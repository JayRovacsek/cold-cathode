//! bf16 primitives and the checkpoint dtype decoders (spec
//! `flash_serve-engine.md` §1, [dtype-table-PTR_DAT_020baa68], with the
//! quantised codecs of spec `flash_serve-dtypes.md` §3–§7 implemented in
//! [`crate::codecs`]).
//!
//! The reference stores norms, vectors and (dequantised) activations as
//! bfloat16 — the binary16 brain format: the top 16 bits of an IEEE-754
//! binary32. [`Bf16`] implements the conversions with round-to-nearest-even
//! (the hardware rounding mode); the port implements them in plain Rust
//! rather than depending on the `half` crate, keeping the Cargo
//! dependency set (and its `cargoHash`) unchanged.
//!
//! The dtype codes the reference table assigns to the resident *float* formats are
//! `0=bf16`, `1=f32`, `2=f16`, `3=i32`, `4=i64`; the quantised codes
//! (`5=q4c`, `7=q8g64`, `16=ht`, `23=i4r`, `24=q6g64`) decode through
//! [`crate::codecs`]. [`decode_tensor`] routes every code: the float
//! formats decode here, the quantised formats delegate to the codecs
//! (dtypes 16/23 additionally need their f16 rotation planes), and the
//! integer codes `3`/`4` (ple metadata, not weights) return `None`.

/// A bf16 value: the high 16 bits of an `f32`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bf16(pub u16);

impl Bf16 {
    /// The bit pattern of `1.0`.
    pub const ONE: Bf16 = Bf16(0x3f80);
    /// The bit pattern of `0.0`.
    pub const ZERO: Bf16 = Bf16(0x0000);

    /// Convert an `f32` to bf16 with round-to-nearest-even: add
    /// `0x7fff + lsb` to the `f32` bits and take the high half — the
    /// standard RNE trick (ties go to the even mantissa, infinities and
    /// normal values round exactly, `f32::MAX` rounds to bf16 infinity).
    pub fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();
        if value.is_nan() {
            // Keep NaN-ness, quiet it (the canonical f16/bf16 conversion).
            return Bf16(((bits >> 16) | 0x0040) as u16);
        }
        let lsb = (bits >> 16) & 1;
        let rounded = bits.wrapping_add(0x7fff + lsb);
        Bf16((rounded >> 16) as u16)
    }

    /// Exact widening back to `f32` (the low 16 bits are zero).
    pub fn to_f32(self) -> f32 {
        f32::from_bits(u32::from(self.0) << 16)
    }

    /// Little-endian bytes (the checkpoint payload order).
    pub fn to_le_bytes(self) -> [u8; 2] {
        self.0.to_le_bytes()
    }

    /// Decode little-endian bytes.
    pub fn from_le_bytes(bytes: [u8; 2]) -> Self {
        Bf16(u16::from_le_bytes(bytes))
    }
}

/// Convert an IEEE-754 binary16 (`f16`) bit pattern to `f32`. Needed for
/// the dtype-2 payloads (`f16`) the reference table names; normals widen
/// exactly, subnormals and NaN payloads are handled explicitly.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let man = (bits & 0x3ff) as u32;
    match exp {
        // Zero and subnormals: value = ±man · 2⁻²⁴ (exact in f32, the
        // mantissa fits in the f32 significand).
        0 => {
            let mag = (man as f32) * f32::from_bits((127 - 24) << 23);
            if sign == 1 {
                -mag
            } else {
                mag
            }
        }
        31 => {
            if man == 0 {
                if sign == 1 {
                    f32::NEG_INFINITY
                } else {
                    f32::INFINITY
                }
            } else {
                // Preserve the NaN payload bit (quiet it).
                f32::from_bits((sign << 31) | 0x7fc0_0000 | (man << 13))
            }
        }
        _ => f32::from_bits((sign << 31) | ((exp + 112) << 23) | (man << 13)),
    }
}

/// Convert an `f32` to an IEEE-754 binary16 (`f16`) bit pattern with
/// round-to-nearest-even (the inverse of [`f16_to_f32`]): normals round
/// their mantissa RNE (a carry promotes the exponent), values below the
/// smallest subnormal round to signed zero, values above the largest
/// finite normal round to infinity.
pub fn f16_from_f32(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    if value.is_nan() {
        return sign | 0x7e00; // quiet the NaN, keep the sign
    }
    if value.is_infinite() {
        return sign | 0x7c00;
    }
    let exp = (bits >> 23) & 0xff;
    let man = bits & 0x007f_ffff;
    if exp == 0 {
        return sign; // f32 subnormals are far below the f16 subnormal range
    }
    let unbiased = exp as i32 - 127;
    if unbiased > 15 {
        return sign | 0x7c00; // overflow → infinity
    }
    if unbiased >= -14 {
        // Normal f16: round the 23-bit mantissa to 10 bits (RNE).
        let rounded = man + 0x0fff + ((man >> 13) & 1);
        let mut mantissa = (rounded >> 13) as u16;
        let mut exponent = (unbiased + 15) as u16;
        if mantissa == 0x400 {
            mantissa = 0;
            exponent += 1;
            if exponent == 0x1f {
                return sign | 0x7c00;
            }
        }
        return sign | (exponent << 10) | mantissa;
    }
    // f16 subnormal: value = M · 2⁻²⁴ with the f32 significand (implicit
    // bit included) shifted right, RNE (M = round(S · 2^(unbiased+1))).
    let shift = (-unbiased - 1) as u32; // 14..=125 for unbiased in −126..−15
    if shift > 24 {
        return sign;
    }
    let significand = man | 0x0080_0000;
    // RNE: half-ulp minus one plus the retained lsb (the same trick as
    // the normal path's `0x0fff`).
    let m = (significand + (1 << (shift - 1)) - 1 + ((significand >> shift) & 1)) >> shift;
    sign | m as u16
}

/// Decode a tensor payload of the resident float dtypes (`0=bf16`,
/// `1=f32`, `2=f16`) into `f32` values; `None` when the payload length
/// does not fit the element size.
pub(crate) fn decode_float_payload(dtype: u32, bytes: &[u8]) -> Option<Vec<f32>> {
    match dtype {
        0 if bytes.len().is_multiple_of(2) => Some(
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|chunk| Bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
                .collect(),
        ),
        1 if bytes.len().is_multiple_of(4) => Some(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_le_bytes(*chunk))
                .collect(),
        ),
        2 if bytes.len().is_multiple_of(2) => Some(
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|chunk| f16_to_f32(u16::from_le_bytes(*chunk)))
                .collect(),
        ),
        _ => None,
    }
}

/// Decode a tensor payload of the given dtype code into `f32` values.
/// The resident float formats (`0=bf16`, `1=f32`, `2=f16`) decode here;
/// the quantised formats (`5=q4c`, `7=q8g64`, `16=ht`, `23=i4r`,
/// `24=q6g64`) delegate to [`crate::codecs::decode`] (dtypes 16/23
/// require their f16 rotation planes in `planes`); the integer codes
/// `3`/`4` have no codec (module docs).
pub fn decode_tensor(
    dtype: u32,
    bytes: &[u8],
    dims: &[u64],
    planes: Option<(&[f32], &[f32])>,
) -> Option<Vec<f32>> {
    match dtype {
        0..=2 => {
            let values = decode_float_payload(dtype, bytes)?;
            let (rows, cols) = crate::codecs::shape(dims)?;
            (rows.checked_mul(cols)? == values.len()).then_some(values)
        }
        _ => crate::codecs::decode(dtype, bytes, dims, planes),
    }
}

/// Encode `f32` values as a little-endian bf16 payload (the fixture
/// builder's writer for dtype-0 tensors).
pub fn encode_bf16(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for value in values {
        out.extend_from_slice(&Bf16::from_f32(*value).to_le_bytes());
    }
    out
}

/// Encode `f32` values as a little-endian f16 payload (the fixture
/// builder's writer for dtype-2 tensors — the quantised codecs' f16
/// rotation planes).
pub fn encode_f16(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for value in values {
        out.extend_from_slice(&f16_from_f32(*value).to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_values_round_trip() {
        for value in [0.0, 1.0, -1.0, 0.5, 2.0, -2.0, 3.140625] {
            let encoded = Bf16::from_f32(value);
            assert_eq!(encoded.to_f32(), value, "bf16 round-trip of {value}");
        }
        assert_eq!(Bf16::ONE.to_f32(), 1.0);
        assert_eq!(Bf16::ONE, Bf16::from_f32(1.0));
    }

    #[test]
    fn rounding_is_nearest_even() {
        // 0x3f808000 (1.00390625) is exactly halfway between the
        // representable 1.0 (0x3f80) and 1.0078125 (0x3f81) — the
        // spacing at 1.0 is 2⁻⁷: RNE picks the even pattern 0x3f80.
        assert_eq!(Bf16::from_f32(f32::from_bits(0x3f80_8000)), Bf16(0x3f80));
        // 0x3f818000 (1.01171875) is exactly halfway between 0x3f81 and
        // 0x3f82: RNE picks the even pattern 0x3f82.
        assert_eq!(Bf16::from_f32(f32::from_bits(0x3f81_8000)), Bf16(0x3f82));
        // Non-ties round to the nearest neighbour.
        assert_eq!(Bf16::from_f32(1.0009), Bf16(0x3f80));
        assert_eq!(Bf16::from_f32(1.005), Bf16(0x3f81));
        // f32::MAX rounds up to bf16 infinity (closer to inf than to the
        // largest finite bf16).
        assert_eq!(Bf16::from_f32(f32::MAX), Bf16(0x7f80));
        assert!(Bf16::from_f32(f32::MAX).to_f32().is_infinite());
        assert!(Bf16::from_f32(f32::NAN).to_f32().is_nan());
    }

    #[test]
    fn f16_conversions() {
        assert_eq!(f16_to_f32(0x3c00), 1.0); // f16 1.0
        assert_eq!(f16_to_f32(0x3800), 0.5); // f16 0.5
        assert_eq!(f16_to_f32(0xbc00), -1.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24)); // smallest subnormal
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn f16_conversions_from_f32() {
        assert_eq!(f16_from_f32(1.0), 0x3c00);
        assert_eq!(f16_from_f32(-1.0), 0xbc00);
        assert_eq!(f16_from_f32(0.5), 0x3800);
        assert_eq!(f16_from_f32(0.0), 0x0000);
        assert_eq!(f16_from_f32(-0.0), 0x8000);
        assert_eq!(f16_from_f32(f32::INFINITY), 0x7c00);
        assert!(f16_from_f32(f32::NAN) & 0x7e00 == 0x7e00);
        // RNE on the discarded mantissa bits: 0x3f811xxx-ish ties.
        // 1.00048828125 is exactly halfway between f16 1.0 and 1.00097…
        // → ties-to-even picks 0x3c00.
        assert_eq!(f16_from_f32(f32::from_bits(0x3f80_1000)), 0x3c00);
        // 1.000732421875 is nearer 1.00097… → 0x3c01.
        assert_eq!(f16_from_f32(f32::from_bits(0x3f80_1800)), 0x3c01);
        // Overflow and the largest finite normal.
        assert_eq!(f32::from_bits(0x4780_0000), 65_536.0);
        assert_eq!(f16_from_f32(65_536.0), 0x7c00); // rounds to infinity
        assert_eq!(f16_from_f32(65_504.0), 0x7bff); // largest finite f16
                                                    // Subnormals: the smallest f16 subnormal and a tie rounding.
        assert_eq!(f16_from_f32(2f32.powi(-24)), 0x0001);
        assert_eq!(f16_from_f32(1.5f32 * 2f32.powi(-24)), 0x0002);
        // Round-trip every exactly-representable sample back through the
        // decoder.
        for bits in [0x0001u16, 0x03ff, 0x0400, 0x3c00, 0x3800, 0xbc00, 0x7bff] {
            assert_eq!(
                f16_from_f32(f16_to_f32(bits)),
                bits,
                "round-trip of {bits:#06x}"
            );
        }
    }

    /// Reference f32→f16 for finite normal inputs (the sweep's domain):
    /// pick the nearest f16 by exact integer comparison, ties to the
    /// even pattern — an independent formulation of the `f16_from_f32`
    /// rounding trick.
    fn ref_f16(bits: u32) -> u16 {
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = (bits >> 23) & 0xff;
        let man = bits & 0x007f_ffff;
        assert!(exp != 0, "sweep inputs are normal f32s");
        let unbiased = exp as i32 - 127;
        assert!(unbiased <= 15, "sweep inputs do not overflow f16");
        if unbiased >= -14 {
            // Normal f16: round the 23-bit mantissa to 10 bits.
            let q = man >> 13;
            let rem = man & 0x1fff;
            let (mut mantissa, mut exponent) = (q, (unbiased + 15) as u32);
            if rem > 0x1000 || (rem == 0x1000 && q & 1 == 1) {
                mantissa = q + 1;
                if mantissa == 0x400 {
                    mantissa = 0;
                    exponent += 1;
                }
            }
            if exponent == 0x1f {
                sign | 0x7c00
            } else {
                sign | ((exponent as u16) << 10) | mantissa as u16
            }
        } else {
            // Subnormal f16: M = round_half_even(significand / 2^shift).
            let shift = (-unbiased - 1) as u32;
            let significand = man | 0x0080_0000;
            let m = if shift > 24 {
                0
            } else {
                let q = significand >> shift;
                let rem = significand & ((1u32 << shift) - 1);
                let half = 1u32 << (shift - 1);
                if rem > half || (rem == half && q & 1 == 1) {
                    q + 1
                } else {
                    q
                }
            };
            // m == 0x400 promotes to the smallest normal via the layout.
            sign | m as u16
        }
    }

    #[test]
    fn f16_subnormal_ties_round_to_even() {
        // 1.25·2⁻²³ = 2.5 ulp of 2⁻²⁴: tie between 0x0002 and 0x0003 →
        // RNE picks the even pattern 0x0002.
        assert_eq!(f16_from_f32(1.25f32 * 2f32.powi(-23)), 0x0002);
        // 2⁻²⁵ = 0.5 ulp: tie between 0x0000 and 0x0001 → 0x0000.
        assert_eq!(f16_from_f32(2f32.powi(-25)), 0x0000);
        // Sweep the f32 exponents 100..=113 (f16 subnormals up to the
        // smallest normal f16) across a few mantissa patterns, asserting
        // against the inline reference.
        for exp in 100u32..=113 {
            for man in [0x000000u32, 0x000001, 0x3f_ffff, 0x40_0000, 0x7f_ffff] {
                let bits = (exp << 23) | man;
                assert_eq!(
                    f16_from_f32(f32::from_bits(bits)),
                    ref_f16(bits),
                    "bits {bits:#08x}"
                );
            }
        }
    }

    #[test]
    fn encode_f16_matches_the_decoder() {
        let values = vec![0.0, 1.0, -2.5, 0.75];
        let bytes = encode_f16(&values);
        assert_eq!(bytes.len(), 8);
        assert_eq!(decode_tensor(2, &bytes, &[1, 4], None).unwrap(), values);
    }

    #[test]
    fn decode_tensor_accepts_the_resident_float_dtypes() {
        // dtype 0: bf16 little-endian.
        let bytes = vec![0x80, 0x3f, 0x00, 0x40]; // 1.0, 2.0
        assert_eq!(
            decode_tensor(0, &bytes, &[2], None).unwrap(),
            vec![1.0, 2.0]
        );
        // dtype 1: f32 little-endian.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1.5f32.to_le_bytes());
        bytes.extend_from_slice(&(-0.25f32).to_le_bytes());
        assert_eq!(
            decode_tensor(1, &bytes, &[1, 2], None).unwrap(),
            vec![1.5, -0.25]
        );
        // dtype 2: f16 little-endian.
        let bytes = 0x3c00u16.to_le_bytes().to_vec();
        assert_eq!(decode_tensor(2, &bytes, &[1, 1], None).unwrap(), vec![1.0]);
        // A payload whose length contradicts dims is rejected.
        assert_eq!(decode_tensor(0, &bytes, &[2, 1], None), None);
        // Integer dtypes have no codec (module docs).
        assert_eq!(decode_tensor(3, &[0, 0, 0, 0], &[1], None), None);
        // A short payload cannot be decoded.
        assert_eq!(decode_tensor(0, &[0], &[1, 1], None), None);
        // Quantised dtypes route to the codecs (tested in crate::codecs);
        // payloads failing the codec layout still yield None here.
        assert_eq!(decode_tensor(5, &[0, 0], &[1, 2], None), None);
        assert_eq!(decode_tensor(16, &[0, 0], &[1, 2], None), None);
        assert_eq!(decode_tensor(23, &[0, 0], &[1, 2], None), None);
        assert_eq!(decode_tensor(24, &[0, 0], &[1, 2], None), None);
    }

    #[test]
    fn encode_bf16_matches_the_decoder() {
        let values = vec![0.0, 1.0, -2.5, 0.75];
        let bytes = encode_bf16(&values);
        assert_eq!(bytes.len(), 8);
        assert_eq!(decode_tensor(0, &bytes, &[1, 4], None).unwrap(), values);
    }
}
