//! bf16 primitives and the checkpoint dtype decoders (spec
//! `flash_serve-engine.md` §1, [dtype-table-PTR_DAT_020baa68]).
//!
//! The reference stores norms, vectors and (dequantised) activations as
//! bfloat16 — the binary16 brain format: the top 16 bits of an IEEE-754
//! binary32. [`Bf16`] implements the conversions with round-to-nearest-even
//! (the hardware rounding mode); the port implements them in plain Rust
//! rather than depending on the `half` crate, keeping the Cargo
//! dependency set (and its `cargoHash`) unchanged.
//!
//! The dtype codes the reference table assigns to the resident formats are
//! `0=bf16`, `1=f32`, `2=f16`, `3=i32`, `4=i64`; the quantised codes
//! (`5=q4c`, `16=ht`, `23=i4r`, `24=q6g64`, …) have no dequant codec
//! specified yet (spec §1 resolved table, §3 `[D]` entries), so
//! [`decode_tensor`] accepts only the resident float formats and returns
//! `None` for everything else — the caller (`model::Model::new`) treats
//! that as "this checkpoint cannot be run for real yet".

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

/// Decode a tensor payload of the given dtype code into `f32` values.
/// Supports the resident float formats (`0=bf16`, `1=f32`, `2=f16`);
/// quantised and integer dtypes return `None` (module docs).
pub fn decode_tensor(dtype: u32, bytes: &[u8]) -> Option<Vec<f32>> {
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

/// Encode `f32` values as a little-endian bf16 payload (the fixture
/// builder's writer for dtype-0 tensors).
pub fn encode_bf16(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for value in values {
        out.extend_from_slice(&Bf16::from_f32(*value).to_le_bytes());
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
    fn decode_tensor_accepts_the_resident_float_dtypes() {
        // dtype 0: bf16 little-endian.
        let bytes = vec![0x80, 0x3f, 0x00, 0x40]; // 1.0, 2.0
        assert_eq!(decode_tensor(0, &bytes).unwrap(), vec![1.0, 2.0]);
        // dtype 1: f32 little-endian.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1.5f32.to_le_bytes());
        bytes.extend_from_slice(&(-0.25f32).to_le_bytes());
        assert_eq!(decode_tensor(1, &bytes).unwrap(), vec![1.5, -0.25]);
        // dtype 2: f16 little-endian.
        let bytes = 0x3c00u16.to_le_bytes().to_vec();
        assert_eq!(decode_tensor(2, &bytes).unwrap(), vec![1.0]);
        // Quantised and integer dtypes have no codec (module docs).
        assert_eq!(decode_tensor(5, &[0, 0]), None);
        assert_eq!(decode_tensor(16, &[0, 0]), None);
        assert_eq!(decode_tensor(23, &[0, 0]), None);
        assert_eq!(decode_tensor(24, &[0, 0]), None);
        assert_eq!(decode_tensor(3, &[0, 0, 0, 0]), None);
        // A short payload cannot be decoded.
        assert_eq!(decode_tensor(0, &[0]), None);
    }

    #[test]
    fn encode_bf16_matches_the_decoder() {
        let values = vec![0.0, 1.0, -2.5, 0.75];
        let bytes = encode_bf16(&values);
        assert_eq!(bytes.len(), 8);
        assert_eq!(decode_tensor(0, &bytes).unwrap(), values);
    }
}
