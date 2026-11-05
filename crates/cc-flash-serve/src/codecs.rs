//! Dequantisation codecs for the quantised checkpoint dtypes (spec
//! `flash_serve-dtypes.md` §3–§7, formulas marked `[D-inferred]` there —
//! the x86 reference only launches the HIP kernels whose value-decoding
//! math lives in the AMDGPU code objects).
//!
//! [`Weights`] is the byte-preserving weight storage the port keeps per
//! tensor: dense tensors hold their decoded `f32` values, quantised
//! tensors keep their raw payload bytes and decode one row at a time
//! ([`Weights::row`]), so a gemv over a quantised matrix costs O(row)
//! rather than materialising the whole tensor per call. The
//! whole-tensor facade is [`decode`] (reached through
//! [`crate::bf16::decode_tensor`]).
//!
//! Per-codec layout decisions (each one the spec's documented reading of
//! an ambiguous on-disk format):
//!
//! * **q4c (5)** — Q4_0 block geometry (per 32 elements: 2-byte f16
//!   scale + 16 packed nibbles, low nibble first, rows padded to 16 B
//!   behind a 64-byte tensor header). The header's source field is a
//!   documented Gap, so the Q4_0 path is the default; the IQ4_NL variant
//!   decodes nibbles through the shipped `kIq4nlValues` codebook
//!   ([`K_IQ4NL_VALUES`], fetched from llama.cpp `kvalues_iq4nl`, which
//!   the reference's `k_gather_iq4nl` kernel confirms).
//! * **q8g64 (7)** — per 64-element block: 4-byte f32 scale + 64 u8
//!   values, `w = scale × i8` (the 4-byte scale encoding is itself
//!   inferred f32, spec §10).
//! * **q6g64 (24)** — per 64-element block: 4-byte f32 scale + 48 bytes
//!   of contiguous little-endian 6-bit packing, `w = scale × (q6 − 32)`;
//!   rows padded to 16 B (`pad16(13K/16)` per row).
//! * **ht (16)** — pure 4-bit packing (low nibble first), no in-payload
//!   scale: `w[r][k] = suh[k] · svh[r] · (nibble − 8)` with the f16
//!   rotation planes (the nibble→value offset and plane application
//!   order are `[D-inferred]`, spec §6).
//! * **i4r (23)** — per 64-element block: 32 bytes of nibbles (low
//!   first) + 1-byte scale, decoded as fp8 e4m3 ([`f8e4m3_to_f32`]; the
//!   fp8-vs-u8 choice is a documented Gap, fp8 chosen for realistic
//!   magnitudes and the engine's fp8 vocabulary): `w[r][k] =
//!   f8e4m3(scale) × (nibble − 8) × suh[k] × svh[r]` (spec §7).
//!
//! Integer dtypes 3/4 (i32/i64) are not weights (spec §2) and have no
//! codec here: [`Weights::from_dtype`] rejects them, matching the
//! previous `decode_tensor` behaviour.

use crate::bf16;

/// The IQ4_NL 16-entry codebook (llama.cpp `kvalues_iq4nl`; the
/// reference ships the same table as the `kIq4nlValues` symbol).
pub const K_IQ4NL_VALUES: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Round up to a 16-byte boundary (the checkpoint's row padding).
pub fn pad16(n: usize) -> usize {
    (n + 15) & !15
}

/// The tensor shape a payload decodes to: 1-D `[n]` → `(1, n)`, 2-D
/// `[rows, cols]` as-is, 3-D `[a, b, c]` (stacked experts, conv1d)
/// flattened to `[a·b, c]` — the same reading `model::Tensor::matrix`
/// uses.
pub fn shape(dims: &[u64]) -> Option<(usize, usize)> {
    match dims {
        [n] => Some((1, *n as usize)),
        [rows, cols] => Some((*rows as usize, *cols as usize)),
        [a, b, c] => Some((*a as usize * *b as usize, *c as usize)),
        _ => None,
    }
}

/// Decode an fp8 e4m3 byte (sign 1 / exponent 4, bias 7 / mantissa 3)
/// to `f32` exactly. `0x7f` is the only NaN pattern (e4m3 has no
/// infinity); the largest finite magnitude is 448 (`0x7e`).
pub fn f8e4m3_to_f32(byte: u8) -> f32 {
    let sign = (byte >> 7) as u32;
    let exp = (byte >> 3) & 0xf;
    let man = byte & 0x7;
    let magnitude = if exp == 0 {
        // Subnormal: man · 2⁻⁶⁻³, exact in f32.
        f32::from(man) * f32::from_bits((127 - 9) << 23)
    } else if exp == 15 && man == 7 {
        return f32::NAN;
    } else {
        f32::from(8 + man) * f32::from_bits(((i32::from(exp) + 127 - 10) as u32) << 23)
    };
    if sign == 1 {
        -magnitude
    } else {
        magnitude
    }
}

/// Encode an `f32` as the nearest fp8 e4m3 byte (exhaustive nearest
/// search over the 254 finite patterns, ties to the smaller pattern;
/// NaN maps to `0x7f`). Used by the fixture builder's i4r encoder.
pub fn f8e4m3_from_f32(value: f32) -> u8 {
    if value.is_nan() {
        return 0x7f;
    }
    if value == 0.0 {
        // Preserve the sign of zero (0x00 / 0x80).
        return if value.is_sign_negative() { 0x80 } else { 0x00 };
    }
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for byte in 0u16..=255 {
        let candidate = f8e4m3_to_f32(byte as u8);
        if candidate.is_nan() {
            continue;
        }
        let err = (candidate - value).abs();
        if err < best_err {
            best_err = err;
            best = byte as u8;
        }
    }
    best
}

/// The q4c source format named by the engine string `Q4C-P source
/// (IQ4_NL, IQ4_XS, IQ3_S, Q4_0)` (spec §3). The 64-byte tensor header's
/// source field is unpinned (spec §10 Gap), so [`Q4cSource::Q40`] is the
/// default and the shipped-codebook variant [`Q4cSource::Iq4Nl`] is
/// selectable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Q4cSource {
    /// Plain 32-element blocks: `w = f16_scale × (nibble − 8)`.
    #[default]
    Q40,
    /// Codebook variant: `w = f16_scale × kIq4nlValues[nibble]`.
    Iq4Nl,
}

/// The byte-preserving weight storage of [`crate::ops::Matrix`]: dense
/// tensors hold decoded values, quantised tensors hold the raw payload
/// (plus the f16 rotation planes for `ht`/`i4r`) and decode per row.
#[derive(Debug, Clone)]
pub enum Weights {
    /// Decoded `f32` values (dtypes 0/1/2), row-major `rows × cols`.
    Dense(Vec<f32>),
    /// dtype 5: 64-byte header + per-row `pad16(9k/16)` Q4_0 blocks.
    Q4c { bytes: Vec<u8>, source: Q4cSource },
    /// dtype 7: per-row `(k/64) × 68` blocks (f32 scale + 64 i8).
    Q8g64 { bytes: Vec<u8> },
    /// dtype 24: per-row `pad16(13k/16)` blocks (f32 scale + 48 packed).
    Q6g64 { bytes: Vec<u8> },
    /// dtype 16: nibble payload + the f16 rotation planes.
    Ht {
        bytes: Vec<u8>,
        suh: Vec<f32>,
        svh: Vec<f32>,
        /// Rows the `svh` plane covers (dims[1] of a 3-D tensor, else
        /// the row count); row `r` reads `svh[r % svh_rows]`.
        svh_rows: usize,
    },
    /// dtype 23: per-row `(k/64) × 33` blocks (32 nibble bytes + fp8
    /// scale) + the f16 rotation planes.
    I4r {
        bytes: Vec<u8>,
        suh: Vec<f32>,
        svh: Vec<f32>,
        /// Rows the `svh` plane covers (as in [`Weights::Ht`]).
        svh_rows: usize,
    },
}

impl Weights {
    /// Build the weights of a tensor payload, validating the payload
    /// against the dtype's byte formula. `planes` carries the decoded
    /// `(suh, svh)` f16 plane values, required by dtypes 16/23.
    /// `None` for dtypes without a codec (3/4) or a payload that fails
    /// its layout check.
    pub fn from_dtype(
        dtype: u32,
        bytes: &[u8],
        dims: &[u64],
        planes: Option<(&[f32], &[f32])>,
    ) -> Option<Weights> {
        let (rows, cols) = shape(dims)?;
        match dtype {
            0..=2 => {
                let data = bf16::decode_float_payload(dtype, bytes)?;
                (data.len() == rows.checked_mul(cols)?).then_some(Weights::Dense(data))
            }
            5 => {
                if !cols.is_multiple_of(32)
                    || bytes.len() != 64 + rows.checked_mul(pad16(9 * cols / 16))?
                {
                    return None;
                }
                Some(Weights::Q4c {
                    bytes: bytes.to_vec(),
                    source: Q4cSource::default(),
                })
            }
            7 => {
                if !cols.is_multiple_of(64)
                    || bytes.len() != rows.checked_mul(cols / 64)?.checked_mul(68)?
                {
                    return None;
                }
                Some(Weights::Q8g64 {
                    bytes: bytes.to_vec(),
                })
            }
            24 => {
                if !cols.is_multiple_of(64)
                    || bytes.len() != rows.checked_mul(pad16(13 * cols / 16))?
                {
                    return None;
                }
                Some(Weights::Q6g64 {
                    bytes: bytes.to_vec(),
                })
            }
            16 => {
                let (suh, svh) = planes?;
                let svh_rows = if dims.len() == 3 {
                    dims[1] as usize
                } else {
                    rows
                };
                if !cols.is_multiple_of(2)
                    || bytes.len() != rows.checked_mul(cols / 2)?
                    || suh.len() != cols
                    || svh.len() != svh_rows
                {
                    return None;
                }
                Some(Weights::Ht {
                    bytes: bytes.to_vec(),
                    suh: suh.to_vec(),
                    svh: svh.to_vec(),
                    svh_rows,
                })
            }
            23 => {
                let (suh, svh) = planes?;
                let svh_rows = if dims.len() == 3 {
                    dims[1] as usize
                } else {
                    rows
                };
                if !cols.is_multiple_of(64)
                    || bytes.len() != rows.checked_mul(cols / 64)?.checked_mul(33)?
                    || suh.len() != cols
                    || svh.len() != svh_rows
                {
                    return None;
                }
                Some(Weights::I4r {
                    bytes: bytes.to_vec(),
                    suh: suh.to_vec(),
                    svh: svh.to_vec(),
                    svh_rows,
                })
            }
            // dtype 3/4 (i32/i64) are ple metadata, not weights (spec §2).
            _ => None,
        }
    }

    /// Dequantise row `row` (the fused row-slice decode the gemv uses).
    pub fn row(&self, cols: usize, row: usize) -> Vec<f32> {
        match self {
            Weights::Dense(data) => data[row * cols..(row + 1) * cols].to_vec(),
            Weights::Q4c { bytes, source } => q4c_row(bytes, cols, *source, row),
            Weights::Q8g64 { bytes } => q8g64_row(bytes, cols, row),
            Weights::Q6g64 { bytes } => q6g64_row(bytes, cols, row),
            Weights::Ht {
                bytes,
                suh,
                svh,
                svh_rows,
            } => ht_row(bytes, cols, row, suh, svh, *svh_rows),
            Weights::I4r {
                bytes,
                suh,
                svh,
                svh_rows,
            } => i4r_row(bytes, cols, row, suh, svh, *svh_rows),
        }
    }

    /// Materialise every value (the whole-tensor facade; the gemv path
    /// uses [`Weights::row`] instead).
    pub fn dequant_all(&self, rows: usize, cols: usize) -> Vec<f32> {
        match self {
            Weights::Dense(data) => data.clone(),
            weights => (0..rows).flat_map(|r| weights.row(cols, r)).collect(),
        }
    }
}

/// Decode a quantised payload into every value (dtypes 5/7/16/23/24;
/// `None` for dtypes without a codec).
pub fn decode(
    dtype: u32,
    bytes: &[u8],
    dims: &[u64],
    planes: Option<(&[f32], &[f32])>,
) -> Option<Vec<f32>> {
    let (rows, cols) = shape(dims)?;
    let weights = Weights::from_dtype(dtype, bytes, dims, planes)?;
    Some(weights.dequant_all(rows, cols))
}

// ---------------------------------------------------------------------------
// q4c (dtype 5)
// ---------------------------------------------------------------------------

/// Dequantise one q4c row: `pad16(9k/16)` bytes at `64 + row × stride`,
/// `k/32` blocks of 18 bytes (2-byte f16 scale + 16 nibble bytes, low
/// nibble = element `2i`).
fn q4c_row(bytes: &[u8], cols: usize, source: Q4cSource, row: usize) -> Vec<f32> {
    let stride = pad16(9 * cols / 16);
    let base = 64 + row * stride;
    let mut out = Vec::with_capacity(cols);
    for block in 0..cols / 32 {
        let off = base + block * 18;
        let scale = bf16::f16_to_f32(u16::from_le_bytes([bytes[off], bytes[off + 1]]));
        for i in 0..16 {
            let byte = bytes[off + 2 + i];
            for nibble in [byte & 0xf, byte >> 4] {
                let value = match source {
                    Q4cSource::Q40 => f32::from(nibble) - 8.0,
                    Q4cSource::Iq4Nl => f32::from(K_IQ4NL_VALUES[usize::from(nibble)]),
                };
                out.push(scale * value);
            }
        }
    }
    out
}

/// Encode `rows × cols` values as a Q4_0 q4c payload (64-byte zero
/// header; per block the f16 scale is `max|w| / 8`, nibbles are
/// `round(w / scale) + 8` clamped to `0..=15`).
pub fn q4c_encode(values: &[f32], rows: usize, cols: usize) -> Option<Vec<u8>> {
    (cols.is_multiple_of(32) && values.len() == rows.checked_mul(cols)?).then(|| {
        let stride = pad16(9 * cols / 16);
        let mut out = vec![0u8; 64 + rows * stride];
        for r in 0..rows {
            for b in 0..cols / 32 {
                let block = &values[r * cols + b * 32..r * cols + (b + 1) * 32];
                let max = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let scale_bits = bf16::f16_from_f32(if max == 0.0 { 0.0 } else { max / 8.0 });
                let scale = bf16::f16_to_f32(scale_bits);
                let off = 64 + r * stride + b * 18;
                out[off..off + 2].copy_from_slice(&scale_bits.to_le_bytes());
                for i in 0..16 {
                    let lo = nibble_quantise(block[2 * i], scale);
                    let hi = nibble_quantise(block[2 * i + 1], scale);
                    out[off + 2 + i] = lo | (hi << 4);
                }
            }
        }
        out
    })
}

/// Quantise one value against a block scale into a 4-bit nibble
/// (`round(w / scale) + 8`, clamped; a zero scale encodes as 8).
fn nibble_quantise(value: f32, scale: f32) -> u8 {
    if scale == 0.0 {
        return 8;
    }
    (f32::round(value / scale) + 8.0).clamp(0.0, 15.0) as u8
}

// ---------------------------------------------------------------------------
// q8g64 (dtype 7)
// ---------------------------------------------------------------------------

/// Dequantise one q8g64 row: `(k/64) × 68` bytes, each block a 4-byte
/// f32 scale followed by 64 i8 values (`w = scale × i8`).
fn q8g64_row(bytes: &[u8], cols: usize, row: usize) -> Vec<f32> {
    let stride = (cols / 64) * 68;
    let base = row * stride;
    let mut out = Vec::with_capacity(cols);
    for block in 0..cols / 64 {
        let off = base + block * 68;
        let scale = f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        for i in 0..64 {
            out.push(scale * f32::from(bytes[off + 4 + i] as i8));
        }
    }
    out
}

/// Encode `rows × cols` values as a q8g64 payload (per block the f32
/// scale is `max|w| / 127`, values are `round(w / scale)` clamped to
/// `i8`).
pub fn q8g64_encode(values: &[f32], rows: usize, cols: usize) -> Option<Vec<u8>> {
    (cols.is_multiple_of(64) && values.len() == rows.checked_mul(cols)?).then(|| {
        let stride = (cols / 64) * 68;
        let mut out = vec![0u8; rows * stride];
        for r in 0..rows {
            for b in 0..cols / 64 {
                let block = &values[r * cols + b * 64..r * cols + (b + 1) * 64];
                let max = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let scale = if max == 0.0 { 0.0 } else { max / 127.0 };
                let off = r * stride + b * 68;
                out[off..off + 4].copy_from_slice(&scale.to_le_bytes());
                for i in 0..64 {
                    let q = if scale == 0.0 {
                        0
                    } else {
                        f32::round(block[i] / scale).clamp(-128.0, 127.0) as i8
                    };
                    out[off + 4 + i] = q as u8;
                }
            }
        }
        out
    })
}

// ---------------------------------------------------------------------------
// q6g64 (dtype 24)
// ---------------------------------------------------------------------------

/// Pack 64 six-bit values into 48 bytes (contiguous little-endian bit
/// order: value `i` occupies bits `6i..6i+5`).
pub fn q6g64_pack_row(values: [u8; 64]) -> [u8; 48] {
    let mut out = [0u8; 48];
    for (i, &value) in values.iter().enumerate() {
        let bit = i * 6;
        let byte = bit / 8;
        let shift = bit % 8;
        let v = value & 0x3f;
        out[byte] |= v << shift;
        if shift > 2 {
            out[byte + 1] |= v >> (8 - shift);
        }
    }
    out
}

/// Inverse of [`q6g64_pack_row`].
pub fn q6g64_unpack_row(bytes: [u8; 48]) -> [u8; 64] {
    let mut out = [0u8; 64];
    for (i, slot) in out.iter_mut().enumerate() {
        let bit = i * 6;
        let byte = bit / 8;
        let shift = bit % 8;
        let mut v = bytes[byte] >> shift;
        if shift > 2 {
            v |= bytes[byte + 1] << (8 - shift);
        }
        *slot = v & 0x3f;
    }
    out
}

/// Dequantise one q6g64 row: `pad16(13k/16)` bytes per row, each 52-byte
/// block a 4-byte f32 scale + 48 packed 6-bit values
/// (`w = scale × (q6 − 32)`).
fn q6g64_row(bytes: &[u8], cols: usize, row: usize) -> Vec<f32> {
    let stride = pad16(13 * cols / 16);
    let base = row * stride;
    let mut out = Vec::with_capacity(cols);
    for block in 0..cols / 64 {
        let off = base + block * 52;
        let scale = f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        let packed: [u8; 48] = bytes[off + 4..off + 52].try_into().unwrap();
        for q in q6g64_unpack_row(packed) {
            out.push(scale * (f32::from(q) - 32.0));
        }
    }
    out
}

/// Encode `rows × cols` values as a q6g64 payload (per block the f32
/// scale is `max|w| / 32`, values are `round(w / scale) + 32` clamped
/// to `0..=63`).
pub fn q6g64_encode(values: &[f32], rows: usize, cols: usize) -> Option<Vec<u8>> {
    (cols.is_multiple_of(64) && values.len() == rows.checked_mul(cols)?).then(|| {
        let stride = pad16(13 * cols / 16);
        let mut out = vec![0u8; rows * stride];
        for r in 0..rows {
            for b in 0..cols / 64 {
                let block = &values[r * cols + b * 64..r * cols + (b + 1) * 64];
                let max = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let scale = if max == 0.0 { 0.0 } else { max / 32.0 };
                let off = r * stride + b * 52;
                out[off..off + 4].copy_from_slice(&scale.to_le_bytes());
                let mut packed = [0u8; 64];
                for i in 0..64 {
                    packed[i] = if scale == 0.0 {
                        32
                    } else {
                        (f32::round(block[i] / scale) + 32.0).clamp(0.0, 63.0) as u8
                    };
                }
                out[off + 4..off + 52].copy_from_slice(&q6g64_pack_row(packed));
            }
        }
        out
    })
}

// ---------------------------------------------------------------------------
// ht (dtype 16)
// ---------------------------------------------------------------------------

/// Dequantise one ht row: `k/2` nibble bytes (low nibble = element
/// `2k`), each value `suh[k] · svh[r % svh_rows] · (nibble − 8)`.
fn ht_row(
    bytes: &[u8],
    cols: usize,
    row: usize,
    suh: &[f32],
    svh: &[f32],
    svh_rows: usize,
) -> Vec<f32> {
    let base = row * (cols / 2);
    let sv = svh[row % svh_rows];
    let mut out = Vec::with_capacity(cols);
    for k in 0..cols {
        let byte = bytes[base + k / 2];
        let nibble = if k % 2 == 0 { byte & 0xf } else { byte >> 4 };
        out.push(suh[k] * sv * (f32::from(nibble) - 8.0));
    }
    out
}

/// Encode `rows × cols` values as an ht nibble payload (per row the
/// nibble scale maps the row's magnitude to `±8`: `nibble =
/// clamp(round(8 + 8·v / max|row|), 0, 15)`; a zero row encodes as 8).
/// The dequantised result re-applies the tensor's planes, so it is not a
/// value-exact round-trip of `values` — the exact-value tests build
/// bytes directly.
pub fn ht_encode(values: &[f32], rows: usize, cols: usize) -> Option<Vec<u8>> {
    (cols.is_multiple_of(2) && values.len() == rows.checked_mul(cols)?).then(|| {
        let mut out = vec![0u8; rows * (cols / 2)];
        for r in 0..rows {
            let row = &values[r * cols..(r + 1) * cols];
            let max = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            for k in 0..cols / 2 {
                let mut byte = 0u8;
                for i in 0..2 {
                    let v = row[2 * k + i];
                    let nibble = if max == 0.0 {
                        8
                    } else {
                        f32::round(8.0 + 8.0 * v / max).clamp(0.0, 15.0) as u8
                    };
                    byte |= nibble << (4 * i);
                }
                out[r * (cols / 2) + k] = byte;
            }
        }
        out
    })
}

// ---------------------------------------------------------------------------
// i4r (dtype 23)
// ---------------------------------------------------------------------------

/// Dequantise one i4r row: `(k/64) × 33` bytes, each block 32 nibble
/// bytes (low nibble = element `2i`) plus a 1-byte fp8 e4m3 scale at
/// byte 32; each value `scale × (nibble − 8) × suh[k] × svh[r %
/// svh_rows]`.
fn i4r_row(
    bytes: &[u8],
    cols: usize,
    row: usize,
    suh: &[f32],
    svh: &[f32],
    svh_rows: usize,
) -> Vec<f32> {
    let stride = (cols / 64) * 33;
    let base = row * stride;
    let sv = svh[row % svh_rows];
    let mut out = Vec::with_capacity(cols);
    for block in 0..cols / 64 {
        let off = base + block * 33;
        let scale = f8e4m3_to_f32(bytes[off + 32]);
        for i in 0..64 {
            let byte = bytes[off + i / 2];
            let nibble = if i % 2 == 0 { byte & 0xf } else { byte >> 4 };
            out.push(scale * (f32::from(nibble) - 8.0) * suh[block * 64 + i] * sv);
        }
    }
    out
}

/// Encode `rows × cols` values as an i4r payload (per block the scale
/// byte is `f8e4m3_from_f32(max|w| / 8)` and the nibbles are
/// `round(w / decoded_scale) + 8` clamped to `0..=15`).
pub fn i4r_encode(values: &[f32], rows: usize, cols: usize) -> Option<Vec<u8>> {
    (cols.is_multiple_of(64) && values.len() == rows.checked_mul(cols)?).then(|| {
        let stride = (cols / 64) * 33;
        let mut out = vec![0u8; rows * stride];
        for r in 0..rows {
            for b in 0..cols / 64 {
                let block = &values[r * cols + b * 64..r * cols + (b + 1) * 64];
                let max = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let scale_byte = f8e4m3_from_f32(if max == 0.0 { 0.0 } else { max / 8.0 });
                let scale = f8e4m3_to_f32(scale_byte);
                let off = r * stride + b * 33;
                for i in 0..64 {
                    let nibble = nibble_quantise(block[i], scale);
                    if i % 2 == 0 {
                        out[off + i / 2] = nibble;
                    } else {
                        out[off + i / 2] |= nibble << 4;
                    }
                }
                out[off + 32] = scale_byte;
            }
        }
        out
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad16_rounds_to_16_byte_boundaries() {
        assert_eq!(pad16(0), 0);
        assert_eq!(pad16(1), 16);
        assert_eq!(pad16(16), 16);
        assert_eq!(pad16(17), 32);
        assert_eq!(pad16(36), 48);
        assert_eq!(pad16(52), 64);
        assert_eq!(pad16(208), 208);
    }

    #[test]
    fn shape_flattens_the_documented_ranks() {
        assert_eq!(shape(&[6]), Some((1, 6)));
        assert_eq!(shape(&[6, 4]), Some((6, 4)));
        assert_eq!(shape(&[4, 16, 64]), Some((64, 64)));
        assert_eq!(shape(&[]), None);
        assert_eq!(shape(&[1, 2, 3, 4]), None);
    }

    #[test]
    fn f8e4m3_matches_the_encoding() {
        assert_eq!(f8e4m3_to_f32(0x00), 0.0);
        assert_eq!(f8e4m3_to_f32(0x80), -0.0);
        assert_eq!(f8e4m3_to_f32(0x38), 1.0); // e=7 (bias 7), m=0
        assert_eq!(f8e4m3_to_f32(0xb8), -1.0);
        assert_eq!(f8e4m3_to_f32(0x39), 1.125); // m=1 → 1 + 1/8
        assert_eq!(f8e4m3_to_f32(0x7e), 448.0); // largest finite
        assert_eq!(f8e4m3_to_f32(0x01), 2f32.powi(-9)); // smallest subnormal
        assert!(f8e4m3_to_f32(0x7f).is_nan());
        // The encoder inverts the decoder exactly on representable values.
        for byte in 0u16..=255 {
            let byte = byte as u8;
            let value = f8e4m3_to_f32(byte);
            if value.is_nan() {
                assert_eq!(f8e4m3_from_f32(value), 0x7f);
            } else {
                assert_eq!(f8e4m3_from_f32(value), byte, "re-encode of {value}");
            }
        }
    }

    #[test]
    fn q4c_row_decodes_a_hand_built_block() {
        // One row of 32 elements: 64-byte header + one row of
        // pad16(9·32/16) = 32 bytes (18-byte block + 14 bytes padding).
        // f16 scale 2.0 (0x4000), nibbles 8,8,7,9,0,15,...
        let bytes = q4c_fixture_bytes();
        let weights = Weights::Q4c {
            bytes: bytes.clone(),
            source: Q4cSource::Q40,
        };
        let row = weights.row(32, 0);
        assert_eq!(row.len(), 32);
        assert_eq!(&row[..6], &[0.0, 0.0, -2.0, 2.0, -16.0, 14.0]);
        assert!(
            row[6..].iter().all(|v| *v == -16.0),
            "zero nibbles → 2.0 · (0 − 8)"
        );
        // The IQ4_NL variant maps the same nibbles through the codebook
        // (k[8] = 1, k[7] = −10, k[9] = 13, k[0] = −127, k[15] = 113).
        let weights = Weights::Q4c {
            bytes,
            source: Q4cSource::Iq4Nl,
        };
        let row = weights.row(32, 0);
        assert_eq!(
            &row[..6],
            &[2.0, 2.0, -20.0, 26.0, -254.0, 226.0],
            "kIq4nlValues × f16 scale 2.0"
        );
        assert!(
            row[6..].iter().all(|v| *v == 2.0 * -127.0),
            "zero nibbles → k[0]"
        );
    }

    fn q4c_fixture_bytes() -> Vec<u8> {
        let mut bytes = vec![0u8; 64 + 32]; // header + padded row
        bytes[64..66].copy_from_slice(&0x4000u16.to_le_bytes()); // scale 2.0
        bytes[66] = 0x88; // elements 0,1 → nibbles 8,8
        bytes[67] = 0x97; // elements 2,3 → nibbles 7,9
        bytes[68] = 0xf0; // elements 4,5 → nibbles 0,15
        bytes
    }

    #[test]
    fn q4c_encode_decode_round_trip_is_within_the_block_error() {
        // Q4_0 encoding: dequantised values match the originals within
        // one nibble step (scale = max|w|/8 → step = scale).
        let values: Vec<f32> = (0..32).map(|i| (i as f32) * 0.25 - 4.0).collect();
        let bytes = q4c_encode(&values, 1, 32).unwrap();
        assert_eq!(bytes.len(), 64 + 32);
        let decoded = q4c_row(&bytes, 32, Q4cSource::Q40, 0);
        let scale = 4.0f32 / 8.0; // max|w| = 4.0 (f16-exact)
        for (got, want) in decoded.iter().zip(&values) {
            assert!((got - want).abs() <= scale, "{got} vs {want}");
        }
    }

    #[test]
    fn q8g64_row_is_exact() {
        // One row of 64 elements: f32 scale 0.5 + i8 values 127, -128,
        // 1, -1, 0, ...
        let mut bytes = vec![0u8; 68];
        bytes[0..4].copy_from_slice(&0.5f32.to_le_bytes());
        bytes[4] = 127;
        bytes[5] = (-128i8) as u8;
        bytes[6] = 1;
        bytes[7] = (-1i8) as u8;
        let weights = Weights::Q8g64 { bytes };
        let row = weights.row(64, 0);
        assert_eq!(&row[..4], &[63.5, -64.0, 0.5, -0.5]);
        assert!(row[4..].iter().all(|v| *v == 0.0));
    }

    #[test]
    fn q8g64_encode_decode_round_trip_is_within_the_quantisation_step() {
        let values: Vec<f32> = (0..64).map(|i| (i as f32) * 0.5 - 16.0).collect();
        let bytes = q8g64_encode(&values, 1, 64).unwrap();
        assert_eq!(bytes.len(), 68);
        let decoded = q8g64_row(&bytes, 64, 0);
        let scale = 16.0f32 / 127.0; // max|w| = 16.0 (values span −16.0..15.5)
        for (got, want) in decoded.iter().zip(&values) {
            assert!(
                (got - want).abs() <= scale.max(f32::EPSILON),
                "{got} vs {want}"
            );
        }
    }

    #[test]
    fn q6g64_pack_unpack_round_trips_all_64_values() {
        let mut values = [0u8; 64];
        for (i, v) in values.iter_mut().enumerate() {
            *v = ((i * 7) as u8) & 0x3f; // every 6-bit pattern, wrapping
        }
        assert_eq!(q6g64_unpack_row(q6g64_pack_row(values)), values);
        // All-ones and all-max edge cases.
        assert_eq!(q6g64_unpack_row(q6g64_pack_row([0u8; 64])), [0u8; 64]);
        assert_eq!(q6g64_unpack_row(q6g64_pack_row([0x3f; 64])), [0x3f; 64]);
    }

    #[test]
    fn q6g64_row_decodes_a_hand_built_block() {
        // One row of 64 elements: stride pad16(13·64/16) = 64 bytes
        // (52-byte block + 12 bytes padding). Scale 1.0, values 32, 33,
        // 31, 0, 63 → 0, 1, -1, -32, 31.
        let mut bytes = vec![0u8; 64];
        bytes[0..4].copy_from_slice(&1.0f32.to_le_bytes());
        let mut packed = [0u8; 64];
        packed[0] = 32;
        packed[1] = 33;
        packed[2] = 31;
        packed[3] = 0;
        packed[4] = 63;
        bytes[4..52].copy_from_slice(&q6g64_pack_row(packed));
        let weights = Weights::Q6g64 { bytes };
        let row = weights.row(64, 0);
        assert_eq!(&row[..5], &[0.0, 1.0, -1.0, -32.0, 31.0]);
        assert!(row[5..].iter().all(|v| *v == -32.0), "zero nibbles → -32");
    }

    #[test]
    fn q6g64_encode_decode_round_trip_is_within_the_quantisation_step() {
        let values: Vec<f32> = (0..64).map(|i| (i as f32) * 0.25 - 8.0).collect();
        let bytes = q6g64_encode(&values, 1, 64).unwrap();
        assert_eq!(bytes.len(), 64); // pad16(52) = 64 for k = 64
        let decoded = q6g64_row(&bytes, 64, 0);
        let scale = 8.0f32 / 32.0; // max|w| = 8.0
        for (got, want) in decoded.iter().zip(&values) {
            assert!((got - want).abs() <= scale, "{got} vs {want}");
        }
    }

    #[test]
    fn ht_row_applies_the_planes_to_the_nibbles() {
        // 2 rows × 4 cols: row0 bytes 0x87, 0x69 (nibbles 7,8,9,6),
        // row1 bytes 0xf8, 0x10 (nibbles 8,15,0,1); suh [1,2,3,4],
        // svh [2,3].
        let bytes = vec![0x87, 0x69, 0xf8, 0x10];
        let weights = Weights::Ht {
            bytes,
            suh: vec![1.0, 2.0, 3.0, 4.0],
            svh: vec![2.0, 3.0],
            svh_rows: 2,
        };
        // w[r][k] = suh[k] · svh[r] · (nibble − 8)
        assert_eq!(weights.row(4, 0), vec![-2.0, 0.0, 6.0, -16.0]);
        assert_eq!(weights.row(4, 1), vec![0.0, 42.0, -72.0, -84.0]);
    }

    #[test]
    fn ht_encode_decode_stays_within_the_nibble_grid() {
        let values: Vec<f32> = (0..8).map(|i| (i as f32) * 0.5 - 1.75).collect();
        let bytes = ht_encode(&values, 1, 8).unwrap();
        assert_eq!(bytes.len(), 4);
        // With unit planes the decoded row is the nibble grid the
        // encoder quantised to.
        let weights = Weights::Ht {
            bytes,
            suh: vec![1.0; 8],
            svh: vec![1.0],
            svh_rows: 1,
        };
        let row = weights.row(8, 0);
        let max = values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        for (got, want) in row.iter().zip(&values) {
            // Decoded value = nibble − 8 ≈ 8·want/max, step 8/max.
            let approx = 8.0 * want / max;
            assert!(
                (got - approx).abs() <= 1.0,
                "{got} vs {approx} (want {want})"
            );
        }
    }

    #[test]
    fn i4r_row_decodes_a_hand_built_block() {
        // One row of 64 elements: 33-byte block, scale byte 0x38 (1.0),
        // nibbles 8,8,7,9,0,15,...; planes suh [1,2,...], svh [3].
        let mut bytes = vec![0u8; 33];
        bytes[32] = 0x38; // fp8 e4m3 1.0
        bytes[0] = 0x88; // elements 0,1 → nibbles 8,8 → 0, 0
        bytes[1] = 0x97; // elements 2,3 → nibbles 7,9 → −1, 1
        bytes[2] = 0xf0; // elements 4,5 → nibbles 0,15 → −8, 7
        let suh: Vec<f32> = (0..64).map(|i| (i as f32) + 1.0).collect();
        let weights = Weights::I4r {
            bytes,
            suh,
            svh: vec![3.0],
            svh_rows: 1,
        };
        let row = weights.row(64, 0);
        // w = scale · (nibble − 8) · suh[k] · svh[0]
        assert_eq!(
            &row[..6],
            &[
                0.0,
                0.0,
                -9.0,
                1.0 * 4.0 * 3.0,
                -8.0 * 5.0 * 3.0,
                7.0 * 6.0 * 3.0
            ]
        );
        assert_eq!(row[6], -8.0 * 7.0 * 3.0, "zero nibble → −8 · suh[6] · svh");
        assert_eq!(row[63], -8.0 * 64.0 * 3.0);
    }

    #[test]
    fn i4r_encode_decode_round_trip_is_within_the_quantisation_step() {
        let values: Vec<f32> = (0..64).map(|i| (i as f32) * 0.25 - 8.0).collect();
        let bytes = i4r_encode(&values, 1, 64).unwrap();
        assert_eq!(bytes.len(), 33);
        let suh = vec![1.0f32; 64];
        let decoded = i4r_row(&bytes, 64, 0, &suh, &[1.0], 1);
        let scale = f8e4m3_to_f32(bytes[32]); // max|w| = 8 → 8/8 = 1.0 exactly
        assert_eq!(scale, 1.0);
        for (got, want) in decoded.iter().zip(&values) {
            assert!((got - want).abs() <= scale, "{got} vs {want}");
        }
    }

    #[test]
    fn from_dtype_validates_every_codec_layout() {
        // Dense dtypes keep the float decode (and reject length mismatches).
        assert!(Weights::from_dtype(0, &[], &[0], None).is_some());
        assert!(Weights::from_dtype(0, &[0], &[1, 1], None).is_none());
        // i32/i64 have no codec.
        assert!(Weights::from_dtype(3, &[0; 4], &[1], None).is_none());
        assert!(Weights::from_dtype(4, &[0; 8], &[1], None).is_none());
        assert!(Weights::from_dtype(99, &[], &[0], None).is_none());
        // q4c: k % 32 and the exact 64 + rows·pad16(9k/16) length
        // (k = 32 → stride pad16(18) = 32 → 96 bytes).
        assert!(Weights::from_dtype(5, &[0; 96], &[1, 32], None).is_some());
        assert!(Weights::from_dtype(5, &[0; 95], &[1, 32], None).is_none());
        assert!(Weights::from_dtype(5, &[0; 96], &[1, 33], None).is_none());
        // q8g64: k % 64 and rows·(k/64)·68.
        assert!(Weights::from_dtype(7, &[0; 68], &[1, 64], None).is_some());
        assert!(Weights::from_dtype(7, &[0; 67], &[1, 64], None).is_none());
        // q6g64: k % 64 and rows·pad16(13k/16).
        assert!(Weights::from_dtype(24, &[0; 64], &[1, 64], None).is_some());
        assert!(Weights::from_dtype(24, &[0; 52], &[1, 64], None).is_none());
        // ht: k even, rows·k/2 bytes, planes of length k and rows.
        let suh = vec![1.0f32; 4];
        let svh = vec![1.0f32; 2];
        assert!(Weights::from_dtype(16, &[0; 4], &[2, 4], Some((&suh, &svh))).is_some());
        assert!(Weights::from_dtype(16, &[0; 4], &[2, 4], None).is_none());
        assert!(Weights::from_dtype(16, &[0; 4], &[2, 5], Some((&suh, &svh))).is_none());
        // i4r: k % 64 and rows·(k/64)·33 plus planes.
        let suh64 = vec![1.0f32; 64];
        assert!(Weights::from_dtype(23, &[0; 33], &[1, 64], Some((&suh64, &[1.0]))).is_some());
        assert!(Weights::from_dtype(23, &[0; 33], &[1, 64], None).is_none());
        // 3-D i4r takes svh from dims[1] (the per-expert plane): rows is
        // a·b, so [4, 16, 64] needs 64 × 33 = 2112 payload bytes.
        let stacked = vec![0u8; 2112];
        assert!(
            Weights::from_dtype(23, &stacked, &[4, 16, 64], Some((&suh64, &[1.0; 16]))).is_some()
        );
        assert!(
            Weights::from_dtype(23, &stacked, &[4, 16, 64], Some((&suh64, &[1.0; 4]))).is_none()
        );
    }

    #[test]
    fn decode_facade_matches_row_by_row_dequantisation() {
        // A 2-row q8g64 tensor: the facade and the per-row decode agree.
        let values: Vec<f32> = (0..128).map(|i| (i as f32) * 0.1 - 6.4).collect();
        let bytes = q8g64_encode(&values, 2, 64).unwrap();
        let all = decode(7, &bytes, &[2, 64], None).unwrap();
        assert_eq!(all.len(), 128);
        let weights = Weights::from_dtype(7, &bytes, &[2, 64], None).unwrap();
        assert_eq!(all, weights.dequant_all(2, 64));
        assert_eq!(
            all,
            weights
                .row(64, 0)
                .into_iter()
                .chain(weights.row(64, 1))
                .collect::<Vec<_>>()
        );
    }
}
