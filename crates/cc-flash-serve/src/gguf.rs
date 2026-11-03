//! Minimal GGUF reader for the repack validation chain.
//!
//! Mirrors the reference binary's full probed validation chain (spec:
//! "Flag table — repack mode" — Input validation; review F2):
//!
//! ```text
//! magic → header length → version → metadata parse → tensor-info records
//!       → MTP head → architecture → qwen4exp metadata schema
//! ```
//!
//! Every error string in this module was probed against the reference
//! binary (v0.18.21) or verified by the code review of `49fd776`/`404947f`.
//! Only the structure is validated; tensor data is never parsed. The
//! reference's next layer (the tensor shape contract, e.g.
//! `gguf: the shape contract failed:\nlayers.1.ple...`) is observed but not
//! yet ported, so a GGUF carrying the full qwen4exp schema still exits 0
//! here (see the `repack-valid-gguf` matrix deferral).

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

/// A failed validation: the exact stderr line and the exit code.
#[derive(Debug, PartialEq, Eq)]
pub struct Failure {
    pub message: String,
    pub code: i32,
}

impl Failure {
    fn not_gguf(path: &Path) -> Self {
        Failure {
            message: format!("gguf: {} is not a GGUF", path.display()),
            code: 2,
        }
    }

    fn truncated(path: &Path) -> Self {
        Failure {
            message: format!("gguf: {}: truncated header", path.display()),
            code: 1,
        }
    }

    fn absurd_string(path: &Path) -> Self {
        Failure {
            message: format!(
                "gguf: {}: absurd string length in the header",
                path.display()
            ),
            code: 1,
        }
    }

    fn version(path: &Path, version: u32) -> Self {
        Failure {
            message: format!(
                "gguf: {}: GGUF version {version} (2 or 3 expected)",
                path.display()
            ),
            code: 1,
        }
    }

    fn head_unopenable(head: &Path) -> Self {
        Failure {
            message: format!(
                "gguf: cannot open the MTP head file {} (HALOGEN_MTP_HEAD)",
                head.display()
            ),
            code: 1,
        }
    }

    fn head_short() -> Self {
        Failure {
            message: "gguf: short read in the head file's header".to_string(),
            code: 1,
        }
    }

    fn tensor_past_shard(path: &Path, name: &str) -> Self {
        Failure {
            message: format!(
                "gguf: {}: {name} runs past the end of the shard",
                path.display()
            ),
            code: 1,
        }
    }

    fn schema_mismatch(key: &str, actual: &str, expected: &str) -> Self {
        Failure {
            message: format!("gguf: {key} = {actual}, expected {expected}"),
            code: 1,
        }
    }

    fn array_missing(key: &str) -> Self {
        Failure {
            message: format!("gguf: {key}: missing from the KV"),
            code: 1,
        }
    }

    fn architecture(arch: &str) -> Self {
        Failure {
            message: format!(
                "gguf: architecture '{arch}': this engine reads qwen4exp (Qwen3.8-Flash-Next) only"
            ),
            code: 1,
        }
    }
}

/// The subset of GGUF metadata `flash_serve` inspects.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Meta {
    pub version: u32,
    pub n_tensors: u64,
    pub n_kv: u64,
    /// `general.architecture`; empty when the key is absent.
    pub architecture: String,
    /// Every metadata entry, rendered as text (needed by the qwen4exp
    /// schema check; `general.architecture` is mirrored in `architecture`).
    pub kv: BTreeMap<String, String>,
}

/// Required qwen4exp scalar metadata keys with the values the reference
/// expects (probed: the reference walks them in this order and reports the
/// first gap as `gguf: KEY = missing, expected N`, or a wrong value as
/// `gguf: KEY = ACTUAL, expected N`).
const QWEN4EXP_SCALARS: &[(&str, &str)] = &[
    ("qwen4exp.ssm.group_count", "16"),
    ("qwen4exp.ssm.time_step_rank", "48"),
    ("qwen4exp.ssm.state_size", "128"),
    ("qwen4exp.block_count", "48"),
    ("qwen4exp.expert_count", "512"),
    ("qwen4exp.embedding_length", "2560"),
];

/// Required qwen4exp array metadata keys (probed: presence-only check — any
/// array value is accepted, a gap reports `gguf: KEY: missing from the KV`).
const QWEN4EXP_ARRAYS: &[&str] = &[
    "qwen4exp.ple.layer_multipliers",
    "qwen4exp.ple.head_offsets",
    "qwen4exp.ple.head_vocab_sizes",
];

/// Validate `path` as a GGUF input for `--repack`/`--repack-hash`.
/// `head` is the resolved MTP head file, or `None` when `--no-head` (or no
/// head is configured).
pub fn inspect(path: &Path, head: Option<&Path>) -> Result<Meta, Failure> {
    let data = fs::read(path).map_err(|_| Failure::not_gguf(path))?;
    if data.len() < 4 || data[..4] != *b"GGUF" {
        return Err(Failure::not_gguf(path));
    }
    // v2/v3 fixed header: magic(4) + version(4) + n_tensors(8) + n_kv(8).
    if data.len() < 24 {
        return Err(Failure::truncated(path));
    }
    let version = u32::from_le_bytes(data[4..8].try_into().expect("4 bytes checked"));
    if version != 2 && version != 3 {
        return Err(Failure::version(path, version));
    }
    let n_tensors = u64::from_le_bytes(data[8..16].try_into().expect("8 bytes checked"));
    let n_kv = u64::from_le_bytes(data[16..24].try_into().expect("8 bytes checked"));

    // Metadata section (probed ordering: the metadata parse precedes the
    // tensor-info and MTP head checks — a file with a broken metadata
    // section and a missing head reports the metadata, not the head).
    let mut off = 24usize;
    let mut kv = BTreeMap::new();
    let mut architecture = String::new();
    for _ in 0..n_kv {
        let key = read_string(&data, &mut off, path)?;
        let kind = read_u32(&data, &mut off, path)?;
        let value = read_typed(&data, &mut off, path, kind, 0)?;
        if key == "general.architecture" {
            architecture = value.clone();
        }
        kv.insert(key, value);
    }
    // Tensor-info section (review F2 layer 1): the n_tensors records must
    // parse and their data ranges must fit the file.
    let tensors = read_tensor_infos(&data, &mut off, n_tensors, path)?;
    check_tensor_ranges(&data, off, &tensors, &kv, path)?;
    // The MTP head is validated after the tensor-info section and before
    // the schema (probed ordering: a complete header with a bad head
    // reports the head, not the architecture).
    if let Some(head_path) = head {
        let head_data = fs::read(head_path).map_err(|_| Failure::head_unopenable(head_path))?;
        // The reference reads a fixed head header; files below it yield
        // "short read" (probed at 20 and 96 bytes).
        if head_data.len() < 128 {
            return Err(Failure::head_short());
        }
    }
    // Architecture + qwen4exp schema (review F2 layer 2). The `?` fallback
    // for a missing arch key was verified by the review of `49fd776`.
    let arch = if architecture.is_empty() {
        "?"
    } else {
        &architecture
    };
    if arch != "qwen4exp" {
        return Err(Failure::architecture(arch));
    }
    validate_qwen4exp_schema(&kv)?;
    Ok(Meta {
        version,
        n_tensors,
        n_kv,
        architecture,
        kv,
    })
}

/// A parsed tensor-info record: name, data offset and byte size.
type Tensor = (String, u64, u64);

/// Read the `count` tensor-info records and return the ones whose dtype the
/// port can size.
///
/// Record layout: name(string) + n_dims(u32) + dims(n_dims×u64) +
/// dtype(u32) + offset(u64). A header declaring more records than the file
/// holds yields `truncated header` (probed: `n_tensors=2^32` in a 24-byte
/// file).
fn read_tensor_infos(
    data: &[u8],
    off: &mut usize,
    count: u64,
    path: &Path,
) -> Result<Vec<Tensor>, Failure> {
    let mut tensors = Vec::new();
    for _ in 0..count {
        let name = read_string(data, off, path)?;
        let n_dims = read_u32(data, off, path)?;
        let mut elements: u64 = 1; // empty product for n_dims = 0 (probed)
        for _ in 0..n_dims {
            elements = elements.saturating_mul(read_u64(data, off, path)?);
        }
        let dtype = read_u32(data, off, path)?;
        let offset = read_u64(data, off, path)?;
        // Probed dtype table: f32 (4 B/element) and f16 (2 B/element) are
        // sized; other dtypes (bf16, quantised forms) are read but not
        // range-checked by the port (the reference sizes quantised types
        // too, e.g. q4_0 — not modelled here).
        if let Some(element_size) = dtype_size(dtype) {
            tensors.push((name, offset, elements.saturating_mul(element_size)));
        }
    }
    Ok(tensors)
}

/// Byte size per element for the tensor dtypes the port range-checks.
fn dtype_size(dtype: u32) -> Option<u64> {
    match dtype {
        0 => Some(4), // f32
        1 => Some(2), // f16
        _ => None,
    }
}

/// Check the data ranges of the parsed tensors (probed:
/// `{name} runs past the end of the shard`). The tensor data section starts
/// `general.alignment` (default 32) bytes past the tensor-info section,
/// exactly as the GGUF format prescribes.
fn check_tensor_ranges(
    data: &[u8],
    tensor_info_end: usize,
    tensors: &[Tensor],
    kv: &BTreeMap<String, String>,
    path: &Path,
) -> Result<(), Failure> {
    let alignment = kv
        .get("general.alignment")
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|a| *a > 0)
        .unwrap_or(32);
    let data_start = align_up(tensor_info_end, alignment);
    for (name, offset, size) in tensors {
        let end = data_start
            .checked_add(*offset)
            .and_then(|o| o.checked_add(*size))
            .ok_or_else(|| Failure::tensor_past_shard(path, name))?;
        if end > data.len() as u64 {
            return Err(Failure::tensor_past_shard(path, name));
        }
    }
    Ok(())
}

fn align_up(value: usize, alignment: u64) -> u64 {
    let alignment = alignment.max(1);
    let value = u64::try_from(value).unwrap_or(u64::MAX);
    let remainder = value % alignment;
    if remainder == 0 {
        value
    } else {
        value + alignment - remainder
    }
}

/// Walk the qwen4exp metadata schema (review F2 layer 2).
fn validate_qwen4exp_schema(kv: &BTreeMap<String, String>) -> Result<(), Failure> {
    for (key, expected) in QWEN4EXP_SCALARS {
        match kv.get(*key) {
            None => return Err(Failure::schema_mismatch(key, "missing", expected)),
            Some(actual) if actual != *expected => {
                return Err(Failure::schema_mismatch(key, actual, expected))
            }
            Some(_) => {}
        }
    }
    for key in QWEN4EXP_ARRAYS {
        if !kv.contains_key(*key) {
            return Err(Failure::array_missing(key));
        }
    }
    Ok(())
}

/// Read a length-prefixed string body, advancing `off`.
fn read_string(data: &[u8], off: &mut usize, path: &Path) -> Result<String, Failure> {
    let len = read_u64(data, off, path)?;
    // Probed: a 34 GiB length is "absurd"; a 1000-byte length in a 72-byte
    // file is "truncated header". The exact cap is inferred as 2^31.
    if len > (1 << 31) {
        return Err(Failure::absurd_string(path));
    }
    let end = off
        .checked_add(len as usize)
        .ok_or_else(|| Failure::truncated(path))?;
    if end > data.len() {
        return Err(Failure::truncated(path));
    }
    let text = String::from_utf8_lossy(&data[*off..end]).into_owned();
    *off = end;
    Ok(text)
}

/// Read one value of the given GGUF value type (0..=10), rendered as text
/// so callers can compare it against the schema (strings verbatim, integers
/// decimal, floats via the reference's `%g` rendering, arrays `[a,b,...]`).
fn read_typed(
    data: &[u8],
    off: &mut usize,
    path: &Path,
    kind: u32,
    depth: u32,
) -> Result<String, Failure> {
    match kind {
        // uint8, int8, bool
        0 | 1 | 7 => {
            check_len(data, *off + 1, path)?;
            let byte = data[*off];
            *off += 1;
            Ok(match kind {
                1 => i8::from_le_bytes([byte]).to_string(),
                _ => byte.to_string(),
            })
        }
        // uint16, int16
        2 | 3 => {
            check_len(data, *off + 2, path)?;
            let bytes = data[*off..*off + 2].try_into().expect("2 bytes checked");
            *off += 2;
            Ok(if kind == 2 {
                u16::from_le_bytes(bytes).to_string()
            } else {
                i16::from_le_bytes(bytes).to_string()
            })
        }
        // uint32, int32, float32
        4..=6 => {
            check_len(data, *off + 4, path)?;
            let bytes = data[*off..*off + 4].try_into().expect("4 bytes checked");
            *off += 4;
            Ok(match kind {
                4 => u32::from_le_bytes(bytes).to_string(),
                5 => i32::from_le_bytes(bytes).to_string(),
                _ => crate::fmt_g(f32::from_le_bytes(bytes) as f64),
            })
        }
        // string
        8 => read_string(data, off, path),
        // float64
        10 => {
            check_len(data, *off + 8, path)?;
            let bytes = data[*off..*off + 8].try_into().expect("8 bytes checked");
            *off += 8;
            Ok(crate::fmt_g(f64::from_le_bytes(bytes)))
        }
        // array: element type (u32) + count (u64) + elements
        9 => {
            if depth > 8 {
                return Err(Failure::truncated(path));
            }
            let element_kind = read_u32(data, off, path)?;
            let count = read_u64(data, off, path)?;
            let element_size = match element_kind {
                0 | 1 | 7 => 1u64,
                2 | 3 => 2,
                4..=6 => 4,
                8 | 9 => 1, // strings/arrays need per-element parsing below
                10 => 8,
                _ => return Err(Failure::truncated(path)),
            };
            let remaining = (data.len() - *off) as u64;
            if count > remaining / element_size.max(1) && element_size > 1 {
                return Err(Failure::truncated(path));
            }
            let mut elements = Vec::new();
            for _ in 0..count.min(4096) {
                elements.push(read_typed(data, off, path, element_kind, depth + 1)?);
            }
            Ok(format!("[{}]", elements.join(",")))
        }
        _ => Err(Failure::truncated(path)),
    }
}

fn check_len(data: &[u8], end: usize, path: &Path) -> Result<(), Failure> {
    if end > data.len() {
        return Err(Failure::truncated(path));
    }
    Ok(())
}

fn read_u64(data: &[u8], off: &mut usize, path: &Path) -> Result<u64, Failure> {
    let end = off.checked_add(8).ok_or_else(|| Failure::truncated(path))?;
    if end > data.len() {
        return Err(Failure::truncated(path));
    }
    let value = u64::from_le_bytes(data[*off..end].try_into().expect("8 bytes checked"));
    *off = end;
    Ok(value)
}

fn read_u32(data: &[u8], off: &mut usize, path: &Path) -> Result<u32, Failure> {
    let end = off.checked_add(4).ok_or_else(|| Failure::truncated(path))?;
    if end > data.len() {
        return Err(Failure::truncated(path));
    }
    let value = u32::from_le_bytes(data[*off..end].try_into().expect("4 bytes checked"));
    *off = end;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scratch_dir;
    use std::io::Write;
    use std::path::PathBuf;

    fn fixture(name: &str, bytes: &[u8]) -> PathBuf {
        let path = scratch_dir().join(name);
        let mut file = fs::File::create(&path).expect("create fixture");
        file.write_all(bytes).expect("write fixture");
        path
    }

    fn header_v3(n_tensors: u64, n_kv: u64) -> Vec<u8> {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&n_tensors.to_le_bytes());
        bytes.extend_from_slice(&n_kv.to_le_bytes());
        bytes
    }

    /// A GGUF v3 carrying the given metadata entries (key, value type,
    /// encoded value bytes) and `n_tensors` tensor-info records.
    fn gguf(kvs: &[(&str, u32, &[u8])], n_tensors: u64) -> Vec<u8> {
        let mut bytes = header_v3(n_tensors, kvs.len() as u64);
        for (key, kind, value) in kvs {
            bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
            bytes.extend_from_slice(key.as_bytes());
            bytes.extend_from_slice(&kind.to_le_bytes());
            bytes.extend_from_slice(value);
        }
        bytes
    }

    /// `gguf` over owned value vectors.
    fn gguf_ref(kvs: &[(&str, u32, Vec<u8>)], n_tensors: u64) -> Vec<u8> {
        gguf(
            &kvs.iter()
                .map(|(k, t, v)| (*k, *t, v.as_slice()))
                .collect::<Vec<_>>(),
            n_tensors,
        )
    }

    /// Encoded string value (length prefix + text).
    fn str_value(text: &str) -> Vec<u8> {
        (text.len() as u64)
            .to_le_bytes()
            .into_iter()
            .chain(text.as_bytes().iter().copied())
            .collect()
    }

    /// A qwen4exp GGUF carrying the given scalar schema entries.
    fn qwen4exp_with(scalars: &[(&str, &str)]) -> Vec<u8> {
        let mut kvs: Vec<(&str, u32, Vec<u8>)> =
            vec![("general.architecture", 8, str_value("qwen4exp"))];
        for (key, value) in scalars {
            let value_bytes = value.parse::<u32>().expect("u32 schema value");
            kvs.push((key, 4, value_bytes.to_le_bytes().to_vec()));
        }
        gguf(
            &kvs.iter()
                .map(|(k, t, v)| (*k, *t, v.as_slice()))
                .collect::<Vec<_>>(),
            0,
        )
    }

    /// A tensor-info record for an f32 tensor with `elements` elements at
    /// `offset`.
    fn f32_tensor(name: &str, elements: u64, offset: u64) -> Vec<u8> {
        let mut bytes = str_value(name);
        bytes.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        bytes.extend_from_slice(&elements.to_le_bytes()); // dims[0]
        bytes.extend_from_slice(&0u32.to_le_bytes()); // dtype f32
        bytes.extend_from_slice(&offset.to_le_bytes());
        bytes
    }

    #[test]
    fn missing_file_is_not_a_gguf() {
        let err = inspect(Path::new("/nonexistent"), None).unwrap_err();
        assert_eq!(err.message, "gguf: /nonexistent is not a GGUF");
        assert_eq!(err.code, 2);
    }

    #[test]
    fn junk_file_is_not_a_gguf() {
        let path = fixture(
            "cc-flash-serve-junk.gguf",
            b"not a gguf at all, longer than 24 bytes here",
        );
        let err = inspect(&path, None).unwrap_err();
        assert_eq!(
            err.message,
            format!("gguf: {} is not a GGUF", path.display())
        );
        assert_eq!(err.code, 2);
        fs::remove_file(path).ok();
    }

    #[test]
    fn magic_but_short_is_truncated() {
        let path = fixture("cc-flash-serve-stub.gguf", b"GGUF");
        let err = inspect(&path, None).unwrap_err();
        assert_eq!(
            err.message,
            format!("gguf: {}: truncated header", path.display())
        );
        assert_eq!(err.code, 1);
        fs::remove_file(path).ok();
    }

    #[test]
    fn bad_version_is_rejected() {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        let path = fixture("cc-flash-serve-v1.gguf", &bytes);
        let err = inspect(&path, None).unwrap_err();
        assert_eq!(
            err.message,
            format!("gguf: {}: GGUF version 1 (2 or 3 expected)", path.display())
        );
        assert_eq!(err.code, 1);
        fs::remove_file(path).ok();
    }

    #[test]
    fn bare_v3_header_reports_unknown_architecture() {
        // Probed (review §4): a bare v3 header parses but the reference
        // rejects the missing architecture with the `?` fallback.
        let path = fixture("cc-flash-serve-v3.gguf", &header_v3(0, 0));
        let err = inspect(&path, None).unwrap_err();
        assert_eq!(
            err.message,
            "gguf: architecture '?': this engine reads qwen4exp (Qwen3.8-Flash-Next) only"
        );
        assert_eq!(err.code, 1);
        fs::remove_file(path).ok();
    }

    #[test]
    fn header_with_llama_architecture_is_rejected() {
        let path = fixture(
            "cc-flash-serve-llama.gguf",
            &gguf(&[("general.architecture", 8, &str_value("llama"))], 0),
        );
        let err = inspect(&path, None).unwrap_err();
        assert_eq!(
            err.message,
            "gguf: architecture 'llama': this engine reads qwen4exp (Qwen3.8-Flash-Next) only"
        );
        assert_eq!(err.code, 1);
        fs::remove_file(path).ok();
    }

    #[test]
    fn qwen4exp_scalar_chain_matches_reference() {
        // Probed chain: each fixture adds the previously demanded key and
        // the reference reports the next gap.
        let cases: Vec<(Vec<(&str, &str)>, String)> = vec![
            (
                vec![],
                "gguf: qwen4exp.ssm.group_count = missing, expected 16".to_string(),
            ),
            (
                vec![("qwen4exp.ssm.group_count", "16")],
                "gguf: qwen4exp.ssm.time_step_rank = missing, expected 48".to_string(),
            ),
            (
                vec![
                    ("qwen4exp.ssm.group_count", "16"),
                    ("qwen4exp.ssm.time_step_rank", "48"),
                ],
                "gguf: qwen4exp.ssm.state_size = missing, expected 128".to_string(),
            ),
            (
                vec![
                    ("qwen4exp.ssm.group_count", "16"),
                    ("qwen4exp.ssm.time_step_rank", "48"),
                    ("qwen4exp.ssm.state_size", "128"),
                ],
                "gguf: qwen4exp.block_count = missing, expected 48".to_string(),
            ),
            (
                vec![
                    ("qwen4exp.ssm.group_count", "16"),
                    ("qwen4exp.ssm.time_step_rank", "48"),
                    ("qwen4exp.ssm.state_size", "128"),
                    ("qwen4exp.block_count", "48"),
                ],
                "gguf: qwen4exp.expert_count = missing, expected 512".to_string(),
            ),
            (
                vec![
                    ("qwen4exp.ssm.group_count", "16"),
                    ("qwen4exp.ssm.time_step_rank", "48"),
                    ("qwen4exp.ssm.state_size", "128"),
                    ("qwen4exp.block_count", "48"),
                    ("qwen4exp.expert_count", "512"),
                ],
                "gguf: qwen4exp.embedding_length = missing, expected 2560".to_string(),
            ),
        ];
        for (scalars, expected) in cases {
            let path = fixture("cc-flash-serve-chain.gguf", &qwen4exp_with(&scalars));
            let err = inspect(&path, None).unwrap_err();
            assert_eq!(err.message, expected, "wrong error for {scalars:?}");
            assert_eq!(err.code, 1);
            fs::remove_file(path).ok();
        }
    }

    #[test]
    fn qwen4exp_wrong_value_is_reported() {
        let path = fixture(
            "cc-flash-serve-wrong.gguf",
            &qwen4exp_with(&[("qwen4exp.ssm.group_count", "3")]),
        );
        let err = inspect(&path, None).unwrap_err();
        assert_eq!(
            err.message,
            "gguf: qwen4exp.ssm.group_count = 3, expected 16"
        );
        assert_eq!(err.code, 1);
        fs::remove_file(path).ok();
    }

    #[test]
    fn qwen4exp_array_chain_matches_reference() {
        // Probed: after the scalars, the array keys report
        // `gguf: KEY: missing from the KV` in order; any array value passes.
        let mut scalars: Vec<(&str, u32, Vec<u8>)> =
            vec![("general.architecture", 8, str_value("qwen4exp"))];
        for (key, expected) in QWEN4EXP_SCALARS {
            scalars.push((
                *key,
                4,
                expected.parse::<u32>().unwrap().to_le_bytes().to_vec(),
            ));
        }
        let kvs_of = |arrays: &[&'static str]| -> Vec<(&'static str, u32, Vec<u8>)> {
            let mut kvs = scalars.clone();
            for key in arrays {
                kvs.push((*key, 9, empty_u32_array()));
            }
            kvs
        };
        for i in 0..QWEN4EXP_ARRAYS.len() {
            let path = fixture(
                "cc-flash-serve-arrays.gguf",
                &gguf_ref(&kvs_of(&QWEN4EXP_ARRAYS[..i]), 0),
            );
            let err = inspect(&path, None).unwrap_err();
            assert_eq!(
                err.message,
                format!("gguf: {}: missing from the KV", QWEN4EXP_ARRAYS[i]),
                "wrong error for array index {i}"
            );
            assert_eq!(err.code, 1);
            fs::remove_file(path).ok();
        }
        // With the full schema the port accepts the file (the reference's
        // next layer — the shape contract — is not ported).
        let path = fixture(
            "cc-flash-serve-full.gguf",
            &gguf_ref(&kvs_of(QWEN4EXP_ARRAYS), 0),
        );
        let meta = inspect(&path, None).expect("full schema parses");
        assert_eq!(meta.architecture, "qwen4exp");
        assert_eq!(meta.version, 3);
        fs::remove_file(path).ok();
    }

    fn empty_u32_array() -> Vec<u8> {
        4u32.to_le_bytes().into_iter().chain([0u8; 8]).collect()
    }

    #[test]
    fn bogus_tensor_count_is_truncated() {
        // Probed (review F2): a complete v3 header declaring
        // n_tensors=2^32 with no records → `truncated header`.
        let path = fixture("cc-flash-serve-badtensors.gguf", &header_v3(1 << 32, 0));
        let err = inspect(&path, None).unwrap_err();
        assert_eq!(
            err.message,
            format!("gguf: {}: truncated header", path.display())
        );
        assert_eq!(err.code, 1);
        fs::remove_file(path).ok();
    }

    #[test]
    fn tensor_record_truncated_is_reported() {
        // n_tensors=1 but the record bytes are missing.
        let path = fixture("cc-flash-serve-onetensor.gguf", &header_v3(1, 0));
        let err = inspect(&path, None).unwrap_err();
        assert_eq!(
            err.message,
            format!("gguf: {}: truncated header", path.display())
        );
        fs::remove_file(path).ok();
    }

    #[test]
    fn tensor_past_shard_is_reported() {
        // Probed: an f32 tensor with 4 elements at offset 0 in a file whose
        // aligned data section leaves no room → `runs past the end`.
        let mut bytes = gguf(&[("general.architecture", 8, &str_value("llama"))], 1);
        bytes.extend_from_slice(&f32_tensor("blk.0.w", 4, 0));
        let path = fixture("cc-flash-serve-shard.gguf", &bytes);
        let err = inspect(&path, None).unwrap_err();
        assert_eq!(
            err.message,
            format!(
                "gguf: {}: blk.0.w runs past the end of the shard",
                path.display()
            )
        );
        assert_eq!(err.code, 1);
        fs::remove_file(path).ok();
    }

    #[test]
    fn tensor_within_shard_parses() {
        // Same fixture padded to align_up(108, 32) + 16 bytes of tensor data
        // → the range check passes and the arch error surfaces.
        let mut bytes = gguf(&[("general.architecture", 8, &str_value("llama"))], 1);
        bytes.extend_from_slice(&f32_tensor("blk.0.w", 4, 0));
        bytes.extend_from_slice(&[0u8; 36]); // data section: 128..144
        let path = fixture("cc-flash-serve-shard16.gguf", &bytes);
        let err = inspect(&path, None).unwrap_err();
        assert_eq!(
            err.message,
            "gguf: architecture 'llama': this engine reads qwen4exp (Qwen3.8-Flash-Next) only"
        );
        fs::remove_file(path).ok();
    }

    #[test]
    fn absurd_string_length_is_rejected() {
        let mut bytes = header_v3(0, 1);
        bytes.extend_from_slice(&(1u64 << 32).to_le_bytes()); // key length 4 GiB
        let path = fixture("cc-flash-serve-absurd.gguf", &bytes);
        let err = inspect(&path, None).unwrap_err();
        assert_eq!(
            err.message,
            format!(
                "gguf: {}: absurd string length in the header",
                path.display()
            )
        );
        assert_eq!(err.code, 1);
        fs::remove_file(path).ok();
    }

    #[test]
    fn string_longer_than_file_is_truncated() {
        let mut bytes = header_v3(0, 1);
        bytes.extend_from_slice(&1000u64.to_le_bytes()); // key length beyond EOF
        let path = fixture("cc-flash-serve-long.gguf", &bytes);
        let err = inspect(&path, None).unwrap_err();
        assert_eq!(
            err.message,
            format!("gguf: {}: truncated header", path.display())
        );
        fs::remove_file(path).ok();
    }

    #[test]
    fn missing_head_file_is_reported() {
        let path = fixture("cc-flash-serve-head.gguf", &header_v3(0, 0));
        let err = inspect(&path, Some(Path::new("/nonexistent"))).unwrap_err();
        assert_eq!(
            err.message,
            "gguf: cannot open the MTP head file /nonexistent (HALOGEN_MTP_HEAD)"
        );
        assert_eq!(err.code, 1);
        fs::remove_file(path).ok();
    }

    #[test]
    fn short_head_file_is_reported() {
        let path = fixture("cc-flash-serve-heads.gguf", &header_v3(0, 0));
        let head = fixture("cc-flash-serve-head.hgn", b"GGUF");
        let err = inspect(&path, Some(&head)).unwrap_err();
        assert_eq!(err.message, "gguf: short read in the head file's header");
        assert_eq!(err.code, 1);
        fs::remove_file(path).ok();
        fs::remove_file(head).ok();
    }
}
