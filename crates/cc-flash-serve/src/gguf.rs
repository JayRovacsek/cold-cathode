//! Minimal GGUF header/metadata reader for the repack validation chain.
//!
//! Mirrors the reference binary's checks (spec: "Flag table — repack mode"
//! plus the probed wording for the version, string-length and MTP-head
//! errors — see the Phase 4 commit notes for the probe evidence). Only the
//! structure is validated; tensor data is never parsed.

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
}

/// The subset of GGUF metadata `flash_serve` inspects.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Meta {
    pub version: u32,
    pub n_tensors: u64,
    pub n_kv: u64,
    /// `general.architecture`; empty when the key is absent.
    pub architecture: String,
}

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
        return Err(Failure {
            message: format!(
                "gguf: {}: GGUF version {version} (2 or 3 expected)",
                path.display()
            ),
            code: 1,
        });
    }
    // The MTP head is validated before the metadata (probed ordering:
    // a complete header with a bad head reports the head, not the
    // architecture).
    if let Some(head_path) = head {
        let head_data = fs::read(head_path).map_err(|_| Failure {
            message: format!(
                "gguf: cannot open the MTP head file {} (HALOGEN_MTP_HEAD)",
                head_path.display()
            ),
            code: 1,
        })?;
        // The reference reads a fixed head header; files below it yield
        // "short read" (probed at 20 and 96 bytes).
        if head_data.len() < 128 {
            return Err(Failure {
                message: "gguf: short read in the head file's header".to_string(),
                code: 1,
            });
        }
    }
    let n_tensors = u64::from_le_bytes(data[8..16].try_into().expect("8 bytes checked"));
    let n_kv = u64::from_le_bytes(data[16..24].try_into().expect("8 bytes checked"));

    let mut off = 24usize;
    let mut architecture = String::new();
    for _ in 0..n_kv {
        let key = read_string(&data, &mut off, path)?;
        let kind = read_u32(&data, &mut off, path)?;
        let value = read_typed(&data, &mut off, path, kind, 0)?;
        if key == "general.architecture" {
            if let Some(text) = value {
                architecture = text;
            }
        }
    }
    Ok(Meta {
        version,
        n_tensors,
        n_kv,
        architecture,
    })
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

/// Read one value of the given GGUF value type (0..=10). Returns the text
/// for strings so callers can capture `general.architecture`.
fn read_typed(
    data: &[u8],
    off: &mut usize,
    path: &Path,
    kind: u32,
    depth: u32,
) -> Result<Option<String>, Failure> {
    match kind {
        // uint8, int8, bool
        0 | 1 | 7 => {
            *off += 1;
            check_len(data, *off, path)?;
            Ok(None)
        }
        // uint16, int16
        2 | 3 => {
            *off += 2;
            check_len(data, *off, path)?;
            Ok(None)
        }
        // uint32, int32, float32
        4..=6 => {
            *off += 4;
            check_len(data, *off, path)?;
            Ok(None)
        }
        // string
        8 => Ok(Some(read_string(data, off, path)?)),
        // float64
        10 => {
            *off += 8;
            check_len(data, *off, path)?;
            Ok(None)
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
            for _ in 0..count.min(4096) {
                read_typed(data, off, path, element_kind, depth + 1)?;
            }
            Ok(None)
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
    use std::io::Write;
    use std::path::PathBuf;

    fn fixture(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
        let path = dir.join(name);
        let mut file = fs::File::create(&path).expect("create fixture");
        file.write_all(bytes).expect("write fixture");
        path
    }

    fn header_v3_with_kv(n_kv: u64) -> Vec<u8> {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&n_kv.to_le_bytes());
        bytes
    }

    /// GGUF v3 with one metadata entry: general.architecture = value.
    fn gguf_with_arch(value: &str) -> Vec<u8> {
        let mut bytes = header_v3_with_kv(1);
        bytes.extend_from_slice(&20u64.to_le_bytes()); // key length
        bytes.extend_from_slice(b"general.architecture");
        bytes.extend_from_slice(&8u32.to_le_bytes()); // value type: string
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
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
        let dir = std::env::temp_dir();
        let path = fixture(
            &dir,
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
        let dir = std::env::temp_dir();
        let path = fixture(&dir, "cc-flash-serve-stub.gguf", b"GGUF");
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
        let dir = std::env::temp_dir();
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        let path = fixture(&dir, "cc-flash-serve-v1.gguf", &bytes);
        let err = inspect(&path, None).unwrap_err();
        assert_eq!(
            err.message,
            format!("gguf: {}: GGUF version 1 (2 or 3 expected)", path.display())
        );
        assert_eq!(err.code, 1);
        fs::remove_file(path).ok();
    }

    #[test]
    fn complete_header_parses_with_no_metadata() {
        let dir = std::env::temp_dir();
        let path = fixture(&dir, "cc-flash-serve-v3.gguf", &header_v3_with_kv(0));
        let meta = inspect(&path, None).expect("parses");
        assert_eq!(meta.version, 3);
        assert_eq!(meta.n_kv, 0);
        assert_eq!(meta.architecture, "");
        fs::remove_file(path).ok();
    }

    #[test]
    fn architecture_is_read_from_metadata() {
        let dir = std::env::temp_dir();
        let path = fixture(
            &dir,
            "cc-flash-serve-qwen.gguf",
            &gguf_with_arch("qwen4exp"),
        );
        let meta = inspect(&path, None).expect("parses");
        assert_eq!(meta.architecture, "qwen4exp");
        fs::remove_file(path).ok();
    }

    #[test]
    fn absurd_string_length_is_rejected() {
        let dir = std::env::temp_dir();
        let mut bytes = header_v3_with_kv(1);
        bytes.extend_from_slice(&(1u64 << 32).to_le_bytes()); // key length 4 GiB
        let path = fixture(&dir, "cc-flash-serve-absurd.gguf", &bytes);
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
        let dir = std::env::temp_dir();
        let mut bytes = header_v3_with_kv(1);
        bytes.extend_from_slice(&1000u64.to_le_bytes()); // key length beyond EOF
        let path = fixture(&dir, "cc-flash-serve-long.gguf", &bytes);
        let err = inspect(&path, None).unwrap_err();
        assert_eq!(
            err.message,
            format!("gguf: {}: truncated header", path.display())
        );
        fs::remove_file(path).ok();
    }

    #[test]
    fn missing_head_file_is_reported() {
        let dir = std::env::temp_dir();
        let path = fixture(&dir, "cc-flash-serve-head.gguf", &header_v3_with_kv(0));
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
        let dir = std::env::temp_dir();
        let path = fixture(&dir, "cc-flash-serve-heads.gguf", &header_v3_with_kv(0));
        let head = fixture(&dir, "cc-flash-serve-head.hgn", b"GGUF");
        let err = inspect(&path, Some(&head)).unwrap_err();
        assert_eq!(err.message, "gguf: short read in the head file's header");
        assert_eq!(err.code, 1);
        fs::remove_file(path).ok();
        fs::remove_file(head).ok();
    }
}
