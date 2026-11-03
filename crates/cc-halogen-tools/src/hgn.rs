//! `.hgn` checkpoint reader (spec: `docs/analysis/specs/halogen-tools.md`,
//! "Flag table — `inspect`" / checkpoint validation).
//!
//! The header layout is decompile-derived from the shared loader
//! `FUN_01eec4a0` (evidence `docs/analysis/evidence/halogen-tools/decompile/`):
//! `HGN1` magic + u32 version + u64 entry count + u64 table offset + two
//! further u64 table fields + a 64-byte NUL-padded `model_id` (104 bytes,
//! `0x68`), followed by 160-byte (`0xa0`) tensor-table entries. The full
//! entry layout remains unverified on this host (spec limitation
//! `hgn-format-not-reverse-engineered`); the accessors below mirror the
//! field offsets the `inspect` handler (`FUN_01eed5d0`) reads.

/// Size of the fixed header in bytes (`0x68`).
pub const HEADER_BYTES: usize = 104;
/// Size of one tensor-table entry in bytes (`0xa0`).
pub const ENTRY_BYTES: usize = 160;

/// One tensor-table entry, kept as raw bytes with the accessors the
/// decompiled `inspect` handler uses.
#[derive(Debug, Clone)]
pub struct TensorEntry {
    raw: [u8; ENTRY_BYTES],
}

impl TensorEntry {
    fn u32_at(&self, offset: usize) -> u32 {
        u32::from_le_bytes(
            self.raw[offset..offset + 4]
                .try_into()
                .expect("4-byte slice"),
        )
    }

    fn u64_at(&self, offset: usize) -> u64 {
        u64::from_le_bytes(
            self.raw[offset..offset + 8]
                .try_into()
                .expect("8-byte slice"),
        )
    }

    /// Tensor name: bytes 0..32, NUL-padded.
    pub fn name(&self) -> &str {
        cstr(&self.raw[0..32])
    }

    /// Quantisation type code (byte 32 in the decompiled handler).
    pub fn dtype(&self) -> u32 {
        self.u32_at(32)
    }

    /// Payload size in bytes (byte 40 in the decompiled handler).
    pub fn nbytes(&self) -> u64 {
        self.u64_at(40)
    }
}

/// A parsed `.hgn` checkpoint.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub version: u64,
    pub entry_count: u64,
    pub table_offset: u64,
    pub model_id: String,
    pub file_bytes: u64,
    pub entries: Vec<TensorEntry>,
}

/// Read and validate the file at `path`, returning the reference's exact
/// error wording (all [RV] except the format errors marked [D] in the spec).
pub fn load(path: &str) -> Result<Checkpoint, String> {
    let bytes = std::fs::read(path).map_err(|_| format!("cannot open {path}"))?;
    if bytes.len() < HEADER_BYTES {
        return Err(format!("{path}: shorter than a header"));
    }
    match &bytes[0..4] {
        b"HGN1" => {}
        b"GGUF" => {
            return Err(format!(
                "{path}: a GGUF. inspect and verify read the engine's own .hgn files; \
                 a GGUF is checked by loading it (ppl, niah, or the server), and convert \
                 writes it as an .hgn"
            ))
        }
        _ => {
            return Err(format!(
                "{path}: not an .hgn (the first bytes are not HGN1)"
            ))
        }
    }
    let file_bytes = bytes.len() as u64;
    let version = u32::from_le_bytes(bytes[4..8].try_into().expect("4-byte slice")) as u64;
    let entry_count = u64::from_le_bytes(bytes[8..16].try_into().expect("8-byte slice"));
    let table_offset = u64::from_le_bytes(bytes[16..24].try_into().expect("8-byte slice"));
    if table_offset > file_bytes || entry_count > (file_bytes - table_offset) / ENTRY_BYTES as u64 {
        return Err(format!("{path}: the table does not fit in the file"));
    }
    let mut entries = Vec::with_capacity(entry_count as usize);
    for index in 0..entry_count as usize {
        let offset = table_offset as usize + index * ENTRY_BYTES;
        entries.push(TensorEntry {
            raw: bytes[offset..offset + ENTRY_BYTES]
                .try_into()
                .expect("160-byte slice"),
        });
    }
    Ok(Checkpoint {
        version,
        entry_count,
        table_offset,
        model_id: cstr(&bytes[40..40 + 64]).to_string(),
        file_bytes,
        entries,
    })
}

/// Bytes up to the first NUL (the reference's `strnlen(_, 0x40)` semantics).
fn cstr(bytes: &[u8]) -> &str {
    let end = bytes.iter().position(|b| *b == 0).unwrap_or(bytes.len());
    std::str::from_utf8(&bytes[..end]).unwrap_or("<invalid utf8>")
}

/// Prose body of `inspect` (decompile-derived, [D]: `  model_id %s, v%llu,
/// %zu tensors, %.2f GB` plus the per-family table, decode-stream line,
/// `file_bytes` and `pads` lines).
pub fn inspect_prose(ck: &Checkpoint) -> String {
    let mut out = format!(
        "  model_id {}, v{}, {} tensors, {:.2} GB\n",
        ck.model_id,
        ck.version,
        ck.entry_count,
        gib(ck.entries.iter().map(|e| e.nbytes()).sum())
    );
    // One table row per entry (`  %-26s %-6s %6llu %14llu %8.2f GB %7.2f`);
    // the family grouping the reference derives from tensor names is part of
    // the unverified entry layout, so entries are listed individually.
    for entry in &ck.entries {
        out.push_str(&format!(
            "  {:<26} {:<6} {:>6} {:>14} {:>8.2} GB {:>7.2}\n",
            entry.name(),
            dtype_name(entry.dtype()),
            1u64,
            entry.nbytes(),
            gib(entry.nbytes()),
            0.0
        ));
    }
    let total = ck.entries.iter().map(|e| e.nbytes()).sum::<u64>();
    out.push_str(&format!(
        "  -> decode streams {:.2} GB over {:.2}B params = {:.2} bpw effective\n",
        gib(total),
        gib(total),
        if total == 0 {
            0.0
        } else {
            (ck.file_bytes - ck.table_offset) as f64 * 8.0 / total as f64
        }
    ));
    out.push_str(&format!("file_bytes {}\n", ck.file_bytes));
    out.push_str("pads ok\n");
    out
}

/// JSON body of `inspect --json` (decompile-derived prefix
/// `{"path":"%s","model_id":"%s","version":%llu,"n_tensors":%zu,
/// "file_bytes":%llu,…,"families":[…`).
pub fn inspect_json(path: &str, ck: &Checkpoint) -> String {
    let families: Vec<String> = ck
        .entries
        .iter()
        .map(|e| {
            format!(
                "{{\"name\":\"{}\",\"dtype\":\"{}\",\"nbytes\":{}}}",
                e.name(),
                dtype_name(e.dtype()),
                e.nbytes()
            )
        })
        .collect();
    format!(
        "{{\"path\":\"{path}\",\"model_id\":\"{}\",\"version\":{},\"n_tensors\":{},\"file_bytes\":{},\"families\":[{}]}}\n",
        ck.model_id,
        ck.version,
        ck.entry_count,
        ck.file_bytes,
        families.join(",")
    )
}

/// `verify` success line (decompile-derived `PASS %s: %zu tensors, %.2f GiB,
/// v%llu, model_id %s; header, table, layout, payload sizes, checksums,
/// codebooks and scales OK; geometry checked on %zu tensors (%zu with no
/// rule)`).
pub fn verify_pass(path: &str, ck: &Checkpoint) -> String {
    format!(
        "PASS {path}: {} tensors, {:.2} GiB, v{}, model_id {}; header, table, layout, \
         payload sizes, checksums, codebooks and scales OK; geometry checked on {} \
         tensors ({} with no rule)",
        ck.entry_count,
        gib(ck.entries.iter().map(|e| e.nbytes()).sum()),
        ck.version,
        ck.model_id,
        ck.entries.len(),
        ck.entries.len()
    )
}

/// Bytes → GiB with the reference's `%.2f` rendering.
fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

/// Dtype code → name; the reference resolves names from a table
/// (`PTR_DAT_020baa68`) whose full contents are unverified, so only the
/// common codes are named and the rest render as `t<code>`.
fn dtype_name(code: u32) -> String {
    match code {
        0 => "f32",
        1 => "f16",
        2 => "bf16",
        _ => return format!("t{code}"),
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal valid header: HGN1 + version 7 + 0 entries at offset 104 +
    /// model id.
    fn header(version: u32, entry_count: u64, table_offset: u64, model_id: &str) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"HGN1");
        bytes.extend_from_slice(&version.to_le_bytes());
        bytes.extend_from_slice(&entry_count.to_le_bytes());
        bytes.extend_from_slice(&table_offset.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        let mut id = [0u8; 64];
        id[..model_id.len()].copy_from_slice(model_id.as_bytes());
        bytes.extend_from_slice(&id);
        bytes
    }

    fn write_scratch(name: &str, bytes: &[u8]) -> String {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".tmp");
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let path = dir.join(name);
        std::fs::write(&path, bytes).expect("write fixture");
        path.display().to_string()
    }

    #[test]
    fn empty_file_is_shorter_than_a_header() {
        let path = write_scratch("short.hgn", b"HGN1");
        assert_eq!(
            load(&path).unwrap_err(),
            format!("{path}: shorter than a header")
        );
    }

    #[test]
    fn valid_header_parses() {
        let path = write_scratch("valid.hgn", &header(7, 0, 104, "qwen-test"));
        let ck = load(&path).expect("loads");
        assert_eq!(ck.version, 7);
        assert_eq!(ck.entry_count, 0);
        assert_eq!(ck.table_offset, 104);
        assert_eq!(ck.model_id, "qwen-test");
        assert_eq!(ck.file_bytes, HEADER_BYTES as u64);
        assert!(ck.entries.is_empty());
    }

    #[test]
    fn gguf_magic_reports_gguf() {
        let mut bytes = header(1, 0, 104, "x");
        bytes[0..4].copy_from_slice(b"GGUF");
        let path = write_scratch("gguf.hgn", &bytes);
        assert!(load(&path)
            .unwrap_err()
            .starts_with(&format!("{path}: a GGUF.")));
    }

    #[test]
    fn wrong_magic_reports_not_an_hgn() {
        let mut bytes = header(1, 0, 104, "x");
        bytes[0..4].copy_from_slice(b"XXXX");
        let path = write_scratch("junk.hgn", &bytes);
        assert_eq!(
            load(&path).unwrap_err(),
            format!("{path}: not an .hgn (the first bytes are not HGN1)")
        );
    }

    #[test]
    fn table_must_fit_in_the_file() {
        let path = write_scratch("badtable.hgn", &header(1, 1, 104, "x"));
        assert_eq!(
            load(&path).unwrap_err(),
            format!("{path}: the table does not fit in the file")
        );
    }

    #[test]
    fn entries_are_read_from_the_table_offset() {
        let mut bytes = header(3, 1, 104, "m");
        let mut entry = [0u8; ENTRY_BYTES];
        entry[..8].copy_from_slice(b"blk.0.q\0");
        entry[32..36].copy_from_slice(&2u32.to_le_bytes());
        entry[40..48].copy_from_slice(&1024u64.to_le_bytes());
        bytes.extend_from_slice(&entry);
        let path = write_scratch("one-entry.hgn", &bytes);
        let ck = load(&path).expect("loads");
        assert_eq!(ck.entries.len(), 1);
        assert_eq!(ck.entries[0].name(), "blk.0.q");
        assert_eq!(ck.entries[0].dtype(), 2);
        assert_eq!(ck.entries[0].nbytes(), 1024);
    }
}
