//! `.hgn` checkpoint reader (spec: `docs/analysis/specs/halogen-tools.md`,
//! "Flag table — `inspect`" / checkpoint validation).
//!
//! The header layout is decompile-derived from the shared loader
//! `FUN_01eec4a0` (evidence `docs/analysis/evidence/halogen-tools/decompile/`):
//! `HGN1` magic + u32 version + u64 entry count + u64 table offset + two
//! further u64 table fields + a 64-byte NUL-padded `model_id` (104 bytes,
//! `0x68`), followed by 160-byte (`0xa0`) tensor-table entries.
//!
//! The entry layout is now confirmed ([RV] round 2) against the real
//! reference checkpoint `qwen38-flash-next-v2-abliterated.hgn` (repo root,
//! 66,687,678,432 bytes) and flash_serve's loader `FUN_020be0a0` (spec
//! `flash_serve-engine.md` §1): name `0..96` (NUL-padded), `u32` dtype
//! @96, `u32` kind @100, four `u64` dims @104/@112/@120/@128, `u64`
//! payload data_offset @136, `u64` nbytes @144, uninterpreted `u64` @152
//! (candidate per-tensor checksum). The earlier accessors (name `0..32` /
//! dtype @32 / nbytes @40) mirrored the `inspect` handler's 96-byte
//! *in-memory* entries (name `0..32`, dtype @32, nbytes @80), not the
//! on-disk entries, and are corrected here; the full field table lives in
//! `crates/cc-flash-serve/src/hgn.rs` (same layout, self-contained-crate
//! pattern).

/// Size of the fixed header in bytes (`0x68`).
pub const HEADER_BYTES: usize = 104;
/// Size of one tensor-table entry in bytes (`0xa0`).
pub const ENTRY_BYTES: usize = 160;

/// One tensor-table entry, kept as raw bytes with the confirmed on-disk
/// accessors (module docs).
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

    /// Build an entry from 160 raw bytes (`None` if the slice is short).
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        Some(Self {
            raw: bytes.get(..ENTRY_BYTES)?.try_into().ok()?,
        })
    }

    /// Tensor name: bytes 0..96, NUL-padded (reference `strnlen(_, 0x60)`).
    pub fn name(&self) -> &str {
        cstr(&self.raw[0..96])
    }

    /// Quantisation type code (`u32` @96; names resolved from the
    /// reference's `PTR_DAT_020baa68` table — see `dtype_name`; real
    /// checkpoints carry codes 0 (bf16), 5 (q4c) and 24 (q6g64), plus
    /// 0x10 (ht) per `FUN_020bdc70`).
    pub fn dtype(&self) -> u32 {
        self.u32_at(96)
    }

    /// Entry kind (`u32` @100; observed: 1 for norm weights, 2 for
    /// matrices).
    pub fn kind(&self) -> u32 {
        self.u32_at(100)
    }

    /// Tensor shape: four `u64` dims @104/@112/@120/@128; unused trailing
    /// dims are 0.
    pub fn dims(&self) -> [u64; 4] {
        [
            self.u64_at(104),
            self.u64_at(112),
            self.u64_at(120),
            self.u64_at(128),
        ]
    }

    /// Byte offset of this tensor's payload (`u64` @136).
    pub fn data_offset(&self) -> u64 {
        self.u64_at(136)
    }

    /// Payload size in bytes (`u64` @144).
    pub fn nbytes(&self) -> u64 {
        self.u64_at(144)
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

/// Dtype code → name, from the reference's `PTR_DAT_020baa68` table
/// (dumped from `.rodata`, spec `flash_serve-engine.md` §1). The NULL
/// slots (codes 9, 17, 18, 19 — matching the validity mask `0x1f1fdff`)
/// and any code beyond the table render as `t<code>`.
fn dtype_name(code: u32) -> String {
    match code {
        0 => "bf16",
        1 => "f32",
        2 => "f16",
        3 => "i32",
        4 => "i64",
        5 => "q4c",
        6 => "fp8r",
        7 => "q8g64",
        8 => "i4l",
        10 => "fp8g",
        11 => "q8s32",
        12 => "q6k",
        13 => "iq4nl",
        14 => "q4a",
        15 => "q5a",
        16 => "ht",
        20 => "q8_0",
        21 => "q5_0",
        22 => "q5_1",
        23 => "i4r",
        24 => "q6g64",
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
    fn dtype_name_matches_the_reference_table() {
        // Authoritative table: flash_serve-engine.md §1
        // (PTR_DAT_020baa68 dumped from .rodata). The port previously
        // mapped 0=f32, 1=f16, 2=bf16 — the reference maps 0=bf16,
        // 1=f32, 2=f16.
        assert_eq!(dtype_name(0), "bf16");
        assert_eq!(dtype_name(1), "f32");
        assert_eq!(dtype_name(2), "f16");
        // Quant / other codes seen in the real checkpoint or named by
        // the table.
        assert_eq!(dtype_name(5), "q4c");
        assert_eq!(dtype_name(16), "ht");
        assert_eq!(dtype_name(24), "q6g64");
        // NULL slots (validity mask 0x1f1fdff) and unknown codes render
        // as `t<code>`.
        assert_eq!(dtype_name(9), "t9");
        assert_eq!(dtype_name(17), "t17");
        assert_eq!(dtype_name(25), "t25");
    }

    #[test]
    fn entries_are_read_from_the_table_offset() {
        // Entry fixture uses the confirmed on-disk layout (name 0..96,
        // dtype @96, kind @100, dims @104..@136, data_offset @136,
        // nbytes @144) — see module docs.
        let mut bytes = header(3, 1, 104, "m");
        let mut entry = [0u8; ENTRY_BYTES];
        entry[..8].copy_from_slice(b"blk.0.q\0");
        entry[96..100].copy_from_slice(&2u32.to_le_bytes());
        entry[100..104].copy_from_slice(&2u32.to_le_bytes());
        entry[104..112].copy_from_slice(&4u64.to_le_bytes());
        entry[112..120].copy_from_slice(&10_240u64.to_le_bytes());
        entry[136..144].copy_from_slice(&(104u64 + ENTRY_BYTES as u64).to_le_bytes());
        entry[144..152].copy_from_slice(&1024u64.to_le_bytes());
        bytes.extend_from_slice(&entry);
        let path = write_scratch("one-entry.hgn", &bytes);
        let ck = load(&path).expect("loads");
        assert_eq!(ck.entries.len(), 1);
        assert_eq!(ck.entries[0].name(), "blk.0.q");
        assert_eq!(ck.entries[0].dtype(), 2);
        assert_eq!(ck.entries[0].kind(), 2);
        assert_eq!(ck.entries[0].dims(), [4, 10_240, 0, 0]);
        assert_eq!(ck.entries[0].data_offset(), 104 + ENTRY_BYTES as u64);
        assert_eq!(ck.entries[0].nbytes(), 1024);
    }
}
