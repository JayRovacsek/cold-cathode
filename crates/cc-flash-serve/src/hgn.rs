//! `.hgn` checkpoint reader for the serve path.
//!
//! The header layout is decompile-derived from the shared loader
//! `FUN_01eec4a0` (evidence `docs/analysis/evidence/halogen-tools/decompile/`)
//! and mirrored from `crates/cc-halogen-tools/src/hgn.rs` rather than
//! cross-coupling the crates (the repo pattern is self-contained crates):
//! `HGN1` magic + u32 version + u64 entry count + u64 table offset + two
//! further u64 table fields + a 64-byte NUL-padded `model_id` (104 bytes,
//! `0x68`), followed by 160-byte (`0xa0`) tensor-table entries.
//!
//! Standing caveat (spec `flash_serve.md` review F3): the layout is
//! decompiled but the reference's `.hgn` writer is unverified, so parsing
//! is defensive and the error wordings match the tools spec exactly. The
//! entry→tensor-data mapping implemented here (payloads follow the entry
//! table in entry order, each `nbytes` long) is the best-evidenced reading
//! of the tools' bits-per-weight formula
//! `(file_bytes - table_offset) * 8 / total_nbytes`, which treats every
//! byte from the table offset on as table + payload; it is marked
//! unverified at each use site.
//!
//! Weight residency: [`load`] reads the whole file into one contiguous
//! `Vec<u8>` (std-only — the crate takes no crates.io dependencies, so no
//! `memmap2`). Memory-mapping the payload region is the eventual
//! optimisation, deferred until the dependency policy allows it; the
//! contiguous read is what makes the process RSS grow when serving.
//!
//! The `--resident-gib` query keeps its own inferred 32-byte header reader
//! (`lib.rs::hgn_header`, review F3: falsified against the reference but
//! pinned by the port's own `write_hgn` output and the matrix); this
//! module supersedes it for real checkpoints on the serve path.

use std::fmt;

/// Size of the fixed header in bytes (`0x68`).
pub const HEADER_BYTES: usize = 104;
/// Size of one tensor-table entry in bytes (`0xa0`).
pub const ENTRY_BYTES: usize = 160;

/// One tensor-table entry, kept as raw bytes with the accessors the
/// decompiled `inspect` handler (`FUN_01eed5d0`) reads.
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

/// A parsed `.hgn` checkpoint: the header, the tensor table and the
/// resident file contents.
#[derive(Clone)]
pub struct Checkpoint {
    pub version: u64,
    pub entry_count: u64,
    pub table_offset: u64,
    pub model_id: String,
    pub file_bytes: u64,
    pub entries: Vec<TensorEntry>,
    /// Total declared weight payload (sum of entry `nbytes`) — the
    /// resident-weight size the `--resident-gib` query reports.
    pub weight_bytes: u64,
    /// The file contents held as one contiguous resident buffer (the
    /// mmap-deferred stand-in for the engine's mapped weights).
    pub data: Vec<u8>,
}

impl fmt::Debug for Checkpoint {
    /// Manual impl so `{:?}` never dumps the resident weight bytes.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Checkpoint")
            .field("version", &self.version)
            .field("entry_count", &self.entry_count)
            .field("table_offset", &self.table_offset)
            .field("model_id", &self.model_id)
            .field("file_bytes", &self.file_bytes)
            .field("weight_bytes", &self.weight_bytes)
            .field("entries", &self.entries)
            .finish()
    }
}

/// Read and validate the file at `path`, returning the reference's exact
/// error wording (all [RV] except the format errors marked [D] in the
/// tools spec).
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
    let weight_bytes = entries.iter().map(TensorEntry::nbytes).sum();
    Ok(Checkpoint {
        version,
        entry_count,
        table_offset,
        model_id: cstr(&bytes[40..40 + 64]).to_string(),
        file_bytes,
        entries,
        weight_bytes,
        data: bytes,
    })
}

/// The checkpoint format string the INFO reply reports (token 20,
/// `checkpoint_format` in serve_api's `_parse_info`): `hgn` for an HGN1
/// file, `gguf` for a GGUF, `None` for anything else (or an unreadable
/// path — the caller decides the fallback). Reads only the 4-byte magic.
pub fn format_of(path: &str) -> Option<&'static str> {
    use std::io::Read;
    let mut magic = [0u8; 4];
    let mut file = std::fs::File::open(path).ok()?;
    file.read_exact(&mut magic).ok()?;
    match &magic {
        b"HGN1" => Some("hgn"),
        b"GGUF" => Some("gguf"),
        _ => None,
    }
}

impl Checkpoint {
    /// Byte offset where the tensor payload region begins: immediately
    /// after the entry table. Unverified mapping (module docs).
    pub fn payload_offset(&self) -> u64 {
        self.table_offset + self.entry_count * ENTRY_BYTES as u64
    }

    /// The resident bytes of entry `index`'s tensor payload, per the
    /// unverified entry→data mapping (payloads follow the entry table in
    /// entry order, each `nbytes` long). The slice is clamped to the bytes
    /// actually present (defensive parsing per the writer caveat).
    pub fn tensor_bytes(&self, index: usize) -> Option<&[u8]> {
        let entry = self.entries.get(index)?;
        let preceding: u64 = self.entries[..index].iter().map(TensorEntry::nbytes).sum();
        let start = self
            .payload_offset()
            .checked_add(preceding)
            .map(|offset| offset.min(self.data.len() as u64) as usize)?;
        let len = (entry.nbytes() as usize).min(self.data.len() - start);
        Some(&self.data[start..start + len])
    }

    /// Warn-level anomaly for the serve log: the declared payloads run
    /// past the end of the file (a truncated or mis-stated checkpoint).
    /// Returns `None` when the payload accounting matches the file size.
    pub fn payload_anomaly(&self, path: &str) -> Option<String> {
        let declared_end = self.payload_offset().checked_add(self.weight_bytes)?;
        if declared_end > self.file_bytes {
            Some(format!(
                "{path}: the tensor payloads run past the end of the file"
            ))
        } else {
            None
        }
    }
}

/// Bytes up to the first NUL (the reference's `strnlen(_, 0x40)` semantics).
fn cstr(bytes: &[u8]) -> &str {
    let end = bytes.iter().position(|b| *b == 0).unwrap_or(bytes.len());
    std::str::from_utf8(&bytes[..end]).unwrap_or("<invalid utf8>")
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

    /// One entry with the given name and payload size (the other entry
    /// fields stay zero).
    fn entry(name: &str, nbytes: u64) -> Vec<u8> {
        let mut entry = [0u8; ENTRY_BYTES];
        entry[..name.len()].copy_from_slice(name.as_bytes());
        entry[32..36].copy_from_slice(&2u32.to_le_bytes());
        entry[40..48].copy_from_slice(&nbytes.to_le_bytes());
        entry.to_vec()
    }

    fn write_scratch(name: &str, bytes: &[u8]) -> String {
        let dir = crate::scratch_dir();
        let path = dir.join(name);
        std::fs::write(&path, bytes).expect("write fixture");
        path.display().to_string()
    }

    #[test]
    fn empty_file_is_shorter_than_a_header() {
        let path = write_scratch("ccfs-hgn-short.hgn", b"HGN1");
        assert_eq!(
            load(&path).unwrap_err(),
            format!("{path}: shorter than a header")
        );
    }

    #[test]
    fn valid_header_parses() {
        let path = write_scratch("ccfs-hgn-valid.hgn", &header(7, 0, 104, "qwen-test"));
        let ck = load(&path).expect("loads");
        assert_eq!(ck.version, 7);
        assert_eq!(ck.entry_count, 0);
        assert_eq!(ck.table_offset, 104);
        assert_eq!(ck.model_id, "qwen-test");
        assert_eq!(ck.file_bytes, HEADER_BYTES as u64);
        assert!(ck.entries.is_empty());
        assert_eq!(ck.weight_bytes, 0);
    }

    #[test]
    fn gguf_magic_reports_gguf() {
        let mut bytes = header(1, 0, 104, "x");
        bytes[0..4].copy_from_slice(b"GGUF");
        let path = write_scratch("ccfs-hgn-gguf.hgn", &bytes);
        assert!(load(&path)
            .unwrap_err()
            .starts_with(&format!("{path}: a GGUF.")));
    }

    #[test]
    fn wrong_magic_reports_not_an_hgn() {
        let mut bytes = header(1, 0, 104, "x");
        bytes[0..4].copy_from_slice(b"XXXX");
        let path = write_scratch("ccfs-hgn-junk.hgn", &bytes);
        assert_eq!(
            load(&path).unwrap_err(),
            format!("{path}: not an .hgn (the first bytes are not HGN1)")
        );
    }

    #[test]
    fn table_must_fit_in_the_file() {
        let path = write_scratch("ccfs-hgn-badtable.hgn", &header(1, 1, 104, "x"));
        assert_eq!(
            load(&path).unwrap_err(),
            format!("{path}: the table does not fit in the file")
        );
    }

    #[test]
    fn entries_are_read_from_the_table_offset() {
        let mut bytes = header(3, 1, 104, "m");
        bytes.extend_from_slice(&entry("blk.0.q", 1024));
        let path = write_scratch("ccfs-hgn-one-entry.hgn", &bytes);
        let ck = load(&path).expect("loads");
        assert_eq!(ck.entries.len(), 1);
        assert_eq!(ck.entries[0].name(), "blk.0.q");
        assert_eq!(ck.entries[0].dtype(), 2);
        assert_eq!(ck.entries[0].nbytes(), 1024);
        assert_eq!(ck.weight_bytes, 1024);
    }

    #[test]
    fn tensor_data_follows_the_entry_table_in_order() {
        // Header + two entries (4 and 8 bytes) + their payloads, per the
        // unverified mapping: entry 0's payload starts right after the
        // table, entry 1's after entry 0's.
        let mut bytes = header(1, 2, 104, "m");
        bytes.extend_from_slice(&entry("blk.0.q", 4));
        bytes.extend_from_slice(&entry("blk.0.k", 8));
        bytes.extend_from_slice(&[0xaau8; 4]);
        bytes.extend_from_slice(&[0xbbu8; 8]);
        let path = write_scratch("ccfs-hgn-payloads.hgn", &bytes);
        let ck = load(&path).expect("loads");
        assert_eq!(ck.payload_offset(), 104 + 2 * ENTRY_BYTES as u64);
        assert_eq!(ck.tensor_bytes(0).expect("entry 0"), &[0xaau8; 4]);
        assert_eq!(ck.tensor_bytes(1).expect("entry 1"), &[0xbbu8; 8]);
        assert_eq!(ck.tensor_bytes(2), None);
        assert_eq!(ck.weight_bytes, 12);
        assert_eq!(ck.payload_anomaly(&path), None);
    }

    #[test]
    fn missing_payload_is_a_warn_anomaly() {
        // Entries declare payloads the file does not carry (the port's own
        // header+table-only writer output): the serve log warns rather
        // than the load failing (the tools spec has no error wording for
        // it, and the writer is unverified).
        let mut bytes = header(1, 1, 104, "m");
        bytes.extend_from_slice(&entry("blk.0.q", 1024));
        let path = write_scratch("ccfs-hgn-truncated.hgn", &bytes);
        let ck = load(&path).expect("loads");
        assert_eq!(
            ck.payload_anomaly(&path).as_deref(),
            Some(format!("{path}: the tensor payloads run past the end of the file").as_str())
        );
        // The resident slice is clamped to the bytes present (here: none).
        assert_eq!(ck.tensor_bytes(0).expect("clamped"), &[] as &[u8]);
    }

    #[test]
    fn format_of_reads_the_magic() {
        let hgn = write_scratch("ccfs-hgn-format.hgn", &header(1, 0, 104, "x"));
        assert_eq!(format_of(&hgn), Some("hgn"));
        let mut gguf = header(1, 0, 104, "x");
        gguf[0..4].copy_from_slice(b"GGUF");
        let gguf = write_scratch("ccfs-gguf-format.gguf", &gguf);
        assert_eq!(format_of(&gguf), Some("gguf"));
        let junk = write_scratch("ccfs-junk-format.bin", b"junk");
        assert_eq!(format_of(&junk), None);
        assert_eq!(format_of("/nonexistent-checkpoint"), None);
    }
}
