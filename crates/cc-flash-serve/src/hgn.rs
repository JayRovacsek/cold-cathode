//! `.hgn` checkpoint reader for the serve path.
//!
//! ## Confirmed on-disk layout (round-2, [RV] against the reference file)
//!
//! The layout below was confirmed against the real reference checkpoint
//! `qwen38-flash-next-v2-abliterated.hgn` (repo root, 66,687,678,432 bytes,
//! sha256 of the first 4 KiB `0ccfefcdb01b67708b05865be5f5778828f2f7027377
//! 313616c48bc8b9ce62a0`), cross-read with flash_serve's own loader
//! `FUN_020be0a0` (evidence
//! `docs/analysis/evidence/flash_serve/engine/fun-020be0a0.json`, spec
//! `flash_serve-engine.md` §1). The real header parses as: magic `HGN1`,
//! version `2`, entry_count `2015`, table_offset `104`, `0x4ec00` (322,560)
//! @24, `66,687,678,432` @32 (the exact file size), model_id
//! `qwen3.8-flash-next-v2`.
//!
//! Header (104 bytes, `0x68` — the reference `pread(fd, buf, 0x68, 0)`):
//!
//! | offset | field |
//! |---|---|
//! | 0 | magic `HGN1` (exact compare) |
//! | 4 | `u32` version — gated to {1, 2} (decompile: `(uint)(version − 3) > 0xfffffffd`; the real file is version 2) |
//! | 8 | `u64` entry_count (reference cap: ≤ `0x100000`) |
//! | 16 | `u64` table_offset (start of the entry table; `104` in the real file) |
//! | 24 | `u64` data_offset (start of the tensor payload region; `322,560` in the real file — 512-byte aligned, 56 bytes past the table end) |
//! | 32 | `u64` declared file size (`66,687,678,432` — matches the real file exactly; the reference's `checkpoint: truncated file %s (size mismatch)` compares this against `fstat`) |
//! | 40 | 64-byte NUL-padded `model_id` (`strnlen(_, 0x40)`) |
//!
//! Entry (160 bytes, `0xa0` stride): name `0..96` (NUL-padded, `strnlen(_,
//! 0x60)`), `u32` dtype @96, `u32` kind @100, four `u64` dims @104/@112/
//! @120/@128 (unused trailing dims are 0), `u64` payload data_offset @136,
//! `u64` payload nbytes @144, and an uninterpreted `u64` @152 (candidate:
//! per-tensor checksum — the tools' `verify` checks "checksums"; the value
//! is hash-like in the real file, e.g. `0x25ddbae32` for `embed_tokens.
//! weight`). The @136/@144 pair is chain-verified in the real file: entry
//! 0's `322,560 + 357,580,864` equals entry 1's data_offset, and entry 1's
//! `357,903,424 + 33,280` equals entry 2's data_offset — so payloads follow
//! each other in entry order and the earlier "payloads follow the table"
//! inference (which computed the payload start as `table_offset +
//! entry_count · 160`) is superseded by the explicit per-entry offset.
//!
//! The port's earlier accessors (name `0..32` / dtype @32 / nbytes @40,
//! round-1 review `.tmp/review-report-engine-round1.md`) matched neither
//! source: they were copied from the tools' 96-byte *in-memory* entries
//! (name `0..32`, dtype @32, nbytes @80), not the 160-byte on-disk entries.
//!
//! ## Error wordings
//!
//! The magic/length/table errors match the tools spec exactly ([RV]/[D],
//! `halogen-tools.md` "Checkpoint validation"). The version-gate error uses
//! the flash_serve engine's wording `checkpoint: unsupported version in %s`
//! ([D], `FUN_020bea30` error set, spec `flash_serve-engine.md` §1): the
//! tools spec has no version error (its loader `FUN_01eec4a0` wording set
//! lacks one) and `FUN_020be0a0` itself rejects bad versions silently, so
//! the engine string is the only derivable reference wording.
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
/// Reference cap on `entry_count` (decompile `FUN_020be0a0`:
/// `entry_count < 0x100001`).
pub const MAX_ENTRY_COUNT: u64 = 0x100_000;

/// The fixed 104-byte header, confirmed layout (module docs).
#[derive(Debug, Clone)]
pub struct Header {
    pub version: u64,
    pub entry_count: u64,
    pub table_offset: u64,
    /// Payload-region start (header @24); the first entry's payload begins
    /// here in the real file.
    pub data_offset: u64,
    /// Declared file size (header @32); the reference compares it against
    /// `fstat` (`checkpoint: truncated file %s (size mismatch)`).
    pub declared_bytes: u64,
    pub model_id: String,
}

impl Header {
    /// Validate and parse the header from the leading bytes of a `.hgn`
    /// file. `file_bytes` is the true file size (the whole-file case is
    /// `bytes.len()`; [`load_prefix`] passes the real size when given only
    /// a prefix). Enforces the reference's version gate {1, 2} and the
    /// `entry_count ≤ 0x100000` cap.
    pub fn parse(path: &str, prefix: &[u8], file_bytes: u64) -> Result<Self, String> {
        if prefix.len() < HEADER_BYTES {
            return Err(format!("{path}: shorter than a header"));
        }
        match &prefix[0..4] {
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
        let version = u32::from_le_bytes(prefix[4..8].try_into().expect("4-byte slice")) as u64;
        // Reference gate (FUN_020be0a0): `(uint)(version − 3) > 0xfffffffd`
        // accepts exactly {1, 2}; wording per module docs.
        if version != 1 && version != 2 {
            return Err(format!("checkpoint: unsupported version in {path}"));
        }
        let entry_count = u64::from_le_bytes(prefix[8..16].try_into().expect("8-byte slice"));
        let table_offset = u64::from_le_bytes(prefix[16..24].try_into().expect("8-byte slice"));
        let data_offset = u64::from_le_bytes(prefix[24..32].try_into().expect("8-byte slice"));
        let declared_bytes = u64::from_le_bytes(prefix[32..40].try_into().expect("8-byte slice"));
        if entry_count > MAX_ENTRY_COUNT
            || table_offset > file_bytes
            || entry_count > (file_bytes - table_offset) / ENTRY_BYTES as u64
        {
            return Err(format!("{path}: the table does not fit in the file"));
        }
        Ok(Self {
            version,
            entry_count,
            table_offset,
            data_offset,
            declared_bytes,
            model_id: cstr(&prefix[40..40 + 64]).to_string(),
        })
    }
}

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

    /// Quantisation type code (`u32` @96; the reference's dtype→name table
    /// `PTR_DAT_020baa68` is still a spec gap — real checkpoints carry at
    /// least codes 0 (f32), 5 and 24, plus 0x10 per `FUN_020bdc70`).
    pub fn dtype(&self) -> u32 {
        self.u32_at(96)
    }

    /// Entry kind (`u32` @100; observed: 1 for norm weights, 2 for
    /// matrices; the reference's keep-trunk query matches dtype 0x10 +
    /// kind 2 on `.mlp.experts.` tensors).
    pub fn kind(&self) -> u32 {
        self.u32_at(100)
    }

    /// Tensor shape: four `u64` dims @104/@112/@120/@128; unused trailing
    /// dims are 0 (e.g. `embed_tokens.weight` → `[248320, 2560, 0, 0]`).
    pub fn dims(&self) -> [u64; 4] {
        [
            self.u64_at(104),
            self.u64_at(112),
            self.u64_at(120),
            self.u64_at(128),
        ]
    }

    /// Byte offset of this tensor's payload (confirmed: entries chain,
    /// `data_offset(n+1) == data_offset(n) + nbytes(n)` in the real file).
    pub fn data_offset(&self) -> u64 {
        self.u64_at(136)
    }

    /// Payload size in bytes (`u64` @144). For quantised dtypes this is
    /// smaller than `prod(dims) · sizeof(dtype)` (the quantised blob, not
    /// the dequantised size).
    pub fn nbytes(&self) -> u64 {
        self.u64_at(144)
    }
}

/// A parsed `.hgn` checkpoint: the header, the tensor table and the
/// resident file contents.
#[derive(Clone)]
pub struct Checkpoint {
    pub header: Header,
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
            .field("header", &self.header)
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
    let file_bytes = bytes.len() as u64;
    load_prefix(path, bytes, file_bytes)
}

/// Parse from a leading prefix of the file at `path` (or of the bytes
/// themselves) given the true `file_bytes`. `load` is the whole-file case;
/// the env-gated real-checkpoint test uses this to validate the header and
/// first entries without reading the full 66 GB. Entries are read only
/// while fully present in the prefix.
pub fn load_prefix(path: &str, prefix: Vec<u8>, file_bytes: u64) -> Result<Checkpoint, String> {
    let header = Header::parse(path, &prefix, file_bytes)?;
    let available = prefix.len().saturating_sub(header.table_offset as usize);
    let present = (available / ENTRY_BYTES) as u64;
    let mut entries = Vec::with_capacity(header.entry_count.min(present) as usize);
    for index in 0..header.entry_count.min(present) as usize {
        let offset = header.table_offset as usize + index * ENTRY_BYTES;
        entries.push(TensorEntry {
            raw: prefix[offset..offset + ENTRY_BYTES]
                .try_into()
                .expect("160-byte slice"),
        });
    }
    let weight_bytes = entries.iter().map(TensorEntry::nbytes).sum();
    Ok(Checkpoint {
        header,
        file_bytes,
        entries,
        weight_bytes,
        data: prefix,
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
    /// Byte offset where the tensor payload region begins (header @24).
    pub fn payload_offset(&self) -> u64 {
        self.header.data_offset
    }

    /// The resident bytes of entry `index`'s tensor payload, at the entry's
    /// confirmed `data_offset` (@136). The slice is clamped to the bytes
    /// actually present (defensive parsing per the prefix reader).
    pub fn tensor_bytes(&self, index: usize) -> Option<&[u8]> {
        let entry = self.entries.get(index)?;
        let start = (entry.data_offset() as usize).min(self.data.len());
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

    /// Minimal valid header: HGN1 + version 2 + `entry_count` entries at
    /// `table_offset`, payload region at `data_offset`, declared size
    /// `declared_bytes`, model id (the confirmed on-disk layout).
    fn header(version: u32, entry_count: u64, table_offset: u64, model_id: &str) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"HGN1");
        bytes.extend_from_slice(&version.to_le_bytes());
        bytes.extend_from_slice(&entry_count.to_le_bytes());
        bytes.extend_from_slice(&table_offset.to_le_bytes());
        // @24 data_offset: payloads start right after the table.
        bytes.extend_from_slice(&(table_offset + entry_count * ENTRY_BYTES as u64).to_le_bytes());
        // @32 declared file size: header + table (no payload bytes).
        bytes.extend_from_slice(&(table_offset + entry_count * ENTRY_BYTES as u64).to_le_bytes());
        let mut id = [0u8; 64];
        id[..model_id.len()].copy_from_slice(model_id.as_bytes());
        bytes.extend_from_slice(&id);
        bytes
    }

    /// One entry with the confirmed field layout (dims/dtype/kind/offset/
    /// nbytes all settable; pass zeros where a field is unused).
    fn entry(
        name: &str,
        dims: [u64; 4],
        dtype: u32,
        kind: u32,
        data_offset: u64,
        nbytes: u64,
    ) -> Vec<u8> {
        let mut entry = [0u8; ENTRY_BYTES];
        entry[..name.len()].copy_from_slice(name.as_bytes());
        entry[96..100].copy_from_slice(&dtype.to_le_bytes());
        entry[100..104].copy_from_slice(&kind.to_le_bytes());
        for (i, dim) in dims.iter().enumerate() {
            entry[104 + i * 8..112 + i * 8].copy_from_slice(&dim.to_le_bytes());
        }
        entry[136..144].copy_from_slice(&data_offset.to_le_bytes());
        entry[144..152].copy_from_slice(&nbytes.to_le_bytes());
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
        // Version 2 matches the real reference checkpoint (the old fixture
        // used 7, which the reference gate now rejects — see
        // version_gate_rejects_unsupported_versions).
        let path = write_scratch("ccfs-hgn-valid.hgn", &header(2, 0, 104, "qwen-test"));
        let ck = load(&path).expect("loads");
        assert_eq!(ck.header.version, 2);
        assert_eq!(ck.header.entry_count, 0);
        assert_eq!(ck.header.table_offset, 104);
        assert_eq!(ck.header.data_offset, 104);
        assert_eq!(ck.header.declared_bytes, 104);
        assert_eq!(ck.header.model_id, "qwen-test");
        assert_eq!(ck.file_bytes, HEADER_BYTES as u64);
        assert!(ck.entries.is_empty());
        assert_eq!(ck.weight_bytes, 0);
    }

    #[test]
    fn version_gate_accepts_one_and_two() {
        // Reference gate (FUN_020be0a0): (uint)(version − 3) > 0xfffffffd
        // → versions 1 and 2 only; the port's write_hgn writes 1, the real
        // reference checkpoint is 2.
        for version in [1u32, 2] {
            let path = write_scratch(
                &format!("ccfs-hgn-v{version}.hgn"),
                &header(version, 0, 104, "x"),
            );
            assert_eq!(load(&path).expect("loads").header.version, version as u64);
        }
    }

    #[test]
    fn version_gate_rejects_unsupported_versions() {
        // Wording: the engine's `checkpoint: unsupported version in %s`
        // (FUN_020bea30 error set — the only derivable reference wording,
        // module docs).
        for version in [0u32, 3, 7] {
            let path = write_scratch(
                &format!("ccfs-hgn-badv{version}.hgn"),
                &header(version, 0, 104, "x"),
            );
            assert_eq!(
                load(&path).unwrap_err(),
                format!("checkpoint: unsupported version in {path}")
            );
        }
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
    fn entry_count_cap_matches_the_reference() {
        // Reference cap entry_count ≤ 0x100000 (FUN_020be0a0). A header
        // declaring one entry past the cap cannot be backed by a real file
        // here, so the table-fit error covers it.
        let path = write_scratch(
            "ccfs-hgn-countcap.hgn",
            &header(1, MAX_ENTRY_COUNT + 1, 104, "x"),
        );
        assert_eq!(
            load(&path).unwrap_err(),
            format!("{path}: the table does not fit in the file")
        );
    }

    #[test]
    fn entries_are_read_from_the_table_offset() {
        // Fixture mirrors the real file's entry 0 field layout (name 96 B,
        // dtype @96, kind @100, dims @104..@136, offset @136, nbytes @144).
        let mut bytes = header(2, 1, 104, "m");
        bytes.extend_from_slice(&entry(
            "blk.0.q",
            [4, 10240, 0, 0],
            2,
            2,
            104 + ENTRY_BYTES as u64,
            1024,
        ));
        let path = write_scratch("ccfs-hgn-one-entry.hgn", &bytes);
        let ck = load(&path).expect("loads");
        assert_eq!(ck.entries.len(), 1);
        assert_eq!(ck.entries[0].name(), "blk.0.q");
        assert_eq!(ck.entries[0].dtype(), 2);
        assert_eq!(ck.entries[0].kind(), 2);
        assert_eq!(ck.entries[0].dims(), [4, 10240, 0, 0]);
        assert_eq!(ck.entries[0].data_offset(), 104 + ENTRY_BYTES as u64);
        assert_eq!(ck.entries[0].nbytes(), 1024);
        assert_eq!(ck.weight_bytes, 1024);
    }

    #[test]
    fn tensor_data_follows_the_entry_data_offsets() {
        // Header + two entries (4 and 8 bytes) + their payloads: entry 0's
        // payload starts at its @136 offset, entry 1's at its own (the
        // confirmed chaining layout, not a computed table-end sum).
        let mut bytes = header(2, 2, 104, "m");
        bytes.extend_from_slice(&entry(
            "blk.0.q",
            [1, 0, 0, 0],
            0,
            1,
            104 + 2 * ENTRY_BYTES as u64,
            4,
        ));
        bytes.extend_from_slice(&entry(
            "blk.0.k",
            [1, 0, 0, 0],
            0,
            1,
            104 + 2 * ENTRY_BYTES as u64 + 4,
            8,
        ));
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
        let mut bytes = header(2, 1, 104, "m");
        bytes.extend_from_slice(&entry(
            "blk.0.q",
            [1, 0, 0, 0],
            0,
            1,
            104 + ENTRY_BYTES as u64,
            1024,
        ));
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

    /// Parse the real reference checkpoint header without reading the full
    /// 66 GB: `COLD_CATHODE_REAL_HGN` points at the file (repo root); the
    /// test skips cleanly when unset so `nix build` and CI don't need it.
    #[test]
    fn real_reference_checkpoint_header() {
        let Some(path) = std::env::var_os("COLD_CATHODE_REAL_HGN") else {
            eprintln!("SKIP real_reference_checkpoint_header: COLD_CATHODE_REAL_HGN unset");
            return;
        };
        let path = path.to_string_lossy().into_owned();
        let file_bytes = std::fs::metadata(&path).expect("real file").len();
        assert_eq!(file_bytes, 66_687_678_432, "unexpected real file size");
        // Header + first three entries (3 * 160 B past the 104-byte header).
        let mut prefix = vec![0u8; 4096];
        let read = {
            use std::io::Read;
            let mut file = std::fs::File::open(&path).expect("open real file");
            file.read(&mut prefix).expect("read prefix")
        };
        prefix.truncate(read);
        let ck = load_prefix(&path, prefix.clone(), file_bytes).expect("real header parses");
        assert_eq!(ck.header.version, 2);
        assert_eq!(ck.header.entry_count, 2015);
        assert_eq!(ck.header.table_offset, 104);
        assert_eq!(ck.header.data_offset, 322_560);
        assert_eq!(ck.header.declared_bytes, file_bytes);
        assert_eq!(ck.header.model_id, "qwen3.8-flash-next-v2");
        // Sample entries (values read from the real bytes, round 2):
        let e0 = TensorEntry::from_bytes(&prefix[104..]).expect("entry 0");
        assert_eq!(e0.name(), "embed_tokens.weight");
        assert_eq!(e0.dtype(), 5);
        assert_eq!(e0.kind(), 2);
        assert_eq!(e0.dims(), [248_320, 2560, 0, 0]);
        assert_eq!(e0.data_offset(), 322_560);
        assert_eq!(e0.nbytes(), 357_580_864);
        let e1 = TensorEntry::from_bytes(&prefix[264..]).expect("entry 1");
        assert_eq!(
            e1.name(),
            "layers.0.attn_hyper_connection.block_inject_weight.weight"
        );
        assert_eq!(e1.dtype(), 24);
        assert_eq!(e1.kind(), 2);
        assert_eq!(e1.dims(), [4, 10_240, 0, 0]);
        assert_eq!(e1.data_offset(), 357_903_424);
        assert_eq!(e1.nbytes(), 33_280);
        let e2 = TensorEntry::from_bytes(&prefix[424..]).expect("entry 2");
        assert_eq!(e2.name(), "layers.0.attn_hyper_connection.hc_norm.weight");
        assert_eq!(e2.dtype(), 0);
        assert_eq!(e2.kind(), 1);
        assert_eq!(e2.dims(), [10_240, 0, 0, 0]);
        assert_eq!(e2.data_offset(), 357_936_704);
        assert_eq!(e2.nbytes(), 20_480);
        // The confirmed entry→payload chaining (module docs): each entry's
        // payload ends exactly where the next begins.
        assert_eq!(e0.data_offset() + e0.nbytes(), e1.data_offset());
        assert_eq!(e1.data_offset() + e1.nbytes(), e2.data_offset());
    }
}
