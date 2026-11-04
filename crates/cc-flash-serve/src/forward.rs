//! Forward-pass entry point (spec `flash_serve-engine.md` §3).
//!
//! The reference's forward pass is `norm → qkv → rope → attention →
//! projection → mlp` per layer over the hybrid `qwen4exp` architecture
//! (spec §2/§3), executed on the GPU through nine embedded AMDGPU code
//! objects. The head count / head dim / GQA configuration and the exact
//! dtype table are still open spec gaps (spec §2 "Gaps register", §3
//! `[D]` entries), so this module ships the entry point the `GEN` path
//! calls today — a documented deterministic stub — with the shape the
//! real attention/MLP math drops into once the head config lands.
//!
//! Stub contract (the parts the rest of the port depends on):
//!
//! * The vocabulary size comes from the loaded checkpoint's
//!   `embed_tokens.weight` entry dims (`[vocab, hidden]` in the real
//!   reference file); checkpoints without that entry fall back to the
//!   frozen-spec vocab, 248 320 (spec §2 [RV]).
//! * The logits are a deterministic pseudo-random distribution seeded by
//!   an FNV-1a hash of the context ids (the reference's hash family,
//!   spec §5). The stub deliberately does not interpret the quantised
//!   weight payloads — the dtype table is a spec gap (§1) — so the real
//!   numerics arrive by replacing [`Forward::logits`]'s body, not the
//!   signature.
//!
//! The signature takes the full context (prompt ids followed by the
//! generated ids so far) and returns the next-token logits, which is the
//! logical contract of the reference's prefill+decode loop; the
//! KV-cache-backed incremental form (spec §5) replaces the body later
//! without touching callers.

use crate::hgn;
use crate::sampling::Rng;

/// Fallback vocabulary size when the checkpoint carries no
/// `embed_tokens.weight` entry (spec §2 [RV]: frozen spec INFO table).
pub const DEFAULT_VOCAB_SIZE: usize = 248_320;

/// The forward-pass entry point over a loaded checkpoint.
#[derive(Debug)]
pub struct Forward<'a> {
    checkpoint: &'a hgn::Checkpoint,
    vocab_size: usize,
}

impl<'a> Forward<'a> {
    /// Bind the forward pass to a loaded checkpoint (the `hgn::load` mmap
    /// path; the GGUF serve path has no resident checkpoint and never
    /// reaches this constructor — see `protocol::handle_line`).
    pub fn new(checkpoint: &'a hgn::Checkpoint) -> Self {
        Forward {
            checkpoint,
            vocab_size: vocab_size(checkpoint),
        }
    }

    /// The vocabulary size the logits vectors are sized to.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// The checkpoint the weights are read from (the future attention/MLP
    /// math's weight source).
    pub fn checkpoint(&self) -> &hgn::Checkpoint {
        self.checkpoint
    }

    /// Next-token logits for `context_ids` (prompt ids followed by the
    /// generated ids so far). Stub implementation: module docs.
    pub fn logits(&self, context_ids: &[i64]) -> Vec<f32> {
        let mut rng = Rng::from_u64(context_seed(context_ids));
        (0..self.vocab_size).map(|_| rng.next_f32()).collect()
    }
}

/// The vocabulary size: `embed_tokens.weight` dims[0] when the checkpoint
/// carries that entry (the real reference file: `[248320, 2560]`),
/// otherwise [`DEFAULT_VOCAB_SIZE`].
pub fn vocab_size(checkpoint: &hgn::Checkpoint) -> usize {
    checkpoint
        .entries
        .iter()
        .find(|entry| entry.name() == "embed_tokens.weight")
        .and_then(|entry| usize::try_from(entry.dims()[0]).ok())
        .filter(|&vocab| vocab > 0)
        .unwrap_or(DEFAULT_VOCAB_SIZE)
}

/// FNV-1a over the context ids (the reference's hash family — spec §5
/// quotes the same prime), seeding the stub's per-context distribution.
fn context_seed(context_ids: &[i64]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for id in context_ids {
        for byte in id.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal `.hgn` fixture: header + one `embed_tokens.weight` entry
    /// with `dims [vocab, 4]` and an empty payload (layout per
    /// `hgn::TensorEntry`).
    fn fixture(vocab: u64) -> hgn::Checkpoint {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"HGN1");
        bytes.extend_from_slice(&2u32.to_le_bytes()); // version
        bytes.extend_from_slice(&1u64.to_le_bytes()); // entry_count
        bytes.extend_from_slice(&104u64.to_le_bytes()); // table_offset
        bytes.extend_from_slice(&264u64.to_le_bytes()); // data_offset
        bytes.extend_from_slice(&264u64.to_le_bytes()); // declared size
        let mut model_id = [0u8; 64];
        model_id[..4].copy_from_slice(b"qwen");
        bytes.extend_from_slice(&model_id);
        let mut entry = [0u8; hgn::ENTRY_BYTES];
        entry[..19].copy_from_slice(b"embed_tokens.weight");
        entry[96..100].copy_from_slice(&0u32.to_le_bytes()); // dtype
        entry[100..104].copy_from_slice(&2u32.to_le_bytes()); // kind
        entry[104..112].copy_from_slice(&vocab.to_le_bytes());
        entry[112..120].copy_from_slice(&4u64.to_le_bytes());
        entry[136..144].copy_from_slice(&264u64.to_le_bytes());
        bytes.extend_from_slice(&entry);
        hgn::load_prefix("ccfs-forward-fixture.hgn", bytes, 264).expect("fixture loads")
    }

    #[test]
    fn vocab_size_comes_from_the_embedding_entry() {
        let ck = fixture(8);
        assert_eq!(vocab_size(&ck), 8);
        assert_eq!(Forward::new(&ck).vocab_size(), 8);
    }

    #[test]
    fn vocab_size_falls_back_to_the_frozen_spec_value() {
        // A header with no embed entry (the port's own write_hgn output):
        // the frozen-spec vocab.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"HGN1");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes()); // entry_count 0
        bytes.extend_from_slice(&104u64.to_le_bytes());
        bytes.extend_from_slice(&104u64.to_le_bytes());
        bytes.extend_from_slice(&104u64.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 64]);
        let ck = hgn::load_prefix("ccfs-forward-empty.hgn", bytes, 104).expect("loads");
        assert_eq!(vocab_size(&ck), DEFAULT_VOCAB_SIZE);
    }

    #[test]
    fn logits_are_deterministic_per_context() {
        let ck = fixture(8);
        let forward = Forward::new(&ck);
        let first = forward.logits(&[10, 20]);
        assert_eq!(first.len(), 8);
        assert_eq!(
            first,
            forward.logits(&[10, 20]),
            "same context, same logits"
        );
        assert_ne!(
            first,
            forward.logits(&[10, 21]),
            "context changes the logits"
        );
        assert!(
            first.iter().all(|logit| logit.is_finite()),
            "stub logits must be finite"
        );
        assert!(
            first.iter().all(|logit| (0.0..1.0).contains(logit)),
            "stub logits must be in [0, 1)"
        );
    }
}
