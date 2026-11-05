//! Forward-pass entry point (spec `flash_serve-engine.md` §3).
//!
//! The reference's forward pass is `norm → qkv → rope → attention →
//! projection → mlp` per layer over the hybrid `qwen4exp` architecture
//! (spec §2/§3), executed on the GPU through nine embedded AMDGPU code
//! objects. This module implements the two paths the `GEN` contract uses:
//!
//! * **Real path** ([`crate::model`] + [`crate::ops`]): embedding lookup,
//!   the 48-layer stack (full-attention layers 3, 7, …, 47 with
//!   `q_norm`/`k_norm` + half-rotary RoPE + GQA attention + `o_proj`;
//!   the other 44 layers run the gated linear-attention block), the MoE
//!   MLP after every block, the top-level hyper-connection mixer and the
//!   `lm_head` gemv. Single-token decode first: prefill is a
//!   single-token loop over the context (the reference's paged slot pool,
//!   spec §5, is ported in [`crate::kvcache`]), and the per-sequence
//!   KV/delta/conv state is memoised in the pool-backed slot
//!   ([`Forward::rollback`] / [`Forward::commit`]) so extending the
//!   context by one token costs one token step, not a full recompute.
//! * **Stub fallback** (the historical behaviour): checkpoints whose
//!   weights cannot be run for real — the port's own write-path fixtures
//!   carry no layer tensors at all, and dtypes without a dequant codec
//!   (the integer codes `3`/`4` on weight entries) still yield no
//!   model — fall back to the deterministic FNV-1a-seeded pseudo-random
//!   distribution (the reference's hash family, spec §5). The real
//!   `qwen38-flash-next-v2-abliterated.hgn` now takes the real path:
//!   its quantised projections (`q4c`/`q8g64`/`ht`/`i4r`/`q6g64`)
//!   decode through [`crate::codecs`] (spec `flash_serve-dtypes.md`),
//!   so the stub only serves fixtures without a full tensor set.
//!
//! The signature takes the full context (prompt ids followed by the
//! generated ids so far) and returns the next-token logits, which is the
//! logical contract of the reference's prefill+decode loop; the
//! KV-cache-backed incremental form (spec §5) replaces the cache
//! implementation later without touching callers.

use std::cell::RefCell;

use crate::hgn;
use crate::kvcache::{self, LayerRuntime, SlotPool, SlotRecord};
use crate::model::{self, Block, Model, NORM_EPS, ROPE_THETA};
use crate::ops;
use crate::sampling::Rng;

/// Fallback vocabulary size when the checkpoint carries no
/// `embed_tokens.weight` entry (spec §2 [RV]: frozen spec INFO table).
pub const DEFAULT_VOCAB_SIZE: usize = 248_320;

/// The forward-pass entry point over a loaded checkpoint.
#[derive(Debug)]
pub struct Forward<'a> {
    checkpoint: &'a hgn::Checkpoint,
    vocab_size: usize,
    /// The real model when the checkpoint exposes the full bf16/f32
    /// tensor set (`model::load`); `None` selects the stub fallback
    /// (module docs).
    model: Option<Model>,
    /// The decode cache as a single-slot pool (the reference's slot pool,
    /// spec §2, ported in [`crate::kvcache`]): created on the first real
    /// logits call, holding the current sequence's slot record.
    pool: RefCell<Option<SlotPool>>,
}

impl<'a> Forward<'a> {
    /// Bind the forward pass to a loaded checkpoint (the `hgn::load` mmap
    /// path; the GGUF serve path has no resident checkpoint and never
    /// reaches this constructor — see `protocol::handle_line`).
    pub fn new(checkpoint: &'a hgn::Checkpoint) -> Self {
        Forward {
            checkpoint,
            vocab_size: vocab_size(checkpoint),
            model: model::load(checkpoint),
            pool: RefCell::new(None),
        }
    }

    /// The vocabulary size the logits vectors are sized to.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// The checkpoint the weights are read from.
    pub fn checkpoint(&self) -> &hgn::Checkpoint {
        self.checkpoint
    }

    /// The real model when the checkpoint is loadable (module docs);
    /// `None` means [`Self::logits`] uses the stub fallback.
    pub fn model(&self) -> Option<&Model> {
        self.model.as_ref()
    }

    /// Next-token logits for `context_ids` (prompt ids followed by the
    /// generated ids so far).
    pub fn logits(&self, context_ids: &[i64]) -> Vec<f32> {
        match &self.model {
            Some(model) if !context_ids.is_empty() => self.real_logits(model, context_ids),
            // Empty contexts have no token to run; the stub's hash of the
            // (empty) context keeps the contract total.
            _ => stub_logits(self.vocab_size, context_ids),
        }
    }

    /// Commit `n` positions from the front of the cached sequence's commit
    /// window (the port's counterpart of the reference's commit-window
    /// mechanics, spec §2: drafted rows whose K/V were written
    /// speculatively become committed). Errors when nothing is cached or
    /// `n` exceeds the uncommitted positions (mirroring [`Self::rollback`]).
    pub fn commit(&self, n: usize) -> Result<(), String> {
        let mut pool = self.pool.borrow_mut();
        match pool.as_mut() {
            Some(pool) => pool.slots_mut()[0].commit(n),
            None => Err(format!("commit({n}): no cached sequence to commit")),
        }
    }

    /// Roll the cached context back by `n` positions, restoring the exact
    /// layer state of the shorter context (spec §2: speculative rows are
    /// rolled back on rejection). Errors when there is no cached sequence
    /// or `n` exceeds the uncommitted positions.
    pub fn rollback(&self, n: usize) -> Result<(), String> {
        let mut pool = self.pool.borrow_mut();
        match pool.as_mut() {
            Some(pool) => pool.slots_mut()[0].rollback(n),
            None => Err(format!("rollback({n}): no cached sequence to roll back")),
        }
    }

    /// Real-path logits: reuse the cached prefix when the context only
    /// grew, otherwise re-run the single-token prefill loop (module docs).
    fn real_logits(&self, model: &Model, context: &[i64]) -> Vec<f32> {
        let mut pool = self.pool.borrow_mut();
        if pool.is_none() {
            *pool = Some(SlotPool::single(
                model,
                self.checkpoint,
                kvcache::kv8_mode(),
            ));
        }
        let slot = &mut pool.as_mut().expect("pool just set").slots_mut()[0];
        if slot.context == context {
            return slot.logits.clone();
        }
        if !context.starts_with(&slot.context) {
            // A shorter or diverging context resets the slot and reruns
            // (the historical cache behaviour).
            *slot = SlotRecord::with_model(model);
        }
        // The loop always runs at least once here (the equal-context case
        // returned above), so the loop's logits are the result.
        let mut logits = Vec::new();
        for &token in context.iter().skip(slot.context.len()) {
            logits = slot
                .append_step(model, token)
                .expect("extending a cached context stays within the slot's max-seq bound");
        }
        logits
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

/// Stub logits: an FNV-1a-seeded pseudo-random distribution over the
/// vocabulary (module docs).
fn stub_logits(vocab_size: usize, context_ids: &[i64]) -> Vec<f32> {
    let mut rng = Rng::from_u64(context_seed(context_ids));
    (0..vocab_size).map(|_| rng.next_f32()).collect()
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

/// One token step: embedding lookup → 48 layers → final mixer → `lm_head`
/// (spec §3; the per-layer stage order is `norm → qkv → rope → attention →
/// projection → mlp` with the hyper-connection mixers around each block).
pub(crate) fn forward_token(
    model: &Model,
    token: i64,
    position: usize,
    layers: &mut [LayerRuntime],
) -> Vec<f32> {
    // Out-of-range ids wrap into the table (the reference validates ids at
    // the protocol layer; the port clamps defensively here).
    let row = token.rem_euclid(model.dims.vocab as i64) as usize;
    let embedding = model.embed.row(row);
    // The top-level hyper-connection expands the embedding into the
    // stream stack (the real file has no top-level `block_inject_weight`,
    // so every stream starts at the raw embedding row — documented
    // assumption, spec §2).
    let mut streams = embedding.repeat(model.dims.num_streams);
    for (index, layer) in model.layers.iter().enumerate() {
        let (block_input, scaled, inject) = layer.attn_mixer.mix(&streams);
        let block_out = match &layer.block {
            Block::Attention(attn) => {
                attention_block(attn, &block_input, position, &mut layers[index])
            }
            Block::Linear(linear) => linear_block(linear, &block_input, &mut layers[index]),
        };
        layer
            .attn_mixer
            .apply_block(&scaled, inject.as_deref(), &block_out, &mut streams);
        let (mlp_input, mlp_scaled, mlp_inject) = layer.mlp_mixer.mix(&streams);
        let mlp_out = layer.mlp.forward(&mlp_input);
        layer
            .mlp_mixer
            .apply_block(&mlp_scaled, mlp_inject.as_deref(), &mlp_out, &mut streams);
    }
    // Final aggregation: the top-level mixer folds the stream stack into
    // the single hidden vector the `lm_head` reads.
    let (final_hidden, _, _) = model.final_mixer.mix(&streams);
    model.lm_head.gemv(&final_hidden)
}

/// The full-attention block (`self_attn`, spec §3 `FUN_01fb3670`): the
/// `q_proj` gemv computes all rows, the rope/attention path consumes the
/// first half (the 24-vs-48 nuance, spec §2); per-head `q_norm`/`k_norm`,
/// half-rotary RoPE, GQA attention over the cached keys/values, `o_proj`.
fn attention_block(
    attn: &model::AttnBlock,
    x: &[f32],
    position: usize,
    runtime: &mut LayerRuntime,
) -> Vec<f32> {
    let all_q = attn.q.gemv(x);
    let mut q = all_q[..attn.q.rows / 2].to_vec();
    for head in q.chunks_mut(attn.head_dim) {
        let normed = ops::rmsnorm(head, &attn.q_norm, NORM_EPS);
        head.copy_from_slice(&normed);
    }
    let mut k = attn.k.gemv(x);
    for head in k.chunks_mut(attn.head_dim) {
        let normed = ops::rmsnorm(head, &attn.k_norm, NORM_EPS);
        head.copy_from_slice(&normed);
    }
    let v = attn.v.gemv(x);
    ops::rope_heads(&mut q, attn.head_dim, position, ROPE_THETA);
    ops::rope_heads(&mut k, attn.head_dim, position, ROPE_THETA);
    runtime.k.extend_from_slice(&k);
    runtime.v.extend_from_slice(&v);
    let out = ops::attention(
        &q,
        &runtime.k,
        &runtime.v,
        attn.q_heads,
        attn.kv_heads,
        attn.head_dim,
        attn.head_dim,
    );
    attn.o.gemv(&out)
}

/// The gated linear-attention block (`linear_attn`, spec §3
/// `FUN_01fa0700`): packed qkv gemv → depthwise conv1d (3-token history)
/// → per-v-head decay/input gates → delta-rule recurrence → per-head norm
/// with the `in_proj_z` output gate → `out_proj`. The v-head j ↔ q/k-head
/// mapping follows the registered sub-gap note (v head j uses q/k head
/// `j · q_heads / v_heads` — `j/3` in the real 16-vs-48 layout).
fn linear_block(linear: &model::LinearBlock, x: &[f32], runtime: &mut LayerRuntime) -> Vec<f32> {
    let qkv_raw = linear.qkv.gemv(x);
    let convolved = ops::conv1d_step(
        &linear.conv_w,
        linear.qkv.rows,
        linear.conv_kernel,
        &mut runtime.conv_hist,
        &qkv_raw,
    );
    let z = linear.z.gemv(x);
    let a = linear.a.gemv(x);
    let b = linear.b.gemv(x);
    let hd = linear.head_dim;
    let vhd = linear.v_head_dim;
    let q_width = linear.q_heads * hd;
    let k_width = linear.k_heads * hd;
    let mut head_outs = vec![0.0; linear.v_heads * vhd];
    for j in 0..linear.v_heads {
        let q_head = (j * linear.q_heads) / linear.v_heads;
        let k_head = (j * linear.k_heads) / linear.v_heads;
        let q = &convolved[q_head * hd..(q_head + 1) * hd];
        let k = &convolved[q_width + k_head * hd..q_width + (k_head + 1) * hd];
        let v = &convolved[q_width + k_width + j * vhd..q_width + k_width + (j + 1) * vhd];
        // α = exp(−exp(A_log) · softplus(a + dt_bias)), β = sigmoid(b)
        // (spec §3 step 5, `k_dn_gates`).
        let alpha = (-(-linear.a_log[j]).exp() * ops::softplus(a[j] + linear.dt_bias[j])).exp();
        let beta = ops::sigmoid(b[j]);
        let state = &mut runtime.delta[j * vhd * hd..(j + 1) * vhd * hd];
        let mut o = ops::delta_rule_step(state, k, v, q, alpha, beta);
        // Per-head norm `[head_dim]` with the gated `in_proj_z` output
        // (RMSNormGated: `rmsnorm(o) · silu(z)`, the Qwen3-Next GDN form).
        let normed = ops::rmsnorm(&o, &linear.norm, NORM_EPS);
        for (i, value) in o.iter_mut().enumerate() {
            *value = normed[i] * ops::silu(z[j * vhd + i]);
        }
        head_outs[j * vhd..(j + 1) * vhd].copy_from_slice(&o);
    }
    linear.out.gemv(&head_outs)
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
    fn stub_logits_are_deterministic_per_context() {
        let ck = fixture(8);
        let forward = Forward::new(&ck);
        assert!(
            forward.model().is_none(),
            "embed-only fixtures take the stub path"
        );
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

    #[test]
    fn real_forward_on_the_mini_checkpoint_returns_vocab_sized_logits() {
        let ck = model::mini_checkpoint();
        let forward = Forward::new(&ck);
        assert!(
            forward.model().is_some(),
            "the mini fixture loads the real path"
        );
        assert_eq!(forward.vocab_size(), model::MINI_VOCAB);
        let logits = forward.logits(&[1, 2, 3]);
        assert_eq!(logits.len(), model::MINI_VOCAB);
        assert!(
            logits.iter().all(|logit| logit.is_finite()),
            "logits {logits:?}"
        );
        assert_ne!(
            logits,
            stub_logits(model::MINI_VOCAB, &[1, 2, 3]),
            "the real path must not return the stub distribution"
        );
    }

    #[test]
    fn real_forward_is_deterministic_and_incremental() {
        let ck = model::mini_checkpoint();
        let forward = Forward::new(&ck);
        let first = forward.logits(&[1, 2]);
        assert_eq!(
            first,
            forward.logits(&[1, 2]),
            "cached context, same logits"
        );
        let extended = forward.logits(&[1, 2, 3]);
        assert!(extended.iter().all(|logit| logit.is_finite()));
        assert_ne!(first, extended, "extending the context changes the logits");
        // A fresh forward recomputes the same logits (no hidden state).
        let fresh = Forward::new(&ck);
        assert_eq!(extended, fresh.logits(&[1, 2, 3]));
        // A shorter (non-extending) context resets the cache and reruns.
        assert_eq!(first, forward.logits(&[1, 2]));
    }

    #[test]
    fn real_forward_wraps_out_of_range_token_ids() {
        let ck = model::mini_checkpoint();
        let forward = Forward::new(&ck);
        let logits = forward.logits(&[model::MINI_VOCAB as i64 + 1]);
        assert_eq!(logits.len(), model::MINI_VOCAB);
        assert!(logits.iter().all(|logit| logit.is_finite()));
    }

    #[test]
    fn rollback_restores_the_shorter_context_through_the_pool() {
        // The forward-level round-trip (the slot-level port is tested in
        // `kvcache`): append five positions, roll three back, and the
        // two-token context must answer with the fresh-recompute logits.
        let ck = model::mini_checkpoint();
        let forward = Forward::new(&ck);
        let five = forward.logits(&[1, 2, 3, 4, 5]);
        assert!(five.iter().all(|logit| logit.is_finite()));
        forward.rollback(3).expect("three positions roll back");
        let fresh = Forward::new(&ck);
        assert_eq!(fresh.logits(&[1, 2]), forward.logits(&[1, 2]));
        // Committing the remaining positions leaves nothing to roll back.
        forward.commit(2).expect("two positions commit");
        assert!(forward.rollback(1).is_err());
        // Extending past the committed prefix still matches a fresh run.
        assert_eq!(five, forward.logits(&[1, 2, 3, 4, 5]));
    }
}
