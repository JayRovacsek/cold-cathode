//! MTP head + speculative decoding (spec `flash_serve-mtp-kv-npu.md` §1).
//!
//! The reference stages a second `.hgn` — `qwen38-flash-next-mtp.hgn`,
//! named by `HALOGEN_MTP_HEAD` — whose single decoder layer (the 13th
//! attention layer, spec §2) drafts tokens ahead of the trunk; the trunk
//! then verifies the drafts and commits the accepted prefix
//! (`spec_verify` `FUN_02008f60` → forward core `FUN_02008fd0`). This
//! module ports the three pieces the `GEN` path needs:
//!
//! * the **head loader/validator** (`FUN_01f7dc50`, [`MtpHead::load`]):
//!   resolves the head's tensor set by name and enforces the spec §1.1
//!   checks (the RMSNorm pair, the `fc_embedding`/`fc_hidden` projection
//!   pair, the hyper-connection mixer, the Q4C-P expert dtype pair, the
//!   fused `gate_up_proj` dtype rule and the trunk FFN-layout match);
//! * the **head forward** ([`MtpHead::logits`]): the spec §1.1 canonical
//!   combine — `RMSNorm(embedding_of_draft_token)` and
//!   `RMSNorm(trunk_hidden_state)` projected through `fc_embedding` /
//!   `fc_hidden`, merged by the head's hyper-connection mixer, run
//!   through the head's decoder layer and scored by the shared trunk
//!   `lm_head` (the head has no `mtp.lm_head.*` table, spec §1.1);
//! * the **draft/verify loop** (`draft_sample` `FUN_0201c790` +
//!   `spec_verify` `FUN_02008f60`, [`MtpEngine`]): `draft(n)` draws `n`
//!   tokens with their logprobs (row 0 is the live sequence — the trunk
//!   logits through the shared `lm_head`, the `q_row = -1` case of
//!   `draft_sample`; rows 1..n are the head's rows), `verify(n)` runs
//!   the trunk over the `n` draft positions (the forward core's
//!   speculative `hipMemcpyAsync` of the `n` token ids), applies the
//!   `k_accept_prob_f32` accept criterion against the
//!   `HALOGEN_MTP_GATE_P` gate and commits/rolls the slot through the
//!   [`crate::kvcache`] commit-window mechanics.
//!
//! Documented port decisions (spec ambiguities, reported in the task
//! notes):
//!
//! * **MTP enablement**: the task gates the feature on `HALOGEN_MTP=1`
//!   (the reference's `FUN_01f7dc50` treats *unset* as on and only
//!   `0` as off; the port requires the explicit `1` so the port's
//!   default behaviour — and the existing contract bytes — stay
//!   unchanged).
//! * **Acceptance rule** (`k_accept_prob_f32`): the classic speculative
//!   criterion `min(1, p_target/p_draft)` scaled by the gate —
//!   `gate · min(1, p_target/p_draft)` — with the two boundary
//!   shortcuts the spec's `0 = off` implies: gate `0` accepts nothing,
//!   gate `1` accepts everything (see [`accept_probability`]). The
//!   decision is one [`crate::sampling::Rng`] draw per scored draft, so
//!   it is deterministic for a seed.
//! * **Row 0**: `k_smpg_accept_final` writes the sequence row's token
//!   unconditionally; the port always emits/commits the first draft
//!   (its `p_target == p_draft`, ratio 1) and applies the criterion to
//!   the speculative rows 1..n only. With the gate at `0` the round
//!   degenerates to a single committed token — byte-identical to
//!   single-token decode for the same seed.
//! * **Head attention**: the reference's head rows address per-row KV in
//!   the slot's verify rings (`slot+0xb8`); the port's CPU model runs
//!   the head's attention row-locally (the current row's K/V only), a
//!   simplification of the ring layout the spec does not pin.
//! * **Row pairing**: the reference's row `i` consumes the trunk hidden
//!   at position `context.len() + i − 2` (the reference reads these from
//!   the verify rings); the port feeds every head row the same
//!   end-of-context hidden (the cached slot vector,
//!   [`crate::forward::Forward::hidden`]), so rows 2..n pair the correct
//!   draft embedding with a stale trunk hidden. Materialising the
//!   per-row hiddens would require the head's own output hidden, which
//!   [`MtpHead::logits`] does not expose.
//! * **Missing-tensor wording**: the `"{name}: not found in the MTP head
//!   checkpoint"` error string is a port fabrication — the spec pins the
//!   reference's `%s: <check>` error shape but not this wording (see
//!   [`MtpHead::load`]'s doc qualification).
//! * **E2E fixture**: the end-to-end tests generate on the quantised
//!   mini trunk (`model::mini_checkpoint_quantised`, hidden 64) with a
//!   synthetic head fixture ([`MtpHead`] over `head_specs()`) — the real
//!   `qwen38-flash-next-mtp.hgn` named by `HALOGEN_MTP_HEAD` is not
//!   available in the test environment, so the fixtures stand in for
//!   both checkpoints.
//! * **Unported knobs**: the reference's S2 sampling knobs
//!   (`HALOGEN_S2_NUCLEUS`, `HALOGEN_S2_QOFF`, `HALOGEN_S2_TSKEW`) and
//!   its falsifier mode (`METHOD 37`) have no port counterpart, and
//!   `HALOGEN_MTP_LAYER_SRC` is not read — the head's decoder layer is
//!   hardcoded to `mtp.layers.0` (spec §1.1's tensor table).

use std::collections::HashMap;

use crate::codecs;
use crate::forward::{self, Forward};
use crate::hgn;
use crate::kvcache::LayerRuntime;
use crate::model::{self, AttnBlock, Mixer, NORM_EPS};
use crate::ops::{self, Matrix};
use crate::sampling::{self, Rng, SampleParams, SampledToken};
use crate::{
    env_strtod, env_strtol, env_var, DEFAULT_MTP_DEPTH, DEFAULT_MTP_DEPTH_HI, DEFAULT_SPEC_N,
};

/// The reference's dtype→name table (`FUN_020bd2b0`, spec
/// `flash_serve-dtypes.md` §1 / engine spec §1) — the port's counterpart,
/// used by the head validator's `got %s / %s` error wording.
pub fn dtype_name(code: u32) -> Option<&'static str> {
    Some(match code {
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
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Configuration (spec §1.3 — the env surface parsed in lib.rs)
// ---------------------------------------------------------------------------

/// The MTP env surface: `HALOGEN_MTP`, `HALOGEN_MTP_DEPTH`,
/// `HALOGEN_MTP_DEPTH_HI`, `HALOGEN_SPEC_N` and `HALOGEN_MTP_GATE_P`.
#[derive(Debug, Clone, PartialEq)]
pub struct MtpConfig {
    /// `HALOGEN_MTP=1` (the port's opt-in, module docs).
    pub enabled: bool,
    /// `HALOGEN_MTP_DEPTH` (default [`DEFAULT_MTP_DEPTH`]).
    pub depth: i64,
    /// Whether `HALOGEN_MTP_DEPTH` was set (drives [`Self::draft_n`]'s
    /// spec §1.3 "unset" rule without re-reading the environment).
    pub depth_set: bool,
    /// `HALOGEN_MTP_DEPTH_HI` (default [`DEFAULT_MTP_DEPTH_HI`] when
    /// `DEPTH` is unset, else 0; forced 0 unless `DEPTH >= 2` and
    /// `HI > DEPTH`, spec §1.3).
    pub depth_hi: i64,
    /// `HALOGEN_SPEC_N` (default [`DEFAULT_SPEC_N`]).
    pub spec_n: i64,
    /// `HALOGEN_MTP_GATE_P`: default `0.9` when set or when
    /// `DEPTH`/`DEPTH_HI` are set, `0.0` (off) otherwise; forced `0.0`
    /// when MTP is off (spec §1.3).
    pub gate_p: f64,
}

impl MtpConfig {
    /// Read the configuration from the process environment (the same
    /// variables `lib.rs` validates; the parsing helpers are shared).
    pub fn from_env() -> Self {
        let enabled = env_var("HALOGEN_MTP").is_some_and(|v| v == "1");
        let depth_raw = env_var("HALOGEN_MTP_DEPTH");
        let depth_set = depth_raw.is_some();
        let depth = depth_raw
            .as_deref()
            .map(crate::strtol)
            .unwrap_or(DEFAULT_MTP_DEPTH);
        let depth_hi_raw = env_var("HALOGEN_MTP_DEPTH_HI");
        let mut depth_hi = match &depth_hi_raw {
            Some(raw) => crate::strtol(raw),
            // HI defaults to 3 iff DEPTH is unset, else 0 (spec §1.3).
            None => i64::from(depth_raw.is_none()) * DEFAULT_MTP_DEPTH_HI,
        };
        // HI is forced to 0 unless DEPTH >= 2 and HI > DEPTH.
        if depth < 2 || depth_hi <= depth {
            depth_hi = 0;
        }
        let spec_n = env_strtol("HALOGEN_SPEC_N", DEFAULT_SPEC_N);
        let gate_p = if !enabled {
            0.0
        } else if env_var("HALOGEN_MTP_GATE_P").is_some() {
            env_strtod("HALOGEN_MTP_GATE_P", 0.9)
        } else if depth_set || depth_hi_raw.is_some() {
            0.9
        } else {
            0.0
        };
        MtpConfig {
            enabled,
            depth,
            depth_set,
            depth_hi,
            spec_n,
            gate_p,
        }
    }

    /// The effective speculative depth: `min(DEPTH, SPEC_N − 1)`, or
    /// `SPEC_N − 1` when `HALOGEN_MTP_DEPTH` is unset (spec §1.3 "the
    /// effective depth used by the gate struct"). Always ≥ 1: the
    /// `.max(1)` clamp deviates from the spec's `SPEC_N − 1 == 0` when
    /// `SPEC_N == 1` (where the reference drafts nothing); the port
    /// keeps one speculative row so every round still commits the live
    /// row — harmless (documented, review 💡-9).
    pub fn draft_n(&self) -> usize {
        let cap = (self.spec_n - 1).max(1);
        if self.depth_set {
            self.depth.min(cap)
        } else {
            cap
        }
        .max(1) as usize
    }
}

// ---------------------------------------------------------------------------
// Head loader/validator (FUN_01f7dc50)
// ---------------------------------------------------------------------------

/// The loaded MTP head: the tensor set the spec §1.1 validator resolves,
/// decoded for the port's CPU forward.
#[derive(Debug, Clone)]
pub struct MtpHead {
    /// `mtp.pre_fc_norm_embedding.weight` (RMSNorm over the draft
    /// token's embedding).
    pub pre_fc_norm_embedding: Vec<f32>,
    /// `mtp.pre_fc_norm_hidden.weight` (RMSNorm over the trunk hidden).
    pub pre_fc_norm_hidden: Vec<f32>,
    /// `mtp.fc_embedding.weight`.
    pub fc_embedding: Matrix,
    /// `mtp.fc_hidden.weight`.
    pub fc_hidden: Matrix,
    /// `mtp.hyper_connection_mixer.*` (the merge mixer over the two
    /// projected streams; no `block_inject_weight` → mean combine, the
    /// same rule as the trunk's top-level mixer).
    pub mixer: Mixer,
    /// `mtp.layers.0.self_attn.*` — the head's single decoder layer
    /// (the 13th attention layer, spec §2).
    pub attn: AttnBlock,
    /// `mtp.layers.0.experts.*` — the head's FFN (fused or split, the
    /// layout flag validated against the trunk's).
    pub ffn: HeadFfn,
    /// Whether the head's FFN is the fused `gate_up_proj` layout.
    pub fused_ffn: bool,
}

/// The head's FFN tensors (the Q4C-P pair the validator checks): fused
/// `gate_up_proj` + `down_proj`, or the split `gate_proj`/`up_proj` +
/// `down_proj`.
#[derive(Debug, Clone)]
pub struct HeadFfn {
    pub gate_up: Matrix,
    pub down: Matrix,
    pub ffn: usize,
}

impl HeadFfn {
    /// `down · (silu(gate·x) ⊙ up·x)` — the single-expert form of the
    /// trunk's `Mlp::expert_out` (the head's FFN is the expert set
    /// itself; the validator resolves no router).
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let gate = self.gate_up.gemv_rows(0, self.ffn, x);
        let up = self.gate_up.gemv_rows(self.ffn, self.ffn, x);
        let act: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| ops::silu(*g) * *u)
            .collect();
        self.down.gemv(&act)
    }
}

/// The head's tensor names the validator resolves (spec §1.1 table).
const HEAD_NORMS: [(&str, &str); 2] = [
    ("pre_fc_norm_embedding", "mtp.pre_fc_norm_embedding.weight"),
    ("pre_fc_norm_hidden", "mtp.pre_fc_norm_hidden.weight"),
];

impl MtpHead {
    /// Port of `FUN_01f7dc50`: resolve the head's tensors by name against
    /// the head checkpoint and validate them against the spec §1.1 table
    /// (wordings pinned by the spec are verbatim; the rest follow the
    /// reference's `%s: <check>` shape — module docs). `trunk`
    /// supplies the FFN layout flag the head's must match (the
    /// `engine+0x9e8+0x700` byte) and the embedding/hidden widths the
    /// head's norms must match (review 🟡-3).
    pub fn load(trunk: &hgn::Checkpoint, head: &hgn::Checkpoint) -> Result<MtpHead, String> {
        let mut by_name: HashMap<&str, usize> = HashMap::new();
        for (index, entry) in head.entries.iter().enumerate() {
            by_name.entry(entry.name()).or_insert(index);
        }
        let entry = |name: &str| -> Result<&hgn::TensorEntry, String> {
            by_name
                .get(name)
                .map(|index| &head.entries[*index])
                .ok_or_else(|| format!("{name}: not found in the MTP head checkpoint"))
        };
        let matrix = |name: &str| -> Result<Matrix, String> {
            let index = *by_name
                .get(name)
                .ok_or_else(|| format!("{name}: not found in the MTP head checkpoint"))?;
            let e = &head.entries[index];
            let bytes = head
                .tensor_bytes(index)
                .ok_or_else(|| format!("{name}: unreadable payload"))?;
            let dims: Vec<u64> = e.dims().into_iter().filter(|d| *d != 0).collect();
            let (rows, cols) = codecs::shape(&dims)
                .ok_or_else(|| format!("{name}: incompatible shape for the head layout"))?;
            let weights = codecs::Weights::from_dtype(e.dtype(), bytes, &dims, None)
                .ok_or_else(|| format!("{name}: undecodable payload"))?;
            Matrix::with_weights(rows, cols, weights)
                .ok_or_else(|| format!("{name}: incompatible shape for the head layout"))
        };
        let vector = |name: &str| -> Result<Vec<f32>, String> {
            let m = matrix(name)?;
            Ok((0..m.rows).map(|r| m.row(r)).collect::<Vec<_>>().concat())
        };

        // The FFN layout flag (engine+0x9e8+0x700): the trunk's fused
        // flag comes from its layer-0 expert tensor names.
        let trunk_fused = trunk
            .entries
            .iter()
            .any(|e| e.name() == "layers.0.mlp.experts.gate_up_proj.weight");
        let trunk_split = trunk
            .entries
            .iter()
            .any(|e| e.name() == "layers.0.mlp.experts.gate_proj.weight");
        if !trunk_fused && !trunk_split {
            return Err("checkpoint: the trunk's FFN layout is unresolved".to_string());
        }
        // Resolve the head's experts tensor before comparing layouts: a
        // head with no experts tensors must report the missing-tensor
        // wording, not the layout-mismatch one (review 💡-10).
        let head_fused = entry("mtp.layers.0.experts.gate_up_proj.weight").is_ok();
        let head_split = entry("mtp.layers.0.experts.gate_proj.weight").is_ok();
        if !head_fused && !head_split {
            return Err(
                "mtp.layers.0.experts.gate_up_proj.weight: not found in the MTP head checkpoint"
                    .to_string(),
            );
        }
        if head_fused != trunk_fused {
            // The reference's `%s` is the resolved experts tensor name
            // (review 💡-8).
            return Err(
                "mtp.layers.0.experts: the head's FFN layout differs from the trunk's".to_string(),
            );
        }

        // The fused gate_up dtype rule (checked before the experts pair,
        // as in the decompile): a fused gate_up is read in place as
        // Q4C-P (5) or bf16 (0) only.
        let gate_name = if head_fused {
            "mtp.layers.0.experts.gate_up_proj.weight"
        } else {
            "mtp.layers.0.experts.gate_proj.weight"
        };
        let gate_dtype = entry(gate_name)?.dtype();
        if head_fused && gate_dtype != 5 && gate_dtype != 0 {
            return Err(
                "mtp.layers.0.experts.gate_up_proj.weight: a fused gate_up is read in place as \
                 Q4C-P or bf16 only"
                    .to_string(),
            );
        }
        // The Q4C-P experts pair: the gate side must be q4c (5), the
        // `down_proj` side q4c (5) or q8s32 (11).
        let down_name = "mtp.layers.0.experts.down_proj.weight";
        let down_dtype = entry(down_name)?.dtype();
        if gate_dtype != 5 || (down_dtype != 5 && down_dtype != 11) {
            let gate_str = dtype_name(gate_dtype).unwrap_or("unknown");
            let down_str = dtype_name(down_dtype).unwrap_or("unknown");
            return Err(format!(
                "mtp.layers.0.experts: the head's experts must be Q4C-P (got {gate_str} / {down_str})"
            ));
        }

        // Resolve the remaining tensors (the spec §1.1 table).
        let mut norms: HashMap<&str, Vec<f32>> = HashMap::new();
        for (_, name) in HEAD_NORMS {
            norms.insert(name, vector(name)?);
        }
        let fc_embedding = matrix("mtp.fc_embedding.weight")?;
        let fc_hidden = matrix("mtp.fc_hidden.weight")?;
        let mixer_norm = vector("mtp.hyper_connection_mixer.hc_norm.weight")?;
        let mixer_down = matrix("mtp.hyper_connection_mixer.input_mix_weight_down.weight")?;
        let mixer_up = matrix("mtp.hyper_connection_mixer.input_mix_weight_up.weight")?;
        let q = matrix("mtp.layers.0.self_attn.q_proj.weight")?;
        let q_norm = vector("mtp.layers.0.self_attn.q_norm.weight")?;
        let k = matrix("mtp.layers.0.self_attn.k_proj.weight")?;
        let k_norm = vector("mtp.layers.0.self_attn.k_norm.weight")?;
        let v = matrix("mtp.layers.0.self_attn.v_proj.weight")?;
        let o = matrix("mtp.layers.0.self_attn.o_proj.weight")?;

        // Shape consistency (the port's counterpart of the reference's
        // tensor-set resolution: every consumer's width must match).
        let hidden = fc_hidden.rows;
        if fc_embedding.rows != hidden
            || fc_embedding.cols != norms["mtp.pre_fc_norm_embedding.weight"].len()
            || fc_hidden.cols != norms["mtp.pre_fc_norm_hidden.weight"].len()
        {
            return Err("mtp.fc_hidden.weight: incompatible shape for the head layout".to_string());
        }
        // Trunk-side width checks (review 🟡-3): the head's embedding
        // path consumes the trunk's embedding rows, so
        // `pre_fc_norm_embedding` must match the trunk's embedding row
        // width (`embed_tokens.weight`'s hidden width, spec §2) — a
        // wrong-model head previously validated OK and debug-panicked in
        // `ops::rmsnorm`. The trunk's hidden width is the same
        // `hidden_per_stream` value, so the hidden path compares against
        // it too.
        let trunk_embed_width = trunk
            .entries
            .iter()
            .find(|e| e.name() == "embed_tokens.weight")
            .and_then(|e| usize::try_from(*e.dims().get(1)?).ok())
            .ok_or_else(|| {
                "checkpoint: the trunk's embed_tokens.weight is unresolved".to_string()
            })?;
        if norms["mtp.pre_fc_norm_embedding.weight"].len() != trunk_embed_width {
            return Err(
                "mtp.pre_fc_norm_embedding.weight: the head's embedding width differs from the trunk's"
                    .to_string(),
            );
        }
        if hidden != trunk_embed_width {
            return Err(
                "mtp.fc_hidden.weight: the head's hidden width differs from the trunk's"
                    .to_string(),
            );
        }
        if mixer_norm.len() != 2 * hidden
            || mixer_down.cols != 2 * hidden
            || mixer_up.rows != 2 * hidden
            || mixer_down.rows != mixer_up.cols
        {
            return Err(
                "mtp.hyper_connection_mixer.hc_norm.weight: incompatible shape for the head layout"
                    .to_string(),
            );
        }
        if q.cols != hidden || o.rows != hidden || o.cols != q.rows / 2 {
            return Err(
                "mtp.layers.0.self_attn.o_proj.weight: incompatible shape for the head layout"
                    .to_string(),
            );
        }
        let head_dim = q_norm.len();
        if head_dim == 0 || q.rows % 2 != 0 || q.rows / 2 != head_dim || k_norm.len() != head_dim {
            return Err(
                "mtp.layers.0.self_attn.q_norm.weight: incompatible shape for the head layout"
                    .to_string(),
            );
        }
        if k.cols != hidden || v.cols != hidden || k.rows % head_dim != 0 || v.rows % head_dim != 0
        {
            return Err(
                "mtp.layers.0.self_attn.k_proj.weight: incompatible shape for the head layout"
                    .to_string(),
            );
        }
        let gate_up = if head_fused {
            matrix("mtp.layers.0.experts.gate_up_proj.weight")?
        } else {
            // The split layout: stack gate/up into the fused shape the
            // port's HeadFfn consumes.
            let gate = matrix("mtp.layers.0.experts.gate_proj.weight")?;
            let up = matrix("mtp.layers.0.experts.up_proj.weight")?;
            if gate.cols != hidden || up.rows != gate.rows || up.cols != gate.cols {
                return Err(
                    "mtp.layers.0.experts.up_proj.weight: incompatible shape for the head layout"
                        .to_string(),
                );
            }
            stack_rows(&[&gate, &up])
        };
        let down = matrix(down_name)?;
        if gate_up.cols != hidden || down.rows != hidden || gate_up.rows % 2 != 0 {
            return Err(format!(
                "{gate_name}: incompatible shape for the head layout"
            ));
        }
        let ffn = gate_up.rows / 2;
        if down.cols != ffn {
            return Err(format!(
                "{down_name}: incompatible shape for the head layout"
            ));
        }

        let q_heads = (q.rows / 2) / head_dim;
        let kv_heads = k.rows / head_dim;
        Ok(MtpHead {
            pre_fc_norm_embedding: norms["mtp.pre_fc_norm_embedding.weight"].clone(),
            pre_fc_norm_hidden: norms["mtp.pre_fc_norm_hidden.weight"].clone(),
            fc_embedding,
            fc_hidden,
            mixer: Mixer {
                norm: mixer_norm,
                down: mixer_down,
                up: mixer_up,
                inject: None,
                streams: 2,
                hidden_per_stream: hidden,
            },
            attn: AttnBlock {
                q,
                q_norm,
                k,
                k_norm,
                v,
                o,
                q_heads,
                kv_heads,
                head_dim,
            },
            ffn: HeadFfn { gate_up, down, ffn },
            fused_ffn: head_fused,
        })
    }

    /// The head forward (spec §1.1 canonical combine): normalise the
    /// draft token's embedding and the trunk hidden state with the
    /// pre-fc RMSNorm pair, project through `fc_embedding`/`fc_hidden`,
    /// merge via the head's hyper-connection mixer, run the head's
    /// decoder layer (row-local attention + the expert FFN) and score
    /// the result with the shared trunk `lm_head`.
    pub fn logits(
        &self,
        trunk: &model::Model,
        trunk_hidden: &[f32],
        position: usize,
        draft_token: i64,
    ) -> Vec<f32> {
        let row = draft_token.rem_euclid(trunk.dims.vocab as i64) as usize;
        let embedding = trunk.embed.row(row);
        let e_normed = ops::rmsnorm(&embedding, &self.pre_fc_norm_embedding, NORM_EPS);
        let h_normed = ops::rmsnorm(trunk_hidden, &self.pre_fc_norm_hidden, NORM_EPS);
        let e_proj = self.fc_embedding.gemv(&e_normed);
        let h_proj = self.fc_hidden.gemv(&h_normed);
        // The head's merge mixer over the [embedding, hidden] stream
        // pair (the same low-rank gate + mean combine as `Mixer::mix`).
        let stack: Vec<f32> = e_proj.iter().chain(&h_proj).copied().collect();
        let (merged, _, _) = self.mixer.mix(&stack);
        // The head's decoder layer: full attention (row-local in the
        // port, module docs) + the expert FFN.
        let mut runtime = LayerRuntime {
            k: Vec::new(),
            v: Vec::new(),
            conv_hist: Vec::new(),
            delta: Vec::new(),
        };
        let attn_out = forward::attention_block(&self.attn, &merged, position, &mut runtime);
        let ffn_out = self.ffn.forward(&attn_out);
        trunk.lm_head.gemv(&ffn_out)
    }
}

/// Stack matrices with identical column counts into one matrix (the
/// split-layout `gate`/`up` pair read as the fused `gate_up` shape).
fn stack_rows(matrices: &[&Matrix]) -> Matrix {
    let cols = matrices[0].cols;
    let rows: usize = matrices.iter().map(|m| m.rows).sum();
    let mut data = Vec::with_capacity(rows * cols);
    for m in matrices {
        for r in 0..m.rows {
            data.extend_from_slice(&m.row(r));
        }
    }
    Matrix::new(rows, cols, data).expect("stacked rows keep the column count")
}

// ---------------------------------------------------------------------------
// Acceptance criterion (k_accept_prob_f32)
// ---------------------------------------------------------------------------

/// The acceptance probability for one draft token (the port's
/// `k_accept_prob_f32`): the classic speculative-sampling criterion
/// `min(1, p_target/p_draft)` scaled by the `HALOGEN_MTP_GATE_P` gate,
/// with the spec's `0 = off` boundary: gate `0` accepts nothing, gate
/// `1` accepts everything (module docs).
pub fn accept_probability(p_target: f64, p_draft: f64, gate_p: f64) -> f64 {
    if gate_p >= 1.0 {
        return 1.0;
    }
    if gate_p <= 0.0 {
        return 0.0;
    }
    let ratio = if p_draft > 0.0 {
        (p_target / p_draft).min(1.0)
    } else {
        1.0
    };
    (gate_p * ratio).clamp(0.0, 1.0)
}

/// Draw the accept/reject decision for one draft token (one
/// [`crate::sampling::Rng`] step — deterministic for a seed).
/// Deviation (review 💡-5): the reference's `draft_sample` refuses
/// `temperature == 0` (§1.2); the port's greedy mode (temperature 0)
/// still scores acceptance through the same criterion, comparing the
/// argmax tokens' probabilities from both distributions.
pub fn accept_draft(p_target: f64, p_draft: f64, gate_p: f64, rng: &mut Rng) -> bool {
    rng.next_f64() < accept_probability(p_target, p_draft, gate_p)
}

// ---------------------------------------------------------------------------
// Draft/verify loop (draft_sample FUN_0201c790 + spec_verify FUN_02008f60)
// ---------------------------------------------------------------------------

/// One drafted token with the logits it was drawn from (row 0: the
/// trunk's logits through the shared `lm_head`; rows 1..n: the head's).
#[derive(Debug, Clone)]
pub struct DraftToken {
    pub token: SampledToken,
    pub logits: Vec<f32>,
}

/// The result of one `GEN` run through the MTP path: the `T` lines, the
/// generated-token count, the done reason and the speculative accounting
/// the `D` line carries (`rounds`/`commit`, engine spec §8).
#[derive(Debug, Clone)]
pub struct MtpGeneration {
    pub lines: Vec<String>,
    pub n_gen: usize,
    pub reason: &'static str,
    pub rounds: usize,
    pub committed: usize,
}

/// The draft/verify engine over a loaded trunk + staged head.
#[derive(Debug)]
pub struct MtpEngine<'a> {
    forward: &'a Forward<'a>,
    model: &'a model::Model,
    head: MtpHead,
    config: MtpConfig,
}

impl<'a> MtpEngine<'a> {
    /// Bind the engine to a forward pass whose checkpoint loads the real
    /// model (the stub path has no trunk hidden state to feed the head).
    pub fn new(forward: &'a Forward<'a>, head: MtpHead, config: MtpConfig) -> Option<Self> {
        let model = forward.model()?;
        Some(MtpEngine {
            forward,
            model,
            head,
            config,
        })
    }

    /// Build the engine from the environment: `None` when MTP is off
    /// (`HALOGEN_MTP != 1`), the head file (`HALOGEN_MTP_HEAD`, default
    /// `/models/qwen38-flash-next-mtp.hgn`) is unreadable or fails
    /// validation (the reference's `asked for drafter … (MTP head not
    /// staged, HALOGEN_MTP=1)` case), or the trunk has no real model.
    pub fn from_env(forward: &'a Forward<'a>) -> Option<Self> {
        let config = MtpConfig::from_env();
        if !config.enabled {
            return None;
        }
        let head_path =
            env_var("HALOGEN_MTP_HEAD").unwrap_or_else(|| crate::DEFAULT_HEAD.to_string());
        let head_ck = hgn::load(&head_path).ok()?;
        let head = MtpHead::load(forward.checkpoint(), &head_ck).ok()?;
        Self::new(forward, head, config)
    }

    /// `draft_sample` port: draw `n` tokens with their logprobs. Row 0
    /// (the live sequence, `q_row = -1`) comes from the trunk's logits;
    /// rows 1..n come from the head's forward over the trunk hidden and
    /// the previous draft's embedding (spec §1.1). `trunk_hidden` is
    /// `Some` whenever `n > 1` (the cached slot vector,
    /// [`crate::forward::Forward::hidden`] — review 🟡-1); the `n == 1`
    /// round uses the trunk logits alone, so callers pass `None`.
    pub fn draft(
        &self,
        n: usize,
        context: &[i64],
        params: &SampleParams,
        rng: &mut Rng,
        trunk_hidden: Option<&[f32]>,
    ) -> Option<Vec<DraftToken>> {
        let mut drafts = Vec::with_capacity(n);
        // Row 0: the live sequence's logits (the shared lm_head on the
        // trunk hidden — the Forward cache answers from the slot).
        let trunk_logits = self.forward.logits(context);
        let first = sampling::sample(&trunk_logits, params, rng)?;
        drafts.push(DraftToken {
            token: first,
            logits: trunk_logits,
        });
        if n > 1 {
            let trunk_hidden = trunk_hidden?;
            for i in 1..n {
                // The row's RoPE position: the predicted token's own
                // position (the trunk's rows use the input-token
                // position — documented choice, review 💡-3).
                let position = context.len() + i;
                let logits = self.head.logits(
                    self.model,
                    trunk_hidden,
                    position,
                    drafts[i - 1].token.index as i64,
                );
                let token = sampling::sample(&logits, params, rng)?;
                drafts.push(DraftToken { token, logits });
            }
        }
        Some(drafts)
    }

    /// `spec_verify` port: run the trunk over the `n` draft positions
    /// (the forward core's speculative write of the `n` token ids),
    /// apply the accept criterion and commit/roll back the slot. Returns
    /// the emitted tokens (the accepted prefix — row 0 is always
    /// committed, module docs) and the committed count, or the
    /// reference's `spec_verify(n=%d): …` wording when the call is
    /// malformed (review 💡-1: `1 <= n <= HALOGEN_SPEC_N`, enough
    /// drafts, and a warm sequence — a non-empty context).
    pub fn verify(
        &self,
        n: usize,
        context: &[i64],
        drafts: &[DraftToken],
        rng: &mut Rng,
    ) -> Result<(Vec<SampledToken>, usize), String> {
        // The reference's `spec_verify` gate (FUN_02008f60): the port
        // replaces the verify-ring check with its own slot state but
        // keeps the reference's wording and the `1 <= n <=
        // HALOGEN_SPEC_N` bound (review 💡-1, replacing the old
        // `assert!`).
        let spec_n = usize::try_from(self.config.spec_n.max(0)).unwrap_or(usize::MAX);
        if n < 1 || n > spec_n || drafts.len() < n || context.is_empty() {
            return Err(format!(
                "spec_verify(n={n}): needs the verify rings (HALOGEN_MTP=1 or a lookup / \
                 grammar drafter), 1 <= n <= {spec_n} (HALOGEN_SPEC_N) and a warm sequence \
                 (pos {})",
                context.len()
            ));
        }
        // Speculative append: write all n draft rows to the slot (the
        // forward core's `hipMemcpyAsync(slot+0xda8, …, n·4)`), keeping
        // the per-position logits chain for scoring.
        let mut chain = vec![self.forward.logits(context)];
        let mut prefix = context.to_vec();
        for draft in drafts.iter().take(n) {
            prefix.push(draft.token.index as i64);
            chain.push(self.forward.logits(&prefix));
        }
        // Row 0 is the live sequence: `k_smpg_accept_final` writes it
        // unconditionally (its p_target == p_draft, ratio 1).
        let mut accepted = vec![drafts[0].token.clone()];
        let mut k = 1usize;
        for i in 1..n {
            let p_target = sampling::token_logprob(&chain[i], drafts[i].token.index).exp();
            let p_draft = sampling::token_logprob(&drafts[i].logits, drafts[i].token.index).exp();
            if !accept_draft(p_target, p_draft, self.config.gate_p, rng) {
                break;
            }
            accepted.push(drafts[i].token.clone());
            k += 1;
        }
        // Commit the accepted prefix, roll back the rejected rows.
        self.forward.rollback(n - k)?;
        self.forward.commit(k)?;
        Ok((accepted, k))
    }

    /// The generation loop: rounds of `draft` → `verify` → commit/rollback
    /// until the token budget runs out or an eos token is emitted.
    pub fn generate(
        &self,
        request: &crate::protocol::GenRequest,
        params: &SampleParams,
        limit: usize,
    ) -> Option<MtpGeneration> {
        let mut rng = Rng::new(params.seed);
        let mut lines = Vec::new();
        let mut context = request.ids.clone();
        let mut n_gen = 0usize;
        let mut reason = "length";
        let mut rounds = 0usize;
        let mut committed = 0usize;
        while n_gen < limit {
            // The gate-0 round degenerates to the live row alone (the
            // module docs' byte-identity rule).
            let n = if self.config.gate_p <= 0.0 {
                1
            } else {
                self.config.draft_n().min(limit - n_gen).max(1)
            };
            // The trunk hidden the head rows consume (review 🟡-1): the
            // cached slot vector (O(1) amortised, replacing the
            // per-round full-context replay); the n == 1 round never
            // uses it, so it is only fetched when the round drafts head
            // rows (syncing the slot first so the cache matches).
            let trunk_hidden = if n > 1 {
                self.forward.logits(&context);
                self.forward.hidden(&context)
            } else {
                None
            };
            let drafts = self.draft(n, &context, params, &mut rng, trunk_hidden.as_deref())?;
            let (emitted, k) = self.verify(n, &context, &drafts, &mut rng).ok()?;
            rounds += 1;
            committed += k;
            // Tokens accepted after an eos within a round stay committed
            // to the slot (verify commits the whole accepted prefix
            // before emission, matching the reference's per-round
            // commit); the port emits only up to the eos, matching
            // `generate_tokens`' clean stop (review 💡-4).
            for token in emitted {
                let token_id = token.index as i64;
                lines.push(format!(
                    "T {} {} {}",
                    request.req,
                    token.index,
                    crate::fmt_g(token.logprob)
                ));
                context.push(token_id);
                n_gen += 1;
                if request.eos.contains(&token_id) {
                    reason = "stop";
                    break;
                }
                if n_gen >= limit {
                    break;
                }
            }
            if reason == "stop" {
                break;
            }
        }
        Some(MtpGeneration {
            lines,
            n_gen,
            reason,
            rounds,
            committed,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bf16;
    use crate::protocol::{gen_response_with, EngineInfo, GenRequest};

    /// One head-fixture entry: name, kind, dims, dtype (mirrors
    /// `model`'s private `FixtureSpec` for the head's tensor set).
    struct HeadSpec(&'static str, u32, Vec<u64>, u32);

    /// Build an `.hgn` checkpoint from head specs (layout per
    /// `hgn::TensorEntry`; bf16 payloads via `bf16::encode_bf16`, q4c
    /// payloads via `codecs::q4c_encode`; other dtypes carry bf16 bytes
    /// the validator rejects before decoding).
    fn head_checkpoint(file: &str, specs: &[HeadSpec]) -> hgn::Checkpoint {
        let entry_count = specs.len() as u64;
        let table_offset = hgn::HEADER_BYTES as u64;
        let data_offset =
            codecs::pad16(table_offset as usize + entry_count as usize * hgn::ENTRY_BYTES) as u64;
        let mut payloads: Vec<u8> = Vec::new();
        let mut entries: Vec<u8> = Vec::new();
        for HeadSpec(name, kind, dims, dtype) in specs {
            let values = fixture_values(name, dims);
            let (rows, cols) = codecs::shape(&dims.clone()).expect("head fixture dims");
            let payload = match dtype {
                5 => codecs::q4c_encode(&values, rows, cols).expect("q4c head payload"),
                _ => bf16::encode_bf16(&values),
            };
            let offset = data_offset + payloads.len() as u64;
            let mut entry = [0u8; hgn::ENTRY_BYTES];
            entry[..name.len()].copy_from_slice(name.as_bytes());
            entry[96..100].copy_from_slice(&dtype.to_le_bytes());
            entry[100..104].copy_from_slice(&kind.to_le_bytes());
            for (i, dim) in dims.iter().take(4).enumerate() {
                entry[104 + i * 8..112 + i * 8].copy_from_slice(&dim.to_le_bytes());
            }
            entry[136..144].copy_from_slice(&offset.to_le_bytes());
            entry[144..152].copy_from_slice(&(payload.len() as u64).to_le_bytes());
            entries.extend_from_slice(&entry);
            payloads.extend_from_slice(&payload);
            payloads.resize(codecs::pad16(payloads.len()), 0);
        }
        let declared = data_offset + payloads.len() as u64;
        let mut bytes = Vec::with_capacity(declared as usize);
        bytes.extend_from_slice(b"HGN1");
        bytes.extend_from_slice(&2u32.to_le_bytes()); // version
        bytes.extend_from_slice(&entry_count.to_le_bytes());
        bytes.extend_from_slice(&table_offset.to_le_bytes());
        bytes.extend_from_slice(&data_offset.to_le_bytes());
        bytes.extend_from_slice(&declared.to_le_bytes());
        let mut model_id = [0u8; 64];
        model_id[..4].copy_from_slice(b"qwen");
        bytes.extend_from_slice(&model_id);
        bytes.extend_from_slice(&entries);
        bytes.resize(data_offset as usize, 0);
        bytes.extend_from_slice(&payloads);
        hgn::load_prefix(file, bytes, declared).expect("head fixture loads")
    }

    /// Deterministic pseudo-random payload values in `[-0.5, 0.5)`,
    /// seeded by the tensor name (the `model` fixtures' scheme).
    fn fixture_values(name: &str, dims: &[u64]) -> Vec<f32> {
        let count = dims.iter().product::<u64>().max(1) as usize;
        let mut rng = Rng::from_u64(fnv1a(name.as_bytes()));
        (0..count).map(|_| rng.next_f32() - 0.5).collect()
    }

    /// FNV-1a over the name bytes (the reference's hash family).
    fn fnv1a(bytes: &[u8]) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
        hash
    }

    /// The valid head tensor set over a hidden-64 trunk (the
    /// `mini_checkpoint_quantised` width): bf16 norms/projections/attn,
    /// q4c fused experts (gate_up [64, 64], down [64, 32]).
    fn head_specs() -> Vec<HeadSpec> {
        vec![
            HeadSpec("mtp.pre_fc_norm_embedding.weight", 1, vec![64], 0),
            HeadSpec("mtp.pre_fc_norm_hidden.weight", 1, vec![64], 0),
            HeadSpec("mtp.fc_embedding.weight", 2, vec![64, 64], 0),
            HeadSpec("mtp.fc_hidden.weight", 2, vec![64, 64], 0),
            HeadSpec("mtp.hyper_connection_mixer.hc_norm.weight", 1, vec![128], 0),
            HeadSpec(
                "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
                2,
                vec![8, 128],
                0,
            ),
            HeadSpec(
                "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
                2,
                vec![128, 8],
                0,
            ),
            HeadSpec("mtp.layers.0.self_attn.q_proj.weight", 2, vec![16, 64], 0),
            HeadSpec("mtp.layers.0.self_attn.q_norm.weight", 1, vec![8], 0),
            HeadSpec("mtp.layers.0.self_attn.k_proj.weight", 2, vec![8, 64], 0),
            HeadSpec("mtp.layers.0.self_attn.k_norm.weight", 1, vec![8], 0),
            HeadSpec("mtp.layers.0.self_attn.v_proj.weight", 2, vec![8, 64], 0),
            HeadSpec("mtp.layers.0.self_attn.o_proj.weight", 2, vec![64, 8], 0),
            HeadSpec(
                "mtp.layers.0.experts.gate_up_proj.weight",
                2,
                vec![64, 64],
                5,
            ),
            HeadSpec("mtp.layers.0.experts.down_proj.weight", 2, vec![64, 32], 5),
        ]
    }

    /// The quantised mini trunk (hidden 64) the e2e tests generate on.
    fn trunk() -> hgn::Checkpoint {
        model::mini_checkpoint_quantised()
    }

    fn head() -> hgn::Checkpoint {
        head_checkpoint("ccfs-mtp-head.hgn", &head_specs())
    }

    fn fixture_info() -> EngineInfo {
        EngineInfo {
            version: "0.18.21".to_string(),
            ctx: 32768,
            max_tok: 8192,
            slots: 4,
            slot_ctx: 32768,
            kv_pool: 0,
            rope_factor: 1.0,
            pld_n: 0,
            pld_k: 0,
            checkpoint_format: "hgn".to_string(),
        }
    }

    #[test]
    fn dtype_name_matches_the_reference_table() {
        for (code, name) in [
            (0, "bf16"),
            (1, "f32"),
            (2, "f16"),
            (5, "q4c"),
            (7, "q8g64"),
            (11, "q8s32"),
            (16, "ht"),
            (23, "i4r"),
            (24, "q6g64"),
        ] {
            assert_eq!(dtype_name(code), Some(name), "dtype {code}");
        }
        assert_eq!(dtype_name(9), None);
        assert_eq!(dtype_name(25), None);
    }

    #[test]
    fn mtp_config_from_env_matches_the_spec_surface() {
        // Shared with the lib.rs env test: Rust runs tests on parallel
        // threads, so every env-mutating test takes the crate lock
        // (review 💡-11: `into_inner` on a poisoned lock keeps the
        // serialisation instead of panicking).
        let _guard = crate::ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        for (key, value) in [
            ("HALOGEN_MTP", "1"),
            ("HALOGEN_MTP_DEPTH", "3"),
            ("HALOGEN_MTP_DEPTH_HI", "5"),
            ("HALOGEN_SPEC_N", "4"),
            ("HALOGEN_MTP_GATE_P", "0.7"),
        ] {
            std::env::set_var(key, value);
        }
        let config = MtpConfig::from_env();
        assert_eq!(
            config,
            MtpConfig {
                enabled: true,
                depth: 3,
                depth_set: true,
                depth_hi: 5,
                spec_n: 4,
                gate_p: 0.7,
            }
        );
        assert_eq!(config.draft_n(), 3); // min(DEPTH, SPEC_N - 1)
        for (key, _) in [
            ("HALOGEN_MTP", "1"),
            ("HALOGEN_MTP_DEPTH", "3"),
            ("HALOGEN_MTP_DEPTH_HI", "5"),
            ("HALOGEN_SPEC_N", "4"),
            ("HALOGEN_MTP_GATE_P", "0.7"),
        ] {
            std::env::remove_var(key);
        }
        // MTP=1 alone: depth defaults to 2, HI to 3, gate 0 (off).
        std::env::set_var("HALOGEN_MTP", "1");
        let defaults = MtpConfig::from_env();
        assert_eq!(
            defaults,
            MtpConfig {
                enabled: true,
                depth: 2,
                depth_set: false,
                depth_hi: 3,
                spec_n: 3,
                gate_p: 0.0,
            }
        );
        assert_eq!(defaults.draft_n(), 2); // SPEC_N - 1 (DEPTH unset)
        std::env::remove_var("HALOGEN_MTP");
        // Unset: disabled, gate forced 0.
        let off = MtpConfig::from_env();
        assert!(!off.enabled);
        assert_eq!(off.gate_p, 0.0);
    }

    #[test]
    fn valid_head_fixture_is_accepted() {
        let loaded = MtpHead::load(&trunk(), &head()).expect("the fixture head validates");
        assert!(loaded.fused_ffn);
        assert_eq!(loaded.ffn.ffn, 32);
        assert_eq!(loaded.attn.head_dim, 8);
        assert_eq!(loaded.attn.q_heads, 1);
        assert_eq!(loaded.attn.kv_heads, 1);
        assert_eq!(loaded.mixer.streams, 2);
        assert_eq!(loaded.mixer.hidden_per_stream, 64);
    }

    #[test]
    fn wrong_expert_dtypes_are_rejected() {
        // A fused gate_up outside {q4c, bf16} hits the fused rule.
        let mut specs = head_specs();
        specs[13].3 = 7; // gate_up_proj → q8g64
        let err = MtpHead::load(&trunk(), &head_checkpoint("ccfs-mtp-head-bad.hgn", &specs))
            .expect_err("q8g64 gate_up must fail");
        assert_eq!(
            err,
            "mtp.layers.0.experts.gate_up_proj.weight: a fused gate_up is read in place as \
             Q4C-P or bf16 only"
        );
        // gate q4c + down outside {q4c, q8s32} hits the experts-pair rule.
        let mut specs = head_specs();
        specs[14].3 = 7; // down_proj → q8g64
        let err = MtpHead::load(&trunk(), &head_checkpoint("ccfs-mtp-head-bad2.hgn", &specs))
            .expect_err("q8g64 down_proj must fail");
        assert_eq!(
            err,
            "mtp.layers.0.experts: the head's experts must be Q4C-P (got q4c / q8g64)"
        );
    }

    #[test]
    fn missing_tensor_is_rejected() {
        let mut specs = head_specs();
        specs.retain(|spec| spec.0 != "mtp.fc_hidden.weight");
        let err = MtpHead::load(
            &trunk(),
            &head_checkpoint("ccfs-mtp-head-missing.hgn", &specs),
        )
        .expect_err("a missing tensor must fail");
        assert_eq!(
            err,
            "mtp.fc_hidden.weight: not found in the MTP head checkpoint"
        );
    }

    #[test]
    fn ffn_layout_mismatch_is_rejected() {
        // The trunk is fused; a head with the split gate/up layout fails
        // (the error prefix is the resolved experts tensor name,
        // review 💡-8).
        let mut specs = head_specs();
        specs.retain(|spec| spec.0 != "mtp.layers.0.experts.gate_up_proj.weight");
        specs.push(HeadSpec(
            "mtp.layers.0.experts.gate_proj.weight",
            2,
            vec![64, 64],
            5,
        ));
        specs.push(HeadSpec(
            "mtp.layers.0.experts.up_proj.weight",
            2,
            vec![64, 64],
            5,
        ));
        let err = MtpHead::load(
            &trunk(),
            &head_checkpoint("ccfs-mtp-head-split.hgn", &specs),
        )
        .expect_err("a split head on a fused trunk must fail");
        assert_eq!(
            err,
            "mtp.layers.0.experts: the head's FFN layout differs from the trunk's"
        );
    }

    #[test]
    fn wrong_width_head_is_rejected() {
        // A head whose embedding path is narrower than the trunk's
        // embedding rows (norm + fc_embedding both 32-wide against the
        // trunk's 64) passes the head-internal checks and must fail the
        // trunk-side comparison (review 🟡-3) instead of debug-panicking
        // in ops::rmsnorm.
        let mut specs = head_specs();
        specs[0] = HeadSpec("mtp.pre_fc_norm_embedding.weight", 1, vec![32], 0);
        specs[2] = HeadSpec("mtp.fc_embedding.weight", 2, vec![64, 32], 0);
        let err = MtpHead::load(
            &trunk(),
            &head_checkpoint("ccfs-mtp-head-narrow.hgn", &specs),
        )
        .expect_err("a head narrower than the trunk's embedding must fail");
        assert_eq!(
            err,
            "mtp.pre_fc_norm_embedding.weight: the head's embedding width differs from the trunk's"
        );
    }

    #[test]
    fn accept_probability_boundaries_and_scaling() {
        // Gate 0 accepts nothing, gate 1 accepts everything (spec §1.3).
        assert_eq!(accept_probability(0.1, 0.9, 0.0), 0.0);
        assert_eq!(accept_probability(0.1, 0.9, 1.0), 1.0);
        // Mid gate: gate · min(1, p_target/p_draft).
        assert!((accept_probability(0.45, 0.9, 0.5) - 0.25).abs() < 1e-12);
        // p_target > p_draft clamps the ratio at 1.
        assert!((accept_probability(0.9, 0.45, 0.5) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn accept_draft_is_deterministic_per_seed() {
        // Gate 1 always accepts (no draw below 1.0 fails), gate 0 never.
        assert!(accept_draft(0.5, 0.5, 1.0, &mut Rng::new(7)));
        assert!(!accept_draft(0.5, 0.5, 0.0, &mut Rng::new(7)));
        // A mid gate replays identically for the same seed.
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        let decisions_a: Vec<bool> = (0..8)
            .map(|_| accept_draft(0.5, 0.5, 0.5, &mut a))
            .collect();
        let decisions_b: Vec<bool> = (0..8)
            .map(|_| accept_draft(0.5, 0.5, 0.5, &mut b))
            .collect();
        assert_eq!(decisions_a, decisions_b);
    }

    #[test]
    fn verify_forced_accept_commits_every_draft() {
        let trunk = trunk();
        let head_ck = head();
        let head = MtpHead::load(&trunk, &head_ck).unwrap();
        let forward = Forward::new(&trunk);
        let config = MtpConfig {
            enabled: true,
            depth: 2,
            depth_set: true,
            depth_hi: 0,
            spec_n: 3,
            gate_p: 1.0,
        };
        let engine = MtpEngine::new(&forward, head, config).unwrap();
        let context = vec![1i64, 2];
        let params = SampleParams::greedy();
        let mut rng = Rng::new(42);
        let hidden = forward::trunk_hidden(engine.model, &context);
        let drafts = engine
            .draft(3, &context, &params, &mut rng, Some(&hidden))
            .unwrap();
        let (emitted, k) = engine.verify(3, &context, &drafts, &mut rng).unwrap();
        assert_eq!(k, 3, "gate 1 accepts every scored draft");
        assert_eq!(emitted.len(), 3);
        for (i, token) in emitted.iter().enumerate() {
            assert_eq!(token.index, drafts[i].token.index);
        }
        // The slot holds prompt(2) + drafts(3) rows with 3 committed:
        // the commit covers the two prompt rows and the first draft row,
        // so the two remaining uncommitted rows are the accepted draft
        // rows 2..3 — committing them empties the window (the kvcache
        // contract).
        assert!(forward.commit(2).is_ok());
        assert!(forward.commit(1).is_err());
    }

    #[test]
    fn verify_forced_reject_commits_only_row_zero() {
        let trunk = trunk();
        let head_ck = head();
        let head = MtpHead::load(&trunk, &head_ck).unwrap();
        let forward = Forward::new(&trunk);
        let config = MtpConfig {
            enabled: true,
            depth: 2,
            depth_set: true,
            depth_hi: 0,
            spec_n: 3,
            gate_p: 0.0,
        };
        let engine = MtpEngine::new(&forward, head, config).unwrap();
        let context = vec![1i64, 2];
        let params = SampleParams::greedy();
        let mut rng = Rng::new(42);
        let hidden = forward::trunk_hidden(engine.model, &context);
        let drafts = engine
            .draft(3, &context, &params, &mut rng, Some(&hidden))
            .unwrap();
        let (emitted, k) = engine.verify(3, &context, &drafts, &mut rng).unwrap();
        assert_eq!(k, 1, "gate 0 keeps only the always-committed row 0");
        assert_eq!(emitted.len(), 1);
        assert_eq!(emitted[0].index, drafts[0].token.index);
        // prompt(2) + drafts(3) rows, rollback(2) dropped the rejected
        // drafts, commit(1) committed the first prompt row → the
        // remaining uncommitted rows are the second prompt row and the
        // accepted draft row 0 (not two prompt rows, review 💡-2).
        assert!(forward.commit(2).is_ok());
        assert!(forward.commit(1).is_err());
    }

    #[test]
    fn verify_partial_accept_commits_the_seeded_prefix() {
        // Review 💡-6: a mid-gate seeded round commits exactly the
        // prefix the rng stream accepts (k = 2 of 3 for this seed), not
        // just the all-accept / all-reject extremes.
        let trunk = trunk();
        let head_ck = head();
        let head = MtpHead::load(&trunk, &head_ck).unwrap();
        let forward = Forward::new(&trunk);
        let config = MtpConfig {
            enabled: true,
            depth: 2,
            depth_set: true,
            depth_hi: 0,
            spec_n: 3,
            gate_p: 0.5,
        };
        let engine = MtpEngine::new(&forward, head, config).unwrap();
        let context = vec![1i64, 2];
        let params = SampleParams::greedy();
        let mut rng = Rng::new(3);
        let hidden = forward::trunk_hidden(engine.model, &context);
        let drafts = engine
            .draft(3, &context, &params, &mut rng, Some(&hidden))
            .unwrap();
        let (emitted, k) = engine.verify(3, &context, &drafts, &mut rng).unwrap();
        assert_eq!(k, 2, "seeded mid-gate round accepts rows 0..1 only");
        assert_eq!(emitted.len(), 2);
        assert_eq!(emitted[0].index, drafts[0].token.index);
        assert_eq!(emitted[1].index, drafts[1].token.index);
        // prompt(2) + drafts(3) rows, rollback(1) dropped the rejected
        // draft row 2, verify's commit(2) committed the prompt rows →
        // the accepted draft rows 0 and 1 remain uncommitted (the same
        // accounting shape as the forced-reject round, review 💡-2).
        assert!(forward.commit(2).is_ok());
        assert!(forward.commit(1).is_err());
    }

    #[test]
    fn verify_rejects_a_malformed_request_with_the_reference_wording() {
        // Review 💡-1: the public verify returns the reference's
        // `spec_verify(n=%d): …` wording instead of panicking.
        let trunk = trunk();
        let head = MtpHead::load(&trunk, &head()).unwrap();
        let forward = Forward::new(&trunk);
        let config = MtpConfig {
            enabled: true,
            depth: 2,
            depth_set: true,
            depth_hi: 0,
            spec_n: 3,
            gate_p: 1.0,
        };
        let engine = MtpEngine::new(&forward, head, config).unwrap();
        let err = engine
            .verify(4, &[1, 2], &[], &mut Rng::new(1))
            .expect_err("n above HALOGEN_SPEC_N must be refused");
        assert_eq!(
            err,
            "spec_verify(n=4): needs the verify rings (HALOGEN_MTP=1 or a lookup / grammar \
             drafter), 1 <= n <= 3 (HALOGEN_SPEC_N) and a warm sequence (pos 2)"
        );
        let err = engine
            .verify(0, &[1, 2], &[], &mut Rng::new(1))
            .expect_err("n = 0 must be refused");
        assert!(err.starts_with("spec_verify(n=0):"));
    }

    #[test]
    fn generate_at_gate_zero_is_byte_identical_to_single_token_decode() {
        let trunk = trunk();
        let head_ck = head();
        let head = MtpHead::load(&trunk, &head_ck).unwrap();
        let config = MtpConfig {
            enabled: true,
            depth: 2,
            depth_set: true,
            depth_hi: 0,
            spec_n: 3,
            gate_p: 0.0,
        };
        let forward = Forward::new(&trunk);
        let engine = MtpEngine::new(&forward, head, config).unwrap();
        let info = fixture_info();
        let request = GenRequest {
            req: 7,
            max_tokens: 4,
            eos: vec![],
            ids: vec![1, 2],
            sample: None,
            progress: false,
        };
        let on = gen_response_with(&request, &info, &forward, Some(&engine));
        let off_forward = Forward::new(&trunk);
        let off = gen_response_with(&request, &info, &off_forward, None);
        let on_lines: Vec<&str> = on.lines().collect();
        let off_lines: Vec<&str> = off.lines().collect();
        // The four T lines are byte-identical (gate 0 ⇒ one token per
        // round, same seed, same rng stream).
        assert_eq!(&on_lines[..4], &off_lines[..4]);
        // The D line carries the speculative accounting: 4 rounds, 4
        // committed (fields [8]/[9], engine spec §8).
        let d_fields: Vec<&str> = on_lines[4].split_whitespace().collect();
        assert_eq!(d_fields[1], "7");
        assert_eq!(d_fields[2], "length");
        assert_eq!(d_fields[4], "4");
        assert_eq!(d_fields[7], "0");
        assert_eq!(d_fields[8], "4");
        assert_eq!(d_fields[9], "4");
        let off_fields: Vec<&str> = off_lines[4].split_whitespace().collect();
        assert_eq!(off_fields[8], "0");
        assert_eq!(off_fields[9], "0");
    }

    #[test]
    fn generate_at_gate_one_drafts_verifies_and_is_deterministic() {
        let trunk = trunk();
        let head_ck = head();
        let head = MtpHead::load(&trunk, &head_ck).unwrap();
        let config = MtpConfig {
            enabled: true,
            depth: 2,
            depth_set: true,
            depth_hi: 0,
            spec_n: 3,
            gate_p: 1.0,
        };
        let info = fixture_info();
        let request = GenRequest {
            req: 7,
            max_tokens: 6,
            eos: vec![],
            ids: vec![1, 2],
            sample: None,
            progress: false,
        };
        let forward = Forward::new(&trunk);
        let engine = MtpEngine::new(&forward, head.clone(), config.clone()).unwrap();
        let on = gen_response_with(&request, &info, &forward, Some(&engine));
        let lines: Vec<&str> = on.lines().collect();
        assert_eq!(lines.len(), 7, "6 T lines + the D line");
        let d_fields: Vec<&str> = lines[6].split_whitespace().collect();
        assert_eq!(d_fields[4], "6"); // n_gen
        assert_eq!(d_fields[8], "3"); // rounds: 6 tokens / 2 per round
        assert_eq!(d_fields[9], "6"); // commit
                                      // The first token matches the MTP-off decode (row 0 is the live
                                      // sequence's sample from the trunk logits).
        let off_forward = Forward::new(&trunk);
        let off = gen_response_with(&request, &info, &off_forward, None);
        assert_eq!(lines[0], off.lines().next().unwrap());
        // Determinism: a second engine on the same fixtures replays the
        // same T lines.
        let forward2 = Forward::new(&trunk);
        let engine2 = MtpEngine::new(&forward2, head, config).unwrap();
        let again = gen_response_with(&request, &info, &forward2, Some(&engine2));
        let first: Vec<&str> = on.lines().take(6).collect();
        let second: Vec<&str> = again.lines().take(6).collect();
        assert_eq!(first, second);
    }
}
