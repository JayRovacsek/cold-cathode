//! The typed tensor set the real forward path runs on (spec
//! `flash_serve-engine.md` §2 tensor inventory, §3 block forwards).
//!
//! [`load`] walks a loaded [`hgn::Checkpoint`], validates the full
//! `qwen4exp` tensor-name set (48 layers, both hyper-connection mixers per
//! layer, the `self_attn`/`linear_attn` block tensors, the MoE MLP and the
//! top-level `embed_tokens`/`lm_head`/`hyper_connection_mixer`), decodes
//! every payload through [`crate::bf16`] and [`crate::codecs`] (including
//! the quantised dtypes of the real `qwen38-flash-next-v2-abliterated.hgn`
//! checkpoint: `q4c`/`q8g64`/`ht`/`i4r`/`q6g64`, spec
//! `flash_serve-dtypes.md`) and derives the shape constants (hidden
//! width, stream count, head counts, expert count) from the entry dims —
//! the same derivation the spec's §2 head-configuration resolution uses
//! (e.g. `num_heads = q_proj.rows / 2 / head_dim` from the 24-vs-48
//! `o_proj` note).
//!
//! Checkpoints whose required tensors carry a dtype without a codec
//! (the integer codes `3`/`4` on weight entries) or a payload that fails
//! its dtype's layout validation yield `None` — the caller falls back to
//! the documented stub (module docs in [`crate::forward`]).
//!
//! The mini-checkpoint fixtures ([`mini_checkpoint`],
//! [`mini_checkpoint_quantised`]) are the test vehicles: the full
//! tensor-name set with shrunk dims (hidden 4, head dims 8, 4 experts of
//! width 8, mix rank 8, 48 layers), every payload bf16 with deterministic
//! pseudo-random values; the quantised variant swaps one tensor per
//! quantised dtype in (with its f16 rotation planes) at hidden 64, the
//! smallest width satisfying every codec's k-group rules.

use std::cmp::Ordering;
use std::collections::HashMap;

use crate::bf16;
use crate::codecs;
use crate::hgn;
use crate::ops::{self, Matrix};
use crate::sampling::Rng;

/// The RMSNorm epsilon used across the port (the reference kernels' value
/// is not exposed; 1e-6 is the Qwen convention).
pub const NORM_EPS: f32 = 1e-6;
/// RoPE base (Qwen convention, not pinned by the spec — reported as an
/// ambiguity).
pub const ROPE_THETA: f32 = 1_000_000.0;
/// Router top-k before clamping to `num_experts` (Qwen3-Next convention,
/// not pinned by the spec — reported as an ambiguity).
pub const DEFAULT_TOP_K: usize = 10;

/// The shape constants, derived from the entry dims (spec §2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dims {
    pub num_layers: usize,
    pub num_streams: usize,
    pub hidden_per_stream: usize,
    pub hidden_total: usize,
    pub mix_rank: usize,
    pub vocab: usize,
    pub num_experts: usize,
    pub expert_dim: usize,
    /// Router top-k (clamped to `num_experts`).
    pub top_k: usize,
}

/// A decoded tensor: its (non-zero) dims and weight storage (dense
/// values or a quantised payload, per [`codecs::Weights`]).
#[derive(Debug, Clone)]
struct Tensor {
    dims: Vec<u64>,
    weights: codecs::Weights,
}

impl Tensor {
    /// The values as a row-major matrix: 2-D tensors directly, 3-D
    /// tensors `[a, b, c]` flattened to `[a·b, c]` (the stacked expert
    /// `gate_up_proj`/`down_proj` and the `[channels, 1, kernel]` conv1d
    /// entries, read as `[channels, kernel]`).
    fn matrix(&self) -> Option<Matrix> {
        match self.dims.as_slice() {
            [rows, cols] => {
                Matrix::with_weights(*rows as usize, *cols as usize, self.weights.clone())
            }
            [a, b, c] => {
                Matrix::with_weights((*a * *b) as usize, *c as usize, self.weights.clone())
            }
            _ => None,
        }
    }

    /// The values of a 1-D tensor (norm gains, `A_log`, `dt_bias`, …).
    fn vector(&self) -> Option<Vec<f32>> {
        (self.dims.len() == 1).then(|| self.weights.row(self.dims[0] as usize, 0))
    }

    /// Every value of the tensor materialised (used where the forward
    /// path needs the raw values rather than a matrix, e.g. the conv1d
    /// kernel weights).
    fn values(&self) -> Option<Vec<f32>> {
        let (rows, cols) = codecs::shape(&self.dims)?;
        Some(self.weights.dequant_all(rows, cols))
    }
}

/// One hyper-connection mixer: `hc_norm` + the low-rank `down`/`up` gate
/// and (per layer) the `block_inject_weight` stream weights (spec §3
/// `FUN_01f8c690`; the exact mixer formula is a documented assumption —
/// the decompile is an open sub-gap).
#[derive(Debug, Clone)]
pub struct Mixer {
    pub norm: Vec<f32>,
    pub down: Matrix,
    pub up: Matrix,
    pub inject: Option<Matrix>,
    pub streams: usize,
    pub hidden_per_stream: usize,
}

impl Mixer {
    /// Pre-mix: normalise the stream stack, gate it through the low-rank
    /// `down`/`up` pair and combine the gated streams into the block
    /// input (`Σ_s inject[s] · scaled_s`, or the stream mean when the
    /// mixer has no `block_inject_weight` — the top-level mixer). Returns
    /// `(block_input, scaled, inject)` for [`Mixer::apply_block`].
    pub fn mix(&self, streams_data: &[f32]) -> (Vec<f32>, Vec<f32>, Option<Vec<f32>>) {
        debug_assert_eq!(streams_data.len(), self.norm.len(), "stream stack width");
        let normed = ops::rmsnorm(streams_data, &self.norm, NORM_EPS);
        let gate = self.up.gemv(&self.down.gemv(&normed));
        let scaled: Vec<f32> = normed.iter().zip(&gate).map(|(v, g)| v * g).collect();
        let inject = self.inject.as_ref().map(|m| m.gemv(&normed));
        let h = self.hidden_per_stream;
        let block_input: Vec<f32> = match &inject {
            Some(inject) => (0..h)
                .map(|i| {
                    (0..self.streams)
                        .map(|s| inject[s] * scaled[s * h + i])
                        .sum()
                })
                .collect(),
            None => (0..h)
                .map(|i| {
                    (0..self.streams).map(|s| scaled[s * h + i]).sum::<f32>() / self.streams as f32
                })
                .collect(),
        };
        (block_input, scaled, inject)
    }

    /// Post-mix: fold the block output back into every stream
    /// (`new_state[s] = scaled_s + inject[s] · block_out`; weight 1 for a
    /// mixer without `block_inject_weight`).
    pub fn apply_block(
        &self,
        scaled: &[f32],
        inject: Option<&[f32]>,
        block_out: &[f32],
        streams_data: &mut [f32],
    ) {
        debug_assert_eq!(streams_data.len(), self.norm.len(), "stream stack width");
        let h = self.hidden_per_stream;
        for s in 0..self.streams {
            let weight = inject.map_or(1.0, |v| v[s]);
            for i in 0..h {
                streams_data[s * h + i] = scaled[s * h + i] + weight * block_out[i];
            }
        }
    }
}

/// The full-attention block (`self_attn`, spec §3 `FUN_01fb3670`).
#[derive(Debug, Clone)]
pub struct AttnBlock {
    pub q: Matrix,
    pub q_norm: Vec<f32>,
    pub k: Matrix,
    pub k_norm: Vec<f32>,
    pub v: Matrix,
    pub o: Matrix,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

/// The gated linear-attention block (`linear_attn`, spec §3 `FUN_01fa0700`
/// — Gated DeltaNet).
#[derive(Debug, Clone)]
pub struct LinearBlock {
    pub qkv: Matrix,
    pub z: Matrix,
    pub a: Matrix,
    pub b: Matrix,
    pub a_log: Vec<f32>,
    pub dt_bias: Vec<f32>,
    /// The depthwise conv1d weights `[channels, kernel]` (always bf16 in
    /// the real file and the fixtures).
    pub conv_w: Vec<f32>,
    pub conv_kernel: usize,
    pub norm: Vec<f32>,
    pub out: Matrix,
    pub q_heads: usize,
    pub k_heads: usize,
    pub v_heads: usize,
    pub head_dim: usize,
    pub v_head_dim: usize,
}

/// The MoE MLP (`mlp`, spec §3 `FUN_01f9f8c0`; the n=1 decode path uses
/// the same math as the small-batch variant `FUN_01f98d30`).
#[derive(Debug, Clone)]
pub struct Mlp {
    pub gate: Matrix,
    pub gate_up: Matrix,
    pub down: Matrix,
    pub shared_gate_proj: Matrix,
    pub shared_up_proj: Matrix,
    pub shared_down_proj: Matrix,
    pub shared_router: Vec<f32>,
    pub ffn: usize,
    pub top_k: usize,
}

/// One decoder layer: pre-mix → block → MLP with its own post-mix, per
/// spec §3 (`FUN_01ffe2e0`: `attn_hyper_connection` mix → block dispatch
/// → `mlp_hyper_connection` mix → MLP).
#[derive(Debug, Clone)]
pub struct Layer {
    pub attn_mixer: Mixer,
    pub block: Block,
    pub mlp_mixer: Mixer,
    pub mlp: Mlp,
}

#[derive(Debug, Clone)]
pub enum Block {
    Attention(AttnBlock),
    Linear(LinearBlock),
}

/// The whole model: the tensor set plus the derived shapes.
#[derive(Debug, Clone)]
pub struct Model {
    pub dims: Dims,
    pub embed: Matrix,
    pub lm_head: Matrix,
    /// The top-level `hyper_connection_mixer` (no `block_inject_weight`
    /// in the real file): used as the final aggregation before `lm_head`.
    pub final_mixer: Mixer,
    pub layers: Vec<Layer>,
}

/// Load the model from a checkpoint, or `None` when the checkpoint lacks
/// the full tensor set (module docs).
pub fn load(checkpoint: &hgn::Checkpoint) -> Option<Model> {
    let mut by_name: HashMap<&str, usize> = HashMap::new();
    for (index, entry) in checkpoint.entries.iter().enumerate() {
        by_name.entry(entry.name()).or_insert(index);
    }
    let tensor = |name: &str| -> Option<Tensor> { load_tensor(checkpoint, &by_name, name) };

    // Layer count: consecutive `layers.N.attn_hyper_connection.hc_norm`
    // from 0 (the real file: 48).
    let num_layers = (0..)
        .map_while(|n| {
            by_name
                .contains_key(format!("layers.{n}.attn_hyper_connection.hc_norm.weight").as_str())
                .then_some(n)
        })
        .count();
    if num_layers == 0 {
        return None;
    }

    // Shapes come from layer 0's mixer + the embedding (spec §2).
    let hc_norm = tensor("layers.0.attn_hyper_connection.hc_norm.weight")?;
    let block_inject = tensor("layers.0.attn_hyper_connection.block_inject_weight.weight")?;
    let hidden_total = hc_norm.dims[0] as usize;
    let num_streams = block_inject.dims[0] as usize;
    if num_streams == 0 || !hidden_total.is_multiple_of(num_streams) {
        return None;
    }
    let hidden_per_stream = hidden_total / num_streams;
    let embed = tensor("embed_tokens.weight")?;
    if embed.dims.len() != 2 || embed.dims[1] as usize != hidden_per_stream {
        return None;
    }
    let vocab = embed.dims[0] as usize;
    let lm_head = tensor("lm_head.weight")?;
    if lm_head.dims.len() != 2
        || lm_head.dims[0] as usize != vocab
        || lm_head.dims[1] as usize != hidden_per_stream
    {
        return None;
    }
    let final_mixer = load_mixer(
        &tensor,
        "hyper_connection_mixer",
        num_streams,
        hidden_per_stream,
    )?;
    let mix_rank = final_mixer.down.rows;

    let mut layers = Vec::with_capacity(num_layers);
    for n in 0..num_layers {
        let prefix = format!("layers.{n}");
        let attn_mixer = load_mixer(
            &tensor,
            &format!("{prefix}.attn_hyper_connection"),
            num_streams,
            hidden_per_stream,
        )?;
        let mlp_mixer = load_mixer(
            &tensor,
            &format!("{prefix}.mlp_hyper_connection"),
            num_streams,
            hidden_per_stream,
        )?;
        let block = if by_name.contains_key(format!("{prefix}.self_attn.q_proj.weight").as_str()) {
            Block::Attention(load_attn(&tensor, &prefix, hidden_per_stream)?)
        } else if by_name.contains_key(format!("{prefix}.linear_attn.in_proj_qkv.weight").as_str())
        {
            Block::Linear(load_linear(&tensor, &prefix, hidden_per_stream)?)
        } else {
            return None;
        };
        let mlp = load_mlp(&tensor, &prefix, hidden_per_stream)?;
        layers.push(Layer {
            attn_mixer,
            block,
            mlp_mixer,
            mlp,
        });
    }

    let num_experts = layers[0].mlp.gate.rows;
    let expert_dim = layers[0].mlp.ffn;
    Some(Model {
        dims: Dims {
            num_layers,
            num_streams,
            hidden_per_stream,
            hidden_total,
            mix_rank,
            vocab,
            num_experts,
            expert_dim,
            top_k: DEFAULT_TOP_K.min(num_experts),
        },
        embed: embed.matrix()?,
        lm_head: lm_head.matrix()?,
        final_mixer,
        layers,
    })
}

/// Load one tensor by name: decode its payload through the dtype codecs
/// (resolving the f16 rotation planes for `ht`/`i4r` tensors per
/// [`plane_names`]) and validate it against the entry's dims.
fn load_tensor(
    checkpoint: &hgn::Checkpoint,
    by_name: &HashMap<&str, usize>,
    name: &str,
) -> Option<Tensor> {
    let &index = by_name.get(name)?;
    let entry = &checkpoint.entries[index];
    let bytes = checkpoint.tensor_bytes(index)?;
    let dims: Vec<u64> = entry.dims().into_iter().filter(|d| *d != 0).collect();
    let planes = resolve_planes(checkpoint, by_name, entry.dtype(), name)?;
    let weights = codecs::Weights::from_dtype(
        entry.dtype(),
        bytes,
        &dims,
        planes.as_ref().map(|(suh, svh)| (&suh[..], &svh[..])),
    )?;
    Some(Tensor { dims, weights })
}

/// The decoded `(suh, svh)` planes a quantised weight needs: `Some(None)`
/// for dtypes without planes, `Some(Some(..))` when an `ht`/`i4r`
/// tensor's planes resolve, `None` when they are missing or malformed
/// (the engine's `an HT linear without its rotation planes` failure,
/// spec §2).
fn resolve_planes(
    checkpoint: &hgn::Checkpoint,
    by_name: &HashMap<&str, usize>,
    dtype: u32,
    name: &str,
) -> Option<Option<(Vec<f32>, Vec<f32>)>> {
    if !matches!(dtype, 16 | 23) {
        return Some(None);
    }
    let (suh_names, svh_name) = plane_names(name)?;
    // The own `<base>.suh` wins when present and must be f16: an own
    // plane of the wrong dtype rejects the tensor rather than silently
    // falling through to the shared `<parent>.su` (the shared name only
    // applies when the own one is absent — the fused expert sets,
    // spec §7).
    let suh = if by_name.contains_key(suh_names[0].as_str()) {
        load_f16_plane(checkpoint, by_name, &suh_names[0])?
    } else {
        suh_names
            .get(1)
            .and_then(|shared| load_f16_plane(checkpoint, by_name, shared))?
    };
    let svh = load_f16_plane(checkpoint, by_name, &svh_name)?;
    Some(Some((suh, svh)))
}

/// Load one f16 plane tensor (spec §2: the planes `must be f16`).
fn load_f16_plane(
    checkpoint: &hgn::Checkpoint,
    by_name: &HashMap<&str, usize>,
    name: &str,
) -> Option<Vec<f32>> {
    let &index = by_name.get(name)?;
    let entry = &checkpoint.entries[index];
    if entry.dtype() != 2 {
        return None;
    }
    bf16::decode_float_payload(2, checkpoint.tensor_bytes(index)?)
}

/// The companion plane names for a quantised `<base>.weight` (spec §2
/// naming rule): the `suh` candidates — the own `<base>.suh`, else the
/// shared `<parent>.su` used by the fused expert sets (spec §7) — and
/// the `svh` name `<base>.svh`.
fn plane_names(name: &str) -> Option<(Vec<String>, String)> {
    let base = name.strip_suffix(".weight")?;
    let mut suh = vec![format!("{base}.suh")];
    if let Some((parent, _)) = base.rsplit_once('.') {
        let shared = format!("{parent}.su");
        if shared != suh[0] {
            suh.push(shared);
        }
    }
    Some((suh, format!("{base}.svh")))
}

/// Load one mixer set (`hc_norm` + `down`/`up` + optional
/// `block_inject_weight`) for a given stream layout.
fn load_mixer(
    tensor: &dyn Fn(&str) -> Option<Tensor>,
    prefix: &str,
    streams: usize,
    hidden_per_stream: usize,
) -> Option<Mixer> {
    let norm = tensor(&format!("{prefix}.hc_norm.weight"))?.vector()?;
    if norm.len() != streams * hidden_per_stream {
        return None;
    }
    let inject = match tensor(&format!("{prefix}.block_inject_weight.weight")) {
        Some(t) => {
            let m = t.matrix()?;
            if m.rows != streams || m.cols != norm.len() {
                return None;
            }
            Some(m)
        }
        None => None,
    };
    let down = tensor(&format!("{prefix}.input_mix_weight_down.weight"))?.matrix()?;
    let up = tensor(&format!("{prefix}.input_mix_weight_up.weight"))?.matrix()?;
    if down.cols != norm.len() || up.rows != norm.len() || up.cols != down.rows {
        return None;
    }
    Some(Mixer {
        norm,
        down,
        up,
        inject,
        streams,
        hidden_per_stream,
    })
}

/// Load and validate the `self_attn` block (spec §2 head resolution:
/// `head_dim = q_norm` width, `num_heads = q_proj.rows / 2 / head_dim` —
/// the 24-vs-48 `o_proj` nuance: the gemv computes all rows, the
/// rope/attention path consumes the first half — `num_kv_heads =
/// k_proj.rows / head_dim`).
fn load_attn(tensor: &dyn Fn(&str) -> Option<Tensor>, prefix: &str, h: usize) -> Option<AttnBlock> {
    let q = tensor(&format!("{prefix}.self_attn.q_proj.weight"))?.matrix()?;
    let q_norm = tensor(&format!("{prefix}.self_attn.q_norm.weight"))?.vector()?;
    let k = tensor(&format!("{prefix}.self_attn.k_proj.weight"))?.matrix()?;
    let k_norm = tensor(&format!("{prefix}.self_attn.k_norm.weight"))?.vector()?;
    let v = tensor(&format!("{prefix}.self_attn.v_proj.weight"))?.matrix()?;
    let o = tensor(&format!("{prefix}.self_attn.o_proj.weight"))?.matrix()?;
    // revisit when the second-half consumer gap closes
    if q.cols != h || k.cols != h || v.cols != h || o.cols != q.rows / 2 || o.rows != h {
        return None;
    }
    let head_dim = q_norm.len();
    let consumed = q.rows / 2;
    if head_dim == 0 || consumed % head_dim != 0 || k.rows % head_dim != 0 || v.rows % head_dim != 0
    {
        return None;
    }
    if k_norm.len() != head_dim {
        return None;
    }
    let q_heads = consumed / head_dim;
    let kv_heads = k.rows / head_dim;
    if v.rows / head_dim != kv_heads {
        return None;
    }
    // The QSA indexer tensors exist on every real attention layer; the
    // port loads them but the dense forward path does not consult them
    // (gated by HALOGEN_QSA_DENSE in the reference, spec §3 step 1).
    tensor(&format!("{prefix}.self_attn.indexer.index_qk_proj.weight"))?;
    tensor(&format!("{prefix}.self_attn.indexer.q_layernorm.weight"))?;
    tensor(&format!("{prefix}.self_attn.indexer.k_layernorm.weight"))?;
    Some(AttnBlock {
        q,
        q_norm,
        k,
        k_norm,
        v,
        o,
        q_heads,
        kv_heads,
        head_dim,
    })
}

/// Load and validate the `linear_attn` block (spec §3 `FUN_01fa0700`:
/// packed `in_proj_qkv` with `v = in_proj_z` width, `q`/`k` from the
/// remainder — the head split is the registered sub-gap, implemented as
/// equal q/k heads; `A_log`/`dt_bias` per v-head; `conv1d` over the
/// packed qkv; per-head `norm` + `out_proj`).
fn load_linear(
    tensor: &dyn Fn(&str) -> Option<Tensor>,
    prefix: &str,
    h: usize,
) -> Option<LinearBlock> {
    let qkv = tensor(&format!("{prefix}.linear_attn.in_proj_qkv.weight"))?.matrix()?;
    let z = tensor(&format!("{prefix}.linear_attn.in_proj_z.weight"))?.matrix()?;
    let a = tensor(&format!("{prefix}.linear_attn.in_proj_a.weight"))?.matrix()?;
    let b = tensor(&format!("{prefix}.linear_attn.in_proj_b.weight"))?.matrix()?;
    let a_log = tensor(&format!("{prefix}.linear_attn.A_log"))?.vector()?;
    let dt_bias = tensor(&format!("{prefix}.linear_attn.dt_bias"))?.vector()?;
    let norm = tensor(&format!("{prefix}.linear_attn.norm.weight"))?.vector()?;
    let out = tensor(&format!("{prefix}.linear_attn.out_proj.weight"))?.matrix()?;
    let conv = tensor(&format!("{prefix}.linear_attn.conv1d.weight"))?;
    if qkv.cols != h
        || z.cols != h
        || a.cols != h
        || b.cols != h
        || out.rows != h
        || out.cols != z.rows
    {
        return None;
    }
    let head_dim = norm.len();
    let v_heads = z.rows.checked_div(head_dim)?;
    let qk_rows = qkv.rows.checked_sub(z.rows)?;
    if head_dim == 0 || qk_rows % (2 * head_dim) != 0 || a.rows != v_heads || b.rows != v_heads {
        return None;
    }
    if a_log.len() != v_heads || dt_bias.len() != v_heads {
        return None;
    }
    // conv1d dims are `[channels, 1, kernel]` in the real file (kind 3);
    // the block keeps the values plus the kernel width (conv1d weights
    // are always bf16 in the real file and the fixtures).
    let conv_kernel = *conv.dims.last()?;
    let conv_w = conv.values()?;
    if conv_w.len() != qkv.rows * conv_kernel as usize {
        return None;
    }
    let conv_kernel = conv_kernel as usize;
    Some(LinearBlock {
        qkv,
        z,
        a,
        b,
        a_log,
        dt_bias,
        conv_w,
        conv_kernel,
        norm,
        out,
        q_heads: qk_rows / (2 * head_dim),
        k_heads: qk_rows / (2 * head_dim),
        v_heads,
        head_dim,
        v_head_dim: head_dim,
    })
}

/// Load and validate the MoE MLP (`mlp.gate` router, the stacked expert
/// `gate_up_proj`/`down_proj`, the gated shared expert).
fn load_mlp(tensor: &dyn Fn(&str) -> Option<Tensor>, prefix: &str, h: usize) -> Option<Mlp> {
    let gate = tensor(&format!("{prefix}.mlp.gate.weight"))?.matrix()?;
    let gate_up = tensor(&format!("{prefix}.mlp.experts.gate_up_proj.weight"))?.matrix()?;
    let down = tensor(&format!("{prefix}.mlp.experts.down_proj.weight"))?.matrix()?;
    let shared_gate_proj =
        tensor(&format!("{prefix}.mlp.shared_expert.gate_proj.weight"))?.matrix()?;
    let shared_up_proj = tensor(&format!("{prefix}.mlp.shared_expert.up_proj.weight"))?.matrix()?;
    let shared_down_proj =
        tensor(&format!("{prefix}.mlp.shared_expert.down_proj.weight"))?.matrix()?;
    let shared_router = tensor(&format!("{prefix}.mlp.shared_expert_gate.weight"))?.matrix()?;
    let num_experts = gate.rows;
    if gate.cols != h || shared_router.rows != 1 || shared_router.cols != h {
        return None;
    }
    if gate_up.cols != h || gate_up.rows % num_experts != 0 || (gate_up.rows / num_experts) % 2 != 0
    {
        return None;
    }
    let ffn = gate_up.rows / num_experts / 2;
    if down.cols != ffn || down.rows != num_experts * h {
        return None;
    }
    // The shared expert's FFN width is its own (the real file's shared
    // expert is narrower than the routed experts); only `ffn` belongs to
    // the routed set.
    let shared_ffn = shared_gate_proj.rows;
    if shared_gate_proj.cols != h || shared_up_proj.rows != shared_ffn {
        return None;
    }
    if shared_down_proj.rows != h || shared_down_proj.cols != shared_ffn {
        return None;
    }
    Some(Mlp {
        gate,
        gate_up,
        down,
        shared_gate_proj,
        shared_up_proj,
        shared_down_proj,
        shared_router: shared_router.row(0),
        ffn,
        top_k: DEFAULT_TOP_K.min(num_experts),
    })
}

impl Mlp {
    /// One expert's output: `down_e · (silu(gate_e·x) ⊙ up_e·x)`.
    pub fn expert_out(&self, expert: usize, x: &[f32]) -> Vec<f32> {
        let h = self.down.rows / self.gate.rows;
        let gate = self.gate_up.gemv_rows(expert * 2 * self.ffn, self.ffn, x);
        let up = self
            .gate_up
            .gemv_rows(expert * 2 * self.ffn + self.ffn, self.ffn, x);
        let act: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| ops::silu(*g) * *u)
            .collect();
        self.down.gemv_rows(expert * h, h, &act)
    }

    /// The MoE forward for one token: softmax over all router logits with
    /// the top-k normalised weights (the Qwen3-Next convention), then the
    /// expert sum plus the sigmoid-gated shared expert (spec §3
    /// `FUN_01f9f8c0`; the n=1 decode path treats the small-batch variant
    /// `FUN_01f98d30` as the same math).
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let h = self.down.rows / self.gate.rows;
        let mut probs = self.gate.gemv(x);
        ops::softmax(&mut probs);
        let mut ranked: Vec<(f32, usize)> =
            probs.into_iter().enumerate().map(|(i, p)| (p, i)).collect();
        ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        let selected: Vec<(f32, usize)> = ranked.into_iter().take(self.top_k).collect();
        let mut out = vec![0.0; h];
        for (weight, expert) in selected {
            let expert_out = self.expert_out(expert, x);
            for (o, value) in out.iter_mut().zip(&expert_out) {
                *o += weight * *value;
            }
        }
        let shared_gate = self.shared_gate_proj.gemv(x);
        let shared_up = self.shared_up_proj.gemv(x);
        let act: Vec<f32> = shared_gate
            .iter()
            .zip(&shared_up)
            .map(|(g, u)| ops::silu(*g) * *u)
            .collect();
        let shared = self.shared_down_proj.gemv(&act);
        let gate = ops::sigmoid(ops::dot(&self.shared_router, x));
        for (o, value) in out.iter_mut().zip(&shared) {
            *o += gate * *value;
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Mini-checkpoint fixture (module docs)
// ---------------------------------------------------------------------------

/// Mini-fixture shape constants.
pub const MINI_LAYERS: usize = 48;
pub const MINI_STREAMS: usize = 4;
pub const MINI_HIDDEN: usize = 4;
pub const MINI_VOCAB: usize = 6;
pub const MINI_EXPERTS: usize = 4;
pub const MINI_FFN: usize = 8;
pub const MINI_MIX_RANK: usize = 8;
pub const MINI_HEAD_DIM: usize = 8;
/// Hidden width of the quantised fixture ([`mini_checkpoint_quantised`]):
/// the smallest value that satisfies every codec's k-group rules (q8g64,
/// q6g64 and i4r need `k % 64 == 0`, q4c needs `k % 32 == 0`).
pub const MINI_HIDDEN_QUANTISED: usize = 64;
/// Attention layers sit every 4th index (3, 7, …, 47) — the real file's
/// pattern (spec §2).
pub fn mini_is_attention_layer(layer: usize) -> bool {
    layer % 4 == 3
}

/// One fixture entry: name, kind, dims and dtype code.
#[derive(Debug, Clone)]
struct FixtureSpec {
    name: String,
    kind: u32,
    dims: Vec<u64>,
    dtype: u32,
}

/// Build the full tensor-name set (48 layers, both mixers, the block
/// tensors for the layer kind, the MoE MLP, the layer-1 PLE extras and the
/// top-level tensors) with the given hidden width, every spec bf16.
fn mini_specs(hidden: usize) -> Vec<FixtureSpec> {
    // (name, kind, dims) — the top-level tensors seed the list.
    let tuples: Vec<(String, u32, Vec<u64>)> = vec![
        (
            "embed_tokens.weight".to_string(),
            2,
            vec![MINI_VOCAB as u64, hidden as u64],
        ),
        (
            "lm_head.weight".to_string(),
            2,
            vec![MINI_VOCAB as u64, hidden as u64],
        ),
        (
            "hyper_connection_mixer.hc_norm.weight".to_string(),
            1,
            vec![(MINI_STREAMS * hidden) as u64],
        ),
        (
            "hyper_connection_mixer.input_mix_weight_down.weight".to_string(),
            2,
            vec![MINI_MIX_RANK as u64, (MINI_STREAMS * hidden) as u64],
        ),
        (
            "hyper_connection_mixer.input_mix_weight_up.weight".to_string(),
            2,
            vec![(MINI_STREAMS * hidden) as u64, MINI_MIX_RANK as u64],
        ),
    ];
    let mut specs: Vec<FixtureSpec> = tuples
        .into_iter()
        .map(|(name, kind, dims)| FixtureSpec {
            name,
            kind,
            dims,
            dtype: 0,
        })
        .collect();
    for layer in 0..MINI_LAYERS {
        let prefix = format!("layers.{layer}");
        for mixer in ["attn_hyper_connection", "mlp_hyper_connection"] {
            specs.push(FixtureSpec {
                name: format!("{prefix}.{mixer}.hc_norm.weight"),
                kind: 1,
                dims: vec![(MINI_STREAMS * hidden) as u64],
                dtype: 0,
            });
            specs.push(FixtureSpec {
                name: format!("{prefix}.{mixer}.block_inject_weight.weight"),
                kind: 2,
                dims: vec![MINI_STREAMS as u64, (MINI_STREAMS * hidden) as u64],
                dtype: 0,
            });
            specs.push(FixtureSpec {
                name: format!("{prefix}.{mixer}.input_mix_weight_down.weight"),
                kind: 2,
                dims: vec![MINI_MIX_RANK as u64, (MINI_STREAMS * hidden) as u64],
                dtype: 0,
            });
            specs.push(FixtureSpec {
                name: format!("{prefix}.{mixer}.input_mix_weight_up.weight"),
                kind: 2,
                dims: vec![(MINI_STREAMS * hidden) as u64, MINI_MIX_RANK as u64],
                dtype: 0,
            });
        }
        if mini_is_attention_layer(layer) {
            // q_proj exposes 2·head_dim rows; the forward consumes half
            // (the 24-vs-48 nuance, spec §2) → 1 query head.
            specs.push(FixtureSpec {
                name: format!("{prefix}.self_attn.q_proj.weight"),
                kind: 2,
                dims: vec![(2 * MINI_HEAD_DIM) as u64, hidden as u64],
                dtype: 0,
            });
            specs.push(FixtureSpec {
                name: format!("{prefix}.self_attn.q_norm.weight"),
                kind: 1,
                dims: vec![MINI_HEAD_DIM as u64],
                dtype: 0,
            });
            specs.push(FixtureSpec {
                name: format!("{prefix}.self_attn.k_proj.weight"),
                kind: 2,
                dims: vec![MINI_HEAD_DIM as u64, hidden as u64],
                dtype: 0,
            });
            specs.push(FixtureSpec {
                name: format!("{prefix}.self_attn.k_norm.weight"),
                kind: 1,
                dims: vec![MINI_HEAD_DIM as u64],
                dtype: 0,
            });
            specs.push(FixtureSpec {
                name: format!("{prefix}.self_attn.v_proj.weight"),
                kind: 2,
                dims: vec![MINI_HEAD_DIM as u64, hidden as u64],
                dtype: 0,
            });
            specs.push(FixtureSpec {
                name: format!("{prefix}.self_attn.o_proj.weight"),
                kind: 2,
                dims: vec![hidden as u64, MINI_HEAD_DIM as u64],
                dtype: 0,
            });
            specs.push(FixtureSpec {
                name: format!("{prefix}.self_attn.indexer.index_qk_proj.weight"),
                kind: 2,
                dims: vec![MINI_HEAD_DIM as u64, hidden as u64],
                dtype: 0,
            });
            specs.push(FixtureSpec {
                name: format!("{prefix}.self_attn.indexer.q_layernorm.weight"),
                kind: 1,
                dims: vec![MINI_HEAD_DIM as u64],
                dtype: 0,
            });
            specs.push(FixtureSpec {
                name: format!("{prefix}.self_attn.indexer.k_layernorm.weight"),
                kind: 1,
                dims: vec![MINI_HEAD_DIM as u64],
                dtype: 0,
            });
        } else {
            // Packed qkv: 1 q + 1 k head (8 each) + 1 v head (8) = 24 rows,
            // matching the real `[10240, 2560]` = q 2048 + k 2048 + v 6144
            // layout at mini scale (spec §3 `FUN_01fa0700`).
            specs.push(FixtureSpec {
                name: format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                kind: 2,
                dims: vec![(3 * MINI_HEAD_DIM) as u64, hidden as u64],
                dtype: 0,
            });
            specs.push(FixtureSpec {
                name: format!("{prefix}.linear_attn.in_proj_z.weight"),
                kind: 2,
                dims: vec![MINI_HEAD_DIM as u64, hidden as u64],
                dtype: 0,
            });
            specs.push(FixtureSpec {
                name: format!("{prefix}.linear_attn.in_proj_a.weight"),
                kind: 2,
                dims: vec![1, hidden as u64],
                dtype: 0,
            });
            specs.push(FixtureSpec {
                name: format!("{prefix}.linear_attn.in_proj_b.weight"),
                kind: 2,
                dims: vec![1, hidden as u64],
                dtype: 0,
            });
            specs.push(FixtureSpec {
                name: format!("{prefix}.linear_attn.A_log"),
                kind: 1,
                dims: vec![1],
                dtype: 0,
            });
            specs.push(FixtureSpec {
                name: format!("{prefix}.linear_attn.dt_bias"),
                kind: 1,
                dims: vec![1],
                dtype: 0,
            });
            specs.push(FixtureSpec {
                name: format!("{prefix}.linear_attn.conv1d.weight"),
                kind: 3,
                dims: vec![(3 * MINI_HEAD_DIM) as u64, 1, 4],
                dtype: 0,
            });
            specs.push(FixtureSpec {
                name: format!("{prefix}.linear_attn.norm.weight"),
                kind: 1,
                dims: vec![MINI_HEAD_DIM as u64],
                dtype: 0,
            });
            specs.push(FixtureSpec {
                name: format!("{prefix}.linear_attn.out_proj.weight"),
                kind: 2,
                dims: vec![hidden as u64, MINI_HEAD_DIM as u64],
                dtype: 0,
            });
        }
        specs.push(FixtureSpec {
            name: format!("{prefix}.mlp.gate.weight"),
            kind: 2,
            dims: vec![MINI_EXPERTS as u64, hidden as u64],
            dtype: 0,
        });
        specs.push(FixtureSpec {
            name: format!("{prefix}.mlp.experts.gate_up_proj.weight"),
            kind: 3,
            dims: vec![MINI_EXPERTS as u64, (2 * MINI_FFN) as u64, hidden as u64],
            dtype: 0,
        });
        specs.push(FixtureSpec {
            name: format!("{prefix}.mlp.experts.down_proj.weight"),
            kind: 3,
            dims: vec![MINI_EXPERTS as u64, hidden as u64, MINI_FFN as u64],
            dtype: 0,
        });
        specs.push(FixtureSpec {
            name: format!("{prefix}.mlp.shared_expert.gate_proj.weight"),
            kind: 2,
            dims: vec![MINI_FFN as u64, hidden as u64],
            dtype: 0,
        });
        specs.push(FixtureSpec {
            name: format!("{prefix}.mlp.shared_expert.up_proj.weight"),
            kind: 2,
            dims: vec![MINI_FFN as u64, hidden as u64],
            dtype: 0,
        });
        specs.push(FixtureSpec {
            name: format!("{prefix}.mlp.shared_expert.down_proj.weight"),
            kind: 2,
            dims: vec![hidden as u64, MINI_FFN as u64],
            dtype: 0,
        });
        specs.push(FixtureSpec {
            name: format!("{prefix}.mlp.shared_expert_gate.weight"),
            kind: 2,
            dims: vec![1, hidden as u64],
            dtype: 0,
        });
        // PLE exists only on layer 1 in the real file; the forward path
        // skips it (out of scope), but the fixture keeps the name set.
        if layer == 1 {
            specs.push(FixtureSpec {
                name: format!("{prefix}.ple.conv1d.weight"),
                kind: 3,
                dims: vec![(MINI_STREAMS * hidden) as u64, 1, 4],
                dtype: 0,
            });
            specs.push(FixtureSpec {
                name: format!("{prefix}.ple.key_proj.weight"),
                kind: 2,
                dims: vec![(MINI_STREAMS * hidden) as u64, hidden as u64],
                dtype: 0,
            });
            specs.push(FixtureSpec {
                name: format!("{prefix}.ple.norm_conv.weight"),
                kind: 1,
                dims: vec![(MINI_STREAMS * hidden) as u64],
                dtype: 0,
            });
            specs.push(FixtureSpec {
                name: format!("{prefix}.ple.norm_key.weight"),
                kind: 1,
                dims: vec![(MINI_STREAMS * hidden) as u64],
                dtype: 0,
            });
            specs.push(FixtureSpec {
                name: format!("{prefix}.ple.norm_query.weight"),
                kind: 1,
                dims: vec![(MINI_STREAMS * hidden) as u64],
                dtype: 0,
            });
            specs.push(FixtureSpec {
                name: format!("{prefix}.ple.ple_embedding.layer_multipliers"),
                kind: 1,
                dims: vec![3],
                dtype: 0,
            });
            specs.push(FixtureSpec {
                name: format!("{prefix}.ple.ple_embedding.ngram_heads_offsets"),
                kind: 1,
                dims: vec![16],
                dtype: 0,
            });
            specs.push(FixtureSpec {
                name: format!("{prefix}.ple.ple_embedding.ngram_heads_vocab_sizes"),
                kind: 1,
                dims: vec![16],
                dtype: 0,
            });
            specs.push(FixtureSpec {
                name: format!("{prefix}.ple.value_proj.weight"),
                kind: 2,
                dims: vec![hidden as u64, hidden as u64],
                dtype: 0,
            });
        }
    }
    specs
}

/// Build the full mini-checkpoint (module docs): the full tensor-name set
/// with mini dims, every payload bf16 with deterministic values, and
/// return the parsed checkpoint.
pub fn mini_checkpoint() -> hgn::Checkpoint {
    build_fixture("ccfs-mini-fixture.hgn", &mini_specs(MINI_HIDDEN))
}

/// The mini fixture with quantised payloads on every quantised dtype the
/// real checkpoint ships (module docs): `embed_tokens` q4c, `lm_head` ht
/// (with its f16 planes), layer 0's `attn_hyper_connection.block_inject_weight`
/// q6g64, layer 0's `mlp.gate` q8g64 and layer 0's
/// `mlp.experts.gate_up_proj` i4r (with the shared `experts.su` and its
/// own `svh`), at hidden [`MINI_HIDDEN_QUANTISED`].
pub fn mini_checkpoint_quantised() -> hgn::Checkpoint {
    let mut specs = mini_specs(MINI_HIDDEN_QUANTISED);
    for (name, dtype) in [
        ("embed_tokens.weight", 5),
        ("lm_head.weight", 16),
        (
            "layers.0.attn_hyper_connection.block_inject_weight.weight",
            24,
        ),
        ("layers.0.mlp.gate.weight", 7),
        ("layers.0.mlp.experts.gate_up_proj.weight", 23),
    ] {
        specs
            .iter_mut()
            .find(|spec| spec.name == name)
            .expect("quantised fixture overrides an existing spec")
            .dtype = dtype;
    }
    // The f16 rotation planes the ht/i4r tensors need (spec §2 naming
    // rule via `plane_names`): lm_head's own pair, and the shared
    // `experts.su` plus the own `gate_up_proj.svh` (spec §7 — the fused
    // expert set ships the shared `su`, not a per-projection `suh`).
    let (lm_suh, lm_svh) = plane_names("lm_head.weight").expect("lm_head planes");
    specs.push(FixtureSpec {
        name: lm_suh[0].clone(),
        kind: 1,
        dims: vec![MINI_HIDDEN_QUANTISED as u64],
        dtype: 2,
    });
    specs.push(FixtureSpec {
        name: lm_svh,
        kind: 1,
        dims: vec![MINI_VOCAB as u64],
        dtype: 2,
    });
    let (gu_suh, gu_svh) =
        plane_names("layers.0.mlp.experts.gate_up_proj.weight").expect("gate_up_proj planes");
    specs.push(FixtureSpec {
        name: gu_suh[1].clone(),
        kind: 1,
        dims: vec![MINI_HIDDEN_QUANTISED as u64],
        dtype: 2,
    });
    specs.push(FixtureSpec {
        name: gu_svh,
        kind: 1,
        dims: vec![(2 * MINI_FFN) as u64],
        dtype: 2,
    });
    build_fixture("ccfs-mini-quantised.hgn", &specs)
}

/// Serialise header + entries + payloads (layout per `hgn::TensorEntry`)
/// and parse the bytes back into a checkpoint.
fn build_fixture(file: &str, specs: &[FixtureSpec]) -> hgn::Checkpoint {
    let entry_count = specs.len() as u64;
    let table_offset = hgn::HEADER_BYTES as u64;
    // The engine requires 16 B-aligned payloads: the data section starts
    // on a 16 B boundary and every stored payload is padded up to one
    // (the entry's nbytes stays the exact payload length, the padding is
    // filler), so each entry's data_offset is 16 B-aligned.
    let data_offset =
        codecs::pad16(table_offset as usize + entry_count as usize * hgn::ENTRY_BYTES) as u64;
    let mut payloads: Vec<u8> = Vec::new();
    let mut entries: Vec<u8> = Vec::new();
    for spec in specs {
        let values = fixture_values(&spec.name, &spec.dims);
        let payload = fixture_payload(spec, &values);
        let offset = data_offset + payloads.len() as u64;
        let mut entry = [0u8; hgn::ENTRY_BYTES];
        entry[..spec.name.len()].copy_from_slice(spec.name.as_bytes());
        entry[96..100].copy_from_slice(&spec.dtype.to_le_bytes());
        entry[100..104].copy_from_slice(&spec.kind.to_le_bytes());
        for (i, dim) in spec.dims.iter().take(4).enumerate() {
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
    // Pad the gap between the table and the data section so the header's
    // data_offset matches the payloads' actual position (the raw table
    // end is 104 + n·160 ≡ 8 (mod 16)).
    bytes.resize(data_offset as usize, 0);
    bytes.extend_from_slice(&payloads);
    hgn::load_prefix(file, bytes, declared).expect("fixture loads")
}

/// The payload bytes for a spec's dtype: the float dtypes encode the
/// deterministic values, the quantised dtypes run the codec encoders.
fn fixture_payload(spec: &FixtureSpec, values: &[f32]) -> Vec<u8> {
    let (rows, cols) = codecs::shape(&spec.dims).expect("fixture dims");
    match spec.dtype {
        0 => bf16::encode_bf16(values),
        2 => bf16::encode_f16(values),
        5 => codecs::q4c_encode(values, rows, cols).expect("q4c fixture payload"),
        7 => codecs::q8g64_encode(values, rows, cols).expect("q8g64 fixture payload"),
        24 => codecs::q6g64_encode(values, rows, cols).expect("q6g64 fixture payload"),
        16 => codecs::ht_encode(values, rows, cols).expect("ht fixture payload"),
        23 => codecs::i4r_encode(values, rows, cols).expect("i4r fixture payload"),
        other => unimplemented!("fixture dtype {other}"),
    }
}

/// Deterministic pseudo-random payload values in `[-0.5, 0.5)`, seeded by
/// the tensor name so every fixture build is byte-identical.
fn fixture_values(name: &str, dims: &[u64]) -> Vec<f32> {
    let count = dims.iter().product::<u64>().max(1) as usize;
    let mut rng = Rng::from_u64(fnv(name.as_bytes()));
    (0..count).map(|_| rng.next_f32() - 0.5).collect()
}

/// FNV-1a over the name bytes (the reference's hash family, spec §5).
fn fnv(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mini_checkpoint_loads_the_full_tensor_set() {
        let ck = mini_checkpoint();
        assert_eq!(ck.entries.len(), 48 * 24 + 9 + 5);
        let model = load(&ck).expect("mini checkpoint loads a real model");
        assert_eq!(model.dims.num_layers, MINI_LAYERS);
        assert_eq!(model.dims.num_streams, MINI_STREAMS);
        assert_eq!(model.dims.hidden_per_stream, MINI_HIDDEN);
        assert_eq!(model.dims.hidden_total, MINI_STREAMS * MINI_HIDDEN);
        assert_eq!(model.dims.mix_rank, MINI_MIX_RANK);
        assert_eq!(model.dims.vocab, MINI_VOCAB);
        assert_eq!(model.dims.num_experts, MINI_EXPERTS);
        assert_eq!(model.dims.expert_dim, MINI_FFN);
        assert_eq!(model.dims.top_k, MINI_EXPERTS); // clamped from 10
        let attn = model
            .layers
            .iter()
            .filter(|l| matches!(l.block, Block::Attention(_)))
            .count();
        assert_eq!(attn, 12, "attention layers every 4th index");
        assert!(model
            .layers
            .iter()
            .enumerate()
            .all(|(i, l)| matches!(l.block, Block::Attention(_)) == mini_is_attention_layer(i)));
    }

    #[test]
    fn load_rejects_an_embed_only_checkpoint() {
        // The protocol-style embed-only fixture is not a full set.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"HGN1");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&104u64.to_le_bytes());
        bytes.extend_from_slice(&264u64.to_le_bytes());
        bytes.extend_from_slice(&264u64.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 64]);
        let mut entry = [0u8; hgn::ENTRY_BYTES];
        entry[..19].copy_from_slice(b"embed_tokens.weight");
        entry[96..100].copy_from_slice(&0u32.to_le_bytes());
        entry[100..104].copy_from_slice(&2u32.to_le_bytes());
        entry[104..112].copy_from_slice(&(MINI_VOCAB as u64).to_le_bytes());
        entry[112..120].copy_from_slice(&(MINI_HIDDEN as u64).to_le_bytes());
        entry[136..144].copy_from_slice(&264u64.to_le_bytes());
        bytes.extend_from_slice(&entry);
        let ck = hgn::load_prefix("ccfs-embed-only.hgn", bytes, 264).expect("loads");
        assert!(load(&ck).is_none(), "embed-only is not a full set");
    }

    #[test]
    fn load_rejects_a_q4c_tensor_whose_shape_fails_the_codec() {
        // A q4c embed with k = 4: the codec requires k % 32 == 0 (and an
        // empty payload), so the tensor cannot be decoded → None.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"HGN1");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&104u64.to_le_bytes());
        bytes.extend_from_slice(&264u64.to_le_bytes());
        bytes.extend_from_slice(&264u64.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 64]);
        let mut entry = [0u8; hgn::ENTRY_BYTES];
        entry[..19].copy_from_slice(b"embed_tokens.weight");
        entry[96..100].copy_from_slice(&5u32.to_le_bytes()); // q4c
        entry[100..104].copy_from_slice(&2u32.to_le_bytes());
        entry[104..112].copy_from_slice(&(MINI_VOCAB as u64).to_le_bytes());
        entry[112..120].copy_from_slice(&(MINI_HIDDEN as u64).to_le_bytes());
        entry[136..144].copy_from_slice(&264u64.to_le_bytes());
        bytes.extend_from_slice(&entry);
        let ck = hgn::load_prefix("ccfs-q4c.hgn", bytes, 264).expect("loads");
        assert!(load(&ck).is_none());
    }

    #[test]
    fn resolve_planes_rejects_an_own_suh_that_is_not_f16() {
        // Review fix: an own `.suh` of the wrong dtype must reject the
        // tensor, not silently fall through to the shared `.su` (the
        // spec requires the planes to be f16).
        let spec = |name: &str, dims: Vec<u64>, dtype: u32| FixtureSpec {
            name: name.to_string(),
            kind: 1,
            dims,
            dtype,
        };
        let weight = FixtureSpec {
            name: "blk.proj.weight".to_string(),
            kind: 2,
            dims: vec![1, 4],
            dtype: 16,
        };
        // Own `.suh` present but bf16 while the shared `.su` is a valid
        // f16 plane: rejected (the old find_map picked the shared plane).
        let ck = build_fixture(
            "ccfs-planes-badsuh.hgn",
            &[
                weight.clone(),
                spec("blk.proj.suh", vec![4], 0),
                spec("blk.proj.svh", vec![1], 2),
                spec("blk.su", vec![4], 2),
            ],
        );
        let by_name: HashMap<&str, usize> = ck
            .entries
            .iter()
            .enumerate()
            .map(|(i, e)| (e.name(), i))
            .collect();
        assert_eq!(resolve_planes(&ck, &by_name, 16, "blk.proj.weight"), None);
        // Own `.suh` present and f16 (the shared `.su` is the bf16 one):
        // resolves to the own plane's values.
        let ck = build_fixture(
            "ccfs-planes-ownsuh.hgn",
            &[
                weight.clone(),
                spec("blk.proj.suh", vec![4], 2),
                spec("blk.proj.svh", vec![1], 2),
                spec("blk.su", vec![4], 0),
            ],
        );
        let by_name: HashMap<&str, usize> = ck
            .entries
            .iter()
            .enumerate()
            .map(|(i, e)| (e.name(), i))
            .collect();
        let planes = resolve_planes(&ck, &by_name, 16, "blk.proj.weight")
            .expect("own f16 suh resolves")
            .expect("planes present");
        assert_eq!(
            planes.0,
            load_f16_plane(&ck, &by_name, "blk.proj.suh").expect("f16 own suh")
        );
        // No own `.suh`: the shared `.su` is used (the fused expert set,
        // spec §7).
        let ck = build_fixture(
            "ccfs-planes-shared.hgn",
            &[
                weight,
                spec("blk.proj.svh", vec![1], 2),
                spec("blk.su", vec![4], 2),
            ],
        );
        let by_name: HashMap<&str, usize> = ck
            .entries
            .iter()
            .enumerate()
            .map(|(i, e)| (e.name(), i))
            .collect();
        let planes = resolve_planes(&ck, &by_name, 16, "blk.proj.weight")
            .expect("shared su resolves")
            .expect("planes present");
        assert_eq!(
            planes.0,
            load_f16_plane(&ck, &by_name, "blk.su").expect("f16 shared su")
        );
    }

    #[test]
    fn load_accepts_the_quantised_mini_checkpoint() {
        // The fixture carrying one tensor per quantised dtype (q4c,
        // q8g64, ht, i4r, q6g64) plus their f16 planes loads as a real
        // model — the regression this change exists for (the real
        // checkpoint previously fell back to the stub).
        let ck = mini_checkpoint_quantised();
        assert_eq!(ck.entries.len(), 48 * 24 + 9 + 5 + 4);
        let model = load(&ck).expect("quantised mini checkpoint loads a real model");
        assert_eq!(model.dims.num_layers, MINI_LAYERS);
        assert_eq!(model.dims.num_streams, MINI_STREAMS);
        assert_eq!(model.dims.hidden_per_stream, MINI_HIDDEN_QUANTISED);
        assert_eq!(model.dims.vocab, MINI_VOCAB);
        assert_eq!(model.dims.num_experts, MINI_EXPERTS);
        assert_eq!(model.dims.expert_dim, MINI_FFN);
        assert_eq!(model.dims.top_k, MINI_EXPERTS); // clamped from 10
    }

    #[test]
    fn fixture_payload_offsets_are_16_byte_aligned() {
        // Review fix: the engine requires 16 B-aligned payloads; the
        // builder pads the data section start and every stored payload
        // to a 16 B boundary (the raw table layout would put payloads
        // at 104 + n·160 ≡ 8 (mod 16)).
        for ck in [mini_checkpoint(), mini_checkpoint_quantised()] {
            for entry in &ck.entries {
                assert_eq!(entry.data_offset() % 16, 0, "{}", entry.name());
            }
        }
    }

    #[test]
    fn mlp_forward_normalises_the_full_router_before_selecting_top_k() {
        // Review fix #2 (Qwen3-Next convention): softmax runs over all
        // router logits and the top-k probabilities are used as expert
        // weights. Softmaxing only the selected subset (the previous
        // behaviour) renormalised them to sum to 1 and inflated the
        // routed output.
        let s = ops::silu(1.0);
        let mlp = Mlp {
            gate: Matrix::new(3, 1, vec![0.0, 1.0, 2.0]).unwrap(),
            // per expert: gate row 1.0, up row 1.0 → act = silu(1.0)
            gate_up: Matrix::new(6, 1, vec![1.0; 6]).unwrap(),
            // expert e's down row is e + 1 → expert_out(e) = (e + 1) · s
            down: Matrix::new(3, 1, vec![1.0, 2.0, 3.0]).unwrap(),
            // the shared expert contributes exactly 0 (silu(0) · 1
            // through a zero down_proj)
            shared_gate_proj: Matrix::new(1, 1, vec![0.0]).unwrap(),
            shared_up_proj: Matrix::new(1, 1, vec![1.0]).unwrap(),
            shared_down_proj: Matrix::new(1, 1, vec![0.0]).unwrap(),
            shared_router: vec![0.0],
            ffn: 1,
            top_k: 2,
        };
        let mut probs = vec![0.0f32, 1.0, 2.0];
        ops::softmax(&mut probs);
        let expected = (3.0 * probs[2] + 2.0 * probs[1]) * s;
        let out = mlp.forward(&[1.0]);
        assert_eq!(out.len(), 1);
        assert!(
            (out[0] - expected).abs() <= expected.abs().max(1.0) * 1e-6,
            "expected {expected}, got {}",
            out[0]
        );
    }

    #[test]
    fn mini_fixture_is_deterministic() {
        let a = mini_checkpoint();
        let b = mini_checkpoint();
        assert_eq!(a.entries.len(), b.entries.len());
        assert!(a
            .entries
            .iter()
            .zip(&b.entries)
            .all(|(x, y)| x.name() == y.name() && x.data_offset() == y.data_offset()));
    }
}
