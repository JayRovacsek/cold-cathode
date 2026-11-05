//! The typed tensor set the real forward path runs on (spec
//! `flash_serve-engine.md` §2 tensor inventory, §3 block forwards).
//!
//! [`load`] walks a loaded [`hgn::Checkpoint`], validates the full
//! `qwen4exp` tensor-name set (48 layers, both hyper-connection mixers per
//! layer, the `self_attn`/`linear_attn` block tensors, the MoE MLP and the
//! top-level `embed_tokens`/`lm_head`/`hyper_connection_mixer`), decodes
//! every payload through [`crate::bf16`] and derives the shape constants
//! (hidden width, stream count, head counts, expert count) from the entry
//! dims — the same derivation the spec's §2 head-configuration resolution
//! uses (e.g. `num_heads = q_proj.rows / 2 / head_dim` from the 24-vs-48
//! `o_proj` note).
//!
//! Checkpoints whose required tensors carry dtypes without a dequant
//! codec (the real `qwen38-flash-next-v2-abliterated.hgn`: `q4c`/`ht`/
//! `i4r`/`q6g64` payloads) yield `None` — the caller falls back to the
//! documented stub (module docs in [`crate::forward`]).
//!
//! The mini-checkpoint fixture ([`mini_checkpoint`]) is the test vehicle:
//! the full tensor-name set with shrunk dims (hidden 4, head dims 8, 4
//! experts of width 8, mix rank 8, 48 layers), every payload bf16 with
//! deterministic pseudo-random values.

use std::cmp::Ordering;
use std::collections::HashMap;

use crate::bf16;
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

/// A decoded tensor: its (non-zero) dims and `f32` values.
#[derive(Debug, Clone)]
struct Tensor {
    dims: Vec<u64>,
    data: Vec<f32>,
}

impl Tensor {
    /// The values as a row-major matrix: 2-D tensors directly, 3-D
    /// tensors `[a, b, c]` flattened to `[a·b, c]` (the stacked expert
    /// `gate_up_proj`/`down_proj` and the `[channels, 1, kernel]` conv1d
    /// entries, read as `[channels, kernel]`).
    fn matrix(&self) -> Option<Matrix> {
        match self.dims.as_slice() {
            [rows, cols] => Matrix::new(*rows as usize, *cols as usize, self.data.clone()),
            [a, b, c] => Matrix::new((*a * *b) as usize, *c as usize, self.data.clone()),
            _ => None,
        }
    }

    fn vector(&self) -> Option<Vec<f32>> {
        (self.dims.len() == 1).then(|| self.data.clone())
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
    pub conv_w: Matrix,
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
/// the full bf16/f32 tensor set (module docs).
pub fn load(checkpoint: &hgn::Checkpoint) -> Option<Model> {
    let mut by_name: HashMap<&str, usize> = HashMap::new();
    for (index, entry) in checkpoint.entries.iter().enumerate() {
        by_name.entry(entry.name()).or_insert(index);
    }
    let tensor = |name: &str| -> Option<Tensor> {
        let &index = by_name.get(name)?;
        let entry = &checkpoint.entries[index];
        let data = bf16::decode_tensor(entry.dtype(), checkpoint.tensor_bytes(index)?)?;
        let dims: Vec<u64> = entry.dims().into_iter().filter(|d| *d != 0).collect();
        let expected: u64 = dims.iter().product();
        if expected == 0 || data.len() as u64 != expected {
            return None;
        }
        Some(Tensor { dims, data })
    };

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
    // Tensor::matrix reads them as `[channels, kernel]`.
    let conv_w = conv.matrix()?;
    if conv_w.rows != qkv.rows {
        return None;
    }
    Some(LinearBlock {
        qkv,
        z,
        a,
        b,
        a_log,
        dt_bias,
        conv_w,
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
        shared_router: shared_router.data.clone(),
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
/// Attention layers sit every 4th index (3, 7, …, 47) — the real file's
/// pattern (spec §2).
pub fn mini_is_attention_layer(layer: usize) -> bool {
    layer % 4 == 3
}

/// Build the full tensor-name set (48 layers, both mixers, the block
/// tensors for the layer kind, the MoE MLP, the layer-1 PLE extras and the
/// top-level tensors) with mini dims, every payload bf16 with
/// deterministic values, and return the parsed checkpoint.
pub fn mini_checkpoint() -> hgn::Checkpoint {
    // (name, kind, dims) — the top-level tensors seed the list.
    let mut specs: Vec<(String, u32, Vec<u64>)> = vec![
        (
            "embed_tokens.weight".to_string(),
            2,
            vec![MINI_VOCAB as u64, MINI_HIDDEN as u64],
        ),
        (
            "lm_head.weight".to_string(),
            2,
            vec![MINI_VOCAB as u64, MINI_HIDDEN as u64],
        ),
        (
            "hyper_connection_mixer.hc_norm.weight".to_string(),
            1,
            vec![(MINI_STREAMS * MINI_HIDDEN) as u64],
        ),
        (
            "hyper_connection_mixer.input_mix_weight_down.weight".to_string(),
            2,
            vec![MINI_MIX_RANK as u64, (MINI_STREAMS * MINI_HIDDEN) as u64],
        ),
        (
            "hyper_connection_mixer.input_mix_weight_up.weight".to_string(),
            2,
            vec![(MINI_STREAMS * MINI_HIDDEN) as u64, MINI_MIX_RANK as u64],
        ),
    ];
    for layer in 0..MINI_LAYERS {
        let prefix = format!("layers.{layer}");
        for mixer in ["attn_hyper_connection", "mlp_hyper_connection"] {
            specs.push((
                format!("{prefix}.{mixer}.hc_norm.weight"),
                1,
                vec![(MINI_STREAMS * MINI_HIDDEN) as u64],
            ));
            specs.push((
                format!("{prefix}.{mixer}.block_inject_weight.weight"),
                2,
                vec![MINI_STREAMS as u64, (MINI_STREAMS * MINI_HIDDEN) as u64],
            ));
            specs.push((
                format!("{prefix}.{mixer}.input_mix_weight_down.weight"),
                2,
                vec![MINI_MIX_RANK as u64, (MINI_STREAMS * MINI_HIDDEN) as u64],
            ));
            specs.push((
                format!("{prefix}.{mixer}.input_mix_weight_up.weight"),
                2,
                vec![(MINI_STREAMS * MINI_HIDDEN) as u64, MINI_MIX_RANK as u64],
            ));
        }
        if mini_is_attention_layer(layer) {
            // q_proj exposes 2·head_dim rows; the forward consumes half
            // (the 24-vs-48 nuance, spec §2) → 1 query head.
            specs.push((
                format!("{prefix}.self_attn.q_proj.weight"),
                2,
                vec![(2 * MINI_HEAD_DIM) as u64, MINI_HIDDEN as u64],
            ));
            specs.push((
                format!("{prefix}.self_attn.q_norm.weight"),
                1,
                vec![MINI_HEAD_DIM as u64],
            ));
            specs.push((
                format!("{prefix}.self_attn.k_proj.weight"),
                2,
                vec![MINI_HEAD_DIM as u64, MINI_HIDDEN as u64],
            ));
            specs.push((
                format!("{prefix}.self_attn.k_norm.weight"),
                1,
                vec![MINI_HEAD_DIM as u64],
            ));
            specs.push((
                format!("{prefix}.self_attn.v_proj.weight"),
                2,
                vec![MINI_HEAD_DIM as u64, MINI_HIDDEN as u64],
            ));
            specs.push((
                format!("{prefix}.self_attn.o_proj.weight"),
                2,
                vec![MINI_HIDDEN as u64, MINI_HEAD_DIM as u64],
            ));
            specs.push((
                format!("{prefix}.self_attn.indexer.index_qk_proj.weight"),
                2,
                vec![MINI_HEAD_DIM as u64, MINI_HIDDEN as u64],
            ));
            specs.push((
                format!("{prefix}.self_attn.indexer.q_layernorm.weight"),
                1,
                vec![MINI_HEAD_DIM as u64],
            ));
            specs.push((
                format!("{prefix}.self_attn.indexer.k_layernorm.weight"),
                1,
                vec![MINI_HEAD_DIM as u64],
            ));
        } else {
            // Packed qkv: 1 q + 1 k head (8 each) + 1 v head (8) = 24 rows,
            // matching the real `[10240, 2560]` = q 2048 + k 2048 + v 6144
            // layout at mini scale (spec §3 `FUN_01fa0700`).
            specs.push((
                format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                2,
                vec![(3 * MINI_HEAD_DIM) as u64, MINI_HIDDEN as u64],
            ));
            specs.push((
                format!("{prefix}.linear_attn.in_proj_z.weight"),
                2,
                vec![MINI_HEAD_DIM as u64, MINI_HIDDEN as u64],
            ));
            specs.push((
                format!("{prefix}.linear_attn.in_proj_a.weight"),
                2,
                vec![1, MINI_HIDDEN as u64],
            ));
            specs.push((
                format!("{prefix}.linear_attn.in_proj_b.weight"),
                2,
                vec![1, MINI_HIDDEN as u64],
            ));
            specs.push((format!("{prefix}.linear_attn.A_log"), 1, vec![1]));
            specs.push((format!("{prefix}.linear_attn.dt_bias"), 1, vec![1]));
            specs.push((
                format!("{prefix}.linear_attn.conv1d.weight"),
                3,
                vec![(3 * MINI_HEAD_DIM) as u64, 1, 4],
            ));
            specs.push((
                format!("{prefix}.linear_attn.norm.weight"),
                1,
                vec![MINI_HEAD_DIM as u64],
            ));
            specs.push((
                format!("{prefix}.linear_attn.out_proj.weight"),
                2,
                vec![MINI_HIDDEN as u64, MINI_HEAD_DIM as u64],
            ));
        }
        specs.push((
            format!("{prefix}.mlp.gate.weight"),
            2,
            vec![MINI_EXPERTS as u64, MINI_HIDDEN as u64],
        ));
        specs.push((
            format!("{prefix}.mlp.experts.gate_up_proj.weight"),
            3,
            vec![
                MINI_EXPERTS as u64,
                (2 * MINI_FFN) as u64,
                MINI_HIDDEN as u64,
            ],
        ));
        specs.push((
            format!("{prefix}.mlp.experts.down_proj.weight"),
            3,
            vec![MINI_EXPERTS as u64, MINI_HIDDEN as u64, MINI_FFN as u64],
        ));
        specs.push((
            format!("{prefix}.mlp.shared_expert.gate_proj.weight"),
            2,
            vec![MINI_FFN as u64, MINI_HIDDEN as u64],
        ));
        specs.push((
            format!("{prefix}.mlp.shared_expert.up_proj.weight"),
            2,
            vec![MINI_FFN as u64, MINI_HIDDEN as u64],
        ));
        specs.push((
            format!("{prefix}.mlp.shared_expert.down_proj.weight"),
            2,
            vec![MINI_HIDDEN as u64, MINI_FFN as u64],
        ));
        specs.push((
            format!("{prefix}.mlp.shared_expert_gate.weight"),
            2,
            vec![1, MINI_HIDDEN as u64],
        ));
        // PLE exists only on layer 1 in the real file; the forward path
        // skips it (out of scope), but the fixture keeps the name set.
        if layer == 1 {
            specs.push((
                format!("{prefix}.ple.conv1d.weight"),
                3,
                vec![(MINI_STREAMS * MINI_HIDDEN) as u64, 1, 4],
            ));
            specs.push((
                format!("{prefix}.ple.key_proj.weight"),
                2,
                vec![(MINI_STREAMS * MINI_HIDDEN) as u64, MINI_HIDDEN as u64],
            ));
            specs.push((
                format!("{prefix}.ple.norm_conv.weight"),
                1,
                vec![(MINI_STREAMS * MINI_HIDDEN) as u64],
            ));
            specs.push((
                format!("{prefix}.ple.norm_key.weight"),
                1,
                vec![(MINI_STREAMS * MINI_HIDDEN) as u64],
            ));
            specs.push((
                format!("{prefix}.ple.norm_query.weight"),
                1,
                vec![(MINI_STREAMS * MINI_HIDDEN) as u64],
            ));
            specs.push((
                format!("{prefix}.ple.ple_embedding.layer_multipliers"),
                1,
                vec![3],
            ));
            specs.push((
                format!("{prefix}.ple.ple_embedding.ngram_heads_offsets"),
                1,
                vec![16],
            ));
            specs.push((
                format!("{prefix}.ple.ple_embedding.ngram_heads_vocab_sizes"),
                1,
                vec![16],
            ));
            specs.push((
                format!("{prefix}.ple.value_proj.weight"),
                2,
                vec![MINI_HIDDEN as u64, MINI_HIDDEN as u64],
            ));
        }
    }

    // Serialise header + entries + payloads (layout per `hgn::TensorEntry`).
    let entry_count = specs.len() as u64;
    let table_offset = hgn::HEADER_BYTES as u64;
    let data_offset = table_offset + entry_count * hgn::ENTRY_BYTES as u64;
    let mut payloads: Vec<u8> = Vec::new();
    let mut entries: Vec<u8> = Vec::new();
    for (name, kind, dims) in &specs {
        let values = fixture_values(name, dims);
        let payload = bf16::encode_bf16(&values);
        let offset = data_offset + payloads.len() as u64;
        let mut entry = [0u8; hgn::ENTRY_BYTES];
        entry[..name.len()].copy_from_slice(name.as_bytes());
        entry[96..100].copy_from_slice(&0u32.to_le_bytes()); // dtype bf16
        entry[100..104].copy_from_slice(&kind.to_le_bytes());
        for (i, dim) in dims.iter().take(4).enumerate() {
            entry[104 + i * 8..112 + i * 8].copy_from_slice(&dim.to_le_bytes());
        }
        entry[136..144].copy_from_slice(&offset.to_le_bytes());
        entry[144..152].copy_from_slice(&(payload.len() as u64).to_le_bytes());
        entries.extend_from_slice(&entry);
        payloads.extend_from_slice(&payload);
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
    bytes.extend_from_slice(&payloads);
    hgn::load_prefix("ccfs-mini-fixture.hgn", bytes, declared).expect("mini fixture loads")
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
    fn load_rejects_quantised_dtypes_without_a_codec() {
        // The real checkpoint's embed is q4c (dtype 5): no codec → None.
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
