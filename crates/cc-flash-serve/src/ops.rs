//! Pure-Rust tensor ops for the real forward path (spec
//! `flash_serve-engine.md` §3 op sequences: `norm → qkv → rope →
//! attention → projection → mlp`, the Gated DeltaNet block and the MoE
//! MLP). Everything operates on `f32` slices (dequantised via
//! [`crate::bf16`] and [`crate::codecs`]); the reference runs the same
//! math through embedded AMDGPU code objects, which the port replaces
//! with straightforward CPU kernels (no `unsafe`, no O(n²) accidents:
//! every kernel is a single pass over its operands).

use crate::codecs::Weights;

/// Row-major matrix backed by [`Weights`]: dense matrices hold their
/// `f32` values (`weights` is `Weights::Dense` with `rows * cols`
/// entries), quantised matrices keep the raw payload and decode one row
/// at a time, so [`Matrix::gemv_rows`] costs O(row) on quantised
/// weights instead of materialising the whole tensor per call.
#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub weights: Weights,
}

impl Matrix {
    /// Build a dense matrix from raw values; `None` when the length
    /// mismatches.
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Option<Self> {
        if rows.checked_mul(cols)? == data.len() {
            Some(Matrix {
                rows,
                cols,
                weights: Weights::Dense(data),
            })
        } else {
            None
        }
    }

    /// Build a matrix from any [`Weights`] variant; `None` when the
    /// weights do not match the shape.
    pub fn with_weights(rows: usize, cols: usize, weights: Weights) -> Option<Self> {
        let matrix = Matrix {
            rows,
            cols,
            weights,
        };
        matrix.weights_matches_shape().then_some(matrix)
    }

    /// The decoded values of row `i` (dense: a slice clone; quantised:
    /// the fused row decode).
    pub fn row(&self, i: usize) -> Vec<f32> {
        self.weights.row(self.cols, i)
    }

    /// The dense values when this matrix is not quantised.
    pub fn dense(&self) -> Option<&[f32]> {
        match &self.weights {
            Weights::Dense(data) => Some(data),
            _ => None,
        }
    }

    /// `y = W x` for the full matrix (`x.len() == cols`, result `rows`).
    pub fn gemv(&self, x: &[f32]) -> Vec<f32> {
        self.gemv_rows(0, self.rows, x)
    }

    /// `y = W[start..start+count] x` — the row-slice gemv the full-attention
    /// block needs (the `q_proj` 24-vs-48 nuance: the gemv computes all
    /// 12 288 rows, the rope/attention path consumes the first 6 144,
    /// spec §2).
    pub fn gemv_rows(&self, start: usize, count: usize, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.cols, "gemv input width");
        debug_assert!(start.saturating_add(count) <= self.rows, "gemv rows");
        match &self.weights {
            // Dense fast path: dot straight over the stored slice.
            Weights::Dense(data) => (0..count)
                .map(|row| {
                    let offset = (start + row) * self.cols;
                    dot(&data[offset..offset + self.cols], x)
                })
                .collect(),
            // Quantised path: dequantise exactly the rows the gemv reads,
            // all through one reused scratch buffer (dot_row) instead of
            // one Vec per row.
            _ => {
                let mut scratch = Vec::with_capacity(self.cols);
                (0..count)
                    .map(|row| self.dot_row(start + row, x, &mut scratch))
                    .collect()
            }
        }
    }

    /// The dot product of row `row` with `x`. The quantised path
    /// dequantises into `scratch` (cleared and refilled), so repeated
    /// row gemvs reuse one allocation — the real `lm_head` gemv decodes
    /// 248 320 rows per token and would otherwise allocate one Vec per
    /// row. The dense path dots the stored slice directly and leaves
    /// `scratch` untouched.
    pub fn dot_row(&self, row: usize, x: &[f32], scratch: &mut Vec<f32>) -> f32 {
        debug_assert!(row < self.rows, "dot_row index");
        match &self.weights {
            Weights::Dense(data) => dot(&data[row * self.cols..(row + 1) * self.cols], x),
            weights => {
                weights.row_into(self.cols, row, scratch);
                dot(scratch, x)
            }
        }
    }

    /// Check the weights against the shape (dense length, or the
    /// quantised payload formula and plane lengths for `rows × cols`).
    fn weights_matches_shape(&self) -> bool {
        match &self.weights {
            Weights::Dense(data) => self.rows.checked_mul(self.cols) == Some(data.len()),
            Weights::Q4c { bytes, .. } => {
                self.cols.is_multiple_of(32)
                    && self
                        .rows
                        .checked_mul(crate::codecs::pad16(9 * self.cols / 16))
                        .and_then(|bytes_per_row| bytes_per_row.checked_add(64))
                        == Some(bytes.len())
            }
            Weights::Q8g64 { bytes } => {
                self.cols.is_multiple_of(64)
                    && self
                        .rows
                        .checked_mul(self.cols / 64)
                        .and_then(|blocks| blocks.checked_mul(68))
                        == Some(bytes.len())
            }
            Weights::Q6g64 { bytes } => {
                self.cols.is_multiple_of(64)
                    && self
                        .rows
                        .checked_mul(crate::codecs::pad16(13 * self.cols / 16))
                        == Some(bytes.len())
            }
            Weights::Ht {
                bytes,
                suh,
                svh,
                svh_rows,
            } => {
                // Mirrors `Weights::from_dtype`'s plane checks so a
                // hand-built variant can't carry planes the row decode
                // would index out of.
                self.cols.is_multiple_of(2)
                    && self.rows.checked_mul(self.cols / 2) == Some(bytes.len())
                    && suh.len() == self.cols
                    && svh.len() == *svh_rows
            }
            Weights::I4r {
                bytes,
                suh,
                svh,
                svh_rows,
            } => {
                self.cols.is_multiple_of(64)
                    && self
                        .rows
                        .checked_mul(self.cols / 64)
                        .and_then(|blocks| blocks.checked_mul(33))
                        == Some(bytes.len())
                    && suh.len() == self.cols
                    && svh.len() == *svh_rows
            }
        }
    }
}

/// Dot product (single pass, no temporaries).
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "dot length");
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// RMSNorm over one vector (`x` with element-wise `gain`, same length):
/// `y = x · rsqrt(mean(x²) + eps) · gain` — the reference's `q_norm` /
/// `k_norm` / `hc_norm` / linear-attn `norm` kernel family (spec §3).
pub fn rmsnorm(x: &[f32], gain: &[f32], eps: f32) -> Vec<f32> {
    debug_assert_eq!(x.len(), gain.len(), "rmsnorm gain width");
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / (x.len().max(1)) as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    x.iter().zip(gain).map(|(v, g)| v * inv * g).collect()
}

/// RoPE with the `rotate_half` (NeoX) pairing: for each head vector of
/// width `head_dim`, the halves `(x[i], x[i + d/2])` rotate by
/// `angle = position · base^(−2i/d)` (spec §3 step 4; `base = 1e6` is the
/// Qwen convention — not pinned by the spec, reported as an ambiguity).
/// Applies to every `head_dim`-wide head packed in `heads`.
pub fn rope_heads(heads: &mut [f32], head_dim: usize, position: usize, base: f32) {
    debug_assert!(head_dim.is_multiple_of(2), "rope head_dim parity");
    debug_assert!(heads.len().is_multiple_of(head_dim), "rope head packing");
    let half = head_dim / 2;
    for head in heads.chunks_mut(head_dim) {
        for i in 0..half {
            let angle = (position as f32) * base.powf(-(2.0 * i as f32) / head_dim as f32);
            let (cos, sin) = (angle.cos(), angle.sin());
            let (x1, x2) = (head[i], head[i + half]);
            head[i] = x1 * cos - x2 * sin;
            head[i + half] = x2 * cos + x1 * sin;
        }
    }
}

/// In-place softmax (max-subtracted for stability).
pub fn softmax(scores: &mut [f32]) {
    let max = scores.iter().fold(f32::NEG_INFINITY, |m, s| m.max(*s));
    let mut sum = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - max).exp();
        sum += *s;
    }
    for s in scores.iter_mut() {
        *s /= sum;
    }
}

/// Grouped-query attention over cached keys/values: `q` packs
/// `q_heads × head_dim`, `k`/`v` pack `n_keys × kv_heads × head_dim` and
/// `n_keys × kv_heads × v_head_dim`. Query head `h` reads KV head
/// `h · kv_heads / q_heads` (the GQA mapping; 24:1 in the real model,
/// spec §2). Returns `q_heads × v_head_dim`.
pub fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    v_head_dim: usize,
) -> Vec<f32> {
    debug_assert_eq!(q.len(), q_heads * head_dim, "attention q shape");
    let n_keys = k.len() / (kv_heads * head_dim).max(1);
    debug_assert_eq!(k.len(), n_keys * kv_heads * head_dim, "attention k shape");
    debug_assert_eq!(v.len(), n_keys * kv_heads * v_head_dim, "attention v shape");
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0; q_heads * v_head_dim];
    for h in 0..q_heads {
        let kv_head = (h * kv_heads) / q_heads.max(1);
        let q_head = &q[h * head_dim..(h + 1) * head_dim];
        let mut scores = vec![0.0; n_keys];
        for j in 0..n_keys {
            let k_row =
                &k[(j * kv_heads + kv_head) * head_dim..(j * kv_heads + kv_head + 1) * head_dim];
            scores[j] = dot(q_head, k_row) * scale;
        }
        softmax(&mut scores);
        let out_head = &mut out[h * v_head_dim..(h + 1) * v_head_dim];
        for j in 0..n_keys {
            let v_row = &v
                [(j * kv_heads + kv_head) * v_head_dim..(j * kv_heads + kv_head + 1) * v_head_dim];
            for (o, value) in out_head.iter_mut().zip(v_row) {
                *o += scores[j] * *value;
            }
        }
    }
    out
}

/// One Gated DeltaNet recurrence step for a single head (spec §3 step 6,
/// `k_dn_recur_step`; the exact β-normalisation/decay order is the open
/// sub-gap, implemented as the standard Gated DeltaNet form):
/// `S ← α(I − β k kᵀ)S + β v kᵀ`, `o = S q` with `S` laid out
/// `v_head_dim × head_dim`. `S` starts zeroed for a fresh sequence.
pub fn delta_rule_step(
    state: &mut [f32],
    k: &[f32],
    v: &[f32],
    q: &[f32],
    alpha: f32,
    beta: f32,
) -> Vec<f32> {
    let v_dim = v.len();
    let k_dim = k.len();
    debug_assert_eq!(state.len(), v_dim * k_dim, "delta state shape");
    debug_assert_eq!(q.len(), k_dim, "delta q width");
    // Sk per row (the projection of the state onto the new key).
    let mut sk = vec![0.0; v_dim];
    for row in 0..v_dim {
        sk[row] = dot(&state[row * k_dim..(row + 1) * k_dim], k);
    }
    let mut o = vec![0.0; v_dim];
    for row in 0..v_dim {
        let row_slice = &mut state[row * k_dim..(row + 1) * k_dim];
        // S' = α·S − α·β·(Sk)⊗k + β·v⊗k, fused into one pass:
        // S'[j] = α·S[j] + β·(v[row] − α·Sk[row])·k[j].
        let delta = beta * (v[row] - alpha * sk[row]);
        let mut acc = 0.0f32;
        for j in 0..k_dim {
            row_slice[j] = alpha * row_slice[j] + delta * k[j];
            acc += row_slice[j] * q[j];
        }
        o[row] = acc;
    }
    o
}

/// Depthwise causal conv1d step (spec §3 step 4: `conv1d.weight`
/// `[channels, kernel]` with a `kernel − 1` token history): `history`
/// keeps the previous frames (oldest first, at most `kernel − 1`), `x` is
/// the newest frame; returns the convolved channel values and shifts the
/// history.
pub fn conv1d_step(
    weight: &[f32],
    channels: usize,
    kernel: usize,
    history: &mut Vec<f32>,
    x: &[f32],
) -> Vec<f32> {
    debug_assert_eq!(x.len(), channels, "conv input width");
    debug_assert_eq!(weight.len(), channels * kernel, "conv weight shape");
    // `kernel - 1` below would underflow when kernel == 0.
    debug_assert!(kernel > 0, "conv kernel width");
    let hist_len = kernel - 1;
    debug_assert!(
        history.len().is_multiple_of(channels.max(1)) && history.len() <= hist_len * channels,
        "conv history packing"
    );
    let available = history.len() / channels.max(1);
    let mut out = vec![0.0; channels];
    for c in 0..channels {
        let w = &weight[c * kernel..(c + 1) * kernel];
        let mut acc = 0.0;
        // History frames pair with the leading taps (a partially filled
        // window left-aligns to the oldest taps); the newest frame `x`
        // uses the last kernel tap.
        for t in 0..available {
            acc += w[hist_len - available + t] * history[t * channels + c];
        }
        acc += w[kernel - 1] * x[c];
        out[c] = acc;
    }
    if history.len() == hist_len * channels {
        history.drain(..channels);
    }
    history.extend_from_slice(x);
    out
}

/// SiLU activation `x · sigmoid(x)`.
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Softplus `ln(1 + eˣ)` with the standard >20 linear fallback.
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Logistic sigmoid.
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(rows: usize, cols: usize, values: &[f32]) -> Matrix {
        Matrix::new(rows, cols, values.to_vec()).expect("matrix shape")
    }

    #[test]
    fn gemv_multiplies_rows_by_the_input() {
        let w = m(2, 3, &[1.0, 2.0, 3.0, 0.0, 1.0, -1.0]);
        assert_eq!(w.gemv(&[1.0, 0.0, 2.0]), vec![7.0, -2.0]);
        // Row-slice variant (the q_proj 24-vs-48 consumption pattern).
        assert_eq!(w.gemv_rows(1, 1, &[1.0, 0.0, 2.0]), vec![-2.0]);
        assert_eq!(w.gemv_rows(0, 1, &[1.0, 0.0, 2.0]), vec![7.0]);
    }

    #[test]
    fn matrix_new_rejects_bad_lengths() {
        assert!(Matrix::new(2, 3, vec![0.0; 6]).is_some());
        assert!(Matrix::new(2, 3, vec![0.0; 5]).is_none());
        assert!(Matrix::new(0, 0, Vec::new()).is_some());
    }

    #[test]
    fn quantised_matrix_row_and_gemv_match_the_codec_decode() {
        // A 4×64 q8g64 matrix (272 B, 16 B-aligned): with_weights
        // validates the payload, row() matches the codec's row decode
        // and gemv agrees with dotting those rows (the O(row) fused path).
        let values: Vec<f32> = (0..256).map(|i| (i as f32) * 0.5 - 64.0).collect();
        let bytes = crate::codecs::q8g64_encode(&values, 4, 64).unwrap();
        let weights = crate::codecs::Weights::from_dtype(7, &bytes, &[4, 64], None).unwrap();
        let matrix = Matrix::with_weights(4, 64, weights.clone()).expect("shape matches");
        assert_eq!(matrix.row(1), weights.row(64, 1));
        assert_eq!(matrix.dense(), None, "quantised weights are not dense");
        let x = vec![1.0f32; 64];
        let expected: Vec<f32> = (0..4).map(|r| matrix.row(r).iter().sum()).collect();
        assert_eq!(matrix.gemv(&x), expected);
        // A shape the payload does not back is rejected.
        assert!(Matrix::with_weights(3, 64, weights).is_none());
    }

    #[test]
    fn dot_row_reuses_one_scratch_and_matches_row_dot() {
        // The quantised gemv path (finding: one Vec per row): dot_row
        // fills the caller's scratch and returns the same value as
        // dotting the row decode, for every row, with one allocation.
        let values: Vec<f32> = (0..256).map(|i| (i as f32) * 0.5 - 64.0).collect();
        let bytes = crate::codecs::q8g64_encode(&values, 4, 64).unwrap();
        let weights = crate::codecs::Weights::from_dtype(7, &bytes, &[4, 64], None).unwrap();
        let matrix = Matrix::with_weights(4, 64, weights).unwrap();
        let x = vec![0.5f32; 64];
        let mut scratch = Vec::new();
        for row in 0..4 {
            let got = matrix.dot_row(row, &x, &mut scratch);
            assert_eq!(got, dot(&matrix.row(row), &x), "row {row}");
            assert_eq!(scratch, matrix.row(row), "scratch holds row {row}");
        }
        // The gemv_rows output equals the per-row dots (scratch reuse
        // doesn't change results).
        assert_eq!(
            matrix.gemv_rows(1, 2, &x),
            vec![
                matrix.dot_row(1, &x, &mut scratch),
                matrix.dot_row(2, &x, &mut scratch)
            ]
        );
        // Dense matrices expose dot_row too (scratch unused).
        let dense = m(2, 3, &[1.0, 2.0, 3.0, 0.0, 1.0, -1.0]);
        assert_eq!(dense.dot_row(1, &[1.0, 0.0, 2.0], &mut scratch), -2.0);
    }

    #[test]
    fn with_weights_validates_the_rotation_planes() {
        // Review fix: with_weights mirrors from_dtype's plane checks —
        // an ht/i4r variant whose suh/svh contradict the shape (or whose
        // svh contradicts svh_rows) is rejected instead of panicking in
        // the row decode.
        let ht = |suh: Vec<f32>, svh: Vec<f32>, svh_rows: usize| crate::codecs::Weights::Ht {
            bytes: vec![0x88; 16],
            suh,
            svh,
            svh_rows,
        };
        assert!(Matrix::with_weights(2, 16, ht(vec![1.0; 16], vec![1.0; 2], 2)).is_some());
        assert!(
            Matrix::with_weights(2, 16, ht(vec![1.0; 15], vec![1.0; 2], 2)).is_none(),
            "suh width"
        );
        assert!(
            Matrix::with_weights(2, 16, ht(vec![1.0; 16], vec![1.0; 3], 2)).is_none(),
            "svh vs svh_rows"
        );
        assert!(
            Matrix::with_weights(2, 16, ht(vec![1.0; 16], vec![1.0; 2], 1)).is_none(),
            "svh_rows vs svh"
        );
        let i4r = |suh: Vec<f32>| crate::codecs::Weights::I4r {
            bytes: vec![0u8; 33],
            suh,
            svh: vec![1.0],
            svh_rows: 1,
        };
        assert!(Matrix::with_weights(1, 64, i4r(vec![1.0; 64])).is_some());
        assert!(
            Matrix::with_weights(1, 64, i4r(vec![1.0; 63])).is_none(),
            "i4r suh width"
        );
    }

    #[test]
    fn rmsnorm_normalises_then_scales_by_gain() {
        let x = vec![1.0, -1.0, 1.0, -1.0];
        let gain = vec![2.0, 2.0, 2.0, 2.0];
        // mean(x²) = 1 → rsqrt(1 + ~0) ≈ 1 → y = x · gain.
        let y = rmsnorm(&x, &gain, 1e-6);
        let want = [2.0f32, -2.0, 2.0, -2.0];
        for (got, want) in y.iter().zip(want.iter()) {
            assert!((got - want).abs() < 1e-4, "rmsnorm {y:?}");
        }
        // Zero input stays finite.
        let z = rmsnorm(&[0.0, 0.0], &[1.0, 1.0], 1e-6);
        assert!(z.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rope_rotates_half_pairs_and_is_identity_at_position_zero() {
        let mut heads = vec![1.0, 0.0, 0.0, 1.0]; // two 2-wide heads
        rope_heads(&mut heads, 2, 0, 1_000_000.0);
        assert_eq!(heads, vec![1.0, 0.0, 0.0, 1.0], "position 0 is identity");
        // Position 1, head_dim 2: angle 1 for the first (and only) pair.
        let mut heads = vec![1.0, 0.0];
        rope_heads(&mut heads, 2, 1, 1_000_000.0);
        let (c, s) = (1.0f32.cos(), 1.0f32.sin());
        assert!((heads[0] - c).abs() < 1e-6);
        assert!((heads[1] - s).abs() < 1e-6);
        // Rotation preserves the norm.
        let norm_before = 1.0f32;
        let norm_after = (heads[0] * heads[0] + heads[1] * heads[1]).sqrt();
        assert!((norm_before - norm_after).abs() < 1e-5);
    }

    #[test]
    fn softmax_normalises_exponentials() {
        let mut s = vec![0.0, 0.0];
        softmax(&mut s);
        assert!((s[0] - 0.5).abs() < 1e-6 && (s[1] - 0.5).abs() < 1e-6);
        let mut s = vec![100.0, 0.0]; // max-subtraction keeps this finite
        softmax(&mut s);
        assert!(s[0] > 0.999_999 && s[1].is_finite());
        let mut s = vec![1.0, 2.0, 3.0];
        softmax(&mut s);
        assert!((s.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn attention_with_one_key_returns_the_value_row() {
        // Single query head, single KV head, head_dim 2, v_head_dim 2.
        let q = vec![1.0, 0.0];
        let k = vec![1.0, 0.0];
        let v = vec![3.0, -1.0];
        let out = attention(&q, &k, &v, 1, 1, 2, 2);
        assert_eq!(out, vec![3.0, -1.0], "one key ⇒ softmax picks its value");
        // GQA mapping: 2 query heads share 1 KV head; both see the same k/v.
        let q = vec![1.0, 0.0, 0.5, 0.0];
        let out = attention(&q, &k, &v, 2, 1, 2, 2);
        assert_eq!(out, vec![3.0, -1.0, 3.0, -1.0]);
    }

    #[test]
    fn delta_rule_step_matches_the_closed_form_on_a_fresh_state() {
        // S = 0, α = 0, β = 1 ⇒ S' = v kᵀ ⇒ o = S' q = v (k·q).
        let mut state = vec![0.0; 4]; // v_dim 2 × k_dim 2
        let k = vec![1.0, 0.0];
        let v = vec![2.0, 5.0];
        let q = vec![0.0, 1.0];
        let o = delta_rule_step(&mut state, &k, &v, &q, 0.0, 1.0);
        assert_eq!(o, vec![0.0, 0.0], "k·q = 0 ⇒ o = 0");
        assert_eq!(state, vec![2.0, 0.0, 5.0, 0.0], "S' = v kᵀ");
        // k·q = 1 ⇒ o = v.
        let mut state = vec![0.0; 4];
        let o = delta_rule_step(&mut state, &k, &v, &k, 0.0, 1.0);
        assert!((o[0] - 2.0).abs() < 1e-6 && (o[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn delta_rule_step_revisits_a_warm_state() {
        // Seed S = v1 k1ᵀ (via one step), then step with k2 = k1, v2,
        // α = 1, β = 1: the delta rule replaces the k1 slot with v2.
        let mut state = vec![0.0; 4];
        delta_rule_step(&mut state, &[1.0, 0.0], &[1.0, 0.0], &[1.0, 0.0], 1.0, 1.0);
        let o = delta_rule_step(&mut state, &[1.0, 0.0], &[7.0, 0.0], &[1.0, 0.0], 1.0, 1.0);
        assert!(
            (o[0] - 7.0).abs() < 1e-5,
            "delta rule overwrites the key: {o:?}"
        );
    }

    #[test]
    fn conv1d_step_uses_the_rolling_history() {
        // 2 channels, kernel 3: w[c] = [1, 2, 4] (oldest tap 1).
        let w = vec![1.0, 2.0, 4.0, 0.5, 1.0, 2.0];
        let mut hist: Vec<f32> = Vec::new();
        // First token: only the newest tap fires.
        let out = conv1d_step(&w, 2, 3, &mut hist, &[1.0, 1.0]);
        assert_eq!(out, vec![4.0, 2.0]);
        // Second token: previous frame shifts to tap 1.
        let out = conv1d_step(&w, 2, 3, &mut hist, &[0.0, 0.0]);
        assert_eq!(out, vec![2.0, 1.0]);
        // Third token fills the window: 1·1 + 2·0 + 4·0 for c0.
        let out = conv1d_step(&w, 2, 3, &mut hist, &[0.0, 0.0]);
        assert_eq!(out, vec![1.0, 0.5]);
        // Fourth token evicts the oldest frame.
        let out = conv1d_step(&w, 2, 3, &mut hist, &[0.0, 0.0]);
        assert_eq!(out, vec![0.0, 0.0]);
        assert_eq!(hist.len(), 2 * 2, "history keeps kernel−1 frames");
    }

    #[test]
    fn activations_match_closed_forms() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-7);
        assert!((silu(0.0) - 0.0).abs() < 1e-7);
        assert!((silu(1.0) - 1.0 / (1.0 + (-1.0f32).exp())).abs() < 1e-7);
        assert!((softplus(0.0) - std::f32::consts::LN_2).abs() < 1e-6);
        assert!((softplus(30.0) - 30.0).abs() < 1e-4, "linear fallback");
        assert!(softplus(-30.0).is_finite());
    }
}
