//! Sampling layer for the `GEN` path (spec `flash_serve-engine.md` §6).
//!
//! The reference samples on the GPU (`halogen::k_sample_f32` plus the
//! `k_smpg_*` staged pipeline — mass → pick → quantised write → accept,
//! spec §6); the port implements the same contract in plain CPU Rust,
//! decoupled from the logits source so it is unit-testable today and is
//! driven by `crate::forward` (the forward-pass entry point) for real
//! requests.
//!
//! Contract sources:
//!
//! 1. The `SAMPLE` clause grammar `SAMPLE %.9g %d %.9g %.9g %d` =
//!    (temperature, top_k, top_p, min_p, seed), emitted by the reference
//!    front-end (serve_api.py:1958, quoted in spec §6).
//! 2. The engine's validation wording `SAMPLE temperature outside (0, 2]
//!    or top_k < 0` (binutils-strings.txt:979, spec §6): the port accepts
//!    the documented `(0, 2]` range plus `temperature = 0` — the greedy
//!    path the serve banner announces (`BATCHED greedy or sampled`,
//!    :3753) — and any `top_k >= 0` (`0` = no limit).
//! 3. The front-end's model-level filter defaults (`{"top_k": 20,
//!    "top_p": 0.95}`, serve_api.py:2759) belong to the front-end's
//!    precedence chain (request `SAMPLE` > `HALOGEN_*` env > model
//!    filters); the engine honours the request's own `SAMPLE` clause and
//!    falls back to greedy when the clause is absent.
//!
//! Filter order matches the reference pipeline shape (mass → pick):
//! temperature scaling, then top-k, then top-p (nucleus), then min-p, then
//! roulette selection from the surviving renormalised distribution. The
//! reported token logprob is the natural log of that final probability —
//! the value the front-end's `t_line_logprob` (serve_api.py:1066–1074)
//! parses from the `T` line.

use crate::{parse_float_strict, strtol};

/// The five `SAMPLE` fields (temperature, top_k, top_p, min_p, seed).
#[derive(Debug, Clone, PartialEq)]
pub struct SampleParams {
    /// `0` selects the greedy path; sampled values live in `(0, 2]`.
    pub temperature: f64,
    /// `0` = no limit; must be `>= 0` (spec §6 validation).
    pub top_k: i64,
    /// Nucleus mass; `1.0` keeps the whole distribution.
    pub top_p: f64,
    /// Drops tokens whose probability is below `min_p × best`; `0` = off.
    pub min_p: f64,
    /// Sampler seed; the port's PRNG is deterministic per seed.
    pub seed: i64,
}

impl SampleParams {
    /// Greedy sampling (temperature 0): always take the argmax. The
    /// default when a `GEN` line carries no `SAMPLE` clause.
    pub fn greedy() -> Self {
        SampleParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: 0,
        }
    }

    /// Validate one `SAMPLE` clause against the reference's rule (module
    /// docs): `temperature == 0 || temperature in (0, 2]`, `top_k >= 0`.
    /// `top_p`/`min_p` carry no reference validation wording and are
    /// accepted verbatim (degenerate values degrade to keep-all/keep-best).
    pub fn new(temperature: f64, top_k: i64, top_p: f64, min_p: f64, seed: i64) -> Option<Self> {
        let temperature_ok = temperature == 0.0 || (0.0 < temperature && temperature <= 2.0);
        if !temperature_ok || top_k < 0 {
            return None;
        }
        Some(SampleParams {
            temperature,
            top_k,
            top_p,
            min_p,
            seed,
        })
    }

    /// Parse the exact `SAMPLE %.9g %d %.9g %.9g %d` field tokens the
    /// front-end emits after the keyword (C-library number semantics via
    /// `crate::parse_float_strict`/`crate::strtol`).
    pub fn parse_sample(fields: [&str; 5]) -> Option<Self> {
        let temperature = parse_float_strict(fields[0])?;
        let top_k = strtol(fields[1]);
        let top_p = parse_float_strict(fields[2])?;
        let min_p = parse_float_strict(fields[3])?;
        let seed = strtol(fields[4]);
        Self::new(temperature, top_k, top_p, min_p, seed)
    }
}

/// Deterministic PRNG (splitmix64): a given `SAMPLE` seed reproduces the
/// token sequence, which the contract tests rely on (the reference's GPU
/// sampler is likewise deterministic per seed — `k_smpg_pick`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: i64) -> Self {
        Self::from_u64(seed as u64)
    }

    pub fn from_u64(seed: u64) -> Self {
        Self(seed)
    }

    /// One splitmix64 step (the reference's FNV/splitmix hash family —
    /// spec §5 quotes the same mixing constants).
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)` with 53-bit mantissa.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in `[0, 1)` with 24-bit mantissa (f32 precision).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// A sampled vocabulary token with the logprob the `T` line reports.
#[derive(Debug, Clone, PartialEq)]
pub struct SampledToken {
    /// Index into the logits slice (the vocabulary token id).
    pub index: usize,
    /// Natural log of the probability the token was sampled with.
    pub logprob: f64,
}

/// Natural log of `softmax(logits)[index]` (max-subtraction form for
/// stability). `index` must be in range (callers derive it from `sample`).
pub fn token_logprob(logits: &[f32], index: usize) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = logits
        .iter()
        .map(|&logit| (f64::from(logit) - max).exp())
        .sum();
    f64::from(logits[index]) - max - sum.ln()
}

/// Draw one token from `logits` under `params`. `None` when there is
/// nothing to sample from (empty slice, or no finite logits).
pub fn sample(logits: &[f32], params: &SampleParams, rng: &mut Rng) -> Option<SampledToken> {
    if logits.is_empty() {
        return None;
    }
    if params.temperature == 0.0 {
        // Greedy: first argmax (the reference's `k_argmax_*` kernels).
        let mut best = (0usize, f32::NEG_INFINITY);
        for (index, &logit) in logits.iter().enumerate() {
            if logit.is_finite() && logit > best.1 {
                best = (index, logit);
            }
        }
        if !best.1.is_finite() {
            return None;
        }
        return Some(SampledToken {
            index: best.0,
            logprob: token_logprob(logits, best.0),
        });
    }
    // Temperature-scaled softmax over the finite logits.
    let inverse_temperature = 1.0 / params.temperature;
    let scored: Vec<(usize, f64)> = logits
        .iter()
        .enumerate()
        .filter(|(_, logit)| logit.is_finite())
        .map(|(index, logit)| (index, f64::from(*logit) * inverse_temperature))
        .collect();
    if scored.is_empty() {
        return None;
    }
    let max = scored
        .iter()
        .map(|(_, score)| *score)
        .fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scored
        .iter()
        .map(|(_, score)| (score - max).exp())
        .collect();
    let total: f64 = exps.iter().sum();

    // top-k: keep the k highest scores (k = 0 keeps everything).
    let mut order: Vec<usize> = (0..scored.len()).collect();
    order.sort_by(|a, b| exps[*b].total_cmp(&exps[*a]));
    let keep = if params.top_k > 0 {
        (params.top_k as usize).min(scored.len())
    } else {
        scored.len()
    };
    let mut kept: Vec<usize> = order[..keep].to_vec();

    // min-p: drop tokens below min_p × the best surviving probability.
    if params.min_p > 0.0 && !kept.is_empty() {
        let best = exps[kept[0]];
        kept.retain(|&index| exps[index] >= params.min_p * best);
    }

    // top-p: keep the shortest descending prefix with cumulative mass
    // >= p (always at least one token).
    if params.top_p < 1.0 && !kept.is_empty() {
        let mut cumulative = 0.0;
        let mut prefix = 0usize;
        for &index in &kept {
            cumulative += exps[index] / total;
            prefix += 1;
            if cumulative >= params.top_p {
                break;
            }
        }
        kept.truncate(prefix.max(1));
    }

    // Roulette selection from the surviving renormalised distribution.
    let kept_total: f64 = kept.iter().map(|&index| exps[index]).sum();
    let mut spin = rng.next_f64() * kept_total;
    let mut chosen = *kept.last()?;
    for &index in &kept {
        spin -= exps[index];
        if spin < 0.0 {
            chosen = index;
            break;
        }
    }
    Some(SampledToken {
        index: scored[chosen].0,
        logprob: (exps[chosen] / kept_total).ln(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn greedy_params() -> SampleParams {
        SampleParams::greedy()
    }

    #[test]
    fn greedy_picks_the_first_argmax_and_reports_its_logprob() {
        let logits = [0.1f32, 0.4, 0.2];
        let token = sample(&logits, &greedy_params(), &mut Rng::new(0)).expect("sampled");
        assert_eq!(token.index, 1);
        // Independent softmax math: ln(e^0.4 / (e^0.1 + e^0.4 + e^0.2)).
        let expected = ((0.4f64).exp() / (0.1f64.exp() + 0.4f64.exp() + 0.2f64.exp())).ln();
        assert!(
            (token.logprob - expected).abs() < 1e-6,
            "{token:?} vs {expected}"
        );
    }

    #[test]
    fn greedy_prefers_the_first_index_on_ties() {
        let logits = [0.5f32, 0.5, 0.5];
        let token = sample(&logits, &greedy_params(), &mut Rng::new(0)).expect("sampled");
        assert_eq!(token.index, 0);
        assert!((token.logprob - (-(3.0f64).ln())).abs() < 1e-6);
    }

    #[test]
    fn greedy_ignores_the_rng_so_the_sequence_repeats_per_seed() {
        let logits = [1.0f32, 3.0, 2.0];
        let a: Vec<usize> = (0..10)
            .map(|_| {
                sample(&logits, &greedy_params(), &mut Rng::new(7))
                    .unwrap()
                    .index
            })
            .collect();
        let b: Vec<usize> = (0..10)
            .map(|_| {
                sample(&logits, &greedy_params(), &mut Rng::new(9))
                    .unwrap()
                    .index
            })
            .collect();
        assert!(a.iter().all(|&i| i == 1));
        assert_eq!(a, b);
    }

    #[test]
    fn empty_or_all_nan_logits_sample_nothing() {
        assert_eq!(sample(&[], &greedy_params(), &mut Rng::new(0)), None);
        assert_eq!(
            sample(&[f32::NAN], &greedy_params(), &mut Rng::new(0)),
            None
        );
        assert_eq!(
            sample(
                &[f32::NAN],
                &SampleParams::new(1.0, 0, 1.0, 0.0, 0).unwrap(),
                &mut Rng::new(0)
            ),
            None
        );
    }

    #[test]
    fn sampled_path_is_deterministic_per_seed() {
        let logits = [0.5f32, -0.25, 1.25, 0.0, -1.0];
        let params = SampleParams::new(1.0, 0, 1.0, 0.0, 42).expect("valid");
        let draw = |seed| {
            let mut rng = Rng::new(seed);
            (0..50)
                .map(|_| sample(&logits, &params, &mut rng).unwrap().index)
                .collect::<Vec<_>>()
        };
        assert_eq!(draw(42), draw(42), "same seed must repeat the sequence");
        assert_ne!(draw(42), draw(43), "different seeds must diverge");
    }

    #[test]
    fn top_k_one_always_picks_the_best_token() {
        let logits = [0.5f32, 2.0, 1.0];
        let params = SampleParams::new(1.0, 1, 1.0, 0.0, 3).expect("valid");
        let mut rng = Rng::new(3);
        for _ in 0..20 {
            assert_eq!(sample(&logits, &params, &mut rng).unwrap().index, 1);
        }
    }

    #[test]
    fn top_p_excludes_the_nucleus_tail() {
        // p ≈ [0.367, 0.367, 0.367, 2.7e-9]: the 0.99 nucleus is the first
        // three tokens; the tail must never be drawn.
        let logits = [1.0f32, 1.0, 1.0, -20.0];
        let params = SampleParams::new(1.0, 0, 0.99, 0.0, 5).expect("valid");
        for seed in 0..200 {
            let token = sample(&logits, &params, &mut Rng::new(seed)).unwrap();
            assert_ne!(token.index, 3, "seed {seed} drew the excluded tail");
        }
    }

    #[test]
    fn min_p_drops_weak_tokens() {
        // p ≈ [0.626, 0.338, 0.004]: min_p 0.1 keeps the first two only.
        let logits = [2.0f32, 1.0, -5.0];
        let params = SampleParams::new(1.0, 0, 1.0, 0.1, 5).expect("valid");
        for seed in 0..200 {
            let token = sample(&logits, &params, &mut Rng::new(seed)).unwrap();
            assert_ne!(token.index, 2, "seed {seed} drew a below-min_p token");
        }
    }

    #[test]
    fn uniform_distribution_covers_every_token() {
        let logits = [0.0f32, 0.0, 0.0, 0.0];
        let params = SampleParams::new(1.0, 0, 1.0, 0.0, 11).expect("valid");
        let mut counts = [0usize; 4];
        let mut rng = Rng::new(11);
        for _ in 0..400 {
            counts[sample(&logits, &params, &mut rng).unwrap().index] += 1;
        }
        for count in counts {
            assert!((50..350).contains(&count), "unbalanced draw {counts:?}");
        }
    }

    #[test]
    fn parse_sample_accepts_the_reference_clause_shape() {
        // `SAMPLE %.9g %d %.9g %.9g %d` as serve_api.py:1958 emits it.
        let params =
            SampleParams::parse_sample(["0.7", "20", "0.95", "0.05", "42"]).expect("parses");
        assert_eq!(
            params,
            SampleParams {
                temperature: 0.7,
                top_k: 20,
                top_p: 0.95,
                min_p: 0.05,
                seed: 42,
            }
        );
    }

    #[test]
    fn parse_sample_enforces_the_reference_validation() {
        // `SAMPLE temperature outside (0, 2] or top_k < 0` (spec §6):
        assert!(SampleParams::new(3.0, 20, 0.95, 0.0, 0).is_none(), "t > 2");
        assert!(SampleParams::new(-1.0, 20, 0.95, 0.0, 0).is_none(), "t < 0");
        assert!(SampleParams::new(2.0, 20, 0.95, 0.0, 0).is_some(), "t == 2");
        assert!(
            SampleParams::new(0.0, 20, 0.95, 0.0, 0).is_some(),
            "t == 0 (greedy)"
        );
        assert!(SampleParams::new(0.7, -1, 0.95, 0.0, 0).is_none(), "k < 0");
        assert!(
            SampleParams::new(0.7, 0, 0.95, 0.0, 0).is_some(),
            "k == 0 (off)"
        );
    }

    #[test]
    fn rng_matches_the_splitmix64_reference_sequence() {
        // Pinned splitmix64 outputs for state 0 (the reference hash family,
        // spec §5 constants).
        let mut rng = Rng::new(0);
        assert_eq!(rng.next_u64(), 0xe220_a839_7b1d_cdaf);
        assert_eq!(rng.next_u64(), 0x6e78_9e6a_a1b9_65f4);
        assert_eq!(rng.next_u64(), 0x06c4_5d18_8009_454f);
        // next_f64 stays in [0, 1) with 53-bit precision.
        let mut rng = Rng::new(1);
        for _ in 0..1000 {
            let value = rng.next_f64();
            assert!((0.0..1.0).contains(&value), "{value} outside [0, 1)");
        }
    }
}
