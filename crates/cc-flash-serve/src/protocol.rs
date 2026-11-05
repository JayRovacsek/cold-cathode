//! The engine line protocol on the serve socket (port 8730).
//!
//! Contract sources, in order of trust:
//!
//! 1. The reference front-end parser, `serve_api.py` (shipped with the
//!    v0.18.21 image, reused as data per plan R4): `probe_live` (:1625–1657)
//!    sends `PING\n` and accepts only `PONG`; `_probe` (:1684–1710) sends
//!    `INFO\n` and accepts a reply whose first whitespace-separated token is
//!    `I` with at least six tokens; `_parse_info` (:1741–1772) maps the
//!    tokens to the capability object and reads the engine version from
//!    token 17 (the value `serve_api: version …, engine version …` prints at
//!    :6771).
//! 2. The reference's own reply format string,
//!    [`evidence/flash_serve/binutils-strings.txt`](../../../docs/analysis/evidence/flash_serve/binutils-strings.txt):3752:
//!    `I %d %d %d %d %d %d %d %lld %d %d %d %d 1 %g %lld %d %s %d %d %s %d %d %d %d %lld %s %d %d %s %d %d`
//!    — 32 tokens, field-for-field the object `_parse_info` builds (the
//!    `%s` at token 17 is the version string).
//! 3. `docs/analysis/specs/api.md` §1 (port 8730 speaks the line protocol)
//!    and its open question §10.2 ("Decide whether the harness stubs the
//!    engine or requires a live one") — this module resolves it: the port
//!    answers INFO itself, so the harness can probe a live ported engine
//!    instead of the fixed fake fixture (decision log 2026-11-03, review
//!    adjudication 5d).
//! 4. The `GEN` grammar and the `T`/`D`/`P`/`G`/`C` response lines,
//!    [`flash_serve-engine.md`](../../../docs/analysis/specs/flash_serve-engine.md)
//!    §8 (the `GEN` case `bcmp`-verified in the connection-handler
//!    decompile; the response formats from serve_api.py's parsers —
//!    `t_line_logprob` :1066–1074, `_done_dict` :1799–1852 instantiated
//!    through the reference's `D` format string
//!    [binutils-strings.txt:1620](../../../docs/analysis/evidence/flash_serve/binutils-strings.txt),
//!    `p_line_progress` :1062–1065 and `g_line` :1337–1348).
//!
//! The whole module is pure except for `serve_connection`/`serve_forever`
//! (which only move bytes) and `EngineInfo::from_serve_args` (which reads
//! the `--ck` file's 4-byte magic via `hgn::format_of` to report the real
//! `checkpoint_format`); `handle_line` is unit-testable without a socket
//! (its `GEN` path reads the loaded checkpoint through `crate::forward`,
//! whose stub contract lives there, and samples through
//! `crate::sampling`).

use std::io::{BufRead, BufReader, ErrorKind, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;
use std::time::Instant;

use crate::forward;
use crate::sampling::{self, SampleParams};
use crate::{env_strtod, env_var, fmt_g, hgn, sscanf_ints, ServeArgs};

/// Everything INFO reports that the port can derive from the parsed serve
/// configuration, the environment and the loaded checkpoint's format (the
/// reference answers from loaded weights; the port answers from its
/// configuration plus the `--ck` file's magic — see `hgn::format_of`).
#[derive(Debug, Clone, PartialEq)]
pub struct EngineInfo {
    /// Token 17 (`version`): the `HALOGEN_IMAGE_VERSION` the launcher pins
    /// (spec flash_serve.md env table: default `unknown`, max 31 chars,
    /// leading spaces trimmed).
    pub version: String,
    /// Token 3 (`ctx`): the `--ctx` value.
    pub ctx: i64,
    /// Token 5 (`default`): the effective max-tok (`--max-tok` clamped to
    /// `--ctx`, the same clamp `run_serve` applies).
    pub max_tok: i64,
    /// Token 10 (`kv_slots`): the `--slots` value.
    pub slots: i64,
    /// Token 11 (`slot_ctx`): each slot covers the full `--ctx` (the
    /// reference's banner describes `--slots` slots over one shared pool).
    pub slot_ctx: i64,
    /// Token 15 (`kv_pool`): the `--kv-pool` value (0 = auto, reported
    /// verbatim; the port does not allocate a pool to resolve it).
    pub kv_pool: i64,
    /// Token 14 (`rope_factor`): `HALOGEN_ROPE_YARN` (default 1.0),
    /// rendered with the reference's `%g`.
    pub rope_factor: f64,
    /// Tokens 18/19 (`pld_n`/`pld_k`): the `HALOGEN_PLD=N,K` pair (0,0 when
    /// unset or unparseable).
    pub pld_n: i64,
    pub pld_k: i64,
    /// Token 20 (`checkpoint_format`): the format of the `--ck` checkpoint
    /// as loaded by `run_serve` — `hgn` for an HGN1 file, `gguf` for a
    /// GGUF (served without resident weights, the port's GGUF load being
    /// repack-only). An unreadable or unrecognised file falls back to the
    /// reference's `_parse_info` default `hgn`.
    pub checkpoint_format: String,
}

impl EngineInfo {
    /// Build the INFO state from parsed serve args plus the environment
    /// (reads `HALOGEN_IMAGE_VERSION`, `HALOGEN_ROPE_YARN`, `HALOGEN_PLD`)
    /// and the `--ck` file's magic (`hgn::format_of`).
    pub fn from_serve_args(opts: &ServeArgs) -> Self {
        let (pld_n, pld_k) = pld_pair(env_var("HALOGEN_PLD"));
        EngineInfo {
            version: version_token(env_var("HALOGEN_IMAGE_VERSION")),
            ctx: opts.ctx,
            max_tok: opts.max_tok.min(opts.ctx),
            slots: opts.slots,
            slot_ctx: opts.ctx,
            kv_pool: opts.kv_pool,
            rope_factor: env_strtod("HALOGEN_ROPE_YARN", 1.0),
            pld_n,
            pld_k,
            checkpoint_format: hgn::format_of(&opts.ck).unwrap_or("hgn").to_string(),
        }
    }
}

/// The version token exactly as the reference reads
/// `HALOGEN_IMAGE_VERSION` (spec flash_serve.md env table: default
/// `unknown`, max 31 chars, leading spaces trimmed). The launcher exports
/// it pinned to the image version (`0.18.21`), so a live port answers with
/// that. An empty value also falls back to `unknown`: an empty token would
/// collapse the whitespace-separated reply and shift every later field out
/// of `_parse_info`'s positions.
pub fn version_token(raw: Option<String>) -> String {
    match raw {
        Some(value) => {
            let trimmed = value.trim_start_matches(' ');
            if trimmed.is_empty() {
                "unknown".to_string()
            } else {
                trimmed.chars().take(31).collect()
            }
        }
        None => "unknown".to_string(),
    }
}

/// The `HALOGEN_PLD=N,K` pair as INFO tokens 18/19 (`sscanf "%d,%d"`, the
/// same parse `validate_env` uses); unset or unparseable → `(0, 0)` (off).
pub fn pld_pair(raw: Option<String>) -> (i64, i64) {
    match raw.and_then(|v| sscanf_ints(&v, 2)) {
        Some(pair) => (pair[0], pair[1]),
        None => (0, 0),
    }
}

/// Render the INFO reply (no trailing newline) from the reference's format
/// string (evidence binutils-strings.txt:3752, field names per
/// serve_api.py `_parse_info`):
///
/// ```text
/// I %d %d %d %d %d %d %d %lld %d %d %d %d 1 %g %lld %d %s %d %d %s %d %d %d %d %lld %s %d %d %s %d %d
/// ```
///
/// The port fills the configuration-derived fields from `EngineInfo` —
/// including token 20 (`checkpoint_format`), now the real format of the
/// loaded `--ck` file — and the remaining weight-derived fields with the
/// honest zero/off values (the port's checkpoint load is the GPU-gated
/// deferred path): `mtp`, `draft_head`, `drafter_weights` (their source is
/// the MTP head checkpoint, which the serve path does not load), `dflash2`,
/// `cache_mb`, `cache_align` (the port allocates no KV pool), `vision`,
/// `grammar`, `composable`, `cc_floor`, `cc_seam`, `cc_bytes`, `tools`,
/// `keep_end`, `masks_images` are 0; `cache_mode` is 1, `indexer_budget`
/// 2048, `kv_cache` "bf16" and `cc_fingerprint` "-" — the same defaults
/// serve_api's own `_fallback_info` (:1712–1717) assumes. Token 13
/// (`sampling`) is the literal `1` of the reference format string.
pub fn info_line(info: &EngineInfo) -> String {
    format!(
        "I 0 0 {ctx} 0 {max_tok} 0 0 0 0 {slots} {slot_ctx} 1 1 {rope} {kv_pool} 0 {version} {pld_n} {pld_k} {checkpoint_format} 0 0 0 0 0 - 2048 0 bf16 0 0",
        ctx = info.ctx,
        max_tok = info.max_tok,
        slots = info.slots,
        slot_ctx = info.slot_ctx,
        rope = fmt_g(info.rope_factor),
        kv_pool = info.kv_pool,
        version = info.version,
        pld_n = info.pld_n,
        pld_k = info.pld_k,
        checkpoint_format = info.checkpoint_format,
    )
}

/// A parsed `GEN` request (spec flash_serve-engine.md §8 grammar,
/// serve_api.py:1952–1990). The port acts on the core fields plus the
/// optional `SAMPLE` clause and `PROGRESS` flag; the remaining optional
/// clauses (`drafter`, `PENALTY`, `SNAP`/`SNAP2`/`SNAP3`, `LEAD3`, `IMG`,
/// `SCHEMA`, `AFTER`, `ESCAPE`, `TOOLS`, `THINK`, `GUARD`, `AGUARD`,
/// `SEG`, `AGENT`, `NOSTORE`, `KEEP`, `KEPEND`) belong to subsystems the
/// port defers (MTP §4, grammar, vision, composable context §5) and are
/// accepted and ignored — the front-end only emits most of them when the
/// INFO reply advertises those capabilities, which `info_line` reports
/// off.
#[derive(Debug, Clone, PartialEq)]
pub struct GenRequest {
    /// The front-end's monotonically increasing request id (`req`).
    pub req: i64,
    /// Tokens to generate (already clamped by the front-end).
    pub max_tokens: i64,
    /// The eos token ids (`n_eos` + list).
    pub eos: Vec<i64>,
    /// The prompt token ids (`n_ids` + list).
    pub ids: Vec<i64>,
    /// The `SAMPLE %.9g %d %.9g %.9g %d` clause (greedy when absent).
    pub sample: Option<SampleParams>,
    /// The `PROGRESS` flag: request `P` progress lines.
    pub progress: bool,
}

/// Parse a `GEN` line ([`GenRequest`] for the grammar). `None` when the
/// core fields (req, max_tokens, eos list, ids list) or a present
/// `SAMPLE` clause are malformed — the port then closes the connection
/// rather than guessing (the reference has no error-reply string for
/// requests it cannot parse).
pub fn parse_gen(line: &str) -> Option<GenRequest> {
    let mut words = line.split_whitespace();
    if words.next()? != "GEN" {
        return None;
    }
    let req = words.next()?.parse::<i64>().ok()?;
    let max_tokens = words.next()?.parse::<i64>().ok()?;
    let n_eos = words.next()?.parse::<i64>().ok()?;
    if max_tokens < 0 || n_eos < 0 {
        return None;
    }
    let eos = count_ids(&mut words, n_eos as usize)?;
    let n_ids = words.next()?.parse::<i64>().ok()?;
    if n_ids < 0 {
        return None;
    }
    let ids = count_ids(&mut words, n_ids as usize)?;
    // Optional clauses in the grammar's emission order: the port acts on
    // SAMPLE and PROGRESS and ignores the rest (GenRequest docs).
    let mut sample = None;
    let mut progress = false;
    while let Some(word) = words.next() {
        if word == "SAMPLE" {
            let fields = [
                words.next()?,
                words.next()?,
                words.next()?,
                words.next()?,
                words.next()?,
            ];
            sample = Some(SampleParams::parse_sample(fields)?);
        } else if word == "PROGRESS" {
            progress = true;
        }
    }
    Some(GenRequest {
        req,
        max_tokens,
        eos,
        ids,
        sample,
        progress,
    })
}

/// Read exactly `count` whitespace-separated token ids (`None` when the
/// line ends early or a field is not an integer). No `with_capacity`: a
/// malformed line declaring a huge count (past `isize::MAX` would panic,
/// a multi-GB count would over-allocate before the parse fails) must
/// close like the reference ("its GEN does not parse",
/// binutils-strings.txt:1617), not abort the connection thread.
fn count_ids<'a>(words: &mut impl Iterator<Item = &'a str>, count: usize) -> Option<Vec<i64>> {
    let mut ids = Vec::new();
    for _ in 0..count {
        ids.push(words.next()?.parse::<i64>().ok()?);
    }
    Some(ids)
}

/// The `D` (done) line, field-for-field the reference's format string
/// ([binutils-strings.txt:1620](../../../docs/analysis/evidence/flash_serve/binutils-strings.txt),
/// field names per serve_api `_done_dict` :1799–1852):
///
/// ```text
/// D %ld %s %d %d %.1f %.1f %d %ld %ld %d %ld %ld %ld %ld %d %d %d %d %d %d %d %d %d %d %.1f %ld %ld %ld %ld %ld %ld %ld
/// ```
///
/// The port fills `req`, `reason`, `n_prompt`, `n_gen`, `decode_ms` and
/// `max_tokens` from the request and its output; `prefill_ms` is 0.0 (the
/// stub performs no separate prefill pass) and the drafter, speculative,
/// PLD, composable, think, disk, draft-vocab and grammar counters are 0 —
/// their subsystems are the deferred phases, matching the honest zeros
/// `info_line` reports.
///
/// `message` (empty for normal completions) is appended after the numeric
/// fields: the reference's refusal shape (`D <req> error <msg>`, spec §8)
/// carries the reason text there — serve_api's `engine_refusal`
/// (:1350–1363) skips the leading numeric fields of `parts[7:]` and
/// renders `the engine refused this request: <message>`.
// The `D` line carries 33 fixed fields (engine spec §8); the parameter
// list mirrors them, so the arity is inherent to the format.
#[allow(clippy::too_many_arguments)]
fn done_line(
    req: i64,
    reason: &str,
    n_prompt: i64,
    n_gen: usize,
    max_tokens: i64,
    decode_ms: f64,
    message: &str,
    rounds: usize,
    commit: usize,
) -> String {
    // Fields [8]/[9] (after the leading `0` at [7]) carry the MTP
    // speculative accounting (engine spec §8); the MTP-off path passes
    // `0 0`, keeping the historical bytes.
    let line = format!(
        "D {req} {reason} {n_prompt} {n_gen} 0.0 {decode_ms:.1} \
         0 {rounds} {commit} 0 0 0 0 0 0 0 {max_tokens} 0 0 0 0 0 0 0 0.0 0 0 0 0 0 0 0"
    );
    if message.is_empty() {
        line
    } else {
        format!("{line} {message}")
    }
}

/// The refusal message the port appends to the `D <req> error …` line
/// when a request cannot be admitted (spec §3: the engine reserves
/// prompt plus max_tokens inside the context).
const OVER_CONTEXT_MESSAGE: &str = "prompt plus max_tokens exceed the context";

/// The reference's generation progress interval in seconds: 30.0, pinned
/// by the live reference journal (review 🟡-1): the journal excerpt
/// archived at
/// [`evidence/flash_serve/live-journal-excerpts.md`](../../../docs/analysis/evidence/flash_serve/live-journal-excerpts.md)
/// shows the reference emitting its generation lines at 30.0 s and 60.0 s
/// of one request (`req 2 generated 1748 tokens, 30.0 s` / `3383 tokens,
/// 60.0 s`). The double `0x403e000000000000` (30.0) in the decompiled
/// `main`'s config initialisation is *not* evidence for this constant:
/// both occurrences assign `local_19a0`, the `HALOGEN_ADMIT_AGE_S`
/// admission-age default (the earlier commit's attribution was wrong; the
/// value itself is right).
const GEN_PROGRESS_INTERVAL_SECS: f64 = 30.0;

/// The reference's generation progress line, instantiated:
/// `flash_serve: req %ld generated %d tokens, %.1f s`
/// ([`evidence/flash_serve/binutils-strings.txt`](../../../docs/analysis/evidence/flash_serve/binutils-strings.txt):1237,
/// observed live as `flash_serve: req 2 generated 1748 tokens, 30.0 s`).
/// The `req` id is the `GEN` request's own id (the reference numbers
/// requests at the protocol layer and every per-request log line names
/// it); `elapsed_s` renders with the format string's `%.1f`.
pub(crate) fn generated_line(req: i64, n_gen: usize, elapsed_s: f64) -> String {
    format!("flash_serve: req {req} generated {n_gen} tokens, {elapsed_s:.1} s")
}

/// The reference's prompt-load progress line, instantiated:
/// `flash_serve: req %ld prefill %d/%d tokens, 0.0 s%s`
/// ([`evidence/flash_serve/binutils-strings.txt`](../../../docs/analysis/evidence/flash_serve/binutils-strings.txt):601).
/// The binary carries a second prefill format string at :2400,
/// `flash_serve: req %ld prefill %d/%d tokens, %.1f s` (measured time, no
/// `%s`): :2400 is the right variant for cold/measured prefills, while
/// :601's fixed `0.0 s` is presumably the cache-hit variant (review
/// 💡-4). Both render identically for the port's instant prefill, so the
/// port keeps the :601 shape. The port's prefill is the prompt itself (the
/// same reading the `P` progress line takes in [`gen_response_with`]), so
/// the line is emitted once with `processed == total` and the format
/// string's fixed `0.0 s` time; the trailing `%s` suffix's variants are
/// not pinned in the evidence, so the port emits it empty (flagged in the
/// port notes).
pub(crate) fn prefill_line(req: i64, processed: i64, total: i64) -> String {
    format!("flash_serve: req {req} prefill {processed}/{total} tokens, 0.0 s")
}

/// The per-request cadence state for [`generated_line`]: emits the
/// reference's line once per elapsed [`GEN_PROGRESS_INTERVAL_SECS`]
/// interval while the token loop runs (the reference's journal shows the
/// 30.0 s / 60.0 s ticks; a request that ends before a tick emits
/// none). Split into a pure `line_at` so the cadence is unit-testable
/// without waiting on a wall clock.
#[derive(Debug)]
pub(crate) struct GenProgress {
    /// The `GEN` request id the lines name (`req %ld`).
    req: i64,
    /// Elapsed seconds at which the next line is due.
    next_s: f64,
}

impl GenProgress {
    pub(crate) fn new(req: i64) -> Self {
        GenProgress {
            req,
            next_s: GEN_PROGRESS_INTERVAL_SECS,
        }
    }

    /// The reference's line when `elapsed_s` has reached the next 30 s
    /// tick (`None` otherwise). The tick boundary advances to the next
    /// whole interval multiple, so an elapsed 30.4 s emits the 30 s tick
    /// and the next fires at 60 s (the reference's observed cadence).
    pub(crate) fn line_at(&mut self, n_gen: usize, elapsed_s: f64) -> Option<String> {
        (elapsed_s >= self.next_s).then(|| {
            // The next whole interval: (floor(elapsed / 30) + 1) · 30 s.
            self.next_s = ((elapsed_s / GEN_PROGRESS_INTERVAL_SECS).floor() + 1.0)
                * GEN_PROGRESS_INTERVAL_SECS;
            generated_line(self.req, n_gen, elapsed_s)
        })
    }

    /// [`Self::line_at`] against the wall clock (`started` = when the
    /// request's generation began, the same instant the `D` line's
    /// `decode_ms` measures).
    pub(crate) fn line(&mut self, n_gen: usize, started: Instant) -> Option<String> {
        self.line_at(n_gen, started.elapsed().as_secs_f64())
    }
}

/// The per-server state the protocol answers from (review 🟡-6): the
/// INFO configuration, the forward pass over the loaded checkpoint and
/// the MTP engine constructed once at server startup — replacing the
/// per-`GEN` rebuild that reloaded and revalidated the head `.hgn` on
/// every request (`gen_response` used to build a fresh
/// `forward::Forward` + `MtpEngine::from_env` per call).
#[derive(Debug)]
pub struct ServeState<'a> {
    /// The INFO capability state (see [`info_line`]).
    pub info: EngineInfo,
    /// The forward pass over the loaded checkpoint; `None` when the
    /// serve path has no resident weights (the GGUF repack-only case —
    /// `GEN` then closes the connection, see [`handle_line`]). The
    /// caller owns the forward pass (`run_serve` keeps it alive for the
    /// server's lifetime) so the MTP engine can hold the
    /// `&'a Forward<'a>` its API requires.
    pub forward: Option<&'a forward::Forward<'a>>,
    /// The MTP engine built from the environment at startup (`None`
    /// when MTP is off or the head is not staged,
    /// [`crate::mtp::MtpEngine::from_env`]).
    pub mtp: Option<crate::mtp::MtpEngine<'a>>,
}

impl<'a> ServeState<'a> {
    /// Build the serve state from the INFO configuration and the
    /// resident forward pass: the environment's MTP engine is
    /// constructed once (review 🟡-6).
    pub fn new(info: EngineInfo, forward: Option<&'a forward::Forward<'a>>) -> Self {
        let mtp = forward.and_then(crate::mtp::MtpEngine::from_env);
        ServeState { info, forward, mtp }
    }
}

/// A destination for the engine's reply lines, delivered line by line as
/// soon as each is ready. The reference never batches a `GEN` reply: the
/// connection-handler decompile
/// ([`evidence/flash_serve/engine/conn-handler-FUN_01ee0e20.json`](../../../docs/analysis/evidence/flash_serve/engine/conn-handler-FUN_01ee0e20.json))
/// shows the raw `send(fd, buf + sent, len - sent, …)` write loop with no
/// user-space buffering, and the live front-end consumes `T` lines as
/// they arrive (serve_api's SSE forwarding with keepalive/gap tracking —
/// `0 SSE frames and 0 chars sent in 18.2s, … 1 keepalives` — only makes
/// sense against an engine that streams each token line while generation
/// runs, spec flash_serve-engine.md §8). The port mirrors that contract:
/// [`SocketSink`] writes and flushes every line on emit, and a
/// `Vec<String>` collects the same lines for the string-returning API
/// ([`handle_line`], [`gen_response_with`]).
pub trait ReplySink {
    /// Deliver one complete protocol line (without its newline) as soon as
    /// it is ready. `false` means the sink is gone (a dead socket): the
    /// caller must stop generating rather than keep producing lines.
    fn emit(&mut self, line: &str) -> bool;
}

impl ReplySink for Vec<String> {
    fn emit(&mut self, line: &str) -> bool {
        self.push(line.to_string());
        true
    }
}

/// The production [`ReplySink`]: every `emit` writes `line` + `\n` to the
/// socket and flushes, so each token reaches the front-end the moment the
/// engine samples it (the reference's per-line `send` contract — the
/// earlier shape built the whole `GEN` reply in memory and wrote it once
/// after generation completed, so a slow first token starved the
/// front-end's SSE stream; see [`gen_response_into`]).
pub struct SocketSink<W: Write> {
    writer: W,
}

impl<W: Write> SocketSink<W> {
    pub fn new(writer: W) -> Self {
        SocketSink { writer }
    }
}

impl<W: Write> ReplySink for SocketSink<W> {
    fn emit(&mut self, line: &str) -> bool {
        writeln!(self.writer, "{line}").is_ok() && self.writer.flush().is_ok()
    }
}

/// The `GEN` answer with the state's forward pass and the optional MTP
/// engine injected (the MTP wiring, `crate::mtp`): the reference reserves
/// prompt plus max_tokens inside the context (spec §3 banner), so a
/// request whose `n_prompt + max_tokens` exceeds `--ctx` answers the
/// refusal shape ([`done_line`] with reason `error` and
/// [`OVER_CONTEXT_MESSAGE`]) instead of generating. Otherwise, when `mtp`
/// is `Some` and the request has a prompt (the reference's warm-sequence
/// rule, spec flash_serve-mtp-kv-npu.md §1), generation runs the
/// speculative draft/verify loop and the `D` line carries its
/// `rounds`/`commit` accounting; otherwise the single-token loop
/// ([`generate_tokens`]) answers with `0 0` accounting, stopping early
/// with reason `stop` when a drawn token is in the request's eos set and
/// with reason `length` when the budget runs out. The `G` grammar line is
/// only emitted by the reference's grammar subsystem (off here, per
/// `info_line`), and `C` lines answer `CSTAT`, not `GEN`.
///
/// Collects the reply into a `String` (the [`handle_line`] shape); the
/// serve path streams it line by line through [`gen_response_into`]
/// instead.
pub(crate) fn gen_response_with(
    request: &GenRequest,
    info: &EngineInfo,
    forward: &forward::Forward<'_>,
    mtp: Option<&crate::mtp::MtpEngine<'_>>,
) -> String {
    let mut lines: Vec<String> = Vec::new();
    gen_response_into(request, info, forward, mtp, &mut lines);
    lines.join("\n")
}

/// The [`gen_response_with`] body streaming each reply line into `sink`
/// the moment it is ready (the reference's per-token write contract,
/// [`ReplySink`]): the `P` line, then every `T` line as its token is
/// sampled, then the `D` line. Returns `false` when the sink died
/// mid-generation (a client hangup): generation stops with the lines
/// already delivered, matching the reference's behaviour once its
/// `send` fails.
///
/// CPU token latency (H2, measured by reasoning from bytes): the real
/// forward pass reads the full 66 687 351 992 B quantised weight set
/// once per token (scalar bf16 gemv across the 48-layer stack,
/// `forward::logits`), so at single-core memory bandwidth (~10–20 GB/s,
/// less with scalar dequant) one token costs seconds to a minute on
/// CPU — the first streamed `T` line of a cold CPU request legitimately
/// arrives that late; the old batched reply hid even completed tokens
/// until the whole generation finished.
fn gen_response_into(
    request: &GenRequest,
    info: &EngineInfo,
    forward: &forward::Forward<'_>,
    mtp: Option<&crate::mtp::MtpEngine<'_>>,
    sink: &mut dyn ReplySink,
) -> bool {
    let n_prompt = request.ids.len() as i64;
    if request
        .max_tokens
        .checked_add(n_prompt)
        .is_none_or(|total| total > info.ctx)
    {
        return sink.emit(&done_line(
            request.req,
            "error",
            n_prompt,
            0,
            request.max_tokens,
            0.0,
            OVER_CONTEXT_MESSAGE,
            0,
            0,
        ));
    }
    let params = request.sample.clone().unwrap_or_else(SampleParams::greedy);
    if request.progress {
        // `P <req> <processed> <total> <cache> <time_ms>` (serve_api
        // p_line_progress :1062–1065): the stub's prefill is the prompt
        // itself — processed/total carry its length, cache and time 0.
        if !sink.emit(&format!("P {} {n_prompt} {n_prompt} 0 0", request.req)) {
            return false;
        }
    }
    // The reference's prompt-load progress line (see [`prefill_line`]):
    // the port's prefill completes with the prompt itself, so the line
    // is emitted once with processed == total (HALOGEN_LOG-gated).
    if n_prompt > 0 {
        crate::log_info(&prefill_line(request.req, n_prompt, n_prompt));
    }
    let started = Instant::now();
    let (n_gen, reason, rounds, commit) =
        match mtp.filter(|_| !request.ids.is_empty()).and_then(|engine| {
            engine.generate_into(request, &params, request.max_tokens as usize, started, sink)
        }) {
            Some(generation) => generation,
            None => {
                let (n_gen, reason) = generate_tokens(
                    request,
                    &params,
                    request.max_tokens as usize,
                    started,
                    &|context| forward.logits(context),
                    sink,
                );
                (n_gen, reason, 0, 0)
            }
        };
    let decode_ms = started.elapsed().as_secs_f64() * 1000.0;
    sink.emit(&done_line(
        request.req,
        reason,
        n_prompt,
        n_gen,
        request.max_tokens,
        decode_ms,
        "",
        rounds,
        commit,
    ))
}

/// The generation loop of [`gen_response_into`]: draws up to `limit` tokens
/// using `logits` as the logits source — the forward pass in production,
/// injectable so the `sample`-returns-`None` stop is testable. Each
/// drawn token's `T` line goes to `sink` immediately (write + flush on
/// the socket, [`ReplySink`]), so the front-end sees every token the
/// moment it is sampled; a dead sink (client hangup) stops the loop with
/// the lines already delivered. Returns the generated-token count and
/// the done reason (`stop` when a drawn token is in the request's eos
/// set, `length` otherwise — also used when the logits source runs dry
/// before the budget).
/// `started` is the instant the request's generation began (the same one
/// the `D` line's `decode_ms` measures); the loop emits the reference's
/// 30 s generation ticks ([`GenProgress`], HALOGEN_LOG-gated) as the
/// token count advances.
fn generate_tokens(
    request: &GenRequest,
    params: &SampleParams,
    limit: usize,
    started: Instant,
    logits: &dyn Fn(&[i64]) -> Vec<f32>,
    sink: &mut dyn ReplySink,
) -> (usize, &'static str) {
    let mut rng = sampling::Rng::new(params.seed);
    let mut context = request.ids.clone();
    let mut n_gen = 0usize;
    let mut reason = "length";
    let mut progress = GenProgress::new(request.req);
    for _ in 0..limit {
        let Some(token) = sampling::sample(&logits(&context), params, &mut rng) else {
            break;
        };
        // `T <req> <token> <logprob>` (serve_api t_line_logprob
        // :1066–1074); no top-k pairs — the port requests none.
        let token_id = token.index as i64;
        if !sink.emit(&format!(
            "T {} {} {}",
            request.req,
            token.index,
            fmt_g(token.logprob)
        )) {
            // The client hung up mid-stream: stop generating (the
            // reference's send failure ends the request the same way).
            break;
        }
        context.push(token_id);
        n_gen += 1;
        if let Some(tick) = progress.line(n_gen, started) {
            crate::log_info(&tick);
        }
        if request.eos.contains(&token_id) {
            reason = "stop";
            break;
        }
    }
    (n_gen, reason)
}

/// Answer one protocol line; `Some(reply)` (without the trailing newline)
/// is written back and the connection stays open, `None` closes it. A
/// `GEN` reply is the streamed response ([`gen_response_with`] collects
/// the `T` lines plus the `D` line; the serve path streams them per line
/// through [`handle_line_into`] instead of collecting).
///
/// Commands, per the reference's verb set (the `<$PING`/`<$INFO` compare
/// strings, binutils-strings.txt:282377–282378, the `GEN` case
/// `bcmp`-verified in the connection-handler decompile, spec
/// flash_serve-engine.md §8, and serve_api's own verb usage: `PING`,
/// `INFO`, `GEN`, `HREL`, `VOCAB`, `MERGES`):
///
/// * `PING` → `PONG` (serve_api `probe_live` accepts only `PONG`).
/// * `INFO` → the capability line (see [`info_line`]).
/// * `GEN …` → the generation response (see [`gen_response_with`]): the
///   request grammar is parsed ([`parse_gen`]), the forward entry point
///   and sampling layer produce the tokens from the state's forward pass
///   (review 🟡-6), and the reply is framed as the `T`/`P`/`D` lines
///   serve_api's `_demux` (:1659–1679) consumes. With
///   `state.forward == None` (a GGUF served repack-only — the port's
///   deferred weight load) the previous deferred behaviour stands and
///   the connection closes; a malformed `GEN` line closes too.
/// * `HREL`, `VOCAB`, `MERGES` and every unknown verb → close. The
///   reference has no error-reply string for unrecognised verbs (it logs
///   `queued connection closed without a request`, :3761, when a connection
///   dies without one), so closing is the evidenced default; the front-end
///   reconnects via `_ensure` (:1774–1779).
pub fn handle_line(request: &str, state: &ServeState<'_>) -> Option<String> {
    let line = request.trim_end_matches(['\r', '\n']);
    let verb = line.split_whitespace().next()?;
    match verb {
        "PING" => Some("PONG".to_string()),
        "INFO" => Some(info_line(&state.info)),
        "GEN" => {
            let request = parse_gen(line)?;
            let forward = state.forward?;
            Some(gen_response_with(
                &request,
                &state.info,
                forward,
                state.mtp.as_ref(),
            ))
        }
        _ => None,
    }
}

/// The [`handle_line`] body streaming every reply line into `sink` as it
/// becomes ready ([`ReplySink`]): `PING`/`INFO` emit their single line,
/// `GEN` streams its `P`/`T`/`D` lines ([`gen_response_into`]). Returns
/// `false` when the connection must close (an unknown or malformed verb,
/// or a dead sink).
fn handle_line_into(line: &str, state: &ServeState<'_>, sink: &mut dyn ReplySink) -> bool {
    let line = line.trim_end_matches(['\r', '\n']);
    let Some(verb) = line.split_whitespace().next() else {
        return false;
    };
    match verb {
        "PING" => sink.emit("PONG"),
        "INFO" => sink.emit(&info_line(&state.info)),
        "GEN" => {
            let (request, forward) = match (parse_gen(line), state.forward) {
                (Some(request), Some(forward)) => (request, forward),
                _ => return false,
            };
            gen_response_into(&request, &state.info, forward, state.mtp.as_ref(), sink)
        }
        _ => false,
    }
}

/// Serve one connection to completion: answer lines until the protocol
/// closes it (or EOF). Every reply line is written and flushed the moment
/// it is ready — `PING`/`INFO` as one line each, a `GEN` reply as its
/// `P`/`T`/`D` lines streamed through [`handle_line_into`] (the
/// reference's per-line `send` contract, [`ReplySink`]); `state` is the
/// shared serve state (the INFO config, the forward pass and the MTP
/// engine — review 🟡-6).
pub fn serve_connection(stream: TcpStream, state: &ServeState<'_>) {
    let mut writer = match stream.try_clone() {
        Ok(clone) => clone,
        Err(_) => return,
    };
    let mut reader = BufReader::new(stream);
    let mut sink = SocketSink::new(&mut writer);
    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        if !handle_line_into(&line, state, &mut sink) {
            break;
        }
    }
}

/// Classify an error returned by [`TcpListener::accept`] (and therefore by
/// `incoming()`): `true` means the error is transient and the accept loop
/// must keep running, `false` means the listening socket is gone and the
/// loop should stop.
///
/// The contract source is the reference's accept loop
/// ([`evidence/flash_serve/decompile/main-FUN_01ede7e0.json`](../../../docs/analysis/evidence/flash_serve/decompile/main-FUN_01ede7e0.json),
/// decompiled `main`): after `accept()` returns -1 it reads `errno` and
/// breaks only on `EINTR` (4) — every other errno, including
/// `ECONNABORTED` (103), falls through and re-accepts. The reference
/// breaks on `EINTR` because its SIGINT/SIGTERM handlers (installed in the
/// same decompile) make `EINTR` the shutdown signal; the port installs no
/// signal handlers, so `Interrupted` is treated as retryable here, while
/// the genuinely fatal kinds (dead fd, wrong family, permission) end the
/// loop as they would in the reference once its socket closes.
pub fn is_transient_accept_error(kind: ErrorKind) -> bool {
    matches!(
        kind,
        ErrorKind::ConnectionAborted
            | ErrorKind::ConnectionReset
            | ErrorKind::Interrupted
            | ErrorKind::WouldBlock
    )
}

/// The accept loop behind `run_serve`: bind succeeded, so answer every
/// connection on its own thread (the reference serves `--slots` concurrent
/// sessions; one thread per connection keeps a persistent probe connection
/// from blocking new clients). `state` (the INFO config, the forward pass
/// and the MTP engine — review 🟡-6) is shared with every connection
/// thread through the scope. Blocks until `incoming()` fails fatally.
///
/// Transient accept errors must not end the loop: on Linux `accept()`
/// returns `ECONNABORTED` when a peer resets a queued connection before
/// `accept()` completes it, and `std::net::TcpListener::incoming()`
/// surfaces that as `Err(io::Error)` (kind `ConnectionAborted`) instead of
/// skipping it — [`is_transient_accept_error`] keeps the loop alive there,
/// matching the reference (which re-accepts on every errno except the
/// `EINTR` its signal handlers use for shutdown). The earlier
/// break-on-any-error version killed the live engine: `serve_api`
/// reconnects per request (open → INFO → GEN → close), the first
/// ECONNABORTED ended the loop, `run_serve` returned 0 and the process
/// exited silently — the observed symptom was the 66 GB weight `Vec`
/// disappearing with no panic or error line after the front-end's first
/// reconnect cycle.
pub fn serve_forever(listener: &TcpListener, state: &ServeState<'_>) {
    // A scoped `thread::spawn` (review 🟡-6): the connection threads
    // borrow the shared `ServeState` instead of cloning the old
    // `(info, checkpoint)` pair per connection.
    thread::scope(|scope| {
        for conn in listener.incoming() {
            let stream = match conn {
                Ok(stream) => stream,
                Err(err) if is_transient_accept_error(err.kind()) => continue,
                Err(_) => break,
            };
            scope.spawn(move || serve_connection(stream, state));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{SocketAddr, TcpListener};
    use std::sync::{mpsc, Arc};
    use std::time::Duration;

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

    /// A small `.hgn` fixture whose only tensor is `embed_tokens.weight`
    /// with `dims [vocab, 4]` (layout per `hgn::TensorEntry`), giving the
    /// forward stub a checkpoint-derived vocab of `vocab`.
    fn fixture_checkpoint(vocab: u64) -> hgn::Checkpoint {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"HGN1");
        bytes.extend_from_slice(&2u32.to_le_bytes()); // version
        bytes.extend_from_slice(&1u64.to_le_bytes()); // entry_count
        bytes.extend_from_slice(&104u64.to_le_bytes()); // table_offset
        bytes.extend_from_slice(&264u64.to_le_bytes()); // data_offset
        bytes.extend_from_slice(&264u64.to_le_bytes()); // declared size
        bytes.extend_from_slice(&[0u8; 64]); // model_id
        let mut entry = [0u8; hgn::ENTRY_BYTES];
        entry[..19].copy_from_slice(b"embed_tokens.weight");
        entry[96..100].copy_from_slice(&0u32.to_le_bytes()); // dtype
        entry[100..104].copy_from_slice(&2u32.to_le_bytes()); // kind
        entry[104..112].copy_from_slice(&vocab.to_le_bytes());
        entry[112..120].copy_from_slice(&4u64.to_le_bytes());
        entry[136..144].copy_from_slice(&264u64.to_le_bytes());
        bytes.extend_from_slice(&entry);
        hgn::load_prefix("ccfs-protocol-fixture.hgn", bytes, 264).expect("fixture loads")
    }

    #[test]
    fn ping_answers_pong() {
        let state = ServeState::new(fixture_info(), None);
        assert_eq!(handle_line("PING", &state).as_deref(), Some("PONG"));
        assert_eq!(handle_line("PING\n", &state).as_deref(), Some("PONG"));
        assert_eq!(handle_line("PING\r\n", &state).as_deref(), Some("PONG"));
        // The reference compares the verb exactly (serve_api sends `PING\n`).
        assert_eq!(handle_line("ping", &state), None);
        assert_eq!(handle_line("PINGO", &state), None);
    }

    #[test]
    fn info_reply_is_the_reference_format_string_instantiated() {
        // The exact 32-token line the reference's format string
        // (binutils-strings.txt:3752) produces for the fixture state;
        // token 17 (the `%s`) carries the version.
        let state = ServeState::new(fixture_info(), None);
        let reply = handle_line("INFO", &state).expect("INFO answers");
        assert_eq!(
            reply,
            "I 0 0 32768 0 8192 0 0 0 0 4 32768 1 1 1 0 0 0.18.21 0 0 hgn 0 0 0 0 0 - 2048 0 bf16 0 0"
        );
        // The exact assertions serve_api._probe makes: first token `I` and
        // at least six tokens; _parse_info's version field (token 17)
        // round-trips.
        let parts: Vec<&str> = reply.split_whitespace().collect();
        assert_eq!(parts[0], "I");
        assert!(parts.len() >= 6);
        assert_eq!(parts.len(), 32);
        assert_eq!(parts[17], "0.18.21");
        assert_eq!(parts[3], "32768"); // ctx
        assert_eq!(parts[5], "8192"); // default max-tok
        assert_eq!(parts[10], "4"); // kv_slots
    }

    #[test]
    fn info_reply_reflects_the_parsed_configuration() {
        let info = EngineInfo {
            version: "0.18.21".to_string(),
            ctx: 4096,
            max_tok: 10,
            slots: 2,
            slot_ctx: 4096,
            kv_pool: 4096,
            rope_factor: 4.0,
            pld_n: 1,
            pld_k: 2,
            checkpoint_format: "hgn".to_string(),
        };
        let line = info_line(&info);
        let parts: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(parts[3], "4096"); // ctx
        assert_eq!(parts[5], "10"); // default max-tok
        assert_eq!(parts[10], "2"); // kv_slots
        assert_eq!(parts[11], "4096"); // slot_ctx
        assert_eq!(parts[14], "4"); // rope_factor, %g-rendered
        assert_eq!(parts[15], "4096"); // kv_pool
        assert_eq!(parts[17], "0.18.21"); // version
        assert_eq!(parts[18], "1"); // pld_n
        assert_eq!(parts[19], "2"); // pld_k
        assert_eq!(parts[20], "hgn"); // checkpoint_format
    }

    #[test]
    fn info_reply_reports_the_real_checkpoint_format() {
        // Token 20 now carries the loaded checkpoint's actual format: a
        // GGUF served without resident weights reports `gguf` (the
        // reference's server reads GGUF checkpoints directly).
        let info = EngineInfo {
            checkpoint_format: "gguf".to_string(),
            ..fixture_info()
        };
        let line = info_line(&info);
        let parts: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(parts[20], "gguf");
    }

    #[test]
    fn gen_without_resident_weights_closes() {
        // The GGUF serve path (repack-only, no resident weights) keeps the
        // deferred behaviour: close, never fake.
        let state = ServeState::new(fixture_info(), None);
        assert_eq!(handle_line("GEN 1 1 0 0", &state), None);
    }

    #[test]
    fn unknown_verbs_close_the_connection() {
        let ck = fixture_checkpoint(8);
        let forward = forward::Forward::new(&ck);
        let state = ServeState::new(fixture_info(), Some(&forward));
        assert_eq!(handle_line("HREL 1", &state), None);
        assert_eq!(handle_line("VOCAB", &state), None);
        assert_eq!(handle_line("MERGES 2 aGk=", &state), None);
        assert_eq!(handle_line("BOGUS", &state), None);
        assert_eq!(handle_line("", &state), None);
        assert_eq!(handle_line("\n", &state), None);
    }

    #[test]
    fn gen_request_parses_the_reference_grammar() {
        // The exact shape serve_api.py:1952–1990 emits: core fields, then
        // the drafter, SAMPLE clause, penalty and PROGRESS flag.
        let request = parse_gen(
            "GEN 12 8 2 151645 151643 3 1 2 3 7 SAMPLE 0.7 20 0.95 0.05 42 PENALTY 1.1 0.6 1.05 PROGRESS",
        )
        .expect("parses");
        assert_eq!(request.req, 12);
        assert_eq!(request.max_tokens, 8);
        assert_eq!(request.eos, vec![151645, 151643]);
        assert_eq!(request.ids, vec![1, 2, 3]);
        assert_eq!(
            request.sample,
            Some(SampleParams {
                temperature: 0.7,
                top_k: 20,
                top_p: 0.95,
                min_p: 0.05,
                seed: 42,
            })
        );
        assert!(request.progress);
        // The bare core (no optional clauses) parses too.
        let request = parse_gen("GEN 1 1 0 0").expect("minimal GEN parses");
        assert_eq!(request.req, 1);
        assert_eq!(request.max_tokens, 1);
        assert!(request.eos.is_empty());
        assert!(request.ids.is_empty());
        assert_eq!(request.sample, None);
        assert!(!request.progress);
    }

    #[test]
    fn malformed_gen_lines_close() {
        let ck = fixture_checkpoint(8);
        let forward = forward::Forward::new(&ck);
        let state = ServeState::new(fixture_info(), Some(&forward));
        for line in [
            "GEN",
            "GEN {\"prompt\":\"hi\"}",
            "GEN x 1 0 0",
            "GEN 1",
            "GEN 1 5",
            "GEN 1 5 0",
            "GEN 1 5 1 0",
            "GEN 1 -1 0 0",
            "GEN 1 5 -1 0",
            // A malformed SAMPLE clause (temperature > 2) fails the request.
            "GEN 1 1 0 0 SAMPLE 3 20 0.95 0.05 42",
            // A SAMPLE clause with fewer than five fields fails the request.
            "GEN 1 1 0 0 SAMPLE 0.7 20",
        ] {
            assert_eq!(handle_line(line, &state), None, "{line} must close");
        }
    }

    #[test]
    fn gen_reply_streams_t_lines_and_a_done_line() {
        let ck = fixture_checkpoint(8);
        let forward = forward::Forward::new(&ck);
        let state = ServeState::new(fixture_info(), Some(&forward));
        let reply = handle_line("GEN 7 3 0 2 10 20", &state).expect("GEN answers");
        let lines: Vec<&str> = reply.split('\n').collect();
        assert_eq!(lines.len(), 4, "3 T lines + the D line: {reply:?}");
        // The T lines carry the greedy tokens the forward stub produces,
        // each with the sampling layer's logprob.
        let mut context = vec![10, 20];
        for line in &lines[..3] {
            let logits = forward.logits(&context);
            let expected =
                sampling::sample(&logits, &SampleParams::greedy(), &mut sampling::Rng::new(0))
                    .expect("sampled");
            assert_eq!(
                *line,
                format!("T 7 {} {}", expected.index, fmt_g(expected.logprob))
            );
            context.push(expected.index as i64);
        }
        // The D line matches the reference's 33-token format string
        // (binutils-strings.txt:1620, fields per serve_api _done_dict).
        let done: Vec<&str> = lines[3].split_whitespace().collect();
        assert_eq!(done.len(), 33, "D line {reply:?}");
        assert_eq!(done[0], "D");
        assert_eq!(done[1], "7"); // req
        assert_eq!(done[2], "length"); // reason: budget exhausted, no eos
        assert_eq!(done[3], "2"); // n_prompt
        assert_eq!(done[4], "3"); // n_gen
        assert_eq!(done[17], "3"); // max_tokens
    }

    #[test]
    fn gen_reply_stops_at_an_eos_token() {
        let ck = fixture_checkpoint(8);
        let forward = forward::Forward::new(&ck);
        let state = ServeState::new(fixture_info(), Some(&forward));
        let logits = forward.logits(&[10, 20]);
        let first_token =
            sampling::sample(&logits, &SampleParams::greedy(), &mut sampling::Rng::new(0))
                .expect("sampled")
                .index;
        let line = format!("GEN 1 5 1 {first_token} 2 10 20");
        let reply = handle_line(&line, &state).expect("GEN answers");
        let lines: Vec<&str> = reply.split('\n').collect();
        assert_eq!(lines.len(), 2, "one T line then the D line: {reply:?}");
        let done: Vec<&str> = lines[1].split_whitespace().collect();
        assert_eq!(done[2], "stop"); // reason: eos drawn
        assert_eq!(done[4], "1"); // n_gen
    }

    #[test]
    fn gen_reply_reports_a_progress_line_when_requested() {
        let ck = fixture_checkpoint(8);
        let forward = forward::Forward::new(&ck);
        let state = ServeState::new(fixture_info(), Some(&forward));
        let reply = handle_line("GEN 1 1 0 0 PROGRESS", &state).expect("GEN answers");
        let lines: Vec<&str> = reply.split('\n').collect();
        assert_eq!(lines.len(), 3, "P + T + D: {reply:?}");
        // `P <req> <processed> <total> <cache> <time_ms>` (serve_api
        // p_line_progress :1062–1065).
        assert_eq!(lines[0], "P 1 0 0 0 0");
    }

    #[test]
    fn generated_line_and_ticks_match_the_reference_evidence() {
        // The reference's format string (binutils-strings.txt:1237)
        // instantiated with the live journal's own values:
        // `flash_serve: req 2 generated 1748 tokens, 30.0 s` and
        // `flash_serve: req 2 generated 3383 tokens, 60.0 s`.
        assert_eq!(
            generated_line(2, 1748, 30.04),
            "flash_serve: req 2 generated 1748 tokens, 30.0 s"
        );
        assert_eq!(
            generated_line(2, 3383, 60.04),
            "flash_serve: req 2 generated 3383 tokens, 60.0 s"
        );
        // Cadence (the live journal's 30.0 s / 60.0 s ticks, see
        // GEN_PROGRESS_INTERVAL_SECS): silent before 30 s, one line per
        // elapsed 30 s interval after.
        let mut progress = GenProgress::new(2);
        assert_eq!(progress.line_at(1, 0.0), None);
        assert_eq!(progress.line_at(1747, 29.99), None);
        assert_eq!(
            progress.line_at(1748, 30.04).as_deref(),
            Some("flash_serve: req 2 generated 1748 tokens, 30.0 s")
        );
        // The next tick boundary is 60 s: nothing between, the next line
        // fires at 60 s with the then-current token count.
        assert_eq!(progress.line_at(2000, 45.0), None);
        assert_eq!(
            progress.line_at(3383, 60.04).as_deref(),
            Some("flash_serve: req 2 generated 3383 tokens, 60.0 s")
        );
        assert_eq!(progress.line_at(3384, 61.0), None);
    }

    #[test]
    fn prefill_line_matches_the_reference_evidence() {
        // The reference's format string (binutils-strings.txt:601)
        // instantiated; the port's prefill is the prompt itself (the P
        // line's reading), so processed == total and the fixed 0.0 s.
        assert_eq!(
            prefill_line(2, 10, 10),
            "flash_serve: req 2 prefill 10/10 tokens, 0.0 s"
        );
        assert_eq!(
            prefill_line(7, 1, 1),
            "flash_serve: req 7 prefill 1/1 tokens, 0.0 s"
        );
    }

    #[test]
    fn gen_reply_never_carries_log_lines_on_the_socket() {
        // The progress lines are the reference's journal output (the
        // port's HALOGEN_LOG channel), never part of the socket reply: a
        // two-token GEN answers with exactly its T lines and D line.
        let ck = fixture_checkpoint(8);
        let forward = forward::Forward::new(&ck);
        let state = ServeState::new(fixture_info(), Some(&forward));
        let reply = handle_line("GEN 2 2 0 1 5", &state).expect("GEN answers");
        assert!(
            !reply.contains("flash_serve:"),
            "log lines never ride the socket: {reply:?}"
        );
        let lines: Vec<&str> = reply.split('\n').collect();
        assert_eq!(lines.len(), 3, "two T lines + the D line: {reply:?}");
        // The exact T lines (the pattern of
        // serve_connection_streams_a_gen_reply_over_a_socket and
        // gen_reply_streams_t_lines_and_a_done_line): each carries the
        // greedy token the forward stub produces for the running context,
        // with the sampling layer's logprob.
        let mut context = vec![5];
        for line in &lines[..2] {
            let logits = forward.logits(&context);
            let expected =
                sampling::sample(&logits, &SampleParams::greedy(), &mut sampling::Rng::new(0))
                    .expect("sampled");
            assert_eq!(
                *line,
                format!("T 2 {} {}", expected.index, fmt_g(expected.logprob))
            );
            context.push(expected.index as i64);
        }
        // The D line's fixed fields (decode_ms measures the wall clock, so
        // the sibling test's field-by-field pattern rather than a full
        // string match).
        let done: Vec<&str> = lines[2].split_whitespace().collect();
        assert_eq!(done.len(), 33, "D line {reply:?}");
        assert_eq!(done[0], "D");
        assert_eq!(done[1], "2"); // req
        assert_eq!(done[2], "length"); // reason: budget exhausted, no eos
        assert_eq!(done[3], "1"); // n_prompt
        assert_eq!(done[4], "2"); // n_gen
        assert_eq!(done[17], "2"); // max_tokens
    }

    #[test]
    fn gen_reply_refuses_a_prompt_over_the_context() {
        // The reference reserves prompt + max_tokens inside the context
        // (spec §3); the refusal is the full D-line shape with reason
        // `error` and the message appended — serve_api's engine_refusal
        // (:1350–1363) skips the numeric fields and renders
        // "the engine refused this request: <message>".
        let info = EngineInfo {
            ctx: 16,
            slot_ctx: 16,
            ..fixture_info()
        };
        let ck = fixture_checkpoint(8);
        let forward = forward::Forward::new(&ck);
        let state = ServeState::new(info, Some(&forward));
        let ids = (0..17).map(|i| i.to_string()).collect::<Vec<_>>().join(" ");
        let reply = handle_line(&format!("GEN 4 1 0 17 {ids}"), &state).expect("GEN answers");
        let parts: Vec<&str> = reply.split_whitespace().collect();
        assert_eq!(
            parts.len(),
            39,
            "33 D-line fields + 6-word message: {reply:?}"
        );
        assert_eq!(parts[0], "D");
        assert_eq!(parts[1], "4"); // req
        assert_eq!(parts[2], "error");
        assert_eq!(parts[3], "17"); // n_prompt
        assert_eq!(parts[4], "0"); // n_gen
        assert_eq!(parts[17], "1"); // max_tokens echoed from the request
        assert_eq!(
            parts[33..].join(" "),
            "prompt plus max_tokens exceed the context"
        );
    }

    #[test]
    fn gen_reply_refuses_when_max_tokens_overflows_the_context() {
        // A huge max_tokens (i64::MAX) plus one prompt id overflows the
        // i64 sum; the checked add must refuse, not panic or clamp.
        let ck = fixture_checkpoint(8);
        let forward = forward::Forward::new(&ck);
        let state = ServeState::new(fixture_info(), Some(&forward));
        let reply = handle_line("GEN 3 9223372036854775807 0 1 5", &state).expect("GEN answers");
        let parts: Vec<&str> = reply.split_whitespace().collect();
        assert_eq!(parts[2], "error");
        assert_eq!(parts[3], "1"); // n_prompt
        assert_eq!(parts[17], "9223372036854775807"); // max_tokens echoed
    }

    #[test]
    fn gen_reply_with_zero_max_tokens_answers_done_only() {
        let ck = fixture_checkpoint(8);
        let forward = forward::Forward::new(&ck);
        let state = ServeState::new(fixture_info(), Some(&forward));
        let reply = handle_line("GEN 9 0 0 2 10 20", &state).expect("GEN answers");
        assert!(!reply.contains('\n'), "D-only reply: {reply:?}");
        let parts: Vec<&str> = reply.split_whitespace().collect();
        assert_eq!(parts.len(), 33, "D line {reply:?}");
        assert_eq!(parts[0], "D");
        assert_eq!(parts[1], "9"); // req
        assert_eq!(parts[2], "length"); // reason: budget exhausted at once
        assert_eq!(parts[3], "2"); // n_prompt
        assert_eq!(parts[4], "0"); // n_gen
        assert_eq!(parts[17], "0"); // max_tokens
    }

    #[test]
    fn generate_tokens_stops_when_the_logits_source_is_empty() {
        // The `sample`-returns-`None` stop (empty logits): no T lines,
        // n_gen 0, reason stays `length`.
        let request = parse_gen("GEN 1 3 0 0").expect("parses");
        let mut lines: Vec<String> = Vec::new();
        let (n_gen, reason) = generate_tokens(
            &request,
            &SampleParams::greedy(),
            3,
            Instant::now(),
            &|_| Vec::new(),
            &mut lines,
        );
        assert!(lines.is_empty(), "T lines: {lines:?}");
        assert_eq!(n_gen, 0);
        assert_eq!(reason, "length");
    }

    #[test]
    fn gen_reply_honours_the_sample_clause() {
        let ck = fixture_checkpoint(8);
        let forward = forward::Forward::new(&ck);
        let state = ServeState::new(fixture_info(), Some(&forward));
        let reply =
            handle_line("GEN 2 1 0 1 5 SAMPLE 0.7 20 0.95 0.05 42", &state).expect("GEN answers");
        let lines: Vec<&str> = reply.split('\n').collect();
        // The sampled token matches the sampling layer run directly with
        // the clause's seed (the port's determinism contract).
        let expected = sampling::sample(
            &forward.logits(&[5]),
            &SampleParams::new(0.7, 20, 0.95, 0.05, 42).expect("valid"),
            &mut sampling::Rng::new(42),
        )
        .expect("sampled");
        assert_eq!(
            lines[0],
            format!("T 2 {} {}", expected.index, fmt_g(expected.logprob))
        );
        assert_eq!(lines[1].split_whitespace().nth(2), Some("length"));
    }

    #[test]
    fn version_token_follows_the_reference_env_rule() {
        // Pure (no process-env mutation, so it never races the lib.rs env
        // tests): the rule is spec flash_serve.md env table — default
        // `unknown`, max 31 chars, leading spaces trimmed.
        assert_eq!(version_token(None), "unknown");
        assert_eq!(version_token(Some("0.18.21".to_string())), "0.18.21");
        assert_eq!(version_token(Some("  0.18.21".to_string())), "0.18.21");
        assert_eq!(
            version_token(Some("0123456789012345678901234567890123456789".to_string())),
            "0123456789012345678901234567890" // truncated to 31 chars
        );
        // Empty falls back to unknown (an empty token would shift the
        // whitespace-separated INFO fields).
        assert_eq!(version_token(Some(String::new())), "unknown");
    }

    #[test]
    fn pld_pair_parses_the_env_pair() {
        assert_eq!(pld_pair(None), (0, 0));
        assert_eq!(pld_pair(Some("bad".to_string())), (0, 0));
        assert_eq!(pld_pair(Some("1,2".to_string())), (1, 2));
    }

    #[test]
    fn transient_accept_errors_keep_the_loop_alive() {
        // ECONNABORTED (a peer resetting a queued connection) is the
        // documented Linux case that std's incoming() surfaces as an Err;
        // the reference re-accepts on it (main-FUN_01ede7e0: only EINTR
        // breaks). The rest of the transient set is retried for the same
        // reason.
        for kind in [
            ErrorKind::ConnectionAborted,
            ErrorKind::ConnectionReset,
            ErrorKind::Interrupted,
            ErrorKind::WouldBlock,
        ] {
            assert!(is_transient_accept_error(kind), "{kind:?} must be retried");
        }
    }

    #[test]
    fn fatal_accept_errors_stop_the_loop() {
        // A dead or unusable listening socket cannot recover by retrying.
        // (The task's `BadFileDescriptor`/`AddressFamily`/`AddressInUse`
        // names are not in this toolchain's std, so the fatal set is
        // exercised through the variants it does have.)
        for kind in [
            ErrorKind::NotConnected,
            ErrorKind::PermissionDenied,
            ErrorKind::ConnectionRefused,
            ErrorKind::BrokenPipe,
            ErrorKind::NotFound,
            ErrorKind::InvalidInput,
            ErrorKind::AddrInUse,
            ErrorKind::AddrNotAvailable,
            ErrorKind::AlreadyExists,
            ErrorKind::TimedOut,
            ErrorKind::WriteZero,
        ] {
            assert!(
                !is_transient_accept_error(kind),
                "{kind:?} must stop the loop"
            );
        }
    }

    #[test]
    fn serve_connection_answers_the_line_protocol_over_a_socket() {
        let info = fixture_info();
        let expected_info = info_line(&info);
        let listener = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0))).expect("bind");
        let addr = listener.local_addr().expect("addr");
        thread::spawn(move || {
            let state = ServeState::new(info, None);
            serve_forever(&listener, &state)
        });
        let mut client = TcpStream::connect(addr).expect("connect");
        client
            .set_read_timeout(Some(Duration::from_secs(5)))
            .expect("timeout");
        client.write_all(b"PING\nINFO\n").expect("write");
        let mut reader = BufReader::new(client.try_clone().expect("clone"));
        let mut ping = String::new();
        reader.read_line(&mut ping).expect("read PONG");
        assert_eq!(ping, "PONG\n");
        let mut info_reply = String::new();
        reader.read_line(&mut info_reply).expect("read INFO reply");
        assert_eq!(info_reply, format!("{expected_info}\n"));
        // The client dropping below gives the server thread EOF.
        drop(client);
    }

    #[test]
    fn sequential_probes_are_answered_while_a_connection_is_open_but_idle() {
        // Covers the probe pattern at the serve_forever level (review
        // 🟡-2): the launcher's wait_for_engine probe and serve_api's
        // capability probe each hold a connection while they wait for a
        // reply, and the engine must answer every one of them. One
        // open-but-idle connection (a probe that connected but has not
        // sent its line — exactly what a connect-only readiness check
        // leaves behind) must not stop two further sequential INFO probes
        // from being served: serve_forever answers each connection on its
        // own thread, so a connection that blocks in read_line never
        // queues the next ones behind it (the single-threaded accept loop
        // this replaced served one connection at a time, and the idle
        // connection starved the probes behind it).
        //
        // Scope: this test spawns serve_forever directly, so it exercises
        // the per-connection threading from the earlier refactor and
        // passes equally on code before the run_serve bind-order fix — it
        // does not cover that fix itself. The bind-order fix lives in
        // `run_serve` (lib.rs), whose startup sequence is not cleanly
        // extractable into a testable `build_state_then_bind` helper
        // (ServeState borrows the Forward, which borrows the Checkpoint —
        // returning the pair from a helper would need a self-referential
        // struct — and the real load needs a GPU and the checkpoint file),
        // so the ordering there is upheld by the code comment above the
        // bind instead.
        let info = fixture_info();
        let expected_info = info_line(&info);
        let listener = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0))).expect("bind");
        let addr = listener.local_addr().expect("addr");
        thread::spawn(move || {
            let state = ServeState::new(info, None);
            serve_forever(&listener, &state)
        });
        // The idle connection: connected, accepted, nothing sent yet.
        let mut idle = TcpStream::connect(addr).expect("idle connect");
        // Two sequential probe connections while the idle one is open.
        for probe in 0..2 {
            let mut client = TcpStream::connect(addr).expect("probe connect");
            client
                .set_read_timeout(Some(Duration::from_secs(5)))
                .expect("timeout");
            client.write_all(b"INFO\n").expect("write INFO");
            let mut reply = String::new();
            BufReader::new(client.try_clone().expect("clone"))
                .read_line(&mut reply)
                .unwrap_or_else(|err| panic!("probe {probe} must be answered while idle: {err}"));
            assert_eq!(reply, format!("{expected_info}\n"), "probe {probe} reply");
        }
        // The idle connection still gets its answer once it sends.
        idle.set_read_timeout(Some(Duration::from_secs(5)))
            .expect("idle timeout");
        idle.write_all(b"PING\n").expect("idle write");
        let mut idle_reader = BufReader::new(idle.try_clone().expect("idle clone"));
        let mut pong = String::new();
        idle_reader.read_line(&mut pong).expect("idle PONG");
        assert_eq!(pong, "PONG\n");
    }

    #[test]
    fn serve_connection_streams_a_gen_reply_over_a_socket() {
        // The wired-up GEN path: one GEN request answers its whole streamed
        // response (T line + D line) on the same connection.
        let info = fixture_info();
        let checkpoint = Arc::new(fixture_checkpoint(8));
        let listener = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0))).expect("bind");
        let addr = listener.local_addr().expect("addr");
        thread::spawn(move || {
            let forward = forward::Forward::new(&checkpoint);
            let state = ServeState::new(info, Some(&forward));
            serve_forever(&listener, &state)
        });
        let mut client = TcpStream::connect(addr).expect("connect");
        client
            .set_read_timeout(Some(Duration::from_secs(5)))
            .expect("timeout");
        client.write_all(b"GEN 1 1 0 0\n").expect("write");
        let mut reader = BufReader::new(client.try_clone().expect("clone"));
        let mut token_line = String::new();
        reader.read_line(&mut token_line).expect("read T line");
        assert!(token_line.starts_with("T 1 "), "{token_line:?}");
        let mut done_line = String::new();
        reader.read_line(&mut done_line).expect("read D line");
        assert!(done_line.starts_with("D 1 length "), "{done_line:?}");
        drop(client);
    }

    /// A [`ReplySink`] that forwards every emitted line to the test over
    /// an mpsc channel, so lines can be observed while the generation
    /// loop is still running.
    struct ChannelSink(mpsc::Sender<String>);

    impl ReplySink for ChannelSink {
        fn emit(&mut self, line: &str) -> bool {
            self.0.send(line.to_string()).is_ok()
        }
    }

    #[test]
    fn generate_tokens_streams_each_t_line_before_the_loop_finishes() {
        // The incremental-delivery proof (the 2026-11-06 live symptom:
        // serve_api saw `0 SSE frames and 0 chars sent in 18.2s` because
        // the old engine collected every T line and wrote the reply only
        // after generation completed). The logits source pauses inside
        // its second call until the test releases it, so the first T
        // line can only be observed while the loop is still running.
        let request = parse_gen("GEN 1 2 0 0").expect("parses");
        let (line_tx, line_rx) = mpsc::channel::<String>();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let generator = thread::spawn(move || {
            let calls = std::cell::Cell::new(0usize);
            let logits = move |_: &[i64]| -> Vec<f32> {
                calls.set(calls.get() + 1);
                if calls.get() == 2 {
                    // Pause after token 1: block until the test has read
                    // the first T line and releases the run.
                    let _ = release_rx.recv();
                }
                vec![0.0, -1.0] // greedy argmax: token 0
            };
            let mut sink = ChannelSink(line_tx);
            generate_tokens(
                &request,
                &SampleParams::greedy(),
                2,
                Instant::now(),
                &logits,
                &mut sink,
            )
        });
        // The first T line arrives while the generator is still paused
        // inside the second token's logits call — the old collect-then-
        // return shape could only produce it after the loop ended.
        let first = line_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("the first T line arrives while generation is still running");
        assert!(first.starts_with("T 1 "), "{first:?}");
        // Nothing else exists yet: the second token is not sampled.
        assert!(
            line_rx.recv_timeout(Duration::from_millis(100)).is_err(),
            "no further line may be emitted while generation is paused"
        );
        // Release the generator: the second T line follows, then the
        // loop ends with the full budget.
        release_tx.send(()).expect("release the paused generator");
        let second = line_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("the second T line after the release");
        assert!(second.starts_with("T 1 "), "{second:?}");
        assert_eq!(generator.join().expect("generator thread"), (2, "length"));
        // Exactly two T lines were ever emitted (no D line: that belongs
        // to gen_response_into, which the socket tests cover).
        assert!(line_rx.recv_timeout(Duration::from_millis(100)).is_err());
    }

    #[test]
    fn socket_sink_writes_and_flushes_every_line_immediately() {
        // The write-per-line half of the contract (the reference's raw
        // `send` per line, conn-handler decompile): each emit reaches the
        // peer while the writer is still open — the old code built the
        // whole reply in memory and flushed once at the end.
        let listener = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0))).expect("bind");
        let addr = listener.local_addr().expect("addr");
        let mut client = TcpStream::connect(addr).expect("connect");
        client
            .set_read_timeout(Some(Duration::from_secs(5)))
            .expect("timeout");
        let server = listener.accept().expect("accept").0;
        let mut sink = SocketSink::new(server);
        assert!(sink.emit("T 1 5 -0.5"), "emit to a live socket");
        let mut line = String::new();
        BufReader::new(&mut client)
            .read_line(&mut line)
            .expect("the line is readable while the sink is still open");
        assert_eq!(line, "T 1 5 -0.5\n");
    }
}
