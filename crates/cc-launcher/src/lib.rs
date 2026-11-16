//! `cold-cathode` implementation library (plan §3: `cc-launcher` →
//! `cold-cathode`): Rust port of the halogen-flash-server launcher. The
//! binary target (`src/main.rs`) is a thin wrapper over this crate so the
//! contract-test harness can depend on it and receive
//! `CARGO_BIN_EXE_cold-cathode`.
//!
//! Frozen contract: `docs/analysis/specs/launcher.md` — the transcription of
//! the v0.18.21 makeWrapper (`$out/bin/halogen-flash-server`) and the
//! `$out/libexec/halogen-entrypoint.sh` it execs. The spec's recommendations
//! are followed verbatim: the usage strings keep the literal program name
//! `entrypoint.sh`, and `HALOGEN_IMAGE_VERSION` is a hard assignment
//! (`0.18.21`, not overridable from the environment).
//!
//! Layer 1 (the makeWrapper env contract) is applied first: every variable
//! from the spec's verbatim wrapper is exported with `${VAR:-default}`
//! semantics (a set-but-empty variable also takes the default). The three
//! path defaults where the plan's Appendix A and the live wrapper differ
//! (`HALOGEN_MATMUL_TUNING_FILE`, `HALOGEN_DRAFT_VOCAB_FILE`, `_hg_hf`) use
//! the live wrapper values: the spec declares the live wrapper the contract
//! ("Variance vs the plan's Appendix A").
//!
//! # v0.20.5 launcher additions (objectives O1–O8)
//!
//! The vendored package is bumped to v0.20.5; the entrypoint additions of
//! that release are ported here: the `HALOGEN_AGENT_GUARD` mode parsing and
//! model injection (O1), the v0.20.5 `cache_gib` reply-end accounting (O2,
//! exposed as `cache_gib` while the rest of `kv_budget_note` stays
//! deferred), the new `FE_ONLY_FLAGS` entries (O3), the `--host ::` API bind
//! (O4), the NPU `vis`/`aud` part handling (O5), the bundled-XRT helpers
//! (O6), the NPU stall watchdog (O7) and the rc-76 host-floor backoff (O8).
//! The rest of the NPU orchestration (`start_npu_engine`, `npu_launch`,
//! `npu_preflight`) remains deferred; these helpers are its ported pieces.
//!
//! # Sibling binary resolution
//!
//! The reference launcher reaches its siblings through store paths and the
//! entrypoint's `_hg_*` override variables. This port resolves every sibling
//! binary in this order:
//!
//! 1. a per-binary `COLD_CATHODE_<NAME>` override — the name uppercased with
//!    `-` mapped to `_` (`COLD_CATHODE_FLASH_SERVE`,
//!    `COLD_CATHODE_HALOGEN_TOOLS`, `COLD_CATHODE_HALOGEN_NPU`,
//!    `COLD_CATHODE_HALOGEN_HEALTHCHECK`);
//! 2. the reference-compatible `_hg_*` override where one exists
//!    (`_hg_flash_serve`, `_hg_npu_bin`, `_hg_python`);
//! 3. a sibling of the running executable — `dirname(current_exe)/<name>`,
//!    which works in `target/debug/` (where cargo places all five workspace
//!    bins) and in `$out/bin` of the nix package;
//! 4. a `PATH` lookup.
//!
//! # Documented simplifications vs the entrypoint
//!
//! * `ulimit -c 0` (entrypoint line 6) is not applied: setting RLIMIT_CORE
//!   needs `setrlimit`, and this crate is std-only. It is invisible on
//!   stdout/stderr/exit codes.
//! * The deep pre-dispatch helpers (`maybe_download`, `check_gguf`,
//!   `check_sidecar`, `legacy_note`, `check_ngram_table`, `check_vision`,
//!   `kv_budget_note`, `gtt_note`, `engine_watchdog`, `stop_engine`,
//!   `wd_gtt_after_exit`, the NPU helpers) are ported structurally: the
//!   host preflight, checkpoint/tokenizer/draft-head checks and their exact
//!   messages are reproduced; the GPU-dependent orchestration (watchdog,
//!   NPU restart, signal traps) and the `memlock_note` weights-lock probe
//!   (whose reference body needs `ckpt_facts`, which execs `flash_serve
//!   --resident-gib`) are not. Those modes are `deferred` in the contract
//!   matrix (`side-effecting-launch`).
//! * `wait_for_engine` (entrypoint lines 749–776, invoked by the spec's
//!   `all` row at lines 1674–1720) is ported in full: a 2s TCP-connect poll
//!   to `127.0.0.1:$ENG_PORT`, the `HALOGEN_ENGINE_WAIT_S` timeout (0 or
//!   unset waits forever), the exact success / engine-exit / timeout
//!   wordings and the 60s heartbeat with its `wd_state` / `wd_gtt_gib`
//!   probes. The reference's optional log tail (`tail -30 "$log"`, line
//!   760) applies only when the caller redirects the engine's output
//!   (bench/sweep pass `/tmp/halogen-engine.log`); `all` has no log, so
//!   the port omits the argument.
//! * `exec` is emulated with `Command::status` (std has no `execve`): the
//!   child's exit code is returned through `main`; a missing program yields
//!   exit 127 and a non-executable one 126, matching bash.

#![warn(unsafe_op_in_unsafe_fn)]

use std::env;
use std::fs;
use std::io::{self, BufRead, BufReader, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{self, Child, Command, Stdio};
use std::thread::sleep;
use std::time::{Duration, Instant};

#[cfg(unix)]
use std::os::unix::process::ExitStatusExt;

/// Pinned by the wrapper as a hard assignment (spec: "pinned, not
/// overridable"; verified live: `HALOGEN_IMAGE_VERSION=9.9` is ignored).
const IMAGE_VERSION: &str = "0.18.21";

/// General usage, verbatim (spec: "Exact usage strings"; the literal
/// `entrypoint.sh` name is kept per the spec's recommendation).
const GENERAL_USAGE: &str =
    "usage: entrypoint.sh [all|engine|api|bench|sweep|convert IN.gguf OUT.hgn|inspect|verify|ppl|niah [FILE] ...]";

/// `convert` usage, verbatim (spec: "Exact usage strings", three lines).
const CONVERT_USAGE: &str = "usage: entrypoint.sh convert IN.gguf OUT.hgn\n  IN: any shard of a llama.cpp GGUF of Qwen3.8-Flash-Next (unsloth's, bartowski's, your own llama-quantize).\n  OUT: the .hgn to write (about 105 GiB for an IQ4_XS build; the table and the draft head are folded in).";

/// The retirement line for the `npu` mode (entrypoint line 1822, verbatim).
const NPU_RETIRED: &str = "halogen: there is no npu mode any more: the NPU's models run in the default mode beside the Flash model; set HALOGEN_NPU_MODELS (and drop the npu argument).";

/// Layer 1 of the spec: every `${VAR:-default}` export of the v0.18.21
/// makeWrapper, in wrapper order. `HALOGEN_IMAGE_VERSION` is handled
/// separately (hard assignment, not a `${VAR:-default}`).
const WRAPPER_DEFAULTS: &[(&str, &str)] = &[
    ("HALOGEN_ARCH", "gfx1151"),
    ("HIP_FORCE_DEV_KERNARG", "1"),
    ("HALOGEN_ATTN_FA", "64"),
    ("HALOGEN_PROMPT_CACHE", "2"),
    ("HALOGEN_PREFILL_CHUNK", "32768"),
    ("HALOGEN_KV_SLOTS", "4"),
    ("HALOGEN_CTX", "262144"),
    ("HALOGEN_MAX_TOK", "32768"),
    ("HALOGEN_MATMUL_ALGOS", "1"),
    ("HALOGEN_MATMUL_BUCKET", "1"),
    (
        "HALOGEN_MATMUL_TUNING_FILE",
        "/nix/store/fn6y2w20kbgjd8da8pwxbbc3gvmiycpy-halogen-flash-server-0.18.21/opt/halogen/flash-tune.plan",
    ),
    (
        "HALOGEN_DRAFT_VOCAB_FILE",
        "/nix/store/fn6y2w20kbgjd8da8pwxbbc3gvmiycpy-halogen-flash-server-0.18.21/opt/halogen/draft-vocab.lists",
    ),
    (
        "_hg_hf",
        "/nix/store/q4wf934v83ab6z79828l7wcgm8a8p7wg-python3-3.14.8-env/bin/hf",
    ),
    ("HALOGEN_TOKENIZER", "/tokenizer"),
    ("HALOGEN_PORT", "8730"),
    ("HALOGEN_BIND", "127.0.0.1"),
    ("HALOGEN_API_PORT", "8731"),
    ("HALOGEN_MAX_TOKENS_CAP", "65536"),
    ("HALOGEN_QUEUE_TIMEOUT", "3600"),
    (
        "HALOGEN_IMAGE_MODES",
        "all engine api bench sweep convert inspect verify ppl niah",
    ),
    ("PYTHONUNBUFFERED", "1"),
    ("HF_HUB_OFFLINE", "1"),
    ("TRANSFORMERS_VERBOSITY", "error"),
];

/// The 37 front-end-only variables (entrypoint lines 1002–1011) that every
/// `flash_serve` spawn strips via `env -u` (spec: "the port must reproduce
/// the `-u` list exactly").
const FE_ONLY_FLAGS: &[&str] = &[
    "HALOGEN_API_KEY",
    "HALOGEN_API_KEY_FILE",
    "HALOGEN_CLIENT_MARKERS",
    "HALOGEN_ENABLE_THINKING",
    "HALOGEN_ENGINE_PING_S",
    "HALOGEN_ENGINE_PROBE_S",
    "HALOGEN_FREQUENCY_PENALTY",
    "HALOGEN_KEEPALIVE_TIMEOUT",
    "HALOGEN_MAX_THINKING_TOKENS",
    "HALOGEN_MAX_TOKENS_DEFAULT",
    "HALOGEN_MIN_P",
    "HALOGEN_MODEL_ID",
    "HALOGEN_NEXT_TOKEN_S",
    "HALOGEN_PRESENCE_PENALTY",
    "HALOGEN_REASONING_EFFORT",
    "HALOGEN_RECORD_DAYS",
    "HALOGEN_RECORD_DIR",
    "HALOGEN_RECORD_GIB",
    "HALOGEN_RECORD_TEXT",
    "HALOGEN_REPETITION_PENALTY",
    "HALOGEN_RETRY_GRACE_S",
    "HALOGEN_SSE_KEEPALIVE_S",
    "HALOGEN_TEMPERATURE",
    "HALOGEN_TEMPLATE_CHECK",
    "HALOGEN_TEMPLATE_UNCHECKED",
    "HALOGEN_THINKING_ANSWER_ROOM",
    "HALOGEN_CHAT_TEMPLATE",
    "HALOGEN_STRICT_TOOLS",
    "HALOGEN_TOKENIZE_REUSE",
    "HALOGEN_TOP_K",
    "HALOGEN_TOP_P",
    "HALOGEN_CACHE_LEAD",
    "HALOGEN_TOOL_DEFER",
    // v0.20.5 (entrypoint lines 1031–1036):
    "HALOGEN_RENDER_ONCE",
    "HALOGEN_TOKENIZE_SPLIT",
    "HALOGEN_TOOL_NONE_MASK",
    "HALOGEN_TOKENIZER_NATIVE",
    "HALOGEN_FORK_AUTO_END_ONLY",
    "HALOGEN_FORK_AUTO_KEEPS",
    "HALOGEN_FORK_HANDLES",
    "HALOGEN_FORK_MAP",
    "HALOGEN_FORK_UNMATCHED_RELEASE",
    "HALOGEN_AGENT_GUARD",
    "HALOGEN_MODEL_NAME_CHECK",
];

/// The reference package root, used as the fallback for the Python
/// front-end's tool scripts until Phase 5 ships them under cold-cathode's
/// own `$out/share/halogen/tools` (spec: "Source → store path rewrites").
const REF_ROOT: &str = "/nix/store/fn6y2w20kbgjd8da8pwxbbc3gvmiycpy-halogen-flash-server-0.18.21";

/// The reference default for `_hg_python` (entrypoint line 1695).
const REF_PYTHON: &str =
    "/nix/store/q4wf934v83ab6z79828l7wcgm8a8p7wg-python3-3.14.8-env/bin/python3";

/// Entry point: `argv` includes `argv[0]`; returns the process exit code.
/// The binary target (`src/main.rs`) forwards its own argv here.
pub fn run(argv: &[String]) -> i32 {
    // Layer 1 — the makeWrapper env contract (spec §Layer 1).
    apply_wrapper_env();

    // Layer 2 pre-dispatch (spec §Layer 2), in entrypoint order.
    set_default("HSA_DISABLE_COREDUMP_ON_EXCEPTION", "1"); // line 5
                                                           // line 6 `ulimit -c 0`: not reproducible in std-only Rust (see module docs).
    if let Err(code) = agent_guard() {
        // v0.20.5 lines 8–27 (right after the coredump export).
        return code;
    }
    oom_adj_raise(); // lines 8–24
    set_default("HALOGEN_VERBOSE", "0"); // line 30
    let checkpoint = resolve_checkpoint("/models"); // lines 32–47
    let eng = match Engine::derive() {
        Ok(eng) => eng,
        Err(Exit) => return 2, // lines 73–77: the YaRN gate exits 2 before the banner
    };
    // line 1000: `[ -n "${HALOGEN_FORK+x}" ] || export HALOGEN_FORK=1` — a
    // set-only check (not `:-` semantics): even an empty value is kept.
    if env::var("HALOGEN_FORK").is_err() {
        env::set_var("HALOGEN_FORK", "1");
    }

    // Banner rule (lines 1666–1669). `${1:-all}` semantics: an empty mode
    // argument also selects `all`.
    let mode = argv
        .get(1)
        .filter(|mode| !mode.is_empty())
        .map(String::as_str)
        .unwrap_or("all");
    if matches!(mode, "inspect" | "verify" | "ppl" | "niah") {
        eprintln!("halogen: halogen-flash-server {IMAGE_VERSION}, mode {mode}");
    } else {
        println!("halogen: halogen-flash-server {IMAGE_VERSION}, mode {mode}");
    }
    let _ = std::io::stdout().flush();

    match mode {
        "all" => start_all(&eng, &checkpoint),
        "engine" => start_engine(&eng, &checkpoint),
        "api" => start_api(&eng, &checkpoint),
        "bench" | "sweep" => start_bench_sweep(mode, &eng, &checkpoint, rest(argv)),
        "convert" => convert(argv),
        "inspect" | "verify" => inspect_verify(mode, argv, &checkpoint),
        "ppl" | "niah" => ppl_niah(mode, argv, &checkpoint),
        _ => {
            if mode == "npu" {
                eprintln!("{NPU_RETIRED}");
            }
            eprintln!("{GENERAL_USAGE}");
            2
        }
    }
}

// ---------------------------------------------------------------------------
// Layer 1 — the makeWrapper env contract
// ---------------------------------------------------------------------------

/// Export every wrapper default with `${VAR:-default}` semantics: the
/// default applies when the variable is unset **or empty**; any non-empty
/// value wins verbatim (spec: "`${VAR:-default}` semantics (locked)").
fn apply_wrapper_env() {
    for (key, default) in WRAPPER_DEFAULTS {
        set_default(key, default);
    }
    // The pinned version is a hard assignment, not `${VAR:-default}`.
    env::set_var("HALOGEN_IMAGE_VERSION", IMAGE_VERSION);
}

/// `export KEY="${KEY:-default}"` — a set-but-empty variable also takes the
/// default.
fn set_default(key: &str, default: &str) {
    if env::var(key).map(|value| value.is_empty()).unwrap_or(true) {
        env::set_var(key, default);
    }
}

/// `oom_adj_raise` (entrypoint lines 8–24), verbatim messages.
fn oom_adj_raise() {
    let want = env::var("HALOGEN_OOM_SCORE_ADJ")
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "500".to_string());
    if want == "0" {
        return;
    }
    let invalid =
        || format!("halogen: HALOGEN_OOM_SCORE_ADJ={want} is not 0..1000; left as inherited");
    let Ok(want_num) = want.parse::<u32>() else {
        eprintln!("{}", invalid());
        return;
    };
    if want_num > 1000 {
        eprintln!("{}", invalid());
        return;
    }
    let proc = env::var("_hg_proc")
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "/proc".to_string());
    let path = format!("{proc}/{}/oom_score_adj", process::id());
    let Ok(raw) = fs::read_to_string(&path) else {
        return; // `cat ... || return 0`
    };
    let cur = raw.trim();
    if cur.is_empty() || !cur.chars().all(|c| c.is_ascii_digit() || c == '-') {
        return;
    }
    let Ok(cur_num) = cur.parse::<i32>() else {
        return;
    };
    if cur_num >= want_num as i32 {
        return;
    }
    if fs::write(&path, format!("{want}\n")).is_ok() {
        eprintln!("halogen: oom_score_adj {cur} -> {want}: if the host runs out of memory, the kernel ends this server before the desktop session or the container runtime (HALOGEN_OOM_SCORE_ADJ=0 leaves it)");
    } else {
        eprintln!("halogen: NOTE could not raise oom_score_adj from {cur} (read-only /proc?); a host out of memory may end other processes first");
    }
}

// ---------------------------------------------------------------------------
// Checkpoint resolution and engine settings (entrypoint lines 26–80)
// ---------------------------------------------------------------------------

/// `default_checkpoint /models` (lines 32–47): v2 unless it is absent and
/// the legacy w4b file exists.
fn default_checkpoint(dir: &str) -> String {
    let v2 = format!("{dir}/qwen38-flash-next-v2.hgn");
    if !is_file(&v2) {
        let w4b = format!("{dir}/qwen38-flash-next-w4b.hgn");
        if is_file(&w4b) {
            return w4b; // the port tracks LEGACY_W4B only where observable
        }
    }
    v2
}

/// `resolve_checkpoint /models` (lines 39–47): honour `HALOGEN_CHECKPOINT`
/// (`:-` semantics) and export the result.
fn resolve_checkpoint(dir: &str) -> String {
    let checkpoint = env::var("HALOGEN_CHECKPOINT")
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| default_checkpoint(dir));
    env::set_var("HALOGEN_CHECKPOINT", &checkpoint);
    checkpoint
}

/// The derived engine settings the dispatch passes to `flash_serve` and
/// `serve_api.py` (entrypoint lines 26–71).
struct Engine {
    port: String,
    api_port: String,
    bind: String,
    slots: String,
    ctx_raw: String,
    pool: String,
    max_tok: String,
    max_tokens_cap: String,
    queue_timeout: String,
}

/// Signal that the YaRN gate aborted the launch with exit 2.
struct Exit;

impl Engine {
    /// Re-derive the settings exactly as the entrypoint does, including the
    /// max-token cap note and the YaRN gate (which exits 2 before the
    /// banner is printed).
    fn derive() -> Result<Engine, Exit> {
        let port = env::var("HALOGEN_PORT").unwrap_or_else(|_| "8730".to_string());
        let api_port = env::var("HALOGEN_API_PORT").unwrap_or_else(|_| "8731".to_string());
        let bind = env::var("HALOGEN_BIND").unwrap_or_else(|_| "127.0.0.1".to_string());
        let slots = env::var("HALOGEN_KV_SLOTS").unwrap_or_else(|_| "4".to_string());
        // Non-numeric HALOGEN_CTX aborts bash via `[ -gt ]`; the port falls
        // back to the default (untested surface, documented).
        let ctx_raw = env::var("HALOGEN_CTX").unwrap_or_else(|_| "262144".to_string());
        let ctx = ctx_raw.parse::<i64>().unwrap_or(262144);

        // KV pool derivation (lines 52–57): ctx*2 clamped to [ctx, 1048576].
        let pool = match env::var("HALOGEN_KV_POOL_POSITIONS")
            .ok()
            .filter(|value| !value.is_empty())
        {
            Some(pool) => pool,
            None => {
                let doubled = ctx.saturating_mul(2);
                let pool = if doubled < ctx {
                    ctx
                } else if doubled > 1_048_576 {
                    1_048_576
                } else {
                    doubled
                };
                pool.to_string()
            }
        };

        // Max-token capping (lines 59–71).
        let max_tok_raw = env::var("HALOGEN_MAX_TOK").unwrap_or_else(|_| "32768".to_string());
        let mut max_tok = max_tok_raw.parse::<i64>().unwrap_or(32768);
        let mut max_tok_out = max_tok_raw.clone();
        if ctx > 262144 {
            if max_tok > 16384 {
                eprintln!("halogen: context {ctx_raw} is past the native 262144: HALOGEN_MAX_TOK {max_tok_raw} is capped at 16384 here (a larger prefill arena leaves a 1M KV cache no room to stay resident).");
                max_tok = 16384;
                max_tok_out = "16384".to_string();
            }
            let prompt_cache = env::var("HALOGEN_PROMPT_CACHE").unwrap_or_else(|_| "2".to_string());
            let cache_inplace =
                env::var("HALOGEN_CACHE_INPLACE").unwrap_or_else(|_| "1".to_string());
            let cache_file = env::var("HALOGEN_CACHE_FILE").unwrap_or_default();
            if prompt_cache != "0" && cache_inplace == "0" && cache_file.is_empty() {
                env::set_var("HALOGEN_CACHE_FILE", "/var/tmp/halogen-cache.snapshot");
                eprintln!("halogen: context {ctx_raw} is past the native 262144 with HALOGEN_CACHE_INPLACE=0: the prompt cache snapshot goes to HALOGEN_CACHE_FILE=/var/tmp/halogen-cache.snapshot (up to 26.6 GiB at 1M; mount fast storage there, or set the path).");
            }
        }
        if max_tok > ctx {
            max_tok_out = ctx.to_string();
        }

        // YaRN gate (lines 73–80).
        let rope_yarn = env::var("HALOGEN_ROPE_YARN").unwrap_or_default();
        if ctx > 262144 && rope_yarn.is_empty() {
            eprintln!("halogen: HALOGEN_CTX={ctx_raw} is past the native 262144. Contexts up to 1048576 need HALOGEN_ROPE_YARN=<factor> (4 for 1M, 2 for 524288), the model's documented static YaRN, which changes its numerics at every position. Set it deliberately, or lower HALOGEN_CTX.");
            return Err(Exit);
        }
        if !rope_yarn.is_empty() && ctx <= 262144 {
            eprintln!("halogen: WARNING: HALOGEN_ROPE_YARN={rope_yarn} with HALOGEN_CTX={ctx_raw} at or under the native 262144. Static YaRN rescales every position; the model card advises it only when the context needs it.");
        }

        Ok(Engine {
            port,
            api_port,
            bind,
            slots,
            ctx_raw,
            pool,
            max_tok: max_tok_out,
            max_tokens_cap: env::var("HALOGEN_MAX_TOKENS_CAP")
                .unwrap_or_else(|_| "65536".to_string()),
            queue_timeout: env::var("HALOGEN_QUEUE_TIMEOUT").unwrap_or_else(|_| "3600".to_string()),
        })
    }

    /// The exact `flash_serve` flag set the entrypoint passes in every
    /// launching mode (spec: "Env-var → binary mapping").
    fn serve_args(&self, bind: &str, checkpoint: &str) -> Vec<String> {
        vec![
            "--ck".to_string(),
            checkpoint.to_string(),
            "--port".to_string(),
            self.port.clone(),
            "--bind".to_string(),
            bind.to_string(),
            "--slots".to_string(),
            self.slots.clone(),
            "--ctx".to_string(),
            self.ctx_raw.clone(),
            "--max-tok".to_string(),
            self.max_tok.clone(),
            "--kv-pool".to_string(),
            self.pool.clone(),
        ]
    }

    /// The `serve_api.py` flag set (entrypoint lines 1073–1079, minus the
    /// mode-dependent `--host`/`--engine` differences).
    fn api_args(&self, tokenizer: &str, host: &str, engine: &str) -> Vec<String> {
        vec![
            "--tokenizer".to_string(),
            tokenizer.to_string(),
            "--engine".to_string(),
            engine.to_string(),
            "--host".to_string(),
            host.to_string(),
            "--port".to_string(),
            self.api_port.clone(),
            "--max-tokens-cap".to_string(),
            self.max_tokens_cap.clone(),
            "--context".to_string(),
            self.ctx_raw.clone(),
            "--queue-timeout".to_string(),
            self.queue_timeout.clone(),
        ]
    }
}

// ---------------------------------------------------------------------------
// Sibling binary resolution (see module docs)
// ---------------------------------------------------------------------------

/// Resolve a workspace sibling binary (`flash_serve`, `halogen-tools`,
/// `halogen-npu`, `halogen-healthcheck`) by the documented order:
/// `COLD_CATHODE_<NAME>` → the reference `_hg_*` override → a sibling of the
/// running executable → `PATH`.
fn resolve_sibling(name: &str, hg_var: Option<&str>) -> Option<PathBuf> {
    let override_var = format!("COLD_CATHODE_{}", name.to_uppercase().replace('-', "_"));
    if let Some(path) = env::var(&override_var).ok().filter(|p| !p.is_empty()) {
        return Some(PathBuf::from(path));
    }
    if let Some(var) = hg_var {
        if let Some(path) = env::var(var).ok().filter(|p| !p.is_empty()) {
            return Some(PathBuf::from(path));
        }
    }
    if let Ok(exe) = env::current_exe() {
        let sibling = exe.parent()?.join(name);
        if sibling.is_file() {
            return Some(sibling);
        }
    }
    env::var("PATH")
        .ok()?
        .split(':')
        .map(|dir| Path::new(dir).join(name))
        .find(|candidate| candidate.is_file())
}

/// Resolve the Python front-end interpreter (`_hg_python`, then the
/// reference store path, then `PATH`).
fn resolve_python() -> Option<PathBuf> {
    if let Some(path) = env::var("_hg_python").ok().filter(|p| !p.is_empty()) {
        return Some(PathBuf::from(path));
    }
    let reference = PathBuf::from(REF_PYTHON);
    if reference.is_file() {
        return Some(reference);
    }
    env::var("PATH")
        .ok()?
        .split(':')
        .map(|dir| Path::new(dir).join("python3"))
        .find(|candidate| candidate.is_file())
}

/// Resolve a Python front-end tool script: the cold-cathode package's own
/// `share/halogen/tools/<name>` when present (the spec's rewrite table maps
/// `/halogen/tools/*` to `$out/share/halogen/tools/*`), else the reference
/// package's copy.
fn share_file(name: &str) -> PathBuf {
    if let Ok(exe) = env::current_exe() {
        if let Some(root) = exe.parent().and_then(|bin| bin.parent()) {
            let candidate = root.join("share/halogen/tools").join(name);
            if candidate.is_file() {
                return candidate;
            }
        }
    }
    Path::new(REF_ROOT).join("share/halogen/tools").join(name)
}

// ---------------------------------------------------------------------------
// Process helpers (emulated exec, file probes)
// ---------------------------------------------------------------------------

/// Spawn `program` with `args`, removing `strip` from the child environment
/// (the entrypoint's `env "${ENGINE_ENV_U[@]}"` pattern), and return its
/// exit code. Emulates `exec` via wait + exit (module docs).
fn exec_program(program: &Path, args: &[&str], strip: &[&str]) -> i32 {
    let mut command = Command::new(program);
    command.args(args);
    for var in strip {
        command.env_remove(var);
    }
    match command.status() {
        Ok(status) => exit_code_of(status),
        // bash: a program that cannot be found exits 127, one that exists
        // but is not executable 126 (matches the healthcheck port).
        Err(err) if err.kind() == io::ErrorKind::NotFound => 127,
        Err(_) => 126,
    }
}

/// Spawn a background child with the same environment-stripping semantics.
fn spawn(program: &Path, args: &[&str], strip: &[&str]) -> Option<Child> {
    let mut command = Command::new(program);
    command.args(args);
    for var in strip {
        command.env_remove(var);
    }
    command.spawn().ok()
}

/// Spawn a background child with stdout and stderr redirected to `log`.
fn spawn_logged(program: &Path, args: &[&str], strip: &[&str], log: &str) -> Option<Child> {
    let Ok(file) = fs::File::create(log) else {
        return None;
    };
    let mut command = Command::new(program);
    command.args(args);
    for var in strip {
        command.env_remove(var);
    }
    let stderr = file.try_clone().ok()?;
    command
        .stdout(Stdio::from(file))
        .stderr(Stdio::from(stderr));
    command.spawn().ok()
}

fn exit_code_of(status: process::ExitStatus) -> i32 {
    #[cfg(unix)]
    let signal = status.signal().map(|sig| 128 + sig);
    #[cfg(not(unix))]
    let signal: Option<i32> = None;
    status.code().or(signal).unwrap_or(1)
}

fn is_file(path: &str) -> bool {
    Path::new(path).is_file()
}

/// bash `[ -r f ] && [ -w f ]` for a file — probed by opening read-write.
fn path_writable(path: &Path) -> bool {
    fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .is_ok()
}

/// bash `dirname`: strip the last component; a bare name maps to `.`.
fn parent_dir(path: &str) -> String {
    match Path::new(path).parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.display().to_string(),
        Some(_) | None => ".".to_string(),
    }
}

/// bash `[ -d dir ] && [ -w dir ]` — std has no `access(2)`, so probe by
/// creating (and removing) a marker file.
fn dir_writable(dir: &str) -> bool {
    if !Path::new(dir).is_dir() {
        return false;
    }
    let marker = Path::new(dir).join(".hgntest");
    let existed = marker.exists();
    if fs::File::create(&marker).is_err() {
        return false;
    }
    if !existed {
        let _ = fs::remove_file(&marker);
    }
    true
}

/// bash `head -c 4 "$file"` — the first four bytes, if readable.
fn head4(path: &str) -> Option<[u8; 4]> {
    use std::io::Read;
    let mut file = fs::File::open(path).ok()?;
    let mut buf = [0u8; 4];
    let read = file.read(&mut buf).ok()?;
    if read == 4 {
        Some(buf)
    } else {
        None
    }
}

/// `du -h "$file" | cut -f1` — the 1 KiB-block human size (used by
/// `need_head`'s success line). Approximation (review L6): derived from the
/// file's apparent size with `{:.1}` formatting, which can differ from GNU
/// du's block-based rounding by 0.1 at some sizes; no live matrix case
/// exercises the feed line — re-derive against GNU du if one is added.
fn du_h(path: &str) -> String {
    let bytes = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let kib = (bytes.div_ceil(1024)).max(1);
    let (value, unit) = if kib < 1024 {
        (kib as f64, "K")
    } else if kib < 1024 * 1024 {
        (kib as f64 / 1024.0, "M")
    } else {
        (kib as f64 / (1024.0 * 1024.0), "G")
    };
    format!("{value:.1}{unit}")
}

// ---------------------------------------------------------------------------
// Preflight helpers (structural ports — see module docs)
// ---------------------------------------------------------------------------

/// `host_preflight` (lines 261–289): the `/dev/kfd` + `/dev/dri` +
/// `/sys` kfd topology checks. Returns false when the host cannot run the
/// server (the caller prints the wrapper line and exits 1).
fn host_preflight() -> bool {
    let sys = env::var("_hg_sys")
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "/sys".to_string());
    let dev = env::var("_hg_dev")
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "/dev".to_string());
    let mut bad = false;
    let kfd = Path::new(&dev).join("kfd");
    if !kfd.exists() {
        if Path::new(&dev).join("dxg").exists() {
            eprintln!("halogen: this looks like WSL2: /dev/dxg is here and /dev/kfd is not. WSL2 is not a supported host for this server; run it on Linux on the machine itself.");
        } else {
            eprintln!(
                "halogen: there is no /dev/kfd in this container, so the GPU was not passed in."
            );
            eprintln!("  Add  --device /dev/kfd --device /dev/dri  and, with podman, --group-add keep-groups (with docker: --group-add video --group-add render).");
        }
        bad = true;
    } else if !path_writable(&kfd) {
        // `[ -r ] && [ -w ]` — the warning goes to stdout (line 272).
        println!("halogen: WARNING /dev/kfd is here but this user may not be able to open it. If the engine cannot find the GPU, check the group mapping: --group-add keep-groups with podman, --group-add video --group-add render with docker.");
    }
    if !bad {
        let has_render = fs::read_dir(Path::new(&dev).join("dri"))
            .map(|entries| {
                entries
                    .flatten()
                    .any(|entry| entry.file_name().to_string_lossy().starts_with("renderD"))
            })
            .unwrap_or(false);
        if !has_render {
            eprintln!("halogen: WARNING there is no render node under /dev/dri in this container; pass --device /dev/dri as well as /dev/kfd.");
        }
    }
    let mut gpus = String::new();
    let mut found = false;
    let nodes = Path::new(&sys).join("class/kfd/kfd/topology/nodes");
    if let Ok(entries) = fs::read_dir(&nodes) {
        for entry in entries.flatten() {
            let properties = entry.path().join("properties");
            if !properties.is_file() {
                continue;
            }
            // awk '$1 == "gfx_target_version" {print $2; exit}'
            let version = fs::read_to_string(&properties)
                .unwrap_or_default()
                .lines()
                .find(|line| line.split_whitespace().next() == Some("gfx_target_version"))
                .and_then(|line| line.split_whitespace().nth(1))
                .unwrap_or("")
                .to_string();
            if version.is_empty() || version == "0" {
                continue;
            }
            gpus.push(' ');
            gpus.push_str(&version);
            if version == "110501" {
                found = true;
            }
        }
    }
    if !gpus.is_empty() && !found {
        eprintln!("halogen: the GPU here is not a Strix Halo (gfx1151): the kernel reports gfx_target_version{gpus}, and this server is built for gfx1151 (110501) only.");
        bad = true;
    }
    !bad
}

/// `need_ckpt` (lines 564–581): host preflight, tuning-plan copy and the
/// checkpoint existence check with the exact message; the download and the
/// GGUF/sidecar/vision/KV checks are deferred (module docs).
fn need_ckpt(checkpoint: &str) -> Result<(), i32> {
    if !host_preflight() {
        eprintln!("halogen: this container cannot run the server as started; nothing was downloaded or loaded.");
        return Err(1);
    }
    tuning_plan_copy();
    if !is_file(checkpoint) {
        eprintln!("halogen: no checkpoint at {checkpoint}");
        eprintln!("  mount it:  -v /path/to/models:/models:ro");
        eprintln!("  or point:  -e HALOGEN_CHECKPOINT=/models/<file>.hgn (or any shard of a GGUF)");
        return Err(1);
    }
    Ok(())
}

/// `need_tokenizer` (lines 733–747), verbatim messages.
fn need_tokenizer(checkpoint: &str) -> Result<(), i32> {
    let mut tokenizer = env::var("HALOGEN_TOKENIZER").unwrap_or_else(|_| "/tokenizer".to_string());
    if !is_file(&format!("{tokenizer}/tokenizer.json")) && tokenizer == "/tokenizer" {
        let beside = format!("{}/tokenizer", parent_dir(checkpoint));
        if is_file(&format!("{beside}/tokenizer.json")) {
            println!("halogen: no /tokenizer mount, using {beside} from the models volume");
            tokenizer = beside;
            env::set_var("HALOGEN_TOKENIZER", &tokenizer);
        }
    }
    if !is_file(&format!("{tokenizer}/tokenizer.json")) {
        eprintln!("halogen: no tokenizer.json in {tokenizer}");
        eprintln!("  mount the weights directory at /models (it contains tokenizer/),");
        eprintln!("  or point HALOGEN_TOKENIZER at a FLAT tokenizer dir");
        eprintln!("  (cp -L out of an HF snapshot; symlinks dangle in a container)");
        return Err(1);
    }
    Ok(())
}

/// `check_defaults` (lines 1061–1067): run `serve_api.py --check-defaults`.
fn check_defaults(eng: &Engine) -> Result<(), i32> {
    const INVALID_DEFAULTS: &str =
        "halogen: a HALOGEN_* request default is invalid (see above); not starting";
    let tokenizer = env::var("HALOGEN_TOKENIZER").unwrap_or_else(|_| "/tokenizer".to_string());
    let Some(python) = resolve_python() else {
        eprintln!("{INVALID_DEFAULTS}");
        return Err(1);
    };
    let args = [
        share_file("serve_api.py").display().to_string(),
        "--tokenizer".to_string(),
        tokenizer,
        "--max-tokens-cap".to_string(),
        eng.max_tokens_cap.clone(),
        "--check-defaults".to_string(),
    ];
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    match exec_program(&python, &args, &[]) {
        0 => Ok(()),
        _ => {
            eprintln!("{INVALID_DEFAULTS}");
            Err(1)
        }
    }
}

/// `need_head` (lines 588–613): resolve and validate the MTP draft head.
fn need_head(dir: &str) -> Result<String, i32> {
    let head = env::var("HALOGEN_MTP_HEAD")
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| format!("{dir}/qwen38-flash-next-mtp.hgn"));
    if !is_file(&head) {
        let download = env::var("HALOGEN_DOWNLOAD").ok().filter(|v| !v.is_empty());
        if let Some(download) = download.filter(|_| dir_writable(dir)) {
            let name = head.rsplit('/').next().unwrap_or(&head).to_string();
            if let Some(hf) = env::var("_hg_hf").ok().filter(|p| !p.is_empty()) {
                println!("halogen: fetching the draft head {name} from {download} (1.4 GiB)");
                exec_program(
                    Path::new(&hf),
                    &["download", &download, &name, "--local-dir", dir],
                    &[],
                );
            }
        }
        if !is_file(&head) {
            eprintln!("halogen: no draft head at {head}");
            eprintln!("  A GGUF trunk needs the engine's MTP head file, qwen38-flash-next-mtp.hgn (1.4 GiB),");
            eprintln!("  from the weights repo. Put it beside the GGUF, point HALOGEN_MTP_HEAD at it, or start");
            eprintln!("  once with HALOGEN_DOWNLOAD set and the models volume mounted read-write.");
            return Err(1);
        }
    }
    if head4(&head) != Some(*b"HGN1") {
        eprintln!("halogen: HALOGEN_MTP_HEAD={head} is not the engine's draft head file (its first bytes are not HGN1).");
        eprintln!("  A GGUF draft head (…-MTP-draft.gguf, llama.cpp's) is not read by this engine. The head it runs is");
        eprintln!("  its own qwen38-flash-next-mtp.hgn (1.4 GiB) from the weights repo: put it beside the GGUF, point");
        eprintln!("  HALOGEN_MTP_HEAD at it, or start once with HALOGEN_DOWNLOAD set and the models volume mounted read-write.");
        return Err(1);
    }
    env::set_var("HALOGEN_MTP_HEAD", &head);
    println!("halogen: draft head {head} ({})", du_h(&head));
    Ok(head)
}

/// `tuning_plan_copy` (lines 481–490): copy the baked tuning plan to /tmp
/// when the environment still points at it.
fn tuning_plan_copy() {
    let baked = WRAPPER_DEFAULTS
        .iter()
        .find(|(key, _)| *key == "HALOGEN_MATMUL_TUNING_FILE")
        .map(|(_, value)| value.to_string())
        .unwrap_or_default();
    let current = env::var("HALOGEN_MATMUL_TUNING_FILE").unwrap_or_default();
    if current != baked || !is_file(&baked) {
        return;
    }
    if fs::copy(&baked, "/tmp/halogen-tune.plan").is_ok() {
        env::set_var("HALOGEN_MATMUL_TUNING_FILE", "/tmp/halogen-tune.plan");
    } else {
        eprintln!("halogen: could not copy the tuning plan to /tmp; the engine reads the baked file and may rewrite it at exit");
    }
}

/// `text_only_mode` (lines 636–641), verbatim message.
fn text_only_mode(mode: &str) {
    let tower = env::var("HALOGEN_VISION_TOWER").unwrap_or_default();
    if !tower.is_empty() {
        eprintln!("halogen {mode}: HALOGEN_VISION_TOWER is ignored in this mode (text only)");
        env::remove_var("HALOGEN_VISION_TOWER");
    }
}

// ---------------------------------------------------------------------------
// Engine wait (entrypoint `wait_for_engine`, lines 749–776)
// ---------------------------------------------------------------------------

/// The entrypoint's connect probe (line 754,
/// `(exec 3<>"/dev/tcp/127.0.0.1/$port")`): a TCP connection to `addr`
/// succeeds exactly when the engine socket accepts.
fn socket_accepts(addr: &str) -> bool {
    TcpStream::connect(addr).is_ok()
}

/// `${HALOGEN_ENGINE_WAIT_S:-0}` (line 751): unset or set-but-empty takes
/// the default 0 (wait forever); a non-numeric value makes bash's
/// `[ -gt 0 ]` fail, so the `if` branch is not taken and bash also waits
/// forever — the port maps any unparseable value to 0 likewise.
fn parse_wait_limit(raw: Option<String>) -> u64 {
    raw.and_then(|value| value.parse::<u64>().ok()).unwrap_or(0)
}

/// `wait_for_engine` (entrypoint lines 749–776; the spec's dispatch table
/// `all` row, lines 1674–1720, documents the exit code "1 if the engine
/// never comes up"): poll `127.0.0.1:port` every 2s (the entrypoint's
/// `sleep 2`, line 770), honouring `HALOGEN_ENGINE_WAIT_S` (0/unset waits
/// forever), and print the exact success (line 755), engine-exit (line 759)
/// and timeout (lines 764–766) wordings plus the 60s heartbeat (lines
/// 771–773). The caller exits 1 when this returns false (lines 1688–1693).
fn wait_for_engine(port: &str, engine: &mut Child) -> bool {
    let limit = parse_wait_limit(env::var("HALOGEN_ENGINE_WAIT_S").ok());
    let mut elapsed: u64 = 0;
    let mut beat: u64 = 0;
    loop {
        if socket_accepts(&format!("127.0.0.1:{port}")) {
            println!("halogen: engine listening after {elapsed}s");
            return true;
        }
        // bash `kill -0 "$pid"` (line 758) succeeds while the child is
        // alive; `try_wait` reaps and reports the same fact.
        if engine.try_wait().ok().flatten().is_some() {
            eprintln!("halogen: the engine exited after {elapsed}s without listening.");
            return false;
        }
        if limit > 0 && elapsed >= limit {
            eprintln!(
                "halogen: HALOGEN_ENGINE_WAIT_S={limit} expired after {elapsed}s and the engine is STILL LOADING (pid {} is alive).",
                engine.id()
            );
            eprintln!("  Nothing is wrong with it yet. Raise or unset HALOGEN_ENGINE_WAIT_S to keep waiting;");
            eprintln!("  0 or unset means wait as long as the load takes.");
            return false;
        }
        sleep(Duration::from_secs(2));
        elapsed += 2;
        beat += 2;
        if beat >= 60 {
            beat = 0;
            println!(
                "halogen: still loading, {elapsed}s elapsed (the engine is alive, state {}; GTT in use {} GiB; the startup lines above name the step)",
                wd_state(engine.id()),
                wd_gtt_gib()
            );
        }
    }
}

/// `wd_stat_state` (entrypoint lines 778–783): the state character of a
/// `/proc/<pid>/task/<tid>/stat` content — bash reads the first line,
/// strips the longest prefix up to the last `") "` (the comm field may
/// contain spaces and parentheses) and prints the next field.
fn wd_stat_state(content: &str) -> String {
    let line = content.lines().next().unwrap_or_default();
    let rest = match line.rfind(") ") {
        Some(index) => &line[index + 2..],
        None => line,
    };
    rest.split(' ').next().unwrap_or(rest).to_string()
}

/// `wd_state` (entrypoint lines 791–800): `D` if any thread of `pid` is
/// uninterruptible, else the main thread's state; `?` when nothing is
/// readable (bash's default when the glob matches no readable file).
fn wd_state(pid: u32) -> String {
    let proc = env::var("_hg_proc")
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "/proc".to_string());
    let tasks = Path::new(&proc).join(pid.to_string()).join("task");
    let mut main = "?".to_string();
    if let Ok(entries) = fs::read_dir(&tasks) {
        for entry in entries.flatten() {
            // bash `[ -r "$f" ] || continue` — probed by reading.
            let Ok(content) = fs::read_to_string(entry.path().join("stat")) else {
                continue;
            };
            let state = wd_stat_state(&content);
            if state == "D" {
                return "D".to_string();
            }
            if entry.file_name().to_string_lossy() == pid.to_string() {
                main = state;
            }
        }
    }
    main
}

/// `wd_gtt_tenths` (entrypoint lines 809–823) as a pure function of the
/// bytes a readable `mem_info_gtt_used` holds: GiB×10 with bash's
/// round-half-to-even (`r` over half, or exactly half with `q` odd, bumps
/// `q`).
fn gtt_tenths_from_bytes(bytes: u64) -> u64 {
    let scaled = bytes.saturating_mul(10);
    let mut q = scaled / 1_073_741_824;
    let r = scaled % 1_073_741_824;
    if r > 536_870_912 || (r == 536_870_912 && q % 2 == 1) {
        q += 1;
    }
    q
}

/// `wd_gtt_gib` (entrypoint lines 824–827): the first readable
/// `$sys/class/drm/card*/device/mem_info_gtt_used` (bash's sorted glob)
/// formatted as `X.Y` GiB; `?` when the sysfs has none (the reference's own
/// fallback, also covered by the `_hg_sys` override).
fn wd_gtt_gib() -> String {
    let sys = env::var("_hg_sys")
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "/sys".to_string());
    gtt_gib_under(Path::new(&sys))
}

/// The `wd_gtt_gib` body rooted at `sys` (split out for unit testing).
fn gtt_gib_under(sys: &Path) -> String {
    let drm = sys.join("class/drm");
    let mut cards: Vec<_> = fs::read_dir(&drm)
        .map(|entries| {
            entries
                .flatten()
                .map(|entry| entry.file_name())
                .filter(|name| name.to_string_lossy().starts_with("card"))
                .collect()
        })
        .unwrap_or_default();
    cards.sort(); // bash expands the glob in sorted order
    for card in cards {
        let Ok(raw) = fs::read_to_string(drm.join(card).join("device/mem_info_gtt_used")) else {
            continue; // bash `[ -r "$f" ] || continue`
        };
        // bash `read -r u < "$f"` then `case "$u" in ''|*[!0-9]*) u=0`:
        // any non-numeric content becomes 0.
        let first_line = raw.lines().next().unwrap_or_default().trim();
        let bytes = first_line.parse::<u64>().unwrap_or(0);
        let tenths = gtt_tenths_from_bytes(bytes);
        return format!("{}.{}", tenths / 10, tenths % 10);
    }
    "?".to_string()
}

// ---------------------------------------------------------------------------
// v0.20.5 additions (objectives O1, O2, O5–O8)
// ---------------------------------------------------------------------------

/// `HALOGEN_AGENT_GUARD` mode parsing (v0.20.5 entrypoint lines 8–27, right
/// after the coredump export). `off`/`0`/`false`/`no`/`none` (any case) are
/// a no-op; `watch`/`hold` inject the guard models into
/// `HALOGEN_NPU_MODELS`; anything else exits 1 with the exact notice.
fn agent_guard() -> Result<(), i32> {
    let raw = env::var("HALOGEN_AGENT_GUARD").unwrap_or_default();
    // bash `tr 'A-Z' 'a-z'`: ASCII-only lowercasing.
    let mode = raw.to_ascii_lowercase();
    match mode.as_str() {
        "" | "0" | "off" | "false" | "no" | "none" => return Ok(()),
        "watch" | "hold" => {}
        _ => {
            eprintln!("halogen: HALOGEN_AGENT_GUARD={raw} is not a mode: use off, watch or hold");
            return Err(1);
        }
    }
    let existing = env::var("HALOGEN_NPU_MODELS").unwrap_or_default();
    env::set_var("HALOGEN_NPU_MODELS", agent_guard_inject(&mode, &existing));
    if mode == "hold" {
        eprintln!("halogen: HALOGEN_AGENT_GUARD={raw}: halogen-spotter-0.6b (reads tool results) and decider-4b (whether a tool acts outside the conversation) are served on the NPU");
    } else {
        eprintln!("halogen: HALOGEN_AGENT_GUARD={raw}: halogen-spotter-0.6b (reads tool results) is served on the NPU");
    }
    Ok(())
}

/// The injection loop (lines 13–20): each guard model is appended to the
/// comma list only when `,models,` does not already contain `,model,`; an
/// empty list takes the model without a leading comma. Existing entries are
/// extended, never duplicated.
fn agent_guard_inject(mode: &str, existing: &str) -> String {
    let set = if mode == "hold" {
        "halogen-spotter-0.6b decider-4b"
    } else {
        "halogen-spotter-0.6b"
    };
    let mut models = existing.to_string();
    for ag in set.split_whitespace() {
        let wrapped = format!(",{models},");
        if wrapped.contains(&format!(",{ag},")) {
            continue;
        }
        if models.is_empty() {
            models = ag.to_string();
        } else {
            models = format!("{models},{ag}");
        }
    }
    models
}

/// `cache_gib` from `kv_budget_note` (v0.20.5 entrypoint line 111) as a
/// function of the derived engine's `ENG_SLOTS`/`ENG_CTX` plus the
/// environment: the awk's `${VAR:-default}` inputs are resolved, then its
/// expression applied. The reply-end term (`rp`, HALOGEN_CACHE_REPLY,
/// suppressed when HALOGEN_PROMPT_CACHE=1) is the 0.20.5 addition. The
/// string comparisons (`ip`, `s3`, `fu`, `rp`, `pm`) follow awk's string
/// semantics; `s`, `br`, `rr`, `n` take awk's numeric semantics (a
/// non-numeric value counts as 0).
pub fn cache_gib(slots: &str, ctx: &str) -> String {
    let get = |key: &str, default: &str| {
        env::var(key)
            .ok()
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| default.to_string())
    };
    let prompt_cache = get("HALOGEN_PROMPT_CACHE", "2");
    let cache_inplace = get("HALOGEN_CACHE_INPLACE", "1");
    let cache_file = env::var("HALOGEN_CACHE_FILE").unwrap_or_default();
    let entries = env::var("HALOGEN_CACHE_ENTRIES").unwrap_or_default();
    let branches = get("HALOGEN_CACHE_BRANCHES", "2");
    let snap3 = get("HALOGEN_CACHE_SNAP3", "1");
    let full = get("HALOGEN_CACHE_FULL", "1");
    let reply = get("HALOGEN_CACHE_REPLY", "1");
    let kv_rows = if env::var("HALOGEN_KV8").unwrap_or_default() == "1" {
        "13832"
    } else {
        "26624"
    };
    let n = if entries.is_empty() {
        let per_slot = 1.0
            + awk_num(&branches) * (if snap3 == "0" { 1.0 } else { 2.0 })
            + (if full == "0" { 0.0 } else { 1.0 })
            + (if reply == "0" || prompt_cache == "1" {
                0.0
            } else {
                1.0
            });
        awk_num(slots) * per_slot
    } else {
        awk_num(&entries)
    };
    let gib = if prompt_cache == "0" || !cache_file.is_empty() {
        0.0
    } else if cache_inplace != "0" {
        n * 115.0 * 1048576.0 / 1073741824.0
    } else {
        awk_num(ctx) * awk_num(kv_rows) / 1073741824.0
    };
    format!("{gib:.1}")
}

/// awk's numeric coercion of a value used arithmetically: a non-numeric
/// string counts as 0 (the port does not emulate awk's leading-number
/// prefix, e.g. `12abc`; no live case presents one).
fn awk_num(value: &str) -> f64 {
    value.parse::<f64>().unwrap_or(0.0)
}

/// One `file` line of the image's NPU record (`npu_pin_files`, entrypoint
/// line 1116): the path under `/models/npu/<id>/`, the size in bytes and
/// the sha256.
pub struct NpuPin {
    pub path: String,
    pub size: u64,
    pub sha: String,
}

/// `npu_pin_files` (entrypoint line 1116): the file records of one model id
/// parsed out of the models.txt text.
pub fn npu_pin_files(record: &str, id: &str) -> Vec<NpuPin> {
    let mut pins = Vec::new();
    for line in record.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() >= 5 && fields[0] == "file" && fields[1] == id {
            if let Ok(size) = fields[3].parse() {
                pins.push(NpuPin {
                    path: fields[2].to_string(),
                    size,
                    sha: fields[4].to_string(),
                });
            }
        }
    }
    pins
}

/// `npu_model_ready`'s pin filter (v0.20.5 entrypoint lines 1307–1311):
/// `devices` now also takes the vis/ and aud/ pins; `vis` and `aud` are the
/// new part types; any other part keeps every pin.
pub fn npu_part_pins<'a>(part: &str, pins: &'a [NpuPin]) -> Vec<&'a NpuPin> {
    pins.iter()
        .filter(|pin| match part {
            "devices" => {
                pin.path.starts_with("devices/")
                    || pin.path.starts_with("vis/")
                    || pin.path.starts_with("aud/")
            }
            "vis" => pin.path.starts_with("vis/"),
            "aud" => pin.path.starts_with("aud/"),
            _ => true,
        })
        .collect()
}

/// `npu_model_ready`'s path list (v0.20.5 entrypoint lines 1326–1332) for a
/// part: the recorded pins when present (and verifiable or downloadable),
/// else the part's fallback set — `vis/devices.hnpm` and
/// `aud/devices.hnpm` are the 0.20.5 additions.
pub fn npu_part_paths(
    part: &str,
    id: &str,
    pins: &[&NpuPin],
    verify: bool,
    download: bool,
    own: &str,
) -> Vec<String> {
    if !pins.is_empty() && (verify || download) {
        return pins.iter().map(|pin| pin.path.clone()).collect();
    }
    match part {
        "vis" => vec!["vis/devices.hnpm".to_string()],
        "aud" => vec!["aud/devices.hnpm".to_string()],
        "devices" => vec!["devices/devices.hnpm".to_string()],
        _ if own != id => vec![format!("{id}.hnpw"), "tokenizer/tokenizer.json".to_string()],
        _ => vec![
            "devices/devices.hnpm".to_string(),
            format!("{id}.hnpw"),
            "tokenizer/tokenizer.json".to_string(),
        ],
    }
}

/// `npu_model_ready`'s .elf check (v0.20.5 entrypoint lines 1347–1360): the
/// exact error when the part's directory under `d` holds no .elf file;
/// `None` when the check passes or the part needs none.
pub fn npu_elf_error(part: &str, id: &str, d: &str, own_is_id: bool) -> Option<String> {
    let (dir, message) = match part {
        "vis" => (
            format!("{d}/vis"),
            format!("halogen npu: {id}: {d}/vis/ holds no .elf file (the NPU's program for the model's vision tower)"),
        ),
        "aud" => (
            format!("{d}/aud"),
            format!("halogen npu: {id}: {d}/aud/ holds no .elf file (the NPU's program for the model's audio encoder)"),
        ),
        _ if part == "devices" || own_is_id => (
            format!("{d}/devices"),
            format!("halogen npu: {id}: {d}/devices/ holds no .elf file (the NPU's program for the model)"),
        ),
        _ => return None,
    };
    if dir_has_elf(&dir) {
        None
    } else {
        Some(message)
    }
}

/// bash `ls "$dir/"*.elf` succeeding: any entry ending in `.elf`.
fn dir_has_elf(dir: &str) -> bool {
    fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .any(|entry| entry.file_name().to_string_lossy().ends_with(".elf"))
        })
        .unwrap_or(false)
}

/// The vis part's `NPU_NEED_K` accounting (v0.20.5 entrypoint lines
/// 1396–1399): the `du -sk` of the model's vis directory, then the early
/// return.
pub fn npu_vis_need_k(d: &str) -> u64 {
    du_k(&format!("{d}/vis"))
}

/// `du -sk` as a 1 KiB-block count: every file's size rounded up to a KiB,
/// summed recursively (the port's approximation, cf. `du_h`'s note).
fn du_k(dir: &str) -> u64 {
    let mut total = 0;
    let mut stack = vec![PathBuf::from(dir)];
    while let Some(path) = stack.pop() {
        let Ok(entries) = fs::read_dir(&path) else {
            continue;
        };
        for entry in entries.flatten() {
            let child = entry.path();
            if child.is_dir() {
                stack.push(child);
            } else {
                let len = entry.metadata().map(|m| m.len()).unwrap_or(0);
                total += len.div_ceil(1024).max(1);
            }
        }
    }
    total
}

/// `npu_finetune_ready`'s convert flags (v0.20.5 entrypoint line 1463):
/// `--vis`/`--aud` with their directories when the fine-tune's base carries
/// them.
pub fn npu_convert_flags(vis: Option<&str>, aud: Option<&str>) -> Vec<String> {
    let mut args = Vec::new();
    if let Some(tower) = vis {
        args.push("--vis".to_string());
        args.push(tower.to_string());
    }
    if let Some(encoder) = aud {
        args.push("--aud".to_string());
        args.push(encoder.to_string());
    }
    args
}

/// The v0.20.5 image's NPU record directory (the `npu/xrt.sha256` and
/// `npu/xrt.version` files). The vendored package ships no /opt/xilinx/xrt,
/// so the host-mount fallback is the live path and the image-XRT branch
/// stays data-driven: it only fires when these files exist.
const NPU_RECORD_DIR: &str =
    "/nix/store/lyvyw7njybryws0gizlvxpx337p33xnh-halogen-flash-server-0.20.5/opt/halogen/npu";

/// The three XRT libraries `npu_xrt_own` checks (entrypoint line 1472).
const XRT_LIBS: [&str; 3] = [
    "libxrt_coreutil.so.2",
    "libxrt_core.so.2",
    "libxrt_driver_xdna.so.2",
];

/// `npu_xrt_own` (v0.20.5 entrypoint lines 1469–1478): how many of the
/// three XRT libraries under `lib_dir` hash-match the image's record. The
/// record's absence answers 0 (the vendored package ships none).
pub fn npu_xrt_own(record: &str, lib_dir: &str, sha256_prog: &str) -> u32 {
    let Ok(contents) = fs::read_to_string(record) else {
        return 0;
    };
    XRT_LIBS
        .iter()
        .filter(|lib| {
            let want = record_line_hash(&contents, lib);
            let got = sha256_of(sha256_prog, &format!("{lib_dir}/{lib}"));
            !want.is_empty() && Some(want) == got
        })
        .count() as u32
}

/// `npu_xrt_own` at the entrypoint's baked paths (v0.20.5 lines 1470–1471
/// with `_hg_root` empty): the image's record against `/opt/xilinx/xrt/lib`.
pub fn npu_xrt_own_default(sha256_prog: &str) -> u32 {
    npu_xrt_own(
        &format!("{NPU_RECORD_DIR}/xrt.sha256"),
        "/opt/xilinx/xrt/lib",
        sha256_prog,
    )
}

/// The sha256 recorded for a library name (awk `$2 == f {print $1}`).
fn record_line_hash(contents: &str, name: &str) -> String {
    for line in contents.lines() {
        let mut fields = line.split_whitespace();
        let (Some(hash), Some(file)) = (fields.next(), fields.next()) else {
            continue;
        };
        if file == name {
            return hash.to_string();
        }
    }
    String::new()
}

/// `${_hg_sha256:-sha256sum}`: the file's hash via the helper program;
/// `None` when the file or the program is missing.
fn sha256_of(prog: &str, path: &str) -> Option<String> {
    let out = Command::new(prog).arg(path).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let first = String::from_utf8(out.stdout)
        .ok()?
        .split_whitespace()
        .next()?
        .to_string();
    Some(first)
}

/// The mixed-mount refusal (v0.20.5 entrypoint lines 1504–1507): a partial
/// match means a host's libraries were mounted over some of the image's.
pub fn npu_xrt_mixed_error(own: u32, ownv: &str) -> Option<String> {
    (own == 1 || own == 2).then(|| format!("halogen npu: /opt/xilinx/xrt/lib mixes a host's XRT libraries with this image's XRT {ownv}. Mount all three of them, or none: with no mount the image uses its own."))
}

/// The startup naming (v0.20.5 entrypoint lines 1534–1538): "XRT <ver>
/// (this image's)" when all three libraries match the record, else the
/// host's naming derived from the mounted file's name.
pub fn npu_xrt_name(ownv: &str, own: u32, mounted: &str) -> String {
    if own == 3 {
        return format!("XRT {ownv} (this image's)");
    }
    let version = mounted
        .strip_prefix("libxrt_coreutil.so.")
        .unwrap_or(mounted);
    if version.contains('.') {
        format!("XRT {version} (the host's)")
    } else {
        "the host's XRT (its version is not in the mounted file's name)".to_string()
    }
}

/// The fallback hint on an engine that exited before ready (v0.20.5
/// entrypoint lines 1671–1673), printed when it ran on the image's own XRT.
pub fn npu_engine_exit_hint(xrt_v: &str, kernel: &str, drv: &str) -> String {
    format!("  It ran on {xrt_v} against the host's kernel {kernel}, driver amdxdna {drv}. If that driver is what refused it, use the host's XRT: -v /opt/xilinx/xrt:/opt/xilinx/xrt:ro, or for a distribution's packages the mounts in the NPU page's host section.")
}

/// The `HALOGEN_NPU_KILL_S` gate of `start_npu_engine` (v0.20.5 entrypoint
/// lines 1661–1663): `${HALOGEN_NPU_KILL_S:-1800}` substitutes the default
/// for a set-but-empty value too (bash's `''` case is unreachable), so only
/// a non-numeric value yields the exact error (the caller exits 1).
pub fn parse_npu_kill_s(raw: Option<String>) -> Result<u64, String> {
    let value = raw
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "1800".to_string());
    if !value.bytes().all(|b| b.is_ascii_digit()) {
        return Err(format!(
            "halogen npu: HALOGEN_NPU_KILL_S={value} is not a number of seconds (0 = off)"
        ));
    }
    Ok(value.parse().unwrap_or(0))
}

/// `npu_running_ms` (v0.20.5 entrypoint lines 1710–1717) as a pure parse of
/// the INFO reply: the digits after `"running_ms":`, `None` when the field
/// is absent or empty.
pub fn parse_running_ms(line: &str) -> Option<u64> {
    let rest = line.split_once("\"running_ms\":")?.1;
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

/// `npu_running_ms` (v0.20.5 entrypoint lines 1710–1717): the INFO 0 probe
/// of the NPU engine's port (bash's `/dev/tcp` with a 5s read timeout).
pub fn npu_running_ms(port: &str) -> Option<u64> {
    let stream = TcpStream::connect(format!("127.0.0.1:{port}")).ok()?;
    stream.set_read_timeout(Some(Duration::from_secs(5))).ok()?;
    (&stream).write_all(b"INFO 0\n").ok()?;
    let mut reader = BufReader::new(&stream);
    let mut line = String::new();
    reader.read_line(&mut line).ok()?;
    parse_running_ms(&line)
}

/// `npu_stall_watchdog` (v0.20.5 entrypoint lines 1718–1758): the background
/// poller. A job holding the NPU past `limit` seconds gets TERM then KILL
/// (the restart itself lives in the `all` mode's loop); D-state threads
/// defer the kill; `parent` bounds the loop (bash's `$$`).
pub fn npu_stall_watchdog(port: &str, limit: u64, step: u64, pidfile: &str, parent: u32) {
    if limit == 0 {
        println!("halogen npu: stall watchdog OFF (HALOGEN_NPU_KILL_S=0)");
        return;
    }
    println!("halogen npu: stall watchdog on: one job holding the NPU for {limit} s takes the NPU engine down and starts it again (HALOGEN_NPU_KILL_S)");
    let start = Instant::now();
    let mut killed = String::new();
    let mut dlast = Duration::ZERO;
    while wd_alive(parent) {
        sleep(Duration::from_secs(step));
        let pid = fs::read_to_string(pidfile)
            .ok()
            .and_then(|raw| raw.lines().next().map(str::to_string))
            .unwrap_or_default();
        let Ok(pid_num) = pid.trim().parse::<u32>() else {
            continue; // bash `case "$pid" in ''|*[!0-9]*) continue`
        };
        let pid = pid_num.to_string();
        if !wd_alive(pid_num) || pid == killed {
            continue;
        }
        let Some(ms) = npu_running_ms(port) else {
            continue;
        };
        let s = ms / 1000;
        if s < limit {
            continue;
        }
        if wd_state(pid_num) == "D" {
            let now = start.elapsed();
            if dlast.is_zero() || now - dlast >= Duration::from_secs(60) {
                dlast = now;
                eprintln!("halogen npu: one job has held the NPU for {s} s (HALOGEN_NPU_KILL_S={limit}) and a thread of the NPU engine is in uninterruptible sleep (state D, inside the kernel). Not taken down: SIGKILL does not reach it and killing the engine there is what leaves the driver holding its memory (issue #79). Deferred; checked again in {step} s.");
            }
            continue;
        }
        eprintln!("halogen npu: one job has held the NPU for {s} s (HALOGEN_NPU_KILL_S={limit}); the device is not coming back. Stopping the NPU engine; it is started again (HALOGEN_NPU_RESTARTS) and the Flash engine serves on. Requests waiting for the NPU are answered 503.");
        kill_pid(&pid, "TERM");
        let mut t = 0;
        while wd_alive(pid_num) && t < 10 {
            // bash `${_hg_npu_term_s:-10}`
            sleep(Duration::from_secs(1));
            t += 1;
        }
        if wd_alive(pid_num) {
            if wd_state(pid_num) == "D" {
                eprintln!("halogen npu: the NPU engine did not exit on SIGTERM and a thread of it is now in uninterruptible sleep (state D); SIGKILL deferred (issue #79).");
            } else {
                eprintln!("halogen npu: the NPU engine did not exit on SIGTERM within {t} s; sending SIGKILL");
                kill_pid(&pid, "KILL");
            }
        }
        killed = pid;
    }
}

/// bash `wd_alive` (entrypoint line 856): the /proc directory exists and the
/// main thread is not a zombie.
fn wd_alive(pid: u32) -> bool {
    let proc = env::var("_hg_proc")
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "/proc".to_string());
    Path::new(&proc).join(pid.to_string()).is_dir() && wd_state(pid) != "Z"
}

/// bash `kill -SIG "$pid"` for a pid that is not our child: the `kill`
/// binary (std has no kill for foreign pids without libc).
fn kill_pid(pid: &str, sig: &str) {
    let _ = Command::new("kill").arg(format!("-{sig}")).arg(pid).spawn();
}

/// The rc-76 memory-floor exit of `npu_again` (v0.20.5 entrypoint line
/// 1783): a floor exit costs no restart count.
pub fn npu_floor_exit(rc: i32) -> bool {
    rc == 76
}

/// The floor retry backoff (v0.20.5 entrypoint lines 1785–1786): the delay
/// doubles from the 30 s default (or `_hg_npu_room_s`) up to the 300 s cap.
pub fn npu_room_s(current: u64, room_default: u64) -> u64 {
    (if current > 0 {
        current.saturating_mul(2)
    } else {
        room_default
    })
    .min(300)
}

/// The floor notice (v0.20.5 entrypoint line 1787), verbatim.
pub fn npu_floor_notice(room_s: u64) -> String {
    format!("halogen npu: the host is under the NPU engine's memory floor (HALOGEN_NPU_HOST_FLOOR_GIB; its line is above): trying again in {room_s} s, until there is room. Its routes answer 503 until then and the Flash engine serves on")
}

// ---------------------------------------------------------------------------
// Mode implementations (spec: "The dispatch table")
// ---------------------------------------------------------------------------

/// `all` (lines 1674–1720): engine + front end. The NPU-alone branch, the
/// watchdog and the wait -n restart loop are deferred (module docs).
fn start_all(eng: &Engine, checkpoint: &str) -> i32 {
    if let Err(code) = need_ckpt(checkpoint) {
        return code;
    }
    if let Err(code) = need_tokenizer(checkpoint) {
        return code;
    }
    if let Err(code) = check_defaults(eng) {
        return code;
    }
    let Some(serve) = resolve_sibling("flash_serve", Some("_hg_flash_serve")) else {
        return 127;
    };
    // `all` hard-codes the bind address (spec: "all/bench hard-code 127.0.0.1").
    let args = eng.serve_args("127.0.0.1", checkpoint);
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let Some(mut engine) = spawn(&serve, &args, FE_ONLY_FLAGS) else {
        return 127;
    };
    println!(
        "halogen: waiting for engine on {} (cold load can take minutes)",
        eng.port
    );
    let _ = std::io::stdout().flush();
    // `wait_for_engine` (entrypoint lines 1688–1693): spawn the API only
    // once the engine socket accepts; if the engine never comes up, kill
    // and reap it and exit 1 (the dispatch table's "1 if the engine never
    // comes up").
    if !wait_for_engine(&eng.port, &mut engine) {
        let _ = engine.kill();
        let _ = engine.wait();
        return 1;
    }
    let api = resolve_python().map(|python| {
        let tokenizer = env::var("HALOGEN_TOKENIZER").unwrap_or_else(|_| "/tokenizer".to_string());
        // The script path comes first (entrypoint: `python3 "$_hg_serve_api"
        // "${API_ARGS[@]}" &`), matching the `api` and `bench` modes.
        let mut api_args = vec![share_file("serve_api.py").display().to_string()];
        // v0.20.5 (entrypoint line 1862): the `all` mode binds the API to `::`.
        api_args.extend(eng.api_args(&tokenizer, "::", &format!("127.0.0.1:{}", eng.port)));
        let api_args: Vec<&str> = api_args.iter().map(String::as_str).collect();
        spawn(&python, &api_args, &[])
    });
    let rc = engine.wait().map(exit_code_of).unwrap_or(127);
    if let Some(Some(mut api)) = api {
        let _ = api.kill();
        let _ = api.wait();
    }
    eprintln!("halogen: a component exited (rc={rc}); shutting down");
    rc
}

/// `engine` (lines 1017–1059): spawn `flash_serve` with the derived flag
/// set, wait, report, exit with the engine's code (`stop_clean` without a
/// signal trap reduces to the engine's rc; module docs).
fn start_engine(eng: &Engine, checkpoint: &str) -> i32 {
    if let Err(code) = need_ckpt(checkpoint) {
        return code;
    }
    let Some(serve) = resolve_sibling("flash_serve", Some("_hg_flash_serve")) else {
        return 127;
    };
    let args = eng.serve_args(&eng.bind, checkpoint);
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let rc = exec_program(&serve, &args, FE_ONLY_FLAGS);
    eprintln!(
        "halogen: engine exited (rc={rc}, the component that ended this: {rc}); shutting down"
    );
    rc
}

/// `api` (lines 1069–1080): `exec serve_api.py` with the exact flag set.
fn start_api(eng: &Engine, checkpoint: &str) -> i32 {
    if let Err(code) = need_tokenizer(checkpoint) {
        return code;
    }
    if let Err(code) = check_defaults(eng) {
        return code;
    }
    let Some(python) = resolve_python() else {
        return 127;
    };
    let tokenizer = env::var("HALOGEN_TOKENIZER").unwrap_or_else(|_| "/tokenizer".to_string());
    let engine = env::var("HALOGEN_ENGINE")
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| format!("127.0.0.1:{}", eng.port));
    let mut args = vec![share_file("serve_api.py").display().to_string()];
    // v0.20.5 (entrypoint line 1100): the `api` mode binds to `::`.
    args.extend(eng.api_args(&tokenizer, "::", &engine));
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    exec_program(&python, &args, &[])
}

/// `bench`/`sweep` (lines 1721–1775): engine + front end + the bench tool.
/// The `/health` poll and the log `tee` are deferred (module docs).
fn start_bench_sweep(mode: &str, eng: &Engine, checkpoint: &str, rest: &[String]) -> i32 {
    text_only_mode(mode);
    if let Err(code) = need_ckpt(checkpoint) {
        return code;
    }
    if let Err(code) = need_tokenizer(checkpoint) {
        return code;
    }
    if let Err(code) = check_defaults(eng) {
        return code;
    }
    let Some(serve) = resolve_sibling("flash_serve", Some("_hg_flash_serve")) else {
        return 127;
    };
    let args = eng.serve_args("127.0.0.1", checkpoint);
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let Some(mut engine) = spawn_logged(&serve, &args, FE_ONLY_FLAGS, "/tmp/halogen-engine.log")
    else {
        return 127;
    };
    println!("halogen bench: loading model (cold load can take minutes)");
    let _ = std::io::stdout().flush();

    let api = resolve_python().and_then(|python| {
        let tokenizer = env::var("HALOGEN_TOKENIZER").unwrap_or_else(|_| "/tokenizer".to_string());
        let mut args = vec![share_file("serve_api.py").display().to_string()];
        args.extend(eng.api_args(&tokenizer, "127.0.0.1", &format!("127.0.0.1:{}", eng.port)));
        let args: Vec<&str> = args.iter().map(String::as_str).collect();
        spawn_logged(
            &python,
            &args,
            &["HALOGEN_API_KEY", "HALOGEN_API_KEY_FILE"],
            "/tmp/halogen-api.log",
        )
    });

    let rc = run_bench_tool(mode, eng, rest);
    let _ = engine.kill();
    let _ = engine.wait();
    if let Some(mut api) = api {
        let _ = api.kill();
        let _ = api.wait();
    }
    rc
}

/// The tool the bench/sweep modes run after the stack is up
/// (lines 1764–1772).
fn run_bench_tool(mode: &str, eng: &Engine, rest: &[String]) -> i32 {
    let Some(python) = resolve_python() else {
        return 127;
    };
    if mode == "sweep" {
        let mut args = vec![
            share_file("halogen-bench.py").display().to_string(),
            "--api".to_string(),
            format!("http://127.0.0.1:{}", eng.api_port),
        ];
        args.extend(rest.iter().cloned());
        let args: Vec<&str> = args.iter().map(String::as_str).collect();
        exec_program(&python, &args, &[])
    } else {
        let positional = |index: usize, default: &str| {
            rest.get(index)
                .filter(|value| !value.is_empty())
                .cloned()
                .unwrap_or_else(|| default.to_string())
        };
        let mut args = vec![share_file("bench-serving.py").display().to_string()];
        args.push(positional(0, "serial,mtp"));
        args.push(positional(1, "256"));
        args.push(positional(2, "low"));
        args.push(positional(3, "1"));
        let args: Vec<&str> = args.iter().map(String::as_str).collect();
        Command::new(&python)
            .args(&args)
            .env("HALOGEN_API", format!("http://127.0.0.1:{}", eng.api_port))
            .env("HALOGEN_API_LOG", "/tmp/halogen-api.log")
            .status()
            .map(exit_code_of)
            .unwrap_or(127)
    }
}

/// `convert IN.gguf OUT.hgn` (lines 1776–1800), fully ported: usage, input
/// checks, draft head, `flash_serve --repack`, result lines.
fn convert(argv: &[String]) -> i32 {
    let input = argv.get(2).cloned().unwrap_or_default();
    let output = argv.get(3).cloned().unwrap_or_default();
    if input.is_empty() || output.is_empty() {
        eprintln!("{CONVERT_USAGE}");
        return 2;
    }
    if !is_file(&input) {
        eprintln!("halogen convert: {input} is not there (mount the models volume and name a file inside it)");
        return 1;
    }
    if head4(&input) != Some(*b"GGUF") {
        eprintln!("halogen convert: {input} is not a GGUF (its first bytes are not GGUF)");
        return 1;
    }
    let outdir = parent_dir(&output);
    if !dir_writable(&outdir) {
        eprintln!(
            "halogen convert: cannot write into {outdir} (is the volume mounted read-write?)"
        );
        return 1;
    }
    let Ok(head) = need_head(&parent_dir(&input)) else {
        return 1;
    };
    println!("halogen convert: {input} -> {output} (the table and the draft head folded in; a few minutes to ten on an NVMe disk)");
    let Some(serve) = resolve_sibling("flash_serve", Some("_hg_flash_serve")) else {
        return 127;
    };
    let rc = exec_program(
        &serve,
        &[
            "--repack",
            &input,
            "--out",
            &output,
            "--with-table",
            "--head",
            &head,
        ],
        &[],
    );
    if rc == 0 {
        println!("halogen convert: done. Start the image with HALOGEN_CHECKPOINT={output} (no GGUF is needed beside it;");
        println!(
            "  the quality sidecar does not apply to a converted trunk, and none is looked for)."
        );
    } else {
        eprintln!("halogen convert: the repack failed (rc={rc}); {output} is not usable and can be removed");
    }
    rc
}

/// `inspect`/`verify` (lines 1801–1821): resolve FILE (first non-flag
/// argument, else the checkpoint), require it, `exec halogen-tools`.
fn inspect_verify(mode: &str, argv: &[String], checkpoint: &str) -> i32 {
    text_only_mode(mode);
    let rest = rest(argv);
    let (file, extra) = split_file(rest, checkpoint);
    if !is_file(&file) {
        eprintln!("halogen {mode}: {file} is not there (mount the models volume and name a file inside it, or set HALOGEN_CHECKPOINT)");
        return 1;
    }
    let Some(tools) = resolve_sibling("halogen-tools", None) else {
        return 127;
    };
    let mut args = vec![mode.to_string(), file];
    args.extend(extra.iter().cloned());
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    exec_program(&tools, &args, &[])
}

/// `ppl`/`niah` (lines 1810–1819): tuning-plan copy, head/ngram checks, the
/// environment line, tokenizer, then `exec halogen_tools.py`.
fn ppl_niah(mode: &str, argv: &[String], checkpoint: &str) -> i32 {
    text_only_mode(mode);
    let rest = rest(argv);
    let (file, extra) = split_file(rest, checkpoint);
    if !is_file(&file) {
        eprintln!("halogen {mode}: {file} is not there (mount the models volume and name a file inside it, or set HALOGEN_CHECKPOINT)");
        return 1;
    }
    tuning_plan_copy();
    if head4(&file) == Some(*b"GGUF") && need_head(&parent_dir(&file)).is_err() {
        return 1;
    }
    // check_ngram_table (lines 526–562) is skipped for non-GGUF files: it
    // parses the .hgn header and is silent whenever the header is
    // unparseable, which is the only case this host can present.
    let tuning = env::var("HALOGEN_MATMUL_TUNING_FILE").unwrap_or_else(|_| "none".to_string());
    let overlay = env::var("HALOGEN_CK_OVERLAY")
        .unwrap_or_else(|_| "beside the checkpoint, if any".to_string());
    let pin = env::var("HALOGEN_FLASH_PIN_TRUNK").unwrap_or_else(|_| "1".to_string());
    eprintln!("halogen {mode}: {file} under this image's engine environment (tuning plan: {tuning}; quality sidecar: {overlay}; trunk pinned: {pin})");
    if need_tokenizer(checkpoint).is_err() {
        return 1;
    }
    let Some(python) = resolve_python() else {
        return 127;
    };
    let tokenizer = env::var("HALOGEN_TOKENIZER").unwrap_or_else(|_| "/tokenizer".to_string());
    let mut args = vec![
        share_file("halogen_tools.py").display().to_string(),
        mode.to_string(),
        file,
        "--tokenizer".to_string(),
        tokenizer,
    ];
    args.extend(extra.iter().cloned());
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    exec_program(&python, &args, &[])
}

/// The arguments after the mode word.
fn rest(argv: &[String]) -> &[String] {
    if argv.len() > 1 {
        &argv[2..]
    } else {
        &[]
    }
}

/// `FILE=""` / `if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then FILE="$1"; shift; fi`
/// followed by `[ -n "$FILE" ] || FILE="$HALOGEN_CHECKPOINT"` (lines
/// 1806–1808): the first non-flag argument is the file; flags stay in the
/// forwarded tail.
fn split_file<'a>(rest: &'a [String], checkpoint: &str) -> (String, &'a [String]) {
    match rest.first() {
        Some(first) if !first.starts_with('-') => (first.clone(), &rest[1..]),
        _ if !checkpoint.is_empty() => (checkpoint.to_string(), rest),
        _ => (String::new(), rest),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::sync::Mutex;

    /// Serialises the socket tests: `closed_port` frees an ephemeral port
    /// and immediately probes it, so a sibling test binding in that window
    /// can make the probe succeed (the flake). A shared lock keeps every
    /// bind/probe pair atomic across the suite. The poison recovery keeps a
    /// failing test from cascading into the others.
    static SOCKET_TEST_LOCK: Mutex<()> = Mutex::new(());

    /// A port nothing listens on: bind an ephemeral listener and drop it,
    /// so a connect to it is refused (the state `wait_for_engine` polls).
    /// Callers hold `SOCKET_TEST_LOCK` for the bind→drop→probe sequence.
    fn closed_port() -> u16 {
        TcpListener::bind("127.0.0.1:0")
            .expect("ephemeral bind")
            .local_addr()
            .expect("ephemeral address")
            .port()
    }

    #[test]
    fn socket_accepts_true_for_a_listening_port() {
        let _guard = SOCKET_TEST_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener binds");
        let addr = listener.local_addr().expect("listener address");
        assert!(socket_accepts(&addr.to_string()));
    }

    #[test]
    fn socket_accepts_false_for_a_closed_port() {
        let _guard = SOCKET_TEST_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        assert!(!socket_accepts(&format!("127.0.0.1:{}", closed_port())));
    }

    #[test]
    fn wait_for_engine_returns_true_once_the_socket_accepts() {
        // The engine child stays alive and the socket accepts on the first
        // probe, so the helper returns true at t=0 without sleeping.
        let _guard = SOCKET_TEST_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener binds");
        let port = listener.local_addr().expect("listener address").port();
        let mut engine = Command::new("sh")
            .args(["-c", "sleep 5"])
            .spawn()
            .expect("spawn sh");
        assert!(wait_for_engine(&port.to_string(), &mut engine));
        let _ = engine.kill();
        let _ = engine.wait();
    }

    #[test]
    fn wait_for_engine_reports_an_exited_engine_on_a_closed_port() {
        // bash's `kill -0` fails once the engine child is gone (line 758);
        // `try_wait` reports the same fact and the helper returns false —
        // the dispatch table's "1 if the engine never comes up". The
        // pre-wait reaps the child so the test never depends on scheduling.
        let _guard = SOCKET_TEST_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let mut engine = Command::new("sh")
            .args(["-c", "exit 0"])
            .spawn()
            .expect("spawn sh");
        engine.wait().expect("sh exits");
        assert!(!wait_for_engine(&closed_port().to_string(), &mut engine));
    }

    #[test]
    fn wait_for_engine_times_out_while_the_engine_stays_alive() {
        // HALOGEN_ENGINE_WAIT_S=1 with the 2s poll: the first probe runs at
        // t=0 (below the limit), the second at t=2 — the expiry wording and
        // false (entrypoint lines 763–769). Takes ~2s (the poll interval).
        let _guard = SOCKET_TEST_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let previous = env::var("HALOGEN_ENGINE_WAIT_S").ok();
        env::set_var("HALOGEN_ENGINE_WAIT_S", "1");
        let mut engine = Command::new("sh")
            .args(["-c", "sleep 5"])
            .spawn()
            .expect("spawn sh");
        let result = wait_for_engine(&closed_port().to_string(), &mut engine);
        match previous {
            Some(value) => env::set_var("HALOGEN_ENGINE_WAIT_S", value),
            None => env::remove_var("HALOGEN_ENGINE_WAIT_S"),
        }
        let _ = engine.kill();
        let _ = engine.wait();
        assert!(!result);
    }

    #[test]
    fn wait_limit_follows_the_reference_default_semantics() {
        assert_eq!(parse_wait_limit(None), 0); // unset → 0 (wait forever)
        assert_eq!(parse_wait_limit(Some(String::new())), 0); // `:-` semantics
        assert_eq!(parse_wait_limit(Some("45".to_string())), 45);
        // bash's `[ abc -gt 0 ]` errors, so the branch is not taken → wait
        // forever; the port maps unparseable values to 0 likewise.
        assert_eq!(parse_wait_limit(Some("abc".to_string())), 0);
    }

    #[test]
    fn wd_stat_state_extracts_the_state_character() {
        assert_eq!(wd_stat_state("123 (sh) S 1 123 123 0 -1 65536"), "S");
        // the comm field may contain spaces and parentheses:
        assert_eq!(wd_stat_state("9 (my sh) D 1 9"), "D");
        assert_eq!(wd_stat_state(""), "");
    }

    #[test]
    fn gtt_tenths_matches_the_reference_arithmetic() {
        assert_eq!(gtt_tenths_from_bytes(0), 0);
        assert_eq!(gtt_tenths_from_bytes(1_073_741_824), 10); // exactly 1 GiB
                                                              // 0.8 GiB: r = 1073741822 > half → rounds up to 8 tenths.
        assert_eq!(gtt_tenths_from_bytes(858_993_459), 8);
        // 0.5 GiB + 1 byte: r = 10 → no rounding.
        assert_eq!(gtt_tenths_from_bytes(536_870_913), 5);
    }

    #[test]
    fn gtt_gib_under_formats_a_sysfs_fixture_and_falls_back_to_question() {
        // Fixture inside the crate's `.tmp/` (repo convention for scratch).
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(".tmp")
            .join(format!("gtt-fixture-{}", process::id()));
        let device = root.join("class/drm/card0/device");
        fs::create_dir_all(&device).expect("fixture dirs");
        fs::write(device.join("mem_info_gtt_used"), "1073741824\n").expect("fixture file");
        assert_eq!(gtt_gib_under(&root), "1.0");
        fs::remove_dir_all(&root).expect("fixture cleanup");
        // No drm sysfs at all → the reference's `?` fallback.
        fs::create_dir_all(root.join("class")).expect("empty class dir");
        assert_eq!(gtt_gib_under(&root), "?");
        fs::remove_dir_all(&root).expect("fixture cleanup");
    }

    // -- v0.20.5 additions ---------------------------------------------------

    #[test]
    fn agent_guard_modes_update_the_environment_in_order() {
        // One test for all env-mutating branches: Rust runs tests in parallel
        // threads over a shared process environment, so these must not race.
        let previous = env::var("HALOGEN_AGENT_GUARD").ok();
        let previous_models = env::var("HALOGEN_NPU_MODELS").ok();

        // Off modes leave HALOGEN_NPU_MODELS untouched.
        for raw in [
            "", "0", "off", "OFF", "false", "FALSE", "no", "none", "NoNe",
        ] {
            env::set_var("HALOGEN_AGENT_GUARD", raw);
            env::set_var("HALOGEN_NPU_MODELS", "decider-4b");
            assert_eq!(agent_guard(), Ok(()), "raw {raw:?} must be a no-op");
            assert_eq!(
                env::var("HALOGEN_NPU_MODELS").ok().as_deref(),
                Some("decider-4b")
            );
        }

        // An unknown mode exits 1 and leaves the list alone.
        env::set_var("HALOGEN_AGENT_GUARD", "watchme");
        env::set_var("HALOGEN_NPU_MODELS", "decider-4b");
        assert_eq!(agent_guard(), Err(1));
        assert_eq!(
            env::var("HALOGEN_NPU_MODELS").ok().as_deref(),
            Some("decider-4b")
        );

        // WATCH (case-insensitive) injects the spotter into an empty list.
        env::set_var("HALOGEN_AGENT_GUARD", "WATCH");
        env::remove_var("HALOGEN_NPU_MODELS");
        assert_eq!(agent_guard(), Ok(()));
        assert_eq!(
            env::var("HALOGEN_NPU_MODELS").ok().as_deref(),
            Some("halogen-spotter-0.6b")
        );

        // HOLD adds both beside an existing entry, without duplication.
        env::set_var("HALOGEN_AGENT_GUARD", "Hold");
        env::set_var("HALOGEN_NPU_MODELS", "x,halogen-spotter-0.6b");
        assert_eq!(agent_guard(), Ok(()));
        assert_eq!(
            env::var("HALOGEN_NPU_MODELS").ok().as_deref(),
            Some("x,halogen-spotter-0.6b,decider-4b")
        );

        match previous {
            Some(value) => env::set_var("HALOGEN_AGENT_GUARD", value),
            None => env::remove_var("HALOGEN_AGENT_GUARD"),
        }
        match previous_models {
            Some(value) => env::set_var("HALOGEN_NPU_MODELS", value),
            None => env::remove_var("HALOGEN_NPU_MODELS"),
        }
    }

    #[test]
    fn agent_guard_inject_watch_appends_spotter_once() {
        assert_eq!(agent_guard_inject("watch", ""), "halogen-spotter-0.6b");
        assert_eq!(
            agent_guard_inject("watch", "decider-4b"),
            "decider-4b,halogen-spotter-0.6b"
        );
        // Already present (inside the list or at either end) → unchanged.
        assert_eq!(
            agent_guard_inject("watch", "x,halogen-spotter-0.6b,y"),
            "x,halogen-spotter-0.6b,y"
        );
        assert_eq!(
            agent_guard_inject("watch", "halogen-spotter-0.6b"),
            "halogen-spotter-0.6b"
        );
    }

    #[test]
    fn agent_guard_inject_hold_appends_both_models() {
        assert_eq!(
            agent_guard_inject("hold", ""),
            "halogen-spotter-0.6b,decider-4b"
        );
        assert_eq!(
            agent_guard_inject("hold", "decider-4b"),
            "decider-4b,halogen-spotter-0.6b"
        );
        assert_eq!(
            agent_guard_inject("hold", "halogen-spotter-0.6b,decider-4b"),
            "halogen-spotter-0.6b,decider-4b"
        );
    }

    #[test]
    fn cache_gib_counts_the_reply_entry_with_the_0_20_5_terms() {
        // Defaults: n = s * (1 + 2*2 + 1 + 1) = 4 * 7 = 28 entries; in-place
        // → 28 * 115 MiB = 3220/1024 GiB = 3.144… → "3.1".
        let previous: Vec<(&str, Option<String>)> = [
            "HALOGEN_PROMPT_CACHE",
            "HALOGEN_CACHE_INPLACE",
            "HALOGEN_CACHE_FILE",
            "HALOGEN_CACHE_ENTRIES",
            "HALOGEN_CACHE_BRANCHES",
            "HALOGEN_CACHE_SNAP3",
            "HALOGEN_CACHE_FULL",
            "HALOGEN_CACHE_REPLY",
            "HALOGEN_KV8",
        ]
        .iter()
        .map(|key| (*key, env::var(key).ok()))
        .collect();
        for key in [
            "HALOGEN_PROMPT_CACHE",
            "HALOGEN_CACHE_INPLACE",
            "HALOGEN_CACHE_FILE",
            "HALOGEN_CACHE_ENTRIES",
            "HALOGEN_CACHE_BRANCHES",
            "HALOGEN_CACHE_SNAP3",
            "HALOGEN_CACHE_FULL",
            "HALOGEN_CACHE_REPLY",
            "HALOGEN_KV8",
        ] {
            env::remove_var(key);
        }
        assert_eq!(cache_gib("4", "262144"), "3.1");
        // HALOGEN_CACHE_REPLY=0 drops the reply-end entry: n = 4 * 6 = 24.
        env::set_var("HALOGEN_CACHE_REPLY", "0");
        assert_eq!(cache_gib("4", "262144"), "2.7");
        // HALOGEN_PROMPT_CACHE=1 also suppresses it (the `pm` guard), even
        // with the reply term unset: n = 4 * 6 = 24.
        env::remove_var("HALOGEN_CACHE_REPLY");
        env::set_var("HALOGEN_PROMPT_CACHE", "1");
        assert_eq!(cache_gib("4", "262144"), "2.7");
        // A cache file zeroes the whole figure.
        env::set_var("HALOGEN_CACHE_FILE", "/var/tmp/halogen-cache.snapshot");
        assert_eq!(cache_gib("4", "262144"), "0.0");
        // An explicit entry count bypasses the derivation.
        env::remove_var("HALOGEN_CACHE_FILE");
        env::remove_var("HALOGEN_PROMPT_CACHE");
        env::set_var("HALOGEN_CACHE_ENTRIES", "10");
        assert_eq!(cache_gib("4", "262144"), "1.1");
        for (key, value) in previous {
            match value {
                Some(value) => env::set_var(key, value),
                None => env::remove_var(key),
            }
        }
    }

    #[test]
    fn fe_only_flags_gained_the_0_20_5_entries() {
        for flag in [
            "HALOGEN_RENDER_ONCE",
            "HALOGEN_TOKENIZE_SPLIT",
            "HALOGEN_TOOL_NONE_MASK",
            "HALOGEN_TOKENIZER_NATIVE",
            "HALOGEN_FORK_UNMATCHED_RELEASE",
            "HALOGEN_AGENT_GUARD",
            "HALOGEN_MODEL_NAME_CHECK",
        ] {
            assert!(FE_ONLY_FLAGS.contains(&flag), "{flag} missing");
        }
    }

    #[test]
    fn npu_pin_files_parses_the_record_for_one_id() {
        let record = "model decider-4b repo=x task=decision\n\
                      file decider-4b devices/devices.hnpm 62341 c58e\n\
                      file decider-4b vis/devices.hnpm 100 abcd\n\
                      file other-id devices/devices.hnpm 1 2\n\
                      file decider-4b decider-4b.hnpw 5692927608 b4bb";
        let pins = npu_pin_files(record, "decider-4b");
        assert_eq!(pins.len(), 3);
        assert_eq!(pins[0].path, "devices/devices.hnpm");
        assert_eq!(pins[0].size, 62341);
        assert_eq!(pins[2].path, "decider-4b.hnpw");
    }

    #[test]
    fn npu_part_devices_now_includes_vis_and_aud_pins() {
        let pins = npu_pin_files(
            "file m devices/devices.hnpm 1 a\nfile m vis/devices.hnpm 2 b\n\
             file m aud/devices.hnpm 3 c\nfile m m.hnpw 4 d\n\
             file m tokenizer/tokenizer.json 5 e",
            "m",
        );
        let devices: Vec<&str> = npu_part_pins("devices", &pins)
            .iter()
            .map(|pin| pin.path.as_str())
            .collect();
        assert_eq!(
            devices,
            [
                "devices/devices.hnpm",
                "vis/devices.hnpm",
                "aud/devices.hnpm"
            ]
        );
        assert_eq!(npu_part_pins("vis", &pins).len(), 1);
        assert_eq!(npu_part_pins("aud", &pins).len(), 1);
    }

    #[test]
    fn npu_part_paths_covers_the_new_vis_and_aud_fallbacks() {
        let empty: Vec<&NpuPin> = Vec::new();
        assert_eq!(
            npu_part_paths("vis", "m", &empty, true, false, "m"),
            ["vis/devices.hnpm"]
        );
        assert_eq!(
            npu_part_paths("aud", "m", &empty, true, false, "m"),
            ["aud/devices.hnpm"]
        );
        assert_eq!(
            npu_part_paths("devices", "m", &empty, true, false, "m"),
            ["devices/devices.hnpm"]
        );
        assert_eq!(
            npu_part_paths("all", "m", &empty, true, false, "base"),
            ["m.hnpw", "tokenizer/tokenizer.json"]
        );
        assert_eq!(
            npu_part_paths("all", "m", &empty, true, false, "m"),
            ["devices/devices.hnpm", "m.hnpw", "tokenizer/tokenizer.json"]
        );
    }

    #[test]
    fn npu_elf_error_reports_the_exact_vis_and_aud_messages() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(".tmp")
            .join(format!("elf-fixture-{}", process::id()));
        fs::create_dir_all(root.join("vis")).expect("vis dir");
        fs::create_dir_all(root.join("aud")).expect("aud dir");
        let d = root.display().to_string();
        // Empty directories → the exact 0.20.5 messages.
        assert_eq!(
            npu_elf_error("vis", "m", &d, false).expect("vis error"),
            format!("halogen npu: m: {d}/vis/ holds no .elf file (the NPU's program for the model's vision tower)")
        );
        assert_eq!(
            npu_elf_error("aud", "m", &d, false).expect("aud error"),
            format!("halogen npu: m: {d}/aud/ holds no .elf file (the NPU's program for the model's audio encoder)")
        );
        fs::write(root.join("vis").join("u0.elf"), "x").expect("elf file");
        assert_eq!(npu_elf_error("vis", "m", &d, false), None);
        // A part with no check needed → None.
        assert_eq!(npu_elf_error("all", "m", &d, false), None);
        fs::remove_dir_all(&root).expect("fixture cleanup");
    }

    #[test]
    fn npu_convert_flags_appends_vis_and_aud() {
        assert_eq!(npu_convert_flags(None, None), Vec::<String>::new());
        assert_eq!(
            npu_convert_flags(Some("/models/npu/b/vis"), None),
            ["--vis", "/models/npu/b/vis"]
        );
        assert_eq!(
            npu_convert_flags(Some("/models/npu/b/vis"), Some("/models/npu/a/aud")),
            ["--vis", "/models/npu/b/vis", "--aud", "/models/npu/a/aud"]
        );
    }

    #[test]
    fn npu_vis_need_k_sums_the_vis_directory() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(".tmp")
            .join(format!("vis-needk-{}", process::id()));
        fs::create_dir_all(root.join("vis")).expect("vis dir");
        fs::write(root.join("vis").join("devices.hnpm"), vec![0u8; 2048]).expect("file");
        fs::write(root.join("vis").join("u0.elf"), vec![0u8; 1024]).expect("file");
        assert_eq!(npu_vis_need_k(&root.display().to_string()), 3);
        fs::remove_dir_all(&root).expect("fixture cleanup");
    }

    #[test]
    fn npu_xrt_own_is_data_driven_about_the_record() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(".tmp")
            .join(format!("xrt-fixture-{}", process::id()));
        fs::create_dir_all(&root).expect("fixture dir");
        // Missing record → 0 (the vendored package's live path).
        assert_eq!(
            npu_xrt_own(
                &root.join("missing.sha256").display().to_string(),
                &root.display().to_string(),
                "sha256sum"
            ),
            0
        );
        // Record present but the library missing → still 0.
        fs::write(root.join("xrt.sha256"), "aa libxrt_coreutil.so.2\n").expect("record");
        assert_eq!(
            npu_xrt_own(
                &root.join("xrt.sha256").display().to_string(),
                &root.display().to_string(),
                "sha256sum"
            ),
            0
        );
        fs::remove_dir_all(&root).expect("fixture cleanup");
    }

    #[test]
    fn npu_xrt_own_counts_a_hash_match() {
        // Fixture inside the crate's `.tmp/`: hash a real file with the same
        // helper the entrypoint uses, then check the record against it.
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(".tmp")
            .join(format!("xrt-match-{}", process::id()));
        fs::create_dir_all(&root).expect("fixture dir");
        fs::write(root.join("libxrt_coreutil.so.2"), "xrt contents").expect("lib file");
        let hash = Command::new("sha256sum")
            .arg(root.join("libxrt_coreutil.so.2"))
            .output()
            .expect("sha256sum")
            .stdout;
        let hash = String::from_utf8(hash).expect("utf8");
        let hash = hash.split_whitespace().next().expect("hash").to_string();
        fs::write(
            root.join("xrt.sha256"),
            format!("{hash} libxrt_coreutil.so.2\n"),
        )
        .expect("record");
        assert_eq!(
            npu_xrt_own(
                &root.join("xrt.sha256").display().to_string(),
                &root.display().to_string(),
                "sha256sum"
            ),
            1
        );
        fs::remove_dir_all(&root).expect("fixture cleanup");
    }

    #[test]
    fn npu_xrt_mixed_and_naming_follow_the_0_20_5_rules() {
        assert!(npu_xrt_mixed_error(0, "2.25").is_none());
        assert!(npu_xrt_mixed_error(3, "2.25").is_none());
        assert_eq!(
            npu_xrt_mixed_error(2, "2.25").expect("mixed"),
            "halogen npu: /opt/xilinx/xrt/lib mixes a host's XRT libraries with this image's XRT 2.25. Mount all three of them, or none: with no mount the image uses its own."
        );
        assert_eq!(
            npu_xrt_name("2.25", 3, "whatever"),
            "XRT 2.25 (this image's)"
        );
        // bash strips only the full `libxrt_coreutil.so.` prefix; a name
        // without a dot after that keeps the whole name as the "version".
        assert_eq!(
            npu_xrt_name("", 0, "libxrt_coreutil.so.2.25"),
            "XRT 2.25 (the host's)"
        );
        assert_eq!(
            npu_xrt_name("", 0, "libxrt_coreutil"),
            "the host's XRT (its version is not in the mounted file's name)"
        );
    }

    #[test]
    fn npu_kill_s_validation_follows_the_reference_case() {
        // `:-` semantics: a set-but-empty value also takes the default
        // (bash's `''` case is unreachable once the default substituted).
        assert_eq!(parse_npu_kill_s(None), Ok(1800)); // unset → `:-1800`
        assert_eq!(parse_npu_kill_s(Some(String::new())), Ok(1800));
        assert_eq!(parse_npu_kill_s(Some("60".to_string())), Ok(60));
        assert_eq!(parse_npu_kill_s(Some("0".to_string())), Ok(0));
        assert_eq!(
            parse_npu_kill_s(Some("abc".to_string())),
            Err(
                "halogen npu: HALOGEN_NPU_KILL_S=abc is not a number of seconds (0 = off)"
                    .to_string()
            )
        );
    }

    #[test]
    fn parse_running_ms_extracts_the_field() {
        assert_eq!(
            parse_running_ms(r#"{"running_ms":12345,"x":1}"#),
            Some(12345)
        );
        assert_eq!(parse_running_ms(r#"{"other":1}"#), None);
        assert_eq!(parse_running_ms(r#"{"running_ms":}"#), None);
        assert_eq!(parse_running_ms(r#"{"running_ms":9z}"#), Some(9));
    }

    #[test]
    fn npu_floor_backoff_doubles_to_the_cap() {
        assert!(npu_floor_exit(76));
        assert!(!npu_floor_exit(1));
        assert_eq!(npu_room_s(0, 30), 30);
        assert_eq!(npu_room_s(30, 30), 60);
        assert_eq!(npu_room_s(240, 30), 300);
        assert_eq!(npu_room_s(300, 30), 300);
        // The floor notice names the doubled delay.
        assert_eq!(
            npu_floor_notice(60),
            "halogen npu: the host is under the NPU engine's memory floor (HALOGEN_NPU_HOST_FLOOR_GIB; its line is above): trying again in 60 s, until there is room. Its routes answer 503 until then and the Flash engine serves on"
        );
    }
}
