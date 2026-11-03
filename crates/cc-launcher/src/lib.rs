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
//!   `kv_budget_note`, `gtt_note`, `wait_for_engine`, `engine_watchdog`,
//!   `stop_engine`, `wd_gtt_after_exit`, the NPU helpers) are ported
//!   structurally: the host preflight, checkpoint/tokenizer/draft-head
//!   checks and their exact messages are reproduced; the GPU-dependent
//!   orchestration (engine wait, watchdog, NPU restart, signal traps) and
//!   the `memlock_note` weights-lock probe (whose reference body needs
//!   `ckpt_facts`, which execs `flash_serve --resident-gib`) are not. Those
//!   modes are `deferred` in the contract matrix
//!   (`side-effecting-launch`).
//! * `exec` is emulated with `Command::status` (std has no `execve`): the
//!   child's exit code is returned through `main`; a missing program yields
//!   exit 127 and a non-executable one 126, matching bash.

#![warn(unsafe_op_in_unsafe_fn)]

use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{self, Child, Command, Stdio};

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
    "HALOGEN_FORK_AUTO_END_ONLY",
    "HALOGEN_FORK_AUTO_KEEPS",
    "HALOGEN_FORK_HANDLES",
    "HALOGEN_FORK_MAP",
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
    let api = resolve_python().map(|python| {
        let tokenizer = env::var("HALOGEN_TOKENIZER").unwrap_or_else(|_| "/tokenizer".to_string());
        let api_args = eng.api_args(&tokenizer, "0.0.0.0", &format!("127.0.0.1:{}", eng.port));
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
    args.extend(eng.api_args(&tokenizer, "0.0.0.0", &engine));
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
