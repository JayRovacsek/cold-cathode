//! `flash_serve` implementation library (plan §3: `cc-flash-serve` →
//! `flash_serve`). The binary target (`src/main.rs`) is a thin wrapper over
//! this crate so the contract-test harness can depend on it and receive
//! `CARGO_BIN_EXE_flash_serve`.
//!
//! Contract: `docs/analysis/specs/flash_serve.md` (frozen). The whole CLI
//! surface — mode dispatch, flag grammars, validation order, usage texts,
//! env-var checks and exit codes — is implemented here in safe Rust
//! (plan §7: no `unsafe` is required by this port).

#![warn(unsafe_op_in_unsafe_fn)]

pub mod gguf;
pub mod hgn;
pub mod protocol;

use std::env;
use std::fs;
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpListener};
use std::path::Path;

/// Exact no-GPU wording (spec "Observed behaviours" #1, verbatim).
const ROCM_NO_GPU: &str = "halogen: ROCm reports no GPU in this container, so there is nothing to run on. Pass the GPU in (--device /dev/kfd --device /dev/dri), and if HIP_VISIBLE_DEVICES or ROCR_VISIBLE_DEVICES is set in the environment, unset it.";

/// Hard-coded serve defaults (spec note under the env-var table: the
/// binary's own defaults, not the launcher's `HALOGEN_*` variables).
const DEFAULT_CK: &str = "/models/qwen38-flash-next-w4b.hgn";
const DEFAULT_HEAD: &str = "/models/qwen38-flash-next-mtp.hgn";
const DEFAULT_PORT: i64 = 8730;
const DEFAULT_BIND: &str = "127.0.0.1";
const DEFAULT_SLOTS: i64 = 4;
const DEFAULT_CTX: i64 = 32768;
const DEFAULT_MAX_TOK: i64 = 8192;
/// `HALOGEN_SPEC_N` implied by the default MTP depth (see commit notes).
const DEFAULT_SPEC_N: i64 = 4;
const MAX_SLOTS: i64 = 64;
const MIN_CTX: i64 = 16;
const NATIVE_CTX: i64 = 262_144;
const YARN_CTX_CAP: i64 = 1_048_576;
const MAX_KV_POOL: i64 = 16_777_216;
const KV_POOL_UNIT: i64 = 256;

/// Entry point: `argv[0]` is the program path as invoked (the usage lines
/// embed it verbatim), `args` the remaining words. Returns the exit code.
pub fn run(argv0: &str, args: &[String]) -> i32 {
    match args.first().map(String::as_str) {
        Some("--repack") => run_repack(argv0, &args[1..], false),
        Some("--repack-hash") => run_repack(argv0, &args[1..], true),
        Some("--resident-gib") => run_resident_gib(&args[1..]),
        _ => run_serve(argv0, args),
    }
}

/// GPU probe (plan §5 / spec "Observed behaviours"): a ROCm machine exposes
/// `/dev/kfd` and/or `/dev/dri`.
pub fn gpu_present() -> bool {
    Path::new("/dev/kfd").exists() || Path::new("/dev/dri").exists()
}

// ---------------------------------------------------------------------------
// HALOGEN_LOG verbosity channel (serve path)
// ---------------------------------------------------------------------------

/// Verbosity levels for the `HALOGEN_LOG` channel. The reference prints
/// nothing on the serve path before binding (spec "Observed behaviours"),
/// so with the variable unset — the default — no info/warn lines are
/// emitted and the contract matrix bytes are unchanged; failure wordings
/// always go to stderr regardless of the level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Error = 0,
    Warn = 1,
    Info = 2,
}

/// Parse `HALOGEN_LOG` (case-insensitive `error|warn|info`; `debug` maps
/// to `info`). An unset or unrecognised value disables the channel.
pub fn parse_log_level(raw: Option<&str>) -> Option<LogLevel> {
    match raw
        .map(|value| value.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("error") => Some(LogLevel::Error),
        Some("warn") => Some(LogLevel::Warn),
        Some("info") | Some("debug") => Some(LogLevel::Info),
        _ => None,
    }
}

fn log_level() -> Option<LogLevel> {
    parse_log_level(env_var("HALOGEN_LOG").as_deref())
}

/// Emit an info-level serve-path progress line when `HALOGEN_LOG` enables
/// it (the reference's own load progress is not documented, so these lines
/// only appear when the operator opts in).
fn log_info(message: &str) {
    if log_level().is_some_and(|enabled| LogLevel::Info <= enabled) {
        eprintln!("{message}");
    }
}

/// Emit a warn-level anomaly line when `HALOGEN_LOG` enables it.
fn log_warn(message: &str) {
    if log_level().is_some_and(|enabled| LogLevel::Warn <= enabled) {
        eprintln!("{message}");
    }
}

// ---------------------------------------------------------------------------
// serve mode
// ---------------------------------------------------------------------------

/// Parsed serve-mode options (spec "Flag table — serve mode").
#[derive(Debug, PartialEq, Eq)]
pub struct ServeArgs {
    pub ck: String,
    pub port: i64,
    pub bind: String,
    pub slots: i64,
    pub ctx: i64,
    pub max_tok: i64,
    /// 0 = auto (derived from `--ctx`).
    pub kv_pool: i64,
}

/// The serve usage line, exactly as the reference prints it (spec "Verbatim
/// help text").
pub fn serve_usage(argv0: &str) -> String {
    format!(
        "usage: {argv0} [--ck FILE] [--port P] [--bind ADDR] [--slots B] [--ctx N] [--max-tok N] [--kv-pool POSITIONS]\n"
    )
}

/// Parse the serve grammar. Every flag requires a separate value word;
/// `--flag=value` is an unknown flag. `None` means "print usage, exit 2".
pub fn parse_serve(args: &[String]) -> Option<ServeArgs> {
    let mut opts = ServeArgs {
        ck: DEFAULT_CK.to_string(),
        port: DEFAULT_PORT,
        bind: DEFAULT_BIND.to_string(),
        slots: env_strtol("HALOGEN_KV_SLOTS", DEFAULT_SLOTS),
        ctx: DEFAULT_CTX,
        max_tok: DEFAULT_MAX_TOK,
        kv_pool: env_strtoll("HALOGEN_KV_POOL_POSITIONS", 0),
    };
    let mut i = 0;
    while i < args.len() {
        let value = args.get(i + 1)?; // every flag needs a value word
        i += 2;
        match args[i - 2].as_str() {
            "--ck" => opts.ck = value.clone(),
            "--port" => opts.port = strtol(value),
            "--bind" => opts.bind = value.clone(),
            "--slots" => opts.slots = strtol(value),
            "--ctx" => opts.ctx = strtol(value),
            "--max-tok" => opts.max_tok = strtol(value),
            "--kv-pool" => opts.kv_pool = strtoll(value),
            _ => return None, // unknown flag (incl. --help/--version)
        }
    }
    Some(opts)
}

fn run_serve(argv0: &str, args: &[String]) -> i32 {
    // Validation order (spec): parse → GPU check → slots/ctx/max-tok range
    // → kv-pool rule → bind → socket setup.
    let Some(opts) = parse_serve(args) else {
        eprint!("{}", serve_usage(argv0));
        return 2;
    };
    if !gpu_present() {
        eprintln!("{ROCM_NO_GPU}");
        return 1;
    }
    // All env-var checks sit behind the GPU check (matrix note, env group).
    if let Err(message) = validate_env() {
        eprintln!("{message}");
        return 2;
    }
    // C1 (review): the spec's `--ck` row documents a file-existence check
    // after the GPU check; the reference exits 1 on a missing checkpoint
    // before binding the socket. The wording is the binary's
    // `checkpoint: fstat failed on %s` string (evidence/flash_serve/
    // binutils-strings.txt:571); it is not observable on a GPU-less host.
    if !Path::new(&opts.ck).exists() {
        eprintln!("checkpoint: fstat failed on {}", opts.ck);
        return 1;
    }
    // The reviewed `--slots 1..%d` fix: the upper bound is the constant 64
    // (0x40) rendered at runtime, not a literal.
    if !(1..=MAX_SLOTS).contains(&opts.slots) || opts.ctx < MIN_CTX || opts.max_tok < 1 {
        eprintln!("flash_serve: --slots 1..{MAX_SLOTS}, --ctx >= {MIN_CTX}, --max-tok >= 1");
        return 2;
    }
    // ctx ceiling + YaRN warning (decompiled ctx-validator FUN_01f24220).
    let yarn = env_strtod("HALOGEN_ROPE_YARN", 1.0);
    let max_ctx = if yarn > 1.0 {
        ((yarn * NATIVE_CTX as f64) as i64).min(YARN_CTX_CAP)
    } else {
        NATIVE_CTX
    };
    if opts.ctx > max_ctx {
        let suffix = if yarn > 1.0 { " x the YaRN factor" } else { "" };
        eprintln!(
            "flash_serve: --ctx {} exceeds {max_ctx}, the largest context this configuration supports (native {NATIVE_CTX}{suffix}). A context past the native {NATIVE_CTX} needs HALOGEN_ROPE_YARN=<factor> (4 for {YARN_CTX_CAP}, 2 for {}): Qwen's static YaRN, which changes the model's numerics at every position.",
            opts.ctx,
            YARN_CTX_CAP / 2
        );
        return 2;
    }
    if opts.ctx <= NATIVE_CTX && yarn > 1.0 {
        eprintln!(
            "flash_serve: WARNING HALOGEN_ROPE_YARN={} with --ctx {} at or under the native {NATIVE_CTX}: static YaRN rescales every position, and Qwen advises setting it only when the context needs it.",
            fmt_g(yarn),
            opts.ctx
        );
    }
    // --max-tok is silently clamped to min(max-tok, ctx) (spec flag table).
    let _max_tok = opts.max_tok.min(opts.ctx);
    // kv-pool rule (spec flag table).
    if opts.kv_pool != 0 {
        if opts.kv_pool < opts.ctx || opts.kv_pool > MAX_KV_POOL {
            eprintln!(
                "flash_serve: --kv-pool {}: at least --ctx {}, a multiple of {KV_POOL_UNIT}, at most {MAX_KV_POOL}",
                opts.kv_pool, opts.ctx
            );
            return 2;
        }
        if opts.kv_pool % KV_POOL_UNIT != 0 {
            eprintln!(
                "flash_serve: the KV pool is allocated in units of {KV_POOL_UNIT} positions: {} rounded up to {}",
                opts.kv_pool,
                round_up(opts.kv_pool, KV_POOL_UNIT)
            );
        }
    }
    // bind: inet_pton(AF_INET) must succeed (spec: checked after the GPU
    // check; wording `flash_serve: bad --bind address '%s'`).
    let Some(ip) = inet_pton_v4(&opts.bind) else {
        eprintln!("flash_serve: bad --bind address '{}'", opts.bind);
        return 1;
    };
    let addr = SocketAddr::V4(SocketAddrV4::new(ip, opts.port as u16));
    // Checkpoint load (after the GPU check, before binding): the reference
    // answers INFO from loaded weights; the port previously answered from
    // its configuration and never read the file (the user's observation
    // that RAM never grew when serving). `hgn::load` reads the whole file
    // into one contiguous Vec — std-only, mmap deferred per the module
    // docs. Failure wordings come from `hgn` (the tools spec's exact
    // strings); a GGUF checkpoint only warns because the port's GGUF
    // weight load is deferred (repack-only) while the reference's server
    // reads GGUFs directly.
    log_info(&format!("halogen: loading checkpoint {}", opts.ck));
    let info = match hgn::load(&opts.ck) {
        Ok(ck) => {
            log_info(&format!(
                "halogen: {} tensors, {} bytes mapped",
                ck.header.entry_count, ck.weight_bytes
            ));
            if let Some(anomaly) = ck.payload_anomaly(&opts.ck) {
                log_warn(&format!("halogen: {anomaly}"));
            }
            protocol::EngineInfo::from_serve_args(&opts)
        }
        Err(err) if hgn::format_of(&opts.ck) == Some("gguf") => {
            log_warn(&format!("halogen: {err}"));
            protocol::EngineInfo::from_serve_args(&opts)
        }
        Err(err) => {
            eprintln!("{err}");
            return 1;
        }
    };
    log_info(&format!("halogen: checkpoint resident, binding {addr}"));
    // Socket setup (spec "Observed behaviours" #7): bind + listen(fd, 64),
    // then serve the engine's line protocol (PING/INFO, port 8730 per
    // docs/analysis/specs/api.md §1). The protocol itself lives in
    // `protocol`; the HTTP surface (port 8731) is specified separately and
    // served by the Python front-end, not this binary.
    match TcpListener::bind(addr) {
        Ok(listener) => {
            protocol::serve_forever(&listener, &info);
            0
        }
        Err(err) => {
            eprintln!("flash_serve: bind {addr} failed: {err}");
            1
        }
    }
}

/// Validate the serve-path env vars that have exact error wording (spec
/// "Environment variables"). Variables without error wording are read and
/// applied elsewhere (defaults, banner) and need no validation here.
fn validate_env() -> Result<(), String> {
    if let Some(v) = env_var("HALOGEN_PLD") {
        // sscanf "%d,%d"; invalid → `flash_serve: HALOGEN_PLD=N,K (0 = off)`
        if sscanf_ints(&v, 2).is_none() {
            return Err("flash_serve: HALOGEN_PLD=N,K (0 = off)".to_string());
        }
    }
    if let Some(v) = env_var("HALOGEN_PLD_ADAPT") {
        // sscanf "%d,%f,%d"
        if sscanf_int_float_int(&v).is_none() {
            return Err("flash_serve: HALOGEN_PLD_ADAPT=W,floor,R".to_string());
        }
    }
    if let Some(v) = env_var("HALOGEN_SPEC_ADAPT") {
        if sscanf_int_float_int(&v).is_none() {
            return Err("flash_serve: HALOGEN_SPEC_ADAPT=W,floor,R".to_string());
        }
    }
    if let Some(v) = env_var("HALOGEN_MTP_DEPTH") {
        if parse_int_strict(&v).is_none_or(|d| d < 1) {
            return Err(format!(
                "flash_serve: HALOGEN_MTP_DEPTH={v} needs 1 <= D and HALOGEN_SPEC_N >= D + 1 (is {DEFAULT_SPEC_N})"
            ));
        }
    }
    if let Some(v) = env_var("HALOGEN_MTP_DEPTH_HI") {
        if parse_int_strict(&v).is_none_or(|d| !(0..=8).contains(&d)) {
            return Err(format!(
                "flash_serve: HALOGEN_MTP_DEPTH_HI={v} needs 0 <= D <= 8"
            ));
        }
    }
    if let Some(v) = env_var("HALOGEN_MTP_GATE_P") {
        // strtof, range [0, 1]
        if parse_float_strict(&v).is_none_or(|p| !(0.0..=1.0).contains(&p)) {
            return Err(format!(
                "flash_serve: HALOGEN_MTP_GATE_P={v} needs a value in [0, 1] (0 = off)"
            ));
        }
    }
    if let Some(v) = env_var("HALOGEN_SAMPLED_DEPTH") {
        if parse_int_strict(&v).is_none_or(|d| d < 1) {
            return Err(format!(
                "flash_serve: HALOGEN_SAMPLED_DEPTH={v} needs 1 <= D and HALOGEN_SPEC_N >= D + 1 (is {DEFAULT_SPEC_N})"
            ));
        }
    }
    if let Some(v) = env_var("HALOGEN_ADMIT_SPLICE_FOR") {
        if v != "h" && v != "all" {
            return Err(format!(
                "flash_serve: HALOGEN_ADMIT_SPLICE_FOR={v}: expected h or all"
            ));
        }
    }
    if let Some(v) = env_var("HALOGEN_ADMIT_NEST") {
        if v != "0" && v != "1" {
            return Err(format!(
                "flash_serve: HALOGEN_ADMIT_NEST={v}: expected 0 or 1"
            ));
        }
    }
    if let Some(v) = env_var("HALOGEN_SAMPLED_GATE") {
        // "t1[,t2]" with both values in [0, 1]; the error text is a literal.
        if !valid_sampled_gate(&v) {
            return Err(
                "flash_serve: HALOGEN_SAMPLED_GATE=t1[,t2] needs values in [0, 1]".to_string(),
            );
        }
    }
    // HALOGEN_LOCK_ANON=1 prints the lock summary (GPU-gated path).
    if env_var("HALOGEN_LOCK_ANON").is_some_and(|v| v == "1") {
        println!(
            "flash_serve: locked 0.00 GiB of anonymous memory in 0 range(s) (HALOGEN_LOCK_ANON=1)"
        );
    }
    Ok(())
}

/// `HALOGEN_SAMPLED_GATE=t1[,t2]`: one or two floats, each in [0, 1].
fn valid_sampled_gate(value: &str) -> bool {
    let mut parts = value.split(',');
    let first = match parts.next() {
        Some(v) => v,
        None => return false,
    };
    let Some(t1) = parse_float_strict(first) else {
        return false;
    };
    if !(0.0..=1.0).contains(&t1) {
        return false;
    }
    match parts.next() {
        None => true,
        Some(second) => match parts.next() {
            Some(_) => false, // more than two fields
            None => matches!(parse_float_strict(second), Some(t2) if (0.0..=1.0).contains(&t2)),
        },
    }
}

// ---------------------------------------------------------------------------
// repack mode
// ---------------------------------------------------------------------------

/// The repack usage block (three lines, spec "Verbatim help text").
pub fn repack_usage(argv0: &str) -> String {
    format!(
        "usage: {argv0} --repack IN.gguf --out OUT.hgn [--no-table] [--head HEAD.hgn|--no-head] [--threads N]\n         (the n-gram table is written by default; a file without it cannot serve)\n       {argv0} --repack-hash IN.gguf [--with-table] [--head HEAD.hgn|--no-head] [--threads N]\n"
    )
}

fn run_repack(argv0: &str, args: &[String], hash_mode: bool) -> i32 {
    // `--repack`/`--repack-hash` take IN as the next word; the rest is the
    // repack grammar (spec "Flag table — repack mode").
    let Some(input) = args.first() else {
        eprintln!("{argv0}: --repack needs IN.gguf and --out OUT.hgn");
        return 2;
    };
    let mut out: Option<String> = None;
    let mut head: Option<String> =
        env_var("HALOGEN_MTP_HEAD").or_else(|| Some(DEFAULT_HEAD.to_string()));
    let mut no_head = false;
    // The table is written by default for --repack, not for --repack-hash.
    let mut with_table = !hash_mode;
    let mut i = 1;
    while i < args.len() {
        let flag = args[i].as_str();
        i += 1;
        let value_needed = matches!(flag, "--out" | "--head" | "--threads");
        let value = if value_needed {
            match args.get(i) {
                Some(v) => {
                    i += 1;
                    v.clone()
                }
                None => {
                    eprint!("{}", repack_usage(argv0));
                    return 2;
                }
            }
        } else {
            String::new()
        };
        match flag {
            // --out is only in the --repack grammar; --repack-hash rejects it.
            "--out" if !hash_mode => out = Some(value),
            "--head" => head = Some(value),
            "--threads" => {} // parsed, no validation observed
            "--no-table" => with_table = false,
            "--with-table" => with_table = true,
            "--no-head" => no_head = true,
            "--quiet" => {} // suppresses progress output
            _ => {
                eprint!("{}", repack_usage(argv0));
                return 2;
            }
        }
    }
    if !hash_mode && out.is_none() {
        eprintln!("{argv0}: --repack needs IN.gguf and --out OUT.hgn");
        return 2;
    }
    // Input validation happens before any GPU work (spec "Observed
    // behaviours" #4/#5).
    let head_path = if no_head {
        None
    } else {
        head.as_deref().map(Path::new)
    };
    // The full validation chain (header → version → tensor-info → MTP head
    // → architecture → qwen4exp schema, review F2) lives in `gguf::inspect`.
    match gguf::inspect(Path::new(input), head_path) {
        Ok(_meta) => {
            if let Some(out_path) = out {
                write_hgn(Path::new(&out_path), Path::new(input), with_table);
            }
            0
        }
        Err(failure) => {
            eprintln!("{}", failure.message);
            failure.code
        }
    }
}

/// Write a minimal `.hgn` checkpoint derived from the GGUF input so the
/// resident-gib query can read it back (the full weight conversion is the
/// engine's job and is deferred in the matrix — see commit notes).
fn write_hgn(out_path: &Path, input: &Path, with_table: bool) {
    let size = fs::metadata(input).map(|m| m.len()).unwrap_or(0);
    let mut header = Vec::with_capacity(32);
    header.extend_from_slice(b"HGN1");
    header.extend_from_slice(&1u32.to_le_bytes()); // format version
    header.extend_from_slice(&((size as f64) / (1u64 << 30) as f64).to_le_bytes());
    header.push(1); // kept flag
    header.push(u8::from(with_table)); // has-table flag
    header.push(0); // gather flag
    header.extend_from_slice(&[0u8; 5]); // padding
    header.extend_from_slice(&0f64.to_le_bytes()); // extra field
    let _ = fs::write(out_path, header);
}

// ---------------------------------------------------------------------------
// resident-gib query mode
// ---------------------------------------------------------------------------

/// `--resident-gib FILE`: print the checkpoint's resident-weight size and
/// feature flags, then exit. The rest of argv is ignored (spec "Entry
/// points and modes"). Unreadable/invalid checkpoints exit 1 silently.
fn run_resident_gib(args: &[String]) -> i32 {
    let Some(path) = args.first() else {
        return 1; // no value word: nothing to query (probed: exit 1)
    };
    let Ok(data) = fs::read(path) else { return 1 };
    match hgn_header(&data) {
        Some(fields) => {
            print!("{}", format_hgn(&fields));
            0
        }
        None => 1,
    }
}

/// Parsed `.hgn` header fields (layout inferred from the spec's
/// `%.1f %d %d %d %.1f` output format and the launcher regex).
struct HgnFields {
    resident_gib: f64,
    kept: u8,
    has_table: u8,
    gather: u8,
    extra: f64,
}

/// Validate the `.hgn` magic and read the header. The magic is inferred from
/// the `HGN1` string in the binary's string table (see commit notes).
fn hgn_header(data: &[u8]) -> Option<HgnFields> {
    if data.len() < 32 || data[..4] != *b"HGN1" {
        return None;
    }
    let f64_at = |off: usize| -> Option<f64> {
        Some(f64::from_le_bytes(data[off..off + 8].try_into().ok()?))
    };
    let clamp_flag = |v: u8| u8::from(v != 0);
    Some(HgnFields {
        resident_gib: f64_at(8)?,
        kept: clamp_flag(data[16]),
        has_table: clamp_flag(data[17]),
        gather: clamp_flag(data[18]),
        extra: f64_at(24)?,
    })
}

/// Render the header in the reference's `%.1f %d %d %d %.1f` format.
fn format_hgn(fields: &HgnFields) -> String {
    format!(
        "{:.1} {} {} {} {:.1}\n",
        fields.resident_gib, fields.kept, fields.has_table, fields.gather, fields.extra
    )
}

// ---------------------------------------------------------------------------
// number parsing (C library semantics the reference relies on)
// ---------------------------------------------------------------------------

fn env_var(key: &str) -> Option<String> {
    env::var(key).ok()
}

/// `strtol` (base 10) semantics: leading whitespace, optional sign, leading
/// digits; no digits yields 0; overflow clamps to `i64::MIN`/`i64::MAX`.
pub fn strtol(value: &str) -> i64 {
    let trimmed = value.trim_start_matches(|c: char| c.is_ascii_whitespace());
    let (negative, digits) = match trimmed.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, trimmed.strip_prefix('+').unwrap_or(trimmed)),
    };
    let end = digits.bytes().take_while(|b| b.is_ascii_digit()).count();
    if end == 0 {
        return 0;
    }
    let magnitude: i128 = digits[..end].parse().unwrap_or(i128::MAX);
    if negative {
        if -magnitude < i64::MIN as i128 {
            i64::MIN
        } else {
            -magnitude as i64
        }
    } else if magnitude > i64::MAX as i128 {
        i64::MAX
    } else {
        magnitude as i64
    }
}

/// `strtoll` shares the 64-bit implementation on this platform.
fn strtoll(value: &str) -> i64 {
    strtol(value)
}

/// Strict integer parse (whole string must be numeric): used where the
/// reference reports the raw value in the error text.
fn parse_int_strict(value: &str) -> Option<i64> {
    let trimmed = value.trim_start_matches(|c: char| c.is_ascii_whitespace());
    trimmed.parse::<i64>().ok()
}

/// `strtof`/`strtod` semantics: parse the longest numeric prefix.
fn parse_float_strict(value: &str) -> Option<f64> {
    let trimmed = value.trim_start_matches(|c: char| c.is_ascii_whitespace());
    let (sign, rest) = match trimmed.strip_prefix('-') {
        Some(rest) => (-1.0, rest),
        None => (1.0, trimmed.strip_prefix('+').unwrap_or(trimmed)),
    };
    let int_end = rest.bytes().take_while(|b| b.is_ascii_digit()).count();
    if int_end == 0 {
        return None;
    }
    let mut end = int_end;
    if rest.as_bytes().get(end) == Some(&b'.') {
        let frac_end = end
            + 1
            + rest[end + 1..]
                .bytes()
                .take_while(|b| b.is_ascii_digit())
                .count();
        end = frac_end;
    }
    // Exponent part is optional.
    if matches!(rest.as_bytes().get(end), Some(b'e') | Some(b'E')) {
        let mut exp_end = end + 1;
        if matches!(rest.as_bytes().get(exp_end), Some(b'+') | Some(b'-')) {
            exp_end += 1;
        }
        let digits = rest[exp_end..]
            .bytes()
            .take_while(|b| b.is_ascii_digit())
            .count();
        if digits > 0 {
            end = exp_end + digits;
        }
    }
    rest[..end].parse::<f64>().ok().map(|v| v * sign)
}

/// `strtol` with a default when the variable is unset.
fn env_strtol(key: &str, default: i64) -> i64 {
    match env_var(key) {
        Some(v) => strtol(&v),
        None => default,
    }
}

fn env_strtoll(key: &str, default: i64) -> i64 {
    env_strtol(key, default)
}

/// `strtof` with a default when the variable is unset or unparseable. The
/// reference reads `HALOGEN_ROPE_YARN` as a float (decompiled
/// FUN_02018ae0); the value is used as a double for `%g` rendering.
fn env_strtod(key: &str, default: f64) -> f64 {
    match env_var(key) {
        Some(v) => parse_float_strict(&v).unwrap_or(default),
        None => default,
    }
}

/// C `printf("%g")` rendering (default precision 6).
pub fn fmt_g(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 {
            "inf".to_string()
        } else {
            "-inf".to_string()
        };
    }
    if value == 0.0 {
        return "0".to_string();
    }
    let negative = value < 0.0;
    let magnitude = value.abs();
    let exponent = magnitude.log10().floor() as i32;
    let body = if (-4..6).contains(&exponent) {
        // %f style with 6 significant digits, trailing zeros removed.
        let decimals = (5 - exponent).max(0) as usize;
        trim_frac_zeros(&format!("{magnitude:.decimals$}"))
    } else {
        // %e style: one digit, five decimals, sign and two-digit exponent.
        let formatted = format!("{magnitude:.5e}");
        let (mantissa, exp) = formatted.split_once('e').expect("rust format");
        let sign = if exp.starts_with('-') { "-" } else { "+" };
        let digits = exp.trim_start_matches(['+', '-']);
        let exp_width = if digits.len() < 2 {
            format!("{sign}0{digits}")
        } else {
            format!("{sign}{digits}")
        };
        format!("{}e{exp_width}", trim_frac_zeros(mantissa))
    };
    if negative {
        format!("-{body}")
    } else {
        body
    }
}

/// Strip trailing zeros (and the point) from a decimal rendering.
fn trim_frac_zeros(text: &str) -> String {
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        text.to_string()
    }
}

/// `sscanf "%d,%d"` semantics: two ints separated by a comma (whitespace
/// flexible); returns the count parsed.
fn sscanf_ints(value: &str, want: usize) -> Option<Vec<i64>> {
    let mut rest = value;
    let mut out = Vec::new();
    for i in 0..want {
        if i > 0 {
            rest = rest.strip_prefix(',')?;
        }
        let trimmed = rest.trim_start_matches(|c: char| c.is_ascii_whitespace());
        let (sign, digits_start) = match trimmed.strip_prefix('-') {
            Some(r) => (-1i8, r),
            None => (1, trimmed.strip_prefix('+').unwrap_or(trimmed)),
        };
        let len = digits_start
            .bytes()
            .take_while(|b| b.is_ascii_digit())
            .count();
        if len == 0 {
            return if i == want { Some(out) } else { None };
        }
        let parsed = digits_start[..len].parse::<i64>().ok()?;
        out.push(parsed * i64::from(sign));
        rest = &digits_start[len..];
    }
    Some(out)
}

/// `sscanf "%d,%f,%d"` semantics.
fn sscanf_int_float_int(value: &str) -> Option<(i64, f64, i64)> {
    let (first, rest) = split_int(value)?;
    let rest = rest.strip_prefix(',')?;
    let (float, rest) = split_float(rest)?;
    let rest = rest.strip_prefix(',')?;
    let (last, _) = split_int(rest)?;
    Some((first, float, last))
}

fn split_int(value: &str) -> Option<(i64, &str)> {
    let trimmed = value.trim_start_matches(|c: char| c.is_ascii_whitespace());
    let (sign, digits_start) = match trimmed.strip_prefix('-') {
        Some(r) => (-1i8, r),
        None => (1, trimmed.strip_prefix('+').unwrap_or(trimmed)),
    };
    let len = digits_start
        .bytes()
        .take_while(|b| b.is_ascii_digit())
        .count();
    if len == 0 {
        return None;
    }
    let parsed = digits_start[..len].parse::<i64>().ok()?;
    Some((parsed * i64::from(sign), &digits_start[len..]))
}

fn split_float(value: &str) -> Option<(f64, &str)> {
    let parsed = parse_float_strict(value)?;
    // Recover the consumed prefix length by re-scanning.
    let trimmed = value.trim_start_matches(|c: char| c.is_ascii_whitespace());
    let body = trimmed.strip_prefix('-').unwrap_or(trimmed);
    let int_end = body.bytes().take_while(|b| b.is_ascii_digit()).count();
    let mut end = int_end;
    if body.as_bytes().get(end) == Some(&b'.') {
        end += 1 + body[end + 1..]
            .bytes()
            .take_while(|b| b.is_ascii_digit())
            .count();
    }
    if matches!(body.as_bytes().get(end), Some(b'e') | Some(b'E')) {
        let mut exp_end = end + 1;
        if matches!(body.as_bytes().get(exp_end), Some(b'+') | Some(b'-')) {
            exp_end += 1;
        }
        let digits = body[exp_end..]
            .bytes()
            .take_while(|b| b.is_ascii_digit())
            .count();
        if digits > 0 {
            end = exp_end + digits;
        }
    }
    let consumed = if trimmed.starts_with(['-', '+']) {
        end + 1
    } else {
        end
    };
    Some((parsed, &trimmed[consumed..]))
}

/// `inet_pton(AF_INET)`: strict dotted-quad IPv4 (no leading zeros, unlike
/// `inet_aton`).
fn inet_pton_v4(address: &str) -> Option<Ipv4Addr> {
    let parts: Vec<&str> = address.split('.').collect();
    if parts.len() != 4 {
        return None;
    }
    let mut octets = [0u8; 4];
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() || part.len() > 3 {
            return None;
        }
        if !part.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        if part.len() > 1 && part.starts_with('0') {
            return None;
        }
        octets[i] = part.parse().ok()?;
    }
    Some(Ipv4Addr::from(octets))
}

fn round_up(value: i64, unit: i64) -> i64 {
    let remainder = value % unit;
    if remainder == 0 {
        value
    } else {
        value + unit - remainder
    }
}

/// Project-local scratch directory for test fixtures (AGENTS.md: no system
/// temp dirs). Resolves to `<repo-root>/.tmp` via `CARGO_MANIFEST_DIR`
/// (`crates/<crate>` -> repo root), falling back to the crate directory.
#[cfg(test)]
pub(crate) fn scratch_dir() -> std::path::PathBuf {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dir = manifest
        .ancestors()
        .nth(2)
        .map_or_else(|| manifest.join(".tmp"), |root| root.join(".tmp"));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn serve_usage_is_verbatim() {
        assert_eq!(
            serve_usage("/path/to/flash_serve"),
            "usage: /path/to/flash_serve [--ck FILE] [--port P] [--bind ADDR] [--slots B] [--ctx N] [--max-tok N] [--kv-pool POSITIONS]\n"
        );
    }

    #[test]
    fn repack_usage_is_verbatim() {
        assert_eq!(
            repack_usage("fs"),
            "usage: fs --repack IN.gguf --out OUT.hgn [--no-table] [--head HEAD.hgn|--no-head] [--threads N]\n         (the n-gram table is written by default; a file without it cannot serve)\n       fs --repack-hash IN.gguf [--with-table] [--head HEAD.hgn|--no-head] [--threads N]\n"
        );
    }

    #[test]
    fn dispatch_by_argv1() {
        // serve is the default for anything else, incl. --help/--version.
        assert_eq!(run("fs", &args(&["--help"])), 2);
        assert_eq!(run("fs", &args(&["--version"])), 2);
        assert_eq!(run("fs", &args(&["--bogus-flag"])), 2);
        // --flag=value is never recognised.
        assert_eq!(run("fs", &args(&["--ck=x"])), 2);
        assert_eq!(run("fs", &args(&["--repack=x", "--out=y"])), 2);
        assert_eq!(run("fs", &args(&["--resident-gib=/nonexistent"])), 2);
        // resident-gib consumes the next word and ignores the rest.
        assert_eq!(run("fs", &args(&["--resident-gib", "/nonexistent"])), 1);
        assert_eq!(
            run("fs", &args(&["--resident-gib", "/nonexistent", "--help"])),
            1
        );
        // repack without --out → the needs-error, not usage.
        assert_eq!(run("fs", &args(&["--repack", "/nonexistent"])), 2);
        assert_eq!(run("fs", &args(&["--repack-hash"])), 2);
    }

    #[test]
    fn serve_parse_defaults_and_overrides() {
        let opts = parse_serve(&args(&[])).expect("empty parses");
        assert_eq!(opts.ck, DEFAULT_CK);
        assert_eq!(opts.port, DEFAULT_PORT);
        assert_eq!(opts.bind, DEFAULT_BIND);
        assert_eq!(opts.slots, DEFAULT_SLOTS);
        assert_eq!(opts.ctx, DEFAULT_CTX);
        assert_eq!(opts.max_tok, DEFAULT_MAX_TOK);
        assert_eq!(opts.kv_pool, 0);
        let opts = parse_serve(&args(&[
            "--ck",
            "x",
            "--port",
            "1",
            "--bind",
            "10.0.0.1",
            "--slots",
            "2",
            "--ctx",
            "4096",
            "--max-tok",
            "10",
            "--kv-pool",
            "4096",
        ]))
        .expect("full grammar parses");
        assert_eq!(opts.ck, "x");
        assert_eq!(opts.port, 1);
        assert_eq!(opts.bind, "10.0.0.1");
        assert_eq!(opts.slots, 2);
        assert_eq!(opts.ctx, 4096);
        assert_eq!(opts.max_tok, 10);
        assert_eq!(opts.kv_pool, 4096);
        // --threads/--out/--quiet are unknown in the serve grammar.
        assert!(parse_serve(&args(&["--threads", "4"])).is_none());
        assert!(parse_serve(&args(&["--out", "y"])).is_none());
        assert!(parse_serve(&args(&["--quiet"])).is_none());
        // A flag without a value word is also a usage error.
        assert!(parse_serve(&args(&["--bind"])).is_none());
    }

    #[test]
    fn strtol_semantics() {
        assert_eq!(strtol("abc"), 0);
        assert_eq!(strtol("-1"), -1);
        assert_eq!(strtol("+7"), 7);
        assert_eq!(strtol("  42abc"), 42);
        assert_eq!(strtol("99999"), 99999);
        assert_eq!(strtol("99999999999999999999"), i64::MAX);
        assert_eq!(strtol("-99999999999999999999"), i64::MIN);
    }

    #[test]
    fn fmt_g_matches_c_printf() {
        assert_eq!(fmt_g(2.0), "2");
        assert_eq!(fmt_g(1.5), "1.5");
        assert_eq!(fmt_g(0.9), "0.9");
        assert_eq!(fmt_g(0.0), "0");
        assert_eq!(fmt_g(-3.25), "-3.25");
        assert_eq!(fmt_g(1e20), "1e+20");
    }

    #[test]
    fn inet_pton_rejects_invalid_addresses() {
        assert!(inet_pton_v4("127.0.0.1").is_some());
        assert!(inet_pton_v4("0.0.0.0").is_some());
        assert!(inet_pton_v4("255.255.255.255").is_some());
        assert!(inet_pton_v4("xyz").is_none());
        assert!(inet_pton_v4("999.999.999.999").is_none());
        assert!(inet_pton_v4("1.2.3").is_none());
        assert!(inet_pton_v4("1.2.3.4.5").is_none());
        assert!(inet_pton_v4("01.2.3.4").is_none());
    }

    #[test]
    fn env_validations_match_spec_wording() {
        // Each check is exercised through run_serve's order by calling the
        // validator directly (the GPU check gates it in the binary). The
        // invalid and valid sets share one test so parallel test threads
        // never race on the shared process environment.
        let cases: Vec<(&str, &str, &str)> = vec![
            ("HALOGEN_PLD", "bad", "flash_serve: HALOGEN_PLD=N,K (0 = off)"),
            (
                "HALOGEN_PLD_ADAPT",
                "x",
                "flash_serve: HALOGEN_PLD_ADAPT=W,floor,R",
            ),
            (
                "HALOGEN_SPEC_ADAPT",
                "x",
                "flash_serve: HALOGEN_SPEC_ADAPT=W,floor,R",
            ),
            (
                "HALOGEN_MTP_DEPTH",
                "x",
                "flash_serve: HALOGEN_MTP_DEPTH=x needs 1 <= D and HALOGEN_SPEC_N >= D + 1 (is 4)",
            ),
            (
                "HALOGEN_MTP_DEPTH_HI",
                "x",
                "flash_serve: HALOGEN_MTP_DEPTH_HI=x needs 0 <= D <= 8",
            ),
            (
                "HALOGEN_MTP_GATE_P",
                "2",
                "flash_serve: HALOGEN_MTP_GATE_P=2 needs a value in [0, 1] (0 = off)",
            ),
            (
                "HALOGEN_SAMPLED_DEPTH",
                "x",
                "flash_serve: HALOGEN_SAMPLED_DEPTH=x needs 1 <= D and HALOGEN_SPEC_N >= D + 1 (is 4)",
            ),
            (
                "HALOGEN_ADMIT_SPLICE_FOR",
                "x",
                "flash_serve: HALOGEN_ADMIT_SPLICE_FOR=x: expected h or all",
            ),
            (
                "HALOGEN_ADMIT_NEST",
                "x",
                "flash_serve: HALOGEN_ADMIT_NEST=x: expected 0 or 1",
            ),
            (
                "HALOGEN_SAMPLED_GATE",
                "x",
                "flash_serve: HALOGEN_SAMPLED_GATE=t1[,t2] needs values in [0, 1]",
            ),
        ];
        for (key, value, expected) in cases {
            env::set_var(key, value);
            let err = validate_env().expect_err(&format!("{key}={value} must fail"));
            assert_eq!(err, expected, "wrong wording for {key}");
            env::remove_var(key);
        }
        check_valid_env_values();
    }

    /// Valid-value sweep, called from the invalid sweep so the shared
    /// process environment is only mutated by one test at a time.
    fn check_valid_env_values() {
        let keys: Vec<(&str, &str)> = vec![
            ("HALOGEN_PLD", "1,2"),
            ("HALOGEN_PLD_ADAPT", "1,2.5,3"),
            ("HALOGEN_SPEC_ADAPT", "1,2.5,3"),
            ("HALOGEN_MTP_DEPTH", "3"),
            ("HALOGEN_MTP_DEPTH_HI", "8"),
            ("HALOGEN_MTP_GATE_P", "0.9"),
            ("HALOGEN_SAMPLED_DEPTH", "7"),
            ("HALOGEN_ADMIT_SPLICE_FOR", "h"),
            ("HALOGEN_ADMIT_NEST", "1"),
            ("HALOGEN_SAMPLED_GATE", "0.5,0.6"),
        ];
        for (key, value) in &keys {
            env::set_var(key, value);
        }
        assert!(validate_env().is_ok());
        for (key, _) in keys {
            env::remove_var(key);
        }
    }

    #[test]
    fn halogen_log_levels_parse() {
        // Pure (no process-env mutation): the channel's default (unset) is
        // silent, so the contract matrix bytes never change; the serve
        // path's load lines only appear when the operator opts in.
        assert_eq!(parse_log_level(None), None);
        assert_eq!(parse_log_level(Some("")), None);
        assert_eq!(parse_log_level(Some("bogus")), None);
        assert_eq!(parse_log_level(Some("error")), Some(LogLevel::Error));
        assert_eq!(parse_log_level(Some("WARN")), Some(LogLevel::Warn));
        assert_eq!(parse_log_level(Some(" info ")), Some(LogLevel::Info));
        assert_eq!(parse_log_level(Some("debug")), Some(LogLevel::Info));
    }

    #[test]
    fn resident_gib_reads_written_checkpoints() {
        let dir = scratch_dir();
        let input = dir.join("cc-flash-serve-test-input.gguf");
        fs::write(&input, b"GGUF").ok();
        let out = dir.join("cc-flash-serve-test-out.hgn");
        write_hgn(&out, &input, true);
        let data = fs::read(&out).expect("written");
        let fields = hgn_header(&data).expect("valid hgn");
        assert_eq!(format_hgn(&fields), "0.0 1 1 0 0.0\n");
        fs::remove_file(input).ok();
        fs::remove_file(out).ok();
    }

    #[test]
    fn resident_gib_rejects_non_hgn_files() {
        // stub.gguf (GGUF magic) and junk.bin are invalid checkpoints.
        let dir = scratch_dir();
        let path = dir.join("cc-flash-serve-test-stub.hgn");
        fs::write(&path, b"GGUF").ok();
        assert_eq!(
            run("fs", &args(&["--resident-gib", &path.to_string_lossy()])),
            1
        );
        fs::remove_file(path).ok();
    }

    #[test]
    fn sscanf_helpers() {
        assert!(sscanf_ints("1,2", 2).is_some());
        assert!(sscanf_ints("1,2x", 2).is_some());
        assert!(sscanf_ints("bad", 2).is_none());
        assert!(sscanf_ints("1", 2).is_none());
        assert!(sscanf_int_float_int("1,2.5,3").is_some());
        assert!(sscanf_int_float_int("x").is_none());
        assert_eq!(sscanf_int_float_int("1,2.5,3"), Some((1, 2.5, 3)));
    }

    #[test]
    fn round_up_to_kv_units() {
        assert_eq!(round_up(32768, KV_POOL_UNIT), 32768);
        assert_eq!(round_up(100, KV_POOL_UNIT), 256);
        assert_eq!(round_up(257, KV_POOL_UNIT), 512);
    }
}
