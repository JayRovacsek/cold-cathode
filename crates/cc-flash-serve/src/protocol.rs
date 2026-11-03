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
//!
//! The whole module is pure except for `serve_connection`/`serve_forever`,
//! which only move bytes; `handle_line` is unit-testable without a socket.

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;

use crate::{env_strtod, env_var, fmt_g, sscanf_ints, ServeArgs};

/// Everything INFO reports that the port can derive from the parsed serve
/// configuration plus the environment (no cross-crate coupling to the
/// `.hgn` reader: the reference answers from loaded weights, the port
/// answers from its configuration, per the "CLI args + env + version" rule).
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
}

impl EngineInfo {
    /// Build the INFO state from parsed serve args plus the environment
    /// (reads `HALOGEN_IMAGE_VERSION`, `HALOGEN_ROPE_YARN`, `HALOGEN_PLD`).
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
/// The port fills the configuration-derived fields from `EngineInfo` and
/// the weight-derived fields with the honest zero/off values (the port's
/// checkpoint load is the deferred GPU-gated path): `mtp`, `draft_head`,
/// `drafter_weights`, `dflash2`, `cache_mb`, `cache_align`, `vision`,
/// `grammar`, `composable`, `cc_floor`, `cc_seam`, `cc_bytes`, `tools`,
/// `keep_end`, `masks_images` are 0; `cache_mode` is 1, `indexer_budget`
/// 2048, `kv_cache` "bf16", `cc_fingerprint` "-" and `checkpoint_format`
/// "hgn" — the same defaults serve_api's own `_fallback_info` (:1712–1717)
/// assumes. Token 13 (`sampling`) is the literal `1` of the reference
/// format string.
pub fn info_line(info: &EngineInfo) -> String {
    format!(
        "I 0 0 {ctx} 0 {max_tok} 0 0 0 0 {slots} {slot_ctx} 1 1 {rope} {kv_pool} 0 {version} {pld_n} {pld_k} hgn 0 0 0 0 0 - 2048 0 bf16 0 0",
        ctx = info.ctx,
        max_tok = info.max_tok,
        slots = info.slots,
        slot_ctx = info.slot_ctx,
        rope = fmt_g(info.rope_factor),
        kv_pool = info.kv_pool,
        version = info.version,
        pld_n = info.pld_n,
        pld_k = info.pld_k,
    )
}

/// Answer one protocol line; `Some(reply)` (without the trailing newline)
/// is written back and the connection stays open, `None` closes it.
///
/// Commands, per the reference's verb set (the `<$PING`/`<$INFO` compare
/// strings, binutils-strings.txt:282377–282378, and serve_api's own verb
/// usage: `PING`, `INFO`, `GEN`, `HREL`, `VOCAB`, `MERGES`):
///
/// * `PING` → `PONG` (serve_api `probe_live` accepts only `PONG`).
/// * `INFO` → the capability line (see [`info_line`]).
/// * `GEN …` — deferred, not faked: the reference answers from loaded
///   weights and the port's checkpoint load is the GPU-gated deferred path.
///   The port closes the connection instead of returning an error line:
///   serve_api's `_demux` (:1659–1679) only consumes lines starting with
///   `T`/`D`/`P`/`G`/`C`, so an error line would be ignored and the
///   request would hang until timeout, while EOF ends every stream cleanly
///   (`_demux` pushes `None`, failing the request immediately).
/// * `HREL`, `VOCAB`, `MERGES` and every unknown verb → close. The
///   reference has no error-reply string for unrecognised verbs (it logs
///   `queued connection closed without a request`, :3761, when a connection
///   dies without one), so closing is the evidenced default; the front-end
///   reconnects via `_ensure` (:1774–1779).
pub fn handle_line(request: &str, info: &EngineInfo) -> Option<String> {
    let line = request.trim_end_matches(['\r', '\n']);
    let verb = line.split_whitespace().next()?;
    match verb {
        "PING" => Some("PONG".to_string()),
        "INFO" => Some(info_line(info)),
        _ => None,
    }
}

/// Serve one connection to completion: answer lines until the protocol
/// closes it (or EOF). The reply framing is one line per request with a
/// trailing `\n`, matching the reference's line-based protocol.
pub fn serve_connection(stream: TcpStream, info: &EngineInfo) {
    let mut writer = match stream.try_clone() {
        Ok(clone) => clone,
        Err(_) => return,
    };
    let mut reader = BufReader::new(stream);
    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let Some(reply) = handle_line(&line, info) else {
            break;
        };
        if writer.write_all(reply.as_bytes()).is_err() || writer.write_all(b"\n").is_err() {
            break;
        }
        let _ = writer.flush();
    }
}

/// The accept loop behind `run_serve`: bind succeeded, so answer every
/// connection on its own thread (the reference serves `--slots` concurrent
/// sessions; one thread per connection keeps a persistent probe connection
/// from blocking new clients). Blocks until `incoming()` errors.
pub fn serve_forever(listener: &TcpListener, info: &EngineInfo) {
    for conn in listener.incoming() {
        let Ok(stream) = conn else { break };
        let info = info.clone();
        thread::spawn(move || serve_connection(stream, &info));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{SocketAddr, TcpListener};
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
        }
    }

    #[test]
    fn ping_answers_pong() {
        let info = fixture_info();
        assert_eq!(handle_line("PING", &info).as_deref(), Some("PONG"));
        assert_eq!(handle_line("PING\n", &info).as_deref(), Some("PONG"));
        assert_eq!(handle_line("PING\r\n", &info).as_deref(), Some("PONG"));
        // The reference compares the verb exactly (serve_api sends `PING\n`).
        assert_eq!(handle_line("ping", &info), None);
        assert_eq!(handle_line("PINGO", &info), None);
    }

    #[test]
    fn info_reply_is_the_reference_format_string_instantiated() {
        // The exact 32-token line the reference's format string
        // (binutils-strings.txt:3752) produces for the fixture state;
        // token 17 (the `%s`) carries the version.
        let info = fixture_info();
        let reply = handle_line("INFO", &info).expect("INFO answers");
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
    }

    #[test]
    fn gen_and_unknown_verbs_close_the_connection() {
        // GEN is deferred (real inference, GPU-gated): close, never fake.
        let info = fixture_info();
        assert_eq!(handle_line("GEN {\"prompt\":\"hi\"}", &info), None);
        assert_eq!(handle_line("GEN", &info), None);
        // The rest of the generation protocol and unknown verbs close too.
        assert_eq!(handle_line("HREL 1", &info), None);
        assert_eq!(handle_line("VOCAB", &info), None);
        assert_eq!(handle_line("MERGES 2 aGk=", &info), None);
        assert_eq!(handle_line("BOGUS", &info), None);
        assert_eq!(handle_line("", &info), None);
        assert_eq!(handle_line("\n", &info), None);
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
    fn serve_connection_answers_the_line_protocol_over_a_socket() {
        let info = fixture_info();
        let expected_info = info_line(&info);
        let listener = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0))).expect("bind");
        let addr = listener.local_addr().expect("addr");
        thread::spawn(move || serve_forever(&listener, &info));
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
}
