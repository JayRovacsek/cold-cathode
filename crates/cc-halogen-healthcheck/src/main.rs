//! `halogen-healthcheck` — Rust port of the 937-byte bash script shipped by
//! halogen-flash-server v0.18.21 (frozen contract:
//! `docs/analysis/specs/halogen-healthcheck.md`).
//!
//! Modes `auto|engine|api`; the tool is silent on success and on failure
//! (the script suppresses every diagnostic). Exit codes: 0 healthy,
//! 1 unhealthy, 2 usage error, 127 when `api` mode cannot find `python3`
//! (reproduced by shelling out, per the spec's recommendation).

use std::env;
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::process::{self, Command, ExitStatus, Stdio};
use std::time::Duration;

/// Hard-coded in the script (not `$0`), so the port must not use argv[0].
const USAGE: &str = "usage: halogen-healthcheck [auto|engine|api]";
const DEFAULT_ENGINE_PORT: &str = "8730";
const DEFAULT_API_PORT: &str = "8731";
const DEFAULT_WAIT: &str = "30";

fn main() {
    // `${1:-auto}`: an absent or empty argv[1] means `auto`.
    let mode = env::args()
        .nth(1)
        .filter(|arg| !arg.is_empty())
        .unwrap_or_else(|| "auto".to_string());
    let engine_port = env_or("HALOGEN_PORT", DEFAULT_ENGINE_PORT);
    let api_port = env_or("HALOGEN_API_PORT", DEFAULT_API_PORT);
    let wait = env_or("HALOGEN_ENGINE_PING_S", DEFAULT_WAIT);

    let code = match mode.as_str() {
        "engine" => engine_ok(&engine_port, &wait),
        "api" => api_ok(&api_port, &wait),
        "auto" => {
            // The script probes the API port first and only falls back to
            // the engine check when that connect fails.
            if tcp_probe(&api_port) {
                api_ok(&api_port, &wait)
            } else {
                engine_ok(&engine_port, &wait)
            }
        }
        _ => {
            eprintln!("{USAGE}");
            2
        }
    };
    process::exit(code);
}

/// `${VAR:-default}` semantics: unset or empty values fall back to default.
fn env_or(key: &str, default: &str) -> String {
    match env::var(key) {
        Ok(value) if !value.is_empty() => value,
        _ => default.to_string(),
    }
}

/// Resolve `127.0.0.1:<port>`. An unparseable port fails the way bash's
/// `/dev/tcp` redirection does (exit 1); service-name ports are not
/// resolved, a documented divergence from bash's `getservbyname` lookup.
fn socket_addr(port: &str) -> Option<SocketAddr> {
    Some(SocketAddr::from((
        [127, 0, 0, 1],
        port.parse::<u16>().ok()?,
    )))
}

/// Bare TCP connect (no I/O) used by `auto` to pick its path.
fn tcp_probe(port: &str) -> bool {
    socket_addr(port).is_some_and(|addr| TcpStream::connect(addr).is_ok())
}

/// `read -t` accepts non-negative (possibly fractional) seconds; any other
/// value makes the read fail, which the script turns into exit 1.
fn parse_wait(wait: &str) -> Option<Duration> {
    Duration::try_from_secs_f64(wait.parse::<f64>().ok()?).ok()
}

/// Engine probe: connect, write `PING\n`, read one line within `wait`, and
/// require it to equal exactly `PONG`. Silent on every path.
fn engine_ok(port: &str, wait: &str) -> i32 {
    let Some(addr) = socket_addr(port) else {
        return 1;
    };
    let Ok(mut stream) = TcpStream::connect(addr) else {
        return 1;
    };
    if stream.write_all(b"PING\n").is_err() {
        return 1;
    }
    let Some(timeout) = parse_wait(wait) else {
        return 1;
    };
    if stream.set_read_timeout(Some(timeout)).is_err() {
        return 1;
    }
    // `read -r` semantics: consume bytes up to the first newline; EOF
    // without one fails even when bytes were read. A trailing CR stays in
    // the line, so CRLF replies fail the comparison like bash's test does.
    let mut line = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        match stream.read(&mut byte) {
            Ok(0) => return 1,
            Ok(_) => {
                line.push(byte[0]);
                if byte[0] == b'\n' {
                    break;
                }
            }
            Err(_) => return 1, // read timeout or I/O error
        }
    }
    line.pop();
    i32::from(line != b"PONG")
}

/// API probe: shell out to `python3` exactly like the script does (a bare
/// `python3` lookup on PATH), so the 127 command-not-found status on hosts
/// without python3 is reproduced. The script runs python with
/// `2>/dev/null`; stdout is inherited (the inline program never writes it).
fn api_ok(port: &str, wait: &str) -> i32 {
    let script = format!(
        "\nimport sys, urllib.request\ntry:\n    \
         urllib.request.urlopen('http://127.0.0.1:{port}/health', timeout={wait})\n\
         except Exception as e:\n    print(e, file=sys.stderr); sys.exit(1)\n"
    );
    match Command::new("python3")
        .arg("-c")
        .arg(&script)
        .stdout(Stdio::inherit())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(mut child) => match child.wait() {
            Ok(status) => exit_code(status),
            Err(_) => 1,
        },
        // bash command-not-found status; 126 covers unexecutables.
        Err(err) if err.kind() == io::ErrorKind::NotFound => 127,
        Err(_) => 126,
    }
}

/// Shell exit-status convention: the child's code, or 128 + signal when it
/// was killed by a signal.
#[cfg(unix)]
fn exit_code(status: ExitStatus) -> i32 {
    use std::os::unix::process::ExitStatusExt;
    status
        .code()
        .unwrap_or_else(|| 128 + status.signal().unwrap_or(0))
}

#[cfg(not(unix))]
fn exit_code(status: ExitStatus) -> i32 {
    status.code().unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    /// Spawn a fake engine on an ephemeral port that answers the first
    /// request with `reply`, and return its port as the script would see it.
    fn fake_engine(reply: &'static [u8]) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("fake engine binds");
        let port = listener.local_addr().expect("fake engine address").port();
        std::thread::spawn(move || {
            let (mut client, _) = listener.accept().expect("fake engine accepts");
            let mut byte = [0u8; 1];
            let mut request = Vec::new();
            while let Ok(1) = client.read(&mut byte) {
                request.push(byte[0]);
                if byte[0] == b'\n' {
                    break;
                }
            }
            assert_eq!(request, b"PING\n", "fake engine request");
            let _ = client.write_all(reply);
        });
        port.to_string()
    }

    #[test]
    fn usage_line_matches_the_script() {
        assert_eq!(USAGE, "usage: halogen-healthcheck [auto|engine|api]");
    }

    #[test]
    fn env_or_falls_back_on_unset_and_empty() {
        assert_eq!(env_or("CC_NOT_SET_AT_ALL", "8730"), "8730");
        // Unique key: parallel tests must not race on it.
        env::set_var("CC_TEST_EMPTY_PORT", "");
        assert_eq!(env_or("CC_TEST_EMPTY_PORT", "8731"), "8731");
        env::remove_var("CC_TEST_EMPTY_PORT");
    }

    #[test]
    fn parse_wait_accepts_integers_and_rejects_junk() {
        assert_eq!(parse_wait("30"), Some(Duration::from_secs(30)));
        assert_eq!(parse_wait("0.5"), Some(Duration::from_millis(500)));
        assert_eq!(parse_wait("abc"), None);
        assert_eq!(parse_wait("-1"), None);
    }

    #[test]
    fn socket_addr_rejects_non_numeric_ports() {
        assert_eq!(
            socket_addr("8730"),
            Some(SocketAddr::from(([127, 0, 0, 1], 8730)))
        );
        assert_eq!(socket_addr("http"), None);
    }

    #[test]
    fn engine_accepts_exact_pong() {
        assert_eq!(engine_ok(&fake_engine(b"PONG\n"), "5"), 0);
    }

    #[test]
    fn engine_rejects_wrong_reply() {
        assert_eq!(engine_ok(&fake_engine(b"PING\n"), "5"), 1);
    }

    #[test]
    fn engine_rejects_crlf_reply() {
        assert_eq!(engine_ok(&fake_engine(b"PONG\r\n"), "5"), 1);
    }

    #[test]
    fn engine_rejects_closed_port() {
        assert_eq!(engine_ok("1", "1"), 1);
    }

    #[test]
    fn engine_rejects_invalid_wait() {
        assert_eq!(engine_ok(&fake_engine(b"PONG\n"), "abc"), 1);
    }

    #[test]
    fn tcp_probe_mirrors_connectability() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port().to_string();
        assert!(tcp_probe(&port));
        assert!(!tcp_probe("1"));
        assert!(!tcp_probe("not-a-port"));
    }
}
