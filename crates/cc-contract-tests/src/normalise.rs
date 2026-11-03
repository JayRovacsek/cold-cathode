//! Normalisation rules from plan §5 (locked): before comparing reference and
//! subject output, argv[0]/program name, timestamps, PIDs, bound port numbers
//! and Nix store paths are replaced with placeholders. Anything not covered
//! by a rule is compared byte-for-byte.
//!
//! The scanners work on bytes (all placeholders are ASCII) so they never
//! panic on multi-byte UTF-8 input.

/// Normalise one captured output stream. `argv0` is the exact path the binary
/// was invoked with (the usage lines embed it verbatim).
pub fn normalise(text: &str, argv0: &str) -> String {
    let mut text = if argv0.is_empty() {
        text.to_string()
    } else {
        text.replace(argv0, "<argv0>")
    };
    text = replace_nix_store_paths(&text);
    text = replace_timestamps(&text);
    text = replace_keyword_number(&text, "pid", "<pid>");
    text = replace_keyword_number(&text, "port", "<port>");
    text = replace_durations(&text);
    text
}

/// Replace every `/nix/store/<hash>-<name>` path with `<nix-store>`.
fn replace_nix_store_paths(text: &str) -> String {
    const PREFIX: &str = "/nix/store/";
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(pos) = rest.find(PREFIX) {
        out.push_str(&rest[..pos]);
        let after = &rest[pos + PREFIX.len()..];
        let hash_len = after
            .bytes()
            .take_while(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
            .count();
        if hash_len > 0 && after.as_bytes().get(hash_len) == Some(&b'-') {
            let name = &after[hash_len + 1..];
            let name_len = name
                .bytes()
                .take_while(|b| {
                    matches!(b, b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'.' | b'_' | b'+' | b'-')
                })
                .count();
            out.push_str("<nix-store>");
            rest = &after[hash_len + 1 + name_len..];
        } else {
            out.push_str(PREFIX);
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

/// Replace the number following a keyword (`pid 4242`, `port: 8730`).
fn replace_keyword_number(text: &str, keyword: &str, placeholder: &str) -> String {
    let bytes = text.as_bytes();
    let needle = keyword.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let matches = bytes.len() - i >= needle.len()
            && bytes[i..i + needle.len()]
                .iter()
                .zip(needle)
                .all(|(a, b)| a.eq_ignore_ascii_case(b));
        if !matches {
            out.push(bytes[i]);
            i += 1;
            continue;
        }
        // Require a separator (`:` or whitespace) before the digits so words
        // like `ports` or `--port P` are left alone.
        let mut j = i + needle.len();
        let mut separator = false;
        if bytes.get(j) == Some(&b':') {
            separator = true;
            j += 1;
        }
        while matches!(bytes.get(j), Some(b' ') | Some(b'\t')) {
            separator = true;
            j += 1;
        }
        let digits_start = j;
        while matches!(bytes.get(j), Some(c) if c.is_ascii_digit()) {
            j += 1;
        }
        if separator && j > digits_start {
            out.extend_from_slice(placeholder.as_bytes());
            i = j;
        } else {
            out.extend_from_slice(needle);
            i += needle.len();
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Replace ISO-8601 style timestamps (`2026-11-03`, optionally with a
/// `HH:MM:SS` time part and `Z`/offset) with `<timestamp>`.
fn replace_timestamps(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if is_date_at(bytes, i) {
            let mut end = i + 10; // YYYY-MM-DD
            if matches!(bytes.get(end), Some(b'T') | Some(b' ')) && is_time_at(bytes, end + 1) {
                end += 9; // HH:MM:SS
                if matches!(bytes.get(end), Some(b'Z')) {
                    end += 1;
                } else if matches!(bytes.get(end), Some(b'+') | Some(b'-'))
                    && is_time_at(bytes, end + 1)
                {
                    end += 6; // ±HH:MM
                }
            }
            out.extend_from_slice(b"<timestamp>");
            i = end;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn is_date_at(bytes: &[u8], i: usize) -> bool {
    let digit = |j: usize| bytes.get(j).is_some_and(u8::is_ascii_digit);
    // Not part of a longer digit run (that is handled by the duration rule).
    let standalone = i == 0 || !bytes[i - 1].is_ascii_digit();
    standalone
        && digit(i)
        && digit(i + 1)
        && digit(i + 2)
        && digit(i + 3)
        && bytes.get(i + 4) == Some(&b'-')
        && digit(i + 5)
        && digit(i + 6)
        && bytes.get(i + 7) == Some(&b'-')
        && digit(i + 8)
        && digit(i + 9)
        && !digit(i + 10)
}

fn is_time_at(bytes: &[u8], j: usize) -> bool {
    let digit = |k: usize| bytes.get(k).is_some_and(u8::is_ascii_digit);
    digit(j)
        && digit(j + 1)
        && bytes.get(j + 2) == Some(&b':')
        && digit(j + 3)
        && digit(j + 4)
        && bytes.get(j + 5) == Some(&b':')
        && digit(j + 6)
        && digit(j + 7)
}

/// Replace durations (`1.5ms`, `42s`, `3us`, `7ns`) with `<duration>`.
fn replace_durations(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if !bytes[i].is_ascii_digit() || (i > 0 && bytes[i - 1].is_ascii_digit()) {
            out.push(bytes[i]);
            i += 1;
            continue;
        }
        let mut j = i;
        while matches!(bytes.get(j), Some(c) if c.is_ascii_digit()) {
            j += 1;
        }
        let mut end = j;
        if bytes.get(j) == Some(&b'.') && matches!(bytes.get(j + 1), Some(c) if c.is_ascii_digit())
        {
            end += 2;
            while matches!(bytes.get(end), Some(c) if c.is_ascii_digit()) {
                end += 1;
            }
        }
        let unit_len = match (bytes.get(end), bytes.get(end + 1)) {
            (Some(b'n'), Some(b's')) | (Some(b'u'), Some(b's')) | (Some(b'm'), Some(b's')) => 2,
            (Some(b's'), _) => 1,
            _ => 0,
        };
        if unit_len == 0 {
            out.extend_from_slice(&bytes[i..j]);
            i = j;
        } else {
            out.extend_from_slice(b"<duration>");
            i = end + unit_len;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replaces_argv0() {
        let out = normalise(
            "usage: /nix/store/abc…-pkg/bin/flash_serve [--ck FILE]",
            "/nix/store/abc…-pkg/bin/flash_serve",
        );
        assert_eq!(out, "usage: <argv0> [--ck FILE]");
    }

    #[test]
    fn replaces_nix_store_paths() {
        let out = normalise(
            "opened /nix/store/fn6y2w20kbgjd8da8pwxbbc3gvmiycpy-halogen-flash-server-0.18.21/bin/flash_serve once",
            "",
        );
        assert_eq!(out, "opened <nix-store>/bin/flash_serve once");
    }

    #[test]
    fn replaces_pids_and_ports() {
        assert_eq!(normalise("worker pid 4242 gone", ""), "worker <pid> gone");
        assert_eq!(
            normalise("listening on 127.0.0.1 port: 8730", ""),
            "listening on 127.0.0.1 <port>"
        );
        // `--port P` has no number after it and must stay untouched.
        assert_eq!(normalise("[--port P]", ""), "[--port P]");
    }

    #[test]
    fn replaces_timestamps() {
        assert_eq!(
            normalise("probe transcript — 2026-11-03T04:14:17Z UTC", ""),
            "probe transcript — <timestamp> UTC"
        );
        assert_eq!(normalise("date 2026-11-03 ok", ""), "date <timestamp> ok");
    }

    #[test]
    fn replaces_durations() {
        assert_eq!(normalise("took 1.5ms done", ""), "took <duration> done");
        assert_eq!(normalise("in 42s", ""), "in <duration>");
        assert_eq!(normalise("1..64 slots", ""), "1..64 slots");
    }

    #[test]
    fn leaves_plain_usage_untouched() {
        let line = "usage: <argv0> [--ck FILE] [--port P] [--bind ADDR] [--slots B] [--ctx N] [--max-tok N] [--kv-pool POSITIONS]";
        assert_eq!(normalise(line, ""), line);
    }
}
