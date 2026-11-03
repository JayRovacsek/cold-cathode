//! `halogen-tools` implementation library (plan §3: `cc-halogen-tools` →
//! `halogen-tools`). The binary target (`src/main.rs`) is a thin wrapper over
//! this crate so the contract-test harness can depend on it and receive
//! `CARGO_BIN_EXE_halogen-tools`.
//!
//! Frozen contract: `docs/analysis/specs/halogen-tools.md`. Every observable
//! string is byte-exact: [RV] facts were runtime-verified against the
//! reference, [D] facts are decompile-derived (Ghidra `main` = `FUN_01ecabf0`
//! and the per-verb handlers in `docs/analysis/evidence/halogen-tools/`).
//!
//! Verb dispatch is an exact, case-sensitive match on `argv[1]`; the help
//! aliases (`--help`, `-h`, `help`) are checked only after the verbs, and
//! `--version` is *not* a version flag (it is an unknown verb).

#![warn(unsafe_op_in_unsafe_fn)]

pub mod hgn;

/// Verbatim help block, printed to stderr on every help path (spec:
/// "Verbatim help text"; byte-identical for no-args, the help aliases and
/// the unknown-verb path [RV]).
const HELP: &str = "halogen-tools: the checkpoint tools\n  inspect FILE.hgn [--json] [--no-hash] [--threads N]\n  verify  FILE.hgn [--threads N]\n  ppl     FILE --ids IDS.bin [--chunk 1024|8] [--out NLL.bin] [--vs OTHER [--vs-out NLL.bin]] [--pin-trunk 0|1]\n          [--ref-out REF.bin [--topk 128]]   write the distribution at every position (top-K, tail, argmax)\n          [--ref REF.bin [--per-pos P.bin]]  score against it: KL(ref || this), top-1 agreement, delta p(target)\n          [--rows-out ROWS.bin --rows LO-HI] the full fp32 logits of positions [LO, HI)\n          [--worst N]                         with --ref: the N positions the two files disagree on most, as ids\n          [--json]                            one object on stdout (the prose goes to stderr)\n          [--seq 262144]                      a longer corpus is scored as consecutive sequences of this length\n  niah    FILE --manifest CASES.tsv [--out DIR] [--max-tok N] [--pin-trunk 0|1]\n  paired  A.bin B.bin [--label-a X] [--label-b Y] [--json]\n";

const USAGE_INSPECT: &str = "usage: inspect FILE.hgn [--json] [--no-hash] [--threads N]";
const USAGE_VERIFY: &str = "usage: verify FILE.hgn [--threads N]";
const USAGE_PPL: &str = "usage: ppl FILE --ids IDS.bin [--chunk 1024|8] [--out NLL.bin] [--vs OTHER [--vs-out NLL.bin]] [--pin-trunk 0|1] [--ref-out REF.bin [--topk K]] [--ref REF.bin [--per-pos P.bin]] [--rows-out ROWS.bin --rows LO-HI]";
const USAGE_NIAH: &str =
    "usage: niah FILE --manifest CASES.tsv [--out DIR] [--max-tok N] [--pin-trunk 0|1]";
const USAGE_PAIRED: &str = "usage: paired A.bin B.bin [--label-a X] [--label-b Y] [--json]";

/// Dispatch on `argv[1]` and return the process exit code. `argv` includes
/// `argv[0]` (the program path as invoked), matching the reference's `argc`/
/// `argv` layout.
pub fn run(argv: &[String]) -> i32 {
    // argc < 2 → help to stderr, exit 2 (spec: "Dispatch rules" [RV]).
    let Some(verb) = argv.get(1) else {
        eprint!("{HELP}");
        return 2;
    };
    match verb.as_str() {
        // Verbs are matched first (exact, case-sensitive), then the help
        // aliases, then everything else is an unknown verb [D order, RV set].
        "inspect" => inspect(argv),
        "verify" => verify(argv),
        "ppl" => ppl(argv),
        "niah" => niah(argv),
        "paired" => paired(argv),
        "--help" | "-h" | "help" => {
            eprint!("{HELP}");
            0
        }
        other => {
            eprintln!("halogen-tools: unknown verb '{other}'");
            eprint!("{HELP}");
            2
        }
    }
}

/// Print a per-verb usage line to stderr (usage strings embed the verb name,
/// never `argv[0]`, so no normalisation is needed).
fn usage(text: &str) -> i32 {
    eprintln!("{text}");
    2
}

/// `inspect FILE.hgn [--json] [--no-hash] [--threads N]`.
fn inspect(argv: &[String]) -> i32 {
    // FILE is the next argv word verbatim, even if it starts with `-`
    // (`inspect --json` → `cannot open --json`) [RV].
    let Some(file) = argv.get(2) else {
        return usage(USAGE_INSPECT);
    };
    let json = has_flag(argv, 2, "--json");
    let _no_hash = has_flag(argv, 2, "--no-hash");
    let _threads = parse_threads(argv);
    match hgn::load(file) {
        Err(err) => {
            eprintln!("{err}");
            2
        }
        Ok(ck) => {
            let body = if json {
                hgn::inspect_json(file, &ck)
            } else {
                hgn::inspect_prose(&ck)
            };
            print!("{body}");
            0
        }
    }
}

/// `verify FILE.hgn [--threads N]`.
fn verify(argv: &[String]) -> i32 {
    let Some(file) = argv.get(2) else {
        return usage(USAGE_VERIFY);
    };
    let _threads = parse_threads(argv);
    match hgn::load(file) {
        // The only verb whose failure output goes to stdout, exit 1 [RV].
        Err(err) => {
            println!("FAIL {err}");
            1
        }
        Ok(ck) => {
            println!("{}", hgn::verify_pass(file, &ck));
            0
        }
    }
}

/// `ppl FILE --ids IDS.bin …` — argument validation; scoring is GPU-gated.
fn ppl(argv: &[String]) -> i32 {
    let Some(file) = argv.get(2) else {
        return usage(USAGE_PPL);
    };
    // `main` rejects a dash-leading FILE with the one-line usage [D].
    if file.starts_with('-') {
        return usage(USAGE_PPL);
    }
    // `--ids` is required (no text front end) [RV]; its value is the word
    // after the flag, scanned from argv[3] on [D].
    let Some(ids) = find_value(argv, 3, "--ids") else {
        eprintln!("ppl: --ids IDS.bin is required (a text front end is not built yet)");
        return 2;
    };
    // File-open checks precede flag-value checks [RV].
    let Ok(ids_bytes) = std::fs::read(ids) else {
        eprintln!("cannot open {ids}");
        return 2;
    };
    let chunk = value(argv, "--chunk").map_or(1024, |s| parse_int(s));
    let seq = value(argv, "--seq").map_or(262144, |s| parse_int(s));
    // Flag-value validation (all [D], spec "Flag table — ppl").
    if ids_bytes.len() / 4 < 2 || chunk < 1 {
        eprintln!("need >= 2 ids and chunk >= 1");
        return 2;
    }
    if chunk > seq {
        eprintln!("ppl: --chunk {chunk} is larger than the sequence (--seq {seq})");
        return 2;
    }
    if !(2..=1048576).contains(&seq) {
        eprintln!("ppl: --seq in [2, 1048576]");
        return 2;
    }
    if value(argv, "--ref-out").is_some() && value(argv, "--ref").is_some() {
        eprintln!("ppl: --ref-out and --ref are one or the other");
        return 2;
    }
    if let Some(topk) = value(argv, "--topk") {
        let topk = parse_int(topk);
        if !(1..=4096).contains(&topk) {
            eprintln!("ppl: --topk in [1, 4096]");
            return 2;
        }
    }
    if value(argv, "--rows-out").is_some() && value(argv, "--rows").is_none() {
        eprintln!("ppl: --rows-out needs --rows LO-HI");
        return 2;
    }
    if value(argv, "--worst").is_some() && value(argv, "--ref").is_none() {
        eprintln!("ppl: --worst needs --ref");
        return 2;
    }
    // Checkpoint load, then engine construction (GPU-gated, spec: "the
    // actual scoring is GPU-gated"; the decompiled runtime message is
    // `ppl: engine %s`).
    match hgn::load(file) {
        Err(err) => {
            eprintln!("{err}");
            2
        }
        Ok(_) => {
            eprintln!("ppl: engine {file}");
            1
        }
    }
}

/// `niah FILE --manifest CASES.tsv …` — argument validation; execution is
/// GPU-gated.
fn niah(argv: &[String]) -> i32 {
    let Some(file) = argv.get(2) else {
        return usage(USAGE_NIAH);
    };
    // `main` rejects a dash-leading FILE with the one-line usage [D].
    if file.starts_with('-') {
        return usage(USAGE_NIAH);
    }
    let Some(manifest) = find_value(argv, 3, "--manifest") else {
        eprintln!(
            "niah: --manifest CASES.tsv is required (the battery builder is not in the image yet)"
        );
        return 2;
    };
    if std::fs::read(manifest).is_err() {
        eprintln!("cannot open {manifest}");
        return 2;
    }
    match hgn::load(file) {
        Err(err) => {
            eprintln!("{err}");
            2
        }
        Ok(_) => {
            eprintln!("niah: engine {file}");
            1
        }
    }
}

/// `paired A.bin B.bin [--label-a X] [--label-b Y] [--json]`.
fn paired(argv: &[String]) -> i32 {
    // Non-flag argv words are collected in order; `--label-a`/`--label-b`
    // consume the following word as their value (defaults `A`/`B` [D]),
    // `--json` is a flag and unknown `--` words are ignored [D].
    let mut positionals: Vec<&String> = Vec::new();
    let mut json = false;
    let mut label_a = "A".to_string();
    let mut label_b = "B".to_string();
    let mut index = 2;
    while index < argv.len() {
        let word = &argv[index];
        if word == "--json" {
            json = true;
            index += 1;
        } else if word == "--label-a" || word == "--label-b" {
            let label = if word == "--label-a" {
                &mut label_a
            } else {
                &mut label_b
            };
            if let Some(next) = argv.get(index + 1) {
                *label = next.clone();
            }
            index += 2;
        } else if word.starts_with("--") {
            index += 1;
        } else {
            positionals.push(word);
            index += 1;
        }
    }
    let (Some(a), Some(b)) = (positionals.first(), positionals.get(1)) else {
        return usage(USAGE_PAIRED);
    };
    // Unreadable A.bin → `cannot open A.bin` (stderr, exit 2) [RV]; differs
    // from `verify`, which prints `FAIL …` on stdout and exits 1.
    let Ok(a_bytes) = std::fs::read(a) else {
        eprintln!("cannot open {a}");
        return 2;
    };
    let Ok(b_bytes) = std::fs::read(b) else {
        eprintln!("cannot open {b}");
        return 2;
    };
    print!(
        "{}",
        paired_report(a, b, &label_a, &label_b, &a_bytes, &b_bytes, json)
    );
    0
}

/// Paired NLL comparison output (decompile-derived prose
/// `paired %s-%s: mean %+.5f nats  t = %+.2f  95%% CI [%+.5f, %+.5f]  PPL
/// ratio %.5f  %s better on %zu / worse on %zu / tied %zu` and the JSON
/// object; the `.bin` score-file layout is [D], so the matrix defers the
/// success path).
fn paired_report(
    a: &str,
    b: &str,
    label_a: &str,
    label_b: &str,
    a_bytes: &[u8],
    b_bytes: &[u8],
    json: bool,
) -> String {
    let scores = |bytes: &[u8]| -> Vec<f64> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk) as f64)
            .collect()
    };
    let (xs, ys) = (scores(a_bytes), scores(b_bytes));
    let n = xs.len().min(ys.len());
    if n == 0 {
        return "paired: no positions\n".to_string();
    }
    let diffs: Vec<f64> = (0..n).map(|i| ys[i] - xs[i]).collect();
    let mean = diffs.iter().sum::<f64>() / n as f64;
    let var = if n > 1 {
        diffs.iter().map(|d| (d - mean) * (d - mean)).sum::<f64>() / (n - 1) as f64
    } else {
        0.0
    };
    let se = (var / n as f64).sqrt();
    let t = if se > 0.0 { mean / se } else { 0.0 };
    // A lower NLL is better: `better` counts positions where B beats A.
    let (better, worse, tied) =
        diffs
            .iter()
            .fold((0usize, 0usize, 0usize), |acc, d| match d.total_cmp(&0.0) {
                std::cmp::Ordering::Less => (acc.0 + 1, acc.1, acc.2),
                std::cmp::Ordering::Greater => (acc.0, acc.1 + 1, acc.2),
                std::cmp::Ordering::Equal => (acc.0, acc.1, acc.2 + 1),
            });
    let winner = if better >= worse { label_b } else { label_a };
    let ppl_ratio = if xs.is_empty() {
        0.0
    } else {
        ys.iter().sum::<f64>() / xs.iter().sum::<f64>()
    };
    if json {
        format!(
            "{{\"tool\":\"halogen-tools paired\",\"warning\":null,\"a\":\"{a}\",\"b\":\"{b}\",\
             \"paired\":{{\"n\":{n},\"mean_diff\":\"{mean:.5}\",\"se\":\"{se:.5}\",\
             \"t\":\"{t:.2}\",\"ci95\":[\"{:.5}\",\"{:.5}\"],\"ppl_ratio\":\"{ppl_ratio:.5}\",\
             \"better\":{better},\"worse\":{worse},\"tied\":{tied}}}}}\n",
            mean - 1.96 * se,
            mean + 1.96 * se
        )
    } else {
        format!(
            "paired {label_a}-{label_b}: mean {mean:+.5} nats  t = {t:+.2}  \
             95% CI [{:+.5}, {:+.5}]  PPL ratio {ppl_ratio:.5}  {winner} better on \
             {better} / worse on {worse} / tied {tied}\n",
            mean - 1.96 * se,
            mean + 1.96 * se
        )
    }
}

/// Exact-match flag scan from `from` on: does any argv word equal `flag`?
/// `--flag=value` is a different word and never matches [D].
fn has_flag(argv: &[String], from: usize, flag: &str) -> bool {
    argv.iter().skip(from).any(|word| word == flag)
}

/// Exact-match flag scan from `from`: the value of `flag` is the word
/// directly after it; a flag without a following word has no value.
fn find_value<'a>(argv: &'a [String], from: usize, flag: &str) -> Option<&'a String> {
    (from..argv.len().saturating_sub(1))
        .find(|i| argv[*i] == flag)
        .map(|i| &argv[i + 1])
}

/// Any-word lookup (used for the ppl/niah flags whose value position is
/// fixed by the grammar).
fn value<'a>(argv: &'a [String], flag: &str) -> Option<&'a String> {
    find_value(argv, 2, flag)
}

/// `--threads N` parser (`FUN_01ef06d0` [D]): scan the words after FILE for
/// an exact `--threads`; the value is the following word, parsed with
/// `strtol` base 10 and clamped to a minimum of 1. The default is 8 and the
/// parser never errors.
fn parse_threads(argv: &[String]) -> i64 {
    if argv.len() < 3 {
        return 8;
    }
    for index in 2..argv.len() - 1 {
        if argv[index] == "--threads" {
            let parsed = parse_int(&argv[index + 1]);
            return if parsed > 1 { parsed } else { 1 };
        }
    }
    8
}

/// `strtol` base-10 semantics for the observable inputs: optional sign plus
/// decimal digits; a non-numeric word yields 0, overflow saturates.
fn parse_int(text: &str) -> i64 {
    let body = text.trim_start();
    let (negative, digits) = match body.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, body.strip_prefix('+').unwrap_or(body)),
    };
    let number: String = digits.chars().take_while(char::is_ascii_digit).collect();
    if number.is_empty() {
        return 0;
    }
    let value = number.parse::<i64>().unwrap_or(i64::MAX);
    if negative {
        -value
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(args: &[&str]) -> Vec<String> {
        std::iter::once("halogen-tools".to_string())
            .chain(args.iter().map(|a| a.to_string()))
            .collect()
    }

    #[test]
    fn help_block_matches_the_reference_bytes() {
        // fwrite lengths from the decompiled `main`: the help block is 0x3ea
        // (1002) bytes including the trailing newline; each usage line's
        // length matches the reference's fwrite size argument.
        assert_eq!(HELP.len(), 0x3ea);
        assert_eq!(USAGE_INSPECT.len() + 1, 0x3b);
        assert_eq!(USAGE_VERIFY.len() + 1, 0x25);
        assert_eq!(USAGE_PPL.len() + 1, 0xd5);
        assert_eq!(USAGE_NIAH.len() + 1, 0x52);
        assert_eq!(USAGE_PAIRED.len() + 1, 0x3f);
    }

    #[test]
    fn no_args_prints_help_with_exit_2() {
        assert_eq!(run(&argv(&[])), 2);
    }

    #[test]
    fn help_aliases_exit_0() {
        for alias in ["--help", "-h", "help"] {
            assert_eq!(run(&argv(&[alias])), 0, "{alias}");
        }
    }

    #[test]
    fn unknown_verbs_exit_2_including_version() {
        // `--version` is not a version flag; `-V` and `INSPECT` are [D].
        for verb in ["--version", "--bogus", "--Help", "HELP", "-V", "INSPECT"] {
            assert_eq!(run(&argv(&[verb])), 2, "{verb}");
        }
    }

    #[test]
    fn missing_file_usage_exits_2_for_every_verb() {
        for verb in ["inspect", "verify", "ppl", "niah", "paired"] {
            assert_eq!(run(&argv(&[verb])), 2, "{verb}");
        }
    }

    #[test]
    fn threads_parser_clamps_and_defaults() {
        assert_eq!(parse_threads(&argv(&["inspect", "x.hgn"])), 8);
        assert_eq!(
            parse_threads(&argv(&["inspect", "x.hgn", "--threads", "4"])),
            4
        );
        assert_eq!(
            parse_threads(&argv(&["inspect", "x.hgn", "--threads", "0"])),
            1
        );
        assert_eq!(
            parse_threads(&argv(&["inspect", "x.hgn", "--threads", "-3"])),
            1
        );
        assert_eq!(
            parse_threads(&argv(&["inspect", "x.hgn", "--threads", "x"])),
            1
        );
        // `--threads=4` is not the `--threads` flag (exact word match).
        assert_eq!(
            parse_threads(&argv(&["inspect", "x.hgn", "--threads=4"])),
            8
        );
        // A dangling `--threads` (no value word) falls back to the default.
        assert_eq!(parse_threads(&argv(&["inspect", "x.hgn", "--threads"])), 8);
    }

    #[test]
    fn parse_int_follows_strtol_prefix_semantics() {
        assert_eq!(parse_int("42"), 42);
        assert_eq!(parse_int("4x"), 4);
        assert_eq!(parse_int("x"), 0);
        assert_eq!(parse_int("-7"), -7);
        assert_eq!(parse_int("+9"), 9);
        assert_eq!(parse_int("99999999999999999999"), i64::MAX);
    }

    #[test]
    fn inspect_reports_cannot_open_for_unreadable_files() {
        // `inspect --json` takes `--json` verbatim as FILE [RV]; the loader
        // error wording is asserted directly (stderr is not capturable
        // in-process without a redirect crate).
        assert_eq!(
            hgn::load("--json").unwrap_err(),
            "cannot open --json".to_string()
        );
        assert_eq!(
            hgn::load("/nonexistent.hgn").unwrap_err(),
            "cannot open /nonexistent.hgn".to_string()
        );
    }
}
