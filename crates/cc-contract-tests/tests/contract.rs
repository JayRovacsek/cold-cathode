//! Differential contract tests for every cold-cathode binary (plan §5, Phase 3).
//!
//! The harness discovers all `*.toml` files in `tests/contract/matrix/`;
//! each filename stem names the subject binary (`flash_serve.toml` →
//! `CARGO_BIN_EXE_flash_serve`, `halogen-tools.toml` →
//! `CARGO_BIN_EXE_halogen-tools`, see `cc_contract_tests::discovery`) and
//! the halogen reference binary (`HALOGEN_<NAME>` env override, then a
//! `PATH` lookup, see `cc_contract_tests::gating`). Every entry runs the
//! halogen reference binary and the cold-cathode binary with identical
//! arguments and environment, normalises both outputs per §5 and compares
//! exit code, stdout and stderr. Failures are collected and reported
//! together so one run yields the full red/green picture.
//!
//! The nix package source (`lib.sourceByRegex`) excludes the root `tests/`
//! directory, so discovery finds nothing there and every test below skips
//! gracefully.

use cc_contract_tests::{discovery, gating, matrix, normalise, repo_root, runner};

/// The workspace binaries with a dedicated test below. A matrix file named
/// after anything else fails `every_matrix_file_names_a_known_binary`.
const KNOWN_STEMS: &[&str] = &[
    "cold-cathode",
    "flash_serve",
    "halogen-healthcheck",
    "halogen-npu",
    "halogen-tools",
];

#[derive(Default)]
struct Tally {
    pass: usize,
    fail: usize,
    skip: usize,
    deferred: usize,
    failures: Vec<String>,
    notes: Vec<String>,
}

/// Resolve `@fixture/<name>` argument tokens to the committed fixture path
/// and `@tmp/<name>` tokens to the project-local `.tmp/` scratch path
/// (AGENTS.md: no system temp dirs).
fn resolve_args(args: &[String]) -> Result<Vec<String>, String> {
    let mut resolved = Vec::with_capacity(args.len());
    for arg in args {
        if let Some(fixture) = arg.strip_prefix("@fixture/") {
            let path = repo_root()
                .map(|root| root.join("crates/cc-contract-tests/fixtures").join(fixture))
                .ok_or_else(|| "repo root unavailable".to_string())?;
            if !path.is_file() {
                return Err(format!("fixture {fixture} missing"));
            }
            resolved.push(path.display().to_string());
        } else if let Some(name) = arg.strip_prefix("@tmp/") {
            let root = repo_root().ok_or_else(|| "repo root unavailable".to_string())?;
            let dir = root.join(".tmp");
            std::fs::create_dir_all(&dir).map_err(|err| format!("cannot create .tmp: {err}"))?;
            resolved.push(dir.join(name).display().to_string());
        } else {
            resolved.push(arg.clone());
        }
    }
    Ok(resolved)
}

/// Execute every case of `cases` belonging to `mode` against the reference
/// and subject binaries resolved for `stem`.
fn collect(stem: &str, mode: &str, cases: &[matrix::Case]) -> Tally {
    let subject = gating::subject_binary(stem);
    let reference = gating::reference_binary(stem);
    let mut tally = Tally::default();

    for case in cases.iter().filter(|case| case.mode == mode) {
        if case.deferred {
            tally.deferred += 1;
            tally
                .notes
                .push(format!("DEFER {}: {}", case.name, case.reason));
            continue;
        }
        if case.requires_gpu && !gating::gpu_present() {
            tally.skip += 1;
            tally.notes.push(format!(
                "SKIP {} (CLI-only mode: no /dev/kfd or /dev/dri)",
                case.name
            ));
            continue;
        }
        let args = match resolve_args(&case.args) {
            Ok(args) => args,
            Err(reason) => {
                tally.skip += 1;
                tally.notes.push(format!("SKIP {}: {reason}", case.name));
                continue;
            }
        };
        let (Some(reference), Some(subject)) = (reference.clone(), subject.clone()) else {
            tally.skip += 1;
            tally.notes.push(format!(
                "SKIP {}: reference or subject binary unavailable",
                case.name
            ));
            continue;
        };
        let reference_exec = runner::run(&reference, &args, &case.env).unwrap_or_else(|err| {
            panic!("{}: reference {}: {err}", case.name, reference.display())
        });
        let subject_exec = runner::run(&subject, &args, &case.env)
            .unwrap_or_else(|err| panic!("{}: subject {}: {err}", case.name, subject.display()));

        let reference_norm =
            normalise::normalise(&reference_exec.stderr, &reference.display().to_string());
        let subject_norm =
            normalise::normalise(&subject_exec.stderr, &subject.display().to_string());
        let reference_out =
            normalise::normalise(&reference_exec.stdout, &reference.display().to_string());
        let subject_out =
            normalise::normalise(&subject_exec.stdout, &subject.display().to_string());

        if reference_exec.code == subject_exec.code
            && reference_norm == subject_norm
            && reference_out == subject_out
        {
            tally.pass += 1;
            continue;
        }
        tally.fail += 1;
        tally.failures.push(format!(
            "FAIL {} [spec: {}]\n  exit: reference {:?}, subject {:?}\n  stdout reference: {:?}\n  stdout subject:   {:?}\n  stderr reference: {:?}\n  stderr subject:   {:?}",
            case.name,
            case.spec,
            reference_exec.code,
            subject_exec.code,
            truncate(&reference_out),
            truncate(&subject_out),
            truncate(&reference_norm),
            truncate(&subject_norm),
        ));
    }
    tally
}

/// Report one mode tally: print the summary (plus notes) or panic with the
/// collected failures.
fn report(mode: &str, tally: Tally) {
    let summary = format!(
        "mode {mode}: {} pass, {} fail, {} skip, {} deferred",
        tally.pass, tally.fail, tally.skip, tally.deferred
    );
    if tally.fail > 0 {
        panic!("{summary}\n{}", tally.failures.join("\n"));
    }
    println!("{summary}");
    for note in &tally.notes {
        println!("{note}");
    }
}

/// Run one mode group of the matrix for `stem`, skipping when the matrix is
/// unavailable (nix build source).
fn run_mode(stem: &str, mode: &str) {
    let Some(path) = discovery::matrix_for(stem) else {
        println!(
            "SKIP {mode}: matrix unavailable (nix build src excludes the root tests/ directory)"
        );
        return;
    };
    let cases = match matrix::load_matrix(&path) {
        Ok(cases) => cases,
        Err(err) => panic!("matrix {}: {err}", path.display()),
    };
    report(mode, collect(stem, mode, &cases));
}

/// Run every mode present in the matrix for `stem` (for binaries whose mode
/// set is not enumerated by dedicated tests the way flash_serve's is).
fn run_all_modes(stem: &str) {
    let Some(path) = discovery::matrix_for(stem) else {
        println!(
            "SKIP {stem}: matrix unavailable (nix build src excludes the root tests/ directory)"
        );
        return;
    };
    let cases = match matrix::load_matrix(&path) {
        Ok(cases) => cases,
        Err(err) => panic!("matrix {}: {err}", path.display()),
    };
    let mut modes: Vec<&str> = Vec::new();
    for case in &cases {
        if !modes.contains(&case.mode.as_str()) {
            modes.push(&case.mode);
        }
    }
    assert!(!modes.is_empty(), "matrix {} has no cases", path.display());
    for mode in modes {
        report(mode, collect(stem, mode, &cases));
    }
}

fn truncate(text: &str) -> String {
    const LIMIT: usize = 240;
    if text.len() <= LIMIT {
        return text.to_string();
    }
    let mut end = LIMIT;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &text[..end])
}

// flash_serve — per-mode tests from the original harness, kept as-is.
#[test]
fn serve_matrix() {
    run_mode("flash_serve", "serve");
}

#[test]
fn dispatch_matrix() {
    run_mode("flash_serve", "dispatch");
}

#[test]
fn repack_matrix() {
    run_mode("flash_serve", "repack");
}

#[test]
fn resident_gib_matrix() {
    run_mode("flash_serve", "resident-gib");
}

#[test]
fn env_matrix() {
    run_mode("flash_serve", "env");
}

// Remaining workspace binaries — one test each, running every mode present.
#[test]
fn halogen_tools_matrix() {
    run_all_modes("halogen-tools");
}

#[test]
fn halogen_npu_matrix() {
    run_all_modes("halogen-npu");
}

#[test]
fn halogen_healthcheck_matrix() {
    run_all_modes("halogen-healthcheck");
}

#[test]
fn cold_cathode_matrix() {
    run_all_modes("cold-cathode");
}

#[test]
fn matrix_covers_every_mode() {
    let Some(path) = discovery::matrix_for("flash_serve") else {
        println!("SKIP: matrix unavailable");
        return;
    };
    let cases = matrix::load_matrix(&path).expect("matrix parses");
    for mode in ["serve", "dispatch", "repack", "resident-gib", "env"] {
        assert!(
            cases.iter().any(|case| case.mode == mode),
            "matrix has no {mode} entries"
        );
    }
}

#[test]
fn every_matrix_file_names_a_known_binary() {
    let Some(root) = repo_root() else {
        println!("SKIP: repo root unavailable");
        return;
    };
    let dir = root.join("tests/contract/matrix");
    if !dir.is_dir() {
        println!(
            "SKIP: matrix directory unavailable (nix build src excludes the root tests/ directory)"
        );
        return;
    }
    let unknown: Vec<String> = discovery::matrices()
        .into_iter()
        .map(|(stem, _)| stem)
        .filter(|stem| !KNOWN_STEMS.contains(&stem.as_str()))
        .collect();
    assert!(
        unknown.is_empty(),
        "matrix files without a dedicated test or dev-dep: {unknown:?}"
    );
}
