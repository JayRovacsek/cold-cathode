//! Differential contract tests for `flash_serve` (plan §5, Phase 3).
//!
//! Each test drives one mode group of `tests/contract/matrix/flash_serve.toml`:
//! every entry runs the halogen reference binary and the cold-cathode binary
//! with identical arguments and environment, normalises both outputs per §5
//! and compares exit code, stdout and stderr. Failures are collected and
//! reported together so one run yields the full red/green picture.

use cc_contract_tests::{gating, matrix, normalise, repo_root, runner};
use std::path::PathBuf;

/// The flash_serve matrix file, when present. The nix package source
/// (`lib.sourceByRegex`) excludes the root `tests/` directory, so the
/// contract tests skip gracefully there.
fn matrix_path() -> Option<PathBuf> {
    let path = repo_root()?.join("tests/contract/matrix/flash_serve.toml");
    path.is_file().then_some(path)
}

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

fn run_mode(mode: &str) {
    let Some(path) = matrix_path() else {
        println!(
            "SKIP {mode}: matrix unavailable (nix build src excludes the root tests/ directory)"
        );
        return;
    };
    let cases = match matrix::load_matrix(&path) {
        Ok(cases) => cases,
        Err(err) => panic!("matrix {}: {err}", path.display()),
    };
    let subject = gating::subject_binary();
    let reference = gating::reference_binary();
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

#[test]
fn serve_matrix() {
    run_mode("serve");
}

#[test]
fn dispatch_matrix() {
    run_mode("dispatch");
}

#[test]
fn repack_matrix() {
    run_mode("repack");
}

#[test]
fn resident_gib_matrix() {
    run_mode("resident-gib");
}

#[test]
fn env_matrix() {
    run_mode("env");
}

#[test]
fn matrix_covers_every_mode() {
    let Some(path) = matrix_path() else {
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
