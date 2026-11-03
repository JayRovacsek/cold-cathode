//! Execute one binary with fixed arguments and environment, capturing the
//! three comparable channels (exit code, stdout, stderr).

use std::io;
use std::path::Path;
use std::process::Command;

/// Captured result of one execution.
#[derive(Debug, Clone)]
pub struct Exec {
    /// Exit code; `None` when the process was killed by a signal.
    pub code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
}

/// Run `bin` with `args` and `env`. The environment is cleared first (plus
/// `PATH`) so results never depend on ambient `HALOGEN_*` variables.
pub fn run(bin: &Path, args: &[String], env: &[(String, String)]) -> io::Result<Exec> {
    let mut command = Command::new(bin);
    command.args(args).env_clear();
    if let Ok(path) = std::env::var("PATH") {
        command.env("PATH", path);
    }
    for (key, value) in env {
        command.env(key, value);
    }
    let output = command.output()?;
    Ok(Exec {
        code: output.status.code(),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    })
}
