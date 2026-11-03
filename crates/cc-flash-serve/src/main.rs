//! `flash_serve` binary: thin wrapper over `cc_flash_serve::run` so the
//! contract-test harness can depend on the library and still exercise the
//! real binary via `CARGO_BIN_EXE_flash_serve`.

use std::process::ExitCode;

fn main() -> ExitCode {
    let mut argv = std::env::args_os();
    let argv0 = argv
        .next()
        .map(|arg| arg.to_string_lossy().into_owned())
        .unwrap_or_default();
    let args: Vec<String> = argv.map(|arg| arg.to_string_lossy().into_owned()).collect();
    ExitCode::from(u8::try_from(cc_flash_serve::run(&argv0, &args)).unwrap_or(1))
}
