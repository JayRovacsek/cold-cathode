//! `cold-cathode` binary: thin wrapper over `cc_launcher::run` so the
//! contract-test harness can depend on the library and still exercise the
//! real binary via `CARGO_BIN_EXE_cold-cathode`.

use std::process::ExitCode;

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();
    ExitCode::from(u8::try_from(cc_launcher::run(&argv)).unwrap_or(1))
}
