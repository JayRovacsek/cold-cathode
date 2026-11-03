//! Binary resolution and GPU gating (plan §5).
//!
//! Every matrix under `tests/contract/matrix/` is named after the binary it
//! exercises (`flash_serve.toml`, `halogen-tools.toml`, `cold-cathode.toml`,
//! ...). The same name resolves both sides of a differential comparison:
//!
//! * subject: `CARGO_BIN_EXE_<name>` — set by cargo for every binary of the
//!   packages `cc-contract-tests` dev-depends on — with a
//!   `target/{debug,release}/<name>` walk-up as fallback.
//! * reference: `HALOGEN_<NAME>`, the name uppercased with `-` mapped to `_`
//!   (`flash_serve` → `HALOGEN_FLASH_SERVE`, `halogen-tools` →
//!   `HALOGEN_HALOGEN_TOOLS`, `cold-cathode` → `HALOGEN_COLD_CATHODE`),
//!   with a `<name>` lookup on `PATH` (the dev shell provides the nix-built
//!   halogen package there — never a mutable install, per plan §5).

use std::env;
use std::path::{Path, PathBuf};

/// GPU probe: a ROCm machine exposes `/dev/kfd` and/or `/dev/dri`. Without
/// either, server tests skip and the run reports "CLI-only mode".
pub fn gpu_present() -> bool {
    Path::new("/dev/kfd").exists() || Path::new("/dev/dri").exists()
}

/// The environment variable that overrides the halogen reference binary for
/// `binary`: `HALOGEN_` plus the name uppercased with `-` replaced by `_`
/// (`flash_serve` → `HALOGEN_FLASH_SERVE`).
pub fn reference_env_var(binary: &str) -> String {
    format!("HALOGEN_{}", binary.to_uppercase().replace('-', "_"))
}

/// The halogen reference binary for `binary` (e.g. `flash_serve`,
/// `halogen-tools`): the `HALOGEN_<NAME>` env var wins, otherwise a `binary`
/// lookup on `PATH` (the dev shell provides the nix-built halogen package
/// there — never a mutable install, per plan §5).
pub fn reference_binary(binary: &str) -> Option<PathBuf> {
    if let Ok(path) = env::var(reference_env_var(binary)) {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }
    env::var("PATH")
        .ok()?
        .split(':')
        .map(|dir| PathBuf::from(dir).join(binary))
        .find(|candidate| candidate.is_file())
}

/// The cold-cathode subject binary for `binary` (e.g. `flash_serve`,
/// `halogen-tools`): `CARGO_BIN_EXE_<name>` (set by cargo for every binary
/// of the packages `cc-contract-tests` dev-depends on), with a
/// `target/{debug,release}` walk-up as fallback.
pub fn subject_binary(binary: &str) -> Option<PathBuf> {
    if let Ok(path) = env::var(format!("CARGO_BIN_EXE_{binary}")) {
        return Some(PathBuf::from(path));
    }
    let root = crate::repo_root()?;
    ["debug", "release"]
        .iter()
        .map(|profile| root.join("target").join(profile).join(binary))
        .find(|candidate| candidate.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_env_vars_follow_the_naming_rule() {
        assert_eq!(reference_env_var("flash_serve"), "HALOGEN_FLASH_SERVE");
        assert_eq!(
            reference_env_var("halogen-tools"),
            "HALOGEN_HALOGEN_TOOLS".to_string()
        );
        assert_eq!(
            reference_env_var("cold-cathode"),
            "HALOGEN_COLD_CATHODE".to_string()
        );
    }
}
