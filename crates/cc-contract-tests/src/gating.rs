//! Binary resolution and GPU gating (plan §5).

use std::env;
use std::path::{Path, PathBuf};

/// GPU probe: a ROCm machine exposes `/dev/kfd` and/or `/dev/dri`. Without
/// either, server tests skip and the run reports "CLI-only mode".
pub fn gpu_present() -> bool {
    Path::new("/dev/kfd").exists() || Path::new("/dev/dri").exists()
}

/// The halogen reference binary: `HALOGEN_FLASH_SERVE` wins, otherwise a
/// `flash_serve` lookup on `PATH` (the dev shell provides the nix-built
/// halogen package there — never a mutable install, per plan §5).
pub fn reference_binary() -> Option<PathBuf> {
    if let Ok(path) = env::var("HALOGEN_FLASH_SERVE") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }
    env::var("PATH")
        .ok()?
        .split(':')
        .map(|dir| PathBuf::from(dir).join("flash_serve"))
        .find(|candidate| candidate.is_file())
}

/// The cold-cathode subject binary: `CARGO_BIN_EXE_flash_serve` (set by
/// cargo for tests of packages that depend on `cc-flash-serve`), with a
/// `target/{debug,release}` walk-up as fallback.
pub fn subject_binary() -> Option<PathBuf> {
    if let Ok(path) = env::var("CARGO_BIN_EXE_flash_serve") {
        return Some(PathBuf::from(path));
    }
    let root = crate::repo_root()?;
    ["debug", "release"]
        .iter()
        .map(|profile| root.join("target").join(profile).join("flash_serve"))
        .find(|candidate| candidate.is_file())
}
