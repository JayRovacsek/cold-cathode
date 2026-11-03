//! Differential contract-test harness for cold-cathode (plan §5, Phase 3).
//!
//! Discovers the per-binary invocation matrices under
//! `tests/contract/matrix/` — one `*.toml` per binary, named after it (see
//! `discovery`) — executes both the halogen reference binary and the
//! cold-cathode binary for each entry, applies the §5 normalisation rules
//! and asserts stdout/stderr and exit-code equality. Binaries are resolved
//! per name (see `gating`): the subject via `CARGO_BIN_EXE_<name>` (with a
//! `target/{debug,release}` fallback), the reference via `HALOGEN_<NAME>` —
//! the name uppercased with `-` mapped to `_`, e.g. `HALOGEN_FLASH_SERVE` —
//! with a `PATH` fallback. The `tests/contract.rs` integration tests drive
//! this library; the unit tests here cover the harness itself.

pub mod discovery;
pub mod gating;
pub mod matrix;
pub mod normalise;
pub mod runner;

use std::path::PathBuf;

/// Repository root, derived from `CARGO_MANIFEST_DIR` (`crates/<crate>` ->
/// repo root) or overridden with `CC_CONTRACT_REPO_ROOT`.
pub fn repo_root() -> Option<PathBuf> {
    if let Ok(root) = std::env::var("CC_CONTRACT_REPO_ROOT") {
        return Some(PathBuf::from(root));
    }
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").ok()?);
    manifest.parent()?.parent().map(PathBuf::from)
}
