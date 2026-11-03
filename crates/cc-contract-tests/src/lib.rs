//! Differential contract-test harness for cold-cathode (plan §5, Phase 3).
//!
//! Reads the invocation matrices under `tests/contract/matrix/`, executes
//! both the halogen reference binary and the cold-cathode binary for each
//! entry, applies the §5 normalisation rules and asserts stdout/stderr and
//! exit-code equality. The `tests/contract.rs` integration tests drive this
//! library; the unit tests here cover the harness itself.

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
