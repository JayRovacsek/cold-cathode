//! Matrix discovery (plan §5): every `*.toml` file under
//! `tests/contract/matrix/` is a per-binary invocation matrix, and the
//! filename stem names the subject binary (`flash_serve.toml` →
//! `flash_serve`, `halogen-tools.toml` → `halogen-tools`).
//!
//! The nix package source (`lib.sourceByRegex`) excludes the root `tests/`
//! directory, so discovery yields nothing there and the integration tests
//! skip gracefully.

use std::fs;
use std::path::{Path, PathBuf};

/// All matrices in `dir` as `(stem, path)` pairs sorted by stem. A missing
/// or unreadable directory yields an empty list; non-`toml` files are
/// ignored.
pub fn matrices_in(dir: &Path) -> Vec<(String, PathBuf)> {
    let mut found = Vec::new();
    let Ok(entries) = fs::read_dir(dir) else {
        return found;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("toml") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        found.push((stem.to_string(), path));
    }
    found.sort();
    found
}

/// All matrices in the repository's `tests/contract/matrix/` directory
/// (empty when the repo root is unavailable, e.g. outside a checkout).
pub fn matrices() -> Vec<(String, PathBuf)> {
    crate::repo_root()
        .map(|root| matrices_in(&root.join("tests/contract/matrix")))
        .unwrap_or_default()
}

/// The matrix file for binary `stem` inside matrix directory `dir`.
pub fn matrix_for_in(dir: &Path, stem: &str) -> Option<PathBuf> {
    matrices_in(dir)
        .into_iter()
        .find(|(name, _)| name == stem)
        .map(|(_, path)| path)
}

/// The matrix file for binary `stem` in the repository's matrix directory.
pub fn matrix_for(stem: &str) -> Option<PathBuf> {
    let dir = crate::repo_root()?.join("tests/contract/matrix");
    matrix_for_in(&dir, stem)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Project-local scratch dir (AGENTS.md: no system temp dirs), unique
    /// per test, removed before and after use.
    fn scratch(name: &str) -> PathBuf {
        let dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("manifest dir"))
            .join(".tmp")
            .join("discovery-tests")
            .join(name);
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    #[test]
    fn lists_toml_files_sorted_by_stem() {
        let dir = scratch("sorted");
        fs::write(dir.join("halogen-npu.toml"), "").expect("write npu");
        fs::write(dir.join("flash_serve.toml"), "").expect("write serve");
        fs::write(dir.join("notes.md"), "").expect("write notes");
        let found = matrices_in(&dir);
        assert_eq!(
            found
                .iter()
                .map(|(stem, _)| stem.as_str())
                .collect::<Vec<_>>(),
            vec!["flash_serve", "halogen-npu"]
        );
        assert_eq!(found[0].1, dir.join("flash_serve.toml"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn missing_directory_is_empty() {
        let missing = scratch("missing").join("no-such-dir");
        assert_eq!(matrices_in(&missing), Vec::<(String, PathBuf)>::new());
    }

    #[test]
    fn matrix_for_in_resolves_by_stem() {
        let dir = scratch("lookup");
        fs::write(dir.join("halogen-tools.toml"), "").expect("write tools");
        assert_eq!(
            matrix_for_in(&dir, "halogen-tools"),
            Some(dir.join("halogen-tools.toml"))
        );
        assert_eq!(matrix_for_in(&dir, "flash_serve"), None);
        let _ = fs::remove_dir_all(dir);
    }
}
