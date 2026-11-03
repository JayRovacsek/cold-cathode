//! Minimal TOML-subset parser for the contract matrices (plan §5).
//!
//! The dev shell has no crates.io access (verified 2026-11-03: `cargo add`
//! cannot reach the registry), so the harness hand-rolls the subset of TOML
//! the matrices use: full-line comments, `[[case]]` tables, and string,
//! boolean, single-line string-array and inline-table values.

use std::fmt;
use std::fs;
use std::path::Path;

/// One matrix entry: an invocation of `flash_serve` plus the spec section it
/// proves (plan §5 traceability requirement).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Case {
    pub name: String,
    /// Grouping key: `serve`, `dispatch`, `repack`, `resident-gib` or `env`.
    pub mode: String,
    /// Citation of the spec section this entry proves.
    pub spec: String,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
    /// Entry is listed but not executable yet (e.g. missing fixture).
    pub deferred: bool,
    pub reason: String,
    /// Entry needs a GPU; skipped in CLI-only mode.
    pub requires_gpu: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub struct MatrixError {
    pub line: usize,
    pub message: String,
}

impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

fn error(line: usize, message: &str) -> MatrixError {
    MatrixError {
        line,
        message: message.to_string(),
    }
}

/// Load and parse a matrix file.
pub fn load_matrix(path: &Path) -> Result<Vec<Case>, MatrixError> {
    let text = fs::read_to_string(path)
        .map_err(|e| error(0, &format!("cannot read {}: {e}", path.display())))?;
    parse_matrix(&text)
}

/// Parse the TOML subset used by the matrices.
pub fn parse_matrix(text: &str) -> Result<Vec<Case>, MatrixError> {
    let mut cases: Vec<Case> = Vec::new();
    let mut current: Option<Case> = None;
    let mut header_line = 0usize;

    for (index, raw) in text.lines().enumerate() {
        let line = raw.trim();
        let line_no = index + 1;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line == "[[case]]" {
            if let Some(case) = current.take() {
                cases.push(validate(case, header_line)?);
            }
            current = Some(Case::default());
            header_line = line_no;
            continue;
        }
        let (key, value) = line
            .split_once('=')
            .ok_or_else(|| error(line_no, &format!("expected `key = value` or [[case]], found {line:?}")))?;
        let key = key.trim();
        let value = value.trim();
        let case = current
            .as_mut()
            .ok_or_else(|| error(line_no, &format!("key {key:?} outside a [[case]] table")))?;
        match key {
            "name" | "mode" | "spec" | "reason" => {
                let text = parse_quoted(value)
                    .ok_or_else(|| error(line_no, &format!("{key} must be a quoted string")))?;
                match key {
                    "name" => case.name = text,
                    "mode" => case.mode = text,
                    "spec" => case.spec = text,
                    _ => case.reason = text,
                }
            }
            "deferred" | "requires_gpu" => {
                let flag = match value {
                    "true" => true,
                    "false" => false,
                    _ => return Err(error(line_no, &format!("{key} must be true or false"))),
                };
                if key == "deferred" {
                    case.deferred = flag;
                } else {
                    case.requires_gpu = flag;
                }
            }
            "args" => case.args = parse_string_array(value, line_no)?,
            "env" => case.env = parse_inline_table(value, line_no)?,
            other => return Err(error(line_no, &format!("unknown key {other:?}"))),
        }
    }
    if let Some(case) = current.take() {
        cases.push(validate(case, header_line)?);
    }
    Ok(cases)
}

fn validate(case: Case, header_line: usize) -> Result<Case, MatrixError> {
    for field in ["name", "mode", "spec"] {
        let value = match field {
            "name" => &case.name,
            "mode" => &case.mode,
            _ => &case.spec,
        };
        if value.is_empty() {
            return Err(error(header_line, &format!("[[case]] is missing {field}")));
        }
    }
    if case.deferred && case.reason.is_empty() {
        return Err(error(header_line, "deferred cases need a reason"));
    }
    Ok(case)
}

fn parse_quoted(value: &str) -> Option<String> {
    let inner = value.strip_prefix('"')?.strip_suffix('"')?;
    Some(inner.to_string())
}

/// Split a comma-separated body (array or inline table) while keeping commas
/// inside quoted strings intact.
fn split_commas(body: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    for ch in body.chars() {
        match ch {
            '"' => {
                in_quotes = !in_quotes;
                current.push(ch);
            }
            ',' if !in_quotes => {
                if !current.trim().is_empty() {
                    parts.push(current.trim().to_string());
                }
                current = String::new();
            }
            _ => current.push(ch),
        }
    }
    if !current.trim().is_empty() {
        parts.push(current.trim().to_string());
    }
    parts
}

fn parse_string_array(value: &str, line: usize) -> Result<Vec<String>, MatrixError> {
    let body = value
        .strip_prefix('[')
        .and_then(|v| v.strip_suffix(']'))
        .ok_or_else(|| error(line, "array value must be on a single line: [\"a\", \"b\"]"))?;
    split_commas(body)
        .iter()
        .map(|part| {
            parse_quoted(part).ok_or_else(|| error(line, &format!("array element {part:?} is not quoted")))
        })
        .collect()
}

fn parse_inline_table(value: &str, line: usize) -> Result<Vec<(String, String)>, MatrixError> {
    let body = value
        .strip_prefix('{')
        .and_then(|v| v.strip_suffix('}'))
        .ok_or_else(|| error(line, "inline table must be on a single line: { KEY = \"value\" }"))?;
    split_commas(body)
        .iter()
        .map(|part| {
            let (key, val) = part
                .split_once('=')
                .ok_or_else(|| error(line, &format!("inline table entry {part:?} lacks `=`")))?;
            Ok((key.trim().to_string(), parse_quoted(val.trim()).ok_or_else(|| error(line, &format!("inline table value for {key:?} is not quoted")))?))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
# comment
[[case]]
name = "serve-no-args"
mode = "serve"
spec = "Observed behaviours #1"
args = []

[[case]]
name = "serve-port"
mode = "serve"
spec = "Flag table — serve mode (--port)"
args = ["--port", "abc"]
env = { HALOGEN_PLD = "1,2" }

[[case]]
name = "repack-valid-gguf"
mode = "repack"
spec = "Flag table — repack mode"
args = ["--repack", "x.gguf"]
deferred = true
reason = "no fixture"
requires_gpu = false
"#;

    #[test]
    fn parses_all_value_shapes() {
        let cases = parse_matrix(SAMPLE).expect("sample parses");
        assert_eq!(cases.len(), 3);
        assert_eq!(cases[0].args, Vec::<String>::new());
        assert_eq!(cases[1].args, vec!["--port".to_string(), "abc".to_string()]);
        assert_eq!(
            cases[1].env,
            vec![("HALOGEN_PLD".to_string(), "1,2".to_string())]
        );
        assert!(cases[2].deferred);
        assert_eq!(cases[2].reason, "no fixture");
        assert!(!cases[2].requires_gpu);
    }

    #[test]
    fn rejects_missing_required_fields() {
        let err = parse_matrix("[[case]]\nname = \"x\"\n").unwrap_err();
        assert_eq!(err.line, 1);
        assert!(err.message.contains("missing mode"));
    }

    #[test]
    fn rejects_key_outside_case() {
        let err = parse_matrix("name = \"x\"\n").unwrap_err();
        assert!(err.message.contains("outside a [[case]]"));
    }

    #[test]
    fn rejects_unknown_key() {
        let err = parse_matrix("[[case]]\nname = \"x\"\nmode = \"m\"\nspec = \"s\"\nnope = \"v\"\n")
            .unwrap_err();
        assert!(err.message.contains("unknown key"));
    }

    #[test]
    fn rejects_deferred_without_reason() {
        let err =
            parse_matrix("[[case]]\nname = \"x\"\nmode = \"m\"\nspec = \"s\"\ndeferred = true\n")
                .unwrap_err();
        assert!(err.message.contains("reason"));
    }
}
