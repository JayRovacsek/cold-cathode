# Unsafe Rust policy

This document mirrors the unsafe Rust policy from the project plan (`full-plan.md` §7,
requirement R6) and records the current status across the workspace.

## Policy

1. Default is safe Rust; `unsafe` requires justification in the **spec**
   ("the C++ original does X via pointer arithmetic; the port mirrors it").
2. All `unsafe` lives in dedicated modules (e.g. `cc-flash-serve::raw`),
   never inline in business logic; safe wrappers expose safe APIs only.
3. Every block carries a `// SAFETY:` comment explaining the invariant.
4. Every unsafe module ships tests exercising its invariants and its failure
   modes (mirrors the contract-test suite's "thoroughly tested" requirement).
5. `#![warn(unsafe_op_in_unsafe_fn)]` crate-wide; clippy `unsafe` lints at
   pedantic level.
6. The README's "Unsafe Rust" section lists every unsafe module with its
   rationale ("clearly described" requirement).

## Current status

| Crate | `unsafe` blocks | Crate-wide lint |
|---|---|---|
| `cc-launcher` | 0 | `#![warn(unsafe_op_in_unsafe_fn)]` |
| `cc-flash-serve` | 0 | `#![warn(unsafe_op_in_unsafe_fn)]` |
| `cc-halogen-tools` | 0 | `#![warn(unsafe_op_in_unsafe_fn)]` |
| `cc-halogen-npu` | 0 | `#![warn(unsafe_op_in_unsafe_fn)]` |
| `cc-halogen-healthcheck` | 0 | `#![warn(unsafe_op_in_unsafe_fn)]` |

**All five crates contain zero `unsafe` blocks.** The `#![warn(unsafe_op_in_unsafe_fn)]`
attribute is set crate-wide in every crate root (`src/lib.rs` for
all five crates), and clippy's `unsafe` lints
are held to pedantic level per rule 5.

Because no port has so far required C++-style pointer arithmetic, no unsafe
module exists yet; if one is introduced, it must follow rules 1–4 (spec
justification, dedicated module, `// SAFETY:` annotation, tests in the same
commit) and be added to the table above and to the README's summary.
