# cold-cathode

> [!WARNING]
> **Work in progress.** The CLI contracts for all five entry points are
> implemented and verified by differential testing, and on ROCm hosts the
> engine now loads the reference's shipped AMDGPU code objects — but the
> GPU compute path (running generation through those kernels) is still
> under construction, so generation latency does not yet match upstream
> halogen. Treat this repository as an in-progress proof of concept, not a
> drop-in replacement.

## What this is

**cold-cathode** is a Rust reimplementation of the `halogen-flash-server`
binary blobs published by
[peonist-ai/halogen-flash-server](https://github.com/peonist-ai/halogen-flash-server)
(v0.18.21). The three vendored ELF binaries (`flash_serve`, `halogen-tools`,
`halogen-npu`), the `halogen-healthcheck` shell script and the
`halogen-flash-server` launcher were reverse-engineered with
[REA](https://github.com/morluto/rea) and rebuilt
in Rust, validated by differential testing against the original binaries: for
every way you can invoke a halogen binary, invoking the cold-cathode equivalent
produces the same stdout, stderr and exit codes. The upstream Python front-end
and data assets (`flash-tune.plan`, `draft-vocab.lists`, `npu/models.txt`) are
reused as data, not rewritten.

## Credit

All the credit for halogen goes to its authors —
[peonist-ai/halogen-flash-server](https://github.com/peonist-ai/halogen-flash-server)
is excellent work, and this repository exists only because it does.

This package is **simply a validation of local model capability, Rust
implementation and agentic flows and skills** — it is not a replacement for
halogen and not a competing product. Upstream halogen remains the canonical
implementation; cold-cathode is a proof-of-concept exercise in what AI agents,
armed with reverse-engineering tooling and structured skills, can faithfully
rebuild from a shipped binary.

## Provenance & licensing

| Item | Detail |
|---|---|
| Upstream source | [`github.com/peonist-ai/halogen-flash-server`](https://github.com/peonist-ai/halogen-flash-server) v0.18.21 |
| Upstream licence | Peonist EULA (`lib.licenses.unfreeRedistributable`) |
| cold-cathode licence | GPL-3.0 (see [`LICENSE`](../LICENSE)) |
| Rust code | Original work, derived from reverse-engineered contracts — not derived from upstream source code |
| Reference derivation | `packages/halogen-flash-server/default.nix`, reconstructed from the upstream Nix derivation; `nix build .#halogen-flash-server` reproduces the v0.18.21 package |
| OCI image digest | `sha256:21006e5347f974241fd11e7bac5fc31759a4187f8267f7b52d8b564986b69821` |

The Rust sources are original implementations written against contracts
(specs) derived from REA evidence — strings, symbols, decompiled control flow
and observed CLI behaviour — so the copyleft licence on the Rust code does not
conflict with the upstream EULA on the vendored binaries. The REA evidence
itself is committed under [`docs/analysis/`](../docs/analysis/) so the
evidence-before-implementation trail is reviewable.

## Quick start

The repository is driven entirely by a Nix flake (single pinned `nixpkgs`
input). It offers two packages:

```console
$ nix build .#cold-cathode          # the Rust reimplementation
$ nix build .#halogen-flash-server  # the upstream v0.18.21 package (reference)
```

Run the reimplementation:

```console
$ result/bin/cold-cathode
```

(or `nix run .#cold-cathode`). For a development shell with the full Rust
toolchain — `cargo`, `rustc`, `clippy`, `rustfmt`, `cargo-nextest`,
`nixfmt-rfc-style` — plus the halogen reference package on `PATH` (needed by
the differential test harness):

```console
$ nix develop
```

The Rust binaries keep the upstream names (`flash_serve`, `halogen-tools`,
`halogen-npu`, `halogen-healthcheck`) so the launcher's dispatch logic ports
1:1; only the launcher itself is renamed (`cold-cathode`). Installing both
packages into one Nix profile therefore collides on the shared bin names —
each package stands alone.

## How it was built

1. **REA reverse engineering.** Each vendored binary was opened with
   [REA](https://github.com/morluto/rea): strings, symbols and imports
   extracted, `main` and CLI argument parsing decompiled, and the raw evidence
   exported to `docs/analysis/evidence/<binary>/`. (`halogen-healthcheck`
   turned out to be a shell script, not an ELF — its contract was derived by
   transcription instead.)
2. **Spec-driven development.** Per-binary contract specs
   (`docs/analysis/specs/<binary>.md`) were written from the evidence —
   complete flag tables, exact error wording, env-var tables, exit codes and
   verbatim help text — reviewed, and then frozen as the source of truth. The
   specs, not the decompiled source, define correctness.
3. **Differential test harness.** `crates/cc-contract-tests` reads per-binary
   invocation matrices (`tests/contract/matrix/*.toml`), executes both the
   halogen binary and the cold-cathode binary for every entry, normalises
   volatile output (argv[0], timestamps, PIDs, ports, store paths), and asserts
   byte equality. Written test-first: the harness ran red against stub binaries
   before any real implementation existed.
4. **Agentic workflow.** The whole project — planning, REA runs, implementation,
   review, docs — was executed by AI agents following codified skills
   (reverse-engineering, TDD, verification-before-completion) with an
   orchestrator/subagent execution model: independent work streams dispatched
   in parallel, every code change reviewed by a separate code-reviewer agent
   before it was considered done.

## Unsafe Rust

The full policy lives in [`docs/unsafe-policy.md`](../docs/unsafe-policy.md).
Summary:

| Crate | `unsafe` blocks | Notes |
|---|---|---|
| `cc-launcher` | 0 | `#![warn(unsafe_op_in_unsafe_fn)]` crate-wide |
| `cc-flash-serve` | 0 | `#![warn(unsafe_op_in_unsafe_fn)]` crate-wide |
| `cc-halogen-tools` | 0 | `#![warn(unsafe_op_in_unsafe_fn)]` crate-wide |
| `cc-halogen-npu` | 0 | `#![warn(unsafe_op_in_unsafe_fn)]` crate-wide |
| `cc-halogen-healthcheck` | 0 | `#![warn(unsafe_op_in_unsafe_fn)]` crate-wide |

**Zero `unsafe` blocks across all five crates.** The policy permits `unsafe`
only where porting C++ patterns demands it (minimised, isolated in dedicated
modules, `// SAFETY:`-annotated, tested); so far no port has needed it.

## Project status

This is a **validation project** — upstream
[halogen-flash-server](https://github.com/peonist-ai/halogen-flash-server) is
the canonical implementation, and cold-cathode exists to demonstrate the
reverse-engineering and rebuild process, not to supersede it.

Current state of the work:

- All five entry points (`cold-cathode`, `flash_serve`, `halogen-tools`,
  `halogen-npu`, `halogen-healthcheck`) are implemented in Rust and their CLI
  contract matrices are in place.
- GPU-gated paths (ROCm-dependent inference behaviour, NPU/XRT probes) are
  deferred on non-GPU hosts; GPU-dependent outputs are compared structurally,
  not token-for-token.
- The HTTP API contract tests are deferred to the API spec
  (`docs/analysis/specs/api.md`); the flash_serve CLI surface is covered.
- Known caveat: the `.hgn` writer format (HGN1 checkpoint layout) was
  decompiled but the writer remains unverified against the original — noted in
  the `halogen-tools` spec.
- The GPU compute path is under active construction: the engine loads the
  reference's shipped AMDGPU code objects on ROCm hosts (module registry
  in place), but routing the forward pass through those kernels — and the
  resulting generation-latency parity with upstream — is still in progress.
- Changes land on `main` through reviewed commits from the `init` working
  branch.
