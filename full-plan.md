# Cold Cathode — full implementation plan

> **Status:** approved for execution · written 2026-10-30 · updated 2026-11-03 · **branch:** `init`
> **Source prompt:** `task.md` (this document supersedes it for execution).
> **Audience:** the long-running agent executing the work. Every rule here is
> binding; anything left open is resolved by asking the user, never by assuming.

---

## 1. Mission

**Cold Cathode** is a Rust reimplementation of the `halogen-flash-server`
binary blobs published by
[peonist-ai/halogen-flash-server](https://github.com/peonist-ai/halogen-flash-server)
(v0.18.21), produced by reverse-engineering the vendored ELF binaries with
[REA](https://github.com/morluto/rea) and validated by differential testing
against the originals.

The project is a **validation exercise**, not a competing product: a
demonstration of local model capability, Rust implementation quality, and
agentic flows and skills. Halogen's excellent work is credited prominently in
`.github/README.md` (see §8).

The user-facing contract is: **for every way you can invoke halogen's
binaries, invoking the cold-cathode equivalent produces the same UX, behaviour
and outcome** (§5 defines "matches" precisely).

---

## 2. Confirmed requirements (non-negotiable)

| # | Requirement | Source |
|---|-------------|--------|
| R1 | The reimplementation language **MUST be Rust** (stable, from nixpkgs). | task.md |
| R2 | The repository **MUST be driven by a Nix flake** that provides all development tools and tests. The flake stays **as simple as possible**: a single `nixpkgs` input pinned to `github:nixos/nixpkgs/3ec636362e1816bc2f0888fa0fa2c69f1bc9e504` (the nix-config pin, so the copied halogen derivation's hashes stay valid and its binary cache is reused). Extra Rust-tooling inputs (e.g. `crane`, `rust-bin`) are added only when a documented need arises. | task.md + user guidance |
| R3 | The flake **offers two packages**: `packages.x86_64-linux.halogen-flash-server` (v0.18.21) — reconstructed from store drv `3mjvnvg02h6xs4lp9qp4ym5pmyz7k734-halogen-flash-server-0.18.21.drv`; the nix-config source path no longer exists on this machine (decision log 2026-11-03) — and `packages.x86_64-linux.cold-cathode`, built from the Rust workspace. | task.md + user guidance |
| R4 | **Scope:** the four upstream binaries (`flash_serve`, `halogen-tools`, `halogen-npu`, `halogen-healthcheck`) **plus** the `halogen-flash-server` launcher (the makeWrapper'd entrypoint). The Python front-end (`share/halogen/tools`) and data assets (`flash-tune.plan`, `draft-vocab.lists`, `npu/models.txt`) are reused as data, **not rewritten**. | clarification Q1 |
| R5 | **Definition of "matches":** for every CLI invocation — identical stdout, stderr and exit codes (help text, version output, argument validation, error wording). For the server — identical HTTP API surface (routes, request/response schemas, error bodies). GPU-dependent inference outputs are compared **structurally, not token-for-token**. | clarification Q2 |
| R6 | **Unsafe Rust** is permitted only where porting C++ patterns demands it: minimised, isolated in dedicated modules behind safe APIs, annotated with `// SAFETY:` comments, and covered by tests. Policy in §7. | task.md |
| R7 | `.github/README.md` explains what this is, credits halogen's excellent work, and states the project is a validation of local model capability, Rust implementation and agentic flows and skills. Content spec in §8. | task.md |
| R8 | **All work happens in a git worktree**; everything converges on the **`init` branch** for review. `main` stays untouched until the review passes. Commits follow the AGENTS.md Conventional Commits rules (lowercase, `Assisted-by:` trailer, message-file workflow). | task.md + clarification Q3 |
| R9 | **REA evidence is committed** to `docs/analysis/` — exported evidence and derived per-binary specs — so the evidence-before-implementation trail is reviewable. | clarification Q4 |
| R10 | The executing agent **never assumes**: it loads the matching skills before each phase (§9), asks the user whenever blocked by ambiguity, and delegates code review per AGENTS.md. | task.md |

---

## 3. Repository layout

```
cold-cathode/
├── flake.nix                              # R2: single nixpkgs input, both packages + devShell
├── flake.lock
├── .gitignore                             # node_modules-style basics + .tmp/, tmp/, target/
├── LICENSE                                # GPL-3.0 (already on main)
├── task.md                                # original prompt (already committed on init)
├── full-plan.md                           # this document (the execution ledger)
├── .github/
│   └── README.md                          # R7 — the repo README (no root README.md)
├── packages/
│   ├── halogen-flash-server/default.nix   # verbatim copy from nix-config (R3)
│   └── cold-cathode/default.nix           # rustPlatform.buildRustPackage over the workspace
├── crates/                                # Cargo workspace, one crate per binary
│   ├── cc-launcher/                       #   → $out/bin/cold-cathode
│   ├── cc-flash-serve/                    #   → $out/bin/flash_serve
│   ├── cc-halogen-tools/                  #   → $out/bin/halogen-tools
│   ├── cc-halogen-npu/                    #   → $out/bin/halogen-npu
│   ├── cc-halogen-healthcheck/            #   → $out/bin/halogen-healthcheck
│   └── cc-contract-tests/                 # §5 differential harness (tests only)
├── tests/contract/matrix/                 # per-binary invocation matrices (*.toml)
└── docs/
    ├── analysis/
    │   ├── evidence/<binary>/             # REA exports (R9)
    │   └── specs/<binary>.md              # derived per-binary contracts (R9)
    └── unsafe-policy.md                   # §7, mirrored in README
```

**Binary naming decision (locked):** the Rust binaries keep the upstream names
(`flash_serve`, `halogen-tools`, `halogen-npu`, `halogen-healthcheck`) so the
launcher's dispatch logic ports 1:1 from `entrypoint.sh` and the contract
harness can compare without renaming heuristics beyond standard argv[0]
normalisation. The launcher itself installs as **`cold-cathode`** (not
`halogen-flash-server`), which is what distinguishes the two packages. The
README documents this; installing both packages into one nix profile collides
on the shared bin names — documented, not solved (each package stands alone).

---

## 4. Execution phases

Phases are strictly ordered; each has an **exit gate** that must be green
before the next begins. Tick the checkboxes in this file as work completes —
this document doubles as the progress ledger.

### Phase 0 — Git baseline ✅ (done at plan approval)
- [x] `init` branch created from `main` via an OpenCode-managed worktree.
- [x] `task.md` + `full-plan.md` committed as the first commits on `init`.

### Phase 1 — Nix foundation (before any Rust code)
**Objective:** the flake exists and builds *before* implementation starts, so
Rust work never happens outside Nix (R2 ordering requirement).
- [x] Write `flake.nix`: single `nixpkgs` input (R2 pin); `packages.x86_64-linux.halogen-flash-server` via `callPackage packages/halogen-flash-server/default.nix`; `packages.x86_64-linux.cold-cathode` via `rustPlatform.buildRustPackage`; `devShells.x86_64-linux.default` with `rustc`, `cargo`, `clippy`, `rustfmt`, `cargo-nextest`, `nixfmt-rfc-style`, and the halogen package on `PATH` (needed by the §5 harness).
- [x] Reconstruct `packages/halogen-flash-server/default.nix` from store drv `3mjvnvg02h6xs4lp9qp4ym5pmyz7k734` (v0.18.21); fidelity verified by drv-path match (`nix eval .#halogen-flash-server.drvPath` equals the store drv).
- [x] Add `.gitignore` (`.tmp/`, `tmp/`, `target/`).
- [x] Create the Cargo workspace skeleton: all six crates compile as stubs (`fn main() {}`); `Cargo.lock` committed.
- [x] **Gate:** `nix flake check` green · `nix build .#halogen-flash-server` green · `nix build .#cold-cathode` green (stubs) · `nix develop` provides the full toolchain.

### Phase 2 — REA reverse engineering
**Objective:** produce the per-binary contracts that everything else is tested
against, with committed evidence (R9).
- [x] Load the `reverse-engineer-anything` skill and follow it for each of the four ELF binaries (opened from the `nix build .#halogen-flash-server` result path). *(healthcheck is a shell script — transcribed instead; see decision log 2026-11-03.)*
- [x] Per binary: extract strings/symbols/imports, decompile `main` and CLI argument parsing, export evidence to `docs/analysis/evidence/<binary>/`.
- [x] Launcher contract needs **no REA** — it is plain source: `deploy/entrypoint.sh` + the makeWrapper env defaults (Appendix A). Transcribe it into `docs/analysis/specs/launcher.md`.
- [x] HTTP route table comes from the upstream Python front-end source (`share/halogen/tools`); transcribe routes/schemas into `docs/analysis/specs/api.md`.
- [x] Write `docs/analysis/specs/<binary>.md` per binary using the spec template: entry points · complete flag table (name, type, default, validation, exact error wording) · env-var table · exit codes · verbatim help text · subcommands · observed behaviours · links to evidence files.
- [x] **Gate:** specs are internally consistent and reviewed (review agent sign-off; ambiguities raised with the user). Specs become the frozen contract for Phase 3. *(all five specs signed off 2026-11-03; flash_serve review approve-with-nits, remaining-specs review PASS — reports in `.tmp/`.)*

### Phase 3 — Differential test harness (tests before implementation)
**Objective:** the harness exists and runs **red** against the stub binaries
before any real implementation — proving the harness works (TDD).
- [x] Implement `crates/cc-contract-tests`: reads `tests/contract/matrix/<binary>.toml` invocation matrices, executes **both** the halogen binary (absolute store path) and the cold-cathode binary for each entry, applies §5 normalisation rules, asserts stdout/stderr/exit-code equality.
- [ ] HTTP contract tests: start each server on an ephemeral port, compare route tables, response schemas and error bodies; GPU-gated per §5. (flash_serve CLI only so far; HTTP surface deferred to `docs/analysis/specs/api.md` — review F4)
- [x] Matrices derived from the Phase 2 specs; every test case cites its spec section (traceability).
- [x] **Gate:** harness runs red against the stubs with a documented failure report; matrix covers every flag, subcommand and error path listed in the specs.

### Phase 4 — Rust implementation (iterative, smallest first)
**Objective:** all five entry points green against the full matrix.
- [x] Order: `cc-halogen-healthcheck` → `cc-halogen-tools` → `cc-halogen-npu` → `cc-flash-serve` → `cc-launcher` (launcher last — it orchestrates the others). (implemented out of order per the 2026-11-03 scoping: `cc-flash-serve` `404947f` first, then `cc-halogen-healthcheck` `e8ed65c`, `cc-halogen-npu` `e189c80`, `cc-halogen-tools` `4867333`, `cc-launcher` `96aba28`; all five reviewed — reports in `.tmp/`)
- [x] Per-binary loop: implement from spec → unit tests → contract tests green → `cargo fmt` + `clippy -D warnings` + `cargo nextest run` green → commit → code review (code-reviewer agent per AGENTS.md) → resolve findings → mark done.
- [x] Unsafe blocks only per §7; each lands with its `// SAFETY:` note and tests in the same commit. (all five crates are std-only with zero `unsafe` — nothing to annotate; every crate carries the crate-wide `#![warn(unsafe_op_in_unsafe_fn)]` attribute per §7.5)
- [x] **Gate:** every contract matrix green (or GPU-gated with documented skips on this machine). (all five matrices green; the cold-cathode matrix runs live via the devShell `HALOGEN_COLD_CATHODE` override — 18 pass + 7 documented defers; the flash_serve HTTP tests remain a Phase 3 F4 carry-over)

### Phase 5 — Packaging and documentation
- [x] Finalise `packages/cold-cathode/default.nix` so `$out/bin` contains all five entry points and `$out/opt/halogen/` data paths match what the launcher expects. (`def23cb` bin names, `5a2064c` devShell export; build verified green)
- [x] Write `.github/README.md per §8; write `docs/unsafe-policy.md` (§7). (`d895c4b`, full-branch review fixes folded in)
- [ ] **Gate:** `nix build .#cold-cathode` green; README reviewed by the user. (build green; ticks on user approval of the merge)

### Phase 6 — Final validation and review
- [x] Full matrix run; `nix flake check`; both packages build. (2026-11-04: `nix flake check` all passed; `nix build .#cold-cathode .#halogen-flash-server --no-link` OK; `cargo test --workspace` 18 suites ok; contract tests 15 unit + 11 integration pass)
- [x] Run a full-branch review (code-reviewer agent); resolve all findings. (2026-11-04: approve-with-nits, no critical findings; F1–F3 applied in this commit — report in `.tmp/review-report-full-branch.md`)
- [x] Present `init` → `main` merge to the user for approval. **Do not merge without it.** (approved 2026-11-04; `main` fast-forwarded to `d2c4551`)

### Phase 7 — Full engine port (scope extension, 2026-11-04)
**Objective:** port the flash_serve engine internals beyond the CLI contract — weights, generation, MTP speculative decoding, KV-cache management, NPU/XRT (user scoping decision, §12); every increment ships its own `HALOGEN_LOG`-gated verbose logging.
- [x] Engine-internals blueprint spec (`docs/analysis/specs/flash_serve-engine.md`, `5a407bc`; round-1 review approve-with-nits, nits fixed in the review-fix commit).
- [x] Checkpoint load into RAM (`crates/cc-flash-serve/src/hgn.rs` + serve-path wiring, `c836d0b`; round-1 review approve-with-nits; entry-layout/version-gate fix landed in `698043a` against the reference checkpoint — round-2 review approve, `.tmp/review-report-engine-round2.md`).
- [x] Launcher `wait_for_engine` before spawning the api (`crates/cc-launcher/src/lib.rs`, `4a6c577`; fixes the live `ConnectionRefusedError` regression from `c836d0b`; round-2 review approve).
- [x] Survive transient accept errors in the serve loop (`crates/cc-flash-serve/src/protocol.rs`, `6ecf49f`; fixes the live silent-engine-death symptom — RAM filled then dropped after the front-end's first reconnect cycle: `serve_forever` broke its accept loop on the first `ECONNABORTED`, which std's `incoming()` surfaces as an `Err`; the reference's decompiled `main` re-accepts on every errno except the `EINTR` its signal handlers use for shutdown. Small fix per AGENTS.md exception — no dedicated review; folded into the next round).
- [ ] Forward pass + sampling (real `GEN` output).
- [ ] MTP speculative decoding.
- [ ] KV-cache management.
- [ ] NPU/XRT integration (`halogen-npu listen`).

---

## 5. Test architecture

**Categories**
1. **CLI contract tests (differential):** every invocation in the matrices
   runs against both binaries; stdout/stderr/exit codes compared after
   normalisation. This is the primary proof of R5.
2. **Unit tests:** per-crate logic (arg parsing, config loading, formatting).
3. **HTTP API contract tests:** route table, request/response schemas, error
   bodies — compared between the two servers.
4. **Property tests** where cheap and meaningful (e.g. arg-parser round-trips).

**Normalisation rules (locked):** before comparison, both outputs are
normalised for — argv[0]/program name, timestamps, PIDs, bound port numbers,
Nix store paths (each `$out` mapped to a placeholder), and timing/duration
values. Any output not covered by a rule is compared byte-for-byte.

**GPU gating:** server tests need a ROCm device. The harness probes
`/dev/dri`; on machines without one, server tests skip with an explicit
warning and the run reports "CLI-only mode". This machine (gfx1151,
`/dev/dri/card1`) runs the full suite. GPU-dependent inference outputs are
compared structurally (schema, finish reason, token count bounds), not
token-for-token (R5).

**Tooling:** `cargo nextest` drives all Rust tests; the halogen reference
binaries are reached via the nix-built package path from the dev shell — never
via a mutable install.

---

## 6. Nix flake design

```nix
# flake.nix — the whole input set (R2)
{
  inputs.nixpkgs.url = "github:nixos/nixpkgs/3ec636362e1816bc2f0888fa0fa2c69f1bc9e504";
  outputs = { self, nixpkgs }: {
    packages.x86_64-linux = {
      halogen-flash-server = pkgs.callPackage ./packages/halogen-flash-server/default.nix {};
      cold-cathode         = pkgs.callPackage ./packages/cold-cathode/default.nix {};
    };
    devShells.x86_64-linux.default = …; # rustc, cargo, clippy, rustfmt, cargo-nextest, nixfmt, halogen on PATH
  };
}
```

- `packages/cold-cathode/default.nix` uses `rustPlatform.buildRustPackage`
  over the workspace (no `crane` unless a documented build-performance need
  appears — then add it as a second input and record it in the decision log).
- The halogen derivation is copied **verbatim** from nix-config; its pinned
  `fetchFromGitHub` and `dockerTools.pullImage` hashes must build unchanged
  against the shared nixpkgs pin (verified at the Phase 1 gate).
- Formatting: `nixfmt-rfc-style` for all `.nix` files (AGENTS.md).

---

## 7. Unsafe Rust policy (R6)

1. Default is safe Rust; `unsafe` requires justification in the **spec**
   ("the C++ original does X via pointer arithmetic; the port mirrors it").
2. All `unsafe` lives in dedicated modules (e.g. `cc-flash-serve::raw`),
   never inline in business logic; safe wrappers expose safe APIs only.
3. Every block carries a `// SAFETY:` comment explaining the invariant.
4. Every unsafe module ships tests exercising its invariants and its failure
   modes (mirrors §5's "thoroughly tested" requirement).
5. `#![warn(unsafe_op_in_unsafe_fn)]` crate-wide; clippy `unsafe` lints at
   pedantic level.
6. The README's "Unsafe Rust" section lists every unsafe module with its
   rationale ("clearly described" requirement).

---

## 8. `.github/README.md` content spec (R7)

Sections, in order:
1. **What this is** — one paragraph: cold-cathode is a Rust reimplementation
   of the halogen-flash-server binary blobs, built by reverse engineering
   (REA) and validated by differential testing against the original.
2. **Credit** — halogen's excellent work, with a link to
   `peonist-ai/halogen-flash-server`; explicit statement that this package is
   *simply a validation of local model capability, Rust implementation and
   agentic flows and skills* — not a replacement or competing product.
3. **Provenance & licensing** — upstream is Peonist EULA (unfreeRedistributable);
   cold-cathode is GPL-3.0; the Rust code is original work derived from
   reverse-engineered contracts.
4. **Quick start** — `nix build`, running `cold-cathode`, both packages.
5. **How it was built** — REA methodology, spec-driven development, the
   differential test harness, and the agentic workflow (skills + subagents).
6. **Unsafe Rust** — pointer to `docs/unsafe-policy.md` and the summary table.
7. **Project status** — validation project; upstream halogen is canonical.

---

## 9. Operating instructions for the executing agent

1. **Session start:** load `using-superpowers`; re-read `full-plan.md`; check
   the checkbox state before doing anything.
2. **Per-phase skill loading (AGENTS.md "Skill & Tool Selection"):**
   - Phase 2 → `reverse-engineer-anything`
   - Phases 3–4 → `test-driven-development` (red-first is already mandated)
   - Every gate → `verification-before-completion` (evidence before claims)
   - Every code change → code-reviewer agent (AGENTS.md rule 2); use
     `requesting-code-review` at phase boundaries
   - Any design fork not answered here → `brainstorming` first, then ask the
     user if still ambiguous (R10)
3. **Never assume (R10):** ambiguity in specs, matrices, or the flake → ask
   the user. Record every resolved ambiguity in the **decision log** (§12).
4. **Git discipline (R8):** work only in the `init` worktree; commit via
   `.tmp/commit-msg.txt` + `git commit -F` (never inline `-m`, never
   `--no-verify`); Conventional Commits, all-lowercase, `Assisted-by:` trailer
   with the executing model's real identifier.
5. **Reporting cadence:** after each phase gate, report to the user: gate
   evidence, updated checkboxes, next phase, open questions.
6. **Cleanup:** remove `.tmp/` after each commit; no system-tmp usage.

---

## 10. Definition of Done

- [ ] All six phases' gates green and ticked in this file.
- [ ] `nix flake check`, `nix build .#halogen-flash-server`, `nix build .#cold-cathode` all green on this machine.
- [ ] Every CLI matrix green (or documented GPU-gated skips); HTTP contract tests green *(deferred to `docs/analysis/specs/api.md` per the Phase 3 F4 annotation — live HTTP surface validation awaits a GPU-capable host)*.
- [ ] `docs/analysis/` contains evidence + specs for all four binaries and the launcher (R9).
- [ ] `.github/README.md` reviewed by the user and credits halogen (R7).
- [ ] Full-branch code review completed with findings resolved.
- [ ] `init` branch presented to the user; merged to `main` only on approval.

---

## 11. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| `flash_serve` decompilation is ambiguous (large stripped C++ binary) | Evidence-first specs (Phase 2) frozen before coding; contract tests — not decompiled source — are the source of truth; ambiguities escalated per R10 |
| GPU-dependent behaviour varies run-to-run | Structural comparison + normalisation rules (§5); GPU probe with explicit skip mode |
| Upstream hashes drift (nixpkgs or image moved) | Flake pins the exact nix-config nixpkgs rev; Phase 1 gate verifies the verbatim copy builds before any Rust work |
| Licensing/IP (engine is Peonist EULA) | Original Rust code derived from contracts; provenance documented in README (§8.3); user consulted before any publication step |
| Scope creep (rewriting the Python front-end) | Explicitly out of scope (R4); anything beyond R4 requires a decision-log entry + user approval |
| Long-running agent loses context | This file is the single source of truth: checkboxes + decision log updated every session (§9) |

---

## Appendix A — Launcher env contract (from the makeWrapper)

The Rust launcher (`cold-cathode`) must reproduce these defaults, each
overridable from the environment (`${VAR:-default}` semantics). Table revised
2026-11-03 to the v0.18.21 makeWrapper (per the store drv):

| Variable | Default | Variable | Default |
|---|---|---|---|
| `HALOGEN_ARCH` | `gfx1151` | `HALOGEN_MATMUL_ALGOS` | `1` |
| `HIP_FORCE_DEV_KERNARG` | `1` | `HALOGEN_MATMUL_BUCKET` | `1` |
| `HALOGEN_ATTN_FA` | `64` | `HALOGEN_MATMUL_TUNING_FILE` | `/1rz4g4znpzjwh1xymhjpm42vipw92pr73vdgl6xs1hycac8kf2n9/opt/halogen/flash-tune.plan` |
| `HALOGEN_PROMPT_CACHE` | `2` | `HALOGEN_DRAFT_VOCAB_FILE` | `/1rz4g4znpzjwh1xymhjpm42vipw92pr73vdgl6xs1hycac8kf2n9/opt/halogen/draft-vocab.lists` |
| `HALOGEN_PREFILL_CHUNK` | `32768` | `HALOGEN_TOKENIZER` | `/tokenizer` |
| `HALOGEN_KV_SLOTS` | `4` | `HALOGEN_PORT` | `8730` |
| `HALOGEN_CTX` | `262144` | `HALOGEN_BIND` | `127.0.0.1` |
| `HALOGEN_MAX_TOK` | `32768` | `HALOGEN_API_PORT` | `8731` |
| `HALOGEN_MAX_TOKENS_CAP` | `65536` | `HALOGEN_QUEUE_TIMEOUT` | `3600` |
| `HALOGEN_IMAGE_VERSION` | `0.18.21` (pinned, not overridable) | `HALOGEN_IMAGE_MODES` | `all engine api bench sweep convert inspect verify ppl niah` |
| `PYTHONUNBUFFERED` | `1` | `TRANSFORMERS_VERBOSITY` | `error` |
| `HF_HUB_OFFLINE` | `1` | `_hg_hf` | `${python3-env}/bin/hf` |

The supported mode list (`all engine api bench sweep convert inspect verify
ppl niah`) defines the launcher's dispatch table; Phase 2 maps each mode to
the binary/tool that serves it.

## Appendix B — Upstream provenance

| Fact | Value |
|---|---|
| Upstream repo | `github.com/peonist-ai/halogen-flash-server` |
| Version | `0.18.21` |
| OCI image digest | `sha256:21006e5347f974241fd11e7bac5fc31759a4187f8267f7b52d8b564986b69821` |
| Upstream licence | Peonist EULA (`lib.licenses.unfreeRedistributable`) |
| Vendored binaries | `flash_serve`, `halogen-tools`, `halogen-npu`, `halogen-healthcheck` |
| cold-cathode licence | GPL-3.0 (existing `LICENSE`) |
| Reference derivation | reconstructed from store drv `3mjvnvg02h6xs4lp9qp4ym5pmyz7k734-halogen-flash-server-0.18.21.drv` (nix-config path unavailable as of 2026-11-03) |

---

## 12. Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-10-30 | Scope = all four binaries + launcher; Python front-end reused as data | user clarification Q1 |
| 2026-10-30 | "Matches" = CLI contract + HTTP API surface; GPU output compared structurally | user clarification Q2 |
| 2026-10-30 | Work in `init` worktree; task.md + full-plan.md committed there; `main` untouched until review | user clarification Q3 |
| 2026-10-30 | REA evidence committed to `docs/analysis/` | user clarification Q4 |
| 2026-10-30 | Flake kept minimal: single nixpkgs input (nix-config pin); extra Rust tooling inputs only when needed | user guidance |
| 2026-10-30 | Rust binaries keep upstream bin names; launcher installs as `cold-cathode` | §3 naming decision |
| 2026-11-03 | Target halogen v0.18.21 (newest available); version not pinned to 0.18.18 | user instruction: version doesn't matter, prefer newer |
| 2026-11-03 | `packages/halogen-flash-server/default.nix` reconstructed from store drv `3mjvnvg02h6xs4lp9qp4ym5pmyz7k734`; fidelity verified by drv-path match | nix-config source path no longer exists on this machine |
| 2026-11-03 | Flake nixpkgs pin `3ec636362e1816bc2f0888fa0fa2c69f1bc9e504` confirmed to match the live system nixpkgs (`BUILD_ID 26.11.20261023.3ec6363`) | store inspection; keeps the binary cache reusable |
| 2026-11-03 | Execution focus this round: server component (`flash_serve` → `cc-flash-serve`) first; Phase 1 and the flash_serve REA run in parallel (disjoint file ownership) | user instruction to reverse and rebuild the server component; coordination skill: maximise concurrency |
| 2026-11-03 | Phase 2 gate passed: all five specs signed off (`flash_serve` `4134bd2`, `halogen-tools` `71bf903`, `halogen-npu` `bf641c7`, `halogen-healthcheck` + `launcher` `47dafa6`, `api` `1de5084`) | review agent sign-off (`.tmp/review-report*.md`) |
| 2026-11-03 | `halogen-healthcheck` is a 937-byte shell script, not an ELF (R4 wording assumed four ELF binaries); contract derived by transcription | live store inspection; spec notes the discrepancy |
| 2026-11-03 | HTTP surface lives on port 8731 only; 8730 is the flash_serve line protocol (PING/PONG), 8740 the NPU protocol — §4/§5 wording predates the finding, `api.md` is authoritative | review adjudication 5c |
| 2026-11-03 | Phase 3 HTTP tests will stub the engine with a fixed fake engine-INFO fixture (makes `/health`, auth gate and §3 error vocabulary testable without GPU) | review adjudication 5d; superseded 2026-11-04 by the live `PING`/`INFO` implementation (`80992e4`, see below) |
| 2026-11-03 | `.hgn` format caveat accepted (HGN1 layout decompiled, writer unverified); npu XRT-stub-based probes accepted (`libxrt_coreutil.so.2` absent on host) | review adjudications 5a/5b |
| 2026-11-03 | REA MCP provider gap (`GHIDRA_INSTALL_DIR` unset; `ps`/`JAVA_HOME` missing in MCP env) — workaround: REA CLI with Ghidra 12.1.4; permanent fix at MCP launch env | flash_serve/halogen-tools/halogen-npu analyses all hit it |
| 2026-11-04 | `cold-cathode` usage strings keep the literal program name `entrypoint.sh` (general + convert usage, npu retirement line verbatim) | the frozen spec's explicit ask (`docs/analysis/specs/launcher.md` §Exact usage strings); recorded here per that ask — launcher review L3 |
| 2026-11-04 | Appendix A path variance reconciled: the live v0.18.21 wrapper uses concrete store paths for `HALOGEN_MATMUL_TUNING_FILE`, `HALOGEN_DRAFT_VOCAB_FILE` and `_hg_hf` where this plan's Appendix A shows the `/1rz4g4znpzjwh1xymhjpm42vipw92pr73vdgl6xs1hycac8kf2n9/…` forms; the spec transcribed the live wrapper and the port follows the spec | spec §Variance vs the plan's Appendix A (open question 1): the live wrapper is the contract — launcher review L3 |
| 2026-11-04 | The port honours `_hg_flash_serve` in every launching mode; the reference honours it only in `all` (hard-codes the store path in engine/bench/convert) | launcher review adjudication: accepted superset — identical behaviour when the var is unset |
| 2026-11-04 | The port omits the entrypoint's `ulimit -c 0` (line 6) | launcher review adjudication: accepted — std-only constraint (`setrlimit` needs libc); invisible on stdout/stderr/exit, the only channels the harness compares |
| 2026-11-04 | Engine INFO stubbing (adjudication 5d / api.md §10.2) superseded: `cc-flash-serve` `80992e4` answers `PING`/`INFO` on the live line protocol (reply derived from the decompiled format string + `serve_api._parse_info`); the GPU-gated `serve_socket_probe` contract test replaces the fixed fake fixture | post-merge live run on the user host showed serve_api's probe failing against the stub; review `.tmp/review-report-80992e4.md` |
| 2026-11-04 | Scope extended beyond the CLI contract: **full engine port** (weights + generation + MTP speculative decoding + KV-cache management + NPU/XRT), opened as Phase 7; logging folded into each increment (`HALOGEN_LOG` pattern) rather than a standalone pass | user scoping decision after observing the CLI-contract port does not load weights into RAM |
| 2026-11-04 | `.hgn` entry-layout contradiction adjudicated: one shared format; the port's in-entry accessors (name 32 B / dtype @32 / nbytes @40) match neither the tools' in-memory entries (nbytes @80) nor flash_serve's on-disk entries (name 96 B / dtype @96 / kind @100); the reference gates header version to {1, 2} while the port reader is ungated — code fix bundled pending a reference-written `.hgn` fixture | engine round-1 review adjudications 1–2 (`.tmp/review-report-engine-round1.md`) |
| 2026-11-04 | `HALOGEN_MTP_DEPTH` default: frozen spec says 3, decompile shows 2 (3 is only the `MTP_DEPTH_HI` default when `DEPTH` unset) — resolve before shipping generation | engine round-1 review adjudication 3 |
| 2026-11-04 | `.hgn` entry layout confirmed against the real reference checkpoint (`qwen3.8-flash-next-v2-abliterated.hgn`, 66,687,678,432 B, hard-linked into the repo root, untracked): version 2, entry_count 2015, model_id `qwen3.8-flash-next-v2`; entry = name 0..96 / dtype @96 / kind @100 / dims u64×4 @104 / data_offset @136 / nbytes @144 / uninterpreted @152; version gate {1,2} added with the engine's `checkpoint: unsupported version in %s` wording (`698043a`) | engine round-2 review approve (`.tmp/review-report-engine-round2.md`) |
| 2026-11-04 | Launcher `all` mode now ports the entrypoint's `wait_for_engine` (2s poll, `HALOGEN_ENGINE_WAIT_S` timeout, exact wordings) before spawning the api — fixes the live `ConnectionRefusedError [Errno 111]` regression from `c836d0b` (`4a6c577`) | engine round-2 review approve |
| 2026-11-04 | Serve-loop accept-error contract adjudicated: std's `incoming()` surfaces `ECONNABORTED` as an `Err` (the reference's raw `accept()` loop retries every errno except the `EINTR` its signal handlers use for shutdown, per `main-FUN_01ede7e0`); the port now retries transient kinds (ConnectionAborted/ConnectionReset/Interrupted/WouldBlock) and breaks only on fatal ones (`6ecf49f`) | live host symptom: engine died silently after the front-end's first reconnect cycle (66 GB weight Vec freed, no panic); small fix, review folded into the next round |
| 2026-11-05 | HIP backend integration (last Phase 7 item): hand-written `extern "C"` FFI for the 52-symbol import catalogue (36 HIP runtime + 13 `hipblasLt*` + 3 mangled `hipblaslt_ext`, plus `__hipPushCallConfiguration`; no hip-sys crate, cargoHash untouched), dlopen-style `libamdhip64.so.7` probe with graceful `Backend::Cpu` fallback, `FUN_01f700f0` device-lifecycle port (visibility-variable precedence, `HALOGEN_GPU_DEVICE` grammar, gfx1151 re-exec rule) with injectable fake, and `FUN_01f82130` `dmalloc` accounting feeding the kvcache pool budget math; `cold-cathode` buildInputs gain `rocmPackages.clr` | completes spec `flash_serve-mtp-kv-npu.md` §3.1; `rocmPackages.clr` (clr-7.2.4) exists at the nixpkgs pin and matches the reference RUNPATH — top-level `pkgs.clr`/`pkgs.hipblaslt` do not; `dlopen`/`dlsym`/`dlclose` declared directly (glibc >= 2.34 exports them from libc) so Cargo.lock stays unchanged |
