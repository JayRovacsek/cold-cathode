# halogen-tools — per-binary contract spec

> **Status:** frozen contract for `cc-halogen-tools` (Phase 3/4).
> **Source binary:** `/nix/store/fn6y2w20kbgjd8da8pwxbbc3gvmiycpy-halogen-flash-server-0.18.21/bin/halogen-tools`
> **Provenance:** halogen-flash-server v0.18.21 (see `full-plan.md` Appendix B).
> **Evidence:** [`docs/analysis/evidence/halogen-tools/`](../evidence/halogen-tools/) (REA MCP session + REA CLI/Ghidra 12.1.4 decompilation + binutils + runtime probes).

Fact provenance markers used throughout: **[RV]** runtime-verified (observed
by executing the reference binary on this host) · **[D]** decompile-derived
(Ghidra 12.1.4 pseudocode, execution not observed for that path).

## Artifact identity

| Fact | Value |
|---|---|
| Format | ELF64, little-endian, `ET_DYN` (PIE), x86-64 |
| Size | 33 311 833 bytes |
| SHA-256 | `f82569b84bcdc4c7e940fba811a6cea038f08c2d90fa6ff265697266c7bf2735` |
| Entry point | `0x1dcab30` |
| ELF ABI tag | GNU/Linux (glibc 2.25-class ABI note; see `evidence/halogen-tools/elf-notes.json`) |
| `DT_NEEDED` | `libhipblaslt.so.1`, `libamdhip64.so.7`, `libstdc++.so.6`, `libm.so.6`, `libgcc_s.so.1`, `libc.so.6` |
| `DT_RUNPATH` | `…/hipblaslt-7.2.4/lib`, `…/clr-7.2.4/lib`, `…/gcc-16.2.0-lib/lib`, `…/gcc-16.2.0-libgcc/lib` |
| Symbol table | stripped (no `.symtab`; `.dynsym` carries weak libstdc++/halogen C++ symbols, e.g. `_ZN7halogen…`) |

The binary shares its engine (same `DT_NEEDED` set, same `HGN1` checkpoint
format) with `flash_serve`; `inspect`/`verify` read the engine's own `.hgn`
files without a GPU, while `ppl`/`niah` construct the engine (GPU-gated).

## Entry points and verbs

One entry point: `$out/bin/halogen-tools`. The launcher's `inspect`/`verify`
modes dispatch to this binary; its `ppl`/`niah` modes exec
`halogen_tools.py`, which is separate from this binary's `ppl`/`niah` verbs
(see `docs/analysis/specs/launcher.md`). `paired` has no launcher mode.

Verb selection is an exact, case-sensitive match on `argv[1]` (decompiled
`main` = `FUN_01ecabf0`, switch on `strlen(argv[1])` + `bcmp`):

| Verb | Grammar | Purpose |
|---|---|---|
| `inspect` | `inspect FILE.hgn [--json] [--no-hash] [--threads N]` | print the checkpoint's header, tensor table and derived stats |
| `verify` | `verify FILE.hgn [--threads N]` | check header/table/layout/checksums; print `PASS`/`FAIL` |
| `ppl` | `ppl FILE --ids IDS.bin …` (see flag table) | corpus perplexity / NLL scoring against the engine |
| `niah` | `niah FILE --manifest CASES.tsv [--out DIR] [--max-tok N] [--pin-trunk 0\|1]` | needle-in-a-haystack battery runner |
| `paired` | `paired A.bin B.bin [--label-a X] [--label-b Y] [--json]` | paired NLL comparison of two `.bin` score files |
| `--help` / `-h` / `help` | — | print help, exit **0** |

Dispatch rules (all **[RV]** unless noted):

- `argc < 2` (no args) → help to **stderr**, exit **2**.
- `--help`, `-h`, `help` → help to **stderr**, exit **0**. (The check order in
  `main` **[D]**: verbs are matched first by length; unmatched words are then
  compared against `--help`, `-h`, `help`; anything else is an unknown verb.)
- Any other unknown verb → `halogen-tools: unknown verb '<argv[1]>'` line
  followed by the help block, both to **stderr**, exit **2** (verified with
  `--version`, `--bogus`, `--Help` **[RV]**; `-V` and `INSPECT` are **[D]** —
  decompile-derived, not in the probe transcript — the verb match is
  case-sensitive and `--version` is **not** a version flag).
- Per-verb usage strings embed the verb name, **not** `argv[0]` (e.g.
  `usage: inspect FILE.hgn …`), so no argv[0] normalisation is needed for
  this binary's usage output.

## Verbatim help text

Printed to **stderr** on no-args (exit 2), on `--help`/`-h`/`help` (exit 0),
and after the unknown-verb line (exit 2). Byte-identical in all three paths
**[RV]**:

```
halogen-tools: the checkpoint tools
  inspect FILE.hgn [--json] [--no-hash] [--threads N]
  verify  FILE.hgn [--threads N]
  ppl     FILE --ids IDS.bin [--chunk 1024|8] [--out NLL.bin] [--vs OTHER [--vs-out NLL.bin]] [--pin-trunk 0|1]
          [--ref-out REF.bin [--topk 128]]   write the distribution at every position (top-K, tail, argmax)
          [--ref REF.bin [--per-pos P.bin]]  score against it: KL(ref || this), top-1 agreement, delta p(target)
          [--rows-out ROWS.bin --rows LO-HI] the full fp32 logits of positions [LO, HI)
          [--worst N]                         with --ref: the N positions the two files disagree on most, as ids
          [--json]                            one object on stdout (the prose goes to stderr)
          [--seq 262144]                      a longer corpus is scored as consecutive sequences of this length
  niah    FILE --manifest CASES.tsv [--out DIR] [--max-tok N] [--pin-trunk 0|1]
  paired  A.bin B.bin [--label-a X] [--label-b Y] [--json]
```

Per-verb usage lines (printed when a verb's required arguments are missing;
all **[RV]**, each a single line to stderr except as noted):

```
usage: inspect FILE.hgn [--json] [--no-hash] [--threads N]
usage: verify FILE.hgn [--threads N]
usage: ppl FILE --ids IDS.bin [--chunk 1024|8] [--out NLL.bin] [--vs OTHER [--vs-out NLL.bin]] [--pin-trunk 0|1] [--ref-out REF.bin [--topk K]] [--ref REF.bin [--per-pos P.bin]] [--rows-out ROWS.bin --rows LO-HI]
usage: niah FILE --manifest CASES.tsv [--out DIR] [--max-tok N] [--pin-trunk 0|1]
usage: paired A.bin B.bin [--label-a X] [--label-b Y] [--json]
```

(The `ppl` usage line is the one-line variant; the multi-line help block above
is a separate string. Both exist verbatim in the binary and match the
observed output **[RV]**.)

## Flag table — `inspect`

| Flag | Type | Default | Validation | Exact error wording | Provenance |
|---|---|---|---|---|---|
| `FILE` (positional) | string path | — | taken verbatim as the next argv word, even if it starts with `-` (`inspect --json` → `cannot open --json`) | `cannot open <file>` → exit 2 | [RV] |
| `--json` | flag | off | — | — | [RV]+[D] |
| `--no-hash` | flag | off | — | — | [RV]+[D] |
| `--threads N` | integer (`strtol`, base 10) | `8` | result clamped to minimum 1 (`max(1, N)`); non-numeric → 1; never an error | — | [D] (`threads-parser-FUN_01ef06d0`) |
| missing `FILE` | — | — | `argc == 2` | `usage: inspect FILE.hgn [--json] [--no-hash] [--threads N]` → exit 2 | [RV] |

Checkpoint validation (shared loader `FUN_01eec4a0`, **[D]**; the
`cannot open` case is **[RV]**):

- unreadable file → `cannot open <file>` (stderr, exit 2) **[RV]**
- file ≤ 103 bytes → `<file>: shorter than a header` (exit 2)
- first 4 bytes ≠ `HGN1` → `<file>: not an .hgn (the first bytes are not HGN1)` (exit 2)
- first 4 bytes = `GGUF` → `<file>: a GGUF. inspect and verify read the engine's own .hgn files; a GGUF is checked by loading it (ppl, niah, or the server), and convert writes it as an .hgn` (exit 2)
- header table size not consistent with file size → `<file>: the table does not fit in the file` / `<file>: the table does not fit in the file` variants (exit 2)

Success output (decompile-derived, not runtime-verified on this host — no
reference `.hgn` is available; see the `.hgn` caveat in
`docs/analysis/specs/flash_serve.md` review F3):

- prose: `  model_id %s, v%llu, %zu tensors, %.2f GB`, per-family table
  `  %-26s %-6s %6llu %14llu %8.2f GB %7.2f`, `  -> decode streams %.2f GB over %.2fB params = %.2f bpw effective`, `file_bytes %llu`, `pads %s`
- `--json`: one object `{"path":"%s","model_id":"%s","version":%llu,"n_tensors":%zu,"file_bytes":%llu,…,"families":[…`

## Flag table — `verify`

| Flag | Type | Default | Validation | Exact error wording | Provenance |
|---|---|---|---|---|---|
| `FILE` (positional) | string path | — | next argv word verbatim | — | [RV] |
| `--threads N` | integer (`strtol`) | `8` | clamped to minimum 1; never an error | — | [D] |
| missing `FILE` | — | — | `argc == 2` | `usage: verify FILE.hgn [--threads N]` → **stderr**, exit 2 | [RV] |

- Unreadable file → `FAIL cannot open <file>` on **stdout** (not stderr),
  exit **1** **[RV]** — the only verb whose failure output goes to stdout.
- Success (decompile-derived): `PASS %s: %zu tensors, %.2f GiB, v%llu, model_id %s; header, table, layout, payload sizes, checksums, codebooks and scales OK; geometry checked on %zu tensors (%zu with no rule)` → exit 0.
- Problem report (decompile-derived): `FAIL %s: %zu problem%s; the first: %s`
  followed by indented problem lines and `  ... and %zu more` → exit 1.
- `--json` is **not** accepted by `verify` (it is consumed as the threads
  scan's noise; unknown flags other than `--threads` are ignored by the
  shared flag scanner) **[D]**.

## Flag table — `ppl`

Grammar (one-line usage, **[RV]**):
`ppl FILE --ids IDS.bin [--chunk 1024|8] [--out NLL.bin] [--vs OTHER [--vs-out NLL.bin]] [--pin-trunk 0|1] [--ref-out REF.bin [--topk K]] [--ref REF.bin [--per-pos P.bin]] [--rows-out ROWS.bin --rows LO-HI]`

| Flag | Type | Default | Validation | Exact error wording | Provenance |
|---|---|---|---|---|---|
| `FILE` (positional) | string path | — | next argv word verbatim | — | [RV] |
| `FILE` starting with `-` | — | — | `main` rejects a dash-leading FILE before the verb runs (`ppl --ids y` → FILE is `--ids`) | one-line usage above → exit 2 | [D] (probed live in Phase 4) |
| `--ids IDS.bin` | string path | — | **required** (no text front end) | `ppl: --ids IDS.bin is required (a text front end is not built yet)` → exit 2 | [RV] |
| `--chunk N` | integer (`strtol`) | `1024` | `>= 1`; `<= --seq` | `need >= 2 ids and chunk >= 1`; `ppl: --chunk %d is larger than the sequence (--seq %d)` | [D] |
| `--out NLL.bin` | string path | empty | — | — | [D] |
| `--vs OTHER` | string path | — | paired comparison corpus | — | [D] |
| `--vs-out NLL.bin` | string path | empty | — | — | [D] |
| `--pin-trunk 0\|1` | integer (`strtol`) | `-1` (auto) | — | — | [D] |
| `--ref-out REF.bin` | string path | — | mutually exclusive with `--ref` | `ppl: --ref-out and --ref are one or the other` → exit 2 | [D] |
| `--topk K` | integer (`strtol`) | `128` | `1..4096` | `ppl: --topk in [1, 4096]` → exit 2 | [D] |
| `--ref REF.bin` | string path | — | mutually exclusive with `--ref-out` | — | [D] |
| `--per-pos P.bin` | string path | — | — | — | [D] |
| `--rows-out ROWS.bin` | string path | — | requires `--rows` | `ppl: --rows-out needs --rows LO-HI` → exit 2 | [D] |
| `--rows LO-HI` | two ints (`sscanf %d-%d`) | — | parse must yield 2 values, `LO >= 0`, `HI > LO` | `--rows LO-HI` (bare usage fragment) → exit 2 | [D] |
| `--worst N` | integer (`strtol`) | `0` | requires `--ref` | `ppl: --worst needs --ref` → exit 2 | [D] |
| `--json` | flag | off | with `--json`, stdout is redirected to stderr (`dup`/`dup2`) so prose goes to stderr and the JSON object to stdout | `ppl: cannot redirect stdout` → exit 2 | [D] |
| `--seq N` | integer (`strtol`) | `262144` | `2..1048576` | `ppl: --seq in [2, %d]` (renders `ppl: --seq in [2, 1048576]`) → exit 2 | [D] |
| missing `FILE` | — | — | `argc == 2` | one-line usage above → exit 2 | [RV] |

Runtime messages (decompile-derived): `ppl: the corpus is longer than one
sequence: scored as %d consecutive sequences of %d tokens (the state resets
at each; a reference dump carries the same number)`, `ppl: cannot make a
temporary file`, `ppl: reference %s: %llu positions, top-%u, sequences of
%llu`, `ppl: engine %s`. Actual scoring is GPU-gated.

## Flag table — `niah`

| Flag | Type | Default | Validation | Exact error wording | Provenance |
|---|---|---|---|---|---|
| `FILE` (positional) | string path | — | next argv word verbatim | — | [RV] |
| `FILE` starting with `-` | — | — | `main` rejects a dash-leading FILE before the verb runs (`niah --manifest y` → FILE is `--manifest`) | one-line usage above → exit 2 | [D] (probed live in Phase 4) |
| `--manifest CASES.tsv` | string path | — | **required** | `niah: --manifest CASES.tsv is required (the battery builder is not in the image yet)` → exit 2 | [RV] |
| `--out DIR` | string path | empty | — | — | [D] |
| `--max-tok N` | integer (`strtol`) | `0` | per-case cap; per-case `> 65536` rejected | `%s: %d tokens to generate; 0 to 65536` | [D] |
| `--pin-trunk 0\|1` | integer (`strtol`) | `-1` (auto) | — | — | [D] |
| missing `FILE` | — | — | `argc == 2` | one-line usage → exit 2 | [RV] |

Manifest/case validation (decompile-derived): `%s: short read`,
`%s: %zu ids + %d generated is past the sequence bound %d` (bound renders
`1048576`), `%s carries the id %d, outside the vocabulary of %d` (renders
`248320`). Runtime messages: `niah: engine %s`, `niah: %zu cases, Tmax %d,
max_tok %d, prefill_chunk %d`, per-case line
`NIAH %-28s T=%-6d prefill %7.2f s (%7.1f tok/s)  decode %2zu in %5.2f s (%5.2f tok/s)`.
Execution is GPU-gated.

## Flag table — `paired`

| Flag | Type | Default | Validation | Exact error wording | Provenance |
|---|---|---|---|---|---|
| `A.bin`, `B.bin` (positionals) | string paths | — | both required; non-flag argv words collected in order | fewer than two files → usage → exit 2 | [RV] |
| `--label-a X` | string | `A` | — | — | [D] (default from `main`: byte `0x41`) |
| `--label-b Y` | string | `B` | — | — | [D] (default byte `0x42`) |
| `--json` | flag | off | — | — | [RV]+[D] |

- Unreadable `A.bin` → `cannot open <A.bin>` (stderr), exit **2** **[RV]**
  (note: differs from `verify`, which prints `FAIL …` on stdout and exits 1).
- Output formats (decompile-derived): prose
  `paired %s-%s: mean %+.5f nats  t = %+.2f  95%% CI [%+.5f, %+.5f]  PPL ratio %.5f  %s better on %zu / worse on %zu / tied %zu`,
  band table header `  band             n   %s NLL   %s NLL   paired diff      t`,
  JSON `{"tool":"halogen-tools paired","warning":%s,…` containing
  `"paired":{"n":%zu,"mean_diff":%s,"se":%s,"t":%s,"ci95":[%s,%s],"ppl_ratio":%s,"better":%zu,"worse":%zu,"tied":%zu},`;
  empty inputs → `paired: no positions`.

## Environment variables

**No environment variables are read on the CLI argument-parsing paths**
**[D]**: `main`, the `inspect`/`verify` handlers, the shared checkpoint
loader and the `--threads` parser contain no `getenv` calls. The binary
imports `getenv` and embeds 274 unique `HALOGEN_[A-Z0-9_]*` names — 391
total string occurrences across 365 lines (counting method:
`grep -o 'HALOGEN_[A-Z0-9_]*' | sort -u` over
[`evidence/halogen-tools/binutils-strings.txt`](../evidence/halogen-tools/binutils-strings.txt))
used by the engine internals (GPU paths, e.g. `HALOGEN_ATTN_QS_BT2_WPE`
referenced in `main`'s engine-tuning tables). The launcher's `HALOGEN_*`
variables are consumed by `halogen-entrypoint.sh`, not by `halogen-tools`
itself.

## Exit codes

| Code | Meaning | Observed via |
|---|---|---|
| 0 | success | `--help`, `-h`, `help` **[RV]**; `inspect`/`verify`/`paired` success paths **[D]** |
| 1 | runtime failure | `verify` failure (`FAIL cannot open …`) **[RV]** |
| 2 | usage / validation | no args **[RV]**, unknown verb **[RV]**, per-verb missing arguments **[RV]**, unreadable `inspect`/`paired` inputs **[RV]**, `ppl`/`niah` missing required flags **[RV]**, checkpoint format errors **[D]**, `ppl` flag-range errors **[D]** |

## Observed behaviours (ground truth, GPU-less host)

1. No args → help (stderr), exit 2; `--help`/`-h`/`help` → help, exit 0;
   `--version`/`--bogus`/`--Help` → `halogen-tools: unknown verb '<x>'` +
   help, exit 2 **[RV]** (`-V` and `INSPECT` are **[D]** — decompile-derived,
   not in the probe transcript).
2. `inspect`/`verify`/`ppl`/`niah`/`paired` with no further args → their
   one-line usage, exit 2 (all five verified).
3. `inspect /nonexistent.hgn` → `cannot open /nonexistent.hgn` (stderr),
   exit 2; `inspect --json` → `cannot open --json` (positional taken
   verbatim).
4. `verify /nonexistent.hgn` → `FAIL cannot open /nonexistent.hgn`
   (**stdout**), exit 1.
5. `ppl /nonexistent` → `ppl: --ids IDS.bin is required (a text front end
   is not built yet)`, exit 2; `niah /nonexistent` → `niah: --manifest
   CASES.tsv is required (the battery builder is not in the image yet)`,
   exit 2.
6. `paired /nonexistent-a.bin` → usage (one file insufficient);
   `paired /nonexistent-a.bin /nonexistent-b.bin [--json]` → `cannot open
   /nonexistent-a.bin`, exit 2.
7. File-open checks precede flag-value checks (`inspect x.hgn --threads x`
   reports `cannot open x.hgn`, never a threads error) **[RV]**; `--threads`
   itself never errors (clamped) **[D]**.
8. `ppl`/`niah` beyond argument validation construct the engine (GPU-gated;
   not exercised on this host).

## Evidence

- [`evidence/halogen-tools/decompile/`](../evidence/halogen-tools/decompile/) — REA CLI + Ghidra 12.1.4 decompilation evidence: `main-FUN_01ecabf0.json` (verb dispatch + inline `ppl`/`niah`/`paired` parsers), `main-analyze-FUN_01ecabf0.json` (full `analyze_function` dossier), `inspect-FUN_01eed5d0.json`, `verify-FUN_01ef0b40.json`, `checkpoint-loader-FUN_01eec4a0.json` (shared `.hgn` validation), `threads-parser-FUN_01ef06d0.json` (`--threads` semantics).
- [`evidence/halogen-tools/native-exports.json`](../evidence/halogen-tools/native-exports.json), [`evidence/halogen-tools/elf-notes.json`](../evidence/halogen-tools/elf-notes.json) — provider-free REA records (identical evidence IDs to the MCP `inspect_native_exports` / `inspect_elf_notes` results).
- [`evidence/halogen-tools/overview-analyze.json`](../evidence/halogen-tools/overview-analyze.json) — REA `analyze` (`binary_overview`) record for the Ghidra analysis snapshot.
- [`evidence/halogen-tools/mcp-session.evidence.json`](../evidence/halogen-tools/mcp-session.evidence.json) — MCP session export (empty: see limitation below).
- [`evidence/halogen-tools/cli-probes.txt`](../evidence/halogen-tools/cli-probes.txt) — runtime probe transcript (32 invocations with separated stdout/stderr and exit codes).
- [`evidence/halogen-tools/binutils-elf-info.txt`](../evidence/halogen-tools/binutils-elf-info.txt), [`binutils-strings.txt`](../evidence/halogen-tools/binutils-strings.txt) — binutils ELF header/dynamic-section dump and full string inventory.

### Method and limitations

- **MCP session export empty.** The REA MCP server (rea-agents 6.8.0) opened
  the target and served provider-free tools, but its evidence ledger retained
  0 records on export (`export_evidence_bundle` → empty bundle) because the
  server environment lacks a `ps` binary (`spawn ps ENOENT` in its
  process-ownership check). The provider-free records were re-obtained via
  the **REA CLI** and carry **identical evidence IDs** to the MCP results
  (`ev_79a8d77a…` exports, `ev_b8ee37d2…` notes), proving equivalence. Deep
  analysis used the REA CLI with `GHIDRA_INSTALL_DIR` → nixpkgs **Ghidra
  12.1.4**, `JAVA_HOME` → nixpkgs **openjdk 21.0.12**, and procps `ps` on
  `PATH` (same environment gap as the flash_serve analysis; escalate by
  fixing the MCP server's launch environment).
- Static analysis never observes execution: facts marked **[D]** are
  corroborated by runtime probes where possible (all **[RV]** facts were
  captured directly) and by decompiled code otherwise.
- The binary is stripped; function names (`FUN_*`) are Ghidra-derived.
- The **success paths** of `inspect`/`verify`/`ppl`/`niah`/`paired` are not
  runtime-verifiable on this host: no reference `.hgn` checkpoint exists here
  and the subject-written stub is rejected by the reference (flash_serve
  review F3 — the `.hgn` format inference remains open; the loader
  decompilation here adds the header layout: `HGN1` magic + version + table
  fields + 64-byte `model_id`, 160-byte table entries).
- GPU-dependent behaviour (`ppl`/`niah` engine execution) is out of scope
  for this spec and gated per plan §5.
