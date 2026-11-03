# halogen-npu — per-binary contract spec

> **Status:** frozen contract for `cc-halogen-npu` (Phase 3/4).
> **Source binary:** `/nix/store/fn6y2w20kbgjd8da8pwxbbc3gvmiycpy-halogen-flash-server-0.18.21/bin/halogen-npu`
> **Provenance:** halogen-flash-server v0.18.21 (see `full-plan.md` Appendix B).
> **Evidence:** [`docs/analysis/evidence/halogen-npu/`](../evidence/halogen-npu/) (REA MCP session export + REA CLI/Ghidra 12.1.4 decompilation + binutils + runtime probes via a stub `libxrt_coreutil.so.2`).

Fact markers used throughout: **[RT]** = runtime-verified (probe transcript),
**[DC]** = decompile-derived (Ghidra 12.1.4), **[S]** = string-table observed.

## Artifact identity

| Fact | Value |
|---|---|
| Format | ELF64, little-endian, `ET_DYN` (PIE), x86-64 |
| Size | 3 066 545 bytes |
| SHA-256 | `2bdd4ffeb9c6cd23a5c37a3e656d192bda83a565f598b5ff015a75bc3b132786` |
| Entry point | `0x3cc30` (ELF vaddr; Ghidra image base `0x100000` → `0x13cc30`) |
| ELF ABI tag | GNU/Linux (`.note.ABI-tag`, `.note.gnu.property` present; no GNU build-id) |
| `DT_NEEDED` | `libxrt_coreutil.so.2`, `libstdc++.so.6`, `libm.so.6`, `libgcc_s.so.1`, `libc.so.6` |
| `DT_RUNPATH` | `…/gcc-16.2.0-lib/lib`, `…/gcc-16.2.0-libgcc/lib` |
| Symbol table | stripped of function symbols (`.symtab` holds only section symbols + 238 `UND` entries; `.dynsym` carries the imports) |
| Runtime dependency | AMD XRT (`libxrt_coreutil.so.2` — `xrt::device`, `xrt::bo`, `xrt::kernel`, `xrt::elf`, `xrt::hw_context`, `xrt::run`); the upstream derivation ignores it (`autoPatchelfIgnoreMissingDeps`) and it is **not** present on this host |

**Loadability caveat [RT]:** invoked directly on this host the binary exits
**127** with `error while loading shared libraries: libxrt_coreutil.so.2`.
All runtime probes below were executed with a purpose-built stub
`libxrt_coreutil.so.2` (no-op definitions for the 23 imported XRT/`_ITM`
symbols) supplied via `LD_LIBRARY_PATH`, so the ELF loads and the real CLI
paths execute. XRT-dependent behaviour beyond argument parsing (actual NPU
execution, engine I/O) is therefore **not** runtime-verified (see
Method and limitations).

## Entry points and modes

One entry point: `$out/bin/halogen-npu`. The launcher
(`libexec/halogen-entrypoint.sh`) invokes it as the NPU engine
(`_hg_npu_dir`/`NPU_PINS` contract, `share/halogen/tools/npu_api.py`).

There is no flag-style CLI: **`argv[1]` selects a subcommand** (verified by
decompilation of `main`/`FUN_00170740` and runtime probes). The program name
in every usage/error string is the **hard-coded literal `halogen-npu`**, not
`argv[0]` — verified by running a copy renamed `renamed-bin`, which still
prints `usage: halogen-npu …` **[RT]** (contract tests need no argv[0]
normalisation for this binary, unlike `flash_serve`).

| Subcommand | In main usage? | Selected when `argv[1]` is |
|---|---|---|
| `selftest` | yes | `selftest` (requires ≥1 further arg) |
| `replay` | yes | `replay` (requires ≥1 further arg) |
| `serve` | yes | `serve` (requires ≥2 further args) |
| `listen` | yes | `listen` (requires ≥3 further args, i.e. `argc >= 5`) **[DC]** |
| `probe` | yes | `probe` (requires ≥1 further arg) |
| `convert` | yes | `convert` (requires ≥2 further args) |
| `flux` | **hidden** | `flux` (requires ≥1 further arg) |
| `gemma` | **hidden** | `gemma` (requires ≥1 further arg) |
| `tts` | **hidden** | `tts` (requires ≥3 further args) |
| `asr` | **hidden** | `asr` (any further arg; `bogus` defaults to the `feat` grammar) |
| `vis` | **hidden** | `vis encode` only (`vis` alone or `vis decode` → main usage) |
| `cache` | **hidden** | `cache ecopy` only (`cache` alone or `cache bogus` → cache usage) |

Any other `argv[1]` (including `--help`, `--version`, `-h`, `-v`,
`--bogus-flag`, `--model`, `--repack`, `edit`, `enc`, `.vis`) prints the
**main usage** to stderr and exits **1** **[RT]**. There is **no `--help` or
`--version` handler** (no comparison against those strings exists in the
decompiled dispatcher **[DC]**).

## Flag table — top level

| Invocation | stdout | stderr | exit |
|---|---|---|---|
| (no args) | — | main usage (316-byte embedded string, `fwrite` in `main` **[DC]**) | 1 **[RT]** |
| `--help` / `--version` / `-h` / `-v` / `--bogus-flag` / `help` / `--model x` / `--repack` / `edit` / `enc` / `.vis` | — | main usage | 1 **[RT]** |

Main usage (verbatim, single line):

```
usage: halogen-npu selftest bf16|norm|head|resid ... | replay DIR [REPS] | serve DEVICES_DIR WEIGHTS [ORDER] [--model DEVICES_DIR WEIGHTS]... | listen DEVICES_DIR WEIGHTS HOST:PORT [ORDER] [--model DEVICES_DIR WEIGHTS]... | probe MODEL_DIR | convert MODEL_DIR DEVICES_DIR OUT [--name ID] [--source SRC] [--if-stale]
```

## Flag tables — subcommands

### `selftest KIND …`

Usage grammar (embedded string): `selftest bf16|norm|head|resid ...` (the
`...` hides further kinds). Decompile confirms the accepted kinds
`bf16`, `head`, `resid`, `hostx`, `gx` **[DC]**; `norm` appears in the usage
string but its comparison was not located in `main`'s inline strings
(escalated as an ambiguity).

| Form | Behaviour | exit |
|---|---|---|
| `selftest` (no kind) | main usage | 1 **[RT]** |
| `selftest bogus` | main usage | 1 **[RT]** |
| `selftest bf16 OUTFILE` (`argc == 4`) | opens `OUTFILE` with `fopen(..., "w")` and writes the bf16 test artefact; with the stub XRT the file is created (6 136 bytes) and nothing prints | 0 **[RT]** |
| `selftest bf16 UNWRITABLE` | `halogen-npu: cannot write <path>` **[DC]** | (not probed) |

### `replay DIR [REPS]`

| Form | stderr | exit |
|---|---|---|
| `replay` (no DIR) | main usage | 1 **[RT]** |
| `replay /nonexistent` | `halogen-npu: cannot open /nonexistent/replay.hnpm` | 1 **[RT]** |

### `serve DEVICES_DIR WEIGHTS [ORDER] [--model DEVICES_DIR WEIGHTS]...`

| Form | stderr | exit |
|---|---|---|
| `serve` (no args) | main usage | 1 **[RT]** |
| `serve /nonexistent /nonexistent` | `halogen-npu: cannot open /nonexistent/devices.hnpm` | 1 **[RT]** |

(`--model` pairs are parsed by the serve handler **[DC]**; the
`devices.hnpm` manifest check precedes any XRT work, so the error is
observable even with the stub library.)

### `listen DEVICES_DIR WEIGHTS HOST:PORT [ORDER] [--model DEVICES_DIR WEIGHTS]...`

Requires `argc >= 5` **[DC]**.

| Form | stderr | exit |
|---|---|---|
| `listen` (no args) | main usage | 1 **[RT]** |
| `listen /nonexistent /nonexistent 127.0.0.1:9999` | `halogen-npu: cannot open /nonexistent/devices.hnpm` | 1 **[RT]** |

### `probe MODEL_DIR` / `convert MODEL_DIR DEVICES_DIR OUT [--name ID] [--source SRC] [--if-stale] [--fold FILE --enc FILE --tokenizer FILE (a speech model)]`

Shared handler `FUN_0022f180` **[DC]**. Sub-usage (verbatim):

```
usage: halogen-npu probe MODEL_DIR | convert MODEL_DIR DEVICES_DIR OUT [--name ID] [--source SRC] [--if-stale] [--fold FILE --enc FILE --tokenizer FILE (a speech model)]
```

| Form | stderr | exit |
|---|---|---|
| `probe` (no DIR) | sub-usage above | 1 **[RT]** |
| `convert /nonexistent` (missing args) | sub-usage above | 1 **[RT]** |
| `probe /nonexistent` | `halogen-npu: /nonexistent holds no config.json (a Hugging Face checkpoint directory); the NPU serves fine-tunes of Qwen3.5-0.8B and 4B (decision; the 4B size also as an embedding model), Qwen3-Embedding-0.6B and 4B (embedding), Qwen3-Reranker-0.6B and 4B (rerank) and Qwen3Guard-Gen-0.6B (moderation)` | 2 **[RT]** |
| `probe DIR` with `config.json` = `{}` | `halogen-npu: DIR is a unknown checkpoint (model_type unknown); the NPU serves fine-tunes of …` (same tail) | 2 **[RT]** |
| `probe DIR` with `config.json` = `{"model_type":"qwen3_dense"}` | `halogen-npu: DIR is a qwen3_dense checkpoint (model_type qwen3_dense); the NPU serves fine-tunes of …` | 2 **[RT]** |
| `probe DIR` with `config.json` = `{"model_type":"qwen35_hybrid"}` | `halogen-npu: DIR is a qwen35_hybrid checkpoint (model_type qwen35_hybrid); the NPU serves fine-tunes of …` | 2 **[RT]** |
| `convert /nonexistent /nonexistent /nonexistent` | same `holds no config.json` error as `probe /nonexistent` | 2 **[RT]** |

Note: even a recognised `model_type` exits 2 when the checkpoint lacks the
weight artefacts the engine expects; the success output format requires a
real checkpoint and is not observable with the stub library (escalated).
Other checkpoint error strings **[S]**: `holds no tokenizer.json (the
model's own tokenizer)`, `is not valid JSON`, `: a guard's chat template on a
checkpoint without its LM head; …`, `: score.weight is not a [2 to 4096,
hidden] matrix of bf16 / f16 / f32; …`, `: a moderation model of the 4B size
is not served; …`, `: a sequence classifier of the 4B size is not served
yet; …`.

### `flux noise|taef2|dit|image|enc|edit …` (hidden)

Sub-usage (verbatim):

```
usage: halogen-npu flux noise SEED RES OUT | taef2 TAEF2 Z H W OUT_RGB [OUT_F32] | dit DEVDIR WEIGHTS RES STEP X CTX OUT [REPS] | image DEVDIR WEIGHTS RES SEED CTX OUT_RGB [OUT_LATENTS] [REPS] | enc TAEF2 RGB W H OUT_Z | edit DEVDIR WEIGHTS SIZE SEED CTX OUT_RGB [options]
```

| Form | stderr | exit |
|---|---|---|
| `flux` / `flux bogus` / `flux noise` | sub-usage above | 1 **[RT]** |

### `gemma DEVDIR WEIGHTS IDS OUT [REPS [BATCH [MASK|- [SOFT|- [IMG|- [AUD]]]]]]` (hidden)

| Form | stderr | exit |
|---|---|---|
| `gemma` / `gemma a` | its own usage (above, verbatim) | 1 **[RT]** |

### `tts SET MODEL REQS OUT [--tf] [--stream N] [--threads N] [--greedy] [--reps N]` (hidden)

| Form | stderr | exit |
|---|---|---|
| `tts` / `tts a b` / `tts a b c` | its own usage (above, verbatim) | 2 **[RT]** |

### `asr encode|decode|feat …` (hidden)

Sub-usages (verbatim, three separate strings):

```
usage: asr encode DEVDIR WEIGHTS IN.wav OUTDIR
usage: asr decode DEVDIR WEIGHTS EMB.f32 PRE.txt POST.txt MAXNEW OUTDIR [STOP,..]
usage: asr feat IN.wav OUTDIR [STEM.safetensors [PREFIX]]
```

| Form | stderr | exit |
|---|---|---|
| `asr` / `asr bogus` | the `feat` usage | 2 **[RT]** |
| `asr encode` | the `encode` usage | 2 **[RT]** |
| `asr decode` | the `decode` usage | 2 **[RT]** |
| `asr feat` | the `feat` usage | 2 **[RT]** |

### `vis encode DEVDIR WEIGHTS IN.rgb H W OUT.f32 [REPS]` (hidden)

| Form | stderr | exit |
|---|---|---|
| `vis` / `vis bogus` / `vis decode` | **main** usage | 1 **[RT]** |
| `vis encode` / `vis encode x` | `usage: vis encode DEVDIR WEIGHTS IN.rgb H W OUT.f32 [REPS]` | 2 **[RT]** |

### `cache ecopy IN OUT CAP` (hidden)

| Form | stderr | exit |
|---|---|---|
| `cache` / `cache bogus` / `cache ecopy` | `usage: halogen-npu cache ecopy IN OUT CAP` | 2 **[RT]** |

## Environment variables

Read on the `main` path **[DC]** (`FUN_0013c2f0`):

| Variable | Default | Behaviour |
|---|---|---|
| `HALOGEN_NPU_ASR_HOST` | unset | if value starts with `1`, `main` sets (no-clobber, `setenv(...,0)`) `HALOGEN_NPU_HOST_THREADS=8`, `HALOGEN_NPU_WAIT_SPIN=0`, `HALOGEN_NPU_HOST_PREWAKE=0` before dispatch |

The binary's `.rodata` contains **92 unique `HALOGEN_NPU_[A-Z0-9_]*`
variable names** (counting method: `grep -o 'HALOGEN_[A-Z0-9_]*' | sort -u`
over [`evidence/halogen-npu/binutils-strings.txt`](../evidence/halogen-npu/binutils-strings.txt)).
The full inventory **[S]** (per-subcommand readers not individually located;
`cc-halogen-npu` must reproduce the four `main`-path variables above and the
names below as its env surface):

`HALOGEN_NPU_ASR_HOST`, `HALOGEN_NPU_ASR_LEDGER`,
`HALOGEN_NPU_ASR_PIECE_S`, `HALOGEN_NPU_ASR_STRETCH_S`,
`HALOGEN_NPU_ASR_THREADS`, `HALOGEN_NPU_ASR_TOKEN`, `HALOGEN_NPU_BATCH`,
`HALOGEN_NPU_CACHE_ENTRIES`, `HALOGEN_NPU_CONTEXTS`,
`HALOGEN_NPU_DEV_SERIAL`, `HALOGEN_NPU_ENGINE_REUSE`,
`HALOGEN_NPU_FAULT_AT`, `HALOGEN_NPU_FAULT_EXIT`,
`HALOGEN_NPU_FLOOR_PROBE`, `HALOGEN_NPU_FLUX_HOST`,
`HALOGEN_NPU_FLUX_LEDGER`, `HALOGEN_NPU_FLUX_NO_TE`,
`HALOGEN_NPU_FLUX_PSYNC`, `HALOGEN_NPU_FLUX_REUSE`,
`HALOGEN_NPU_FLUX_SETS`, `HALOGEN_NPU_FLUX_THREADS`,
`HALOGEN_NPU_FLUX_WAKE`, `HALOGEN_NPU_GEMMA_FUSEPACK`,
`HALOGEN_NPU_GEMMA_G512`, `HALOGEN_NPU_GEMMA_GELUTAB`,
`HALOGEN_NPU_GEMMA_PROBE`, `HALOGEN_NPU_GEMMA_REUSE`,
`HALOGEN_NPU_GEMMA_S2`, `HALOGEN_NPU_GEMMA_S2_ROWS`,
`HALOGEN_NPU_GEMMA_SMX`, `HALOGEN_NPU_GEMMA_THREADS`,
`HALOGEN_NPU_GEMMA_VNORM`, `HALOGEN_NPU_GEMMA_VPACK`,
`HALOGEN_NPU_GEMMA_VQ4`, `HALOGEN_NPU_GEMMA_WINPACK`,
`HALOGEN_NPU_GEN_DBG`, `HALOGEN_NPU_GEN_DIGEST`,
`HALOGEN_NPU_GEN_GUESTS`, `HALOGEN_NPU_GEN_SHARE`,
`HALOGEN_NPU_GS_SESSIONS`, `HALOGEN_NPU_GUARD_ROOM`,
`HALOGEN_NPU_HEAP_MIB`, `HALOGEN_NPU_HOST_AWAKE`,
`HALOGEN_NPU_HOST_FLOOR_GIB`, `HALOGEN_NPU_HOST_NT`,
`HALOGEN_NPU_HOST_PREWAKE`, `HALOGEN_NPU_HOST_PSYNC`,
`HALOGEN_NPU_HOST_SPIN`, `HALOGEN_NPU_HOST_THREADS`,
`HALOGEN_NPU_JOBLOG`, `HALOGEN_NPU_LFM3_HELPER`,
`HALOGEN_NPU_LFM3_PROF`, `HALOGEN_NPU_PFX`, `HALOGEN_NPU_QUEUE`,
`HALOGEN_NPU_REBIND`, `HALOGEN_NPU_SHORT_PATH`,
`HALOGEN_NPU_SPEAK_LEAD_MS`, `HALOGEN_NPU_SPEAK_RT`,
`HALOGEN_NPU_SPEAK_WAIT_MS`, `HALOGEN_NPU_SPLICE`,
`HALOGEN_NPU_SPLICE_MAX`, `HALOGEN_NPU_SPLICE_TOKENS`,
`HALOGEN_NPU_SYNC_NARROW`, `HALOGEN_NPU_SYNC_WHOLE`,
`HALOGEN_NPU_TTS_HEAD_THREADS`, `HALOGEN_NPU_TTS_PREFILL`,
`HALOGEN_NPU_TTS_ROOM`, `HALOGEN_NPU_TTS_THREADS`,
`HALOGEN_NPU_VIS_ATTN6`, `HALOGEN_NPU_VIS_CHECK`,
`HALOGEN_NPU_VIS_DUMP`, `HALOGEN_NPU_VIS_FUSE`,
`HALOGEN_NPU_VIS_GAP_US`, `HALOGEN_NPU_VIS_GELU`,
`HALOGEN_NPU_VIS_HASH`, `HALOGEN_NPU_VIS_HOST_ATTN`,
`HALOGEN_NPU_VIS_KEEP`, `HALOGEN_NPU_VIS_PACK`,
`HALOGEN_NPU_VIS_POOL`, `HALOGEN_NPU_VIS_REUSE`,
`HALOGEN_NPU_VIS_STAGE`, `HALOGEN_NPU_VIS_THREADS`,
`HALOGEN_NPU_WAIT_MS`, `HALOGEN_NPU_WAIT_PREDICT`,
`HALOGEN_NPU_WAIT_SPIN`.

## Exit codes

| Code | Meaning | Observed via |
|---|---|---|
| 0 | success | `selftest bf16 OUTFILE` **[RT]**; serve/listen success is GPU/XRT-gated |
| 1 | usage error (main usage), missing `devices.hnpm` / `replay.hnpm`, sub-usage for `flux`/`gemma`/`probe`/`selftest`-missing-kind | runtime probes **[RT]** |
| 2 | sub-usage for `cache`/`tts`/`asr`/`vis encode`, invalid/unusable checkpoint in `probe`/`convert` | runtime probes **[RT]** |
| 127 | dynamic-loader failure (missing `libxrt_coreutil.so.2`) — the state of this host without the stub | runtime probe **[RT]** |

Note the asymmetry (contract-relevant): the **top-level** usage error exits
**1**, while **subcommand-internal** usage errors exit **2** for
`cache`/`tts`/`asr`/`vis encode` but **1** for `flux`/`gemma`/`probe`
(missing args)/`selftest` (missing kind) **[RT]**.

## Verbatim help text

`halogen-npu` has **no `--help` handler**: any unrecognised `argv[1]`
(including `--help` and `--version`) prints the main usage to **stderr** and
exits **1**. The program name is the hard-coded literal `halogen-npu`
(not `argv[0]`) **[RT]**. All usage strings (main + per-subcommand) are
listed verbatim in the flag tables above and captured in
[`evidence/halogen-npu/binutils-strings.txt`](../evidence/halogen-npu/binutils-strings.txt)
(lines 4419, 4519, 4733, 4772, 4812, 4863, 4897, 4912, 4914, 4916).

## Observed behaviours (ground truth, stub-XRT host)

1. Direct invocation (no stub) → `error while loading shared libraries:
   libxrt_coreutil.so.2`, exit 127 **[RT]**.
2. No args → main usage, exit 1; stdout empty **[RT]**.
3. `--help` / `--version` / `-h` / `-v` / `--bogus-flag` / `help` → main
   usage, exit 1 (no help/version output exists) **[RT]**.
4. Renamed binary (`renamed-bin`) still prints `halogen-npu` as the program
   name → the name is embedded, not derived from `argv[0]` **[RT]**.
5. `serve /nonexistent /nonexistent` and `listen …` → `cannot open
   <dir>/devices.hnpm`, exit 1 (manifest check precedes XRT) **[RT]**.
6. `replay /nonexistent` → `cannot open <dir>/replay.hnpm`, exit 1 **[RT]**.
7. `probe /nonexistent` → `holds no config.json …` error, exit 2; a
   `config.json` with `{}` → `is a unknown checkpoint (model_type unknown)`;
   `model_type` `qwen3_dense` / `qwen35_hybrid` → recognised-family error,
   exit 2 **[RT]**.
8. `selftest bf16 FILE` → writes the test artefact to FILE (6 136 bytes with
   the stub), exit 0, no stdout/stderr **[RT]**.
9. `main` reads `HALOGEN_NPU_ASR_HOST` and, when `1`, seeds
   `HALOGEN_NPU_HOST_THREADS=8`, `HALOGEN_NPU_WAIT_SPIN=0`,
   `HALOGEN_NPU_HOST_PREWAKE=0` (no-clobber) **[DC]**.

## Evidence

- [`evidence/halogen-npu/mcp-session.evidence.json`](../evidence/halogen-npu/mcp-session.evidence.json) — REA MCP session export (`open_binary`, `inspect_native_exports` + its `inspect_analysis_view`, `inspect_elf_notes`).
- [`evidence/halogen-npu/decompile/`](../evidence/halogen-npu/decompile/) — REA CLI + Ghidra 12.1.4 decompilation: `entry-0x13cc30.c`, `main-FUN_0013c2f0.c`, `cli-dispatch-FUN_00170740.c`, `probe-convert-FUN_0022f180.c`, plus `manifest.json` (full export: 3 720 functions, analysis-profile digest `ae80ece8…`).
- [`evidence/halogen-npu/cli-probes.txt`](../evidence/halogen-npu/cli-probes.txt) — full runtime probe transcript (46 invocations via the stub library, with stdout/stderr/exit codes).
- [`evidence/halogen-npu/binutils-elf-info.txt`](../evidence/halogen-npu/binutils-elf-info.txt), [`binutils-strings.txt`](../evidence/halogen-npu/binutils-strings.txt) — binutils ELF info and full string inventory.

### Method and limitations

- The REA MCP server (rea-agents 6.8.0) has **no deep-analysis provider
  configured** (`GHIDRA_INSTALL_DIR` unset in its environment):
  `open_binary` and provider-free tools work, but `analyze_function` returns
  `provider_unavailable`. Deep analysis ran through the **REA CLI** with
  `GHIDRA_INSTALL_DIR` → nixpkgs **Ghidra 12.1.4**. Two further environment
  gaps were worked around and should be escalated: the CLI requires a
  **procps-compatible `ps` on PATH** (linked from the nix store), and
  Ghidra requires a **JDK 21+** (`JAVA_HOME` set to the nix-store
  `openjdk-21.0.12.1+1`).
- **The binary cannot execute on this host without AMD XRT**: the upstream
  derivation ignores the missing `libxrt_coreutil.so.2`, so all runtime
  probes used a stub shared object providing its 23 imported symbols. The
  stub exercises the real argument-parsing/validation paths (which precede
  XRT use), but XRT-dependent behaviour (device enumeration, engine I/O,
  inference output, `serve`/`listen` success paths) remains unverified —
  GPU/XRT-gated per plan §5.
- Static analysis never observes execution: facts marked **[DC]** are
  corroborated by runtime probes where possible.
- The binary is stripped; `FUN_*` names are Ghidra-derived.
- **Escalated ambiguities:** (a) the `selftest` usage advertises `norm` but
  no `norm` comparison was located in `main`'s inline strings (the decompile
  confirms `bf16|head|resid|hostx|gx`); (b) the `probe` success output
  format for a fully valid checkpoint is not observable without real weights;
  (c) per-subcommand readers of the 88 non-`main` `HALOGEN_NPU_*` variables
  are inventoried by name only, not individually located.
