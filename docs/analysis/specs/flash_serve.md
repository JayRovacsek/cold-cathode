# flash_serve — per-binary contract spec

> **Status:** frozen contract for `cc-flash-serve` (Phase 3/4).
> **Source binary:** `/nix/store/fn6y2w20kbgjd8da8pwxbbc3gvmiycpy-halogen-flash-server-0.18.21/bin/flash_serve`
> **Provenance:** halogen-flash-server v0.18.21 (see `full-plan.md` Appendix B).
> **Evidence:** [`docs/analysis/evidence/flash_serve/`](../evidence/flash_serve/) (REA MCP session export + REA CLI/Ghidra 12.1.4 decompilation + binutils + runtime probes).

## Artifact identity

| Fact | Value |
|---|---|
| Format | ELF64, little-endian, `ET_DYN` (PIE), x86-64 |
| Size | 33 732 361 bytes |
| SHA-256 | `7c92fe3f8ac544c4fb690a69e9ce359fa7b8d453a8eab8e9a1289c6f2309c012` |
| Entry point | `0x1dde5d0` |
| ELF ABI tag | GNU/Linux, kernel ≥ 3.2.0 (no GNU build-id) |
| `DT_NEEDED` | `libhipblaslt.so.1`, `libamdhip64.so.7`, `libstdc++.so.6`, `libm.so.6`, `libgcc_s.so.1`, `libc.so.6` |
| `DT_RUNPATH` | `…/hipblaslt-7.2.4/lib`, `…/clr-7.2.4/lib`, `…/gcc-16.2.0-lib/lib`, `…/gcc-16.2.0-libgcc/lib` |
| Symbol table | stripped (`.symtab` empty; `.dynsym` carries weak libstdc++ symbols) |
| Embedded GPU code | 9 AMDGPU code objects (see `evidence/flash_serve/mcp-session.evidence.json`) |

## Entry points and modes

One entry point: `$out/bin/flash_serve`. It is invoked directly or by the
launcher (`libexec/halogen-entrypoint.sh`, which passes `--ck/--port/--bind/
--slots/--ctx/--max-tok/--kv-pool` from the `HALOGEN_*` environment).

There are no subcommand words; the **mode is selected by `argv[1]`** (verified
by decompilation of the mode dispatcher `FUN_01f23680` and runtime probes):

| Mode | Selected when `argv[1]` is | Purpose |
|---|---|---|
| **serve** (default) | anything else (incl. nothing, `--help`, unknown flags) | run the HTTP inference engine server |
| **repack** | `--repack` (or `--repack-hash`) | convert a GGUF checkpoint to `.hgn` (or hash it) |
| **resident-gib query** | `--resident-gib` | print the checkpoint's resident-weight size and feature flags, then exit |

Mode precedence details (all observed):

- `--ck x --repack y --out z` → **serve** mode (`--repack` unknown in serve grammar → usage, exit 2).
- `--version --repack x --out y` → **serve** usage (only `argv[1]` selects the mode).
- `--out y --repack x` → **serve** usage (`--out` unknown in serve grammar).
- `--repack x --out y --port 1` → **repack** usage (`--port` unknown in repack grammar).
- `--repack=x` / `--resident-gib=/nonexistent` → **serve** usage (`--flag=value` syntax is not supported anywhere; only `--flag value`).
- In resident-gib mode the rest of `argv` is ignored (`--resident-gib /nonexistent --help` → exit 1, no usage).

## Flag table — serve mode

Usage grammar: `[--ck FILE] [--port P] [--bind ADDR] [--slots B] [--ctx N] [--max-tok N] [--kv-pool POSITIONS]`
(all flags optional; every flag requires a separate value argument — `--flag=value` is an unknown flag).

| Flag | Type | Default | Validation | Exact error wording |
|---|---|---|---|---|
| `--ck FILE` | string path | `/models/qwen38-flash-next-w4b.hgn` | file existence is checked only after the GPU check (not observable on a GPU-less host; `--ck /nonexistent` there yields the ROCm error instead) | — |
| `--port P` | integer (`strtol`, base 10) | `8730` | none at parse time (observed: `abc`, `-1`, `0`, `99999`, `70000` all accepted and reach the GPU check) | — |
| `--bind ADDR` | string, IPv4 dotted-quad | `127.0.0.1` | `inet_pton(AF_INET)` must succeed; checked **after** the GPU check | `flash_serve: bad --bind address '%s'` → exit 1 |
| `--slots B` | integer (`strtol`) | `4`, or `$HALOGEN_KV_SLOTS` if set (flag overrides env) | `1..64` | format string `flash_serve: --slots 1..%d, --ctx >= 16, --max-tok >= 1` → exit 2 (the upper bound is runtime-rendered, not a literal: the decompiled `main` passes the constant `64` (`0x40`), so it renders `flash_serve: --slots 1..64, --ctx >= 16, --max-tok >= 1` under the default configuration) |
| `--ctx N` | integer (`strtol`) | `32768` | `>= 16`; also `<=` the largest supported context (native `262144`; with `HALOGEN_ROPE_YARN=f>1`: `min(f·262144, 1048576)`) | range: format string `flash_serve: --slots 1..%d, --ctx >= 16, --max-tok >= 1` → exit 2 (renders `1..64` under the default configuration, see the `--slots` row). too-large: `flash_serve: --ctx %d exceeds %lld, the largest context this configuration supports (native %d%s). A context past the native %d needs HALOGEN_ROPE_YARN=<factor> (4 for %d, 2 for %d): Qwen's static YaRN, which changes the model's numerics at every position.` → exit 2 |
| `--max-tok N` | integer (`strtol`) | `8192` | `>= 1`; silently clamped to `min(max-tok, ctx)` | format string `flash_serve: --slots 1..%d, --ctx >= 16, --max-tok >= 1` → exit 2 (renders `1..64` under the default configuration, see the `--slots` row) |
| `--kv-pool POSITIONS` | 64-bit integer (`strtoll`) | `0` = auto (becomes `ctx`, rounded up to a multiple of 256), or `$HALOGEN_KV_POOL_POSITIONS` if set (flag overrides env) | `>= ctx`, a multiple of 256 (otherwise rounded up with an informational message), `<= 16777216` | `flash_serve: --kv-pool %lld: at least --ctx %d, a multiple of 256, at most %lld` → exit 2. informational (valid but unaligned): `flash_serve: the KV pool is allocated in units of 256 positions: %lld rounded up to %lld` |
| `--help` | — | — | **not recognised** in serve mode (behaves as an unknown flag) | serve usage → exit 2 |
| `--version` | — | — | **not recognised** anywhere (no version output exists in the binary) | serve usage → exit 2 |

Validation order in serve mode (from decompiled `main` + probes):
**parse argv → GPU check → slots/ctx/max-tok range → kv-pool rule → bind `inet_pton` → socket setup.**
On a GPU-less host every value-validation error is therefore masked by the ROCm error (observed for `--bind xyz`, `--slots 0`, `--ctx 0`, `--kv-pool 100`).

## Flag table — repack mode

Usage grammar (three lines, covering both the `--repack` and `--repack-hash` forms):

```
usage: %s --repack IN.gguf --out OUT.hgn [--no-table] [--head HEAD.hgn|--no-head] [--threads N]
         (the n-gram table is written by default; a file without it cannot serve)
       %s --repack-hash IN.gguf [--with-table] [--head HEAD.hgn|--no-head] [--threads N]
```

| Flag | Type | Default | Notes / exact errors |
|---|---|---|---|
| `--repack IN` | string path (value = next argv word) | — | requires `--out`; missing → `%s: --repack needs IN.gguf and --out OUT.hgn` (argv[0]-prefixed) → exit 2 |
| `--repack-hash IN` | string path (value = next argv word) | — | `--out` not required; missing IN → same `--repack needs IN.gguf and --out OUT.hgn` error → exit 2 |
| `--out OUT` | string path | — | required for `--repack`; unknown in serve grammar |
| `--no-table` | flag | table written by default for `--repack` | — |
| `--with-table` | flag | table **not** written for `--repack-hash` unless given | accepted alongside `--no-table` (last-wins via grammar; no conflict error observed) |
| `--head HEAD.hgn` | string path | `$HALOGEN_MTP_HEAD`, else `/models/qwen38-flash-next-mtp.hgn` | — |
| `--no-head` | flag | — | clears the head |
| `--threads N` | integer (`strtol`) | — | no validation observed |
| `--quiet` | flag | off | suppresses progress output; unknown in serve grammar |
| `--help` | — | — | not recognised (unknown flag in repack grammar) → repack usage → exit 2 |

Input validation (observed, happens before any GPU work):

- non-GGUF input → `gguf: %s is not a GGUF` → exit 2
- GGUF magic but short file → `gguf: %s: truncated header` → exit 1

## Flag table — resident-gib query mode

`--resident-gib FILE` (value = next argv word; rest of argv ignored).

- Readable checkpoint: prints one stdout line in format `%.1f %d %d %d %.1f\n`
  (resident-weight GiB, kept-flag, has-table-flag, gather-flag, extra field —
  field semantics partially inferred; the launcher parses it with the regex
  `^[0-9]+\.[0-9] [01]( [01]){0,2}( [0-9]+\.[0-9])?$`) → exit 0.
- Unreadable/invalid checkpoint (e.g. `/nonexistent`, or a stub GGUF): **no output**, exit 1.
- No GPU check, no usage output.

> Provenance note (review C2): the literal string `--resident-gib` is not
> visible in `binutils-strings.txt` nor in the two committed decompiled
> functions (likely stored in a form binutils `strings` does not extract,
> e.g. a length-prefixed string table). Its existence is substantiated by
> the probe transcript (`--resident-gib /nonexistent` → exit 1 with no usage
> output, distinct from the unknown-flag path, which prints usage and exits
> 2) and by the `%.1f %d %d %d %.1f` output format in
> `mode-dispatcher-FUN_01f23680.json`.

> `.hgn` format caveat (review F3): the checkpoint layout inferred by the
> Phase 4 port (`HGN1` magic + version + f64 GiB + flag bytes, derived from
> the `HGN1` string in `binutils-strings.txt`) is **falsified**: probing the
> reference with a subject-written `.hgn` (`--resident-gib sub.hgn`) yields
> exit 1 with no output — the reference rejects it — while the subject reads
> it back and prints `0.0 1 1 0 0.0` with exit 0. The reference's actual
> `.hgn` format differs from the inference; decompiling the resident-gib
> handler, or obtaining a reference-produced checkpoint, is required before
> the deferred success case can be un-deferred (matrix reason
> `hgn-format-not-reverse-engineered`).

## Environment variables

Read on the CLI paths (main / mode dispatcher / GPU selector). The binary
contains 353 unique `HALOGEN_[A-Z0-9_]*` variable names — 545 total string
occurrences (counting method, review C1: `grep -o 'HALOGEN_[A-Z0-9_]*' | sort -u`
over the full inventory in
[`evidence/flash_serve/binutils-strings.txt`](../evidence/flash_serve/binutils-strings.txt));
the table below covers the variables read by the argument-parsing and
startup paths that `cc-flash-serve` must reproduce.

| Variable | Default | Used by | Validation / exact error wording |
|---|---|---|---|
| `HALOGEN_KV_SLOTS` | `4` | serve (`--slots` default) | `strtol`; range as `--slots` |
| `HALOGEN_KV_POOL_POSITIONS` | `0` (auto) | serve (`--kv-pool` default) | `strtoll`; range as `--kv-pool` |
| `HALOGEN_KV_POOL` | `1` (any value ≠ `0`) | serve | `0` selects the non-pooled KV layout (per-slot pool); affects kv-pool validation path |
| `HALOGEN_IMAGE_VERSION` | `unknown` | serve banner | max 31 chars; leading-space trimmed |
| `HALOGEN_ROPE_YARN` | unset (= 1.0) | serve ctx check | `>1` extends max ctx to `min(f·262144, 1048576)`; setting it with `ctx <= 262144` warns: `flash_serve: WARNING HALOGEN_ROPE_YARN=%g with --ctx %d at or under the native %d: static YaRN rescales every position, and Qwen advises setting it only when the context needs it.` |
| `HALOGEN_GPU_DEVICE` | unset | GPU selector | device selection; then `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, `CUDA_VISIBLE_DEVICES`, `GPU_DEVICE_ORDINAL` are consulted in that order |
| `HALOGEN_MTP_HEAD` | `/models/qwen38-flash-next-mtp.hgn` | repack (`--head` default) | — |
| `HALOGEN_PREFILL_KEEP_TRUNK` | `0` | resident-gib query | `1` fills the second output field from the checkpoint |
| `HALOGEN_PLD` | — | serve | `N,K` (`sscanf %d,%d`); invalid → `flash_serve: HALOGEN_PLD=N,K (0 = off)` → exit 2 |
| `HALOGEN_PLD_ADAPT` | — | serve | `%d,%f,%d`; invalid → `flash_serve: HALOGEN_PLD_ADAPT=W,floor,R` → exit 2 |
| `HALOGEN_PLD_LOOKBACK` | — | serve | `strtol` |
| `HALOGEN_PLD_AGREE` | — | serve | `strtol` |
| `HALOGEN_SPEC_ADAPT` | — | serve | `%d,%f,%d`; invalid → `flash_serve: HALOGEN_SPEC_ADAPT=W,floor,R` → exit 2 |
| `HALOGEN_MTP_DEPTH` | `3` | serve | `strtol`; invalid → `flash_serve: HALOGEN_MTP_DEPTH=%s needs 1 <= D and HALOGEN_SPEC_N >= D + 1 (is %d)` → exit 2 |
| `HALOGEN_MTP_DEPTH_HI` | — | serve | `strtol`; invalid → `flash_serve: HALOGEN_MTP_DEPTH_HI=%s needs 0 <= D <= 8` → exit 2 |
| `HALOGEN_MTP_GATE_P` | `0.9` | serve | `strtof`, range `[0,1]`; invalid → `flash_serve: HALOGEN_MTP_GATE_P=%s needs a value in [0, 1] (0 = off)` → exit 2 |
| `HALOGEN_SAMPLED_DEPTH` | `7` | serve | `strtol`; invalid → `flash_serve: HALOGEN_SAMPLED_DEPTH=%s needs 1 <= D and HALOGEN_SPEC_N >= D + 1 (is %d)` → exit 2 |
| `HALOGEN_ADMIT_CHUNK` | — | serve | `strtol` |
| `HALOGEN_ADMIT_TICKS` | `1` | serve | `strtol`, minimum 1 |
| `HALOGEN_ADMIT_ORDER` | `1` (lifo) | serve | `fifo` → 0, anything else → 1 |
| `HALOGEN_ADMIT_AGE_S` | — | serve | `strtod`; `<= 0` → 0 |
| `HALOGEN_ADMIT_SHORT` | `0` | serve | `strtol`, minimum 0 |
| `HALOGEN_ADMIT_SPLICE_MS` | `0` | serve | `strtol`, minimum 0 |
| `HALOGEN_ADMIT_SPLICE_FOR` | — | serve | `h` or `all`; invalid → `flash_serve: HALOGEN_ADMIT_SPLICE_FOR=%s: expected h or all` → exit 2 |
| `HALOGEN_ADMIT_NEST` | — | serve | `0` or `1`; invalid → `flash_serve: HALOGEN_ADMIT_NEST=%s: expected 0 or 1` → exit 2 |
| `HALOGEN_ADMIT_NEST_MAX` | `0` | serve | `strtol`, minimum 0 |
| `HALOGEN_PREFILL_CANCEL` | — | serve | `0` disables prefill cancellation |
| `HALOGEN_LOCK_ANON` | — | serve | `1` locks anonymous memory; prints `flash_serve: locked %.2f GiB of anonymous memory in %d range(s) (HALOGEN_LOCK_ANON=1)%s%s` |
| `HALOGEN_SAMPLED_GATE` | — | serve | `t1[,t2]` values in `[0,1]`; invalid → `flash_serve: HALOGEN_SAMPLED_GATE=t1[,t2] needs values in [0, 1]` → exit 2 |

Note: `--port`, `--bind`, `--ctx`, `--max-tok` and `--ck` defaults are
**hard-coded** in the binary (8730, 127.0.0.1, 32768, 8192,
`/models/qwen38-flash-next-w4b.hgn`); the launcher's `HALOGEN_PORT`,
`HALOGEN_BIND`, `HALOGEN_CTX`, `HALOGEN_MAX_TOK`, `HALOGEN_CHECKPOINT`
variables are translated into flags by `halogen-entrypoint.sh`, not read by
`flash_serve` itself.

## Exit codes

| Code | Meaning | Observed via |
|---|---|---|
| 0 | success | resident-gib query success; repack success; serve exits 0 on clean shutdown (GPU-gated) |
| 1 | runtime failure | no GPU (`--help`-less serve on GPU-less host), bad `--bind` address, unreadable checkpoint in resident-gib mode, truncated GGUF, `hipSetDevice` failure |
| 2 | usage / validation error | unknown flag, missing flag value, slots/ctx/max-tok range, kv-pool rule, env-var format errors, non-GGUF repack input, `--repack` without `--out` |

## Verbatim help text

`flash_serve` has **no `--help` handler**: any unrecognised first argument
(including `--help` and `--version`) prints the mode's usage to **stderr** and
exits **2**. The usage lines embed `argv[0]` verbatim (contract tests must
normalise argv[0] per §5 of the plan).

Serve mode (single line):

```
usage: <argv0> [--ck FILE] [--port P] [--bind ADDR] [--slots B] [--ctx N] [--max-tok N] [--kv-pool POSITIONS]
```

Repack mode (three lines):

```
usage: <argv0> --repack IN.gguf --out OUT.hgn [--no-table] [--head HEAD.hgn|--no-head] [--threads N]
         (the n-gram table is written by default; a file without it cannot serve)
       <argv0> --repack-hash IN.gguf [--with-table] [--head HEAD.hgn|--no-head] [--threads N]
```

## Observed behaviours (ground truth, GPU-less host)

1. `flash_serve` (no args) → stderr `halogen: ROCm reports no GPU in this container, so there is nothing to run on. Pass the GPU in (--device /dev/kfd --device /dev/dri), and if HIP_VISIBLE_DEVICES or ROCR_VISIBLE_DEVICES is set in the environment, unset it.`, exit 1.
2. `--help` / `--version` / `--bogus-flag` → serve usage, exit 2 (stdout empty).
3. `--port abc`, `--port 99999`, `--slots x`, `--ctx abc`, `--max-tok -5`, `--kv-pool 100`, `--bind xyz` → all reach the GPU check (exit 1) — value validation is gated behind it.
4. `--repack /nonexistent` → `%s: --repack needs IN.gguf and --out OUT.hgn` (exit 2); `--repack /nonexistent --out /tmp/x` → `gguf: /nonexistent is not a GGUF` (exit 2); a stub GGUF → `gguf: …: truncated header` (exit 1).
5. `--repack-hash /nonexistent` → `gguf: /nonexistent is not a GGUF` (exit 2) — no `--out` needed.
6. `--resident-gib /nonexistent` → exit 1, no output.
7. Serve startup (GPU present, from decompilation): `socket(AF_INET, SOCK_STREAM)`, `setsockopt(SO_REUSEADDR)`, `bind()` to `--bind:--port`, `listen(fd, 64)`, then the HTTP protocol served by the engine (HTTP route surface is specified separately in `docs/analysis/specs/api.md`).
8. GPU check (from decompilation of `FUN_01f700f0`): `hipGetDeviceCount`; device selection consults `HALOGEN_GPU_DEVICE`, `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, `CUDA_VISIBLE_DEVICES`, `GPU_DEVICE_ORDINAL`; on failure prints the ROCm message and `exit(1)`; on success may `execv` re-exec and `hipSetDevice`.

## Evidence

- [`evidence/flash_serve/mcp-session.evidence.json`](../evidence/flash_serve/mcp-session.evidence.json) — REA MCP session export: records for `inspect_native_exports` (dynamic symbol table), `inspect_elf_notes` (ABI tag), `inspect_analysis_view` (×2) and `inspect_gpu_code_objects` (9 AMDGPU code objects); the SHA-256 above is recorded in each record's `artifact` field (the `open_binary` operation itself is not an exported record — review C3).
- [`evidence/flash_serve/decompile/`](../evidence/flash_serve/decompile/) — REA CLI + Ghidra 12.1.4 decompilation evidence: `main-FUN_01ede7e0.json` (serve parser, defaults, validation), `mode-dispatcher-FUN_01f23680.json` (repack/repack-hash/resident-gib grammar), `ctx-validator-FUN_01f24220.json` (ctx/YaRN limits), `gpu-guard-FUN_01f70090.json` + `gpu-selector-FUN_01f700f0.json` (HIP device check), `set-ops-FUN_01f25a20.json`.
- [`evidence/flash_serve/cli-probes.txt`](../evidence/flash_serve/cli-probes.txt) — full runtime probe transcript (49 captured invocations — 46 from the initial session plus 3 follow-up probes, review F2 — with stdout/stderr/exit codes).
- [`evidence/flash_serve/search-procedure-main.json`](../evidence/flash_serve/search-procedure-main.json), [`search-string-rocm.json`](../evidence/flash_serve/search-string-rocm.json), [`xrefs-string-rocm.json`](../evidence/flash_serve/xrefs-string-rocm.json), [`search-string-not-a-gguf.json`](../evidence/flash_serve/search-string-not-a-gguf.json), [`search-string-truncated-header.json`](../evidence/flash_serve/search-string-truncated-header.json) — REA string/procedure search evidence.
- [`evidence/flash_serve/binutils-elf-info.txt`](../evidence/flash_serve/binutils-elf-info.txt), [`binutils-strings.txt`](../evidence/flash_serve/binutils-strings.txt) — supplementary binutils evidence (used because the REA MCP server's string inventory requires a configured decompiler provider).

### Method and limitations

- The REA MCP server (rea-agents 6.8.0) had **no deep-analysis provider
  configured** (no `GHIDRA_INSTALL_DIR`/Hopper/IDA/Kuna in its environment);
  `open_binary` and the provider-free tools worked, but `search_strings` and
  `binary_overview` returned `provider_unavailable`. Deep analysis was
  therefore performed through the **REA CLI** (`rea.mjs` from the same
  nix-store package) with `GHIDRA_INSTALL_DIR` pointing at nixpkgs
  **Ghidra 12.1.4**, reusing a local analysis snapshot. This is recorded as an
  environment gap to escalate (set `GHIDRA_INSTALL_DIR` where REA is launched
  and restart the MCP client to restore MCP-side decompilation).
- Static analysis never observes execution: behaviours marked "from
  decompilation" (port range absence, bind validation wording, socket setup,
  GPU enumeration) are corroborated by runtime probes where possible and by
  decompiled code otherwise.
- The binary is stripped; function names (`FUN_*`) are Ghidra-derived.
- GPU-dependent behaviour (actual serving, HTTP responses, inference output)
  is out of scope for this spec and gated per plan §5.
