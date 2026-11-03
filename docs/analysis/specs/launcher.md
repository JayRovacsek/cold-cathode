# launcher — contract spec (`halogen-flash-server` entrypoint)

> **Status:** frozen contract for `cc-launcher` (Phase 3/4; implemented last
> per `full-plan.md` §4 Phase 4 ordering).
> **Source artifacts (plain source — no REA, per `full-plan.md` §4 Phase 2):**
>
> | Artifact | Path | Size | SHA-256 |
> |---|---|---|---|
> | makeWrapper script (`$out/bin/halogen-flash-server`) | `/nix/store/fn6y2w20kbgjd8da8pwxbbc3gvmiycpy-halogen-flash-server-0.18.21/bin/halogen-flash-server` | 1 795 B | `f6757d40964525f7965cab30490ac2ae07b5ed5b4056b37966d60cfc60e52c01` |
> | upstream source entrypoint | `/nix/store/ksqcsnk4xigk07r02hib5p94znwv8pzb-source/deploy/entrypoint.sh` (v0.18.21 source tree) | 87 686 B (1 824 lines) | `c7f8fbfcb65b42b2b1841f7f75f9292ea43c1b3f297cea814e8d54e66560375e` |
> | installed entrypoint (`$out/libexec/halogen-entrypoint.sh`) | `/nix/store/fn6y2w20kbgjd8da8pwxbbc3gvmiycpy-halogen-flash-server-0.18.21/libexec/halogen-entrypoint.sh` | 89 876 B | `a6046db54e0c9a32042dc5a0c883f9d8978c216567ebeb485ec8e3329bb0932f` |
>
> **Provenance:** halogen-flash-server v0.18.21 (see `full-plan.md` Appendix B).
> **Evidence:** the files themselves (plain source); no `docs/analysis/evidence/`
> export is required for this contract.

## Architecture: two layers

The upstream `halogen-flash-server` command is **two files**:

1. **The makeWrapper script** (`$out/bin/halogen-flash-server`, 26 lines):
   exports the environment defaults (Appendix A of the plan), then
   `exec`s the entrypoint with `"$@"`.
2. **The entrypoint** (`$out/libexec/halogen-entrypoint.sh`, 1 824 lines):
   all validation, derivation and mode dispatch.

The Rust `cold-cathode` launcher must reproduce **both layers** (the plan's
§3 binary-naming decision: "the launcher's dispatch logic ports 1:1 from
`entrypoint.sh`"; the Rust binary installs as `cold-cathode` and its
`$out/libexec` layout must match what the wrapper points at).

### Source → store path rewrites

The installed entrypoint differs from `deploy/entrypoint.sh` **only** by
Nix path substitutions (verified by diff; the size delta is path-length
expansion). The port must apply the equivalent rewrites against its own
`$out`:

| In `deploy/entrypoint.sh` | In `$out/libexec/halogen-entrypoint.sh` |
|---|---|
| `#!/bin/bash` | `#!/nix/store/nnm9gdr70ihc6zc9298mzz8jpcgaciql-bash-5.3p20/bin/bash` |
| `python3` (bare) | `/nix/store/q4wf934v83ab6z79828l7wcgm8a8p7wg-python3-3.14.8-env/bin/python3` |
| `/usr/local/bin/flash_serve` | `$out/bin/flash_serve` (or via `${_hg_flash_serve:-…}`) |
| `/usr/local/bin/halogen-tools`, `/usr/local/bin/halogen-npu` | `$out/bin/halogen-tools`, `$out/bin/halogen-npu` |
| `/halogen/tools/serve_api.py`, `halogen-bench.py`, `bench-serving.py`, `halogen_tools.py` | `$out/share/halogen/tools/<same names>` |
| `/opt/halogen/flash-tune.plan`, `/opt/halogen/npu/models.txt` | `$out/opt/halogen/flash-tune.plan`, `$out/opt/halogen/npu/models.txt` |

## Layer 1 — the makeWrapper env contract

Verbatim wrapper (26 lines):

```bash
#! /nix/store/nnm9gdr70ihc6zc9298mzz8jpcgaciql-bash-5.3p20/bin/bash -e
export HALOGEN_ARCH="${HALOGEN_ARCH:-gfx1151}"
export HIP_FORCE_DEV_KERNARG="${HIP_FORCE_DEV_KERNARG:-1}"
export HALOGEN_ATTN_FA="${HALOGEN_ATTN_FA:-64}"
export HALOGEN_PROMPT_CACHE="${HALOGEN_PROMPT_CACHE:-2}"
export HALOGEN_PREFILL_CHUNK="${HALOGEN_PREFILL_CHUNK:-32768}"
export HALOGEN_KV_SLOTS="${HALOGEN_KV_SLOTS:-4}"
export HALOGEN_CTX="${HALOGEN_CTX:-262144}"
export HALOGEN_MAX_TOK="${HALOGEN_MAX_TOK:-32768}"
export HALOGEN_MATMUL_ALGOS="${HALOGEN_MATMUL_ALGOS:-1}"
export HALOGEN_MATMUL_BUCKET="${HALOGEN_MATMUL_BUCKET:-1}"
export HALOGEN_MATMUL_TUNING_FILE="${HALOGEN_MATMUL_TUNING_FILE:-<out>/opt/halogen/flash-tune.plan}"
export HALOGEN_DRAFT_VOCAB_FILE="${HALOGEN_DRAFT_VOCAB_FILE:-<out>/opt/halogen/draft-vocab.lists}"
export _hg_hf="${_hg_hf:-/nix/store/q4wf934v83ab6z79828l7wcgm8a8p7wg-python3-3.14.8-env/bin/hf}"
export HALOGEN_TOKENIZER="${HALOGEN_TOKENIZER:-/tokenizer}"
export HALOGEN_PORT="${HALOGEN_PORT:-8730}"
export HALOGEN_BIND="${HALOGEN_BIND:-127.0.0.1}"
export HALOGEN_API_PORT="${HALOGEN_API_PORT:-8731}"
export HALOGEN_MAX_TOKENS_CAP="${HALOGEN_MAX_TOKENS_CAP:-65536}"
export HALOGEN_QUEUE_TIMEOUT="${HALOGEN_QUEUE_TIMEOUT:-3600}"
export HALOGEN_IMAGE_VERSION="0.18.21"
export HALOGEN_IMAGE_MODES="${HALOGEN_IMAGE_MODES:-all engine api bench sweep convert inspect verify ppl niah}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
exec "<out>/libexec/halogen-entrypoint.sh"  "$@"
```

(`<out>` = `/nix/store/fn6y2w20kbgjd8da8pwxbbc3gvmiycpy-halogen-flash-server-0.18.21`;
the wrapper's exec line contains two spaces before `"$@"`.)

### `${VAR:-default}` semantics (locked)

- Every default uses `${VAR:-default}`: the default applies only when the
  variable is **unset or empty**; a set-but-empty `VAR=""` also takes the
  default, and any non-empty value wins verbatim (no validation at this
  layer).
- **Exception:** `HALOGEN_IMAGE_VERSION="0.18.21"` is a hard assignment —
  **not overridable** from the environment (verified: `HALOGEN_IMAGE_VERSION=9.9
  halogen-flash-server bogus` still prints `0.18.21`). The plan's Appendix A
  marks it "pinned, not overridable" — confirmed.
- The wrapper runs bash with `-e` (`#! … bash -e`): any failing `export`
  aborts before the entrypoint runs.
- The wrapper exports the variables, so **all child processes inherit them**
  (this is how `flash_serve` receives e.g. `HALOGEN_MATMUL_ALGOS` without
  flags — see the flash_serve spec note that `--port/--bind/--ctx/--max-tok/--ck`
  are translated into flags by the entrypoint, while other `HALOGEN_*`
  variables reach the engine through the environment).

### Variance vs the plan's Appendix A (escalate)

Appendix A lists the tuning/vocab defaults as
`/1rz4g4znpzjwh1xymhjpm42vipw92pr73vdgl6xs1hycac8kf2n9/opt/halogen/…` and
`_hg_hf` as `${python3-env}/bin/hf`. The live v0.18.21 wrapper uses the
concrete store paths shown above (`…-halogen-flash-server-0.18.21/opt/halogen/…`
and `…-python3-3.14.8-env/bin/hf`). The Appendix A forms look like
placeholder/normalised paths from an earlier revision. The live wrapper is
transcribed here as the contract; the orchestrator should reconcile
Appendix A in the decision log.

## Layer 2 — entrypoint pre-dispatch behaviour

Everything below runs **before** the mode dispatch, for every mode (lines
3–1015 of the entrypoint), and is observable on stdout/stderr:

1. `set -euo pipefail`.
2. `export HSA_DISABLE_COREDUMP_ON_EXCEPTION="${HSA_DISABLE_COREDUMP_ON_EXCEPTION:-1}"`;
   `ulimit -c 0` (line 5–6).
3. **`oom_adj_raise`** (lines 8–24): unless `HALOGEN_OOM_SCORE_ADJ=0`,
   validates `0..1000` (else `halogen: HALOGEN_OOM_SCORE_ADJ=$want is not
   0..1000; left as inherited` → stderr, continue), reads
   `${_hg_proc:-/proc}/$$/oom_score_adj`; if the current value is lower
   than the wanted one, writes it and prints
   `halogen: oom_score_adj <cur> -> <want>: if the host runs out of memory, the kernel ends this server before the desktop session or the container runtime (HALOGEN_OOM_SCORE_ADJ=0 leaves it)`
   (stderr); if `/proc` is read-only: `halogen: NOTE could not raise
   oom_score_adj from <cur> (read-only /proc?); a host out of memory may
   end other processes first` (stderr). **Contract tests must control the
   inherited oom score** (or set `HALOGEN_OOM_SCORE_ADJ=0`) to make output
   deterministic.
4. `ENG_PORT="${HALOGEN_PORT:-8730}"`, `API_PORT="${HALOGEN_API_PORT:-8731}"`,
   `BIND="${HALOGEN_BIND:-127.0.0.1}"` (re-derived; identical defaults to
   the wrapper).
5. `export HALOGEN_VERBOSE="${HALOGEN_VERBOSE:-0}"`.
6. **Checkpoint resolution** (`default_checkpoint` / `resolve_checkpoint`,
   lines 32–47): if `HALOGEN_CHECKPOINT` is unset/empty, it becomes
   `/models/qwen38-flash-next-v2.hgn`, **except** when that file is absent
   and `/models/qwen38-flash-next-w4b.hgn` exists → use w4b and set the
   internal `LEGACY_W4B=1` flag. The result is exported.
7. `ENG_SLOTS="${HALOGEN_KV_SLOTS:-4}"`, `ENG_CTX="${HALOGEN_CTX:-262144}"`.
8. **KV pool derivation** (lines 52–57): `ENG_POOL="${HALOGEN_KV_POOL_POSITIONS:-}"`;
   when empty: `ENG_POOL = min(max(ENG_CTX*2, ENG_CTX), 1048576)` (i.e.
   `ctx*2` clamped to ≤ 1 048 576).
9. **Max-token capping** (lines 59–71): `ENG_MAX_TOK="${HALOGEN_MAX_TOK:-32768}"`;
   if `ENG_CTX > 262144` and `ENG_MAX_TOK > 16384` → cap to 16384 with the
   exact note `halogen: context $ENG_CTX is past the native 262144:
   HALOGEN_MAX_TOK $ENG_MAX_TOK is capped at 16384 here (a larger prefill
   arena leaves a 1M KV cache no room to stay resident).`; also, with
   `HALOGEN_PROMPT_CACHE≠0`, `HALOGEN_CACHE_INPLACE=0` and no
   `HALOGEN_CACHE_FILE`, exports `HALOGEN_CACHE_FILE=/var/tmp/halogen-cache.snapshot`
   with its own note. Finally `ENG_MAX_TOK = min(ENG_MAX_TOK, ENG_CTX)`.
10. **YaRN gate** (lines 73–80): `HALOGEN_CTX > 262144` without
    `HALOGEN_ROPE_YARN` → error (exact text at lines 75) and **exit 2**;
    `HALOGEN_ROPE_YARN` set with `ctx ≤ 262144` → `halogen: WARNING: …`
    (stderr, continues).
11. **Front-end-only flag stripping** (lines 1002–1015): `HALOGEN_FORK`
    defaults to `1` (exported); the list `FE_ONLY_FLAGS` (37 variables:
    `HALOGEN_API_KEY HALOGEN_API_KEY_FILE HALOGEN_CLIENT_MARKERS
    HALOGEN_ENABLE_THINKING HALOGEN_ENGINE_PING_S HALOGEN_ENGINE_PROBE_S
    HALOGEN_FREQUENCY_PENALTY HALOGEN_KEEPALIVE_TIMEOUT
    HALOGEN_MAX_THINKING_TOKENS HALOGEN_MAX_TOKENS_DEFAULT HALOGEN_MIN_P
    HALOGEN_MODEL_ID HALOGEN_NEXT_TOKEN_S HALOGEN_PRESENCE_PENALTY
    HALOGEN_REASONING_EFFORT HALOGEN_RECORD_DAYS HALOGEN_RECORD_DIR
    HALOGEN_RECORD_GIB HALOGEN_RECORD_TEXT HALOGEN_REPETITION_PENALTY
    HALOGEN_RETRY_GRACE_S HALOGEN_SSE_KEEPALIVE_S HALOGEN_TEMPERATURE
    HALOGEN_TEMPLATE_CHECK HALOGEN_TEMPLATE_UNCHECKED
    HALOGEN_THINKING_ANSWER_ROOM HALOGEN_CHAT_TEMPLATE HALOGEN_STRICT_TOOLS
    HALOGEN_TOKENIZE_REUSE HALOGEN_TOP_K HALOGEN_TOP_P HALOGEN_CACHE_LEAD
    HALOGEN_TOOL_DEFER HALOGEN_FORK_AUTO_END_ONLY HALOGEN_FORK_AUTO_KEEPS
    HALOGEN_FORK_HANDLES HALOGEN_FORK_MAP`) is turned into
    `ENGINE_ENV_U=(-u VAR …)` — every `flash_serve` spawn uses
    `env "${ENGINE_ENV_U[@]}" …` so these front-end-only variables are
    **stripped from the engine's environment** (the engine must not see
    them; the port must reproduce the `-u` list exactly).

Mode-dependent helpers invoked per dispatch (not pre-dispatch): `need_ckpt`
(host preflight, download, tuning-plan copy, checkpoint/sidecar/ngram/vision
checks, KV-budget and GTT notes), `need_tokenizer`, `check_defaults`,
`wait_for_engine`, `engine_watchdog`, `stop_engine`, `wd_gtt_after_exit`,
and the NPU helpers. Their exact strings live in the entrypoint lines cited
in the dispatch table.

## The dispatch table

`argv[1]` selects the mode (default `all`). **Banner rule** (lines
1666–1669): modes `inspect|verify|ppl|niah` print
`halogen: halogen-flash-server ${HALOGEN_IMAGE_VERSION:-unknown}, mode <mode>`
to **stderr**; every other mode prints
`halogen: halogen-flash-server ${HALOGEN_IMAGE_VERSION:-unknown}, mode ${1:-all}`
to **stdout**.

| Mode | Serves via | Behaviour (entrypoint lines) | Exit codes |
|---|---|---|---|
| `all` (default) | `flash_serve` + `serve_api.py` (+ optional `halogen-npu`) | (1674–1720) if `_hg_npu_alone=1` and `HALOGEN_NPU_MODELS` set → `start_npu_alone`; else `need_ckpt; need_tokenizer; check_defaults`, optional NPU engine, spawn `env "${ENGINE_ENV_U[@]}" ${_hg_flash_serve:-…/flash_serve} --ck "$HALOGEN_CHECKPOINT" --port "$ENG_PORT" --bind 127.0.0.1 --slots "$ENG_SLOTS" --ctx "$ENG_CTX" --max-tok "$ENG_MAX_TOK" --kv-pool "$ENG_POOL" &`, print `halogen: waiting for engine on $ENG_PORT (cold load can take minutes)`, `wait_for_engine`, spawn `serve_api.py` (with `${NPU_API_ARGS[@]}`), start `engine_watchdog`, then a `wait -n` loop with NPU-restart logic (`npu_exited_alone`/`npu_again`) and `all_takedown` | 0 on clean stop (`stop_clean`: rc ∈ {0,143}); engine exit code otherwise; 1 if the engine never comes up |
| `engine` | `flash_serve` | (1017–1059) `start_engine`: `need_ckpt`, `memlock_note`, spawn `flash_serve` with the same flag set (`--bind "$BIND"` here, not hard-coded 127.0.0.1), `wait_for_engine`, watchdog, `wait`, `stop_engine`, `wd_gtt_after_exit`; prints `halogen: stop requested; engine exited (rc=$RC)` or `halogen: engine exited (rc=$RC, the component that ended this: $WRC); shutting down` | 0 on clean stop; else the engine's rc |
| `api` | `serve_api.py` | (1069–1080) `need_tokenizer`, `check_defaults`, then **`exec`** `python3 …/serve_api.py --tokenizer "$HALOGEN_TOKENIZER" --engine "${HALOGEN_ENGINE:-127.0.0.1:$ENG_PORT}" --host 0.0.0.0 --port "$API_PORT" --max-tokens-cap "${HALOGEN_MAX_TOKENS_CAP:-65536}" --context "$ENG_CTX" --queue-timeout "${HALOGEN_QUEUE_TIMEOUT:-3600}"` | python3's exit status |
| `bench` | engine + api + `bench-serving.py` | (1721–1775) `text_only_mode` (unsets `HALOGEN_VISION_TOWER` with a note if set), `need_ckpt; need_tokenizer; check_defaults`; engine stdout/err → `/tmp/halogen-engine.log`; api (spawned with `env -u HALOGEN_API_KEY -u HALOGEN_API_KEY_FILE`) → `tee /tmp/halogen-api.log`; poll `/health` up to 150×2 s (`halogen bench: the front-end never answered /health on $API_PORT after 300s.` → exit 1); then `HALOGEN_API=… HALOGEN_API_LOG=… python3 …/bench-serving.py "${1:-serial,mtp}" "${2:-256}" "${3:-low}" "${4:-1}"` (positional args **after** the mode shift, with those defaults); kills both, exits with the tool's rc | tool's rc; 1 if the front end never answers |
| `sweep` | engine + api + `halogen-bench.py` | same preamble; runs `python3 …/halogen-bench.py --api "http://127.0.0.1:$API_PORT" "$@"` (all remaining args forwarded verbatim) | tool's rc |
| `convert IN.gguf OUT.hgn` | `flash_serve --repack` | (1776–1800) requires exactly two more args else usage → **exit 2** (exact 3-line text below); `IN` must exist and start with `GGUF`; `OUT`'s dir must be writable; `need_head "$(dirname "$IN")"`; runs `flash_serve --repack "$IN" --out "$OUT" --with-table --head "$HALOGEN_MTP_HEAD"`; on success prints `halogen convert: done. …`; on failure `halogen convert: the repack failed (rc=$RC); $OUT is not usable and can be removed` | 2 (usage), 1 (missing/unwritable inputs), or flash_serve's rc |
| `inspect [FILE] …` | `halogen-tools` | (1801–1821) `text_only_mode`; `FILE` = first non-flag arg, else `$HALOGEN_CHECKPOINT`; must exist (`halogen inspect: $FILE is not there (mount the models volume and name a file inside it, or set HALOGEN_CHECKPOINT)` → exit 1); **`exec …/bin/halogen-tools "$MODE" "$FILE" "$@"`** (installed form; see `docs/analysis/specs/halogen-tools.md` when written) | halogen-tools' rc; 1 if FILE missing |
| `verify [FILE] …` | `halogen-tools` | same as `inspect`, mode word `verify` | same |
| `ppl [FILE] …` | `halogen_tools.py` | (1810–1819) `tuning_plan_copy`; GGUF file → `need_head`, else `HALOGEN_CHECKPOINT="$FILE" check_ngram_table >&2`; prints `halogen ppl: $FILE under this image's engine environment (tuning plan: ${HALOGEN_MATMUL_TUNING_FILE:-none}; quality sidecar: ${HALOGEN_CK_OVERLAY:-beside the checkpoint, if any}; trunk pinned: ${HALOGEN_FLASH_PIN_TRUNK:-1})` (stderr); `need_tokenizer`; **`exec python3 …/halogen_tools.py ppl "$FILE" --tokenizer "$HALOGEN_TOKENIZER" "$@"`** | python tool's rc |
| `niah [FILE] …` | `halogen_tools.py` | same as `ppl`, mode word `niah` | same |
| `npu` | — (retired) | (1822) stderr `halogen: there is no npu mode any more: the NPU's models run in the default mode beside the Flash model; set HALOGEN_NPU_MODELS (and drop the npu argument).` then the general usage → **exit 2** | 2 |
| anything else | — | general usage (below) → **exit 2** | 2 |

### Exact usage strings

General (stdout modes print the banner first; usage goes to **stderr**):

```
usage: entrypoint.sh [all|engine|api|bench|sweep|convert IN.gguf OUT.hgn|inspect|verify|ppl|niah [FILE] ...]
```

Note the usage text is the literal string `entrypoint.sh` (not `$0`) — the
Rust launcher `cold-cathode` must decide whether to keep it verbatim
(contract fidelity) or substitute its own name; **recommendation: keep
`entrypoint.sh` verbatim** and record the choice in the decision log.

`convert` usage (stderr, exit 2):

```
usage: entrypoint.sh convert IN.gguf OUT.hgn
  IN: any shard of a llama.cpp GGUF of Qwen3.8-Flash-Next (unsloth's, bartowski's, your own llama-quantize).
  OUT: the .hgn to write (about 105 GiB for an IQ4_XS build; the table and the draft head are folded in).
```

## Env-var → binary mapping

| Env var | Consumed as |
|---|---|
| `HALOGEN_CHECKPOINT` | `flash_serve --ck` (all/engine/bench/sweep); `halogen-tools` FILE default; `halogen_tools.py` FILE default |
| `HALOGEN_PORT` | `flash_serve --port`; healthcheck `engine` target; `serve_api.py --engine 127.0.0.1:$PORT` |
| `HALOGEN_BIND` | `flash_serve --bind` (engine mode only; `all`/`bench` hard-code `127.0.0.1`) |
| `HALOGEN_KV_SLOTS` | `flash_serve --slots` (also read natively by the binary — see flash_serve spec) |
| `HALOGEN_CTX` | `flash_serve --ctx`; `serve_api.py --context` |
| `HALOGEN_KV_POOL_POSITIONS` | `flash_serve --kv-pool` (via the `ENG_POOL` derivation) |
| `HALOGEN_MAX_TOK` | `flash_serve --max-tok` (via the `ENG_MAX_TOK` derivation) |
| `HALOGEN_API_PORT` | `serve_api.py --port` (host `0.0.0.0` in all/engine; `127.0.0.1` in bench/sweep) |
| `HALOGEN_MAX_TOKENS_CAP` | `serve_api.py --max-tokens-cap` |
| `HALOGEN_QUEUE_TIMEOUT` | `serve_api.py --queue-timeout` |
| `HALOGEN_TOKENIZER` | `serve_api.py`/`halogen_tools.py --tokenizer` (with the `/tokenizer` → models-volume fallback in `need_tokenizer`) |
| `HALOGEN_MTP_HEAD` | `flash_serve --head` (convert mode; `need_head` derivation) |
| `HALOGEN_ENGINE` | `serve_api.py --engine` override (api mode only) |
| `HALOGEN_IMAGE_VERSION` | banner text (pinned by the wrapper) |
| `HALOGEN_IMAGE_MODES` | advertises the mode list (informational; the dispatch is hard-coded) |
| the 37 `FE_ONLY_FLAGS` | stripped from every `flash_serve` spawn via `env -u` (the engine must not see them) |
| remaining `HALOGEN_*` (e.g. `HALOGEN_MATMUL_ALGOS`, `HALOGEN_ATTN_FA`, `HALOGEN_ARCH`, `HIP_FORCE_DEV_KERNARG`) | reach `flash_serve` through the inherited environment (see the flash_serve spec's env-var table) |

### Internal `_hg_*` indirection variables

The entrypoint routes every binary/path reference through overridable
`_hg_*` variables (used by the test/dev harness to point at alternates):
`_hg_hf`, `_hg_python`, `_hg_flash_serve`, `_hg_npu_bin`, `_hg_npu_pins`,
`_hg_npu_dir`, `_hg_npu_out`, `_hg_npu_alone`, `_hg_npu_backoff`,
`_hg_proc`, `_hg_sys`, `_hg_dev`, `_hg_root`, `_hg_hub_list`, `_hg_sha256`,
`_hg_memlock`, `_hg_sys` (see lines 1082–1083, 1109, 1367, 1462, 1536,
1576, 1643, 1680, 1695, 1729, 1739, 1765, 1769, 1791, 1818, 1820 and the
`_hg_proc`/`_hg_sys` uses in the oom/watchdog helpers). The store wrapper
sets only `_hg_hf`; the rest default to the store paths shown in the
rewrite table. The Rust port should honour the same variables.

## Ground truth (this host, 2026-11-03)

`/models` absent; `HALOGEN_IMAGE_VERSION` override ignored (pinned).
Every invocation emits the `oom_score_adj 0 -> 500: …` line (stderr)
because the inherited score is 0 (see §Layer 2 item 3).

| Invocation | stdout | stderr | exit |
|---|---|---|---|
| `halogen-flash-server bogus` | `halogen: halogen-flash-server 0.18.21, mode bogus` | oom line + general usage | 2 |
| `halogen-flash-server convert` | `halogen: halogen-flash-server 0.18.21, mode convert` | oom line + convert usage (3 lines) | 2 |
| `halogen-flash-server npu` | `halogen: halogen-flash-server 0.18.21, mode npu` | oom line + retirement line + general usage | 2 |
| `halogen-flash-server inspect` | *(empty — inspect banners on stderr)* | oom line + `halogen: halogen-flash-server 0.18.21, mode inspect` + `halogen inspect: /models/qwen38-flash-next-v2.hgn is not there (mount the models volume and name a file inside it, or set HALOGEN_CHECKPOINT)` | 1 |
| `HALOGEN_IMAGE_VERSION=9.9 halogen-flash-server bogus` | `halogen: halogen-flash-server 0.18.21, mode bogus` | (as above) | 2 |

## Exit-code summary

| Code | Context |
|---|---|
| 0 | clean shutdown (`stop_clean`: rc ∈ {0,143}); tool success in exec modes |
| 1 | missing checkpoint/tokenizer/head; engine failed to start; NPU preflight failures; GGUF not downloadable; tool failure via exec |
| 2 | unknown mode, retired `npu` mode, `convert` usage, `HALOGEN_CTX > 262144` without `HALOGEN_ROPE_YARN` |
| 137 | engine unkillable after SIGKILL (`stop_engine`) |
| child's rc | `api`/`convert`/`inspect`/`verify`/`ppl`/`niah` propagate the exec'd program's status |

## Open questions (escalate to the orchestrator)

1. **Appendix A path variance** (see §Layer 1): the plan's placeholder paths
   vs the live store paths — reconcile in the decision log.
2. **Usage string `entrypoint.sh`**: keep verbatim in `cold-cathode`
   (recommended) or substitute the new binary name?
3. **oom_score_adj line in contract tests**: environment-dependent; the
   matrix should set `HALOGEN_OOM_SCORE_ADJ=0` (or pin the inherited score)
   for deterministic output.
4. **`docs/analysis/specs/halogen-tools.md` / `halogen-npu.md`** don't exist
   yet; the launcher spec cross-references them as the Phase 2 REA targets
   for the `inspect`/`verify` and NPU paths.

## Method and limitations

- Plain-source transcription (plan §4 Phase 2: "Launcher contract needs no
  REA — it is plain source"). The three artifacts are hashed above; the
  source→store diff is fully accounted for by the path-rewrite table.
- Runtime probes cover only side-effect-free paths (unknown/retired modes,
  `convert` usage, `inspect` with no models volume). Modes that start the
  stack (`all`, `engine`, `api`, `bench`, `sweep`, `ppl`, `niah`) require
  a checkpoint and GPU and are exercised by the Phase 3 contract harness
  instead.
- The entrypoint's deep validation logic (download planning, GGUF checks,
  NPU pin verification, watchdog internals — lines 82–998) is summarised
  here; the file itself is the normative text, and the Phase 3 matrices
  should cite line ranges.
