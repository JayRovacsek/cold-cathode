# Objectives: features added between halogen v0.18.21 and v0.20.5

Provenance: reverse-engineering diff between the two vendored builds now
symlinked in the repo root (`result-0.18.21` → the initial v0.18.21 store
output, `result-new` → the bumped v0.20.5 store output). Method: REA
`open_binary`/`inspect_rust_binary` for target handling with a binutils
(`strings`/`nm`) fallback for the string and symbol diff (the decompiler
providers are unconfigured in this environment); plain diffs for the
Python tool tree, entrypoint and wrapper. Raw evidence lives in
[`evidence/version-diff-0.18.21-0.20.5/`](evidence/version-diff-0.18.21-0.20.5/);
file names cited below refer to that directory. The upstream
`CHANGELOG.md` (vendored there as `changelog-v0.20.5.md`) was used only
for corroboration; every objective below has store-output evidence.

Scope corrections (verified, NOT new in 0.20.5 — do not re-implement):
the GTT preflight (`gtt_note`), the engine PING/PONG watchdog (including
`HALOGEN_ENGINE_WATCHDOG_DEFER_S` compact-stall deferral) and the NPU
download planning (`NPU_NEED_K`) all already exist in v0.18.21.

Out of port scope: the GPU/NPU-internal numeric clusters (DN
state/swap/tap, wide-tick batch decode, thin-router padding, speech
codec numerics, vision-tower numerics) are engine-internal and excluded
from these objectives; only their API-visible plumbing is listed.

## 1. Launcher (`libexec/halogen-entrypoint.sh` → `cc-launcher`)

- [x] **O1 — `HALOGEN_AGENT_GUARD` mode parsing.** Accept `off`/`watch`/`hold`
      (case-insensitive, plus `0`/`false`/`no`/`none`); `watch` injects
      `halogen-spotter-0.6b` into `HALOGEN_NPU_MODELS`, `hold` also injects
      `decider-4b`; an unknown mode exits 1 with the exact notice.
      Evidence: `entrypoint.diff` L7–26.
- [x] **O2 — Reply-cache memory accounting.** `cache_gib` counts one extra
      cache entry per slot for the reply-end cache when
      `HALOGEN_PROMPT_CACHE != 1` and `HALOGEN_CACHE_REPLY` is unset.
      Evidence: `entrypoint.diff` L36.
- [x] **O3 — New `FE_ONLY_FLAGS` entries.** Strip `HALOGEN_RENDER_ONCE`,
      `HALOGEN_TOKENIZE_SPLIT`, `HALOGEN_TOOL_NONE_MASK`,
      `HALOGEN_TOKENIZER_NATIVE`, `HALOGEN_FORK_UNMATCHED_RELEASE`,
      `HALOGEN_AGENT_GUARD`, `HALOGEN_MODEL_NAME_CHECK` from the engine
      environment. Evidence: `entrypoint.diff` L71–77.
- [x] **O4 — IPv6 API bind.** Serve with `--host ::` instead of `0.0.0.0`
      in all serve modes. Evidence: `entrypoint.diff` L108, 464, 494.
- [x] **O5 — `vis`/`aud` NPU download parts.** `npu_model_ready` gains
      vision/audio part types with their own `devices.hnpm`/`.elf` checks
      and `NPU_NEED_K` accounting; `npu_finetune_ready` passes `--vis`/
      `--aud` to convert. Evidence: `entrypoint.diff` L133–176, 194–217.
- [x] **O6 — Bundled-XRT handling (`npu_xrt_own`).** Verify the three XRT
      libs against `npu/xrt.sha256`, refuse mixed host/image mounts, name
      "this image's XRT" at startup. Note: the vendored package does not
      ship `/opt/xilinx/xrt`, so implement the host-mount fallback path.
      Evidence: `entrypoint.diff` L223–273.
- [x] **O7 — NPU stall watchdog (`HALOGEN_NPU_KILL_S`).** Background poller
      reads `running_ms` via `INFO 0`; a job holding the NPU past the
      limit (default 1800 s) gets TERM→KILL and a restart; D-state threads
      defer the kill; pidfile `/tmp/halogen-npu.pid`. Evidence:
      `entrypoint.diff` L360–408.
- [x] **O8 — Host-floor backoff restart (rc 76).** A memory-floor exit costs
      no restart count; retries back off 30 s → 300 s doubling
      (`NPU_PAUSED`, `NPU_ROOM_S`). Evidence: `entrypoint.diff` L439–448.

## 2. API front-end (`serve_api.py` → `cc-flash-serve` / contract surface)

> **Status:** satisfied by the vendored v0.20.5 package — plan R4 reuses the
> Python front-end as data, so these need no Rust port work; the bump already
> delivered them.

- [ ] **O9 — Agent guard module.** Port `agent_guard.py` (Guard class,
      watch/hold modes, spotter classification: ACT_WORDS, threshold 0.25,
      window/stride, scope/side system prompts) wired via
      `app.state.agent_guard` and `gen = aguard.wrap(...)`.
      Evidence: new file; `py-diff-serve_api.txt` L5027–5031, 7024.
- [ ] **O10 — `tool_addition`/`tool_removal` blocks on `/v1/messages`.**
      Mid-session Claude Code tool changes rendered as "Tool added: …"
      system blocks plus a `<tools>` block from the defer map;
      `with_added_tools` appends the definition.
      Evidence: `py-diff-serve_api.txt` L448–482, new defs.
- [ ] **O11 — `model_not_found` 404 refusals.** A halogen model name that
      isn't loaded gets 404 on chat/completions, messages and responses
      (`flash_model_refusal`, `npu_api.unloaded_refusal`); unknown foreign
      names are still answered, logged once.
      Evidence: `py-diff-serve_api.txt` L1928, 3226–3286, 429.
- [ ] **O12 — `ignore_eos` on `/v1/completions`.** Request field; `[]` when
      set, else the eos ids, passed to generate.
      Evidence: `py-diff-serve_api.txt` L3498, 5398.
- [ ] **O13 — `tool_choice:"none"` mask.** `TOOL_NONE_MASK` (default on)
      plus a `nocall` guard so a tool call cannot arrive as text when
      tools exist in history. Evidence: `py-diff-serve_api.txt` L769, 3052.
- [ ] **O14 — `check_tool_schemas`.** Non-object tool `parameters` → 400
      naming the tool (previously a 500/slot leak).
      Evidence: `py-diff-serve_api.txt` new def L1147.
- [ ] **O15 — Native tokenizer normalisation.** `HALOGEN_TOKENIZER_NATIVE`
      with `native_split`/`split_encode`/`TOKENIZE_SPLIT` so Indic-script
      prompts tokenise as trained. Evidence: `py-diff-serve_api.txt` L3125–3130.
- [ ] **O16 — Render-once template path.** `RENDER_ONCE` with
      `gen_suffix`/`_gen_block_names`/`view_segments`/`lookup_view`
      prompt-lookup rewrite speedups. Evidence: `py-diff-serve_api.txt`
      L3030–3086, 2944–2974.
- [ ] **O17 — Startup warm-up.** `HALOGEN_START_WARM` (default 1): one
      short internal request before the API reports ready.
      Evidence: `py-diff-serve_api.txt` L1244–1247.
- [ ] **O18 — Fork unmatched release.** `HALOGEN_FORK_UNMATCHED_RELEASE`:
      unmatched `X-Halogen-From`/`Keep` state released early.
      Evidence: `py-diff-serve_api.txt` L2935.
- [ ] **O19 — Reasoning-close accounting.** `reasoning_closed_by` /
      `end_of_turn_in_reasoning` in usage details; `think_prose_end`.
      Evidence: `py-diff-serve_api.txt` L1153–1161, 2070–2080.
- [ ] **O20 — NPU shim routes.** `npu_wire_routes`/`_npu_shim`/
      `_npu_messages_req`: NPU generation models served on the main API's
      chat/messages/responses (3 new `add_api_route`s).
      Evidence: `py-diff-serve_api.txt` new defs L4793–4968.
- [ ] **O21 — `/health` capability text.** Speech voices/modes and
      per-model `architecture.input_modalities` (ovis/omni/vl/aud/
      speech_out). Evidence: `py-diff-npu_api.txt` L887, 946.

## 3. NPU surface (`npu_api.py` / `npu_sets.py` / `models.txt`)

> **Status:** satisfied by the vendored v0.20.5 package — plan R4 reuses the
> Python front-end and `models.txt` as data, so these need no Rust port work;
> the bump already delivered them.

- [ ] **O22 — d1-2B media + box/boxes/estimate questions.** `box`
      (5 slots), `boxes` (≤16 instances), `estimate` with range/step grid;
      `d1_media`/`d1_picture`/`d1_video` with vision_start/image/video
      tokens. Evidence: `py-diff-npu_sets.txt` L17–60; npu_api new defs.
- [ ] **O23 — LFM3-VL-2B routing.** Pictures/video on chat routes
      (`vl_ids`/`vl_media`/`vl_picture`); models.txt record with
      `devices=lfm3-1.7b vis=…`. Vision-tower numerics stay out of scope.
      Evidence: npu_api new defs; `models.diff`.
- [ ] **O24 — LFM3-Audio-2B request handling.** `aud_ids`,
      `speech_sampling`, `speech_voice` (female/male + OpenAI aliases),
      `wav_header`/`wav_info`, `wants_audio_out`, `modalities`. Codec
      numerics stay out of scope. Evidence: npu_api new defs;
      `py-diff-npu_api.txt` L516–527, 1515–1535.
- [ ] **O25 — LFM3-Omni-2B bounds.** `omni_bounds` (4–384 frames, 2 MP).
      Evidence: npu_api new def; models.txt omni lines.
- [ ] **O26 — OvisOCR3.** `ovis_default_max`, `render_ovis`; models.txt
      `ovisocr3` record (Qwen3_5 arch). Evidence: npu_api new defs;
      `models.diff`; npu strings `Qwen3_5ForConditionalGeneration`.
- [ ] **O27 — NPU token cache/split.** `HALOGEN_NPU_TOK_CACHE`/`_TOK_SPLIT`/
      `_TAIL_HOLD` with `tok_ids`/`_tok_piece`/`record_ids`/`lfm3_cached`.
      Evidence: `py-diff-npu_api.txt` L1061–1248.
- [ ] **O28 — d3-* strict key validation.** Unknown keys → 422 naming where
      text belongs. Evidence: `py-diff-npu_api.txt` L1017, 1046.
- [ ] **O29 — NPU 503 error codes.** `npu_memory` alongside
      `engine_busy`/`host_memory`. Evidence: `py-diff-npu_api.txt` L22–24.
- [ ] **O30 — `tool_parse` whole-tail close.** `_whole_tail_close`/
      `_tail_start`: args quoting closing tags arrive whole.
      Evidence: `py-diff-tool_parse.txt`.
- [ ] **O31 — models.txt new records.** `halogen-spotter-0.6b`,
      `lfm3-{vl,audio,omni}-2b(-fast)`, `ovisocr3`; d1-2b placeholder →
      full 16-line record; `aud=` doc. Evidence: `models.diff`.

## 4. Engine-visible flags (`flash_serve` / `halogen-npu` strings)

- [x] **O32 — Reply-end cache bookkeeping (API-visible part).** Refusal and
      drop reasons ("a reply-end entry", "dropped: its history",
      "req %ld reply entry: …") and the prompt-cache budget line
      `…+ %d reply]`. KV internals stay out of scope.
      Evidence: `strings-new-only-flash_serve.txt`.
- [x] **O33 — Think-block end-of-turn close (API-visible part).**
      `HALOGEN_THINK_EOT_CLOSE` plumbing, "NOCALL needs GUARD".
      Evidence: `strings-new-only-flash_serve.txt`.
- [x] **O34 — Answer-guard heading rule flag.** `HALOGEN_ANSWER_GUARD_HEADING`
      plumbing only. Evidence: `strings-new-only-flash_serve.txt`.
- [x] **O35 — KV-pool shortening warning.** "kv pool: request %ld was
      shortened by the room its region had left…" log behaviour.
      Evidence: `strings-new-only-flash_serve.txt`.
- [x] **O36 — NPU speech-out flags (plumbing).** `HALOGEN_NPU_AOUT_{CHUNK,
      FIRST,FRAMES,THREADS,CODEC_THREADS,DUMP}` accepted; codec numerics
      out of scope. Evidence: `newenvs-halogen-npu.txt`.
- [x] **O37 — Convert stamping.** "halogen-npu convert: could not write %s,
      so the next start converts again" / "weights stamp 1" behaviour.
      Evidence: `strings-new-only-halogen-npu.txt`.
- [x] **O38 — Media validation errors.** "a malformed media request/
      soundtrack/audio clip", "an audio clip is not base64" error paths.
      Evidence: `strings-new-only-halogen-npu.txt`.

## Excluded as out of scope (recorded for completeness)

- DN state/swap/tap, `forward_batch_wide`/tick rows, thin router and
  padded buffers, ngram arrival prefetch, MTP gate/exact sampling
  (flash_serve internals); LFM3 long-prompt speedups (halogen-npu
  internals). Evidence: `strings-new-only-flash_serve.txt`,
  `newenvs-halogen-npu.txt`.
- `bin/halogen-healthcheck`: byte-identical between the two versions
  (md5 `9bee2d57…`), nothing to do.
- `bin/halogen-tools`: no new subcommands (`HALOGEN_IMAGE_MODES`
  identical); its new-only strings mirror the flash_serve set.
- Changelog's "anyOf ≤32 branches" (#263): the relevant strings are
  byte-identical in both binaries, so the change is an unobservable
  numeric constant — not verifiable from the store outputs.