# cold-cathode — HTTP API surface contract (`api.md`)

> **Phase 2 deliverable** (full-plan §4): the HTTP route table, request/response
> schemas and error bodies of the upstream `halogen-flash-server` v0.18.21
> HTTP API, transcribed from the shipped Python front-end source
> (`share/halogen/tools`, reused as data per R4 — not rewritten).
> **Source of truth:**
> `/nix/store/fn6y2w20kbgjd8da8pwxbbc3gvmiycpy-halogen-flash-server-0.18.21/share/halogen/tools/serve_api.py`
> (6785 lines; line citations below) and
> `…/share/halogen/tools/npu_api.py` (3492 lines).
> **Status:** frozen contract for Phase 3 HTTP contract tests (plan §5 category 3).

---

## 1. Ports, modes and which server owns which surface

| Port | Env var (default) | Protocol | Served by | Upstream modes that start it |
|---|---|---|---|---|
| **8730** | `HALOGEN_PORT` | **line protocol, not HTTP** (TCP; `PING`→`PONG`, `INFO`, `GEN …`) | `flash_serve` (C++ engine) | `engine`, `all`, `bench`, `sweep` |
| **8731** | `HALOGEN_API_PORT` | **HTTP/1.1** (uvicorn, `http="h11"`, `ws="none"` — serve_api.py:887) | `serve_api.py` (FastAPI app, title `halogen`) | `api`, `all`, `bench`, `sweep` |
| **8740** | `HALOGEN_NPU_PORT` | line protocol, not HTTP | `halogen-npu listen` | `all` (when `HALOGEN_NPU_MODELS` set), and the retired `npu` argument |

**The HTTP route table lives entirely on port 8731.** Port 8730 is the engine's
own line protocol (documented in `flash_serve.md`, not here); the front-end
proxies generation to it. This resolves the plan's wording "routes for the API
port 8731 / serve port 8730": only 8731 has HTTP routes.

Mode → server mapping (from `libexec/halogen-entrypoint.sh`):

- `api` — `serve_api.py --tokenizer … --engine 127.0.0.1:8730 --host 0.0.0.0 --port 8731` (entrypoint:1069–1080).
- `all` — starts `flash_serve` on 8730, then `serve_api.py --host 0.0.0.0 --port 8731` (entrypoint:1695–1702); with `HALOGEN_NPU_MODELS` set it also starts `halogen-npu` on 8740 and passes `--npu 127.0.0.1:8740 --npu-tokenizer …` to the front-end (entrypoint:1678–1682, 1701).
- `all` with `_hg_npu_alone=1` + `HALOGEN_NPU_MODELS` — `serve_api.py --engine none --npu 127.0.0.1:8740 …`, which delegates the whole app to `npu_api.serve_only` (serve_api.py:6695–6703, npu_api.py:3399–3492).
- `bench` / `sweep` — same front-end but bound to **127.0.0.1** instead of 0.0.0.0, with `HALOGEN_API_KEY`/`HALOGEN_API_KEY_FILE` unset (entrypoint:1739–1745).
- `convert`, `inspect`, `verify`, `ppl`, `niah` — no HTTP surface.

**GPU gating (plan §5):** every route on 8731 needs a reachable `flash_serve`
engine (ROCm device) except `GET /health` and `GET /api/hello`, which answer
without it (`/health` replies **503** with `status: "engine_unresponsive"`
when the engine does not answer its probe — serve_api.py:4997–4999). The
NPU-mounted routes additionally need the NPU engine (`/dev/accel/accel0`);
when `halogen-npu` is down its routes answer 503 (entrypoint:1634).

---

## 2. Route table (port 8731, `serve_api.py`)

| Method | Path | Handler (line) | Notes |
|---|---|---|---|
| `GET` | `/health` | 4725 | auth-open; 200 when engine live, **503** + same body when not |
| `GET` | `/metrics` | 5001 | Prometheus text exposition, `media_type="text/plain; version=0.0.4; charset=utf-8"` |
| `GET` | `/halogen/handles` | 5032 | **only when `HALOGEN_FORK=1`** |
| `DELETE` | `/halogen/handles/{name}` | 5043 | **only when `HALOGEN_FORK=1`**; 404 `no handle named {name}` |
| `GET` | `/cache` | 5052 | 501 `this engine reports no prompt cache` when unavailable |
| `GET` | `/v1/models` | 5072 | OpenAI model list; appends NPU entries when `--npu` |
| `POST` | `/v1/completions` | 6198 | |
| `POST` | `/v1/chat/completions` | 6220 | |
| `POST` | `/v1/responses` | 6314 | OpenAI Responses API |
| `POST` | `/v1/messages` | 6526 | Anthropic Messages API |
| `POST` | `/v1/messages/count_tokens` | 6573 | Anthropic-compatible |
| `POST` | `/tokenize` | 6590 | llama.cpp-compatible |
| `POST` | `/detokenize` | 6608 | llama.cpp-compatible |
| `GET` | `/props` | 6622 | llama.cpp-compatible |
| `GET` | `/slots` | 6637 | llama.cpp-compatible |
| `GET`/`HEAD` | `/api/hello` | 6642 | 200, empty body; auth-open |

Unknown paths get Starlette's default 404; the `StarletteHTTPException`
handler re-shapes it per §3 (e.g. `GET /v1/chat/completions` → 405
`method_not_allowed` in OpenAI error shape).

### NPU-mounted routes (`npu_api.Front.add_routes`, npu_api.py:3297–3397)

Mounted on the same app when `serve_api.py` runs with `--npu HOST:PORT`
(serve_api.py:6723), and on the standalone `serve_only` app (npu_api.py:3414):

| Method | Path | Request model |
|---|---|---|
| `POST` | `/v1/embeddings` | `EmbReq` (3303) |
| `POST` | `/v1/rerank` | `RerankReq` (3315) |
| `POST` | `/v1/moderations` | `ModReq` (3328) |
| `POST` | `/classify` | `ClsReq` (3339) |
| `POST` | `/v1/systemone` | `S1Req` (3355) |
| `POST` | `/v1/images/generations` | `ImgReq` (3365) |
| `POST` | `/v1/images/edits` | raw `Request` (3383) |
| `POST` | `/v1/images/variations` | raw `Request` (3387) |
| `POST` | `/v1/audio/transcriptions` | raw `Request` (3391) |
| `POST` | `/v1/audio/speech` | raw `Request` (3395) |

Routing rule on the shared routes: a request whose `model` is owned by the
NPU front (`Front.owns`, npu_api.py:1529) is proxied to the NPU engine
(`npu.chat` / `npu.complete`), otherwise it goes to the Flash engine.
`/v1/completions` with a non-`lfm3` NPU model returns 400 `wrong_route`
(6200–6203); `/v1/responses` with an NPU model returns 400 `wrong_route`
(6316–6317); `/v1/messages` with an NPU model returns an **Anthropic-shaped**
400 (6528–6529). The `wrong_route` wording per task (npu_api.py:1564–1594)
is in §6.4.

Standalone NPU-only app (`serve_only`, npu_api.py:3399–3492) exposes:
`GET /health`, `GET /v1/models`, `POST /v1/chat/completions`, all
`add_routes` routes, and `POST /v1/completions` **only when an `lfm3`-style
model is loaded** (3447–3455). Its `/health` returns 503 when the NPU engine
is not ok (3473).

---

## 3. Error contract (exact shapes and wording)

Two error vocabularies, selected by request path
(`wire_error`, serve_api.py:3514–3517): paths starting with `/v1/messages`
answer in Anthropic shape, everything else in OpenAI shape.

### 3.1 OpenAI shape (`oai_error`, 335–338)

```json
{"error": {"message": "<text>", "type": "<type>", "param": null, "code": "<code|null>"}}
```

`ERROR_TYPES` (43–45): `400 invalid_request_error`, `401 authentication_error`,
`403 permission_error`, `404 not_found_error`, `429 rate_limit_error`;
any other status → `server_error`.

### 3.2 Anthropic shape (`anth_error`, 348–354)

```json
{"type": "error", "error": {"type": "<type>", "message": "<text>"}}
```

`ANTH_ERROR_TYPES` (340–343): `400 invalid_request_error`,
`401 authentication_error`, `403 permission_error`, `404 not_found_error`,
`413 request_too_large`, `429 rate_limit_error`, `503 overloaded_error`,
`529 overloaded_error`; any other status → `api_error`.

### 3.3 Framework-level handlers (serve_api.py:3519–3539)

- `StarletteHTTPException` → `wire_error(status, str(detail))`; a 503 gets
  `code: "engine_busy"` on the OpenAI shape.
- Unhandled `Exception` → 500, message
  `internal error: {ExceptionClassName}: {str(exc)[:300]}`, code `server_error`
  (traceback printed to stderr).
- `RequestValidationError` (bad JSON body vs the pydantic model) → 400,
  message = `"; ".join(f"{loc}: {msg}")` over pydantic errors with the
  `body` path prefix stripped (`loc` becomes `body` when empty), or
  `invalid request` when there are no parts.

### 3.4 Recurring exact-wording errors

| Status | Route(s) | Exact message |
|---|---|---|
| 401 | any authed route | `invalid API key` (key present but wrong) / `an API key is required: send \`Authorization: Bearer <key>\` or \`x-api-key: <key>\``; code `invalid_api_key`; response header `www-authenticate: Bearer` (394–431) |
| 503 | generation routes | `timed out waiting for the engine: halogen serves one request at a time (batch-1) and the queue did not clear` (`EngineBusy`, 1330–1335) |
| 400 | `/v1/chat/completions`, `/v1/completions`, `/v1/responses` | `max_tokens {n} exceeds this server's cap of {max_cap}. The cap is server policy, not a model limit: halogen is batch-1, so one long request holds the GPU for its whole run and every other client queues behind it. Raise it with --max-tokens-cap; /health reports the live value as max_tokens_cap.` (5801–5808) |
| 400 | same | `max_tokens {n} is not valid: send a whole number of 1 or more, or -1 for the most this server allows (now {max_cap}).` (`-1` = "fill the context", 5771–5777) |
| 400 | `/v1/messages` | `prompt is too long: {n} tokens > {limit} maximum` (5812–5815); other wires: `the prompt does not fit: it is {n} tokens and the context is {limit}, {-room} tokens over. Shorten the prompt, or raise the context (HALOGEN_CTX, and HALOGEN_ROPE_YARN for more than the native 262144).` (5816–5820) |
| 400 | `/v1/chat/completions` | `continue_final_message cannot be combined with a forced tool choice: one resumes the assistant turn already in the history, the other starts a new call` (6243–6247) |
| 502 | `/v1/chat/completions` | `tool_choice required a call and the model did not name a function: retry, or name the function in tool_choice` (6172–6177) |
| 400 | `/v1/responses` | `previous_response_id is not supported: this server keeps no conversation between requests. Send the conversation so far in \`input\` (the earlier assistant messages, function calls and their outputs) with each request.` (6318–6325; only when `input` itself carries no history — otherwise ignored with a once-only log) |
| 400 | `/v1/responses` | `input is empty: /v1/responses needs at least one non-system turn` (6376–6378) |
| 400 | `/v1/messages` | `messages is empty: /v1/messages needs at least one user turn` (6509–6511) |
| 400 | `/tokenize` | `content is required and must be a string` (6594–6595); `the body is not JSON` / `the body must be a JSON object` (`llama_json`, 3477–3484) |
| 400 | `/detokenize` | `tokens is required: a list of token ids` / `a token id is outside the vocabulary` (6608–6620) |
| 400 | `/v1/messages/count_tokens` | `the body is not JSON` / `the body must be a JSON object` / first 300 chars of the MessagesReq validation error (6573–6588) |
| 404 | `/halogen/handles/{name}` | `no handle named {name}` (5043–5051) |
| 501 | `/cache` | `this engine reports no prompt cache` (5052–5056) |
| 404 | standalone NPU `/v1/chat/completions` | `The model \`{model}\` does not exist or is not loaded here (loaded: {a, b, c})`, code `model_not_found` (npu_api.py:3484–3488) |

### 3.5 Sampling validation (`check_sampling`, 3575–3656) — exact wording

- `temperature={v} is outside 0.0..2.0. Values past the range are NOT clamped, because a clamped request is indistinguishable, from the response, from one that was served as asked.` (same pattern for `top_p` 0.0..1.0 and `min_p` 0.0..1.0)
- `top_k={v} must be >= 0 (0 disables the filter).`
- `logprobs with stream=true is not implemented. The logprob column is collected and returned with the finished response. Send stream=false, or drop logprobs.`
- `top_logprobs={v!r} must be an integer in 0..20.`
- `top_logprobs needs logprobs=true (the OpenAI rule): send both.`
- `top_logprobs` (or `logprobs at temperature 0`) `covers the first generated token only: send max_tokens 1, which is the classifier shape (one forward pass, the next-token distribution). With temperature > 0 and no top_logprobs, every sampled token carries its own logprob.`
- `logprobs with response_format is not supported: the schema's mask picks the token, and an unmasked logprob would not describe that pick. Drop one of the two.`
- `repetition_penalty must be a finite number greater than 0 (1 = off); got {v!r}`
- greedy + penalties: `{field} applies to the sampler, and this request decodes greedy (temperature absent or 0). Send temperature > 0, or drop {field}: a request silently served without it would be indistinguishable from one that honoured it.` (fields: `presence_penalty`, `frequency_penalty`, `repetition_penalty` ≠ 1, `logit_bias`)
- engine without sampling: `this engine build decodes greedy only (a forward pass and an argmax); it does not sample. Omit \`temperature\` or send \`temperature: 0\`. /health reports what this build supports.`
- `response_format (\`text.format\` on /v1/responses) {type!r} cannot be enforced here: the engine's vocabulary handshake failed ({why})` (or `the engine (version {v}) has no constrained decoding; 0.8.0 and later do`) + `. /health reports \`structured_output\`. Omit the field and parse the reply, or prompt for JSON and validate it.`
- drafter: `drafter '{name}' unknown; use serial | mtp | dflash2` (3661–3669; `DRAFTERS = {"serial": 0, "mtp": 1, "dflash2": 2}`, 3301)
- `logit_bias key {k!r} is not a token id` / `logit_bias token id {tid} is outside 0..248319` / `logit_bias has {n} entries, over this server's cap of 20480` (3254–3268)
- `prompt_cache_policy must be one of "default", "no_store" (got {v!r})` (2821–2831)
- `reasoning_effort '{v}' unsupported; use minimal|low|medium|high|xhigh|max|none` (3346–3353; `EFFORT_MAP` maps `minimal→low`, `high→xhigh`, `max→xhigh`, 2410–2416)
- ChatReq validators (pydantic `ValueError`s, surfaced via §3.3):
  `continue_final_message and add_generation_prompt cannot both be true: one continues the final assistant message, the other opens a new one`;
  `continue_final_message needs the last message to be an assistant message to continue; got role '{role}'` (or `no messages`);
  `chat_template_kwargs must be an object`;
  `chat_template_kwargs contains unsupported key(s) {…}; this server honours reasoning_effort, enable_thinking, preserve_thinking, which may also be sent as top-level fields`;
  `conflicting {k}: {top!r} at the top level and {nested!r} in chat_template_kwargs; they are the same control under two names, send one`;
  `conflicting max_thinking_tokens: {cur!r} and {value!r} (as {name}); they are the same control under two names, send one`;
  `{name} must be a whole number >= 1` (for each of `thinking_budget_tokens`, `thinking_budget`, `thinking_token_budget`);
  `reasoning must be an object ({enabled, effort, max_tokens})`; `reasoning.effort must be a string`; `reasoning.max_tokens must be a whole number >= 1`; `reasoning.enabled must be true or false`;
  `thinking.type must be enabled, disabled or adaptive`; `thinking.budget_tokens must be a whole number >= 1`; `thinking must be an object ({type, budget_tokens}) or a level`.
- content parts: `this server has no video path; a \`video\` content part cannot be served` (330–332).

---

## 4. Authentication (`ApiKeyGate`, serve_api.py:356–438)

- Configured by `HALOGEN_API_KEY` (comma-separated keys) **or**
  `HALOGEN_API_KEY_FILE` (file, ≤65536 bytes, ASCII). Both set →
  `SystemExit("serve_api: HALOGEN_API_KEY and HALOGEN_API_KEY_FILE are both set; set one")`.
  Empty/whitespace-only value → `SystemExit` (`… is set but holds no key. Unset it to serve without one; an empty value is refused so that a key that failed to expand cannot leave the port open`).
  Keys must be printable ASCII without spaces
  (`a key may hold only printable ASCII without spaces (a comma separates keys)`).
- Key accepted via `Authorization: Bearer <key>` **or** `x-api-key: <key>`
  (constant-time compare against every configured key).
- Open paths (no key): `/health`, `/api/hello` (`AUTH_OPEN_PATHS`, 356).
  `OPTIONS` requests are never gated.
- With no keys configured, everything is open (no middleware installed).
- Failure response: 401 in the path-appropriate shape (§3.1/§3.2) with
  header `www-authenticate: Bearer`.

---

## 5. Request schemas

### 5.1 `CompletionReq` — `POST /v1/completions` (2833–2855)

| Field | Type | Default | Notes |
|---|---|---|---|
| `model` | str | `MODEL_ID` (`HALOGEN_MODEL_ID`, default `halogen-qwen3.8-flash-next`) | |
| `prompt` | str | `""` | tokenised with `add_special_tokens=False` |
| `prompt_cache_policy` | object | `None` | `"default"` / `"no_store"` (§3.5) |
| `max_tokens` | int | `COMPLETION_MAX_TOKENS` = `HALOGEN_MAX_TOKENS_DEFAULT` or **128** | `-1` = max the server allows |
| `stream` | bool | `false` | |
| `stop` | list[str] \| str \| None | `None` | empty strings dropped |
| `return_progress` | bool \| None | `None` | |
| `stream_options` | dict \| None | `None` | `{"include_usage": true}` adds a final usage frame |
| `drafter` | str \| None | `None` | `serial` \| `mtp` \| `dflash2` |
| `temperature` `top_p` `seed` `top_k` `min_p` `presence_penalty` `frequency_penalty` `repetition_penalty` | see §3.5 | `None` | validated by `check_sampling` |
| `logit_bias` | dict \| None | `None` | keys = token ids (0..248319), ≤20480 entries |
| `logprobs` | int \| bool \| None | `None` | |
| `top_logprobs` | int \| None | `None` | 0..20 |
| `n` | int \| None | `None` | |
| `response_format` | dict \| None | `None` | `json_schema` / `json_object` |

### 5.2 `ChatReq` — `POST /v1/chat/completions` (2860–3027)

All of §5.1's sampling fields except `logprobs` is `bool | None`; plus:

| Field | Type | Default | Notes |
|---|---|---|---|
| `messages` | list[dict] | **required** | |
| `user`, `prompt_cache_key`, `safety_identifier` | str \| None | `None` | recorded, not sent to engine |
| `metadata` | dict \| None | `None` | `session_id`, `user_id` recorded |
| `max_completion_tokens`, `max_output_tokens` | int \| None | `None` | aliases; precedence `max_completion_tokens` > `max_output_tokens` > `max_tokens`; conflicting values log once (`serve_api: the request names two token budgets; used {used}={v}, not …`) |
| `guard` | dict \| str \| bool \| None | `None` | streaming guard; also `x-halogen-guard` header; without npu_guard_stream → 400 `guard: not available in this build` |
| `reasoning_effort` | str \| None | `None` | §3.5 values |
| `max_thinking_tokens` | int \| None | `None` | default `HALOGEN_MAX_THINKING_TOKENS` |
| `enable_thinking`, `preserve_thinking` | bool \| None | `None` | |
| `thinking_budget_tokens`, `thinking_budget`, `thinking_token_budget` | int \| None | `None` | aliases of `max_thinking_tokens` (≥1) |
| `reasoning` | dict \| None | `None` | `{enabled, effort, max_tokens}` folded into the fields above |
| `thinking` | dict \| str \| None | `None` | `{type, budget_tokens}` or a level string |
| `tools` | list[dict] \| None | `None` | |
| `tool_choice` | str \| dict \| None | `None` | `auto`/`none`/`required`/`{type: function, function: {name}}` |
| `parallel_tool_calls` | bool \| None | `None` | |
| `chat_template_kwargs` | dict \| None | `None` | only `reasoning_effort`, `enable_thinking`, `preserve_thinking` (§3.5) |
| `continue_final_message`, `add_generation_prompt` | bool \| None | `None` | §3.5 validators |
| `max_tokens` default | int | `CHAT_MAX_TOKENS` = `HALOGEN_MAX_TOKENS_DEFAULT` or **8192** | |

`n` is **accepted but ignored** on this endpoint (logged once:
`serve_api: 'n' accepted but IGNORED by this endpoint`, 6225–6229;
`SAMPLING_FIELDS = ("n",)`).

### 5.3 `MessagesReq` — `POST /v1/messages` (3029–3045)

`model_config = {"extra": "allow"}` (unknown fields logged once:
`serve_api: /v1/messages ignored the field '{f}'`).

| Field | Type | Default |
|---|---|---|
| `model` | str \| None | `None` |
| `max_tokens` | int | **required** (above `max_cap` it is **clamped to the cap**, not a 400 — logged once: `serve_api: /v1/messages asked for max_tokens {want}, above this server's cap of {max_cap}; on this route such a request runs with {max_cap} (the other routes answer 400). /health reports the cap as max_tokens_cap. Said once.`) |
| `messages` | list | **required** |
| `prompt_cache_policy` | object | `None` |
| `system` | str \| list \| None | `None` |
| `stream` | bool | `false` |
| `tools` | list \| None | `None` (Anthropic-defined tools dropped with a once-only log) |
| `tool_choice` | dict \| None | `None` |
| `thinking` | dict \| None | `None` (`{type: enabled|disabled|adaptive, budget_tokens, display}`; `display != "omitted"` controls whether reasoning is echoed) |
| `output_config` | dict \| None | `None` (`{format: {type: json_schema, schema}, effort}` — `format` maps to `response_format`, `effort` to `reasoning_effort`) |
| `temperature`, `top_p`, `top_k` | float/int \| None | `None` |
| `stop_sequences` | list[str] \| None | `None` |
| `metadata` | dict \| None | `None` |

### 5.4 `ResponsesReq` — `POST /v1/responses` (3047–3072)

`model_config = {"extra": "allow"}`.

| Field | Type | Default |
|---|---|---|
| `model` | str | `MODEL_ID` |
| `prompt_cache_policy` | object | `None` |
| `instructions` | str \| None | `None` |
| `input` | list[dict] \| str | `""` |
| `tools` | list[dict] \| None | `None` (non-function tools dropped with a once-only log; `namespace` tools flattened, collisions renamed `ns__name`) |
| `tool_choice` | str \| dict \| None | `None` (flat `{type: function, name}` normalised to `{type: function, function: {name}}`) |
| `parallel_tool_calls` | bool \| None | `None` |
| `max_output_tokens` | int \| None | `None` (defaults to `CHAT_MAX_TOKENS`) |
| `stream` | bool | `false` |
| `temperature`, `top_p` | float \| None | `None` |
| `drafter` | str \| None | `None` |
| `store`, `previous_response_id` | bool \| None / str \| None | `None` (see §3.4) |
| `include` | list[str] \| None | `None` |
| `reasoning` | dict \| None | `None` (`{summary}` non-empty/≠`none` adds a `summary_text` part) |
| `prompt_cache_key`, `safety_identifier`, `metadata`, `client_metadata`, `service_tier`, `user` | | `None` |
| `text` | dict \| None | `None` (`text.format` → `response_format`) |

### 5.5 NPU request models (npu_api.py:3303–3377)

All use `model_config = ConfigDict(extra="allow")`; every field defaults to
`None`/`""`/`False` as listed:

- `EmbReq`: `model: str = ""`, `input`, `encoding_format`, `dimensions`, `user`.
- `RerankReq`: `model: str = ""`, `query`, `documents`, `top_n`, `return_documents: False`, `instruction`.
- `ModReq`: `model: str = ""`, `input`, `messages`, `strict`.
- `ClsReq`: `model: None`, `input`, `messages`, `truncate_prompt_tokens`, `truncation_side`, `padding`, `add_special_tokens`, `use_activation`, `user`.
- `S1Req`: `model`, `state`, `questions`.
- `ImgReq`: `model`, `prompt`, `n`, `size`, `quality`, `response_format`, `output_format`, `seed`, `steps`, `num_inference_steps`, `user`.

Raw-body routes (`/v1/images/edits`, `/v1/images/variations`,
`/v1/audio/transcriptions`, `/v1/audio/speech`) parse multipart/JSON inside
the handler (audio limits: `ASR_MAX_S = 1200.0`, `ASR_MAX_BYTES = 25 MiB`,
`SPEECH_MAX_CHARS = 4096`, language table `ASR_LANGS`/`ASR_ISO`,
npu_api.py:42–56).

---

## 6. Response schemas

### 6.1 `POST /v1/chat/completions` (non-stream, 6155–6196)

```json
{"id": "chatcmpl-…", "created": <unix_s>, "model": "<MODEL_ID>",
 "object": "chat.completion",
 "choices": [{"index": 0, "finish_reason": "stop|length|tool_calls",
              "message": {"role": "assistant", "content": "<text>",
                          "reasoning_content": "<thinking, when present>",
                          "tool_calls": [ … ] },
              "logprobs": {"content": [ … ] } }],
 "usage": { … }, "timings": { … }}
```

- `finish_reason` becomes `tool_calls` when tool calls were parsed (unless
  `length`); a forced `tool_choice` with no call and `finish != length`
  → 502 (§3.4).
- `usage` (`usage()`, 4061–4077): `prompt_tokens`, `completion_tokens`,
  `total_tokens`; `prompt_tokens_details.cached_tokens` when non-zero;
  `completion_tokens_details.reasoning_tokens` (+ `reasoning_closed_at`,
  `reasoning_closed_by`), `end_of_turn_kept`, `tool_call_from_reasoning`.
- `timings` (`timings()`, 4090–4120+): `prompt_n`, `predicted_n`,
  `prompt_ms`, `predicted_ms`, `prompt_per_second`, `predicted_per_second`,
  optional `cache_n`, `disk_restore_n`, `disk_restore_ms`, `prefix_n`,
  `max_tokens_clamped_from`/`_to`, drafter round counters.
- **Normalisation note (plan §5):** `created`, timings and ids vary per run;
  contract tests compare structure, not values.

### 6.2 `POST /v1/completions` (non-stream)

Same envelope with `"object": "text_completion"` and
`choices[0].text` instead of `message` (6183–6196).

### 6.3 `POST /v1/responses` (non-stream, 6102–6154)

```json
{"id": "resp_…", "object": "response", "created_at": <unix_s>,
 "status": "completed|incomplete", "model": "<MODEL_ID>",
 "output": [ {"id": "rs_…", "type": "reasoning", "status": "completed",
              "summary": [ … ], "content": [{"type": "reasoning_text", …}],
              "encrypted_content": null},
            {"id": "msg_…", "type": "message", "role": "assistant",
             "status": "completed",
             "content": [{"type": "output_text", "text": …, "annotations": []}]},
            {"id": "fc_…", "type": "function_call",
             "status": "completed|incomplete", "name": …, "arguments": …,
             "call_id": …, "namespace": … } ],
 "parallel_tool_calls": <bool>, "max_output_tokens": <int>,
 "usage": { … }, "timings": { … },
 "incomplete_details": {"reason": "max_output_tokens"} }
```

`usage` here is `responses_usage` (4079–4088): `input_tokens`,
`output_tokens`, `total_tokens`, `input_tokens_details`,
`output_tokens_details`. `status` is `incomplete` iff `finish == length`.

### 6.4 `POST /v1/messages` (non-stream, 6097–6101)

```json
{"id": "msg_…", "type": "message", "role": "assistant",
 "model": "<MODEL_ID>",
 "content": [ {"type": "text", …}, {"type": "tool_use", "id": …, "name": …,
                                        "input": { … } } ],
 "stop_reason": "end_turn|max_tokens|tool_use|stop_sequence",
 "stop_sequence": <string|null>,
 "usage": {"input_tokens": …, "output_tokens": …,
           "cache_read_input_tokens": …, "cache_creation_input_tokens": 0},
 "timings": { … }}
```

(`messages_stop` 4360–4367, `messages_usage` 4353–4358, `tool_input` parses
arguments JSON or `{}`.)

### 6.5 Streaming (`stream: true`)

- OpenAI wires (`sse`, 4254–4314): `media_type="text/event-stream"` (6014).
  Frames: `data: {"id","created","model","object":"chat.completion.chunk"|"text_completion","choices":[{"index":0,"finish_reason":<null|final>,"delta":{…}}],"usage":null}\n\n`;
  first chat frame carries `{"role":"assistant","content":""}`; tool-call
  deltas streamed; final frame carries `finish_reason` + `timings`; with
  `stream_options.include_usage` an extra `{"choices":[],"usage":{…},"timings":{…}}`
  frame; stream ends `data: [DONE]\n\n`. Mid-stream errors emit
  `data: {"error": {…oai error object…}}` then `[DONE]`.
- Anthropic wire (`sse_messages`, 4376+): named events
  `event: message_start\ndata: {"type":"message_start","message":{…}}\n\n`,
  content-block starts/deltas/stops, `message_delta`/`message_stop`.
- NPU routes stream their own SSE per handler.

### 6.6 `GET /health` (4725–4999)

200 when the engine answers its probe, **503 with the same body** when not
(`status: "engine_unresponsive"`). Body keys (values dynamic): `status`,
`model`, `endpoints` (sorted `/v1/*` paths), `cache_counters`, `metrics`,
`context`, `capability_probe`, `rope_scaling`, `indexer_budget`, `vision`
(`{enabled, endpoints, input, max_pixels, size_multiple, disabled_because?}`),
`version` (`{api, engine, match}`), `modes` (`HALOGEN_IMAGE_MODES` split),
`checkpoint_format`, `chat_template`, `engine` (`{responds, probe_s, detail?}`),
`busy`, `slots`, `slot_ctx`, `kv_pool_positions`, `kv_cache`, `in_flight`,
`busy_for_s`, `queued`, `longest_wait_s`, optional `admission`, `decode`,
`drafters_available`, `drafter_default`, `prompt_lookup`,
`shortlist_draft_head`, `drafter_weights_loaded`, `prompt_cache`,
`composable_context`, `tool_calls`, `supported`, `token_budget_aliases`,
`max_tokens_default`, `reasoning_effort_default`, `reasoning_effort_values`,
`token_budget_covers_reasoning`, `max_thinking_tokens_default`,
`thinking_answer_room`, `end_of_turn_guard`, `answer_guard`, `stop_ids`,
`thinking_control_aliases`, `server_defaults`, `server_defaults_rule`,
`chat_template_kwargs`, `error_format: "openai"`, `accepted_but_ignored`,
`not_implemented`, `structured_output` (large static contract incl.
`strict_tools`), `sampling`, `max_tokens_cap`, `max_tokens_over_cap: "400"`;
plus `npu` (when `--npu`) and `request_records` (when `HALOGEN_RECORD_DIR`
set). The full literal block is at serve_api.py:4725–4999 — the Rust port
must reproduce this object field-for-field given the same engine INFO.

### 6.7 `GET /metrics` (5001–5029)

Prometheus text (0.0.4). Engine metrics plus front-end extras:
`halogen:engine_up`, `halogen:busy_for_seconds`, `halogen:queue_waited_total`,
`halogen:queue_wait_seconds_total`, `halogen:queue_wait_longest_seconds`,
`halogen:queue_gave_up_total`, `halogen:build_info{version,engine_version,model}`
(plus cache metric rows).

### 6.8 `GET /v1/models` (5072–5090)

```json
{"object": "list", "has_more": false, "first_id": "<MODEL_ID>",
 "last_id": "<MODEL_ID>",
 "data": [{"id": "<MODEL_ID>", "object": "model", "type": "model",
           "display_name": "<MODEL_ID>", "created_at": "1970-01-01T00:00:00Z",
           "owned_by": "halogen", "created": 0, "max_model_len": <ctx>,
           "context_length": <ctx>, "meta": {"n_ctx_train": …, "n_ctx": …},
           "max_tokens_cap": <cap>, "max_tokens_default": <default>,
           "architecture": {"input_modalities": ["text"(,"image")],
                            "output_modalities": ["text"]}}]}
```

NPU entries appended as
`{"id", "object": "model", "type": "model", "display_name", "created_at",
"owned_by": "halogen", "created": 0, "max_model_len", "context_length",
"backend": "npu", "task", "source"}` (npu_api.py:1532–1538).

### 6.9 llama.cpp-compatible endpoints

- `GET /props` (6622–6635): `{model_alias, total_slots,
  default_generation_settings: {n_ctx, params: {n_predict}}, chat_template,
  modalities: {vision}, build_info: "halogen <version>" (or `"halogen dev"`),
  is_sleeping: false}`.
- `GET /slots` (6637–6640): `[{"id": i, "n_ctx": n_ctx}, …]` for each slot.
- `POST /tokenize` (6590–6604): body `{content, add_special=false,
  parse_special=true, with_pieces=false}` → `{"tokens": [ids…]}` or
  `{"tokens": [{"id", "piece"}, …]}` with `with_pieces`.
- `POST /detokenize` (6608–6620): body `{tokens: [ids…]}` →
  `{"content": "<text>"}` (decoded with `skip_special_tokens=False`,
  `clean_up_tokenization_spaces=False`).
- `GET|HEAD /api/hello` (6642–6644): 200, empty body.
- `GET /cache` (5052–5070): engine cache stats + `hit_rate`,
  `token_hit_rate`, `pool.usage_ratio`, `snapshot_places` (when
  `cache_mode ≥ 2`).
- `GET /halogen/handles` (5032–5041, `HALOGEN_FORK=1` only):
  `{handles: [{name, tokens, idle_s, age_s, map?}], cap, ttl_s}`;
  `DELETE /halogen/handles/{name}` → `{"released": name}`.

---

## 7. Headers

**Set by the server:**

- `Content-Type: application/json` (all JSON routes; FastAPI default),
  `text/plain; version=0.0.4; charset=utf-8` (`/metrics`),
  `text/event-stream` (SSE).
- `www-authenticate: Bearer` on 401 responses (428).
- `X-Halogen-From: used|unmatched|unknown` and `X-Halogen-From-Ids: same|differs`
  on chat/messages/responses replies when `HALOGEN_FORK=1` (5726–5735).
- Server software header comes from uvicorn (`server: uvicorn`) — covered by
  the harness normalisation, not by this spec.

**Read by the server:**

- `Authorization: Bearer …`, `x-api-key` (auth, §4).
- `X-Halogen-Priority` — `1|true|high|yes` marks a priority request (1106–1113).
- `X-Halogen-Keep` — handle name, 1–64 chars `[A-Za-z0-9_.:-]`; invalid →
  400 `X-Halogen-Keep takes a name of 1 to 64 letters, digits or _.:-`
  (5710–5713). `X-Halogen-From` pairs with it (fork feature, `HALOGEN_FORK=1`).
- `x-claude-code-session-id`, `x-claude-code-agent-id`,
  `x-claude-code-parent-agent-id` — fork mapping (1119–1121).
- `x-halogen-guard` — streaming guard spec (5744, chat route).
- `Upgrade: websocket`/`h2c` offers are logged once per protocol and answered
  over HTTP/1.1 (`UpgradeNote` middleware, 899–931; no response change).

---

## 8. Server defaults (env → behaviour)

`DEFAULTS` (serve_api.py:2466–2482) — each validated at startup with
`SystemExit("serve_api: {NAME}={raw!r} is not a number")` /
`… is outside {lo}..{hi}; values are not clamped, fix the variable`:

| Env var | Range | Feeds |
|---|---|---|
| `HALOGEN_MAX_TOKENS_DEFAULT` | int 1..2³⁰ | `ChatReq.max_tokens` (else 8192), `CompletionReq.max_tokens` (else 128); > `--max-tokens-cap` → SystemExit (6680–6685) |
| `HALOGEN_TEMPERATURE` | float 0..2 | sampling default |
| `HALOGEN_TOP_P` | float 0..1 | filter default |
| `HALOGEN_TOP_K` | int 0..2³⁰ | filter default (else model's `top_k: 20`) |
| `HALOGEN_MIN_P` | float 0..1 | |
| `HALOGEN_PRESENCE_PENALTY` / `HALOGEN_FREQUENCY_PENALTY` | float −2..2 | |
| `HALOGEN_REPETITION_PENALTY` | float >0 | |
| `HALOGEN_REASONING_EFFORT` | effort values | |
| `HALOGEN_ENABLE_THINKING` | bool | |
| `HALOGEN_MAX_THINKING_TOKENS` | int 1..2³⁰ | |
| `HALOGEN_THINKING_ANSWER_ROOM` | int 0..2³⁰ | else `max(1024, 15% of max_tokens)` |

Other front-end env (from the makeWrapper, Appendix A): `HALOGEN_MODEL_ID`,
`HALOGEN_IMAGE_VERSION` (→ `API_VERSION`), `HALOGEN_STRICT_TOOLS`,
`HALOGEN_SCHEMA_ESCAPE`, `HALOGEN_CACHE_SNAP3`, `HALOGEN_CACHE_LEAD`,
`HALOGEN_VISION_MAX_PIXELS`, `HALOGEN_FORK`, `HALOGEN_TOKENIZE_REUSE`,
`HALOGEN_KEEPALIVE_TIMEOUT` (uvicorn `timeout_keep_alive`, default 300),
`HALOGEN_ACCESS_LOG` (`quiet|all|off`, quiet hides `/health` and `/metrics`),
`HALOGEN_EOS_GUARD`, `HALOGEN_ANSWER_GUARD`, `HALOGEN_TEMPLATE_CHECK`,
`HALOGEN_TEMPLATE_UNCHECKED`, `HALOGEN_CHAT_TEMPLATE`, `HALOGEN_TOOL_DEFER`,
`HALOGEN_RECORD_*` (request records), `HALOGEN_NPU_STALL_S` (npu_api).
The full launcher table is in `launcher.md`; the front-end-only subset is
also listed there as `FE_ONLY_FLAGS` (entrypoint:1002–1015).

---

## 9. CLI of the front-end (for completeness; owned by `launcher.md`)

`serve_api.py --tokenizer DIR --engine HOST:PORT|none [--npu HOST:PORT]
[--npu-tokenizer DIR]… [--host 0.0.0.0] [--port 8731] [--context 32768]
[--queue-timeout 3600] [--max-tokens-cap 65536] [--check-defaults]`
(6648–6678). `--check-defaults` prints
`serve_api: request defaults …` (or `none set (built-in)`), runs the template
check and exits 0 (6686–6693).

---

## 10. Ambiguities to escalate

1. **Port 8730 is not HTTP.** The plan's §4 wording ("routes … serve port
   8730") maps to the engine's line protocol, not HTTP routes. This spec
   documents the HTTP surface (8731 + NPU mount); the 8730 protocol belongs
   to `flash_serve.md`. Confirm the orchestrator agrees with this split.
2. **`/health` body fidelity.** The body is ~270 lines of literals mixed with
   engine-reported values (serve_api.py:4725–4999). The Rust port must
   reproduce the literal structure exactly; the contract test needs a fixed
   fake engine INFO to compare against. Decide whether the harness stubs the
   engine or requires a live one (GPU-gated per §5).
3. **`request_records` side-channel.** When `HALOGEN_RECORD_DIR` is set,
   `/health` gains a `request_records` section and responses are recorded;
   its schema lives in `request_records.py` (721 lines) and is only
   summarised here. Escalate if the Phase 3 matrix must cover it.
4. **NPU route bodies.** The NPU handlers (embeddings/rerank/images/audio)
   are 3.5k lines of per-task logic; this spec fixes their routes and request
   models, but response bodies are transcribed only at the level the plan's
   HTTP contract tests need. Full per-route response schemas can be expanded
   if the matrix requires them.
5. **`/v1/messages` cap clamping vs 400 elsewhere** (6440–6451) is an
   intentional asymmetry ("on this route such a request runs with {max_cap}
   (the other routes answer 400)") — flagging so the Rust port doesn't
   "fix" it.
6. **Bind address differs by mode** (`0.0.0.0` for `api`/`all`,
   `127.0.0.1` for `bench`/`sweep`) — the harness should bind ephemeral
   ports anyway (plan §5 normalisation covers port numbers).
