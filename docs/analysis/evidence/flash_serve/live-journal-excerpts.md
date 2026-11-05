# flash_serve — live journal excerpts (host missingno, 2026-11-06)

Archived so the port's log-line implementations stay verifiable
(review 💡-3): `cc-flash-serve`'s `generated_line` test asserts the
reference's exact journal wording (`req 2 generated 1748 tokens, 30.0 s`
/ `3383 tokens, 60.0 s`), and the `GEN_PROGRESS_INTERVAL_SECS` constant
(review 🟡-1) is pinned by this journal — not by the decompiled `main`,
whose double `0x403e000000000000` is the `HALOGEN_ADMIT_AGE_S`
admission-age default (`local_19a0`), not a generation-log interval.

## Live reference journal excerpt

Source: systemd journal of the reference `halogen-flash-server` on host
`missingno`, captured 2026-11-06 (the user's 06:39 session).

```
Nov 06 06:39:01 missingno halogen-flash-server[13749]: INFO:     127.0.0.1:53816 - "POST /v1/chat/completions HTTP/1.1" 200 OK
Nov 06 06:39:33 missingno halogen-flash-server[13687]: flash_serve: req 2 generated 1748 tokens, 30.0 s
Nov 06 06:40:03 missingno halogen-flash-server[13687]: flash_serve: req 2 generated 3383 tokens, 60.0 s
```

## Cold-cathode run excerpt (capability-probe failure sequence)

Source: the user's cold-cathode run on host `missingno`, 2026-11-06 (the
same 06:39 session; the run whose startup probe failure commit 03b6591
fixed — the launcher reported the engine listening while serve_api's
capability probe timed out 3 × 5 s).

```
halogen: backend: cpu
halogen: loading checkpoint /srv/halogen-models/qwen38-flash-next-v2.hgn
halogen: 2015 tensors, 66687351992 bytes mapped
halogen: checkpoint resident, binding 127.0.0.1:8730
halogen: engine listening after 2s
serve_api: chat template /srv/halogen-models/tokenizer/chat_template.jinja sha256 c3cf9e34abf4f9e3, thinking control probe passed
serve_api: chat template cache check passed: the history is reusable in 6 of 6 turn shapes and the model's own reply in 4 (6 ms)
/nix/store/fn6y2w20kbgjd8da8pwxbbc3gvmiycpy-halogen-flash-server-0.18.21/share/halogen/tools/serve_api.py:6727: DeprecationWarning:
        on_event is deprecated, use lifespan event handlers instead.

        Read more about it in the
        [FastAPI docs for Lifespan Events](https://fastapi.tiangolo.com/advanced/events/).

  @app.on_event("startup")
INFO:     Started server process [11943]
INFO:     Waiting for application startup.
serve_api: capability probe FAILED: the engine did not answer INFO in 3 x 5 s.
```
