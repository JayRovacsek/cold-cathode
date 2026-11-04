# flash_serve engine-internals spec (full port blueprint)

Scope: the engine internals of `halogen-flash-server` v0.18.21
(`flash_serve`, sha256 `7c92fe3f8ac544c4fb690a69e9ce359fa7b8d453a8eab8e9a1289c6f2309c012`)
that the full port of `cc-flash-serve` must reproduce. This complements —
and does not restate — the frozen CLI contract in
[`flash_serve.md`](flash_serve.md); where the frozen spec inferred a
behaviour and this pass verified it, the correction is noted inline.

Provenance markers used throughout:

- **[RV]** — runtime-verified: observed by executing the reference binary
  (see [`evidence/flash_serve/cli-probes.txt`](../evidence/flash_serve/cli-probes.txt)
  and [`evidence/flash_serve/engine/probes.txt`](../evidence/flash_serve/engine/probes.txt)),
  or read from the shipped `serve_api.py`
  (`/nix/store/fn6y2w20kbgjd8da8pwxbbc3gvmiycpy-halogen-flash-server-0.18.21/share/halogen/tools/serve_api.py`,
  the reference front-end that speaks the engine protocol), or read from
  `binutils-strings.txt` (a string is runtime-visible even when its
  consumer is not decompiled).
- **[D]** — decompile-derived: read from Ghidra pseudocode of the
  reference binary (committed under
  [`evidence/flash_serve/decompile/`](../evidence/flash_serve/decompile/)
  and [`evidence/flash_serve/engine/`](../evidence/flash_serve/engine/)).

Reference binary paths in function addresses use Ghidra addresses
(`FUN_xxxxxxxx`); file offsets are `address − 0x100000` for the first
`PT_LOAD` (verified by resolving string operands of `bcmp`/`fwrite` calls
against the raw binary bytes).

## 1. `.hgn` weight layout

- **[D]** Reference header (from `FUN_020be0a0`, newly decompiled,
  [engine/fun-020be0a0.json](../evidence/flash_serve/engine/fun-020be0a0.json)):
  `pread(fd, buf, 0x68, 0)` — a **104-byte header**; magic bytes
  `0x314e4748` = `HGN1` (exact compare); `u64` entry_count @8 (must be
  ≤ `0x100000`); `u64` table_offset @16 (the entry table is read from this
  offset). The port's header
  ([`crates/cc-halogen-tools/src/hgn.rs`](../../crates/cc-halogen-tools/src/hgn.rs):
  `HGN1` + `u32` version @4 + `u64` entry_count @8 + `u64` table_offset
  @16 + 2 reserved `u64`s + 64-byte `model_id`) matches on magic,
  entry_count and table_offset.
- **[D]** **New contradiction — entry layout:** `FUN_020be0a0` iterates
  entries with stride `0xa0` (160 B, matching the port) but reads the name
  via `strnlen(entry, 0x60)` (96-byte name field), the dtype as `u32`
  @**96** and a second `u32` @**100** (queried values: dtype `0x10`,
  kind `2`), whereas the port's `hgn.rs` reads `name[0..32]`, `u32` dtype
  @32, `u64` nbytes @40. The port's entry layout (derived from the
  `halogen-tools` binary) does not match flash_serve's loader.
- **[D]** **New contradiction — version gate:** the reference accepts only
  versions where `(uint)(version − 3) > 0xfffffffd`, i.e. version ∈ {1, 2}
  (Ghidra renders this from the header's `u32` @4; type-propagation warning
  present, so re-verify). The port writes version **7** — the likely
  mechanical cause of the review-F3 rejection of subject-written `.hgn`
  files.
- **[RV]** Review F3 caveat (carried from the frozen spec): the reference
  **rejects** a subject-written `.hgn` (`--resident-gib sub.hgn` → exit 1,
  no output).
- **[D]** The reference's checkpoint-loading functions:
  - `FUN_020bd2d0(path, out)` — parse a checkpoint into an in-memory
    descriptor (called by the resident-gib path,
    [`mode-dispatcher-FUN_01f23680.json`](../evidence/flash_serve/decompile/mode-dispatcher-FUN_01f23680.json);
    decompile pending).
  - `FUN_020bd7d0(out, descriptor, flag)` — build the tensor-name list: a
    vector of 5-qword entries `{ name_ptr, name_len, … }` (shape inferred
    from its callers `FUN_020bd680`/`FUN_020bdc70`).
  - `FUN_020be4e0(descriptor, …) → double` — resident-weight GiB:
    decompiled this pass,
    [engine/fun-020be4e0.json](../evidence/flash_serve/engine/fun-020be4e0.json);
    walks the tensor name→size map and honours `HALOGEN_CK_OVERLAY_SKIP`
    (string-verified in `binutils-strings.txt`; delimiter not yet
    extracted) when computing the total.
  - `FUN_020bea30` — engine construction step fed by the checkpoint
    (see §3): the mmap/registration loader (source path
    `/src/halogen/src/checkpoint.cpp`), staged in
    [engine/fun-020bea30.json](../evidence/flash_serve/engine/fun-020bea30.json).
    Its error set completes the header validation picture:
    `checkpoint: bad magic in %s`, `checkpoint: unsupported version in %s`,
    `checkpoint: truncated file %s (size mismatch)`, `checkpoint: entry
    out of bounds in %s`, `checkpoint: cannot open %s`, `checkpoint:
    fstat failed on %s`, `checkpoint: mmap failed on %s`, `checkpoint: %s:
    an HT tensor whose shape and size do not match`, `checkpoint: %s: an
    I4R tensor whose group, shape or size this engine does not read`;
    logging `checkpoint: mapped %.1f GB, NOT pinned (Pin::None)` /
    `checkpoint: registered %.1f GB in %.1f s (%.1f GB/s, %s)%s`; reads
    `HALOGEN_WEIGHTS_LOCK`.
- **[RV]** The engine distinguishes three checkpoint classes — the INFO
  `checkpoint_format` token ∈ {`hgn`, `gguf`, `gguf-cache`} (frozen spec
  flag table) — and logs `checkpoint: GGUF ready in %.1f s: %zu tensors
  (%zu trunk, 1 table, %zu head)` (
  [`binutils-strings.txt:3735`](../evidence/flash_serve/binutils-strings.txt)),
  i.e. tensors are grouped into *trunk*, *table* and *head* sections.
- **[RV]** Quant formats named in strings: `W4B`, `I4R`, `Q8S32`, `Q6G64`,
  `Q4C-P` (e.g. `mtp.layers.0.experts: the head's experts must be Q4C-P
  (got %s / %s)`,
  [`binutils-strings.txt:1488`](../evidence/flash_serve/binutils-strings.txt)).
- **[D]** Dtype code → name resolves through a table at `PTR_DAT_020baa68`
  whose full contents are unverified; only `0=f32`, `1=f16`, `2=bf16` are
  confirmed (port `dtype_name`, hgn.rs:216–224). Dtype code `0x10` exists
  in real checkpoints (queried by `FUN_020bdc70`, §7) but its name is a
  documented gap — revisit by dumping the table at `PTR_DAT_020baa68`.
- **[RV]** The MTP head is a *separate* `.hgn` file —
  `qwen38-flash-next-mtp.hgn` from the weights repo, named by
  `HALOGEN_MTP_HEAD`; a llama.cpp MTP draft GGUF is not read
  ([`binutils-strings.txt:957`](../evidence/flash_serve/binutils-strings.txt),
  [:1997](../evidence/flash_serve/binutils-strings.txt)).
- **[D]** GGUF path: the port's
  [`crates/cc-flash-serve/src/gguf.rs`](../../crates/cc-flash-serve/src/gguf.rs)
  validates the `qwen4exp` metadata schema (§2) before the engine accepts a
  GGUF trunk; `gguf-cache` is the serialised form of a loaded checkpoint
  (INFO token set above).

## 2. Model architecture constants

- **[RV]** Architecture string `qwen4exp` (Qwen3.8-Flash-Next); the engine
  reads `qwen4exp` only: `gguf: architecture '{arch}': this engine reads
  qwen4exp (Qwen3.8-Flash-Next) only` (
  [gguf.rs:109](../../crates/cc-flash-serve/src/gguf.rs), string-verified).
- **[RV]** Required scalar metadata and their fixed values (probed against
  the reference via `qwen4exp-minimal.gguf`,
  [gguf.rs:134–139](../../crates/cc-flash-serve/src/gguf.rs)):
  | key | value |
  |---|---|
  | `qwen4exp.block_count` | 48 |
  | `qwen4exp.embedding_length` | 2560 |
  | `qwen4exp.expert_count` | 512 |
  | `qwen4exp.ssm.group_count` | 16 |
  | `qwen4exp.ssm.time_step_rank` | 48 |
  | `qwen4exp.ssm.state_size` | 128 |
- **[RV]** Required array metadata (presence-only check):
  `qwen4exp.ple.layer_multipliers`, `qwen4exp.ple.head_offsets`,
  `qwen4exp.ple.head_vocab_sizes`
  ([gguf.rs:145–147](../../crates/cc-flash-serve/src/gguf.rs)).
- **[RV]** Vocab size 248320 (frozen spec INFO table;
  [flash_serve.md](flash_serve.md)).
- **[RV]** Native context 262144 (`0x40000`): `HALOGEN_CTX` default
  (launcher.md:141) and serve usage `[--ctx N]`
  ([engine/probes.txt](../evidence/flash_serve/engine/probes.txt)).
- **[D]** Context extension beyond native: YaRN scaling computed by
  `FUN_02018ae0`; max context `min(factor · 262144, 1048576)`; the error
  renders the allowed factors literally (`"(4 for 1048576, 2 for 524288)"`,
  [ctx-validator-FUN_01f24220.json](../evidence/flash_serve/decompile/ctx-validator-FUN_01f24220.json)).
- **[D]** Attention head count, head dimension and GQA ratio are **not yet
  extracted** — documented gap. Revisit: the model-config constructor
  (reachable from the engine ctor chain, §3) and the attention kernel names
  `k_attn_fa2`, `k_attn_qs`, `k_attn_fd`, `k_rope_blocks`,
  `k_rope_w_mrope` (mrope ⇒ multi-resolution RoPE variants) in
  `binutils-strings.txt`.

## 3. Forward pass

- **[RV]** Per-layer stage timing template (verbose progress):
  `norm %.2f s  qkv %.2f s  rope %.2f s  ATTN %.2f s (%.1f%%, %.2f TFLOPS)
  proj %.2f s  mlp %.2f s` — the forward pass per layer is
  **norm → qkv → rope → attention → projection → mlp**
  (`binutils-strings.txt`, quoted in
  [flash_serve.md](flash_serve.md)).
- **[RV]** Hybrid layer composition, from tensor-name strings:
  `self_attn` with per-head `q_norm`/`k_norm`; `linear_attn` with `conv`,
  `in_proj_z`, `out_proj` (gated linear-attention branch); MoE with a
  router, 512 `experts` and a `shared_expert`; PLE ngram heads
  (`qwen4exp.ple.*`, §2); `hyper_connection_mixer` (also present in the
  MTP head, §4).
- **[D]** Engine construction chain from `main`
  ([main-FUN_01ede7e0.json](../evidence/flash_serve/decompile/main-FUN_01ede7e0.json)):
  `FUN_01f24460(ck, pool, ctx, max_tok, slots, pooled)` → `FUN_020bea30` →
  `FUN_01f24580` → `FUN_01f24690` (checkpoint → engine object →
  runtime init). `FUN_020bea30` decompile pending (batch in progress).
- **[D]** Startup phases announced via `FUN_01f76330`:
  `preparing weights` → `weights ready; reserving working memory` →
  `opening the socket` → `model ready`
  ([main-FUN_01ede7e0.json](../evidence/flash_serve/decompile/main-FUN_01ede7e0.json)),
  plus `preparing composable
  context` when the composable-context subsystem is on (§5).
- **[D]** `FUN_01f24460` (ctor step 1, staged in
  [engine/](../evidence/flash_serve/engine/)) resolves the checkpoint
  size: if `FUN_020c4aa0()` returns > 0 (a known GGUF) it prints
  `kv pool: %s is a GGUF; %.1f GiB of resident weights once repacked
  (plus the head)`; otherwise it parses via `FUN_020bd2d0` +
  `FUN_020be4e0` (same pair as the resident-gib query, §7), then calls
  `FUN_02019610(gib, keep_trunk, pool, ctx, max_tok, slots, ngram_flag)`
  for the pooled-KV setup when the model is pooled.
- **[RV]** GPU execution: 9 AMDGPU code objects embedded in the binary
  ([binutils-elf-info.txt](../evidence/flash_serve/binutils-elf-info.txt));
  kernel symbol names in `binutils-strings.txt` cover attention
  (`k_attn_*`), RoPE (`k_rope_*`), sampling (§6) and argmax.
- **[D]** The serve banner summarises the configured engine:
  `flash_serve: listening on %s:%d, %d slots over ONE %lld-position KV
  pool, each request up to %d (it reserves prompt + max_tokens), max_tok
  %d, BATCHED greedy or sampled; MTP depth-%d%s %s`
  ([binutils-strings.txt:3753](../evidence/flash_serve/binutils-strings.txt)).

## 4. MTP head (multi-token prediction)

- **[RV]** Head tensor names (all in
  `binutils-strings.txt`; line refs):
  `mtp.pre_fc_norm_embedding.weight` (:355), `mtp.pre_fc_norm_hidden.weight`
  (:745), `mtp.fc_embedding.weight` (:621 — via `mtp.fc_hidden.weight`
  :621 pair), `mtp.fc_hidden.weight` (:621),
  `mtp.hyper_connection_mixer.hc_norm.weight` (:356),
  `mtp.hyper_connection_mixer.input_mix_weight_up.weight` (:492),
  `mtp.hyper_connection_mixer.input_mix_weight_down.weight` (:2703),
  and the `mtp.layers.0.` prefix (:2284).
- **[RV]** Head validation strings: `the MTP head file is not an .hgn (a
  llama.cpp MTP draft GGUF is not read; the engine's own head is
  qwen38-flash-next-mtp.hgn from the weights repo, HALOGEN_MTP_HEAD names
  it)` (:957); `: the MTP head file carries no mtp.* tensors` (:1593);
  `mtp.layers.0.experts: the head's experts must be Q4C-P (got %s / %s)`
  (:1488).
- **[D]** `FUN_01f24c10`/`FUN_01f24d00` (previously guessed as MTP-head
  loaders) are actually the **composable-context** config ctor and
  initialiser (see §5); the MTP-head loader function remains a gap —
  revisit via `FUN_01f24580`/`FUN_01f24690` (the remaining ctor chain
  steps).
- **[RV]** `HALOGEN_MTP=0` turns the head off (frozen spec env table); the
  connection handler refuses a drafter without it:
  `asked for drafter ` + ` (MTP head not staged, HALOGEN_MTP=1)`
  ([conn-handler-FUN_01ee0e20.json](../evidence/flash_serve/engine/conn-handler-FUN_01ee0e20.json)).
- **[D]** **Contradiction (must resolve before the port ships):** the
  frozen CLI spec documents `HALOGEN_MTP_DEPTH` default **3**, but the
  mode-dispatcher decompile
  ([mode-dispatcher-FUN_01f23680.json](../evidence/flash_serve/decompile/mode-dispatcher-FUN_01f23680.json))
  shows default **DEPTH=2** (with the `HI` default 3 only when `DEPTH` is
  unset). The port currently follows the frozen spec (3).
- **[D]** Related defaults (same decompile): `SAMPLED_DEPTH` 7,
  `MTP_GATE_P` 0.9, PLD (prompt-lookup decoding) `N=3, K=3`, PLD adapt
  `W=32, floor=1.0, R=64`.
- **[RV]** `spec_verify(n=%d): needs the verify rings (HALOGEN_MTP=1 or a
  lookup / grammar drafter), 1 <= n <= %d (HALOGEN_SPEC_N) and a warm
  sequence (pos %d)` (frozen spec validation wording).
- **[RV]** The serve banner embeds the active depth: `MTP depth-%d%s %s`
  (:3753, §3).

## 5. KV cache

- **[RV]** Cost model: `HALOGEN_KV_POOL_POSITIONS … the pool, about 29.5
  KiB a position (262144 = 7.2 GiB); it cannot go below one context`
  ([binutils-strings.txt:287](../evidence/flash_serve/binutils-strings.txt)).
- **[D]** Pool cap `16777216` (`0x1000000`) positions, allocated in
  256-position units (ctx-validator decompile,
  [ctx-validator-FUN_01f24220.json](../evidence/flash_serve/decompile/ctx-validator-FUN_01f24220.json));
  matches the launcher derivation `min(max(ctx·2, ctx), 1048576)`
  (launcher.md:142–144).
- **[RV]** Slot allocator errors: `slots_alloc(%d) on a pooled model: use
  slots_alloc_pooled` (:1782), `slots_alloc_pooled(%d): 1 <= B <= %d`
  (:1783), `slots_alloc_pooled(%d): %d slots already hold private KVs`
  (:1784).
- **[RV]** One shared pool per engine, per-request reservation:
  `%d slots over ONE %lld-position KV pool, each request up to %d (it
  reserves prompt + max_tokens)` (:3753, §3); `HALOGEN_KV_POOL_FIT` env
  (frozen spec env table).
- **[D]** Admission/waiting logic (connection handler,
  [conn-handler-FUN_01ee0e20.json](../evidence/flash_serve/engine/conn-handler-FUN_01ee0e20.json)):
  `req %ld waits: no room for %d positions`; on admission
  `req %ld admitted: slot %d, region base %d cap %d, prompt %d, drafter
  %d, %s`; kept-state accounting
  `; %d held positions are kept states'%s, freed when their handle is
  released or idle past HALOGEN_FORK_TTL_S` (and the variant
  ` (%d of them only auto keeps')`).
- **[RV]** Region/busy/held/free-gap status strings and the `HREL`
  (handle-release) verb exist (§8); frozen spec documents the
  `HALOGEN_FORK`/`HALOGEN_FORK_TTL_S` front-end pair.
- **[D]** **Composable context** subsystem (`FUN_01f24c10` config ctor +
  `FUN_01f24d00` initialiser, both staged in
  [engine/](../evidence/flash_serve/engine/)): enabled by
  `HALOGEN_COMPOSABLE_CONTEXT` (default off); `HALOGEN_COMPOSABLE_CONTEXT_FLOOR`
  (strtol, min 64, default `0x800` = 2048),
  `HALOGEN_COMPOSABLE_CONTEXT_SEAM` (strtol, min 0; non-zero rejected:
  `HALOGEN_COMPOSABLE_CONTEXT_SEAM is not built in this release (0 is the
  only value)`), `HALOGEN_COMPOSABLE_CONTEXT_BYTES` (strtod, default
  `0x100000000` = 4 GiB store). Requires `HALOGEN_PROMPT_CACHE=2` and
  `HALOGEN_KV_POOL=1`; conflicts with `HALOGEN_KV8=1` (`composing
  re-rotates cached keys, which the 8-bit cache does not do`). Banner:
  `flash_serve: composable context ON (…): message units of %d tokens or
  more are extracted as they prefill and COMPOSED at any later offset
  behind the same system prompt; %.1f GiB of host memory for the store,
  fingerprint %s. NOT bitwise: …`. The store object is FNV-1a-hashed
  (seed `0x14650fb0739d0383`, prime `0x100000001b3`).

## 6. Sampling

- **[RV]** Engine sampling kernels (C++ symbols in
  `binutils-strings.txt`): `halogen::k_sample_f32` (:2446),
  `halogen::k_smpg_level<1|2|3>` (:783, :1964, :409),
  `halogen::k_smpg_pick` (:1142), `halogen::k_smpg_rpick` (:2720),
  `halogen::k_smpg_mass` (:1932), `halogen::k_smpg_qwrite` (:1933),
  `halogen::k_smpg_accept_final` (:1143), `k_argmax_part` (:1548),
  `k_argmax_final` (:2745), `k_argmax_f32_legacy` (:2627). The
  `k_smpg_level<N>` family + `SmpScratch` structure indicate a staged
  pipeline (mass computation → pick → quantised write → speculative
  accept-final), consistent with the MTP verify path (§4).
- **[RV]** Validation wording: `SAMPLE temperature outside (0, 2] or top_k
  < 0`; `malformed SAMPLE` (:979); `PENALTY/REPEAT/BIAS need SAMPLE with
  temperature > 0` (:1762).
- **[RV]** `SAMPLE` clause grammar (emitted by the reference front-end):
  `SAMPLE %.9g %d %.9g %.9g %d` = (temperature, top_k, top_p, min_p, seed)
  (serve_api.py:1958).
- **[RV]** Model-level filter defaults: `MODEL_FILTERS_BUILTIN = {"top_k":
  20, "top_p": 0.95}` (serve_api.py:2759), overridable by the model's
  `generation_config.json` (serve_api.py:2770–2802); precedence: request
  `SAMPLE` > `HALOGEN_*` env > model filters.
- **[RV]** Greedy vs sampled is announced in the banner (`BATCHED greedy or
  sampled`, :3753); `T` lines carry per-token logprobs with optional
  top-k pairs (serve_api.py `t_line_logprob`, :1066–1074).

## 7. `--resident-gib` query

Exact algorithm, from
[mode-dispatcher-FUN_01f23680.json](../evidence/flash_serve/decompile/mode-dispatcher-FUN_01f23680.json)
(lines ~362–406) **[D]**, with the helper functions now decompiled
([engine/](../evidence/flash_serve/engine/)):

1. `FUN_020bd2d0(path, &info)` parses the checkpoint (GGUF or `.hgn`,
   §1) into a descriptor.
2. `local_90 = (double)FUN_020be4e0(&info, …)` → resident-weight GiB.
3. If `local_90 ≤ 0` → **exit 1, no output** (matches the [RV] probes:
   stub GGUF, minimal GGUF and the ELF itself all exit 1 silently —
   [engine/probes.txt](../evidence/flash_serve/engine/probes.txt)).
4. Otherwise print `printf("%.1f %d %d %d %.1f\n", gib, keep_trunk,
   has_ngram, has_dtype16, extra)` and exit 0, where:
   - `keep_trunk` = 0 unless `HALOGEN_PREFILL_KEEP_TRUNK == "1"`, in which
     case `FUN_020be0a0(path)` **[D]** (matches frozen spec env table).
     `FUN_020be0a0` re-opens the `.hgn`, validates the header (§1) and
     scans the entry table: entries with dtype `0x10` and kind `2` whose
     name contains `.mlp.experts.` (and not `lm_head`) set the result —
     i.e. the flag reports whether the checkpoint carries a quantised
     expert trunk to "keep" (Ghidra control flow is partially mangled;
     re-verify before implementing).
   - `has_ngram` = `FUN_020bd680(&info) == 1` — scans tensor names for the
     substring `ngram_embedding` (memchr `'n'` + two overlapping 8-byte
     compares `"ngram_em"`/`"mbedding"`) **[D]**. This *corrects* the
     frozen spec's inferred field name "has-table-flag": the flag is the
     PLE ngram-embedding presence (§2/§3).
   - `has_dtype16` = `FUN_020bdc70(&info, 0x10) == 1` — 1 iff any tensor
     carries dtype code `0x10` **[D]** (corrects the inferred
     "gather-flag"; the dtype's name is the §1 gap).
   - `extra` = `FUN_020bdd90(&info)` rendered `%.1f` **[D]** — decompile
     pending (a 3.1 MB function; the batched re-run was cancelled and is
     queued).
5. The launcher accepts the line with the regex
   `^[0-9]+\.[0-9] [01]( [01]){0,2}( [0-9]+\.[0-9])?$` **[RV]**
   (launcher.md:99–102).

## 8. GEN protocol (engine line protocol)

- **[D]** Request verbs (connection handler
  [conn-handler-FUN_01ee0e20.json](../evidence/flash_serve/engine/conn-handler-FUN_01ee0e20.json),
  `switch` on the first token's length, each case `bcmp`-verified against
  the binary's string table):

  | token length | verbs |
  |---|---|
  | 1 | `X` (cancel) |
  | 3 | `GEN` |
  | 4 | `HREL`, `VIMG`, `PARK` |
  | 5 | `CSTAT`, `VOCAB` |
  | 6 | `MERGES`, `RESUME` |

  `PING` and `INFO` are answered on a separate fast path (not in this
  handler; the port's
  [`crates/cc-flash-serve/src/protocol.rs`](../../crates/cc-flash-serve/src/protocol.rs)
  implements them; the length-prefixed string fragments `<$PING`/`<$INFO`
  at [binutils-strings.txt:282377–282378](../evidence/flash_serve/binutils-strings.txt)
  corroborate the string-table storage noted in frozen-spec review C2).
- **[RV]** `GEN` line grammar (serve_api.py:1952–1990):
  `GEN <req> <max_tokens> <n_eos> [eos…] <n_ids> <ids…> [drafter]
  [SAMPLE %.9g %d %.9g %.9g %d] [penalty] [SNAP x] [SNAP2 x] [SNAP3 x]
  [LEAD3] [IMG n f h w …] [SCHEMA json|b64] [AFTER n] [ESCAPE n ids…]
  [TOOLS n m b64 [CALL b64] [ONE]] [THINK a b n ids…] [GUARD a b c d e]
  [AGUARD id] [SEG n ids…] [PROGRESS] [AGENT] [NOSTORE] [KEEP n]
  [KEPEND]\n`.
- **[RV]** Response lines (serve_api.py): `T` (token, optional logprob +
  top-k pairs, :1066–1074), `D` (done; fields per `_done_dict`,
  :1799–1852: reason, n_prompt, n_gen, prefill_ms, decode_ms, drafter,
  rounds, commit, n_cached, shared, pld_rounds, pld_accepted, pld_drafted,
  n_composed, room_from, max_tokens, …), `P` (progress: processed, total,
  cache, time_ms, :1062–1065), `G` (grammar enforcement note, base64,
  :1337–1348), `C` (status/cancel). `D <req> error <msg>` → HTTP 400 at
  the front-end (`engine_refusal`, :1350–1363).
- **[D]** `CSTAT` reply: a `C …` line of ~50 fields (counters from the
  engine object; zero-case template
  `"C 0 0 0 0 0 0 0 0 0 0.0 0.0 0.0 0.0 0 0 0.0 … %d %.1f %ld %ld %d %d
  %d %d %ld %ld %ld"` at conn-handler :2143).
- **[D]** `VIMG` (image attach): `VIMG req %ld: %dx%d at token %d (%zu B)`;
  rejection `flash_serve: VIMG %dx%d rejected (multiple of %d, at most
  %lld pixels)`; mismatch `VIMG geometry disagrees with the GEN line`;
  no-tower refusal `this server has no vision tower
  (HALOGEN_VISION_TOWER is unset)`.
- **[D]** `PARK`/`RESUME` (pause/resume a decoding request):
  `req %ld parked after %d tokens`, `PARK %ld ignored: not decoding`,
  `req %ld resumed at %d tokens`, `RESUME %ld ignored: not parked`.
- **[D]** `VOCAB`/`MERGES` (upload the HGV1 vocabulary / grammar-drafter
  merges): replies `V ok %u %.0f` and `M ok `; logs
  `flash_serve: grammar: vocabulary of %u tokens (%zu plain, longest %zu
  B) in %.0f ms` and `flash_serve: grammar drafter: %ld merges in %.0f ms`
  (HGV1 blob format: frozen spec / api.md).
- **[D]** Queueing: `GEN queued (%zu bytes), %zu waiting, %zu active`;
  a probe connection that closes without a request logs
  `queued connection closed without a request (a health probe does this)`;
  the handler services one request per connection.
- **[D]** `X <req>` cancels a queued/active request (verb table above;
  port's protocol.rs currently closes on `GEN` only — the full port must
  implement the remaining verbs).
- **[D]** Tracing: `trace %10.1f ms` line emitted when request tracing is
  on; `HALOGEN_S239_EAGER_ERASE` env read in the handler.

## Gaps register (for the port's next analysis pass)

| § | Gap | Revisit address / source |
|---|---|---|
| 1 | dtype code `0x10` name | table `PTR_DAT_020baa68` |
| 1 | reference `.hgn` entry layout vs port `hgn.rs` (name 96 B/dtype @96 vs name 32 B/dtype @32) | `FUN_020be0a0` (staged) vs port reader; confirm with a reference-produced checkpoint |
| 1 | `FUN_020bd2d0` full decompile | decompile pending |
| 2 | attention head count / head dim / GQA | model-config ctor; `k_attn_*`/`k_rope_*` kernels |
| 4 | MTP-head loader function (`FUN_01f24c10`/`FUN_01f24d00` resolved as composable-context instead) | `FUN_01f24580`/`FUN_01f24690` ctor steps |
| 5 | `FUN_020bdd90` (resident-gib `extra` field) | 3.1 MB function, re-run queued |
| 4 | `HALOGEN_MTP_DEPTH` default 2 vs 3 contradiction | mode-dispatcher vs frozen spec |
