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

- **[RV]** Reference header (from `FUN_020be0a0`,
  [engine/fun-020be0a0.json](../evidence/flash_serve/engine/fun-020be0a0.json),
  confirmed round 2 against the real checkpoint
  `qwen38-flash-next-v2-abliterated.hgn` — repo root, 66,687,678,432 bytes,
  sha256 of the first 4 KiB
  `0ccfefcdb01b67708b05865be5f5778828f2f7027377313616c48bc8b9ce62a0`):
  `pread(fd, buf, 0x68, 0)` — a **104-byte header**; magic bytes
  `0x314e4748` = `HGN1` (exact compare); `u64` entry_count @8 (must be
  ≤ `0x100000`); `u64` table_offset @16 (the entry table is read from this
  offset; `104` in the real file). The two further `u64`s are now resolved:
  @24 is the **payload-region start** (`322,560` = `0x4ec00` in the real
  file — 512-byte aligned, 56 bytes past the table end, and exactly entry
  0's data offset) and @32 is the **declared file size** (`66,687,678,432`
  — matches the real file byte-exactly; this is what
  `checkpoint: truncated file %s (size mismatch)` compares against `fstat`).
  The 64-byte `model_id` @40 reads `qwen3.8-flash-next-v2`. The port's
  header
  ([`crates/cc-halogen-tools/src/hgn.rs`](../../crates/cc-halogen-tools/src/hgn.rs)
  and [`crates/cc-flash-serve/src/hgn.rs`](../../crates/cc-flash-serve/src/hgn.rs))
  now matches the full layout including @24/@32.
- **[RV]** **Entry layout (resolved round 2):** `FUN_020be0a0` iterates
  entries with stride `0xa0` (160 B) reading the name via `strnlen(entry,
  0x60)` (96-byte name field), the dtype as `u32` @**96** and a second `u32`
  @**100** (the keep-trunk query matches dtype `0x10` + kind `2` on
  `.mlp.experts.` names). The real checkpoint confirms the full entry:
  name `0..96` (NUL-padded), `u32` dtype @96, `u32` kind @100, four `u64`
  dims @104/@112/@120/@128 (unused trailing dims 0), `u64` payload
  data_offset @136, `u64` nbytes @144, and an uninterpreted `u64` @152
  (candidate per-tensor checksum — the tools' `verify` checks "checksums";
  hash-like values in the real file, e.g. `0x25ddbae32` for
  `embed_tokens.weight`). The @136/@144 pair is chain-verified in the real
  bytes: entry 0 (`embed_tokens.weight`, dtype 5, kind 2, dims
  `[248320, 2560, 0, 0]`) `322,560 + 357,580,864` equals entry 1's
  data_offset (`layers.0.attn_hyper_connection.block_inject_weight.weight`,
  dtype 24, kind 2, dims `[4, 10240, 0, 0]`, nbytes 33,280), and entry 1 +
  its nbytes equals entry 2's data_offset
  (`layers.0.attn_hyper_connection.hc_norm.weight`, dtype 0, kind 1, dims
  `[10240, 0, 0, 0]`, nbytes 20,480). The port's earlier accessors (name
  `0..32` / dtype @32 / nbytes @40) matched neither source — they were
  copied from the tools' 96-byte *in-memory* entries (name `0..32`, dtype
  @32, nbytes @80), not the on-disk entries — and are corrected in both
  `cc-flash-serve` and `cc-halogen-tools` (round-2 fix; the tools' matrix
  stays green because its `.hgn` success cases are deferred and its live
  error cases don't read entries).
- **[RV]** **Version gate (resolved round 2):** the reference accepts only
  versions where `(uint)(version − 3) > 0xfffffffd`, i.e. version ∈ {1, 2}
  (Ghidra renders this from the header's `u32` @4; type-propagation warning
  present — the real checkpoint's version `2` confirms the gate reads the
  header `u32` @4 as documented). The port's `write_hgn` writes version
  **1** in every git revision (7 appears only in test fixtures — round-1
  review correction); the actual cause of the review-F3 rejection of
  subject-written `.hgn` files is the 32-byte written file failing the
  reference's `pread == 0x68` header-size check. The port's `cc-flash-serve`
  reader now implements the {1, 2} gate (round-2 fix) with the engine's
  wording `checkpoint: unsupported version in %s` ([D], `FUN_020bea30`
  error set — `FUN_020be0a0` itself rejects silently and the tools spec has
  no version error, so this is the only derivable wording); the
  `cc-halogen-tools` reader stays ungated (its spec error table has no
  version entry and its v7 fixtures are pinned by the matrix).
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
- **[D]** Dtype code → name table **resolved round 3**: the table at
  `PTR_DAT_020baa68` lives in the shipped **`halogen-tools`** binary (the
  same address in `flash_serve` holds code; the symbol came from the
  tools' `inspect` handler,
  [inspect-FUN_01eed5d0.json](../evidence/halogen-tools/decompile/inspect-FUN_01eed5d0.json));
  dumped via `rea read-bytes`
  ([dtype-table-PTR_DAT_020baa68.json](../evidence/flash_serve/engine/dtype-table-PTR_DAT_020baa68.json)
  + the `.rodata` blob it indexes):
  `0=bf16`, `1=f32`, `2=f16`, `3=i32`, `4=i64`, `5=q4c`, `6=fp8r`,
  `7=q8g64`, `8=i4l`, `9=NULL`, `10=fp8g`, `11=q8s32`, `12=q6k`,
  `13=iq4nl`, `14=q4a`, `15=q5a`, `16=ht`, `17–19=NULL`, `20=q8_0`,
  `21=q5_0`, `22=q5_1`, `23=i4r`, `24=q6g64`.
  The port's `dtype_name`
  ([hgn.rs](../../crates/cc-halogen-tools/src/hgn.rs), `0=f32`, `1=f16`,
  `2=bf16`) is wrong — the reference maps `0=bf16`, `1=f32`, `2=f16`; this
  resolves the
  round-2 contradiction (real `hc_norm.weight` dtype 0 with 2 B/element)
  and names the codes seen in the real checkpoint: `5` → `q4c`
  (`embed_tokens.weight`), `24` → `q6g64`
  (`layers.0.attn_hyper_connection.block_inject_weight.weight`) and
  `0x10` → `ht` (queried by `FUN_020bdc70`, §7). The validity mask
  `0x1f1fdff` matches the NULL slots (codes 9, 17, 18, 19).
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
- **[RV]** Attention head configuration **resolved round 3** (real
  checkpoint tensor table + `FUN_01f82410`): `head_dim = 256`
  (`self_attn.q_norm`/`k_norm` width), `num_heads = 48`
  (`self_attn.q_proj.weight` `[12288, 2560]`), `num_kv_heads = 2`
  (`self_attn.k_proj`/`v_proj.weight` `[512, 2560]`) → GQA ratio 24:1.
  Full attention exists in 12 of the 48 main layers (indices 3, 7, …, 47 —
  every 4th; the other 36 layers are `linear_attn`) plus the MTP head; the
  KV-pool sizing function `FUN_01f82410` cross-checks this: per position it
  returns `(K+V)·13 + 2880` with `K = V = 0x400` (512 × bf16) in bf16 mode
  and `0x208`/`0x220` (512+8 / 512+32 — fp8 payloads with scale blocks) in
  KV8 mode (`FUN_01f6ff70() == 1`); the `× 0xd` is the 12 main attention
  layers + 1 MTP layer
  ([fun-01f82410.json](../evidence/flash_serve/engine/fun-01f82410.json),
  [fun-02019610.json](../evidence/flash_serve/engine/fun-02019610.json)).
  `self_attn.o_proj.weight` `[2560, 6144]` **resolved round 4 [D]**
  (full-attention block forward `FUN_01fb3670`, §3,
  [fun-01fb3670.json](../evidence/flash_serve/engine/fun-01fb3670.json)):
  the block's attention core operates on **24 query heads × 256 =
  6144** — the RoPE wrapper receives head count `0x18` (24) for q (and
  `0x2` for k), the attention kernel's q slot stride is `0x3000` bytes
  (6144 fp8 activations), and the `o_proj` gemv contracts exactly
  `0x1800` (6144) → `0xa00` (2560). The `q_proj` gemv still computes
  all 12288 rows (the weight table exposes `q_proj` as two 6144-row
  halves `DAT_02127a40`/`DAT_02127a48` plus a combined handle
  `DAT_02127a38`); the rope/attention path consumes the 6144 (24
  heads) via the layer+`0x460` buffer fed by the `HALOGEN_QK_PREP`
  kernel from the gemv output (layer+`0x280`). Which consumer uses the
  second 6144-row half is an open sub-gap (Gaps register; candidates:
  the QSA indexer path, the MTP head, a second attention pass).

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
- **[D]** **Layer dispatcher and block forwards (resolved round 4):**
  `FUN_01ffe2e0(model, layer, tokens)`
  ([fun-01ffe2e0.json](../evidence/flash_serve/engine/fun-01ffe2e0.json))
  executes one decoder layer: hyper-connection pre-mix `FUN_01f8c690(…, 0)`
  → block dispatch on the layer-type byte
  (`*(model->layers + layer·0xd08) == 1` → full attention `FUN_01fb3670`,
  else linear attention `FUN_01fa0700`) → MLP `FUN_01f9f8c0` (a
  small-batch variant `FUN_01f98d30` runs first for token counts < 9 when
  the layer's own buffers at +0x800/+0x8f0/+0x9e0 are staged; the result
  flag is stored at model+`0x57f`) → hyper-connection post-mix
  `FUN_01f8c690(…, 1)`. With verbose timing on, `hipEventRecord` slots
  indexed by layer (`DAT_0212a0a0 + layer·7`, 7 qwords per layer for the
  48 main layers; 5 stages used: pre-mix start/end, attention end, MLP
  end, post-mix end) bracket the stages — the decompile counterpart of
  the per-stage timing template above. The function iterates the layer
  range (tail: `param_2 + 1` while `< 0x2f`), so one call drives a full
  48-layer forward for the token batch.
- **[D]** **Full-attention block forward `FUN_01fb3670`**
  ([fun-01fb3670.json](../evidence/flash_serve/engine/fun-01fb3670.json))
  — op sequence per token batch, confirming the frozen
  `norm → qkv → rope → attention → projection` template:
  1. QSA index gemv `indexer.index_qk_proj` `[640, 2560]` → layer+`0x470`
     (gated by `HALOGEN_QSA_DENSE`; the staging error text names the same
     indexer and its `block_topk`);
  2. qkv projection gemvs: `q_proj` 12288 rows (combined handle
     `DAT_02127a38`, also exposed as two 6144-row halves
     `DAT_02127a40`/`DAT_02127a48`) → layer+`0x280`; `k_proj`/`v_proj`
     512 rows each → layer+`0x2f8`/`0x370`;
  3. `HALOGEN_QK_PREP` kernel (grid `tokens × 0x1b`) — the per-head
     `q_norm`/`k_norm` [256] application (§2), writing the rope inputs
     layer+`0x460` (q, 6144 wide) and layer+`0x468` (k);
  4. RoPE wrapper `FUN_01f88ea0`: q 24 heads × `0x100` (256),
     k 2 heads × `0x100` (the 24-vs-48 head asymmetry is resolved in the
     §2 `o_proj` note);
  5. KV-cache append (`hipMemcpyAsync`, device pointers layer+`0xc30` /
     `0xc38`, 1024 B per token = 2 × 512 B fp8 k+v, matching the §2 KV8
     sizing);
  6. attention kernel — `k_attn_fd` for batches n ≤ 8 via `FUN_01fb9970`
     (error text `forward_batch: the per-row attention is k_attn_fd only
     (HALOGEN_ATTN_FD=%d, n=%d > %d?)`);
  7. `o_proj` gemv `[2560, 6144]` → layer+`0x3e8` (the block output
     hidden-state slot).
- **[D]** **Linear-attention block forward `FUN_01fa0700` — Gated
  DeltaNet math (resolved round 4)**
  ([fun-01fa0700.json](../evidence/flash_serve/engine/fun-01fa0700.json)):
  identified as Gated DeltaNet from the `k_dn_*`/`k_dnc_*` kernel family
  in `binutils-strings.txt` and the §2 tensor set. Op sequence:
  1. gemv `in_proj_qkv` `[10240, 2560]` → block+`0x8` (packed q/k/v:
     v = 6144 = 48 heads × 128; q/k = 2048 each inferred from the
     10240 − 6144 split — the head split is an open sub-gap, Gaps
     register);
  2. gemv `in_proj_z` `[6144, 2560]` → block+`0x80` (output gate, v-width);
  3. gemvs `in_proj_a` / `in_proj_b` `[48, 2560]` → block+`0xf8` /
     `0x170` (per-v-head decay/input scalars);
  4. depthwise causal conv1d `conv1d.weight` `[10240, 4]` over the packed
     qkv, with a 3-token history state (`hipMemset` of `0xf000` =
     3 × 10240 × 2 B at `flash_model.hip:3433`);
  5. decay gates from `A_log` / `dt_bias` `[48]` (`k_dn_gates` kernel
     family);
  6. delta-rule recurrence (`k_dn_recur_step`): state 48 × 128 × 128
     bf16 (`0x180000` bytes per slot, confirmed in the model ctor
     `FUN_01f70c20` arena layout), step kernel grid 48 × block 128;
  7. per-head `norm` `[128]` and gemv `out_proj` `[2560, 6144]`.
  The exact recurrence formula (β normalisation and decay application
  order) remains an open sub-gap pending the `k_dn_recur_step` GPU
  code-object disassembly (Gaps register).
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
  `switch` on the first token's length; the `GEN`/`VIMG`/`PARK` cases are
  `bcmp`-verified in the decompile, while `X`, `HREL`, `VOCAB`, `CSTAT`,
  `MERGES` and `RESUME` are attested only by binutils strings — review
  correction, round-1 review `.tmp/review-report-engine-round1.md`):

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
| 1 | ~~dtype code `0x10` name; codes `5`/`24` observed in the real checkpoint (round 2) also unnamed, and the real `hc_norm.weight` (dtype 0, 2 B/elem) contradicts the port's `0=f32` mapping~~ **resolved round 3 [D]:** full table dumped from `halogen-tools` `PTR_DAT_020baa68` (§1, [dtype-table-PTR_DAT_020baa68.json](../evidence/flash_serve/engine/dtype-table-PTR_DAT_020baa68.json)): `0=bf16`, `1=f32`, `2=f16`, `5=q4c`, `16=ht`, `24=q6g64`, …; port `dtype_name` mapping corrected in §1 (the same address in `flash_serve` holds code, not the table) | — |
| 1 | ~~reference `.hgn` entry layout vs port `hgn.rs`~~ **resolved round 2 [RV]:** on-disk entry = name `0..96` / dtype u32 @96 / kind u32 @100 / dims u64×4 @104..@136 / data_offset u64 @136 / nbytes u64 @144 / uninterpreted u64 @152; confirmed against `qwen38-flash-next-v2-abliterated.hgn` (66,687,678,432 B, first-4 KiB sha256 `0ccfefcdb01b67708b05865be5f5778828f2f7027377313616c48bc8b9ce62a0`) and `FUN_020be0a0`; port readers in `cc-flash-serve` + `cc-halogen-tools` corrected (the old name `0..32`/dtype @32/nbytes @40 accessors were the tools' in-memory layout, not on-disk) | — |
| 1 | ~~version gate {1, 2} vs the port's ungated reader~~ **resolved round 2:** `cc-flash-serve` reader gates to {1, 2} (wording `checkpoint: unsupported version in %s` per `FUN_020bea30`); real file is version 2; `cc-halogen-tools` stays ungated (no version error in its spec table, v7 fixtures pinned by the matrix) | — |
| 1 | `FUN_020bd2d0` full decompile | decompile pending |
| 1 | ~~dequantisation codecs for the quantised dtypes of the real checkpoint (q4c / q8g64 / q6g64 / ht / i4r)~~ **specified round 5 [D]:** companion spec [`flash_serve-dtypes.md`](flash_serve-dtypes.md) — per-dtype block layouts + dequant formulas verified against all 2015 table entries (the table also contains dtype `7` q8g64, 18 tensors, beyond the review's `5`/`16`/`23`/`24` expectation); residual sub-gaps (q4c 64-B header contents, `qparam` storage, `FUN_01f991e0` + variant classifiers, GPU code-object math) listed in that spec's §10 | — |
| 2 | ~~attention head count / head dim / GQA~~ **resolved round 3 [RV]:** `num_heads = 48`, `head_dim = 256`, `num_kv_heads = 2` (GQA 24:1) from the real checkpoint's `self_attn` tensor dims + `FUN_01f82410`'s KV sizing (§2, [fun-01f82410.json](../evidence/flash_serve/engine/fun-01f82410.json)); 12 main attention layers (3, 7, …, 47) + 1 MTP layer = the `× 13` multiplier; ~~`o_proj` input 6144 (half the `q_proj` output) noted for follow-up~~ **resolved round 4 [D]** (§2 `o_proj` note / §3 `FUN_01fb3670`: the attention core is 24 q-heads × 256 = 6144; the second `q_proj` half's consumer is registered below) | — |
| 3 | consumer of the second 6144-row `q_proj` half (the rope/attention path consumes 6144 via layer+`0x460`; `o_proj` contracts 6144) | candidates: QSA indexer path, MTP head, a second attention pass — re-visit `FUN_01fb3670` branches + the `DAT_02127a38`/`DAT_02127a40`/`DAT_02127a48` handle set in `FUN_01f766f0` ([fun-01f766f0.json](../evidence/flash_serve/engine/fun-01f766f0.json)) |
| 3 | linear_attn q/k head split (inferred 2048 + 2048 from `in_proj_qkv` 10240 − v 6144; no `0x800` constant appears in `FUN_01fa0700`) | `k_dn_*` kernel arguments in the AMDGPU code objects |
| 3 | exact Gated DeltaNet recurrence formula (β normalisation, decay application order) | `k_dn_recur_step` GPU code-object disassembly |
| 4 | MTP-head loader function (`FUN_01f24c10`/`FUN_01f24d00` resolved as composable-context instead) | `FUN_01f24580`/`FUN_01f24690` ctor steps |
| 5 | `FUN_020bdd90` (resident-gib `extra` field) | 3.1 MB function, re-run queued |
| 4 | `HALOGEN_MTP_DEPTH` default 2 vs 3 contradiction | mode-dispatcher vs frozen spec |
