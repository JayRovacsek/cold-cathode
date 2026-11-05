# flash_serve dequantisation-codec spec (dtype codecs of the real checkpoint)

Scope: the dequantisation codecs the engine of `halogen-flash-server`
v0.18.21 (`flash_serve`, sha256
`7c92fe3f8ac544c4fb690a69e9ce359fa7b8d453a8eab8e9a1289c6f2309c012`)
applies to every dtype present in the real checkpoint
`qwen38-flash-next-v2-abliterated.hgn` (66,687,678,432 B). This
complements — and does not restate — the `.hgn` layout and dtype-code
table in [`flash_serve-engine.md`](flash_serve-engine.md) §1.

**File choice (documented per task brief):** specified as a new companion
file rather than a section inside `flash_serve-engine.md`, following the
[`flash_serve-mtp-kv-npu.md`](flash_serve-mtp-kv-npu.md) precedent — the
engine spec is concurrently owned and parallel edits to one file risk
clobbering. The engine spec's Gaps register gains one row pointing here.

Provenance markers (same vocabulary as the engine spec): **[RV]** —
runtime-verified / shipped-string / shipped-file observed; **[D]** —
Ghidra-decompile-derived (Ghidra 12.1.4 via the `rea` CLI against the
reference binary). New evidence lives under
[`evidence/flash_serve/dtypes/`](../evidence/flash_serve/dtypes/);
previously committed evidence under `evidence/flash_serve/engine/` and
the shipped-string dump `evidence/flash_serve/binutils-strings.txt` are
linked inline.

**Method note (honesty about coverage):** the x86 binary is stripped
(only 54 libstdc++ dynamic symbols), so the `k_*` kernel names live in
`.rodata` strings and the HIP code objects, not the procedure table. The
x86 functions decompiled here are **launchers**: they select and
parameterise the HIP kernel, while the value-decoding math itself lives
in the AMDGPU code objects, which the x86 Ghidra provider cannot
decompile. Every per-dtype "dequant formula" below is therefore marked
**[D-inferred]** where it is derived from kernel signatures, payload
arithmetic and validation strings rather than from observed decode
instructions.

---

## 1. Checkpoint dtype histogram

Source: the entry-table dump
[`evidence/flash_serve/dtypes/table-dump.tsv`](../evidence/flash_serve/dtypes/table-dump.tsv)
(2015 entries, dumped from the real checkpoint by the round-2 tooling;
columns `name / dtype / kind / dims / data_offset / nbytes`). The
nbytes formula per dtype is re-checked against **all 2015 entries** by
[`evidence/flash_serve/dtypes/verify-nbytes-formulas.awk`](../evidence/flash_serve/dtypes/verify-nbytes-formulas.awk)
(0 mismatches).

| dtype | name | tensors | bytes | payload formula (verified) |
|---|---|---:|---:|---|
| 0 | bf16 | 353 | 199,188,736 | `numel × 2` |
| 2 | f16 | 818 | 5,063,680 | `numel × 2` (all entries are scale/rotation-plane vectors, §2) |
| 4 | i64 | 3 | 280 | `numel × 8` |
| 5 | q4c | 124 | 1,788,699,296 | `64 + (numel/K) × pad16(9K/16)` (§3) |
| 7 | q8g64 | 18 | 94,786,560 | `numel × 17/16` (§4) |
| 16 | ht | 313 | 1,782,579,200 | `numel / 2` (§6) |
| 23 | i4r | 96 | 62,285,414,400 | `numel × 33/64` (§7) |
| 24 | q6g64 | 290 | 531,619,840 | `N × pad16(13K/16)` (§5) |

- **[RV]** The review brief expected quantised dtypes `5`/`16`/`23`/`24`
  only; the table **also contains dtype `7` (q8g64, 18 tensors)** plus
  the plain dtypes `0`/`2`/`4`. All eight observed codes are specified
  here. The remaining table codes (`1=f32`, `3=i32`, `6=fp8r`, `8=i4l`,
  `10=fp8g`, `11=q8s32`, `12=q6k`, `13=iq4nl`, `14=q4a`, `15=q5a`,
  `20=q8_0`, `21=q5_0`, `22=q5_1` — engine spec §1 table) do not occur
  in this checkpoint.
- **[RV]** Sum of per-dtype bytes: 66,687,351,992 B; plus the 104-byte
  header, the 322,400-byte entry table (2015 × 160 B) and the 56-byte
  gap to the payload start (`322,560`, engine spec §1) accounts for the
  file size 66,687,678,432 B with per-tensor alignment padding.
- **[RV]** Every quantised dtype named in the engine's validation
  strings has a findable kernel family in the shipped strings (no
  "dtype without kernel" case): `q4c` → `k_deq_q4cp*`/`k_lq4w_grp`/
  `q4moe::*`, `q8g64` → `k_deq_q8g64`/`k_lq8w_grp`/`k_linear_q8`,
  `q6g64` → `k_deq_q6g64`/`k_lq6w_grp`, `ht` → `k_ht_*`, `i4r` →
  `k_i4r_*` (full inventories in §3–§7).

---

## 2. Plain dtypes: bf16 (0), f16 (2), i64 (4)

- **[RV] bf16 (code 0)** — `numel × 2`, row-major, no codec. 353
  tensors: the RMSNorm weights (`*.hc_norm.weight`, `*.q_norm.weight`,
  `*.k_norm.weight`, `*.indexer.{q,k}_layernorm.weight`,
  `mtp.pre_fc_norm_{hidden,embedding}.weight`), the MoE router
  `*.mlp.gate.weight` [512, 2560], the linear-attention state
  (`A_log` [36], `dt_bias` [36], `convd.weight` [36], `norm.weight`
  [36]), the `layers.1.ple.*` projection set, and the MTP-head norm/gate
  mirrors. Dequantisation is the identity (already the compute dtype).
- **[RV] f16 (code 2)** — `numel × 2`. **All 818 f16 entries are
  companion scale vectors**, never standalone weights: 361 `.suh`, 409
  `.svh` and 48 `.su` names (361 + 409 + 48 = 818 exactly). The naming
  rule is `<tensor name minus ".weight">.suh|.svh` (e.g.
  `layers.0.linear_attn.in_proj_qkv.suh` [2560] / `.svh` [10240] for
  `…in_proj_qkv.weight` [10240, 2560]); `.su` occurs only as the shared
  `layers.N.mlp.experts.su` [2560] (§7). The engine's own wording calls
  them the HT linear's **rotation planes**: `[RV]`
  `binutils-strings.txt:499` `%s: an HT linear without its rotation
  planes (%s.suh%s, %s.svh)` and `:3174` `%s: an HT linear's rotation
  planes (%s.suh, %s.svh) must be f16 of %lld and %lld values`.
- **[RV] i64 (code 4)** — `numel × 8`. 3 tensors, all `ple` metadata:
  `layers.1.ple.ple_embedding.layer_multipliers` [3],
  `…ngram_heads_offsets` [16], `…ngram_heads_vocab_sizes` [16]. Not
  weights; no codec.

---

## 3. q4c (code 5) — the Q4C-P packed container

- **[RV] Tensor set (124):** `embed_tokens.weight` [248320, 2560], the
  48 `layers.N.mlp.shared_expert_gate.weight` [1, 2560], 36
  `layers.N.linear_attn.in_proj_a` / 36 `…in_proj_b` weights, and the
  three MTP-head entries
  (`mtp.layers.0.mlp.shared_expert_gate.weight`,
  `mtp.layers.0.mlp.experts.gate_up_proj.weight` [512, 1280, 2560],
  `mtp.layers.0.mlp.experts.down_proj.weight` [512, 2560, 640]).
- **[RV] Payload size formula:** `64 + (numel/K) × pad16(9K/16)` where
  `K` is the last dim (verified on all 124 entries, both the 2-D and the
  3-D expert-tensor forms — the 3-D MTP entries carry the same 64-byte
  prefix: `943,718,464 = 64 + 512×1280×1440`). The engine validates it
  verbatim: `binutils-strings.txt:1794`
  `%s: Q4C-P size %lld != table nbytes %llu, the payload layout this
  kernel assumes is NOT the one on disk`.
- **[RV] Block layout:** `9K/16 = 18·K/32` bytes per row — the Q4_0
  block geometry (per 32 elements: 2-byte f16 scale + 16 bytes of
  packed nibbles), each row padded to a 16-byte boundary. The 64-byte
  prefix is a tensor-level header whose contents the x86 path does not
  parse (Gaps).
- **[RV] Source formats:** `binutils-strings.txt:1449` names the legal
  sources — `Q4C-P source (IQ4_NL, IQ4_XS, IQ3_S, Q4_0)`; `:3545`
  `%s: expected Q4C-P, got %s qparam %u`; `:365` `%s: expected bf16 or
  Q4C-P, got %s qparam %u`; `:205` `%sgate_up_proj.weight: a fused
  gate_up is read in place as Q4C-P or bf16 only`. The shipped
  `kIq4nlValues` table (16-entry IQ4_NL codebook) and the
  `k_gather_iq4nl(const u8*, const long*, u16*)` kernel confirm the
  IQ4_NL sub-format is decoded via codebook lookup, while the v32
  kernels decode the Q4_0-style 32-element blocks.
- **[D-inferred] Dequant formula (Q4_0 path):** for each 32-element
  block, `w[k] = f16_scale × (nibble(k) − 8)` with the low nibble of
  each byte holding element `2i` (Q4_0 convention). The IQ4_NL path
  instead maps each 4-bit code through `kIq4nlValues` scaled by the
  tensor f16 scale. Exact per-kernel arithmetic lives in the AMDGPU
  code objects.
- **[RV] Kernel family (from `binutils-strings.txt`):**
  `k_deq_q4cp(const u8*, u16*, long×5)`, `k_deq_q4cp_f32(const u8*,
  float*, long×5)`, `k_deq_q4cp_sel(const u8*, u16*, const int*,
  long×5)`, `k_deq_q4cp_v32<bool>` (32-element blocks), `k_deq_q4cp_v8`
  / `k_deq_q4cp_v8u<int,bool>` (8-byte-unit variants, `int ∈ {1,2,4,8}`),
  `k_deq_q4cp_probe<int,int>` (source-format probes, `int ∈ {1,4}`),
  `k_lq4w_grp<dtype,…>` (load-weight-group kernels instantiated for
  dtype codes 1–8, block 16/32 and a shift bool wired to
  `HALOGEN_LQ4W_SH`), `k_lq4w_pair<1,16,1>` (the PAIR form), and the
  MoE gemv family `q4moe::k_q4moe<int,int,int,Tune,int>` /
  `q4moe::k_q4moe6<…>`. The `DeqCb` callback struct passed to the
  `k_deq_q4cp_*` kernels is the `HALOGEN_DEQ_CB` knob.
- **[RV] Env knobs:** `HALOGEN_LQ4W_SH` (decompile
  [dtypes/env-lq4w-sh-getter-FUN_0205e4d0.json](../evidence/flash_serve/dtypes/env-lq4w-sh-getter-FUN_0205e4d0.json)
  — `getenv("HALOGEN_LQ4W_SH")` → `strtol`, default 0),
  `HALOGEN_DEQ_KERN`, `HALOGEN_DEQ_CB`, `HALOGEN_DEQ_PROBE`,
  `HALOGEN_DEQ_STREAM`, `HALOGEN_FLASH_DEQ_V8`, `HALOGEN_Q4MOE_FORM`,
  `HALOGEN_Q4MOE_LINE`, `HALOGEN_LQ8_WAVE`.

---

## 4. q8g64 (code 7) — 8-bit, group 64 (present beyond the review's expectation)

- **[RV] Tensor set (18):** every MTP-head matrix —
  `mtp.fc_embedding.weight` / `mtp.fc_hidden.weight` [2560, 2560],
  `mtp.hyper_connection_mixer.{input_mix_weight_up,input_mix_weight_down}.weight`,
  `mtp.layers.0.attn_hyper_connection.*` and
  `mtp.layers.0.mlp_hyper_connection.*` (block_inject [4, 10240],
  input_mix up/down), `mtp.layers.0.self_attn.{q,k,v,o}_proj.weight`,
  `mtp.layers.0.self_attn.indexer.index_qk_proj.weight`,
  `mtp.layers.0.mlp.shared_expert.{gate,up,down}_proj.weight`. The
  trunk uses `ht`/`q6g64`/`bf16` instead; only the head ships q8g64.
- **[RV] Payload size formula:** `numel × 17/16` — 68 bytes per
  64-element block = **4-byte scale + 64 u8 values**. Verified on all
  18 entries (e.g. [2560, 2560] → 6,963,200). Engine wording
  (`binutils-strings.txt:1792`/`:3173`): `%s: Q8G64 needs K %% 64 == 0,
  a 16B-aligned payload and nbytes = numel*17/16 (K=%lld,
  nbytes=%llu)`.
- **[D-inferred] Dequant formula:** symmetric int8 per 64-element group,
  `w[k] = scale × q`; the 4-byte scale encoding (f32 vs u32) is not
  directly observed on the x86 side (the 2-byte alternative belongs to
  `q8s32`, whose formula `N*pad16(17K/16)` = 34 B per 32 elements is a
  separate codec — `binutils-strings.txt:2162`).
- **[RV] Kernel family:** `k_deq_q8g64(const u8*, u16*, uint, uint,
  long)`, `k_lq8w_grp<dtype,32,2>` (dtypes 1–7), `k_linear_q8(const u8*,
  const u16*, u16*, long, long)` and `k_linear_q8s32(const u8*, const
  u16*, u16*, long, long)`.
- **[RV] Env knob:** `HALOGEN_LQ8_WAVE`.

---

## 5. q6g64 (code 24) — 6-bit, group 64

- **[RV] Tensor set (290):** the hyper-connection mixers of all 48
  trunk layers (`layers.N.attn_hyper_connection.{input_mix_weight_up,
  input_mix_weight_down,block_inject_weight}.weight` and the
  `mlp_hyper_connection` mirrors — 6 × 48) plus the top-level
  `hyper_connection_mixer.{input_mix_weight_up,input_mix_weight_down}.weight`.
- **[RV] Payload size formula:** `N × pad16(13K/16)` — 52 bytes per
  64-element block = **4-byte scale + 48 bytes of 6-bit packed values**
  (64 × 6 / 8 = 48). Verified on all 290 entries (e.g. [4, 10240] →
  33,280). Engine wording (`binutils-strings.txt:868`):
  `%s: Q6G64 needs K %% 64 == 0, a 16B-aligned payload, qparam 64 and
  nbytes = N * pad16(13K/16) (K=%lld, nbytes=%llu)` — the `qparam 64`
  names the group size stored with the tensor.
- **[D-inferred] Dequant formula:** symmetric 6-bit per 64-element
  group, `w[k] = f32_scale × (q6(k) − 32)`; the 4-byte scale encoding is
  inferred from the block arithmetic (same 4-byte scale pattern as
  q8g64).
- **[RV] Kernel family:** `k_deq_q6g64(const u8*, u16*, long, long)`,
  `k_lq6w_grp<1|2|3, 32, {2,4,8}>`.
- **[RV] Cross-check:** `linear_f32: %s is Q8G64 / Q6G64 and has no f32
  path` (`binutils-strings.txt:505`) — these two codecs have no f32
  output variant, unlike q4c (`k_deq_q4cp_f32` exists).

---

## 6. ht (code 16) — 4-bit weights with f16 rotation planes

- **[RV] Tensor set (313):** `lm_head.weight` [248320, 2560], the 12
  main `self_attn.{q,k,v,o}_proj` + 12 `self_attn.indexer.index_qk_proj`,
  the 36 `linear_attn.{in_proj_qkv, in_proj_z,out_proj}`, and the 3 × 48
  `mlp.shared_expert.{gate,up,down}_proj` weights.
- **[RV] Payload size formula:** `numel / 2` — pure 4-bit packing with
  **no in-payload scale** (verified on all 313 entries). The scales live
  in the companion f16 **rotation planes** `.suh` (length K) and `.svh`
  (length N) — every ht tensor has both (§2; the engine validates their
  dtype and lengths: `…must be f16 of %lld and %lld values`).
- **[RV] Format checks:** `checkpoint: %s: an HT tensor whose shape and
  size do not match` (`:947`), `checkpoint: %s: an HT tensor in a form
  this engine does not read (codebook, tile order or bits)` (`:819`),
  `%sexperts: HT needs both projections in HT and the pinned path
  (got %s / %s)` (`:1774`), and the loading rule `%s: HT weights are
  read in place and need the pinned path (HALOGEN_FLASH_PIN_TRUNK must
  not be 0)` (`:1646`).
- **[D-inferred] Dequant formula:** the `k_ht_deq_orig<3|4>(const u32*
  packed, int, const DF16* suh, const DF16* svh, u16* out, int)`
  signature shows the payload consumed as `u32` words (8 nibbles per
  word, low-first) and scaled by the two f16 plane vectors (`DF16_` =
  packed f16 pair); the template parameter selects the 3-bit or 4-bit
  variant and the checkpoint's `numel/2` tensors are the 4-bit one. The
  exact nibble→value mapping (offset/scale application order) is GPU-side
  math.
- **[RV] Kernel family (from `binutils-strings.txt`):** `k_ht_deq_orig`,
  `k_ht_deq_t<3|4>` (transpose dequant), `k_ht_gemv_dn/dn_g/dn_t/dn_ts/
  dn_tv` (down-proj gemv), `k_ht_gemv_gu/gu_gs/gu_t/gu_ts` (gate_up
  gemv), `k_ht_lin<3|4,{1,2,4,5,6,8}>` + `k_ht_lin_grp` +
  `k_ht_lin_grp_rt` (linear gemm), `k_ht_moe_*` (dn/dn2/dn2w/dn2x/dn3/
  gu/gu2/gu2p/gu2w/gu3/gu3e — MoE gemv), `k_ht_items_group<2|4>` /
  `k_ht_items_slots` (build the `HtItem` descriptor array), `k_ht_rot_*`
  (activation rotation: `rot_in`, `rot_out`, `rot_out_f32`, `rot_rows`,
  `rot_rows_h`, `rot_rows_xsum`), `k_ht_tg/tg3/tg_sum`. Shared structs:
  `HtItem`, `HtSd`, `HtFold`, `HtGrp`, `HtRt`.
- **[RV] Env knobs:** `HALOGEN_HT_AB_GRP`, `HALOGEN_HT_DEC_FUSE`,
  `HALOGEN_HT_DEC_GROUP`, `HALOGEN_HT_DEQ_KFAST`, `HALOGEN_HT_DEQ_PRE`,
  `HALOGEN_HT_DEQ_W16`, `HALOGEN_HT_DN1_PIN`, `HALOGEN_HT_DN_FOLD`,
  `HALOGEN_HT_DN_VW`, `HALOGEN_HT_GEMV_KL`, `HALOGEN_HT_GRP4_ROWS`,
  `HALOGEN_HT_GU3E`, `HALOGEN_HT_GU_FORM`, `HALOGEN_HT_GU_SPLIT`,
  `HALOGEN_HT_GU_XS`, `HALOGEN_HT_GU_XS_PAD`, `HALOGEN_HT_GU_XS_PAD3`,
  `HALOGEN_HT_LIN_BLOCKS`, `HALOGEN_HT_LIN_DIRECT`, `HALOGEN_HT_LIN_GRP`,
  `HALOGEN_HT_LIN_PIN`, `HALOGEN_HT_LIN_TM56`, `HALOGEN_HT_MOE_DNX`,
  `HALOGEN_HT_MOE_DV`, `HALOGEN_HT_MOE_GEMM`, `HALOGEN_HT_MOE_M128`,
  `HALOGEN_HT_MOE_M128_DN`, `HALOGEN_HT_MOE_M128_GU`, `HALOGEN_HT_MOE_SUM`,
  `HALOGEN_HT_MOE_V2`, `HALOGEN_HT_MOE_WPE`, `HALOGEN_HT_SD_RIDE_ROWS`,
  `HALOGEN_HT_TG_BLOCKS`, `HALOGEN_HT_TG_DEC`, `HALOGEN_HT_TG_M128`,
  `HALOGEN_HT_TRUNK_GEMM`, `HALOGEN_HT_TRUNK_ORIG`,
  `HALOGEN_HT_TRUNK_RESIDENT` (renamed `HALOGEN_PREFILL_KEEP_TRUNK`,
  `:497`), `HALOGEN_HT_U24`.

---

## 7. i4r (code 23) — 4-bit rows with in-payload scale + f16 planes

- **[RV] Tensor set (96):** the trunk MoE expert weights only — 48
  `layers.N.mlp.experts.gate_up_proj.weight` [512, 1280, 2560] and 48
  `layers.N.mlp.experts.down_proj.weight` [512, 2560, 640]. (The MTP
  head's experts are q4c, §3; the shared experts are ht, §6.)
- **[RV] Payload size formula:** `numel × 33/64` — 33 bytes per
  64-element block = **32 bytes of packed nibbles + 1-byte scale**
  (verified on all 96 entries: gate_up 865,075,200 = 512×1280×2560×33/64;
  down_proj 432,537,600). The engine validates group, shape and size:
  `checkpoint: %s: an I4R tensor whose group, shape or size this engine
  does not read` (`binutils-strings.txt:3249`) and `%sexperts: I4R needs
  both projections in I4R ([%d, %d, %d] gate_up, [%d, %d, %d] down),
  64-B aligned, on the pinned path (got %s / %s)` (`:858`).
- **[RV] Companion planes:** `gate_up_proj` uses the **shared**
  `layers.N.mlp.experts.su` [2560] (one f16 vector for all 512 experts —
  the 48 `.su` entries of §2) plus its own `.svh` [1280];
  `down_proj` carries its own `.suh` [640] + `.svh` [2560]. The
  `k_i4r_rows<K>` kernels are instantiated exactly at the two K values
  (`k_i4r_rows<2560>` and `k_i4r_rows<640>`).
- **[D-inferred] Dequant formula:** 4-bit nibbles (32 B per 64-element
  block) scaled by the 1-byte in-payload scale (encoding not observed on
  the x86 side — fp8/u8 candidates) and the f16 `.su`/`.svh` planes,
  which the kernels take alongside the payload
  (`k_i4r_rows<2560>(const u8*, long, int, const DF16*, int, const int*,
  float*)`). Exact arithmetic is GPU-side.
- **[RV] Kernel family:** `k_i4r_fold(const float*, int, const float*,
  const DF16*, const u16*, const float*, u16*)`, `k_i4r_fold_rows`,
  `k_i4r_fold_scatter_norm`, `k_i4r_gu_epi`, `k_i4r_rows_epi`,
  `k_i4r_rows_fl(_g<2|4>/_o)`, `k_i4r_rows_gl<1|2>(_g<1|2,2|4>/_o/_sdp/
  _gl_sdp)`, `k_i4r_rows_rot`, `k_i4r_zscan(const u16*, long, int*)`.
  Shared structs: `HtItem` (the ht descriptor is reused) and `I4rSd`.
- **[RV] Env knobs:** `HALOGEN_I4R_ARITH`, `HALOGEN_I4R_DEC_GROUP`,
  `HALOGEN_I4R_DN_FOLD`, `HALOGEN_I4R_FOLD_BATCH`, `HALOGEN_I4R_FUSE`,
  `HALOGEN_I4R_GRP4_ROWS`, `HALOGEN_I4R_HFUSE`, `HALOGEN_I4R_LAST`,
  `HALOGEN_I4R_SDFUSE`, `HALOGEN_I4R_SORTED`; the MoE path logs under
  the `moe_i4r_:` prefix (`:353`, `:1498`, `:2151`).

---

## 8. The x86 gemv dispatch chain (how a tensor's kernel is chosen)

- **[D]** `gemv-dispatcher-FUN_01f9b1b0.json`
  ([dtypes/gemv-dispatcher-FUN_01f9b1b0.json](../evidence/flash_serve/dtypes/gemv-dispatcher-FUN_01f9b1b0.json),
  source string `/src/halogen/src/flash_ops.h`) —
  `FUN_01f9b1b0(long *op, long input, long weight, uint N, int tokens,
  ulong K)` tries the prebuilt kernel handles on the tensor's op object
  in priority order:
  1. `FUN_01f991e0(op[0xe], …)` — gated by the `DAT_02129870` one-time
     init (decompile pending, Gaps);
  2. `FUN_0202cda0`
     ([dtypes/gemv-bf16-smallbatch-FUN_0202cda0.json](../evidence/flash_serve/dtypes/gemv-bf16-smallbatch-FUN_0202cda0.json))
     — the bf16 small-batch path: requires `K % 32 == 0`, tokens 1–8 and
     `op[0] ≠ 0`; the kernel handle comes from the `HALOGEN_LQ4W_SH`
     getter `FUN_0205e4d0` and the grid is selected by a `switch` on the
     token count (1–8) and N thresholds (`N < 9`, `N > 0x1407`,
     `K < 0x1400`, `K > 0xa1f`);
  3. `FUN_02031950` (slot `op[2]`,
     [dtypes/gemv-slot2-FUN_02031950.json](../evidence/flash_serve/dtypes/gemv-slot2-FUN_02031950.json)),
  4. `FUN_02033560` (slot `op[5]`,
     [dtypes/gemv-slot5-FUN_02033560.json](../evidence/flash_serve/dtypes/gemv-slot5-FUN_02033560.json)),
  5. `FUN_02034ef0` (slot `op[3]`,
     [dtypes/gemv-slot3-FUN_02034ef0.json](../evidence/flash_serve/dtypes/gemv-slot3-FUN_02034ef0.json)),
  6. `FUN_02036b00` (slot `op[4]`,
     [dtypes/gemv-slot4-FUN_02036b00.json](../evidence/flash_serve/dtypes/gemv-slot4-FUN_02036b00.json))
     — each a per-variant launcher with the same token-count `switch`
     and an 8-token chunking loop when called from the dispatcher tail;
  7. inline `__hipPushCallConfiguration` blocks for the `N % 8` /
     `N % 16` / `N % 32` grid variants (Ghidra renders the kernel-arg
     pushes as the non-returning `__hipPushCallConfiguration` pattern);
  8. `FUN_0203c2d0` (slot `op[1]`, tokens ≥ 9,
     [dtypes/gemv-slot1-multitok-FUN_0203c2d0.json](../evidence/flash_serve/dtypes/gemv-slot1-multitok-FUN_0203c2d0.json)).
- **[D]** Prefill paths: `HALOGEN_FLASH_PREFILL_GEMM` /
  `HALOGEN_FLASH_BF16_GEMM` gate the bf16 GEMM entries
  (`FUN_01ff4e10`, `FUN_021113d0`, `FUN_020215d0`, `FUN_02021700`,
  `FUN_01ff66e0`, `FUN_02003010`) reached from the same dispatcher.
- **[D]** The per-dtype variant classifier `FUN_02059c00(op, input,
  weight)` (cached in `DAT_02129cd0`, consumed by `FUN_0202cda0` and
  `FUN_0203c2d0`) and `FUN_02058530` (cached in `DAT_02129c60`) are not
  yet decompiled (Gaps) — the slot↔dtype assignment beyond the bf16
  slots is therefore inferred from the engine spec §3 handle set
  (`DAT_02127a38/40/48/68/70` in `FUN_01f766f0`).
- **[D]** Supporting helpers decompiled this pass:
  `tensor-name-hash-lookup-FUN_020c4420.json` (the name→tensor hash-map
  lookup used to resolve `op` objects), `error-exit-helper-FUN_020c2460.json`
  (`fprintf(stderr) + exit(1)`), `gguf-magic-check-FUN_020c4a30.json`
  (4-byte read compared against `0x46554747` = `GGUF`),
  `kfd-topology-FUN_020bf500.json` (reads
  `/sys/class/kfd/kfd/topology/nodes/%d/properties`),
  `meminfo-rlimit-FUN_020bf730.json` (rlimit-based memory sizing), and
  `search-procedures-kernels.json` (the procedure-table search showing
  the stripped symbol set).

---

## 9. Evidence index (`evidence/flash_serve/dtypes/`)

| file | role |
|---|---|
| `table-dump.tsv` | the 2015-entry entry-table dump of the real checkpoint (histogram source) |
| `verify-nbytes-formulas.awk` | re-checks every table entry's `nbytes` against the §1 formulas (0 mismatches) |
| `gemv-dispatcher-FUN_01f9b1b0.json` | the gemv dispatcher (§8) |
| `gemv-bf16-smallbatch-FUN_0202cda0.json` | bf16 small-batch launcher (`K % 32`, tokens 1–8) |
| `gemv-slot2-FUN_02031950.json`, `gemv-slot5-FUN_02033560.json`, `gemv-slot3-FUN_02034ef0.json`, `gemv-slot4-FUN_02036b00.json`, `gemv-slot1-multitok-FUN_0203c2d0.json` | per-slot variant launchers (§8) |
| `env-lq4w-sh-getter-FUN_0205e4d0.json` | `HALOGEN_LQ4W_SH` env getter |
| `tensor-name-hash-lookup-FUN_020c4420.json`, `error-exit-helper-FUN_020c2460.json`, `gguf-magic-check-FUN_020c4a30.json`, `kfd-topology-FUN_020bf500.json`, `meminfo-rlimit-FUN_020bf730.json` | supporting helpers (§8) |
| `search-procedures-kernels.json` | procedure-table search (stripped-binary context) |

All JSON files are raw `rea` CLI evidence envelopes (`{"evidence_id":…}`)
from the Ghidra 12.1.4 provider against the reference binary, matching
the format of the previously committed evidence.

---

## 10. Gaps (dtypes-specific; the engine spec's register links here)

| Gap | Next step |
|---|---|
| q4c 64-byte tensor-header contents | dump the header bytes of a q4c tensor and diff across tensors (the x86 launchers never parse it) |
| `qparam` storage location (named in the q6g64/q4c/q8g64 validation strings; the on-disk entry's uninterpreted `u64` @152 is the candidate) | read @152 for tensors of each dtype and compare against the expected `qparam` (64 for q6g64/q8g64) |
| `FUN_01f991e0` (the `op[0xe]` gemv path) | decompile pending |
| `FUN_02059c00` / `FUN_02058530` variant classifiers | decompile to complete the slot↔dtype assignment |
| exact GPU-side value-decoding math (all quantised dtypes) | AMDGPU code-object disassembly (outside the x86 provider's reach) |
| i4r 1-byte in-payload scale encoding (fp8 vs u8) | code-object disassembly or a crafted-checkpoint probe |
| q8g64/q6g64 4-byte scale encoding (f32 vs u32) | same |
