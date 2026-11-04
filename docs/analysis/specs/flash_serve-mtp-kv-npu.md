# flash_serve MTP / KV-cache / NPU-XRT spec (engine-spec companion)

Scope: the three engine subsystems left unmapped by
[`flash_serve-engine.md`](flash_serve-engine.md) after §1 (`.hgn` layout)
and §2 (architecture constants) were resolved — the **MTP head +
speculative decoding**, the **KV-cache management**, and the **NPU/XRT
backend**. This complements §3–§8 of the engine spec and does not restate
them.

**File choice (documented per task brief):** these subsystems are specified
in a new companion file rather than as new sections inside
`flash_serve-engine.md`, because (a) the engine spec is concurrently owned
by the forward-path agent (§1/§2) and parallel edits to one file risk
clobbering, and (b) the engine spec is already ~480 lines. The §4
`HALOGEN_MTP_DEPTH` contradiction entry in the engine spec is **resolved
here** (see §1.3); the engine spec's §4/§5 text stays as written by its
owner.

Provenance markers (same vocabulary as the engine spec): **[RV]** —
runtime-verified / shipped-string / shipped-file observed; **[D]** —
Ghidra-decompile-derived (Ghidra 12.1.4 via the `rea` CLI against
`flash_serve`, sha256
`7c92fe3f8ac544c4fb690a69e9ce359fa7b8d453a8eab8e9a1289c6f2309c012`).
New evidence lives under
[`evidence/flash_serve/mtp/`](../evidence/flash_serve/mtp/),
[`evidence/flash_serve/kv-cache/`](../evidence/flash_serve/kv-cache/) and
[`evidence/flash_serve/npu/`](../evidence/flash_serve/npu/); previously
committed evidence under `evidence/flash_serve/engine/` and
`evidence/flash_serve/decompile/` is linked inline.

---

## 1. MTP head + speculative decoding

### 1.1 Head file and tensor set

- **[RV]** The head is a separate `.hgn` — `qwen38-flash-next-mtp.hgn`
  from the weights repo, named by `HALOGEN_MTP_HEAD`; a llama.cpp MTP
  draft GGUF is not read (engine spec §1/§4;
  `binutils-strings.txt:957`). `HALOGEN_MTP=0` turns the head off; the
  connection handler refuses a drafter without it (`asked for drafter ` +
  ` (MTP head not staged, HALOGEN_MTP=1)`,
  [engine/conn-handler-FUN_01ee0e20.json](../evidence/flash_serve/engine/conn-handler-FUN_01ee0e20.json)).
- **[D]** **Head loader/validator `FUN_01f7dc50`**
  ([mtp/mtp-head-validator-FUN_01f7dc50.json](../evidence/flash_serve/mtp/mtp-head-validator-FUN_01f7dc50.json))
  — resolves the head's tensors by name and caches them in the engine
  object (fields `+0x276c0…+0x27890+`), validating:
  | tensor | check |
  |---|---|
  | `mtp.pre_fc_norm_embedding.weight` | resolved by name (RMSNorm over the draft token's embedding) |
  | `mtp.pre_fc_norm_hidden.weight` | resolved by name (RMSNorm over the trunk hidden state) |
  | `mtp.fc_embedding.weight`, `mtp.fc_hidden.weight` | the projection pair applied to the two normalised inputs |
  | `mtp.hyper_connection_mixer.hc_norm.weight`, `…input_mix_weight_up.weight`, `…input_mix_weight_down.weight` | resolved by name |
  | `mtp.layers.0.*` | the head's single decoder layer |
  | `mtp.layers.0.experts` | dtype must be `5` (`q4c`) or `11` (`q8s32`) — the **Q4C-P** pair; else `mtp.layers.0.experts: the head's experts must be Q4C-P (got %s / %s)` (dtype names via `FUN_020bd2b0`) |
  | `…gate_up_proj.weight` | `a fused gate_up is read in place as Q4C-P or bf16 only` |
  | FFN layout | `the head's FFN layout differs from the trunk's` — the head's fused/split FFN flag must equal the trunk's (`engine+0x9e8+0x700`) |
- **[D]** How the head consumes the trunk state: the tensor set defines the
  canonical MTP combine — `RMSNorm(embedding_of_draft_token)` and
  `RMSNorm(trunk_hidden_state)` are projected through `fc_embedding` /
  `fc_hidden` and merged (hyper-connection mixer), then run through the
  head's single `mtp.layers.0` decoder layer (full attention — engine
  spec §2 counts it as the 13th attention layer) to produce draft
  logits. The draft logits are sampled by the shared sampling pipeline
  (§1.2), not a separate head matmul string — the head reuses the trunk
  `lm_head` table (no `mtp.lm_head.*` strings exist).

### 1.2 Draft sampling and the verify path

- **[D]** **`draft_sample` — `FUN_0201c790`**
  ([mtp/draft-sample-FUN_0201c790.json](../evidence/flash_serve/mtp/draft-sample-FUN_0201c790.json),
  source `/src/halogen/src/flash_model.hip`):
  - signature `(float temperature, slot* , …, void* token_out, uint q_row)`;
    `q_row` selects the speculative query row (`-1` = the live sequence,
    `0..63` = a ring row; `q_row > 7` with no ring →
    `draft_sample: q row %d was not kept`, exit 1);
  - refuses when the head is not staged
    (`draft_sample: the MTP head is not staged (HALOGEN_MTP=1)`,
    checks `slot+0x119c`/`slot+0x13a0`) and when `temperature == 0`
    (`temperature 0 is the GREEDY path and must stay on argmax()`);
  - stages a 0x28-byte parameter block (temperature·`tskew`, `qoff`,
    top-k/top-p/min-p/seed fields) into the slot's device parameter
    buffer `slot+0xdb8` via `hipMemcpy(…, 0x28, HostToDevice)`;
  - launches the head forward+sample via `FUN_0201ba20`
    ([mtp/mtp-sample-launcher-FUN_0201ba20.json](../evidence/flash_serve/mtp/mtp-sample-launcher-FUN_0201ba20.json)):
    pre-fills the token buffer `slot+0xdc0` with `0xff` sentinels
    (`hipMemsetAsync`), then dispatches the kernel
    (`__hipPushCallConfiguration` + `FUN_0201be90`, the `k_smpg_*` family
    of engine spec §6);
  - reads the sampled token back from `slot+0xdc0` (4 B
    DeviceToHost) and the logprob from `slot+0xe00` (sampled) or
    `slot+0xe08` (greedy), adding `qoff` to the returned logprob;
  - **falsifier mode** [D]: when the S2 knobs are non-default the first
    call prints `draft_sample: FALSIFIER ACTIVE: nucleus %.4f, qoff
    %+.4f, tskew %.4f. Every proposal from this process is deliberately
    wrong (METHOD 37).` — a built-in self-test that forces wrong drafts;
  - per-slot counters `slot+0xea8` (calls) / `slot+0xeb0` (ns) are
    updated.
- **[D]** S2 tuning knobs (guarded getters, all `strtof`):
  `HALOGEN_S2_NUCLEUS` ([mtp/s2-nucleus-FUN_0201cd70.json](../evidence/flash_serve/mtp/s2-nucleus-FUN_0201cd70.json)),
  `HALOGEN_S2_QOFF` ([mtp/s2-qoff-FUN_0201cda0.json](../evidence/flash_serve/mtp/s2-qoff-FUN_0201cda0.json)),
  `HALOGEN_S2_TSKEW` ([mtp/s2-tskew-FUN_0201cdd0.json](../evidence/flash_serve/mtp/s2-tskew-FUN_0201cdd0.json)).
- **[D]** **`spec_verify` — `FUN_02008f60`**
  ([mtp/spec-verify-FUN_02008f60.json](../evidence/flash_serve/mtp/spec-verify-FUN_02008f60.json)):
  validates `1 <= n <= slot[+0x14]` (the request's `HALOGEN_SPEC_N`),
  requires the verify rings (`slot[+0x119d]` — set by `HALOGEN_MTP=1` or
  a lookup/grammar drafter), requires a warm sequence (`slot[+0x630] >
  0`) and `slot[+0x3dc] == 0`; then sets `slot[0]=1` and delegates to
  the forward core. Error wording: `spec_verify(n=%d): needs the verify
  rings (HALOGEN_MTP=1 or a lookup / grammar drafter), 1 <= n <= %d
  (HALOGEN_SPEC_N) and a warm sequence (pos %d)`.
- **[D]** **Forward core — `FUN_02008fd0`**
  ([mtp/forward-core-FUN_02008fd0.json](../evidence/flash_serve/mtp/forward-core-FUN_02008fd0.json)):
  the shared `forward(n tokens)` entry used by `GEN`/`spec_verify`:
  validates `n ∈ [1, ring]`, clamps out-of-vocab token ids (`> 0x3c9ff`)
  to 0 with `token id %d (row %d of %d) reached %s, outside the
  vocabulary; it is read as 0 and the request is ended`, enforces
  `sequence would reach %d > max_seq %d`, records the speculative
  commit-window fields (`+0x8c4/+0x8c8/+0x8cc` from the ring counters
  `+0x8b8/+0x8bc/+0x8c0`, only when the delta is a multiple of 64),
  stages the token ids into `slot+0xbc8` and uploads them with
  `hipMemcpyAsync(slot+0xda8, …, n·4, HostToDevice)`, then launches the
  batched forward kernel (`__hipPushCallConfiguration(n|0x100000000, …)`).
- **[RV]** The accept/reject kernels themselves: `halogen::k_accept_prob_f32`
  (`binutils-strings.txt:225`) computes per-row acceptance probabilities
  and `halogen::k_smpg_accept_final` (:1143) commits the accepted prefix
  (engine spec §6). The accept gate threshold is `HALOGEN_MTP_GATE_P`
  (§1.3); a sampled draft is accepted with probability derived from
  `k_accept_prob_f32` against the gate (the classic speculative-sampling
  criterion), and `k_smpg_accept_final` writes the accepted tokens into
  the sequence.
- **[D]** Per-request speculative accounting (emitted by the connection
  handler trace, `req %ld spec: %ld rounds %ld accepted; chain %ld rounds
  %ld drafted %ld accepted; lookup %ld rounds %ld drafted %ld accepted;
  grammar %ld rounds %ld drafted %ld accepted; %ld serial; list %ld
  tokens %ld off; %ld lookup chains kept under a hook (D3); %ld joint
  rounds live (D4)`) — counters read from the request object at
  `+0x8/+0x10` (MTP rounds/accepted), `+0x88/0x90/0x98` (chain),
  `+0x48/0x68/0x70` (lookup), `+0x78/0x80` (grammar), `+0x18` (serial),
  `+0xc0/200` (list tokens/off), `+0xe8` (D3), `+0xf0` (D4). These are
  the `D`-line fields `rounds`/`commit`/`pld_rounds`/`pld_accepted`/
  `pld_drafted`/`n_composed` of the frozen spec (engine spec §8).
- **[RV]** `joint_verify(%d segments, %d rows): needs the head's rings
  (HALOGEN_MTP), 1 <= segments <= %d and rows <= %d (HALOGEN_SPEC_N)`
  (`binutils-strings.txt:3423`) — the multi-segment variant of
  `spec_verify` (the D4 "joint rounds").

### 1.3 `HALOGEN_MTP_DEPTH` — contradiction resolved

The engine spec §4 carried a conflict: frozen CLI spec says default **3**,
an earlier decompile suggested **2**. The full env-parsing block in
`main` (`FUN_01ede7e0`,
[decompile/main-FUN_01ede7e0.json](../evidence/flash_serve/decompile/main-FUN_01ede7e0.json),
lines ~854–935) **[D]** settles it:

```c
local_19d4._0_4_ = 2;                       /* DEPTH default */
pcVar11 = getenv("HALOGEN_MTP_DEPTH");
if (pcVar11 != NULL) {
    iVar10 = strtol(pcVar11);               /* DEPTH = env value */
    /* valid iff 0 < D and (SPEC_N >= D or D == 1 or MTP off), else:
       "flash_serve: HALOGEN_MTP_DEPTH=%s needs 1 <= D and HALOGEN_SPEC_N >= D + 1 (is %d)" */
}
iVar10 = 2;                                 /* unset → 2 */
…
pcVar11 = getenv("HALOGEN_MTP_DEPTH");
uVar25 = (pcVar11 == NULL) * 3;             /* DEPTH_HI default = 3 iff DEPTH unset, else 0 */
if (iVar10 < 2 || uVar25 <= iVar10) uVar25 = 0;  /* HI only when DEPTH ≥ 2 and HI > DEPTH */
```

- **`HALOGEN_MTP_DEPTH` default = 2** (the value the serve banner renders
  as `MTP depth-%d`). The frozen spec's **3 is `HALOGEN_MTP_DEPTH_HI`** —
  the upper speculative depth used when `DEPTH` is unset — conflated with
  `DEPTH`. The port must default `DEPTH=2`, `DEPTH_HI=3`.
- `HALOGEN_MTP_DEPTH_HI` range-check: `0 <= D <= 8`
  (`flash_serve: HALOGEN_MTP_DEPTH_HI=%s needs 0 <= D <= 8`); `HI` is
  forced to 0 unless `DEPTH ≥ 2` and `HI > DEPTH`.
- Validation of `DEPTH`: `1 <= D` and `HALOGEN_SPEC_N >= D + 1` (the
  `D == 1` and MTP-off exemptions are Ghidra-visible in the branch).
- **`HALOGEN_MTP_GATE_P`** default `0.9`; the effective gate is `0.0`
  (off) unless `GATE_P` is explicitly set **or** `DEPTH`/`DEPTH_HI` are
  set (then `0.9`), and forced to `0.0` when `HALOGEN_MTP` is off;
  range `[0, 1]` (`0 = off`).
- **Effective depth** used by the gate struct: `min(DEPTH, SPEC_N − 1)`,
  or `SPEC_N − 1` when `HALOGEN_MTP_DEPTH` is unset.
- Related defaults (same decompile, engine spec §4): `SAMPLED_DEPTH` 7
  (`HALOGEN_SAMPLED_DEPTH`, same validation wording), `HALOGEN_SAMPLED_GATE=t1[,t2]`
  (parsed by `FUN_01f25020`), PLD `N=3, K=3`, PLD adapt `W=32,
  floor=1.0, R=64`, `HALOGEN_PLD_LOOKBACK`, `HALOGEN_PLD_AGREE`,
  `HALOGEN_SPEC_ADAPT=w,floor,r`.
- The port (currently following the frozen spec's 3) must change
  `HALOGEN_MTP_DEPTH` default to **2** and add `HALOGEN_MTP_DEPTH_HI`
  default **3**.

---

## 2. KV-cache management

- **[D]** **Per-position sizing — `FUN_01f82410`** (engine spec §2,
  [engine/fun-01f82410.json](../evidence/flash_serve/engine/fun-01f82410.json)):
  per position the engine reserves `(K + V) · 13 + 0xb40` bytes, with
  `K = V = 0x400` (512 elements × bf16 = the 2 KV heads × 256 dim) in
  bf16 mode → **29 504 B/position** (the frozen spec's "about 29.5 KiB a
  position, 262144 = 7.2 GiB"), and `K = 0x208` (512 B fp8 payload + 8 B
  scales), `V = 0x220` (512 B + 32 B scales) in KV8 mode → 16 816
  B/position. The `× 13` is the 12 main attention layers + 1 MTP layer
  (engine spec §2).
- **[D]** **KV8 mode — `FUN_01f6ff70`**
  ([kv-cache/kv8-mode-FUN_01f6ff70.json](../evidence/flash_serve/kv-cache/kv8-mode-FUN_01f6ff70.json)):
  `HALOGEN_KV8` ∈ {unset → 0, `0` → 0, `1` → 1 (the 8-bit KV cache), `2`
  → 2}; any other value exits with `HALOGEN_KV8=%s: 1 (the 8-bit KV
  cache) or unset`. The value is memoised (`__cxa_guard`) and consulted
  by the sizing function and the allocator. (Mode `2` exists in the
  parser; the sizing/allocator branches only special-case `1` — mode 2
  currently sizes like bf16.)
- **[D]** **Pool budget — `FUN_02019610`**
  ([engine/fun-02019610.json](../evidence/flash_serve/engine/fun-02019610.json)):
  computes the device-side budget as host RAM − resident weights −
  reserve (n-gram page cache + OS), charged additionally by
  `HALOGEN_PREFILL_KEEP_TRUNK=1` (trunk kept unpacked beside the pool)
  and the vision tower scratch (`HALOGEN_VISION_MAX_PIXELS`), then
  derives the pool position count; cap `0x1000000` positions rounded to
  256-position units (engine spec §5,
  [decompile/ctx-validator-FUN_01f24220.json](../evidence/flash_serve/decompile/ctx-validator-FUN_01f24220.json)).
- **[D]** **Slot allocator — `FUN_020120b0` (`slots_alloc_pooled`)**
  ([kv-cache/slots-alloc-pooled-FUN_020120b0.json](../evidence/flash_serve/kv-cache/slots-alloc-pooled-FUN_020120b0.json)):
  - `slots_alloc_pooled(B)` with `B ∈ [1, 64]`
    (`slots_alloc_pooled(%d): 1 <= B <= %d`); on a non-pooled model
    `slots_alloc(%d) on a pooled model: use slots_alloc_pooled` is the
    sibling error (engine spec §5);
  - refuses when existing non-pooled slots hold private KVs
    (`slots_alloc_pooled(%d): %d slots already hold private KVs`) unless
    switching to pooled mode (`engine+0x345`);
  - per new slot (slot record size `0x160` B, stored in the vector at
    `engine+0x328…0x338`, count at `+0x340`) it allocates the device
    buffers via `FUN_01f82130`
    ([kv-cache/device-alloc-FUN_01f82130.json](../evidence/flash_serve/kv-cache/device-alloc-FUN_01f82130.json),
    a `hipMalloc` wrapper with global byte/count accounting):
    | size | role |
    |---|---|
    | `0x300000` (3 MiB) | activation/logit staging block |
    | `0x14000` (80 KiB) | secondary staging |
    | `0x300` (768 B) | small control block |
    | `0x10` + `0x2d000` (16 B + 184 KiB) | state block pair |
    | `0xf2800` (993 280 B) | **the slot's KV region** — the stride `draft_sample` uses to address per-row KV (`q_row · 0xf2800 + ring_base`, §1.2) |
    | `0x5000` (20 KiB) | alternate state block |
  - prints `slots: %d -> %d POOLED, %.1f MiB each of O(1) state; the KV
    is ONE pool of %d positions (%.1f MiB) carved into per-request
    regions` — confirming the one-pool-carved-into-regions model of the
    serve banner (engine spec §3/§5); the per-slot MiB figure is the
    summed allocations above divided by the new-slot count, and the
    per-position bytes are recomputed with the KV8 branch (`0x400`
    bf16 / `0x214` KV8 per unit).
- **[D]** **Per-request interaction** (from §1.2/§1.3 decompiles): the
  request/slot object carries the KV ring pointers at `+0x90` (empty
  case) / `+0x98 + i·8` (ring of up to `0x40` block pointers), the
  sequence length at `+0x298`, the max-seq bound at `+0x390` (with the
  `(the bound slot's KV region)` qualifier when the slot is a bound
  clone), the current position at `+0x630`, and the keep-flag ring at
  `+0xf94/+0xf9c` (which rows' K/V were kept — `draft_sample: q row %d
  was not kept`). The forward core uploads token ids to `slot+0xda8`
  and the attention kernels write K/V into the slot's region at the
  position-derived offset.
- **[RV]** Admission/waiting and the composable-context subsystem are
  engine spec §5 (conn-handler); the fork/keep accounting
  (`HALOGEN_FORK`/`HALOGEN_FORK_TTL_S`, `HREL`) is unchanged by this
  pass. `HALOGEN_KV_POOL_FIT` and the pool-fit error wording
  (`kv pool: %lld positions need ~%.1f GiB (plus %.1f of margin) and %s
  cannot spare it. LOWERING THE POOL TO %lld (~%.1f GiB).`,
  `binutils-strings.txt:89`) belong to the budget function above.
- **[D]** **Speculative-decode interaction**: the verify rings
  (`slot+0x119d`, sized by `HALOGEN_SPEC_N`, `slot+0x14`) hold the
  drafted-token rows whose K/V are written speculatively and rolled back
  on rejection; `spec_verify(n)` re-runs the forward over `n` rows
  (§1.2), and the commit-window fields (`+0x8c4…0x8cf`) record how many
  speculative positions become committed. The MTP head's own KV (the
  13th attention layer) lives in the same per-slot region (the `× 13`
  in `FUN_01f82410`).

---

## 3. NPU / XRT backend

- **Key finding [RV]:** `flash_serve` itself **does not use XRT** — a
  case-insensitive whole-word search for `xrt`/`npu` in its string table
  returns zero matches, and its ELF imports contain no XRT symbols. Its
  device backend is the **AMD HIP stack**: `DT_NEEDED libhipblaslt.so.1`
  + `libamdhip64.so.7`, RUNPATH pointing at `hipblaslt-7.2.4` and
  `clr-7.2.4` (AMD OpenCL runtime), and 9 embedded AMDGPU code objects in
  `.hipFatBinSegment` (engine spec §3). The **XRT integration surface of
  the product lives in the sibling binary `halogen-npu`** (same nix
  package), whose contract is frozen in
  [`halogen-npu.md`](halogen-npu.md). The port's "NPU backend" therefore
  has two targets, specified separately below.

### 3.1 flash_serve's device surface (HIP + hipBLASLt)

- **[RV]** Complete HIP/hipBLASLt import list:
  [npu/hip-imported-symbols.json](../evidence/flash_serve/npu/hip-imported-symbols.json)
  (36 HIP runtime symbols + the 15-function hipBLASLt matmul family +
  3 `hipblaslt_ext::` helpers). The Rust port's device layer must cover:
  device lifecycle (`hipSetDevice`/`hipGetDevice`/`hipGetDeviceCount`/
  `hipGetDevicePropertiesR0600`/`hipDeviceGetPCIBusId`), memory
  (`hipMalloc`/`hipMemcpy{,Async,2D}`/`hipMemset{,Async,D32Async}`/
  `hipHostMalloc`/`hipHostRegister`/`hipHostGetDevicePointer`), streams
  and events (`hipStreamCreateWith{Flags,Priority}`/`hipStreamSynchronize`/
  `hipStreamWaitEvent`/`hipEvent*`), kernel launch (`hipLaunchKernel` +
  the `__hipPushCallConfiguration` fast path), occupancy
  (`hipOccupancyMaxActiveBlocksPerMultiprocessor`), and the hipBLASLt
  descriptor/heuristic/matmul path (incl. the tuning-file handling,
  engine spec §3).
- **[D]** Device selection — `FUN_01f700f0` (engine spec,
  [decompile/gpu-selector-FUN_01f700f0.json](../evidence/flash_serve/decompile/gpu-selector-FUN_01f700f0.json)):
  enumerates devices (`hipGetDeviceCount` → `hipGetDevicePropertiesR0600`
  → `hipDeviceGetPCIBusId`), honours `HIP_VISIBLE_DEVICES` /
  `ROCR_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` (rewriting/unsetting
  them), and binds with `hipSetDevice`; the guard `FUN_01f70090`
  produces the no-GPU message (engine spec §3).
- **[D]** Weight transfer path: the checkpoint is mmap'd and registered
  once (`FUN_020bea30`, engine spec §1); the unpinned trunk stages
  through a compacting `hipMemcpy` path measured at 11.4 GB/s
  (`trunk NOT pinned (HALOGEN_FLASH_PIN_TRUNK=0), bf16 staging and the
  compacting hipMemcpy path, measured at 11.4 GB/s`,
  `binutils-strings.txt:1718`); pinned host buffers use
  `hipHostRegister`/`hipHostGetDevicePointer` (`HALOGEN_PIN_HOSTBUF`,
  `HALOGEN_WEIGHTS_LOCK`, `HALOGEN_LOCK_ANON` — engine spec §1/§7).
- **[D]** Kernel launch pattern (observed in every device-touching
  function, e.g. `FUN_0201ba20` §1.2, `FUN_02008fd0` §1.2): stage
  arguments into per-slot device buffers, then `__hipPushCallConfiguration`
  + the fatbin kernel entry; the 9 code objects supply the `k_attn_*`,
  `k_rope_*`, `k_smpg_*`, `k_argmax_*`, `k_accept_prob_f32` kernels
  (engine spec §3/§6).

### 3.2 halogen-npu's XRT surface (the NPU engine proper)

- **[RV]** `halogen-npu` `DT_NEEDED`s `libxrt_coreutil.so.2` (AMD XRT);
  the complete import surface — 20 `xrt::` symbols + 2 `_ITM_*` hooks —
  is catalogued in
  [npu/xrt-imported-symbols.json](../evidence/flash_serve/npu/xrt-imported-symbols.json)
  (derived from the existing
  [halogen-npu xrt-stub](../evidence/halogen-npu/xrt-stub/libxrt_coreutil-stub.c)
  and its frozen spec).
- **Device/session lifecycle** [RV, from the frozen halogen-npu spec]:
  `xrt::device(index)` → `xrt::elf(kernel_elf_path)` →
  `xrt::hw_context(device, elf)` → `xrt::ext::kernel(hw_context, name)` →
  `xrt::run(kernel)`; arguments bound with `xrt::run::set_arg_at_index(i,
  bo)`, launched with `run::start()`, synchronised with
  `run::wait(timeout_ms)` / polled with `run::state()`.
- **Weight/activation transfer** [RV]: `xrt::bo(device, size, flags,
  bank)` buffer objects, `bo::map()` for host access,
  `bo::sync(xclBOSyncDirection, size, offset)` for H2D/D2H transfers, and
  the sub-buffer constructor `xrt::bo(existing, offset, size)` for
  slicing weight blobs — i.e. one `bo` per tensor/activation with
  offset-sliced reuse, mirroring the HIP staging design.
- **[D]** Launch wrapper `FUN_001afec0` (halogen-npu,
  [npu/halogen-npu-xrt-run-launcher-FUN_001afec0.json](../evidence/flash_serve/npu/halogen-npu-xrt-run-launcher-FUN_001afec0.json)):
  iterates the bound-argument vector, binds each (`FUN_001af340` →
  `set_arg_at_index`), timestamps, calls `xrt::run::start()`, records
  the launch duration and bumps global launch counters — the pattern the
  Rust port reproduces per kernel invocation.
- **Port note:** `flash_serve` (the server being ported in this repo's
  current phase) needs **no XRT**; the XRT surface only matters when
  porting `halogen-npu` (Phase 3/4 of the full plan), where the frozen
  `halogen-npu.md` contract governs.

---

## 4. Gaps closed / remaining (this pass)

| item | status |
|---|---|
| engine spec §4 `HALOGEN_MTP_DEPTH` 2-vs-3 contradiction | **resolved** — default 2; the frozen spec's 3 is `HALOGEN_MTP_DEPTH_HI` (§1.3) |
| engine spec §4 "MTP-head loader function remains a gap" | **closed** — `FUN_01f7dc50` (§1.1); `FUN_01f24580`/`FUN_01f24690` remain the ctor steps but the loader is now identified |
| engine spec §5 KV-cache data structure | **closed** — slot records + per-slot device buffers + one-pool carving (§2) |
| remaining MTP unknown | the exact accept-probability formula inside `k_accept_prob_f32` (GPU kernel; needs the code-object disassembly, not the x86 binary) |
| remaining KV unknown | the rollback mechanics for rejected speculative rows (inferred from the keep-flag ring + commit-window fields; the kernel-side write/rollback is in the code objects) |
| remaining NPU unknown | none for `flash_serve` (HIP surface fully enumerated); `halogen-npu` XRT runtime behaviour stays unverified per its own spec's limitations (no real `libxrt_coreutil.so.2` on host) |
