# flash_serve `.hip_fatbin` inventory

Static extraction and parsing of the `.hip_fatbin` ELF section from the
reference binary (halogen-flash-server 0.18.21). No device loading was
performed (no `/dev/dri` available); this is a file-level analysis only.

## Source

- Binary: `/nix/store/fn6y2w20kbgjd8da8pwxbbc3gvmiycpy-halogen-flash-server-0.18.21/bin/flash_serve`
- Section: `.hip_fatbin`, file offset `0x62000`, size `0x1d3b008` (30,650,376 bytes)

## Extraction

`objcopy` requires a writable output context, so the section was dumped
via a copy of the binary:

```sh
cp /nix/store/fn6y2w20kbgjd8da8pwxbbc3gvmiycpy-halogen-flash-server-0.18.21/bin/flash_serve .tmp/fatbin-inventory/flash_serve
objcopy --dump-section .hip_fatbin=fatbin.bin .tmp/fatbin-inventory/flash_serve /dev/null
# fatbin.bin: 30,650,376 bytes
```

## Container format

The section is **not** a classic HIP fatbinary (magic `0x46626378`
`"x\`F"`). It is a concatenation of four **clang offload bundles**
(magic string `__CLANG_OFFLOAD_BUNDLE__`), one per GPU code object
bundle. The first bundle's entry table does not parse per the documented
clang layout (the fields at the expected entry-table offset read as
ASCII fragments of the target-triple strings, e.g. entry size
`0x363838782d74736f` = `"ost-x86_"`-style text), so entry enumeration
used a fallback: magic-string scan for bundle headers plus `strings`
harvesting of target IDs and mangled kernel symbols.

Contradicts the port spec's claim of 9 code objects — the section
contains **4** bundles.

## Bundles

All four bundles pair a `host-x86_64-unknown-linux-gnu` entry with a
`hipv4-amdgcn-amd-amdhsa--gfx1151` entry. **gfx1151 is the only GPU
target** in the binary.

| Bundle | Offset    | Size (bytes) | `.kd` kernel symbols |
|--------|-----------|--------------|----------------------|
| 0      | `0x0`     | 24,399,872   | 1,739                |
| 1      | `0x1745000` | 3,555,328  | 175                  |
| 2      | `0x1aa9000` | 499,712    | 19                   |
| 3      | `0x1b23000` | 2,195,464  | 67                   |
| **Total** |        | 30,650,376   | **2,000**            |

## Kernel families (from `.kd` symbols in the fatbin)

| Kernel base name    | Variants |
|---------------------|----------|
| `k_attn_qs`         | 32       |
| `k_attn_fa2`        | 30       |
| `k_dn_fused16`      | 19       |
| `k_dn_recur_rows`   | 14       |
| `k_attn`            | 7        |
| `k_dn_recur_step`   | 5        |
| `k_smpg_*` (8 names)| 8        |
| `k_rope*` (8 names) | 8        |
| `k_attn_fd8`        | 2        |
| `k_dn_gates`        | 1        |

The full symbol list (2,001 unique mangled names) is in
`.tmp/fatbin-inventory/halogen-symbols.txt`; per-bundle lists are in
`.tmp/fatbin-inventory/per-object-kernels.txt`.

## Cross-check against `binutils-strings.txt`

Evidence names follow the convention `k_<mangled>` (no leading
underscore); fatbin symbols are `_<mangled>.kd`, so matching was done by
checking `'_' + name[2:]` against the fatbin symbol set.

- Evidence `k_*` names in `docs/analysis/evidence/flash_serve/binutils-strings.txt`: **37**
- Found in fatbin: **34**
- Not found: **3** — `k_layernorm`, `k_norm`, `k_proj` (bare,
  non-mangled names; likely substrings of other mangled symbols rather
  than kernel entries in this section)

## Notes / caveats

- The 2,000 `.kd` symbols cover only the kernels embedded in this
  section; the binary's full kernel surface (e.g. `k_deq_*`,
  `k_ht_moe_*`, `k_linear_*` families) is also present in the fatbin
  symbol set and listed in the artifact files.
- Parsing script retained at `.tmp/fatbin-inventory/parse_bundle.py`.

## Implications for the port

- **Delivery**: the port's flake already packages the reference binary
  (`packages/halogen-flash-server`), so the section is extracted at
  build time (`objcopy --dump-section .hip_fatbin=…`) and installed to
  `$out/share/cold-cathode/flash_serve.hipfb`; no 30 MB blob is vendored
  into git.
- **Loading**: the reference itself uses static HIP registration
  (`hipFatBinSegment` / `hipLaunchKernel` imports, see
  [npu/hip-imported-symbols.json](../npu/hip-imported-symbols.json));
  the port's equivalent is `hipModuleLoadData` + `hipModuleGetFunction`
  + `hipModuleLaunchKernel` (the `hipModule*` family must be added to
  `hip.rs`'s FFI). At startup the port parses the section (magic-scan
  for the four bundle headers, extract the `gfx1151` payloads) and
  builds a mangled-name → `hipFunction_t` registry.
- **Target constraint**: `gfx1151` only — matches the user's host (the
  reference runs there); the port inherits this constraint until the
  bundle gains other targets.
