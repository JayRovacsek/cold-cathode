//! GPU code-object loading (the reference's `.hip_fatbin` section).
//!
//! The reference binary `flash_serve` embeds its GPU code objects in the
//! `.hip_fatbin` ELF section as clang offload bundles (evidence
//! [`flash_serve-mtp-kv-npu.md`](../../../docs/analysis/specs/flash_serve-mtp-kv-npu.md)
//! and
//! [`fatbin-inventory.md`](../../../docs/analysis/evidence/flash_serve/engine/fatbin-inventory.md)):
//! four bundles targeting `gfx1151` only, ~2,000 `.kd` kernel symbols.
//! The Nix build extracts that section into the port's own
//! `$out/share/cold-cathode/flash_serve.hipfb` and bakes the path into
//! the binary (the `HALOGEN_FATBIN` compile-time default, overridable at
//! runtime), so the port can load the reference's shipped kernels through
//! the `hipModule*` API ([`crate::hip::ffi::hipModuleLoadData`]) instead
//! of shipping its own compiled kernels.
//!
//! This module is the foundation for GPU compute: it parses the bundle
//! container, extracts the kernel-name registry, and resolves kernel
//! names to `hipFunction_t` handles. No compute routing happens yet —
//! the CPU forward path stays behaviourally unchanged.
//!
//! Container format (probed against the extracted section, see the
//! inventory doc): each bundle starts with the 24-byte magic
//! `__CLANG_OFFLOAD_BUNDLE__`, followed by a `u64` entry count and
//! `count` variable-length entries `{u64 offset, u64 size, u64
//! triple_len, triple bytes}`. The documented clang layout terminates
//! each triple with a NUL byte; the reference's bundles omit it (the
//! documented reading misparses every entry after the first, which is
//! the inventory's "corrupted table"). [`parse_bundle_at`] therefore
//! tries the documented NUL-terminated variant first and falls back to
//! the no-NUL variant, so both the real file and a spec-conformant
//! bundle parse.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::env;
use std::ffi::CString;
use std::fs;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use crate::hip::{self, ffi};

/// The clang offload bundle magic.
const MAGIC: &[u8; 24] = b"__CLANG_OFFLOAD_BUNDLE__";

/// One entry of a bundle's entry table: the payload location (relative
/// to the bundle start) and the target triple naming the device it is
/// built for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BundleEntry {
    pub offset: u64,
    pub size: u64,
    pub triple: String,
}

/// A parsed bundle: its position in the containing blob plus its entry
/// table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bundle {
    pub start: usize,
    pub entries: Vec<BundleEntry>,
}

impl Bundle {
    /// The payload bytes of `entry` within `blob` (empty for size-0
    /// entries; `None` when the entry points outside the blob).
    pub fn entry_payload<'a>(&self, blob: &'a [u8], entry: &BundleEntry) -> Option<&'a [u8]> {
        let start = self
            .start
            .checked_add(usize::try_from(entry.offset).ok()?)?;
        let end = start.checked_add(usize::try_from(entry.size).ok()?)?;
        blob.get(start..end)
    }
}

/// Whether `triple` names an AMD GPU code object (as opposed to a host
/// entry). The reference's device triple is
/// `hipv4-amdgcn-amd-amdhsa--gfx1151`; the check accepts any `amdgcn`
/// triple so future targets load without a code change.
pub fn is_amdgcn_triple(triple: &str) -> bool {
    triple.contains("amdgcn")
}

fn u16_at(data: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(data.get(at..at + 2)?.try_into().ok()?))
}

fn u32_at(data: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(data.get(at..at + 4)?.try_into().ok()?))
}

fn u64_at(data: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_le_bytes(data.get(at..at + 8)?.try_into().ok()?))
}

/// The NUL-terminated string starting at `at` (without the terminator).
fn cstr_at(data: &[u8], at: usize) -> Option<&[u8]> {
    let end = data.get(at..)?.iter().position(|&byte| byte == 0)?;
    data.get(at..at + end)
}

/// Parses the entry table of the bundle starting at `at`, trying the
/// documented NUL-terminated entry variant first and falling back to the
/// no-NUL variant the reference's bundles actually use (see the module
/// docs). A bundle whose entry table parses under neither variant is
/// skipped by [`find_bundles`].
pub fn parse_bundle_at(blob: &[u8], at: usize) -> Option<Bundle> {
    let data = blob.get(at..)?;
    let entries = parse_entry_table(data, true).or_else(|| parse_entry_table(data, false))?;
    Some(Bundle { start: at, entries })
}

fn parse_entry_table(data: &[u8], nul_terminated: bool) -> Option<Vec<BundleEntry>> {
    let count = usize::try_from(u64_at(data, MAGIC.len())?).ok()?;
    // Every entry needs at least the 24-byte fixed part; a count that
    // cannot fit is a false magic hit or a corrupt table.
    if count == 0 || count > data.len() / 24 {
        return None;
    }
    let mut at = MAGIC.len() + 8;
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        let offset = u64_at(data, at)?;
        let size = u64_at(data, at + 8)?;
        let triple_len = usize::try_from(u64_at(data, at + 16)?).ok()?;
        at += 24;
        let triple_bytes = data.get(at..at.checked_add(triple_len)?)?;
        if triple_len == 0 || !triple_bytes.is_ascii() {
            return None;
        }
        at += triple_len;
        if nul_terminated {
            if data.get(at) != Some(&0) {
                return None;
            }
            at += 1;
        }
        // The payload must lie inside the bundle's own bytes; this is
        // what rejects the NUL-variant misreading of the reference's
        // no-NUL tables (the shifted offset overflows or runs past the
        // end).
        let end = offset.checked_add(size)?;
        if usize::try_from(end).ok()? > data.len() {
            return None;
        }
        entries.push(BundleEntry {
            offset,
            size,
            triple: String::from_utf8(triple_bytes.to_vec()).ok()?,
        });
    }
    Some(entries)
}

/// Finds every bundle in `blob` (the extracted `.hip_fatbin` section or
/// any other concatenation of clang offload bundles), keeping the ones
/// whose entry table parses.
pub fn find_bundles(blob: &[u8]) -> Vec<Bundle> {
    let mut bundles = Vec::new();
    let mut at = 0;
    while let Some(rel) = find_magic(blob, at) {
        if let Some(bundle) = parse_bundle_at(blob, rel) {
            bundles.push(bundle);
        }
        at = rel + MAGIC.len();
    }
    bundles
}

fn find_magic(blob: &[u8], from: usize) -> Option<usize> {
    (from..blob.len()).find(|&at| blob[at..].starts_with(MAGIC))
}

/// The `(triple, payload)` pairs of every amdgcn code object in `blob`,
/// skipping host entries.
pub fn amdgcn_payloads(blob: &[u8]) -> Vec<(String, &[u8])> {
    let mut payloads = Vec::new();
    for bundle in find_bundles(blob) {
        for entry in &bundle.entries {
            if !is_amdgcn_triple(&entry.triple) {
                continue;
            }
            if let Some(payload) = bundle.entry_payload(blob, entry) {
                payloads.push((entry.triple.clone(), payload));
            }
        }
    }
    payloads
}

// ---------------------------------------------------------------------------
// ELF64 symbol extraction (the kernel names one code object exports)
// ---------------------------------------------------------------------------

const SHT_SYMTAB: u32 = 2;
const SHT_DYNSYM: u32 = 11;
const STT_OBJECT: u8 = 1;
const STT_FUNC: u8 = 2;

/// The registry key for a code-object symbol: the `.kd` suffix clang
/// appends to the kernel-descriptor objects is stripped, so the mangled
/// function name and its `name.kd` alias resolve to the same kernel.
pub fn normalize_kernel_name(name: &str) -> &str {
    name.strip_suffix(".kd").unwrap_or(name)
}

/// The kernel names exported by one amdgcn code object (an ELF64
/// payload): every `STT_FUNC` symbol plus every `STT_OBJECT` symbol
/// named `*.kd`, normalised with [`normalize_kernel_name`] and
/// deduplicated. `None` when `elf` is not a little-endian ELF64 object.
pub fn elf_kernel_names(elf: &[u8]) -> Option<Vec<String>> {
    if elf.get(..4) != Some(b"\x7fELF") || elf.get(4) != Some(&2) || elf.get(5) != Some(&1) {
        return None;
    }
    let e_shoff = usize::try_from(u64_at(elf, 0x28)?).ok()?;
    let e_shentsize = usize::from(u16_at(elf, 0x3a)?);
    let e_shnum = usize::from(u16_at(elf, 0x3c)?);
    if e_shentsize < 64 || e_shoff.checked_add(e_shnum.checked_mul(e_shentsize)?)? > elf.len() {
        return None;
    }
    let mut sections = Vec::with_capacity(e_shnum);
    for index in 0..e_shnum {
        let sh = e_shoff + index * e_shentsize;
        sections.push((
            u32_at(elf, sh + 4)?,                           // sh_type
            usize::try_from(u64_at(elf, sh + 0x18)?).ok()?, // sh_offset
            usize::try_from(u64_at(elf, sh + 0x20)?).ok()?, // sh_size
            usize::try_from(u32_at(elf, sh + 0x28)?).ok()?, // sh_link
        ));
    }
    let mut names = BTreeSet::new();
    for &(sh_type, sh_offset, sh_size, sh_link) in &sections {
        if sh_type != SHT_SYMTAB && sh_type != SHT_DYNSYM {
            continue;
        }
        let strtab = *sections.get(sh_link)?;
        let strtab_data = elf.get(strtab.1..strtab.1.checked_add(strtab.2)?)?;
        let symbols = elf.get(sh_offset..sh_offset.checked_add(sh_size)?)?;
        for symbol in symbols.as_chunks::<24>().0 {
            let name = cstr_at(
                strtab_data,
                usize::try_from(u32::from_le_bytes(symbol[0..4].try_into().ok()?)).ok()?,
            )
            .and_then(|bytes| std::str::from_utf8(bytes).ok());
            let Some(name) = name else { continue };
            match symbol[4] & 0xf {
                STT_FUNC => {
                    names.insert(name.to_string());
                }
                STT_OBJECT if name.ends_with(".kd") => {
                    names.insert(normalize_kernel_name(name).to_string());
                }
                _ => {}
            }
        }
    }
    Some(names.into_iter().collect())
}

// ---------------------------------------------------------------------------
// The kernel registry (module handles + lazy function resolution)
// ---------------------------------------------------------------------------

/// One loaded code object: the `hipModule_t` handle.
#[derive(Clone, Copy, Debug)]
struct LoadedModule {
    handle: ffi::hipModule_t,
}

/// The loaded GPU code objects and the mangled-name → module registry.
/// Function handles are resolved lazily through `hipModuleGetFunction`
/// and cached, so a 2,000-kernel registry costs one hash lookup per
/// name at startup instead of 2,000 FFI calls.
///
/// Safety: `kernels` and `modules` are immutable after construction;
/// `cache` is behind a `Mutex`. The raw pointers themselves are only
/// read, never dereferenced through this type (the FFI calls take them
/// by value), so `Send`/`Sync` are sound.
#[derive(Debug)]
pub struct KernelRegistry {
    modules: Vec<LoadedModule>,
    kernels: BTreeMap<String, usize>,
    cache: Mutex<HashMap<String, ffi::hipFunction_t>>,
}

unsafe impl Send for KernelRegistry {}
unsafe impl Sync for KernelRegistry {}

impl KernelRegistry {
    /// Builds the registry from the bundle container in `data`, loading
    /// every amdgcn payload through `load_module`. Loading is
    /// all-or-nothing: the first payload the loader rejects fails the
    /// whole build with the reason (the serve path logs it and continues
    /// without a registry).
    pub fn with_loader(
        data: &[u8],
        load_module: impl Fn(&[u8]) -> Option<ffi::hipModule_t>,
    ) -> Result<Self, String> {
        let mut modules = Vec::new();
        let mut kernels: BTreeMap<String, usize> = BTreeMap::new();
        for bundle in find_bundles(data) {
            for entry in &bundle.entries {
                if !is_amdgcn_triple(&entry.triple) {
                    continue;
                }
                let payload = bundle
                    .entry_payload(data, entry)
                    .ok_or_else(|| format!("{}: entry points outside the file", entry.triple))?;
                let Some(handle) = load_module(payload) else {
                    return Err(format!("hipModuleLoadData failed for {}", entry.triple));
                };
                let names = elf_kernel_names(payload).unwrap_or_default();
                let index = modules.len();
                for name in &names {
                    kernels.entry(name.clone()).or_insert(index);
                }
                modules.push(LoadedModule { handle });
            }
        }
        Ok(Self {
            modules,
            kernels,
            cache: Mutex::new(HashMap::new()),
        })
    }

    /// Loads the container in `data` through the live HIP runtime
    /// (`hipModuleLoadData` per amdgcn payload).
    pub fn load(runtime: &hip::HipRuntime, data: &[u8]) -> Result<Self, String> {
        Self::with_loader(data, |payload| {
            let mut module = core::ptr::null_mut();
            let error =
                unsafe { (runtime.hipModuleLoadData)(&mut module, payload.as_ptr().cast()) };
            if error == 0 && !module.is_null() {
                Some(module)
            } else {
                None
            }
        })
    }

    /// The number of loaded code-object modules.
    pub fn module_count(&self) -> usize {
        self.modules.len()
    }

    /// The number of distinct kernel names across all modules.
    pub fn kernel_count(&self) -> usize {
        self.kernels.len()
    }

    /// The module index providing `name` (the `.kd` suffix is stripped,
    /// so the mangled name and its `.kd` alias resolve identically).
    pub fn lookup(&self, name: &str) -> Option<usize> {
        self.kernels.get(normalize_kernel_name(name)).copied()
    }

    /// The `hipFunction_t` for `name`, resolved through
    /// `hipModuleGetFunction` on first use and cached afterwards.
    pub fn function(&self, runtime: &hip::HipRuntime, name: &str) -> Option<ffi::hipFunction_t> {
        let name = normalize_kernel_name(name);
        if let Some(cached) = self.cache.lock().ok()?.get(name).copied() {
            return Some(cached);
        }
        let handle = self.modules.get(self.lookup(name)?)?.handle;
        let c_name = CString::new(name).ok()?;
        let mut func = core::ptr::null_mut();
        let error = unsafe { (runtime.hipModuleGetFunction)(&mut func, handle, c_name.as_ptr()) };
        if error != 0 || func.is_null() {
            return None;
        }
        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(name.to_string(), func);
        }
        Some(func)
    }
}

// ---------------------------------------------------------------------------
// Path resolution and the process-wide registry
// ---------------------------------------------------------------------------

/// The fatbin file the loader reads: `HALOGEN_FATBIN` at runtime, else
/// the compile-time default baked by the Nix build (the same pattern as
/// the `--ck` default).
pub fn fatbin_path() -> Option<PathBuf> {
    env::var_os("HALOGEN_FATBIN")
        .map(PathBuf::from)
        .or_else(|| option_env!("HALOGEN_FATBIN").map(PathBuf::from))
}

/// The process-wide registry: the first call loads the fatbin through
/// the live HIP runtime and remembers the outcome (success or the
/// failure reason); later calls return the same result without touching
/// the filesystem or the runtime again.
pub fn init() -> Result<&'static KernelRegistry, &'static str> {
    static INIT: OnceLock<Result<KernelRegistry, String>> = OnceLock::new();
    match INIT.get_or_init(try_init) {
        Ok(registry) => Ok(registry),
        Err(reason) => Err(reason.as_str()),
    }
}

fn try_init() -> Result<KernelRegistry, String> {
    let runtime = hip::runtime().ok_or_else(|| "hip runtime not loaded".to_string())?;
    let path = fatbin_path().ok_or_else(|| "no fatbin path (set HALOGEN_FATBIN)".to_string())?;
    let data = fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    KernelRegistry::load(runtime, &data)
}

// ---------------------------------------------------------------------------
// Tests (device-free: parsing and registry logic need no GPU)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal little-endian ELF64 object exporting `symbols`
    /// (`(name, st_info)` pairs) through `.symtab`/`.strtab`.
    fn synthetic_elf(symbols: &[(&str, u8)]) -> Vec<u8> {
        let mut strtab = Vec::from(b"\x00".as_slice());
        let mut symtab = Vec::new();
        for (name, st_type) in symbols {
            let name_off = strtab.len() as u32;
            strtab.extend_from_slice(name.as_bytes());
            strtab.push(0);
            symtab.extend_from_slice(&name_off.to_le_bytes()); // st_name
            symtab.push((1u8 << 4) | *st_type); // st_info (STB_GLOBAL | type)
            symtab.push(0); // st_other
            symtab.extend_from_slice(&1u16.to_le_bytes()); // st_shndx
            symtab.extend_from_slice(&0u64.to_le_bytes()); // st_value
            symtab.extend_from_slice(&0u64.to_le_bytes()); // st_size
        }
        let symtab_off = 64u64 + 3 * 64;
        let strtab_off = symtab_off + symtab.len() as u64;
        let mut elf = Vec::from(b"\x7fELF".as_slice());
        elf.extend_from_slice(&[2, 1, 1, 0]); // class64, little-endian, version, osabi
        elf.extend_from_slice(&[0u8; 8]); // abiversion + padding (completes e_ident)
        elf.extend_from_slice(&3u16.to_le_bytes()); // e_type ET_DYN
        elf.extend_from_slice(&0xe000u16.to_le_bytes()); // e_machine AMDGCN
        elf.extend_from_slice(&1u32.to_le_bytes()); // e_version
        elf.extend_from_slice(&0u64.to_le_bytes()); // e_entry
        elf.extend_from_slice(&0u64.to_le_bytes()); // e_phoff
        elf.extend_from_slice(&64u64.to_le_bytes()); // e_shoff
        elf.extend_from_slice(&0u32.to_le_bytes()); // e_flags
        elf.extend_from_slice(&64u16.to_le_bytes()); // e_ehsize
        elf.extend_from_slice(&56u16.to_le_bytes()); // e_phentsize
        elf.extend_from_slice(&0u16.to_le_bytes()); // e_phnum
        elf.extend_from_slice(&64u16.to_le_bytes()); // e_shentsize
        elf.extend_from_slice(&3u16.to_le_bytes()); // e_shnum
        elf.extend_from_slice(&2u16.to_le_bytes()); // e_shstrndx
        assert_eq!(elf.len(), 64);
        elf.extend_from_slice(&[0u8; 64]); // shdr[0]: SHT_NULL
        section_header(&mut elf, 0, SHT_SYMTAB, symtab_off, symtab.len() as u64, 2);
        section_header(&mut elf, 0, 3, strtab_off, strtab.len() as u64, 0);
        elf.extend_from_slice(&symtab);
        elf.extend_from_slice(&strtab);
        elf
    }

    fn section_header(
        elf: &mut Vec<u8>,
        sh_name: u32,
        sh_type: u32,
        sh_offset: u64,
        sh_size: u64,
        sh_link: u32,
    ) {
        elf.extend_from_slice(&sh_name.to_le_bytes());
        elf.extend_from_slice(&sh_type.to_le_bytes());
        elf.extend_from_slice(&0u64.to_le_bytes()); // sh_flags
        elf.extend_from_slice(&0u64.to_le_bytes()); // sh_addr
        elf.extend_from_slice(&sh_offset.to_le_bytes());
        elf.extend_from_slice(&sh_size.to_le_bytes());
        elf.extend_from_slice(&sh_link.to_le_bytes());
        elf.extend_from_slice(&0u32.to_le_bytes()); // sh_info
        elf.extend_from_slice(&8u64.to_le_bytes()); // sh_addralign
        elf.extend_from_slice(&24u64.to_le_bytes()); // sh_entsize
    }

    /// Serialises a bundle from `(triple, payload)` items, with or
    /// without the documented per-entry NUL terminator.
    fn build_bundle(items: &[(&str, &[u8])], nul_terminated: bool) -> Vec<u8> {
        let table_len = items.iter().fold(MAGIC.len() + 8, |acc, (triple, _)| {
            acc + 24 + triple.len() + usize::from(nul_terminated)
        });
        let mut bundle = Vec::from(MAGIC.as_slice());
        bundle.extend_from_slice(&(items.len() as u64).to_le_bytes());
        let mut payload_off = table_len;
        for (triple, payload) in items {
            bundle.extend_from_slice(&(payload_off as u64).to_le_bytes());
            bundle.extend_from_slice(&(payload.len() as u64).to_le_bytes());
            bundle.extend_from_slice(&(triple.len() as u64).to_le_bytes());
            bundle.extend_from_slice(triple.as_bytes());
            if nul_terminated {
                bundle.push(0);
            }
            payload_off += payload.len();
        }
        for (_, payload) in items {
            bundle.extend_from_slice(payload);
        }
        bundle
    }

    const HOST_TRIPLE: &str = "host-x86_64-unknown-linux-gnu";
    const DEVICE_TRIPLE: &str = "hipv4-amdgcn-amd-amdhsa--gfx1151";
    const KERNEL: &str = "_ZN7halogen6k_attnEPKtS2_S2_lS2_PtPKiii";

    #[test]
    fn synthetic_bundle_round_trips_through_the_nul_terminated_variant() {
        let elf = synthetic_elf(&[(KERNEL, STT_FUNC), (&format!("{KERNEL}.kd"), STT_OBJECT)]);
        let bundle_bytes = build_bundle(
            &[
                (HOST_TRIPLE, b"host-code".as_slice()),
                (DEVICE_TRIPLE, &elf),
            ],
            true,
        );
        let bundle = parse_bundle_at(&bundle_bytes, 0).expect("the fixture must parse");
        assert_eq!(
            bundle.entries,
            vec![
                BundleEntry {
                    offset: (32 + 24 + HOST_TRIPLE.len() + 1 + 24 + DEVICE_TRIPLE.len() + 1) as u64,
                    size: 9,
                    triple: HOST_TRIPLE.to_string(),
                },
                BundleEntry {
                    offset: (32 + 24 + HOST_TRIPLE.len() + 1 + 24 + DEVICE_TRIPLE.len() + 1 + 9)
                        as u64,
                    size: elf.len() as u64,
                    triple: DEVICE_TRIPLE.to_string(),
                },
            ]
        );
        let payloads = amdgcn_payloads(&bundle_bytes);
        assert_eq!(payloads.len(), 1);
        assert_eq!(payloads[0].0, DEVICE_TRIPLE);
        assert_eq!(payloads[0].1, elf.as_slice());
    }

    #[test]
    fn the_real_no_nul_layout_parses_via_the_fallback_variant() {
        // The reference's bundles have no NUL after the triple; the
        // documented variant misreads entry 1 (its offset shifts by one
        // byte and fails validation), so the fallback must produce the
        // same entries.
        let elf = synthetic_elf(&[(KERNEL, STT_FUNC)]);
        let bundle_bytes = build_bundle(
            &[(HOST_TRIPLE, b"".as_slice()), (DEVICE_TRIPLE, &elf)],
            false,
        );
        let bundle = parse_bundle_at(&bundle_bytes, 0).expect("the no-NUL fixture must parse");
        assert_eq!(bundle.entries[1].triple, DEVICE_TRIPLE);
        assert_eq!(bundle.entries[1].size, elf.len() as u64);
        assert_eq!(
            bundle.entry_payload(&bundle_bytes, &bundle.entries[1]),
            Some(elf.as_slice())
        );
    }

    #[test]
    fn a_multi_bundle_blob_yields_every_bundle() {
        let first = build_bundle(
            &[(HOST_TRIPLE, b"a".as_slice()), (DEVICE_TRIPLE, b"elf-a")],
            true,
        );
        let second = build_bundle(
            &[(HOST_TRIPLE, b"b".as_slice()), (DEVICE_TRIPLE, b"elf-b")],
            false,
        );
        let mut blob = first.clone();
        blob.extend_from_slice(&second);
        let bundles = find_bundles(&blob);
        assert_eq!(bundles.len(), 2);
        assert_eq!(bundles[0].start, 0);
        assert_eq!(bundles[1].start, first.len());
        let payloads = amdgcn_payloads(&blob);
        assert_eq!(payloads.len(), 2);
        assert_eq!(payloads[0].1, b"elf-a");
        assert_eq!(payloads[1].1, b"elf-b");
    }

    #[test]
    fn elf_kernel_names_collect_funcs_and_kd_objects_and_deduplicate() {
        let elf = synthetic_elf(&[
            (KERNEL, STT_FUNC),
            (&format!("{KERNEL}.kd"), STT_OBJECT),
            ("k_other", STT_FUNC),
            ("k_other.kd", STT_OBJECT),
            ("some_data", STT_OBJECT), // plain object: not a kernel
        ]);
        assert_eq!(
            elf_kernel_names(&elf).expect("the fixture must parse"),
            vec![KERNEL.to_string(), "k_other".to_string()]
        );
        assert_eq!(elf_kernel_names(b"not an elf"), None);
    }

    #[test]
    fn registry_resolves_both_the_mangled_name_and_the_kd_alias() {
        let elf = synthetic_elf(&[(KERNEL, STT_FUNC), (&format!("{KERNEL}.kd"), STT_OBJECT)]);
        let bundle_bytes = build_bundle(
            &[(HOST_TRIPLE, b"".as_slice()), (DEVICE_TRIPLE, &elf)],
            false,
        );
        let registry = KernelRegistry::with_loader(&bundle_bytes, |_| Some(core::ptr::null_mut()))
            .expect("the fixture must load");
        assert_eq!(registry.module_count(), 1); // the host entry is skipped
        assert_eq!(registry.kernel_count(), 1);
        assert_eq!(registry.lookup(KERNEL), Some(0));
        assert_eq!(registry.lookup(&format!("{KERNEL}.kd")), Some(0));
        assert_eq!(registry.lookup("k_missing"), None);
    }

    #[test]
    fn registry_loading_is_all_or_nothing() {
        let elf = synthetic_elf(&[(KERNEL, STT_FUNC)]);
        let bundle_bytes = build_bundle(
            &[(HOST_TRIPLE, b"".as_slice()), (DEVICE_TRIPLE, &elf)],
            false,
        );
        let error = KernelRegistry::with_loader(&bundle_bytes, |_| None)
            .expect_err("a rejecting loader must fail the build");
        assert!(
            error.contains("hipModuleLoadData failed for hipv4-amdgcn-amd-amdhsa--gfx1151"),
            "unexpected wording: {error}"
        );
    }

    #[test]
    fn the_extracted_reference_section_parses_when_present() {
        // Cross-check against the real `.hip_fatbin` extraction from the
        // inventory work; skipped when the file is absent (the nix
        // build's source filter excludes .tmp/).
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../.tmp/fatbin-inventory/fatbin.bin"
        );
        let Ok(data) = fs::read(path) else { return };
        assert_eq!(find_bundles(&data).len(), 4);
        let payloads = amdgcn_payloads(&data);
        assert_eq!(payloads.len(), 4);
        assert!(payloads.iter().all(|(triple, _)| triple == DEVICE_TRIPLE));
        let registry = KernelRegistry::with_loader(&data, |_| Some(core::ptr::null_mut()))
            .expect("the real section must load");
        assert_eq!(registry.module_count(), 4);
        // The inventory documents 2,000 `.kd` symbols; the four payloads
        // also export q4moe-namespace kernels, so the registry holds at
        // least the documented set.
        assert!(
            registry.kernel_count() >= 2000,
            "expected >= 2000 kernels, got {}",
            registry.kernel_count()
        );
        assert!(registry
            .lookup("_ZN7halogen10k_dv_quantEPKtPKiS3_iPhli")
            .is_some());
    }
}
