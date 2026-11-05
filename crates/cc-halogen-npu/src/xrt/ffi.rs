//! The hand-written FFI surface for AMD XRT's `libxrt_coreutil.so.2` —
//! the reference `halogen-npu` binary's `DT_NEEDED` NPU runtime (spec
//! [`halogen-npu.md`](../../../../docs/analysis/specs/halogen-npu.md),
//! [npu/xrt-imported-symbols.json](../../../../docs/analysis/evidence/flash_serve/npu/xrt-imported-symbols.json)).
//!
//! The catalogue lists 20 `xrt::` C++ symbols (+2 `_ITM_*` clone-table
//! hooks). C++ symbols are mangled, so every entry point is declared with
//! `#[link_name]` set to the exact mangled name from the catalogue — the
//! same pattern as `cc-flash-serve`'s `hip.rs` mangled `hipblaslt_ext::`
//! helpers. The library is loaded dlopen-style at runtime; when it cannot
//! be loaded (the usual case — see the module docs of [`crate::xrt`]) the
//! port takes the documented no-device path.
//!
//! Signature modelling notes (mirroring the `hip.rs` precedent of declaring
//! the surface for completeness): C++ references are modelled as pointers;
//! `std::string const&` parameters as `*const c_char`; the `xrt::info::device`
//! and `xclBOSyncDirection` enums and `xrt::kernel::state` (the return of
//! `xrt::run::state`) as `c_uint` (the psABI passes non-negative enums in
//! registers). Constructors are declared with the stub's void shape — the
//! real ABI returns the `this` pointer in `rax`, which the port ignores.

use core::ffi::{c_char, c_int, c_uint, c_void};
use std::ffi::CString;

/// The `DT_NEEDED` soname of the reference binary (spec "Artifact identity").
pub const LIBRARY_SONAME: &str = "libxrt_coreutil.so.2";
/// The unversioned dev-name fallback (mirrors the `hip.rs` probe pattern).
pub const LIBRARY_DEV_NAME: &str = "libxrt_coreutil.so";

pub const RTLD_LAZY: c_int = 1;

extern "C" {
    pub fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    pub fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    pub fn dlclose(handle: *mut c_void) -> c_int;
}

/// The 20 `xrt::` symbols of the import catalogue (the JSON's order):
/// `(demangled signature, mangled link name)` pairs. The mangled names are
/// the ones the reference binary imports (cross-checked against
/// [the xrt-stub](../../../../docs/analysis/evidence/halogen-npu/xrt-stub/libxrt_coreutil-stub.c)
/// by `ffi_surface_matches_the_import_catalogue`).
pub const XRT_SYMBOLS: [(&str, &str); 20] = [
    (
        "xrt::device::device(unsigned int)",
        "_ZN3xrt6deviceC1Ej",
    ),
    ("xrt::device::~device()", "_ZN3xrt6deviceD1Ev"),
    (
        "xrt::device::get_info_std(xrt::info::device, xrt::detail::abi const&) const",
        "_ZNK3xrt6device12get_info_stdENS_4info6deviceERKNS_6detail3abiE",
    ),
    (
        "xrt::elf::elf(std::string const&)",
        "_ZN3xrt3elfC1ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE",
    ),
    (
        "xrt::hw_context::hw_context(xrt::device const&, xrt::elf const&)",
        "_ZN3xrt10hw_contextC1ERKNS_6deviceERKNS_3elfE",
    ),
    ("xrt::hw_context::~hw_context()", "_ZN3xrt10hw_contextD1Ev"),
    (
        "xrt::ext::kernel::kernel(xrt::hw_context const&, std::string const&)",
        "_ZN3xrt3ext6kernelC1ERKNS_10hw_contextERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE",
    ),
    ("xrt::kernel::~kernel()", "_ZN3xrt6kernelD2Ev"),
    (
        "xrt::run::run(xrt::kernel const&)",
        "_ZN3xrt3runC1ERKNS_6kernelE",
    ),
    ("xrt::run::~run()", "_ZN3xrt3runD1Ev"),
    (
        "xrt::run::set_arg_at_index(int, xrt::bo const&)",
        "_ZN3xrt3run16set_arg_at_indexEiRKNS_2boE",
    ),
    ("xrt::run::start()", "_ZN3xrt3run5startEv"),
    (
        "xrt::run::wait(std::chrono::duration<long, std::ratio<1l, 1000l>> const&) const",
        "_ZNK3xrt3run4waitERKNSt6chrono8durationIlSt5ratioILl1ELl1000EEEE",
    ),
    ("xrt::run::state() const", "_ZNK3xrt3run5stateEv"),
    (
        "xrt::bo::bo(xrt::device const&, unsigned long, xrt::bo::flags, unsigned int)",
        "_ZN3xrt2boC1ERKNS_6deviceEmNS0_5flagsEj",
    ),
    (
        "xrt::bo::bo(xrt::bo const&, unsigned long, unsigned long)",
        "_ZN3xrt2boC1ERKS0_mm",
    ),
    ("xrt::bo::~bo()", "_ZN3xrt2boD1Ev"),
    ("xrt::bo::map()", "_ZN3xrt2bo3mapEv"),
    (
        "xrt::bo::sync(xclBOSyncDirection, unsigned long, unsigned long)",
        "_ZN3xrt2bo4syncE18xclBOSyncDirectionmm",
    ),
    ("xrt::bo::size() const", "_ZNK3xrt2bo4sizeEv"),
];

/// The 2 `_ITM_*` clone-table hooks of the catalogue (normally libgcc weak
/// symbols; the reference binary imports them and the stub defines them).
/// They carry no API surface, so they are catalogued but not declared.
pub const ITM_SYMBOLS: [&str; 2] = ["_ITM_deregisterTMCloneTable", "_ITM_registerTMCloneTable"];

// --- the 20 `xrt::` entry points (the catalogue's exact mangled names) ----

extern "C" {
    // xrt::device
    /// `xrt::device::device(unsigned int)` — opens the device at `index`.
    #[link_name = "_ZN3xrt6deviceC1Ej"]
    pub fn device_ctor(device: *mut c_void, index: c_uint);
    /// `xrt::device::~device()`.
    #[link_name = "_ZN3xrt6deviceD1Ev"]
    pub fn device_dtor(device: *mut c_void);
    /// `xrt::device::get_info_std(xrt::info::device, xrt::detail::abi const&) const`
    /// — the real XRT returns a `std::string` (sret); the stub declares
    /// `void` and the port never calls it (the reference's `listen` path
    /// uses it only for device-name logging).
    #[link_name = "_ZNK3xrt6device12get_info_stdENS_4info6deviceERKNS_6detail3abiE"]
    pub fn device_get_info_std(device: *const c_void, query: c_uint, abi: *const c_void);

    // xrt::elf
    /// `xrt::elf::elf(std::string const&)` — loads the kernel ELF at `path`.
    #[link_name = "_ZN3xrt3elfC1ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE"]
    pub fn elf_ctor(elf: *mut c_void, path: *const c_char);
    // The catalogue imports no `xrt::elf::~elf()` — the reference never
    // destroys the elf object (see the session port's note).

    // xrt::hw_context
    /// `xrt::hw_context::hw_context(xrt::device const&, xrt::elf const&)`.
    #[link_name = "_ZN3xrt10hw_contextC1ERKNS_6deviceERKNS_3elfE"]
    pub fn hw_context_ctor(context: *mut c_void, device: *const c_void, elf: *const c_void);
    /// `xrt::hw_context::~hw_context()`.
    #[link_name = "_ZN3xrt10hw_contextD1Ev"]
    pub fn hw_context_dtor(context: *mut c_void);

    // xrt::ext::kernel / xrt::kernel
    /// `xrt::ext::kernel::kernel(xrt::hw_context const&, std::string const&)`.
    #[link_name = "_ZN3xrt3ext6kernelC1ERKNS_10hw_contextERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE"]
    pub fn ext_kernel_ctor(kernel: *mut c_void, context: *const c_void, name: *const c_char);
    /// `xrt::kernel::~kernel()` (the base-object destructor variant `D2`).
    #[link_name = "_ZN3xrt6kernelD2Ev"]
    pub fn kernel_dtor(kernel: *mut c_void);

    // xrt::run
    /// `xrt::run::run(xrt::kernel const&)`.
    #[link_name = "_ZN3xrt3runC1ERKNS_6kernelE"]
    pub fn run_ctor(run: *mut c_void, kernel: *const c_void);
    /// `xrt::run::~run()`.
    #[link_name = "_ZN3xrt3runD1Ev"]
    pub fn run_dtor(run: *mut c_void);
    /// `xrt::run::set_arg_at_index(int, xrt::bo const&)`.
    #[link_name = "_ZN3xrt3run16set_arg_at_indexEiRKNS_2boE"]
    pub fn run_set_arg_at_index(run: *mut c_void, index: c_int, bo: *const c_void);
    /// `xrt::run::start()`.
    #[link_name = "_ZN3xrt3run5startEv"]
    pub fn run_start(run: *mut c_void);
    /// `xrt::run::wait(std::chrono::duration<long, milli> const&) const` —
    /// the duration (milliseconds) is passed by const reference.
    #[link_name = "_ZNK3xrt3run4waitERKNSt6chrono8durationIlSt5ratioILl1ELl1000EEEE"]
    pub fn run_wait(run: *const c_void, timeout_ms: *const i64);
    /// `xrt::run::state() const` — returns `xrt::kernel::state` (the stub
    /// declares `void`; the real XRT returns the enum, modelled as `c_uint`).
    #[link_name = "_ZNK3xrt3run5stateEv"]
    pub fn run_state(run: *const c_void) -> c_uint;

    // xrt::bo
    /// `xrt::bo::bo(xrt::device const&, unsigned long, xrt::bo::flags, unsigned int)`.
    #[link_name = "_ZN3xrt2boC1ERKNS_6deviceEmNS0_5flagsEj"]
    pub fn bo_ctor_device(
        bo: *mut c_void,
        device: *const c_void,
        size: u64,
        flags: c_uint,
        bank: c_uint,
    );
    /// `xrt::bo::bo(xrt::bo const&, unsigned long, unsigned long)` — the
    /// offset-sliced sub-buffer constructor (parent, offset, size).
    #[link_name = "_ZN3xrt2boC1ERKS0_mm"]
    pub fn bo_ctor_sub(bo: *mut c_void, parent: *const c_void, offset: u64, size: u64);
    /// `xrt::bo::~bo()`.
    #[link_name = "_ZN3xrt2boD1Ev"]
    pub fn bo_dtor(bo: *mut c_void);
    /// `xrt::bo::map()` — returns the host mapping (the stub declares
    /// `void`; the real XRT returns `void*`, modelled as `*mut c_void`).
    #[link_name = "_ZN3xrt2bo3mapEv"]
    pub fn bo_map(bo: *mut c_void) -> *mut c_void;
    /// `xrt::bo::sync(xclBOSyncDirection, unsigned long, unsigned long)`.
    #[link_name = "_ZN3xrt2bo4syncE18xclBOSyncDirectionmm"]
    pub fn bo_sync(bo: *mut c_void, direction: c_uint, size: u64, offset: u64);
    /// `xrt::bo::size() const`.
    #[link_name = "_ZNK3xrt2bo4sizeEv"]
    pub fn bo_size(bo: *const c_void) -> u64;
}

/// The 20 resolved entry points of [`XRT_SYMBOLS`], loaded dlopen-style
/// (the `cc-flash-serve` `hip.rs` pattern). Any missing symbol yields
/// `None` from [`XrtRuntime::open`] — the port then treats XRT as absent.
#[derive(Clone, Copy)]
pub struct XrtRuntime {
    pub device_ctor: unsafe extern "C" fn(*mut c_void, c_uint),
    pub device_dtor: unsafe extern "C" fn(*mut c_void),
    pub device_get_info_std: unsafe extern "C" fn(*const c_void, c_uint, *const c_void),
    pub elf_ctor: unsafe extern "C" fn(*mut c_void, *const c_char),
    pub hw_context_ctor: unsafe extern "C" fn(*mut c_void, *const c_void, *const c_void),
    pub hw_context_dtor: unsafe extern "C" fn(*mut c_void),
    pub ext_kernel_ctor: unsafe extern "C" fn(*mut c_void, *const c_void, *const c_char),
    pub kernel_dtor: unsafe extern "C" fn(*mut c_void),
    pub run_ctor: unsafe extern "C" fn(*mut c_void, *const c_void),
    pub run_dtor: unsafe extern "C" fn(*mut c_void),
    pub run_set_arg_at_index: unsafe extern "C" fn(*mut c_void, c_int, *const c_void),
    pub run_start: unsafe extern "C" fn(*mut c_void),
    pub run_wait: unsafe extern "C" fn(*const c_void, *const i64),
    pub run_state: unsafe extern "C" fn(*const c_void) -> c_uint,
    pub bo_ctor_device: unsafe extern "C" fn(*mut c_void, *const c_void, u64, c_uint, c_uint),
    pub bo_ctor_sub: unsafe extern "C" fn(*mut c_void, *const c_void, u64, u64),
    pub bo_dtor: unsafe extern "C" fn(*mut c_void),
    pub bo_map: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    pub bo_sync: unsafe extern "C" fn(*mut c_void, c_uint, u64, u64),
    pub bo_size: unsafe extern "C" fn(*const c_void) -> u64,
}

/// Resolves `symbol` from an open library handle as the function-pointer
/// type `$ty`, or returns `None` from the enclosing `open` when the symbol
/// is missing (the port then treats XRT as absent).
macro_rules! bind {
    ($handle:expr, $symbol:expr, $ty:ty) => {{
        let name = CString::new($symbol).expect("symbol name without interior nul");
        let ptr = unsafe { dlsym($handle, name.as_ptr()) };
        if ptr.is_null() {
            return None;
        }
        unsafe { core::mem::transmute::<*mut c_void, $ty>(ptr) }
    }};
}

impl XrtRuntime {
    /// Opens `path` with `RTLD_LAZY` and resolves every symbol of
    /// [`XRT_SYMBOLS`]; `None` when the library cannot be opened or any
    /// symbol is missing.
    pub fn open(path: &str) -> Option<Self> {
        let name = CString::new(path).ok()?;
        let handle = unsafe { dlopen(name.as_ptr(), RTLD_LAZY) };
        if handle.is_null() {
            return None;
        }
        Some(Self {
            device_ctor: bind!(handle, "_ZN3xrt6deviceC1Ej", unsafe extern "C" fn(*mut c_void, c_uint)),
            device_dtor: bind!(handle, "_ZN3xrt6deviceD1Ev", unsafe extern "C" fn(*mut c_void)),
            device_get_info_std: bind!(handle, "_ZNK3xrt6device12get_info_stdENS_4info6deviceERKNS_6detail3abiE", unsafe extern "C" fn(*const c_void, c_uint, *const c_void)),
            elf_ctor: bind!(handle, "_ZN3xrt3elfC1ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE", unsafe extern "C" fn(*mut c_void, *const c_char)),
            hw_context_ctor: bind!(handle, "_ZN3xrt10hw_contextC1ERKNS_6deviceERKNS_3elfE", unsafe extern "C" fn(*mut c_void, *const c_void, *const c_void)),
            hw_context_dtor: bind!(handle, "_ZN3xrt10hw_contextD1Ev", unsafe extern "C" fn(*mut c_void)),
            ext_kernel_ctor: bind!(handle, "_ZN3xrt3ext6kernelC1ERKNS_10hw_contextERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE", unsafe extern "C" fn(*mut c_void, *const c_void, *const c_char)),
            kernel_dtor: bind!(handle, "_ZN3xrt6kernelD2Ev", unsafe extern "C" fn(*mut c_void)),
            run_ctor: bind!(handle, "_ZN3xrt3runC1ERKNS_6kernelE", unsafe extern "C" fn(*mut c_void, *const c_void)),
            run_dtor: bind!(handle, "_ZN3xrt3runD1Ev", unsafe extern "C" fn(*mut c_void)),
            run_set_arg_at_index: bind!(handle, "_ZN3xrt3run16set_arg_at_indexEiRKNS_2boE", unsafe extern "C" fn(*mut c_void, c_int, *const c_void)),
            run_start: bind!(handle, "_ZN3xrt3run5startEv", unsafe extern "C" fn(*mut c_void)),
            run_wait: bind!(handle, "_ZNK3xrt3run4waitERKNSt6chrono8durationIlSt5ratioILl1ELl1000EEEE", unsafe extern "C" fn(*const c_void, *const i64)),
            run_state: bind!(handle, "_ZNK3xrt3run5stateEv", unsafe extern "C" fn(*const c_void) -> c_uint),
            bo_ctor_device: bind!(handle, "_ZN3xrt2boC1ERKNS_6deviceEmNS0_5flagsEj", unsafe extern "C" fn(*mut c_void, *const c_void, u64, c_uint, c_uint)),
            bo_ctor_sub: bind!(handle, "_ZN3xrt2boC1ERKS0_mm", unsafe extern "C" fn(*mut c_void, *const c_void, u64, u64)),
            bo_dtor: bind!(handle, "_ZN3xrt2boD1Ev", unsafe extern "C" fn(*mut c_void)),
            bo_map: bind!(handle, "_ZN3xrt2bo3mapEv", unsafe extern "C" fn(*mut c_void) -> *mut c_void),
            bo_sync: bind!(handle, "_ZN3xrt2bo4syncE18xclBOSyncDirectionmm", unsafe extern "C" fn(*mut c_void, c_uint, u64, u64)),
            bo_size: bind!(handle, "_ZNK3xrt2bo4sizeEv", unsafe extern "C" fn(*const c_void) -> u64),
        })
    }

    /// The runtime availability probe: opens the reference's `DT_NEEDED`
    /// soname first, then the unversioned dev name (the `hip.rs` pattern).
    pub fn load() -> Option<Self> {
        [LIBRARY_SONAME, LIBRARY_DEV_NAME]
            .into_iter()
            .find_map(Self::open)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Collects every `#[link_name]` attribute string from the module
    /// source (worded to avoid the attribute pattern it scans for).
    fn declared_link_names(source: &str) -> Vec<String> {
        const PREFIX: &str = concat!("#[link_name", " = \"");
        let mut names = Vec::new();
        let mut rest = source;
        while let Some((_, tail)) = rest.split_once(PREFIX) {
            let Some((name, tail)) = tail.split_once('"') else {
                break;
            };
            names.push(name.to_string());
            rest = tail;
        }
        names
    }

    #[test]
    fn ffi_surface_matches_the_import_catalogue() {
        assert_eq!(XRT_SYMBOLS.len(), 20);
        assert_eq!(ITM_SYMBOLS.len(), 2);
        // Every catalogued mangled name must appear exactly once as a
        // `#[link_name]` declaration, and no undeclared names may exist
        // (the non-vacuous pattern from `cc-flash-serve`'s hip.rs tests).
        let declared = declared_link_names(include_str!("ffi.rs"));
        assert_eq!(
            declared.len(),
            XRT_SYMBOLS.len(),
            "the FFI surface declares a different number of symbols than the catalogue"
        );
        for (_, mangled) in XRT_SYMBOLS {
            let count = declared
                .iter()
                .filter(|name| name.as_str() == mangled)
                .count();
            assert_eq!(
                count, 1,
                "catalogued symbol {mangled} must have exactly one FFI declaration"
            );
        }
        for (demangled, _) in XRT_SYMBOLS {
            assert!(
                include_str!("ffi.rs").contains(demangled),
                "catalogued signature {demangled} is not documented in the FFI surface"
            );
        }
    }

    #[test]
    fn catalogue_matches_the_evidence() {
        // Cross-check against the evidence JSON and the stub C source. The
        // nix build's source filter excludes docs/, so the check is
        // best-effort there (same policy as the hip.rs tests).
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/analysis/evidence");
        let json_path = format!("{root}/flash_serve/npu/xrt-imported-symbols.json");
        if let Ok(json) = std::fs::read_to_string(json_path) {
            for (demangled, _) in XRT_SYMBOLS {
                assert!(
                    json.contains(&format!("\"{demangled}\"")),
                    "catalogued signature {demangled} is not in the evidence JSON"
                );
            }
            for symbol in ITM_SYMBOLS {
                assert!(
                    json.contains(&format!("\"{symbol}\"")),
                    "catalogued _ITM symbol {symbol} is not in the evidence JSON"
                );
            }
        }
        let stub_path = format!("{root}/halogen-npu/xrt-stub/libxrt_coreutil-stub.c");
        if let Ok(stub) = std::fs::read_to_string(stub_path) {
            for (_, mangled) in XRT_SYMBOLS {
                assert!(
                    stub.contains(mangled),
                    "mangled symbol {mangled} is not defined by the xrt stub"
                );
            }
            for symbol in ITM_SYMBOLS {
                assert!(
                    stub.contains(symbol),
                    "_ITM symbol {symbol} is not defined by the xrt stub"
                );
            }
        }
    }
}
