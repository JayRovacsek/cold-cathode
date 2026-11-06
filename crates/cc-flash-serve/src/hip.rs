//! HIP backend integration (plan §7 Phase 7 — spec
//! [`flash_serve-mtp-kv-npu.md`](../../../docs/analysis/specs/flash_serve-mtp-kv-npu.md)
//! §3.1).
//!
//! The reference binary `flash_serve` links the AMD HIP stack
//! (`libamdhip64.so.7` + `libhipblaslt.so.1`, RUNPATH `clr-7.2.4` /
//! `hipblaslt-7.2.4` — evidence
//! [npu/hip-imported-symbols.json](../../../docs/analysis/evidence/flash_serve/npu/hip-imported-symbols.json)).
//! This module hand-writes the `extern "C"` FFI surface for the imported
//! symbols (no hip-sys crate: the port keeps Cargo.lock and the package
//! cargoHash untouched), loads `libamdhip64.so.7` dlopen-style at runtime,
//! and ports the two device-side functions the plan covers:
//!
//! * `FUN_01f700f0` — the device lifecycle: enumerate devices, read the
//!   PCI bus id, apply the `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES`
//!   / `CUDA_VISIBLE_DEVICES` / `GPU_DEVICE_ORDINAL` precedence, resolve
//!   `HALOGEN_GPU_DEVICE` (`<n>` | `first` | `auto`), bind with
//!   `hipSetDevice` and print the device banner (the reference's
//!   setenv/unsetenv/execv re-exec rule is documented at [`bind_device`]).
//! * `FUN_01f82130` — the `dmalloc:` device-allocation accounting feeding
//!   the kvcache pool budget math ([`DeviceAllocAccount`]).
//!
//! When `libamdhip64.so.7` cannot be loaded (the usual case on a
//! CPU-only host) [`active`] reports [`Backend::Cpu`] and the CPU forward
//! path stays the default. The reference has no CPU fallback — without
//! HIP it exits with the `ROCM_NO_GPU` message (lib.rs), which the serve
//! path still enforces before this module is consulted; the reference
//! behaviour when HIP is absent is therefore the `ROCM_NO_GPU` exit, and
//! the port's `Backend::Cpu` exists so the binary remains usable and
//! testable off-GPU.

use std::env;
use std::ffi::CStr;
use std::ffi::CString;
use std::ffi::{c_char, c_float, c_int, c_uint, c_void};
use std::process;
use std::sync::{Mutex, OnceLock};

use crate::kvcache;

/// The hand-written FFI surface of the reference's device imports: the
/// 36 HIP runtime symbols, the 13 `hipblasLt*` functions and the 3
/// mangled `hipblaslt_ext::` helpers of
/// [npu/hip-imported-symbols.json](../../../docs/analysis/evidence/flash_serve/npu/hip-imported-symbols.json),
/// plus `__hipPushCallConfiguration` (named by spec §3.1 as the
/// `hipLaunchKernel` fast path; it lives in the same library but is not
/// part of the 36-symbol runtime list) and the three `dl*` entry points
/// used to load the library (declared directly — glibc ≥ 2.34 exports
/// them from libc, so no `libc` crate dependency is needed).
pub mod ffi {
    #![allow(non_snake_case, non_camel_case_types)]

    use core::ffi::{c_char, c_float, c_int, c_uint, c_void};

    // --- scalar aliases -----------------------------------------------------

    pub type hipError_t = c_int;
    pub type hipStream_t = *mut c_void;
    pub type hipEvent_t = *mut c_void;
    pub type hipMemcpyKind = c_uint;

    /// The `dim3` launch dimensions (the `__hipPushCallConfiguration` and
    /// `hipLaunchKernel` arguments).
    #[repr(C)]
    #[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
    pub struct dim3 {
        pub x: c_uint,
        pub y: c_uint,
        pub z: c_uint,
    }

    /// The R0600 `hipDeviceProp_t` layout: `name` (the device name the
    /// reference compares against `gfx1151`) followed by the remaining
    /// 1216 bytes of the struct (size 0x5c0 per the decompile; the port
    /// only reads `name`).
    #[repr(C, align(8))]
    #[derive(Clone, Copy)]
    pub struct hipDevicePropStructR0600 {
        pub name: [c_char; 256],
        pub _rest: [u8; 1216],
    }

    // --- hipBLASLt handles and descriptors ----------------------------------

    #[repr(C)]
    pub struct _hipblasLtHandle_t {
        pub _opaque: [u8; 0],
    }
    #[repr(C)]
    pub struct _hipblasLtMatmulDesc_t {
        pub _opaque: [u8; 0],
    }
    #[repr(C)]
    pub struct _hipblasLtMatrixLayout_t {
        pub _opaque: [u8; 0],
    }
    #[repr(C)]
    pub struct _hipblasLtMatmulPreference_t {
        pub _opaque: [u8; 0],
    }
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct _hipblasLtMatmulAlgo_t {
        pub data: [u64; 8],
    }
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct _hipblasLtMatmulHeuristicResult_t {
        pub algo: _hipblasLtMatmulAlgo_t,
        pub workspace: usize,
        pub state: c_uint,
        pub wavesCount: c_float,
        pub reserved: c_int,
    }

    pub type hipblasLtHandle_t = *mut _hipblasLtHandle_t;
    pub type hipblasLtMatmulDesc_t = *mut _hipblasLtMatmulDesc_t;
    pub type hipblasLtMatrixLayout_t = *mut _hipblasLtMatrixLayout_t;
    pub type hipblasLtMatmulPreference_t = *mut _hipblasLtMatmulPreference_t;
    pub type hipblasLtMatmulAlgo_t = _hipblasLtMatmulAlgo_t;
    pub type hipblasLtMatmulHeuristicResult_t = _hipblasLtMatmulHeuristicResult_t;

    // The hipBLASLt enums (C++ enums with non-negative enumerators; the
    // psABI passes them as `unsigned int`).
    pub type hipblaslt_status = c_uint;
    pub type hipblasLtComputeType_t = c_uint;
    pub type hipblasLtScaleType_t = c_uint;
    pub type hipblasLtMatmulDescAttributes_t = c_uint;
    pub type hipblasLtMatmulPreferenceAttributes_t = c_uint;
    pub type hipDataType = c_uint;

    // --- the HIP runtime declarations (the 36 imported symbols) -------------

    extern "C" {
        pub fn hipDeviceGetPCIBusId(pciBusId: *mut c_char, len: c_int, device: c_int)
            -> hipError_t;
        pub fn hipDeviceGetStreamPriorityRange(
            leastPriority: *mut c_int,
            greatestPriority: *mut c_int,
            device: c_int,
        ) -> hipError_t;
        pub fn hipDeviceSynchronize() -> hipError_t;
        pub fn hipEventCreate(event: *mut hipEvent_t) -> hipError_t;
        pub fn hipEventCreateWithFlags(event: *mut hipEvent_t, flags: c_uint) -> hipError_t;
        pub fn hipEventDestroy(event: hipEvent_t) -> hipError_t;
        pub fn hipEventElapsedTime(
            ms: *mut c_float,
            start: hipEvent_t,
            stop: hipEvent_t,
        ) -> hipError_t;
        pub fn hipEventRecord(event: hipEvent_t, stream: hipStream_t) -> hipError_t;
        pub fn hipEventSynchronize(event: hipEvent_t) -> hipError_t;
        pub fn hipFuncGetAttributes(attr: *mut c_void, func: *const c_void) -> hipError_t;
        pub fn hipGetDevice(device: *mut c_int) -> hipError_t;
        pub fn hipGetDeviceCount(count: *mut c_int) -> hipError_t;
        pub fn hipGetDevicePropertiesR0600(
            prop: *mut hipDevicePropStructR0600,
            device: c_int,
        ) -> hipError_t;
        pub fn hipGetErrorString(error: hipError_t) -> *const c_char;
        pub fn hipGetLastError() -> hipError_t;
        pub fn hipHostFree(ptr: *mut c_void) -> hipError_t;
        pub fn hipHostGetDevicePointer(
            devicePointer: *mut *mut c_void,
            hostPointer: *mut c_void,
            flags: c_uint,
        ) -> hipError_t;
        pub fn hipHostMalloc(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> hipError_t;
        pub fn hipHostRegister(ptr: *mut c_void, size: usize, flags: c_uint) -> hipError_t;
        pub fn hipHostUnregister(ptr: *mut c_void) -> hipError_t;
        pub fn hipLaunchKernel(
            functionToLaunch: *const c_void,
            gridDim: dim3,
            blockDim: dim3,
            args: *mut *mut c_void,
            sharedMemBytes: usize,
            stream: hipStream_t,
        ) -> hipError_t;
        pub fn hipMalloc(devPtr: *mut *mut c_void, size: usize) -> hipError_t;
        pub fn hipMemcpy(
            dst: *mut c_void,
            src: *const c_void,
            size: usize,
            kind: hipMemcpyKind,
        ) -> hipError_t;
        pub fn hipMemcpy2D(
            dst: *mut c_void,
            dpitch: usize,
            src: *const c_void,
            spitch: usize,
            width: usize,
            height: usize,
            kind: hipMemcpyKind,
        ) -> hipError_t;
        pub fn hipMemcpyAsync(
            dst: *mut c_void,
            src: *const c_void,
            size: usize,
            kind: hipMemcpyKind,
            stream: hipStream_t,
        ) -> hipError_t;
        pub fn hipMemset(devPtr: *mut c_void, value: c_int, count: usize) -> hipError_t;
        pub fn hipMemsetAsync(
            devPtr: *mut c_void,
            value: c_int,
            count: usize,
            stream: hipStream_t,
        ) -> hipError_t;
        pub fn hipMemsetD32Async(
            devPtr: *mut c_void,
            value: c_uint,
            count: usize,
            stream: hipStream_t,
        ) -> hipError_t;
        pub fn hipOccupancyMaxActiveBlocksPerMultiprocessor(
            numBlocks: *mut c_int,
            func: *const c_void,
            blockSize: c_int,
            dynamicSMemSize: usize,
        ) -> hipError_t;
        pub fn hipRuntimeGetVersion(runtimeVersion: *mut c_int) -> hipError_t;
        pub fn hipSetDevice(device: c_int) -> hipError_t;
        pub fn hipStreamCreateWithFlags(stream: *mut hipStream_t, flags: c_uint) -> hipError_t;
        pub fn hipStreamCreateWithPriority(
            stream: *mut hipStream_t,
            flags: c_uint,
            priority: c_int,
        ) -> hipError_t;
        pub fn hipStreamDestroy(stream: hipStream_t) -> hipError_t;
        pub fn hipStreamSynchronize(stream: hipStream_t) -> hipError_t;
        pub fn hipStreamWaitEvent(
            stream: hipStream_t,
            event: hipEvent_t,
            flags: c_uint,
        ) -> hipError_t;
    }

    // --- the hipBLASLt declarations (the 13 imported `hipblasLt*`) ----------

    extern "C" {
        pub fn hipblasLtCreate(handle: *mut hipblasLtHandle_t) -> hipblaslt_status;
        pub fn hipblasLtDestroy(handle: hipblasLtHandle_t) -> hipblaslt_status;
        pub fn hipblasLtGetVersion() -> usize;
        pub fn hipblasLtMatmul(
            handle: hipblasLtHandle_t,
            computeDesc: hipblasLtMatmulDesc_t,
            alpha: *const c_void,
            a: *const c_void,
            adesc: hipblasLtMatrixLayout_t,
            b: *const c_void,
            bdesc: hipblasLtMatrixLayout_t,
            beta: *const c_void,
            c: *const c_void,
            cdesc: hipblasLtMatrixLayout_t,
            d: *mut c_void,
            ddesc: hipblasLtMatrixLayout_t,
            algo: *const _hipblasLtMatmulAlgo_t,
            workspace: *mut c_void,
            workspaceSizeInBytes: usize,
            stream: hipStream_t,
        ) -> hipblaslt_status;
        pub fn hipblasLtMatmulAlgoGetHeuristic(
            handle: hipblasLtHandle_t,
            operationDesc: hipblasLtMatmulDesc_t,
            adesc: hipblasLtMatrixLayout_t,
            bdesc: hipblasLtMatrixLayout_t,
            cdesc: hipblasLtMatrixLayout_t,
            ddesc: hipblasLtMatrixLayout_t,
            preference: hipblasLtMatmulPreference_t,
            request: c_int,
            results: *mut _hipblasLtMatmulHeuristicResult_t,
            returnedResults: *mut c_int,
        ) -> hipblaslt_status;
        pub fn hipblasLtMatmulDescCreate(
            desc: *mut hipblasLtMatmulDesc_t,
            computeType: hipblasLtComputeType_t,
            scaleType: hipblasLtScaleType_t,
        ) -> hipblaslt_status;
        pub fn hipblasLtMatmulDescDestroy(desc: hipblasLtMatmulDesc_t) -> hipblaslt_status;
        pub fn hipblasLtMatmulDescSetAttribute(
            desc: hipblasLtMatmulDesc_t,
            attr: hipblasLtMatmulDescAttributes_t,
            buf: *const c_void,
            size: usize,
        ) -> hipblaslt_status;
        pub fn hipblasLtMatmulPreferenceCreate(
            pref: *mut hipblasLtMatmulPreference_t,
        ) -> hipblaslt_status;
        pub fn hipblasLtMatmulPreferenceDestroy(
            pref: hipblasLtMatmulPreference_t,
        ) -> hipblaslt_status;
        pub fn hipblasLtMatmulPreferenceSetAttribute(
            pref: hipblasLtMatmulPreference_t,
            attr: hipblasLtMatmulPreferenceAttributes_t,
            buf: *const c_void,
            size: usize,
        ) -> hipblaslt_status;
        pub fn hipblasLtMatrixLayoutCreate(
            layout: *mut hipblasLtMatrixLayout_t,
            datatype: hipDataType,
            rows: u64,
            cols: u64,
            ld: i64,
        ) -> hipblaslt_status;
        pub fn hipblasLtMatrixLayoutDestroy(layout: hipblasLtMatrixLayout_t) -> hipblaslt_status;
    }

    // --- the mangled `hipblaslt_ext::` helpers (the 3 C++ helpers) ----------
    //
    // The C++ signatures take/return references and a `std::string`; the
    // declarations model them as pointers (the port declares the surface
    // for completeness and never calls them through this module).

    extern "C" {
        #[link_name = "_ZN13hipblaslt_ext16getIndexFromAlgoER22_hipblasLtMatmulAlgo_t"]
        pub fn hipblaslt_ext_getIndexFromAlgo(algo: *mut _hipblasLtMatmulAlgo_t) -> c_int;
        #[link_name = "_ZN13hipblaslt_ext17getAlgosFromIndexEPvRSt6vectorIiSaIiEERS1_I33_hipblasLtMatmulHeuristicResult_tSaIS5_EE"]
        pub fn hipblaslt_ext_getAlgosFromIndex(
            handle: *mut c_void,
            index: *mut c_void,
            results: *mut c_void,
        );
        #[link_name = "_ZN13hipblaslt_ext23getSolutionNameFromAlgoB5cxx11EPvR22_hipblasLtMatmulAlgo_t"]
        pub fn hipblaslt_ext_getSolutionNameFromAlgo(
            handle: *mut c_void,
            algo: *mut _hipblasLtMatmulAlgo_t,
        ) -> *mut c_char;
    }

    // --- the launch fast path and the loader ---------------------------------

    // The `hipLaunchKernel` fast path (spec §3.1): pushes the launch
    // configuration consumed by the fatbin kernel entry.
    extern "C" {
        pub fn __hipPushCallConfiguration(
            gridDim: dim3,
            blockDim: dim3,
            sharedMemBytes: usize,
            stream: *mut c_void,
        ) -> hipError_t;
    }

    // --- the module API (code-object loading) --------------------------------
    //
    // The `hipModule*` entry points the fatbin loader ([`crate::fatbin`])
    // uses to load the reference's shipped GPU code objects. They live in
    // the same `libamdhip64.so.7` but are not part of the reference's
    // import catalogue (the binary embeds its code objects in the
    // `.hip_fatbin` section instead of loading them at runtime), so they
    // are catalogued separately in [`HIP_MODULE_SYMBOLS`].

    pub type hipModule_t = *mut c_void;
    pub type hipFunction_t = *mut c_void;

    extern "C" {
        pub fn hipModuleLoadData(module: *mut hipModule_t, image: *const c_void) -> hipError_t;
        pub fn hipModuleLoadDataEx(
            module: *mut hipModule_t,
            image: *const c_void,
            numOptions: c_uint,
            options: *mut c_void,
            optionValues: *mut *mut c_void,
        ) -> hipError_t;
        pub fn hipModuleGetFunction(
            func: *mut hipFunction_t,
            module: hipModule_t,
            name: *const c_char,
        ) -> hipError_t;
        pub fn hipModuleLaunchKernel(
            func: hipFunction_t,
            gridDimX: c_uint,
            gridDimY: c_uint,
            gridDimZ: c_uint,
            blockDimX: c_uint,
            blockDimY: c_uint,
            blockDimZ: c_uint,
            sharedMemBytes: c_uint,
            stream: hipStream_t,
            kernelParams: *mut *mut c_void,
            extra: *mut *mut c_void,
        ) -> hipError_t;
        pub fn hipModuleGetGlobal(
            devPtr: *mut *mut c_void,
            bytes: *mut usize,
            module: hipModule_t,
            name: *const c_char,
        ) -> hipError_t;
        pub fn hipModuleUnload(module: hipModule_t) -> hipError_t;
    }

    pub const RTLD_LAZY: c_int = 1;

    extern "C" {
        pub fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        pub fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        pub fn dlclose(handle: *mut c_void) -> c_int;
    }

    // --- the symbol catalogues (the JSON's exact order) ----------------------

    /// The 36 HIP runtime symbols of the import catalogue.
    pub const HIP_RUNTIME_SYMBOLS: [&str; 36] = [
        "hipDeviceGetPCIBusId",
        "hipDeviceGetStreamPriorityRange",
        "hipDeviceSynchronize",
        "hipEventCreate",
        "hipEventCreateWithFlags",
        "hipEventDestroy",
        "hipEventElapsedTime",
        "hipEventRecord",
        "hipEventSynchronize",
        "hipFuncGetAttributes",
        "hipGetDevice",
        "hipGetDeviceCount",
        "hipGetDevicePropertiesR0600",
        "hipGetErrorString",
        "hipGetLastError",
        "hipHostFree",
        "hipHostGetDevicePointer",
        "hipHostMalloc",
        "hipHostRegister",
        "hipHostUnregister",
        "hipLaunchKernel",
        "hipMalloc",
        "hipMemcpy",
        "hipMemcpy2D",
        "hipMemcpyAsync",
        "hipMemset",
        "hipMemsetAsync",
        "hipMemsetD32Async",
        "hipOccupancyMaxActiveBlocksPerMultiprocessor",
        "hipRuntimeGetVersion",
        "hipSetDevice",
        "hipStreamCreateWithFlags",
        "hipStreamCreateWithPriority",
        "hipStreamDestroy",
        "hipStreamSynchronize",
        "hipStreamWaitEvent",
    ];

    /// The 13 `hipblasLt*` symbols of the import catalogue.
    pub const HIPBLASLT_SYMBOLS: [&str; 13] = [
        "hipblasLtCreate",
        "hipblasLtDestroy",
        "hipblasLtGetVersion",
        "hipblasLtMatmul",
        "hipblasLtMatmulAlgoGetHeuristic",
        "hipblasLtMatmulDescCreate",
        "hipblasLtMatmulDescDestroy",
        "hipblasLtMatmulDescSetAttribute",
        "hipblasLtMatmulPreferenceCreate",
        "hipblasLtMatmulPreferenceDestroy",
        "hipblasLtMatmulPreferenceSetAttribute",
        "hipblasLtMatrixLayoutCreate",
        "hipblasLtMatrixLayoutDestroy",
    ];

    /// The 3 mangled `hipblaslt_ext::` symbols of the import catalogue.
    pub const HIPBLASLT_EXT_SYMBOLS: [&str; 3] = [
        "_ZN13hipblaslt_ext16getIndexFromAlgoER22_hipblasLtMatmulAlgo_t",
        "_ZN13hipblaslt_ext17getAlgosFromIndexEPvRSt6vectorIiSaIiEERS1_I33_hipblasLtMatmulHeuristicResult_tSaIS5_EE",
        "_ZN13hipblaslt_ext23getSolutionNameFromAlgoB5cxx11EPvR22_hipblasLtMatmulAlgo_t",
    ];

    /// The 6 `hipModule*` entry points the fatbin loader uses to load the
    /// reference's code objects (exported by the same library, but not
    /// part of the import catalogue the reference binary itself links).
    pub const HIP_MODULE_SYMBOLS: [&str; 6] = [
        "hipModuleLoadData",
        "hipModuleLoadDataEx",
        "hipModuleGetFunction",
        "hipModuleLaunchKernel",
        "hipModuleGetGlobal",
        "hipModuleUnload",
    ];
}

// ---------------------------------------------------------------------------
// Runtime loading (dlopen-style) and backend selection
// ---------------------------------------------------------------------------

/// Resolves `symbol` from an open library handle as the function-pointer
/// type `$ty`, or returns `None` from the enclosing `load` when the
/// symbol is missing (the port then treats the runtime as absent). A
/// missing symbol closes the handle first, so a failed load does not
/// leak the dlopen reference.
macro_rules! bind {
    ($handle:expr, $symbol:expr, $ty:ty) => {{
        let name = CString::new($symbol).expect("symbol name without interior nul");
        let ptr = unsafe { ffi::dlsym($handle, name.as_ptr()) };
        if ptr.is_null() {
            unsafe { ffi::dlclose($handle) };
            return None;
        }
        unsafe { core::mem::transmute::<*mut c_void, $ty>(ptr) }
    }};
}

/// The 36 HIP runtime entry points plus the 6 `hipModule*` entry points
/// (the [`ffi::HIP_MODULE_SYMBOLS`] family), resolved from
/// `libamdhip64.so.7`. The hipBLASLt surface (the `ffi::HIPBLASLT_SYMBOLS` /
/// `ffi::HIPBLASLT_EXT_SYMBOLS` families) lives in `libhipblaslt.so.1`;
/// the port declares it in [`ffi`] for surface completeness and loads it
/// through the same dlopen pattern when the matmul path is exercised.
#[allow(non_snake_case)]
pub struct HipRuntime {
    pub hipDeviceGetPCIBusId: unsafe extern "C" fn(*mut c_char, c_int, c_int) -> ffi::hipError_t,
    pub hipDeviceGetStreamPriorityRange:
        unsafe extern "C" fn(*mut c_int, *mut c_int, c_int) -> ffi::hipError_t,
    pub hipDeviceSynchronize: unsafe extern "C" fn() -> ffi::hipError_t,
    pub hipEventCreate: unsafe extern "C" fn(*mut ffi::hipEvent_t) -> ffi::hipError_t,
    pub hipEventCreateWithFlags:
        unsafe extern "C" fn(*mut ffi::hipEvent_t, c_uint) -> ffi::hipError_t,
    pub hipEventDestroy: unsafe extern "C" fn(ffi::hipEvent_t) -> ffi::hipError_t,
    pub hipEventElapsedTime:
        unsafe extern "C" fn(*mut c_float, ffi::hipEvent_t, ffi::hipEvent_t) -> ffi::hipError_t,
    pub hipEventRecord: unsafe extern "C" fn(ffi::hipEvent_t, ffi::hipStream_t) -> ffi::hipError_t,
    pub hipEventSynchronize: unsafe extern "C" fn(ffi::hipEvent_t) -> ffi::hipError_t,
    pub hipFuncGetAttributes: unsafe extern "C" fn(*mut c_void, *const c_void) -> ffi::hipError_t,
    pub hipGetDevice: unsafe extern "C" fn(*mut c_int) -> ffi::hipError_t,
    pub hipGetDeviceCount: unsafe extern "C" fn(*mut c_int) -> ffi::hipError_t,
    pub hipGetDevicePropertiesR0600:
        unsafe extern "C" fn(*mut ffi::hipDevicePropStructR0600, c_int) -> ffi::hipError_t,
    pub hipGetErrorString: unsafe extern "C" fn(ffi::hipError_t) -> *const c_char,
    pub hipGetLastError: unsafe extern "C" fn() -> ffi::hipError_t,
    pub hipHostFree: unsafe extern "C" fn(*mut c_void) -> ffi::hipError_t,
    pub hipHostGetDevicePointer:
        unsafe extern "C" fn(*mut *mut c_void, *mut c_void, c_uint) -> ffi::hipError_t,
    pub hipHostMalloc: unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> ffi::hipError_t,
    pub hipHostRegister: unsafe extern "C" fn(*mut c_void, usize, c_uint) -> ffi::hipError_t,
    pub hipHostUnregister: unsafe extern "C" fn(*mut c_void) -> ffi::hipError_t,
    pub hipLaunchKernel: unsafe extern "C" fn(
        *const c_void,
        ffi::dim3,
        ffi::dim3,
        *mut *mut c_void,
        usize,
        ffi::hipStream_t,
    ) -> ffi::hipError_t,
    pub hipMalloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> ffi::hipError_t,
    pub hipMemcpy: unsafe extern "C" fn(
        *mut c_void,
        *const c_void,
        usize,
        ffi::hipMemcpyKind,
    ) -> ffi::hipError_t,
    pub hipMemcpy2D: unsafe extern "C" fn(
        *mut c_void,
        usize,
        *const c_void,
        usize,
        usize,
        usize,
        ffi::hipMemcpyKind,
    ) -> ffi::hipError_t,
    pub hipMemcpyAsync: unsafe extern "C" fn(
        *mut c_void,
        *const c_void,
        usize,
        ffi::hipMemcpyKind,
        ffi::hipStream_t,
    ) -> ffi::hipError_t,
    pub hipMemset: unsafe extern "C" fn(*mut c_void, c_int, usize) -> ffi::hipError_t,
    pub hipMemsetAsync:
        unsafe extern "C" fn(*mut c_void, c_int, usize, ffi::hipStream_t) -> ffi::hipError_t,
    pub hipMemsetD32Async:
        unsafe extern "C" fn(*mut c_void, c_uint, usize, ffi::hipStream_t) -> ffi::hipError_t,
    pub hipOccupancyMaxActiveBlocksPerMultiprocessor:
        unsafe extern "C" fn(*mut c_int, *const c_void, c_int, usize) -> ffi::hipError_t,
    pub hipRuntimeGetVersion: unsafe extern "C" fn(*mut c_int) -> ffi::hipError_t,
    pub hipSetDevice: unsafe extern "C" fn(c_int) -> ffi::hipError_t,
    pub hipStreamCreateWithFlags:
        unsafe extern "C" fn(*mut ffi::hipStream_t, c_uint) -> ffi::hipError_t,
    pub hipStreamCreateWithPriority:
        unsafe extern "C" fn(*mut ffi::hipStream_t, c_uint, c_int) -> ffi::hipError_t,
    pub hipStreamDestroy: unsafe extern "C" fn(ffi::hipStream_t) -> ffi::hipError_t,
    pub hipStreamSynchronize: unsafe extern "C" fn(ffi::hipStream_t) -> ffi::hipError_t,
    pub hipStreamWaitEvent:
        unsafe extern "C" fn(ffi::hipStream_t, ffi::hipEvent_t, c_uint) -> ffi::hipError_t,
    pub hipModuleLoadData:
        unsafe extern "C" fn(*mut ffi::hipModule_t, *const c_void) -> ffi::hipError_t,
    pub hipModuleLoadDataEx: unsafe extern "C" fn(
        *mut ffi::hipModule_t,
        *const c_void,
        c_uint,
        *mut c_void,
        *mut *mut c_void,
    ) -> ffi::hipError_t,
    pub hipModuleGetFunction: unsafe extern "C" fn(
        *mut ffi::hipFunction_t,
        ffi::hipModule_t,
        *const c_char,
    ) -> ffi::hipError_t,
    pub hipModuleLaunchKernel: unsafe extern "C" fn(
        ffi::hipFunction_t,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        ffi::hipStream_t,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> ffi::hipError_t,
    pub hipModuleGetGlobal: unsafe extern "C" fn(
        *mut *mut c_void,
        *mut usize,
        ffi::hipModule_t,
        *const c_char,
    ) -> ffi::hipError_t,
    pub hipModuleUnload: unsafe extern "C" fn(ffi::hipModule_t) -> ffi::hipError_t,
}

impl HipRuntime {
    /// Opens `libamdhip64.so.7` (the reference's `DT_NEEDED` soname; the
    /// unversioned `libamdhip64.so` is the fallback) with `RTLD_LAZY` and
    /// resolves every symbol of [`ffi::HIP_RUNTIME_SYMBOLS`]. Any missing
    /// symbol yields `None` — the port then runs on [`Backend::Cpu`].
    pub fn load() -> Option<Self> {
        let handle = open_library()?;
        Some(Self {
            hipDeviceGetPCIBusId: bind!(
                handle,
                "hipDeviceGetPCIBusId",
                unsafe extern "C" fn(*mut c_char, c_int, c_int) -> ffi::hipError_t
            ),
            hipDeviceGetStreamPriorityRange: bind!(
                handle,
                "hipDeviceGetStreamPriorityRange",
                unsafe extern "C" fn(*mut c_int, *mut c_int, c_int) -> ffi::hipError_t
            ),
            hipDeviceSynchronize: bind!(
                handle,
                "hipDeviceSynchronize",
                unsafe extern "C" fn() -> ffi::hipError_t
            ),
            hipEventCreate: bind!(
                handle,
                "hipEventCreate",
                unsafe extern "C" fn(*mut ffi::hipEvent_t) -> ffi::hipError_t
            ),
            hipEventCreateWithFlags: bind!(
                handle,
                "hipEventCreateWithFlags",
                unsafe extern "C" fn(*mut ffi::hipEvent_t, c_uint) -> ffi::hipError_t
            ),
            hipEventDestroy: bind!(
                handle,
                "hipEventDestroy",
                unsafe extern "C" fn(ffi::hipEvent_t) -> ffi::hipError_t
            ),
            hipEventElapsedTime: bind!(
                handle,
                "hipEventElapsedTime",
                unsafe extern "C" fn(
                    *mut c_float,
                    ffi::hipEvent_t,
                    ffi::hipEvent_t,
                ) -> ffi::hipError_t
            ),
            hipEventRecord: bind!(
                handle,
                "hipEventRecord",
                unsafe extern "C" fn(ffi::hipEvent_t, ffi::hipStream_t) -> ffi::hipError_t
            ),
            hipEventSynchronize: bind!(
                handle,
                "hipEventSynchronize",
                unsafe extern "C" fn(ffi::hipEvent_t) -> ffi::hipError_t
            ),
            hipFuncGetAttributes: bind!(
                handle,
                "hipFuncGetAttributes",
                unsafe extern "C" fn(*mut c_void, *const c_void) -> ffi::hipError_t
            ),
            hipGetDevice: bind!(
                handle,
                "hipGetDevice",
                unsafe extern "C" fn(*mut c_int) -> ffi::hipError_t
            ),
            hipGetDeviceCount: bind!(
                handle,
                "hipGetDeviceCount",
                unsafe extern "C" fn(*mut c_int) -> ffi::hipError_t
            ),
            hipGetDevicePropertiesR0600: bind!(
                handle,
                "hipGetDevicePropertiesR0600",
                unsafe extern "C" fn(*mut ffi::hipDevicePropStructR0600, c_int) -> ffi::hipError_t
            ),
            hipGetErrorString: bind!(
                handle,
                "hipGetErrorString",
                unsafe extern "C" fn(ffi::hipError_t) -> *const c_char
            ),
            hipGetLastError: bind!(
                handle,
                "hipGetLastError",
                unsafe extern "C" fn() -> ffi::hipError_t
            ),
            hipHostFree: bind!(
                handle,
                "hipHostFree",
                unsafe extern "C" fn(*mut c_void) -> ffi::hipError_t
            ),
            hipHostGetDevicePointer: bind!(
                handle,
                "hipHostGetDevicePointer",
                unsafe extern "C" fn(*mut *mut c_void, *mut c_void, c_uint) -> ffi::hipError_t
            ),
            hipHostMalloc: bind!(
                handle,
                "hipHostMalloc",
                unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> ffi::hipError_t
            ),
            hipHostRegister: bind!(
                handle,
                "hipHostRegister",
                unsafe extern "C" fn(*mut c_void, usize, c_uint) -> ffi::hipError_t
            ),
            hipHostUnregister: bind!(
                handle,
                "hipHostUnregister",
                unsafe extern "C" fn(*mut c_void) -> ffi::hipError_t
            ),
            hipLaunchKernel: bind!(
                handle,
                "hipLaunchKernel",
                unsafe extern "C" fn(
                    *const c_void,
                    ffi::dim3,
                    ffi::dim3,
                    *mut *mut c_void,
                    usize,
                    ffi::hipStream_t,
                ) -> ffi::hipError_t
            ),
            hipMalloc: bind!(
                handle,
                "hipMalloc",
                unsafe extern "C" fn(*mut *mut c_void, usize) -> ffi::hipError_t
            ),
            hipMemcpy: bind!(
                handle,
                "hipMemcpy",
                unsafe extern "C" fn(
                    *mut c_void,
                    *const c_void,
                    usize,
                    ffi::hipMemcpyKind,
                ) -> ffi::hipError_t
            ),
            hipMemcpy2D: bind!(
                handle,
                "hipMemcpy2D",
                unsafe extern "C" fn(
                    *mut c_void,
                    usize,
                    *const c_void,
                    usize,
                    usize,
                    usize,
                    ffi::hipMemcpyKind,
                ) -> ffi::hipError_t
            ),
            hipMemcpyAsync: bind!(
                handle,
                "hipMemcpyAsync",
                unsafe extern "C" fn(
                    *mut c_void,
                    *const c_void,
                    usize,
                    ffi::hipMemcpyKind,
                    ffi::hipStream_t,
                ) -> ffi::hipError_t
            ),
            hipMemset: bind!(
                handle,
                "hipMemset",
                unsafe extern "C" fn(*mut c_void, c_int, usize) -> ffi::hipError_t
            ),
            hipMemsetAsync: bind!(
                handle,
                "hipMemsetAsync",
                unsafe extern "C" fn(
                    *mut c_void,
                    c_int,
                    usize,
                    ffi::hipStream_t,
                ) -> ffi::hipError_t
            ),
            hipMemsetD32Async: bind!(
                handle,
                "hipMemsetD32Async",
                unsafe extern "C" fn(
                    *mut c_void,
                    c_uint,
                    usize,
                    ffi::hipStream_t,
                ) -> ffi::hipError_t
            ),
            hipOccupancyMaxActiveBlocksPerMultiprocessor: bind!(
                handle,
                "hipOccupancyMaxActiveBlocksPerMultiprocessor",
                unsafe extern "C" fn(*mut c_int, *const c_void, c_int, usize) -> ffi::hipError_t
            ),
            hipRuntimeGetVersion: bind!(
                handle,
                "hipRuntimeGetVersion",
                unsafe extern "C" fn(*mut c_int) -> ffi::hipError_t
            ),
            hipSetDevice: bind!(
                handle,
                "hipSetDevice",
                unsafe extern "C" fn(c_int) -> ffi::hipError_t
            ),
            hipStreamCreateWithFlags: bind!(
                handle,
                "hipStreamCreateWithFlags",
                unsafe extern "C" fn(*mut ffi::hipStream_t, c_uint) -> ffi::hipError_t
            ),
            hipStreamCreateWithPriority: bind!(
                handle,
                "hipStreamCreateWithPriority",
                unsafe extern "C" fn(*mut ffi::hipStream_t, c_uint, c_int) -> ffi::hipError_t
            ),
            hipStreamDestroy: bind!(
                handle,
                "hipStreamDestroy",
                unsafe extern "C" fn(ffi::hipStream_t) -> ffi::hipError_t
            ),
            hipStreamSynchronize: bind!(
                handle,
                "hipStreamSynchronize",
                unsafe extern "C" fn(ffi::hipStream_t) -> ffi::hipError_t
            ),
            hipStreamWaitEvent: bind!(
                handle,
                "hipStreamWaitEvent",
                unsafe extern "C" fn(ffi::hipStream_t, ffi::hipEvent_t, c_uint) -> ffi::hipError_t
            ),
            hipModuleLoadData: bind!(
                handle,
                "hipModuleLoadData",
                unsafe extern "C" fn(*mut ffi::hipModule_t, *const c_void) -> ffi::hipError_t
            ),
            hipModuleLoadDataEx: bind!(
                handle,
                "hipModuleLoadDataEx",
                unsafe extern "C" fn(
                    *mut ffi::hipModule_t,
                    *const c_void,
                    c_uint,
                    *mut c_void,
                    *mut *mut c_void,
                ) -> ffi::hipError_t
            ),
            hipModuleGetFunction: bind!(
                handle,
                "hipModuleGetFunction",
                unsafe extern "C" fn(
                    *mut ffi::hipFunction_t,
                    ffi::hipModule_t,
                    *const c_char,
                ) -> ffi::hipError_t
            ),
            hipModuleLaunchKernel: bind!(
                handle,
                "hipModuleLaunchKernel",
                unsafe extern "C" fn(
                    ffi::hipFunction_t,
                    c_uint,
                    c_uint,
                    c_uint,
                    c_uint,
                    c_uint,
                    c_uint,
                    c_uint,
                    ffi::hipStream_t,
                    *mut *mut c_void,
                    *mut *mut c_void,
                ) -> ffi::hipError_t
            ),
            hipModuleGetGlobal: bind!(
                handle,
                "hipModuleGetGlobal",
                unsafe extern "C" fn(
                    *mut *mut c_void,
                    *mut usize,
                    ffi::hipModule_t,
                    *const c_char,
                ) -> ffi::hipError_t
            ),
            hipModuleUnload: bind!(
                handle,
                "hipModuleUnload",
                unsafe extern "C" fn(ffi::hipModule_t) -> ffi::hipError_t
            ),
        })
    }
}

/// Opens the HIP runtime library: the reference's `DT_NEEDED` soname
/// first, then the unversioned dev name.
fn open_library() -> Option<*mut c_void> {
    for soname in ["libamdhip64.so.7", "libamdhip64.so"] {
        let name = CString::new(soname).ok()?;
        let handle = unsafe { ffi::dlopen(name.as_ptr(), ffi::RTLD_LAZY) };
        if !handle.is_null() {
            return Some(handle);
        }
    }
    None
}

// ---------------------------------------------------------------------------

/// The compute backend the port runs on. The reference only ever runs on
/// HIP (without it it exits with the `ROCM_NO_GPU` message); the port
/// falls back to [`Backend::Cpu`] — the CPU forward path of
/// [`crate::forward`] — whenever the HIP runtime cannot be loaded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Hip,
}

impl Backend {
    pub fn label(&self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Hip => "hip",
        }
    }
}

/// The pure backend-selection rule: HIP when the runtime is loadable,
/// CPU otherwise (kept separate from [`runtime`] so tests can exercise
/// both branches without a GPU).
pub fn select_backend(loadable: bool) -> Backend {
    if loadable {
        Backend::Hip
    } else {
        Backend::Cpu
    }
}

/// The process-wide HIP runtime probe (loaded once on first call;
/// `RTLD_LAZY` keeps the cost low when the library exists).
pub fn runtime() -> Option<&'static HipRuntime> {
    static RUNTIME: OnceLock<Option<HipRuntime>> = OnceLock::new();
    RUNTIME.get_or_init(HipRuntime::load).as_ref()
}

/// The backend the process runs on (the serve path logs it under
/// `HALOGEN_LOG`).
pub fn active() -> Backend {
    select_backend(runtime().is_some())
}

// ---------------------------------------------------------------------------
// Device lifecycle (the `FUN_01f700f0` port)
// ---------------------------------------------------------------------------

/// The device name `HALOGEN_GPU_DEVICE=auto` and the re-exec rule look
/// for (the Strix Halo APU).
const STRIX_HALO: &str = "gfx1151";

/// One enumerated device: the HIP index, the `hipGetDevicePropertiesR0600`
/// name and the `hipDeviceGetPCIBusId` string (the reference reads all
/// three before binding; the port models them as data so the selection
/// logic is testable without a GPU).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceEntry {
    pub index: i32,
    pub name: String,
    pub pci_bus_id: Option<String>,
}

/// The device-selection environment the reference reads: the four
/// ROCm/CUDA visibility variables plus `HALOGEN_GPU_DEVICE`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DeviceEnv {
    pub hip_visible_devices: Option<String>,
    pub rocr_visible_devices: Option<String>,
    pub cuda_visible_devices: Option<String>,
    pub gpu_device_ordinal: Option<String>,
    pub halogen_gpu_device: Option<String>,
}

impl DeviceEnv {
    /// Reads the five variables from the process environment.
    pub fn from_env() -> Self {
        Self {
            hip_visible_devices: env::var("HIP_VISIBLE_DEVICES").ok(),
            rocr_visible_devices: env::var("ROCR_VISIBLE_DEVICES").ok(),
            cuda_visible_devices: env::var("CUDA_VISIBLE_DEVICES").ok(),
            gpu_device_ordinal: env::var("GPU_DEVICE_ORDINAL").ok(),
            halogen_gpu_device: env::var("HALOGEN_GPU_DEVICE").ok(),
        }
    }

    /// The reference's visibility precedence: `HIP_VISIBLE_DEVICES` →
    /// `ROCR_VISIBLE_DEVICES` → `CUDA_VISIBLE_DEVICES` →
    /// `GPU_DEVICE_ORDINAL`. An empty value still counts as set (an
    /// empty `HIP_VISIBLE_DEVICES` hides every device, which is how the
    /// reference reaches the `ROCM_NO_GPU` message on a machine that has
    /// a GPU).
    pub fn visible_devices(&self) -> Option<&str> {
        self.hip_visible_devices
            .as_deref()
            .or(self.rocr_visible_devices.as_deref())
            .or(self.cuda_visible_devices.as_deref())
            .or(self.gpu_device_ordinal.as_deref())
    }
}

/// The resolved device plus the visibility rewrite the reference applies
/// before binding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceSelection {
    pub device: DeviceEntry,
    /// The reference re-execs (`setenv("HIP_VISIBLE_DEVICES", index)` +
    /// `unsetenv("HALOGEN_GPU_DEVICE")` + `execv("/proc/self/exe")`) when
    /// no visibility variable is set, more than one device exists and the
    /// selected device is the Strix Halo GPU — pinning the process to the
    /// APU (the decompile also consults a `DAT_021276e8` flag whose
    /// provenance is unresolved; the port treats the rule as the
    /// decompile's control flow). The port records the rewrite instead
    /// of exec-ing.
    pub visible_devices_rewrite: Option<i32>,
    /// Whether the reference would print the device banner: it does so
    /// when the selection is not the trivial single-device default (more
    /// than one device in view, a non-zero device, or an explicit
    /// `HALOGEN_GPU_DEVICE` request).
    pub banner_shown: bool,
}

impl DeviceSelection {
    /// The device banner the reference prints after binding:
    /// `halogen: GPU device {i}: {name}` — with `, the Strix Halo GPU.`
    /// appended for the APU.
    pub fn banner(&self) -> String {
        if self.device.name == STRIX_HALO {
            format!(
                "halogen: GPU device {}: {}, the Strix Halo GPU.",
                self.device.index, self.device.name
            )
        } else {
            format!(
                "halogen: GPU device {}: {}",
                self.device.index, self.device.name
            )
        }
    }

    /// The re-exec line the reference prints alongside the banner
    /// (`fprintf("%s\n%s\n", banner, reexec)`) when the rewrite rule
    /// restarts the process pinned to a single device; `None` when no
    /// re-exec happens.
    pub fn reexec_message(&self) -> Option<String> {
        self.visible_devices_rewrite.map(|index| {
            format!(
                "halogen: starting the engine again with only device {index}, and this build runs on the Strix Halo GPU ({})",
                self.device.name
            )
        })
    }
}

/// The wording the reference prints when the re-exec fails and the
/// process continues with every GPU in view.
pub fn reexec_failure_message(error: &str) -> String {
    format!("halogen: could not start the engine again ({error}); it runs with every GPU in view, as before")
}

/// Resolves a `HALOGEN_GPU_DEVICE` request against the enumerated
/// devices: a device number, `first` (device 0 — the default when the
/// variable is unset) or `auto` (the Strix Halo GPU).
pub fn resolve_request(devices: &[DeviceEntry], request: &str) -> Result<DeviceEntry, String> {
    let list = device_list(devices);
    if request == "first" {
        return devices
            .first()
            .cloned()
            .ok_or_else(|| crate::ROCM_NO_GPU.to_string());
    }
    if request == "auto" {
        return devices
            .iter()
            .find(|device| device.name == STRIX_HALO)
            .cloned()
            .ok_or_else(|| format!("halogen: ROCm offers no Strix Halo GPU ({list})"));
    }
    if let Ok(index) = request.parse::<i32>() {
        if let Some(device) = devices.iter().find(|device| device.index == index) {
            return Ok(device.clone());
        }
    }
    Err(format!(
        "halogen: HALOGEN_GPU_DEVICE=\"{request}\" is not a device number. It takes one HIP device number (0, 1, ...), or `first` for device 0, or `auto` for the Strix Halo GPU. Devices ROCm offers: {list}"
    ))
}

/// Renders the device list the error wordings append (`0 (gfx1151),
/// 1 (gfx9...)`).
fn device_list(devices: &[DeviceEntry]) -> String {
    devices
        .iter()
        .map(|device| format!("{} ({})", device.index, device.name))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The pure device-selection rule (the `FUN_01f700f0` decision tree):
/// empty enumeration → the `ROCM_NO_GPU` wording; otherwise resolve the
/// request (default `first`) and compute the re-exec rewrite rule.
pub fn select_device(devices: &[DeviceEntry], env: &DeviceEnv) -> Result<DeviceSelection, String> {
    if devices.is_empty() {
        return Err(crate::ROCM_NO_GPU.to_string());
    }
    let request = env.halogen_gpu_device.as_deref().unwrap_or("first");
    let device = resolve_request(devices, request)?;
    let no_visibility_vars = env.hip_visible_devices.is_none()
        && env.rocr_visible_devices.is_none()
        && env.cuda_visible_devices.is_none()
        && env.gpu_device_ordinal.is_none();
    let visible_devices_rewrite =
        if no_visibility_vars && devices.len() > 1 && device.name == STRIX_HALO {
            Some(device.index)
        } else {
            None
        };
    Ok(DeviceSelection {
        device: device.clone(),
        visible_devices_rewrite,
        banner_shown: should_show_banner(devices, &device, env),
    })
}

/// Whether the reference prints the device banner: it does so whenever
/// the selection is not the trivial single-device default (more than one
/// device in view, a non-zero device, or an explicit non-empty
/// `HALOGEN_GPU_DEVICE` request).
fn should_show_banner(devices: &[DeviceEntry], device: &DeviceEntry, env: &DeviceEnv) -> bool {
    devices.len() > 1
        || device.index != 0
        || env
            .halogen_gpu_device
            .as_ref()
            .is_some_and(|value| !value.is_empty())
}

/// The full lifecycle port: select the device, then bind it through
/// `set_device` (the reference's `hipSetDevice`; the injectable closure
/// keeps the path testable). A failed bind produces the reference's
/// `halogen: cannot use GPU device {index}.` wording.
pub fn bind_device(
    devices: &[DeviceEntry],
    env: &DeviceEnv,
    set_device: impl Fn(i32) -> Result<(), i32>,
) -> Result<DeviceSelection, String> {
    let selection = select_device(devices, env)?;
    set_device(selection.device.index)
        .map_err(|_| format!("halogen: cannot use GPU device {}.", selection.device.index))?;
    Ok(selection)
}

// ---------------------------------------------------------------------------
// Device-allocation accounting (the `FUN_01f82130` port)
// ---------------------------------------------------------------------------

/// The running total of device allocations: the reference's `dmalloc:`
/// accounting feeding the kvcache pool budget math (the `total ... GiB,
/// #N` counters of the log line and the `after ... GiB in N allocations`
/// wording of the failure report).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceAllocAccount {
    pub total_bytes: u64,
    pub count: u64,
}

impl DeviceAllocAccount {
    /// Records one successful allocation of `bytes`.
    pub fn record(&mut self, bytes: u64) {
        self.total_bytes += bytes;
        self.count += 1;
    }

    /// The success log line `dmalloc: %10.3f MiB  (total %8.3f GiB, #%ld)`
    /// for the just-recorded allocation of `bytes`.
    pub fn success_line(&self, bytes: u64) -> String {
        format!(
            "dmalloc: {:>10.3} MiB  (total {:>8.3} GiB, #{})",
            bytes as f64 / MIB,
            self.total_bytes as f64 / GIB,
            self.count
        )
    }

    /// The failure report: the `dmalloc: FAILED ...` line, the reference's
    /// budget guidance block and the `HIP %s:%d: %s` line naming
    /// `/src/halogen/src/flash_ops.h:161` (the source location the
    /// reference's HIP wrapper reports for the failed `hipMalloc`).
    pub fn failure_report(&self, requested: u64, error: &str) -> String {
        format!(
            "dmalloc: FAILED requesting {:.3} GiB after {:.3} GiB in {} allocations ({error})\n{DMALLOC_GUIDANCE}\nHIP /src/halogen/src/flash_ops.h:161: {error}",
            requested as f64 / GIB,
            self.total_bytes as f64 / GIB,
            self.count
        )
    }
}

const MIB: f64 = 1_048_576.0;
const GIB: f64 = 1_073_741_824.0;

/// The guidance block the reference prints after a failed device
/// allocation (binutils-strings.txt:1785-1790).
const DMALLOC_GUIDANCE: &str = "dmalloc: this is the device budget, and on this machine device memory IS system memory.
  LOWER ONE OF THESE, in this order:
    HALOGEN_KV_POOL_POSITIONS  positions resident across all conversations (~29.5 KiB each, so 262144 = ~7.2 GiB, 524288 = ~14.4, 786432 = ~21.6)
    HALOGEN_MAX_TOK            the single-call prefill arena (32768 = ~16.7 GiB, 16384 = ~8.4, at ~9% of prefill speed)
  HALOGEN_KV_SLOTS IS NOT A MEMORY KNOB since 0.3: the slots share one pool and a slot costs ~115 MiB.
  HALOGEN_DMALLOC_LOG=1 prints every allocation over 64 MiB, which is this configuration's budget measured.";

/// The eight device-buffer sizes of one slot record (the reference's
/// eight `hipMalloc` calls per slot; the kvcache module owns the
/// per-field constants).
const SLOT_RECORD_SIZES: [usize; 8] = [
    kvcache::ACTIVATION_STAGING,
    kvcache::SECONDARY_STAGING,
    kvcache::SMALL_CONTROL,
    kvcache::STATE_SMALL,
    kvcache::STATE_MAIN,
    kvcache::KV_REGION,
    kvcache::ALT_SMALL_CONTROL,
    kvcache::ALT_STATE_BLOCK,
];

/// The total device bytes one slot record allocates (the kvcache pool
/// budget math's per-slot cost).
pub fn slot_record_bytes() -> u64 {
    SLOT_RECORD_SIZES.iter().map(|&size| size as u64).sum()
}

/// Accounts the eight allocations of one slot record and returns the
/// total bytes recorded.
pub fn account_slot_record(account: &mut DeviceAllocAccount) -> u64 {
    let mut total = 0;
    for size in SLOT_RECORD_SIZES {
        account.record(size as u64);
        total += size as u64;
    }
    total
}

/// The `HALOGEN_DMALLOC_LOG` threshold (the reference's `dmalloc_log`
/// setting): unset or `0` disables the log, `1` logs every allocation
/// over 64 MiB, any other value is a byte count parsed like `strtol`
/// base 0 (decimal, `0x` hex or `0` octal, stopping at the first
/// invalid digit).
pub fn dmalloc_log_threshold(raw: Option<&str>) -> u64 {
    match raw.map(str::trim) {
        None | Some("") | Some("0") => 0,
        Some("1") => 64 * 1_048_576,
        Some(other) => strtol_base0(other).max(0) as u64,
    }
}

/// Parses `value` like C `strtol` with base 0 (decimal, `0x` hex or `0`
/// octal, stopping at the first invalid digit; overflow saturates).
fn strtol_base0(value: &str) -> i64 {
    let (negative, rest) = match value.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, value.strip_prefix('+').unwrap_or(value)),
    };
    let (radix, digits) = if rest.starts_with("0x") || rest.starts_with("0X") {
        (16, &rest[2..])
    } else if rest.starts_with('0') && rest.len() > 1 {
        (8, &rest[1..])
    } else {
        (10, rest)
    };
    let valid: String = digits.chars().take_while(|c| c.is_digit(radix)).collect();
    if valid.is_empty() {
        return 0;
    }
    let magnitude = u64::from_str_radix(&valid, radix).unwrap_or(i64::MAX as u64);
    if negative {
        if magnitude > i64::MAX as u64 {
            i64::MIN
        } else {
            -(magnitude as i64)
        }
    } else {
        magnitude.min(i64::MAX as u64) as i64
    }
}

/// The reference's logging rule: log when a threshold is set and the
/// allocation reaches it.
pub fn should_log(threshold: u64, bytes: u64) -> bool {
    threshold != 0 && threshold <= bytes
}

/// The `FUN_01f82130` core, injectable for tests: performs the
/// allocation, updates the account, and returns the log line (when the
/// `HALOGEN_DMALLOC_LOG` threshold logs it) or the failure report.
pub fn dmalloc_with(
    account: &mut DeviceAllocAccount,
    bytes: u64,
    threshold: u64,
    allocate: impl FnOnce(u64) -> Result<(), String>,
) -> Result<Option<String>, String> {
    match allocate(bytes) {
        Ok(()) => {
            account.record(bytes);
            Ok(if should_log(threshold, bytes) {
                Some(account.success_line(bytes))
            } else {
                None
            })
        }
        Err(error) => Err(account.failure_report(bytes, &error)),
    }
}

/// The process-wide allocation account (the reference's global counters).
static DMALLOC_ACCOUNT: Mutex<DeviceAllocAccount> = Mutex::new(DeviceAllocAccount {
    total_bytes: 0,
    count: 0,
});

/// The reference's `dmalloc`: `hipMalloc`s `bytes` through the loaded
/// runtime, logs the success line when `HALOGEN_DMALLOC_LOG` enables it,
/// and on failure prints the failure report and exits 1 (the reference
/// cannot serve without its device buffers).
pub fn dmalloc(runtime: &HipRuntime, bytes: u64) -> *mut c_void {
    let threshold = dmalloc_log_threshold(env::var("HALOGEN_DMALLOC_LOG").ok().as_deref());
    let mut device_ptr = core::ptr::null_mut();
    let error = match unsafe { (runtime.hipMalloc)(&mut device_ptr, bytes as usize) } {
        0 => None,
        code => Some(error_string(runtime, code)),
    };
    let mut account = DMALLOC_ACCOUNT
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    match dmalloc_with(&mut account, bytes, threshold, |_| match &error {
        None => Ok(()),
        Some(message) => Err(message.clone()),
    }) {
        Ok(_) => device_ptr,
        Err(report) => {
            drop(account);
            eprintln!("{report}");
            process::exit(1);
        }
    }
}

/// Renders a HIP error code through the runtime's `hipGetErrorString`
/// (the reference's `HIP %s:%d: %s` message body).
pub fn error_string(runtime: &HipRuntime, error: c_int) -> String {
    let raw = unsafe { (runtime.hipGetErrorString)(error) };
    if raw.is_null() {
        format!("hipError {error}")
    } else {
        unsafe { CStr::from_ptr(raw) }
            .to_string_lossy()
            .into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kvcache::SlotRecord;

    #[test]
    fn ffi_surface_matches_the_import_catalogue() {
        assert_eq!(ffi::HIP_RUNTIME_SYMBOLS.len(), 36);
        assert_eq!(ffi::HIPBLASLT_SYMBOLS.len(), 13);
        assert_eq!(ffi::HIPBLASLT_EXT_SYMBOLS.len(), 3);
        assert_eq!(ffi::HIP_MODULE_SYMBOLS.len(), 6);
        // Every catalogue symbol must appear in this file's FFI surface.
        // The HIP runtime and hipblasLt C APIs are declared as
        // `pub fn <name>`; the mangled C++ hipblaslt_ext API is declared
        // via `#[link_name = "<mangled name>"]` attributes on the
        // Rust-side names, so a bare `pub fn <mangled name>` search would
        // miss them.
        let source = include_str!("hip.rs");
        for symbol in ffi::HIP_RUNTIME_SYMBOLS
            .iter()
            .chain(ffi::HIPBLASLT_SYMBOLS.iter())
            .chain(ffi::HIP_MODULE_SYMBOLS.iter())
        {
            assert!(
                source.contains(&format!("pub fn {symbol}")),
                "imported symbol {symbol} has no FFI declaration"
            );
        }
        for symbol in ffi::HIPBLASLT_EXT_SYMBOLS.iter() {
            assert!(
                source.contains(&format!("#[link_name = \"{symbol}\"]")),
                "imported symbol {symbol} has no FFI declaration"
            );
        }
        // Cross-check the catalogues against the evidence JSON: the union
        // of the three const arrays must match the `hip_symbols` list
        // exactly, in both directions. The nix build's source filter
        // excludes docs/, so the check is best-effort there.
        let json_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../docs/analysis/evidence/flash_serve/npu/hip-imported-symbols.json"
        );
        let Ok(json) = std::fs::read_to_string(json_path) else {
            return;
        };
        let catalogue = hip_symbols_from_json(&json)
            .expect("the evidence JSON must contain a hip_symbols array");
        let declared: std::collections::BTreeSet<&str> = ffi::HIP_RUNTIME_SYMBOLS
            .iter()
            .chain(ffi::HIPBLASLT_SYMBOLS.iter())
            .chain(ffi::HIPBLASLT_EXT_SYMBOLS.iter())
            .copied()
            .collect();
        let listed: std::collections::BTreeSet<&str> =
            catalogue.iter().map(String::as_str).collect();
        assert_eq!(
            declared, listed,
            "the ffi symbol constants drift from the import catalogue"
        );
    }

    // Extract the quoted strings from the `hip_symbols` array of the
    // evidence JSON. The crate has no JSON parser dependency, so this
    // walks the array by hand: everything between `"hip_symbols": [` and
    // the closing `]` is a comma-separated list of quoted names.
    fn hip_symbols_from_json(json: &str) -> Option<Vec<String>> {
        let (_, rest) = json.split_once("\"hip_symbols\"")?;
        let (_, rest) = rest.split_once('[')?;
        let (body, _) = rest.split_once(']')?;
        let names = body
            .split(',')
            .map(str::trim)
            .map_while(|entry| {
                let (open, entry) = entry.split_once('"')?;
                debug_assert!(open.is_empty(), "entries must start with a quote");
                let (name, close) = entry.split_once('"')?;
                debug_assert!(close.is_empty(), "entries must end with a quote");
                Some(name.to_owned())
            })
            .collect::<Vec<_>>();
        (!names.is_empty()).then_some(names)
    }

    #[test]
    fn the_reexec_message_names_the_device_the_engine_restarts_on() {
        // The reference prints the re-exec line alongside the banner
        // (`fprintf "%s\n%s\n"`); the port only has a meaningful message
        // when a rewrite rule actually picked a single device.
        let rewritten = DeviceSelection {
            device: device(1, "gfx1151 (Radeon 8060S)"),
            visible_devices_rewrite: Some(1),
            banner_shown: true,
        };
        assert_eq!(
            rewritten.reexec_message().as_deref(),
            Some("halogen: starting the engine again with only device 1, and this build runs on the Strix Halo GPU (gfx1151 (Radeon 8060S))")
        );

        let passthrough = DeviceSelection {
            device: device(0, "gfx1151 (Radeon 8060S)"),
            visible_devices_rewrite: None,
            banner_shown: true,
        };
        assert_eq!(passthrough.reexec_message(), None);
    }

    #[test]
    fn the_reexec_failure_message_falls_back_to_every_gpu_in_view() {
        assert_eq!(
            reexec_failure_message("no such file"),
            "halogen: could not start the engine again (no such file); it runs with every GPU in view, as before"
        );
    }

    fn device(index: i32, name: &str) -> DeviceEntry {
        DeviceEntry {
            index,
            name: name.to_string(),
            pci_bus_id: Some(format!("0000:03:00.{index}")),
        }
    }

    #[test]
    fn device_selection_follows_the_visible_devices_precedence() {
        // HIP → ROCR → CUDA → GPU_DEVICE_ORDINAL.
        let env = DeviceEnv {
            rocr_visible_devices: Some("9".to_string()),
            cuda_visible_devices: Some("8".to_string()),
            gpu_device_ordinal: Some("7".to_string()),
            ..DeviceEnv::default()
        };
        assert_eq!(env.visible_devices(), Some("9"));
        let env = DeviceEnv {
            cuda_visible_devices: Some("8".to_string()),
            gpu_device_ordinal: Some("7".to_string()),
            ..DeviceEnv::default()
        };
        assert_eq!(env.visible_devices(), Some("8"));
        let env = DeviceEnv {
            gpu_device_ordinal: Some("7".to_string()),
            ..DeviceEnv::default()
        };
        assert_eq!(env.visible_devices(), Some("7"));
        assert_eq!(DeviceEnv::default().visible_devices(), None);

        // An empty HIP_VISIBLE_DEVICES hides every device: the empty
        // enumeration produces the reference's no-GPU wording.
        let hidden = DeviceEnv {
            hip_visible_devices: Some(String::new()),
            ..DeviceEnv::default()
        };
        assert_eq!(
            select_device(&[], &hidden),
            Err(crate::ROCM_NO_GPU.to_string())
        );

        // HIP="1,2": HIP makes physical devices 1 and 2 visible,
        // remapping them to HIP indices 0 and 1; the fixture lists the
        // devices with their physical indices and the default request
        // (`first`) selects the first entry.
        let devices = vec![device(1, "gfx1151"), device(2, "gfx9 gfx")];
        let env = DeviceEnv {
            hip_visible_devices: Some("1,2".to_string()),
            ..DeviceEnv::default()
        };
        assert_eq!(select_device(&devices, &env).unwrap().device.index, 1);
    }

    #[test]
    fn resolve_request_accepts_number_first_and_auto() {
        let devices = vec![device(0, "gfx9 gfx"), device(1, "gfx1151")];
        assert_eq!(resolve_request(&devices, "0").unwrap(), devices[0]);
        assert_eq!(resolve_request(&devices, "1").unwrap(), devices[1]);
        assert_eq!(resolve_request(&devices, "first").unwrap(), devices[0]);
        assert_eq!(resolve_request(&devices, "auto").unwrap(), devices[1]);
        // `auto` with no Strix Halo GPU.
        let only_discrete = vec![device(0, "gfx9 gfx")];
        assert_eq!(
            resolve_request(&only_discrete, "auto"),
            Err("halogen: ROCm offers no Strix Halo GPU (0 (gfx9 gfx))".to_string())
        );
        // An unresolvable request carries the reference's full wording.
        assert_eq!(
            resolve_request(&devices, "cat"),
            Err("halogen: HALOGEN_GPU_DEVICE=\"cat\" is not a device number. It takes one HIP device number (0, 1, ...), or `first` for device 0, or `auto` for the Strix Halo GPU. Devices ROCm offers: 0 (gfx9 gfx), 1 (gfx1151)".to_string())
        );
        assert_eq!(
            resolve_request(&devices, "7"),
            Err("halogen: HALOGEN_GPU_DEVICE=\"7\" is not a device number. It takes one HIP device number (0, 1, ...), or `first` for device 0, or `auto` for the Strix Halo GPU. Devices ROCm offers: 0 (gfx9 gfx), 1 (gfx1151)".to_string())
        );
    }

    #[test]
    fn the_gfx1151_reexec_rewrite_rule() {
        // All four visibility variables unset, more than one device and
        // the selected device is the Strix Halo GPU → the reference
        // re-execs with HIP_VISIBLE_DEVICES pinned to the device index.
        let devices = vec![device(0, "gfx1151"), device(1, "gfx9 gfx")];
        let selection = select_device(&devices, &DeviceEnv::default()).unwrap();
        assert_eq!(selection.visible_devices_rewrite, Some(0));
        // A non-Strix selection, a single device, or any visibility
        // variable set suppresses the rewrite.
        let env = DeviceEnv {
            halogen_gpu_device: Some("1".to_string()),
            ..DeviceEnv::default()
        };
        assert_eq!(
            select_device(&devices, &env)
                .unwrap()
                .visible_devices_rewrite,
            None
        );
        assert_eq!(
            select_device(&[device(0, "gfx1151")], &DeviceEnv::default())
                .unwrap()
                .visible_devices_rewrite,
            None
        );
        let env = DeviceEnv {
            rocr_visible_devices: Some("0".to_string()),
            ..DeviceEnv::default()
        };
        assert_eq!(
            select_device(&devices, &env)
                .unwrap()
                .visible_devices_rewrite,
            None
        );
    }

    #[test]
    fn bind_device_reports_set_device_failures_and_banners() {
        let devices = vec![device(0, "gfx1151"), device(1, "gfx9 gfx")];
        let env = DeviceEnv::default();
        // Successful bind: the banner matches the reference's wording.
        let selection = bind_device(&devices, &env, |_| Ok(())).unwrap();
        assert_eq!(
            selection.banner(),
            "halogen: GPU device 0: gfx1151, the Strix Halo GPU."
        );
        let discrete = DeviceEnv {
            halogen_gpu_device: Some("1".to_string()),
            ..DeviceEnv::default()
        };
        let selection = bind_device(&devices, &discrete, |_| Ok(())).unwrap();
        assert_eq!(selection.banner(), "halogen: GPU device 1: gfx9 gfx");
        // Failed bind: the reference's `cannot use GPU device %d.`.
        assert_eq!(
            bind_device(&devices, &env, |_| Err(1)),
            Err("halogen: cannot use GPU device 0.".to_string())
        );
    }

    #[test]
    fn backend_selection_falls_back_to_cpu_without_the_runtime() {
        assert_eq!(select_backend(false), Backend::Cpu);
        assert_eq!(select_backend(true), Backend::Hip);
        assert_eq!(Backend::Cpu.label(), "cpu");
        assert_eq!(Backend::Hip.label(), "hip");
        // `active` agrees with the probe in both environments (a nix
        // build with rocmPackages.clr in buildInputs resolves the
        // runtime; a plain host run reports Cpu).
        assert_eq!(active(), select_backend(runtime().is_some()));
    }

    #[test]
    fn device_env_from_env_reads_the_five_variables() {
        let _guard = crate::ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        env::set_var("HIP_VISIBLE_DEVICES", "4");
        env::set_var("ROCR_VISIBLE_DEVICES", "3");
        env::set_var("CUDA_VISIBLE_DEVICES", "2");
        env::set_var("GPU_DEVICE_ORDINAL", "1");
        env::set_var("HALOGEN_GPU_DEVICE", "first");
        let env = DeviceEnv::from_env();
        assert_eq!(env.hip_visible_devices.as_deref(), Some("4"));
        assert_eq!(env.rocr_visible_devices.as_deref(), Some("3"));
        assert_eq!(env.cuda_visible_devices.as_deref(), Some("2"));
        assert_eq!(env.gpu_device_ordinal.as_deref(), Some("1"));
        assert_eq!(env.halogen_gpu_device.as_deref(), Some("first"));
        for name in [
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
            "GPU_DEVICE_ORDINAL",
            "HALOGEN_GPU_DEVICE",
        ] {
            env::remove_var(name);
        }
        assert_eq!(DeviceEnv::from_env(), DeviceEnv::default());
    }

    #[test]
    fn slot_record_bytes_matches_the_kvcache_record() {
        let slot = SlotRecord::default();
        let expected = slot.activation_staging as u64
            + slot.secondary_staging as u64
            + slot.small_control as u64
            + slot.state_small as u64
            + slot.state_main as u64
            + slot.kv_region as u64
            + slot.alt_small_control as u64
            + slot.alt_state_block as u64;
        assert_eq!(slot_record_bytes(), expected);
        let mut account = DeviceAllocAccount::default();
        assert_eq!(account_slot_record(&mut account), expected);
        assert_eq!(account.total_bytes, expected);
        assert_eq!(account.count, 8);
    }

    #[test]
    fn dmalloc_log_threshold_parsing() {
        assert_eq!(dmalloc_log_threshold(None), 0);
        assert_eq!(dmalloc_log_threshold(Some("0")), 0);
        assert_eq!(dmalloc_log_threshold(Some("1")), 64 * 1_048_576);
        assert_eq!(dmalloc_log_threshold(Some("0x1000")), 4096);
        assert_eq!(dmalloc_log_threshold(Some("8388608")), 8_388_608);
        assert_eq!(dmalloc_log_threshold(Some("garbage")), 0);
        // The logging rule: a set threshold logs allocations that reach
        // it.
        assert!(!should_log(0, u64::MAX));
        assert!(should_log(64 * 1_048_576, 64 * 1_048_576));
        assert!(!should_log(64 * 1_048_576, 64 * 1_048_576 - 1));
    }

    #[test]
    fn dmalloc_accounting_lines_match_the_reference_formats() {
        let mut account = DeviceAllocAccount::default();
        account.record(0x30_0000);
        assert_eq!(
            account.success_line(0x30_0000),
            "dmalloc:      3.000 MiB  (total    0.003 GiB, #1)"
        );
        let report = account.failure_report(0xf_2800, "hipErrorOutOfMemory");
        assert!(report.starts_with(
            "dmalloc: FAILED requesting 0.001 GiB after 0.003 GiB in 1 allocations (hipErrorOutOfMemory)"
        ));
        assert!(report.lines().last().is_some_and(
            |line| line == "HIP /src/halogen/src/flash_ops.h:161: hipErrorOutOfMemory"
        ));
    }

    #[test]
    fn dmalloc_with_updates_the_account_only_on_success() {
        // Success below the threshold: no line, account updated.
        let mut account = DeviceAllocAccount::default();
        assert_eq!(
            dmalloc_with(&mut account, 16, 64 * 1_048_576, |_| Ok(())),
            Ok(None)
        );
        assert_eq!((account.total_bytes, account.count), (16, 1));
        // Success at the threshold: the log line.
        assert_eq!(
            dmalloc_with(&mut account, 64 * 1_048_576, 64 * 1_048_576, |_| Ok(())),
            Ok(Some(
                "dmalloc:     64.000 MiB  (total    0.063 GiB, #2)".to_string()
            ))
        );
        // Failure: the report, account untouched.
        let before = account;
        let failure = dmalloc_with(&mut account, 16, 0, |_| {
            Err("hipErrorOutOfMemory".to_string())
        });
        assert!(failure.is_err());
        assert_eq!(account, before);
    }
}
