//! AMD XRT integration for the `halogen-npu` engine paths (Phase 7).
//!
//! The reference binary `DT_NEEDED`s `libxrt_coreutil.so.2` (AMD XRT); the
//! complete import surface — 20 `xrt::` symbols + 2 `_ITM_*` hooks — is
//! frozen in
//! [npu/xrt-imported-symbols.json](../../../../docs/analysis/evidence/flash_serve/npu/xrt-imported-symbols.json).
//! This module ports that surface:
//!
//! * [`ffi`] — the hand-written `extern "C"` declarations (mangled
//!   `#[link_name]`s, mirroring `cc-flash-serve`'s `hip.rs` pattern) and
//!   the dlopen-style [`ffi::XrtRuntime`] probe.
//! * [`session`] — the typed lifecycle port (`Device → Elf → HwContext →
//!   Kernel → Run`, plus `Bo` buffer objects with offset-sliced
//!   sub-buffers) over an injectable [`session::Backend`] trait.
//!
//! **Runtime availability at the nixpkgs pin:** neither `xilinx-xrt` nor
//! `amd-xrt` is packaged in nixpkgs at the pinned revision
//! (`3ec636362e1816bc2f0888fa0fa2c69f1bc9e504`; verified via
//! `pkgs/by-name/{xi,am}`), matching the spec's own finding that the
//! library "is **not** present on this host" (the upstream derivation
//! ignores the missing dependency with `autoPatchelfIgnoreMissingDeps`).
//! The probe therefore reports [`ListenInit::NoDevice`] on this host and
//! the binary takes the documented no-device path; when a real (or stub)
//! `libxrt_coreutil.so.2` is on the loader search path, the probe resolves
//! the full symbol table and the typed lifecycle becomes usable.
//!
//! The reference's `listen` success output is GPU/XRT-gated and not
//! observable with the stub library (spec ambiguity (b)), so the port
//! performs the XRT init attempt silently and exits 0 on both paths.

pub mod ffi;
pub mod session;

use std::sync::OnceLock;

pub use ffi::XrtRuntime;
pub use session::{
    Backend, Bo, BoFlags, Device, Elf, HwContext, Kernel, Run, RunState, SyncDirection, XrtError,
    XrtResult,
};

/// The process-wide XRT runtime probe (dlopen once, cached — the
/// `cc-flash-serve` `hip.rs` pattern). `None` when `libxrt_coreutil.so.2`
/// cannot be loaded or any catalogued symbol is missing.
pub fn runtime() -> Option<&'static XrtRuntime> {
    static RUNTIME: OnceLock<Option<XrtRuntime>> = OnceLock::new();
    RUNTIME.get_or_init(XrtRuntime::load).as_ref()
}

/// Whether the XRT runtime is available to this process.
pub fn available() -> bool {
    runtime().is_some()
}

/// The outcome of the `listen` engine-initialisation attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ListenInit {
    /// XRT loaded and device 0 opened — the engine path.
    Xrt,
    /// XRT absent (or device 0 unopenable) — the documented no-device path
    /// (the reference exits 127 without the library; the port exits 0).
    NoDevice,
}

/// The `listen` engine init: attempt XRT init when the runtime is
/// available, falling back to the documented no-device behaviour
/// otherwise. The reference's `listen` success output is GPU/XRT-gated and
/// unobservable (spec ambiguity (b)), so nothing is printed on either
/// path; the port's contract output for a valid invocation stays empty
/// with exit 0.
pub fn listen_init() -> ListenInit {
    match runtime() {
        // The reference's engine opens device 0 (the `HALOGEN_NPU_DEV_SERIAL`
        // / devices.hnpm contract); a failed open degrades to the no-device
        // path rather than aborting, mirroring the port's GPU/XRT-gated
        // exit-0 policy.
        Some(runtime) if Device::open(0, runtime).is_ok() => ListenInit::Xrt,
        _ => ListenInit::NoDevice,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn listen_init_agrees_with_the_availability_probe() {
        // The invariant: `Xrt` only when the runtime probe succeeded (an
        // available runtime whose device 0 cannot be opened still degrades
        // to `NoDevice`).
        match (listen_init(), available()) {
            (ListenInit::Xrt, true) | (ListenInit::NoDevice, _) => {}
            (ListenInit::Xrt, false) => {
                panic!("listen_init reported Xrt while the runtime probe reports unavailable");
            }
        }
    }
}
