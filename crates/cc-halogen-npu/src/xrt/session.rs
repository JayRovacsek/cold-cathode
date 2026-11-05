//! The typed device/session lifecycle port over the [`crate::xrt::ffi`]
//! surface, mirroring the catalogue's lifecycle semantics (spec
//! [`flash_serve-mtp-kv-npu.md`](../../../../docs/analysis/specs/flash_serve-mtp-kv-npu.md)
//! §3.2): `xrt::device(index)` → `xrt::elf(path)` → `xrt::hw_context(dev,
//! elf)` → `xrt::ext::kernel(ctx, name)` → `xrt::run(kernel)`, with
//! `run::set_arg_at_index` / `run::start` / `run::wait` / `run::state` and
//! `xrt::bo` buffer objects (create / map / sync, plus offset-sliced
//! sub-buffers).
//!
//! Every stage is a distinct Rust type produced only by the preceding
//! stage (`Device::load_elf` → `Elf`, `Elf::create_context` →
//! `HwContext`, …), so an illegal transition (starting a run twice,
//! waiting on an unstarted run, slicing a sub-buffer out of its parent)
//! is a runtime error at the exact catalogue-defined point, and the
//! borrow checker prevents a child from outliving its parent.
//!
//! The operations run through the injectable [`Backend`] trait: the real
//! implementation is [`crate::xrt::ffi::XrtRuntime`] (dlopen-resolved FFI);
//! tests inject a fake to exercise every state transition and error path
//! without a device.

use std::collections::BTreeMap;
use std::ffi::{c_void, CString};
use std::fmt;

#[cfg(test)]
use std::cell::{Cell, RefCell};

use crate::xrt::ffi::XrtRuntime;

/// An opaque XRT object handle (the `this` pointer of the C++ object the
/// backend constructed).
pub type Handle = *mut c_void;

/// A failed XRT operation: the catalogue entry point that failed plus the
/// backend's message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XrtError {
    pub stage: &'static str,
    pub message: String,
}

impl XrtError {
    pub(crate) fn new(stage: &'static str, message: impl Into<String>) -> Self {
        Self {
            stage,
            message: message.into(),
        }
    }
}

impl fmt::Display for XrtError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "xrt: {}: {}", self.stage, self.message)
    }
}

impl std::error::Error for XrtError {}

pub type XrtResult<T> = Result<T, XrtError>;

/// The `xrt::bo::flags` parameter type. The catalogue types the constructor
/// argument but the evidence does not enumerate the set; the port models it
/// as a newtype over the raw `unsigned` with XRT's default value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoFlags(pub u32);

impl BoFlags {
    /// XRT's default (cacheable) flag value for `xrt::bo::flags`.
    pub const NORMAL: Self = Self(0);
}

/// The `xclBOSyncDirection` parameter of `xrt::bo::sync` (the catalogue
/// names the type; the numeric values follow XRT's `xclhal2.h`, which the
/// local evidence does not enumerate).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SyncDirection {
    /// `XCL_BO_SYNC_BO_TO_DEVICE` (3): host → device.
    HostToDevice,
    /// `XCL_BO_SYNC_BO_FROM_DEVICE` (4): device → host.
    DeviceToHost,
}

impl SyncDirection {
    /// The raw enumerator value passed to `xrt::bo::sync`.
    pub fn as_raw(self) -> u32 {
        match self {
            SyncDirection::HostToDevice => 3,
            SyncDirection::DeviceToHost => 4,
        }
    }
}

/// The `xrt::kernel::state` value returned by `xrt::run::state()`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunState {
    /// `running` (0): the run is submitted or executing.
    Running,
    /// `completed` (1): the run has finished.
    Completed,
}

impl RunState {
    /// The raw enumerator value.
    pub fn as_raw(self) -> u32 {
        match self {
            RunState::Running => 0,
            RunState::Completed => 1,
        }
    }

    /// Decodes a raw value; `None` for values outside the catalogue.
    pub fn from_raw(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(RunState::Running),
            1 => Some(RunState::Completed),
            _ => None,
        }
    }
}

/// The XRT operations the typed lifecycle performs. The real implementation
/// is [`XrtRuntime`] (the dlopen-resolved FFI table); tests inject a fake.
/// The method set mirrors the catalogue's callable surface — the
/// never-invoked `device_get_info_std` is FFI-only ([`crate::xrt::ffi`]
/// declares it, the typed lifecycle never calls it) — including the
/// catalogue's omission of an `xrt::elf` destructor (the reference never
/// destroys the elf object, so [`Elf`] has no close).
pub trait Backend {
    fn device_open(&self, index: u32) -> XrtResult<Handle>;
    fn device_close(&self, handle: Handle);
    fn elf_load(&self, device: Handle, path: &str) -> XrtResult<Handle>;
    fn hw_context_create(&self, device: Handle, elf: Handle) -> XrtResult<Handle>;
    fn hw_context_close(&self, handle: Handle);
    fn kernel_bind(&self, context: Handle, name: &str) -> XrtResult<Handle>;
    fn kernel_close(&self, handle: Handle);
    fn run_create(&self, kernel: Handle) -> XrtResult<Handle>;
    fn run_close(&self, handle: Handle);
    fn run_set_arg(&self, run: Handle, index: i32, bo: Handle) -> XrtResult<()>;
    fn run_start(&self, run: Handle) -> XrtResult<()>;
    fn run_wait(&self, run: Handle, timeout_ms: u64) -> XrtResult<()>;
    fn run_state(&self, run: Handle) -> XrtResult<RunState>;
    fn bo_create(&self, device: Handle, size: u64, flags: BoFlags, bank: u32) -> XrtResult<Handle>;
    fn bo_slice(&self, parent: Handle, offset: u64, size: u64) -> XrtResult<Handle>;
    fn bo_close(&self, handle: Handle);
    fn bo_map(&self, bo: Handle) -> XrtResult<*mut c_void>;
    fn bo_sync(
        &self,
        bo: Handle,
        direction: SyncDirection,
        size: u64,
        offset: u64,
    ) -> XrtResult<()>;
    fn bo_size(&self, bo: Handle) -> XrtResult<u64>;
}

// ---------------------------------------------------------------------------
// The real backend: the dlopen-resolved FFI table
// ---------------------------------------------------------------------------

/// XRT C++ object storage as the port models it: an 8-aligned 64-byte heap
/// slot handed to the constructor as `this` and reclaimed after the
/// destructor. Against the reference stub (the only `libxrt_coreutil.so.2`
/// available at the nixpkgs pin) the constructors are no-ops, so the slot
/// content is never read; against a real XRT the slot sizes would need the
/// XRT object layout, which the frozen evidence does not record (the
/// spec's existing "XRT-dependent behaviour unverified" limitation).
/// Per-type sizing note: XRT's handle classes are pimpl-only — 8 bytes for
/// the pimpl pointer, at most 16 with inline scalar members — so the fixed
/// 64-byte slot is a conservative over-allocation, not a measured layout.
#[repr(align(8))]
#[allow(dead_code)] // the field defines the slot's size/alignment; the
                    // C++ object bytes are written through the FFI, never read
struct Slot(std::mem::MaybeUninit<[u8; 64]>);

fn alloc_slot() -> Handle {
    Box::into_raw(Box::new(Slot(std::mem::MaybeUninit::uninit()))).cast()
}

/// Reclaims a slot produced by [`alloc_slot`] (the caller must have run the
/// matching destructor first).
///
/// # Safety
/// `handle` must come from [`alloc_slot`] and its C++ destructor must have
/// been called exactly once.
unsafe fn free_slot(handle: Handle) {
    unsafe {
        drop(Box::from_raw(handle.cast::<Slot>()));
    }
}

// The handles this impl receives always come from its own constructors (the
// typed wrappers below enforce it), so the raw-pointer dereference contract
// the lint checks is upheld by construction.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
impl Backend for XrtRuntime {
    fn device_open(&self, index: u32) -> XrtResult<Handle> {
        let handle = alloc_slot();
        unsafe { (self.device_ctor)(handle, index) };
        Ok(handle)
    }

    fn device_close(&self, handle: Handle) {
        unsafe {
            (self.device_dtor)(handle);
            free_slot(handle);
        }
    }

    fn elf_load(&self, _device: Handle, path: &str) -> XrtResult<Handle> {
        // `std::string const&` is modelled as a C string (the ffi.rs
        // signature-modelling note).
        let path =
            CString::new(path).map_err(|_| XrtError::new("elf_load", "path contains a nul"))?;
        let handle = alloc_slot();
        unsafe { (self.elf_ctor)(handle, path.as_ptr()) };
        Ok(handle)
    }

    fn hw_context_create(&self, device: Handle, elf: Handle) -> XrtResult<Handle> {
        let handle = alloc_slot();
        unsafe { (self.hw_context_ctor)(handle, device.cast_const(), elf.cast_const()) };
        Ok(handle)
    }

    fn hw_context_close(&self, handle: Handle) {
        unsafe {
            (self.hw_context_dtor)(handle);
            free_slot(handle);
        }
    }

    fn kernel_bind(&self, context: Handle, name: &str) -> XrtResult<Handle> {
        let name =
            CString::new(name).map_err(|_| XrtError::new("kernel_bind", "name contains a nul"))?;
        let handle = alloc_slot();
        unsafe { (self.ext_kernel_ctor)(handle, context.cast_const(), name.as_ptr()) };
        Ok(handle)
    }

    fn kernel_close(&self, handle: Handle) {
        unsafe {
            (self.kernel_dtor)(handle);
            free_slot(handle);
        }
    }

    fn run_create(&self, kernel: Handle) -> XrtResult<Handle> {
        let handle = alloc_slot();
        unsafe { (self.run_ctor)(handle, kernel.cast_const()) };
        Ok(handle)
    }

    fn run_close(&self, handle: Handle) {
        unsafe {
            (self.run_dtor)(handle);
            free_slot(handle);
        }
    }

    fn run_set_arg(&self, run: Handle, index: i32, bo: Handle) -> XrtResult<()> {
        unsafe { (self.run_set_arg_at_index)(run, index, bo.cast_const()) };
        Ok(())
    }

    fn run_start(&self, run: Handle) -> XrtResult<()> {
        unsafe { (self.run_start)(run) };
        Ok(())
    }

    fn run_wait(&self, run: Handle, timeout_ms: u64) -> XrtResult<()> {
        // The duration parameter is a const reference to the i64
        // milliseconds value (the ffi.rs signature-modelling note).
        let timeout_ms = i64::try_from(timeout_ms).unwrap_or(i64::MAX);
        unsafe { (self.run_wait)(run.cast_const(), &timeout_ms) };
        Ok(())
    }

    fn run_state(&self, run: Handle) -> XrtResult<RunState> {
        let raw = unsafe { (self.run_state)(run.cast_const()) };
        RunState::from_raw(raw)
            .ok_or_else(|| XrtError::new("run_state", format!("unknown state value {raw}")))
    }

    fn bo_create(&self, device: Handle, size: u64, flags: BoFlags, bank: u32) -> XrtResult<Handle> {
        let handle = alloc_slot();
        unsafe { (self.bo_ctor_device)(handle, device.cast_const(), size, flags.0, bank) };
        Ok(handle)
    }

    fn bo_slice(&self, parent: Handle, offset: u64, size: u64) -> XrtResult<Handle> {
        let handle = alloc_slot();
        unsafe { (self.bo_ctor_sub)(handle, parent.cast_const(), offset, size) };
        Ok(handle)
    }

    fn bo_close(&self, handle: Handle) {
        unsafe {
            (self.bo_dtor)(handle);
            free_slot(handle);
        }
    }

    fn bo_map(&self, bo: Handle) -> XrtResult<*mut c_void> {
        let ptr = unsafe { (self.bo_map)(bo) };
        if ptr.is_null() {
            return Err(XrtError::new("bo_map", "mapping returned null"));
        }
        Ok(ptr)
    }

    fn bo_sync(
        &self,
        bo: Handle,
        direction: SyncDirection,
        size: u64,
        offset: u64,
    ) -> XrtResult<()> {
        unsafe { (self.bo_sync)(bo, direction.as_raw(), size, offset) };
        Ok(())
    }

    fn bo_size(&self, bo: Handle) -> XrtResult<u64> {
        Ok(unsafe { (self.bo_size)(bo.cast_const()) })
    }
}

// ---------------------------------------------------------------------------
// The typed lifecycle: Device → Elf → HwContext → Kernel → Run, and Bo
// ---------------------------------------------------------------------------

/// `xrt::device(index)` — an opened device ([`crate::xrt::ffi::XRT_SYMBOLS`]
/// entries 1–3). Dropping it runs `~device()`.
pub struct Device<'a> {
    backend: &'a dyn Backend,
    handle: Handle,
    index: u32,
}

impl<'a> Device<'a> {
    /// `xrt::device::device(unsigned int)` — opens device `index`.
    pub fn open(index: u32, backend: &'a dyn Backend) -> XrtResult<Self> {
        Ok(Self {
            handle: backend.device_open(index)?,
            backend,
            index,
        })
    }

    pub fn index(&self) -> u32 {
        self.index
    }

    pub fn handle(&self) -> Handle {
        self.handle
    }

    /// `xrt::elf::elf(path)` — loads the kernel ELF, keeping the device
    /// borrowed so the elf cannot outlive it.
    pub fn load_elf<'s>(&'s self, path: &str) -> XrtResult<Elf<'s>> {
        Ok(Elf {
            backend: self.backend,
            device: self.handle,
            path: path.to_string(),
            handle: self.backend.elf_load(self.handle, path)?,
        })
    }

    /// `xrt::bo::bo(device, size, flags, bank)` — allocates a buffer object
    /// on this device.
    pub fn alloc_bo<'s>(&'s self, size: u64, flags: BoFlags, bank: u32) -> XrtResult<Bo<'s>> {
        Ok(Bo {
            backend: self.backend,
            handle: self.backend.bo_create(self.handle, size, flags, bank)?,
            size,
            offset: 0,
        })
    }
}

impl Drop for Device<'_> {
    fn drop(&mut self) {
        self.backend.device_close(self.handle);
    }
}

/// `xrt::elf::elf(path)` — a loaded kernel ELF (catalogue entry 4). The
/// catalogue imports no `xrt::elf` destructor — the reference never
/// destroys the elf object — so this type has no `Drop`.
pub struct Elf<'a> {
    backend: &'a dyn Backend,
    device: Handle,
    path: String,
    handle: Handle,
}

impl<'a> Elf<'a> {
    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn handle(&self) -> Handle {
        self.handle
    }

    /// `xrt::hw_context::hw_context(device, elf)` — creates the hardware
    /// context for this ELF.
    pub fn create_context<'s>(&'s self) -> XrtResult<HwContext<'s>> {
        Ok(HwContext {
            backend: self.backend,
            handle: self.backend.hw_context_create(self.device, self.handle)?,
        })
    }
}

/// `xrt::hw_context(device, elf)` — a hardware context (catalogue entries
/// 5–6). Dropping it runs `~hw_context()`.
pub struct HwContext<'a> {
    backend: &'a dyn Backend,
    handle: Handle,
}

impl<'a> HwContext<'a> {
    pub fn handle(&self) -> Handle {
        self.handle
    }

    /// `xrt::ext::kernel::kernel(context, name)` — binds a kernel by name.
    pub fn kernel<'s>(&'s self, name: &str) -> XrtResult<Kernel<'s>> {
        Ok(Kernel {
            backend: self.backend,
            context: self.handle,
            name: name.to_string(),
            handle: self.backend.kernel_bind(self.handle, name)?,
        })
    }
}

impl Drop for HwContext<'_> {
    fn drop(&mut self) {
        self.backend.hw_context_close(self.handle);
    }
}

/// `xrt::ext::kernel(context, name)` — a bound kernel (catalogue entries
/// 7–8). Dropping it runs `~kernel()`.
pub struct Kernel<'a> {
    backend: &'a dyn Backend,
    #[allow(dead_code)] // mirrors the catalogue: the run ctor keeps the
    // context's kernel alive; the borrow checker already
    // enforces it via `create_run`.
    context: Handle,
    name: String,
    handle: Handle,
}

impl<'a> Kernel<'a> {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn handle(&self) -> Handle {
        self.handle
    }

    /// `xrt::run::run(kernel)` — creates a run object for this kernel.
    pub fn create_run<'s>(&'s self) -> XrtResult<Run<'s>> {
        Ok(Run {
            backend: self.backend,
            handle: self.backend.run_create(self.handle)?,
            args: BTreeMap::new(),
            started: false,
            state: None,
        })
    }
}

impl Drop for Kernel<'_> {
    fn drop(&mut self) {
        self.backend.kernel_close(self.handle);
    }
}

/// `xrt::run(kernel)` — a run object (catalogue entries 9–14). The
/// launch-wrapper semantics of `FUN_001afec0` (bind every argument, then
/// `start()`, then `wait()`/`state()`) are enforced as state transitions:
/// `start()` requires at least one bound argument and cannot repeat;
/// `set_arg` is rejected once started; `wait`/`state` require a started
/// run. Dropping it runs `~run()`.
pub struct Run<'a> {
    backend: &'a dyn Backend,
    handle: Handle,
    /// The bound argument slots as raw bo handles. The stored handle is
    /// not lifetime-tied to the `Bo` it came from (storing `&'a Bo` would
    /// need extra lifetime parameters on `Run` and reject the common
    /// create-run-then-bind-argument order), so the handle dangles if the
    /// `Bo` drops before the run — currently benign because the map is
    /// only read via [`Run::bound_args`].
    args: BTreeMap<i32, Handle>,
    started: bool,
    state: Option<RunState>,
}

impl<'a> Run<'a> {
    pub fn handle(&self) -> Handle {
        self.handle
    }

    /// The number of argument slots currently bound (the launcher's
    /// bound-argument vector).
    pub fn bound_args(&self) -> usize {
        self.args.len()
    }

    /// `xrt::run::set_arg_at_index(index, bo)` — binds `bo` to argument
    /// `index` (re-binding replaces the previous buffer).
    pub fn set_arg(&mut self, index: i32, bo: &Bo<'_>) -> XrtResult<()> {
        if self.started {
            return Err(XrtError::new(
                "run_set_arg",
                "cannot bind arguments to a started run",
            ));
        }
        self.backend.run_set_arg(self.handle, index, bo.handle)?;
        self.args.insert(index, bo.handle);
        Ok(())
    }

    /// `xrt::run::start()` — launches the run. Requires at least one bound
    /// argument (the `FUN_001afec0` launcher binds every argument before
    /// calling `start()`) and cannot be called twice.
    pub fn start(&mut self) -> XrtResult<()> {
        if self.started {
            return Err(XrtError::new("run_start", "run already started"));
        }
        if self.args.is_empty() {
            return Err(XrtError::new(
                "run_start",
                "no arguments bound (the launcher binds every argument before start)",
            ));
        }
        self.backend.run_start(self.handle)?;
        self.started = true;
        self.state = Some(RunState::Running);
        Ok(())
    }

    /// `xrt::run::wait(timeout)` — waits up to `timeout_ms` for completion.
    pub fn wait(&self, timeout_ms: u64) -> XrtResult<()> {
        if !self.started {
            return Err(XrtError::new("run_wait", "run not started"));
        }
        self.backend.run_wait(self.handle, timeout_ms)
    }

    /// `xrt::run::state()` — polls the run state and caches it.
    pub fn state(&mut self) -> XrtResult<RunState> {
        if !self.started {
            return Err(XrtError::new("run_state", "run not started"));
        }
        let state = self.backend.run_state(self.handle)?;
        self.state = Some(state);
        Ok(state)
    }

    /// The state recorded by the last `state()`/`start()`, if any.
    pub fn cached_state(&self) -> Option<RunState> {
        self.state
    }
}

impl Drop for Run<'_> {
    fn drop(&mut self) {
        self.backend.run_close(self.handle);
    }
}

/// `xrt::bo` — a buffer object (catalogue entries 15–20). A root buffer is
/// created with [`Device::alloc_bo`]; [`Bo::slice`] produces offset-sliced
/// sub-buffers (the `xrt::bo(existing, offset, size)` constructor) for the
/// one-bo-per-tensor weight-blob carving the spec describes. Dropping it
/// runs `~bo()`.
pub struct Bo<'a> {
    backend: &'a dyn Backend,
    handle: Handle,
    size: u64,
    /// The absolute offset within the root allocation (0 for a root bo,
    /// `parent.offset + requested offset` for a sub-buffer).
    offset: u64,
}

impl<'a> Bo<'a> {
    pub fn handle(&self) -> Handle {
        self.handle
    }

    /// The size of this buffer (or sub-buffer) in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// The absolute offset of this buffer within the root allocation
    /// (0 for a root bo).
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// `xrt::bo::bo(parent, offset, size)` — an offset-sliced sub-buffer.
    /// The slice must fit inside the parent (`offset + size <= parent.size`,
    /// checked with `checked_add` against overflow).
    pub fn slice<'s>(&'s self, offset: u64, size: u64) -> XrtResult<Bo<'s>> {
        let end = offset
            .checked_add(size)
            .ok_or_else(|| XrtError::new("bo_slice", "offset + size overflows u64"))?;
        if end > self.size {
            return Err(XrtError::new(
                "bo_slice",
                format!(
                    "sub-buffer {offset}..{end} exceeds buffer size {}",
                    self.size
                ),
            ));
        }
        Ok(Bo {
            backend: self.backend,
            handle: self.backend.bo_slice(self.handle, offset, size)?,
            size,
            offset: self.offset + offset,
        })
    }

    /// `xrt::bo::map()` — the host mapping of this buffer.
    pub fn map(&self) -> XrtResult<*mut c_void> {
        self.backend.bo_map(self.handle)
    }

    /// `xrt::bo::sync(direction, size, offset)` — a partial transfer; the
    /// range must fit inside this buffer.
    pub fn sync(&self, direction: SyncDirection, size: u64, offset: u64) -> XrtResult<()> {
        let end = offset
            .checked_add(size)
            .ok_or_else(|| XrtError::new("bo_sync", "offset + size overflows u64"))?;
        if end > self.size {
            return Err(XrtError::new(
                "bo_sync",
                format!("sync {offset}..{end} exceeds buffer size {}", self.size),
            ));
        }
        self.backend.bo_sync(self.handle, direction, size, offset)
    }

    /// `sync(direction, self.size(), 0)` — transfer the whole buffer.
    pub fn sync_all(&self, direction: SyncDirection) -> XrtResult<()> {
        self.sync(direction, self.size, 0)
    }
}

impl Drop for Bo<'_> {
    fn drop(&mut self) {
        self.backend.bo_close(self.handle);
    }
}

// ---------------------------------------------------------------------------
// The injectable fake backend (test seam)
// ---------------------------------------------------------------------------

/// A fake [`Backend`] that records every call and can inject failures at
/// any stage, so the typed lifecycle's state transitions are testable
/// without a device (the `cc-flash-serve` `hip.rs` injectable-closure
/// pattern, widened to the whole catalogue surface).
#[cfg(test)]
#[derive(Default)]
pub(crate) struct Fake {
    fail_stage: Option<&'static str>,
    calls: RefCell<Vec<String>>,
    next_handle: Cell<u64>,
    state: Cell<Option<RunState>>,
}

#[cfg(test)]
impl Fake {
    /// A fake where every operation succeeds.
    pub(crate) fn ok() -> Self {
        Self::default()
    }

    /// A fake where the operation named `stage` fails (every other
    /// operation succeeds).
    pub(crate) fn failing(stage: &'static str) -> Self {
        Self {
            fail_stage: Some(stage),
            ..Self::default()
        }
    }

    /// The recorded call log (e.g. `["device_open(0)", "elf_load(k.elf)"]`).
    pub(crate) fn calls(&self) -> Vec<String> {
        self.calls.borrow().clone()
    }

    /// The state `run_state` reports (default `Running`).
    pub(crate) fn set_state(&self, state: RunState) {
        self.state.set(Some(state));
    }

    /// Records `call` and applies the injected failure for `stage`.
    fn record(&self, stage: &'static str, call: String) -> XrtResult<()> {
        self.calls.borrow_mut().push(call);
        if self.fail_stage == Some(stage) {
            return Err(XrtError::new(stage, "injected failure"));
        }
        Ok(())
    }

    /// The next synthetic handle (never null).
    fn fresh_handle(&self) -> Handle {
        self.next_handle.set(self.next_handle.get() + 1);
        self.next_handle.get() as *mut c_void
    }
}

#[cfg(test)]
impl Backend for Fake {
    fn device_open(&self, index: u32) -> XrtResult<Handle> {
        self.record("device_open", format!("device_open({index})"))?;
        Ok(self.fresh_handle())
    }

    fn device_close(&self, handle: Handle) {
        // The close methods cannot fail in the fake: `record` only fails for
        // constructor stages (the trait's close signatures are infallible).
        let _ = self.record("device_close", format!("device_close({handle:p})"));
    }

    fn elf_load(&self, _device: Handle, path: &str) -> XrtResult<Handle> {
        self.record("elf_load", format!("elf_load({path})"))?;
        Ok(self.fresh_handle())
    }

    fn hw_context_create(&self, _device: Handle, _elf: Handle) -> XrtResult<Handle> {
        self.record("hw_context_create", "hw_context_create".to_string())?;
        Ok(self.fresh_handle())
    }

    fn hw_context_close(&self, handle: Handle) {
        let _ = self.record("hw_context_close", format!("hw_context_close({handle:p})"));
    }

    fn kernel_bind(&self, _context: Handle, name: &str) -> XrtResult<Handle> {
        self.record("kernel_bind", format!("kernel_bind({name})"))?;
        Ok(self.fresh_handle())
    }

    fn kernel_close(&self, handle: Handle) {
        let _ = self.record("kernel_close", format!("kernel_close({handle:p})"));
    }

    fn run_create(&self, _kernel: Handle) -> XrtResult<Handle> {
        self.record("run_create", "run_create".to_string())?;
        Ok(self.fresh_handle())
    }

    fn run_close(&self, handle: Handle) {
        let _ = self.record("run_close", format!("run_close({handle:p})"));
    }

    fn run_set_arg(&self, _run: Handle, index: i32, _bo: Handle) -> XrtResult<()> {
        self.record("run_set_arg", format!("run_set_arg({index})"))
    }

    fn run_start(&self, _run: Handle) -> XrtResult<()> {
        self.record("run_start", "run_start".to_string())
    }

    fn run_wait(&self, _run: Handle, timeout_ms: u64) -> XrtResult<()> {
        self.record("run_wait", format!("run_wait({timeout_ms})"))
    }

    fn run_state(&self, _run: Handle) -> XrtResult<RunState> {
        self.record("run_state", "run_state".to_string())?;
        Ok(self.state.get().unwrap_or(RunState::Running))
    }

    fn bo_create(
        &self,
        _device: Handle,
        size: u64,
        _flags: BoFlags,
        _bank: u32,
    ) -> XrtResult<Handle> {
        self.record("bo_create", format!("bo_create({size})"))?;
        Ok(self.fresh_handle())
    }

    fn bo_slice(&self, _parent: Handle, offset: u64, size: u64) -> XrtResult<Handle> {
        self.record("bo_slice", format!("bo_slice({offset},{size})"))?;
        Ok(self.fresh_handle())
    }

    fn bo_close(&self, handle: Handle) {
        let _ = self.record("bo_close", format!("bo_close({handle:p})"));
    }

    fn bo_map(&self, _bo: Handle) -> XrtResult<*mut c_void> {
        self.record("bo_map", "bo_map".to_string())?;
        Ok(0x2000 as *mut c_void)
    }

    fn bo_sync(
        &self,
        _bo: Handle,
        direction: SyncDirection,
        size: u64,
        offset: u64,
    ) -> XrtResult<()> {
        self.record("bo_sync", format!("bo_sync({direction:?},{size},{offset})"))
    }

    fn bo_size(&self, _bo: Handle) -> XrtResult<u64> {
        self.record("bo_size", "bo_size".to_string())?;
        Ok(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifecycle_walks_the_catalogue_chain() {
        let fake = Fake::ok();
        // The scope ends before `calls` is read, so the destructors'
        // entries are part of the snapshot.
        {
            let device = Device::open(0, &fake).unwrap();
            let elf = device.load_elf("kernel.elf").unwrap();
            let context = elf.create_context().unwrap();
            let kernel = context.kernel("mlm").unwrap();
            let mut run = kernel.create_run().unwrap();
            let bo = device.alloc_bo(4096, BoFlags::NORMAL, 0).unwrap();
            run.set_arg(0, &bo).unwrap();
            run.set_arg(1, &bo).unwrap();
            assert_eq!(run.bound_args(), 2);
            run.start().unwrap();
            assert_eq!(run.state().unwrap(), RunState::Running);
            run.wait(1000).unwrap();
        }
        let calls = fake.calls();
        assert_eq!(
            &calls[..11],
            [
                "device_open(0)",
                "elf_load(kernel.elf)",
                "hw_context_create",
                "kernel_bind(mlm)",
                "run_create",
                "bo_create(4096)",
                "run_set_arg(0)",
                "run_set_arg(1)",
                "run_start",
                "run_state",
                "run_wait(1000)",
            ]
        );
        // Every destructor ran exactly once, in reverse declaration order
        // (the elf has no destructor — the catalogue imports none). The
        // handles are the fake's counter: device 0x1, elf 0x2, context 0x3,
        // kernel 0x4, run 0x5, bo 0x6.
        assert_eq!(
            &calls[11..],
            [
                "bo_close(0x6)",
                "run_close(0x5)",
                "kernel_close(0x4)",
                "hw_context_close(0x3)",
                "device_close(0x1)",
            ]
        );
    }

    #[test]
    fn run_start_requires_bound_arguments_and_rejects_double_start() {
        let fake = Fake::ok();
        {
            let device = Device::open(0, &fake).unwrap();
            let elf = device.load_elf("kernel.elf").unwrap();
            let context = elf.create_context().unwrap();
            let kernel = context.kernel("mlm").unwrap();
            let mut run = kernel.create_run().unwrap();
            assert_eq!(
                run.start(),
                Err(XrtError::new(
                    "run_start",
                    "no arguments bound (the launcher binds every argument before start)"
                ))
            );
            let bo = device.alloc_bo(16, BoFlags::NORMAL, 0).unwrap();
            run.set_arg(0, &bo).unwrap();
            run.start().unwrap();
            assert_eq!(
                run.start(),
                Err(XrtError::new("run_start", "run already started"))
            );
            assert_eq!(
                run.set_arg(1, &bo),
                Err(XrtError::new(
                    "run_set_arg",
                    "cannot bind arguments to a started run"
                ))
            );
        }
        // The fake saw exactly one run_start despite the second attempt.
        assert_eq!(
            fake.calls()
                .iter()
                .filter(|call| call.as_str() == "run_start")
                .count(),
            1
        );
    }

    #[test]
    fn wait_and_state_require_a_started_run() {
        let fake = Fake::ok();
        {
            let device = Device::open(0, &fake).unwrap();
            let elf = device.load_elf("kernel.elf").unwrap();
            let context = elf.create_context().unwrap();
            let kernel = context.kernel("mlm").unwrap();
            let mut run = kernel.create_run().unwrap();
            assert_eq!(
                run.wait(10),
                Err(XrtError::new("run_wait", "run not started"))
            );
            assert_eq!(
                run.state(),
                Err(XrtError::new("run_state", "run not started"))
            );
        }
    }

    #[test]
    fn run_state_reflects_the_backend() {
        let fake = Fake::ok();
        {
            let device = Device::open(0, &fake).unwrap();
            let elf = device.load_elf("kernel.elf").unwrap();
            let context = elf.create_context().unwrap();
            let kernel = context.kernel("mlm").unwrap();
            let mut run = kernel.create_run().unwrap();
            let bo = device.alloc_bo(16, BoFlags::NORMAL, 0).unwrap();
            run.set_arg(0, &bo).unwrap();
            run.start().unwrap();
            assert_eq!(run.cached_state(), Some(RunState::Running));
            fake.set_state(RunState::Completed);
            assert_eq!(run.state().unwrap(), RunState::Completed);
            assert_eq!(run.cached_state(), Some(RunState::Completed));
        }
    }

    #[test]
    fn every_stage_propagates_backend_failures() {
        // Each constructor/operation surfaces the backend error with its
        // catalogue stage; the chain still closes cleanly (Drop).
        {
            let fake = Fake::failing("device_open");
            assert_eq!(
                Device::open(0, &fake).err(),
                Some(XrtError::new("device_open", "injected failure"))
            );
        }
        {
            let fake = Fake::failing("elf_load");
            let device = Device::open(0, &fake).unwrap();
            assert_eq!(
                device.load_elf("kernel.elf").err(),
                Some(XrtError::new("elf_load", "injected failure"))
            );
        }
        {
            let fake = Fake::failing("hw_context_create");
            let device = Device::open(0, &fake).unwrap();
            let elf = device.load_elf("kernel.elf").unwrap();
            assert_eq!(
                elf.create_context().err(),
                Some(XrtError::new("hw_context_create", "injected failure"))
            );
        }
        {
            let fake = Fake::failing("kernel_bind");
            let device = Device::open(0, &fake).unwrap();
            let elf = device.load_elf("kernel.elf").unwrap();
            let context = elf.create_context().unwrap();
            assert_eq!(
                context.kernel("mlm").err(),
                Some(XrtError::new("kernel_bind", "injected failure"))
            );
        }
        {
            let fake = Fake::failing("run_create");
            let device = Device::open(0, &fake).unwrap();
            let elf = device.load_elf("kernel.elf").unwrap();
            let context = elf.create_context().unwrap();
            let kernel = context.kernel("mlm").unwrap();
            assert_eq!(
                kernel.create_run().err(),
                Some(XrtError::new("run_create", "injected failure"))
            );
        }
        {
            let fake = Fake::failing("bo_create");
            let device = Device::open(0, &fake).unwrap();
            assert_eq!(
                device.alloc_bo(16, BoFlags::NORMAL, 0).err(),
                Some(XrtError::new("bo_create", "injected failure"))
            );
        }
        {
            let fake = Fake::failing("run_set_arg");
            let device = Device::open(0, &fake).unwrap();
            let elf = device.load_elf("kernel.elf").unwrap();
            let context = elf.create_context().unwrap();
            let kernel = context.kernel("mlm").unwrap();
            let mut run = kernel.create_run().unwrap();
            let bo = device.alloc_bo(16, BoFlags::NORMAL, 0).unwrap();
            assert_eq!(
                run.set_arg(0, &bo),
                Err(XrtError::new("run_set_arg", "injected failure"))
            );
            // The failed binding was not recorded, so start still refuses.
            assert!(run.start().is_err());
        }
    }

    #[test]
    fn bo_sub_buffer_offset_math() {
        let fake = Fake::ok();
        let device = Device::open(0, &fake).unwrap();
        let weights = device.alloc_bo(0x10000, BoFlags::NORMAL, 0).unwrap();
        assert_eq!((weights.size(), weights.offset()), (0x10000, 0));

        // One bo per tensor, offset-sliced from the weight blob.
        let head = weights.slice(0x1000, 0x800).unwrap();
        assert_eq!((head.size(), head.offset()), (0x800, 0x1000));
        // Nested slicing composes: the absolute offset accumulates.
        let nested = head.slice(0x20, 0x10).unwrap();
        assert_eq!((nested.size(), nested.offset()), (0x10, 0x1020));

        // Bounds: the slice must fit inside its parent.
        assert_eq!(
            weights.slice(0x1000, 0xF001).err(),
            Some(XrtError::new(
                "bo_slice",
                "sub-buffer 4096..65537 exceeds buffer size 65536"
            ))
        );
        assert_eq!(
            weights.slice(0x10001, 0).err(),
            Some(XrtError::new(
                "bo_slice",
                "sub-buffer 65537..65537 exceeds buffer size 65536"
            ))
        );
        assert_eq!(
            weights.slice(u64::MAX, 1).err(),
            Some(XrtError::new("bo_slice", "offset + size overflows u64"))
        );

        // Transfers validate against the (sub-)buffer, not the root.
        assert_eq!(
            head.sync(SyncDirection::HostToDevice, 0x801, 0),
            Err(XrtError::new(
                "bo_sync",
                "sync 0..2049 exceeds buffer size 2048"
            ))
        );
        assert_eq!(
            head.sync(SyncDirection::HostToDevice, 0x100, 0x701),
            Err(XrtError::new(
                "bo_sync",
                "sync 1793..2049 exceeds buffer size 2048"
            ))
        );
        assert_eq!(head.sync(SyncDirection::HostToDevice, 0x800, 0), Ok(()));
        assert_eq!(head.sync_all(SyncDirection::DeviceToHost), Ok(()));
        assert_eq!(head.map(), Ok(0x2000 as *mut c_void));
    }

    #[test]
    fn run_state_decodes_only_catalogue_values() {
        assert_eq!(RunState::from_raw(0), Some(RunState::Running));
        assert_eq!(RunState::from_raw(1), Some(RunState::Completed));
        assert_eq!(RunState::from_raw(2), None);
        assert_eq!(RunState::Running.as_raw(), 0);
        assert_eq!(RunState::Completed.as_raw(), 1);
        assert_eq!(SyncDirection::HostToDevice.as_raw(), 3);
        assert_eq!(SyncDirection::DeviceToHost.as_raw(), 4);
    }
}
