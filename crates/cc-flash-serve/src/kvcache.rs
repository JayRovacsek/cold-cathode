//! KV-cache management (spec `flash_serve-mtp-kv-npu.md` §2).
//!
//! The reference manages the KV cache as one device pool of N positions
//! carved into per-request slot regions ([`SlotPool`], ported from
//! `slots_alloc_pooled` / `FUN_020120b0`), sized by a host-RAM budget
//! ([`Budget`], ported from `FUN_02019610`), with per-position byte costs
//! that depend on the `HALOGEN_KV8` mode ([`Kv8Mode`], ported from
//! `FUN_01f6ff70` and the sizing function `FUN_01f82410`).
//!
//! The port stores the slot's K/V rows as `f32` (the shape `forward`
//! computes) rather than bf16/fp8 bytes: the byte math governs only the
//! budget and region sizing, keeping the port's logits byte-stable while
//! the reference's quantised KV codecs are unspecified (spec §1). The
//! reference's device-buffer fields of the 0x160-byte slot record are
//! ported as named fields recording the `FUN_01f82130` allocation sizes,
//! and the §2 request-object fields (spec_n, sequence length, position,
//! keep-flag ring, commit window, verify rings) are ported as the
//! bookkeeping fields of [`SlotRecord`].
//!
//! Rollback (spec §2 "Speculative-decode interaction": drafted rows whose
//! K/V are written speculatively and rolled back on rejection) is
//! implemented with per-position state snapshots ([`SlotRecord::prefixes`]
//! plus the [`SlotRecord::commit_window`] counter): a caller appends
//! positions, then either commits them ([`SlotRecord::commit`]) or discards
//! the last n ([`SlotRecord::rollback`]), restoring the exact prior state.

use std::env;
use std::fs;
use std::process;
use std::sync::OnceLock;

use crate::hgn;
use crate::model::{Block, Model};

/// One pool unit, in positions (spec §2: the pool is allocated in units of
/// 256; `lib.rs`'s `KV_POOL_UNIT` is the same constant on the serve path).
pub const POOL_UNIT: usize = 256;
/// The pool ceiling: `0x1000000` positions (spec §2, engine spec §5;
/// `lib.rs`'s `MAX_KV_POOL`).
pub const MAX_POOL_POSITIONS: usize = 0x0100_0000;
/// The maximum serving-slot count (`slots_alloc_pooled`'s `0x40` bound).
pub const MAX_SLOTS: usize = 64;
/// The `--max-tok` default the budget formula charges the prefill arena
/// against when the caller does not choose (the binary's own default).
const DEFAULT_MAX_TOK: usize = 8192;
/// The weights size the reference charges when the checkpoint reports no
/// resident weights (`0x4050eccccccccccd` as a double = 68.8 GiB).
const UNLISTED_WEIGHTS_GIB: f64 = 68.8;
/// The `HALOGEN_HOST_RESERVE_GIB` default (20 GiB for the n-gram page
/// cache and the OS).
const DEFAULT_HOST_RESERVE_GIB: f64 = 20.0;
/// Bytes in one GiB.
const GIB: f64 = 1073741824.0;
/// KiB in one GiB (`/proc/meminfo` reports KiB).
const KIB_PER_GIB: f64 = 1_048_576.0;

// ---------------------------------------------------------------------------
// KV8 mode (FUN_01f6ff70) and per-position sizing (FUN_01f82410)
// ---------------------------------------------------------------------------

/// The `HALOGEN_KV8` setting. The reference's parser accepts unset/empty
/// and `0` (off), `1` (the 8-bit KV cache) and `2` (a mode that exists in
/// the parser but which the sizing and allocator branches only ever
/// special-case as `1 == kv8`; mode 2 currently sizes like bf16, spec §2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kv8Mode {
    /// Unset, empty, or `0`: bf16 K/V rows.
    Off,
    /// `1`: the 8-bit KV cache.
    Kv8,
    /// `2`: parsed but sized like bf16 (see [`bytes_per_position`]).
    Extended,
}

/// Parse a `HALOGEN_KV8` value with the reference's grammar. The error
/// string is the reference's exact wording (`HALOGEN_KV8=%s: 1 (the 8-bit
/// KV cache) or unset`); any value other than empty, `0`, `1` or `2` —
/// including multi-character strings like `00` or `10` — is rejected.
pub fn parse_kv8(raw: Option<&str>) -> Result<Kv8Mode, String> {
    match raw {
        None | Some("") | Some("0") => Ok(Kv8Mode::Off),
        Some("1") => Ok(Kv8Mode::Kv8),
        Some("2") => Ok(Kv8Mode::Extended),
        Some(value) => Err(format!(
            "HALOGEN_KV8={value}: 1 (the 8-bit KV cache) or unset"
        )),
    }
}

/// The memoised `HALOGEN_KV8` mode (the reference memoises the parse with
/// `__cxa_guard` and exits 1 on an invalid value). The port memoises with
/// [`OnceLock`] and reproduces the exact stderr wording before exiting.
pub fn kv8_mode() -> Kv8Mode {
    static MODE: OnceLock<Kv8Mode> = OnceLock::new();
    *MODE.get_or_init(
        || match parse_kv8(env::var("HALOGEN_KV8").ok().as_deref()) {
            Ok(mode) => mode,
            Err(message) => {
                eprintln!("{message}");
                process::exit(1);
            }
        },
    )
}

/// The per-position K/V row sizes in bytes for a mode (the `K`/`V` of
/// `FUN_01f82410`): bf16 rows are `0x400` each (512 elements × bf16, the
/// 2 KV heads × 256 dim); KV8 rows are `K = 0x208` (512 B fp8 payload +
/// 8 B scales) and `V = 0x220` (512 B + 32 B scales). Mode 2 sizes like
/// bf16 (spec §2: the sizing branch only special-cases `1`).
pub fn kv_row_bytes(mode: Kv8Mode) -> (usize, usize) {
    match mode {
        Kv8Mode::Kv8 => (0x208, 0x220),
        Kv8Mode::Off | Kv8Mode::Extended => (0x400, 0x400),
    }
}

/// The bytes the engine reserves per position: `(K + V) · 13 + 0xb40`,
/// where the ×13 is the 12 main attention layers + 1 MTP layer (spec §2).
///
/// The decompile evaluates this to 29 504 B in bf16 mode (matching the
/// frozen spec's "about 29.5 KiB a position") and 16 712 B in KV8 mode —
/// the spec prose quotes 16 816 for KV8, an arithmetic slip in the prose
/// (the formula and the `0x208`/`0x220` row sizes give 16 712); the port
/// implements the decompile formula.
pub fn bytes_per_position(mode: Kv8Mode) -> usize {
    let (k, v) = kv_row_bytes(mode);
    (k + v) * 13 + 0xb40
}

// ---------------------------------------------------------------------------
// Pool budget (FUN_02019610)
// ---------------------------------------------------------------------------

/// The pool-budget inputs (ported from `FUN_02019610`). The reference
/// derives the device-side budget from host RAM less resident weights and
/// reserve, then charges the non-pool terms (per-slot O(1) state, the
/// n-gram page cache, the prefill arena, the kept trunk, the KV8 extra
/// rows, the vision tower and the speculative rows) before dividing by the
/// per-position cost.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Budget {
    /// Host RAM in GiB (the reference reads `MemTotal` from
    /// `/proc/meminfo`; [`host_ram_gib`] does the same).
    pub host_ram_gib: f64,
    /// Resident weights in GiB; 0 clamps to [`UNLISTED_WEIGHTS_GIB`] the
    /// way the reference clamps a checkpoint with no reported weights.
    pub weights_gib: f64,
    /// `HALOGEN_HOST_RESERVE_GIB` (default 20 GiB; negative values clamp
    /// to the default, as the reference's clamp does).
    pub host_reserve_gib: f64,
    /// The serving-slot count, charged 0.115 GiB each.
    pub slots: usize,
    /// Whether the n-gram page cache is on: the OS/n-gram charge is 8.8
    /// GiB on, 3.1 GiB off, and the prefill coefficient is 3.1 GiB on,
    /// 8.8 GiB off (the two terms swap between the branches).
    pub ngram_on: bool,
    /// `--max-tok`: the prefill arena is `max_tok × coefficient / 32768`
    /// GiB.
    pub max_tok: usize,
    /// `HALOGEN_PREFILL_KEEP_TRUNK=1` charges the unpacked trunk beside
    /// the pool (GiB).
    pub keep_trunk_gib: f64,
    /// The KV8 extra rows (GiB): the reference computes this from the
    /// n-gram row count (`FUN_01f35410`) only when `HALOGEN_KV8=1`; the
    /// port exposes it as an injectable field.
    pub kv8_extra_gib: f64,
    /// The vision tower charge (GiB): resident weights plus worst-case
    /// scratch at `HALOGEN_VISION_MAX_PIXELS` (the reference's
    /// `HALOGEN_VISION_TOWER` branch); injectable in the port.
    pub vision_gib: f64,
    /// The speculative-decode rows beyond the two default rows (GiB);
    /// injectable in the port.
    pub spec_rows_gib: f64,
}

impl Budget {
    /// Read the environment- and checkpoint-derived inputs: host RAM from
    /// `/proc/meminfo`, the reserve from `HALOGEN_HOST_RESERVE_GIB`, the
    /// weights from the checkpoint's table, with the optional charges
    /// defaulted to zero.
    pub fn from_checkpoint(checkpoint: &hgn::Checkpoint, slots: usize, max_tok: usize) -> Self {
        Budget {
            host_ram_gib: host_ram_gib(),
            weights_gib: checkpoint.weight_bytes as f64 / GIB,
            host_reserve_gib: host_reserve_gib(),
            slots,
            ngram_on: false,
            max_tok,
            ..Budget::default()
        }
    }

    /// Device bytes available: `max(0, host RAM − weights − reserve)` with
    /// the reference's zero-weights clamp.
    pub fn device_bytes(&self) -> u64 {
        let weights = if self.weights_gib > 0.0 {
            self.weights_gib
        } else {
            UNLISTED_WEIGHTS_GIB
        };
        ((self.host_ram_gib - weights - self.host_reserve_gib).max(0.0) * GIB) as u64
    }

    /// The non-pool charges in bytes (per-slot state, n-gram page cache,
    /// prefill arena, kept trunk, KV8 extra rows, vision tower, spec rows).
    pub fn non_pool_bytes(&self, mode: Kv8Mode) -> u64 {
        let (ngram_os, prefill_coeff) = if self.ngram_on {
            (8.8, 3.1)
        } else {
            (3.1, 8.8)
        };
        let mut gib = self.slots as f64 * 0.115
            + ngram_os
            + prefill_coeff * self.max_tok as f64 / 32768.0
            + self.keep_trunk_gib;
        // The reference adds the KV8 extra rows only in mode 1 (the
        // `DAT_02127900 == 1` branch); the port keeps the field injectable
        // but gates it the same way.
        if mode == Kv8Mode::Kv8 {
            gib += self.kv8_extra_gib;
        }
        gib += self.vision_gib + self.spec_rows_gib;
        (gib * GIB) as u64
    }

    /// The pool position count: the device bytes the non-pool charges leave
    /// over, divided by the per-position cost, capped at
    /// [`MAX_POOL_POSITIONS`] and rounded down to [`POOL_UNIT`] multiples.
    pub fn pool_positions(&self, mode: Kv8Mode) -> usize {
        let available = self
            .device_bytes()
            .saturating_sub(self.non_pool_bytes(mode));
        positions_from_bytes(available, mode)
    }
}

/// The pool size for `available` spare bytes: whole positions at
/// [`bytes_per_position`], capped at [`MAX_POOL_POSITIONS`] and rounded
/// down to 256-position units (the reference's `& 0xffffffffffffff00`).
pub fn positions_from_bytes(available_bytes: u64, mode: Kv8Mode) -> usize {
    let positions =
        (available_bytes / bytes_per_position(mode) as u64).min(MAX_POOL_POSITIONS as u64);
    positions as usize / POOL_UNIT * POOL_UNIT
}

/// `MemTotal` from `/proc/meminfo` in GiB (the reference reads the same
/// file); 0.0 when the file is absent or the line is unparseable.
pub fn host_ram_gib() -> f64 {
    let Ok(text) = fs::read_to_string("/proc/meminfo") else {
        return 0.0;
    };
    let Some(line) = text.lines().find(|line| line.starts_with("MemTotal:")) else {
        return 0.0;
    };
    // `sscanf "MemTotal: %lf kB"`: the first number after the label.
    let kib = line
        .split_whitespace()
        .nth(1)
        .and_then(crate::parse_float_strict)
        .unwrap_or(0.0);
    kib / KIB_PER_GIB
}

/// `HALOGEN_HOST_RESERVE_GIB` with the reference's semantics: unset → 20,
/// a negative value → 20, otherwise the parsed value.
pub fn host_reserve_gib() -> f64 {
    match env::var("HALOGEN_HOST_RESERVE_GIB") {
        Ok(raw) => {
            let value = crate::parse_float_strict(&raw).unwrap_or(0.0);
            if value < 0.0 {
                DEFAULT_HOST_RESERVE_GIB
            } else {
                value
            }
        }
        Err(_) => DEFAULT_HOST_RESERVE_GIB,
    }
}

// ---------------------------------------------------------------------------
// Slot record (the 0x160-byte record of FUN_020120b0 + the §2 request object)
// ---------------------------------------------------------------------------

// The `FUN_01f82130` device-buffer sizes of the 0x160-byte slot record,
// with the record offsets from the decompiled stack frame.
const ACTIVATION_STAGING: usize = 0x30_0000; // +0x000, 3 MiB
const SECONDARY_STAGING: usize = 0x1_4000; // +0x018, 80 KiB
const SMALL_CONTROL: usize = 0x300; // +0x078, 768 B
const STATE_SMALL: usize = 0x10; // +0x090, 16 B
const STATE_MAIN: usize = 0x2_d000; // +0x098, 184 KiB
const KV_REGION: usize = 0xf_2800; // +0x0b8, 993 280 B
const ALT_SMALL_CONTROL: usize = 0x300; // +0x0d0, 20 KiB block pair
const ALT_STATE_BLOCK: usize = 0x5000; // +0x0e0

/// Per-layer recurrent state: the accumulated (post-RoPE) keys/values for
/// full-attention layers, the conv1d history and delta-rule states for
/// linear-attention layers (moved from `forward`, where the token step
/// fills it).
#[derive(Debug, Clone, PartialEq)]
pub struct LayerRuntime {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub conv_hist: Vec<f32>,
    pub delta: Vec<f32>,
}

/// One serving slot. The reference's slot record is 0x160 bytes of device
/// buffer pointers (the `FUN_01f82130` allocations, ported as the size
/// fields below) plus the §2 request-object fields (spec_n, the KV ring
/// pointers, sequence length, max-seq bound, position, keep-flag ring,
/// ring counters, commit window, MTP staging and verify rings). The port's
/// decode state (the context tokens, the per-layer K/V rows, the last
/// logits and the prefix snapshots) lives alongside them; the reference
/// keeps the K/V bytes in the `kv_region` device buffer, the port keeps
/// them as `f32` rows (module docs).
#[derive(Debug, Clone)]
pub struct SlotRecord {
    // --- the 0x160-byte device-buffer record (FUN_020120b0) ---
    /// +0x000 — activation/logit staging block (`0x300000` B).
    pub activation_staging: usize,
    /// +0x018 — secondary staging (`0x14000` B).
    pub secondary_staging: usize,
    /// +0x078 — small control block (`0x300` B).
    pub small_control: usize,
    /// +0x090 — state block pair, small half (`0x10` B).
    pub state_small: usize,
    /// +0x098 — state block pair, main half (`0x2d000` B).
    pub state_main: usize,
    /// +0x0b8 — the slot's KV region (`0xf2800` B): the stride
    /// `draft_sample` uses to address per-row KV (spec §1.2).
    pub kv_region: usize,
    /// +0x0d0 — alternate small control block (`0x300` B).
    pub alt_small_control: usize,
    /// +0x0e0 — alternate state block (`0x5000` B).
    pub alt_state_block: usize,
    // --- the §2 request-object fields ---
    /// +0x14 — `HALOGEN_SPEC_N` for the request (0 = unset).
    pub spec_n: u32,
    /// +0x90 — the empty-case KV ring base pointer (0: the port has no
    /// device buffers).
    pub kv_ring_base: u64,
    /// +0x98 + i·8 — the ring of up to `0x40` block pointers.
    pub verify_ring: Vec<u64>,
    /// +0x298 — the sequence length (the port: `context.len()`).
    pub seq_len: u64,
    /// +0x390 — the max-seq bound (0 = unset in the port).
    pub max_seq: u64,
    /// +0x630 — the current position (the port: the next write index).
    pub position: u64,
    /// +0xf94/+0xf9c — the keep-flag ring: which rows' K/V were kept
    /// (`draft_sample: q row %d was not kept`). The port keeps one flag
    /// per resident row; an appended row is always kept.
    pub keep_flags: Vec<bool>,
    /// +0x8b8/+0x8bc/+0x8c0 — the KV ring counters (head/tail/capacity).
    pub ring_counters: [u32; 3],
    /// +0x8c4/+0x8c8/+0x8cc — the commit window: how many speculative
    /// positions are staged but not yet committed. The port models it as
    /// `[uncommitted, 0, 0]` (see [`Self::commit`] and [`Self::rollback`]).
    pub commit_window: [u32; 3],
    /// +0x119c — the MTP staging flag.
    pub mtp_staged: u8,
    /// +0x119d — the number of verify rings (sized by `spec_n`).
    pub verify_rings: u8,
    // --- the port's decode state ---
    /// The token ids resident in the slot (prompt ids followed by the
    /// generated ids so far).
    pub context: Vec<i64>,
    /// The KV region's contents: per-layer K/V rows and recurrent state.
    pub layers: Vec<LayerRuntime>,
    /// The logits produced by the last token step.
    pub logits: Vec<f32>,
    /// Snapshots of every shorter prefix: index `i` holds the layer state
    /// and logits after the first `i + 1` tokens. This is what
    /// [`Self::rollback`] restores from.
    pub prefixes: Vec<(Vec<LayerRuntime>, Vec<f32>)>,
}

impl Default for SlotRecord {
    fn default() -> Self {
        Self::new()
    }
}

impl SlotRecord {
    /// An empty slot carrying the reference's record constants (no model:
    /// the KV rows stay empty until [`Self::with_model`] or a pool
    /// constructor supplies them).
    pub fn new() -> Self {
        SlotRecord {
            activation_staging: ACTIVATION_STAGING,
            secondary_staging: SECONDARY_STAGING,
            small_control: SMALL_CONTROL,
            state_small: STATE_SMALL,
            state_main: STATE_MAIN,
            kv_region: KV_REGION,
            alt_small_control: ALT_SMALL_CONTROL,
            alt_state_block: ALT_STATE_BLOCK,
            spec_n: 0,
            kv_ring_base: 0,
            verify_ring: Vec::new(),
            seq_len: 0,
            max_seq: 0,
            position: 0,
            keep_flags: Vec::new(),
            ring_counters: [0; 3],
            commit_window: [0; 3],
            mtp_staged: 0,
            verify_rings: 0,
            context: Vec::new(),
            layers: Vec::new(),
            logits: Vec::new(),
            prefixes: Vec::new(),
        }
    }

    /// An empty slot whose KV rows are sized for `model`'s layers.
    pub fn with_model(model: &Model) -> Self {
        let mut slot = Self::new();
        slot.layers = new_runtimes(model);
        slot
    }

    /// Run one token step on the slot (the `forward` token step over the
    /// slot's layer state) and record the position: the previous state
    /// becomes a prefix snapshot, the new logits become current, and the
    /// row is marked kept and uncommitted.
    pub fn append_step(&mut self, model: &Model, token: i64) -> Vec<f32> {
        if !self.context.is_empty() {
            // The outgoing current state is the snapshot for the prefix
            // ending at the previous token.
            self.prefixes
                .push((self.layers.clone(), self.logits.clone()));
        }
        let position = self.context.len();
        let logits = crate::forward::forward_token(model, token, position, &mut self.layers);
        self.context.push(token);
        self.keep_flags.push(true);
        self.commit_window[0] = self.commit_window[0].saturating_add(1);
        self.seq_len = self.context.len() as u64;
        self.position = self.context.len() as u64;
        self.logits = logits.clone();
        logits
    }

    /// The number of appended positions not yet committed (the port's
    /// counterpart of the +0x8c4…0x8cf commit window).
    pub fn uncommitted(&self) -> usize {
        self.commit_window[0] as usize
    }

    /// Commit the last `n` appended positions (spec §2: the commit-window
    /// fields record how many speculative positions become committed).
    pub fn commit(&mut self, n: usize) {
        self.commit_window[0] = self.commit_window[0].saturating_sub(n as u32);
    }

    /// Discard the last `n` appended positions, restoring the exact layer
    /// state and logits of the shorter context (the reference's rollback
    /// of rejected speculative rows). At least one position must remain,
    /// and only uncommitted positions can be rolled back.
    pub fn rollback(&mut self, n: usize) -> Result<(), String> {
        if n == 0 {
            return Ok(());
        }
        if n >= self.context.len() {
            return Err(format!(
                "rollback({n}): the slot holds {} positions, at most {} can be rolled back",
                self.context.len(),
                self.context.len() - 1
            ));
        }
        if n > self.uncommitted() {
            return Err(format!(
                "rollback({n}): only {} positions are uncommitted",
                self.uncommitted()
            ));
        }
        for _ in 0..n {
            let (layers, logits) = self
                .prefixes
                .pop()
                .expect("n <= context.len() - 1 == prefixes.len()");
            self.layers = layers;
            self.logits = logits;
            self.context.pop();
            self.keep_flags.pop();
        }
        self.commit_window[0] -= n as u32;
        self.seq_len = self.context.len() as u64;
        self.position = self.context.len() as u64;
        Ok(())
    }
}

/// Fresh per-layer runtimes for a new sequence (empty KV, zeroed conv
/// history and delta states; moved from `forward`).
pub fn new_runtimes(model: &Model) -> Vec<LayerRuntime> {
    model
        .layers
        .iter()
        .map(|layer| match &layer.block {
            Block::Attention(_) => LayerRuntime {
                k: Vec::new(),
                v: Vec::new(),
                conv_hist: Vec::new(),
                delta: Vec::new(),
            },
            Block::Linear(linear) => LayerRuntime {
                k: Vec::new(),
                v: Vec::new(),
                conv_hist: Vec::new(),
                delta: vec![0.0; linear.v_heads * linear.v_head_dim * linear.head_dim],
            },
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Slot pool (slots_alloc_pooled, FUN_020120b0)
// ---------------------------------------------------------------------------

/// The engine's slot pool: `B` slots sharing one KV pool of N positions
/// carved into per-request regions (the reference's
/// `slots: %d -> %d POOLED ... the KV is ONE pool of %d positions ...
/// carved into per-request regions`).
#[derive(Debug, Clone)]
pub struct SlotPool {
    slots: Vec<SlotRecord>,
    positions: usize,
    positions_per_slot: usize,
    mode: Kv8Mode,
}

impl SlotPool {
    /// Port of `slots_alloc_pooled(B)` on a fresh engine (no pre-existing
    /// slots). The error wordings are the reference's exact strings.
    pub fn alloc_pooled(b: usize, positions: usize, mode: Kv8Mode) -> Result<SlotPool, String> {
        Self::alloc_pooled_with_private_slots(b, positions, mode, 0)
    }

    /// The full port: `existing_private_slots` is the reference's slot
    /// count before the call (the vector at `engine+0x328…0x338`). A
    /// non-pooled engine with more than one private slot refuses to switch
    /// (`slots_alloc_pooled(%d): %d slots already hold private KVs`);
    /// with one or zero the pool grows to `b` slots.
    pub fn alloc_pooled_with_private_slots(
        b: usize,
        positions: usize,
        mode: Kv8Mode,
        existing_private_slots: usize,
    ) -> Result<SlotPool, String> {
        if b == 0 || b > MAX_SLOTS {
            return Err(format!("slots_alloc_pooled({b}): 1 <= B <= {MAX_SLOTS}"));
        }
        if existing_private_slots > 1 {
            return Err(format!(
                "slots_alloc_pooled({b}): {existing_private_slots} slots already hold private KVs"
            ));
        }
        // The pool is carved into B equal per-request regions (documented
        // assumption: the reference prints the carve but the region sizes
        // are not exposed per slot).
        let positions_per_slot = positions / b;
        let mut slots = vec![SlotRecord::new(); b];
        for slot in &mut slots {
            slot.max_seq = positions_per_slot as u64;
        }
        Ok(SlotPool {
            slots,
            positions,
            positions_per_slot,
            mode,
        })
    }

    /// The port's single-slot pool for `Forward`: `B = 1`, positions sized
    /// by [`Budget::from_checkpoint`] (the port has no `--slots`/`--max-tok`
    /// surface here, so the binary's defaults are used).
    pub fn single(model: &Model, checkpoint: &hgn::Checkpoint, mode: Kv8Mode) -> SlotPool {
        let budget = Budget::from_checkpoint(checkpoint, 1, DEFAULT_MAX_TOK);
        let positions = budget.pool_positions(mode);
        let mut pool = Self::alloc_pooled(1, positions, mode).expect("B = 1 is always in range");
        pool.slots[0] = SlotRecord::with_model(model);
        pool
    }

    /// The pool's total position count (the "ONE pool of %d positions").
    pub fn positions(&self) -> usize {
        self.positions
    }

    /// The per-request region size (the pool divided by the slot count).
    pub fn positions_per_slot(&self) -> usize {
        self.positions_per_slot
    }

    /// The KV8 mode the pool was sized with.
    pub fn mode(&self) -> Kv8Mode {
        self.mode
    }

    /// The pool's slots.
    pub fn slots(&self) -> &[SlotRecord] {
        &self.slots
    }

    /// The pool's mutable slots.
    pub fn slots_mut(&mut self) -> &mut [SlotRecord] {
        &mut self.slots
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv8_mode_parsing_matches_the_reference_grammar() {
        assert_eq!(parse_kv8(None), Ok(Kv8Mode::Off));
        assert_eq!(parse_kv8(Some("")), Ok(Kv8Mode::Off));
        assert_eq!(parse_kv8(Some("0")), Ok(Kv8Mode::Off));
        assert_eq!(parse_kv8(Some("1")), Ok(Kv8Mode::Kv8));
        assert_eq!(parse_kv8(Some("2")), Ok(Kv8Mode::Extended));
        // Everything else — including multi-character strings — exits with
        // the reference's wording.
        assert_eq!(
            parse_kv8(Some("x")),
            Err("HALOGEN_KV8=x: 1 (the 8-bit KV cache) or unset".to_string())
        );
        assert_eq!(
            parse_kv8(Some("00")),
            Err("HALOGEN_KV8=00: 1 (the 8-bit KV cache) or unset".to_string())
        );
        assert_eq!(
            parse_kv8(Some("10")),
            Err("HALOGEN_KV8=10: 1 (the 8-bit KV cache) or unset".to_string())
        );
    }

    #[test]
    fn per_position_bytes_match_the_sizing_function() {
        // bf16: (0x400 + 0x400) · 13 + 0xb40 = 29 504 (the frozen spec's
        // "about 29.5 KiB a position"); KV8: (0x208 + 0x220) · 13 + 0xb40
        // = 16 712 (the spec prose's 16 816 is an arithmetic slip, see the
        // function docs); mode 2 sizes like bf16.
        assert_eq!(kv_row_bytes(Kv8Mode::Off), (0x400, 0x400));
        assert_eq!(kv_row_bytes(Kv8Mode::Kv8), (0x208, 0x220));
        assert_eq!(kv_row_bytes(Kv8Mode::Extended), (0x400, 0x400));
        assert_eq!(bytes_per_position(Kv8Mode::Off), 29_504);
        assert_eq!(bytes_per_position(Kv8Mode::Kv8), 16_712);
        assert_eq!(bytes_per_position(Kv8Mode::Extended), 29_504);
    }

    #[test]
    fn slot_alloc_bounds_and_private_kv_refusal() {
        assert_eq!(
            SlotPool::alloc_pooled(0, 256, Kv8Mode::Off).unwrap_err(),
            "slots_alloc_pooled(0): 1 <= B <= 64"
        );
        assert_eq!(
            SlotPool::alloc_pooled(65, 256, Kv8Mode::Off).unwrap_err(),
            "slots_alloc_pooled(65): 1 <= B <= 64"
        );
        let pool = SlotPool::alloc_pooled(4, 1024, Kv8Mode::Off).expect("in range");
        assert_eq!(pool.slots().len(), 4);
        assert_eq!(pool.positions(), 1024);
        assert_eq!(pool.positions_per_slot(), 256);
        assert_eq!(pool.mode(), Kv8Mode::Off);
        // A non-pooled engine with more than one private slot refuses.
        assert_eq!(
            SlotPool::alloc_pooled_with_private_slots(4, 1024, Kv8Mode::Off, 2).unwrap_err(),
            "slots_alloc_pooled(4): 2 slots already hold private KVs"
        );
        // One existing slot is kept and the pool grows to B.
        let pool = SlotPool::alloc_pooled_with_private_slots(4, 1024, Kv8Mode::Kv8, 1)
            .expect("one private slot may switch to pooled");
        assert_eq!(pool.slots().len(), 4);
        // The boundary B = 64 is valid.
        assert!(SlotPool::alloc_pooled(64, 64 * 256, Kv8Mode::Off).is_ok());
    }

    #[test]
    fn pool_budget_matches_the_decompile_formula() {
        // 128 GiB host, 64 GiB weights, 20 GiB reserve, 4 slots, n-gram
        // on, max-tok 8192, KV8 extras 1 GiB, vision 0.5 GiB, spec rows
        // 0.25 GiB — the formula mirrors FUN_02019610 with explicit
        // literals so the test does not re-run the implementation.
        let budget = Budget {
            host_ram_gib: 128.0,
            weights_gib: 64.0,
            host_reserve_gib: 20.0,
            slots: 4,
            ngram_on: true,
            max_tok: 8192,
            keep_trunk_gib: 0.0,
            kv8_extra_gib: 1.0,
            vision_gib: 0.5,
            spec_rows_gib: 0.25,
        };
        // device = (128 − 64 − 20) GiB; non-pool = 4·0.115 + 8.8 +
        // 3.1·8192/32768 + 1 + 0.5 + 0.25 GiB (the KV8 extra is charged
        // in mode 1 only).
        let available = ((128.0 - 64.0 - 20.0) * GIB) as u64
            - ((4.0 * 0.115 + 8.8 + 3.1 * 8192.0 / 32768.0 + 1.0 + 0.5 + 0.25) * GIB) as u64;
        assert_eq!(
            budget.pool_positions(Kv8Mode::Kv8),
            positions_from_bytes(available, Kv8Mode::Kv8)
        );
        // Without the KV8 extra the pool is larger (the extra is only
        // charged in mode 1).
        let plain = Budget {
            kv8_extra_gib: 0.0,
            ..budget
        };
        assert!(plain.pool_positions(Kv8Mode::Kv8) >= budget.pool_positions(Kv8Mode::Kv8));
    }

    #[test]
    fn pool_budget_clamps_weights_caps_and_rounds() {
        // A checkpoint with no reported weights charges 68.8 GiB (the C
        // clamp).
        let budget = Budget {
            host_ram_gib: 100.0,
            ..Budget::default()
        };
        assert_eq!(budget.device_bytes(), ((100.0 - 68.8) * GIB) as u64);
        // A host smaller than weights + reserve yields zero device bytes.
        let tiny = Budget {
            host_ram_gib: 10.0,
            ..Budget::default()
        };
        assert_eq!(tiny.device_bytes(), 0);
        // The 0x1000000 cap and the 256-unit rounding.
        assert_eq!(
            positions_from_bytes((MAX_POOL_POSITIONS as u64 + 4096) * 29_504, Kv8Mode::Off),
            MAX_POOL_POSITIONS
        );
        assert_eq!(positions_from_bytes(29_504 * 513, Kv8Mode::Off), 512);
        assert_eq!(positions_from_bytes(29_504 * 512 - 1, Kv8Mode::Off), 256);
        assert_eq!(positions_from_bytes(0, Kv8Mode::Kv8), 0);
    }

    #[test]
    fn append_and_rollback_round_trip_matches_a_fresh_recompute() {
        let ck = crate::model::mini_checkpoint();
        let m = crate::model::load(&ck).expect("the mini fixture loads the real path");
        let mut slot = SlotRecord::with_model(&m);
        for token in [1i64, 2, 3, 4, 5] {
            slot.append_step(&m, token);
        }
        assert_eq!(slot.context, vec![1, 2, 3, 4, 5]);
        assert_eq!(slot.uncommitted(), 5);
        // Roll the last three positions back.
        slot.rollback(3)
            .expect("three uncommitted positions roll back");
        assert_eq!(slot.context, vec![1, 2]);
        assert_eq!(slot.uncommitted(), 2);
        // The restored state must equal a fresh two-token recompute.
        let mut fresh = SlotRecord::with_model(&m);
        for token in [1i64, 2] {
            fresh.append_step(&m, token);
        }
        assert_eq!(slot.logits, fresh.logits);
        assert_eq!(slot.layers, fresh.layers);
        // Committing the remaining positions leaves nothing to roll back.
        slot.commit(2);
        assert_eq!(slot.uncommitted(), 0);
        assert!(slot.rollback(1).is_err());
        // And a further append continues from the restored state.
        let logits3 = slot.append_step(&m, 3);
        let mut fresh3 = fresh.clone();
        assert_eq!(
            logits3,
            fresh3.append_step(&m, 3),
            "appending after a rollback matches the unbroken context"
        );
    }
}
