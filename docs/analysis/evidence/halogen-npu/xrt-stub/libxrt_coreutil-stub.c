/*
 * libxrt_coreutil.so.2 stub — rebuilt 2026-11-03 for cold-cathode contract testing.
 *
 * halogen-npu links against AMD XRT's libxrt_coreutil.so.2 (DT_NEEDED), which
 * is absent on this host (the upstream derivation sets autoPatchelfIgnoreMissingDeps).
 * Direct invocation exits 127 with "error while loading shared libraries".
 * This stub provides no-op definitions for the XRT/`_ITM` symbols imported via
 * .dynsym (the 20 `xrt::` symbols plus the 2 `_ITM_*` clone-table symbols) so
 * the ELF loads and the real argument-parsing/validation paths execute.
 *
 * The symbol set was extracted from the reference binary with:
 *   readelf -W --dyn-syms <store>/bin/halogen-npu | awk '$7 == "UND"' | grep -E 'xrt|_ITM'
 * matching the stub described in docs/analysis/specs/halogen-npu.md
 * ("no-op definitions for the 23 imported XRT/_ITM symbols").
 *
 * XRT-dependent behaviour beyond argument parsing (device enumeration, engine
 * I/O, inference) is NOT exercised — see the spec's "Method and limitations".
 *
 * Rebuild:
 *   cc -shared -fPIC -nostdlib -Wl,-soname,libxrt_coreutil.so.2 \
 *      -o libxrt_coreutil.so.2 libxrt_coreutil-stub.c
 */

/* xrt::hw_context */
void _ZN3xrt10hw_contextC1ERKNS_6deviceERKNS_3elfE(void) {}
void _ZN3xrt10hw_contextD1Ev(void) {}

/* xrt::bo */
void _ZN3xrt2bo3mapEv(void) {}
void _ZN3xrt2bo4syncE18xclBOSyncDirectionmm(void) {}
void _ZN3xrt2boC1ERKNS_6deviceEmNS0_5flagsEj(void) {}
void _ZN3xrt2boC1ERKS0_mm(void) {}
void _ZN3xrt2boD1Ev(void) {}
unsigned long _ZNK3xrt2bo4sizeEv(void) { return 0; }

/* xrt::elf */
void _ZN3xrt3elfC1ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(void) {}

/* xrt::ext::kernel */
void _ZN3xrt3ext6kernelC1ERKNS_10hw_contextERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(void) {}
void _ZN3xrt6kernelD2Ev(void) {}

/* xrt::run */
void _ZN3xrt3run16set_arg_at_indexEiRKNS_2boE(void) {}
void _ZN3xrt3run5startEv(void) {}
void _ZN3xrt3runC1ERKNS_6kernelE(void) {}
void _ZN3xrt3runD1Ev(void) {}
void _ZNK3xrt3run4waitERKNSt6chrono8durationIlSt5ratioILl1ELl1000EEEE(void) {}
void _ZNK3xrt3run5stateEv(void) {}

/* xrt::device */
void _ZN3xrt6deviceC1Ej(void) {}
void _ZN3xrt6deviceD1Ev(void) {}
void _ZNK3xrt6device12get_info_stdENS_4info6deviceERKNS_6detail3abiE(void) {}

/* GCC transactional-memory clone table hooks (normally libgcc_s weak symbols) */
void _ITM_deregisterTMCloneTable(void) {}
void _ITM_registerTMCloneTable(void) {}
