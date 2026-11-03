// rea decompile_all
// name: FUN_0013c2f0
// address: 0x13c2f0
// body: available
// range: 0x13c2f0-0x13c35d inclusive
// non_contiguous: false
// provider: Ghidra 12.1.4 (ghidra)
// analysis_profile: ae80ece8c83379eddabbdf379bbffca5da84c7df7fa7c159f45fa810468f85d9
// limitations: The session serves operations only after Ghidra reports default auto-analysis complete; incomplete analysis does not expose partial results.; The imported Program and temporary project are ephemeral and deleted on close. Annotation operations edit session database metadata; original executable bytes are never written.; Cancelling an active Ghidra request terminates its ephemeral database and discards session annotations. The selected target remains open; the next provider query imports the original artifact again.; Default-space addresses use lowercase 0x-prefixed hexadecimal; other address spaces use <percent-encoded-space>:0x<hex>.; Pseudocode is Ghidra decompiler output, not original source and not text-equivalent to Hopper output.; External functions and functions without an analyzable body return null; other decompiler failures remain explicit.
// pseudocode: provider-specific; not original source


void FUN_0013c2f0(undefined4 param_1,undefined8 param_2)

{
  char *pcVar1;
  
  pcVar1 = getenv("HALOGEN_NPU_ASR_HOST");
  if ((pcVar1 != (char *)0x0) && (*pcVar1 == '1')) {
    setenv("HALOGEN_NPU_HOST_THREADS","8",0);
    setenv("HALOGEN_NPU_WAIT_SPIN","0",0);
    setenv("HALOGEN_NPU_HOST_PREWAKE","0",0);
  }
  FUN_00170740(param_1,param_2);
  return;
}

