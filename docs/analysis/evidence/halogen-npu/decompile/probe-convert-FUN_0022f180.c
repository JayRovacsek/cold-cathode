// rea decompile_all
// name: FUN_0022f180
// address: 0x22f180
// body: available
// range: 0x22f180-0x22f51d inclusive
// range: 0x22f538-0x22f601 inclusive
// range: 0x22f608-0x22f7c6 inclusive
// range: 0x22f818-0x22f865 inclusive
// range: 0x22f870-0x22f8c0 inclusive
// range: 0x22f8c8-0x22f94e inclusive
// range: 0x22fa38-0x22fa89 inclusive
// range: 0x22fa90-0x22fabe inclusive
// range: 0x22fac0-0x22faec inclusive
// range: 0x22faf0-0x22fb2c inclusive
// range: 0x22fb61-0x22fb7f inclusive
// range: 0x22fe65-0x22fe89 inclusive
// range: 0x22feaa-0x22febb inclusive
// range: 0x22fedc-0x22ff2d inclusive
// non_contiguous: true
// provider: Ghidra 12.1.4 (ghidra)
// analysis_profile: ae80ece8c83379eddabbdf379bbffca5da84c7df7fa7c159f45fa810468f85d9
// limitations: The session serves operations only after Ghidra reports default auto-analysis complete; incomplete analysis does not expose partial results.; The imported Program and temporary project are ephemeral and deleted on close. Annotation operations edit session database metadata; original executable bytes are never written.; Cancelling an active Ghidra request terminates its ephemeral database and discards session annotations. The selected target remains open; the next provider query imports the original artifact again.; Default-space addresses use lowercase 0x-prefixed hexadecimal; other address spaces use <percent-encoded-space>:0x<hex>.; Ghidra Listing reports a terminal call at 0x22f519 to 0x17ffc0 std::__cxx11::string::_M_dispose (external=false, thunk=false, hasNoReturn=true); fallthrough is excluded by Ghidra's flow model. This records Ghidra's FunctionManager flag, not an independent verification that the callee cannot return.; Ghidra Listing reports a terminal call at 0x22f5e5 to 0x10d660 <EXTERNAL>::operator.delete (external=false, thunk=true, hasNoReturn=true); fallthrough is excluded by Ghidra's flow model. This records Ghidra's FunctionManager flag, not an independent verification that the callee cannot return.; Ghidra Listing reports a terminal call at 0x22f7c2 to 0x17ffc0 std::__cxx11::string::_M_dispose (external=false, thunk=false, hasNoReturn=true); fallthrough is excluded by Ghidra's flow model. This records Ghidra's FunctionManager flag, not an independent verification that the callee cannot return.; Ghidra Listing reports a terminal call at 0x22f917 to 0x10d660 <EXTERNAL>::operator.delete (external=false, thunk=true, hasNoReturn=true); fallthrough is excluded by Ghidra's flow model. This records Ghidra's FunctionManager flag, not an independent verification that the callee cannot return.; Ghidra Listing reports a terminal call at 0x22f94a to 0x17ffc0 std::__cxx11::string::_M_dispose (external=false, thunk=false, hasNoReturn=true); fallthrough is excluded by Ghidra's flow model. This records Ghidra's FunctionManager flag, not an independent verification that the callee cannot return.; Ghidra Listing reports a terminal call at 0x22fb28 to 0x17ffc0 std::__cxx11::string::_M_dispose (external=false, thunk=false, hasNoReturn=true); fallthrough is excluded by Ghidra's flow model. This records Ghidra's FunctionManager flag, not an independent verification that the callee cannot return.; Ghidra Listing reports a terminal call at 0x22fb7b to 0x17ffc0 std::__cxx11::string::_M_dispose (external=false, thunk=false, hasNoReturn=true); fallthrough is excluded by Ghidra's flow model. This records Ghidra's FunctionManager flag, not an independent verification that the callee cannot return.; Pseudocode is Ghidra decompiler output, not original source and not text-equivalent to Hopper output.; External functions and functions without an analyzable body return null; other decompiler failures remain explicit.
// pseudocode: provider-specific; not original source


undefined8 FUN_0022f180(int param_1,undefined8 *param_2)

{
  char cVar1;
  byte bVar2;
  undefined1 *puVar3;
  int iVar4;
  long *local_500;
  long local_4f8;
  long local_4f0 [2];
  undefined1 *local_4e0;
  undefined8 local_4d8;
  undefined1 local_4d0 [16];
  undefined1 *local_4c0;
  undefined8 local_4b8;
  undefined1 local_4b0 [16];
  undefined1 *local_4a0;
  undefined8 local_498;
  undefined1 local_490 [16];
  undefined1 *local_480;
  undefined8 local_478;
  undefined1 local_470 [16];
  undefined1 *local_460;
  undefined8 local_458;
  undefined1 local_450 [176];
  undefined1 local_3a0 [32];
  long *local_380;
  long local_378;
  undefined1 local_370;
  undefined7 uStack_36f;
  undefined1 *local_360;
  undefined8 local_358;
  undefined1 local_350 [16];
  undefined1 *local_340;
  undefined8 local_338;
  undefined1 local_330 [16];
  undefined1 *local_320;
  undefined8 local_318;
  undefined1 local_310 [16];
  undefined1 *local_300;
  undefined8 local_2f8;
  undefined1 local_2f0 [16];
  undefined1 *local_2e0;
  undefined8 local_2d8;
  undefined1 local_2d0 [16];
  undefined1 *local_2c0;
  undefined8 local_2b8;
  undefined1 local_2b0 [16];
  undefined4 local_2a0;
  undefined1 local_29c;
  undefined1 *local_298;
  undefined8 local_290;
  undefined1 local_288 [16];
  undefined8 local_278;
  undefined8 uStack_270;
  undefined8 uStack_268;
  undefined8 uStack_260;
  undefined8 local_258;
  undefined8 uStack_250;
  undefined8 uStack_248;
  undefined8 uStack_240;
  undefined1 local_238 [16];
  undefined4 local_220 [2];
  undefined8 local_218;
  undefined4 *local_210;
  undefined4 *local_208;
  undefined8 local_200;
  undefined1 local_1f8;
  undefined8 local_1f4;
  undefined8 uStack_1ec;
  undefined8 uStack_1e4;
  undefined8 uStack_1dc;
  undefined8 local_1d0;
  undefined4 local_1c8;
  undefined8 local_1c0;
  undefined4 local_1b8;
  undefined8 local_1b0;
  undefined8 uStack_1a8;
  undefined8 uStack_1a0;
  undefined8 uStack_198;
  undefined8 local_190;
  undefined8 local_188;
  undefined2 local_180;
  undefined8 local_178;
  undefined8 local_170;
  undefined8 uStack_168;
  undefined8 uStack_160;
  undefined8 uStack_158;
  undefined8 local_150;
  undefined8 uStack_148;
  undefined8 uStack_140;
  undefined8 uStack_138;
  undefined8 local_130;
  undefined1 *local_128;
  undefined8 local_120;
  undefined1 local_118 [16];
  undefined1 *local_108;
  undefined8 local_100;
  undefined1 local_f8 [16];
  undefined1 local_e8 [16];
  undefined1 local_d8;
  undefined1 *local_d0;
  undefined8 local_c8;
  undefined1 local_c0 [16];
  undefined1 *local_b0;
  undefined8 local_a8;
  undefined1 local_a0 [16];
  undefined8 local_90;
  undefined1 local_88;
  undefined8 local_80;
  undefined8 uStack_78;
  undefined8 uStack_70;
  undefined8 uStack_68;
  undefined1 *local_60;
  undefined8 local_58;
  undefined1 local_50 [16];
  
                    /* catch() { ... } // from try @ 0022f17a with catch @ 0022f198 */
  puVar3 = &LAB_0038ae38_1;
  if (0 < param_1) {
    puVar3 = (undefined1 *)*param_2;
  }
  FUN_001e04a0(&local_500,puVar3);
  if (local_4f8 == 5) {
                    /* try { // try from 0022faf6 to 0032fafa has its CatchHandler @ 0022fad6 */
    if (((int)*local_500 == 0x626f7270) && (*(char *)((long)local_500 + 4) == 'e')) {
      if (param_1 == 2) {
        FUN_001e04a0((string *)&local_360,param_2[1]);
        FUN_001fad90((string *)&local_360);
                    /* WARNING: Subroutine does not return */
        std::__cxx11::string::_M_dispose((string *)&local_360);
      }
      goto LAB_0022f580;
    }
    if (((int)*local_500 == 0x626f7270) && (*(char *)((long)local_500 + 4) == 'e')) {
      if (param_1 == 2) {
        FUN_001e04a0((string *)&local_360,param_2[1]);
        FUN_001f9ff0((string *)&local_360);
                    /* WARNING: Subroutine does not return */
                    /* try { // try from 0022fb7f to 0032fb83 has its CatchHandler @ 0022fbd1 */
        std::__cxx11::string::_M_dispose((string *)&local_360);
      }
      goto LAB_0022f580;
    }
  }
  cVar1 = FUN_0017e3f0(&local_500,"probe");
  if ((param_1 == 2) && (cVar1 != '\0')) {
    FUN_001e04a0(local_3a0,param_2[1]);
    local_360 = local_350;
    local_358 = 0;
    local_340 = local_330;
                    /* try { // try from 0022f250 to 0032f254 has its CatchHandler @ 0022ed39 */
                    /* catch() { ... } // from try @ 0022f45a with catch @ 0022f256 */
    local_320 = local_310;
    local_300 = local_2f0;
    local_2e0 = local_2d0;
    local_2c0 = local_2b0;
    local_298 = local_288;
    local_210 = local_220;
    local_350[0] = 0;
    local_338 = 0;
    local_330[0] = 0;
    local_318 = 0;
    local_310[0] = 0;
    local_2f8 = 0;
    local_2f0[0] = 0;
    local_2d8 = 0;
    local_2d0[0] = 0;
    local_2b8 = 0;
    local_2b0[0] = 0;
    local_2a0 = 0;
    local_29c = 0;
    local_290 = 0;
    local_288[0] = 0;
    local_220[0] = 0;
    local_218 = 0;
    local_200 = 0;
    local_1f8 = 0;
    local_278 = 0;
    uStack_270 = 0;
    uStack_268 = 0;
    uStack_260 = 0;
    local_258 = 0;
    uStack_250 = 0;
    uStack_248 = 0;
    uStack_240 = 0;
    local_238 = (undefined1  [16])0x0;
    local_1f4 = 0;
    uStack_1ec = 0;
    uStack_1e4 = 0;
    uStack_1dc = 0;
    local_180 = 0;
    local_128 = local_118;
    local_108 = local_f8;
    local_d0 = local_c0;
    local_b0 = local_a0;
    local_60 = local_50;
    local_1d0 = 0;
    local_1c8 = 0;
    local_1c0 = 0;
    local_1b8 = 0;
    local_190 = 0;
    local_188 = 0;
    local_178 = 0;
    local_130 = 0;
    local_120 = 0;
    local_118[0] = 0;
    local_100 = 0;
    local_f8[0] = 0;
    local_d8 = 0;
    local_c8 = 0;
                    /* try { // try from 0022f45a to 0032f45e has its CatchHandler @ 0022f256 */
    local_c0[0] = 0;
    local_a8 = 0;
    local_a0[0] = 0;
    local_90 = 0;
    local_88 = 0;
    local_58 = 0;
    local_50[0] = 0;
    local_1b0 = 0;
    uStack_1a8 = 0;
    uStack_1a0 = 0;
    uStack_198 = 0;
    local_170 = 0;
    uStack_168 = 0;
    uStack_160 = 0;
    uStack_158 = 0;
    local_150 = 0;
    uStack_148 = 0;
    uStack_140 = 0;
    uStack_138 = 0;
    local_e8 = (undefined1  [16])0x0;
    local_80 = 0;
    uStack_78 = 0;
    uStack_70 = 0;
    uStack_68 = 0;
    local_380 = (long *)&local_370;
    local_378 = 0;
    local_370 = 0;
    local_208 = local_210;
    cVar1 = FUN_002108c0(local_3a0,&local_360,(string *)&local_380);
    if (cVar1 == '\0') {
                    /* try { // try from 0022fe6b to 0032fe6f has its CatchHandler @ 0022ffbc */
      fprintf(stderr,"halogen-npu: %s\n",local_380);
    }
    else {
      printf("family=%s task=%s base=%s sig=%s\n",local_340,local_320,local_300,local_2e0);
    }
                    /* WARNING: Subroutine does not return */
    std::__cxx11::string::_M_dispose((string *)&local_380);
  }
LAB_0022f580:
  bVar2 = FUN_0017e3f0(&local_500,"convert");
  if ((3 < param_1 & bVar2) == 0) {
                    /* try { // try from 0022f5bc to 0032f5c0 has its CatchHandler @ 0022f5c1 */
    fwrite("usage: halogen-npu probe MODEL_DIR | convert MODEL_DIR DEVICES_DIR OUT [--name ID] [--source SRC] [--if-stale] [--fold FILE --enc FILE --tokenizer FILE (a speech model)]\n"
           ,1,0xaa,stderr);
                    /* catch() { ... } // from try @ 0022f5bc with catch @ 0022f5c1 */
    if (local_500 == local_4f0) {
      return 1;
    }
                    /* WARNING: Subroutine does not return */
    operator_delete(local_500,local_4f0[0] + 1);
  }
  local_4e0 = local_4d0;
  local_4d8 = 0;
                    /* catch() { ... } // from try @ 0022f7ce with catch @ 0022f620 */
  local_4c0 = local_4b0;
  local_4a0 = local_490;
  local_480 = local_470;
  local_460 = local_450;
  local_4d0[0] = 0;
  local_4b8 = 0;
  local_4b0[0] = 0;
  local_498 = 0;
  local_490[0] = 0;
  local_478 = 0;
  local_470[0] = 0;
                    /* try { // try from 0022f693 to 0032f697 has its CatchHandler @ 0022f698 */
                    /* catch() { ... } // from try @ 0022f693 with catch @ 0022f698 */
  local_458 = 0;
  local_450[0] = 0;
  if (param_1 != 4) {
    iVar4 = 4;
    do {
                    /* try { // try from 0022f736 to 0032f73a has its CatchHandler @ 0022f73b */
      FUN_001e04a0(&local_380,param_2[iVar4]);
                    /* catch() { ... } // from try @ 0022f736 with catch @ 0022f73b */
      if (local_378 == 6) {
        if (((int)*local_380 == 0x616e2d2d) && (*(short *)((long)local_380 + 4) == 0x656d)) {
          if (param_1 <= iVar4 + 1) {
LAB_0022f780:
            FUN_001ecf40((string *)&local_360,&LAB_0038bf72_2,local_380);
            fprintf(stderr,"halogen-npu: %s\n",local_360);
                    /* WARNING: Subroutine does not return */
            std::__cxx11::string::_M_dispose((string *)&local_360);
          }
          FUN_001e6020(&local_4e0,param_2[(long)iVar4 + 1]);
        }
        else {
          if (((int)*local_380 != 0x6f662d2d) ||
             ((*(short *)((long)local_380 + 4) != 0x646c || (param_1 <= iVar4 + 1))))
          goto LAB_0022f780;
          FUN_001e6020(&local_4a0,param_2[(long)iVar4 + 1]);
        }
LAB_0022f900:
        iVar4 = iVar4 + 1;
        if (local_380 != (long *)&local_370) {
LAB_0022f90c:
                    /* WARNING: Subroutine does not return */
          operator_delete(local_380,CONCAT71(uStack_36f,local_370) + 1);
        }
      }
      else {
        if (local_378 == 8) {
          if ((*local_380 != 0x656372756f732d2d) || (param_1 <= iVar4 + 1)) goto LAB_0022f780;
          FUN_001e6020(&local_4c0,param_2[(long)iVar4 + 1]);
          goto LAB_0022f900;
        }
        if (local_378 == 5) {
                    /* catch() { ... } // from try @ 0022faf6 with catch @ 0022fad6 */
          if ((((int)*local_380 != 0x6e652d2d) || (*(char *)((long)local_380 + 4) != 'c')) ||
             (param_1 <= iVar4 + 1)) goto LAB_0022f780;
                    /* catch() { ... } // from try @ 0022fc3f with catch @ 0022fad7 */
          FUN_001e6020(&local_480,param_2[(long)iVar4 + 1]);
          goto LAB_0022f900;
        }
        if (local_378 == 0xb) {
          if (((*local_380 != 0x696e656b6f742d2d) || (*(int *)((long)local_380 + 7) != 0x72657a69))
             || (param_1 <= iVar4 + 1)) goto LAB_0022f780;
          FUN_001e6020(&local_460,param_2[(long)iVar4 + 1]);
          goto LAB_0022f900;
        }
        if (((local_378 != 10) || (*local_380 != 0x6174732d66692d2d)) ||
           ((short)local_380[1] != 0x656c)) goto LAB_0022f780;
        if (local_380 != (long *)&local_370) goto LAB_0022f90c;
      }
      iVar4 = iVar4 + 1;
    } while (iVar4 < param_1);
  }
  FUN_001e04a0((string *)&local_360,param_2[1]);
  FUN_001fad90((string *)&local_360);
                    /* WARNING: Subroutine does not return */
  std::__cxx11::string::_M_dispose((string *)&local_360);
}

