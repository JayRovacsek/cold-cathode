// rea decompile_all
// name: FUN_00170740
// address: 0x170740
// body: available
// range: 0x112fbf-0x112fca inclusive
// range: 0x113402-0x113418 inclusive
// range: 0x170740-0x1708eb inclusive
// range: 0x1708f1-0x170cd3 inclusive
// range: 0x170cd9-0x170e48 inclusive
// range: 0x170e61-0x170e6c inclusive
// range: 0x170e72-0x171043 inclusive
// range: 0x17108f-0x171238 inclusive
// range: 0x17125f-0x1715da inclusive
// range: 0x1715f6-0x1716cf inclusive
// range: 0x1716f6-0x171721 inclusive
// range: 0x171727-0x171958 inclusive
// range: 0x171974-0x1719c0 inclusive
// range: 0x1719c6-0x171b5d inclusive
// range: 0x171ffc-0x172044 inclusive
// range: 0x1721b8-0x1721f5 inclusive
// range: 0x172bc9-0x172c15 inclusive
// range: 0x172c1b-0x1734a1 inclusive
// range: 0x1734a7-0x1734cf inclusive
// range: 0x1734d8-0x173730 inclusive
// range: 0x17373e-0x1737a2 inclusive
// range: 0x1737e4-0x173848 inclusive
// non_contiguous: true
// provider: Ghidra 12.1.4 (ghidra)
// analysis_profile: ae80ece8c83379eddabbdf379bbffca5da84c7df7fa7c159f45fa810468f85d9
// limitations: The session serves operations only after Ghidra reports default auto-analysis complete; incomplete analysis does not expose partial results.; The imported Program and temporary project are ephemeral and deleted on close. Annotation operations edit session database metadata; original executable bytes are never written.; Cancelling an active Ghidra request terminates its ephemeral database and discards session annotations. The selected target remains open; the next provider query imports the original artifact again.; Default-space addresses use lowercase 0x-prefixed hexadecimal; other address spaces use <percent-encoded-space>:0x<hex>.; Ghidra Listing reports a terminal call at 0x112fc6 to 0x10d390 <EXTERNAL>::std::__throw_length_error (external=false, thunk=true, hasNoReturn=true); fallthrough is excluded by Ghidra's flow model. This records Ghidra's FunctionManager flag, not an independent verification that the callee cannot return.; Ghidra Listing reports a terminal call at 0x113414 to 0x17ffc0 std::__cxx11::string::_M_dispose (external=false, thunk=false, hasNoReturn=true); fallthrough is excluded by Ghidra's flow model. This records Ghidra's FunctionManager flag, not an independent verification that the callee cannot return.; Ghidra Listing reports a terminal call at 0x1708e7 to 0x17ffc0 std::__cxx11::string::_M_dispose (external=false, thunk=false, hasNoReturn=true); fallthrough is excluded by Ghidra's flow model. This records Ghidra's FunctionManager flag, not an independent verification that the callee cannot return.; Ghidra Listing reports a terminal call at 0x170ccf to 0x17ffc0 std::__cxx11::string::_M_dispose (external=false, thunk=false, hasNoReturn=true); fallthrough is excluded by Ghidra's flow model. This records Ghidra's FunctionManager flag, not an independent verification that the callee cannot return.; Ghidra Listing reports a terminal call at 0x170cfb to 0x17ffc0 std::__cxx11::string::_M_dispose (external=false, thunk=false, hasNoReturn=true); fallthrough is excluded by Ghidra's flow model. This records Ghidra's FunctionManager flag, not an independent verification that the callee cannot return.; Ghidra Listing reports a terminal call at 0x170e3c to 0x17ffc0 std::__cxx11::string::_M_dispose (external=false, thunk=false, hasNoReturn=true); fallthrough is excluded by Ghidra's flow model. This records Ghidra's FunctionManager flag, not an independent verification that the callee cannot return.; Ghidra Listing reports a terminal call at 0x170e44 to 0x17ffc0 std::__cxx11::string::_M_dispose (external=false, thunk=false, hasNoReturn=true); fallthrough is excluded by Ghidra's flow model. This records Ghidra's FunctionManager flag, not an independent verification that the callee cannot return.; Ghidra Listing reports a terminal call at 0x170e68 to 0x17ffc0 std::__cxx11::string::_M_dispose (external=false, thunk=false, hasNoReturn=true); fallthrough is excluded by Ghidra's flow model. This records Ghidra's FunctionManager flag, not an independent verification that the callee cannot return.; Ghidra Listing reports a terminal call at 0x17103f to 0x17ffc0 std::__cxx11::string::_M_dispose (external=false, thunk=false, hasNoReturn=true); fallthrough is excluded by Ghidra's flow model. This records Ghidra's FunctionManager flag, not an independent verification that the callee cannot return.; Ghidra Listing reports a terminal call at 0x171234 to 0x17ffc0 std::__cxx11::string::_M_dispose (external=false, thunk=false, hasNoReturn=true); fallthrough is excluded by Ghidra's flow model. This records Ghidra's FunctionManager flag, not an independent verification that the callee cannot return.; Ghidra Listing reports a terminal call at 0x17158e to 0x10d660 <EXTERNAL>::operator.delete (external=false, thunk=true, hasNoReturn=true); fallthrough is excluded by Ghidra's flow model. This records Ghidra's FunctionManager flag, not an independent verification that the callee cannot return.; Ghidra Listing reports a terminal call at 0x1715
// limitations_truncated: true; decompile-index.json keeps the complete list
// pseudocode: provider-specific; not original source


/* WARNING: Removing unreachable block (ram,0x0017370b) */
/* WARNING: Removing unreachable block (ram,0x0017376a) */
/* WARNING: Removing unreachable block (ram,0x00173810) */
/* WARNING: Removing unreachable block (ram,0x00112fbf) */
/* WARNING: Removing unreachable block (ram,0x00112fb3) */
/* WARNING: Globals starting with '_' overlap smaller symbols at the same address */

undefined8 FUN_00170740(int param_1,long param_2)

{
  long *plVar1;
  undefined1 auVar2 [16];
  undefined1 auVar3 [32];
  undefined1 auVar4 [32];
  undefined1 auVar5 [32];
  undefined1 auVar6 [32];
  undefined1 auVar7 [32];
  undefined8 uVar8;
  undefined8 uVar9;
  undefined8 __src;
  ulong *__s1;
  char cVar10;
  undefined2 uVar11;
  int iVar12;
  undefined4 uVar13;
  uint uVar14;
  uint uVar15;
  uint uVar16;
  uint uVar17;
  uint uVar18;
  FILE *__stream;
  ulong uVar19;
  ulong uVar20;
  ulong uVar21;
  ulong uVar22;
  long lVar23;
  ulong uVar24;
  void *__ptr;
  void *__ptr_00;
  int *piVar25;
  istream *piVar26;
  long lVar27;
  int iVar28;
  double *pdVar29;
  undefined1 *puVar30;
  long lVar31;
  void *pvVar32;
  long lVar33;
  undefined4 *puVar34;
  char *pcVar35;
  undefined8 *puVar36;
  double *pdVar37;
  byte bVar38;
  double dVar39;
  double dVar40;
  double dVar41;
  undefined1 auVar42 [32];
  undefined1 auVar43 [32];
  undefined1 auVar44 [32];
  undefined1 auVar45 [16];
  undefined1 auVar46 [32];
  float fVar47;
  undefined1 auVar48 [32];
  undefined1 auVar49 [32];
  undefined1 auVar50 [32];
  undefined1 auVar51 [32];
  undefined1 auVar52 [32];
  undefined1 auVar53 [32];
  undefined1 auVar54 [32];
  long lVar55;
  undefined1 auVar56 [32];
  undefined1 auVar57 [32];
  undefined1 auVar58 [32];
  undefined1 auVar59 [32];
  undefined1 auVar60 [64];
  undefined1 auVar61 [32];
  undefined1 auVar62 [32];
  undefined1 *local_5e0;
  int local_5c0;
  long local_4d0;
  int local_4c8;
  int local_4c4 [3];
  undefined8 local_4b8;
  long *local_4b0;
  int *piStack_4a8;
  undefined1 local_4a0 [16];
  undefined8 local_490;
  undefined1 local_480 [16];
  undefined8 local_470;
  undefined1 local_460 [16];
  undefined8 local_450;
  string local_420 [32];
  void *local_400 [4];
  void *local_3e0 [2];
  long local_3d0;
  undefined8 local_3c0;
  undefined4 local_3b8;
  undefined8 *local_3a0;
  undefined8 *local_398;
  long local_390;
  ulong *local_380;
  undefined8 local_378;
  ulong local_370 [2];
  long *local_360 [2];
  long local_350 [2];
  timespec local_340;
  undefined8 local_330;
  undefined4 local_328 [10];
  undefined8 local_300;
  undefined8 uStack_2f8;
  undefined8 uStack_2f0;
  long lStack_2e8;
  undefined1 local_2e0 [16];
  undefined1 *local_2d0;
  undefined8 local_2c8;
  undefined1 local_2c0 [16];
  undefined1 local_2b0 [16];
  undefined8 local_2a0;
  undefined4 local_298;
  undefined8 local_290;
  undefined4 local_288;
  undefined1 local_280 [16];
  undefined8 local_270;
  undefined1 local_260 [16];
  ulong local_250;
  undefined4 local_248 [2];
  undefined4 local_240;
  undefined8 uStack_238;
  undefined8 uStack_230;
  undefined8 uStack_228;
  undefined8 local_220;
  undefined8 uStack_218;
  undefined8 uStack_210;
  undefined8 uStack_208;
  undefined1 local_200 [16];
  ulong *puStack_1f0;
  undefined8 uStack_1e8;
  ulong auStack_1e0 [4];
  long local_1c0;
  uint uStack_1bc;
  undefined4 local_1b8;
  undefined4 uStack_1b4;
  undefined8 uStack_1b0;
  undefined4 uStack_1a8;
  undefined4 uStack_1a4;
  undefined8 uStack_1a0;
  undefined8 uStack_198;
  undefined8 uStack_190;
  undefined1 auStack_188 [16];
  undefined1 local_178 [32];
  undefined1 local_158 [32];
  undefined8 local_138;
  undefined4 local_130;
  undefined1 local_128 [16];
  undefined1 local_118 [16];
  undefined8 local_108;
  undefined8 uStack_100;
  undefined4 local_f0 [2];
  undefined8 local_e8;
  undefined4 *local_e0;
  undefined4 *local_d8;
  undefined8 local_d0;
  undefined4 local_c0 [2];
  undefined8 local_b8;
  undefined4 *local_b0;
  undefined4 *local_a8;
  undefined8 local_a0;
  undefined8 local_98;
  undefined4 local_90;
  undefined8 local_88;
  undefined8 uStack_80;
  undefined8 uStack_78;
  undefined8 uStack_70;
  undefined8 local_68;
  undefined8 uStack_60;
  undefined8 local_58;
  
  bVar38 = 0;
  local_4d0 = param_2;
  local_4c4[0] = param_1;
  if (param_1 < 2) {
    fwrite(&LAB_0038e8f7_1,1,0x13c,stderr);
    return 1;
  }
  FUN_0013eae0(local_420,*(undefined8 *)(param_2 + 8));
  cVar10 = FUN_0017e3f0(local_420,"selftest");
  if ((cVar10 != '\0') && (2 < local_4c4[0])) {
    FUN_0013eae0((string *)&local_3c0,*(undefined8 *)(local_4d0 + 0x10));
    cVar10 = FUN_0017e3f0((string *)&local_3c0,&DAT_0038abb3);
    if ((cVar10 != '\0') && (local_4c4[0] == 4)) {
      FUN_0013eae0(&local_340,*(undefined8 *)(local_4d0 + 0x18));
      __stream = fopen((char *)local_340.tv_sec,"w");
      if (__stream == (FILE *)0x0) {
        FUN_0014ecf0((string *)local_260,"cannot write ",local_340.tv_sec,local_340.tv_nsec);
        fprintf(stderr,"halogen-npu: %s\n",local_260._0_8_);
                    /* WARNING: Subroutine does not return */
        std::__cxx11::string::_M_dispose((string *)local_260);
      }
      lVar27 = 0;
      auVar48._8_4_ = 0x80000000;
      auVar48._0_8_ = 0x8000000080000000;
      auVar48._12_4_ = 0x80000000;
      auVar48._16_4_ = 0x80000000;
      auVar48._20_4_ = 0x80000000;
      auVar48._24_4_ = 0x80000000;
      auVar48._28_4_ = 0x80000000;
      auVar42._8_4_ = 0x7f800000;
      auVar42._0_8_ = 0x7f8000007f800000;
      auVar42._12_4_ = 0x7f800000;
      auVar42._16_4_ = 0x7f800000;
      auVar42._20_4_ = 0x7f800000;
      auVar42._24_4_ = 0x7f800000;
      auVar42._28_4_ = 0x7f800000;
      auVar42 = vpsubd_avx2(auVar42,auVar48);
      do {
        auVar60 = ZEXT1664((undefined1  [16])0x0);
        iVar28 = (int)lVar27 << 0x18;
        iVar12 = 0;
        auVar62._4_4_ = iVar28;
        auVar62._0_4_ = iVar28;
        auVar62._8_4_ = iVar28;
        auVar62._12_4_ = iVar28;
        auVar62._16_4_ = iVar28;
        auVar62._20_4_ = iVar28;
        auVar62._24_4_ = iVar28;
        auVar62._28_4_ = iVar28;
        auVar61._8_4_ = 0x7fffffff;
        auVar61._0_8_ = 0x7fffffff7fffffff;
        auVar61._12_4_ = 0x7fffffff;
        auVar61._16_4_ = 0x7fffffff;
        auVar61._20_4_ = 0x7fffffff;
        auVar61._24_4_ = 0x7fffffff;
        auVar61._28_4_ = 0x7fffffff;
        auVar58._8_4_ = 0xffff;
        auVar58._0_8_ = 0xffff0000ffff;
        auVar58._12_4_ = 0xffff;
        auVar58._16_4_ = 0xffff;
        auVar58._20_4_ = 0xffff;
        auVar58._24_4_ = 0xffff;
        auVar58._28_4_ = 0xffff;
        auVar6 = ram0x00397020;
        do {
          lVar33 = auVar6._0_8_;
          auVar7._0_8_ = lVar33 + 4;
          lVar23 = auVar6._8_8_;
          auVar7._8_8_ = lVar23 + 4;
          lVar31 = auVar6._16_8_;
          auVar7._16_8_ = lVar31 + 4;
          lVar55 = auVar6._24_8_;
          auVar7._24_8_ = lVar55 + 4;
          auVar59._0_8_ = lVar33 + 8;
          auVar59._8_8_ = lVar23 + 8;
          auVar59._16_8_ = lVar31 + 8;
          auVar59._24_8_ = lVar55 + 8;
          auVar56._0_8_ = lVar33 + 0xc;
          auVar56._8_8_ = lVar23 + 0xc;
          auVar56._16_8_ = lVar31 + 0xc;
          auVar56._24_8_ = lVar55 + 0xc;
          iVar12 = iVar12 + 1;
          auVar43 = vperm2i128_avx2(auVar6,auVar7,0x20);
          auVar44 = vperm2i128_avx2(auVar6,auVar7,0x31);
          auVar57._8_4_ = 1;
          auVar57._0_8_ = 0x100000001;
          auVar57._12_4_ = 1;
          auVar57._16_4_ = 1;
          auVar57._20_4_ = 1;
          auVar57._24_4_ = 1;
          auVar57._28_4_ = 1;
          auVar44 = vpshufd_avx2(auVar44,0xd8);
          auVar43 = vpshufd_avx2(auVar43,0xd8);
          auVar51 = vperm2i128_avx2(auVar59,auVar56,0x20);
          auVar53 = vpunpcklqdq_avx2(auVar43,auVar44);
          auVar43 = vperm2i128_avx2(auVar59,auVar56,0x31);
          auVar44 = vpshufd_avx2(auVar51,0xd8);
          auVar43 = vpshufd_avx2(auVar43,0xd8);
          auVar52 = vpsllq_avx2(auVar6,1);
          auVar53 = vpor_avx2(auVar62,auVar53);
          auVar44 = vpunpcklqdq_avx2(auVar44,auVar43);
          auVar50 = vpsllq_avx2(auVar7,1);
          auVar6 = vpand_avx2(auVar53,auVar61);
          auVar51 = vpor_avx2(auVar62,auVar44);
          auVar49 = vpsllq_avx2(auVar59,1);
          auVar6 = vpsubd_avx2(auVar6,auVar48);
          auVar44 = vpand_avx2(auVar51,auVar61);
          auVar43 = vpcmpgtd_avx2(auVar6,auVar42);
          auVar6._0_8_ = lVar33 + 0x10;
          auVar6._8_8_ = lVar23 + 0x10;
          auVar6._16_8_ = lVar31 + 0x10;
          auVar6._24_8_ = lVar55 + 0x10;
          auVar44 = vpsubd_avx2(auVar44,auVar48);
          auVar46 = vpsllq_avx2(auVar56,1);
          auVar7 = vpcmpgtd_avx2(auVar44,auVar42);
          auVar44 = vpand_avx2(auVar58,auVar43);
          auVar43 = vpand_avx2(auVar58,auVar7);
          auVar44 = vpackusdw_avx2(auVar44,auVar43);
          auVar43 = vpsrld_avx2(auVar53,0x10);
          auVar43 = vpand_avx2(auVar43,auVar57);
          auVar59 = vpermq_avx2(auVar44,0xd8);
          auVar44._0_4_ = auVar43._0_4_ + auVar53._0_4_ + 0x7fff;
          auVar44._4_4_ = auVar43._4_4_ + auVar53._4_4_ + 0x7fff;
          auVar44._8_4_ = auVar43._8_4_ + auVar53._8_4_ + 0x7fff;
          auVar44._12_4_ = auVar43._12_4_ + auVar53._12_4_ + 0x7fff;
          auVar44._16_4_ = auVar43._16_4_ + auVar53._16_4_ + 0x7fff;
          auVar44._20_4_ = auVar43._20_4_ + auVar53._20_4_ + 0x7fff;
          auVar44._24_4_ = auVar43._24_4_ + auVar53._24_4_ + 0x7fff;
          auVar44._28_4_ = auVar43._28_4_ + auVar53._28_4_ + 0x7fff;
          auVar43 = vpsrld_avx2(auVar51,0x10);
          auVar7 = vpand_avx2(auVar43,auVar57);
          auVar44 = vpsrld_avx2(auVar44,0x10);
          auVar43._0_4_ = auVar7._0_4_ + auVar51._0_4_ + 0x7fff;
          auVar43._4_4_ = auVar7._4_4_ + auVar51._4_4_ + 0x7fff;
          auVar43._8_4_ = auVar7._8_4_ + auVar51._8_4_ + 0x7fff;
          auVar43._12_4_ = auVar7._12_4_ + auVar51._12_4_ + 0x7fff;
          auVar43._16_4_ = auVar7._16_4_ + auVar51._16_4_ + 0x7fff;
          auVar43._20_4_ = auVar7._20_4_ + auVar51._20_4_ + 0x7fff;
          auVar43._24_4_ = auVar7._24_4_ + auVar51._24_4_ + 0x7fff;
          auVar43._28_4_ = auVar7._28_4_ + auVar51._28_4_ + 0x7fff;
          auVar44 = vpand_avx2(auVar58,auVar44);
          auVar43 = vpsrld_avx2(auVar43,0x10);
          auVar43 = vpand_avx2(auVar58,auVar43);
          auVar44 = vpackusdw_avx2(auVar44,auVar43);
          auVar54._8_4_ = 0x7fc07fc0;
          auVar54._0_8_ = 0x7fc07fc07fc07fc0;
          auVar54._12_4_ = 0x7fc07fc0;
          auVar54._16_4_ = 0x7fc07fc0;
          auVar54._20_4_ = 0x7fc07fc0;
          auVar54._24_4_ = 0x7fc07fc0;
          auVar54._28_4_ = 0x7fc07fc0;
          auVar44 = vpermq_avx2(auVar44,0xd8);
          auVar44 = vpblendvb_avx2(auVar44,auVar54,auVar59);
          auVar43 = vpmovzxwd_avx2(auVar44._0_16_);
          auVar53._0_8_ = auVar52._0_8_ + 1;
          auVar53._8_8_ = auVar52._8_8_ + 1;
          auVar53._16_8_ = auVar52._16_8_ + 1;
          auVar53._24_8_ = auVar52._24_8_ + 1;
          auVar7 = vpmovzxdq_avx2(auVar43._0_16_);
          auVar56 = vpsrlq_avx2(auVar53,0x20);
          auVar59 = vpsrlq_avx2(auVar7,0x20);
          auVar51._0_8_ = auVar50._0_8_ + 1;
          auVar51._8_8_ = auVar50._8_8_ + 1;
          auVar51._16_8_ = auVar50._16_8_ + 1;
          auVar51._24_8_ = auVar50._24_8_ + 1;
          auVar59 = vpmuludq_avx2(auVar59,auVar53);
          auVar56 = vpmuludq_avx2(auVar56,auVar7);
          auVar43 = vpmovzxdq_avx2(auVar43._16_16_);
          auVar7 = vpmuludq_avx2(auVar53,auVar7);
          auVar53 = vpsrlq_avx2(auVar51,0x20);
          auVar44 = vpmovzxwd_avx2(auVar44._16_16_);
          auVar53 = vpmuludq_avx2(auVar53,auVar43);
          auVar50._0_8_ = auVar49._0_8_ + 1;
          auVar50._8_8_ = auVar49._8_8_ + 1;
          auVar50._16_8_ = auVar49._16_8_ + 1;
          auVar50._24_8_ = auVar49._24_8_ + 1;
          auVar49._0_8_ = auVar46._0_8_ + 1;
          auVar49._8_8_ = auVar46._8_8_ + 1;
          auVar49._16_8_ = auVar46._16_8_ + 1;
          auVar49._24_8_ = auVar46._24_8_ + 1;
          auVar46._0_8_ = auVar56._0_8_ + auVar59._0_8_;
          auVar46._8_8_ = auVar56._8_8_ + auVar59._8_8_;
          auVar46._16_8_ = auVar56._16_8_ + auVar59._16_8_;
          auVar46._24_8_ = auVar56._24_8_ + auVar59._24_8_;
          auVar56 = vpsrlq_avx2(auVar43,0x20);
          auVar57 = vpsllq_avx2(auVar46,0x20);
          auVar59 = vpmuludq_avx2(auVar51,auVar43);
          auVar43 = vpmuludq_avx2(auVar56,auVar51);
          auVar51 = vpsrlq_avx2(auVar50,0x20);
          auVar52._0_8_ = auVar53._0_8_ + auVar43._0_8_;
          auVar52._8_8_ = auVar53._8_8_ + auVar43._8_8_;
          auVar52._16_8_ = auVar53._16_8_ + auVar43._16_8_;
          auVar52._24_8_ = auVar53._24_8_ + auVar43._24_8_;
          auVar46 = vpsllq_avx2(auVar52,0x20);
          auVar43 = vpmovzxdq_avx2(auVar44._0_16_);
          auVar53 = vpsrlq_avx2(auVar43,0x20);
          auVar44 = vpmovzxdq_avx2(auVar44._16_16_);
          auVar56 = vpmuludq_avx2(auVar50,auVar43);
          auVar43 = vpmuludq_avx2(auVar51,auVar43);
          auVar53 = vpmuludq_avx2(auVar53,auVar50);
          auVar50 = vpsrlq_avx2(auVar44,0x20);
          auVar51 = vpmuludq_avx2(auVar49,auVar44);
          auVar3._0_8_ = auVar43._0_8_ + auVar53._0_8_;
          auVar3._8_8_ = auVar43._8_8_ + auVar53._8_8_;
          auVar3._16_8_ = auVar43._16_8_ + auVar53._16_8_;
          auVar3._24_8_ = auVar43._24_8_ + auVar53._24_8_;
          auVar43 = vpsrlq_avx2(auVar49,0x20);
          auVar44 = vpmuludq_avx2(auVar43,auVar44);
          auVar43 = vpmuludq_avx2(auVar50,auVar49);
          auVar53 = vpsllq_avx2(auVar3,0x20);
          auVar4._0_8_ = auVar44._0_8_ + auVar43._0_8_;
          auVar4._8_8_ = auVar44._8_8_ + auVar43._8_8_;
          auVar4._16_8_ = auVar44._16_8_ + auVar43._16_8_;
          auVar4._24_8_ = auVar44._24_8_ + auVar43._24_8_;
          auVar44 = vpsllq_avx2(auVar4,0x20);
          auVar2._0_8_ = auVar60._0_8_ +
                         auVar7._0_8_ + auVar57._0_8_ + auVar59._0_8_ + auVar46._0_8_ +
                         auVar56._0_8_ + auVar53._0_8_ + auVar51._0_8_ + auVar44._0_8_;
          auVar2._8_8_ = auVar60._8_8_ +
                         auVar7._8_8_ + auVar57._8_8_ + auVar59._8_8_ + auVar46._8_8_ +
                         auVar56._8_8_ + auVar53._8_8_ + auVar51._8_8_ + auVar44._8_8_;
          auVar5._16_8_ =
               auVar60._16_8_ +
               auVar7._16_8_ + auVar57._16_8_ + auVar59._16_8_ + auVar46._16_8_ +
               auVar56._16_8_ + auVar53._16_8_ + auVar51._16_8_ + auVar44._16_8_;
          auVar5._0_16_ = auVar2;
          auVar5._24_8_ =
               auVar60._24_8_ +
               auVar7._24_8_ + auVar57._24_8_ + auVar59._24_8_ + auVar46._24_8_ +
               auVar56._24_8_ + auVar53._24_8_ + auVar51._24_8_ + auVar44._24_8_;
          auVar60 = ZEXT3264(auVar5);
        } while (iVar12 != 0x100000);
        auVar2 = vpaddq_avx(auVar5._16_16_,auVar2);
        auVar45 = vpsrldq_avx(auVar2,8);
        auVar2 = vpaddq_avx(auVar2,auVar45);
        fprintf(__stream,"%llu %llu\n",lVar27,auVar2._0_8_);
        lVar27 = lVar27 + 1;
      } while (lVar27 != 0x100);
      fclose(__stream);
                    /* WARNING: Subroutine does not return */
      std::__cxx11::string::_M_dispose((string *)&local_340);
    }
    cVar10 = FUN_0017e3f0((string *)&local_3c0,&LAB_0038c638_1);
    if ((cVar10 != '\0') && (7 < local_4c4[0])) {
      uVar13 = 0x400;
      if (local_4c4[0] != 8) {
        uVar13 = __isoc23_strtol(*(undefined8 *)(local_4d0 + 0x40),0,10);
      }
      FUN_0013eae0(local_260,*(undefined8 *)(local_4d0 + 0x38));
      FUN_0013eae0(&local_340,*(undefined8 *)(local_4d0 + 0x30));
      FUN_0013eae0(local_360,*(undefined8 *)(local_4d0 + 0x28));
      FUN_0013eae0(&local_380,*(undefined8 *)(local_4d0 + 0x20));
      FUN_0013eae0((string *)&local_3a0,*(undefined8 *)(local_4d0 + 0x18));
      FUN_00156770((string *)&local_3a0,&local_380,local_360,&local_340,local_260,uVar13);
                    /* WARNING: Subroutine does not return */
      std::__cxx11::string::_M_dispose((string *)&local_3a0);
    }
    cVar10 = FUN_0017e3f0((string *)&local_3c0,"head");
    if ((cVar10 == '\0') || (local_4c4[0] < 6)) {
      cVar10 = FUN_0017e3f0((string *)&local_3c0,"resid");
      if ((cVar10 == '\0') || (local_4c4[0] != 4)) {
        cVar10 = FUN_0017e3f0((string *)&local_3c0,"hostx");
        if ((cVar10 == '\0') || (local_4c4[0] != 10)) {
          cVar10 = FUN_0017e3f0((string *)&local_3c0,&DAT_00389d24);
          if ((cVar10 != '\0') && (local_4c4[0] == 7)) {
            FUN_0013eae0(local_260,*(undefined8 *)(local_4d0 + 0x30));
            FUN_0013eae0(&local_340,*(undefined8 *)(local_4d0 + 0x28));
            FUN_0013eae0(local_360,*(undefined8 *)(local_4d0 + 0x20));
            FUN_0013eae0((string *)&local_380,*(undefined8 *)(local_4d0 + 0x18));
            FUN_001cea10((string *)&local_380,local_360,&local_340,local_260);
                    /* WARNING: Subroutine does not return */
            std::__cxx11::string::_M_dispose((string *)&local_380);
          }
          cVar10 = FUN_0017e3f0((string *)&local_3c0,&LAB_0038bc28);
          if ((cVar10 != '\0') && (local_4c4[0] == 4)) {
            FUN_0013eae0((string *)local_260,*(undefined8 *)(local_4d0 + 0x18));
            FUN_00159560((string *)local_260);
                    /* WARNING: Subroutine does not return */
            std::__cxx11::string::_M_dispose((string *)local_260);
          }
          fwrite(&LAB_0038e8f7_1,1,0x13c,stderr);
                    /* WARNING: Subroutine does not return */
          std::__cxx11::string::_M_dispose((string *)&local_3c0);
        }
        uVar14 = __isoc23_strtol(*(undefined8 *)(local_4d0 + 0x48),0,10);
        uVar15 = __isoc23_strtol(*(undefined8 *)(local_4d0 + 0x40),0,10);
        uVar16 = __isoc23_strtol(*(undefined8 *)(local_4d0 + 0x38),0,10);
        uVar17 = __isoc23_strtol(*(undefined8 *)(local_4d0 + 0x30),0,10);
        uVar18 = __isoc23_strtol(*(undefined8 *)(local_4d0 + 0x28),0,10);
        uVar13 = __isoc23_strtol(*(undefined8 *)(local_4d0 + 0x20),0,10);
        lVar27 = __isoc23_strtoll(*(undefined8 *)(local_4d0 + 0x18),0,10);
        uVar24 = lVar27 * 2 + 0x3fU & 0xffffffffffffffc0;
        pvVar32 = (void *)aligned_alloc(0x40,lVar27 * 4 + 0x3fU & 0xffffffffffffffc0);
        __ptr = (void *)aligned_alloc(0x40,uVar24);
        __ptr_00 = (void *)aligned_alloc(0x40);
        uVar19 = 0x9e3779b97f4a7c15;
        for (lVar33 = 0; lVar27 != lVar33; lVar33 = lVar33 + 1) {
          uVar19 = uVar19 * 0x5851f42d4c957f2d + 0x14057b7ef767814f;
          *(float *)((long)pvVar32 + lVar33 * 4) =
               ((float)(int)((uVar19 >> 0x21) % 0x30d41) - 100000.0) / 997.0;
          fVar47 = ((float)(int)((uVar19 >> 0x28) % 0x4e21) - 10000.0) / 313.0;
          uVar11 = (undefined2)((int)fVar47 + 0x7fff + ((uint)fVar47 >> 0x10 & 1) >> 0x10);
          if (0x7f800000 < (uint)ABS(fVar47)) {
            uVar11 = 0x7fc0;
          }
          *(undefined2 *)((long)__ptr + lVar33 * 2) = uVar11;
          *(undefined2 *)((long)__ptr_00 + lVar33 * 2) = 0;
        }
        FUN_001c47e0();
        if (uVar16 != 0) {
          FUN_001c09c0();
        }
        pdVar37 = (double *)0x0;
        pdVar29 = (double *)0x0;
        local_4c8 = uVar17 + 1;
        local_5c0 = 0;
        do {
          if (local_4c8 < local_5c0) {
            FUN_001c09c0(local_260,0);
            FUN_0013f950(0,pdVar29);
            FUN_0013f950(0,pdVar37);
            dVar39 = *(double *)(((ulong)((long)pdVar29 >> 3) >> 1) * 8);
            printf("hostx n %zu th %d gap %d spin %d nt %d flush %d reps %d: step med %.3f min %.3f p90 %.3f max %.3f ms (%.1f GB/s at 12 B an element); flush med %.3f ms\n"
                   ,dVar39,_DAT_00000000,
                   *(undefined8 *)(((ulong)(((long)pdVar29 >> 3) * 9) / 10) * 8),pdVar29[-1],
                   (((double)lVar27 * 12.0) / dVar39) / 1000000.0,
                   *(undefined8 *)(((ulong)((long)pdVar37 >> 3) >> 1) * 8),lVar27,uVar13,
                   (ulong)uVar18,(ulong)uVar16,(ulong)uVar15,(ulong)uVar14,(ulong)uVar17);
            free(pvVar32);
            free(__ptr);
            free(__ptr_00);
                    /* WARNING: Subroutine does not return */
            operator_delete((void *)0x0,0);
          }
          if (0 < (int)uVar18) {
            local_340.tv_nsec = ((long)(int)uVar18 % 1000000) * 1000;
            local_340.tv_sec = (long)(int)uVar18 / 1000000;
            do {
              iVar28 = nanosleep(&local_340,&local_340);
              if (iVar28 != -1) break;
              piVar25 = __errno_location();
            } while (*piVar25 == 4);
          }
          dVar39 = (double)FUN_001ae410();
          if (uVar14 != 0) {
            for (uVar19 = 0; uVar19 < uVar24; uVar19 = uVar19 + 0x40) {
              clflush(*(undefined1 *)((long)__ptr + uVar19));
              clflush(*(undefined1 *)((long)__ptr_00 + uVar19));
            }
          }
          dVar40 = (double)FUN_001ae410();
          FUN_001c1210(local_260,pvVar32,__ptr,__ptr_00,lVar27,uVar15 != 0);
          dVar41 = (double)FUN_001ae410();
          if (1 < local_5c0) {
            if (pdVar29 == (double *)0x0) {
                    /* WARNING: Subroutine does not return */
              operator_new(8);
            }
            *pdVar29 = dVar41 - dVar40;
            pdVar29 = pdVar29 + 1;
            if (pdVar37 == (double *)0x0) {
                    /* WARNING: Subroutine does not return */
              operator_new(8);
            }
            *pdVar37 = dVar40 - dVar39;
            pdVar37 = pdVar37 + 1;
          }
          local_5c0 = local_5c0 + 1;
        } while( true );
      }
      lVar23 = __isoc23_strtoll(*(undefined8 *)(local_4d0 + 0x18),0,10);
      FUN_0013e3d0(local_400,lVar23);
      FUN_0013e3d0(local_3e0,lVar23);
      local_3a0 = (undefined8 *)0x0;
      local_390 = 0;
      FUN_0017fd10(&local_3a0,lVar23,local_260);
      lVar27 = local_390;
      puVar36 = local_3a0;
      local_380 = (ulong *)0x0;
      local_370[0] = 0;
      FUN_0017fd10(&local_380,lVar23,local_260);
      uVar24 = local_370[0];
      __s1 = local_380;
      local_360[0] = (long *)0x0;
      local_350[0] = 0;
      FUN_0017fd10(local_360,lVar23,local_260);
      lVar33 = local_350[0];
      plVar1 = local_360[0];
      uVar19 = 0x9e3779b97f4a7c15;
      for (lVar31 = 0; lVar23 != lVar31; lVar31 = lVar31 + 1) {
        uVar19 = uVar19 * 0x5851f42d4c957f2d + 0x14057b7ef767814f;
        fVar47 = ((float)(int)((uVar19 >> 0x21) % 0x30d41) - 100000.0) / 997.0;
        *(float *)((long)local_3e0[0] + lVar31 * 4) = fVar47;
        *(float *)((long)local_400[0] + lVar31 * 4) = fVar47;
        fVar47 = ((float)(int)((uVar19 >> 0x28) % 0x4e21) - 10000.0) / 313.0;
        uVar11 = (undefined2)((int)fVar47 + 0x7fff + ((uint)fVar47 >> 0x10 & 1) >> 0x10);
        if (0x7f800000 < (uint)ABS(fVar47)) {
          uVar11 = 0x7fc0;
        }
        *(undefined2 *)((long)puVar36 + lVar31 * 2) = uVar11;
      }
      FUN_001c47e0(&local_340,1);
      FUN_001c47e0(local_260,8);
      dVar39 = (double)FUN_001ae410();
      FUN_001c1210(&local_340,local_400[0],puVar36,__s1,lVar23,0);
      dVar40 = (double)FUN_001ae410();
      FUN_001c1210(local_260,local_3e0[0],puVar36,plVar1,lVar23,0);
      dVar41 = (double)FUN_001ae410();
      iVar28 = memcmp(local_400[0],local_3e0[0],lVar23 * 4);
      if (iVar28 == 0) {
        iVar28 = memcmp(__s1,plVar1,lVar23 * 2);
        pcVar35 = "BITWISE";
        if (iVar28 == 0) goto LAB_00171537;
      }
      pcVar35 = "DIFFERS";
LAB_00171537:
      printf("resid_copy n %zu: 1 thread %.3f ms, 8 threads %.3f ms, %s\n",dVar40 - dVar39,
             dVar41 - dVar40,lVar23,pcVar35);
      FUN_001c0500(local_260);
      FUN_001c0500(&local_340);
      if (plVar1 != (long *)0x0) {
                    /* WARNING: Subroutine does not return */
        operator_delete(plVar1,lVar33 - (long)plVar1);
      }
      if (__s1 == (ulong *)0x0) {
        if (puVar36 == (undefined8 *)0x0) {
                    /* WARNING: Subroutine does not return */
          operator_delete(local_3e0[0],local_3d0 - (long)local_3e0[0]);
        }
                    /* WARNING: Subroutine does not return */
        operator_delete(puVar36,lVar27 - (long)puVar36);
      }
                    /* WARNING: Subroutine does not return */
      operator_delete(__s1,uVar24 - (long)__s1);
    }
    local_5c0 = 0x400;
    if (local_4c4[0] != 6) {
      local_5c0 = __isoc23_strtol(*(undefined8 *)(local_4d0 + 0x30),0,10);
    }
    FUN_0013eae0(&local_340,*(undefined8 *)(local_4d0 + 0x28));
    FUN_0013eae0(local_360,*(undefined8 *)(local_4d0 + 0x20));
    FUN_0013eae0(&local_380,*(undefined8 *)(local_4d0 + 0x18));
    local_470 = 0;
    local_480 = (undefined1  [16])0x0;
    cVar10 = FUN_00148810(&local_380,local_480);
    uVar8 = local_480._0_8_;
    if (cVar10 != '\0') {
      local_450 = 0;
      uVar9 = local_480._8_8_;
      local_460 = (undefined1  [16])0x0;
      cVar10 = FUN_00148810(local_360,local_460);
      __src = local_460._0_8_;
      if (cVar10 != '\0') {
        uVar19 = uVar9 - uVar8;
        uVar24 = (long)local_5c0 * 4;
        uVar20 = uVar19 / uVar24;
        if (uVar19 % uVar24 == 0) {
          uVar21 = local_460._8_8_ - local_460._0_8_;
          uVar22 = uVar21 / uVar24;
          if (uVar21 % uVar24 == 0) {
            FUN_0013e3d0(local_400,uVar20 * (long)local_5c0);
            FUN_0013e3d0(local_3e0,(long)local_5c0 * uVar22);
            FUN_0013e3d0(&local_3a0,uVar20 * uVar22);
            memcpy(local_400[0],(void *)uVar8,uVar19);
            memcpy(local_3e0[0],(void *)__src,uVar21);
            lVar27 = 0;
            for (uVar19 = 0; uVar20 != uVar19; uVar19 = uVar19 + 1) {
              pvVar32 = local_3e0[0];
              for (uVar21 = 0; uVar22 != uVar21; uVar21 = uVar21 + 1) {
                uVar13 = FUN_001c3b20(local_400[0],pvVar32,local_5c0);
                *(undefined4 *)((long)local_3a0 + uVar21 * 4 + lVar27 * 4) = uVar13;
                pvVar32 = (void *)((long)pvVar32 + uVar24);
              }
              lVar27 = lVar27 + uVar22;
              local_400[0] = (void *)((long)local_400[0] + uVar24);
            }
            cVar10 = FUN_0013ddb0(local_340.tv_sec,local_3a0,(long)local_398 - (long)local_3a0);
            if (cVar10 == '\0') {
              FUN_0014ecf0((string *)local_260,"cannot write ",local_340.tv_sec,local_340.tv_nsec);
              fprintf(stderr,"halogen-npu: %s\n",local_260._0_8_);
                    /* WARNING: Subroutine does not return */
              std::__cxx11::string::_M_dispose((string *)local_260);
            }
            if (local_3a0 == (undefined8 *)0x0) {
                    /* WARNING: Subroutine does not return */
              operator_delete(local_3e0[0],local_3d0 - (long)local_3e0[0]);
            }
                    /* WARNING: Subroutine does not return */
            operator_delete(local_3a0,local_390 - (long)local_3a0);
          }
        }
        FUN_0013eae0(local_260,"sizes do not match H");
        fprintf(stderr,"halogen-npu: %s\n",local_260._0_8_);
        goto LAB_00171037;
      }
    }
    FUN_0013eae0(local_260,"cannot read hn or rows");
    fprintf(stderr,"halogen-npu: %s\n",local_260._0_8_);
LAB_00171037:
                    /* WARNING: Subroutine does not return */
    std::__cxx11::string::_M_dispose((string *)local_260);
  }
  cVar10 = FUN_0017e3f0(local_420,&LAB_0038aca8);
  if (cVar10 != '\0') {
    FUN_001d4ed0(local_4c4[0] + -2,local_4d0 + 0x10);
    goto LAB_00170cf4;
  }
  cVar10 = FUN_0017e3f0(local_420,"probe");
  if ((cVar10 != '\0') || (cVar10 = FUN_0017e3f0(local_420,"convert"), cVar10 != '\0')) {
    FUN_0022f180(local_4c4[0] + -1,local_4d0 + 8);
    goto LAB_00170cf4;
  }
  cVar10 = FUN_0017e3f0(local_420,&DAT_00389d35);
  if (cVar10 != '\0') {
    FUN_0028b0c0(local_4c4[0] + -2,local_4d0 + 0x10);
    goto LAB_00170cf4;
  }
  cVar10 = FUN_0017e3f0(local_420,&DAT_00389bc0);
  if (cVar10 != '\0') {
    FUN_00316af0(local_4c4[0] + -2,local_4d0 + 0x10);
    goto LAB_00170cf4;
  }
  cVar10 = FUN_0017e3f0(local_420,"gemma");
  if (cVar10 != '\0') {
    FUN_002e2b50(local_4c4[0] + -2,local_4d0 + 0x10);
    goto LAB_00170cf4;
  }
  cVar10 = FUN_0017e3f0(local_420,&DAT_00389110);
  if (cVar10 != '\0') {
    puVar30 = &LAB_0038ae38_1;
    if (2 < local_4c4[0]) {
      puVar30 = *(undefined1 **)(local_4d0 + 0x10);
    }
    FUN_0013eae0(local_260,puVar30);
    cVar10 = FUN_0017e3f0(local_260,"decode");
    lVar27 = local_4d0 + 0x10;
    iVar28 = local_4c4[0] + -2;
    if (cVar10 == '\0') {
      cVar10 = FUN_0017e3f0(local_260,"encode");
      if (cVar10 == '\0') {
        FUN_00387730(iVar28,lVar27);
      }
      else {
        FUN_00384a80();
      }
    }
    else {
      FUN_00385b50(iVar28,lVar27);
    }
LAB_001708e2:
                    /* WARNING: Subroutine does not return */
    std::__cxx11::string::_M_dispose((string *)local_260);
  }
  cVar10 = FUN_0017e3f0(local_420,&LAB_0038d076_1);
  if (cVar10 == '\0') {
    cVar10 = FUN_0017e3f0(local_420,&LAB_0038d076_1);
    if ((cVar10 != '\0') && (2 < local_4c4[0])) {
      FUN_0013eae0((string *)local_260,*(undefined8 *)(local_4d0 + 0x10));
      FUN_0017e3f0((string *)local_260,"gelucheck");
                    /* WARNING: Subroutine does not return */
      std::__cxx11::string::_M_dispose((string *)local_260);
    }
  }
  else if (2 < local_4c4[0]) {
    FUN_0013eae0((string *)local_260,*(undefined8 *)(local_4d0 + 0x10));
    FUN_0017e3f0((string *)local_260,"encode");
                    /* WARNING: Subroutine does not return */
    std::__cxx11::string::_M_dispose((string *)local_260);
  }
  cVar10 = FUN_0017e3f0(local_420,"replay");
  local_4b0 = &local_4d0;
  piStack_4a8 = local_4c4;
  if (cVar10 == '\0') {
    local_490 = 0;
    local_4b8 = 0;
    local_4a0 = (undefined1  [16])0x0;
    cVar10 = FUN_0017e3f0(local_420,"serve");
    if (cVar10 == '\0') {
LAB_00171a05:
      cVar10 = FUN_0017e3f0(local_420,"listen");
      if ((cVar10 == '\0') || (local_4c4[0] < 5)) goto LAB_001721b8;
      cVar10 = FUN_00149d30(&local_4b0,1,local_4a0);
      uVar8 = local_4b8;
      if (cVar10 != '\0') {
        FUN_0013eae0((string *)local_400);
        local_250 = 0;
        local_260 = (undefined1  [16])0x0;
        puVar34 = local_248;
        for (lVar27 = 10; lVar27 != 0; lVar27 = lVar27 + -1) {
          *puVar34 = 0;
          puVar34 = puVar34 + (ulong)bVar38 * -2 + 1;
        }
        puStack_1f0 = auStack_1e0;
        uStack_1e8 = 0;
        local_1c0 = 0;
        auStack_1e0[0] = auStack_1e0[0] & 0xffffffffffffff00;
        local_1b8 = 0;
        uStack_1b0 = 0;
        uStack_1a8 = 0;
        uStack_190 = 0;
        local_220 = 0;
        uStack_218 = 0;
        uStack_210 = 0;
        uStack_208 = 0;
        local_200 = (undefined1  [16])0x0;
        auStack_1e0[2] = 0;
        auStack_1e0[3] = 0;
        uStack_1a0 = 0;
        uStack_198 = 0;
        cVar10 = FUN_0016c310(local_260,local_4a0,uVar8,4);
        if (cVar10 == '\0') {
          FUN_0016d740(local_260);
                    /* WARNING: Subroutine does not return */
          std::__cxx11::string::_M_dispose((string *)local_400);
        }
        auVar45._8_8_ = 0;
        auVar45._0_8_ = local_460._8_8_;
        local_460 = auVar45 << 0x40;
                    /* WARNING: Subroutine does not return */
        operator_new(0xf0);
      }
LAB_00172021:
      fwrite(&LAB_0038e8f7_1,1,0x13c,stderr);
    }
    else {
      if (local_4c4[0] < 4) goto LAB_001721b8;
      cVar10 = FUN_00149d30(&local_4b0,0,local_4a0,&local_4b8);
      if (cVar10 == '\0') goto LAB_00172021;
      local_330 = 0;
      local_340.tv_sec = 0;
      local_340.tv_nsec = 0;
      puVar34 = local_328;
      for (lVar27 = 10; lVar27 != 0; lVar27 = lVar27 + -1) {
        *puVar34 = 0;
        puVar34 = puVar34 + (ulong)bVar38 * -2 + 1;
      }
      local_2d0 = local_2c0;
      local_2c8 = 0;
      local_298 = 0;
      local_290 = 0;
      local_288 = 0;
      local_2c0[0] = 0;
      local_2a0 = 0;
      local_270 = 0;
      local_300 = 0;
      uStack_2f8 = 0;
      uStack_2f0 = 0;
      lStack_2e8 = 0;
      local_2e0 = (undefined1  [16])0x0;
      local_2b0 = (undefined1  [16])0x0;
      local_280 = (undefined1  [16])0x0;
      cVar10 = FUN_0016c310(&local_340,local_4a0,local_4b8);
      if (cVar10 != '\0') {
        if (local_340.tv_nsec == local_340.tv_sec) {
          pcVar35 = "READY embedding\n";
          if (lStack_2e8 != 0) {
            pcVar35 = "READY image\n";
          }
          printf(pcVar35);
        }
        else {
          lVar27 = *(long *)local_340.tv_sec;
          FUN_001c2190(*(undefined4 *)(lVar27 + 0x170));
          printf("READY hidden %d layers %d order %s threads %d wait_ms %d op %d%s\n",
                 (ulong)*(uint *)(lVar27 + 0x128),(ulong)*(uint *)(lVar27 + 300));
        }
        fflush(stdout);
        local_130 = 0;
        local_f0[0] = 0;
        local_e8 = 0;
        local_e0 = local_f0;
        local_1b8 = 0;
        uStack_1b4 = 0;
        uStack_1b0 = 0;
        uStack_1a8 = 0;
        uStack_1a4 = 0;
        uStack_1a0 = 0;
        local_d0 = 0;
        local_c0[0] = 0;
        local_b0 = local_c0;
        uStack_198 = 0;
        uStack_190 = 0;
        auStack_188 = (undefined1  [16])0x0;
        local_178 = ZEXT1632(ZEXT816(0));
        local_260._8_8_ = 0;
        local_260._0_8_ = &local_250;
        local_240 = 0;
        auStack_1e0[3] = 0;
        local_250 = local_250 & 0xffffffffffffff00;
        local_1c0 = (ulong)uStack_1bc << 0x20;
        local_138 = 0;
        local_b8 = 0;
        local_a0 = 0;
        local_98 = 0;
        local_90 = 0;
        uStack_238 = 0;
        uStack_230 = 0;
        uStack_228 = 0;
        local_220 = 0;
        uStack_218 = 0;
        uStack_210 = 0;
        uStack_208 = 0;
        local_200 = ZEXT816(0);
        puStack_1f0 = (ulong *)0x0;
        uStack_1e8 = 0;
        auStack_1e0[0] = 0;
        auStack_1e0[1] = 0;
        auStack_1e0[2] = 0;
        local_128 = (undefined1  [16])0x0;
        local_118 = (undefined1  [16])0x0;
        local_108 = 0;
        uStack_100 = 0;
        local_88 = 0;
        uStack_80 = 0;
        uStack_78 = 0;
        uStack_70 = 0;
        local_68 = 0;
        uStack_60 = 0;
        local_58 = 0;
        local_3c0 = 0x100000000;
        local_380 = local_370;
        local_3b8 = 0;
        local_378 = 0;
        local_370[0] = local_370[0] & 0xffffffffffffff00;
        local_158 = local_178;
        local_d8 = local_e0;
        local_a8 = local_b0;
        while( true ) {
          plVar1 = *(long **)(&DAT_003e7910 + *(long *)(std::cin + -0x18));
          if (plVar1 == (long *)0x0) {
            std::__throw_bad_cast();
                    /* WARNING: Subroutine does not return */
            std::__cxx11::string::_M_dispose((string *)&local_380);
          }
          if ((char)plVar1[7] == '\0') {
            std::ctype<char>::_M_widen_init();
            cVar10 = '\n';
            if (*(code **)(*plVar1 + 0x30) != std::ctype<char>::do_widen) {
              cVar10 = (**(code **)(*plVar1 + 0x30))(plVar1,10);
            }
          }
          else {
            cVar10 = *(char *)((long)plVar1 + 0x43);
          }
          piVar26 = std::getline<char,std::char_traits<char>,std::allocator<char>>
                              ((istream *)&std::cin,(string *)&local_380,cVar10);
          if (((byte)piVar26[*(long *)(*(long *)piVar26 + -0x18) + 0x20] & 5) != 0) break;
          FUN_0014c470(&local_3a0,(string *)&local_380);
          if (local_398 == local_3a0) {
            FUN_0017c390(&local_3a0);
          }
          else {
            if ((local_3a0[1] == 4) && (*(int *)*local_3a0 == 0x54495551)) {
              FUN_0017c390(&local_3a0);
              break;
            }
            FUN_00166e20(local_360,&local_340,&local_3a0,local_260,&local_3c0);
            fputs((char *)local_360[0],stdout);
            fputc(10,stdout);
            fflush(stdout);
            puVar36 = local_3a0;
            if (local_360[0] != local_350) {
                    /* WARNING: Subroutine does not return */
              operator_delete(local_360[0],local_350[0] + 1);
            }
            for (; local_398 != puVar36; puVar36 = puVar36 + 4) {
              if ((undefined8 *)*puVar36 != puVar36 + 2) {
                    /* WARNING: Subroutine does not return */
                operator_delete((undefined8 *)*puVar36,puVar36[2] + 1);
              }
            }
            if (local_3a0 != (undefined8 *)0x0) {
                    /* WARNING: Subroutine does not return */
              operator_delete(local_3a0,local_390 - (long)local_3a0);
            }
          }
        }
                    /* WARNING: Subroutine does not return */
        std::__cxx11::string::_M_dispose((string *)&local_380);
      }
      FUN_0016d740(&local_340);
    }
  }
  else {
    if (2 < local_4c4[0]) {
      iVar28 = local_4c4[0];
      if (local_4c4[0] != 3) {
        iVar28 = __isoc23_strtol(*(undefined8 *)(local_4d0 + 0x18),0,10);
      }
      FUN_0013eae0(local_260,*(undefined8 *)(local_4d0 + 0x10));
      FUN_0016e020(local_260,iVar28);
      goto LAB_001708e2;
    }
    local_490 = 0;
    local_4b8 = 0;
    local_4a0 = (undefined1  [16])0x0;
    cVar10 = FUN_0017e3f0(local_420,"serve");
    if (cVar10 == '\0') goto LAB_00171a05;
LAB_001721b8:
    fwrite(&LAB_0038e8f7_1,1,0x13c,stderr);
  }
  local_5e0 = local_4a0;
  FUN_001827e0(local_5e0);
LAB_00170cf4:
                    /* WARNING: Subroutine does not return */
  std::__cxx11::string::_M_dispose(local_420);
}

