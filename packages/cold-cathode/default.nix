{
  lib,
  rustPlatform,
  rocmPackages,
  binutils,
  halogen-flash-server,
}:

rustPlatform.buildRustPackage {
  pname = "cold-cathode";
  version = "0.1.0";

  src = lib.sourceByRegex ../.. [
    "^Cargo\\.(toml|lock)$"
    "^crates(/.*)?$"
  ];

  cargoHash = "sha256-Z5dDZu+lBtpPrjIFyobzxacxYoz4eH+rT2hM3MITguo=";

  # The HIP libraries the port dlopens at runtime (`libamdhip64.so.7`
  # from clr, `libhipblaslt.so.1` from hipblaslt): mirrors the
  # reference binary's RUNPATH (clr-7.2.4 + hipblaslt-7.2.4, see
  # docs/analysis/evidence/flash_serve/npu/hip-imported-symbols.json)
  # so the nix-built binary resolves them; buildInputs do not affect
  # cargoHash. The explicit rpath flag bakes both library directories
  # into every binary's RUNPATH the way the reference binary's RUNPATH
  # does — buildInputs alone only affect linking via pkg-config-style
  # flags. dontPatchELF is required: stdenv's patchelf setup hook runs
  # `patchelf --shrink-rpath` in the fixup phase, which keeps only the
  # RUNPATH entries that satisfy a DT_NEEDED entry. The ROCm libraries
  # are dlopened, never linked, so their entries were dropped (the
  # previous build's RUNPATH shrank to just glibc + gcc, dlopen failed
  # on ROCm hosts, and the port logged `halogen: backend: cpu`).
  buildInputs = [
    rocmPackages.clr
    rocmPackages.hipblaslt
  ];
  RUSTFLAGS = [
    "-Clink-arg=-Wl,-rpath,${rocmPackages.hipblaslt}/lib:${rocmPackages.clr}/lib"
  ];
  dontPatchELF = true;

  # The GPU code objects the port loads through the hipModule* API: the
  # reference binary's `.hip_fatbin` section, extracted at build time
  # into this output (docs/analysis/evidence/flash_serve/engine/
  # fatbin-inventory.md). `preBuild` exports the path so `src/fatbin.rs`
  # bakes it in via `option_env!("HALOGEN_FATBIN")` (a plain derivation
  # attribute would reach rustc as the literal `$out/...`, since shell
  # variables are not re-expanded inside env var values); a runtime
  # `HALOGEN_FATBIN` still wins. The `/dev/null` output argument
  # discards objcopy's copied object: the store input is read-only, so
  # objcopy cannot write its default `<input>.o` next to it.
  preBuild = ''
    export HALOGEN_FATBIN=$out/share/cold-cathode/flash_serve.hipfb
  '';
  postInstall = ''
    mkdir -p $out/share/cold-cathode
    ${binutils}/bin/objcopy \
      --dump-section .hip_fatbin=$out/share/cold-cathode/flash_serve.hipfb \
      ${halogen-flash-server}/bin/flash_serve /dev/null
  '';

  meta.mainProgram = "cold-cathode";
}
