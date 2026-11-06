{
  lib,
  rustPlatform,
  rocmPackages,
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

  meta.mainProgram = "cold-cathode";
}
