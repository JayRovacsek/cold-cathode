{ lib, rustPlatform, rocmPackages }:

rustPlatform.buildRustPackage {
  pname = "cold-cathode";
  version = "0.1.0";

  src = lib.sourceByRegex ../.. [
    "^Cargo\\.(toml|lock)$"
    "^crates(/.*)?$"
  ];

  cargoHash = "sha256-Z5dDZu+lBtpPrjIFyobzxacxYoz4eH+rT2hM3MITguo=";

  # The HIP runtime the port dlopens at runtime (`libamdhip64.so.7`):
  # mirrors the reference binary's RUNPATH (clr-7.2.4, see
  # docs/analysis/evidence/flash_serve/npu/hip-imported-symbols.json)
  # so the nix-built binary resolves it; buildInputs do not affect
  # cargoHash.
  buildInputs = [ rocmPackages.clr ];

  meta.mainProgram = "cold-cathode";
}
