{ lib, rustPlatform }:

rustPlatform.buildRustPackage {
  pname = "cold-cathode";
  version = "0.1.0";

  src = lib.sourceByRegex ../.. [
    "^Cargo\\.(toml|lock)$"
    "^crates(/.*)?$"
  ];

  cargoHash = "sha256-B0SvDFXn9AW2gv0rfIuDIaeybaRQw7pgXIdjeR4Ow/4=";

  meta.mainProgram = "cold-cathode";
}
