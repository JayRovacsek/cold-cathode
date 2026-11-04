{ lib, rustPlatform }:

rustPlatform.buildRustPackage {
  pname = "cold-cathode";
  version = "0.1.0";

  src = lib.sourceByRegex ../.. [
    "^Cargo\\.(toml|lock)$"
    "^crates(/.*)?$"
  ];

  cargoHash = "sha256-Z5dDZu+lBtpPrjIFyobzxacxYoz4eH+rT2hM3MITguo=";

  meta.mainProgram = "cold-cathode";
}
