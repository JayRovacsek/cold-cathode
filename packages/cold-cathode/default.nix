{ lib, rustPlatform }:

rustPlatform.buildRustPackage {
  pname = "cold-cathode";
  version = "0.1.0";

  src = lib.sourceByRegex ../.. [
    "^Cargo\\.(toml|lock)$"
    "^crates(/.*)?$"
  ];

  cargoHash = "sha256-kJgt08OfjDqoblquhq7P9hvMNa/APoFcNmNpgyTNBhM=";

  meta.mainProgram = "cold-cathode";
}
