{ lib, rustPlatform }:

rustPlatform.buildRustPackage {
  pname = "cold-cathode";
  version = "0.1.0";

  src = lib.sourceByRegex ../.. [
    "^Cargo\\.(toml|lock)$"
    "^crates(/.*)?$"
  ];

  cargoHash = "sha256-gP36D92DSLcdLalenspR1tkGnGHtySTbogc1KKOOeYw=";

  meta.mainProgram = "cold-cathode";
}
