{
  description = "Cold Cathode - Rust reimplementation of the halogen-flash-server binary blobs";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/3ec636362e1816bc2f0888fa0fa2c69f1bc9e504";

  outputs =
    { self, nixpkgs }:
    let
      pkgs = import nixpkgs {
        system = "x86_64-linux";
        config = {
          allowUnfree = true;
        };
      };
    in
    {
      packages.x86_64-linux = {
        halogen-flash-server = pkgs.callPackage ./packages/halogen-flash-server/default.nix { };
        cold-cathode = pkgs.callPackage ./packages/cold-cathode/default.nix { };
      };

      devShells.x86_64-linux.default = pkgs.mkShell {
        packages = [
          pkgs.cargo
          pkgs.cargo-nextest
          pkgs.clippy
          pkgs.nixfmt-rfc-style
          pkgs.rustc
          pkgs.rustfmt
          self.packages.x86_64-linux.halogen-flash-server
        ];
      };
    };
}
