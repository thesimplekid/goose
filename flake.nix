{
  description = "Could not find `protoc`. If `protoc`";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };
        
        rustNightly = pkgs.rust-bin.nightly.latest.default.override {
          extensions = [ "rust-src" ];
        };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            rustNightly
            pkg-config
            openssl
            nodejs_20
            nodePackages.typescript
            nodePackages.typescript-language-server
            protobuf
          ];

          RUST_SRC_PATH = "${rustNightly}/lib/rustlib/src/rust/library";
        };
      }
    );
}
