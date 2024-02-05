{
  description = "picogpt-rs";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python3.withPackages
          (ps: with ps; [ numpy regex requests tqdm fire tensorflow ]);

      in {
        devShells.default =
          pkgs.mkShell { packages = with pkgs; [ rustc cargo clippy python ]; };
      });
}
