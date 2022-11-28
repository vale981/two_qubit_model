{
  description = "Operators for a general model of two interacting qubits coupled to two baths.";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    utils.url = "github:vale981/hiro-flake-utils";
    poetry2nix.url = "github:nix-community/poetry2nix";
    utils.inputs.poetry.follows = "poetry2nix";
    utils.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, utils, nixpkgs, ... }:
    (utils.lib.poetry2nixWrapper nixpkgs {
      name = "two_qubit_model";
      shellPackages = pkgs: with pkgs; [ black pyright ];
      python = pkgs: pkgs.python39Full;
      # shellOverride = (oldAttrs: {
      #   shellHook = ''
      #               export PYTHONPATH=/home/hiro/src/hops/:$PYTHONPATH
      #               export PYTHONPATH=/home/hiro/src/hopsflow/:$PYTHONPATH
      #               export PYTHONPATH=/home/hiro/src/stocproc/:$PYTHONPATH
      #               '';
      # });
      poetryArgs = {
        projectDir = ./.;
      };
    });
}
