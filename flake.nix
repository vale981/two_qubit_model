{
  description = "Operators for a general model of two interacting qubits coupled to two baths.";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    utils.url = "github:vale981/hiro-flake-utils";
  };

  outputs = { self, utils, nixpkgs, ... }:
    (utils.lib.poetry2nixWrapper nixpkgs {
      name = "two_qubit_model";
      shellPackages = pkgs: with pkgs; [ black pyright ];
      python = pkgs: pkgs.python39;
      shellOverride = (oldAttrs: {
        shellHook = ''
                    export PYTHONPATH=/home/hiro/src/hops/:$PYTHONPATH
                    export PYTHONPATH=/home/hiro/src/hopsflow/:$PYTHONPATH
                    export PYTHONPATH=/home/hiro/src/stocproc/:$PYTHONPATH
                    '';
      });
      poetryArgs = {
        projectDir = ./.;
      };
    });
}
