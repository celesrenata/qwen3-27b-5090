{
  description = "Qwen3.8-27B inference engine for one RTX 5090 (SM120)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = inputs@{ self, nixpkgs, ... }:
    let
      lib = nixpkgs.lib;
      system = "x86_64-linux";

      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          nvidia.acceptLicense = true;
        };
        overlays = [
          (final: prev: {
            # anyio (transitive dep of transformers->huggingface-hub->httpx, needed
            # by `serve`) installs fine; only its own test phase fails on the
            # build pool under Python 3.12 (test_tls_connectable + a few asyncio
            # threading tests) — environment-dependent stdlib behavior, not a
            # package defect. The server only imports anyio; skip the check.
            #
            # The override must go through the interpreter's `packageOverrides`,
            # not a top-level `python312Packages.anyio` — all A/B-verified on
            # this nixpkgs rev:
            #  - `serve` is `python312.withPackages (ps: with ps; [...])`, which
            #    resolves modules from the *interpreter's* passthru set
            #    `python312.pkgs` (with-packages.nix: `buildEnv.override {
            #    extraLibs = f pythonPackages }` where pythonPackages IS the
            #    interpreter's pkgs, built inside passthrufun.nix).
            #  - A top-level overlay that sets `python312Packages.anyio` (plain
            #    derivation or overridePythonAttrs) never touches `python312.pkgs`,
            #    so `serve.drvPath` stays byte-identical — verified repeatedly:
            #    the served closure still referenced the check-running anyio drv.
            #  - `python312.overrideAttrs (oa: oa // { pkgs = ... })` fails
            #    outright: `pkgs` is a passthru, not a derivation attr.
            #
            # `python312.override { packageOverrides = ... }` is the actual hook
            # into that internal set (cpython/default.nix:95; passthrufun.nix
            # composes it into the set via makeScopeWithSplicing'). Verified: the
            # patched anyio lands in serve's closure, `serve.drvPath` changes,
            # and the resulting anyio has zero check-phase deps (hypothesis /
            # trustme / pytest-* / uvloop / psutil all drop out), so the build
            # pool never compiles the failing test suite.
            python312 = prev.python312.override {
              packageOverrides = self: super: {
                anyio = super.anyio.overridePythonAttrs {
                  doCheck = false;
                };
              };
            };
          })
        ];
      };

      cuda = pkgs.cudaPackages_13_3;  # matches the dev image (nvidia/cuda:13.3.0-devel)

      # CUTLASS v4.8 (sm_120a block-scaled FP4), pinned to the exact commit the engine was developed
      # against (dev/Dockerfile arg CUTLASS_COMMIT). nixpkgs cudaPackages.cutlass (4.7.1) predates the
      # block-scaled MMA features the engine uses, so we fetch the source directly.
      # fetchgit (not fetchFromGitHub): GitHub's on-demand commit tarballs are byte-non-deterministic,
      # so any pinned tarball hash eventually rots. fetchgit hashes the checked-out tree deterministically
      # and matches how the dev Docker image provisions CUTLASS (git clone + checkout, no submodules).
      cutlass = pkgs.fetchgit {
        url = "https://github.com/NVIDIA/cutlass";
        rev = "098de2a652cf8f00fd70b2df54051c7eccbb855a";
        hash = "sha256-FiXSg6DMUAV9Gf5LOePKAwmGLVPxphq6MVwHZwwsKDM=";
        fetchSubmodules = false;
      };

      # --- engine: C++/CUDA shared library (libq27.so) ---
      engine = pkgs.stdenv.mkDerivation (finalAttrs: {
        pname = "qwen38-engine";
        version = "0.1.0";
        src = self + "/engine";

        nativeBuildInputs = [
          pkgs.cmake
          pkgs.ninja
          cuda.cuda_nvcc
        ];

        buildInputs = [
          cuda.cuda_cudart
          (lib.getOutput "static" cuda.cuda_cudart)
        ];

        env.CUTLASS_DIR = cutlass;

        # The CMakeLists already sets CMAKE_CUDA_ARCHITECTURES=120a, CMAKE_BUILD_TYPE=Release,
        # CMAKE_CUDA_STANDARD=20. We only need to tell nixpkgs to use CMake+Ninja.
        #
        # Rely on the nixpkgs cmake hook (auto-registered because `cmake` is in
        # nativeBuildInputs) to drive configure/build/install instead of hand-rolling
        # buildPhase. A manual `cmake -B build` in buildPhase collided with the hook's
        # auto-configure in configurePhase (which had already created `build/` and left
        # the cwd inside it), so the second configure treated the build dir as the source
        # dir: "The source directory /build/engine/build does not contain CMakeLists.txt".
        cmakeConfigureArgs = [ "-G" "Ninja" ];

        meta = with pkgs.lib; {
          description = "Qwen3.8-27B inference engine (C++/CUDA, SM120)";
          license = licenses.mit;
          platforms = [ "x86_64-linux" ];
        };
      });

      # --- serve: Python 3.12 environment for the frontend server ---
      # The frontend (frontend/server.py) is a stdlib http.server; at startup it imports
      # qwen27b.weights (torch + safetensors) and transformers.AutoTokenizer, and on image
      # requests it imports qwen27b.vision (torch, torchvision, PIL).
      # The C++ engine itself runs in its own process (worker.py → libq27.so) and never
      # touches PyTorch — but the parent server process needs it to load the tokenizer.
      serve = pkgs.python312.withPackages (ps: with ps; [
        torch-bin
        torchvision-bin
        transformers
        safetensors
        numpy
        pillow
      ]);

      # --- repoFiles: the Python-side source tree the server runs from at runtime ---
      # server.py computes ROOT = Path(__file__).parents[1] and reads frontend/, engine/python/,
      # reference/ and bench/draft-vocab/vocab-128k.txt relative to it. The systemd unit starts
      # the server from this store path, so the whole runtime layout must be present on disk.
      repoFiles = pkgs.stdenv.mkDerivation {
        pname = "qwen38-runtime-files";
        version = "0.1.0";
        src = self;
        installPhase = ''
          mkdir -p $out
          cp -r frontend $out/frontend
          cp -r reference $out/reference
          mkdir -p $out/engine
          cp -r engine/python $out/engine/python
          mkdir -p $out/bench/draft-vocab
          cp bench/draft-vocab/vocab-128k.txt $out/bench/draft-vocab/
        '';
        meta = with pkgs.lib; {
          description = "Qwen3.8-27B runtime files (frontend + reference + draft vocab)";
          license = licenses.mit;
          platforms = [ "x86_64-linux" ];
        };
      };

      # --- checks (run by `nix flake check`) ---

      # Frontend tests are pure stdlib (no torch, no GPU): the parsing and
      # admission logic. Fast and cheap — always runs in CI.
      frontendTests = let
        pyEnv = pkgs.python312.withPackages (ps: with ps; [
          pytest
        ]);
      in pkgs.stdenv.mkDerivation {
        pname = "frontend-tests";
        version = "0.1.0";
        src = self;
        nativeBuildInputs = [ pyEnv ];
        doCheck = true;
        checkPhase = "cd frontend && python -m pytest -q tests/";
        # nixpkgs >= 25.x requires the builder to create $out; a pure-test
        # derivation must still produce (empty) output.
        installPhase = "mkdir -p $out";
      };

      # CPU-runnable reference tests (GDN math, quantization). The snapshot
      # tests (test_model_cpu, test_vision, test_vs_transformers) need the
      # 20 GB checkpoint and are excluded from CI.
      referenceTests = let
        pyEnv = pkgs.python312.withPackages (ps: with ps; [
          torch-bin
          transformers
          safetensors
          numpy
          pillow
          pytest
        ]);
      in pkgs.stdenv.mkDerivation {
        pname = "reference-tests";
        version = "0.1.0";
        src = self;
        nativeBuildInputs = [ pyEnv ];
        doCheck = true;
        checkPhase = "cd reference && PYTHONPATH=. python -m pytest -q tests/test_gdn.py tests/test_quant.py";
        # nixpkgs >= 25.x requires the builder to create $out; a pure-test
        # derivation must still produce (empty) output.
        installPhase = "mkdir -p $out";
      };

    in {
      packages.${system} = {
        default = engine;
        engine = engine;
        serve = serve;
        repoFiles = repoFiles;
      };

      devShells.${system}.default = pkgs.mkShell {
        buildInputs = [
          pkgs.cmake
          pkgs.ninja
          cuda.cuda_nvcc
          cuda.cuda_cudart
        ] ++ [
          (pkgs.python312.withPackages (ps: with ps; [
            torch-bin
            torchvision-bin
            transformers
            safetensors
            numpy
            pillow
            httpx
            pytest
          ]))
        ];
        env.CUTLASS_DIR = cutlass;
      };

      # NixOS module: systemd service for the engine.
      # Usage (in your NixOS flake):
      #   nixosSystem = lib.nixosSystem {
      #     specialArgs = { inherit self; };
      #     configuration = {
      #       imports = [ qwen27b.nixosModules.qwen27b ];
      #       services.qwen27b.enable = true;
      #       services.qwen27b.modelPath = "/path/to/hf-cache";
      #     };
      #   };
      nixosModules.qwen27b = import ./modules/default.nix;

      checks.${system} = {
        frontend-tests = frontendTests;
        reference-tests = referenceTests;
      };
    };
}
