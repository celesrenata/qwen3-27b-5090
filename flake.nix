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
            # anyio (transitive dep of transformers/httpx, needed by `serve`)
            # installs fine; only its own test phase fails on the build pool
            # under Python 3.12 (test_tls_connectable + a few asyncio threading
            # tests) — environment-dependent stdlib behavior, not a package
            # defect. The server only imports anyio; skip the check.
            #
            # Why every "skip the check" trick fails here, all A/B-verified
            # on this nixpkgs rev:
            #  - `anyio.override { doCheck = false }` — `doCheck` isn't a named
            #    arg of the package function, so makeOverridable drops it.
            #  - `anyio.overrideAttrs (o: o // { doCheck = false; })` — returns
            #    a __functor (makeOverridablePythonPackage, python-packages-
            #    base.nix:80) that evaluates to the SAME drvPath in an overlay
            #    context; the box's own overlays/debugpy.nix uses this form and
            #    is likewise a no-op (invisible only because debugpy's tests
            #    happen to pass).
            #  - `anyio.overridePythonAttrs { doCheck = false; ... }` — changes
            #    the isolated `anyio.drvPath`, but the result does NOT propagate
            #    into a downstream `python312.withPackages` closure: an overlay
            #    that sets `python312Packages.anyio` to this functor leaves
            #    `serve.drvPath` byte-identical to the un-overlaid base, and
            #    the served closure still references the un-patched anyio drv.
            #
            # Replacing `python312Packages.anyio` with a *plain* derivation
            # DOES propagate (sentinel test confirmed `serve.drvPath` changes),
            # so rebuild anyio directly from its upstream recipe — same src /
            # hash / runtime deps, minus the test suite and its deps
            # (hypothesis, trustme, pytest-*, uvloop, psutil) that only the
            # check phase needed. Also cheaper: no check deps to fetch/build.
            #
            # `final` (not `prev`) so the rebuilt anyio picks up any further
            # overlays downstream; only anyio itself is replaced, so this
            # cannot recurse.
            python312Packages.anyio = final.python312Packages.buildPythonPackage rec {
              pname = "anyio";
              version = "4.14.2";
              pyproject = true;
              doCheck = false;

              src = final.fetchFromGitHub {
                owner = "agronholm";
                repo = "anyio";
                tag = version;
                # identical to upstream nixpkgs anyio 4.14.2 fetch hash
                hash = "sha256-MEU0c8/NI1vlyNtBsg/hGLv6DR619ZqoZzNY1eJLEWM=";
              };

              # build-system
              build-system = [ final.python312Packages.setuptools-scm ];

              # runtime deps: idna + typing-extensions on < 3.13
              dependencies =
                [ final.python312Packages.idna ]
                ++ lib.optionals (final.pythonOlder "3.13") [
                  final.python312Packages.typing-extensions
                ];

              meta = (prev.python312Packages.anyio.overridePythonAttrs { }.meta or { }) // {
                description = "Another async I/O library for Python";
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

      # --- engine: C++/CUDA shared library (libq27.so) + q27-sfcheck diagnostic tool ---
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
