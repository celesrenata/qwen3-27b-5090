# NixOS module: systemd service for the Qwen3.8-27B engine on one RTX 5090.
#
# Requires the flake's `self` to be passed via specialArgs:
#   nixosSystem = lib.nixosSystem {
#     specialArgs = { inherit self; };
#     configuration = {
#       imports = [ (import ./flake.nix) ... .nixosModules.qwen27b ];
#       services.qwen27b.enable = true;
#       services.qwen27b.modelPath = "/path/to/hf-cache";
#     };
#   };

{ self, config, lib, pkgs, ... }:
let
  cfg = config.services.qwen27b;
  engine = self.packages."x86_64-linux".engine;
  serve = self.packages."x86_64-linux".serve;
  repoSrc = self.packages."x86_64-linux".repoFiles;
in {
  options.services.qwen27b = {
    enable = lib.mkEnableOption "Qwen3.8-27B inference engine on one RTX 5090";

    port = lib.mkOption {
      type = lib.types.port;
      default = 8000;
      description = "Port to listen on (bound to 127.0.0.1 by the server).";
    };

    modelPath = lib.mkOption {
      type = lib.types.path;
      description = "Path to the Hugging Face cache containing the weights " +
        "(run scripts/download-weights.sh first, or point at a pre-populated cache).";
    };

    requireRTX5090 = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Refuse to start unless the GPU is an RTX 5090. " +
        "Set to false on other hardware for testing (the engine may not run).";
    };

    extraEnv = lib.mkOption {
      type = lib.types.attrsOf lib.types.str;
      default = { };
      description = "Additional environment variables passed to the engine " +
        "(e.g. MAX_SEQS, DECODE_SHARE, …).";
    };
  };

  config = lib.mkIf cfg.enable {
    systemd.services.qwen27b = {
      description = "Qwen3.8-27B inference engine (RTX 5090)";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" ];

      # Guard: the engine is written for SM120 (RTX 5090) and will not build
      # or run on other GPUs. Fail early with a clear message.
      # nvidia-smi comes from the NVIDIA driver's .bin output, which the
      # hardware.nvidia module adds to the system PATH.
      preStart = lib.mkIf cfg.requireRTX5090 ''
        command -v nvidia-smi >/dev/null || {
          echo "ERROR: nvidia-smi not found; enable hardware.nvidia in the NixOS config." >&2
          exit 1
        }
        GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
        if [[ "$GPU_MODEL" != *"RTX 5090"* ]]; then
          echo "ERROR: expected an RTX 5090, got '$GPU_MODEL'." >&2
          echo "Set services.qwen27b.requireRTX5090 = false to override." >&2
          exit 1
        fi
      '';

      serviceConfig = {
        Environment = [
          "Q27_LIB=${engine}/lib/libq27.so"
          "PYTHONPATH=${repoSrc}/reference"
          "HF_HOME=${cfg.modelPath}"
          "HF_HUB_OFFLINE=1"
          # Serving defaults (compose.yaml / README)
          "SERVED_MODEL_NAME=qwen3.8-27b"
          "MAX_TOKENS=262144"
          "MAX_SEQS=8"
          "ADMIT_KV=1"
          "KV_RESERVE_TOKENS=4096"
          "DECODE_SHARE=0.5"
          "SCHED_QUANTUM_MS=50"
          "BATCH_DECODE=1"
          "PREFILL_CHUNK=2048"
          "KV_DTYPE=nvfp4"
          "WEIGHTS=gdnfp4"
        ]
        # User-supplied overrides
        ++ lib.mapAttrsToList (name: value: "${name}=${value}") cfg.extraEnv;

        ExecStart = "${serve}/bin/python ${repoSrc}/frontend/server.py --port ${toString cfg.port}";

        # The engine pins ~16 GB of host RAM (embedding table, prefix cache, vision tower).
        MemoryAccounting = true;

        # The CUDA context must be available before the engine starts.
        After = [ "nvidia-persistenced.service" ];
      };
    };
  };
}
