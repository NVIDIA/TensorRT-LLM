#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Kimi K3 GSM8K evaluation on 16 NVIDIA Blackwell GPUs.
#
# Submit the job with the checkpoint and container paths:
#
#   sbatch examples/kimi_k3/run_gsm8k_kimi_k3.sbatch \
#       --model /path/to/kimi-k3-checkpoint \
#       --image /path/to/tensorrt-llm-container.sqsh
#
# Pass --reuse to additionally enable KV-cache block reuse
# (eval_extra_llm_options_reuse.yaml); chunked prefill is enabled by
# default.
#
# REPO defaults to the submit directory (override by exporting REPO), and
# must contain the built TensorRT-LLM checkout. The #SBATCH partition and
# account below are placeholders — edit them or override on the sbatch
# command line (sbatch --partition=... --account=...), along with any
# other scheduler-specific options your cluster requires.
#
#SBATCH --job-name=kimi-k3-gsm8k
#SBATCH --partition=batch
#SBATCH --account=${account}
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=kimi-k3-gsm8k-%j.log

set -euo pipefail

usage() {
    echo "Usage: sbatch $0 --model PATH --image PATH [--reuse]"
}

MODEL=""
CONTAINER_IMAGE=""
MODE=default
while [[ $# -gt 0 ]]; do
    case "$1" in
        --reuse)
            MODE=reuse
            shift
            ;;
        --model)
            [[ $# -ge 2 ]] || { echo "error: --model requires a value" >&2; usage >&2; exit 2; }
            MODEL=$2
            shift 2
            ;;
        --model=*)
            MODEL=${1#*=}
            shift
            ;;
        --image)
            [[ $# -ge 2 ]] || { echo "error: --image requires a value" >&2; usage >&2; exit 2; }
            CONTAINER_IMAGE=$2
            shift 2
            ;;
        --image=*)
            CONTAINER_IMAGE=${1#*=}
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[[ -n "$MODEL" ]] || { echo "error: --model is required" >&2; usage >&2; exit 2; }
[[ -n "$CONTAINER_IMAGE" ]] || { echo "error: --image is required" >&2; usage >&2; exit 2; }
[[ -e "$MODEL" ]] || { echo "error: model path does not exist: $MODEL" >&2; exit 2; }
[[ -e "$CONTAINER_IMAGE" ]] || { echo "error: image path does not exist: $CONTAINER_IMAGE" >&2; exit 2; }

REPO=${REPO:-$SLURM_SUBMIT_DIR}

# Per-mode LLM options. All engine options live in the selected yaml
# (single source of truth); trtllm-eval lets explicit CLI flags override
# --config, so none are duplicated here.
# - reuse: KV-cache block reuse (enable_block_reuse: true in the yaml).
case "$MODE" in
    reuse)
        EVAL_CONFIG=$REPO/examples/kimi_k3/eval_extra_llm_options_reuse.yaml
        ;;
    *)
        EVAL_CONFIG=$REPO/examples/kimi_k3/eval_extra_llm_options.yaml
        ;;
esac

# Mount every input at its host path so the paths are valid inside too.
# EXTRA_MOUNTS (comma-separated src:dst:flags) lets git-worktree submits
# also mount the built main checkout that the worktree's artifact
# symlinks resolve into (submit with --export=ALL).
MOUNTS="$REPO:$REPO:rw,$MODEL:$MODEL:ro"
# Optional extra container mounts (srun --container-mounts syntax). Needed
# e.g. when $REPO is a git worktree whose build artifacts (.venv-3.12,
# bindings *.so) are symlinks into the main checkout.
if [ -n "${EXTRA_MOUNTS:-}" ]; then
    MOUNTS+=",$EXTRA_MOUNTS"
fi

srun --mpi=pmix \
    --container-image="$CONTAINER_IMAGE" \
    --container-mount-home \
    --container-mounts="$MOUNTS" \
    bash -c "
        set -x
        ulimit -n 65536

        # Node-local Triton cache: ~/.triton on shared NFS races across
        # ranks during autotune JIT (stale file handles).
        export TRITON_CACHE_DIR=/tmp/triton-cache-rank\${SLURM_PROCID:-0}
        mkdir -p \"\$TRITON_CACHE_DIR\"

        # Node-local flashinfer cache: the default (\$HOME) is shared NFS and
        # races across ranks during cubin download (stale file handles).
        # These caches are cold on every run (~minutes of JIT per rank);
        # the GSM8K dataset cache is \$HOME/.cache/huggingface, provided by
        # --container-mount-home above.
        export FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer-rank\${SLURM_PROCID:-0}

        # Node-local HF remote-code modules cache: all ranks share \$HOME
        # (mounted at /root), and racing mkdir of
        # ~/.cache/huggingface/modules at startup escapes pathlib's
        # exist_ok on NFS (stale attribute cache fails the is_dir()
        # recheck after EEXIST) -> 'FileExistsError: /root/.cache' spam
        # from every rank. Only the trust_remote_code modules move
        # node-local; the dataset/hub cache stays shared for reuse.
        export HF_MODULES_CACHE=/tmp/hf-modules-rank\${SLURM_PROCID:-0}

        # Run trtllm-eval from the in-place installation created in the
        # README Prerequisites section (build_wheel.py creates .venv-3.12).
        export PATH=\"$REPO/.venv-3.12/bin:\$PATH\"

        # Log a running partial score every N completed responses (0 = off).
        export TLLM_EVAL_PARTIAL_SCORES_EVERY=\"\${TLLM_EVAL_PARTIAL_SCORES_EVERY:-100}\"
        # Cap in-flight requests: may cost some throughput vs submit-all, but yields steady partial scores (early failure signal) instead of one burst at the end (0 = off).
        export TLLM_EVAL_MAX_IN_FLIGHT=\"\${TLLM_EVAL_MAX_IN_FLIGHT:-0}\"

        # Import tensorrt_llm from $REPO, not from wherever the venv's
        # in-place install points (for git-worktree submits those differ;
        # without this the job silently tests the main checkout's code).
        export PYTHONPATH=\"$REPO\${PYTHONPATH:+:\$PYTHONPATH}\"

        exec '$REPO/tensorrt_llm/llmapi/trtllm-llmapi-launch' python3 \
            '$REPO/.venv-3.12/bin/trtllm-eval' \
            --model \"$MODEL\" \
            --backend pytorch \
            --config '$EVAL_CONFIG' \
            gsm8k
    "
