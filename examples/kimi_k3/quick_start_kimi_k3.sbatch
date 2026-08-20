#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Kimi K3 quick start on 16 NVIDIA Blackwell GPUs.
#
# Submit the job with the checkpoint and container paths:
#
#   sbatch examples/kimi_k3/quick_start_kimi_k3.sbatch \
#       --model /path/to/kimi-k3-checkpoint \
#       --image /path/to/tensorrt-llm-container.sqsh
#
# The #SBATCH partition and account below are placeholders — edit them or
# override on the sbatch command line (sbatch --partition=... --account=...).
# REPO defaults to the submit directory (override by exporting REPO) and
# must contain the built TensorRT-LLM checkout.
#
#SBATCH --job-name=kimi-k3-quick-start
#SBATCH --partition=batch
#SBATCH --account=${account}
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=00:40:00
#SBATCH --output=kimi-k3-quick-start-%j.log

set -euo pipefail

usage() {
    echo "Usage: sbatch $0 --model PATH --image PATH [--enable-block-reuse]"
}

MODEL=""
CONTAINER_IMAGE=""
FEATURE_ARGS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --enable-block-reuse)
            # Forwarded verbatim to quick_start_kimi_k3.py (see its --help).
            FEATURE_ARGS="$FEATURE_ARGS $1"
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

MOUNTS="$REPO:$REPO:rw,$MODEL:$MODEL:ro"
# Optional extra container mounts (srun --container-mounts syntax)
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
        export FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer-rank\${SLURM_PROCID:-0}

        # Run python3 from the in-place installation created in the README
        # Prerequisites section (build_wheel.py creates .venv-3.12).
        export PATH=\"$REPO/.venv-3.12/bin:\$PATH\"

        # Import tensorrt_llm from $REPO, not from wherever the venv's
        # in-place install points (for git-worktree submits those differ;
        # without this the job silently tests the main checkout's code).
        export PYTHONPATH=\"$REPO\${PYTHONPATH:+:\$PYTHONPATH}\"

        exec '$REPO/tensorrt_llm/llmapi/trtllm-llmapi-launch' python3 \
            '$REPO/examples/kimi_k3/quick_start_kimi_k3.py' \
            --model \"$MODEL\" \
            --tp-size 16$FEATURE_ARGS
    "
