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
# Pass --sa to evaluate with suffix-automaton speculative decoding
# (selects eval_extra_llm_options_sa.yaml and the SA-required
# max_batch_size of 8).
#
# REPO defaults to the submit directory (override by exporting REPO), and
# must contain the built TensorRT-LLM checkout. Provide scheduler-specific
# options on the sbatch command line as needed.
#
#SBATCH --job-name=kimi-k3-gsm8k
#SBATCH --partition=batch
#SBATCH --account=coreai_comparch_trtllm
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=kimi-k3-gsm8k-%j.log

set -euo pipefail

usage() {
    echo "Usage: sbatch $0 --model PATH --image PATH [--sa]"
}

MODEL=""
CONTAINER_IMAGE=""
SA=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sa)
            SA=1
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

# SA speculative decoding needs its own LLM options (eager, plain-EP; see
# the yaml) and max_batch_size <= 8 (SpeculativeState buffers).
if [[ "$SA" == "1" ]]; then
    EVAL_CONFIG=$REPO/examples/kimi_k3/eval_extra_llm_options_sa.yaml
    MAX_BATCH_SIZE=8
else
    EVAL_CONFIG=$REPO/examples/kimi_k3/eval_extra_llm_options.yaml
    MAX_BATCH_SIZE=32
fi

# Mount every input at its host path so the paths are valid inside too.
MOUNTS="$REPO:$REPO:rw,$MODEL:$MODEL:ro"

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

        # Run trtllm-eval from the in-place installation created in the
        # README Prerequisites section (build_wheel.py creates .venv-3.12).
        # Invoke it through python3 explicitly: the console script's shebang
        # records the build-time venv path, so a PATH lookup alone breaks
        # with an obscure 'bad interpreter' if the repo is mounted anywhere
        # other than its build-time location.
        export PATH=\"$REPO/.venv-3.12/bin:\$PATH\"

        exec '$REPO/tensorrt_llm/llmapi/trtllm-llmapi-launch' python3 \
            '$REPO/.venv-3.12/bin/trtllm-eval' \
            --model \"$MODEL\" \
            --backend pytorch \
            --tp_size 16 \
            --max_batch_size $MAX_BATCH_SIZE \
            --max_seq_len 8192 \
            --max_num_tokens 8192 \
            --kv_cache_free_gpu_memory_fraction 0.25 \
            --disable_kv_cache_reuse \
            --trust_remote_code \
            --config '$EVAL_CONFIG' \
            gsm8k
    "
