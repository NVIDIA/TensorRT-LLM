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
# Pass --reuse to additionally enable KV-cache block reuse
# (eval_extra_llm_options_reuse.yaml); chunked prefill is enabled by
# default.
#
# Pass --dflash /path/to/drafter to evaluate with DFlash speculative
# decoding (selects eval_extra_llm_options_dflash.yaml with the drafter
# path substituted in). With a synthetic drafter
# (make_synthetic_dflash_drafter.py) this is a wiring smoke only.
#
# --sa, --reuse, and --dflash are mutually exclusive.
#
# REPO defaults to the submit directory (override by exporting REPO), and
# must contain the built TensorRT-LLM checkout. Provide scheduler-specific
# options on the sbatch command line as needed.
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
    echo "Usage: sbatch $0 --model PATH --image PATH [--sa | --reuse | --dflash DRAFTER_PATH]"
}

MODEL=""
CONTAINER_IMAGE=""
MODE=default
SPEC_STATS_DEFAULT=0
DFLASH_DRAFTER=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sa|--reuse)
            [[ "$MODE" == default ]] || { echo "error: --sa, --reuse, and --dflash are mutually exclusive" >&2; usage >&2; exit 2; }
            MODE=${1#--}
            shift
            ;;
        --dflash)
            [[ $# -ge 2 ]] || { echo "error: --dflash requires a drafter checkpoint path" >&2; usage >&2; exit 2; }
            [[ "$MODE" == default ]] || { echo "error: --sa, --reuse, and --dflash are mutually exclusive" >&2; usage >&2; exit 2; }
            MODE=dflash
            DFLASH_DRAFTER=$2
            shift 2
            ;;
        --dflash=*)
            [[ "$MODE" == default ]] || { echo "error: --sa, --reuse, and --dflash are mutually exclusive" >&2; usage >&2; exit 2; }
            MODE=dflash
            DFLASH_DRAFTER=${1#*=}
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

# Per-mode LLM options.
# - sa: SA speculative decoding needs its own LLM options (eager, plain-EP;
#   see the yaml) and max_batch_size <= 8 (SpeculativeState buffers).
# - reuse: KV-cache block reuse; drops --disable_kv_cache_reuse so the
#   yaml's enable_block_reuse takes effect.
# - dflash: DFlash speculative decoding against the --dflash drafter; the
#   drafter path is substituted into a per-job copy of the DFlash yaml
#   (the checked-in file carries a placeholder). Smaller warmup/activation
#   footprint and KV budget: drafter weights, capture buffer, and DFlash
#   context-KV slots leave less headroom than the SA config assumes
#   (0.25/8192 OOM'd in warmup on GB300).
MAX_BATCH_SIZE=32
MAX_NUM_TOKENS=8192
KV_FRAC=0.25
KV_REUSE_ARG="--disable_kv_cache_reuse"
case "$MODE" in
    sa)
        EVAL_CONFIG=$REPO/examples/kimi_k3/eval_extra_llm_options_sa.yaml
        MAX_BATCH_SIZE=8
        SPEC_STATS_DEFAULT=1
        ;;
    reuse)
        EVAL_CONFIG=$REPO/examples/kimi_k3/eval_extra_llm_options_reuse.yaml
        KV_REUSE_ARG=""
        ;;
    dflash)
        [[ -e "$DFLASH_DRAFTER" ]] || { echo "error: drafter path does not exist: $DFLASH_DRAFTER" >&2; exit 2; }
        EVAL_CONFIG=$REPO/examples/kimi_k3/.eval_dflash_runtime.$SLURM_JOB_ID.yaml
        sed "s|speculative_model: .*|speculative_model: $DFLASH_DRAFTER|" \
            "$REPO/examples/kimi_k3/eval_extra_llm_options_dflash.yaml" > "$EVAL_CONFIG"
        MAX_BATCH_SIZE=8
        MAX_NUM_TOKENS=4096
        KV_FRAC=0.20
        SPEC_STATS_DEFAULT=1
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
[[ -n "$DFLASH_DRAFTER" ]] && MOUNTS="$MOUNTS,$DFLASH_DRAFTER:$DFLASH_DRAFTER:ro"
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
        # Spec-dec acceptance summary (AL/AR) at eval end; default ON for the
        # sa/dflash modes, off otherwise (explicit env still wins either way).
        export TLLM_EVAL_SPEC_STATS=\"\${TLLM_EVAL_SPEC_STATS:-$SPEC_STATS_DEFAULT}\"

        # Import tensorrt_llm from $REPO, not from wherever the venv's
        # in-place install points (for git-worktree submits those differ;
        # without this the job silently tests the main checkout's code).
        export PYTHONPATH=\"$REPO\${PYTHONPATH:+:\$PYTHONPATH}\"

        exec '$REPO/tensorrt_llm/llmapi/trtllm-llmapi-launch' python3 \
            '$REPO/.venv-3.12/bin/trtllm-eval' \
            --model \"$MODEL\" \
            --backend pytorch \
            --tp_size 16 \
            --max_batch_size $MAX_BATCH_SIZE \
            --max_seq_len 8192 \
            --max_num_tokens $MAX_NUM_TOKENS \
            --kv_cache_free_gpu_memory_fraction $KV_FRAC \
            $KV_REUSE_ARG \
            --trust_remote_code \
            --config '$EVAL_CONFIG' \
            gsm8k
    "
