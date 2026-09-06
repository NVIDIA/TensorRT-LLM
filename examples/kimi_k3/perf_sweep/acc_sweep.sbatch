#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Kimi K3 ACCURACY sweep (GSM8K) for the serving recipes measured by
# perf_sweep.sbatch: same per-mode yaml, same env, same tp/batch/token CLI
# flags (keep the mode blocks below in sync with perf_sweep.sbatch by hand).
# One in-process trtllm-eval run per job; expect ~96.5 +/- 0.5 on the full
# 1319-question set.
#
# Usage:
#   sbatch [--nodes=N ...] acc_sweep.sbatch \
#       --mode dep16|tep16|tep8 --model PATH --image PATH [--repo PATH] [--tag NAME]
#
# The #SBATCH partition and account below are placeholders — edit them or
# override on the sbatch command line (sbatch --partition=... --account=...).
#
#SBATCH --job-name=kimi-k3-acc
#SBATCH --partition=batch
#SBATCH --account=${account}
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=kimi-k3-acc-%j.log

set -euo pipefail

MODE=""; MODEL=""; CONTAINER_IMAGE=""; TAG=""; REPO_ARG=""; ISL=8192; OSL=1024
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE=$2; shift 2 ;;
        --model) MODEL=$2; shift 2 ;;
        --image) CONTAINER_IMAGE=$2; shift 2 ;;
        --tag) TAG=$2; shift 2 ;;
        --repo) REPO_ARG=$2; shift 2 ;;
        *) echo "error: unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$MODE" && -n "$MODEL" && -n "$CONTAINER_IMAGE" ]] || {
    echo "error: --mode, --model and --image are required" >&2; exit 2; }
[[ -e "$MODEL" && -e "$CONTAINER_IMAGE" ]] || { echo "error: bad model/image path" >&2; exit 2; }

# Same batch shape and tuned env as perf_sweep.sbatch (see that script
# for the per-knob rationale).
case "$MODE" in
    dep16)
        TP_SIZE=16; MAX_BATCH_SIZE=64
        MODE_ENV="KIMI_K3_FP8_WEIGHT_READ=1"
        ;;
    tep16)
        TP_SIZE=16; MAX_BATCH_SIZE=16
        MODE_ENV="KIMI_K3_FP8_WEIGHT_READ=1 KIMI_K3_FP8_WEIGHT_READ_KDA=0 KIMI_K3_FP8_WEIGHT_READ_GATE_UP=1"
        # BF16 KDA read + FP8 gate_up at tp16 measured ~40% better
        # tps/user at c1 than the base FP8 recipe, at equal-or-better
        # GSM8K accuracy. Do NOT add the separated-routing env here: at
        # tp16 it costs accuracy on its own and, combined with the FP8
        # gate_up read, collapses GSM8K entirely; that pair is validated
        # for tep8 only.
        ;;
    tep8)
        TP_SIZE=8; MAX_BATCH_SIZE=16
        MODE_ENV="KIMI_K3_FP8_WEIGHT_READ=1 KIMI_K3_FP8_WEIGHT_READ_GATE_UP=1 TLLM_TRTLLMGEN_FORCE_SEPARATED_ROUTING=1"
        ;;
    *) echo "error: unknown --mode: $MODE (dep16|tep16|tep8)" >&2; exit 2 ;;
esac

MAX_SEQ_LEN=$((ISL + OSL + 128))
MAX_NUM_TOKENS=$((ISL + MAX_BATCH_SIZE))

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO_ARG:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
WORK_DIR=$SLURM_SUBMIT_DIR/kimi-k3-acc-$MODE${TAG:+-$TAG}-$SLURM_JOB_ID
mkdir -p "$WORK_DIR"

# Same serving yaml as perf_sweep.sbatch, verbatim.
CONFIG_FILE=$WORK_DIR/serve_config.yaml
case "$MODE" in
    dep16)
        cat > "$CONFIG_FILE" <<'EOF'
enable_attention_dp: true
moe_expert_parallel_size: 16
disable_overlap_scheduler: false
enable_chunked_prefill: true
cuda_graph_config:
  max_batch_size: 64
  enable_padding: true
moe_config:
  max_num_tokens: 33024
  use_low_precision_moe_combine: true
kv_cache_config:
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.6
stream_interval: 10
EOF
        ;;
    tep16)
        cat > "$CONFIG_FILE" <<'EOF'
enable_attention_dp: false
moe_expert_parallel_size: 16
disable_overlap_scheduler: false
enable_chunked_prefill: true
cuda_graph_config:
  max_batch_size: 16
kv_cache_config:
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.25
stream_interval: 10
EOF
        ;;
    tep8)
        cat > "$CONFIG_FILE" <<'EOF'
enable_attention_dp: false
moe_expert_parallel_size: 8
disable_overlap_scheduler: false
enable_chunked_prefill: true
cuda_graph_config:
  max_batch_size: 16
kv_cache_config:
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.25
stream_interval: 10
EOF
        ;;
esac

CACHE_DIR=${CACHE_DIR:-$SLURM_SUBMIT_DIR/.kimi_k3_cache}
mkdir -p "$CACHE_DIR"
ENV_CMD="true"
[[ -n "$MODE_ENV" ]] && ENV_CMD="export $MODE_ENV"
MOUNTS="$REPO:$REPO:rw,$MODEL:$MODEL:ro,$WORK_DIR:$WORK_DIR:rw,$CACHE_DIR:$CACHE_DIR:rw"

srun --mpi=pmix --kill-on-bad-exit=1 \
    --container-image="$CONTAINER_IMAGE" \
    --container-mount-home \
    --container-mounts="$MOUNTS" \
    bash -c "
        set -x
        ulimit -n 65536
        export PYTHONNOUSERSITE=1
        $ENV_CMD
        export TRITON_CACHE_DIR=/tmp/triton-cache-rank\${SLURM_PROCID:-0}
        mkdir -p \"\$TRITON_CACHE_DIR\"
        export FLASHINFER_WORKSPACE_BASE=$CACHE_DIR/fi_workspace
        export FLASHINFER_CUBIN_DIR=$CACHE_DIR/fi_cubins
        # HF dataset/hub cache: \$HOME/.cache is dangling on remote ranks in
        # some cluster images, so keep HF under CACHE_DIR (matches
        # perf_sweep.sbatch); the gsm8k dataset is served from this cache.
        export HF_HOME=\${HF_HOME:-$CACHE_DIR/hf_home}
        mkdir -p \"\$FLASHINFER_WORKSPACE_BASE\" \"\$FLASHINFER_CUBIN_DIR\" \"\$HF_HOME\"
        # Node-local HF remote-code modules cache: racing mkdir over shared
        # \$HOME escapes pathlib's exist_ok on NFS (see run_eval_kimi_k3).
        export HF_MODULES_CACHE=/tmp/hf-modules-rank\${SLURM_PROCID:-0}
        # Running partial score every N responses (early failure signal).
        export TLLM_EVAL_PARTIAL_SCORES_EVERY=\"\${TLLM_EVAL_PARTIAL_SCORES_EVERY:-100}\"
        export PATH=\"$REPO/.venv-3.12/bin:\$PATH\"
        export PYTHONPATH=\"$REPO\${PYTHONPATH:+:\$PYTHONPATH}\"
        # Offline compute nodes cannot reach the Hub: point lm_eval at the
        # pre-downloaded snapshot when present (falls back to the hub id).
        GSM8K_DATASET_PATH=\${GSM8K_DATASET_PATH:-\$(ls -d \"\$HF_HOME\"/hub/datasets--openai--gsm8k/snapshots/*/ 2>/dev/null | head -1)}
        DATASET_ARGS=\"\"
        [ -n \"\$GSM8K_DATASET_PATH\" ] && DATASET_ARGS=\"--dataset_path \$GSM8K_DATASET_PATH\"
        exec '$REPO/tensorrt_llm/llmapi/trtllm-llmapi-launch' python3 \
            '$REPO/.venv-3.12/bin/trtllm-eval' \
            --model \"$MODEL\" \
            --backend pytorch \
            --tp_size $TP_SIZE \
            --max_batch_size $MAX_BATCH_SIZE \
            --max_num_tokens $MAX_NUM_TOKENS \
            --max_seq_len $MAX_SEQ_LEN \
            --trust_remote_code \
            --config '$CONFIG_FILE' \
            gsm8k \$DATASET_ARGS
    " 2>&1 | tee "$WORK_DIR/eval.log"

grep -E "average accuracy|exact_match" "$WORK_DIR/eval.log" | tail -4 || true
echo "Accuracy run complete; log in $WORK_DIR/eval.log"
