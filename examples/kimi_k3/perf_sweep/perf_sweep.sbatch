#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Kimi K3 serving perf sweep (8K-in/1K-out): one trtllm-serve launch
# per job, all requested concurrencies measured against it.
#
# Modes bake the tuned serving recipes (measured on GB300 NVL; treat them
# as starting points for your own cluster):
#   tep8   TP8 attention + EP8 MoE, 8 GPU / 2 nodes  (interactive, c 1-16)
#   tep16  TP16 attention + EP16 MoE, 16 GPU / 4 nodes (low latency, c 1-4)
#   dep16  attention-DP + EP16 MoE, 16 GPU / 4 nodes (throughput, c 16-1024)
#
# Usage:
#   sbatch [--nodes=N ...] perf_sweep.sbatch \
#       --mode dep16|tep16|tep8 --model PATH --image PATH \
#       [--repo PATH] [--concurrencies "16 32 64"] [--isl N] [--osl N] [--tag NAME]
#
# Submit ALL jobs of a comparison batch at the same time: a concurrent
# multi-hundred-GB weight load depresses in-flight DEP16 c>=128
# measurements by ~30%.
#
# The #SBATCH partition and account below are placeholders — edit them or
# override on the sbatch command line (sbatch --partition=... --account=...).
#
#SBATCH --job-name=kimi-k3-sweep
#SBATCH --partition=batch
#SBATCH --account=${account}
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=kimi-k3-sweep-%j.log

set -euo pipefail

MODE=""; MODEL=""; CONTAINER_IMAGE=""; TAG=""; ISL=8192; OSL=1024
CONCURRENCIES=""; REPO_ARG=""; PORT=8000
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE=$2; shift 2 ;;
        --model) MODEL=$2; shift 2 ;;
        --image) CONTAINER_IMAGE=$2; shift 2 ;;
        --tag) TAG=$2; shift 2 ;;
        --isl) ISL=$2; shift 2 ;;
        --osl) OSL=$2; shift 2 ;;
        --concurrencies) CONCURRENCIES=$2; shift 2 ;;
        --repo) REPO_ARG=$2; shift 2 ;;
        *) echo "error: unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$MODE" && -n "$MODEL" && -n "$CONTAINER_IMAGE" ]] || {
    echo "error: --mode, --model and --image are required" >&2; exit 2; }
[[ -e "$MODEL" && -e "$CONTAINER_IMAGE" ]] || { echo "error: bad model/image path" >&2; exit 2; }

# Per-mode batch shape and the tuned env. MAX_BATCH_SIZE is PER-RANK for
# dep16 (attention DP), GLOBAL for the TEP modes.
#   KIMI_K3_FP8_WEIGHT_READ=1  opt-in FP8 (lossy) read of the replicated
#       BF16 weights — default OFF in the loader, so every mode sets it
#       explicitly: all recipes below were tuned and measured with it on.
#   KIMI_K3_FP8_WEIGHT_READ_KDA=0  keep the KDA q/k/v/g/o projections in
#       BF16 within the enabled master (the most precision-sensitive slice).
#   KIMI_K3_FP8_WEIGHT_READ_GATE_UP=1  FP8 shared-expert gate_up read at TEP
#       (with the master on it defaults on only under attention DP): +8-9%
#       with stream_interval >= 10.
#   TLLM_TRTLLMGEN_FORCE_SEPARATED_ROUTING=1  host-side routing; the in-cubin
#       top-k tier register-spills at decode batch 5-64 (33 us/layer).
case "$MODE" in
    dep16)
        TP_SIZE=16; MAX_BATCH_SIZE=64
        DEFAULT_CONCURRENCIES="16 32 64 128 256 512 1024"
        MODE_ENV="KIMI_K3_FP8_WEIGHT_READ=1"
        ;;
    tep16)
        TP_SIZE=16; MAX_BATCH_SIZE=16
        DEFAULT_CONCURRENCIES="1 2 4 8 16"
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
        DEFAULT_CONCURRENCIES="1 2 4 8 16"
        MODE_ENV="KIMI_K3_FP8_WEIGHT_READ=1 KIMI_K3_FP8_WEIGHT_READ_GATE_UP=1 TLLM_TRTLLMGEN_FORCE_SEPARATED_ROUTING=1"
        ;;
    *) echo "error: unknown --mode: $MODE (dep16|tep16|tep8)" >&2; exit 2 ;;
esac
CONCURRENCIES=${CONCURRENCIES:-$DEFAULT_CONCURRENCIES}

MAX_SEQ_LEN=$((ISL + OSL + 128))
MAX_NUM_TOKENS=$((ISL + MAX_BATCH_SIZE))

# Default repo = the checkout containing this script (expects .venv-3.12
# built in place, custom FlashInfer installed; see examples/kimi_k3/README.md).
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO_ARG:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
WORK_DIR=$SLURM_SUBMIT_DIR/kimi-k3-sweep-$MODE${TAG:+-$TAG}-$SLURM_JOB_ID
mkdir -p "$WORK_DIR"

# Tuned serving config per mode. stream_interval 10 removes the ~18 ms/token
# host streaming floor (TPOT metric unaffected: (latency-TTFT)/(len-1)).
CONFIG_FILE=$WORK_DIR/serve_config.yaml
case "$MODE" in
    dep16)
        # moe max_num_tokens 33024: fewer serialized A2A chunk rounds per
        # prefill wave (TTFT -4-6%). use_low_precision_moe_combine: FP8 A2A
        # combine payload, +3.6-4.6% at c64-128 (GSM8K 96.74). enable_padding:
        # ADP graphs are all-or-nothing without it.
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

# Shared warm caches: without a pre-populated FLASHINFER_CUBIN_DIR, ranks on
# egress-less compute nodes can hang fetching cubins during FMHA warmup.
CACHE_DIR=${CACHE_DIR:-$SLURM_SUBMIT_DIR/.kimi_k3_cache}
mkdir -p "$CACHE_DIR"
ENV_CMD="true"
[[ -n "$MODE_ENV" ]] && ENV_CMD="export $MODE_ENV"

MOUNTS="$REPO:$REPO:rw,$MODEL:$MODEL:ro,$WORK_DIR:$WORK_DIR:rw,$CACHE_DIR:$CACHE_DIR:rw"
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | awk 'NR == 1')

SERVER_PID=""
cleanup() { [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

srun --mpi=pmix --kill-on-bad-exit=1 \
    --container-image="$CONTAINER_IMAGE" \
    --container-mount-home \
    --container-mounts="$MOUNTS" \
    --output="$WORK_DIR/serve.log" \
    bash -c "
        set -x
        ulimit -n 65536
        export PYTHONNOUSERSITE=1
        $ENV_CMD
        export TRITON_CACHE_DIR=/tmp/triton-cache-rank\${SLURM_PROCID:-0}
        mkdir -p \"\$TRITON_CACHE_DIR\"
        export FLASHINFER_WORKSPACE_BASE=$CACHE_DIR/fi_workspace
        export FLASHINFER_CUBIN_DIR=$CACHE_DIR/fi_cubins
        export HF_HOME=\${HF_HOME:-$CACHE_DIR/hf_home}
        mkdir -p \"\$FLASHINFER_WORKSPACE_BASE\" \"\$FLASHINFER_CUBIN_DIR\" \"\$HF_HOME\"
        export PATH=\"$REPO/.venv-3.12/bin:\$PATH\"
        export PYTHONPATH=\"$REPO\${PYTHONPATH:+:\$PYTHONPATH}\"
        # Non-leader MPI workers inherit their TRT-LLM log level from the
        # environment rather than the trtllm-serve CLI.
        export TLLM_LOG_LEVEL=\${TLLM_LOG_LEVEL:-info}
        exec '$REPO/tensorrt_llm/llmapi/trtllm-llmapi-launch' trtllm-serve \
            \"$MODEL\" \
            --backend pytorch --host 0.0.0.0 --port $PORT \
            --tp_size $TP_SIZE --max_batch_size $MAX_BATCH_SIZE \
            --max_num_tokens $MAX_NUM_TOKENS --max_seq_len $MAX_SEQ_LEN \
            --trust_remote_code --config '$CONFIG_FILE'
    " &
SERVER_PID=$!

# Self-cancel if not healthy so the allocation never idles.
deadline=$((SECONDS + 1500))
until curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null || (( SECONDS >= deadline )); then
        echo "error: server not healthy within 25 min (see $WORK_DIR/serve.log)" >&2
        scancel "$SLURM_JOB_ID"; sleep 120; exit 1
    fi
    sleep 15
done
echo "Server healthy; sweeping c = $CONCURRENCIES (see $WORK_DIR/client.log)."

# One warmup pass then ONE measured run per point: num_prompts = c x 5
# (c < 256) or c x 3, at max-concurrency = c. Metric of record:
# tps/user = 1000 / median_tpot_ms from each concurrency_<c>.json.
srun --overlap --kill-on-bad-exit=1 --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    --container-image="$CONTAINER_IMAGE" \
    --container-mount-home \
    --container-mounts="$MOUNTS" \
    --output="$WORK_DIR/client.log" \
    bash -c "
        set -x
        export PYTHONNOUSERSITE=1
        # Same cache redirects as the server block: importing tensorrt_llm
        # pulls in flashinfer, which creates \$HOME-based workspace/cache
        # dirs, and \$HOME resolves to an unwritable /root in this container.
        export TRITON_CACHE_DIR=/tmp/triton-cache-client
        export FLASHINFER_WORKSPACE_BASE=$CACHE_DIR/fi_workspace
        export FLASHINFER_CUBIN_DIR=$CACHE_DIR/fi_cubins
        export HF_HOME=\${HF_HOME:-$CACHE_DIR/hf_home}
        export HF_MODULES_CACHE=/tmp/hf-modules-client
        mkdir -p \"\$TRITON_CACHE_DIR\" \"\$FLASHINFER_WORKSPACE_BASE\" \\
            \"\$FLASHINFER_CUBIN_DIR\" \"\$HF_HOME\" \"\$HF_MODULES_CACHE\"
        export PATH=\"$REPO/.venv-3.12/bin:\$PATH\"
        export PYTHONPATH=\"$REPO\${PYTHONPATH:+:\$PYTHONPATH}\"
        # Same cache redirects as the server step: importing tensorrt_llm on
        # the client pulls in flashinfer, whose workspace defaults under
        # \$HOME/.cache — a dangling path on some cluster images.
        export FLASHINFER_WORKSPACE_BASE=$CACHE_DIR/fi_workspace
        export FLASHINFER_CUBIN_DIR=$CACHE_DIR/fi_cubins
        export HF_HOME=\${HF_HOME:-$CACHE_DIR/hf_home}
        export TRITON_CACHE_DIR=/tmp/triton-cache-client
        mkdir -p \"\$TRITON_CACHE_DIR\"
        run_client() {
            python3 -m tensorrt_llm.serve.scripts.benchmark_serving \
                --model \"$MODEL\" --host 127.0.0.1 --port $PORT \
                --dataset-name random --random-ids --tokenize-on-client \
                --random-input-len $ISL --random-output-len $OSL \
                --ignore-eos --trust-remote-code \"\$@\"
        }
        for c in $CONCURRENCIES; do
            reps=5; [ \$c -ge 256 ] && reps=3
            run_client --num-prompts \$c --max-concurrency \$c
            run_client --num-prompts \$((c * reps)) --max-concurrency \$c \
                --save-result --result-dir '$WORK_DIR' \
                --result-filename \"concurrency_\${c}.json\"
        done
    "

echo "Sweep complete; results in $WORK_DIR"
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
