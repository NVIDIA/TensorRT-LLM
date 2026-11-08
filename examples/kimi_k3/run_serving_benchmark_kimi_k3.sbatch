#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Kimi K3 serving benchmark on 16 NVIDIA Blackwell GPUs.
#
# Brings up trtllm-serve with attention DP + 16-way expert parallelism, then
# drives the in-repo benchmark client (tensorrt_llm/serve/scripts/
# benchmark_serving.py) with synthetic random 8K-input/1K-output requests.
# Per concurrency C: one warmup pass (C requests, discarded), then one
# measured pass (10*C requests) saved as JSON.
#
# Submit from the repository root of a built TensorRT-LLM checkout (see the
# README Prerequisites section):
#
#   sbatch examples/kimi_k3/run_serving_benchmark_kimi_k3.sbatch \
#       --model /path/to/kimi-k3-checkpoint \
#       --image /path/to/tensorrt-llm-container.sqsh
#
# Optional arguments: --isl N (default 8192), --osl N (default 1024),
# --concurrencies "64 256". REPO defaults to the submit directory (override
# by exporting REPO) and must contain the built TensorRT-LLM checkout.
# Results and server/client logs go to kimi-k3-serving-benchmark-<job-id>/
# in the submission directory. The #SBATCH partition and account below are
# placeholders — edit them or override on the sbatch command line
# (sbatch --partition=... --account=...).
#
# Note: server initialization and weight loading take 10-15 minutes before
# timed benchmark traffic starts; if your cluster reaps idle-GPU jobs, grant
# this job an exemption via your site's mechanism.
#
#SBATCH --job-name=kimi-k3-serving-benchmark
#SBATCH --partition=batch
#SBATCH --account=${account}
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=03:00:00
#SBATCH --output=kimi-k3-serving-benchmark-%j.log

set -euo pipefail

usage() {
    echo "Usage: sbatch $0 --model PATH --image PATH [--isl N] [--osl N] [--concurrencies \"64 256\"]"
}

MODEL=""
CONTAINER_IMAGE=""
ISL=8192
OSL=1024
CONCURRENCIES="64 256"
PORT=8000
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model|--image|--isl|--osl|--concurrencies)
            [[ $# -ge 2 ]] || { echo "error: $1 requires a value" >&2; usage >&2; exit 2; }
            VALUE=$2
            OPT=$1
            shift 2
            ;;
        --model=*|--image=*|--isl=*|--osl=*|--concurrencies=*)
            VALUE=${1#*=}
            OPT=${1%%=*}
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
    case "$OPT" in
        --model) MODEL=$VALUE ;;
        --image) CONTAINER_IMAGE=$VALUE ;;
        --isl) ISL=$VALUE ;;
        --osl) OSL=$VALUE ;;
        --concurrencies) CONCURRENCIES=$VALUE ;;
    esac
done

[[ -n "$MODEL" ]] || { echo "error: --model is required" >&2; usage >&2; exit 2; }
[[ -n "$CONTAINER_IMAGE" ]] || { echo "error: --image is required" >&2; usage >&2; exit 2; }
[[ -e "$MODEL" ]] || { echo "error: model path does not exist: $MODEL" >&2; exit 2; }
[[ -e "$CONTAINER_IMAGE" ]] || { echo "error: image path does not exist: $CONTAINER_IMAGE" >&2; exit 2; }
[[ "$ISL" =~ ^[0-9]+$ && "$OSL" =~ ^[0-9]+$ ]] || { echo "error: --isl/--osl must be integers" >&2; exit 2; }
for concurrency in $CONCURRENCIES; do
    [[ "$concurrency" =~ ^[0-9]+$ ]] || { echo "error: bad concurrency: $concurrency" >&2; exit 2; }
done

# 9344 at the default 8192/1024, matching the verified runs exactly.
MAX_SEQ_LEN=$((ISL + OSL + 128))

REPO=${REPO:-$SLURM_SUBMIT_DIR}
WORK_DIR=$SLURM_SUBMIT_DIR/kimi-k3-serving-benchmark-$SLURM_JOB_ID
mkdir -p "$WORK_DIR"

# Serving config; free_gpu_memory_fraction 0.25 leaves headroom for the
# KDA/SSM state cache on top of the model weights.
CONFIG_FILE=$WORK_DIR/serve_config.yaml
cat > "$CONFIG_FILE" <<'EOF'
enable_attention_dp: true
moe_expert_parallel_size: 16
disable_overlap_scheduler: false
cuda_graph_config:
  max_batch_size: 32
enable_chunked_prefill: true
kv_cache_config:
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.25
EOF

# Shared warm caches, persistent across runs: without a pre-populated
# FLASHINFER_CUBIN_DIR, ranks on egress-less compute nodes can hang fetching
# cubins during the TRTLLM-Gen FMHA warmup.
CACHE_DIR=${CACHE_DIR:-$SLURM_SUBMIT_DIR/.kimi_k3_cache}
mkdir -p "$CACHE_DIR"

MOUNTS="$REPO:$REPO:rw,$MODEL:$MODEL:ro,$WORK_DIR:$WORK_DIR:rw,$CACHE_DIR:$CACHE_DIR:rw"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | awk 'NR == 1')

SERVER_PID=""
cleanup() { [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

# Server: all 16 ranks; rank 0 (this node) binds the HTTP endpoint.
# --kill-on-bad-exit=1: any failed rank kills the whole step immediately.
srun --mpi=pmix --kill-on-bad-exit=1 \
    --container-image="$CONTAINER_IMAGE" \
    --container-mount-home \
    --container-mounts="$MOUNTS" \
    --output="$WORK_DIR/serve.log" \
    bash -c "
        set -x
        ulimit -n 65536

        # Node-local Triton cache: ~/.triton on shared NFS races across
        # ranks during autotune JIT (stale file handles).
        export TRITON_CACHE_DIR=/tmp/triton-cache-rank\${SLURM_PROCID:-0}
        mkdir -p \"\$TRITON_CACHE_DIR\"
        # Shared warm caches (see the CACHE_DIR comment above).
        export FLASHINFER_WORKSPACE_BASE=$CACHE_DIR/fi_workspace
        export FLASHINFER_CUBIN_DIR=$CACHE_DIR/fi_cubins
        export HF_HOME=\${HF_HOME:-$CACHE_DIR/hf_home}
        mkdir -p \"\$FLASHINFER_WORKSPACE_BASE\" \"\$FLASHINFER_CUBIN_DIR\" \"\$HF_HOME\"

        # Exercise CommunicationFactory AUTO selection rather than inheriting
        # a caller-side forced communication strategy.
        unset TRTLLM_FORCE_COMM_METHOD

        # In-place venv of the built checkout (build_wheel.py creates .venv-3.12).
        export PATH=\"$REPO/.venv-3.12/bin:\$PATH\"

        # The leader receives --log_level through trtllm-serve. Spawned MPI
        # workers read TLLM_LOG_LEVEL instead, so export it for all ranks.
        export TLLM_LOG_LEVEL=\${TLLM_LOG_LEVEL:-info}

        exec '$REPO/tensorrt_llm/llmapi/trtllm-llmapi-launch' trtllm-serve \
            \"$MODEL\" \
            --backend pytorch \
            --host 0.0.0.0 \
            --port $PORT \
            --tp_size 16 \
            --max_batch_size 32 \
            --max_num_tokens 8192 \
            --max_seq_len $MAX_SEQ_LEN \
            --trust_remote_code \
            --config '$CONFIG_FILE'
    " &
SERVER_PID=$!

# Wait up to 25 min for health; on timeout or server death, cancel the job
# so the allocation never sits on idle GPUs.
deadline=$((SECONDS + 1500))
until curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null || (( SECONDS >= deadline )); then
        echo "error: server not healthy within 25 min (see $WORK_DIR/serve.log); cancelling job" >&2
        scancel "$SLURM_JOB_ID"
        sleep 120
        exit 1
    fi
    sleep 15
done
echo "Server is healthy; starting benchmark sweep (see $WORK_DIR/client.log)."

# Client on the head node, alongside the server. The random dataset is
# generated client-side (--random-ids --tokenize-on-client) so the server
# sees exactly ISL input tokens; --ignore-eos fixes output length at OSL.
srun --overlap --kill-on-bad-exit=1 --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    --container-image="$CONTAINER_IMAGE" \
    --container-mount-home \
    --container-mounts="$MOUNTS" \
    --output="$WORK_DIR/client.log" \
    bash -c "
        set -x
        export PATH=\"$REPO/.venv-3.12/bin:\$PATH\"

        run_client() {
            python3 -m tensorrt_llm.serve.scripts.benchmark_serving \
                --model \"$MODEL\" \
                --host 127.0.0.1 \
                --port $PORT \
                --dataset-name random \
                --random-ids \
                --tokenize-on-client \
                --random-input-len $ISL \
                --random-output-len $OSL \
                --ignore-eos \
                --trust-remote-code \
                \"\$@\"
        }

        for concurrency in $CONCURRENCIES; do
            echo \"=== Warmup: concurrency \$concurrency ===\"
            run_client --num-prompts \$concurrency --max-concurrency \$concurrency

            echo \"=== Measured: concurrency \$concurrency ===\"
            run_client \
                --num-prompts \$((concurrency * 10)) \
                --max-concurrency \$concurrency \
                --save-result \
                --result-dir '$WORK_DIR' \
                --result-filename \"concurrency_\${concurrency}.json\"
        done
    "

echo "Benchmark complete; results in $WORK_DIR"
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
