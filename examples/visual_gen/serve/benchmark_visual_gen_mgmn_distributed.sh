#!/bin/bash
#SBATCH -A <account>    # parameter
#SBATCH -p <partition>  # parameter
#SBATCH -t 01:00:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -o visual-gen-serve-distributed.out
#SBATCH -e visual-gen-serve-distributed.err
#SBATCH -J visual-gen-serve-distributed

##############################################################################
# OVERVIEW:
# This script demonstrates running the VisualGen serving benchmark (trtllm-serve
# + benchmark client) on SLURM with distributed inference across multiple GPUs
# and nodes using Ulysses sequence parallelism.
#
# Two SLURM job steps run concurrently:
#   Step 1 (background): Launch trtllm-serve across all allocated nodes/GPUs.
#                        The server handles distributed inference internally.
#   Step 2 (foreground): Wait for the server to become healthy on the rank-0
#                        node, then run the benchmark client against it.
#
# WHAT TO MODIFY:
# 1. SLURM Parameters (lines above):
#    - Replace <account> with your SLURM account name
#    - Replace <partition> with your SLURM partition name
#    - Adjust -N, --ntasks-per-node, and --gpus-per-node to match your config
#    - Total GPUs must equal: dit_cfg_size * dit_ulysses_size (set in SERVER_CONFIG)
#
# 2. Environment Variables (set before running sbatch, or edit defaults below):
#    - CONTAINER_IMAGE: Docker image or enroot .sqsh image with TensorRT-LLM installed
#    - MOUNT_DIR:       host directory to mount into the container (default: $HOME)
#    - MOUNT_DEST:      mount destination path inside the container (default: $HOME)
#    - PROJECT_ROOT:    path to TensorRT-LLM source on the shared filesystem
#    - MODEL:           HuggingFace Hub model ID or local path (default: Wan-AI/Wan2.2-T2V-A14B-Diffusers)
#    - SERVER_CONFIG:   YAML config for trtllm-serve; set dit_cfg_size * dit_ulysses_size
#                       to match total allocated GPUs (default: examples/visual_gen/serve/configs/wan.yml inside PROJECT_ROOT)
#    - BACKEND:         benchmark backend (default: openai-videos)
#    - SERVER_PORT:     HTTP port for trtllm-serve (default: 8000)
#    - SERVER_TIMEOUT:  seconds to wait for server readiness (default: 3600)
#    - MASTER_PORT:     NCCL rendezvous port (default: 29500)
#    - RESULT_DIR:      directory to save benchmark results (default: ./benchmark_results)
#
# 3. Benchmark parameters (SIZE, NUM_FRAMES, FPS, NUM_INFERENCE_STEPS, etc.):
#    Override via environment variables or edit defaults in the section below.
#
# EXAMPLE USAGE:
#   export CONTAINER_IMAGE="/path/to/tensorrt-llm.sqsh"
#   export PROJECT_ROOT="/path/to/TensorRT-LLM"
#   export MODEL="Wan-AI/Wan2.2-T2V-A14B-Diffusers"
#   export SERVER_CONFIG="/path/to/wan.yml"
#   sbatch benchmark_visual_gen_mgmn_distributed.sh
#
##############################################################################

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------

export PROJECT_ROOT="${PROJECT_ROOT:-/path/to/TensorRT-LLM}"
export CONTAINER_IMAGE="${CONTAINER_IMAGE:-/path/to/tensorrt-llm.sqsh}"
export MOUNT_DIR="${MOUNT_DIR:-$HOME}"
export MOUNT_DEST="${MOUNT_DEST:-$HOME}"

export MODEL="${MODEL:-Wan-AI/Wan2.2-T2V-A14B-Diffusers}"
export SERVER_CONFIG="${SERVER_CONFIG:-${PROJECT_ROOT}/examples/visual_gen/serve/configs/wan.yml}"
export BACKEND="${BACKEND:-openai-videos}"
export SERVER_PORT="${SERVER_PORT:-8000}"

# Generation defaults
export SIZE="${SIZE:-1280x720}"
export NUM_FRAMES="${NUM_FRAMES:-81}"
export FPS="${FPS:-16}"
export NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
export GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.0}"
export SEED="${SEED:-42}"

# Benchmark defaults
export NUM_PROMPTS="${NUM_PROMPTS:-3}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
export PROMPT="${PROMPT:-A cat walks through a field of flowers, with the wind blowing gently}"

# Output
export RESULT_DIR="${RESULT_DIR:-./benchmark_results}"

# ---------------------------------------------------------------------------
# Derived values — do not edit
# ---------------------------------------------------------------------------

# Rank-0 node hostname: used for NCCL rendezvous and as the server endpoint
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT="${MASTER_PORT:-29500}"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

wait_for_server() {
    local url="http://${MASTER_ADDR}:${SERVER_PORT}/health"
    local max_wait=${SERVER_TIMEOUT:-3600}  # 60 minutes for model loading + warmup on NFS
    local elapsed=0
    local interval=5

    echo "Waiting for server at ${url} ..."
    while [ $elapsed -lt $max_wait ]; do
        if curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null | grep -q "200"; then
            echo "Server is ready (took ${elapsed}s)"
            return 0
        fi
        sleep $interval
        elapsed=$((elapsed + interval))
        if [ $((elapsed % 30)) -eq 0 ]; then
            echo "  Still waiting... (${elapsed}s elapsed)"
        fi
    done
    echo "ERROR: Server did not become ready within ${max_wait}s" >&2
    return 1
}

cleanup() {
    if [ -n "${SERVER_PID:-}" ]; then
        echo "Stopping server (PID: $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

echo "============================================"
echo "VisualGen Serving Benchmark (multi-node)"
echo "============================================"
echo "Model:               ${MODEL}"
echo "Server config:       ${SERVER_CONFIG}"
echo "Server:              http://${MASTER_ADDR}:${SERVER_PORT}"
echo "Nodes:               ${SLURM_NNODES}"
echo "GPUs per node:       ${SLURM_GPUS_PER_NODE}"
echo "Backend:             ${BACKEND}"
echo "Size:                ${SIZE}"
if [ "$BACKEND" = "openai-videos" ]; then
echo "Num frames:          ${NUM_FRAMES}"
echo "FPS:                 ${FPS}"
fi
echo "Inference steps:     ${NUM_INFERENCE_STEPS}"
echo "Guidance scale:      ${GUIDANCE_SCALE}"
echo "Num prompts:         ${NUM_PROMPTS}"
echo "Max concurrency:     ${MAX_CONCURRENCY}"
echo "Result dir:          ${RESULT_DIR}"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Launch distributed server across all nodes (background)
# ---------------------------------------------------------------------------

export SERVER_CMD="trtllm-serve ${MODEL} --host 0.0.0.0 --port ${SERVER_PORT}"
if [ -n "$SERVER_CONFIG" ]; then
    SERVER_CMD="${SERVER_CMD} --extra_visual_gen_options ${SERVER_CONFIG}"
fi

echo "Step 1: Starting distributed server..."
echo "  Command: ${SERVER_CMD}"
echo ""

srun -l \
    --export=ALL \
    --container-image "${CONTAINER_IMAGE}" \
    --container-workdir "${PROJECT_ROOT}" \
    --container-mounts=${MOUNT_DIR}:${MOUNT_DEST} \
    sh -c 'eval "${SERVER_CMD}"' &

SERVER_PID=$!
trap cleanup EXIT
echo "  Server srun PID: ${SERVER_PID}"

wait_for_server

# ---------------------------------------------------------------------------
# Step 2: Run benchmark (rank-0 node only)
# ---------------------------------------------------------------------------

export BENCHMARK_CMD="python -m tensorrt_llm.serve.scripts.benchmark_visual_gen \
    --model ${MODEL} \
    --backend ${BACKEND} \
    --host ${MASTER_ADDR} \
    --port ${SERVER_PORT} \
    --prompt \"${PROMPT}\" \
    --num-prompts ${NUM_PROMPTS} \
    --size ${SIZE} \
    --num-inference-steps ${NUM_INFERENCE_STEPS} \
    --guidance-scale ${GUIDANCE_SCALE} \
    --seed ${SEED} \
    --max-concurrency ${MAX_CONCURRENCY} \
    --save-result \
    --save-detailed \
    --result-dir ${RESULT_DIR} \
    --metric-percentiles 50,90,99"

if [ "$BACKEND" = "openai-videos" ]; then
    BENCHMARK_CMD="${BENCHMARK_CMD} --num-frames ${NUM_FRAMES} --fps ${FPS}"
fi

echo ""
echo "Step 2: Running benchmark..."
echo "  Command: ${BENCHMARK_CMD}"
echo ""

mkdir -p "${RESULT_DIR}"

srun -l \
    --overlap \
    --ntasks=1 --nodes=1 --nodelist="${MASTER_ADDR}" \
    --export=ALL \
    --container-image "${CONTAINER_IMAGE}" \
    --container-workdir "${PROJECT_ROOT}" \
    --container-mounts=${MOUNT_DIR}:${MOUNT_DEST} \
    sh -c 'eval "${BENCHMARK_CMD}"'

echo ""
echo "============================================"
echo "Benchmark complete. Results in: ${RESULT_DIR}"
echo "============================================"
