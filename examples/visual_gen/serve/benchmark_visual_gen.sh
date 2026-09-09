#!/bin/bash
# Benchmark VisualGen serving with trtllm-serve
#
# This script demonstrates how to:
#   1. Start a trtllm-serve server for VisualGen
#   2. Run the benchmark_visual_gen.py client against it
#
# Usage:
#   # Set model path (HF model ID or local path)
#   export MODEL=Wan-AI/Wan2.2-T2V-A14B-Diffusers
#
#   # Optional: customize server config
#   export SERVER_CONFIG=./configs/wan22.yml
#
#   # Run the benchmark
#   ./benchmark_visual_gen.sh
#
# Optional (for MP4/H.264 video output):
#   apt-get install ffmpeg   # or: conda install ffmpeg
#   Without ffmpeg, videos are saved as AVI/MJPEG using a pure-Python encoder.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "${SCRIPT_DIR}/../../.." && pwd)"}

MODEL=${MODEL:-"Wan-AI/Wan2.2-T2V-A14B-Diffusers"}
SERVER_CONFIG=${SERVER_CONFIG:-"${SCRIPT_DIR}/configs/wan22.yml"}
BACKEND=${BACKEND:-"openai-videos"}
HOST=${HOST:-"127.0.0.1"}
PORT=${PORT:-8000}

# Generation defaults
SIZE=${SIZE:-"720x1280"}
NUM_FRAMES=${NUM_FRAMES:-81}
FPS=${FPS:-16}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-50}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-5.0}
SEED=${SEED:-42}

# Benchmark defaults
NUM_PROMPTS=${NUM_PROMPTS:-3}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-1}
PROMPT=${PROMPT:-"A cat walks through a field of flowers, with the wind blowing gently"}

# Server lifecycle defaults
SERVER_TIMEOUT=${SERVER_TIMEOUT:-3600}
SERVER_SHUTDOWN_TIMEOUT=${SERVER_SHUTDOWN_TIMEOUT:-60}

# Output
RESULT_DIR=${RESULT_DIR:-"./benchmark_results"}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

wait_for_server() {
    local url="http://${HOST}:${PORT}/health"
    local max_wait=$SERVER_TIMEOUT  # 60 minutes for model loading + warmup on NFS
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
    echo "ERROR: Server did not become ready within ${max_wait}s"
    return 1
}

cleanup() {
    local exit_code=$?
    local elapsed=0

    trap - EXIT
    trap '' INT TERM

    if [ -n "${SERVER_PID:-}" ]; then
        echo "Stopping server (PID: $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        while kill -0 "$SERVER_PID" 2>/dev/null && [ "$elapsed" -lt "$SERVER_SHUTDOWN_TIMEOUT" ]; do
            sleep 1
            elapsed=$((elapsed + 1))
        done
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Server did not stop within ${SERVER_SHUTDOWN_TIMEOUT}s; sending SIGKILL..."
            kill -KILL "$SERVER_PID" 2>/dev/null || true
        fi
        wait "$SERVER_PID" 2>/dev/null || true
    fi

    return "$exit_code"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

echo "============================================"
echo "VisualGen Serving Benchmark"
echo "============================================"
echo "Model:               $MODEL"
echo "Backend:             $BACKEND"
echo "Server:              http://${HOST}:${PORT}"
echo "Size:                $SIZE"
if [ "$BACKEND" = "openai-videos" ]; then
echo "Num frames:          $NUM_FRAMES"
echo "FPS:                 $FPS"
fi
echo "Inference steps:     $NUM_INFERENCE_STEPS"
echo "Guidance scale:      $GUIDANCE_SCALE"
echo "Num prompts:         $NUM_PROMPTS"
echo "Max concurrency:     $MAX_CONCURRENCY"
echo "Result dir:          $RESULT_DIR"
echo "============================================"
echo ""

# Step 1: Start server
SERVER_CMD="trtllm-serve ${MODEL} --host ${HOST} --port ${PORT}"
if [ -n "$SERVER_CONFIG" ]; then
    SERVER_CMD="${SERVER_CMD} --visual_gen_args ${SERVER_CONFIG}"
fi

echo "Step 1: Starting server..."
echo "  Command: ${SERVER_CMD}"

SERVER_LOG="${RESULT_DIR}/server.log"
mkdir -p "${RESULT_DIR}"

# Route SIGINT and SIGTERM through exit instead of Bash's signal-dependent
# default handling. This guarantees the EXIT cleanup and preserves the
# conventional caller-visible statuses, 130 and 143.
SERVER_PID=
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

$SERVER_CMD > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "  Server PID: $SERVER_PID"
echo "  Server log: $SERVER_LOG"

wait_for_server

# Step 2: Run benchmark
echo ""
echo "Step 2: Running benchmark..."

BENCHMARK_CMD="python -m tensorrt_llm.serve.scripts.benchmark_visual_gen \
    --model ${MODEL} \
    --backend ${BACKEND} \
    --host ${HOST} \
    --port ${PORT} \
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

BENCHMARK_LOG="${RESULT_DIR}/benchmark.log"

echo "  Command: ${BENCHMARK_CMD}"
echo "  Benchmark log: ${BENCHMARK_LOG}"
echo ""

eval $BENCHMARK_CMD 2>&1 | tee "${BENCHMARK_LOG}"

echo ""
echo "============================================"
echo "Benchmark complete. Results in: ${RESULT_DIR}"
echo "============================================"
