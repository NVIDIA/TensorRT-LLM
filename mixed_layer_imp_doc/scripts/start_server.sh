#!/usr/bin/env bash
# Run INSIDE the container — activates venv and starts trtllm-serve.
# Usage: bash /code/tensorrt_llm/scripts/serve/start_server.sh [port]

set -euo pipefail


VENV="/code/tensorrt_llm/.venv-3.12"
export LD_LIBRARY_PATH="/usr/local/cuda-13.1/targets/x86_64-linux/lib:${VENV}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"
export HF_HOME=/data/huggingface


PORT="${1:-8010}"
# MODEL="Qwen/Qwen3-8B"
# MODEL="nvidia/Qwen3-8B-FP8"
# MODEL="Qwen/Qwen3-VL-2B-Thinking-FP8"
MODEL="Qwen/Qwen3-4B-Thinking-2507-FP8"

SCRIPT_DIR=/scripts/serve
CONFIG="${SCRIPT_DIR}/config.yaml"


# Activate venv
# shellcheck disable=SC1091


# Prepend venv torch/lib so the correct libc10.so is found at runtime.
# CUDA 13.1 lib must come before the venv's nvidia/cu13/lib — the venv ships
# libnvJitLink.so.13 at version 13.0 but TensorRT-LLM requires >= 13.1.


# Pin to GPU 3 (already enforced by --gpus device=3 in start_docker.sh,
# but set here explicitly in case the container has multiple GPUs visible)
export CUDA_VISIBLE_DEVICES=3

echo "Starting TensorRT-LLM server"
echo "  Model   : ${MODEL}"
echo "  GPU     : CUDA device 3"
echo "  Port    : ${PORT}"
echo "  Config  : ${CONFIG}"
echo "  HF cache: ${HF_HOME}"
echo " CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# CUDA_LAUNCH_BLOCKING=1 
# CUDA_LAUNCH_BLOCKING=1  
trtllm-serve "${MODEL}" \
    --max_batch_size 128 \
    --max_num_tokens 8192 \
    --max_seq_len 32768 \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --config "${CONFIG}"
