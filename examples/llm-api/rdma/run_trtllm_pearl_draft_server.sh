#!/usr/bin/env bash
set -euo pipefail

# Product-style PEARL draft-server launcher.
#
# GPU visibility is intentionally controlled here.  Example:
#
#   DRAFT_VISIBLE_GPUS=7 ./examples/llm-api/rdma/run_trtllm_pearl_draft_server.sh \
#     /scratch.trt_llm_data/llm-models/Qwen3/Qwen3-0.6B

MODEL="${1:-/scratch.trt_llm_data/llm-models/Qwen3/Qwen3-0.6B}"
VISIBLE_GPUS="${DRAFT_VISIBLE_GPUS:-7}"
CONTROL_PORT="${PEARL_DRAFT_CONTROL_PORT:-47331}"
NIC="${PEARL_DRAFT_NIC:-mlx5_0}"
TRANSPORT="${PEARL_DRAFT_TRANSPORT:-ibverbs}"
TRACE_LOG="${PEARL_DRAFT_TRACE_LOG:-/code/tensorrt_llm/tmp/pearl_draft_server.jsonl}"
PYTHONPATH_VALUE="${PYTHONPATH:-/code/tensorrt_llm}"

mkdir -p "$(dirname "${TRACE_LOG}")"

export PYTHONPATH="${PYTHONPATH_VALUE}"
export CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}"
export GPU_ID="${GPU_ID:-0}"
export TLLM_WORKER_USE_SINGLE_PROCESS="${TLLM_WORKER_USE_SINGLE_PROCESS:-1}"

exec python3 examples/llm-api/rdma/trtllm_pearl_draft_server.py \
  --model "${MODEL}" \
  --backend pyexecutor \
  --transport "${TRANSPORT}" \
  --nic "${NIC}" \
  --control-port "${CONTROL_PORT}" \
  --data-port 0 \
  --backend-init-timeout-s 3600 \
  --trtllm-stream-max-tokens "${PEARL_TRTLLM_STREAM_MAX_TOKENS:-2048}" \
  --prefetch-wait-timeout-s "${PEARL_PREFETCH_WAIT_TIMEOUT_S:-0.05}" \
  --trace-log "${TRACE_LOG}"
