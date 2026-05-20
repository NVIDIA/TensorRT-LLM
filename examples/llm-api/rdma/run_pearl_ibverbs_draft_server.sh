#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "${REPO_DIR}"

OUT_DIR="${OUT_DIR:-${REPO_DIR}/tmp/pearl_ibverbs_one_case}"
mkdir -p "${OUT_DIR}"

# Set DRAFT_MODEL="" to force lazy-load from the target's TcpModelInit.
DRAFT_MODEL="${DRAFT_MODEL-/scratch.trt_llm_data/llm-models/llama-3.1-model/Llama-3.1-8B-Instruct}"
DRAFT_VISIBLE_GPUS="${DRAFT_VISIBLE_GPUS:-7}"
DRAFT_CONTROL_PORT="${DRAFT_CONTROL_PORT:-47331}"
TRANSPORT="${TRANSPORT:-ibverbs}"
NIC="${NIC:-mlx5_0}"
PREFETCH_WAIT_TIMEOUT_S="${PREFETCH_WAIT_TIMEOUT_S:-0.05}"
TRTLLM_STREAM_MAX_TOKENS="${TRTLLM_STREAM_MAX_TOKENS:-16384}"
TRACE_LOG="${DRAFT_TRACE_LOG:-${OUT_DIR}/draft_trace.jsonl}"
LOG_FILE="${DRAFT_LOG_FILE:-${OUT_DIR}/draft_server.log}"

export PYTHONPATH="${PYTHONPATH:-${REPO_DIR}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DRAFT_VISIBLE_GPUS}}"
export GPU_ID="${GPU_ID:-0}"
export TLLM_WORKER_USE_SINGLE_PROCESS="${TLLM_WORKER_USE_SINGLE_PROCESS:-1}"
export PEARL_PREFETCH_WAIT_TIMEOUT_S="${PEARL_PREFETCH_WAIT_TIMEOUT_S:-${PREFETCH_WAIT_TIMEOUT_S}}"
export PEARL_TRTLLM_STREAM_MAX_TOKENS="${PEARL_TRTLLM_STREAM_MAX_TOKENS:-${TRTLLM_STREAM_MAX_TOKENS}}"

MODEL_ARGS=()
if [[ -n "${DRAFT_MODEL}" ]]; then
  MODEL_ARGS+=(--model "${DRAFT_MODEL}")
fi

echo "===== PEARL draft server ====="
echo "repo: ${REPO_DIR}"
echo "out_dir: ${OUT_DIR}"
echo "model: ${DRAFT_MODEL:-<lazy-load from target>}"
echo "gpu: ${CUDA_VISIBLE_DEVICES}"
echo "transport: ${TRANSPORT}"
echo "nic: ${NIC}"
echo "control_port: ${DRAFT_CONTROL_PORT}"
echo "trtllm_stream_max_tokens: ${TRTLLM_STREAM_MAX_TOKENS}"
echo "trace_log: ${TRACE_LOG}"

python3 examples/llm-api/rdma/trtllm_pearl_draft_server.py \
  "${MODEL_ARGS[@]}" \
  --backend trtllm \
  --transport "${TRANSPORT}" \
  --nic "${NIC}" \
  --control-port "${DRAFT_CONTROL_PORT}" \
  --data-port 0 \
  --backend-init-timeout-s 3600 \
  --trtllm-stream-max-tokens "${TRTLLM_STREAM_MAX_TOKENS}" \
  --prefetch-wait-timeout-s "${PREFETCH_WAIT_TIMEOUT_S}" \
  --trace-log "${TRACE_LOG}" \
  2>&1 | tee "${LOG_FILE}"
