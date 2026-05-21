#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Launch aggregated trtllm-serve for gpt-oss-120b on a single GPU.
#
# By default this **backgrounds** the server (nohup, separate log file) so you
# can run scripts/bench_ttft.sh from the same shell -- no second docker exec
# required. Use --foreground to keep it attached.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 scripts/launch_agg.sh [config_basename]
#   CUDA_VISIBLE_DEVICES=0 scripts/launch_agg.sh repro_agg_tp1_nospec  # ablation
#   CUDA_VISIBLE_DEVICES=0 scripts/launch_agg.sh --foreground          # attached
#
# Defaults to MODEL and EAGLE_CKPT pointing at shared local checkpoints (no
# HF downloads). Override either env var if you need a different checkpoint.
#
# config_basename defaults to repro_agg_tp1_eagle3.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

FOREGROUND=0
CONFIG_NAME="repro_agg_tp1_eagle3"
for arg in "$@"; do
  case "${arg}" in
    --foreground|-f) FOREGROUND=1 ;;
    --background|-b) FOREGROUND=0 ;;
    -*) echo "Unknown flag: ${arg}" >&2; exit 2 ;;
    *)  CONFIG_NAME="${arg}" ;;
  esac
done

SRC_CONFIG="${REPO_ROOT}/configs/${CONFIG_NAME}.yaml"
RUN_CONFIG="${REPO_ROOT}/configs/.runtime_${CONFIG_NAME}.yaml"

# Default to the shared local checkpoint to avoid HF downloads.
MODEL="${MODEL:-/home/scratch.trt_llm_data_ci/llm-models/gpt_oss/gpt-oss-120b}"
# Stable API id, so the bench client never depends on the local checkpoint path.
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-openai/gpt-oss-120b}"
HOST="${HOST:-localhost}"
PORT="${PORT:-8000}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
PID_DIR="${REPO_ROOT}/logs/pids"
mkdir -p "${LOG_DIR}" "${PID_DIR}"

# Belt-and-suspenders: prevent any HF download attempt.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
# Required for the /v1/completions handler to set sampling_params.return_perf_metrics=True
# per request, which is what populates the /perf_metrics deque used by
# benchmark_serving.py --save-request-time-breakdown.
# (Chat completions never sets this in current TRT-LLM, regardless of the
# server-level return_perf_metrics: true setting -- see
# tensorrt_llm/serve/openai_server.py:1383 vs 1045-1050.)
export TRTLLM_KVCACHE_TIME_OUTPUT_PATH="${TRTLLM_KVCACHE_TIME_OUTPUT_PATH:-${LOG_DIR}/agg_kvcache_time}"

if [[ ! -f "${SRC_CONFIG}" ]]; then
  echo "ERROR: ${SRC_CONFIG} not found" >&2
  exit 1
fi

# Default: shared local copy of the released nvidia/gpt-oss-120b-Eagle3-v3 model
# (a.k.a. the pre-release "Eagle3-next" name the cookbook configs refer to).
EAGLE_CKPT="${EAGLE_CKPT:-/home/scratch.simengl_sw_3/trt_repos/hf_models/nvidia/gpt-oss-120b-Eagle3-v3}"
sed "s|EAGLE_CKPT_PLACEHOLDER|${EAGLE_CKPT}|g" "${SRC_CONFIG}" > "${RUN_CONFIG}"
echo "Resolved config -> ${RUN_CONFIG}"
echo "  EAGLE_CKPT=${EAGLE_CKPT}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  MODEL=${MODEL}"
echo "  served_model_name=${SERVED_MODEL_NAME}"
echo "  ${HOST}:${PORT}"
echo "  log dir=${LOG_DIR}"
echo "  background=$([[ ${FOREGROUND} -eq 0 ]] && echo yes || echo no)"

if [[ ${FOREGROUND} -eq 1 ]]; then
  exec trtllm-serve "${MODEL}" \
    --host "${HOST}" --port "${PORT}" --backend pytorch \
    --served_model_name "${SERVED_MODEL_NAME}" \
    --extra_llm_api_options "${RUN_CONFIG}" \
    2>&1 | tee "${LOG_DIR}/agg.log"
fi

# Background launch.
nohup trtllm-serve "${MODEL}" \
  --host "${HOST}" --port "${PORT}" --backend pytorch \
  --served_model_name "${SERVED_MODEL_NAME}" \
  --extra_llm_api_options "${RUN_CONFIG}" \
  > "${LOG_DIR}/agg.log" 2>&1 &
AGG_PID=$!
echo "${AGG_PID}" > "${PID_DIR}/agg.pid"
echo "agg PID ${AGG_PID} -> tail -f ${LOG_DIR}/agg.log"

# Wait for /health.
TIMEOUT="${HEALTH_TIMEOUT:-1800}"
URL="http://${HOST}:${PORT}/health"
echo "Waiting for ${URL} (timeout ${TIMEOUT}s) ..."
START=$(date +%s)
while true; do
  if ! kill -0 "${AGG_PID}" 2>/dev/null; then
    echo "ERROR: trtllm-serve process exited. Tail of log:" >&2
    tail -n 80 "${LOG_DIR}/agg.log" >&2 || true
    exit 1
  fi
  if curl -sf -o /dev/null "${URL}"; then
    echo "agg server ready after $(( $(date +%s) - START ))s"
    exit 0
  fi
  if (( $(date +%s) - START > TIMEOUT )); then
    echo "ERROR: ${URL} did not become healthy within ${TIMEOUT}s" >&2
    exit 1
  fi
  sleep 5
done
