#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Launch a 1ctx + 1gen disaggregated trtllm-serve setup for gpt-oss-120b.
#
# Spawns three processes in the background, redirecting each to its own log:
#   - ctx worker (CUDA_VISIBLE_DEVICES=<CTX_GPU>, port 8001)
#   - gen worker (CUDA_VISIBLE_DEVICES=<GEN_GPU>, port 8002)
#   - disagg proxy (port 8000) once both workers are healthy
#
# Env vars:
#   EAGLE_CKPT      path or HF id of Eagle draft checkpoint (defaults to the
#                   shared local copy of nvidia/gpt-oss-120b-Eagle3-v3,
#                   the released-name for the cookbook's "Eagle3-next").
#   CTX_GPU         GPU index for ctx worker (default 0)
#   GEN_GPU         GPU index for gen worker (default 1)
#   CTX_CONFIG_BASE / GEN_CONFIG_BASE / PROXY_CONFIG_BASE
#                   config basenames (no .yaml). Defaults match the cookbook.
#   MODEL           HF id or local path of the model
#
# After this script returns, all three processes are running in the background.
# Stop them with scripts/stop_disagg.sh.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

# Default: shared local copy of the released nvidia/gpt-oss-120b-Eagle3-v3 model
# (a.k.a. the pre-release "Eagle3-next" name the cookbook configs refer to).
EAGLE_CKPT="${EAGLE_CKPT:-/home/scratch.simengl_sw_3/trt_repos/hf_models/nvidia/gpt-oss-120b-Eagle3-v3}"
CTX_GPU="${CTX_GPU:-0}"
GEN_GPU="${GEN_GPU:-1}"
# Default to the shared local checkpoint to avoid HF downloads.
MODEL="${MODEL:-/home/scratch.trt_llm_data_ci/llm-models/gpt_oss/gpt-oss-120b}"
# Stable API id, so the bench client never depends on the local checkpoint path.
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-openai/gpt-oss-120b}"
CTX_CONFIG_BASE="${CTX_CONFIG_BASE:-repro_disagg_ctx_tp1}"
GEN_CONFIG_BASE="${GEN_CONFIG_BASE:-repro_disagg_gen_tp1}"
PROXY_CONFIG_BASE="${PROXY_CONFIG_BASE:-repro_disagg_proxy}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
PID_DIR="${REPO_ROOT}/logs/pids"
mkdir -p "${LOG_DIR}" "${PID_DIR}"

# Belt-and-suspenders: prevent any HF download attempt from worker / proxy.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
# Required for /v1/completions to set sampling_params.return_perf_metrics=True
# per request (see openai_server.py:1383). Without this env var, /perf_metrics
# returns an empty list and benchmark_serving's --save-request-time-breakdown
# is a no-op even though the server-level return_perf_metrics: true is set.
export TRTLLM_KVCACHE_TIME_OUTPUT_PATH="${TRTLLM_KVCACHE_TIME_OUTPUT_PATH:-${LOG_DIR}/disagg_kvcache_time}"

resolve_config() {
  local base="$1"
  local src="${REPO_ROOT}/configs/${base}.yaml"
  local out="${REPO_ROOT}/configs/.runtime_${base}.yaml"
  if [[ ! -f "${src}" ]]; then
    echo "ERROR: ${src} not found" >&2
    return 1
  fi
  sed "s|EAGLE_CKPT_PLACEHOLDER|${EAGLE_CKPT}|g" "${src}" > "${out}"
  echo "${out}"
}

wait_for_health() {
  local url="$1"
  local label="$2"
  local timeout="${3:-1200}"
  local start
  start=$(date +%s)
  echo "Waiting for ${label} (${url}) ..."
  while true; do
    if curl -sf -o /dev/null "${url}"; then
      echo "${label} ready after $(( $(date +%s) - start ))s"
      return 0
    fi
    if (( $(date +%s) - start > timeout )); then
      echo "ERROR: ${label} did not become healthy within ${timeout}s" >&2
      return 1
    fi
    sleep 5
  done
}

CTX_RUN_CONFIG="$(resolve_config "${CTX_CONFIG_BASE}")"
GEN_RUN_CONFIG="$(resolve_config "${GEN_CONFIG_BASE}")"
PROXY_CONFIG="${REPO_ROOT}/configs/${PROXY_CONFIG_BASE}.yaml"

echo "ctx config -> ${CTX_RUN_CONFIG} (GPU ${CTX_GPU})"
echo "gen config -> ${GEN_RUN_CONFIG} (GPU ${GEN_GPU})"
echo "proxy     -> ${PROXY_CONFIG}"
echo "model     -> ${MODEL}"
echo "EAGLE_CKPT-> ${EAGLE_CKPT}"

CUDA_VISIBLE_DEVICES="${CTX_GPU}" nohup trtllm-serve "${MODEL}" \
  --host localhost --port 8001 --backend pytorch --server_role CONTEXT \
  --served_model_name "${SERVED_MODEL_NAME}" \
  --extra_llm_api_options "${CTX_RUN_CONFIG}" \
  > "${LOG_DIR}/ctx.log" 2>&1 &
echo $! > "${PID_DIR}/ctx.pid"
echo "ctx PID $(cat "${PID_DIR}/ctx.pid")"

CUDA_VISIBLE_DEVICES="${GEN_GPU}" nohup trtllm-serve "${MODEL}" \
  --host localhost --port 8002 --backend pytorch --server_role GENERATION \
  --served_model_name "${SERVED_MODEL_NAME}" \
  --extra_llm_api_options "${GEN_RUN_CONFIG}" \
  > "${LOG_DIR}/gen.log" 2>&1 &
echo $! > "${PID_DIR}/gen.pid"
echo "gen PID $(cat "${PID_DIR}/gen.pid")"

wait_for_health "http://localhost:8001/health" "ctx worker"
wait_for_health "http://localhost:8002/health" "gen worker"

nohup trtllm-serve disaggregated -c "${PROXY_CONFIG}" \
  > "${LOG_DIR}/proxy.log" 2>&1 &
echo $! > "${PID_DIR}/proxy.pid"
echo "proxy PID $(cat "${PID_DIR}/proxy.pid")"

wait_for_health "http://localhost:8000/health" "disagg proxy"
echo "All three services healthy. Disagg endpoint -> http://localhost:8000"
