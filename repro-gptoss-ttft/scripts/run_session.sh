#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end benchmarking session orchestrator. Launches the appropriate
# trtllm-serve setup (agg or disagg), runs the requested client (rwlt or
# bench_serving), and ALWAYS tears the servers down -- even on Ctrl-C or
# client failure -- so the next session starts on a cold engine.
#
# Usage:
#   scripts/run_session.sh <layout> <client> <label> [client_extra_args...]
#
#     layout: agg | disagg
#     client: rwlt_smoke | rwlt_baseline | bench_serving
#     label : output dir basename under {results,rwlt-results}/<label>
#
# Examples:
#   # Agg + RWLT smoke (~3 min, validates the toolchain)
#   CUDA_VISIBLE_DEVICES=0 scripts/run_session.sh agg rwlt_smoke agg
#
#   # Disagg + RWLT baseline (~10-15 min, 30 trajectories at conc=1)
#   CTX_GPU=0 GEN_GPU=1 scripts/run_session.sh disagg rwlt_baseline disagg
#
#   # Agg + benchmark_serving (32 random-IDs prompts at conc=1)
#   CUDA_VISIBLE_DEVICES=0 scripts/run_session.sh agg bench_serving agg
#
# Env vars forwarded to launchers:
#   CUDA_VISIBLE_DEVICES (agg)
#   CTX_GPU, GEN_GPU (disagg)
#   EAGLE_CKPT, MODEL, SERVED_MODEL_NAME (both)
#   SEED, CONCS (rwlt only)
set -uo pipefail

cd "$(dirname "$0")/.."
REPRO_DIR="$(pwd)"

LAYOUT="${1:?usage: run_session.sh <layout> <client> <label> [args...]}"
CLIENT="${2:?usage: run_session.sh <layout> <client> <label> [args...]}"
LABEL="${3:?usage: run_session.sh <layout> <client> <label> [args...]}"
shift 3 || true

case "${LAYOUT}" in
  agg|disagg) ;;
  *) echo "ERROR: layout must be 'agg' or 'disagg', got ${LAYOUT}" >&2; exit 2 ;;
esac
case "${CLIENT}" in
  rwlt_smoke|rwlt_baseline|bench_serving) ;;
  *) echo "ERROR: client must be rwlt_smoke|rwlt_baseline|bench_serving, got ${CLIENT}" >&2; exit 2 ;;
esac

teardown() {
  echo
  echo "[run_session] tearing down ${LAYOUT} servers ..."
  if [[ "${LAYOUT}" == "agg" ]]; then
    "${REPRO_DIR}/scripts/stop_agg.sh" || true
  else
    "${REPRO_DIR}/scripts/stop_disagg.sh" || true
  fi
  echo "[run_session] teardown complete."
}
# Trap covers normal exit, Ctrl-C, and bench failures.
trap teardown EXIT INT TERM

echo "[run_session] launching ${LAYOUT} ..."
# Allow overriding the config basenames per session so the same wrapper can
# drive baseline (repro_*_eagle3) and ablation (repro_*_tuned, _nospec, ...) runs.
AGG_CONFIG_BASE="${AGG_CONFIG_BASE:-repro_agg_tp1_eagle3}"
if [[ "${LAYOUT}" == "agg" ]]; then
  "${REPRO_DIR}/scripts/launch_agg.sh" "${AGG_CONFIG_BASE}"
  BASE_URL_HTTP="http://localhost:8000"
  BASE_URL_V1="http://localhost:8000/v1"
else
  # launch_disagg.sh honours CTX_CONFIG_BASE / GEN_CONFIG_BASE / PROXY_CONFIG_BASE
  # env vars already; nothing extra to do here.
  "${REPRO_DIR}/scripts/launch_disagg.sh"
  BASE_URL_HTTP="http://localhost:8000"
  BASE_URL_V1="http://localhost:8000/v1"
fi

echo "[run_session] running client: ${CLIENT} (label=${LABEL}) ..."
case "${CLIENT}" in
  rwlt_smoke|rwlt_baseline)
    "${REPRO_DIR}/scripts/run_rwlt.sh" "${LABEL}" "${BASE_URL_V1}" "${CLIENT}" "$@"
    ;;
  bench_serving)
    "${REPRO_DIR}/scripts/bench_ttft.sh" "${LABEL}" "${BASE_URL_HTTP}" "$@"
    ;;
esac

echo "[run_session] client finished. (teardown will run on exit)"
