#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run the RWLT client against an already-running trtllm-serve agg or disagg
# endpoint on the same node. No sflow, no sbatch, no srun.
#
# Usage:
#   scripts/run_rwlt.sh <label> [base_url] [rwlt_config_basename]
#     label                -- output dir name under rwlt-results/ (e.g. agg or disagg)
#     base_url             -- defaults to http://localhost:8000/v1
#     rwlt_config_basename -- defaults to rwlt_smoke (smoke test; pass
#                             rwlt_baseline for the 30-trajectory baseline)
#
# Env vars:
#   AA_REPO       -- path to /home/scratch.bbuddharaju_gpu/artificial-analysis
#                    (default uses that path)
#   MODEL_API_ID  -- request body "model" field; defaults to openai/gpt-oss-120b
#                    (matches our trtllm-serve --served_model_name default)
#   SEED          -- override seed in the RWLT config (default keeps config value)
#   CONCS         -- override concurrencies, e.g. "1,2,4"
set -euo pipefail

cd "$(dirname "$0")/.."
REPRO_DIR="$(pwd)"

LABEL="${1:?usage: run_rwlt.sh <label> [base_url] [rwlt_config_basename]}"
BASE_URL="${2:-http://localhost:8000/v1}"
CONFIG_NAME="${3:-rwlt_smoke}"

AA_REPO="${AA_REPO:-/home/scratch.bbuddharaju_gpu/artificial-analysis}"
MODEL_API_ID="${MODEL_API_ID:-openai/gpt-oss-120b}"

SRC_CONFIG="${REPRO_DIR}/configs/${CONFIG_NAME}.yaml"
RESULTS_DIR="${REPRO_DIR}/rwlt-results/${LABEL}"
mkdir -p "${RESULTS_DIR}"

if [[ ! -f "${SRC_CONFIG}" ]]; then
  echo "ERROR: ${SRC_CONFIG} not found" >&2
  exit 1
fi

# Build CLI overrides only when the user actually set them.
EXTRA_ARGS=()
if [[ -n "${SEED:-}" ]]; then EXTRA_ARGS+=(--seed "${SEED}"); fi
if [[ -n "${CONCS:-}" ]]; then EXTRA_ARGS+=(--concurrencies "${CONCS}"); fi

echo "RWLT label    : ${LABEL}"
echo "AA_REPO       : ${AA_REPO}"
echo "base_url      : ${BASE_URL}"
echo "model api id  : ${MODEL_API_ID}"
echo "rwlt config   : ${SRC_CONFIG}"
echo "results dir   : ${RESULTS_DIR}"
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo "extra args    : ${EXTRA_ARGS[*]}"

# `uv run` resolves rwlt's pyproject.toml deps into a venv under
# ${AA_REPO}/.venv on first call (~10s); subsequent runs are instant.
cd "${AA_REPO}"
uv run --project "${AA_REPO}" -- \
  python3 -m rwlt.run \
    --config "${SRC_CONFIG}" \
    --base-url "${BASE_URL}" \
    --model "${MODEL_API_ID}" \
    --api-key not-required \
    --results-dir "${RESULTS_DIR}" \
    --request-log-path "${RESULTS_DIR}/rwlt_requests.jsonl" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${RESULTS_DIR}/${LABEL}.rwlt.log"

echo "Wrote:"
ls -la "${RESULTS_DIR}"
