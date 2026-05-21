#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Run benchmark_serving.py at concurrency=1 with a long-prompt random workload
# that roughly matches the cookbook request (13,576 input tokens, 144 output).
# Saves both the summary JSON and the per-request *-perf_metrics.json so we
# can rebuild the cookbook stage breakdown.
#
# Usage:
#   scripts/bench_ttft.sh <label> [base_url] [extra args ...]
#
#   label    -> directory name under results/ (e.g. "agg" or "disagg")
#   base_url -> defaults to http://localhost:8000
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

LABEL="${1:?usage: bench_ttft.sh <label> [base_url] [extra args...]}"
shift
BASE_URL="${1:-http://localhost:8000}"
if [[ $# -ge 1 ]]; then shift || true; fi

# The serve launchers set --served_model_name openai/gpt-oss-120b. Use the same id
# so the OpenAI API "model" field matches; benchmark_serving will use it as the
# tokenizer fallback too, which is also fine because we point HF_HUB at the
# local hub cache via env (no downloads).
MODEL="${MODEL:-openai/gpt-oss-120b}"
# benchmark_serving needs to actually load the tokenizer; force it to the local
# checkpoint path to avoid any HF download regardless of HF_HUB_OFFLINE.
TOKENIZER="${TOKENIZER:-/home/scratch.trt_llm_data_ci/llm-models/gpt_oss/gpt-oss-120b}"
ISL="${ISL:-13576}"
OSL="${OSL:-144}"
# 32 sequential conc=1 requests. With --random-ids each prompt is fully
# distinct, so block_reuse cannot mask repeated prefills across requests.
# The first request still pays the warmup tax; analysis script drops it.
NUM_PROMPTS="${NUM_PROMPTS:-32}"
CONCURRENCY="${CONCURRENCY:-1}"
SEED="${SEED:-42}"

# Avoid any HF download for the tokenizer used by benchmark_serving.py.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

RESULT_DIR="${REPO_ROOT}/results/${LABEL}"
mkdir -p "${RESULT_DIR}"

echo "Bench label   : ${LABEL}"
echo "Endpoint      : ${BASE_URL}"
echo "Model         : ${MODEL}"
echo "ISL/OSL       : ${ISL}/${OSL}"
echo "num prompts   : ${NUM_PROMPTS} (first is warmup)"
echo "concurrency   : ${CONCURRENCY}"
echo "Result dir    : ${RESULT_DIR}"

## IMPORTANT: We deliberately use --backend openai (which hits /v1/completions)
## instead of openai-chat. Reason: in current TRT-LLM only the completions
## handler sets per-request sampling_params.return_perf_metrics=True (gated
## behind the TRTLLM_KVCACHE_TIME_OUTPUT_PATH env var, set by launch_agg.sh
## and launch_disagg.sh). The chat handler does not, so /perf_metrics returns
## an empty list and --save-request-time-breakdown is a no-op.
## See tensorrt_llm/serve/openai_server.py:1383.
python3 -m tensorrt_llm.serve.scripts.benchmark_serving \
  --model "${MODEL}" \
  --tokenizer "${TOKENIZER}" \
  --backend openai \
  --base-url "${BASE_URL}" \
  --endpoint "/v1/completions" \
  --dataset-name random \
  --random-input-len "${ISL}" \
  --random-output-len "${OSL}" \
  --random-ids \
  --num-prompts "${NUM_PROMPTS}" \
  --max-concurrency "${CONCURRENCY}" \
  --ignore-eos \
  --percentile-metrics "ttft,tpot,itl,e2el" \
  --seed "${SEED}" \
  --save-result \
  --save-request-time-breakdown \
  --result-dir "${RESULT_DIR}" \
  --result-filename "${LABEL}.json" \
  "$@" \
  2>&1 | tee "${RESULT_DIR}/${LABEL}.bench.log"

echo "Wrote:"
ls -la "${RESULT_DIR}"
