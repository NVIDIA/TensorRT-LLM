#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# run_bench.sh - Representative single-node aggregate workload driver.
#
# Prepares a token-norm-dist dataset matching the workload's ISL/OSL and runs
# `trtllm-bench throughput` (PyTorch backend). Uses the exact flag form that
# TRT-LLM's own CI uses (tests/integration/defs/test_e2e.py), so it should
# stay valid. Invoked per-workload by run_workloads.sh via the suite `command:`.
#
# Reads (exported by run_workloads.sh from the suite, or set directly):
#   WL_MODEL_PATH   (required) absolute path to the model checkpoint
#   WL_ISL, WL_OSL  sequence lengths (accept "1024" or "1k"); default 1024
#   WL_TP           tensor-parallel size; default 8
#   WL_CONC         concurrency (passed through only via BENCH_EXTRA_ARGS)
# Optional:
#   BENCH_NUM_REQUESTS  dataset size (default 256)
#   BENCH_EXTRA_ARGS    appended verbatim to `throughput` (e.g. "--streaming
#                       --concurrency 1024 --extra_llm_api_options cfg.yaml")
#                       -- use this for cluster-known flags so we never guess.
#   BENCH_DATASET       reuse an existing dataset file instead of preparing one

set -euo pipefail

to_int() { case "$1" in *k|*K) echo $(( ${1%[kK]} * 1024 ));; *) echo "${1:-0}";; esac; }

MODEL_PATH="${WL_MODEL_PATH:?run_bench: set model_path in the suite (or WL_MODEL_PATH)}"
ISL=$(to_int "${WL_ISL:-1024}")
OSL=$(to_int "${WL_OSL:-1024}")
TP="${WL_TP:-8}"
NREQ="${BENCH_NUM_REQUESTS:-256}"

echo "[INFO] run_bench: model=$MODEL_PATH ISL=$ISL OSL=$OSL TP=$TP num_requests=$NREQ"

DATASET="${BENCH_DATASET:-}"
if [[ -z "$DATASET" ]]; then
    DATASET=$(mktemp /tmp/bolt_bench_ds_XXXXXX.jsonl)
    trtllm-bench --model "$MODEL_PATH" prepare-dataset \
        --output "$DATASET" \
        token-norm-dist \
        --input-mean "$ISL" --output-mean "$OSL" \
        --input-stdev 0 --output-stdev 0 \
        --num-requests "$NREQ"
fi

# Pass BOTH --model and --model_path: with --model_path set, trtllm-bench uses
# the LOCAL checkpoint and skips the HF snapshot_download (so a local path in
# --model isn't treated as a repo id to download).
# shellcheck disable=SC2086
trtllm-bench --model "$MODEL_PATH" --model_path "$MODEL_PATH" throughput \
    --tp "$TP" \
    --dataset "$DATASET" \
    --backend pytorch \
    ${BENCH_EXTRA_ARGS:-}
