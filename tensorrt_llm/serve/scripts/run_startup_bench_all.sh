#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Master script: runs the full v2 startup benchmark matrix (14 configs x 5 runs).
#
# Usage:
#   ./run_startup_bench_all.sh [--runs <N>] [--result-dir <dir>] [--nfs-qwen72b <path>] [--nfs-ds70b <path>]
#
# The NFS paths are required for S2/S3 tier tests. If omitted, S2/S3 tests are skipped.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$SCRIPT_DIR/run_startup_bench.sh"
AGG="$SCRIPT_DIR/aggregate_startup_results.py"

RUNS=5
RESULT_DIR="/tmp/trtllm-startup-bench-v2"
NFS_QWEN72B=""
NFS_DS70B=""
PORT=8020

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs) RUNS="$2"; shift 2 ;;
        --result-dir) RESULT_DIR="$2"; shift 2 ;;
        --nfs-qwen72b) NFS_QWEN72B="$2"; shift 2 ;;
        --nfs-ds70b) NFS_DS70B="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

QWEN_7B="Qwen/Qwen2.5-7B-Instruct"
QWEN_72B="Qwen/Qwen2.5-72B-Instruct"
DS_7B="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DS_70B="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

run_bench() {
    local label="$1"; shift
    echo ""
    echo "################################################################"
    echo "# $label"
    echo "################################################################"
    "$BENCH" "$@" --runs "$RUNS" --result-dir "$RESULT_DIR" --port "$PORT"
}

echo "Starting full v2 benchmark matrix ($RUNS runs each)"
echo "Results: $RESULT_DIR"
echo ""

# --- Part 1: Model Size Scaling (S1 remote cold) ---
run_bench "B1-S1: Qwen 7B remote cold"      --model "$QWEN_7B"  --tp 1 --tier s1
run_bench "B2-S1: Qwen 72B remote cold"      --model "$QWEN_72B" --tp 8 --tier s1
run_bench "B3-S1: DeepSeek 7B remote cold"   --model "$DS_7B"    --tp 1 --tier s1
run_bench "B4-S1: DeepSeek 70B remote cold"  --model "$DS_70B"   --tp 8 --tier s1

# --- Part 2: Storage Tier Comparison (large models) ---
if [[ -n "$NFS_QWEN72B" ]]; then
    run_bench "B2-S2: Qwen 72B NFS cache"         --model "$QWEN_72B" --tp 8 --tier s2 --nfs-path "$NFS_QWEN72B"
    run_bench "B2-S3: Qwen 72B local node cache"   --model "$QWEN_72B" --tp 8 --tier s3 --nfs-path "$NFS_QWEN72B"
else
    echo "SKIP: B2-S2, B2-S3 (no --nfs-qwen72b provided)"
fi

if [[ -n "$NFS_DS70B" ]]; then
    run_bench "B4-S2: DeepSeek 70B NFS cache"         --model "$DS_70B" --tp 8 --tier s2 --nfs-path "$NFS_DS70B"
    run_bench "B4-S3: DeepSeek 70B local node cache"   --model "$DS_70B" --tp 8 --tier s3 --nfs-path "$NFS_DS70B"
else
    echo "SKIP: B4-S2, B4-S3 (no --nfs-ds70b provided)"
fi

# --- Part 3: Autotuner Impact (Group C) ---
run_bench "B2-S1-C: Qwen 72B no autotuner"     --model "$QWEN_72B" --tp 8 --tier s1 --no-autotuner
run_bench "B4-S1-C: DeepSeek 70B no autotuner"  --model "$DS_70B"   --tp 8 --tier s1 --no-autotuner

# --- Part 4: Serving Config Sensitivity (Group D) ---
run_bench "B2-S1-D1: Qwen 72B large config"    --model "$QWEN_72B" --tp 8 --tier s1 --bs 64 --nt 8192
run_bench "B4-S1-D1: DeepSeek 70B large config" --model "$DS_70B"   --tp 8 --tier s1 --bs 64 --nt 8192
run_bench "B2-S1-D2: Qwen 72B long seq"        --model "$QWEN_72B" --tp 8 --tier s1 --seq-len 16384
run_bench "B4-S1-D2: DeepSeek 70B long seq"    --model "$DS_70B"   --tp 8 --tier s1 --seq-len 16384

# --- Post-processing ---
echo ""
echo "================================================================"
echo "All benchmarks complete. Running aggregation..."
echo "================================================================"

for dir in "$RESULT_DIR"/*/; do
    if [[ -d "$dir/run_1" ]]; then
        echo "Aggregating: $dir"
        python "$AGG" "$dir" || echo "  WARNING: aggregation failed for $dir"
    fi
done

echo ""
echo "================================================================"
echo "Full v2 benchmark matrix complete."
echo "Results: $RESULT_DIR"
echo "================================================================"
