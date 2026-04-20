#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Master script: runs the full v2 startup benchmark matrix.
#
# Flow for each large model:
#   1. S1 runs (download to /tmp tmpfs)
#   2. Copy last S1 download to NFS (not timed)
#   3. S2 runs (NFS cold, page cache dropped)
#   4. S3 runs (NFS warm, page cache hot)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$SCRIPT_DIR/run_startup_bench.sh"
AGG="$SCRIPT_DIR/aggregate_startup_results.py"

RUNS=3
RESULT_DIR="/home/scratch.chienchunh_coreai/dev/startup-bench-v2"
PORT=8050

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs) RUNS="$2"; shift 2 ;;
        --result-dir) RESULT_DIR="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

QWEN_7B="Qwen/Qwen2.5-7B-Instruct"
QWEN_72B="Qwen/Qwen2.5-72B-Instruct"
DS_7B="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DS_70B="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

NFS_MODEL_CACHE="$RESULT_DIR/_nfs_model_cache"
mkdir -p "$NFS_MODEL_CACHE"

run_bench() {
    local label="$1"; shift
    echo ""
    echo "################################################################"
    echo "# $label"
    echo "################################################################"
    "$BENCH" "$@" --runs "$RUNS" --result-dir "$RESULT_DIR" --port "$PORT"
}

# Copy the last S1 download from /tmp to NFS for S2/S3 use.
# Args: $1=model_short_name
copy_s1_to_nfs() {
    local model_short="$1"
    local s1_tmp="/tmp/trtllm-s1-${model_short}-run${RUNS}"
    local nfs_dest="$NFS_MODEL_CACHE/$model_short"

    if [[ ! -d "$s1_tmp/hf-hub" ]]; then
        echo "WARNING: S1 download not found at $s1_tmp/hf-hub, skipping NFS copy"
        return 1
    fi

    local snapshot_dir
    snapshot_dir=$(find "$s1_tmp/hf-hub" -type d -name "snapshots" 2>/dev/null | head -1)
    if [[ -z "$snapshot_dir" ]]; then
        echo "WARNING: No snapshots dir found in $s1_tmp/hf-hub"
        return 1
    fi

    local model_path
    model_path=$(find "$snapshot_dir" -mindepth 1 -maxdepth 1 -type d | head -1)
    if [[ -z "$model_path" ]]; then
        echo "WARNING: No model snapshot found in $snapshot_dir"
        return 1
    fi

    echo ""
    echo ">>> Copying S1 download to NFS (not timed as benchmark)"
    echo "    Source: $model_path"
    echo "    Dest:   $nfs_dest"
    rm -rf "$nfs_dest"
    cp -rL "$model_path" "$nfs_dest"
    echo "    Copy complete: $(du -sh "$nfs_dest" | awk '{print $1}')"
    return 0
}

echo "Starting full v2 benchmark matrix ($RUNS runs each)"
echo "Results: $RESULT_DIR"
echo ""

# ============================================================
# Part 1: Model Size Scaling (S1 remote cold, all 4 models)
# ============================================================
run_bench "B1-S1: Qwen 7B remote cold"      --model "$QWEN_7B"  --tp 1 --tier s1
run_bench "B2-S1: Qwen 72B remote cold"      --model "$QWEN_72B" --tp 8 --tier s1
run_bench "B3-S1: DeepSeek 7B remote cold"   --model "$DS_7B"    --tp 1 --tier s1
run_bench "B4-S1: DeepSeek 70B remote cold"  --model "$DS_70B"   --tp 8 --tier s1

# ============================================================
# Part 2: Storage Tier Comparison (large models only)
# Copy S1 downloads to NFS, then run S2 (cold NFS) and S3 (warm cache)
# ============================================================
echo ""
echo "================================================================"
echo "Preparing NFS model cache for S2/S3 tests..."
echo "================================================================"

QWEN72B_NFS=""
DS70B_NFS=""

if copy_s1_to_nfs "Qwen2.5-72B-Instruct"; then
    QWEN72B_NFS="$NFS_MODEL_CACHE/Qwen2.5-72B-Instruct"
fi

if copy_s1_to_nfs "DeepSeek-R1-Distill-Llama-70B"; then
    DS70B_NFS="$NFS_MODEL_CACHE/DeepSeek-R1-Distill-Llama-70B"
fi

if [[ -n "$QWEN72B_NFS" ]]; then
    run_bench "B2-S2: Qwen 72B NFS cache"        --model "$QWEN_72B" --tp 8 --tier s2 --nfs-path "$QWEN72B_NFS"
    run_bench "B2-S3: Qwen 72B local node cache"  --model "$QWEN_72B" --tp 8 --tier s3 --nfs-path "$QWEN72B_NFS"
else
    echo "SKIP: B2-S2, B2-S3 (S1 download not available for Qwen 72B)"
fi

if [[ -n "$DS70B_NFS" ]]; then
    run_bench "B4-S2: DeepSeek 70B NFS cache"        --model "$DS_70B" --tp 8 --tier s2 --nfs-path "$DS70B_NFS"
    run_bench "B4-S3: DeepSeek 70B local node cache"  --model "$DS_70B" --tp 8 --tier s3 --nfs-path "$DS70B_NFS"
else
    echo "SKIP: B4-S2, B4-S3 (S1 download not available for DeepSeek 70B)"
fi

# ============================================================
# Part 3: Autotuner Impact (Group C)
# ============================================================
run_bench "B2-S1-C: Qwen 72B no autotuner"     --model "$QWEN_72B" --tp 8 --tier s1 --no-autotuner
run_bench "B4-S1-C: DeepSeek 70B no autotuner"  --model "$DS_70B"   --tp 8 --tier s1 --no-autotuner

# ============================================================
# Part 4: Serving Config Sensitivity (Group D)
# ============================================================
run_bench "B2-S1-D1: Qwen 72B large config"    --model "$QWEN_72B" --tp 8 --tier s1 --bs 64 --nt 8192
run_bench "B4-S1-D1: DeepSeek 70B large config" --model "$DS_70B"   --tp 8 --tier s1 --bs 64 --nt 8192
run_bench "B2-S1-D2: Qwen 72B long seq"        --model "$QWEN_72B" --tp 8 --tier s1 --seq-len 16384
run_bench "B4-S1-D2: DeepSeek 70B long seq"    --model "$DS_70B"   --tp 8 --tier s1 --seq-len 16384

# ============================================================
# Post-processing
# ============================================================
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
