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
# Startup benchmark runner for TensorRT-LLM.
# Runs a single configuration N times and collects startup profile artifacts.
#
# Usage:
#   ./run_startup_bench.sh --model <hf_id> --tp <N> --tier <s1|s2|s3> \
#       [--runs <N>] [--port <P>] [--bs <B>] [--nt <T>] [--seq-len <L>] \
#       [--no-autotuner] [--result-dir <dir>] [--nfs-path <path>]
#
# Examples:
#   # B1-S1: Qwen 7B remote cold, 5 runs
#   ./run_startup_bench.sh --model Qwen/Qwen2.5-7B-Instruct --tp 1 --tier s1 --runs 5
#
#   # B2-S1-C: Qwen 72B remote cold, autotuner OFF
#   ./run_startup_bench.sh --model Qwen/Qwen2.5-72B-Instruct --tp 8 --tier s1 --runs 5 --no-autotuner
#
#   # B2-S1-D1: Qwen 72B, large serving config
#   ./run_startup_bench.sh --model Qwen/Qwen2.5-72B-Instruct --tp 8 --tier s1 --runs 5 --bs 64 --nt 8192
#
#   # B2-S1-D2: Qwen 72B, long sequence
#   ./run_startup_bench.sh --model Qwen/Qwen2.5-72B-Instruct --tp 8 --tier s1 --runs 5 --seq-len 16384
#
#   # B2-S2: Qwen 72B NFS cache (requires --nfs-path)
#   ./run_startup_bench.sh --model Qwen/Qwen2.5-72B-Instruct --tp 8 --tier s2 --runs 5 \
#       --nfs-path /path/to/nfs/Qwen2.5-72B-Instruct

set -euo pipefail

# --- Defaults ---
MODEL=""
TP=1
TIER="s1"
RUNS=5
PORT=8020
BS=4
NT=1024
SEQ_LEN=4096
NO_AUTOTUNER=false
RESULT_DIR="/tmp/trtllm-startup-bench"
NFS_PATH=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --tp) TP="$2"; shift 2 ;;
        --tier) TIER="$2"; shift 2 ;;
        --runs) RUNS="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --bs) BS="$2"; shift 2 ;;
        --nt) NT="$2"; shift 2 ;;
        --seq-len) SEQ_LEN="$2"; shift 2 ;;
        --no-autotuner) NO_AUTOTUNER=true; shift ;;
        --result-dir) RESULT_DIR="$2"; shift 2 ;;
        --nfs-path) NFS_PATH="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "ERROR: --model is required"
    exit 1
fi

if [[ "$TIER" == "s2" && -z "$NFS_PATH" ]]; then
    echo "ERROR: --nfs-path is required for tier s2 (NFS cache)"
    exit 1
fi

MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
CONFIG_TAG="tp${TP}_bs${BS}_nt${NT}_sl${SEQ_LEN}"
if $NO_AUTOTUNER; then CONFIG_TAG="${CONFIG_TAG}_noauto"; fi
RUN_TAG="${MODEL_SHORT}_${TIER}_${CONFIG_TAG}"

echo "============================================"
echo "Startup Benchmark Configuration"
echo "============================================"
echo "  Model:       $MODEL"
echo "  TP:          $TP"
echo "  Tier:        $TIER"
echo "  Runs:        $RUNS"
echo "  Port:        $PORT"
echo "  Batch size:  $BS"
echo "  Max tokens:  $NT"
echo "  Seq len:     $SEQ_LEN"
echo "  Autotuner:   $(if $NO_AUTOTUNER; then echo OFF; else echo ON; fi)"
echo "  Result dir:  $RESULT_DIR"
echo "  Run tag:     $RUN_TAG"
echo "============================================"

mkdir -p "$RESULT_DIR"

# --- Build extra trtllm-serve args ---
EXTRA_SERVE_ARGS=""
if $NO_AUTOTUNER; then
    AUTOTUNER_CFG="$RESULT_DIR/_autotuner_off.yaml"
    echo "enable_autotuner: false" > "$AUTOTUNER_CFG"
    EXTRA_SERVE_ARGS="--extra_llm_api_options $AUTOTUNER_CFG"
fi

# --- Helper: kill server and wait ---
cleanup_server() {
    local pid=$1
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
}

# --- Run loop ---
for i in $(seq 1 "$RUNS"); do
    RUN_DIR="$RESULT_DIR/${RUN_TAG}/run_${i}"
    mkdir -p "$RUN_DIR"
    echo ""
    echo ">>> Run $i/$RUNS  [$(date '+%Y-%m-%d %H:%M:%S')]"
    echo "    Output: $RUN_DIR"

    # --- Tier-specific setup ---
    case "$TIER" in
        s1)
            # Remote cold: isolated empty HF cache
            export HF_HOME="$RUN_DIR/hf-home"
            export HUGGINGFACE_HUB_CACHE="$RUN_DIR/hf-hub"
            mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE"
            SERVE_MODEL="$MODEL"
            ;;
        s2)
            # NFS cache: use existing NFS path, drop page cache
            unset HF_HOME HUGGINGFACE_HUB_CACHE 2>/dev/null || true
            if command -v sudo &>/dev/null; then
                sync && echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || \
                    echo "    WARNING: Could not drop page cache (no sudo?)"
            fi
            SERVE_MODEL="$NFS_PATH"
            ;;
        s3)
            # Local node cache: use NFS path but do NOT drop cache (warm from prior run)
            unset HF_HOME HUGGINGFACE_HUB_CACHE 2>/dev/null || true
            if [[ -z "$NFS_PATH" ]]; then
                # If no NFS path, S3 uses the HF cache from a prior S1 run
                echo "    NOTE: S3 without --nfs-path relies on prior S1 run populating HF cache"
                SERVE_MODEL="$MODEL"
            else
                SERVE_MODEL="$NFS_PATH"
            fi
            ;;
        *)
            echo "ERROR: Unknown tier '$TIER'. Use s1, s2, or s3."
            exit 1
            ;;
    esac

    # --- Start server ---
    export TRTLLM_PROFILE_STARTUP=1
    export TRTLLM_STARTUP_PROFILE_OUTPUT="$RUN_DIR/startup_profile_server.json"
    export TRT_LLM_NO_LIB_INIT=1

    trtllm-serve "$SERVE_MODEL" \
        --backend pytorch \
        --host 127.0.0.1 \
        --port "$PORT" \
        --tensor_parallel_size "$TP" \
        --max_batch_size "$BS" \
        --max_num_tokens "$NT" \
        --max_seq_len "$SEQ_LEN" \
        $EXTRA_SERVE_ARGS \
        > "$RUN_DIR/server.log" 2>&1 &
    SERVER_PID=$!

    # --- Run benchmark client ---
    python "$SCRIPT_DIR/benchmark_serving.py" \
        --backend openai \
        --base-url "http://127.0.0.1:$PORT" \
        --model "$MODEL" \
        --tokenizer "$MODEL" \
        --dataset-name random --random-ids \
        --num-prompts 1 \
        --random-input-len 16 --random-output-len 8 \
        --request-rate inf \
        --save-result \
        --save-startup-metrics \
        --startup-timeout 7200 \
        --result-dir "$RUN_DIR" \
        > "$RUN_DIR/benchmark.log" 2>&1 || {
            echo "    WARNING: Benchmark client failed on run $i"
            cleanup_server $SERVER_PID
            continue
        }

    # --- Shutdown server ---
    cleanup_server $SERVER_PID
    echo "    Done. Server profile: $RUN_DIR/startup_profile_server.json"
done

echo ""
echo "============================================"
echo "All $RUNS runs complete for $RUN_TAG"
echo "Results in: $RESULT_DIR/${RUN_TAG}/"
echo "============================================"
echo ""
echo "Next: run post-processing to extract median/min/max:"
echo "  python $SCRIPT_DIR/aggregate_startup_results.py $RESULT_DIR/${RUN_TAG}/"
