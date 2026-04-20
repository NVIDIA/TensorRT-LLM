#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Startup benchmark runner for TensorRT-LLM.
#
# Storage tiers:
#   s1 - Remote cold: fresh HF download to /tmp (tmpfs) each run
#   s2 - NFS cold: model on NFS, page cache dropped before each run
#   s3 - Local warm: model on NFS, page cache warm (2nd replica scenario)
#
# S2/S3 require --nfs-path pointing to a local model directory on NFS.

set -euo pipefail

MODEL=""
TP=1
TIER="s1"
RUNS=5
PORT=8020
BS=4
NT=1024
SEQ_LEN=4096
NO_AUTOTUNER=false
RESULT_DIR="/home/scratch.chienchunh_coreai/dev/startup-bench-v2"
NFS_PATH=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
    echo "ERROR: --model is required"; exit 1
fi

if [[ "$TIER" == "s2" || "$TIER" == "s3" ]] && [[ -z "$NFS_PATH" ]]; then
    echo "ERROR: --nfs-path is required for tier $TIER"; exit 1
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
if [[ -n "$NFS_PATH" ]]; then echo "  NFS path:    $NFS_PATH"; fi
echo "============================================"

mkdir -p "$RESULT_DIR"

EXTRA_SERVE_ARGS=""
if $NO_AUTOTUNER; then
    AUTOTUNER_CFG="$RESULT_DIR/_autotuner_off.yaml"
    echo "enable_autotuner: false" > "$AUTOTUNER_CFG"
    EXTRA_SERVE_ARGS="--extra_llm_api_options $AUTOTUNER_CFG"
fi

cleanup_server() {
    local pid=$1
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
}

for i in $(seq 1 "$RUNS"); do
    RUN_DIR="$RESULT_DIR/${RUN_TAG}/run_${i}"
    mkdir -p "$RUN_DIR"
    echo ""
    echo ">>> Run $i/$RUNS  [$(date '+%Y-%m-%d %H:%M:%S')]"
    echo "    Output: $RUN_DIR"

    PREV_S2_COPY=""  # track previous S2 copy for cleanup
    case "$TIER" in
        s1)
            # Remote cold: download to /tmp (tmpfs) for fast I/O
            S1_TMP="/tmp/trtllm-s1-${MODEL_SHORT}-run${i}"
            rm -rf "$S1_TMP"
            mkdir -p "$S1_TMP/hf-home" "$S1_TMP/hf-hub"
            export HF_HOME="$S1_TMP/hf-home"
            export HUGGINGFACE_HUB_CACHE="$S1_TMP/hf-hub"
            SERVE_MODEL="$MODEL"
            ;;
        s2)
            # NFS cold: copy model to a fresh per-run directory so page cache
            # has never seen these inodes. This guarantees true NFS reads
            # without needing drop_caches privileges.
            unset HF_HOME HUGGINGFACE_HUB_CACHE 2>/dev/null || true
            export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
            S2_COPY="$RESULT_DIR/_s2_nfs_cold/${MODEL_SHORT}_run${i}"
            # Clean up previous run's copy to save disk space
            if [[ $i -gt 1 ]]; then
                PREV_S2_COPY="$RESULT_DIR/_s2_nfs_cold/${MODEL_SHORT}_run$((i-1))"
                echo "    Cleaning up previous S2 copy: $PREV_S2_COPY"
                rm -rf "$PREV_S2_COPY" &
            fi
            echo "    Copying model to fresh NFS path for cold read: $S2_COPY"
            mkdir -p "$S2_COPY"
            cp -rL "$NFS_PATH"/* "$S2_COPY/"
            echo "    Copy complete: $(du -sh "$S2_COPY" | awk '{print $1}')"
            SERVE_MODEL="$S2_COPY"
            ;;
        s3)
            # Local warm: model on NFS, page cache hot from prior access
            unset HF_HOME HUGGINGFACE_HUB_CACHE 2>/dev/null || true
            export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
            SERVE_MODEL="$NFS_PATH"
            ;;
        *)
            echo "ERROR: Unknown tier '$TIER'"; exit 1 ;;
    esac

    RUN_PORT=$((PORT + i - 1))

    export TRTLLM_PROFILE_STARTUP=1
    export TRTLLM_STARTUP_PROFILE_OUTPUT="$RUN_DIR/startup_profile_server.json"

    TOKENIZER_ARGS=""
    CLIENT_TOKENIZER="$MODEL"
    if [[ "$TIER" == "s2" || "$TIER" == "s3" ]]; then
        TOKENIZER_ARGS="--tokenizer $SERVE_MODEL"
        CLIENT_TOKENIZER="$NFS_PATH"
    fi

    trtllm-serve "$SERVE_MODEL" \
        --backend pytorch \
        --host 127.0.0.1 \
        --port "$RUN_PORT" \
        --tensor_parallel_size "$TP" \
        --max_batch_size "$BS" \
        --max_num_tokens "$NT" \
        --max_seq_len "$SEQ_LEN" \
        $TOKENIZER_ARGS \
        $EXTRA_SERVE_ARGS \
        > "$RUN_DIR/server.log" 2>&1 &
    SERVER_PID=$!

    python "$SCRIPT_DIR/benchmark_serving.py" \
        --backend openai \
        --base-url "http://127.0.0.1:$RUN_PORT" \
        --model "$MODEL" \
        --tokenizer "$CLIENT_TOKENIZER" \
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
            sleep 2
            continue
        }

    cleanup_server $SERVER_PID
    sleep 2

    echo "    Done. Server profile: $RUN_DIR/startup_profile_server.json"
done

# Clean up last S2 copy if applicable
if [[ "$TIER" == "s2" ]]; then
    LAST_S2="$RESULT_DIR/_s2_nfs_cold/${MODEL_SHORT}_run${RUNS}"
    echo "Cleaning up last S2 copy: $LAST_S2"
    rm -rf "$LAST_S2"
fi

echo ""
echo "============================================"
echo "All $RUNS runs complete for $RUN_TAG"
echo "Results in: $RESULT_DIR/${RUN_TAG}/"
echo "============================================"
