#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Failover floor benchmark for TensorRT-LLM.
#
# Measures the lower bound on activation latency by sending requests to a
# fully warm trtllm-serve instance. Reports TTFT and end-to-end latency.
# This characterizes the floor that any failover scheme (e.g., GMS shadow
# import) must beat.
#
# Usage:
#   run_failover_floor_bench.sh \
#       --model <local_path_or_hf_id> \
#       --tp <int> \
#       [--runs 3] \
#       [--measurements 10] \
#       [--input-len 16] \
#       [--output-len 8] \
#       [--port 8070] \
#       [--result-dir <path>] \
#       [--tokenizer <path>]
#
# Output (per config):
#   <result-dir>/<model>_tp<tp>/run_<i>/m_<m>.json   per-measurement TTFT/E2E
#   <result-dir>/<model>_tp<tp>/run_<i>/summary.json per-run median/min/max
#   <result-dir>/<model>_tp<tp>/aggregate.json       across-run median-of-medians

set -euo pipefail

MODEL=""
TOKENIZER=""
TP=1
RUNS=3
MEASUREMENTS=10
INPUT_LEN=16
OUTPUT_LEN=8
PORT=8070
MAX_BATCH_SIZE=4
MAX_NUM_TOKENS=1024
MAX_SEQ_LEN=4096
RESULT_DIR="/tmp/trtllm-failover-floor"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --tokenizer) TOKENIZER="$2"; shift 2 ;;
        --tp) TP="$2"; shift 2 ;;
        --runs) RUNS="$2"; shift 2 ;;
        --measurements) MEASUREMENTS="$2"; shift 2 ;;
        --input-len) INPUT_LEN="$2"; shift 2 ;;
        --output-len) OUTPUT_LEN="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --max-batch-size) MAX_BATCH_SIZE="$2"; shift 2 ;;
        --max-num-tokens) MAX_NUM_TOKENS="$2"; shift 2 ;;
        --max-seq-len) MAX_SEQ_LEN="$2"; shift 2 ;;
        --result-dir) RESULT_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "ERROR: --model is required"; exit 1
fi

if [[ -z "$TOKENIZER" ]]; then
    TOKENIZER="$MODEL"
fi

MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
TAG="${MODEL_SHORT}_tp${TP}"
CONFIG_DIR="$RESULT_DIR/$TAG"
mkdir -p "$CONFIG_DIR"

echo "============================================"
echo "Failover Floor Benchmark"
echo "============================================"
echo "  Model:        $MODEL"
echo "  Tokenizer:    $TOKENIZER"
echo "  TP:           $TP"
echo "  Runs:         $RUNS"
echo "  Measurements: $MEASUREMENTS"
echo "  Request:      input=$INPUT_LEN output=$OUTPUT_LEN tokens"
echo "  Port:         $PORT"
echo "  Result dir:   $CONFIG_DIR"
echo "============================================"

cleanup_server() {
    local pid=$1
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
}

for run_id in $(seq 1 "$RUNS"); do
    RUN_DIR="$CONFIG_DIR/run_${run_id}"
    mkdir -p "$RUN_DIR"
    RUN_PORT=$((PORT + run_id - 1))

    echo ""
    echo ">>> Run $run_id/$RUNS  [$(date '+%H:%M:%S')]  port=$RUN_PORT"

    # 1) Start server
    trtllm-serve "$MODEL" \
        --backend pytorch \
        --host 127.0.0.1 \
        --port "$RUN_PORT" \
        --tensor_parallel_size "$TP" \
        --max_batch_size "$MAX_BATCH_SIZE" \
        --max_num_tokens "$MAX_NUM_TOKENS" \
        --max_seq_len "$MAX_SEQ_LEN" \
        --tokenizer "$TOKENIZER" \
        > "$RUN_DIR/server.log" 2>&1 &
    SERVER_PID=$!

    # 2) Wait for /health (max 600s)
    ready=0
    for i in $(seq 1 120); do
        sleep 5
        if curl -sf "http://127.0.0.1:$RUN_PORT/health" >/dev/null 2>&1; then
            ready=1
            echo "    Server ready in ~$((i * 5))s"
            break
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "    SERVER CRASHED — see $RUN_DIR/server.log"
            break
        fi
    done

    if [ $ready -eq 0 ]; then
        echo "    SKIPPING: server failed to become ready"
        cleanup_server $SERVER_PID
        continue
    fi

    # 3) Send 1 warmup request (results discarded)
    echo "    Sending warmup request..."
    curl -sf -X POST "http://127.0.0.1:$RUN_PORT/v1/completions" \
        -H 'Content-Type: application/json' \
        -d '{"model":"'"$MODEL_SHORT"'","prompt":"Hello","max_tokens":'"$OUTPUT_LEN"',"temperature":0}' \
        > "$RUN_DIR/warmup_response.json" 2>&1 || true

    # 4) Send N measurement requests via streaming API; capture TTFT and E2E
    echo "    Sending $MEASUREMENTS measurement requests..."
    for m in $(seq 1 "$MEASUREMENTS"); do
        python3 - "$RUN_PORT" "$MODEL_SHORT" "$RUN_DIR/m_${m}.json" "$OUTPUT_LEN" <<'PYEOF'
import json
import sys
import time
import urllib.request

port, model, out_path, output_len = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])

url = f"http://127.0.0.1:{port}/v1/completions"
body = json.dumps({
    "model": model,
    "prompt": "Hello",
    "max_tokens": output_len,
    "temperature": 0,
    "stream": True,
}).encode()

req = urllib.request.Request(
    url, data=body,
    headers={"Content-Type": "application/json"},
    method="POST",
)

t_start = time.perf_counter()
ttft = None
chunks = 0
with urllib.request.urlopen(req, timeout=30) as resp:
    for line in resp:
        line = line.decode().strip()
        if not line.startswith("data: "):
            continue
        if line == "data: [DONE]":
            break
        if ttft is None:
            ttft = time.perf_counter() - t_start
        chunks += 1
e2e = time.perf_counter() - t_start

with open(out_path, "w") as f:
    json.dump({"ttft_s": ttft, "e2e_s": e2e, "chunks": chunks}, f)
PYEOF
    done

    # 5) Per-run aggregation
    python3 - "$RUN_DIR" "$MEASUREMENTS" <<'PYEOF'
import json
import statistics
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
n = int(sys.argv[2])

ttfts, e2es = [], []
for m in range(1, n + 1):
    p = run_dir / f"m_{m}.json"
    if not p.exists():
        continue
    try:
        d = json.loads(p.read_text())
        if d.get("ttft_s") is not None:
            ttfts.append(d["ttft_s"])
        if d.get("e2e_s") is not None:
            e2es.append(d["e2e_s"])
    except Exception:
        pass

summary = {
    "n_measurements": len(ttfts),
    "ttft_median_ms": statistics.median(ttfts) * 1000 if ttfts else None,
    "ttft_min_ms": min(ttfts) * 1000 if ttfts else None,
    "ttft_max_ms": max(ttfts) * 1000 if ttfts else None,
    "e2e_median_ms": statistics.median(e2es) * 1000 if e2es else None,
    "e2e_min_ms": min(e2es) * 1000 if e2es else None,
    "e2e_max_ms": max(e2es) * 1000 if e2es else None,
}
(run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
if summary["ttft_median_ms"] is not None:
    print(f"    TTFT median: {summary['ttft_median_ms']:.1f}ms"
          f"  E2E median: {summary['e2e_median_ms']:.1f}ms")
PYEOF

    cleanup_server $SERVER_PID
    sleep 5  # let GPU memory drain
done

# 6) Cross-run aggregation: median-of-medians
python3 - "$CONFIG_DIR" "$RUNS" <<'PYEOF'
import json
import statistics
import sys
from pathlib import Path

config_dir = Path(sys.argv[1])
n_runs = int(sys.argv[2])

run_summaries = []
for r in range(1, n_runs + 1):
    p = config_dir / f"run_{r}" / "summary.json"
    if not p.exists():
        continue
    try:
        run_summaries.append(json.loads(p.read_text()))
    except Exception:
        pass

if not run_summaries:
    print("    NO RUNS SUCCEEDED")
    sys.exit(0)

ttft_medians = [s["ttft_median_ms"] for s in run_summaries if s["ttft_median_ms"] is not None]
e2e_medians = [s["e2e_median_ms"] for s in run_summaries if s["e2e_median_ms"] is not None]

result = {
    "n_runs": len(run_summaries),
    "ttft_median_of_medians_ms": statistics.median(ttft_medians) if ttft_medians else None,
    "ttft_min_ms": min(s["ttft_min_ms"] for s in run_summaries
                      if s["ttft_min_ms"] is not None) if ttft_medians else None,
    "ttft_max_ms": max(s["ttft_max_ms"] for s in run_summaries
                      if s["ttft_max_ms"] is not None) if ttft_medians else None,
    "e2e_median_of_medians_ms": statistics.median(e2e_medians) if e2e_medians else None,
    "e2e_min_ms": min(s["e2e_min_ms"] for s in run_summaries
                     if s["e2e_min_ms"] is not None) if e2e_medians else None,
    "e2e_max_ms": max(s["e2e_max_ms"] for s in run_summaries
                     if s["e2e_max_ms"] is not None) if e2e_medians else None,
}
(config_dir / "aggregate.json").write_text(json.dumps(result, indent=2))
print(f"\n*** {config_dir.name} aggregate ***")
print(f"  TTFT (median of {result['n_runs']} runs): "
      f"{result['ttft_median_of_medians_ms']:.1f}ms  "
      f"(range: {result['ttft_min_ms']:.1f}-{result['ttft_max_ms']:.1f}ms)")
print(f"  E2E  (median of {result['n_runs']} runs): "
      f"{result['e2e_median_of_medians_ms']:.1f}ms  "
      f"(range: {result['e2e_min_ms']:.1f}-{result['e2e_max_ms']:.1f}ms)")
PYEOF

echo ""
echo "============================================"
echo "All $RUNS runs complete for $TAG"
echo "Results in: $CONFIG_DIR/"
echo "============================================"
