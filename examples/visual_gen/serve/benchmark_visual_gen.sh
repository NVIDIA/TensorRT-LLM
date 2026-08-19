#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Benchmark Cosmos3 VisualGen serving on an already-allocated single node.
#
# The script starts trtllm-serve, waits for /health, runs the online benchmark,
# validates the result JSON, and retains the server/client logs and run metadata.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "${SCRIPT_DIR}/../../.." && pwd)"}

MODE=${MODE:-t2v}
MODEL=${MODEL:-nvidia/Cosmos3-Nano}
# Use ${VAR-default}, not ${VAR:-default}: an explicitly empty config is valid.
SERVER_CONFIG=${SERVER_CONFIG-"${PROJECT_ROOT}/examples/visual_gen/configs/cosmos3-nano-1gpu.yaml"}
BACKEND=${BACKEND-}
INPUT_REFERENCE=${INPUT_REFERENCE-}
EXTRA_PARAMS=${EXTRA_PARAMS-"{}"}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
PYTHON_BIN=${PYTHON_BIN:-python3}

# Generation parameters are optional. Omission preserves checkpoint defaults.
SIZE=${SIZE-}
SECONDS_TO_GENERATE=${SECONDS_TO_GENERATE-}
FPS=${FPS-}
NUM_FRAMES=${NUM_FRAMES-}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS-}
GUIDANCE_SCALE=${GUIDANCE_SCALE-}
SEED=${SEED-}
NEGATIVE_PROMPT=${NEGATIVE_PROMPT-}

NUM_PROMPTS=${NUM_PROMPTS:-3}
REQUEST_RATE=${REQUEST_RATE:-inf}
BURSTINESS=${BURSTINESS:-1.0}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-1}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-21600}
METRIC_PERCENTILES=${METRIC_PERCENTILES:-50,90,99}
PROMPT=${PROMPT:-"A cinematic scene with natural motion and consistent subjects"}
NUM_GPUS_VALUE=${NUM_GPUS-}
SAVE_DETAILED=${SAVE_DETAILED:-true}
DRY_RUN=${DRY_RUN:-false}

RUN_TIMESTAMP=$(date -u +%Y%m%d-%H%M%S)
RESULT_DIR=${RESULT_DIR:-"./benchmark_results/${RUN_TIMESTAMP}-${MODE}"}
SERVER_LOG="${RESULT_DIR}/server.log"
BENCHMARK_LOG="${RESULT_DIR}/benchmark.log"
RESULT_JSON="${RESULT_DIR}/result.json"
METADATA_JSON="${RESULT_DIR}/metadata.json"

fail() {
    echo "ERROR: $*" >&2
    exit 2
}

validate_boolean() {
    local name=$1
    local value=$2
    case "$value" in
        true|false) ;;
        *) fail "${name} must be true or false, got '${value}'" ;;
    esac
}

print_command() {
    local arg
    printf '  '
    for arg in "$@"; do
        printf '%q ' "$arg"
    done
    printf '\n'
}

wait_for_server() {
    local url="http://${HOST}:${PORT}/health"
    local max_wait=${SERVER_TIMEOUT:-3600}
    local elapsed=0
    local interval=5
    local status

    echo "Waiting for server at ${url} ..."
    while [ "$elapsed" -lt "$max_wait" ]; do
        if [ -n "${SERVER_PID:-}" ] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
            fail "Server exited before becoming healthy; inspect ${SERVER_LOG}"
        fi
        status=$(curl -s -o /dev/null -w '%{http_code}' "$url" 2>/dev/null || true)
        if [ "$status" = "200" ]; then
            echo "Server is ready (took ${elapsed}s)"
            return 0
        fi
        sleep "$interval"
        elapsed=$((elapsed + interval))
        if [ $((elapsed % 30)) -eq 0 ]; then
            echo "  Still waiting... (${elapsed}s elapsed)"
        fi
    done
    fail "Server did not become ready within ${max_wait}s; inspect ${SERVER_LOG}"
}

cleanup() {
    if [ -n "${SERVER_PID:-}" ]; then
        echo "Stopping server (PID: ${SERVER_PID})..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}

validate_boolean SAVE_DETAILED "$SAVE_DETAILED"
validate_boolean DRY_RUN "$DRY_RUN"

case "$MODE" in
    t2i)
        EXPECTED_BACKEND=openai-images
        ;;
    t2v|i2v|v2v|t2av|ti2av)
        EXPECTED_BACKEND=openai-videos
        ;;
    *)
        fail "MODE must be one of t2i, t2v, i2v, v2v, t2av, or ti2av; got '${MODE}'"
        ;;
esac

if [ -n "$BACKEND" ] && [ "$BACKEND" != "$EXPECTED_BACKEND" ]; then
    fail "MODE=${MODE} requires BACKEND=${EXPECTED_BACKEND}; got BACKEND=${BACKEND}"
fi
BACKEND=$EXPECTED_BACKEND

case "$MODE" in
    i2v|v2v|ti2av)
        if [ -z "$INPUT_REFERENCE" ]; then
            fail "MODE=${MODE} requires INPUT_REFERENCE pointing to an image or video file"
        fi
        if [ ! -f "$INPUT_REFERENCE" ]; then
            fail "INPUT_REFERENCE file does not exist: ${INPUT_REFERENCE}"
        fi
        ;;
    *)
        if [ -n "$INPUT_REFERENCE" ]; then
            fail "MODE=${MODE} does not accept INPUT_REFERENCE; use i2v, v2v, or ti2av"
        fi
        ;;
esac

if [ -n "$SERVER_CONFIG" ] && [ ! -f "$SERVER_CONFIG" ]; then
    fail "SERVER_CONFIG file does not exist: ${SERVER_CONFIG}"
fi

case "$MODE" in
    t2av|ti2av)
        case "$MODEL" in
            *Cosmos3-Edge*|*cosmos3-edge*)
                fail "Cosmos3-Edge has no audio tower; use Cosmos3-Nano or Cosmos3-Super"
                ;;
        esac
        if [ "$DRY_RUN" = "false" ]; then
            command -v ffmpeg >/dev/null 2>&1 || fail \
                "Audio modes require FFmpeg for MP4 audio muxing; the AVI fallback drops audio"
            command -v ffprobe >/dev/null 2>&1 || fail \
                "Audio modes require ffprobe so the result can be verified"
        fi
        ;;
esac

EXTRA_BODY=$(
    "$PYTHON_BIN" -c '
import json
import sys

mode, raw_extra_params = sys.argv[1:]
try:
    extra_params = json.loads(raw_extra_params or "{}")
except json.JSONDecodeError as exc:
    raise SystemExit(f"ERROR: malformed EXTRA_PARAMS JSON: {exc}") from exc
if not isinstance(extra_params, dict):
    raise SystemExit("ERROR: EXTRA_PARAMS must be a JSON object")

required = {}
body = {}
if mode == "t2i":
    required["output_type"] = "image"
if mode in {"t2av", "ti2av"}:
    required["enable_audio"] = True
    body["format"] = "mp4"

for key, expected in required.items():
    if key in extra_params and extra_params[key] != expected:
        raise SystemExit(
            f"ERROR: MODE={mode} requires EXTRA_PARAMS.{key}={expected!r}; "
            f"got {extra_params[key]!r}"
        )
    extra_params[key] = expected

if extra_params:
    body["extra_params"] = extra_params
print(json.dumps(body, separators=(",", ":")))
' "$MODE" "$EXTRA_PARAMS"
)

NUM_GPUS_EXPLICIT=false
if [ -n "$NUM_GPUS_VALUE" ]; then
    NUM_GPUS_EXPLICIT=true
elif [ -n "$SERVER_CONFIG" ]; then
    NUM_GPUS_OUTPUT=$(
        "$PYTHON_BIN" -c '
import sys
import yaml
from tensorrt_llm.commands.utils import get_visual_gen_num_gpus

with open(sys.argv[1], encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file) or {}
print(get_visual_gen_num_gpus(config))
' "$SERVER_CONFIG"
    )
    # TensorRT-LLM may print an import-time version banner to stdout. The
    # resolver's integer is deliberately the final line.
    NUM_GPUS_VALUE=${NUM_GPUS_OUTPUT##*$'\n'}
else
    NUM_GPUS_VALUE=1
fi

case "$NUM_GPUS_VALUE" in
    ''|*[!0-9]*) fail "NUM_GPUS must be a positive integer, got '${NUM_GPUS_VALUE}'" ;;
esac
if [ "$NUM_GPUS_VALUE" -lt 1 ]; then
    fail "NUM_GPUS must be at least 1, got '${NUM_GPUS_VALUE}'"
fi

SERVER_CMD=(trtllm-serve "$MODEL" --host "$HOST" --port "$PORT")
if [ -n "$SERVER_CONFIG" ]; then
    SERVER_CMD+=(--visual_gen_args "$SERVER_CONFIG")
fi

BENCHMARK_CMD=(
    "$PYTHON_BIN" -m tensorrt_llm.serve.scripts.benchmark_visual_gen
    --model "$MODEL"
    --backend "$BACKEND"
    --host "$HOST"
    --port "$PORT"
    --prompt "$PROMPT"
    --num-prompts "$NUM_PROMPTS"
    --request-rate "$REQUEST_RATE"
    --burstiness "$BURSTINESS"
    --max-concurrency "$MAX_CONCURRENCY"
    --request-timeout "$REQUEST_TIMEOUT"
    --metric-percentiles "$METRIC_PERCENTILES"
    --save-result
    --result-dir "$RESULT_DIR"
    --result-filename "$(basename "$RESULT_JSON")"
)

if [ "$SAVE_DETAILED" = "true" ]; then
    BENCHMARK_CMD+=(--save-detailed)
fi
if [ -n "$SIZE" ]; then
    BENCHMARK_CMD+=(--size "$SIZE")
fi
if [ -n "$SECONDS_TO_GENERATE" ]; then
    BENCHMARK_CMD+=(--seconds "$SECONDS_TO_GENERATE")
fi
if [ -n "$FPS" ]; then
    BENCHMARK_CMD+=(--fps "$FPS")
fi
if [ -n "$NUM_FRAMES" ]; then
    BENCHMARK_CMD+=(--num-frames "$NUM_FRAMES")
fi
if [ -n "$NUM_INFERENCE_STEPS" ]; then
    BENCHMARK_CMD+=(--num-inference-steps "$NUM_INFERENCE_STEPS")
fi
if [ -n "$GUIDANCE_SCALE" ]; then
    BENCHMARK_CMD+=(--guidance-scale "$GUIDANCE_SCALE")
fi
if [ -n "$SEED" ]; then
    BENCHMARK_CMD+=(--seed "$SEED")
fi
if [ -n "$NEGATIVE_PROMPT" ]; then
    BENCHMARK_CMD+=(--negative-prompt "$NEGATIVE_PROMPT")
fi
if [ "$EXTRA_BODY" != "{}" ]; then
    BENCHMARK_CMD+=(--extra-body "$EXTRA_BODY")
fi
if [ -n "$INPUT_REFERENCE" ]; then
    BENCHMARK_CMD+=(--input-reference "$INPUT_REFERENCE")
fi
case "$MODE" in
    t2av|ti2av) BENCHMARK_CMD+=(--require-audio) ;;
esac
if [ "$NUM_GPUS_EXPLICIT" = "true" ] || [ -z "$SERVER_CONFIG" ]; then
    BENCHMARK_CMD+=(--num-gpus "$NUM_GPUS_VALUE")
else
    BENCHMARK_CMD+=(--visual-gen-args "$SERVER_CONFIG")
fi

INPUT_REFERENCE_BASENAME=
if [ -n "$INPUT_REFERENCE" ]; then
    INPUT_REFERENCE_BASENAME=$(basename "$INPUT_REFERENCE")
fi
CONFIG_METADATA=$SERVER_CONFIG
if [ -z "$CONFIG_METADATA" ]; then
    CONFIG_METADATA=checkpoint-defaults
fi
BENCHMARK_CMD+=(
    --metadata
    "mode=${MODE}"
    "server_config=${CONFIG_METADATA}"
    "input_reference=${INPUT_REFERENCE_BASENAME}"
)

mkdir -p "$RESULT_DIR"
"$PYTHON_BIN" -c '
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    output_path,
    mode,
    model,
    server_config,
    num_gpus,
    request_rate,
    max_concurrency,
    input_reference,
    request_body,
) = sys.argv[1:]
request_body = json.loads(request_body)
metadata = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "mode": mode,
    "model": model,
    "server_config": server_config,
    "num_gpus": int(num_gpus),
    "request_rate": request_rate,
    "max_concurrency": int(max_concurrency),
    "input_reference": input_reference or None,
    "extra_params": request_body.get("extra_params", {}),
    "request_body": request_body,
}
Path(output_path).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
' \
    "$METADATA_JSON" \
    "$MODE" \
    "$MODEL" \
    "$CONFIG_METADATA" \
    "$NUM_GPUS_VALUE" \
    "$REQUEST_RATE" \
    "$MAX_CONCURRENCY" \
    "$INPUT_REFERENCE_BASENAME" \
    "$EXTRA_BODY"

echo "VisualGen Serving Benchmark"
echo "Model:               ${MODEL}"
echo "Mode:                ${MODE}"
echo "Backend:             ${BACKEND}"
echo "Server config:       ${CONFIG_METADATA}"
echo "GPUs:                ${NUM_GPUS_VALUE}"
echo "Request rate:        ${REQUEST_RATE}"
echo "Max concurrency:     ${MAX_CONCURRENCY}"
echo "Input reference:     ${INPUT_REFERENCE_BASENAME:-none}"
echo "Result directory:    ${RESULT_DIR}"
echo "Server command:"
print_command "${SERVER_CMD[@]}"
echo "Benchmark command:"
print_command "${BENCHMARK_CMD[@]}"

if [ "$DRY_RUN" = "true" ]; then
    echo "DRY_RUN=true; commands were validated but not executed."
    exit 0
fi

echo "Starting server; log: ${SERVER_LOG}"
"${SERVER_CMD[@]}" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap cleanup EXIT

wait_for_server
SERVER_LOG_START_LINE=$(wc -l <"$SERVER_LOG" | tr -d ' ')

echo "Running benchmark; log: ${BENCHMARK_LOG}"
set +e
"${BENCHMARK_CMD[@]}" 2>&1 | tee "$BENCHMARK_LOG"
BENCHMARK_STATUS=${PIPESTATUS[0]}
set -e
if [ "$BENCHMARK_STATUS" -ne 0 ]; then
    fail "Benchmark client exited with status ${BENCHMARK_STATUS}; inspect ${BENCHMARK_LOG}"
fi
if [ ! -f "$RESULT_JSON" ]; then
    fail "Benchmark completed without writing ${RESULT_JSON}"
fi

"$PYTHON_BIN" - "$RESULT_JSON" "$SERVER_LOG" "$SERVER_LOG_START_LINE" \
    "$NUM_PROMPTS" "$NUM_GPUS_VALUE" "$SAVE_DETAILED" <<'PY'
import json
import re
import sys
from pathlib import Path

result_path = Path(sys.argv[1])
server_log_path = Path(sys.argv[2])
start_line = int(sys.argv[3])
expected_requests = int(sys.argv[4])
expected_num_gpus = int(sys.argv[5])
save_detailed = sys.argv[6] == "true"

result = json.loads(result_path.read_text(encoding="utf-8"))
completed = int(result.get("completed", -1))
total_requests = int(result.get("total_requests", -1))
reported_num_gpus = int(result.get("num_gpus", -1))
if total_requests != expected_requests or completed != total_requests:
    raise SystemExit(
        "ERROR: benchmark result is incomplete: "
        f"completed={completed}, total={total_requests}, expected={expected_requests}"
    )
if reported_num_gpus != expected_num_gpus:
    raise SystemExit(
        "ERROR: benchmark result GPU count mismatch: "
        f"reported={reported_num_gpus}, expected={expected_num_gpus}"
    )
if result.get("mode") in {"t2av", "ti2av"} and result.get("audio_validated") is not True:
    raise SystemExit("ERROR: audio benchmark result was not validated with ffprobe")

run_lines = server_log_path.read_text(encoding="utf-8", errors="replace").splitlines()[start_line:]
decode_pattern = re.compile(r"(?:Video|Image) decoded in ([0-9]+(?:\.[0-9]+)?)s")
step_pattern = re.compile(r"num_inference_steps=([0-9]+)")
decode_values = [float(match.group(1)) for line in run_lines if (match := decode_pattern.search(line))]
resolved_steps = [int(match.group(1)) for line in run_lines if (match := step_pattern.search(line))]

# The first post-startup generation is the client's validation request. Keep
# only the final N entries, which correspond to the N measured requests.
if len(decode_values) >= expected_requests + 1:
    measured_decode_values = decode_values[-expected_requests:]
    result["mean_vision_decode"] = sum(measured_decode_values) / len(measured_decode_values)
    if save_detailed:
        result["vision_decodes"] = measured_decode_values

if resolved_steps:
    measured_steps = resolved_steps[-expected_requests:]
    if len(measured_steps) == expected_requests and len(set(measured_steps)) == 1:
        step_count = measured_steps[0]
        result["resolved_num_inference_steps"] = step_count
        result["mean_seconds_per_denoising_step"] = result["mean_denoise"] / step_count
        if save_detailed and "denoises" in result:
            result["seconds_per_denoising_step"] = [
                denoise / step_count for denoise in result["denoises"]
            ]

temp_path = result_path.with_suffix(result_path.suffix + ".tmp")
temp_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
temp_path.replace(result_path)

print("Verified benchmark result:")
print(f"  completed requests: {completed}/{total_requests}")
print(f"  reported GPUs: {reported_num_gpus}")
print(f"  Avg. Diffusion Time (s): {result['mean_denoise']:.4f}")
print(f"  Avg. Generation Time (s): {result['mean_generation']:.4f}")
if "mean_seconds_per_denoising_step" in result:
    print(
        "  Avg. Seconds per Denoising Step (s/it): "
        f"{result['mean_seconds_per_denoising_step']:.4f}"
    )
if "mean_vision_decode" in result:
    print(f"  Avg. Vision Decode Time (s): {result['mean_vision_decode']:.4f}")
else:
    print("  Avg. Vision Decode Time (s): unavailable; inspect the server log")
print(f"  Request Latency (s): {result['mean_latency']:.4f}")
PY

echo "Benchmark complete. Artifacts:"
echo "  Server log:    ${SERVER_LOG}"
echo "  Benchmark log: ${BENCHMARK_LOG}"
echo "  Result JSON:   ${RESULT_JSON}"
echo "  Metadata JSON: ${METADATA_JSON}"
echo "Total pipeline time has no benchmark-schema field and remains in the server log."
