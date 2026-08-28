#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Run a standalone online VisualGen benchmark against an existing server.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "${SCRIPT_DIR}/../../.." && pwd)"}

MODE=${MODE:-t2v}
MODEL=${MODEL:-nvidia/Cosmos3-Nano}
# Use ${VAR-default}, not ${VAR:-default}: an explicitly empty config is valid.
SERVER_CONFIG=${SERVER_CONFIG-"${PROJECT_ROOT}/examples/visual_gen/configs/cosmos3-nano-1gpu.yaml"}
SERVER_LOG_PATH=${SERVER_LOG_PATH-}
SERVER_PROCESS_PID=${SERVER_PROCESS_PID-}
BACKEND=${BACKEND-}
INPUT_REFERENCE=${INPUT_REFERENCE-}
ACTION_JSON=${ACTION_JSON-}
TRANSFER_CONTROLS=${TRANSFER_CONTROLS-}
EXTRA_PARAMS=${EXTRA_PARAMS-"{}"}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
PYTHON_BIN=${PYTHON_BIN:-python3}
SERVER_TIMEOUT=${SERVER_TIMEOUT:-3600}

# Generation parameters are optional. Omission preserves checkpoint defaults.
SIZE=${SIZE-}
SECONDS_TO_GENERATE=${SECONDS_TO_GENERATE-}
FPS=${FPS-}
NUM_FRAMES=${NUM_FRAMES-}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS-}
GUIDANCE_SCALE=${GUIDANCE_SCALE-}
SEED=${SEED-}
NEGATIVE_PROMPT=${NEGATIVE_PROMPT-}
OUTPUT_FORMAT=${OUTPUT_FORMAT-}

NUM_PROMPTS=${NUM_PROMPTS:-3}
REQUEST_RATE=${REQUEST_RATE:-inf}
BURSTINESS=${BURSTINESS:-1.0}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-1}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-21600}
METRIC_PERCENTILES=${METRIC_PERCENTILES:-50,90,99}
PROMPT=${PROMPT:-"A cinematic scene with natural motion and consistent subjects"}
NUM_GPUS_VALUE=${NUM_GPUS-}
SAVE_DETAILED=${SAVE_DETAILED:-true}
SAVE_MEDIA=${SAVE_MEDIA:-false}
DRY_RUN=${DRY_RUN:-false}

RUN_TIMESTAMP=$(date -u +%Y%m%d-%H%M%S)
RESULT_DIR=${RESULT_DIR:-"./benchmark_results/${RUN_TIMESTAMP}-${MODE}"}
BENCHMARK_LOG="${RESULT_DIR}/benchmark.log"
RESULT_JSON="${RESULT_DIR}/result.json"
METADATA_JSON="${RESULT_DIR}/metadata.json"
CLIENT_MEDIA_DIR=

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
    local elapsed=0
    local interval=5
    local status

    echo "Waiting for existing server at ${url} ..."
    while [ "$elapsed" -lt "$SERVER_TIMEOUT" ]; do
        if [ -n "$SERVER_PROCESS_PID" ] && ! kill -0 "$SERVER_PROCESS_PID" 2>/dev/null; then
            fail "Benchmark server part exited before becoming healthy; inspect ${SERVER_LOG_PATH:-its server log}"
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
    fail "Server at ${HOST}:${PORT} did not become ready within ${SERVER_TIMEOUT}s"
}

verify_server_model() {
    local url="http://${HOST}:${PORT}/v1/models"
    local response

    if ! response=$(curl -sS --fail "$url"); then
        fail "Could not read ${url}; cannot verify which model the existing server loaded"
    fi
    if ! "$PYTHON_BIN" -c '
import json
import sys

expected_model, raw_response = sys.argv[1:]
try:
    response = json.loads(raw_response)
except json.JSONDecodeError as exc:
    raise SystemExit(f"invalid JSON from /v1/models: {exc}") from exc
model_ids = [item.get("id") for item in response.get("data", []) if isinstance(item, dict)]
if expected_model not in model_ids:
    raise SystemExit(
        f"expected model {expected_model!r}, but /v1/models reported {model_ids!r}"
    )
' "$MODEL" "$response"
    then
        fail "Existing server at ${HOST}:${PORT} does not report MODEL=${MODEL}"
    fi
    echo "Verified existing server model: ${MODEL}"
}

validate_boolean SAVE_DETAILED "$SAVE_DETAILED"
validate_boolean SAVE_MEDIA "$SAVE_MEDIA"
validate_boolean DRY_RUN "$DRY_RUN"

case "$PORT" in
    ''|*[!0-9]*) fail "PORT must be an integer from 1 through 65535, got '${PORT}'" ;;
esac
if [ "$PORT" -lt 1 ] || [ "$PORT" -gt 65535 ]; then
    fail "PORT must be an integer from 1 through 65535, got '${PORT}'"
fi
case "$SERVER_TIMEOUT" in
    ''|*[!0-9]*) fail "SERVER_TIMEOUT must be a non-negative integer, got '${SERVER_TIMEOUT}'" ;;
esac

case "$MODE" in
    t2i)
        EXPECTED_BACKEND=openai-images
        ;;
    t2v|i2v|v2v|transfer|t2av|ti2av|policy|forward_dynamics|inverse_dynamics)
        EXPECTED_BACKEND=openai-videos
        ;;
    *)
        fail \
            "MODE must be one of t2i, t2v, i2v, v2v, transfer, t2av, ti2av, " \
            "policy, forward_dynamics, or inverse_dynamics; got '${MODE}'"
        ;;
esac

if [ -n "$BACKEND" ] && [ "$BACKEND" != "$EXPECTED_BACKEND" ]; then
    fail "MODE=${MODE} requires BACKEND=${EXPECTED_BACKEND}; got BACKEND=${BACKEND}"
fi
BACKEND=$EXPECTED_BACKEND

if [ -n "$INPUT_REFERENCE" ] && [ ! -f "$INPUT_REFERENCE" ]; then
    fail "INPUT_REFERENCE file does not exist: ${INPUT_REFERENCE}"
fi
if [ -n "$ACTION_JSON" ] && [ ! -f "$ACTION_JSON" ]; then
    fail "ACTION_JSON file does not exist: ${ACTION_JSON}"
fi
TRANSFER_CONTROLS_METADATA={}

case "$MODE" in
    i2v|v2v|ti2av|policy|forward_dynamics|inverse_dynamics)
        if [ -z "$INPUT_REFERENCE" ]; then
            fail "MODE=${MODE} requires INPUT_REFERENCE pointing to an image or video file"
        fi
        if [ "$MODE" = "forward_dynamics" ] && [ -z "$ACTION_JSON" ]; then
            fail "MODE=forward_dynamics requires ACTION_JSON containing a [T, D] trajectory"
        fi
        case "$MODE:$MODEL" in
            policy:*Cosmos3-Edge*|policy:*cosmos3-edge*|\
            forward_dynamics:*Cosmos3-Edge*|forward_dynamics:*cosmos3-edge*|\
            inverse_dynamics:*Cosmos3-Edge*|inverse_dynamics:*cosmos3-edge*)
                fail "Cosmos3-Edge action weights are not supported; use Cosmos3-Nano or Cosmos3-Super"
                ;;
        esac
        ;;
    transfer)
        if [ -z "$TRANSFER_CONTROLS" ]; then
            fail "MODE=transfer requires a non-empty TRANSFER_CONTROLS JSON object"
        fi
        TRANSFER_CONTROLS_METADATA=$(
            "$PYTHON_BIN" -c '
import json
import sys
from pathlib import Path

raw_controls, input_reference = sys.argv[1:]
try:
    controls = json.loads(raw_controls)
except json.JSONDecodeError as exc:
    raise SystemExit(f"ERROR: malformed TRANSFER_CONTROLS JSON: {exc}") from exc
if not isinstance(controls, dict) or not controls:
    raise SystemExit("ERROR: TRANSFER_CONTROLS must be a non-empty JSON object")

allowed_hints = {"edge", "blur", "depth", "seg", "wsm"}
auto_hints = {"edge", "blur"}
metadata = {}
for hint, control_reference in controls.items():
    if hint not in allowed_hints:
        raise SystemExit(
            "ERROR: TRANSFER_CONTROLS keys must be edge, blur, depth, seg, or wsm; "
            f"got {hint!r}"
        )
    if control_reference is True:
        if hint not in auto_hints:
            raise SystemExit(
                f"ERROR: TRANSFER_CONTROLS.{hint}=true is unsupported; only edge and "
                "blur can derive a control from INPUT_REFERENCE"
            )
        if not input_reference:
            raise SystemExit(
                f"ERROR: TRANSFER_CONTROLS.{hint}=true requires INPUT_REFERENCE"
            )
        metadata[hint] = "input_reference"
        continue
    if not isinstance(control_reference, str) or not control_reference:
        raise SystemExit(
            f"ERROR: TRANSFER_CONTROLS.{hint} must be true or a non-empty "
            "control-media path"
        )
    control_path = Path(control_reference).expanduser()
    if not control_path.is_file():
        raise SystemExit(
            f"ERROR: TRANSFER_CONTROLS.{hint} file does not exist: {control_reference}"
        )
    metadata[hint] = control_path.name

print(json.dumps(metadata, separators=(",", ":")))
' "$TRANSFER_CONTROLS" "$INPUT_REFERENCE"
        )
        case "$MODEL" in
            *Cosmos3-Edge*|*cosmos3-edge*)
                fail "Cosmos3-Edge does not support Transfer; use Cosmos3-Nano or Cosmos3-Super"
                ;;
        esac
        ;;
    *)
        if [ -n "$INPUT_REFERENCE" ]; then
            fail \
                "MODE=${MODE} does not accept INPUT_REFERENCE; use i2v, v2v, " \
                "transfer, ti2av, policy, forward_dynamics, or inverse_dynamics"
        fi
        ;;
esac

if [ "$MODE" != "transfer" ] && [ -n "$TRANSFER_CONTROLS" ]; then
    fail "TRANSFER_CONTROLS requires MODE=transfer"
fi
if [ "$MODE" != "forward_dynamics" ] && [ -n "$ACTION_JSON" ]; then
    fail "ACTION_JSON requires MODE=forward_dynamics"
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

mode, raw_extra_params, output_format = sys.argv[1:]
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
    allowed_formats = {"png", "webp", "jpeg", "safetensors", "pt"}
else:
    allowed_formats = {"mp4", "avi", "auto", "safetensors", "pt"}
if mode in {"t2av", "ti2av"}:
    required["enable_audio"] = True
    body["format"] = "mp4"
if mode in {"policy", "forward_dynamics", "inverse_dynamics"}:
    required["action_mode"] = mode
    allowed_formats = {"auto", "safetensors", "pt"}
    if not any(extra_params.get(key) is not None for key in ("domain_name", "domain_id")):
        raise SystemExit(
            f"ERROR: MODE={mode} requires EXTRA_PARAMS.domain_name or EXTRA_PARAMS.domain_id"
        )
if output_format:
    if output_format not in allowed_formats:
        raise SystemExit(
            f"ERROR: OUTPUT_FORMAT={output_format!r} is not valid for MODE={mode}"
        )
    if mode in {"t2av", "ti2av"} and output_format != "mp4":
        raise SystemExit(f"ERROR: MODE={mode} requires OUTPUT_FORMAT=mp4")
    body["format"] = output_format

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
' "$MODE" "$EXTRA_PARAMS" "$OUTPUT_FORMAT"
)

NUM_GPUS_EXPLICIT=false
if [ -n "$NUM_GPUS_VALUE" ]; then
    NUM_GPUS_EXPLICIT=true
elif [ -n "$SERVER_CONFIG" ]; then
    if [ ! -f "$SERVER_CONFIG" ]; then
        fail \
            "SERVER_CONFIG file does not exist: ${SERVER_CONFIG}; set NUM_GPUS " \
            "explicitly when the client cannot read the server config"
    fi
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
if [ "$SAVE_MEDIA" = "true" ]; then
    CLIENT_MEDIA_DIR="${RESULT_DIR}/media"
    BENCHMARK_CMD+=(--media-dir "$CLIENT_MEDIA_DIR")
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
if [ -n "$ACTION_JSON" ]; then
    BENCHMARK_CMD+=(--action-json "$ACTION_JSON")
fi
if [ -n "$TRANSFER_CONTROLS" ]; then
    BENCHMARK_CMD+=(--transfer-controls "$TRANSFER_CONTROLS")
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
ACTION_JSON_BASENAME=
if [ -n "$ACTION_JSON" ]; then
    ACTION_JSON_BASENAME=$(basename "$ACTION_JSON")
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
    "action_json=${ACTION_JSON_BASENAME}"
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
    action_json,
    transfer_controls,
    request_body,
    save_media,
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
    "action_json": action_json or None,
    "transfer_controls": json.loads(transfer_controls),
    "extra_params": request_body.get("extra_params", {}),
    "request_body": request_body,
    "save_media": save_media == "true",
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
    "$ACTION_JSON_BASENAME" \
    "$TRANSFER_CONTROLS_METADATA" \
    "$EXTRA_BODY" \
    "$SAVE_MEDIA"

echo "VisualGen Benchmark Client"
echo "Model:               ${MODEL}"
echo "Mode:                ${MODE}"
echo "Backend:             ${BACKEND}"
echo "Server address:      ${HOST}:${PORT}"
echo "Server config:       ${CONFIG_METADATA}"
echo "GPUs:                ${NUM_GPUS_VALUE}"
echo "Request rate:        ${REQUEST_RATE}"
echo "Max concurrency:     ${MAX_CONCURRENCY}"
echo "Input reference:     ${INPUT_REFERENCE_BASENAME:-none}"
echo "Action trajectory:   ${ACTION_JSON_BASENAME:-none}"
echo "Transfer controls:   ${TRANSFER_CONTROLS_METADATA}"
echo "Client media:        ${CLIENT_MEDIA_DIR:-disabled}"
echo "Server metrics log:  ${SERVER_LOG_PATH:-disabled}"
echo "Result directory:    ${RESULT_DIR}"
echo "Benchmark command:"
print_command "${BENCHMARK_CMD[@]}"

if [ "$DRY_RUN" = "true" ]; then
    echo "DRY_RUN=true; the client command was validated but not executed."
    exit 0
fi

wait_for_server
verify_server_model

SERVER_LOG_START_LINE=0
if [ -n "$SERVER_LOG_PATH" ]; then
    if [ ! -f "$SERVER_LOG_PATH" ]; then
        fail "SERVER_LOG_PATH does not exist: ${SERVER_LOG_PATH}"
    fi
    SERVER_LOG_START_LINE=$(wc -l <"$SERVER_LOG_PATH" | tr -d ' ')
fi

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

"$PYTHON_BIN" - "$RESULT_JSON" "$SERVER_LOG_PATH" "$SERVER_LOG_START_LINE" \
    "$NUM_PROMPTS" "$NUM_GPUS_VALUE" "$SAVE_DETAILED" "$SAVE_MEDIA" "$CLIENT_MEDIA_DIR" \
    2>&1 <<'PY' | tee -a "$BENCHMARK_LOG"
import json
import re
import sys
from pathlib import Path

from tensorrt_llm.serve.scripts.benchmark_visual_gen import (
    _summarize_denoising_step_times,
    _summarize_total_pipeline_times,
)

result_path = Path(sys.argv[1])
server_log_value = sys.argv[2]
start_line = int(sys.argv[3])
expected_requests = int(sys.argv[4])
expected_num_gpus = int(sys.argv[5])
save_detailed = sys.argv[6] == "true"
save_media = sys.argv[7] == "true"
client_media_dir = Path(sys.argv[8]) if save_media else None

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
if save_media:
    media_files = result.get("media_files")
    if not isinstance(media_files, list) or len(media_files) != expected_requests:
        saved_count = len(media_files) if isinstance(media_files, list) else "unavailable"
        raise SystemExit(
            "ERROR: benchmark did not retain one client media file per measured request: "
            f"saved={saved_count}, expected={expected_requests}"
        )
    missing_media = [
        media_file
        for media_file in media_files
        if not (client_media_dir / media_file).is_file()
    ]
    if missing_media:
        raise SystemExit(f"ERROR: retained client media files are missing: {missing_media}")

run_lines = []
if server_log_value:
    server_log_path = Path(server_log_value)
    run_lines = server_log_path.read_text(encoding="utf-8", errors="replace").splitlines()[
        start_line:
    ]

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

total_pipeline_summary = _summarize_total_pipeline_times(run_lines, expected_requests)
result.update(total_pipeline_summary)
if not save_detailed:
    result.pop("total_pipeline_times", None)

if resolved_steps:
    measured_steps = resolved_steps[-expected_requests:]
    if len(measured_steps) == expected_requests and len(set(measured_steps)) == 1:
        step_count = measured_steps[0]
        result["resolved_num_inference_steps"] = step_count
        result["mean_seconds_per_denoising_step"] = result["mean_denoise"] / step_count
        step_timing_summary = _summarize_denoising_step_times(
            run_lines, expected_requests, step_count
        )
        result.update(step_timing_summary)
        if not save_detailed:
            result.pop("denoising_step_times", None)

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
if "mean_denoising_step_time" in result:
    step_percentiles = result["percentiles_denoising_step_time"]
    print(
        "  Denoising Step Time Mean (s/it): "
        f"{result['mean_denoising_step_time']:.4f}"
    )
    print(f"  Denoising Step Time P95 (s/it): {step_percentiles['p95']:.4f}")
    print(f"  Denoising Step Time P99 (s/it): {step_percentiles['p99']:.4f}")
else:
    print("  Denoising Step Time P95/P99: unavailable; provide SERVER_LOG_PATH")
if "mean_vision_decode" in result:
    print(f"  Avg. Vision Decode Time (s): {result['mean_vision_decode']:.4f}")
else:
    print("  Avg. Vision Decode Time (s): unavailable; provide SERVER_LOG_PATH")
print(f"  Request Latency (s): {result['mean_latency']:.4f}")
if save_media:
    print(f"  Retained media files: {len(result['media_files'])} in {client_media_dir}")
if "mean_total_pipeline_time" in result:
    print(
        "  Total Pipeline Time (s/request), mean: "
        f"{result['mean_total_pipeline_time']:.4f} (server log)"
    )
else:
    print(
        "  Total Pipeline Time (s/request), mean: "
        "unavailable; provide SERVER_LOG_PATH"
    )
PY

echo "Benchmark complete. Artifacts:"
echo "  Benchmark log: ${BENCHMARK_LOG}"
echo "  Result JSON:   ${RESULT_JSON}"
echo "  Metadata JSON: ${METADATA_JSON}"
if [ -n "$SERVER_LOG_PATH" ]; then
    echo "  Server log:    ${SERVER_LOG_PATH}"
fi
if [ "$SAVE_MEDIA" = "true" ]; then
    echo "  Client media:  ${CLIENT_MEDIA_DIR}"
fi
