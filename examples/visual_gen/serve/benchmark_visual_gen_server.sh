#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Start a standalone TensorRT-LLM VisualGen server for online benchmarks.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "${SCRIPT_DIR}/../../.." && pwd)"}

MODEL=${MODEL:-nvidia/Cosmos3-Nano}
# Use ${VAR-default}, not ${VAR:-default}: an explicitly empty config is valid.
SERVER_CONFIG=${SERVER_CONFIG-"${PROJECT_ROOT}/examples/visual_gen/configs/cosmos3-nano-1gpu.yaml"}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
PYTHON_BIN=${PYTHON_BIN:-python3}
SERVER_TIMEOUT=${SERVER_TIMEOUT:-3600}
DRY_RUN=${DRY_RUN:-false}

RUN_TIMESTAMP=$(date -u +%Y%m%d-%H%M%S)
RESULT_DIR=${RESULT_DIR:-"./benchmark_results/${RUN_TIMESTAMP}-server"}
SERVER_LOG_PATH=${SERVER_LOG_PATH:-"${RESULT_DIR}/server.log"}
SERVER_MEDIA_DIR=
SERVER_PID=

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

ensure_server_port_available() {
    if ! "$PYTHON_BIN" - "$HOST" "$PORT" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
addr_info = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
address_family = (
    socket.AF_INET6
    if all(info[0] == socket.AF_INET6 for info in addr_info)
    else socket.AF_INET
)

with socket.socket(address_family, socket.SOCK_STREAM) as server_socket:
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server_socket.bind((host, port))
    except OSError:
        raise SystemExit(1)
PY
    then
        fail \
            "Cannot start VisualGen server on ${HOST}:${PORT}; port is already in use. " \
            "Stop the existing server or set PORT to a free port."
    fi
}

wait_for_server() {
    local health_host=$HOST
    local url
    local elapsed=0
    local interval=5
    local status

    case "$health_host" in
        0.0.0.0) health_host=127.0.0.1 ;;
        ::) health_host='[::1]' ;;
    esac
    url="http://${health_host}:${PORT}/health"

    echo "Waiting for server at ${url} ..."
    while [ "$elapsed" -lt "$SERVER_TIMEOUT" ]; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            fail "Server exited before becoming healthy; inspect ${SERVER_LOG_PATH}"
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
    fail "Server did not become ready within ${SERVER_TIMEOUT}s; inspect ${SERVER_LOG_PATH}"
}

cleanup() {
    if [ -n "$SERVER_PID" ]; then
        echo "Stopping server (PID: ${SERVER_PID})..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=
    fi
    if [ -n "$SERVER_MEDIA_DIR" ] && [ -d "$SERVER_MEDIA_DIR" ]; then
        case "${SERVER_MEDIA_DIR##*/}" in
            .server-media.*)
                if ! rm -rf -- "$SERVER_MEDIA_DIR"; then
                    echo "WARNING: Failed to remove temporary server media: ${SERVER_MEDIA_DIR}" >&2
                fi
                ;;
            *)
                echo "WARNING: Refusing to remove unexpected server media path: ${SERVER_MEDIA_DIR}" >&2
                ;;
        esac
        SERVER_MEDIA_DIR=
    fi
}

on_signal() {
    exit 130
}

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

if [ -n "$SERVER_CONFIG" ] && [ ! -f "$SERVER_CONFIG" ]; then
    fail "SERVER_CONFIG file does not exist: ${SERVER_CONFIG}"
fi

SERVER_CMD=(trtllm-serve "$MODEL" --host "$HOST" --port "$PORT")
if [ -n "$SERVER_CONFIG" ]; then
    SERVER_CMD+=(--visual_gen_args "$SERVER_CONFIG")
fi

echo "VisualGen Benchmark Server"
echo "Model:             ${MODEL}"
echo "Server config:     ${SERVER_CONFIG:-checkpoint-defaults}"
echo "Listen address:    ${HOST}:${PORT}"
echo "Server log:        ${SERVER_LOG_PATH}"
echo "Server command:"
print_command "${SERVER_CMD[@]}"

if [ "$DRY_RUN" = "true" ]; then
    echo "DRY_RUN=true; the server command was validated but not executed."
    exit 0
fi

mkdir -p "$RESULT_DIR" "$(dirname "$SERVER_LOG_PATH")"
ensure_server_port_available
SERVER_MEDIA_DIR=$(mktemp -d "${RESULT_DIR}/.server-media.XXXXXX")
trap cleanup EXIT
trap on_signal INT TERM

echo "Starting server; log: ${SERVER_LOG_PATH}"
TRTLLM_MEDIA_STORAGE_PATH="$SERVER_MEDIA_DIR" \
    "${SERVER_CMD[@]}" >"$SERVER_LOG_PATH" 2>&1 &
SERVER_PID=$!

wait_for_server
echo "Server will run until this script receives SIGINT or SIGTERM."

set +e
wait "$SERVER_PID"
SERVER_STATUS=$?
set -e
SERVER_PID=
if [ "$SERVER_STATUS" -ne 0 ]; then
    echo "ERROR: VisualGen server exited with status ${SERVER_STATUS}; inspect ${SERVER_LOG_PATH}" >&2
fi
exit "$SERVER_STATUS"
