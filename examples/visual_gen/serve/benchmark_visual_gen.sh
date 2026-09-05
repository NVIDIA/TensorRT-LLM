#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Run the standalone VisualGen benchmark server and client as one lifecycle.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "${SCRIPT_DIR}/../../.." && pwd)"}

MODE=${MODE:-t2v}
MODEL=${MODEL:-nvidia/Cosmos3-Nano}
# Use ${VAR-default}, not ${VAR:-default}: an explicitly empty config is valid.
SERVER_CONFIG=${SERVER_CONFIG-"${PROJECT_ROOT}/examples/visual_gen/configs/cosmos3-nano-1gpu.yaml"}
INPUT_REFERENCE_TYPE=${INPUT_REFERENCE_TYPE-}
ACTION_JSON=${ACTION_JSON-}
TRANSFER_CONTROLS=${TRANSFER_CONTROLS-}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
PYTHON_BIN=${PYTHON_BIN:-python3}
DRY_RUN=${DRY_RUN:-false}

RUN_TIMESTAMP=$(date -u +%Y%m%d-%H%M%S)
RESULT_DIR=${RESULT_DIR:-"./benchmark_results/${RUN_TIMESTAMP}-${MODE}"}
SERVER_LOG_PATH=${SERVER_LOG_PATH:-"${RESULT_DIR}/server.log"}
SERVER_SCRIPT=${SERVER_SCRIPT:-"${SCRIPT_DIR}/benchmark_visual_gen_server.sh"}
CLIENT_SCRIPT=${CLIENT_SCRIPT:-"${SCRIPT_DIR}/benchmark_visual_gen_client.sh"}
SERVER_PART_PID=

fail() {
    echo "ERROR: $*" >&2
    exit 2
}

cleanup() {
    if [ -n "$SERVER_PART_PID" ]; then
        echo "Stopping benchmark server part (PID: ${SERVER_PART_PID})..."
        kill "$SERVER_PART_PID" 2>/dev/null || true
        wait "$SERVER_PART_PID" 2>/dev/null || true
        SERVER_PART_PID=
    fi
}

on_signal() {
    exit 130
}

case "$DRY_RUN" in
    true|false) ;;
    *) fail "DRY_RUN must be true or false, got '${DRY_RUN}'" ;;
esac
if [ ! -x "$SERVER_SCRIPT" ]; then
    fail "Server benchmark part is not executable: ${SERVER_SCRIPT}"
fi
if [ ! -x "$CLIENT_SCRIPT" ]; then
    fail "Client benchmark part is not executable: ${CLIENT_SCRIPT}"
fi

export MODE MODEL SERVER_CONFIG HOST PORT PYTHON_BIN DRY_RUN RESULT_DIR SERVER_LOG_PATH
export INPUT_REFERENCE_TYPE ACTION_JSON TRANSFER_CONTROLS

echo "VisualGen Serving Benchmark (server + client)"
echo "Result directory: ${RESULT_DIR}"

if [ "$DRY_RUN" = "true" ]; then
    "$SERVER_SCRIPT"
    "$CLIENT_SCRIPT"
    echo "DRY_RUN=true; both standalone parts were validated but not executed."
    exit 0
fi

trap cleanup EXIT
trap on_signal INT TERM

"$SERVER_SCRIPT" &
SERVER_PART_PID=$!

set +e
SERVER_PROCESS_PID="$SERVER_PART_PID" "$CLIENT_SCRIPT"
CLIENT_STATUS=$?
set -e
if [ "$CLIENT_STATUS" -ne 0 ]; then
    fail "Benchmark client part exited with status ${CLIENT_STATUS}"
fi

cleanup
trap - EXIT
echo "Combined benchmark complete."
