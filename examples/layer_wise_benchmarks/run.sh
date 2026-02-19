#!/bin/bash

set -euo pipefail

if [ -v OMPI_COMM_WORLD_SIZE ]; then
    export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
    export RANK=$OMPI_COMM_WORLD_RANK
    export LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
    export NODE_RANK=$OMPI_COMM_WORLD_NODE_RANK
fi

if [ "$RANK" -eq 0 ]; then
    export TLLM_LOG_LEVEL=INFO
fi

PROFILE_DIR=${PROFILE_DIR:-profiles}
mkdir -p -- "$PROFILE_DIR"

PROFILE=${PROFILE:-1}
BACKTRACE=${BACKTRACE:-0}
GPU_METRICS=${GPU_METRICS:-0}
if [ "$PROFILE" -eq 1 ]; then
    PROFILE_CMD=(
        nsys profile
        -t cuda,nvtx
        --cpuctxsw none --cuda-event-trace false
        --cuda-graph-trace node
        -c cudaProfilerApi --capture-range-end stop
        -o "${PROFILE_DIR}/report_np${WORLD_SIZE}_rank${RANK}.nsys-rep"
        --force-overwrite true
    )
    if [ "$BACKTRACE" -eq 1 ]; then
        PROFILE_CMD+=(--python-backtrace=cuda --cudabacktrace all)
    else
        PROFILE_CMD+=(-s none)
    fi
    if [ "$GPU_METRICS" -eq 1 ]; then
        PROFILE_CMD+=(
            --gpu-metrics-devices $LOCAL_RANK
            --gpu-metrics-frequency 10000
        )
    fi
else
    PROFILE_CMD=()
fi

SCRIPT_PATH=$(realpath --relative-to="$(pwd)" -- "$(dirname -- "$0")"/run.py)

set -x
${PROFILE_CMD[@]+"${PROFILE_CMD[@]}"} bash -o pipefail -c \
    'python3 -u "$1" "${@:3}" 2>&1 | tee "$2/report_np'"${WORLD_SIZE}"'_rank'"${RANK}"'.log"' \
    bash "$SCRIPT_PATH" "$PROFILE_DIR" "$@"
