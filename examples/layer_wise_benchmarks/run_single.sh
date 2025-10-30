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

PROFILE=${PROFILE:-1}
GPU_METRICS=${GPU_METRICS:-0}
if [ "$PROFILE" -eq 1 ]; then
    PROFILE_FOLDER=profiles/run_single
    mkdir -p ${PROFILE_FOLDER}
    PROFILE_CMD="nsys profile
        -t cuda,nvtx -s none
        --cpuctxsw none --cuda-event-trace false
        --cuda-graph-trace node
        -c cudaProfilerApi --capture-range-end stop
        -o ${PROFILE_FOLDER}/run_single_ep${WORLD_SIZE}_rank${RANK}.nsys-rep
        --force-overwrite true"
    if [ "$GPU_METRICS" -eq 1 ]; then
        PROFILE_CMD+=" --gpu-metrics-devices $LOCAL_RANK
            --gpu-metrics-frequency 10000"
    fi
else
    PROFILE_CMD=
fi

set -x
$PROFILE_CMD python3 -u run_single.py "$@"
