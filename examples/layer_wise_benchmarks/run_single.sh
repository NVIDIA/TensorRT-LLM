#!/bin/bash

set -euo pipefail

if [ -v OMPI_COMM_WORLD_SIZE ]; then
    export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
    export RANK=$OMPI_COMM_WORLD_RANK
    export LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
    export NODE_RANK=$OMPI_COMM_WORLD_NODE_RANK
fi

export TRTLLM_FORCE_ALLTOALL_METHOD=MNNVL

PROFILE=${PROFILE:-1}
if [ "$PROFILE" -eq 1 ]; then
    PROFILE_FOLDER=profiles/run_single
    mkdir -p ${PROFILE_FOLDER}
    PROFILE_CMD="nsys profile --wait primary \
        -t cuda,nvtx -s none \
        --cpuctxsw none --cuda-event-trace false \
        --cuda-graph-trace node \
        -c cudaProfilerApi --capture-range-end stop \
        -o ${PROFILE_FOLDER}/run_single_ep${WORLD_SIZE}_rank${RANK}.nsys-rep \
        --force-overwrite true"
else
    PROFILE_CMD=
fi

set -x
$PROFILE_CMD python3 -u run_single.py
