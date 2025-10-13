#!/bin/bash

set -euo pipefail

CONTAINER_MOUNTS=$(realpath "$(pwd)/../.."):$(realpath "$(pwd)/../..")

if [ "${SLURM_JOB_ID:-}" == "" ]; then
    echo "Please set SLURM_JOB_ID"
    exit 1
fi

set -x
srun --mpi=pmix \
    -N "$NODES" \
    --ntasks-per-node $(($NP / $NODES)) \
    --container-name "layer_wise_benchmarks" \
    --container-mounts "$CONTAINER_MOUNTS" \
    --container-workdir "$(realpath "$(pwd)")" \
    "$@"
