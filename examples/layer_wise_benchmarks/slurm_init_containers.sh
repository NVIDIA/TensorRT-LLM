#!/bin/bash

set -euo pipefail

# DOCKER_IMAGE=
# CONTAINER_MOUNTS=

if [ "${SLURM_JOB_ID:-}" == "" ]; then
    echo "Please set SLURM_JOB_ID"
    exit 1
fi

NODES=$(squeue -j $SLURM_JOB_ID -h -o "%D")

if [ "${DOCKER_IMAGE:-}" == "" ]; then
    source ../../jenkins/current_image_tags.properties

    MACHINE="$(uname -m)"
    if [ "$MACHINE" == "x86_64" ]; then
        DOCKER_IMAGE=$LLM_DOCKER_IMAGE
    elif [ "$MACHINE" == "aarch64" ]; then
        DOCKER_IMAGE=$LLM_SBSA_DOCKER_IMAGE
    else
        echo "Unsupported machine hardware name \"$MACHINE\""
    fi

    DOCKER_IMAGE="${DOCKER_IMAGE/\//#}"
    echo "DOCKER_IMAGE was not set, set to $DOCKER_IMAGE"
fi

set -x
srun --mpi=pmix \
    -N "$NODES" \
    --ntasks-per-node 1 \
    --container-image "$DOCKER_IMAGE" \
    --container-name "layer_wise_benchmarks" \
    --container-mounts "$CONTAINER_MOUNTS" \
    --container-workdir "$(pwd)" \
    pip install -e ../..
