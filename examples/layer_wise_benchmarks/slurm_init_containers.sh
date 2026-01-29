#!/bin/bash

set -euo pipefail

# CONTAINER_IMAGE=
CONTAINER_NAME=${CONTAINER_NAME:-layer_wise_benchmarks}
TRTLLM_ROOT=$(realpath -- "$(dirname -- "$0")"/../..)
CONTAINER_MOUNTS="$TRTLLM_ROOT:$TRTLLM_ROOT"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "Please set SLURM_JOB_ID"
    exit 1
fi

NODES=$(squeue -j $SLURM_JOB_ID -h -o "%D")

if [ -z "${CONTAINER_IMAGE:-}" ]; then
    # Read Docker image from current_image_tags.properties
    MACHINE="$(srun -N 1 uname -m)"
    if [ "$MACHINE" == "x86_64" ]; then
        DOCKER_IMAGE=$(source "$TRTLLM_ROOT/jenkins/current_image_tags.properties" && echo "$LLM_DOCKER_IMAGE")
    elif [ "$MACHINE" == "aarch64" ]; then
        DOCKER_IMAGE=$(source "$TRTLLM_ROOT/jenkins/current_image_tags.properties" && echo "$LLM_SBSA_DOCKER_IMAGE")
    else
        echo "Unsupported machine hardware name \"$MACHINE\""
        exit 1
    fi

    # Change "urm.nvidia.com/sw-tensorrt-docker/..." to "urm.nvidia.com#sw-tensorrt-docker/..." to bypass credentials
    DOCKER_IMAGE="${DOCKER_IMAGE/\//#}"
    echo "CONTAINER_IMAGE was not set, using Docker image $DOCKER_IMAGE"

    # Import to .sqsh file
    SQSH_FILE_NAME=$(printf '%s\n' "$DOCKER_IMAGE" |
                     awk -F'#' '{print $2}' |
                     awk -F':' '{gsub(/\//,"+",$1); print $1"+"$2".sqsh"}')
    CONTAINER_IMAGE="$TRTLLM_ROOT/enroot/$SQSH_FILE_NAME"
    if [ ! -f "$CONTAINER_IMAGE" ]; then
        echo "Container image file $CONTAINER_IMAGE does not exist, importing ..."
        srun -N 1 --pty enroot import -o "$CONTAINER_IMAGE" "docker://$DOCKER_IMAGE"
    fi
fi

WORKDIR=$(realpath -- "$(pwd)")

set -x
srun -N "$NODES" \
    --ntasks-per-node 1 \
    --container-image "$CONTAINER_IMAGE" \
    --container-name "$CONTAINER_NAME" \
    --container-mounts "$CONTAINER_MOUNTS" \
    --container-workdir "$WORKDIR" \
bash -c 'cd "$1" &&
    pip install -U packaging &&
    pip install -r requirements.txt --no-build-isolation &&
    pip install -e .' bash "$TRTLLM_ROOT"
