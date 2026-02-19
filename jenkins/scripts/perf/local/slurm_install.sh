#!/bin/bash

# Set up error handling
set -xEeuo pipefail
trap 'rc=$?; echo "Error in file ${BASH_SOURCE[0]} on line $LINENO: $BASH_COMMAND (exit $rc)"; exit $rc' ERR

# Source bash utilities for retry_command
source "$(dirname "${BASH_SOURCE[0]}")/../../bash_utils.sh"

slurm_build_wheel() {
    if [ "${BUILD_WHEEL:-false}" != "true" ]; then
        echo "BUILD_WHEEL is not true, skipping wheel build"
        return
    fi

    build_lock_file="build_wheel_lock_${SLURM_JOB_ID:-local}.lock"
    if [ "${SLURM_NODEID:-0}" -eq 0 ] && [ "${SLURM_LOCALID:-0}" -eq 0 ]; then
        cd $jobWorkspace
        if [ -f "$build_lock_file" ]; then
            rm -f "$build_lock_file"
        fi

        echo "Building wheel on node ${SLURM_NODEID:-0}, task ${SLURM_LOCALID:-0}"
        retry_command bash -c "cd $llmSrcNode && rm -rf .venv-3.12 && python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt --benchmarks --use_ccache --cuda_architectures '100-real' --clean -c"

        cd $jobWorkspace
        echo "(Writing build wheel lock) Lock file: $build_lock_file"
        touch "$build_lock_file"
    else
        cd $jobWorkspace
        echo "(Waiting for build wheel lock) Lock file: $build_lock_file"
        while [ ! -f "$build_lock_file" ]; do
            sleep 10
        done
    fi
    echo "Build wheel completed"
}

slurm_install_setup() {
    lock_file="install_lock_job_${SLURM_JOB_ID:-local}_node_${SLURM_NODEID:-0}.lock"
    if [ "${SLURM_LOCALID:-0}" -eq 0 ]; then
        cd /tmp
        if [ -f "$lock_file" ]; then
            rm -f "$lock_file"
        fi

        echo "(Installing TensorRT-LLM and requirements) Current directory: $(pwd)"
        retry_command bash -c "cd $llmSrcNode && pip install --retries 10 -e . && pip install --retries 10 -r requirements-dev.txt"

        cd /tmp
        echo "(Writing install lock) Current directory: $(pwd)"
        touch "$lock_file"
    else
        cd /tmp
        echo "(Waiting for install lock) Current directory: $(pwd)"
        while [ ! -f "$lock_file" ]; do
            sleep 10
        done
    fi
    echo "Install completed"
}

# Only run when script is executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    slurm_build_wheel
    slurm_install_setup
fi
