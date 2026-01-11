#!/bin/bash

# Set up error handling
set -xEeuo pipefail
trap 'rc=$?; echo "Error in file ${BASH_SOURCE[0]} on line $LINENO: $BASH_COMMAND (exit $rc)"; exit $rc' ERR

# Source utilities
bashUtilsPath="$(dirname "${BASH_SOURCE[0]}")/$(basename "${BASH_SOURCE[0]}" | sed 's/slurm_install\.sh/bash_utils.sh/')"
source "$bashUtilsPath"

slurm_install_setup() {
    cd $resourcePathNode
    llmSrcNode=$resourcePathNode/TensorRT-LLM/src

    # Use unique lock file for this job ID
    lock_file="install_lock_job_${SLURM_JOB_ID:-local}_node_${SLURM_NODEID:-0}.lock"

    if [ $SLURM_LOCALID -eq 0 ]; then
        if [ -f "$lock_file" ]; then
            rm -f "$lock_file"
        fi

        retry_command bash -c "wget -nv $llmTarfile && tar -zxf $tarName"
        which python3
        python3 --version
        retry_command apt-get install -y libffi-dev
        nvidia-smi && nvidia-smi -q && nvidia-smi topo -m
        if [[ $pytestCommand == *--run-ray* ]]; then
            retry_command pip3 install --retries 10 ray[default]
        fi
        retry_command bash -c "cd $llmSrcNode && pip3 install --retries 10 -r requirements-dev.txt"
        retry_command bash -c "cd $resourcePathNode && pip3 install --retries 10 --force-reinstall --no-deps TensorRT-LLM/tensorrt_llm-*.whl"
        gpuUuids=$(nvidia-smi -q | grep "GPU UUID" | awk '{print $4}' | tr '\n' ',' || true)
        hostNodeName="${HOST_NODE_NAME:-$(hostname -f || hostname)}"
        echo "HOST_NODE_NAME = $hostNodeName ; GPU_UUIDS = $gpuUuids ; STAGE_NAME = $stageName"
        echo "(Writing install lock) Current directory: $(pwd)"
        touch "$lock_file"
    else
        echo "(Waiting for install lock) Current directory: $(pwd)"
        while [ ! -f "$lock_file" ]; do
            sleep 10
        done
    fi
}

# Only run slurm_install_setup when script is executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    slurm_install_setup
fi
