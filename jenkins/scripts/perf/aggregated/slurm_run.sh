#!/bin/bash

set -Eeuo pipefail

if [ -n "${DISAGG_SERVING_TYPE:-}" ]; then
    install_lock_file="install_lock.lock.${SLURM_JOB_ID}.$(hostname).${DISAGG_SERVING_TYPE}"
else
    install_lock_file="install_lock.lock.${SLURM_JOB_ID}.$(hostname)"
fi
if [ $SLURM_LOCALID -eq 0 ]; then
    echo "Installing dependencies on $(hostname) for process $SLURM_PROCID"
    cd $llmSrcNode
    pip install -e .
    pip install -r requirements-dev.txt
    hostname
    nvidia-smi
    echo "Installation completed on $(hostname)"
    touch $install_lock_file
else
    echo "Waiting for dependencies installation on $(hostname) for process $SLURM_PROCID"
    start_time=$(date +%s)
    while [ ! -f $install_lock_file ]; do
        sleep 5
        elapsed=$(($(date +%s) - start_time))
        echo "Still waiting... elapsed time: ${elapsed}s"
    done
    total_elapsed=$(($(date +%s) - start_time))
    echo "Dependencies installation complete. Total wait time: ${total_elapsed}s"
fi

# Change to test directory
llmapiLaunchScript="$llmSrcNode/tensorrt_llm/llmapi/trtllm-llmapi-launch"
chmod +x $llmapiLaunchScript
cd $llmSrcNode/tests/integration/defs

# Get trtllm wheel path and add to pytest command
trtllmWhlPath=$(pip3 show tensorrt_llm | grep Location | cut -d ' ' -f 2)
trtllmWhlPath=$(echo "$trtllmWhlPath" | sed 's/[[:space:]]+/_/g')
echo "TRTLLM WHEEL PATH: $trtllmWhlPath"

# Set up library paths
containerPipLLMLibPath=$(pip3 show tensorrt_llm | grep "Location" | awk -F ":" '{ gsub(/ /, "", $2); print $2"/tensorrt_llm/libs"}')
containerPipLLMLibPath=$(echo "$containerPipLLMLibPath" | sed 's/[[:space:]]+/_/g')
containerLDLibPath=$LD_LIBRARY_PATH
containerLDLibPath=$(echo "$containerLDLibPath" | sed 's/[[:space:]]+/_/g')
if [[ "$containerLDLibPath" != *"$containerPipLLMLibPath"* ]]; then
  containerLDLibPath="$containerPipLLMLibPath:$containerLDLibPath"
  containerLDLibPath="${containerLDLibPath%:}"
fi
export LD_LIBRARY_PATH=$containerLDLibPath

echo "Full Command: $pytestCommand"

# Execute pytest command
eval $pytestCommand
echo "Rank${SLURM_PROCID} Pytest finished execution"
