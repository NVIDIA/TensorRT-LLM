#!/bin/bash

# Set up error handling
set -Eeuo pipefail
trap 'rc=$?; echo "Error in file ${BASH_SOURCE[0]} on line $LINENO: $BASH_COMMAND (exit $rc)"; exit $rc' ERR

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

llmapiLaunchScript="$llmSrcNode/tensorrt_llm/llmapi/trtllm-llmapi-launch"
chmod +x $llmapiLaunchScript
cd $llmSrcNode/tests/integration/defs

containerPipLLMLibPath=$(pip3 show tensorrt_llm | grep "Location" | awk -F ":" '{ gsub(/ /, "", $2); print $2"/tensorrt_llm/libs"}')
containerPipLLMLibPath=$(echo "$containerPipLLMLibPath" | sed 's/[[:space:]]+/_/g')
containerLDLibPath=$LD_LIBRARY_PATH
containerLDLibPath=$(echo "$containerLDLibPath" | sed 's/[[:space:]]+/_/g')
if [[ "$containerLDLibPath" != *"$containerPipLLMLibPath"* ]]; then
  containerLDLibPath="$containerPipLLMLibPath:$containerLDLibPath"
  containerLDLibPath="${containerLDLibPath%:}"
fi
export LD_LIBRARY_PATH=$containerLDLibPath
echo "Library Path:"
echo "$LD_LIBRARY_PATH"
env | sort

echo "Full Command: $pytestCommand"

# For single-node test runs, clear all environment variables related to Slurm and MPI.
# This prevents test processes (e.g., pytest) from incorrectly initializing MPI
# when running under a single-node srun environment.
# TODO: check if we can take advantage of --export=None arg when execute srun instead
# of unset them in the script
 if [ "${SLURM_JOB_NUM_NODES:-1}" -eq 1 ]; then
    for v in ${!PMI@} ${!PMIX@} ${!MPI@} ${!OMPI@} ${!SLURM@}; do
        if [ "$v" != "SLURM_PROCID" ]; then
            unset "$v"
        fi
    done
 fi

eval $pytestCommand
echo "Rank${SLURM_PROCID:-0} Pytest finished execution"
