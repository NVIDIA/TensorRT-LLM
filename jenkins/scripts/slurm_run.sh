#!/bin/bash

# Set up error handling
set -Eeuo pipefail
trap 'rc=$?; echo "Error in file ${BASH_SOURCE[0]} on line $LINENO: $BASH_COMMAND (exit $rc)"; exit $rc' ERR

cd $resourcePathNode
llmSrcNode=$resourcePathNode/TensorRT-LLM/src

set_value_in_command() {
    # Parameters
    local key="$1"
    local value="$2"
    local command="$3"

    # Transform the key
    local placeholder="__PLACEHOLDER_${key}__"

    # Check if placeholder exists
    if [[ "$command" != *"$placeholder"* ]]; then
        echo "Error: placeholder '$placeholder' not found in the command" >&2
        return 1
    fi

    # Replace all occurrences
    local result="${command//${placeholder}/${value}}"

    # Return the result
    echo "$result"
}

resultsPath=$jobWorkspace/results
mkdir -p $resultsPath
if [ $SLURM_LOCALID -eq 0 ]; then
    # save job ID in $jobWorkspace/slurm_job_id.txt for later job to retrieve
    echo $SLURM_JOB_ID > $jobWorkspace/slurm_job_id.txt

    wget -nv $llmTarfile
    tar -zxf $tarName
    which python3
    python3 --version
    apt-get install -y libffi-dev
    nvidia-smi && nvidia-smi -q && nvidia-smi topo -m
    if [[ $pytestCommand == *--run-ray* ]]; then
        pip3 install ray[default]
    fi
    cd $llmSrcNode && pip3 install --retries 1 -r requirements-dev.txt
    cd $resourcePathNode &&  pip3 install --force-reinstall --no-deps TensorRT-LLM/tensorrt_llm-*.whl
    git config --global --add safe.directory "*"
    gpuUuids=$(nvidia-smi -q | grep "GPU UUID" | awk '{print $4}' | tr '\n' ',' || true)
    hostNodeName="${HOST_NODE_NAME:-$(hostname -f || hostname)}"
    echo "HOST_NODE_NAME = $hostNodeName ; GPU_UUIDS = $gpuUuids ; STAGE_NAME = $stageName"
    touch install_lock.lock
else
    while [ ! -f install_lock.lock ]; do
        sleep 5
    done
fi


llmapiLaunchScript="$llmSrcNode/tensorrt_llm/llmapi/trtllm-llmapi-launch"
chmod +x $llmapiLaunchScript
cd $llmSrcNode/tests/integration/defs

# get trtllm wheel path and add to pytest command
trtllmWhlPath=$(pip3 show tensorrt_llm | grep Location | cut -d ' ' -f 2)
trtllmWhlPath=$(echo "$trtllmWhlPath" | sed 's/[[:space:]]+/_/g')
echo "TRTLLM WHEEL PATH: $trtllmWhlPath"
if [ $SLURM_LOCALID -eq 0 ]; then
    sed -i "s|---wheel_path---|$trtllmWhlPath|g" "$coverageConfigFile"
fi
pytestCommand=$(set_value_in_command "TRTLLM_WHL_PATH" "$trtllmWhlPath" "$pytestCommand")

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
echo "Rank${SLURM_PROCID} Pytest finished execution"

if [ "$perfMode" = "true" ]; then
    if [[ "$stageName" == *PyTorch* ]]; then
        basePerfFilename="base_perf_pytorch.csv"
    else
        basePerfFilename="base_perf.csv"
    fi
    basePerfPath="$llmSrcNode/tests/integration/defs/perf/$basePerfFilename"
    echo "Check Perf Result"
    python3 $llmSrcNode/tests/integration/defs/perf/sanity_perf_check.py \
        $stageName/perf_script_test_results.csv \
        $basePerfPath
    echo "Check Perf Result"
    python3 $llmSrcNode/tests/integration/defs/perf/create_perf_comparison_report.py \
        --output_path $stageName/report.pdf \
        --files $stageName/perf_script_test_results.csv \
        $basePerfPath
fi
