#!/bin/bash
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
    wget -nv $llmTarfile
    tar -zxf $tarName
    which python3
    python3 --version
    apt-get install -y libffi-dev
    nvidia-smi
    cd $llmSrcNode && pip3 install --retries 1 -r requirements-dev.txt
    cd $resourcePathNode &&  pip3 install --force-reinstall --no-deps TensorRT-LLM/tensorrt_llm-*.whl
    git config --global --add safe.directory "*"
    gpuUuids=$(nvidia-smi -q | grep "GPU UUID" | awk '{print $4}' | tr '\n' ',' || true)
    echo "HOST_NODE_NAME = $hostname ; GPU_UUIDS = =$gpuUuids ; STAGE_NAME = $stageName"
    touch install_lock.lock
else
    while [ ! -f install_lock.lock ]; do
        sleep 5
    done
fi

export UCX_TLS=^gdr_copy
cd $llmSrcNode/tests/integration/defs

# get trtllm wheel path and add to pytest command
trtllmWhlPath=$(pip3 show tensorrt_llm | grep Location | cut -d ' ' -f 2)
trtllmWhlPath=$(echo "$trtllmWhlPath" | sed 's/[[:space:]]+/_/g')
echo "TRTLLM WHEEL PATH: $trtllmWhlPath"
pytestCommand=$(set_value_in_command "TRTLLM_WHL_PATH" "$trtllmWhlPath" "$pytestCommand")

# generate .coveragerc in workspace and add file path to pytest command
cat << EOF > $jobWorkspace/.coveragerc
[run]
branch = True
data_file = $jobWorkspace/.coverage.$stageName
[paths]
source =
    $llmSrcNode/tensorrt_llm/
    $trtllmWhlPath/tensorrt_llm/
EOF
pytestCommand=$(set_value_in_command "coverageConfigFile" "$jobWorkspace/.coveragerc" "$pytestCommand")

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
echo "Running: $testCase"
echo "Full Command: $pytestCommand"
eval $pytestCommand

if [ "$perfMode" = "true" ]; then
    if [[ "$stageName" == *PyTorch* ]]; then
        basePerfFilename="base_perf_pytorch.csv"
    else
        basePerfFilename="base_perf.csv"
    fi
    basePerfPath="$llmSrcNode/tests/integration/defs/perf/$basePerfFilename"
    echo "Check perf result"
    python3 $llmSrcNode/tests/integration/defs/perf/sanity_perf_check.py \
        $stageName/perf_script_test_results.csv \
        $basePerfPath
    echo "Check perf report"
    python3 $llmSrcNode/tests/integration/defs/perf/create_perf_comparison_report.py \
        --output_path $stageName/report.pdf \
        --files $stageName/perf_script_test_results.csv \
        $basePerfPath
fi
