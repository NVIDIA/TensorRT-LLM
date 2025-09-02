#!/bin/bash
cd $resourcePathNode
llmSrcNode=$resourcePathNode/TensorRT-LLM/src

# generate .coveragerc in workspace
coverageConfigFile="$jobWorkspace/.coveragerc"
cat << EOF > "$coverageConfigFile"
[run]
branch = True
data_file = $jobWorkspace/.coverage.$stageName
[paths]
source =
    $llmSrcNode/tensorrt_llm/
    ---wheel_path---/tensorrt_llm/
EOF

resultsPath=$jobWorkspace/results
mkdir -p $resultsPath
if [ $SLURM_LOCALID -eq 0 ]; then
    wget -nv $llmTarfile
    tar -zxf $tarName
    which python3
    python3 --version
    apt-get install -y libffi-dev
    nvidia-smi && nvidia-smi -q && nvidia-smi topo -m
    cd $llmSrcNode && pip3 install --retries 1 -r requirements-dev.txt
    cd $resourcePathNode &&  pip3 install --force-reinstall --no-deps TensorRT-LLM/tensorrt_llm-*.whl
    git config --global --add safe.directory "*"
    gpuUuids=$(nvidia-smi -q | grep "GPU UUID" | awk '{print $4}' | tr '\n' ',' || true)
    echo "HOST_NODE_NAME = $HOST_NODE_NAME ; GPU_UUIDS = =$gpuUuids ; STAGE_NAME = $stageName"
    touch install_lock.lock
else
    while [ ! -f install_lock.lock ]; do
        sleep 5
    done
fi
testList="$testList_$splitId"
export CPP_TEST_TIMEOUT_OVERRIDDEN=$pytestTestTimeout
export LLM_ROOT=$llmSrcNode
export LLM_MODELS_ROOT=$MODEL_CACHE_DIR
export UCX_TLS=^gdr_copy
cd $llmSrcNode/tests/integration/defs
testCmdLines=(
    "$llmSrcNode/tensorrt_llm/llmapi/trtllm-llmapi-launch"
    "pytest"
    "-v"
    "--timeout-method=thread"
    "--timeout=$pytestTestTimeout"
    "--test-list=$testListPathNode"
    "--waives-file=$waivesListPathNode"
    "--rootdir $llmSrcNode/tests/integration/defs"
    "--test-prefix=$stageName"
    "--splits $splits"
    "--group $splitId"
    "--output-dir=$jobWorkspace/"
    "--csv=$resultsPath/report.csv"
    "--junit-xml $resultsPath/results.xml"
    "-o junit_logging=out-err"
)
if [ "$perfMode" = "true" ]; then
    testCmdLines+=(
        "--perf"
        "--perf-log-formats csv"
        "--perf-log-formats yaml"
    )
fi
trtllmWhlPath=$(pip3 show tensorrt_llm | grep Location | cut -d ' ' -f 2)
trtllmWhlPath=$(echo "$trtllmWhlPath" | sed 's/[[:space:]]+/_/g')
echo "TRTLLM WHEEL PATH: $trtllmWhlPath"
sed -i "s|---wheel_path---|$trtllmWhlPath|g" "$coverageConfigFile"
testCmdLines+=(
    "--cov=$llmSrcNode/examples/"
    "--cov=$llmSrcNode/tensorrt_llm/"
    "--cov=$trtllmWhlPath/tensorrt_llm/"
    "--cov-report="
    "--cov-config=$coverageConfigFile"
)
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
fullCmd="${testCmdLines[*]}"
echo "Running: $testCase"
echo "Full Command: $fullCmd"
eval $fullCmd
