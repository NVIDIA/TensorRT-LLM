#!/bin/bash

set -Eeuo pipefail

# Setup paths
llmapiLaunchScript="${llmsrc}/tensorrt_llm/llmapi/trtllm-llmapi-launch"
chmod +x $llmapiLaunchScript
cd ${llmsrc}/tests/integration/defs

# Get trtllm wheel path
trtllmWhlPath=$(pip3 show tensorrt_llm | grep Location | cut -d ' ' -f 2)
trtllmWhlPath=$(echo "$trtllmWhlPath" | sed 's/[[:space:]]+/_/g')
echo "TRTLLM WHEEL PATH: $trtllmWhlPath"

# Setup library paths
containerPipLLMLibPath=$(pip3 show tensorrt_llm | grep "Location" | awk -F ":" '{ gsub(/ /, "", $2); print $2"/tensorrt_llm/libs"}')
containerPipLLMLibPath=$(echo "$containerPipLLMLibPath" | sed 's/[[:space:]]+/_/g')
containerLDLibPath=$LD_LIBRARY_PATH
containerLDLibPath=$(echo "$containerLDLibPath" | sed 's/[[:space:]]+/_/g')
if [[ "$containerLDLibPath" != *"$containerPipLLMLibPath"* ]]; then
  containerLDLibPath="$containerPipLLMLibPath:$containerLDLibPath"
  containerLDLibPath="${containerLDLibPath%:}"
fi
export LD_LIBRARY_PATH=$containerLDLibPath

# Run pytest without llmapilaunch for DISAGG_SERVER or BENCHMARK mode.
if [[ "${DISAGG_SERVER_IDX}" == "BENCHMARK" ]] || [[ "${DISAGG_SERVER_IDX}" == "DISAGG_SERVER" ]]; then
    echo "Running pytest without llmapilaunch"
    CompletePytestCommand="$pytestPrefix $pytestCommand"
else
    echo "Running pytest with llmapilaunch"
    CompletePytestCommand="$pytestPrefix $LLMAPILaunch $pytestCommand"
fi

echo "Full Command: $CompletePytestCommand"

eval "$CompletePytestCommand"
echo "Rank${SLURM_PROCID} Pytest finished execution"
