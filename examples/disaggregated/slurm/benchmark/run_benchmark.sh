#!/bin/bash

# Add error handling
set -e
set -u
trap 'echo "Error occurred at line $LINENO"; exit 1' ERR

# Add parameter validation
if [ "$#" -lt 9 ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 model_name dataset_file multi_round concurrency_list streaming log_path hostname port"
    exit 1
fi

model_name=$1
dataset_file=$2
multi_round=$3
num_gen_servers=$4
concurrency_list=$5
streaming=$6
log_path=$7
hostname=$8
port=$9

# check process id is not 0
if [[ ${SLURM_PROCID} != "0" ]]; then
    echo "Process id is ${SLURM_PROCID} for loadgen, exiting"
    exit 0
fi

echo "Hostname: ${hostname}, Port: ${port}"
echo "Starting benchmark..."
for concurrency in ${concurrency_list}; do
    concurrency=$((concurrency * num_gen_servers))
    num_prompts=$((concurrency * multi_round))
    echo "Benchmarking with concurrency ${concurrency} ... ${num_prompts} prompts"
    mkdir -p ${log_path}/concurrency_${concurrency}
    python -m tensorrt_llm.serve.scripts.benchmark_serving \
        --model ${model_name} \
        --backend openai \
        --host ${hostname} \
        --port ${port} \
        --dataset-name "trtllm_custom" \
        --dataset-path ${dataset_file} \
        --num-prompts ${num_prompts} \
        --max-concurrency ${concurrency} \
        --trust-remote-code \
        --ignore-eos \
        --no-test-input \
        --save-result \
        --result-dir "${log_path}/concurrency_${concurrency}" \
        --result-filename "result.json" \
        --percentile-metrics "ttft,tpot,itl,e2el" \
        $(if [ "${streaming}" = "false" ]; then echo "--non-streaming"; fi)
    echo "Benchmark with concurrency ${concurrency} done"
done
