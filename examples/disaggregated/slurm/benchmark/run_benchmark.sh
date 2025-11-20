#!/bin/bash

# Add error handling
set -e
set -u
trap 'echo "Error occurred at line $LINENO"; exit 1' ERR

# Add parameter validation
if [ "$#" -lt 7 ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 model_name dataset_file multi_round concurrency_list streaming log_path"
    exit 1
fi

model_name=$1
dataset_file=$2
multi_round=$3
num_gen_servers=$4
concurrency_list=$5
streaming=$6
log_path=$7

# check process id is not 0
if [[ ${SLURM_PROCID} != "0" ]]; then
    echo "Process id is ${SLURM_PROCID} for loadgen, exiting"
    exit 0
fi

config_file=${log_path}/server_config.yaml

# Extract hostname and port from config file (server is already healthy)
hostname=$(grep -i "hostname:" ${config_file} | awk '{print $2}')
port=$(grep -i "port:" ${config_file} | awk '{print $2}')
if [ -z "$hostname" ] || [ -z "$port" ]; then
    echo "Error: Failed to extract hostname or port from config file"
    exit 1
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
        --ignore-eos \
        --no-test-input \
        --save-result \
        --result-dir "${log_path}/concurrency_${concurrency}" \
        --result-filename "result.json" \
        --percentile-metrics "ttft,tpot,itl,e2el" \
        $(if [ "${streaming}" = "false" ]; then echo "--non-streaming"; fi)
    echo "Benchmark with concurrency ${concurrency} done"
done

job_id=${SLURM_JOB_ID}
if [ -n "${job_id}" ]; then
    echo "${SLURM_JOB_NODELIST}" > ${log_path}/job_${job_id}.txt
fi
