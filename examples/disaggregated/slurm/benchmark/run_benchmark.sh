#!/bin/bash

# Add error handling
set -e
set -u
trap 'echo "Error occurred at line $LINENO"; exit 1' ERR

# Add parameter validation
if [ "$#" -lt 7 ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 model_name input_seq_len output_seq_len dataset_file multi_round num_gen_servers concurrency_list streaming log_path enable_time_breakdown"
    exit 1
fi

model_name=$1
input_seq_len=$2
output_seq_len=$3
dataset_file=$4
multi_round=$5
num_gen_servers=$6
concurrency_list=$7
streaming=$8
log_path=$9
enable_time_breakdown=${10:-false}

# check process id is not 0
if [[ ${SLURM_PROCID} != "0" ]]; then
    echo "Process id is ${SLURM_PROCID} for loadgen, exiting"
    exit 0
fi

config_file=${log_path}/server_config.yaml

# check if the config file exists every 10 seconds timeout 1800 seconds
timeout=1800
start_time=$(date +%s)
while [ ! -f ${config_file} ]; do
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    if [ $elapsed -ge $timeout ]; then
        echo "Error: Config file ${config_file} not found within ${timeout} seconds"
        exit 1
    fi
    if [ $((elapsed % 30)) -eq 0 ]; then
        echo "Waiting for config file... (${elapsed}s elapsed)"
    fi
    sleep 10
done

# grep the host and port from the config file
hostname=$(grep -i "hostname:" ${config_file} | awk '{print $2}')
port=$(grep -i "port:" ${config_file} | awk '{print $2}')
if [ -z "$hostname" ] || [ -z "$port" ]; then
    echo "Error: Failed to extract hostname or port from config file"
    exit 1
fi
echo "Hostname: ${hostname}, Port: ${port}"

# check server is health by curl every 10 seconds timeout 1800 seconds
timeout=1800
start_time=$(date +%s)
while ! curl -s -o /dev/null -w "%{http_code}" http://${hostname}:${port}/health; do
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    if [ $elapsed -ge $timeout ]; then
        echo "Error: Server is not healthy after ${timeout} seconds"
        exit 1
    fi
    if [ $((elapsed % 30)) -eq 0 ]; then
        echo "Waiting for server to be healthy... (${elapsed}s elapsed)"
    fi
    sleep 10
done

# try client

do_get_logs(){
    worker_log_path=$1
    output_folder=$2

    # Check if log file exists
    if [ ! -f "${worker_log_path}" ]; then
        echo "Warning: Worker log file ${worker_log_path} not found"
        touch "${output_folder}/gen_only.txt"
        touch "${output_folder}/ctx_only.txt"
        return 0
    fi

    # Create output folder if it doesn't exist
    mkdir -p "${output_folder}"

    # Extract metrics with better error handling
    if ! grep -a "'num_ctx_requests': 0, 'num_ctx_tokens': 0" "${worker_log_path}" > "${output_folder}/gen_only.txt" 2>/dev/null; then
        echo "Note: No generation-only metrics found in ${worker_log_path}"
        touch "${output_folder}/gen_only.txt"
    fi

    if ! grep -a "'num_generation_tokens': 0" "${worker_log_path}" > "${output_folder}/ctx_only.txt" 2>/dev/null; then
        echo "Note: No context-only metrics found in ${worker_log_path}"
        touch "${output_folder}/ctx_only.txt"
    fi
}

echo "Starting benchmark..."
for concurrency in ${concurrency_list}; do
    concurrency=$((concurrency * num_gen_servers))
    num_prompts=$((concurrency * multi_round))
    echo "Benchmarking with concurrency ${concurrency} ... ${num_prompts} prompts"
    mkdir -p ${log_path}/concurrency_${concurrency}

    # Set dataset arguments based on whether dataset_file is provided
    if [ -z "${dataset_file}" ]; then
        dataset_args="--dataset-name random --random-ids"
    else
        dataset_args="--dataset-name trtllm_custom --dataset-path ${dataset_file}"
    fi

    python -m tensorrt_llm.serve.scripts.benchmark_serving \
        --model ${model_name} \
        --backend openai \
        --host ${hostname} \
        --port ${port} \
        ${dataset_args} \
        --num-prompts ${num_prompts} \
        --max-concurrency ${concurrency} \
        --ignore-eos \
        --random-input-len "${input_seq_len}" \
        --random-output-len "${output_seq_len}" \
        --no-test-input \
        --save-result \
        --result-dir "${log_path}/concurrency_${concurrency}" \
        --result-filename "result.json" \
        --percentile-metrics "ttft,tpot,itl,e2el" \
        $(if [ "${streaming}" = "false" ]; then echo "--non-streaming"; fi) \
        $(if [ "${enable_time_breakdown}" = "true" ]; then echo "--save-request-time-breakdown"; fi)
    echo "Benchmark with concurrency ${concurrency} done"
done

job_id=${SLURM_JOB_ID}
if [ -n "${job_id}" ]; then
    echo "${SLURM_JOB_NODELIST}" > ${log_path}/job_${job_id}.txt
fi

echo "Benchmark done, gracefully shutting down server and workers..."
kill -9 $(ps aux | grep '[s]tart_server.sh' | awk '{print $2}') >/dev/null 2>&1 || true
kill -9 $(ps aux | grep '[s]tart_worker.sh' | awk '{print $2}') >/dev/null 2>&1 || true
kill -9 $(ps aux | grep '[t]rtllm-serve' | awk '{print $2}') >/dev/null 2>&1 || true
sleep 20  # Give processes some time to clean up

# Check if there are any remaining processes
if pgrep -f "trtllm-serve"; then
    echo "Warning: Some processes may still be running"
else
    echo "All processes successfully terminated"
fi
