#!/bin/bash
set -u
set -e
set -x
trap 'echo "Error occurred at line $LINENO"; exit 1' ERR

# Add parameter validation
if [ "$#" -lt 7 ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 isl osl multi_round model_name concurrency_list streaming log_path"
    exit 1
fi

isl=$1
osl=$2
multi_round=$3
model_name=$4
concurrency_list=$5
streaming=$6
log_path=$7

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

# download sharedgpt for benchmarking
shared_gpt_path=/tmp/ShareGPT_V3_unfiltered_cleaned_split.json
if [ ! -f ${shared_gpt_path} ]; then
    echo "Downloading sharedgpt..."
    wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json -O ${shared_gpt_path}
fi

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
    log_path=$1
    output_folder=$2

    for gen_file in ${log_path}/output_gen_*.log; do
        if [ -f "$gen_file" ]; then
            index=$(basename "$gen_file" | sed 's/output_gen_\(.*\)\.log/\1/')
            grep -a "'num_ctx_requests': 0, 'num_ctx_tokens': 0" "$gen_file" > "${output_folder}/gen_only_${index}.txt" || true
        fi
    done

    for ctx_file in ${log_path}/output_ctx_*.log; do
        if [ -f "$ctx_file" ]; then
            index=$(basename "$ctx_file" | sed 's/output_ctx_\(.*\)\.log/\1/')
            grep -a "'num_generation_tokens': 0" "$ctx_file" > "${output_folder}/ctx_only_${index}.txt" || true
        fi
    done
}

# run the loadgen
echo "Starting benchmark..."
for concurrency in ${concurrency_list}; do
    mkdir -p ${log_path}/concurrency_${concurrency}
    max_count=$((${concurrency} * ${multi_round}))
    echo "Running loadgen with concurrency: ${concurrency}, max_count: ${max_count}"
    python -m tensorrt_llm.serve.scripts.benchmark_serving \
        --model ${model_name} \
        --tokenizer ${model_name} \
        --dataset-name random \
        --dataset-path ${shared_gpt_path} \
        --random-input-len ${isl} \
        --random-output-len ${osl} \
        --random-prefix-len 0 \
        --num-prompts ${max_count} \
        --max-concurrency ${concurrency} \
        --host ${hostname} \
        --port ${port} \
        --ignore-eos \
        --no-test-input \
        $(if [ "${streaming}" = "false" ]; then echo "--non-streaming"; fi)

    do_get_logs ${log_path} ${log_path}/concurrency_${concurrency}
    echo "done for ${concurrency} in folder ${log_path}/concurrency_${concurrency}"
done

echo "Benchmark done, gracefully shutting down server and workers..."
kill -9 $(ps aux | grep '[s]tart_server.sh' | awk '{print $2}') >/dev/null 2>&1 || true
kill -9 $(ps aux | grep '[s]tart_worker.sh' | awk '{print $2}') >/dev/null 2>&1 || true
kill -9 $(ps aux | grep '[t]rtllm-serve' | awk '{print $2}') >/dev/null 2>&1 || true
sleep 20  # Give processes some time to clean up

# Check if there are remaining processes
if pgrep -f "trtllm-serve"; then
    echo "Warning: Some processes may still be running"
else
    echo "All processes successfully terminated"
fi
