#!/bin/bash

# Enable strict error handling
set -euo pipefail
trap 'echo "Error occurred at line $LINENO"; exit 1' ERR

# Constants
readonly TIMEOUT=1800  # 30 minutes
readonly HEALTH_CHECK_INTERVAL=10
readonly STATUS_UPDATE_INTERVAL=30
readonly BENCH_SERVING_REPO="https://github.com/kedarpotdar-nv/bench_serving.git"
readonly BENCH_SERVING_DIR="/tmp/bench_serving"
readonly BENCH_SCRIPT="${BENCH_SERVING_DIR}/benchmark_serving.py"

usage() {
    cat << EOF
Usage: $0 model_name input_seq_len output_seq_len ratio multi_round num_gen_servers concurrency_list streaming log_path
    model_name      : Name of the model to benchmark
    input_seq_len   : Input sequence length
    output_seq_len  : Output sequence length
    ratio          : Random range ratio
    multi_round    : Number of multi rounds
    num_gen_servers: Number of generation servers
    concurrency_list: List of concurrency values
    streaming      : Enable/disable streaming (true/false)
    log_path       : Path to store logs
EOF
    exit 1
}

# Validate arguments
[[ $# -lt 9 ]] && usage

# Parse arguments
model_name=$1
input_seq_len=$2
output_seq_len=$3
ratio=$4
multi_round=$5
num_gen_servers=$6
concurrency_list=$7
streaming=$8
log_path=$9

# Exit if not primary process
[[ ${SLURM_PROCID:-0} != "0" ]] && { echo "Process id is ${SLURM_PROCID} for loadgen, exiting"; exit 0; }

config_file="${log_path}/server_config.yaml"

wait_for_file() {
    local file=$1
    local start_time=$(date +%s)

    while [ ! -f "${file}" ]; do
        local elapsed=$(($(date +%s) - start_time))
        [[ $elapsed -ge $TIMEOUT ]] && { echo "Error: File ${file} not found within ${TIMEOUT} seconds"; exit 1; }
        [[ $((elapsed % STATUS_UPDATE_INTERVAL)) -eq 0 ]] && echo "Waiting for file... (${elapsed}s elapsed)"
        sleep $HEALTH_CHECK_INTERVAL
    done
}

wait_for_server() {
    local host=$1
    local port=$2
    local start_time=$(date +%s)

    while ! curl -s -o /dev/null -w "%{http_code}" "http://${host}:${port}/health"; do
        local elapsed=$(($(date +%s) - start_time))
        [[ $elapsed -ge $TIMEOUT ]] && { echo "Error: Server not healthy after ${TIMEOUT} seconds"; exit 1; }
        [[ $((elapsed % STATUS_UPDATE_INTERVAL)) -eq 0 ]] && echo "Waiting for server... (${elapsed}s elapsed)"
        sleep $HEALTH_CHECK_INTERVAL
    done
}

extract_server_info() {
    local config=$1
    hostname=$(grep -i "hostname:" "${config}" | awk '{print $2}')
    port=$(grep -i "port:" "${config}" | awk '{print $2}')
    [[ -z "$hostname" || -z "$port" ]] && { echo "Error: Failed to extract hostname or port from config file"; exit 1; }
    echo "Hostname: ${hostname}, Port: ${port}"
}

do_get_logs() {
    local worker_log_path=$1
    local output_folder=$2
    grep -a "'num_ctx_requests': 0, 'num_ctx_tokens': 0" "${worker_log_path}" > "${output_folder}/gen_only.txt" || true
    grep -a "'num_generation_tokens': 0" "${worker_log_path}" > "${output_folder}/ctx_only.txt" || true
}

cleanup_processes() {
    echo "Cleaning up processes..."
    pkill -f 'start_server.sh|start_worker_e2e.sh|trtllm-serve' || true
    sleep 20  # Allow time for cleanup

    if pgrep -f "trtllm-serve"; then
        echo "Warning: Some processes may still be running"
    else
        echo "All processes successfully terminated"
    fi
}

# Main execution flow
wait_for_file "${config_file}"
extract_server_info "${config_file}"

# Clean up and clone benchmark repository
if [ -d "${BENCH_SERVING_DIR}" ]; then
    echo "Removing existing benchmark directory..."
    rm -rf "${BENCH_SERVING_DIR}"
fi
echo "Cloning benchmark repository..."
git clone "${BENCH_SERVING_REPO}" "${BENCH_SERVING_DIR}"

# Wait for server to be healthy
wait_for_server "${hostname}" "${port}"

# Run benchmarks
echo "Starting benchmark..."
for concurrency in ${concurrency_list}; do
    concurrency=$((concurrency * num_gen_servers))
    num_prompts=$((concurrency * multi_round))
    output_dir="${log_path}/concurrency_${concurrency}"

    echo "Benchmarking with concurrency ${concurrency} ... ${num_prompts} prompts"
    mkdir -p "${output_dir}"

    python "${BENCH_SCRIPT}" \
        --model "${model_name}" \
        --host "${hostname}" \
        --port "${port}" \
        --dataset-name random \
        --num-prompts "${num_prompts}" \
        --max-concurrency "${concurrency}" \
        --ignore-eos \
        --random-input-len "${input_seq_len}" \
        --random-output-len "${output_seq_len}" \
        --random-range-ratio "${ratio}" \
        --save-result \
        --use-chat-template \
        --result-dir "${output_dir}" \
        --result-filename "result.json" \
        --percentile-metrics "ttft,tpot,itl,e2el" \
        $([ "${streaming}" = "false" ] && echo "--non-streaming")

    echo "Benchmark with concurrency ${concurrency} done"
done

# Save job information
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "${SLURM_JOB_NODELIST}" > "${log_path}/job_${SLURM_JOB_ID}.txt"
fi

# Cleanup
cleanup_processes
