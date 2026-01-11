#!/bin/bash

# Enable strict error handling
set -euo pipefail
trap 'echo "Error occurred at line $LINENO"; exit 1' ERR

# Constants
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
    hostname       : Hostname of the server
    port           : Port of the server
EOF
    exit 1
}

# Validate arguments
[[ $# -lt 11 ]] && usage

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
hostname=${10}
port=${11}

# check process id is not 0
if [[ ${SLURM_PROCID} != "0" ]]; then
    echo "Process id is ${SLURM_PROCID} for loadgen, exiting"
    exit 0
fi

echo "Hostname: ${hostname}, Port: ${port}"

# Clean up and clone benchmark repository
if [ -d "${BENCH_SERVING_DIR}" ]; then
    echo "Removing existing benchmark directory..."
    rm -rf "${BENCH_SERVING_DIR}"
fi
echo "Cloning benchmark repository..."
git clone "${BENCH_SERVING_REPO}" "${BENCH_SERVING_DIR}"

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
        --trust-remote-code \
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
