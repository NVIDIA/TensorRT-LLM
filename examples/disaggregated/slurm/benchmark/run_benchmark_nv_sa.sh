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
Usage: $0 model_name input_seq_len output_seq_len ratio multi_round num_gen_servers concurrency_list streaming log_path hostname port ucx_warmup_requests
    model_name          : Name of the model to benchmark
    input_seq_len       : Input sequence length
    output_seq_len      : Output sequence length
    ratio               : Random range ratio
    multi_round         : Number of multi rounds
    num_gen_servers     : Number of generation servers
    concurrency_list    : List of concurrency values
    streaming           : Enable/disable streaming (true/false)
    log_path            : Path to store logs
    hostname            : Hostname of the server
    port                : Port of the server
    ucx_warmup_requests : Number of UCX warmup requests
EOF
    exit 1
}

# Validate arguments
[[ $# -lt 12 ]] && usage

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
ucx_warmup_requests=${12}

# check process id is not 0
if [[ ${SLURM_PROCID} != "0" ]]; then
    echo "Process id is ${SLURM_PROCID} for loadgen, exiting"
    exit 0
fi

echo "Hostname: ${hostname}, Port: ${port}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/benchmark_utils.sh"

# Clean up and clone benchmark repository
if [ -d "${BENCH_SERVING_DIR}" ]; then
    echo "Removing existing benchmark directory..."
    rm -rf "${BENCH_SERVING_DIR}"
fi
echo "Cloning benchmark repository..."
git clone "${BENCH_SERVING_REPO}" "${BENCH_SERVING_DIR}"

setup_start_logs "${SLURM_JOB_ID}" "${log_path}"

# warmup requests for ucx connections
if [ "${ucx_warmup_requests}" -gt 0 ]; then
    echo "warming up ucx connections with small requests... ${ucx_warmup_requests}"
    python -m tensorrt_llm.serve.scripts.benchmark_serving \
        --model ${model_name} \
        --dataset-name random \
        --random-ids \
        --random-input-len 100 \
        --random-output-len 10 \
        --num-prompts ${ucx_warmup_requests} \
        --host ${hostname} \
        --port ${port} \
        --ignore-eos \
        --non-streaming
    echo "UCX warmup done"
fi

# Run benchmarks
echo "Starting benchmark..."
for concurrency in ${concurrency_list}; do
    concurrency=$((concurrency * num_gen_servers))
    num_prompts=$((concurrency * multi_round))
    output_dir="${log_path}/concurrency_${concurrency}"
    echo "Benchmarking with concurrency ${concurrency} ... ${num_prompts} prompts"
    mkdir -p "${output_dir}"
    do_process_all_logs ${log_path}/ ${log_path}/concurrency_${concurrency} "line" "${log_path}"

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

    # Print failed request count (consistent with non-nv_sa benchmark format)
    python - "${output_dir}/result.json" <<-'PYEOF'
	import json
	import sys

	try:
	    with open(sys.argv[1], encoding="utf-8") as f:
	        d = json.load(f)
	    failed = d["num_prompts"] - d["completed"]
	    print(f"Total failed requests: {failed}")
	except (OSError, json.JSONDecodeError, KeyError) as exc:
	    print(f"WARNING: failed to read request counts from {sys.argv[1]}: {exc}", file=sys.stderr)
	PYEOF

    echo "Benchmark with concurrency ${concurrency} done"
    do_process_all_logs ${log_path}/ ${log_path}/concurrency_${concurrency} "log" "${log_path}"
done
