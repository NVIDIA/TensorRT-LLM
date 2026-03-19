#!/bin/bash

# aiperf-based benchmark script for disaggregated serving
# Args: model_name dataset_file multi_round num_gen_servers concurrency_list streaming log_path hostname port ucx_warmup_requests

set -e
set -u
trap 'echo "Error occurred at line $LINENO"; exit 1' ERR

if [ "$#" -lt 10 ]; then
    echo "Error: Missing required arguments, got $# arguments, args: $@"
    echo "Usage: $0 model_name dataset_file multi_round num_gen_servers concurrency_list streaming log_path hostname port ucx_warmup_requests"
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
ucx_warmup_requests=${10}

# check process id is not 0
if [[ ${SLURM_PROCID} != "0" ]]; then
    echo "Process id is ${SLURM_PROCID} for loadgen, exiting"
    exit 0
fi

# Always install/upgrade aiperf to ensure we have the version with trust_remote_code fix
# (container may have an older version with parallel_decode.py that lacks trust_remote_code)
echo "Installing aiperf..."
pip install --force-reinstall --no-deps 'aiperf @ git+https://github.com/ai-dynamo/aiperf.git@ac3d91652e5e024bfb4ac38d48603423aad666bc'

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
        --trust-remote-code \
        --non-streaming
    echo "UCX warmup done"
fi

# Trust remote code globally for custom tokenizers in parallel workers
export HF_HUB_TRUST_REMOTE_CODE=1

echo "Hostname: ${hostname}, Port: ${port}"
echo "Starting aiperf benchmark..."

concurrency_list=$(echo "${concurrency_list}" | tr ',' ' ')
for concurrency in ${concurrency_list}; do
    concurrency=$((concurrency))
    request_count=$((concurrency * multi_round))
    # benchmark_duration: 20min per round
    benchmark_duration=$((multi_round * 1200))
    echo "Benchmarking with concurrency ${concurrency} ... ${request_count} requests, duration ${benchmark_duration}s"
    mkdir -p ${log_path}/concurrency_${concurrency}

    aiperf profile \
        -m ${model_name} \
        --tokenizer ${model_name} \
        --tokenizer-trust-remote-code \
        --url http://${hostname}:${port} \
        --streaming \
        --ui simple \
        --input-file ${dataset_file} \
        --artifact-dir ${log_path}/concurrency_${concurrency} \
        --concurrency ${concurrency} \
        --concurrency-ramp-duration 60 \
        --custom-dataset-type mooncake_trace \
        --benchmark-duration ${benchmark_duration} \
        --benchmark-grace-period 60 \
        --workers-max 200 \
        --request-timeout-seconds 1200 \
        --profile-export-level records \
        --extra-inputs ignore_eos:true \
        --request-count ${request_count} \
        --record-processors 8

    echo "Benchmark with concurrency ${concurrency} done"
done

# Fetch perf metrics from disagg server
echo "Fetching perf metrics from http://${hostname}:${port}/perf_metrics ..."
curl -s "http://${hostname}:${port}/perf_metrics" > ${log_path}/perf_metrics.json 2>&1 || true
if [ -s "${log_path}/perf_metrics.json" ]; then
    echo "Perf metrics saved to ${log_path}/perf_metrics.json"
else
    echo "Warning: perf_metrics response was empty or endpoint not available"
fi
