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

echo "Installing aiperf..."
aiperf_repo=${AIPERF_REPO:-}
if [[ -n "${aiperf_repo}" ]]; then
    if [[ ! -d "${aiperf_repo}" ]]; then
        echo "AIPerf repository does not exist: ${aiperf_repo}" >&2
        exit 1
    fi
    echo "Installing AIPerf from workspace: ${aiperf_repo}"
    git -C "${aiperf_repo}" log -1 --oneline
    pip install --force-reinstall --no-deps "${aiperf_repo}"
else
    pip install --force-reinstall --no-deps 'aiperf @ git+https://github.com/ai-dynamo/aiperf.git@ac3d91652e5e024bfb4ac38d48603423aad666bc'
fi
if [[ -n "${AIPERF_DATASETS_VERSION:-}" ]]; then
    echo "Installing Hugging Face datasets ${AIPERF_DATASETS_VERSION}..."
    pip install --force-reinstall --no-deps "datasets==${AIPERF_DATASETS_VERSION}"
fi
python -c 'import datasets; print(f"Hugging Face datasets: {datasets.__version__}")'


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

if [[ -n "${AIPERF_RUNTIME_TMPDIR:-}" ]]; then
    mkdir -p "${AIPERF_RUNTIME_TMPDIR}"
    export TMPDIR="${AIPERF_RUNTIME_TMPDIR}"
    echo "AIPerf runtime TMPDIR: ${TMPDIR}"
fi

concurrency_list=$(echo "${concurrency_list}" | tr ',' ' ')
for concurrency in ${concurrency_list}; do
    concurrency=$((concurrency))
    request_count=$((concurrency * multi_round))
    # benchmark_duration: 20min per round
    benchmark_duration=$((multi_round * 1200))
    echo "Benchmarking with concurrency ${concurrency} ... ${request_count} requests, duration ${benchmark_duration}s"
    mkdir -p "${log_path}/concurrency_${concurrency}"

    dataset_args=()
    stop_args=(
        --benchmark-duration "${benchmark_duration}"
        --benchmark-grace-period "${AIPERF_BENCHMARK_GRACE_PERIOD:-60}"
    )
    if [[ -n "${AIPERF_PUBLIC_DATASET:-}" ]]; then
        dataset_args+=(--public-dataset "${AIPERF_PUBLIC_DATASET}")
        if [[ "${AIPERF_PUBLIC_DATASET}" == "weka_hf" ]]; then
            if [[ -z "${AIPERF_HF_WEKA_REPO:-}" ]]; then
                echo "AIPERF_HF_WEKA_REPO is required for weka_hf" >&2
                exit 1
            fi
            dataset_args+=(--hf-weka-repo "${AIPERF_HF_WEKA_REPO}")
        fi
        if [[ -n "${AIPERF_NUM_DATASET_ENTRIES:-}" ]]; then
            dataset_args+=(--num-dataset-entries "${AIPERF_NUM_DATASET_ENTRIES}")
        fi
        if [[ -n "${AIPERF_EXCLUDE_TRACE_IDS:-}" ]]; then
            dataset_args+=(--exclude-trace-id "${AIPERF_EXCLUDE_TRACE_IDS}")
        fi
        if [[ -n "${AIPERF_MAX_CONTEXT_LENGTH:-}" ]]; then
            dataset_args+=(--max-context-length "${AIPERF_MAX_CONTEXT_LENGTH}")
        fi
        if [[ "${AIPERF_NO_FIXED_SCHEDULE:-0}" == "1" ]]; then
            dataset_args+=(--no-fixed-schedule)
        fi
        if [[ "${AIPERF_IGNORE_TRACE_DELAYS:-0}" == "1" ]]; then
            dataset_args+=(--ignore-trace-delays)
        fi
    else
        dataset_args+=(--input-file "${dataset_file}" --custom-dataset-type mooncake_trace)
        stop_args+=(--request-count "${request_count}")
    fi

    aiperf profile \
        -m "${model_name}" \
        --tokenizer "${model_name}" \
        --tokenizer-trust-remote-code \
        --url "http://${hostname}:${port}" \
        --endpoint-type chat \
        --streaming \
        --ui simple \
        "${dataset_args[@]}" \
        --artifact-dir "${log_path}/concurrency_${concurrency}" \
        --concurrency "${concurrency}" \
        --concurrency-ramp-duration 60 \
        "${stop_args[@]}" \
        --workers-max 200 \
        --request-timeout-seconds 1200 \
        --profile-export-level records \
        --extra-inputs ignore_eos:true \
        --record-processors 8

    if ! find "${log_path}/concurrency_${concurrency}" -type f \
        -name profile_export_aiperf.json -size +0c -print -quit | grep -q .; then
        echo "AIPerf did not produce profile_export_aiperf.json" >&2
        exit 1
    fi


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
