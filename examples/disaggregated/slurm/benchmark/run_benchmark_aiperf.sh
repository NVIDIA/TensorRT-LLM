#!/bin/bash

# AIPerf-based benchmark script for disaggregated serving.
# Args: model_name dataset_file multi_round num_gen_servers concurrency_list streaming log_path hostname port ucx_warmup_requests

set -euo pipefail
trap 'echo "Error occurred at line $LINENO"; exit 1' ERR

if [ "$#" -lt 10 ]; then
    echo "Error: Missing required arguments, got $# arguments, args: $*"
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

strip_outer_quotes() {
    local value="$1"
    if [[ "${value}" == \'*\' ]] && [[ "${value}" == *\' ]]; then
        value="${value:1:${#value}-2}"
    fi
    if [[ "${value}" == \"*\" ]] && [[ "${value}" == *\" ]]; then
        value="${value:1:${#value}-2}"
    fi
    printf "%s" "${value}"
}

for env_name in HF_HOME XDG_CACHE_HOME PIP_CACHE_DIR AIPERF_SOURCE_DIR AIPERF_CUSTOM_DATASET_TYPE AIPERF_BENCHMARK_DURATION_SECONDS AIPERF_MAX_CONTEXT_LENGTH AIPERF_TOKENIZER AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT AIPERF_DATASET_CONFIGURATION_TIMEOUT AIPERF_TRACE_SPLIT_DIR AIPERF_ARTIFACT_BASE_DIR TMPDIR HF_MODULES_CACHE; do
    if [ -n "${!env_name:-}" ]; then
        printf -v "${env_name}" "%s" "$(strip_outer_quotes "${!env_name}")"
        export "${env_name}"
    fi
done

export HF_HOME=${HF_HOME:-/lustre/fsw/coreai_comparch_trtllm/lizhiz/.cache/huggingface}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${log_path}/cache/xdg}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-${log_path}/cache/pip}
export TMPDIR=${TMPDIR:-${log_path}/cache/tmp}
export HF_MODULES_CACHE=${HF_MODULES_CACHE:-${log_path}/cache/hf_modules}
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=${AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT:-900}
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=${AIPERF_DATASET_CONFIGURATION_TIMEOUT:-900}
export HF_HUB_TRUST_REMOTE_CODE=1
trace_split_root="${AIPERF_TRACE_SPLIT_DIR:-${log_path}/weka_trace_json}"
artifact_base_dir="${AIPERF_ARTIFACT_BASE_DIR:-${log_path}}"
mkdir -p "${HF_HOME}" "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}" "${TMPDIR}" "${HF_MODULES_CACHE}" "${trace_split_root}" "${artifact_base_dir}"

install_aiperf() {
    local source_dir="${AIPERF_SOURCE_DIR:-/lustre/fsw/coreai_comparch_trtllm/lizhiz/offload-sa-k25/aiperf}"

    if [ -d "${source_dir}/src/aiperf" ]; then
        echo "Installing AIPerf from local source: ${source_dir}"
        python -m pip install --force-reinstall --no-deps -e "${source_dir}"
    else
        echo "Local AIPerf source not found at ${source_dir}; installing Cam's AgentX branch"
        python -m pip install --force-reinstall --no-deps \
            'aiperf @ git+https://github.com/cquil11/aiperf.git@cjq/agentx-v0.3-subagents'
    fi
}

prepare_dataset() {
    dataset_type="${AIPERF_CUSTOM_DATASET_TYPE:-}"
    if [ -z "${dataset_type}" ]; then
        case "${dataset_file}" in
            *.jsonl) dataset_type="weka_trace" ;;
            *) dataset_type="mooncake_trace" ;;
        esac
    fi

    dataset_input="${dataset_file}"
    if [ "${dataset_type}" = "weka_trace" ] && [ -f "${dataset_file}" ] && [[ "${dataset_file}" == *.jsonl ]]; then
        dataset_input="${trace_split_root}"
        if [ ! -f "${dataset_input}/.complete" ]; then
            echo "Splitting Weka JSONL ${dataset_file} into ${dataset_input}"
            mkdir -p "${dataset_input}"
            python - "${dataset_file}" "${dataset_input}" <<'PY'
import json
import pathlib
import re
import sys

src = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
complete = out / ".complete"

if complete.exists():
    print(f"Weka JSON split already complete: {out}")
    raise SystemExit(0)

existing = list(out.glob("*.json"))
if existing:
    raise SystemExit(
        f"{out} contains JSON files but .complete is absent; use a fresh log directory"
    )

count = 0
with src.open("r", encoding="utf-8") as fh:
    for idx, line in enumerate(fh):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        trace_id = str(obj.get("id") or obj.get("trace_id") or f"trace_{idx:06d}")
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", trace_id).strip("._")
        safe = (safe or f"trace_{idx:06d}")[:180]
        with (out / f"{idx:06d}_{safe}.json").open("w", encoding="utf-8") as out_f:
            json.dump(obj, out_f, separators=(",", ":"))
        count += 1

complete.write_text(f"{count}\n", encoding="utf-8")
print(f"Wrote {count} Weka trace JSON files to {out}")
PY
        fi
    fi
}

install_aiperf
python -m pip install "crick~=0.0.8"
prepare_dataset

# warmup requests for ucx connections
if [ "${ucx_warmup_requests}" -gt 0 ]; then
    echo "warming up ucx connections with small requests... ${ucx_warmup_requests}"
    python -m tensorrt_llm.serve.scripts.benchmark_serving \
        --model "${model_name}" \
        --dataset-name random \
        --random-ids \
        --random-input-len 100 \
        --random-output-len 10 \
        --num-prompts "${ucx_warmup_requests}" \
        --host "${hostname}" \
        --port "${port}" \
        --ignore-eos \
        --trust-remote-code \
        --non-streaming
    echo "UCX warmup done"
fi

streaming_args=()
case "${streaming}" in
    true|True|TRUE|1|yes|Yes|YES) streaming_args=(--streaming) ;;
    *) streaming_args=() ;;
esac

run_weka_trace_benchmark() {
    local concurrency=$1
    local artifact_dir="${artifact_base_dir}/concurrency_${concurrency}"
    local benchmark_duration="${AIPERF_BENCHMARK_DURATION_SECONDS:-1800}"
    local tokenizer="${AIPERF_TOKENIZER:-${model_name}}"
    local max_context_length="${AIPERF_MAX_CONTEXT_LENGTH:-167000}"

    echo "Benchmarking Weka trace with concurrency ${concurrency}, duration ${benchmark_duration}s"
    mkdir -p "${artifact_dir}"

    aiperf profile \
        --scenario inferencex-agentx-mvp \
        --model "${model_name}" \
        --tokenizer "${tokenizer}" \
        --tokenizer-trust-remote-code \
        --max-context-length "${max_context_length}" \
        --endpoint-type chat \
        --url "http://${hostname}:${port}" \
        "${streaming_args[@]}" \
        --ui simple \
        --input-file "${dataset_input}" \
        --artifact-dir "${artifact_dir}" \
        --concurrency "${concurrency}" \
        --concurrency-ramp-duration "${AIPERF_CONCURRENCY_RAMP_DURATION_SECONDS:-60}" \
        --custom-dataset-type weka_trace \
        --benchmark-duration "${benchmark_duration}" \
        --benchmark-grace-period "${AIPERF_BENCHMARK_GRACE_PERIOD_SECONDS:-60}" \
        --workers-max "${AIPERF_WORKERS_MAX:-200}" \
        --request-timeout-seconds "${AIPERF_REQUEST_TIMEOUT_SECONDS:-1200}" \
        --profile-export-level records \
        --extra-inputs ignore_eos:true \
        --record-processors "${AIPERF_RECORD_PROCESSORS:-8}" \
        --use-server-token-count
}

run_trace_benchmark() {
    local concurrency=$1
    local request_count=$((concurrency * multi_round))
    local benchmark_duration=$((multi_round * 1200))
    local artifact_dir="${artifact_base_dir}/concurrency_${concurrency}"

    echo "Benchmarking ${dataset_type} with concurrency ${concurrency} ... ${request_count} requests, duration ${benchmark_duration}s"
    mkdir -p "${artifact_dir}"

    aiperf profile \
        --model "${model_name}" \
        --tokenizer "${model_name}" \
        --tokenizer-trust-remote-code \
        --url "http://${hostname}:${port}" \
        "${streaming_args[@]}" \
        --ui simple \
        --input-file "${dataset_input}" \
        --artifact-dir "${artifact_dir}" \
        --concurrency "${concurrency}" \
        --concurrency-ramp-duration 60 \
        --custom-dataset-type "${dataset_type}" \
        --benchmark-duration "${benchmark_duration}" \
        --benchmark-grace-period 60 \
        --workers-max 200 \
        --request-timeout-seconds 1200 \
        --profile-export-level records \
        --extra-inputs ignore_eos:true \
        --request-count "${request_count}" \
        --record-processors 8
}

echo "Hostname: ${hostname}, Port: ${port}"
echo "AIPerf trace split dir: ${trace_split_root}"
echo "AIPerf artifact base dir: ${artifact_base_dir}"
echo "Starting aiperf benchmark with dataset_type=${dataset_type}, dataset_input=${dataset_input}"

concurrency_list=$(echo "${concurrency_list}" | tr ',' ' ')
for concurrency in ${concurrency_list}; do
    concurrency=$((concurrency))
    if [ "${dataset_type}" = "weka_trace" ]; then
        run_weka_trace_benchmark "${concurrency}"
    else
        run_trace_benchmark "${concurrency}"
    fi
    echo "Benchmark with concurrency ${concurrency} done"
done

# Fetch perf metrics from disagg server
echo "Fetching perf metrics from http://${hostname}:${port}/perf_metrics ..."
curl -s "http://${hostname}:${port}/perf_metrics" > "${log_path}/perf_metrics.json" 2>&1 || true
if [ -s "${log_path}/perf_metrics.json" ]; then
    echo "Perf metrics saved to ${log_path}/perf_metrics.json"
else
    echo "Warning: perf_metrics response was empty or endpoint not available"
fi
