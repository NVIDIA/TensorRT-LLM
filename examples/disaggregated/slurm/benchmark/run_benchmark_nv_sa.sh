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
ucx_warmup_requests=${12}

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

do_get_logs(){
    local input_file=$1
    local output_file=$2
    local mode=$3
    local start_line=$4
    # check mode is ctx or gen
    if [ "${mode}" = "ctx" ]; then
        sed -n "${start_line},\$p" ${input_file} | grep -a "'num_generation_tokens': 0" > ${output_file} || true
    elif [ "${mode}" = "gen" ]; then
        sed -n "${start_line},\$p" ${input_file} | grep -a "'num_ctx_requests': 0, 'num_ctx_tokens': 0" > ${output_file} || true
    else
        echo "Invalid mode: ${mode}"
        return 1
    fi
    return 0
}

do_process_all_logs(){
    local input_folder=$1
    local output_folder=$2
    local mode=$3
    if [ "${mode}" != "line" ] && [ "${mode}" != "log" ] && [ "${mode}" != "clean" ]; then
        echo "Invalid mode: ${mode}"
        exit 1
    fi
    local ctx_log
    local ctx_num
    local gen_log
    local gen_num
    local line_count
    local start_line
    for ctx_log in ${input_folder}/3_output_CTX_*.log; do
        if [ -f "${ctx_log}" ]; then
            ctx_num=$(basename "${ctx_log}" | sed 's/3_output_CTX_\([0-9]*\)\.log/\1/')
            if [ "${mode}" = "line" ]; then
                line_count=$(wc -l < ${ctx_log})
                echo ${line_count} > ${output_folder}/ctx_only_line_${ctx_num}.txt
            elif [ "${mode}" = "log" ]; then
                if [ ! -f "${output_folder}/ctx_only_line_${ctx_num}.txt" ]; then
                    start_line=0
                else
                    start_line=$(cat ${output_folder}/ctx_only_line_${ctx_num}.txt)
                    rm -f ${output_folder}/ctx_only_line_${ctx_num}.txt
                fi
                do_get_logs ${ctx_log} ${output_folder}/ctx_only_${ctx_num}.txt "ctx" ${start_line}
            elif [ "${mode}" = "clean" ]; then
                rm -f ${ctx_log}
            fi
        fi
    done
    # process all the gen log files in the input folder
    for gen_log in ${input_folder}/3_output_GEN_*.log; do
        if [ -f "${gen_log}" ]; then
            gen_num=$(basename "${gen_log}" | sed 's/3_output_GEN_\([0-9]*\)\.log/\1/')
            if [ "${mode}" = "line" ]; then
                line_count=$(wc -l < ${gen_log})
                echo ${line_count} > ${output_folder}/gen_only_line_${gen_num}.txt
            elif [ "${mode}" = "log" ]; then
                if [ ! -f "${output_folder}/gen_only_line_${gen_num}.txt" ]; then
                    start_line=0
                else
                    start_line=$(cat ${output_folder}/gen_only_line_${gen_num}.txt)
                    rm -f ${output_folder}/gen_only_line_${gen_num}.txt
                fi
                do_get_logs ${gen_log} ${output_folder}/gen_only_${gen_num}.txt "gen" ${start_line}
            elif [ "${mode}" = "clean" ]; then
                rm -f ${gen_log}
            fi
        fi
    done
    if [ "${mode}" = "clean" ]; then
        if [ -d "${tmp_start_logs}" ]; then
            mkdir -p ${log_path}/start_logs
            cp ${tmp_start_logs}/3_output_CTX_*.log ${log_path}/start_logs/ 2>/dev/null || true
            cp ${tmp_start_logs}/3_output_GEN_*.log ${log_path}/start_logs/ 2>/dev/null || true
            rm -rf ${tmp_start_logs}
        fi
    fi
}

tmp_start_logs=/tmp/${SLURM_JOB_ID}/start_logs
mkdir -p ${tmp_start_logs}
cp ${log_path}/3_output_CTX_*.log ${tmp_start_logs}/ 2>/dev/null || true
cp ${log_path}/3_output_GEN_*.log ${tmp_start_logs}/ 2>/dev/null || true

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
    do_process_all_logs ${log_path}/ ${log_path}/concurrency_${concurrency} "line"

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
    do_process_all_logs ${log_path}/ ${log_path}/concurrency_${concurrency} "log"
done
# do_process_all_logs ${log_path}/ ${log_path}/concurrency_${concurrency} "clean"
