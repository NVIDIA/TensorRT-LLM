#!/bin/bash

# Add error handling
set -e
set -u
trap 'echo "Error occurred at line $LINENO"; exit 1' ERR

# Add parameter validation
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

echo "Hostname: ${hostname}, Port: ${port}"
echo "Starting benchmark..."
for concurrency in ${concurrency_list}; do
    concurrency=$((concurrency * num_gen_servers))
    num_prompts=$((concurrency * multi_round))
    echo "Benchmarking with concurrency ${concurrency} ... ${num_prompts} prompts"
    mkdir -p ${log_path}/concurrency_${concurrency}
    do_process_all_logs ${log_path}/ ${log_path}/concurrency_${concurrency} "line"
    python -m tensorrt_llm.serve.scripts.benchmark_serving \
        --model ${model_name} \
        --backend openai \
        --host ${hostname} \
        --port ${port} \
        --dataset-name "trtllm_custom" \
        --dataset-path ${dataset_file} \
        --num-prompts ${num_prompts} \
        --max-concurrency ${concurrency} \
        --trust-remote-code \
        --ignore-eos \
        --no-test-input \
        --save-result \
        --result-dir "${log_path}/concurrency_${concurrency}" \
        --result-filename "result.json" \
        --percentile-metrics "ttft,tpot,itl,e2el" \
        $(if [ "${streaming}" = "false" ]; then echo "--non-streaming"; fi)
    echo "Benchmark with concurrency ${concurrency} done"
    do_process_all_logs ${log_path}/ ${log_path}/concurrency_${concurrency} "log"
done
# do_process_all_logs ${log_path}/ ${log_path}/concurrency_${concurrency} "clean"
