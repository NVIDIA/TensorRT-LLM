#!/bin/bash
set -u
set -e
set -x

num_ctx_servers=$1
num_gen_servers=$2
work_dir=$3
script_dir=$4
enable_time_breakdown=$5
perf_metrics_max_requests=$6

python3 ${script_dir}/gen_server_config.py \
    --num_ctx_servers ${num_ctx_servers} \
    --num_gen_servers ${num_gen_servers} \
    --work_dir ${work_dir} \
    $(if [ "${enable_time_breakdown}" = "true" ]; then echo "--enable_time_breakdown"; fi) \
    --perf_metrics_max_requests ${perf_metrics_max_requests}
echo "server config generated to ${work_dir}/server_config.yaml"

trtllm-serve disaggregated -c ${work_dir}/server_config.yaml -t 7200 -r 7200
