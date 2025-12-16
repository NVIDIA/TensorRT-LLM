#!/bin/bash
set -u
set -e
set -x

num_ctx_servers=$1
num_gen_servers=$2
work_dir=$3
script_dir=$4
server_env_var=$5

python3 ${script_dir}/gen_server_config.py \
    --num_ctx_servers ${num_ctx_servers} \
    --num_gen_servers ${num_gen_servers} \
    --work_dir ${work_dir}
echo "Server config generated to ${work_dir}/server_config.yaml"

# Export server environment variables from config
for env_var in ${server_env_var}; do
    export "${env_var}"
    echo "Exported: ${env_var}"
done

trtllm-serve disaggregated -c ${work_dir}/server_config.yaml -t 7200 -r 7200
