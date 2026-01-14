#!/bin/bash
set -u
set -e
set -x

config_file=$1
server_env_var=$2

# Export server environment variables from config
for env_var in ${server_env_var}; do
    export "${env_var}"
    echo "Exported: ${env_var}"
done

trtllm-serve disaggregated -c ${config_file} -t 7200 -r 7200
