#!/bin/bash
set -euo pipefail

# Parse arguments
full_logdir=${1}
accuracy_model=${2}
accuracy_tasks=${3}
model_path=${4}
model_args_extra=${5}

echo "Starting accuracy evaluation..."
echo "Log directory: ${full_logdir}"

# Parse hostname and port from server_config.yaml (server is already healthy)
config_file="${full_logdir}/server_config.yaml"

hostname=$(grep -i "hostname:" ${config_file} | awk '{print $2}')
port=$(grep -i "port:" ${config_file} | awk '{print $2}')

if [ -z "$hostname" ] || [ -z "$port" ]; then
    echo "Error: Failed to extract hostname or port from config file"
    exit 1
fi

echo "Hostname: ${hostname}, Port: ${port}"
base_url="http://${hostname}:${port}/v1/completions"
echo "Using base_url: ${base_url}"

# Install lm_eval and run evaluation
echo "Installing lm_eval[api] and running evaluation..."
pip install lm_eval[api]==0.4.8

echo "Running lm_eval with tasks: ${accuracy_tasks}..."
lm_eval --model ${accuracy_model} \
    --tasks ${accuracy_tasks} \
    --model_args model=${model_path},base_url=${base_url},${model_args_extra} \
    --trust_remote_code

echo "Accuracy evaluation completed successfully"
