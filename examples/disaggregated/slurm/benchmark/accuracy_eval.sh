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

# Parse hostname and port from server_config.yaml
config_file="${full_logdir}/server_config.yaml"

# Wait for server_config.yaml to be created
max_wait=1800
wait_count=0
while [ ! -f "${config_file}" ] && [ ${wait_count} -lt ${max_wait} ]; do
    echo "Waiting for server_config.yaml to be created..."
    sleep 1
    wait_count=$((wait_count + 1))
done

if [ ${wait_count} -ge ${max_wait} ]; then
    echo "Error: server_config.yaml not found after ${max_wait} seconds"
    exit 1
fi

# grep the host and port from the config file
hostname=$(grep -i "hostname:" ${config_file} | awk '{print $2}')
port=$(grep -i "port:" ${config_file} | awk '{print $2}')

if [ -z "$hostname" ] || [ -z "$port" ]; then
    echo "Error: Failed to extract hostname or port from config file"
    exit 1
fi

echo "Hostname: ${hostname}, Port: ${port}"
base_url="http://${hostname}:${port}/v1/completions"
echo "Using base_url: ${base_url}"

# check server is health by curl every 10 seconds timeout 1800 seconds
timeout=1800
start_time=$(date +%s)
while ! curl -s -o /dev/null -w "%{http_code}" http://${hostname}:${port}/health; do
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    if [ $elapsed -ge $timeout ]; then
        echo "Error: Server is not healthy after ${timeout} seconds"
        exit 1
    fi
    if [ $((elapsed % 30)) -eq 0 ]; then
        echo "Waiting for server to be healthy... (${elapsed}s elapsed)"
    fi
    sleep 10
done

# Install lm_eval and run evaluation
echo "Installing lm_eval[api] and running evaluation..."
pip install lm_eval[api]==0.4.8

echo "Running lm_eval with tasks: ${accuracy_tasks}..."
lm_eval --model ${accuracy_model} \
    --tasks ${accuracy_tasks} \
    --model_args model=${model_path},base_url=${base_url},${model_args_extra} \
    --trust_remote_code

echo "Accuracy evaluation completed successfully"
