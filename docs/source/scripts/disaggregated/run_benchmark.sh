#!/bin/bash

set -e
set -u
trap 'echo "Error occurred at line $LINENO"; exit 1' ERR

if [ "$#" -lt 7 ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 isl osl multi_round model_name concurrency_list streaming log_path"
    exit 1
fi

isl=$1
osl=$2
multi_round=$3
model_name=$4
concurrency_list=$5
streaming=$6
log_path=$7

set -x
config_file=${log_path}/config.yaml

# check if the config file exists every 10 seconds timeout 1800 seconds
timeout=1800
start_time=$(date +%s)
while [ ! -f ${config_file} ]; do
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    if [ $elapsed -ge $timeout ]; then
        echo "Error: Config file ${config_file} not found within ${timeout} seconds"
        exit 1
    fi
    if [ $((elapsed % 30)) -eq 0 ]; then
        echo "Waiting for config file... (${elapsed}s elapsed)"
    fi
    sleep 10
done

# grep the host and port from the config file
hostname=$(grep -i "hostname:" ${config_file} | awk '{print $2}')
port=$(grep -i "port:" ${config_file} | awk '{print $2}')
if [ -z "$hostname" ] || [ -z "$port" ]; then
    echo "Error: Failed to extract hostname or port from config file"
    exit 1
fi
echo "Hostname: ${hostname}, Port: ${port}"

# check server is health by curl every 10 seconds timeout 1800 seconds
timeout=1800
start_time=$(date +%s)
while ! curl -s -o /dev/null -w "%{http_code}" http://${hostname}:${port}/health; do
    hostname=$(grep -i "hostname:" ${config_file} | awk '{print $2}')
    port=$(grep -i "port:" ${config_file} | awk '{print $2}')
    echo "Hostname: ${hostname}, Port: ${port}"
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

# run the benchmark
for concurrency in ${concurrency_list}; do
    mkdir -p ${log_path}/concurrency_${concurrency}
    max_count=$((${concurrency} * ${multi_round}))
    echo "Running benchmark with concurrency: ${concurrency}, max_count: ${max_count}"
    # run your benchmark here
    python run_benchmark.py --model_name ${model_name} \
            --isl ${isl} \
            --osl ${osl} \
            --concurrency ${concurrency} \
            --max_count ${max_count} \
            --log_path ${log_path}/concurrency_${concurrency}
    echo "done for ${concurrency} in folder ${log_path}/concurrency_${concurrency}"
done

echo "Benchmark done, gracefully shutting down server and workers..."
pkill -f "start_worker.sh" || true
pkill -f "trtllm-serve" || true
sleep 20  #

if pgrep -f "trtllm-serve"; then
    echo "Warning: Some processes may still be running"
else
    echo "All processes successfully terminated"
fi
