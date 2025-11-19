#!/bin/bash
set -euo pipefail

# Parse arguments
work_dir=$1

# Constants for health check
readonly TIMEOUT=1800  # 30 minutes
readonly HEALTH_CHECK_INTERVAL=10
readonly STATUS_UPDATE_INTERVAL=30

config_file="${work_dir}/server_config.yaml"

echo "Waiting for server to be ready..."
echo "Config file: ${config_file}"

# Wait for server_config.yaml to be created
echo "Step 1: Waiting for config file..."
start_time=$(date +%s)
while [ ! -f "${config_file}" ]; do
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))

    if [ $elapsed -ge $TIMEOUT ]; then
        echo "Error: Config file ${config_file} not found within ${TIMEOUT} seconds"
        exit 1
    fi

    if [ $((elapsed % STATUS_UPDATE_INTERVAL)) -eq 0 ] && [ $elapsed -gt 0 ]; then
        echo "Waiting for config file... (${elapsed}s elapsed)"
    fi

    sleep $HEALTH_CHECK_INTERVAL
done

echo "Config file found!"

# Extract hostname and port from the config file
hostname=$(grep -i "hostname:" ${config_file} | awk '{print $2}')
port=$(grep -i "port:" ${config_file} | awk '{print $2}')

if [ -z "$hostname" ] || [ -z "$port" ]; then
    echo "Error: Failed to extract hostname or port from config file"
    exit 1
fi

echo "Server address: ${hostname}:${port}"

# Wait for server to be healthy
echo "Step 2: Waiting for server to be healthy..."
start_time=$(date +%s)
while ! curl -s -o /dev/null -w "%{http_code}" "http://${hostname}:${port}/health" > /dev/null 2>&1; do
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))

    if [ $elapsed -ge $TIMEOUT ]; then
        echo "Error: Server not healthy after ${TIMEOUT} seconds"
        exit 1
    fi

    if [ $((elapsed % STATUS_UPDATE_INTERVAL)) -eq 0 ] && [ $elapsed -gt 0 ]; then
        echo "Waiting for server to be healthy... (${elapsed}s elapsed)"
    fi

    sleep $HEALTH_CHECK_INTERVAL
done

echo "Server is healthy and ready to accept requests!"
