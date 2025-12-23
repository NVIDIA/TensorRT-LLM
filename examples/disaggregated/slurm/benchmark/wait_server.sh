#!/bin/bash
set -euo pipefail

# Parse arguments
hostname=$1
port=$2

# Constants for health check
readonly TIMEOUT=1800  # 30 minutes
readonly HEALTH_CHECK_INTERVAL=10
readonly STATUS_UPDATE_INTERVAL=30


# Wait for server to be healthy
echo "Waiting for server ${hostname}:${port} to be healthy..."
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
