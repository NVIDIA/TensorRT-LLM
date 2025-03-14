#!/bin/bash
set -x
set -e

device_count=1
while getopts "c:" opt; do
    case "$opt" in
        c)
            if [ -n "$OPTARG" ]; then
                device_count=$OPTARG
            fi
            ;;
    esac
done

start_port=12478
end_port=$((start_port + device_count))

# Function to check if server is running on a specific port
check_server() {
    local port=$1
    if curl -s http://localhost:$port > /dev/null; then
        echo "Server is running on http://localhost:$port"
        return 0
    else
        return 1
    fi
}

# Kill processes for ports in the range
for ((port = start_port; port <= end_port; port++)); do
    if check_server $port; then
        kill -9 $(lsof -t -i :$port)
        echo "Close server on port: $port"
    fi
done
