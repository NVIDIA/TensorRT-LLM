#!/bin/bash

# Retry a command with a specified number of retries and interval.
# Arguments:
#   max_retries (optional): The maximum number of times to retry the command. Default: 3.
#   interval (optional): The time in seconds to wait between retries. Default: 60.
#   command: The command to run and its arguments.
# Usage:
#   retry_command [max_retries] [interval] command...
#   If only one numeric argument is provided, it is treated as max_retries.
function retry_command() {
    local max_retries=3
    local interval=60

    if [[ "$1" =~ ^[0-9]+$ ]]; then
        max_retries=$1
        shift
    fi

    if [[ "$1" =~ ^[0-9]+$ ]]; then
        interval=$1
        shift
    fi

    local cmd=("$@")

    local count=0
    local rc=0

    while [ $count -lt $max_retries ]; do
        if "${cmd[@]}"; then
            return 0
        fi
        rc=$?
        count=$((count + 1))
        echo "Command failed with exit code $rc. Attempt $count/$max_retries."
        if [ $count -lt $max_retries ]; then
            echo "Retrying in $interval seconds..."
            sleep $interval
        fi
    done

    echo "Command failed after $max_retries attempts."
    return $rc
}
