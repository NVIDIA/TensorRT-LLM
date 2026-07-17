#!/bin/bash

# Retry a command with a specified number of retries and interval.
# Arguments:
#   max_retries (optional): The maximum number of times to retry the command. Default: 3.
#   interval (optional): The time in seconds to wait between retries. Default: 60.
#   --timeout <seconds> (optional): Per-attempt timeout in seconds. Default: no timeout.
#   command: The command to run and its arguments.
# Usage:
#   retry_command [max_retries] [interval] [--timeout <seconds>] command...
#   If only one numeric argument is provided, it is treated as max_retries.
function retry_command() {
    local max_retries=3
    local interval=60
    local cmd_timeout=0

    if [[ "$1" =~ ^[0-9]+$ ]]; then
        max_retries=$1
        shift
    fi

    if [[ "$1" =~ ^[0-9]+$ ]]; then
        interval=$1
        shift
    fi

    if [[ "$1" == "--timeout" ]]; then
        cmd_timeout=$2
        shift 2
    fi

    local cmd=("$@")

    local count=0
    local rc=0

    while [ $count -lt $max_retries ]; do
        if [ $cmd_timeout -gt 0 ]; then
            if timeout $cmd_timeout "${cmd[@]}"; then
                return 0
            fi
        else
            if "${cmd[@]}"; then
                return 0
            fi
        fi
        rc=$?
        count=$((count + 1))
        if [ $rc -eq 124 ] && [ $cmd_timeout -gt 0 ]; then
            echo "Command timed out after ${cmd_timeout}s. Attempt $count/$max_retries."
        else
            echo "Command failed with exit code $rc. Attempt $count/$max_retries."
        fi
        if [ $count -lt $max_retries ]; then
            echo "Retrying in $interval seconds..."
            sleep $interval
        fi
    done

    echo "Command failed after $max_retries attempts."
    return $rc
}
