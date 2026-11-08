#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Retry a command with a specified number of retries and interval.
# Arguments:
#   max_retries (optional): The maximum number of times to retry the command. Default: 3.
#   interval (optional): The time in seconds to wait between retries. Default: 60.
#   --timeout <seconds> (optional): Per-attempt timeout in seconds. Default: no timeout.
#   --retry-on <statuses> (optional): Comma-separated exit statuses to retry; a failure with any
#       other status is returned at once. Default: every failure is retried.
#   command: The command to run and its arguments.
# Usage:
#   retry_command [max_retries] [interval] [--timeout <seconds>] [--retry-on <statuses>] command...
#   If only one numeric argument is provided, it is treated as max_retries.
function retry_command() {
    local max_retries=3
    local interval=60
    local cmd_timeout=0
    local retry_on=""

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

    if [[ "$1" == "--retry-on" ]]; then
        retry_on=$2
        shift 2
    fi

    local cmd=("$@")

    local count=0
    local rc=0

    while [ $count -lt $max_retries ]; do
        if [ $cmd_timeout -gt 0 ]; then
            if timeout $cmd_timeout "${cmd[@]}"; then
                return 0
            else
                # Capture here: after `fi`, $? is the `if` statement's status.
                rc=$?
            fi
        else
            if "${cmd[@]}"; then
                return 0
            else
                # Capture here: after `fi`, $? is the `if` statement's status.
                rc=$?
            fi
        fi
        count=$((count + 1))
        if [ $rc -eq 124 ] && [ $cmd_timeout -gt 0 ]; then
            echo "Command timed out after ${cmd_timeout}s. Attempt $count/$max_retries."
        else
            echo "Command failed with exit code $rc. Attempt $count/$max_retries."
        fi
        if [ -n "$retry_on" ] && [[ ",$retry_on," != *",$rc,"* ]]; then
            echo "Exit code $rc is not retried."
            return $rc
        fi
        if [ $count -lt $max_retries ]; then
            echo "Retrying in $interval seconds..."
            sleep $interval
        fi
    done

    echo "Command failed after $max_retries attempts."
    return $rc
}
