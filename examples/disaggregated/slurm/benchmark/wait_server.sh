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

set -euo pipefail

# Parse arguments
hostname=$1
port=$2
timeout_s=${3:-1800}

if [[ ! "${timeout_s}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: timeout must be a positive integer in seconds" >&2
    exit 2
fi

# Constants for health check
readonly TIMEOUT="${timeout_s}"
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
