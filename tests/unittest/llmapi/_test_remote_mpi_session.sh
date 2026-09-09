#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -ex

task=$1

echo "Starting remote MPI session test with task: $task"
echo "MPI processes: 2"

timeout_seconds=60
if [ "$task" = "flashinfer_workspace" ]; then
    timeout_seconds=180
fi

# Add timeout to prevent infinite hanging
timeout "$timeout_seconds" mpirun --allow-run-as-root -np 2 trtllm-llmapi-launch python3 _run_mpi_comm_task.py --task_type "$task"

echo "Remote MPI session test completed"
