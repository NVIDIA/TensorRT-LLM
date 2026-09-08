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

# TODO(dlfw-26.08): drop once the nested-spawn failure below is settled.
# Under Open MPI 5 the DVM that rank 0's MPI_Comm_spawn starts refuses to fork
# as root, and PMIx rejects a hostname of 31 characters or more. Both are meant
# to be handled by environment variables set outside this script, and this dump
# is here to record whether they actually reached the process that needs them.
echo "----- MPI environment as seen by this shell -----"
env | grep -E '^(PRTE_|OMPI_|PMIX_|PMI_)' | sort || echo "(none set)"
echo "hostname: $(hostname) (${#HOSTNAME} chars)"
echo "-------------------------------------------------"

# Add timeout to prevent infinite hanging
timeout "$timeout_seconds" mpirun --allow-run-as-root -np 2 trtllm-llmapi-launch python3 _run_mpi_comm_task.py --task_type "$task"

echo "Remote MPI session test completed"
