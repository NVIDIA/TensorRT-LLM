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

# Rank 0 nests an MPI_Comm_spawn inside this launch, and the DVM that spawn
# starts inherits the rank's environment -- not this shell's. mpirun does not
# forward the environment it was given, so the two variables that DVM needs have
# to be named explicitly:
#
#   *_ALLOW_RUN_AS_ROOT  Open MPI 5's PRRTE refuses to fork a DVM as root without
#                        them. The --allow-run-as-root below only covers mpirun
#                        itself, not the nested spawn.
#   PMIX_HOSTNAME        PMIx rejects a hostname of 31 characters or more during
#                        the singleton handshake, and CI pod names are 63.
#
# The caller (jenkins/L0_Test.groovy) sets all of them; -x is what carries them
# across. Keep it in sync if that list changes.
timeout "$timeout_seconds" mpirun --allow-run-as-root \
    -x OMPI_ALLOW_RUN_AS_ROOT -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM \
    -x PRTE_ALLOW_RUN_AS_ROOT -x PRTE_ALLOW_RUN_AS_ROOT_CONFIRM \
    -x PMIX_HOSTNAME \
    -np 2 trtllm-llmapi-launch python3 _run_mpi_comm_task.py --task_type "$task"

echo "Remote MPI session test completed"
