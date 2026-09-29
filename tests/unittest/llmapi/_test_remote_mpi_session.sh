#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -ex

task=$1

echo "Starting remote MPI session test with task: $task"
echo "MPI processes: 2"

timeout_seconds=60

# mpirun does not forward the environment it was given to launched ranks, so
# the ALLOW_RUN_AS_ROOT pair Open MPI 5's PRRTE needs has to be named
# explicitly (--allow-run-as-root below only covers mpirun's own launch of
# the 2 ranks). The caller (jenkins/L0_Test.groovy) sets them; -x is what
# carries them across. Keep it in sync if that list changes.
# Shutdown knobs a test may set (the "hang" task shortens them so a wedged
# worker world is aborted within seconds); forwarded only when set so mpirun
# does not warn about missing variables.
extra_env_args=()
for var in TLLM_MGMN_SHUTDOWN_GRACE_SECONDS TLLM_LLMAPI_LAUNCH_STOP_TIMEOUT TLLM_LOG_LEVEL; do
    if [ -n "${!var+x}" ]; then
        extra_env_args+=(-x "$var")
    fi
done

timeout "$timeout_seconds" mpirun --allow-run-as-root \
    -x OMPI_ALLOW_RUN_AS_ROOT -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM \
    -x PRTE_ALLOW_RUN_AS_ROOT -x PRTE_ALLOW_RUN_AS_ROOT_CONFIRM \
    "${extra_env_args[@]}" \
    -np 2 trtllm-llmapi-launch python3 _run_mpi_comm_task.py --task_type "$task"

echo "Remote MPI session test completed"
