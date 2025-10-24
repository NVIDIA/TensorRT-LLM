#!/bin/bash
set -ex

task=$1

echo "Starting remote MPI session test with task: $task"

echo "TLLM_SPAWN_EXTRA_MAIN_PROCESS: $TLLM_SPAWN_EXTRA_MAIN_PROCESS"

# Add timeout to prevent infinite hanging
timeout 60 mpirun --allow-run-as-root -np 2 trtllm-llmapi-launch python3 _run_mpi_comm_task.py --task_type $task

echo "Remote MPI session test completed"
