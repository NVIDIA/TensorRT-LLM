#!/bin/bash
#SBATCH -A <account>
#SBATCH -p <partition>
#SBATCH -t 01:00:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH -o logs/trtllm-bench.out
#SBATCH -e logs/trtllm-bench.err
#SBATCH -J trtllm-bench

### Run trtllm-bench on Slurm

# NOTE, this feature is experimental and may not work on all systems.
# The trtllm-llmapi-launch is a script that launches the LLM-API code on
# Slurm-like systems, and can support multi-node and multi-GPU setups.

# Note that, the number of MPI processes should be the same as the model world
# size. e.g. For tensor_parallel_size=16, you may use 2 nodes with 8 gpus for
# each, or 4 nodes with 4 gpus for each or other combinations.

# This docker image should have tensorrt_llm installed, or you need to install
# it in the task.
container_image=<docker_image>
mount_dir=<mount_dir>
mount_dest=<mount_dest>
workdir=<workdir>

# Just launch trtllm-bench job with trtllm-llmapi-launch command.
srun -l \
    --container-image=${container_image} \
    --container-mounts=${mount_dir}:${mount_dest} \
    --container-workdir=${workdir} \
    --export=ALL,PYTHONPATH=${TEKIT_ROOT} \
    --mpi=pmix \
    bash -c "
        trtllm-llmapi-launch \
         trtllm-bench \
            --model <model_name> \
            --model_path <model_path> \
            throughput \
            --dataset <dataset_path> \
            <remaining_options>
    "
