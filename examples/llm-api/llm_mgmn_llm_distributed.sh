#!/bin/bash
#SBATCH -A <account>    # parameter
#SBATCH -p <partition>  # parameter
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH -o logs/llmapi.out
#SBATCH -e logs/llmapi.err
#SBATCH -J llmapi-distributed

### Run llm_inference_distributed.py on Slurm

# NOTE, this feature is experimental and may not work on all systems.
# The trtllm-llmapi-launch is a script that launches the LLM-API code on
# Slurm-like systems, and can support multi-node and multi-GPU setups.

# Free to change the -N and --ntasks-per-node to fit your needs, here the
# example is for tensor_parallel_size=2, thus we use world-size of 2.

# Note that, the number of MPI processes should be the same as the model world
# size. e.g. For tensor_parallel_size=16, you may use 2 nodes with 8 gpus for
# each, or 4 nodes with 4 gpus for each.


# This docker image should have tensorrt_llm installed, or you need to install
# it in the task.
container_image=<docker_image>
mount_dir=<mount_dir>
mount_dest=<mount_dest>
workdir=<workdir>

# Adjust the paths to run
export script=$TEKIT_ROOT/examples/llm-api/llm_inference_distributed.py

# Just launch llm_inference_distributed.py with trtllm-llmapi-launch command.
srun -l \
    --container-image=${container_image} \
    --container-mounts=${mount_dir}:${mount_dest} \
    --container-workdir=${workdir} \
    --export=ALL \
    --mpi=pmix \
    bash -c "
        trtllm-llmapi-launch python3 $script"
