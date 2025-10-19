#!/bin/bash
#SBATCH -A <account>    # parameter
#SBATCH -p <partition>  # parameter
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH -o logs/llmapi-distributed.out
#SBATCH -e logs/llmapi-distributed.err
#SBATCH -J llmapi-distributed-task

### :section Slurm
### :title Run LLM-API with pytorch backend on Slurm
### :order 0

# NOTE, this feature is experimental and may not work on all systems.
# The trtllm-llmapi-launch is a script that launches the LLM-API code on
# Slurm-like systems, and can support multi-node and multi-GPU setups.

# Note that, the number of MPI processes should be the same as the model world
# size. e.g. For tensor_parallel_size=16, you may use 2 nodes with 8 gpus for
# each, or 4 nodes with 4 gpus for each or other combinations.

# This docker image should have tensorrt_llm installed, or you need to install
# it in the task.

# The following variables are expected to be set in the environment:
# You can set them via --export in the srun/sbatch command.
#   CONTAINER_IMAGE: the docker image to use, you'd better install tensorrt_llm in it, or install it in the task.
#   MOUNT_DIR: the directory to mount in the container
#   MOUNT_DEST: the destination directory in the container
#   WORKDIR: the working directory in the container
#   SOURCE_ROOT: the path to the TensorRT LLM source
#   PROLOGUE: the prologue to run before the script
#   LOCAL_MODEL: the local model directory to use, NOTE: downloading from HF is
#      not supported in Slurm mode, you need to download the model and put it in
#      the LOCAL_MODEL directory.

# Adjust the paths to run
export script=$SOURCE_ROOT/examples/llm-api/quickstart_advanced.py

# Just launch the PyTorch example with trtllm-llmapi-launch command.
srun -l \
    --container-image=${CONTAINER_IMAGE} \
    --container-mounts=${MOUNT_DIR}:${MOUNT_DEST} \
    --container-workdir=${WORKDIR} \
    --export=ALL \
    --mpi=pmix \
    bash -c "
        $PROLOGUE
        export PATH=$PATH:~/.local/bin
        trtllm-llmapi-launch python3 $script \
            --model_dir $LOCAL_MODEL \
            --prompt 'Hello, how are you?' \
            --tp_size 2
    "
