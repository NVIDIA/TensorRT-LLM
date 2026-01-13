#!/bin/bash
#SBATCH -A <account>    # parameter
#SBATCH -p <partition>  # parameter
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH -o logs/llmapi-distributed.out
#SBATCH -e logs/llmapi-distributed.err
#SBATCH -J llmapi-distributed-task

##############################################################################
# OVERVIEW:
# This script demonstrates running a custom LLM API Python script on SLURM
# with distributed inference support. It executes quickstart_advanced.py with
# tensor parallelism across multiple GPUs/nodes.
#
# WHAT TO MODIFY:
# 1. SLURM Parameters (lines 2-9):
#    - Replace <account> with your SLURM account name
#    - Replace <partition> with your SLURM partition name
#    - Adjust -N (number of nodes) based on your TP size
#    - Adjust --ntasks-per-node (GPUs per node) to match your setup
#
# 2. Environment Variables (set before running sbatch):
#    - CONTAINER_IMAGE: Docker image with TensorRT-LLM installed
#    - MOUNT_DIR: Host directory to mount in container
#    - MOUNT_DEST: Container mount destination path
#    - WORKDIR: Working directory inside container
#    - SOURCE_ROOT: Path to TensorRT-LLM source code
#    - PROLOGUE: Commands to run before main task (e.g., module loads)
#    - LOCAL_MODEL: Path to your pre-downloaded model directory
#
# 3. Script Configuration (lines 39, 51-54):
#    - Line 39: Change $script to point to your own Python script
#    - Line 52: Modify --model_dir to use your model path
#    - Line 53: Customize --prompt with your test prompt
#    - Line 54: Adjust --tp_size to match your node/GPU setup
#
# EXAMPLE USAGE:
#   export CONTAINER_IMAGE="nvcr.io/nvidia/tensorrt_llm:latest"
#   export LOCAL_MODEL="/path/to/llama-model"
#   sbatch llm_mgmn_llm_distributed.sh
#
# NOTE: This is a template - you can replace quickstart_advanced.py with any
#       LLM API Python script. The trtllm-llmapi-launch wrapper handles the
#       distributed execution setup automatically.
##############################################################################

### :section Slurm
### :title Run LLM-API with pytorch backend on Slurm
### :order 0

# NOTE, this feature is experimental and may not work on all systems.
# The trtllm-llmapi-launch is a script that launches the LLM-API code on
# Slurm-like systems, and can support multi-node and multi-GPU setups.

# IMPORTANT: Total MPI processes (nodes × ntasks-per-node) must equal tp_size.
# e.g. For tensor_parallel_size=16, you may use 2 nodes with 8 gpus for
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
            --tp_size 2 \
            --max_batch_size 256
    "
