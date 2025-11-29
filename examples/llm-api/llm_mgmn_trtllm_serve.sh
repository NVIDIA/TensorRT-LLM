#!/bin/bash
#SBATCH -A <account>
#SBATCH -p <partition>
#SBATCH -t 01:00:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH -o logs/trtllm-serve.out
#SBATCH -e logs/trtllm-serve.err
#SBATCH -J trtllm-serve

##############################################################################
# OVERVIEW:
# This script launches trtllm-serve (OpenAI-compatible API server) on SLURM
# with multi-node, multi-GPU support. The server will start on all allocated
# nodes and serve the model with tensor parallelism.
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
#    - ADDITIONAL_OPTIONS: (Optional) Extra trtllm-serve options
#
# 3. Server Configuration (lines 51-55):
#    - --tp_size 16: Adjust tensor parallelism to match your node/GPU setup
#    - --host 0.0.0.0: Server bind address (0.0.0.0 allows external access)
#
# EXAMPLE USAGE:
#   export CONTAINER_IMAGE="nvcr.io/nvidia/tensorrt_llm:latest"
#   export LOCAL_MODEL="/path/to/llama-model"
#   sbatch llm_mgmn_trtllm_serve.sh
#
# NOTE: After the server starts, you can send requests to it using curl or
#       the OpenAI Python client. Check the output logs for the server address.
##############################################################################

### :title Run trtllm-serve with pytorch backend on Slurm
### :order 2
### :section Slurm

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

echo "Starting trtllm-serve..."
# Just launch trtllm-serve job with trtllm-llmapi-launch command.
srun -l \
    --container-image=${CONTAINER_IMAGE} \
    --container-mounts=${MOUNT_DIR}:${MOUNT_DEST} \
    --container-workdir=${WORKDIR} \
    --export=ALL,PYTHONPATH=${SOURCE_ROOT} \
    --mpi=pmix \
    bash -c "
        set -ex
        $PROLOGUE
        export PATH=$PATH:~/.local/bin

        trtllm-llmapi-launch \
         trtllm-serve $LOCAL_MODEL \
            --tp_size 16 \
            --host 0.0.0.0 \
            ${ADDITIONAL_OPTIONS}
    "
