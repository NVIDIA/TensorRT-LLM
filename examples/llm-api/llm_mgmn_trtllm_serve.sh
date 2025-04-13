#!/bin/bash
#SBATCH -A <account>
#SBATCH -p <partition>
#SBATCH -t 01:00:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH -o logs/trtllm-serve.out
#SBATCH -e logs/trtllm-serve.err
#SBATCH -J trtllm-serve

### Run trtllm-serve with pytorch backend on Slurm

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
#   SOURCE_ROOT: the path to the TensorRT-LLM source
#   PROLOGUE: the prologue to run before the script
#   LOCAL_MODEL: the local model directory to use, NOTE: downloading from HF is
#      not supported in Slurm mode, you need to download the model and put it in
#      the LOCAL_MODEL directory.

export prepare_dataset="$SOURCE_ROOT/benchmarks/cpp/prepare_dataset.py"
export data_path="$WORKDIR/token-norm-dist.txt"

echo "Preparing dataset..."
srun -l \
    -N 1 \
    -n 1 \
    --container-image=${CONTAINER_IMAGE} \
    --container-name="prepare-name" \
    --container-mounts=${MOUNT_DIR}:${MOUNT_DEST} \
    --container-workdir=${WORKDIR} \
    --export=ALL \
    --mpi=pmix \
    bash -c "
        $PROLOGUE
        python3 $prepare_dataset \
            --tokenizer=$LOCAL_MODEL \
            --stdout token-norm-dist \
            --num-requests=100 \
            --input-mean=128 \
            --output-mean=128 \
            --input-stdev=0 \
            --output-stdev=0 > $data_path
    "

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
            --backend pytorch \
            ${ADDITIONAL_OPTIONS}
    "
