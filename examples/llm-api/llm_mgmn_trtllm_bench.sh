#!/bin/bash
#SBATCH -A <account>
#SBATCH -p <partition>
#SBATCH -t 01:00:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH -o logs/trtllm-bench.out
#SBATCH -e logs/trtllm-bench.err
#SBATCH -J trtllm-bench

##############################################################################
# OVERVIEW:
# This script runs trtllm-bench throughput benchmarking on SLURM with multi-node,
# multi-GPU setup. It prepares a synthetic dataset and then benchmarks the model
# using the PyTorch backend with tensor parallelism.
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
#    - MODEL_NAME: Name of the model to benchmark
#    - EXTRA_ARGS: (Optional) Additional benchmark arguments
#
# 3. Model Configuration (lines 87-94):
#    - --tp 16: Adjust tensor parallelism size to match your node/GPU setup
#    - --num-requests (line 56): Change number of benchmark requests
#    - --input-mean/output-mean (lines 57-58): Adjust token lengths
#
# EXAMPLE USAGE:
#   export CONTAINER_IMAGE="nvcr.io/nvidia/tensorrt_llm:latest"
#   export LOCAL_MODEL="/path/to/llama-model"
#   export MODEL_NAME="meta-llama/Llama-2-7b-hf"
#   sbatch llm_mgmn_trtllm_bench.sh
##############################################################################

### :title Run trtllm-bench with pytorch backend on Slurm
### :order 1
### :section Slurm

# NOTE, this feature is experimental and may not work on all systems.
# The trtllm-llmapi-launch is a script that launches the LLM-API code on
# Slurm-like systems, and can support multi-node and multi-GPU setups.

# IMPORTANT: Total MPI processes (nodes × ntasks-per-node) must equal tensor_parallel_size.
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
        trtllm-bench --model=$LOCAL_MODEL prepare-dataset \
            --output $data_path \
            token-norm-dist \
            --num-requests=100 \
            --input-mean=128 \
            --output-mean=128 \
            --input-stdev=0 \
            --output-stdev=0
    "

echo "Running benchmark..."
# Just launch trtllm-bench job with trtllm-llmapi-launch command.

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

        # This is optional
        cat > /tmp/pytorch_extra_args.txt << EOF
cuda_graph_config: null
print_iter_log: true
enable_attention_dp: false
EOF

        # launch the benchmark
        trtllm-llmapi-launch \
         trtllm-bench \
            --model $MODEL_NAME \
            --model_path $LOCAL_MODEL \
            throughput \
            --dataset $data_path \
            --backend pytorch \
            --tp 16 \
            --config /tmp/pytorch_extra_args.txt \
            $EXTRA_ARGS
    "
