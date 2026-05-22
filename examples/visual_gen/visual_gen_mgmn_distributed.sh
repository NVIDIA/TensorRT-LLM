#!/bin/bash
#SBATCH -A <account>    # parameter
#SBATCH -p <partition>  # parameter
#SBATCH -t 01:00:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -o visual-gen-distributed.out
#SBATCH -e visual-gen-distributed.err
#SBATCH -J visual-gen-distributed

##############################################################################
# OVERVIEW:
# This script demonstrates running visual generation (e.g., Wan T2V) on SLURM
# with distributed inference support across multiple GPUs and nodes using
# Ulysses sequence parallelism.
#
# WHAT TO MODIFY:
# 1. SLURM Parameters (lines above):
#    - Replace <account> with your SLURM account name
#    - Replace <partition> with your SLURM partition name
#    - Adjust -N (number of nodes) to match your parallelism config
#    - Adjust --ntasks-per-node and --gpus-per-node to match GPUs per node
#    - Total GPUs must equal: cfg_size * ulysses_size
#
# 2. Environment Variables (set before running sbatch, or edit defaults below):
#    - CONTAINER_IMAGE: Docker image or enroot .sqsh image with TensorRT-LLM installed
#    - MOUNT_DIR:       host directory to mount into the container (default: $HOME)
#    - MOUNT_DEST:      mount destination path inside the container (default: $HOME)
#    - PROJECT_DIR:     path to TensorRT-LLM source on the shared filesystem
#    - MODEL_PATH:      HuggingFace Hub model ID or local path (default: Wan-AI/Wan2.2-T2V-A14B-Diffusers)
#    - OUTPUT_PATH:        path to save generated video (default: output.avi)
#    - ATTENTION_BACKEND:  attention backend to use (default: FA4; options: VANILLA, TRTLLM, FA4)
#    - CFG_SIZE:           CFG parallel size (1 or 2)
#    - ULYSSES_SIZE:       Ulysses sequence parallel size
#    - MASTER_PORT:        NCCL rendezvous port (default: 29500)
#
# EXAMPLE USAGE:
#   export CONTAINER_IMAGE="/path/to/tensorrt-llm.sqsh"
#   export PROJECT_DIR="/path/to/TensorRT-LLM"
#   export MODEL_PATH="Wan-AI/Wan2.2-T2V-A14B-Diffusers"
#   export OUTPUT_PATH="output.avi"
#   sbatch visual_gen_mgmn_distributed.sh
#
##############################################################################

# ---------------------------------------------------------------------------
# Configuration — override any of these via environment before calling sbatch
# ---------------------------------------------------------------------------

PROJECT_DIR="${PROJECT_DIR:-/path/to/TensorRT-LLM}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/path/to/tensorrt-llm.sqsh}"
MOUNT_DIR="${MOUNT_DIR:-$HOME}"
MOUNT_DEST="${MOUNT_DEST:-$HOME}"

MODEL_PATH="${MODEL_PATH:-Wan-AI/Wan2.2-T2V-A14B-Diffusers}"
PROMPT="${PROMPT:-A cat playing piano}"
OUTPUT_PATH="${OUTPUT_PATH:-output.avi}"

# Generation parameters
HEIGHT="${HEIGHT:-720}"
WIDTH="${WIDTH:-1280}"
NUM_FRAMES="${NUM_FRAMES:-81}"
NUM_STEPS="${NUM_STEPS:-40}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-FA4}"

# Parallelism
CFG_SIZE="${CFG_SIZE:-2}"
ULYSSES_SIZE="${ULYSSES_SIZE:-2}"

# ---------------------------------------------------------------------------
# Derived values — do not edit
# ---------------------------------------------------------------------------

NUM_GPUS=$(( SLURM_NNODES * SLURM_GPUS_PER_NODE ))

# Determine rank-0 node hostname for NCCL rendezvous
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT="${MASTER_PORT:-29500}"

# ---------------------------------------------------------------------------
# Validate parallelism config against allocated GPU count
# ---------------------------------------------------------------------------

EXPECTED_GPUS=$(( CFG_SIZE * ULYSSES_SIZE ))
if [ "${NUM_GPUS}" -ne "${EXPECTED_GPUS}" ]; then
    echo "ERROR: NUM_GPUS=${NUM_GPUS} but cfg_size(${CFG_SIZE}) * ulysses_size(${ULYSSES_SIZE}) = ${EXPECTED_GPUS}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Build the run command
# ---------------------------------------------------------------------------

RUN_CMD="python examples/visual_gen/visual_gen_wan_t2v.py \
        --model_path '${MODEL_PATH}' \
        --prompt '${PROMPT}' \
        --height ${HEIGHT} --width ${WIDTH} --num_frames ${NUM_FRAMES} \
        --steps ${NUM_STEPS} \
        --attention_backend ${ATTENTION_BACKEND} \
        --cfg_size ${CFG_SIZE} \
        --ulysses_size ${ULYSSES_SIZE} \
        --output_path '${OUTPUT_PATH}'"

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

srun -l \
    --export=ALL \
    --container-image "${CONTAINER_IMAGE}" \
    --container-workdir "${PROJECT_DIR}" \
    --container-mounts=${MOUNT_DIR}:${MOUNT_DEST} \
    sh -c "${RUN_CMD}"
