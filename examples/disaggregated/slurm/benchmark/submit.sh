#!/bin/bash
set -euo pipefail

# Configuration
SLURM_FILE="disaggr_torch.slurm"

# SLURM Configuration
PARTITION="<partition>"
ACCOUNT="<account>"
JOB_TIME="02:00:00"
JOB_NAME="<job_name>"

##############################################################
# User Configuration - Review and edit the following variables

# Hardware Configuration
export GPUS_PER_NODE=4  # Modify this with your hardware configuration

# Benchmark Configuration
export USE_NV_SA_BENCHMARK=false   # Whether to use NVIDIA SA benchmark script instead of default one
export ISL=1024                    # Input sequence length
export OSL=1024                    # Output sequence length
export MULTI_ROUND=8              # Number of benchmark rounds
export BENCHMARK_RATIO=0.8        # Benchmark ratio
export STREAMING=true             # Enable streaming mode
export CACHE_MAX_TOKENS=4608     # Cache transceiver max tokens
# Dataset file for benchmarking
export DATASET_FILE="path/to/dataset/file"

# Environment Configuration
# Directories mount to the container
export MOUNT_DIR="path/to/mount/directory"
# Container image
export CONTAINER_IMAGE=""
# Path to the model directory
export MODEL_PATH="path/to/model/directory"
# Path to the TensorRT-LLM repository
export TRTLLM_REPO="path/to/tensorrt-llm/repository"
# Set to true to do a clean build of TensorRT-LLM from source
export BUILD_WHEEL=false

# Workspace Configuration
export WORK_DIR=$(pwd) # path to the work directory containing the scripts

# Profiling Configuration
export NSYS_ON=false  # Set to true to enable profiling

##############################################################

# Check if SLURM file exists
if [[ ! -f "${SLURM_FILE}" ]]; then
    echo "Error: SLURM script '${SLURM_FILE}' not found" >&2
    exit 1
fi

# Validate required paths
[[ ! -d "${MOUNT_DIR}" ]] && { echo "Error: MOUNT_DIR '${MOUNT_DIR}' not found" >&2; exit 1; }
[[ ! -d "${MODEL_PATH}" ]] && { echo "Error: MODEL_PATH not found: ${MODEL_PATH}" >&2; exit 1; }
[[ ! -d "${WORK_DIR}" ]] && { echo "Error: WORK_DIR '${WORK_DIR}' not found" >&2; exit 1; }
[[ ! -f "${DATASET_FILE}" ]] && { echo "Error: DATASET_FILE '${DATASET_FILE}' not found" >&2; exit 1; }

# Calculate required nodes based on tensor parallel size and server count
calc_nodes() {
    local tp_size=$1
    local num_servers=$2
    echo $(( (tp_size + GPUS_PER_NODE - 1) / GPUS_PER_NODE * num_servers ))
}

# Submit a single benchmark job
run_single() {
    # Context server params
    local ctx_num=$1
    local ctx_tp_size=$2
    local ctx_batch_size=$3
    local ctx_max_num_tokens=$4
    local ctx_enable_attention_dp=$5
    local ctx_gpu_frac=$6
    # Generation server params
    local gen_num=$7
    local gen_tp_size=$8
    local gen_batch_size=$9
    local gen_max_num_tokens=${10}
    local gen_enable_attention_dp=${11}
    local gen_gpu_memory_fraction=${12}
    local gen_mtp_size=${13}
    local gen_eplb_num_slots=${14}
    local gen_concurrency_list=${15}

    # Calculate total nodes needed
    local gen_nodes=$(calc_nodes "$gen_tp_size" "$gen_num")
    local ctx_nodes=$(calc_nodes "$ctx_tp_size" "$ctx_num")
    local total_nodes=$((ctx_nodes + gen_nodes))
    local total_tasks=$((total_nodes * GPUS_PER_NODE))

    # Handle SLURM reservation if needed
    local reservation_str=""
    [[ $gen_eplb_num_slots -gt 0 ]] && reservation_str="--reservation=sla_res_fw_11"

    # Submit job
    set -x
    sbatch \
        --partition="${PARTITION}" \
        --gres=gpu:${GPUS_PER_NODE} \
        --account="${ACCOUNT}" \
        --time="${JOB_TIME}" \
        --job-name="${JOB_NAME}" \
        --nodes="${total_nodes}" \
        --ntasks="${total_tasks}" \
        --ntasks-per-node="${GPUS_PER_NODE}" \
        --segment="${total_nodes}" \
        ${reservation_str} \
        "${SLURM_FILE}" \
        "${ctx_num}" "${ctx_tp_size}" "${ctx_batch_size}" "${ctx_max_num_tokens}" "${ctx_enable_attention_dp}" "${ctx_gpu_frac}" \
        "${gen_num}" "${gen_tp_size}" "${gen_batch_size}" "${gen_max_num_tokens}" "${gen_enable_attention_dp}" \
        "${gen_gpu_memory_fraction}" "${gen_eplb_num_slots}" "${gen_mtp_size}" "${gen_concurrency_list}"
    set +x
}

# Example benchmark configuration
#      CTX: num tp_size batch tokens attn_dp gpu_frac  GEN: num tp_size batch tokens attn_dp gpu_frac mtp eplb concurrency
run_single  1   4       4     4608   true    0.85           1   8       32    128    false   "0.9"    3   0    "16"
