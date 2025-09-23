#!/bin/bash
set -euo pipefail

# Configuration
slurm_file="disaggr_torch.slurm"

# SLURM Configuration
partition="<partition>"
account="<account>"
job_time="02:00:00"
job_name="<job_name>"

##############################################################
# User Configuration - Review and edit the following variables

numa_bind=true
benchmark_mode="e2e" # e2e or gen_only

# Hardware Configuration
gpus_per_node=4  # Modify this with your hardware configuration

# Benchmark Configuration
use_nv_sa_benchmark=false   # Whether to use NVIDIA SA benchmark script instead of default one
isl=1024                   # Input sequence length
osl=1024                   # Output sequence length
multi_round=8              # Number of benchmark rounds
benchmark_ratio=0.8        # Benchmark ratio
streaming=true             # Enable streaming mode
cache_max_tokens=4608     # Cache transceiver max tokens
seq_offset=203  # Offset added to sequence lengths
# Dataset file for benchmarking
dataset_file="<dataset_file>"

# Environment Configuration
# Directories mount to the container
container_mount="<container_mount>" # path1:path1,path2:path2
# Container image
container_image="<container_image>"
# Path to the model directory
model_path="<model_path>"
# Path to the TensorRT-LLM repository
trtllm_repo="<trtllm_repo>"
# Set to true to do a clean build of TensorRT-LLM from source
build_wheel=false

# Workspace Configuration
work_dir=$(pwd) # path to the work directory containing the scripts

# Profiling Configuration
nsys_on=false  # Set to true to enable profiling

##############################################################

# Check if SLURM file exists
if [[ ! -f "${slurm_file}" ]]; then
    echo "Error: SLURM script '${slurm_file}' not found" >&2
    exit 1
fi

# Validate required paths
[[ ! -d "${model_path}" ]] && { echo "Error: model_path not found: ${model_path}" >&2; exit 1; }
[[ ! -d "${work_dir}" ]] && { echo "Error: work_dir '${work_dir}' not found" >&2; exit 1; }
[[ ! -f "${dataset_file}" ]] && { echo "Error: dataset_file '${dataset_file}' not found" >&2; exit 1; }

# Calculate required nodes based on tensor parallel size and server count
calc_nodes() {
    local tp_size=$1
    local num_servers=$2
    echo $(( (tp_size + gpus_per_node - 1) / gpus_per_node * num_servers ))
}

# Submit a single benchmark job
run_single() {
    # Context server params
    local ctx_num=$1
    local ctx_tp_size=$2
    local ctx_pp_size=$3
    local ctx_batch_size=$4
    local ctx_max_num_tokens=$5
    local ctx_enable_attention_dp=$6
    local ctx_gpu_frac=$7
    # Generation server params
    local gen_num=$8
    local gen_tp_size=$9
    local gen_pp_size=${10}
    local gen_batch_size=${11}
    local gen_max_num_tokens=${12}
    local gen_enable_attention_dp=${13}
    local gen_gpu_frac=${14}
    local gen_eplb_num_slots=${15}
    local mtp_size=${16}
    local gen_concurrency_list=${17}

    # Calculate total nodes needed
    local gen_nodes=$(calc_nodes "$gen_tp_size" "$gen_num")
    local ctx_nodes=$(calc_nodes "$ctx_tp_size" "$ctx_num")
    local total_nodes=$((ctx_nodes + gen_nodes))
    local total_tasks=$((total_nodes * gpus_per_node))

    # Handle SLURM reservation if needed
    local reservation_str=""
    [[ $gen_eplb_num_slots -gt 0 ]] && reservation_str="--reservation=sla_res_fw_11"

    # Submit job
    set -x
    sbatch \
        --partition="${partition}" \
        --gres=gpu:${gpus_per_node} \
        --account="${account}" \
        --time="${job_time}" \
        --job-name="${job_name}" \
        --nodes="${total_nodes}" \
        --ntasks="${total_tasks}" \
        --ntasks-per-node="${gpus_per_node}" \
        --segment="${total_nodes}" \
        ${reservation_str} \
        "${slurm_file}" \
        "${ctx_num}" "${ctx_tp_size}" "${ctx_pp_size}" "${ctx_batch_size}" "${ctx_max_num_tokens}" "${ctx_enable_attention_dp}" "${ctx_gpu_frac}" \
        "${gen_num}" "${gen_tp_size}" "${gen_pp_size}" "${gen_batch_size}" "${gen_max_num_tokens}" "${gen_enable_attention_dp}" "${gen_gpu_frac}" \
        "${gen_eplb_num_slots}" "${mtp_size}" "${gen_concurrency_list}" \
        "${gpus_per_node}" "${use_nv_sa_benchmark}" "${isl}" "${osl}" "${multi_round}" "${benchmark_ratio}" \
        "${streaming}" "${cache_max_tokens}" "${dataset_file}" "${container_mount}" "${container_image}" \
        "${model_path}" "${trtllm_repo}" "${build_wheel}" "${work_dir}" "${nsys_on}" "${seq_offset}" "${numa_bind}" "${benchmark_mode}"
    set +x
}

# Example benchmark configuration
#          |------------------- context -----------------|  |---------------------- generation ----------------------|
#           num  tp  pp  batch  tokens  attn_dp  gpu_frac    num  tp  pp  batch  tokens  attn_dp  gpu_frac  eplb  mtp  concurrency
# 1k-1k
run_single  1    4   1   4      4608    true     0.85        4    8   1   32     128     false    "0.9"     0    3    "1 2 4 8 16 36"
run_single  1    4   1   4      4608    true     0.85        1    16  1   64     256     true     "0.7"     0    3    "512 1075"
run_single  2    4   1   4      4608    true     0.85        1    16  1   128    256     true     "0.7"     0    1    "2150"
run_single  1    4   1   4      4608    true     0.85        1    32  1   16     64      true     "0.6"     0    3    "512"
run_single  1    4   1   4      4608    true     0.85        1    8   1   256    512     true     "0.8"     0    1    "2252"
run_single  1    4   1   4      4608    true     0.85        4    8   1   128    128     false    "0.9"     0    0    "1 2 4 8 16 32 64 141"
run_single  1    4   1   4      4608    true     0.85        1    32  1   32     32      true     "0.7"     0    0    "1075"
run_single  1    4   1   4      4608    true     0.85        1    16  1   64     64      true     "0.75"    0    0    "1075"
run_single  2    4   1   4      4608    true     0.85        1    16  1   256    256     true     "0.75"    0    0    "2048 4300"
run_single  1    4   1   4      4608    true     0.85        1    8   1   512    512     true     "0.8"     0    0    "4300"

# # 8k-1k, please also modify the isl,osl and cache_max_tokens above.
# run_single  1    4   1   4      8448    true     0.75        3    8   1   32     32      false    "0.9"     0    0    "1 2 4 8 16 34"
# run_single  4    4   1   4      8448    true     0.75        1    32  1   16     16      true     "0.7"     0    0    "256 538"
# run_single  7    4   1   4      8448    true     0.75        1    32  1   32     32      true     "0.7"     0    0    "1075"
# run_single  6    4   1   4      8448    true     0.75        1    16  1   64     64      true     "0.75"    0    0    "1075"
# run_single  8    4   1   4      8448    true     0.75        1    16  1   128    128     true     "0.75"    0    0    "2150"
# run_single  5    4   1   4      8448    true     0.75        1    8   1   256    256     true     "0.8"     0    0    "2150"
# run_single  1    4   1   4      8448    true     0.75        3    8   1   16     64      false    "0.9"     0    3    "1 2 4 8 18"
# run_single  5    4   1   4      8448    true     0.75        1    32  1   8      32      true     "0.7"     0    3    "128 269"
# run_single  8    4   1   4      8448    true     0.75        1    32  1   16     64      true     "0.7"     0    3    "538"
# run_single  6    4   1   4      8448    true     0.75        1    16  1   32     128     true     "0.75"    0    3    "538"
# run_single  8    4   1   4      8448    true     0.75        1    16  1   64     256     true     "0.75"    0    2    "1075"
# run_single  5    4   1   4      8448    true     0.75        1    8   1   128    256     true     "0.8"     0    1    "1075"
# run_single  6    4   1   4      8448    true     0.75        1    8   1   256    512     true     "0.8"     0    1    "2150"
