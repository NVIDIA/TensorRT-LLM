#!/bin/bash

# !!!
# Make sure that SLURM parameters are correctly set in `disaggr_torch.slurm` before executing this script.
# !!!

# concurrency 8
concurrency=8
ctx_num=1
total_node_num=8
ntasks_per_node=4 # 4 GPUs per GB200 node
ntasks=$((total_node_num * ntasks_per_node))

# `--segment` makes sure that all nodes are in the same NVLink domain
# disaggr_torch.slurm arguments:
#   num_ctx_servers=$1
#   ctx_tp_size=$2
#   ctx_batch_size=$3
#   ctx_max_num_tokens=$4
#   ctx_enable_attention_dp=$5
#   num_gen_servers=$6
#   gen_tp_size=$7
#   gen_batch_size=$8
#   gen_max_num_tokens=$9
#   gen_enable_attention_dp=${10}
#   gen_gpu_memory_fraction=${11}
#   eplb_num_slots=${12}
#   mtp_size=${13}
#   concurrency=${14}

# This command starts a job with 8 nodes, 32 GPUs in total.
# The server will include 4 context workers with DEP4, and 1 generation worker with DEP8.
sbatch --nodes=${total_node_num} \
    --ntasks=${ntasks} \
    --ntasks-per-node=${ntasks_per_node} \
    --gres=gpu:${ntasks_per_node} \
    --segment=${total_node_num} \
    disaggr_torch.slurm \
        ${ctx_num} 4 4 4480 true 1 8 1024 1024 true "0.8" 0 0 "$concurrency"
