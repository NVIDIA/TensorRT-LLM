#!/bin/bash

partition=<partition>
account=<account>
job_name=<job_name>
container_image=<container_image>
mounts=<mounts>  # e.g. /mnt/data:/mnt/data
workdir=<workdir>  # Path to disaggr_torch.slurm
model_dir=<model_dir>  # Path to the model checkpoint
repo_dir=<repo_dir>  # Path to the repo to install TensorRT-LLM, if this is empty, the pre-installed version will be used

ntasks_per_node=4 # 4 GPUs per GB200 node
total_node_num=8
ntasks=$((total_node_num * ntasks_per_node))

concurrency=8
isl=1024
osl=1024
multi_round=10
streaming=true
benchmark_mode=e2e

args=(
    1 4 1 4 4480 true "0.75"   # Context servers arguments
    1 8 1 1024 1024 true "0.8" # Generation servers arguments
    0 0                        # Other arguments
    $concurrency               # Benchmarking arguments
    $isl
    $osl
    $multi_round
    $streaming
    $container_image           # User specific arguments
    $mounts
    $workdir
    $model_dir
    $benchmark_mode
    $repo_dir
)

# This command starts a job with 8 nodes, 32 GPUs in total.
# The server will include 4 context workers with TP=4, PP=1, and 1 generation worker with TP=8, PP=1.
# `--segment` makes sure that all nodes are in the same NVLink domain
sbatch --nodes=${total_node_num} \
    --ntasks=${ntasks} \
    --ntasks-per-node=${ntasks_per_node} \
    --partition=${partition} \
    --account=${account} \
    --job-name=${job_name} \
    --gres=gpu:${ntasks_per_node} \
    --segment=${total_node_num} \
    ${workdir}/disaggr_torch.slurm "${args[@]}"
