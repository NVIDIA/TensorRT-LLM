#!/bin/bash

echo "Please find the \`disaggr_torch.slurm\` script in the \`examples/disaggregated/slurm/benchmark/\` directory."

partition=<partition>
account=<account>
job_name=<job_name>
container_image=<container_image>
mounts=<mounts>  # e.g. /mnt/data:/mnt/data
workdir=<workdir>  # Path to disaggr_torch.slurm
model_dir=<model_dir>  # Path to the model checkpoint
repo_dir=<repo_dir>  # Path to the repo to install TensorRT-LLM, if this is empty, the pre-installed version will be used

mtp_size=0
ntasks_per_node=4 # 4 GPUs per GB200 node, 8 GPUs per B200 node

isl=1024
osl=1024
multi_round=10
streaming=true
benchmark_mode=e2e

# dep16 eplb0, 256, 288
for b in 1 64 1024; do
    for eplb_num_slots in 0 256 288; do
        concurrency=$((b * 16))
        ctx_node_num=$(((concurrency + 5499)/5500)) # $(((concurrency + 10999)/11000)) for B200
        ctx_num=${ctx_node_num} # $((ctx_node_num * 2)) for B200
        total_node_num=$((ctx_node_num + 4)) # $((ctx_node_num + 2)) for B200
        ntasks=$((total_node_num * ntasks_per_node))

        args=(
            ${ctx_num} 4 1 4 4480 true "0.85"   # Context servers arguments
            1 16 1 1024 1024 true "0.7"       # Generation servers arguments
            $eplb_num_slots $mtp_size  # Other arguments
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

        sbatch --nodes=${total_node_num} \
            --ntasks=${ntasks} \
            --ntasks-per-node=${ntasks_per_node} \
            --partition=${partition} \
            --account=${account} \
            --job-name=${job_name} \
            --gres=gpu:${ntasks_per_node} \
            --segment=${total_node_num} \
            ${workdir}/disaggr_torch.slurm "${args[@]}"
    done
done

# dep32 eplb288
for b in 512; do
    concurrency=$((b * 32))
    ctx_node_num=$(((concurrency + 5499)/5500)) # $(((concurrency + 10999)/11000)) for B200
    ctx_num=${ctx_node_num} # $((ctx_node_num * 2)) for B200
    total_node_num=$((ctx_node_num + 8)) # $((ctx_node_num + 4)) for B200
    ntasks=$((total_node_num * ntasks_per_node))
    eplb_num_slots=288

    args=(
        ${ctx_num} 4 1 4 4480 true "0.85"   # Context servers arguments
        1 32 1 1024 1024 true "0.7"  # Generation servers arguments
        $eplb_num_slots $mtp_size  # Other arguments
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

    sbatch --nodes=${total_node_num} \
        --ntasks=${ntasks} \
        --ntasks-per-node=${ntasks_per_node} \
        --partition=${partition} \
        --account=${account} \
        --job-name=${job_name} \
        --gres=gpu:${ntasks_per_node} \
        --segment=${total_node_num} \
        ${workdir}/disaggr_torch.slurm "${args[@]}"
done
