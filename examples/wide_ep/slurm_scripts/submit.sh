#!/bin/bash

# !!!
# Please find the `disaggr_torch.slurm` script in the `examples/disaggregated/slurm/` directory.
# Make sure that SLURM parameters are correctly set in `disaggr_torch.slurm` before executing this script.
# !!!

mtp_size=0
ntasks_per_node=4 # 4 GPUs per GB200 node

# dep8
for b in 1 64 1024; do
    concurrency=$((b * 8))
    ctx_num=$(((concurrency + 5499)/5500))
    total_node_num=$((ctx_num + 2))
    ntasks=$((total_node_num * ntasks_per_node))
    sbatch --nodes=${total_node_num} --ntasks=${ntasks} --ntasks-per-node=${ntasks_per_node} --segment=${total_node_num} disaggr_torch.slurm ${ctx_num} 4 4 4480 true 1 8 1024 1024 true "0.8" 0 "$mtp_size" "$concurrency"
done

# dep16 eplb0, 256, 288
for b in 1 64 1024; do
    concurrency=$((b * 16))
    ctx_num=$(((concurrency + 5499)/5500))
    total_node_num=$((ctx_num + 4))
    ntasks=$((total_node_num * ntasks_per_node))
    sbatch --nodes=${total_node_num} --ntasks=${ntasks} --ntasks-per-node=${ntasks_per_node} --segment=${total_node_num} disaggr_torch.slurm ${ctx_num} 4 4 4480 true 1 16 1024 1024 true "0.7" 0 "$mtp_size" "$concurrency"
    sbatch --nodes=${total_node_num} --ntasks=${ntasks} --ntasks-per-node=${ntasks_per_node} --segment=${total_node_num} disaggr_torch.slurm ${ctx_num} 4 4 4480 true 1 16 1024 1024 true "0.7" 256 "$mtp_size" "$concurrency"
    sbatch --nodes=${total_node_num} --ntasks=${ntasks} --ntasks-per-node=${ntasks_per_node} --segment=${total_node_num} disaggr_torch.slurm ${ctx_num} 4 4 4480 true 1 16 1024 1024 true "0.7" 288 "$mtp_size" "$concurrency"
done

# dep32 eplb288
for b in 512; do
    concurrency=$((b * 32))
    ctx_num=$(((concurrency + 5499)/5500))
    total_node_num=$((ctx_num + 8))
    ntasks=$((total_node_num * ntasks_per_node))
    sbatch --nodes=${total_node_num} --ntasks=${ntasks} --ntasks-per-node=${ntasks_per_node} --segment=${total_node_num} disaggr_torch.slurm ${ctx_num} 4 4 4480 true 1 32 1024 1024 true "0.7" 288 "$mtp_size" "$concurrency"
done
