#!/bin/bash
mtp_size=0

# dep8
for b in 1 64 1024; do
    concurrency=$((b * 8))
    ctx_num=$(((concurrency + 5499)/5500))
    total_gpu_num=$((ctx_num + 2))
    total_tasks=$((total_gpu_num * 4))
    sbatch --nodes=${total_gpu_num} --ntasks=${total_tasks} --ntasks-per-node=4 --segment=${total_gpu_num} disaggr_torch.slurm ${ctx_num} 4 4 4480 true 1 8 1024 1024 true "0.8" 0 "$mtp_size" "$concurrency"
done

# dep16 eplb0, 256, 288
for b in 1 64 1024; do
    concurrency=$((b * 16))
    ctx_num=$(((concurrency + 5499)/5500))
    total_gpu_num=$((ctx_num + 4))
    total_tasks=$((total_gpu_num * 4))
    sbatch --nodes=${total_gpu_num} --ntasks=${total_tasks} --ntasks-per-node=4 --segment=${total_gpu_num} disaggr_torch.slurm ${ctx_num} 4 4 4480 true 1 16 1024 1024 true "0.7" 0 "$mtp_size" "$concurrency"
    sbatch --nodes=${total_gpu_num} --ntasks=${total_tasks} --ntasks-per-node=4 --segment=${total_gpu_num} disaggr_torch.slurm ${ctx_num} 4 4 4480 true 1 16 1024 1024 true "0.7" 256 "$mtp_size" "$concurrency"
    sbatch --nodes=${total_gpu_num} --ntasks=${total_tasks} --ntasks-per-node=4 --segment=${total_gpu_num} disaggr_torch.slurm ${ctx_num} 4 4 4480 true 1 16 1024 1024 true "0.7" 288 "$mtp_size" "$concurrency"
done

# dep32 eplb288
for b in 512; do
    concurrency=$((b * 32))
    ctx_num=$(((concurrency + 5499)/5500))
    total_gpu_num=$((ctx_num + 8))
    total_tasks=$((total_gpu_num * 4))
    sbatch --nodes=${total_gpu_num} --ntasks=${total_tasks} --ntasks-per-node=4 --segment=${total_gpu_num} disaggr_torch.slurm ${ctx_num} 4 4 4480 true 1 32 1024 1024 true "0.7" 288 "$mtp_size" "$concurrency"
done
