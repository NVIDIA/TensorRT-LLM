#! /bin/bash

slurm_file=disaggr_torch.slurm

# ctx1dep4_gen1tep4, max_batch16
for c in 1 2 4 8 16 32 48 64; do
    sbatch --nodes=2 --ntasks=8 --ntasks-per-node=4  ${slurm_file} 1 4 1 8300 true 1 4 32 32 false "0.95" "$c" ctx1dep4_gen1tep4_${c}
done

# ctx2dep4_gen1tep4, max_batch 64
for c in 64 96 128; do
    sbatch --nodes=3 --ntasks=12 --ntasks-per-node=4  ${slurm_file} 2 4 1 8300 true 1 4 64 64 false "0.9" "$c" ctx2dep4_gen1tep4_${c}
done

for c in 128 192 256; do
    sbatch --nodes=4 --ntasks=16 --ntasks-per-node=4  ${slurm_file} 3 4 1 8300 true 1 4 32 32 true "0.9" "$c" ctx3dep4_gen1dep4_${c}
done

for c in 256 384 512; do
    sbatch --nodes=5 --ntasks=20 --ntasks-per-node=4  ${slurm_file} 4 4 1 8300 true 1 4 64 64 true "0.9" "$c" ctx4dep4_gen1dep4_${c}
done

# ctx5dep4_gen1dep4, max_batch
for c in 256 384 512; do
    sbatch --nodes=6 --ntasks=24 --ntasks-per-node=4  ${slurm_file} 5 4 1 8300 true 1 4 64 64 true "0.9" "$c" ctx5dep4_gen1dep4_${c}
done

# ctx7dep4_gen1dep4
for c in 512 768 1024; do
    sbatch --nodes=8 --ntasks=32 --ntasks-per-node=4  ${slurm_file} 7 4 1 8300 true 1 4 128 128 true "0.9" "$c" ctx7dep4_gen1dep4_${c}
done

# ctx8dep4_gen1dep4
for c in 512 768 1024; do
    sbatch --nodes=9 --ntasks=36 --ntasks-per-node=4  ${slurm_file} 8 4 1 8300 true 1 4 128 128 true "0.9" "$c" ctx8dep4_gen1dep4_${c}
done
