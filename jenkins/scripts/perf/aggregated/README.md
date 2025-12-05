# Multinode Aggr Perf Test Scripts

This directory contains scripts to run pytest's multinode aggregated perf tests locally. CI will not use these scripts.

## Files

- **submit.py**: Main submission script that reads config, calculates resources, submits the job, and monitors execution
- **slurm_launch_draft.sh**: Draft SLURM batch script that orchestrates the disaggregated test execution
- **slurm_run.sh**: SLURM run script for local run.

## Usage

### Basic Usage

Run python3 submit.py to generate slurm_launch.sh, then sbatch slurm_launch.sh.

```bash
# For dlcluster
python3 submit.py \
  --config-yaml /home/scratch.chenfeiz_gpu/another_repo/tekit-2/jenkins/scripts/perf/aggregated/l0_gb200_multi_nodes.yaml \
  --draft-launch-sh /home/scratch.chenfeiz_gpu/another_repo/tekit-2/jenkins/scripts/perf/aggregated/slurm_launch_draft.sh \
  --launch-sh /home/scratch.chenfeiz_gpu/another_repo/tekit-2/jenkins/scripts/perf/aggregated/slurm_launch.sh \
  --run-sh /home/scratch.chenfeiz_gpu/another_repo/tekit-2/jenkins/scripts/perf/aggregated/slurm_run.sh \
  --stage-name GB200-8_GPUs-2_Nodes-PyTorch-Perf-Sanity-Post-Merge-1 \
  --build-wheel  # Uncomment to build wheel before running tests
```

```bash
# For oci
python3 submit.py \
  --config-yaml /lustre/fsw/portfolios/coreai/users/chenfeiz/repo/trtllm-1/jenkins/scripts/perf/aggregated/l0_gb200_multi_nodes.yaml \
  --draft-launch-sh /lustre/fsw/portfolios/coreai/users/chenfeiz/repo/trtllm-1/jenkins/scripts/perf/aggregated/slurm_launch_draft.sh \
  --launch-sh /lustre/fsw/portfolios/coreai/users/chenfeiz/repo/trtllm-1/jenkins/scripts/perf/aggregated/slurm_launch.sh \
  --run-sh /lustre/fsw/portfolios/coreai/users/chenfeiz/repo/trtllm-1/jenkins/scripts/perf/aggregated/slurm_run.sh \
  --stage-name GB200-8_GPUs-2_Nodes-PyTorch-Perf-Sanity-Post-Merge-1 \
  --build-wheel  # Uncomment to build wheel before running tests
```
