# Disaggregated Perf Test Scripts

This directory contains scripts to run multinode disaggregated perf tests using SLURM, supporting both CI and local pytest runs.

## Files

- **submit.py**: Main submission script that reads config, calculates resources, submits the job, and monitors execution
- **slurm_launch_draft.sh**: Draft SLURM batch script that orchestrates the disaggregated test execution
- **slurm_run.sh**: SLURM run script for local run.

## Usage for Local Test

Run python3 submit.py to generate slurm_launch.sh, then sbatch slurm_launch.sh.

```bash
# For ptyche
python3 submit.py \
  --config-yaml /lustre/fsw/coreai_comparch_trtllm/chenfeiz/repos/trtllm-1/jenkins/scripts/perf/disaggregated/l0_gb200_multi_nodes_disagg.yaml \
  --draft-launch-sh /lustre/fsw/coreai_comparch_trtllm/chenfeiz/repos/trtllm-1/jenkins/scripts/perf/disaggregated/slurm_launch_draft.sh \
  --launch-sh /lustre/fsw/coreai_comparch_trtllm/chenfeiz/repos/trtllm-1/jenkins/scripts/perf/disaggregated/slurm_launch.sh \
  --run-sh /lustre/fsw/coreai_comparch_trtllm/chenfeiz/repos/trtllm-1/jenkins/scripts/perf/disaggregated/slurm_run.sh \
  --stage-name GB200-8_GPUs-2_Nodes-PyTorch-Perf-Sanity-Disagg

# For oci
python3 submit.py \
  --config-yaml /lustre/fsw/portfolios/coreai/users/chenfeiz/repo/trtllm-1/jenkins/scripts/perf/disaggregated/l0_gb200_multi_nodes_disagg.yaml \
  --draft-launch-sh /lustre/fsw/portfolios/coreai/users/chenfeiz/repo/trtllm-1/jenkins/scripts/perf/disaggregated/slurm_launch_draft.sh \
  --launch-sh /lustre/fsw/portfolios/coreai/users/chenfeiz/repo/trtllm-1/jenkins/scripts/perf/disaggregated/slurm_launch.sh \
  --run-sh /lustre/fsw/portfolios/coreai/users/chenfeiz/repo/trtllm-1/jenkins/scripts/perf/disaggregated/slurm_run.sh \
  --stage-name GB200-8_GPUs-2_Nodes-PyTorch-Perf-Sanity-Disagg

# For dlcluster
python3 submit.py \
  --config-yaml /home/scratch.chenfeiz_gpu/another_repo/tekit-2/jenkins/scripts/perf/disaggregated/l0_gb200_multi_nodes_disagg.yaml \
  --draft-launch-sh /home/scratch.chenfeiz_gpu/another_repo/tekit-2/jenkins/scripts/perf/disaggregated/slurm_launch_draft.sh \
  --launch-sh /home/scratch.chenfeiz_gpu/another_repo/tekit-2/jenkins/scripts/perf/disaggregated/slurm_launch.sh \
  --run-sh /home/scratch.chenfeiz_gpu/another_repo/tekit-2/jenkins/scripts/perf/disaggregated/slurm_run.sh \
  --stage-name GB200-8_GPUs-2_Nodes-PyTorch-Perf-Sanity-Disagg
```
