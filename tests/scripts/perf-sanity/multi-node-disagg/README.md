# Multinode Disagg Perf Test Scripts

This directory contains scripts to run pytest's multinode disaggregated perf tests using SLURM.

## Files

- **submit.py**: Main submission script that reads config, calculates resources, submits the job, and monitors execution
- **launch.sh**: SLURM batch script that orchestrates the disaggregated test execution
- **install.sh**: Installation script that runs on each node to set up the environment
- **run.sh**: Execution script that runs the actual pytest command with appropriate configuration

## Example

```bash
python submit.py \
  --partition 36x2-a01r \
  --jobname blackwell-dsr1.v1 \
  --account coreai_comparch_trtllm \
  --trtllmsrc /lustre/fsw/coreai_comparch_trtllm/chenfeiz/repos/trtllm-2 \
  --jobworkspace /lustre/fsw/coreai_comparch_trtllm/chenfeiz/repos/trtllm-2/pytest_disagg_workspace-$(date +%Y%m%d_%H%M%S) \
  --test-name l0_gb200_multi_nodes_disagg-r1_fp4_v2_dep8_mtp1 \
  --mounts "/home/chenfeiz/:/home/chenfeiz/,/home/chenfeiz/.cache:/root/.cache,/lustre/fsw/coreai_comparch_trtllm/chenfeiz/:/lustre/fsw/coreai_comparch_trtllm/chenfeiz/"
  --llm-models-root "/lustre/fsw/coreai_comparch_trtllm/chenfeiz/models"
  # --build-wheel  # Uncomment to build wheel before running tests
```

## Arguments

- `--partition`: SLURM partition to use (required)
- `--jobname`: SLURM job name (required)
- `--account`: SLURM account (required)
- `--trtllmsrc`: Path to the TRT-LLM repository source (required)
- `--jobworkspace`: Directory for job outputs and logs (required)
- `--stagename`: Stage name for the test (optional)
- `--test-name`: Test name, e.g., l0_gb200_multi_nodes_disagg-r1_fp4_v2_dep8_mtp1 (required)
- `--mounts`: Container mount directories (required)
- `--llm-models-root`: LLM models root directory (optional, default: /home/scratch.trt_llm_data/llm-models)
- `--build-wheel`: Build TensorRT-LLM wheel before running tests (optional flag)
