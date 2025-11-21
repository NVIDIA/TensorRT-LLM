# Multinode Aggr Perf Test Scripts

This directory contains scripts to run pytest's multinode aggr perf tests using SLURM.

## Files

- **submit.py**: Main submission script that reads config, calculates resources, and submits the job
- **launch.sh**: SLURM batch script that orchestrates the test execution
- **install.sh**: Installation script that runs on each node to set up the environment
- **run.sh**: Execution script that runs the actual pytest command

## Usage

### Basic Usage

```bash
python3 submit.py \
  --partition gb200nvl72_preprod \
  --jobname coreai_comparch_trtllm_aggr \
  --account blackwell \
  --trtllmsrc /home/scratch.chenfeiz_gpu/another_repo/tekit-2 \
  --jobworkspace /home/scratch.chenfeiz_gpu/another_repo/tekit-2/pytest_aggr_workspace-$(date +%Y%m%d_%H%M%S) \
  --test-name l0_gb200_multi_nodes-r1_fp4_v2_dep8_mtp1 \
  --mounts "/home/chenfeiz/:/home/chenfeiz/,/home/chenfeiz/.cache:/root/.cache,/home/scratch.chenfeiz_gpu:/home/scratch.chenfeiz_gpu,/home/scratch.trt_llm_data:/home/scratch.trt_llm_data" \
  # --build-wheel  # Uncomment to build wheel before running tests
```

### Arguments

- `--partition`: SLURM partition to use (required)
- `--jobname`: SLURM job name for SLURM (required)
- `--account`: SLURM account (required)
- `--trtllmsrc`: Path to the TRT-LLM repository source (required)
- `--jobworkspace`: Directory for job outputs and logs (required)
- `--stagename`: Stage name for the test (optional)
- `--test-name`: Test name, e.g., l0_gb200_multi_nodes-r1_fp4_v2_dep8_mtp1 (required)
- `--mounts`: Mounts directories (required)
- `--llm-models-root`: LLM models root directory (optional)
- `--build-wheel`: Build TensorRT-LLM wheel before running tests (optional flag)
