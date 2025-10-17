# TensorRT-LLM Wide-EP Benchmark Scripts

This directory contains scripts for benchmarking TensorRT-LLM wide-ep performance using SLURM job scheduler.

## ⚠️ DISCLAIMER

**These scripts are currently not QA'ed and are provided for demonstration purposes only.**

Please note that:

- These scripts have not undergone formal quality assurance testing
- They are intended for demonstration and educational purposes
- Use at your own risk in production environments
- Always review and test scripts thoroughly before running in your specific environment

## Scripts Overview

### Core Scripts

Note that, core implementation of the slurm scripts are included in `examples/disaggregated/slurm/benchmark`.

1. `submit_e2e.sh` - Main entry point for submitting E2E benchmark jobs
2. `submit_gen_only.sh` - Main entry point for submitting gen-only benchmark jobs
3. `process_gen_iterlog.py` - Processes benchmark results and generates reports

## Usage

### Prerequisites

Before running the scripts, ensure you have:
- Access to a SLURM cluster
- Container image with TensorRT-LLM installed
- Model files accessible on the cluster
- Required environment variables set

### Run E2E Benchmarks

```bash
# Refer to `examples/disaggregated/slurm/benchmark/`
# Please find the `disaggr_torch.slurm` script in the `examples/disaggregated/slurm/benchmark/` directory.
# Make sure that SLURM parameters are correctly set in `disaggr_torch.slurm` before executing this script.
./submit_e2e.sh
```


### Run gen-only Benchmarks and post-processes the results using `process_gen_iterlog.py`

```bash
./submit_gen_only.sh

python3 process_gen_iterlog.py --dir_prefix <path>
```
`process_gen_iterlog.py` will be responsible for:
- Parses iteration logs from workers
- Calculates throughput metrics
- Generates CSV reports
- Supports MTP (Multi-Token Prediction) analysis
