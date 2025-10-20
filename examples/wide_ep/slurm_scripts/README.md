# TensorRT LLM Wide-EP Benchmark Scripts

This directory contains scripts for benchmarking TensorRT LLM wide-ep performance using SLURM job scheduler.

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

1. `process_gen_iterlog.py` - Processes benchmark results and generates reports

## Usage

### Prerequisites

Before running the scripts, ensure you have:
- Access to a SLURM cluster
- Container image with TensorRT LLM installed
- Model files accessible on the cluster
- Required environment variables set

### Run Benchmarks

```bash
# Please find the `submit.py` script and an example `config.yaml` in the `examples/disaggregated/slurm/benchmark/` directory.
python3 submit.py -c your_config.yaml
```
