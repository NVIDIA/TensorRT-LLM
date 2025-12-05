# TensorRT LLM Wide-EP Benchmark Scripts

This directory contains scripts for benchmarking TensorRT LLM wide-ep performance using SLURM job scheduler.

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
# Please find the `submit.py` script in the `examples/disaggregated/slurm/benchmark/` directory.
# An example `config.yaml` for wide EP: `examples/wide_ep/slurm_scripts/config.yaml`.
python3 submit.py -c config.yaml
```
