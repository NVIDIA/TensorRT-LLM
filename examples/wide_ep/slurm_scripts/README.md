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

1. **`submit.sh`** - Main entry point for submitting benchmark jobs
2. **`disaggr_torch.slurm`** - SLURM job script orchestrating the entire benchmark
3. **`gen_yaml.py`** - Generates configuration files for serving setup
4. **`start_server.sh`** - Starts the inference server
5. **`start_worker.sh`** - Starts the worker processes
6. **`run_benchmark.sh`** - Executes the benchmark workload
7. **`process_gen_iterlog.py`** - Processes benchmark results and generates reports

## Usage

### Prerequisites

Before running the scripts, ensure you have:
- Access to a SLURM cluster
- Container image with TensorRT-LLM installed
- Model files accessible on the cluster
- Required environment variables set

### Configuration

Edit the following variables in `submit.sh` and `disaggr_torch.slurm`:

```bash
# In disaggr_torch.slurm
container_image=${container_image}     # Your container image
mount_dir=${mount_dir}                 # Mount directory path
model_dir=${model_dir}                 # Model directory path
```

### Running Benchmarks

1. **Submit benchmark jobs**:
   ```bash
   ./submit.sh
   ```

2. **Monitor job progress**:
   ```bash
   squeue -u $USER
   ```

3. **View results**:
   Results are saved in `bm_20250703_deepseek-r1-{isl}-{osl}/` directory

## Script Details

### `submit.sh`
Main entry script that submits multiple SLURM jobs with different configurations:
- **DEP8**: 8-way parallelism for decode servers
- **DEP16**: 16-way parallelism with different EPLB slot configurations
- **DEP32**: 32-way parallelism for high-throughput scenarios

Parameters tested:
- Concurrency levels: 1x, 64x, 1024x multipliers
- EPLB slots: 0, 256, 288
- Different parallelism sizes

### `disaggr_torch.slurm`
SLURM job script that:
1. Sets up container environment
2. Generates configuration files
3. Starts server and workers
4. Executes benchmarks
5. Cleans up processes

**Key parameters**:
- `num_ctx_servers`: Number of context servers
- `ctx_tp_size`: Tensor parallel size for context servers
- `num_gen_servers`: Number of generation servers
- `gen_tp_size`: Tensor parallel size for generation servers
- `concurrency`: Number of concurrent requests

### `gen_yaml.py`
Generates YAML configuration files with:
- Server topology and resource allocation
- Network configuration (hostnames, ports)
- Memory and batch size settings
- Optimization parameters (CUDA graphs, KV cache)

**Key features**:
- Automatic node and task allocation
- Support for attention data parallelism
- MoE load balancing configuration
- Speculative decoding (MTP) support

### `start_server.sh` & `start_worker.sh`
- **Server**: Starts the main inference server with API endpoint
- **Workers**: Starts MPI workers for distributed processing
- Support for profiling with NSight Systems
- Environment variable configuration for optimizations

### `run_benchmark.sh`
Executes benchmarking using TensorRT-LLM's benchmark_serving tool:
- Downloads ShareGPT dataset for realistic workloads
- Waits for server health checks
- Runs load testing with specified concurrency
- Collects performance metrics
- Gracefully shuts down services

**Metrics collected**:
- Throughput (tokens/second)
- Latency (request completion time)
- Context vs generation only statistics

### `process_gen_iterlog.py`
Post-processes benchmark results:
- Parses iteration logs from workers
- Calculates throughput metrics
- Generates CSV reports
- Supports MTP (Multi-Token Prediction) analysis
