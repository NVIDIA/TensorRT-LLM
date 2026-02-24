# Wide-EP SLURM Benchmark Scripts

This directory contains configuration files and utilities for benchmarking TensorRT-LLM Wide Expert Parallelism (Wide-EP) performance on SLURM-managed clusters.

## Overview

The Wide-EP benchmarking infrastructure leverages the [disaggregated serving benchmark framework](../../disaggregated/slurm/benchmark/) to evaluate MoE model performance with expert parallelism at scale. This directory provides:

- **Configuration templates** for Wide-EP deployments (`config.yaml`)
- **Post-processing utilities** for benchmark analysis (`process_gen_iterlog.py`)

### Core Implementation

The core SLURM submission and execution logic is implemented in [`examples/disaggregated/slurm/benchmark/`](../../disaggregated/slurm/benchmark/). The scripts in that directory handle:
- Job submission to SLURM clusters
- Multi-node distributed execution
- Worker initialization and coordination
- Benchmark execution and result collection

## Files in This Directory

### `config.yaml`

Example configuration file for Wide-EP benchmarks. Key sections include:

- **SLURM Configuration**: Cluster-specific settings (partition, account, job parameters)
- **Benchmark Mode**: Testing parameters (concurrency, sequence lengths, streaming mode)
- **Hardware Configuration**: GPU topology and server counts
- **Environment**: Container images, model paths, and environment variables
- **Worker Configuration**: Detailed settings for generation and context workers, including:
  - Parallelism settings (TP, EP, PP)
  - MoE configuration with load balancer settings
  - CUDA graph and KV cache configurations
  - Speculative decoding parameters

See the inline comments in [`config.yaml`](config.yaml) for detailed parameter descriptions.

### `process_gen_iterlog.py`

Post-processing script that analyzes benchmark iteration logs to generate performance reports. This script:
- Parses generation worker iteration logs
- Computes throughput and latency statistics
- Generates summary reports for benchmark results

## Usage

### Prerequisites

Before running benchmarks, ensure you have:

1. **SLURM Cluster Access**: Valid account and partition allocation
2. **Container Environment**:
   - NVIDIA Container Toolkit configured
   - Required device mappings (e.g., `/dev/nvidia-caps-imex-channels` for GB200/GB300 NVL72, `/dev/gdrdrv` for GDRCopy)
3. **Model Files**: Checkpoint files accessible from all cluster nodes
4. **Configuration**: Updated `config.yaml` with your cluster-specific settings

### Configuration Setup

1. Copy and customize the example configuration:

```bash
cp config.yaml my_benchmark_config.yaml
```

2. Update the following required fields in `my_benchmark_config.yaml`:
   - `slurm.partition`: Your SLURM partition name
   - `slurm.account`: Your SLURM account
   - `environment.container_image`: Path to your TensorRT-LLM container
   - `environment.model_path`: Path to your model checkpoint
   - `environment.work_dir`: Working directory for benchmark outputs
   - `environment.container_mount`: Mount paths for the container

3. Adjust hardware configuration to match your setup:
   - `hardware.gpus_per_node`: GPUs available per node
   - `hardware.num_ctx_servers`: Number of context processing servers
   - `hardware.num_gen_servers`: Number of generation servers

### Running Benchmarks

Submit a benchmark job using the `submit.py` script from the disaggregated benchmark directory:

```bash
# Navigate to the benchmark submission directory
cd ../../disaggregated/slurm/benchmark/

# Submit the job with your configuration
python3 submit.py -c ../../../wide_ep/slurm_scripts/my_benchmark_config.yaml
```

The script will:
1. Validate your configuration
2. Submit a SLURM job with the specified parameters
3. Launch distributed workers across the allocated nodes
4. Execute the benchmark workload
5. Collect results in the specified working directory

### Monitoring and Results

After submission, monitor your job:

```bash
# Check job status
squeue -u $USER

# View job output (replace <job_id> with your SLURM job ID)
tail -f slurm-<job_id>.out

# Check worker logs in the working directory
ls <work_dir>/logs/
```

Benchmark results will be saved in your configured `work_dir`, including:
- Iteration logs from generation and context workers
- Performance metrics and throughput statistics
- System logs and error reports

### Post-Processing Results

Process generation iteration logs to extract performance metrics:

```bash
python3 process_gen_iterlog.py <path_to_gen_iter_log>
```
