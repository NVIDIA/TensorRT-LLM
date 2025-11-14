# Disaggregated Inference Benchmark Scripts

This directory contains scripts to run disaggregated inference benchmarks using TensorRT-LLM and SLURM. The benchmark system uses Python for orchestration and YAML for configuration.

## Overview

The benchmarking process is orchestrated through a combination of Python scripts and YAML configuration:

1. `config.yaml`: The main configuration file that defines all benchmark parameters including SLURM settings, hardware configuration, worker settings, and benchmark modes.
2. `disaggr_torch.slurm`: The SLURM script that sets up and runs a single benchmark experiment based on the YAML configuration.
3. Python scripts for configuration and execution:
   - Worker configuration generation
   - Server configuration generation
   - Benchmark execution and metrics collection

## Configuration (config.yaml)

The benchmark is configured through a YAML file with the following sections:

### 1. SLURM Configuration
```yaml
slurm:
  script_file: "disaggr_torch.slurm"
  partition: "<partition>"
  account: "<account>"
  job_time: "02:00:00"
  job_name: "<job_name>"
  numa_bind: true
```

### 2. Benchmark Mode
```yaml
benchmark:
  mode: "e2e"  # Options: e2e, gen_only
  use_nv_sa_benchmark: false
  multi_round: 8
  benchmark_ratio: 0.8
  streaming: true
```

### 3. Hardware Configuration
```yaml
hardware:
  gpus_per_node: 4
  num_ctx_servers: 1
  num_gen_servers: 1
```

### 4. Sequence Configuration
```yaml
sequence:
  input_length: 1024
  output_length: 1024
```

### 5. Environment Configuration
```yaml
environment:
  container_mount: "<container_mount>"  # Format: path1:path1,path2:path2
  container_image: "<container_image>"
  model_path: "<model_path>"
  trtllm_repo: "<trtllm_repo>"
  build_wheel: false
  dataset_file: "<dataset_file>"
  work_dir: "<full_path_to_work_dir>"
```

### 6. Worker Configuration
The worker configuration section defines detailed settings for both context and generation workers:

```yaml
worker_config:
  concurrency_list: "16"
  eplb_num_slots: 0
  mtp_size: 0
  gen:
    tensor_parallel_size: 16
    pipeline_parallel_size: 1
    max_batch_size: 64
    max_num_tokens: 64
    enable_attention_dp: true
    # Additional generation worker settings...
  ctx:
    tensor_parallel_size: 4
    pipeline_parallel_size: 1
    max_batch_size: 4
    max_num_tokens: 4608
    enable_attention_dp: true
    # Additional context worker settings...
```

## Running the Benchmark

The benchmark system now uses a more streamlined approach with configuration defined in YAML and execution handled by Python scripts.

### Step 1: Configure the Benchmark

Edit the `config.yaml` file to set up your benchmark parameters. The configuration is organized into logical sections:

1. SLURM settings (partition, account, time limits)
2. Hardware configuration (GPUs, server counts)
3. Benchmark parameters (mode, sequence lengths, streaming)
4. Environment settings (container, model paths)
5. Worker configurations (parallelism, batch sizes, memory settings)

### Step 2: Launch the Benchmark

The benchmark can be launched using the SLURM system:

```bash
sbatch disaggr_torch.slurm
```

The SLURM script will:
1. Read and validate the YAML configuration
2. Set up the container environment
3. Configure and start the workers and servers
4. Execute the benchmark
5. Collect and store metrics

### Benchmark Modes

The system supports two primary benchmark modes:

1. **End-to-End (e2e)**: Tests the complete pipeline including both context and generation phases
2. **Generation Only (gen_only)**: Focuses on testing just the generation phase

Configure the mode in the YAML file:
```yaml
benchmark:
  mode: "e2e"  # or "gen_only"
```

### Metrics Collection

The benchmark system collects various performance metrics:

- TTFT (Time to First Token)
- TPOT (Throughput Over Time)
- ITL (Inter-Token Latency)
- E2EL (End-to-End Latency)

Metrics are automatically collected and stored in the work directory specified in the configuration.

### Advanced Features

1. **NVIDIA SA Benchmark Integration**
   ```yaml
   benchmark:
     use_nv_sa_benchmark: true
   ```

2. **Profiling Support**
   ```yaml
   profiling:
     nsys_on: true
   ```

3. **Custom Worker Settings**
   The worker configuration section allows detailed customization of both context and generation workers, including:
   - Tensor and pipeline parallelism
   - Batch sizes and token limits
   - Memory management
   - Cache configuration
   - MoE settings (if applicable)

4. **Container and Build Options**
   ```yaml
   environment:
     build_wheel: true  # Build TensorRT-LLM from source
     container_mount: "path1:path1,path2:path2"
   ```

### Output and Logs

Benchmark results and logs are stored in the specified work directory, including:
- Performance metrics
- Worker and server logs
- Profiling data (if enabled)
- Error logs and diagnostics

The system automatically organizes outputs by benchmark run and configuration.
