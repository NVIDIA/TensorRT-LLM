# Disaggregated Inference Benchmark Scripts

This directory contains scripts to run disaggregated inference benchmarks using TensorRT-LLM and SLURM. The benchmark system uses Python for orchestration and YAML for configuration.

## Overview

The benchmarking process is orchestrated through a combination of Python scripts and YAML configuration:

1. **`submit.py`**: Main entry point for submitting benchmark jobs. Handles configuration validation, worker config generation, and SLURM job submission.
2. **`config.yaml`**: The main configuration file that defines all benchmark parameters including SLURM settings, hardware configuration, worker settings, and benchmark modes.
3. **`disaggr_torch.slurm`**: The SLURM batch script that sets up the container environment, initializes workers, and runs benchmarks.
4. **Supporting scripts**:
   - `start_worker.sh`: Initializes context and generation workers
   - `start_server.sh`: Starts the disaggregated serving coordinator
   - `wait_server.sh`: Waits for server readiness before benchmarking
   - `run_benchmark.sh` / `run_benchmark_nv_sa.sh`: Execute benchmark workloads
   - `accuracy_eval.sh`: Runs accuracy evaluation using lm_eval
   - `gen_server_config.py`: Generates server configuration from worker settings

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
  extra_args: ""  # Additional SLURM arguments (e.g., "--gres=gpu:4 --exclude=node1")
  set_segment: true # Optional: whether to set the segment for the job
  numa_bind: true  # Enable NUMA binding for GB200 NVL72
```

### 2. Benchmark Configuration
```yaml
benchmark:
  mode: "e2e"  # Options: e2e (end-to-end), gen_only (generation only)
  use_nv_sa_benchmark: false  # Use NVIDIA SA benchmark script
  multi_round: 8  # Number of benchmark rounds
  benchmark_ratio: 0.8  # Fraction of requests to benchmark
  streaming: true  # Enable streaming mode
  concurrency_list: "16"  # Comma-separated list of concurrency levels to test
  input_length: 1024  # Input sequence length
  output_length: 1024  # Output sequence length
  dataset_file: "<dataset_file>"  # Path to dataset file
```

### 3. Hardware Configuration
```yaml
hardware:
  gpus_per_node: 4  # GPUs per node in your cluster
  num_ctx_servers: 1  # Number of context processing servers
  num_gen_servers: 1  # Number of generation servers
```

### 4. Environment Configuration
```yaml
environment:
  container_mount: "<container_mount>"  # Format: path1:path1,path2:path2
  container_image: "<container_image>"  # Path to TensorRT-LLM container
  model_path: "<model_path>"  # Path to model checkpoint
  trtllm_repo: "<trtllm_repo>"  # Path to TensorRT-LLM repository
  build_wheel: false  # Set true to build TensorRT-LLM from source
  trtllm_wheel_path: ""  # Path to pre-built wheel (if not building from source)
  work_dir: "<full_path_to_work_dir>"  # Working directory for outputs
  worker_env_var: "TLLM_LOG_LEVEL=INFO ..."  # Environment variables for workers
  server_env_var: "TRTLLM_SERVER_DISABLE_GC=1"  # Environment variables for server
```

### 5. Worker Configuration
The worker configuration section defines detailed settings for both context and generation workers:

```yaml
worker_config:
  gen:
    tensor_parallel_size: 8
    moe_expert_parallel_size: 8  # For MoE models
    enable_attention_dp: true  # Enable attention data parallelism
    # Additional generation worker settings...

  ctx:
    tensor_parallel_size: 4
    moe_expert_parallel_size: 4
    enable_attention_dp: true
    # Additional context worker settings...
```

## Running the Benchmark

The benchmark system uses a streamlined approach with configuration defined in YAML and execution handled by the `submit.py` Python script.

### Prerequisites

Before running benchmarks, ensure you have:

1. **SLURM cluster access** with valid partition and account
2. **Container environment** with NVIDIA Container Toolkit configured
3. **Model checkpoint** files accessible from all cluster nodes
4. **Required device mappings** configured (e.g., `/dev/gdrdrv` for GDRCopy)
5. **Python 3** with PyYAML installed

### Step 1: Configure the Benchmark

Create or edit a configuration YAML file based on `config.yaml`. Update the following required fields:

1. **SLURM settings**: partition, account, job time limits
2. **Hardware configuration**: GPUs per node, server counts
3. **Benchmark parameters**: mode, sequence lengths, concurrency, streaming
4. **Environment settings**: container image and mount paths, model path, work directory
5. **Worker configurations**: parallelism settings, batch sizes, memory configurations

Example:
```bash
cp config.yaml my_benchmark.yaml
# Edit my_benchmark.yaml with your settings
```

### Step 2: Submit the Benchmark Job

Use the `submit.py` script to submit your benchmark job:

```bash
# Submit a single configuration
python3 submit.py -c my_benchmark.yaml

# Or submit multiple configurations from a directory
python3 submit.py -d ./configs/
```

The submission script will:
1. Validate the YAML configuration
2. Calculate required nodes based on parallelism settings
3. Generate worker configuration files
4. Submit the SLURM job with appropriate parameters

The SLURM job (via `disaggr_torch.slurm`) will then:
1. Start the container environment
2. Install or build TensorRT-LLM (if configured)
3. Launch context and generation workers
4. Start the disaggregated serving coordinator
5. Execute the benchmark workload
6. Run accuracy evaluation (if enabled)
7. Collect and store all metrics and logs

### Monitoring and Results

After submitting your job, you can monitor its progress:

```bash
# Check job status
squeue -u $USER

# View job output (replace <job_id> with your SLURM job ID)
tail -f slurm-<job_id>.out

# Monitor worker logs in the work directory
ls <work_dir>/<date>/<isl-osl>/<config>/logs/
```

Results are automatically organized in the work directory:
```
<work_dir>/
  └── <YYYYMMDD>/
      └── <isl>-<osl>/
          └── ctx<N>_gen<M>_dep<X>_batch<Y>_eplb<Z>_mtp<W>/
              ├── logs/
              ├── ctx_config.yaml
              ├── gen_config.yaml
              ├── job_info.txt
              └── bench.log
```

### Benchmark Modes

The system supports three primary benchmark modes:

1. **End-to-End (e2e)**: Tests the complete disaggregated inference pipeline including both context processing and token generation phases
2. **Generation Only (gen_only)**: Focuses solely on testing the generation phase with pre-cached KV data
3. **Generation Only No Context (gen_only_no_context)**: Skips launching context workers entirely by setting `TRTLLM_DISAGG_BENCHMARK_GEN_ONLY=1`. This is useful when you only want to benchmark the generation phase without allocating resources for context workers.

Configure the mode in the YAML file:
```yaml
benchmark:
  mode: "e2e"  # or "gen_only" or "gen_only_no_context"
```

### Metrics Collection

The benchmark system collects various performance metrics:

- **TTFT** (Time to First Token): Latency from request submission to first token generation
- **TPOT** (Time Per Output Token): Average time to generate each token
- **ITL** (Inter-Token Latency): Latency between consecutive tokens
- **E2EL** (End-to-End Latency): Total request latency from input to completion
- **Throughput**: Requests per second and tokens per second

Metrics are automatically collected from worker iteration logs and stored in the work directory.

### Advanced Features

#### 1. Accuracy Evaluation

Enable accuracy evaluation using the lm_eval framework:

```yaml
accuracy:
  enable_accuracy_test: true
  model: "local-completions"
  tasks: "gsm8k,hellaswag,mmlu"  # Comma-separated task list
  model_args_extra: "num_concurrent=512,max_retries=3,tokenized_requests=false,timeout=1200,max_gen_toks=256,max_length=4096"
```

Accuracy results will be saved in `<log_dir>/accuracy_eval/` after benchmark completion.

#### 2. NVIDIA Nsight Systems Profiling

Enable profiling to analyze performance bottlenecks:

```yaml
profiling:
  nsys_on: true
  ctx_profile_range: "10-30"  # Profile iterations 10-30 for context workers
  gen_profile_range: "200-250"  # Profile iterations 200-250 for generation workers
```

Profiling data (`.nsys-rep` files) will be saved in the log directory.

#### 3. Batch Job Submission

Submit multiple benchmark configurations at once:

```bash
# Create a directory with multiple config files
mkdir -p ./configs
cp config.yaml ./configs/config1.yaml
cp config.yaml ./configs/config2.yaml
# Edit each config...

# Submit all configurations
python3 submit.py -d ./configs/
```

Each configuration will be submitted as a separate SLURM job.

#### 4. Custom TensorRT-LLM Installation

Build from source:
```yaml
environment:
  trtllm_repo: "/path/to/TensorRT-LLM"
  build_wheel: true  # Builds wheel on one node
```

Or install from pre-built wheel:
```yaml
environment:
  trtllm_wheel_path: "/path/to/tensorrt_llm-*.whl"
  trtllm_repo: ""
  build_wheel: false
```
