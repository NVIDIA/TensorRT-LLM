# Aggregated Inference Benchmark Scripts

This directory contains scripts to run aggregated inference benchmarks using TensorRT-LLM and SLURM. The benchmark system uses Python for orchestration and YAML for configuration.

## Overview

The benchmarking process is orchestrated through a combination of Python scripts and YAML configuration:

1. **`submit.py`**: Main entry point for submitting benchmark jobs. Handles configuration validation, server command generation, and SLURM job submission.
2. **`config.yaml`**: The main configuration file that defines all benchmark parameters including SLURM settings, hardware configuration, server settings, and benchmark workload.
3. **`aggregated_torch.slurm`**: The SLURM batch script that sets up the container environment, installs/builds TensorRT-LLM, launches the server, and runs benchmarks.
4. **Supporting scripts**:
   - `start_server.sh`: Launches `trtllm-serve` on each rank with optional NUMA binding
   - `wait_server.sh`: Polls the server's `/health` endpoint until it is ready to accept requests
   - `extra_llm_api_options.yaml`: Additional LLM API options (KV cache, CUDA graphs, MoE backend, etc.) forwarded to `trtllm-serve`

## Configuration (config.yaml)

The benchmark is configured through a YAML file with the following sections:

### 1. SLURM Configuration
```yaml
slurm:
  script_file: "aggregated_torch.slurm"
  partition: "<partition>"
  account: "<account>"
  job_time: "5:00:00"
  job_name: "<job_name>"
  extra_args: ""  # Additional SLURM arguments (e.g., "--gres=gpu:4 --exclude=node1")
  set_segment: true # Optional: whether to set the segment for the job
  numa_bind: true  # Enable NUMA binding for GB200/GB300 NVL72
```

### 2. Benchmark Configuration
```yaml
benchmark:
  input_length: 4096   # Input sequence length
  output_length: 65536 # Output sequence length
  num_prompts: 1024    # Total number of prompts to send to the server
  max_concurrency: 256 # Maximum number of in-flight requests
  streaming: true      # Enable streaming mode
  dataset_name: "<dataset_file>"  # Dataset name (e.g., "random") or path to dataset file
```

### 3. Hardware Configuration
```yaml
hardware:
  gpus_per_node: 4  # GPUs per node in your cluster
```

### 4. Server Configuration
The server configuration defines the settings for the single aggregated `trtllm-serve` instance:

```yaml
server:
  model_path: "<path_to_model>"      # Path to model checkpoint
  tp_size: 8                          # Tensor-parallel size
  ep_size: 8                          # Expert-parallel size (for MoE models)
  pp_size: 1                          # Pipeline-parallel size
  max_batch_size: 512
  max_num_tokens: 8192
  backend: "pytorch"
  trust_remote_code: true
  port: 8000
  extra_llm_api_options: "extra_llm_api_options.yaml"  # Path to extra LLM API options file
```

The `extra_llm_api_options` file allows fine-grained control over KV cache, CUDA graphs, MoE backend selection, attention data parallelism, and other runtime features. An example:

```yaml
cuda_graph_config:
  enable_padding: true
  batch_sizes:
  - 512

kv_cache_config:
  enable_block_reuse: false
  dtype: auto

moe_config:
  backend: TRTLLM
  use_low_precision_moe_combine: true

enable_attention_dp: true
```

### 5. Environment Configuration
```yaml
environment:
  container_mount: "<container_mount>"    # Format: path1:path1,path2:path2
  container_image: "<container_image>"    # Path to TensorRT-LLM container
  trtllm_repo: "<trtllm_repo>"            # Path to TensorRT-LLM repository
  build_wheel: false                       # Set true to build TensorRT-LLM from source
  cuda_architectures: "100-real"          # Target CUDA architectures when building
  trtllm_wheel_path: ""                    # Path to pre-built wheel (if not building from source)
  work_dir: "<full_path_to_work_dir>"     # Working directory for outputs
  worker_env_var: "TLLM_LOG_LEVEL=INFO ..." # Environment variables passed to the server ranks
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
2. **Hardware configuration**: GPUs per node
3. **Benchmark parameters**: input/output lengths, number of prompts, concurrency, streaming, dataset
4. **Server settings**: model path, parallelism sizes, batch sizes, port
5. **Environment settings**: container image and mount paths, work directory, TensorRT-LLM source/wheel

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

# Preview the generated sbatch command without submitting
python3 submit.py -c my_benchmark.yaml --dry-run
```

The submission script will:
1. Validate the YAML configuration
2. Calculate required nodes based on parallelism settings
3. Generate `start_server_cmds.sh`, `wait_server_cmds.sh`, and `client_cmds.sh` in the log directory
4. Submit the SLURM job with appropriate parameters

The SLURM job (via `aggregated_torch.slurm`) will then:
1. Start the container environment
2. Install or build TensorRT-LLM (if configured)
3. Launch the aggregated `trtllm-serve` instance across all ranks
4. Wait for the server to become healthy
5. Execute the benchmark client (`tensorrt_llm.serve.scripts.benchmark_serving`)
6. Collect and store all metrics and logs

### Monitoring and Results

After submitting your job, you can monitor its progress:

```bash
# Check job status
squeue -u $USER

# View job output (replace <job_id> with your SLURM job ID)
tail -f <log_dir>/slurm-<job_id>.out

# Follow the server and benchmark logs
tail -f <log_dir>/3_output_server.log
tail -f <log_dir>/5_bench.log
```

Results are automatically organized in the log directory:
```
<work_dir>/logs/
  └── <YYYYMMDD-HHMMSS>/
      └── <isl>-<osl>/
          └── agg_tp<T>_ep<E>_pp<P>_batch<B>[_<config_stem>]/
              ├── config.yaml
              ├── extra_llm_api_options.yaml
              ├── environment.txt
              ├── start_server_cmds.sh
              ├── wait_server_cmds.sh
              ├── client_cmds.sh
              ├── 1_container_launch.log
              ├── 2_install.log        # or 2_build.log when building from source
              ├── 3_output_server.log
              ├── 4_wait_server.log
              ├── 5_bench.log
              ├── 6_done_<job_id>.txt
              ├── benchmark_serving_results.json
              ├── slurm-<job_id>.out
              └── slurm-<job_id>.err
```

### Metrics Collection

The benchmark system collects various performance metrics via `tensorrt_llm.serve.scripts.benchmark_serving`:

- **TTFT** (Time to First Token): Latency from request submission to first token generation
- **TPOT** (Time Per Output Token): Average time to generate each token
- **ITL** (Inter-Token Latency): Latency between consecutive tokens
- **Throughput**: Requests per second and tokens per second

Results are saved to `benchmark_serving_results.json` in the log directory along with percentile breakdowns for TTFT, TPOT, and ITL.

### Advanced Features

#### 1. Tuning LLM API Options

Runtime behavior of `trtllm-serve` can be tuned through `extra_llm_api_options.yaml`. Common knobs include:

- **KV cache**: `kv_cache_config.enable_block_reuse`, `kv_cache_config.dtype`, `kv_cache_config.free_gpu_memory_fraction`
- **CUDA graphs**: `cuda_graph_config.enable_padding`, `cuda_graph_config.batch_sizes`
- **MoE backend**: `moe_config.backend` (e.g., `TRTLLM`, `CUTLASS`), `moe_config.use_low_precision_moe_combine`
- **Attention data parallelism**: `enable_attention_dp`

Point `server.extra_llm_api_options` in `config.yaml` at your customized file (relative paths are resolved against the work directory).

#### 2. Batch Job Submission

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

Each configuration will be submitted as a separate SLURM job, and the config file stem is appended to the log directory name to keep results separated.

#### 3. Custom TensorRT-LLM Installation

Build from source:
```yaml
environment:
  trtllm_repo: "/path/to/TensorRT-LLM"
  build_wheel: true            # Builds wheel on one node
  cuda_architectures: "100-real"  # Optional, e.g. "90-real;100-real"
```

Or install from pre-built wheel:
```yaml
environment:
  trtllm_wheel_path: "/path/to/tensorrt_llm-*.whl"
  trtllm_repo: ""
  build_wheel: false
```

If neither `trtllm_wheel_path` nor `trtllm_repo` is provided, the TensorRT-LLM version shipped in the container image is used.
