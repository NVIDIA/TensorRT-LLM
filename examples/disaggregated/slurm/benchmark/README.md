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
  numa_bind: true  # Enable NUMA binding for GB200/GB300 NVL72
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
  # compact_packing: false  # Opt-in; see note below
```

By default each worker owns whole nodes (round-robin), so cross-worker
traffic never crosses a node boundary. Set `compact_packing: true` to pack
all workers into `ceil(total_gpus / gpus_per_node)` nodes, allowing two
workers to share a physical node when their GPU counts don't align with
node boundaries (e.g. two TP=6 ctx workers fit in 3 four-GPU nodes via
4+2 / 2+4). This is **only recommended on full-mesh NVLink fabrics like
GB200/GB300 NVL72**, where the cross-worker NVLink traffic on the shared
node is free; on PCIe or partitioned-NVLink hosts the shared node will
become a bottleneck and you should leave it off.

### 4. Environment Configuration
```yaml
environment:
  container_mount: "<container_mount>"  # Format: path1:path1,path2:path2
  container_image: "<container_image>"  # Path to TensorRT-LLM container
  model_path: "<model_path>"  # Path to model checkpoint
  trtllm_repo: "<trtllm_repo>"  # Path to TensorRT-LLM repository
  build_wheel: false  # Set true to build TensorRT-LLM from source
  trtllm_wheel_path: ""  # Path to pre-built wheel (if not building from source)
  server_health_timeout: 1800  # Seconds to wait for the server to become healthy
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

#### Mooncake store pool (optional)

To back the deployment with a [Mooncake](https://github.com/kvcache-ai/Mooncake) store, so a prompt prefix computed by one context server can be replayed by another, configure the connector in both worker sections. The job then starts one `mooncake_master` step of its own and waits for the manifest before any worker launches; `disaggr_torch.slurm` detects this by finding `mooncake-store` in either rendered config.

```yaml
worker_config:
  ctx:
    # ... parallelism as above ...
    kv_cache_config:
      use_kv_cache_manager_v2: true
    kv_connector_config:
      connector: mooncake-store
      mooncake_store:
        pool: file://__LOG_DIR__/pool.json
        role: both             # reads and writes the pool
        segment_size: 160GiB   # per rank
        model_key: minimax-m3-fp4
        run_dir: __MOONCAKE_RUN_DIR__
        master_timeout: 900

  gen:
    # ... parallelism as above ...
    kv_connector_config:
      connector: mooncake-store
      mooncake_store:
        pool: file://__LOG_DIR__/pool.json
        role: capacity         # lends memory, transfers nothing
        segment_size: 160GiB   # the same per-rank figure
        model_key: minimax-m3-fp4
        run_dir: __MOONCAKE_RUN_DIR__
        master_timeout: 900
```

Write the `__LOG_DIR__` and `__MOONCAKE_RUN_DIR__` placeholders literally; the job substitutes them at launch. Set the rest as follows.

| Key | What to set | If you get it wrong |
| --- | --- | --- |
| `pool` | `file://__LOG_DIR__/pool.json`, identical on every server. | Servers naming different manifests join different pools and share nothing. |
| `role` | `both` on context, `capacity` on generation. | See [Roles and sizing](#roles-and-sizing). |
| `segment_size` | The same per-rank figure on both sides. | See [Roles and sizing](#roles-and-sizing). |
| `model_key` | Any stable name for this checkpoint, identical on every server. Required. | Servers that disagree share nothing; two *different* checkpoints sharing one name read each other's KV. |
| `run_dir` | `__MOONCAKE_RUN_DIR__`. Context and generation need **separate** directories. | Servers sharing a directory render one client config between them. With more than one server per side the job fails at startup, naming the other server's claim. |
| `master_timeout` | Seconds to wait for the master; 900 covers container start on another node. | Too short and the server fails at startup — which is the point, since a pool that never came up otherwise shows up only as missing cache hits. |

`__LOG_DIR__` becomes the job's log directory, which is not known when the config is written. `__MOONCAKE_RUN_DIR__` becomes `<log dir>/mooncake_<role>_<instance>`, substituted per server by `start_worker.sh` — which is why it cannot be spelled out as a `__LOG_DIR__` path, that being one directory for the whole job where each server needs its own. Both stay inside the log directory because the ranks `srun` starts do not inherit the leader's environment and read the rendered client config back from disk.

##### Roles and sizing

Neither side sets `host_cache_size` or `disk_cache_size`, because the connector forces both to 0 for every role, `capacity` included. The pool is the deployment's offload tier, and a native one would claim a second share of the same node's DRAM, which on the generation side is the DRAM it just lent the pool. Size `segment_size` against the whole node's memory on that basis.

Both sides contribute the same amount per rank, so pool capacity is `total ranks x segment_size` and the generation side — which normally has far more ranks and far more host DRAM — supplies most of it. For 2 context servers at DP2 and 5 generation servers at TP4, that is 24 ranks and 3840 GiB, of which decode holds 83%.

The roles differ only in traffic. `capacity` drives none: those ranks mount their segment and never look up, load or save, and they register no KV cache with Mooncake, so they need no GPUDirect RDMA and keep their scheduler policy and block reuse unchanged.

Keep `segment_size` identical in both sections. Each server sees only its own value, so a mismatch is otherwise invisible; the run's report names the distinct sizes it finds. Note also that `segment_size` is claimed **per rank**, so a node's demand is `ranks_on_node x segment_size` — 4 x 160 GiB on a 4-GPU TP4 node. The connector checks this against available host memory and refuses at startup rather than letting the OOM killer arrive during weight loading.

##### Checking the pool afterwards

After the run, `${full_logdir}/9_mooncake_summary.log` reports the capacity the pool actually had and where blocks landed, read from the records each rank wrote rather than from log messages. The same report is available directly:

```bash
trtllm-serve mooncake_pool_report --run_dir <full_logdir>
```

The job fails rather than running without a pool, so a master that never starts stops the run instead of quietly costing it every cache hit.

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
