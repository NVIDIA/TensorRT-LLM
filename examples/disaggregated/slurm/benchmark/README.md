# Disaggregated Inference Benchmark Scripts

This directory contains scripts to run disaggregated inference benchmarks using TensorRT-LLM and SLURM.

## Overview

The benchmarking process is orchestrated through a set of shell scripts and Python scripts that work together:

1.  `submit.sh`: The main entry point for submitting benchmark jobs to SLURM. It runs a parameter sweep by calling `sbatch` with different configurations.
2.  `disaggr_torch.slurm`: The SLURM script that sets up and runs a single benchmark experiment. It launches a container, generates configuration files, starts the server and workers, and runs the benchmark client.
3.  `gen_worker_config.py`: A Python script that generates the worker configuration YAML file needed by `trtllm-serve`. It determines the worker configuration based on SLURM environment variables and script arguments.
4.  `gen_server_config.py`: A Python script that generates the server configuration YAML file needed by `trtllm-serve`. It determines the server configuration based on the number of context and generation servers.
5.  `start_worker.sh`: A shell script responsible for starting disaggregated workers using `trtllm-serve` on each allocated machine.
6.  `start_server.sh`: A shell script responsible for starting disaggregated server using `trtllm-serve` on each allocated machine.
7.  `run_benchmark.sh`: A shell script that waits for the server to be healthy and then runs the actual benchmark client (`run_benchmark.py`, not included in this directory).

## File Descriptions

### `submit.sh`

This script is used to submit SLURM jobs for running benchmarks with specific configurations. It provides helper functions to calculate required nodes and submit jobs with the right parameters.

The script includes a user configuration section where you can set various parameters:

1. Hardware Configuration:
   - `GPUS_PER_NODE`: Number of GPUs per node (default: 4)

2. Benchmark Configuration:
   - `USE_NV_SA_BENCHMARK`: Whether to use NVIDIA SA benchmark script
   - `ISL`: Input sequence length
   - `OSL`: Output sequence length
   - `MULTI_ROUND`: Number of benchmark rounds
   - `BENCHMARK_RATIO`: Benchmark ratio
   - `STREAMING`: Enable streaming mode
   - `CACHE_MAX_TOKENS`: Cache transceiver max tokens
   - `DATASET_FILE`: Path to dataset file for benchmarking

3. Environment Configuration:
   - `MOUNT_DIR`: Directory to mount in container
   - `CONTAINER_IMAGE`: Path to container image

4. Model Configuration:
   - `MODEL_PATH`: Path to model directory
   - `TRTLLM_REPO`: Path to TensorRT-LLM repository
   - `BUILD_WHEEL`: Whether to build TensorRT-LLM from source

5. Workspace and Profiling Configuration:
   - `WORK_DIR`: Path to work directory
   - `NSYS_ON`: Path for nsys profiling output (empty to disable)

**Usage:**

The script provides a `run_single` function that takes all the necessary parameters for both context and generation servers. Example usage:

```bash
#      CTX: num tp_size batch tokens attn_dp gpu_frac  GEN: num tp_size batch tokens attn_dp gpu_frac mtp eplb concurrency
run_single  1   4       4     4608   true    0.85           1   8       32    128    false   "0.9"    3   0    "16"
```

The script automatically calculates the required number of nodes based on the tensor parallel size and server count.

### `disaggr_torch.slurm`

This is the core SLURM script for a single benchmark run. It is not meant to be run directly, but rather submitted via `sbatch` (e.g., by `submit.sh`).

It takes the following arguments in order:

1.  `num_ctx_servers`: Number of context servers.
2.  `ctx_tp_size`: Tensor parallel size for context servers.
3.  `ctx_batch_size`: Max batch size for context servers.
4.  `ctx_max_num_tokens`: Max number of tokens for context servers.
5.  `ctx_enable_attention_dp`: `true` or `false` to enable attention DP for context servers.
6.  `ctx_gpu_frac`: GPU memory fraction for context servers.
7.  `num_gen_servers`: Number of generation servers.
8.  `gen_tp_size`: Tensor parallel size for generation servers.
9.  `gen_batch_size`: Max batch size for generation servers.
10. `gen_max_num_tokens`: Max number of tokens for generation servers.
11. `gen_enable_attention_dp`: `true` or `false` to enable attention DP for generation servers.
12. `gen_gpu_memory_fraction`: GPU memory fraction for generation servers.
13. `eplb_num_slots`: Number of slots for eplb.
14. `mtp_size`: Number of nextn layers for MTP.
15. `concurrency_list`: Space-separated list of concurrencies for benchmarking.

### `gen_worker_config.py`

This Python script generates the worker configuration YAML file that configures the `trtllm-serve` workers. It creates separate configurations for context and generation workers with different tensor parallelism, batch sizes, and other parameters.

**Usage:**

The script is called from within `disaggr_torch.slurm`. It takes numerous arguments to define the model, parallelism, and worker configurations for both context and generation phases.

### `gen_server_config.py`

This Python script generates the server configuration YAML file that configures the `trtllm-serve` disaggregated server. It reads hostname information from the work directory and creates a configuration that specifies the URLs for context and generation servers.

**Usage:**

The script is called from within `start_server.sh`. It takes arguments for the number of context and generation servers and the work directory.

### `start_worker.sh`

This script starts a `trtllm-serve disaggregated_mpi_worker`. It is launched by `srun` from the `disaggr_torch.slurm` script on all allocated nodes.

**Arguments:**

1.  `worker_type`: Either "CTX" or "GEN" to specify the worker type.
2.  `worker_index`: Index of the worker instance.
3.  `model_dir`: Path to the model directory.
4.  `worker_port`: Port for the worker to listen on.
5.  `benchmark_mode`: Either "ctx_only" or "gen_only" for benchmarking.
6.  `batch_size`: Batch size for the worker.
7.  `enable_pdl`: `true` or `false` for enabling PDL.
8.  `enable_profiling`: `true` or `false` for enabling profiling.
9.  `work_dir`: Work directory for logs and configuration.
10. `nsys_on`: Path for nsys profiling output (empty string to disable).

### `start_server.sh`

This script starts the `trtllm-serve disaggregated` server. It first generates the server configuration using `gen_server_config.py`, then starts the server process.

**Arguments:**

1.  `num_ctx_servers`: Number of context servers.
2.  `num_gen_servers`: Number of generation servers.
3.  `work_dir`: Work directory for logs and configuration.
4.  `script_dir`: Directory containing the scripts.

### `run_benchmark.sh` and `run_benchmark_nv_sa.sh`

The benchmark can be run using either the default benchmark script (`run_benchmark.sh`) or the NVIDIA SA benchmark script (`run_benchmark_nv_sa.sh`), controlled by the `USE_NV_SA_BENCHMARK` environment variable.

**Default Benchmark Script Arguments (`run_benchmark.sh`):**

1.  `model_name`: Path to the model directory.
2.  `dataset_file`: Path to the dataset file for benchmarking.
3.  `multi_round`: Number of rounds for the benchmark.
4.  `num_gen_servers`: Number of generation servers.
5.  `concurrency_list`: Space-separated list of concurrencies.
6.  `streaming`: `true` or `false` for streaming mode.
7.  `log_path`: Path to the log directory.

**NVIDIA SA Benchmark Script Arguments (`run_benchmark_nv_sa.sh`):**

1.  `model_name`: Path to the model directory.
2.  `isl`: Input sequence length.
3.  `osl`: Output sequence length.
4.  `benchmark_ratio`: Ratio for benchmarking.
5.  `multi_round`: Number of rounds for the benchmark.
6.  `num_gen_servers`: Number of generation servers.
7.  `concurrency_list`: Space-separated list of concurrencies.
8.  `streaming`: `true` or `false` for streaming mode.
9.  `log_path`: Path to the log directory.

## Workflow

1.  Configure the environment variables in `submit.sh` (e.g., sequence lengths, dataset file, model path, container image).
2.  The user runs `./submit.sh` with appropriate parameters for context and generation servers.
3.  `submit.sh` calculates required nodes and submits the job to SLURM using `sbatch disaggr_torch.slurm`.
4.  For each job, SLURM allocates resources and runs `disaggr_torch.slurm`.
5.  `disaggr_torch.slurm` validates all required environment variables.
6.  `disaggr_torch.slurm` starts the container and optionally builds/installs TensorRT-LLM.
7.  `disaggr_torch.slurm` runs `gen_worker_config.py` to create worker configuration files.
8.  `disaggr_torch.slurm` uses `srun` to launch `start_worker.sh` on allocated nodes for context and generation workers.
9.  `disaggr_torch.slurm` generates server configuration using `gen_server_config.py` and starts the server with `start_server.sh`.
10. `disaggr_torch.slurm` runs either `run_benchmark.sh` or `run_benchmark_nv_sa.sh` based on `USE_NV_SA_BENCHMARK` setting.
11. The benchmark script executes the benchmark for each concurrency level.
12. After completion, processes are cleaned up and logs are stored in the specified log directory.
