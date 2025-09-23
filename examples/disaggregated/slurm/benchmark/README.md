# Disaggregated Inference Benchmark Scripts

This directory contains scripts to run disaggregated inference benchmarks using TensorRT-LLM and SLURM.

## Overview

The benchmarking process is orchestrated through a set of shell scripts and Python scripts that work together:

1.  `submit.sh`: The main entry point for submitting benchmark jobs to SLURM. It runs a parameter sweep by calling `sbatch` with different configurations. Supports both context and generation server configurations with pipeline parallelism.
2.  `disaggr_torch.slurm`: The SLURM script that sets up and runs a single benchmark experiment. It launches a container, optionally builds TensorRT-LLM from source, generates configuration files, starts the server and workers, and runs the benchmark client.
3.  `gen_worker_config.py`: A Python script that generates the worker configuration YAML file needed by `trtllm-serve`. It determines the worker configuration based on SLURM environment variables and script arguments, supporting both context and generation workers with tensor/pipeline parallelism.
4.  `gen_server_config.py`: A Python script that generates the server configuration YAML file needed by `trtllm-serve`. It determines the server configuration based on the number of context and generation servers.
5.  `start_worker.sh`: A shell script responsible for starting disaggregated workers using `trtllm-serve` on each allocated machine. Supports both context and generation workers with profiling capabilities.
6.  `start_server.sh`: A shell script responsible for starting disaggregated server using `trtllm-serve` on each allocated machine.
7.  `run_benchmark.sh`: A shell script that waits for the server to be healthy and then runs the actual benchmark client. Supports streaming mode and various metrics collection.

## File Descriptions

### `submit.sh`

This script is used to submit SLURM jobs for running benchmarks with specific configurations. It provides helper functions to calculate required nodes and submit jobs with the right parameters.

The script includes a user configuration section where you can set various parameters:

1. SLURM Configuration:
   - `partition`: SLURM partition to use
   - `account`: SLURM account to use
   - `job_time`: Job time limit
   - `job_name`: Name of the job

2. Hardware Configuration:
   - `gpus_per_node`: Number of GPUs per node (default: 4)

3. Benchmark Configuration:
   - `use_nv_sa_benchmark`: Whether to use NVIDIA SA benchmark script
   - `isl`: Input sequence length
   - `osl`: Output sequence length
   - `multi_round`: Number of benchmark rounds
   - `benchmark_ratio`: Benchmark ratio
   - `streaming`: Enable streaming mode
   - `cache_max_tokens`: Cache transceiver max tokens
   - `dataset_file`: Path to dataset file for benchmarking

4. Environment Configuration:
   - `mount_dir`: Directory to mount in container
   - `container_image`: Path to container image
   - `model_path`: Path to model directory
   - `trtllm_repo`: Path to TensorRT-LLM repository
   - `build_wheel`: Whether to build TensorRT-LLM from source

5. Workspace and Profiling Configuration:
   - `work_dir`: Path to work directory
   - `nsys_on`: Enable nsys profiling (true/false)

**Usage:**

The script provides a `run_single` function that takes all the necessary parameters for both context and generation servers. Example usage:

```bash
#      CTX: num tp_size pp_size batch tokens attn_dp gpu_frac  GEN: num tp_size pp_size batch tokens attn_dp gpu_frac eplb mtp concurrency
run_single  1   4       1       4     4608   true    0.85           1   8       1       32    128    false   "0.9"    0    3    "16"
```

The script automatically calculates the required number of nodes based on the tensor parallel size and server count.

### `disaggr_torch.slurm`

This is the core SLURM script for a single benchmark run. It is not meant to be run directly, but rather submitted via `sbatch` (e.g., by `submit.sh`).

It takes the following arguments in order:

1.  `num_ctx_servers`: Number of context servers.
2.  `ctx_tp_size`: Tensor parallel size for context servers.
3.  `ctx_pp_size`: Pipeline parallel size for context servers.
4.  `ctx_batch_size`: Max batch size for context servers.
5.  `ctx_max_num_tokens`: Max number of tokens for context servers.
6.  `ctx_enable_attention_dp`: `true` or `false` to enable attention DP for context servers.
7.  `ctx_gpu_frac`: GPU memory fraction for context servers.
8.  `num_gen_servers`: Number of generation servers.
9.  `gen_tp_size`: Tensor parallel size for generation servers.
10. `gen_pp_size`: Pipeline parallel size for generation servers.
11. `gen_batch_size`: Max batch size for generation servers.
12. `gen_max_num_tokens`: Max number of tokens for generation servers.
13. `gen_enable_attention_dp`: `true` or `false` to enable attention DP for generation servers.
14. `gen_gpu_memory_fraction`: GPU memory fraction for generation servers.
15. `eplb_num_slots`: Number of slots for eplb.
16. `mtp_size`: Number of nextn layers for MTP.
17. `concurrency_list`: Space-separated list of concurrencies for benchmarking.
18. `gpus_per_node`: Number of GPUs per node.
19. `use_nv_sa_benchmark`: Whether to use NVIDIA SA benchmark script.
20. `isl`: Input sequence length.
21. `osl`: Output sequence length.
22. `multi_round`: Number of benchmark rounds.
23. `benchmark_ratio`: Benchmark ratio.
24. `streaming`: Enable streaming mode.
25. `cache_max_tokens`: Cache transceiver max tokens.
26. `dataset_file`: Path to dataset file for benchmarking.
27. `mount_dir`: Directory to mount in container.
28. `container_image`: Path to container image.
29. `model_path`: Path to model directory.
30. `trtllm_repo`: Path to TensorRT-LLM repository.
31. `build_wheel`: Whether to build TensorRT-LLM from source.
32. `work_dir`: Path to work directory.
33. `nsys_on`: Enable nsys profiling.

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
5.  `enable_pdl`: `true` or `false` for enabling PDL.
6.  `work_dir`: Work directory for logs and configuration.
7.  `nsys_on`: Enable nsys profiling (true/false).

### `start_server.sh`

This script starts the `trtllm-serve disaggregated` server. It first generates the server configuration using `gen_server_config.py`, then starts the server process.

**Arguments:**

1.  `num_ctx_servers`: Number of context servers.
2.  `num_gen_servers`: Number of generation servers.
3.  `work_dir`: Work directory for logs and configuration.
4.  `script_dir`: Directory containing the scripts.

### `run_benchmark.sh` and `run_benchmark_nv_sa.sh`

The benchmark can be run using either the default benchmark script (`run_benchmark.sh`) or the NVIDIA SA benchmark script (`run_benchmark_nv_sa.sh`), controlled by the `use_nv_sa_benchmark` parameter.

**Default Benchmark Script Arguments (`run_benchmark.sh`):**

1.  `model_name`: Path to the model directory.
2.  `dataset_file`: Path to the dataset file for benchmarking.
3.  `multi_round`: Number of rounds for the benchmark.
4.  `num_gen_servers`: Number of generation servers.
5.  `concurrency_list`: Space-separated list of concurrencies.
6.  `streaming`: `true` or `false` for streaming mode.
7.  `log_path`: Path to the log directory.

The script supports various metrics collection including:
- TTFT (Time to First Token)
- TPOT (Throughput Over Time)
- ITL (Inter-Token Latency)
- E2EL (End-to-End Latency)

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

1.  Configure the parameters in `submit.sh` (e.g., SLURM settings, sequence lengths, dataset file, model path, container image).
2.  The user runs `./submit.sh` with appropriate parameters for context and generation servers.
3.  `submit.sh` calculates required nodes based on tensor/pipeline parallelism and submits the job to SLURM using `sbatch disaggr_torch.slurm`.
4.  For each job, SLURM allocates resources and runs `disaggr_torch.slurm`.
5.  `disaggr_torch.slurm` validates all required parameters.
6.  `disaggr_torch.slurm` starts the container and optionally builds/installs TensorRT-LLM from source.
7.  `disaggr_torch.slurm` runs `gen_worker_config.py` to create worker configuration files with tensor/pipeline parallelism settings.
8.  `disaggr_torch.slurm` uses `srun` to launch `start_worker.sh` on allocated nodes for context and generation workers.
9.  `disaggr_torch.slurm` generates server configuration using `gen_server_config.py` and starts the server with `start_server.sh`.
10. `disaggr_torch.slurm` runs either `run_benchmark.sh` or `run_benchmark_nv_sa.sh` based on `use_nv_sa_benchmark` setting.
11. The benchmark script executes the benchmark for each concurrency level, collecting various metrics.
12. After completion, processes are gracefully terminated and logs are stored in the specified log directory.
