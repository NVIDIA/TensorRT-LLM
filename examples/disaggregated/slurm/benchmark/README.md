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

This script is used to submit multiple SLURM jobs for running benchmarks with different parameters. It iterates through various configurations and uses `sbatch` to submit `disaggr_torch.slurm` for each one.

**Usage:**

```bash
./submit.sh
```

You can modify the loops in this script to change the parameter space for the benchmark sweep.

### `disaggr_torch.slurm`

This is the core SLURM script for a single benchmark run. It is not meant to be run directly, but rather submitted via `sbatch` (e.g., by `submit.sh`).

It takes the following arguments in order:

1.  `num_ctx_servers`: Number of context servers.
2.  `ctx_tp_size`: Tensor parallel size for context servers.
3.  `ctx_batch_size`: Max batch size for context servers.
4.  `ctx_max_num_tokens`: Max number of tokens for context servers.
5.  `ctx_enable_attention_dp`: `true` or `false` to enable attention DP for context servers.
6.  `num_gen_servers`: Number of generation servers.
7.  `gen_tp_size`: Tensor parallel size for generation servers.
8.  `gen_batch_size`: Max batch size for generation servers.
9.  `gen_max_num_tokens`: Max number of tokens for generation servers.
10. `gen_enable_attention_dp`: `true` or `false` to enable attention DP for generation servers.
11. `gen_gpu_memory_fraction`: GPU memory fraction for generation servers.
12. `concurrency_list`: A space-separated list of concurrencies to test (e.g., "1 2 4 8").
13. `sub_file`: A subdirectory name for logs.

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
5.  `benchmark_mode`: Benchmark mode setting.
6.  `concurrency`: Concurrency level.
7.  `enable_pdl`: `true` or `false`.
8.  `work_dir`: Work directory for logs and configuration.
9.  `nsys_on`: Whether to enable nsys profiling.

### `start_server.sh`

This script starts the `trtllm-serve disaggregated` server. It first generates the server configuration using `gen_server_config.py`, then starts the server process.

**Arguments:**

1.  `num_ctx_servers`: Number of context servers.
2.  `num_gen_servers`: Number of generation servers.
3.  `work_dir`: Work directory for logs and configuration.
4.  `script_dir`: Directory containing the scripts.

### `run_benchmark.sh`

This script orchestrates the execution of the benchmark client. It waits for the configuration files to be created and for the server's `/health` endpoint to respond, then it runs the benchmark.

**Arguments:**

1.  `isl`: Input sequence length.
2.  `osl`: Output sequence length.
3.  `multi_round`: Number of rounds for the benchmark.
4.  `model_name`: Name of the model being benchmarked.
5.  `concurrency_list`: Space-separated list of concurrencies.
6.  `streaming`: `true` or `false`.
7.  `log_path`: Path to the log directory.

## Workflow

1.  Make sure that SLURM parameters are correctly set in `disaggr_torch.slurm`.
2.  The user runs `./submit.sh`.
3.  `submit.sh` submits one or more jobs to SLURM by calling `sbatch disaggr_torch.slurm` with different parameters.
4.  For each job, SLURM allocates resources and runs `disaggr_torch.slurm`.
5.  `disaggr_torch.slurm` runs `gen_worker_config.py` to create worker configuration files.
6.  `disaggr_torch.slurm` uses `srun` to launch `start_worker.sh` on all nodes, starting the MPI workers for both context and generation phases.
7.  `disaggr_torch.slurm` starts the main `trtllm-serve` process using `start_server.sh`, which generates the server configuration using `gen_server_config.py`.
8.  `disaggr_torch.slurm` runs `run_benchmark.sh` which waits for the server to be ready.
9.  `run_benchmark.sh` executes the benchmark for each concurrency level specified.
10.  After the benchmark, `run_benchmark.sh` and `disaggr_torch.slurm` attempt to kill the server and worker processes.
11. Logs for each run are stored in a subdirectory specified by the `sub_file` parameter.
