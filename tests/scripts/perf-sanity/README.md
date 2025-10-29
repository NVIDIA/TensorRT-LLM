# TensorRT-LLM Benchmark Test System

Benchmarking scripts for TensorRT-LLM serving performance tests with configuration-driven test cases and CSV report generation.

## Overview

- Run performance benchmarks across multiple model configurations
- Manage test cases through YAML configuration files
- Support selective execution of specific test cases

## Scripts Overview

### 1. `benchmark_config.yaml` - Test Case Configuration
**Purpose**: Defines all benchmark test cases in a structured YAML format.

**Structure**:
```yaml
server_configs:
  - name: "r1_fp4_dep4"
    model_name: "deepseek_r1_0528_fp4"
    tp: 4
    ep: 4
    pp: 1
    attention_backend: "TRTLLM"
    moe_backend: "CUTLASS"
    moe_max_num_tokens: ""
    enable_attention_dp: true
    enable_chunked_prefill: false
    max_num_tokens: 2176
    disable_overlap_scheduler: false
    kv_cache_dtype: "fp8"
    enable_block_reuse: false
    free_gpu_memory_fraction: 0.8
    max_batch_size: 256
    enable_padding: true
    client_configs:
      - name: "con1_iter1_1024_1024"
        concurrency: 1
        iterations: 1
        isl: 1024
        osl: 1024
        random_range_ratio: 0.0
      - name: "con8_iter1_1024_1024"
        concurrency: 8
        iterations: 1
        isl: 1024
        osl: 1024
        random_range_ratio: 0.0

  - name: "r1_fp4_tep4"
    model_name: "deepseek_r1_0528_fp4"
    tp: 4
    ep: 4
    pp: 1
    attention_backend: "TRTLLM"
    moe_backend: "CUTLASS"
    moe_max_num_tokens: ""
    enable_attention_dp: false
    enable_chunked_prefill: false
    max_num_tokens: 2176
    disable_overlap_scheduler: false
    kv_cache_dtype: "fp8"
    enable_block_reuse: false
    free_gpu_memory_fraction: 0.8
    max_batch_size: 256
    enable_padding: true
    client_configs:
      - name: "con1_iter1_1024_1024"
        concurrency: 1
        iterations: 1
        isl: 1024
        osl: 1024
        random_range_ratio: 0.0
      - name: "con8_iter1_1024_1024"
        concurrency: 8
        iterations: 1
        isl: 1024
        osl: 1024
        random_range_ratio: 0.0
```

### 2. `run_benchmark_serve.py` - Main Benchmark Runner
**Purpose**: Executes performance benchmarks based on YAML configuration files.

**Usage**:
```bash
python run_benchmark_serve.py --log_folder <log_folder> --config_file <config_file> [--select <select_pattern>] [--timeout 5400]
```

**Arguments**:
- `--log_folder`: Directory to store benchmark logs (required)
- `--config_file`: Path to YAML configuration file (required)
- `--select`: Select pattern for specific Server and Client Config. (optional, default: all test cases)
- `--timeout`: Timeout for server setup. (optional, default: 3600 seconds)

**Examples**:
```bash
# Select
python run_benchmark_serve.py --log_folder ./results --config_file benchmark_config.yaml --select "r1_fp4_dep4:con8_iter1_1024_1024,r1_fp4_tep4:con1_iter1_1024_1024"

```

### 3. `parse_benchmark_results.py` - Results Parser
**Purpose**: Print log's perf.

**Arguments**:
- `--log_folder`: Directory to store benchmark logs (required)

**Usage**:
```bash
python parse_benchmark_results.py --log_folder <log_folder>
```


### 4. `benchmark-serve.sh` - SLURM Job Script
**Usage**:
```bash
sbatch benchmark-serve.sh [IMAGE] [bench_dir] [log_folder] [select_pattern]
```

**Parameters**:
- `IMAGE`: Docker image (default: tensorrt-llm-staging/release:main-x86_64)
- `bench_dir`: Directory containing config file and benchmark scripts (default: current directory)
- `log_folder`: Directory containing output logs and csv. (default: current directory)
- `select_pattern`: Select pattern (default: default - all test cases)

**Examples**:
```bash

bench_dir="/path/to/benchmark/scripts"
log_folder="/path/to/store/output/files"
sbatch --reservation=RES--COM-3970 --qos=reservation -D ${log_folder} ${bench_dir}/benchmark-serve.sh urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm-staging/release:main-x86_64 ${bench_dir} ${log_folder} "r1_fp4_dep4:con8_iter1_1024_1024,r1_fp4_tep4:con1_iter1_1024_1024"

```
