# TensorRT-LLM Benchmark Test System

Benchmarking scripts for TensorRT-LLM serving performance tests with configuration-driven test cases and CSV report generation.

## Overview

- Run performance benchmarks across multiple model configurations
- Manage test cases through YAML configuration files
- Generate comprehensive CSV reports with complete test case coverage
- Support selective execution of specific test cases

## Scripts Overview

### 1. `benchmark_config.yaml` - Test Case Configuration
**Purpose**: Defines all benchmark test cases in a structured YAML format.

**Structure**:
```yaml
test_cases:
  - id: 1
    model: "70B-FP8"
    gpus: 1
    tp: 1
    ep: 1
    attn_backend: "TRTLLM"
    moe_backend: ""
    enable_attention_dp: false
    free_gpu_mem_fraction: 0.9
    max_batch_size: 512
    isl: 1024
    osl: 1024
    max_num_tokens: 16384
    moe_max_num_tokens: ""
    concurrency_iterations:
      - [1, 10]
      - [8, 10]
      - [64, 5]
      - [512, 2]
```

**Configuration Fields**:
- `id`: Unique identifier for the test case
- `model`: Model name (e.g., "70B-FP8", "Scout-FP4")
- `gpus`: Number of GPUs to use
- `tp`: Tensor parallelism size
- `ep`: Expert parallelism size
- `attn_backend`: Attention backend ("TRTLLM", "FLASHINFER")
- `moe_backend`: MoE backend ("DEEPGEMM", "TRTLLM", "CUTLASS", "")
- `enable_attention_dp`: Enable attention data parallelism
- `free_gpu_mem_fraction`: GPU memory fraction to reserve
- `max_batch_size`: Maximum batch size
- `isl`: Input sequence length
- `osl`: Output sequence length
- `max_num_tokens`: Maximum number of tokens
- `moe_max_num_tokens`: Maximum number of tokens for MoE
- `concurrency_iterations`: List of [concurrency, iteration] pairs


### 2. `run_benchmark_serve.py` - Main Benchmark Runner
**Purpose**: Executes performance benchmarks based on YAML configuration files.

**Usage**:
```bash
python run_benchmark_serve.py --output_folder <output_folder> --config_file <config_file> [--skip <skip_pattern>] [--select <select_pattern>]
```

**Arguments**:
- `--output_folder`: Directory to store benchmark results (required)
- `--config_file`: Path to YAML configuration file (required)
- `--skip`: Skip pattern for specific test cases/concurrencies (optional, default: no skipping)
- `--select`: Select pattern for specific test cases/concurrencies (optional, default: all test cases)

**Examples**:
```bash
# Run all test cases
python run_benchmark_serve.py --output_folder results --config_file benchmark_config.yaml --skip default --select default

# Skip specific test cases
python run_benchmark_serve.py --output_folder results --config_file benchmark_config.yaml --skip "2-1,4"

# Run specific concurrencies from specific test cases
python run_benchmark_serve.py --output_folder results --config_file benchmark_config.yaml --select "1,2-3"

```

**Skip Pattern**:
Format: `"test_case1,test_case2,test_case3"` or `"test_case1-concurrency1,test_case2-concurrency3"`
- `"2,4"`: Skip test cases 2 and 4 entirely
- `"2-1,4-2"`: Skip test case 2's 1st concurrency and test case 4's 2nd concurrency
- `"default"` or empty: No skipping (default)

**Select Pattern**:
Format: `"test_case1,test_case2,test_case3"` or `"test_case1-concurrency1,test_case2-concurrency3"`
- `"1,3,5"`: Run only test cases 1, 3, and 5 (all concurrencies)
- `"1-1,2-3"`: Run test case 1's 1st concurrency and test case 2's 3rd concurrency
- `"default"` or empty: Run all test cases (default)


### 3. `parse_benchmark_results.py` - Results Parser
**Purpose**: Parses benchmark log files and generates comprehensive CSV reports with all test cases from the configuration file.

**Usage**:
```bash
python parse_benchmark_results.py --input_folder <input_folder> --output_csv <output_csv> --config_file <config_file>
```

**Arguments**:
- `input_folder`: Folder containing benchmark log files (serve.*.log) (required)
- `output_csv`: Output CSV filename for the results table (required)
- `config_file`: Path to benchmark_config.yaml file (required)

**Examples**:
```bash
python parse_benchmark_results.py --config_file ./benchmark_logs --output_csv results.csv --input_folder ./benchmark_config.yaml

```

### 4. `benchmark-serve.sh` - SLURM Job Script
**Usage**:
```bash
sbatch benchmark-serve.sh [IMAGE] [bench_dir] [output_dir] [select_pattern] [skip_pattern]
```

**Parameters**:
- `IMAGE`: Docker image (default: tensorrt-llm-staging/release:main-x86_64)
- `bench_dir`: Directory containing config file and benchmark scripts (default: current directory)
- `output_dir`: Directory containing output logs and csv. (default: current directory)
- `select_pattern`: Select pattern (default: default - all test cases)
- `skip_pattern`: Skip pattern (default: default - no skipping)

**Examples**:
```bash

bench_dir="/path/to/benchmark/scripts"
output_dir="/path/to/store/output/files"
sbatch --reservation=RES--COM-3970 --qos=reservation -D ${output_dir} ${bench_dir}/benchmark-serve.sh urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm-staging/release:main-x86_64 ${bench_dir} ${output_dir} "1-1" ""

```
