# TRT-LLM Benchmark System

A benchmarking system for TensorRT-LLM serving performance evaluation with configuration-driven test cases and CSV report generation.

## Overview

- Run performance benchmarks across multiple model configurations
- Manage test cases through YAML configuration files
- Generate CSV reports with detailed metrics
- Support selective execution of specific test cases
- Run efficiently on multi-GPU nodes

## Scripts Overview

### 1. `run_benchmark_serve.py` - Main Benchmark Runner
**Purpose**: Executes performance benchmarks based on YAML configuration files.

**Features**:
- Reads test cases from YAML configuration files
- Supports selective execution with skip/select patterns
- Efficient server management (start once per test case, run multiple benchmarks, then kill)
- Robust error handling and graceful failure recovery

**Usage**:
```bash
python run_benchmark_serve.py --output_folder <output_folder> --commit <commit> --config_file <config_file> [--skip <skip_pattern>] [--select <select_pattern>]
```

**Arguments**:
- `--output_folder`: Directory to store benchmark results (required)
- `--commit`: Git commit ID for tracking (required)
- `--config_file`: Path to YAML configuration file (required)
- `--skip`: Skip pattern for specific test cases/concurrencies (optional)
- `--select`: Select pattern for specific test cases (optional)

**Examples**:
```bash
# Run all test cases
python run_benchmark_serve.py --output_folder results --commit abc123 --config_file benchmark_config.yaml

# Skip specific test cases
python run_benchmark_serve.py --output_folder results --commit abc123 --config_file benchmark_config.yaml --skip "2,4"

# Run only specific test cases
python run_benchmark_serve.py --output_folder results --commit abc123 --config_file benchmark_config.yaml --select "1,3,5"

# Skip specific concurrency levels
python run_benchmark_serve.py --output_folder results --commit abc123 --config_file benchmark_config.yaml --skip "2-1,4-0"
```

### 2. `benchmark_config.yaml` - Test Case Configuration
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

### 3. `parse_benchmark_results.py` - Results Parser
**Purpose**: Parses benchmark log files and generates CSV reports.

**Features**:
- Extracts configuration information from log files
- Parses performance metrics (throughput, latency)
- Generates structured CSV tables
- Relies on log content for accurate configuration data

**Usage**:
```bash
python parse_benchmark_results.py <folder_name>
```

**Output**:
- CSV file named `<folder_name>.csv` in the parent directory
- Console output with detailed configuration and metrics
- Columns include: model_name, GPUs, TP, EP, attn_backend, moe_backend, enable_attention_dp, free_gpu_mem_fraction, max_batch_size, ISL, OSL, max_num_tokens, moe_max_num_tokens, Concurrency, TPS/System, TPS/User

**Example**:
```bash
python parse_benchmark_results.py benchmark.run.12345.2024-01-15-10:30:00.abc123
```

### 4. `benchmark-serve.sh` - SLURM Job Script
**Purpose**: SLURM job script for running benchmarks on GPU nodes.

**Features**:
- Requests 8-GPU node for 8 hours
- Runs benchmarks in Docker container
- Automatically generates CSV reports
- Supports Slack notifications (if available)

**Usage**:
```bash
sbatch benchmark-serve.sh [image] [commit] [config_file] [skip_pattern] [select_pattern]
```

**Parameters**:
- `image`: Docker image (default: tensorrt-llm-staging/release:main-x86_64)
- `config_file`: YAML config file (default: benchmark_config.yaml)
- `skip_pattern`: Skip pattern (default: empty)
- `select_pattern`: Select pattern (default: empty)

**Examples**:
```bash
# Run with defaults
sbatch benchmark-serve.sh

# Run with custom parameters
sbatch benchmark-serve.sh custom_image custom_commit custom_config.yaml "2,4" "1,3,5"
```

## Pattern Syntax

### Skip Pattern
Format: `"test_case[-concurrency_index]"`
- `"2,4"`: Skip test cases 2 and 4 entirely
- `"2-1,4-0"`: Skip test case 2's 1st concurrency and test case 4's 0th concurrency

### Select Pattern
Format: `"test_case1,test_case2,test_case3"`
- `"1,3,5"`: Run only test cases 1, 3, and 5
- `""` (empty): Run all test cases (default)

## Workflow

1. **Prepare Configuration**: Edit `benchmark_config.yaml` to define test cases
2. **Run Benchmarks**: 
   ```bash
   # Local
   python run_benchmark_serve.py --output_folder results --commit abc123 --config_file benchmark_config.yaml
   # SLURM
   sbatch benchmark-serve.sh
   ```
3. **Generate Reports**: `python parse_benchmark_results.py results`

## Output Structure

```
results/
├── serve.model1.tp1.ep1.attnTRTLLM.moe.gpu0.9.batch512.isl1024.osl1024.tokens16384.moetokens.concurrency1.iter10.log
├── serve.model1.tp1.ep1.attnTRTLLM.moe.gpu0.9.batch512.isl1024.osl1024.tokens16384.moetokens.concurrency8.iter10.log
├── serve.model2.tp4.ep1.attnTRTLLM.moe.gpu0.9.batch1024.isl1024.osl1024.tokens16384.moetokens.concurrency1.iter10.log
├── trtllm-serve.model1.tp1.ep1.attnTRTLLM.moe.gpu0.9.batch512.isl1024.osl1024.tokens16384.moetokens.log
├── trtllm-serve.model2.tp4.ep1.attnTRTLLM.moe.gpu0.9.batch1024.isl1024.osl1024.tokens16384.moetokens.log
└── results.csv (generated by parse script)
```

## Model Support

Supported models: 70B-FP8/FP4, Scout-FP8/FP4, R1-FP8/FP4

Configurable parameters: Tensor/Expert parallelism, attention/MoE backends, memory fractions, batch sizes, sequence lengths
