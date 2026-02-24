# TensorRT-LLM Disaggregated Benchmark Framework

A YAML-based testing framework for TensorRT-LLM disaggregated serving performance and accuracy benchmarks.

## Overview

This framework provides a simple, maintainable approach to benchmark testing using YAML configuration files. Each test configuration is defined in a separate YAML file, with automatic test discovery and execution through pytest.

## Key Features

- **YAML Configuration**: Each test has its own independent YAML configuration file
- **Automatic Test Discovery**: Tests are automatically discovered from the config directory structure
- **Default Metrics**: Built-in default metrics configuration for common test scenarios
- **GPU Filtering**: Automatically filters tests based on hardware compatibility
- **Flexible Override**: Override default configurations as needed for special cases
- **Test Categories**: Support for both performance (perf) and accuracy tests
- **Multiple Test Types**: Support for disagg (disaggregated) and wideep architectures

## Directory Structure

```
test_configs/
├── disagg/                    # Disaggregated serving tests
│   ├── perf/                  # Performance tests
│   └── accuracy/              # Accuracy tests (optional)
└── wideep/                    # Wide-deep tests
    ├── perf/
    └── accuracy/
```

## YAML Configuration

### Minimal Configuration Example

```yaml
metadata:
  model_name: "deepseek-r1-fp4"
  precision: "fp4"
  supported_gpus: ["GB200"]

slurm:
  partition: "<partition>"
  account: "<account>"
  job_time: "02:00:00"

benchmark:
  mode: "e2e"
  streaming: true
  concurrency_list: "1 2 4 8 16 36"
  input_length: 1024
  output_length: 1024

hardware:
  gpus_per_node: 4
  num_ctx_servers: 1
  num_gen_servers: 4

environment:
  container_mount: "<container_mount>"
  container_image: "<container_image>"
  model_path: "<model_path>"

worker_config:
  gen:
    tensor_parallel_size: 8
    moe_expert_parallel_size: 8
    max_batch_size: 32
    max_num_tokens: 32
    max_seq_len: 2251
    # ... other gen worker configs
  
  ctx:
    tensor_parallel_size: 4
    moe_expert_parallel_size: 4
    max_batch_size: 4
    max_num_tokens: 4608
    max_seq_len: 2251
    # ... other ctx worker configs
```

### Custom Metrics (Optional)

Most tests use default metrics. To customize:

```yaml
benchmark:
  metrics:
    log_file: "custom_benchmark.log"
    extractor_pattern: "Custom Pattern:\s+([0-9.]+)"
    metric_names: ["CUSTOM_METRIC"]
```

## GPU Support

Currently supports **OCI GB200** only. The framework is designed to support additional GPU types in the future.

All configurations must specify:
```yaml
metadata:
  supported_gpus: ["GB200"]
```

## Configuration Validation

The framework validates configurations before execution:

1. **gen_max_tokens**: Must equal `gen_max_batch_size * (mtp_size + 1)` when MTP is enabled
2. **streaming**: Must be `true`
3. **max_seq_len**: Both ctx and gen must be > (input_length + output_length)

## Running Tests

### Run all tests
```bash
poetry run pytest --disagg test_disagg.py -s -vv
```

### Run from test list
```bash
poetry run pytest --disagg test_disagg.py -s -vv --disagg-test-list=./testlist/disagg.txt
```

### Run specific tests
```bash
# Run only performance tests
poetry run pytest --disagg test_disagg.py -s -vv -m perf

# Run only accuracy tests
poetry run pytest --disagg test_disagg.py -s -vv -m accuracy

# Run specific test by ID
poetry run pytest --disagg test_disagg.py -s -vv -k "deepseek-r1-fp4_1k1k"
```

## Batch Job Submission

The framework supports automatic batch job submission to maximize parallelism in SLURM cluster environments. Instead of submitting jobs one-by-one, it groups test cases into batches and submits entire batches when needed.

### Quick Start

**Default batch size (5 jobs per batch):**
```bash
# Run all tests with default batching
poetry run pytest --disagg test_disagg.py -s -vv

# Run with test list
poetry run pytest --disagg test_disagg.py -s -vv --disagg-test-list=./testlist/all.txt
```

**Custom batch size:**
```bash
# Set batch size via command line
poetry run pytest --disagg test_disagg.py -s -vv --disagg-batch-size=10

# Set batch size via environment variable
export DISAGG_BATCH_SIZE=20
poetry run pytest --disagg test_disagg.py -s -vv

# Submit all jobs at once (unlimited batch)
poetry run pytest --disagg test_disagg.py -s -vv --disagg-batch-size=0
```

### How Batch Submission Works

```
Pytest Collection Phase:
  - Collects all test cases (e.g., 100 tests)
  - BatchManager splits them into batches (e.g., 20 batches of 5)

Pytest Execution Phase:
  Test 0 runs:
    -> Triggers submission of Batch 0 (jobs 0-4)
    -> Waits for job 0 to complete
  
  Test 1-4 run:
    -> Batch 0 already submitted, directly wait for completion
  
  Test 5 runs:
    -> Triggers submission of Batch 1 (jobs 5-9)
    -> Waits for job 5 to complete
  
  ... and so on
```

### Key Benefits

- **Parallel Execution**: All jobs in a batch run simultaneously on SLURM cluster
- **Reduced Wait Time**: Total time ≈ MAX(job time) instead of SUM(job times)
- **Automatic Management**: No need to manually split test lists
- **Lazy Loading**: Only submits batches when needed

### Configuration Options

**Priority**: Command line option > Environment variable > Default (5)

**Examples:**

```bash
# Small batch for quick testing
poetry run pytest --disagg test_disagg.py -s -vv --disagg-batch-size=3 \
  --disagg-test-list=./testlist/debug.txt

# Large batch for production
poetry run pytest --disagg test_disagg.py -s -vv --disagg-batch-size=50 \
  --disagg-test-list=./testlist/all.txt

# Submit all at once
poetry run pytest --disagg test_disagg.py -s -vv --disagg-batch-size=0
```

### Timeout Configuration

The default timeout for waiting for job completion is **10 hours (36000 seconds)**, which accounts for:
- SLURM queue wait time
- Job execution time
- Buffer for delays

### Performance Comparison

**Before (Sequential Submission):**
```
Case 1: submit + wait (1.5h) = 1.5h
Case 2: submit + wait (1.5h) = 1.5h
Case 3: submit + wait (1.5h) = 1.5h
...
Total: 50 × 1.5h = 75 hours
```

**After (Batch Submission, batch_size=50):**
```
Batch 0 (50 jobs): submitted in parallel
  Case 1: wait (1.5h)
  Case 2-50: wait (0s, already done)

Total: ~1.5 hours
```

**Speedup: 50x**

### Troubleshooting

**Check BatchManager initialization:**
```
======================================================================
Batch Manager Initialized
Batch size: 5 jobs per batch
======================================================================

Total test configs: 20
Total batches: 4
```

**Monitor batch submission:**
```
======================================================================
Submitting Batch 0
Range: [0:5] (5 jobs)
======================================================================

  [  1/5] Job 1234 <- test_config_id_1
  [  2/5] Job 1235 <- test_config_id_2
  ...
```

**If jobs timeout frequently:**
- Check SLURM queue status
- Consider reducing batch size to avoid resource contention
- Verify that timeout (36000s) is sufficient for your workload

## Test Naming Convention

Tests are automatically named using the format:
```
{test_type}_{category}_{config_filename}
```

Example: `disagg_perf_deepseek-r1-fp4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp0_ccb-NIXL`

## File Naming Convention

Configuration files should follow this format:
```
{model}_{benchmark_type}_{config_details}.yaml
```

Examples:
- `deepseek-r1-fp4_1k1k_ctx1_gen1_dep32_bs32_eplb0_mtp0_ccb-NIXL.yaml`
- `deepseek-r1-fp4_8k1k_ctx1_gen3_tep8_bs32_eplb0_mtp0_ccb-UCX.yaml`

Where:
- `1k1k`, `8k1k`: Input/output lengths (1024/1024, 8192/1024)
- `ctx1_gen1`: Context and generation server counts
- `dep32` or `tep8`: Data parallel (dep) or tensor parallel (tep) configuration
- `bs32`: Batch size
- `eplb0`: Expert parallel load balancing slots
- `mtp0`: Multi-token prediction layers
- `ccb-NIXL` or `ccb-UCX`: Communication backend

## Key Configuration Fields

### Metadata
- `model_name`: Model identifier
- `precision`: Model precision (fp4, fp8, etc.)
- `supported_gpus`: List of compatible GPU types

### Benchmark
- `mode`: Benchmark mode (e2e, gen_only, ctx_only)
- `streaming`: Enable streaming (must be true)
- `input_length`, `output_length`: Sequence lengths
- `concurrency_list`: Concurrency levels to test

### Worker Config
- `tensor_parallel_size`: Tensor parallelism degree
- `moe_expert_parallel_size`: MoE expert parallelism
- `max_batch_size`: Maximum batch size
- `max_num_tokens`: Maximum tokens per batch
- `max_seq_len`: Maximum sequence length
- `speculative_config`: Multi-token prediction settings (optional)

## Test Output

Test results are saved to:
- Performance metrics: `{OUTPUT_PATH}/perf_script_test_results.csv`
- Test logs: `{OUTPUT_PATH}/disagg_benchmark_{timestamp}.log`

## Environment Variables

- `GPU_TYPE`: Current GPU type (default: GB200)
- `OUTPUT_PATH`: Directory for test results and logs
- `WORK_DIR`: Working directory for benchmark execution
- `DISAGG_BATCH_SIZE`: Default batch size for job submission (default: 5)
- `DEBUG_MODE`: Enable debug mode (set to "1" to skip job submission)
- `DEBUG_JOB_ID`: Job ID to use in debug mode

## Debug Mode

For local testing without SLURM submission:

```bash
export DEBUG_MODE=1
export DEBUG_JOB_ID=12345
poetry run pytest --disagg test_disagg.py -s -vv
```

## Architecture

The framework consists of:

1. **ConfigLoader**: Scans and loads YAML configurations
2. **ConfigValidator**: Validates configuration correctness
3. **BatchManager**: Manages batch job submission for parallel execution
4. **JobManager**: Handles SLURM job submission and monitoring
5. **LogParser**: Extracts metrics from benchmark logs
6. **TestCaseTracker**: Tracks test execution timing
7. **ResultSaver**: Saves results to CSV

## Benefits

- **Simple**: YAML-based configuration, no code changes needed
- **Maintainable**: Each test is a separate file
- **Flexible**: Override defaults only when needed
- **Scalable**: Easy to add new tests and models
- **Reliable**: Automatic validation before execution
- **Traceable**: Comprehensive logging and result tracking

## Adding New Tests

1. Create a new YAML file in `test_configs/{test_type}/{category}/`
2. Configure the test parameters
3. Run pytest - the test will be automatically discovered

No code changes required!

---

For detailed configuration options and advanced usage, refer to the inline comments in the YAML configuration files.
