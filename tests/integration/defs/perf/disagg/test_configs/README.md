# Test Configurations

This directory contains YAML-based test configuration files for disaggregated benchmark testing.

## Directory Structure

```
test_configs/
└── disagg/                    # Disaggregated test type
    ├── perf/                  # Performance test configurations
    └── accuracy/              # Accuracy test configurations (to be added)
```

## Configuration File Naming Convention

File name format: `{model}_{benchmark_type}_{config_details}.yaml`

- **model**: Model name (e.g., deepseek-r1-fp4, Qwen3-8B-FP8)
- **benchmark_type**: Benchmark type (e.g., 1k1k, 8k1k)
- **config_details**: Configuration details (e.g., tep8_bs32_mtp3_nixl)

### Configuration Details Explanation

- **tep/dep**: Tensor Parallel (TEP) or Data Parallel with attention_dp (DEP)
- **number**: TP size (e.g., tep8 means TP=8, dep16 means TP=16+DP)
- **bs**: Batch Size (e.g., bs32 means batch_size=32)
- **mtp**: MTP layers (e.g., mtp3 means 3 MTP layers, no mtp means not used)
- **nixl/ucx**: Cache transceiver backend (NIXL or UCX)

## Generated Configuration File List

### DeepSeek R1 FP4 (1k1k)

1. `deepseek-r1-fp4_1k1k_tep8_bs32_mtp3_nixl.yaml` - TEP8, BS32, MTP3, NIXL
2. `deepseek-r1-fp4_1k1k_tep8_bs32_nixl.yaml` - TEP8, BS32, no MTP, NIXL
3. `deepseek-r1-fp4_1k1k_dep16_bs128_mtp3_nixl.yaml` - DEP16, BS128, MTP3, NIXL (high concurrency)
4. `deepseek-r1-fp4_1k1k_dep32_bs32_nixl.yaml` - DEP32, BS32, no MTP, NIXL (very high TP)
5. `deepseek-r1-fp4_1k1k_tep8_bs32_mtp3_ucx.yaml` - TEP8, BS32, MTP3, UCX backend

### DeepSeek R1 FP4 (8k1k)

6. `deepseek-r1-fp4_8k1k_tep8_bs16_mtp3_nixl.yaml` - TEP8, BS16, MTP3, NIXL (long sequence)

### Qwen3-235B-A22B FP4

7. `Qwen3-235B-A22B-FP4_1k1k_tep8_bs32_mtp3_nixl.yaml` - TEP8, BS32, MTP3, NIXL

### DeepSeek V3 Lite FP8

8. `deepseek-v3-lite-fp8_1k1k_tep4_bs4_ucx.yaml` - TEP4, BS4, UCX (small model)

### Qwen3-8B FP8

9. `Qwen3-8B-FP8_1k1k_tep4_bs4_nixl.yaml` - TEP4, BS4, NIXL (small model)

## Configuration Source

All configurations are converted from `DisaggConfig.MODEL_CONFIGS` in `disagg_config.py`.

## Metadata Fields Explanation

Each YAML configuration file contains a `metadata` node to identify test metadata:

```yaml
metadata:
  model_name: "deepseek-r1-fp4"     # Model name
  precision: "fp4"                   # Precision type (fp4, fp8, etc.)
  supported_gpus: ["GB200", "GB300"] # List of supported GPU types
```

### Dynamic benchmark_type Generation

`benchmark_type` (e.g., 1k1k, 8k1k) is no longer parsed from the filename, but dynamically generated from the YAML `sequence` configuration:

- `input_length: 1024, output_length: 1024` → `1k1k`
- `input_length: 8192, output_length: 1024` → `8k1k`
- `input_length: 16384, output_length: 2048` → `16k2k`

**Advantages**:
- ✅ Configuration file is the single source of truth
- ✅ Avoids inconsistency between filename and actual configuration
- ✅ Easy to programmatically modify configuration without renaming files
- ✅ `metadata` field facilitates future extension of other metadata

### GPU Type Filtering

The system automatically filters configurations based on the current GPU type (via the `GPU_TYPE` environment variable):

```bash
export GPU_TYPE=GB200
python list_configs.py  # Only shows configurations supporting GB200

export GPU_TYPE=H100
python list_configs.py  # Only shows configurations supporting H100
```

## Usage

### 1. View Configurations with `list_configs.py`

```bash
# List all configurations
python list_configs.py

# View performance test configurations
python list_configs.py --category perf -v

# View specific model configurations
python list_configs.py --model deepseek-r1-fp4 --show-metrics
```

### 2. Submit a Single Job with `submit.py`

```bash
python disagg/slurm/benchmark/submit.py -c test_configs/disagg/perf/deepseek-r1-fp4_1k1k_tep8_bs32_mtp3_nixl.yaml
```

### 3. Batch Submit Jobs in a Directory with `submit.py`

```bash
python disagg/slurm/benchmark/submit.py -d test_configs/disagg/perf/
```

### 4. Run Tests with pytest

```bash
# Run all tests
pytest test_disagg_yaml.py -v

# Run only performance tests
pytest test_disagg_yaml.py -k "perf" -v

# Run specific model
pytest test_disagg_yaml.py -k "deepseek-r1-fp4" -v

# View detailed output
pytest test_disagg_yaml.py -s -vv
```

## Metrics Configuration

All performance test configurations use the following metrics by default:

- **log_file**: `benchmark_result.log`
- **metric_names**: `["DISAGG_SERVER_TTFT", "DISAGG_SERVER_E2EL"]`
- **extractor_pattern**: Predefined TTFT/E2EL extraction pattern

To customize metrics, add a `metrics` configuration under the `benchmark` node in the YAML file. See `solution4.md` for details.

## Notes

1. **Environment Paths**: All paths (container_image, model_path, work_dir, etc.) are based on the Lustre file system
2. **GPU Type**: Most configurations are for GB200 GPU, some small model configurations support H100/B200/B300
3. **Concurrency List**: `concurrency_list` is adjusted based on the configured throughput capability
4. **Default Configuration**: All configurations use default perf metrics, no need to explicitly configure in YAML

## Extending Configurations

To add new test configurations:

1. Create a new YAML file in the corresponding directory
2. Follow the file naming convention
3. Refer to the structure of existing configuration files
4. Override default metrics configuration as needed

## Configuration Mapping

| Original BenchmarkConfig Parameter | YAML Configuration Path |
|-----------------------------------|------------------------|
| `isl`, `osl` | `sequence.input_length`, `sequence.output_length` |
| `cache_transceiver_max_num_tokens` | `worker_config.gen.cache_transceiver_config.max_tokens_in_buffer` |
| `cache_transceiver_backend` | `worker_config.gen.cache_transceiver_config.backend` |
| `ctx_num`, `gen_num` | `hardware.num_ctx_servers`, `hardware.num_gen_servers` |
| `gen_tp_size` | `worker_config.gen.tensor_parallel_size` |
| `gen_batch_size` | `worker_config.gen.max_batch_size` |
| `gen_enable_attention_dp` | `worker_config.gen.enable_attention_dp` |
| `gen_mtp_size` | `worker_config.gen.speculative_config.num_nextn_predict_layers` |
| `concurrency_list` | `benchmark.concurrency_list` |
| `extractor_pattern`, `metric_names` | `benchmark.metrics` (optional, uses perf config by default) |
