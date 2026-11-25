# TensorRT LLM Performance Test Flow (Default PyTorch Flow)

## Overview
This document describes the complete TensorRT LLM performance testing workflow, particularly for the default PyTorch backend testing process for release testing.

## 1. Test Scripts

### Main Test Script
The main script for TensorRT LLM performance testing is `test_perf.py`, which is responsible for executing all performance test cases.

### Performance Metrics
For trtllm-bench, the test extracts the following key performance metrics from logs:

- **BUILD_TIME**: Model build time
- **INFERENCE_TIME**: Inference time
- **TOTAL_OUTPUT_THROUGHPUT**: Total output token throughput (system-wide)
- **SEQ_THROUGHPUT**: Sequence throughput
- **FIRST_TOKEN_TIME**: First token generation time
- **OUTPUT_TOKEN_TIME**: Output token time

## 2. Detailed Test Flow

### 2.1 Dataset Preparation

#### Without LoRA
```python
data_cmd += [
    "trtllm-bench", f"--model={tokenizer_dir}",
        "prepare-dataset", "--output", dataset_path, "token-norm-dist",
        f"--num-requests={self._config.num_reqs}",
        f"--input-mean={input_len}", f"--output-mean={output_len}",
        f"--input-stdev={istdev}", f"--output-stdev={ostdev}"
]
```

#### With LoRA
```python
"trtllm-bench", f"--model={tokenizer_dir}",
    "prepare-dataset", "--output", dataset_path,
    f"--rand-task-id 0 {nloras-1}",
    f"--lora-dir={lora_dir}",
    f"token-norm-dist",
    f"--num-requests={self._config.num_reqs}",
    f"--input-mean={input_len}", f"--output-mean={output_len}",
    f"--input-stdev={istdev}", f"--output-stdev={ostdev}"
```

### 2.2 PyTorch Configuration Generation
The runner merges the configuration selected by the test name and writes the resulting YAML for `trtllm-bench` or `trtllm-serve`.

### 2.3 Calling trtllm-bench for Throughput Testing

#### Basic Command
```python
benchmark_cmd = [
    self._benchmark_script,
    f"--model={model_name}",
    f"--model_path={model_dir}",
    "throughput",
    f"--dataset={dataset_path}",
    f"--max_batch_size={self._config.max_batch_size}",
    f"--max_num_tokens={self._config.max_num_tokens}",
    f"--report_json={report_path}",
]
```

#### Backend Selection
```python
benchmark_cmd += [f"--backend={self._config.backend}"]
```

#### Optional Parameter Configuration
```python
if self._config.num_reqs > 0:
    benchmark_cmd += [f"--num_requests={self._config.num_reqs}"]
if self._config.concurrency != -1:
    benchmark_cmd += [f"--concurrency={self._config.concurrency}"]
if self._config.ep_size != None:
    benchmark_cmd += [f"--ep={self._config.ep_size}"]
if self._config.tp_size > 1:
    benchmark_cmd += [f"--tp={self._config.tp_size}"]
if self._config.pp_size > 1:
    benchmark_cmd += [f"--pp={self._config.pp_size}"]
if self._config.streaming == "streaming":
    benchmark_cmd += [f"--streaming"]
```

#### PyTorch Launch Configuration
```python
if self._config.backend == "pytorch":
    import yaml
    config = self._get_model_yaml_config()
    print_info(f"pytorch model config: {config}")
    with open('config.yml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    benchmark_cmd += [
        f"--config=config.yml"
    ]
```

## 3. Test Scheduling

### 3.1 Full Test Cycles

1. **llm_perf_core.yml** - Release performance test; the former `llm_perf_full.yml`, `llm_perf_cluster.yml` and `llm_perf_sanity.yml` were merged into it
   - [test_lists/qa/llm_perf_core.yml](../../test_lists/qa/llm_perf_core.yml)
2. **llm_perf_disagg.yml** - Disaggregated serving performance test
   - [test_lists/qa/llm_perf_disagg.yml](../../test_lists/qa/llm_perf_disagg.yml)
3. **llm_perf_multinode.yml** - Multi-node performance test
   - [test_lists/qa/llm_perf_multinode.yml](../../test_lists/qa/llm_perf_multinode.yml)

## 4. Test Configuration Description

### 4.1 PyTorch Model Configuration

Put model settings in the final `-cfg:` suffix. Explicit test-name options such as
`maxbs`, `maxnt`, `tp`, `pp`, `ep`, `kv_frac` and `moe` override the selected
YAML. Fields omitted from both use the runner and TensorRT-LLM defaults.

- Select a file in [examples/configs/curated](../../../../examples/configs/curated)
  by filename, for example `-cfg:deepseek-v4-pro-latency.yaml`.
- Use a named preset for workloads whose historical settings would make the ID too long.
  For example, `-cfg:lightning_mtp` supplies the shared KV, MTP and streaming settings
  for both Nemotron Lightning variants. Each case selects its own `moe:` backend.
  These presets are declared together in [pytorch_model_config.py](pytorch_model_config.py).
- Set a case-specific field directly with `-cfg:enable_attention_dp=true`.
  Append comma-separated overrides to a selection, or combine selections from left
  to right with `+`, as in `-cfg:glm53_flash_fp8+mtp3`.
  Use YAML scalar values and dotted field names, for example
  `cfg:speculative_config.decoding_type=NGram,max_draft_len=3`. The runner passes these fields to
  the same configuration validation as a YAML file. Put paths and lists in the selected recipe or preset.
  Later selections override earlier ones and merge nested mappings.

Each Core and Spark case selects at most one model YAML. Complete recipes come from
`examples/configs/curated/`. Cases using named Python presets get all their shared
settings from that module. Draft paths prefixed with `${LLM_MODELS_ROOT}/` resolve
against the test model cache at launch.

The test parameter also names the engine directory. Keep both the full parameter and
the exported benchmark name within 255 bytes. Reuse a shared profile when inlining
its fields would exceed that limit.

### 4.1 Test Case Configuration
- Test cases are defined in YAML configuration files
- Support for different models, precisions, batch sizes, etc.
- Support for LoRA and standard model testing

### 4.2 Performance Baseline
- Compare regression of each release on internal TRT-Perf dashboard

### 4.3 Result Analysis
- Generates detailed performance reports
- Supports performance trend analysis
- View performance data and compare between different runs on internal TRT-Perf dashboard

## 5. Runtime Environment Requirements

### 5.1 Dependency Installation
```bash
pip install -r ./TensorRT-LLM/requirements.txt
pip install -r ./TensorRT-LLM/requirements-dev.txt
```

### 5.2 Hardware Requirements
- CUDA-capable GPU
- Sufficient GPU memory for model loading
- Recommended to use B200/GB200 or higher performance GPU for cluster testing

## 6. Reproduce Steps

To reproduce the performance tests locally, follow these steps:

### 6.1 Install Dependencies
```bash
pip install -r requirements-dev.txt
pip install -r requirements.txt
```

### 6.2 Navigate to Test Directory
```bash
cd tests/integration/defs
```

### 6.3 Add Test Case to Test List
```bash
echo "perf/test_perf.py::test_perf[<test-case-id>]" >> perf_test.txt
```

### 6.4 Run Performance Test
```bash
pytest -v -s --test-prefix=H100_80GB_HBM3 --test-list=perf_test.txt \
  -R=<test-case-id> --output-dir=./output --perf \
  --perf-log-formats=csv -o junit_logging=out-err
```

### 6.5 Command Parameters Explanation
- `--test-prefix=H100_80GB_HBM3`: Specifies the test environment prefix
- `--test-list`: Points to the test list file containing test cases
- `-R`: Filter for specific test patterns
- `--output-dir=./output`: Specifies the output directory for test results
- `--perf`: Enables performance testing mode
- `--perf-log-formats=csv`: Outputs performance logs in CSV format
- `-o junit_logging=out-err`: Configures JUnit logging output

## 7. Related Documentation
- [Sanity Perf Check Introduction](README.md)
