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
- **TOKEN_THROUGHPUT**: Token throughput
- **SEQ_THROUGHPUT**: Sequence throughput
- **FIRST_TOKEN_TIME**: First token generation time
- **OUTPUT_TOKEN_TIME**: Output token time

## 2. Detailed Test Flow

### 2.1 Dataset Preparation

#### Without LoRA
```python
prepare_data_script = os.path.join(self._llm_root, "benchmarks", "cpp", "prepare_dataset.py")
data_cmd += [
    "python3", prepare_data_script, "--stdout",
    f"--tokenizer={tokenizer_dir}", f"token-norm-dist",
    f"--num-requests={self._config.num_reqs}",
    f"--input-mean={input_len}", f"--output-mean={output_len}",
    f"--input-stdev={istdev}", f"--output-stdev={ostdev}",
    f" > {dataset_path}"
]
```

#### With LoRA
```python
"python3", prepare_data_script, f"--stdout",
    f"--rand-task-id 0 {nloras-1}",
    f"--tokenizer={tokenizer_dir}", f"--lora-dir={lora_dir}",
    f"token-norm-dist",
    f"--num-requests={self._config.num_reqs}",
    f"--input-mean={input_len}", f"--output-mean={output_len}",
    f"--input-stdev={istdev}", f"--output-stdev={ostdev}",
    f" > {dataset_path}"
```

### 2.2 PyTorch Configuration Generation
In `pytorch_model_config.py`, we override PyTorch configurations for certain specific cases and generate YAML configuration files.

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
if self._config.backend != "pytorch":
    benchmark_cmd += [
        f"--backend=tensorrt", f"--engine_dir={engine_dir}"
    ]
else:
    benchmark_cmd += ["--backend=pytorch"]
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

#### PyTorch Default Configuration
```python
# Use default YAML configuration
if self._config.backend == "pytorch":
    import yaml
    config = get_model_yaml_config(self._config.to_string(),
                                   lora_dirs=self.lora_dirs)
    print_info(f"pytorch model config: {config}")
    with open('extra-llm-api-config.yml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    benchmark_cmd += [
        f"--extra_llm_api_options=extra-llm-api-config.yml"
    ]
```

## 3. Test Scheduling

### 3.1 Full Test Cycles

1. **llm_perf_full.yml** - Release performance test
   - [test_lists/qa/llm_perf_full.yml](../../test_lists/qa/llm_perf_full.yml)
2. **llm_perf_cluster.yml** - Cluster performance test(for Blackwell)
   - [test_lists/qa/llm_perf_cluster.yml](../../test_lists/qa/llm_perf_cluster.yml)
3. **llm_perf_nim.yml** - NIM performance test
   - [test_lists/qa/llm_perf_nim.yml](../../test_lists/qa/llm_perf_nim.yml)

### 3.2 Sanity Test Cycles

- **llm_perf_sanity.yml** - Release performance sanity test
  - [test_lists/qa/llm_perf_sanity.yml](../../test_lists/qa/llm_perf_sanity.yml)

## 4. Test Configuration Description

### 4.1 PyTorch Model Configuration

The default PyTorch configuration is defined in [pytorch_model_config.py](pytorch_model_config.py) and can be overridden for specific test patterns. For example:

```python
{
    'patterns': [
        'qwen3_235b_a22b_fp4-bench-pytorch-float4-maxbs:512-maxnt:2048-input_output_len:1000,2000-con:8-ep:8-gpus:8',
    ],
    'config': {
        'enable_attention_dp': False,
        'moe_config': {
            'backend': 'TRTLLM'
        }
    }
}
```

This configuration allows you to customize PyTorch-specific settings for different model patterns while maintaining the base configuration as a fallback.

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
echo "perf/test_perf.py::test_perf[llama_v3.3_70b_instruct_fp8-bench-pytorch-float8-input_output_len:128,128]" >> perf_test.txt
```

### 6.4 Run Performance Test
```bash
pytest -v -s --test-prefix=H100_80GB_HBM3 --test-list=perf_test.txt -R=llama_v3.3_70b_instruct_fp8-bench-pytorch-float8-input_output_len:128,128 --output-dir=./output --perf --perf-log-formats=csv -o junit_logging=out-err
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
