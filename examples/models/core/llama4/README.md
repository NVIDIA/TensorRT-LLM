# Llama4-Maverick

This document shows how to run Llama4-Maverick on B200 with PyTorch workflow and how to run performance benchmarks


## Table of Contents

- [Performance Benchmarks](#performance-benchmarks)
  - [B200 Max-throughput](#b200-max-throughput)
  - [B200 Min-latency](#b200-min-latency)
  - [B200 Balanced](#b200-balanced)
- [Advanced Configuration](#advanced-configuration)
  - [Configuration tuning](#configuration-tuning)
  - [Troubleshooting](#troubleshooting)
    - [Out of memory issues](#out-of-memory-issues)


## Performance Benchmarks

This section provides the steps to launch TensorRT LLM server and run performance benchmarks for different scenarios.


### B200 Max-throughput


#### 1. Prepare TensorRT LLM extra configs
```bash
cat >./extra-llm-api-config.yml <<EOF
enable_attention_dp: true
stream_interval: 2
cuda_graph_config:
  max_batch_size: 512
  enable_padding: true
EOF
```
Explanation:
- `enable_attention_dp`: Enable attention Data Parallel which is recommend to enable in high concurrency.
- `stream_interval`: The iteration interval to create responses under the streaming mode.
- `cuda_graph_config`: CUDA Graph config.
  - `max_batch_size`: Max CUDA graph batch size to capture.
  - `enable_padding`: Whether to enable CUDA graph padding.


#### 2. Launch trtllm-serve OpenAI-compatible API server
TensorRT LLM supports nvidia TensorRT Model Optimizer quantized FP8 checkpoint
``` bash
trtllm-serve nvidia/Llama-4-Maverick-17B-128E-Instruct-FP8 \
    --max_batch_size 512 \
    --tp_size 8 \
    --ep_size 8 \
    --num_postprocess_workers 2 \
    --trust_remote_code \
    --extra_llm_api_options ./extra-llm-api-config.yml
```


#### 3. Run performance benchmarks
TensorRT LLM provides a benchmark tool to benchmark trtllm-serve
Prepare a new terminal and run `benchmark_serving`
```bash
python -m tensorrt_llm.serve.scripts.benchmark_serving \
        --model nvidia/Llama-4-Maverick-17B-128E-Instruct-FP8 \
        --dataset-name random \
        --ignore-eos \
        --num-prompts 8192 \
        --random-input-len 1024 \
        --random-output-len 2048 \
        --random-ids \
        --max-concurrency 1024 \
```


### B200 Min-latency


#### 1. Prepare TensorRT LLM extra configs
```bash
cat >./extra-llm-api-config.yml <<EOF
enable_attention_dp: false
enable_min_latency: true
stream_interval: 2
cuda_graph_config:
  max_batch_size: 8
  enable_padding: true
EOF
```
Explanation:
- `enable_attention_dp`: Enable attention Data Parallel which is recommend to disable in low concurrency.
- `enable_min_latency` Enable optimizations for low latency scenarios, where concurrency is very small (like 1 or 2).
- `stream_interval`: The iteration interval to create responses under the streaming mode.
- `cuda_graph_config`: CUDA Graph config.
  - `max_batch_size`: Max CUDA graph batch size to capture.
  - `enable_padding`: Whether to enable CUDA graph padding.


#### 2. Launch trtllm-serve OpenAI-compatible API server
TensorRT LLM supports nvidia TensorRT Model Optimizer quantized FP8 checkpoint.
``` bash
trtllm-serve nvidia/Llama-4-Maverick-17B-128E-Instruct-FP8 \
    --max_batch_size 8 \
    --tp_size 8 \
    --ep_size 1 \
    --trust_remote_code \
    --extra_llm_api_options ./extra-llm-api-config.yml
```


#### 3. Run performance benchmark
TensorRT LLM provides a benchmark tool to benchmark trtllm-serve
Prepare a new terminal and run `benchmark_serving`
```bash
python -m tensorrt_llm.serve.scripts.benchmark_serving \
        --model nvidia/Llama-4-Maverick-17B-128E-Instruct-FP8 \
        --dataset-name random \
        --ignore-eos \
        --num-prompts 1000 \
        --random-input-len 1024 \
        --random-output-len 2048 \
        --random-ids \
        --max-concurrency 1 \
```

### B200 Balanced


#### 1. Prepare TensorRT LLM extra configs
```bash
cat >./extra-llm-api-config.yml <<EOF
stream_interval: 2
cuda_graph_config:
  max_batch_size: 1024
  enable_padding: true
EOF
```
Explanation:
- `stream_interval`: The iteration interval to create responses under the streaming mode.
- `cuda_graph_config`: CUDA Graph config.
  - `max_batch_size`: Max CUDA graph batch size to capture.
  - `enable_padding`: Whether to enable CUDA graph padding.


#### 2. Launch trtllm-serve OpenAI-compatible API server
TensorRT LLM supports nvidia TensorRT Model Optimizer quantized FP8 checkpoint.
``` bash
trtllm-serve nvidia/Llama-4-Maverick-17B-128E-Instruct-FP8 \
    --tp_size 8 \
    --ep_size 2 \
    --num_postprocess_workers 2 \
    --trust_remote_code \
    --extra_llm_api_options ./extra-llm-api-config.yml
```


#### 3. Run performance benchmark
TensorRT LLM provides a benchmark tool to benchmark trtllm-serve
Prepare a new terminal and run `benchmark_serving`
```bash
python -m tensorrt_llm.serve.scripts.benchmark_serving \
        --model nvidia/Llama-4-Maverick-17B-128E-Instruct-FP8 \
        --dataset-name random \
        --ignore-eos \
        --num-prompts 1000 \
        --random-input-len 1024 \
        --random-output-len 2048 \
        --random-ids \
        --max-concurrency 64 \
```

## Advanced Configuration

### Configuration tuning

- **Attention DP** only provides throughput gains in high concurrency scenarios. Consider disabling it for low concurrency and enabling it for high concurrency. The concurrency threshold needs to be tuned based on your specific ISL/OSL configuration.
- **Expert Parallel (EP)** usually benefits in high concurrency scenarios. The `ep_size` needs to be tuned based on your specific ISL/OSL and concurrency configuration.
- `enable_min_latency` enables optimizations for low latency scenarios, where concurrency is very small (like 1 or 2). The concurrency threshold needs to be tuned based on your specific ISL/OSL configuration.
- `stream_interval` and `num_postprocess_workers` are both used to reduce streaming mode overhead. `stream_interval` controls the iteration interval to create responses under streaming mode, which benefits performance across all concurrency levels. `num_postprocess_workers` controls the number of processes used for postprocessing generated tokens, which provides benefits in high concurrency scenarios. These values need to be tuned based on your specific ISL/OSL and concurrency configuration.
- `max_batch_size` and `max_num_tokens` can easily affect the performance. The default values for them are already carefully designed and should deliver good performance on overall cases, however, you may still need to tune it for peak performance.
- `max_batch_size` should not be too low to bottleneck the throughput. Note with Attention DP, the the whole system's max_batch_size will be `max_batch_size*dp_size`.
- CUDA grah `max_batch_size` should be same value as TensorRT LLM server's `max_batch_size`.
- For more details on `max_batch_size` and `max_num_tokens`, refer to [Tuning Max Batch Size and Max Num Tokens](../../../../docs/source/performance/performance-tuning-guide/tuning-max-batch-size-and-max-num-tokens.md).

### Troubleshooting

#### Out of memory issues

It's possible to see OOM issues in some cases. Considering reducing `kv_cache_free_gpu_mem_fraction` to a smaller value as a workaround. We're working on investigating and addressing the problem.
