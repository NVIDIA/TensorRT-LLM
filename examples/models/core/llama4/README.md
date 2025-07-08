# Llama4-Maverick

This document shows how to run Llama4-Maverick on B200 with PyTorch workflow and how to run performance benchmark


## Table of Contents

- [Llama4-Maverick](#Llama4-Maverick)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites: Install TensorRT-LLM and download models](#prerequisites-install-tensorrt-llm-and-download-models)
      - [1. Download TensorRT-LLM](#1-download-tensorrt-llm)
      - [2. Build and run TensorRT-LLM container](#2-build-and-run-tensorrt-llm-container)
      - [3. Compile and Install TensorRT-LLM](#3-compile-and-install-tensorrt-llm)
  - [Launching the server and run performance benchmark](#launching-the-server-and-run-performance-benchmark)
  - [Exploring more ISL/OSL combinations](#exploring-more-islosl-combinations)
    - [Out of memory issues](#out-of-memory-issues)


## Prerequisites: Install TensorRT-LLM and download models

This section can be skipped if you already have TensorRT-LLM installed.

#### 1. Download TensorRT-LLM

**You can also find more comprehensive instructions to install TensorRT-LLM in this [TensorRT-LLM installation guide](https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-linux.html), refer to that guide for common issues if you encounter any here.**

``` bash
# Prerequisites
apt-get update && apt-get -y install git git-lfs
git lfs install

# Replace with your actual path
YOUR_WORK_PATH=<YOUR_WORK_PATH>

# Clone the TensorRT-LLM repository
cd $YOUR_WORK_PATH
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git submodule update --init --recursive
git lfs pull
```
**Note**: Replace `<*_PATH>` to your actual path.


#### 2. Build and run TensorRT-LLM container

``` bash
cd TensorRT-LLM
make -C docker run LOCAL_USER=1 DOCKER_RUN_ARGS="-v $YOUR_MODEL_PATH:$YOUR_MODEL_PATH:ro -v $YOUR_WORK_PATH:$YOUR_WORK_PATH"
```
Here we set `LOCAL_USER=1` argument to set up the local user instead of root account inside the container, you can remove it if running as root inside container is fine.

#### 3. Compile and Install TensorRT-LLM
Here we compile the Blackwell only source inside the container:

``` bash
python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt --benchmarks --cuda_architectures "100-real"  --python_bindings --clean
```

Install and set environment variables:

```bash
pip install --user build/tensorrt_llm*.whl
export PATH=${HOME}/.local/bin:${PATH}
export PYTHONPATH=`pwd`
```

## Launching the server and run performance benchmark

This section provides the steps to launch TensorRT-LLM server and run performance benchmark for max-throughput scenarios.

All below commands here are assumed to be running inside the container started by `make -C docker run ...` command mentioned in the [Build and run TensorRT-LLM container section](#3-build-and-run-tensorrt-llm-container)


### 1. Prepare TensorRT-LLM extra configs
```bash
cat >./extra-llm-api-config.yml <<EOF
enable_attention_dp: true
stream_interval: 4
cuda_graph_config:
  max_batch_size: 1024
  padding_enabled: true
EOF
```
Explanation:
- `enable_attention_dp`: Enable attention Data Parallel which is recommend to enable in high concurrency.
- `stream_interval`: The iteration interval to create responses under the streaming mode.
- `cuda_graph_config`: CUDA Graph config.
  - `max_batch_size`: Max cuda graph batch size to capture.
  - `padding_enabled`: Whether to enable CUDA graph padding.


### 2. Launch trtllm-serve OpenAI-compatible API server
TensorRT-LLM supports nvidia TensorRT Model Optimizer quantized FP8 checkpoint
``` bash
trtllm-serve nvidia/Llama-4-Maverick-17B-128E-Instruct-FP8 \
    --backend pytorch \
    --max_batch_size 1024 \
    --tp_size 8 \
    --ep_size 8 \
    --trust_remote_code \
    --extra_llm_api_options ./extra-llm-api-config.yml
```
With Attention DP one, the whole system's max_batch_size will be max_batch_size*tp_size


### 3. Run performance benchmark
TensorRT-LLM provides a benchmark tool to benchmark trtllm-serve
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


## Exploring more ISL/OSL combinations

Currently, there are some features that need to be enabled through a user-defined file `extra-llm-api-config.yml`, such as CUDA graph, and attention dp. We're working on to enable those features by default, so that users can get good out-of-the-box performance.

Note that, `max_batch_size` and `max_num_tokens` can easily affect the performance. The default values for them are already carefully designed and should deliver good performance on overall cases, however, you may still need to tune it for peak performance.

Generally, you should make sure that `max_batch_size` is not too low to bottleneck the throughput, and `max_num_tokens` needs to be large enough so that it covers the max input sequence length of the samples in dataset.

For more details on `max_batch_size` and `max_num_tokens`, refer to [Tuning Max Batch Size and Max Num Tokens](../performance/performance-tuning-guide/tuning-max-batch-size-and-max-num-tokens.md).

### Out of memory issues

It's possible seeing OOM issues on some cases. Considering reducing `kv_cache_free_gpu_mem_fraction` to a smaller value as a workaround. We're working on the investigation and addressing the problem.
