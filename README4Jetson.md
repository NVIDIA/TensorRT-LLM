<div align="center">

TensorRT-LLM Deployment on Jetson Orin
===========================

<div align="left">

# Table of Contents

- [1. Installation](#1-installation)
- [2. Build and Run](#2-build-and-run)
- [3. Reference Memory Usage](#3-reference-memory-usage)
- [4. Reference Benchmark Performance ](#4-reference-benchmark-performance )


# 1 Installation
## 1.1 Install Jetpack 6.1
Install Jetpack 6.1 with CUDA, cuDNN and TensorRT with the help of SDK Manager and then boost and lock Jetson to MAX-N clock by the following commonds.
```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

## 1.2 Install Prerequisites
```bash
sudo apt-get update
sudo apt-get install -y python3-pip libopenblas-dev git-lfs ccache
wget https://raw.githubusercontent.com/pytorch/pytorch/9b424aac1d70f360479dd919d6b7933b5a9181ac/.ci/docker/common/install_cusparselt.sh
export CUDA_VERSION=12.6
sudo -E bash ./install_cusparselt.sh
python3 -m pip install numpy=='1.26.1'
```

## 1.3 Install Jetson TensorRT-LLM
```bash
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git checkout v0.12.0-jetson
git lfs pull
python3 scripts/build_wheel.py --clean --cuda_architectures 87 -DENABLE_MULTI_DEVICE=0 --build_type Release --benchmarks --use_ccache
pip install build/tensorrt_llm-*.whl
```

# 2 Build and Run
We take the Meta-Llama-3-8B-Instruct INT4-GPTQ as example.
## 2.1 Build the Engine with INT4-GPTQ
```bash
git clone https://huggingface.co/MaziyarPanahi/Meta-Llama-3-8B-Instruct-GPTQ

python convert_checkpoint.py --model_dir Meta-Llama-3-8B-Instruct-GPTQ --output_dir tllm_checkpoint_1gpu_gptq --dtype float16 --use_weight_only --weight_only_precision int4_gptq  --per_group

export PATH=/home/nvidia/.local/bin:$PATH
trtllm-build --checkpoint_dir tllm_checkpoint_1gpu_gptq --output_dir engine_1gpu_gptq --gemm_plugin float16
```
## 2.2 Run the Engine

```bash
python3 ../run.py --max_output_len=50 --tokenizer_dir Meta-Llama-3-8B-Instruct --engine_dir=engine_1gpu_gptq --use_mmap
```

# 3 Reference Memory Usage
## Static Batching

1. No CUDA_LAZY_LOADING

| w/wo mmap | GPU Memory Usage(GB) | Total Memory Usage(GB)|
| :-: | :-: |:-: |
| without mmap | 7.8 |16.2 |
| with mmap | 7.8 |11.2|

2. With CUDA_LAZY_LOADING (Default)

| w/wo mmap | GPU Memory Usage(GB) | Total Memory Usage(GB)|
| :-: | :-: |:-: |
| without mmap | 6.8 |12.3 |
| with mmap | 6.8 |7.3|

## Inflight Batching
In inflight batching mode, 90% of the memory is allocated by default for paged KV cache. Therefore, using mmap will almost not reduce the overall memory consumption, instead it will increase the available hardware memory.

| w/wo mmap | GPU Memory Usage(GB) | Total Memory Usage(GB)|
| :-: | :-: |:-: |
| without mmap | 44.0 |50.5 |
| with mmap | 48.9 |49.7|

 **NOTE:** To control the memory allocation for KV cache, you can utilize the ```--kv_cache_free_gpu_memory_fraction``` parameter when executing ```run.py```.


# 4 Reference Benchmark Performance

**Platform**: Jetson Orin 64GB, MAX-N, Jetpack 6.1

**Framework**: TRT-LLM v0.12.0-jetson

**Model**: Llama-3-8B

**Reference Command**:
```bash
cpp/build/benchmarks/gptSessionBenchmark --engine_dir path/to/engine --batch_size "1" --input_output_len "128,128" --enable_cuda_graph
```

## INT4 Default

This config be achieved by specifying `--weight_only_precision int4` when running `convert_checkpoint.py`

|  Batch Size  | Input Length | Output Length |Context (ms) | Decode (token/s) |
| :-: | :-: | :-: | :-:|:-: |
|1   |     128       | 128      |   91      |   35.9     |
|1   |     512       | 512      |   260     |   35.2     |
|1   |     1024      | 512      |   582     |   34.7     |

## INT4 GPTQ

This config be achieved by specifying `--weight_only_precision int4_gptq --per_group` when running `convert_checkpoint.py`

|  Batch Size  | Input Length | Output Length |Context (ms) | Decode (token/s) |
| :-: | :-: | :-: | :-:|:-: |
|1   |     128       | 128      |   105     |   34.5     |
|1   |     512       | 512      |   292     |   33.7     |
|1   |     1024      | 512      |   337     |   33.3     |
