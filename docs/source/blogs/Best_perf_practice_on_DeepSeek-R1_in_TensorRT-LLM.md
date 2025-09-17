# How to get best performance on DeepSeek-R1 in TensorRT LLM

NVIDIA has announced world-record DeepSeek-R1 inference performance at NVIDIA GTC 2025. A single NVIDIA DGX system with eight NVIDIA Blackwell GPUs can achieve over 250 tokens per second per user or a maximum throughput of over 30,000 tokens per second on the massive, state-of-the-art 671 billion parameter DeepSeek-R1 model. [NVIDIA Blackwell Delivers World-Record DeepSeek-R1 Inference Performance](https://developer.nvidia.com/blog/nvidia-blackwell-delivers-world-record-deepseek-r1-inference-performance/)

In this blog, we share the configurations and procedures about how to reproduce the number on both B200 and H200 with PyTorch workflow.

## Table of Contents

- [How to get best performance on DeepSeek-R1 in TensorRT LLM](#how-to-get-best-performance-on-deepseek-r1-in-tensorrt-llm)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites: Install TensorRT LLM and download models](#prerequisites-install-tensorrt-llm-and-download-models)
      - [1. Download TensorRT LLM](#1-download-tensorrt-llm)
      - [2. Download the DeepSeek R1 models](#2-download-the-deepseek-r1-models)
      - [3. Build and run TensorRT LLM container](#3-build-and-run-tensorrt-llm-container)
      - [4. Compile and Install TensorRT LLM](#4-compile-and-install-tensorrt-llm)
      - [5. Optional: Tune GPU clocks](#5-optional-tune-gpu-clocks)
      - [6. Dataset preparation](#6-dataset-preparation)
  - [Reproducing steps](#reproducing-steps)
    - [B200 min-latency](#b200-min-latency)
      - [Expected Results](#expected-results)
    - [B200 max-throughput for R1-0528 with FP8 KV cache](#b200-max-throughput-for-r1-0528-with-fp8-kv-cache)
      - [Benchmark](#benchmark)
      - [Expected Result Format](#expected-result-format)
    - [B200 max-throughput for R1 with FP16 KV cache](#b200-max-throughput-for-r1-with-fp16-kv-cache)
      - [Benchmark](#benchmark-1)
      - [Expected Result Format](#expected-result-format-1)
    - [H200 min-latency](#h200-min-latency)
      - [Expected Result Format](#expected-result-format-2)
    - [H200 max-throughput](#h200-max-throughput)
      - [Expected Result Format](#expected-result-format-3)
  - [Exploring more ISL/OSL combinations](#exploring-more-islosl-combinations)
    - [WIP: Enable more features by default](#wip-enable-more-features-by-default)
    - [Not supported: MLA chunked context support on Hopper](#not-supported-mla-chunked-context-support-on-hopper)
    - [Out of memory issues](#out-of-memory-issues)


## Prerequisites: Install TensorRT LLM and download models

This section can be skipped if you already have TensorRT LLM installed and have already downloaded the DeepSeek R1 model checkpoint.

#### 1. Download TensorRT LLM

**You can also find more comprehensive instructions to install TensorRT LLM in this [TensorRT LLM installation guide](https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-linux.html), refer to that guide for common issues if you encounter any here.**

``` bash
# Prerequisites
apt-get update && apt-get -y install git git-lfs
git lfs install

# Replace with your actual path
YOUR_WORK_PATH=<YOUR_WORK_PATH>

# Clone the TensorRT LLM repository
cd $YOUR_WORK_PATH
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git submodule update --init --recursive
git lfs pull
```
**Note**: Replace `<*_PATH>` to your actual path.

#### 2. Download the DeepSeek R1 models

For NVIDIA Blackwell GPUs, it's recommended to use the [FP4 quantized version of DeepSeek R1](https://huggingface.co/nvidia/DeepSeek-R1-FP4) to get the best performance.
For NVIDIA Hopper GPUs, it's recommended to use the FP8 version of the DeepSeek R1 model.

```bash
# Replace with your actual path
YOUR_MODEL_PATH=<YOUR_MODEL_PATH>
cd $YOUR_MODEL_PATH

## Download FP4 model for Blackwell GPUs
git clone https://huggingface.co/nvidia/DeepSeek-R1-FP4

## Download FP8 model for Hopper GPUs
## FP8 model also works for Blackwell, but FP4 has the best performance on Blackwell.
git clone https://huggingface.co/deepseek-ai/DeepSeek-R1
```

#### 3. Build and run TensorRT LLM container

``` bash
cd TensorRT-LLM
make -C docker run LOCAL_USER=1 DOCKER_RUN_ARGS="-v $YOUR_MODEL_PATH:$YOUR_MODEL_PATH:ro -v $YOUR_WORK_PATH:$YOUR_WORK_PATH"
```
Here we set `LOCAL_USER=1` argument to set up the local user instead of root account inside the container, you can remove it if running as root inside container is fine.

#### 4. Compile and Install TensorRT LLM
Here we compile the source inside the container:

``` bash
python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt --benchmarks --cuda_architectures "90-real;100-real"  --python_bindings --clean
```
You can set the cuda_architectures to "100-real" if targeting Blackwell only, and "90-real" to target Hopper only to save some build time.

Install and set environment variables:

```bash
pip install --user build/tensorrt_llm*.whl
export PATH=${HOME}/.local/bin:${PATH}
export PYTHONPATH=`pwd`
```

#### 5. Optional: Tune GPU clocks
```
sudo nvidia-smi -pm 0; sudo nvidia-smi -pm 1; sudo nvidia-smi boost-slider --vboost 4
```
The boost-slider option will tune the GPU clock and can get you slight perf increase, for B200 min-latency scenarios it's about 8 TPS/USER.
This is not a required step, it's provided here to make sure the perf numbers in this doc can be reproduced more closely to our internal run.

#### 6. Dataset preparation

The trtllm-bench tool requires a dataset file to read prompts and output sequence length of each prompt. Format details of this dataset file can be seen in [preparing a dataset](
https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html#preparing-a-dataset).

For min-latency benchmarking, **real dataset is required** since the MTP accept rate is affected by the dataset thus affecting the performance. You can use your own dataset following the format described in the link above.

For the max-throughput benchmarking, synthetic dataset is enough to be representative, since it does not use MTP.
The command to generate synthetic dataset will be attached to the max throughput section.

## Reproducing steps

This section provides the reproducing steps for NVIDIA Blackwell B200 and H200 GPUs, for both min-latency and max-throughput scenarios.

All the benchmarking is done by the trtllm-bench command line tool provided in the TensorRT LLM installation, see [TensorRT LLM Benchmarking](https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html) for details of this tool.

For brevity, we only provide the commands to reproduce the perf numbers without detailed explanation of the tools and options in this doc.

All these commands here are assumed to be running inside the container started by `make -C docker run ...` command mentioned in the [Build and run TensorRT LLM container section](#3-build-and-run-tensorrt-llm-container)

### B200 min-latency
Our benchmark results are based on **Batch = 1, ISL = 1K, OSL = 2K, num_requests = 10 from real dataset**

To do the benchmark, run the following command:

```bash
YOUR_DATA_PATH=<your dataset file following the format>

cat >./extra-llm-api-config.yml<<EOF
moe_config:
  backend: TRTLLM
speculative_config:
    decoding_type: MTP
    num_nextn_predict_layers: 3
EOF

export TRTLLM_ENABLE_PDL=1

trtllm-bench --model nvidia/DeepSeek-R1-FP4 \
    throughput \
    --dataset $YOUR_DATA_PATH \
    --num_requests 10 \
    --concurrency 1 \
    --max_batch_size 1 \
    --tp 8 \
    --ep 2 \
    --extra_llm_api_options ./extra-llm-api-config.yml
```

Explanation:
- `trtllm-bench`: A CLI benchmarking utility that aims to make it easier for users to reproduce our officially published. See [TensorRT LLM Benchmarking](https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html) for details.
- `--dataset`: Prompt dataset used to benchmark. Our official benchmark dataset has ISL = 1K, OSL = 2K
- `--num_requests`: Num requests used for the benchmark.
- `--concurrency`: Total concurrency for the system.
- `--max_batch_size`: Max batch size in each rank.
- `--tp`: Tensor parallel size.
- `--ep`: Expert parallel size.
- `--extra_llm_api_options`: Used to specify some extra config. The content of the file is as follows:

#### Expected Results
The perf can be different when using different datasets and different machines.

```
===========================================================
= PERFORMANCE OVERVIEW
===========================================================
Request Throughput (req/sec):                     0.1341
Total Output Throughput (tokens/sec):             274.4168
Per User Output Throughput (tokens/sec/user):     274.7188
Per GPU Output Throughput (tokens/sec/gpu):       34.3021
Total Token Throughput (tokens/sec):              414.0461
Total Latency (ms):                               74561.7520
Average request latency (ms):                     7456.1219
```
### B200 max-throughput for R1-0528 with FP8 KV cache

Due to our evaluation found that FP8 KV cache does not introduce obvious accuracy drop compared to BF16 KV cache. See [Precision strategy](./tech_blog/blog3_Optimizing_DeepSeek_R1_Throughput_on_NVIDIA_Blackwell_GPUs.md#precision-strategy), the latest [DeepSeek-R1-0528-FP4](https://huggingface.co/nvidia/DeepSeek-R1-0528-FP4) checkpoint had enabled FP8 KV cache by-default.

We are seeing meaningful speedup using FP8 KV cache, thus refreshing the numbers here. The results are reproduced with TensorRT LLM commit b6261862419c33d6ce2313aff1e7116067d6037d.

!! Note that the exact command to reproduce numbers can change as the API/options are refactored, the option and numbers here is a reference at given exact commit.

#### Benchmark
```bash
cat >./extra-llm-api-config.yml <<EOF
cuda_graph_config:
  enable_padding: true
  batch_sizes:
  - 896
  - 512
  - 256
  - 128
  - 64
  - 32
  - 16
  - 8
  - 4
  - 2
  - 1
print_iter_log: true
kv_cache_dtype: fp8
enable_attention_dp: true
EOF
trtllm-bench  --model nvidia/DeepSeek-R1-0528-FP4
     throughput
     --dataset ${YOUR_DATA_PATH}
     --tp 8  --ep 8
     --extra_llm_api_options ./extra-llm-api-config.yml
     --max_batch_size 896
     --max_num_tokens 2048
     --kv_cache_free_gpu_mem_fraction 0.93
     --concurrency 7168
     --num_requests 114688
```
#### Expected Result Format
```
===========================================================
= PERFORMANCE OVERVIEW
===========================================================
Request Throughput (req/sec):                     21.0675
Total Output Throughput (tokens/sec):             43146.2042
Total Token Throughput (tokens/sec):              65100.6376
Total Latency (ms):                               5443839.8140
Average request latency (ms):                     332826.9898
Per User Output Throughput [w/ ctx] (tps/user):   6.1806
Per GPU Output Throughput (tps/gpu):              5393.2755
```

### B200 max-throughput for R1 with FP16 KV cache
Our benchmark results are based on **Batch = 3072, ISL = 1K, OSL = 2K, num_requests = 49152 from synthetic dataset**.

The results are reproduced with TensorRT LLM commit b6261862419c33d6ce2313aff1e7116067d6037d.

!! Note that the exact command to reproduce numbers can change as the API/options are refactored, the option and numbers here is a reference at given exact commit.

#### Benchmark
To do the benchmark, run the following command:

```bash
# generate synthetic dataset
python ${YOUR_WORK_PATH}/benchmarks/cpp/prepare_dataset.py \
        --stdout \
        --tokenizer nvidia/DeepSeek-R1-FP4 \
        token-norm-dist \
        --input-mean 1024 --output-mean 2048 \
        --input-stdev 0 --output-stdev 0 \
        --num-requests 49152 > dataset.txt

YOUR_DATA_PATH=./dataset.txt

cat >./extra-llm-api-config.yml <<EOF
cuda_graph_config:
  enable_padding: true
  batch_sizes:
  - 1
  - 2
  - 4
  - 8
  - 16
  - 32
  - 64
  - 128
  - 256
  - 384
print_iter_log: ${PRINT_ITER_LOG}
enable_attention_dp: true
EOF

trtllm-bench -m nvidia/DeepSeek-R1-FP4 \
    throughput \
    --tp 8 \
    --ep 8 \
    --warmup 0 \
    --dataset ${YOUR_DATA_PATH} \
    --max_batch_size 384 \
    --max_num_tokens 1536 \
    --num_requests 49152 \
    --concurrency 3072 \
    --kv_cache_free_gpu_mem_fraction 0.85 \
    --extra_llm_api_options ./extra-llm-api-config.yml
```

#### Expected Result Format
The perf might be different from different datasets and machines
```
===========================================================
= PERFORMANCE OVERVIEW
===========================================================
Request Throughput (req/sec):                     17.7657
Total Output Throughput (tokens/sec):             36384.0838
Total Token Throughput (tokens/sec):              54576.1257
Total Latency (ms):                               2766684.9197
Average request latency (ms):                     172321.7206
Per User Output Throughput [w/ ctx] (tps/user):   11.9263
Per GPU Output Throughput (tps/gpu):              4548.0105
```

### H200 min-latency
Our benchmark results are based on **Batch = 1, ISL = 1K, OSL = 2K, num_requests = 10 from real dataset**
To do the benchmark, run the following command:

```bash
YOUR_DATA_PATH=<your dataset file following the format>

cat >./extra-llm-api-config.yml<<EOF
speculative_config:
    decoding_type: MTP
    num_nextn_predict_layers: 3
EOF

trtllm-bench --model deepseek-ai/DeepSeek-R1 \
    throughput \
    --dataset $YOUR_DATA_PATH \
    --num_requests 10 \
    --max_batch_size 1 \
    --tp 8 \
    --ep 4 \
    --concurrency 1 \
    --extra_llm_api_options ./extra-llm-api-config.yml
```

#### Expected Result Format

The perf might be different from different datasets and machines
```
===========================================================
= PERFORMANCE OVERVIEW
===========================================================
Request Throughput (req/sec):                     0.0772
Total Output Throughput (tokens/sec):             158.0669
Per User Output Throughput (tokens/sec/user):     158.1196
Per GPU Output Throughput (tokens/sec/gpu):       19.7584
Total Latency (ms):                               129498.2168
Average request latency (ms):                     12945.9379
```

### H200 max-throughput
Our benchmark results are based on **Batch = 1024, ISL = 1K, OSL = 2K, num_requests = 5120 from real dataset**
To do the benchmark, run the following command:

```bash
# generate synthetic dataset
python ${YOUR_WORK_PATH}/benchmarks/cpp/prepare_dataset.py \
        --stdout \
        --tokenizer deepseek-ai/DeepSeek-R1 \
        token-norm-dist \
        --input-mean 1024 --output-mean 2048 \
        --input-stdev 0 --output-stdev 0 \
        --num-requests 5120 > dataset.txt
YOUR_DATA_PATH=./dataset.txt

cat >./extra-llm-api-config.yml<<EOF
cuda_graph_config:
  batch_sizes:
  - 128
enable_attention_dp: true
EOF

# Use NVCC for DeepGEMM JIT compilation
export TRTLLM_DG_JIT_USE_NVCC=1

trtllm-bench -m deepseek-ai/DeepSeek-R1 \
    throughput \
    --tp 8 \
    --ep 8 \
    --warmup 0 \
    --dataset $YOUR_DATA_PATH \
    --max_batch_size 128 \
    --max_num_tokens 1151 \
    --num_requests 5120 \
    --concurrency 1024 \
    --kv_cache_free_gpu_mem_fraction 0.8 \
    --extra_llm_api_options ./extra-llm-api-config.yml
```

#### Expected Result Format
The perf might be different from different datasets and machines

```
===========================================================
= PERFORMANCE OVERVIEW
===========================================================
Request Throughput (req/sec):                     5.6100
Total Output Throughput (tokens/sec):             11489.2671
Per User Output Throughput (tokens/sec/user):     11.3476
Per GPU Output Throughput (tokens/sec/gpu):       1436.1584
Total Token Throughput (tokens/sec):              17233.9007
Total Latency (ms):                               912656.9938
Average request latency (ms):                     181540.5739
```

## Exploring more ISL/OSL combinations

To benchmark TensorRT LLM on DeepSeek models with more ISL/OSL combinations, you can use `prepare_dataset.py` to generate the dataset and use similar commands mentioned in the previous section. TensorRT LLM is working on enhancements that can make the benchmark process smoother.
### WIP: Enable more features by default

Currently, there are some features that need to be enabled through a user-defined file `extra-llm-api-config.yml`, such as CUDA graph, overlap scheduler and attention dp. We're working on to enable those features by default, so that users can get good out-of-the-box performance on DeepSeek models.

Note that, `max_batch_size` and `max_num_tokens` can easily affect the performance. The default values for them are already carefully designed and should deliver good performance on overall cases, however, you may still need to tune it for peak performance.

Generally, you should make sure that `max_batch_size` is not too low to bottleneck the throughput, and `max_num_tokens` needs to be large enough so that it covers the max input sequence length of the samples in dataset, as mentioned in below section "WIP: Chunked context support on DeepSeek models".

For more details on `max_batch_size` and `max_num_tokens`, refer to [Tuning Max Batch Size and Max Num Tokens](../performance/performance-tuning-guide/tuning-max-batch-size-and-max-num-tokens.md).

### MLA chunked context

MLA currently supports the chunked context feature on both Hopper and Blackwell GPUs. You can use `--enable_chunked_context` to enable it. This feature is primarily designed to reduce TPOT (Time Per Output Token). The default chunk size is set to `max_num_tokens`. If you want to achieve a lower TPOT, you can appropriately reduce the chunk size. However, please note that this will also decrease overall throughput. Therefore, a trade-off needs to be considered. 

For more details on `max_num_tokens`, refer to [Tuning Max Batch Size and Max Num Tokens](../performance/performance-tuning-guide/tuning-max-batch-size-and-max-num-tokens.md).

### Out of memory issues

It's possible seeing OOM issues on some cases. Considering reducing `kv_cache_free_gpu_mem_fraction` to a smaller value as a workaround. We're working on the investigation and addressing the problem.
