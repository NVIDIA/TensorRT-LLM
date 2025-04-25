# How to get best performance on DeepSeek-R1 in TensorRT-LLM

NVIDIA has announced world-record DeepSeek-R1 inference performance at NVIDIA GTC 2025. A single NVIDIA DGX system with eight NVIDIA Blackwell GPUs can achieve over 250 tokens per second per user or a maximum throughput of over 30,000 tokens per second on the massive, state-of-the-art 671 billion parameter DeepSeek-R1 model. [NVIDIA Blackwell Delivers World-Record DeepSeek-R1 Inference Performance](https://developer.nvidia.com/blog/nvidia-blackwell-delivers-world-record-deepseek-r1-inference-performance/)

In this blog, we share the configurations and procedures about how to reproduce the number on both B200 and H200 with PyTorch workflow.

## Prerequisites: Install TensorRT-LLM and download models

This section can be skipped if you already have TensorRT-LLM installed and have already downloaded the DeepSeek R1 model checkpoint.

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

#### 3. Build and run TensorRT-LLM container

``` bash
cd TensorRT-LLM
make -C docker run LOCAL_USER=1 DOCKER_RUN_ARGS="-v $YOUR_MODEL_PATH:$YOUR_MODEL_PATH:ro -v $YOUR_WORK_PATH:$YOUR_WORK_PATH"
```
Here we set `LOCAL_USER=1` argument to set up the local user instead of root account inside the container, you can remove it if running as root inside container is fine.

#### 4. Compile and Install TensorRT-LLM
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

All the benchmarking is done by the trtllm-bench command line tool provided in the TensorRT-LLM installation, see [TensorRT-LLM Benchmarking](https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html) for details of this tool.

For brevity, we only provide the commands to reproduce the perf numbers without detailed explanation of the tools and options in this doc.

All these commands here are assumed to be running inside the container started by `make -C docker run ...` command mentioned in the [Build and run TensorRT-LLM container section](#3-build-and-run-tensorrt-llm-container)

### B200 min-latency
Our benchmark results are based on **Batch = 1, ISL = 1K, OSL = 2K, num_requests = 10 from real dataset**

To do the benchmark, run the following command:

```bash
YOUR_DATA_PATH=<your dataset file following the format>

cat >./extra-llm-api-config.yml<<EOF
pytorch_backend_config:
    enable_overlap_scheduler: true
    use_cuda_graph: true
    moe_backend: TRTLLM
speculative_config:
    decoding_type: MTP
    num_nextn_predict_layers: 3
EOF

export TRTLLM_ENABLE_PDL=1

trtllm-bench --model nvidia/DeepSeek-R1-FP4 \
    throughput \
    --dataset $YOUR_DATA_PATH \
    --backend pytorch \
    --num_requests 10 \
    --concurrency 1 \
    --max_batch_size 1 \
    --tp 8 \
    --ep 2 \
    --extra_llm_api_options ./extra-llm-api-config.yml
```

Explanation:
- `trtllm-bench`: A CLI benchmarking utility that aims to make it easier for users to reproduce our officially published. See [TensorRT-LLM Benchmarking](https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html) for details.
- `--dataset`: Prompt dataset used to benchmark. Our official benchmark dataset has ISL = 1K, OSL = 2K
- `--backend`: Inference backend. Here we use PyTorch backend.
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

### B200 max-throughput
Our benchmark results are based on **Batch = 3072, ISL = 1K, OSL = 2K, num_requests = 49152 from synthetic dataset**

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
pytorch_backend_config:
    use_cuda_graph: true
    cuda_graph_padding_enabled: true
    cuda_graph_batch_sizes:
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
    print_iter_log: true
    enable_overlap_scheduler: true
enable_attention_dp: true
EOF

trtllm-bench -m nvidia/DeepSeek-R1-FP4 \
    throughput \
    --tp 8 \
    --ep 8 \
    --warmup 0 \
    --dataset ${YOUR_DATA_PATH} \
    --backend pytorch \
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
Request Throughput (req/sec):                     17.3885
Total Output Throughput (tokens/sec):             35611.5942
Per User Output Throughput (tokens/sec/user):     11.6701
Per GPU Output Throughput (tokens/sec/gpu):       4451.4493
Total Latency (ms):                               2826700.0758
Average request latency (ms):                     176064.1921
```

### H200 min-latency
Our benchmark results are based on **Batch = 1, ISL = 1K, OSL = 2K, num_requests = 10 from real dataset**
To do the benchmark, run the following command:

```bash
YOUR_DATA_PATH=<your dataset file following the format>

cat >./extra-llm-api-config.yml<<EOF
pytorch_backend_config:
    enable_overlap_scheduler: true
    use_cuda_graph: true
speculative_config:
    decoding_type: MTP
    num_nextn_predict_layers: 3
EOF

trtllm-bench --model deepseek-ai/DeepSeek-R1 \
    throughput \
    --dataset $YOUR_DATA_PATH \
    --backend pytorch \
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
pytorch_backend_config:
    use_cuda_graph: true
    cuda_graph_batch_sizes:
    - 128
    enable_overlap_scheduler: true
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
    --backend pytorch \
    --max_batch_size 128 \
    --max_num_tokens 1127 \
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
Request Throughput (req/sec):                     5.1532
Total Output Throughput (tokens/sec):             10553.8445
Per User Output Throughput (tokens/sec/user):     10.4199
Per GPU Output Throughput (tokens/sec/gpu):       1319.2306
Total Token Throughput (tokens/sec):              15707.0888
Total Latency (ms):                               993548.8470
Average request latency (ms):                     197768.0434
```
