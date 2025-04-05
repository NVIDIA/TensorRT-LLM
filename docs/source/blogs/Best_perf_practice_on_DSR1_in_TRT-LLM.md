# How to get best performance on DSR1 in TRT-LLM

NVIDIA has announced world-record DeepSeek-R1 inference performance at NVIDIA GTC 2025. A single NVIDIA DGX system with eight NVIDIA Blackwell GPUs can achieve over 250 tokens per second per user or a maximum throughput of over 30,000 tokens per second on the massive, state-of-the-art 671 billion parameter DeepSeek-R1 model. [NVIDIA Blackwell Delivers World-Record DeepSeek-R1 Inference Performance](https://developer.nvidia.com/blog/nvidia-blackwell-delivers-world-record-deepseek-r1-inference-performance/)

In this blog, we share the configurations and procedures about how to reproduce the number on both B200 and H200 with Pytorch workflow.

## B200 NVL8
### Prerequisites

``` bash
# Prerequisites
apt-get update && apt-get -y install git git-lfs
git lfs install

# Improve GPU performance
sudo nvidia-smi -pm 0; sudo nvidia-smi -pm 1; sudo nvidia-smi boost-slider --vboost 4

# Replace with your actual path
YOUR_WORK_PATH=<YOUR_WORK_PATH>
YOUR_MODEL_PATH=<YOUR_MODEL_PATH>
YOUR_DATA_PATH=<YOUR_DATA_PATH>

# Clone the TensorRT-LLM repository
cd $YOUR_WORK_PATH
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git submodule update --init --recursive
git lfs pull

# Clone the DeepSeek-R1-FP4 model
cd $YOUR_MODEL_PATH
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/nvidia/DeepSeek-R1-FP4
git lfs pull  # Download the full model weight will take a long time
```
**Note**: Replace `<*_PATH>` to your actual path. 

#### Build Docker 
Create a docker and run:

``` bash
cd TensorRT-LLM
make -C docker jenkins_run LOCAL_USER=1 DOCKER_RUN_ARGS="-v $YOUR_MODEL_PATH:$YOUR_MODEL_PATH:ro -v $YOUR_DATA_PATH:$YOUR_DATA_PATH:ro -v $YOUR_WORK_PATH:$YOUR_WORK_PATH"
```
Here we set `LOCAL_USER=1` argument to set up the local user account inside the container.

#### Compile and Install
Here we compile the source inside the container:

``` bash
python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt --benchmarks --use_ccache --cuda_architectures "100-real"  --python_bindings --clean
```

Install and set environment variables:

```bash
pip install --user build/tensorrt_llm*.whl
export PATH=${HOME}/.local/bin:${PATH}
export PYTHONPATH=`pwd`
```
### B200 min-latency
Our benchmark results are based on **Batch = 1, ISL = 1K, OSL = 2K, num_requests = 10 from real dataset**

#### Benchmark
To do the benchmark, run the following command:

```bash
export TRTLLM_ENABLE_PDL=1

DS_R1_NVFP4_ALLMOE_MODEL_PATH=$YOUR_MODEL_PATH/DeepSeek-R1-FP4
trtllm-bench --model deepseek-ai/DeepSeek-R1 \
    --model_path $DS_R1_NVFP4_ALLMOE_MODEL_PATH \
    throughput \
    --dataset $YOUR_DATA_PATH \
    --backend pytorch \
    --num_requests 10 \
    --concurrency 1 \
    --max_batch_size 1 \
    --tp 8 \
    --ep 4 \
    --extra_llm_api_options ./extra-llm-api-config.yml
```

Explanation:
- `trtllm-bench`: A CLI benchmarking utility that aims to make it easier for users to reproduce our officially published. [TensorRT-LLM Benchmarking](https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html).
- `--dataset`: Prompt dataset used to benchmark. Our official benchmark dataset has ISL = 1K, OSL = 2K
- `--backend`: Inference backend. Here we use Pytorch backend.
- `--num_requests`: Num requests used for the benchmark.
- `--concurrency`: Total concurrency for the system.
- `--max_batch_size`: Max batch size in each rank.
- `--tp`: Tensor parallel size.
- `--ep`: Expert parallel size.
- `--extra_llm_api_options`: Used to specify some extra config. The content of the file is as follows:

    ``` yaml
    pytorch_backend_config:
        enable_overlap_scheduler: true
        use_cuda_graph: true
    speculative_config:
        decoding_type: MTP
        num_nextn_predict_layers: 3
    ```


#### Expected Result Format
The perf might be different from different datasets and machines

``` 
===========================================================                                                                                                     
= PERFORMANCE OVERVIEW  
===========================================================
Request Throughput (req/sec):                     0.1244   
Total Output Throughput (tokens/sec):             254.5535
Per User Output Throughput (tokens/sec/user):     254.7634
Per GPU Output Throughput (tokens/sec/gpu):       31.8192  
Total Latency (ms):                               80368.1616
Average request latency (ms):                     8036.7546
```

## B200 max-throughput
Our benchmark results are based on **Batch = 3072, ISL = 1K, OSL = 2K, num_requests = 49152 from real dataset**

#### Benchmark
To do the benchmark, run the following command:

```bash
DS_R1_NVFP4_ALLMOE_MODEL_PATH=$YOUR_MODEL_PATH/DeepSeek-R1-FP4
trtllm-bench -m deepseek-ai/DeepSeek-R1 \
    --model_path $DS_R1_NVFP4_ALLMOE_MODEL_PATH \
    throughput \
    --tp 8 \
    --ep 8 \
    --warmup 0 \
    --dataset $YOUR_DATA_PATH \
    --backend pytorch \
    --max_batch_size 384 \
    --max_num_tokens 1536 \
    --num_requests 49152 \
    --concurrency 3072 \
    --kv_cache_free_gpu_mem_fraction 0.85 \
    --extra_llm_api_options ./extra-llm-api-config.yml
```

Explanation:
- `trtllm-bench`: A CLI benchmarking utility that aims to make it easier for users to reproduce our officially published. [TensorRT-LLM Benchmarking](https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html).
- `--dataset`: Prompt dataset used to benchmark. our official benchmark dataset has ISL = 1K, OSL = 2K
- `--backend`: Inference backend. Here we use Pytorch backend. 
- `--tp 8`: Tensor parallel size is 8.
- `--ep 8`: Expert parallel size is 8.
- `--max_batch_size`: Max batch size in each rank.
- `--max_num_tokens`: Max num tokens in each rank.
- `--num_requests`: Num requests used for the benchmark.
- `--concurrency`: Total concurrency for the system.
- `--kv_cache_free_gpu_mem_fraction`: Mem fraction used to hold kv cache tokens.
- `--extra_llm_api_options`: Used to specify some extra config. The content of the file is as follows:

    ``` yaml
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
    ```


#### Expected Result Format
The perf might be different from different datasets and machines

``` 
===========================================================
= PERFORMANCE OVERVIEW
===========================================================
Request Throughput (req/sec):                     17.5825
Total Output Throughput (tokens/sec):             36008.8948
Per User Output Throughput (tokens/sec/user):     11.7967
Per GPU Output Throughput (tokens/sec/gpu):       4501.1119
Total Latency (ms):                               2795511.9559
Average request latency (ms):                     174119.3317
```

## H200 NVL8
### Prerequisites

``` bash
# Prerequisites
apt-get update && apt-get -y install git git-lfs
git lfs install

# Replace with your actual path
YOUR_WORK_PATH=<YOUR_WORK_PATH>
YOUR_MODEL_PATH=<YOUR_MODEL_PATH>
YOUR_DATA_PATH=<YOUR_DATA_PATH>

# Clone the TensorRT-LLM repository
cd $YOUR_WORK_PATH
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git submodule update --init --recursive
git lfs pull

# Clone the DeepSeek-R1 model
cd $YOUR_MODEL_PATH
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/deepseek-ai/DeepSeek-R1
git lfs pull  # Download the full model weight will take a long time
```
**Note**: Replace `<*_PATH>` to your actual path. 

#### Build Docker 
Create a docker and run:

``` bash
cd TensorRT-LLM
make -C docker jenkins_run LOCAL_USER=1 DOCKER_RUN_ARGS="-v $YOUR_MODEL_PATH:$YOUR_MODEL_PATH:ro -v $YOUR_DATA_PATH:$YOUR_DATA_PATH:ro -v $YOUR_WORK_PATH:$YOUR_WORK_PATH"
```
Here we set `LOCAL_USER=1` argument to set up the local user account inside the container.

#### Compile and Install
Here we compile the source inside the container:

``` bash
python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt --benchmarks --use_ccache --cuda_architectures "90-real"  --python_bindings --clean
```

Install and set environment variables:

```bash
pip install --user build/tensorrt_llm*.whl
export PATH=${HOME}/.local/bin:${PATH}
export PYTHONPATH=`pwd`
```
### H200 min-latency
Our benchmark results are based on **Batch = 1, ISL = 1K, OSL = 2K, num_requests = 10 from real dataset**

#### Benchmark
To do the benchmark, run the following command:

```bash
# Enable DeepGEMM
export TRTLLM_DG_ENABLED=1

DS_R1_MODEL_PATH=$YOUR_MODEL_PATH/DeepSeek-R1
trtllm-bench --model deepseek-ai/DeepSeek-R1 \
    --model_path $DS_R1_MODEL_PATH \
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

Explanation:
- `trtllm-bench`: A CLI packags benchmarking utility that aims to make it easier for users to reproduce our officially published. [TensorRT-LLM Benchmarking](https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html).
- `--dataset`: Prompt dataset used to benchmark. our official benchmark dataset has ISL = 1K, OSL = 2K
- `--backend`: Inference backend. Here we use Pytorch backed. 
- `--tp 8`: Tensor parallel size is 8.
- `--ep 4`: Expert parallel size is 4.
- `--extra_llm_api_options`: Used to specify some extra config. The content of the file is as follows:

    ``` yaml
    pytorch_backend_config:
        enable_overlap_scheduler: true
        use_cuda_graph: true
    speculative_config:
        decoding_type: MTP
        num_nextn_predict_layers: 3
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

#### Benchmark
To do the benchmark, run the following command:

```bash
export TRTLLM_DG_ENABLED=1

DS_R1_MODEL_PATH=$YOUR_MODEL_PATH/DeepSeek-R1
trtllm-bench -m deepseek-ai/DeepSeek-R1 \
    --model_path $DS_R1_MODEL_PATH \
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

Explanation:
- `trtllm-bench`: A CLI benchmarking utility that aims to make it easier for users to reproduce our officially published. [TensorRT-LLM Benchmarking](https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html).
- `--dataset`: Prompt dataset used to benchmark. our official benchmark dataset has ISL = 1K, OSL = 2K
- `--backend`: Inference backend. Here we use Pytorch backend. 
- `--tp 8`: Tensor parallel size is 8.
- `--ep 8`: Expert parallel size is 8.
- `--max_batch_size`: Max batch size in each rank.
- `--max_num_tokens`: Max num tokens in each rank.
- `--num_requests`: Num requests used for the benchmark.
- `--concurrency`: Total concurrency for the system.
- `--kv_cache_free_gpu_mem_fraction`: Mem fraction used to hold kv cache tokens.
- `--extra_llm_api_options`: Used to specify some extra config. The content of the file is as follows:

    ``` yaml
        pytorch_backend_config:
            use_cuda_graph: true
            cuda_graph_batch_sizes:
            - 128
            enable_overlap_scheduler: true
        enable_attention_dp: true
    ```


#### Expected Result Format
The perf might be different from different datasets and machines

``` 
===========================================================
= PERFORMANCE OVERVIEW
===========================================================
Request Throughput (req/sec):                     5.1942
Total Output Throughput (tokens/sec):             10637.7380
Per User Output Throughput (tokens/sec/user):     10.4899
Per GPU Output Throughput (tokens/sec/gpu):       1329.7173
Total Latency (ms):                               985713.3137
Average request latency (ms):                     195228.9468
```

