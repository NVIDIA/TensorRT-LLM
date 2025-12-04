# Deployment Guide for Kimi K2 Thinking on TensorRT LLM - Blackwell

## Introduction

This is a quick-start guide for running the Kimi K2 Thinking model on TensorRT LLM. It focuses on a working setup with recommended defaults.

## Prerequisites

* GPU: NVIDIA Blackwell Architecture
* OS: Linux
* Drivers: CUDA Driver 575 or Later
* Docker with NVIDIA Container Toolkit installed
* Python3 and python3-pip (Optional, for accuracy evaluation only)

## Models

* NVFP4 model: [Kimi-K2-Thinking-NVFP4](https://huggingface.co/nvidia/Kimi-K2-Thinking-NVFP4)


## Deploy Kimi K2 Thinking on DGX B200 through Docker

### Prepare Docker image

Build and run the docker container. See the [Docker guide](../../../docker/README.md) for details.
```
cd TensorRT-LLM

make -C docker release_build IMAGE_TAG=kimi-k2-thinking-local

make -C docker release_run IMAGE_NAME=tensorrt_llm IMAGE_TAG=kimi-k2-thinking-local LOCAL_USER=1
```

### Launch the TensorRT LLM Server

Prepare a `EXTRA_OPTIONS_YAML_FILE` that specify LLM API arguments when deploying the model, an example YAML file is as follows:

```yaml
max_batch_size: 128
max_num_tokens: 8448
max_seq_len: 8212
tensor_parallel_size: 8
moe_expert_parallel_size: 8
enable_attention_dp: true
pipeline_parallel_size: 1
print_iter_log: true
kv_cache_config:
  free_gpu_memory_fraction: 0.75
  dtype: fp8
cache_transceiver_config:
  backend: UCX
  max_tokens_in_buffer: 8448
trust_remote_code: true
```

This YAML file specifies configurations that deploy the model with 8-way expert parallelism for the MoE part, and 8-way attention data parallelism. It also enables `trust_remote_code`, so that it works with the Kimi K2 Thinking customize [tokenizer](https://huggingface.co/nvidia/Kimi-K2-Thinking-NVFP4/blob/main/tokenization_kimi.py).


With the `EXTRA_OPTIONS_YAML_FILE`, use the following example command to launch the TensorRT LLM server with the Kimi-K2-Thinking-NVFP4 model from within the container.

```shell
trtllm-serve nvidia/Kimi-K2-Thinking-NVFP4 \
    --host 0.0.0.0 --port 8000 \
    --extra_llm_api_options ${EXTRA_OPTIONS_YAML_FILE}
```

TensorRT LLM will load weights and select the best kernels during starting, and the server is successfully launched when the following log is shown:

```
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:8000 (Press CTRL+C to quit)
```

You can query the health/readiness of the server using:

```shell
curl -s -o /dev/null -w "Status: %{http_code}\n" "http://localhost:8000/health"
```

When the `Status: 200` code is returned, the server is ready for queries.

## Deploy Kimi K2 Thinking on GB200 NVL72 through SLURM with wide EP and disaggregated serving

TensorRT LLM provides a set of SLURM scripts that could be easily configured through YAML files, and automatically launch SLURM jobs on the GB200 NVL72 clusters for deploying, benchmarking and accuracy testing purposes. The scripts are located at `examples/disaggregated/slurm/benchmark`, and refer to [this page](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/wide_ep/slurm_scripts) for more details and example wide EP config file.

For Kimi K2 Thinking, an example to configure SLURM arguments and the scripts is as the following:

```yaml
# SLURM Configuration
slurm:
  script_file: "disaggr_torch.slurm"
  partition: "<partition>"
  account: "<account>"
  job_time: "02:00:00"
  job_name: "<job_name>"
  extra_args: "" # Cluster specific arguments, e.g. "--gres=gpu:4 --exclude=node1,node2"
  numa_bind: true # Only enable for GB200 NVL72

# Benchmark Mode
benchmark:
  mode: "e2e"  # Options: e2e, gen_only
  use_nv_sa_benchmark: false  # Whether to use NVIDIA SA benchmark script
  multi_round: 8  # Number of benchmark rounds
  benchmark_ratio: 0.8  # Benchmark ratio
  streaming: true  # Enable streaming mode
  concurrency_list: "16"
  input_length: 1024  # Input sequence length
  output_length: 1024  # Output sequence length
  dataset_file: "<dataset_file>"

# Hardware Configuration
hardware:
  gpus_per_node: 4  # Modify this with your hardware configuration
  num_ctx_servers: 4  # Number of context servers
  num_gen_servers: 1  # Number of generation servers

# Environment Configuration
environment:
  container_mount: "<container_mount>"  # Format: path1:path1,path2:path2
  container_image: "<container_image>"
  model_path: "<model_path>"
  trtllm_repo: "<trtllm_repo>"
  build_wheel: false  # Don't build the wheel when launching multiple jobs
  trtllm_wheel_path: ""  # Path to pre-built TensorRT-LLM wheel. If provided, install from this wheel instead
  work_dir: "<full_path_to_work_dir>"
  worker_env_var: "TLLM_LOG_LEVEL=INFO TRTLLM_SERVER_DISABLE_GC=1 TRTLLM_WORKER_DISABLE_GC=1 TRTLLM_ENABLE_PDL=1 ENROOT_ALLOW_DEV=yes"
  server_env_var: "TRTLLM_SERVER_DISABLE_GC=1"

# Worker Configuration
worker_config:
  gen:
    tensor_parallel_size: 32
    moe_expert_parallel_size: 32
    enable_attention_dp: true
    enable_lm_head_tp_in_adp: true
    pipeline_parallel_size: 1
    max_batch_size: 128
    max_num_tokens: 128
    max_seq_len: 9236
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
      - 512
      - 768
      - 1024
      - 2048
      - 128
    print_iter_log: true
    kv_cache_config:
      enable_block_reuse: false
      free_gpu_memory_fraction: 0.6
      dtype: fp8
    moe_config:
      backend: WIDEEP
      use_low_precision_moe_combine: true
      load_balancer:
        num_slots: 416
        layer_updates_per_iter: 1
    cache_transceiver_config:
      backend: UCX
      max_tokens_in_buffer: 8448
    stream_interval: 20
    num_postprocess_workers: 4
    trust_remote_code: true
  ctx:
    max_batch_size: 1
    max_num_tokens: 8448
    max_seq_len: 8212
    tensor_parallel_size: 4
    moe_expert_parallel_size: 4
    enable_attention_dp: true
    pipeline_parallel_size: 1
    print_iter_log: true
    cuda_graph_config: null
    disable_overlap_scheduler: true
    kv_cache_config:
      enable_block_reuse: false
      free_gpu_memory_fraction: 0.75
      dtype: fp8
    cache_transceiver_config:
      backend: UCX
      max_tokens_in_buffer: 8448
    trust_remote_code: true
```

It includes SLURM specific configurations, benchmark and hardware details, and environment settings. The `worker_config` field includes detailed settings for context and generation servers when deploying a disaggregated server, and each performs as a list of LLM API arguments.

To launch SLURM jobs with the YAML config file, execute the following command:
```shell
cd <TensorRT LLM root>/examples/disaggregated/slurm/benchmark
python3 submit.py -c config.yaml
```

## Query the OpenAI-compatible API Endpoint

After the TensorRT LLM server is set up and shows Application startup complete, you can send requests to the server.

```shell
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json"  -d '{
    "model": "nvidia/Kimi-K2-Thinking-NVFP4",
    "messages": [
        {
            "role": "user",
            "content": "Where is New York?"
        }
    ],
    "max_tokens": 128,
    "top_p": 1.0
}' -w "\n"
```

Example response:

```

```

## Benchmark

To benchmark the performance of your TensorRT LLM server you can leverage the built-in `benchmark_serving.py` script. To do this first creating a wrapper `bench.sh` script.

```shell
cat <<'EOF' > bench.sh
#!/usr/bin/env bash
set -euo pipefail

concurrency_list="1 2 4 8 16 32 64 128 256"
multi_round=5
isl=1024
osl=1024
result_dir=/tmp/kimi_k2_thinking_output

for concurrency in ${concurrency_list}; do
    num_prompts=$((concurrency * multi_round))
    python -m tensorrt_llm.serve.scripts.benchmark_serving \
        --model nvidia/Kimi-K2-Thinking-NVFP4 \
        --backend openai \
        --dataset-name "random" \
        --random-input-len ${isl} \
        --random-output-len ${osl} \
        --random-prefix-len 0 \
        --random-ids \
        --num-prompts ${num_prompts} \
        --max-concurrency ${concurrency} \
        --ignore-eos \
        --tokenize-on-client \
        --percentile-metrics "ttft,tpot,itl,e2el"
done
EOF
chmod +x bench.sh
```

If you want to save the results to a file add the following options.

```shell
--save-result \
--result-dir "${result_dir}" \
--result-filename "concurrency_${concurrency}.json"
```

For more benchmarking options see [benchmark_serving.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/serve/scripts/benchmark_serving.py)

Run `bench.sh` to begin a serving benchmark.

```shell
./bench.sh
```
