# Running a High Performance GPT-OSS-120B Inference Server with TensorRT-LLM

NVIDIA has [announced](https://developer.nvidia.com/blog/delivering-1-5-m-tps-inference-on-nvidia-gb200-nvl72-nvidia-accelerates-openai-gpt-oss-models-from-cloud-to-edge/) day-0 support for OpenAI's new open-source model series, [gpt-oss](https://openai.com/index/introducing-gpt-oss/). In the guide below, we will walk you through how to launch your own
high-performance TensorRT-LLM server for **gpt-oss-120b** for inference.

**Low-latency** use cases aim to maximize the number of tokens per second per user (tps/user) with limited concurrency.

For **max-throughput**, the goal is to maximize the tokens produced per GPU per second (tps/gpu). While tps/user indicates user experience quality, tps/gpu measures the economic efficiency of the system.

## Prerequisites

- 1x NVIDIA B200/GB200/H200 GPU (more GPUs could be used for lower latency and higher throughput)
- Fast SSD storage for model weights
- Access to the gpt-oss-120b model checkpoint

We have a forthcoming guide for achieving great performance on H100; however, this guide focuses on the GPUs listed above.

## Install TensorRT-LLM

In this section, we introduce several ways to install TensorRT-LLM.

###  NGC Docker Image of dev branch

Day-0 support for gpt-oss is provided via the NGC container image `nvcr.io/nvidia/tensorrt-llm/release:gpt-oss-dev`. This image was built on top of the pre-day-0 **dev branch**. This container is multi-platform and will run on both x64 and arm64 architectures.

Run the following docker command to start the TensorRT-LLM container in interactive mode:

```bash
docker run --rm --ipc=host -it \
  --ulimit stack=67108864 \
  --ulimit memlock=-1 \
  --gpus all \
  -p 8000:8000 \
  -e TRTLLM_ENABLE_PDL=1 \
  -v ~/.cache:/root/.cache:rw \
  nvcr.io/nvidia/tensorrt-llm/release:gpt-oss-dev \
  /bin/bash
```


Explanation of the command:
- Automatically removes the container when stopped (`--rm`)
- Allows container to interact with the host's IPC resources and shared memory for optimal performance (`--ipc=host`)
- Runs the container in interactive mode (`-it`)
- Sets up shared memory and stack limits for optimal performance
- Maps port 8000 from the container to the host
- Enables PDL for performance optimization

Additionally, the container mounts your user `.cache` directory to save the downloaded model checkpoints, which are stored in `~/.cache/huggingface/hub/` by default. This prevents having to redownload the weights each time you rerun the container. You can also download the weights to a custom location (we assume `${local_model_path}` is the path to the local model weights).

### Build from source

Support for gpt-oss has been [merged](https://github.com/NVIDIA/TensorRT-LLM/pull/6645) into the **main branch** of TensorRT-LLM. As we continue to optimize gpt-oss performance, you can build TensorRT-LLM from source to get the latest features and support. Please refer to the [doc](https://nvidia.github.io/TensorRT-LLM/latest/installation/build-from-source-linux.html) if you want to build from source yourself.


### Regular Release of TensorRT-LLM

Since gpt-oss has been supported on the main branch, you can get TensorRT-LLM out of the box through its regular release in the future. Please check the latest [release notes](https://github.com/NVIDIA/TensorRT-LLM/releases) to keep track of the support status. The release is provided as [NGC Container Image](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release/tags) or [pip Python wheel](https://pypi.org/project/tensorrt-llm/#history). You can find instructions on pip install [here](https://nvidia.github.io/TensorRT-LLM/installation/linux.html).


## Performance Benchmarking and Model Serving

This guide covers how to configure for both low-latency and max-throughput cases, as well as how to benchmark end-to-end performance.

### Prepare the dataset
Before getting started, we need to prepare a dataset of randomized tokens for benchmarking:

```bash
python benchmarks/cpp/prepare_dataset.py \
  --stdout \
  --tokenizer openai/gpt-oss-120b \
  token-norm-dist \
  --input-mean 1024 \
  --output-mean 2048 \
  --input-stdev 0 \
  --output-stdev 0 \
  --num-requests 20000 > gpt-oss-120b-1k2k.txt
```

### Low-latency Use Case

The low-latency configuration maximizes tps/user under limited concurrency (e.g., 1, 4, 8, or 16 users). Please set the number of GPUs and concurrency according to your specific situation and workload.

```bash
num_gpus=8
max_batch_size=1
```


#### Creating the Extra Options Configuration

Create a YAML configuration file, `low_latency.yaml`, as follows:

```bash
cat <<EOF > low_latency.yaml
enable_attention_dp: false
cuda_graph_config:
    max_batch_size: ${max_batch_size}
    enable_padding: true
moe_config:
    backend: TRTLLM
EOF
```

Key takeaways:
- `enable_attention_dp` is set to `false` to use TP instead of DP for attention.
s- `cuda_graph_config.max_batch_size` is the maximum batch size for CUDA graph.
- `cuda_graph_config.enable_padding` is set to `true` to enable CUDA graph padding.
- `moe_config.backend` is set to `TRTLLM` to use the `trtllm-gen` MoE kernels which are optimized for low concurrency.


> Note: If you are using NVIDIA H200 GPUs please set the `moe_config.backend` to `TRITON` to use the OpenAI Triton MoE kernel regardless of use case. See the section [(H200/H100 Only) Using OpenAI Triton Kernels for MoE](#h200h100-only-using-openai-triton-kernels-for-moe) for more details.


#### Run the benchmark
Use `trtllm-bench` to benchmark the performance of your system:

```bash
trtllm-bench \
    --model openai/gpt-oss-120b \
    --model_path ${local_model_path} \
    throughput \
    --backend pytorch \
    --tp ${num_gpus} \
    --ep 1 \
    --extra_llm_api_options low_latency.yaml \
    --dataset gpt-oss-120b-1k2k.txt \
    --max_batch_size ${max_batch_size} \
    --concurrency ${max_batch_size} \
    --num_requests $((max_batch_size * 10)) \
    --kv_cache_free_gpu_mem_fraction 0.9 \
    --streaming \
    --warmup 0 \
    --report_json low_latency_benchmark.json
```

`--max_batch_size` controls the maximum batch size that the inference engine could serve, while `--concurrency` is the number of concurrent requests that the benchmarking client is sending. `--num_requests` is set to 10 times of `--concurrency` to run enough number of requests.

Note that you can set `--ep` to a value larger than 1, which will enable mixed TP/EP for MoE. In minimum-latency scenarios, we recommend a small EP size to avoid load imbalance in MoE.

For reference, we achieve **420 tps/user** with 8x B200 GPUs and batch size 1.


### Max-Throughput Use Case

The max-throughput configuration maximizes tps/gpu at high concurrency levels. With increasing concurrency, we trade per-user latency for higher throughput that saturates the system's GPUs. Using input sequence length (isl) of 1k and output sequence length (osl) of 2k, we can currently achieve a batch size of 640 with 8x B200 GPUs.

```bash
num_gpus=8
max_batch_size=640
```


#### Creating the Extra Options Configuration

Like before, create a YAML configuration file, `max_throughput.yaml`, as follows:

```bash
cat <<EOF > max_throughput.yaml
enable_attention_dp: true
cuda_graph_config:
    max_batch_size: ${max_batch_size}
    enable_padding: true
stream_interval: 10
moe_config:
    backend: CUTLASS
EOF
```

Compared to the low-latency configuration, we:
- set `enable_attention_dp` to `true` to use attention DP which is better for high throughput.
- set `stream_interval` to 10 to stream results to the client every 10 tokens. At high concurrency, the detokenization overhead of streaming mode cannot be hidden under GPU execution time, so `stream_interval` serves as a workaround to reduce this overhead.
- set `moe_config.backend` to `CUTLASS` to use the `CUTLASS` MoE kernels which are optimized for high throughput.

#### Run the benchmark

Run the following command to benchmark the throughput of your system:

```bash
trtllm-bench \
    --model openai/gpt-oss-120b \
    --model_path ${local_model_path} \
    throughput \
    --backend pytorch \
    --tp ${num_gpus} \
    --ep ${num_gpus} \
    --extra_llm_api_options max_throughput.yaml \
    --dataset gpt-oss-120b-1k2k.txt \
    --max_batch_size ${max_batch_size} \
    --concurrency $((max_batch_size * num_gpus)) \
    --num_requests $((max_batch_size * num_gpus * 3)) \
    --kv_cache_free_gpu_mem_fraction 0.9 \
    --streaming \
    --warmup 0 \
    --report_json max_throughput_benchmark.json
```

Note:
- `CUTLASS` MoE backend only supports pure EP for MoE, so we set `--ep` to `num_gpus`.
- When using `enable_attention_dp`, `max_batch_size` describes the maximum batch size for each local rank, so to saturate the system, we need to multiply `max_batch_size` by `num_gpus` for `--concurrency`.
- `--num_requests` is set to 3 times `--concurrency` to run enough number of requests.

Currently, the best throughput **19.5k tps/gpu** is achieved with DP4EP4 using 4x B200 GPUs and over **20k tps/gpu** on GB200 GPUs due to slightly better performance of GB200, which translates to over **1.5M tps** on a GB200 NVL72 system. In theory, even better tps/gpu could be achieved with larger world size due to larger allowable batch size and smaller MoE weights per-GPU, but the communication implementation for >4GPUs is suboptimal and we are actively working on improving it.



## Launch the TensorRT-LLM Server

We can use `trtllm-serve` to serve the model by translating the benchmark commands above. For low-latency configuration, run:

```bash
trtllm-serve \
  gpt-oss-120b \  # Or ${local_model_path}
  --host 0.0.0.0 \
  --port 8000 \
  --backend pytorch \
  --tp_size ${num_gpus} \
  --ep_size 1  \
  --extra_llm_api_options low_latency.yaml \
  --kv_cache_free_gpu_memory_fraction 0.9 \
  --max_batch_size ${max_batch_size} \  # E.g., 1
  --trust_remote_code
```

The initialization may take several minutes as it loads and optimizes the models.

For max-throughput configuration, run:

```bash
trtllm-serve \
  gpt-oss-120b \  # Or ${local_model_path}
  --host 0.0.0.0 \
  --port 8000 \
  --backend pytorch \
  --tp_size ${num_gpus} \
  --ep_size ${num_gpus} \
  --extra_llm_api_options max_throughput.yaml \
  --kv_cache_free_gpu_memory_fraction 0.9 \
  --max_batch_size ${max_batch_size} \  # E.g., 640 
  --trust_remote_code
```



### Test the Server with a Sample Request


To check the server's health and readiness:

```bash
curl -s -o /dev/null -w "Status: %{http_code}\n" "http://localhost:8000/health"
```

When the `Status: 200` code is returned, the server is ready for queries. Note that the
very first query may take longer due to initialization and compilation.

Once the server is running, you can test it with a simple curl request:

```bash
curl localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "openai/gpt-oss-120b",
    "messages": [
        {
            "role": "user",
            "content": "What is NVIDIA's advantage for inference?"
        }
    ],
    "max_tokens": 1024,
    "top_p": 0.9
}' -w "\n"
```

<details><summary><b>Show Example Output</b></summary>

```bash
{
  "id": "chatcmpl-c440e2a3e7e14cd699295afc3739bf42",
  "object": "chat.completion",
  "created": 1754358426,
  "model": "openai/gpt-oss-120b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "<|channel|>analysis<|message|>The user asks: \"What is NVIDIA's advantage for inference?\" The user wants:
         \"What is NVIDIA's advantage for inference?\" Likely they want a detailed answer about why NVIDIA has advantages for
          inference tasks (e.g., GPU acceleration, TensorRT, software stack, Tensor Cores, hardware, performance, low latency,
          etc.). The user wants a short answer? Not specified. The user wants a straightforward answer. Probably a brief
          explanation: hardware features like Tensor cores, optimized libraries (TensorRT), ecosystem, software stack,
          integration with frameworks, cuDNN, inference GPU architecture, high throughput, low latency, FP16, INT8, etc.\n\nWe
          need to produce a response: Provide NVIDIA's advantage for inference. Provide specifics: hardware (Tensor cores, RT
          cores, Ampere architecture), software (TensorRT, CUDA, cuDNN, Triton Inference Server), developer ecosystem, network
          effects, etc.\n\nAlso mention inference performance vs competition, DPUs, etc.\n\nWe'll answer concisely but thorough.
          \n\nWe also must not mention we are ChatGPT or mention policy. Provide final answer.\n\nLet's produce a clear,
          well-structured answer.\n\n<|end|><|start|>assistant<|channel|>final<|message|>**NVIDIA’s edge in AI inference comes
          from a tightly‑integrated stack of hardware, software, and ecosystem tools that together deliver the
          highest‑throughput, lowest‑latency execution of deep‑learning models.**\n\n| **Layer** | **Key Advantages** | **Why
          It Matters for Inference** |\n|-----------|-------------------|---------------------------------|\n| **GPU
          Architecture** | • **Tensor Cores** (FP16/FP8/INT8) – dedicated matrix‑multiply units that give 2–8× higher ops/W
          than classic CUDA cores. <br>• **CUDA SMs with mixed‑precision support** – can run FP16/INT8/INT4 with minimal
          accuracy loss. <br>• **NVLink / NVSwitch** – high‑bandwidth, low‑latency inter‑GPU communication for multi‑GPU
          serving. | Maximizes raw arithmetic density, letting a single GPU handle many concurrent requests. |\n|
          **Inference‑focused Silicon** | • **A100 / H100 “Tensor‑Core” GPUs** – FP8 support (H100) and sparsity acceleration
          (up to 2× throughput). <br>• **T4/Ti, L40, L4, Jetson edge modules** – power‑optimized variants for data‑center,
          edge, and robotics. | Provides the right performance‑per‑watt for cloud, on‑prem, or edge deployments. |\n|
          **Software Stack** | • **TensorRT** – a compiler & runtime that fuses layers, applies precision calibration, and
          auto‑tunes kernels for each GPU. <br>• **CUDA, cuDNN, cuBLAS** – low‑level libraries tuned for every generation of
          GPU. <br>• **Triton Inference Server** – model‑agnostic serving, model‑versioning, batching, GPU sharing, and
          scaling across CPUs/GPUs. | Turns raw GPU power into production‑ready, low‑latency services with minimal engineering
          effort. |\n| **Model Optimizations** | • **Quantization (INT8/FP8) & Structured Sparsity** – supported natively by
          TensorRT and the hardware. <br>• **Automatic Mixed‑Precision (AMP)** – retains accuracy while cutting compute. |
          Reduces memory bandwidth and compute cost while keeping accuracy within acceptable bounds. |\n| **Ecosystem &
          Compatibility** | • **Broad framework support** – TensorFlow, PyTorch, ONNX, JAX, etc., all compile to TensorRT.
          <br>• **NVIDIA NGC** – pre‑optimized model zoo, containers, and reference pipelines. <br>• **MLOps tools** –
          NVIDIA Merlin, Clara, Metropolis, etc., for recommendation, medical, vision pipelines. | Engineers can
          plug‑and‑play, accelerate, and ship models faster. |\n| **Scalability & Deployment Flexibility** | • **DGX Cloud,
          EGX, Jetson, and Orin** – end‑to‑end solutions from cloud to edge. <br>• **Multi‑Instance GPU (MIG)** – partition
          a single A100 into up to 7 isolated inference instances. <br>• **NVIDIA AI Enterprise** – managed software suite
          for on‑prem data‑centers. | Allows the same code to run on a laptop, an edge device, or a massive data‑center
          cluster. |\n| **Performance Benchmarks** | • **Industry‑leading latency/throughput** on MLPerf Inference (FP8,
          INT8). <br>• **Sparsity‑aware kernels** give >2× speedup on H100 with < 0.1 % accuracy loss. | Demonstrates
          real‑world advantage in the most respected benchmark suite. |\n|",
        "reasoning_content": null,
        "tool_calls": []
      },
      "logprobs": null,
      "finish_reason": "length",
      "stop_reason": null,
      "disaggregated_params": null
    }
  ],
  "usage": {
    "prompt_tokens": 17,
    "total_tokens": 1041,
    "completion_tokens": 1024
  },
  "prompt_token_ids": null
}

```
</details>

The server exposes a standard OpenAI-compatible API endpoint that accepts JSON
requests. You can adjust parameters like `max_tokens`, `temperature`, and
others according to your needs.


## (H200/H100 Only) Using OpenAI Triton Kernels for MoE

OpenAI ships a set of Triton kernels optimized for its MoE models. TensorRT-LLM can leverage these kernels for Hopper-based GPUs like NVIDIA's H200 for optimal performance. `TRTLLM` MoE backend is not supported on Hopper, and `CUTLASS` backend support is still ongoing.  Please enable `TRITON` backend with the steps below if you are running on Hopper GPUs.

### Installing OpenAI Triton

The `nvcr.io/nvidia/tensorrt-llm/release:gpt-oss-dev` has prepared Triton already (`echo $TRITON_ROOT` could reveal the path). In other situations, you will need to build and install a specific version of Triton. Please follow the instructions in this [link](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/gpt_oss#using-openai-triton-kernels-for-moe).


### Selecting Triton as the MoE backend

To use the Triton MoE backend with **trtllm-serve** (or other commands), add this snippet to the YAML file passed via `--extra_llm_api_options`:

```yaml
moe_config:
  backend: TRITON
```


## Troubleshooting Tips

- If you encounter CUDA out-of-memory errors, try reducing `--max_batch_size`, `--max_num_tokens`, or `--kv_cache_free_gpu_memory_fraction`. See the [doc](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/performance/performance-tuning-guide/tuning-max-batch-size-and-max-num-tokens.md) for the explanation of these parameters.
- Add `print_iter_log: true` to extra LLM API options YAML file to inspect the per-iteration log.
- Check GPU utilization with `nvidia-smi` while the server is running to inspect GPU status and memory usage.
- If the container fails to start, verify that the NVIDIA Container Toolkit is properly installed
- For connection issues, make sure port 8000 is not being used by another application
