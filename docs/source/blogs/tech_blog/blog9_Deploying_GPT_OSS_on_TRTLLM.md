# Running a High Performance GPT-OSS-120B Inference Server with TensorRT LLM

In the guide below, we will walk you through how to launch your own
high-performance TensorRT LLM server for **gpt-oss-120b** for inference.
This guide covers both low-latency and max-throughput cases.

The typical use case for **low-latency**, is when we try to maximize the number of tokens per second per user with a limited concurrency (4, 8 or 16 users).

For **maximum throughput**, the goal is to maximize the amount of tokens produced per GPU per second. The former is an indication of how fast a system can produce tokens, the latter measures how many tokens a "chip" can generate per unit of time.


## Prerequisites

- 1x NVIDIA B200/GB200/H200 GPU (8x NVIDIA B200/H200 GPUs or 4x GB200 GPUs in a single node recommended for higher performance)
- CUDA Toolkit 12.8 or later
- Docker with NVIDIA Container Toolkit installed
- Fast SSD storage for model weights
- Access to the gpt-oss-120b model checkpoint

We have a forthcoming guide for getting great performance on H100, however this guide focuses on the above GPUs.


## Launching the TensorRT LLM docker container

The container image that you will use will be pulled from NVIDIA's NGC. This container is multi-platform and will run on both x64 and arm64 architectures: `nvcr.io/nvidia/tensorrt-llm/release:gpt-oss-dev`

Run the follow docker command to start the TensorRT LLM container in interactive mode:

```bash
docker run --rm --ipc=host -it \
  --ulimit stack=67108864 \
  --ulimit memlock=-1 \
  --gpus all \
  -p 8000:8000 \
  -e TRTLLM_ENABLE_PDL=1 \
  -e TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL=True \
  -v ~/.cache:/root/.cache:rw \
  nvcr.io/nvidia/tensorrt-llm/release:gpt-oss-dev \
  /bin/bash
```


This command:
- Automatically removes the container when stopped (`--rm`)
- Allows container to interact with the host's IPC resources and shared memory for optimal performance (`--ipc=host`)
- Runs the container in interactive mode (`-it`)
- Sets up shared memory and stack limits for optimal performance
- Maps port 8000 from the container to your host
- enables PDL for low-latency perf optimization
- disables parallel weight loading

Lastly the container mounts your user `.cache` directory to save the downloaded model checkpoints which are saved to `~/.cache/huggingface/hub/` by default. This prevents having to redownload the weights each time you rerun the container.


## Running the TensorRT LLM Server

As pointed out in the introduction, this guide covers low-latency and max-throughput cases. Each requires a different configurations and commands to run. We will first cover the Low-Latency use-case, followed by the max throughput use-case.

### Low-latency Use-Case

#### Creating the Extra Options Configuration

To run a server for low-latency workloads, create a YAML configuration file, `low_latency.yaml`, as follows:

```yaml
cat <<EOF > low_latency.yaml
enable_attention_dp: false
enable_mixed_sampler: true
cuda_graph_config:
    max_batch_size: 8
    enable_padding: true
moe_config:
    backend: TRTLLM
EOF
```

> Note: If you are using NVIDIA H200 GPUs it is highly recommended to set the `moe_config.backend` to TRITON to use the OpenAI Triton MoE kernel. See the section [(H200 Only) Using OpenAI Triton Kernels for MoE](#h200-only-using-openai-triton-kernels-for-moe) for more details.


#### Launching TensorRT LLM Serve

To launch the TensorRT LLM Server to serve the model with the **low latency** config, run the following command. Commands for different GPU configurations are provided (1xGPU, 8xGPU, 4xGPU):

<details open> <summary>1x B200/GB200/H200</summary>

```bash
mpirun -n 1 --oversubscribe --allow-run-as-root \
trtllm-serve  openai/gpt-oss-120b \
  --host 0.0.0.0 \
  --port 8000 \
  --backend pytorch \
  --tp_size 1 \
  --ep_size 1 \
  --trust_remote_code \
  --extra_llm_api_options low_latency.yaml \
  --kv_cache_free_gpu_memory_fraction 0.75
```
</details>

<details> <summary>8x B200/H200</summary>

```bash
mpirun -n 1 --oversubscribe --allow-run-as-root \
trtllm-serve  openai/gpt-oss-120b \
  --host 0.0.0.0 \
  --port 8000 \
  --backend pytorch \
  --tp_size 8 \
  --ep_size 8 \
  --trust_remote_code \
  --extra_llm_api_options low_latency.yaml \
  --kv_cache_free_gpu_memory_fraction 0.75
```
</details>

<details> <summary>4x GB200/B200/H200</summary>

```bash
mpirun -n 1 --oversubscribe --allow-run-as-root \
trtllm-serve  openai/gpt-oss-120b \
  --host 0.0.0.0 \
  --port 8000 \
  --backend pytorch \
  --tp_size 4 \
  --ep_size 4 \
  --trust_remote_code \
  --extra_llm_api_options low_latency.yaml \
  --kv_cache_free_gpu_memory_fraction 0.75
```
</details>




### Max-Throughput Use-Case

#### Creating the Extra Options Configuration

To run a server for max-throughput workloads, create a YAML configuration file,
`max_throughput.yaml`, as follows:

```yaml
cat <<EOF > max_throughput.yaml
enable_attention_dp: true
cuda_graph_config:
    max_batch_size: 640
    enable_padding: true
stream_interval: 10
moe_config:
    backend: CUTLASS
EOF
```

> Note: If you are using NVIDIA H200 GPUs it is highly recommended to set the `moe_config.backend` to TRITON to use the OpenAI Triton MoE kernel. See the section [(H200 Only) Using OpenAI Triton Kernels for MoE](#h200-only-using-openai-triton-kernels-for-moe) for more details.

#### Launching TensorRT LLM Serve

To launch the TensorRT LLM Server to serve the model with the **max throughput** config, run the following command. Commands for different GPU configurations are provided (1xGPU, 8xGPU, 4xGPU):

<details open> <summary>1x B200/GB200/H200</summary>

```bash
mpirun -n 1 --oversubscribe --allow-run-as-root \
trtllm-serve  openai/gpt-oss-120b \
  --host 0.0.0.0 \
  --port 8000 \
  --backend pytorch \
  --tp_size 1 \
  --ep_size 1 \
  --max_batch_size 640 \
  --trust_remote_code \
  --extra_llm_api_options max_throughput.yaml \
  --kv_cache_free_gpu_memory_fraction 0.9
```
</details>

<details> <summary>8x B200/H200</summary>

```bash
mpirun -n 1 --oversubscribe --allow-run-as-root \
trtllm-serve  openai/gpt-oss-120b \
  --host 0.0.0.0 \
  --port 8000 \
  --backend pytorch \
  --tp_size 8 \
  --ep_size 8 \
  --max_batch_size 640 \
  --trust_remote_code \
  --extra_llm_api_options max_throughput.yaml \
  --kv_cache_free_gpu_memory_fraction 0.9
```
</details>

<details> <summary>4x GB200/B200/H200</summary>

```bash
mpirun -n 1 --oversubscribe --allow-run-as-root \
trtllm-serve  openai/gpt-oss-120b \
  --host 0.0.0.0 \
  --port 8000 \
  --backend pytorch \
  --tp_size 4 \
  --ep_size 4 \
  --max_batch_size 640 \
  --trust_remote_code \
  --extra_llm_api_options max_throughput.yaml \
  --kv_cache_free_gpu_memory_fraction 0.9
```
</details>


This command:
- Maps port 8000 from the container to your host
- Uses the PyTorch backend and specifies the tensor and expert parallel sizes
- References the low latency or max throughput configuration file for extra options
- Configures memory settings for optimal performance
- Enables all GPUs with attention data parallelism for the max throughput scenario

The initialization may take several minutes as it loads and optimizes the models.


## (H200 Only) Using OpenAI Triton Kernels for MoE

OpenAI ships a set of Triton kernels optimized for its MoE models. TensorRT LLM can leverage these kernels for Hopper based GPUs like NVIDIA's H200 for best performance. The NGC TensorRT LLM container image mentioned above already includes the required kernels so you do not need to build or install them. It is highly recommended to enable them with the steps below:

### Selecting Triton as the MoE backend

To use the Triton MoE backend with **trtllm-serve** (or other similar commands) add this snippet to the YAML file passed via `--extra_llm_api_options`:

```yaml
moe_config:
  backend: TRITON
```

Alternatively the TRITON backend can be enabled by passing the CLI flag to the trtllm-server command at runtime:

```bash
--moe_backend TRITON
```


## Test the Server with a Sample Request

You can query the health/readiness of the server using

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
            "content": "What is NVIDIAs advantage for inference?"
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


## Troubleshooting Tips

- If you encounter CUDA out-of-memory errors, try reducing `max_batch_size`, `max_seq_len`, or `--kv_cache_free_gpu_memory_fraction`
- Ensure your model checkpoints are compatible with the expected format
- For performance issues, check GPU utilization with `nvidia-smi` while the server is running
- If the container fails to start, verify that the NVIDIA Container Toolkit is properly installed
- For connection issues, make sure port 8000 is not being used by another application


## Performance Tuning

The configuration provided is optimized for 8xB200 GPUs, but you can adjust
several parameters for your specific workload:

- `max_batch_size`: Controls how many requests can be batched together
- `max_draft_len`: The number of tokens Eagle can speculate ahead
- `kv_cache_free_gpu_memory_fraction`: Controls memory allocation for the KV cache
