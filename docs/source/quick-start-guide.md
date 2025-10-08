(quick-start-guide)=

# Quick Start Guide

This is the starting point to try out TensorRT LLM. Specifically, this Quick Start Guide enables you to quickly get set up and send HTTP requests using TensorRT LLM.


## Launch Docker Container

The [TensorRT LLM container](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release/tags) maintained by NVIDIA contains all of the required dependencies pre-installed. You can start the container on a machine with NVIDIA GPUs via:

```bash
docker run --rm -it --ipc host --gpus all --ulimit memlock=-1 --ulimit stack=67108864 -p 8000:8000 nvcr.io/nvidia/tensorrt-llm/release:x.y.z
```


(deploy-with-trtllm-serve)=
## Deploy Online Serving with trtllm-serve

You can use the `trtllm-serve` command to start an OpenAI compatible server to interact with a model.
To start the server, you can run a command like the following example inside a Docker container:

```bash
trtllm-serve "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

```{note}
If you are running `trtllm-serve` inside a Docker container, you have two options for sending API requests:
1. Expose a port (e.g., 8000) to allow external access to the server from outside the container.
2. Open a new terminal and use the following command to directly attach to the running container:
```bash
docker exec -it <container_id> bash
```

After the server has started, you can access well-known OpenAI endpoints such as `v1/chat/completions`.
Inference can then be performed using examples similar to the one provided below, from a separate terminal.

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Accept: application/json" \
    -d '{
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "messages":[{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Where is New York? Tell me in a single sentence."}],
        "max_tokens": 32,
        "temperature": 0
    }'
```

_Example Output_

```json
{
  "id": "chatcmpl-ef648e7489c040679d87ed12db5d3214",
  "object": "chat.completion",
  "created": 1741966075,
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "New York is a city in the northeastern United States, located on the eastern coast of the state of New York.",
        "tool_calls": []
      },
      "logprobs": null,
      "finish_reason": "stop",
      "stop_reason": null
    }
  ],
  "usage": {
    "prompt_tokens": 43,
    "total_tokens": 69,
    "completion_tokens": 26
  }
}
```

For detailed examples and command syntax, refer to the [trtllm-serve](commands/trtllm-serve/trtllm-serve.rst) section.

## Run Offline Inference with LLM API
The LLM API is a Python API designed to facilitate setup and inference with TensorRT LLM directly within Python. It enables model optimization by simply specifying a HuggingFace repository name or a model checkpoint. The LLM API streamlines the process by managing model loading, optimization, and inference, all through a single `LLM` instance.

Here is a simple example to show how to use the LLM API with TinyLlama.

```{literalinclude} ../../examples/llm-api/quickstart_example.py
    :language: python
    :linenos:
```

You can also directly load pre-quantized models [quantized checkpoints on Hugging Face](https://huggingface.co/collections/nvidia/model-optimizer-66aa84f7966b3150262481a4) in the LLM constructor.
To learn more about the LLM API, check out the [](llm-api/index) and [](examples/llm_api_examples).

## Quick Start for Popular Models

Below is a table containing one-line `trtllm-serve` commands that can be used to easily deploy popular models including DeepSeek-R1, gpt-oss, Llama 4, Qwen3, and more. The LLM API configuration settings have been optimized for the listed inference scenarios, though you may be able to improve performance further with additional tuning for your use case.

We maintain the LLM API configuration files with recommended settings in the [`examples/configs`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/configs) directory. The `trtllm-serve` commands below can be run as-is within the TensorRT LLM Docker container since the config files will be available at `/app/tensorrt_llm/examples/configs`. 

This table is designed to be simple; for detailed model-specific deployment guides, check out the [Model Recipes](deployment-guide/index.rst).

| Model Name | GPU | Inference Scenario | Config | Command |
|------|------|------|------|------|
| [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) | H100, H200 | Max Throughput | [deepseek-r1-throughput.yaml](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/deepseek-r1-throughput.yaml) | `trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options /app/tensorrt_llm/examples/configs/deepseek-r1-throughput.yaml` |
| [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) | B200, GB200 | Max Throughput | [deepseek-r1-deepgemm.yaml](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/deepseek-r1-deepgemm.yaml) | `trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options /app/tensorrt_llm/examples/configs/deepseek-r1-deepgemm.yaml` |
| [DeepSeek-R1 (NVFP4)](https://huggingface.co/nvidia/DeepSeek-R1-FP4) | B200, GB200 | Max Throughput | [deepseek-r1-throughput.yaml](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/deepseek-r1-throughput.yaml) | `trtllm-serve nvidia/DeepSeek-R1-FP4 --extra_llm_api_options /app/tensorrt_llm/examples/configs/deepseek-r1-throughput.yaml` |
| [DeepSeek-R1 (NVFP4)](https://huggingface.co/nvidia/DeepSeek-R1-FP4-v2) | B200, GB200 | Min Latency | [deepseek-r1-latency.yaml](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/deepseek-r1-latency.yaml) | `trtllm-serve nvidia/DeepSeek-R1-FP4-v2 --extra_llm_api_options /app/tensorrt_llm/examples/configs/deepseek-r1-latency.yaml` |
| [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) | Any | Max Throughput | [gpt-oss-120b-throughput.yaml](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/gpt-oss-120b-throughput.yaml) | `trtllm-serve openai/gpt-oss-120b --extra_llm_api_options /app/tensorrt_llm/examples/configs/gpt-oss-120b-throughput.yaml` |
| [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) | Any | Min Latency | [gpt-oss-120b-latency.yaml](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/gpt-oss-120b-latency.yaml) | `trtllm-serve openai/gpt-oss-120b --extra_llm_api_options /app/tensorrt_llm/examples/configs/gpt-oss-120b-latency.yaml` |
| Qwen3 family (e.g. [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)) | Any | Max Throughput | [qwen3.yaml](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/qwen3.yaml) | `trtllm-serve Qwen/Qwen3-30B-A3B --extra_llm_api_options /app/tensorrt_llm/examples/configs/qwen3.yaml` |
| [Llama-3.3-70B (FP8)](https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-FP8) | Any | Max Throughput | [llama-3.3-70b.yaml](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/llama-3.3-70b.yaml) | `trtllm-serve nvidia/Llama-3.3-70B-Instruct-FP8 --extra_llm_api_options /app/tensorrt_llm/examples/configs/llama-3.3-70b.yaml` |
| [Llama 4 Scout (FP8)](https://huggingface.co/nvidia/Llama-4-Scout-17B-16E-Instruct-FP8) | Any | Max Throughput | [llama-4-scout.yaml](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/llama-4-scout.yaml) | `trtllm-serve nvidia/Llama-4-Scout-17B-16E-Instruct-FP8 --extra_llm_api_options /app/tensorrt_llm/examples/configs/llama-4-scout.yaml` |

## Next Steps

In this Quick Start Guide, you have:

- Learned how to deploy a model with `trtllm-serve` for online serving
- Explored the LLM API for offline inference with TensorRT LLM

To continue your journey with TensorRT LLM, explore these resources:

- **[Installation Guide](installation/index.rst)** - Detailed installation instructions for different platforms
- **[Deployment Guide](examples/llm_api_examples)** - Comprehensive examples for deploying LLM inference in various scenarios
- **[Model Support](models/supported-models.md)** - Check which models are supported and how to add new ones
- **CLI Reference** - Explore TensorRT LLM command-line tools:
  - [`trtllm-serve`](commands/trtllm-serve/trtllm-serve.rst) - Deploy models for online serving
  - [`trtllm-bench`](commands/trtllm-bench.rst) - Benchmark model performance
  - [`trtllm-eval`](commands/trtllm-eval.rst) - Evaluate model accuracy
