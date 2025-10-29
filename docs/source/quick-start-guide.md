(quick-start-guide)=

# Quick Start Guide

This is the starting point to try out TensorRT LLM. Specifically, this Quick Start Guide enables you to quickly get set up and send HTTP requests using TensorRT LLM.


## Launch Docker on a node with NVIDIA GPUs deployed

```bash
docker run --rm -it --ipc host --gpus all --ulimit memlock=-1 --ulimit stack=67108864 -p 8000:8000 nvcr.io/nvidia/tensorrt-llm/release:x.y.z
```


(deploy-with-trtllm-serve)=
## Deploy online serving with trtllm-serve

You can use the `trtllm-serve` command to start an OpenAI compatible server to interact with a model.
To start the server, you can run a command like the following example inside a Docker container:

```bash
trtllm-serve "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

```{note}
If you are running trtllm-server inside a Docker container, you have two options for sending API requests:
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

## Run Offline inference with LLM API
The LLM API is a Python API designed to facilitate setup and inference with TensorRT LLM directly within Python. It enables model optimization by simply specifying a HuggingFace repository name or a model checkpoint. The LLM API streamlines the process by managing model loading, optimization, and inference, all through a single `LLM` instance.

Here is a simple example to show how to use the LLM API with TinyLlama.

```{literalinclude} ../../examples/llm-api/quickstart_example.py
    :language: python
    :linenos:
```

You can also directly load pre-quantized models [quantized checkpoints on Hugging Face](https://huggingface.co/collections/nvidia/model-optimizer-66aa84f7966b3150262481a4) in the LLM constructor.
To learn more about the LLM API, check out the [](llm-api/index) and [](examples/llm_api_examples).

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
