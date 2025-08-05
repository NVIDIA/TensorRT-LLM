(quick-start-guide)=

# Quick Start Guide

This is the starting point to try out TensorRT-LLM. Specifically, this Quick Start Guide enables you to quickly get set up and send HTTP requests using TensorRT-LLM.

## Installation

There are multiple ways to install and run TensorRT-LLM. For most users, the options below should be ordered from simple to complex. The approaches are equivalent in terms of the supported features.

Note: **This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.**

1. [](installation/containers)

1. Pre-built release wheels on [PyPI](https://pypi.org/project/tensorrt-llm) (see [](installation/linux))

1. [Building from source](installation/build-from-source-linux)

The following examples can most easily be executed using the prebuilt [Docker release container available on NGC](https://registry.ngc.nvidia.com/orgs/nvstaging/teams/tensorrt-llm/containers/release) (see also [release.md](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docker/release.md) on GitHub). Ensure to run these commands as a user with appropriate permissions, preferably `root`, to streamline the setup process.


## Launch Docker on a node with NVIDIA GPUs deployed.

```bash
docker run --ipc host --gpus all -it nvcr.io/nvidia/tensorrt-llm/release
```
## Run Offline inference with LLM API
The LLM API is a Python API designed to facilitate setup and inference with TensorRT-LLM directly within Python. It enables model optimization by simply specifying a HuggingFace repository name or a model checkpoint. The LLM API streamlines the process by managing checkpoint conversion, engine building, engine loading, and model inference, all through a single Python object.

Here is a simple example to show how to use the LLM API with TinyLlama.

```{literalinclude} ../../examples/llm-api/quickstart_example.py
    :language: python
    :linenos:
```

You can also directly load TensorRT Model Optimizer's [quantized checkpoints on Hugging Face](https://huggingface.co/collections/nvidia/model-optimizer-66aa84f7966b3150262481a4) in the LLM constructor.
To learn more about the LLM API, check out the [](llm-api/index) and [](examples/llm_api_examples).

(deploy-with-trtllm-serve)=
## Deploy online serving with trtllm-serve

You can use the `trtllm-serve` command to start an OpenAI compatible server to interact with a model.
To start the server, you can run a command like the following example inside a Docker container:

```bash
trtllm-serve "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```


> [!NOTE]
> If you are running `trtllm-server` inside a Docker container, you have two options for sending API requests:

> 1. Expose port `8000` to access the server from outside the container.

> 2. Open a new terminal and use the following command to directly attach to the running container:

> ```bash
> docker exec -it <container_id> bash
> ```

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

For detailed examples and command syntax, refer to the [trtllm-serve](commands/trtllm-serve.rst) section.

1. Expose port `8000` to access the server from outside the container.

2. Open a new terminal and use the following command to directly attach to the running container:

```bash:docs/source/quick-start-guide.md
docker exec -it <container_id> bash
```

## Next Steps

In this Quick Start Guide, you:

- Saw an example of the LLM API
- Learned about deploying a model with `trtllm-serve`

For more examples, refer to:

- [examples](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples) for showcases of how to run a quick benchmark on latest LLMs.

## Related Information

- [Best Practices Guide](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/index.html)
- [Support Matrix](https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html)
