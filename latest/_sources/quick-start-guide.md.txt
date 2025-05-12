(quick-start-guide)=

# Quick Start Guide

This is the starting point to try out TensorRT-LLM. Specifically, this Quick Start Guide enables you to quickly get setup and send HTTP requests using TensorRT-LLM.

## LLM API
The LLM API is a Python API designed to facilitate setup and inference with TensorRT-LLM directly within Python. It enables model optimization by simply specifying a HuggingFace repository name or a model checkpoint. The LLM API streamlines the process by managing checkpoint conversion, engine building, engine loading, and model inference, all through a single Python object.

Here is a simple example to show how to use the LLM API with TinyLlama.

```{literalinclude} ../../examples/llm-api/quickstart_example.py
    :language: python
    :linenos:
```

You can also directly load TensorRT Model Optimizer's [quantized checkpoints on Hugging Face](https://huggingface.co/collections/nvidia/model-optimizer-66aa84f7966b3150262481a4) in the LLM constructor.
To learn more about the LLM API, check out the [](llm-api/index) and [](llm-api-examples/index).

(deploy-with-trtllm-serve)=
## Deploy with trtllm-serve

You can use the `trtllm-serve` command to start an OpenAI compatible server to interact with a model.
To start the server, you can run a command like the following example:

```bash
trtllm-serve "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

After the server starts, you can access familiar OpenAI endpoints such as `v1/chat/completions`.
You can run inference such as the following example:

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

For examples and command syntax, refer to the [trtllm-serve](commands/trtllm-serve.rst) section.


## Model Definition API

### Prerequisites

- This quick start uses the Meta Llama 3.1 model. This model is subject to a particular [license](https://llama.meta.com/llama-downloads/). To download the model files, agree to the terms and [authenticate with Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct?clone=true).

- Complete the [installation](./installation/linux.md) steps.

- Pull the weights and tokenizer files for the chat-tuned variant of the Llama 3.1 8B model from the [Hugging Face Hub](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct).

  ```console
  git clone https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
  ```

(quick-start-guide-compile)=
### Compile the Model into a TensorRT Engine

Use the [Llama model definition](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama) from the `examples/llama` directory of the GitHub repository.
The model definition is a minimal example that shows some of the optimizations available in TensorRT-LLM.

```console
# From the root of the cloned repository, start the TensorRT-LLM container
make -C docker release_run LOCAL_USER=1

# Log in to huggingface-cli
# You can get your token from huggingface.co/settings/token
huggingface-cli login --token *****

# Convert the model into TensorRT-LLM checkpoint format
cd examples/llama
pip install -r requirements.txt
pip install --upgrade transformers # Llama 3.1 requires transformer 4.43.0+ version.
python3 convert_checkpoint.py --model_dir Meta-Llama-3.1-8B-Instruct --output_dir llama-3.1-8b-ckpt

# Compile model
trtllm-build --checkpoint_dir llama-3.1-8b-ckpt \
    --gemm_plugin float16 \
    --output_dir ./llama-3.1-8b-engine
```

When you create a model definition with the TensorRT-LLM API, you build a graph of operations from [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) primitives that form the layers of your neural network. These operations map to specific kernels; prewritten programs for the GPU.

In this example, we included the `gpt_attention` plugin, which implements a FlashAttention-like fused attention kernel, and the `gemm` plugin, that performs matrix multiplication with FP32 accumulation. We also called out the desired precision for the full model as FP16, matching the default precision of the weights that you downloaded from Hugging Face. For more information about plugins and quantizations, refer to the [Llama example](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama) and {ref}`precision` section.

### Run the Model

Now that you have the model engine, run the engine and perform inference.

```console
python3 ../run.py --engine_dir ./llama-3.1-8b-engine  --max_output_len 100 --tokenizer_dir Meta-Llama-3.1-8B-Instruct --input_text "How do I count to nine in French?"
```

### Deploy with Triton Inference Server

To create a production-ready deployment of your LLM, use the [Triton Inference Server backend for TensorRT-LLM](https://github.com/triton-inference-server/tensorrtllm_backend) to leverage the TensorRT-LLM C++ runtime for rapid inference execution and include optimizations like in-flight batching and paged KV caching. Triton Inference Server with the TensorRT-LLM backend is available as a [pre-built container through NVIDIA NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags).

1. Clone the TensorRT-LLM backend repository:

```console
cd ..
git clone https://github.com/triton-inference-server/tensorrtllm_backend.git
cd tensorrtllm_backend
```

2. Refer to [End to end workflow to run llama 7b](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/llama.md) in the TensorRT-LLM backend repository to deploy the model with Triton Inference Server.

## Next Steps

In this Quick Start Guide, you:

- Saw an example of the LLM API
- Learned about deploying a model with `trtllm-serve`
- Learned about the Model Definition API

For more examples, refer to:

- [examples/](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples) for showcases of how to run a quick benchmark on latest LLMs.

## Related Information

- [Best Practices Guide](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/index.html)
- [Support Matrix](https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html)
