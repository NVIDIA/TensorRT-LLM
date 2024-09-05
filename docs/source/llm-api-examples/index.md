# LLM Examples Introduction

## Supported Models

* Llama (including variants Mistral, Mixtral, InternLM)
* GPT (including variants Starcoder-1/2, Santacoder)
* Gemma-1/2
* Phi-1/2/3
* ChatGLM (including variants glm-10b, chatglm, chatglm2, chatglm3, glm4)
* QWen-1/1.5/2
* Falcon
* Baichuan-1/2
* GPT-J

## Model Preparation

The `LLM` class supports the following types of model inputs:

1. **Hugging Face Hub**: triggers a download from the Hugging Face model hub, such as `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.
2. **Local Hugging Face models**: uses a locally stored Hugging Face model.
3. **Local TensorRT-LLM engine**: built by `trtllm-build` tool or saved by the Python LLM API.

All kinds of the model inputs can be seamlessly integrated with the API, and the `LLM(model=<any-model-path>)` constructor can accommodate models in any of the preceding formats.

### Hugging Face Hub

Given the popularity of the Hugging Face model hub, the API supports the Hugging Face format as one of the starting points.
To use the API with Llama 3.1 models, download the model from the [Meta Llama 3.1 8B model page](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B) by using the following command:

```console
git clone https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
```

### From TensorRT-LLM Engine

There are two ways to build the TensorRT-LLM engine:

1. You can build the TensorRT-LLM engine from the Hugging Face model directly with the `trtllm-build` tool and then save the engine to disk for later use.
Refer to the [README](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama) in the `examples/llama` repository on GitHub.
2. Use an `LLM` instance to save one:

   ```python
   llm = LLM(<model-path>)

   # Save engine to local disk
   llm.save(<engine-dir>)
   ```
