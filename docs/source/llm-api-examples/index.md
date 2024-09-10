# LLM Examples Introduction
Here is a simple example to show how to use the LLM with TinyLlama.
```{eval-rst}
.. literalinclude:: ../../../examples/llm-api/quickstart_example.py
    :language: python
    :linenos:
```

The LLM API can be used for both offline or online usage. See more examples of the LLM API here:
* [LLM Generate](https://nvidia.github.io/TensorRT-LLM/llm-api-examples/llm_generate.html)
* [LLM Generate Distributed](https://nvidia.github.io/TensorRT-LLM/llm-api-examples/llm_generate_distributed.html)
* [LLM Generate Async](https://nvidia.github.io/TensorRT-LLM/llm-api-examples/llm_generate_async.html)
* [LLM Generate Async Streaming](https://nvidia.github.io/TensorRT-LLM/llm-api-examples/llm_generate_async_streaming.html)
* [LLM Quantization](https://nvidia.github.io/TensorRT-LLM/llm-api-examples/llm_quantization.html)
* [LLM Auto Parallel](https://nvidia.github.io/TensorRT-LLM/llm-api-examples/llm_auto_parallel.html)

For more details on how to fully utilize this API, check out:

* [Common customizations](https://nvidia.github.io/TensorRT-LLM/llm-api-examples/customization.html)
* [LLM API Reference](https://nvidia.github.io/TensorRT-LLM/llm-api/index.html)


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

The `LLM` class supports input from any of following:

1. **Hugging Face Hub**: triggers a download from the Hugging Face model hub, such as `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.
2. **Local Hugging Face models**: uses a locally stored Hugging Face model.
3. **Local TensorRT-LLM engine**: built by `trtllm-build` tool or saved by the Python LLM API.

Any of these formats can be used interchangeably with the LLM(model=<any-model-path>) constructor.
The following sections how to use get these different formats for the LLM API.


### Hugging Face Hub

Using the hugging face hub is as simple as specifying the repo name in the LLM constructor

```python
llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
```
#### Local Hugging Face Models
Given the popularity of the Hugging Face model hub, the API supports the Hugging Face format as one of the starting points.
To use the API with Llama 3.1 models, download the model from the [Meta Llama 3.1 8B model page](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B) by using the following command:

```console
git lfs install
git clone https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
```
After the model downloading finished, we can load the model as below.
```python
llm = LLM(model=<path_to_meta_llama_from_hf>)
```

Note that using this model is subject to a [particular](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) license. Agree to the terms and [authenticate with HuggingFace](https://huggingface.co/meta-llama/Meta-Llama-3-8B?clone=true) to begin the download.

### From TensorRT-LLM Engine

There are two ways to build the TensorRT-LLM engine:

1. You can build the TensorRT-LLM engine from the Hugging Face model directly with the [`trtllm-build`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/setup.py#L126) tool and then save the engine to disk for later use.
Refer to the [README](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama) in the [`examples/llama`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama) repository on GitHub.

   After the engine building is finished, we can load the model as below.
   ```python
   llm = LLM(model=<path_to_trt_engine>)
   ```

2. Use an `LLM` instance to create the engine and persist to local disk:

   ```python
   llm = LLM(<model-path>)

   # Save engine to local disk
   llm.save(<engine-dir>)
   ```
The engine can be reloaded like above.
