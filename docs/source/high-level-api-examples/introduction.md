# High Level API(HLAPI) Introduction

## Concept


## HLAPI Supported Model
* LLaMA (including variants Mistral, Mixtral, InternLM)
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

1. **Hugging Face model name**: triggers a download from the Hugging Face model hub, e.g. `TinyLlama/TinyLlama-1.1B-Chat-v1.0` in the quickstart.
2. **Local Hugging Face models**: uses a locally stored Hugging Face model.
3. **Local TensorRT-LLM engine**: built by `trtllm-build` tool or saved by the HLAPI


All kinds of the model inputs can be seamlessly integrated with the HLAPI, and the `LLM(model=<any-model-path>)` construcotr can accommodate models in any of the above formats.

Let's delve into the preparation of the three kinds of local model formats.

### Option 1: From Hugging Face models
Given its popularity, the TensorRT-LLM HLAPI chooses to support Hugging Face format as one of the start points, to use the HLAPI on LLaMA3.1 models, you need to download the model from [LLaMA3.1 8B model page](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B) via below command
```bash
git clone https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
```

### Option 2: From TensorRT-LLM engine
There are two ways to build the TensorRT-LLM engine:

1. You can build the TensorRT-LLM engine from the Hugging Face model directly with the `trtllm-build` tool, and save the engine to disk for later use.  Please consult the LLaMA's [README](../llama/README.md).
2. Use the HLAPI to save one:

```python
llm = LLM(<model-path>)

# Save engine to local disk
llm.save(<engine-dir>)
```
