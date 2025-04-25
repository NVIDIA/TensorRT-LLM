# API Introduction

The LLM API is a high-level Python API and designed for LLM workflows.
This API is under development and might have breaking changes in the future.

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
* Mamba-1/2

## Model Preparation

The `LLM` class supports input from any of following:

1. **Hugging Face Hub**: Triggers a download from the Hugging Face model hub, such as `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.
2. **Local Hugging Face models**: Uses a locally stored Hugging Face model.
3. **Local TensorRT-LLM engine**: Built by `trtllm-build` tool or saved by the Python LLM API.

Any of these formats can be used interchangeably with the ``LLM(model=<any-model-path>)`` constructor.

The following sections describe how to use these different formats for the LLM API.

### Hugging Face Hub

Using the Hugging Face Hub is as simple as specifying the repo name in the LLM constructor:

```python
llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
```

You can also directly load TensorRT Model Optimizer's [quantized checkpoints](https://huggingface.co/collections/nvidia/model-optimizer-66aa84f7966b3150262481a4) on Hugging Face Hub in the same way.

### Local Hugging Face Models

Given the popularity of the Hugging Face model hub, the API supports the Hugging Face format as one of the starting points.
To use the API with Llama 3.1 models, download the model from the [Meta Llama 3.1 8B model page](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B) by using the following command:

```console
git lfs install
git clone https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
```

After the model download is complete, you can load the model:

```python
llm = LLM(model=<path_to_meta_llama_from_hf>)
```

Using this model is subject to a [particular](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) license. Agree to the terms and [authenticate with Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3-8B?clone=true) to begin the download.

### Local TensorRT-LLM Engine

There are two ways to build a TensorRT-LLM engine:

1. You can build the TensorRT-LLM engine from the Hugging Face model directly with the [`trtllm-build`](../commands/trtllm-build.rst) tool and then save the engine to disk for later use.
Refer to the [README](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama) in the [`examples/llama`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama) repository on GitHub.

   After the engine building is finished, we can load the model:

   ```python
   llm = LLM(model=<path_to_trt_engine>)
   ```

2. Use an `LLM` instance to create the engine and persist to local disk:

   ```python
   llm = LLM(<model-path>)

   # Save engine to local disk
   llm.save(<engine-dir>)
   ```

   The engine can be loaded using the `model` argument as shown in the first approach.

## Tips and Troubleshooting

The following tips typically assist new LLM API users who are familiar with other APIs that are part of TensorRT-LLM:

- RuntimeError: only rank 0 can start multi-node session, got 1

  There is no need to add an `mpirun` prefix for launching single node multi-GPU inference with the LLM API.

  For example, you can run `python llm_inference_distributed.py` to perform multi-GPU on a single node.

- Hang issue on Slurm Node

  If you experience a hang or other issue on a node managed with Slurm, add prefix `mpirun -n 1 --oversubscribe --allow-run-as-root` to your launch script.

  For example, try `mpirun -n 1 --oversubscribe --allow-run-as-root python llm_inference_distributed.py`.

- MPI_ABORT was invoked on rank 1 in communicator MPI_COMM_WORLD with errorcode 1.

  Because the LLM API relies on the `mpi4py` library, put the LLM class in a function and protect the main entrypoint to the program under the `__main__` namespace to avoid a [recursive spawn](https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#mpipoolexecutor) process in `mpi4py`.

  This limitation is applicable for multi-GPU inference only.

- Cannot quit after generation

  The LLM instance manages threads and processes, which may prevent its reference count from reaching zero. To address this issue, there are two common solutions:
  1. Wrap the LLM instance in a function, as demonstrated in the quickstart guide. This will reduce the reference count and trigger the shutdown process.
  2. Use LLM as an contextmanager, with the following code: `with LLM(...) as llm: ...`, the shutdown methed will be invoked automatically once it goes out of the `with`-statement block.
