# LLM API Introduction

The LLM API is a high-level Python API designed to streamline LLM inference workflows.

It supports a broad range of use cases, from single-GPU setups to multi-GPU and multi-node deployments, with built-in support for various parallelism strategies and advanced features. The LLM API integrates seamlessly with the broader inference ecosystem, including NVIDIA [Dynamo](https://github.com/ai-dynamo/dynamo).

While the LLM API simplifies inference workflows with a high-level interface, it is also designed with flexibility in mind. Under the hood, it uses a PyTorch-native and modular backend, making it easy to customize, extend, or experiment with the runtime.


## Quick Start Example
A simple inference example with TinyLlama using the LLM API:

```{literalinclude} ../../../examples/llm-api/quickstart_example.py
    :language: python
    :linenos:
```

For more advanced usage including distributed inference, multimodal, and speculative decoding, please refer to this [README](../../../examples/llm-api/README.md).

## Model Input

The `LLM()` constructor accepts either a Hugging Face model ID or a local model path as input.

### 1. Using a Model from the Hugging Face Hub

To load a model directly from the [Hugging Face Model Hub](https://huggingface.co/), simply pass its model ID (i.e., repository name) to the LLM constructor. The model will be automatically downloaded:

```python
llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
```

You can also use [quantized checkpoints](https://huggingface.co/collections/nvidia/model-optimizer-66aa84f7966b3150262481a4) (FP4, FP8, etc) of popular models provided by NVIDIA in the same way.

### 2. Using a Local Hugging Face Model

To use a model from local storage, first download it manually:

```console
git lfs install
git clone https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
```

Then, load the model by specifying a local directory path:

```python
llm = LLM(model=<local_path_to_model>)
```

> **Note:** Some models require accepting specific [license agreements](https://ai.meta.com/resources/models-and-libraries/llama-downloads/). Make sure you have agreed to the terms and authenticated with Hugging Face before downloading.

## Startup Metrics

For the PyTorch backend, the beta `LLM.startup_metrics` property reports weight-loading timings from
worker rank 0. Values are wall-clock seconds. The property returns an empty dictionary when the
backend does not provide startup metrics.

```python
llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
print(llm.startup_metrics)
```

A typical result has the following structure:

```json
{
  "model_loader": {
    "total_model_loading_seconds":  1.971,
    "checkpoint_preparation_seconds": 1.177,
    "weight_population_seconds": 0.598,
    "post_load_processing_seconds": 0.005
  }
}
```

The `model_loader` object contains timings for the main LLM weights. If a draft model is used,
additional field `draft_checkpoint_preparation_seconds` and `draft_weight_population_seconds` will appear.
A `draft_model_loader` object can also appear in deprecated 2-model style MTP setting.

| Metric | Description |
|--------|-------------|
| `total_model_loading_seconds` | Overall model construction and loading interval measured after checkpoint configuration validation. It includes the named phases below. |
| `checkpoint_preparation_seconds` | Time spent warming up, parsing and preparing checkpoint tensors for the model. Some checkpoint formats can populate model storage directly during this phase. |
| `weight_population_seconds` | Time spent copying prepared checkpoint tensors into model parameters on GPUs. This metric can be absent for formats that populate weights directly during the above checkpoint preparation phase. |
| `draft_checkpoint_preparation_seconds` | Checkpoint preparation time for draft weights loaded as part of the model loader. |
| `draft_weight_population_seconds` | Weight population time for draft weights loaded as part of the model loader. |
| `post_load_processing_seconds` | Time spent in format-specific hooks and model finalization, including post-load weight transformation, quantization and memory cleanup. |

`trtllm-serve` exposes the same rank-0 payload in the `startup_metrics` field of the
`GET /server_info` response:

```console
curl http://localhost:8000/server_info
```

```json
{
  "startup_metrics": {
    "model_loader": {
      "total_model_loading_seconds": 1.971,
      ...
    }
  }
}
```


## Tips and Troubleshooting

The following tips typically assist new LLM API users who are familiar with other APIs that are part of TensorRT-LLM:

### RuntimeError: only rank 0 can start multi-node session, got 1

  There is no need to add an `mpirun` prefix for launching single node multi-GPU inference with the LLM API.

  For example, you can run `python llm_inference_distributed.py` to perform multi-GPU on a single node.

### Hang issue on Slurm Node

  If you experience a hang or other issue on a node managed with Slurm, add prefix `mpirun -n 1 --oversubscribe --allow-run-as-root` to your launch script.

  For example, try `mpirun -n 1 --oversubscribe --allow-run-as-root python llm_inference_distributed.py`.

### MPI_ABORT was invoked on rank 1 in communicator MPI_COMM_WORLD with errorcode 1.

  Because the LLM API relies on the `mpi4py` library, put the LLM class in a function and protect the main entrypoint to the program under the `__main__` namespace to avoid a [recursive spawn](https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#mpipoolexecutor) process in `mpi4py`.

  This limitation is applicable for multi-GPU inference only.

### FlashInfer JIT workspace for dynamically spawned MPI workers

When the LLM API dynamically spawns multiple MPI workers, each worker is given
its own persistent cache slot for FlashInfer JIT artifacts. Sharing one
workspace lets a worker relink a module's `.so` in place while a peer has it
mapped, which raises `SIGBUS` in the peer; the per-worker slots avoid that. The
isolation preserves compiled artifacts between launches, and downloaded cubins
remain in FlashInfer's shared cache. Set
`TRTLLM_FLASHINFER_WORKSPACE_PER_PROCESS=0` to restore the shared workspace.
Both the isolation and this variable can be removed once FlashInfer guards
source generation and linking against concurrent readers.

An explicitly configured `FLASHINFER_WORKSPACE_BASE` takes precedence. Workers
started outside the LLM API's dynamic MPI pool must configure their own
workspace isolation.

### Cannot quit after generation

  The LLM instance manages threads and processes, which may prevent its reference count from reaching zero. To address this issue, there are two common solutions:
  1. Wrap the LLM instance in a function, as demonstrated in the quickstart guide. This will reduce the reference count and trigger the shutdown process.
  2. Use LLM as a context manager, with the following code: `with LLM(...) as llm: ...`, the shutdown method will be invoked automatically once it goes out of the `with`-statement block.

### Single node hanging when using `docker run --net=host`

The root cause may be related to `mpi4py`. There is a [workaround](https://github.com/mpi4py/mpi4py/discussions/491#discussioncomment-12660609) suggesting a change from `--net=host` to `--ipc=host`, or setting the following environment variables:

```bash
export OMPI_MCA_btl_tcp_if_include=lo
export OMPI_MCA_oob_tcp_if_include=lo
```

Another option to improve compatibility with `mpi4py` is to launch the task using:

```bash
mpirun -n 1 --oversubscribe --allow-run-as-root python my_llm_task.py
```

This command can help avoid related runtime issues.
