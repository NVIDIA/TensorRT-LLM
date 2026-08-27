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

For the PyTorch backend, the beta `LLM.startup_metrics` property reports executor construction,
model-engine warmup, and weight-loading timings from worker rank 0. Values are wall-clock seconds.
The property returns an empty dictionary when the backend does not provide startup metrics.

```python
llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
print(llm.startup_metrics)
```

A typical result has the following structure:

```json
{
  "initial_model_engine": {
    "attention_warmup_seconds": 2.911,
    "general_warmup_seconds": 0.387,
    "autotuner_warmup_seconds": 0.041,
    "mamba_hybrid_warmup_seconds": 0.000,
    "cuda_graph_warmup_seconds": 0.566,
    "cuda_graph_capture_seconds": 0.950,
    "dg_paged_mqa_warmup_seconds": 0.000,
    "cute_dsl_radix_topk_warmup_seconds": 0.000,
    "memory_pool_prepopulation_seconds": 0.094,
    "kv_cache_cleanup_seconds": 0.194,
    "total_warmup_seconds": 5.485
  },
  "final_model_engine": {
    "attention_warmup_seconds": 0.057,
    "general_warmup_seconds": 0.492,
    "autotuner_warmup_seconds": 0.046,
    "mamba_hybrid_warmup_seconds": 0.000,
    "cuda_graph_warmup_seconds": 0.599,
    "cuda_graph_capture_seconds": 0.991,
    "dg_paged_mqa_warmup_seconds": 0.000,
    "cute_dsl_radix_topk_warmup_seconds": 0.000,
    "memory_pool_prepopulation_seconds": 0.097,
    "kv_cache_cleanup_seconds": 1.657,
    "total_warmup_seconds": 4.230
  },
  "py_executor": {
    "config_and_checkpoint_loader_initialization_seconds": 0.004,
    "model_engine_creation_seconds": 1.554,
    "sampler_creation_seconds": 1.195,
    "initial_kv_cache_creation_seconds": 0.025,
    "speculative_decoding_resource_manager_creation_seconds": 0.000,
    "speculative_drafter_creation_seconds": 0.000,
    "initial_py_executor_creation_seconds_for_kv_cache_estimation": 5.766,
    "kv_cache_capacity_configuration_seconds": 0.075,
    "final_kv_cache_creation_seconds": 0.057,
    "final_py_executor_creation_seconds": 4.459,
    "worker_start_seconds": 0.000,
    "total_executor_creation_seconds": 13.373
  },
  "model_loader": {
    "checkpoint_preparation_seconds": 0.719,
    "weight_population_seconds": 0.272,
    "post_load_processing_seconds": 0.004,
    "total_model_loading_seconds": 1.545
  }
}
```

The `py_executor` property contains the timed scopes in `create_py_executor()`, which creates the [PyExecutor](../developer-guide/overview.md).

| PyExecutor metric | Scope |
|-----------------|-------|
| `config_and_checkpoint_loader_initialization_seconds` | Initialize the model config and checkpoint loader. |
| `model_engine_creation_seconds` | Construct the main model engine, which includes loading model weights. |
| `draft_model_engine_creation_seconds` | Construct the separate draft model engine in the deprecated two-model MTP setting. |
| `guided_decoder_creation_seconds` | Construct guided-decoding resources. |
| `sampler_creation_seconds` | Construct the sampler. |
| `initial_kv_cache_creation_seconds` | Construct the temporary KV cache used to estimate final capacity. |
| `speculative_decoding_resource_manager_creation_seconds` | Construct the speculative-decoding resource manager. |
| `speculative_drafter_creation_seconds` | Construct the speculative drafter. |
| `initial_py_executor_creation_seconds_for_kv_cache_estimation` | Construct the temporary PyExecutor for KV cache capacity estimation, which includes the initial model engine warmup. |
| `kv_cache_capacity_configuration_seconds` | Determine final KV cache capacity from the temporary executor and available memory. |
| `final_kv_cache_creation_seconds` | Construct the final KV cache retained for serving. |
| `final_py_executor_creation_seconds` | Using the final KV cache, construct the final PyExecutor retained for serving, which includes the final model engine warmup. |
| `worker_start_seconds` | Start the final PyExecutor worker. |
| `total_executor_creation_seconds` | Total time for the `create_py_executor()` call, including all applicable scopes above. |

The two model-engine properties contain times for various `PyTorchModelEngine` warmup stages.
`initial_model_engine` measures the timings for the initial model engine creation as part of the
`py_executor.initial_py_executor_creation_seconds_for_kv_cache_estimation` timing.
Likewise, `final_model_engine` measures the timings for the final model engine creation as part of
the `py_executor.final_py_executor_creation_seconds` timing.
The deprecated two-model speculative decoding setting may produce the additional properties
`initial_draft_model_engine` and `final_draft_model_engine`.

Each model-engine property contains the warmup scopes below. Which scopes appear depends on the model
and configuration; for example, encoder-decoder and context-parallel configurations can skip some
stages.

| Model-engine metric | Scope |
|---------------------|-------|
| `attention_warmup_seconds` | Warm up the attention backend and kernels. |
| `general_warmup_seconds` | Warm up general input shapes and release temporary workspaces. |
| `autotuner_warmup_seconds` | Run kernel autotuning warmup. |
| `mamba_hybrid_warmup_seconds` | Warm up Mamba hybrid kernels, when applicable. |
| `cuda_graph_warmup_seconds` | Run the warmup-only CUDA graph pass. |
| `cuda_graph_capture_seconds` | Capture CUDA graphs for serving. |
| `dg_paged_mqa_warmup_seconds` | Warm up DeepGEMM paged-MQA metadata, when applicable. |
| `cute_dsl_radix_topk_warmup_seconds` | Warm up the CuTe DSL radix top-k kernel, when applicable. |
| `memory_pool_prepopulation_seconds` | Pre-populate the memory pool with maximum-shape allocations. |
| `kv_cache_cleanup_seconds` | Check and clear invalid KV cache values produced during warmup. |
| `total_warmup_seconds` | Complete model-engine warmup, including KV cache cleanup. |

The `model_loader` property contains timings for loading the main LLM weights. If a draft model is used,
additional fields `draft_checkpoint_preparation_seconds` and `draft_weight_population_seconds` will appear.
A `draft_model_loader` property can also appear in the deprecated two-model MTP setting.
The `py_executor.model_engine_creation_seconds` timing includes the total `model_loader` timing.

| Metric | Description |
|--------|-------------|
| `checkpoint_preparation_seconds` | Time spent warming up, parsing and preparing checkpoint tensors for the model. Some checkpoint formats can populate model storage directly during this phase. |
| `weight_population_seconds` | Time spent copying prepared checkpoint tensors into model parameters on GPUs. This metric can be absent for formats that populate weights directly during the above checkpoint preparation phase. |
| `draft_checkpoint_preparation_seconds` | Checkpoint preparation time for draft weights loaded as part of the model loader. |
| `draft_weight_population_seconds` | Weight population time for draft weights loaded as part of the model loader. |
| `post_load_processing_seconds` | Time spent in format-specific hooks and model finalization, including post-load weight transformation, quantization and memory cleanup. |
| `total_model_loading_seconds` | Overall model construction and loading interval measured after checkpoint configuration validation. It includes the named phases below. |

`trtllm-serve` exposes the same rank-0 payload in the `startup_metrics` field of the
`GET /server_info` response:

```console
curl http://localhost:8000/server_info
```

```json
{
  "startup_metrics": {
    "model_loader": {
      "total_model_loading_seconds": 1.545,
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

When the LLM API dynamically spawns multiple MPI workers, users affected by
concurrent FlashInfer source-generation races can enable persistent, per-worker
cache slots for FlashInfer JIT artifacts. Set
`TRTLLM_FLASHINFER_WORKSPACE_PER_PROCESS=1` before creating the LLM instance.
The workaround preserves compiled artifacts between launches, and downloaded
cubins remain in FlashInfer's shared cache. It is disabled by default and can
be removed once FlashInfer guards source generation before writing shared
workspace files.

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
