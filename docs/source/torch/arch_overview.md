# Architecture Ovewiew

TensorRT-LLM is a toolkit designed to create optimized solutions for Large Language Model (LLM) inference.
Besides TensorRT, PyTorch can also serve as the backend for TensorRT-LLM. This document provides an overview of the PyTorch Backend architecture.

## Top Level API

The interface for PyTorch backend is `tensorrt._torch.LLM`.

```python
from tensorrt_llm._torch import LLM
llm = LLM(model=<path_to_llama_from_hf>)
```

The `LLM` also manages the tokenization and detokenization processes of the input.

## PyExecutor


Similar to the TensorRT backend, which uses [Executor API](../advanced/executor.md), the PyTorch backend employs a `PyExecutor` class.
This class has a similar interface to Executor, allowing it to be integrated into LLM as an alternative backend.
Key components of the `PyExecutor` include:

- Model Engine: Holds the language model and efficiently supports single-step model forward.
- Decoder: Generates output tokens based on Model Engine outputs. Currently, only greedy search is supported.
- Scheduler: Decides whether to allocate resources (like KV Cache) for a request and whether to run forward for each request at the current step.

The single-step flow of PyExecutor involves:

- Fetching new requests from the request queue, if any.
- Scheduling some requests.
- Running model forward for scheduled requests.
- Running the decoder using the model forward outputs for the scheduled requests.
- Adding output tokens for each request and handling finished requests.

## Model Engine

The core component of `PyExecutor` is the `ModelEngine`, responsible for executing the model's forward pass efficiently on the GPU.
The key method of `ModelEngine` is `forward`, which handles the forward pass computation.
For the PyTorch backend, the derived class is `PyTorchModelEngine`, declared in [pytorch_model_engine.py](../../../tensorrt_llm/_torch/pyexecutor/pytorch_model_engine.py).

## Decoder

The Decoder generates output tokens based on Model Engine outputs and supports greedy search decoding.

## Scheduler

The scheduler operates in two steps:

1. CapacityScheduler: Determines if there are enough resources to accommodate a request.
2. MicroBatchScheduler: Selects some requests for the model to run forward.

Both CapacityScheduler and MicroBatchScheduler currently use C++ bindings.
However, since the interfaces are implemented in Python, customization is possible.
The document [scheduler.md](./scheduler.md) explains how to implement customized scheduling logic.

## ResourceManager

`ResourceManager` helps allocate and manage these resources that may be needed to run inference for a single request.
It is a container of objects inherited from `BaseResourceManager`, each managing a specific type of resource.
There are three important interfaces for `BaseResourceManager`:

- `prepare_resources`: Called at each step before model forward in PyExecutor for the current batch.
- `update_resources`: Called at each step finish for the current batch.
- `free_resources`: Called at each request finish.

One crucial resource is the KV Cache for transformer models. The `BaseResourceManager` for KV Cache is `KVCacheManager`.

### KVCacheManager

Currently, the KVCacheManager uses C++ binding. However, customization in Python is possible, as its interface is implemented in Python.
The document [kv_cache_manager.md](./kv_cache_manager.md) details how to implement a customized KVCacheManager.
