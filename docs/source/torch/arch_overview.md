# Architecture Overview

TensorRT LLM is a toolkit designed to create optimized solutions for Large Language Model (LLM) inference.
PyTorch is the sole execution backend of TensorRT LLM; the former TensorRT engine backend has been removed (see the [migration guide](../legacy/tensorrt-backend-removal.md)). This document provides an overview of the PyTorch backend architecture.

## Top Level API

The interface for PyTorch backend is `tensorrt_llm.LLM`.

```python
from tensorrt_llm import LLM
llm = LLM(model=<path_to_llama_from_hf>)
```

The `LLM` also manages the tokenization and detokenization processes of the input.

## PyExecutor

Key components of the `PyExecutor` include:

- Model Engine: Holds the language model and efficiently supports single-step model forward.
- Decoder: Generates output tokens based on Model Engine outputs.
- Scheduler: Decides whether to allocate resources (like KV Cache) for a request and whether to run forward for each request at the current step.

The single-step flow of PyExecutor involves:

- Fetching new requests from the request queue, if any.
- Scheduling some requests.
- Running model forward for scheduled requests.
- Running the decoder using the model forward outputs for the scheduled requests.
- Adding output tokens for each request and handling finished requests.

## Multimodal Request Data

See the [multimodal developer guide](../developer-guide/multimodal.md) for
shared data contracts, process boundaries, and request ownership.

## Model Engine

[`ModelEngine`](../../../tensorrt_llm/_torch/pyexecutor/model_engine.py) loads the model, resolves capacity and execution settings, constructs a model runner, and coordinates its lifecycle.
`PyExecutor` calls `ModelEngine.forward` with scheduled requests, runtime resources, and the current speculative decoding state; the engine builds `ScheduledInputs` and delegates to a `ScheduledModelRunner`.
`EncoderExecutor` instead passes `PackedInputs` through the engine to a `PackedModelRunner`.

The concrete runner owns input preparation, model execution, CUDA graphs, metadata, and output processing.
`EncoderDecoderRunner` combines decoder execution with a separate encoder stage, which `PyExecutor` schedules through explicit encoder entry points.
Multimodal item encoding also has a separate scheduled entry point.
The engine retains the shared services and coordinates warmup, late-installed hooks, and resource release.

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
