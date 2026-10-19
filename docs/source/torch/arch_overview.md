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

Multimodal preprocessing and execution use the existing `MultimodalParams` in
[inputs/multimodal.py](../../../tensorrt_llm/inputs/multimodal.py).
The executor keeps request-wide data in `LlmRequest.py_multimodal_params`, while
encoder selection and LLM forwards create separate Params views.

| Type or member | Responsibility |
|---|---|
| `MultimodalParams.multimodal_data` | Processed model inputs, precomputed embeddings or shared-tensor handles, and model-specific auxiliary data. |
| `MultimodalParams.item_metadata` | A property reading the single `MultimodalItemMetadata` record stored in `multimodal_data`. |
| `MultimodalItemMetadata` | Prompt-ordered references to processed items, optional encoder costs and output row counts, and input partitions. |
| `MultimodalItemPartition` | A small value describing one input field's item axis and optional packed lengths. It lives inside `item_metadata.input_partitions`. |
| `MultimodalInput` | Processor-side hash, prompt-position, and exact-run data converted into the existing C++ request fields by the worker. |
| `MultimodalRuntimeData` | Derived information for the current LLM forward or prefill chunk. |
| `MultimodalEncoderRequestState` | Request-owned readiness and references to cached encoder outputs; the model still owns the `TensorLRUCache`. |

The process boundary is:

```text
Main / input processor
  -> prompt token IDs + MultimodalParams (processed data and item metadata)
  -> existing worker handoff: restore tensors; convert MultimodalInput to C++ fields
Executor / LlmRequest.py_multimodal_params
  -> admission creates encoder_state where item scheduling is supported
  -> selected encoder Params: selected data and metadata, no cache ownership
  -> LLM forward Params: embeddings and current-chunk runtime, no cache ownership
  -> executor cleanup releases request-owned cache references
```

After conversion, the worker clears the processor-side `multimodal_input`;
the executor uses the C++ request fields for that information. Text-only requests
do not need an aggregate. Execution views leave `encoder_state` unset, and the
request-wide Params do not carry the current chunk's runtime data. Existing
prefetch events retain their stream synchronization role. Live cache references
and CUDA events are not added to the cross-process payload.

An item reference identifies a processed item within its modality. It does not
necessarily identify one original media object or one tensor: a video can expand
into multiple processed items, and a field can pack multiple items in one tensor.
Partitions describe selection: a shared field has no item axis, a batched field
has an item axis, and a packed field also supplies per-item lengths. Contiguous
tensor selections can share storage; reordered selections may copy. Treat shared
input storage as read-only. Model hooks and `EncoderGroup` still determine
compatible batches, padding, masks, encoder arguments, and output fusion.

Prompt multimodal token counts, physical encoder token costs, and output embedding
row counts serve different purposes. Unknown costs or row counts remain `None`;
metadata alone does not enable item scheduling. Producers and admission boundaries
validate the required contracts rather than revalidating all items per forward.
Legacy `mm_item_order`, `multimodal_embedding_lengths`, and the three-field
`MultimodalEncoderItemMetadata` remain compatibility inputs. Migrated producers
publish the shared metadata record; the executor's output-length helper reads its
rows and rejects conflicting legacy lengths.

For encoder/prefill disaggregation (E/P), model processor hooks return
`(prompt_token_ids, MultimodalParams)` and use `MultimodalParams.from_embeddings`
to attach embeddings or their handles. See the hook contract in
[inputs/registry.py](../../../tensorrt_llm/inputs/registry.py).
`DisaggregatedParams` remains the transport envelope; this local result does not
replace the wire protocol. The prefill worker still rebuilds model-specific prompt
layout. Shared data contracts do not change the supported model/modality matrix
or introduce a common batching policy for every model.

External orchestrators can pass the existing `DisaggregatedParams` to the LLM API;
the prefill-side processor builds the local Params. Custom processors implementing
the E/P hook must follow its tuple return contract. Check those callers separately
from transport fields when changing this interface.

## Model Engine

The core component of `PyExecutor` is the `ModelEngine`, responsible for executing the model's forward pass efficiently on the GPU.
The key method of `ModelEngine` is `forward`, which handles the forward pass computation.
For the PyTorch backend, the derived class is `PyTorchModelEngine`, declared in [model_engine.py](../../../tensorrt_llm/_torch/pyexecutor/model_engine.py).

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
