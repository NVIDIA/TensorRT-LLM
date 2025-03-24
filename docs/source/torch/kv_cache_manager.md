# KV Cache Manager

In Transformer-based models, the KV (Key-Value) Cache is a mechanism used to optimize decoding efficiency, particularly during autoregressive generation tasks.
Since KV Cache requires memory to store, it is also an important resource.
In TensorRT-LLM, KV Cache is managed by the `KVCacheManager`.

## KV Cache Manager Introduction

`KVCacheManager` is a type of resource manager, inheriting from `BaseResourceManager`.
Therefore, it implements the interfaces declared by `BaseResourceManager`.

Note: As the project evolves, these interfaces may change.

## Interfaces

The interfaces from `BaseResourceManager` include:

- **prepare_resources**: Called at each step before model forward in `PyExecutor` for the current batch.
  In `KVCacheManager`, this involves allocating KV Cache memory. This allocation varies depending on the request type.
  For requests entering the context phase for the first time, KV Cache needs to be allocated for the entire context.
  For requests already in the generation phase, KV Cache is allocated for the upcoming step.
  If KV Cache is organized in blocks and free space is available within a block, actual allocation may not occur.
- **update_resources**: Called at the end of each step for the current batch to update allocated resources.
  For KV Cache, updates may not be necessary, so this function currently performs no operations.
  If KV Cache reuse is supported in Python, updates like KV Cache Radix Tree management occurs here.
- **free_resources**: Called when a request finishes to free the resources allocated for that request.
  For KV Cache, if reuse is not enabled, the KV Cache memory used by the request should be recycled.
  In the C++ binding implementation, this might involve calling the binding's `remove_sequence` method to free the KV Cache memory related to that request.


There are also two interfaces designed for `CapacityScheduler`:

- **get_max_resource_count**: Queries the maximum number of resources available. For `KVCacheManager`, this is usually the maximum number of KV Cache blocks.
- **get_needed_resource_to_completion**: Computes the resources needed for a single request to complete.
  `CapacityScheduler` uses this to sum up the total resources needed and determine if new requests can be accommodated.

In addition to the `BaseResourceManager` interfaces, `KVCacheManager` has interfaces related to the `ModelEngine` in use.
For `PyTorchModelEngine`, common interfaces include:

- **get_batch_cache_indices**: Takes a list of `LlmRequest` and returns a `Dict[List[int]]`, indicating the block IDs for each request.
- **get_buffers**: Returns the buffer of the KV Cache pool for a given layer, used by the attention backend. The shape might be [`num_blocks`, 2, `num_tokens_per_block`, `num_kv_heads`, `head_dim`].
- **get_num_free_blocks**: Returns the number of free blocks available for allocation.

There are also interfaces for warming up `PyTorchModelEngine`, especially when using CUDA graphs:

- **add_padding_request**: Adds a sequence of context length 1 to KV Cache as a warmup request.
  This is optional if CUDA Graph is not used in your proof of concept.

## Customize KV Cache Manager

To customize `KVCacheManager`, implement all the necessary interfaces.
Then, integrate it into the `PyExecutor`. For the PyTorch backend, the relevant code is in [pytorch_model_registry.py](../../../tensorrt_llm/_torch/pyexecutor/backend_registries/pytorch_model_registry.py).
In the `create_pytorch_model_based_executor` function, the `KVCacheManager` is instantiated as follows:

```python
    kv_cache_manager = KVCacheManager(
        executor_config.kv_cache_config,
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=model_engine.model.config.num_hidden_layers,
        num_kv_heads=model_engine.model.config.num_key_value_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=max_num_requests,
        mapping=mapping,
        dtype=kv_cache_dtype,
    )
```

For local testing or proof of concept, update these lines to use your implementation.
Then, test it to ensure the `PyExecutor` runs with your customized `KVCacheManager`.
