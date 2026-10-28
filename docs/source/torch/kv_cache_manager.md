# KV Cache Manager

In Transformer-based models, the KV (Key-Value) Cache is a mechanism used to optimize decoding efficiency, particularly during autoregressive generation tasks.
Since KV Cache requires memory to store, it is also an important resource.
In TensorRT LLM, KV Cache is managed by the `KVCacheManager`.

For details of the TensorRT LLM `KVCacheManager` implementation see [KV Cache Management](../legacy/advanced/kv-cache-management.md).

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
  If KV Cache reuse is supported in Python, updates like KV Cache Radix Tree management occur here.
- **free_resources**: Called when a request finishes to free the resources allocated for that request.
  For KV Cache, if reuse is not enabled, the KV Cache memory used by the request should be recycled.
  In the C++ binding implementation, this might involve calling the binding's `remove_sequence` method to free the KV Cache memory related to that request.


There are also two interfaces designed for `CapacityScheduler`:

- **get_max_resource_count**: Queries the maximum number of resources available. For `KVCacheManager`, this is usually the maximum number of KV Cache blocks.
- **get_needed_resource_to_completion**: Computes the resources needed for a single request to complete.
  `CapacityScheduler` uses this to sum up the total resources needed and determine if new requests can be accommodated.

In addition to the `BaseResourceManager` interfaces, `KVCacheManager` has interfaces related to the `ModelEngine` in use.
For `ModelEngine`, common interfaces include:

- **get_batch_cache_indices**: Takes a list of `LlmRequest` and returns a `Dict[List[int]]`, indicating the block IDs for each request.
- **get_buffers**: Returns the buffer of the KV Cache pool for a given layer, used by the attention backend. The shape might be [`num_blocks`, 2, `num_tokens_per_block`, `num_kv_heads`, `head_dim`].
- **get_num_free_blocks**: Returns the number of free blocks available for allocation.

There are also interfaces for warming up `ModelEngine`, especially when using CUDA graphs:

- **add_padding_request**: Adds a sequence of context length 1 to KV Cache as a warmup request.
  This is optional if CUDA Graph is not used in your proof of concept.

## Customize KV Cache Manager

To customize `KVCacheManager`, implement all the necessary interfaces.
Then, integrate it into the `PyExecutor`. For the PyTorch backend, the relevant code is in [_util.py](../../../tensorrt_llm/_torch/pyexecutor/_util.py).
In `KvCacheCreator._create_kv_cache_manager`, the `KVCacheManager` is instantiated along these lines:

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


## Native V2 buffer reuse policies

Custom cache managers can override a layer's reuse policy per buffer through
`tensorrt_llm.runtime.kv_cache_manager_v2`. Both attention and SSM layers default
to `ReusePolicy.REQUIRED`; a buffer's `reuse_policy=None` inherits its layer's
policy. For example, a custom recurrent consumer with independent state indices
can configure:

```python
from tensorrt_llm.runtime.kv_cache_manager_v2 import BufferConfig, ReusePolicy, SsmLayerConfig

layer = SsmLayerConfig(
    layer_id=0,
    reuse_policy=ReusePolicy.REQUIRED,
    buffers=[
        BufferConfig(role="state", size=4096),
        BufferConfig(role="checkpoint", size=2048, reuse_policy=ReusePolicy.OPTIONAL),
        BufferConfig(role="scratch", size=1024, reuse_policy=ReusePolicy.PRIVATE),
    ],
)
```

REQUIRED buffers constrain the reusable prefix. OPTIONAL buffers are candidates
at that endpoint; select their group IDs through `resume(optional_reuse_groups=...)`
on the first resume. Attention candidates retain the block-aligned sliding-window
checkpoint contract; an SSM snapshot must match the endpoint exactly. PRIVATE
buffers remain request-local and are never published as reusable checkpoints.
The consumer must initialize or reconstruct private and unselected optional state
before inference. Unselected optional snapshots count as neither hits nor misses.

Use `manager.get_buffer_group_id(layer_id, role)` to resolve a buffer and
`manager.buffer_grouping[group_id]` to enumerate membership. Buffers with the same
lifecycle properties share a group, including across layers; selecting a group
selects all of its buffers. Policy overrides add at most three policy variants
per attention window/sink/sparsity combination and at most three SSM groups.
Coalescing remains within lifecycle groups and compatible buffer sizes/coalescing
groups.

`get_layer_group_id(layer_id)` and `layer_grouping` are deprecated in documentation
only and produce no deprecation warnings. The former returns a uniform layer's
group and raises `ValueError` for a split layer. The latter preserves its
partition-of-layers result and raises if any layer is split. `LayerGroupId` retains
its name; it identifies a lifecycle group, not a physical storage pool.

Existing Python cache-manager consumers continue using the legacy layer APIs
and require one lifecycle per layer. They do not adopt per-buffer policy splits;
the legacy API rejects ambiguous layer mappings. Consumers can migrate to the
buffer APIs when they add support for independent indices. Current Mamba kernels
couple recurrent, convolution, PLE and replay buffers through one state-slot
index; separate replay indices are a follow-up. This native configuration does
not add an LLM API configuration field.

The native page-index converter also supports unequal layer shares in scratch
slots, including a layer's block crossing two physical slots. Existing attention
scratch-table kernels require whole layer blocks per slot. Custom consumers of
unequal shares that cross slots must use the native converter or supply compatible
index generation; existing Python consumers have not adopted that layout.
