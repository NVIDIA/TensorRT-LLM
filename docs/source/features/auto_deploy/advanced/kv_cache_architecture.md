# KV Cache Architecture

## Overview

The caching system in AutoDeploy manages KV caches for attention layers, SSM/convolution states for Mamba models, and other stateful resources. The architecture is built around three key concepts:

1. **Resource Handlers** - Abstract descriptions of cache resources (shape, dtype, layout)
1. **CachedSequenceInterface** - The central manager that collects handlers and allocates caches
1. **KVCacheManager / MambaHybridCacheManager** - Low-level memory managers from the executor

## Flowchart: Cache Collection and Allocation Pipeline

```{mermaid}
flowchart TD
    subgraph Phase1["PHASE 1: RESOURCE HANDLER COLLECTION"]
        A1[AttentionDescriptor implementations]
        A2["get_cache_initializers(node, config)"]
        A3["cm.add_resource(k_indexed, handler)"]
        A1 --> A2
        A2 -->|"Returns ResourceHandlerDict"| A3
    end

    subgraph Phase2["PHASE 2: INITIALIZE CACHES"]
        B1["_resource_lookup: Dict"]
        B2["Initialize _caches with None values"]
        B3["Order matches _resource_lookup"]
        B1 --> B2
        B2 --> B3
    end

    subgraph Phase3["PHASE 3: COMPATIBILITY CHECKING"]
        C1["_identify_managed_kv_resources()"]
        C2["_identify_managed_state_resources()"]
    end

    subgraph Phase4["PHASE 4: CACHE MANAGER CREATION"]
        D1{"Has state resources?"}
        D2["Create MambaHybridCacheManager"]
        D3["Create KVCacheManager"]
        D4["VIEW ASSIGNMENT"]
        D5["self._caches = tensor_view"]
        D1 -->|YES| D2
        D1 -->|NO| D3
        D2 --> D4
        D3 --> D4
        D4 --> D5
    end

    subgraph Phase5["PHASE 5: UNMANAGED RESOURCE ALLOCATION"]
        E1["handler.allocate(sequence_info)"]
    end

    subgraph Phase6["PHASE 6: OPTIONAL RESIZE"]
        F1["Recreate KVCacheManager with optimal capacity"]
    end

    Phase1 --> Phase2
    Phase2 --> Phase3
    Phase3 --> Phase4
    Phase4 --> Phase5
    Phase5 --> Phase6
```

## Detailed Pipeline Flow

For reference, here's the detailed text-based flow:

```text
PHASE 1: RESOURCE HANDLER COLLECTION (insert_cached_attention transform)
├── AttentionDescriptor implementations (e.g., FlashinferCachedAttention)
├── get_cache_initializers(node, config)
│   ├── Extracts shapes from FakeTensors
│   └── Returns ResourceHandlerDict {"kv_cache": KVPagedResourceHandler, ...}
└── For each attention node (idx=0,1,2...):
    └── cm.add_resource(f"{k}_{idx}", handler)  → Stores in CachedSequenceInterface

PHASE 2: INITIALIZE CACHES (initialize_resources)
├── _resource_lookup contains all collected handlers
└── Initialize _caches dict with None values (same order as _resource_lookup)

PHASE 3: COMPATIBILITY CHECKING (dynamically from _resource_lookup)
├── _identify_managed_kv_resources()
│   ├── Iterate _resource_lookup, find first KVPagedResourceHandler → kv_ref
│   ├── All KVPagedResourceHandlers matching kv_ref (head_dim, dtype, layout) → kv_managed
│   └── Non-matching handlers → local allocation later
└── _identify_managed_state_resources()
    ├── Iterate _resource_lookup, find first SSMResourceHandler → ssm_ref
    ├── Iterate _resource_lookup, find first CausalConvResourceHandler → conv_ref
    ├── Check n_groups constraint: conv_dim = head_dim*num_heads + 2*n_groups*d_state
    └── If constraint fails → conv_ref = None (local allocation)

PHASE 4: CACHE MANAGER CREATION (_create_kv_cache_manager)
├── Has state resources? (ssm_managed or conv_managed)
│   ├── YES → Create MambaHybridCacheManager (manages KV + SSM + Conv)
│   └── NO  → Create KVCacheManager (manages paged KV only)
└── View Assignment:
    ├── _assign_kv_cache_views()         → manager.get_buffers(idx)
    └── _create_and_assign_state_views() → manager.get_ssm_states/get_conv_states

PHASE 5: UNMANAGED RESOURCE ALLOCATION
└── For resources where self._caches[name] is None:
    ├── self._caches[name] = handler.allocate(sequence_info)
    └── Track in _unmanaged_resources list (for proper .to() handling)

PHASE 6: OPTIONAL RESIZE (resize_kv_cache transform)
├── Run forward pass to measure activation memory
├── Shutdown existing KVCacheManager
├── Compute: mem_for_paged = (free_mem - non_paged - forward_mem) * free_gpu_fraction
└── Recreate KVCacheManager with optimal max_tokens
```

## Key Resource Handler Types

| Handler Type | Managed By | Buffer Source | Use Case |
|--------------|------------|---------------|----------|
| `KVPagedResourceHandler` | `KVCacheManager` | `get_buffers(idx)` | Paged KV caches for attention |
| `SSMResourceHandler` | `MambaHybridCacheManager` | `get_ssm_states(layer)` | Mamba SSM state |
| `CausalConvResourceHandler` | `MambaHybridCacheManager` | `get_conv_states(layer)` | Mamba causal conv state |
| `StateResourceHandler` | Local allocation | `handler.allocate()` | Generic per-sequence state |
| `UnpagedResourceHandler` | Local allocation | `handler.allocate()` | Unpaged per-token resources |

## Key Files and Their Responsibilities

### `tensorrt_llm/_torch/auto_deploy/custom_ops/attention_interface.py`

- **`ResourceHandler`** (abstract base): Interface for allocating resources
- **`KVPagedResourceHandler`**: Describes paged KV cache with `num_kv_heads`, `head_dim`, `dtype`, `kv_layout`
- **`SSMResourceHandler`**: Describes Mamba SSM state with `num_heads`, `head_dim`, `d_state`
- **`CausalConvResourceHandler`**: Describes causal conv state with `conv_dim`, `d_conv`
- **`AttentionDescriptor.get_cache_initializers()`**: Returns `ResourceHandlerDict` mapping names to handlers

### `tensorrt_llm/_torch/auto_deploy/transform/library/kvcache.py`

- **`InsertCachedAttention`**: Iterates over attention nodes, calls `get_cache_initializers()`, and registers handlers via `cm.add_resource()`
- **`InitializeCache`**: Triggers `cm.initialize_resources()` to allocate all caches
- **`ResizeKVCache`**: Runs forward pass, measures memory, and calls `cm.resize_kv_cache_manager()`

### `tensorrt_llm/_torch/auto_deploy/shim/interface.py`

- **`CachedSequenceInterface`**: Central class managing all caches
  - `_resource_lookup`: Dict of all registered resource handlers
  - `_unmanaged_resources`: List tracking locally-allocated (non-managed) resource names
  - `add_resource()`: Stores handlers in `_resource_lookup`
  - `initialize_resources()`: Initializes caches, creates cache managers, assigns views
  - `_identify_managed_kv_resources()`: Finds compatible KV handlers from `_resource_lookup`
  - `_identify_managed_state_resources()`: Finds compatible SSM/Conv handlers with constraint checking
  - `_create_kv_cache_manager()`: Creates `KVCacheManager` or `MambaHybridCacheManager`
  - `_allocate_unmanaged_resources()`: Allocates resources not managed by cache managers, tracks in `_unmanaged_resources`

## Example Flow: FlashInfer Attention

```python
# In flashinfer_attention.py
class FlashinferCachedAttention(AttentionDescriptor):
    @classmethod
    def get_cache_initializers(cls, source_attn_node, cache_config):
        k_fake = source_attn_node.args[1].meta["val"]
        return {
            "kv_cache": KVPagedResourceHandler(
                num_kv_heads=k_fake.shape[2],
                head_dim=k_fake.shape[3],
                dtype=cls.resolve_cache_dtype(cache_config.dtype, k_fake.dtype),
                kv_factor=2,
                kv_layout="HND",
            )
        }
```

**This handler gets:**

1. **Collected** by `InsertCachedAttention` → `cm.add_resource("kv_cache_0", handler)`
1. **Stored** in `_resource_lookup` during `initialize_resources()`
1. **Identified** as manageable by `_identify_managed_kv_resources()` if compatible with other KV handlers
1. **View assigned** via `self._caches["kv_cache_0"] = manager.get_buffers(0, kv_layout="HND")`

## Compatibility Rules

### KV Cache Compatibility (for KVCacheManager)

Handlers are compatible if they match on:

- `head_dim`
- `dtype`
- `kv_factor`
- `kv_layout`

Note: `num_kv_heads` can differ (supports GQA/MQA with varying head counts per layer).

### State Resource Compatibility (for MambaHybridCacheManager)

**SSM Resources**: Compatible if `state_shape` and `dtype` match.

**Conv Resources**: Compatible if `state_shape` and `dtype` match, **AND** the n_groups constraint holds:

```text
conv_dim = head_dim * num_heads + 2 * n_groups * d_state
```

If this constraint cannot be satisfied with integer `n_groups >= 0`, Conv resources fall back to local allocation.
