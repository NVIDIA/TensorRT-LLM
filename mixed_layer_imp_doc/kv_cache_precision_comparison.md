# TensorRT-LLM KV Cache: Uniform FP8/BF16 vs Uniform NVFP4 vs Mixed Precision

This document compares the three KV cache precision modes in TensorRT-LLM, covering memory pool
structure, Python-to-C++ construction flow, and the `kv_cache_pool_pointers` tensor format.
All examples use **Qwen3-8B** as a reference: 36 transformer layers, 8 KV heads per layer,
`head_dim=128`, `tokens_per_block=16`.

---

## 1. Memory Pool Structure

### C++ Pool Layout

The fundamental unit of KV cache storage is a **pool** — a contiguous GPU buffer shared across
layers. Pools are allocated inside `WindowBlockManager::allocatePools()` in `kvCacheManager.cpp`.

Within a single attention-window group, layers are **grouped into pools by their KV head count**.
All layers sharing the same number of KV heads share one pool. For Qwen3-8B (uniform 8 KV heads),
this means one pool holds all 36 layers:

```
Pool shape: [num_blocks, num_layers_in_pool, kvFactor, tokens_per_block × head_dim]
          = [num_blocks, 36,                 2,         16 × 128               ]
          = [num_blocks, 36, 2, 2048]
```

- `num_blocks`: total paged blocks allocated (e.g. 2000)
- `num_layers_in_pool`: layers sharing this pool (36 for Qwen3-8B)
- `kvFactor=2`: one slice for K, one for V (1 for K-only cache types)
- `tokens_per_block × head_dim`: the data per head per block

Each **block** is a page in the paged-attention scheme. Block metadata
(`KVCacheBlock`) is separate — it only stores the block ID and index into the pool;
the actual tensor lives in the pool buffer.

`num_pools > 1` arises when:
- Different layers have different KV head counts (variable GQA / MLA hybrids)
- Different layers use different attention window sizes (sliding window attention)

In those cases, `pool_mapping[layer] = [pool_idx, layer_within_pool]` routes each layer
to the correct pool.

---

### Uniform FP8 / BF16

One pool, element dtype = FP8 (1 byte) or BF16 (2 bytes).

```
C++ pool (FP8 example):
  dtype  : kFP8
  shape  : [num_blocks, 36, 2, 2048]
  bytes  : num_blocks × 36 × 2 × 2048 × 1
```

No separate scale pool is needed — FP8 uses a global per-tensor scale (`kv_scale_orig_quant`
passed as a scalar to the attention kernel), not per-block scales.

### Uniform NVFP4

NVFP4 stores data at 4 bits per element (packed as INT8, 2 values per byte) with
**per-block FP8 scale factors** (1 scale per 16 NVFP4 elements). This requires two
parallel pools per logical pool:

```
C++ data pool:
  dtype  : kINT8  (2 NVFP4 elems packed per byte)
  shape  : [num_blocks, 36, 2, 2048 / 2]       ← half the element count
          = [num_blocks, 36, 2, 1024]

C++ scale pool:
  dtype  : kFP8   (1 scale per 16 NVFP4 elems)
  shape  : [num_blocks, 36, 2, 2048 / 16]
          = [num_blocks, 36, 2, 128]
```

The scale pool is created in `WindowBlockManager::createBlockScalePools()` with
`containsBlockScales=true` and `sizePerHead /= 16`.

### Mixed Precision

Each dtype group gets its **own independent set of pools**. For example, FP8 for layers 0–1
and NVFP4 for layers 2–35:

```
Group A — FP8 (layers 0–1):
  data pool shape: [num_blocks_A, 2,  2, 2048]   dtype: kFP8

Group B — NVFP4 (layers 2–35):
  data pool shape: [num_blocks_B, 34, 2, 1024]   dtype: kINT8
  scale pool shape:[num_blocks_B, 34, 2, 128]    dtype: kFP8
```

Memory budgets are split proportionally by `bytes_per_token × num_layers_in_group` across groups
(see `_util.py:948-956`), so each group gets a fair share of GPU memory.

---

## 2. Python-to-C++ Construction Flow

### Uniform Precision (`KVCacheManager`, `resource_manager.py:254`)

A single `KVCacheManager` wraps one C++ `KVCacheManagerImpl`:

```
Python KVCacheManager.__init__()
  │
  ├─ get_pp_layers()              # filter layers for this pipeline-parallel stage
  ├─ compute num_kv_heads_per_layer (with TP sharding)
  ├─ build KvCacheConfigCpp + ModelConfigCpp
  │
  └─ KVCacheManagerCpp(**kwargs)  # C++ constructor (via nanobind)
       │
       ├─ BlockManager(numKvHeadsPerLayer, sizePerHead, tokensPerBlock, ...)
       │    └─ WindowBlockManager per unique attention-window size
       │         └─ KVCacheBlockPool per unique numKvHeads group
       │              └─ allocatePools() → cudaMalloc GPU buffer
       │
       └─ returns impl

  impl.allocate_pools(False)
  self.kv_cache_pool_pointers = impl.get_block_pool_pointers()   # (num_pools, 2)

  # NVFP4 only: attach scale pool pointers on dim=-1
  scale_ptrs = impl.get_block_scale_pool_pointers()
  if scale_ptrs.numel() > 0:
      self.kv_cache_pool_pointers = torch.stack([data_ptrs, scale_ptrs], dim=-1)
      # shape becomes: (num_pools, 2, 2)

  self.kv_cache_pool_mapping = impl.get_layer_to_pool_mapping()  # (num_local_layers, 2)
```

### Mixed Precision (`MixedPrecisionKVCacheManager`, `resource_manager.py:1645`)

A compositor that owns multiple `KVCacheManager` instances, one per dtype group:

```
_util.py: create_mixed_precision_kv_cache_manager()
  │
  ├─ group layers by dtype:  {FP8: [0,1], NVFP4: [2..35]}
  ├─ split GPU budget proportionally by bytes_per_token × num_layers
  │
  └─ for each (dtype, layers) group:
       ├─ build layer_mask = [i in group for i in range(num_hidden_layers)]
       ├─ deep-copy kv_cache_config with proportional max_gpu_total_bytes
       └─ KVCacheManager(dtype=dtype, layer_mask=layer_mask, ...)
            └─ same C++ construction as uniform case above, but for subset of layers

  MixedPrecisionKVCacheManager(sub_managers=[mgr_fp8, mgr_nvfp4], per_layer_dtype_map)
       │
       ├─ _build_merged_pool_pointers()   → (total_pools, 2, 2)
       └─ _build_merged_pool_mapping()    → (num_local_layers, 2)  with offset pool indices
```

---

## 3. `kv_cache_pool_pointers` Tensor

This is a **CPU tensor of raw GPU (and host) memory addresses**. It stays on CPU because the
attention kernel reads it as a host-side lookup table — for each layer it reads the base address
of the pool and then uses `kv_cache_block_offsets` to find the right block at runtime.

### Uniform FP8 / BF16

```
shape: (num_pools, 2)    dtype: int64

[pool_idx, 0]  →  GPU primary pool base address
[pool_idx, 1]  →  host secondary pool address  (0 if no CPU offloading)

Qwen3-8B example (1 pool):
  [[gpu_data_ptr, 0]]
```

### Uniform NVFP4

```
shape: (num_pools, 2, 2)    dtype: int64

[pool_idx, 0, 0]  →  GPU  primary  data  ptr
[pool_idx, 0, 1]  →  GPU  primary  scale ptr
[pool_idx, 1, 0]  →  host secondary data  ptr   (0 if no offloading)
[pool_idx, 1, 1]  →  host secondary scale ptr   (0 if no offloading)

Qwen3-8B example (1 pool):
  [[[data_gpu_ptr, scale_gpu_ptr],
    [0,            0            ]]]
```

Built by stacking `get_block_pool_pointers()` and `get_block_scale_pool_pointers()` on `dim=-1`
(`resource_manager.py:545-548`).

### Mixed Precision

```
shape: (total_pools_across_all_managers, 2, 2)    dtype: int64

Always 3D. Non-FP4 pools are zero-padded in dim 2:
  [pool_idx, j, 0]  →  data  ptr  (primary/secondary)
  [pool_idx, j, 1]  →  scale ptr  (0 for FP8/BF16 pools)

Example: FP8 layers 0–1  +  NVFP4 layers 2–35  (3 pools total):
  pool 0 (FP8 data):    [[fp8_gpu_ptr,   0              ], [0, 0]]
  pool 1 (NVFP4 data):  [[fp4_gpu_ptr,   fp4_scale_ptr  ], [0, 0]]
  pool 2 (NVFP4 scale): (absorbed into pool 1 dim 2 — not a separate entry)
```

`_build_merged_pool_pointers()` (`resource_manager.py:1746`) normalises all sub-manager tensors
to 3D before `torch.cat(parts, dim=0)`. NVFP4 sub-managers pass through as-is; FP8/BF16
sub-managers are zero-padded.

### Comparison Table

| | Uniform FP8/BF16 | Uniform NVFP4 | Mixed |
|---|---|---|---|
| Shape | `(num_pools, 2)` | `(num_pools, 2, 2)` | `(total_pools, 2, 2)` |
| `[i, j, 0]` | data ptr | data ptr | data ptr |
| `[i, j, 1]` | N/A | **scale ptr** | scale ptr (0 for non-FP4) |
| `j=0` | primary (GPU) | primary (GPU) | primary (GPU) |
| `j=1` | secondary (host) | secondary (host) | secondary (host) |
| Who builds it | C++ `get_block_pool_pointers()` | C++ + `torch.stack(..., dim=-1)` | `_build_merged_pool_pointers()` |
| `num_pools=1` when | uniform head count + single window | uniform head count + single window | N/A (sum across groups) |

---

## 4. `kv_cache_pool_mapping`

Companion tensor that tells the attention kernel which pool each layer uses:

```
shape: (num_local_layers, 2)    dtype: int32

[layer_local_idx, 0]  →  pool_idx           (indexes into pool_pointers dim 0)
[layer_local_idx, 1]  →  layer_within_pool  (indexes into pool dim 1)
```

For uniform Qwen3-8B (all 36 layers → pool 0):
```
[[0, 0], [0, 1], [0, 2], ..., [0, 35]]
```

For mixed FP8 (layers 0–1, pool 0) + NVFP4 (layers 2–35, pool 1):
```
[[0, 0], [0, 1], [1, 0], [1, 1], ..., [1, 33]]
  ^FP8^            ^--------- NVFP4 -----------^
```
Pool indices in the mixed case are offset by `_pool_offsets[id(mgr)]` in
`_build_merged_pool_mapping()` (`resource_manager.py:1780`) so they correctly index
into the concatenated `pool_pointers` tensor.
