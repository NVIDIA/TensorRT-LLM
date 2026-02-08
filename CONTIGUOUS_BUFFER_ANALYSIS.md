# Contiguous Buffer Approach - Analysis and Why It Fails

## What This Branch Contains

This branch (`eg/trtllm_attn_contiguous_buffer_attempt`) contains an attempt to fix Nemotron with TRT-LLM attention by allocating contiguous buffers for the KV cache.

## The Problem This Tried to Solve

KVCacheManager allocates memory as:
```
[max_num_pages, num_layers, kv_factor, flattened_features]
```

When you call `get_buffers(layer_idx)`, it returns:
```python
result = full_buffer[:, layer_idx, :, :]  # Non-contiguous!
```

This creates a non-contiguous view because there are gaps where other layers' data lives.

## The Attempted Solution (This Branch)

Added code in `trtllm_attention.py` (lines 1043-1071):
```python
if not kv_cache.is_contiguous():
    # Allocate contiguous buffer per layer
    _global_state._contiguous_kv_buffers[layer_idx] = torch.empty(
        kv_cache.shape, dtype=kv_cache.dtype, device=kv_cache.device
    )
    # Copy data
    _global_state._contiguous_kv_buffers[layer_idx].copy_(kv_cache)
    kv_cache = _global_state._contiguous_kv_buffers[layer_idx]
```

## Why This Fails

### Memory Requirements

**Llama-3-8B:**
- 32 attention layers
- max_batch_size=64, max_seq_len=4096
- Buffer per layer: 0.5 GB
- **Total: 16 GB** ✅ Fits in memory

**Nemotron-Nano-30B:**
- 6 attention layers  
- max_batch_size=384, max_seq_len=65536
- Buffer per layer: 12 GB
- **Total: 72 GB** ❌ OOM during CUDA graph capture!

### Why It's Unnecessary

The TRT-LLM kernel (`thop.attention`) **does NOT use the kv_cache tensor directly**!

Instead it uses:
- `host_kv_cache_pool_pointers` - points to BASE address of entire pool
- `host_kv_cache_pool_mapping` - offset for each layer

The kernel computes:
```
layer_cache_ptr = pool_pointers[0] + pool_mapping[layer_idx][offset]
```

This accesses the underlying **contiguous** pool memory directly, so the non-contiguous per-layer view doesn't matter!

## The Correct Solution (HEAD Version)

The HEAD version already works correctly:

1. Sets pool pointers from AD's KVCacheManager:
```python
state.host_kv_cache_pool_pointers[0, 0] = ad_pool_pointers[0, 0].item()
state.host_kv_cache_pool_mapping[layer_idx, 0] = ad_pool_mapping[layer_idx, 0].item()
```

2. TRT-LLM kernel uses these pointers to access contiguous memory

3. No `.contiguous()` calls needed!

## What to Keep from This Branch

1. **llm_args.py** shortcut fix (lines 279-295):
   - Ensures `attn_backend: trtllm` creates `insert_cached_attention` config
   - Sets default `stage: cache_init`

2. Maybe some CUDA graph warmup enhancements in `torch_cudagraph.py`

## What to Discard

1. Contiguous buffer allocation code in `trtllm_attention.py`
2. Any changes that remove pool pointer usage

## How Llama Works Efficiently in HEAD

- Uses pool pointers (no contiguous buffers needed)
- Native HND mode enabled via `TrtllmKVResourceHandler`
- No memory copies during forward pass
- Efficient ~6650 tok/s throughput

## Next Steps

1. Go back to base branch
2. Cherry-pick only the llm_args.py fix
3. Verify Nemotron works with pool pointers (no OOM)
4. Test both Llama and Nemotron for accuracy and performance
