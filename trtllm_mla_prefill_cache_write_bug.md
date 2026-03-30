# MLA Context (Prefill) Cache Write Bug in AutoDeploy

## Latest Status

All `thop.attention` scalar/metadata params now match PT backend exactly (verified
via side-by-side dump comparison). With `torch-simple` + `tokens_per_block=64`:
- **192/256 requests succeed** with 2-token prefills
- **64/256 fail** with illegal memory access
- Page allocations differ (AD sequential, PT scattered) but both are valid
- **Q values match** between PT and AD (identical min/max range)
- **K/V values differ 50-100x**: PT K=±10, V=±0.3; AD K=±470, V=±800
- `latent_cache` values are nearly identical (±12 both) — compressed_kv matches
- The K/V magnitude issue suggests AD's `kv_b_proj` expansion produces different
  results, possibly from weight transform differences in the `fuse_rope_into_trtllm_mla`
  graph transform or different weight loading for the FP8 model

### Root Cause: Mixed Prefill+Decode Batches
The crash occurs ONLY in mixed batches (prefill + decode sequences together).
Pure prefill batches work correctly — verified by forcing all sequences to
prefill at once via high `max_num_tokens`. The context wrapper receives
metadata tensors that include decode sequences' entries, confusing the C++
kernel's cache write path.

Fix: slice metadata to only include prefill sequences before passing to
the context wrapper's plan/run.

### Investigation: K/V Magnitude Difference
The K/V expansion is `K,V = kv_b_proj(compressed_kv)` where `kv_b_proj_weight` has
shape `[num_heads*(qk_nope_head_dim+v_head_dim), kv_lora_rank]` = `[8192, 512]`.
The `compressed_kv` (and `latent_cache`) values match between PT and AD. So the
weight itself must differ. The `fuse_rope_into_trtllm_mla` transform modifies the
model graph and may affect how `kv_b_proj_weight` is stored or applied.

With `torch-cudagraph`, the failures cascade through CUDA graph corruption.

## Reproduction Commit

- **Branch**: `eg/ds_with_mla_enablement`
- **Repro commit SHA**: `8e949a9b4d` (`trtllm_mla: BF16 KV cache, warmup skip, Flash MLA + prefill cache write bug`)
- This commit contains the crashing `_call_thop_attention_mla` call with `update_kv_cache=True`
  in `_handle_prefill_thop`. Run the benchmark command below to reproduce the illegal memory access.

## Summary

`thop.attention` with `is_mla_enable=True` and `attention_input_type=context_only` crashes with
**illegal memory access** when writing latent_cache to the paged KV cache in the AutoDeploy (AD)
pipeline. The same kernel with the same MLA parameters works correctly in the PyTorch (PT) backend.

The crash is specifically in the **cache write** path — the FMHA attention computation itself
works fine (confirmed by passing `update_kv_cache=False`, which completes without error).

**Current workaround**: Use batched `torch.nn.functional.scaled_dot_product_attention` (SDPA)
for prefill attention + manual `index_copy_` for cache writes. This works end-to-end but is
~19% slower than the PT backend (29.7 vs 36.6 tps/user at ISL/OSL=128/128).

## Reproduction

```bash
# Config: ds_ad.yaml
# runtime: trtllm
# attn_backend: trtllm
# compile_backend: torch-cudagraph
# model_factory: AutoModelForCausalLM
# skip_loading_weights: true
# max_seq_len: 512
# transforms:
#   insert_cached_mla_attention:
#     backend: trtllm_mla
#   fuse_rope_into_trtllm_mla:
#     stage: cache_init
#     enabled: true

trtllm-bench --model deepseek-ai/DeepSeek-V3-Lite \
    --model_path ${LLM_MODELS_ROOT}/DeepSeek-V3-Lite/fp8 \
    throughput \
    --dataset DeepSeek-V3-Lite_128_128_64.inp \
    --backend _autodeploy --max_batch_size 64 --max_num_tokens 4096 \
    --extra_llm_api_options ds_ad.yaml
```

**Hardware**: NVIDIA H100 NVL (SM90), single GPU.

## Root Cause Analysis

### What works

| Path | update_kv_cache | Result |
|------|----------------|--------|
| Context FMHA read (attention computation) | False | Works — no crash |
| Context FMHA write (cache populate) | True | **Crashes** — illegal memory access |
| Decode FMHA (via TrtllmAttentionWrapper) | True | Works — Flash MLA path |
| Decode cache write (via mla_rope_generation) | True | Works |
| Manual cache write (index_copy_) | N/A | Works |

### The bug: context cache write addresses

When `thop.attention` runs with `attention_input_type=context_only` and `update_kv_cache=True`,
the C++ kernel writes `latent_cache` (compressed KV, 576 values per token) to the paged KV cache.
The write address is computed from:

- `kv_cache_block_offsets[pool, seq, K_or_V, block_idx]` — page-level offset
- `host_kv_cache_pool_mapping[layer_idx]` — layer offset within page
- `host_kv_cache_pool_pointers[pool]` — base pointer of the pool

The address computation: `pool_ptr + (block_offset + layer_offset) * tokens_per_block * head_dim`

**AD computes block_offsets** via `ragged_to_block_table_triton`:
```
block_offsets[seq, K, block] = page_id * block_offset_multiplier
block_offsets[seq, V, block] = page_id * block_offset_multiplier  (kv_factor=1)
```
Where `block_offset_multiplier = 30` (= num_layers × kv_factor = 30 × 1).

**Verified**: AD's block_offsets produce correct addresses for the pool layout:
```
kv_cache shape = [48646, 1, 1, 32, 576]
kv_cache stride(0) = 552960 = 30 * 18432 = 30 * tpb * head_dim
pool_ptr = kv_cache.data_ptr()  (same as pool start)
block_offsets[seq0] = [0, 30, 60, 90]  (pages 0,1,2,3 × multiplier 30)
```

Address for page P, layer L: `(P*30 + L) * 32 * 576` — mathematically correct.

### Why it still crashes

Despite the addresses being mathematically correct, the C++ context MLA write kernel
produces an illegal memory access. The exact cause is inside the C++ kernel
(`tensorrt_llm/thop/attentionOp.cpp`) and is not directly observable from Python.

Possible explanations:

1. **Head count mismatch in write stride**: The context MLA kernel may compute per-token
   write size as `num_kv_heads * some_dim` instead of using the MLA latent dimension (576).
   With `num_kv_heads=1` and `head_size=192`, it might write `1 * 192 = 192` values but
   stride by `576` (latent dim), or vice versa. Any mismatch causes out-of-bounds writes
   across page boundaries.

2. **Block offset interpretation differs between context and decode paths**: The decode
   path uses `mla_rope_generation` (a separate kernel) for cache writes, which correctly
   interprets the block offsets for the MLA layout. The context path uses a different
   code path in `AttentionOp` that may apply additional transformations to the block
   offsets (e.g., scaling by `num_kv_heads`) that are incompatible with `kv_factor=1`.

3. **Interleaved layout assumption**: The context write path may assume `kv_factor=2`
   (separate K/V slots) even when `kv_factor=1` (shared MLA latent). This would cause
   it to write V data at `block_offset + 1` (next layer's slot), corrupting adjacent
   layers' data and potentially overflowing the pool at the last layer.

## Why the PT Backend Works

The PT backend uses `KVCacheManager.copy_batch_block_offsets()` to fill the
`kv_cache_block_offsets` tensor. This C++ method:

```python
# PT backend (trtllm.py:1213)
kv_cache_manager.copy_batch_block_offsets(
    self.kv_cache_block_offsets,  # dst GPU tensor
    self.request_ids,              # which requests
    self.beam_width,
    self.num_contexts,
    self.num_seqs,
)
```

Key differences from AD:

| Aspect | PT Backend | AD Pipeline |
|--------|-----------|-------------|
| Block offset source | `KVCacheManager.copy_batch_block_offsets()` (C++) | `ragged_to_block_table_triton` + `page_id * multiplier` (Triton) |
| Request tracking | `request_ids` from scheduler | `cache_loc` pages from SequenceInfo |
| Pool pointer | From `kv_cache_manager.host_kv_cache_pool_pointers` | From `kv_cache.data_ptr()` |
| Pool mapping | From `kv_cache_manager.host_kv_cache_pool_mapping` | Manually constructed `[layer_idx, 0→pool, 1→layer]` |

The critical difference: `copy_batch_block_offsets` is a C++ method that fills block offsets
using the KVCacheManager's **internal block table** — the same data structure that the C++
attention kernels use to compute addresses. This guarantees consistency between the offset
encoding and the kernel's address computation.

AD computes offsets independently from `cache_loc` pages using a Triton kernel. While the
**values** appear mathematically equivalent (`page_id * 30` in both cases), the C++ context
MLA write kernel may expect a subtly different encoding that `copy_batch_block_offsets`
provides (e.g., per-pool scaling, beam-width adjustments, or V-offset conventions).

### Why AD's decode works but context doesn't

- **Decode cache write**: Uses `mla_rope_generation` kernel, which is a standalone CUDA
  kernel that directly computes write addresses from `block_ids_per_seq` (page IDs) and
  cache metadata. This kernel was written specifically for MLA and handles `kv_factor=1`
  correctly.

- **Context cache write**: Uses the generic `AttentionOp::contextMlaWriteCache()` path
  inside `thop.attention`, which was designed for the PT backend's `copy_batch_block_offsets`
  encoding. This path may apply additional address transformations that are correct for
  `copy_batch_block_offsets` output but incorrect for AD's `page_id * multiplier` encoding.

## Things Tried

### 1. Context wrapper with num_kv_heads=32 (PT backend's context params)
**Approach**: Use `TrtllmAttentionWrapper` with `head_size=192, num_kv_heads=32` (matching
PT's `self.mha`).
**Result**: Same illegal memory access. The kernel writes 32 KV heads' worth of data into
a cache slot sized for 1 head (kv_factor=1), causing massive overflow.

### 2. Direct thop.attention with num_kv_heads=1
**Approach**: Call `_call_thop_attention_mla` with `num_kv_heads=1, head_size=192`.
**Result**: Illegal memory access in cache write. The FMHA read works (`update_kv_cache=False`)
but the write crashes.

### 3. update_kv_cache=False + manual cache write
**Approach**: Disable C++ cache write, do manual `index_copy_` instead.
**Result**: C++ asserts `"KV cache update cannot be disabled now"` during real inference.
The assertion is global (not per-AttentionOp) and cannot be bypassed by using separate
`layer_idx` values.

### 4. Scratch pool pointer redirect
**Approach**: Pass a scratch buffer's pointer as `host_kv_cache_pool_pointers` to absorb
the C++ kernel's writes, then do manual writes to the real cache.
**Result**: Scratch buffer too small — the C++ kernel's address computation spans the
entire pool (30× the per-layer kv_cache tensor size due to interleaving). Allocating
a full-sized scratch (50+ GB) is impractical.

### 5. FP8 KV cache (originally auto-detected)
**Approach**: AD auto-enabled FP8 KV cache for FP8 weight models.
**Result**: The standard FMHA has no kernel for FP8 + MLA on SM90. PT backend uses
BF16 KV cache (confirmed: `quant_config.quant_mode.has_fp8_kv_cache() == False`).
Fixed by using BF16 KV cache to match PT.

### 6. Flash MLA metadata for decode
**Approach**: Allocate `flash_mla_tile_scheduler_metadata` and `flash_mla_num_splits`,
call `thop.compute_flash_mla_metadata` before decode wrapper's plan/run.
**Result**: Required for FP8 KV cache on SM90. Not needed after switching to BF16 cache,
but kept for future FP8 support.

### 7. Batched SDPA + manual cache write (current workaround)
**Approach**: Use `torch.nn.functional.scaled_dot_product_attention` for prefill attention
(Flash Attention backend) + `index_copy_` for cache writes.
**Result**: **Works end-to-end**. 29.7 tps/user (vs 36.6 PT backend). ~19% slower due to
SDPA overhead and per-sequence cache write loop.

## Suggested Fix (C++ side)

The most robust fix would be one of:

1. **Expose `update_kv_cache=False` for context**: Remove the global assertion that
   prevents disabling cache updates. This would allow AD to use C++ FMHA for attention
   and manual cache writes. The assertion exists for safety but is overly restrictive.

2. **Fix context MLA cache write for AD's block offset encoding**: The `contextMlaWriteCache`
   code path in `AttentionOp` should correctly handle the `page_id * num_layers` block
   offset encoding used by AD's `ragged_to_block_table_triton`. This may require
   adding a flag or adjusting the address computation to match `mla_rope_generation`'s
   approach.

3. **Use `mla_rope_generation` for context cache writes**: Since `mla_rope_generation`
   correctly writes to the paged cache with AD's block offsets, a similar kernel could
   be used for the context phase's cache population. This would bypass the `AttentionOp`
   cache write entirely.

4. **Expose `copy_batch_block_offsets` to AD**: Add `request_ids` to the AD SequenceInfo
   pipeline and call `KVCacheManager.copy_batch_block_offsets` in the host-side prepare
   function. This would produce identical block offsets to the PT backend.
