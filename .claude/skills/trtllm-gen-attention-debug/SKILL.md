---
name: trtllm-gen-attention-debug
description: >-
  Debug and develop the trtllm_gen attention backend (FlashInfer + C++ custom ops).
  Covers KV cache layout differences (V1 vs V2), block table construction,
  separate_q_kv_output handling, speculative decoding compatibility, CUDA graph
  safety, and known pitfalls. Use when working on trtllm_gen.py, trtllm.py
  dispatch logic, FlashInfer integration, or debugging attention accuracy issues
  with TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1.
---

# trtllm_gen Attention Backend — Debug & Development Guide

## Architecture Overview

`trtllm_gen` is a drop-in replacement for `thop.attention()` on Blackwell (SM100+).
It reuses the same C++ preprocessing kernels but replaces the FMHA kernel with
FlashInfer's trtllm-gen kernels.

```
thop.attention path:
  build_decoder_info -> qkv_preprocessing -> XQA/MMHA kernel -> kv_cache_postprocessing

trtllm_gen path:
  build_decoder_info -> qkv_preprocessing -> FlashInfer FMHA -> kv_cache_postprocessing
```

Key files:
- `tensorrt_llm/_torch/attention_backend/trtllm_gen.py` — Python backend
- `tensorrt_llm/_torch/attention_backend/trtllm.py` — Dispatch logic
- `cpp/tensorrt_llm/thop/trtllmGenQKVProcessOp.cpp` — C++ custom ops
- `cpp/tensorrt_llm/thop/attentionOp.h` — KV cache pool pointer helpers
- `flashinfer/flashinfer/prefill.py` — FlashInfer prefill API
- `flashinfer/flashinfer/decode.py` — FlashInfer decode API

## Critical Design Constraints

### 1. KV Cache: C++ Writes, FlashInfer Reads

The C++ `qkv_preprocessing` kernel writes K/V to the paged KV cache using
**global block offsets** from `kv_cache_block_offsets`. FlashInfer reads the
KV cache via a **flat-block pool tensor** + `block_tables`.

These two indexing schemes must be compatible. See [architecture.md](references/architecture.md)
for the full memory layout.

### 2. `separate_q_kv_output` Must Match Phase Semantics

| Phase      | thop behavior              | trtllm_gen requirement          |
|------------|----------------------------|---------------------------------|
| Context    | Q read from packed QKV ptr | `False` for BF16; strided view  |
| Generation | Q always separate buffer   | `True` always                   |

When `separate_q_kv_output=False`, Q stays in the packed QKV buffer after RoPE.
FlashInfer needs a separate Q tensor, so extract via zero-copy strided view:
```python
q_size = num_heads * head_size
q_processed = qkv_input[:, :q_size].view(-1, num_heads, head_size)
```

### 3. FlashInfer Uses Separate K/V Page Indices

FlashInfer (with PR #2770 `uses_shared_paged_kv_idx=False`) expects
`block_tables[batch, 2, max_blocks]` where dim-1 index 0 = K offsets,
dim-1 index 1 = V offsets. Always pass `uses_shared_paged_kv_idx=False`.

### 4. No Host-Device Sync in Hot Paths

Any `.cpu()`, `.tolist()`, `.item()`, or `logger.info(tensor)` call in
`run_context` / `run_generation` will break CUDA graph capture and cause
`CUDA_ERROR_ILLEGAL_ADDRESS` at replay time. Gate all diagnostics with
`if not common::isCapturing(stream)` in C++ or move them outside the
capture region in Python.

### 5. `total_num_blocks` for `from_blob` Must Match Pool Size

`torch::from_blob` creates a view — the declared shape must not exceed
allocated memory. Use the actual pool size from KVCacheManager:
```python
# V1 (KVCacheManager): physical pages * layers * kv_factor
total = mgr.blocks_in_primary_pool * mgr.num_local_layers * kv_factor
# V2 (KVCacheManagerV2): use impl API
total = mgr.impl.get_page_index_upper_bound(layer_offset, Role.KEY)
```

## MLA (Multi-head Latent Attention) Support

MLA is used by DeepSeek V2/V3 models. It has fundamentally different data flows
for context vs generation phases, and `trtllm_gen` handles them via separate
code paths and FMHA kernels.

### MLA Dispatch Logic (CRITICAL)

MLA uses **two separate attention wrappers** in the model (`tensorrt_llm/_torch/modules/attention.py`):

| Wrapper | Phase | `head_dim` | `num_kv_heads` | `is_fused_qkv` |
|---------|-------|-----------|----------------|-----------------|
| `self.mha` | Context (non-absorption) | `qk_nope_head_dim + qk_rope_head_dim` (192) | `num_heads` (16/128) | `False` (k provided) |
| `self.mqa` | Generation (absorption) | `kv_lora_rank + qk_rope_head_dim` (576) | 1 | `True` (k is None) |

The dispatch in `trtllm.py` uses `is_fused_qkv = not metadata.is_cross and k is None`:

- **MLA context**: k is provided → `is_fused_qkv=False` → `is_supported()` returns
  `False` ("MLA context with separate Q/K/V not yet supported") → **thop handles context**
- **MLA generation**: k is None → `is_fused_qkv=True` → `is_supported()` may return
  `True` → **trtllm_gen handles generation** via `run_mla_generation()`

**This means**: if you see MLA accuracy issues with `TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1`,
the bug is either in `run_mla_generation()`, or in non-MLA layers that also use trtllm_gen.
MLA context is NOT handled by trtllm_gen unless you modified `is_supported()`.

### MLA Context Path (ragged FMHA — for future enablement)

The `run_context()` MLA branch uses `trtllm_ragged_attention_deepseek` with dense Q/K/V:

1. `mla_rope_context` — C++ kernel: applies RoPE to Q/K in-place, writes compressed
   latent to paged KV cache
2. `trtllm_ragged_attention_deepseek` — FlashInfer: runs FMHA on dense Q/K/V

**Critical parameters**:
- `bmm1_scale`: Must use `1/(q_scaling * sqrt(qk_nope_head_dim + qk_rope_head_dim))`,
  NOT `1/sqrt(head_size)`. For MHA path `head_size = qk_head_dim = 192` so they're equal,
  but for MQA path `head_size = 576` which would be wrong.
- `max_kv_len`: For ragged attention, flashinfer uses this for kernel selection and grid
  config. `host_past_key_value_lengths = 0` for fresh context; may need `input_seq_length`.
- K/V shapes: MLA non-absorption K has `num_heads` heads (not 1), each with dim 192.

FlashInfer detects DeepSeek format via:
```python
is_dsr1 = query.shape[2] == 192 and key.shape[2] == 192 and value.shape[2] == 128
```

### MLA Generation Path (`run_mla_generation`)

Uses `flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla` with paged KV cache.

- Query shape: `[batch, q_len_per_req, num_heads, kv_lora_rank + qk_rope_head_dim]`
- KV cache: `[num_pages, kv_factor, page_size, kv_lora_rank + qk_rope_head_dim]`
  (squeeze dim 2 if ndim=5)
- `bmm1_scale = 1/(q_scaling * sqrt(qk_nope_head_dim + qk_rope_head_dim))`
- `block_tables` must be padded to `128 // tokens_per_block` superblock alignment
- Workspace: `view(-1, 4)` for int32 semaphores

### C++ thop MLA Context Reference (`attentionOp.cpp` ~line 2781)

```cpp
// Non-absorption MLA context FMHA params:
fmhaParams.attentionInputLayout = AttentionInputLayout::SEPARATE_Q_K_V;
fmhaParams.numKvHeads = mNumHeads;     // ALL heads (16/128), NOT num_kv_heads!
fmhaParams.headSize = qk_nope_head_dim + qk_rope_head_dim;  // 192
fmhaParams.headSizeV = qk_nope_head_dim;  // 128 (equals v_head_dim for DeepSeek)
```

### DeepSeek Dimension Reference

| Param | DS V2 Lite | DS V3 |
|-------|-----------|-------|
| num_attention_heads | 16 | 128 |
| num_key_value_heads (MLA layers) | 16 | 128 |
| qk_nope_head_dim | 128 | 128 |
| qk_rope_head_dim | 64 | 64 |
| v_head_dim | 128 | 128 |
| kv_lora_rank | 512 | 512 |
| head_size (MHA context) | 192 | 192 |
| head_size (MQA generation) | 576 | 576 |

## Known Bugs & Pitfalls

See [known-bugs.md](references/known-bugs.md) for the full catalog of bugs
found during the initial integration, including root causes and fixes.

## Debugging Methodology

See [debug-methodology.md](references/debug-methodology.md) for the
systematic approach used to isolate accuracy regressions.

## Quick Checklist for New Changes

1. Run with `TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1`
2. Test both V1 (`KVCacheManager`) and V2 (`KVCacheManagerV2`) KV cache
3. Test with speculative decoding (Eagle3) — multi-token gen dispatch
4. Test with quantized models (FP8/FP4) — `separate_q_kv_output` paths
5. Test with sparse attention — `is_supported()` must reject unsupported configs
6. Verify no host-device syncs in hot paths (CUDA graph safe)
7. Compare `TRTLLM_DUMP_ATTN_PARAMS` output between thop and trtllm_gen paths
8. Test MLA models (DeepSeek V2 Lite / V3) — verify dispatch logging shows
   correct path per layer (thop for context, trtllm_gen for generation)
9. For MLA context enablement: verify `bmm1_scale`, `max_kv_len`, K/V num_heads
