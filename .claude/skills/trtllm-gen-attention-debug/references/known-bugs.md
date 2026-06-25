# Known Bugs & Fixes — trtllm_gen Attention Backend

## Bug 1: rotary_embedding_inv_freq Zeros in Generation

**Symptom**: Accuracy 0.531 (vs baseline 85.0). Generation-phase Q had wrong
RoPE rotation, producing slightly off attention scores.

**Root Cause**: `build_decoder_info` kernel was skipped during generation
(returned `is_needed=False` for single-token decode), leaving
`gen_ws.rotary_inv_freq` buffer uninitialized (all zeros). The `thop` path
had a fallback to use the pre-computed `params.rotary_inv_freq` from Python,
but `trtllm_gen` did not replicate this.

**Fix**: Check `build_decoder_info` return value and conditionally use the
fallback:
```python
if is_build_decoder_info_kernel_needed:
    rotary_inv_freq_buf = gen_ws.rotary_inv_freq  # filled by kernel
else:
    rotary_inv_freq_buf = params.rotary_inv_freq   # pre-computed fallback
```

**Impact**: Accuracy 0.531 -> 0.758

---

## Bug 2: FlashInfer Workspace Overflow

**Symptom**: Intermittent data corruption, accuracy stuck at 0.758 despite
parameter alignment.

**Root Cause**: `TRTLLM_GEN_WORKSPACE_SIZE` was 32MB, insufficient for
FlashInfer's internal scratch. The workspace overflowed silently into
adjacent workspace sub-buffers (e.g., `q_buf`, `cu_seqlens`).

**Fix**: Increase to 128MB:
```python
TRTLLM_GEN_WORKSPACE_SIZE = 128 * 1024 * 1024  # 128 MB
```

**Lesson**: FlashInfer does not report workspace overflow errors. When
accuracy issues persist after parameter alignment, suspect workspace size.

---

## Bug 3: Block Table Index Mismatch (Core Accuracy Bug)

**Symptom**: Context-phase attention output mostly zeros; generation-phase
output highly inflated. Accuracy 0.910.

**Root Cause**: FlashInfer's `block_tables` came from
`kv_cache_manager.block_ids_per_seq`, which uses per-layer page IDs
(e.g., 255). Meanwhile, the C++ `qkv_preprocessing` writes KV cache using
global block offsets from `kv_cache_block_offsets` (e.g., 18360). These two
coordinate systems are incompatible — FlashInfer reads from wrong memory.

**Fix**: Derive `block_tables` directly from `kv_cache_block_offsets`
(same source as C++ writes), not from `block_ids_per_seq`. With FlashInfer
PR #2770 (`uses_shared_paged_kv_idx=False`), pass the raw 3D offsets
`[batch, 2, max_blocks]` directly:
```python
pool_idx = int(host_kv_cache_pool_mapping[layer_idx, 0])
block_tables = kv_cache_block_offsets[pool_idx, batch_start:batch_start+batch_size]
```

**Impact**: Accuracy 0.910 -> 90.523 (PASSED)

---

## Bug 4: V1 KV Cache CUDA_ERROR_ILLEGAL_ADDRESS

**Symptom**: `CUDA_ERROR_ILLEGAL_ADDRESS` crash with V1 KVCacheManager
after applying the block table fix from Bug 3.

**Root Cause (multi-factor)**:

1. **Workspace regression**: `TRTLLM_GEN_WORKSPACE_SIZE` accidentally
   reverted to 32MB during code iteration.

2. **Hardcoded divisor**: Initial fix used `k_offsets // 2` (correct for V2
   only). V1's `get_buffers()` returns a strided view where
   `stride(0) = num_layers * single_kv_block_elems`, requiring a different
   divisor. This is now moot since we use raw offsets without division.

3. **Host-device sync in CUDA graph**: Diagnostic `logger.info()` calls
   with `.cpu().tolist()` caused synchronization inside the CUDA graph
   capture region.

**Fix**: Restore 128MB workspace, remove all sync-causing diagnostics from
hot paths, use raw offsets + `uses_shared_paged_kv_idx=False` (no division).

---

## Bug 5: separate_q_kv_output Mismatch (BF16 Garbage Output)

**Symptom**: Non-quantized BF16 model (Llama-3.1-8B) produced garbage output
with `trtllm_gen` path. V1 KV cache, no speculative decoding.

**Root Cause**: `trtllm_gen.py` set
`separate_q_kv_output = params.paged_context_fmha or params.cross_attention`,
which was `False` for BF16 models. This meant `qkv_preprocessing` did NOT
copy Q to a separate buffer — Q stayed in the packed QKV buffer.

But `trtllm_gen.py` then read Q from `ctx_ws.q_buf` (which was
uninitialized), feeding garbage to FlashInfer.

The C++ `thop` path works because its FMHA kernel reads Q directly from the
packed QKV buffer via `fmhaParams.qkvPtr`, so it doesn't need a separate Q.

**Fix (generation phase)**: Always set `separate_q_kv_output=True` for
generation, matching thop's implicit behavior:
```python
separate_q_kv_output=True,  # generation always needs separate Q
```

**Fix (context phase)**: Conditionally extract Q:
```python
if separate_q_kv_output:
    q_processed = ctx_ws.q_buf.view(num_tokens, num_heads, head_size)
else:
    q_size = num_heads * head_size
    q_processed = qkv_input[:, :q_size].view(-1, num_heads, head_size)
```

The strided view works because FlashInfer's trtllm-gen kernel correctly
handles non-contiguous Q via `qStrideTokens` / `qStrideHeads`.

---

## Bug 6: Speculative Decoding CUBLAS_STATUS_EXECUTION_FAILED

**Symptom**: `CUBLAS_STATUS_EXECUTION_FAILED` in `o_proj` during CUDA graph
warmup for Eagle3 test with `trtllm_gen` enabled.

**Root Cause**: `trtllm_gen` was incorrectly selected for multi-token
speculative decoding. On Blackwell, `is_spec_decoding_enabled` is forced
`False`, so `is_supported()` returned `True`. But the actual generation had
multiple tokens per request, causing FlashInfer decode to compute wrong
`batch_size`, leading to out-of-bounds `block_tables` reads and downstream
CUBLAS failures.

**Fix**: Add `has_multi_token_gen` check in `trtllm.py` dispatch:
```python
has_multi_token_gen = (spec_decoding_generation_lengths is not None
                       and predicted_tokens_per_seq > 1)
if not has_multi_token_gen and trtllm_gen.is_supported(...)[0]:
    trtllm_gen_attention(...)
```

---

## Bug 7: RocketKV Sparse Attention False Dispatch

**Symptom**: RocketKV sparse attention test failed with 0.0 accuracy.

**Root Cause**: `trtllm_gen.is_supported()` call in `trtllm.py` did not
pass `sparse_kv_indices` and `sparse_attn_indices`. Without these,
`is_supported()` returned `True`. `trtllm_gen` performed full causal
attention instead of the required sparse pattern.

**Fix**: Pass sparse parameters to `is_supported()`:
```python
trtllm_gen.is_supported(
    ...,
    sparse_kv_indices=self.sparse_kv_indices,
    sparse_attn_indices=self.sparse_attn_indices,
)
```

---

## Bug 8: total_num_blocks Overestimation (UB)

**Symptom**: No immediate crash but potential undefined behavior.

**Root Cause**: Formula
`kv_cache_block_offsets.size(1) * kv_cache_block_offsets.size(-1) * kv_factor`
computes `max_batch * max_blocks_per_seq * kv_factor` — unrelated to actual
pool allocation. `torch::from_blob` with oversized shape creates a tensor
view extending beyond allocated GPU memory.

**Fix**: Derive exact pool size from KVCacheManager attributes:
```python
if hasattr(mgr, 'blocks_in_primary_pool'):
    total = mgr.blocks_in_primary_pool * mgr.num_local_layers * kv_factor
else:
    total = mgr.impl.get_page_index_upper_bound(layer_offset, Role.KEY)
```

---

## Investigation: MLA Context Accuracy (DeepSeek V3 Lite)

**Status**: Under investigation. Root cause not yet confirmed.

**Symptom**: DeepSeek V3 Lite GSM8K 5-shot accuracy drops from 64.74 to ~57.8
when `TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1`.

**Key findings so far**:

1. **MLA context falls back to thop** — `is_fused_qkv=False` for MLA context
   (k is provided), so `is_supported()` returns False. MLA context is handled
   by thop, not trtllm_gen.

2. **MLA generation uses trtllm_gen** — `is_fused_qkv=True` for MLA generation
   (k is None), so `run_mla_generation()` is used.

3. **bmm1_scale is correct** — For MHA context, `head_size = qk_head_dim = 192`,
   so `common_params['bmm1_scale'] = 1/sqrt(192)` matches the MLA-specific scale.

4. **max_kv_len=0 for fresh context** — `host_past_key_value_lengths = 0` for
   fresh context requests. This is passed to ragged attention (if MLA context is
   enabled). Flashinfer uses this for grid config, but changing to
   `input_seq_length` did not improve accuracy (possibly within noise).

5. **Non-MLA layers exist** — DeepSeek V2 Lite has standard MHA layers (layer 0)
   that use trtllm_gen for both context and generation. These could be the source.

**Next steps**:
- Add per-layer dispatch logging to confirm exactly which paths each layer takes
- Add per-layer output digests to identify which layers produce divergent output
- Compare non-MLA layer outputs between thop and trtllm_gen paths

---

## Bug 9: torch.ops Dispatch for CPU Tensors

**Symptom**: `Could not run 'trtllm::build_kv_cache_buffers' with arguments
from the 'CPU' backend`.

**Root Cause**: `host_kv_cache_pool_pointers` and `host_kv_cache_pool_mapping`
are CPU tensors. PyTorch op dispatch selects backend based on the first
tensor argument. The op was registered for CUDA only.

**Fix**: Use nanobind function call instead of `torch.ops` dispatch:
```python
from tensorrt_llm.bindings.internal import thop
kv_pool, kv_scale_pool = thop.build_kv_cache_buffers(...)
```
