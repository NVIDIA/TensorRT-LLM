# Debugging Methodology — trtllm_gen Attention Backend

## Phase 1: Reproduce & Bisect

### Test Commands

```bash
# V2 KV cache accuracy test (W4 quantized)
TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1 \
  pytest tests/integration/defs/accuracy/test_llm_api_pytorch.py::TestGPTOSS::test_w4_1gpu[v2_kv_cache-True-True-trtllm-auto] -s -v

# V1 KV cache speculative decoding (Eagle3)
TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1 \
  pytest tests/integration/defs/llmapi/test_llm_examples.py::test_llmapi_speculative_decoding_eagle3 -s -v

# Non-spec-decode V1 baseline (Llama-3.1-8B BF16)
TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1 \
  pytest tests/unittest/_torch/speculative/test_eagle3.py::test_llama_eagle3[True-TRTLLM-True-False-False-False-True-False-False-False] -s -v
```

### A/B Comparison

Run the same test with `TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=0` (thop path)
and `=1` (trtllm_gen path). Redirect to separate log files:
```bash
TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=0 TRTLLM_DUMP_ATTN_PARAMS=stdout \
  python -m pytest <test> -s -v > log_thop.txt 2>&1

TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1 TRTLLM_DUMP_ATTN_PARAMS=stdout \
  python -m pytest <test> -s -v > log_tllm.txt 2>&1
```

## Phase 2: Parameter Alignment

### TRTLLM_DUMP_ATTN_PARAMS

Set `TRTLLM_DUMP_ATTN_PARAMS=stdout` to dump all C++ kernel parameters.
The C++ `dumpAttnParams()` function emits tagged entries like:
```
[ATTN_PARAMS:qkv_preprocessing:call_42] batch_size=4 head_num=32 ...
```

Compare parameters between thop and trtllm_gen paths. Key parameters to
check first:
- `batch_size` — should match
- `max_input_seq_len`, `max_kv_seq_len` — phase-dependent
- `separate_q_kv_output` — True in gen, conditional in context
- `generation_phase` — True for generation, False for context
- `layer_idx` / `global_layer_idx` — layer mapping
- `kv_cache_quant_mode` — quantization mode

### Call-ID Based Comparison

Each kernel invocation gets a unique monotonic ID via
`static std::atomic<uint64_t>` in the kernel wrapper. This allows
chronological 1:1 comparison between thop and trtllm_gen paths even
across layers and phases.

Strategy: dump first N calls from both paths, skip CUDA graph warmup
outputs (look for lines after "Fetching responses:" in the log).

## Phase 3: Tensor Diagnostics

### Attention Output Digest

Add digest logging in `trtllm.py` after the attention call:
```python
if os.environ.get("TRTLLM_DEBUG_ATTN_OUTPUT"):
    t = output.float()
    logger.warning(
        f"[ATTN_OUT] layer={layer_idx} "
        f"min={t.min().item():.6f} max={t.max().item():.6f} "
        f"mean={t.mean().item():.6f} norm={t.norm().item():.6f} "
        f"first8={t.flatten()[:8].tolist()}"
    )
```

Red flags:
- All zeros in context phase -> FlashInfer reading wrong KV memory
- Highly inflated values in generation -> block table pointing to stale data
- NaN/Inf -> workspace overflow or uninitialized buffer

### Per-Layer Tensor Dump

For deep investigation, dump full tensors per layer to files:
```python
import os
dump_dir = os.environ.get("TRTLLM_DEBUG_DUMP_DIR")
if dump_dir:
    torch.save(q_processed, f"{dump_dir}/layer{layer_idx}_q.pt")
    torch.save(block_tables, f"{dump_dir}/layer{layer_idx}_bt.pt")
```

## Phase 4: KV Cache Verification

### Block Table Comparison

Compare block tables between FlashInfer path and thop's internal tables:
```python
# FlashInfer's block_tables (from kv_cache_block_offsets)
bt_fi = kv_cache_block_offsets[pool_idx, batch_start:batch_start+bs]

# thop's block_ids_per_seq (per-layer page IDs, different coordinate)
bt_thop = kv_cache_manager.block_ids_per_seq[layer_idx]
```

These use DIFFERENT coordinate systems and should NOT be directly compared.
The correct validation is to verify that `block_tables` entries index valid
locations in the flat-block `kv_pool` tensor.

### KV Cache Content Verification

Read back KV cache values at specific block offsets to verify writes:
```python
# Check if qkv_preprocessing wrote to the correct location
k_offset = int(kv_cache_block_offsets[pool_idx, 0, 0, 0])  # first K block
kv_block = kv_pool[k_offset]  # [num_kv_heads, tokens_per_block, head_dim]
# Should contain non-zero values after qkv_preprocessing
```

## Phase 5: CUDA Graph Safety

### Detection

Add `CUDA_LAUNCH_BLOCKING=1` to disable async launches. If a crash moves
to a different location, the original crash site was in a CUDA graph.

### Common Sync Violations

| Pattern | Where | Fix |
|---------|-------|-----|
| `tensor.cpu()` | Python hot path | Remove or gate with env var |
| `tensor.tolist()` | Logger calls | Remove from hot path |
| `tensor.item()` | Scalar extraction | Use `tensor[0]` if on GPU |
| `logger.info(f"...{tensor}")` | Debug logging | Gate with `if not capturing` |
| `fprintf(stderr, ...)` in C++ | dumpAttnParams | Gate with `!isCapturing(stream)` |

### Verification

Run with CUDA graph enabled (default) and verify no
`CUDA_ERROR_ILLEGAL_ADDRESS` or `cudaErrorStreamCaptureUnsupported`.

## Phase 6: MLA-Specific Debugging

### Step 1: Verify Dispatch Path Per Layer

MLA accuracy issues are often caused by misunderstanding which code path each layer
actually takes. Add temporary logging in `TrtllmAttentionWrapper.run()`:

```python
if self.is_mla_enable:
    from tensorrt_llm.logger import logger
    logger.warning(
        f"[MLA-DEBUG] layer={self.layer_idx} is_mla={self.is_mla_enable} "
        f"is_fused_qkv={is_fused_qkv} k_is_none={k is None} "
        f"attn_input_type={self.attention_input_type} "
        f"trtllm_gen_supported={_supported} reason={_reason} "
        f"num_heads={self.num_heads} num_kv_heads={self.num_kv_heads} "
        f"head_size={self.head_size}"
    )
```

**Expected for DeepSeek MLA layers:**
- Context: `is_fused_qkv=False`, `trtllm_gen_supported=False` → thop
- Generation: `is_fused_qkv=True`, `trtllm_gen_supported=True` → trtllm_gen

### Step 2: Check MLA Dimension Consistency

Verify dimensions match between model config and attention parameters:

```python
# In run_context or run_mla_generation, add:
logger.warning(
    f"[MLA-DIM] Q={q.shape} K={k.shape if k is not None else None} "
    f"V={v.shape if v is not None else None} "
    f"head_size={params.head_size} qk_nope={params.qk_nope_head_dim} "
    f"qk_rope={params.qk_rope_head_dim} v_dim={params.v_head_dim} "
    f"kv_lora_rank={params.kv_lora_rank} "
    f"num_heads={params.num_heads} num_kv_heads={params.num_kv_heads}"
)
```

**Key checks:**
- `head_size` should be 192 for MHA context, 576 for MQA generation
- `num_kv_heads` should equal `num_heads` for MLA context, 1 for MLA generation
- K.shape should be `[N, num_heads * 192]` for context (not `[N, 1 * 192]`)

### Step 3: Compare bmm1_scale

For MLA, the softmax scale depends on which attention wrapper is active:

| Path | head_size | bmm1_scale |
|------|-----------|-----------|
| MHA context | 192 | 1/sqrt(192) ≈ 0.0722 |
| MQA generation | 576 | 1/sqrt(576) ≈ 0.0417 |

But `common_params['bmm1_scale']` uses `head_size` from the wrapper.
`run_mla_generation` correctly overrides with its own scale; verify `run_context`
MLA path does the same if enabled.

### Step 4: Verify KV Cache Compatibility

MLA context (thop) writes compressed latent to KV cache.
MLA generation (trtllm_gen) reads from the same cache.

Check that `_build_block_tables()` produces correct page indices for MLA:
- MLA KV cache: 1 KV head, `kv_lora_rank + qk_rope_head_dim` head_dim
- `get_buffers()` returns `[pages, kv_factor, 1, page_size, head_dim]`
- After squeeze(2): `[pages, kv_factor, page_size, head_dim]`
- Block table divisor must match this layout

### Step 5: Check Non-MLA Layers

DeepSeek V2 Lite has mixed architecture (layer 0 may be standard MHA).
Non-MLA layers use trtllm_gen for BOTH context and generation when enabled.
Verify these layers produce correct output:

```bash
# Add per-layer output digest to identify which layers diverge
TRTLLM_DEBUG_ATTN_OUTPUT=1 TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1 \
  pytest <test> -s -v 2>&1 | grep ATTN_OUT
```

### MLA Debugging Decision Tree

```
MLA accuracy drop with TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1?
  |
  +-- Step 1: Add dispatch logging
  |     |
  |     +-- MLA context uses thop? (expected: yes)
  |     |     +-- Yes -> MLA context is NOT the issue
  |     |     +-- No -> Check is_supported() modifications
  |     |
  |     +-- MLA generation uses trtllm_gen? (expected: yes)
  |           +-- Yes -> Check run_mla_generation params
  |           +-- No -> Check is_fused_qkv logic
  |
  +-- Step 2: Is the drop from MLA gen or non-MLA layers?
  |     |
  |     +-- Disable trtllm_gen for gen only -> still broken?
  |     |     +-- Yes -> Non-MLA layers are the issue
  |     |     +-- No -> MLA generation is the issue
  |     |
  |     +-- Check per-layer output digests for divergence
  |
  +-- Step 3: If MLA generation is broken:
  |     +-- Check bmm1_scale (576 vs 192 mixup?)
  |     +-- Check block_tables (superblock padding?)
  |     +-- Check KV cache squeeze (ndim=5 → ndim=4?)
  |     +-- Check Q reshape (4D: [batch, q_len, heads, dim])
  |
  +-- Step 4: If non-MLA layers are broken:
        +-- Follow standard (non-MLA) debugging flow
        +-- Check separate_q_kv_output
        +-- Check workspace size
```

## Useful Environment Variables

| Variable | Values | Purpose |
|----------|--------|---------|
| `TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION` | 0/1 | Enable trtllm_gen backend |
| `TRTLLM_DUMP_ATTN_PARAMS` | stdout | Dump C++ kernel parameters |
| `CUDA_LAUNCH_BLOCKING` | 0/1 | Serialize CUDA launches |
| `TLLM_WORKER_USE_SINGLE_PROCESS` | 0/1 | Single-process mode for debugging |
| `TRTLLM_DEBUG_ATTN_OUTPUT` | 0/1 | Dump attention output digests |

## Debugging Decision Tree

```
Accuracy wrong?
  |
  +-- Check TRTLLM_DUMP_ATTN_PARAMS: parameters match thop?
  |     |
  |     +-- No -> Fix parameter passing in trtllm_gen.py
  |     +-- Yes -> Continue
  |
  +-- Check attention output digest per layer:
  |     |
  |     +-- All zeros (context) -> KV cache read issue
  |     |     |
  |     |     +-- Check block_tables vs kv_cache_block_offsets
  |     |     +-- Check kv_pool construction (total_num_blocks, from_blob)
  |     |     +-- Check uses_shared_paged_kv_idx=False
  |     |
  |     +-- Garbage / huge values -> Q input issue
  |     |     |
  |     |     +-- Check separate_q_kv_output setting
  |     |     +-- Check if q_buf is initialized
  |     |     +-- Check FP4/FP8 reinterpretation
  |     |
  |     +-- Slightly off -> RoPE or scale issue
  |           |
  |           +-- Check rotary_embedding_inv_freq (zeros?)
  |           +-- Check bmm1_scale / bmm2_scale
  |           +-- Check workspace size (overflow?)
  |
  +-- CUDA crash?
        |
        +-- ILLEGAL_ADDRESS -> memory bounds issue
        |     |
        |     +-- Check total_num_blocks vs actual pool size
        |     +-- Check host-device sync in CUDA graph
        |     +-- Check workspace size
        |
        +-- CUBLAS_STATUS_EXECUTION_FAILED -> shape mismatch
              |
              +-- Check if multi-token gen was incorrectly dispatched
              +-- Check batch_size / q_len_per_req
```
