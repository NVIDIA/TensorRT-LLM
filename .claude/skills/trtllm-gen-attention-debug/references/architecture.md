# trtllm_gen Attention Backend — Architecture

## Data Flow

```
                       +-------------------+
                       |  qkv_input (QKV)  |
                       +--------+----------+
                                |
                    +-----------v-----------+
                    | qkv_preprocessing     |  C++ kernel (shared with thop)
                    | - RoPE on Q and K     |
                    | - Write K/V to cache  |  <-- uses kv_cache_block_offsets
                    | - Optionally copy Q   |      (global block indices)
                    +-----------+-----------+
                                |
              +-----------------+------------------+
              |                                    |
     separate_q_kv_output=True          separate_q_kv_output=False
     Q in ctx_ws.q_buf                  Q stays in packed QKV buffer
              |                                    |
              +-------+   +------------------------+
                      |   |
                      v   v
              +-------+---+--------+
              | FlashInfer FMHA    |  reads KV via flat-block pool tensor
              | - prefill (ctx)    |  + block_tables[batch, 2, blocks]
              | - decode  (gen)    |
              +--------+-----------+
                       |
           +-----------v-----------+
           | kv_cache_postprocessing|  C++ kernel
           +-----------------------+
```

## KV Cache Memory Layout

### Raw Pool (GPU Memory)

The KV cache pool is a contiguous GPU allocation. Blocks are interleaved:

```
Pool memory (flat byte buffer):
+----------+----------+----------+----------+----------+----------+---
| Page0    | Page0    | Page0    | Page0    | Page1    | Page1    | ...
| Layer0_K | Layer0_V | Layer1_K | Layer1_V | Layer0_K | Layer0_V |
+----------+----------+----------+----------+----------+----------+---
  block 0    block 1    block 2    block 3    block 4    block 5
```

Each "block" = `num_kv_heads * tokens_per_block * head_dim` elements.

### kv_cache_block_offsets

Shape: `[num_pools, batch, 2, max_blocks_per_seq]`

- dim-2 index 0 = K block offsets (global indices into the flat pool)
- dim-2 index 1 = V block offsets

Example for batch=0, seq with 2 pages:
```
K offsets: [0, 4, ...]   (page0_layer0_K=0, page1_layer0_K=4)
V offsets: [1, 5, ...]   (page0_layer0_V=1, page1_layer0_V=5)
```

### build_kv_cache_buffers C++ Op

Constructs a flat-block view via `torch::from_blob`:
```
kv_pool shape: [total_num_blocks, num_kv_heads, tokens_per_block, head_dim]
```

The pool pointer is offset to the correct layer within the pool using
`buildKvCachePoolPointers()`, which computes:
```
intra_pool_offset = layer_idx_in_cache_pool * kv_factor * block_size_bytes
```

FlashInfer receives `kv_cache = (kv_pool, kv_pool)` — K and V share the
same pool tensor but are indexed by separate block offsets in `block_tables`.

### block_tables Construction

`block_tables` is a Python slice directly from `kv_cache_block_offsets`:
```python
pool_idx = int(host_kv_cache_pool_mapping[layer_idx, 0])
block_tables = kv_cache_block_offsets[pool_idx, batch_start:batch_start+batch_size]
# shape: [batch_size, 2, max_blocks_per_seq]
```

No division or conversion needed — raw offsets index directly into the
flat-block pool tensor.

### V1 vs V2 KVCacheManager Differences

| Property | V1 (KVCacheManager) | V2 (KVCacheManagerV2) |
|----------|--------------------|-----------------------|
| Pool layout | Pages interleaved across layers | Same |
| `get_buffers()` | Strided view (stride(0) = num_layers * block_size) | Contiguous tensor |
| `blocks_in_primary_pool` | Available | Available |
| `num_local_layers` | Available | Available |
| `layer_offsets` | Available | Available |
| `total_num_blocks` formula | `blocks_in_primary_pool * num_local_layers * kv_factor` | Same, or `impl.get_page_index_upper_bound()` |

## FlashInfer Integration Details

### uses_shared_paged_kv_idx (PR #2770)

Default FlashInfer assumes K and V share page indices (2D block_tables).
TRT-LLM uses separate K/V indices (3D block_tables `[batch, 2, blocks]`).

Set `uses_shared_paged_kv_idx=False` on all FlashInfer calls:
- `flashinfer.prefill.trtllm_batch_context_with_kv_cache()`
- `flashinfer.decode.trtllm_batch_decode_with_kv_cache()`

### KV Layout

Default: `"HND"` = `[max_pages, kv_factor, num_kv_heads, page_size, head_dim]`

### Workspace

FlashInfer needs a workspace buffer for internal scratch. Minimum 128MB
(`TRTLLM_GEN_WORKSPACE_SIZE`). Smaller values cause silent data corruption
from buffer overflow.

## Speculative Decoding Support

### Dispatch in trtllm.py

`trtllm_gen` does NOT support multi-token speculative decoding where
`predicted_tokens_per_seq > 1`. The dispatch logic in `trtllm.py` must
bypass `trtllm_gen` in this case.

However, on Blackwell `is_spec_decoding_enabled` may be forced `False`
even when Eagle3 is active. The actual check is:
```python
has_multi_token_gen = (spec_decoding_generation_lengths is not None
                       and predicted_tokens_per_seq > 1)
```

### FlashInfer Decode with Multiple Query Tokens

For Eagle3 on Blackwell (`predicted_tokens_per_seq == 1` but
`input_seq_length > 1`), FlashInfer decode needs `q_len_per_req`
to correctly compute batch_size from the flattened query tensor:
```python
if is_multi_token_gen:
    # Variable lengths: pass cum_seq_lens_q, set q_len_per_req=None
    flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        ..., q_len_per_req=None, cum_seq_lens_q=cu_seqlens)
else:
    # Uniform lengths: pass q_len_per_req=input_seq_length
    flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        ..., q_len_per_req=input_seq_length)
```

## MLA (Multi-head Latent Attention) Data Flow

MLA uses a fundamentally different data flow from standard GQA/MHA attention.
The model's `kv_a_proj` compresses hidden states to a latent representation,
and `kv_b_proj` decompresses to per-head K/V.

### MLA Context Phase (Non-Absorption Mode)

```
Model Layer:
  hidden_states -> kv_a_proj -> compressed_kv [N, kv_lora_rank + qk_rope_head_dim]
                                    |
                          +---------+---------+
                          |                   |
                   latent_cache           k_pe
                   [N, kv_lora_rank]   [N, qk_rope_head_dim]
                          |
                   kv_b_proj ->  k_nope [N, num_heads, qk_nope_head_dim]
                                 v      [N, num_heads, v_head_dim]

  q_a_proj -> q_b_proj -> q [N, num_heads, qk_nope_head_dim + qk_rope_head_dim]

  K = concat(k_nope, k_pe.expand(num_heads)) -> [N, num_heads, 192]
  V = v -> [N, num_heads, 128]

Attention Backend (trtllm_gen MLA context path):
  mla_rope_context(q_buf, k_buf, v_buf, latent_cache, cos_sin_cache, ...)
    - Applies RoPE to Q's rope portion in-place
    - Applies RoPE to K's rope portion in-place
    - Writes compressed latent + RoPE'd rope to paged KV cache
    - (Does NOT decompress K/V — model already did that via kv_b_proj)

  trtllm_ragged_attention_deepseek(Q, K, V, ...)
    - Dense Q/K/V (NOT from KV cache)
    - Q: [N, num_heads, 192], K: [N, num_heads, 192], V: [N, num_heads, 128]
    - Output: [N, num_heads, v_head_dim=128]
```

**Key insight**: For MLA non-absorption context, `num_kv_heads = num_heads` (all heads
have unique K/V from `kv_b_proj`). This differs from standard GQA where `num_kv_heads < num_heads`.

### MLA Generation Phase (Absorption Mode)

```
Model Layer:
  hidden_states -> q_a_proj -> q_b_proj -> q_nope [N, num_heads, qk_nope_head_dim]
                                           q_pe   [N, num_heads, qk_rope_head_dim]

  Absorption: q_absorbed = q_nope @ kv_b_proj_trans -> [N, num_heads, kv_lora_rank]
  Q = concat(q_absorbed, q_pe) -> [N, num_heads, kv_lora_rank + qk_rope_head_dim]

Attention Backend (run_mla_generation):
  qkv_preprocessing(Q_fused, ...)
    - Applies RoPE to Q's rope portion
    - Writes Q to gen_ws.q_buf
    - Writes KV to paged cache (NOT needed for gen — already in cache from context)

  flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(...)
    - Reads Q: [batch, q_len, num_heads, kv_lora_rank + qk_rope_head_dim]
    - Reads KV from paged cache: [pages, kv_factor, page_size, kv_lora_rank + qk_rope_head_dim]
    - KV cache has 1 effective head (compressed), num_kv_heads dim squeezed
    - Output: [batch, q_len, num_heads, kv_lora_rank]
```

### MLA KV Cache Layout

MLA stores the **compressed** representation (not decompressed K/V):

```
Per token in KV cache:
  [kv_lora_rank bytes (nope latent) | qk_rope_head_dim bytes (RoPE'd rope)]
  Total: kv_lora_rank + qk_rope_head_dim = 576 elements per token

Paged KV cache shape (after get_buffers + squeeze):
  [num_pages, kv_factor, page_size, kv_lora_rank + qk_rope_head_dim]
```

Unlike standard attention where K and V are stored separately (kv_factor=2),
MLA stores a single compressed representation. The `kv_factor` depends on the
KV cache manager implementation.

### MLA Block Table Padding

The MLA decode kernel requires block_tables aligned to superblock boundaries:
```python
pages_per_superblock = 128 // tokens_per_block
if pages_per_superblock > 1:
    num_blocks = block_tables.size(-1)
    remainder = num_blocks % pages_per_superblock
    if remainder != 0:
        pad = pages_per_superblock - remainder
        block_tables = F.pad(block_tables, (0, pad), value=0)
```

## FP4/FP8 Quantized Paths

### Q Buffer Reinterpretation (FP4 KV Cache)

In FP4 path, `qkv_preprocessing` writes FP8 Q data into a buffer typed as
the model dtype (e.g., BF16). To correctly read it:
```python
# gen_ws.q_buf is typed as BF16 (2 bytes/element)
# but C++ wrote FP8 data (1 byte/element)
q_processed = (
    gen_ws.q_buf.view(torch.uint8)               # raw bytes
    [:num_tokens * num_heads * head_size]          # trim to FP8 count
    .view(torch.float8_e4m3fn)                     # reinterpret as FP8
    .view(num_tokens, num_heads, head_size)        # reshape
)
```

The `uint8` intermediate is necessary because the nominal dtype of
`q_buf` doesn't match what the kernel actually wrote.
