# Attention Developer Guide

## Scope

This guide covers the TRT-LLM PyTorch attention stack:

- `tensorrt_llm/_torch/modules/attention.py`
- `tensorrt_llm/_torch/attention_backend/`
- `tensorrt_llm/_torch/attention_backend/sparse/`

Use it when modifying the current implementation or adding a new model's
attention behavior. It covers standard `Attention`, MLA, dense backends, and
sparse backends. It does not cover
`tensorrt_llm/_torch/attention_backend/star_flashinfer.py`, which is planned
for deprecation.

## How to Read the Stack

Attention in TRT-LLM is split across four layers:

1. module wrapper (`Attention` or `MLA`)
2. backend class selected by `config.attn_backend`
3. metadata subtype and runtime buffers
4. KV-cache manager and decode-time cache semantics

Keep these four questions separate:

1. What math happens at the module layer around the backend call?
2. Which backend family can execute the core attention path?
3. Which metadata subtype and runtime contract does that backend require?
4. What KV-cache ownership and decode-time semantics does the path assume?

The same module math can still require a different backend, metadata subtype,
KV-cache manager, or a fallback from `TRTLLM` to `VANILLA`. Attention work is
not only score computation. It also includes how the backend reads, writes,
appends, and reuses KV cache, especially during decode.

## 1. Module Layer Reference

### 1.1 `Attention`: the module wrapper around the backend

`Attention` is not just the backend call. It owns the logic around the backend:

- QKV projection and output projection
- TP/CP reshaping and mapping setup
- fused or split QKV handling
- optional unfused RoPE
- optional output gating
- optional LoRA injection
- passing masks, sinks, and metadata into the backend

At a high level:

```text
hidden_states
  -> qkv_proj
  -> optional LoRA
  -> optional gate split
  -> optional unfused RoPE
  -> fused/split QKV conversion
  -> backend.forward(...)
  -> optional output gate
  -> o_proj
```

Important extension points in `Attention`:

- `apply_rope()`
- `apply_qk_norm()`
- `convert_qkv()`

If a source model needs extra Q/K processing, gating, scaling, or projection
layout logic, the first question is whether it can stay at this module layer
without changing the outer runtime contract.

### 1.2 `MLA`: a separate module on top of the same backend system

`MLA` is a separate module in `attention.py`. It keeps module-level
projections and tensor transforms in `MLA`, delegates core execution to backend
objects, and depends on metadata and KV-cache contract. It owns:

- low-rank Q decomposition
- low-rank KV decomposition
- absorbed MLA path
- DSA-specific dispatch
- short-seq MHA routing
- MLA-specific RoPE and latent-cache flow

`MLA` has two projection layouts.

Non-lite MLA (`is_lite == False`):

```text
MLA(nn.Module)
├── kv_a_proj_with_mqa
├── q_a_layernorm
├── q_b_proj
├── kv_a_layernorm
├── kv_b_proj
├── mha   # dense backend, used by short-seq MHA path
├── mqa   # sparse / DSA backend
└── short_seq_mha_threshold
```

Lite MLA (`is_lite == True`):

```text
MLA(nn.Module)
├── kv_a_proj_with_mqa
├── q_proj   # also assigned to q_b_proj
├── kv_a_layernorm
├── kv_b_proj
├── mha
├── mqa
└── short_seq_mha_threshold
```

In lite mode there is no separate `q_a_proj`, `q_a_layernorm`, or `kv_a_proj`.
`q_proj` is used as `q_b_proj`.

### 1.3 MLA dispatch reference

For DSA-style MLA models, the dispatch is:

```text
forward()
  -> forward_impl_with_dsa()
     -> forward_dsa_proj()
     -> forward_dsa_attn()
        -> context tokens?
           -> yes: forward_context_dsa()
              -> short-seq gate?
                 -> yes: forward_context()
                    -> forward_context_default()
                    -> forward_context_with_cached_kv()
                    -> forward_context_with_chunked_prefill()
                 -> no: absorption / sparse MLA path
        -> generation tokens?
           -> yes: forward_generation_dsa()
```

For MLA maintenance:

- the short-seq MHA path should stay inside `forward_context()`
- do not bypass the dispatcher and call `forward_context_default()` directly

`forward_context_default()` only handles fresh context. It does not cover
cached-KV or chunked-context cases.

### 1.4 Practical MLA notes

Structure and dispatch:

- `is_lite` changes the projection structure, not just a small code path.
- `self.is_dsa == True` means the DSA path is active.
- `self.mqa` is the sparse DSA backend.
- `self.mha` is a dense backend used only for the short-seq dense context path.
- `self.mha` being present does not mean DSA is disabled.
- `_should_use_short_mha()` uses `max_ctx_kv_len` when available, not just the
  new-token count.

`torch.compile` path:

- the compiled path may use custom-op based execution paths
- under `torch.compile`, `_should_use_short_mha()` returns `False`, so the
  split DSA path is always used

Helix CP path:

- `_helix_cp_allgather_input()` runs before the attention body on layers after
  the first
- `_helix_cp_output_projection()` runs after the attention body

## 2. Backend Layer Reference

### 2.1 Backend selection

Backends are created by:

- `get_attention_backend()`
- `create_attention()`

The backend is chosen from:

- `config.attn_backend`
- optional `sparse_attention_config`
- optional MLA parameters

Base backend families:

| Backend name | Class | Metadata subtype | Notes |
|---|---|---|---|
| `TRTLLM` | `TrtllmAttention` | `TrtllmAttentionMetadata` | Standard backend path |
| `VANILLA` | `VanillaAttention` | `VanillaAttentionMetadata` | Torch fallback path |
| `FLASHINFER` | `FlashInferAttention` | `FlashInferAttentionMetadata` | FlashInfer planning/runtime path |

### 2.2 Sparse backend families

Sparse attention is not selected by a separate top-level module. It is resolved
through `sparse_attention_config` on top of a base backend family.

Current sparse registrations:

| Base backend | Sparse algorithm | Resulting backend class | Metadata subtype | KV-cache manager |
|---|---|---|---|---|
| `TRTLLM` | `rocket` | `RocketTrtllmAttention` | `RocketTrtllmAttentionMetadata` | `RocketKVCacheManager` |
| `TRTLLM` | `dsa` | `DSATrtllmAttention` | `DSAtrtllmAttentionMetadata` | `DSACacheManager` |
| `TRTLLM` | `skip_softmax` | `TrtllmAttention` | `TrtllmAttentionMetadata` | standard `KVCacheManager` |
| `VANILLA` | `rocket` | `RocketVanillaAttention` | `RocketVanillaAttentionMetadata` | `RocketKVCacheManager` |
| `VANILLA` | `dsa` | unsupported | none | none |
| `VANILLA` | `skip_softmax` | unsupported | none | none |
| `FLASHINFER` | sparse variants | unsupported | none | none |

### 2.3 Backend contract

All backends implement the `AttentionBackend` interface.

The core contract is:

- `forward(q, k, v, metadata, attention_mask=..., **kwargs)`
- `Metadata` subtype
- coarse capability hooks:
  - `support_fused_rope()`
  - `support_fused_qkv()`
  - `support_mla()`

Those capability hooks are coarse checks. They do not prove that every
required operator or sparse path already exists.

### 2.4 Capability reference

The current coarse capability picture is:

| Backend family | Fused RoPE | Fused QKV input | MLA |
|---|---|---|---|
| `TrtllmAttention` | yes | yes | yes |
| `VanillaAttention` | no | no | no |
| `FlashInferAttention` | no | no | no |

Sparse subclasses inherit the base backend family and then add sparse-specific
metadata and cache behavior.

### 2.5 `TRTLLM` internal kernel paths

`TrtllmAttention` can dispatch to `trtllm_gen.py` for supported dense cases.
That is an internal fast path, not a separate top-level backend selection.

It only applies to a subset of dense cases. If it does not apply,
`TrtllmAttention` stays on its regular runtime path.

## 3. Runtime Contract Reference

### 3.1 Metadata families

All backend metadata types inherit from `AttentionMetadata`.

#### 3.1.1 Base `AttentionMetadata`

The common contract includes:

- sequence-length and request-level runtime state
- KV-cache manager and KV-cache parameters
- runtime feature flags
- optional sparse-attention state
- optional CUDA-graph buffer management

The base metadata also supports cross attention through `seq_lens_kv` and
cross sub-metadata. Backend support is not uniform:

- `FlashInferAttention` has explicit cross-attention handling
- `TrtllmAttention` currently asserts that cross attention is not supported

#### 3.1.2 `TrtllmAttentionMetadata`

`TrtllmAttentionMetadata` is the main metadata family for the standard TRTLLM
path.

It extends the base metadata with:

- paged-KV block information
- request and sequence state for TRTLLM runtime execution
- runtime state for chunked prefill, speculative decode, and Helix
- MLA-specific runtime state when MLA is active

If a source attention implementation needs paged KV, chunked prefill,
FlashMLA, speculative decoding, or Helix-aware execution, the fit question is
mostly a `TrtllmAttentionMetadata` fit question.

#### 3.1.3 `VanillaAttentionMetadata`

`VanillaAttentionMetadata` mainly prepares:

- base attention metadata
- simple cache-index information for torch-side cache access

The actual attention computation is mostly done in torch. Use it when the
current `Attention` module boundary still fits but the fused TRTLLM path is
too restrictive.

#### 3.1.4 `FlashInferAttentionMetadata`

`FlashInferAttentionMetadata` adds a planning-oriented runtime contract:

- workspace and planning state
- page-table style KV metadata
- prefill and decode wrapper state

This backend requires a planning step and a paged-table runtime shape.

#### 3.1.5 Sparse metadata families

Sparse backends extend the base metadata family rather than inventing an
unrelated attention interface.

`DSAtrtllmAttentionMetadata` extends `TrtllmAttentionMetadata` with:

- indexer-side sparse runtime state
- top-k and token-to-request routing state
- extra state for sparse context and generation flows

`RocketTrtllmAttentionMetadata` and `RocketVanillaAttentionMetadata` add:

- sparse-window and routing state
- sparse offsets and sequence state
- KT-cache related state

### 3.2 KV-cache and decode-time semantics

The main question is not just "does the backend read K and V?" but:

- who owns the cache
- what cache layout the backend assumes
- how new tokens are appended
- whether decode updates happen in place
- how pages or blocks are indexed
- whether cached KV can be revisited during context
- whether sparse state must be maintained alongside KV

A backend may support the score computation you want, but still be the wrong
fit because it assumes a different KV-cache layout or a different decode-time
update pattern.

#### 3.2.1 Common paged-KV model

All current `_torch` backends are built on top of paged KV cache.

At the cache-manager level, `KVCacheManager.get_buffers()` exposes a per-layer
view of the primary pool in two common layouts:

- `NHD`: `[num_pages, kv_factor, tokens_per_block, num_kv_heads, head_dim]`
- `HND`: `[num_pages, kv_factor, num_kv_heads, tokens_per_block, head_dim]`

For standard dense attention, `kv_factor = 2`, which means separate K and V
planes.

For MLA-style cache, `kv_factor = 1`. The cache stores one latent-cache tensor
per token rather than separate K and V planes.

The main differences across backends are:

- which tensor view or block table the backend expects
- whether the backend writes cache internally or Python writes it explicitly
- whether extra side cache is required in addition to the main KV cache

#### 3.2.2 `TRTLLM` backend

At the C++ boundary, the main contract is a block-offset table plus pool
pointers, not only a reshaped cache tensor view.

In the Python metadata path, this appears as fields such as:

- `kv_cache_block_offsets`
- `host_kv_cache_pool_pointers`
- `host_kv_cache_pool_mapping`
- `cache_indirection`

Read and write behavior:

- decode and prefill cache updates are backend-managed
- dense cache writes go through backend ops such as `qkv_preprocessing`
- some sparse updates go through backend postprocessing
- the regular path reads cached KV through the block-table and pool-pointer
  contract

Python does not update the main KV cache with `index_copy_` in the regular
`TRTLLM` path.

#### 3.2.3 `TRTLLM` internal `trtllm_gen` path

`trtllm_gen.py` is not a separate top-level backend, but it has its own KV
view assumptions.

It still starts from the same paged cache state, but it bridges the TRTLLM
block-offset format into the page-table shape expected by the FlashInfer
`trtllm_gen` kernels.

In practice:

- cache writes still begin from TRTLLM preprocessing
- the fast path converts K-side block offsets into shared page indices
- it reads cache again through `KVCacheManager.get_buffers(...)`

This path is narrower than the main `TRTLLM` backend:

- dense only
- no MLA
- no sparse attention
- no cross attention
- fused QKV only

If `trtllm_gen` does not fit, that does not rule out the main `TRTLLM`
backend.

#### 3.2.4 `FlashInfer` backend

`FlashInfer` also uses paged KV cache, but its runtime contract is more
directly page-table oriented. It reads the cache through
`kv_cache_manager.get_buffers(...)` using the layout requested by
`metadata.kv_layout`.

Read and write behavior:

- Python explicitly appends current K and V into paged cache
- FlashInfer wrappers then read from that paged cache using page-table metadata

#### 3.2.5 `VANILLA` backend

`VANILLA` gets a paged cache tensor from `kv_cache_manager.get_buffers()` and
request block ids from `block_ids_per_seq`.

Read and write behavior:

- Python writes K and V directly into cache
- Python slices the same cache tensor to rebuild key and value states
- sparse token filtering, if any, also happens around this Python-side path

#### 3.2.6 MLA cached-context semantics

MLA adds a different cache shape and different decode-time assumptions.

The main difference is that MLA cached state is not regular dense K and V.
The paged cache stores latent-cache state, and backend ops handle:

- appending latent cache into paged storage
- applying RoPE as part of that flow
- loading paged cached state back for attention use

MLA fit cannot be judged from attention math alone. The module and backend also
have to agree on:

- latent-cache layout
- paged-KV read path
- paged-KV write path
- cached-context and chunked-context behavior

The short-seq MHA path is only correct if cached-KV behavior stays inside the
top-level `forward_context()` dispatcher.

#### 3.2.7 Sparse side-cache semantics

Sparse backends may change more than the attention score path. They may also
add side caches.

`skip_softmax` keeps the standard `KVCacheManager`.

`DSA` uses `DSACacheManager`:

- the main cache is still the paged MLA-style cache
- DSA also adds `indexer_k_cache`
- this side cache is used for sparse indexing and top-k related work

`Rocket` uses `RocketKVCacheManager`:

- the main cache still follows the base backend family
- Rocket also adds `KT` cache
- this side cache is used for sparse routing and sparse block selection

When evaluating a new sparse attention implementation, check:

- the main KV-cache contract
- the side-cache contract

## 4. Evaluating New Attention

### 4.1 First-pass fit against the current `TRTLLM` backend

For a new model, first compare it against the current `TRTLLM` backend surface
in four parts.

| What to compare | Current `TRTLLM` backend surface |
|---|---|
| Module-layer math around the backend call | `Attention` and `MLA` already handle QKV projection, fused or split QKV conversion, optional unfused RoPE, Q/K normalization hooks, output gating, LoRA, TP/CP handling, and Helix wrappers. |
| Backend-side execution | `TrtllmAttention` handles the standard dense path. It supports fused RoPE, fused QKV input, MLA mode, sliding-window style `attention_window_size`, mRoPE config, attention sinks, and sparse variants through registered sparse backends. |
| Metadata and runtime contract | The direct path expects `TrtllmAttentionMetadata` or one of its sparse subclasses. That means the new attention must still fit request-based metadata, paged-KV metadata, runtime feature flags, and any extra sparse or MLA buffers required by the path. |
| KV-cache semantics | The direct path assumes paged-KV ownership. Dense TRTLLM uses the standard `KVCacheManager`. Sparse paths can swap in `DSACacheManager` or `RocketKVCacheManager`. Backend choice also implies assumptions about KV-cache layout, decode-time reads, and inplace append or update behavior. If the new attention needs a different cache ownership model or different decode-time cache semantics, it is not a direct fit. |

A practical check:

1. Write down the new attention in four buckets:
   - module math before or after the backend call
   - backend features it needs
   - metadata and runtime state it needs
   - KV-cache ownership, layout assumptions, and update rules it needs
2. Compare each bucket against the current `TRTLLM` backend surface above.
3. Treat the first mismatch as the current blocker.

For direct fit, ask these questions in order:

1. Can `Attention` or `MLA` express the new math with module-side code only?
2. Can `TrtllmAttention.forward(...)` express the backend call shape that is
   needed?
   - fused or split QKV
   - dense or sparse path
   - RoPE or mRoPE handling
   - windowed masking
   - MLA or non-MLA
3. Can the runtime state be stored in `TrtllmAttentionMetadata` or a known
   sparse metadata subclass?
4. Can the KV-cache behavior stay inside the current paged-KV and cache-manager
   model?

If the answer stays "yes" through all four questions, start with the current
`TRTLLM` backend and then check the runtime contract in Section 3.

### 4.2 Detailed checklist

#### 4.2.1 Module-level math

Check all math around the backend call:

- Q/K/V layout
- fused or split QKV input
- MQA or GQA structure
- Q/K normalization
- extra QK scaling terms
- output gating
- pre-backend and post-backend transforms

The first question is whether this math can stay at the module layer by
overriding or extending `Attention` / `MLA`, without changing the outer runtime
contract.

#### 4.2.2 Backend capability mapping

Check which backend family can actually run the source behavior:

- `TRTLLM`
- `VANILLA`
- `FLASHINFER`
- sparse variants on top of those families

Then check capability assumptions explicitly:

- fused RoPE
- fused QKV input
- MLA support
- sparse operator support
- chunked-context support

Do not use backend name alone as proof of support. For a first-pass fit check,
use Section 4.1 before looking at other backends.

#### 4.2.3 Positional embedding and mask contract

Check:

- whether RoPE is applied outside the backend or fused into it
- standard RoPE vs mRoPE
- causal, full, sliding-window, or custom masks
- any sink-token or mask-side special logic

#### 4.2.4 Metadata and runtime contract

Check which metadata subtype is required:

- `TrtllmAttentionMetadata`
- `VanillaAttentionMetadata`
- `FlashInferAttentionMetadata`
- sparse metadata subclasses

Then check whether the source behavior needs runtime state such as:

- request-level state
- KV-cache manager handles and KV-cache parameters
- runtime feature flags
- planning or sparse buffers
- CUDA-graph assumptions

#### 4.2.5 KV-cache ownership and decode-time semantics

Check:

- how K/V are appended
- what KV-cache layout the backend expects
- how tokens are indexed in cache
- whether cached KV is revisited during context
- whether decode updates require inplace cache writes
- whether cache reuse or chunked prefill is required
- whether speculative decoding assumptions are involved
- whether sparse state is attached to cache ownership

### 4.3 Bring-up order

Start with the `TRTLLM` backend when the new attention fits the existing
runtime contract or only needs limited changes. If that path is too costly for
initial bring-up or quick experiments, use `VANILLA` to validate the math and
outer module behavior first, then re-evaluate whether the implementation
should move back to `TRTLLM`.

Working rules:

- Stay on `Attention` or `MLA` plus an existing backend family when possible.
- Extend the `TRTLLM` backend path before adding a new backend.
- Extend module-level hooks before adding a new backend.
- Follow an existing sparse family pattern before adding a new sparse
  abstraction.
- Treat cache-manager mismatch as a real blocker.

## 5. Key File Map

| File | Role |
|------|------|
| `tensorrt_llm/_torch/modules/attention.py` | Standard attention and MLA module logic |
| `tensorrt_llm/_torch/attention_backend/interface.py` | Backend contract, base metadata, capability hooks |
| `tensorrt_llm/_torch/attention_backend/utils.py` | Backend and sparse-backend selection |
| `tensorrt_llm/_torch/attention_backend/trtllm.py` | TRTLLM backend and metadata |
| `tensorrt_llm/_torch/attention_backend/trtllm_gen.py` | Internal dense fast path used by `TrtllmAttention` on supported Blackwell configurations |
| `tensorrt_llm/_torch/attention_backend/vanilla.py` | Torch fallback backend and metadata |
| `tensorrt_llm/_torch/attention_backend/flashinfer.py` | FlashInfer backend and metadata |
| `tensorrt_llm/_torch/attention_backend/sparse/dsa.py` | DSA sparse backend, metadata, indexer, cache manager |
| `tensorrt_llm/_torch/attention_backend/sparse/kernel.py` | Triton helper kernels used by sparse attention implementations |
| `tensorrt_llm/_torch/attention_backend/sparse/rocket.py` | Rocket sparse backends, metadata, cache manager |
| `tensorrt_llm/_torch/attention_backend/sparse/utils.py` | Sparse backend and sparse cache-manager registration |
| `tensorrt_llm/_torch/models/modeling_deepseekv3.py` | DeepSeek weight loading, including `kv_b_proj` layout transforms |

## 6. Testing Notes

- `mla.to(device)` does not move `mla.mqa.indexer` weights automatically.
- Copy indexer weights explicitly in A/B tests.
- Initialize `kv_b_proj` weights in loaded TRT-LLM layout, not HuggingFace
  layout.
- Test lite and non-lite MLA separately when changing projection logic.
- Test eager and compiled paths separately when changing DSA MLA dispatch or
  custom-op behavior.
- Test fresh context, cached context, chunked context, and generation
  separately.
- Any dispatch change touching `forward_context()` needs chunked-context tests.

Useful tests:

- `tests/unittest/_torch/attention/test_attention.py`
- `tests/unittest/_torch/attention/test_attention_mla.py`
- `tests/unittest/_torch/attention/test_vanilla_attention.py`
- `tests/unittest/_torch/attention/test_flashinfer_attention.py`
- `tests/unittest/_torch/attention/sparse/test_short_seq_mha.py`
- `tests/unittest/_torch/attention/sparse/test_sparse_mla_forward.py`

## 7. Anti-Patterns

- Do not treat attention work as "math only".
- Do not treat backend choice as independent from metadata choice.
- Do not treat KV-cache semantics as a small implementation detail.
- Do not call `forward_context_default()` directly for chunked MLA context.
- Do not duplicate RoPE handling before checking the fused path.
- Do not assume `self.mha is not None` means DSA is disabled.
