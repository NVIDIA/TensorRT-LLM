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

`MLA` has two projection layouts: non-lite (`is_lite == False`) and lite
(`is_lite == True`). In lite mode there is no separate Q low-rank compression
stage. `is_lite` changes the projection structure, not just a small code path.

**Non-DSA MLA dispatch.** Non-DSA MLA uses absorption: Q, K, V are projected
through low-rank decomposition and then absorbed into the backend call. The
context path still dispatches through `forward_context()`, which handles fresh
context, cached-KV context, and chunked prefill as separate cases. Chunked
prefill routing depends on architecture (SM90 vs SM100+). Do not assume all
context goes through the same handler.

**DSA dispatch.** DSA-style MLA adds a further split: projection and attention
are separated, context and generation paths are separated, and a short-seq
gate inside the context path can route to a dense MHA fallback.

**For both DSA and non-DSA MLA:** the short-seq MHA path should stay inside
`forward_context()`. Do not bypass the dispatcher and call
`forward_context_default()` directly — it only handles fresh context, not
cached-KV or chunked-context cases.

**Practical notes:**

- `self.mha` being present does not mean DSA is disabled.
- Helix CP wraps the attention body with allgather/output-projection helpers.

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
through `sparse_attention_config` on top of a base backend family. Sparse
selection can change the backend class, metadata subtype, and KV-cache manager.

Current sparse registrations:

| Base backend | Sparse algorithm | Resulting backend class | KV-cache manager |
|---|---|---|---|
| `TRTLLM` | `rocket` | `RocketTrtllmAttention` | `RocketKVCacheManager` |
| `TRTLLM` | `dsa` | `DSATrtllmAttention` | `DSACacheManager` |
| `TRTLLM` | `skip_softmax` | `TrtllmAttention` | standard `KVCacheManager` |
| `VANILLA` | `rocket` | `RocketVanillaAttention` | `RocketKVCacheManager` |
| `VANILLA` | `dsa` | unsupported | — |
| `FLASHINFER` | sparse variants | unsupported | — |

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

| Backend family | Fused RoPE | Fused QKV input | MLA |
|---|---|---|---|
| `TrtllmAttention` | yes | yes | yes |
| `VanillaAttention` | no | no | no |
| `FlashInferAttention` | no | no | no |

Sparse subclasses inherit the base backend family and then add sparse-specific
metadata and cache behavior.

### 2.5 `TRTLLM` internal kernel paths

`TrtllmAttention` can dispatch to `trtllm_gen.py` for supported dense cases.
That is an internal fast path, not a separate top-level backend selection. It
is disabled by default and gated by `TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION`.

It only applies to a narrow subset of dense cases. If it does not apply or is
not enabled, `TrtllmAttention` stays on its regular runtime path.

## 3. Runtime Contract Reference

### 3.1 Metadata families

All backend metadata types inherit from `AttentionMetadata`. The base contract
includes sequence-length and request-level state, KV-cache manager and
parameters, runtime feature flags, optional sparse state, and optional
CUDA-graph buffer management.

**`TrtllmAttentionMetadata`** is the main metadata family. It adds paged-KV
block information, TRTLLM runtime state, chunked-prefill/speculative-decode/Helix
state, and MLA-specific state. If a source attention needs paged KV, chunked
prefill, FlashMLA, speculative decoding, or Helix-aware execution, the fit
question is mostly a `TrtllmAttentionMetadata` fit question.

**`VanillaAttentionMetadata`** is lighter — base metadata plus simple
cache-index information. Use it when the `Attention` module boundary fits but
the fused TRTLLM path is too restrictive.

**`FlashInferAttentionMetadata`** adds a planning-oriented contract with
workspace, page-table KV metadata, and prefill/decode wrapper state.

**Sparse metadata** families extend the base backend metadata with
sparse-specific runtime state (indexer buffers, routing state, side-cache
state).

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
`KVCacheManager.get_buffers()` exposes a per-layer view of the primary pool:

- For standard dense attention, `kv_factor = 2` (separate K and V planes).
- For MLA-style cache, `kv_factor = 1` (one latent-cache tensor per token).

The main differences across backends:

| Backend | Cache write | Cache read | Notes |
|---|---|---|---|
| `TRTLLM` | Backend-managed (C++ ops like `qkv_preprocessing`) | Block-table + pool pointers | Python does not call `index_copy_` in the regular path |
| `VANILLA` | Python-side (`index_copy_`) | Python-side slicing from same cache tensor | Most direct path for inspecting what is written to cache |
| `FlashInfer` | Python-side (explicit append) | Page-table metadata | Requires a planning step and its own page-table shape |

#### 3.2.2 `TRTLLM` internal `trtllm_gen` path

`trtllm_gen.py` is not a separate backend, but it has its own KV view
assumptions. It bridges the TRTLLM block-offset format into the page-table
shape expected by its kernels. This path is narrower than the main `TRTLLM`
backend (dense only, no MLA, no sparse, fused QKV only). If `trtllm_gen` does
not fit, that does not rule out the main `TRTLLM` backend.

#### 3.2.3 MLA cached-context semantics

MLA cached state is not regular dense K and V. The paged cache stores
latent-cache state, and backend ops handle:

- appending latent cache into paged storage
- applying RoPE as part of that flow
- loading paged cached state back for attention use

MLA fit cannot be judged from attention math alone. The module and backend must
agree on latent-cache layout, paged-KV read/write paths, and cached/chunked
context behavior. The short-seq MHA path is only correct if cached-KV behavior
stays inside the top-level `forward_context()` dispatcher.

#### 3.2.4 Sparse side-cache semantics

Sparse backends may add side caches beyond the main KV cache:

- `skip_softmax` keeps the standard `KVCacheManager`, no side cache.
- `DSA` uses `DSACacheManager` and adds `indexer_k_cache` for sparse indexing.
- `Rocket` uses `RocketKVCacheManager` and adds `KT` cache for sparse routing.

When evaluating new sparse attention, check both the main KV-cache contract
and the side-cache contract.

## 4. Evaluating New Attention

### 4.1 First-pass fit

For a new model, compare it against the current stack in four parts:

1. **Module math**: can `Attention` or `MLA` express the new math with
   module-side code only?
2. **Backend execution**: can `TrtllmAttention.forward(...)` handle the needed
   call shape (fused/split QKV, dense/sparse, RoPE/mRoPE, MLA/non-MLA)?
3. **Metadata**: can the runtime state fit in `TrtllmAttentionMetadata` or a
   known sparse subclass?
4. **KV-cache**: can the cache behavior stay inside the current paged-KV and
   cache-manager model?

If yes to all four, start with the `TRTLLM` backend. Treat the first mismatch
as the current blocker.

### 4.2 Checklist

#### Module-level math

- Q/K/V layout, fused or split QKV, MQA/GQA structure
- Q/K normalization, extra scaling, output gating
- Pre-backend and post-backend transforms

#### Backend capability

- Which backend family can run the source behavior
- Fused RoPE, fused QKV, MLA, sparse, chunked-context support
- Do not use backend name alone as proof of support

#### Positional embedding and masking

- RoPE applied outside vs fused, standard RoPE vs mRoPE
- Causal, full, sliding-window, or custom masks

#### Metadata and runtime contract

- Which metadata subtype, what runtime state is needed
- CUDA-graph assumptions

#### KV-cache ownership

- How K/V are appended, what layout, how indexed
- Cache reuse, chunked prefill, speculative decoding assumptions
- Sparse side-cache requirements

### 4.3 Bring-up order

Start with `TRTLLM` when the new attention fits or only needs limited changes.
Use `VANILLA` for quick bring-up or experiments when the module boundary fits
but the fused path is too costly to change initially.

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
| `tensorrt_llm/_torch/attention_backend/trtllm_gen.py` | Internal dense fast path |
| `tensorrt_llm/_torch/attention_backend/vanilla.py` | Torch fallback backend and metadata |
| `tensorrt_llm/_torch/attention_backend/flashinfer.py` | FlashInfer backend and metadata |
| `tensorrt_llm/_torch/attention_backend/sparse/` | DSA, Rocket sparse backends, metadata, cache managers |

## 6. Testing Notes

- Test lite and non-lite MLA separately when changing projection logic.
- Test eager and compiled paths separately when changing DSA MLA dispatch.
- Test fresh context, cached context, chunked context, and generation
  separately.
- Any dispatch change touching `forward_context()` needs chunked-context tests.

Key test files:

- `tests/unittest/_torch/attention/test_attention.py`
- `tests/unittest/_torch/attention/test_attention_mla.py`
- `tests/unittest/_torch/attention/test_vanilla_attention.py`
- `tests/unittest/_torch/attention/test_flashinfer_attention.py`
- `tests/unittest/_torch/attention/sparse/`

## 7. Anti-Patterns

- Do not treat attention work as "math only".
- Do not treat backend choice as independent from metadata choice.
- Do not treat KV-cache semantics as a small implementation detail.
- Do not call `forward_context_default()` directly for chunked MLA context.
- Do not duplicate RoPE handling before checking the fused path.
- Do not assume `self.mha is not None` means DSA is disabled.
