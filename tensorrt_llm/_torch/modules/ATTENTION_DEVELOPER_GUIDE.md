# Attention Developer Guide

## Scope

This guide covers the TRT-LLM PyTorch attention stack:

- `tensorrt_llm/_torch/modules/attention.py`
- `tensorrt_llm/_torch/attention_backend/`
- `tensorrt_llm/_torch/attention_backend/sparse/`

Use it when modifying the current implementation or adding a new model's
attention behavior. It covers standard `Attention`, Multi-head Latent
Attention (MLA), dense backends, and sparse backends. It does not cover
`tensorrt_llm/_torch/attention_backend/star_flashinfer.py`, which is planned
for deprecation.

## Glossary

| Acronym | Meaning |
|---|---|
| MLA | Multi-head Latent Attention |
| DSA | DeepSeek Sparse Attention |
| MHA | Multi-Head Attention |
| MQA | Multi-Query Attention |
| GQA | Grouped-Query Attention |
| RoPE | Rotary Position Embedding |
| mRoPE | Multimodal Rotary Position Embedding |
| TP | Tensor Parallelism |
| CP | Context Parallelism |
| KV | Key/Value |

## How to Read the Stack

Attention in TRT-LLM is split across four layers:

1. module wrapper (`Attention` or `MLA`)
2. backend class selected by `config.attn_backend`
3. metadata subtype and runtime buffers
4. key/value (KV) cache manager and decode-time cache semantics

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
- tensor parallelism (TP) / context parallelism (CP) reshaping and mapping setup
- fused or split QKV handling
- optional unfused Rotary Position Embedding (RoPE)
- optional output gating
- optional LoRA injection
- collecting masks, sinks, output buffers, and other per-forward options into
  `AttentionForwardArgs`
- passing Q/K/V, metadata, and `AttentionForwardArgs` into the backend

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

`MLA` (Multi-head Latent Attention) is a separate module in `attention.py`.
Like `Attention`, it keeps module-level projection logic in the module,
delegates core execution to a backend object, and depends on metadata and
KV-cache contract. At a high level, it owns:

- low-rank Q decomposition
- low-rank KV decomposition
- absorbed MLA path
- MLA-specific RoPE and latent-cache flow
- integration points for sparse attention paths

`MLA` has two projection layouts: non-lite (`is_lite == False`) and lite
(`is_lite == True`). In lite mode there is no separate Q low-rank compression
stage. `is_lite` changes the projection structure, not just a small code path.

Dense MLA and current sparse MLA variants still use the same module/backend/
metadata/KV-cache split described above. Sparse-specific routing may currently
pass through `MLA`, but that should be treated as an implementation detail
rather than a stable design boundary.

For MLA-related tasks, first check whether the work fits the current
projection structure, can stay on an existing backend and metadata family, and
can preserve the current latent-cache / paged-KV contract. If it can, the
task usually stays within the existing MLA stack. If it depends on sparse
helper-level control flow, read `attention.py` and the relevant sparse
backend code directly.

## 2. Backend Layer Reference

### 2.1 Backend selection

`config.attn_backend` selects the base backend family. If
`ModelConfig.sparse_attention_config` is set, the backend class is selected
from that user-facing config's algorithm. The attention module separately
lowers the config to `SparseParams` and passes those params to the constructed
backend instance. MLA parameters can further affect backend construction.

Base backend families:

| Backend name | Class | Metadata subtype | Notes |
|---|---|---|---|
| `TRTLLM` | `TrtllmAttention` | `TrtllmAttentionMetadata` | Standard backend path |
| `VANILLA` | `VanillaAttention` | `VanillaAttentionMetadata` | Torch fallback path |
| `FLASHINFER` | `FlashInferAttention` | `FlashInferAttentionMetadata` | FlashInfer planning/runtime path |

### 2.2 Sparse backend families

Sparse attention is not selected by a separate top-level module. User-facing
`SparseAttentionConfig` objects live in LLM / VisualGen args and `ModelConfig`.
Attention modules use those configs to select sparse backend classes, then
lower the configs into `SparseParams` for backend construction. KV-cache
managers stay model-scope and consume the user-facing config directly.
Sparse metadata consumes `SparseMetadataParams`, derived independently from the
same user-facing config.

Sparse registrations are defined in `attention_backend/sparse/utils.py`. Check
that file for the current supported combinations, as they may change over time.

### 2.3 Backend contract

All backends implement the `AttentionBackend` interface.

The core contract is:

- `forward(q, k, v, metadata, forward_args=..., **kwargs)`
- `Metadata` subtype
- `AttentionForwardArgs` for per-forward optional arguments such as masks,
  output buffers, scales, RoPE/mRoPE inputs, MLA buffers, and sparse inputs
- coarse capability hooks:
  - `support_fused_rope()`
  - `support_fused_qkv()`
  - `support_mla()`

`**kwargs` is only a temporary compatibility path. It is merged into
`AttentionForwardArgs`, rejects unknown fields, and must not be mixed with
an explicit `forward_args`.

Those capability hooks are coarse checks. They do not prove that every
required operator or sparse path already exists.

### 2.4 Capability reference

Check each backend's capability hooks (`support_fused_rope()`,
`support_fused_qkv()`, `support_mla()`) directly in the code. `TrtllmAttention`
currently supports all three; other backends may not. These capabilities can
change over time.

Sparse subclasses inherit the base backend family and then add sparse-specific
metadata and cache behavior.

## 3. Runtime Contract Reference

### 3.1 Metadata families

All backend metadata types inherit from `AttentionMetadata`. The base contract
includes sequence-length and request-level state, KV-cache manager and
parameters, runtime feature flags, and optional CUDA-graph buffer management.
Sparse metadata subtypes consume only `SparseMetadataParams`, not
backend-owned `SparseParams`.

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

When KV cache is enabled, all current `_torch` backends use paged KV cache.
`VanillaAttention` also has a separate no-KV-cache path for models that do not
use cache. `KVCacheManager.get_buffers()` exposes a per-layer view of the
primary pool:

- For standard dense attention, `kv_factor = 2` (separate K and V planes).
- For MLA-style cache, `kv_factor = 1` (one latent-cache tensor per token).

The main differences across backends:

| Backend | Cache write | Cache read |
|---|---|---|
| `TRTLLM` | Backend-managed (C++ ops) | Block-table + pool pointers |
| `VANILLA` | Python-side | Python-side slicing |
| `FLASHINFER` | Python-side (explicit append) | Page-table metadata |

#### 3.2.2 `TRTLLM` internal FMHA libraries

`TrtllmAttention` dispatches attention through an ordered list of internal FMHA
libraries. `FlashInferTrtllmGenFmha` integrates trtllm-gen kernels from
FlashInfer into the `TRTLLM` backend, and `FallbackFmha` calls the regular
`thop.attention` runtime path. These are not separate attention backends.

`TLLM_FMHA_LIBS` controls the ordered list. Unset means
`flashinfer_trtllm_gen,fallback`; use `TLLM_FMHA_LIBS=fallback` or
`TLLM_FMHA_LIBS=-flashinfer_trtllm_gen` to force the fallback path. Each FMHA
library exposes `is_available()` for module/static environment checks and
`is_supported()` for per-forward request checks.

The FMHA package is split by role:

- `fmha/interface.py` defines the `Fmha` runtime contract.
- `fmha/phased.py` defines `PhasedFmha`, which handles mixed context/generation
  requests and dispatches them to phase-specific hooks.
- `fmha/flashinfer_trtllm_gen.py` implements the FlashInfer trtllm-gen FMHA
  library.
- `fmha/fallback.py` implements the regular `thop.attention` fallback library.
- `fmha/registry.py` owns `TLLM_FMHA_LIBS` parsing and library ordering.

Use `PhasedFmha` for libraries that need separate context/generation or MHA/MLA
entry points. Use `Fmha` directly for libraries that already own the full
request shape.

#### 3.2.3 MLA cached-context semantics

MLA cached state is not regular dense K and V. The paged cache stores
latent-cache state rather than separate K and V planes. Backend ops handle
appending, RoPE application, and loading cached state for attention use.

MLA fit cannot be judged from attention math alone. The module and backend must
agree on latent-cache layout, paged-KV read/write paths, and cached/chunked
context behavior. Read the MLA section of `attention.py` and the relevant
backend code for the current implementation details.

#### 3.2.4 Sparse side-cache semantics

Sparse backends may add side caches beyond the main KV cache. Some sparse
algorithms keep the standard cache manager; others replace it with a
sparse-aware cache manager that adds side caches for indexing or routing.

When evaluating new sparse attention, check both the main KV-cache contract
and the side-cache contract. See `attention_backend/sparse/` for the current
sparse cache managers and their side-cache structures.

## 4. Evaluating New Attention

### 4.1 First-pass fit

When evaluating a new attention path, compare it against the same four layers
used throughout this guide:

1. **Module layer**: can `Attention` or `MLA` express the required math with
   module-side changes only?
2. **Backend layer**: can the current `TRTLLM` backend family handle the
   required execution shape?
3. **Runtime contract**: can the state fit in an existing metadata family?
4. **KV-cache semantics**: can the cache behavior stay within the current
   paged-KV and cache-manager model?

If yes to all four, start with the `TRTLLM` backend. Treat the first mismatch
as the current blocker.

### 4.2 What to check

- **Module layer**
  Q/K/V layout, fused or split QKV, MQA/GQA structure, Q/K normalization,
  extra scaling, output gating, and pre-backend or post-backend transforms.

- **Backend layer**
  Which backend family can run the source behavior, and whether it needs fused
  RoPE, fused QKV, MLA, sparse, or chunked-context support. Do not use backend
  name alone as proof of support.

- **Positional embedding and masking**
  Whether RoPE is applied outside or fused, whether the path needs mRoPE, and
  whether masking fits the current causal, full, sliding-window, or custom
  paths.

- **Runtime contract**
  Which metadata subtype is needed, what runtime state must be carried, and
  whether the path depends on CUDA-graph assumptions.

- **KV-cache semantics**
  How K/V are appended, what layout is assumed, how cached state is indexed and
  reused, whether chunked prefill or speculative decoding matters, and whether
  sparse side caches are required.

### 4.3 Default bring-up order

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
| `tensorrt_llm/_torch/attention_backend/fmha/` | Internal TRTLLM FMHA libraries |
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
- Do not bypass MLA's context dispatcher for chunked or cached-KV cases.
- Do not duplicate RoPE handling before checking the fused path.
