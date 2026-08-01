# Sparse Attention Development Guide

This guide is for contributors adding a new sparse attention algorithm
to TensorRT LLM. It walks through the framework hooks each algorithm
plugs into and the registration steps needed for the runtime to pick up
the new backend.

For the user-facing configuration surface, see
[Sparse Attention](../features/sparse-attention.md). For the design
rationale and high-level architecture diagrams, see the
[Sparse Attention tech blog][tech-blog].

[tech-blog]: ../blogs/tech_blog/blog17_Sparse_Attention_in_TensorRT-LLM.md

- [Two integration levels](#two-integration-levels)
- [Lowered sparse parameters](#lowered-sparse-parameters)
- [Framework-level sparse attention](#framework-level-sparse-attention)
  - [Prediction hooks](#prediction-hooks)
  - [AttentionOp behavior](#attentionop-behavior)
  - [Auxiliary memory pools](#auxiliary-memory-pools)
- [Adding a new framework-level algorithm](#adding-a-new-framework-level-algorithm)
  - [1. Configuration class](#1-configuration-class)
  - [2. Prediction module](#2-prediction-module)
  - [3. Auxiliary memory](#3-auxiliary-memory)
  - [4. Registration and dispatch](#4-registration-and-dispatch)
- [Kernel-level sparse attention](#kernel-level-sparse-attention)
- [Roadmap](#roadmap)

## Two integration levels

TensorRT LLM's sparse attention algorithms fall into two categories.

- **Framework-level**: the algorithm runs a *prediction* step that emits
  sparse indices, which are then consumed by a shared `AttentionOp` to
  produce sparse KV cache updates and/or sparse attention computation.
  Examples: **RocketKV** (page-level, MQA/GQA), **DSA** (token-level,
  MLA).
- **Kernel-level**: sparsity is implemented entirely inside the
  attention kernel — there is no external prediction or gather step.
  The kernel decides what to skip from runtime values such as Softmax
  scores. Example: **Skip Softmax Attention (BLASST)**. The only
  framework dependency is `sparse_attention_config` plumbing for
  selecting the backend; everything else lives in the kernel.

This guide focuses primarily on the framework-level path. Kernel-level
algorithms reuse the same configuration surface but skip the prediction
and memory-management sections below.

## Lowered sparse parameters

Sparse attention has two configuration layers.

- **User-facing sparse configs** live in `tensorrt_llm/llmapi/llm_args.py`
  for LLM and `tensorrt_llm/visual_gen/sparse_attention.py` for
  VisualGen. They are the Python/YAML surface and may also merge data
  from checkpoint `config.json`.
- **Lowered sparse params** live under
  `tensorrt_llm/_torch/attention_backend/sparse/`. They are backend-owned
  runtime objects consumed by attention implementations and metadata
  builders.

The lowering boundary is intentional: `AttentionBackend` instances
should not keep or interpret user-facing config objects. Before an
attention backend is constructed, the model layer calls
`to_sparse_params(...)` on the user config. That method resolves
per-model, per-layer, checkpoint, and default values into an
algorithm-specific `SparseParams` dataclass, or returns `None` when the
algorithm should not apply to that layer. The resolved object is then
passed to `create_attention(..., sparse_params=...)` and stored on the
backend instance.

Algorithms that need sparse metadata, auxiliary buffers, or per-batch
runtime state also implement `to_sparse_metadata_params(...)`. This
returns an algorithm-specific `SparseMetadataParams` object for
`AttentionMetadata`, analogous to how `to_sparse_params(...)` returns
`SparseParams` for `AttentionBackend`. Keep them separate: metadata
params describe allocation and runtime metadata state, while sparse
params describe per-attention-layer kernel or prediction behavior.

When adding a new algorithm, define concrete parameter dataclasses next
to the backend implementation, implement the two lowering methods on the
public config class, and make backend code consume only the lowered
params.

## Framework-level sparse attention

Framework-level sparse attention primarily targets approaches that
leverage **token/sequence sparsity** — for many queries only a small
fraction of historical tokens meaningfully contribute to the output,
and the framework exploits that in a GPU-friendly, structured way.
The attention operator provides unified APIs for both **sparse
computation** and **sparse KV cache**, so algorithm authors only need
to identify the important query/key pairs; everything else (KV cache
layout, kernel dispatch, page alignment) is handled by the framework.

It is built around three layers:

- **Prediction module** — generates `sparse_kv_indices` (which KV
  tokens to keep in cache) and `sparse_attn_indices` (which KV pages or
  tokens to attend to during compute).
- **`AttentionOp`** — consumes those indices via pre/post kernels and
  drives the core attention kernels. The op already understands
  page-level sparsity for MQA/GQA in the generation phase, token-level
  sparsity for MLA in both phases, and token-level KV compression in
  the context phase for MQA/GQA.
- **Auxiliary memory subsystem** — manages any extra pools (KT cache,
  indexer K cache, …) alongside the main KV cache.

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/media/sparse_attention_framework.png" alt="Framework support for sparse attention in TensorRT LLM" width="800">
</figure>
</div>
<p align="center"><sub><em>Figure 1: Framework support for sparse attention in TensorRT LLM.</em></sub></p>

Architecturally, each sparse attention algorithm subclasses the shared
`AttentionBackend` and supplies its own `sparse_kv_predict` /
`sparse_attn_predict` implementations. Different attention layers
within a single model can use different backends, so a model can mix
sparse attention strategies layer by layer. The shared `AttentionOp`
performs the actual computation and is not modified by individual
algorithms.

The current capability matrix is:

| Attention type | Context phase | Generation phase |
|---|---|---|
| MQA / MHA / GQA | sparse KV cache | sparse computation (page-level) |
| MLA | sparse computation (token-level) | sparse computation (token-level) |

Context-phase sparse computation for MQA/GQA and dynamic generation-phase
KV eviction are tracked as future work.

### Prediction hooks

`AttentionBackend` exposes two prediction methods that algorithm-specific
subclasses override:

```python
sparse_kv_indices, sparse_kv_offsets = self.sparse_kv_predict(q, k, metadata, **kwargs)
sparse_attn_indices, sparse_attn_offsets = self.sparse_attn_predict(q, k, metadata, **kwargs)
```

Different KV heads are allowed to emit different sparse index sets; Q
heads that map to the same KV head share the KV head's sparse pattern.

Algorithm implementations live under
`tensorrt_llm/_torch/attention_backend/sparse/`:

- `rocket.py`, `dsa.py` — concrete algorithms.
- `kernel.py` — custom Triton kernels (importance scoring, Top-K).
- `utils.py` — dispatch helpers.

### AttentionOp behavior

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/media/sparse_attention_op.png" alt="Sparse attention operator workflow in TensorRT LLM" width="800">
</figure>
</div>
<p align="center"><sub><em>Figure 2: Sparse attention operator workflow in TensorRT LLM.</em></sub></p>

For MQA/GQA, the op runs `gatherKvPageOffsetsKernel` before the
generation-phase attention kernel. It takes the (potentially unordered
or finer-grained) sparse indices and maps them to ordered, page-aligned
KV cache offsets, also producing an updated per-head effective KV
length. The downstream attention kernel reads only those pages. Today
MQA/GQA sparse computation is supported at **block (page) granularity**
in the generation phase only.

After context attention, `updateSparseKvCacheAfterFmha` post-processes
the KV cache: it selects the important KV tokens and rewrites the
corresponding K/V vectors in place to shrink the cache. The indices
must be **sorted** so the in-place gather is safe; this preserves
compatibility with features such as chunked prefill at the cost of an
extra write.

For sparse MLA, the kernel consumes token-level indices directly, so
`gatherKvPageOffsetsKernel` is bypassed — both context and generation
phases are supported at token granularity. The sparse MLA path
currently expects **global** KV cache pool addresses with token-level
offsets, not request-local logical positions. Sparse KV cache for MLA
is not yet supported.

### Auxiliary memory pools

Two paths exist for managing auxiliary tensors today; new algorithms
should prefer **`KVCacheManagerV2`** when starting fresh.

- **`KVCacheManagerV2` (recommended for new work)**: Python-side,
  hierarchical, supports heterogeneous pools per layer with automatic
  coalescing within a lifecycle group. Adding an auxiliary pool only
  requires defining a per-layer `AttentionLayerConfig` and `BufferConfig`.
- **`KVCacheManager` (legacy path used by RocketKV/DSA today)**: either
  inherit from it at the Python level (RocketKV's `RocketKVCacheManager`),
  or integrate directly into the C++ `KVCacheManager` (DSA's indexer K
  cache). The Python path is faster to iterate on; the C++ path is
  required for KV cache reuse and disaggregated serving.

Note: algorithms that evict KV blocks generally cannot coexist with the
standard KV cache block reuse, because eviction changes block contents
per request. Low-rank-only approaches like DSA's indexer K cache can
still reuse blocks.

## Adding a new framework-level algorithm

The four steps below cover what the runtime needs in order to dispatch a
new algorithm end-to-end. The order matches the natural development
flow — config first, then prediction, then memory, then registration.

### 1. Configuration class

Define a configuration class in `tensorrt_llm/llmapi/llm_args.py`
inheriting from `BaseSparseAttentionConfig`. Hold all user-tunable
parameters here and pick a unique `algorithm` discriminator literal.

```python
class MySparseAttentionConfig(BaseSparseAttentionConfig):
    algorithm: Literal["my_algo"] = "my_algo"
    topk: int = 64
    # ... other parameters
```

Add the new class to the discriminated `SparseAttentionConfig` union at
the bottom of the file.

### 2. Prediction module

Create a new backend class inheriting from `TrtllmAttention` (or
`VanillaAttention` if appropriate) in
`tensorrt_llm/_torch/attention_backend/sparse/`. Override one or both
prediction methods.

**`sparse_kv_predict(self, q, k, metadata, **kwargs)`**

- **Behavior**: return the indices of tokens to retain in the KV cache.
- **Outputs**:
  - `sparse_kv_indices`: shape `(nHeads, nTokens)` — token indices on
    the sequence dimension, where `nHeads` is the number of KV heads
    and `nTokens` is the total selected tokens across the batch.
  - `sparse_kv_offsets`: shape `(nBatch + 1)` — sample boundaries; the
    indices for head `h` and sample `n` are
    `sparse_kv_indices[h, sparse_kv_offsets[n]:sparse_kv_offsets[n+1]]`.
- **Constraint**: indices must be **sorted** so the post-attention
  in-place gather (`updateSparseKvCacheAfterFmha`) is safe. The sort
  cost buys compatibility with chunked prefill and similar features.

**`sparse_attn_predict(self, q, k, metadata, **kwargs)`**

- **Behavior**: return the sparse indices used by the generation-phase
  attention computation.
- **Outputs**:
  - `sparse_attn_indices`: shape `(nHeads, nBlocks)` — block indices on
    the KV sequence dimension. Block size is set by the algorithm via
    `sparse_attn_indices_block_size` (arbitrary value supported).
  - `sparse_attn_offsets`: shape `(nBatch + 1)` — same semantics as
    above.
- **Constraint**: today only **page-level** granularity is supported
  for MQA/GQA sparse computation, and the generation-phase path uses
  TRTLLM-GEN kernels (NVIDIA Blackwell SM 100+).

Prediction is on the critical path and can dominate latency in
low-latency scenarios. Plan for custom kernels (Triton or CUDA) rather
than relying on generic PyTorch ops.

### 3. Auxiliary memory

If the algorithm needs extra tensors beyond the main KV cache:

- **`KVCacheManagerV2` (preferred for new algorithms)**: define a
  per-layer `AttentionLayerConfig` and a `BufferConfig` for the
  auxiliary buffer; the V2 manager groups layers by lifecycle and
  coalesces buffers automatically. No C++ changes required.
- **Python-level custom manager (legacy `KVCacheManager`)**: subclass
  `KVCacheManager`, reuse `BlockManager` for the auxiliary pool, and
  override `get_cache_size_per_token` / `get_cache_bytes_per_token` so
  the runtime allocates enough GPU memory, plus
  `add_dummy_requests` / `prepare_resources` so the pool gets the right
  resources at request time. Easier to iterate; no KV cache reuse or
  disagg-serving.
- **C++ integrated manager**: extend the C++ `KVCacheManager` itself.
  Required for advanced features (KV cache reuse, disaggregated
  serving). Significantly higher implementation cost.

### 4. Registration and dispatch

- Register the new config + backend in
  `tensorrt_llm/_torch/attention_backend/sparse/utils.py` and
  `tensorrt_llm/_torch/pyexecutor/_util.py` so the runtime routes
  requests to your backend when the config is present.
- If your algorithm exposes new C++ parameters, plumb them through
  `cpp/tensorrt_llm/thop/attentionOp.cpp` and
  `cpp/tensorrt_llm/kernels/sparseAttentionKernels.h`.

## Kernel-level sparse attention

Kernel-level algorithms reuse the same `sparse_attention_config`
selection but bypass the prediction and memory-management hooks
entirely. Implementation lives inside the attention kernel; the only
framework wiring is:

- A new config subclass with its own `algorithm` discriminator.
- A lowered `SparseParams` object that carries the resolved kernel
  settings.
- A switch inside the attention backend (e.g.,
  `_torch/attention_backend/trtllm_gen.py`) that reads the lowered params
  and enables the kernel-side fast path.

Skip Softmax Attention follows this pattern — see the
[BLASST tech blog](../blogs/tech_blog/blog16_Accelerating_Long_Context_Inference_with_Skip_Softmax_Attention.md)
for the kernel-side specifics.

## Roadmap

- **Sparse computation in context phase for MQA/MHA/GQA** — extend
  framework coverage to context-phase sparse compute.
- **Dynamic eviction in generation phase** — exploring block-level
  eviction as a compromise that keeps KV cache flexibility manageable.
- **Unified auxiliary memory management** — let custom auxiliary pools
  inherit KV-cache features (reuse, offloading) by default.
- **Code refactoring** — as more algorithms land, unify the
  framework-level scaffolding for maintainability.
