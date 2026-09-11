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
  sparse indices. A hook-based implementation can pass those indices to the
  shared `AttentionOp`, while a dedicated backend can own prediction and
  sparse computation end to end. Examples: **RocketKV** (token-level prompt
  eviction plus page-level MHA/MQA/GQA decode selection), **DSA**
  (token-level MLA), and **MiniMax-M3** (block-level GQA).
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
  `tensorrt_llm/_torch/attention/backends/sparse/`. They are backend-owned
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
and the framework exploits that in a GPU-friendly, structured way. On
the shared `AttentionOp` integration path, the operator provides APIs
for both **sparse computation** and **sparse KV cache** and owns KV-cache
layout conversion, kernel dispatch, and page alignment. An algorithm
with a dedicated attention implementation can instead perform those
steps in its backend while still using the common sparse config,
metadata, cache-manager, and registry framework.

The shared `AttentionOp` path is built around three layers:

- **Prediction module** — generates `sparse_kv_indices` (which KV
  tokens to keep in cache) and `sparse_attn_indices` (which KV pages or
  tokens to attend to during compute).
- **`AttentionOp`** — consumes those indices via pre/post kernels and
  drives the core attention kernels. The op already understands
  page-level sparsity for MHA/MQA/GQA in the generation phase,
  token-level MQA/GQA and MLA sparsity in both phases, and token-level
  KV compression in the context phase for MHA/MQA/GQA.
- **Auxiliary memory subsystem** — manages any extra pools (KT cache,
  indexer K cache, …) alongside the main KV cache.

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/media/sparse_attention_framework.png" alt="Framework support for sparse attention in TensorRT LLM" width="800">
</figure>
</div>
<p align="center"><sub><em>Figure 1: Framework support for sparse attention in TensorRT LLM.</em></sub></p>

Hook-based `TrtllmAttention` implementations supply `sparse_kv_predict` /
`sparse_attn_predict` and reuse the shared `AttentionOp` stack. Algorithms that
select whole KV blocks instead supply `block_sparse_attn_predict`; their routes
bypass `AttentionOp` and run on the generic block-sparse FMHA described in the
[feature guide](../features/sparse-attention.md#block-sparse-mha-mqa-gqa). RocketKV's
`VanillaAttention` implementation instead uses per-request Python hooks. A
dedicated backend can implement sparse computation directly; MiniMax-M3's
default Triton backend follows this model. Different attention layers within a
model can use different backends, so sparse strategies can be mixed layer by
layer.

The current capability matrix is:

| Attention type | Context phase | Generation phase |
|---|---|---|
| MQA / GQA | sparse KV cache and sparse computation (token-level) | sparse computation (token- or page-level) |
| MHA | sparse KV cache | sparse computation (page-level) |
| MLA | sparse computation (token-level) | sparse computation (token-level) |
| Block-sparse MHA / MQA / GQA | sparse computation (block-level, contiguous Q/K/V without a KV cache) | sparse computation (block-level, paged) |

Dynamic generation-phase KV eviction is tracked as future work.

### Prediction hooks

`TrtllmAttention`-based sparse backends expose three prediction methods that
algorithm-specific subclasses override:

```python
sparse_kv_indices, sparse_kv_offsets = self.sparse_kv_predict(q, k, metadata, forward_args)
sparse_attn_indices, sparse_attn_offsets = self.sparse_attn_predict(q, k, metadata, forward_args)
block_sparse_inputs = self.block_sparse_attn_predict(q, k, v, metadata, forward_args)
```

`prepare_sparse_runtime_params` in `sparse/hooks.py` runs all three hooks once
per call regardless of whether the backend carries `SparseParams`, applies the
SkipSoftmax threshold schedule when the backend carries `SkipSoftmaxParams`,
and returns a new per-call `SparseRuntimeParams` built from the caller's
`AttentionForwardArgs.sparse_runtime_params` plus the hook results. The core
forward assigns the returned carrier back to that field before FMHA dispatch.
Backends that need runtime state outside the three hooks (DSA's auxiliary pool
pointer, DeepSeek-V4's per-token KV lengths) write it into the caller's carrier
before or inside their hooks, and `prepare_sparse_runtime_params` carries those
fields over.
`AttentionForwardArgs.sparse_backend_args` carries
algorithm inputs from the module to the backend, while
`AttentionForwardArgs.sparse_runtime_params` carries the complete lowered state
from the backend through FMHA dispatch to `AttentionOp`.

`SparseRuntimeParams.block_sparse_inputs` is the nested carrier for optional,
algorithm-neutral `BlockSparseForwardInputs`. `Fmha.is_supported()` rejects a
request that carries routes for every library that does not declare
`supports_block_sparse_inputs`, so a dense kernel never silently ignores them;
the selected block-sparse FMHA then validates and consumes the field. `AttentionForwardArgs` defaults
the field to an empty `SparseRuntimeParams()`; the core forward always
overwrites it with the carrier prepared for the current call.

`block_sparse_attn_predict` runs even when the backend has no `SparseParams`.
Its default implementation hands through
`SparseBackendForwardArgs.block_sparse_inputs`, so an attention module that
predicts routes before the core forward only needs to place the complete
payload in `sparse_backend_args`. Algorithms that predict inside the backend
override the hook, read `metadata` for the batch layout and `forward_args` for
per-call state such as `timestep`, and return `None` for dense phases.

The core contract owns this runtime transport and general block-sparse FMHA
execution. Algorithm integrations own their prediction policy, effective Q/K/V
preparation, and any post-processing around the normal core forward.

Different KV heads are allowed to emit different sparse index sets; Q
heads that map to the same KV head share the KV head's sparse pattern.

Algorithm implementations live under
`tensorrt_llm/_torch/attention/backends/sparse/`:

- `rocket/` — RocketKV backend, metadata, cache manager, parameters, and kernels.
- `dsa/` — DSA backend, indexer, metadata, cache manager, parameters, custom ops, and kernels.
- `deepseek_v4/` — DeepSeek-V4 backend, indexer, metadata, cache manager,
  parameters, module hooks, and index conversion kernels.
- `minimax_m3/` — MiniMax-M3 Triton and packaged block-sparse backends,
  indexer implementations, metadata, and `KVCacheManagerV2` integration.
- `skip_softmax/` — SkipSoftmax parameter parsing and runtime scheduler.
- `hooks.py` — typed MLA/Attention module adapters and common backend
  prediction orchestration.
- `registry.py` — backend, metadata, and cache-manager dispatch helpers.

### AttentionOp behavior

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/media/sparse_attention_op.png" alt="Sparse attention operator workflow in TensorRT LLM" width="800">
</figure>
</div>
<p align="center"><sub><em>Figure 2: Sparse attention operator workflow in TensorRT LLM.</em></sub></p>

For page-sparse MHA/MQA/GQA, the op runs `gatherKvPageOffsetsKernel`
before the generation-phase attention kernel. It takes the (potentially
unordered or finer-grained) sparse indices and maps them to ordered,
page-aligned KV cache offsets, also producing an updated per-head
effective KV length. The downstream attention kernel reads only those
pages.

Token-sparse MQA/GQA uses physical KV-cache token indices directly. It
supports packed context and generation computation, including a linear
sequence of draft tokens. Query heads in the same KV group share the KV
head's per-query token list.

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
offsets, not request-local logical positions. MLA does not support the shared
`sparse_kv_indices` in-place compaction path. DeepSeek-V4's model-native
compressed-history pools use a separate cache path.

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

The four steps below describe the hook-based `AttentionOp` integration
path. A dedicated backend reuses the configuration, auxiliary-memory,
and registration steps but owns its prediction and sparse computation
contracts. The order matches the natural development flow — config
first, then prediction, then memory, then registration.

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

Create a new backend class inheriting from `TrtllmAttention` in
`tensorrt_llm/_torch/attention/backends/sparse/`. Override one or more of the
three prediction methods. A `VanillaAttention` implementation instead overrides
`_single_request_sparse_kv_predict` and
`_single_request_sparse_attn_predict` with its per-request Python contract.

**`sparse_kv_predict(self, q, k, metadata, forward_args)`**

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

**`sparse_attn_predict(self, q, k, metadata, forward_args)`**

- **Behavior**: return the sparse indices used by attention computation in
  the context phase, generation phase, or both, as supported by the backend.
- **Outputs**:
  - `sparse_attn_indices`: backend-specific sparse token or block indices.
    Token-sparse MQA/GQA uses shape
    `(nKvHeads, nQueryTokens, topK)` with physical KV-pool token indices
    and no offsets. Page-sparse attention uses request-local block indices;
    the algorithm declares their block size through
    `sparse_attn_indices_block_size`.
  - `sparse_attn_offsets`: optional and backend-specific. RocketKV uses
    `(numGenerations + 1)` request boundaries for its flattened page
    selections. Token-sparse MQA/GQA and DSA leave it unset. DeepSeek-V4
    uses the field for secondary compressed-pool indices.
- **Constraint**: token-sparse MQA/GQA and page-sparse MHA/MQA/GQA use
  different index layouts. Match the selected kernel contract; do not
  pass request-local block indices to the physical-token path.

**`block_sparse_attn_predict(self, q, k, v, metadata, forward_args)`**

- **Behavior**: return the `BlockSparseForwardInputs` consumed by the
  general block-sparse FMHA, or `None` for a dense call.
- **Outputs**: block geometry plus exactly one route representation
  (BSR `block_indptr`/`block_indices` or a packed `exact_block_bits`
  bitmask), optional K/V summaries for proxy routes, and optional
  `kv_valid_bits` masking ragged KV tails.
- **Default**: hands through `SparseBackendForwardArgs.block_sparse_inputs`,
  so modules that predict before the core forward do not override it.
  Override it to predict inside the backend from the flattened Q/K/V,
  the batch layout in `metadata`, and per-call state in `forward_args`.

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

- Register the new config and backend in
  `tensorrt_llm/_torch/attention/backends/sparse/registry.py`. Update executor
  wiring only when the algorithm requires behavior beyond the registry's
  generic dispatch.
- If the algorithm customizes module-layer behavior, implement and register a
  concrete `MLASparseHooks` or `AttentionSparseHooks` adapter from the
  algorithm's `module.py`.
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
- A switch inside the attention backend, such as
  `_torch/attention/backends/trtllm.py` or an implementation under
  `_torch/attention/backends/fmha/`, that reads the lowered params and enables
  the kernel-side fast path.

Skip Softmax Attention follows this pattern — see the
[BLASST tech blog](../blogs/tech_blog/blog16_Accelerating_Long_Context_Inference_with_Skip_Softmax_Attention.md)
for the kernel-side specifics.

## Roadmap

- **Dynamic eviction in generation phase** — exploring block-level
  eviction as a compromise that keeps KV cache flexibility manageable.
- **Unified auxiliary memory management** — let custom auxiliary pools
  inherit KV-cache features (reuse, offloading) by default.
