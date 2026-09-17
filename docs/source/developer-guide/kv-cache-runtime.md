# KV Cache Runtime Developer Guide

## Scope

This guide covers the `_torch` KV-cache runtime contract:

- `tensorrt_llm/_torch/pyexecutor/resource_manager.py`
- `tensorrt_llm/_torch/pyexecutor/_util.py`
- `tensorrt_llm/_torch/pyexecutor/kv_cache_transceiver.py`
- `tensorrt_llm/_torch/disaggregation/`
- `tensorrt_llm/runtime/kv_cache_manager_v2/`
- speculative relocation and paged KV cache builder code under `cpp/tensorrt_llm/`

Use this guide when a change touches:

- KV cache layout
- paged KV cache mapping
- block or pool mapping
- chunked prefill
- speculative decoding relocation
- KV cache transfer or disaggregated serving

This guide is the storage/runtime guide paired with
`tensorrt_llm/_torch/modules/ATTENTION_DEVELOPER_GUIDE.md`.

If the problem is only module math, backend choice, or metadata fit, start with
the attention guide. If the problem changes how KV is sized, stored, indexed,
relocated, or transferred, start here.

## Glossary

| Term | Meaning |
| --- | --- |
| KV cache layout contract | The combined logical and physical contract used to store and interpret K/V cache |
| Logical KV cache layout | The model-facing interpretation of KV cache, such as real `num_kv_heads`, real `head_dim`, and real head grouping |
| Physical KV cache layout | The cache-facing representation used to size pages, pools, offsets, transfer state, and element storage such as `dtype` |
| Paged KV cache | KV cache stored in fixed-size blocks or pages rather than one dense tensor |
| Chunked prefill | Runtime feature that splits context execution into chunks while preserving full causal semantics |
| Speculative relocation | Runtime path that rewrites KV block locations after draft-token acceptance |
| Transceiver | KV transfer path for disaggregated serving |

## Architecture

Keep these roles separate:

1. cache manager owns storage
2. attention and backend consume that storage
3. speculative relocation rewrites storage location metadata
4. transceiver paths move KV between ranks or processes

One runtime path can accept a KV cache layout change while another rejects it,
because each path carries its own KV cache layout assumptions.

## 1. Core KV Cache Layout Contract

### 1.1 Logical KV cache layout vs physical KV cache layout

In the evaluated `_torch` runtime paths in this guide, logical KV cache layout
and physical KV cache layout are not represented as separate end-to-end
contracts.

Logical KV cache layout includes:

- real `num_kv_heads`
- real `head_dim`
- GQA grouping derived from those values

Physical KV cache layout includes:

- bytes per token
- bytes per block
- slot size
- pool grouping
- page-table metadata
- element storage such as `dtype`

This distinction matters because some quantities still cross both layers. In
the current stack:

- `num_kv_heads` is not storage-only
- `head_dim` is not storage-only

Any workaround that changes either value must first answer whether the change
affects logical KV cache layout, physical KV cache layout, or both.

The runtime-v2 core is buffer-oriented, while the current `_torch` bring-up
path still carries logical KV cache layout assumptions above that core.

### 1.2 What the cache manager actually owns

The cache manager owns storage, not attention semantics. Its responsibilities
include:

- page and pool allocation
- bytes-per-token accounting
- request-to-block mapping
- block reuse and free-space tracking
- page-table or block-table views for downstream consumers

Treat `get_buffers()` as one Python view of the cache, not as the universal KV
cache layout contract.

### 1.3 What the attention/backend path actually owns

Attention and its backend consume KV storage. They decide:

- how K/V are written
- how K/V are read
- how runtime `num_kv_heads` and `head_size` are interpreted
- whether cached KV is revisited in context or decode

On the dense `TRTLLM` path, this is driven by paged KV cache runtime
descriptors rather than by one high-level K/V tensor shape.

## 2. Runtime Paths

This guide distinguishes two different "default" ideas:

- **deployed default path**: the path an existing model family uses today
- **bring-up target path**: the path this guide uses as the default evaluation
  target for unsupported or new models

Those are not always the same. Many existing model families still run on:

- `TRTLLM + KVCacheManager / KVCacheManagerCpp`

For new-model evaluation and heterogeneous KV cache layout work, this guide
uses:

- `TRTLLM + KVCacheManagerV2`

as the default bring-up target unless there is a model-specific reason to stay
on `KVCacheManager`.

Section 3 gives the current support matrix for these paths.

### 2.1 Bring-up target: `TRTLLM + KVCacheManagerV2`

This is the default evaluated path for unsupported-model bring-up in this
guide.

Why it is a KV cache layout path:

- the `_torch` stack translates logical KV cache layout into a storage
  contract
- the runtime interprets that stored KV through paged KV cache descriptors
- both sides must agree on heads, bytes, and block offsets

The limiting layer here is the current `_torch` translation layer above the V2
core.

### 2.2 Chunked prefill

For non-MLA attention, chunked prefill is not a separate attention
implementation. On the V2 bring-up target, it reuses the same cache object and
the same paged-context TRTLLM path.

Why it is a KV cache layout path:

- earlier chunks are written into paged KV cache
- later chunks must read that same stored KV correctly
- execution order changes, but the KV cache layout contract does not

### 2.3 Speculative decoding

The relevant boundary here is draft-token relocation.

Why it is a KV cache layout path:

- it rewrites or reindexes existing KV after draft-token acceptance
- it must preserve the same block-level interpretation used by the write path

### 2.4 Disaggregated serving

This splits into two paths.

#### C++ transceiver path

Why it is a KV cache layout path:

- it transfers KV between executors or ranks
- sender and receiver must agree on the same stored KV cache contract

#### Python V2 transceiver path

Why it is a KV cache layout path:

- it exports storage state from the V2 cache manager
- it also carries metadata and peer-compatibility checks
- both views must describe the same KV cache layout

## 3. Feature-by-Path Support Matrix

This matrix summarizes the evaluated runtime paths in this guide today.
"Supported" means the path can carry that KV cache layout feature within its
current runtime contract.

| KV cache layout feature | Bring-up target: `TRTLLM + KVCacheManagerV2` | Chunked prefill on the non-MLA V2 bring-up target | Speculative decoding relocation | C++ transceiver path | Python V2 transceiver path |
| --- | --- | --- | --- | --- | --- |
| Uniform logical and physical KV cache layout | Supported | Supported | Supported | Supported | Supported |
| Per-layer `num_kv_heads` with aligned physical KV cache layout | Supported | Supported | Not supported | Supported | Not supported |
| Per-layer `head_dim` with aligned logical and physical KV cache layout | Not supported | Not supported | Not supported | Not supported | Not supported |
| Per-layer `dtype` as a physical KV cache layout change | Not supported | Not supported | Not supported | Not supported | Not supported |
| Separate logical and physical KV cache layout | Not supported | Not supported | Not supported | Not supported | Not supported |

## 4. Evaluating KV Cache Layout Changes

When a change touches `num_kv_heads`, `head_dim`, `dtype`, or any derived KV
cache layout, check all of these paths:

1. module construction
2. backend runtime parameters
3. QKV preprocessing and KV write path
4. cache bytes-per-token accounting
5. pool mapping or page-table construction
6. shaped-view consumers such as `get_buffers()`
7. chunked prefill
8. speculative relocation
9. KV transfer and peer routing

Do not claim support from one path alone.

## 5. KV Cache Layout Heterogeneity

This section is model-family-agnostic. It covers any runtime where different
layers use different KV cache layout.

### 5.1 Current support status

Section 3 is the current support matrix.

At the runtime-v2-core level, heterogeneous physical buffer sizing is already
part of the config model. In the evaluated runtime paths in this guide, the
unsupported rows come from the `_torch` translation layer and from
conditional runtime paths with additional KV cache layout requirements.

### 5.2 Why support boundaries differ

The support boundary differs by path because the stack is split into layers
with different responsibilities:

- the V2 core is buffer-oriented
- the `_torch` bring-up path translates model quantities into that buffer
  contract
- speculative relocation adds its own layout assumptions
- transfer paths add metadata and compatibility assumptions

So the end-to-end matrix in Section 3 is narrower than the V2 core API by
itself.

### 5.3 Extension principles

Use these rules for extensions:

1. keep logical KV cache layout and physical KV cache layout aligned unless
   the runtime explicitly models both
2. preserve one consistent byte-stride contract between cache-manager-side
   indexing and runtime-side block interpretation
3. treat speculative relocation and transceiver metadata as part of the same
   support boundary, not as separate afterthoughts
4. promote support incrementally:
   - first on the bring-up target
   - then on chunked prefill
   - then on speculative and disaggregated paths

### 5.4 Future extension boundary

This guide uses the following extension order:

1. heterogeneous KV cache layout on the TRTLLM V2 bring-up target
2. chunked prefill on that same target
3. speculative decoding relocation
4. KV transfer and transceiver metadata

A design with separate logical and physical KV cache layout requires explicit
runtime fields for both contracts across:

- logical KV cache layout used by attention
- physical KV cache layout used by allocation, relocation, and transfer

Those separate runtime fields do not exist in the evaluated paths in this
guide.

## 6. Promotion Rules

Use these support labels carefully:

- **bring-up-target support** means the evaluated path in this guide,
  `TRTLLM + KVCacheManagerV2`, is covered.
- **complete support** means the bring-up target is covered and all relevant
  conditional paths are also covered.

Do not upgrade a claim to complete support unless the relevant conditional
paths are covered:

- chunked prefill
- speculative decoding
- disaggregated serving

## 7. Testing Notes

For bring-up-target support, test:

1. monolithic prefill
2. paged KV cache decode
3. chunked prefill
4. one hard configuration per active KV cache layout
5. one test that checks cache-manager pool mapping and runtime KV cache layout agree

For complete support, also test:

6. speculative decoding with accepted draft tokens
7. C++ transceiver path
8. Python V2 transceiver path

## 8. Anti-Patterns

- Do not treat a KV cache layout change as only a memory-sizing change. It can
  also affect runtime interpretation, relocation, and transfer.
- Do not change model-facing KV quantities such as `num_kv_heads` or
  `head_dim` just to make storage look uniform, unless the runtime explicitly
  supports different model-facing and storage-facing KV cache layouts.
- Do not treat bring-up-target support as complete support. In this guide,
  complete support also includes the relevant chunked-prefill, speculative,
  and disaggregated paths.
- Do not judge a KV cache layout change from one API or one helper alone.
  Follow the full runtime path from cache write to cache read, relocation, and
  transfer.
- Do not treat a runtime-v2-core limitation and an `_torch` integration
  limitation as the same problem. Check which layer actually owns the
  restriction.

## 9. Key File Map

- `tensorrt_llm/_torch/pyexecutor/resource_manager.py`
- `tensorrt_llm/_torch/pyexecutor/_util.py`
- `tensorrt_llm/_torch/pyexecutor/kv_cache_transceiver.py`
- `tensorrt_llm/_torch/disaggregation/native/rank_info.py`
- `tensorrt_llm/_torch/disaggregation/native/mixers/attention/peer.py`
- `tensorrt_llm/_torch/disaggregation/resource/kv_extractor.py`
- `tensorrt_llm/runtime/kv_cache_manager_v2/_config.py`
- `tensorrt_llm/runtime/kv_cache_manager_v2/_storage/_config.py`
- `cpp/tensorrt_llm/thop/attentionOp.cpp`
- `cpp/tensorrt_llm/thop/parallelDecodeKVCacheUpdateOp.cpp`
- `cpp/tensorrt_llm/kernels/speculativeDecoding/kvCacheUpdateKernels.cu`
- `cpp/tensorrt_llm/kernels/unfusedAttentionKernels/unfusedAttentionKernels_2_template.h`
