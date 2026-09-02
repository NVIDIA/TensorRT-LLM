<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KV Cache Compression

- [Overview](#overview)
- [Compression Methods](#compression-methods)
  - [Cold-Page Quantization](#cold-page-quantization)
  - [TriAttention](#triattention)
- [When Compression Runs](#when-compression-runs)
  - [Iteration-Driven Methods](#iteration-driven-methods)
  - [Storage-Bound Methods](#storage-bound-methods)
- [Support](#support)
- [Verification](#verification)
- [Further Reading](#further-reading)

## Overview

Long-context and agentic workloads can retain large amounts of reusable KV
cache. If GPU, Host, or Disk cache capacity is exhausted, useful prefixes are
evicted and must be recomputed. KV cache compression reduces this pressure by
changing either how KV data is represented in storage or how much KV data is
retained.

Compression runs at well-defined safe points outside the Attention kernel. For
example, a method can compress the cache after a prefill or generation step, or
when KVCM moves a Page from the GPU's hot layout into the cold representation
used by Host or Disk, or restores it to the GPU. The method uses the current
cache state either to reduce what is retained or to encode and decode a compact
Page representation.

```text
Prefill, generation, or a hot/cold Page transition reaches a safe boundary
  |
  v
Compression manager reads the current KV-cache state
  |
  +-- iteration-driven method: reduce or compact the retained KV state
  |
  `-- storage-bound method: encode hot Pages into a compact cold format and
                            decode them when they return
  |
  v
Use the resulting smaller cache for storage, transfer, or later inference
  |
  v
Continue inference through the existing KVCM and Attention interfaces
```

Depending on the method, this design can reduce KV-cache storage, transfer
bytes, Attention work, or a combination of them without adding
compression-specific code to the model or Attention kernel. A method can use
its own scoring or transform kernels. Each method is designed to preserve useful
KV information while minimizing its impact on accuracy and output quality. The
exact trade-off depends on the method, its settings, and the workload, and
should be validated before deployment.

`KvCacheConfig` continues to control cache capacity, levels, reuse, offloading,
and Page lifetime. `KvCacheCompressionConfig` selects how KV is compressed at
the supported boundaries. A concrete method must understand the cache layout it
transforms; it can preserve unsupported or non-Attention state losslessly, or
reject a layout that it cannot handle. `SparseAttentionConfig` is orthogonal: it
changes how Attention selects or processes KV during computation.

Currently, only one KV cache compression method can be enabled for each LLM
instance.

## Compression Methods

| Method | When it runs | What it changes | Primary benefit |
| --- | --- | --- | --- |
| Cold-page quantization | When Pages move between the GPU and a Host or Disk cache tier | The stored representation of cold Attention KV | More KV Pages per cold-tier byte and fewer bytes transferred |
| TriAttention | Periodically during generation | The set of KV tokens retained in the cache | Lower KV-cache memory usage and Attention work for long generation |

### Cold-Page Quantization

Cold-page quantization stores supported Attention KV buffers in NVFP4 while
their Pages reside in Host or Disk memory. The GPU cache continues to use the
model's normal runtime KV type, such as FP16, BF16, or FP8.

```text
GPU hot Page (runtime KV type)
  -- encode and offload --> Host/Disk cold Page (NVFP4)
  <-- onboard and decode --
GPU hot Page (runtime KV type)
```

As a result, the active Attention implementation does not need to consume the
cold representation. Page identity, token identity, block reuse, and the
Attention-visible GPU layout remain unchanged. The benefit is
workload-dependent: it is largest when cold-tier capacity or Page migration is
a bottleneck.

Enable the feature with the C++ KV cache manager V2 and a nonzero Host or Disk
cache:

```python
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import (
    ColdPageQuantizationCompressionConfig,
    KvCacheConfig,
)

llm = LLM(
    model="<path_or_hf_id>",
    backend="pytorch",
    kv_cache_config=KvCacheConfig(
        use_kv_cache_manager_v2=True,
        host_cache_size=8 << 30,
    ),
    kv_cache_compression_config=ColdPageQuantizationCompressionConfig(
        quant="nvfp4",
    ),
)
```

Cold-page NVFP4 is different from an active NVFP4 KV cache. The former stores
NVFP4 only in a cold cache tier and restores the runtime type before Attention;
the latter sets `KvCacheConfig(dtype="nvfp4")` and keeps active GPU KV in
NVFP4. See [Quantization](quantization.md) for active KV-cache quantization.
For complete single-GPU and disaggregated-serving configurations, see the
[NVFP4 cold-page compression example](source:examples/kv_cache_compression/nvfp4_cold_page.md).

### TriAttention

[TriAttention](https://arxiv.org/abs/2604.04921) (ICML 2026) periodically scores
generation KV tokens, retains the most useful tokens, and physically compacts
the cache. The prompt remains preserved, and the model's standard Attention
implementation runs over the compacted cache.

TriAttention requires an offline calibration file. A minimal configuration is:

```python
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import (
    KvCacheConfig,
    TriAttentionKvCacheCompressionConfig,
)

llm = LLM(
    model="<path_or_hf_id>",
    backend="pytorch",
    kv_cache_config=KvCacheConfig(use_kv_cache_manager_v2=True),
    kv_cache_compression_config=TriAttentionKvCacheCompressionConfig(
        budget=2048,
        beta=128,
        eviction_mode="union",
        calibration_path="/path/to/model-calibration.pt",
    ),
)
```

For calibration, configuration parameters, and current requirements, see the
[detailed TriAttention example](source:examples/kv_cache_compression/triattention.md).

## When Compression Runs

Compression methods use one of two execution models. Iteration-driven methods
run from PyExecutor's request and iteration lifecycle. Storage-bound methods
run only when KVCM migrates a Page across a hot/cold representation boundary.
A method implements only the model and stages it needs.

### Iteration-Driven Methods

PyExecutor dispatches semantic lifecycle hooks through
`KVCacheCompressionManager`. TriAttention uses the generation-end hook to run
periodic, budget-triggered token eviction.

| Trigger | Manager method | Purpose |
| --- | --- | --- |
| A request enters its first prefill chunk | `on_request_init()` | Initialize request-local compression state |
| A request finishes its final prefill chunk | `on_context_step_end()` | Run optional context-bound compression |
| Before a generation forward step | `on_generation_step_begin()` | Prepare generation-step state when required |
| After a generation forward step | `on_generation_step_end()` | Run periodic or budget-triggered compression, such as TriAttention |
| A request finishes or aborts | `on_request_finish()` | Release request-local compression state |

### Storage-Bound Methods

KVCM calls a storage-bound codec provider only when migration changes the Page
representation. NVFP4 cold-page quantization implements these two batched
operations and does not register for per-iteration callbacks.

| Trigger | Codec-provider method | Purpose |
| --- | --- | --- |
| A hot Page moves to a cold tier | `encode_cold_pages()` | Encode and transfer a batch of Pages to cold storage |
| A cold Page returns to the GPU | `decode_cold_pages()` | Transfer and restore a batch of Pages to the runtime layout |

KVCM still decides when Pages migrate and owns their Slots, streams, events,
rollback, and mapping publication. It publishes the new cache level only after
the codec operation is safely enqueued. For method signatures and ownership
rules, see the [KV Cache Compression Development Guide](../developer-guide/kv-cache-compression-development.md).

## Support

The two methods share the compression framework but support different cache
structures.

| Cache structure | NVFP4 cold-page quantization | TriAttention |
| --- | --- | --- |
| MHA/GQA Attention KV | Supported | Supported for full-Attention KV |
| MLA Attention KV | Supported | Not supported |
| GDN, SSM, and Conv state | Skipped by quantization and preserved losslessly | Not supported |
| DSA and other Attention side buffers | Preserved losslessly | Not supported |
| DeepSeek-V4 specialized sparse cache | Not supported | Not supported |

\* Both methods currently require the PyTorch backend, KVCM V2, and an
SM100-family GPU (SM100 or SM103). NVFP4 cold-page quantization additionally
requires the native C++ KVCM V2 backend and a nonzero Host or Disk cache.
TriAttention requires a model-specific offline calibration file. See each
method's detailed example for its remaining requirements and validated modes.

### Tested Models

NVFP4 cold-page quantization has been tested with the following model families:

- Qwen3 family
- Qwen3.5 family
- GLM family, including GLM-5.2
- DeepSeek-R1 family

TriAttention has been tested with Qwen3-8B. These are tested-model lists, not
exhaustive support lists. Other models that use a supported KV-cache structure
are expected to work; consult the method example for method-specific
restrictions.

## Verification

Configuring a compression method does not by itself prove that compression ran.
Cold-page quantization is exercised only when the workload creates real
GPU-to-cold and cold-to-GPU Page migration. Verify that a Host or Disk cache is
enabled and that the workload produces offload and reuse. For a hybrid model,
seeing Attention KV compressed while GDN, SSM, or Conv state remains lossless is
expected behavior.

## Further Reading

- [NVFP4 Cold-Page KV-Cache Compression example](source:examples/kv_cache_compression/nvfp4_cold_page.md)
- [TriAttention KV-Cache Compression example](source:examples/kv_cache_compression/triattention.md)
- [KV Cache Compression Development Guide](../developer-guide/kv-cache-compression-development.md)
- [KVCacheManagerV2 Cold-Page Codec Design](../developer-guide/kv-cache-cold-page-codec.md)
- [KV Cache System](kvcache.md)
- [Sparse Attention](sparse-attention.md)
