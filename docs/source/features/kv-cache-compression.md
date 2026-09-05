<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KV Cache Compression

- [Overview](#overview)
- [When Compression Runs](#when-compression-runs)
  - [Iteration-Driven Methods](#iteration-driven-methods)
  - [Storage-Bound Methods](#storage-bound-methods)
- [Compression Methods](#compression-methods)
  - [Cold-Page Quantization](#cold-page-quantization)
  - [TriAttention](#triattention)
- [Support](#support)
- [Further Reading](#further-reading)

## Overview

Long-context and agentic workloads can accumulate large amounts of reusable KV
state across prompts, tool interactions, and intermediate reasoning. When the
available GPU, Host, or Disk capacity cannot retain enough of that state, the
serving system must evict and later reconstruct useful context. At scale, this
creates substantial redundant work, increases request latency, and limits the
scale and efficiency of KV-cache reuse.

Existing KV cache compression methods use a range of techniques. Token eviction
removes selected KV entries, quantization represents KV values at lower
precision, and other methods use compact encodings or transformations to reduce
the retained representation. Depending on the method and workload, these
techniques can reduce Attention work, cache footprint, data movement, and
recomputation while increasing effective cache capacity. Lossy methods may
trade some output quality for those savings, so their accuracy and quality
impact must be evaluated for the target workload.

As an LLM serving system, TensorRT-LLM KV cache compression supports both
method-level techniques, such as eviction and quantization, and system-level
co-design across storage, transfer, and execution. Compression can be
co-designed with the KV-cache storage hierarchy, data-transfer path, and
inference lifecycle to optimize storage capacity, data movement, and
computation together. This includes storage-aware compressed layouts,
compression placed on transfer paths, and fused or co-optimized compression and
transfer operations.

TensorRT-LLM organizes its integration points along this lifecycle dimension. A
compression method observes the current KV state at an appropriate boundary,
applies a method-specific transformation, and makes the resulting state
available to the existing inference path. These integration points sit outside
the Attention kernel, so compression policies do not require model-specific or
compression-specific branches in Attention. An iteration-driven implementation
can transform retained KV state after a prefill or generation step, while a
storage-bound implementation can encode Pages as they move to Host or Disk and
decode them when they return.

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
its own scoring or transform kernels. Compression can affect accuracy and output
quality; the exact trade-off depends on the method, its settings, and the
workload, and must be validated before deployment.

`KvCacheCompressionConfig` selects the compression method and its
algorithm-specific policy. It is used alongside two related but distinct
configurations: `KvCacheConfig` controls cache capacity, levels, reuse,
offloading, and Page lifetime, while `SparseAttentionConfig` controls how
Attention selects or processes KV during computation. Neither configuration
selects a compression method. A concrete compression method must understand the
cache layout it transforms; it can preserve unsupported or non-Attention state
losslessly, or reject a layout that it cannot handle.

## When Compression Runs

Compression methods use one or both integration models. Iteration-driven
methods run from PyExecutor's request and iteration lifecycle. Storage-bound
methods run only when KVCM migrates a Page across a hot/cold representation
boundary. A method implements only the integration models and stages it needs.

### Iteration-Driven Methods

PyExecutor dispatches semantic lifecycle hooks through
`KVCacheCompressionManager`. TriAttention uses the generation-end hook to run
periodic, budget-triggered token eviction.

| Trigger | Manager method | Purpose |
| --- | --- | --- |
| A request enters its first prefill chunk | `on_request_init()` | Initialize request-local compression state |
| A request finishes its final prefill chunk | `on_context_step_end()` | Run optional context-bound compression |
| Before each scheduled forward iteration | `on_generation_step_begin()` | Inspect or prepare the generation cohort when required |
| After each scheduled forward iteration and the native KVCM update | `on_generation_step_end()` | Process the generation cohort, such as periodic TriAttention eviction |
| A request finishes or aborts | `on_request_finish()` | Release request-local compression state |

### Storage-Bound Methods

KVCM calls a storage-bound codec provider only when migration changes the Page
representation. NVFP4 cold-page quantization implements these two batched
operations and does not register for per-iteration callbacks.

At the native storage boundary, KVCM invokes `IKvCacheColdPageCodec::encode()`
or `IKvCacheColdPageCodec::decode()`. A codec backed by a Python compression
provider delegates those operations to the provider hooks shown below.

| Trigger | Native codec method | Python provider hook | Purpose |
| --- | --- | --- | --- |
| A hot Page moves to a cold tier | `encode()` | `encode_cold_pages()` | Encode and transfer a batch of Pages to cold storage |
| A cold Page returns to the GPU | `decode()` | `decode_cold_pages()` | Transfer and restore a batch of Pages to the runtime layout |

KVCM still decides when Pages migrate and owns their Slots, streams, completion
ordering, rollback, and mapping publication. For method signatures and
ownership rules, see the
[KV Cache Compression Development Guide](../developer-guide/kv-cache-compression-development.md).

## Compression Methods

TensorRT-LLM currently supports compression methods through both lifecycle
integration models. Only one KV cache compression method can be enabled for
each LLM instance.

| Method | When it runs | What it changes | Primary benefit |
| --- | --- | --- | --- |
| Cold-page quantization | When Pages move between the GPU and a Host or Disk cache tier | The stored representation of cold Attention KV | More KV Pages per cold-tier byte and fewer bytes transferred |
| TriAttention | Periodically during generation | The set of KV tokens retained in the cache | Lower KV-cache memory usage and Attention work for long generation |

### Cold-Page Quantization

Cold-page quantization encodes supported Attention KV into a smaller numerical
representation while its Pages reside in Host or Disk memory. A format
implementation can reuse the quantization algorithm and optimized conversion
primitives from an existing quantization path, then combine them with Page
migration in a cold-page codec. This builds on established formats, scale
contracts, and rounding behavior instead of defining a separate numerical
format only for storage, making the accuracy trade-off easier to understand and
validate.

NVFP4 is the first supported cold-page quantization format. It stores eligible
Attention KV in NVFP4 in the cold tiers, while the GPU cache continues to use
the model's normal runtime KV type, such as FP16, BF16, or FP8.

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

## Support

The two methods share the compression framework but support different cache
structures. Platform and method requirements are noted below.[^support-requirements]

| Cache structure | NVFP4 cold-page quantization | TriAttention |
| --- | --- | --- |
| MHA Attention KV | Supported | Not supported |
| MQA Attention KV | Supported | Not supported |
| GQA Attention KV | Supported | Restricted; see the TriAttention example |
| MLA Attention KV | Supported | Not supported |
| GDN, SSM, and Conv state | Skipped by quantization and preserved losslessly | Not supported |
| DSA and other Attention side buffers | Preserved losslessly | Not supported |
| DeepSeek-V4 specialized sparse cache | Not supported | Not supported |

[^support-requirements]: Both methods currently require the PyTorch backend,
    KVCM V2, and an NVIDIA GPU with compute capability SM100 or SM103. NVFP4
    cold-page quantization additionally requires the native C++ KVCM V2 backend
    and a nonzero Host or Disk cache. TriAttention requires a model-specific
    offline calibration file and currently supports only BF16 GQA pools with
    group size 4 or 8 and the score geometry listed in its detailed example.
    See each method's example for its remaining requirements and validated
    modes.

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

## Further Reading

- [NVFP4 Cold-Page KV-Cache Compression example](source:examples/kv_cache_compression/nvfp4_cold_page.md)
- [TriAttention KV-Cache Compression example](source:examples/kv_cache_compression/triattention.md)
- [KV Cache Compression Development Guide](../developer-guide/kv-cache-compression-development.md)
- [KVCacheManagerV2 Cold-Page Codec Design](../developer-guide/kv-cache-cold-page-codec.md)
- [KV Cache System](kvcache.md)
- [Sparse Attention](sparse-attention.md)
