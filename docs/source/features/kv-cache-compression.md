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
the retained representation. TensorRT-LLM already provides mature paths for
applying these techniques during model forward computation. Active KV-cache
quantization stores and processes KV at lower precision, while Sparse Attention
can evict or select tokens and skip low-contribution work during prefill or
generation. Depending on the method and workload, these techniques can reduce
Attention work, cache footprint, data movement, and recomputation while
increasing effective cache capacity. Lossy methods may trade some output
quality for those savings, so their accuracy and quality impact must be
evaluated for the target workload.

Beyond these forward-pass paths, an LLM serving system creates additional
opportunities for KV cache compression at stable boundaries between model
forward steps or when KV Pages move across cache tiers. This page introduces
the KV cache compression framework for these lifecycle points, enabling
system-level co-design across storage, transfer, and execution.
Compression can be co-designed with the KV-cache storage hierarchy,
data-transfer path, and inference lifecycle to optimize storage capacity, data
movement, and computation together. This includes storage-aware compressed
layouts, compression placed on transfer paths, and fused or co-optimized
compression and transfer operations.

The framework organizes its integration points along this lifecycle dimension.
A compression method observes the current KV state at an appropriate boundary,
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
Compression method reads the current KV-cache state
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
Continue inference through the existing cache-management and Attention paths
```

Depending on the method, this design can reduce KV-cache storage, transfer
bytes, Attention work, or a combination of them without adding
compression-specific code to the model or Attention kernel. A method can use
its own scoring or transform kernels. Compression can affect accuracy and output
quality; the exact trade-off depends on the method, its settings, and the
workload, and must be validated before deployment.

Together with this framework, TensorRT-LLM exposes three related but distinct
KV-cache optimization paths.

Active KV-cache management (configured through `KvCacheConfig`) controls cache
capacity, levels, reuse, offloading, Page lifetime, and the active KV dtype.
Selecting a lower-precision dtype enables active KV-cache quantization, which
keeps the GPU KV cache in that format and reads or writes it as part of each
forward step; see [Quantization](quantization.md).

The [Sparse Attention](sparse-attention.md) framework (configured through
`SparseAttentionConfig`) supports token eviction, token selection, and masking
or skipping low-contribution work. These operations run within prefill or
generation forward computation and change which KV entries Attention retains or
processes.

The KV cache compression framework (configured through
`KvCacheCompressionConfig`) selects a compression method and its
algorithm-specific policy. Its methods run at stable cache-lifecycle boundaries
outside model forward computation, such as between forward steps or when a Page
moves across cache tiers. These paths select distinct execution flows. A
concrete compression method must understand the cache layout it transforms; it
can preserve unsupported or non-Attention state losslessly, or reject a layout
that it cannot handle. This page focuses on the KV cache compression framework.

## When Compression Runs

Unlike Sparse Attention and active KV-cache quantization, which run during model
forward computation, the KV cache compression framework unifies methods that
run at stable cache-lifecycle boundaries outside model forward computation. It
currently supports two integration models:
iteration-driven methods run between model iterations, while storage-bound
methods run when a Page moves across a hot/cold representation boundary. A
method can use one or both integration models and implements only the stages it
needs.

### Iteration-Driven Methods

Iteration-driven methods run between model forward steps. After prefill builds
the initial KV state, compression can run before generation begins or between
successive generation iterations. Each later forward step consumes the updated
KV state. TriAttention follows this flow and applies budget-triggered token
eviction periodically during generation.

### Storage-Bound Methods

Storage-bound methods run when KV Pages move across cache tiers. During
offloading, a hot GPU Page is encoded into a compressed representation as it
moves to Host or Disk storage. During onboarding, the cold Page is transferred
back and decoded into the runtime GPU representation before it is reused.
The cache manager continues to manage Page migration and storage, while the
compression method defines the representation transform. For extension APIs
and ownership rules, see the
[KV Cache Compression Development Guide](../developer-guide/kv-cache-compression-development.md).

## Compression Methods

TensorRT-LLM currently supports compression methods through both lifecycle
integration models. Only one KV cache compression method can be enabled for
each LLM instance.

| Method | When it runs | What it changes | Primary benefit |
| --- | --- | --- | --- |
| Cold-page quantization (NVFP4) | When Pages move between the GPU and a Host or Disk cache tier | The stored representation of cold Attention KV | More KV Pages per cold-tier byte and fewer bytes transferred |
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
structures. Both share the same general platform requirements.[^general-requirements]

| Cache structure | NVFP4 cold-page quantization[^cold-page-requirements] | TriAttention[^triattention-requirements] |
| --- | --- | --- |
| MHA Attention KV | Supported | Supported |
| MQA Attention KV | Supported | Supported |
| GQA Attention KV | Supported | Supported |
| MLA Attention KV | Supported | Not supported |
| GDN, SSM, and Conv state | Skipped by quantization and preserved losslessly | Not supported |
| DSA and other Attention side buffers | Preserved losslessly | Not supported |
| DeepSeek-V4 specialized sparse cache | Not supported | Not supported |

[^general-requirements]: Both methods currently require the PyTorch backend,
    KVCM V2, and an NVIDIA GPU with compute capability SM100 or SM103.
[^cold-page-requirements]: NVFP4 cold-page quantization additionally requires
    the native C++ KVCM V2 backend and a nonzero Host or Disk cache. See the
    [NVFP4 cold-page compression example](source:examples/kv_cache_compression/nvfp4_cold_page.md)
    for its remaining requirements and validated modes.
[^triattention-requirements]: TriAttention requires a model-specific offline
    calibration file. See the
    [detailed TriAttention example](source:examples/kv_cache_compression/triattention.md)
    for its remaining requirements and validated modes.

### Tested Models

NVFP4 cold-page quantization has been tested with the following model families:

- Qwen3 family
- Qwen3.5 family
- GLM family, including GLM-5.2
- DeepSeek-R1 family

TriAttention has been tested with the following model families:

- Qwen3 family
- GPT-OSS family
- Llama 3 family

These are tested-model lists, not exhaustive support lists. Other models that
use a supported KV-cache structure are expected to work; consult the method
examples for method-specific requirements.

## Further Reading

- [NVFP4 Cold-Page KV-Cache Compression example](source:examples/kv_cache_compression/nvfp4_cold_page.md)
- [TriAttention KV-Cache Compression example](source:examples/kv_cache_compression/triattention.md)
- [KV Cache Compression Development Guide](../developer-guide/kv-cache-compression-development.md)
- [KVCacheManagerV2 Cold-Page Codec Design](../developer-guide/kv-cache-cold-page-codec.md)
- [KV Cache System](kvcache.md)
- [Sparse Attention](sparse-attention.md)
