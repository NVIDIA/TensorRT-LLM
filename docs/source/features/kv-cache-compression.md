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
- [Support](#support)
- [Verification](#verification)
- [Further Reading](#further-reading)

## Overview

Long-context and agentic workloads can retain large amounts of reusable KV
cache. If GPU, Host, or Disk cache capacity is exhausted, useful prefixes are
evicted and must be recomputed. KV cache compression reduces this pressure by
changing either how KV data is represented in storage or how much KV data is
retained.

TensorRT LLM keeps compression separate from cache management and Attention:

```text
KvCacheConfig
  `-- capacity, cache levels, reuse, offloading, and Page lifecycle

KvCacheCompressionConfig
  `-- compression method, representation, and algorithm-specific policy

SparseAttentionConfig
  `-- which KV entries Attention reads or skips
```

This separation lets the KV cache manager remain responsible for Pages,
allocation, migration, and publication while a compression method owns its
compression policy. In particular, cold-page quantization does not change the
GPU KV format consumed by Attention.

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

### TriAttention

TriAttention periodically scores generation KV tokens, retains the most useful
tokens, and physically compacts the cache. The prompt remains preserved, and
the model's standard Attention implementation runs over the compacted cache.

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
[TriAttention example](source:examples/kv_cache_compression/triattention.md).

## When Compression Runs

`KVCacheCompressionManager` exposes lifecycle methods for request- and
iteration-driven methods, together with cold-page encode and decode methods.
Each compression method implements only the stages it needs.

| Trigger | Manager method | Purpose |
| --- | --- | --- |
| A request enters its first prefill chunk | `on_request_init()` | Initialize request-local compression state |
| A request finishes its final prefill chunk | `on_context_step_end()` | Run optional context-bound compression |
| Before a generation forward step | `on_generation_step_begin()` | Prepare generation-step state when required |
| After a generation forward step | `on_generation_step_end()` | Run periodic or budget-triggered compression, such as TriAttention |
| A hot Page moves to a cold tier | `encode_cold_pages()` | Encode and transfer a batch of Pages to cold storage |
| A cold Page returns to the GPU | `decode_cold_pages()` | Transfer and restore a batch of Pages to the runtime layout |
| A request finishes or aborts | `on_request_finish()` | Release request-local compression state |

The KV cache manager still decides when Pages migrate and publishes their new
cache level only after the operation completes. For method signatures and
ownership rules, see the [KV Cache Compression Development Guide](../developer-guide/kv-cache-compression-development.md).

## Support

The following table summarizes the cache structures handled by NVFP4 cold-page
quantization.

| Cache structure | Compression support |
| --- | --- |
| MHA/GQA Attention KV | Supported |
| MLA Attention KV | Supported |
| GDN, SSM, and Conv state | Skipped by quantization and preserved losslessly |
| DSA and other Attention side buffers | Preserved losslessly |
| DeepSeek-V4 specialized sparse cache | Not supported |

Cold-page NVFP4 currently requires the PyTorch backend, the native C++ KV cache
manager V2, and an SM100-family GPU. Algorithm-specific requirements can differ;
refer to the corresponding example before enabling a method.

### Tested Models

Cold-page quantization has been tested with the following model families:

- Qwen3 family
- Qwen3.5 family
- GLM family, including GLM-5.2
- DeepSeek-R1 family

This is a tested-model list, not an exhaustive support list. Other models that
use a supported KV-cache structure are expected to work.

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
