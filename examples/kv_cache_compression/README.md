<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KV Cache Compression

TensorRT-LLM KV-cache compression methods reduce the storage required for
reusable KV cache. They run alongside KVCacheManagerV2 while preserving its
ownership of Pages, cache levels, migration, and reuse. Currently, only one KV
cache compression method can be enabled for each LLM instance.

| Method | Use it to | Example |
| --- | --- | --- |
| Cold-page quantization | Store supported Attention KV in NVFP4 while its Pages reside in Host or Disk cache | [NVFP4 cold-page compression](nvfp4_cold_page.md) |
| TriAttention | Periodically evict lower-importance generation tokens and compact the GPU KV cache | [TriAttention](triattention.md) |

For a feature overview and configuration guidance, see
[KV Cache Compression](https://nvidia.github.io/TensorRT-LLM/features/kv-cache-compression.html).
To implement another compression method, see the
[KV Cache Compression Development Guide](https://nvidia.github.io/TensorRT-LLM/developer-guide/kv-cache-compression-development.html).
