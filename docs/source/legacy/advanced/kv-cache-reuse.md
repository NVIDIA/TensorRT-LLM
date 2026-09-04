(kv-cache-reuse)=

# KV cache reuse

```{caution}
This page describes **prefix KV cache reuse** concepts. The TensorRT engine
build path (`trtllm-build`, `gptManagerBenchmark`) was removed; configure reuse
with the PyTorch / LLM API `KvCacheConfig` (or `trtllm-serve` YAML). See
[KV Cache System](../../features/kvcache.md),
[How to Change KV Cache Behavior](../../examples/kvcacheconfig.md), and
[TensorRT Backend Removal](../tensorrt-backend-removal.md).
```

This document describes how kv cache pages can be shared and reused by requests that start with the same prompt. This can greatly lower first token latency, the time it takes before the first output token is generated. Many use cases can benefit from this, including multi-turn requests and system prompts.

## How to enable kv cache reuse

On the PyTorch backend, KV cache block reuse is **enabled by default**. Control it through `KvCacheConfig` when constructing `LLM`, or via the equivalent YAML under `kv_cache_config` for `trtllm-serve` / `trtllm-bench` / `trtllm-eval`.

Python:

```python
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig

# Reuse is on by default; set enable_block_reuse=False to disable.
kv_cache_config = KvCacheConfig(enable_block_reuse=True)
llm = LLM(model="<model>", kv_cache_config=kv_cache_config)
```

`trtllm-serve` YAML:

```yaml
kv_cache_config:
  enable_block_reuse: true
```

The legacy `trtllm-build --use_paged_context_fmha` step and
`gptManagerBenchmark --enable_kv_cache_reuse` / Triton
`enable_kv_cache_reuse` knobs applied to the removed TensorRT engine path and
are no longer used.

### Enable kv cache reuse for p-tuning

When using p-tuning, different requests may use same fake input ids (i.e. prompt ids whose values are larger than vocabulary size). That may lead to incorrect kv cache reuse, since TRT-LLM could not distinguish these requests only by input ids. To enable kv cache reuse for p-tuning correctly, users should provide an extra id (uint64) for each input id. Extra ids for normal input ids (i.e. text token ids) should always be 0, while fake input ids should have extra ids which are larger than 0. Requests using same prompt embeddings should use same extra ids, while requests using different prompt embeddings should use different extra ids.

Example:
Assume vocabulary size is 100, which means normal text token ids are in range [0, 99] and prompt ids start from 100.

```python
# Request 1 uses prompt embedding table 1
input_ids = [100, 101, 102, 103, 1, 2, 3, 4]
extra_ids = [1,   1,   1,   1,   0, 0, 0, 0]

# Request 2 uses prompt embedding table 2
input_ids = [100, 101, 102, 103, 1, 2, 3, 4]
extra_ids = [2,   2,   2,   2,   0, 0, 0, 0]

# Request 3 uses prompt embedding table 1 and different text tokens
input_ids = [100, 101, 102, 103, 5, 6, 7, 8]
extra_ids = [1,   1,   1,   1,   0, 0, 0, 0]
```

## Performance expectations

KV cache state can be reused when two requests start with the same partial prompt. This reduces first token latency, the time it takes until the first output token is generated. Bigger savings are realized when the shared prompt is longer, relative to the overall prompt length. The biggest saving is realized when two identical requests are run back-to-back, in which case the latency for the first output token approaches latency for subsequent tokens.

## Situations that can prevent kv cache reuse

There are a few pitfalls that can prevent kv cache reuse when that seems possible. KV cache state only becomes reusable after the request that computed the state terminates. If you have a shared system prompt, the first request will compute kv cache state for the system prompt, the second request will reuse it, but only if the second request launches after the first request completed. If you run with a large batch-size, it is likely that many requests that share a common system prompt will be launched before the first request has terminated. No reuse will occur until one of the requests terminate, then subsequently scheduled requests can reuse.

Kv cache state for system prompts will remain reusable until memory is needed for launching a new request or propagating an existing one. When this happens, reusable blocks are evicted based on LRU. System prompts that are frequently used have a better chance of remaining reusable, but there is no guarantee since launching new requests take priority over possible reuse. Running with a larger batch size, or larger output sequence lengths for example will reduce the probability of kv cache blocks being reused, since it increases memory needs.

KV cache state is stored in blocks, each block holds multiple tokens. Only full blocks can be shared by multiple requests, thus the block size matters. Partially matched blocks can also be reused, but that creates a new copy of the block for each sequence. The block size is a trade-off, larger block size may improve efficiency of compute kernels, but it reduces the likelihood of kv cache state reuse. The default is `tokens_per_block=32` on `KvCacheConfig` (must be a power of 2). Set it when constructing the engine, for example:

```python
kv_cache_config = KvCacheConfig(tokens_per_block=32)
```

or in YAML:

```yaml
kv_cache_config:
  tokens_per_block: 32
```

Partial reuse is controlled by `enable_partial_reuse` / `copy_on_partial_reuse`
on the same config (both default to `true`).

## Offloading to host memory

Offloading to host memory increases likelihood of kv cache reuse. Reusable blocks that are needed for higher priority tasks, like propagating an already running request, are copied to a buffer in host memory instead of being evicted. This greatly extends the amount of memory available for reuse, allowing blocks to remain reusable much longer. On the other hand, offloading of blocks (and subsequent onboarding when a block is reused) has some cost since the blocks must be copied from CPU to GPU memory and vice versa. This cost is negligible on Grace-Hopper machines, and small enough to yield a net benefit for many use cases on x86 machines with Hopper GPUs. Offloading is unlikely to yield benefits on older architectures because of the (relatively) slow link between GPU and host memory.

Set `KvCacheConfig.host_cache_size` to the desired host buffer size in bytes
(for example `45000000000` for ~45 GiB). On x86, large pinned allocations can
take tens of seconds once at startup.

```python
kv_cache_config = KvCacheConfig(host_cache_size=45000000000)
```

```yaml
kv_cache_config:
  host_cache_size: 45000000000
```

The legacy `gptManagerBenchmark --kv_host_cache_bytes`, Triton
`kv_cache_host_memory_bytes`, and Executor `hostCacheSize` knobs mapped to the
removed TensorRT engine path; use `host_cache_size` on `KvCacheConfig` instead.