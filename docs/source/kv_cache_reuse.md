# KV cache reuse

This document describes how kv cache pages can be shared and reused by requests that start with the same prompt. This can greatly lower first token latency, the time it takes before the first output token is generated. Many use cases can benefit from this, including multi-turn requests and system prompts.

## How to enable kv cache reuse

There are two steps to enabling kv cache reuse.

1. Model must support it

KV cache reuse requires the model to be built for paged context attention. This is done with `trtllm-build`:

```trtllm-build --use_paged_context_fmha enable```

2. KV cache reuse must be enabled in KVCacheManager

If you are running gptManagerBenchmark application, you can enable kv cache reuse with a command-line switch:

```gptManagerBenchmark --enable_kv_cache_reuse enable```

If you are running a Triton server, you can enable kv cache reuse with a parameter:

```
parameters: {
  key: "enable_kv_cache_reuse"
  value: {
    string_value: "true"
  }
}
```

If you are writing your own application using Executor API, you can enable kv cache reuse by including `enableBlockReuse=true` when you create the `KvCacheConfig` object.

GptManager API has been deprecated, but if you have an old application that is using GptManager API, you can enable kv cache reuse with an optional parameter:

* `TrtGptModelOptionalParams` class encapsulates the following fields:
  - `kvCacheConfig`
    - `enableBlockReuse` (default: `false`) allow reuse of previously computed KV cache blocks across requests. This is expected to optimize memory use and computation.

GptSession is scheduled to be obsoleted and does not support kv cache reuse.

## Performance expectations

KV cache state can be reused when two requests start with the same partial prompt. This reduces first token latency, the time it takes until the first output token is generated. Bigger savings are realized when the shared prompt is longer, relative to the overall prompt length. The biggest saving is realized when two identical requests are run back-to-back, in which case the latency for the first output token approaches latency for subsequent tokens.

## Situations that can prevent kv cache reuse

There are a few pitfalls that can prevent kv cache reuse when that seems possible. KV cache state only becomes reusable after the request that computed the state terminates. If you have a shared system prompt, the first request will compute kv cache state for the system prompt, the second request will reuse it, but only if the second request launches after the first request completed. If you run with a large batch-size, it is likely that many requests that share a common system prompt will be launched before the first request has terminated. No reuse will occur until one of the requests terminate, then subsequently scheduled requests can reuse.

Kv cache state for system prompts will remain reusable until memory is needed for launching a new request or propagating an existing one. When this happens, reusable blocks are evicted based on LRU. System prompts that are frequently used have a better chance of remaining reusable, but there is no guarantee since launching new requests take priority over possible reuse. Running with a larger batch size, or larger output sequence lengths for example will reduce the probability of kv cache blocks being reused, since it increases memory needs.

KV cache state is stored in blocks, each block holds multiple tokens. Only full blocks can be shared by multiple requests, thus the block size matters. The block size is a trade-off, larger block size may improve efficiency of compute kernels, but it reduces the likelihood of kv cache state reuse. The block defaults to 128 tokens, this can be changed when the model is built with the trtllm-build command, for example

```trtllm-build --tokens_per_block 32 ...```

will create a model where one KV cache block can hold 32 tokens. Note that tokens_per_block must be a power of 2.

## Offloading to host memory

Offloading to host memory increases likelihood of kv cache reuse. Reusable blocks that are needed for higher priority tasks, like propagating an already running request, are copied to a buffer in host memory instead of being evicted. This greatly extends the amount of memory available for reuse, allowing blocks to remain reusable much longer. On the other hand, offloading of blocks (and subsequent onboarding when a block is reused) has some cost since the blocks must be copied from CPU to GPU memory and vice versa. This cost is negligible on Grace-Hopper machines, and small enough to yield a net benefit for many use cases on x86 machines with Hopper GPUs. Offloading is unlikely to yield benefits on older architectures because of the (relatively) slow link between GPU and host memory.

If you are running gptManagerBenchmark, you can enable offloading with a command-line switch. For example,

```gptManagerBenchmark --kv_host_cache_bytes 45000000000```

will create a 45 GiB offloading buffer in host memory. Note that this buffer is pinned memory, allocating a lot of pinned memory on x86 machines can take a substantial amount of time (10s of seconds). This is a one-time cost.

If you are running a Triton server, you can enable offloading to host memory with the kv_cache_host_memory_bytes parameter. For example, adding this to your model config file will create a 45 GiB offloading buffer in host memory.

```
parameters: {
  key: "kv_cache_host_memory_bytes"
  value: {
    string_value: "45000000000"
  }
}
```

If you are writing your own application using Executor API, you can enable offloading to host by including `hostCacheSize=45000000000` when you create the `KvCacheConfig` object. This will create a 45 GiB offloading buffer in host memory.

GptManager API has been deprecated, but if you have an existing application that is using GptManager API, you can enable offloading with an optional parameter:

* `TrtGptModelOptionalParams` class encapsulates the following fields:
  - `kvCacheConfig`
    - `hostCacheSize` (default: `0`) size in bytes of host buffer used to offload kv cache pages upon eviction from gpu memory.

GptSession is scheduled to be obsoleted and does not support kv cache block offloading.
