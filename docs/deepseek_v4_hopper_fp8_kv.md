<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4 Hopper FP8 KV Cache with Sparse MLA

This document describes the Hopper (SM90) sparse-MLA path for
DeepSeek-V4.

## Summary

DeepSeek-V4 uses two paged attention pools:

- a sliding-window (SWA) pool; and
- a compressed pool for layers whose compression ratio is greater than one.

The Blackwell implementation consumes those pools through the absorption
FMHA path. Hopper instead uses FlashMLA:

- context and BF16 fallback decode call `flash_mla_sparse_fwd` once per
  pool and merge the results using their log-sum-exp values; and
- ratio-4 decode uses FlashMLA's native `sparse_decode_fwd` extension,
  whose MODEL1 format supports both pools in one kernel call.

FlashMLA is pinned to `15f13e5`, which includes the MODEL1 FP8 sparse
decode API and the source layout used by this integration.

## Runtime dispatch

`MLA.forward_context_sparse_mla` and
`MLA.forward_generation_sparse_mla` dispatch by GPU architecture:

```text
SM100+:
    existing DeepSeek-V4 absorption FMHA path

SM90 context:
    forward_sparse_mla_deepseek_v4_bf16

SM90 generation, ratio 4:
    forward_sparse_decode_deepseek_v4_fp8

SM90 generation, other ratios:
    forward_sparse_mla_deepseek_v4_bf16
```

The Hopper metadata path prepares the context and generation indirection
buffers consumed by `mla_rope_append_paged_kv_assign_q` and
`mla_rope_generation`. These fused helpers apply RoPE and update the
standard TRT-LLM paged cache before FlashMLA reads it.

## BF16 dual-pool path

`forward_sparse_mla_deepseek_v4_bf16` performs the following steps:

1. Apply RoPE and update the paged cache.
2. Convert SWA and compressed local positions into pool-relative
   physical token indices using the current DeepSeek-V4 block tables.
3. Gather the referenced entries and dequantize only those FP8 cache
   entries to BF16 when necessary.
4. Invoke `flash_mla_sparse_fwd` independently for SWA and compressed
   KV.
5. Merge both outputs with the natural-log LSE values returned by
   FlashMLA, then apply the attention sink once to the combined softmax.

Attention sinks, when present in the checkpoint, are applied after the
pool LSEs are merged so they participate exactly once in the global
softmax.

## MODEL1 FP8 shadow

FlashMLA's Hopper sparse-decode kernel expects MODEL1 KV data with the
following logical token contents:

```text
logical token:
  448 bytes  FP8 nope, quantized in 64-element blocks
  128 bytes  BF16 RoPE
    8 bytes  seven E8M0 scale bytes plus padding
  ---------
584 bytes total
```

The standard paged FP8 pool stores 512 FP8 bytes per token with a
per-tensor scale, so decode maintains a persistent MODEL1 shadow for
each pool. Within each cache block, FlashMLA stores all 576-byte data
regions first, followed by all 8-byte scale regions. The shadow state
consists of:

- `_fp8_shadows[attention_type]`: MODEL1 data plus one dummy block;
- `_fp8_block_fill_gpu[attention_type]`: the converted token count for
  each physical block;
- `_fp8_update_grids`: cached request/block/token grids; and
- `_fp8_offsets_cache`: cached byte offsets used by vectorized scatter.

Context invalidates the fill trackers. The first decode then repopulates
all referenced tokens, while steady-state decode converts only newly
filled token positions. Invalid grid entries write to the extra dummy
block, keeping the tensor shapes and scatter pattern stable for CUDA
graph capture.

## Key files

| File | Responsibility |
| --- | --- |
| `3rdparty/fetch_content.json` | FlashMLA revision |
| `cpp/tensorrt_llm/flash_mla/CMakeLists.txt` | FlashMLA SM90/SM100 sources |
| generated `tensorrt_llm/flash_mla/` package | sparse prefill wrapper from the pinned dependency |
| `tensorrt_llm/_torch/modules/mla.py` | Hopper dispatch and shadow conversion |
| `tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/deepseek_v4.py` | Hopper metadata preparation |
| `tensorrt_llm/_torch/attention_backend/trtllm.py` | fused MLA argument normalization |
| `tensorrt_llm/_torch/models/modeling_deepseekv4.py` | Hopper CUTLASS MXFP4 scale loading |

## Known limitation

Context uses FlashMLA's BF16 sparse-prefill kernel. With FP8 KV cache it
therefore gathers and dequantizes the selected entries into a temporary
BF16 buffer before the call. A future native FP8 sparse-prefill kernel
would remove that conversion and temporary buffer.
