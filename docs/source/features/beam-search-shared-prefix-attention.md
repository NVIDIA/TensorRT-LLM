<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Shared-Prefix Attention for Wide Beam Search (Prototype)

This feature targets **wide beam search**: `beam_width` in the tens or more in `SamplingParams`
(see [Sampling](sampling.md)), with long prompts, as in semantic-ID recommendation. Every beam of
a request starts from the same prompt, but during decode the attention kernel still reads the
prompt's KV cache from GPU memory once per beam at every step. At wide beams these redundant
reads dominate decode time. The faster decode kernel (XQA) also rejects batches where the query
heads per KV head times the beam width exceeds 32, so wide beams fall back to the slowest decode
kernel.

**Shared-prefix attention** reads the shared prompt once per request and attends each beam's own
generated tokens separately, through the beam search's cache indirection. It then combines the
two partial results exactly with the log-sum-exp. Prefill is unaffected.

This feature is a prototype: its settings may change.

## Support

- Validated on H100 (SM90) and GB200 (SM100) with the default `fa2` prefix kernel. Other GPUs are
  untested; the FA4 backend (below) is SM90 only and declines on other architectures.
- PyTorch backend, KV cache manager V2 (`use_kv_cache_manager_v2: true`). With the legacy
  manager, planning raises an error instead of falling back.
- bf16 or fp16 KV cache, head size 64, 128 or 256, no sliding window, attention sinks or custom
  masks, no speculative decoding, no MLA.
- Models whose attention module applies the position embedding itself (for example Qwen3).
  Layers that need the attention backend to apply RoPE, ALiBi or a relative bias are not served
  and silently use the regular attention.
- CUDA graphs. A step that also carries context requests runs the context phase on the regular
  attention and the generation rows on this library.
- Batches or layers the library does not support use the regular attention automatically. Two
  cases raise an error instead: the legacy KV cache manager, and a beam that generates more than
  `TLLM_BEAM_SHARED_PREFIX_MAX_TAIL` tokens under CUDA graphs.

## Usage

Set the environment variables before starting `trtllm-serve` or creating the `LLM`. They are read once per engine, not per step:

```bash
export TLLM_FMHA_LIBS=+beam_shared_prefix
```

No per-request setting is needed.

**How wide?** The gain grows with beam width, prompt length and model size, and there is no
fixed threshold. For a rough guide, on H100 with a 0.6B-parameter model and prompts of a few
hundred to a few thousand tokens, we measured no clear gain at beam width 14, about 1.3 to 1.7
times faster end to end at beam width 30, and about 2 times at beam width 96 with 3,000-token
prompts. Narrow beams (a few) and short prompts are not expected to benefit; measure your own
workload.

| Variable | Default | Meaning |
|---|---|---|
| `TLLM_BEAM_SHARED_PREFIX_BACKEND` | `fa2` | Prefix kernel: FlashInfer `fa2`, `fa3` or `auto`, or `fa4` for the FlashAttention-4 kernels shared with the n-sample shared-prefix library (SM90, bf16, head size 64 or 128; otherwise the library declines with a warning). |
| `TLLM_BEAM_SHARED_PREFIX_MAX_TAIL` | `8` | Most generated tokens per beam under CUDA graphs, where the length of each beam's own part must be fixed when the graph is captured. A longer generation fails at planning time instead of being truncated. Raise it for longer outputs; the per-step cost grows with it, so keep it close to the real output length. |
| `TLLM_BEAM_SHARED_PREFIX_FUSED_TAIL` | `1` | Set to `0` to run the tail attention and merge as separate PyTorch ops (the numerical reference). |
| `TLLM_BEAM_SHARED_PREFIX_CHECK` | `0` | Set to `1` to synchronize after every stage and compare the page table and prefix attention with dense references. Debugging aid; do not use for measurements. |

Beam search keeps near-tied beams or drops them depending on tiny numerical differences, so the
returned beam set can differ from the regular kernel's by a beam or two even though each beam's
score is as accurate.
