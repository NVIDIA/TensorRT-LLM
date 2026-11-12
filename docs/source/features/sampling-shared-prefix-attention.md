<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Shared-Prefix Attention for n-Sample Requests (Prototype)

When a request sets `n > 1` in `SamplingParams` (see [Sampling](sampling.md)), TensorRT-LLM
generates `n` samples from the same prompt, for use cases such as best-of-n sampling. With
KV cache block reuse (on by default), the samples share the prompt's KV cache pages. During
decode, however, the attention kernel still reads the prompt's KV cache from GPU memory once
per sample at every step. These redundant reads add GPU time that grows with prompt length,
`n`, and KV cache size per token, especially for models with full multi-head attention.

**Shared-prefix attention** addresses this by reading the shared prompt once per group of
samples and combining the result with each sample's own tokens. Prefill is unaffected. For
small groups with short prompts, where sharing does not pay off, a size gate (see
[Usage](#usage)) attends the samples without sharing.

This feature is a prototype: its settings and metrics may change.

## Support

- H100 (SM90) GPUs. Other GPUs run the regular attention.
- PyTorch backend, KV cache manager V2 (`use_kv_cache_manager_v2: true`), KV cache block
  reuse, bf16 KV cache.
- Qwen2-architecture models (dense MHA/GQA, head size 64 or 128, no sliding window).
- Linear speculative decoding (for example EAGLE-3) with at least 3 draft tokens per step.
- CUDA graphs and the overlap scheduler.
- Decode-only steps. A step that also carries a prefill runs the regular attention.

## Usage

Set the environment variable before starting `trtllm-serve` or creating the `LLM`:

```bash
export TLLM_FMHA_LIBS=+sampling_shared_prefix,+flash_attn4_generation
```

`+sampling_shared_prefix` enables shared-prefix attention. `+flash_attn4_generation` runs
the remaining decode batches through FlashAttention-4.

No per-request setting is needed: the samples of each `n > 1` request are grouped
automatically. Samples without a group to share with, including `n = 1` requests, are
attended on their own. In a decode step without groups, the regular attention runs unless
the step replays a CUDA graph; such a step runs through shared-prefix attention, with every
sample attended on its own.

| Variable | Default | Meaning |
|---|---|---|
| `TLLM_SAMPLING_SHARED_PREFIX_MIN_GROUP` | `2` | Fewest samples in a group. |
| `TLLM_SAMPLING_SHARED_PREFIX_MIN_SHARED_TOKENS` | unset | Fewest shared prefix tokens of a group, counted in full KV pages. When unset, a group with full multi-head attention shares only if it saves enough KV reads; a GQA group always shares. |
| `TLLM_SAMPLING_SHARED_PREFIX_MAX_GROUPS` | `8` | Most groups that share in one step; further groups are attended as usual. |

## Metrics

Three counters show how often shared-prefix attention runs. Each counts request-steps: one
sample decoded in one decode step. The counters are cumulative and exclude warm-up.

| Prometheus counter | Iteration-stats field | Counts samples that... |
|---|---|---|
| `trtllm_shared_prefix_attn_request_steps_total` | `requestSteps` | were decoded |
| `trtllm_shared_prefix_attn_shareable_request_steps_total` | `shareableRequestSteps` | share at least one full KV page with another sample in the step, in steps shared-prefix attention plans |
| `trtllm_shared_prefix_attn_shared_request_steps_total` | `sharedRequestSteps` | were decoded with shared-prefix attention |

To enable them, add to the `trtllm-serve` config file:

```yaml
enable_iter_perf_stats: true
return_perf_metrics: true
```

The counters then appear in two places:

- `/prometheus/metrics`: the Prometheus counters above, one series per `dp_rank`.
- `/metrics`: the `sharedPrefixAttnStats` field of each iteration record.

To read them:

- `shared / request_steps` is the fraction of decoded samples that used shared-prefix
  attention. In Prometheus:

  ```
  rate(trtllm_shared_prefix_attn_shared_request_steps_total[5m])
    / rate(trtllm_shared_prefix_attn_request_steps_total[5m])
  ```

- `shareable - shared` counts samples that had a prefix to share but did not use it,
  because the size gate rejected their group or the step already had `MAX_GROUPS` groups.
- `request_steps - shareable` counts samples with nothing to share, or in steps that
  shared-prefix attention does not plan (for example, a step that also ran a prefill or
  has no draft tokens).
