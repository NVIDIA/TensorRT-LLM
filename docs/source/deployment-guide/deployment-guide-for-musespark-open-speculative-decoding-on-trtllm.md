<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Deployment Guide for Muse Spark Open Suffix-Automaton Speculative Decoding on TensorRT LLM - Blackwell

## Introduction

Muse Spark Open supports two kinds of speculative decoding in TensorRT LLM: a DFlash2 drafter, covered in the [Muse Spark Open deployment guide](deployment-guide-for-musespark-open-on-trtllm.md#speculative-decoding), and model-free suffix-automaton (SA) drafting, covered here.

SA drafts by matching the request's own token history: it needs no draft model and no extra weights, the target model verifies the drafts in its own forward pass, and both run inside CUDA graphs. It pays when the output repeats earlier text — structured output, code edits, tool-call loops — and costs a verification of `1 + max_draft_len` positions per step when it does not.

## Recipe

Start from the recommended configuration in the [deployment guide](deployment-guide-for-musespark-open-on-trtllm.md#recommended-performance-settings) (`examples/configs/curated/musespark-open.yaml`) and add:

```yaml
speculative_config:
  decoding_type: SA
  max_draft_len: 2
  max_matching_ngram_size: -1   # match the longest suffix in the request's history
```

Everything else stays as in the recommended configuration, including the `TRTLLM` attention backend, the FP8 KV cache, and the overlap scheduler. Launch the server with the same `trtllm-serve` command as the deployment guide.

## Limits

SA uses the one-model speculative path, which rejects requests that set `min_tokens`, `bad_words`, `no_repeat_ngram_size`, `embedding_bias`, `top_p_decay`, or a logits post-processor; drop those fields from the request or serve it without speculation.

## Accuracy

Draft-token acceptance is strict: an accepted draft token is exactly the token the target model emits at that position, so SA reproduces the target model's accuracy. GSM8K (5-shot, all 1,319 samples) on one GB300 NVL node:

| Configuration | GSM8K accuracy |
|---|---|
| Recommended configuration | 94.84 |
| Recommended configuration with SA, `max_draft_len: 2` | 94.81 |

## Performance

SA's gain depends on how much of the output it can find in the request's own history, so it varies with the workload. Two measurements on one GB300 NVL node, with the recommended configuration with and without SA (`max_draft_len: 2`):

* **Coding prompts at the default reasoning effort** (80 prompts, temperature 0, 8 concurrent requests): the mean acceptance length was 1.31 tokens per step, and SA decoded about 4 % slower than no speculation, because the extra verification work outweighed the few accepted drafts.
* **The deployment guide's [benchmark command](deployment-guide-for-musespark-open-on-trtllm.md#benchmarking-performance)** (random-token prompts, 8,192 input and 1,024 output tokens, `ignore_eos`), output tokens per second:

| Concurrent requests | Without speculation | With SA |
|---|---|---|
| 1 | 117 | 216 |
| 32 | 1,318 | 1,315 |
| 256 | 2,846 | 2,773 |

SA pays when much of the output repeats text the request already contains — structured output that echoes its input, code edits, tool-call loops — and mostly at low concurrency, where decoding is latency-bound. On free-form reasoning text, and at high concurrency, expect no gain. Measure the acceptance length on your own traffic before enabling SA: with `per_request_spec_decode_stats: true` in the configuration, each chat completion reports its accepted and drafted token counts.

## Credit

The suffix-automaton drafter and its GPU kernels are Baseten's `sa_spec` contribution to TensorRT LLM.
