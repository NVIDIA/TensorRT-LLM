# NGram Speculative Decoding

This document shows how to run a model with NGram speculative decoding
(supported as `ASSISTED_GENERATION` in transformers and vLLM, source:
[GitHub](https://github.com/apoorvumang/prompt-lookup-decoding/tree/main))
in TensorRT LLM.

## Overview

NGram builds a pattern pool from the prompt and previously generated tokens
and proposes draft tokens by matching the tail of the current sequence
against that pool. It has 2 hyperparameters that control the process of
generation:

- `max_draft_len`: the maximum number of tokens provided as draft tokens in
  one iteration, which is usually from 4 to 10 in common usage (default
  value: 4). Empirically, the larger the value is, the higher acceptance rate
  but higher overhead is expected at the same time, so the right balance
  based on the models and application scenarios needs to be found.
- `max_matching_ngram_size`: the maximum number of tokens extracted from the
  tail of the input prompt or generated output as a pattern, which is used to
  search corresponding draft tokens (default value: 2). Empirically, the
  larger the value is, the more precise context can be matched from the
  existed sequence, indicating higher acceptance rate, but the higher
  probability of miss-match and higher overhead appear, which fall back to
  normal generation (one token per iteration).

## Support Matrix

  * GPU Compute Capability >= 8.0 (Ampere or newer)
  * FP16 / BF16 / FP8
  * Paged KV Cache
  * Tensor Parallel

## Usage

```bash
python3 examples/llm-api/quickstart_advanced.py \
    --spec_decode_max_draft_len 4 \
    --max_matching_ngram_size 2 \
    --disable_overlap_scheduler \
    --disable_kv_cache_reuse
```

With the LLM API, configure NGram through `NGramDecodingConfig`
(`speculative_config`). See the
[speculative decoding documentation](https://nvidia.github.io/TensorRT-LLM/features/speculative-decoding.html)
for details.
