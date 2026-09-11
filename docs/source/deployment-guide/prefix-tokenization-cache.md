# Prefix-Tokenization Cache in TensorRT LLM

## Overview

In multi-turn agentic serving, each turn's prompt is usually the previous turn's
prompt plus a small delta, yet the frontend re-tokenizes the whole prompt on
every turn. On a GLM-5.2 disaggregated context server with ~38k-token prompts,
tokenization accounted for 47.4% of context-server wall-clock at
43.7 ms/request.

The prefix-tokenization cache remembers the token ids of recent prompts. When a
new prompt extends a cached one, it tokenizes only the tail and splices it onto
the cached ids, bringing tokenization to 5.49 ms/request in the same setup.

Tokenizing a tail in isolation is not generally identical to tokenizing the
whole prompt, because BPE merges can straddle the split. The cache therefore
backs off a fixed number of tokens from the end of the cached prefix,
re-tokenizes from there, and accepts the splice only if the re-tokenized ids
equal the cached ids over that seam. Otherwise the prompt is tokenized in full.
The output is always identical to tokenizing the whole prompt.

## Enabling the cache

The cache is **off by default**. Enable it with:

```bash
export TLLM_PREFIX_TOKEN_CACHE=1
```

| Environment variable | Default | Behavior |
|---|---:|---|
| `TLLM_PREFIX_TOKEN_CACHE` | unset | Set to exactly `1` to enable the cache. Any other value leaves it disabled. |
| `TLLM_PREFIX_TOKEN_CACHE_ENTRIES` | 512 | Maximum number of cached prompts. Eviction is least-recently-used. |
| `TLLM_PREFIX_TOKEN_CACHE_MAX_CHARS` | 67108864 | Maximum total characters of cached prompt text. Cached ids are stored as int32, about one byte per character of English text, so this bounds host memory to roughly twice this many bytes. |
| `TLLM_PREFIX_TOKEN_CACHE_MIN_CHARS` | 4096 | Prompts shorter than this are tokenized normally and never cached. |

Each value must be a positive integer. A malformed value raises at server
startup so that misconfiguration fails loudly rather than silently disabling a
feature you turned on.

## Scope and safety

- Each `DefaultInputProcessor` owns its own cache, so cached ids are never
  shared across tokenizers.
- The cache requires a fast (Rust-backed) tokenizer, because it relies on
  character offsets. With a slow tokenizer the cache is disabled with a warning.
- The cache is used only when the tokenizer would be called exactly as the
  cache calls it: `add_special_tokens=False` and no prompt truncation. Chat
  completions apply the chat template and tokenize with
  `add_special_tokens=False`, so they benefit. `/v1/completions` defaults to
  `add_special_tokens=True` and is not accelerated.
- A prompt that extends a cached entry replaces that entry, so a conversation
  costs one entry regardless of how many turns it has. Lookup is bucketed by a
  hash of the first `TLLM_PREFIX_TOKEN_CACHE_MIN_CHARS` characters.
- An unexpected exception inside the cache disables it, with a warning, and the
  request is tokenized normally. A cache problem never fails a request.

## Measured effect

An A/B on GLM-5.2 (GB300, disaggregated, matched pair, 3600 s, ~29.6k requests
per arm, 0.34% error rate in both, identical configuration except the
environment variable):

| Metric | OFF | ON | Delta |
|---|---:|---:|---:|
| TTFT p50 (ms) | 1135.5 | 938.9 | **−17.3%** |
| TTFT avg (ms) | 1897.3 | 1786.0 | −5.9% |
| Total throughput (tokens/s) | 1,161,644 | 1,162,562 | +0.08% |
| Inter-token latency (ms) | 11.258 | 11.316 | +0.5% |

This is a **latency win, not a throughput win**, at that operating point:
total throughput and inter-token latency are flat. Do not expect a throughput
improvement from it.

The splice was also checked against real `gpt2`, Llama-3, and Qwen3
tokenizers on 300 randomly grown multi-turn prompts each, with deltas
deliberately starting mid-word: 300/300 identical ids, 280 cache hits, and 0
re-synchronization fallbacks per tokenizer.
