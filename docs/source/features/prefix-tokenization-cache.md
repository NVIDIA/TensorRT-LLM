# Prefix-Tokenization Cache

## Overview

In multi-turn agentic serving, each turn's prompt is usually the previous turn's
prompt plus a small delta, yet the frontend re-tokenizes the whole prompt on
every turn. With long prompts, this can make tokenization a significant share
of the time a context server spends per request.

The prefix-tokenization cache remembers the token ids of recent prompts. When a
new prompt extends a cached one, it tokenizes only the tail and splices it onto
the cached ids, so the cost of a turn scales with the delta rather than the
whole prompt. The new prompt may also rewrite the last tokens of the cached one
(those after the seam described below), as a chat template does when it
replaces the previous turn's generation prefix with the response.

Agentic clients also rewrite the end of the history between turns, for example
by trimming a tool result or replacing the last message, so the new prompt is
no longer an extension of the cached one. The cache records a checkpoint every
256 tokens of a cached prompt. When a new prompt diverges from a cached one, the
cache reuses the cached ids up to the checkpoint one interval before the last
checkpoint inside their longest common prefix, and tokenizes only the rest. The
same is tried when a new prompt extends a much shorter cached prompt, such as
another conversation with the same opening, and the later split is used.

Tokenizing a tail in isolation is not generally identical to tokenizing the
whole prompt, because BPE merges can straddle the split. The cache therefore
backs off a fixed number of tokens from the end of the cached prefix (for a
diverging prompt, at least one checkpoint interval before the divergence),
re-tokenizes from there, and accepts the splice only if the re-tokenized ids
equal the cached ids over that seam (for a diverging prompt, over the whole
interval up to the next checkpoint). Otherwise the prompt is tokenized in full. Every split is moved to
the first token of a character, so a character tokenized into several tokens is
never split. The output is always identical to tokenizing the whole prompt.

Some tokenizers alter every text they tokenize; for example, a `tokenizer.json`
whose normalizer prepends `▁` (as for Phi-3 and Mixtral-8x7B-v0.1) adds a space
to every tail. Such a shift can pass the check at the seam, so the cache probes
the tokenizer when it is created and stays disabled, with a warning, if a tail
can be shifted.

## Enabling the cache

The cache is **off by default**. Enable it with the `enable_tokenization_cache`
LLM argument, either in the `trtllm-serve` configuration file:

```yaml
enable_tokenization_cache: true
```

or from Python:

```python
from tensorrt_llm import LLM

llm = LLM(model="zai-org/GLM-5.2", enable_tokenization_cache=True)
```

The cache size can be tuned with environment variables:

| Environment variable | Default | Behavior |
|---|---:|---|
| `TLLM_PREFIX_TOKEN_CACHE_ENTRIES` | 512 | Maximum number of cached prompts. Eviction is least-recently-used. |
| `TLLM_PREFIX_TOKEN_CACHE_MAX_CHARS` | 67108864 | Maximum total characters of cached prompt text. Cached ids are stored as int32, about one byte per character of English text, so this bounds host memory to roughly twice this many bytes. |
| `TLLM_PREFIX_TOKEN_CACHE_MIN_CHARS` | 4096 | Prompts shorter than this are tokenized normally and never cached. |
| `TLLM_PREFIX_TOKEN_CACHE_ROUTE_BY_CONVERSATION` | 0 | Set to `1` to tokenize each conversation in one serving frontend; see [Multiple serving frontends](#multiple-serving-frontends). |

The sizing values must be positive integers, and the routing switch must be `0`
or `1`. A malformed value raises at server startup so that misconfiguration
fails loudly rather than silently disabling a feature you turned on.

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
- A prompt that reuses a cached entry replaces that entry, so a conversation
  costs one entry regardless of how many turns it has. A prompt that diverges
  from a cached entry before its seam is added next to it instead: the
  divergence may be a branch, such as a subagent started from a shared history,
  and the original conversation can still continue from the old entry. Lookup is bucketed by a
  hash of the first `TLLM_PREFIX_TOKEN_CACHE_MIN_CHARS` characters, and each
  entry keeps a short probe of the characters before its seam so that
  conversations sharing a long system preamble are told apart without
  comparing the preamble.
- The router of a disaggregated server uses the same cache for its routing
  token ids when its `context_servers` / `generation_servers` section sets
  `enable_tokenization_cache`. The candidate scan runs outside the cache lock.
- An unexpected exception inside the cache disables it, with a warning, and the
  request is tokenized normally. A cache problem never fails a request.
- The cache logs its configuration at startup and, every 1000 cache-eligible
  requests (routed ones included), an `INFO` line with the hit rate (and how many hits were on
  diverging prompts), the number of re-synchronization fallbacks, how many
  prompts were routed to another frontend and how many were tokenized locally
  because routing failed (see below), and its occupancy, so a cache that is
  enabled but never hitting is visible in the server log.

## Multiple serving frontends

`trtllm-serve` starts 8 frontend processes by default. When you enable the
cache for multi-turn traffic, use a single serving frontend by passing
`--num_serve_frontends 1` (or setting `num_serve_frontends: 1` in the
`--config` YAML file). Multiple frontends are available only with the classic
IPC executor, so this section applies only there.

The cache lives in the memory of one frontend process and is not shared
between processes. With `num_serve_frontends: K`, the K frontend processes
share the serving port through `SO_REUSEPORT`, and the kernel assigns each new
TCP connection to one of them without looking at the request. A client that
opens a new connection per request, as many benchmark clients do, therefore
sends each turn of a conversation to an arbitrary frontend. That frontend
usually has not seen the conversation's previous turn, so the prompt is
tokenized in full.

On an agentic multi-turn trace with prompts of about 160k tokens on average
(GLM-5.3 and its tokenizer on GB300), at one session, the p90 time spent in the
frontend was about 165 ms with 8 frontends and 44 ms with one. Decode throughput
per user was the same from 1 to 32 concurrent sessions, the range measured,
because at that concurrency the frontend is not the bottleneck.

Multiple frontends help when a single Python frontend cannot keep up, typically
at high request rates with short prompts. That is not the workload this cache
targets.

A client that can pick the frontend itself avoids the problem: with
`serve_frontend_affinity_port_stride` set, each frontend also listens on its own
port, and a disaggregated server sends every context request that carries a
conversation id to the port of the frontend that conversation maps to.

Otherwise, if you need several frontends, set
`TLLM_PREFIX_TOKEN_CACHE_ROUTE_BY_CONVERSATION=1` to tokenize each conversation
in one frontend. A chat request's conversation id (the same id attention DP
routes by: from `conversation_params.conversation_id` or headers such as
`X-Correlation-ID`, or, on a disaggregated context server, the sub-agent parent
id forwarded by the disaggregated server) is mapped to an owner frontend, the
same one the affinity ports use, and the frontend the request arrived at sends
the prompt text to the owner over a local Unix socket and gets back the token
ids; the request itself stays where it arrived. Routing helps only when the id
is stable across a conversation's turns: with an id that is fresh on every
request, as many gateways set `X-Correlation-ID`, it only adds a hop per turn.

A request without a conversation id is tokenized locally. So is one whose owner
is not serving, does not answer within 5 seconds, or cannot tokenize the
prompt; these are counted as route failures in the periodic log line. An owner
that times out is logged with a warning and not asked again for 30 seconds.
Each frontend keeps one connection to every other frontend, shared by its
tokenizing threads, so with K frontends each one holds 2 × (K − 1) extra Unix
sockets and K − 1 extra threads.

The switch is off by default. When it is set but cannot take effect, with a
single frontend or with the cache disabled, the server logs a warning; when it
takes effect, each frontend logs an `INFO` line. On the trace above with 8
frontends, it brought the p90 frontend time at one session from about 165 ms to
about 58 ms.

## Measured effect

An A/B on GLM-5.2 (GB300, disaggregated, matched pair, 3600 s, ~29.6k requests
per arm, identical configuration except `enable_tokenization_cache`):

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

Splicing diverging prompts was checked on 300 multi-turn prompts whose history
is rewritten between turns: all 300 matched tokenizing the whole prompt, 99.3%
were cache hits, and tokenization took about a tenth of the time of
tokenizing every prompt in full.

## Node-shared prompt token cache

`prompt_token_cache_config` (prototype) keeps prompt token ids in files that
every serving frontend of a node shares, so a turn still hits the cache when
another frontend (`--num_serve_frontends`) served the previous turn. Only the
DeepSeek-V4.1 input processor (checkpoints with a vision tower) uses it; other
input processors ignore it with a warning.

```yaml
prompt_token_cache_config:
  path: /dev/shm/trtllm-prompt-tokens  # node-local, the same for every frontend
  max_bytes: 1073741824                # 1 GiB
```

- `path` is a directory on a node-local file system such as a tmpfs, created if
  missing. Every frontend of the server must use the same directory, which must
  be owned by the serving user and not writable by group or others.
- `max_bytes` caps the space the entries take, each counted in whole file
  system blocks; the least recently used entries are evicted first.
- A prompt reuses the ids of its longest cached prefix that ends at an
  `<｜Assistant｜>` marker and tokenizes only the rest, so the ids equal a full
  tokenization. Startup fails unless the tokenizer is fast and keeps that marker
  as one added token that does not strip its neighbours.
