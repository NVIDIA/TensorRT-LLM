---
name: trtllm-modeling-v2-bounded-replay
description: Load when adding sliding-window (SWA) bounded replay to a target — `decoder_replay` (the layers after the last KV-source layer run over only the last `window` prompt tokens) or `encoder_replay` (a prefix-cache hit rebuilds sliding-window KV by re-forwarding the last `window` cached tokens). Deliberately not output-identical, and gated by a work witness, because a unit that implements nothing passes accuracy too.
---

# trtllm-modeling-v2-bounded-replay

`--features decoder_replay` and `--features encoder_replay`. Both belong to a
model that keeps two kinds of KV, global (often compressed) KV and
sliding-window (SWA) KV, and that was post-trained to tolerate truncated SWA
state. Such a model may be a **causal encoder-decoder**: the first block of
layers is the "encoder", the rest the "decoder", and decoder global KV is
projected from the encoder's final hidden state rather than from each decoder
layer's own. "Encoder" here is those causal layers — **never the vision
tower** — and "replay" is neither CUDA-graph replay nor recurrent-state
replay. The checkpoint's tech report says whether it was trained for either
feature; quote it in the plan.

## The definitions

**Bounded replay** replays only the most recent `n_win` tokens and truncates
SWA to the replay segment, accepting approximate states: for a replay starting
at position `r`, a query at position `i` attends to SWA keys in
`[max(r, i - n_win + 1), i]`.

**Encoder bounded replay** makes prefix caching depend only on global KV, so
SWA KV can leave the persistent cache. On a prefix hit the last `n_win` cached
tokens are replayed together with the uncached suffix. The replayed tokens
regenerate only SWA KV, reusing the cached global KV without recomputing or
overwriting it; the uncached suffix generates both. The replayed prefix state
is approximate by design.

**Decoder bounded replay** bounds the decoder's forward pass to `n_win`
tokens. Decoder global KV comes from the encoder's output, so the only thing
standing between prefill and stopping at the encoder is decoder SWA KV: at
every prefill the last `n_win` prompt tokens' encoder outputs run through the
decoder layers under the same SWA truncation, and the decoder SWA KV that
produces serves decoding only, never prefix caching. Not mathematically
equivalent to the full computation; the training simulated it.

`n_win` is the checkpoint's sliding window, read from its config.

## The property that shapes the gates

**Neither feature is output-identical.** Both truncate sliding-window state on
purpose, so the accuracy gate is a real check of them — unlike host offload or
disaggregation, where it is blind.

It is not a sufficient check. The truncation was designed, and post-trained,
to "barely compromise" accuracy. So a unit that implements **nothing** lands
inside the same tolerance, reports green, and ships a target that does the
full prefill while its Coverage section claims the saving. The only
observable that differs is **work**: rows the decoder ran, rows a prefix hit
re-forwarded. Only the target can count those, so it prints them, and the
**replay witness** judges them against a window and a workload the spec
declared before assembly. See **Gates**.

## decoder_replay — the target owns all of it

Target-local: no runtime component, no scheduler change.

- **Which layers are bounded.** The decoder's: for example the layers after
  the last KV-source layer (the last entry of a `kv_source_layer_ids` config
  key). The boundary is a decision, not a fact. A tech report may start the
  decoder *at* the last KV-source layer, while an implementation may start it
  one layer later and run that layer over every prompt token, because its
  compressor is what produces the decoder's global KV. Pick one, say why, and
  record it — the plan's open risks carry it.
- **Context phase.** Every layer up to the boundary — including the last
  KV-source layer's compressor and index-K projection — runs over all prompt
  tokens. The bounded layers run over only the last `min(N, window)` prompt
  tokens of each context request, with SWA truncated at
  `r = N - min(N, window)`. Only the last token feeds the head. Main and
  indexer KV stay exact; only decoder SWA KV is approximate.
- **Only in the final context chunk.** Under chunked prefill the bounded
  layers do nothing in earlier chunks and run once, in the last. The last
  `window` prompt tokens can straddle chunks, so the hidden state entering the
  first bounded layer must be **retained per request across chunks** for the
  tail — a buffer the target owns, keyed by request, freed at the final chunk.
  Dropping it silently replays fewer than `window` tokens and still passes
  accuracy.
- **What it cannot return.** The bounded layers never computed rows before the
  tail, so prompt logprobs, context logits and full hidden-state capture do
  not exist under this feature. The target must fail fast rather than return
  rows it did not compute.
- The decoder SWA KV the replay produces serves decoding only — it is not
  something prefix caching may reuse.

The contract, compactly: after the last KV-source layer, run the remaining
layers over only the last `window` tokens of a prefill. Main and indexer KV
stay exact; nothing is replayed. Deterministic for a fixed prompt and chunk
size.

## encoder_replay — runtime and target, and the runtime half is new

| responsibility | owner |
|---|---|
| the prefix cache keeps only global KV (compressed KV + indexer K) for a cached prefix; SWA KV is not held by it | runtime |
| on a prefix hit, schedule the last `min(window, reused)` cached tokens as **replay** rows ahead of the uncached suffix, charged against the token budget | runtime |
| a reuse boundary aligned to the compress ratio — even, for ratio 2 — or backed off to one | runtime |
| replay rows write **SWA KV only**: compressor skipped, indexer K not written, global KV never overwritten | target |
| replay rows produce no logits and are never sampled | target |
| the uncached suffix writes both global and SWA KV, as in an ordinary prefill | target |

An odd boundary on a ratio-2 layer splits a compressed pair: one half is
cached and the other was never computed, and the result is a quietly wrong
global entry rather than an error. Refuse the combination in the runtime; the
witness checks it when the spec declares `reuse_alignment`.

**TRT has none of the runtime half today.** `DeepseekV4CacheManager` in
`tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/cache_manager.py`
keeps SWA as a sliding pool whose blocks reuse depends on, and does not back
off a reuse match (`_supports_reuse_match_backoff` is false there). A target
cannot own the scheduler, so name that work in the plan as runtime work, with
its owner — a target that fakes it by re-forwarding the whole prefix on a hit
is exactly what the witness's accounting check catches.

Where another engine already implements it, read that for the mechanics and
name it in the plan. The shape is: re-run the cached tokens
`[end - window, end)` through the layers with the compressor skipped, keep SWA
in a per-request ring, and charge at most `window` replay tokens per request
to the scheduler.

## Interaction rules, and saying which combinations work

Combinations a published implementation of these features refuses:

| feature | refused together with |
|---|---|
| `encoder_replay` | prefill CUDA graphs, attention DP, PD disaggregation, LoRA |
| `decoder_replay` | prefill CUDA graphs, attention DP |

**No implementation anywhere combines bounded replay with attention DP.** A
`dep` member of a unified target is therefore unexplored ground, not a
configuration with a known answer.

The target must **say** which combinations it supports:

- a fail-fast assert for each combination it does not, naming the reason — a
  combination that silently runs without replay is the silent wrong this
  package exists to prevent, and it passes accuracy;
- the spec's `members:` narrowed for the unit to the members it supports, so
  the gate loop does not ask for the rest:
  `members: {default: [tp4, tep4, dep4], decoder_replay: [tp4, tep4]}`;
- and the same two statements in the target's Coverage section.

"Prefill CUDA graphs" means a captured **context** forward. TRT captures
generation steps by default and runs context eagerly, so the conflict appears
only where context forwards are captured (piecewise capture, compile). The
replay row selection is data-dependent; keep it out of any capture.

## Gates

boot → accuracy → **replay witness**, all under `modeling_v2=require`.

**The switch.** Each feature is turned off by an environment variable the
target reads at construction — for example
`MYMODEL_DECODER_REPLAY=0` — and the spec names it as the
witness's `contrast_env`. Off must mean the full computation, not an error:
the witness runs both.

**The counter.** When `TRTLLM_MODELING_V2_WITNESS=1` is set, the target prints
one line per context request, when that request's context phase completes
(after its final chunk):

```
MODELING_V2_WITNESS prompt_tokens=4103 reused_tokens=0 replayed_tokens=0 forwarded_tokens=4103 decoder_tokens=128
```

Space-separated `key=<int>`, on one line; anything may precede the marker.
Every field is counted for that request over its whole context phase:

| field | what it counts |
|---|---|
| `prompt_tokens` | the request's prompt length |
| `reused_tokens` | prompt tokens served from the prefix cache |
| `replayed_tokens` | cached tokens forwarded again in replay mode |
| `forwarded_tokens` | rows the model's first layer processed |
| `decoder_tokens` | rows the first bounded layer processed |

`decoder_replay` reads `forwarded_tokens` and `decoder_tokens`;
`encoder_replay` reads the other four. Print them all when the target
implements both.

**Count rows the layer actually ran**, from the shape of the tensor it ran
over, accumulated per request id across chunks — not a formula of the prompt
length. A counter that prints `min(prompt, window)` passes the ON leg and is
caught by the OFF leg, which is the leg's purpose; do not make the OFF leg
the only thing standing between a formula and a pass. On a multi-rank run,
printing from every rank is fine: each verdict is per line.

**The verdicts**, against the spec's declared values:

| feature | replay ON | replay OFF (`contrast_env`) |
|---|---|---|
| `decoder_replay` | `decoder_tokens <= window` for every request, and the long prompt was forwarded cold (`forwarded_tokens >= prompt_tokens`) | `decoder_tokens == forwarded_tokens` for every request |
| `encoder_replay` | `replayed_tokens == min(window, reused_tokens)` and `forwarded_tokens == prompt_tokens - reused_tokens + replayed_tokens` for every request; at least one hit with `reused_tokens > window`; `reused_tokens` a multiple of `reuse_alignment` | `replayed_tokens == 0`, and the same accounting identity |

No counter line at all is a FAIL that says the target printed no witness
counter. The witness builds its own workload from the declared sizes — one
long prompt for `decoder_replay`; two prompts sharing a long prefix, with
block reuse on, for `encoder_replay` — so the boot spec's short prompts, which
never exceed the window, are not used.

**The spec**, declared before assembly and never derived from the run:

```yaml
features: [decoder_replay, encoder_replay]
accuracy:
  witness:
    decoder_replay:
      contrast_env: {MYMODEL_DECODER_REPLAY: "0"}
      window: 128                # the checkpoint's sliding window
      prompt_tokens: 4096        # > window; at most 16384
    encoder_replay:
      contrast_env: {MYMODEL_ENCODER_REPLAY: "0"}
      window: 128
      prefix_tokens: 4096        # > window; at most 16384
      reuse_alignment: 2         # e.g. the compress ratio
```

`runtime.max_seq_len` (through `extra_flags`) must admit the longest witness
prompt, and the `encoder_replay` unit's runtime must keep block reuse on — the
witness refuses to launch otherwise, since no request could hit the cache.

## What the gates still do not tell you

That prefill got faster. The witness says rows were skipped; whether a 32K
prompt now prefills in about half the time is a perf measurement for tuning.
And that the approximation is the post-trained one: accuracy within tolerance
is the evidence for that, which is why this feature's accuracy gate is not a
formality.

## Report additions

- the layer boundary chosen for `decoder_replay`, and why
- which combinations the target supports, the asserts that refuse the rest,
  and the member set the unit was gated at
- for `encoder_replay`, where the runtime half lives and who owns it
- the witness lines' ON and OFF values for the longest request, verbatim
- accuracy with replay against the anchor, and which anchor that was
