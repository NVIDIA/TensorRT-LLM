# Sliding-window attention, alternating with full attention

A model whose layers alternate between a short attention window and full
causal attention. Written from the gpt-oss-120b/sm_100/tp1 onboard
(36 layers, `layer_types` alternating `sliding_attention` /
`full_attention`, `sliding_window: 128`); the mechanism recurs across
families (Gemma, Mistral, Ministral).

## The window is a mask, not an addressing mode

This is the fact everything else follows from, and the one an upstream
comment gets wrong. TensorRT-LLM's attention op appends every new token
at its **absolute** position — entry `t // tokens_per_block` of the
sequence's block-offsets row, slot `t % tokens_per_block` — and wraps
nothing, not at the window and not at `max_seq_len`. `attention_window_size`
only restricts which cached keys a query may read. Any cyclic reuse of
pool memory is the *caller's* page mapping. Measured on sm_100: a comment or
doc that says the kernel "wraps KV reads modulo the attention window" is wrong
about the mechanism, and sizing a pool from it is wrong too.

**Boundary:** a query at absolute position `p` attends to
`[p - W + 1, p]` — exactly `W` keys, never `W + 1`. `W` is a token count,
not a page count, and may be shorter than one page.

## What a target owes

- The block-offsets row still needs `ceil(total_kv / tokens_per_block)`
  entries, window or no window. Only entries holding at least one
  in-window token must point at live pages: aging the older ones out to
  garbage leaves the decode output bitwise identical.
- `sequence_length` / `host_past_key_value_lengths` must stay the
  **global** cached+new count. It is both the window origin and the append
  position, so capping it to `W` moves the write to slot `W - 1` and
  truncates the attended range — silently. `context_lengths` and
  `max_seq_len` are inert on generation rows.
- Prefill is **not** shrunk by the window. One call's new tokens must land
  in distinct physical slots, or the writes race and leave torn cache rows
  nondeterministically, with no error.
- A caller-side page ring (`ring[j % P]`) does make the pool wrap, bounded
  by `P * tokens_per_block >= W`. Below that bound the result is silently
  wrong rather than rejected.

## The route that works, and what it does not buy

**Per-layer windows over one shared pool are the correctness route.**
`attention_window_size` is a per-call scalar, so alternating layers can
address one pool, one layer→pool mapping and one block-offset table
inside a single batch. Because a target built on the catalog does not
construct the engine's own attention modules, **the runtime is never told
about the window at all**: the KV manager sizes one pool for full
attention and returns an identity `host_kv_cache_pool_mapping`. The
per-layer window is then a pure modeling-side argument needing no
`llm_args` support and no page-ring work. Both FMHA kernel families
(`...PagedKvDense...` and `...PagedKvSlidingOrChunkedCausal...`) appear at
warmup. What picks between them is the batch's longest KV run against the
window, not the layer's window setting: a windowed layer whose history
still fits inside the window takes the *dense* kernel and crosses over on
the first step past it, so one sequence pays both JIT compiles over its
lifetime (`thop_attention`, *Notes*).

**It saves no memory.** The shared pool is still sized for the
full-attention layers, so the alternation costs exactly what an
all-full-attention model would. A real footprint reduction needs
per-window pools, i.e. `num_pools > 1` addressing.

Before spending a certification on that: on gpt-oss-120b it was measured
*not* to be the win. KV capacity never bound (36.3% of the pool used at
concurrency 256), the per-layer window already cuts the FMHA read cost
where it matters (23.9 µs sliding vs 100.3 µs dense per layer at
concurrency 256), and stock trtllm sizes the **same** single
full-window pool on that checkpoint — so it is not a difference between
the two systems at all. Check those three things before assuming
multi-pool addressing is worth the run.
