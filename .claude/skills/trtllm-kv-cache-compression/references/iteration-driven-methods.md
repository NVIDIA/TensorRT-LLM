<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Iteration-Driven Compression Methods

An iteration-driven method changes the retained KV cache state between
forward steps: it evicts tokens and compacts what is left. The KV cache
compression development guide lists the hooks. This file covers the
contracts around them. Each one comes from a real bug or review finding.

Symbols are named rather than given line numbers. Locate them on the current
main before relying on them.

## 1. Lengths: Logical, Physical, History, Capacity

Keep four quantities apart:

| Quantity | Meaning | Who changes it |
|---|---|---|
| Logical length | Tokens the request has produced, prompt included | The request itself |
| Physical length | Tokens actually stored in the KV cache after compaction | The method, by compacting |
| `py_num_compressed_tokens` | Logical minus physical, accumulated over rounds | The method, when it publishes a round |
| KV cache manager V2 history and capacity | KV cache manager V2's own accounting of the request | KV cache manager V2, and the method through `resize(capacity, None)` |

Rules:

- **Declare the divergence.** `changes_physical_kv_length = True` on the
  config makes `bind_kv_cache_managers()` set `kv_compression_manages_history`
  on the target and draft KV cache managers. KV cache manager V2 then leaves
  history alone during generation resizes. Do not set history yourself.
- **Publish, then shrink.** Add the round's evicted count to
  `py_num_compressed_tokens`, then call `resize(new_capacity, None)` to
  return trailing pages.
- **Attention sees physical; rotary position embedding (RoPE) sees
  logical.** Input preparation subtracts `py_num_compressed_tokens` from the
  cached-token count passed to attention. Positions for new tokens still come
  from the logical length.
- **Keep keys as written.** Keys in the KV cache keep the rotation they were
  written with. A method that does not re-rotate keys must never feed a
  physical index into a RoPE computation.
- **No RoPE fusion.** The config flag also disables RoPE fusion in attention,
  because a fused kernel would rotate by the physical position.

## 2. Protected Tokens

Write down which tokens a round may never move or drop:

- **The prompt**, if the method keeps it. TriAttention keeps every prompt
  token and evicts only generated tokens beyond its budget.
- **The protected tail**: tokens written by the in-flight step, the extra
  KV cache tokens of the attention layout, and the slots reserved for draft
  tokens. TriAttention reserves
  `num_extra_kv_tokens + reserved_draft_tokens + 1`.
- **The next prepared batch.** Under the overlap scheduler, iteration N+1 is
  already prepared when iteration N's `update_resources` runs.
  - Requests in that prepared batch need an extra protected tail covering the
    token it will write and its draft slots.
  - TriAttention records the prepared batch in `on_generation_step_begin` and
    widens the protected tail for those requests.
- **Sliding-window layers.** On sliding-window layers, keep the latest
  window plus the protected tail. The budget must be at least the window
  size.

Test each boundary: a round that ends exactly at the budget, a request in the
prepared batch, a request whose protected tail spans a page boundary.

## 3. Cadence and Due Rules

- Run the round in `on_generation_step_end`, after KV cache manager V2 has
  applied the step's capacity change and any speculative rewind. The executor
  updates the target KV cache manager V2 before the other resource managers,
  and the KV cache compression manager is registered last, after any draft
  KV cache manager V2. Do not reorder either. A KV cache compression manager
  that runs before KV cache manager V2's update has its compaction undone by
  that update.
- Skip dummy requests (CUDA-graph padding and warm-up), requests whose
  generation is complete, and requests still in context initialization. Dummy
  requests never pass through `on_request_init`.
- Make the due rule a pure function of published counts: confirmed
  generated tokens, the budget, and the period. Speculative steps can confirm
  several tokens at once, so coalesce to at most one round per update.
- Skip cheaply when nothing is due. The skip path runs every step, so keep
  it free of device synchronization.

## 4. Selection Versus Movement

- Selection (scores, top-k, tie-breaking) is the method's policy. Movement
  is generic.
- Reuse `build_compaction_params()` and `compact()` from
  `tensorrt_llm/_torch/kv_cache_compression/compaction.py`.
  - They move kept tokens and the protected tail inside each pool group, in
    place, in ascending order.
  - Dense rows get the kept tokens followed by the protected tail.
    Sliding-window rows get the latest window followed by the protected tail.
- Read block tables from a snapshot taken for the round. Read pools through
  the KV cache manager V2 accessors for the expected layout. Do not cache raw
  pointers across rounds.
- Order the round on the stream the next forward pass uses. Record a
  completion event, and make any other stream that reads the result wait on
  it.

## 5. CUDA Graphs and the Overlap Scheduler

- **Outside the CUDA graph.** Rounds run between forward steps, outside CUDA
  graph capture. Warm up kernels and allocate workspaces during
  `bind_kv_cache_managers()`, never inside a captured region.
- **Stable addresses.** CUDA-graph-captured attention metadata reads the
  published cached-token counts. Update those counts through the existing
  preparation path; do not add new captured tensors.
- **Overlap scheduler.** Compare outputs end to end with the overlap
  scheduler on and off, locally on real runs, and write the commands and
  results in the PR description. In pytest, the section 2 boundary test for
  a request in the prepared batch covers the overlap-scheduler ordering.
  When output diverges only with the overlap scheduler on, consider three
  causes: a missing dependency between GPU streams, fixed with an event that
  the consumer stream waits on, as TriAttention makes the KV cache manager
  V2 stream wait for its compaction event; a host buffer that the next step
  overwrites while an asynchronous copy still reads it; and kernel inputs,
  such as batch shapes, that occur only with the overlap scheduler. A full
  device synchronization narrows the search but does not identify the
  cause; check the kernel inputs and which buffers are still in use.
- **Fast paths.** `changes_physical_kv_length = True` disables the steady
  generation-input fast paths. Measure that cost on the target workload.

## 6. Speculative Decoding

- A one-model drafter whose layers share the target KV cache manager V2 is
  compacted with the target model.
- An independent draft KV cache manager V2 must be compacted with the same
  kept ordinals. TriAttention compacts every draft layer as dense, with the
  target's selection.
- Rewinds of rejected draft tokens happen in KV cache manager V2 before the
  round. Protect the reserved draft slots in the protected tail.
- Admission must list the supported speculative modes explicitly. TriAttention
  supports them only with `eviction_mode="union"`, and only for one-model
  multi-token prediction (MTP) and one-model EAGLE3.

## 7. Parallelism

- **Tensor parallelism (TP).** Each rank holds a shard of the KV heads. A
  selection shared across heads, like a union, must be identical on every
  rank: gather the per-rank candidates and fold them. Per-head selections
  must slice the calibration and score inputs per rank.
- **Attention data parallelism.** Each rank owns different requests. Use the
  per-rank head count, not the global one.
- **Pipeline parallelism.** Map local layers to global layer indices for
  calibration and per-layer tables.
- **Context-parallel Helix** is rejected by admission today.

## 8. Request State and Teardown

- Create request-local state in `on_request_init`, and release it in
  `on_request_finish`, which also runs on abort.
- `on_request_finish` runs before KV cache manager V2 frees the request's
  blocks. Never free or resize blocks there.
- Size per-request device buffers at bind time from the maximum batch size.
  Reject configurations whose offsets would exceed 32-bit indexing, or use
  64-bit offsets.

## 9. Calibration and Model Inputs

- Load calibration once at initialization, convert it to the runtime layout,
  and validate it against the model: layer count per pipeline-parallel rank,
  head count, RoPE parameters.
- Fail with an actionable message when something is unsupported, such as
  per-layer-type RoPE parameters, missing layer types on a sliding-window
  model, or no full-attention layer.
- Never compute calibration in TensorRT-LLM. Document the upstream tool that
  produces the artifact.

## 10. Kernels and Performance

- Batch rounds across requests. Fall back to per-request scoring only where
  the batch path is infeasible, and make that a measured, documented choice.
- Fuse scoring, statistics, and selection when intermediates are large.
  Compare the fused selection bit for bit with the unfused reference,
  including tie-breaking.
- Keep host work per round small. Write metadata into pinned buffers and copy
  it once. Avoid per-request device synchronization.
- Report the round's cost in the performance comparison, both amortized per
  step and as the latency spike in the round's step. Report the costs of the
  disabled fast paths and of disabled RoPE fusion separately.

## 11. Worked Example: TriAttention

| Aspect | On main |
|---|---|
| Config | `TriAttentionKvCacheCompressionConfig`: `budget` (the KV cache compression manager rejects values above 16384 at construction), `beta` period, `eviction_mode` (`union`, `per_head`, `per_layer_perhead`), `normalize_scores` (forced on, with a warning, for `union`), `calibration_path` |
| KV cache compression manager | `TriAttentionCompressionManager` in `tensorrt_llm/_torch/kv_cache_compression/triattention/triattention.py` |
| Hooks | `on_request_init` (position-range check), `on_generation_step_begin` (remembers the prepared batch), `on_generation_step_end` (runs due rounds) |
| Round | Move offsets → pinned metadata copied to the GPU → block-offset snapshot → fused scoring → union fold across TP ranks → top-k selection with tie-breaking → `compact()` → publish counts → resize |
| Layers | Full-attention layers and kernel-masked sliding-window layers; the draft KV cache is compacted as dense with the target's kept ordinals |
| Hardware | Blackwell (SM100 family) GPUs |
| Not supported or not validated | Per `examples/kv_cache_compression/triattention.md`: multi-head attention (MHA), multi-head latent attention (MLA), hybrid recurrent state-space model (SSM) pools, and native sliding-eviction layouts are unsupported; TP above 1, attention data parallelism, and disaggregated serving are not validated end to end. Re-check the doc before relying on this row |

Lessons specific to this method:
- All three eviction modes are part of the algorithm. Keep them through
  refactors.
- Evaluate with the reference implementation's protocol and calibration. Its
  default setting matches `union`.
- Short-output tasks never trigger a round.
- An accuracy gap can come from answer extraction rather than eviction.
  Check the scorer before blaming the method.
