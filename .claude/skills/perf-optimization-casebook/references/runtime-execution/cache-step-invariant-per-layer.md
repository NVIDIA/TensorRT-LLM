---
id: case-cache-step-invariant-per-layer
type: case
family: runtime-execution
maturity: full
bottleneck: [host-overhead]
signals: [gpu-idle-between-steps, host-prep-on-critical-path]
architectures: [any-sm]
model_scope: [model-agnostic, sparse-attention, mla]
phase: [any-phase]
patterns: [pattern-hoist-step-invariant-host-work]
accuracy_risk: lossless
apply_via_kind: [code-change]
knobs: []
specialists: [perf-host-optimization, perf-host-analysis]
commits: []
interactions:
  - {case: case-hoist-torch-compile-closures, relation: composes-with, note: "same PR (#12581), same host-overhead theme"}
measured: []
---

# Cache step-invariant values across the per-layer attention loop

> Part of the [Runtime / Execution casebook](index.md) · schema: [case-template](../case-template.md)

- **Applies when:** host-overhead-bound serving — the PyExecutor loop is
  CPU-bound, the GPU shows idle gaps between steps, and a function called **once
  per decoder layer** recomputes values that are actually invariant across all
  layers within a forward step. The signal it targets: per-layer attention-prep
  code that re-derives KV-pool views/reshapes, stride factors, and block-table /
  request-index slices that depend only on the pool shape and batch dimensions,
  not the layer index. Confirm host-bound first with `perf-host-analysis`
  (GPU-idle ratio, per-iteration host breakdown) — this only pays off when the
  loop is host-bound. Most acute with many layers × non-trivial per-layer Python/
  CUDA prep. Instance: the DSA indexer's
  `transform_local_topk_and_prepare_pool_view()`.
- **Mechanism:** the per-layer call recomputes step-invariant derived tensors
  (pool `squeeze/view` reshape, `stride_factor = num_layers * tokens_per_block`,
  context/gen block-table and request-index slices) on *every* layer, paying
  redundant Python + CUDA overhead N times. These depend only on the KV-pool
  shape and batch dims, so compute them once per step, cache them on the metadata
  object (`_cached_*`), and reuse across layers; invalidate once per step.
- **Generalizes to:** the pattern "**loop-invariant code motion on the host hot
  path** — hoist any per-layer/per-iteration work whose inputs are constant
  within the step out of the loop and cache it once." Carries to: other per-layer
  metadata prep (RoPE/positional tables, slot-mapping slices, mask precompute),
  per-step-constant reshapes/views, anything keyed off pool shape or batch dims;
  and the same trick in any framework loop over homogeneous layers. Adapt by:
  identifying which derived values are *truly* step-invariant (not
  layer-dependent), caching them on the per-step metadata object, invalidating
  once per step **before** the first per-layer use, and guarding validity with a
  cheap key (here a boolean flag + `id(kv_cache_manager)`). Keep cache fields as
  plain instance attributes (init in `__init__`) so they stay invisible to
  dataclass/`torch.compile` introspection.
- **Apply via:** **not a server config knob** — a host-side code change in the
  attention-metadata class. Delegate optimization to **perf-host-optimization**
  (line_profiler / nsys host rounds) and detection to **perf-host-analysis**.
  Touch points (prior art, TRT-LLM repo):
  `tensorrt_llm/_torch/attention/backends/sparse/dsa/metadata.py` —
  `transform_local_topk_and_prepare_pool_view()` (now reads `_cached_*`),
  `_ensure_pool_view_cached()` / `_invalidate_pool_view_cache()`, and the
  invalidation calls in `prepare()` and `on_update_kv_lens()`.
- **Expected effect:** lower per-layer host time in attention prep → smaller
  GPU-idle gaps between steps and higher throughput when host-bound; no change
  when GPU-bound. PR #12581 reports ~50µs×2 per layer (~4–5ms per forward step)
  of host overhead removed for DSA — treat that as the author's estimate and
  **record the local Δ** (per-iteration host time, GPU-idle ratio, throughput)
  from a host-profiling run; do not present the PR figure as a locally-achieved
  result.
- **Accuracy risk:** lossless — identical values, computed once instead of N
  times. Correctness hinges entirely on the cache being invalidated every step
  before the first per-layer read; a stale cache would silently feed wrong
  slices.
- **Verify:** correctness — DSA e2e / sparse-attention tests still pass; for a
  multi-layer step assert the cached pool view / slices equal the per-layer
  recompute, and re-check after a batch-shape or `kv_cache_manager` change. Perf
  — `perf-host-analysis`: per-iteration host time in attention prep, GPU-idle
  ratio, and throughput before vs after.
- **Rollback:** revert to per-layer recompute; trigger: any correctness
  regression or >5% host-time / throughput regression. Likely failure mode to
  watch: a step path that mutates batch dims or swaps `kv_cache_manager` without
  calling the invalidation → stale cache.
- **Prior art:** PR #12581 ("Multiple host perf
  optimizations for DSA part"); `dsa.py` paths above. Owning skill:
  **perf-host-optimization**; detection: **perf-host-analysis**. Related: the [torch.compile-closure-hoist case](hoist-torch-compile-closures.md) (same PR, same host-overhead theme).
