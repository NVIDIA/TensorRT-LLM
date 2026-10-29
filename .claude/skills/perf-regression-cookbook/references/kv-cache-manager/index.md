# Regression Cookbook — KV-cache manager

This module is the KV-cache manager and the state caches beside it: the block
pool and its size estimators, the reuse/prefix-cache accounting that credits
cached blocks back to a scheduling budget, and the recurrent/SSM state caches
used by mamba-hybrid models. Two kinds of defect recur. Either an **estimator
term** goes stale after a cache-layout change — a dtype size, a layer count, a
budget split — so the planner believes there is less memory than there is and
fewer blocks or smaller batches fit; or the cache's *bookkeeping* grows onto the
per-step path, adding host work and syncs to decode preparation. First thing to
check: the reported KV-block count / pool size across builds at identical config.
The perf-sanity `kv_cache_size` metric exists for exactly this, and a capacity
defect leaves per-step GPU time flat.

## Recurring patterns in this module

- **Capacity accounting error** — dtype-size, layer-count or budget-split terms
  in an estimator go stale when the layout they describe changes, or a
  conservative credit under-counts what reuse already provides. Audit every
  estimator term when the layout changes, and assert estimates against actual
  allocation.
  _(Instances: the DSA indexer K-cache counted at KV dtype size; the
  MAX_UTILIZATION reuse token budget. Two spec-decode siblings — EAGLE3 draft KV
  over-allocated by `max_draft_len×` (PR #15017) and host KV budget
  double-counted with offloading (PR #13130) — were removed on 2026-08-12
  because their PRs describe memory over-allocation fixes and state no perf
  symptom. They are real accounting defects of exactly this shape; read those
  PRs directly if you are auditing spec-decode KV sizing, because the
  surviving cases cover dtype-size and reuse-budget terms only.)_
- **Per-step sync added** — cache management runs inside decode preparation, so
  a per-slot device→host read added there is paid every iteration. The trace
  shows GPU idle waiting on the host with kernel durations unchanged.
  _(Instance: mamba-hybrid prefix caching adding per-slot D2H syncs.)_
- **Host work on the hot path** — the same case's Python-side per-slot loop is
  host cost in its own right, independent of the sync.
  _(Instance: mamba-hybrid recurrent-state D2H syncs.)_
- **Fast path silently fell back** — the *dtype* a state cache is allocated in is
  a kernel-eligibility input: allocating in fp32 made the bf16-state decode
  kernel ineligible, so the cache decided which kernel ran.
  _(Instance: the Qwen3.5 GDN state cache allocated in fp32.)_

## Cases

| Case | Symptom (signal) | Class |
|------|------------------|-------|
| [DSA indexer K-cache counted at KV dtype size, shrinking the KV pool](dsa-kcache-dtype-size-estimate.md) | paged KV cache 147.32 → 133.44 GiB on GLM-5 FP8; primary blocks 44708 → 49324 (~10%) with the fix | capacity-accounting-error |
| [Mamba-hybrid prefix caching added per-slot D2H syncs to decode prep](mamba-hybrid-recurrent-state-d2h-syncs.md) | `numGenReq × (1 + max_draft_len)` memcpy+sync pairs per iteration in `_prepare_inputs` (PR #14003 nsys) | sync-introduced, host-work-added |
| [MAX_UTILIZATION reuse token budget under-credits cached blocks](max-utilization-reuse-token-budget.md) | prefill batches smaller than expected with block reuse on; context requests serialize to ~1/iter, inflating TTFT (#15065) | capacity-accounting-error, scheduler-batching-regression |
| [Qwen3.5 GDN state cache allocated in fp32, disabling the bf16-state decode kernel](qwen35-ssm-cache-fp32-fallback.md) | ~20% serving throughput loss (fix diff's docstring) | fast-path-fallback |
