---
id: case-dsa-kcache-dtype-size-estimate
type: regression-case
family: memory-and-capacity
module: kv-cache-manager
maturity: full
regression_class: [capacity-accounting-error]
signals: [kv-capacity-drop, perf-ci-bar-failure]
subsystems: [kv-cache]
introduced_via: [prior-fix-side-effect]
phase: [any-phase]
patterns: [pattern-capacity-accounting-error]
nvbugs: ["6255037"]
commits: ["6254f3a1612c"]
success_prs: [15088]
failed_prs: []
---

# DSA indexer K-cache counted at KV dtype size, shrinking the KV pool ~10%

> Part of the [KV-cache manager regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `6255037` · commit `6254f3a1612c` · PR #15088 —
  "Count DSA indexer K-cache correctly as UINT8 in KV cache size estimate".
- **Symptom:** Reported paged KV cache creation dropped 147.32 → 133.44 GiB
  (~13.9 GiB) on the GLM-5 FP8 DSA perf case
  (`glm_5_fp8-bench-pytorch-float8-...-ep:8-gpus:8`); surfaced via the perf CI
  `test_perf_metric_kv_cache_size` metric. With the fix, primary KV blocks rose
  44708 → 49324 (~10%) (numbers from the PR description).
- **Root cause:** `DSACacheManager` in
  `tensorrt_llm/_torch/attention_backend/sparse/dsa.py` folded the DSA indexer
  K-cache into a `kv_factor` multiplied by the KV-cache dtype size, counting
  the UINT8 indexer pool at 2 bytes/element (BF16). The C++ allocator counts it
  per-pool at 1 byte, so the Python estimate overstated bytes/token
  (110,448 vs 100,152 for the bug's config — a phantom 10,296 B/token) and the
  planner sized the pool for ~10% fewer tokens than actually fit.
- **How introduced:** PR #13745 (commit `13ca44a117`, Gemma4 multi-head_dim
  pools) changed `cacheSizeBytes` in
  `cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp` from the manager-wide
  `mDataType` size to the per-pool dtype size; the DSA Python estimator was not
  updated in tandem and kept the old everything-at-KV-dtype math.
- **Fix mechanism:** `get_cache_size_per_token` and `get_cache_bytes_per_token`
  now add the indexer K-cache as a separate byte term at 1 byte/element
  (raw UINT8, matching `WindowBlockManager::allocatePools`), instead of scaling
  it via a dtype-multiplied `kv_factor`; the FP4 half-width data portion is
  still handled via `indexer_data_dim`.
- **Detection signal:** KV-block count / "paged KV cache" allocation size drops
  across builds with no config change while free GPU memory is unchanged;
  compare `grep "primary blocks" <serve log>` (or the perf-sanity
  `kv_cache_size` metric) between builds, and audit dtype terms in
  `DSACacheManager.get_cache_bytes_per_token`.
- **Prevention/guard:** No new test in the fix PR (validated by re-running the
  perf case locally); the perf CI `kv_cache_size` metric bar is the existing
  guard. Gap: no assert that the Python bytes/token estimate matches the C++
  per-pool allocation math for mixed-dtype cache pools.
- **Generalizes to:** `pattern-capacity-accounting-error` — an estimator term
  goes stale when the allocation layout changes. Carries to: any Python-side
  memory estimator duplicating C++ allocator math (drift after either side
  changes); other mixed-dtype cache pools (FP4/FP8 scale tensors, spec-decode
  hidden-state caches) counted at the wrong element size; per-pool dtype
  refactors that silently invalidate manager-wide-dtype assumptions elsewhere.
