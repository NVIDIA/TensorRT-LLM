---
id: case-max-utilization-reuse-token-budget
type: regression-case
family: memory-and-capacity
module: kv-cache-manager
maturity: full
regression_class: [capacity-accounting-error, scheduler-batching-regression]
signals: [throughput-drop, kv-capacity-drop, ttft-increase]
subsystems: [kv-cache, scheduler-executor]
introduced_via: [incomplete-coverage]
phase: [prefill]
patterns: [pattern-capacity-accounting-error]
nvbugs: ["6266370"]
commits: ["9635f7d0d096"]
success_prs: [15066]
failed_prs: []
---

# MAX_UTILIZATION reuse token budget under-credits cached blocks, shrinking prefill batches

> Part of the [KV-cache manager regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `6266370` · commit `9635f7d0d096` · PR #15066 —
  "[https://nvbugs/6266370][fix] Fix MAX_UTILIZATION reuse token budget on
  main" (base `main`, merged 2026-06-10). The same fix was written first as
  PR #15065 "[None][fix] Use all reusable blocks for MAX_UTILIZATION token
  budget" against the **`feat/bench_x`** feature branch and closed unmerged
  2026-06-10 — "Skipping merge to `feat/bench_x`. Merged to `main`:
  #15066" — so it is neither a second fix nor a rejected attempt, just the
  feature-branch original. Note #15065's title carries no bug id.
- **Symptom:** Prefill throughput bottlenecked under MAX_UTILIZATION
  capacity scheduling with KV block reuse: the micro batch scheduler
  admitted "smaller batch sizes than expected" because reuse tokens were
  credited with "a very conservative estimate" (PR #15066). The
  feature-branch original #15065 spells out the consequence: one request
  saturates `max_num_tokens` and "context requests serialize to ~1/iter --
  inflating TTFT at high concurrency". No number is stated.
- **Root cause:** In `KVCacheManager::getNeededBlocksOneStep`
  (`cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp`), the estimated
  reusable tokens stored via `req.setEstimatedReusableTokens(...)` for the
  micro batch scheduler's *token* budget were derived from
  `summary.reusableBlocksAllocated` — only cached blocks with active refs.
  Free-but-cached prefix blocks also skip recompute (the engine recovers
  their KV via `prepopulatedPromptLen`), so the token budget under-credited
  reuse and serialized context requests.
- **How introduced:** unknown — not stated in the PR. Git history shows the
  estimated-reusable-tokens mechanism landed in PR #11637 (commit
  `313e274e65`), which credited *all* cached blocks in the
  GUARANTEED_NO_EVICT path (`getRemainingBlocksToCompletion`) but only
  allocated blocks in the MAX_UTILIZATION path — the gap this fix closes.
- **Fix mechanism:** Splits the two budgets explicitly: the block
  (capacity) budget keeps `reusableBlocksAllocated` (a free-but-cached
  block still pulls one block from the free pool when reused, so crediting
  it would double-count the eviction policy's free count and over-admit),
  while the token (compute) budget now uses `summary.reusableBlocksAll`,
  matching the GUARANTEED_NO_EVICT accounting in
  `getRemainingBlocksToCompletion`.
- **Detection signal:** With `capacity_scheduler_policy: MAX_UTILIZATION`
  and block reuse enabled, context batches stay small on cache-hit-heavy
  workloads even with free KV blocks available. The diagnostic is a
  histogram of *requests per context iteration* against the token budget:
  mostly single-request context iterations whose uncached-token count sits
  far below the token budget mean the budget is not what limits admission.
  Cheap A/B: rerun under GUARANTEED_NO_EVICT, whose estimator already
  credits all cached blocks; if the serialization disappears, the
  MAX_UTILIZATION accounting is the difference. Then verify the token
  budget credits all cached blocks with
  `grep -n "setEstimatedReusableTokens" cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp`
  — the `getNeededBlocksOneStep` site must use `reusableBlocksAll`, not
  `reusableBlocksAllocated`.
- **Prevention/guard:** PR #15066 added
  `TEST(KVCacheManagerReuseAccountingTest, NeededBlocksOneStepCreditsFreeCachedReuseInTokenBudget)`
  in `cpp/tests/unit_tests/batch_manager/kvCacheManagerTest.cpp`, pinning
  that free cached blocks count toward the token budget. Gap: no perf-CI
  bar compares admitted context batch size across scheduler policies for a
  reuse-heavy workload, which would have caught the serialization directly.
- **Generalizes to:** `pattern-capacity-accounting-error` — a scheduler
  budget term uses a stricter block state than the runtime actually
  requires. Carries to: any place block *capacity* accounting and token
  *compute* accounting are conflated (they legitimately differ per this
  fix); divergence between MAX_UTILIZATION and GUARANTEED_NO_EVICT
  estimators after a change touches only one; reuse crediting after
  eviction-policy or ref-count semantics change; chunked-prefill token
  budgets that double-count reuse across chunks.
