---
id: case-model-engine-warmup-orchestration-gaps
type: regression-case
family: execution-and-graph
module: jit-and-warmup
maturity: full
regression_class: [warmup-jit-gap, memory-footprint-regression]
signals: [startup-time-increase, memory-usage-increase, perf-ci-bar-failure]
subsystems: [runtime-python, moe, autotuner]
introduced_via: [incomplete-coverage]
phase: [any-phase]
patterns: [pattern-warmup-coverage-gap, pattern-unaccounted-startup-residency]
nvbugs: ["5963665"]
commits: ["a0b53e66a6f3"]
success_prs: [12407]
failed_prs: []
---

# ModelEngine warmup: gated on torch.compile, missing the (1,0) shape, and holding MoE workspaces into autotuning

> Part of the [JIT & warmup regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `5963665` · commit `a0b53e66a6f3` · PR #12407 —
  "[https://nvbugs/5963665][refactor] Refactor warmup orchestration in
  ModelEngine".
- **Symptom:** GB200 disagg `ctx_only` perf regressions in the L0 post-merge
  perf CI on `main`. **The PR states no symptom and no number** —
  it is filed as a refactor. Read the diff, not the title: the refactor fixes
  three independent defects, and a reader who trusts the `[refactor]` tag will
  skip the case entirely.
- **Root cause:** three defects in one warmup path, all in
  `tensorrt_llm/_torch/pyexecutor/model_engine.py`.
  **(1) Warmup gated on torch.compile.** `_run_torch_compile_warmup` opened with
  `if not self._torch_compile_enabled: return` — so with torch.compile **off**
  there was no general warmup at all, and every first-touch cost (kernel
  autotune-adjacent allocation, lazy init, first-shape specialization) landed in
  the first served iterations. The function's *name* is why this survived review:
  it reads as "warmup for torch.compile", and the body was in fact the only
  general warmup.
  **(2) The `(1, 0)` shape was never warmed.** The warmup shapes were held in an
  unordered `set` of four `(num_tokens, num_gen_tokens)` tuples and iterated
  sorted by magnitude; `(1, 0)` — one token, no generation tokens, i.e. the
  minimal context shape — was absent. A ctx-only disagg worker runs exactly that
  regime.
  **(3) C++ MoE workspaces stayed resident into autotuner warmup.**
  `FusedMoeRunner` allocated its workspaces during warmup and never released
  them, so the autotuner probed free memory with those workspaces still held.
- **How introduced:** `incomplete-coverage` — each defect is a hole in a warmup
  grid / lifecycle that was correct for the configuration it was written for
  (torch.compile on; larger shapes; a single warmup consumer).
- **Fix mechanism:** split warmup **configuration** from warmup **execution**, and
  make the shape list explicit and ordered:
  `[(1,0), (1,1), (curr_max_num_tokens,0), (max_batch_size,max_batch_size), (2,0)]`
  deduped with `list(dict.fromkeys(...))` (order-preserving, unlike the old `set`),
  run under a `can_run_general_warmup` predicate — **not** under the torch.compile
  flag — and inside `with self.no_cuda_graph()`. Then release the workspaces before
  autotuning: a new `MoERunner.clear_all_workspaces()` (C++
  `FusedMoeRunner::clearWorkspaces()` under `mMutex`, `moeOp.cpp` +11/−1, exposed
  via `torch_custom_ops.py` +12) followed by `gc.collect()` and
  `torch.cuda.empty_cache()`. `model_engine.py` is +77/−43.
- **Detection signal:** the log line changed, and the *old* line is the pre-fix
  marker: `grep -n 'Running torch.compile warmup' <log>` — present ⇒ pre-fix
  (the PR **deletes** that `logger.info`). Post-fix the shapes are announced
  individually, so `grep -n 'Run warmup with 1 tokens, include 0 generation
  tokens' <log>` confirms the `(1,0)` shape is actually warmed. For defect (3),
  compare reported free memory / KV pool size at autotune time across builds.
- **Prevention/guard:** **no test was added** for any of the three. Two rules
  worth carrying: a warmup routine's gate must be "can we warm up?", never "is
  feature X enabled?" — name the function for what it warms, not for the feature
  that motivated it; and a warmup shape grid should be an explicit ordered list
  including the *degenerate* shapes (1 token, 0 generation tokens), because a
  magnitude-sorted `set` hides which shapes are missing.
- **Generalizes to:** `pattern-warmup-coverage-gap` (defects 1 and 2) and
  `pattern-unaccounted-startup-residency` (defect 3 — memory held across the
  free-memory probe comes off KV capacity 1:1). Carries to any engine whose
  warmup, autotune and capacity-sizing steps run in one sequence: audit what is
  *resident* at each hand-off, and audit the shape grid for the smallest legal
  shape, not just the largest.
