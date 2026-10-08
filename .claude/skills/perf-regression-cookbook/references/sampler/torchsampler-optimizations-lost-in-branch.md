---
id: case-torchsampler-optimizations-lost-in-branch
type: regression-case
family: execution-and-graph
module: sampler
maturity: full
regression_class: [optimization-lost-in-port]
signals: [host-time-increase, throughput-drop]
subsystems: [sampler]
introduced_via: [unknown]
phase: [decode]
patterns: [pattern-optimization-lost-in-port]
nvbugs: ["5820922"]
commits: ["a5768ce3167b", "57c1ecf12919"]
success_prs: [11315, 11636]
failed_prs: []
---

# TorchSampler host-overhead optimizations lost on a branch

> Part of the [Sampler regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `5820922`. **Two PRs, two branches — cite #11315
  first.** The fix was authored against `release/1.2` and reached `main` only
  as part of a mass integration, so bug→PR and hash→PR lookups disagree:
  - PR #11315 · commit `57c1ecf12919` · base `release/1.2`, merged
    2026-02-11 — `[https://nvbugs/5820922][perf] Improve TorchSampler
    performance by reducing host overhead` (`sampler.py` +22/-8, the only
    file). **This is the nvbug's fix PR.** Its commit is reachable only from
    `origin/release/1.2` and `origin/release/1.2.1`
    (`git branch -r --contains 57c1ecf12919`) — it is nowhere in `main`.
  - PR #11636 · commit `a5768ce3167b` · base `main`, merged 2026-02-24 —
    `[None][chroe] Mass integration of release/1.2 - 5th` (head
    `mi-release-1.2-5`), which is what put the fix on `main`. Titled
    `[None]`, so **no bug id links it**; `gh pr list --search 5820922`
    never returns it.
  Read as: a fix on a release branch is not on `main`, and the carryover that
  lands it drops the bug id. Here the hash in `main`'s history
  (`a5768ce3167b`) belongs to a mass integration spanning many unrelated
  changes, while the hash that actually is this fix (`57c1ecf12919`) cannot
  be found on `main` at all. Per the NVBug, the rerun measured
  `llama_v3.2_1b-bench-pytorch-bfloat16-…-gpus:2` on the base and on target
  commit 57c1ecf, i.e. against the release-branch hash.
- **Symptom:** TorchSampler host overhead higher than it should be — the
  branch was missing main-branch host-time optimizations, so per-iteration
  sampling cost more CPU time (tracked as an NVBug perf issue; no
  quantitative delta stated in the PR).
- **Root cause:** previously-shipped main-branch optimizations were absent:
  per the PR, iterative accesses to `torch.Tensor` objects had to be
  re-replaced with plain lists, and the resolved sampling strategy was being
  recomputed from `SamplingParams` on every iteration instead of being
  computed once per request.
- **How introduced:** the PR states it "Reapplied changes from Main branch",
  i.e. the optimizations were lost on a branch relative to main; the specific
  commit/PR that dropped them is unknown — not stated in the PR.
- **Fix mechanism:** #11315's commit (`57c1ecf12919`) caches the resolved strategy on the
  request object (`request.py_sampling_strategy` in `_request_strategy` of
  `tensorrt_llm/_torch/pyexecutor/sampler.py`), skipping the cache only when
  beam search is used (`_request_sampling_params_cachable`); the PR
  description additionally cites reapplying the list-based replacement of
  iterative `torch.Tensor` accesses.
- **Detection signal:** growing host span around sampling in an nsys trace
  while the same build on main is fine; the decisive check is diffing the hot
  file against main: `git diff main -- tensorrt_llm/_torch/pyexecutor/sampler.py`
  on the regressing branch, or
  `grep -n "py_sampling_strategy" tensorrt_llm/_torch/pyexecutor/sampler.py`
  to confirm the strategy cache is present.
- **Prevention/guard:** none added by the fix (no test coverage listed in the
  PR); the gap is a branch-vs-main perf parity bar for sampler host time, or
  a unit test asserting strategy resolution happens at most once per request.
  Note the fix was deliberately partial: per the NVBug, larger changes were
  kept off the release branch without proper testing this close to the 1.2
  code freeze, and the remaining regressions were to be tracked in a new bug
  scoped to 1.3 — so #11315 is the code-freeze-scoped subset, and the
  residual sampler host overhead moved to a follow-on 1.3 bug (its id is not
  recorded).
- **Generalizes to:** `pattern-optimization-lost-in-port` — a shipped perf
  fix silently missing after a port/branch/refactor; carries to release
  branches that cherry-pick features but not perf fixes, rewrites of a
  sampler/scheduler that re-derive per-request state each step, ports of a
  Python hot path that reintroduce per-element tensor indexing, and refactors
  that drop a memoization attribute because it looks like dead state.
