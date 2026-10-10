---
id: case-allreduce-strategy-lut
type: regression-case
family: communication
module: communication
maturity: full
regression_class: [communication-regression, kernel-selection-regression]
signals: [throughput-drop, itl-increase, slower-kernel-in-trace]
subsystems: [communication, runtime-cpp]
introduced_via: [pre-existing-gap]
phase: [any-phase]
patterns: [pattern-selection-heuristic-too-coarse]
nvbugs: ["5540494", "5531839", "5517023"]
commits: ["fd4311e6a396"]
success_prs: [7870]
failed_prs: []
---

# AllReduce AUTO keyed only on token count, so it picked a one-shot kernel where NCCL was faster

> Part of the [Communication regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbugs `5540494`, `5531839`, `5517023` · commit `fd4311e6a396` ·
  PR #7870 — replaces the AUTO heuristic with a measured best-strategy table.
- **Symptom:** two directions, both in the PR. Regression: "perf regression due to
  using a one-shot kernel instead of NCCL on A100/H100" — AUTO chose a custom
  kernel that lost to plain NCCL on those SMs. Upside on the fixed path: "3-4 %
  perf gain in concurrencies of 256 and 512" on Deepseek-R1. Three bugs collapse
  into one case because they are one root cause and one fix PR.
- **Root cause:** AUTO decided from **token count alone** — a single threshold
  (default 128, overridable) plus a `world_size <= 2` special case — and below the
  threshold always returned `MIN_LATENCY`, i.e. the one-shot custom kernel. The
  inputs it ignored are exactly the ones that determine the crossover:
  `hidden_size` (message bytes per token), the **fusion op** attached to the
  allreduce (which changes how much work the custom kernel is amortising), and the
  **SM architecture** (the custom kernels' advantage over NCCL is
  generation-dependent, and on A100/H100 it can be negative). A one-dimensional
  heuristic over a five-dimensional crossover surface is right in the region it was
  tuned on and wrong everywhere else — and because it is *deterministic*, it is
  wrong reproducibly, which reads as a product regression rather than a tuning gap.
- **How introduced:** `pre-existing-gap` — no culprit commit. The heuristic was
  always this coarse; each bug is a workload that walked into a region where it
  mispredicts.
- **Fix mechanism:** a generated lookup table,
  `AllReduceBestStrategyTable[sm][tp][fusion_op][hidden_size][num_tokens]`,
  populated from measurement for SM90 and SM100, with **out-of-range ⇒ NCCL** as
  the safe default. Explicit user requests for `ONESHOT` / `TWOSHOT` are still
  honoured (AUTO is the only path that changes). The kernels themselves also move
  from stronger orderings to `relaxed` stores plus a single
  `fence.release.sys`. What the PR **removes** matters as much: a `TORCH_CHECK`,
  three `TLLM_LOG_WARNING` fallback paths, and the env overrides
  `OVERRIDE_HEURISTIC_ALLREDUCE_STRATEGY` and
  `ALLREDUCE_AUTO_HEURISTIC_MIN_LATENCY_THRESHOLD_TOKEN_NUM` — so a runbook or
  script that sets either variable is silently a no-op after this commit.
- **Detection signal:** the log line was renamed, which dates any log precisely:
  `grep -n "AllReduceOp runtime strategy for rank" <log>` is post-fix,
  `grep -n "AllReducePlugin strategy for rank" <log>` is pre-fix. Diagnose a
  suspected misprediction by pinning the strategy — run the same case with
  `ONESHOT`, `TWOSHOT` and `NCCL` explicitly; if an explicit choice beats AUTO, the
  table (or its coverage for your `sm`/`tp`/`hidden_size`) is the problem, not the
  kernels. Beware the corollary: the table covers **SM90 and SM100 only**, so on
  other architectures the OOB fallback lands on NCCL by design.
- **Prevention/guard:** no unit test asserts a table entry; the guard is
  structural — the table is generated from measurement, and unknown regions
  degrade to NCCL rather than to a guess. Rule: a selection heuristic must fall
  back to the **portable** implementation outside its measured envelope, and
  removing env overrides removes the field escape hatch, so the fallback has to be
  right.
- **Generalizes to:** `pattern-selection-heuristic-too-coarse`; carries to MoE
  backend selection, attention-backend choice, GEMM tactic heuristics and
  `all_reduce`/`all_gather` strategy pickers — anywhere a threshold on one
  dimension stands in for a multi-dimensional crossover. See also
  `case-allreduce-host-overhead-small-model-tp` in this family for the
  *host-overhead* failure mode of the same subsystem.
