---
id: case-disagg-fill-throttle-slow-start
type: regression-case
family: execution-and-graph
module: disagg-orchestrator
maturity: full
regression_class: [scheduler-batching-regression]
signals: [throughput-drop, perf-ci-bar-failure]
subsystems: [scheduler-executor]
introduced_via: [prior-fix-side-effect]
phase: [decode]
patterns: [pattern-admission-throttle-misconfigured]
nvbugs: ["6204488"]
commits: ["f20858cc05c5"]
success_prs: [14475]
failed_prs: []
---

# Fixed disagg fill-admission cap stretches ramp-up at high concurrency

> Part of the [Disagg orchestrator regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `6204488` · commit `f20858cc05c5` · PR #14475 —
  Replace fixed disagg fill throttle with slow-start ramp.
- **Symptom:** 7–13 % lower output token throughput (per the fix PR) on
  post-merge perf-sanity gen-only disagg configs — DeepSeek-R1, GPT-OSS, and
  Kimi-K2.5 1k1k on B200 / GB200 / GB300; surfaced via post-merge perf-sanity
  CI.
- **Root cause:** the benchmark-disagg fill-phase admission throttle in
  `PyExecutor._pop_from_waiting_queue` capped new-request admission at a fixed
  `tp_size` per iteration, so at high concurrency filling the server takes
  `con/tp_size` iterations — a long serialized ramp charged to wall-clock
  before steady-state generation.
- **How introduced:** PR #13347 (commit `ef160ad0f2e6`) added the fixed
  `tp_size`/iter cap to stop a preloaded queue from popping every request into
  `DISAGG_GENERATION_INIT` in one iteration, which tripped PR #12206's
  insufficient-KV fail-fast under transient ADP-router imbalance. The cap was
  sized for that repro (Kimi-K2 8k1k con=8192), not for high-concurrency 1k1k
  throughput configs — a prior-fix side effect.
- **Fix mechanism:** replace the fixed cap with a doubling slow-start ramp
  (`_fill_admit_cap` in `py_executor.py`): iter 0 still admits at most
  `tp_size` requests (preserving the iter-0 burst protection), each subsequent
  iter doubles the cap, saturating at `total_max` in
  `ceil(log2(total_max / tp_size))` iterations (~10 iters for con=4096, tp=8).
  The cap resets to 0 when the fill phase ends.
- **Detection signal:** gen-only disagg benchmark throughput drops while
  steady-state per-iteration behavior is unchanged — the loss is in the fill
  ramp; inspect the throttle with
  `grep -n "_fill_admit_cap\|_benchmark_fill_phase_active" tensorrt_llm/_torch/pyexecutor/py_executor.py`
  and check whether admitted-request count grows per iteration or stays flat.
- **Prevention/guard:** PR #14475 added
  `tests/unittest/_torch/executor/test_benchmark_disagg.py::TestBenchmarkFillAdmissionFlowControl`
  pinning the ramp shape (iter-0 cap, doubling, O(log2) saturation, reset,
  warmup bypass); the integration test
  `test_disaggregated_benchmark_gen_only_insufficient_kv` keeps the original
  #13347 repro green. Gap: only post-merge perf-sanity caught the wall-clock
  cost — no pre-merge perf check covers fill-ramp duration.
- **Generalizes to:** `pattern-admission-throttle-misconfigured` — a fixed
  admission/fill cap sized for one concurrency regime over-throttles another;
  carries to any scheduler rate limiter with a hard per-iteration constant
  (waiting-queue pop limits, KV-transfer pacing, warmup request feeding), to
  hang/OOM fixes that clamp batching aggressively instead of adaptively, and
  to connection/stream ramp-up logic in serving frontends where slow-start
  beats fixed windows.
