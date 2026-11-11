# Regression Cookbook — Scheduler & executor

This module is the request-loop layer above the model: the executor's per-iteration
bookkeeping (request queues, cancellation sets, per-request attribute handling),
the ModelEngine's per-request preparation, and the overlap scheduler's decision
about when a first token may be emitted. Everything here is host code on the
critical path of every iteration, so the two failure shapes are (a) per-iteration
host work that grows with batch size or with run length, and (b) a batching /
emission decision that delays a token without making anything slower. First thing
to check: per-iteration host time versus device step time — if device time is flat
and the gap grew, the defect is in this module, not in a kernel.

## Recurring patterns in this module

- **Host work on the hot path** — per-request or per-iteration Python work in the
  executor loop, paid before any kernel launches. Two shapes seen: per-request
  attribute handling in ModelEngine preparation, and a collection that is appended
  to and never pruned, so the cost *grows over the run* — a slope, not a level,
  which a short probe reproduces poorly and a fixed-duration bar can miss entirely.
  For the growing kind, compare early versus late iterations within one run rather
  than run-to-run means.
  _(Instances: the unbounded canceled-request-id growth; the ModelEngine
  per-request attribute overhead.)_
- **Delayed first-token emission** — the scheduler holds a token it already has,
  so TTFT rises with no change in throughput and no slow code anywhere. Look for a
  gating condition on when a response is released, and for the knob that restores
  the earlier behavior, before profiling.
  _(Instance: the overlap scheduler first-token delay, gated by
  `enable_early_first_token_response`.)_

## Cases

| Case | Symptom (signal) | Class |
|------|------------------|-------|
| [Canceled request ids grow unbounded in the executor loop](canceled-req-ids-unbounded-growth.md) | gradual slowdown of a long-running server; degradation is a slope, and the case states no magnitude | host-work-added |
| [ModelEngine per-request attribute overhead in decode preparation](model-engine-per-request-attr-overhead.md) | gpt-oss-120b gen-only `mean_gen_worker_per_iter_device_step_time` 8.384 → 8.691, 7.869 after the fix | host-work-added |
| [Overlap scheduler delays the first token](overlap-scheduler-first-token-delay.md) | TTFT elevated with throughput unchanged; magnitude not quantified in the PR | scheduler-batching-regression |
