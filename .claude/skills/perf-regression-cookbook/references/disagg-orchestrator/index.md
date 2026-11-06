# Regression Cookbook — Disagg orchestrator

This module is the disaggregated-serving front end: the router/proxy process
that accepts requests and hands them to context and generation workers, plus
the admission policy that decides how fast it fills them. It runs no model code
at all — its whole job is host-side Python — so both regressions recorded here
are host defects that starve otherwise healthy workers. First thing to check on
a disagg throughput drop with unchanged worker-side kernels: whether the router
process itself is the bottleneck (interpreter pauses, per-request work) and
whether the admission cap is sized for the concurrency actually being run.

## Recurring patterns in this module

- **Admission throttle misconfigured** — a fixed fill/admission cap sized for
  one concurrency regime over-throttles another, stretching ramp-up so the
  measured window never reaches steady state. Check the cap against the
  concurrency, not against its original design point.
  _(Instance: the fixed disagg fill-admission cap.)_
- **Host work on the hot path** — the router is a single Python process on the
  critical path of every request, so anything that stalls it (here, garbage
  collection pauses at high concurrency) idles every worker behind it. The
  discriminator is cheap: the same run with GC disabled recovers the number.
  _(Instance: Python GC pauses in the disagg router.)_

## Cases

| Case | Symptom (signal) | Class |
|------|------------------|-------|
| [Fixed disagg fill-admission cap stretches ramp-up at high concurrency](disagg-fill-throttle-slow-start.md) | 7–13% lower output token throughput on gen-only disagg perf-sanity configs (B200 / GB200 / GB300) | scheduler-batching-regression |
| [Python GC pauses in the disagg router starved CTX/GEN workers](disagg-server-gc-pauses.md) | `ctx4_gen1_dep32_batch256_eplb0_mtp1`: 187,466 tok/s with GC vs 240,964 tok/s with GC off (−22%) | host-work-added |
