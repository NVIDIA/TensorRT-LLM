# Regression Cookbook — Autotuner

This module is the autotuner: `tensorrt_llm/_torch/autotuner.py` and the
`AutoTuner`-aware op wrappers that consult it to choose a tactic for a shape.
Its cost is almost entirely **host-side** — a cache lookup and a dispatch
decision taken on the way to a kernel that is itself unchanged — so regressions
here show up as host time or GPU idle between steps while an nsys kernel diff
finds nothing. First thing to check: how many autotuner lookups happen per op
call, and whether an op that gained a wrapper is now tuned *inside* another
tuned op.

## Recurring patterns in this module

- **Host work on the hot path** — every layer of tuning wrapping adds a cache
  lookup per call, and nesting a tuned op inside a tuned op pays for both. The
  kernels do not change, so compare host spans across builds rather than looking
  for a new kernel; and beware windows where several changes move the number,
  which read as a staircase of partial recoveries rather than one step.
  _(Instance: the unified NVFP4 GEMM's two-level nested autotuning.)_

## Cases

| Case | Symptom (signal) | Class |
|------|------------------|-------|
| [Unified NVFP4 GEMM: two-level nested autotuning paid a double cache lookup per call](nested-nvfp4-gemm-autotune-dispatch.md) | MLPerf Llama3.1-405B TP2PP2 on GB200/B200 −8%, 182 → 172 tps/gpu; bisect is a staircase across five commits | host-work-added |
