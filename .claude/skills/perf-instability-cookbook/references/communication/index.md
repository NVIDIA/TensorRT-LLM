# Instability Cookbook — Communication

This module is TRT-LLM's collective layer — the AllReduce path and its
tactic selection under `tensorrt_llm/_torch/distributed/`, including the
NCCL_SYMMETRIC registered-buffer (zero-copy) fast path and the AllReduce
autotuner that sweeps its tactics. Every case here is an **opportunistic fast
collective whose eligibility is decided at runtime**: buffer registration can
fail, is forbidden inside a CUDA-graph capture region, or hits an upstream NCCL
bug on one SKU — and the run then either silently takes plain NCCL (correct,
slower) or does not complete at all. The observable is therefore variance in
tuner selections and downstream perf numbers rather than a stable mean.

First thing to check: grep the run log for NCCL_SYMMETRIC skip / fallback lines
(`grep -nE 'NCCL_SYMMETRIC|graph.capture|register' bench.log`) and establish
*when* registration happens relative to graph capture; then confirm the platform
is not one with a known-bad tactic.

## Recurring patterns in this module

- **Opportunistic collective fallback** — the fast path degrades to plain NCCL
  on a corner case: registration failure, or a lazy first-use registration that
  lands inside a CUDA-graph capture region (where registration is not
  permitted). Correctness survives; the tuner's picks and the measured number
  do not. Check that a fallback path exists *and* that symmetric buffers are
  preallocated before any capture.
  _(Instances: [graceful fallback foundation](nccl-symmetric-graceful-fallback.md),
  [preallocation for autotuning](nccl-symmetric-preallocation-for-autotuning.md).)_
  Two further corner cases of the same pattern — the long-context OOM *hang*
  (PR #11870, nvbug 5930934) and the load-time *segfault* on library
  version mismatch (PR #12015, nvbugs 5923949 / 5803120) — were removed on
  2026-08-12: both are deterministic failures with no varying metric, so they are
  not instability precedents. If you are chasing a hang or a segfault on this
  path they are real failure modes; read PRs #11870 / #12015 directly.
- **Platform-conditional disable** — one platform hits a bug in the upstream
  collective library, and a workload-wide disable would over-solve it. Prefer a
  targeted gate at tactic-selection time. Check the product/SKU string
  (`nvidia-smi -q | grep -E 'Product Name|SKU'`) and then the gate itself
  (`grep -nE 'GB10|Spark|is_gb10' tensorrt_llm/_torch/distributed/`); note that
  on issue #12715 *every* NCCL env knob a reader would reach for first —
  `NCCL_MNNVL_ENABLE=0`, `NCCL_GIN_ENABLE=0`, `NCCL_IB_MERGE_NICS=0`,
  `NCCL_SYMMETRIC_ENABLE=0`, `TRTLLM_ALLREDUCE_STRATEGY=NCCL` — failed to
  resolve the hang, which is why the fix had to be code-side.
  _(Instance: [NCCL_SYMMETRIC off on GB10 Spark](nccl-symmetric-off-on-gb10-spark.md).)_

## Cases

| Case | Symptom (signal) | Class |
|------|------------------|-------|
| [NCCL_SYMMETRIC needs graceful fallback on registration failure and graph capture](nccl-symmetric-graceful-fallback.md) | registration failure had no fallback — error, degraded or hung behaviour depending on the workload; registration inside graph capture always fails | opportunistic-collective-fallback |
| [NCCL_SYMMETRIC autotuning triggered registration inside graph capture](nccl-symmetric-preallocation-for-autotuning.md) | tactics whose lazy first-use registration landed in a capture region fell back to plain NCCL → rep-to-rep variance in tuner selections | opportunistic-collective-fallback |
| [NCCL_SYMMETRIC AllReduce fails on GB10 (DGX Spark)](nccl-symmetric-off-on-gb10-spark.md) | segfault under `AllreduceOp::run` during autotuner warmup, or an indefinite hang whose last line is the MNNVL probe (`finalMNNVL=0`); same config passes on 1.2.0rc6, fails on 1.3.0rc10 | platform-conditional-disable |
