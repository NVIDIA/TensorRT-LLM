---
id: case-nccl-symmetric-preallocation-for-autotuning
type: instability-case
family: communication-negotiation
module: communication
maturity: full
instability_class: [opportunistic-collective-fallback]
signals: [autotuner-cache-miss-during-capture, eager-fallback-in-log]
subsystems: [communication, autotuner]
introduced_via: [prior-fix-side-effect]
phase: [warmup]
patterns: [pattern-opportunistic-collective-fallback]
nvbugs: []
commits: ["5130cbd73e9f"]
success_prs: [11326]
failed_prs: []
---

# NCCL_SYMMETRIC autotuning triggered registration inside graph capture — preallocate instead

> Part of the [Communication instability cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** commit `5130cbd73e9f` · PR #11326 — Pre-Allocation for
  Auto-Tuning NCCL_SYMMETRIC. Follow-up to
  [case-nccl-symmetric-graceful-fallback](nccl-symmetric-graceful-fallback.md)
  (#11042).
- **Symptom (variance signature):** with #11042's graceful fallback in
  place, the AllReduce autotuner still exercised NCCL_SYMMETRIC tactics
  whose buffer registration could land inside a CUDA-graph capture region
  during the same tuning window — each of those tactics fell back to
  plain NCCL, producing rep-to-rep variance in tuner selections and
  measurement noise for downstream perf tests.
- **Root cause:** the autotuner allocates and registers symmetric buffers
  lazily on first use per tensor size, and its "first use" for a given
  size sometimes coincided with graph capture — during which registration
  is disallowed (see #11042).
- **How introduced:** NCCL_SYMMETRIC was integrated into the AllReduce
  autotuner without a pre-registration step; the tuner effectively raced
  against graph capture.
- **Fix mechanism:** pre-allocate symmetric buffers for the tensor sizes
  the AllReduce autotuner will exercise, *before* graph capture. The
  overall memory footprint is unchanged (the allocation is only moved
  earlier); the pre-allocated buffers are also reused during the regular
  run, avoiding extra allocations later.
- **Detection signal:** autotuner tactic list where NCCL_SYMMETRIC tactics
  fell back to plain NCCL during tuning; per-size registration timestamps
  landing inside a graph-capture nvtx range —
  `grep -n 'NCCL_SYMMETRIC\|pre_alloc\|symm_buffer' tensorrt_llm/_torch/distributed/`.
- **Prevention/guard:** any tuner that exercises tactics with resource-
  registration side effects must complete registration before any graph
  capture; a review checklist for adding tactics should include this.
- **Generalizes to:** `pattern-opportunistic-collective-fallback`; carries
  to any tuner sweep of tactics that allocate on first use (cutlass
  workspaces, DeepGEMM buffers, symmetric-memory pools) racing with
  graph capture / warmup gating.
