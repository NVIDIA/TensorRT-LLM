# Regression Cookbook — MoE

This module is the Mixture-of-Experts stack: the MoE op and its multiple kernel
backends (CUTLASS, TRITON, TRT-LLM-gen), the router GEMM and its template
instantiations, the expert-parallel all-to-all dispatch/combine path, and the
activation helpers between them. It is the hottest single subsystem in the large
generation-side workloads, so anything added per iteration here — a mask
computation, a missing kernel specialization, an extra activation launch — shows
up directly in per-iteration device step time. First thing to check: which
backend and which kernel instantiations actually ran (the log dumps the backend;
a profile names the kernel), because "the MoE got slower" is usually "a different
kernel ran".

## Recurring patterns in this module

- **Kernel swap regressed** — a kernel or helper is replaced/extended and the new
  path is slower on the shape that matters. Seen twice: a `doActivation`
  refactor whose side effect took the helper from 164,477 ns to 967,955 ns, and a
  routing/backend change carried along with an uplift. Measure the swapped kernel
  in isolation on the target shape, not just end-to-end.
  _(Instances: the doActivation optimization side effect; the a2a rank mask.)_
- **Inactive-feature guard in the hot device loop** — work added to support a
  feature that this run does not use still executes every iteration when the guard
  sits inside the per-iteration path rather than around it. Hoist the check to
  setup, or make the feature-off path skip the work entirely.
  _(Instance: the MoE all-to-all rank mask computed on the hot path.)_
- **Fast path silently fell back** — a missing template instantiation makes the
  dispatcher pick a generic kernel that is functionally right and much slower; the
  profile names it explicitly (`cutlass_80_simt_sgemm_…_align1` is the tell).
  Grep the profile for SIMT / `align1` fallbacks before hunting elsewhere.
  _(Instance: the router GEMM missing instantiation.)_
- **A pinned dependency holds back kernel perf** — some of this module's
  performance lives in an external kernel package, so the in-tree pin decides the
  achievable number and there is no culprit commit in TRT-LLM at all. When a MoE
  gap has no bisectable commit, check the dependency version in the window.
  _(Instance: the triton-kernels 3.6.0 uplift + routing port.)_

## Cases

| Case | Symptom (signal) | Class |
|------|------------------|-------|
| [MoE all-to-all rank mask computed on the hot path](moe-a2a-rank-mask-hot-path.md) | GB200 DEP32 gen-worker per-iter device step time 12.88 → 14.43 ms | kernel-swap-regressed, communication-regression |
| [A doActivation "optimization" side effect made the helper 6× slower](moe-doactivation-optimization-side-effect.md) | doActivation 164,477 ns → 967,955 ns; DS-R1 DEP8 bs=512 B200 TPS 45,741 → 44,597 | kernel-swap-regressed |
| [Router GEMM missing instantiation falls back to a SIMT SGEMM](router-gemm-missing-instantiation.md) | GLM-5 FP8 MTP=3 B200 TP=8: ~5.4%, ~1 ms extra per decode iteration; `cutlass_80_simt_sgemm_64x64_8x5_tn_align1` in the profile | fast-path-fallback |
| [triton-kernels pinned below the 3.6.0 uplift + routing port](triton-kernels-36-uplift-and-routing-port.md) | TRITON MoE backend perf deficit with no culprit commit; fixed by triton 3.5.1 → 3.6.0 | dependency-regression |
