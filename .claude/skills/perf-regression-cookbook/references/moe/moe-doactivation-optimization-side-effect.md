---
id: case-moe-doactivation-optimization-side-effect
type: regression-case
family: kernel-and-fusion
module: moe
maturity: full
regression_class: [kernel-swap-regressed]
signals: [slower-kernel-in-trace, throughput-drop]
subsystems: [moe]
introduced_via: [kernel-change, incomplete-coverage]
phase: [any-phase]
patterns: [pattern-kernel-swap-regressed]
nvbugs: ["5799917"]
commits: ["7bb371553fa3"]
success_prs: [11165]
failed_prs: []
---

# CUTLASS MoE doActivation optimization for FP8/W4A8 regressed NVFP4/MXFP4

> Part of the [MoE regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `5799917` · commit `7bb371553fa3` · PR #11165 —
  "Recover from CUTLASS MoE doActivation perf regression for MXFP4/NVFP4
  dtype"; related: nvbug `5799917` · commit `aa5e1d9fe8d1` · PR #11733 — same
  subject, cherry-pick onto `release/1.2`.
  **Branch pair — expect two PR numbers for this one defect.** A
  bug→PR lookup on 5799917 returns *two* merged PRs with the same title and
  the same single changed file: #11165 (`baseRefName=main`, merged
  2026-02-26 04:16Z, mergeCommit `7bb371553fa3`) and #11733
  (`baseRefName=release/1.2`, merged the same day at 08:00Z, mergeCommit
  `aa5e1d9fe8d1`). Only `7bb371553fa3` is on `main`, so that is the commit
  recorded here; `aa5e1d9fe8d1` exists solely on the 1.2 release branch.
  Do not substitute one hash for the other, and do not read the pair as two
  fix attempts.
- **Symptom:** After PR #9838, the `doActivationKernel` in the CUTLASS MoE
  path got drastically slower for NVFP4/MXFP4 output in end-to-end runs:
  on DS-R1 DEP8 bs=512 (B200), doActivation average went 164,477 ns →
  967,955 ns and total TPS with trtllm-bench dropped 45,741 → 44,597
  (numbers from the PR #11165 tables). Isolated kernel benchmarks still
  showed a *gain*, so only e2e traces exposed the regression.
- **Root cause:** PR #9838 optimized doActivation for FP8/W4A8 on Hopper but
  overlooked the NVFP4/MXFP4 variant, where a CTA may or may not perform
  scale-factor (SF) padding depending on its blockIdx. The rewrite dropped
  the grid-stride loop (one CTA per token chunk → too many short-running
  CTAs, load imbalance) and instantiated four `if constexpr` copies of the
  main loop (prequant_scale × bias), bloating kernel binary size so e2e
  runs stalled on "No Instructions" (instruction-fetch) — which is why the
  isolated benchmark saw the gain but e2e did not.
- **How introduced:** PR #9838 (commit `9cae7277ea9c`, "Apply fusion for
  W4AFP8_AWQ MoE"), named explicitly in the fix PR description.
- **Fix mechanism:** In
  `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu`:
  re-introduce a grid-stride loop over tokens and cap `grid_x` at
  `sm_count * max_blocks_per_sm` (via `getMultiProcessorCount` /
  `getMaxActiveBlocksPerSM`, plus `__launch_bounds__`); shrink binary size
  by replacing the four compile-time main-loop instantiations with runtime
  `bias_ptr`/`prequant_scale` branches; handle the NVFP4/MXFP8 N-dim SF
  padding with a second grid-stride loop over the same grid instead of
  dedicated extra blocks beyond `num_valid_tokens`.
  Result: doActivation avg 133,009 ns and TPS 49,879 (1.09x over the
  pre-#9838 baseline).
- **Detection signal:** In an nsys trace, `doActivationKernel` duration
  balloons vs the previous build while GEMMs are unchanged; e2e TPS drops
  although isolated kernel microbenchmarks of the same kernel look fine.
  Check with `nsys stats --report cuda_gpu_kern_sum report.nsys-rep | grep
  doActivation` and compare avg/max against the good build; in ncu, look
  for a high "No Instructions" warp-stall fraction on the kernel.
- **Prevention/guard:** PR #11165 lists no new test (Test Coverage empty).
  Gap: kernel micro-optimizations validated only in isolation and only on
  the target dtype/arch — a guard would be a per-dtype (FP8, W4A8, NVFP4,
  MXFP4) doActivation perf check plus an e2e trace diff, since binary-size
  stalls only manifest under real instruction-cache pressure.
- **Generalizes to:** `pattern-kernel-swap-regressed` — an optimization for
  one dtype/shape regime regresses another it also serves. Carries to: any
  shared template kernel where one dtype's fast path removes a grid-stride
  loop others rely on; `if constexpr` fan-out that bloats binary size and
  stalls e2e but not microbenchmarks; SF-padding/quantized variants of a
  kernel skipped during validation of a rewrite; occupancy-tuned launches
  that assume one arch's SM count.
