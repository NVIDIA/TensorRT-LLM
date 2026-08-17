---
id: case-mnnvl-twoshot-allreduce
type: case
family: communication
maturity: full
bottleneck: [communication, sync]
signals: [allreduce-dominates, kernel-far-from-sol]
architectures: [sm90, sm100]
model_scope: [model-agnostic, dense, moe, deepseek-v3r1]
phase: [any-phase]
patterns: [pattern-tune-collective-kernel-not-algorithm]
accuracy_risk: lossless
apply_via_kind: [config-knob, kernel-change]
knobs: [allreduce_strategy]
specialists: [perf-nsight-compute-analysis, kernel-cuda-specialist]
commits: ['6e1aee6fd68a']
eligibility:
  - "num_token*token_dim divisible by ELTS_PER_LOAD (4 for BF16, 2 for FP32), as of #5934"
measured: []
---

# Optimize the collective kernel itself (two-shot / Lamport-sync MNNVL AllReduce)

> Part of the [Communication casebook](index.md) · schema: [case-template](../case-template.md)

- **Commits:** `6e1aee6fd68a` [fix] Performance Optimization for MNNVL TwoShot Kernel (#5934).
- **Applies when:** communication-bound + signals: TP AllReduce on a multi-node NVLink (MNNVL) domain; medium/large messages where two-shot (reduce-scatter + allgather) beats one-shot; `allreduce_strategy=MNNVL` or `TWOSHOT` already selected and the AR kernel (not the network) is the bottleneck.
- **Mechanism:** tunes the MNNVL two-shot AllReduce CUDA kernel rather than swapping algorithms. Lamport-style flag sync with triple-buffered flags (`buffer_flags % 3`), grid reworked so only a subset of CTAs do the lamport sync; cross-CTA completion via `red.async.release.global.gpu.add.u32` atomics + spin on `offset_access_ptr < gridDim.x*gridDim.y`; data via vectorized `ld.volatile.global.v4.f32`/`v2.f32` over NVLink-multicast buffers (`mcastDeviceMemory`), assuming 8B-atomic writes and `num_token*token_dim` divisible by ELTS_PER_LOAD (4 BF16 / 2 FP32).
- **Generalizes to:** "make the collective kernel faster instead of changing the algorithm" — buffer-flag/Lamport sync, grid-rightsizing, vectorized multicast loads carry to one-shot AllReduce, allgather/reduce-scatter, custom NVLS/multimem collectives; adapt by matching alignment/divisibility assumptions and the flag lifecycle.
- **Apply via:** `allreduce_strategy=MNNVL` (or `TWOSHOT`) in `_torch/model_config.py` (`AllReduceStrategyType`: NCCL=0, MIN_LATENCY=1, UB=2, AUTO=3, ONESHOT=4, TWOSHOT=5, LOWPRECISION=6, MNNVL=7, NCCL_SYMMETRIC=8). Delegate kernel profiling to **perf-nsight-compute-analysis**; kernel edits to **kernel-cuda-specialist**.
- **Expected effect:** lower MNNVL TwoShot AllReduce kernel latency at targeted message sizes; no number — measured Δ to be recorded from run.
- **Accuracy risk:** lossless (same reduction math/precision; only sync, grid layout, vectorized access changed).
- **Verify:** AllReduce kernel time (ncu/nsys) + end-to-end TP latency vs pre-patch; numeric parity of reduced output across ranks; respect alignment/divisibility preconditions.
- **Rollback:** switch `allreduce_strategy` to `AUTO`/`NCCL` or revert the kernel commit. Trigger: correctness/sync failure or regression on the target shapes.
- **Prior art:** PR #5934. Files: `cpp/.../kernels/communicationKernels/mnnvlTwoShotAllreduceKernels.cu`, `runtime/mcastDeviceMemory.{h,cpp}`, `modeling_deepseekv3.py`, `kernels/customAllReduceKernels.h`. Owning specialist: **kernel-cuda-specialist**.
