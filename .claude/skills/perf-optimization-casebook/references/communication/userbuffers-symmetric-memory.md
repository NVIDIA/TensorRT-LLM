---
id: case-userbuffers-symmetric-memory
type: case
family: communication
maturity: full
bottleneck: [communication, launch, memory]
signals: [allreduce-dominates, hbm-roundtrip-between-ops]
architectures: [sm90, sm100]
model_scope: [model-agnostic, dense, moe]
phase: [any-phase]
patterns: [pattern-preregister-symmetric-comm-buffers]
accuracy_risk: mixed
apply_via_kind: [config-knob]
knobs: [allreduce_strategy]
specialists: [perf-torch-cuda-graphs, perf-host-optimization]
commits: ['dca6397d1e2d']
eligibility:
  - "AR input dtype FP16/BF16 (UB extra_check gate)"
interactions:
  - {case: case-fuse-ar-epilogue, relation: composes-with, note: "UB registration is the in-place backing that enables the fused AR+RMSNorm(+quant) variant"}
measured: []
---

# UserBuffers / symmetric memory — pre-register comm buffers to enable AllReduce+quant fusion

> Part of the [Communication casebook](index.md) · schema: [case-template](../case-template.md)

- **Commits:** `dca6397d1e2d` feat: Introduce UB allocator for pytorch flow (#3257).
- **Applies when:** communication-bound + signals: TP AllReduce in the PyTorch flow where AR is followed by residual-add + RMSNorm (+ optional FP8/FP4 quant); per-step comm-buffer allocation or an unfused AR→norm→quant chain shows up; want a symmetric-memory AllReduce path. UB requires FP16/BF16 AR input.
- **Mechanism:** a global `UserBuffersManager` singleton hands out pre-registered symmetric UserBuffers tensors (`create_userbuffers_tensor` op, `allocate_userbuffers`, `search_buffer`). Buffers reserved at warm-up via `initialize_userbuffers_manager(tp_size, ...)` so "there is no dynamic allocation during inference." This registration unlocks the fused UB AllReduce path: `ub_allreduce.py` rewrites the graph to fuse quant into the AllReduce as an in-place UB impl (`copy_to_userbuffers`), convert supported AllReduces to UB, fuse the producer to write directly into the userbuffer, and drop `userbuffers_allreduce_finalize` when chained — running `allreduce(..., strategy=UB, fusion=RESIDUAL_RMS_NORM_QUANT_FP8)` via launchers like `allreduce2_userbuff_inplace_rmsnorm_quant_fp4_launcher`.
- **Generalizes to:** "pre-register symmetric comm buffers to enable in-place fused collectives + kill per-step allocation"; carries to NCCL symmetric AllReduce, AR+RMSNorm fusion, AR+quant (FP8/FP4) fusion; adapt by routing producers to write into the registered buffer and gating on dtype support.
- **Apply via:** `allreduce_strategy=UB` (`AllReduceStrategyType::UB=2`); UB manager auto-inits at warm-up; AR+RMSNorm(+FP8/FP4 quant) fusion applied by the torch-compile UB pattern pass (`register_ub_patterns`). Delegate to **perf-torch-cuda-graphs** / **perf-host-optimization** for capture + host-overhead validation.
- **Expected effect:** lower AllReduce+norm(+quant) latency via in-place fusion + no per-step buffer allocation; no number — measured Δ to be recorded from run.
- **Accuracy risk:** mixed — UB AllReduce+RMSNorm fusion is lossless; the FP8/FP4-quant fusion variants are lossy (emit FP8/FP4) and need an accuracy record + rollback. UB AR requires FP16/BF16 input (`extra_check` gate).
- **Verify:** AR+norm latency + step time vs unfused/NCCL; for quant-fusion variants, accuracy/parity vs unfused FP8/FP4; `multi_gpu/test_user_buffers.py`.
- **Rollback:** `allreduce_strategy` → `AUTO`/`NCCL` (disables UB fusion). Trigger: capture/registration failure or accuracy regression on the quant-fused variant.
- **Prior art:** PR #3257. Files: `cpp/.../kernels/userbuffers/userbuffersManager.{h,cpp}`, `thop/userbuffersTensor.cpp`, `thop/allreduceOp.cpp`, `_torch/compilation/patterns/ub_allreduce.py`, `pyexecutor/model_engine.py`. Related: the [AR-epilogue fusion case](../kernel-and-fusion/fuse-ar-epilogue.md) (UB is the in-place backing for it).
