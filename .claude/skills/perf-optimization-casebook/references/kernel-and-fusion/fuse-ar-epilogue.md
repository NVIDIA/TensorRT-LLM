---
id: case-fuse-ar-epilogue
type: case
family: kernel-and-fusion
maturity: full
bottleneck: [communication, launch, memory]
signals: [allreduce-dominates, many-small-kernels, hbm-roundtrip-between-ops]
architectures: [any-sm]
model_scope: [dense, moe, model-agnostic]
phase: [any-phase]
patterns: [pattern-fuse-chain-feeding-gemm]
accuracy_risk: mixed
apply_via_kind: [config-knob]
knobs: []
specialists: [kernel-cuda-specialist, perf-optimization]
commits: ['14d94a385641', '7b08677c0f15', '6d1f2d0fd700']
eligibility:
  - "TP > 1 with a per-layer allreduce → add → rms_norm → (quant) chain"
  - "dtype fp16/bf16 and AllReduce strategy != UB (UB patterns register only when ub_enabled)"
interactions:
  - {case: case-userbuffers-symmetric-memory, relation: alternative-to, note: UB-backed in-place variant of the same epilogue fusion}
measured: []
---

# Fuse the TP collective epilogue — AllReduce + Residual-add + RMSNorm (+ Quant)

> Part of the [Kernel & Fusion casebook](index.md) · schema: [case-template](../case-template.md)

- **Commits:** `14d94a385641` feat: Add non UB AR + Residual + Norm + Quant fusion (#6320); related: `7b08677c0f15` Add fused allreduce+RMSNorm op + optional residual (#12201), `6d1f2d0fd700` Finalize + Allreduce + add + rmsnorm fusion (#4756, MoE-finalize variant).
- **Applies when:** communication-bound TP layer epilogue where `allreduce` is immediately followed by residual-add → RMSNorm → (optional) quant. Signals: TP>1; per-layer pattern `allreduce → add → rms_norm → static_quantize_e4m3_per_tensor`/`fp4_quantize`; many small kernels + collective launches per layer; UserBuffers unavailable/undesired.
- **Mechanism:** collapses the post-AllReduce residual+norm(+quant) chain into one fused `torch.ops.trtllm.allreduce` carrying an `AllReduceFusionOp`, so residual-add, RMSNorm and quant ride inside the collective kernel's epilogue instead of launching separately — removing intermediate HBM round-trips and per-op launch overhead, emitting the quantized tensor (+scale) directly. Non-UB variant = a `torch.compile` PatternMatcher rewrite (`register_ar_fusions` in `patterns/ar_residual_norm.py`); UB patterns add only when `ub_enabled`.
- **Generalizes to:** "fuse the collective's epilogue into the collective kernel"; carries to TP RMSNorm/LayerNorm epilogues, ReduceScatter+norm, MoE finalize+AllReduce+add+norm, any all-reduce-then-pointwise chain; adapt by choosing the right `AllReduceFusionOp` for the output set, gating on dtype (fp16/bf16) and strategy != UB, and (in-place residual) verifying the residual's last user is the fused node.
- **Apply via:** enable `torch.compile` so the rewrite fires (`register_ar_fusions(custom_passes, ub_enabled)`); ops from `AllReduceFusionOp`: `RESIDUAL_RMS_NORM`, `RESIDUAL_RMS_NORM_QUANT_FP8`, `RESIDUAL_RMS_NORM_QUANT_NVFP4`, `*_OUT_QUANT_*`, plus `MOE_FINALIZE_ALLREDUCE_RESIDUAL_RMS_NORM`. Delegate to **kernel-cuda-specialist** (C++ epilogue) / **perf-optimization**. (UB-backed in-place variant: see the [UserBuffers case](../communication/userbuffers-symmetric-memory.md).)
- **Expected effect:** fewer launches + less memory traffic per TP layer → lower per-token latency, most visible at TP>1 + small batch; no number stated — measured Δ to be recorded from run.
- **Accuracy risk:** AR+residual+norm is lossless (same math, fused order); the `*_QUANT_FP8`/`*_QUANT_NVFP4` epilogues are lossy (quant in-kernel on normed output) — parity-check.
- **Verify:** per-layer kernel count / nsys (fused AllReduce vs separate add/norm/quant); for quant variants run an accuracy eval vs unfused. Tests: `multi_gpu/test_allreduce.py`, `test_user_buffers.py`, `thop/test_moe.py`.
- **Rollback:** disable `torch.compile` (pattern won't register) or restrict to `RESIDUAL_RMS_NORM` (drop quant fusion). Trigger: accuracy regression on quant variants, or UB-availability mismatch.
- **Prior art:** PRs #6320, #12201, #4756. Files: `_torch/compilation/patterns/ar_residual_norm.py`, `compilation/backend.py` (`ub_enabled`), `cpp/.../thop/allreduceOp.cpp`, `kernels/.../moeAllReduceFusionKernels.cu`. Owning specialist: **kernel-cuda-specialist**.
