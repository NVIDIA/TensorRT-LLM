---
id: case-fuse-add-norm-quant
type: case
family: kernel-and-fusion
maturity: full
bottleneck: [memory, launch]
signals: [many-small-kernels, hbm-roundtrip-between-ops]
architectures: [any-sm]
model_scope: [llama-family, model-agnostic]
phase: [any-phase]
patterns: [pattern-fuse-chain-feeding-gemm]
accuracy_risk: lossy
apply_via_kind: [code-change, config-knob, env-var]
knobs: [TRTLLM_DISABLE_NVFP4_LAYERNORM_FUSION]
specialists: [kernel-cuda-specialist, kernel-triton-specialist]
commits: ['211c44b95199', '72cd7d824bb0']
eligibility:
  - "input dtype fp16/bf16; NO collective in the chain (TP epilogue with allreduce belongs to the AR-epilogue case)"
  - "NVFP4 fused path: sm == 100 or sm == 103"
  - "NVFP4 layernorm fusion is default-OFF: TRTLLM_DISABLE_NVFP4_LAYERNORM_FUSION defaults to '1'; set '0' to enable"
interactions:
  - {case: case-fuse-ar-epilogue, relation: alternative-to, note: "the collective sibling — when a TP allreduce precedes the add→norm→quant chain, fuse into the collective instead"}
measured: []
---

# Fuse local residual-add + RMSNorm + quant (no collective) in one kernel / torch.compile pattern

> Part of the [Kernel & Fusion casebook](index.md) · schema: [case-template](../case-template.md)

- **Commits:** `211c44b95199` Adding torch ext API for FusedAddRMSNormQuant kernel (#9905); related: `72cd7d824bb0` Fuse add + norm + fp8 quant pattern (#12674).
- **Applies when:** memory-bound layer epilogue with NO collective: `residual = x + residual; normed = rms_norm(residual); q = quantize(normed)` per layer. Signals: separate add, RMSNorm and `static_quantize_e4m3_per_tensor`/`fp4_quantize` kernels; fp16/bf16 → FP8/NVFP4 before the next GEMM; norm output only consumed by the quantizer.
- **Mechanism:** (1) direct op `torch.ops.trtllm.fused_add_rms_norm_quant(...)` (`thop/fusedAddRMSNormQuant.cpp`, fp16/bf16) doing add+RMSNorm+NVFP4 quant in one kernel, returning packed FP4 + swizzled scales + updated residual — invoked from `RMSNorm(quantize_type="nvfp4")` and wired into `modeling_llama.py`. (2) `torch.compile` rewrite (`patterns/residual_add_norm.py::register_add_norm_quant`) matching `add → flashinfer_rmsnorm → static_quantize_e4m3_per_tensor` → `flashinfer_fused_add_rmsnorm_quant`. Both remove norm-output + residual round-trips and two launches/layer.
- **Generalizes to:** "fuse a pointwise+reduction+quant chain that feeds the next GEMM into one op"; carries to add+LayerNorm+quant, gated-norm+quant, FP8 vs NVFP4 output, and the collective variant (sibling of the [AR-epilogue case](fuse-ar-epilogue.md)); adapt by choosing the op for the quant dtype, honoring fp16/bf16-in, passing the next GEMM's `input_scale`, and (in-place) verifying residual's last user is the add.
- **Apply via:** `RMSNorm(..., quantize_type="nvfp4")` for the direct path; enable `torch.compile` for the FP8 rewrite. **NVFP4 layernorm fusion gated by env `TRTLLM_DISABLE_NVFP4_LAYERNORM_FUSION` (default "1" = OFF; set "0" to enable).** Delegate to **kernel-cuda-specialist** / **kernel-triton-specialist**.
- **Expected effect:** fewer kernels + less HBM traffic per epilogue → lower latency in memory-bound regimes; no number — measured Δ to be recorded from run.
- **Accuracy risk:** lossy — quant fused on normed output may differ from unfused quant order/rounding; that is why NVFP4 fusion ships default-OFF. Always parity-check.
- **Verify:** kernel count / nsys; numeric parity vs unfused add→norm→quant; accuracy eval (`test_nvfp4_with_norm_quant`, NVFP4 Llama-3.1-8B, SM100/103 only).
- **Rollback:** keep `TRTLLM_DISABLE_NVFP4_LAYERNORM_FUSION=1` and/or disable `torch.compile`. Trigger: accuracy regression, or non-SM100/103 hardware.
- **Prior art:** PRs #9905, #12674. Files: `cpp/.../thop/fusedAddRMSNormQuant.cpp`, `_torch/modules/rms_norm.py`, `modeling_llama.py`, `compilation/patterns/residual_add_norm.py`. Owning specialist: **kernel-cuda-specialist**.
