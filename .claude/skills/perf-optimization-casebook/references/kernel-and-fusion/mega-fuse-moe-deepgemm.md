---
id: case-mega-fuse-moe-deepgemm
type: case
family: kernel-and-fusion
maturity: full
bottleneck: [launch, memory]
signals: [many-small-kernels, hbm-roundtrip-between-ops]
architectures: [sm100]
model_scope: [moe, deepseek-v3r1, model-agnostic]
phase: [decode]
patterns: [pattern-fuse-chain-feeding-gemm, pattern-hw-matched-lowprec-gemm-backend]
accuracy_risk: lossy
apply_via_kind: [config-knob]
knobs: [moe_backend]
specialists: [kernel-cute-specialist]
commits: ['6e069b69ef3c']
eligibility:
  - "sm == 100"
  - "quant exactly W4A8_MXFP4_MXFP8; SwiGLU activation"
  - "expert-parallel only: moe_tp_size == 1 and cluster_size == 1"
  - "hidden_size % 128 == 0 and intermediate_size % 128 == 0"
  - "live gate: mega_moe/backend.py::can_implement — bounds as of 6e069b69ef3c"
interactions:
  - {feature: TRTLLMGenFusedMoE, relation: alternative-to, note: "same W4A8_MXFP4_MXFP8 math, different kernel — the parity reference and rollback target"}
measured: []
---

# Mega-fuse the entire MoE pipeline into one DeepGEMM launch (EP-dispatch + GEMM1 + SwiGLU + GEMM2 + EP-combine)

> Part of the [Kernel & Fusion casebook](index.md) · schema: [case-template](../case-template.md)

- **Commits:** `6e069b69ef3c` [feat] Add MegaMoEDeepGemmFusedMoE backend wrapping DeepGEMM fp8_fp4_mega_moe (#13384).
- **Applies when:** launch-bound (many small per-stage MoE kernels + EP all-to-all) on sm_100; quant exactly `W4A8_MXFP4_MXFP8`; expert-parallel only (`moe_tp_size=1`, `cluster_size=1`); `hidden_size%128==0` and `intermediate_size%128==0`; SwiGLU activation.
- **Mechanism:** replaces the multi-kernel MoE sequence (separate EP dispatch, GEMM1, SwiGLU, GEMM2, EP combine) with one fused kernel `deep_gemm.fp8_fp4_mega_moe`. Math matches `TRTLLMGenFusedMoE`'s W4A8_MXFP4_MXFP8 path; win is one launch vs 5+ stages, collapsing launch overhead + intermediate HBM round-trips. Inputs MXFP8 per-token quantized; weights pre-transformed via `transform_weights_for_mega_moe`.
- **Generalizes to:** "mega-fuse a multi-stage pipeline whose stages are individually launch/memory-bound into one kernel"; carries to other fused-MoE megakernels, fused attention pipelines, any EP-dispatch+GEMM+combine chain; adapt by matching strict preconditions (arch, quant, EP-only, shape%128) and providing a clean fallback when unmet.
- **Apply via:** `moe_backend='MEGAMOE_DEEPGEMM'` (resolved in `create_moe.py::get_moe_cls`); self-gates via `can_implement(...)`, raises `_MegaMoEUnavailable` (graceful fallback) if bundled `tensorrt_llm.deep_gemm` lacks `fp8_fp4_mega_moe`. Delegate to **kernel-cute-specialist** / DeepGEMM owners.
- **Expected effect:** fewer launches + less intermediate memory traffic → lower MoE decode latency; no number (code notes fused MXFP8 quant ~11 us/launch vs per-seq Triton fallback) — measured Δ to be recorded from run.
- **Accuracy risk:** lossy (W4A8_MXFP4_MXFP8 MoE GEMM) but output-equivalent to the existing `TRTLLMGenFusedMoE` W4A8 backend by design (same math, different kernel) — parity is against that backend, not bf16.
- **Verify:** numeric parity vs `TRTLLMGenFusedMoE` same model/quant; task accuracy vs that backend; MoE collapses to a single kernel in nsys.
- **Rollback:** `moe_backend` → `TRTLLM` or `CUTLASS`. Trigger: `can_implement` rejects env (non-SM100, wrong quant, tp>1), missing DeepGEMM symbols, or parity failure.
- **Prior art:** PR #13384. Files: `_torch/moe/fused_moe/mega_moe/backend.py` (`can_implement`), `create_moe.py`, DeepGEMM `fp8_fp4_mega_moe`/`transform_weights_for_mega_moe`. Owning specialist: **kernel-cute-specialist**.
