---
id: case-hw-matched-lowprec-moe-gemm
type: case
family: kernel-and-fusion
maturity: full
bottleneck: [compute, memory]
signals: [gemm-dominates-step, weight-reads-dominate]
architectures: [sm89, sm90, sm100]
model_scope: [moe, deepseek-v3r1, model-agnostic]
phase: [any-phase]
patterns: [pattern-hw-matched-lowprec-gemm-backend]
accuracy_risk: lossy
apply_via_kind: [config-knob]
knobs: [moe_backend]
specialists: [kernel-cuda-specialist, kernel-cute-specialist, perf-nsight-compute-analysis, trtllm-moe-develop, perf-sweep-challenger]
commits: ['7bb0a78631de', '20b42912cef7']
eligibility:
  - "sm == 100: FP8 128x128 block-scale grouped GEMM via moe_backend=DEEPGEMM — requires the deep_gemm dependency and an FP8 block-scale checkpoint"
  - "sm == 90 or sm == 89: W4A8/W4AFP8 — requires a ModelOpt-calibrated W4AFP8 checkpoint (quant_config.is_int4_weight_only_per_group())"
measured: []
---

# Pick the hardware-matched low-precision MoE GEMM (FP8 block-scale on Blackwell, W4A8 on Hopper)

> Part of the [Kernel & Fusion casebook](index.md) · schema: [case-template](../case-template.md)

- **Commits:** `7bb0a78631de` Deepseek R1 FP8 Support on Blackwell (#6486); related: `20b42912cef7` [feat] Support DeepSeek-R1 W4A8 on Hopper (#4123).
- **Applies when:** compute-bound (and weight-memory-bound) MoE on DeepSeek-class models, choosing expert-GEMM precision/kernel to match the GPU's tensor-core support. Blackwell sm_100: FP8 **128×128 block-scale** grouped GEMM (DeepGEMM). Hopper sm_90 / Ada sm_89: **W4A8/W4AFP8** (INT4 weights + FP8 activations) to shrink weight bytes and hit FP8 tensor cores.
- **Mechanism:** Blackwell — `DeepGemmFusedMoE` calls `deep_gemm` `m_grouped` FP8 GEMM (`deepgemm_fp8_group_blockwise_gemm`, FP8 e4m3 → BF16) with 128-wide block scales as **UE8M0** (`per_token_cast_to_fp8_e8m0`, `per_block_cast_to_fp8_e8m0`, amax/448.0), re-laying weight scales via `resmooth_to_fp8_e8m0`/`transform_sf_into_required_layout` when `has_fp8_block_scales() and sm==100`; per-token act quant in a Triton kernel. Hopper — `has_w4afp8` packs experts as `torch.quint4x2` with `use_w4a8_group_scaling=True`, per-group 1×128 weight scales + per-tensor act scales (`FusedMoEQuantScalesW4A8`); mixed-precision CUTLASS MoE GEMM does INT4×FP8. Both halve/quarter expert weight bytes; Blackwell prefers native FP8 block-scale, Hopper goes INT4-weight to cut bandwidth and reuse FP8 math.
- **Generalizes to:** "select the low-precision GEMM format and kernel backend by GPU tensor-core capability, not one-size-fits-all"; carries to dense linears (Blackwell FP8 block-scale vs Hopper W4A8/W4A16), NVFP4 on Blackwell, other grouped-expert models (Mixtral, Qwen-MoE); adapt by matching format to arch, wiring the right scale layout (UE8M0 vs 1×128 group), keeping a non-quant fallback.
- **Apply via:** `moe_backend` (`model_config.moe_backend`, e.g. `"DEEPGEMM"`; `--moe_backend DEEPGEMM`) on Blackwell with an FP8 block-scale checkpoint; on Hopper load a W4AFP8 checkpoint (ModelOpt-calibrated act scales) so `quant_config` reports `is_int4_weight_only_per_group()`. Blackwell path needs the `deep_gemm` dependency. Delegate to **kernel-cuda-specialist** / **kernel-cute-specialist**; profile with **perf-nsight-compute-analysis**.
- **Expected effect:** higher MoE GEMM throughput + lower expert-weight memory on the matched GPU (FP8 ~½ BF16 bytes; W4A8 ~¼ weight bytes); no number — measured Δ to be recorded from run.
- **Accuracy risk:** lossy (FP8 e4m3 block-scale; W4A8 = INT4 weights + FP8 activations). Each needs an on-disk accuracy record (GSM8K/MMLU) + rollback criterion before promotion. W4A8 depends on a calibrated act-scale file; bad calibration degrades accuracy.
- **Verify:** MoE GEMM throughput + weight footprint; accuracy parity vs a higher-precision backend (CUTLASS BF16/FP8); Blackwell confirm UE8M0 layout (sm_100 gate), Hopper confirm W4A8 group scaling (sm_90).
- **Rollback:** switch `moe_backend` to `CUTLASS` (or BF16/FP8-per-tensor) / load higher-precision checkpoint. Trigger: accuracy regression beyond threshold, or backend unsupported on the running SM.
- **Prior art:** PRs #6486, #4123. Files (Blackwell): `_torch/modules/fused_moe/fused_moe_deepgemm.py`, `create_moe.py`, `quantization/utils/fp8_utils.py`, `modeling_deepseekv3.py`. (Hopper): `_torch/modules/fused_moe.py` (`has_w4afp8`, `FusedMoEQuantScalesW4A8`), `thop/moeOp.cpp`, `examples/quantization/quantize_mixed_precision_moe.py`. Owning skill: **trtllm-moe-develop**; gate with **perf-sweep-challenger**.
