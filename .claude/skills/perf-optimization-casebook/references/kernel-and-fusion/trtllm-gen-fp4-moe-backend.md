---
id: case-trtllm-gen-fp4-moe-backend
type: case
family: kernel-and-fusion
maturity: full
bottleneck: [compute]
signals: [gemm-dominates-step, hbm-roundtrip-between-ops]
architectures: [sm100]
model_scope: [moe, deepseek-v3r1, model-agnostic]
phase: [decode]
patterns: [pattern-hw-matched-lowprec-gemm-backend]
accuracy_risk: lossy
apply_via_kind: [config-knob]
knobs: [moe_backend]
specialists: [kernel-cute-specialist, trtllm-moe-develop]
commits: ['31624b079a12']
eligibility:
  - "sm == 100 (sm_100a prebuilt cubins)"
  - "quant is NVFP4 or FP8_BLOCK_SCALES; the cubin gemmList must cover the (dtype, tileN, transposeMmaOutput) combination"
measured: []
---

# Select the trtllm-gen FP4 grouped-GEMM MoE backend for Blackwell DeepSeek

> Part of the [Kernel & Fusion casebook](index.md) · schema: [case-template](../case-template.md)

- **Commits:** `31624b079a12` feat: [Deepseek] Add trtllm-gen MOE FP4 MOE backend (#3387).
- **Applies when:** compute-bound MoE grouped-GEMM on sm_100a; quant is `NVFP4`/`FP8_BLOCK_SCALES`; DeepSeek-V3/R1 (or other block-scale MoE); currently on default `CUTLASS` MoE backend and grouped-GEMM dominates the decode step.
- **Mechanism:** swaps the per-expert grouped GEMM to trtllm-gen prebuilt SM100a cubins (`blockScaleMoe`) that natively consume FP4/FP8 block scales, with routing fused into the moe_runner — avoiding the CUTLASS path's extra type conversions and a redundant FP4 temp output. Deliberately runs routing reduction in fp32 with tanh, and disables PDL for fc1 to stabilize numerics.
- **Generalizes to:** "pick a hardware-specialized low-precision grouped-GEMM backend over the generic one when the GEMM is the bottleneck and the dtype is natively supported"; carries to FP8 block-scale MoE, MXFP4, other Blackwell MoE models (KimiK2, Qwen-next); adapt by checking the cubin/gemmList covers your (dtype, tileN, transposeMmaOutput) and quant exclude_modules match (trtllm-gen excludes `*kv_b_proj*`, `*k_b_proj*`, `*eh_proj` for FP8 block scales).
- **Apply via:** `moe_backend='TRTLLM'` (on `ModelConfig`/`PyTorchConfig`, default `'CUTLASS'`; `--moe_backend {CUTLASS,TRTLLM}`). Gate `FusedMoE.is_trtllm()`; dispatches `forward_trtllmgen` → `torch.ops.trtllm.fp8_block_scale_moe_runner`/`fp4_block_scale_moe_runner`. Delegate to **kernel-cute-specialist** only if a cubin variant is missing.
- **Expected effect:** higher MoE grouped-GEMM throughput / lower decode latency on Blackwell; no number — measured Δ to be recorded from run.
- **Accuracy risk:** lossy (FP4/FP8 block-scale GEMM). Mitigations: fp32 routing reduction + tanh, PDL off for fc1. NVFP4 weights use a distinct loader (`is_trtllm_nvfp4`) so weight layout must match.
- **Verify:** MMLU / task accuracy vs CUTLASS baseline (commit adds `mmlu_llmapi.py` hooks, extends `test_moe.py`); grouped-GEMM kernel time drops in nsys.
- **Rollback:** `moe_backend='CUTLASS'`. Trigger: accuracy regression beyond tolerance, or a dtype/shape lacking a trtllm-gen cubin.
- **Prior art:** PR #3387. Files: `_torch/modules/fused_moe.py` (`is_trtllm`/`forward_trtllmgen`), `_torch/model_config.py` (`moe_backend`), `cpp/.../trtllmGenKernels/blockScaleMoe/{runner.cu,gemmList.h,RoutingKernel.cu}`, `thop/fp4BlockScaleMoe.cpp`. Owning skill: **trtllm-moe-develop**; cubins: **kernel-cute-specialist**.
