---
id: case-fold-scale-swizzle-into-kernel
type: case
family: kernel-and-fusion
maturity: full
bottleneck: [host-overhead, launch, memory]
signals: [many-small-kernels, hbm-roundtrip-between-ops]
architectures: [sm90, sm100]
model_scope: [moe, model-agnostic]
phase: [any-phase]
patterns: [pattern-fuse-chain-feeding-gemm]
accuracy_risk: lossless
apply_via_kind: [kernel-change, code-change]
knobs: []
specialists: [kernel-cuda-specialist, trtllm-moe-develop]
commits: ['2f2f5cc72c51']
measured: []
---

# Fold a host-side pre-pass into the consuming kernel (scale-factor swizzle)

> Part of the [Kernel & Fusion casebook](index.md) · schema: [case-template](../case-template.md)

- **Commits:** `2f2f5cc72c51` [TRTLLM-6744][feat] Remove input_sf swizzle for module WideEPMoE (#6231).
- **Applies when:** NVFP4 WideEPMoE with expert-parallel comm where a separate `swizzle_sf(...)` op was launched on the host path to re-layout the activation scale-factor tensor into SWIZZLED layout before the grouped GEMM — adding a kernel launch + full SF read/write per MoE layer; signals: a standalone swizzle/transpose op before MoE GEMM in profiles, host launch overhead in MoE dispatch.
- **Mechanism:** the MoE GEMM's activation-SF setup is taught to consume a non-swizzled (linear) SF directly: a `swizzled_input_sf` flag is threaded through `torch.ops.trtllm.fused_moe` → `moeOp.cpp` → CUTLASS `moe_kernels.cu`, which branches on the flag and reads SF in linear layout when `false`. The three `swizzle_sf(...)` calls in `WideEPMoE.forward` are deleted.
- **Generalizes to:** "fold a standalone layout/transform pre-pass into the kernel that consumes the data"; carries to other quantized GEMMs that pre-swizzle scales, weight-SF layout passes, any element-wise relayout (transpose/pack) feeding a GEMM; adapt by adding a layout flag to the consumer kernel and handling the un-transformed layout in its load path, then deleting the producer op.
- **Apply via:** plumb a layout flag (`swizzled_input_sf`) through the op chain; delegate to **kernel-cuda-specialist** (CUTLASS) and **trtllm-moe-develop** for module wiring.
- **Expected effect:** one fewer kernel launch + one fewer full SF read/write per MoE layer → lower host launch overhead and memory traffic in FP4 WideEPMoE; direction only — measured Δ (MoE-layer time, launch count, throughput) to be recorded from run.
- **Accuracy risk:** lossless when the flag is correct (kernel reads SF in its actual layout). Risk = layout/flag mismatch silently corrupting FP4 GEMM output — parity-check on first enable.
- **Verify:** profile shows the swizzle op gone from MoE path; MoE-layer latency/launch count down; NVFP4 MoE output parity vs pre-swizzle path.
- **Rollback:** set `swizzled_input_sf=True` and restore the `swizzle_sf(...)` calls. Trigger: FP4 MoE accuracy mismatch (layout/flag inconsistency).
- **Prior art:** PR #6231. Files: `_torch/custom_ops/torch_custom_ops.py`, `_torch/moe/fused_moe/fused_moe_wide_ep.py`, `cpp/.../thop/moeOp.cpp`, `cutlass_kernels/moe_gemm/moe_kernels.cu`. Owning specialist: **kernel-cuda-specialist**.
