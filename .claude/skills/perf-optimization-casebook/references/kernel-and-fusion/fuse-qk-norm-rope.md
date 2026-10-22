---
id: case-fuse-qk-norm-rope
type: case
family: kernel-and-fusion
maturity: full
bottleneck: [launch, memory]
signals: [many-small-kernels, hbm-roundtrip-between-ops]
architectures: [any-sm]
model_scope: [qwen3-moe, gemma3, llama-family, model-agnostic]
phase: [any-phase]
patterns: [pattern-fuse-chain-feeding-gemm]
accuracy_risk: lossless
apply_via_kind: [default-on, code-change]
knobs: [fuse_qk_norm_rope]
specialists: [kernel-cuda-specialist]
commits: ['9c4b8f66b454']
eligibility:
  - "model applies per-head RMSNorm to Q and K before RoPE (qk_norm_type == pre_rope)"
  - "bf16 only; rope variant/layout must be supported by the fused kernel (fall back to unfused otherwise)"
measured: []
---

# Fuse QK-Norm with RoPE in attention pre-processing

> Part of the [Kernel & Fusion casebook](index.md) · schema: [case-template](../case-template.md)

- **Commits:** `9c4b8f66b454` feat: Integration of Fused QKNorm+RoPE (#4611).
- **Applies when:** launch/memory-bound attention pre-processing on models applying per-head RMSNorm to Q and K before RoPE (`qk_norm_type == pre_rope`). Signals: separate `q_norm`/`k_norm` RMSNorm + a separate RoPE kernel per layer; QKV laid out for an in-place op.
- **Mechanism:** replaces (Q/K RMSNorm) → (RoPE) with one in-place op `torch.ops.trtllm.fused_qk_norm_rope(qkv, ...)` that normalizes Q,K and applies rotary in a single kernel over packed QKV — saving two RMSNorm launches + round-trips and the standalone RoPE launch per layer. Flag `fuse_qk_norm_rope` (default True on supporting models); when set, RoPE is removed from base attention and `apply_rope` dispatches to `apply_qk_norm_rope`. Registered in-place → CUDA-graph / torch.compile safe.
- **Generalizes to:** "fold the norm that immediately precedes a positional/elementwise transform into that transform's kernel"; carries to post-rope QK-norm, DiT QK-norm+RoPE, partial-rotary/YaRN variants, split+norm+rope; adapt by matching norm placement (pre vs post rope), rotary interleave/mRoPE layout, and supported dtype (bf16); fall back to unfused when freq derivation doesn't match.
- **Apply via:** set `fuse_qk_norm_rope=True` on the attention module (exposed in `modeling_qwen3.py` etc.); op `torch.ops.trtllm.fused_qk_norm_rope`. Delegate to **kernel-cuda-specialist** to wire a new model.
- **Expected effect:** fewer attention pre-processing kernels per layer → lower attention latency / better launch efficiency at small batch; no number — measured Δ to be recorded from run.
- **Accuracy risk:** lossless in principle (same QK-norm+RoPE math, one kernel) but bf16-only and layout-sensitive — enabling on a non-bf16/interleaved-rope model without kernel support is a correctness risk; parity-check Q/K on a new model.
- **Verify:** kernel count / nsys (one fused op vs two norms + rope); Q/K output parity vs unfused; downstream accuracy.
- **Rollback:** `fuse_qk_norm_rope=False` (falls back to separate `apply_qk_norm` + `rotary_emb`). Trigger: Q/K parity mismatch, non-bf16 dtype, unsupported rope variant.
- **Prior art:** PR #4611. Files: `_torch/attention/qk_norm_attention.py` (`apply_qk_norm_rope`), `modeling_qwen3.py`, `_torch/compilation/utils.py` (in-place reg). Owning specialist: **kernel-cuda-specialist**.
