---
id: case-fuse-moe-routing-kernel
type: case
family: kernel-and-fusion
maturity: full
bottleneck: [launch, memory]
signals: [routing-hot-path, many-small-kernels, hbm-roundtrip-between-ops, slow-path-fallback]
architectures: [any-sm]
model_scope: [moe, deepseek-v3r1, kimi-k2, model-agnostic]
phase: [decode]
patterns: [pattern-fuse-small-op-chain, pattern-reuse-warp-primitive, pattern-bounded-fastpath-fallback]
accuracy_risk: lossless
apply_via_kind: [kernel-change]
knobs: []
specialists: [kernel-cuda-specialist]
commits: ['c8b9998acb8b', '84926bcb6f14']
log_markers:
  - "use the original pytorch implementation"   # is_fused=False fallback warning
eligibility:
  - "model uses the noaux_tc router (sigmoid gating + per-expert correction bias, grouped top-k)"
  - "fused-path bounds live in tensorrt_llm/_torch/modules/fused_moe/routing.py::Deepseekv3RoutingImpl.noaux_tc (the is_fused guard) — read them from YOUR checkout before judging eligibility"
  - "as of 84926bcb6f14, n_group > 1: requires top_k <= 8 and num_experts <= 256 and experts_per_group <= 32 and experts_per_group * topk_group <= 256"
  - "as of 84926bcb6f14, n_group == 1: requires num_experts <= 1024 and top_k <= 32"
interactions:
  - {case: case-pdl, relation: composes-with, note: fused kernel can overlap neighbors via PDL (TRTLLM_ENABLE_PDL)}
measured: []
---

# Fuse MoE grouped-top-k routing into one per-token CUDA kernel

> Part of the [Kernel & Fusion casebook](index.md) · schema: [case-template](../case-template.md)

- **Commits:** `c8b9998acb8b` Optimize the routing kernel for DeepseekV3 (#7761);
  related: `84926bcb6f14` Update the deepseek routing (#13186) — widens fused-path
  config eligibility so more workloads stay fused.
- **Applies when:** the MoE **router/gating** step is on the decode hot path and
  shows up as many tiny back-to-back kernels with GPU-idle gaps (launch-bound,
  small tensors), **or** the config silently fell back to the slow PyTorch router
  (an `is_fused=False` `warnings.warn(...)` "use the original pytorch
  implementation") — classify with `perf-host-analysis` / `perf-nsight-systems`
  first. The model uses the **noaux_tc** router (sigmoid gating + per-expert
  correction bias, `n_group` groups, `topk_group`, top-`k` experts,
  `routed_scaling_factor`): DeepSeek-V3/R1, Kimi K2. (Qwen3 models use
  Renormalize-style routing, not noaux_tc — out of scope here.) The current path
  computes scores as a chain of PyTorch elementwise ops (sigmoid, `+bias`) that
  materializes `scores` / `scores_with_bias` to HBM, then runs a separate
  group-score kernel (writing a `group_scores` intermediate) and a generic
  bitonic-sort top-k. **Bounded config** (else it must fall back to PyTorch):
  as of `84926bcb6f14`: `n_group>1` → `top_k ≤ 8`, `num_experts ≤ 256`,
  `experts_per_group ≤ 32`, `experts_per_group·topk_group ≤ 256`; `n_group==1` →
  `num_experts ≤ 1024`, `top_k ≤ 32`. The live guard is
  `routing.py::Deepseekv3RoutingImpl.noaux_tc` — re-read it from your checkout.
- **Counter-signals:** config outside the fused-path bounds (fails the
  `is_fused` guard — no win without re-deriving bounds for the new shape,
  and the kernel hard-checks rather than degrades); prefill-heavy / large-batch
  regimes where routing is a negligible share of step time (measure the routing
  share first); workloads requiring bit-exact parity with the PyTorch router's
  tie-breaking order on equal scores.
- **Mechanism:** collapse the whole router — sigmoid, bias add, per-group score
  (sum of top-2), top-group selection, in-group expert top-k, and normalize ×
  `routed_scaling_factor` — into **one block-per-token kernel**. Removes the
  PyTorch elementwise launches and the `scores_with_bias` / `group_scores`
  HBM round-trips (kernel now takes raw `logits` + `bias`), and replaces the
  generic bitonic-sort top-k with a warp-shuffle packed-value top-k reduction
  (the same `moeTopKFuncs.cuh::reduceTopK` the TRTLLM-Gen MoE routing kernels
  use). Net: fewer launches + less HBM traffic + cheaper top-k; optionally
  overlaps with neighbors via PDL.
- **Generalizes to:** the pattern "collapse a small launch-bound op-chain into
  one fused per-token kernel, reusing an existing warp primitive." Carries to
  other grouped/top-k routers (e.g. Llama-4 MoE — similar shape, different bounds),
  fused sampling top-k/top-p (reuse `reduceTopK`), and any per-token
  "activation → reduction → select" chain. Adapt by: re-deriving the
  supported-config bounds for the new shape and keeping the Python `is_fused`
  guard in sync with them, and reusing an existing reduction primitive over a new
  sort. The bounded-guard-plus-fallback is part of the pattern, not incidental.
- **Apply via:** **not a server config knob** — a kernel/op change. Delegate to
  **kernel-cuda-specialist** (raw CUDA C++):
  `cpp/tensorrt_llm/kernels/noAuxTcKernels.cu` (+`.h`), the op binding
  `cpp/tensorrt_llm/thop/noAuxTcOp.cpp`, and the model routing path
  `tensorrt_llm/_torch/models/modeling_deepseekv3.py` (`noaux_tc`). Reuse the
  existing `reduceTopK` rather than writing a new sort. PDL is gated on
  `getEnvEnablePDL()` (`TRTLLM_ENABLE_PDL`).
- **Expected effect:** lower MoE-routing latency, fewer kernel launches, and a
  smaller inter-kernel GPU-idle gap around gating; reduced HBM traffic from the
  eliminated intermediates. Largest at **decode / small token counts** where
  launch overhead dominates. Measured Δ to be recorded from an nsys trace
  (routing kernel count + total duration) before vs after — PR #7761 ships the
  optimization but states no public speedup; do **not** cite a number not
  measured locally.
- **Accuracy risk:** lossless — numerically equivalent (same sigmoid / bias /
  group-top-k / expert-top-k / normalize math, within fp tolerance). Still a
  kernel rewrite, so **verify vs reference**; tie-breaking on equal scores can
  differ from the PyTorch path.
- **Verify:** correctness — `pytest tests/unittest/_torch/thop/parallel/test_noaux_tc.py`
  and `test_moe.py`, plus the C++ `routingDeepSeekTest`; compare
  `topk_values`/`topk_indices` to the PyTorch reference within tolerance,
  including tie cases. Perf — nsys before/after: routing kernel count, total
  duration, inter-kernel gap, and the fused kernel's SOL. Also confirm
  out-of-bound configs (`top_k>8`, experts past the cap) take the PyTorch
  fallback (the `is_fused=False` warning) and still produce correct results.
- **Rollback:** force the PyTorch router (`self.is_fused = False`) or revert the
  op to the precompute-scores signature; trigger: any accuracy mismatch vs
  reference outside tolerance, or >5% perf regression. Note the kernel
  **hard-checks** unsupported configs (`TLLM_CHECK`), so the Python guard bounds
  must stay in sync with the kernel's supported bounds — a mismatch errors
  instead of falling back.
- **Prior art:** PRs #7761 (`[TRTLLM-8637]`), #13186 (widens fused-path
  eligibility; `tensorrt_llm/_torch/modules/fused_moe/routing.py`
  `Deepseekv3RoutingImpl`);
  `cpp/tensorrt_llm/kernels/noAuxTcKernels.cu`, `moeTopKFuncs.cuh`,
  `cpp/tensorrt_llm/thop/noAuxTcOp.cpp`,
  `tensorrt_llm/_torch/models/modeling_deepseekv3.py`; analogous TRTLLM-Gen MoE
  routing kernels under `kernels/trtllmGenKernels/blockScaleMoe/Routing*.cu`.
  Owning specialist: **kernel-cuda-specialist**. Related: `perf-host-analysis`
  (inter-kernel gap), `perf-nsight-systems` (launch count).
