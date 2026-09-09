---
id: case-reevaluate-fusion-boundary-per-dtype
type: case
family: kernel-and-fusion
maturity: full
bottleneck: [compute, launch]
signals: [many-small-kernels, mixed-dtype-checkpoint]
architectures: [any-sm]
model_scope: [sparse-attention, deepseek-v32, model-agnostic]
phase: [any-phase]
patterns: [pattern-reevaluate-fusion-boundary]
accuracy_risk: lossless
apply_via_kind: [code-change]
knobs: []
specialists: [kernel-cuda-specialist, trtllm-moe-develop]
commits: ['2f45640c19', 'e57d83c5dc', '4b915d4f9b']
interactions:
  - {case: case-ranking-only-precision-tf32, relation: composes-with, note: "the same indexer GEMM — fuse/unfuse boundary here, precision lever there"}
measured: []
---

# Fusion is not monotonic — re-evaluate the GEMM fuse boundary per dtype/checkpoint

> Part of the [Kernel & Fusion casebook](index.md) · schema: [case-template](../case-template.md)

- **Commits:** `2f45640c19` [feat] Unfuse indexer.wk from attention GEMM for
  DS-V3.2 NVFP4 (#11989); related: `e57d83c5dc` Fuse QK down_proj with indexer K +
  weight_proj for FP4 ckpt (#8771, the original fuse), `4b915d4f9b` Fuse indexer
  wk + weights_proj into a single GEMM (#12055, the narrower re-fuse).
- **Applies when:** you are deciding whether to fuse several small GEMMs that read
  the **same activation** into one wide GEMM (QKV projections, gate+up, LoRA A/B, a
  down-projection feeding multiple heads). The trap: a fusion that was a clear win
  for one checkpoint becomes **illegal or suboptimal** for another. Signals: a
  **mixed-precision checkpoint** puts one fused participant in a different dtype
  (e.g. attention weights NVFP4 but the indexer's `wk` still BF16); a fused op
  forces a computation that a cheaper path could otherwise **skip**; the fused GEMM
  asserts all participants share a dtype.
- **Mechanism:** concatenating co-resident projections into one wide GEMM (weights
  cat along the out-dim, materialized once in `post_load_weights`) cuts launches
  and shares the activation read — **but only when the fused weights share a
  dtype**. #8771 fused QK down_proj + `indexer.wk` + `weights_proj` into the
  oversized `kv_a_proj_with_mqa` for an FP4 checkpoint (all same dtype). A later
  FP4-*attention* checkpoint left `indexer.wk` in BF16 while the attention
  projections went NVFP4, so the shared-dtype fuse was no longer valid → #11989
  **unfused** `indexer.wk` back to a standalone GEMM, and **sited it so the
  short-MHA path can elide it** (when the indexer isn't needed the extra GEMM is
  skipped). #12055 then **re-fused at a narrower, same-(fp32)-dtype boundary**
  (`wk`+`weights_proj` only). The optimum moved as the checkpoint dtypes changed.
- **Generalizes to:** the pattern "**the optimal fusion boundary is
  dtype/checkpoint/backend-dependent — fusion is not unconditionally a win.**"
  Before fusing GEMMs that share an operand, check they share a dtype/quant format;
  when a mixed-precision checkpoint splits them, **unfuse** the odd one out; and
  weigh whether keeping an op **separate unlocks a conditional skip** (elide it when
  its output isn't consumed). Carries to: quantized QKV/gate-up fusion across
  FP8/NVFP4/BF16 checkpoints, LoRA-adapter fusion, any "fuse the projections"
  refactor applied blindly across checkpoints. Adapt by: gating the fuse on a
  runtime dtype check, keeping the boundary configurable per checkpoint, and
  materializing the fused weight once at load — not per forward.
- **Apply via:** **not a server knob** — model-definition code (Linear sizing +
  `post_load_weights` weight materialization, with a dtype guard). Delegate to
  **kernel-cuda-specialist** (GEMM) / **trtllm-moe-develop** or the model author for
  module wiring.
- **Expected effect:** correct + fastest fusion *for the checkpoint at hand* — fused
  where dtypes align (fewer launches, one activation read), unfused where a
  mixed-precision checkpoint or a skip-path makes separate better. Direction only —
  measured Δ (projection GEMM time, launch count, throughput) to be recorded per
  checkpoint.
- **Accuracy risk:** **lossless** — fusing or unfusing GEMMs is numerically
  equivalent (same math, combined or split); the risk is a *correctness* bug (a
  dtype-mismatched fuse), not precision loss. The dtype-guard assert is what makes
  an illegal fuse fail loudly instead of silently corrupting.
- **Verify:** the fused/unfused path produces identical outputs vs the reference on
  each target checkpoint; projection GEMM time + launch count before vs after **on
  the specific checkpoint** (a win on one checkpoint does not transfer). Confirm the
  skip-path (if relied on) actually elides the unfused op.
- **Rollback:** flip the fuse flag / restore the separate (or fused) GEMM. Trigger:
  a dtype-mismatch assert, an accuracy mismatch, or no measured benefit on the
  target checkpoint.
- **Prior art:** PRs #11989, #8771, #12055. Files:
  `tensorrt_llm/_torch/models/modeling_deepseekv3.py` (`DeepseekV32Attention`,
  `fuse_a_indexer_k_weight`, `post_load_weights`),
  `tensorrt_llm/_torch/attention/mla.py`,
  `tensorrt_llm/_torch/attention/backends/sparse/dsa/indexer.py`. Owning specialists:
  **kernel-cuda-specialist**, **trtllm-moe-develop**. Related: the
  [ranking-only TF32 case](ranking-only-precision-tf32.md) (same indexer GEMM, the
  precision lever) and the many "fuse the chain" cases in this family — this is the
  caveat that keeps them from being applied blindly.
