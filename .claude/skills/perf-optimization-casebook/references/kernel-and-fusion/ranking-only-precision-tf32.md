---
id: case-ranking-only-precision-tf32
type: case
family: kernel-and-fusion
maturity: full
bottleneck: [compute]
signals: [kernel-far-from-sol, slow-path-fallback, small-batch-decode]
architectures: [any-sm]
model_scope: [sparse-attention, deepseek-v32, model-agnostic]
phase: [any-phase]
patterns: [pattern-ranking-only-reduced-precision]
accuracy_risk: lossy
apply_via_kind: [code-change]
knobs: []
specialists: [kernel-cuda-specialist, perf-nsight-compute-analysis]
commits: ['24bd7a2c20', '4b915d4f9b', '79a6c9742b']
interactions:
  - {case: case-reevaluate-fusion-boundary-per-dtype, relation: composes-with, note: "the same indexer GEMM — precision lever here, fuse/unfuse boundary there"}
measured: []
---

# Run a ranking-only GEMM in TF32 — and verify the tensor-core path is actually taken

> Part of the [Kernel & Fusion casebook](index.md) · schema: [case-template](../case-template.md)

- **Commits:** `24bd7a2c20` [perf] Force enable TF32 tensor cores for DSA indexer
  fused GEMM (#13452); related: `4b915d4f9b` Fuse indexer wk + weights_proj into a
  single TF32 GEMM (#12055), `79a6c9742b` [fix] Use fp32 for indexer weight_proj
  GEMM (#9243, the precision contract these optimize around).
- **Applies when:** a GEMM (or reduction) produces a value that is used **only to
  rank or select** downstream — not as a precise numeric operand. The archetype:
  DSA's lightning indexer projects hidden states to per-token **index scores**
  whose sole use is a top-k KV selection; the score magnitudes never feed a
  precise computation. Signals: a scoring/gating/routing GEMM feeding an
  `argmax` / `top-k` / threshold; the GEMM runs in FP32 "for safety" and shows up
  as a CUDA-core SIMT SGEMM (not a tensor-core GEMM) in ncu; small-M (decode) so
  it's launch/latency-sensitive.
- **Mechanism:** two coupled moves. (1) **The output only ranks, so drop the
  compute to TF32** — keep FP32 as the *storage/accumulation contract* (a
  softmax-scale factor still deserves FP32, #9243) but let the multiply run on
  TF32 tensor cores (~10-bit mantissa), which is far faster than FP32 CUDA-core
  SGEMM and precise enough to preserve the ranking. (2) **Verify the tensor-core
  path is actually taken** — asking for FP32 does *not* guarantee TF32: a custom
  `cublas_mm` pinned to `CUBLAS_COMPUTE_32F` (and cuBLASLt heuristics for small M)
  silently fall back to CUDA-core SGEMM (#12055 asked for TF32 but didn't get it).
  Force it with `torch.backends.cuda.matmul.allow_tf32 = True` **and** an entry
  point that honors the flag — `F.linear` routes through PyTorch's cuBLAS handle
  and dispatches `CUBLAS_COMPUTE_32F_FAST_TF32` (#13452 wraps the call in a
  `_tf32_matmul_enabled()` context manager and swaps `cublas_mm → F.linear`).
- **Generalizes to:** the pattern "**a computation whose result is consumed only
  for ranking/selection tolerates reduced precision — run it on the fast
  tensor-core path (TF32 / BF16), keep the higher-precision contract only where a
  scale/softmax needs it, and confirm the accelerated path is dispatched (small-M
  GEMMs silently fall back to CUDA cores).**" Carries to: MoE router logits before
  `argmax`/top-k, retrieval/similarity scoring for nearest-neighbour selection,
  attention-sink or landmark scoring, any gate whose output only picks. Adapt by:
  (a) confirming the output is ranking-only (a downstream select, not a numeric
  operand); (b) choosing the precision the hardware accelerates; (c) **verifying
  dispatch** (ncu: tensor-core GEMM, not SIMT) rather than trusting the requested
  compute type.
- **Apply via:** **not a server knob** — a code change at the GEMM call site
  (`allow_tf32` context + `F.linear` in place of a fixed-compute-type custom op).
  Delegate to **kernel-cuda-specialist**; confirm the dispatched kernel with
  **perf-nsight-compute-analysis** (the whole point is that the requested
  precision ≠ the executed kernel).
- **Expected effect:** the scoring GEMM moves off CUDA cores onto tensor cores →
  lower indexer/scoring latency at small M, with the selection unchanged.
  Direction only — measured Δ (GEMM time, kernel name/tensor-core utilization,
  decode throughput) to be recorded from the run.
- **Accuracy risk:** **lossy but bounded** — TF32 rounding perturbs the score
  values, but they are used only to rank, so the selected set is (almost always)
  identical. Lower-risk than KV/weight quant, but still verify: it can in principle
  flip a near-tie selection. Confirm top-k selection overlap and e2e accuracy are
  unchanged on enable; keep an accuracy note if promoting as a default.
- **Verify:** correctness — top-k index overlap vs the FP32 path on representative
  logits; DSA e2e accuracy (GSM8K/MMLU) within tolerance. Perf — ncu confirms a
  tensor-core GEMM (e.g. `…_tf32…`), not a SIMT SGEMM; GEMM time and throughput
  before vs after.
- **Rollback:** remove the `allow_tf32` context (revert to the FP32 path). Trigger:
  any selection/accuracy drift beyond tolerance, or no measured speedup (path was
  already tensor-core).
- **Prior art:** PRs #13452, #12055, #9243. Files:
  `tensorrt_llm/_torch/attention/backends/sparse/dsa.py` (`_tf32_matmul_enabled`,
  `_fused_wk_wp_weight`, `F.linear` vs `torch.ops.trtllm.cublas_mm`). Owning
  specialist: **kernel-cuda-specialist**; dispatch check:
  **perf-nsight-compute-analysis**. Related: the [fusion-boundary case](reevaluate-fusion-boundary-per-dtype.md)
  (the same indexer GEMM, the orthogonal fuse/unfuse lesson).
