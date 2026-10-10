---
id: case-router-gemm-missing-instantiation
type: regression-case
family: kernel-and-fusion
module: moe
maturity: full
regression_class: [fast-path-fallback]
signals: [slower-kernel-in-trace, itl-increase, throughput-drop]
subsystems: [moe, gemm-kernel]
introduced_via: [incomplete-coverage]
phase: [decode]
patterns: [pattern-fast-path-silent-fallback]
nvbugs: ["6108841"]
commits: ["7d2bed7820ce"]
success_prs: [13740]
failed_prs: []
---

# Router GEMM silently falls back to non-tensor-core cuBLAS SIMT on GLM-5

> Part of the [MoE regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `6108841` · commit `7d2bed7820ce` · PR #13740 —
  "add hidden_dim=6144 router GEMM instantiation for GLM-5". **This case is
  one fixed sub-cause of a still-open bug, not the bug's resolution.**
  6108841 is a performance bug from a customer report: a 40–50% decode-phase
  gap between TRT-LLM and TileRT serving GLM-5 at MTP=3, batch size 1. It is
  still open after 111 days. Per the NVBug, #13740 — which extends the
  DeepSeek-V3 router kernel to GLM-5's hidden dimension — narrowed the gap to
  TileRT only a little: from 0.50x to 0.56x TileRT's output throughput. The
  remaining gap is being chased with a fused `ExpertSelectUpGateSiLU`-style
  mega-kernel, which is a different problem — so do not read this case as
  "the GLM-5 decode gap was a missing instantiation".
- **Symptom:** ~5.4% / ~1 ms extra per decode iteration on GLM-5 FP8 MTP=3
  BS=1 ISL=1K, B200 TP=8 (numbers from the PR description); the MoE routing
  GEMM ran as `cutlass_80_simt_sgemm_64x64_8x5_tn_align1` — an Ampere SIMT
  kernel with no tensor cores — instead of the min-latency custom kernel.
- **Root cause:** The `dsv3_router_gemm` custom kernel only had compile-time
  template instantiations for `hidden_dim=7168` (DeepSeek-V3). GLM-5 has
  `hidden_size=6144` with 256 MoE experts, so the shape check in
  `dsv3_router_gemm_op` failed and the op silently fell back to
  `cublas_mm_out`, which dispatched BF16xBF16->FP32 to the slow SIMT SGEMM
  on Blackwell.
- **How introduced:** incomplete coverage — the min-latency router GEMM
  shipped instantiated only for DeepSeek-V3's hidden dim; no specific
  regressing commit is named in the PR. The gap surfaced when GLM-5 reused
  the same op with a hidden dim the kernel was never compiled for.
- **Fix mechanism:** Adds the 16 `num_tokens` (1..16) template
  instantiations of `invokeRouterGemm<__nv_bfloat16, N, 256, 6144>` in
  `cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3RouterGemm.cu`
  (the kernel template is generic over `kHiddenDim` when divisible by
  VPT*kBlockSize = 1024) and makes `dsv3_router_gemm_op` in
  `cpp/tensorrt_llm/thop/dsv3RouterGemmOp.cpp` dispatch between K=7168 and
  K=6144 paths; unsupported shapes still fall back to `cublas_mm_out`.
- **Detection signal:** nsys decode trace shows a `simt_sgemm` (or any
  non-tensor-core cuBLAS/cutlass SGEMM) where the router GEMM should be;
  check the op's supported dims against the model's hidden size with
  `grep -n "kHiddenDim" cpp/tensorrt_llm/thop/dsv3RouterGemmOp.cpp` —
  if the model's `hidden_size` is not listed, every routing GEMM call is
  taking the cuBLAS fallback.
- **Prevention/guard:** PR #13740 extended
  `tests/unittest/_torch/thop/parallel/test_dsv3_router_gemm.py` to cover
  `hidden_size` in {7168, 6144}, but that only pins shapes already known.
  The fallback branch is still silent (the diff marks it only with a
  `// fallback to cublas, can be slow` comment) — a one-shot log warning
  when `cublas_mm_out` is taken for an in-range `num_tokens` would surface
  the next uncovered hidden dim immediately.
- **Generalizes to:** `pattern-fast-path-silent-fallback` — a custom fast
  kernel is gated on exact compile-time shapes and new models silently miss
  it. Carries to: any shape/arch/dtype-gated custom op whose fallback is a
  generic library call (new hidden dims, expert counts, or head dims);
  min-latency kernels reused by a new model family without re-checking the
  instantiation list; dtype gates (BF16-only paths) when a model ships FP16;
  arch gates that fall back to pre-Hopper kernels on newer GPUs.
