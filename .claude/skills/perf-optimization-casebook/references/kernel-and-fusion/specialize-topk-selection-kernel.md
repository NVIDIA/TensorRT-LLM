---
id: case-specialize-topk-selection-kernel
type: case
family: kernel-and-fusion
maturity: full
bottleneck: [launch, compute]
signals: [many-small-kernels, recompilation-churn]
architectures: [any-sm]
model_scope: [sparse-attention, deepseek-v32, model-agnostic]
phase: [decode]
patterns: [pattern-specialize-hotpath-topk]
accuracy_risk: lossless
apply_via_kind: [kernel-change, config-knob]
knobs: [use_cute_dsl_topk, enable_heuristic_topk]
specialists: [kernel-cuda-specialist, kernel-cute-writing, perf-nsight-compute-analysis]
commits: ['2e7769d1e8', '941a54c66a', 'e940e58eb9', 'b206f682f5', '29fac6b673', 'b41876f144']
interactions:
  - {case: case-sparse-mla-topk-attention, relation: composes-with, note: the attention kernel is the consumer of the indices this kernel produces}
measured: []
---

# Specialize the hot-path top-k / selection kernel (don't ship a generic sort)

> Part of the [Kernel & Fusion casebook](index.md) · schema: [case-template](../case-template.md)

- **Commits:** `2e7769d1e8` [feat] Add customized topk and unit tests for DSA
  (#8882); related: `941a54c66a` Update the indexer topK (#9255), `e940e58eb9`
  [perf] Enable CuteDSL indexer_top_k in model (#12236), `b206f682f5` Indexer
  TopK: single-block / multi-pass radix (#14268), `29fac6b673` Temporally-
  Correlated Heuristic-guided Indexer TopK (#12385), `b41876f144` DSv4 prep:
  IndexerTopK primitives (#15381).
- **Applies when:** a **top-k / selection** runs on the hot path — DSA's indexer
  picks the top-k KV per query every step; also MoE grouped-top-k routing, sampling
  top-k/top-p, beam search. Signals: a `torch.topk(...)` (a full sort) plus a chain
  of `arange` / `masked_fill` / cast / index-arithmetic ops in the profile;
  Triton/PyTorch selection recompiling or launching many small kernels; the op
  runs at every decode step so its launch + sort cost is disproportionate. You
  only need the top-k **indices** (and then post-process them), not a full sort.
- **Mechanism:** replace the general sort + its downstream index/mask arithmetic
  with **one custom radix-select kernel** that emits exactly the top-k indices
  (and does the causal mask / local-index conversion inline) — a histogram/radix
  select over the float bit-pattern (e.g. 11/11/10-bit passes) avoids the
  O(N log N) sort and the mask-tensor round-trips (#8882). Then **keep several
  exact implementations and dispatch per regime**: single-CTA vs single-pass
  multi-CTA (CuteDSL, chosen by whether the launch fits **one SM wave**
  `num_rows·ctas_per_group ≤ num_sms`, folding dtype-dependent sync cost — fp32 =
  4 radix rounds vs fp16/bf16 = 2 — into the crossover, #12236); single-block vs
  multi-pass radix with per-row DRAM ping-pong scratch for large N (#14268). Two
  **short-circuits** cut passes without changing the result: a **coarse
  low-precision first pass** (fp16 histogram) that resolves the common case before
  the exact fp32 passes and falls back when it can't disambiguate (#9255), and a
  **warm-start from the previous decode step's selection** — seed the k-th-value
  threshold search from the prior step's top-k, since consecutive autoregressive
  steps select nearly the same tokens (#12385).
- **Generalizes to:** the pattern "**the hot-path top-k/selection is a specialized
  kernel, not a generic sort — fuse the selection with its index/mask
  post-processing into one radix-select, keep multiple exact variants, and dispatch
  by `(k, N, dtype, SM-wave-occupancy)`.**" Two reusable short-circuit sub-patterns
  travel with it: **cheap-check-first** (a coarse low-precision pass that resolves
  the common case, exact fallback otherwise) and **warm-start from the previous
  iteration** when the answer drifts slowly across steps (autoregressive decode —
  also applies to per-step threshold/quantile/routing searches). Carries to: MoE
  grouped-top-k routing (cf. the [routing-kernel case](fuse-moe-routing-kernel.md)),
  sampling top-k/top-p, beam-search top-k, argmax/threshold selection. Adapt by:
  writing the radix-select for your `k`; choosing the parallelization tier by
  N/occupancy; and pre-allocating any scratch so the op is CUDA-graph-capturable
  (#14268 exposes `scratch` for graph-stable addresses).
- **Apply via:** **not a server knob** for the core kernel (auto-dispatched);
  opt-in variants are config-gated — `use_cute_dsl_topk`, `enable_heuristic_topk`
  (`DeepSeekSparseAttentionConfig` in `llm_args.py`). Delegate the CUDA-C++
  radix-select to **kernel-cuda-specialist**, the CuteDSL variant to
  **kernel-cute-writing**; profile with **perf-nsight-compute-analysis**.
- **Expected effect:** lower per-step selection time (no full sort, fewer launches,
  fewer passes) → smaller GPU-idle and higher decode throughput where selection is
  hot. PR #12236 reports ~1.14× over always-single-CTA for the CuteDSL path; treat
  as the author's figure and **record the local Δ** (selection kernel time,
  distinct-compile count, throughput) from the run — do not present a PR number as
  a locally-achieved result.
- **Accuracy risk:** **lossless** — every variant returns the exact top-k (the
  coarse pass only brackets and falls back to exact; radix-select is exact). One
  flag: the temporal warm-start (#12385) has a bounded refinement loop that
  *targets* exactness (unit test asserts parity with `torch.topk` even with poor
  seeds) but could deviate under pathological non-convergence — **verify accuracy
  on first enable** of `enable_heuristic_topk`.
- **Verify:** correctness — kernel output equals `torch.topk` on the same logits,
  including edge cases (ties, all-equal, N<k); DSA e2e accuracy unchanged; re-check
  when enabling the heuristic/CuteDSL variants. Perf — selection kernel time and
  launch/compile count before vs after; confirm the chosen tier matches the
  `(k,N,dtype)` regime (ncu / dispatch logs).
- **Rollback:** fall back to the reference `torch.topk` path (kept as reference/
  fallback), or disable the opt-in variant (`use_cute_dsl_topk=False`,
  `enable_heuristic_topk=False`). Trigger: any index mismatch vs reference or a
  perf regression in an untuned `(k,N)` regime.
- **Prior art:** PRs #8882, #9255, #12236, #14268, #12385, #15381. Files:
  `cpp/tensorrt_llm/kernels/indexerTopK.cu`, `heuristic_topk.cuh`, `IndexerTopK.h`,
  `cpp/tensorrt_llm/thop/IndexerTopKOp.cpp`,
  `tensorrt_llm/_torch/attention/backends/sparse/dsa/indexer.py`,
  `.../custom_ops/cute_dsl_kernels/blackwell/top_k/single_pass_multi_cta_radix_topk.py`.
  Owning specialists: **kernel-cuda-specialist**, **kernel-cute-writing**. Related:
  the [MoE routing-kernel case](fuse-moe-routing-kernel.md) (same "collapse a
  selection op-chain into one per-token kernel" idea) and the
  [sparse-MLA top-k attention case](sparse-mla-topk-attention.md) (the consumer of
  these indices).
