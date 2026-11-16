---
id: case-dsa-indexer-host-overhead
type: regression-case
family: execution-and-graph
module: attention-fmha
maturity: full
regression_class: [host-work-added, sync-introduced]
signals: [host-time-increase, itl-increase, gpu-idle-between-steps, many-small-kernels, throughput-drop]
subsystems: [attention-kernel, cuda-graph]
introduced_via: [pre-existing-gap]
phase: [any-phase]
patterns: [pattern-host-work-on-hot-path, pattern-per-step-sync-added]
nvbugs: ["5983390"]
commits: ["6601758d3a8a", "d8eb3a601c9d", "edbb4b2c1d3d", "ae84aaddb6f1", "7e477ba8bfe8"]
success_prs: [12322, 12445, 12581, 12631, 12503]
failed_prs: []
---

# DSA indexer host overhead and per-step syncs dominate DeepSeek-V3.2 latency

> Part of the [Attention & FMHA regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `5983390` · commit `6601758d3a8a` · PR #12322 —
  Kernel fusions in `_gather_k_cache_for_chunk` of Indexer in DSA;
  related: nvbug `5983390` · commit `d8eb3a601c9d` · PR #12445 — Remove
  redundant D2H sync to optimize perf;
  related: nvbug `5983390` · commit `7e477ba8bfe8` · PR #12503 — Split MLA
  DSA custom op for piecewise CUDA graph capture;
  related: nvbug `5983390` · commit `edbb4b2c1d3d` · PR #12581 — Multiple
  host perf optimizations for DSA part;
  related: nvbug `5983390` · commit `ae84aaddb6f1` · PR #12631 — Reduce host
  overhead in DSA MLA attention path.
- **Symptom:** Decode throughput/latency below expectation on DeepSeek-V3
  (.2) DSA sparse attention, 8xB200 TP8/EP8 with piecewise CUDA graph;
  surfaced via nsys profiles: ~65% of per-iteration GPU time in
  DSA-launched kernels (PR #12322), and 2× 12–15 ms `cudaStreamSynchronize`
  stalls (~25 ms bubble) per context forward step (PR #12445). PR #12322
  alone recovered +6.5% system throughput and -7.8% ITL p95 at conc=16.
- **Root cause:** Under piecewise CUDA graph the DSA attention region is
  the only eagerly-executed part, so its host cost is fully exposed:
  (a) `_gather_k_cache_for_chunk` issued ~8–12 small eager PyTorch ops per
  chunk (arange, broadcast add, advanced indexing); (b) a debug assert in
  `_compute_slot_mappings` was guarded by `is_current_stream_capturing()`
  only, so its `.all()` forced a 1-byte D2H memcpy + stream sync on the
  eager GPU path, twice per context step; (c) step-invariant pool views and
  block-table slices recomputed per layer (~50 us × 2 per layer, ~4–5 ms per
  step) plus `torch.compile` closures re-traced per call in `mtp.py`;
  (d) per-layer `sum().item()` recomputed batch structure; (e) the monolithic
  `mla_custom_op_inplace` kept even token-wise projections out of capture.
- **How introduced:** unknown — not stated in the PRs; the DSA indexer path
  shipped with these costs (below-expectation new-backend perf, not a
  regression from a faster state).
- **Fix mechanism:** Fuse the gather into one Triton kernel, later replaced
  by C++ ops `trtllm::indexer_k_cache_gather_op` /
  `convert_req_index_to_global`; re-guard the assert on
  `block_indices_in_seq.is_cuda`; cache step-invariant views in
  `_ensure_pool_view_cached()`; pass precomputed `num_contexts` /
  `num_ctx_tokens` into `thop::attention`; split the MLA op into graph-safe
  `mla_dsa_proj` + eager `mla_dsa_attn_inplace` so projections are captured.
- **Detection signal:** nsys shows a dense stream of tiny kernels plus
  `cudaStreamSynchronize` inside the eager (non-graphed) attention region
  while graphed regions are idle-free; check
  `grep -n "\.item()\|\.all()\|\.cpu()" tensorrt_llm/_torch/attention_backend/sparse/dsa.py`
  and nsys `cuda_api_sum` for per-step `cudaStreamSynchronize`/`Memcpy DtoH`.
- **Prevention/guard:** PR #12581 added functional unit tests
  (`tests/unittest/_torch/attention/sparse/test_cpp_custom_ops.py`) but no
  guard counts launches or syncs per step — a per-iteration kernel-launch /
  D2H-sync budget check on the eager region is the remaining gap. Review
  rule: debug asserts touching GPU tensors must be device-guarded, not just
  capture-guarded.
- **Generalizes to:** pattern-host-work-on-hot-path and
  pattern-per-step-sync-added; carries to any new attention backend running
  eagerly inside a piecewise-graphed model (its host overhead becomes
  first-order), debug asserts calling `.all()`/`.item()` on GPU tensors in
  prepare/metadata paths, per-layer recomputation of step-invariant
  metadata, and `torch.compile`-decorated closures defined inside method
  bodies that re-trace every call.
