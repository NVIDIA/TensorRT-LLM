---
id: case-skip-sparse-path-when-degenerate
type: case
family: runtime-execution
maturity: full
bottleneck: [compute, launch]
signals: [attention-hot-path, small-batch-decode, long-context]
architectures: [any-sm]
model_scope: [model-agnostic, sparse-attention, mla, deepseek-v32]
phase: [any-phase]
patterns: [pattern-skip-degenerate-sparse-path]
accuracy_risk: lossless
apply_via_kind: [config-knob, env-var, code-change]
knobs: [skip_indexer_for_short_seqs, seq_len_threshold, TRTLLM_MLA_SHORT_SEQ_MHA_THRESHOLD, q_split_threshold]
specialists: [trtllm-serve-config-guide, perf-torch-cuda-graph-specialist]
commits: ['8f144d9282', '695d7a0bdd', '6f3acc0614']
interactions:
  - {case: case-sparse-mla-topk-attention, relation: depends-on, note: length-gates the sparse top-k machinery that case implements}
measured: []
---

# Skip the sparse/approximate path when it degenerates to the exact one

> Part of the [Runtime / Execution casebook](index.md) · schema: [case-template](../case-template.md)

- **Commits:** `8f144d9282` [feat] Skip DS-v3.2 indexer MQA and Top-K for short
  sequences (#9524); related: `695d7a0bdd` [perf] Short-sequence MHA optimization for
  DSA MLA prefill (#11677), `6f3acc0614` [perf] Long-sequence token-parallel indexer
  prefill (#11871, the opposite-regime move).
- **Applies when:** a **sparse / approximate** code path has a regime where it produces
  the **same result as the dense / exact** path but still pays the selection overhead.
  The archetype: DSA's sparse top-k over a sequence **shorter than the top-k budget**
  selects *every* token — so the lightning-indexer MQA + top-k are pure overhead, and
  dense attention is both faster and identical. Signals: a selection/approximation step
  whose selectivity → 1 below some size (short sequences, small expert counts, low
  concurrency); the sparse path carries extra machinery (indexer, absorption BMMs,
  larger head_dim) that dense attention avoids at that scale.
- **Mechanism:** length-gate the path. Below a KV-length threshold, **skip the sparse
  selection entirely** and either run dense attention (`forward_context` dense MHA:
  `kv_b_proj` expansion + standard fused attention, `topk_indices=None`, #11677) or fill
  the selection buffer with the **trivially-constructed exact result** (a dense causal
  index pattern, `prepare_dense_topk_indices` / `_get_dense_topk_indices`, #9524). This
  drops the whole indexer MQA-logits + top-k + absorption cost for short-sequence
  batches. Because the fast path **changes control flow**, add a **CUDA-graph-key
  dimension** (`short_seq_len_mode` in the graph key,
  `needs_separate_short_long_cuda_graphs()`) so short-only and long/mixed batches each
  keep a captured graph instead of falling back to eager.
- **Generalizes to:** the pattern "**detect when a sparse/approximate path degenerates
  to the dense/exact answer below a threshold and skip the selection work, substituting
  a trivially-constructed exact result — and add a graph-key dimension so the
  control-flow fast-path stays CUDA-graph-captured.**" Carries to: sparse/block-sparse
  or sliding-window attention that is dense below a length; MoE that effectively routes
  to all experts when `n_experts` is small; speculative decoding disabled where accept
  rate is ~0; any approximate reduction exact below N. The complementary **long-sequence
  regime** move travels with it: when the per-token selection work is *replicated* across
  TP ranks, **shard it along the token axis and all-gather** (`q_split_threshold`, #11871)
  — turning replicated work into divided work. Adapt by: finding the regime where
  approximate == exact (or where a different parallelization wins), constructing the exact
  result cheaply, and keeping every path graph-captured.
- **Apply via:** config knobs — `skip_indexer_for_short_seqs` / `seq_len_threshold`, env
  `TRTLLM_MLA_SHORT_SEQ_MHA_THRESHOLD` (short-path), `q_split_threshold` (long-path
  token-parallel); the graph-key split is code (`cuda_graph_runner.py`). Delegate the
  YAML knobs to **trtllm-serve-config-guide** and the graph-key/path work to
  **perf-torch-cuda-graph-specialist** / the model author.
- **Expected effect:** short-sequence batches skip the indexer/top-k/absorption overhead
  → lower latency at small scale with identical output; long-sequence prefill divides the
  indexer work across TP ranks. Direction only — measured Δ (short-batch step latency,
  long-prefill indexer time, throughput) to be recorded per regime.
- **Accuracy risk:** **lossless** — for sequences ≤ top-k, dense attention (or a dense
  causal index set) *is* the exact result the sparse path would compute (it would select
  all valid tokens); the token-parallel split is an exact data-parallel decomposition
  reconstructed by all-gather. The risk is a threshold/graph-key bug, not precision.
- **Verify:** correctness — short-path output parity vs a standalone dense reference
  (`test_short_seq_mha.py`) and vs the sparse path at the boundary length; long-path parity
  across `q_split_threshold` values; confirm both regimes stay graph-captured (no eager
  fallback). Perf — short-batch latency and long-prefill indexer time before vs after.
- **Rollback:** `skip_indexer_for_short_seqs=False` / raise the threshold / negative
  `q_split_threshold` to disable. Trigger: any boundary-length mismatch, an eager fallback
  regressing captured batches, or no benefit at the target lengths.
- **Prior art:** PRs #9524, #11677, #11871. Files:
  `tensorrt_llm/_torch/attention/backends/sparse/dsa/metadata.py` (`prepare_dense_topk_indices`),
  `tensorrt_llm/_torch/attention/backends/sparse/dsa/params.py`
  (`skip_indexer_for_short_seqs`, `q_split_threshold`),
  `tensorrt_llm/_torch/attention/attention.py` (`_should_use_short_mha`,
  `forward_context_dsa`), `tensorrt_llm/_torch/pyexecutor/cuda_graph_runner.py`
  (`short_seq_len_mode`). Owning skills: **trtllm-serve-config-guide**,
  **perf-torch-cuda-graph-specialist**. Related: the
  [sparse-MLA top-k attention](../kernel-and-fusion/sparse-mla-topk-attention.md) and
  [top-k selection](../kernel-and-fusion/specialize-topk-selection-kernel.md) cases (the
  machinery this skips), and [remove attention-DP padding](attention-dp-padding.md)
  ("process only the real work" sibling).
