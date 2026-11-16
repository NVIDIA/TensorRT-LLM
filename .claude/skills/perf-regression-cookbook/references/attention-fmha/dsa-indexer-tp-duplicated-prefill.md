---
id: case-dsa-indexer-tp-duplicated-prefill
type: regression-case
family: execution-and-graph
module: attention-fmha
maturity: full
regression_class: [redundant-cross-rank-work]
signals: [ttft-increase, throughput-drop]
subsystems: [attention-kernel]
introduced_via: [pre-existing-gap]
phase: [prefill]
patterns: [pattern-redundant-work-across-ranks]
nvbugs: ["5892646"]
commits: ["6f3acc0614b2"]
success_prs: [11871]
failed_prs: []
---

# DSA indexer prefill duplicated on every TP rank in long-context chunked prefill

> Part of the [Attention & FMHA regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `5892646` · commit `6f3acc0614b2` · PR #11871 —
  `[perf] Long-sequence token-parallel optimization for DSA indexer prefill`.
- **Symptom:** High TTFT for long-context DeepSeek-V3.2 (DSA sparse attention)
  chunked prefill under attention TP; PR reports up to 2.3× TTFT improvement
  after the fix for a 150k-ISL sequence with 32k chunk size at concurrency 1,
  TEP8 (nvfp4 checkpoint) on a single DGX node with 8×B200.
- **Root cause:** In the `Indexer.sparse_attn_indexer` chunked-prefill path
  (invoked from `Indexer.forward`;
  `tensorrt_llm/_torch/attention_backend/sparse/dsa.py`), every TP rank ran
  the full indexer computation — `fp8_mqa_logits` plus top-k over
  `q_fp8[chunk.token_start:chunk.token_end]` — for the *entire* q-token range
  of each chunk. With TP applied to attention, the same indexer work was
  repeated `tp_size` times; wasted compute scales with world size and with
  context length.
- **How introduced:** No regressing commit is named — the DSA indexer
  chunked-prefill path never split q tokens across ranks (pre-existing gap in
  the DeepSeek-V3.2 sparse-attention implementation, not a regression from a
  previously faster state).
- **Fix mechanism:** Token parallelism for the indexer: when the number of
  packed tokens in a prefill chunk exceeds `q_split_threshold` (default 8192,
  configurable in `DeepSeekSparseAttentionConfig`; negative disables), each
  rank computes logits/top-k only for its contiguous q-token slice
  (`chunk_num_token * tp_rank // tp_size` ..), then an `allgather` over
  `metadata.mapping` (dim 0, per-rank sizes) reassembles the full
  `topk_indices_buffer`. Stateless across iterations; skipped when
  `enable_attention_dp` or `tp_size == 1`.
- **Detection signal:** nsys prefill trace shows identical-duration
  `fp8_mqa_logits` / indexer top-k kernels on every TP rank (no per-rank
  shrink as TP grows), and long-context TTFT does not improve with TP size;
  check the knob with
  `grep -n "q_split_threshold" tensorrt_llm/_torch/attention_backend/sparse/dsa.py tensorrt_llm/llmapi/llm_args.py`
  and confirm `sparse_attention_config.q_split_threshold >= 0` in the serve
  YAML.
- **Prevention/guard:** PR adds a DeepSeek-V3.2 chunked-prefill end-to-end
  test in `tests/integration/defs/accuracy/test_llm_api_pytorch.py` with
  `q_split_threshold=0` (split always on, max_num_tokens=512) — a correctness
  guard for the split path; no perf bar guards against the duplication class
  itself.
- **Generalizes to:** `pattern-redundant-work-across-ranks` — the same
  computation duplicated on every rank instead of split. Carries to: other
  auxiliary computations replicated under TP (spec-decode drafting, routing /
  gating preprocessing); per-rank re-tokenization or input preprocessing in
  multi-rank serving; KV-cache side computations (indexers, block selection)
  that sit outside the TP-sharded attention kernel; DP-attention variants
  where the eligible split axis differs (here the split is disabled under
  `enable_attention_dp`).
