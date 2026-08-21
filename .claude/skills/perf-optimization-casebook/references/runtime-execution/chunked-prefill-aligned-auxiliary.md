---
id: case-chunked-prefill-aligned-auxiliary
type: case
family: runtime-execution
maturity: full
bottleneck: [memory]
signals: [memory-capacity-bound, long-context, high-concurrency]
architectures: [any-sm]
model_scope: [model-agnostic, mla, sparse-attention, deepseek-v32]
phase: [prefill]
patterns: [pattern-chunk-prefill-align-auxiliary]
accuracy_risk: lossless
apply_via_kind: [config-knob, code-change]
knobs: [enable_chunked_prefill]
specialists: [trtllm-serve-config-guide, kernel-cuda-specialist]
commits: ['b10137fdd5', '78bb245554']
interactions:
  - {case: case-free-mla-intermediates, relation: alternative-to, note: the other prefill-memory lever}
  - {case: case-auxiliary-cache-in-kv-manager, relation: depends-on, note: the chunk gather reads from that cache}
measured: []
---

# Chunked prefill to bound memory — and align an auxiliary structure's chunking to it

> Part of the [Runtime / Execution casebook](index.md) · schema: [case-template](../case-template.md)

- **Commits:** `b10137fdd5` [feat] Support MLA chunked prefill for DeepSeek V3.2
  (#9376); related: `78bb245554` [fix] Better align MLA chunking with indexer
  chunking for DSV32 (#10552).
- **Applies when:** long-context **prefill** where materializing the full sequence at
  once blows up activation/logit memory — MLA context attention over a long ISL, and
  especially a **quadratic** intermediate (the DSA indexer's full-sequence MQA logits
  are O(L²)). Signals: prefill OOM or capped ISL at high concurrency; a large
  full-sequence intermediate on the prefill path; you want to overlap prefill with
  decode. Most acute when an **auxiliary structure** (sparse indexer, landmark
  selector) also materializes something per-sequence.
- **Mechanism:** split a long prefill into fixed-size **KV chunks** so peak
  activation/logit memory is bounded by the chunk size, not the sequence length. MLA
  context attention runs chunk-by-chunk against cached KV
  (`enable_context_mla_with_cached_kv`); the RoPE kernel recomputes a **cache-aware
  position offset per chunk** (`cached_offset = cache_seq_len − current_seq_len`) so
  positions are correct when earlier chunks are already cached
  (`applyMLARopeAndAssignQKVKernelOptContext`). The reusable subtlety: the **auxiliary
  indexer must gather only the *current chunk's* K** from cache
  (`_gather_k_cache_for_chunk`, `slot_mapping_*_fullkv`) instead of the full sequence,
  and it must **inherit MLA's chunk boundaries** rather than running its own second
  chunking scheme — #10552 makes the indexer treat each MLA chunk as one indexer chunk
  (dropping a redundant `host_cached_tokens.sum().item()` host sync and a
  boundary-mismatch class).
- **Generalizes to:** the pattern "**chunk long prefill into fixed windows to bound
  the activation/logit footprint to chunk size; when an auxiliary structure exists, it
  must (a) gather only the current chunk's data, (b) recompute cache-aware position
  offsets per chunk, and (c) inherit the primary chunker's boundaries — never run two
  chunking schemes over the same data.**" Carries to: chunked prefill for standard
  MHA/GQA, chunked context with sparse/landmark/retrieval selection, any per-chunk
  auxiliary state; and the general rule "when two subsystems chunk the same sequence,
  the secondary defers to the primary's boundaries." Adapt by: choosing the chunk size
  (memory vs launch trade-off), making every auxiliary gather chunk-local, and aligning
  boundaries.
- **Apply via:** config `enable_chunked_prefill` (+ the chunk-size knob); the
  auxiliary-alignment is a code change on the auxiliary path. Delegate the YAML knob to
  **trtllm-serve-config-guide**; the chunk-local auxiliary gather + boundary alignment
  to **kernel-cuda-specialist** / the model author.
- **Expected effect:** bounded prefill memory → longer ISL / higher concurrency fits,
  and prefill can overlap decode; the quadratic indexer intermediate becomes per-chunk.
  Direction only — measured Δ (peak prefill memory, max ISL/concurrency, TTFT) to be
  recorded from a long-context run.
- **Accuracy risk:** **lossless** — chunking is a tiling/scheduling of an exact
  computation; the only risk is a boundary/offset bug (misaligned chunks or a wrong
  cache offset), which is a *correctness* failure, not precision loss. Alignment
  (#10552) exists precisely to keep the two chunkers consistent.
- **Verify:** correctness — DSA/MLA prefill output parity vs non-chunked (GSM8K or a
  logit check), especially at chunk boundaries and when part of the sequence is cached;
  confirm the indexer reads only the current chunk. Perf — peak prefill memory and max
  ISL/concurrency before vs after; TTFT.
- **Rollback:** `enable_chunked_prefill=False`. Trigger: accuracy mismatch at chunk
  boundaries, or no memory benefit at the target ISL.
- **Prior art:** PRs #9376, #10552. Files:
  `tensorrt_llm/_torch/attention/backends/sparse/dsa.py` (`_gather_k_cache_for_chunk`,
  `enable_context_mla_with_cached_kv`, `split_prefill_chunks`, chunk-spec alignment),
  `cpp/tensorrt_llm/kernels/mlaKernels.cu` (`applyMLARopeAndAssignQKVKernelOptContext`,
  cache offset). Owning skills: **trtllm-serve-config-guide** (knob),
  **kernel-cuda-specialist** (auxiliary gather). Related:
  [free MLA intermediates](free-mla-intermediates.md) (the other prefill-memory lever)
  and the [auxiliary-cache-in-manager case](auxiliary-cache-in-kv-manager.md) (the
  cache the chunk gather reads from).
