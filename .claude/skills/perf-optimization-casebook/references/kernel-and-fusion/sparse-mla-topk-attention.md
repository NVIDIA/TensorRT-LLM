---
id: case-sparse-mla-topk-attention
type: case
family: kernel-and-fusion
maturity: full
bottleneck: [compute, memory]
signals: [attention-hot-path, kv-reads-dominate, long-context, high-concurrency]
architectures: [sm100]
model_scope: [sparse-attention, mla, deepseek-v32, model-agnostic]
phase: [any-phase]
patterns: [pattern-attend-selected-kv-subset, pattern-write-direct-into-paged-cache, pattern-early-exit-ragged-bounds]
accuracy_risk: mixed
apply_via_kind: [kernel-change]
knobs: [kv_cache_config.dtype]
specialists: [kernel-cuda-specialist, perf-nsight-compute-analysis]
commits: ['f0dc746738', '497a07021d', '389b73c349']
eligibility:
  - "model is trained with sparse attention (DSA-style selection is the intended computation)"
  - "sm >= 100 (trtllm-gen sparse MLA kernels)"
  - "sparse_mla_topk < typical seqLenKv (else selection degenerates to dense plus overhead)"
interactions:
  - {case: case-specialize-topk-selection-kernel, relation: depends-on, note: producer of the top-k indices this kernel consumes}
  - {case: case-fp8-mla-kv-cache, relation: composes-with, note: the quantized-KV variant rides on it (lossy part)}
  - {case: case-split-mla-reduction-kernel, relation: composes-with, note: "independent levers on the same FMHA reduction; they co-apply — verify the combined effect"}
  - {case: case-skip-sparse-path-when-degenerate, relation: composes-with, note: guards the short-seq degenerate regime}
measured: []
---

# Attend only to the top-k–selected KV in the attention kernel (sparse MLA)

> Part of the [Kernel & Fusion casebook](index.md) · schema: [case-template](../case-template.md)

- **Commits:** `f0dc746738` [feat] Add trtllm-gen sparse MLA kernels to support
  per-Tensor FP8 KV Cache (#8692); related: `497a07021d` optimized sparse mla
  kernels && fix unspecified cuda launch (#8866), `389b73c349` [fix] Remove FP8
  K/V buffer from TRTLLM sparse MLA attention kernel (#9529).
- **Applies when:** an attention layer only needs to attend to a *selected subset*
  of the KV history — a natively-sparse-attention model (DeepSeek V3.2's DSA,
  where a lightning indexer picks the top-k KV per query), or block-sparse /
  sliding-window / landmark attention. Signals: full-sequence FMHA is the hot
  path yet most KV positions are masked out or unselected; long context / high
  concurrency where reading all KV is wasteful; MLA on SM100+ trtllm-gen FMHA.
  The selection indices are produced upstream (see the [top-k selection kernel
  case](specialize-topk-selection-kernel.md)); this case is the *consumer* — the
  attention kernel that uses them.
- **Counter-signals:** models **not** trained with sparse attention — feeding a
  dense-trained model selected-KV attention changes the computation itself (a
  real accuracy change, out of scope for this precedent); short sequences where
  `seqLenKv <= sparse_mla_topk` (selection degenerates to dense plus selection
  overhead — see [skip-sparse-path](../runtime-execution/skip-sparse-path-when-degenerate.md));
  non-MLA / non-trtllm-gen attention backends where the sparse dispatch does
  not exist.
- **Mechanism:** feed the FMHA generation kernel the selected KV page indices
  (`kvPageIdxPtr = sparse_attn_indices`) and bound its KV loop by
  `maxAttentionWindow = min(maxSeqLenKv, sparse_mla_topk)`, so it does O(top-k)
  work instead of O(seqLen). Three reinforcing moves: (1) run MLA in
  **absorption/latent space** so the RoPE kernel writes K/V straight into the
  paged KV cache (latent dims `kv_lora_rank+qk_rope_head_dim`=576 /
  `kv_lora_rank`=512) with per-tensor FP8, no separate materialization
  (`applyMLARopeAndAssignQKVKernelOptContext`, `absorption_mode`); (2) in the
  cross-CTA reduction, **early-exit CTAs whose query index exceeds the ragged
  `seqLenQ`** and clamp `seqLenKv = min(seqLenKv, sparse_mla_topk)` so the reduce
  only spans selected tiles; use exact page granularity (`numTokensPerPage=1`)
  for the scattered top-k gather (#8866); (3) **drop the redundant FP8 K/V scratch
  buffers** once K/V already live in the paged cache
  (`fp8_k_buf_size=fp8_v_buf_size=0`, `kPtr=vPtr=nullptr`) (#9529).
- **Generalizes to:** the pattern "**make the attention kernel operate on a
  selected subset of KV rather than the full sequence** — pass the selected
  page/block indices and bound the kernel's KV loop and reduction by the
  selection count." Carries to: block-sparse attention, sliding-window / streaming
  attention, landmark / NSA-style selection, any masked attention where the mask
  is structured enough to skip whole tiles. Two reusable corollaries travel with
  it: **write K/V straight into the paged cache** (absorption/latent layout) to
  avoid a materialization pass, and **early-exit CTAs beyond the ragged
  per-sequence length** so padded work is free. Adapt by: choosing the kernel's
  selection-index input (page vs token indices), setting the loop/reduction bound
  to the selection budget, and matching page granularity to how scattered the
  access is.
- **Apply via:** **not a server knob** — an FMHA kernel/dispatch change,
  auto-selected for DSA/sparse-MLA on SM100+ (`useSparseMLA()`, `mSparseMla`,
  `forward_absorption` / `forward_generation_dsa`). The selection budget
  (`sparse_mla_topk`) is a model/config property; the per-tensor FP8 KV rides on
  `kv_cache_config.dtype` (see [FP8 MLA KV](fp8-mla-kv-cache.md)). Delegate kernel
  work to **kernel-cuda-specialist**; profile with **perf-nsight-compute-analysis**.
- **Expected effect:** attention work drops from O(seqLen) to O(top-k) → lower
  MLA decode/prefill latency and KV bandwidth at long context; reclaimed FMHA
  workspace from the dropped scratch buffers. Direction only — measured Δ (attn
  kernel time, KV bytes/step, workspace) to be recorded from the run.
- **Accuracy risk:** **mixed.** The sparse *dispatch* itself is lossless relative
  to the model's sparse-attention specification (DSA is trained sparse — attending
  to the selected top-k **is** the intended computation, not an approximation the
  optimization introduces). The **per-tensor FP8 (later FP4) KV cache is lossy**
  in exactly the way [FP8 MLA KV](fp8-mla-kv-cache.md) is → accuracy record +
  rollback when the quantized-KV variant is enabled. The dropped scratch buffer
  (#9529) is lossless (K/V read from cache instead of a duplicate).
- **Verify:** correctness — sparse-MLA forward parity vs the reference sparse
  computation (`test_sparse_mla_forward.py`); DSA e2e accuracy (GSM8K / GPQA
  references) unchanged, and for FP8/FP4 KV a parity check vs BF16 KV. Perf —
  attn kernel time and KV bytes/step before vs after; confirm the reduction spans
  only top-k tiles and padded CTAs early-exit (ncu).
- **Rollback:** fall back to the dense MLA FMHA path (and restore FP8 K/V scratch
  if the buffer removal is implicated). Trigger: sparse-kernel correctness
  mismatch, FP8/FP4-KV accuracy drop beyond the recorded threshold, or the
  sparse-MLA kernel unavailable for the SM/head config.
- **Prior art:** PRs #8692, #8866, #9529. Files:
  `cpp/tensorrt_llm/kernels/mlaKernels.cu`
  (`applyMLARopeAndAssignQKVKernelOptContext`, `absorption_mode`),
  `cpp/tensorrt_llm/kernels/fmhaDispatcher.cpp` (`useSparseMLA()`, `mSparseMla`),
  `cpp/tensorrt_llm/kernels/.../fmhaReduction.cu`,
  `cpp/tensorrt_llm/common/attentionOp.cpp` (`mFP8ContextMLA`, K/V buffer sizing),
  `tensorrt_llm/_torch/attention/attention.py` (`forward_absorption`,
  `forward_generation_dsa`), `tensorrt_llm/_torch/attention/backends/sparse/dsa.py`
  (`sparse_mla_topk`). Owning specialist: **kernel-cuda-specialist**. Related: the
  [top-k selection kernel](specialize-topk-selection-kernel.md) (the producer of
  the indices), [split MLA reduction](split-mla-reduction-kernel.md) (the same
  FMHA reduction, a different lever), and [FP8 MLA KV](fp8-mla-kv-cache.md) (the
  quantized-KV precursor).
