---
id: case-split-custom-op-for-piecewise-capture
type: case
family: runtime-execution
maturity: full
bottleneck: [launch]
signals: [many-small-kernels, slow-path-fallback]
architectures: [any-sm]
model_scope: [model-agnostic, sparse-attention, mla]
phase: [decode]
patterns: [pattern-split-op-at-graphable-seam]
accuracy_risk: lossless
apply_via_kind: [code-change]
knobs: []
specialists: [perf-torch-cuda-graph-specialist, kernel-cuda-specialist]
commits: ['7e477ba8bf', 'b72ee4fd89']
eligibility:
  - "pays off only with torch_compile_enabled + torch_compile_piecewise_cuda_graph on"
interactions:
  - {case: case-piecewise-cuda-graph, relation: depends-on, note: "producer side of the same mechanism; pays off only with piecewise capture on"}
measured: []
---

# Split a monolithic custom op at the graphable/non-graphable seam for CUDA-graph capture

> Part of the [Runtime / Execution casebook](index.md) · schema: [case-template](../case-template.md)

- **Commits:** `7e477ba8bf` [perf] Split MLA DSA custom op for piecewise CUDA graph
  capture (#12503); related: `b72ee4fd89` [fix] Route DSA attention through the MLA
  custom op for torch.compile compatibility (#12186, the precondition).
- **Applies when:** a single large fused **custom op** mixes graph-capturable and
  non-capturable work, so piecewise CUDA graph must leave the *whole* op eager and you
  lose capture coverage on the graphable part. Signals: launch-bound generation on the
  `torch.compile` + piecewise-CUDA-graph path; one custom op that does both shape-static
  token-wise math (projections, RoPE, quant) **and** batch/data-dependent work
  (KV-cache scatter, top-k, length-adaptive branches); the op is a partition boundary so
  its graphable half never gets captured. Prerequisite failure mode: the op is called
  **directly in eager Python** (bypassing its registered `torch.ops` entry), so
  `torch.compile` can't even trace it (#12186).
- **Mechanism:** first make the op **traceable** — route it through its registered
  `torch.library` custom op (`mla_custom_op_inplace`) instead of calling the impl
  directly, so `torch.compile` sees one opaque node (#12186). Then **split it at the
  seam** (#12503): hoist the **shape-static, metadata-free, token-wise** compute into a
  capturable op (`trtllm::mla_dsa_proj` — `kv_a_proj`, layernorms, `q_b_proj`, the
  indexer's `pre_indexer_proj`: cublas mm + RoPE + FP8 quant + weight scale; **no
  batch metadata, no KV-cache access**), and leave the **batch/data-dependent** work as
  a separate op **excluded** from capture (`trtllm::mla_dsa_attn_inplace` —
  `sparse_attn_indexer`'s K-cache scatter + top-k + attention dispatch). Register the
  attention op as a partition boundary in `piecewise_optimizer.py`. Two enabling tricks
  keep the capturable half **unconditional**: a **constant-arity `register_fake`** (Op 1
  always returns the same number of tensors) and **forcing straight-line control flow
  under compile** (`should_use_short_mha()` returns `False` when `is_torch_compiling()`,
  so a length-adaptive branch doesn't fork the traced graph).
- **Generalizes to:** the pattern "**split a monolithic custom op at the
  graphable/non-graphable boundary — put the shape-static, metadata-free compute in its
  own capturable op and leave the batch/data-dependent part as a separate excluded op —
  and force straight-line control flow under compile (constant-arity `register_fake`,
  disable length-adaptive branches) so the capturable half is unconditional.**" The
  precondition — "**route work through its registered `torch.ops` custom op, never call
  the impl directly in eager, or `torch.compile` can't trace it**" — is a reusable gotcha
  on its own. Carries to: attention wrappers mixing projection + cache access, MoE
  dispatch+compute+combine, any fused op with a data-dependent tail. This is the
  **op-authoring (producer) side** of piecewise CUDA graph, complementary to
  [turning piecewise capture on](piecewise-cuda-graph.md) (the consumer side). Adapt by:
  locating the seam (which sub-steps touch batch metadata / the cache / data-dependent
  branches), splitting into two registered ops, marking the data-dependent one a
  boundary, and making the capturable op's signature/control-flow constant under compile.
- **Apply via:** **not a server knob** — custom-op authoring; it pays off only with
  `torch_compile_enabled` + `torch_compile_piecewise_cuda_graph` on (see the
  [piecewise case](piecewise-cuda-graph.md)). Delegate to
  **perf-torch-cuda-graph-specialist** + **kernel-cuda-specialist** / the model author.
- **Expected effect:** the projection/RoPE/quant half gets captured into the CUDA graph
  (fewer launches / lower per-step latency); only the data-dependent attention runs
  eager. Direction only — measured Δ (graph coverage, per-step launch count, latency) to
  be recorded from the run.
- **Accuracy risk:** **lossless** — an op-boundary refactor; identical math, just split
  across two registered ops. Verify the split didn't drop/reorder a side effect (the
  in-place output arg must be marked in `inplace_info()`).
- **Verify:** correctness — DSA e2e accuracy unchanged with piecewise CUDA graph +
  torch.compile on (`test_nvfp4_multi_gpus_piecewise_cuda_graph`, GSM8K); outputs match
  the eager path. Perf — confirm Op 1 is captured and Op 2 is the boundary (ad-conf-check
  / nsys graph coverage); per-step launch count and latency before vs after.
- **Rollback:** disable piecewise capture (`torch_compile_piecewise_cuda_graph=false`)
  or fall back to the single-op eager path. Trigger: capture errors, an accuracy
  mismatch, or the split adding net overhead.
- **Prior art:** PRs #12503, #12186. Files:
  `tensorrt_llm/_torch/attention/backends/sparse/dsa/custom_ops.py`
  (`trtllm::mla_dsa_proj` + `register_fake`, `trtllm::mla_dsa_attn_inplace` +
  `register_fake`), `tensorrt_llm/_torch/attention/backends/sparse/dsa/module.py`
  (`forward_dsa_proj` / `_forward_dsa_attn`, `should_use_short_mha`),
  `tensorrt_llm/_torch/compilation/piecewise_optimizer.py`,
  `tensorrt_llm/_torch/compilation/utils.py` (`inplace_info`). Owning specialist:
  **perf-torch-cuda-graph-specialist**. Related: the
  [piecewise CUDA graph case](piecewise-cuda-graph.md) (the consumer side of the same
  mechanism).
