---
id: case-piecewise-cuda-graph
type: case
family: runtime-execution
maturity: full
bottleneck: [launch]
signals: [many-small-kernels, slow-path-fallback]
architectures: [any-sm]
model_scope: [model-agnostic]
phase: [decode]
patterns: [pattern-graph-the-graphable]
accuracy_risk: lossless
apply_via_kind: [config-knob]
knobs: [torch_compile_piecewise_cuda_graph, cuda_graph_batch_sizes]
specialists: [perf-torch-cuda-graphs, perf-torch-cuda-graph-specialist]
commits: ['91bf5e6a8e73']
eligibility:
  - "requires torch_compile_enabled; asserts torch_compile_fullgraph=True"
measured: []
---

# Piecewise CUDA Graph — capture stable regions, leave dynamic/unsupported ops out

> Part of the [Runtime / Execution casebook](index.md) · schema: [case-template](../case-template.md)

- **Commits:** `91bf5e6a8e73` Add Piecewise CUDA Graph Support (#3804).
- **Applies when:** launch-bound generation where full-iteration CUDA-graph capture is blocked by ops that can't/shouldn't be captured (notably the fused `attention` op). Signals: launch overhead dominates between kernels; full graph unavailable; you already use `torch.compile`.
- **Mechanism:** splits the compiled FX graph at the attention boundary and CUDA-graph-captures only the contiguous stable submodules around it, leaving attention eager. Splits on `torch.ops.trtllm.attention` (rewritten to `attention_inplace`, no new tensor → safe resume); attention submods go in `exclude_modules_id`; a `PiecewiseInterpreter` wraps each non-excluded `submod` in a `PiecewiseRunner` capturing one graph per `cuda_graph_batch_sizes` entry. All pieces share one `graph_pool_handle`. Each piece replays as a single launch.
- **Generalizes to:** "graph the graphable, run the rest eagerly"; carries to any region broken by a dynamic-shape op, data-dependent control flow, a non-capturable custom op, or spec-decode bookkeeping; adapt by choosing the split op, ensuring safe resume after it (in-place/no-new-tensor), and enumerating token-count buckets.
- **Apply via:** config `torch_compile_piecewise_cuda_graph: true` (requires `torch_compile_enabled`, asserts `torch_compile_fullgraph=True`) + `cuda_graph_batch_sizes`; flags `--use_torch_compile --use_piecewise_cuda_graph`. Delegate to **perf-torch-cuda-graphs** / **perf-torch-cuda-graph-specialist**.
- **Expected effect:** reduced launch overhead / lower per-step latency by graphing stable regions; attention stays eager; no number — measured Δ to be recorded from run.
- **Accuracy risk:** lossless — graph replay reproduces the same kernels. Care: a warmup flag `set_enable_piecewise_cuda_graph_capture_flag(False)` disables capture during runs that "would produce wrong results"; token counts not captured fall back to non-graph.
- **Verify:** confirm pieces captured/replayed (ad-conf-check); nsys graph coverage vs eager attention; outputs vs eager and full-graph.
- **Rollback:** `torch_compile_piecewise_cuda_graph=false`. Trigger: capture errors, wrong warmup results, or token counts persistently missing from `cuda_graph_batch_sizes`.
- **Prior art:** PR #3804. Files: `_torch/compilation/piecewise_optimizer.py`, `compilation/backend.py`, `pyexecutor/config.py`. Owning specialist: **perf-torch-cuda-graph-specialist**.
