---
id: case-multi-stream-shared-routed-expert
type: case
family: runtime-execution
maturity: full
bottleneck: [launch, compute]
signals: [small-batch-decode, many-small-kernels]
architectures: [any-sm]
model_scope: [model-agnostic, moe, mla, deepseek-v3r1]
phase: [decode]
patterns: [pattern-overlap-independent-work]
accuracy_risk: lossless
apply_via_kind: [config-knob, code-change]
knobs: [cuda_graph_config, use_cuda_graph]
specialists: [perf-torch-cuda-graphs, perf-sweep-workflow, perf-torch-cuda-graph-specialist]
commits: ['4855431d3d4e']
eligibility:
  - "CUDA graphs must be enabled (gate: do_multi_stream = is_graph_capturing() and aux_stream is not None)"
interactions:
  - {feature: cuda-graphs, relation: depends-on, note: stream switching only pays off under graph capture/replay}
measured: []
---

# Overlap shared-expert and routed-expert (MoE) compute on a side CUDA stream

> Part of the [Runtime / Execution casebook](index.md) · schema: [case-template](../case-template.md)

- **Commits:** `4855431d3d4e` [Deepseek] Redesign multi-stream API (#3459).
- **Applies when:** launch/exposed-latency bound (GPU not saturated, small batch / low concurrency, min-latency mode) on a model with two independent sub-computations per layer — shared experts vs routed MoE, or the two MLA layernorms (q vs kv) / MLA `_run_bmm` vs `_concat_kv_cache`. Requires CUDA Graphs enabled, because stream switching has host overhead only hidden under graph capture/replay.
- **Mechanism:** issues two independent ops on separate CUDA streams (default = `fn0`, `aux_stream` = `fn1`) with `cuda.Event` record/wait pairs ordering them, so the GPU runs them concurrently instead of serializing — filling SM gaps a single small kernel leaves. Gate is literally `do_multi_stream = is_graph_capturing() and aux_stream is not None`.
- **Generalizes to:** "issue two data-independent kernels on a side stream to overlap them"; carries to MoE shared+routed split, MLA dual-layernorm, attention/MoE-chunking overlap (codebase enumerates `AuxStreamType = {Attention, MoeShared, MoeChunkingOverlap}`), and EPLB statistic/weight movement; adapt by identifying a true independent pair, allocating an aux stream + two events, and only enabling under graph capture.
- **Apply via:** `maybe_execute_in_parallel(fn0, fn1, event0, event1, aux_stream)` (`_torch/modules/multi_stream_utils.py`); aux streams created per `AuxStreamType` in model `__init__`. Operationally, enable by turning on CUDA Graphs (`cuda_graph_config` / `use_cuda_graph`) — multi-stream then self-activates. Delegate to **perf-torch-cuda-graphs**; measure with **perf-sweep-workflow**.
- **Expected effect:** lower per-iteration latency in the low-latency/small-batch regime (overlap of shared- and routed-expert work); no number — measured Δ to be recorded from run.
- **Accuracy risk:** lossless (pure scheduling; identical math, only ordered by events).
- **Verify:** TPOT/ITL + end-to-end latency at low concurrency with vs without CUDA Graphs; confirm two streams concurrently active in nsys. No parity check needed.
- **Rollback:** pass `aux_stream=None` (disables multi-stream) or disable CUDA Graphs — falls back to sequential `fn0(); fn1()`. Trigger: throughput regression at high batch (tuned for low latency), or no measured overlap.
- **Prior art:** PR #3459. Files: `_torch/modules/multi_stream_utils.py` (`maybe_execute_in_parallel`), `modeling_deepseekv3.py` (`Deepseekv3MoE.forward`), `_torch/attention/mla.py` (MLA), `_torch/utils.py` (`AuxStreamType`, `EventType`). Owning specialist: **perf-torch-cuda-graph-specialist**.
