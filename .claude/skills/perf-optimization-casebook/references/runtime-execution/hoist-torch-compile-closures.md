---
id: case-hoist-torch-compile-closures
type: case
family: runtime-execution
maturity: full
bottleneck: [host-overhead]
signals: [recompilation-churn, gpu-idle-between-steps, host-prep-on-critical-path]
architectures: [any-sm]
model_scope: [model-agnostic, spec-decode, sparse-attention]
phase: [any-phase]
patterns: [pattern-stable-torch-compile-target]
accuracy_risk: lossless
apply_via_kind: [code-change]
knobs: []
specialists: [perf-host-optimization, perf-host-analysis, perf-torch-cuda-graphs]
commits: []
interactions:
  - {case: case-cache-step-invariant-per-layer, relation: composes-with, note: "same PR (#12581), same host-overhead theme"}
measured: []
---

# Hoist torch.compile closures out of hot methods (and drop torch.compile from trivial ops)

> Part of the [Runtime / Execution casebook](index.md) · schema: [case-template](../case-template.md)

- **Applies when:** host-overhead-bound serving where a hot per-call method
  **defines a `@torch.compile` (or `@maybe_compile`)-decorated function inside
  its body**, so a fresh compiled callable is created on every call; symptom is
  repeated graph tracing / recompilation (visible via `TORCH_LOGS=recompiles` or
  host time spent in Dynamo) and GPU-idle gaps. Also applies when a *trivial*
  single op (e.g. an in-place `x[:n] += 1`) is wrapped in `torch.compile` for no
  real benefit. Confirm host-bound first (`perf-host-analysis`). Instances:
  `mtp.py::forward()` and `dsa.py::prepare_dense_topk_indices()`.
- **Mechanism:** a `@torch.compile` decorator applied to a nested function
  rebinds a new optimized callable each call; the compiled-artifact cache keys
  off the function object, so per-call redefinition causes re-tracing/re-guarding
  (and can defeat caching entirely), adding host overhead every iteration.
  Hoisting the decorated function to a stable class method / module-level def
  lets the compiled artifact persist and be reused. For a one-line op the compile
  machinery costs more than it saves — run it eager.
- **Generalizes to:** the pattern "**a `torch.compile` target must be a stable,
  long-lived callable; never (re)define it on the hot path, and don't compile
  trivial work.**" Carries to: any `@torch.compile` / `torch.compile(...)`-wrapped
  closure created inside `forward`/`step`/`prepare`; lambdas passed to
  `torch.compile` per call; tiny elementwise/in-place ops wrapped "just in case."
  Adapt by: moving the decorated function to method/module scope and passing only
  the few tensors it needs (e.g. `seq_lens_cuda`, not the whole `attn_metadata`)
  so its guards stay small; or removing the decorator when the body is a single
  cheap op. Confirm host-bound and inspect recompiles first.
- **Apply via:** **not a server config knob** — a host-side code change. Detect
  with **perf-host-analysis** (+ `TORCH_LOGS=recompiles` / Dynamo timing);
  optimize via **perf-host-optimization**. Touch points (prior art, TRT-LLM
  repo): `tensorrt_llm/_torch/speculative/mtp.py`
  (`prepare_position_ids_and_last_tokens` hoisted from a `forward()` closure to a
  method; `update_kv_lens` decorator dropped and inlined as
  `kv_lens_cuda[:batch_size] += 1`), and
  `tensorrt_llm/_torch/attention/backends/sparse/dsa/metadata.py` (`_get_dense_topk_indices`
  hoisted out of `prepare_dense_topk_indices` into a `@maybe_compile` method).
- **Expected effect:** fewer `torch.compile` recompilations and lower Dynamo/host
  time per call → smaller GPU-idle gaps and higher throughput when host-bound;
  largest where the method runs every iteration (decode / spec-dec). Record the
  local Δ (recompile count, per-iteration host time, throughput) — PR #12581
  ships it without a standalone number; do **not** cite one not measured locally.
- **Accuracy risk:** lossless — same computation; only *where/when* it is compiled
  changes (`torch.compile` is output-equivalent within fp tolerance, and the
  dropped-compile op runs identical eager math).
- **Verify:** correctness — spec-dec / MTP and DSA tests still pass; outputs match
  pre-change within tolerance. Perf — `TORCH_LOGS=recompiles` shows the function
  compiled once, not per-call; per-iteration host time and throughput before vs
  after.
- **Rollback:** re-inline the closure / re-add the decorator; trigger: any
  correctness regression or >5% host-time / throughput regression.
- **Prior art:** PR #12581; `mtp.py` + `dsa.py`
  paths above. Owning skill: **perf-host-optimization**; detection:
  **perf-host-analysis**; `torch.compile` background:
  **perf-torch-cuda-graphs** (it covers `torch.compile` as a CUDA-graph API).
  Related: the [loop-invariant-hoist case](cache-step-invariant-per-layer.md) (same PR).
