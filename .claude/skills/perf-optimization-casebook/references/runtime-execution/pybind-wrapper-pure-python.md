---
id: case-pybind-wrapper-pure-python
type: case
family: runtime-execution
maturity: full
bottleneck: [host-overhead, launch]
signals: [gpu-idle-between-steps, host-prep-on-critical-path, high-concurrency]
architectures: [any-sm]
model_scope: [model-agnostic]
phase: [any-phase]
patterns: [pattern-push-language-boundary-off-hot-path]
accuracy_risk: lossless
apply_via_kind: [code-change]
knobs: []
specialists: [perf-host-optimization, perf-host-analysis]
commits: []
measured: []
---

# Re-implement a hot-path binding wrapper in pure Python to cut pybind overhead

> Part of the [Runtime / Execution casebook](index.md) · schema: [case-template](../case-template.md)

- **Applies when:** host-overhead-bound serving — the PyExecutor loop is
  CPU-bound, the GPU shows idle gaps between steps / low utilization, and host
  time sits in per-iteration response/request handling rather than kernels.
  Confirm with `perf-host-analysis` (GPU-idle ratio, host-prep-exposed ratio,
  per-iteration host breakdown) **before** doing this — it is an invasive
  refactor, not a knob. The signal it targets: a Python object on the
  per-response/per-request hot path is a thin wrapper over a C++ binding type
  (`bindings.executor.Response` / `Result`) whose attributes are read many times
  per iteration via `__getattr__`, and/or is pickled to cross the worker→client
  process boundary. Most acute at high concurrency / high request rate, where
  response handling runs every iteration.
- **Mechanism:** every attribute read on a pybind-wrapped object is a Python↔C++
  crossing with fixed overhead, so a wrapper that delegates via `__getattr__`
  pays it per field per iteration, and an object pickled for IPC is marshalled
  twice. Making the hot object pure-Python (a `@dataclass`) turns those reads
  into plain attribute lookups; mirroring the few C++-derived values it needs
  onto `py_*` attributes once removes the remaining crossings; and serializing
  the C++ `Result` in a single native call (returning bytes + the `is_final`
  flag) with lazy deserialize on the consumer replaces N per-attribute crossings
  plus a second pickle with one bulk serialize.
- **Generalizes to:** the pattern "**push language-boundary crossings off the
  host hot path** — pure-Python hot objects, `py_*`-mirrored C++ fields read
  once, and bulk-serialize-once + lazy-deserialize for anything that must cross a
  process boundary." Carries to: other executor binding-wrappers
  (request / sampling-params / result proxies — the `py_*` mirror convention is
  already the idiom, extend it); any tight Python loop reading attributes off a
  C-extension proxy (nanobind / Cython / ctypes — hoist the reads to locals, cf.
  the `begin_compute` local replacing the `request.context_current_position`
  property read in the same PR); and IPC paths that pickle a native object
  (serialize once in native code, defer deserialize to the consumer, lift only
  the flag the dispatcher branches on). Adapt by: profiling the host loop first;
  confirming the object is genuinely hot (many per-iteration crossings) before
  refactoring; and keeping the lazy-deserialize seam correct — the consumer must
  deserialize before touching real fields, and any control-flow flag it needs
  pre-deserialize (here `is_final`) must be hoisted out.
- **Apply via:** **not a server config knob** — a host-side code change in the
  executor/binding layer. Delegate optimization to **perf-host-optimization**
  (line_profiler / nsys host rounds) and detection to **perf-host-analysis**.
  Touch points (prior art, TRT-LLM repo): the Python wrapper
  `tensorrt_llm/_torch/pyexecutor/llm_request.py` (`LlmResponse` / `LlmResult`,
  `py_*` fields), the loop `tensorrt_llm/_torch/pyexecutor/py_executor.py`
  (`_handle_responses`), the lazy-deserialize seam
  `tensorrt_llm/executor/result.py` (`_handle_response`), and the C++
  bulk-serialize entry points `createSerializedResult` / `deserialize_result`
  (`cpp/.../batch_manager/llmRequest.cpp` + pybind bindings).
- **Expected effect:** lower host/CPU time per executor iteration in
  response/request handling → higher throughput and smaller GPU-idle gaps when
  the loop is host-bound; no change when it is GPU-bound. Largest at high
  concurrency / high request rate. Measured Δ to be recorded from a
  host-profiling run (per-iteration host time, GPU-idle ratio, throughput before
  vs after) — PR #5224 is "a step of #3034" and reports no standalone speedup;
  do **not** cite a number not measured locally.
- **Accuracy risk:** lossless — purely changes how data is marshalled across the
  binding/IPC boundary; outputs (tokens, and logits/logprobs via `PyResult`) are
  unchanged.
- **Verify:** correctness — return-logits / logprobs and streaming tests still
  pass (e.g. `tests/unittest/_torch/test_return_logits.py`); confirm final-token
  semantics intact (`is_final` still drives `request_done` in
  `_handle_responses`) and that every consumer deserializes before reading result
  fields. Perf — `perf-host-analysis` / `perf-host-optimization`: per-iteration
  host time in response handling, GPU-idle ratio, and throughput at high
  concurrency, before vs after.
- **Rollback:** revert to the binding-wrapper `LlmResponse` / `LlmResult`;
  trigger: any correctness regression (logits / logprobs / streaming) or >5%
  host-time / throughput regression. Likely failure mode to watch: a consumer
  reading result fields without first calling `deserialize()` on the lazy seam.
- **Prior art:** PR #5224 (a step of #3034), "Re-implement LlmResponse in Python
  to reduce host overhead of pybind"; TRT-LLM paths above. Owning skill:
  **perf-host-optimization**; detection: **perf-host-analysis**. Related
  runtime/execution levers: [overlap scheduler](overlap-scheduler.md), CUDA graphs ([piecewise](piecewise-cuda-graph.md), [padding](cuda-graph-padding.md)).
