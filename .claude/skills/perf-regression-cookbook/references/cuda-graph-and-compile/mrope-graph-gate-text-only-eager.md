---
id: case-mrope-graph-gate-text-only-eager
type: regression-case
family: execution-and-graph
module: cuda-graph-and-compile
maturity: full
regression_class: [cuda-graph-regression, fast-path-fallback]
signals: [throughput-drop, itl-increase, host-time-increase, gpu-idle-between-steps]
subsystems: [cuda-graph]
introduced_via: [incomplete-coverage]
phase: [decode]
patterns: [pattern-fast-path-silent-fallback]
nvbugs: ["6346545", "6346546"]
commits: ["a8f0efc57fbf"]
success_prs: [15589]
failed_prs: []
---

# mRoPE delta-cache seeding gate keeps text-only decode permanently eager

> Part of the [CUDA graph & compile regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbugs `6346546` / `6346545` · commit `a8f0efc57fbf` ·
  PR #15589 — "[fix] fix mRoPE CUDA graph gate for text requests".
- **Symptom:** Qwen3.5 **pure-text** decode lost throughput and gained
  `gpu_time`, because every decode step ran eager instead of replaying a CUDA
  graph. The PR carries no numbers (only a CodeRabbit summary), so the
  magnitude is not public; the mechanism (graph replay lost on every decode
  step) implies a large, every-step decode regression on text-only traffic.
- **Root cause:** `CUDAGraphRunner.maybe_get_cuda_graph` refused a graph
  whenever `self.config.use_mrope` and any request in
  `batch.generation_requests` had `py_mrope_delta_cache_slot != py_seq_slot`
  — the intent being one eager step per seq slot to seed the model-side mRoPE
  delta cache before replay. Text-only requests carry **no** mRoPE position
  delta, so that slot is never seeded, the predicate stays true forever, and
  every decode step of a text-only workload on a `use_mrope` model (per the
  fix's own code comment: "Qwen3.5 configs normalized to text-only
  decoding") ran eager and never replayed a CUDA graph.
- **How introduced:** PR #11943 `[TRTLLM-12427][perf] Qwen2.5/3/3.5-VL
  Performance Optimization` (merge commit `1283c6b31976`). The same PR that
  replaced the per-request `MultimodalParams` mRoPE tensors in the graph's
  shared static tensors with a device-side `mrope_delta_read_seq_slots` cache
  added this seeding gate to `maybe_get_cuda_graph` —
  `git log -S py_mrope_delta_cache_slot -- tensorrt_llm/_torch/pyexecutor/cuda_graph_runner.py`
  returns only that commit. Culprit and fix share an author.
- **Fix mechanism:** narrows the gate to requests that actually have a delta.
  Two new static helpers in
  `tensorrt_llm/_torch/pyexecutor/cuda_graph_runner.py`:
  `_get_mrope_position_delta(request)` (reads `py_mrope_position_delta`, else
  `py_multimodal_data["mrope_config"]["mrope_position_deltas"]`) and
  `_needs_mrope_delta_cache_update(request)`, which returns False for a dummy
  request or `py_seq_slot is None`, False once
  `py_mrope_delta_cache_slot == py_seq_slot`, and — the fix — False when the
  request carries no delta at all. `maybe_get_cuda_graph` calls that helper,
  so text-only requests are graph-eligible again.
- **Detection signal:** a large (>2x), *every-step* decode regression on a
  multimodal-capable model driven with text-only prompts, with nsys showing
  per-op eager launches and host-bound inter-step gaps where a graph replay
  used to be. Audit the eligibility predicate:
  `grep -n -A8 "config.use_mrope and any" tensorrt_llm/_torch/pyexecutor/cuda_graph_runner.py`
  — a pre-fix build compares `py_mrope_delta_cache_slot != py_seq_slot` with
  no check that the request produces a delta; post-fix it calls
  `_needs_mrope_delta_cache_update`. Cheap A/B that needs no profiler: rerun
  the same text-only case with CUDA graphs disabled — numbers identical to
  the enabled run mean no graph was ever being replayed.
- **Prevention/guard:** gap. PR #15589 touches only `cuda_graph_runner.py`
  (+31/−6) and adds no test; its single approving review carries an empty
  body, so nothing pins that a text-only request on a `use_mrope` model is
  graph-eligible. General guard: a predicate that forces the slow path
  "until state X is seeded" must also assert that the input can ever produce
  X — and a graph gate that stays closed for a whole run should log once
  rather than fail silent, which is what kept this drop invisible to the
  graph-enabled config.
- **Generalizes to:** `pattern-fast-path-silent-fallback` — a "run one eager
  step to seed a cache" guard becomes permanent for the input class that
  never populates that cache, so the fast path never re-arms. Carries to:
  seeding/warmup gates keyed on cache-slot equality that some request class
  can never satisfy; multimodal-capable configs serving text-only traffic
  (any per-request multimodal state consulted on a graph-eligibility path);
  CUDA-graph gates keyed on a static config flag (`use_mrope`) instead of the
  per-batch presence of the data that flag implies; lazy-init "only the first
  call is slow" paths where the first call never happens.
