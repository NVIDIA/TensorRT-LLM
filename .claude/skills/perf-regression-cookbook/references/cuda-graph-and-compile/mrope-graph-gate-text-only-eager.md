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

- **Provenance:** nvbug `6346546` · commit `a8f0efc57fbf` · PR #15589 —
  "[fix] fix mRoPE CUDA graph gate for text requests". Covers duplicate nvbug
  `6346545` (same culprit, smaller Qwen3.5 models — 6346545 is marked as a
  duplicate of 6346546, and all 7 of its cases were re-measured at the same
  bisect boundary, per the NVBug). **Do not fold in nvbug `6419078`**, which
  6346546 links as a related Qwen3.5 filing: it is a performance bug but a
  different defect — RTX 6000D, 1.3.0rc19 → rc20, 13–17% — and per its NVBug,
  6405760, 6418453, 6419078 and 6419139 are duplicates of one issue, fixed
  independently by PR #15632 (`[TRTLLM-12950][perf] DSv4 follow-up: DeepGEMM
  and MegaMoE`, merged 2026-07-03). Same model family, different root cause
  and different fix PR, so the folding rule excludes it.
- **Symptom:** Qwen3.5 **pure-text** bench cases lost throughput and gained
  `gpu_time` on GB200-OCI in the 1.3.0rc18 → rc19 QA release sweep.
  6346546: `qwen3.5_397b_a17b_fp4-bench-pytorch-float4-maxbs:512-maxnt:2048-input_output_len:1000,1000-con:512-ep:4-gpus:4`
  — `gpu_time` 77216.6 → 189899 (+145.93%), `total_token_throughput`
  13261.4 → 5392.33. 6346545: qwen3.5_9b bfloat16, Inference Time up
  26%–738% across 7 cases (e.g. `input_output_len:500,2000` `gpu_time`
  43718.3 → 94607.1, +116.40%; worst re-measured case
  `maxbs:1-input_output_len:1000,1000-reqs:10-con:1` 508 → 56.9 tok/s,
  −88.8%). Caught by the QA perf sweep, **not** perf CI — per the NVBug,
  there is no good way to verify it on CI, so the fix was merged and left
  for the QA perf sweep to confirm (the PR itself carries no review body).
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
  added this seeding gate to `maybe_get_cuda_graph`. Bisect on 6346546 pins
  the entire drop to that one commit (`82ca2c5e` last-good 13267.28 →
  `1283c6b3` 5398.96 tok/s, −59.3%) and exonerates the other two candidates,
  #15219 and #14398, which both measured within 1% of last-good. Culprit and
  fix share an author.
- **Fix mechanism:** narrows the gate to requests that actually have a delta.
  Two new static helpers in
  `tensorrt_llm/_torch/pyexecutor/cuda_graph_runner.py`:
  `_get_mrope_position_delta(request)` (reads `py_mrope_position_delta`, else
  `py_multimodal_data["mrope_config"]["mrope_position_deltas"]`) and
  `_needs_mrope_delta_cache_update(request)`, which returns False for a dummy
  request or `py_seq_slot is None`, False once
  `py_mrope_delta_cache_slot == py_seq_slot`, and — the fix — False when the
  request carries no delta at all. `maybe_get_cuda_graph` calls that helper,
  so text-only requests are graph-eligible again. Measured fix vs no-fix on
  the bug: total token throughput 8211.0 → 14195.8 tok/s (+72.9%), total
  latency 124711 → 72134 ms (−42.2%).
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
  body, and the request to add Qwen3.5 CI coverage lives in the NVBug, not
  the PR, so nothing pins that a text-only request on a `use_mrope` model is
  graph-eligible. General guard: a predicate that forces the slow path
  "until state X is seeded" must also assert that the input can ever produce
  X — and a graph gate that stays closed for a whole run should log once
  rather than fail silent, which is what made a 2.5x drop invisible until a
  release-to-release sweep.
- **Generalizes to:** `pattern-fast-path-silent-fallback` — a "run one eager
  step to seed a cache" guard becomes permanent for the input class that
  never populates that cache, so the fast path never re-arms. Carries to:
  seeding/warmup gates keyed on cache-slot equality that some request class
  can never satisfy; multimodal-capable configs serving text-only traffic
  (any per-request multimodal state consulted on a graph-eligibility path);
  CUDA-graph gates keyed on a static config flag (`use_mrope`) instead of the
  per-batch presence of the data that flag implies; lazy-init "only the first
  call is slow" paths where the first call never happens.
