# Regression Cookbook — CUDA graph & compile

This module is the graph-capture and `torch.compile` machinery: the CUDA-graph
runner and its batch-size candidate list, the piecewise-CUDA-graph capture set
and the padding rule that maps a runtime shape onto a captured one, and the
compile gates that decide which regions are graphed at all
(`tensorrt_llm/_torch/compilation/`, `cuda_graph_runner.py`). Nothing here
computes a result — it decides *how* the same work is driven — so a defect is
almost never a slower kernel. It is a shape that no longer has a graph and
quietly runs eager, a shape that gets padded up to a far larger captured one,
or a compile wrapper whose host-side cost exceeds what graphing saves. First
thing to check: for the exact shapes the workload actually runs, was a graph
replayed, and which captured bucket did it land in.

## Recurring patterns in this module

- **Fast path silently fell back** — a capture set, an eligibility gate or a
  seeding predicate quietly routes a shape to eager. Nothing errors and the
  config is unchanged, so confirm replay per shape instead of trusting that
  graphs are "on". _(Instances: piecewise capture set missing reachable
  `num_tokens`; the mRoPE delta-cache seeding gate that keeps text-only decode
  permanently eager.)_
- **Padding bucket overshoot** — pad-to-nearest means a bucket far above its
  neighbour rounds the whole gap below it up to worst-case work, so **adding a
  bucket is not monotonically safe**. Check the distance between adjacent
  candidates, not just that the shape is covered.
  _(Instance: the force-appended piecewise capture ceiling.)_
- **Default change regressed a tuned workload** — flipping a default re-tunes
  every workload that was tuned against the old one; a config that named no
  batch-size list inherits the change silently.
  _(Instance: the changed CUDA-graph default batch sizes.)_
- **Host work on the hot path** — `torch.compile` on regions whose ops are too
  small to amortize guard/dispatch cost adds host time inside the piecewise
  region. Kernel names and durations are unchanged, so an nsys kernel diff finds
  nothing — compare the host spans.
  _(Instance: torch.compile on small MLA context ops.)_

## Cases

| Case | Symptom (signal) | Class |
|------|------------------|-------|
| [Changed default CUDA-graph batch-size list regressed a tuned GB200 workload](cudagraph-default-batch-sizes.md) | `gpt_oss_fp4_dep4_1k8k-con2560_iter5_1k8k` perf-sanity bar failure after the +64 batch-size default | cuda-graph-regression |
| [mRoPE delta-cache seeding gate keeps text-only decode permanently eager](mrope-graph-gate-text-only-eager.md) | Qwen3.5 pure-text decode runs eager every step; large throughput / `gpu_time` regression | cuda-graph-regression, fast-path-fallback |
| [torch.compile on small MLA context ops costs host time inside piecewise graphs](piecewise-attention-torch-compile-host-overhead.md) | extra host time and TTFT in the MLA context path under piecewise CUDA graph | host-work-added |
| [Piecewise CUDA-graph capture set misses reachable `num_tokens`](piecewise-cudagraph-capture-coverage.md) | ISLs 100/107/121/127 pad to 128 with no graph and run eager | cuda-graph-regression, fast-path-fallback |
| [Force-appended piecewise-graph capture ceiling makes padding run the 65536-token graph](piecewise-cudagraph-far-ceiling-append.md) | gpt-oss-120b GB300 serving ~12% QPS loss, TTFT more than doubled; 51.40 → 44.3–44.7 QPS | cuda-graph-regression |
