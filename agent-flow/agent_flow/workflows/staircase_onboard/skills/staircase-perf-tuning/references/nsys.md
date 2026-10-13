# Profiling a target through the substrate

## Capture

`bench.py --nsys <prefix>` (or `<perf workdir>/scripts/nsys_capture.sh`) inside a spool job. Two modes:

- **session (default)**: `nsys launch -t cuda,nvtx --cuda-graph-trace node trtllm-serve ...`, then
  `nsys start -o <prefix>` `--nsys-delay` seconds into the first measured point and `nsys stop`
  `--nsys-duration` seconds later. Nothing inside the server calls the profiler API.
- **api**: TensorRT-LLM's documented recipe, `TLLM_PROFILE_START_STOP=<A>-<B> nsys profile -t cuda,nvtx
  -c cudaProfilerApi --cuda-graph-trace node --capture-range-end=stop --gpu-metrics-devices=none`. On
  this target's single-process server the in-process stop never returned (2026-10-13, nsys 2026.5.1,
  GB300) -- first because `cudaProfilerStop` held the GIL (fixed in `pyexecutor/profiling.py`), then
  inside CUPTI. Use it where it works; do not add `osrt`.

`--cuda-graph-trace node` shows every kernel inside a replayed graph; `graph` shows one range per replay.
Boot under CUPTI is several times slower; the harness allows 90 minutes. A profiled run's throughput is
perturbed: use a throwaway label and never quote it.

Then summarize per step (standard library only):

```bash
python3 <PKG>/tune/nsys_summary.py <prefix>.sqlite --top 15 --json <prefix>.summary.json
```

It attributes kernels to the executor step whose host range launched them and reports, per step class
(decode / mixed / prefill): GPU busy time, span, idle in span, kernels per step, time per kernel family,
the top kernels, and the executor's host ranges.

### Window arithmetic

Iterations count from the engine's first step after warmup requests start arriving. With the
harness defaults (16 warmup requests of ISL tokens and 64 output tokens at concurrency 16), the
warmup takes roughly `16 * ceil(ISL / max_num_tokens) + 64` iterations. Then:

- **con 1, decode**: request 1 spends `ceil(ISL / max_num_tokens)` iterations in prefill and OSL-1 in
  decode. A window of 40 iterations a few hundred after warmup is pure single-sequence decode.
- **con N, steady decode**: the first wave's prefills take about `N * ceil(ISL/max_num_tokens)`
  iterations (each mixed with the decodes already admitted); a window after that and before
  OSL is pure batch-N decode.
- **mixed / prefill**: a window inside the first wave's admission shows prefill chunks with
  decode rows.

Confirm the window from the NVTX ranges (each executor step is one range carrying its batch
composition) rather than trusting the arithmetic.

## GUI-free extraction

```bash
nsys stats --report cuda_gpu_kern_sum --format csv -o <prefix> <prefix>.nsys-rep   # per kernel name
nsys stats --report cuda_api_sum      --format csv -o <prefix> <prefix>.nsys-rep   # launches, graph replays
nsys stats --report nvtx_sum          --format csv -o <prefix> <prefix>.nsys-rep   # executor steps
nsys export --type sqlite -o <prefix>.sqlite <prefix>.nsys-rep                     # for custom queries
```

Per decode step: divide each kernel family's total time by the number of steps in the window
(count `cudaGraphLaunch` or the step NVTX ranges). Report per step:

- kernel count and time per family (GEMM by shape, norm, elementwise, attention, recurrence, MoE,
  routing, copy);
- **exclusive** GPU time (union of intervals) vs the step's wall time: the difference is GPU idle,
  i.e. host- or launch-bound time;
- the count of kernels < 3 us -- at small batch the step is often a launch-latency chain even
  inside a graph.

Classify: GPU idle > 15% of the step -> host/launch bound (fewer kernels, graph coverage,
host knobs); a single family > 30% -> that family is the lever; many tiny kernels -> fusion.
