# Profiling a target through the substrate

## Capture

TensorRT-LLM's own recipe, as `bench.py --nsys <prefix>` runs it:

```bash
TLLM_PROFILE_START_STOP=<A>-<B> nsys profile -o <prefix> -f true -t cuda,nvtx \
  -c cudaProfilerApi --cuda-graph-trace node --capture-range-end=stop \
  --gpu-metrics-devices=none trtllm-serve <model> --host ... --port ... --config <yaml>
```

- `-c cudaProfilerApi` + `TLLM_PROFILE_START_STOP` collect only iterations A..B of the executor.
- `--capture-range-end=stop` keeps the server serving after the window. nsys's default
  (`stop-shutdown`) tore the server down mid-sweep and, with `osrt` tracing added, hung the capture
  with the report never written (observed 2026-10-12, nsys 2026.5.1, GB300). Do not add `osrt`.
- `--cuda-graph-trace node` shows every kernel inside a replayed graph; `graph` shows one range per
  replay (cheaper, no per-kernel attribution).
- Boot under CUPTI is several times slower; the harness allows 90 minutes.
- A profiled run's throughput is perturbed. Use a throwaway label; never quote it.

Use `<perf workdir>/scripts/nsys_capture.sh` (or the same flags) inside a whole-node spool job.

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
