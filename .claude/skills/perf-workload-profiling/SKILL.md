---
name: perf-workload-profiling
description: >
  Code instrumentation for timing workloads. Two scenarios:
  (1) Training loop — inject manual timing to report per-iteration latency,
  throughput (samples/sec), and data load time. (2) Standalone kernel/op —
  write CUDA event timing code with warmup, per-iteration statistics, and
  anti-pattern avoidance. Also covers NVTX annotation for labeling profiler
  timelines.
  NOT for: running or analyzing profiler tools (nsys, ncu, Nsight Systems,
  Nsight Compute), writing kernels (Triton, CuTe, CUDA), applying
  optimizations (CUDA Graphs, gradient checkpointing, fusion), or
  interpreting roofline/SOL% metrics.
  Triggers: "measure throughput", "benchmark this function", "time my
  training loop", "samples per second", "NVTX annotate", "instrument my
  dataloader", "data load time", "kernel timing", "how do I time".
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Workload Profiling

## Quick Reference

Pick ONE path based on the workload type:

| Workload | Approach | Section |
|----------|----------|---------|
| Training loop | Manual `torch.cuda.synchronize()` + `time.perf_counter()` with warmup | Loop Workloads — Manual Timing |
| Single kernel or op | Write CUDA event benchmark (pre-allocate, warmup, event pairs) | Non-Loop Workloads — CUDA Event Benchmarking |
| Add timeline labels for nsys | Use `@nvtx.annotate` decorator or context manager | NVTX Reference |

## Principles

- **Measure, don't guess.** Every performance claim must trace back to profiler output or structured measurement data. Never invent metrics.
- **Isolate steady-state.** Warmup costs (CUDA context init, cuDNN autotuning, JIT compilation) distort measurements. Always exclude warmup iterations before collecting data.
- **Use hardware timing.** CUDA events measure GPU time precisely. CPU timers (`time.perf_counter()`) include host overhead and miss asynchronous execution.
- **No sync inside measurement loops.** Each `torch.cuda.synchronize()` adds 10-50us overhead. Record CUDA events asynchronously, sync once at the end.
- **Pre-allocate everything.** Tensors, events, compiled kernels — all before the timing loop. For CuTe DSL kernels, pre-compile with `cute.compile()`.
- **Minimize profiler interference.** Start with lightweight measurement (manual timing for latency/throughput) and escalate to heavier tools (Kineto, nsys, ncu) only when lighter tools cannot answer the question.

## Loop Workloads — Manual Timing

For training loops and iterative workloads, use manual `torch.cuda.synchronize()` + `time.perf_counter()` timing with warmup to measure per-iteration latency, throughput, and data load time.

### Injection Template

Read the user's training script, understand the dataloader and loop structure, then inject timing code.

```python
import time
import torch

WARMUP = 5
NUM_ITERS = 30
BATCH_SIZE = 128  # global batch size for throughput calculation

iter_times = []
data_times = []

for i, batch in enumerate(dataloader):
    if i >= WARMUP + NUM_ITERS:
        break

    t_data_end = time.perf_counter()

    torch.cuda.synchronize()
    t_start = time.perf_counter()

    # ... existing training loop body ...

    torch.cuda.synchronize()
    t_end = time.perf_counter()

    if i >= WARMUP:
        iter_ms = (t_end - t_start) * 1000
        iter_times.append(iter_ms)
        if i > 0:
            data_times.append((t_data_end - prev_iter_end) * 1000)
        print(f"[{i:04d}]: iter {iter_ms:.2f} ms, fps {BATCH_SIZE / (iter_ms / 1000):.2f}")

    prev_iter_end = t_end

import statistics
print(f"Average: iter {statistics.mean(iter_times):.2f} ms, "
      f"fps {BATCH_SIZE / (statistics.mean(iter_times) / 1000):.2f}")
```

### Interpreting Results

- **iter (ms)**: Wall-clock time per iteration (compute + communication, excluding data loading)
- **data (ms)**: Time spent in dataloader between iterations. If `data / iter > 0.2`, data loading is a bottleneck.
- **fps**: Global throughput in samples/second. Use with known FLOPs-per-sample to compute MFU.

### Limitations

Manual timing reports **aggregate** iteration timing — not per-sub-phase breakdown (forward, backward, optimizer). When the user asks **where time is spent within compute**:

1. Add `torch.cuda.synchronize()` + `time.perf_counter()` around each sub-phase for a one-off diagnosis, OR
2. Add NVTX annotations and run with `nsys profile` for timeline visualization.

## Non-Loop Workloads — CUDA Event Benchmarking

For single kernels, one-shot inference, or standalone operations, write CUDA event benchmarking code directly.

### PyTorch: Simple (Mean Only)

```python
import torch

def benchmark(fn, warmup=50, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters  # ms per iteration
```

### PyTorch: Detailed (Per-Iteration Stats)

```python
import torch
import statistics

def benchmark_detailed(fn, warmup=50, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()

    torch.cuda.synchronize()
    times = [starts[i].elapsed_time(ends[i]) for i in range(iters)]

    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "min_ms": min(times),
        "max_ms": max(times),
    }
```

### Anti-Patterns

| Anti-Pattern | Problem |
|--------------|---------|
| `torch.cuda.synchronize()` before AND after each iteration | Adds ~10-50us overhead per iteration |
| `time.perf_counter()` for GPU timing | Measures CPU time, misses async GPU execution |
| Missing warmup | First iterations include JIT, clock ramp-up, context init |
| Allocating tensors inside measurement loop | Allocation overhead pollutes timing |
| Reporting only mean | Hides variance, outliers, bimodal distributions |

For additional benchmarking templates (CUDA Graph, CuTe DSL, Triton, Raw CUDA), see [references/benchmarking-patterns.md](references/benchmarking-patterns.md).

## NVTX Reference

NVTX (NVIDIA Tools Extension) adds named annotations to profiler timelines. Use NVTX to label phases (forward, backward, optimizer) for readability in nsys — not for measurement.

```python
import nvtx

# Decorator — annotates every call
@nvtx.annotate("training_step", color="blue")
def training_step():
    ...

# Context manager — annotates a code block
with nvtx.annotate("data_loading", color="green"):
    batch = next(dataloader)
```

- **Do** annotate training phases (forward, backward, optimizer, data loading) for nsys timeline clarity.
- **Do not** annotate for measurement — use CUDA events or manual timing instead.
- **Do not** over-annotate — too many fine-grained ranges add visual clutter and minor overhead.

For NVTX domains, categories, payloads, and legacy API details, see [references/nvtx-api.md](references/nvtx-api.md).

## References

- [references/benchmarking-patterns.md](references/benchmarking-patterns.md) — CUDA Graph, CuTe DSL, Triton, Raw CUDA templates; warmup guidance; GPU hardware properties; reporting format
- [references/nvtx-api.md](references/nvtx-api.md) — Domains, categories, payloads, legacy push/pop API
- [references/pytorch-profiler-api.md](references/pytorch-profiler-api.md) — PyTorch 2.0+ profiler API changes (`device_time` vs deprecated `cuda_time`)
