<!--
SPDX-FileCopyrightText: Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Benchmarking Patterns

Detailed patterns and guidance for GPU benchmarking beyond the quick-reference in SKILL.md.

## Warmup Considerations

| Factor | Warmup Iterations |
|--------|-------------------|
| JIT compilation (torch.compile, XLA) | 10-20 iterations |
| GPU clock stabilization | 5-10 iterations |
| CUDA context initialization | 1-2 iterations |
| Autotuning (cuBLAS, cuDNN) | 1-5 iterations |
| **Recommended default** | **50 iterations** |

## GPU Hardware Properties

Query hardware specs for bandwidth/throughput analysis via `torch.cuda.get_device_properties()`:

```python
props = torch.cuda.get_device_properties(0)
props.name               # "NVIDIA H100 80GB HBM3"
props.total_memory        # bytes (e.g. 85899345920)
props.memory_clock_rate   # kHz (e.g. 2619000)
props.memory_bus_width    # bits (e.g. 5120)
```

Compute theoretical peak memory bandwidth:

```python
peak_bw_gbs = 2 * (props.memory_clock_rate * 1e3) * (props.memory_bus_width / 8) / 1e9
```

**Common wrong attribute names** (these do NOT exist on the properties object):
`total_mem`, `mem_clock_rate`, `bus_width`. Always use the names above.

## CuTe DSL Pre-Compilation

CuTe DSL kernels use JIT compilation. **You must pre-compile with `cute.compile()` before timing**, otherwise the kernel recompiles on every call and timing is inaccurate.

```python
# Pre-compile the @cute.jit host function
compiled = cute.compile(host_fn, ...)

# Now safe to benchmark
x = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
out = torch.empty_like(x)

for _ in range(warmup):
    compiled(x, out)
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(iters):
    compiled(x, out)
end.record()
torch.cuda.synchronize()
mean_ms = start.elapsed_time(end) / iters
```

## CUDA Graph Timing

When benchmarking CUDA Graph replay, follow this sequence:

1. **Eager warmup** — Run the function normally to stabilize GPU clocks and trigger JIT
2. **Graph capture** — Capture the function into a CUDA Graph
3. **Replay warmup** — Replay the graph 10+ times to stabilize graph execution
4. **Measure replay** — Time graph replay with CUDA events

The eager warmup and replay warmup are both necessary. Graph replay has different execution characteristics than eager mode.

```python
import torch

def benchmark_cuda_graph(fn, warmup=50, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()

    # Warmup graph replay
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters
```

## Triton Kernel Timing

Use Triton's built-in benchmark utility:

```python
import triton

result_ms = triton.testing.do_bench(
    lambda: kernel[grid](args),
    warmup=50,
    rep=100,
)
```

## Raw CUDA (C++) Timing

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

for (int i = 0; i < warmup; i++)
    kernel<<<grid, block>>>(args);
cudaDeviceSynchronize();

cudaEventRecord(start);
for (int i = 0; i < iters; i++)
    kernel<<<grid, block>>>(args);
cudaEventRecord(stop);
cudaDeviceSynchronize();

float ms;
cudaEventElapsedTime(&ms, start, stop);
printf("Mean: %.3f ms\n", ms / iters);
```

## When to Use What

| Scenario | Recommended Approach |
|----------|---------------------|
| Quick comparison | Simple CUDA events (start/end pair) |
| Detailed analysis | Per-iteration events with statistics |
| CUDA Graph timing | Capture, warmup replay, then measure replay |
| CuTe DSL kernels | `cute.compile()` once, then CUDA events |
| Triton kernels | Use `triton.testing.do_bench()` |
| Training loop | Manual timing (see SKILL.md Loop Workloads section) |
| Cross-framework comparison | Use nsys for consistent measurement |

## Reporting Format

Always report with units and context:

```
Kernel: matmul_fp16 (M=4096, N=4096, K=4096)
Device: NVIDIA H100
Latency: 1.234 +/- 0.012 ms (mean +/- std, n=100)
Throughput: 123.4 TFLOP/s
```

Include:
- Kernel/operation name and input dimensions
- Device info
- Mean +/- std (or median + min/max)
- Sample count
- Throughput if applicable
