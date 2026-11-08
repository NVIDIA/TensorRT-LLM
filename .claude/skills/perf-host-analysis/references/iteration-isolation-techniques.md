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

# Iteration Isolation Techniques for nsys Trace Analysis

## Problem

nsys traces of LLM inference benchmarks contain multiple phases:
1. Model loading and initialization
2. JIT compilation (e.g., flashinfer, torch.compile)
3. Warmup iterations
4. Ramp-up phase (increasing batch size as requests arrive)
5. Steady-state generation (peak batch, generation-only)
6. Drain phase (decreasing batch as requests complete)
7. Teardown

Analyzing the full trace or using naive windowing (e.g., kernel density) leads to
false conclusions because non-inference phases dominate metrics.

## Technique 1: Allreduce Kernel Pattern (Recommended for TP)

### When to Use
- Tensor Parallel (TP) configurations with TP >= 2
- Each forward step has a fixed number of allreduce operations

### How It Works
Each transformer layer typically executes 2-4 allreduce operations per forward step
(column-parallel + row-parallel linear layers). The total is deterministic:

| Model | Allreduce per Step |
|-------|-------------------|
| Llama 3.2 1B (TP=2) | 66 |
| Llama 3.1 8B (TP=2) | 128 |

### Algorithm

```python
# 1. Find all allreduce kernels
allreduce_events = query("SELECT start, end FROM kernels WHERE name LIKE '%allreduce%'")

# 2. Group by gap threshold (< 1ms = same iteration)
iterations = group_by_gap(allreduce_events, threshold=1_000_000)  # 1ms in ns

# 3. Find common iteration size
common_size = mode([len(iter) for iter in iterations])

# 4. Detect phase boundaries (> 100ms gap)
phases = split_by_gap(iterations, threshold=100_000_000)  # 100ms

# 5. Benchmark phase = last phase with common iteration size
bench_phase = last_phase_with_common_size(phases, common_size)
```

### Advantages
- Deterministic: allreduce count per step is fixed by model architecture
- Robust: allreduce kernels are always present in TP configurations
- No false positives from JIT phases (JIT doesn't launch allreduce)

### Limitations
- Requires TP >= 2 (no allreduce in single-GPU)
- Assumes allreduce pattern is stable across versions

## Technique 2: NVTX Forward Step Ranges

### When to Use
- TRT-LLM traces with NVTX instrumentation enabled
- Any TP configuration including single-GPU

### How It Works
TRT-LLM instruments each forward step with NVTX:
```
[Executor] _forward_step N: X ctx reqs, Y gen reqs
```

This encodes:
- Step number (N)
- Context request count (X) — prefill requests
- Generation request count (Y) — decode requests

### Algorithm

```python
# 1. Get all [Executor] _forward_step NVTX ranges
steps = query("""
    SELECT n.start, n.end, s.value
    FROM NVTX_EVENTS n
    JOIN StringIds s ON n.textId = s.id
    WHERE s.value LIKE '%[Executor] _forward_step%'
""")

# 2. Parse ctx/gen counts from text
for step in steps:
    step.ctx_reqs = parse_ctx(step.text)
    step.gen_reqs = parse_gen(step.text)

# 3. De-duplicate TP ranks (group entries within 100us)
unique_steps = dedup_by_proximity(steps, threshold=100_000)

# 4. Filter to steady state
max_gen = max(s.gen_reqs for s in unique_steps)
steady = [s for s in unique_steps if s.ctx_reqs == 0 and s.gen_reqs == max_gen]
```

### Advantages
- Works for any TP configuration including single-GPU
- Directly encodes workload characteristics (ctx vs gen)
- Supports filtering to specific workload phases

### Limitations
- Requires NVTX instrumentation (enabled by default in TRT-LLM)
- TP de-duplication needed (each rank reports independently)
- NVTX text is stored via textId, not direct text field

## Technique 3: Kernel Density Windowing (NOT Recommended)

### Why It Fails

This approach selects the time window with the highest kernel density as the
"benchmark region." It fails because:

1. **JIT compilation phases have high kernel density.** Flashinfer JIT compiles
   hundreds of elementwise/reduce kernels in rapid succession, creating a denser
   kernel pattern than actual inference.

2. **The selected window may contain zero model kernels.** In our case study,
   the density-based approach selected a window (t=6.61-7.60s) that contained
   ONLY elementwise/reduce kernels from JIT compilation and ZERO GEMM/attention
   kernels.

3. **No workload validation.** The approach has no way to verify that the selected
   window actually contains model forward passes.

### When It Might Work
- Simple single-kernel benchmarks without JIT
- Traces where inference is the only compute-intensive phase
- As a coarse first approximation, validated by kernel type checking

## Combining Techniques

The most robust approach combines allreduce iteration detection (structural) with
NVTX step filtering (semantic):

1. Use allreduce grouping to find the benchmark iteration window
2. Use NVTX step ranges within that window for per-step metrics
3. Use NVTX ctx/gen counts to filter to steady-state generation
4. Cross-validate: allreduce iteration count should match NVTX step count

## Phase Identification Summary

| Phase | Characteristics | How to Identify |
|-------|----------------|-----------------|
| Model loading | No GPU kernels | First kernels in trace |
| JIT compilation | Many small elementwise/reduce kernels, no allreduce | High kernel density, no model kernels |
| Warmup | Few iterations, small batch | Low step numbers, few ctx/gen reqs |
| Ramp-up | Increasing gen reqs per step | ctx > 0 in NVTX |
| Steady-state | Fixed max gen reqs, 0 ctx | ctx = 0, gen = max in NVTX |
| Drain | Decreasing gen reqs per step | gen < max, ctx = 0 |
| Teardown | No forward steps | After last NVTX step |
