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

# Advanced Profiling Guide

Replay modes, MPI/multi-process profiling, JIT kernel profiling, CUDA graphs, PM sampling, customization, and occupancy calculator.

## Replay Modes

ncu collects metrics by replaying kernels multiple times (one pass per metric group). The replay mode determines how this works.

### Kernel Replay (Default)

```bash
ncu --replay-mode kernel ...
```

Saves/restores kernel-written memory between passes. Best for most single-process profiling.

### Application Replay

```bash
ncu --replay-mode application ...
```

Reruns the entire application for each metric collection pass. Use when kernel replay causes issues (e.g., kernels with side effects).

Options:
- `--app-replay-buffer file|memory` — data buffering strategy
- `--app-replay-match name|grid|all` — kernel matching between runs
- `--app-replay-mode strict|balanced|relaxed` — matching strictness

### Range Replay

```bash
ncu --replay-mode range --profile-from-start off ...
```

Replays CUDA API call ranges between `cudaProfilerStart()` and `cudaProfilerStop()`. Good for profiling specific regions of complex applications.

### Application Range Replay

```bash
ncu --replay-mode app-range --profile-from-start off ...
```

Reruns the application to collect metrics for specified ranges. Combines application replay with range markers.

### Choosing a Replay Mode

| Scenario | Recommended Mode |
|----------|------------------|
| Standard kernel profiling | `kernel` (default) |
| Kernels with global side effects | `application` |
| Specific code regions | `range` with profiler markers |
| Framework/JIT kernels | `range` or `kernel` with `--profile-from-start off` |

## Profiling JIT-Compiled Kernels (Triton/cuTile/CuTeDSL)

JIT-compiled kernels trigger autotuning on first invocation. Isolate actual execution:

### Method 1: Profiler Markers

```python
# Warmup (includes JIT compilation + autotuning)
for _ in range(5):
    result = kernel(inputs)
torch.cuda.synchronize()

# Profile only the steady-state execution
torch.cuda.cudart().cudaProfilerStart()
for _ in range(3):
    result = kernel(inputs)
    torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
```

```bash
ncu --profile-from-start off \
    --kernel-name regex:"target_kernel" \
    --launch-count 3 \
    -- python script.py
```

### Method 2: Launch Skip

```bash
ncu --launch-skip 10 --launch-count 3 \
    --kernel-name regex:"target_kernel" \
    -- python script.py
```

Skip enough launches to pass autotuning. The exact skip count depends on the framework.

### Method 3: NVTX Ranges

```python
import torch
torch.cuda.nvtx.range_push("profile_region")
result = kernel(inputs)
torch.cuda.nvtx.range_pop()
```

```bash
ncu --nvtx --nvtx-include "profile_region/" \
    -- python script.py
```

## CUDA Graph Profiling

### Per-Node Profiling (Default)

```bash
ncu --graph-profiling node ...
```

Profiles each kernel node in the graph individually.

### Whole-Graph Profiling

```bash
ncu --graph-profiling graph ...
```

Profiles the entire graph as a single workload. Useful for understanding graph-level behavior.

## Multi-Process and MPI Profiling

### Profile All Processes

```bash
ncu --target-processes all -o report mpirun -np 4 app
```

### Per-Rank Reports

```bash
mpirun -np 4 ncu -o report_%q{OMPI_COMM_WORLD_RANK} app
```

### Synchronized Profiling (NCCL/NVSHMEM)

For dependent kernels across ranks that must be profiled together:

```bash
mpirun -np 4 ncu --communicator=tcp --communicator-num-peers=4 \
    --lockstep-kernel-launch -o report app
```

Restrict synchronization to specific NVTX ranges:

```bash
mpirun -np 4 ncu --communicator=tcp --communicator-num-peers=4 \
    --lockstep-nvtx-include "nccl/" -o report app
```

## PM Sampling

Lower-overhead alternative to full section profiling. Periodically samples metrics instead of replaying.

```bash
ncu --section PmSampling --pm-sampling-interval 0 ...
```

- `--pm-sampling-interval 0` — auto interval
- `--pm-sampling-buffer-size 0` — auto buffer
- `--pm-sampling-max-passes 0` — auto passes

### Warp State Sampling

```bash
ncu --warp-sampling-interval auto --warp-sampling-max-passes 5 ...
```

Captures periodic warp state snapshots. Lower overhead than `WarpStateStats` section.

## Profile Series

Automated profiling with varying parameters to find optimal configurations:

```bash
ncu --section SpeedOfLight --kernel-name regex:"kernel" \
    --launch-count 10 -- python sweep.py
```

Useful for comparing different block sizes, shared memory configs, or algorithm variants.

## Customization

### Custom Section Files

Section files (`.section` format, Protocol Buffer) define what metrics to collect and how to display them. Located in the `sections/` folder of the installation.

```bash
ncu --section-folder /path/to/custom/sections ...
ncu --list-sections ...   # Verify custom sections are discovered
```

### Derived Metrics

Compose new metrics from existing ones using math expressions (addition, subtraction, multiplication, division). Defined in section files, computed at collection time.

### Python Rules

Rules implement automated analysis logic:

```python
def get_identifier():
    return "my_custom_rule"

def get_name():
    return "My Custom Analysis"

def apply(handle):
    # Access metrics and add recommendations
    pass
```

```bash
ncu --list-rules ...   # Verify custom rules are discovered
```

## Occupancy Calculator (Python)

The `ncu_occupancy` module (in `extras/python/`) calculates theoretical occupancy for different kernel configurations.

```python
import ncu_occupancy as occ

calc = occ.OccupancyCalculator(major=8, minor=0)  # SM 8.0 (A100)

params = occ.OccupancyParameters(
    threads_per_block=256,
    registers_per_thread=32,
    shared_mem_per_block=2048
)

occupancy = calc.get_sm_occupancy()
limiters = calc.get_occupancy_limiters()
# Returns: [OccupancyLimiter.REGISTERS, ...]

optimal = calc.get_optimal_occupancy()  # Finds best config
```

### OccupancyVariable Enum

Variables that can be swept for optimization:
- `THREADS_PER_BLOCK`
- `REGISTERS_PER_THREAD`
- `SHARED_MEMORY_PER_BLOCK`
- `BLOCK_BARRIERS`

### GPU Data

```python
gpu_data = occ.get_gpu_data(major=8, minor=0)
# Returns: SM count, register limits, warp sizes, memory constraints
```

## Reproducibility

### Clock Control

```bash
ncu --clock-control base ...    # Lock to base frequency (default)
ncu --clock-control boost ...   # Lock to boost frequency
```

Fixed-frequency profiling produces more reproducible results.

### Cache Control

```bash
ncu --cache-control all ...     # Flush L1/L2 between replays (default)
ncu --cache-control none ...    # No flushing (shows cache-warm behavior)
```
