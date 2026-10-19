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
ncu --replay-mode range ...                                     # cudaProfilerStart/Stop ranges
ncu --replay-mode range --nvtx --nvtx-include "region/" ...     # NVTX ranges
```

Replays CUDA API call ranges between `cudaProfilerStart()` and `cudaProfilerStop()`, or NVTX ranges selected with `--nvtx-include`. Good for profiling specific regions of complex applications. Range replay rejects `--profile-from-start off`.

### Application Range Replay

```bash
ncu --replay-mode app-range ...
```

Reruns the application to collect metrics for the same ranges. Combines application replay with range markers.

### Choosing a Replay Mode

| Scenario | Recommended Mode |
|----------|------------------|
| Standard kernel profiling | `kernel` (default) |
| Kernels with global side effects | `application` |
| Specific code regions | `range` with profiler markers |
| Framework/JIT kernels | `kernel` with `--profile-from-start off`, or `range` |

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

Skip enough launches to pass autotuning. The exact skip count depends on the framework:
Triton's autotuner launches every config many times, and `--launch-skip 10` still profiled
an autotuning launch in testing. Check the profiled grid and block size against the chosen
config, or use Method 1.

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
mpirun -np 4 ncu --communicator=tcp --communicator-tcp-num-peers=4 \
    --lockstep-kernel-launch -o report app
```

Restrict synchronization to specific NVTX ranges; `--lockstep-nvtx-include` is rejected
without `--lockstep-kernel-launch`:

```bash
mpirun -np 4 ncu --communicator=tcp --communicator-tcp-num-peers=4 --lockstep-kernel-launch \
    --lockstep-nvtx-include "nccl/" -o report app
```

With the TCP communicator, ncu appends the rank to the `-o` name (`report0`, `report1`, ...), so
`%q{OMPI_COMM_WORLD_RANK}` is not needed. In testing with Nsight Compute 2026.3.1 and Open MPI,
the reports were complete once written, but the `ncu` processes did not always exit cleanly
afterwards (some hung, some returned non-zero), and a leftover one held the communicator port
(default 49217) so the next run waited for peers forever. Check for the reports rather than the
exit code, kill leftover `ncu` processes, or pass a free `--communicator-tcp-port`.

When one `ncu` launches every rank on one node (`torchrun`, or `mpirun` under `ncu`),
`--communicator shmem` (Nsight Compute 2026.1+) synchronizes them without TCP and writes one
report. It supports kernel and range replay only:

```bash
ncu --communicator shmem --communicator-shmem-num-peers 2 -k regex:nccl -o report \
    torchrun --nnodes=1 --nproc_per_node=2 app.py
```

Without a communicator, profiling a kernel that waits on another rank hangs. On NVLink systems
where NCCL uses NVLS (NVLink SHARP), kernel replay of NCCL kernels fails with `Failed to save
memory for replay`: set `NCCL_NVLS_ENABLE=0` for the profiling run. `--replay-mode application`
with the TCP communicator also works, but reruns the whole job once per pass. Range replay
fails when NCCL's `cuMem*` allocation calls fall inside the range (unsupported during capture).

## PM Sampling

Samples performance-monitor metrics at a fixed interval while the kernel runs, giving a
timeline of how its behavior changes over its runtime (for example a tail where SMs go
idle). It is not a cheaper substitute for the other sections: the `PmSampling` section is
collected over several replay passes like they are.

```bash
ncu --section PmSampling --pm-sampling-interval 0 ...
```

- `--pm-sampling-interval 0` — auto interval
- `--pm-sampling-buffer-size 0` — auto buffer
- `--pm-sampling-max-passes 0` — auto passes
- `--warp-samples-per-interval 0` — auto warp samples per interval; `--disable-pm-warp-sampling` turns
  off the sampled warp stall reasons that `PmSampling` shows next to the metrics

### Warp State Sampling

```bash
ncu --section SourceCounters --warp-sampling-interval auto --warp-sampling-max-passes 5 ...
```

These options tune warp state sampling: the per-instruction stall samples of `SourceCounters`
and, from Nsight Compute 2026.2, the per-warp-slot samples that `WarpStateStats` shows next to
its counter-based stall breakdown. On their own they collect nothing. `PmSampling` has its own
warp sampling options (above).

## Profile Series

Profile Series, which profiles one kernel repeatedly with varying parameters,
is a feature of the Nsight Compute UI's Interactive Profile activity; the CLI
has no equivalent. From the CLI, profile each variant of a sweep and compare
the results:

```bash
ncu --section SpeedOfLight --kernel-name regex:"kernel" \
    --launch-count 10 -- python sweep.py
```

## Customization

### Custom Section Files

Section files (`.section` format, Protocol Buffer) define what metrics to collect and how to display them. The stock files ship in the `sections/` folder of the installation; ncu copies them to `~/Documents/NVIDIA Nsight Compute/<version>/Sections` and loads that copy.

`--section-folder` replaces the default search path: pass it again for the stock folder to keep
the stock sections and rules.

```bash
ncu --section-folder /path/to/custom/sections --list-sections    # Verify custom sections are discovered
ncu --section-folder /path/to/stock/sections --section-folder /path/to/custom/sections ...
```

### Derived Metrics

Compose new metrics from existing ones using math expressions (addition, subtraction, multiplication, division). Defined in section files, computed at collection time.

### Python Rules

Rules implement automated analysis logic:

```python
import NvRules

def get_identifier():
    return "my_custom_rule"

def get_name():
    return "My Custom Analysis"

def get_description():
    return "What the rule checks"

def get_section_identifier():
    return "SpeedOfLight"    # Runs when this section is collected

def apply(handle):
    ctx = NvRules.get_context(handle)
    # Access metrics and add recommendations
    ctx.frontend().message("Printed under the section")
```

Declare `get_section_identifier()`: in testing with Nsight Compute 2026.3.1, rules without it
were listed by `--list-rules` but printed nothing when profiling, even with `--rule`. The
optional `evaluate(handle)` declares metrics the rule needs. NVIDIA's templates are in
`extras/RuleTemplates/` of the installation. From 2026.3, rules run in an embedded Python that
ignores the user's packages and Python environment variables such as `PYTHONPATH`;
`NV_NSIGHT_PYTHON_ISOLATED=0` turns that off.

```bash
ncu --section-folder /path/to/rules --list-rules   # Verify custom rules are discovered
```

## Occupancy Calculator (Python)

The `ncu_occupancy` module (in `extras/python/`) calculates theoretical occupancy for different kernel configurations.

```python
import ncu_occupancy as occ

calc = occ.OccupancyCalculator(8, 0)  # compute capability 8.0 (A100)

params = occ.OccupancyParameters(
    threads_per_block=256,
    registers_per_thread=32,
    shared_mem_per_block=2048,
    shared_mem_size=32768,  # shared memory carveout, bytes
)

occupancy = calc.get_sm_occupancy(params)
limiters = calc.get_occupancy_limiters(params)
utilization = calc.get_resource_utilization(params)

# Best config found by varying the listed variables, others held fixed
optimal = calc.get_optimal_occupancy(params, [occ.OccupancyVariable.THREADS_PER_BLOCK])
```

Set `shared_mem_size` to the kernel's shared memory carveout, the `Shared Memory Configuration
Size` in LaunchStats; `occ.get_gpu_data(...)["shared_mem_size_configs"]` lists the valid values.
It defaults to 0, and with 0 this example reports 25% occupancy and no limiter instead of 100%.

### OccupancyVariable Enum

Variables that can be swept for optimization:
- `THREADS_PER_BLOCK`
- `REGISTERS_PER_THREAD`
- `SHARED_MEMORY_PER_BLOCK`
- `BLOCK_BARRIERS`

### GPU Data

```python
gpu_data = occ.get_gpu_data(major=8, minor=0)
# Returns per-architecture limits (no SM count): registers, warps, threads and blocks per SM,
# shared memory size configs, allocation granularities
```

## Reproducibility

### Clock Control

```bash
ncu --clock-control base ...    # Lock to base frequency (default before 2026.1)
ncu --clock-control boost ...   # Lock to boost frequency (default from 2026.1)
ncu --clock-control none ...    # Leave clocks alone, e.g. when locked with nvidia-smi
```

Fixed-frequency profiling produces more reproducible results. `--pipeline-boost-state stable`
(the default) also holds the Tensor Core boost state steady across runs.

### Cache Control

```bash
ncu --cache-control all ...     # Flush L1/L2 between replays (default)
ncu --cache-control none ...    # No flushing (shows cache-warm behavior)
```
