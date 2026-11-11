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

# Nsight Compute CLI Reference

Complete command-line reference for `ncu` (NVIDIA Nsight Compute CLI).

## Command Syntax

```bash
ncu [options] [--] <application> [application-arguments]
```

## Kernel Filtering

### By Name

```bash
ncu -k "volta_fp16_gemm" app              # Exact match
ncu -k regex:"gemm" app                    # Regex partial match
ncu -k regex:"(gemm|conv)" app             # Multiple patterns
ncu --kernel-name-base demangled -k "foo" app  # Match demangled names
```

### By Launch Index

```bash
ncu -s 5 -c 3 app       # Skip first 5 launches, profile next 3
ncu --kernel-id ::foo:2 app   # Second invocation of "foo"
ncu --kernel-id ::regex:^.*foo$: app  # All kernels ending in "foo"
```

### By NVTX Range

```bash
ncu --nvtx --nvtx-include "training/" app
ncu --nvtx --nvtx-include "Domain-A@range_name/" app
ncu --nvtx --nvtx-include "[bottom_range" app         # Bottom of stack
ncu --nvtx --nvtx-include "A_range/*/B_range" app     # Nested ranges
ncu --nvtx --nvtx-include "regex:iter_[0-9]+/" app    # Regex ranges
```

NVTX quantifiers: `/` sequence, `[` stack bottom, `]` stack top, `+` exactly one between, `*` zero or more between. Escape literal quantifiers with `\\`.

### By Call Stack

```bash
# Native (C/C++) call stack filtering
ncu --call-stack-type native --native-include "ModuleA@FileA.cpp@FuncA" app

# Python call stack filtering
ncu --call-stack-type python --python-include "FileA.py@FuncA" python script.py
```

Format: `<Module>@<File>@<Function>` (module and file optional).

A C++ function matches only by its full demangled signature as `nm -C` prints it, such
as `launch(Bufs const&)`; its bare name matches nothing. A function whose last statement
is the kernel launch can be tail-called off the stack at -O2/-O3: match its caller, or
build with `-fno-optimize-sibling-calls`. "No kernels were profiled" means no frame matched.

## Section and Metric Collection

### List Available Sections/Sets

```bash
ncu --list-sets app          # Section sets (basic, detailed, full, etc.)
ncu --list-sections app      # Individual sections
ncu --list-metrics app       # Metrics from active sections
ncu --list-rules app         # Available analysis rules
```

### Collect Specific Sections

```bash
ncu --section SpeedOfLight app
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis app
ncu --section "regex:.*Stats" app          # Regex section matching
```

### Collect Section Sets

```bash
ncu --set basic app       # LaunchStats, Occupancy, SpeedOfLight, WorkloadDistribution
ncu --set detailed app    # basic + ComputeWorkloadAnalysis, MemoryWorkloadAnalysis(_Chart), SourceCounters, RooflineChart, Tile
ncu --set full app        # Nearly all sections (~9000 metrics in Nsight Compute 2026.3)
ncu --set roofline app    # SpeedOfLight + all roofline charts
```

### Collect Individual Metrics

```bash
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed app
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed app
```

A throughput metric needs its full name: a bare `dram__throughput` fails with
"Failed to find metric". List a metric's full names with `--query-metrics-mode suffix`.

### Query Metric Availability

```bash
ncu --query-metrics app                                # Base names
ncu --query-metrics-mode suffix --metrics sm__throughput app
ncu --query-metrics-mode all app                       # Full metric names
ncu --query-metrics-collection pmsampling app           # PM sampling metrics
```

## Output Formats

### Console Pages

```bash
ncu --page details app    # Section-grouped results (default)
ncu --page raw app        # All metrics per kernel (flat)
ncu --page source app     # Source code correlation
ncu --page session app    # Launch settings and device info
```

### CSV Output

```bash
ncu --csv app                        # CSV to stdout
ncu --csv --page raw app             # All metrics in CSV
ncu --csv --print-units base app     # Base unit scaling
ncu --print-fp app                   # Floating point formatting
```

### Report Files

```bash
ncu -o report app                    # report.ncu-repz (2026.3+) or report.ncu-rep (older)
ncu -o report.ncu-rep app            # Name the extension to fix the file name
ncu -o report_%h_%p app              # With hostname and PID macros
ncu -o report_%q{OMPI_COMM_WORLD_RANK} app  # With env var macro
ncu -f -o report app                 # Force overwrite
```

File macro expansions: `%h` hostname, `%p` PID, `%q{VAR}` env var, `%i` auto-increment, `%%` literal %.

Without an extension, `-o` adds `.ncu-repz` (zstd-compressed) from Nsight Compute 2026.3 on and
`.ncu-rep` before. `--import` and `ncu_report` read both formats.

### Source Code Display

```bash
ncu --page source --print-source sass app          # SASS assembly (the default view)
ncu --page source --print-source cuda app          # CUDA-C source
ncu --page source --print-source cuda,sass app     # Both correlated
```

`--print-source` requires `--page source`. The CUDA-C views need a build with
`-lineinfo`; the SASS view and its per-instruction metrics do not. Other views are `ptx`
and, for CUDA Tile kernels, `tileir`, `cuda,tileir`, `tileir,ptx` and `tileir,sass`
(Nsight Compute 2026.3+).

### Summary Modes

```bash
ncu --print-summary per-gpu app      # Per GPU
ncu --print-summary per-kernel app   # Per kernel type
ncu --print-summary per-nvtx app     # Per NVTX context
```

### Metric Instances

```bash
ncu --print-metric-instances none app     # GPU aggregate only
ncu --print-metric-instances values app   # Aggregate + per-instance
ncu --print-metric-instances details app  # With correlation IDs
```

## Report Import/Export

```bash
ncu --import report.ncu-rep --page details
ncu --import report.ncu-rep --page raw --csv
ncu --import old.ncu-rep --export new.ncu-rep --kernel-name "regex:foo"
```

## Cache and Clock Control

```bash
--cache-control all          # Flush caches before replays (default)
--cache-control none         # No cache flushing
--clock-control base         # Base frequency (default before Nsight Compute 2026.1)
--clock-control boost        # Boost frequency, or base where boost is unsupported (default from 2026.1)
--clock-control force-boost  # Boost frequency, no fallback
--clock-control none         # No clock changes
--clock-control reset        # Reset GPU clocks and exit
--pipeline-boost-state stable # Stable Tensor Core boost state (default); dynamic lets it vary
```

## Replay Modes

```bash
--replay-mode kernel         # Individual kernel replay (default)
--replay-mode application    # Full application reruns
--replay-mode range          # Ranges from cudaProfilerStart/Stop, or NVTX with --nvtx-include
--replay-mode app-range      # Application-level range replay
```

## Profiler Start Control

```bash
ncu --profile-from-start off app    # Wait for cudaProfilerStart()
```

Pair with profiler markers in code:
```python
torch.cuda.cudart().cudaProfilerStart()
# ... profiled region ...
torch.cuda.cudart().cudaProfilerStop()
```

## Device Selection

```bash
ncu --devices 0,2 app        # Profile specific GPUs
```

## CUDA Graph Profiling

```bash
ncu --graph-profiling node app    # Individual nodes (default)
ncu --graph-profiling graph app   # Entire graph as workload
```

## Multi-Process and MPI

```bash
# Profile all processes
ncu --target-processes all -o report mpirun app

# Per-rank reports
mpirun ncu -o report_%q{OMPI_COMM_WORLD_RANK} app

# Synchronized profiling (for NCCL/NVSHMEM dependent kernels)
mpirun -np 4 ncu --communicator=tcp --communicator-tcp-num-peers=4 \
  --lockstep-kernel-launch -o report app

# Restrict synchronization to specific NVTX ranges (requires --lockstep-kernel-launch)
mpirun ncu --communicator=tcp --communicator-tcp-num-peers=4 --lockstep-kernel-launch \
  --lockstep-nvtx-include "nccl/" -o report app

# One ncu over every rank of one node (2026.1+)
ncu --communicator shmem --communicator-shmem-num-peers 2 -k regex:nccl -o report \
  torchrun --nnodes=1 --nproc_per_node=2 app.py
```

On NVLink systems where NCCL uses NVLS, NCCL kernels need `NCCL_NVLS_ENABLE=0` under kernel
replay; see `advanced-profiling.md`.

### Process Filtering

```bash
ncu --target-processes-filter "MatrixMul" app       # Exact name
ncu --target-processes-filter "regex:Matrix" app     # Regex
ncu --target-processes-filter "exclude:MatrixMul" app
```

## MPS (Multi-Process Service)

`ncu --mps client` only launches a client, suspended; a separate `ncu --mps control` process
attaches to the clients and profiles them, so profiling options go on the control process:

```bash
ncu --mps client ./client_app 1                    # Launch each client (add --nvtx if ranges use NVTX)
ncu --mps client ./client_app 2
ncu --mps control --mps-num-clients 2 --replay-mode range -o report    # Wait for 2 clients, profile
ncu --mps control --replay-mode range -o report ./client_app           # Or launch a single client itself
```

Launch a client with `--mps primary-client` to limit the profiled window to that client's
duration. Only kernel and range replay work; prefer range replay, because with kernel replay
each client contributes a single kernel launch. MPS profiling is CLI-only.

## Configuration Files

ncu reads `config.ncu-cfg` from the current working directory, then from
`$HOME/.config/NVIDIA Corporation/`. A stray one in the working directory
silently changes every run there; `--config-file off` ignores it.

```ini
[Launch-and-attach]
-c = 1
--section = LaunchStats, Occupancy

[Import]
--open-in-ui
```

```bash
ncu --config-file on app                            # Enable (default)
ncu --config-file-path /path/config.ncu-cfg app     # Custom path
```

## Response Files

```bash
ncu @myoptions.txt app    # Read options from file
```

## PM and Warp Sampling

These options tune sampling that a section requests: PM sampling for `PmSampling`, warp state
sampling for `SourceCounters`. Without such a section they collect nothing.

```bash
ncu --section PmSampling --pm-sampling-interval 0 app              # Auto interval
ncu --section SourceCounters --warp-sampling-interval auto app     # Auto warp sampling
ncu --section SourceCounters --warp-sampling-max-passes 5 app
```

## Kernel Renaming

```bash
ncu --rename-kernels-path renames.yaml --kernel-name "MyKernel" app
ncu --rename-kernels-export on -o report app   # Export names for renaming
```

The export writes `$HOME/.config/NVIDIA Corporation/ncu-kernel-renames.yaml`
(or `--rename-kernels-path`), and later runs apply it automatically: renaming
is on by default and also reads `ncu-kernel-renames.yaml` from the working
directory.

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `NV_COMPUTE_PROFILER_DISABLE_STOCK_FILE_DEPLOYMENT` | Skip versioned section dir |
| `NV_COMPUTE_PROFILER_LOCAL_CONNECTION_OVERRIDE` | Connection: `uds`, `tcp`, `named-pipes` |
| `NV_COMPUTE_PROFILER_DISABLE_CONCURRENT_PROFILING` | Single-profiler system lock |
| `NV_NSIGHT_PYTHON_ISOLATED` | `0` lets rules use the user's Python packages and `PYTHONPATH` (isolated by default from 2026.3) |

## Miscellaneous Options

```bash
ncu --null-stdin app                # Suppress stdin blocking
ncu --check-exit-code yes app       # Validate app exit code
ncu --support-32bit app             # Only for processes launched from a 32-bit app; 2026.1.1 hung on a 64-bit app, 2026.3.1 did not
ncu --section-folder /path app      # Custom section file location
```
