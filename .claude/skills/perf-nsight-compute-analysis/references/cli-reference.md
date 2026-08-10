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
ncu --set detailed app    # basic + ComputeWorkloadAnalysis, MemoryWorkloadAnalysis, SourceCounters
ncu --set full app        # All sections (~8051 metrics)
ncu --set roofline app    # SpeedOfLight + all roofline charts
```

### Collect Individual Metrics

```bash
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed app
ncu --metrics sm__throughput,dram__throughput app
```

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
ncu -o report app                    # Save as report.ncu-rep
ncu -o report_%h_%p app              # With hostname and PID macros
ncu -o report_%q{OMPI_COMM_WORLD_RANK} app  # With env var macro
ncu -f -o report app                 # Force overwrite
```

File macro expansions: `%h` hostname, `%p` PID, `%q{VAR}` env var, `%i` auto-increment, `%%` literal %.

### Source Code Display

```bash
ncu --print-source sass app          # SASS assembly
ncu --print-source cuda app          # CUDA-C source
ncu --print-source cuda,sass app     # Both correlated
```

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
--clock-control base         # Base frequency (default)
--clock-control boost        # Boost frequency
--clock-control none         # No clock changes
```

## Replay Modes

```bash
--replay-mode kernel         # Individual kernel replay (default)
--replay-mode application    # Full application reruns
--replay-mode range          # Range-based (cudaProfilerStart/Stop)
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
mpirun -np 4 ncu --communicator=tcp --communicator-num-peers=4 \
  --lockstep-kernel-launch -o report app

# Restrict synchronization to specific NVTX ranges
mpirun ncu --communicator=tcp --communicator-num-peers=4 \
  --lockstep-nvtx-include "nccl/" -o report app
```

### Process Filtering

```bash
ncu --target-processes-filter "MatrixMul" app       # Exact name
ncu --target-processes-filter "regex:Matrix" app     # Regex
ncu --target-processes-filter "exclude:MatrixMul" app
```

## MPS (Multi-Process Service)

```bash
ncu --mps client app                    # Profile as MPS client
ncu --mps primary-client app            # Primary client role
ncu --mps-num-clients 4 app             # Expected client count
```

## Configuration Files

Default location: `$HOME/.config/NVIDIA Corporation/config.ncu-cfg`

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

```bash
ncu --pm-sampling-interval 0 app          # Auto interval
ncu --warp-sampling-interval auto app     # Auto warp sampling
ncu --warp-sampling-max-passes 5 app
```

## Kernel Renaming

```bash
ncu --rename-kernels-path renames.yaml --kernel-name "MyKernel" app
ncu --rename-kernels-export on -o report app   # Export names for renaming
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `NV_COMPUTE_PROFILER_DISABLE_STOCK_FILE_DEPLOYMENT` | Skip versioned section dir |
| `NV_COMPUTE_PROFILER_LOCAL_CONNECTION_OVERRIDE` | Connection: `uds`, `tcp`, `named-pipes` |
| `NV_COMPUTE_PROFILER_DISABLE_CONCURRENT_PROFILING` | Single-profiler system lock |

## Miscellaneous Options

```bash
ncu --null-stdin app                # Suppress stdin blocking
ncu --check-exit-code yes app       # Validate app exit code
ncu --support-32bit app             # Linux 32-bit support
ncu --section-folder /path app      # Custom section file location
```
