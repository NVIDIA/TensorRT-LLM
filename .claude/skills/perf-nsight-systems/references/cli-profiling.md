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

# nsys profile CLI Reference

Complete CLI reference for `nsys profile` — the primary collection command.

## Command Syntax

```bash
nsys profile [options] [--] <application> [application-arguments]
```

## CUDA Tracing Options

### --trace / -t (API selection)

Comma-separated list (no spaces). Defaults: `cuda,opengl,nvtx,osrt`.

DL-relevant trace targets:

| Target | What it traces |
|--------|---------------|
| `cuda` | CUDA runtime and driver API calls |
| `nvtx` | NVTX annotations (ranges, markers) |
| `osrt` | OS runtime calls (malloc, pthread, file I/O) |
| `cudnn` | cuDNN API calls |
| `cublas` | cuBLAS API calls |
| `cublas-verbose` | cuBLAS with extended info |
| `cusolver` | cuSolver API calls |
| `cusparse-verbose` | cuSPARSE with extended info |
| `mpi` | MPI calls |
| `ucx` | UCX communication calls |
| `python-gil` | Python GIL acquisition/release |
| `gds` | GPUDirect Storage |
| `none` | Disable all tracing |

```bash
# Typical DL profiling
nsys profile -t cuda,nvtx,cudnn,cublas,osrt -- python train.py

# Distributed DL
nsys profile -t cuda,nvtx,mpi,ucx -- torchrun --nproc_per_node=8 train.py

# Minimal overhead
nsys profile -t cuda,nvtx -- python train.py
```

### CUDA-Specific Trace Options

| Flag | Default | Description |
|------|---------|-------------|
| `--cuda-graph-trace` | `graph` | `graph` (low overhead) or `node` (per-node detail) |
| `--cuda-memory-usage` | `false` | Track GPU memory per kernel (significant overhead) |
| `--cuda-um-cpu-page-faults` | `false` | Track Unified Memory CPU page faults (significant overhead) |
| `--cuda-um-gpu-page-faults` | `false` | Track Unified Memory GPU page faults (significant overhead) |
| `--cuda-event-trace` | `false` | Trace CUDA event completion |
| `--cuda-trace-scope` | `process-tree` | `process-tree` or `system-wide` |
| `--cuda-flush-interval` | `0` | ms between buffer saves (10000 for Embedded) |
| `--flush-on-cudaprofilerstop` | `true` | Flush CUDA buffers on `cudaProfilerStop()` |

### CUDA Backtrace Collection

```bash
# Backtraces for all CUDA APIs taking > 1000ns
nsys profile --cudabacktrace=all -- python train.py

# Only kernel launches and sync calls with custom threshold
nsys profile --cudabacktrace=kernel:5000,sync:10000 -- python train.py
```

Options: `all`, `none`, `kernel`, `memory`, `sync`, `other` (comma-separated). Append `:threshold_ns` per type. Requires CPU sampling enabled.

## Python and PyTorch Options

| Flag | Default | Description |
|------|---------|-------------|
| `--python-backtrace` | `none` | Collect Python backtraces on CUDA API calls |
| `--python-sampling` | `false` | Enable Python backtrace sampling |
| `--python-sampling-frequency` | `1000` | Sampling rate in Hz (1-2000) |
| `--python-functions-trace` | none | Path to JSON with NVTX annotations for Python functions |
| `--pytorch` | `none` | Auto-annotate PyTorch ops |
| `--dask` | `none` | Auto-annotate Dask tasks |

### --pytorch values

| Value | Effect |
|-------|--------|
| `autograd-nvtx` | NVTX ranges around autograd ops |
| `autograd-shapes-nvtx` | NVTX with tensor shape info |
| `functions-trace` | Trace PyTorch Python functions |
| `none` | Disable |

```bash
# PyTorch with autograd NVTX + shapes
nsys profile --pytorch=autograd-shapes-nvtx -t cuda,nvtx -- python train.py

# PyTorch + Python sampling
nsys profile --pytorch=autograd-nvtx --python-sampling=true \
    -t cuda,nvtx -- python train.py
```

## Duration and Timing

| Flag | Default | Description |
|------|---------|-------------|
| `--delay` / `-y` | `0` | Start collection after N seconds |
| `--duration` / `-d` | unlimited | Collect for N seconds |
| `--start-later` / `-Y` | off | Delay until `nsys start` is called |

```bash
# Profile 30 seconds, skip first 10
nsys profile -y 10 -d 30 -- python train.py
```

## Capture Range Control

### --capture-range / -c

| Value | Trigger |
|-------|---------|
| `none` | Profile entire run (default) |
| `cudaProfilerApi` | Start/stop via `cudaProfilerStart()/Stop()` |
| `nvtx` | Start/stop on NVTX range entry/exit |
| `hotkey` | F-key trigger |

### --capture-range-end

| Value | Behavior |
|-------|----------|
| `stop-shutdown` | Stop collection and exit (default) |
| `stop` | Stop collection, app continues |
| `repeat[:N]` | Restart on next trigger (optionally N times) |
| `repeat-shutdown:N` | Repeat N times then shutdown |

### NVTX-Triggered Profiling

```bash
# Profile only the "training" NVTX range
nsys profile -c nvtx -p "training@my_domain" \
    --capture-range-end=repeat:5 -- python train.py

# Profile a specific iteration range
nsys profile -c nvtx -p "iteration" -- python train.py
```

`--nvtx-capture` / `-p` format: `range@domain`, `range`, `range@*`.

### NVTX Domain Filtering

| Flag | Description |
|------|-------------|
| `--nvtx-domain-include` | Only trace these domains (comma-separated) |
| `--nvtx-domain-exclude` | Exclude these domains |

## Output Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output` / `-o` | `report#` | Report filename (auto-numbered) |
| `--force-overwrite` / `-f` | `false` | Overwrite existing files |
| `--stats` | `false` | Print summary stats after collection |
| `--export` | `none` | Auto-export: `sqlite`, `arrow`, `hdf`, `jsonlines`, `parquetdir`, `text` |

Filename macros: `%h` hostname, `%p` PID, `%q{ENV_VAR}` env variable, `%%` literal %.

```bash
# Per-rank output for distributed training
nsys profile -o report_%q{RANK}_%h -t cuda,nvtx,mpi -- torchrun train.py

# Auto-generate stats and SQLite export
nsys profile --stats=true --export=sqlite -o my_profile -- python train.py
```

## CPU Sampling

| Flag | Default | Description |
|------|---------|-------------|
| `--sample` / `-s` | `process-tree` | `process-tree`, `system-wide`, `none` |
| `--sampling-frequency` | `1000` | Hz (100-8000) |
| `--backtrace` / `-b` | `auto` | `fp`, `lbr`, `dwarf`, `none` |
| `--samples-per-backtrace` | `4` (DWARF) | IP samples per backtrace (max 32) |

## GPU Metrics

| Flag | Default | Description |
|------|---------|-------------|
| `--gpu-metrics-devices` | `none` | GPU IDs to monitor (use `help` for list) |
| `--gpu-metrics-frequency` | `10000` | Hz (10-200000) |
| `--gpu-metrics-set` | auto | Metric set alias or `file:path` |
| `--gpuctxsw` | `false` | Trace GPU context switches |

```bash
# Collect GPU metrics on all devices at 1kHz
nsys profile --gpu-metrics-devices=all --gpu-metrics-frequency=1000 \
    -t cuda,nvtx -- python train.py
```

## MPI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mpi-impl` | auto-detect | `openmpi` or `mpich` |

```bash
# MPI with explicit implementation
nsys profile --mpi-impl=openmpi -t cuda,nvtx,mpi \
    -- mpirun -np 4 python train.py
```

## InfiniBand / Network Monitoring

| Flag | Description |
|------|-------------|
| `--nic-metrics` | Collect NIC/HCA metrics (system scope) |
| `--ib-switch-metrics-devices` | Switch GUIDs for IB metrics |
| `--ib-switch-congestion-devices` | Switch GUIDs for congestion events |
| `--ib-net-info-devices` | NIC names for network discovery |

## Process and Environment

| Flag | Default | Description |
|------|---------|-------------|
| `--env-var` / `-e` | none | Set env vars (`A=B,C=D`) |
| `--kill` | `sigterm` | Signal on exit: `none`, `sigkill`, `sigterm` |
| `--stop-on-exit` / `-x` | `true` | Stop when process exits |
| `--resolve-symbols` | `true` | Resolve symbols in report |
| `--session-new` | auto | Name the profiling session |
| `--command-file` | none | Load options from file |

## Callback Commands

| Flag | Description |
|------|-------------|
| `--after-collection-start` | Execute command after collection starts |
| `--after-report-ready` | Execute after report is generated |

Both receive `NSYS_SESSION_NAME`, `NSYS_CALLBACK_NAME` env vars. `--after-report-ready` also gets `NSYS_REPORT_PATH`.

## Key Defaults Summary

| Setting | Default |
|---------|---------|
| Traced APIs | `cuda,opengl,nvtx,osrt` |
| CPU sampling | `process-tree` (if app launched) |
| Duration | Unlimited |
| Report name | `report#` (auto-numbered) |
| Exit signal | `SIGTERM` |
| Symbol resolution | Enabled |
| GPU metrics | Disabled |
