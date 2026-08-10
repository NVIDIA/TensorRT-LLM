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

# Nsight Compute Sections Guide

All available `ncu --section` names, what they measure, and when to use each.

## Core Analysis Sections

### SpeedOfLight

**Command**: `--section SpeedOfLight`

The most important section. Provides GPU throughput as percentage of theoretical peak.

**Key Metrics:**
- Compute (SM) Throughput (%)
- Memory Throughput (%)

**When to Use:** Always start here. Classifies the kernel bottleneck type.

### Occupancy

**Command**: `--section Occupancy`

Warp occupancy and resource limiters.

**Key Metrics:**
- Achieved Occupancy vs Theoretical Occupancy
- Occupancy limiters: registers, shared memory, blocks
- Registers per thread, shared memory per block

**When to Use:** When both throughputs <40% (latency-bound), or to diagnose resource constraints.

### MemoryWorkloadAnalysis

**Command**: `--section MemoryWorkloadAnalysis`

Deep dive into memory access patterns and cache behavior.

**Key Metrics:**
- L1/L2 cache hit rates
- Memory load/store throughput
- Coalescing efficiency
- Shared memory bank conflicts

**When to Use:** When memory throughput >60% (memory-bound kernel).

### ComputeWorkloadAnalysis

**Command**: `--section ComputeWorkloadAnalysis`

Compute pipeline utilization breakdown.

**Key Metrics:**
- Pipeline utilization: FP32, FP64, tensor, integer
- Instruction issue rates
- Warp scheduler activity

**When to Use:** When compute throughput >60% (compute-bound kernel).

### SchedulerStats

**Command**: `--section SchedulerStats`

Warp scheduler statistics.

**Key Metrics:**
- Eligible warps per scheduler
- Scheduler issue rate
- Warp stall reasons (summary)

**When to Use:** When occupancy is low or scheduler seems underutilized.

### WarpStateStats

**Command**: `--section WarpStateStats`

Detailed warp stall reason breakdown.

**Key Metrics:**
- Stall cycles by reason: memory dependency, execution dependency, sync, etc.
- Active vs stalled warp ratio

**When to Use:** After SchedulerStats shows stalls; identifies root cause.

### InstructionStats

**Command**: `--section InstructionStats`

Instruction mix and execution statistics.

**Key Metrics:**
- Instructions executed by type
- Instruction throughput
- Predicated instructions

**When to Use:** Instruction-bound kernels, or detailed compute analysis.

### LaunchStats

**Command**: `--section LaunchStats`

Kernel launch configuration parameters.

**Key Metrics:**
- Grid and block dimensions
- Shared memory config
- Register usage per thread

**When to Use:** Verify launch configuration. Always include with Occupancy for latency-bound diagnosis.

## Memory Sections

### WorkloadDistribution

**Command**: `--section WorkloadDistribution`

How work is distributed across GPU resources.

**Key Metrics:**
- SM utilization distribution
- Memory unit utilization

**When to Use:** Identify load imbalance across SMs.

### SourceCounters

**Command**: `--section SourceCounters`

Source-level metrics (requires `-lineinfo` compilation flag).

**Key Metrics:**
- Per-line execution counts
- Per-line stall reasons

**When to Use:** Correlate metrics to source code lines.

## Roofline Sections

### SpeedOfLight_RooflineChart

**Command**: `--section SpeedOfLight_RooflineChart`

Overview roofline chart: kernel position vs memory and compute roofs.

### SpeedOfLight_HierarchicalSingleRooflineChart

**Command**: `--section SpeedOfLight_HierarchicalSingleRooflineChart`

Hierarchical roofline for FP32 with L1, L2, DRAM roofs.

### SpeedOfLight_HierarchicalHalfRooflineChart

**Command**: `--section SpeedOfLight_HierarchicalHalfRooflineChart`

Hierarchical roofline for FP16 operations.

### SpeedOfLight_HierarchicalTensorRooflineChart

**Command**: `--section SpeedOfLight_HierarchicalTensorRooflineChart`

Hierarchical roofline for tensor core operations.

## Multi-GPU Sections

### Nvlink

**Command**: `--section Nvlink`

NVLink utilization. **When to Use:** Multi-GPU workloads with NVLink.

### Nvlink_Tables / Nvlink_Topology

**Command**: `--section Nvlink_Tables --section Nvlink_Topology`

Detailed NVLink bandwidth, port mappings, and topology.

### NumaAffinity

**Command**: `--section NumaAffinity`

NUMA affinity and CPU-GPU distance. **When to Use:** Investigating CPU-GPU affinity issues.

## Special Sections

### PmSampling / PmSampling_WarpStates

**Command**: `--section PmSampling`

Performance monitoring sampling (lower overhead than full profiling).

### Tile

**Command**: `--section Tile`

Tile statistics for tiled/blocked algorithms.

### C2CLink

**Command**: `--section C2CLink`

Chip-to-chip link analysis (Grace Hopper systems).

## Section Sets

Pre-defined collections for convenience:

| Set | Sections Included | ~Metrics | When to Use |
|-----|-------------------|----------|-------------|
| `basic` | LaunchStats, Occupancy, SpeedOfLight, WorkloadDistribution | 213 | Quick overview |
| `detailed` | basic + ComputeWorkloadAnalysis, MemoryWorkloadAnalysis, SourceCounters | 996 | Standard analysis |
| `full` | All sections | 8051 | Exhaustive (slow) |
| `roofline` | SpeedOfLight + all roofline charts | 6679 | Roofline focus |
| `nvlink` | Nvlink, Nvlink_Tables, Nvlink_Topology | 122 | Multi-GPU focus |

**Recommendation**: Prefer individual `--section` flags for targeted, faster analysis. Use `--set` only for broad exploration.

## Bottleneck-to-Section Map

| Classification | Sections to Add |
|----------------|-----------------|
| Compute-bound (SM >60%) | `ComputeWorkloadAnalysis` |
| Memory-bound (Mem >60%) | `MemoryWorkloadAnalysis` |
| Latency-bound (both <40%) | `LaunchStats`, `Occupancy` |
| Warp stalls | `WarpStateStats`, `SchedulerStats` |
| Instruction-bound | `InstructionStats` |
| Source-level analysis | `SourceCounters` (needs `-lineinfo`) |
| Load imbalance | `WorkloadDistribution` |

## Combining Sections

```bash
# Memory-bound deep dive
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy ...

# Compute-bound deep dive
ncu --section SpeedOfLight --section ComputeWorkloadAnalysis --section InstructionStats ...

# Full occupancy investigation
ncu --section Occupancy --section SchedulerStats --section WarpStateStats ...
```

Each additional section increases profiling time. Add only what the diagnosis requires.
