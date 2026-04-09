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

# Bottleneck Analysis Guide

Root-cause analysis and optimization strategies for each bottleneck type identified by SOL% classification.

## SOL% Classification

| Compute % | Memory % | Bottleneck | Primary Section |
|-----------|----------|------------|-----------------|
| >60 | <40 | **Compute-bound** | ComputeWorkloadAnalysis |
| <40 | >60 | **Memory-bound** | MemoryWorkloadAnalysis |
| <40 | <40 | **Latency-bound** | LaunchStats + Occupancy |
| 40-60 | 40-60 | **Balanced** | Profile deeper with detailed sections |

Additional signals:
- Duration <10us with many launches: **Launch-overhead bound** (use nsys first)
- Both <40% but occupancy >50%: **Instruction-bound** (check InstructionStats)

## SOL% Performance Levels

| SOL% | Level | Action |
|------|-------|--------|
| >80% | Excellent | Minor tuning only |
| 60-80% | Good | Targeted optimization |
| 40-60% | Fair | Significant optimization needed |
| <40% | Poor | Major rework needed |

## Compute-Bound Kernels

**Symptoms:** Compute throughput >60%, Memory throughput <40%. Heavy arithmetic operations.

**Key Metrics:**
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` — compute throughput
- `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` — tensor core usage
- `smsp__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_elapsed` — FP64 usage

**Root Causes:**
- Heavy FP operations without tensor core usage
- Inefficient math (FP64 when FP32 suffices)
- Warp divergence in compute paths

**Optimization Priority:**
1. Enable tensor cores (FP16/BF16/FP8 operations)
2. Use faster math intrinsics (`--use_fast_math`)
3. Reduce warp divergence
4. Algorithmic improvements (reduce FLOPs)

## Memory-Bound Kernels

**Symptoms:** Memory throughput >60%, Compute throughput <40%. Low arithmetic intensity.

**Key Metrics:**
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` — DRAM bandwidth
- `l1tex__t_sector_hit_rate.pct` — L1 cache hit rate
- `lts__t_sector_hit_rate.pct` — L2 cache hit rate

**Root Causes:**
- Large data movement per operation
- Poor cache utilization (many misses)
- Uncoalesced memory access
- Shared memory bank conflicts

**Optimization Priority:**
1. Kernel fusion to increase arithmetic intensity
2. Improve data locality and cache reuse
3. Use shared memory for frequently accessed data
4. Ensure coalesced global memory access
5. Lower precision formats (FP16, INT8) to reduce bandwidth

**Memory becomes limiting when:** hardware units are fully utilized (Mem Busy), communication bandwidth between units is exhausted (Max Bandwidth), or memory instruction issue rate is maxed (Mem Pipes Busy).

## Latency-Bound (Low Occupancy)

**Symptoms:** Both throughputs <40%, Achieved occupancy <50%.

**Key Metrics:**
- `sm__warps_active.avg.pct_of_peak_sustained_active` — active warps
- `launch__occupancy_limit_registers` — register limit
- `launch__occupancy_limit_shared_mem` — shared memory limit
- `launch__occupancy_limit_blocks` — block limit

**Root Causes:**
- High register usage per thread (>64)
- Large shared memory per block
- Small block sizes
- Block dimension limitations

**Diagnosis Table:**

| Symptom | Cause | Solution |
|---------|-------|----------|
| Theoretical occupancy < 50% | Register pressure | `--maxrregcount`, occupancy hint |
| Theoretical occupancy < 50% | Shared memory | Reduce shared memory per block |
| Achieved << Theoretical | Workload imbalance | Adjust grid/block dimensions |
| Both throughputs <40% | Low occupancy | Check LaunchStats for limiting resource |

**Optimization Priority:**
1. Reduce register usage (`--maxrregcount`)
2. Reduce shared memory per block
3. Adjust block dimensions (128-256 threads typical)
4. Trade register usage for memory access if net-positive

## Instruction-Bound

**Symptoms:** High instruction count relative to useful compute. Warp divergence indicators.

**Key Metrics:**
- `smsp__inst_executed.sum` — total instructions
- `smsp__thread_inst_executed_per_inst_executed.ratio` — divergence indicator (ideal=32)
- `smsp__inst_executed_pipe_cbu.avg.pct_of_peak_sustained_elapsed` — control flow overhead

**Root Causes:**
- Excessive control flow (branches)
- Warp divergence
- Many low-throughput instructions
- Instruction cache misses

**Optimization Priority:**
1. Simplify control flow
2. Use predication instead of branches
3. Reorganize data to reduce divergence
4. Unroll loops where beneficial

## Launch-Overhead Bound

**Symptoms:** Very short kernel durations (<10us), many launches, high CPU time between them.

**Key Metrics:**
- `gpu__time_duration.sum` — kernel duration
- Launch count and CPU-GPU gaps (from nsys trace)

**Note:** This is better diagnosed with `nsys` (Nsight Systems) which shows the system-level timeline. Use `ncu` to confirm individual kernel performance after identifying hot kernels with `nsys`.

**Optimization Priority:**
1. CUDA Graphs to batch kernel launches
2. Fuse small kernels together
3. Increase work per kernel launch
4. Persistent kernels for repeated small work

## Quick Optimization Map

| Bottleneck Type | First Try | Second Try | Third Try |
|-----------------|-----------|------------|-----------|
| Compute-bound | Enable tensor cores | Mixed precision | Algorithmic opt |
| Memory-bound | Kernel fusion | Improve locality | Shared memory |
| Latency-bound | Adjust block size | Reduce registers | Increase parallelism |
| Instruction-bound | Simplify control flow | Use predication | Loop unrolling |
| Launch-overhead | CUDA Graphs | Kernel fusion | Persistent kernels |

## ncu vs nsys

| Tool | Scope | Overhead | Purpose |
|------|-------|----------|---------|
| **nsys** | System-level | 5-10% | Find which kernels to optimize |
| **ncu** | Kernel-level | 10-100x slower | Understand why a kernel is slow |

Use nsys first to identify top kernels by GPU time, then ncu for deep analysis of those specific kernels.
