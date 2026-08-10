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

# Memory Analysis Guide

Interpreting memory hierarchy performance in Nsight Compute: memory chart, memory tables, and cache analysis.

## Memory Chart Overview

The Memory Chart visualizes the GPU memory hierarchy as a diagram of interconnected units:

- **Logical Units (green):** High-level memory abstractions (shared memory, caches) that code interacts with
- **Physical Units (blue):** Actual hardware (cache slices, memory controllers)
- **Links:** Data flow paths between units with bandwidth metrics
- **Ports:** Interface points where data transfers occur

## Memory Hierarchy: From Fastest to Slowest

### Registers
- Per-thread, fastest access
- Limited per SM; high usage limits occupancy
- Spills go to local memory (device memory speed)

### Shared Memory
- On-chip, per SM, shared by threads in the same block
- Organized in 32 banks for parallel access
- Bank conflicts serialize access, reducing throughput
- Configured alongside L1 cache (shared capacity)

### L1/TEX Cache
- Per-SM, unified with shared memory
- Handles global, local, shared, texture, and surface memory operations
- Two instances per TPC on modern architectures

### L2 Cache
- Global cache shared by all SMs
- All GPU units communicate to main memory through L2
- Operates in physical-address space
- Handles compression and atomic operations

### Device Memory (DRAM)
- Off-chip, highest capacity, highest latency
- Highest bandwidth when access is coalesced (128-byte transactions)

## Key Memory Metrics

### Cache Hit Rates

| Metric | Good | Poor | Meaning |
|--------|------|------|---------|
| `l1tex__t_sector_hit_rate.pct` | >80% | <50% | L1 data locality |
| `lts__t_sector_hit_rate.pct` | >80% | <50% | L2 data locality |

Low hit rates indicate:
- Working set exceeds cache capacity
- Poor spatial locality (scattered access patterns)
- No data reuse between threads

### Bandwidth Utilization

| Metric | Meaning |
|--------|---------|
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | % of peak DRAM bandwidth used |
| `l1tex__throughput.avg.pct_of_peak_sustained_elapsed` | L1 throughput utilization |
| `lts__throughput.avg.pct_of_peak_sustained_elapsed` | L2 throughput utilization |

### Coalescing Efficiency

Global memory loads/stores should coalesce into minimal 128-byte transactions. Metrics to watch:

| Metric | Ideal | Problem Indicator |
|--------|-------|-------------------|
| Global Load Efficiency | >90% | <50% = uncoalesced reads |
| Global Store Efficiency | >90% | <50% = uncoalesced writes |

**Causes of poor coalescing:**
- Strided access patterns (accessing every Nth element)
- Misaligned base addresses
- Structure-of-arrays vs array-of-structures layout

### Shared Memory Bank Conflicts

Shared memory has 32 banks. Conflicts occur when multiple threads in a warp access different addresses in the same bank.

**Metric:** Shared memory wavefronts per request > 1 indicates conflicts.

**Fixes:**
- Pad shared memory arrays to avoid stride conflicts
- Rearrange access patterns for conflict-free access
- Use `__shfl_*` warp shuffle when possible

## Memory Bottleneck Diagnosis

### Step 1: Is the Kernel Memory-Bound?

Check SpeedOfLight: Memory Throughput >60% and Compute Throughput <40%.

### Step 2: Where in the Memory Hierarchy?

Run `--section MemoryWorkloadAnalysis` and check:

| Finding | Interpretation | Action |
|---------|---------------|--------|
| High DRAM throughput, low cache hit | Data not reused | Improve locality, kernel fusion |
| Low DRAM throughput, high L2 hit | Good L2 caching | Check if L1 can be improved |
| Low DRAM throughput, low L2 hit | Bandwidth underused | Check coalescing, occupancy |
| High shared memory traffic | Possible bank conflicts | Check wavefronts per request |

### Step 3: Identify the Memory Limiter

Memory performance is limited when any of these saturates:
- **Mem Busy:** Hardware units fully utilized
- **Max Bandwidth:** Communication bandwidth between units exhausted
- **Mem Pipes Busy:** Memory instruction issue rate maxed

## Memory Tables (MemoryWorkloadAnalysis Detail)

### Shared Memory Table
Per-block shared memory usage, bank conflict counts, access efficiency.

### L1/TEX Cache Table
Hit/miss rates, transaction patterns, sector access details.

### L2 Cache Table
Request counts, hits, misses, bandwidth utilization per slice.

### L2 Cache Eviction Policies Table
Eviction behavior when cache capacity is exceeded.

### Device Memory Table
Global memory access patterns, DRAM bandwidth consumption.

## Common Memory Optimization Patterns

### Kernel Fusion
Combine multiple memory-bound kernels to reuse data in registers/shared memory instead of writing back to DRAM between kernels.

### Tiling with Shared Memory
Load a tile of data from global memory into shared memory, compute on it, then write results back. Reduces redundant global memory reads.

### Data Layout Optimization
- **Structure of Arrays (SoA):** Better coalescing for GPU
- **Array of Structures (AoS):** Often leads to uncoalesced access
- Align data to 128-byte boundaries

### Precision Reduction
Use FP16/BF16/INT8 where accuracy permits. Halves/quarters the memory bandwidth requirement per element.
