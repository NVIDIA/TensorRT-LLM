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

# Roofline Analysis Guide

Understanding and interpreting roofline charts in Nsight Compute.

## Roofline Model Concept

The roofline model plots kernel performance against arithmetic intensity to identify whether a kernel is compute-bound or memory-bound.

- **X-axis:** Arithmetic Intensity (FLOP/byte) — ratio of compute operations to memory bytes accessed
- **Y-axis:** Performance (FLOP/s or operations/s)
- **Roofline:** Two ceilings that form an inverted "V" shape:
  - **Memory roof (sloped):** Peak bandwidth limits performance at low arithmetic intensity
  - **Compute roof (flat):** Peak compute limits performance at high arithmetic intensity
  - **Ridge point:** Where the two roofs meet; the minimum arithmetic intensity to be compute-bound

## Roofline Sections

### SpeedOfLight_RooflineChart

**Command:** `ncu --section SpeedOfLight_RooflineChart ...`

Overview roofline showing kernel position relative to peak memory and compute roofs. Best for quick visual classification.

### Hierarchical Roofline Charts

Show multiple memory roofs at different cache levels (L1, L2, DRAM), revealing where in the memory hierarchy the bottleneck occurs.

| Section | Precision | Command |
|---------|-----------|---------|
| `SpeedOfLight_HierarchicalSingleRooflineChart` | FP32 | `--section SpeedOfLight_HierarchicalSingleRooflineChart` |
| `SpeedOfLight_HierarchicalHalfRooflineChart` | FP16 | `--section SpeedOfLight_HierarchicalHalfRooflineChart` |
| `SpeedOfLight_HierarchicalTensorRooflineChart` | Tensor Core | `--section SpeedOfLight_HierarchicalTensorRooflineChart` |

Use the precision-specific chart matching your kernel's dominant operation type.

## Interpreting Roofline Position

### Position Relative to Roofs

| Position | Meaning | Action |
|----------|---------|--------|
| Below memory roof (left side) | Memory-bound, not reaching bandwidth limit | Improve coalescing, cache usage |
| On memory roof (left side) | Memory-bound, bandwidth saturated | Increase arithmetic intensity (fusion, less data movement) |
| Below compute roof (right side) | Compute-bound, not reaching peak compute | Enable tensor cores, reduce divergence |
| On compute roof (right side) | Compute-bound, near peak compute | Algorithmic improvement needed |
| Far below both roofs | Latency-bound or other issue | Check occupancy, launch overhead |

### Hierarchical Roofline Interpretation

When a kernel is below the DRAM roof but close to the L2 roof, the bottleneck is L2 bandwidth. When close to L1 roof, L1 is the bottleneck. This guides where to optimize:

| Bottleneck Level | Optimization |
|------------------|-------------|
| DRAM bandwidth | Reduce global memory traffic, increase data reuse |
| L2 bandwidth | Improve L1 hit rate, reduce working set |
| L1 bandwidth | Improve shared memory usage, reduce bank conflicts |

## Arithmetic Intensity

**Definition:** FLOP/byte = total floating-point operations / total bytes transferred from memory.

**Low AI (<1):** Memory-bound. Performance limited by how fast data can be moved.
**High AI (>10):** Compute-bound. Performance limited by how fast operations can execute.
**Ridge point AI:** Where memory and compute roofs meet. Depends on GPU architecture.

### Improving Arithmetic Intensity

1. **Kernel fusion:** Combine operations that share data
2. **Tiling:** Compute on cached data blocks instead of streaming
3. **Data reuse:** Ensure each loaded byte is used multiple times
4. **Precision reduction:** Fewer bytes per element = higher AI at same FLOP count

## Using Roofline for Optimization Guidance

### Step 1: Identify Position
Run the appropriate roofline section and locate the kernel dot on the chart.

### Step 2: Determine Bound
- Left of ridge point → memory-bound
- Right of ridge point → compute-bound
- Far below both roofs → other bottleneck (occupancy, latency)

### Step 3: Set Target
- Memory-bound kernel: move right (increase AI) or up (improve bandwidth utilization)
- Compute-bound kernel: move up (improve compute utilization)
- The roof itself is the theoretical maximum; closing the gap to the roof is the goal

### Step 4: Track Progress
Re-profile after optimization. The kernel dot should move up (better performance) and/or right (better AI).

## Roofline with Profile Series

Use profile series to run the same kernel with different configurations and compare their roofline positions:

```bash
ncu --section SpeedOfLight_RooflineChart \
    --kernel-name regex:"my_kernel" \
    --launch-count 5 \
    -- python sweep.py
```

This plots multiple dots, showing how different launch configs or input sizes affect the compute/memory balance.
