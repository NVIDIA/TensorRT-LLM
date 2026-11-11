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

# Nsight Compute Metrics Guide

GPU hardware model, metric naming conventions, and key metrics for performance analysis.

## GPU Hardware Model

### Compute Model

- **Grid**: Collection of thread blocks launched by a kernel
- **Block (CTA)**: Group of threads executing on one SM, sharing shared memory
- **Warp**: 32 threads executing in lockstep (the scheduling unit)
- **Thread**: Individual execution unit

### Streaming Multiprocessor (SM)

Each SM contains 4 sub-partitions (SMSPs), each with:
- Warp scheduler + dispatch unit
- Register file
- Execution units: integer (ALU), floating-point (FMA), load/store (LSU), special function (XU)
- Tensor cores

Shared across the SM: the unified L1 data cache / shared memory and the texture units. On
Blackwell, 5th-generation MMA (`tcgen05`) is issued by a single thread to the tensor-core
pipeline (`tc`), which drives all four sub-partitions and writes the accumulator to Tensor
Memory (TMEM) instead of registers.

### Memory Hierarchy

| Level | Location | Scope | Key Property |
|-------|----------|-------|--------------|
| Registers | SM | Per-thread | Fastest, limited per SM |
| Shared Memory | SM (on-chip) | Per-block | 32 banks, user-managed |
| L1/TEX Cache | SM (on-chip) | Per-SM | Unified with shared memory |
| L2 Cache | Global | All SMs | Between L1 and DRAM |
| Device Memory (DRAM) | Off-chip | Global | Highest capacity, highest latency |
| Local Memory | Off-chip | Per-thread | Register spill, in device memory |

## Metric Naming Convention

Pattern: `unit__(subunit?)_(pipestage?)_quantity_(qualifiers?)`

### Units (GPU Components)

| Unit | Component |
|------|-----------|
| `sm` | Streaming Multiprocessor |
| `smsp` | SM Sub-Partition |
| `l1tex` | L1/Texture Cache |
| `lts` / `ltc` | L2 Cache |
| `dram` | Device Memory |
| `fbpa` | Framebuffer Partition |
| `gpc` | General Processing Cluster |
| `tpc` | Thread Processing Cluster |
| `gpu` | Whole GPU |
| `ctc` | NVLink Chip-to-Chip (C2C) link to a Grace CPU |

### Pipelines (Execution Units)

| Pipeline | Purpose |
|----------|---------|
| `alu` | Integer arithmetic |
| `fma` | Fused multiply-add (FP32) |
| `fp64` | Double precision |
| `lsu` | Load/store |
| `tensor` | Tensor core operations (MMA, including Blackwell `tcgen05` MMA) |
| `tc` | Blackwell tensor-core pipeline (`UTC*` instructions: MMA, copies, barriers) |
| `tex` | Texture operations |
| `tma` | Tensor memory accelerator |
| `tmem` | Blackwell Tensor Memory loads and stores |
| `xu` | Transcendental/conversion |
| `cbu` | Control/branch |

### Quantities

| Quantity | Meaning |
|----------|---------|
| `inst_executed` | Assembly (SASS) instructions executed |
| `request` | Command sent to a hardware unit |
| `sector` | 32-byte aligned memory chunk |
| `tag` | Cache line identifier |
| `wavefront` | Maximum work package per pipeline stage |
| `cycles_active` | Cycles the unit was active |
| `cycles_elapsed` | Total elapsed cycles |

## Metric Types

### Counters
Raw or calculated values. Roll-up suffixes: `.sum`, `.avg`, `.min`, `.max`.

Example: `smsp__inst_executed.sum` = total instructions across all SMSPs.

### Ratios
Three sub-metrics:
- `.pct` — percentage (0-100)
- `.ratio` — raw ratio
- `.max_rate` — theoretical max

Example: `l1tex__t_sector_hit_rate.pct` = L1 cache hit rate as percentage.

### Throughputs
Peak rate percentages:
- `.pct_of_peak_sustained_active` — % of peak while unit was active
- `.pct_of_peak_sustained_elapsed` — % of peak over total elapsed time

Example: `sm__throughput.avg.pct_of_peak_sustained_elapsed` = compute throughput.

## Key Metrics by Category

### Throughput (SOL% Classification)

| Metric | Meaning | Use |
|--------|---------|-----|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | Compute (SM) Throughput | SOL% classification |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | Memory Throughput (busiest memory unit) | SOL% classification |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | DRAM throughput | DRAM bandwidth use |
| `gpu__time_duration.sum` | Kernel duration (ns) | Launch overhead check |

### Occupancy

| Metric | Meaning | Good Value |
|--------|---------|------------|
| `sm__warps_active.avg.pct_of_peak_sustained_active` | Achieved occupancy | >50% |
| `sm__maximum_warps_per_active_cycle_pct` | Theoretical occupancy | 100% ideal |
| `launch__occupancy_limit_registers` | Blocks per SM that registers allow | Lowest limit sets theoretical |
| `launch__occupancy_limit_shared_mem` | Blocks per SM that shared memory allows | Lowest limit sets theoretical |
| `launch__occupancy_limit_warps` | Blocks per SM that the warp count allows | Lowest limit sets theoretical |
| `launch__occupancy_limit_blocks` | Blocks per SM the hardware allows | Lowest limit sets theoretical |

### Cache Performance

| Metric | Meaning | Good Value |
|--------|---------|------------|
| `l1tex__t_sector_hit_rate.pct` | L1 cache hit rate | >80% |
| `lts__t_sector_hit_rate.pct` | L2 cache hit rate | >80% |
| `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum` | Shared memory ops | Check for bank conflicts |

### Compute Pipeline

| Metric | Meaning |
|--------|---------|
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | Tensor core utilization |
| `smsp__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_elapsed` | FP64 usage |
| `smsp__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed` | FP32 FMA usage |

### Instruction Analysis

| Metric | Meaning | Ideal |
|--------|---------|-------|
| `smsp__inst_executed.sum` | Total instructions | Lower is better |
| `smsp__thread_inst_executed_per_inst_executed.ratio` | Warp divergence | 32 (no divergence) |
| `smsp__inst_executed_pipe_cbu.avg.pct_of_peak_sustained_elapsed` | Control flow overhead | Low |

### Launch Configuration

| Metric | Meaning | Typical |
|--------|---------|---------|
| `launch__grid_size` | Total blocks | >100 for good utilization |
| `launch__block_size` | Threads per block | 128-256 |
| `launch__registers_per_thread` | Registers/thread | <64 |
| `launch__shared_mem_per_block` | Shared memory/block | <48KB |

## Cycle Metrics

Each hardware unit tracks:
- `cycles_elapsed` — total cycles during measurement
- `cycles_active` — cycles the unit was doing work
- `cycles_stalled` — cycles the unit could not take new data because its output was blocked
- `cycles_idle` — cycles the unit was idle

Ratio: `active / elapsed` = utilization percentage.

## Instanced Metrics

Some metrics have multiple instances (e.g., per-instruction source metrics). Access with correlation IDs to map back to source lines.

```bash
ncu --print-metric-instances details app  # Show correlation IDs
```

## Out-of-Range Values

Metric values can occasionally appear out of range due to:
- Asynchronous GPU activity from other processes
- Multi-pass collection variations between passes
- Very short kernels where measurement overhead dominates

Mitigation: profile on isolated GPUs, increase workload size, use `--launch-count` > 1.
