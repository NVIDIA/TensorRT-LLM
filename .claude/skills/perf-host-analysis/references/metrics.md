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

# Metric Definitions

All metrics are derived from an nsys trace exported to SQLite. Core metrics (M1, M2, M4, M5) are extracted via direct SQL queries. M3 (host prep overlap) requires Python-based range intersection.

## Metric 1: GPU Idle Ratio

**Definition**: The fraction of the analysis window where no GPU kernel is executing.

```
gpu_idle_ratio = gpu_idle_time_us / total_time_us
```

Where:
- `total_time_us` = `MAX(kernel.end) - MIN(kernel.start)` across all rows in `CUPTI_ACTIVITY_KIND_KERNEL` (the analysis window spanning first kernel start to last kernel end)
- `gpu_active_time_us` = sum of durations of **merged** GPU kernel ranges (overlapping kernel intervals are unioned into non-overlapping ranges to avoid double-counting concurrent kernels on different SMs)
- `gpu_idle_time_us` = `total_time_us - gpu_active_time_us` (the gaps between merged GPU active ranges)

**Source**: Computed from `CUPTI_ACTIVITY_KIND_KERNEL` table — merge overlapping kernel intervals, then subtract from analysis window. See SKILL.md Detection Step 2 for SQL.

**Threshold**: `gpu_idle_ratio > GPU_IDLE_RATIO_THRESHOLD`

## Metric 2: CUDA Launch API Overhead Ratio

**Definition**: The fraction of total wall-clock time spent inside `cudaLaunchKernel` on the host/CPU side.

```
launch_overhead_ratio = cudaLaunchKernel_total_us / total_time_us
```

Where:
- `cudaLaunchKernel_total_us` = `SUM(end - start)` for all rows in `CUPTI_ACTIVITY_KIND_RUNTIME` where `nameId` resolves to `"cudaLaunchKernel"`
- `total_time_us` = same analysis window as Metric 1

**Source**: Extracted from nsys SQLite via `CUPTI_ACTIVITY_KIND_RUNTIME` table

**SQL extraction**:

```sql
SELECT SUM(end - start) / 1000.0 AS cudaLaunchKernel_total_us
FROM CUPTI_ACTIVITY_KIND_RUNTIME
WHERE nameId IN (
    SELECT id FROM StringIds WHERE value = 'cudaLaunchKernel'
);
```

**Threshold**: `launch_overhead_ratio > LAUNCH_OVERHEAD_RATIO_THRESHOLD`

## Metric 3: Host-Side Preparation Overhead

Three sub-metrics characterize host prep overhead from different angles. All share the same base measurements (refer to schedule diagrams in SKILL.md):

**Base measurements**:
- `host_prep_total_us` = total duration of all NVTX ranges matching the host preparation marker in the `TensorRT-LLM` domain
- `host_prep_exposed_us` = portion of those NVTX ranges where **no GPU kernel is executing** (computed by intersecting each NVTX range with the merged GPU idle gaps)
- If `host_prep_total_us == 0` (no matching NVTX ranges found), all three sub-metrics are **not applicable** and excluded from the verdict

**Configurable NVTX range name**: The host-prep marker defaults to `_prepare_tp_inputs` but may differ across TRT-LLM versions. Common alternatives: `_prepare_inputs`, `prepare_inputs`. Check available NVTX ranges with:
```sql
SELECT DISTINCT s.value, COUNT(*)
FROM NVTX_EVENTS n JOIN StringIds s ON n.textId = s.id
WHERE n.end > 0 AND s.value LIKE '%prepare%input%'
GROUP BY s.value ORDER BY COUNT(*) DESC;
```

**Source**: Computed in Python by intersecting NVTX ranges with merged GPU idle gaps (see SKILL.md Detection Step 4). M3 is optional — the core metrics M1/M2/M4/M5 are usually sufficient for the Detection verdict.

### Metric 3a: Pipeline Efficiency (exposed ratio)

```
host_prep_exposed_ratio = host_prep_exposed_us / host_prep_total_us
```

**Answers**: "What fraction of host prep time is exposed (GPU idle) vs hidden (overlapping GPU execution)?"

A ratio of 0.0 means host prep is fully hidden behind GPU work (ideal pipelining, Scenario A). A ratio of 1.0 means the GPU is idle during all host prep (no overlap at all, Scenario B).

**Threshold**: `host_prep_exposed_ratio > HOST_PREP_EXPOSED_RATIO_THRESHOLD`

**Limitation**: Does not tell you the magnitude of the performance impact. A workload with 100 us total host prep and 60% exposed ratio has only 60 us of waste — negligible in a 10 ms iteration.

### Metric 3b: Performance Impact (exposed as fraction of wall time)

```
host_prep_perf_impact = host_prep_exposed_us / total_time_us
```

**Answers**: "How much of the total wall-clock time is wasted on exposed host prep?"

This directly measures the **throughput cost** of host overhead. If `host_prep_perf_impact = 0.15`, then making host prep instant would improve throughput by ~15%. This is the primary metric for deciding **whether host overhead is worth optimizing**.

```
Scenario C, annotated with performance impact:

time ──────────────────────────────────────────────────────────────→

Host: |prep N|launch|post|======== prep N+1 ========|launch|post|
GPU:         |======== kernels N ========|   ↑IDLE↑  |kernels N+1|
                                             ^^^^^^^
|←──────────────────── total_time_us ──────────────────────────→|
                                             |←───→|
                                             host_prep_perf_impact
                                             = this gap / total_time
```

**Threshold**: `host_prep_perf_impact > HOST_PREP_PERF_IMPACT_THRESHOLD`

### Metric 3c: GPU Idle Attribution (root cause)

```
host_prep_idle_attribution = host_prep_exposed_us / gpu_idle_time_us
```

**Answers**: "Of all the time the GPU is idle, how much is caused by host prep?"

GPU idle time can have multiple causes: host prep, NCCL waits, cudaStreamSynchronize, memory allocation, Python GIL contention, etc. This metric isolates host prep's contribution. A high attribution (>0.50) confirms host prep is the **dominant cause** of GPU idle time, not communication or other stalls.

```
Example: GPU idle = 1000 us total
    host_prep_exposed = 700 us  → attribution = 70% (host prep is main cause)
    remaining 300 us            → other causes (NCCL, sync, alloc, etc.)
```

**Threshold**: `host_prep_idle_attribution > HOST_PREP_IDLE_ATTRIBUTION_THRESHOLD`

### How the Three Sub-Metrics Work Together

| Scenario | 3a (pipeline) | 3b (perf impact) | 3c (attribution) | Interpretation |
|----------|---------------|-------------------|-------------------|----------------|
| Heavy host prep, fully exposed | 0.95 | 0.20 | 0.85 | Host prep is the bottleneck — large, exposed, dominant |
| Heavy host prep, well hidden | 0.10 | 0.01 | 0.05 | Host prep is slow but GPU hides it — not a problem |
| Light host prep, fully exposed | 0.90 | 0.02 | 0.15 | Host prep is exposed but tiny — other idle causes dominate |
| Heavy host prep, GPU idle from NCCL | 0.60 | 0.08 | 0.25 | Host prep is partially exposed but NCCL is the bigger idle cause |

**Decision**: Host prep is a confirmed bottleneck when **both** 3b (performance impact) and 3c (attribution) cross their thresholds — the exposed overhead is large enough to matter AND it is the main cause of GPU idle time.

## Metric 4: GPU Utilization (Time-Based)

**Definition**: The fraction of the analysis window where at least one GPU kernel is executing.

```
gpu_utilization = gpu_active_time_us / total_time_us
```

This is the complement of Metric 1 (`gpu_utilization = 1 - gpu_idle_ratio`), reported separately because it is the most intuitive indicator. This is **time-based utilization** (is any kernel running?), not SM occupancy or compute throughput.

**Source**: Complement of M1: `gpu_utilization = 1 - gpu_idle_ratio`. Computed from same merged kernel ranges.

**Threshold**: `gpu_utilization < GPU_UTILIZATION_THRESHOLD`

## Metric 5 (Caveat): NCCL Communication Ratio

**Definition**: The fraction of total GPU active time spent in NCCL collective operations.

```
nccl_ratio = nccl_kernel_total_us / gpu_active_time_us
```

Where:
- `nccl_kernel_total_us` = `SUM(end - start)` for kernel rows in `CUPTI_ACTIVITY_KIND_KERNEL` where `shortName` contains `"nccl"` (case-insensitive)

**Purpose**: This is NOT a host-overhead indicator. It is a **caveat metric**. If NCCL dominates, GPU idle time may be caused by communication stalls rather than host overhead. The verdict should note this caveat when NCCL ratio is high.

**SQL extraction** (note: `shortName` is an integer ID — must join with `StringIds`):

```sql
SELECT SUM(k.end - k.start) / 1000.0 AS nccl_kernel_total_us
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.shortName = s.id
WHERE s.value LIKE '%nccl%';
```

**Threshold**: `nccl_ratio > NCCL_RATIO_CAVEAT_THRESHOLD`
