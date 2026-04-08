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

# Output Format

## Report Template

```
## Host Overhead Verdict: YES / NO

### Aggregate Evidence

| Metric                            | Value  | Threshold | Crossed? |
|-----------------------------------|--------|-----------|----------|
| GPU idle ratio                    | 42.1%  | >30%      | YES      |
| cudaLaunchKernel overhead ratio   | 3.2%   | >10%      | NO       |
| Host prep exposed ratio (3a)      | 55.0%  | >50%      | YES      |
| Host prep perf impact (3b)        | 12.5%  | >5%       | YES      |
| Host prep idle attribution (3c)   | 72.0%  | >50%      | YES      |
| GPU utilization (time-based)      | 57.9%  | <60%      | YES      |

Metrics crossed: 5 / 6 applicable
Host prep confirmed bottleneck: YES (3b AND 3c both crossed)

### Per-Phase Breakdown

#### Context Phase (N iterations)

| Metric                         | Value  | Threshold | Crossed? |
|--------------------------------|--------|-----------|----------|
| GPU idle ratio                 | 48.2%  | >30%      | YES      |
| GPU utilization                | 51.8%  | <60%      | YES      |
| cudaLaunchKernel ratio         | 8.5%   | >10%      | NO       |
| Host prep exposed ratio (3a)   | 62.0%  | >50%      | YES      |
| Host prep perf impact (3b)     | 18.0%  | >5%       | YES      |
| Host prep idle attribution (3c)| 80.0%  | >50%      | YES      |

Phase verdict: YES (5/6 crossed)
Host prep confirmed bottleneck: YES (3b AND 3c both crossed)

#### Generation Phase (M iterations, CUDA graphs: enabled/disabled)

| Metric                         | Value  | Threshold | Crossed? |
|--------------------------------|--------|-----------|----------|
| GPU idle ratio                 | 5.1%   | >15%      | NO       |
| GPU utilization                | 94.9%  | <80%      | NO       |
| cudaLaunchKernel ratio         | 0.8%   | >10%      | NO       |
| Host prep exposed ratio (3a)   | N/A    | >50%      | N/A      |
| Host prep perf impact (3b)     | N/A    | >5%       | N/A      |
| Host prep idle attribution (3c)| N/A    | >50%      | N/A      |

Phase verdict: NO (0/3 crossed)

### Interpretation

Host overhead is concentrated in **context phase** iterations.
Generation phase is healthy — CUDA graphs are effectively hiding host work.
Optimization should focus on context-phase host preparation.

### Caveat (if NCCL ratio > threshold)

> NCCL communication accounts for Z% of GPU active time.
> GPU idle gaps may be partially caused by communication stalls,
> not solely host overhead.

### Next Steps

**If YES** → Use `host-perf-optimization` skill to profile and optimize.
Start with line_profiler on `_prepare_tp_inputs`.

**If NO** → Host overhead is not the bottleneck.
Use `perf-nsight-compute-analysis` for kernel-level SOL% analysis.
Use `trace-interpretation` for full bottleneck classification.
```

## Integration JSON

When called from `perf-analysis`, return structured data:

```json
{
  "verdict": "YES",
  "aggregate_metrics": {
    "gpu_idle_ratio": 0.421,
    "launch_overhead_ratio": 0.032,
    "host_prep_exposed_ratio": 0.550,
    "host_prep_perf_impact": 0.125,
    "host_prep_idle_attribution": 0.720,
    "gpu_utilization": 0.579,
    "nccl_ratio": 0.05
  },
  "crossed_count": 5,
  "applicable_count": 6,
  "host_prep_confirmed": true,
  "nccl_caveat": false,
  "phases": {
    "context": {
      "iteration_count": 5,
      "verdict": "YES",
      "metrics": {
        "gpu_idle_ratio": 0.482,
        "gpu_utilization": 0.518,
        "launch_overhead_ratio": 0.085,
        "host_prep_exposed_ratio": 0.620,
        "host_prep_perf_impact": 0.180,
        "host_prep_idle_attribution": 0.800
      },
      "host_prep_confirmed": true,
      "crossed_count": 5,
      "applicable_count": 6
    },
    "generation": {
      "iteration_count": 95,
      "verdict": "NO",
      "cuda_graphs_detected": true,
      "metrics": {
        "gpu_idle_ratio": 0.051,
        "gpu_utilization": 0.949,
        "launch_overhead_ratio": 0.008,
        "host_prep_exposed_ratio": null,
        "host_prep_perf_impact": null,
        "host_prep_idle_attribution": null
      },
      "host_prep_confirmed": false,
      "crossed_count": 0,
      "applicable_count": 3
    }
  }
}
```

## Handoff to Optimization

When handing off to `perf-host-optimization`, append an `optimization_handoff` block to the Integration JSON. This provides the optimization skill with triage data so it can prioritize without re-running analysis.

```json
{
  "optimization_handoff": {
    "verdict": "YES",
    "primary_bottleneck_phase": "context",
    "host_prep_confirmed": true,
    "top_regressing_ops": [
      {"nvtx_name": "_update_requests", "delta_us_per_step": 310, "source_function": "PyExecutor._update_requests"},
      {"nvtx_name": "broadcast_requests", "delta_us_per_step": 250, "source_function": "PyExecutor.broadcast_requests"},
      {"nvtx_name": "_fetch_new_requests", "delta_us_per_step": 234, "source_function": "PyExecutor._fetch_new_requests"}
    ],
    "recommended_first_target": "_prepare_tp_inputs",
    "wall_time_per_step_us": 3977.7,
    "inter_step_gap_p50_us": 4467.6,
    "kernels_per_step": 21.9
  }
}
```

**Field descriptions:**
- `top_regressing_ops`: NVTX operations sorted by absolute regression (us/step), from Root Cause comparison. Only present when two traces are compared.
- `recommended_first_target`: The function to profile first with line_profiler. Defaults to `_prepare_tp_inputs` when host_prep_confirmed is true.
- `wall_time_per_step_us` / `inter_step_gap_p50_us`: Baseline metrics for the optimization skill's stopping criteria.
