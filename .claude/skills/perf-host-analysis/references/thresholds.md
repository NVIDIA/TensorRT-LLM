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

# Configurable Thresholds

All thresholds are variables. Adjust based on the workload characteristics (batch size, model size, hardware).

**Rationale**: Default thresholds are calibrated from Llama 3.2 1B (TP=2, B200) profiling runs (Feb 2025). They may need adjustment for larger models (where GPU time dominates and host overhead is naturally lower) or different hardware (where CPU speed or PCIe bandwidth changes the host/GPU balance). When in doubt, start with defaults and adjust if the verdict contradicts visual inspection of the nsys timeline.

## Aggregate Thresholds

| Variable | Default | Description |
|----------|---------|-------------|
| `GPU_IDLE_RATIO_THRESHOLD` | 0.30 | GPU idle fraction above which host overhead is suspected |
| `LAUNCH_OVERHEAD_RATIO_THRESHOLD` | 0.10 | cudaLaunchKernel time fraction above which launch overhead is excessive |
| `HOST_PREP_EXPOSED_RATIO_THRESHOLD` | 0.50 | Exposed host prep fraction above which host work is serializing the pipeline (pipeline efficiency) |
| `HOST_PREP_PERF_IMPACT_THRESHOLD` | 0.05 | Exposed host prep as fraction of wall time above which it materially degrades throughput |
| `HOST_PREP_IDLE_ATTRIBUTION_THRESHOLD` | 0.50 | Fraction of GPU idle time caused by host prep above which host prep is the dominant idle cause |
| `GPU_UTILIZATION_THRESHOLD` | 0.60 | GPU time-based utilization below which the GPU is underutilized |
| `NCCL_RATIO_CAVEAT_THRESHOLD` | 0.20 | NCCL fraction of GPU time above which communication is a confounding factor |

## Per-Phase Thresholds

Context and generation iterations have fundamentally different characteristics. Context iterations run eagerly (no CUDA graphs) with variable-length sequences, while generation iterations use CUDA graph replay with fixed batch sizes. Apply tighter thresholds to generation iterations because CUDA graphs should eliminate most host overhead.

| Variable | Context Default | Generation Default | Description |
|----------|----------------|-------------------|-------------|
| `GPU_IDLE_RATIO_THRESHOLD` | 0.30 | 0.15 | Generation with CUDA graphs should have very little idle time |
| `GPU_UTILIZATION_THRESHOLD` | 0.60 | 0.80 | Generation should have high GPU utilization thanks to graph replay |
| `LAUNCH_OVERHEAD_RATIO_THRESHOLD` | 0.10 | 0.10 | Same across phases |
| `HOST_PREP_EXPOSED_RATIO_THRESHOLD` | 0.50 | 0.50 | Same across phases |
| `HOST_PREP_PERF_IMPACT_THRESHOLD` | 0.05 | 0.05 | Same across phases |
| `HOST_PREP_IDLE_ATTRIBUTION_THRESHOLD` | 0.50 | 0.50 | Same across phases |
| `NCCL_RATIO_CAVEAT_THRESHOLD` | 0.20 | 0.20 | Same across phases |
