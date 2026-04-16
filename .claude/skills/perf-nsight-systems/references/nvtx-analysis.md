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

# NVTX Statistical Reports

Reference for NVTX-related statistical reports in `nsys stats`.

## Running NVTX Reports

```bash
# All NVTX reports
nsys stats -r nvtx_sum,nvtx_pushpop_sum,nvtx_kern_sum,nvtx_gpu_proj_sum report.nsys-rep

# Specific report
nsys stats -r nvtx_pushpop_sum report.nsys-rep

# CSV output
nsys stats -r nvtx_gpu_proj_sum --format csv report.nsys-rep
```

## nvtx_sum — Combined NVTX Range Summary

Summary of all NVTX ranges (both Push/Pop and Start/End styles).

| Column | Meaning |
|--------|---------|
| Time(%) | % of total NVTX time |
| Total Time | Cumulative range duration |
| Instances | Number of range entries |
| Avg / Med / Min / Max / StdDev | Duration statistics |
| Style | `PushPop` or `StartEnd` |
| Range | Range name string |

**DL use case**: Identify which annotated training phases (forward, backward,
optimizer, data loading) consume the most wall-clock time.

## nvtx_pushpop_sum — Push/Pop Range Summary

Summary of nested Push/Pop ranges only (the most common style in DL code).

| Column | Meaning |
|--------|---------|
| Time(%) | % of total Push/Pop range time |
| Total Time | Cumulative duration |
| Instances | Count |
| Avg / Med / Min / Max / StdDev | Duration statistics |
| Range | Range name |

**DL use case**: With `--pytorch=autograd-nvtx`, each autograd op gets a Push/Pop
range. This report shows which ops dominate.

## nvtx_pushpop_trace — Push/Pop Range Trace

Chronological trace of individual Push/Pop range instances.

| Column | Meaning |
|--------|---------|
| Start / End / Duration | Timing |
| Child Duration | Time in child ranges |
| Level | Nesting depth |
| Parent | Parent range name |
| Name Tree | Full nesting path with level markers |
| Process / Thread | Caller identity |

**DL use case**: Trace individual iteration timing. Check if certain iterations
are slower (outliers in the trace).

## nvtx_startend_sum — Start/End Range Summary

Same as pushpop_sum but for Start/End ranges (less common in DL).

## nvtx_kern_sum — NVTX Range Kernel Summary

CUDA kernels grouped by their enclosing NVTX range. Maps GPU work back to
annotated code regions.

| Column | Meaning |
|--------|---------|
| NVTX Range | Enclosing range name |
| Style | PushPop or StartEnd |
| Range Instances | How many times the range executed |
| Kernel Instances | Total kernel launches within |
| Avg / Med / Min / Max / StdDev | Kernel duration statistics |
| Kernel Name | CUDA kernel function name |
| Process / Thread | Host thread that launched |

**DL use case**: See which training phase (forward, backward) launches which
kernels and how much GPU time each consumes.

```bash
# Which kernels run during each NVTX-annotated phase?
nsys stats -r nvtx_kern_sum report.nsys-rep
```

## nvtx_gpu_proj_sum — GPU Projection Summary

Projects NVTX CPU ranges to GPU execution timeline. Shows total GPU time
attributable to each annotated region.

| Column | Meaning |
|--------|---------|
| Projected Duration | GPU time within this NVTX range |
| Original Duration | CPU-side range duration |
| GPU Op Count | Number of GPU operations launched |
| Avg / Med / Min / Max / StdDev | Projected duration stats |
| Range | NVTX range name |
| Style | PushPop or StartEnd |
| Level | Nesting depth |

**DL use case**: "How much GPU time does the forward pass consume vs backward
pass?" This report answers that directly by projecting CPU annotations to GPU
activity.

```bash
nsys stats -r nvtx_gpu_proj_sum report.nsys-rep
```

## nvtx_gpu_proj_trace — GPU Projection Trace

Per-instance projection records with timestamps.

| Column | Meaning |
|--------|---------|
| Projected Start / End / Duration | GPU-side timing |
| Original Start / End / Duration | CPU-side timing |
| GPU Op Count | Operations in this instance |
| Level | Nesting depth |
| Parent | Parent range |
| Range Stack | Full range hierarchy |
| Process / Thread | Identity |

**DL use case**: Track per-iteration GPU projection to find iteration-to-iteration
variance.

## Common NVTX Analysis Patterns

### Pattern: Training Phase Breakdown

```bash
# Profile with PyTorch auto-annotations
nsys profile --pytorch=autograd-nvtx -t cuda,nvtx -o train -- python train.py

# See which phases dominate CPU time
nsys stats -r nvtx_pushpop_sum train.nsys-rep

# See which phases dominate GPU time
nsys stats -r nvtx_gpu_proj_sum train.nsys-rep

# See kernel-to-phase mapping
nsys stats -r nvtx_kern_sum train.nsys-rep
```

### Pattern: Iteration Time Variability

```bash
# With manual NVTX iteration markers
nsys stats -r nvtx_pushpop_trace --format csv train.nsys-rep | \
    grep "iteration" | sort -t, -k3 -n -r | head -20
```

### Pattern: GPU Projection Efficiency

Compare original (CPU) duration to projected (GPU) duration. Large difference
means the GPU is idle during part of the annotated region:

```bash
nsys stats -r nvtx_gpu_proj_sum --format csv train.nsys-rep
# Check: if Projected Duration << Original Duration → GPU idle within range
```
