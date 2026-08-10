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

# Advanced Recipes for Deep Learning

DL-relevant `nsys recipe` commands for multi-file and visual analysis.

## Running Recipes

```bash
# Basic syntax
nsys recipe <recipe-name> [recipe-args] -- <input-files>

# Get help for a specific recipe
nsys recipe <recipe-name> --help

# Multiple input files (multi-node/multi-run)
nsys recipe <recipe-name> -- rank0.nsys-rep rank1.nsys-rep rank2.nsys-rep
```

Recipes produce output directories containing:
- CSV/Parquet data files
- Plotly HTML visualizations (summary graphs, heatmaps, box plots)
- `.nsys-analysis` files openable in Nsight Systems GUI or Jupyter

## CUDA Analysis Recipes

### cuda_api_sum — CUDA API Summary

Summarizes CUDA API functions and execution times across one or more reports.

```bash
nsys recipe cuda_api_sum -- report.nsys-rep
```

### cuda_gpu_kern_sum — GPU Kernel Summary

GPU kernel execution statistics with timing distribution.

```bash
nsys recipe cuda_gpu_kern_sum -- report.nsys-rep
```

### cuda_gpu_kern_hist — Kernel Duration Histogram

Probability distribution of kernel durations. Useful for identifying bimodal
distributions (e.g., warmup kernels vs steady-state).

```bash
nsys recipe cuda_gpu_kern_hist -- report.nsys-rep
```

### cuda_gpu_kern_pace — Kernel Pace Analysis

Tracks kernel launch consistency over time. Detects irregular pacing that
indicates pipeline stalls or scheduling issues.

```bash
nsys recipe cuda_gpu_kern_pace -- report.nsys-rep
```

### cuda_gpu_mem_size_sum / cuda_gpu_mem_time_sum

Memory operation summaries by size and time respectively.

```bash
nsys recipe cuda_gpu_mem_size_sum -- report.nsys-rep
nsys recipe cuda_gpu_mem_time_sum -- report.nsys-rep
```

### cuda_gpu_time_util_map — CUDA Kernel Utilization Heatmap

Heatmap showing CUDA kernel utilization % over time. Reveals phases of high/low
GPU activity across the training run.

```bash
nsys recipe cuda_gpu_time_util_map -- report.nsys-rep
```

### cuda_memcpy_async / cuda_memcpy_sync / cuda_memset_sync / cuda_api_sync

Recipe versions of the expert system rules. Same detection logic but with
recipe output format (data files + visualizations).

```bash
nsys recipe cuda_memcpy_sync -- report.nsys-rep
nsys recipe cuda_api_sync -- report.nsys-rep
```

## GPU Utilization Recipes

### gpu_gaps — GPU Idle Period Analysis

Identifies and visualizes GPU idle periods exceeding a threshold.

```bash
nsys recipe gpu_gaps -- report.nsys-rep
```

### gpu_time_util — GPU Time Utilization

Detects low GPU utilization regions with configurable threshold.

```bash
nsys recipe gpu_time_util -- report.nsys-rep
```

### gpu_metric_util_map — GPU Metric Utilization Heatmap

Heatmap for SM Active, SM Issue, and Tensor Active metrics over time. Requires
GPU metrics collection (`--gpu-metrics-devices`).

```bash
# Collect with GPU metrics
nsys profile --gpu-metrics-devices=all -t cuda,nvtx -o report -- python train.py

# Analyze
nsys recipe gpu_metric_util_map -- report.nsys-rep
```

## NCCL / Distributed Recipes

### nccl_sum — NCCL Function Summary

Summarizes NCCL collective operations (AllReduce, AllGather, etc.) with timing.

```bash
nsys recipe nccl_sum -- rank0.nsys-rep rank1.nsys-rep
```

### nccl_gpu_overlap_trace — Communication/Compute Overlap

Traces overlap between NCCL communication kernels and compute kernels. Key
metric for distributed training efficiency.

```bash
nsys recipe nccl_gpu_overlap_trace -- rank0.nsys-rep rank1.nsys-rep
```

### nccl_gpu_time_util_map — NCCL + Compute Heatmap

Heatmap showing NCCL and compute kernel utilization over time. Visualizes
communication/computation balance per rank.

```bash
nsys recipe nccl_gpu_time_util_map -- rank0.nsys-rep rank1.nsys-rep
```

### nccl_gpu_proj_sum — NCCL GPU Projection Summary

GPU projection of NCCL operations.

```bash
nsys recipe nccl_gpu_proj_sum -- rank0.nsys-rep rank1.nsys-rep
```

## NVTX Recipes

### nvtx_sum — NVTX Range Summary

Combined summary of Push/Pop and Start/End ranges.

```bash
nsys recipe nvtx_sum -- report.nsys-rep
```

### nvtx_gpu_proj_sum / nvtx_gpu_proj_trace — NVTX GPU Projection

Project NVTX CPU ranges to GPU execution. Shows which GPU work belongs to which
annotated code region.

```bash
nsys recipe nvtx_gpu_proj_sum -- report.nsys-rep
nsys recipe nvtx_gpu_proj_trace -- report.nsys-rep
```

### nvtx_gpu_proj_pace — NVTX GPU Projection Pace

Tracks consistency of GPU work projected from NVTX ranges over time.

```bash
nsys recipe nvtx_gpu_proj_pace -- report.nsys-rep
```

### nvtx_pace — NVTX Range Pace

Tracks NVTX range timing consistency. Useful for monitoring iteration time
stability in training loops.

```bash
nsys recipe nvtx_pace -- report.nsys-rep
```

## Network and System Recipes

### mpi_sum — MPI Function Summary

```bash
nsys recipe mpi_sum -- rank0.nsys-rep rank1.nsys-rep
```

### mpi_gpu_time_util_map — MPI + GPU Utilization Heatmap

```bash
nsys recipe mpi_gpu_time_util_map -- rank0.nsys-rep rank1.nsys-rep
```

### nvlink_sum — NVLink Throughput Summary

```bash
nsys recipe nvlink_sum -- report.nsys-rep
```

### network_sum / network_traffic_map — Network Analysis

```bash
nsys recipe network_sum -- report.nsys-rep
nsys recipe network_traffic_map -- report.nsys-rep
```

## Comparison Recipe

### diff — Statistical Comparison

Compare two recipe outputs to find regressions or improvements.

```bash
# First generate two recipe outputs
nsys recipe cuda_gpu_kern_sum -- baseline.nsys-rep
nsys recipe cuda_gpu_kern_sum -- optimized.nsys-rep

# Then compare
nsys recipe diff -- baseline_output optimized_output
```

## Jupyter Notebook Integration

Recipe outputs (`.nsys-analysis` files) can be opened as Jupyter notebooks:

1. Open in Nsight Systems GUI: `File > Open > .nsys-analysis`
2. Click the notebook icon to launch Jupyter
3. Execute cells for interactive Plotly visualizations

## Dask for Large-Scale Analysis

Recipes support distributed computation via Dask:

```bash
# Set Dask scheduler
export NSYS_DASK_SCHEDULER_FILE=/path/to/scheduler.json
nsys recipe cuda_gpu_kern_sum -- many_reports/*.nsys-rep
```

Config via `~/.config/dask/` YAML files or `DASK_*` environment variables.
