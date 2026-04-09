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

# Expert System Analysis

Automated rule-based detection of known GPU performance anti-patterns.

## Running Expert Analysis

```bash
# Run all rules
nsys analyze -r all report.nsys-rep

# Run specific rules
nsys analyze -r cuda_memcpy_sync,gpu_gaps report.nsys-rep

# List available rules with descriptions
nsys analyze --help-rules ALL

# CSV output
nsys analyze -r all --format csv report.nsys-rep
```

## CUDA Synchronous Operation Rules

These rules detect host-blocking CUDA operations that should be asynchronous.

### cuda_memcpy_async — Async Memcpy with Pageable Memory

**Detects**: `cudaMemcpyAsync` calls that use pageable (non-pinned) memory,
which silently becomes synchronous.

**Impact**: Host thread blocks until transfer completes, serializing CPU/GPU work.

**Fix**: Use pinned (page-locked) memory:
```python
# PyTorch: pin_memory in DataLoader
loader = DataLoader(dataset, pin_memory=True)

# PyTorch: explicit pinning
tensor = tensor.pin_memory()

# CUDA C: cudaMallocHost or cudaHostAlloc
cudaMallocHost(&ptr, size);
```

### cuda_memcpy_sync — Synchronous Memory Transfer

**Detects**: `cudaMemcpy` calls (synchronous by definition) blocking the host.

**Impact**: Host cannot submit new work while transfer runs.

**Fix**: Replace with async variants:
```python
# PyTorch: non_blocking transfers
tensor_gpu = tensor_cpu.to(device, non_blocking=True)

# CUDA C
cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
```

### cuda_memset_sync — Synchronous Memset

**Detects**: `cudaMemset` calls blocking the host.

**Fix**: Use `cudaMemsetAsync` on a stream.

### cuda_api_sync — Synchronization API Calls

**Detects**: Calls that block the host until GPU work completes:
- `cudaDeviceSynchronize`
- `cudaStreamSynchronize`
- `cudaEventSynchronize`

**Impact**: Host idles while waiting. Over-synchronization serializes the pipeline.

**Fix**:
- Minimize sync calls — sync once per iteration, not per kernel
- Use `cudaStreamWaitEvent` for stream-to-stream dependencies
- Use `cudaEventQuery` for non-blocking checks
- Prefer stream-ordered operations and callbacks

## GPU Utilization Rules

### gpu_gaps — GPU Idle Periods

**Detects**: Periods where a GPU has no active kernels or memory operations.
Default threshold: 500ms (configurable).

**Impact**: GPU sitting idle = wasted compute. Common in DL when:
- Data loading is too slow (CPU-bound pipeline)
- Excessive synchronization between steps
- Host-side computation between kernel launches
- Communication blocking compute in distributed training

**Investigation approach**:
1. Check timeline around the gap
2. Look at CPU sampling data — what is the host doing?
3. Check OS runtime (osrt) calls — is it blocked on I/O?
4. Check for sync calls just before the gap

### gpu_time_util — Low GPU Utilization Regions

**Detects**: Time regions where GPU utilization % is below threshold. Divides
analysis timespan into equal chunks and measures kernel activity per chunk.

**Note**: This measures **time utilization** (fraction of time with active
kernels), not **resource utilization** (SM occupancy).

**Impact**: Regions with sparse kernel activity indicate pipeline stalls.

**Investigation approach**:
1. Identify low-utilization chunks on the timeline
2. Check if kernels are short with long gaps between them
3. Look for CPU-side bottlenecks causing submission delays
4. Consider kernel fusion or batching to increase density

## Interpreting Expert System Output

Each rule produces results with:

| Field | Meaning |
|-------|---------|
| Rule Name | Which anti-pattern was detected |
| Description | What the rule checks |
| Advice | Recommended fix |
| Results table | Specific instances found |

### DL-Specific Interpretation

**Common findings in DL workloads:**

| Finding | Likely cause | Fix |
|---------|-------------|-----|
| Many `cuda_memcpy_sync` | `DataLoader` without `pin_memory` | `pin_memory=True` |
| `cuda_memcpy_async` flagged | Pageable host tensors | Pin memory before async transfer |
| Frequent `cudaDeviceSynchronize` | Eager mode sync per op | Use `torch.cuda.synchronize()` sparingly |
| Large GPU gaps | Slow data pipeline | Increase `num_workers`, use prefetching |
| Low GPU utilization | Small batches or frequent sync | Increase batch size, reduce sync points |

## Combining with Stats Reports

```bash
# Full analysis pipeline
nsys analyze -r all report.nsys-rep        # Find anti-patterns
nsys stats -r cuda_api_sum report.nsys-rep  # Quantify API time
nsys stats -r cuda_kern_exec_sum report.nsys-rep  # Check launch overhead
```
