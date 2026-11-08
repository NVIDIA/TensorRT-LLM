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

# Preparing Applications for Profiling

How to annotate and scope profiling for DL workloads.

## Focused Profiling — Why and When

By default, `nsys profile` captures the entire application run. For DL training
this is wasteful — initialization, data loading, and warmup dominate. Focus
collection on steady-state iterations to get actionable data.

### Three Approaches

| Method | Best for | Control granularity | Preferred |
|--------|----------|-------------------|-----------|
| `cudaProfilerStart/Stop` | Code-controlled regions | Fine | **Yes** |
| NVTX capture range | Named region triggering | Fine, reusable | |
| `--delay` / `--duration` | Quick time-based windowing | Coarse | |

**Always prefer `cudaProfilerApi`** — it gives exact control over which
iterations are captured, cleanly excludes warmup, JIT compilation, and
initialization overhead.

## Method 1 (Preferred): cudaProfilerStart / cudaProfilerStop

Insert profiler markers in your training loop:

### PyTorch

```python
import torch

# Warmup phase — profiler is off
for epoch in range(num_epochs):
    for i, batch in enumerate(dataloader):
        if i == 5:
            torch.cuda.cudart().cudaProfilerStart()

        # Training step
        loss = model(batch)
        loss.backward()
        optimizer.step()

        if i == 10:
            torch.cuda.cudart().cudaProfilerStop()
            break
```

### Collect with capture range

```bash
nsys profile -c cudaProfilerApi \
    --capture-range-end=stop-shutdown \
    -t cuda,nvtx -- python train.py
```

### C/C++ CUDA

```c
#include <cuda_profiler_api.h>

// In training loop
cudaProfilerStart();
// ... profiled region ...
cudaProfilerStop();
```

## Method 2: NVTX-Triggered Capture

Annotate code with NVTX ranges, then capture only matching ranges.

### Add NVTX Ranges in Python

```python
import torch.cuda.nvtx as nvtx

for epoch in range(num_epochs):
    for i, batch in enumerate(dataloader):
        nvtx.range_push(f"iteration_{i}")

        nvtx.range_push("forward")
        output = model(batch)
        nvtx.range_pop()

        nvtx.range_push("backward")
        loss.backward()
        nvtx.range_pop()

        nvtx.range_push("optimizer")
        optimizer.step()
        nvtx.range_pop()

        nvtx.range_pop()  # iteration
```

### Collect specific NVTX range

```bash
# Profile only the "iteration_5" range
nsys profile -c nvtx -p "iteration_5" \
    --capture-range-end=stop -- python train.py

# Profile iterations 5-9 (repeat 5 times)
nsys profile -c nvtx -p "iteration_5" \
    --capture-range-end=repeat:5 -- python train.py

# Profile by domain
nsys profile -c nvtx -p "training@my_domain" -- python train.py
```

## Method 3: Time-Based Windowing (Fallback)

Simple but imprecise — use only when you cannot modify the application code.

```bash
# Skip first 60s (warmup), profile for 30s
nsys profile -y 60 -d 30 -t cuda,nvtx -- python train.py
```

You must estimate when steady-state begins. Prefer Method 1 instead.

## NVTX Annotations — Markers and Ranges

### Push/Pop Ranges (Nested)

```python
import torch.cuda.nvtx as nvtx

# Simple range
nvtx.range_push("data_loading")
batch = next(dataloader)
nvtx.range_pop()

# Context manager
with nvtx.range("forward_pass"):
    output = model(input)
```

### Named Ranges with Colors

```python
import nvtx

# Using the nvtx Python package (pip install nvtx)
@nvtx.annotate("train_step", color="blue")
def train_step(model, batch):
    output = model(batch)
    loss = criterion(output, batch.labels)
    loss.backward()
    return loss

# Context manager with domain
with nvtx.annotate("data_load", domain="dataloader"):
    batch = next(loader)
```

### Resource Naming

Name threads, CUDA streams, and devices for clearer timeline view:

```python
import ctypes
libcudart = ctypes.CDLL("libcudart.so")

# Name a CUDA stream
nvtx.name_cuda_stream(stream, "compute_stream")
```

## PyTorch Automatic Annotations

Skip manual NVTX — let nsys annotate PyTorch ops automatically:

```bash
# Autograd op names as NVTX ranges
nsys profile --pytorch=autograd-nvtx -t cuda,nvtx -- python train.py

# Autograd ops with tensor shape info
nsys profile --pytorch=autograd-shapes-nvtx -t cuda,nvtx -- python train.py
```

This inserts NVTX ranges around every autograd operation (forward + backward).

## DL Profiling Patterns

### Pattern: Profile Steady-State Training Iterations (Preferred)

```python
# In training script — add profiler markers to skip warmup
for i, batch in enumerate(train_loader):
    if i == warmup_iters:
        torch.cuda.cudart().cudaProfilerStart()
    train_step(model, batch)
    if i == warmup_iters + profile_iters:
        torch.cuda.cudart().cudaProfilerStop()
        break
```

```bash
nsys profile -c cudaProfilerApi \
    -t cuda,nvtx,cudnn,cublas \
    --pytorch=autograd-nvtx \
    -o train_profile -- python train.py
```

### Pattern: Distributed Training (Multi-GPU)

```bash
nsys profile -t cuda,nvtx,mpi,ucx \
    --pytorch=autograd-nvtx \
    -o profile_%q{RANK} \
    -- torchrun --nproc_per_node=8 train.py
```

### Pattern: CUDA Graph Profiling

```bash
# Trace individual graph nodes
nsys profile --cuda-graph-trace=node \
    -t cuda,nvtx -- python train_with_graphs.py
```
