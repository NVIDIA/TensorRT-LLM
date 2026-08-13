---
name: perf-torch-sync-free
description: >-
  Identify and eliminate host-device synchronizations in PyTorch code. Detects
  sync points (.item(), .cpu(), boolean indexing, torch.tensor on CUDA),
  classifies false vs true dependencies, provides sync-free alternatives.
  Triggers: sync-free, synchronization, .item(), .cpu(), host-device sync,
  eliminate syncs, CPU stall, non_blocking, set_sync_debug_mode,
  cudaStreamSynchronize, cudaEventSynchronize, remove syncs, async GPU.
tags:
  - synchronization
  - performance
  - pytorch
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Writing Sync-Free PyTorch Code

Sync-free code means the CPU continuously queues work to the GPU without
waiting for GPU operations to complete. When host-device synchronizations
are eliminated, the GPU works continuously without idle stalls.

Every host-device synchronization ultimately calls one of three CUDA driver
APIs that block the CPU thread:

- `cuEventSynchronize` -- CPU waits until a specific GPU event completes
- `cuStreamSynchronize` -- CPU waits until all work on a stream finishes
- `cuCtxSynchronize` -- CPU waits until all work across all streams finishes

## When to Use

Reach for this skill when you encounter:

- **Triggers**: User wants to remove host-device synchronizations, eliminate
  CPU stalls from GPU waits, make code async/sync-free, remove `.item()` or
  `.cpu()` calls that block the CPU, or understand why specific PyTorch
  operations cause synchronization
- **Symptoms**: Frequent `cudaStreamSynchronize` in nsys profiles,
  warnings from `torch.cuda.set_sync_debug_mode`, training throughput
  limited by CPU-GPU round-trips, `.item()` or `.cpu()` calls in hot loops
- **Keywords**: "sync-free", "synchronization", ".item()", ".cpu()",
  "host-device sync", "eliminate syncs", "CPU stall", "non_blocking",
  "set_sync_debug_mode", "cudaStreamSynchronize", "cudaEventSynchronize",
  "remove syncs", "async GPU", "CPU waiting on GPU"

Do NOT use this skill for:

- Applying CUDA Graphs or reducing kernel launch overhead (use
  `perf-torch-cuda-graphs` instead)
- Profiling GPU kernels, system timelines, or finding GPU idle time (use
  `perf-nsight-compute-analysis` or `perf-nsight-systems`)
- Kernel optimization or code generation (use `kernel-triton-writing`)
- Optimizing NCCL communication or distributed training collective
  operations
- Reducing GPU memory usage or gradient checkpointing
- General model compilation with `torch.compile`

## Requirements

| Dependency | Version | Notes |
|------------|---------|-------|
| PyTorch | >=2.0 | With CUDA support |
| NVIDIA GPU | Any | CUDA-capable |
| Nsight Systems | Optional | For comprehensive sync detection via `nsys` |

## Workflow

### Step 1: Detect Synchronizations

Use one or both methods to find sync points in the code.

**Quick detection** -- PyTorch sync debug mode prints a warning with stack
trace on every synchronization:

```python
import torch

# Enable at the start of the region you want to check
torch.cuda.set_sync_debug_mode('warn')   # prints warning + stack trace
# torch.cuda.set_sync_debug_mode('error')  # raises exception on sync

# Run your training step / forward pass here
train_step(model, batch)

torch.cuda.set_sync_debug_mode(0)  # disable
```

This mode only detects syncs going through PyTorch's wrapped
`cuStreamSynchronize`. Third-party libraries calling CUDA sync APIs
directly are not detected.

**Comprehensive detection** -- Nsight Systems captures all sync calls
including those from extensions and libraries:

```bash
nsys profile --capture-range=cudaProfilerApi \
             --python-sampling=true \
             --backtrace=dwarf \
             python your_script.py
```

In the Nsight Systems GUI, check the **CUDA API** timeline row and search
for `cudaStreamSynchronize`, `cudaEventSynchronize`, or
`cudaDeviceSynchronize`. The call stack panel shows which Python line
triggered each sync.

### Step 2: Classify -- False vs True Dependencies

After detecting syncs, classify each one before deciding how to fix it.

**False dependencies** (avoidable) -- CPU does not actually need the GPU
result. These can be eliminated without changing program logic:

- Debug prints left in hot paths (`print(loss.item())`)
- Unnecessary `.item()` calls for logging that could be deferred
- Using `.cuda()` instead of `.to('cuda', non_blocking=True)`
- Using `.type(torch.LongTensor)` instead of `.type(torch.long)`
- Creating tensors from Python objects directly on CUDA

**True dependencies** (require restructuring) -- CPU genuinely needs the
GPU value to proceed:

- **Control flow dependency**: `if loss.item() > threshold:` -- CPU
  branches on a GPU-computed value
- **Dynamic memory allocation**: `output = x[mask]` -- output size depends
  on GPU computation
- **CPU computation using GPU values**: computing statistics for logging,
  updating learning rates from metrics

True dependencies require restructuring: move logic to GPU
(`torch.where()`), delay to end of iteration, or accept that those parts
stay outside any CUDA Graph capture region.

### Step 3: Eliminate Systematically

Apply fixes in order of increasing difficulty. Start with easy wins.

**1. Remove redundancy** -- Delete operations that do not need to happen:
- Remove debug prints and logging from hot loops
- Delete unnecessary `.item()` calls
- Eliminate duplicate synchronizations

**2. Use `non_blocking=True`** -- Make transfers async where CPU does not
immediately use the result:

```python
# Before (syncs)
x_gpu = x_cpu.cuda()
x_cpu = x_gpu.cpu()

# After (async, no sync)
x_gpu = x_cpu.to('cuda', non_blocking=True)
x_cpu = x_gpu.to('cpu', non_blocking=True)   # only if CPU does not use x_cpu immediately
```

Only use `non_blocking=True` for GPU-to-CPU when the CPU does not
immediately read the result. Otherwise the CPU may operate on incomplete
data.

**3. Switch to sync-free API alternatives** -- See the Quick Reference
Table below for a condensed mapping of common patterns.

**4. Delay synchronization to end of iteration** -- Move logging and
validation to after the optimizer step rather than mid-forward/backward:

```python
# Before: sync mid-iteration
loss = model(batch)
print(f"Loss: {loss.item()}")    # cuStreamSynchronize
loss.backward()

# After: delay to end of iteration
loss = model(batch)
loss.backward()
optimizer.step()
print(f"Loss: {loss.item()}")    # sync is outside the hot path
```

**5. Coalesce multiple syncs into one** -- If you need several GPU values
on CPU, gather them and transfer once:

```python
# Before: 3 separate syncs
loss_val = loss.item()           # cuStreamSynchronize
acc_val = accuracy.item()        # cuStreamSynchronize
gnorm_val = grad_norm.item()     # cuStreamSynchronize

# After: 1 sync
metrics = torch.stack([loss, accuracy, grad_norm])
vals = metrics.cpu()             # single cuStreamSynchronize
loss_val, acc_val, gnorm_val = vals.tolist()
```

**6. Offload logic to GPU** -- Replace CPU-side logic with GPU-native ops:

```python
# Before: CPU control flow (syncs)
if loss.item() > threshold:
    result = a
else:
    result = b

# After: GPU-side selection (no sync)
result = torch.where(loss > threshold, a, b)

# Before: Python max (syncs)
val = max(x_gpu[0, 0], x_gpu[0, 1])

# After: torch.max (no sync)
val = torch.max(x_gpu[0, 0], x_gpu[0, 1])
```

**7. Exclude unavoidable syncs from capture range** (last resort) -- If a
sync cannot be eliminated, keep it outside the CUDA Graph capture region
and graph only the sync-free sections. Partial graphing is better than no
graphing.

### Step 4: Verify

Re-run detection to confirm syncs are eliminated:

```python
torch.cuda.set_sync_debug_mode('error')  # will raise if any sync remains
train_step(model, batch)
torch.cuda.set_sync_debug_mode(0)
```

Or re-profile with Nsight Systems and confirm no `cudaStreamSynchronize` /
`cudaEventSynchronize` / `cudaDeviceSynchronize` calls appear in the
target region.

## Quick Reference Table

| Sync-Inducing Pattern | Sync-Free Alternative |
|----------------------|----------------------|
| **Device Transfers** | |
| `.cpu()` or `.to('cpu')` | `.to('cpu', non_blocking=True)` (fire-and-forget only) |
| `.cuda()` or `.to('cuda')` | `.to('cuda', non_blocking=True)` |
| `.type(torch.LongTensor)` | `.type(torch.long)` (dtype conversion, stays on GPU) |
| **Tensor Creation** | |
| `torch.tensor(obj, device='cuda')` | Create on CPU, then `.to('cuda', non_blocking=True)` |
| `torch.tensor(0, device='cuda')` | `torch.zeros(1, device='cuda', dtype=...).squeeze()` |
| `torch.as_tensor(arr, device='cuda')` | Create on CPU, then `.to('cuda', non_blocking=True)` |
| `torch.cuda.BoolTensor(list)` | `torch.tensor(list, device='cpu').to('cuda', non_blocking=True)` |
| **Control Flow** | |
| `.item()` in conditionals | `torch.where()` or move outside critical region |
| `if gpu_tensor:` | Keep logic on GPU with `torch.where()` |
| Python `max(a, b)` on GPU tensors | `torch.max(a, b)` |
| `torch.is_nonzero(t)` | Avoid; use GPU-side comparisons |
| **Indexing** | |
| `x_gpu[idx_cpu]` or `x_gpu[idx_list]` | `x_gpu[idx_gpu]` (keep indices on same device) |
| `x_gpu[idx] = 0` (scalar assignment) | `x_gpu[idx] = zero_gpu` (GPU tensor value) |
| `x[i:j]` with CUDA tensor bounds | `x[:, s]` with `s = torch.arange(i, j, device='cuda')` |
| **Dynamic Shapes** | |
| `x_gpu[mask_gpu]` (masked selection) | `torch.where(mask_gpu, x_gpu, 0)` (fixed shape) |
| `torch.nonzero(mask)` | `torch.where()` or move outside critical region |
| `torch.masked_select(x, mask)` | `torch.where(mask, x, 0)` |
| `torch.unique(x)` | Avoid in hot path; precompute if possible |
| `torch.repeat_interleave(x, r)` | Specify `output_size=N` if known |

## Finding More Information

- **Tier 1 (this file)**: Workflow, classification, elimination strategies,
  and quick reference table
- **Tier 2 (`references/sync-patterns.md`)**: Comprehensive pattern catalog
  with 9 categories, full code examples showing sync-inducing and sync-free
  versions, and the specific CUDA driver API triggered by each pattern
