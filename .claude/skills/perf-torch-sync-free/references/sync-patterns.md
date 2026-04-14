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

# Sync Patterns Reference

Comprehensive catalog of PyTorch operations that cause host-device
synchronization, organized by category. Each section shows the sync-inducing
pattern, which CUDA driver API is triggered, and the sync-free alternative.

**Variable naming conventions**:

- `x_scalar` -- CPU scalar value
- `x_list` -- Python list
- `x_cpu` -- CPU tensor (on host)
- `x_gpu` -- GPU tensor (on device)
- `zero_gpu` -- `torch.zeros(1, device='cuda')`

---

## 1. Explicit Synchronizations

Direct calls that block the CPU waiting for GPU completion.

```python
stream = torch.cuda.Stream()
event = torch.cuda.Event()

event.record()
event.wait()              # cudaStreamWaitEvent -- non-blocking for host
event.synchronize()       # cuEventSynchronize -- BLOCKS CPU
stream.synchronize()      # cuStreamSynchronize -- BLOCKS CPU
torch.cuda.synchronize()  # cuCtxSynchronize -- BLOCKS CPU, waits for ALL streams
```

| Call | CUDA Driver API | Blocks CPU? |
|------|----------------|-------------|
| `event.wait()` | `cudaStreamWaitEvent` | No (GPU-side only) |
| `event.synchronize()` | `cuEventSynchronize` | Yes |
| `stream.synchronize()` | `cuStreamSynchronize` | Yes |
| `torch.cuda.synchronize()` | `cuCtxSynchronize` | Yes (all streams) |

**Fix**: Remove blocking syncs or move them outside the performance-critical
region. Use `event.wait()` (GPU-side stream ordering) instead of
`event.synchronize()` (CPU-blocking) when you only need stream dependencies.

---

## 2. Tensor Movement Between Devices

Moving tensors between CPU and GPU triggers synchronization by default.

```python
x_gpu = torch.randint(0, 10, (3, 4), device='cuda')

# GPU -> CPU
x_cpu = x_gpu.cpu()                                    # cuStreamSynchronize
x_cpu = x_gpu.to(device='cpu', non_blocking=True)      # no sync (async)

# CPU -> GPU
idx_cpu = torch.tensor([2, 0, 1], dtype=torch.int64)
idx_gpu = idx_cpu.cuda()                               # cuStreamSynchronize
idx_gpu = idx_cpu.cuda(non_blocking=True)               # no sync (async)
idx_gpu = idx_cpu.to(device='cuda', non_blocking=True)  # no sync (async)

# Type conversion pitfall
x_gpu.type(torch.long)        # dtype conversion, stays on GPU -- no sync
x_gpu.type(torch.LongTensor)  # creates CPU tensor -- cuStreamSynchronize
```

| Pattern | CUDA Driver API | Sync-Free Alternative |
|---------|----------------|----------------------|
| `.cpu()` | `cuStreamSynchronize` | `.to('cpu', non_blocking=True)` |
| `.cuda()` | `cuStreamSynchronize` | `.to('cuda', non_blocking=True)` |
| `.to(device=...)` | `cuStreamSynchronize` | Add `non_blocking=True` |
| `.type(torch.LongTensor)` | `cuStreamSynchronize` | `.type(torch.long)` |

**Important**: `non_blocking=True` for GPU-to-CPU is only safe when the CPU
does not immediately read the result. For fire-and-forget transfers (e.g.,
logging tensors to a buffer), it is safe. For transfers followed by CPU
computation on the result, you must sync or use a CUDA event to ensure
completion.

---

## 3. Tensor Creation and Allocation

Creating tensors from Python objects or NumPy arrays directly on CUDA syncs.

```python
# From Python scalar -- syncs
torch.tensor(0, device='cuda')                          # cuStreamSynchronize
torch.tensor(0).to(device='cuda')                       # cuStreamSynchronize
torch.tensor(0).to(device='cuda', non_blocking=True)    # no sync (async)

# Sync-free scalar creation
zero = torch.zeros(1, device='cuda', dtype=torch.int64).squeeze()  # no sync

# From NumPy array -- syncs
arr = np.array([1, 2, 3])
arr_gpu = torch.as_tensor(arr, device='cuda')           # cuStreamSynchronize

# Two-step approach -- no sync
arr_cpu = torch.as_tensor(arr, device='cpu')
arr_gpu = arr_cpu.to(device='cuda', non_blocking=True)  # no sync (async)

# From Python list -- syncs
bool_list = [False]
torch.cuda.BoolTensor(bool_list)                        # cuStreamSynchronize

# Two-step approach -- no sync
bool_cpu = torch.tensor(bool_list, dtype=torch.bool, device='cpu')
bool_gpu = bool_cpu.to(device='cuda', non_blocking=True)  # no sync (async)

# Direct allocation functions -- no sync
torch.zeros(3, 4, device='cuda')    # no sync
torch.ones(3, 4, device='cuda')     # no sync
torch.empty(3, 4, device='cuda')    # no sync
torch.randn(3, 4, device='cuda')    # no sync
```

| Pattern | CUDA Driver API | Sync-Free Alternative |
|---------|----------------|----------------------|
| `torch.tensor(obj, device='cuda')` | `cuStreamSynchronize` | Create on CPU + `.to('cuda', non_blocking=True)` |
| `torch.as_tensor(arr, device='cuda')` | `cuStreamSynchronize` | Create on CPU + `.to('cuda', non_blocking=True)` |
| `torch.cuda.BoolTensor(list)` | `cuStreamSynchronize` | `torch.tensor(list, device='cpu').to(...)` |
| `torch.zeros/ones/empty(device='cuda')` | None | Already sync-free |

**Rule of thumb**: `torch.zeros()`, `torch.ones()`, `torch.empty()`,
`torch.randn()` with `device='cuda'` do not sync. Anything that copies
Python/NumPy data to CUDA syncs. Use the two-step pattern: create on CPU,
transfer with `non_blocking=True`.

---

## 4. Data-Dependent Control Flow

Using GPU tensor values in Python control flow forces a sync to materialize
the value on CPU.

```python
loss = torch.randn(1, device="cuda")

# Explicit .item() -- syncs
loss.item()                          # cuStreamSynchronize

# Implicit .item() via truth testing -- syncs
torch.is_nonzero(loss)               # cuStreamSynchronize
if loss:                             # cuStreamSynchronize (implicit is_nonzero)
    pass

# Boolean conditions from GPU comparisons -- syncs
flag = x_gpu.sum() > 16             # creates bool GPU tensor
if flag:                             # cuStreamSynchronize (implicit is_nonzero)
    pass

flag = x_gpu.isnan().any()
if flag:                             # cuStreamSynchronize
    pass

# Python builtins on GPU tensors -- syncs
max(x_gpu[0, 0], x_gpu[0, 1])       # cuStreamSynchronize (Python max needs CPU values)

# GPU-native alternative -- no sync
torch.max(x_gpu[0, 0], x_gpu[0, 1])  # no sync, result stays on GPU
```

**Common real-world example** -- AMP GradScaler `_maybe_opt_step`:

```python
# This pattern appears in torch.amp.grad_scaler
if not sum(v.item() for v in optimizer_state["found_inf_per_device"].values()):
    # cuStreamSynchronize from each .item() call
    retval = optimizer.step(*args, **kwargs)
```

| Pattern | CUDA Driver API | Sync-Free Alternative |
|---------|----------------|----------------------|
| `.item()` | `cuStreamSynchronize` | Avoid; use tensor ops |
| `if gpu_tensor:` | `cuStreamSynchronize` | `torch.where(cond, a, b)` |
| `torch.is_nonzero(t)` | `cuStreamSynchronize` | GPU-side comparison |
| Python `max(a, b)` | `cuStreamSynchronize` | `torch.max(a, b)` |

**Solutions**: (1) Use `torch.where()` instead of `if/else`, (2) use
`torch.max()` instead of Python `max()`, (3) move data-dependent control
flow outside the critical region, (4) implement as custom CUDA kernel.

---

## 5. Indexing Tensors

Indexing GPU tensors with CPU indices or assigning Python scalars triggers
implicit device transfers.

```python
x_gpu = torch.randint(0, 10, (3, 4), device='cuda')
idx_list = [2, 0, 1]
idx_cpu = torch.tensor(idx_list, dtype=torch.int64, device='cpu')
idx_gpu = idx_cpu.to(device='cuda', non_blocking=True)
zero_gpu = torch.zeros(1, device='cuda')

# Integer scalar indexing -- no sync
x_gpu[0]                              # no sync

# CPU index tensors / Python lists -- sync
x_gpu[idx_list]                        # cuStreamSynchronize (implicit .to('cuda'))
x_gpu[idx_cpu]                         # cuStreamSynchronize (implicit .to('cuda'))

# GPU index tensor -- no sync
x_gpu[idx_gpu]                         # no sync

# Scalar assignment -- sync
x_gpu[idx_gpu] = 0                     # cuStreamSynchronize (scalar 0 transferred to GPU)

# GPU tensor assignment -- no sync
x_gpu[idx_gpu] = zero_gpu              # no sync

# Explicit index_select -- no sync
torch.index_select(x_gpu, 0, idx_gpu)  # no sync
```

| Pattern | CUDA Driver API | Sync-Free Alternative |
|---------|----------------|----------------------|
| `x_gpu[idx_cpu]` | `cuStreamSynchronize` | `x_gpu[idx_gpu]` |
| `x_gpu[idx_list]` | `cuStreamSynchronize` | `x_gpu[idx_gpu]` |
| `x_gpu[idx] = 0` | `cuStreamSynchronize` | `x_gpu[idx] = zero_gpu` |
| `x_gpu[0]` | None | Already sync-free |

**Rule**: Keep indices on the same device as the indexed tensor. For
assignments, use GPU tensor values instead of Python scalars.

---

## 6. Slicing with Tensor Indices

Python slice syntax (`x[start:stop]`) requires integer values. CUDA tensor
bounds trigger `.item()` calls.

```python
x = torch.randn((32, 32), device='cuda')
i = torch.tensor([8], dtype=torch.long, device='cuda')
j = torch.tensor([16], dtype=torch.long, device='cuda')
s = torch.tensor(list(range(8, 16)), dtype=torch.long, device='cuda')

# Python integer bounds -- no sync
x[8:16]                 # no sync
x[:, 8:16]              # no sync

# CUDA tensor bounds -- syncs (each bound calls .item())
x[i:j]                  # cuStreamSynchronize x2
x[:, i:j]               # cuStreamSynchronize x2
x[i:j, i:j]             # cuStreamSynchronize x4

# Index select with GPU tensor -- no sync
x[:, s]                 # no sync
x[s, :]                 # no sync
```

| Pattern | CUDA Driver API | Sync-Free Alternative |
|---------|----------------|----------------------|
| `x[cuda_i:cuda_j]` | `cuStreamSynchronize` (per bound) | `x[:, s]` with `s = torch.arange(...)` on GPU |

**Fix**: Use Python integers for slice bounds when known at Python execution
time. For GPU-computed bounds, build a GPU index tensor:
`s = torch.arange(start, end, device='cuda')` and use `x[:, s]` or
`torch.index_select()` instead of slice syntax.

---

## 7. Operations with Dynamic Output Shapes

Operations producing dynamically-sized outputs sync to determine the
allocation size on CPU.

### Masked Selection

The most common case. The output size depends on how many mask elements
are `True`, which is a GPU computation.

```python
mask_gpu = x_gpu > 5

# Dynamic shape operations -- sync
torch.nonzero(mask_gpu)                    # cuStreamSynchronize
x_gpu[mask_gpu]                            # cuStreamSynchronize (implicit nonzero)
torch.where(mask_gpu)                      # cuStreamSynchronize (implicit nonzero as_tuple)
x_gpu[torch.where(mask_gpu)]               # cuStreamSynchronize
torch.masked_select(x_gpu, mask_gpu)       # cuStreamSynchronize

# CPU mask avoids sync (size known on CPU)
mask_cpu = mask_gpu.to(device='cpu', non_blocking=True)
# (must sync or wait before using mask_cpu)
x_gpu[mask_cpu]                            # no sync (size determined from CPU tensor)

# Fixed-shape alternatives -- no sync
x_gpu & mask_gpu                           # bitwise_and, same shape as input
torch.where(mask_gpu, x_gpu, 0)            # same shape as input, fills 0 where False

# Correct mean over masked elements
torch.where(mask_gpu, x_gpu, 0).float().sum() / torch.sum(mask_gpu)
```

### Masked Assignment

Assignment into masked positions behaves differently from selection.

```python
# These sync (dynamic indexing or tensor value metadata)
x_gpu[torch.where(mask_gpu)] = -1          # cuStreamSynchronize x2
x_gpu[mask_gpu] = zero_gpu                 # cuStreamSynchronize

# These do NOT sync (no new allocation needed)
x_gpu[mask_gpu] = -1                       # no sync (scalar broadcast)
x_gpu.masked_fill_(mask_gpu, -1)           # no sync

# torch.where with 3 args -- no sync
x_gpu = torch.where(mask_gpu, zero_gpu, x_gpu)  # no sync
```

### Backward with Masked Indexing

```python
x_fp32 = x_gpu.float().requires_grad_()
grad_gpu = torch.randn_like(x_fp32)

x_fp32.backward(grad_gpu)                               # no sync
x_fp32[0].backward(grad_gpu[0])                          # no sync
x_fp32[idx_gpu].backward(grad_gpu[idx_gpu])               # no sync
x_fp32[mask_cpu].backward(grad_gpu[mask_cpu])             # no sync
x_fp32[mask_gpu].backward(grad_gpu[mask_gpu])             # cuStreamSynchronize x2
```

### Other Dynamic Shape Operations

```python
y = torch.zeros((2, 2), device='cuda')
repeats = torch.tensor([1, 2], device='cpu').to('cuda', non_blocking=True)

torch.repeat_interleave(y, repeats, dim=0)                # cuStreamSynchronize x2
torch.repeat_interleave(y, repeats, dim=0, output_size=3)  # no sync (size specified)

torch.unique(idx_gpu)              # cuStreamSynchronize
torch.unique_consecutive(idx_gpu)  # cuStreamSynchronize
```

| Pattern | CUDA Driver API | Sync-Free Alternative |
|---------|----------------|----------------------|
| `torch.nonzero(mask)` | `cuStreamSynchronize` | `torch.where(mask, x, 0)` |
| `x_gpu[mask_gpu]` | `cuStreamSynchronize` | `torch.where(mask_gpu, x_gpu, 0)` |
| `torch.masked_select()` | `cuStreamSynchronize` | `torch.where(mask, x, 0)` |
| `x_gpu[mask_gpu] = val_gpu` | `cuStreamSynchronize` | `torch.where(mask, val, x)` or `masked_fill_` |
| `x_gpu[mask_gpu] = -1` | None | Already sync-free |
| `torch.repeat_interleave(x, r)` | `cuStreamSynchronize` | Specify `output_size=N` |
| `torch.unique(x)` | `cuStreamSynchronize` | Avoid in hot path; precompute |

---

## 8. Embedding Layers

Embedding backward can sync depending on PyTorch version and index count.

```python
embedding = torch.nn.Embedding(50304, 768).cuda()
idx = torch.randint(0, 50304, (4, 1024), dtype=torch.int64)
idx = idx.to(device='cuda', non_blocking=True)

y = embedding(idx)
y.sum().backward()  # cuStreamSynchronize x2 in older PyTorch
```

**Root cause**: When the number of indices exceeds 3072 and PyTorch uses
CUB < 1.16, the embedding backward falls back to
`thrust::unique_by_key_copy` which syncs. With CUB >= 1.16, it uses
`cuda::cub::unique_by_key` which is sync-free.

**Fix**: Upgrade to recent PyTorch (>= 2.0) which includes CUB >= 1.16.
This is a framework-level fix; no user code changes needed.

---

## 9. Distributed Operations

Distributed collectives and barriers involve synchronization.

```python
import torch.distributed as dist

# Synchronous collective -- syncs via events internally
dist.all_reduce(x_gpu)                    # cuEventSynchronize (internal)

# Async collective -- returns immediately
work = dist.all_reduce(x_gpu, async_op=True)
work.wait()                               # cuEventSynchronize (explicit wait)

# Global barrier -- syncs all ranks
dist.barrier()                            # cuCtxSynchronize
```

**DDP logging syncs**: In older PyTorch versions (< 2.0), DistributedDataParallel
invokes `cuEventSynchronize` x5 for internal performance logging. See
[torch/nn/parallel/distributed.py](https://github.com/pytorch/pytorch/blob/v1.10.0/torch/nn/parallel/distributed.py#L855).

| Pattern | CUDA Driver API | Sync-Free Alternative |
|---------|----------------|----------------------|
| `dist.all_reduce(x)` | `cuEventSynchronize` | `dist.all_reduce(x, async_op=True)` |
| `work.wait()` | `cuEventSynchronize` | Delay `.wait()` as long as possible |
| `dist.barrier()` | `cuCtxSynchronize` | Minimize usage; avoid in hot loops |

**Fix**: Use `async_op=True` where possible. Ensure NCCL >= 2.9.6 which
supports CUDA Graph capture of collectives. Upgrade PyTorch to avoid DDP
logging syncs.
