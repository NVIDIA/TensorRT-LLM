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

# Memory Operations & Data Movement

## Framework Tensor Conversion

### Explicit: from_dlpack (Preferred for Performance)

```python
from cutlass.cute.runtime import from_dlpack

mA = from_dlpack(torch_tensor, assumed_align=16)
# assumed_align=16 → LDG.128/STG.128
# assumed_align=32 → LDG.256/STG.256 (Blackwell)

# Additional options
mA = from_dlpack(tensor, use_32bit_stride=True)     # Smaller strides
mA = from_dlpack(tensor, enable_tvm_ffi=True)        # TVM FFI path
```

Overhead: ~2-3 microseconds per call. Cache converted tensors for
repeated use.

### Implicit: Pass torch.Tensor Directly

```python
@cute.jit
def foo(tensor):  # Accepts torch.Tensor
    # Auto-converted to CuTe tensor with dynamic layout
    print(tensor.layout)  # (?,?):(?,1)
```

Leading dimension stride fixed at 1. Broadcast strides (0) preserved.

### Raw Pointers

```python
from cutlass.cute.runtime import make_ptr

ptr = make_ptr(
    cutlass.Float16,
    torch_tensor.data_ptr(),
    cute.AddressSpace.gmem,
    assumed_align=32,
)
layout = cute.make_ordered_layout((M, K), order=(0, 1))
mA = cute.make_tensor(ptr, layout=layout)
```

Bypasses DLPack overhead entirely.

## Dynamic vs Static Layout Control

```python
# Static (fixed shape, optimal codegen)
mA = from_dlpack(tensor)  # Shape baked into compilation

# Dynamic (varying shapes, single compilation)
mA = from_dlpack(tensor).mark_layout_dynamic()
# Or pass torch.Tensor directly (auto-dynamic)

# Fine-grained dynamic
mA = from_dlpack(tensor).mark_compact_shape_dynamic(
    mode=0,              # Which dimension is dynamic
    divisibility=2,      # Alignment constraint
    stride_order=(1,0),  # Custom stride ordering
)
```

## Global Memory Access

### Vectorized Loads/Stores

Alignment hints enable vector instructions:

| assumed_align | Instruction | FP16 elements/thread |
|--------------|-------------|---------------------|
| 16 | LDG.128 / STG.128 | 8 |
| 32 | LDG.256 / STG.256 | 16 |

### Load/Store with Cache Hints

```python
# Low-level load with eviction policy
val = cute.arch.load(ptr, cache_mode)

# Low-level store with coherence hints
cute.arch.store(ptr, val, cache_mode)
```

## Shared Memory (SMEM)

### Static Allocation

```python
smem = cute.arch.alloc_smem(dtype, layout)
```

### Dynamic Allocation

```python
smem_ptr = cute.arch.get_dyn_smem(dtype, offset)
```

### SmemAllocator (Utils)

```python
from cutlass.utils import SmemAllocator

allocator = SmemAllocator()
smem_A = allocator.allocate_tensor(cutlass.Float16, layout_A, swizzle)
smem_B = allocator.allocate_tensor(cutlass.Float16, layout_B, swizzle)
total_smem = allocator.total_bytes()
```

### Swizzled Layouts for Bank Conflict Avoidance

```python
# Architecture-specific SMEM layout atoms
from cutlass.cute.nvgpu.warpgroup import make_smem_layout_atom, SmemLayoutAtomKind

smem_atom = make_smem_layout_atom(SmemLayoutAtomKind.K_SW128, cutlass.Float16)
```

## TMA (Tensor Memory Accelerator) — Hopper+

TMA transfers entire tiles between global and shared memory with a single
instruction. A TMA descriptor encodes: base pointer, data type, dimensions,
strides, swizzle pattern, and out-of-bounds behavior.

### TMA Copy Atoms

```python
from cutlass.cute.nvgpu.cpasync import (
    CopyBulkTensorTileG2SOp,      # Global → Shared
    CopyBulkTensorTileS2GOp,      # Shared → Global
    CopyBulkTensorTileG2SMulticastOp,  # Global → Shared (multicast)
    CopyReduceBulkTensorTileS2GOp,     # Shared → Global (with reduction)
)

# Create TMA atom
tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
    op=CopyBulkTensorTileG2SOp(),
    gmem_tensor=mA,
    smem_layout_=smem_layout,
    cta_tiler=cta_tiler,
)

# Partition
gmem_tma, smem_tma = cute.nvgpu.cpasync.tma_partition(
    tma_atom, tma_tensor, smem_tensor)
```

### TMA Descriptor Management

```python
cute.nvgpu.cpasync.prefetch_descriptor(tma_desc)
cute.nvgpu.cpasync.update_tma_descriptor(desc, base_addr, shape, stride)
cute.nvgpu.cpasync.fence_tma_desc_acquire()
cute.nvgpu.cpasync.fence_tma_desc_release()
```

## cp.async — Hopper

Non-bulk asynchronous global → shared memory copy:

```python
from cutlass.cute.nvgpu.cpasync import CopyG2SOp, LoadCacheMode

copy_op = CopyG2SOp(cache_mode=LoadCacheMode.always_)
```

Commit and wait:
```python
cute.arch.cp_async_commit_group()
cute.arch.cp_async_wait_group(0)  # Wait for all outstanding
```

## Tensor Memory (TMEM) — Blackwell

Dedicated register file for MMA operations on Blackwell.

```python
from cutlass.cute.nvgpu.tcgen05 import Ld16x128bOp, St16x128bOp, Repetition

# TMEM → Register
ld_op = Ld16x128bOp(repeat=Repetition.x1)

# Register → TMEM
st_op = St16x128bOp(repeat=Repetition.x1)

# Allocation
tmem_ptr = cute.arch.alloc_tmem(num_columns)
cute.arch.dealloc_tmem(tmem_ptr)
```

## Copy Atom Patterns

| Pattern | Source → Dest | Architecture |
|---------|--------------|-------------|
| Universal copy | Any → Any | All |
| cp.async (CopyG2SOp) | Global → Shared | SM80+ |
| TMA tile (G2S/S2G) | Global ↔ Shared | SM90+ |
| LdMatrix | Shared → Register | SM80+ |
| StMatrix | Register → Shared | SM80+ |
| TMEM load/store | TMEM ↔ Register | SM100 |
| SMEM → TMEM (S2T) | Shared → TMEM | SM100 |

## Zero-Copy Design

All tensor conversions (from_dlpack, implicit) share underlying memory.
No data duplication. Source tensor must outlive CuTe tensor.
