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

# CuTe Tensors & Partitioning

## Tensor Fundamentals

A Tensor combines two components:
- **Engine**: Iterator/pointer holding actual data
- **Layout**: Defines logical coordinates and memory mapping

```python
# Create tensor from pointer + layout
tensor = cute.make_tensor(ptr, layout)
tensor = cute.make_tensor(cute.make_gmem_ptr(ptr), shape)
```

## Memory Spaces

Tensors are tagged with memory space information:

| Space | Tag | Description |
|-------|-----|-------------|
| Global | `gmem` | Device DRAM, accessible by all threads |
| Shared | `smem` | Per-CTA fast memory, ~100KB-200KB |
| Register | `rmem` | Per-thread, fastest, limited |
| Tensor Memory | `tmem` | Blackwell MMA register file |

```python
# Python DSL: from_dlpack creates gmem tensors
mA = cute.runtime.from_dlpack(torch_tensor, assumed_align=16)

# Shared memory allocation
smem_ptr = cute.arch.alloc_smem(dtype, layout)

# Register tensor (owning)
rmem = cute.make_rmem_tensor(layout)
```

## Tensor Creation

### From Framework Tensors (Nonowning)
```python
from cutlass.cute.runtime import from_dlpack

# Explicit conversion (preferred for perf)
mA = from_dlpack(torch_tensor, assumed_align=16)

# Implicit conversion (auto dynamic layout)
@cute.jit
def foo(tensor):  # Pass torch.Tensor directly
    ...
```

Parameters for `from_dlpack`:
- `assumed_align`: Byte alignment hint (16 for LDG.128, 32 for LDG.256)
- `use_32bit_stride`: Reduces register usage for small tensors
- `enable_tvm_ffi`: For TVM FFI compilation path

### From Raw Pointers
```python
from cutlass.cute.runtime import make_ptr
ptr = make_ptr(cutlass.Float16, address, cute.AddressSpace.gmem, assumed_align=16)
layout = cute.make_ordered_layout((M, K, L), order=(0, 1, 2))
mA = cute.make_tensor(ptr, layout=layout)
```

### Register/Fragment Tensors (Owning)
```python
rmem = cute.make_rmem_tensor(layout)
frag = cute.make_fragment(shape, dtype)
```

## Accessing Tensors

```python
# By coordinate
val = tensor[(m, n)]
val = tensor[i].load()  # Load from memory

# Store
tensor[(m, n)] = value

# Slicing with underscore (retain dimension)
row = tensor[(:, n)]     # All rows at column n
col = tensor[(m, :)]     # All columns at row m
```

## Tiling (Division) Operations

Tiling splits a tensor into sub-tiles using layout algebra division.
Result has two logical groups: **tile elements** and **remainder/rest**.

| Operation | Result Structure | Use Case |
|-----------|-----------------|----------|
| `logical_divide(T, tiler)` | `((TileM,RestM), ...)` | Preserve semantics |
| `zipped_divide(T, tiler)` | `((Tile...), (Rest...))` | **Most common** |
| `tiled_divide(T, tiler)` | `((Tile...), Rest0, ...)` | Flat remainder |
| `flat_divide(T, tiler)` | `(Tile0, Tile1, Rest0, ...)` | Fully flat |

```python
# Element-wise kernel pattern: vectorize last dimension
vec_size = 16 // (mA.element_type.width // 8)  # 8 for FP16
tiler = (1, vec_size)                            # 2D tiler
gA = cute.zipped_divide(mA, tiler)
# gA[tile_coord, rest_coord] accesses vectorized chunks
```

## Partitioning Strategies

### Inner Partitioning (Tile Assignment)
Distribute tiles across CTAs/warps. Preserves tile dimensions.

```python
# Assign tiles to thread blocks
gA = cute.zipped_divide(mA, cta_tiler)
cta_gA = gA[(:, :), (blockIdx_x, blockIdx_y)]
```

### Outer Partitioning (Thread Distribution)
Distribute tile elements across threads within a CTA.

```python
gA = cute.zipped_divide(mA, tiler)
# Each thread handles tiles at stride
thread_gA = gA[thread_idx, (:, :)]
```

### local_tile / local_partition
Convenience wrappers combining tiling + coordinate selection:

```python
# CTA-level tiling
cta_A = cute.local_tile(mA, cta_tiler, cta_coord)

# Thread-level partitioning
thr_A = cute.local_partition(sA, thread_layout, thread_idx)
```

## Predication (Bounds Checking)

When tiles don't divide evenly, use predication to mask invalid accesses.

**Pattern**: Create an identity tensor, apply same tiling, compare coordinates.

```python
# Create coordinate tensor
cA = cute.make_identity_tensor(cute.shape(mA))

# Apply same tiling as data tensor
tiled_cA = cute.zipped_divide(cA, tiler)

# Check bounds
if cutlass.dynamic_expr(thread_idx < total_tiles):
    coord = tiled_cA[(None, (mi, ni))]
    # coord gives the logical position; compare against shape
```

In CuTe DSL element-wise kernels, use `cutlass.dynamic_expr` for bounds:
```python
if cutlass.dynamic_expr(thread_idx < total_tiles):
    a_val = gA[(None, (mi, ni))].load()
    gC[(None, (mi, ni))] = result
```

## Zero-Copy Design

Converted tensors (via `from_dlpack`) share underlying memory with source
tensors — no data duplication. The source tensor must outlive the CuTe tensor.

## Key Tensor Properties

```python
tensor.shape        # Shape tuple
tensor.stride       # Stride tuple
tensor.layout       # Combined layout
tensor.element_type # Data type (e.g., cutlass.Float16)
tensor.memspace     # Memory space tag
cute.size(tensor)   # Total elements
cute.rank(tensor)   # Number of top-level modes
```
