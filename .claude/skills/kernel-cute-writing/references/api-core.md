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

# API Reference: cute Module (Core)

## Decorators

### @cute.jit

JIT-compiled host function, callable from Python.

```python
@cute.jit
def host_fn(tensor: cute.Tensor, flag: cutlass.Constexpr):
    ...
```

Parameters:
- `preprocessor` (bool, default `True`): Enable AST rewrite for control flow
- `no_cache` (bool, default `False`): Skip caching, recompile each call

### @cute.kernel

GPU kernel function. Must be launched from `@cute.jit`.

```python
@cute.kernel
def my_kernel(gA: cute.Tensor, gC: cute.Tensor):
    ...

# Launch from @cute.jit:
my_kernel(gA, gC).launch(grid=(G,1,1), block=(T,1,1), smem=bytes)
```

Parameters: `grid`, `block`, `cluster` (int tuples), `smem` (bytes)

### @cute.struct

Define C structures in Python for shared memory, arguments, etc.

```python
@cute.struct
class MyStruct:
    field_a: cutlass.Float32
    field_b: cute.Array[cutlass.Float16, 8]
```

## Compilation

```python
compiled = cute.compile(fn, *args, options="")
compiled(*runtime_args)

# Access generated code
compiled.__ptx__    # PTX assembly string
compiled.__cubin__  # Binary kernel bytes
compiled.__mlir__   # MLIR IR string
```

## Layout Operations

### Creation

```python
cute.make_layout(shape, stride=None)           # From shape + optional stride
cute.make_identity_layout(shape)               # Coord → coord identity
cute.make_ordered_layout(shape, order)         # Custom dimension ordering
cute.make_composed_layout(inner, outer)        # Composed (swizzled) layout
```

### Algebra

```python
cute.composition(layout_A, layout_B)           # R(c) = A(B(c))
cute.complement(layout, cosize)                # Elements not in layout
cute.coalesce(layout)                          # Merge contiguous modes
cute.ceil_div(a, b)                            # Ceiling division
```

### Division / Tiling

```python
cute.logical_divide(layout, tiler)             # ((Tile,Rest), ...)
cute.zipped_divide(layout, tiler)              # ((Tile...), (Rest...))
cute.tiled_divide(layout, tiler)               # ((Tile...), Rest0, Rest1)
cute.flat_divide(layout, tiler)                # (Tile0, Tile1, Rest0, Rest1)
```

### Partitioning

```python
cute.local_partition(tensor, layout, idx)      # Thread-level partition
cute.local_tile(tensor, tiler, coord)          # CTA-level tile selection
```

### Properties

```python
cute.rank(x)           # Number of top-level modes
cute.depth(x)          # Maximum nesting level
cute.size(x)           # Total elements (product of shape)
cute.cosize(x)         # Memory footprint
cute.leading_dim(x)    # Stride-1 dimension extent
cute.is_static(x)      # True if all values compile-time known
cute.is_major(x, d)    # True if dimension d is stride-1
```

## Tensor Operations

### Creation

```python
cute.make_tensor(ptr, layout)                  # From pointer + layout
cute.make_identity_tensor(shape)               # Coordinate mapping tensor
cute.make_rmem_tensor(layout)                  # Register memory tensor
cute.make_fragment(shape, dtype)               # Register fragment
```

### Data Initialization

```python
cute.full(shape, value, dtype)                 # Filled tensor
cute.full_like(tensor, value)                  # Fill with same shape
cute.zeros_like(tensor)                        # Zero-filled copy
cute.ones_like(tensor)                         # One-filled copy
cute.empty_like(tensor)                        # Uninitialized copy
```

### Manipulation

```python
tensor.load()                                  # Load from memory
tensor.to(dtype)                               # Type cast
cute.reshape(tensor, new_shape)                # Reshape
cute.broadcast_to(tensor, new_shape)           # Broadcast

cute.flatten(x)                                # Flatten nested structure
cute.unflatten(x, profile)                     # Rebuild nested structure
cute.group_modes(layout, start, end)           # Group mode range
cute.select(layout, *indices)                  # Extract sublayout
cute.slice_(layout, coord)                     # Slice at coordinate
cute.get(x, *indices)                          # Nested element access
```

### Conditional

```python
cute.where(condition, true_val, false_val)     # Element-wise select
cute.any_(tensor)                              # Any element true
cute.all_(tensor)                              # All elements true
```

### Coordinate Conversion

```python
cute.crd2idx(coord, shape, stride)             # Coordinate → linear index
cute.idx2crd(idx, shape)                       # Linear index → coordinate
cute.slice_and_offset(coord, layout)           # Combined slice + offset
```

## Math Operations

Available in `cute.math`:

| Function | Description |
|----------|-------------|
| `cute.math.exp(x)` | Exponential |
| `cute.math.log(x)` | Natural logarithm |
| `cute.math.tanh(x)` | Hyperbolic tangent |
| `cute.math.sin(x)` | Sine |
| `cute.math.cos(x)` | Cosine |
| `cute.math.sqrt(x)` | Square root |
| `cute.math.rsqrt(x)` | Reciprocal square root |
| `cute.math.erf(x)` | Error function |

Arithmetic: `+`, `-`, `*`, `/` work directly on CuTe tensor values.

**No sigmoid** — implement as: `1.0 / (1.0 + cute.math.exp(-x))`

## Copy Operations

```python
cute.copy(src, dst)                            # Default copy
cute.copy(copy_atom, src, dst)                 # With specific atom
cute.copy_if(predicate, src, dst)              # Conditional copy
```

## GEMM

```python
cute.gemm(tiled_mma, A, B, C)                 # Tiled matrix multiply
```

Dispatch modes:
- `(V) × (V) → (V)`: Element-wise
- `(M,K) × (N,K) → (M,N)`: Standard GEMM
- `(V,M,K) × (V,N,K) → (V,M,N)`: Batched GEMM

## Printing

```python
print(value)                    # Compile-time only (Python print)
cute.printf(fmt, *args)         # Runtime GPU printf (adds PTX overhead)
cute.print_tensor(tensor)       # Formatted tensor display
cute.print_latex(layout)        # LaTeX/TiKZ visualization
```

## Utility

```python
cute.assume(condition)          # Compiler optimization hint
cute.static(value)              # Force static evaluation
cute.E(idx)                     # Static basis element
cute.sym_int()                  # Symbolic integer for fake tensors
```

## Control Flow Helpers

```python
cutlass.dynamic_expr(cond)      # Runtime conditional guard
cutlass.const_expr(cond)        # Compile-time conditional
cutlass.Constexpr               # Type annotation for compile-time args
cutlass.range(n)                # IR loop with optional attributes
cutlass.range_constexpr(n)      # Compile-time unrolled loop
```
