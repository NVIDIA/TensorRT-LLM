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

# CuTe Layouts & Layout Algebra

## Layout Fundamentals

A Layout is a `(Shape, Stride)` tuple that maps coordinates to indices.
Layouts are **functions from integers to integers**.

```python
# Layout (4,8):(1,4) maps 2D coordinate (i,j) to index i*1 + j*4
# This is a 4x8 column-major layout
```

### IntTuple

The recursive building block. An IntTuple is either an integer or a tuple
of IntTuples. Operations:
- `rank(t)`: Number of top-level elements
- `depth(t)`: Maximum nesting level
- `size(t)`: Product of all leaf integers
- `get<I>(t)`: Access I-th element

### Shape and Stride

Both are IntTuples. Shape defines the coordinate domain; Stride defines the
index mapping. The index for natural coordinate `c` with stride `d` is the
inner product: `index = sum(c_i * d_i)`.

### Layout Creation

```python
# Python DSL
layout = cute.make_layout((4, 8), (1, 4))        # Column-major 4x8
layout = cute.make_layout((4, 8))                  # Default stride (col-major)
layout = cute.make_ordered_layout((M, N, K), order=(0, 1, 2))  # Custom order
identity = cute.make_identity_layout((M, N))       # Maps coords to themselves
```

Default stride generation: `LayoutLeft` (column-major) when stride omitted.

## Coordinate Systems

A single Layout can be indexed with multiple coordinate types via
**colexicographical ordering** (right-to-left):

For shape `(3, (2, 3))`:
- **1-D**: 0–17 (linearized index)
- **2-D**: `(i, j)` where `i ∈ [0,3)`, `j ∈ [0,6)`
- **Natural (h-D)**: `(i, (j, k))` matching shape hierarchy

Functions: `idx2crd(idx, shape)` and `crd2idx(coord, shape, stride)`.

## Static vs Dynamic Layouts

**Static layouts**: All shape values known at compile time. Enable full
optimization (unrolling, vectorization). Each distinct shape requires
separate compilation.

```python
# Static: pass CuTe tensor with fixed shape
a_cute = from_dlpack(a_torch)  # Shape baked into compilation
```

**Dynamic layouts**: Shape modes replaced with runtime values (`?`).
Single compilation handles varying input shapes.

```python
# Dynamic: pass torch.Tensor directly (auto mark_layout_dynamic)
compiled = cute.compile(foo, a_torch)  # Shape: (?,?):(?,1)
compiled(a_torch)  # Works with any shape
compiled(b_torch)  # Same compiled function, different shape
```

**Fine-grained control**:
```python
t = from_dlpack(tensor).mark_layout_dynamic(leading_dim=1)
t = from_dlpack(tensor).mark_compact_shape_dynamic(mode=0, divisibility=2)
```

## Layout Algebra Operations

### Coalesce

Simplifies layout by merging adjacent modes with contiguous strides.
Preserves functional behavior while reducing rank.

Adjacent modes `s0:d0` and `s1:d1` combine when `d1 == s0 * d0`.

### Composition

Functional composition: `R(c) = A(B(c))`. Produces a new layout.

```python
result = cute.composition(layout_A, layout_B)
```

Key property: `compatible(B, result)` — B's coordinates work for the result.

### Complement

Finds layout of elements NOT selected by the input layout. The complement
is disjoint, ordered, and bounded.

```python
# complement(4:1, 24) = 6:4
# Input selects {0,1,2,3}, complement selects {4,8,12,16,20}
```

### Division Operations

Split layout A into two modes: selected elements and remainder.
Formula: `A ÷ B := A ∘ (B, B*)`  where `B*` is complement.

| Variant | Result Shape | Use Case |
|---------|-------------|----------|
| `logical_divide` | `((TileM,RestM), (TileN,RestN))` | Preserve mode semantics |
| `zipped_divide` | `((TileM,TileN), (RestM,RestN))` | Group tile vs rest |
| `tiled_divide` | `((TileM,TileN), RestM, RestN)` | Flatten remainder |
| `flat_divide` | `(TileM, TileN, RestM, RestN)` | Fully flat |

`zipped_divide` is the most common in CuTe DSL element-wise kernels.

### Product Operations

Create layout where first mode is A and second replicates A via B.

| Variant | Pattern | Use Case |
|---------|---------|----------|
| `blocked_product` | Block distribution | Contiguous tile assignment |
| `raked_product` | Cyclic distribution | Round-robin assignment |

## Swizzle Patterns

Swizzles are bit permutations applied to layout indices to avoid shared
memory bank conflicts. Defined by three parameters:
- **MBase**: Constant bits (untouched)
- **BBits**: Mask bits (XOR source)
- **SShift**: Shift distance (XOR target)

```python
swizzle = cute.Swizzle(MBase=3, BBits=2, SShift=3)
layout = cute.make_composed_layout(swizzle, base_layout)
```

## Key Properties

| Property | Description | Function |
|----------|-------------|----------|
| Rank | Number of top-level modes | `cute.rank(layout)` |
| Size | Total elements in domain | `cute.size(layout)` |
| Cosize | Codomain extent (memory footprint) | `cute.cosize(layout)` |
| Depth | Maximum nesting level | `cute.depth(layout)` |
| Leading dim | Stride-1 dimension | `cute.leading_dim(layout)` |

## Layout Compatibility

Shape A is compatible with Shape B if sizes match and all valid coordinates
in A are valid in B:
- `24` compatible with `(4,6)` ✓
- `((2,2),6)` compatible with `((2,2),(3,2))` ✓
- `((2,3),4)` NOT compatible with `((2,2),(3,2))` ✗

## Constraints

- Only 32-bit shapes/strides supported in CuTe layouts (64-bit planned)
- Layout algebra operations require JIT compilation — cannot be used in
  native Python outside `@cute.jit` / `@cute.kernel`
- `Layout` types cannot be passed as native Python function arguments
