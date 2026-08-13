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

# API Reference: cute.nvgpu Module

GPU-specific MMA and Copy operations organized by architecture.

## Top-Level (Architecture-Agnostic)

```python
from cutlass.cute import nvgpu
```

Contains common enums and operations shared across architectures.

## Warp Submodule (SM80+)

```python
from cutlass.cute.nvgpu import warp
```

### MMA Operations

**MmaF16BF16Op** — Half/BFloat16 warp-level MMA:
```python
mma = warp.MmaF16BF16Op(
    ab_dtype=cutlass.Float16,       # or cutlass.BFloat16
    acc_dtype=cutlass.Float32,
    shape_mnk=(16, 8, 16),          # Instruction tile dimensions
)
```

**MmaMXF4Op** — MXF4 warp-level MMA (microscaling FP4):
```python
mma = warp.MmaMXF4Op(
    ab_dtype=..., acc_dtype=..., sf_type=...,
)
```

**MmaMXF4NVF4Op** — MXF4+NVF4 warp-level MMA

### Matrix Load/Store

**LdMatrix** (Shared → Register):
```python
ld = warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=1)
ld = warp.LdMatrix16x8x8bOp(transpose=False, num_matrices=1)
ld = warp.LdMatrix16x16x8bOp(transpose=False, num_matrices=1)
```

**StMatrix** (Register → Shared):
```python
st = warp.StMatrix8x8x16bOp(transpose=False, num_matrices=1)
st = warp.StMatrix16x8x8bOp(transpose=False, num_matrices=1)
```

### Runtime Fields

```python
warp.Field.ACCUMULATE   # Accumulator control
warp.Field.SFA          # Scale factor A
warp.Field.SFB          # Scale factor B
```

## Warpgroup Submodule (SM90+)

```python
from cutlass.cute.nvgpu import warpgroup
```

### MMA Operations

**MmaF16BF16Op** — Warpgroup MMA (128 threads):
```python
mma = warpgroup.MmaF16BF16Op(
    ab_dtype=cutlass.Float16,
    acc_dtype=cutlass.Float32,
    instruction_shape=(64, 128, 16),
    a_src=warpgroup.OperandSource.SMEM,
    a_major_mode=warpgroup.OperandMajorMode.K,
    b_major_mode=warpgroup.OperandMajorMode.K,
)
```

**MmaF8Op** — FP8 warpgroup MMA:
```python
mma = warpgroup.MmaF8Op(
    a_dtype=cutlass.Float8E4M3,
    b_dtype=cutlass.Float8E5M2,
    acc_dtype=cutlass.Float32,
    instruction_shape=(64, 128, 32),
    a_src=warpgroup.OperandSource.SMEM,
    a_major_mode=warpgroup.OperandMajorMode.K,
    b_major_mode=warpgroup.OperandMajorMode.K,
)
```

### Shared Memory Layout

```python
smem_atom = warpgroup.make_smem_layout_atom(
    warpgroup.SmemLayoutAtomKind.K_SW128,
    cutlass.Float16,
)
```

Kinds: `MN_INTER`, `MN_SW32`, `MN_SW64`, `MN_SW128`,
       `K_INTER`, `K_SW32`, `K_SW64`, `K_SW128`

### Synchronization

```python
warpgroup.fence()              # Warpgroup fence
warpgroup.commit_group()       # Commit instruction batch
warpgroup.wait_group(n)        # Wait for group n to complete
```

## cpasync Submodule (SM80+)

```python
from cutlass.cute.nvgpu import cpasync
```

### Non-Bulk Copy

```python
copy_op = cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.always_)
```

### TMA Copy Operations (SM90+)

```python
# Global → Shared (tile mode)
g2s = cpasync.CopyBulkTensorTileG2SOp(cta_group=cpasync.CtaGroup.ONE)

# Global → Shared (multicast)
g2s_mc = cpasync.CopyBulkTensorTileG2SMulticastOp()

# Shared → Global
s2g = cpasync.CopyBulkTensorTileS2GOp()

# Shared → Global (with reduction)
s2g_red = cpasync.CopyReduceBulkTensorTileS2GOp(reduction_op="ADD")
```

### TMA Atom Construction

```python
tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
    op=cpasync.CopyBulkTensorTileG2SOp(),
    gmem_tensor=mA,
    smem_layout_=smem_layout,
    cta_tiler=cta_tiler,
    num_multicast=1,
    internal_type=None,  # Optional type casting
)
gmem_tma, smem_tma = cpasync.tma_partition(tma_atom, tma_tensor, smem_tensor)
```

### TMA Descriptor Management

```python
cpasync.prefetch_descriptor(tma_desc)
cpasync.copy_tensormap(dest, src)
cpasync.update_tma_descriptor(desc, base_addr, shape, stride)
cpasync.fence_tma_desc_acquire()
cpasync.fence_tma_desc_release()
cpasync.cp_fence_tma_desc_release()
cpasync.create_tma_multicast_mask(cluster_layout, cta_coord)
```

### DSMEM

```python
cpasync.CopyDsmemStoreOp()  # Async distributed shared memory store
```

## tcgen05 Submodule (SM100 / Blackwell)

```python
from cutlass.cute.nvgpu import tcgen05
```

### MMA Operations

All MMA constructors require `instruction_shape`, `cta_group`, `a_src`,
`a_major_mode`, `b_major_mode`.

```python
# F16/BF16
mma = tcgen05.MmaF16BF16Op(
    ab_dtype=cutlass.Float16, acc_dtype=cutlass.Float32,
    instruction_shape=(M, N, K),
    cta_group=tcgen05.CtaGroup.TWO,  # or ONE
    a_src=tcgen05.OperandSource.SMEM,
    a_major_mode=tcgen05.OperandMajorMode.K,
    b_major_mode=tcgen05.OperandMajorMode.K,
)

# Other MMA ops (same constructor pattern):
tcgen05.MmaTF32Op(...)       # TensorFloat-32
tcgen05.MmaI8Op(...)         # INT8
tcgen05.MmaFP8Op(...)        # FP8
tcgen05.MmaMXF8Op(...)       # Block-scaled FP8
tcgen05.MmaMXF4Op(...)       # Block-scaled FP4
tcgen05.MmaMXF4NVF4Op(...)   # Block-scaled FP4+NVF4
```

### TMEM Load/Store

```python
# TMEM → Register
ld = tcgen05.Ld16x64bOp(repeat=tcgen05.Repetition.x1, pack=tcgen05.Pack.NONE)
ld = tcgen05.Ld16x128bOp(repeat=tcgen05.Repetition.x1)
ld = tcgen05.Ld16x256bOp(repeat=tcgen05.Repetition.x1)
ld = tcgen05.Ld32x32bOp(repeat=tcgen05.Repetition.x1)

# Register → TMEM
st = tcgen05.St16x128bOp(repeat=tcgen05.Repetition.x1, unpack=tcgen05.Unpack.NONE)
```

Repetition values: `x1, x2, x4, x8, x16, x32, x64, x128`

### Shared Memory Layout

```python
smem_atom = tcgen05.make_smem_layout_atom(
    tcgen05.SmemLayoutAtomKind.MN_SW128, cutlass.Float16)
```

Kinds: `MN_INTER`, `MN_SW32`, `MN_SW64`, `MN_SW128`, `MN_SW128_32B`,
       `K_INTER`, `K_SW32`, `K_SW64`, `K_SW128`

### Utility Functions

```python
tcgen05.tile_to_mma_shape(atom, tile_shape, order)
tcgen05.commit(mbar_ptr, mask, cta_group)
tcgen05.is_tmem_load(atom) / tcgen05.is_tmem_store(atom)
tcgen05.get_tmem_copy_properties(atom)
tcgen05.make_tmem_copy(atom, tmem_tensor)
tcgen05.make_s2t_copy(atom, tmem_tensor)  # SMEM → TMEM
tcgen05.make_umma_smem_desc(src, layout, major, next_src)
```

### Runtime Fields

```python
tcgen05.Field.ACCUMULATE
tcgen05.Field.NEGATE_A
tcgen05.Field.NEGATE_B
tcgen05.Field.SFA
tcgen05.Field.SFB
```
