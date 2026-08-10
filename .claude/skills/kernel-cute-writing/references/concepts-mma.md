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

# MMA Operations & Tensor Cores

## MMA Atom Hierarchy

CuTe abstracts matrix multiply-accumulate hardware through three layers:

1. **Operation Struct**: Wraps a single PTX instruction. Named to encode
   architecture, dimensions, types, and transpose mode.
   Example: `SM70_8x8x4_F32F16F16F32_NT`

2. **MMA_Traits**: Metadata including compute types, shape (M,N,K), and
   thread-value layout mappings for A, B, C matrices.

3. **Atom**: Combines Operation + Traits. Decouples thread/data layouts
   from instruction call sites.

## Thread-Value (TV) Layouts

An MMA atom defines how threads and values map to matrix elements:

```
TV Layout: (Thread, Value) → (M, N)
```

- Thread dimension: Which thread holds which matrix element
- Value dimension: Which register within a thread holds which element

This mapping is architecture-specific and must match the hardware instruction.

## TiledMMA

TiledMMA scales a single atom across larger tiles by replicating it spatially:

```python
# Python DSL
mma_atom = cute.nvgpu.tcgen05.MmaF16BF16Op(
    ab_dtype=cutlass.Float16,
    acc_dtype=cutlass.Float32,
    instruction_shape=(M, N, K),
    cta_group=cute.nvgpu.tcgen05.CtaGroup.ONE,
    a_src=OperandSource.SMEM,
    a_major_mode=OperandMajorMode.K,
    b_major_mode=OperandMajorMode.K,
)
tiled_mma = cute.make_tiled_mma(mma_atom)
```

TiledMMA provides partition methods:
```python
thr_mma = tiled_mma.get_slice(thread_idx)
tCsA = thr_mma.partition_A(sA)  # A operand partition
tCsB = thr_mma.partition_B(sB)  # B operand partition
tCrC = thr_mma.partition_C(gC)  # C accumulator partition
```

## Architecture-Specific MMA

### Ampere (SM80): Warp-Level MMA

Warp-level operations process 32 threads cooperatively.

```python
from cutlass.cute.nvgpu.warp import MmaF16BF16Op

mma = MmaF16BF16Op(
    ab_dtype=cutlass.Float16,
    acc_dtype=cutlass.Float32,
    shape_mnk=(16, 8, 16),
)
```

Supported types: FP16, BF16
Accumulator: FP32

### Hopper (SM90): Warpgroup MMA (WGMMA)

Warpgroup = 4 warps (128 threads). GMMA reads directly from shared memory
for the A operand (no register stage needed for A).

```python
from cutlass.cute.nvgpu.warpgroup import MmaF16BF16Op, MmaF8Op

mma_f16 = MmaF16BF16Op(
    ab_dtype=cutlass.Float16,
    acc_dtype=cutlass.Float32,
    instruction_shape=(64, N, 16),  # N varies: 8-256
    a_src=OperandSource.SMEM,
    a_major_mode=OperandMajorMode.K,
    b_major_mode=OperandMajorMode.K,
)

mma_f8 = MmaF8Op(
    a_dtype=cutlass.Float8E4M3,
    b_dtype=cutlass.Float8E5M2,
    acc_dtype=cutlass.Float32,
    instruction_shape=(64, N, 32),
)
```

Supported types: FP16, BF16, FP8 (E4M3, E5M2)
Shared memory layout atoms via `make_smem_layout_atom()`.

Warpgroup synchronization:
```python
cute.nvgpu.warpgroup.fence()
cute.nvgpu.warpgroup.commit_group()
cute.nvgpu.warpgroup.wait_group(0)
```

### Blackwell (SM100): tcgen05 UMMA

Uses Tensor Memory (TMEM) — a dedicated register file for MMA. Supports
2-CTA instructions for doubled throughput.

```python
from cutlass.cute.nvgpu.tcgen05 import (
    MmaF16BF16Op, MmaFP8Op, MmaTF32Op, MmaI8Op,
    MmaMXF8Op, MmaMXF4Op, MmaMXF4NVF4Op,
    CtaGroup, OperandSource, OperandMajorMode,
)

mma = MmaF16BF16Op(
    ab_dtype=cutlass.Float16,
    acc_dtype=cutlass.Float32,
    instruction_shape=(M, N, K),
    cta_group=CtaGroup.TWO,    # 2-CTA mode
    a_src=OperandSource.SMEM,
    a_major_mode=OperandMajorMode.K,
    b_major_mode=OperandMajorMode.K,
)
```

Supported types: FP16, BF16, TF32, INT8, FP8, MX formats (MXF8, MXF4)

TMEM operations:
```python
from cutlass.cute.nvgpu.tcgen05 import Ld16x128bOp, St16x128bOp

# Load from TMEM to registers
ld_atom = Ld16x128bOp(repeat=Repetition.x1)

# Store from registers to TMEM
st_atom = St16x128bOp(repeat=Repetition.x1)
```

Runtime-modifiable fields: `ACCUMULATE`, `NEGATE_A`, `NEGATE_B`, `SFA`, `SFB`

## Supported Data Types by Architecture

| Architecture | Input Types | Accumulator Types |
|-------------|-------------|-------------------|
| Ampere (SM80) | FP16, BF16 | FP32 |
| Hopper (SM90) | FP16, BF16, FP8 | FP32 |
| Blackwell (SM100) | FP16, BF16, TF32, INT8, FP8, MXF8, MXF4 | FP32, INT32 |

## Shared Memory Layout Atoms

Architecture-specific shared memory layouts for efficient MMA access:

```python
# Hopper
from cutlass.cute.nvgpu.warpgroup import make_smem_layout_atom, SmemLayoutAtomKind
smem_atom = make_smem_layout_atom(SmemLayoutAtomKind.K_SW128, cutlass.Float16)

# Blackwell
from cutlass.cute.nvgpu.tcgen05 import make_smem_layout_atom, SmemLayoutAtomKind
smem_atom = make_smem_layout_atom(SmemLayoutAtomKind.MN_SW128, cutlass.Float16)
```

Swizzle variants: `INTER` (no swizzle), `SW32`, `SW64`, `SW128`, `SW128_32B`
