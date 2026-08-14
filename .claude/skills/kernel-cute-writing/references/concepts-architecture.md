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

# CuTe DSL: Architecture & Core Concepts

## What is CuTe DSL

CuTe DSL is a Python-based domain-specific language for GPU kernel development,
part of CUTLASS 4.x. It provides Python abstractions over CUTLASS C++ templates,
enabling rapid iteration and easier prototyping while maintaining full hardware
control.

CuTe DSL is **not** a replacement for CUTLASS C++. It complements the existing
C++ APIs. The Python DSL and CuTe C++ share an isomorphic programming model —
knowledge transfers between them.

## Compilation Pipeline

```
Python Code → AST Rewrite → Custom IR → MLIR → PTX → SASS (via ptxas)
```

Three-stage process:
1. **Python → IR**: AST preprocessing converts Python control flow to structured
   IR. Arithmetic is traced via proxy objects.
2. **IR → MLIR**: Intermediate representation lowered through MLIR infrastructure.
3. **MLIR → PTX → SASS**: Final code generation via ptxas from CUDA toolkit.

No NVCC or NVRTC required — the `nvidia-cutlass-dsl` wheel contains everything.

## Core Abstractions

### Layouts
A Layout is a `(Shape, Stride)` tuple implementing a mapping from coordinates
to indices. Layouts are hierarchically multidimensional — they can represent
functions beyond simple row/column major. In CuTe, Layout is a first-class
citizen used for both data organization and thread distribution.

### Tensors
A Tensor combines an Engine (iterator/pointer to data) with a Layout (defining
logical coordinates and memory mapping). Tensors abstract data organization
and storage details, enabling uniform algorithms over any memory layout.

### Atoms
Atoms represent fundamental hardware operations:
- **MMA Atoms**: Matrix multiply-accumulate instructions (tensor cores)
- **Copy Atoms**: Memory copy operations (LDG, STS, TMA, cp.async)

Atoms decouple thread/data layouts from instruction call sites.

### Tiled Operations
TiledMMA and TiledCopy scale atoms across thread blocks and warps. They define
how atoms are replicated and distributed to cover larger tile dimensions.

## Architecture Support

CuTe DSL supports NVIDIA GPUs starting with Ampere (SM80):

| Architecture | SM | Key Features |
|-------------|-----|-------------|
| Ampere | SM80 | FP16/BF16 tensor cores, cp.async |
| Hopper | SM90 | WGMMA, TMA, thread block clusters |
| Blackwell | SM100 | tcgen05 UMMA, tensor memory, 2-CTA instructions |

## Key Terminology

| Term | Definition |
|------|-----------|
| **Layout** | (Shape, Stride) tuple mapping coordinates to indices |
| **Tensor** | Pointer + Layout representing multidimensional array |
| **Atom** | Fundamental hardware operation (MMA or Copy) |
| **Fragment** | Register-backed array holding a thread's tile portion |
| **Tile** | Tensor partition with compile-time extents |
| **Residue** | Partial tile requiring predication |
| **Warp** | 32 threads executing in lock-step |
| **Warpgroup** | 4 warps (128 threads) for GMMA on Hopper+ |
| **CTA** | Cooperative Thread Array (thread block) |
| **TMA** | Tensor Memory Accelerator (Hopper+ hardware unit) |
| **TMEM** | Tensor Memory (Blackwell register file for MMA) |
| **Cosize** | Physical memory footprint of a layout |
| **Swizzle** | Bit permutation to avoid shared memory bank conflicts |

## Relationship to CUTLASS C++

- **CUTLASS 2.x/3.x**: C++ template APIs for GEMM and convolution. Still
  maintained and receiving updates.
- **CuTe C++**: Low-level C++ library for layouts, tensors, and algorithms.
  Foundation for CUTLASS 3.x kernels.
- **CuTe DSL**: Python DSL fully isomorphic with CuTe C++. Same concepts
  (layouts, tensors, atoms) expressed in Python with JIT compilation.
- **CUTLASS Python** (deprecated): Old Python interface for instantiating
  C++ kernels. Replaced by CuTe DSL.

## System Requirements

- Linux x86_64 only (no Windows support)
- Python 3.10–3.13
- CUDA driver compatible with Toolkit 12.9+ (driver ≥ 575.51.03)
- NVIDIA GPU with SM80+ (Ampere or newer)
