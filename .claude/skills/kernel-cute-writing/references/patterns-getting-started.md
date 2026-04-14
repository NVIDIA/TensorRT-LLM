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

# Getting Started with CuTe DSL

## Installation

### From PyPI (Stable)
```bash
# CUDA Toolkit 12.9
pip install nvidia-cutlass-dsl

# CUDA Toolkit 13.1
pip install nvidia-cutlass-dsl[cu13]
```

### From GitHub (Latest)
```bash
git clone https://github.com/NVIDIA/cutlass.git

# CUDA 12.9
./cutlass/python/CuTeDSL/setup.sh --cu12

# CUDA 13.1
./cutlass/python/CuTeDSL/setup.sh --cu13
```

### Remove Previous Installations
```bash
pip uninstall nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base nvidia-cutlass-dsl-libs-cu13 -y
```

### Optional Packages
```bash
pip install torch jupyter
pip install apache-tvm-ffi           # For TVM FFI compilation
pip install torch-c-dlpack-ext       # Faster torch tensor handling
```

### Environment
```bash
export PYTHONUNBUFFERED=1            # For Jupyter notebooks
export CUTE_DSL_CACHE_DIR=/path/     # Persistent JIT cache
```

## Primary Decorators

### @cute.jit — Host Function
Declares a JIT-compiled function callable from Python. Runs on host,
launches kernels, performs tensor setup.

```python
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.jit
def host_fn(mA: cute.Tensor, mC: cute.Tensor):
    # Compute tiling, launch kernel
    vec_size = 16 // (mA.element_type.width // 8)
    tiler = (1, vec_size)
    gA = cute.zipped_divide(mA, tiler)
    gC = cute.zipped_divide(mC, tiler)

    num_threads = 256
    num_tiles = cute.size(gA.shape[1])
    num_blocks = (num_tiles + num_threads - 1) // num_threads

    my_kernel(gA, gC).launch(
        grid=(num_blocks, 1, 1),
        block=(num_threads, 1, 1),
    )
```

Parameters:
- `preprocessor` (default `True`): Enable AST rewrite for control flow
- `no_cache` (default `False`): Force fresh compilation each call

### @cute.kernel — Device Function
Declares a GPU kernel. Cannot be called directly from Python — must be
launched from a `@cute.jit` function.

```python
@cute.kernel
def my_kernel(gA: cute.Tensor, gC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx
    m, n = gA.shape[1]
    ni = thread_idx % n
    mi = thread_idx // n
    total_tiles = m * n

    if cutlass.dynamic_expr(thread_idx < total_tiles):
        a_val = gA[(None, (mi, ni))].load()
        c_val = a_val + a_val
        gC[(None, (mi, ni))] = c_val
```

## Calling Conventions

| Caller → Callee | Allowed | Notes |
|-----------------|---------|-------|
| Python → @jit | Yes | Entry point |
| Python → @kernel | **No** | RuntimeError |
| @jit → @jit | Yes | Inlined at compile time |
| @jit → @kernel | Yes | GPU kernel launch |
| @kernel → @kernel | **No** | Not supported |

## Type Annotations

```python
import cutlass

@cute.jit
def foo(
    tensor: cute.Tensor,              # Dynamic argument (in JIT signature)
    dtype: cutlass.Constexpr,         # Compile-time constant (not in signature)
    n: cutlass.Constexpr[int],        # Typed compile-time constant
):
    ...
```

- **Dynamic arguments** (default): Runtime values, included in JIT signature
- **Constexpr arguments**: Compile-time constants, baked into generated code

## First Kernel: Element-wise Double

```python
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def double_kernel(gA: cute.Tensor, gC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    thread_idx = bidx * bdim + tidx
    m, n = gA.shape[1]
    total = m * n
    ni = thread_idx % n
    mi = thread_idx // n
    if cutlass.dynamic_expr(thread_idx < total):
        val = gA[(None, (mi, ni))].load()
        gC[(None, (mi, ni))] = val + val

@cute.jit
def double_host(mA: cute.Tensor, mC: cute.Tensor):
    vec_size = 16 // (mA.element_type.width // 8)
    tiler = (1, vec_size)
    gA = cute.zipped_divide(mA, tiler)
    gC = cute.zipped_divide(mC, tiler)
    threads = 256
    tiles = cute.size(gA.shape[1])
    blocks = (tiles + threads - 1) // threads
    double_kernel(gA, gC).launch(grid=(blocks,1,1), block=(threads,1,1))

# Usage
x = torch.randn(1024, 512, dtype=torch.float16, device="cuda")
out = torch.empty_like(x)
double_host(from_dlpack(x, assumed_align=16), from_dlpack(out, assumed_align=16))
torch.testing.assert_close(out, x * 2, rtol=1e-3, atol=1e-3)
```

## Compile for Reuse

```python
# Pre-compile to avoid JIT overhead on subsequent calls
compiled = cute.compile(double_host, from_dlpack(x, assumed_align=16),
                        from_dlpack(out, assumed_align=16))
compiled(from_dlpack(x, assumed_align=16), from_dlpack(out, assumed_align=16))
```

## Key Imports

```python
import cutlass                        # Top-level: types, Constexpr, dynamic_expr
import cutlass.cute as cute           # Core DSL: jit, kernel, layout ops, math
from cutlass.cute.runtime import from_dlpack, make_ptr  # Tensor conversion
from cutlass.cute import arch         # Thread/block indexing, sync, memory
from cutlass.cute import nvgpu        # GPU-specific MMA/copy atoms
```
