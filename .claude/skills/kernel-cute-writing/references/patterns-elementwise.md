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

# Element-wise Kernel Patterns

## Invariant Principles

These four principles are **non-negotiable** — apply all to every kernel.

### P1: Alignment Hints for Vectorized Memory Access

```python
from cutlass.cute.runtime import from_dlpack

# Always specify assumed_align to enable vector loads/stores
mA = from_dlpack(tensor, assumed_align=16)  # LDG.128/STG.128
mA = from_dlpack(tensor, assumed_align=32)  # LDG.256/STG.256 (Blackwell)
```

Without alignment hints, compiler generates scalar loads (8x LDG.U16)
instead of vector loads (1x LDG.128).

### P2: Compile-time vec_size from Element Type

```python
@cute.jit
def host_fn(mA, mC):
    vec_size = load_bytes // (mA.element_type.width // 8)  # width is bits
```

Derive `vec_size` from tensor element type — no extra dtype parameter.
This keeps the host function signature to tensors only (required for
`cute.compile`).

### P3: Use zipped_divide for Coalesced Access

```python
tiler = (..., vec_size)  # vec_size on the vectorized dimension
gA = cute.zipped_divide(mA, tiler)
```

### P4: Bounds Checking with cutlass.dynamic_expr

```python
if cutlass.dynamic_expr(thread_idx < total_tiles):
    # ... load, compute, store
```

## Critical Rules

### No Early Return in @cute.kernel

```python
# WRONG — DSLAstPreprocessorError
@cute.kernel
def kernel(gA, gC):
    if thread_idx >= total_tiles:
        return  # ERROR

# CORRECT — predicated execution
@cute.kernel
def kernel(gA, gC):
    if cutlass.dynamic_expr(thread_idx < total_tiles):
        gC[...] = gA[...].load() + gA[...].load()
```

### Scalar Multiplication Type Promotion

Multiplying by Python int promotes FP16 to FP32:

```python
# WRONG — type mismatch
a_val = gA[...].load()  # Float16
c_val = a_val * 2       # Promotes to Float32!
gC[...] = c_val         # ERROR: Float32 → Float16

# CORRECT — use addition
c_val = a_val + a_val   # Stays Float16

# CORRECT — explicit cast
c_val = (a_val * scalar).to(cutlass.Float16)
```

### No cute.math.sigmoid

```python
# WRONG
sigmoid_val = cute.math.sigmoid(x)  # AttributeError

# CORRECT
sigmoid_val = 1.0 / (1.0 + cute.math.exp(-x))
```

## Pattern Variations

### Variation A: Unary Op (1 input → 1 output)

```python
@cute.kernel
def unary_kernel(gA, gC):
    # ... index calculation ...
    a_val = gA[idx].load()
    c_val = cute.math.exp(a_val)
    gC[idx] = c_val
```

### Variation B: Binary Op (2 inputs → 1 output)

```python
@cute.kernel
def binary_kernel(gA, gB, gC):
    a_val = gA[idx].load()
    b_val = gB[idx].load()
    c_val = a_val + b_val
    gC[idx] = c_val
```

### Variation C: In-place Op

```python
@cute.kernel
def inplace_kernel(gA):
    a_val = gA[idx].load()
    gA[idx] = a_val + a_val
```

### Variation D: 1D Tensor

```python
@cute.jit
def host_1d(mA, mC):
    vec_size = 16 // (mA.element_type.width // 8)
    tiler = (vec_size,)  # 1D tiler
    gA = cute.zipped_divide(mA, tiler)
```

### Variation E: 3D/Batched Tensor

```python
@cute.jit
def host_3d(mA, mC):
    vec_size = 16 // (mA.element_type.width // 8)
    tiler = (1, 1, vec_size)  # Vectorize last dimension
    gA = cute.zipped_divide(mA, tiler)
```

## Complete Reference: 2D Unary Op

```python
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def elementwise_kernel(gA: cute.Tensor, gC: cute.Tensor):
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
        c_val = a_val + a_val  # Replace with your operation
        gC[(None, (mi, ni))] = c_val

@cute.jit
def elementwise_host(mA: cute.Tensor, mC: cute.Tensor):
    load_bytes = 16  # 32 for Blackwell
    vec_size = load_bytes // (mA.element_type.width // 8)
    tiler = (1, vec_size)
    gA = cute.zipped_divide(mA, tiler)
    gC = cute.zipped_divide(mC, tiler)
    num_threads = 256
    num_tiles = cute.size(gA.shape[1])
    num_blocks = (num_tiles + num_threads - 1) // num_threads
    elementwise_kernel(gA, gC).launch(
        grid=(num_blocks, 1, 1), block=(num_threads, 1, 1))

def run_kernel(input_t: torch.Tensor) -> torch.Tensor:
    output_t = torch.empty_like(input_t)
    elementwise_host(
        from_dlpack(input_t, assumed_align=16),
        from_dlpack(output_t, assumed_align=16),
    )
    return output_t
```

## Common Operations

| Operation | Implementation | Notes |
|-----------|---------------|-------|
| ReLU | `cute.where(a > 0, a, 0)` | |
| GELU | Use `cute.math` functions | Approximation or exact |
| Tanh | `cute.math.tanh(a)` | |
| Exp | `cute.math.exp(a)` | |
| Sigmoid | `1.0 / (1.0 + cute.math.exp(-x))` | No `cute.math.sigmoid` |
| SiLU | `a * (1.0/(1.0+cute.math.exp(-a)))` | Fused sigmoid × input |
| Add | `a + b` | Binary |
| Mul | `a * b` | Binary |

Available `cute.math` functions: `exp`, `log`, `tanh`, `sin`, `cos`,
`rsqrt`, `sqrt`, `erf`

## Vector Size Reference

| Instruction | Bytes/Thread | FP16/BF16 vec | FP32 vec | Alignment |
|-------------|-------------|---------------|----------|-----------|
| LDG.128 | 16 | 8 | 4 | 16 |
| LDG.256 | 32 | 16 | 8 | 32 |

## Test Harness Template

```python
import torch
from kernel import run_kernel

def test_correctness():
    M, N = 1024, 512
    x = torch.randn((M, N), dtype=torch.float16, device="cuda")
    output = run_kernel(x)
    expected = x * 2  # Replace with reference
    torch.testing.assert_close(output, expected, rtol=1e-3, atol=1e-3)
```

## Benchmark Template

Use `cute.compile()` in setup to pre-compile once. Without it, kernels
recompile every iteration, producing inaccurate timing.

```python
def setup():
    import torch, cutlass.cute as cute
    from kernel import host_fn
    x = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
    out = torch.empty_like(x)
    x_cute = cute.runtime.from_dlpack(x, enable_tvm_ffi=True).mark_layout_dynamic()
    out_cute = cute.runtime.from_dlpack(out, enable_tvm_ffi=True).mark_layout_dynamic()
    compiled = cute.compile(host_fn, x_cute, out_cute, options="--enable-tvm-ffi")
    return compiled, (x, out)

def setup_ref():
    import torch
    x = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
    return lambda x: x * 2, (x,)  # Replace with your reference op
```

Then run the benchmark script:
```bash
python scripts/benchmark_kernel.py --script {output_dir}/bench_kernel.py
```

## Adaptation Checklist

| Question | Affects |
|----------|---------|
| How many input tensors? | Parameters, `.load()` calls |
| How many output tensors? | Store operations |
| Tensor rank? (1D, 2D, 3D) | Tiler shape, index calculation |
| In-place? | Same tensor for input/output |
| Which dimension to vectorize? | `vec_size` position in tiler |
| Target arch? | `load_bytes` (16 or 32), alignment |
