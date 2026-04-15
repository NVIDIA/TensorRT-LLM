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

# GEMM Kernel Patterns

## Three-Level Tiling Strategy

GEMM kernels partition work across three concurrency levels:

### Level 1: CTA (Thread Block) Tiling

Distribute M×N output tiles across CTAs. Each CTA handles a
`bM × bN × bK` tile from global memory.

```python
cta_tiler = (bM, bN, bK)  # e.g., (128, 128, 64)
# blockIdx.x → M tile, blockIdx.y → N tile
# K dimension iterated in mainloop
```

Larger tiles → fewer global memory fetches → better DRAM utilization.
But oversized tiles leave threads idle for small problems.

### Level 2: Copy Partitioning (Global → Shared Memory)

Thread layouts partition CTA tiles for efficient data movement.

```python
# Thread layout for copying A: 32 threads × 8 elements each
thread_layout_A = cute.make_layout((32, 8))

# Partition global tile among threads
thr_A = cute.local_partition(gA, thread_layout_A, thread_idx)
thr_sA = cute.local_partition(sA, thread_layout_A, thread_idx)

# Copy: each thread loads its subtensor
cute.copy(copy_atom, thr_A, thr_sA)
```

### Level 3: Compute Partitioning (Shared → Registers)

Separate partitioning for MMA operations (different from copy layout):

```python
thr_mma = tiled_mma.get_slice(thread_idx)
tCsA = thr_mma.partition_A(sA)   # A from shared memory
tCsB = thr_mma.partition_B(sB)   # B from shared memory
tCrC = thr_mma.partition_C(gC)   # C accumulator in registers
```

Using distinct layouts for copy vs compute preserves logical consistency.

## Basic Mainloop Structure

```python
# Pseudocode — iterate over K tiles
for k_tile in range(K_TILES):
    # 1. Copy global → shared (using copy partitioning)
    cute.copy(copy_atom_A, thr_gA[:, :, k_tile], thr_sA)
    cute.copy(copy_atom_B, thr_gB[:, :, k_tile], thr_sB)

    # 2. Synchronize
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()

    # 3. MMA shared → registers (using compute partitioning)
    cute.gemm(tiled_mma, tCsA, tCsB, tCrC)

    cute.arch.sync_threads()
```

## Shared Memory Layout

Pad strides to avoid bank conflicts:

```python
# Column-major with padding: stride = bM + 1
smem_layout_A = cute.make_layout((bM, bK), (1, bM + 1))

# Or use swizzled layouts for better patterns
smem_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
    SmemLayoutAtomKind.K_SW128, cutlass.Float16)
```

Shared memory allocation:
```python
smem_A = cute.arch.alloc_smem(cutlass.Float16, smem_layout_A)
```

## TiledCopy for Efficient Data Movement

```python
# Specify copy instruction, thread layout, and values per thread
copy_atom_A = cute.make_tiled_copy(
    CopyAtom(UniversalCopy_uint128_t, element_type),
    thread_layout,    # e.g., (32, 8)
    value_layout,     # e.g., (4, 1) — 4 elements per thread
)
thr_copy = copy_atom_A.get_slice(thread_idx)
src_tensor = thr_copy.partition_S(global_A)
dst_tensor = thr_copy.partition_D(shared_A)
cute.copy(copy_atom_A, src_tensor, dst_tensor)
```

## Software Pipelining

Double-buffer shared memory to overlap computation with data loading:

```python
# Stage 0: prefetch first tile
cute.copy(copy_atom, thr_gA[:,:,0], thr_sA[0])
cute.copy(copy_atom, thr_gB[:,:,0], thr_sB[0])

for k in range(1, K_TILES):
    # Load next tile into alternate buffer
    cute.copy(copy_atom, thr_gA[:,:,k], thr_sA[k % 2])

    # Compute on current buffer
    cute.arch.sync_threads()
    cute.gemm(tiled_mma, tCsA[1-k%2], tCsB[1-k%2], tCrC)
    cute.arch.sync_threads()
```

For Hopper+, use `PipelineTmaAsync` for multi-stage pipelining:
```python
pipeline = PipelineTmaAsync.create(
    num_stages=num_stages,
    producer_group=producer_cg,
    consumer_group=consumer_cg,
    barrier_storage=smem_barrier,
)
```

## Split-K Parallelization

Divide K dimension across multiple CTAs. Each CTA computes a partial sum,
then a reduction kernel combines results.

```
Problem: M=128, N=128, K=4096, split=16
→ 16 batched GEMMs of M=128, N=128, K=256
→ Reduction kernel sums 16 partial results
```

Useful when M×N is too small to fill the GPU but K is large.

## Accumulator Initialization

Zero-initialize the register accumulator **before** the mainloop.
There is no `cute.fill()` — use `cute.zeros_like` or `cute.full_like`:

```python
# Create zero-initialized accumulator (preferred)
tCrC = cute.zeros_like(tCrC)

# Or fill with a specific value
tCrC = cute.full_like(tCrC, 0.0)

# Loop-based alternative (works in all contexts)
for i in range(cute.size(tCrC)):
    tCrC[i] = cutlass.Float32(0.0)
```

## Epilogue

After accumulation, write results to global memory:

```python
# Predicated store for boundary tiles
for i in range(cute.size(tCrC)):
    if elem_less(tCcC[i], shape(mC)):
        tCgC[i] = alpha * tCrC[i] + beta * tCgC[i]
```

### Fused Epilogue (Bias + Activation)

Fuse bias addition and activation into the epilogue to avoid an extra
global memory round-trip. Apply element-wise ops in registers before
the predicated store:

```python
# Load bias vector (shared across M rows, one value per N column)
# bias shape: (N,) — broadcast across rows
for i in range(cute.size(tCrC)):
    if elem_less(tCcC[i], shape(mC)):
        # Get the column index for bias lookup
        col_idx = tCcC[i][1]  # N dimension
        val = alpha * tCrC[i] + tCgBias[col_idx]  # GEMM + bias
        # Apply activation in registers (e.g., GELU)
        val = val * cutlass.Float32(0.5) * (
            cutlass.Float32(1.0) + cute.math.erf(
                val * cutlass.Float32(0.7071067811865476)
            )
        )
        tCgC[i] = val
```

This pattern avoids writing the intermediate GEMM result to global memory
and reading it back for the activation pass.

## Pre-compilation for GEMM (`cute.compile` + `mark_layout_dynamic`)

GEMM tensors are 3D `(mode0, mode1, L)` with specific memory layouts.
Use `mark_layout_dynamic` + `mark_compact_shape_dynamic` so a single
compilation handles arbitrary problem sizes:

```python
# A: (M, K, L), M-major (stride-1 on dim 0, column-major)
# B: (N, K, L), N-major (stride-1 on dim 0)
# C: (M, N, L), N-major (stride-1 on dim 1, row-major)

div = 128 // AB_DTYPE.width  # 8 for fp16, 4 for fp32

def _mark_gemm_tensor(torch_tensor, leading_dim, stride_order):
    """Mark a 3D GEMM tensor for dynamic compilation."""
    return (
        from_dlpack(torch_tensor, assumed_align=16)
        .mark_layout_dynamic(leading_dim=leading_dim)
        .mark_compact_shape_dynamic(
            mode=leading_dim,
            stride_order=stride_order,
            divisibility=div,
        )
    )

# Fake tensors for compilation (small — shape doesn't matter)
fake_A = torch.empty(8, 8, 1, dtype=torch.float16, device="cuda")
    .permute(2, 1, 0)                   # M-major physical layout
fake_B = torch.empty(8, 8, 1, dtype=torch.float16, device="cuda")
    .permute(2, 1, 0)                   # N-major physical layout
fake_C = torch.empty(8, 8, 1, dtype=torch.float16, device="cuda")

mA = _mark_gemm_tensor(fake_A, leading_dim=0, stride_order=(2, 1, 0))
mB = _mark_gemm_tensor(fake_B, leading_dim=0, stride_order=(2, 1, 0))
mC = _mark_gemm_tensor(fake_C, leading_dim=1, stride_order=(2, 0, 1))

compiled = cute.compile(gemm_host, mA, mB, mC)
```

**Key rules:**
- `leading_dim` = the dimension with stride 1 (0 for column-major, 1 for row-major)
- `stride_order` = physical dimension ordering (outermost → innermost)
- `divisibility` = alignment in elements (128 bits / dtype width)
- Use `fake_tensors` ≥ 8 elements per dim to avoid alignment inference issues

At runtime, mark the actual tensors the same way before calling `compiled()`.

## Autotuning Search Space

Key tunable parameters for GEMM kernels:

| Parameter | Description | Typical Values |
|-----------|-------------|---------------|
| `mma_tiler_mn` | MMA tile dimensions | (128,128), (128,256), (256,128) |
| `cluster_shape_mn` | CTAs per cluster | (1,1), (2,1), (1,2), (2,2) |
| `num_stages` | Pipeline depth | 2–8 |
| `use_2cta_instrs` | Blackwell 2-CTA mode | True/False |
| `use_tma_store` | TMA for epilogue | True/False |

### Autotuning Best Practices

- 5–10 warmup iterations for GPU stabilization
- 100–1000 timed iterations for statistical validity
- Use CUDA events with synchronization for timing
- Lock GPU clocks via `nvidia-smi`
- Cache kernels by config key to avoid recompilation:

```python
cache_key = f"{dtype}x{acc_dtype}x{mma_tiler}x{cluster}x{stages}"
if cache_key not in kernel_cache:
    kernel_cache[cache_key] = cute.compile(kernel_fn, ...)
```

## Naming Convention

Pattern `tCsA` means "partitioning pattern tC applied to tensor sA":
- First letter: `t` (thread-partitioned)
- Second letter: partitioning scheme (C for compute, A for copy-A)
- Remaining: source tensor (sA = shared A, gA = global A, rC = register C)
