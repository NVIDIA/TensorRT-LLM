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

# Reduction Kernel Patterns

Row-wise reduction kernels (softmax, layernorm, RMSNorm) use a scalar-per-thread
loop with warp shuffle + shared memory to reduce across all threads.

**Do NOT use `zipped_divide`** for reductions — it is an element-wise pattern.
Reduction kernels use scalar element access `gX[(row, col)]` with a strided loop.

## Architecture: One CTA per Row

```python
NUM_THREADS = 256
WARP_SIZE = 32
NUM_WARPS = NUM_THREADS // WARP_SIZE  # 8

@cute.kernel
def reduction_kernel(gX: cute.Tensor, gOut: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    N = gX.shape[1]

    # Shared memory for cross-warp reduction
    smem_ptr = cute.arch.alloc_smem(cutlass.Float32, NUM_WARPS + 1)
    smem = cute.make_tensor(smem_ptr, cute.make_layout((NUM_WARPS + 1,)))

    warp_id = tidx // WARP_SIZE
    lane_id = tidx % WARP_SIZE

    # Each thread loops at stride NUM_THREADS
    for col in range(tidx, N, NUM_THREADS):
        val = gX[(row, col)]
        ...
```

Grid: `(M, 1, 1)` — one CTA per row. Block: `(NUM_THREADS, 1, 1)`.
This works with **any dtype** (float16, float32) and **any N** (no alignment
constraints).

## Warp-Level Reduction

### Preferred: Built-in `warp_reduction_max` / `warp_reduction_sum`

These wrap `shuffle_sync_bfly` internally and are simpler to use:

```python
# Per-thread local max
local_max = cutlass.Float32(-1e30)
for col in range(tidx, N, NUM_THREADS):
    val = gX[(row, col)]
    local_max = cute.arch.fmax(val, local_max)

# Single call reduces across the warp
local_max = cute.arch.warp_reduction_max(local_max)
```

```python
# Per-thread local sum
local_sum = cutlass.Float32(0.0)
for col in range(tidx, N, NUM_THREADS):
    local_sum = local_sum + gX[(row, col)]

local_sum = cute.arch.warp_reduction_sum(local_sum)
```

### Alternative: Manual Butterfly Shuffle

If `warp_reduction_max/sum` is unavailable, use `shuffle_sync_bfly` directly.
**Do NOT use `warp_redux_sync("fmax")`** — `redux.f32` is not supported
on SM90 for float max/min operations.

```python
from cutlass.cute.math import arith

# Max: use arith.maxnumf (not cute.math.max which does not exist)
for offset in [16, 8, 4, 2, 1]:
    other = cute.arch.shuffle_sync_bfly(local_max, offset, 0xFFFFFFFF)
    local_max = arith.maxnumf(local_max, other)

# Sum: use addition
for offset in [16, 8, 4, 2, 1]:
    other = cute.arch.shuffle_sync_bfly(local_sum, offset, 0xFFFFFFFF)
    local_sum = local_sum + other
```

## Cross-Warp Reduction via Shared Memory

After warp-level reduction, lane 0 of each warp writes to shared memory,
then thread 0 reduces across warps and broadcasts the result:

```python
# Lane 0 of each warp writes partial result
if lane_id == 0:
    smem[(warp_id,)] = local_val

cute.arch.sync_threads()

# Thread 0 reduces across warps and broadcasts
if tidx == 0:
    result = smem[(0,)]
    for w in range(1, NUM_WARPS):
        wval = smem[(w,)]
        result = result + wval   # or arith.maxnumf for max
    smem[(NUM_WARPS,)] = result  # broadcast slot

cute.arch.sync_threads()

# All threads read the broadcast result
final_val = smem[(NUM_WARPS,)]
```

Shared memory layout: `(NUM_WARPS + 1,)` — one slot per warp + one broadcast
slot. Total: `(NUM_WARPS + 1) * 4` bytes.

## Complete Patterns

### Softmax (3-pass: max → exp+sum → normalize)

```python
# Pass 1: row max (see "Max Reduction" above)
# Pass 2: exp(x - max) and sum
local_sum = cutlass.Float32(0.0)
for col in range(tidx, N, NUM_THREADS):
    val = gX[(row, col)]
    exp_val = cute.math.exp(val - row_max)
    local_sum = local_sum + exp_val
# ... warp + cross-warp sum reduction ...

# Pass 3: normalize
inv_sum = 1.0 / row_sum
for col in range(tidx, N, NUM_THREADS):
    val = gX[(row, col)]
    exp_val = cute.math.exp(val - row_max)
    gOut[(row, col)] = exp_val * inv_sum
```

### LayerNorm (3-pass: mean → variance → normalize)

```python
# Pass 1: sum for mean
local_sum = cutlass.Float32(0.0)
for col in range(tidx, N, NUM_THREADS):
    local_sum = local_sum + gX[(row, col)]
# ... warp + cross-warp sum reduction ...
mean = row_sum / cutlass.Float32(N)

# Pass 2: sum of squared differences for variance
local_var = cutlass.Float32(0.0)
for col in range(tidx, N, NUM_THREADS):
    diff = gX[(row, col)] - mean
    local_var = local_var + diff * diff
# ... warp + cross-warp sum reduction ...
var = var_sum / cutlass.Float32(N)
rstd = cute.math.rsqrt(var + cutlass.Float32(eps))

# Pass 3: normalize with weight and bias
for col in range(tidx, N, NUM_THREADS):
    x_norm = (gX[(row, col)] - mean) * rstd
    gOut[(row, col)] = x_norm * gW[(col,)] + gB[(col,)]
```

### RMSNorm (2-pass: sum-of-squares → normalize)

```python
# Pass 1: sum of squares
local_ss = cutlass.Float32(0.0)
for col in range(tidx, N, NUM_THREADS):
    val = gX[(row, col)]
    local_ss = local_ss + val * val
# ... warp + cross-warp sum reduction ...
rstd = cute.math.rsqrt(ss_sum / cutlass.Float32(N) + cutlass.Float32(eps))

# Pass 2: normalize with weight
for col in range(tidx, N, NUM_THREADS):
    gOut[(row, col)] = gX[(row, col)] * rstd * gW[(col,)]
```

## Host Wrapper and Compilation

Use `cute.compile()` to pre-compile the kernel once. Without it, `@cute.jit`
recompiles on every call (~20-50ms overhead), which dominates runtime for
fast kernels.

```python
@cute.jit
def reduction_host(mX: cute.Tensor, mOut: cute.Tensor):
    M = mX.shape[0]
    reduction_kernel(mX, mOut).launch(
        grid=(M, 1, 1),
        block=(NUM_THREADS, 1, 1),
    )

# Pre-compile once — cache by column count for dynamic shapes
_compile_cache: dict = {}

def kernel_fn(x: torch.Tensor) -> torch.Tensor:
    N = x.shape[1]
    if N not in _compile_cache:
        fake_x = torch.empty(1, N, dtype=x.dtype, device=x.device)
        fake_out = torch.empty_like(fake_x)
        _compile_cache[N] = cute.compile(
            reduction_host,
            from_dlpack(fake_x, assumed_align=16),
            from_dlpack(fake_out, assumed_align=16),
        )
    out = torch.empty_like(x)
    _compile_cache[N](
        from_dlpack(x, assumed_align=16),
        from_dlpack(out, assumed_align=16),
    )
    return out
```

## Known Limitations

- **`warp_redux_sync` with `fmax`/`fmin`:** Not supported on SM90 for float32.
  Use `shuffle_sync_bfly` + `arith.maxnumf` instead.
- **`cute.math.max` does not exist.** Use `arith.maxnumf` from
  `cutlass.cute.math.arith`.
- **`cta_norm.py` example is fp16-optimized.** Its vectorized `elems_per_thread=8`
  assumes 16-bit elements (128-bit loads / 16-bit = 8). For fp32, this breaks
  the tiled_copy layout. Use the scalar-loop pattern above for fp32 reductions
  instead of adapting `cta_norm.py`.
