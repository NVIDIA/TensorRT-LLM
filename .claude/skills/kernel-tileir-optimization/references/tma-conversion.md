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

# TMA Descriptor Conversion for Dot-Related Kernels

TMA (Tensor Memory Accelerator) descriptors are MANDATORY for dot-related kernels
on TileIR. They replace manual pointer arithmetic with hardware-accelerated loads.

## Conversion Steps

### Step 1: Import TensorDescriptor

```python
from triton.tools.tensor_descriptor import TensorDescriptor
```

Requires nvtriton installed as `triton`.

### Step 2: Create Descriptors in Wrapper

Use dummy block sizes (updated dynamically by pre-hook):

```python
a_desc = TensorDescriptor.from_tensor(a, [1, 1])
b_desc = TensorDescriptor.from_tensor(b, [1, 1])
c_desc = TensorDescriptor.from_tensor(c, [1, 1])
```

### Step 3: Add Pre-Hook for Dynamic Block Sizes

```python
def tma_set_block_size_hook(nargs):
    BLOCK_M = nargs["BLOCK_SIZE_M"]
    BLOCK_N = nargs["BLOCK_SIZE_N"]
    BLOCK_K = nargs["BLOCK_SIZE_K"]

    nargs["a_desc"].block_shape = [BLOCK_M, BLOCK_K]
    nargs["b_desc"].block_shape = [BLOCK_N, BLOCK_K]  # Note: transposed
    nargs["c_desc"].block_shape = [BLOCK_M, BLOCK_N]

    # Prevent oversubscription with 2CTA
    if "NUM_SMS" in nargs and "NUM_CTAS" in nargs:
        nargs["NUM_SMS"] = nargs["NUM_SMS"] // nargs["NUM_CTAS"]
```

### Step 4: Replace Pointer Arithmetic with Descriptor Loads

Before (pointer arithmetic):

```python
a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
a = tl.load(a_ptrs, mask=mask, other=0.0)
```

After (TMA descriptor):

```python
a = a_desc.load([offs_am, offs_k])
```

## Key Differences from Pointer-Based Access

- No manual pointer arithmetic
- No explicit masking (TMA handles bounds checking)
- Descriptors passed to kernel instead of pointers + strides
- Block sizes configured dynamically via pre-hook

## Pre-Hook Wiring

Always pass `pre_hook` when creating autotune configs:

```python
@triton.autotune(
    configs=get_configs(pre_hook=tma_set_block_size_hook),  # Required!
    key=["M", "N", "K"]
)
```

Without the pre-hook, TMA descriptors retain dummy `[1, 1]` block sizes,
causing runtime errors or silently incorrect results.

## Matrix Transposition

For GEMM kernels:
- In wrapper: pass `b.T.contiguous()` if B is row-major
- In kernel: use `tl.dot(a, b.T, accumulator)`
- Mismatch between descriptor layout and kernel access produces wrong results silently

## References

- [NV Triton Repository](https://github.com/triton-lang/Triton-to-tile-IR)
- [TileIR Performance Tuning Tips](https://github.com/triton-lang/Triton-to-tile-IR/blob/main/third_party/tileir/PerformanceTuningTips.md)
- Triton Tutorials: `09-persistent-matmul.py` for TMA patterns
