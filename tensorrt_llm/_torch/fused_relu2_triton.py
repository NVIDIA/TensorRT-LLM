# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-kernel ``square(relu(x))``.

The eager form runs two elementwise passes over the activation, so it reads and
writes the tensor twice. Models with a large relu2 MLP (Nemotron-H's shared
expert is 32768 x 3712 bf16 at ISL 32k) spend a measurable slice of the forward
there purely on the extra round trip. Fusing halves the traffic; the result is
bit-identical because the squaring still happens in fp32 before the store, which
is what ``torch.square(F.relu(x))`` does for a half-precision input.
"""

import torch
import triton
import triton.language as tl

# One 128-bit access per thread at 8 warps; measured at ~97% of achievable copy
# bandwidth on SM103, so there is nothing to gain from autotuning here (and a
# fixed launch config stays CUDA-graph capturable).
_BLOCK = 4096
_NUM_WARPS = 8

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


@triton.jit
def _relu2_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.maximum(x, 0.0)
    tl.store(out_ptr + offs, (r * r).to(out_ptr.dtype.element_ty), mask=mask)


def is_eligible(x: torch.Tensor) -> bool:
    return x.is_cuda and x.dtype in _SUPPORTED_DTYPES and x.is_contiguous() and x.numel() > 0


def fused_relu2(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """``square(relu(x))`` in one pass. Caller must check ``is_eligible``."""
    out = torch.empty_like(x) if out is None else out
    n = x.numel()
    _relu2_kernel[(triton.cdiv(n, _BLOCK),)](x, out, n, BLOCK=_BLOCK, num_warps=_NUM_WARPS)
    return out
