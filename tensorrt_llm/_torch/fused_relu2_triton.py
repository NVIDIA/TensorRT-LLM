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
    # tl.maximum defaults to PropagateNan.NONE, which returns 0.0 for a NaN input
    # where torch.relu propagates it. Ask for the eager behaviour explicitly.
    r = tl.maximum(x, 0.0, propagate_nan=tl.PropagateNan.ALL)
    tl.store(out_ptr + offs, (r * r).to(out_ptr.dtype.element_ty), mask=mask)


def is_eligible(x: torch.Tensor) -> bool:
    """Whether ``fused_relu2`` may stand in for the eager form on this tensor.

    True iff ``x`` is a non-empty contiguous CUDA tensor of bf16/fp16/fp32;
    anything else stays on the eager path.
    """
    return x.is_cuda and x.dtype in _SUPPORTED_DTYPES and x.is_contiguous() and x.numel() > 0


def fused_relu2(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """``square(relu(x))`` in one pass. Caller must check ``is_eligible``.

    ``out`` is indexed linearly over ``x.numel()`` elements, so anything but a
    matching contiguous buffer is a memory error rather than a wrong result:
    too small overruns the allocation, and a strided view lands the values in
    the wrong logical positions. Check it here rather than trusting the caller.
    """
    if out is None:
        out = torch.empty_like(x)
    else:
        if (
            out.shape != x.shape
            or out.dtype != x.dtype
            or out.device != x.device
            or not out.is_contiguous()
        ):
            raise ValueError(
                "fused_relu2 out must be a contiguous tensor matching x's shape, dtype and "
                f"device; got shape={tuple(out.shape)} dtype={out.dtype} device={out.device} "
                f"contiguous={out.is_contiguous()} for x shape={tuple(x.shape)} "
                f"dtype={x.dtype} device={x.device}"
            )
        # Each thread reads and writes the same linear index, so writing straight
        # back over the input is fine. A *shifted* view of the same storage is
        # not: programs are unordered, so one block's store can land on an element
        # another block has yet to load and the result stops being square(relu(x)).
        # Shape and dtype already match, so both spans have the same length and
        # equal base pointers imply an identical element mapping.
        if out.data_ptr() != x.data_ptr():
            nbytes = x.numel() * x.element_size()
            if x.data_ptr() < out.data_ptr() + nbytes and out.data_ptr() < x.data_ptr() + nbytes:
                raise ValueError(
                    "fused_relu2 out partially overlaps x; pass a separate buffer, or x "
                    "itself -- exact aliasing is supported, a shifted view is not"
                )
    n = x.numel()
    _relu2_kernel[(triton.cdiv(n, _BLOCK),)](x, out, n, BLOCK=_BLOCK, num_warps=_NUM_WARPS)
    return out
