# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP32 copy of the Kimi K3 router input."""

import os

import torch
import triton
import triton.language as tl

# Rows from which the vectorized kernel replaces the elementwise cast; below, both take a few
# microseconds and the elementwise cast saves the Triton launch.
_MIN_ROWS = 512
_BLOCK = 8192
_ENABLED = os.environ.get("TRTLLM_K3_ROUTER_INPUT_CAST", "1") == "1"


@triton.jit
def _bf16_to_fp32_kernel(x, y, size, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < size
    tl.store(y + offsets, tl.load(x + offsets, mask=mask).to(tl.float32), mask=mask)


def router_input_fp32(x: torch.Tensor) -> torch.Tensor:
    """Return ``x.to(torch.float32)``.

    Every BF16 value is an FP32 value, so the result is the same bit for bit on either path. A
    contiguous 2-D BF16 CUDA tensor with at least ``_MIN_ROWS`` rows is converted by a vectorized
    kernel (about 6.6 TB/s on GB300, against 2.4 TB/s for the elementwise cast);
    ``TRTLLM_K3_ROUTER_INPUT_CAST=0`` keeps the elementwise cast.
    """
    if not (
        _ENABLED
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.dim() == 2
        and x.shape[0] >= _MIN_ROWS
        and x.is_contiguous()
        and not torch.compiler.is_compiling()
        and not (torch.is_grad_enabled() and x.requires_grad)
    ):
        return x.to(torch.float32)
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    size = x.numel()
    _bf16_to_fp32_kernel[(triton.cdiv(size, _BLOCK),)](x, out, size, BLOCK=_BLOCK, num_warps=8)
    return out
