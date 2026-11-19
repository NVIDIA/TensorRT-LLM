# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fused SiTU-gated multiply (Triton) and its launcher.

``situ_and_mul`` launches the kernel on a packed ``[num_tokens, 2 * d]`` ``[gate | up]`` input and returns a new
``[num_tokens, d]`` tensor; the family op ``ops/activation/situ_and_mul`` calls it directly.
"""

from __future__ import annotations

from typing import Mapping

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._utils import get_sm_version


@triton.jit
def situ_and_mul_kernel(
    o_ptr,
    o_stride,
    x_ptr,
    x_stride,
    d,
    beta,
    linear_beta,
    BLOCK_SIZE: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
) -> None:
    """Fused :class:`SituAndMul` on a packed ``[gate | up]`` row layout."""
    i = tl.program_id(axis=0).to(tl.int64)
    j = tl.program_id(axis=1)

    o_row_ptr = o_ptr + o_stride * i
    x_row_ptr = x_ptr + x_stride * i

    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_wait()

    offsets = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < d

    gate = tl.load(x_row_ptr + offsets, mask=mask).to(tl.float32)
    up = tl.load(x_row_ptr + offsets + d, mask=mask).to(tl.float32)

    situ_a = beta * tldevice.tanh(gate / beta) * tl.sigmoid(gate)
    up = linear_beta * tldevice.tanh(up / linear_beta)
    result = situ_a * up

    tl.store(o_row_ptr + offsets, result, mask=mask)

    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def situ_and_mul(x: torch.Tensor, beta: float, linear_beta: float) -> torch.Tensor:
    """Run fused SiTU on a packed ``[num_tokens, 2 * d]`` input."""
    b, n = x.shape

    assert n % 2 == 0
    if x.stride(1) != 1:
        raise ValueError("situ_and_mul requires a contiguous last dimension")
    d = n // 2
    output = torch.empty((b, d), dtype=x.dtype, device=x.device)

    def grid(meta: Mapping[str, int]) -> tuple[int, int]:
        return (b, triton.cdiv(d, meta["BLOCK_SIZE"]))

    launch_with_pdl = get_env_enable_pdl() and get_sm_version() >= 90
    situ_and_mul_kernel[grid](
        o_ptr=output,
        o_stride=output.stride(0),
        x_ptr=x,
        x_stride=x.stride(0),
        d=d,
        beta=float(beta),
        linear_beta=float(linear_beta),
        BLOCK_SIZE=1024,
        LAUNCH_WITH_PDL=launch_with_pdl,
        launch_pdl=launch_with_pdl,
    )
    return output
