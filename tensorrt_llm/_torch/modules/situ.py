# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SiTU gated activation and its fused Triton implementation."""

from __future__ import annotations

from typing import Mapping, Optional

import torch
import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]
import triton.language.extra.libdevice as tldevice  # type: ignore[import]
from torch import nn


class SituAndMul(nn.Module):
    """SiTU activation with gate/up multiplicative gating.

    Runs the math in fp32 for numerical stability, then casts back to the
    input dtype. CUDA inputs optionally use the fused Triton implementation;
    CPU and meta inputs keep the eager reference path.
    """

    def __init__(
        self,
        *,
        beta: float = 1.0,
        linear_beta: Optional[float] = None,
        use_fused_activation: bool = False,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta
        self.use_fused_activation = use_fused_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fused_activation and x.is_cuda:
            return torch.ops.trtllm.situ_and_mul(
                x.reshape(-1, x.shape[-1]), self.beta, self.linear_beta
            ).reshape(*x.shape[:-1], x.shape[-1] // 2)

        d = x.shape[-1] // 2
        gate = x[..., :d].to(torch.float32)
        up = x[..., d:].to(torch.float32)
        situ_a = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return (situ_a * up).to(x.dtype)


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
    HAS_LINEAR_BETA: tl.constexpr,
) -> None:
    """Fused :class:`SituAndMul` on a packed ``[gate | up]`` row layout."""
    i = tl.program_id(axis=0).to(tl.int64)
    j = tl.program_id(axis=1)

    o_row_ptr = o_ptr + o_stride * i
    x_row_ptr = x_ptr + x_stride * i

    offsets = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < d

    gate = tl.load(x_row_ptr + offsets, mask=mask).to(tl.float32)
    up = tl.load(x_row_ptr + offsets + d, mask=mask).to(tl.float32)

    situ_a = beta * tldevice.tanh(gate / beta) * tl.sigmoid(gate)
    if HAS_LINEAR_BETA:
        up = linear_beta * tldevice.tanh(up / linear_beta)
    result = situ_a * up

    tl.store(o_row_ptr + offsets, result, mask=mask)


@torch.library.custom_op("trtllm::situ_and_mul", mutates_args=())
def situ_and_mul(x: torch.Tensor, beta: float, linear_beta: Optional[float] = None) -> torch.Tensor:
    """Run fused SiTU on a packed ``[num_tokens, 2 * d]`` input."""
    b, n = x.shape

    assert n % 2 == 0
    if x.stride(1) != 1:
        raise ValueError("situ_and_mul requires a contiguous last dimension")
    d = n // 2
    output = torch.empty((b, d), dtype=x.dtype, device=x.device)

    def grid(meta: Mapping[str, int]) -> tuple[int, int]:
        return (b, triton.cdiv(d, meta["BLOCK_SIZE"]))

    situ_and_mul_kernel[grid](
        o_ptr=output,
        o_stride=output.stride(0),
        x_ptr=x,
        x_stride=x.stride(0),
        d=d,
        beta=float(beta),
        linear_beta=float(linear_beta) if linear_beta is not None else 1.0,
        BLOCK_SIZE=1024,
        HAS_LINEAR_BETA=linear_beta is not None,
    )
    return output


@situ_and_mul.register_fake
def _(x: torch.Tensor, beta: float, linear_beta: Optional[float] = None) -> torch.Tensor:
    b, n = x.shape
    assert n % 2 == 0
    return x.new_empty((b, n // 2))
