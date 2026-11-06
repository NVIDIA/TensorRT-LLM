# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RMS normalization over the last dim in the source model's fp32 op order, rounded once."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: The Beam decoder block's input norm: one 5120-wide row per token, bf16, eps 1e-6.
_BLOCK_INPUT = dict(width=5120, dtype=torch.bfloat16, eps=1e-6)


class BeamRmsnormFp32(OpWrapper):
    """`x / sqrt(mean(x^2, -1) + eps) * weight`, one eager torch op per source step.

    `x` is `[T, D]` bf16 and `weight` `[D]` bf16. The steps, as the source model computes
    them::

        x32    = x.float()                                   [T, D] fp32
        rms    = x32.pow(2).mean(-1, keepdim=True)           [T, 1] fp32
                    .add(eps).rsqrt()
        normed = x32 * rms                                   [T, D] fp32
        out    = (normed * weight).to(x.dtype)               [T, D] bf16, weight promoted

    Returns `out`, a new tensor; `x` is not mutated.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="block input norm, 1 token", spec=dict(_BLOCK_INPUT, tokens=1)),
        Cell(why="block input norm, 17 tokens", spec=dict(_BLOCK_INPUT, tokens=17)),
        Cell(
            why="block input norm, 129 tokens: the longest context",
            spec=dict(_BLOCK_INPUT, tokens=129),
        ),
    )

    note = """
    Not the catalog `flashinfer_rmsnorm` entry. That kernel also reduces in fp32 and rounds
    once, but it sums the squares in its own order and computes its own reciprocal square
    root. On identical bf16 inputs its output differs from the source model's by one bf16
    unit wherever the fp32 product lands near a rounding boundary. The difference is a few
    elements per 129 tokens of real hidden states. Every projection that reads the normalized
    row then carries it.

    This entry issues the source's torch ops in the source's order. Its result equals the
    source's when both run the same torch reduction kernel for the same row width. The
    entry does not guard against a different torch build that changes that kernel's
    summation order.
    """

    def __call__(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        x32 = x.float()
        rms = x32.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
        normed = x32 * rms
        return (normed * weight).to(x.dtype)

    def reference(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        """Statistics and products in fp64, rounded once to `x.dtype`."""
        x64 = x.double()
        rms = torch.rsqrt(x64.pow(2).mean(dim=-1, keepdim=True) + eps)
        return (x64 * rms * weight.double()).to(x.dtype)

    def is_valid(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
        # A weight of another dtype or width broadcasts or promotes without raising, and an
        # fp32 input removes the one rounding the projections' input carries.
        assert x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16, (
            f"x and weight must be torch.bfloat16, got {x.dtype} and {weight.dtype}"
        )
        assert weight.dim() == 1 and weight.shape[0] == x.shape[-1], (
            f"weight must be [{x.shape[-1]}], got {tuple(weight.shape)}"
        )


beam_rmsnorm_fp32 = BeamRmsnormFp32()
