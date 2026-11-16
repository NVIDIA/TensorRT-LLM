# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Depth-scaled residual add, in the activation dtype."""

from __future__ import annotations

import math

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: A block's two adds as (branch, residual) scales: (1, sqrt(n - 1)) / sqrt(n) for the n-th
#: add of the residual stream. Block i's attention add is n = 2(i + 1) and its feed-forward
#: add the next one: n = 2 and 3 for block 0, n = 8 and 9 for block 3.
_ATTENTION_ADD = (1.0 / math.sqrt(2.0), math.sqrt(1.0) / math.sqrt(2.0))
_FEED_FORWARD_ADD = (1.0 / math.sqrt(3.0), math.sqrt(2.0) / math.sqrt(3.0))
_BLOCK3_ATTENTION_ADD = (1.0 / math.sqrt(8.0), math.sqrt(7.0) / math.sqrt(8.0))
_BLOCK3_FEED_FORWARD_ADD = (1.0 / math.sqrt(9.0), math.sqrt(8.0) / math.sqrt(9.0))


def depth_scales(layer_idx: int, is_attention: bool) -> tuple[float, float]:
    """(branch, residual) scales of a block's attention or feed-forward residual add.

    The n-th add of the residual stream scales the branch by 1 / sqrt(n) and the stream
    by sqrt(n - 1) / sqrt(n); block i's attention add is the 2(i + 1)-th, its
    feed-forward add the next one.
    """
    num_adds = 2 * (layer_idx + 1) + (0 if is_attention else 1)
    denom = math.sqrt(num_adds)
    return 1.0 / denom, math.sqrt(num_adds - 1) / denom


_HIDDEN = dict(hidden=5120, dtype=torch.bfloat16)


class BeamDepthResidual(OpWrapper):
    """`branch * branch_scale + residual * residual_scale`, a new tensor.

    Three activation-dtype operations -- two scalings and the add -- each rounded to
    that dtype, as the source model computes them. The scales are Python floats and
    reach the kernel as fp32.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="block 0 attention add, 17 tokens",
            spec=dict(_HIDDEN, tokens=17, scales=_ATTENTION_ADD),
        ),
        Cell(
            why="block 0 feed-forward add, 1 token",
            spec=dict(_HIDDEN, tokens=1, scales=_FEED_FORWARD_ADD),
        ),
        Cell(
            why="block 0 feed-forward add, 129 tokens: the longest context",
            spec=dict(_HIDDEN, tokens=129, scales=_FEED_FORWARD_ADD),
        ),
        Cell(
            why="block 3 attention add, 17 tokens",
            spec=dict(_HIDDEN, tokens=17, scales=_BLOCK3_ATTENTION_ADD),
        ),
        Cell(
            why="block 3 feed-forward add, 17 tokens",
            spec=dict(_HIDDEN, tokens=17, scales=_BLOCK3_FEED_FORWARD_ADD),
        ),
    )

    note = """
    Not fused with the norm before it: the source rounds the norm output, then each
    scaled term, then the sum. A fused fp32 add-and-scale rounds once and is a
    different result.
    """

    def __call__(
        self,
        branch: torch.Tensor,
        residual: torch.Tensor,
        branch_scale: float,
        residual_scale: float,
    ) -> torch.Tensor:
        return branch * branch_scale + residual * residual_scale

    def reference(
        self,
        branch: torch.Tensor,
        residual: torch.Tensor,
        branch_scale: float,
        residual_scale: float,
    ) -> torch.Tensor:
        """Each product and the sum in fp32, rounded to the activation dtype in turn."""
        dtype = branch.dtype
        scaled_branch = (branch.float() * branch_scale).to(dtype)
        scaled_residual = (residual.float() * residual_scale).to(dtype)
        return (scaled_branch.float() + scaled_residual.float()).to(dtype)

    def is_valid(
        self,
        branch: torch.Tensor,
        residual: torch.Tensor,
        branch_scale: float,
        residual_scale: float,
    ) -> None:
        # Broadcasting accepts mismatched shapes, mixed dtypes promote, and a tensor
        # scale in the activation dtype rounds the coefficient itself. None raise.
        assert branch.shape == residual.shape, (
            f"branch {tuple(branch.shape)} and residual {tuple(residual.shape)} must match"
        )
        assert branch.dtype == residual.dtype, "branch and residual must share one dtype"
        assert isinstance(branch_scale, float) and isinstance(residual_scale, float), (
            "scales must be Python floats"
        )


beam_depth_residual = BeamDepthResidual()
