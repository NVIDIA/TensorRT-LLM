# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 MoE residual update on SM107: ``residual + bf16(routed + shared)`` in one kernel."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.moe.rubin_moe_residual import _fused_moe_residual_add as _launch

HIDDEN = 7168


class RubinMoeResidualAdd(OpWrapper):
    """``residual + (routed + shared)`` for BF16 ``[rows, 7168]`` operands, new BF16 output.

    Two BF16 roundings, as the unfused module path has: the MoE output ``routed + shared`` is
    rounded to BF16, then added to the residual and rounded again. One Triton launch up to 96
    rows of contiguous CUDA tensors; with ``any_rows`` (a context step) one launch at every row
    count above 0; otherwise the same two additions in torch.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = tuple(
        Cell(why=why, spec=dict(rows=rows))
        for rows, why in (
            (1, "one row"),
            (8, "TEP8 generation b1 k7: 8 rows"),
            (64, "helix8 generation b8 k7: 64 rows"),
            (96, "DEP16 generation b12 k7 and helix8 b16 k5: 96 rows, the fused maximum"),
            (97, "one row past the fused kernel: the torch path"),
        )
    ) + tuple(
        Cell(why=why, spec=dict(rows=rows, any_rows=True))
        for rows, why in (
            (16384, "any_rows 16384: a DEP16 context step's 16k chunk"),
            (16383, "any_rows 16383: a partial last block"),
            (4095, "any_rows 4095: a short context step"),
            (97, "any_rows 97: one row past the generation kernel"),
            (1, "any_rows 1: one row"),
            (0, "any_rows 0: a context rank without rows, the torch path"),
        )
    )

    note = """
    The result is bitwise the unfused ``residual + (routed + shared)`` on every path: each output
    element is a function of the three input elements at its index, with the same two BF16
    roundings. The kernel keeps no state and allocates its output per call; it captures in a CUDA
    graph. ``any_rows`` changes which launch serves the rows, never the arithmetic. NaN inputs
    give NaN outputs whose payload may differ from torch's; the model's inputs are never NaN.
    """

    def __call__(
        self,
        routed: torch.Tensor,
        shared: torch.Tensor,
        residual: torch.Tensor,
        *,
        any_rows: bool = False,
    ) -> torch.Tensor:
        return _launch(routed, shared, residual, any_rows=any_rows)

    def reference(
        self,
        routed: torch.Tensor,
        shared: torch.Tensor,
        residual: torch.Tensor,
        *,
        any_rows: bool = False,
    ) -> torch.Tensor:
        """FP32 sums with the op's two BF16 roundings (the first one is the op's definition, not
        an approximation); ``any_rows`` picks a launch, not a result."""
        moe = (routed.float() + shared.float()).to(torch.bfloat16)
        return (residual.float() + moe.float()).to(torch.bfloat16)

    def is_valid(
        self,
        routed: torch.Tensor,
        shared: torch.Tensor,
        residual: torch.Tensor,
        *,
        any_rows: bool = False,
    ) -> None:
        """The torch path broadcasts: operands of different shapes are added without an error."""
        assert routed.shape == shared.shape == residual.shape, (
            f"routed {tuple(routed.shape)}, shared {tuple(shared.shape)} and residual "
            f"{tuple(residual.shape)} must have one shape"
        )
        assert routed.dtype == shared.dtype == residual.dtype == torch.bfloat16, (
            "operands must be BF16"
        )


rubin_moe_residual_add = RubinMoeResidualAdd()
