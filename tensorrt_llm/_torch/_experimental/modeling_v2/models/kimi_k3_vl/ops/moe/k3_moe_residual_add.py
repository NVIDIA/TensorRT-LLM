# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MoE block's residual update with both BF16 rounding boundaries kept, in one launch."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_moe import kimi_k3_moe_residual as _residual


class K3MoeResidualAdd(OpWrapper):
    """``residual + bf16(routed + shared)`` as PyTorch's two BF16 adds compute it; new tensor."""

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    #: Decode rows of every deployment that fuses the residual (helix8, helix16, dep16), up to the
    #: kernel's 96-row bound, and one row above it (the PyTorch adds).
    CELLS: tuple[Cell, ...] = (
        Cell(why="one row", spec={"m": 1}),
        Cell(why="eight requests of 8 rows", spec={"m": 64}),
        Cell(why="the kernel's 96-row bound", spec={"m": 96}),
        Cell(why="above the bound: the PyTorch path", spec={"m": 128}),
    )

    note = """
    One Triton kernel for contiguous BF16 ``[M, 7168]`` operands of ``1 <= M <= 96`` rows on one
    CUDA device (no autograd); every other input takes ``residual + (routed + shared)`` in PyTorch,
    with the same result. The output is a new tensor shaped like ``routed``.
    """

    def __call__(
        self, routed: torch.Tensor, shared: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        return _residual._fused_moe_residual_add(routed, shared, residual)

    def reference(
        self, routed: torch.Tensor, shared: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        """FP32 adds rounded to BF16 after each, as the two PyTorch BF16 adds."""
        moe = (routed.float() + shared.float()).to(torch.bfloat16)
        return (residual.float() + moe.float()).to(torch.bfloat16)


k3_moe_residual_add = K3MoeResidualAdd()
