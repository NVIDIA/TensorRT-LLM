# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RMSNorm of MoonViT's blocks and projector: torch's ``rms_norm``."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class MoonViTRmsNorm(OpWrapper):
    """``x / sqrt(mean(x^2) + eps) * weight`` over the last dimension, in ``x.dtype``.

    ``eps=None`` is torch's default: ``torch.finfo(x.dtype).eps``.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="a block norm: 1024 wide, default eps, one 64x64 image of rows",
            spec=dict(rows=4096, width=1024, eps=None),
        ),
        Cell(why="a block norm on the smallest 2x2 image", spec=dict(rows=4, width=1024, eps=None)),
        Cell(
            why="the projector's post-norm: 7168 wide, eps 1e-5, one 64x64 image of merged rows",
            spec=dict(rows=1024, width=7168, eps=1e-5),
        ),
    )
    note = """
    torch's `F.rms_norm`, the call `torch.nn.RMSNorm` makes, so the bits are
    those of the engine's MoonViT (its block norms, its final norm and its
    projector's post-norm are `torch.nn.RMSNorm`). The block norms use the
    default eps, which depends on the input dtype (bf16: 2^-7); pass it as
    None to keep that. No metadata or state.
    """

    def __call__(
        self, x: torch.Tensor, weight: torch.Tensor, eps: Optional[float] = None
    ) -> torch.Tensor:
        return F.rms_norm(x, tuple(weight.shape), weight, eps)

    def reference(
        self, x: torch.Tensor, weight: torch.Tensor, eps: Optional[float] = None
    ) -> torch.Tensor:
        """fp32, rounded once."""
        eps = torch.finfo(x.dtype).eps if eps is None else eps
        x32 = x.float()
        out = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
        return out.to(x.dtype)


moonvit_rms_norm = MoonViTRmsNorm()
