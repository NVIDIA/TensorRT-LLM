# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GELU of MoonViT: the tanh form in the block MLPs, the exact (erf) form in the projector."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class MoonViTGelu(OpWrapper):
    """``x * Phi(x)`` elementwise, in ``x.dtype``: ``approximate="tanh"`` takes the tanh form of ``Phi``,
    ``"none"`` the exact erf form."""

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="a block MLP's 4096-wide hidden rows, tanh form",
            spec=dict(rows=4096, width=4096, approximate="tanh"),
        ),
        Cell(
            why="the projector's 4096-wide merged rows, exact form",
            spec=dict(rows=1024, width=4096, approximate="none"),
        ),
        Cell(
            why="the smallest 2x2 image: 4 block rows, tanh form",
            spec=dict(rows=4, width=4096, approximate="tanh"),
        ),
    )
    note = """
    torch's `F.gelu`: the tanh form is the engine MoonViT MLP's activation and
    the exact form its projector's `torch.nn.GELU()`, so the bits are theirs.
    No metadata or state.
    """

    def __call__(self, x: torch.Tensor, approximate: str) -> torch.Tensor:
        return F.gelu(x, approximate=approximate)

    def reference(self, x: torch.Tensor, approximate: str) -> torch.Tensor:
        """fp64 closed forms, rounded once."""
        x64 = x.double()
        if approximate == "tanh":
            inner = math.sqrt(2.0 / math.pi) * (x64 + 0.044715 * x64.pow(3))
            out = 0.5 * x64 * (1.0 + torch.tanh(inner))
        else:
            out = 0.5 * x64 * (1.0 + torch.erf(x64 / math.sqrt(2.0)))
        return out.to(x.dtype)

    def is_valid(self, x: torch.Tensor, approximate: str) -> None:
        if approximate not in ("tanh", "none"):
            raise ValueError(f"approximate must be 'tanh' or 'none', got {approximate!r}")


moonvit_gelu = MoonViTGelu()
