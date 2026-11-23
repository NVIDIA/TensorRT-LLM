# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vision-tower GELU, the exact erf form (`vision_config.hidden_act == "gelu"`), not the tanh approximation."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: The vision tower's `intermediate_size`: the block MLP's fc1 output width.
MLP_WIDTH = 4304

#: The merger's `hidden_size * spatial_merge_size**2`: its 2x2 conv output width.
MERGER_WIDTH = 4608

_SQRT_2 = math.sqrt(2.0)


class Gelu(OpWrapper):
    """`0.5 * x * (1 + erf(x / sqrt(2)))` elementwise, a new tensor out in `x`'s dtype."""

    ARCHS = frozenset({Arch.SM_103})

    # spec `dist`: "uniform" over [lo, hi] (default), or "tower"/"merger" -- the block MLP's fc1 output and the
    # merger's proj.0 output, built through synthetic layers at the checkpoint's weight statistics.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="block MLP at 512x512: 1024 patch rows x 4304, the tower's own activand",
            spec=dict(shape=(1024, MLP_WIDTH), dtype=torch.bfloat16, dist="tower"),
        ),
        Cell(
            why="block MLP at 1024x1024: 4096 patch rows x 4304 bf16",
            spec=dict(shape=(4096, MLP_WIDTH), dtype=torch.bfloat16, lo=-16.0, hi=16.0),
        ),
        Cell(
            why="block MLP envelope: 16384 patch rows (max_vit_seq_len) x 4304, the tower's own activand",
            spec=dict(shape=(16384, MLP_WIDTH), dtype=torch.bfloat16, dist="tower"),
        ),
        Cell(
            why="merger at 512x512: 256 merged blocks x 4608 bf16",
            spec=dict(shape=(256, MERGER_WIDTH), dtype=torch.bfloat16, lo=-16.0, hi=16.0),
        ),
        Cell(
            why="merger at 1024x1024: 1024 merged blocks x 4608, built the way the merger builds it",
            spec=dict(shape=(1024, MERGER_WIDTH), dtype=torch.bfloat16, dist="merger"),
        ),
        Cell(
            why="merger envelope: 4096 merged blocks from 16384 patches x 4608 bf16",
            spec=dict(shape=(4096, MERGER_WIDTH), dtype=torch.bfloat16, lo=-16.0, hi=16.0),
        ),
    )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)

    def reference(self, x: torch.Tensor) -> torch.Tensor:
        """The closed form in fp64, rounded once to `x.dtype`."""
        x64 = x.double()
        return (0.5 * x64 * (1.0 + torch.erf(x64 / _SQRT_2))).to(x.dtype)


gelu = Gelu()
