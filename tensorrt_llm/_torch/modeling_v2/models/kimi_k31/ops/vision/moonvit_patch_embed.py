# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MoonViT patch embedding: the bias-free patch convolution plus the learnable 2-D position table."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


def _position_rows(
    pos_weight: torch.Tensor, time_weight: torch.Tensor, grid: Tuple[int, int, int]
) -> torch.Tensor:
    """The position rows of one ``(t, h, w)`` grid, frame-major then row-major."""
    t, h, w = grid
    height, width, dim = pos_weight.shape
    if (h, w) == (height, width):
        rows = pos_weight.flatten(end_dim=1)
    else:
        rows = (
            F.interpolate(pos_weight.permute((2, 0, 1)).unsqueeze(0), size=(h, w), mode="bicubic")
            .squeeze(0)
            .permute((1, 2, 0))
            .flatten(end_dim=1)
        )
    if t > 1:
        rows = rows.unsqueeze(0).repeat(t, 1, 1) + time_weight[:t]
    return rows.reshape(-1, dim)


class MoonViTPatchEmbed(OpWrapper):
    """``[N, 3, p, p]`` patches -> ``[N, D]`` rows: each patch through the ``p x p``, stride-``p`` convolution
    ``proj_weight`` ``[D, 3, p, p]`` (no bias), plus its position row.

    A grid ``(t, h, w)`` owns ``t * h * w`` consecutive patches. Its position rows are the ``[H, W, D]`` table
    ``pos_weight`` itself when ``(h, w) == (H, W)`` and the table resampled bicubically to ``h x w`` otherwise, each
    frame ``i`` of a multi-frame grid adding the temporal row ``time_weight[i]`` (``[frames, 1, D]``).
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one image at the table's own 64x64 grid: the table rows as they are",
            spec=dict(grids=((1, 64, 64),)),
        ),
        Cell(
            why="one 32x48 image: the bicubic-resampled table",
            spec=dict(grids=((1, 32, 48),)),
        ),
        Cell(
            why="three images of one call, the smallest 2x2 grid among them",
            spec=dict(grids=((1, 2, 2), (1, 64, 64), (1, 20, 34))),
        ),
        Cell(
            why="a 4-frame grid: the temporal rows on the resampled table",
            spec=dict(grids=((4, 16, 24),)),
        ),
    )
    note = """
    Kimi K3.1's MoonViT input stage, in the tower's dtype (bf16): the patch
    convolution is torch's conv2d on the [N, 3, 14, 14] patches (one output
    pixel each), the resample torch's bicubic interpolate (align_corners
    False) on the table, and the sum one elementwise add. These are the calls
    the engine's MoonViT3d patch embedding makes, in its order, so the rows
    are its bits. No metadata or state; the grids are host values. `pixel_values`
    must hold exactly the patches of `grids`, in grid order.

    The processor sends single-frame image grids; the temporal rows serve only
    multi-frame grids.
    """

    def __call__(
        self,
        pixel_values: torch.Tensor,
        proj_weight: torch.Tensor,
        pos_weight: torch.Tensor,
        time_weight: torch.Tensor,
        grids: Sequence[Tuple[int, int, int]],
    ) -> torch.Tensor:
        x = F.conv2d(pixel_values, proj_weight, None, tuple(proj_weight.shape[-2:]))
        x = x.view(x.size(0), -1)
        return x + torch.cat([_position_rows(pos_weight, time_weight, grid) for grid in grids])

    def reference(
        self,
        pixel_values: torch.Tensor,
        proj_weight: torch.Tensor,
        pos_weight: torch.Tensor,
        time_weight: torch.Tensor,
        grids: Sequence[Tuple[int, int, int]],
    ) -> torch.Tensor:
        """fp32: each patch dotted with every filter, plus the fp32-resampled table, rounded once."""
        patches = pixel_values.float().flatten(1)
        x = patches @ proj_weight.float().flatten(1).t()
        pos = torch.cat(
            [_position_rows(pos_weight.float(), time_weight.float(), grid) for grid in grids]
        )
        return (x + pos).to(pixel_values.dtype)

    def is_valid(
        self,
        pixel_values: torch.Tensor,
        proj_weight: torch.Tensor,
        pos_weight: torch.Tensor,
        time_weight: torch.Tensor,
        grids: Sequence[Tuple[int, int, int]],
    ) -> None:
        if tuple(pixel_values.shape[1:]) != tuple(proj_weight.shape[1:]):
            raise ValueError("each patch must be one convolution window (one output pixel)")
        if any(t > time_weight.shape[0] for t, _, _ in grids):
            raise ValueError("a grid has more frames than the temporal table")


moonvit_patch_embed = MoonViTPatchEmbed()
