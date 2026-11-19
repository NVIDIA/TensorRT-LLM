# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MoonViT's 2-D rotary embedding: q and k of every patch rotated by the complex angles of its (row, column)."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Kimi K3.1's MoonViT attention: 12 heads of 128, a 512 x 512 angle table, theta 1e4.
HEADS, HEAD_DIM, TABLE_SIDE, THETA = 12, 128, 512, 10000.0


def moonvit_rope_table(
    head_dim: int, max_height: int, max_width: int, theta: float, device: torch.device
) -> torch.Tensor:
    """The ``[max_height, max_width, head_dim / 2]`` complex64 angle table: per position, the column angles and the
    row angles of each frequency, interleaved (column first)."""
    flat_pos = torch.arange(max_height * max_width, device=device).float()
    x_pos = flat_pos % max_width
    y_pos = flat_pos // max_width
    dim_range = torch.arange(0, head_dim, 4, device=device)[: head_dim // 4].float()
    freqs = 1.0 / (theta ** (dim_range / head_dim))
    x_cis = torch.polar(torch.ones_like(torch.outer(x_pos, freqs)), torch.outer(x_pos, freqs))
    y_cis = torch.polar(torch.ones_like(torch.outer(y_pos, freqs)), torch.outer(y_pos, freqs))
    freqs_cis = torch.cat([x_cis.unsqueeze(-1), y_cis.unsqueeze(-1)], dim=-1)
    return freqs_cis.reshape(max_height, max_width, -1)


def moonvit_rope_angles(table: torch.Tensor, grids: Sequence[Tuple[int, int, int]]) -> torch.Tensor:
    """The ``[sum t*h*w, head_dim / 2]`` angles of the patches of ``grids``: each frame's ``h x w`` corner of
    ``table``, row-major."""
    width = table.shape[-1]
    return torch.cat([table[:h, :w].reshape(-1, width).repeat(t, 1) for t, h, w in grids], dim=0)


class MoonViTRope2d(OpWrapper):
    """Rotate ``q`` and ``k`` (``[N, heads, head_dim]``) by ``freqs_cis`` (``[N, head_dim / 2]`` complex64): each
    consecutive pair ``(x[2i], x[2i+1])`` of a head, as a complex number, times ``freqs_cis[n, i]``.

    Fresh ``(q_out, k_out)`` in the inputs' shapes and dtypes; the product is taken in fp32 and rounded once.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one 64x64 image, q and k as head views of one fused qkv projection",
            spec=dict(grids=((1, 64, 64),), heads=HEADS, head_dim=HEAD_DIM),
        ),
        Cell(
            why="three images of one call, the smallest 2x2 grid among them",
            spec=dict(grids=((1, 2, 2), (1, 32, 48), (1, 20, 34)), heads=HEADS, head_dim=HEAD_DIM),
        ),
        Cell(
            why="the table's far corner: a 512x512 grid",
            spec=dict(grids=((1, 512, 512),), heads=HEADS, head_dim=HEAD_DIM),
        ),
    )
    note = """
    The engine's MoonViT rotary embedding, call for call: q and k are made
    fp32 (`.float()`, so strided head views of the qkv projection are read as
    they are), viewed as complex pairs, multiplied by the complex64 angles and
    cast back with `type_as`. `moonvit_rope_table` builds the angle table the
    way the engine's MoonViT does (on the device, once per tower) and
    `moonvit_rope_angles` gathers a call's rows from it. No metadata; eager
    torch elementwise kernels, never captured.
    """

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        freqs_cis = freqs_cis.unsqueeze(-2)
        q_pairs = torch.view_as_complex(q.float().view(*q.shape[:-1], -1, 2))
        k_pairs = torch.view_as_complex(k.float().view(*k.shape[:-1], -1, 2))
        q_out = torch.view_as_real(q_pairs * freqs_cis).flatten(-2)
        k_out = torch.view_as_real(k_pairs * freqs_cis).flatten(-2)
        return q_out.type_as(q), k_out.type_as(k)

    def reference(
        self, q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """fp64 real arithmetic: the 2x2 rotation of each pair by the angle's cos and sin, rounded once."""
        cos = freqs_cis.real.double().unsqueeze(-2)
        sin = freqs_cis.imag.double().unsqueeze(-2)

        def rotate(x: torch.Tensor) -> torch.Tensor:
            x64 = x.double()
            even, odd = x64[..., 0::2], x64[..., 1::2]
            out = torch.empty_like(x64)
            out[..., 0::2] = even * cos - odd * sin
            out[..., 1::2] = even * sin + odd * cos
            return out.to(x.dtype)

        return rotate(q), rotate(k)

    def is_valid(self, q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor) -> None:
        if q.shape[0] != freqs_cis.shape[0] or k.shape[0] != freqs_cis.shape[0]:
            raise ValueError("freqs_cis must hold one row per token of q and k")
        if q.ndim != 3 or k.ndim != 3:
            raise ValueError("q and k must be [N, heads, head_dim]")


moonvit_rope_2d = MoonViTRope2d()
