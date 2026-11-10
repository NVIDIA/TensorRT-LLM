# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vision position-embedding bilinear interpolation onto one image's patch grid, in
spatial-merge order, by the stock Qwen3-VL tower's Triton kernel
(`_triton_pos_embed_interpolate` in `_torch/models/modeling_qwen3vl.py`), one call per image.
The NVFP4 default target's entry; the two checkpoints share the bf16 vision tower.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.models.modeling_qwen3vl import _triton_pos_embed_interpolate

NUM_GRID = 48
HIDDEN = 1152
MERGE = 2


def _ckpt(t: int, h: int, w: int) -> dict:
    """The checkpoint's bf16 table at one `(t, h, w)` grid."""
    return dict(
        t=t,
        h=h,
        w=w,
        num_grid_per_side=NUM_GRID,
        m_size=MERGE,
        dtype=torch.bfloat16,
        hidden=HIDDEN,
    )


class TritonPosEmbedInterpolate(OpWrapper):
    """Resample a square row-major `[N*N, hidden]` table onto one `(t, h, w)`
    patch grid; return a fresh `[t*h*w, hidden]` tensor in `dtype`.

    Corner-aligned bilinear (`align_corners=True`): row `i` of `h` samples the
    table at `i * (N - 1) / (h - 1)` (0 when `h == 1`), `lo = floor`,
    `hi = min(lo + 1, N - 1)`; `h` indexes table rows (stride `N`), `w`
    columns. Rows come out in spatial-merge order -- whole `m_size x m_size`
    blocks row-major, row-major inside a block -- which is what the patch
    merger's 4-to-1 reshape consumes. `t` repeats the frame's rows.

    Arithmetic is the HF reference's (`fast_pos_embed_interpolate`): fp32 corner
    weights rounded to `dtype`, multiply-adds in `dtype` (not an fp32 accumulation).
    """

    ARCHS = frozenset({Arch.SM_103})

    # The tower is replicated on every rank, so these are every member's shapes.
    CELLS: tuple[Cell, ...] = (
        Cell(why="one 4x8 image: 32 rows, both sides below the table's 48", spec=_ckpt(1, 4, 8)),
        Cell(why="the 62x90 serving grid: 5580 rows, one image per call", spec=_ckpt(1, 62, 90)),
        Cell(why="a 3-frame 6x10 video: t repeats one frame's 60 rows", spec=_ckpt(3, 6, 10)),
        Cell(
            why="128x256: 32768 rows, the tower's per-call cap and the envelope's largest",
            spec=_ckpt(1, 128, 256),
        ),
    )

    note = """
    Source: the Triton kernel `_bilinear_pos_embed_kernel` behind the stock
    `_triton_pos_embed_interpolate` (not a registered op). Measured on GB300 within 4 / 1
    bf16 ulp (element / relative RMS) of the fp64 reference. `is_valid` refuses what the
    kernel computes silently wrong: a strided table, a `num_grid_per_side` that does not
    match the table, an empty side, `h` or `w` not divisible by `m_size`. Hard limit:
    `t*h*w*hidden` below 2^31 (int32 output offset).
    Not certified: tables with uncorrelated neighbouring rows (can exceed the
    4-ulp band), fp16 or fp32 tables, a `dtype` other than the table's, `m_size`
    other than 2, a hidden size other than 1152, a CPU table, and CUDA-graph
    capture (the vision tower is eager only).
    """

    def __call__(
        self,
        embed_weight: torch.Tensor,
        t: int,
        h: int,
        w: int,
        num_grid_per_side: int,
        m_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return _triton_pos_embed_interpolate(
            embed_weight, t, h, w, num_grid_per_side, m_size, dtype
        )

    def reference(
        self,
        embed_weight: torch.Tensor,
        t: int,
        h: int,
        w: int,
        num_grid_per_side: int,
        m_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """The four-corner weighted sum in fp64 at the exact sample positions, rounded once to
        fp32 (not to `dtype`, so a band in `dtype` ulp measures the op's error alone)."""
        n = num_grid_per_side
        hidden = embed_weight.shape[1]
        device = embed_weight.device
        grid = embed_weight.double().reshape(n, n, hidden)

        def positions(size: int) -> torch.Tensor:
            index = torch.arange(size, dtype=torch.float64, device=device)
            return index * ((n - 1) / (size - 1)) if size > 1 else torch.zeros_like(index)

        h_pos, w_pos = positions(h), positions(w)
        h_lo, w_lo = h_pos.floor().long(), w_pos.floor().long()
        h_hi = (h_lo + 1).clamp(max=n - 1)
        w_hi = (w_lo + 1).clamp(max=n - 1)
        d_h = (h_pos - h_lo)[:, None, None]
        d_w = (w_pos - w_lo)[None, :, None]
        raster = (
            grid[h_lo][:, w_lo] * ((1 - d_h) * (1 - d_w))
            + grid[h_lo][:, w_hi] * ((1 - d_h) * d_w)
            + grid[h_hi][:, w_lo] * (d_h * (1 - d_w))
            + grid[h_hi][:, w_hi] * (d_h * d_w)
        )
        rows = (
            raster.reshape(h // m_size, m_size, w // m_size, m_size, hidden)
            .permute(0, 2, 1, 3, 4)
            .reshape(h * w, hidden)
        )
        return (rows.repeat(t, 1) if t > 1 else rows).float()

    def is_valid(
        self,
        embed_weight: torch.Tensor,
        t: int,
        h: int,
        w: int,
        num_grid_per_side: int,
        m_size: int,
        dtype: torch.dtype,
    ) -> None:
        assert embed_weight.is_contiguous(), (
            "embed_weight must be contiguous; the kernel indexes it as dense rows "
            "and reads a strided table wrongly without raising"
        )
        assert num_grid_per_side * num_grid_per_side == embed_weight.shape[0], (
            f"num_grid_per_side {num_grid_per_side} does not describe a "
            f"{embed_weight.shape[0]}-row table; the kernel resamples the grid it is "
            f"told, in or out of the table"
        )
        assert t > 0 and h > 0 and w > 0, (
            f"empty grid side (t, h, w) = ({t}, {h}, {w}); the kernel returns no rows "
            f"rather than raising"
        )
        assert h % m_size == 0 and w % m_size == 0, (
            f"grid sides h={h}, w={w} are not whole m_size={m_size} merge blocks; "
            f"the kernel's bare assert is all that stops an out-of-grid read"
        )


triton_pos_embed_interpolate = TritonPosEmbedInterpolate()
