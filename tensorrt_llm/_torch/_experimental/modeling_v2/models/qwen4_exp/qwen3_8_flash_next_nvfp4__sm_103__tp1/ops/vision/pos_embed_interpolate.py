# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vision position-embedding bilinear interpolation onto each image's patch
grid, emitted in spatial-merge order.

Tier-3 composed: index and corner-weight arithmetic on `[h]`, `[w]` and
`[h, w]` tensors, then one `torch.nn.functional.embedding_bag` for the whole
batch. No registered op interpolates anything, and `modules/` has no vision
position-embedding callable. The exact-math sibling,
`_triton_pos_embed_interpolate` in `models/modeling_qwen3vl.py`, is a private
symbol of another model's modeling file; the other in-tree resamplers are
bicubic and/or `align_corners=False`, a different function.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: This checkpoint: num_position_embeddings 2304 (a 48 x 48 grid), hidden 1152,
#: spatial_merge_size 2.
NUM_GRID = 48
HIDDEN = 1152
MERGE = 2


def _merge_order(x: torch.Tensor, height: int, width: int, merge: int) -> torch.Tensor:
    """`[h, w, ...]` raster -> `[h*w, ...]` in (h_block, w_block, row, col) order."""
    tail = x.shape[2:]
    return (
        x.reshape(height // merge, merge, width // merge, merge, *tail)
        .permute(0, 2, 1, 3, *range(4, 4 + len(tail)))
        .reshape(height * width, *tail)
    )


class PosEmbedInterpolate(OpWrapper):
    """Resample a square row-major `[N*N, hidden]` table onto each `(t, h, w)`
    patch grid; return `[sum(t*h*w), hidden]` in the table's dtype.

    Corner-aligned bilinear: `pos = linspace(0, N - 1, n)` per axis, `lo =
    floor(pos)`, `hi = min(lo + 1, N - 1)`; `h` indexes table rows (stride `N`),
    `w` columns. Rows come out in spatial-merge order -- whole `merge x merge`
    blocks row-major, row-major inside a block -- which is what the patch
    merger's 4-to-1 reshape consumes; raster order is silently wrong. `t`
    repeats the frame's rows (no temporal axis); images concatenate in order.
    The four-term sum accumulates in fp32 and rounds once.

    Fusion boundary: the position embeddings alone. Adding them to the patch
    embeddings, the 2-D rope and the attention segmentation are the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: grid (tuple of (t, h, w)), merge, dtype, rows (table rows), hidden.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="a 4-image batch incl. the 62x90 serving grid",
            spec=dict(
                grid=((1, 4, 8), (2, 6, 4), (1, 48, 48), (1, 62, 90)),
                merge=MERGE,
                dtype=torch.bfloat16,
                rows=NUM_GRID**2,
                hidden=HIDDEN,
            ),
        ),
        Cell(
            why="video: 3 frames repeat one 6x10 frame",
            spec=dict(
                grid=((3, 6, 10),),
                merge=MERGE,
                dtype=torch.bfloat16,
                rows=NUM_GRID**2,
                hidden=HIDDEN,
            ),
        ),
        # The largest grid one call carries (32768 patches).
        Cell(
            why="checkpoint table bf16, grid [[1, 128, 256]]",
            spec=dict(
                grid=((1, 128, 256),),
                merge=MERGE,
                dtype=torch.bfloat16,
                rows=NUM_GRID**2,
                hidden=HIDDEN,
            ),
        ),
    )

    note = """
    Metadata consumed: none. `grid_thw` is host ints (a device tensor works but
    `int()` syncs per element). The table is not mutated; the result is fresh.

    Loud failures, left to torch: an `h` or `w` not divisible by
    `spatial_merge_size` (or a merge larger than the grid) raises "shape ... is
    invalid for input of size ..."; a non-2-D table raises "weight has to be a
    2D Tensor". An empty grid side returns `[0, hidden]` rather than raising.

    At `h = w = N` the samples are exactly `arange(N)` and the result is
    bitwise the table in merge order. The bf16 result is within one bf16
    half-ulp of the fp64 value (0.0147 at 62x90); a bf16 multiply-add chain
    (a literal HF transcription) lands up to one bf16 ulp (0.031) from this.

    CUDA graphs: capturable, but the grid is a host argument, so it is frozen
    at capture. Cost is ~225 us of launch-bound index construction per image,
    flat in token count; the gather is the only part that scales.

    Certified at the checkpoint's bf16 table on a 4-image batch (4x8, a 2x6x4
    video, the 48x48 exact fit and the 62x90 serving grid), a 3-frame 6x10
    video and the 128x256 grid one call carries at most. Not certified: a CPU
    table (it runs and returns on CPU); fp16 and fp32 tables; a
    `spatial_merge_size` other than the checkpoint's 2.
    """

    def __call__(
        self,
        pos_embed_weight: torch.Tensor,
        grid_thw: Sequence[Sequence[int]],
        spatial_merge_size: int,
    ) -> torch.Tensor:
        num_grid_per_side = math.isqrt(pos_embed_weight.shape[0])
        device = pos_embed_weight.device
        merge = spatial_merge_size
        idx_parts = []
        weight_parts = []

        for frames, height, width in grid_thw:
            frames, height, width = int(frames), int(height), int(width)

            # Corner-aligned; non-negative positions, so the int64 cast floors.
            h_pos = torch.linspace(
                0, num_grid_per_side - 1, height, device=device, dtype=torch.float32
            )
            w_pos = torch.linspace(
                0, num_grid_per_side - 1, width, device=device, dtype=torch.float32
            )
            h_lo = h_pos.to(torch.int64)
            w_lo = w_pos.to(torch.int64)
            h_hi = (h_lo + 1).clamp(max=num_grid_per_side - 1)
            w_hi = (w_lo + 1).clamp(max=num_grid_per_side - 1)
            dh = (h_pos - h_lo).unsqueeze(1)
            dw = (w_pos - w_lo).unsqueeze(0)

            row_lo = (h_lo * num_grid_per_side).unsqueeze(1)
            row_hi = (h_hi * num_grid_per_side).unsqueeze(1)
            col_lo = w_lo.unsqueeze(0)
            col_hi = w_hi.unsqueeze(0)

            # [h, w, 4]: corners in (lo,lo) (lo,hi) (hi,lo) (hi,hi) order.
            idx = torch.stack(
                (row_lo + col_lo, row_lo + col_hi, row_hi + col_lo, row_hi + col_hi), dim=-1
            )
            weights = torch.stack(
                ((1 - dh) * (1 - dw), (1 - dh) * dw, dh * (1 - dw), dh * dw), dim=-1
            )

            # Spatial-merge reorder; `reshape` is loud when a side does not divide.
            merged_h, merged_w = height // merge, width // merge
            idx = (
                idx.reshape(merged_h, merge, merged_w, merge, 4)
                .permute(0, 2, 1, 3, 4)
                .reshape(height * width, 4)
            )
            weights = (
                weights.reshape(merged_h, merge, merged_w, merge, 4)
                .permute(0, 2, 1, 3, 4)
                .reshape(height * width, 4)
            )

            if frames > 1:
                idx = idx.repeat(frames, 1)
                weights = weights.repeat(frames, 1)
            idx_parts.append(idx)
            weight_parts.append(weights)

        idx = idx_parts[0] if len(idx_parts) == 1 else torch.cat(idx_parts)
        weights = weight_parts[0] if len(weight_parts) == 1 else torch.cat(weight_parts)

        # One launch: each row of `idx` is a 4-element bag scaled by `weights`
        # and summed, fp32 table and fp32 weights.
        table = pos_embed_weight.float()
        out = F.embedding_bag(idx, table, mode="sum", per_sample_weights=weights)
        return out.to(pos_embed_weight.dtype)

    def reference(
        self,
        pos_embed_weight: torch.Tensor,
        grid_thw: Sequence[Sequence[int]],
        spatial_merge_size: int,
    ) -> torch.Tensor:
        """The explicit four-corner weighted sum in fp64, rounded once.

        Shares the entry's sample positions (fp32 `linspace`) but not its
        gather or accumulation; the test cross-checks the structure separately
        against `F.interpolate`.
        """
        num_grid = math.isqrt(pos_embed_weight.shape[0])
        hidden = pos_embed_weight.shape[1]
        grid = pos_embed_weight.double().view(num_grid, num_grid, hidden)
        device = pos_embed_weight.device
        outs = []
        for frames, height, width in grid_thw:
            frames, height, width = int(frames), int(height), int(width)
            h_pos = torch.linspace(0, num_grid - 1, height, device=device)
            w_pos = torch.linspace(0, num_grid - 1, width, device=device)
            h_lo, w_lo = h_pos.to(torch.int64), w_pos.to(torch.int64)
            h_hi = (h_lo + 1).clamp(max=num_grid - 1)
            w_hi = (w_lo + 1).clamp(max=num_grid - 1)
            d_h = (h_pos - h_lo).double().unsqueeze(1)
            d_w = (w_pos - w_lo).double().unsqueeze(0)
            raster = (
                grid[h_lo][:, w_lo] * ((1 - d_h) * (1 - d_w))[..., None]
                + grid[h_lo][:, w_hi] * ((1 - d_h) * d_w)[..., None]
                + grid[h_hi][:, w_lo] * (d_h * (1 - d_w))[..., None]
                + grid[h_hi][:, w_hi] * (d_h * d_w)[..., None]
            )
            rows = _merge_order(raster, height, width, spatial_merge_size)
            outs.append(rows.repeat(frames, 1) if frames > 1 else rows)
        out = torch.cat(outs) if len(outs) > 1 else outs[0]
        return out.to(pos_embed_weight.dtype)

    def is_valid(
        self,
        pos_embed_weight: torch.Tensor,
        grid_thw: Sequence[Sequence[int]],
        spatial_merge_size: int,
    ) -> None:
        # With 2300 rows isqrt truncates to 47, every gather index stays in
        # range, and the result is a correctly shaped resample of the wrong grid.
        num_rows = pos_embed_weight.shape[0]
        side = math.isqrt(num_rows)
        assert side * side == num_rows, (
            f"pos_embed_weight must hold a square grid; got {num_rows} rows, "
            f"which is not a perfect square"
        )


pos_embed_interpolate = PosEmbedInterpolate()
