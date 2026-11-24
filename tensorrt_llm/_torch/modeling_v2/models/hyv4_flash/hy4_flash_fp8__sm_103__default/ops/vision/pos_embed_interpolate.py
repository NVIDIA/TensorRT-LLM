# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vision position-embedding interpolation onto one frame's patch grid (HF `interpolate_pos_encoding`), in Triton."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: `max_image_size // patch_size` = 2048 // 16 positions per side; the table has 128**2 + 1 rows (row 0 is cls).
POSITION_EDGE = 128
HIDDEN = 1152


def _ckpt(t: int, h: int, w: int) -> dict:
    return dict(
        t=t,
        h=h,
        w=w,
        position_edge=POSITION_EDGE,
        dtype=torch.bfloat16,
        hidden=HIDDEN,
    )


@triton.jit(do_not_specialize=["H", "W"])
def _bilinear_pos_embed_kernel(
    embed_ptr,
    output_ptr,
    H,
    W,
    h_scale,
    w_scale,
    EDGE: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One output row: the fp32 bilinear blend of the four table rows around its patch, in raster order."""
    pid = tl.program_id(0)
    spatial = pid % (H * W)

    row = spatial // W
    col = spatial % W

    # torch's bilinear source index with align_corners=False: scale * (dst + 0.5) - 0.5, clamped at 0.
    h_frac = tl.maximum((row.to(tl.float32) + 0.5) * h_scale - 0.5, 0.0)
    w_frac = tl.maximum((col.to(tl.float32) + 0.5) * w_scale - 0.5, 0.0)

    hf = h_frac.to(tl.int32)
    wf = w_frac.to(tl.int32)
    hc = tl.minimum(hf + 1, EDGE - 1)
    wc = tl.minimum(wf + 1, EDGE - 1)

    dh = h_frac - hf.to(tl.float32)
    dw = w_frac - wf.to(tl.float32)
    h0 = 1.0 - dh
    w0 = 1.0 - dw

    # The table's row 0 is the cls token: the patch grid starts at row 1.
    r00 = embed_ptr + (1 + hf * EDGE + wf) * HIDDEN
    r01 = embed_ptr + (1 + hf * EDGE + wc) * HIDDEN
    r10 = embed_ptr + (1 + hc * EDGE + wf) * HIDDEN
    r11 = embed_ptr + (1 + hc * EDGE + wc) * HIDDEN
    out = output_ptr + pid * HIDDEN

    for d in tl.range(0, HIDDEN, BLOCK_D):
        cols = d + tl.arange(0, BLOCK_D)
        mask = cols < HIDDEN

        e00 = tl.load(r00 + cols, mask=mask).to(tl.float32)
        e01 = tl.load(r01 + cols, mask=mask).to(tl.float32)
        e10 = tl.load(r10 + cols, mask=mask).to(tl.float32)
        e11 = tl.load(r11 + cols, mask=mask).to(tl.float32)

        top = w0 * e00 + dw * e01
        bot = w0 * e10 + dw * e11
        tl.store(out + cols, h0 * top + dh * bot, mask=mask)


def _source_scales(h: int, w: int, position_edge: int) -> tuple[float, float]:
    """Source-coordinate scales `1 / scale_factor` for the HF `scale_factor=(size + 0.1) / edge`.

    torch's bilinear copies its input when the output size equals the input size, so a grid at the table's own
    size on both axes is the identity (scale 1.0); this is whole-op, not per-axis.
    """
    if h == position_edge and w == position_edge:
        return 1.0, 1.0
    return 1.0 / ((h + 0.1) / position_edge), 1.0 / ((w + 0.1) / position_edge)


class PosEmbedInterpolate(OpWrapper):
    """Resample the `[position_edge**2 + 1, hidden]` table (row 0 cls, skipped) onto a `(t, h, w)` grid.

    Returns a fresh `[t*h*w, hidden]` tensor in `dtype`: raster order over `(h, w)`, the frame repeated `t` times.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one 32x32 image: 1024 rows, the processor's 512-pixel minimum",
            spec=_ckpt(1, 32, 32),
        ),
        Cell(
            why="a 48x80 image: 3840 rows, a 768x1280 resize, both sides off the table's grid",
            spec=_ckpt(1, 48, 80),
        ),
        Cell(
            why="an 8-frame 32x64 video: 16384 rows, the pixel budget's largest total",
            spec=_ckpt(8, 32, 64),
        ),
        Cell(
            why="a 1024-frame 4x4 video: the maximum frame count, 16 rows per frame",
            spec=_ckpt(1024, 4, 4),
        ),
        Cell(
            why="128x128: 16384 rows, the table's own grid and the single-frame maximum",
            spec=_ckpt(1, 128, 128),
        ),
        Cell(
            why="96x128: 12288 rows, one side at the table's size, the other not",
            spec=_ckpt(1, 96, 128),
        ),
    )

    note = """
    The output offset `program_id * hidden` is int32, so `t*h*w*hidden` must stay below 2^31.
    """

    def __call__(
        self,
        embed_weight: torch.Tensor,
        t: int,
        h: int,
        w: int,
        position_edge: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        hidden = embed_weight.shape[1]
        total = t * h * w
        output = torch.empty(total, hidden, device=embed_weight.device, dtype=dtype)
        h_scale, w_scale = _source_scales(h, w, position_edge)
        _bilinear_pos_embed_kernel[(total,)](
            embed_weight,
            output,
            h,
            w,
            h_scale,
            w_scale,
            position_edge,
            hidden,
            triton.next_power_of_2(hidden),
        )
        return output

    def is_valid(
        self,
        embed_weight: torch.Tensor,
        t: int,
        h: int,
        w: int,
        position_edge: int,
        dtype: torch.dtype,
    ) -> None:
        # Each violation below returns a plausible result instead of raising.
        if embed_weight.dim() != 2:
            raise AssertionError(f"embed_weight must be 2-D, got {embed_weight.dim()}-D")
        if not embed_weight.is_contiguous():
            raise AssertionError(
                f"embed_weight must be contiguous, got strides {tuple(embed_weight.stride())}"
            )
        if position_edge * position_edge + 1 != embed_weight.shape[0]:
            raise AssertionError(
                f"position_edge {position_edge} does not describe {embed_weight.shape[0]} rows"
            )
        if min(t, h, w) <= 0:
            raise AssertionError(f"t, h, w must be positive, got t={t} h={h} w={w}")

    def reference(
        self,
        embed_weight: torch.Tensor,
        t: int,
        h: int,
        w: int,
        position_edge: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """The four-corner weighted sum in fp64 on the `_source_scales` grid, rounded once to `dtype`."""
        edge = position_edge
        hidden = embed_weight.shape[1]
        grid = embed_weight.double()[1:].reshape(edge, edge, hidden)
        dev = grid.device

        h_scale, w_scale = _source_scales(h, w, edge)
        src_h = (torch.arange(h, dtype=torch.float64, device=dev) + 0.5) * h_scale - 0.5
        src_w = (torch.arange(w, dtype=torch.float64, device=dev) + 0.5) * w_scale - 0.5
        src_h = src_h.clamp_min(0.0)
        src_w = src_w.clamp_min(0.0)
        hf = src_h.floor().to(torch.int64)
        wf = src_w.floor().to(torch.int64)
        hc = (hf + 1).clamp_max(edge - 1)
        wc = (wf + 1).clamp_max(edge - 1)
        dh = (src_h - hf.double()).view(h, 1, 1)
        dw = (src_w - wf.double()).view(1, w, 1)

        top = (1.0 - dw) * grid[hf][:, wf] + dw * grid[hf][:, wc]
        bot = (1.0 - dw) * grid[hc][:, wf] + dw * grid[hc][:, wc]
        rows = ((1.0 - dh) * top + dh * bot).reshape(h * w, hidden)
        return rows.repeat(t, 1).to(dtype)


pos_embed_interpolate = PosEmbedInterpolate()
