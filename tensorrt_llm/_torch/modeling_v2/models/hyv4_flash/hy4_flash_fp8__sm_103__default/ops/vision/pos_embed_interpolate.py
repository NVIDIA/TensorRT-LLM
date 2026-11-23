# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vision position-embedding interpolation onto one frame's patch grid.

Hy4-Flash's vision tower resamples its learned patch-position table onto each
image or video frame's grid: `HYV4VLVisionPatchEmbed.interpolate_pos_encoding`
in the HF reference
(`src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py`). The table's row 0 is
the cls token and is skipped; the remaining `position_edge**2` rows are read as
a square grid, bilinear-resampled with `align_corners=False` and an explicit
`scale_factor` of `(size + 0.1) / position_edge` (the tower's own sampling grid,
not the plain `size=` interpolation SigLIP uses), cast to the patch
embeddings' dtype, and repeated `t` times for a video.

`__call__` allocates the output and makes one launch of the Triton JIT kernel
`_bilinear_pos_embed_kernel`: the four corner gathers, the bilinear weights,
the frame repeat and the raster row order in one pass. The kernel's launch
shape is adapted from the qwen4_exp family's
`qwen3_8_flash_next_fp8__sm_103__unified/ops/vision/triton_pos_embed_interpolate.py`
(another target's op, copied here, never imported); that entry samples
Qwen3-VL's corner-aligned grid in spatial-merge order, so only the
one-program-per-row structure carries over. Concatenating several frames' rows
is the caller's.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: This checkpoint's table: `max_image_size // patch_size` = 2048 // 16 = 128
#: positions per side, 128**2 + 1 rows (row 0 is the cls token) at the vision
#: tower's 1152 channels.
POSITION_EDGE = 128
HIDDEN = 1152


def _ckpt(t: int, h: int, w: int) -> dict:
    """The checkpoint's bf16 table at one `(t, h, w)` frame grid."""
    return dict(
        t=t,
        h=h,
        w=w,
        position_edge=POSITION_EDGE,
        dtype=torch.bfloat16,
        hidden=HIDDEN,
    )


# The grid sides vary per image: not specialized, so one compile serves every
# grid.
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
    """One output row: the bilinear blend of the table's four corner rows around
    the row's patch, in raster order, the frame repeated `T` times by the grid."""
    pid = tl.program_id(0)
    spatial = pid % (H * W)

    row = spatial // W
    col = spatial % W

    # torch's `area_pixel_compute_source_index` for bilinear, align_corners=False:
    # scale * (dst + 0.5) - 0.5, a negative result clamped to 0. `h_scale` is
    # `1 / scale_factor` as fp32, the same value torch derives from the
    # `scale_factor` the HF path passes.
    h_frac = tl.maximum((row.to(tl.float32) + 0.5) * h_scale - 0.5, 0.0)
    w_frac = tl.maximum((col.to(tl.float32) + 0.5) * w_scale - 0.5, 0.0)

    # Non-negative after the clamp, so truncation is floor (torch's `int h1 = h1r`).
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

    # The corner weights stay fp32 and the blend runs in fp32; the store rounds
    # once to the output dtype, as the reference's `.to(dtype)` does.
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
    """The source-coordinate scales torch's bilinear uses for one `(h, w)` grid.

    `F.interpolate` with an explicit `scale_factor` samples at `1 / scale_factor`
    -- except that its bilinear kernel copies the input unchanged when the output
    size equals the input size, and a grid at the table's own size computes
    exactly that size. Measured on torch 2.14: `128x128` on the 128-edge table is
    bit-exact against scale 1.0 and 7.3e-01 away from `1 / ((128 + 0.1) / 128)`,
    while `127x128`, `128x126` and `96x128` all follow `1 / scale_factor`. The
    identity is whole-op, not per-axis, so both axes take 1.0 only when both
    sides are the table's size.

    The scales are computed in double and used as fp32, the way torch derives
    them from the `scale_factor` the HF path passes.
    """
    if h == position_edge and w == position_edge:
        return 1.0, 1.0
    return 1.0 / ((h + 0.1) / position_edge), 1.0 / ((w + 0.1) / position_edge)


def _pos_embed_interpolate(
    embed_weight: torch.Tensor,
    t: int,
    h: int,
    w: int,
    position_edge: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """One launch over a `(t, h, w)` grid: `[t * h * w, hidden]` in raster order."""
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


class PosEmbedInterpolate(OpWrapper):
    """Resample the square `[position_edge**2 + 1, hidden]` table onto one
    `(t, h, w)` patch grid; return a fresh `[t*h*w, hidden]` tensor in `dtype`.

    Row 0 of `embed_weight` is the cls token and is not read. The remaining
    rows are the `position_edge x position_edge` grid, row-major: `h` indexes
    grid rows (stride `position_edge`), `w` columns. Bilinear, no corner
    alignment: output row `i` of `h` samples source position
    `(i + 0.5) * position_edge / (h + 0.1) - 0.5` (clamped at 0), `lo =
    floor`, `hi = min(lo + 1, position_edge - 1)`; the `+ 0.1` is the HF
    tower's own scale factor and is what distinguishes this op from SigLIP's
    `size=` interpolation. Rows come out in raster order -- row-major over
    `(h, w)` -- and `t` repeats the frame's `h*w` rows.

    A grid at the table's own size -- `h == w == position_edge`, the 2048x2048
    image -- is the identity: torch's bilinear copies its input when the output
    size equals the input size, so the `+ 0.1` grid never applies there. That is
    whole-op, not per-axis: a `96x128` grid follows the `+ 0.1` grid on both
    axes. See `_source_scales`.

    Arithmetic: the four corner weights and the blend are fp32; the output is
    rounded once to `dtype`, which is the HF path's `.float()` interpolation
    followed by `.to(embeddings.dtype)`.

    Fusion boundary: one frame grid's position embeddings, repeated over `t`.
    The patch embeddings, their addition, the merger and joining images are the
    caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: t, h, w, position_edge, dtype, hidden. The tower is replicated on
    # every rank, so these are every member's shapes.
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
    Source: a Triton JIT kernel, compiled at first call for the device it runs
    on (the cells ran on sm_103). `H` and `W` are int32 and not specialized, so
    every grid shares one compile; `h_scale`/`w_scale` are fp32, and `EDGE`
    (128), `HIDDEN` (1152) and `BLOCK_D` (2048) are constexprs, so another
    table geometry compiles another kernel. Not a registered
    `torch.ops.trtllm` op.

    Accuracy over the six cells, in bf16 ulp of `assert_within_ulp`'s row-scale
    unit, element / relative RMS:
    - against the entry's fp64 reference: at most 1.896 / 0.041, 99.74% of
      elements bit-exact, the identity grid 100%;
    - against the HF path on the same table: at most 0.955 / 0.0027, 99.997%
      bit-exact, the identity grid 100%.
    The test gates 3.0 / 1.0. A boundary flip moves an element by one bf16 ulp
    of its own value, which reads up to 2.0 in this unit, so 3.0 is the derived
    ceiling rather than a measured one. The closest wrong reading of the grid
    is the identity taken per-axis (53.1 ulp); `size=` without the `+ 0.1` is
    116.2, corner alignment 214.6, the cls row 581.1, and the qwen4_exp row
    order 629.9. The qwen4_exp entry's bf16-weight arithmetic sits at 1.984 --
    inside the band: a different arithmetic, not a different grid, so the test
    measures it and does not refuse it.

    The identity at the table's own size is torch's, measured on 2.14:
    `128x128` is bit-exact against scale 1.0 and 7.3e-01 away from the `+ 0.1`
    grid, while `127x128`, `128x126` and `96x128` follow `1 / scale_factor`. It
    is whole-op, not per-axis.

    Silent, refused by `is_valid` (each measured on GB300):
    - a non-contiguous table: the kernel reads rows of `shape[1]` at stride
      `shape[1]`. A column slice of a buffer twice as wide (strides
      (2304, 1)) returned the right shape, differing from the same values made
      contiguous by up to 4.984.
    - `position_edge ** 2 + 1 != embed_weight.shape[0]`: the grid side is an
      argument, not read off the table. 64 on the 128-edge table returned a
      correctly shaped result 4.258 off.
    - a 3-D table: `hidden` is `shape[1]`, so the output's width becomes that
      dimension (a [2, 9, 1152] table with edge 8 returned [16, 9]). Passed as
      [1, 16385, 1152] the rows are read far past the tensor.
    - an empty side (`t`, `h` or `w` of 0): returns `[0, hidden]`, no error,
      where the HF path's `F.interpolate` raises on a zero output size.

    Hard limits: the output offset is `program_id * hidden` in int32, so
    `t*h*w*hidden` must stay below 2^31 (1,864,135 rows at hidden 1152; the
    envelope's largest total is 16384 rows, the video pixel budget). The grid
    sides are host ints, exact in fp32 up to 2^24 rows.

    Not certified: a table dtype other than bf16, a `dtype` other than the
    table's, a `position_edge` other than 128, a `hidden` other than 1152, a
    CPU table, `t*h*w` above 16384 rows, a grid side above `position_edge`, and
    CUDA-graph capture (the vision tower is eager; the grid sides are host
    arguments, frozen at capture).
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
        return _pos_embed_interpolate(embed_weight, t, h, w, position_edge, dtype)

    def is_valid(
        self,
        embed_weight: torch.Tensor,
        t: int,
        h: int,
        w: int,
        position_edge: int,
        dtype: torch.dtype,
    ) -> None:
        if embed_weight.dim() != 2:
            raise AssertionError(
                f"embed_weight must be 2-D [position_edge**2 + 1, hidden], got {embed_weight.dim()}-D"
            )
        if not embed_weight.is_contiguous():
            raise AssertionError(
                f"embed_weight must be contiguous; the kernel reads rows of shape[1] at "
                f"stride shape[1], strides {tuple(embed_weight.stride())}"
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
        """The explicit four-corner weighted sum in fp64, rounded once to `dtype`.

        Sample positions are `(i + 0.5) * scale - 0.5` in fp64, clamped at 0, with
        `scale` from `_source_scales` -- the same grid the op samples, including
        the identity at the table's own size. `dtype` is applied at the end, so
        both sides carry one rounding.
        """
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
