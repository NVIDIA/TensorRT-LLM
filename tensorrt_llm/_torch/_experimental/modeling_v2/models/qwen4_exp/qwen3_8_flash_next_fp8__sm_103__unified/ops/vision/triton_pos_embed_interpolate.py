# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vision position-embedding bilinear interpolation onto one image's patch
grid, emitted in spatial-merge order, by the stock Qwen3-VL tower's fused
Triton kernel.

`__call__` allocates the output and makes one launch of the Triton JIT kernel
`_bilinear_pos_embed_kernel`: the four corner gathers, the bilinear weights and
the merge-order row permutation in one pass. The kernel and its launch are the
ones `tensorrt_llm/_torch/models/modeling_qwen3vl.py` runs for the stock vision
towers (`Qwen3VisionModel.fast_pos_embed_interpolate`, one call per image), with
the grid sides left unspecialized so one compile serves every image.
Concatenating several images' rows is the caller's. The FP8 unified target's entry; the
NVFP4 default target keeps its own copy (the two checkpoints share the bf16 vision tower).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: This checkpoint: num_position_embeddings 2304 (a 48 x 48 grid), hidden 1152,
#: spatial_merge_size 2.
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


# The grid sides vary per image: not specialized, so one compile serves every grid.
@triton.jit(do_not_specialize=["H", "W"])
def _bilinear_pos_embed_kernel(
    embed_ptr,
    output_ptr,
    H,
    W,
    h_scale,
    w_scale,
    NUM_GRID: tl.constexpr,
    M_SIZE: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One output row: the bilinear blend of the table's four corner rows around the
    row's patch, the patch taken in spatial-merge order."""
    pid = tl.program_id(0)
    total_spatial = H * W
    spatial_idx = pid % total_spatial

    num_blocks_w = W // M_SIZE
    block_idx = spatial_idx // (M_SIZE * M_SIZE)
    local_idx = spatial_idx % (M_SIZE * M_SIZE)
    br = block_idx // num_blocks_w
    bc = block_idx % num_blocks_w
    lr = local_idx // M_SIZE
    lc = local_idx % M_SIZE
    row = br * M_SIZE + lr
    col = bc * M_SIZE + lc

    h_frac = row.to(tl.float32) * h_scale
    w_frac = col.to(tl.float32) * w_scale

    hf = tl.math.floor(h_frac).to(tl.int32)
    wf = tl.math.floor(w_frac).to(tl.int32)
    hc = tl.minimum(hf + 1, NUM_GRID - 1)
    wc = tl.minimum(wf + 1, NUM_GRID - 1)

    dh = h_frac - hf.to(tl.float32)
    dw = w_frac - wf.to(tl.float32)
    w11 = dh * dw
    w10 = dh - w11
    w01 = dw - w11
    w00 = 1.0 - dh - w01

    off00 = (hf * NUM_GRID + wf) * HIDDEN_DIM
    off01 = (hf * NUM_GRID + wc) * HIDDEN_DIM
    off10 = (hc * NUM_GRID + wf) * HIDDEN_DIM
    off11 = (hc * NUM_GRID + wc) * HIDDEN_DIM
    out_off = pid * HIDDEN_DIM

    # The corner weights rounded to the output dtype; the multiply-adds run in it.
    out_dtype = output_ptr.dtype.element_ty
    w00_c = w00.to(out_dtype)
    w01_c = w01.to(out_dtype)
    w10_c = w10.to(out_dtype)
    w11_c = w11.to(out_dtype)

    for d in tl.range(0, HIDDEN_DIM, BLOCK_D):
        cols = d + tl.arange(0, BLOCK_D)
        mask = cols < HIDDEN_DIM

        e00 = tl.load(embed_ptr + off00 + cols, mask=mask)
        e01 = tl.load(embed_ptr + off01 + cols, mask=mask)
        e10 = tl.load(embed_ptr + off10 + cols, mask=mask)
        e11 = tl.load(embed_ptr + off11 + cols, mask=mask)

        val = w00_c * e00 + w01_c * e01 + w10_c * e10 + w11_c * e11

        tl.store(output_ptr + out_off + cols, val, mask=mask)


def _pos_embed_interpolate(
    embed_weight: torch.Tensor,
    t: int,
    h: int,
    w: int,
    num_grid_per_side: int,
    m_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """One launch over a `(t, h, w)` grid: `[t * h * w, hidden]` in spatial-merge order."""
    hidden_dim = embed_weight.shape[1]
    total_out = t * h * w
    output = torch.empty(total_out, hidden_dim, device=embed_weight.device, dtype=dtype)
    h_scale = (num_grid_per_side - 1) / (h - 1) if h > 1 else 0.0
    w_scale = (num_grid_per_side - 1) / (w - 1) if w > 1 else 0.0
    _bilinear_pos_embed_kernel[(total_out,)](
        embed_weight,
        output,
        h,
        w,
        h_scale,
        w_scale,
        num_grid_per_side,
        m_size,
        hidden_dim,
        triton.next_power_of_2(hidden_dim),
    )
    return output


class TritonPosEmbedInterpolate(OpWrapper):
    """Resample a square row-major `[N*N, hidden]` table onto one `(t, h, w)`
    patch grid; return a fresh `[t*h*w, hidden]` tensor in `dtype`.

    Corner-aligned bilinear (`align_corners=True`): row `i` of `h` samples the
    table at `i * (N - 1) / (h - 1)` (0 when `h == 1`), `lo = floor`,
    `hi = min(lo + 1, N - 1)`; `h` indexes table rows (stride `N`), `w`
    columns. Rows come out in spatial-merge order -- whole `m_size x m_size`
    blocks row-major, row-major inside a block -- which is what the patch
    merger's 4-to-1 reshape consumes. `t` repeats the frame's rows.

    Arithmetic is the HF reference's (`fast_pos_embed_interpolate`): the four
    corner weights are computed in fp32, rounded to `dtype`, and the
    multiply-adds run in `dtype`. Not an fp32 accumulation; see `note`.

    Fusion boundary: one image's position embeddings. Adding them to the patch
    embeddings, the 2-D rope, and joining images are the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: t, h, w, num_grid_per_side, m_size, dtype, hidden. The tower is
    # replicated on every rank, so these are every member's shapes.
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
    Source: a Triton JIT kernel, compiled at first call for the device it runs
    on: the stock Qwen3-VL towers' kernel (`modeling_qwen3vl`), kept here so its
    grid sides can stay unspecialized. `H` and `W` are int32 and not specialized,
    so every grid shares one compile (the tower's `warm_up` makes it at load; the
    test holds one variant); the scales are fp32, and `NUM_GRID`, `M_SIZE`,
    `HIDDEN_DIM` (1152) and `BLOCK_D` (2048) are constexprs, so another table
    geometry compiles another kernel. Not a registered `torch.ops.trtllm` op.

    Accuracy, against the exact (fp64) corner-aligned bilinear value, as
    row-scale element error / relative RMS in bf16 ulp (`assert_within_ulp`):
    at most 3.522 / 0.752 over 50 seeds of the test's smooth table (a bicubic
    upsampled normal field, neighbour cosine 0.911 against the checkpoint
    table's 0.907), and at most 3.349 / 0.745 on the checkpoint's own
    `model.visual.pos_embed.weight`. The test gates 4 / 1. The closest wrong
    variant (align_corners=False) lands at 41.3 / 15.2 or more; raster row
    order and swapped table axes above 400 / 300.

    Not certified on tables whose neighbouring rows are uncorrelated: a normal
    table with independent rows reaches 4.182 / 0.879 over 20 seeds.
    Averaging uncorrelated corners shrinks the output's row maximum below the
    corners' magnitude, and the bf16 rounding error scales with the corners. A
    learned 2-D position table is not such a table.

    Silent, refused by `is_valid` (each measured on GB300):
    - a strided table: the kernel indexes `embed_weight` as dense rows of
      `hidden`. A column slice of a wider buffer returned, differing from the
      same values made contiguous by up to 5.05.
    - `num_grid_per_side ** 2 != embed_weight.shape[0]`: the grid side is an
      argument, not read off the table. 47 on the 2304-row table returned a
      correctly shaped result 564 / 340 ulp off.
    - an empty side (`t`, `h` or `w` of 0): returns `[0, hidden]`, no error.
    - `h` or `w` not divisible by `m_size`: a 6x9 grid returned a finite
      `[54, 1152]` tensor, its rows read from positions outside the grid and the
      table.

    Hard limits: the output offset is `program_id * hidden` in int32, so
    `t*h*w*hidden` must stay below 2^31 (1,864,135 rows at hidden 1152; the
    tower's per-call cap is 32768). Positions are `row * scale` in fp32,
    exact up to 2^24 rows per side.

    Not certified: fp16 or fp32 tables, a `dtype` other than the table's,
    `m_size` other than 2, a hidden size other than 1152, a CPU table, and
    CUDA-graph capture (the call site, the vision tower, is eager only; the
    grid is a host argument, frozen at capture).
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
        return _pos_embed_interpolate(embed_weight, t, h, w, num_grid_per_side, m_size, dtype)

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
        """The explicit four-corner weighted sum in fp64, returned as fp32.

        Sample positions are the exact `i * (N - 1) / (size - 1)` in fp64.
        `dtype` is not applied: the result is the exact value rounded once to
        fp32, so a test's band in `dtype` ulp measures the op's error alone
        rather than adding the reference's own rounding to `dtype`.
        """
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
            f"the kernel would read rows outside the grid and the table"
        )


triton_pos_embed_interpolate = TritonPosEmbedInterpolate()
