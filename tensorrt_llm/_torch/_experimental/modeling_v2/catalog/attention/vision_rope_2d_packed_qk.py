# SPDX-FileCopyrightText: Copyright (c) 2025, Tri Dao.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Adapted from flash-attn 2 (https://github.com/Dao-AILab/flash-attention),
# `flash_attn/ops/triton/rotary.py`: `rotary_kernel` and `apply_rotary`.
"""Vision-tower 2-D rotary, in place on the q|k prefix of the packed qkv projection.

The rotation half of the vision tower's 2-D axial rotary (`vision_rope_2d__r52` is the
whole op, angle table included). The angle table depends only on the request's
`grid_thw`, so the caller builds it once per tower forward -- spatial-merge `(row, col)`
per token, fp64 angles rounded once to fp32, then `cos` / `sin` -- and every one of the 27
blocks hands it here with that block's qkv projection output.

Tier 5: one Triton launch, flash-attn 2's `rotary_kernel` (BSD-3-Clause, header above)
cut down to the path this call takes. flash-attn 2.x is not a pinned dependency
(`requirements.txt` pins only flash-attn 4, and the stock Qwen VL towers guard the 2.x
import), so the kernel is vendored here rather than imported. What changed from the
origin: one sequence (no batch axis, `cu_seqlens` or `seqlen_offsets`), NeoX pairing only
(no `INTERLEAVED` or `CONJUGATE`), int64 token offsets, in place only, a direct
`kernel[grid]` launch instead of `torch.library.wrap_triton`, and 4 heads per program
where the origin takes 2 (see `note`). The arithmetic, `BLOCK_M` 8 and the masked
half-width loads are the origin's.

Nothing in the tree serves it in one launch: the registered rope ops take 1-D
positions, `fused_dit_qk_norm_rope` always RMS-normalizes q and k, and FlashInfer's
`apply_rope_with_cos_sin_cache_inplace` raised "misaligned address" at head size 72
(measured by the tp1 target's `ops/attention/vision_rope_2d`).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .._op import Arch, Cell, OpWrapper

#: This checkpoint's tower: 1152 / 16 heads = a 72-wide head, spatial_merge_size 2, theta 1e4
#: (`model.visual.blocks.N.attn.qkv.weight` is [3456, 1152], so a qkv row is 3 * 16 * 72).
NUM_HEADS = 16
HEAD_DIM = 72

#: Heads x tokens per program: the origin's 8 tokens for a rotary width <= 128, and 4 heads
#: (measured faster than the origin's 2 at every cell; see `note`).
_BLOCK_H = 4
_BLOCK_M = 8


@triton.jit
def _rotary_neox_inplace_kernel(
    X,
    COS,
    SIN,
    seqlen,
    nheads,
    seqlen_ro,
    stride_x_seqlen,
    stride_x_nheads,
    stride_x_headdim,
    # ROTARY_DIM is constexpr so the compiler sees the mask is constant every 8 elements
    # and issues 128-bit loads (the origin's comment).
    ROTARY_DIM: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    ROTARY_DIM_HALF: tl.constexpr = ROTARY_DIM // 2
    pid_head = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    if pid_m * BLOCK_M >= seqlen:
        return

    rh = pid_head * BLOCK_H + tl.arange(0, BLOCK_H)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # Token offsets in int64: rows x row stride passes 2^31 at 621379 rows of a 3456-wide row.
    rm64 = rm.to(tl.int64)
    rk_half = tl.arange(0, BLOCK_K // 2)

    COS = COS + (rm64[:, None] * ROTARY_DIM_HALF + rk_half[None, :])
    SIN = SIN + (rm64[:, None] * ROTARY_DIM_HALF + rk_half[None, :])
    mask_cs = (rm[:, None] < seqlen_ro) & (rk_half[None, :] < ROTARY_DIM_HALF)
    cos = tl.load(COS, mask=mask_cs, other=1.0).to(tl.float32)
    sin = tl.load(SIN, mask=mask_cs, other=0.0).to(tl.float32)

    # The 1st and 2nd halves of each head, rotated as pairs (i, i + ROTARY_DIM_HALF), stored
    # back where they were read.
    X = X + (
        rh[:, None, None] * stride_x_nheads
        + rm64[None, :, None] * stride_x_seqlen
        + rk_half[None, None, :] * stride_x_headdim
    )
    mask = (
        (rh[:, None, None] < nheads)
        & (rm[None, :, None] < seqlen)
        & (rk_half[None, None, :] < ROTARY_DIM_HALF)
    )
    x0 = tl.load(X, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(X + ROTARY_DIM_HALF * stride_x_headdim, mask=mask, other=0.0).to(tl.float32)
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos
    tl.store(X, o0, mask=mask)
    tl.store(X + ROTARY_DIM_HALF * stride_x_headdim, o1, mask=mask)


class VisionRope2dPackedQk(OpWrapper):
    """Rotate every head of `qk` by its token's angles, in place; return `qk`.

        half = head_dim // 2
        qk[t, h, i]        <- x[i] cos[t, i] - x[i + half] sin[t, i]
        qk[t, h, i + half] <- x[i] sin[t, i] + x[i + half] cos[t, i]        i < half

    NeoX half-split pairing `(i, i + half)` over the whole head, in fp32, rounded once on
    the store. `qk` is the `[T, 2 * heads, head_dim]` q|k prefix of the tower's packed
    `[T, 3 * heads, head_dim]` qkv projection, read and written through its strides; the
    v heads that follow in each row are never addressed. `cos` / `sin` are the caller's
    per-token fp32 tables `[T, half]`: the height angles, then the width angles.

    Fusion boundary: the rotation. The angle table, the qkv projection, the split into
    q / k / v and the attention are the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: grid -- `grid_thw` rows (frames, height, width) in patch units, from which
    # the test builds the caller's angle table the way the target does; grid_dtype --
    # the integer dtype of that host `grid_thw` (it reaches only the table build);
    # dtype -- the qkv projection's dtype. Heads 16 x 72, merge 2, theta 1e4 throughout.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="62x90 serving image, int32 host grid: 5580 rows",
            spec=dict(grid=((1, 62, 90),), grid_dtype=torch.int32, dtype=torch.bfloat16),
        ),
        Cell(
            why="62x90 with the processor's int64 grid_thw: 5580 rows",
            spec=dict(grid=((1, 62, 90),), grid_dtype=torch.int64, dtype=torch.bfloat16),
        ),
        Cell(
            why="video (3,8,8) + image (1,16,24), two grid rows: 576 rows",
            spec=dict(grid=((3, 8, 8), (1, 16, 24)), grid_dtype=torch.int64, dtype=torch.bfloat16),
        ),
        Cell(
            why="video (64,16,32): 32768 rows, the tower's per-call ceiling",
            spec=dict(grid=((64, 16, 32),), grid_dtype=torch.int64, dtype=torch.bfloat16),
        ),
    )

    note = """
    Metadata consumed: none. Call site: the VisionTower's 27 blocks (multimodal unit),
    eager only -- the table bakes in the request's grid. Rank-invariant: every member
    (tp4, tep4, dep4) runs the tower at these shapes.

    Source: Triton, JIT-compiled at first call for the running device; the cells ran that
    path on sm_103 (GB300). Launch: `(cdiv(heads, 4), cdiv(T, 8))` programs, one per 4
    heads x 8 tokens, each loading its 8 tokens' cos / sin once and both half-heads of
    its 4 heads with 64-lane masked loads (36 live). Token offsets are int64, so no
    address product overflows (the origin's int32 product would pass 2^31 at 621379 rows
    of the 3456-wide qkv row); the cos / sin index is `t * half`.

    Cost on GB300, 50 calls: device time (back to back in one CUDA graph) 2.51 / 11.39 /
    74.72 us at 576 / 5580 / 32768 rows (4.2 TB/s of q|k read + write plus the table at
    32768); eager, as the tower calls it, 15.6 / 16.4 / 75.8 us between events -- host
    launch bound below a few thousand rows. The origin's 2 heads per program measured
    2.76 / 12.95 / 86.03 us device time on the same operands; 4 x 8 was the fastest or
    within 3% of it among 15 (heads, tokens, warps) shapes swept from 2 to 16 heads and
    2 to 64 tokens. For scale, the composed torch rotation it replaces
    (`vision_rope_2d__r52`, table excluded) measured 206 / 1217 us at 5580 / 32768
    rows, and flash-attn's own `apply_rotary` 108 / 107 us (host bound).

    Preconditions: `qk` 3-D, any strides (the kernel reads all three); `cos` and `sin`
    fp32, contiguous, both `[T, head_dim // 2]`. In place: the return value is `qk`
    itself, and the caller's qkv buffer holds the rotated q|k on return.

    Numerics: bitwise equal to `reference` (a complex multiply in fp32, rounded once) and
    to `vision_rope_2d__r52.reference` (positions by nested loops, fp64 angles, `polar`)
    at every cell, q and k alike: relative L2 0, cosine 1.

    Silent, refused by `is_valid` (each demonstrated unguarded in the test):
      - a bf16 (or any non-fp32) table: the kernel widens it and rotates by rounded
        angles;
      - a table with fewer rows than `qk`: rows past it load cos 1 / sin 0, so the tail
        tokens come back unrotated;
      - a table narrower than half the head: it rotates a prefix as pairs
        `(i, i + width)` and leaves the rest of the head untouched (partial rotary);
        wider would write across the next head;
      - a non-contiguous table (e.g. the first half of a `[T, head_dim]` duplicated
        table): it is indexed as `t * half + i` and reads the wrong angles.

    Not certified: float16 / float32 / float8 q|k; head geometries other than 16 x 72;
    partial rotary; grids past the cells (sides past 90, calls past 32768 rows); T = 0;
    CUDA-graph capture; out-of-place use.
    """

    def __call__(self, qk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        seqlen, nheads, _ = qk.shape
        rotary_dim = 2 * cos.shape[-1]
        grid = (triton.cdiv(nheads, _BLOCK_H), triton.cdiv(seqlen, _BLOCK_M))
        _rotary_neox_inplace_kernel[grid](
            qk,
            cos,
            sin,
            seqlen,
            nheads,
            cos.shape[0],
            qk.stride(0),
            qk.stride(1),
            qk.stride(2),
            ROTARY_DIM=rotary_dim,
            BLOCK_K=triton.next_power_of_2(rotary_dim),
            BLOCK_H=_BLOCK_H,
            BLOCK_M=_BLOCK_M,
        )
        return qk

    def reference(self, qk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """A fresh tensor: each head's halves as one complex number, times `cos + i sin`, in
        fp32; rounded once to `qk`'s dtype. `qk` is not mutated."""
        half = qk.shape[-1] // 2
        x = qk.float()
        rotor = torch.complex(cos.float(), sin.float()).unsqueeze(-2)
        rotated = torch.complex(x[..., :half], x[..., half:]) * rotor
        return torch.cat((rotated.real, rotated.imag), dim=-1).to(qk.dtype)

    def is_valid(self, qk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
        assert cos.dtype == torch.float32 and sin.dtype == torch.float32, (
            f"cos and sin must be fp32; got {cos.dtype} / {sin.dtype} -- the kernel widens a "
            "narrower table and rotates by its rounded angles"
        )
        half = qk.shape[-1] // 2
        assert cos.shape == (qk.shape[0], half) and sin.shape == cos.shape, (
            f"cos and sin must be [T, head_dim // 2] = [{qk.shape[0]}, {half}]; got "
            f"{tuple(cos.shape)} / {tuple(sin.shape)} -- a short table leaves the tail tokens "
            "unrotated, a narrow one rotates only a prefix of each head"
        )
        assert cos.is_contiguous() and sin.is_contiguous(), (
            "cos and sin must be contiguous; the kernel indexes them as t * half + i and a "
            "strided table reads the wrong angles"
        )


vision_rope_2d_packed_qk = VisionRope2dPackedQk()
