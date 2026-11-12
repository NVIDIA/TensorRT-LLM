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

The caller builds the per-token `cos` / `sin` table once per tower forward from `grid_thw`.
The kernel is flash-attn 2's `rotary_kernel` (header above) cut down to one sequence, NeoX
pairing, int64 token offsets and in-place writes; the registered rope ops take 1-D positions only.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: This checkpoint's bf16 tower: 1152 / 16 heads = a 72-wide head, a qkv row of 3 * 16 * 72.
NUM_HEADS = 16
HEAD_DIM = 72

#: Heads x tokens per program.
_BLOCK_H = 4
_BLOCK_M = 8


@triton.jit(do_not_specialize=["seqlen", "seqlen_ro"])
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
    # Constexpr so the compiler sees the mask is constant every 8 elements (128-bit loads).
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

    # Pairs (i, i + ROTARY_DIM_HALF), stored back where they were read.
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

    In fp32, rounded once on the store. `qk` is the `[T, 2 * heads, head_dim]` q|k prefix of
    the packed `[T, 3 * heads, head_dim]` qkv, addressed through its strides (the v heads are
    never touched). `cos` / `sin` are fp32 `[T, half]`: the height angles, then the width angles.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: grid -- `grid_thw` rows the test builds the angle table from; grid_dtype -- its
    # host dtype; dtype -- the qkv dtype. Heads 16 x 72, merge 2, theta 1e4 throughout.
    CELLS: tuple[Cell, ...] = (
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
    Source: the Triton kernel above. Called eagerly by every vision block; the tower is
    replicated, so every member runs these shapes. Bitwise equal to `reference` at every cell.

    Preconditions: `qk` 3-D, any strides; `cos` and `sin` fp32, contiguous, `[T, head_dim // 2]`.
    `is_valid` refuses the silent failures: a non-fp32 table (rotates by rounded angles), a
    short table (tail tokens unrotated), a narrow or wide table (partial rotation, or the last
    k head writes into the first v head), a strided table (wrong angles).

    Not certified: float16 / float32 / float8 q|k; head geometries other than 16 x 72; partial
    rotary; calls past 32768 rows (the tower's per-call ceiling); T = 0; CUDA-graph capture;
    out-of-place use.
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
        """Each head's halves as one complex number times `cos + i sin`, in fp32; `qk` is not
        mutated."""
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
