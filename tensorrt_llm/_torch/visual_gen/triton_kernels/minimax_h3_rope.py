# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 split-half RoPE with eager BF16 rounding boundaries."""

import torch
import triton
import triton.language as tl


@triton.jit
def _minimax_h3_rope_kernel(
    X,
    C,
    S,
    Y,
    N: tl.constexpr,
    SEQ: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    ROT: tl.constexpr,
    XB: tl.constexpr,
    XS: tl.constexpr,
    XH: tl.constexpr,
    XD: tl.constexpr,
    CS: tl.constexpr,
    CD: tl.constexpr,
    SS: tl.constexpr,
    SD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    d = i % DIM
    h = i // DIM % HEADS
    s = i // (DIM * HEADS) % SEQ
    b = i // (DIM * HEADS * SEQ)
    x = tl.load(X + b * XB + s * XS + h * XH + d * XD, i < N, 0).to(tl.float32)
    partner = tl.where(d < ROT // 2, d + ROT // 2, d - ROT // 2)
    xp = tl.load(
        X + b * XB + s * XS + h * XH + partner * XD,
        (i < N) & (d < ROT),
        0,
    ).to(tl.float32)
    rotated = tl.where(d < ROT // 2, -xp, xp)
    c = tl.load(C + s * CS + d * CD, (i < N) & (d < ROT), 0)
    sn = tl.load(S + s * SS + d * SD, (i < N) & (d < ROT), 0)
    # Eager casts the tables and materializes both products in BF16 before
    # adding them. Keeping these boundaries prevents output drift from fusion.
    c = c.to(tl.bfloat16).to(tl.float32)
    sn = sn.to(tl.bfloat16).to(tl.float32)
    left = (x * c).to(tl.bfloat16).to(tl.float32)
    right = (rotated * sn).to(tl.bfloat16).to(tl.float32)
    y = tl.where(d < ROT, left + right, x)
    tl.store(Y + i, y, i < N)


def apply_minimax_h3_rope_bf16(
    hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Fuse inference RoPE for validated CUDA BF16 ``[B, S, H, D]`` inputs."""
    _, seq, heads, dim = hidden_states.shape
    out = torch.empty(hidden_states.shape, device=hidden_states.device, dtype=hidden_states.dtype)
    if out.numel() == 0:
        return out
    with torch.cuda.device(hidden_states.device):
        _minimax_h3_rope_kernel[(triton.cdiv(out.numel(), 1024),)](
            hidden_states,
            cos,
            sin,
            out,
            out.numel(),
            seq,
            heads,
            dim,
            cos.shape[-1],
            *hidden_states.stride(),
            *cos.stride(),
            *sin.stride(),
            BLOCK=1024,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
