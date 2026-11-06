# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The drafter's QK norm + RoPE as Triton tile helpers, bitwise ``drafter_attn_pdl``'s CUDA kernel.

``qk_norm_rope_pdl`` (``kernels/drafter_attn_pdl/_qk_norm_rope_pdl``, built without fast math) runs one warp per
(token, head) of a 128-wide head, four consecutive elements per lane, and computes, in this order (read from its
SASS on sm_107a, NeoX rotation, full rotary width, no YaRN):

* ``ss`` = ``fma(e3, e3, fma(e2, e2, fma(e1, e1, fma(e0, e0, +0))))`` per lane, then a butterfly over lane bits
  4, 3, 2, 1, 0 (``ss + shfl_xor(ss, m)`` for ``m`` = 16 ... 1);
* ``r = rsqrt.approx(fma(ss, 1 / 128, eps))`` (one FFMA: the division by 128 became a product and contracted);
* ``y = e * (r * w)`` (two products, the weight's first);
* ``c = div.rn(lg2.approx(base) * -2, rotary_dim)``; for element ``d``: ``freq = ex2.approx(float(d % 64) * c)``,
  ``theta = freq * float(position)``, ``(sin, cos) = (sin.approx(theta), cos.approx(theta))``;
* the partner ``p = -y[d + 64]`` for ``d < 64`` else ``y[d - 64]`` (the shuffle over 16 lanes, negated on the low
  half), and ``out = fma(cos, y, sin * p) * attention_factor``, rounded to BF16 (``cvt.rn``).

None of it flushes subnormals. Each step is inline PTX here (``*.rn``, ``*.approx`` without ``.ftz``), so the
compiler can neither contract nor reassociate across them; ptxas expands each ``.approx`` instruction as it does
for the CUDA build. The butterfly's tree is formed with ``tl.sum`` over size-2 axes, each a single commutative add.

The values are held in the kernel's lane shape: four ``[R, 32]`` tensors, ``x_i[:, l]`` = element ``4 l + i``, and
the four partners loaded directly from their columns. They are joined into the ``[R, 128]`` tile only at the end.

``qk_fold_rows`` / ``qk_fold_packed_q`` return a BF16 tile for the window attention's split kernel;
``qk_norm_rope_rows`` applies ``qk_fold_rows`` to a copy of a qkv tensor, with ``qk_norm_rope_pdl``'s contract,
for the unit test.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

#: The geometry the helpers are written for: a 128-wide head, NeoX halves of 64, 32 lanes of 4 elements.
HEAD_DIM = 128


@triton.jit
def _mul_rn(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _fma_rn(a, b, c):
    return tl.inline_asm_elementwise(
        "fma.rn.f32 $0, $1, $2, $3;", "=r,r,r,r", [a, b, c], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _div_rn(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _rsqrt_approx(x):
    return tl.inline_asm_elementwise(
        "rsqrt.approx.f32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _lg2_approx(x):
    return tl.inline_asm_elementwise(
        "lg2.approx.f32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _ex2_approx(x):
    return tl.inline_asm_elementwise(
        "ex2.approx.f32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _sin_approx(x):
    return tl.inline_asm_elementwise(
        "sin.approx.f32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _cos_approx(x):
    return tl.inline_asm_elementwise(
        "cos.approx.f32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _lane_load(ptr, row_off, row_ok, col):
    """``[R, 32]`` FP32: element ``col[l]`` of each row's head, one value per lane position ``l``."""
    return tl.load(ptr + row_off[:, None] + col[None, :], mask=row_ok[:, None], other=0.0).to(
        tl.float32
    )


@triton.jit
def _lane_norm(ptr, row_off, row_ok, w_ptr, eps, R: tl.constexpr):
    """The QK norm of ``R`` heads in the CUDA kernel's lane shape.

    Returns ``(y0, y1, y2, y3, p0, p1, p2, p3)``, each ``[R, 32]`` FP32: ``y_i[:, l]`` = normalized element
    ``4 l + i``, ``p_i[:, l]`` = its RoPE partner (the normalized element ``(4 l + i + 64) % 128``, negated on lanes
    0-15). The partner is loaded and normalized here instead of shuffled; it has the same ``r`` and weight
    element, so it is the value the shuffle would deliver.
    """
    lane = tl.arange(0, 32)
    col = 4 * lane
    pcol = 4 * ((lane + 16) % 32)
    x0 = _lane_load(ptr, row_off, row_ok, col)
    x1 = _lane_load(ptr, row_off, row_ok, col + 1)
    x2 = _lane_load(ptr, row_off, row_ok, col + 2)
    x3 = _lane_load(ptr, row_off, row_ok, col + 3)
    s = _fma_rn(x0, x0, tl.zeros_like(x0))
    s = _fma_rn(x1, x1, s)
    s = _fma_rn(x2, x2, s)
    s = _fma_rn(x3, x3, s)
    s = tl.sum(tl.reshape(s, [R, 2, 16]), axis=1)
    s = tl.sum(tl.reshape(s, [R, 2, 8]), axis=1)
    s = tl.sum(tl.reshape(s, [R, 2, 4]), axis=1)
    s = tl.sum(tl.reshape(s, [R, 2, 2]), axis=1)
    s = tl.sum(s, axis=1)
    inv = _rsqrt_approx(_fma_rn(s, 0.0078125, eps))[:, None]
    low = (lane < 16)[None, :]
    y0 = _mul_rn(x0, _mul_rn(inv, tl.load(w_ptr + col).to(tl.float32)[None, :]))
    y1 = _mul_rn(x1, _mul_rn(inv, tl.load(w_ptr + col + 1).to(tl.float32)[None, :]))
    y2 = _mul_rn(x2, _mul_rn(inv, tl.load(w_ptr + col + 2).to(tl.float32)[None, :]))
    y3 = _mul_rn(x3, _mul_rn(inv, tl.load(w_ptr + col + 3).to(tl.float32)[None, :]))
    p0 = _mul_rn(
        _lane_load(ptr, row_off, row_ok, pcol),
        _mul_rn(inv, tl.load(w_ptr + pcol).to(tl.float32)[None, :]),
    )
    p1 = _mul_rn(
        _lane_load(ptr, row_off, row_ok, pcol + 1),
        _mul_rn(inv, tl.load(w_ptr + pcol + 1).to(tl.float32)[None, :]),
    )
    p2 = _mul_rn(
        _lane_load(ptr, row_off, row_ok, pcol + 2),
        _mul_rn(inv, tl.load(w_ptr + pcol + 2).to(tl.float32)[None, :]),
    )
    p3 = _mul_rn(
        _lane_load(ptr, row_off, row_ok, pcol + 3),
        _mul_rn(inv, tl.load(w_ptr + pcol + 3).to(tl.float32)[None, :]),
    )
    p0 = tl.where(low, -p0, p0)
    p1 = tl.where(low, -p1, p1)
    p2 = tl.where(low, -p2, p2)
    p3 = tl.where(low, -p3, p3)
    return y0, y1, y2, y3, p0, p1, p2, p3


@triton.jit
def _lane_freq(base, ELEM: tl.constexpr):
    """``[32]``: the RoPE frequency of element ``4 l + ELEM`` (half-dimension ``(4 l + ELEM) % 64``)."""
    lane = tl.arange(0, 32)
    c = _div_rn(_mul_rn(_lg2_approx(tl.zeros([32], tl.float32) + base), -2.0), 128.0)
    return _ex2_approx(_mul_rn(((4 * lane + ELEM) % 64).to(tl.float32), c))


@triton.jit
def _lane_rotate(y, p, cs, sn, attention_factor):
    return _mul_rn(_fma_rn(cs, y, _mul_rn(sn, p)), attention_factor)


@triton.jit
def _lane_assemble(o0, o1, o2, o3, R: tl.constexpr):
    """``[R, 128]`` from the four lane tensors: column ``4 l + i`` = ``o_i[:, l]``."""
    pairs = tl.join(tl.join(o0, o1), tl.join(o2, o3))  # [R, 32, c, p]: element 2 p + c
    return tl.reshape(tl.permute(pairs, (0, 1, 3, 2)), [R, 128])


@triton.jit
def qk_fold_packed_q(
    ptr,
    row_off,
    row_ok,
    w_ptr,
    pos_ptr,
    pos_row0,
    eps,
    base,
    attention_factor,
    ROWS: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
):
    """``qk_fold_rows`` for the window attention's packed query tile (row ``g * M + i`` = head ``g`` of block row
    ``i``, position ``pos_ptr[pos_row0 + i]``; ``M`` a power of two dividing ``ROWS``): the same per-element
    operations, with the angle, sine and cosine formed once per (block row, element) -- ``[M, 32]`` per lane
    element -- and broadcast over the heads."""
    y0, y1, y2, y3, p0, p1, p2, p3 = _lane_norm(ptr, row_off, row_ok, w_ptr, eps, ROWS)
    pos_f = tl.load(pos_ptr + pos_row0 + tl.arange(0, M)).to(tl.float32)[:, None]
    t0 = _mul_rn(_lane_freq(base, 0)[None, :], pos_f)
    t1 = _mul_rn(_lane_freq(base, 1)[None, :], pos_f)
    t2 = _mul_rn(_lane_freq(base, 2)[None, :], pos_f)
    t3 = _mul_rn(_lane_freq(base, 3)[None, :], pos_f)
    G: tl.constexpr = ROWS // M
    o0 = _lane_rotate(
        tl.reshape(y0, [G, M, 32]),
        tl.reshape(p0, [G, M, 32]),
        _cos_approx(t0)[None, :, :],
        _sin_approx(t0)[None, :, :],
        attention_factor,
    )
    o1 = _lane_rotate(
        tl.reshape(y1, [G, M, 32]),
        tl.reshape(p1, [G, M, 32]),
        _cos_approx(t1)[None, :, :],
        _sin_approx(t1)[None, :, :],
        attention_factor,
    )
    o2 = _lane_rotate(
        tl.reshape(y2, [G, M, 32]),
        tl.reshape(p2, [G, M, 32]),
        _cos_approx(t2)[None, :, :],
        _sin_approx(t2)[None, :, :],
        attention_factor,
    )
    o3 = _lane_rotate(
        tl.reshape(y3, [G, M, 32]),
        tl.reshape(p3, [G, M, 32]),
        _cos_approx(t3)[None, :, :],
        _sin_approx(t3)[None, :, :],
        attention_factor,
    )
    out = _lane_assemble(
        tl.reshape(o0, [ROWS, 32]),
        tl.reshape(o1, [ROWS, 32]),
        tl.reshape(o2, [ROWS, 32]),
        tl.reshape(o3, [ROWS, 32]),
        ROWS,
    )
    return tl.where(row_ok[:, None], out, 0.0).to(tl.bfloat16)


@triton.jit
def qk_fold_rows(
    ptr,
    row_off,
    row_ok,
    w_ptr,
    pos_ptr,
    pos_idx,
    eps,
    base,
    attention_factor,
    R: tl.constexpr,
    D: tl.constexpr,
):
    """``[R, D]`` BF16: the heads at ``ptr + row_off[r]`` normalized with the weight at ``w_ptr`` and rotated to
    position ``pos_ptr[pos_idx[r]]``, as the CUDA kernel writes them. Rows outside ``row_ok`` are zeros."""
    y0, y1, y2, y3, p0, p1, p2, p3 = _lane_norm(ptr, row_off, row_ok, w_ptr, eps, R)
    pos_f = tl.load(pos_ptr + pos_idx, mask=row_ok, other=0).to(tl.float32)[:, None]
    t0 = _mul_rn(_lane_freq(base, 0)[None, :], pos_f)
    t1 = _mul_rn(_lane_freq(base, 1)[None, :], pos_f)
    t2 = _mul_rn(_lane_freq(base, 2)[None, :], pos_f)
    t3 = _mul_rn(_lane_freq(base, 3)[None, :], pos_f)
    out = _lane_assemble(
        _lane_rotate(y0, p0, _cos_approx(t0), _sin_approx(t0), attention_factor),
        _lane_rotate(y1, p1, _cos_approx(t1), _sin_approx(t1), attention_factor),
        _lane_rotate(y2, p2, _cos_approx(t2), _sin_approx(t2), attention_factor),
        _lane_rotate(y3, p3, _cos_approx(t3), _sin_approx(t3), attention_factor),
        R,
    )
    return tl.where(row_ok[:, None], out, 0.0).to(tl.bfloat16)


@triton.jit
def _qk_norm_rope_rows_kernel(
    qkv_ptr,
    out_ptr,
    q_w_ptr,
    k_w_ptr,
    pos_ptr,
    num_tokens,
    stride_row,
    eps,
    base,
    attention_factor,
    NQ: tl.constexpr,
    R: tl.constexpr,
    D: tl.constexpr,
):
    head = tl.program_id(0)
    rows = tl.arange(0, R)
    row_ok = rows < num_tokens
    dims = tl.arange(0, D)
    row_off = rows.to(tl.int64) * stride_row + head * D
    if head < NQ:
        out = qk_fold_rows(
            qkv_ptr, row_off, row_ok, q_w_ptr, pos_ptr, rows, eps, base, attention_factor, R, D
        )
    else:
        out = qk_fold_rows(
            qkv_ptr, row_off, row_ok, k_w_ptr, pos_ptr, rows, eps, base, attention_factor, R, D
        )
    tl.store(out_ptr + row_off[:, None] + dims[None, :], out, mask=row_ok[:, None])


def qk_norm_rope_rows(
    qkv: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    base: float,
    attention_factor: float = 1.0,
) -> torch.Tensor:
    """``qk_norm_rope_pdl``'s values on a copy of ``qkv`` ``[T, (Hq + 2 Hk) * 128]`` (NeoX, full rotary, no YaRN).

    Returns the copy: the query and key heads normalized and rotated, the value heads as given.
    """
    tokens = qkv.shape[0]
    out = qkv.clone()
    _qk_norm_rope_rows_kernel[(num_heads_q + num_heads_k,)](
        qkv,
        out,
        q_weight,
        k_weight,
        positions,
        tokens,
        qkv.stride(0),
        float(eps),
        float(base),
        float(attention_factor),
        NQ=num_heads_q,
        R=max(16, triton.next_power_of_2(tokens)),
        D=HEAD_DIM,
        num_warps=4,
    )
    return out
