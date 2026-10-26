# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SWQ: the shared expert's SwiGLU-OAI written directly as the down projection's MXFP8 activation."""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


@triton.jit
def _rcp_approx_ftz(x):
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _swq_kernel(
    x_ptr,
    x_stride,
    q_ptr,
    sf_ptr,
    M,
    INTER: tl.constexpr,
    NB: tl.constexpr,
    NB_PAD: tl.constexpr,
    P2: tl.constexpr,
    swiglu_limit: tl.constexpr,
    swiglu_alpha: tl.constexpr,
    swiglu_beta: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
):
    # One program per row of the 128-row-padded scale layout; rows past M only zero their scale bytes.
    r = tl.program_id(0)
    blk = tl.arange(0, P2)
    blk_ok = blk < NB_PAD
    # The 128x4 swizzle of trtllm::mxfp8_quantize(x, True) (quantization.cuh get_sf_out_offset_128x4).
    sf_off = (
        (blk % 4)
        + (blk // 4) * 512
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + (r // 128) * (128 * NB_PAD)
    )
    if r >= M:
        tl.store(sf_ptr + sf_off, tl.zeros([P2], dtype=tl.uint8), mask=blk_ok)
        return
    cols = tl.arange(0, P2 * 32)
    valid = cols < INTER
    row = x_ptr + r.to(tl.int64) * x_stride
    # silu_and_mul_kernel's arithmetic (modules/swiglu.py), then its bf16 store.
    a = tl.load(row + cols, mask=valid, other=0.0).to(tl.float32)
    b = tl.load(row + cols + INTER, mask=valid, other=0.0).to(tl.float32)
    if HAS_LIMIT:
        a = tl.minimum(a, swiglu_limit)
        b = tl.clamp(b, -swiglu_limit, swiglu_limit)
    v = (a * tl.sigmoid(swiglu_alpha * a) * (b + swiglu_beta)).to(tl.bfloat16).to(tl.float32)
    # cvt_warp_fp16_to_mxfp8: SF = amax * rcp.approx(448), UE8M0 toward +inf, scale = rcp.approx(SF), E4M3 RN.
    v2 = tl.reshape(v, (P2, 32))
    amax = tl.max(tl.abs(v2), axis=1)
    sf = amax * _rcp_approx_ftz(tl.full([P2], 448.0, tl.float32))
    bits = sf.to(tl.int32, bitcast=True)
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    e = tl.where(exp > 0, exp + (mant != 0).to(tl.int32), tl.where(bits > 0x400000, 1, 0))
    e = tl.minimum(e, 254)
    # At code 0 the scale 2^-127 is a float subnormal, so rcp.approx.ftz returns inf (native publishes
    # inf/NaN there, found by a tiny-amax oracle). Use its exact reciprocal 2^127 instead; every other code keeps
    # native's rcp.approx (exact on a normal power of two), so each block native emits finite stays bitwise.
    pinv = tl.where(
        e == 0, 1.7014118346046923e38, _rcp_approx_ftz((e << 23).to(tl.float32, bitcast=True))
    )
    inv = tl.where(amax != 0.0, pinv, 0.0)
    q = tl.reshape((v2 * inv[:, None]).to(tl.float8e4nv), (P2 * 32,))
    tl.store(q_ptr + r.to(tl.int64) * INTER + cols, q, mask=valid)
    tl.store(sf_ptr + sf_off, tl.where(blk < NB, e, 0).to(tl.uint8), mask=blk_ok)


class M3Swq(OpWrapper):
    """``silu_and_mul`` (SwiGLU-OAI) over ``x = [g | u]`` (bf16 [M, 2I]) rounded to bf16, then MXFP8-quantized as
    ``trtllm::mxfp8_quantize(h, True)``: (E4M3 [M, I], UE8M0 scales in the 128x4-swizzled layout), bit for bit, in one
    launch. The TP8 shared expert's down projection (intermediate 384, no GEMV row table) reads it at ``SWQ_ROWS``."""

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(why="shared expert, the smallest row count it serves", spec=dict(m=9, inter=384)),
        Cell(
            why="shared expert, a 32-row decode verify (c8; the widest R1-VR-TP8 step)",
            spec=dict(m=32, inter=384),
        ),
        Cell(
            why="TP2 shared expert (width 1536), the first row past its down GEMV (17)",
            spec=dict(m=17, inter=1536),
        ),
        Cell(
            why="TP2 shared expert (width 1536), the K7 c4 step (32 rows)",
            spec=dict(m=32, inter=1536),
        ),
    )

    note = """
    Bitwise to the two-kernel path (silu_and_mul's fp32 arithmetic and bf16 rounding, then mxfp8_quantize's
    rcp.approx scale rule): q bytes and every scale byte, padding included (checked on GB300 and VR).
    One deliberate exception: a block whose scale code is 0 (amax / 448 <= 2^-127), where native's
    rcp.approx.ftz of the subnormal scale is inf and its e4m3 bytes encode inf/NaN, is quantized with the exact
    reciprocal 2^127 instead, so it stays finite; its scale byte is native's.
    """

    def __call__(
        self,
        x: torch.Tensor,
        limit: Optional[float],
        alpha: Optional[float],
        beta: Optional[float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m, i = x.shape[0], x.shape[1] // 2
        nb = i // 32
        nb_pad = (nb + 3) // 4 * 4
        q = torch.empty(m, i, dtype=torch.float8_e4m3fn, device=x.device)
        sf = torch.empty((m + 127) // 128 * 128 * nb_pad, dtype=torch.uint8, device=x.device)
        _swq_kernel[((m + 127) // 128 * 128,)](
            x,
            x.stride(0),
            q,
            sf,
            m,
            INTER=i,
            NB=nb,
            NB_PAD=nb_pad,
            P2=triton.next_power_of_2(nb_pad),
            swiglu_limit=limit or 0.0,
            swiglu_alpha=alpha if alpha is not None else 1.0,
            swiglu_beta=beta if beta is not None else 0.0,
            HAS_LIMIT=limit is not None and limit > 0.0,
        )
        return q, sf

    def reference(
        self,
        x: torch.Tensor,
        limit: Optional[float],
        alpha: Optional[float],
        beta: Optional[float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The two kernels it replaces (bitwise is the contract)."""
        h = torch.ops.trtllm.silu_and_mul(
            x, swiglu_limit=limit, swiglu_alpha=alpha, swiglu_beta=beta
        )
        q, sf = torch.ops.trtllm.mxfp8_quantize(h, True)
        return q.view(h.shape), sf

    def is_valid(
        self,
        x: torch.Tensor,
        limit: Optional[float],
        alpha: Optional[float],
        beta: Optional[float],
    ) -> None:
        assert x.dtype == torch.bfloat16 and x.stride(-1) == 1, "bf16 [M, 2I] with contiguous rows"
        assert (x.shape[1] // 2) % 32 == 0, "I must be a multiple of the 32-element MXFP8 block"


m3_swq = M3Swq()
