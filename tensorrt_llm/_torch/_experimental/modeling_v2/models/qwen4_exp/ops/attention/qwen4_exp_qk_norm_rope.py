# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gemma QK norm + partial NeoX rope for both of an attention layer's planes, one launch.

The main plane (q and k heads of the fused input projection) is rewritten in
place; the QSA index plane's query heads are read from the same projection row
and written, normalized and rotated, into a separate buffer. The rotation reads
the target's own fp32 `[positions, rotary_dim]` cos|sin table.

Two kernels, one result: a Triton kernel, and from `_CUTE_MIN_TOKENS` tokens (a mixed
step's prefill chunk) a persistent CuTe DSL kernel that is bitwise the Triton kernel's
eight-head launch and streams the projection at about twice its rate.
"""

from __future__ import annotations

import struct
from typing import Dict, Tuple

import cutlass
import cutlass.cute as cute
import torch
import triton
import triton.language as tl
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.kernels.blackwell.utils import make_ptr


@triton.jit
def _norm_rope_tile(
    src,
    dst,
    heads,
    head_mask,
    weight,
    positions,
    num_tokens,
    token,
    cos_sin,
    stride_cs,
    eps,
    D: tl.constexpr,
    ROT: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    WEIGHT_SPLIT: tl.constexpr,
    weight_k,
    dst_head_stride,
):
    """One `[len(heads), D]` tile: rms over each head, `* (1 + w)`, rotate the
    first ROT dims in NeoX halves. fp32 throughout, one rounding at the store."""
    HALF: tl.constexpr = ROT // 2
    d = tl.arange(0, D)
    rot = d < ROT
    partner = tl.where(d < HALF, d + HALF, tl.where(rot, d - HALF, d))
    hm = head_mask[:, None]
    x = tl.load(src + heads[:, None] * D + d[None, :], mask=hm, other=0.0).to(tl.float32)
    xp = tl.load(src + heads[:, None] * D + partner[None, :], mask=hm & rot[None, :], other=0.0).to(
        tl.float32
    )
    w = tl.load(weight + d).to(tl.float32)
    wp = tl.load(weight + partner).to(tl.float32)
    if WEIGHT_SPLIT > 0:
        # Heads at or past WEIGHT_SPLIT take the second weight (the k heads).
        is_k = (heads >= WEIGHT_SPLIT)[:, None]
        wk = tl.load(weight_k + d).to(tl.float32)
        wkp = tl.load(weight_k + partner).to(tl.float32)
        w2 = tl.where(is_k, wk[None, :], w[None, :]) + 1.0
        wp2 = tl.where(is_k, wkp[None, :], wp[None, :]) + 1.0
    else:
        w2 = w[None, :] + 1.0
        wp2 = wp[None, :] + 1.0
    rstd = tl.rsqrt(tl.sum(x * x, axis=1) / D + eps)
    y = x * rstd[:, None] * w2
    yp = xp * rstd[:, None] * wp2
    # Interleaved mRoPE: pair j reads coordinate 1 when j % 3 == 1 and
    # j < 3 * S1, coordinate 2 when j % 3 == 2 and j < 3 * S2, else 0.
    j = d % HALF
    p0 = tl.load(positions + token)
    p1 = tl.load(positions + num_tokens + token)
    p2 = tl.load(positions + 2 * num_tokens + token)
    pos = tl.where((j % 3 == 1) & (j < 3 * S1), p1, tl.where((j % 3 == 2) & (j < 3 * S2), p2, p0))
    row = cos_sin + pos.to(tl.int64) * stride_cs
    c = tl.load(row + j, mask=rot, other=1.0)
    s = tl.load(row + HALF + j, mask=rot, other=0.0)
    s = tl.where(d < HALF, -s, s)
    out = y * c[None, :] + yp * s[None, :]
    tl.store(
        dst + heads[:, None] * dst_head_stride + d[None, :],
        out.to(dst.dtype.element_ty),
        mask=hm,
    )


@triton.jit(do_not_specialize=["num_tokens"])
def _qwen4_exp_qk_norm_rope_kernel(
    proj,
    stride_proj,
    index_q,
    stride_iq_token,
    stride_iq_head,
    positions,
    num_tokens,
    cos_sin,
    stride_cs,
    q_weight,
    k_weight,
    index_q_weight,
    eps,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    IH: tl.constexpr,
    ID: tl.constexpr,
    IQ_COL: tl.constexpr,
    ROT: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    IS1: tl.constexpr,
    IS2: tl.constexpr,
    HPB: tl.constexpr,
    MAIN_PROGRAMS: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    program = tl.program_id(1)
    row = proj + token * stride_proj
    if program < MAIN_PROGRAMS:
        main_heads = program * HPB + tl.arange(0, HPB)
        _norm_rope_tile(
            row, row, main_heads, main_heads < HQ + HK, q_weight, positions, num_tokens, token,
            cos_sin, stride_cs, eps, D, ROT, S1, S2, HQ, k_weight, D,
        )  # fmt: skip
    else:
        index_heads = tl.arange(0, IH)
        _norm_rope_tile(
            row + IQ_COL, index_q + token * stride_iq_token, index_heads, index_heads < IH,
            index_q_weight, positions, num_tokens, token, cos_sin, stride_cs, eps,
            ID, ROT, IS1, IS2, 0, index_q_weight, stride_iq_head,
        )  # fmt: skip


def _launch(num_tokens: int) -> Tuple[int, int]:
    """Main-plane heads per program and warps per program, by token count.

    Measured on GB300 (CUDA-graph replay, 12 rotating layers, us per call;
    HPB/warps):

        tokens   1/2    2/2    2/4    4/4    8/8    32/8
        1        1.76   1.84   1.87   1.99   1.92   2.82
        8        1.83   1.97   1.97   2.02   2.05   2.91
        32       2.11   2.14   2.18   2.14   2.16   2.99
        64       2.48   2.39   2.47   2.44   2.48   3.03
        128      3.24   3.19   3.46   3.12   2.96   3.12
        2048     29.7   25.4   30.8   24.0   22.6   25.8
        8192     115    98.5   121    95.7   89.5   122
    """
    if num_tokens <= 16:
        return 1, 2
    if num_tokens <= 64:
        return 2, 2
    return 8, 8


#: This checkpoint's geometry: 24 q + 2 k heads of 256 at the head of the
#: 14080-wide fused projection row, 4 index q heads of 128 at column 13312, a
#: 64-dim rotary prefix, `mrope_section` [11, 11, 10] on the main plane and
#: coordinate 0 alone on the index plane.
CKPT = dict(
    num_heads_q=24,
    num_heads_k=2,
    head_dim=256,
    index_heads=4,
    index_head_dim=128,
    index_q_col=13312,
    proj_width=14080,
    index_scratch_heads=6,
    rotary_dim=64,
    mrope_section1=11,
    mrope_section2=10,
    index_mrope_section1=0,
    index_mrope_section2=0,
)


# ---- CuTe DSL path: prefill-size token counts -------------------------------------------

#: Token count from which the op runs the CuTe kernel. GB300, graph replay, us per call (Triton
#: -> CuTe): 512 8.4 -> 10.3, 768 12.3 -> 8.2, 1024 14.5 -> 8.3, 4096 45.2 -> 18.5, 16384
#: 182 -> 82.7. The Triton kernel runs its eight-head launch from 65 tokens on, and the CuTe
#: kernel is bitwise that launch.
_CUTE_MIN_TOKENS = 1024
#: A token's stage in a warp's ring: the lanes' four chunks (32 x 16 bytes each), the cos and
#: sin of the 32 rotary pairs, and three positions.
_STAGE = 4 * 32 * 16 + 2 * 32 * 4 + 16
_STAGE_CS = 4 * 32 * 16
_STAGE_POS = _STAGE_CS + 2 * 32 * 4
#: Stages in a warp's ring (the next token's loads in flight while one is formed).
_RING = 2
_CTAS_PER_SM = 3


def _iv(x, loc, ip):
    return x.ir_value(loc=loc, ip=ip)


def _asm(ret, args, text, constraints, loc, ip, side_effects=False):
    return llvm.inline_asm(
        ret,
        [_iv(a, loc, ip) for a in args],
        text,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _asm4(args, text, constraints, loc, ip, side_effects=False):
    ty = T.i32()
    res = _asm(
        llvm.StructType.get_literal([ty] * 4), args, text, constraints, loc, ip, side_effects
    )
    return [cutlass.Int32(llvm.extractvalue(ty, res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _ldg_v4(addr, *, loc=None, ip=None):
    return _asm4(
        [cutlass.Int64(addr)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def _ldg_b32(addr, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(T.i32(), [cutlass.Int64(addr)], "ld.global.b32 $0, [$1];", "=r,l", loc, ip, True)
    )


@dsl_user_op
def _stg_v4(addr, off: int, a, b, c, d, *, loc=None, ip=None):
    """16 bytes at `addr + off`."""
    _asm(
        T.i32(),
        [cutlass.Int64(addr)] + [cutlass.Int32(v) for v in (a, b, c, d)],
        "{ st.global.v4.b32 [$1+%d], {$2, $3, $4, $5}; mov.u32 $0, 0; }" % off,
        "=r,l,r,r,r,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def _lds_v4(addr, *, loc=None, ip=None):
    return _asm4(
        [cutlass.Int32(addr)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def _lds_b32(addr, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(T.i32(), [cutlass.Int32(addr)], "ld.shared.b32 $0, [$1];", "=r,r", loc, ip, True)
    )


@dsl_user_op
def _cp16(dst, src, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [cutlass.Int32(dst), cutlass.Int64(src)],
        "{ cp.async.cg.shared.global [$1], [$2], 16; mov.u32 $0, 0; }",
        "=r,r,l",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def _cp4(dst, src, pred, *, loc=None, ip=None):
    """4 bytes where `pred` is not 0."""
    _asm(
        T.i32(),
        [cutlass.Int32(dst), cutlass.Int64(src), cutlass.Int32(pred)],
        "{ .reg .pred p; setp.ne.b32 p, $3, 0; @p cp.async.ca.shared.global [$1], [$2], 4; "
        "mov.u32 $0, 0; }",
        "=r,r,l,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def _cp_commit(*, loc=None, ip=None):
    _asm(T.i32(), [], "{ cp.async.commit_group; mov.u32 $0, 0; }", "=r", loc, ip, True)


@dsl_user_op
def _cp_wait(n: int, *, loc=None, ip=None):
    _asm(T.i32(), [], "{ cp.async.wait_group %d; mov.u32 $0, 0; }" % n, "=r", loc, ip, True)


@dsl_user_op
def _bf16_lo(w, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [cutlass.Int32(w)],
            "{ .reg .b32 t; shl.b32 t, $1, 16; mov.b32 $0, t; }",
            "=f,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _bf16_hi(w, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [cutlass.Int32(w)],
            "{ .reg .b32 t; and.b32 t, $1, 0xFFFF0000; mov.b32 $0, t; }",
            "=f,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _unpack2(w, *, loc=None, ip=None):
    """A bf16x2 word -> its two values as an fp32x2 pair, the low element first."""
    return cutlass.Int64(
        _asm(
            T.i64(),
            [cutlass.Int32(w)],
            "{ .reg .b32 a, b; shl.b32 a, $1, 16; and.b32 b, $1, 0xFFFF0000; mov.b64 $0, {a, b}; }",
            "=l,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _pair(lo, hi, *, loc=None, ip=None):
    """Two fp32 bit patterns -> one fp32x2 pair."""
    return cutlass.Int64(
        _asm(
            T.i64(),
            [cutlass.Int32(lo), cutlass.Int32(hi)],
            "mov.b64 $0, {$1, $2};",
            "=l,r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _splat2(x, *, loc=None, ip=None):
    return cutlass.Int64(
        _asm(T.i64(), [cutlass.Float32(x)], "mov.b64 $0, {$1, $1};", "=l,f", loc, ip)
    )


@dsl_user_op
def _mul2(a, b, *, loc=None, ip=None):
    return cutlass.Int64(
        _asm(
            T.i64(),
            [cutlass.Int64(a), cutlass.Int64(b)],
            "mul.rn.f32x2 $0, $1, $2;",
            "=l,l,l",
            loc,
            ip,
        )
    )


@dsl_user_op
def _fma2(a, b, c, *, loc=None, ip=None):
    return cutlass.Int64(
        _asm(
            T.i64(),
            [cutlass.Int64(a), cutlass.Int64(b), cutlass.Int64(c)],
            "fma.rn.f32x2 $0, $1, $2, $3;",
            "=l,l,l,l",
            loc,
            ip,
        )
    )


@dsl_user_op
def _add2(a, b, *, loc=None, ip=None):
    return cutlass.Int64(
        _asm(
            T.i64(),
            [cutlass.Int64(a), cutlass.Int64(b)],
            "add.rn.f32x2 $0, $1, $2;",
            "=l,l,l",
            loc,
            ip,
        )
    )


@dsl_user_op
def _hsum2(a, *, loc=None, ip=None):
    """The low plus the high element of a pair."""
    return cutlass.Float32(
        _asm(
            T.f32(),
            [cutlass.Int64(a)],
            "{ .reg .f32 x, y; mov.b64 {x, y}, $1; add.rn.f32 $0, x, y; }",
            "=f,l",
            loc,
            ip,
        )
    )


@dsl_user_op
def _bf16x2_of(a, *, loc=None, ip=None):
    """A pair rounded (RN) into one bf16x2 word, the low element low."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [cutlass.Int64(a)],
            "{ .reg .f32 x, y; mov.b64 {x, y}, $1; cvt.rn.bf16x2.f32 $0, y, x; }",
            "=r,l",
            loc,
            ip,
        )
    )


@dsl_user_op
def _neg_bits(w, *, loc=None, ip=None):
    """An fp32 bit pattern with its sign flipped (the Triton kernel's `xor`)."""
    return cutlass.Int32(
        _asm(T.i32(), [cutlass.Int32(w)], "xor.b32 $0, $1, 0x80000000;", "=r,r", loc, ip)
    )


@dsl_user_op
def _fma(a, b, c, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [cutlass.Float32(a), cutlass.Float32(b), cutlass.Float32(c)],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _mul(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [cutlass.Float32(a), cutlass.Float32(b)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _add(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [cutlass.Float32(a), cutlass.Float32(b)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _rstd(ss, eps, width_hex: str, *, loc=None, ip=None):
    """`rsqrt(ss / width + eps)` as the Triton kernel's `div.full`, `add`, `rsqrt.approx.ftz`:
    the `add` has no rounding mode there, so ptxas fuses it with the division (a multiply by
    the reciprocal) into one FFMA -- here the same."""
    return cutlass.Float32(
        _asm(
            T.f32(),
            [cutlass.Float32(ss), cutlass.Float32(eps)],
            "{ .reg .f32 t; div.full.f32 t, $1, " + width_hex + "; add.f32 t, t, $2; "
            "rsqrt.approx.ftz.f32 $0, t; }",
            "=f,f,f",
            loc,
            ip,
        )
    )


def _hex(x: float) -> str:
    return "0f%08X" % struct.unpack("<I", struct.pack("<f", float(x)))[0]


class _NormRopeRing:
    """The Triton kernel's eight-head launch at prefill-size token counts, persistent: CTA b
    forms tokens b, b + G, ... (G CTAs).

    Warps: `hq / 4` q warps of 4 heads, one k warp (the 2 k heads; lanes 16-31 repeat 0-15),
    one index warp (the 4 index heads). Lane (h = L // 8, c = L % 8) owns chunks c + 8m of its
    head (a chunk is 8 dims, 16 bytes; m < 4, m < 2 on the index plane): chunk c is its rotary
    chunk, and that chunk's NeoX partner is lane L ^ 4's. A warp's next token -- its chunks,
    its cos|sin entries and the positions of the token after -- arrives by cp.async in a ring
    of the warp's own shared memory while it forms the current one; its weights stay in
    registers.

    Bitwise the Triton kernel: the main plane's sum of squares is its tree (per chunk
    `(fma(x0, x0, x2 * x2) + fma(x4, x4, x6 * x6)) + (the odd dims)`, then its butterfly over
    a head's 32 chunks -- chunk bits 4 and 3 inside the lane, 2, 1, 0 across lanes), the index
    plane's its two-warp tree over a head's 64 dim pairs (pair bits 4, 3, 2 across lanes, 1, 0
    and 5 inside), the products are its fp32x2 products in its order, and dims past the rotary
    prefix take its `y * 1 + (w2 * (rstd * 0)) * 0` as `y + w2 * 0` (equal for every finite
    rstd, and both NaN otherwise).
    """

    def __init__(self, hq: int, iq_col: int, s1: int, s2: int, is1: int, is2: int):
        self.hq, self.iq_col = hq, iq_col
        self.sections = ((s1, s2), (s1, s2), (is1, is2))
        self.q_warps = hq // 4
        self.warps = self.q_warps + 2
        self.depth = _RING - 1

    @cute.jit
    def __call__(
        self,
        proj: cute.Tensor,
        iq_ptr: cute.Pointer,
        pos_ptr: cute.Pointer,
        cs_ptr: cute.Pointer,
        qw_ptr: cute.Pointer,
        kw_ptr: cute.Pointer,
        iw_ptr: cute.Pointer,
        tokens: cutlass.Int32,
        s_proj: cutlass.Int32,
        s_iq: cutlass.Int32,
        s_cs: cutlass.Int32,
        eps: cutlass.Float32,
        ctas: cutlass.Int32,
        stream,
    ):
        # `proj` arrives as a tensor: the TVM-FFI entry takes its stream from a GPU tensor.
        self.kernel(
            proj.iterator.toint(),
            s_proj,
            iq_ptr.toint(),
            s_iq,
            pos_ptr.toint(),
            tokens,
            cs_ptr.toint(),
            s_cs,
            qw_ptr.toint(),
            kw_ptr.toint(),
            iw_ptr.toint(),
            eps,
        ).launch(
            grid=[ctas, 1, 1],
            block=[32 * self.warps, 1, 1],
            smem=self.warps * _RING * _STAGE,
            min_blocks_per_mp=_CTAS_PER_SM,
            stream=stream,
        )

    def _lane(self, kind, warp, lane):
        """(head, rotary chunk, chunks, first projection column) of a lane; kind 0 q, 1 k, 2 index."""
        c = lane % 8
        if kind == 2:
            h = lane // 8
            return h, c, 2, self.iq_col + h * 128 + c * 8
        if kind == 1:
            h = self.hq + (lane // 8) % 2
        else:
            h = warp * 4 + lane // 8
        return h, c, 4, h * 256 + c * 8

    def _issue(
        self, kind, lane, stage, pstage, t, tn, tokens, src, nch, pos_base, cs_base, s_cs, p
    ):
        """Token t's chunks (from `src`) and cos|sin (its positions `p`) into `stage`; token tn's
        positions into `pstage`. One cp.async group."""
        for m in range(nch):
            _cp16(stage + (m * 32 + lane) * 16, src + m * 128)
        s1, s2 = self.sections[kind]
        r = p[0]
        if s1 or s2:
            m3 = lane % 3
            r = cutlass.select_(
                (m3 == 1) & (lane < 3 * s1),
                p[1],
                cutlass.select_((m3 == 2) & (lane < 3 * s2), p[2], p[0]),
            )
        crow = cs_base + cutlass.Int64(r) * cutlass.Int64(s_cs) * 4
        one = cutlass.Int32(1)
        _cp4(stage + _STAGE_CS + lane * 4, crow + cutlass.Int64(lane) * 4, one)
        _cp4(stage + _STAGE_CS + 128 + lane * 4, crow + cutlass.Int64(32 + lane) * 4, one)
        k = cutlass.min(lane, cutlass.Int32(2))
        _cp4(
            pstage + _STAGE_POS + k * 4,
            pos_base + (cutlass.Int64(k) * cutlass.Int64(tokens) + cutlass.Int64(tn)) * 4,
            cutlass.select_(lane < 3, one, cutlass.Int32(0)),
        )
        _cp_commit()

    def _form(self, kind, lane, c, nch, stage, ww, dst, eps):
        """A token from its stage: the norm and rotation of the lane's chunks, stored at `dst`."""
        xw = [_lds_v4(stage + (m * 32 + lane) * 16) for m in range(nch)]
        if kind == 2:
            t = []
            for m in range(nch):
                x = []
                for w in xw[m]:
                    x += [_bf16_lo(w), _bf16_hi(w)]
                t.append(
                    [_fma(x[2 * k], x[2 * k], _mul(x[2 * k + 1], x[2 * k + 1])) for k in range(4)]
                )
            for off in (4, 2, 1):
                t = [[_add(v, cute.arch.shuffle_sync_bfly(v, offset=off)) for v in tm] for tm in t]
            u = [_add(_add(tm[0], tm[2]), _add(tm[1], tm[3])) for tm in t]
            v = _add(u[0], u[1])
            width_hex = _hex(128)
        else:
            sc = []
            for m in range(nch):
                P = [_unpack2(w) for w in xw[m]]
                sc.append(
                    _hsum2(
                        _add2(
                            _fma2(P[0], P[0], _mul2(P[1], P[1])),
                            _fma2(P[2], P[2], _mul2(P[3], P[3])),
                        )
                    )
                )
            v = _add(_add(sc[0], sc[2]), _add(sc[1], sc[3]))
            for off in (4, 2, 1):
                v = _add(v, cute.arch.shuffle_sync_bfly(v, offset=off))
            width_hex = _hex(256)
        rstd = _splat2(_rstd(v, eps, width_hex))
        # cos|sin of the rotary chunk's 8 pairs; sin negated on the first half's dims.
        j0 = (c % 4) * 8
        cw = _lds_v4(stage + _STAGE_CS + j0 * 4) + _lds_v4(stage + _STAGE_CS + j0 * 4 + 16)
        sw = _lds_v4(stage + _STAGE_CS + 128 + j0 * 4) + _lds_v4(
            stage + _STAGE_CS + 128 + j0 * 4 + 16
        )
        sw = [cutlass.select_(c < 4, _neg_bits(w), w) for w in sw]
        cos = [_pair(cw[2 * i], cw[2 * i + 1]) for i in range(4)]
        sin = [_pair(sw[2 * i], sw[2 * i + 1]) for i in range(4)]
        partner = [cute.arch.shuffle_sync_bfly(w, offset=4) for w in xw[0]]
        ones = _splat2(cutlass.Float32(1.0))
        w2 = [_add2(_unpack2(w), ones) for w in ww[0]]
        wp2 = [_add2(_unpack2(w), ones) for w in ww[4]]
        words = []
        for i in range(4):
            y = _mul2(w2[i], _mul2(rstd, _unpack2(xw[0][i])))
            yp = _mul2(wp2[i], _mul2(rstd, _unpack2(partner[i])))
            words.append(_bf16x2_of(_fma2(y, cos[i], _mul2(yp, sin[i]))))
        _stg_v4(dst, 0, *words)
        zero = _splat2(cutlass.Float32(0.0))
        for m in range(1, nch):
            w2 = [_add2(_unpack2(w), ones) for w in ww[m]]
            words = []
            for i in range(4):
                y = _mul2(w2[i], _mul2(rstd, _unpack2(xw[m][i])))
                words.append(_bf16x2_of(_add2(y, _mul2(w2[i], zero))))
            _stg_v4(dst, m * 128, *words)

    def _prologue(self, kind, warp, lane, bid, ctas, tokens, sm, proj_base, s_proj, pos_base,
                  cs_base, s_cs):  # fmt: skip
        """The groups of the CTA's first `depth` tokens."""
        _, _, nch, col = self._lane(kind, warp, lane)
        last = tokens - 1
        for k in range(self.depth):
            t = cutlass.min(bid + k * ctas, last)
            tn = cutlass.min(bid + (k + self.depth) * ctas, last)
            p = [
                _ldg_b32(pos_base + (cutlass.Int64(tokens) * j + cutlass.Int64(t)) * 4)
                for j in range(3)
            ]
            src = proj_base + (cutlass.Int64(t) * cutlass.Int64(s_proj) + cutlass.Int64(col)) * 2
            self._issue(
                kind, lane, sm + k * _STAGE, sm + ((k + self.depth) % _RING) * _STAGE, t, tn,
                tokens, src, nch, pos_base, cs_base, s_cs, p,
            )  # fmt: skip

    def _step(self, kind, warp, lane, i, bid, ctas, tokens, sm, ww, proj_base, s_proj, iq_base,
              s_iq, pos_base, cs_base, s_cs, eps):  # fmt: skip
        """The CTA's i-th token: wait for its group, send the group of token i + depth, form it."""
        h, c, nch, col = self._lane(kind, warp, lane)
        last = tokens - 1
        _cp_wait(self.depth - 1)
        cute.arch.sync_warp()
        stage_n = sm + ((i + self.depth) % _RING) * _STAGE
        p = [_lds_b32(stage_n + _STAGE_POS + 4 * j) for j in range(3)]
        t = bid + i * ctas
        tn = cutlass.min(t + self.depth * ctas, last)
        tnn = cutlass.min(t + 2 * self.depth * ctas, last)
        src = proj_base + (cutlass.Int64(tn) * cutlass.Int64(s_proj) + cutlass.Int64(col)) * 2
        self._issue(
            kind, lane, stage_n, sm + ((i + 2 * self.depth) % _RING) * _STAGE, tn, tnn, tokens,
            src, nch, pos_base, cs_base, s_cs, p,
        )  # fmt: skip
        if kind == 2:
            dst = (
                iq_base
                + (cutlass.Int64(t) * cutlass.Int64(s_iq) + cutlass.Int64(h * 128 + c * 8)) * 2
            )
        else:
            dst = proj_base + (cutlass.Int64(t) * cutlass.Int64(s_proj) + cutlass.Int64(col)) * 2
        self._form(kind, lane, c, nch, sm + (i % _RING) * _STAGE, ww, dst, eps)

    @cute.kernel
    def kernel(
        self,
        proj_base: cutlass.Int64,
        s_proj: cutlass.Int32,
        iq_base: cutlass.Int64,
        s_iq: cutlass.Int32,
        pos_base: cutlass.Int64,
        tokens: cutlass.Int32,
        cs_base: cutlass.Int64,
        s_cs: cutlass.Int32,
        qw: cutlass.Int64,
        kw: cutlass.Int64,
        iw: cutlass.Int64,
        eps: cutlass.Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        ctas, _, _ = cute.arch.grid_dim()
        warp = tidx // 32
        lane = tidx % 32
        buf = cutlass.utils.SmemAllocator().allocate_tensor(
            cutlass.Float32,
            cute.make_layout((self.warps * _RING * _STAGE // 4,), stride=(1,)),
            byte_alignment=16,
        )
        sm = cutlass.Int32(buf.iterator.toint()) + warp * (_RING * _STAGE)
        n = (tokens - bid + ctas - 1) // ctas
        # The lane's weight chunks c + 8m (m < 4) and c ^ 4 as bf16 words, by the warp's plane;
        # the index weight's 16 chunks clamp the chunks its warp does not read.
        c = lane % 8
        is_q = warp < self.q_warps
        is_k = warp == self.q_warps
        w_base = cutlass.select_(is_q, qw, cutlass.select_(is_k, kw, iw))
        lim = cutlass.select_(is_q | is_k, cutlass.Int32(31), cutlass.Int32(15))
        ww = [
            _ldg_v4(w_base + cutlass.Int64(cutlass.min(ch, lim) * 8) * 2)
            for ch in [c + 8 * m for m in range(4)] + [c ^ 4]
        ]
        common = (lane, bid, ctas, tokens, sm, proj_base, s_proj, pos_base, cs_base, s_cs)
        if warp < self.q_warps:
            self._prologue(0, warp, *common)
        elif warp == self.q_warps:
            self._prologue(1, warp, *common)
        else:
            self._prologue(2, warp, *common)
        rest = (bid, ctas, tokens, sm, ww, proj_base, s_proj, iq_base, s_iq, pos_base, cs_base,
                s_cs, eps)  # fmt: skip
        for i in cutlass.range(n, unroll=1):
            if warp < self.q_warps:
                self._step(0, warp, lane, i, *rest)
            elif warp == self.q_warps:
                self._step(1, warp, lane, i, *rest)
            else:
                self._step(2, warp, lane, i, *rest)
        _cp_wait(0)


_RING_KERNELS: Dict[tuple, object] = {}
_SM_COUNT: Dict[int, int] = {}


def _cute_ok(
    proj: torch.Tensor,
    index_q: torch.Tensor,
    cos_sin: torch.Tensor,
    weights: Tuple[torch.Tensor, ...],
    num_heads_q: int,
    num_heads_k: int,
    head_dim: int,
    index_heads: int,
    index_head_dim: int,
    index_q_col: int,
    rotary_dim: int,
) -> bool:
    """The geometry and layouts the CuTe kernel covers; anything else runs the Triton kernel."""
    bf = torch.bfloat16
    return (
        num_heads_q % 4 == 0
        and num_heads_k == 2
        and head_dim == 256
        and index_heads == 4
        and index_head_dim == 128
        and rotary_dim == 64
        and index_q_col % 8 == 0
        and proj.dtype == index_q.dtype == bf
        and all(w.dtype == bf and w.is_contiguous() and w.data_ptr() % 16 == 0 for w in weights)
        and cos_sin.dtype == torch.float32
        and cos_sin.stride(1) == 1
        and proj.data_ptr() % 16 == 0
        and proj.stride(0) % 8 == 0
        and index_q.data_ptr() % 16 == 0
        and index_q.stride(0) % 8 == 0
        and index_q.stride(1) == index_head_dim
    )


def _launch_cute(
    proj: torch.Tensor,
    index_q: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    index_q_weight: torch.Tensor,
    num_heads_q: int,
    index_q_col: int,
    eps: float,
    sections: Tuple[int, int, int, int],
) -> None:
    key = (num_heads_q, index_q_col) + tuple(sections)
    fn = _RING_KERNELS.get(key)
    if fn is None:
        gm = cute.AddressSpace.gmem
        bf = cutlass.BFloat16
        fn = cute.compile(
            _NormRopeRing(num_heads_q, index_q_col, *sections),
            from_dlpack(proj, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            make_ptr(bf, index_q.data_ptr(), gm, assumed_align=16),
            make_ptr(cutlass.Int32, position_ids.data_ptr(), gm, assumed_align=4),
            make_ptr(cutlass.Float32, cos_sin.data_ptr(), gm, assumed_align=4),
            make_ptr(bf, q_weight.data_ptr(), gm, assumed_align=16),
            make_ptr(bf, k_weight.data_ptr(), gm, assumed_align=16),
            make_ptr(bf, index_q_weight.data_ptr(), gm, assumed_align=16),
            *(cutlass.Int32(1),) * 4,
            cutlass.Float32(1e-6),
            cutlass.Int32(1),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--opt-level 2 --enable-tvm-ffi",
        )
        _RING_KERNELS[key] = fn
    dev = proj.device.index
    if dev not in _SM_COUNT:
        _SM_COUNT[dev] = torch.cuda.get_device_properties(proj.device).multi_processor_count
    tokens = proj.shape[0]
    fn(
        proj,
        index_q.data_ptr(),
        position_ids.data_ptr(),
        cos_sin.data_ptr(),
        q_weight.data_ptr(),
        k_weight.data_ptr(),
        index_q_weight.data_ptr(),
        tokens,
        proj.stride(0),
        index_q.stride(0),
        cos_sin.stride(0),
        eps,
        min(tokens, _SM_COUNT[dev] * _CTAS_PER_SM),
    )


def _launch_triton(
    proj: torch.Tensor,
    index_q: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    index_q_weight: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    head_dim: int,
    index_heads: int,
    index_head_dim: int,
    index_q_col: int,
    rotary_dim: int,
    eps: float,
    sections: Tuple[int, int, int, int],
) -> None:
    num_tokens = proj.shape[0]
    hpb, warps = _launch(num_tokens)
    main_programs = triton.cdiv(num_heads_q + num_heads_k, hpb)
    _qwen4_exp_qk_norm_rope_kernel[(num_tokens, main_programs + 1)](
        proj,
        proj.stride(0),
        index_q,
        index_q.stride(0),
        index_q.stride(1),
        position_ids,
        num_tokens,
        cos_sin,
        cos_sin.stride(0),
        q_weight,
        k_weight,
        index_q_weight,
        eps,
        HQ=num_heads_q,
        HK=num_heads_k,
        D=head_dim,
        IH=index_heads,
        ID=index_head_dim,
        IQ_COL=index_q_col,
        ROT=rotary_dim,
        S1=sections[0],
        S2=sections[1],
        IS1=sections[2],
        IS2=sections[3],
        HPB=hpb,
        MAIN_PROGRAMS=main_programs,
        num_warps=warps,
    )


class Qwen4ExpQkNormRope(OpWrapper):
    """Both planes of one attention layer's QK norm + rope, in one launch.

    Main plane, in place on `proj`: heads `[0, Hq)` (weight `q_weight`) and
    `[Hq, Hq + Hk)` (weight `k_weight`), each `head_dim` wide at the start of
    every row, become `rope(rmsnorm(x) * (1 + w))`. Index plane: the
    `index_heads` heads of `index_head_dim` starting at column `index_q_col` of
    `proj` are read, normalized with `index_q_weight` and rotated, and written
    to `index_q` (`[T, index_heads, index_head_dim]`, any token/head stride,
    unit dim stride); `proj`'s index columns are not written. Every other
    column of `proj` is left alone. Returns `(proj, index_q)`.

    The rotation is NeoX over the first `rotary_dim` dims of a head: pair `j`
    is `(j, j + rotary_dim / 2)`, rotated by `cos_sin[p, j]` and
    `cos_sin[p, rotary_dim / 2 + j]` (`cos_sin` is the fp32 `[P, rotary_dim]`
    table, cos block then sin block). `position_ids` is the contiguous int32
    `[3, T]`; pair `j` reads coordinate 1 when `j % 3 == 1 and j < 3 * s1`,
    coordinate 2 when `j % 3 == 2 and j < 3 * s2`, coordinate 0 otherwise --
    interleaved mRoPE with the plane's own `(s1, s2)`; `(0, 0)` is the plain
    rope on coordinate 0. fp32 throughout, one rounding at the store.

    Fusion boundary: norm and rotation of both planes. The projection GEMM
    before it, and the KV/index-cache writes and attention after it, are the
    caller's. The index plane's raw key is not touched, so the index-cache
    store may run before or after this call.
    """

    ARCHS = frozenset({Arch.SM_103})

    # Four cells, one per compiled launch the forward reaches: the Triton kernel's 1-, 2-
    # and 8-head launches (by decode batch size) and the CuTe kernel (prefill).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode M=1: one token on the 1-head/2-warp launch, three distinct mRoPE rows",
            spec=dict(tokens=1, **CKPT),
        ),
        Cell(
            why="decode M=17: first on the 2-head launch (13 main programs)",
            spec=dict(tokens=17, **CKPT),
        ),
        Cell(
            why="decode M=128: max concurrency, 8-head launch (last main program half masked), "
            "positions up to the table end (262144 rows)",
            spec=dict(tokens=128, max_position=262144, **CKPT),
        ),
        Cell(why="prefill 16384 tokens: max_num_tokens", spec=dict(tokens=16384, **CKPT)),
    )

    note = """
    Metadata consumed: none (positions are an operand). Below `_CUTE_MIN_TOKENS`
    tokens one Triton kernel, no autotuner: grid `(T, cdiv(Hq + Hk, HPB) + 1)`;
    the last program of a token row runs the whole index plane as one
    `[index_heads, index_head_dim]` tile. HPB and warps by token count from
    `_launch` (its docstring has the table). Replaces, per attention layer, the
    CUDA `fused_qk_norm_rope` main call, the index-scratch copy and the second
    `fused_qk_norm_rope` call: 6.67 -> 1.76 us at M=1, 9.15 -> 2.96 at M=128, 185
    -> 89.5 at 8192 tokens (graph replay).

    From `_CUTE_MIN_TOKENS` tokens, where the geometry is this checkpoint's (`_cute_ok`:
    head dims 256 / 128, 2 k heads, 4 index heads, a 64-dim rotary prefix, 16-byte
    aligned rows, contiguous index heads), the CuTe DSL `_NormRopeRing`: three
    persistent CTAs an SM, each forming every G-th token, a warp's next token
    prefetched by cp.async (its class docstring has the mapping). 16384 tokens: 182 -> 82.7 us in the
    microbenchmark (the Triton kernel was L1-bound; the ring runs at ~65% of DRAM peak,
    bounded by load latency at three CTAs an SM).

    Numerics: fp32 throughout, one rounding at the store; the CuTe kernel is bitwise
    the Triton kernel's eight-head launch (measured at 1024-16384 tokens, signed zeros
    included; not pinned by a test). Against the fp64
    reference over the same table every cell is within one output rounding
    step (measured max 1.48 ulp of the row scale, relative RMS <= 0.011 ulp).
    Against the CUDA pair it replaces (measured, not pinned by a test), <= 1.41
    ulp of the row scale and 0.09 ulp RMS (that op derives cos/sin with `__sincosf`, whose phase error grows
    with position). Three agreeing position rows select the same table row for
    every pair, so a text batch is exactly the plain rope on row 0.

    Preconditions in `is_valid`: the main q/k columns end at or before
    `index_q_col` (the planes do not overlap, so reading the index plane while
    the main plane is rewritten in place is race-free); `index_q` does not
    alias `proj`; positions below the table length (an out-of-range position
    reads past the table).

    Hard limits: token offsets in int64; positions index the table in int64.

    CUDA graphs: the Triton kernel captured at M=4 replays bitwise an eager call on
    inputs rewritten in place.

    Not certified: non-zero index-plane mRoPE sections (the target passes (0, 0)),
    capture of the CuTe kernel, `T == 0`, fp16/fp32 activations, `rotary_dim` > either head width,
    non-power-of-two head widths or head counts (tl.arange), YaRN, non-Gemma
    norm, a k plane on the index side (its scratch slot is not written).
    """

    def __call__(
        self,
        proj: torch.Tensor,
        index_q: torch.Tensor,
        position_ids: torch.Tensor,
        cos_sin: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        index_q_weight: torch.Tensor,
        num_heads_q: int,
        num_heads_k: int,
        head_dim: int,
        index_heads: int,
        index_head_dim: int,
        index_q_col: int,
        rotary_dim: int,
        eps: float,
        mrope_section1: int,
        mrope_section2: int,
        index_mrope_section1: int = 0,
        index_mrope_section2: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_tokens = proj.shape[0]
        if num_tokens == 0:
            return proj, index_q
        sections = (mrope_section1, mrope_section2, index_mrope_section1, index_mrope_section2)
        weights = (q_weight, k_weight, index_q_weight)
        if num_tokens >= _CUTE_MIN_TOKENS and _cute_ok(
            proj, index_q, cos_sin, weights, num_heads_q, num_heads_k, head_dim, index_heads,
            index_head_dim, index_q_col, rotary_dim,
        ):  # fmt: skip
            _launch_cute(
                proj, index_q, position_ids, cos_sin, *weights, num_heads_q, index_q_col, eps,
                sections,
            )  # fmt: skip
        else:
            _launch_triton(
                proj, index_q, position_ids, cos_sin, *weights, num_heads_q, num_heads_k,
                head_dim, index_heads, index_head_dim, index_q_col, rotary_dim, eps, sections,
            )  # fmt: skip
        return proj, index_q

    def reference(
        self,
        proj: torch.Tensor,
        index_q: torch.Tensor,
        position_ids: torch.Tensor,
        cos_sin: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        index_q_weight: torch.Tensor,
        num_heads_q: int,
        num_heads_k: int,
        head_dim: int,
        index_heads: int,
        index_head_dim: int,
        index_q_col: int,
        rotary_dim: int,
        eps: float,
        mrope_section1: int,
        mrope_section2: int,
        index_mrope_section1: int = 0,
        index_mrope_section2: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """`(proj after, index_q's written values)` in fp64 over the same table,
        rounded once; neither input is mutated."""
        num_tokens = proj.shape[0]
        half = rotary_dim // 2
        table = cos_sin.double()
        pos = position_ids.long()
        j = torch.arange(half, device=proj.device)

        def angles(s1: int, s2: int):
            coord = torch.zeros(half, dtype=torch.long, device=proj.device)
            coord[(j % 3 == 1) & (j < 3 * s1)] = 1
            coord[(j % 3 == 2) & (j < 3 * s2)] = 2
            p = pos[coord, :].t()  # [T, half]
            return table[p, j][:, None, :], table[p, half + j][:, None, :]

        def plane(x: torch.Tensor, w: torch.Tensor, cos, sin) -> torch.Tensor:
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w)
            x1, x2 = x[..., :half], x[..., half:rotary_dim]
            rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)
            return torch.cat([rotated, x[..., rotary_dim:]], -1)

        out = proj.double().clone()
        main_heads = num_heads_q + num_heads_k
        main_cols = main_heads * head_dim
        w = torch.cat(
            [
                q_weight.double().expand(num_heads_q, head_dim),
                k_weight.double().expand(num_heads_k, head_dim),
            ]
        )
        main = out[:, :main_cols].reshape(num_tokens, main_heads, head_dim)
        out[:, :main_cols] = plane(main, w, *angles(mrope_section1, mrope_section2)).reshape(
            num_tokens, main_cols
        )
        iq = proj[:, index_q_col : index_q_col + index_heads * index_head_dim].double()
        iq = plane(
            iq.reshape(num_tokens, index_heads, index_head_dim),
            index_q_weight.double(),
            *angles(index_mrope_section1, index_mrope_section2),
        )
        return out.to(proj.dtype), iq.to(index_q.dtype)

    def is_valid(
        self,
        proj: torch.Tensor,
        index_q: torch.Tensor,
        position_ids: torch.Tensor,
        cos_sin: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        index_q_weight: torch.Tensor,
        num_heads_q: int,
        num_heads_k: int,
        head_dim: int,
        index_heads: int,
        index_head_dim: int,
        index_q_col: int,
        rotary_dim: int,
        *unused_args: object,
        **unused_kwargs: object,
    ) -> None:
        num_tokens = proj.shape[0]
        assert proj.stride(-1) == 1, "proj must have a unit column stride"
        assert (num_heads_q + num_heads_k) * head_dim <= index_q_col, (
            "the index q columns overlap the main q/k heads: the in-place main write races "
            "the index read"
        )
        assert index_q_col + index_heads * index_head_dim <= proj.shape[1], (
            "the index q columns run past the projection row"
        )
        assert index_q.shape == (num_tokens, index_heads, index_head_dim), index_q.shape
        assert index_q.stride(-1) == 1, "index_q must have a unit dim stride"
        lo, hi = index_q.data_ptr(), proj.data_ptr()
        assert lo >= hi + proj.numel() * proj.element_size() or hi >= lo + (
            index_q.shape[0] * index_q.stride(0) * index_q.element_size()
        ), "index_q aliases proj"
        assert position_ids.shape == (3, num_tokens) and position_ids.is_contiguous(), (
            f"position_ids must be a contiguous [3, {num_tokens}]; got {tuple(position_ids.shape)}"
        )
        assert position_ids.dtype == torch.int32, position_ids.dtype
        assert cos_sin.dtype == torch.float32 and cos_sin.shape[1] == rotary_dim, cos_sin.shape
        assert rotary_dim <= min(head_dim, index_head_dim) and rotary_dim % 2 == 0, rotary_dim
        if num_tokens:
            assert int(position_ids.min()) >= 0, "negative position"
            assert int(position_ids.max()) < cos_sin.shape[0], "position past the rope table"


qwen4_exp_qk_norm_rope = Qwen4ExpQkNormRope()
