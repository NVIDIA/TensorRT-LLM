# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BF16 -> MXFP8 (E4M3 + 1x32 UE8M0, R128c4 scales) for the dense block-scaled GEMM (CuTe DSL).

``mxfp8_quantize_r128c4(x)`` returns what ``torch.ops.trtllm.mxfp8_quantize(x, True, 32)`` returns,
bit for bit on every input (NaN and infinity included). A lane owns 16 contiguous values of a row
(half a 1x32 group: two 16-byte loads, one 16-byte code store); ``lpr`` lanes cover a row segment
of ``16 * lpr`` values and a warp a tile of ``rows`` such segments, all loads issued before any
conversion. The family op ``ops/gemm/rubin_mxfp8_quantize`` calls it.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

SF_VEC = 32
#: Values per lane: half of a 1x32 group, so a group is a lane pair (lane, lane ^ 1).
_LANE_VALUES = 16


def _asm(res, operands, text, constraints, *, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        operands,
        text,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _iv(value, *, loc=None, ip=None):
    return Int32(value).ir_value(loc=loc, ip=ip)


def _lv(value, *, loc=None, ip=None):
    return Int64(value).ir_value(loc=loc, ip=ip)


def _words(res, n, *, loc=None, ip=None):
    return tuple(Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(n))


@dsl_user_op
def _griddepcontrol_wait(*, loc=None, ip=None):
    _asm(None, [], "griddepcontrol.wait;", "", loc=loc, ip=ip)


@dsl_user_op
def _griddepcontrol_launch_dependents(*, loc=None, ip=None):
    _asm(None, [], "griddepcontrol.launch_dependents;", "", loc=loc, ip=ip)


@dsl_user_op
def _ldg_32b_if(addr, pred, *, loc=None, ip=None):
    """The 32 bytes at ``addr`` (two 16-byte loads) if ``pred`` is non-zero, else eight zero words."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 8),
        [_lv(addr, loc=loc, ip=ip), _iv(pred, loc=loc, ip=ip)],
        "{\n.reg .pred p;\nsetp.ne.b32 p, $9, 0;\n"
        + "".join(f"mov.b32 ${i}, 0;\n" for i in range(8))
        + "@p ld.global.v4.b32 {$0, $1, $2, $3}, [$8];\n"
        + "@p ld.global.v4.b32 {$4, $5, $6, $7}, [$8+16];\n}",
        "=r,=r,=r,=r,=r,=r,=r,=r,l,r",
        loc=loc,
        ip=ip,
    )
    return _words(res, 8, loc=loc, ip=ip)


@dsl_user_op
def _stg_16b_if(addr, v0, v1, v2, v3, pred, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip)] + [_iv(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3, pred)],
        "{\n.reg .pred p;\nsetp.ne.b32 p, $5, 0;\n@p st.global.v4.b32 [$0], {$1, $2, $3, $4};\n}",
        "l,r,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _stg_4b_if(addr, v, pred, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _iv(v, loc=loc, ip=ip), _iv(pred, loc=loc, ip=ip)],
        "{\n.reg .pred p;\nsetp.ne.b32 p, $2, 0;\n@p st.global.b32 [$0], $1;\n}",
        "l,r,r",
        loc=loc,
        ip=ip,
    )


def _lane_ptx() -> str:
    """One lane's 16 values (operands ``$5..$12``, BF16x2 words, element ``2 i`` in the low half of
    word ``i``); results ``$0..$3`` their E4M3 codes and ``$4`` the scale word of the four groups
    starting at this lane (bytes of lanes +0, +2, +4, +6; used by lanes with ``lane % 8 == 0``).

    The C++ quantizer's arithmetic (quantization.cuh ``cvt_warp_fp16_to_mxfp8``, no fast math):
    ``vecMax`` = ``__hmax2`` (NaN-skipping ``max.bf16x2``) of ``|x|`` over the even and over the odd
    elements, then ``even > odd ? even : odd``; ``sf = vecMax * fl(1/448)`` (``mul.rn``); the byte
    ``cvt.rp.satfinite.ue8m0``; the multiplier ``2^(127 - byte)`` (1 for byte 0, 0 for
    ``vecMax == 0``, the subnormals 2^-127 / 2^-128 for bytes 254 / 255). The multiplier is a power
    of two, so the BF16x2 product is the FP32 one and ``cvt.rn.satfinite.e4m3x2.bf16x2`` rounds it
    as the C++ FP32 conversion does.
    """
    lines = [
        "{",
        ".reg .b32 a<8>, t, lo, hi, eb, sb, mb, r;",
        ".reg .f32 feven, fodd, amax, sf;",
        ".reg .b16 e, h0, h1;",
        ".reg .pred pg, pn, p0, p254, p255;",
    ]
    lines += [f"abs.bf16x2 a{i}, ${5 + i};" for i in range(8)]
    lines += [f"max.bf16x2 a{i}, a{i}, a{i + 1};" for i in (0, 2, 4, 6)]
    lines += ["max.bf16x2 a0, a0, a2;", "max.bf16x2 a4, a4, a6;", "max.bf16x2 a0, a0, a4;"]
    lines += ["shfl.sync.bfly.b32 t, a0, 1, 0x1f, 0xffffffff;", "max.bf16x2 a0, a0, t;"]
    lines += [
        "shl.b32 lo, a0, 16;",
        "and.b32 hi, a0, 0xFFFF0000;",
        "mov.b32 feven, lo;",
        "mov.b32 fodd, hi;",
        "setp.gt.f32 pg, feven, fodd;",
        "selp.f32 amax, feven, fodd, pg;",
        "mul.rn.f32 sf, amax, 0f3B124925;",
        "cvt.rp.satfinite.ue8m0x2.f32 e, sf, sf;",
        "cvt.u32.u16 eb, e;",
        "and.b32 eb, eb, 255;",
        "setp.neu.f32 pn, amax, 0f00000000;",
        "setp.eq.u32 p0, eb, 0;",
        "setp.eq.u32 p254, eb, 254;",
        "setp.eq.u32 p255, eb, 255;",
        "sub.u32 sb, 254, eb;",
        "shl.b32 sb, sb, 23;",
        "selp.b32 sb, 0x00400000, sb, p254;",
        "selp.b32 sb, 0x00200000, sb, p255;",
        "selp.b32 sb, 0x3F800000, sb, p0;",
        "selp.b32 sb, sb, 0, pn;",
        "shr.u32 mb, sb, 16;",
        "shl.b32 r, mb, 16;",
        "or.b32 mb, mb, r;",
    ]
    for j in range(4):
        lines += [
            f"mul.rn.bf16x2 r, ${5 + 2 * j}, mb;",
            "cvt.rn.satfinite.e4m3x2.bf16x2 h0, r;",
            f"mul.rn.bf16x2 r, ${6 + 2 * j}, mb;",
            "cvt.rn.satfinite.e4m3x2.bf16x2 h1, r;",
            f"mov.b32 ${j}, {{h0, h1}};",
        ]
    # every shuffle reads the source lane's own byte (eb); the word accumulates in r
    lines.append("mov.b32 r, eb;")
    for i, d in enumerate((2, 4, 6)):
        lines += [
            f"shfl.sync.down.b32 t, eb, {d}, 0x1f, 0xffffffff;",
            f"shl.b32 t, t, {8 * (i + 1)};",
            "or.b32 r, r, t;",
        ]
    lines += ["mov.b32 $4, r;", "}"]
    return "\n".join(lines)


_LANE_PTX = _lane_ptx()


@dsl_user_op
def _quant_lane(words, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 5),
        [_iv(w, loc=loc, ip=ip) for w in words],
        _LANE_PTX,
        ",".join(["=r"] * 5 + ["r"] * 8),
        loc=loc,
        ip=ip,
    )
    return _words(res, 5, loc=loc, ip=ip)


class Mxfp8QuantizeR128c4Kernel:
    """``x`` BF16 ``[M, K]`` -> E4M3 ``[M, K]`` + UE8M0 R128c4 scales ``[pad128(M) * pad4(K / 32)]``.

    One warp per item: ``rows`` rows x ``16 * lpr`` values (``lpr`` lanes per row, ``32 / lpr``
    rows per load instruction), items ordered column segment fastest, over ``pad128(M)`` rows and
    ``pad4(K / 32) * 32`` columns. Values past ``K`` read as zero and rows past ``M`` load nothing,
    so their groups store scale byte 0; codes are stored only inside ``[M, K]``. Scale word of
    (row ``r``, groups ``4c..4c+3``) at ``((r // 128) * pad4(K / 32) / 4 + c) * 512 + (r % 32) * 16
    + (r // 32 % 4) * 4``.
    """

    def __init__(self, k: int, threads: int, rows: int, lpr: int, use_pdl: bool):
        assert lpr in (8, 16, 32) and threads % 32 == 0 and rows % (32 // lpr) == 0
        assert 128 % rows == 0
        self.k = k
        self.sf_cols = -(-(k // SF_VEC) // 4) * 4
        self.threads = threads
        self.warps = threads // 32
        self.rows = rows
        self.lpr = lpr
        self.rows_per_load = 32 // lpr
        self.seg_cols = _LANE_VALUES * lpr
        self.segs = -(-self.sf_cols * SF_VEC // self.seg_cols)
        self.use_pdl = use_pdl

    @cute.jit
    def __call__(
        self, x_addr: Int64, q_addr: Int64, s_addr: Int64, m: Int32, stream: cuda.CUstream
    ):
        total = ((m + 127) // 128) * (128 // self.rows) * self.segs
        ctas = (total + self.warps - 1) // self.warps
        self.kernel(x_addr, q_addr, s_addr, m, total).launch(
            grid=(ctas, 1, 1), block=(self.threads, 1, 1), stream=stream, use_pdl=self.use_pdl
        )

    @cute.kernel
    def kernel(self, x_addr: Int64, q_addr: Int64, s_addr: Int64, m: Int32, total: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = tidx & 31
        item = bidx * self.warps + (tidx >> 5)
        if cutlass.const_expr(self.use_pdl):
            _griddepcontrol_wait()
        live = Int32(item < total)
        rb = item // self.segs
        seg = item - rb * self.segs
        r0 = rb * self.rows + lane // self.lpr
        col = seg * self.seg_cols + (lane % self.lpr) * _LANE_VALUES
        g = col >> 5
        in_k = live & Int32(col < self.k)
        loads = self.rows // self.rows_per_load
        words = []
        for j in cutlass.range_constexpr(loads):
            r = r0 + j * self.rows_per_load
            src = x_addr + (Int64(r) * self.k + col) * 2
            words.append(_ldg_32b_if(src, in_k & Int32(r < m)))
        for j in cutlass.range_constexpr(loads):
            r = r0 + j * self.rows_per_load
            c0, c1, c2, c3, sw = _quant_lane(words[j])
            _stg_16b_if(q_addr + Int64(r) * self.k + col, c0, c1, c2, c3, in_k & Int32(r < m))
            rr = r & 127
            sf = Int64(r >> 7) * (128 * self.sf_cols) + (g >> 2) * 512 + (rr & 31) * 16
            sf = sf + (rr >> 5) * 4
            _stg_4b_if(s_addr + sf, sw, live & Int32((lane & 7) == 0) & Int32(g < self.sf_cols))
        if cutlass.const_expr(self.use_pdl):
            _griddepcontrol_launch_dependents()


@functools.cache
def _compiled(k: int, threads: int, rows: int, lpr: int, use_pdl: bool):
    dummy = Int64(1 << 20)
    return cute.compile(
        Mxfp8QuantizeR128c4Kernel(k, threads, rows, lpr, use_pdl),
        dummy,
        dummy,
        dummy,
        Int32(1),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


def default_config(m: int, k: int) -> tuple[int, int, int]:
    """``(threads, rows, lpr)`` for an ``[m, k]`` input: 4-row x 128-value warp tiles up to 4M
    values (a launch is mostly fill and drain), 4 x 256 up to 24M, 8 x 256 above. Within 1% of the
    best of 9 tiles on sm_107 at every K3.1 prefill shape (256..8192 rows x K 768..12288)."""
    elems = m * k
    if elems <= 1 << 22:
        return 128, 4, 8
    if elems <= 24 << 20:
        return 128, 4, 16
    return 128, 8, 16


def mxfp8_quantize_r128c4(
    x: torch.Tensor,
    config: tuple[int, int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(E4M3 [M, K], UE8M0 R128c4 scales [pad128(M) * pad4(K / 32)])`` of a BF16 ``[M, K]``."""
    m, k = x.shape
    if x.dtype != torch.bfloat16 or not x.is_contiguous() or k % SF_VEC:
        raise ValueError("mxfp8_quantize_r128c4 needs a contiguous BF16 [M, K] with K % 32 == 0")
    if x.data_ptr() % 16:
        raise ValueError("mxfp8_quantize_r128c4 needs a 16-byte aligned input")
    sf_cols = -(-(k // SF_VEC) // 4) * 4
    sf_rows = -(-m // 128) * 128
    q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty(sf_rows * sf_cols, dtype=torch.uint8, device=x.device)
    if m == 0:
        return q, s
    threads, rows, lpr = config or default_config(m, k)
    _compiled(k, threads, rows, lpr, get_env_enable_pdl())(
        Int64(x.data_ptr()),
        Int64(q.data_ptr()),
        Int64(s.data_ptr()),
        Int32(m),
        cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream),
    )
    return q, s
