# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Token-major gated RMSNorm of the KDA prefill output at prefill row counts (CuTe DSL).

For rows ``r = t * heads + h`` of ``x`` (bf16 ``[M, 128]``) and the gate ``z[t, h, :]`` read from a token-major
``[T, heads, 128]`` view (head blocks contiguous, any token stride), one launch writes

    ``out[r] = bf16(x[r] * rstd(r) * w * gate(z[t, h]))``,  ``rstd = 1 / sqrt(mean(x[r]^2) + eps)``,

``gate`` = ``sigmoid(z)`` (KDA) or ``z * sigmoid(z)`` (GDN), bit for bit the rows of the four-row token-major Triton
kernel (``_rms_norm_gated_fwd_multirow_kernel`` in ``rubin_rms_norm_gated.py``) that it replaces at these row
counts.

Arithmetic: the instructions that kernel's SASS executes, each written with an explicit rounding mode so that ptxas
cannot contract or reassociate them differently:

* the sum of squares in that kernel's order: Triton gives each lane four consecutive columns ``4v .. 4v + 3`` of a
  row and reduces them as ``fma(x0, x0, x2 * x2) + fma(x1, x1, x3 * x3)`` (one ``mul.f32x2``, one
  ``fma.rn.f32x2``, one add), then over the 32 lanes with the butterfly ``xor 16, 8, 4, 2, 1``;
* ``var + eps`` as one ``fma.rn.f32(sum, 1 / 128, eps)`` (ptxas folds ``div.full.f32 sum, 128.0`` into the add),
  ``sqrt.approx.ftz.f32`` and ``div.full.f32 1.0, sqrt`` for ``rstd``;
* ``tl.sigmoid(z)`` = ``div.full.f32 1.0, ex2.approx(z * -log2(e)) + 1``: the multiply, ``ex2.approx.ftz.f32``,
  the add, then ``div.full.f32``'s expansion for a divisor of at least 1 (``rcp.approx`` of the divisor, or of a
  quarter of it scaled back by a quarter above 2^126). The flush-to-zero ``ex2`` gives the same sum as the
  non-flushing one: it differs only below 2^-126, where ``1 + e`` rounds to 1 either way;
* ``((rstd * x) * w) * gate`` (``silu``: ``gate = sigmoid * z``), each a round-to-nearest multiply, and one
  round-to-nearest conversion to bf16.

Layout: a warp handles two rows at a time (a pair). Lane ``L`` holds eight columns ``8 (L % 16) .. + 7`` of row
``L // 16`` of the pair: it loads them as one 16-byte vector each of ``x`` and ``z`` and stores 16 bytes of the
output. Those columns are the Triton lanes ``v = 2 (L % 16)`` and ``v + 1``, so the lane forms both lanes' partial
sums, runs the first four butterfly levels (``xor 16, 8, 4, 2`` over ``v``) as shuffles ``xor 8, 4, 2, 1`` inside
its half-warp and the last level (``xor 1``) as its own add: the same tree, the same bits.

Schedule: a resident grid (at most ``ctas_per_sm`` CTAs of ``warps`` warps per SM, the register cap set to fit
them) strides over the pairs in batches: in each batch a warp takes ``depth`` consecutive pairs, issues all their
loads, then computes and stores them. The batch's first pair reads every load of the batch through a zero mask
(``_fold``), so ptxas cannot sink a load into the batch's arithmetic. ``prefetch`` instead loads a batch's
successor before computing it (two batches in registers), held above the batch's stores by a store that never
executes (``_issue_point``). The weight's eight columns of a lane are loaded once. The schedule does not change a
bit.
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda

#: Row length (the KDA head dim) the kernel is built for.
WIDTH = 128
#: Columns per lane and row of a pair (one 16-byte vector of bf16).
_LANE_COLS = 8
_OPTIONS = "--enable-tvm-ffi"

_NEG_LOG2E = "0fBFB8AA3B"  # -log2(e), tl.sigmoid's ex2 argument scale
_INV_N = "0f3C000000"  # 1 / 128, div.full.f32 x, 128.0 as ptxas folds it
_ONE = "0f3F800000"
_QUARTER = "0f3E800000"
_TWO_126 = "0f7E800000"  # div.full.f32's large-divisor threshold


def _asm(res, operands, text, constraints, side_effects=False, *, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        operands,
        text,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _fv(value, *, loc=None, ip=None):
    return Float32(value).ir_value(loc=loc, ip=ip)


def _iv(value, *, loc=None, ip=None):
    return Int32(value).ir_value(loc=loc, ip=ip)


def _lv(value, *, loc=None, ip=None):
    return Int64(value).ir_value(loc=loc, ip=ip)


def _unpack(res, ty, cls, count, *, loc=None, ip=None):
    return [cls(llvm.extractvalue(ty(), res, [i], loc=loc, ip=ip)) for i in range(count)]


# ---- memory ----


@dsl_user_op
def _ld_v4(addr, pred, *, loc=None, ip=None):
    """16 bytes at ``addr`` (zeros where ``pred`` is 0, memory untouched). A plain (coherent) load: ptxas keeps
    it above every later store, which is what holds a batch's loads together (see :func:`_issue_point`)."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_lv(addr, loc=loc, ip=ip), _iv(pred, loc=loc, ip=ip)],
        "{\n.reg .pred p;\nsetp.ne.b32 p, $5, 0;\nmov.b32 $0, 0;\nmov.b32 $1, 0;\nmov.b32 $2, 0;\n"
        "mov.b32 $3, 0;\n@p ld.global.L1::no_allocate.v4.b32 {$0, $1, $2, $3}, [$4];\n}",
        "=r,=r,=r,=r,l,r",
        True,
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32, Int32, 4, loc=loc, ip=ip)


@dsl_user_op
def _issue_point(addr, never, *, loc=None, ip=None):
    """A store that never executes (``never`` is 0 at run time, which ptxas cannot know): the loads before it in
    program order must be issued before it, so ptxas cannot sink a batch's loads into the batch's arithmetic."""
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _iv(never, loc=loc, ip=ip)],
        "{\n.reg .pred p;\nsetp.ne.b32 p, $1, 0;\n@p st.global.b32 [$0], 0;\n}",
        "l,r",
        True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_v4(addr, words, pred, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip)]
        + [_iv(w, loc=loc, ip=ip) for w in words]
        + [_iv(pred, loc=loc, ip=ip)],
        "{\n.reg .pred p;\nsetp.ne.b32 p, $5, 0;\n@p st.global.v4.b32 [$0], {$1, $2, $3, $4};\n}",
        "l,r,r,r,r,r",
        True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _fold(acc, word, zero, *, loc=None, ip=None):
    """``acc ^ (word & zero)``: ``acc`` itself at run time (``zero`` is 0), but data-dependent on ``word``."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(acc, loc=loc, ip=ip), _iv(word, loc=loc, ip=ip), _iv(zero, loc=loc, ip=ip)],
            "{\n.reg .b32 t;\nand.b32 t, $2, $3;\nxor.b32 $0, $1, t;\n}",
            "=r,r,r,r",
            loc=loc,
            ip=ip,
        )
    )


# ---- arithmetic ----


@dsl_user_op
def _bf16x2_to_f32(word, *, loc=None, ip=None):
    """``(float(lo), float(hi))`` of a packed bf16x2 word (exact)."""
    res = _asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [_iv(word, loc=loc, ip=ip)],
        "{\n.reg .b32 t;\nshl.b32 t, $2, 16;\nmov.b32 $0, t;\nand.b32 t, $2, 0xffff0000;\nmov.b32 $1, t;\n}",
        "=f,=f,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.f32, Float32, 2, loc=loc, ip=ip)


@dsl_user_op
def _f32_bits(word, *, loc=None, ip=None):
    return Float32(
        _asm(T.f32(), [_iv(word, loc=loc, ip=ip)], "mov.b32 $0, $1;", "=f,r", loc=loc, ip=ip)
    )


@dsl_user_op
def _sum_sq4(x0, x1, x2, x3, *, loc=None, ip=None):
    """One Triton lane's partial: ``fma(x0, x0, x2 * x2) + fma(x1, x1, x3 * x3)``."""
    return Float32(
        _asm(
            T.f32(),
            [_fv(v, loc=loc, ip=ip) for v in (x0, x1, x2, x3)],
            "{\n.reg .b64 a, b, p, f;\n.reg .f32 lo, hi;\nmov.b64 a, {$1, $2};\nmov.b64 b, {$3, $4};\n"
            "mul.rn.f32x2 p, b, b;\nfma.rn.f32x2 f, a, a, p;\nmov.b64 {lo, hi}, f;\nadd.rn.f32 $0, lo, hi;\n}",
            "=f,f,f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bfly_add(value, offset: int, *, loc=None, ip=None):
    """``value + shfl.bfly(value, offset)`` over the full warp."""
    return Float32(
        _asm(
            T.f32(),
            [_fv(value, loc=loc, ip=ip)],
            "{\n.reg .f32 t;\n"
            f"shfl.sync.bfly.b32 t, $1, {offset}, 0x1f, 0xffffffff;\n"
            "add.rn.f32 $0, $1, t;\n}",
            "=f,f",
            True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _add(a, b, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_fv(a, loc=loc, ip=ip), _fv(b, loc=loc, ip=ip)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _rstd(total, eps, *, loc=None, ip=None):
    """``1 / sqrt(total / 128 + eps)``: one FFMA, ``sqrt.approx.ftz``, ``div.full``."""
    return Float32(
        _asm(
            T.f32(),
            [_fv(total, loc=loc, ip=ip), _fv(eps, loc=loc, ip=ip)],
            "{\n.reg .f32 t, q, one;\n"
            f"fma.rn.f32 t, $1, {_INV_N}, $2;\nsqrt.approx.ftz.f32 q, t;\nmov.b32 one, {_ONE};\n"
            "div.full.f32 $0, one, q;\n}",
            "=f,f,f",
            loc=loc,
            ip=ip,
        )
    )


def _gate_out_text(silu: bool) -> str:
    """Two outputs of one lane: $0 the packed bf16 pair; $1 rstd, $2/$3 x, $4/$5 w, $6/$7 z."""
    lines = [
        "{",
        ".reg .b64 rr, xx, ww, zz, c2, one2, q2, t2, e2, d2, r2, s2, g2, y2;",
        ".reg .f32 t0, t1, e0, e1, d0, d1, r0, r1, s0, s1, lo, hi, c, one, quarter;",
        ".reg .pred p0, p1;",
        f"mov.b32 c, {_NEG_LOG2E};",
        f"mov.b32 one, {_ONE};",
        f"mov.b32 quarter, {_QUARTER};",
        "mov.b64 rr, {$1, $1};",
        "mov.b64 xx, {$2, $3};",
        "mov.b64 ww, {$4, $5};",
        "mov.b64 zz, {$6, $7};",
        "mov.b64 c2, {c, c};",
        "mov.b64 one2, {one, one};",
        # sigmoid(z) = 1 / (ex2(z * -log2 e) + 1)
        "mul.rn.f32x2 t2, zz, c2;",
        "mov.b64 {t0, t1}, t2;",
        "ex2.approx.ftz.f32 e0, t0;",
        "ex2.approx.ftz.f32 e1, t1;",
        "mov.b64 e2, {e0, e1};",
        "add.rn.f32x2 d2, e2, one2;",
        "mov.b64 {d0, d1}, d2;",
        # div.full.f32 1.0, d for d >= 1 (or NaN): above 2^126 the reciprocal of d / 4, scaled by 1 / 4
        f"setp.gt.f32 p0, d0, {_TWO_126};",
        f"setp.gt.f32 p1, d1, {_TWO_126};",
        "@p0 mul.rn.f32 d0, d0, quarter;",
        "@p1 mul.rn.f32 d1, d1, quarter;",
        "rcp.approx.ftz.f32 r0, d0;",
        "rcp.approx.ftz.f32 r1, d1;",
        "selp.f32 s0, quarter, one, p0;",
        "selp.f32 s1, quarter, one, p1;",
        "mov.b64 r2, {r0, r1};",
        "mov.b64 s2, {s0, s1};",
        "mul.rn.f32x2 g2, r2, s2;",
    ]
    if silu:
        lines.append("mul.rn.f32x2 g2, g2, zz;")
    lines += [
        # ((rstd * x) * w) * gate, one rounding to bf16
        "mul.rn.f32x2 y2, rr, xx;",
        "mul.rn.f32x2 y2, y2, ww;",
        "mul.rn.f32x2 y2, y2, g2;",
        "mov.b64 {lo, hi}, y2;",
        "cvt.rn.bf16x2.f32 $0, hi, lo;",
        "}",
    ]
    return "\n".join(lines)


_GATE_OUT = {silu: _gate_out_text(silu) for silu in (False, True)}


@dsl_user_op
def _gate_out2(rstd, x0, x1, w0, w1, z0, z1, silu: bool, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_fv(v, loc=loc, ip=ip) for v in (rstd, x0, x1, w0, w1, z0, z1)],
            _GATE_OUT[silu],
            "=r,f,f,f,f,f,f,f",
            loc=loc,
            ip=ip,
        )
    )


class RmsNormGatedTokenMajorKernel:
    """``out = bf16(rmsnorm(x) * w * gate(z))`` over ``rows`` rows of 128, ``z`` token-major (see the module)."""

    def __init__(
        self,
        heads: int,
        silu: bool,
        w_f32: bool,
        warps: int,
        ctas_per_sm: int,
        depth: int,
        prefetch: bool,
    ):
        self.heads = heads
        self.silu = silu
        self.w_f32 = w_f32
        self.warps = warps
        self.ctas_per_sm = ctas_per_sm
        self.depth = depth
        self.prefetch = prefetch

    @cute.jit
    def __call__(
        self,
        x_addr: Int64,
        z_addr: Int64,
        w_addr: Int64,
        out: cute.Tensor,
        x_stride: Int64,
        z_stride: Int64,
        rows: Int32,
        ctas: Int32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        # ``out`` (bf16 [rows, 128], contiguous) is the one tensor argument: TVM-FFI takes the launch stream from its
        # device. ``min_blocks_per_mp`` caps the registers so that ``ctas_per_sm`` CTAs fit on an SM: the grid
        # (at most that many per SM) is resident at once, as the strided schedule assumes.
        self.kernel(
            x_addr, z_addr, w_addr, out.iterator.toint(), x_stride, z_stride, rows, eps
        ).launch(
            grid=(ctas, 1, 1),
            block=(32 * self.warps, 1, 1),
            stream=stream,
            min_blocks_per_mp=self.ctas_per_sm,
        )

    def load_pair(self, x_addr, z_addr, x_stride, z_stride, rows, pair, half, col):
        """The lane's 16-byte vectors of ``x`` and ``z`` in pair ``pair`` (zeros past ``rows``)."""
        row = pair * 2 + half
        ok = Int32(row < rows)
        tok = row // self.heads
        head = row - tok * self.heads
        xw = _ld_v4(x_addr + (Int64(row) * x_stride + Int64(col)) * 2, ok)
        zw = _ld_v4(z_addr + (Int64(tok) * z_stride + Int64(head * WIDTH + col)) * 2, ok)
        return xw, zw

    def compute_pair(self, xw, zw, wf, eps, o_addr, rows, pair, half, col):
        row = pair * 2 + half
        xf = []
        for j in range(4):
            lo, hi = _bf16x2_to_f32(xw[j])
            xf.append(lo)
            xf.append(hi)
        # Triton lanes v = 2 (L % 16) (columns 0-3 of the lane) and v + 1 (columns 4-7).
        a = _sum_sq4(xf[0], xf[1], xf[2], xf[3])
        b = _sum_sq4(xf[4], xf[5], xf[6], xf[7])
        for offset in range(4):
            a = _bfly_add(a, 8 >> offset)
            b = _bfly_add(b, 8 >> offset)
        rstd = _rstd(_add(a, b), eps)
        words = []
        for j in range(4):
            z0, z1 = _bf16x2_to_f32(zw[j])
            words.append(
                _gate_out2(
                    rstd, xf[2 * j], xf[2 * j + 1], wf[2 * j], wf[2 * j + 1], z0, z1, self.silu
                )
            )
        _st_v4(o_addr + (Int64(row) * WIDTH + Int64(col)) * 2, words, Int32(row < rows))

    @cute.kernel
    def kernel(
        self,
        x_addr: Int64,
        z_addr: Int64,
        w_addr: Int64,
        o_addr: Int64,
        x_stride: Int64,
        z_stride: Int64,
        rows: Int32,
        eps: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        lane = tidx % 32
        half = lane // 16
        col = (lane % 16) * _LANE_COLS
        warp = bidx * self.warps + tidx // 32
        warps_total = gdim * self.warps
        pairs = (rows + 1) // 2
        span = warps_total * self.depth  # pairs per batch over the grid
        batches = (pairs + span - 1) // span
        never = Int32(rows < 0)  # 0 at run time; ptxas cannot know it

        wf = []
        if cutlass.const_expr(self.w_f32):
            for k in cutlass.range_constexpr(2):
                for word in _ld_v4(w_addr + Int64(col * 4 + 16 * k), Int32(1)):
                    wf.append(_f32_bits(word))
        else:
            for word in _ld_v4(w_addr + Int64(col * 2), Int32(1)):
                lo, hi = _bf16x2_to_f32(word)
                wf.append(lo)
                wf.append(hi)

        if cutlass.const_expr(not self.prefetch):
            for batch in cutlass.range(batches, unroll=1):
                first = (batch * warps_total + warp) * self.depth
                loads = []
                for d in cutlass.range_constexpr(self.depth):
                    loads.append(
                        self.load_pair(
                            x_addr, z_addr, x_stride, z_stride, rows, first + d, half, col
                        )
                    )
                # The first pair's arithmetic reads every load of the batch (through a zero mask), so ptxas issues
                # them all before it instead of sinking them into the batch.
                dep = Int32(0)
                for d in cutlass.range_constexpr(self.depth):
                    dep = _fold(_fold(dep, loads[d][0][3], never), loads[d][1][3], never)
                loads[0][0][0] = loads[0][0][0] ^ dep
                for d in cutlass.range_constexpr(self.depth):
                    self.compute_pair(
                        loads[d][0], loads[d][1], wf, eps, o_addr, rows, first + d, half, col
                    )
        else:
            # Two batches per trip: batch 2i's vectors arrive in ``buf`` (loaded one trip earlier), batch 2i + 1's
            # in registers of this trip; each is loaded before its predecessor is computed.
            buf = cute.make_rmem_tensor((self.depth * 8,), Int32)
            first = warp * self.depth
            for d in cutlass.range_constexpr(self.depth):
                xw, zw = self.load_pair(
                    x_addr, z_addr, x_stride, z_stride, rows, first + d, half, col
                )
                for j in cutlass.range_constexpr(4):
                    buf[d * 8 + j] = xw[j]
                    buf[d * 8 + 4 + j] = zw[j]
            for trip in cutlass.range((batches + 1) // 2, unroll=1):
                cur = (2 * trip * warps_total + warp) * self.depth
                nxt = cur + warps_total * self.depth
                loads = []
                for d in cutlass.range_constexpr(self.depth):
                    loads.append(
                        self.load_pair(x_addr, z_addr, x_stride, z_stride, rows, nxt + d, half, col)
                    )
                _issue_point(o_addr, never)
                for d in cutlass.range_constexpr(self.depth):
                    xw = [buf[d * 8 + j] for j in range(4)]
                    zw = [buf[d * 8 + 4 + j] for j in range(4)]
                    self.compute_pair(xw, zw, wf, eps, o_addr, rows, cur + d, half, col)
                after = nxt + warps_total * self.depth
                for d in cutlass.range_constexpr(self.depth):
                    xw, zw = self.load_pair(
                        x_addr, z_addr, x_stride, z_stride, rows, after + d, half, col
                    )
                    for j in cutlass.range_constexpr(4):
                        buf[d * 8 + j] = xw[j]
                        buf[d * 8 + 4 + j] = zw[j]
                _issue_point(o_addr, never)
                for d in cutlass.range_constexpr(self.depth):
                    self.compute_pair(
                        loads[d][0], loads[d][1], wf, eps, o_addr, rows, nxt + d, half, col
                    )


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------

#: Launch shape: warps per CTA, CTAs per SM, pairs per warp and batch, prefetch of the next batch.
DEFAULT_CONFIG = (8, 3, 4, False)


@functools.cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@functools.cache
def _compiled(
    heads: int, silu: bool, w_f32: bool, warps: int, ctas_per_sm: int, depth: int, prefetch: bool
):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "rubin_rms_norm_gated_token_major_cute compiles on the first call of a variant; make that call "
            "before CUDA graph capture"
        )
    kernel = RmsNormGatedTokenMajorKernel(heads, silu, w_f32, warps, ctas_per_sm, depth, prefetch)
    addr = Int64(1 << 20)
    out = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (cute.sym_int(), WIDTH), stride_order=(1, 0)
    )
    return cute.compile(
        kernel,
        addr,
        addr,
        addr,
        out,
        Int64(WIDTH),
        Int64(WIDTH),
        Int32(2),
        Int32(1),
        Float32(1e-5),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options=_OPTIONS,
    )


def eligible(x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether the kernel takes this call: bf16 ``x`` ``[M, 128]`` with unit column stride and 16-byte aligned rows,
    bf16 ``z`` ``[T, heads, 128]`` with contiguous head blocks and 16-byte aligned token rows, a contiguous bf16 or
    fp32 ``weight`` ``[128]``, all on one device, ``M < 2^31``."""
    if x.dim() != 2 or z.dim() != 3 or x.shape[1] != WIDTH or z.shape[2] != WIDTH:
        return False
    if x.dtype is not torch.bfloat16 or z.dtype is not torch.bfloat16:
        return False
    if weight.dtype not in (torch.bfloat16, torch.float32) or weight.numel() != WIDTH:
        return False
    if not weight.is_contiguous() or z.shape[0] * z.shape[1] != x.shape[0] or x.shape[0] >= 2**31:
        return False
    if x.stride(1) != 1 or z.stride(2) != 1 or z.stride(1) != WIDTH:
        return False
    if x.stride(0) % _LANE_COLS or z.stride(0) % _LANE_COLS:
        return False
    if (x.data_ptr() | z.data_ptr() | weight.data_ptr()) % 16:
        return False
    return x.device == z.device == weight.device


def rms_norm_gated_token_major_cute(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    silu: bool,
    *,
    config: tuple[int, int, int, bool] = DEFAULT_CONFIG,
) -> torch.Tensor:
    """``bf16(rmsnorm(x) * weight * gate(z))`` as a new contiguous ``[M, 128]`` tensor, ``gate`` = ``z * sigmoid(z)``
    with ``silu`` else ``sigmoid(z)``; the inputs must be :func:`eligible`. ``config``: (warps per CTA, CTAs per SM,
    pairs per warp and batch, prefetch); it does not change a bit. Compiles once per variant (heads, gate, weight
    dtype, config) on its first call, which must not be under CUDA graph capture."""
    rows = x.shape[0]
    out = torch.empty((rows, WIDTH), dtype=torch.bfloat16, device=x.device)
    if rows == 0:
        return out
    warps, ctas_per_sm, depth, prefetch = config
    pairs = (rows + 1) // 2
    ctas = max(1, min(ctas_per_sm * _num_sms(x.device.index), -(-pairs // (warps * depth))))
    _compiled(z.shape[1], silu, weight.dtype is torch.float32, warps, ctas_per_sm, depth, prefetch)(
        x.data_ptr(),
        z.data_ptr(),
        weight.data_ptr(),
        out,
        x.stride(0),
        z.stride(0),
        rows,
        ctas,
        eps,
    )
    return out


__all__ = [
    "DEFAULT_CONFIG",
    "RmsNormGatedTokenMajorKernel",
    "WIDTH",
    "eligible",
    "rms_norm_gated_token_major_cute",
]
