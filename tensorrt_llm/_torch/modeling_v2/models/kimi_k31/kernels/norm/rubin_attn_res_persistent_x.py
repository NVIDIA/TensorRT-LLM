# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 attention-residual mix + RMSNorm on a persistent grid, with the generation step's fused epilogues.

For ``T`` rows of hidden size 7168, the bank ``block`` ``[N - 1, T, 7168]`` (``2 <= N <= 9``) and the prefix row
``x``, one launch computes per row

* ``rsigma_n = rsqrt(mean(v_n^2) + eps)``, ``logit_n = rsigma_n * <v_n, res_w * rms_w>`` over the candidates
  ``v_n`` (the bank rows, then ``x``),
* an online softmax over chunks of three candidates: ``acc = sum_n exp(logit_n - m) v_n``,
  ``s = sum_n exp(logit_n - m)``,
* ``out = bf16(acc * rsqrt(mean(acc^2) + out_eps * s^2) * out_w)`` (the mixture's norm with ``s`` folded in),

where ``x`` is ``prefix``, ``bf16(prefix + addend)`` or ``bf16(prefix + bf16(addend + addend2))`` (the MoE residual
update), and the epilogues store the updated prefix (with an addend), ``float(out)`` (``f32``) and the MXFP8 form of
``out`` (``mx``: E4M3 codes and the 128x4 swizzled UE8M0 scales of ``quantize_with_block_size<FP16_TO_MXFP8, bf16,
32, true, 32>``, the padding rows' scales zero).

This is the engine's former ``attn_res_add_rmsnorm_persistent_x_fwd`` (``attnResFwdPersistentX.cu``) in CuTe DSL,
byte for byte. Schedule (the C++ kernel's): one CTA of 288 threads per SM (``grid_cap``: ``min(SMs, T)``) loops over
the rows. Warp 0's elected lane streams each row's candidates, three per chunk, into a two-deep ring of shared-memory
buffers with ``cp.async.bulk`` (the chunk holding the prefix also brings the addends); warps 1-8 form two groups of
four that own alternate 1024-column tiles (three each) and one half of the last tile: each thread holds 28 values of
a candidate (three vectors of 8, one of 4). Per chunk a thread loads its values from shared memory (adding the
addends to the prefix and storing the updated row), parks them in TMEM (``tcgen05.st``), accumulates its sums of
squares and dot products, releases the buffer, and the warps reduce the statistics (warp butterflies, then eight
partials per candidate through shared memory); every consumer thread then forms the logits and the softmax weights
and folds the chunk into its accumulator from TMEM (``tcgen05.ld``). The output norm reduces the accumulator's sum
of squares the same way; the output norm weight arrives once per CTA by ``cp.async.bulk``. PDL: the kernel inits its
barriers before ``griddepcontrol.wait`` and releases its dependents after its last store.

Arithmetic: every float instruction is the one the C++ binary executes (it was built with ``--use_fast_math``; the
instructions were read from its SASS): FTZ ``FMUL`` / ``FADD`` / ``FFMA`` with round-to-nearest, packed ``fma.rn.f32x2``
/ ``add.rn.f32x2`` / ``mul.rn.f32x2``, ``ex2.approx.ftz`` and ``rsqrt.approx.ftz``, the shuffle trees in the same
order, ``x / 7168`` as the multiply by ``0x39124925`` (fused with ``+ eps`` into one FFMA for the candidate norms,
as ptxas fused ``div.approx`` and the add) and the bf16 adds of the addends. The MXFP8 epilogue issues the IEEE
instructions of the catalog quantizer (built without fast math): ``mul.rn.f32`` by ``0x3B124925``
(``rcp.approx.ftz(448)`` as ptxas folds it) and by the scale, ``cvt.rp.satfinite.ue8m0x2``, a non-flushing zero test
and the non-FTZ ``ex2.approx`` (argument halved and result squared below -126). Each instruction is written with an
explicit rounding mode, so ptxas cannot contract or reassociate differently from the C++ build.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

#: Hidden size the kernel is specialized for.
HIDDEN_SIZE = 7168
#: Candidate counts (bank rows + the prefix row) the kernel is built for.
MIN_CANDIDATES, MAX_CANDIDATES = 2, 9

_K_TILE = 1024
_THREADS = 288  # 1 producer warp + 8 consumer warps
_CONSUMER_THREADS = 256
_CONSUMER_WARPS = 8
_NC = 3  # candidates per chunk
_CD = 2  # chunk pipeline depth
_VEC = 8
_SLICES = 4  # per thread: three 8-wide vectors and the last tile's 4-wide one
_ACC = 28  # values per thread
_TMEM_COLS = 256
_TMEM_GROUP_COLS = _SLICES * _NC * _VEC  # 96
_NAMED_BARRIER = 8  # the C++ kernel's FIRST_USER_NAMED_BARRIER + 0
_ROW_BYTES = HIDDEN_SIZE * 2
_SF_COLS = HIDDEN_SIZE // 32  # 224 UE8M0 scales per row
_PLAN_BYTES = 256  # bar_ready[2], bar_consumed[2], bar_out_w, ws_stats[8][3] (float2), tmem_base

_INV_H = "0f39124925"  # 1 / 7168 as ptxas folds div.approx.ftz.f32 x, 7168.0
_LOG2E = "0f3FB8AA3B"
_NEG_FLT_MAX = "0fFF7FFFFF"
_RCP_448 = "0f3B124925"  # rcp.approx.ftz.f32(448.0) as ptxas folds it


# ---------------------------------------------------------------------------
# Inline PTX
# ---------------------------------------------------------------------------


def _asm(res, operands, text, constraints, *, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        operands,
        text,
        constraints,
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _asm_se(res, operands, text, constraints, *, loc=None, ip=None):
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


def _fv(value, *, loc=None, ip=None):
    return Float32(value).ir_value(loc=loc, ip=ip)


def _unpack(res, ty, cls, count, *, loc=None, ip=None):
    return [cls(llvm.extractvalue(ty(), res, [i], loc=loc, ip=ip)) for i in range(count)]


def _f32_op(text, *args, loc=None, ip=None):
    constraints = "=f," + ",".join("f" for _ in args)
    return Float32(
        _asm(T.f32(), [_fv(a, loc=loc, ip=ip) for a in args], text, constraints, loc=loc, ip=ip)
    )


# ---- scalar FP32 (the fast-math build's instructions, rounding explicit) ----


@dsl_user_op
def _const(bits: str, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [], f"mov.b32 $0, {bits};", "=f", loc=loc, ip=ip))


@dsl_user_op
def _mul_ftz(a, b, *, loc=None, ip=None):
    return _f32_op("mul.rn.ftz.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def _add_ftz(a, b, *, loc=None, ip=None):
    return _f32_op("add.rn.ftz.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def _fma_ftz(a, b, c, *, loc=None, ip=None):
    return _f32_op("fma.rn.ftz.f32 $0, $1, $2, $3;", a, b, c, loc=loc, ip=ip)


@dsl_user_op
def _max_ftz(a, b, *, loc=None, ip=None):
    return _f32_op("max.ftz.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def _rsigma(sum_sq, eps, *, loc=None, ip=None):
    """``rsqrtf(sum_sq / 7168 + eps)``: one FFMA.FTZ (ptxas fused div.approx and the add), MUFU.RSQ."""
    return _f32_op(
        f"{{\n.reg .f32 t;\nfma.rn.ftz.f32 t, $1, {_INV_H}, $2;\nrsqrt.approx.ftz.f32 $0, t;\n}}",
        sum_sq,
        eps,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _exp_diff(a, b, *, loc=None, ip=None):
    """``exp2f((a - b) * LOG2E)``: FADD.FTZ, FMUL.FTZ, MUFU.EX2."""
    return _f32_op(
        "{\n.reg .f32 t;\nsub.rn.ftz.f32 t, $1, $2;\n"
        f"mul.rn.ftz.f32 t, t, {_LOG2E};\nex2.approx.ftz.f32 $0, t;\n}}",
        a,
        b,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _out_rsigma(sum_sq, eps, s, *, loc=None, ip=None):
    """``rsqrtf(sum_sq / 7168 + eps * s * s)``: FMUL.FTZ by 1/7168, FMUL.FTZ ``eps * s``, FFMA.FTZ
    ``s * (eps * s) + sum_sq / 7168``, MUFU.RSQ."""
    return _f32_op(
        "{\n.reg .f32 a, b, c;\n"
        f"mul.rn.ftz.f32 a, $1, {_INV_H};\nmul.rn.ftz.f32 b, $2, $3;\n"
        "fma.rn.ftz.f32 c, $3, b, a;\nrsqrt.approx.ftz.f32 $0, c;\n}",
        sum_sq,
        eps,
        s,
        loc=loc,
        ip=ip,
    )


# ---- packed FP32 pairs (the C++ kernel's float2 inline PTX) ----


def _pair_op(text, pairs, *, loc=None, ip=None):
    regs = [f"r{i}" for i in range(len(pairs))]
    body = "".join(f"mov.b64 {r}, {{${2 + 2 * i}, ${3 + 2 * i}}};\n" for i, r in enumerate(regs))
    res = _asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [_fv(v, loc=loc, ip=ip) for p in pairs for v in p],
        "{\n.reg .b64 "
        + ", ".join(regs + ["rd"])
        + ";\n"
        + body
        + text
        + "mov.b64 {$0, $1}, rd;\n}",
        "=f,=f," + ",".join("f" for _ in range(2 * len(pairs))),
        loc=loc,
        ip=ip,
    )
    return tuple(_unpack(res, T.f32, Float32, 2, loc=loc, ip=ip))


@dsl_user_op
def _ffma2(a, b, c, *, loc=None, ip=None):
    """``fma.rn.f32x2``: ``(a.x * b.x + c.x, a.y * b.y + c.y)``."""
    return _pair_op("fma.rn.f32x2 rd, r0, r1, r2;\n", (a, b, c), loc=loc, ip=ip)


@dsl_user_op
def _fadd2(a, b, *, loc=None, ip=None):
    return _pair_op("add.rn.f32x2 rd, r0, r1;\n", (a, b), loc=loc, ip=ip)


@dsl_user_op
def _fmul2(a, b, *, loc=None, ip=None):
    return _pair_op("mul.rn.f32x2 rd, r0, r1;\n", (a, b), loc=loc, ip=ip)


# ---- bf16 ----


@dsl_user_op
def _bf16x2_to_f32(word, *, loc=None, ip=None):
    """``(float(lo), float(hi))`` of a packed bf16x2 word (exact)."""
    res = _asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [_iv(word, loc=loc, ip=ip)],
        "{\n.reg .b32 t;\nshl.b32 t, $2, 16;\nmov.b32 $0, t;\n"
        "and.b32 t, $2, 0xffff0000;\nmov.b32 $1, t;\n}",
        "=f,=f,r",
        loc=loc,
        ip=ip,
    )
    return tuple(_unpack(res, T.f32, Float32, 2, loc=loc, ip=ip))


@dsl_user_op
def _f32_to_bf16x2(lo, hi, *, loc=None, ip=None):
    """Packed ``{bf16_rn(hi), bf16_rn(lo)}``: two ``cvt.rn.bf16.f32`` (``__float2bfloat16``)."""
    return Int32(
        _asm(
            T.i32(),
            [_fv(lo, loc=loc, ip=ip), _fv(hi, loc=loc, ip=ip)],
            "{\n.reg .b16 l, h;\ncvt.rn.bf16.f32 l, $1;\ncvt.rn.bf16.f32 h, $2;\nmov.b32 $0, {l, h};\n}",
            "=r,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _hadd2(a, b, *, loc=None, ip=None):
    """``__hadd2`` on bf16x2 words (round to nearest, subnormals kept)."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(a, loc=loc, ip=ip), _iv(b, loc=loc, ip=ip)],
            "add.rn.bf16x2 $0, $1, $2;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bf16x2_hi_lo_f32(word, *, loc=None, ip=None):
    """``float(bf16)`` of both halves as raw FP32 bits ``(lo << 16, hi & 0xffff0000)``."""
    res = _asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [_iv(word, loc=loc, ip=ip)],
        "shl.b32 $0, $2, 16;\nand.b32 $1, $2, 0xffff0000;",
        "=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32, Int32, 2, loc=loc, ip=ip)


# ---- shuffles ----


@dsl_user_op
def _shfl_bfly(value, offset: int, *, loc=None, ip=None):
    return Float32(
        _asm_se(
            T.f32(),
            [_fv(value, loc=loc, ip=ip)],
            f"shfl.sync.bfly.b32 $0, $1, {offset}, 0x1f, 0xffffffff;",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shfl_down8(value, offset: int, *, loc=None, ip=None):
    """``__shfl_down_sync(0xffffffff, value, offset, 8)``."""
    return Float32(
        _asm_se(
            T.f32(),
            [_fv(value, loc=loc, ip=ip)],
            f"shfl.sync.down.b32 $0, $1, {offset}, 0x181f, 0xffffffff;",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shfl_idx(value, lane: int, *, loc=None, ip=None):
    return Float32(
        _asm_se(
            T.f32(),
            [_fv(value, loc=loc, ip=ip)],
            f"shfl.sync.idx.b32 $0, $1, {lane}, 0x1f, 0xffffffff;",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


# ---- barriers, PDL, TMA, TMEM ----


@dsl_user_op
def _elect_one(*, loc=None, ip=None):
    """1 on the lane ``elect.sync`` picks in a full warp, else 0."""
    return Int32(
        _asm_se(
            T.i32(),
            [],
            "{\n.reg .pred p;\n.reg .b32 r;\nelect.sync r|p, 0xffffffff;\nselp.b32 $0, 1, 0, p;\n}",
            "=r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bar_sync_all(*, loc=None, ip=None):
    _asm_se(None, [], "bar.sync 0;", "", loc=loc, ip=ip)


@dsl_user_op
def _bar_sync_consumers(*, loc=None, ip=None):
    _asm_se(None, [], f"bar.sync {_NAMED_BARRIER}, {_CONSUMER_THREADS};", "", loc=loc, ip=ip)


@dsl_user_op
def _syncwarp(*, loc=None, ip=None):
    _asm_se(None, [], "bar.warp.sync 0xffffffff;", "", loc=loc, ip=ip)


@dsl_user_op
def _griddepcontrol_wait(*, loc=None, ip=None):
    _asm_se(None, [], "griddepcontrol.wait;", "", loc=loc, ip=ip)


@dsl_user_op
def _griddepcontrol_launch_dependents(*, loc=None, ip=None):
    _asm_se(None, [], "griddepcontrol.launch_dependents;", "", loc=loc, ip=ip)


@dsl_user_op
def _mbar_init(addr, count: int, *, loc=None, ip=None):
    _asm_se(
        None,
        [_iv(addr, loc=loc, ip=ip)],
        f"mbarrier.init.shared::cta.b64 [$0], {count};",
        "r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _fence_mbar_init(*, loc=None, ip=None):
    _asm_se(None, [], "fence.mbarrier_init.release.cluster;", "", loc=loc, ip=ip)


@dsl_user_op
def _mbar_expect_tx(addr, nbytes: int, *, loc=None, ip=None):
    _asm_se(
        None,
        [_iv(addr, loc=loc, ip=ip)],
        f"mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], {nbytes};",
        "r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mbar_arrive(addr, *, loc=None, ip=None):
    _asm_se(
        None,
        [_iv(addr, loc=loc, ip=ip)],
        "{\n.reg .b64 state;\nmbarrier.arrive.shared::cta.b64 state, [$0];\n}",
        "r",
        loc=loc,
        ip=ip,
    )


_WAIT_LABEL = [0]


@dsl_user_op
def _mbar_wait(addr, parity, *, loc=None, ip=None):
    """Spin on ``mbarrier.try_wait.parity`` until the phase with ``parity`` completed."""
    _WAIT_LABEL[0] += 1
    lab = f"K31ARX_WAIT{_WAIT_LABEL[0]}"
    _asm_se(
        None,
        [_iv(addr, loc=loc, ip=ip), _iv(parity, loc=loc, ip=ip)],
        "{\n.reg .pred p;\n"
        f"{lab}:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [$0], $1;\n"
        f"@!p bra {lab};\n}}",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bulk_g2s(smem_dst, gmem_src, nbytes: int, mbar, *, loc=None, ip=None):
    _asm_se(
        None,
        [
            _iv(smem_dst, loc=loc, ip=ip),
            _lv(gmem_src, loc=loc, ip=ip),
            _iv(nbytes, loc=loc, ip=ip),
            _iv(mbar, loc=loc, ip=ip),
        ],
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
        "r,l,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _fence_proxy_async_smem(*, loc=None, ip=None):
    _asm_se(None, [], "fence.proxy.async.shared::cta;", "", loc=loc, ip=ip)


@dsl_user_op
def _tmem_alloc(slot_addr, *, loc=None, ip=None):
    _asm_se(
        None,
        [_iv(slot_addr, loc=loc, ip=ip)],
        f"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [$0], {_TMEM_COLS};",
        "r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _tmem_dealloc(taddr, *, loc=None, ip=None):
    _asm_se(
        None,
        [_iv(taddr, loc=loc, ip=ip)],
        f"tcgen05.dealloc.cta_group::1.sync.aligned.b32 $0, {_TMEM_COLS};",
        "r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _tmem_st(taddr, values, *, loc=None, ip=None):
    """``tcgen05.st.sync.aligned.32x32b.x{4,8}.b32`` of FP32 ``values`` at column ``taddr``."""
    n = len(values)
    regs = ", ".join(f"r{i}" for i in range(n))
    moves = "".join(f"mov.b32 r{i}, ${i + 1};\n" for i in range(n))
    _asm_se(
        None,
        [_iv(taddr, loc=loc, ip=ip)] + [_fv(v, loc=loc, ip=ip) for v in values],
        "{\n.reg .b32 "
        + regs
        + ";\n"
        + moves
        + f"tcgen05.st.sync.aligned.32x32b.x{n}.b32 [$0], {{{regs}}};\n}}",
        "r," + ",".join("f" for _ in range(n)),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _tmem_wait_st(*, loc=None, ip=None):
    _asm_se(None, [], "tcgen05.wait::st.sync.aligned;", "", loc=loc, ip=ip)


@dsl_user_op
def _tmem_ld_wait(taddrs, width: int, *, loc=None, ip=None):
    """``tcgen05.ld.sync.aligned.32x32b.x{width}.b32`` at each column of ``taddrs``, then
    ``tcgen05.wait::ld``; ``len(taddrs)`` lists of ``width`` FP32 values."""
    n = len(taddrs) * width
    regs = [f"r{i}" for i in range(n)]
    body = ""
    for k in range(len(taddrs)):
        body += (
            f"tcgen05.ld.sync.aligned.32x32b.x{width}.b32 {{"
            + ", ".join(regs[k * width : (k + 1) * width])
            + f"}}, [${n + k}];\n"
        )
    body += "tcgen05.wait::ld.sync.aligned;\n"
    body += "".join(f"mov.b32 ${i}, r{i};\n" for i in range(n))
    res = _asm_se(
        llvm.StructType.get_literal([T.f32()] * n),
        [_iv(a, loc=loc, ip=ip) for a in taddrs],
        "{\n.reg .b32 " + ", ".join(regs) + ";\n" + body + "}",
        ",".join(["=f"] * n + ["r"] * len(taddrs)),
        loc=loc,
        ip=ip,
    )
    flat = _unpack(res, T.f32, Float32, n, loc=loc, ip=ip)
    return [flat[k * width : (k + 1) * width] for k in range(len(taddrs))]


# ---- shared / global memory ----


@dsl_user_op
def _lds_words(addr, count: int, *, loc=None, ip=None):
    """``count`` (1, 2 or 4) 32-bit words from shared memory."""
    if count == 1:
        return [
            Int32(
                _asm_se(
                    T.i32(),
                    [_iv(addr, loc=loc, ip=ip)],
                    "ld.shared.b32 $0, [$1];",
                    "=r,r",
                    loc=loc,
                    ip=ip,
                )
            )
        ]
    outs = ", ".join(f"${i}" for i in range(count))
    res = _asm_se(
        llvm.StructType.get_literal([T.i32()] * count),
        [_iv(addr, loc=loc, ip=ip)],
        f"ld.shared.v{count}.b32 {{{outs}}}, [${count}];",
        ",".join(["=r"] * count + ["r"]),
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32, Int32, count, loc=loc, ip=ip)


@dsl_user_op
def _lds_f32x2_if(valid, addr, *, loc=None, ip=None):
    """Two FP32 from shared memory if ``valid != 0``, else zeros."""
    res = _asm_se(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [_iv(valid, loc=loc, ip=ip), _iv(addr, loc=loc, ip=ip)],
        "{\n.reg .pred p;\nsetp.ne.u32 p, $2, 0;\nmov.b32 $0, 0f00000000;\nmov.b32 $1, 0f00000000;\n"
        "@p ld.shared.v2.f32 {$0, $1}, [$3];\n}",
        "=&f,=&f,r,r",
        loc=loc,
        ip=ip,
    )
    return tuple(_unpack(res, T.f32, Float32, 2, loc=loc, ip=ip))


@dsl_user_op
def _lds_f32_if(valid, addr, *, loc=None, ip=None):
    return Float32(
        _asm_se(
            T.f32(),
            [_iv(valid, loc=loc, ip=ip), _iv(addr, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.ne.u32 p, $1, 0;\nmov.b32 $0, 0f00000000;\n"
            "@p ld.shared.f32 $0, [$2];\n}",
            "=&f,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sts_f32x2(addr, a, b, *, loc=None, ip=None):
    _asm_se(
        None,
        [_iv(addr, loc=loc, ip=ip), _fv(a, loc=loc, ip=ip), _fv(b, loc=loc, ip=ip)],
        "st.shared.v2.f32 [$0], {$1, $2};",
        "r,f,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ldg_words(addr, count: int, *, loc=None, ip=None):
    """``count`` (2 or 4) 32-bit words from global memory."""
    outs = ", ".join(f"${i}" for i in range(count))
    res = _asm_se(
        llvm.StructType.get_literal([T.i32()] * count),
        [_lv(addr, loc=loc, ip=ip)],
        f"ld.global.v{count}.b32 {{{outs}}}, [${count}];",
        ",".join(["=r"] * count + ["l"]),
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32, Int32, count, loc=loc, ip=ip)


@dsl_user_op
def _stg_words(addr, words, *, loc=None, ip=None):
    """Store 1, 2 or 4 32-bit words to global memory."""
    n = len(words)
    if n == 1:
        text = "st.global.b32 [$0], $1;"
    else:
        text = f"st.global.v{n}.b32 [$0], {{" + ", ".join(f"${i + 1}" for i in range(n)) + "};"
    _asm_se(
        None,
        [_lv(addr, loc=loc, ip=ip)] + [_iv(w, loc=loc, ip=ip) for w in words],
        text,
        "l," + ",".join("r" for _ in range(n)),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _stg_u8_if(valid, addr, value, *, loc=None, ip=None):
    _asm_se(
        None,
        [_iv(valid, loc=loc, ip=ip), _lv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "{\n.reg .pred p;\nsetp.ne.u32 p, $0, 0;\n@p st.global.b8 [$1], $2;\n}",
        "r,l,r",
        loc=loc,
        ip=ip,
    )


# ---- MXFP8 (the catalog quantizer's IEEE instructions) ----


@dsl_user_op
def _bf16x2_absmax(words, *, loc=None, ip=None):
    """``__habs2`` of each word, folded by ``__hmax2(acc, abs)`` in order."""
    n = len(words)
    body = "abs.bf16x2 m, $1;\n"
    for i in range(1, n):
        body += f"abs.bf16x2 a, ${i + 1};\nmax.bf16x2 m, m, a;\n"
    return Int32(
        _asm(
            T.i32(),
            [_iv(w, loc=loc, ip=ip) for w in words],
            "{\n.reg .b32 m, a;\n" + body + "mov.b32 $0, m;\n}",
            "=r," + ",".join("r" for _ in range(n)),
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shfl_bfly_b32(value, offset: int, *, loc=None, ip=None):
    return Int32(
        _asm_se(
            T.i32(),
            [_iv(value, loc=loc, ip=ip)],
            f"shfl.sync.bfly.b32 $0, $1, {offset}, 0x1f, 0xffffffff;",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bf16x2_max(a, b, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(a, loc=loc, ip=ip), _iv(b, loc=loc, ip=ip)],
            "max.bf16x2 $0, $1, $2;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _mx_scale(packed_max, *, loc=None, ip=None):
    """``(UE8M0 byte, FP32 scale)`` of a 32-element block whose ``|max|`` halves are ``packed_max``.

    ``vecMax = float(__hmax(lo, hi))``; the byte ``cvt.rp.satfinite.ue8m0x2.f32(vecMax * 0x3B124925)`` (an IEEE
    multiply); the scale 0 when ``vecMax`` is +-0 (an integer test: subnormal and NaN maxima are not zero), 1 for
    byte 0, else ``exp2f(127 - byte)`` as the non-FTZ ``ex2.approx.f32`` runs: ``ex2.approx.ftz`` with the argument
    halved and the result squared below -126."""
    res = _asm(
        llvm.StructType.get_literal([T.i32(), T.f32()]),
        [_iv(packed_max, loc=loc, ip=ip)],
        "{\n"
        ".reg .b16 lo, hi, mx, e2;\n"
        ".reg .b32 v, e;\n"
        ".reg .f32 f, s, z, t, r, q, k;\n"
        ".reg .pred pz, pe, pr;\n"
        "mov.b32 {lo, hi}, $2;\n"
        "max.bf16 mx, lo, hi;\n"
        "cvt.f32.bf16 f, mx;\n"
        f"mul.rn.f32 s, f, {_RCP_448};\n"
        "mov.b32 z, 0f00000000;\n"
        "cvt.rp.satfinite.ue8m0x2.f32 e2, z, s;\n"
        "cvt.u32.u16 e, e2;\n"
        "and.b32 e, e, 255;\n"
        "mov.b32 v, f;\n"
        "and.b32 v, v, 0x7fffffff;\n"
        "setp.eq.u32 pz, v, 0;\n"
        "setp.eq.u32 pe, e, 0;\n"
        "cvt.rn.f32.u32 t, e;\n"
        "mov.b32 k, 0f42FE0000;\n"
        "sub.rn.ftz.f32 t, k, t;\n"
        "setp.geu.f32 pr, t, 0fC2FC0000;\n"
        "@!pr mul.rn.f32 t, t, 0f3F000000;\n"
        "ex2.approx.ftz.f32 r, t;\n"
        "@!pr mul.rn.f32 r, r, r;\n"
        "mov.f32 q, r;\n"
        "@pe mov.f32 q, 0f3F800000;\n"
        "@pz mov.f32 q, 0f00000000;\n"
        "mov.b32 $0, e;\n"
        "mov.f32 $1, q;\n"
        "}",
        "=r,=f,r",
        loc=loc,
        ip=ip,
    )
    return Int32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)), Float32(
        llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)
    )


@dsl_user_op
def _e4m3_codes(words, scale, *, loc=None, ip=None):
    """E4M3 codes of the bf16 values in ``words`` (2 or 4 packed words) times ``scale``: each widened,
    ``mul.rn.f32`` by the scale, ``cvt.rn.satfinite.e4m3x2.f32``; returns ``len(words) // 2`` words, value 0 in
    the lowest byte."""
    n = len(words)
    body = "".join(
        f"shl.b32 t, ${n // 2 + i}, 16;\nmov.b32 a{2 * i}, t;\n"
        f"and.b32 t, ${n // 2 + i}, 0xffff0000;\nmov.b32 a{2 * i + 1}, t;\n"
        for i in range(n)
    )
    body += "".join(f"mul.rn.f32 a{i}, a{i}, ${n // 2 + n};\n" for i in range(2 * n))
    body += "".join(
        f"cvt.rn.satfinite.e4m3x2.f32 h{i}, a{2 * i + 1}, a{2 * i};\n" for i in range(n)
    )
    body += "".join(f"mov.b32 ${i}, {{h{2 * i}, h{2 * i + 1}}};\n" for i in range(n // 2))
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * (n // 2)),
        [_iv(w, loc=loc, ip=ip) for w in words] + [_fv(scale, loc=loc, ip=ip)],
        "{\n.reg .b32 t;\n.reg .f32 "
        + ", ".join(f"a{i}" for i in range(2 * n))
        + ";\n.reg .b16 "
        + ", ".join(f"h{i}" for i in range(n))
        + ";\n"
        + body
        + "}",
        ",".join(["=r"] * (n // 2) + ["r"] * n + ["f"]),
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32, Int32, n // 2, loc=loc, ip=ip)


def _sf_offset(row, k_idx):
    """``get_sf_out_offset_128x4`` (batch 0, 224 scale columns: 56 K tiles) as an Int32 byte offset."""
    return (
        (row >> 7) * (56 * 512)
        + (k_idx >> 2) * 512
        + (row & 31) * 16
        + ((row & 127) >> 5) * 4
        + (k_idx & 3)
    )


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


class AttnResPersistentXKernel:
    """The persistent mix with ``n`` candidates and one flag set (see the module docstring).

    ``has_delta``: ``x = bf16(prefix + addend)``, stored as the updated prefix; ``delta2``: ``x = bf16(prefix +
    bf16(addend + addend2))``; ``f32_out``: also store ``float(out)``; ``mx_out``: also store the MXFP8 form.
    """

    def __init__(
        self,
        n: int,
        has_delta: bool,
        delta2: bool,
        f32_out: bool,
        mx_out: bool,
        use_pdl: bool,
    ):
        if not MIN_CANDIDATES <= n <= MAX_CANDIDATES:
            raise ValueError(f"n={n} not in [{MIN_CANDIDATES}, {MAX_CANDIDATES}]")
        if delta2 and not has_delta:
            raise ValueError("delta2 is a second addend: it needs has_delta")
        self.n = n
        self.has_delta = has_delta
        self.delta2 = delta2
        self.f32_out = f32_out
        self.mx_out = mx_out
        self.use_pdl = use_pdl
        self.n_delta = (2 if delta2 else 1) if has_delta else 0
        self.num_chunks = (n + _NC - 1) // _NC
        self.v_bytes = _CD * _NC * _ROW_BYTES
        self.delta_bytes = self.n_delta * _CD * _ROW_BYTES
        self.out_w_off = self.v_bytes + self.delta_bytes
        self.plan_off = self.out_w_off + _ROW_BYTES
        self.smem_bytes = self.plan_off + _PLAN_BYTES

    @cute.jit
    def __call__(
        self,
        block_addr: Int64,
        layer_addr: Int64,
        upd_addr: Int64,
        add_addr: Int64,
        add2_addr: Int64,
        res_w_addr: Int64,
        rms_w_addr: Int64,
        out_w_addr: Int64,
        output: cute.Tensor,
        f32_addr: Int64,
        codes_addr: Int64,
        sf_addr: Int64,
        rows: Int32,
        grid: Int32,
        rms_eps: Float32,
        out_eps: Float32,
        stream: cuda.CUstream,
    ):
        # ``output`` (bf16 [T, 7168]) is the one tensor argument: TVM-FFI takes the launch stream from its device.
        @cute.struct
        class SharedStorage:
            raw: cute.struct.Align[cute.struct.MemRange[cutlass.Int8, self.smem_bytes], 128]

        self.shared_storage = SharedStorage
        self.kernel(
            block_addr,
            layer_addr,
            upd_addr,
            add_addr,
            add2_addr,
            res_w_addr,
            rms_w_addr,
            out_w_addr,
            output.iterator.toint(),
            f32_addr,
            codes_addr,
            sf_addr,
            rows,
            rms_eps,
            out_eps,
        ).launch(
            grid=(grid, 1, 1),
            block=(_THREADS, 1, 1),
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        block_addr: Int64,
        layer_addr: Int64,
        upd_addr: Int64,
        add_addr: Int64,
        add2_addr: Int64,
        res_w_addr: Int64,
        rms_w_addr: Int64,
        out_w_addr: Int64,
        out_addr: Int64,
        f32_addr: Int64,
        codes_addr: Int64,
        sf_addr: Int64,
        rows: Int32,
        rms_eps: Float32,
        out_eps: Float32,
    ):
        n = self.n
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        num_ctas, _, _ = cute.arch.grid_dim()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        smem = storage.raw.data_ptr().toint()
        v_smem = smem
        delta_smem = smem + self.v_bytes
        out_w_smem = smem + self.out_w_off
        plan = smem + self.plan_off
        bar_ready = plan  # [CD] x 8 bytes
        bar_consumed = plan + 16  # [CD] x 8 bytes
        bar_out_w = plan + 32
        ws_stats = plan + 48  # [8 warps][3] float2
        tmem_slot = plan + 240

        if warp == 0:
            if _elect_one() != 0:
                for i in cutlass.range_constexpr(_CD):
                    _mbar_init(bar_ready + 8 * i, 1)
                    _mbar_init(bar_consumed + 8 * i, _CONSUMER_WARPS)
                _mbar_init(bar_out_w, 1)
                _fence_mbar_init()

        _griddepcontrol_wait()

        if warp == 1:
            _tmem_alloc(tmem_slot)
        _bar_sync_all()

        if warp == 0:
            if _elect_one() != 0:
                _mbar_expect_tx(bar_out_w, _ROW_BYTES)
                _bulk_g2s(out_w_smem, out_w_addr, _ROW_BYTES, bar_out_w)

        if warp == 0:
            if _elect_one() != 0:
                # ---- producer: each row's candidates, three per chunk, into the two-deep ring ----
                gci = Int32(0)
                for tb in cutlass.range(bidx, rows, num_ctas, unroll=1):
                    for ci in cutlass.range_constexpr(self.num_chunks):
                        ns = ci * _NC
                        an = min(_NC, n - ns)
                        prefix_chunk = 0 <= n - 1 - ns < an
                        slot = gci & 1
                        _mbar_wait(bar_consumed + slot * 8, ((gci >> 1) & 1) ^ 1)
                        nbytes = an * _ROW_BYTES
                        if cutlass.const_expr(self.has_delta and prefix_chunk):
                            nbytes += self.n_delta * _ROW_BYTES
                        ready = bar_ready + slot * 8
                        _mbar_expect_tx(ready, nbytes)
                        for k in cutlass.range_constexpr(an):
                            src = ns + k
                            if cutlass.const_expr(src < n - 1):
                                gaddr = (
                                    block_addr + (Int64(src) * Int64(rows) + Int64(tb)) * _ROW_BYTES
                                )
                            else:
                                gaddr = layer_addr + Int64(tb) * _ROW_BYTES
                            _bulk_g2s(
                                v_smem + (slot * _NC + k) * _ROW_BYTES, gaddr, _ROW_BYTES, ready
                            )
                        if cutlass.const_expr(self.has_delta and prefix_chunk):
                            _bulk_g2s(
                                delta_smem + slot * _ROW_BYTES,
                                add_addr + Int64(tb) * _ROW_BYTES,
                                _ROW_BYTES,
                                ready,
                            )
                            if cutlass.const_expr(self.delta2):
                                _bulk_g2s(
                                    delta_smem + (_CD + slot) * _ROW_BYTES,
                                    add2_addr + Int64(tb) * _ROW_BYTES,
                                    _ROW_BYTES,
                                    ready,
                                )
                        gci = gci + 1
        else:
            self._consumer(
                upd_addr,
                res_w_addr,
                rms_w_addr,
                out_addr,
                f32_addr,
                codes_addr,
                sf_addr,
                rows,
                rms_eps,
                out_eps,
                tidx,
                bidx,
                num_ctas,
                warp,
                lane,
                v_smem,
                delta_smem,
                out_w_smem,
                bar_ready,
                bar_consumed,
                bar_out_w,
                ws_stats,
                tmem_slot,
            )

        _bar_sync_all()
        _griddepcontrol_launch_dependents()
        if warp == 1:
            _tmem_dealloc(_lds_words(tmem_slot, 1)[0])

    @cute.jit
    def _consumer(
        self,
        upd_addr: Int64,
        res_w_addr: Int64,
        rms_w_addr: Int64,
        out_addr: Int64,
        f32_addr: Int64,
        codes_addr: Int64,
        sf_addr: Int64,
        rows: Int32,
        rms_eps: Float32,
        out_eps: Float32,
        tidx: Int32,
        bidx: Int32,
        num_ctas: Int32,
        warp: Int32,
        lane: Int32,
        v_smem: Int32,
        delta_smem: Int32,
        out_w_smem: Int32,
        bar_ready: Int32,
        bar_consumed: Int32,
        bar_out_w: Int32,
        ws_stats: Int32,
        tmem_slot: Int32,
    ):
        n = self.n
        comp_tid = tidx - 32
        comp_wid = warp - 1
        group = Int32(comp_wid >= 4)
        ct = comp_tid & 127
        # The thread's columns: tiles 2 * si + group (si < 3, 8 values) and half `group` of tile 6 (4 values).
        h_base = [(si * 2 + group) * _K_TILE + ct * _VEC for si in range(3)]
        h_base.append(6 * _K_TILE + group * (_K_TILE // 2) + ct * 4)
        widths = (8, 8, 8, 4)
        my_tmem = _lds_words(tmem_slot, 1)[0] + group * _TMEM_GROUP_COLS

        # q = rms_w * res_w (FMUL.FTZ), 28 per thread.
        q = []
        for si in cutlass.range_constexpr(_SLICES):
            nw = widths[si] // 2
            rw = _ldg_words(rms_w_addr + Int64(h_base[si]) * 2, nw)
            sw = _ldg_words(res_w_addr + Int64(h_base[si]) * 2, nw)
            for k in cutlass.range_constexpr(nw):
                r0, r1 = _bf16x2_to_f32(rw[k])
                s0, s1 = _bf16x2_to_f32(sw[k])
                q.append(_mul_ftz(r0, s0))
                q.append(_mul_ftz(r1, s1))

        zero = _const("0f00000000")
        neg_flt_max = _const(_NEG_FLT_MAX)
        stat_n = lane >> 3
        stat_w = lane & 7

        gci = Int32(0)
        for tb in cutlass.range(bidx, rows, num_ctas, unroll=1):
            row_off = Int64(tb) * _ROW_BYTES
            m_running = neg_flt_max
            s_running = zero
            acc = [zero] * _ACC
            for ci in cutlass.range_constexpr(self.num_chunks):
                ns = ci * _NC
                an = min(_NC, n - ns)
                prefix_n = n - 1 - ns
                slot = gci & 1
                _mbar_wait(bar_ready + slot * 8, (gci >> 1) & 1)

                # ---- pass A: values to TMEM, per-candidate sum of squares and dot product ----
                sq = [(zero, zero)] * _NC
                dot = [(zero, zero)] * _NC
                buf = v_smem + slot * (_NC * _ROW_BYTES)
                for si in cutlass.range_constexpr(_SLICES):
                    nw = widths[si] // 2
                    for k in cutlass.range_constexpr(an):
                        words = _lds_words(buf + k * _ROW_BYTES + h_base[si] * 2, nw)
                        if cutlass.const_expr(self.has_delta and k == prefix_n):
                            add = _lds_words(delta_smem + slot * _ROW_BYTES + h_base[si] * 2, nw)
                            if cutlass.const_expr(self.delta2):
                                add2 = _lds_words(
                                    delta_smem + (_CD + slot) * _ROW_BYTES + h_base[si] * 2, nw
                                )
                                add = [_hadd2(add[j], add2[j]) for j in range(nw)]
                            words = [_hadd2(words[j], add[j]) for j in range(nw)]
                            _stg_words(upd_addr + row_off + Int64(h_base[si]) * 2, words)
                        f = []
                        for j in cutlass.range_constexpr(nw):
                            lo, hi = _bf16x2_to_f32(words[j])
                            f.append((lo, hi))
                        _tmem_st(my_tmem + (si * _NC + k) * _VEC, [v for p in f for v in p])
                        for j in cutlass.range_constexpr(nw):
                            sq[k] = _ffma2(f[j], f[j], sq[k])
                            dot[k] = _ffma2(
                                f[j], (q[si * 8 + 2 * j], q[si * 8 + 2 * j + 1]), dot[k]
                            )

                _fence_proxy_async_smem()
                _syncwarp()
                if lane == 0:
                    _mbar_arrive(bar_consumed + slot * 8)
                _tmem_wait_st()

                # ---- statistics: warp butterflies, then the 8 warp partials of each candidate ----
                pairs = []
                for k in cutlass.range_constexpr(_NC):
                    if cutlass.const_expr(k < an):
                        pairs.append((_add_ftz(sq[k][0], sq[k][1]), _add_ftz(dot[k][0], dot[k][1])))
                    else:
                        pairs.append((zero, zero))
                for off in cutlass.range_constexpr(5):
                    offset = 16 >> off
                    for k in cutlass.range_constexpr(_NC):
                        other = (_shfl_bfly(pairs[k][0], offset), _shfl_bfly(pairs[k][1], offset))
                        pairs[k] = _fadd2(pairs[k], other)
                if lane == 0:
                    for k in cutlass.range_constexpr(_NC):
                        _sts_f32x2(ws_stats + (comp_wid * _NC + k) * 8, pairs[k][0], pairs[k][1])
                _bar_sync_consumers()
                tx, ty = _lds_f32x2_if(Int32(stat_n < _NC), ws_stats + (stat_w * _NC + stat_n) * 8)
                _bar_sync_consumers()
                for off in cutlass.range_constexpr(3):
                    offset = 4 >> off
                    tx = _add_ftz(tx, _shfl_down8(tx, offset))
                    ty = _add_ftz(ty, _shfl_down8(ty, offset))
                # Lanes 0, 8, 16 hold candidates 0, 1, 2 (the other lanes' values are not read).
                local_logit = _mul_ftz(ty, _rsigma(tx, rms_eps))
                logits = [_shfl_idx(local_logit, k * _CONSUMER_WARPS) for k in range(an)]

                # ---- online softmax over the chunk ----
                m_chunk = neg_flt_max
                for k in cutlass.range_constexpr(an):
                    m_chunk = _max_ftz(m_chunk, logits[k])
                m_new = _max_ftz(m_running, m_chunk)
                corr = _exp_diff(m_running, m_new)
                w = []
                w_sum = zero
                for k in cutlass.range_constexpr(an):
                    w.append(_exp_diff(logits[k], m_new))
                    w_sum = _add_ftz(w_sum, w[k])

                # ---- pass B: acc = acc * corr + sum_k w_k v_k (values from TMEM) ----
                for si in cutlass.range_constexpr(_SLICES):
                    nw = widths[si] // 2
                    a = [
                        _fmul2((acc[si * 8 + 2 * j], acc[si * 8 + 2 * j + 1]), (corr, corr))
                        for j in range(nw)
                    ]
                    vals = _tmem_ld_wait(
                        [my_tmem + (si * _NC + k) * _VEC for k in range(an)], widths[si]
                    )
                    for k in cutlass.range_constexpr(an):
                        for j in cutlass.range_constexpr(nw):
                            a[j] = _ffma2((w[k], w[k]), (vals[k][2 * j], vals[k][2 * j + 1]), a[j])
                    for j in cutlass.range_constexpr(nw):
                        acc[si * 8 + 2 * j] = a[j][0]
                        acc[si * 8 + 2 * j + 1] = a[j][1]
                s_running = _fma_ftz(corr, s_running, w_sum)
                m_running = m_new
                gci = gci + 1

            # ---- output norm: sum of squares of acc, s folded into the epsilon ----
            osq2 = (zero, zero)
            for si in cutlass.range_constexpr(_SLICES):
                for j in cutlass.range_constexpr(widths[si] // 2):
                    p = (acc[si * 8 + 2 * j], acc[si * 8 + 2 * j + 1])
                    osq2 = _ffma2(p, p, osq2)
            if tb == bidx:
                _mbar_wait(bar_out_w, 0)
            osq = _add_ftz(osq2[0], osq2[1])
            for off in cutlass.range_constexpr(5):
                osq = _add_ftz(osq, _shfl_bfly(osq, 16 >> off))
            if lane == 0:
                _sts_f32x2(ws_stats + comp_wid * (_NC * 8), osq, zero)
            _bar_sync_consumers()
            total = _lds_f32_if(Int32(lane < _CONSUMER_WARPS), ws_stats + lane * (_NC * 8))
            _bar_sync_consumers()
            for off in cutlass.range_constexpr(3):
                total = _add_ftz(total, _shfl_down8(total, 4 >> off))
            out_rsigma = _shfl_idx(_out_rsigma(total, out_eps, s_running), 0)

            # ---- outputs ----
            for si in cutlass.range_constexpr(_SLICES):
                nw = widths[si] // 2
                h = h_base[si]
                wwords = _lds_words(out_w_smem + h * 2, nw)
                packed = []
                for j in cutlass.range_constexpr(nw):
                    w0, w1 = _bf16x2_to_f32(wwords[j])
                    y0 = _mul_ftz(_mul_ftz(acc[si * 8 + 2 * j], out_rsigma), w0)
                    y1 = _mul_ftz(_mul_ftz(acc[si * 8 + 2 * j + 1], out_rsigma), w1)
                    packed.append(_f32_to_bf16x2(y0, y1))
                _stg_words(out_addr + row_off + Int64(h) * 2, packed)
                if cutlass.const_expr(self.f32_out):
                    bits = []
                    for j in cutlass.range_constexpr(nw):
                        bits.extend(_bf16x2_hi_lo_f32(packed[j]))
                    f32_row = f32_addr + Int64(tb) * (HIDDEN_SIZE * 4) + Int64(h) * 4
                    _stg_words(f32_row, bits[0:4])
                    if cutlass.const_expr(nw == 4):
                        _stg_words(f32_row + 16, bits[4:8])
                if cutlass.const_expr(self.mx_out):
                    lanes = 32 // widths[si]  # lanes per 32-element block: 4 (8 wide) or 8 (4 wide)
                    lmax = _bf16x2_absmax(packed)
                    for e in cutlass.range_constexpr(lanes.bit_length() - 1):
                        lmax = _bf16x2_max(_shfl_bfly_b32(lmax, 1 << e), lmax)
                    sfv, scale = _mx_scale(lmax)
                    _stg_u8_if(
                        Int32((lane & (lanes - 1)) == 0),
                        sf_addr + Int64(_sf_offset(tb, h >> 5)),
                        sfv,
                    )
                    codes = _e4m3_codes(packed, scale)
                    _stg_words(codes_addr + Int64(tb) * HIDDEN_SIZE + Int64(h), codes)

        if cutlass.const_expr(self.mx_out):
            # The quantizer zeroes the scale bytes of the last 128-row tile's padding rows.
            pad_rows = (rows + 127) // 128 * 128
            for r in cutlass.range(rows + bidx, pad_rows, num_ctas, unroll=1):
                _stg_u8_if(
                    Int32(comp_tid < _SF_COLS),
                    sf_addr + Int64(_sf_offset(r, comp_tid)),
                    Int32(0),
                )


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


@functools.cache
def _capability_major(index: int) -> int:
    return torch.cuda.get_device_capability(index)[0]


def supported(device: torch.device | int | None = None) -> bool:
    """Whether the kernel runs on ``device`` (default: the current one): an SM 10.x GPU (tcgen05 and TMEM)."""
    if not torch.cuda.is_available():
        return False
    if device is None:
        index = torch.cuda.current_device()
    elif isinstance(device, int):
        index = device
    else:
        index = device.index if device.index is not None else torch.cuda.current_device()
    return _capability_major(index) == 10


def flag_set_supported(addend: bool, addend2: bool, f32: bool, mx: bool) -> bool:
    """The flag sets the kernel is built for (the C++ op's instantiations): none, addend, addend + addend2, FP32
    copy with or without the addend, addend + addend2 + MXFP8 form."""
    if addend2 and not addend:
        return False
    if mx:
        return addend and addend2 and not f32
    if f32:
        return not addend2
    return True


@functools.cache
def _num_sms(device_index: int) -> int:
    """The uncapped grid: one CTA per SM of the device. (The C++ launcher's opt-in SM reserve is an
    environment lever, which the family's flat path does not read; the grid size never changes a
    byte.)"""
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@functools.cache
def _default_pdl() -> bool:
    """``TRTLLM_ENABLE_PDL``, read once (as the C++ launcher's ``getEnvEnablePDL``)."""
    return bool(get_env_enable_pdl())


@functools.cache
def _compiled(n: int, has_delta: bool, delta2: bool, f32_out: bool, mx_out: bool, use_pdl: bool):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "rubin_attn_res_persistent_x compiles on the first call of a variant; make that call before CUDA "
            "graph capture"
        )
    kernel = AttnResPersistentXKernel(n, has_delta, delta2, f32_out, mx_out, use_pdl)
    addr = Int64(1 << 20)
    output = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (cute.sym_int(), HIDDEN_SIZE), stride_order=(1, 0)
    )
    return cute.compile(
        kernel,
        *([addr] * 8),
        output,
        *([addr] * 3),
        Int32(1),
        Int32(1),
        Float32(1e-5),
        Float32(1e-5),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _bad(name: str, t: torch.Tensor, what: str) -> ValueError:
    return ValueError(
        f"attn_res persistent x: {name} must be {what} (contiguous, 16-byte aligned, on the inputs' device); got "
        f"{tuple(t.shape)} {t.dtype} on {t.device}, contiguous={t.is_contiguous()}, address {t.data_ptr():#x}"
    )


def _fits(t: torch.Tensor, dtype: torch.dtype, numel: int, device: int) -> bool:
    return (
        t.dtype is dtype
        and t.numel() == numel
        and t.is_contiguous()
        and t.get_device() == device
        and t.data_ptr() % 16 == 0
    )


def run_persistent_x(
    prefix: torch.Tensor,
    addend: torch.Tensor | None,
    addend2: torch.Tensor | None,
    block: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    out_weight: torch.Tensor,
    rms_eps: float,
    out_eps: float,
    *,
    output: torch.Tensor,
    updated: torch.Tensor | None = None,
    output_f32: torch.Tensor | None = None,
    mx_codes: torch.Tensor | None = None,
    mx_scales: torch.Tensor | None = None,
    grid_cap: bool = False,
    use_pdl: bool | None = None,
) -> None:
    """Launch on the current stream into the given outputs.

    Inputs (contiguous, 16-byte aligned, on one SM 10.x device): ``prefix`` bf16 ``[T, 7168]`` (``T >= 1``),
    ``addend`` / ``addend2`` bf16 ``[T, 7168]`` or None (``addend2`` needs ``addend``), ``block`` bf16
    ``[N - 1, T, 7168]`` with ``2 <= N <= 9``, the weights bf16 ``[7168]``. Outputs: ``output`` bf16 ``[T, 7168]``;
    ``updated`` bf16 ``[T, 7168]`` with an addend (not ``prefix``'s memory); ``output_f32`` fp32 ``[T, 7168]`` for
    the FP32 copy; ``mx_codes`` (1-byte ``[T, 7168]``) and ``mx_scales`` (1-byte ``[pad128(T) * 224]``) for the
    MXFP8 form. The flag set must be one of :func:`flag_set_supported`'s. ``grid_cap``: ``min(SMs, T)`` CTAs
    instead of one per SM. ``use_pdl``: default ``TRTLLM_ENABLE_PDL``. Compiles once per variant (``N`` and
    flag set) on its first call, which must not be under CUDA graph capture.
    """
    if prefix.dim() != 2 or prefix.shape[1] != HIDDEN_SIZE or prefix.shape[0] < 1:
        raise ValueError(
            f"attn_res persistent x: prefix must be [T >= 1, {HIDDEN_SIZE}], got {tuple(prefix.shape)}"
        )
    rows = prefix.shape[0]
    n = block.shape[0] + 1 if block.dim() == 3 else 0
    if not MIN_CANDIDATES <= n <= MAX_CANDIDATES or block.shape[1:] != prefix.shape:
        raise ValueError(
            f"attn_res persistent x: block must be [N - 1, T, {HIDDEN_SIZE}] with {MIN_CANDIDATES} <= N <= "
            f"{MAX_CANDIDATES} and T = {rows}, got {tuple(block.shape)}"
        )
    f32, mx = output_f32 is not None, mx_codes is not None
    if (
        not flag_set_supported(addend is not None, addend2 is not None, f32, mx)
        or (mx_scales is None) == mx
    ):
        raise ValueError(
            "attn_res persistent x: unsupported flag set (addend="
            f"{addend is not None}, addend2={addend2 is not None}, f32={f32}, mx={mx}, mx_scales="
            f"{mx_scales is not None})"
        )
    dev = prefix.get_device()
    if dev < 0 or _capability_major(dev) != 10:
        raise RuntimeError("attn_res persistent x needs an SM 10.x GPU")
    if rows * HIDDEN_SIZE * (n - 1) >= 2**31:
        raise ValueError(
            f"attn_res persistent x: {rows} rows x {n - 1} bank rows exceed 32-bit element offsets"
        )
    bf16 = torch.bfloat16
    numel = rows * HIDDEN_SIZE
    checks = [
        ("prefix", prefix, bf16, numel),
        ("block", block, bf16, numel * (n - 1)),
        ("res_weight", res_weight, bf16, HIDDEN_SIZE),
        ("rms_weight", rms_weight, bf16, HIDDEN_SIZE),
        ("out_weight", out_weight, bf16, HIDDEN_SIZE),
        ("output", output, bf16, numel),
    ]
    if addend is not None:
        if updated is None or updated.data_ptr() == prefix.data_ptr():
            raise ValueError(
                "attn_res persistent x: an addend needs `updated`, not prefix's memory"
            )
        checks += [("addend", addend, bf16, numel), ("updated", updated, bf16, numel)]
    if addend2 is not None:
        checks.append(("addend2", addend2, bf16, numel))
    if f32:
        checks.append(("output_f32", output_f32, torch.float32, numel))
    if mx:
        checks.append(("mx_codes", mx_codes, mx_codes.dtype, numel))
        if mx_codes.element_size() != 1 or mx_scales.element_size() != 1:
            raise ValueError("attn_res persistent x: mx_codes and mx_scales hold bytes")
        if (
            mx_scales.numel() != (rows + 127) // 128 * 128 * _SF_COLS
            or not mx_scales.is_contiguous()
            or mx_scales.get_device() != dev
        ):
            raise _bad("mx_scales", mx_scales, "[pad128(T) * 224] bytes")
    for name, t, dtype, count in checks:
        if not _fits(t, dtype, count, dev):
            raise _bad(name, t, f"{dtype} with {count} elements")
    if use_pdl is None:
        use_pdl = _default_pdl()
    num_sms = _num_sms(dev)
    _compiled(n, addend is not None, addend2 is not None, f32, mx, bool(use_pdl))(
        block.data_ptr(),
        prefix.data_ptr(),
        updated.data_ptr() if addend is not None else 0,
        addend.data_ptr() if addend is not None else 0,
        addend2.data_ptr() if addend2 is not None else 0,
        res_weight.data_ptr(),
        rms_weight.data_ptr(),
        out_weight.data_ptr(),
        output,
        output_f32.data_ptr() if f32 else 0,
        mx_codes.data_ptr() if mx else 0,
        mx_scales.data_ptr() if mx else 0,
        rows,
        min(num_sms, rows) if grid_cap else num_sms,
        float(rms_eps),
        float(out_eps),
    )


def attn_res_add_rmsnorm_persistent_x(
    prefix: torch.Tensor,
    addend: torch.Tensor | None,
    addend2: torch.Tensor | None,
    block: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    out_weight: torch.Tensor,
    rms_eps: float,
    out_eps: float,
    f32: bool = False,
    mx: bool = False,
    grid_cap: bool = False,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
]:
    """``(updated, output, output_f32, codes, scales)`` on ``prefix`` ``[T, 7168]``, ``block`` ``[N - 1, T, 7168]``
    and the mix's weights (see :func:`run_persistent_x`): new tensors, except that without an addend ``updated``
    IS ``prefix``. The epilogues not asked for are None; ``codes`` is float8_e4m3fn ``[T, 7168]``, ``scales`` uint8
    ``[pad128(T) * 224]`` (the 128x4 swizzled layout, padding rows zero)."""
    rows = prefix.shape[0]
    output = torch.empty_like(prefix)
    updated = torch.empty_like(prefix) if addend is not None else None
    output_f32 = (
        torch.empty(prefix.shape, dtype=torch.float32, device=prefix.device) if f32 else None
    )
    codes = scales = None
    if mx:
        codes = torch.empty(prefix.shape, dtype=torch.float8_e4m3fn, device=prefix.device)
        scales = torch.empty(
            (rows + 127) // 128 * 128 * _SF_COLS, dtype=torch.uint8, device=prefix.device
        )
    run_persistent_x(
        prefix,
        addend,
        addend2,
        block,
        res_weight,
        rms_weight,
        out_weight,
        rms_eps,
        out_eps,
        output=output,
        updated=updated,
        output_f32=output_f32,
        mx_codes=codes,
        mx_scales=scales,
        grid_cap=grid_cap,
    )
    return (prefix if updated is None else updated), output, output_f32, codes, scales


__all__ = [
    "HIDDEN_SIZE",
    "MAX_CANDIDATES",
    "MIN_CANDIDATES",
    "AttnResPersistentXKernel",
    "attn_res_add_rmsnorm_persistent_x",
    "flag_set_supported",
    "run_persistent_x",
    "supported",
]
