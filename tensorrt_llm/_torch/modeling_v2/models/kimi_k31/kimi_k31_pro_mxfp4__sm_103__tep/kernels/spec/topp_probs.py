# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Top-p target probabilities of the sampled DFlash2 verify (CuTe DSL): one thread-block cluster per row.

``probs[r] = softmax(logits[r] / T[r])`` restricted to the smallest descending prefix whose mass reaches
``top_p[r]`` (the token that crosses ``top_p`` included) and renormalized; zeros elsewhere. A row with
``top_p >= 1`` gets the plain temperature softmax. No top-k, no min-p (advanced sampling mode NO_TOPK).

One launch; one cluster of ``C`` CTAs per row (8 x 512 threads by default), each CTA owning a contiguous
``vocab / C`` slice that one bulk copy stages in shared memory. The slice stays there until the output
store, so the logits are read once and the probabilities written once; every phase runs on all C CTAs.

1. Local max ``m_c`` of the slice; ``R_c = ceil(m_c * k)`` with ``k = log2(e) / T``; each weight
   ``w_c = ex2.approx(x * k - R_c)`` replaces its logit in shared memory. Weights at or above a floor
   ``F`` (a power of two <= (1 - top_p) / (2 vocab), so everything below it weighs less than
   (1 - top_p) of the row and the cut never falls there) go into an exact histogram: bucket = the
   weight's bits 29..20 (1/8 binade); mass = the 24-bit mantissa added as two 12-bit halves with native
   u32 shared atomics, so a bucket's mass is an exact integer in units of its binade. Weights below
   ``F`` go to a float sum.
2. Cluster barrier 1. Every CTA reads every peer's ``R_c``, floor sum and 128 binade totals over DSMEM
   and combines them in rank order, so every CTA computes bit-identical values. ``R = max R_c``; a
   CTA's weights move to the row's scale by the exact exponent shift ``R - R_c`` (whole binades, so
   bucket integers add exactly). The binade masses (double, one warp) give ``S``,
   ``target = top_p * S`` and the crossing binade; that binade's 8 buckets, read from every peer, give
   the crossing bucket in exact integers.
3. Every CTA pushes its keys in that bucket (the candidates) and their count into every peer's shared
   memory (``st.shared::cluster``); cluster barrier 2. No DSMEM read happens after it.
4. Each CTA alone, identically: an exact 1024-bucket histogram of the candidates' bits 19..10 and a
   block scan give the crossing sub-bucket; the candidates left in it (at most 32) are ranked by one
   warp (more than 32: one more 10-bit histogram). Result: the cut weight ``t`` (an exact key) and
   the kept mass ``sum(w >= t)`` as an exact integer, turned into ``1 / kept`` once.
   When some CTA has more than ``CAND_TOTAL / C`` candidates (very flat rows), the whole cluster
   takes two cluster-wide radix passes (bits 19..10, then 9..0) instead: same result, bit for bit.
5. ``p = w * (1 / kept)`` for ``w >= t``, else 0, stored straight from shared memory.

Tie rule: every token whose weight equals the cut weight is kept (the engine's fused sampler's rule;
``filtered_probs`` keeps tied tokens in id order until the mass is reached). Equal logits in one CTA's
slice get equal weights; in different slices (different ``R_c``) their weights can differ in the last
bit, so a tie spanning slices can split at the cut. The weights come from
``ex2.approx`` as the engine's ``__expf`` does, so the distribution is the reference's, not bit-equal
to it or to the engine: a token sitting on the top-p cut can land on either side.

Deterministic: the histograms are integers and every float reduction runs in a fixed order, so a row's
output is bit-identical run to run, with or without PDL, and on either candidate path. It depends on
the cluster size (the slices set ``R_c``), not on the other rows of the call. Needs fp32 logits with a
16-byte-aligned row-contiguous layout and ``vocab % (cluster * threads * 4) == 0``; the slice must fit
shared memory (``vocab / cluster * 4`` bytes plus ~33 KB). The first call per configuration compiles
(outside CUDA-graph capture); the launch allocates nothing and never syncs the host.
"""

from __future__ import annotations

import functools
import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Float32, Float64, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

LOG2E = 1.4426950408889634
NB = 1024  # buckets per radix pass (10 bits)
NG1 = 128  # binades of pass 1 (8 buckets each)
NG = 32  # groups of passes 2 and 3 (32 buckets each)
CAND_TOTAL = 2048  # candidates a CTA holds: cand_total / cluster from each peer
F_MIN_BITS = 27 << 23  # 2^-100: lowest floor

# ---------------------------------------------------------------------------
# Inline PTX helpers
# ---------------------------------------------------------------------------


def _asm(res, ops, text, cons, *, loc=None, ip=None, se=True):
    return llvm.inline_asm(
        res,
        ops,
        text,
        cons,
        has_side_effects=se,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _iv(v, *, loc=None, ip=None):
    return Int32(v).ir_value(loc=loc, ip=ip)


def _lv(v, *, loc=None, ip=None):
    return Int64(v).ir_value(loc=loc, ip=ip)


def _fv(v, *, loc=None, ip=None):
    return Float32(v).ir_value(loc=loc, ip=ip)


def _dv(v, *, loc=None, ip=None):
    return Float64(v).ir_value(loc=loc, ip=ip)


def _f32(text, *args, loc=None, ip=None):
    cons = "=f," + ",".join("f" for _ in args)
    return Float32(
        _asm(T.f32(), [_fv(a, loc=loc, ip=ip) for a in args], text, cons, loc=loc, ip=ip, se=False)
    )


@dsl_user_op
def ex2(x, *, loc=None, ip=None):
    return _f32("ex2.approx.ftz.f32 $0, $1;", x, loc=loc, ip=ip)


@dsl_user_op
def fma_rn(a, b, c, *, loc=None, ip=None):
    return _f32("fma.rn.f32 $0, $1, $2, $3;", a, b, c, loc=loc, ip=ip)


@dsl_user_op
def mul_rn(a, b, *, loc=None, ip=None):
    return _f32("mul.rn.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def add_rn(a, b, *, loc=None, ip=None):
    return _f32("add.rn.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def fmax(a, b, *, loc=None, ip=None):
    return _f32("max.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def rcp_rn(a, *, loc=None, ip=None):
    return _f32("div.rn.f32 $0, 0f3F800000, $1;", a, loc=loc, ip=ip)


@dsl_user_op
def ceil_s32(x, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_fv(x, loc=loc, ip=ip)],
            "cvt.rpi.s32.f32 $0, $1;",
            "=r,f",
            loc=loc,
            ip=ip,
            se=False,
        )
    )


@dsl_user_op
def s32_f32(x, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_iv(x, loc=loc, ip=ip)],
            "cvt.rn.f32.s32 $0, $1;",
            "=f,r",
            loc=loc,
            ip=ip,
            se=False,
        )
    )


@dsl_user_op
def bits_of(x, *, loc=None, ip=None):
    return Int32(
        _asm(T.i32(), [_fv(x, loc=loc, ip=ip)], "mov.b32 $0, $1;", "=r,f", loc=loc, ip=ip, se=False)
    )


@dsl_user_op
def f32_of(x, *, loc=None, ip=None):
    return Float32(
        _asm(T.f32(), [_iv(x, loc=loc, ip=ip)], "mov.b32 $0, $1;", "=f,r", loc=loc, ip=ip, se=False)
    )


@dsl_user_op
def u64_f64(x, *, loc=None, ip=None):
    return Float64(
        _asm(
            T.f64(),
            [_lv(x, loc=loc, ip=ip)],
            "cvt.rn.f64.u64 $0, $1;",
            "=d,l",
            loc=loc,
            ip=ip,
            se=False,
        )
    )


@dsl_user_op
def f32_to_f64(x, *, loc=None, ip=None):
    return Float64(
        _asm(
            T.f64(),
            [_fv(x, loc=loc, ip=ip)],
            "cvt.f64.f32 $0, $1;",
            "=d,f",
            loc=loc,
            ip=ip,
            se=False,
        )
    )


@dsl_user_op
def f64_to_f32(x, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_dv(x, loc=loc, ip=ip)],
            "cvt.rn.f32.f64 $0, $1;",
            "=f,d",
            loc=loc,
            ip=ip,
            se=False,
        )
    )


@dsl_user_op
def dadd(a, b, *, loc=None, ip=None):
    return Float64(
        _asm(
            T.f64(),
            [_dv(a, loc=loc, ip=ip), _dv(b, loc=loc, ip=ip)],
            "add.rn.f64 $0, $1, $2;",
            "=d,d,d",
            loc=loc,
            ip=ip,
            se=False,
        )
    )


@dsl_user_op
def dmul(a, b, *, loc=None, ip=None):
    return Float64(
        _asm(
            T.f64(),
            [_dv(a, loc=loc, ip=ip), _dv(b, loc=loc, ip=ip)],
            "mul.rn.f64 $0, $1, $2;",
            "=d,d,d",
            loc=loc,
            ip=ip,
            se=False,
        )
    )


@dsl_user_op
def dsetp_ge(a, b, *, loc=None, ip=None):
    """1 if a >= b else 0."""
    return Int32(
        _asm(
            T.i32(),
            [_dv(a, loc=loc, ip=ip), _dv(b, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.ge.f64 p, $1, $2;\nselp.u32 $0, 1, 0, p;\n}",
            "=r,d,d",
            loc=loc,
            ip=ip,
            se=False,
        )
    )


@dsl_user_op
def dsetp_lt(a, b, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_dv(a, loc=loc, ip=ip), _dv(b, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.lt.f64 p, $1, $2;\nselp.u32 $0, 1, 0, p;\n}",
            "=r,d,d",
            loc=loc,
            ip=ip,
            se=False,
        )
    )


@dsl_user_op
def unit_f64(e, *, loc=None, ip=None):
    """2^(e - 150) as a double: the mass of mantissa integer 1 in binade (exponent field) e."""
    return Float64(
        _asm(
            T.f64(),
            [_iv(e, loc=loc, ip=ip)],
            "{\n.reg .b64 t;\n.reg .b32 x;\nadd.s32 x, $1, 873;\ncvt.u64.u32 t, x;\nshl.b64 t, t, 52;\n"
            "mov.b64 $0, t;\n}",
            "=d,r",
            loc=loc,
            ip=ip,
            se=False,
        )
    )


@dsl_user_op
def lds_v4(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [_iv(addr, loc=loc, ip=ip)],
        "ld.shared.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def sts_v4(addr, a, b, c, d, *, loc=None, ip=None):
    _asm(
        None,
        [
            _iv(addr, loc=loc, ip=ip),
            _iv(a, loc=loc, ip=ip),
            _iv(b, loc=loc, ip=ip),
            _iv(c, loc=loc, ip=ip),
            _iv(d, loc=loc, ip=ip),
        ],
        "st.shared.v4.u32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def lds_u32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(), [_iv(addr, loc=loc, ip=ip)], "ld.shared.u32 $0, [$1];", "=r,r", loc=loc, ip=ip
        )
    )


@dsl_user_op
def sts_u32(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _iv(v, loc=loc, ip=ip)],
        "st.shared.u32 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def lds_f32(addr, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(), [_iv(addr, loc=loc, ip=ip)], "ld.shared.f32 $0, [$1];", "=f,r", loc=loc, ip=ip
        )
    )


@dsl_user_op
def sts_f32(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _fv(v, loc=loc, ip=ip)],
        "st.shared.f32 [$0], $1;",
        "r,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def lds_u64(addr, *, loc=None, ip=None):
    return Int64(
        _asm(
            T.i64(), [_iv(addr, loc=loc, ip=ip)], "ld.shared.u64 $0, [$1];", "=l,r", loc=loc, ip=ip
        )
    )


@dsl_user_op
def sts_u64(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _lv(v, loc=loc, ip=ip)],
        "st.shared.u64 [$0], $1;",
        "r,l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def lds_f64(addr, *, loc=None, ip=None):
    return Float64(
        _asm(
            T.f64(), [_iv(addr, loc=loc, ip=ip)], "ld.shared.f64 $0, [$1];", "=d,r", loc=loc, ip=ip
        )
    )


@dsl_user_op
def sts_f64(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _dv(v, loc=loc, ip=ip)],
        "st.shared.f64 [$0], $1;",
        "r,d",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_add_if(pred, addr, v, *, loc=None, ip=None):
    """``atomicAdd(smem u32 [addr], v)`` where ``pred != 0``."""
    _asm(
        None,
        [_iv(pred, loc=loc, ip=ip), _iv(addr, loc=loc, ip=ip), _iv(v, loc=loc, ip=ip)],
        "{\n.reg .pred p;\nsetp.ne.u32 p, $0, 0;\n@p red.shared.add.u32 [$1], $2;\n}",
        "r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ldc_u64(addr, rank, *, loc=None, ip=None):
    """u64 at ``addr`` in the shared memory of cluster CTA ``rank``."""
    return Int64(
        _asm(
            T.i64(),
            [_iv(addr, loc=loc, ip=ip), _iv(rank, loc=loc, ip=ip)],
            "{\n.reg .b32 a;\nmapa.shared::cluster.u32 a, $1, $2;\nld.shared::cluster.u64 $0, [a];\n}",
            "=l,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ldc_u32(addr, rank, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(addr, loc=loc, ip=ip), _iv(rank, loc=loc, ip=ip)],
            "{\n.reg .b32 a;\nmapa.shared::cluster.u32 a, $1, $2;\nld.shared::cluster.u32 $0, [a];\n}",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cluster_sync(*, loc=None, ip=None):
    _asm(
        None,
        [],
        "barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;",
        "",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cluster_arrive(*, loc=None, ip=None):
    _asm(None, [], "barrier.cluster.arrive.release.aligned;", "", loc=loc, ip=ip)


@dsl_user_op
def cluster_wait(*, loc=None, ip=None):
    _asm(None, [], "barrier.cluster.wait.acquire.aligned;", "", loc=loc, ip=ip)


@dsl_user_op
def bar_sync(*, loc=None, ip=None):
    _asm(None, [], "bar.sync 0;", "", loc=loc, ip=ip)


@dsl_user_op
def mbar_init(addr, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip)],
        "mbarrier.init.shared::cta.b64 [$0], 1;\nfence.mbarrier_init.release.cluster;",
        "r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def bulk_load(dst, src, mbar, nbytes: int, *, loc=None, ip=None):
    """Arm ``mbar`` for ``nbytes`` and bulk-copy ``nbytes`` from global ``src`` to shared ``dst``."""
    _asm(
        None,
        [_iv(dst, loc=loc, ip=ip), _lv(src, loc=loc, ip=ip), _iv(mbar, loc=loc, ip=ip)],
        f"mbarrier.arrive.expect_tx.shared::cta.b64 _, [$2], {nbytes};\n"
        f"cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], {nbytes}, [$2];",
        "r,l,r",
        loc=loc,
        ip=ip,
    )


_WAIT = [0]


@dsl_user_op
def mbar_wait0(addr, *, loc=None, ip=None):
    _WAIT[0] += 1
    lab = f"K31TOPP_W{_WAIT[0]}"
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip)],
        "{\n.reg .pred p;\n"
        f"{lab}:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [$0], 0;\n"
        f"@!p bra {lab};\n}}",
        "r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ldg_f32(addr, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(), [_lv(addr, loc=loc, ip=ip)], "ld.global.f32 $0, [$1];", "=f,l", loc=loc, ip=ip
        )
    )


@dsl_user_op
def stg_v4(addr, a, b, c, d, *, loc=None, ip=None):
    _asm(
        None,
        [
            _lv(addr, loc=loc, ip=ip),
            _fv(a, loc=loc, ip=ip),
            _fv(b, loc=loc, ip=ip),
            _fv(c, loc=loc, ip=ip),
            _fv(d, loc=loc, ip=ip),
        ],
        "st.global.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,f,f,f,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def shfl_bfly_f32(v, off: int, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_fv(v, loc=loc, ip=ip)],
            f"shfl.sync.bfly.b32 $0, $1, {off}, 0x1f, 0xffffffff;",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def shfl_up_f64(v, off: int, *, loc=None, ip=None):
    return Float64(
        _asm(
            T.f64(),
            [_dv(v, loc=loc, ip=ip)],
            "{\n.reg .b32 lo, hi;\nmov.b64 {lo, hi}, $1;\n"
            f"shfl.sync.up.b32 lo, lo, {off}, 0x0, 0xffffffff;\n"
            f"shfl.sync.up.b32 hi, hi, {off}, 0x0, 0xffffffff;\n"
            "mov.b64 $0, {lo, hi};\n}",
            "=d,d",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def shfl_idx_f64(v, lane: int, *, loc=None, ip=None):
    return Float64(
        _asm(
            T.f64(),
            [_dv(v, loc=loc, ip=ip)],
            "{\n.reg .b32 lo, hi;\nmov.b64 {lo, hi}, $1;\n"
            f"shfl.sync.idx.b32 lo, lo, {lane}, 0x1f, 0xffffffff;\n"
            f"shfl.sync.idx.b32 hi, hi, {lane}, 0x1f, 0xffffffff;\n"
            "mov.b64 $0, {lo, hi};\n}",
            "=d,d",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ballot(pred, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(pred, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.ne.u32 p, $1, 0;\nvote.sync.ballot.b32 $0, p, 0xffffffff;\n}",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def lowest_bit(x, *, loc=None, ip=None):
    """Index of the lowest set bit of a nonzero word."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(x, loc=loc, ip=ip)],
            "{\n.reg .b32 t;\nbrev.b32 t, $1;\nclz.b32 $0, t;\n}",
            "=r,r",
            loc=loc,
            ip=ip,
            se=False,
        )
    )


@dsl_user_op
def highest_bit(x, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(), [_iv(x, loc=loc, ip=ip)], "bfind.u32 $0, $1;", "=r,r", loc=loc, ip=ip, se=False
        )
    )


@dsl_user_op
def shfl_up_u64(v, off: int, *, loc=None, ip=None):
    return Int64(
        _asm(
            T.i64(),
            [_lv(v, loc=loc, ip=ip)],
            "{\n.reg .b32 lo, hi;\nmov.b64 {lo, hi}, $1;\n"
            f"shfl.sync.up.b32 lo, lo, {off}, 0x0, 0xffffffff;\n"
            f"shfl.sync.up.b32 hi, hi, {off}, 0x0, 0xffffffff;\n"
            "mov.b64 $0, {lo, hi};\n}",
            "=l,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def shfl_bfly_u64(v, off: int, *, loc=None, ip=None):
    return Int64(
        _asm(
            T.i64(),
            [_lv(v, loc=loc, ip=ip)],
            "{\n.reg .b32 lo, hi;\nmov.b64 {lo, hi}, $1;\n"
            f"shfl.sync.bfly.b32 lo, lo, {off}, 0x1f, 0xffffffff;\n"
            f"shfl.sync.bfly.b32 hi, hi, {off}, 0x1f, 0xffffffff;\n"
            "mov.b64 $0, {lo, hi};\n}",
            "=l,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def shfl_up_u32(v, off: int, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(v, loc=loc, ip=ip)],
            f"shfl.sync.up.b32 $0, $1, {off}, 0x0, 0xffffffff;",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def stc_u32(addr, rank, v, *, loc=None, ip=None):
    """u32 store to ``addr`` in the shared memory of cluster CTA ``rank``."""
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _iv(rank, loc=loc, ip=ip), _iv(v, loc=loc, ip=ip)],
        "{\n.reg .b32 a;\nmapa.shared::cluster.u32 a, $0, $1;\nst.shared::cluster.u32 [a], $2;\n}",
        "r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_min_u32(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _iv(v, loc=loc, ip=ip)],
        "red.shared.min.u32 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ddiv(a, b, *, loc=None, ip=None):
    return Float64(
        _asm(
            T.f64(),
            [_dv(a, loc=loc, ip=ip), _dv(b, loc=loc, ip=ip)],
            "div.rn.f64 $0, $1, $2;",
            "=d,d,d",
            loc=loc,
            ip=ip,
            se=False,
        )
    )


@dsl_user_op
def ceil_s64(x, *, loc=None, ip=None):
    return Int64(
        _asm(
            T.i64(),
            [_dv(x, loc=loc, ip=ip)],
            "cvt.rpi.s64.f64 $0, $1;",
            "=l,d",
            loc=loc,
            ip=ip,
            se=False,
        )
    )


@dsl_user_op
def shfl_lane_u64(v, lane, *, loc=None, ip=None):
    return Int64(
        _asm(
            T.i64(),
            [_lv(v, loc=loc, ip=ip), _iv(lane, loc=loc, ip=ip)],
            "{\n.reg .b32 lo, hi;\nmov.b64 {lo, hi}, $1;\n"
            "shfl.sync.idx.b32 lo, lo, $2, 0x1f, 0xffffffff;\nshfl.sync.idx.b32 hi, hi, $2, 0x1f, 0xffffffff;\n"
            "mov.b64 $0, {lo, hi};\n}",
            "=l,l,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def atom_add_u32(addr, v, *, loc=None, ip=None):
    """``atomicAdd(smem u32 [addr], v)``, returning the old value."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(addr, loc=loc, ip=ip), _iv(v, loc=loc, ip=ip)],
            "atom.shared.add.u32 $0, [$1], $2;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def shfl_idx_u32(v, lane: int, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(v, loc=loc, ip=ip)],
            f"shfl.sync.idx.b32 $0, $1, {lane}, 0x1f, 0xffffffff;",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def shfl_lane_u32(v, lane, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(v, loc=loc, ip=ip), _iv(lane, loc=loc, ip=ip)],
            "shfl.sync.idx.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def shfl_idx_u64(v, lane: int, *, loc=None, ip=None):
    return Int64(
        _asm(
            T.i64(),
            [_lv(v, loc=loc, ip=ip)],
            "{\n.reg .b32 lo, hi;\nmov.b64 {lo, hi}, $1;\n"
            f"shfl.sync.idx.b32 lo, lo, {lane}, 0x1f, 0xffffffff;\n"
            f"shfl.sync.idx.b32 hi, hi, {lane}, 0x1f, 0xffffffff;\n"
            "mov.b64 $0, {lo, hi};\n}",
            "=l,l",
            loc=loc,
            ip=ip,
        )
    )


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


class TopPProbsKernel:
    """One cluster of ``cluster`` CTAs (``threads`` threads each) per row; see the module docstring.

    ``enable_pdl``: launch with programmatic dependent launch; everything that does not read the
    previous grid's outputs (barrier init, histogram zeroing) runs before ``griddepcontrol.wait``. The
    kernel does not trigger its dependents early: they launch when it completes. (An early trigger lets
    the next grid's CTAs land beside these, two 111 KB CTAs per SM, and measured 3-5 us slower per call
    at 10-15 rows back to back.)
    """

    def __init__(
        self,
        vocab: int,
        cluster: int,
        threads: int,
        enable_pdl: bool,
        cand_total: int = CAND_TOTAL,
    ):
        if vocab % (cluster * threads * 4):
            raise ValueError(f"vocab {vocab} is not a whole number of float4s per thread")
        if threads not in (256, 512, 1024) or cluster not in (2, 4, 8, 16):
            raise ValueError(f"cluster {cluster} threads {threads}")
        self.V, self.C, self.NT = vocab, cluster, threads
        self.enable_pdl = enable_pdl
        self.slice = vocab // cluster
        self.e4 = self.slice // (4 * threads)
        self.warps = threads // 32
        self.h = threads // NG1  # peer slices of the pass-1 binade merge
        self.capc = cand_total // cluster  # candidates one CTA may push
        self.cand_total = cand_total
        off = 0

        def take(nbytes, align=16):
            nonlocal off
            off = (off + align - 1) // align * align
            o = off
            off += nbytes
            return o

        self.o_data = take(self.slice * 4, 1024)
        # Pass-1 buckets ({hi, lo} u32 while built, then u64); later the local pass on bits 19..10.
        self.o_bufA = take(NB * 8)
        self.o_grpA = take(NG1 * 8)  # pass-1 binade totals (u64); slow pass-3 group totals
        self.o_bufB = take(NB * 8)  # own candidate list before the push; local pass b; slow pass 2
        self.o_grpB = take(NG * 8)
        self.o_part = take(max(self.h * NG1, cluster * NG, cluster * 8) * 8)
        self.o_mass = take(NG1 * 8)  # merged binade masses (f64)
        self.o_stat = take(8)  # this CTA's (R_c, floor sum)
        self.o_peer = take(cluster * 8)  # every peer's (R_c, floor sum)
        self.o_red = take(2 * 32 * 4)
        self.o_wsum = take(32 * 8)
        self.o_state = take(128)
        self.o_cand = take(cand_total * 4)  # pushed candidates [source rank][slot]
        self.o_cnt = take(cluster * 4)  # pushed candidate counts [source rank]
        self.o_mbar = take(8, 8)
        self.smem_bytes = (off + 1023) // 1024 * 1024

    @cute.jit
    def __call__(
        self,
        mLogits: cute.Tensor,
        mOut: cute.Tensor,
        mTemps: cute.Tensor,
        mTopP: cute.Tensor,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            buf: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, self.smem_bytes // 4], 1024]

        self.shared_storage = SharedStorage
        self.kernel(mLogits, mOut, mTemps, mTopP).launch(
            grid=(mLogits.shape[0] * self.C, 1, 1),
            block=(self.NT, 1, 1),
            cluster=(self.C, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    # -- block-level helpers ------------------------------------------------------------------

    @cute.jit
    def block_max(self, v, red, warp, lane):
        for off in (16, 8, 4, 2, 1):
            v = fmax(v, shfl_bfly_f32(v, off))
        if lane == 0:
            sts_f32(red + warp * 4, v)
        bar_sync()
        r = lds_f32(red)
        for w in cutlass.range_constexpr(1, self.warps):
            r = fmax(r, lds_f32(red + w * 4))
        return r

    @cute.jit
    def block_sum(self, v, red, warp, lane):
        for off in (16, 8, 4, 2, 1):
            v = add_rn(v, shfl_bfly_f32(v, off))
        if lane == 0:
            sts_f32(red + warp * 4, v)
        bar_sync()
        r = lds_f32(red)
        for w in cutlass.range_constexpr(1, self.warps):
            r = add_rn(r, lds_f32(red + w * 4))
        return r

    @cute.jit
    def to_u64(self, buf, tidx):
        """{hi, lo} u32 bucket pairs -> u64 ``hi * 4096 + lo`` in place."""
        for j in cutlass.range_constexpr(NB // self.NT):
            d = tidx + j * self.NT
            hi = lds_u32(buf + d * 8)
            lo = lds_u32(buf + d * 8 + 4)
            sts_u64(buf + d * 8, Int64(hi) * Int64(4096) + Int64(lo))

    @cute.jit
    def zero(self, buf, nwords: cutlass.Constexpr[int], tidx):
        for j in cutlass.range_constexpr((nwords + self.NT - 1) // self.NT):
            i = tidx + j * self.NT
            if cutlass.const_expr((j + 1) * self.NT <= nwords):
                sts_u32(buf + i * 4, Int32(0))
            else:
                if i < nwords:
                    sts_u32(buf + i * 4, Int32(0))

    @cute.jit
    def shifted(self, b, shift):
        """The row-scale weight bits of local weight bits ``b`` (0 when it underflows)."""
        k = b - (shift << 23)
        return cutlass.select_((b >> 23) > shift, k, Int32(0))

    @cute.jit
    def local_hist(
        self,
        smem,
        buf,
        shift,
        prefix,
        pshift: cutlass.Constexpr[int],
        dshift: cutlass.Constexpr[int],
        tidx,
    ):
        """Exact-mass histogram of digit ``(key >> dshift) & 1023`` over keys with ``key >> pshift == prefix``."""
        for i in cutlass.range_constexpr(self.e4):
            v = lds_v4(smem + self.o_data + (tidx + i * self.NT) * 16)
            for j in cutlass.range_constexpr(4):
                key = self.shifted(v[j], shift)
                hit = cutlass.select_((key >> pshift) == prefix, Int32(1), Int32(0))
                d = (key >> dshift) & (NB - 1)
                mant = (key & 0x7FFFFF) | 0x800000
                red_add_if(hit, buf + d * 8, mant >> 12)
                red_add_if(hit, buf + d * 8 + 4, mant & 0xFFF)

    @cute.jit
    def cand_hist(
        self,
        smem,
        buf,
        prefix,
        pshift: cutlass.Constexpr[int],
        dshift: cutlass.Constexpr[int],
        tidx,
    ):
        """``local_hist`` over the pushed candidates."""
        n: cutlass.Constexpr = self.cand_total
        lg: cutlass.Constexpr = self.capc.bit_length() - 1
        for k in cutlass.range_constexpr((n + self.NT - 1) // self.NT):
            i = tidx + k * self.NT
            q = i >> lg
            s = i & (self.capc - 1)
            if s < lds_u32(smem + self.o_cnt + q * 4):
                key = lds_u32(smem + self.o_cand + i * 4)
                hit = cutlass.select_((key >> pshift) == prefix, Int32(1), Int32(0))
                d = (key >> dshift) & (NB - 1)
                mant = (key & 0x7FFFFF) | 0x800000
                red_add_if(hit, buf + d * 8, mant >> 12)
                red_add_if(hit, buf + d * 8 + 4, mant & 0xFFF)

    @cute.jit
    def block_cross(self, smem, buf, rem, tidx, warp, lane):
        """The crossing bucket of 1024 u64 buckets in ``buf`` (one binade, exact integer masses), scanned from
        bucket 1023 down: the largest bucket whose suffix mass reaches ``rem`` (>= 1), else the lowest nonempty
        one. Returns (bucket, mass above it, its mass), uniform over the CTA."""
        bpt: cutlass.Constexpr = NB // self.NT
        state = smem + self.o_state
        wsum = smem + self.o_wsum
        ms = [Int64(0)] * bpt
        loc = Int64(0)
        for r in cutlass.range_constexpr(bpt):
            ms[r] = lds_u64(buf + (NB - 1 - bpt * tidx - r) * 8)
            loc = loc + ms[r]
        incl = loc
        for off in (1, 2, 4, 8, 16):
            up = shfl_up_u64(incl, off)
            if lane >= off:
                incl = incl + up
        if lane == 31:
            sts_u64(wsum + warp * 8, incl)
        if tidx == 0:
            sts_u32(state + 72, Int32(0))
            sts_u32(state + 76, Int32(NB))
        bar_sync()
        pre = Int64(0)
        for w in cutlass.range_constexpr(self.warps - 1):
            pre = pre + cutlass.select_(warp > w, lds_u64(wsum + w * 8), Int64(0))
        run = pre + incl - loc
        for r in cutlass.range_constexpr(bpt):
            if ms[r] > Int64(0):
                red_min_u32(state + 76, NB - 1 - bpt * tidx - r)
                if run < rem:
                    if run + ms[r] >= rem:
                        sts_u32(state + 72, Int32(1))
                        sts_u32(state + 96, NB - 1 - bpt * tidx - r)
                        sts_u64(state + 80, run)
                        sts_u64(state + 88, ms[r])
            run = run + ms[r]
        bar_sync()
        if lds_u32(state + 72) == 0:
            low = lds_u32(state + 76)
            run2 = pre + incl - loc
            for r in cutlass.range_constexpr(bpt):
                if NB - 1 - bpt * tidx - r == low:
                    sts_u32(state + 96, low)
                    sts_u64(state + 80, run2)
                    sts_u64(state + 88, ms[r])
                run2 = run2 + ms[r]
            bar_sync()
        idx = lds_u32(state + 96)
        above = lds_u64(state + 80)
        mass = lds_u64(state + 88)
        bar_sync()
        return idx, above, mass

    @cute.jit
    def gather_last(self, smem, prefix, tidx):
        """Up to 32 pushed candidates with ``key >> 10 == prefix`` into part[], counted at state + 112."""
        n: cutlass.Constexpr = self.cand_total
        lg: cutlass.Constexpr = self.capc.bit_length() - 1
        for k in cutlass.range_constexpr((n + self.NT - 1) // self.NT):
            i = tidx + k * self.NT
            q = i >> lg
            s = i & (self.capc - 1)
            if s < lds_u32(smem + self.o_cnt + q * 4):
                key = lds_u32(smem + self.o_cand + i * 4)
                if (key >> 10) == prefix:
                    slot = atom_add_u32(smem + self.o_state + 112, Int32(1))
                    if slot < 32:
                        sts_u32(smem + self.o_part + slot * 4, key)

    @cute.jit
    def warp_rank(self, smem, n, rem, lane):
        """Warp 0, n <= 32 keys in part[]: the crossing key (the largest key whose mass of keys >= it reaches
        ``rem``; else the smallest key) and the mass of the keys >= it. Writes state + 116 (key), + 120 (mass)."""
        valid = lane < n
        key = Int32(0)
        if valid:
            key = lds_u32(smem + self.o_part + lane * 4)
        mant = cutlass.select_(valid, Int64((key & 0x7FFFFF) | 0x800000), Int64(0))
        above = Int64(0)
        eq = Int64(0)
        tot = Int64(0)
        for j in cutlass.range_constexpr(32):
            kj = shfl_idx_u32(key, j)
            mj = shfl_idx_u64(mant, j)
            above = above + cutlass.select_(kj > key, mj, Int64(0))
            eq = eq + cutlass.select_(kj == key, mj, Int64(0))
            tot = tot + mj
        incl = above + eq
        vi = cutlass.select_(valid, Int32(1), Int32(0))
        hit = (
            vi
            & cutlass.select_(above < rem, Int32(1), Int32(0))
            & cutlass.select_(incl >= rem, Int32(1), Int32(0))
        )
        bal = ballot(hit)
        if bal == 0:
            bal = ballot(vi & cutlass.select_(incl == tot, Int32(1), Int32(0)))
        win = lowest_bit(bal)
        tkey = shfl_lane_u32(key, win)
        tinc = shfl_lane_u64(incl, win)
        if lane == 0:
            sts_u32(smem + self.o_state + 116, tkey)
            sts_u64(smem + self.o_state + 120, tinc)

    @cute.jit
    def group32(self, buf, grp, tidx, lane):
        """u64 totals of 32 groups of 32 buckets (256 threads: 8 a group, 4 buckets each)."""
        if tidx < 256:
            g = tidx >> 3
            s = Int64(0)
            for r in cutlass.range_constexpr(4):
                s = s + lds_u64(buf + (g * 32 + (tidx & 7) * 4 + r) * 8)
            for off in (4, 2, 1):
                s = s + shfl_bfly_u64(s, off)
            if (tidx & 7) == 0:
                sts_u64(grp + g * 8, s)

    @cute.jit
    def merge32(self, smem, src, base_idx, tidx):
        """part[p][i] = peer p's u64 at ``src + 8 * (base_idx + i)``, i < 32, every peer p."""
        n: cutlass.Constexpr = self.C * 32
        for j in cutlass.range_constexpr((n + self.NT - 1) // self.NT):
            i = tidx + j * self.NT
            if i < n:
                p = i >> 5
                q = i & 31
                sts_u64(smem + self.o_part + i * 8, ldc_u64(src + (base_idx + q) * 8, p))

    @cute.jit
    def cross32(self, smem, rem, lane):
        """Warp 0: the crossing among 32 merged entries part[p][i] (summed over peers p; exact integers of one
        binade); lane l takes entry 31 - l, so lanes run in descending order: the largest entry whose suffix
        mass reaches ``rem``, else the lowest nonempty one. Returns (entry, mass above it, its mass), uniform."""
        idx = 31 - lane
        s = Int64(0)
        for pp in cutlass.range_constexpr(self.C):
            s = s + lds_u64(smem + self.o_part + (pp * 32 + idx) * 8)
        incl = s
        for off in (1, 2, 4, 8, 16):
            up = shfl_up_u64(incl, off)
            if lane >= off:
                incl = incl + up
        excl = incl - s
        nonempty = cutlass.select_(s > Int64(0), Int32(1), Int32(0))
        hit = cutlass.select_(excl < rem, Int32(1), Int32(0)) & cutlass.select_(
            incl >= rem, Int32(1), Int32(0)
        )
        bal = ballot(hit & nonempty)
        nz = ballot(nonempty)
        win = Int32(31)
        if bal != 0:
            win = lowest_bit(bal)
        else:
            if nz != 0:
                win = highest_bit(nz)
        return 31 - win, shfl_lane_u64(excl, win), shfl_lane_u64(s, win)

    @cute.jit
    def slow_pass(
        self,
        smem,
        buf,
        grp,
        shift,
        prefix,
        pshift: cutlass.Constexpr[int],
        dshift: cutlass.Constexpr[int],
        rem,
        tidx,
        warp,
        lane,
    ):
        """One cluster-wide radix pass (digit ``(key >> dshift) & 1023`` of the keys with ``key >> pshift ==
        prefix``): local exact histogram, a cluster barrier, then 32 group totals and the crossing group's 32
        buckets read from every peer. Returns (digit, mass above it, its mass) in integers of the binade."""
        state = smem + self.o_state
        self.zero(buf, NB * 2, tidx)
        bar_sync()
        self.local_hist(smem, buf, shift, prefix, pshift, dshift, tidx)
        bar_sync()
        self.to_u64(buf, tidx)
        bar_sync()
        self.group32(buf, grp, tidx, lane)
        cluster_sync()
        self.merge32(smem, grp, Int32(0), tidx)
        bar_sync()
        if warp == 0:
            g, ga, gm = self.cross32(smem, rem, lane)
            if lane == 0:
                sts_u32(state + 8, g)
                sts_u64(state + 104, ga)
        bar_sync()
        gs = lds_u32(state + 8)
        gabove = lds_u64(state + 104)
        bar_sync()
        self.merge32(smem, buf, gs * 32, tidx)
        bar_sync()
        if warp == 0:
            b, ba, bm = self.cross32(smem, rem - gabove, lane)
            if lane == 0:
                sts_u32(state + 12, gs * 32 + b)
                sts_u64(state + 80, gabove + ba)
                sts_u64(state + 88, bm)
        bar_sync()
        dig = lds_u32(state + 12)
        above = lds_u64(state + 80)
        mass = lds_u64(state + 88)
        bar_sync()
        return dig, above, mass

    @cute.kernel
    def kernel(
        self, mLogits: cute.Tensor, mOut: cute.Tensor, mTemps: cute.Tensor, mTopP: cute.Tensor
    ):
        C, NT, V = self.C, self.NT, self.V
        # Raw global addresses: the slices move by bulk copy and 16-byte vector stores.
        logits = Int64(mLogits.iterator.toint())
        out = Int64(mOut.iterator.toint())
        temps = Int64(mTemps.iterator.toint())
        top_ps = Int64(mTopP.iterator.toint())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        smem = storage.buf.data_ptr().toint()
        data = smem + self.o_data
        bufA = smem + self.o_bufA
        bufB = smem + self.o_bufB
        grpA = smem + self.o_grpA
        red = smem + self.o_red
        state = smem + self.o_state
        peer = smem + self.o_peer
        part = smem + self.o_part
        mass = smem + self.o_mass
        mbar = smem + self.o_mbar

        row = bidx // C
        rank = bidx - row * C
        gbase = Int64(row) * Int64(V * 4) + Int64(rank * self.slice * 4)

        if tidx == 0:
            mbar_init(mbar)
        self.zero(bufA, NB * 2, tidx)
        if cutlass.const_expr(self.enable_pdl):
            # Everything below reads the previous grid's outputs (logits, temperatures, top-p).
            cute.arch.griddepcontrol_wait()
        if tidx == 0:
            bulk_load(data, logits + gbase, mbar, self.slice * 4)

        temp = ldg_f32(temps + Int64(row) * Int64(4))
        top_p = ldg_f32(top_ps + Int64(row) * Int64(4))
        tinv = Float32(1.0e6)
        if temp > Float32(0.0):
            tinv = rcp_rn(temp)
        kf = mul_rn(tinv, Float32(LOG2E))
        need = top_p < Float32(1.0)
        fbits = Int32(0x7F800000)  # +inf: nothing histogrammed
        if need:
            fl = mul_rn(add_rn(Float32(1.0), -top_p), Float32(0.5 / V))
            fbits = cutlass.max(bits_of(fl) & 0x7F800000, Int32(F_MIN_BITS))
        floor_w = f32_of(fbits)
        bar_sync()
        mbar_wait0(mbar)

        # 1. local max, weights, exact histogram above the floor.
        mloc = Float32(-math.inf)
        for i in cutlass.range_constexpr(self.e4):
            v = lds_v4(data + (tidx + i * NT) * 16)
            for j in cutlass.range_constexpr(4):
                mloc = fmax(mloc, f32_of(v[j]))
        mc = self.block_max(mloc, red, warp, lane)
        rc = cutlass.max(ceil_s32(mul_rn(mc, kf)), Int32(-(1 << 24)))
        rcf = -s32_f32(rc)
        sfl = Float32(0.0)
        for i in cutlass.range_constexpr(self.e4):
            a = data + (tidx + i * NT) * 16
            v = lds_v4(a)
            kb = [Int32(0)] * 4
            for j in cutlass.range_constexpr(4):
                w = ex2(fma_rn(f32_of(v[j]), kf, rcf))
                b = bits_of(w)
                kb[j] = b
                hit = cutlass.select_(w >= floor_w, Int32(1), Int32(0))
                d = b >> 20
                mant = (b & 0x7FFFFF) | 0x800000
                red_add_if(hit, bufA + d * 8, mant >> 12)
                red_add_if(hit, bufA + d * 8 + 4, mant & 0xFFF)
                sfl = add_rn(sfl, cutlass.select_(hit != 0, Float32(0.0), w))
            sts_v4(a, kb[0], kb[1], kb[2], kb[3])
        sfc = self.block_sum(sfl, red + 128, warp, lane)  # its bar.sync also closes the atomics
        self.to_u64(bufA, tidx)
        bar_sync()
        if tidx < NG1:
            gs = Int64(0)
            for j in cutlass.range_constexpr(8):
                gs = gs + lds_u64(bufA + (tidx * 8 + j) * 8)
            sts_u64(grpA + tidx * 8, gs)
        if tidx == 0:
            sts_u32(smem + self.o_stat, rc)
            sts_f32(smem + self.o_stat + 4, sfc)
        cluster_sync()  # barrier 1

        # 2. row scale, sum, crossing binade and bucket.
        if tidx < C:
            sts_u32(peer + tidx * 8, ldc_u32(smem + self.o_stat, tidx))
            sts_u32(peer + tidx * 8 + 4, ldc_u32(smem + self.o_stat + 4, tidx))
        bar_sync()
        rmax = lds_u32(peer)
        for pp in cutlass.range_constexpr(1, C):
            rmax = cutlass.max(rmax, lds_u32(peer + pp * 8))
        sfloor = Float32(0.0)
        for pp in cutlass.range_constexpr(C):
            ex = 127 + lds_u32(peer + pp * 8) - rmax
            fscale = f32_of(cutlass.select_(ex > 0, ex << 23, Int32(0)))
            sfloor = add_rn(sfloor, mul_rn(lds_f32(peer + pp * 8 + 4), fscale))
        shift = rmax - rc
        thr = Int32(0)
        scale = rcp_rn(sfloor)  # a row without the filter: the plain softmax
        if need:
            # Binade totals of the row: part[q][e] summed over peers q, q + h, ...
            eg = tidx & (NG1 - 1)
            qg = tidx >> 7
            acc = Int64(0)
            if cutlass.const_expr(self.h <= C):
                for jj in cutlass.range_constexpr(C // self.h):
                    pr = qg + jj * self.h
                    srcg = eg + rmax - lds_u32(peer + pr * 8)
                    if srcg < NG1:
                        acc = acc + ldc_u64(grpA + srcg * 8, pr)
            else:
                if qg < C:
                    srcg = eg + rmax - lds_u32(peer + qg * 8)
                    if srcg < NG1:
                        acc = acc + ldc_u64(grpA + srcg * 8, qg)
            sts_u64(part + tidx * 8, acc)
            bar_sync()
            if tidx < NG1:
                gt = lds_u64(part + tidx * 8)
                for jj in cutlass.range_constexpr(1, min(self.h, C)):
                    gt = gt + lds_u64(part + (jj * NG1 + tidx) * 8)
                sts_f64(mass + tidx * 8, dmul(u64_f64(gt), unit_f64(tidx)))
            bar_sync()
            if warp == 0:
                ms = [Float64(0.0)] * 4
                lsum = Float64(0.0)
                for r in cutlass.range_constexpr(4):
                    ms[r] = lds_f64(mass + (127 - 4 * lane - r) * 8)
                    lsum = dadd(lsum, ms[r])
                incl_scan = lsum
                for off in (1, 2, 4, 8, 16):
                    lup = shfl_up_f64(incl_scan, off)
                    if lane >= off:
                        incl_scan = dadd(incl_scan, lup)
                shist = shfl_idx_f64(incl_scan, 31)
                stot = dadd(shist, f32_to_f64(sfloor))
                tgt = dmul(f32_to_f64(top_p), stot)
                if dsetp_lt(tgt, Float64(1.0e-300)) != 0:
                    tgt = Float64(1.0e-300)
                lrun = dadd(incl_scan, -lsum)
                hit_e = Int32(-1)
                hit_above = Float64(0.0)
                low_e = Int32(-1)
                low_above = Float64(0.0)
                for r in cutlass.range_constexpr(4):
                    cross = dsetp_lt(lrun, tgt) & dsetp_ge(dadd(lrun, ms[r]), tgt)
                    if cross != 0:
                        hit_e = 127 - 4 * lane - r
                        hit_above = lrun
                    if dsetp_lt(Float64(0.0), ms[r]) != 0:
                        low_e = 127 - 4 * lane - r
                        low_above = lrun
                    lrun = dadd(lrun, ms[r])
                bal = ballot(cutlass.select_(hit_e >= 0, Int32(1), Int32(0)))
                win = Int32(0)
                if bal != 0:
                    win = lowest_bit(bal)
                else:
                    # Rounding left the target past the total: the lowest nonempty binade.
                    win = highest_bit(ballot(cutlass.select_(low_e >= 0, Int32(1), Int32(0))))
                    hit_e = low_e
                    hit_above = low_above
                if lane == win:
                    sts_u32(state, hit_e)
                    sts_f64(state + 16, hit_above)
                    sts_f64(state + 24, tgt)
                    sts_f64(state + 64, stot)
            bar_sync()
            e1 = lds_u32(state)
            base1 = lds_f64(state + 16)
            target = lds_f64(state + 24)
            unit = unit_f64(e1)
            # The 8 buckets of binade e1 from every peer: part[p][j]; lanes j < 8 sum them.
            if tidx < C * 8:
                pb = tidx >> 3
                jb = tidx & 7
                srcb = e1 + rmax - lds_u32(peer + pb * 8)
                vb = Int64(0)
                if srcb < NG1:
                    vb = ldc_u64(bufA + (srcb * 8 + jb) * 8, pb)
                sts_u64(part + tidx * 8, vb)
            bar_sync()
            if tidx < 8:
                bsum = Int64(0)
                for pp in cutlass.range_constexpr(C):
                    bsum = bsum + lds_u64(part + (pp * 8 + tidx) * 8)
                sts_u64(mass + tidx * 8, bsum)
            bar_sync()
            if tidx == 0:
                # Everything below is exact integers in units of binade e1: rem1 = units still needed below
                # the binade's top.
                rem1 = ceil_s64(ddiv(dadd(target, -base1), unit))
                rem1 = cutlass.max(rem1, Int64(1))
                brun = Int64(0)
                bd = Int32(-1)
                bhi = Int64(0)
                blow = Int32(-1)
                blow_hi = Int64(0)
                for jj in cutlass.range_constexpr(8):
                    jx = 7 - jj
                    bm = lds_u64(mass + jx * 8)
                    if bm > Int64(0):
                        if bd < 0:
                            if brun + bm >= rem1:
                                bd = e1 * 8 + jx
                                bhi = brun
                        blow = e1 * 8 + jx
                        blow_hi = brun
                    brun = brun + bm
                if bd < 0:
                    bd = blow
                    bhi = blow_hi
                sts_u32(state + 4, bd)
                sts_u64(state + 32, bhi)
                sts_u64(state + 40, rem1 - bhi)
            bar_sync()
            d1 = lds_u32(state + 4)
            above1 = lds_u64(state + 32)
            rem2 = lds_u64(state + 40)

            # 3. Candidates (keys in bucket d1) pushed to every peer: count, scan, compact, push.
            ccount = Int32(0)
            for i in cutlass.range_constexpr(self.e4):
                v = lds_v4(data + (tidx + i * NT) * 16)
                for j in cutlass.range_constexpr(4):
                    key = self.shifted(v[j], shift)
                    ccount = ccount + cutlass.select_((key >> 20) == d1, Int32(1), Int32(0))
            cinc = ccount
            for off in (1, 2, 4, 8, 16):
                cup = shfl_up_u32(cinc, off)
                if lane >= off:
                    cinc = cinc + cup
            if lane == 31:
                sts_u32(red + warp * 4, cinc)
            bar_sync()
            cpre = Int32(0)
            ctot = Int32(0)
            for wq in cutlass.range_constexpr(self.warps):
                cw = lds_u32(red + wq * 4)
                cpre = cpre + cutlass.select_(warp > wq, cw, Int32(0))
                ctot = ctot + cw
            slot = cpre + cinc - ccount
            for i in cutlass.range_constexpr(self.e4):
                v = lds_v4(data + (tidx + i * NT) * 16)
                for j in cutlass.range_constexpr(4):
                    key = self.shifted(v[j], shift)
                    if (key >> 20) == d1:
                        if slot < self.capc:
                            sts_u32(bufB + slot * 4, key)
                        slot = slot + 1
            bar_sync()
            # Slot sl of this CTA's list to peer (rank + q) % C: consecutive threads store consecutive
            # slots of one peer.
            lgc: cutlass.Constexpr = self.capc.bit_length() - 1
            own_count = cutlass.min(ctot, Int32(self.capc))
            for k in cutlass.range_constexpr(self.cand_total // NT):
                ix = tidx + k * NT
                sl = ix & (self.capc - 1)
                if sl < own_count:
                    stc_u32(
                        smem + self.o_cand + (rank * self.capc + sl) * 4,
                        (rank + (ix >> lgc)) & (C - 1),
                        lds_u32(bufB + sl * 4),
                    )
            if tidx < C:
                stc_u32(smem + self.o_cnt + rank * 4, tidx, ctot)
            cluster_sync()  # barrier 2
            over = Int32(0)
            for qq in cutlass.range_constexpr(C):
                over = over | cutlass.select_(
                    lds_u32(smem + self.o_cnt + qq * 4) > self.capc, Int32(1), Int32(0)
                )
            kint = Int64(0)
            if over == 0:
                cluster_arrive()  # nothing remote is read past barrier 2
                # 4. Exact selection among the candidates, locally (every CTA, identically).
                self.zero(bufA, NB * 2, tidx)
                if tidx == 0:
                    sts_u32(state + 112, Int32(0))
                bar_sync()
                self.cand_hist(smem, bufA, d1, 20, 10, tidx)
                bar_sync()
                self.to_u64(bufA, tidx)
                bar_sync()
                d2, a2, m2 = self.block_cross(smem, bufA, rem2, tidx, warp, lane)
                self.gather_last(smem, (d1 << 10) | d2, tidx)
                bar_sync()
                nlast = lds_u32(state + 112)
                if nlast <= 32:
                    if warp == 0:
                        self.warp_rank(smem, nlast, rem2 - a2, lane)
                    bar_sync()
                    thr = lds_u32(state + 116)
                    kint = above1 + a2 + lds_u64(state + 120)
                else:
                    self.zero(bufB, NB * 2, tidx)
                    bar_sync()
                    self.cand_hist(smem, bufB, (d1 << 10) | d2, 10, 0, tidx)
                    bar_sync()
                    self.to_u64(bufB, tidx)
                    bar_sync()
                    d3, a3, m3 = self.block_cross(smem, bufB, rem2 - a2, tidx, warp, lane)
                    thr = (d1 << 20) | (d2 << 10) | d3
                    kint = above1 + a2 + a3 + m3
            else:
                d2, a2, m2 = self.slow_pass(
                    smem, bufB, smem + self.o_grpB, shift, d1, 20, 10, rem2, tidx, warp, lane
                )
                # bufA / grpA are free: every peer finished reading them before barrier 2.
                d3, a3, m3 = self.slow_pass(
                    smem, bufA, grpA, shift, (d1 << 10) | d2, 10, 0, rem2 - a2, tidx, warp, lane
                )
                thr = (d1 << 20) | (d2 << 10) | d3
                kint = above1 + a2 + a3 + m3
                cluster_arrive()
            if tidx == 0:
                sts_f32(state + 100, rcp_rn(f64_to_f32(dadd(base1, dmul(u64_f64(kint), unit)))))
            bar_sync()
            scale = lds_f32(state + 100)
        else:
            cluster_arrive()

        # 5. Probabilities.
        obase = out + gbase
        for i in cutlass.range_constexpr(self.e4):
            v = lds_v4(data + (tidx + i * NT) * 16)
            o = [Float32(0.0)] * 4
            for j in cutlass.range_constexpr(4):
                key = self.shifted(v[j], shift)
                o[j] = cutlass.select_(key >= thr, mul_rn(f32_of(key), scale), Float32(0.0))
            stg_v4(obase + Int64((tidx + i * NT) * 16), o[0], o[1], o[2], o[3])
        cluster_wait()


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------

#: Default cluster size and CTA width: 8 CTAs of 512 threads per row.
DEFAULT_CLUSTER = 8
DEFAULT_THREADS = 512
#: Largest dynamic shared memory a CTA may opt into on SM 10.x.
_MAX_SMEM = 232448


@functools.cache
def smem_bytes(vocab: int, cluster: int = DEFAULT_CLUSTER, threads: int = DEFAULT_THREADS) -> int:
    """Shared memory one CTA needs for ``vocab`` (raises on an unsupported shape)."""
    return TopPProbsKernel(vocab, cluster, threads, False).smem_bytes


@functools.cache
def supports(vocab: int, cluster: int = DEFAULT_CLUSTER, threads: int = DEFAULT_THREADS) -> bool:
    """Whether the kernel takes rows of ``vocab`` at this configuration."""
    try:
        return smem_bytes(vocab, cluster, threads) <= _MAX_SMEM
    except ValueError:
        return False


@functools.cache
def _compiled(vocab: int, cluster: int, threads: int, enable_pdl: bool):
    kernel = TopPProbsKernel(vocab, cluster, threads, enable_pdl)
    rows = cute.sym_int()
    rows_logits = cute.runtime.make_fake_compact_tensor(
        Float32, (rows, vocab), stride_order=(1, 0), assumed_align=16
    )
    rows_out = cute.runtime.make_fake_compact_tensor(
        Float32, (rows, vocab), stride_order=(1, 0), assumed_align=16
    )
    temps = cute.runtime.make_fake_compact_tensor(Float32, (cute.sym_int(),), assumed_align=4)
    top_p = cute.runtime.make_fake_compact_tensor(Float32, (cute.sym_int(),), assumed_align=4)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        rows_logits,
        rows_out,
        temps,
        top_p,
        stream,
        options="--opt-level 2 --enable-tvm-ffi",
    )


def topp_probs(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_p: torch.Tensor,
    out: torch.Tensor,
    enable_pdl: bool,
    *,
    cluster: int = DEFAULT_CLUSTER,
    threads: int = DEFAULT_THREADS,
) -> torch.Tensor:
    """``out[rows, vocab]`` = each row's top-p distribution (see the module docstring), one launch on the
    current torch stream.

    ``logits`` and ``out`` float32 ``[rows, vocab]`` row-contiguous and 16-byte aligned; ``temperatures`` and
    ``top_p`` float32 with one contiguous entry per row. ``cluster`` / ``threads`` pick the configuration
    (the result depends on ``cluster``). The first call per configuration compiles; later calls are
    graph-capture safe.
    """
    rows, vocab = logits.shape
    if rows == 0:
        return out
    _compiled(vocab, cluster, threads, bool(enable_pdl))(logits, out, temperatures, top_p)
    return out


__all__ = [
    "CAND_TOTAL",
    "DEFAULT_CLUSTER",
    "DEFAULT_THREADS",
    "TopPProbsKernel",
    "topp_probs",
    "smem_bytes",
    "supports",
]
