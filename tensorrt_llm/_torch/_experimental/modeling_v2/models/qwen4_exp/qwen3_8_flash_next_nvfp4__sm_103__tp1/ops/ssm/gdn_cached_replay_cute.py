# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GDN cached-replay verify step on warp tensor cores (CuTe DSL), a drop-in for the Triton
`_cached_replay_kernel` (`fused_recurrent_gated_delta_rule_cached_replay_update`) at this
checkpoint's geometry (H=16 q/k heads, HV=48 v heads, K=V=128,
history 16, bf16 checkpoint state, T=1..8 new tokens).

One CTA serves one (request, q/k head, split): the 3 v heads of a q/k head hold 3 x 128 = 384
state rows of 128 (K innermost), cut into 24 m-tiles of 16 rows; a CTA owns `tpc` consecutive
m-tiles and each of its warps `tpc / warps` of them (strided by the warp count).

Per m-tile a warp computes, on mma.sync m16n8k16 (bf16 in, fp32 out), with the 16 state rows as the
A operand and the T normalized keys and T scaled queries as the 8 (T <= 4) or 16 (T <= 8) B columns,
    D[v, k_t] = h0[v, :] . k_t        D[v, q_t] = h0[v, :] . q_t
The head dimension is permuted identically in A and B (lane quad c holds dims 32m+8c..32m+8c+7), so
16-byte loads give the fragments directly. A warp's first tile is loaded into registers, its other
tiles by 1-D bulk copies (TMA, mbarrier) into shared memory -- the CTA's whole state share is in
flight from its start -- with an L2 evict-first policy (the state is read once a step). The history
term is one more MMA: the history u rows (shared memory, ldmatrix.trans) times the [16 x 2T]
coefficient matrix (k_t . k_j or q_t . k_j) exp(G_t - G_j), rounded to bf16 as the Triton kernel
rounds it. The T x T forward substitution and the output combination run per state row on CUDA
cores. With `commit`, a row whose history wraps (`p + T > 16`) also writes the advanced checkpoint
`exp(g_start) h0 + sum_j exp(g_start - G_j) u_j k_j^T` over the tile it just used (a
[16 x 16] x [16 x 128] MMA, u split into a bf16 hi and lo part as the Triton commit does), so the
checkpoint is read once and written once and the engine's all-layer commit is not needed.

Given the engine's replay work items, each CTA reads its row's (row, slot, pending count, half) in
one load; otherwise slot, then pending count / half. With PDL the state and history (not written by
the previous kernel) are read before `griddepcontrol.wait`; without it the step's own small inputs
are issued first. Small batches (the launch is latency-bound there) bring every state tile in by
bulk copy as soon as the slot is known (`pf=0`: no tile held in registers through the prologue) and
take the tiles' state dots before the first barrier, while the history is still in flight
(`early_acc`). `_Replay.kernel_p` is an experimental persistent variant (slower; not used).
"""

from __future__ import annotations

from typing import Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass import const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T as _ty
from cutlass.cutlass_dsl import dsl_user_op

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

H, HV, K, V, HIST = 16, 48, 128, 128, 16
GRP = HV // H
QKV_WIDTH = 2 * H * K + HV * V
TILES = GRP * (V // 16)  # m-tiles of one (request, q/k head)
PU = 272  # shared row pitch (bytes) of 128-element bf16 rows read by ldmatrix (conflict-free)
PK = 320  # shared row pitch of the history keys read as permuted 16-byte fragments
LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453

# ---------------------------------------------------------------------------------------
# inline PTX helpers
# ---------------------------------------------------------------------------------------


def _ir(x, cls, loc, ip):
    return cls(x).ir_value(loc=loc, ip=ip)


def _asm(ret, args, text, cons, loc, ip, side=False):
    return llvm.inline_asm(
        ret,
        args,
        text,
        cons,
        has_side_effects=side,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _unpack(res, n, ty, loc, ip):
    return [
        ty(
            llvm.extractvalue(
                _ty.i32() if ty is cutlass.Int32 else _ty.f32(), res, [i], loc=loc, ip=ip
            )
        )
        for i in range(n)
    ]


def _make_ldg4(qual):
    @dsl_user_op
    def _ldg4(gaddr, *, loc=None, ip=None):
        res = _asm(
            llvm.StructType.get_literal([_ty.i32()] * 4),
            [_ir(gaddr, cutlass.Int64, loc, ip)],
            f"ld.global{qual}.v4.b32 {{$0, $1, $2, $3}}, [$4];",
            "=r,=r,=r,=r,l",
            loc,
            ip,
            side=True,
        )
        return _unpack(res, 4, cutlass.Int32, loc, ip)

    return _ldg4


@dsl_user_op
def _policy_evict_first(*, loc=None, ip=None):
    """An L2 cache policy: evict first (data read once per step)."""
    return cutlass.Int64(
        _asm(_ty.i64(), [], "createpolicy.fractional.L2::evict_first.b64 $0, 1.0;", "=l", loc, ip)
    )


@dsl_user_op
def _ldg4_hint(gaddr, pol, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([_ty.i32()] * 4),
        [_ir(gaddr, cutlass.Int64, loc, ip), _ir(pol, cutlass.Int64, loc, ip)],
        "ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {$0, $1, $2, $3}, [$4], $5;",
        "=r,=r,=r,=r,l,l",
        loc,
        ip,
        side=True,
    )
    return _unpack(res, 4, cutlass.Int32, loc, ip)


_LDG4 = {
    "plain": _make_ldg4(""),
    "na": _make_ldg4(".L1::no_allocate"),
    "nc": _make_ldg4(".nc.L1::no_allocate"),
    "nc256": _make_ldg4(".nc.L1::no_allocate.L2::256B"),
    "cs": _make_ldg4(".cs"),
    "lu": _make_ldg4(".lu"),
}
_ldg4 = _LDG4["plain"]


@dsl_user_op
def _ldg4p(gaddr, pred, *, loc=None, ip=None):
    """16 bytes, or zeros where `pred` is 0 (nothing read)."""
    res = _asm(
        llvm.StructType.get_literal([_ty.i32()] * 4),
        [_ir(gaddr, cutlass.Int64, loc, ip), _ir(pred, cutlass.Int32, loc, ip)],
        "{ .reg .pred p; setp.ne.b32 p, $5, 0; mov.b32 $0, 0; mov.b32 $1, 0; mov.b32 $2, 0; "
        "mov.b32 $3, 0; @p ld.global.v4.b32 {$0, $1, $2, $3}, [$4]; }",
        "=r,=r,=r,=r,l,r",
        loc,
        ip,
        side=True,
    )
    return _unpack(res, 4, cutlass.Int32, loc, ip)


@dsl_user_op
def _stg4(gaddr, a, b, c, d, *, loc=None, ip=None):
    _asm(
        _ty.i32(),
        [_ir(gaddr, cutlass.Int64, loc, ip)]
        + [_ir(x, cutlass.Int32, loc, ip) for x in (a, b, c, d)],
        "{ st.global.v4.b32 [$1], {$2, $3, $4, $5}; mov.u32 $0, 0; }",
        "=r,l,r,r,r,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _stg_bf16(gaddr, x, *, loc=None, ip=None):
    """One f32 rounded to bf16 and stored (2 bytes)."""
    _asm(
        _ty.i32(),
        [_ir(gaddr, cutlass.Int64, loc, ip), _ir(x, cutlass.Float32, loc, ip)],
        "{ .reg .b16 h; cvt.rn.bf16.f32 h, $2; st.global.b16 [$1], h; mov.u32 $0, 0; }",
        "=r,l,f",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _stg_f32(gaddr, x, *, loc=None, ip=None):
    _asm(
        _ty.i32(),
        [_ir(gaddr, cutlass.Int64, loc, ip), _ir(x, cutlass.Float32, loc, ip)],
        "{ st.global.f32 [$1], $2; mov.u32 $0, 0; }",
        "=r,l,f",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _ldg_i32(gaddr, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(
            _ty.i32(),
            [_ir(gaddr, cutlass.Int64, loc, ip)],
            "ld.global.b32 $0, [$1];",
            "=r,l",
            loc,
            ip,
            side=True,
        )
    )


@dsl_user_op
def _ldg_f32(gaddr, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            _ty.f32(),
            [_ir(gaddr, cutlass.Int64, loc, ip)],
            "ld.global.f32 $0, [$1];",
            "=f,l",
            loc,
            ip,
            side=True,
        )
    )


@dsl_user_op
def _ldg_f32p(gaddr, pred, *, loc=None, ip=None):
    """An f32, or 0 where `pred` is 0 (nothing read)."""
    return cutlass.Float32(
        _asm(
            _ty.f32(),
            [_ir(gaddr, cutlass.Int64, loc, ip), _ir(pred, cutlass.Int32, loc, ip)],
            "{ .reg .pred q; setp.ne.b32 q, $2, 0; mov.f32 $0, 0f00000000; @q ld.global.f32 $0, [$1]; }",
            "=f,l,r",
            loc,
            ip,
            side=True,
        )
    )


@dsl_user_op
def _ldg_b16p(gaddr, pred, *, loc=None, ip=None):
    """16 bits zero-extended (0 where `pred` is 0); converted where used, so the load does not
    stall the issuing thread."""
    return cutlass.Int32(
        _asm(
            _ty.i32(),
            [_ir(gaddr, cutlass.Int64, loc, ip), _ir(pred, cutlass.Int32, loc, ip)],
            "{ .reg .pred q; setp.ne.b32 q, $2, 0; mov.b32 $0, 0; @q ld.global.u16 $0, [$1]; }",
            "=r,l,r",
            loc,
            ip,
            side=True,
        )
    )


@dsl_user_op
def _b16_f32(x, *, loc=None, ip=None):
    """bf16 bits in the low half -> f32."""
    return cutlass.Float32(
        _asm(
            _ty.f32(),
            [_ir(x, cutlass.Int32, loc, ip)],
            "{ .reg .b32 t; shl.b32 t, $1, 16; mov.b32 $0, t; }",
            "=f,r",
            loc,
            ip,
            side=True,
        )
    )


@dsl_user_op
def _f32_bits(x, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(_ty.i32(), [_ir(x, cutlass.Float32, loc, ip)], "mov.b32 $0, $1;", "=r,f", loc, ip)
    )


@dsl_user_op
def _bits_f32(x, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(_ty.f32(), [_ir(x, cutlass.Int32, loc, ip)], "mov.b32 $0, $1;", "=f,r", loc, ip)
    )


@dsl_user_op
def _ldg_bf16p(gaddr, pred, *, loc=None, ip=None):
    """A bf16 widened to f32, or 0 where `pred` is 0 (nothing read)."""
    return cutlass.Float32(
        _asm(
            _ty.f32(),
            [_ir(gaddr, cutlass.Int64, loc, ip), _ir(pred, cutlass.Int32, loc, ip)],
            "{ .reg .pred q; .reg .b16 h; setp.ne.b32 q, $2, 0; mov.b16 h, 0; @q ld.global.b16 h, [$1]; "
            "cvt.f32.bf16 $0, h; }",
            "=f,l,r",
            loc,
            ip,
            side=True,
        )
    )


@dsl_user_op
def _ldg_bf16(gaddr, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            _ty.f32(),
            [_ir(gaddr, cutlass.Int64, loc, ip)],
            "{ .reg .b16 h; ld.global.b16 h, [$1]; cvt.f32.bf16 $0, h; }",
            "=f,l",
            loc,
            ip,
            side=True,
        )
    )


@dsl_user_op
def _cp16z(saddr, gaddr, nbytes, *, loc=None, ip=None):
    """cp.async of 16 bytes, zero-filled past `nbytes` (0 or 16)."""
    _asm(
        _ty.i32(),
        [
            _ir(saddr, cutlass.Int32, loc, ip),
            _ir(gaddr, cutlass.Int64, loc, ip),
            _ir(nbytes, cutlass.Int32, loc, ip),
        ],
        "{ cp.async.cg.shared.global [$1], [$2], 16, $3; mov.u32 $0, 0; }",
        "=r,r,l,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _cp_commit(*, loc=None, ip=None):
    _asm(_ty.i32(), [], "{ cp.async.commit_group; mov.u32 $0, 0; }", "=r", loc, ip, side=True)


@dsl_user_op
def _cp_wait_all(*, loc=None, ip=None):
    _asm(_ty.i32(), [], "{ cp.async.wait_all; mov.u32 $0, 0; }", "=r", loc, ip, side=True)


@dsl_user_op
def _sts_i32(saddr, a, *, loc=None, ip=None):
    _asm(
        _ty.i32(),
        [_ir(saddr, cutlass.Int32, loc, ip), _ir(a, cutlass.Int32, loc, ip)],
        "{ st.shared.b32 [$1], $2; mov.u32 $0, 0; }",
        "=r,r,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _sts_f32(saddr, a, *, loc=None, ip=None):
    _asm(
        _ty.i32(),
        [_ir(saddr, cutlass.Int32, loc, ip), _ir(a, cutlass.Float32, loc, ip)],
        "{ st.shared.f32 [$1], $2; mov.u32 $0, 0; }",
        "=r,r,f",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _sts2_f32(saddr, a, b, *, loc=None, ip=None):
    _asm(
        _ty.i32(),
        [
            _ir(saddr, cutlass.Int32, loc, ip),
            _ir(a, cutlass.Float32, loc, ip),
            _ir(b, cutlass.Float32, loc, ip),
        ],
        "{ st.shared.v2.f32 [$1], {$2, $3}; mov.u32 $0, 0; }",
        "=r,r,f,f",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _sts_bf16(saddr, a, *, loc=None, ip=None):
    _asm(
        _ty.i32(),
        [_ir(saddr, cutlass.Int32, loc, ip), _ir(a, cutlass.Float32, loc, ip)],
        "{ .reg .b16 h; cvt.rn.bf16.f32 h, $2; st.shared.b16 [$1], h; mov.u32 $0, 0; }",
        "=r,r,f",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _lds_i32(saddr, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(
            _ty.i32(),
            [_ir(saddr, cutlass.Int32, loc, ip)],
            "ld.shared.b32 $0, [$1];",
            "=r,r",
            loc,
            ip,
            side=True,
        )
    )


@dsl_user_op
def _lds_f32(saddr, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            _ty.f32(),
            [_ir(saddr, cutlass.Int32, loc, ip)],
            "ld.shared.f32 $0, [$1];",
            "=f,r",
            loc,
            ip,
            side=True,
        )
    )


@dsl_user_op
def _lds_bf16(saddr, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            _ty.f32(),
            [_ir(saddr, cutlass.Int32, loc, ip)],
            "{ .reg .b16 h; ld.shared.b16 h, [$1]; cvt.f32.bf16 $0, h; }",
            "=f,r",
            loc,
            ip,
            side=True,
        )
    )


@dsl_user_op
def _lds4_i32(saddr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([_ty.i32()] * 4),
        [_ir(saddr, cutlass.Int32, loc, ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc,
        ip,
        side=True,
    )
    return _unpack(res, 4, cutlass.Int32, loc, ip)


@dsl_user_op
def _lds4_f32(saddr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([_ty.f32()] * 4),
        [_ir(saddr, cutlass.Int32, loc, ip)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        loc,
        ip,
        side=True,
    )
    return _unpack(res, 4, cutlass.Float32, loc, ip)


@dsl_user_op
def _ldsm4t(saddr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([_ty.i32()] * 4),
        [_ir(saddr, cutlass.Int32, loc, ip)],
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc,
        ip,
        side=True,
    )
    return _unpack(res, 4, cutlass.Int32, loc, ip)


@dsl_user_op
def _mma(a, b, c, *, loc=None, ip=None):
    """D = A B + C on m16n8k16 (bf16 A/B, f32 C/D); a: 4 regs, b: 2 regs, c: 4 f32."""
    res = _asm(
        llvm.StructType.get_literal([_ty.f32()] * 4),
        [_ir(v, cutlass.Int32, loc, ip) for v in a]
        + [_ir(v, cutlass.Int32, loc, ip) for v in b]
        + [_ir(v, cutlass.Float32, loc, ip) for v in c],
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        loc,
        ip,
    )
    return _unpack(res, 4, cutlass.Float32, loc, ip)


@dsl_user_op
def _bf2_pack(lo, hi, *, loc=None, ip=None):
    """Two f32 -> bf16x2 (lo in the low half)."""
    return cutlass.Int32(
        _asm(
            _ty.i32(),
            [_ir(lo, cutlass.Float32, loc, ip), _ir(hi, cutlass.Float32, loc, ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _bf2_lo_v(w, *, loc=None, ip=None):
    """`_bf2_lo` pinned in program order (for prefetched words: a pure conversion may be hoisted
    up to the load and stall the issuing warp there)."""
    return cutlass.Float32(
        _asm(
            _ty.f32(),
            [_ir(w, cutlass.Int32, loc, ip)],
            "{ .reg .b16 lo, hi; mov.b32 {lo, hi}, $1; cvt.f32.bf16 $0, lo; }",
            "=f,r",
            loc,
            ip,
            side=True,
        )
    )


@dsl_user_op
def _bf2_hi_v(w, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            _ty.f32(),
            [_ir(w, cutlass.Int32, loc, ip)],
            "{ .reg .b16 lo, hi; mov.b32 {lo, hi}, $1; cvt.f32.bf16 $0, hi; }",
            "=f,r",
            loc,
            ip,
            side=True,
        )
    )


@dsl_user_op
def _bf2_lo(w, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            _ty.f32(),
            [_ir(w, cutlass.Int32, loc, ip)],
            "{ .reg .b16 lo, hi; mov.b32 {lo, hi}, $1; cvt.f32.bf16 $0, lo; }",
            "=f,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _bf2_hi(w, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            _ty.f32(),
            [_ir(w, cutlass.Int32, loc, ip)],
            "{ .reg .b16 lo, hi; mov.b32 {lo, hi}, $1; cvt.f32.bf16 $0, hi; }",
            "=f,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _ex2(x, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            _ty.f32(),
            [_ir(x, cutlass.Float32, loc, ip)],
            "ex2.approx.ftz.f32 $0, $1;",
            "=f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _lg2(x, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            _ty.f32(),
            [_ir(x, cutlass.Float32, loc, ip)],
            "lg2.approx.ftz.f32 $0, $1;",
            "=f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _div_approx(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            _ty.f32(),
            [_ir(a, cutlass.Float32, loc, ip), _ir(b, cutlass.Float32, loc, ip)],
            "div.approx.ftz.f32 $0, $1, $2;",
            "=f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _div_rn(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            _ty.f32(),
            [_ir(a, cutlass.Float32, loc, ip), _ir(b, cutlass.Float32, loc, ip)],
            "div.rn.f32 $0, $1, $2;",
            "=f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _sqrt_rn(x, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(_ty.f32(), [_ir(x, cutlass.Float32, loc, ip)], "sqrt.rn.f32 $0, $1;", "=f,f", loc, ip)
    )


@dsl_user_op
def _mbar_init(saddr, count, *, loc=None, ip=None):
    _asm(
        _ty.i32(),
        [_ir(saddr, cutlass.Int32, loc, ip), _ir(count, cutlass.Int32, loc, ip)],
        "{ mbarrier.init.shared::cta.b64 [$1], $2; mov.u32 $0, 0; }",
        "=r,r,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _fence_mbar_init(*, loc=None, ip=None):
    _asm(
        _ty.i32(),
        [],
        "{ fence.mbarrier_init.release.cluster; mov.u32 $0, 0; }",
        "=r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _bulk_g2s(saddr, gaddr, nbytes, mbar, *, loc=None, ip=None):
    """mbarrier.arrive.expect_tx + one 1-D bulk copy global -> shared of `nbytes`."""
    _asm(
        _ty.i32(),
        [
            _ir(saddr, cutlass.Int32, loc, ip),
            _ir(gaddr, cutlass.Int64, loc, ip),
            _ir(nbytes, cutlass.Int32, loc, ip),
            _ir(mbar, cutlass.Int32, loc, ip),
        ],
        "{ mbarrier.arrive.expect_tx.shared::cta.b64 _, [$4], $3; "
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2], $3, [$4]; "
        "mov.u32 $0, 0; }",
        "=r,r,l,r,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _bulk_g2s_hint(saddr, gaddr, nbytes, mbar, pol, *, loc=None, ip=None):
    """`_bulk_g2s` with an L2 cache policy."""
    _asm(
        _ty.i32(),
        [
            _ir(saddr, cutlass.Int32, loc, ip),
            _ir(gaddr, cutlass.Int64, loc, ip),
            _ir(nbytes, cutlass.Int32, loc, ip),
            _ir(mbar, cutlass.Int32, loc, ip),
            _ir(pol, cutlass.Int64, loc, ip),
        ],
        "{ mbarrier.arrive.expect_tx.shared::cta.b64 _, [$4], $3; "
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [$1], [$2], $3, [$4], $5; "
        "mov.u32 $0, 0; }",
        "=r,r,l,r,r,l",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _mbar_wait(mbar, phase, *, loc=None, ip=None):
    _asm(
        _ty.i32(),
        [_ir(mbar, cutlass.Int32, loc, ip), _ir(phase, cutlass.Int32, loc, ip)],
        "{ .reg .pred P1; LAB_WAIT: mbarrier.try_wait.parity.shared::cta.b64 P1, [$1], $2; "
        "@P1 bra DONE; bra LAB_WAIT; DONE: mov.u32 $0, 0; }",
        "=r,r,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _mbar_arrive(mbar, *, loc=None, ip=None):
    _asm(
        _ty.i32(),
        [_ir(mbar, cutlass.Int32, loc, ip)],
        "{ mbarrier.arrive.shared::cta.b64 _, [$1]; mov.u32 $0, 0; }",
        "=r,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _fence_proxy_async(*, loc=None, ip=None):
    _asm(
        _ty.i32(), [], "{ fence.proxy.async.shared::cta; mov.u32 $0, 0; }", "=r", loc, ip, side=True
    )


@dsl_user_op
def _cp_wait1(*, loc=None, ip=None):
    _asm(_ty.i32(), [], "{ cp.async.wait_group 1; mov.u32 $0, 0; }", "=r", loc, ip, side=True)


def _exp(x):
    return _ex2(x * LOG2E)


def _shfl_xor(x, m):
    return cute.arch.shuffle_sync_bfly(x, offset=m)


# ---------------------------------------------------------------------------------------
# the kernel
# ---------------------------------------------------------------------------------------


class _Replay:
    def __init__(
        self,
        *,
        T,
        tpc,
        warps,
        commit,
        pdl,
        ld="nc",
        pf=1,
        minb=1,
        l2norm=True,
        gating=True,
        g_f32=False,
        b_f32=False,
        early_trigger=True,
        lo_split=True,
        state_before_wait=True,
        smem_tiles=True,
        persistent=False,
        gmax=0,
        spec_hist=False,
        qk_first=True,
        use_wi=False,
        evict_first=True,
        early_acc=False,
    ):
        assert 1 <= T <= 8
        assert tpc in (24, 12, 8, 4, 2, 1) and tpc % warps == 0
        self.T = T
        self.TP = 4 if T <= 4 else 8
        self.NT = self.TP // 4
        self.tpc, self.W = tpc, warps
        self.tpw = tpc // warps
        self.S = TILES // tpc
        self.nhv = 3 if tpc == 24 else (2 if tpc == 12 else 1)
        self.commit, self.pdl = commit, pdl
        # pf = 0 (shared-memory tiles only): every state tile goes by bulk copy, issued as soon
        # as the slot is known, and no state tile is held in registers through the prologue
        self.ld, self.pf, self.minb = ld, max(0 if smem_tiles else 1, min(pf, self.tpw)), minb
        # the tiles' state dots before the first barrier, while the history is in flight
        self.use_early = bool(early_acc) and not persistent
        self.l2norm, self.gating, self.g_f32, self.b_f32 = l2norm, gating, g_f32, b_f32
        self.early_trigger, self.lo_split = early_trigger, lo_split
        self.state_before_wait = state_before_wait
        # tiles past the first `pf` of a warp: bulk-copied into shared memory at the start
        self.persistent, self.gmax = persistent, gmax
        if persistent:
            assert not (g_f32 or b_f32), "the persistent variant takes bf16 raw g / beta"
            self.pf, smem_tiles = 1, True
        self.nsm = self.tpw - self.pf if smem_tiles else 0
        self.spec_hist = spec_hist and not persistent
        self.qk_first = qk_first
        # (row, slot, pending count, half) per item from the engine's replay work items
        self.use_wi = use_wi and not persistent
        self.evict_first = evict_first
        nset = 2 if (persistent or self.spec_hist) else 1
        self.nset = nset
        self.RP = (2 * self.TP + 4) * 4  # per-warp scratch row pitch (bytes)
        o = 0
        tset = 2 if persistent else 1
        self.o_st = o
        o += tset * warps * self.nsm * 4096
        self.o_mb = o
        o += -(-(tset * warps * self.nsm * 8) // 16) * 16
        self.o_u = o
        o += nset * self.nhv * 16 * PU
        self.o_kh = o
        o += nset * 16 * PK
        self.o_meta = o
        o += 256 * 16 if persistent else 0
        self.o_kp = o
        o += 16 * PU if commit else 0
        self.o_v = o
        o += (2 if persistent else 1) * self.nhv * T * 256
        self.o_x = o
        o += warps * 16 * self.RP
        self.o_kk = o
        o += 16 * 8 * 4
        self.o_cf = o
        o += self.nhv * 16 * 16 * 2
        self.o_gh = o
        o += nset * self.nhv * 16 * 4
        self.o_dec = o
        o += self.nhv * 16 * 4
        self.o_g = o
        o += self.nhv * 8 * 4
        self.o_eg = o
        o += self.nhv * 8 * 4
        self.o_beta = o
        o += self.nhv * 8 * 4
        self.o_a = o
        o += self.nhv * 64 * 4
        self.o_cq = o
        o += self.nhv * 64 * 4
        self.o_egs = o
        o += 16
        self.smem = -(-o // 128) * 128 + 128

    @cute.jit
    def __call__(
        self,
        sidx_t: cute.Tensor,
        q_ptr: cutlass.Int64,
        k_ptr: cutlass.Int64,
        v_ptr: cutlass.Int64,
        q_sn: cutlass.Int64,
        q_st: cutlass.Int64,
        k_sn: cutlass.Int64,
        k_st: cutlass.Int64,
        v_sn: cutlass.Int64,
        v_st: cutlass.Int64,
        g_ptr: cutlass.Int64,
        g_st: cutlass.Int64,
        b_ptr: cutlass.Int64,
        b_st: cutlass.Int64,
        alog_ptr: cutlass.Int64,
        dtb_ptr: cutlass.Int64,
        o_ptr: cutlass.Int64,
        h_ptr: cutlass.Int64,
        h_ss: cutlass.Int64,
        pnat_ptr: cutlass.Int64,
        buf_ptr: cutlass.Int64,
        ou_ptr: cutlass.Int64,
        ok_ptr: cutlass.Int64,
        og_ptr: cutlass.Int64,
        wi_ptr: cutlass.Int64,
        n_rows: cutlass.Int32,
        scale: cutlass.Float32,
        stream,
    ):
        if const_expr(self.persistent):
            n_items = n_rows * (H * self.S)
            self.kernel_p(
                sidx_t.iterator.toint(),
                q_ptr,
                k_ptr,
                v_ptr,
                q_sn,
                q_st,
                k_sn,
                k_st,
                v_sn,
                v_st,
                g_ptr,
                g_st,
                b_ptr,
                b_st,
                alog_ptr,
                dtb_ptr,
                o_ptr,
                h_ptr,
                h_ss,
                pnat_ptr,
                buf_ptr,
                ou_ptr,
                ok_ptr,
                og_ptr,
                scale,
                n_items,
            ).launch(
                grid=[cutlass.min(n_items, self.gmax), 1, 1],
                block=[32 * self.W, 1, 1],
                smem=self.smem,
                stream=stream,
                use_pdl=self.pdl,
                min_blocks_per_mp=self.minb,
            )
        else:
            self.kernel(
                sidx_t.iterator.toint(),
                q_ptr,
                k_ptr,
                v_ptr,
                q_sn,
                q_st,
                k_sn,
                k_st,
                v_sn,
                v_st,
                g_ptr,
                g_st,
                b_ptr,
                b_st,
                alog_ptr,
                dtb_ptr,
                o_ptr,
                h_ptr,
                h_ss,
                pnat_ptr,
                buf_ptr,
                ou_ptr,
                ok_ptr,
                og_ptr,
                wi_ptr,
                scale,
            ).launch(
                grid=[n_rows * (H * self.S), 1, 1],
                block=[32 * self.W, 1, 1],
                smem=self.smem,
                stream=stream,
                use_pdl=self.pdl,
                min_blocks_per_mp=self.minb,
            )

    # -- straight-line helpers (traced inline) ------------------------------------------------

    def _tile_addr(self, h_ptr, slot64, h_ss, h, sp, warp, i, g, c):
        mt = sp * self.tpc + warp + self.W * i
        hv = h * GRP + mt // 8
        v0 = (mt % 8) * 16
        base = h_ptr + (slot64 * h_ss + cutlass.Int64(hv * V + v0) * K) * 2
        return base + cutlass.Int64(g * 256 + c * 16)

    def _qk_raw(self, nt, q_ptr, k_ptr, q_sn, q_st, k_sn, k_st, n64, h, g, c, scale, live):
        """Raw bf16 words of key / query column nt * 8 + g at this lane's head dims, the
        column's kind and its scale."""
        T, TP = self.T, self.TP
        zi = cutlass.Int32(0)
        t = g % 4 if TP == 4 else g
        valid = (t < T) & live
        tc = cutlass.min(t, T - 1)
        qa = q_ptr + (n64 * q_sn + cutlass.Int64(tc) * q_st + cutlass.Int64(h * K + c * 8)) * 2
        ka = k_ptr + (n64 * k_sn + cutlass.Int64(tc) * k_st + cutlass.Int64(h * K + c * 8)) * 2
        if TP == 4:
            is_q = g >= 4
            src = cutlass.Int64(cutlass.select_(is_q, qa, ka))
            mult = cutlass.Float32(cutlass.select_(is_q, scale, cutlass.Float32(1.0)))
        elif nt == 1:
            is_q = g >= 0
            src = qa
            mult = scale
        else:
            is_q = g < 0
            src = ka
            mult = cutlass.Float32(1.0)
        vpred = cutlass.Int32(cutlass.select_(valid, cutlass.Int32(1), zi))
        raw = []
        for jj in range(4):
            raw.extend(_ldg4p(src + jj * 64, vpred))
        return raw, is_q, mult

    def _qk_frag(self, raw, is_q, mult):
        """L2-normalized (queries also scaled) bf16 B fragment from `_qk_raw`."""
        ss = cutlass.Float32(0.0)
        fl = []
        for e in range(16):
            lo = _bf2_lo_v(raw[e])
            hi = _bf2_hi_v(raw[e])
            fl.append(lo)
            fl.append(hi)
            ss = ss + lo * lo + hi * hi
        if self.l2norm:
            ss = ss + _shfl_xor(ss, 1)
            ss = ss + _shfl_xor(ss, 2)
            inv = _div_rn(mult, _sqrt_rn(ss + 1e-6))
            return [_bf2_pack(fl[2 * e] * inv, fl[2 * e + 1] * inv) for e in range(16)]
        return [
            cutlass.Int32(
                cutlass.select_(is_q, _bf2_pack(fl[2 * e] * mult, fl[2 * e + 1] * mult), raw[e])
            )
            for e in range(16)
        ]

    def _gate_raw(self, pred, hv, n64, g_ptr, g_st, b_ptr, b_st, alog_ptr, dtb_ptr):
        """Raw g / beta of the T tokens for v head `hv` and its A_log / dt_bias (predicated)."""
        T = self.T
        zf = cutlass.Float32(0.0)
        pi = cutlass.Int32(cutlass.select_(pred, cutlass.Int32(1), cutlass.Int32(0)))
        alv, dtb = zf, zf
        if self.gating:
            alv = _ldg_f32p(alog_ptr + cutlass.Int64(hv) * 4, pi)
            dtb = _ldg_f32p(dtb_ptr + cutlass.Int64(hv) * 4, pi)
        graws, braws = [], []
        for t in range(T):
            if self.g_f32:
                graws.append(
                    _ldg_f32p(g_ptr + (n64 * T + t) * g_st * 4 + cutlass.Int64(hv) * 4, pi)
                )
            else:
                graws.append(
                    _ldg_b16p(g_ptr + (n64 * T + t) * g_st * 2 + cutlass.Int64(hv) * 2, pi)
                )
            if self.b_f32:
                braws.append(
                    _ldg_f32p(b_ptr + (n64 * T + t) * b_st * 4 + cutlass.Int64(hv) * 4, pi)
                )
            else:
                braws.append(
                    _ldg_b16p(b_ptr + (n64 * T + t) * b_st * 2 + cutlass.Int64(hv) * 2, pi)
                )
        return graws, braws, alv, dtb

    def _smem_tile(self, s_st, s_mb, i, g, c, zi):
        """This lane's A fragment words of the warp's staged tile i (after its bulk copy landed).

        Rows g and g + 8; chunk order by row parity keeps each quarter-warp's two rows on
        disjoint banks.
        """
        kslot = i - self.pf
        _mbar_wait(s_mb + kslot * 8, zi)
        tb = s_st + kslot * 4096 + g * 256 + c * 16
        godd = (g % 2) == 1
        raw = []
        for ii in range(4):
            jsw = (g % 2) ^ ii
            raw.append(_lds4_i32(tb + jsw * 64))
            raw.append(_lds4_i32(tb + 2048 + jsw * 64))
        cur = []
        for jj in range(4):
            for rr in range(2):
                for e in range(4):
                    cur.append(
                        cutlass.Int32(
                            cutlass.select_(godd, raw[2 * (jj ^ 1) + rr][e], raw[2 * jj + rr][e])
                        )
                    )
        return cur

    def _tile_dots(self, cur, bq, zf):
        """D = h0 [16 x 128] times the T keys / T scaled queries.

        Two MMA chains over the head dimension (even / odd 16-wide k-steps), summed.
        """
        NT = self.NT
        acc0 = [[zf, zf, zf, zf] for _ in range(NT)]
        acc1 = [[zf, zf, zf, zf] for _ in range(NT)]
        for jj in range(4):
            for s in range(2):
                a = (
                    cur[8 * jj + 2 * s],
                    cur[8 * jj + 4 + 2 * s],
                    cur[8 * jj + 2 * s + 1],
                    cur[8 * jj + 4 + 2 * s + 1],
                )
                for nt in range(NT):
                    bfr = (bq[nt][4 * jj + 2 * s], bq[nt][4 * jj + 2 * s + 1])
                    if s == 0:
                        acc0[nt] = _mma(a, bfr, acc0[nt])
                    else:
                        acc1[nt] = _mma(a, bfr, acc1[nt])
        return [[acc0[nt][e] + acc1[nt][e] for e in range(4)] for nt in range(NT)]

    def _g_f32(self, x):
        return x if self.g_f32 else _b16_f32(x)

    def _b_f32(self, x):
        return x if self.b_f32 else _b16_f32(x)

    def _load_tile(self, addr, pol=None):
        regs = []
        for jj in range(4):
            for off in (jj * 64, 2048 + jj * 64):
                if pol is not None:
                    regs.extend(_ldg4_hint(addr + off, pol))
                else:
                    regs.extend(_LDG4[self.ld](addr + off))
        return regs

    @cute.kernel
    def kernel(
        self,
        sidx_ptr: cutlass.Int64,
        q_ptr: cutlass.Int64,
        k_ptr: cutlass.Int64,
        v_ptr: cutlass.Int64,
        q_sn: cutlass.Int64,
        q_st: cutlass.Int64,
        k_sn: cutlass.Int64,
        k_st: cutlass.Int64,
        v_sn: cutlass.Int64,
        v_st: cutlass.Int64,
        g_ptr: cutlass.Int64,
        g_st: cutlass.Int64,
        b_ptr: cutlass.Int64,
        b_st: cutlass.Int64,
        alog_ptr: cutlass.Int64,
        dtb_ptr: cutlass.Int64,
        o_ptr: cutlass.Int64,
        h_ptr: cutlass.Int64,
        h_ss: cutlass.Int64,
        pnat_ptr: cutlass.Int64,
        buf_ptr: cutlass.Int64,
        ou_ptr: cutlass.Int64,
        ok_ptr: cutlass.Int64,
        og_ptr: cutlass.Int64,
        wi_ptr: cutlass.Int64,
        scale: cutlass.Float32,
    ):
        T: cutlass.Constexpr = self.T
        TP: cutlass.Constexpr = self.TP
        NT: cutlass.Constexpr = self.NT
        W: cutlass.Constexpr = self.W
        TPW: cutlass.Constexpr = self.tpw
        TPC: cutlass.Constexpr = self.tpc
        NHV: cutlass.Constexpr = self.nhv
        S: cutlass.Constexpr = self.S
        NTHR: cutlass.Constexpr = 32 * self.W
        RP: cutlass.Constexpr = self.RP

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx % 32
        g = lane // 4
        c = lane % 4
        sp = bidx % S
        nh = bidx // S
        h = nh % H
        if const_expr(self.use_wi):
            wi = _LDG4["plain"](wi_ptr + cutlass.Int64(nh // H) * 16)
            n = wi[0]
        else:
            n = nh // H
        n64 = cutlass.Int64(n)
        zf = cutlass.Float32(0.0)
        zi = cutlass.Int32(0)

        smem = cutlass.utils.SmemAllocator()
        base_t = smem.allocate_tensor(
            cutlass.Int8, cute.make_layout((self.smem - 128,), stride=(1,)), byte_alignment=128
        )
        sbase = cutlass.Int32(base_t.iterator.toint())
        s_u = sbase + self.o_u
        s_kh = sbase + self.o_kh
        s_kp = sbase + self.o_kp
        s_v = sbase + self.o_v
        s_x = sbase + self.o_x + warp * (16 * RP)
        s_kk = sbase + self.o_kk
        s_cf = sbase + self.o_cf
        s_gh = sbase + self.o_gh
        s_dec = sbase + self.o_dec
        s_g = sbase + self.o_g
        s_eg = sbase + self.o_eg
        s_beta = sbase + self.o_beta
        s_a = sbase + self.o_a
        s_cq = sbase + self.o_cq
        s_egs = sbase + self.o_egs

        if const_expr(self.pdl and not self.state_before_wait and not self.qk_first):
            cute.arch.griddepcontrol_wait()

        NSM: cutlass.Constexpr = self.nsm
        s_st = sbase + self.o_st + warp * (NSM * 4096)
        s_mb = sbase + self.o_mb + warp * (NSM * 8)
        if const_expr(NSM > 0):
            if lane == 0:
                for k_ in cutlass.range_constexpr(NSM):
                    _mbar_init(s_mb + k_ * 8, cutlass.Int32(1))
                _fence_mbar_init()

        hv_first = (sp * TPC) // 8
        hv0 = h * GRP + hv_first
        gi = tidx - (32 * (W - 1) if W > 1 else 0)
        gpred = (gi >= 0) & (gi < NHV)
        if const_expr(self.qk_first):
            # the step's own inputs first: they are L2-resident and would otherwise queue behind
            # the state stream
            if const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
                if const_expr(self.early_trigger):
                    cute.arch.griddepcontrol_launch_dependents()
            pre = [
                self._qk_raw(
                    nt, q_ptr, k_ptr, q_sn, q_st, k_sn, k_st, n64, h, g, c, scale, tidx >= 0
                )
                for nt in range(NT)
            ]
            gpre = self._gate_raw(
                gpred,
                hv0 + cutlass.min(cutlass.max(gi, zi), NHV - 1),
                n64,
                g_ptr,
                g_st,
                b_ptr,
                b_st,
                alog_ptr,
                dtb_ptr,
            )
            n_v0: cutlass.Constexpr = NHV * T * 16
            for it0 in cutlass.range_constexpr(-(-n_v0 // NTHR)):
                ev0 = tidx + it0 * NTHR
                if ev0 < n_v0:
                    hl0 = ev0 // (T * 16)
                    tv0 = (ev0 // 16) % T
                    qv0 = ev0 % 16
                    sv0 = (
                        v_ptr
                        + (
                            n64 * v_sn
                            + cutlass.Int64(tv0) * v_st
                            + cutlass.Int64((hv0 + hl0) * V + qv0 * 8)
                        )
                        * 2
                    )
                    _cp16z(s_v + (hl0 * T + tv0) * 256 + qv0 * 16, sv0, cutlass.Int32(16))
            _cp_commit()

        if const_expr(self.use_wi):
            slot = wi[1]
        else:
            slot = _ldg_i32(sidx_ptr + n64 * 4)
        if slot >= 0:
            slot64 = cutlass.Int64(slot)
            # ---- 1. the state tiles: bulk copies first, then the register tile -----------------
            pol = _policy_evict_first() if const_expr(self.evict_first) else None
            if const_expr(NSM > 0):
                if lane == 0:
                    for k_ in cutlass.range_constexpr(NSM):
                        if const_expr(self.evict_first):
                            _bulk_g2s_hint(
                                s_st + k_ * 4096,
                                self._tile_addr(
                                    h_ptr, slot64, h_ss, h, sp, warp, self.pf + k_, 0, 0
                                ),
                                cutlass.Int32(4096),
                                s_mb + k_ * 8,
                                pol,
                            )
                        else:
                            _bulk_g2s(
                                s_st + k_ * 4096,
                                self._tile_addr(
                                    h_ptr, slot64, h_ss, h, sp, warp, self.pf + k_, 0, 0
                                ),
                                cutlass.Int32(4096),
                                s_mb + k_ * 8,
                            )
            tiles = []
            for i in cutlass.range_constexpr(self.pf):
                tiles.append(
                    self._load_tile(self._tile_addr(h_ptr, slot64, h_ss, h, sp, warp, i, g, c), pol)
                )

            # ---- 2. history bookkeeping and the history rows into shared memory ------------
            if const_expr(self.use_wi):
                p = wi[2]
                rb = wi[3]
            else:
                p = _ldg_i32(pnat_ptr + slot64 * 4)
                rb = _ldg_i32(buf_ptr + slot64 * 4)
            wrap = (p + T) > HIST
            wb = cutlass.Int32(cutlass.select_(wrap, 1 - rb, rb))
            woff = cutlass.Int32(cutlass.select_(wrap, zi, p))
            rslot = slot64 * 2 + cutlass.Int64(rb)
            g_rd = og_ptr + rslot * (HV * HIST * 4)
            wslot = slot64 * 2 + cutlass.Int64(wb)
            u_rd = ou_ptr + rslot * (HIST * HV * V * 2)
            k_rd = ok_ptr + rslot * (HIST * H * K * 2)
            u_wr = ou_ptr + wslot * (HIST * HV * V * 2)
            k_wr = ok_ptr + wslot * (HIST * H * K * 2)
            g_wr = og_ptr + wslot * (HV * HIST * 4)

            n_u: cutlass.Constexpr = NHV * 256
            if const_expr(self.spec_hist):
                # both halves, every row, before the pending count / half are known; rows j >= p
                # are masked where they are used
                for hb in cutlass.range_constexpr(2):
                    hslot = slot64 * 2 + hb
                    u_h = ou_ptr + hslot * (HIST * HV * V * 2)
                    k_h = ok_ptr + hslot * (HIST * H * K * 2)
                    g_h = og_ptr + hslot * (HV * HIST * 4)
                    for it in cutlass.range_constexpr(-(-n_u // NTHR)):
                        e = tidx + it * NTHR
                        if e < n_u:
                            hl = e // 256
                            j = (e // 16) % 16
                            qq = e % 16
                            src = u_h + cutlass.Int64(((j * HV + hv0 + hl) * V + qq * 8) * 2)
                            _cp16z(
                                s_u + hb * (NHV * 16 * PU) + hl * (16 * PU) + j * PU + qq * 16,
                                src,
                                cutlass.Int32(16),
                            )
                    for it in cutlass.range_constexpr(-(-256 // NTHR)):
                        e = tidx + it * NTHR
                        if e < 256:
                            j = e // 16
                            qq = e % 16
                            src = k_h + cutlass.Int64(((j * H + h) * K + qq * 8) * 2)
                            _cp16z(s_kh + hb * (16 * PK) + j * PK + qq * 16, src, cutlass.Int32(16))
                    if tidx < NHV * 4:
                        hl = tidx // 4
                        qq = tidx % 4
                        src = g_h + cutlass.Int64(((hv0 + hl) * HIST + qq * 4) * 4)
                        _cp16z(s_gh + hb * (NHV * 64) + hl * 64 + qq * 16, src, cutlass.Int32(16))
            else:
                for it in cutlass.range_constexpr(-(-n_u // NTHR)):
                    e = tidx + it * NTHR
                    if e < n_u:
                        hl = e // 256
                        j = (e // 16) % 16
                        qq = e % 16
                        jc = cutlass.min(
                            j, cutlass.max(p - 1, zi)
                        )  # an in-bounds row for a zero fill
                        src = u_rd + cutlass.Int64(((jc * HV + hv0 + hl) * V + qq * 8) * 2)
                        _cp16z(
                            s_u + hl * (16 * PU) + j * PU + qq * 16,
                            src,
                            cutlass.Int32(cutlass.select_(j < p, cutlass.Int32(16), zi)),
                        )
                for it in cutlass.range_constexpr(-(-256 // NTHR)):
                    e = tidx + it * NTHR
                    if e < 256:
                        j = e // 16
                        qq = e % 16
                        jc = cutlass.min(j, cutlass.max(p - 1, zi))
                        src = k_rd + cutlass.Int64(((jc * H + h) * K + qq * 8) * 2)
                        _cp16z(
                            s_kh + j * PK + qq * 16,
                            src,
                            cutlass.Int32(cutlass.select_(j < p, cutlass.Int32(16), zi)),
                        )
                if tidx < NHV * 4:
                    hl = tidx // 4
                    qq = tidx % 4
                    src = g_rd + cutlass.Int64(((hv0 + hl) * HIST + qq * 4) * 4)
                    _cp16z(s_gh + hl * 64 + qq * 16, src, cutlass.Int32(16))
            _cp_commit()

            if const_expr(self.spec_hist):
                s_u = s_u + rb * (NHV * 16 * PU)
                s_kh = s_kh + rb * (16 * PK)
                s_gh = s_gh + rb * (NHV * 64)

            # ---- 3. the step's own inputs (written by the previous kernel) ------------------
            if const_expr(not self.qk_first):
                if const_expr(self.pdl and self.state_before_wait):
                    cute.arch.griddepcontrol_wait()
                if const_expr(self.pdl and self.early_trigger):
                    cute.arch.griddepcontrol_launch_dependents()
                n_v: cutlass.Constexpr = NHV * T * 16
                for it in cutlass.range_constexpr(-(-n_v // NTHR)):
                    e = tidx + it * NTHR
                    if e < n_v:
                        hl = e // (T * 16)
                        t = (e // 16) % T
                        qq = e % 16
                        src = (
                            v_ptr
                            + (
                                n64 * v_sn
                                + cutlass.Int64(t) * v_st
                                + cutlass.Int64((hv0 + hl) * V + qq * 8)
                            )
                            * 2
                        )
                        _cp16z(s_v + (hl * T + t) * 256 + qq * 16, src, cutlass.Int32(16))
                _cp_commit()

            # key / query B fragments: column nt * 8 + g of the m16n8 tiles; lane quad c holds
            # head dims 32 jj + 8 c .. 32 jj + 8 c + 7.
            bq = []
            for nt in cutlass.range_constexpr(NT):
                if const_expr(self.qk_first):
                    rq = pre[nt]
                else:
                    rq = self._qk_raw(
                        nt, q_ptr, k_ptr, q_sn, q_st, k_sn, k_st, n64, h, g, c, scale, tidx >= 0
                    )
                bq.append(self._qk_frag(rq[0], rq[1], rq[2]))

            # the new keys into the history (one CTA of the head, warp 0)
            if sp == 0:
                if warp == 0:
                    # column g < T of tile 0 is key g (both column layouts)
                    if g < T:
                        kfr = bq[0]
                        dst = k_wr + cutlass.Int64((((woff + g) * H + h) * K + c * 8) * 2)
                        for jj in cutlass.range_constexpr(4):
                            _stg4(
                                dst + jj * 64,
                                kfr[4 * jj],
                                kfr[4 * jj + 1],
                                kfr[4 * jj + 2],
                                kfr[4 * jj + 3],
                            )

            # gating, cumulative decays, the G history (a thread per v head, off warp 0)
            if gpred:
                hl = gi
                hv = hv0 + hl
                gs = _ldg_f32p(
                    g_rd + cutlass.Int64(hv * HIST + p - 1) * 4,
                    cutlass.Int32(cutlass.select_(p > 0, cutlass.Int32(1), zi)),
                )
                if const_expr(self.qk_first):
                    graws, braws, alv, dtb = gpre
                else:
                    graws, braws, alv, dtb = self._gate_raw(
                        gpred, hv, n64, g_ptr, g_st, b_ptr, b_st, alog_ptr, dtb_ptr
                    )
                aexp = _exp(alv)
                owns = (hv_first + hl) * 8 >= sp * TPC
                gl = zf
                for t in cutlass.range_constexpr(T):
                    graw = self._g_f32(graws[t])
                    braw = self._b_f32(braws[t])
                    if const_expr(self.gating):
                        x = graw + dtb
                        spl = cutlass.Float32(
                            cutlass.select_(x <= 20.0, LN2 * _lg2(1.0 + _ex2(x * LOG2E)), x)
                        )
                        gt = -aexp * spl
                        beta = _div_approx(cutlass.Float32(1.0), 1.0 + _ex2(-braw * LOG2E))
                    else:
                        gt = graw
                        beta = braw
                    gl = gl + gt
                    gabs = gs + gl
                    _sts_f32(s_g + (hl * 8 + t) * 4, gabs)
                    _sts_f32(s_eg + (hl * 8 + t) * 4, _exp(gabs))
                    _sts_f32(s_beta + (hl * 8 + t) * 4, beta)
                    if owns:
                        _stg_f32(
                            g_wr + cutlass.Int64((hv * HIST + woff + t) * 4),
                            cutlass.Float32(cutlass.select_(wrap, gl, gabs)),
                        )
                _sts_f32(s_egs + hl * 4, _exp(gs))

            # new-token dot products: rows (keys | queries) x key columns
            if warp == 0:
                d = [zf, zf, zf, zf]
                for jj in cutlass.range_constexpr(4):
                    for s in cutlass.range_constexpr(2):
                        b0 = bq[0][4 * jj + 2 * s]
                        b1 = bq[0][4 * jj + 2 * s + 1]
                        if const_expr(NT == 1):
                            a = (b0, zi, b1, zi)
                        else:
                            a = (b0, bq[1][4 * jj + 2 * s], b1, bq[1][4 * jj + 2 * s + 1])
                        d = _mma(a, (b0, b1), d)
                _sts2_f32(s_kk + (g * 8 + 2 * c) * 4, d[0], d[1])
                _sts2_f32(s_kk + ((g + 8) * 8 + 2 * c) * 4, d[2], d[3])

            # the tiles' state dots need only the state and the new keys / queries: run them
            # while the history is still in flight
            early = []
            if const_expr(self.use_early):
                for i in cutlass.range_constexpr(TPW):
                    if const_expr(NSM == 0):
                        if const_expr(i + self.pf < TPW):
                            tiles.append(
                                self._load_tile(
                                    self._tile_addr(
                                        h_ptr, slot64, h_ss, h, sp, warp, i + self.pf, g, c
                                    )
                                )
                            )
                    if const_expr(i < self.pf or NSM == 0):
                        early.append(self._tile_dots(tiles[i], bq, zf))
                    else:
                        early.append(
                            self._tile_dots(self._smem_tile(s_st, s_mb, i, g, c, zi), bq, zf)
                        )

            _cp_wait_all()
            cute.arch.sync_threads()

            # ---- 4. history-key dots, commit operand, per-head scalars ----------------------
            hk_warp: cutlass.Constexpr = 1 if W > 1 else 0
            if warp == hk_warp:
                dh = [[zf, zf, zf, zf] for _ in range(NT)]
                for jj in cutlass.range_constexpr(4):
                    rg = _lds4_i32(s_kh + g * PK + jj * 64 + c * 16)
                    rh = _lds4_i32(s_kh + (g + 8) * PK + jj * 64 + c * 16)
                    for s in cutlass.range_constexpr(2):
                        a = (rg[2 * s], rh[2 * s], rg[2 * s + 1], rh[2 * s + 1])
                        for nt in cutlass.range_constexpr(NT):
                            dh[nt] = _mma(
                                a, (bq[nt][4 * jj + 2 * s], bq[nt][4 * jj + 2 * s + 1]), dh[nt]
                            )
                # history coefficients (k_t . k_j or q_t . k_j) exp(G_t - G_j), j < p, as bf16
                # B fragments [head][column][j]
                for hl in cutlass.range_constexpr(NHV):
                    gh_g = _lds_f32(s_gh + (hl * 16 + g) * 4)
                    gh_h = _lds_f32(s_gh + (hl * 16 + g + 8) * 4)
                    for nt in cutlass.range_constexpr(NT):
                        for e in cutlass.range_constexpr(2):
                            col = nt * 8 + 2 * c + e
                            if const_expr(TP == 4):
                                tt = (2 * c + e) % 4
                            else:
                                tt = 2 * c + e
                            tv = tt < T
                            gtt = _lds_f32(s_g + (hl * 8 + tt) * 4)
                            vg = cutlass.Float32(
                                cutlass.select_((g < p) & tv, dh[nt][e] * _exp(gtt - gh_g), zf)
                            )
                            vh = cutlass.Float32(
                                cutlass.select_(
                                    ((g + 8) < p) & tv, dh[nt][2 + e] * _exp(gtt - gh_h), zf
                                )
                            )
                            _sts_bf16(s_cf + ((hl * 16 + col) * 16 + g) * 2, vg)
                            _sts_bf16(s_cf + ((hl * 16 + col) * 16 + g + 8) * 2, vh)

            if const_expr(self.commit):
                if wrap:
                    for it in cutlass.range_constexpr(-(-256 // NTHR)):
                        e = tidx + it * NTHR
                        if e < 256:
                            j = e // 16
                            jj = (e // 4) % 4
                            aa = e % 4
                            w4 = _lds4_i32(s_kh + j * PK + jj * 64 + aa * 16)
                            for s4 in cutlass.range_constexpr(4):
                                wv = w4[s4]
                                if const_expr(self.spec_hist):
                                    wv = cutlass.Int32(cutlass.select_(j < p, wv, zi))
                                _sts_i32(s_kp + j * PU + (16 * jj + 4 * s4 + aa) * 4, wv)

            n_ts: cutlass.Constexpr = NHV * T * T
            for it in cutlass.range_constexpr(-(-n_ts // NTHR)):
                e = tidx + it * NTHR
                if e < n_ts:
                    hl = e // (T * T)
                    t = (e // T) % T
                    s = e % T
                    gt = _lds_f32(s_g + (hl * 8 + t) * 4)
                    gsv = _lds_f32(s_g + (hl * 8 + s) * 4)
                    bt = _lds_f32(s_beta + (hl * 8 + t) * 4)
                    dec = _exp(gt - gsv)
                    kk = _lds_f32(s_kk + (t * 8 + s) * 4)
                    qk = _lds_f32(s_kk + ((TP + t) * 8 + s) * 4)
                    av = cutlass.Float32(cutlass.select_(s < t, bt * kk * dec, zf))
                    cv = cutlass.Float32(cutlass.select_(s <= t, qk * dec, zf))
                    _sts_f32(s_a + (hl * 64 + t * 8 + s) * 4, av)
                    _sts_f32(s_cq + (hl * 64 + t * 8 + s) * 4, cv)
            if const_expr(self.commit):
                if tidx < NHV * 16:
                    hl = tidx // 16
                    j = tidx % 16
                    gsv = cutlass.Float32(0.0)
                    if p > 0:
                        gsv = _lds_f32(s_gh + (hl * 16 + p - 1) * 4)
                    ghj = _lds_f32(s_gh + (hl * 16 + j) * 4)
                    _sts_f32(
                        s_dec + (hl * 16 + j) * 4,
                        cutlass.Float32(cutlass.select_(j < p, _exp(gsv - ghj), zf)),
                    )
            cute.arch.sync_threads()

            # ---- 5. the warp's m-tiles ------------------------------------------------------
            jrow = (lane % 8) + 8 * (lane // 16)
            v8 = 8 * ((lane // 8) % 2)
            r = lane % 16
            hf = lane // 16
            godd = (g % 2) == 1
            mask01 = cutlass.Int32(
                cutlass.select_(2 * c < p, cutlass.Int32(0xFFFF), zi)
            ) | cutlass.Int32(cutlass.select_(2 * c + 1 < p, cutlass.Int32(-65536), zi))
            mask89 = cutlass.Int32(
                cutlass.select_(2 * c + 8 < p, cutlass.Int32(0xFFFF), zi)
            ) | cutlass.Int32(cutlass.select_(2 * c + 9 < p, cutlass.Int32(-65536), zi))
            for i in cutlass.range_constexpr(TPW):
                if const_expr(NSM == 0 and not self.use_early):
                    if const_expr(i + self.pf < TPW):
                        tiles.append(
                            self._load_tile(
                                self._tile_addr(h_ptr, slot64, h_ss, h, sp, warp, i + self.pf, g, c)
                            )
                        )
                if const_expr(i < self.pf or NSM == 0):
                    cur = tiles[i]
                elif const_expr(self.use_early):
                    # a staged tile is read again from shared memory only for its commit
                    cur = None
                else:
                    # rows g and g + 8 of the staged tile; chunk order by row parity keeps each
                    # quarter-warp's two rows on disjoint banks
                    kslot = i - self.pf
                    _mbar_wait(s_mb + kslot * 8, zi)
                    tb = s_st + kslot * 4096 + g * 256 + c * 16
                    raw = []
                    for ii in cutlass.range_constexpr(4):
                        jsw = (g % 2) ^ ii
                        raw.append(_lds4_i32(tb + jsw * 64))
                        raw.append(_lds4_i32(tb + 2048 + jsw * 64))
                    cur = []
                    for jj in cutlass.range_constexpr(4):
                        for rr in cutlass.range_constexpr(2):
                            for e in cutlass.range_constexpr(4):
                                cur.append(
                                    cutlass.Int32(
                                        cutlass.select_(
                                            godd, raw[2 * (jj ^ 1) + rr][e], raw[2 * jj + rr][e]
                                        )
                                    )
                                )
                mt = sp * TPC + warp + W * i
                hvr = mt // 8
                hv = h * GRP + hvr
                hl = hvr - hv_first
                v0 = (mt % 8) * 16

                if const_expr(self.use_early):
                    acc = early[i]
                else:
                    acc = self._tile_dots(cur, bq, zf)

                au = _ldsm4t(s_u + hl * (16 * PU) + jrow * PU + (v0 + v8) * 2)
                if const_expr(self.spec_hist):
                    au = [cutlass.Int32(au[q4] & (mask01 if q4 < 2 else mask89)) for q4 in range(4)]
                hacc = []
                for nt in cutlass.range_constexpr(NT):
                    cfa = s_cf + ((hl * 16 + nt * 8 + g) * 16 + 2 * c) * 2
                    bc = (_lds_i32(cfa), _lds_i32(cfa + 16))
                    hacc.append(_mma(au, bc, [zf, zf, zf, zf]))

                for nt in cutlass.range_constexpr(NT):
                    col0 = nt * 8 + 2 * c
                    if const_expr(TP == 4):
                        t0 = (2 * c) % 4
                    else:
                        t0 = 2 * c
                    eg0 = _lds_f32(s_eg + (hl * 8 + t0) * 4)
                    eg1 = _lds_f32(s_eg + (hl * 8 + t0 + 1) * 4)
                    _sts2_f32(
                        s_x + g * RP + col0 * 4,
                        eg0 * acc[nt][0] + hacc[nt][0],
                        eg1 * acc[nt][1] + hacc[nt][1],
                    )
                    _sts2_f32(
                        s_x + (g + 8) * RP + col0 * 4,
                        eg0 * acc[nt][2] + hacc[nt][2],
                        eg1 * acc[nt][3] + hacc[nt][3],
                    )
                cute.arch.sync_warp()

                # per state row: the forward substitution and the outputs
                xs = []
                ys = []
                for q4 in cutlass.range_constexpr(TP // 4):
                    xs.extend(_lds4_f32(s_x + r * RP + q4 * 16))
                    ys.extend(_lds4_f32(s_x + r * RP + (TP + 4 * q4) * 4))
                us = []
                for t in cutlass.range_constexpr(T):
                    vt = _lds_bf16(s_v + ((hl * T + t) * 128 + v0 + r) * 2)
                    bt = _lds_f32(s_beta + (hl * 8 + t) * 4)
                    u = bt * (vt - xs[t])
                    for s in cutlass.range_constexpr(t):
                        u = u - _lds_f32(s_a + (hl * 64 + t * 8 + s) * 4) * us[s]
                    us.append(u)
                if hf == 0:
                    for t in cutlass.range_constexpr(T):
                        o = ys[t]
                        for s in cutlass.range_constexpr(t + 1):
                            o = o + _lds_f32(s_cq + (hl * 64 + t * 8 + s) * 4) * us[s]
                        _stg_bf16(o_ptr + (((n64 * T + t) * HV + hv) * V + v0 + r) * 2, o)
                else:
                    for t in cutlass.range_constexpr(T):
                        _stg_bf16(
                            u_wr + cutlass.Int64((((woff + t) * HV + hv) * V + v0 + r) * 2), us[t]
                        )
                cute.arch.sync_warp()

                # the checkpoint commit of a wrapping row, over the tile just used
                if const_expr(self.commit):
                    if wrap:
                        dc = [_lds_f32(s_dec + (hl * 16 + 2 * c + jx) * 4) for jx in [0, 1, 8, 9]]
                        egs = _lds_f32(s_egs + hl * 4)
                        ahi = []
                        alo = []
                        for q4 in cutlass.range_constexpr(4):
                            dl = dc[0] if q4 < 2 else dc[2]
                            dhh = dc[1] if q4 < 2 else dc[3]
                            x0 = _bf2_lo(au[q4]) * dl
                            x1 = _bf2_hi(au[q4]) * dhh
                            hw = _bf2_pack(x0, x1)
                            ahi.append(hw)
                            alo.append(_bf2_pack(x0 - _bf2_lo(hw), x1 - _bf2_hi(hw)))
                        if const_expr(cur is None):
                            curc = self._smem_tile(s_st, s_mb, i, g, c, zi)
                        else:
                            curc = cur
                        taddr = self._tile_addr(h_ptr, slot64, h_ss, h, sp, warp, i, g, c)
                        kp_row = (
                            s_kp + ((lane % 8) + 8 * ((lane // 8) % 2)) * PU + 16 * (lane // 16)
                        )
                        for jj in cutlass.range_constexpr(4):
                            bb0 = _ldsm4t(kp_row + (4 * jj) * 16)
                            bb1 = _ldsm4t(kp_row + (4 * jj + 2) * 16)
                            ng = []
                            nh8 = []
                            for s4 in cutlass.range_constexpr(4):
                                bsrc = bb0 if s4 < 2 else bb1
                                bf = (bsrc[2 * (s4 % 2)], bsrc[2 * (s4 % 2) + 1])
                                dcm = _mma(ahi, bf, [zf, zf, zf, zf])
                                if const_expr(self.lo_split):
                                    dcm = _mma(alo, bf, dcm)
                                wg = curc[8 * jj + s4]
                                wh = curc[8 * jj + 4 + s4]
                                ng.append(
                                    _bf2_pack(
                                        egs * _bf2_lo(wg) + dcm[0], egs * _bf2_hi(wg) + dcm[1]
                                    )
                                )
                                nh8.append(
                                    _bf2_pack(
                                        egs * _bf2_lo(wh) + dcm[2], egs * _bf2_hi(wh) + dcm[3]
                                    )
                                )
                            _stg4(taddr + jj * 64, ng[0], ng[1], ng[2], ng[3])
                            _stg4(taddr + (2048 + jj * 64), nh8[0], nh8[1], nh8[2], nh8[3])

            if const_expr(self.pdl and not self.early_trigger):
                cute.arch.griddepcontrol_launch_dependents()

    # -- the persistent variant: CTAs loop over (request, head, split) items, the next item's
    # state tiles and history in flight while the current one computes -------------------------

    @cute.kernel
    def kernel_p(
        self,
        sidx_ptr: cutlass.Int64,
        q_ptr: cutlass.Int64,
        k_ptr: cutlass.Int64,
        v_ptr: cutlass.Int64,
        q_sn: cutlass.Int64,
        q_st: cutlass.Int64,
        k_sn: cutlass.Int64,
        k_st: cutlass.Int64,
        v_sn: cutlass.Int64,
        v_st: cutlass.Int64,
        g_ptr: cutlass.Int64,
        g_st: cutlass.Int64,
        b_ptr: cutlass.Int64,
        b_st: cutlass.Int64,
        alog_ptr: cutlass.Int64,
        dtb_ptr: cutlass.Int64,
        o_ptr: cutlass.Int64,
        h_ptr: cutlass.Int64,
        h_ss: cutlass.Int64,
        pnat_ptr: cutlass.Int64,
        buf_ptr: cutlass.Int64,
        ou_ptr: cutlass.Int64,
        ok_ptr: cutlass.Int64,
        og_ptr: cutlass.Int64,
        scale: cutlass.Float32,
        n_items: cutlass.Int32,
    ):
        T: cutlass.Constexpr = self.T
        TP: cutlass.Constexpr = self.TP
        NT: cutlass.Constexpr = self.NT
        W: cutlass.Constexpr = self.W
        TPW: cutlass.Constexpr = self.tpw
        TPC: cutlass.Constexpr = self.tpc
        NHV: cutlass.Constexpr = self.nhv
        S: cutlass.Constexpr = self.S
        NTHR: cutlass.Constexpr = 32 * self.W
        RP: cutlass.Constexpr = self.RP
        NSM: cutlass.Constexpr = self.nsm
        SU: cutlass.Constexpr = NHV * 16 * PU
        SK: cutlass.Constexpr = 16 * PK
        SG: cutlass.Constexpr = NHV * 64
        ST: cutlass.Constexpr = W * NSM * 4096
        SMB: cutlass.Constexpr = W * NSM * 8

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gsz, _, _ = cute.arch.grid_dim()
        warp = tidx // 32
        lane = tidx % 32
        g = lane // 4
        c = lane % 4
        zf = cutlass.Float32(0.0)
        zi = cutlass.Int32(0)

        smem = cutlass.utils.SmemAllocator()
        base_t = smem.allocate_tensor(
            cutlass.Int8, cute.make_layout((self.smem - 128,), stride=(1,)), byte_alignment=128
        )
        sbase = cutlass.Int32(base_t.iterator.toint())
        s_u0 = sbase + self.o_u
        s_kh0 = sbase + self.o_kh
        s_gh0 = sbase + self.o_gh
        s_meta = sbase + self.o_meta
        s_kp = sbase + self.o_kp
        s_v = sbase + self.o_v
        s_x = sbase + self.o_x + warp * (16 * RP)
        s_kk = sbase + self.o_kk
        s_cf = sbase + self.o_cf
        s_dec = sbase + self.o_dec
        s_g = sbase + self.o_g
        s_eg = sbase + self.o_eg
        s_beta = sbase + self.o_beta
        s_a = sbase + self.o_a
        s_cq = sbase + self.o_cq
        s_egs = sbase + self.o_egs
        s_st0 = sbase + self.o_st + warp * (NSM * 4096)
        s_mb0 = sbase + self.o_mb + warp * (NSM * 8)

        nk = (n_items - bidx + gsz - 1) // gsz
        # ---- the items' slot / pending count / half, once --------------------------------------
        for m_t in cutlass.range(tidx, nk, NTHR):
            m_it = bidx + m_t * gsz
            m_n = m_it // (H * S)
            m_sl = _ldg_i32(sidx_ptr + cutlass.Int64(m_n) * 4)
            m_p = zi
            m_b = zi
            if m_sl >= 0:
                m_p = _ldg_i32(pnat_ptr + cutlass.Int64(m_sl) * 4)
                m_b = _ldg_i32(buf_ptr + cutlass.Int64(m_sl) * 4)
            _sts_i32(s_meta + m_t * 16, m_sl)
            _sts_i32(s_meta + m_t * 16 + 4, m_p)
            _sts_i32(s_meta + m_t * 16 + 8, m_b)
        if const_expr(NSM > 0):
            if lane == 0:
                for k_ in cutlass.range_constexpr(2 * NSM):
                    _mbar_init(s_mb0 + (k_ // NSM) * SMB + (k_ % NSM) * 8, cutlass.Int32(1))
                _fence_mbar_init()
        cute.arch.sync_threads()

        cur0 = cute.make_rmem_tensor(cute.make_layout((32,), stride=(1,)), cutlass.Int32)
        nxt0 = cute.make_rmem_tensor(cute.make_layout((32,), stride=(1,)), cutlass.Int32)
        NQ: cutlass.Constexpr = 16 * NT
        GN: cutlass.Constexpr = 2 * T
        qkc = cute.make_rmem_tensor(cute.make_layout((NQ,), stride=(1,)), cutlass.Int32)
        qkn = cute.make_rmem_tensor(cute.make_layout((NQ,), stride=(1,)), cutlass.Int32)
        gtc = cute.make_rmem_tensor(cute.make_layout((GN,), stride=(1,)), cutlass.Int32)
        gtn = cute.make_rmem_tensor(cute.make_layout((GN,), stride=(1,)), cutlass.Int32)
        gfc = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), cutlass.Float32)
        gfn = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), cutlass.Float32)
        SVB: cutlass.Constexpr = NHV * T * 256
        jrow = (lane % 8) + 8 * (lane // 16)
        v8 = 8 * ((lane // 8) % 2)
        r = lane % 16
        hf = lane // 16
        godd = (g % 2) == 1
        gi = tidx - (32 * (W - 1) if W > 1 else 0)
        gpred = (gi >= 0) & (gi < NHV)
        gic = cutlass.min(cutlass.max(gi, zi), NHV - 1)

        for k in cutlass.range(nk):
            # ---- issue: item 0 (first pass only), then item k + 1; everything an item reads
            # from global memory is in flight one item ahead ----------------------------------
            for q2 in cutlass.range_constexpr(2):
                kk = k + q2
                if const_expr(q2 == 0):
                    do_it = k == 0
                else:
                    do_it = kk < nk
                if do_it:
                    st_ = kk % 2
                    it_ = bidx + kk * gsz
                    sp_ = it_ % S
                    nh_ = it_ // S
                    h_ = nh_ % H
                    n64_ = cutlass.Int64(nh_ // H)
                    sl = _lds_i32(s_meta + kk * 16)
                    pp = _lds_i32(s_meta + kk * 16 + 4)
                    bb = _lds_i32(s_meta + kk * 16 + 8)
                    hvf = (sp_ * TPC) // 8
                    hv0_ = h_ * GRP + hvf
                    if const_expr(q2 == 0):
                        if const_expr(self.pdl):
                            cute.arch.griddepcontrol_wait()
                            if const_expr(self.early_trigger):
                                cute.arch.griddepcontrol_launch_dependents()
                    # the item's own inputs first (they are needed first), then its history, then its
                    # state: a lane's requests come back roughly in order
                    if sl >= 0:
                        sl64 = cutlass.Int64(sl)
                        n_v: cutlass.Constexpr = NHV * T * 16
                        for it2 in cutlass.range_constexpr(-(-n_v // NTHR)):
                            e = tidx + it2 * NTHR
                            if e < n_v:
                                hl = e // (T * 16)
                                t = (e // 16) % T
                                qq = e % 16
                                src = (
                                    v_ptr
                                    + (
                                        n64_ * v_sn
                                        + cutlass.Int64(t) * v_st
                                        + cutlass.Int64((hv0_ + hl) * V + qq * 8)
                                    )
                                    * 2
                                )
                                _cp16z(
                                    s_v + st_ * SVB + (hl * T + t) * 256 + qq * 16,
                                    src,
                                    cutlass.Int32(16),
                                )
                        for nt in cutlass.range_constexpr(NT):
                            rq = self._qk_raw(
                                nt,
                                q_ptr,
                                k_ptr,
                                q_sn,
                                q_st,
                                k_sn,
                                k_st,
                                n64_,
                                h_,
                                g,
                                c,
                                scale,
                                tidx >= 0,
                            )
                            for e in cutlass.range_constexpr(16):
                                if const_expr(q2 == 0):
                                    qkc[nt * 16 + e] = rq[0][e]
                                else:
                                    qkn[nt * 16 + e] = rq[0][e]
                        gr = self._gate_raw(
                            gpred, hv0_ + gic, n64_, g_ptr, g_st, b_ptr, b_st, alog_ptr, dtb_ptr
                        )
                        for t in cutlass.range_constexpr(T):
                            if const_expr(q2 == 0):
                                gtc[t] = gr[0][t]
                                gtc[T + t] = gr[1][t]
                            else:
                                gtn[t] = gr[0][t]
                                gtn[T + t] = gr[1][t]
                        if const_expr(q2 == 0):
                            gfc[0] = gr[2]
                            gfc[1] = gr[3]
                        else:
                            gfn[0] = gr[2]
                            gfn[1] = gr[3]
                        rsl = sl64 * 2 + cutlass.Int64(bb)
                        u_rd_ = ou_ptr + rsl * (HIST * HV * V * 2)
                        k_rd_ = ok_ptr + rsl * (HIST * H * K * 2)
                        g_rd_ = og_ptr + rsl * (HV * HIST * 4)
                        n_u: cutlass.Constexpr = NHV * 256
                        for it2 in cutlass.range_constexpr(-(-n_u // NTHR)):
                            e = tidx + it2 * NTHR
                            if e < n_u:
                                hl = e // 256
                                j = (e // 16) % 16
                                qq = e % 16
                                jc = cutlass.min(j, cutlass.max(pp - 1, zi))
                                src = u_rd_ + cutlass.Int64(
                                    ((jc * HV + hv0_ + hl) * V + qq * 8) * 2
                                )
                                _cp16z(
                                    s_u0 + st_ * SU + hl * (16 * PU) + j * PU + qq * 16,
                                    src,
                                    cutlass.Int32(cutlass.select_(j < pp, cutlass.Int32(16), zi)),
                                )
                        for it2 in cutlass.range_constexpr(-(-256 // NTHR)):
                            e = tidx + it2 * NTHR
                            if e < 256:
                                j = e // 16
                                qq = e % 16
                                jc = cutlass.min(j, cutlass.max(pp - 1, zi))
                                src = k_rd_ + cutlass.Int64(((jc * H + h_) * K + qq * 8) * 2)
                                _cp16z(
                                    s_kh0 + st_ * SK + j * PK + qq * 16,
                                    src,
                                    cutlass.Int32(cutlass.select_(j < pp, cutlass.Int32(16), zi)),
                                )
                        if gpred:
                            # a v head's G row by the thread that gates that head (read back by it)
                            for qq in cutlass.range_constexpr(4):
                                src = g_rd_ + cutlass.Int64(((hv0_ + gi) * HIST + qq * 4) * 4)
                                _cp16z(s_gh0 + st_ * SG + gi * 64 + qq * 16, src, cutlass.Int32(16))
                        vals = self._load_tile(
                            self._tile_addr(h_ptr, sl64, h_ss, h_, sp_, warp, 0, g, c)
                        )
                        for e in cutlass.range_constexpr(32):
                            if const_expr(q2 == 0):
                                cur0[e] = vals[e]
                            else:
                                nxt0[e] = vals[e]
                        if const_expr(NSM > 0):
                            if lane == 0:
                                for k_ in cutlass.range_constexpr(NSM):
                                    _bulk_g2s(
                                        s_st0 + st_ * ST + k_ * 4096,
                                        self._tile_addr(
                                            h_ptr, sl64, h_ss, h_, sp_, warp, 1 + k_, 0, 0
                                        ),
                                        cutlass.Int32(4096),
                                        s_mb0 + st_ * SMB + k_ * 8,
                                    )
                    else:
                        if const_expr(NSM > 0):
                            if lane == 0:
                                for k_ in cutlass.range_constexpr(NSM):
                                    _mbar_arrive(s_mb0 + st_ * SMB + k_ * 8)
                _cp_commit()  # item 0 (empty later) / item k + 1

            # ---- item k ---------------------------------------------------------------------
            st = k % 2
            it = bidx + k * gsz
            sp = it % S
            nh = it // S
            h = nh % H
            n = nh // H
            n64 = cutlass.Int64(n)
            slot = _lds_i32(s_meta + k * 16)
            p = _lds_i32(s_meta + k * 16 + 4)
            rb = _lds_i32(s_meta + k * 16 + 8)
            s_u = s_u0 + st * SU
            s_kh = s_kh0 + st * SK
            s_gh = s_gh0 + st * SG
            s_st = s_st0 + st * ST
            s_mb = s_mb0 + st * SMB
            s_vk = s_v + st * SVB
            ph = (k // 2) % 2
            slot64 = cutlass.Int64(cutlass.max(slot, zi))
            wrap = (p + T) > HIST
            wb = cutlass.Int32(cutlass.select_(wrap, 1 - rb, rb))
            woff = cutlass.Int32(cutlass.select_(wrap, zi, p))
            hv_first = (sp * TPC) // 8
            hv0 = h * GRP + hv_first
            wslot = slot64 * 2 + cutlass.Int64(wb)
            u_wr = ou_ptr + wslot * (HIST * HV * V * 2)
            k_wr = ok_ptr + wslot * (HIST * H * K * 2)
            g_wr = og_ptr + wslot * (HV * HIST * 4)

            live = slot >= 0
            if live:
                bq = []
                for nt in cutlass.range_constexpr(NT):
                    rw = [qkc[nt * 16 + e] for e in range(16)]
                    if const_expr(TP == 4):
                        is_q = g >= 4
                        mult = cutlass.Float32(cutlass.select_(is_q, scale, cutlass.Float32(1.0)))
                    elif const_expr(nt == 1):
                        is_q = g >= 0
                        mult = scale
                    else:
                        is_q = g < 0
                        mult = cutlass.Float32(1.0)
                    bq.append(self._qk_frag(rw, is_q, mult))

                # the new keys into the history (one CTA of the head, warp 0)
                if sp == 0:
                    if warp == 0:
                        if g < T:
                            kfr = bq[0]
                            dst = k_wr + cutlass.Int64((((woff + g) * H + h) * K + c * 8) * 2)
                            for jj in cutlass.range_constexpr(4):
                                _stg4(
                                    dst + jj * 64,
                                    kfr[4 * jj],
                                    kfr[4 * jj + 1],
                                    kfr[4 * jj + 2],
                                    kfr[4 * jj + 3],
                                )

                # new-token dot products: rows (keys | queries) x key columns
                if warp == 0:
                    d = [zf, zf, zf, zf]
                    for jj in cutlass.range_constexpr(4):
                        for s in cutlass.range_constexpr(2):
                            b0 = bq[0][4 * jj + 2 * s]
                            b1 = bq[0][4 * jj + 2 * s + 1]
                            if const_expr(NT == 1):
                                a = (b0, zi, b1, zi)
                            else:
                                a = (b0, bq[1][4 * jj + 2 * s], b1, bq[1][4 * jj + 2 * s + 1])
                            d = _mma(a, (b0, b1), d)
                    _sts2_f32(s_kk + (g * 8 + 2 * c) * 4, d[0], d[1])
                    _sts2_f32(s_kk + ((g + 8) * 8 + 2 * c) * 4, d[2], d[3])

                _cp_wait1()
                # gating, cumulative decays, the G history (a thread per v head, off warp 0); the
                # thread's own G row copy has landed
                if gpred:
                    hl = gi
                    hv = hv0 + hl
                    gs = cutlass.Float32(
                        cutlass.select_(
                            p > 0, _lds_f32(s_gh + (hl * 16 + cutlass.max(p - 1, zi)) * 4), zf
                        )
                    )
                    graws = [_b16_f32(gtc[t]) for t in range(T)]
                    braws = [_b16_f32(gtc[T + t]) for t in range(T)]
                    alv = gfc[0]
                    dtb = gfc[1]
                    aexp = _exp(alv)
                    owns = (hv_first + hl) * 8 >= sp * TPC
                    gl = zf
                    for t in cutlass.range_constexpr(T):
                        graw = graws[t]
                        braw = braws[t]
                        if const_expr(self.gating):
                            x = graw + dtb
                            spl = cutlass.Float32(
                                cutlass.select_(x <= 20.0, LN2 * _lg2(1.0 + _ex2(x * LOG2E)), x)
                            )
                            gt = -aexp * spl
                            beta = _div_approx(cutlass.Float32(1.0), 1.0 + _ex2(-braw * LOG2E))
                        else:
                            gt = graw
                            beta = braw
                        gl = gl + gt
                        gabs = gs + gl
                        _sts_f32(s_g + (hl * 8 + t) * 4, gabs)
                        _sts_f32(s_eg + (hl * 8 + t) * 4, _exp(gabs))
                        _sts_f32(s_beta + (hl * 8 + t) * 4, beta)
                        if owns:
                            _stg_f32(
                                g_wr + cutlass.Int64((hv * HIST + woff + t) * 4),
                                cutlass.Float32(cutlass.select_(wrap, gl, gabs)),
                            )
                    _sts_f32(s_egs + hl * 4, _exp(gs))
                cute.arch.sync_threads()

                hk_warp: cutlass.Constexpr = 1 if W > 1 else 0
                if warp == hk_warp:
                    dh = [[zf, zf, zf, zf] for _ in range(NT)]
                    for jj in cutlass.range_constexpr(4):
                        rg = _lds4_i32(s_kh + g * PK + jj * 64 + c * 16)
                        rh = _lds4_i32(s_kh + (g + 8) * PK + jj * 64 + c * 16)
                        for s in cutlass.range_constexpr(2):
                            a = (rg[2 * s], rh[2 * s], rg[2 * s + 1], rh[2 * s + 1])
                            for nt in cutlass.range_constexpr(NT):
                                dh[nt] = _mma(
                                    a, (bq[nt][4 * jj + 2 * s], bq[nt][4 * jj + 2 * s + 1]), dh[nt]
                                )
                    for hl in cutlass.range_constexpr(NHV):
                        gh_g = _lds_f32(s_gh + (hl * 16 + g) * 4)
                        gh_h = _lds_f32(s_gh + (hl * 16 + g + 8) * 4)
                        for nt in cutlass.range_constexpr(NT):
                            for e in cutlass.range_constexpr(2):
                                col = nt * 8 + 2 * c + e
                                if const_expr(TP == 4):
                                    tt = (2 * c + e) % 4
                                else:
                                    tt = 2 * c + e
                                tv = tt < T
                                gtt = _lds_f32(s_g + (hl * 8 + tt) * 4)
                                vg = cutlass.Float32(
                                    cutlass.select_((g < p) & tv, dh[nt][e] * _exp(gtt - gh_g), zf)
                                )
                                vh = cutlass.Float32(
                                    cutlass.select_(
                                        ((g + 8) < p) & tv, dh[nt][2 + e] * _exp(gtt - gh_h), zf
                                    )
                                )
                                _sts_bf16(s_cf + ((hl * 16 + col) * 16 + g) * 2, vg)
                                _sts_bf16(s_cf + ((hl * 16 + col) * 16 + g + 8) * 2, vh)

                if const_expr(self.commit):
                    if wrap:
                        for it2 in cutlass.range_constexpr(-(-256 // NTHR)):
                            e = tidx + it2 * NTHR
                            if e < 256:
                                j = e // 16
                                jj = (e // 4) % 4
                                aa = e % 4
                                w4 = _lds4_i32(s_kh + j * PK + jj * 64 + aa * 16)
                                for s4 in cutlass.range_constexpr(4):
                                    _sts_i32(s_kp + j * PU + (16 * jj + 4 * s4 + aa) * 4, w4[s4])

                n_ts: cutlass.Constexpr = NHV * T * T
                for it2 in cutlass.range_constexpr(-(-n_ts // NTHR)):
                    e = tidx + it2 * NTHR
                    if e < n_ts:
                        hl = e // (T * T)
                        t = (e // T) % T
                        s = e % T
                        gt = _lds_f32(s_g + (hl * 8 + t) * 4)
                        gsv = _lds_f32(s_g + (hl * 8 + s) * 4)
                        bt = _lds_f32(s_beta + (hl * 8 + t) * 4)
                        dec = _exp(gt - gsv)
                        kkv = _lds_f32(s_kk + (t * 8 + s) * 4)
                        qkv = _lds_f32(s_kk + ((TP + t) * 8 + s) * 4)
                        av = cutlass.Float32(cutlass.select_(s < t, bt * kkv * dec, zf))
                        cv = cutlass.Float32(cutlass.select_(s <= t, qkv * dec, zf))
                        _sts_f32(s_a + (hl * 64 + t * 8 + s) * 4, av)
                        _sts_f32(s_cq + (hl * 64 + t * 8 + s) * 4, cv)
                if const_expr(self.commit):
                    if tidx < NHV * 16:
                        hl = tidx // 16
                        j = tidx % 16
                        gsv = cutlass.Float32(0.0)
                        if p > 0:
                            gsv = _lds_f32(s_gh + (hl * 16 + p - 1) * 4)
                        ghj = _lds_f32(s_gh + (hl * 16 + j) * 4)
                        _sts_f32(
                            s_dec + (hl * 16 + j) * 4,
                            cutlass.Float32(cutlass.select_(j < p, _exp(gsv - ghj), zf)),
                        )
                cute.arch.sync_threads()

                for i in cutlass.range_constexpr(TPW):
                    if const_expr(i == 0):
                        cur = [cur0[e] for e in range(32)]
                    else:
                        kslot = i - 1
                        _mbar_wait(s_mb + kslot * 8, ph)
                        tb = s_st + kslot * 4096 + g * 256 + c * 16
                        raw = []
                        for ii in cutlass.range_constexpr(4):
                            jsw = (g % 2) ^ ii
                            raw.append(_lds4_i32(tb + jsw * 64))
                            raw.append(_lds4_i32(tb + 2048 + jsw * 64))
                        cur = []
                        for jj in cutlass.range_constexpr(4):
                            for rr in cutlass.range_constexpr(2):
                                for e in cutlass.range_constexpr(4):
                                    cur.append(
                                        cutlass.Int32(
                                            cutlass.select_(
                                                godd, raw[2 * (jj ^ 1) + rr][e], raw[2 * jj + rr][e]
                                            )
                                        )
                                    )
                    mt = sp * TPC + warp + W * i
                    hvr = mt // 8
                    hv = h * GRP + hvr
                    hl = hvr - hv_first
                    v0 = (mt % 8) * 16

                    acc0 = [[zf, zf, zf, zf] for _ in range(NT)]
                    acc1 = [[zf, zf, zf, zf] for _ in range(NT)]
                    for jj in cutlass.range_constexpr(4):
                        for s in cutlass.range_constexpr(2):
                            a = (
                                cur[8 * jj + 2 * s],
                                cur[8 * jj + 4 + 2 * s],
                                cur[8 * jj + 2 * s + 1],
                                cur[8 * jj + 4 + 2 * s + 1],
                            )
                            for nt in cutlass.range_constexpr(NT):
                                bfr = (bq[nt][4 * jj + 2 * s], bq[nt][4 * jj + 2 * s + 1])
                                if const_expr(s == 0):
                                    acc0[nt] = _mma(a, bfr, acc0[nt])
                                else:
                                    acc1[nt] = _mma(a, bfr, acc1[nt])
                    acc = [[acc0[nt][e] + acc1[nt][e] for e in range(4)] for nt in range(NT)]

                    au = _ldsm4t(s_u + hl * (16 * PU) + jrow * PU + (v0 + v8) * 2)
                    hacc = []
                    for nt in cutlass.range_constexpr(NT):
                        cfa = s_cf + ((hl * 16 + nt * 8 + g) * 16 + 2 * c) * 2
                        bc = (_lds_i32(cfa), _lds_i32(cfa + 16))
                        hacc.append(_mma(au, bc, [zf, zf, zf, zf]))

                    for nt in cutlass.range_constexpr(NT):
                        col0 = nt * 8 + 2 * c
                        if const_expr(TP == 4):
                            t0 = (2 * c) % 4
                        else:
                            t0 = 2 * c
                        eg0 = _lds_f32(s_eg + (hl * 8 + t0) * 4)
                        eg1 = _lds_f32(s_eg + (hl * 8 + t0 + 1) * 4)
                        _sts2_f32(
                            s_x + g * RP + col0 * 4,
                            eg0 * acc[nt][0] + hacc[nt][0],
                            eg1 * acc[nt][1] + hacc[nt][1],
                        )
                        _sts2_f32(
                            s_x + (g + 8) * RP + col0 * 4,
                            eg0 * acc[nt][2] + hacc[nt][2],
                            eg1 * acc[nt][3] + hacc[nt][3],
                        )
                    cute.arch.sync_warp()

                    xs = []
                    ys = []
                    for q4 in cutlass.range_constexpr(TP // 4):
                        xs.extend(_lds4_f32(s_x + r * RP + q4 * 16))
                        ys.extend(_lds4_f32(s_x + r * RP + (TP + 4 * q4) * 4))
                    us = []
                    for t in cutlass.range_constexpr(T):
                        vt = _lds_bf16(s_vk + ((hl * T + t) * 128 + v0 + r) * 2)
                        bt = _lds_f32(s_beta + (hl * 8 + t) * 4)
                        u = bt * (vt - xs[t])
                        for s in cutlass.range_constexpr(t):
                            u = u - _lds_f32(s_a + (hl * 64 + t * 8 + s) * 4) * us[s]
                        us.append(u)
                    if hf == 0:
                        for t in cutlass.range_constexpr(T):
                            o = ys[t]
                            for s in cutlass.range_constexpr(t + 1):
                                o = o + _lds_f32(s_cq + (hl * 64 + t * 8 + s) * 4) * us[s]
                            _stg_bf16(o_ptr + (((n64 * T + t) * HV + hv) * V + v0 + r) * 2, o)
                    else:
                        for t in cutlass.range_constexpr(T):
                            _stg_bf16(
                                u_wr + cutlass.Int64((((woff + t) * HV + hv) * V + v0 + r) * 2),
                                us[t],
                            )
                    cute.arch.sync_warp()

                    if const_expr(self.commit):
                        if wrap:
                            dc = [
                                _lds_f32(s_dec + (hl * 16 + 2 * c + jx) * 4) for jx in [0, 1, 8, 9]
                            ]
                            egs = _lds_f32(s_egs + hl * 4)
                            ahi = []
                            alo = []
                            for q4 in cutlass.range_constexpr(4):
                                dl = dc[0] if q4 < 2 else dc[2]
                                dhh = dc[1] if q4 < 2 else dc[3]
                                x0 = _bf2_lo(au[q4]) * dl
                                x1 = _bf2_hi(au[q4]) * dhh
                                hw = _bf2_pack(x0, x1)
                                ahi.append(hw)
                                alo.append(_bf2_pack(x0 - _bf2_lo(hw), x1 - _bf2_hi(hw)))
                            taddr = self._tile_addr(h_ptr, slot64, h_ss, h, sp, warp, i, g, c)
                            kp_row = (
                                s_kp + ((lane % 8) + 8 * ((lane // 8) % 2)) * PU + 16 * (lane // 16)
                            )
                            for jj in cutlass.range_constexpr(4):
                                bb0 = _ldsm4t(kp_row + (4 * jj) * 16)
                                bb1 = _ldsm4t(kp_row + (4 * jj + 2) * 16)
                                ng = []
                                nh8 = []
                                for s4 in cutlass.range_constexpr(4):
                                    bsrc = bb0 if s4 < 2 else bb1
                                    bf = (bsrc[2 * (s4 % 2)], bsrc[2 * (s4 % 2) + 1])
                                    dcm = _mma(ahi, bf, [zf, zf, zf, zf])
                                    if const_expr(self.lo_split):
                                        dcm = _mma(alo, bf, dcm)
                                    wg = cur[8 * jj + s4]
                                    wh = cur[8 * jj + 4 + s4]
                                    ng.append(
                                        _bf2_pack(
                                            egs * _bf2_lo(wg) + dcm[0], egs * _bf2_hi(wg) + dcm[1]
                                        )
                                    )
                                    nh8.append(
                                        _bf2_pack(
                                            egs * _bf2_lo(wh) + dcm[2], egs * _bf2_hi(wh) + dcm[3]
                                        )
                                    )
                                _stg4(taddr + jj * 64, ng[0], ng[1], ng[2], ng[3])
                                _stg4(taddr + (2048 + jj * 64), nh8[0], nh8[1], nh8[2], nh8[3])

            for ce in cutlass.range_constexpr(32):
                cur0[ce] = nxt0[ce]
            for ce in cutlass.range_constexpr(NQ):
                qkc[ce] = qkn[ce]
            for ce in cutlass.range_constexpr(GN):
                gtc[ce] = gtn[ce]
            for ce in cutlass.range_constexpr(2):
                gfc[ce] = gfn[ce]
            cute.arch.sync_threads()

        if const_expr(self.pdl and not self.early_trigger):
            cute.arch.griddepcontrol_launch_dependents()


# ---------------------------------------------------------------------------------------
# host side
# ---------------------------------------------------------------------------------------

_COMPILED: dict = {}


def parse_config(text: str) -> Optional[dict]:
    """'default', positional 'tpc,warps,pf,minb[,smem_tiles[,persistent[,occ]]]' or 'key=value,...'."""
    if text in ("", "default"):
        return None
    if "=" in text:
        out = {}
        for kv in text.split(","):
            k_, v_ = kv.split("=")
            if v_.lstrip("-").isdigit():
                out[k_] = int(v_)
            elif v_ in ("True", "False"):
                out[k_] = v_ == "True"
            else:
                out[k_] = v_
        for b_ in (
            "smem_tiles",
            "persistent",
            "spec_hist",
            "qk_first",
            "lo_split",
            "early_trigger",
            "state_before_wait",
            "early_acc",
        ):
            if b_ in out:
                out[b_] = bool(out[b_])
        return out
    f = [int(x) for x in text.split(",")]
    names = ("tpc", "warps", "pf", "minb", "smem_tiles", "persistent", "occ")
    out = dict(zip(names, f))
    for b_ in ("smem_tiles", "persistent"):
        if b_ in out:
            out[b_] = bool(out[b_])
    return out


def default_config(n: int, t: int, commit: bool = False) -> dict:
    """Launch mapping by request count, measured on GB300 at T = 4 (36 cold layers): small
    batches split a head's 24 m-tiles over more, smaller CTAs, every state tile by bulk copy
    issued as soon as the slot is known and the state dots taken before the history lands (a
    latency-bound launch: 1 request 3.3 -> 2.8 us); from 32 requests (or with the in-kernel
    commit from 16) CTAs of 8 m-tiles x 4 warps, capped at 5 (4 with the commit) resident per
    SM's register file, keep the most state in flight."""
    if n <= 2:
        return dict(tpc=2, warps=2, pf=0, minb=4, early_acc=True)
    if n <= 8:
        return dict(tpc=4, warps=4, pf=0, minb=4, early_acc=True)
    if commit:
        return dict(tpc=8, warps=4, pf=1, minb=4)
    if n <= 16:
        return dict(tpc=24, warps=8, pf=1, minb=2)
    return dict(tpc=8, warps=4, pf=1, minb=5)


def _compile(key, kern):
    fn = _COMPILED.get(key)
    if fn is None:
        dummy = torch.zeros(4, dtype=torch.int32, device="cuda")
        i64 = cutlass.Int64(0)
        fn = cute.compile(
            kern,
            from_dlpack(dummy, assumed_align=4).mark_layout_dynamic(leading_dim=0),
            *([i64] * 24),
            cutlass.Int32(1),
            cutlass.Float32(1.0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--opt-level 2 --enable-tvm-ffi",
        )
        _COMPILED[key] = fn
    return fn


def cached_replay_update(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ssm_states: torch.Tensor,
    state_indices: torch.Tensor,
    old_u: torch.Tensor,
    old_k: torch.Tensor,
    old_G: torch.Tensor,
    old_beta: Optional[torch.Tensor],
    cache_buf_idx: torch.Tensor,
    prev_num_accepted_tokens: torch.Tensor,
    history_size: int,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    launch_with_pdl: bool = False,
    replay_work_items: Optional[torch.Tensor] = None,
    n_writes: Optional[torch.Tensor] = None,
    packed_qkv: Optional[torch.Tensor] = None,
    use_all_layer_commit: bool = False,
    output: Optional[torch.Tensor] = None,
    in_kernel_commit: Optional[bool] = None,
    config: Optional[dict] = None,
    **ignored,
) -> torch.Tensor:
    """Same contract as `fused_recurrent_gated_delta_rule_cached_replay_update` (bf16 state).

    Without work items a wrapping row commits its checkpoint in-kernel, as the Triton op does;
    with work items + `use_all_layer_commit` it does not (the caller runs the all-layer commit).
    `in_kernel_commit` overrides that choice (True: commit in-kernel even when the batch is
    partitioned -- then the caller must NOT run the all-layer commit for this layer).
    Given work items, the kernel reads each row's (row, slot, pending count, half) from them (one
    load instead of two dependent ones), so they must list every one of the N rows, as the engine's
    builder over all decode rows does (`config={"use_wi": False}` falls back to `state_indices`).
    Work items with `in_kernel_commit=True` need no `use_all_layer_commit` (the caller runs none).
    The Triton launch knobs (`block_v`, `num_warps`, `use_l2_*`, `main_num_stages`) are ignored.
    """
    del old_beta
    N, T, Hq, Kd = k.shape
    assert (Hq, Kd) == (H, K) and v.shape[2:] == (HV, V) and history_size == HIST, "geometry"
    assert 1 <= T <= 8
    assert ssm_states.dtype == torch.bfloat16, "bf16 checkpoint state only"
    assert ssm_states.stride()[1:] == (V * K, K, 1)
    assert old_u.dtype == torch.bfloat16 and old_k.dtype == torch.bfloat16
    assert old_G.dtype == torch.float32
    for x in (old_u, old_k, old_G):
        assert x.is_contiguous()
    assert g.shape == (N, T, HV) and beta.shape == (N, T, HV)
    assert g.stride(2) == 1 and beta.stride(2) == 1
    assert g.stride(0) == T * g.stride(1) and beta.stride(0) == T * beta.stride(1)
    assert g.dtype in (torch.bfloat16, torch.float32) and beta.dtype in (
        torch.bfloat16,
        torch.float32,
    )
    if (replay_work_items is None) != (n_writes is None):
        raise ValueError("replay_work_items and n_writes must either both be set or both be None")
    if replay_work_items is not None and not use_all_layer_commit and not in_kernel_commit:
        raise ValueError("Partitioned replay requires the all-layer commit")
    if replay_work_items is None and use_all_layer_commit:
        raise ValueError("use_all_layer_commit requires replay work items")
    commit = replay_work_items is None if in_kernel_commit is None else bool(in_kernel_commit)
    if scale is None:
        scale = K**-0.5
    if output is None:
        output = torch.empty((N, T, HV, V), dtype=q.dtype, device=q.device)
    assert output.is_contiguous() and output.dtype == torch.bfloat16
    if N == 0:
        return output
    if packed_qkv is not None:
        assert packed_qkv.shape == (N * T, QKV_WIDTH) and packed_qkv.is_contiguous()
        base = packed_qkv.data_ptr()
        es = packed_qkv.element_size()
        qp, kp, vp = base, base + H * K * es, base + 2 * H * K * es
        qs = ks = vs = (T * QKV_WIDTH, QKV_WIDTH)
    else:
        for x, nh, d in ((q, H, K), (k, H, K), (v, HV, V)):
            assert x.stride(3) == 1 and x.stride(2) == d, "head rows must be dense"
        qp, kp, vp = q.data_ptr(), k.data_ptr(), v.data_ptr()
        qs, ks, vs = q.stride()[:2], k.stride()[:2], v.stride()[:2]
    gating = A_log is not None
    if gating:
        assert dt_bias is not None
        A_log = A_log.float().contiguous()
        dt_bias = dt_bias.float().contiguous()
        ap, dp = A_log.data_ptr(), dt_bias.data_ptr()
    else:
        ap = dp = 0
    cfg = dict(default_config(N, T, commit))
    if config:
        cfg.update(config)
    # with PDL, issue the state / history (not written by the previous kernel) before the wait;
    # without it, the step's own small inputs first
    cfg.setdefault("qk_first", not bool(launch_with_pdl))
    occ = cfg.pop("occ", 1)
    use_wi = (
        bool(cfg.pop("use_wi", True))
        and replay_work_items is not None
        and not cfg.get("persistent")
    )
    cfg["use_wi"] = use_wi
    if cfg.get("persistent"):
        sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        cfg["gmax"] = occ * sms
        items = N * H * (TILES // cfg["tpc"])
        assert items <= 256 * min(items, cfg["gmax"]), "more than 256 items per CTA"
    else:
        cfg.pop("gmax", None)
    pdl = bool(launch_with_pdl)
    kw = dict(
        T=T,
        commit=commit,
        pdl=pdl,
        l2norm=bool(use_qk_l2norm_in_kernel),
        gating=gating,
        g_f32=g.dtype == torch.float32,
        b_f32=beta.dtype == torch.float32,
        **cfg,
    )
    key = tuple(sorted(kw.items()))
    fn = _COMPILED.get(key)
    if fn is None:
        fn = _compile(key, _Replay(**kw))
    sidx = state_indices if state_indices.dtype == torch.int32 else state_indices.to(torch.int32)
    fn(
        sidx,
        qp,
        kp,
        vp,
        qs[0],
        qs[1],
        ks[0],
        ks[1],
        vs[0],
        vs[1],
        g.data_ptr(),
        g.stride(1),
        beta.data_ptr(),
        beta.stride(1),
        ap,
        dp,
        output.data_ptr(),
        ssm_states.data_ptr(),
        ssm_states.stride(0),
        prev_num_accepted_tokens.data_ptr(),
        cache_buf_idx.data_ptr(),
        old_u.data_ptr(),
        old_k.data_ptr(),
        old_G.data_ptr(),
        replay_work_items.data_ptr() if use_wi else 0,
        N,
        float(scale),
    )
    return output


class GdnCachedReplayCute(OpWrapper):
    """`cached_replay_update` as a catalog entry: one verify step of `N` sequences x `T` tokens
    through one GDN layer under the cached-replay state layout, bf16 checkpoint state only,
    with the same contract as `fused_recurrent_gated_delta_rule_cached_replay_update` (see that
    entry's docstring: history rebuild, gated delta rule over the T new tokens, history append,
    and -- `in_kernel_commit` -- the checkpoint advance of a wrapping row written in place).

    With `in_kernel_commit=True` every wrapping row commits here at any `N`: the caller must
    not also run the manager's all-layer commit (the checkpoint would advance twice).

    Fusion boundary: as the Triton entry's.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: the Triton entry's test harness (`_Bench`) spec -- path, n sequences x t tokens,
    # consecutive steps with random acceptance, pool slots, layers -- run with the in-kernel
    # commit at every step.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="MTP3 verify at con1: N=1, T=4, in-kernel commit",
            spec=dict(path="small", n=1, t=4, steps=6),
        ),
        Cell(
            why="MTP3 verify at con32: N=32, T=4, in-kernel commit",
            spec=dict(path="all_layer", n=32, t=4, steps=4, slots=72),
        ),
        Cell(
            why="MTP3 verify at con256: N=256, T=4, in-kernel commit",
            spec=dict(path="all_layer", n=256, t=4, steps=3, slots=300, layers=1),
        ),
    )

    note = """
    Metadata consumed: as the Triton entry (slots, per-slot pending count and history half),
    or the engine's replay work items (row, slot, pending count, half) over every row in one
    load each. Call site: `_replay_verify` over a bf16 recurrent pool, `in_kernel_commit=True`
    (with the work items the metadata builds, if given, and no all-layer commit); the target
    switches the manager's all-layer commit off when it binds a bf16 pool. `g` / `beta` may be
    the in_proj's column views (any row stride). Compiles one kernel per (T, launch mapping,
    commit, PDL, ...) on first use (5-10 s): warm-up runs before CUDA-graph capture cover it.
    Not supported: fp32 state, T > 8, another geometry. Optimized for T = 4; T = 1..8 are
    tested for correctness. Every launch mapping is bitwise every other (the per-row
    arithmetic does not depend on it).

    GB300, T = 4, in-kernel commit, 25 % of the rows wrapping, PDL off, a CUDA graph of 36
    layers each streaming its own state, us a layer: 1 request 2.8 (2.76 with the work
    items), 4: 4.8 (4.4), 32: 17.0 (16.2), 128: 52.3 (51.9), 256: 100.9 (99.6).
    """

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return cached_replay_update(*args, **kwargs)

    def reference(self, *args, **kwargs):
        raise NotImplementedError(
            "the multi-step cell harness drives the Triton entry's reference (see the test)"
        )


gdn_cached_replay_cute = GdnCachedReplayCute()
