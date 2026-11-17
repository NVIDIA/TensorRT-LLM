# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GDN cached-replay verify step on warp tensor cores (CuTe DSL), a drop-in for the Triton
`_cached_replay_kernel` (`fused_recurrent_gated_delta_rule_cached_replay_update`) at this
checkpoint's geometry (K=V=128, history 16, bf16 checkpoint state, T=1..8 new tokens), over
H q/k heads and HV = 3 H value heads: a whole layer's 16 / 48, or a tp4/tep4 rank's 4 / 12.

Tier 5 (CuTe DSL written for this checkpoint); the head counts are compile-time parameters of
`_Replay` (`h`, `hv`).

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
one load (scalar loads when the items carry the replay slot as a fifth field); otherwise slot, then
pending count / half. The history and its bookkeeping are addressed by a replay slot of their own
(`replay_indices`, the checkpoint slot when not given) at the slot strides of their buffers, so a
manager may keep them in pool pages apart from the checkpoint. With PDL the state and history (not written by
the previous kernel) are read before `griddepcontrol.wait`; without it the step's own small inputs
are issued first. Small batches (the launch is latency-bound there) bring every state tile in by
bulk copy as soon as the slot is known (`pf=0`: no tile held in registers through the prologue) and
take the tiles' state dots before the first barrier, while the history is still in flight
(`early_acc`).
"""

from __future__ import annotations

from typing import Optional

import cutlass
import cutlass.cute as cute
import torch
import torch.nn.functional as F
from cutlass import const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T as _ty
from cutlass.cutlass_dsl import dsl_user_op

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

K, V, HIST = 128, 128, 16
#: Value heads a q/k head serves. The q/k and value head counts themselves are compile-time
#: parameters (`_Replay(h=..., hv=...)`, from the call's shapes): `GEOMETRIES`.
GRP = 3
#: `(q/k heads, value heads)` the kernel is compiled for: a whole layer (an attention-DP
#: rank's), and a tp4/tep4 rank's quarter.
GEOMETRIES = ((16, 48), (4, 12))
TILES = GRP * (V // 16)  # m-tiles of one (request, q/k head)
PU = 272  # shared row pitch (bytes) of 128-element bf16 rows read by ldmatrix (conflict-free)
PK = 320  # shared row pitch of the history keys read as permuted 16-byte fragments
WI_REPLAY_SLOT = 4  # a work item's replay-slot field, when it has one (`REPLAY_WORK_REPLAY_SLOT`)
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
        spec_hist=False,
        qk_first=True,
        use_wi=False,
        wi_w=4,
        rsep=False,
        evict_first=True,
        early_acc=False,
        h=16,
        hv=48,
    ):
        assert 1 <= T <= 8
        assert (h, hv) in GEOMETRIES, (h, hv)
        self.H, self.HV = h, hv
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
        self.use_early = bool(early_acc)
        self.l2norm, self.gating, self.g_f32, self.b_f32 = l2norm, gating, g_f32, b_f32
        self.early_trigger, self.lo_split = early_trigger, lo_split
        self.state_before_wait = state_before_wait
        # tiles past the first `pf` of a warp: bulk-copied into shared memory at the start
        self.nsm = self.tpw - self.pf if smem_tiles else 0
        self.spec_hist = spec_hist
        self.qk_first = qk_first
        # (row, slot, pending count, half) per item from the engine's replay work items
        self.use_wi = use_wi
        # int32 fields per work item: 4 is one 16-byte load, anything else scalar loads
        self.wi_w = wi_w
        # the history and its bookkeeping have replay slots of their own (`replay_indices`)
        self.rsep = rsep
        self.evict_first = evict_first
        nset = 2 if self.spec_hist else 1
        self.nset = nset
        self.RP = (2 * self.TP + 4) * 4  # per-warp scratch row pitch (bytes)
        o = 0
        self.o_st = o
        o += warps * self.nsm * 4096
        self.o_mb = o
        o += -(-(warps * self.nsm * 8) // 16) * 16
        self.o_u = o
        o += nset * self.nhv * 16 * PU
        self.o_kh = o
        o += nset * 16 * PK
        self.o_kp = o
        o += 16 * PU if commit else 0
        self.o_v = o
        o += self.nhv * T * 256
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
        ridx_ptr: cutlass.Int64,
        u_ss: cutlass.Int64,
        k_ss: cutlass.Int64,
        g_ss: cutlass.Int64,
        pnat_ss: cutlass.Int64,
        buf_ss: cutlass.Int64,
        n_rows: cutlass.Int32,
        scale: cutlass.Float32,
        stream,
    ):
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
            ridx_ptr,
            u_ss,
            k_ss,
            g_ss,
            pnat_ss,
            buf_ss,
            scale,
        ).launch(
            grid=[n_rows * (self.H * self.S), 1, 1],
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

    def _work_item(self, wi_ptr, item):
        """Work item `item`'s int32 fields: (row, slot, pending count, half), then the replay
        slot when the history has slots of its own and the items carry them."""
        base = wi_ptr + cutlass.Int64(item) * (4 * self.wi_w)
        if self.wi_w == 4:
            return _LDG4["plain"](base)
        fields = WI_REPLAY_SLOT + 1 if self.rsep and self.wi_w > WI_REPLAY_SLOT else 4
        return [_ldg_i32(base + 4 * f) for f in range(fields)]

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
        ridx_ptr: cutlass.Int64,
        u_ss: cutlass.Int64,
        k_ss: cutlass.Int64,
        g_ss: cutlass.Int64,
        pnat_ss: cutlass.Int64,
        buf_ss: cutlass.Int64,
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
        H: cutlass.Constexpr = self.H
        HV: cutlass.Constexpr = self.HV

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
            wi = self._work_item(wi_ptr, nh // H)
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
        # the history and bookkeeping row: the checkpoint's slot unless it has its own
        rslot = slot
        if const_expr(self.rsep):
            if const_expr(self.use_wi and self.wi_w > WI_REPLAY_SLOT):
                rslot = wi[WI_REPLAY_SLOT]
            else:
                rslot = _ldg_i32(ridx_ptr + n64 * 4)
        if slot >= 0:
            slot64 = cutlass.Int64(slot)
            rs64 = cutlass.Int64(rslot)
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
                p = _ldg_i32(pnat_ptr + rs64 * pnat_ss)
                rb = _ldg_i32(buf_ptr + rs64 * buf_ss)
            wrap = (p + T) > HIST
            wb = cutlass.Int32(cutlass.select_(wrap, 1 - rb, rb))
            woff = cutlass.Int32(cutlass.select_(wrap, zi, p))
            # a replay slot's two history halves are packed behind its slot stride
            u_sl = ou_ptr + rs64 * u_ss
            k_sl = ok_ptr + rs64 * k_ss
            g_sl = og_ptr + rs64 * g_ss
            g_rd = g_sl + cutlass.Int64(rb) * (HV * HIST * 4)
            u_rd = u_sl + cutlass.Int64(rb) * (HIST * HV * V * 2)
            k_rd = k_sl + cutlass.Int64(rb) * (HIST * H * K * 2)
            u_wr = u_sl + cutlass.Int64(wb) * (HIST * HV * V * 2)
            k_wr = k_sl + cutlass.Int64(wb) * (HIST * H * K * 2)
            g_wr = g_sl + cutlass.Int64(wb) * (HV * HIST * 4)

            n_u: cutlass.Constexpr = NHV * 256
            if const_expr(self.spec_hist):
                # both halves, every row, before the pending count / half are known; rows j >= p
                # are masked where they are used
                for hb in cutlass.range_constexpr(2):
                    u_h = u_sl + hb * (HIST * HV * V * 2)
                    k_h = k_sl + hb * (HIST * H * K * 2)
                    g_h = g_sl + hb * (HV * HIST * 4)
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
                        a_hi = []
                        a_lo = []
                        for q4 in cutlass.range_constexpr(4):
                            dl = dc[0] if q4 < 2 else dc[2]
                            dhh = dc[1] if q4 < 2 else dc[3]
                            x0 = _bf2_lo(au[q4]) * dl
                            x1 = _bf2_hi(au[q4]) * dhh
                            hw = _bf2_pack(x0, x1)
                            a_hi.append(hw)
                            a_lo.append(_bf2_pack(x0 - _bf2_lo(hw), x1 - _bf2_hi(hw)))
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
                                dcm = _mma(a_hi, bf, [zf, zf, zf, zf])
                                if const_expr(self.lo_split):
                                    dcm = _mma(a_lo, bf, dcm)
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


# ---------------------------------------------------------------------------------------
# host side
# ---------------------------------------------------------------------------------------

_COMPILED: dict = {}


def default_config(n: int, t: int, commit: bool = False) -> dict:
    """Launch mapping by request count: small batches split a head's 24 m-tiles over more,
    smaller CTAs, every state tile by bulk copy issued as soon as the slot is known and the
    state dots taken before the history lands (a latency-bound launch); from 32 requests (or
    with the in-kernel commit from 16) CTAs of 8 m-tiles x 4 warps, capped at 5 (4 with the
    commit) resident per SM's register file, keep the most state in flight."""
    if n <= 2:
        return dict(tpc=2, warps=2, pf=0, minb=4, early_acc=True)
    if n <= 8:
        return dict(tpc=4, warps=4, pf=0, minb=4, early_acc=True)
    if commit:
        return dict(tpc=8, warps=4, pf=1, minb=4)
    if n <= 16:
        return dict(tpc=24, warps=8, pf=1, minb=2)
    return dict(tpc=8, warps=4, pf=1, minb=5)


def _kernel_kw(
    n: int,
    t: int,
    h: int,
    hv: int,
    *,
    commit: bool,
    pdl: bool,
    l2norm: bool,
    gating: bool,
    g_f32: bool,
    b_f32: bool,
    work_items: bool,
    config: Optional[dict] = None,
    work_item_width: int = 4,
    separate_replay_slots: bool = False,
) -> dict:
    """The kernel's compile-time parameters for a call of `n` requests: its key in `_COMPILED`.

    `work_item_width` is the int32 fields per work item; `separate_replay_slots` says the
    history is addressed by replay slots of its own (`replay_indices`)."""
    cfg = dict(default_config(n, t, commit))
    if config:
        cfg.update(config)
    # with PDL, issue the state / history (not written by the previous kernel) before the wait;
    # without it, the step's own small inputs first
    cfg.setdefault("qk_first", not pdl)
    cfg["use_wi"] = bool(cfg.pop("use_wi", True)) and work_items
    cfg["wi_w"] = work_item_width if cfg["use_wi"] else 4
    cfg["rsep"] = separate_replay_slots
    return dict(
        T=t,
        h=h,
        hv=hv,
        commit=commit,
        pdl=pdl,
        l2norm=l2norm,
        gating=gating,
        g_f32=g_f32,
        b_f32=b_f32,
        **cfg,
    )


def precompile(
    t: int,
    h: int,
    hv: int,
    max_requests: int,
    *,
    pdl: bool,
    partition_from: int,
    work_item_width: int = 4,
    separate_replay_slots: bool = False,
) -> int:
    """Compile every kernel a verify step of 1..`max_requests` requests launches, the way a
    target calls this entry: work items, and with them no in-kernel commit, from
    `partition_from` requests on; the q/k L2 norm and the gating in-kernel; bf16 `a` / `b`.
    A compile costs 5-10 s, so a target runs this at bind, before any step: the captured
    decode graphs do not reach every mapping (padded graphs skip 9 to 15 requests, which an
    eager mixed step carries). Returns the number of distinct kernels."""
    keys = set()
    for n in range(1, max_requests + 1):
        partitioned = n >= partition_from
        kw = _kernel_kw(
            n,
            t,
            h,
            hv,
            commit=not partitioned,
            pdl=pdl,
            l2norm=True,
            gating=True,
            g_f32=False,
            b_f32=False,
            work_items=partitioned,
            work_item_width=work_item_width,
            separate_replay_slots=separate_replay_slots,
        )
        key = tuple(sorted(kw.items()))
        if key not in keys:
            keys.add(key)
            _compile(key, _Replay(**kw))
    return len(keys)


def _compile(key, kern):
    fn = _COMPILED.get(key)
    if fn is None:
        dummy = torch.zeros(4, dtype=torch.int32, device="cuda")
        i64 = cutlass.Int64(0)
        fn = cute.compile(
            kern,
            from_dlpack(dummy, assumed_align=4).mark_layout_dynamic(leading_dim=0),
            *([i64] * 30),
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
    replay_indices: Optional[torch.Tensor] = None,
    **ignored,
) -> torch.Tensor:
    """Same contract as `fused_recurrent_gated_delta_rule_cached_replay_update` (bf16 state).

    `replay_indices` (`state_indices` when None) address the history (`old_u` / `old_k` /
    `old_G`) and its bookkeeping (`prev_num_accepted_tokens` / `cache_buf_idx`); each
    replay slot's rows are packed behind a slot stride of 16-byte multiples, and the
    bookkeeping may have any element stride (a manager keeps both in one row per slot).

    Without work items a wrapping row commits its checkpoint in-kernel, as the Triton op does;
    with work items + `use_all_layer_commit` it does not (the caller runs the all-layer commit).
    `in_kernel_commit` overrides that choice (True: commit in-kernel even when the batch is
    partitioned -- then the caller must NOT run the all-layer commit for this layer).
    Given work items, the kernel reads each row's (row, slot, pending count, half) from them (one
    load instead of two dependent ones), so they must list every one of the N rows, as the engine's
    builder over all decode rows does (`config={"use_wi": False}` falls back to `state_indices`).
    Work items with `in_kernel_commit=True` need no `use_all_layer_commit` (the caller runs none);
    with separate replay slots their fifth field (`REPLAY_WORK_REPLAY_SLOT`) gives each row's
    replay slot, else `replay_indices` does.
    The Triton launch knobs (`block_v`, `num_warps`, `use_l2_*`, `main_num_stages`) are ignored.
    """
    del old_beta
    N, T, H, Kd = k.shape
    HV = v.shape[2]
    assert (H, HV) in GEOMETRIES and Kd == K and v.shape[3] == V and history_size == HIST, (
        f"geometry: {H} q/k x {Kd}, {HV} v x {v.shape[3]}, history {history_size}"
    )
    qkv_width = 2 * H * K + HV * V
    assert 1 <= T <= 8
    assert ssm_states.dtype == torch.bfloat16, "bf16 checkpoint state only"
    assert ssm_states.stride()[1:] == (V * K, K, 1)
    assert old_u.dtype == torch.bfloat16 and old_k.dtype == torch.bfloat16
    assert old_G.dtype == torch.float32
    for name, x, rows in (
        ("old_u", old_u, (2, HIST, HV, V)),
        ("old_k", old_k, (2, HIST, H, K)),
        ("old_G", old_G, (2, HV, HIST)),
    ):
        assert tuple(x.shape[1:]) == rows and x[0].is_contiguous(), (
            f"{name} is {tuple(x.shape)} with strides {x.stride()}; the kernel reads packed "
            f"[slots, {', '.join(map(str, rows))}] rows"
        )
        assert x.data_ptr() % 16 == 0 and (x.stride(0) * x.element_size()) % 16 == 0, (
            f"{name}: the kernel copies 16-byte chunks, so its base and slot stride must be "
            "16-byte aligned"
        )
    for name, x in (
        ("prev_num_accepted_tokens", prev_num_accepted_tokens),
        ("cache_buf_idx", cache_buf_idx),
    ):
        assert x.dim() == 1 and x.dtype == torch.int32, f"{name}: one int32 per replay slot"
    rsep = replay_indices is not None and replay_indices.data_ptr() != state_indices.data_ptr()
    if rsep:
        assert (
            replay_indices.dim() == 1
            and replay_indices.shape[0] >= N
            and replay_indices.dtype == torch.int32
            and replay_indices.is_contiguous()
        ), "replay_indices: one contiguous int32 replay slot per row"
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
        assert packed_qkv.shape == (N * T, qkv_width) and packed_qkv.is_contiguous()
        base = packed_qkv.data_ptr()
        es = packed_qkv.element_size()
        qp, kp, vp = base, base + H * K * es, base + 2 * H * K * es
        qs = ks = vs = (T * qkv_width, qkv_width)
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
    if replay_work_items is not None:
        assert (
            replay_work_items.dim() == 2
            and replay_work_items.shape[1] >= 4
            and replay_work_items.dtype == torch.int32
            and replay_work_items.is_contiguous()
        ), "replay_work_items: contiguous int32 rows of (row, slot, pending count, half[, ...])"
    kw = _kernel_kw(
        N,
        T,
        H,
        HV,
        commit=commit,
        pdl=bool(launch_with_pdl),
        l2norm=bool(use_qk_l2norm_in_kernel),
        gating=gating,
        g_f32=g.dtype == torch.float32,
        b_f32=beta.dtype == torch.float32,
        work_items=replay_work_items is not None,
        config=config,
        work_item_width=replay_work_items.shape[1] if replay_work_items is not None else 4,
        separate_replay_slots=rsep,
    )
    use_wi = kw["use_wi"]
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
        replay_indices.data_ptr() if rsep else 0,
        old_u.stride(0) * old_u.element_size(),
        old_k.stride(0) * old_k.element_size(),
        old_G.stride(0) * old_G.element_size(),
        prev_num_accepted_tokens.stride(0) * prev_num_accepted_tokens.element_size(),
        cache_buf_idx.stride(0) * cache_buf_idx.element_size(),
        N,
        float(scale),
    )
    return output


#: A tp4/tep4 rank's GDN heads: 16 key / 48 value split four ways.
RANK_K_HEADS, RANK_V_HEADS = 4, 12
#: A tp1 or dep4 rank's GDN heads: the whole layer's.
ONE_RANK_K_HEADS, ONE_RANK_V_HEADS = 16, 48
#: The stock tep4 engine's recurrent-state slots a rank (bf16 state), and the GDN layers whose
#: states share one slot page in the stock manager: the slot stride is 36 layers' states.
TEP4_SLOTS, PAGE_LAYERS = 1874, 36


def _tep4_cell(why, path, n, t, **spec) -> Cell:
    """A tp4/tep4 rank's heads over the stock tep4 engine's bf16 state page."""
    return Cell(
        why=f"tep4 {why}",
        spec=dict(
            path=path,
            n=n,
            t=t,
            state_dtype=torch.bfloat16,
            H=RANK_K_HEADS,
            HV=RANK_V_HEADS,
            history_size=HIST,
            slots=TEP4_SLOTS,
            page_layers=PAGE_LAYERS,
            **spec,
        ),
    )


def _whole_layer_cell(why, path, n, t, **spec) -> Cell:
    """The whole layer's heads (tp1, dep4) over a bf16 state pool."""
    return Cell(
        why=f"16/48 {why}",
        spec=dict(
            path=path,
            n=n,
            t=t,
            state_dtype=torch.bfloat16,
            H=ONE_RANK_K_HEADS,
            HV=ONE_RANK_V_HEADS,
            history_size=HIST,
            **spec,
        ),
    )


class GdnCachedReplayCute(OpWrapper):
    """`cached_replay_update` as a catalog entry: one verify step of `N` sequences x `T` tokens
    through one GDN layer under the cached-replay state layout, bf16 checkpoint state only, with
    the contract of `fused_recurrent_gated_delta_rule_cached_replay_update` (history rebuild,
    gated delta rule over the T new tokens, history append, and the checkpoint advance of a
    wrapping row written in place when no work items are given -- or, `in_kernel_commit=True`,
    always). With work items and `use_all_layer_commit` it commits nothing (the manager's
    all-layer commit does), exactly as the Triton entry.

    The q/k and value head counts are compile-time parameters taken from the call's shapes
    (`GEOMETRIES`: a whole layer's 16 / 48 and a tp4/tep4 rank's 4 / 12), and `g` / `beta` may
    be strided column views of the input projection (any row stride, unit head stride).

    Fusion boundary: as the Triton entry's.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: the Triton entry's test harness (`_Bench`) spec -- path, n sequences x t
    # tokens, consecutive steps with random acceptance, start, accept, dup, H / HV, pool slots,
    # page_layers (the stock manager's coalesced state page) -- plus `strided_gates` (g / beta
    # handed over as column views of a `[N*T, conv + z + 2 HV]` projection, as the target does).
    CELLS: tuple[Cell, ...] = (
        _tep4_cell(
            "small n=1 T=4: MTP3 verify at con1, in-kernel commit on the page's last slot",
            "small",
            1,
            4,
            steps=8,
            strided_gates=True,
        ),
        _tep4_cell(
            "small n=4 T=4: the largest captured ll batch",
            "small",
            4,
            4,
            steps=6,
            strided_gates=True,
        ),
        _tep4_cell(
            "small n=4 T=4 dup: rows 2..3 share one CUDA-graph padding slot",
            "small",
            4,
            4,
            steps=6,
            dup=(2, 3),
            strided_gates=True,
        ),
        _tep4_cell("small n=2 T=2: MTP1 verify", "small", 2, 2, steps=8, strided_gates=True),
        _tep4_cell(
            "small n=1 T=8 full accept: MTP7 appends to the history's end, then wraps",
            "small",
            1,
            8,
            steps=8,
            start="fresh",
            accept="full",
            strided_gates=True,
        ),
        _tep4_cell(
            "all_layer n=16 T=4: the smallest partitioned batch, all-layer commit",
            "all_layer",
            16,
            4,
            steps=6,
            strided_gates=True,
        ),
        _tep4_cell("all_layer n=128 T=4: a large partitioned batch", "all_layer", 128, 4, steps=4),
        _whole_layer_cell(
            "small n=1 T=3: MTP2 verify at con1 (the tp1 AgentX engines), in-kernel commit",
            "small",
            1,
            3,
            steps=6,
            strided_gates=True,
        ),
        _whole_layer_cell(
            "small n=1 T=4: MTP3 verify at con1, in-kernel commit",
            "small",
            1,
            4,
            steps=6,
            strided_gates=True,
        ),
        _whole_layer_cell(
            "all_layer n=32 T=4: work items + the all-layer commit",
            "all_layer",
            32,
            4,
            steps=4,
            slots=72,
            strided_gates=True,
        ),
        _whole_layer_cell(
            "all_layer n=256 T=4: a large partitioned batch",
            "all_layer",
            256,
            4,
            steps=3,
            slots=300,
            layers=1,
        ),
    )

    note = """
    Call site: `_gdn_verify` (`DecodeTarget._gdn_block` / `PrefillTarget._gdn_block` of a verify
    step) over the bf16 recurrent pool, at a tp4/tep4 rank's 4 / 12 heads (tp1 and dep4: the
    whole layer's 16 / 48): q/k/v from the conv's fresh contiguous `[N*T, 2560]` rows
    (`packed_qkv`; 10240 at 16 / 48), `g` / `beta` the input projection's raw
    a / b columns read through its row stride (no copies), `A_log` / `dt_bias` fp32, the
    layer's view of the stock manager's bf16 state page (slot stride 36 layers' states), the
    history buffers `[slots, 2, 16, 12|4, 128]` / `[slots, 2, 12, 16]` (addressed by the
    manager's replay slots, `replay_indices`, when it keeps the history in slots of its own;
    the work items then carry the replay slot as a fifth field), and work items +
    `use_all_layer_commit` exactly when the Triton entry gets them (N >= 16 with the manager's
    all-layer commit on). `launch_with_pdl=True`: the Triton conv before it signals its
    dependents at its start (`launch_dependent_kernels`), so the kernel streams the checkpoint
    and the history -- which nothing in the step writes ahead of it -- while the conv runs, and
    reads q/k/v, a and b only after `griddepcontrol.wait`. Compiles one kernel per (geometry,
    T, launch mapping, commit, PDL, ...) on first use (5-10 s); `precompile` builds every one a
    target's verify steps reach, and the target runs it at bind (the captured decode graphs
    alone skip 9 to 15 requests, which an eager mixed step carries).

    Certified at the whole layer's 16 / 48 heads (tp1, dep4) over a bf16 state: the in-kernel
    commit at N = 1 and T = 3, 4, the all-layer commit at N = 32 and 256 (T = 4). Certified at
    4 / 12 heads over a bf16 checkpoint (cells: the stock tep4 engine's 36-layer
    state page at 1874 slots, rows on both ends of it, several consecutive verify steps from
    warm histories with random acceptance): the in-kernel commit at N = 1, 2, 4 and T = 2, 4,
    8 (T = 8 with full acceptance appending to the history's end, then wrapping), CUDA-graph
    padding rows sharing one slot, and the all-layer commit at N = 16 and 128 (T = 4); g / beta
    as column views of a `[N*T, 4120]` projection. Against the Triton entry's bands (fractions
    of each gate): output elements <= 0.50 of 4 bf16 ulps of their token block, relative RMS
    <= 0.71 of 1.5 ulps, bf16 checkpoint <= 0.68 of 2 ulps of its head's absmax, `old_u` <=
    0.50 of 4 ulps, `old_G` <= 0.06 -- the Triton entry's own margins (0.52 / 0.64 / 0.84 /
    0.50 / 0.08). A reference with the key heads reversed is refused by the output gate.
    CUDA-graph replay (PDL on and off, N = 1, 4 and 16, at both geometries) is bitwise eager,
    and the 4 / 12 launch mappings are bitwise each other at N = 1, 2, 4 and T = 2, 4, 8.

    Numerics: the Triton kernel's rounding points (bf16 L2-normalized q / k and history
    coefficients into the tensor-core dots, fp32 recurrence, fp32 commit rounded once to the
    bf16 pool, bf16 history u / k, fp32 G); the dot accumulation order differs, so outputs
    match the Triton kernel to a few bf16 ulps, not bitwise. Every launch mapping is bitwise
    every other (the per-row arithmetic does not depend on it).

    Not supported: fp32 state, T > 8, a geometry outside `GEOMETRIES`, K or V other than 128,
    history other than 16, pre-gated g / beta (`A_log=None` is compiled but not certified).
    The Triton launch knobs (`block_v`, `num_warps`, `use_l2_*`, `main_num_stages`) are ignored.
    """

    def __call__(
        self,
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
        old_beta: torch.Tensor,
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
        block_v: Optional[int] = None,
        num_warps: Optional[int] = None,
        use_l2_state_cache: Optional[bool] = None,
        use_l2_streaming_inputs: Optional[bool] = None,
        use_l2_shared_inputs: Optional[bool] = None,
        main_num_stages: Optional[int] = None,
        packed_qkv: Optional[torch.Tensor] = None,
        use_all_layer_commit: bool = False,
        output: Optional[torch.Tensor] = None,
        in_kernel_commit: Optional[bool] = None,
        config: Optional[dict] = None,
        replay_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return cached_replay_update(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            ssm_states=ssm_states,
            state_indices=state_indices,
            old_u=old_u,
            old_k=old_k,
            old_G=old_G,
            old_beta=old_beta,
            cache_buf_idx=cache_buf_idx,
            prev_num_accepted_tokens=prev_num_accepted_tokens,
            history_size=history_size,
            scale=scale,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            A_log=A_log,
            dt_bias=dt_bias,
            launch_with_pdl=launch_with_pdl,
            replay_work_items=replay_work_items,
            n_writes=n_writes,
            packed_qkv=packed_qkv,
            use_all_layer_commit=use_all_layer_commit,
            output=output,
            in_kernel_commit=in_kernel_commit,
            config=config,
            replay_indices=replay_indices,
        )

    def reference(
        self,
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
        old_beta: torch.Tensor,
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
        block_v: Optional[int] = None,
        num_warps: Optional[int] = None,
        use_l2_state_cache: Optional[bool] = None,
        use_l2_streaming_inputs: Optional[bool] = None,
        use_l2_shared_inputs: Optional[bool] = None,
        main_num_stages: Optional[int] = None,
        packed_qkv: Optional[torch.Tensor] = None,
        use_all_layer_commit: bool = False,
        output: Optional[torch.Tensor] = None,
        in_kernel_commit: Optional[bool] = None,
        config: Optional[dict] = None,
        replay_indices: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(out, ssm_states, old_u, old_k, old_G)` as the call leaves them;
        mutates nothing. The state is rebuilt from the checkpoint and history
        in fp32 and the new tokens are stepped one at a time in fp32 -- no
        matmul, so no TF32 -- and every store is rounded once to its buffer's
        dtype. The history is read at whatever dtype and depth its buffers
        have, so the same function driven on fp32 history deeper than any run
        ever fills is the exact token-by-token recurrence.

        Rows with a negative slot keep `output`'s prior content, or are NaN
        (unspecified) when no `output` is given. `old_beta`, `launch_with_pdl`,
        `n_writes` and the launch-tuning knobs are ignored; `replay_work_items`
        only decides whether a wrap commits the checkpoint here, unless `in_kernel_commit`
        says (True: every wrap commits here; False: none does); `config` is ignored."""
        N, T, H, K = k.shape
        HV, V = v.shape[2], v.shape[3]
        rep = HV // H
        dev = v.device
        scale = K**-0.5 if scale is None else scale
        if packed_qkv is not None:
            rows = packed_qkv.view(N, T, -1)
            q = rows[..., : H * K].view(N, T, H, K)
            k = rows[..., H * K : 2 * H * K].view(N, T, H, K)
            v = rows[..., 2 * H * K :].view(N, T, HV, V)
        qf, kf, vf = q.float(), k.float(), v.float()
        if use_qk_l2norm_in_kernel:
            qf = qf / torch.sqrt(qf.square().sum(-1, keepdim=True) + 1e-6)
            kf = kf / torch.sqrt(kf.square().sum(-1, keepdim=True) + 1e-6)
        qf = (qf * scale).repeat_interleave(rep, dim=2)  # [N, T, HV, K]
        kv = kf.repeat_interleave(rep, dim=2)  # [N, T, HV, K]
        if A_log is not None:
            gf = -torch.exp(A_log.float()) * F.softplus(g.float() + dt_bias.float())
            bf = torch.sigmoid(beta.float())
        else:
            gf, bf = g.float(), beta.float()

        out_dtype = q.dtype if output is None else output.dtype
        out = (
            torch.full((N, T, HV, V), float("nan"), device=dev)
            if output is None
            else output.float().clone()
        )
        pool, u_buf, k_buf, G_buf = (x.clone() for x in (ssm_states, old_u, old_k, old_G))
        live = (state_indices[:N] >= 0).nonzero().flatten()
        if live.numel() == 0:
            return out.to(out_dtype), pool, u_buf, k_buf, G_buf
        slot = state_indices[:N][live].long()
        rslot = slot if replay_indices is None else replay_indices[:N][live].long()
        p = prev_num_accepted_tokens[rslot].long()
        b = cache_buf_idx[rslot].long()
        M, depth = live.numel(), old_u.shape[2]

        # 1. the pending state: checkpoint plus the p history entries of half b
        pending = torch.arange(depth, device=dev)[None, :] < p[:, None]  # [M, depth]
        Gh = torch.where(pending[:, None, :], old_G[rslot, b].float(), 0.0)  # [M, HV, depth]
        Uh = torch.where(pending[:, :, None, None], old_u[rslot, b].float(), 0.0)
        Kh = torch.where(pending[:, :, None, None], old_k[rslot, b].float(), 0.0)
        Kh = Kh.repeat_interleave(rep, dim=2)  # [M, depth, HV, K]
        last = (p - 1).clamp(min=0)[:, None, None].expand(M, HV, 1)
        g_start = torch.where(p[:, None] > 0, Gh.gather(2, last).squeeze(2), 0.0)  # [M, HV]
        weight = torch.where(pending[:, None, :], torch.exp(g_start[:, :, None] - Gh), 0.0)
        h = torch.exp(g_start)[:, :, None, None] * ssm_states[slot].float()  # [M, HV, V, K]
        for j in range(depth):
            h = h + weight[:, :, j, None, None] * Uh[:, j, :, :, None] * Kh[:, j, :, None, :]
        checkpoint = h

        # 2. the new tokens, in order
        gl, bl = gf[live], bf[live]  # [M, T, HV]
        ql, kl, vl = qf[live], kv[live], vf[live]
        G_local = torch.zeros(M, HV, device=dev)
        u_new, G_new = [], []
        for t in range(T):
            h = h * torch.exp(gl[:, t])[:, :, None, None]
            u = bl[:, t, :, None] * (vl[:, t] - (h * kl[:, t, :, None, :]).sum(-1))  # [M, HV, V]
            h = h + u[:, :, :, None] * kl[:, t, :, None, :]
            out[live, t] = (h * ql[:, t, :, None, :]).sum(-1)
            G_local = G_local + gl[:, t]
            u_new.append(u)
            G_new.append(G_local.clone())

        # 3. the history append, and the checkpoint commit on a wrap
        wrap = p + T > history_size
        w_buf = torch.where(wrap, 1 - b, b)
        w_off = torch.where(wrap, 0, p)
        for t in range(T):
            pos = w_off + t
            u_buf[rslot, w_buf, pos] = u_new[t].to(u_buf.dtype)
            k_buf[rslot, w_buf, pos] = kf[live, t].to(k_buf.dtype)
            G_store = torch.where(wrap[:, None], G_new[t], g_start + G_new[t])
            G_buf[rslot, w_buf, :, pos] = G_store.to(G_buf.dtype)
        commit = replay_work_items is None if in_kernel_commit is None else bool(in_kernel_commit)
        if commit and bool(wrap.any()):
            pool[slot[wrap]] = checkpoint[wrap].to(pool.dtype)
        return out.to(out_dtype), pool, u_buf, k_buf, G_buf

    def is_valid(
        self,
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
        old_beta: torch.Tensor,
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
        block_v: Optional[int] = None,
        num_warps: Optional[int] = None,
        use_l2_state_cache: Optional[bool] = None,
        use_l2_streaming_inputs: Optional[bool] = None,
        use_l2_shared_inputs: Optional[bool] = None,
        main_num_stages: Optional[int] = None,
        packed_qkv: Optional[torch.Tensor] = None,
        use_all_layer_commit: bool = False,
        output: Optional[torch.Tensor] = None,
        in_kernel_commit: Optional[bool] = None,
        config: Optional[dict] = None,
        replay_indices: Optional[torch.Tensor] = None,
    ) -> None:
        N, _, H, K = k.shape
        HV, V = v.shape[2], v.shape[3]
        assert state_indices.dim() == 1 and state_indices.shape[0] >= N, (
            f"state_indices has shape {tuple(state_indices.shape)} for {N} sequences -- the "
            "kernel reads one slot per sequence with no bound, so a shorter tensor hands the "
            "trailing rows whatever follows it in memory, and they write that slot's history"
        )
        # The history is addressed by the replay slots, the state slots when none are given.
        if replay_indices is None:
            history_slots, slot_space = ssm_states.shape[0], "the recurrent pool"
        else:
            assert replay_indices.dim() == 1 and replay_indices.shape[0] >= N, (
                f"replay_indices has shape {tuple(replay_indices.shape)} for {N} sequences -- "
                "the kernel reads one replay slot per sequence with no bound"
            )
            history_slots, slot_space = int(replay_indices[:N].max()) + 1, "the replay slots"
        slots = old_u.shape[0]
        assert slots >= history_slots, (
            f"the history buffers have {slots} slots for {history_slots} of {slot_space} -- a "
            "slot past the history's end is read and written out of bounds"
        )
        for name, tensor in (
            ("prev_num_accepted_tokens", prev_num_accepted_tokens),
            ("cache_buf_idx", cache_buf_idx),
        ):
            assert tensor.dim() == 1 and tensor.shape[0] >= history_slots, (
                f"{name} has shape {tuple(tensor.shape)}; the kernel reads one element per "
                f"history slot ({history_slots} of {slot_space}), at the slot's index times "
                "the element stride"
            )
        expected = {
            "old_u": (old_u, (slots, 2, history_size, HV, V)),
            "old_k": (old_k, (slots, 2, history_size, H, K)),
            "old_G": (old_G, (slots, 2, HV, history_size)),
        }
        for name, (tensor, shape) in expected.items():
            assert tuple(tensor.shape) == shape, (
                f"{name} has shape {tuple(tensor.shape)}; history_size={history_size} and this "
                f"geometry need {shape} -- the kernel addresses the history by history_size and "
                "the head geometry, never by the buffer's own shape"
            )


gdn_cached_replay_cute = GdnCachedReplayCute()
