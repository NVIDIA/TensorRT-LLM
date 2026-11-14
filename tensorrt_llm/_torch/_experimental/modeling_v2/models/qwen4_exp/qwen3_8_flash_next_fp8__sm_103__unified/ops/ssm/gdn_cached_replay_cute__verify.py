# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GDN cached-replay verify step on warp tensor cores (CuTe DSL): the Triton
`_cached_replay_kernel` (`fused_recurrent_gated_delta_rule_cached_replay_update`) at this
checkpoint's geometry (K=V=128, history 16, bf16 state, T=1..8 new tokens) over H q/k heads and
HV = 3 H value heads (`GEOMETRIES`, compile-time parameters of `_Replay`).

A CTA serves one (request, q/k head, split): the q/k head's 384 state rows (3 v heads x 128, K
innermost) are 24 m-tiles of 16 rows, `tpc` of them a CTA. Per m-tile a warp computes the state
dots D[v, k_t] = h0[v, :] . k_t and D[v, q_t] = h0[v, :] . q_t on mma.sync m16n8k16 (bf16 in,
fp32 out), and the history term as one more MMA with the [16 x 2T] coefficients rounded to bf16
as the Triton kernel rounds them; the T x T forward substitution runs per state row on CUDA
cores. With `commit`, a row whose history wraps (`p + T > 16`) writes the advanced checkpoint over
the tile it just used (u split into bf16 hi and lo, as the Triton commit does).

With PDL the state and history (not written by the previous kernel) are read before
`griddepcontrol.wait`, the step's own inputs after it. The history and its bookkeeping are
addressed by a replay slot of their own (`replay_indices`, the checkpoint slot when not given) at
the slot strides of their buffers, so a manager may keep them in pool pages apart from the
checkpoint.
"""

from __future__ import annotations

from typing import Callable, Optional

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
#: Value heads a q/k head serves.
GRP = 3
#: `(q/k heads, value heads)` compiled for: a whole layer (tp1, attention DP), a tp2/tep2 and a
#: tp4/tep4 rank.
GEOMETRIES = ((16, 48), (8, 24), (4, 12))
TILES = GRP * (V // 16)  # m-tiles of one (request, q/k head)
PU = 272  # shared row pitch (bytes) of 128-element bf16 rows read by ldmatrix (conflict-free)
PK = 320  # shared row pitch of the history keys read as permuted 16-byte fragments
WI_REPLAY_SLOT = 4  # a work item's replay-slot field, when it has one (`REPLAY_WORK_REPLAY_SLOT`)
LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453

# -- inline PTX helpers ------------------------------------------------------------------------


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
    """16 bits zero-extended (0 where `pred` is 0), converted where used so the load does not stall."""
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
    """`_bf2_lo` pinned in program order, so the conversion of a prefetched word is not hoisted
    up to its load."""
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
def _mul_bf16(a, b, *, loc=None, ip=None):
    """bf16 x bf16 rounded to bf16 (`mul.rn.bf16`), bf16 bits in the low half of an i32."""
    return cutlass.Int32(
        _asm(
            _ty.i32(),
            [_ir(a, cutlass.Int32, loc, ip), _ir(b, cutlass.Int32, loc, ip)],
            "{ .reg .b16 x, y, z; cvt.u16.u32 x, $1; cvt.u16.u32 y, $2; mul.rn.bf16 z, x, y; "
            "cvt.u32.u16 $0, z; }",
            "=r,r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _conv4_silu(p0, p1, p2, p3, *, loc=None, ip=None):
    """The verify conv's per-element tail in the Triton `_causal_conv1d_update_kernel`'s PTX:
    the bf16 tap products summed in fp32 in tap order, SiLU as `acc / (1 + ex2.approx(acc *
    -log2e))` (`div.full.f32`), rounded to bf16 (bits in the low half of an i32)."""
    return cutlass.Int32(
        _asm(
            _ty.i32(),
            [_ir(x, cutlass.Int32, loc, ip) for x in (p0, p1, p2, p3)],
            "{ .reg .b16 h0, h1, h2, h3, o; .reg .f32 a, t, e, d, y; "
            "cvt.u16.u32 h0, $1; cvt.u16.u32 h1, $2; cvt.u16.u32 h2, $3; cvt.u16.u32 h3, $4; "
            "cvt.f32.bf16 a, h0; add.f32 a, a, 0f00000000; add.rn.f32.bf16 a, h1, a; "
            "add.rn.f32.bf16 a, h2, a; add.rn.f32.bf16 a, h3, a; "
            "mul.f32 t, a, 0fBFB8AA3B; ex2.approx.f32 e, t; add.f32 d, e, 0f3F800000; "
            "div.full.f32 y, a, d; cvt.rn.bf16.f32 o, y; cvt.u32.u16 $0, o; }",
            "=r,r,r,r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _sts4(saddr, a, b, c, d, *, loc=None, ip=None):
    _asm(
        _ty.i32(),
        [_ir(saddr, cutlass.Int32, loc, ip)]
        + [_ir(x, cutlass.Int32, loc, ip) for x in (a, b, c, d)],
        "{ st.shared.v4.b32 [$1], {$2, $3, $4, $5}; mov.u32 $0, 0; }",
        "=r,r,r,r,r,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _pack16(lo, hi, *, loc=None, ip=None):
    """Two bf16 bit patterns (low halves of i32) packed into one word, `lo` in the low half."""
    return cutlass.Int32(
        _asm(
            _ty.i32(),
            [_ir(lo, cutlass.Int32, loc, ip), _ir(hi, cutlass.Int32, loc, ip)],
            "prmt.b32 $0, $1, $2, 0x5410;",
            "=r,r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _half16(w, hi, *, loc=None, ip=None):
    """The low (`hi` False) or high half of a word, zero-extended."""
    sel = "0x4432" if hi else "0x4410"
    return cutlass.Int32(
        _asm(
            _ty.i32(),
            [_ir(w, cutlass.Int32, loc, ip)],
            f"prmt.b32 $0, $1, 0, {sel};",
            "=r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _stg_b16(gaddr, x, *, loc=None, ip=None):
    _asm(
        _ty.i32(),
        [_ir(gaddr, cutlass.Int64, loc, ip), _ir(x, cutlass.Int32, loc, ip)],
        "{ .reg .b16 h; cvt.u16.u32 h, $2; st.global.b16 [$1], h; mov.u32 $0, 0; }",
        "=r,l,r",
        loc,
        ip,
        side=True,
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


# -- the folded gated RMS norm (`_Replay(norm=True)`): the instructions of the Triton
# `_rms_norm_gated_fwd_multirow_kernel` (one row a warp, four columns a lane), so its bits.


@dsl_user_op
def _ldg2_i32(gaddr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([_ty.i32()] * 2),
        [_ir(gaddr, cutlass.Int64, loc, ip)],
        "ld.global.v2.b32 {$0, $1}, [$2];",
        "=r,=r,l",
        loc,
        ip,
        side=True,
    )
    return _unpack(res, 2, cutlass.Int32, loc, ip)


@dsl_user_op
def _lds2_i32(saddr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([_ty.i32()] * 2),
        [_ir(saddr, cutlass.Int32, loc, ip)],
        "ld.shared.v2.b32 {$0, $1}, [$2];",
        "=r,=r,r",
        loc,
        ip,
        side=True,
    )
    return _unpack(res, 2, cutlass.Int32, loc, ip)


@dsl_user_op
def _stg2_i32(gaddr, a, b, *, loc=None, ip=None):
    _asm(
        _ty.i32(),
        [
            _ir(gaddr, cutlass.Int64, loc, ip),
            _ir(a, cutlass.Int32, loc, ip),
            _ir(b, cutlass.Int32, loc, ip),
        ],
        "{ st.global.v2.b32 [$1], {$2, $3}; mov.u32 $0, 0; }",
        "=r,l,r,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _gn_sumsq(x0, x1, x2, x3, *, loc=None, ip=None):
    """A lane's sum of squares in the Triton kernel's order: (x0 x0 + x2 x2) + (x1 x1 + x3 x3)."""
    return cutlass.Float32(
        _asm(
            _ty.f32(),
            [_ir(x, cutlass.Float32, loc, ip) for x in (x0, x1, x2, x3)],
            "{ .reg .f32 p2, p3, a, b; mul.rn.f32 p2, $3, $3; mul.rn.f32 p3, $4, $4; "
            "fma.rn.f32 a, $1, $1, p2; fma.rn.f32 b, $2, $2, p3; add.rn.f32 $0, a, b; }",
            "=f,f,f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _add_rn(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            _ty.f32(),
            [_ir(a, cutlass.Float32, loc, ip), _ir(b, cutlass.Float32, loc, ip)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _mul_rn(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            _ty.f32(),
            [_ir(a, cutlass.Float32, loc, ip), _ir(b, cutlass.Float32, loc, ip)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _gn_rstd(sumsq, eps, *, loc=None, ip=None):
    """1 / sqrt(sum / 128 + eps): `div.full.f32`, `add.f32`, `sqrt.approx.ftz.f32`, `div.full.f32`."""
    return cutlass.Float32(
        _asm(
            _ty.f32(),
            [_ir(sumsq, cutlass.Float32, loc, ip), _ir(eps, cutlass.Float32, loc, ip)],
            "{ .reg .f32 n, v, a, q, one; mov.b32 n, 0f43000000; mov.b32 one, 0f3F800000; "
            "div.full.f32 v, $1, n; add.rn.f32 a, $2, v; sqrt.approx.ftz.f32 q, a; "
            "div.full.f32 $0, one, q; }",
            "=f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _gn_sigmoid(z, *, loc=None, ip=None):
    """sigmoid(z) as Triton's `tl.sigmoid`: 1 / (1 + ex2.approx(z * -log2(e)))."""
    return cutlass.Float32(
        _asm(
            _ty.f32(),
            [_ir(z, cutlass.Float32, loc, ip)],
            "{ .reg .f32 t, e, d, one; mov.b32 one, 0f3F800000; mul.rn.f32 t, $1, 0fBFB8AA3B; "
            "ex2.approx.f32 e, t; add.rn.f32 d, e, one; div.full.f32 $0, one, d; }",
            "=f,f",
            loc,
            ip,
        )
    )


def _exp(x):
    return _ex2(x * LOG2E)


def _shfl_xor(x, m):
    return cute.arch.shuffle_sync_bfly(x, offset=m)


# -- the kernel --------------------------------------------------------------------------------


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
        fold=False,
        norm=False,
        norm_eps=1e-6,
    ):
        assert 1 <= T <= 8
        # the gated RMS norm folded in: a CTA holds one value head's 128 outputs of its T rows
        self.norm = bool(norm)
        self.norm_eps = float(norm_eps)
        assert not self.norm or (tpc == 8 and T <= warps), "the folded norm: one value head a CTA"
        assert (h, hv) in GEOMETRIES, (h, hv)
        # the verify conv folded in: a CTA computes its head's q / k and its value heads' v rows
        # from the projection, the conv pool and the taps; a value head belongs to one CTA
        assert not fold or tpc in (8, 24), "the folded conv needs whole value heads a CTA"
        self.fold = bool(fold)
        if self.fold:
            qk_first = False
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
        # pf = 0: every state tile by bulk copy as soon as the slot is known, none in registers
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
        # the folded conv's q | k rows: T x 256 bf16
        self.o_cv = o
        o += T * 512 if self.fold else 0
        # the folded norm's rows: T x 128 bf16 (the outputs, rounded as they would be stored)
        self.o_on = o
        o += T * 256 if self.norm else 0
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
        cx_ptr: cutlass.Int64,
        cx_st: cutlass.Int64,
        cw_ptr: cutlass.Int64,
        cs_ptr: cutlass.Int64,
        cs_ss: cutlass.Int64,
        win_ptr: cutlass.Int64,
        win_sn: cutlass.Int64,
        win_st: cutlass.Int64,
        wrow_ptr: cutlass.Int64,
        zn_ptr: cutlass.Int64,
        zn_st: cutlass.Int64,
        gw_ptr: cutlass.Int64,
        on_ptr: cutlass.Int64,
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
            cx_ptr,
            cx_st,
            cw_ptr,
            cs_ptr,
            cs_ss,
            win_ptr,
            win_sn,
            win_st,
            wrow_ptr,
            zn_ptr,
            zn_st,
            gw_ptr,
            on_ptr,
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

    def _qk_raw_smem(self, nt, s_cv, g, c, scale):
        """`_qk_raw` over the folded conv's shared q | k rows (the kernel's fold prologue)."""
        T, TP = self.T, self.TP
        zi = cutlass.Int32(0)
        t = g % 4 if TP == 4 else g
        valid = t < T
        tc = cutlass.min(t, T - 1)
        if TP == 4:
            is_q = g >= 4
        elif nt == 1:
            is_q = g >= 0
        else:
            is_q = g < 0
        mult = cutlass.Float32(cutlass.select_(is_q, scale, cutlass.Float32(1.0)))
        src = (
            s_cv + tc * 512 + cutlass.Int32(cutlass.select_(is_q, zi, cutlass.Int32(256))) + c * 16
        )
        raw = []
        for jj in range(4):
            w4 = _lds4_i32(src + jj * 64)
            for e in range(4):
                raw.append(cutlass.Int32(cutlass.select_(valid, w4[e], zi)))
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
        """This lane's A fragment words of staged tile i: rows g and g + 8, chunk order by row
        parity so each quarter-warp's two rows are on disjoint banks."""
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
        """D = h0 [16 x 128] times the T keys / scaled queries, two MMA chains (even / odd k-steps)."""
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
        cx_ptr: cutlass.Int64,
        cx_st: cutlass.Int64,
        cw_ptr: cutlass.Int64,
        cs_ptr: cutlass.Int64,
        cs_ss: cutlass.Int64,
        win_ptr: cutlass.Int64,
        win_sn: cutlass.Int64,
        win_st: cutlass.Int64,
        wrow_ptr: cutlass.Int64,
        zn_ptr: cutlass.Int64,
        zn_st: cutlass.Int64,
        gw_ptr: cutlass.Int64,
        on_ptr: cutlass.Int64,
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
        s_cv = sbase + self.o_cv
        s_on = sbase + self.o_on

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
            # the step's own (L2-resident) inputs first, ahead of the state stream
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
                # both halves, every row, before p / half are known; rows j >= p masked at use
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
                if const_expr(self.fold):
                    # The verify conv of this CTA's channels for the T rows, a thread an 8-channel
                    # group. The owning CTA writes the per-row windows (q/k: split 0); the pool is
                    # not advanced (the engine's acceptance promote writes it).
                    ngr: cutlass.Constexpr = 32 + NHV * 16
                    wrow = cutlass.Int64(_ldg_i32(wrow_ptr + n64 * 4))
                    for itc in cutlass.range_constexpr(-(-ngr // NTHR)):
                        gc = tidx + itc * NTHR
                        if gc < ngr:
                            ec = gc * 8
                            is_v = ec >= 256
                            vv = ec - 256
                            hlc = vv // 128
                            vc = vv - hlc * 128
                            ch = cutlass.Int32(
                                cutlass.select_(
                                    is_v,
                                    2 * H * K + (hv0 + hlc) * V + vc,
                                    cutlass.select_(
                                        ec < 128, h * K + ec, H * K + h * K + (ec - 128)
                                    ),
                                )
                            )
                            dst = cutlass.Int32(
                                cutlass.select_(is_v, s_v + (hlc * T) * 256 + vc * 2, s_cv + ec * 2)
                            )
                            pitch = cutlass.Int32(
                                cutlass.select_(is_v, cutlass.Int32(256), cutlass.Int32(512))
                            )
                            owns = is_v | (sp == 0)
                            ch64 = cutlass.Int64(ch)
                            wa = cw_ptr + ch64 * 8
                            ww = []
                            for q4 in cutlass.range_constexpr(4):
                                ww.extend(_LDG4["plain"](wa + q4 * 16))
                            pa = cs_ptr + (slot64 * cs_ss + ch64 * 3) * 2
                            pw = []
                            for q4 in cutlass.range_constexpr(3):
                                pw.extend(_LDG4["plain"](pa + q4 * 16))
                            xw = []
                            for tt in cutlass.range_constexpr(T):
                                xw.append(
                                    _LDG4["plain"](cx_ptr + ((n64 * T + tt) * cx_st + ch64) * 2)
                                )
                            # per channel: taps = pool[0..2], x rows; weights w[0..3]
                            outs = [[] for _ in range(T)]
                            wins = [[] for _ in range(T)]
                            for cl in cutlass.range_constexpr(8):
                                wt = [
                                    _half16(ww[(cl * 4 + j) // 2], (cl * 4 + j) % 2 == 1)
                                    for j in range(4)
                                ]
                                taps = [
                                    _half16(pw[(cl * 3 + j) // 2], (cl * 3 + j) % 2 == 1)
                                    for j in range(3)
                                ]
                                for tt in cutlass.range_constexpr(T):
                                    taps.append(_half16(xw[tt][cl // 2], cl % 2 == 1))
                                for tt in cutlass.range_constexpr(T):
                                    outs[tt].append(
                                        _conv4_silu(
                                            _mul_bf16(taps[tt], wt[0]),
                                            _mul_bf16(taps[tt + 1], wt[1]),
                                            _mul_bf16(taps[tt + 2], wt[2]),
                                            _mul_bf16(taps[tt + 3], wt[3]),
                                        )
                                    )
                                    for j in cutlass.range_constexpr(3):
                                        wins[tt].append(taps[tt + 1 + j])
                            for tt in cutlass.range_constexpr(T):
                                o = outs[tt]
                                _sts4(
                                    dst + tt * pitch,
                                    _pack16(o[0], o[1]),
                                    _pack16(o[2], o[3]),
                                    _pack16(o[4], o[5]),
                                    _pack16(o[6], o[7]),
                                )
                                if owns:
                                    wv = wins[tt]
                                    wbase = (
                                        win_ptr
                                        + (wrow * win_sn + cutlass.Int64(tt) * win_st + ch64 * 3)
                                        * 2
                                    )
                                    for q4 in cutlass.range_constexpr(3):
                                        _stg4(
                                            wbase + q4 * 16,
                                            _pack16(wv[q4 * 8], wv[q4 * 8 + 1]),
                                            _pack16(wv[q4 * 8 + 2], wv[q4 * 8 + 3]),
                                            _pack16(wv[q4 * 8 + 4], wv[q4 * 8 + 5]),
                                            _pack16(wv[q4 * 8 + 6], wv[q4 * 8 + 7]),
                                        )
                    cute.arch.sync_threads()
                n_v: cutlass.Constexpr = 0 if self.fold else NHV * T * 16
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

            # key / query B fragments: column nt * 8 + g; lane quad c holds dims 32 jj + 8 c + 0..7
            bq = []
            for nt in cutlass.range_constexpr(NT):
                if const_expr(self.qk_first):
                    rq = pre[nt]
                elif const_expr(self.fold):
                    rq = self._qk_raw_smem(nt, s_cv, g, c, scale)
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

            # the state dots need no history: run them while it is in flight
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
                # history coefficients (k_t . k_j or q_t . k_j) exp(G_t - G_j), j < p, bf16 B fragments
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

            # the folded norm's gate and weight words (after any dependency wait), landing during the tiles
            if const_expr(self.norm):
                nz0 = zi
                nz1 = zi
                nw0 = zi
                nw1 = zi
                if warp < T:
                    hvp = h * GRP + sp
                    rowp = n64 * T + cutlass.Int64(warp)
                    colp = cutlass.Int64(4 * lane)
                    zpre = _ldg2_i32(zn_ptr + (rowp * zn_st + cutlass.Int64(hvp * V) + colp) * 2)
                    wpre = _ldg2_i32(gw_ptr + colp * 2)
                    nz0 = zpre[0]
                    nz1 = zpre[1]
                    nw0 = wpre[0]
                    nw1 = wpre[1]

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
                    # rows g and g + 8 of the staged tile, chunk order by row parity (bank-disjoint)
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
                        if const_expr(self.norm):
                            # the CTA's one value head: staged for the norm below
                            _sts_bf16(s_on + (t * V + v0 + r) * 2, o)
                        else:
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

            # ---- 6. the folded gated RMS norm: warp t takes row t of the CTA's value head ------
            if const_expr(self.norm):
                cute.arch.sync_threads()
                if warp < T:
                    hvn = h * GRP + sp  # tpc 8: the CTA's value head
                    row64 = n64 * T + cutlass.Int64(warp)
                    col = cutlass.Int64(4 * lane)
                    xw = _lds2_i32(s_on + (warp * V + 4 * lane) * 2)
                    zw = [nz0, nz1]
                    ww = [nw0, nw1]
                    xs4 = [_bf2_lo(xw[0]), _bf2_hi(xw[0]), _bf2_lo(xw[1]), _bf2_hi(xw[1])]
                    ssq = _gn_sumsq(xs4[0], xs4[1], xs4[2], xs4[3])
                    for m_ in cutlass.range_constexpr(5):
                        ssq = _add_rn(ssq, _shfl_xor(ssq, 16 >> m_))
                    rstd = _gn_rstd(ssq, cutlass.Float32(self.norm_eps))
                    ws4 = [_bf2_lo(ww[0]), _bf2_hi(ww[0]), _bf2_lo(ww[1]), _bf2_hi(ww[1])]
                    zs4 = [_bf2_lo(zw[0]), _bf2_hi(zw[0]), _bf2_lo(zw[1]), _bf2_hi(zw[1])]
                    ys4 = [
                        _mul_rn(_mul_rn(_mul_rn(rstd, xs4[e]), ws4[e]), _gn_sigmoid(zs4[e]))
                        for e in range(4)
                    ]
                    _stg2_i32(
                        on_ptr + ((row64 * HV + cutlass.Int64(hvn)) * V + col) * 2,
                        _bf2_pack(ys4[0], ys4[1]),
                        _bf2_pack(ys4[2], ys4[3]),
                    )

            if const_expr(self.pdl and not self.early_trigger):
                cute.arch.griddepcontrol_launch_dependents()


# -- host side ---------------------------------------------------------------------------------

_COMPILED: dict = {}


def default_config(n: int, t: int, commit: bool = False) -> dict:
    """Launch mapping by request count: smaller CTAs for small (latency-bound) batches; from 32
    requests (or 8 with the commit) CTAs of 8 m-tiles x 4 warps, at most 5 (4) an SM."""
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
            *([i64] * 43),
            cutlass.Int32(1),
            cutlass.Float32(1.0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--opt-level 2 --enable-tvm-ffi",
        )
        _COMPILED[key] = fn
    return fn


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
    fold: bool = False,
    norm_eps: Optional[float] = None,
    work_item_width: int = 4,
    separate_replay_slots: bool = False,
) -> dict:
    """The kernel's compile-time parameters for a call of `n` requests: its key in `_COMPILED`.
    `fold` is a `conv_fold` call, `norm_eps` a `gated_norm` call's epsilon, `work_item_width`
    the int32 fields per work item, and `separate_replay_slots` says the history is addressed by
    replay slots of its own (`replay_indices`)."""
    cfg = dict(default_config(n, t, commit))
    if config:
        cfg.update(config)
    if fold:
        if cfg.get("tpc") not in (8, 24):
            # small batches' mappings split a value head over CTAs: take the whole-head ones
            whole = (
                dict(tpc=8, warps=4, pf=1, minb=5)
                if n > 16
                else dict(tpc=24, warps=8, pf=1, minb=2)
            )
            cfg.update(whole)
            cfg.pop("early_acc", None)
        cfg["fold"] = True
    if norm_eps is not None:
        if cfg.get("tpc") != 8 or cfg.get("warps") != 4:
            cfg.update(tpc=8, warps=4)
            cfg["pf"] = min(int(cfg.get("pf", 1)), 2)
            cfg.setdefault("minb", 5)
        cfg["norm"] = True
        cfg["norm_eps"] = norm_eps
    # PDL: state / history before the wait; without it, the step's own inputs first
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


def precompile_keys(
    t: int,
    h: int,
    hv: int,
    max_requests: int,
    *,
    partition_from: int,
    launch: Callable[[int], tuple],
    conv_fold: Optional[Callable[[int], bool]] = None,
    fold_norm: Optional[Callable[[int], bool]] = None,
    norm_eps: Optional[float] = None,
    work_item_width: int = 4,
    separate_replay_slots: bool = False,
) -> set:
    """The `_COMPILED` key of every kernel a verify step of 1..`max_requests` requests launches,
    the way a target calls this entry: work items, and with them no in-kernel commit, from
    `partition_from` requests on; the q/k L2 norm and the gating in-kernel; bf16 `a` / `b`.
    `launch(n)` returns `(pdl, config)` for a batch of `n` requests (config None: the default
    mapping), `conv_fold(n)` whether it folds the conv, and `fold_norm(n)` whether a
    generation-only batch folds the gated norm (epsilon `norm_eps`); a mixed step's verify rows
    never fold it, so such a batch reaches both kernels. `work_item_width` is the int32 fields
    of the manager's work items and `separate_replay_slots` whether it keeps the history at
    replay slots of its own (every step then passes `replay_indices`)."""
    keys = set()
    for n in range(1, max_requests + 1):
        partitioned = n >= partition_from
        n_pdl, config = launch(n)
        norms = [None] + ([norm_eps] if fold_norm is not None and fold_norm(n) else [])
        for eps in norms:
            kw = _kernel_kw(
                n,
                t,
                h,
                hv,
                commit=not partitioned,
                pdl=bool(n_pdl),
                l2norm=True,
                gating=True,
                g_f32=False,
                b_f32=False,
                work_items=partitioned,
                config=config,
                fold=conv_fold is not None and conv_fold(n),
                norm_eps=eps,
                work_item_width=work_item_width,
                separate_replay_slots=separate_replay_slots,
            )
            keys.add(tuple(sorted(kw.items())))
    return keys


def precompile(t: int, h: int, hv: int, max_requests: int, **launches) -> int:
    """Compile every kernel of `precompile_keys(t, h, hv, max_requests, **launches)`. A compile
    costs about 0.5 s on GB300, so a target runs this at bind, before any step: the captured
    decode graphs do not reach every mapping (padded graphs skip 9 to 15 requests, which an
    eager mixed step carries). Returns the number of distinct kernels."""
    keys = precompile_keys(t, h, hv, max_requests, **launches)
    for key in keys:
        _compile(key, _Replay(**dict(key)))
    return len(keys)


def conv_fold_supported(x, weight, pool, window) -> bool:
    """Whether `cached_replay_update(conv_fold=...)` takes these operands: bf16, the layouts it
    reads, and 16-byte alignment of every 8-channel group (vector loads / stores)."""
    tensors = (x, weight, pool, window)
    if any(t.dtype != torch.bfloat16 for t in tensors):
        return False
    if x.stride(1) != 1 or not weight.is_contiguous() or weight.shape[1] != 4:
        return False
    if pool.stride()[1:] != (3, 1) or window.stride()[2:] != (3, 1):
        return False
    strides = (x.stride(0), pool.stride(0), window.stride(0), window.stride(1))
    return all(t.data_ptr() % 16 == 0 for t in tensors) and all(
        (2 * st) % 16 == 0 for st in strides
    )


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
    config: Optional[dict] = None,
    conv_fold: Optional[dict] = None,
    gated_norm: Optional[dict] = None,
    replay_indices: Optional[torch.Tensor] = None,
    **ignored,
) -> torch.Tensor:
    """Same contract as `fused_recurrent_gated_delta_rule_cached_replay_update` (bf16 state).

    A wrapping row commits its checkpoint in-kernel unless work items + `use_all_layer_commit`
    are given. Work items must list all N rows (`config={"use_wi": False}` reads
    `state_indices` instead); with separate replay slots their fifth field
    (`REPLAY_WORK_REPLAY_SLOT`) gives each row's replay slot, else `replay_indices` does. The
    Triton launch knobs are ignored.

    `replay_indices` (`state_indices` when None) address the history (`old_u` / `old_k` /
    `old_G`) and its bookkeeping (`prev_num_accepted_tokens` / `cache_buf_idx`); each replay
    slot's rows are packed behind a slot stride of 16-byte multiples, and the bookkeeping may
    have any element stride (a manager keeps both in one row per slot).

    `conv_fold` folds the verify conv in, bitwise the conv kernel: `x` (the q|k|v projection
    columns), `weight` `[C, 4]`, `pool` `[slots, C, 3]` (read, not advanced: the engine's
    acceptance promote rewrites it from `window`), `window` `[rows, T, C, 3]` and `window_rows`;
    the q/k/v arguments then give shapes only. Needs `tpc` 8 or 24.

    `gated_norm` folds `rms_norm_gated_token_major` (sigmoid gate) in, bitwise: `z`, `weight`,
    `eps`; the un-normed outputs are not stored. Needs `tpc` 8, 4 warps and T <= 4.
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
    if replay_work_items is not None and not use_all_layer_commit:
        raise ValueError("Partitioned replay requires the all-layer commit")
    if replay_work_items is None and use_all_layer_commit:
        raise ValueError("use_all_layer_commit requires replay work items")
    commit = replay_work_items is None
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
    fold_args = (0,) * 9
    if conv_fold is not None:
        cx, cw, cs = conv_fold["x"], conv_fold["weight"], conv_fold["pool"]
        win, wrows = conv_fold["window"], conv_fold["window_rows"]
        assert cx.shape == (N * T, qkv_width) and cx.stride(1) == 1 and cx.dtype == torch.bfloat16
        assert cw.shape == (qkv_width, 4) and cw.is_contiguous() and cw.dtype == torch.bfloat16
        assert cs.shape[1:] == (qkv_width, 3) and cs.stride()[1:] == (3, 1)
        assert cs.dtype == torch.bfloat16
        assert win.shape[1:] == (T, qkv_width, 3) and win.stride()[2:] == (3, 1)
        assert win.dtype == torch.bfloat16
        assert wrows.dtype == torch.int32 and wrows.shape[0] >= N and wrows.is_contiguous()
        assert conv_fold_supported(cx, cw, cs, win), (
            "conv_fold operands: layout / 16-byte alignment"
        )
        fold_args = (
            cx.data_ptr(), cx.stride(0), cw.data_ptr(), cs.data_ptr(), cs.stride(0),
            win.data_ptr(), win.stride(0), win.stride(1), wrows.data_ptr(),
        )  # fmt: skip
    norm_args = (0,) * 4
    norm_eps = None
    if gated_norm is not None:
        zn, gw = gated_norm["z"], gated_norm["weight"]
        assert T <= 4, "the folded norm takes a warp a row"
        assert (
            zn.shape == (N * T, HV, V) and zn.stride()[1:] == (V, 1) and zn.dtype == torch.bfloat16
        )
        assert gw.shape == (V,) and gw.is_contiguous() and gw.dtype == torch.bfloat16
        assert zn.data_ptr() % 8 == 0 and zn.stride(0) % 4 == 0 and gw.data_ptr() % 8 == 0
        norm_eps = float(gated_norm["eps"])
        norm_args = (zn.data_ptr(), zn.stride(0), gw.data_ptr(), output.data_ptr())
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
        fold=conv_fold is not None,
        norm_eps=norm_eps,
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
        *fold_args,
        *norm_args,
        N,
        float(scale),
    )
    return output


#: A tp4/tep4 rank's GDN heads.
RANK_K_HEADS, RANK_V_HEADS = 4, 12
#: A tep4 engine rank's state slots, and the GDN layers sharing one slot page (slot stride 36
#: layers' states).
TEP4_SLOTS, PAGE_LAYERS = 1874, 36
#: The same 26.5 GB page at a dep4 rank's (48-head) and a tp2/tep2 rank's (24-head) slot
#: stride: the top slot sits 1.3e10 elements past the base, as tep4's does.
DEP4_SLOTS = 468
TEP2_SLOTS = 937


def _tep2_spec(path, n, t, **spec) -> dict:
    """A tp2/tep2 rank's heads, 8 / 24, over a bf16 stock-manager page."""
    return dict(
        path=path,
        n=n,
        t=t,
        state_dtype=torch.bfloat16,
        H=8,
        HV=24,
        history_size=HIST,
        slots=TEP2_SLOTS,
        page_layers=PAGE_LAYERS,
        **spec,
    )


def _tep4_spec(path, n, t, **spec) -> dict:
    """A tp4/tep4 rank's heads over the stock tep4 engine's bf16 state page."""
    return dict(
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
    )


def _dep4_spec(path, n, t, **spec) -> dict:
    """A dep4 rank's heads, all 16 / 48 under attention DP, over a bf16 stock-manager page."""
    return dict(
        path=path,
        n=n,
        t=t,
        state_dtype=torch.bfloat16,
        H=16,
        HV=48,
        history_size=HIST,
        slots=DEP4_SLOTS,
        page_layers=PAGE_LAYERS,
        **spec,
    )


class GdnCachedReplayCute(OpWrapper):
    """`cached_replay_update`: one verify step of `N` sequences x `T` tokens through one GDN
    layer, the contract of `fused_recurrent_gated_delta_rule_cached_replay_update` (bf16 state).
    Head counts from the call's shapes (`GEOMETRIES`); `g` / `beta` may be strided column views
    of the input projection (unit head stride).
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: the test harness's `_Bench` spec (path, n x t, steps with random acceptance, start,
    # accept, dup, H / HV, slots, page_layers) plus `strided_gates` (g / beta as column views of
    # the projection). The small (in-kernel commit) and all_layer paths at each geometry.
    CELLS: tuple[Cell, ...] = (
        # tep4: 4 / 12 heads.
        Cell(
            why="tep4 small n=1 T=4: MTP3 verify at con1, in-kernel commit on the page's last slot",
            spec=_tep4_spec("small", 1, 4, steps=8, strided_gates=True),
        ),
        Cell(
            why="tep4 small n=1 T=8 full accept: MTP7 appends to the history's end, then wraps",
            spec=_tep4_spec(
                "small", 1, 8, steps=8, start="fresh", accept="full", strided_gates=True
            ),
        ),
        Cell(
            why="tep4 all_layer n=16 T=4: the smallest partitioned batch, all-layer commit",
            spec=_tep4_spec("all_layer", 16, 4, steps=6, strided_gates=True),
        ),
        Cell(
            why="tep4 all_layer n=32 T=8: the widest verify below N=128",
            spec=_tep4_spec("all_layer", 32, 8, steps=6, strided_gates=True),
        ),
        Cell(
            why="tep4 all_layer n=128 T=4: a large partitioned batch",
            spec=_tep4_spec("all_layer", 128, 4, steps=4),
        ),
        # dep4: 16 / 48 heads.
        Cell(
            why="dep4 small n=1 T=4: the in-kernel commit of the page's last slot (467)",
            spec=_dep4_spec("small", 1, 4, steps=8, strided_gates=True),
        ),
        Cell(
            why="dep4 small n=8 T=4 dup: rows 5..7 share one CUDA-graph padding slot",
            spec=_dep4_spec("small", 8, 4, steps=6, dup=(5, 6, 7), strided_gates=True),
        ),
        # tep2: 8 / 24 heads, in-kernel commit below 16 requests, all-layer from 16.
        Cell(
            why="tep2 small n=4 T=4: a 16-row verify, in-kernel commit",
            spec=_tep2_spec("small", 4, 4, steps=6, strided_gates=True),
        ),
        Cell(
            why="tep2 all_layer n=32 T=4: a 128-row verify, all-layer commit",
            spec=_tep2_spec("all_layer", 32, 4, steps=4, strided_gates=True),
        ),
        Cell(
            why="dep4 all_layer n=128 T=4: a rank's largest verify",
            spec=_dep4_spec("all_layer", 128, 4, steps=4, strided_gates=True),
        ),
    )

    note = """
    Source: this file's CuTe DSL kernel (`_Replay`), one compile per (geometry, T, launch
    mapping, commit, PDL, folds) on first use (about 0.5 s): the warm-up before CUDA-graph capture
    must cover it. Called by `_gdn_verify` over the bf16 recurrent pool (slot stride 36 layers'
    states) when the rank's GDN heads are a `GEOMETRIES` pair, with work items +
    `use_all_layer_commit` exactly when the Triton entry gets them (N >= 16). The history is
    addressed by the manager's replay slots (`replay_indices`) when it keeps the history in
    slots of its own; the work items then carry the replay slot as a fifth field. PDL only at up
    to 24 (request x value head) pairs: the conv before it triggers its dependents at its start.

    Numerics: the Triton kernel's rounding points (bf16 L2-normalized q / k and history
    coefficients, fp32 recurrence, fp32 commit rounded once to the bf16 pool, bf16 history u / k,
    fp32 G); the dot order differs, so outputs match it to a few bf16 ulps, not bitwise. Launch
    mappings are bitwise each other, and CUDA-graph replay is bitwise eager.

    Certified on GB300 at the cells (rows at both ends of the page, consecutive steps from warm
    histories with random acceptance) within the Triton entry's test bands. Not supported: fp32
    state, T > 8, other geometries, K or V other than 128, history other than 16, pre-gated
    g / beta (`A_log=None` compiles but is not certified).
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
        config: Optional[dict] = None,
        conv_fold: Optional[dict] = None,
        gated_norm: Optional[dict] = None,
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
            config=config,
            conv_fold=conv_fold,
            gated_norm=gated_norm,
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
        config: Optional[dict] = None,
        conv_fold: Optional[dict] = None,
        replay_indices: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(out, ssm_states, old_u, old_k, old_G)` as the call leaves them; mutates nothing.
        The state is rebuilt from the checkpoint and history in fp32 and the new tokens stepped
        one at a time in fp32 (no matmul, so no TF32), every store rounded once to its buffer's
        dtype. Rows with a negative slot keep `output`'s prior content (NaN without `output`).
        The history and its bookkeeping are read and written at `replay_indices` (the checkpoint
        slots when None). `replay_work_items` only decides whether a wrap commits here; the
        launch knobs, `config`, `old_beta` and `n_writes` are ignored."""
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
        commit = replay_work_items is None
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
        config: Optional[dict] = None,
        conv_fold: Optional[dict] = None,
        gated_norm: Optional[dict] = None,
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
