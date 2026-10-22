# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode-row W8A16 projection: `y[M, N] = x[M, K] @ dequant(W)^T` with FP8 e4m3 weights
(128 x 128 block scales, fp32) and bf16 activations that are never quantized.

CuTe DSL (inline PTX), one PDL launch, CUDA-graph capturable, M = 1 .. 32.

Math: swap-AB warp MMA (`mma.sync.m16n8k16`, bf16 in, fp32 accumulate). The A operand is
16 weight rows dequantized in registers with `cvt.rn.bf16x2.e4m3x2` (exact: every e4m3
value is a bf16); B is up to 8 activation rows per n8 tile, read as raw bf16. Each 128-wide
k-block is accumulated on its own (four independent MMA chains) and folded into the running
fp32 sum times its block scale, so the only roundings are fp32 sums and the final bf16 store.

Weight layout (`pack_weight`): 2 KB units of 16 rows x 128 k in exactly the order the MMA
fragments read them (one conflict-free `ld.shared.v4` per lane per two k16 steps), stored
k-block-major `[K/128][ceil(N/16)][2048]` or, for small N, tile-major
`[ceil(N/16)][K/128][2048]`. K is permuted identically for both operands inside every
128-block (lane quad t of k16 step j holds k = 32 (j // 2) + 8 t + 4 (j % 2) + 0..3), which
leaves the dot product unchanged and makes every activation read a 16-byte vector.

Work split (`Plan`): a CTA owns a contiguous range of 16-row tiles over its K range (all of K,
or 1/ks of it in a `ks`-CTA cluster reduced over DSMEM). A weight-producer warp streams the
CTA's units with `cp.async.bulk` into a ring of shared-memory slots (a slot = `kw` k-blocks of
every tile of the CTA) completed on mbarriers; the static weight needs no PDL wait, so the ring
fills while the kernel ahead finishes. Consumer warp (kw_i, tw_i) takes k-block kw_i of a slot
for tiles tw_i, tw_i + tw, .... Activations are read only after `griddepcontrol.wait`, either
straight into registers a few stages ahead (few stages per warp) or by an activation-producer
warp with 16-byte `cp.async` into the same ring slots (long K). The `kw` k-block partials are
summed in shared memory in a fixed order and stored as bf16: results are deterministic.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import torch
from cutlass import const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack, make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

TILE_N = 16
BLOCK_K = 128
UNIT = TILE_N * BLOCK_K  # bytes of one (16-row, 128-k) unit
FP8_MAX = 448.0
MAX_ROWS = 32


# ------------------------------------------------------------------------------------------
# inline PTX helpers
# ------------------------------------------------------------------------------------------


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
        ty(llvm.extractvalue(T.i32() if ty is cutlass.Int32 else T.f32(), res, [i], loc=loc, ip=ip))
        for i in range(n)
    ]


@dsl_user_op
def _lds4(saddr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_ir(saddr, cutlass.Int32, loc, ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc,
        ip,
        side=True,
    )
    return _unpack(res, 4, cutlass.Int32, loc, ip)


@dsl_user_op
def _lds_f32(saddr, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [_ir(saddr, cutlass.Int32, loc, ip)],
            "ld.shared.f32 $0, [$1];",
            "=f,r",
            loc,
            ip,
            side=True,
        )
    )


@dsl_user_op
def _sts_f32(saddr, v, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(saddr, cutlass.Int32, loc, ip), _ir(v, cutlass.Float32, loc, ip)],
        "{ st.shared.f32 [$1], $2; mov.u32 $0, 0; }",
        "=r,r,f",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _ldg4p(gaddr, pred, *, loc=None, ip=None):
    """16 bytes from global (L1-cached, read-only) if `pred` != 0, else zeros."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_ir(gaddr, cutlass.Int64, loc, ip), _ir(pred, cutlass.Int32, loc, ip)],
        "{ .reg .pred p; setp.ne.b32 p, $5, 0; mov.b32 $0, 0; mov.b32 $1, 0; mov.b32 $2, 0; "
        "mov.b32 $3, 0; @p ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4]; }",
        "=r,=r,=r,=r,l,r",
        loc,
        ip,
        side=True,
    )
    return _unpack(res, 4, cutlass.Int32, loc, ip)


@dsl_user_op
def _ldg_f32(gaddr, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [_ir(gaddr, cutlass.Int64, loc, ip)],
            "ld.global.nc.f32 $0, [$1];",
            "=f,l",
            loc,
            ip,
            side=True,
        )
    )


@dsl_user_op
def _stg_bf16(gaddr, v, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(gaddr, cutlass.Int64, loc, ip), _ir(v, cutlass.Float32, loc, ip)],
        "{ .reg .b16 h; cvt.rn.bf16.f32 h, $2; st.global.b16 [$1], h; mov.u32 $0, 0; }",
        "=r,l,f",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _ldg_bf16p(gaddr, pred, *, loc=None, ip=None):
    """One bf16 from global as fp32 if `pred` != 0, else 0."""
    return cutlass.Float32(
        _asm(
            T.f32(),
            [_ir(gaddr, cutlass.Int64, loc, ip), _ir(pred, cutlass.Int32, loc, ip)],
            "{ .reg .pred p; .reg .b16 h; setp.ne.b32 p, $2, 0; mov.b16 h, 0; @p ld.global.b16 h, [$1]; "
            "cvt.f32.bf16 $0, h; }",
            "=f,l,r",
            loc,
            ip,
            side=True,
        )
    )


@dsl_user_op
def _ldg_s32p(gaddr, pred, dflt, *, loc=None, ip=None):
    """One int32 from global if `pred` != 0, else `dflt`."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [
                _ir(gaddr, cutlass.Int64, loc, ip),
                _ir(pred, cutlass.Int32, loc, ip),
                _ir(dflt, cutlass.Int32, loc, ip),
            ],
            "{ .reg .pred p; setp.ne.b32 p, $2, 0; mov.b32 $0, $3; @p ld.global.b32 $0, [$1]; }",
            "=r,l,r,r",
            loc,
            ip,
            side=True,
        )
    )


@dsl_user_op
def _round_bf16(x, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [_ir(x, cutlass.Float32, loc, ip)],
            "{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; cvt.f32.bf16 $0, h; }",
            "=f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _dq_chunk(w0, w1, w2, w3, *, loc=None, ip=None):
    """Four words of e4m3 (rows g / g+8 of k16 step 2c, then of step 2c+1) -> the eight bf16x2
    A registers of the two steps: (R0, R1, R2, R3) = (g lo, g+8 lo, g hi, g+8 hi)."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 8),
        [_ir(v, cutlass.Int32, loc, ip) for v in (w0, w1, w2, w3)],
        "{ .reg .b16 a0, a1, b0, b1, c0, c1, d0, d1; "
        "mov.b32 {a0, a1}, $8; mov.b32 {b0, b1}, $9; mov.b32 {c0, c1}, $10; mov.b32 {d0, d1}, $11; "
        "cvt.rn.bf16x2.e4m3x2 $0, a0; cvt.rn.bf16x2.e4m3x2 $1, b0; "
        "cvt.rn.bf16x2.e4m3x2 $2, a1; cvt.rn.bf16x2.e4m3x2 $3, b1; "
        "cvt.rn.bf16x2.e4m3x2 $4, c0; cvt.rn.bf16x2.e4m3x2 $5, d0; "
        "cvt.rn.bf16x2.e4m3x2 $6, c1; cvt.rn.bf16x2.e4m3x2 $7, d1; }",
        "=r,=r,=r,=r,=r,=r,=r,=r,r,r,r,r",
        loc,
        ip,
    )
    return _unpack(res, 8, cutlass.Int32, loc, ip)


@dsl_user_op
def _mma(a, b, c, *, loc=None, ip=None):
    """D = A B + C on m16n8k16, bf16 A/B, fp32 C/D; a: 4 regs, b: 2 regs, c: 4 f32."""
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
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
def _mbar_init(addr, count, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(addr, cutlass.Int32, loc, ip), _ir(count, cutlass.Int32, loc, ip)],
        "{ mbarrier.init.shared::cta.b64 [$1], $2; mov.u32 $0, 0; }",
        "=r,r,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _mbar_init_fence(*, loc=None, ip=None):
    _asm(
        T.i32(),
        [],
        "{ fence.mbarrier_init.release.cluster; mov.u32 $0, 0; }",
        "=r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _mbar_expect_tx(addr, nbytes, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(addr, cutlass.Int32, loc, ip), _ir(nbytes, cutlass.Int32, loc, ip)],
        "{ mbarrier.arrive.expect_tx.shared::cta.b64 _, [$1], $2; mov.u32 $0, 0; }",
        "=r,r,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _mbar_arrive(addr, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(addr, cutlass.Int32, loc, ip)],
        "{ mbarrier.arrive.shared::cta.b64 _, [$1]; mov.u32 $0, 0; }",
        "=r,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _mbar_wait(addr, parity, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(addr, cutlass.Int32, loc, ip), _ir(parity, cutlass.Int32, loc, ip)],
        "{ .reg .pred p; WAIT: mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; @!p bra WAIT; "
        "mov.u32 $0, 0; }",
        "=r,r,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _bulk_g2s(saddr, gaddr, nbytes, mbar, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [
            _ir(saddr, cutlass.Int32, loc, ip),
            _ir(gaddr, cutlass.Int64, loc, ip),
            _ir(nbytes, cutlass.Int32, loc, ip),
            _ir(mbar, cutlass.Int32, loc, ip),
        ],
        "{ cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2], $3, [$4]; "
        "mov.u32 $0, 0; }",
        "=r,r,l,r,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _cp16(saddr, gaddr, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(saddr, cutlass.Int32, loc, ip), _ir(gaddr, cutlass.Int64, loc, ip)],
        "{ cp.async.cg.shared.global [$1], [$2], 16; mov.u32 $0, 0; }",
        "=r,r,l",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _cp_async_mbar_arrive(mbar, *, loc=None, ip=None):
    """Arrive on `mbar` once this thread's earlier cp.async copies have landed."""
    _asm(
        T.i32(),
        [_ir(mbar, cutlass.Int32, loc, ip)],
        "{ cp.async.mbarrier.arrive.noinc.shared::cta.b64 [$1]; mov.u32 $0, 0; }",
        "=r,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _mapa(addr, rank, *, loc=None, ip=None):
    """This CTA's shared-memory address `addr` in cluster CTA `rank`'s window."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [_ir(addr, cutlass.Int32, loc, ip), _ir(rank, cutlass.Int32, loc, ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _st_dsmem_f32(addr, v, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(addr, cutlass.Int32, loc, ip), _ir(v, cutlass.Float32, loc, ip)],
        "{ st.shared::cluster.f32 [$1], $2; mov.u32 $0, 0; }",
        "=r,r,f",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _cluster_arrive_release(*, loc=None, ip=None):
    _asm(
        T.i32(),
        [],
        "{ barrier.cluster.arrive.release.aligned; mov.u32 $0, 0; }",
        "=r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _cluster_arrive_relaxed(*, loc=None, ip=None):
    _asm(
        T.i32(),
        [],
        "{ barrier.cluster.arrive.relaxed.aligned; mov.u32 $0, 0; }",
        "=r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _cluster_wait(*, loc=None, ip=None):
    _asm(T.i32(), [], "{ barrier.cluster.wait.aligned; mov.u32 $0, 0; }", "=r", loc, ip, side=True)


@dsl_user_op
def _cluster_wait_acquire(*, loc=None, ip=None):
    _asm(
        T.i32(),
        [],
        "{ barrier.cluster.wait.acquire.aligned; mov.u32 $0, 0; }",
        "=r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _bulk_prefetch_l2(gaddr, nbytes, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(gaddr, cutlass.Int64, loc, ip), _ir(nbytes, cutlass.Int32, loc, ip)],
        "{ cp.async.bulk.prefetch.L2.global [$1], $2; mov.u32 $0, 0; }",
        "=r,l,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _bar_sync(bar_id, nthreads, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(bar_id, cutlass.Int32, loc, ip), _ir(nthreads, cutlass.Int32, loc, ip)],
        "{ bar.sync $1, $2; mov.u32 $0, 0; }",
        "=r,r,r",
        loc,
        ip,
        side=True,
    )


# ------------------------------------------------------------------------------------------
# kernel
# ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Plan:
    """Launch shape. `grid` CTAs, each a contiguous range of <= ceil(tiles / grid) 16-row
    tiles; `kw` x `tw` consumer warps (k-blocks per ring slot x tile groups); `slots` ring
    slots; `l2_ahead` slots past the ring prefetched into L2 at the start."""

    grid: int
    kw: int
    tw: int
    slots: int
    l2_ahead: int = 0  # rolling window of slots past the ring prefetched into L2
    xreg: int = 0  # activations: 0 auto, 1 registers, 2 through the ring
    xdepth: int = 0  # register mode: stages of activations loaded ahead (0: auto)
    ks: int = 1  # cluster K split: `ks` CTAs share a tile range, each 1/ks of K, reduced over DSMEM


#: Most activation-fragment registers a consumer may hold to take the register path.
XREG_MAX = 64
XRS = 320  # bytes of one activation row slice (128 bf16 + 64 B pad: conflict-free fragment reads)


class _Fp8Gemv:
    def __init__(self, *, n, k, m, plan: Plan, pdl, tile_major, conv_dim=0, pad_slot=-1):
        assert k % BLOCK_K == 0 and 1 <= m <= MAX_ROWS
        self.n, self.k, self.m, self.pdl = n, k, m, pdl
        self.tile_major = tile_major
        # conv1d epilogue (`conv_dim` > 0): columns below conv_dim of a non-pad row get the
        # causal conv step (window of 3, 4 taps) + SiLU; see `fp8_gemv_conv1d`
        self.conv_dim, self.pad_slot = conv_dim, pad_slot
        assert conv_dim == 0 or plan.ks == 1
        self.nt8 = -(-m // 8)
        self.n_tiles = -(-n // TILE_N)
        self.nkb = k // BLOCK_K
        self.grid = plan.grid
        self.ks = plan.ks
        assert plan.grid % self.ks == 0 and self.nkb % self.ks == 0, (plan.grid, self.ks, self.nkb)
        self.groups = plan.grid // self.ks
        self.nkb_c = self.nkb // self.ks  # k-blocks per CTA
        self.rt = -(-self.n_tiles // self.groups)
        self.kw, self.tw, self.slots, self.l2_ahead = plan.kw, plan.tw, plan.slots, plan.l2_ahead
        assert self.nkb_c % self.kw == 0, (self.nkb_c, self.kw)
        self.n_stages = self.nkb_c // self.kw
        self.slots = min(self.slots, self.n_stages)
        self.l2_ahead = max(0, min(self.l2_ahead, self.n_stages - self.slots))
        self.rtw = -(-self.rt // self.tw)
        self.cw = self.kw * self.tw
        # Activations: with few stages a warp loads all of its fragments into registers right
        # after the PDL wait (lowest latency); otherwise they stream through the ring slots.
        self.xreg = self.n_stages * self.nt8 * 16 <= XREG_MAX if plan.xreg == 0 else plan.xreg == 1
        # register mode: stages of activation fragments in flight ahead of the one in use
        self.xdepth = plan.xdepth or max(1, XREG_MAX // (self.nt8 * 16) - 1)
        self.xdepth = min(self.xdepth, self.n_stages)
        # + weight producer warp (+ activation producer warp)
        self.threads = (self.cw + (1 if self.xreg else 2)) * 32
        self.w_bytes = self.kw * self.rt * UNIT
        self.x_bytes = 0 if self.xreg else self.kw * m * XRS
        self.slot_bytes = -(-(self.w_bytes + self.x_bytes) // 128) * 128
        self.np = self.nt8 * 8
        self.o_red = self.slots * self.slot_bytes
        red_bytes = self.kw * self.rt * self.np * TILE_N * 4
        # block scales of the (at most nsb) 128-row blocks the CTA's tiles touch
        self.nsb = self.rt // 8 + 2
        self.o_sc = self.o_red + -(-red_bytes // 128) * 128
        self.o_bar = self.o_sc + -(-(self.nsb * self.nkb * 4) // 128) * 128
        self.o_cred = self.o_bar + 3 * self.slots * 8
        cred_bytes = self.ks * self.rt * m * TILE_N * 4 if self.ks > 1 else 0
        self.smem = self.o_cred + cred_bytes + 128

    @cute.jit
    def __call__(
        self,
        out_t: cute.Tensor,
        x_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        s_ptr: cute.Pointer,
        x_stride: cutlass.Int32,
        cs_ptr: cute.Pointer,
        cw_ptr: cute.Pointer,
        idx_ptr: cute.Pointer,
        cs_s0: cutlass.Int32,
        cs_s1: cutlass.Int32,
        cs_s2: cutlass.Int32,
        stream,
    ):
        args = (
            out_t.iterator.toint(),
            x_ptr.toint(),
            w_ptr.toint(),
            s_ptr.toint(),
            x_stride,
            cs_ptr.toint(),
            cw_ptr.toint(),
            idx_ptr.toint(),
            cs_s0,
            cs_s1,
            cs_s2,
        )
        if const_expr(self.ks > 1):
            self.kernel(*args).launch(
                grid=[self.grid, 1, 1],
                block=[self.threads, 1, 1],
                smem=self.smem,
                stream=stream,
                cluster=[self.ks, 1, 1],
                use_pdl=self.pdl,
            )
        else:
            self.kernel(*args).launch(
                grid=[self.grid, 1, 1],
                block=[self.threads, 1, 1],
                smem=self.smem,
                stream=stream,
                use_pdl=self.pdl,
            )

    @cute.jit
    def _issue_w(self, dst, w_base, kbs, t0, nt, mbar):
        """Bulk copies of a stage's units (k-blocks kbs .. kbs + kw, tiles t0 .. t0 + nt)."""
        kw: cutlass.Constexpr = self.kw
        rt: cutlass.Constexpr = self.rt
        if const_expr(self.tile_major):
            for tl in cutlass.range_constexpr(rt):
                if tl < nt:
                    _bulk_g2s(
                        dst + tl * (kw * UNIT),
                        w_base + (cutlass.Int64(t0 + tl) * self.nkb + kbs) * UNIT,
                        kw * UNIT,
                        mbar,
                    )
        else:
            for j in cutlass.range_constexpr(kw):
                _bulk_g2s(
                    dst + j * (rt * UNIT),
                    w_base + (cutlass.Int64(kbs + j) * self.n_tiles + t0) * UNIT,
                    nt * UNIT,
                    mbar,
                )

    @cute.jit
    def _prefetch_w(self, w_base, kbs, t0, nt):
        kw: cutlass.Constexpr = self.kw
        rt: cutlass.Constexpr = self.rt
        if const_expr(self.tile_major):
            for tl in cutlass.range_constexpr(rt):
                if tl < nt:
                    _bulk_prefetch_l2(
                        w_base + (cutlass.Int64(t0 + tl) * self.nkb + kbs) * UNIT, kw * UNIT
                    )
        else:
            for j in cutlass.range_constexpr(kw):
                _bulk_prefetch_l2(
                    w_base + (cutlass.Int64(kbs + j) * self.n_tiles + t0) * UNIT, nt * UNIT
                )

    @cute.jit
    def _compute(self, acc, cur, slot, kb, kwi, twi, t0, nt, b0, scs, lane):
        """This warp's tiles of k-block `kb` (in ring slot `slot`) times the activation fragments
        `cur`, folded into `acc` with the block scale."""
        kw: cutlass.Constexpr = self.kw
        tw: cutlass.Constexpr = self.tw
        rt: cutlass.Constexpr = self.rt
        nt8: cutlass.Constexpr = self.nt8
        nkb: cutlass.Constexpr = self.nkb
        for i in cutlass.range_constexpr(self.rtw):
            tl = twi + i * tw
            if tl < nt:
                sc = _lds_f32(scs + (((t0 + tl) // 8 - b0) * nkb + kb) * 4)
                if const_expr(self.tile_major):
                    ua = slot + (tl * kw + kwi) * UNIT + lane * 16
                else:
                    ua = slot + (kwi * rt + tl) * UNIT + lane * 16
                wv = [_lds4(ua + c * 512) for c in range(4)]
                # one accumulator chain per 16-byte chunk (two k16 steps): 4-way ILP
                zero = cutlass.Float32(0.0)
                ch = [[[zero] * 4 for _ in range(nt8)] for _ in range(4)]
                for c in cutlass.range_constexpr(4):
                    a = _dq_chunk(wv[c][0], wv[c][1], wv[c][2], wv[c][3])
                    for st in cutlass.range_constexpr(2):
                        for j in cutlass.range_constexpr(nt8):
                            ch[c][j] = _mma(
                                a[4 * st : 4 * st + 4],
                                [cur[j][4 * c + 2 * st], cur[j][4 * c + 2 * st + 1]],
                                ch[c][j],
                            )
                for j in cutlass.range_constexpr(nt8):
                    for e in cutlass.range_constexpr(4):
                        part = (ch[0][j][e] + ch[1][j][e]) + (ch[2][j][e] + ch[3][j][e])
                        acc[i, j, e] = acc[i, j, e] + part * sc

    @cute.jit
    def _issue_x(self, xs, src, xh, xc, x_stride):
        """cp.async of one stage's activations into the slot's activation area `xs`."""
        m: cutlass.Constexpr = self.m
        n_rc: cutlass.Constexpr = self.kw * self.m
        for i in cutlass.range_constexpr(-(-n_rc // 2)):
            rc = xh + 2 * i
            xj = rc // m
            xr = rc - xj * m
            ok = rc < n_rc
            if const_expr(n_rc % 2 == 0):
                ok = rc >= 0
            if ok:
                _cp16(
                    xs + rc * XRS + xc * 16,
                    src + cutlass.Int64(xr) * cutlass.Int64(x_stride) * 2 + cutlass.Int64(xj * 256),
                )

    @cute.kernel
    def kernel(
        self,
        out_base: cutlass.Int64,
        x_base: cutlass.Int64,
        w_base: cutlass.Int64,
        s_base: cutlass.Int64,
        x_stride: cutlass.Int32,
        cs_base: cutlass.Int64,
        cw_base: cutlass.Int64,
        idx_base: cutlass.Int64,
        cs_s0: cutlass.Int32,
        cs_s1: cutlass.Int32,
        cs_s2: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        g = lane // 4
        t = lane % 4
        m: cutlass.Constexpr = self.m
        n_tiles: cutlass.Constexpr = self.n_tiles
        nkb: cutlass.Constexpr = self.nkb
        rt: cutlass.Constexpr = self.rt
        kw: cutlass.Constexpr = self.kw
        tw: cutlass.Constexpr = self.tw
        cw: cutlass.Constexpr = self.cw
        slots: cutlass.Constexpr = self.slots
        n_stages: cutlass.Constexpr = self.n_stages
        slot_bytes: cutlass.Constexpr = self.slot_bytes
        w_bytes: cutlass.Constexpr = self.w_bytes
        nt8: cutlass.Constexpr = self.nt8
        np_: cutlass.Constexpr = self.np
        rtw: cutlass.Constexpr = self.rtw
        n_out: cutlass.Constexpr = self.n
        l2a: cutlass.Constexpr = self.l2_ahead

        smem = cutlass.utils.SmemAllocator()
        base_t = smem.allocate_tensor(
            cutlass.Int8, cute.make_layout((self.smem - 128,), stride=(1,)), byte_alignment=128
        )
        sbase = cutlass.Int32(base_t.iterator.toint())
        ring = sbase
        red = sbase + self.o_red
        scs = sbase + self.o_sc
        bars = sbase + self.o_bar  # wfull[slots], xfull[slots], empty[slots]

        ks: cutlass.Constexpr = self.ks
        groups: cutlass.Constexpr = self.groups
        grp = bidx // ks
        rank = bidx % ks
        kb0 = rank * self.nkb_c
        t0 = (grp * n_tiles) // groups
        nt = ((grp + 1) * n_tiles) // groups - t0
        b0 = t0 // 8
        n_blocks: cutlass.Constexpr = -(-self.n // 128)
        n_sc: cutlass.Constexpr = self.nsb * nkb
        for q in cutlass.range_constexpr(-(-n_sc // self.threads)):
            qi = tidx + q * self.threads
            if qi < n_sc:
                blk = b0 + qi // nkb
                if blk < n_blocks:
                    _sts_f32(
                        scs + qi * 4, _ldg_f32(s_base + (cutlass.Int64(blk) * nkb + qi % nkb) * 4)
                    )

        if warp == cw:
            if lane == 0:
                for sl in cutlass.range_constexpr(slots):
                    _mbar_init(bars + 8 * sl, 1)
                    _mbar_init(bars + 8 * (slots + sl), 32)
                    _mbar_init(bars + 8 * (2 * slots + sl), cw)
                _mbar_init_fence()
        cute.arch.sync_threads()
        if const_expr(self.ks > 1):
            _cluster_arrive_relaxed()  # matched by the wait before the first DSMEM store
        if const_expr(self.pdl):
            cute.arch.griddepcontrol_launch_dependents()

        if warp == cw:
            # ---- weight producer (static data: no PDL wait). Slot layout: unit (j, tile) at
            # (j * rt + tile) * UNIT for a k-block-major weight, (tile * kw + j) * UNIT for a
            # tile-major one; one bulk copy per k-block row resp. per tile. ----
            if lane == 0:
                for ps in cutlass.range(n_stages):
                    psl = ps % slots
                    if ps >= slots:
                        _mbar_wait(bars + 8 * (2 * slots + psl), ((ps // slots) - 1) & 1)
                    if const_expr(l2a > 0):
                        # rolling window: stages slots .. slots + l2a - 1 at the start, then one
                        # stage l2a past each refill
                        if ps == 0:
                            for la in cutlass.range_constexpr(slots, slots + l2a):
                                self._prefetch_w(w_base, kb0 + la * kw, t0, nt)
                        if ps >= slots:
                            if ps + l2a < n_stages:
                                self._prefetch_w(w_base, kb0 + (ps + l2a) * kw, t0, nt)
                    _mbar_expect_tx(bars + 8 * psl, kw * nt * UNIT)
                    self._issue_w(
                        ring + psl * slot_bytes, w_base, kb0 + ps * kw, t0, nt, bars + 8 * psl
                    )
        elif warp == cw + 1:
            # ---- activation producer (ring mode): after the PDL wait, a stage's rows into its slot ----
            if const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
            xh = lane // 16
            xc = lane % 16
            x_lane = x_base + cutlass.Int64(xc * 16)
            for xs_ in cutlass.range(n_stages):
                xsl = xs_ % slots
                if xs_ >= slots:
                    _mbar_wait(bars + 8 * (2 * slots + xsl), ((xs_ // slots) - 1) & 1)
                self._issue_x(
                    ring + xsl * slot_bytes + w_bytes,
                    x_lane + cutlass.Int64((kb0 + xs_ * kw) * 256),
                    xh,
                    xc,
                    x_stride,
                )
                _cp_async_mbar_arrive(bars + 8 * (slots + xsl))
        else:
            # ---- consumers ----
            kwi = warp % kw
            twi = warp // kw
            n_ep: cutlass.Constexpr = -(-(rt * m * TILE_N) // (cw * 32))
            if const_expr(self.conv_dim > 0):
                # The conv operands of this thread's epilogue outputs -- slot, 4 taps and the 3-value
                # window -- before the PDL wait: the slots are host-filled, and the window is this
                # layer's, last written by this kernel a step ago.
                cvs = cute.make_rmem_tensor(
                    cute.make_layout((n_ep, 8), stride=(8, 1)), cutlass.Float32
                )
                cslots = cute.make_rmem_tensor(
                    cute.make_layout((n_ep,), stride=(1,)), cutlass.Int32
                )
                for ic in cutlass.range_constexpr(n_ep):
                    ec = tidx + ic * (cw * 32)
                    rc_ = ec % TILE_N
                    nc_ = (ec // TILE_N) % m
                    tc_ = ec // (TILE_N * m)
                    oc_ = (t0 + tc_) * TILE_N + rc_
                    okc = (tc_ < nt) & (oc_ < self.conv_dim)
                    sl_c = _ldg_s32p(
                        idx_base + cutlass.Int64(nc_) * 4, cutlass.Int32(okc), self.pad_slot
                    )
                    cslots[ic] = sl_c
                    okw = cutlass.Int32(okc & (sl_c != self.pad_slot))
                    for q in cutlass.range_constexpr(4):
                        cvs[ic, q] = _ldg_bf16p(cw_base + (cutlass.Int64(oc_) * 4 + q) * 2, okw)
                    wbase = cs_base + (cutlass.Int64(sl_c) * cs_s0 + cutlass.Int64(oc_) * cs_s1) * 2
                    for q in cutlass.range_constexpr(3):
                        cvs[ic, 4 + q] = _ldg_bf16p(wbase + cutlass.Int64(q) * cs_s2 * 2, okw)
            if const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
            acc = cute.make_rmem_tensor(
                cute.make_layout((rtw, nt8, 4), stride=(nt8 * 4, 4, 1)), cutlass.Float32
            )
            acc.fill(0.0)
            if const_expr(self.xreg):
                # fragments of this warp's k-block of a stage: row 8j + g, lane quad t reads
                # elements 32c + 8t .. + 7; loaded `xdepth` stages ahead (the stage loop is
                # unrolled, so every buffer is its own registers)
                xd: cutlass.Constexpr = self.xdepth
                xsrc = [
                    x_base
                    + cutlass.Int64(8 * j + g) * cutlass.Int64(x_stride) * 2
                    + cutlass.Int64((kb0 + kwi) * (BLOCK_K * 2) + t * 16)
                    for j in range(nt8)
                ]
                xok = [cutlass.Int32(8 * j + g < m) for j in range(nt8)]
                xbuf = {}
                for s0 in cutlass.range_constexpr(xd):
                    xbuf[s0] = [
                        [
                            v
                            for c in range(4)
                            for v in _ldg4p(xsrc[j] + (s0 * kw * (BLOCK_K * 2) + 64 * c), xok[j])
                        ]
                        for j in range(nt8)
                    ]
                for s in cutlass.range_constexpr(n_stages):
                    if const_expr(s + xd < n_stages):
                        xbuf[s + xd] = [
                            [
                                v
                                for c in range(4)
                                for v in _ldg4p(
                                    xsrc[j] + ((s + xd) * kw * (BLOCK_K * 2) + 64 * c), xok[j]
                                )
                            ]
                            for j in range(nt8)
                        ]
                    sl = s % slots
                    slot = ring + sl * slot_bytes
                    _mbar_wait(bars + 8 * sl, (s // slots) & 1)
                    self._compute(
                        acc, xbuf.pop(s), slot, kb0 + s * kw + kwi, kwi, twi, t0, nt, b0, scs, lane
                    )
                    cute.arch.sync_warp()
                    if lane == 0:
                        _mbar_arrive(bars + 8 * (2 * slots + sl))
            else:
                # activation row 8j + g (clamped to m - 1: columns past m are discarded), k-block of
                # this warp: lane quad t reads elements 32c + 8t .. + 7
                x_rows = [cutlass.min(8 * j + g, m - 1) * XRS + t * 16 for j in range(nt8)]
                for s in cutlass.range(n_stages):
                    sl = s % slots
                    par = (s // slots) & 1
                    slot = ring + sl * slot_bytes
                    _mbar_wait(bars + 8 * (slots + sl), par)
                    xa = slot + w_bytes + kwi * (m * XRS)
                    cur = [
                        [v for c in range(4) for v in _lds4(xa + x_rows[j] + 64 * c)]
                        for j in range(nt8)
                    ]
                    _mbar_wait(bars + 8 * sl, par)
                    self._compute(
                        acc, cur, slot, kb0 + s * kw + kwi, kwi, twi, t0, nt, b0, scs, lane
                    )
                    cute.arch.sync_warp()
                    if lane == 0:
                        _mbar_arrive(bars + 8 * (2 * slots + sl))

            # ---- k-block partials -> shared memory [kw][rt][np][16] fp32 ----
            for i in cutlass.range_constexpr(rtw):
                wtl = twi + i * tw
                if wtl < nt:
                    for j in cutlass.range_constexpr(nt8):
                        n0 = 8 * j + 2 * t
                        p = red + (((kwi * rt + wtl) * np_ + n0) * TILE_N + g) * 4
                        _sts_f32(p, acc[i, j, 0])
                        _sts_f32(p + TILE_N * 4, acc[i, j, 1])
                        _sts_f32(p + 8 * 4, acc[i, j, 2])
                        _sts_f32(p + TILE_N * 4 + 8 * 4, acc[i, j, 3])
            _bar_sync(1, cw * 32)
            if const_expr(ks == 1):
                for it in cutlass.range_constexpr(n_ep):
                    e_ = tidx + it * (cw * 32)
                    er = e_ % TILE_N
                    en = (e_ // TILE_N) % m
                    etl = e_ // (TILE_N * m)
                    orow = (t0 + etl) * TILE_N + er
                    if (etl < nt) & (orow < n_out):
                        tot = cutlass.Float32(0.0)
                        for q in cutlass.range_constexpr(kw):
                            tot = tot + _lds_f32(
                                red + (((q * rt + etl) * np_ + en) * TILE_N + er) * 4
                            )
                        if const_expr(self.conv_dim > 0):
                            p_ = _round_bf16(tot)
                            res = p_
                            if (orow < self.conv_dim) & (cslots[it] != self.pad_slot):
                                cv = (
                                    cvs[it, 0] * cvs[it, 4]
                                    + cvs[it, 1] * cvs[it, 5]
                                    + cvs[it, 2] * cvs[it, 6]
                                    + cvs[it, 3] * p_
                                )
                                res = cv / (1.0 + cute.math.exp(-cv))
                                wb = (
                                    cs_base
                                    + (
                                        cutlass.Int64(cslots[it]) * cs_s0
                                        + cutlass.Int64(orow) * cs_s1
                                    )
                                    * 2
                                )
                                _stg_bf16(wb, cvs[it, 5])
                                _stg_bf16(wb + cutlass.Int64(cs_s2) * 2, cvs[it, 6])
                                _stg_bf16(wb + cutlass.Int64(cs_s2) * 4, p_)
                            _stg_bf16(out_base + (cutlass.Int64(en) * n_out + orow) * 2, res)
                        else:
                            _stg_bf16(out_base + (cutlass.Int64(en) * n_out + orow) * 2, tot)
        if const_expr(ks > 1):
            # The CTA's K-share of each output goes to cluster rank (tile % ks), slot
            # [sender][tile][row of M][16]; after the cluster barrier the owner sums the shares in
            # rank order and stores. Every thread of the cluster takes part in the barriers.
            _cluster_wait()
            if warp < cw:
                items1: cutlass.Constexpr = rt * m * TILE_N
                for it1 in cutlass.range_constexpr(-(-items1 // (cw * 32))):
                    e1 = tidx + it1 * (cw * 32)
                    er1 = e1 % TILE_N
                    en1 = (e1 // TILE_N) % m
                    etl1 = e1 // (TILE_N * m)
                    if etl1 < nt:
                        tot1 = cutlass.Float32(0.0)
                        for q1 in cutlass.range_constexpr(kw):
                            tot1 = tot1 + _lds_f32(
                                red + (((q1 * rt + etl1) * np_ + en1) * TILE_N + er1) * 4
                            )
                        cslot = (
                            sbase
                            + self.o_cred
                            + (((rank * rt + etl1) * m + en1) * TILE_N + er1) * 4
                        )
                        _st_dsmem_f32(_mapa(cslot, etl1 % ks), tot1)
            _cluster_arrive_release()
            _cluster_wait_acquire()
            if warp < cw:
                items2: cutlass.Constexpr = rt * m * TILE_N
                for it2 in cutlass.range_constexpr(-(-items2 // (cw * 32))):
                    f_ = tidx + it2 * (cw * 32)
                    fr = f_ % TILE_N
                    fn_ = (f_ // TILE_N) % m
                    ftl = f_ // (TILE_N * m)
                    frow = (t0 + ftl) * TILE_N + fr
                    if (ftl < nt) & (frow < n_out) & (ftl % ks == rank):
                        tot2 = cutlass.Float32(0.0)
                        for q2 in cutlass.range_constexpr(ks):
                            tot2 = tot2 + _lds_f32(
                                sbase
                                + self.o_cred
                                + (((q2 * rt + ftl) * m + fn_) * TILE_N + fr) * 4
                            )
                        _stg_bf16(out_base + (cutlass.Int64(fn_) * n_out + frow) * 2, tot2)


# ------------------------------------------------------------------------------------------
# host side
# ------------------------------------------------------------------------------------------


@dataclass
class PackedFp8Weight:
    """`data`: uint8 `[K/128, ceil(N/16), 2048]` (see `pack_weight`); `scale`: fp32
    `[ceil(N/128), K/128]` contiguous; logical shape `[n, k]`."""

    data: torch.Tensor
    scale: torch.Tensor
    n: int
    k: int
    tile_major: bool = False


def quantize_fp8_block(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16/fp32 `[N, K]` -> (e4m3 codes `[N, K]`, fp32 scales `[ceil(N/128), K/128]`),
    scale = amax(128x128 block) / 448, code = round_to_e4m3(w / scale)."""
    n, k = w.shape
    assert k % BLOCK_K == 0
    nb = -(-n // 128)
    wp = torch.zeros(nb * 128, k, dtype=torch.float32, device=w.device)
    wp[:n] = w.float()
    blocks = wp.reshape(nb, 128, k // BLOCK_K, BLOCK_K)
    amax = blocks.abs().amax((1, 3))
    scale = (amax / FP8_MAX).clamp_min(1e-30)
    codes = (blocks / scale[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).reshape(nb * 128, k)[:n]
    return codes.to(torch.float8_e4m3fn), scale.contiguous()


def dequantize_fp8_block(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """fp32 `[N, K]` = code * scale of its 128x128 block."""
    n, k = codes.shape
    s = scale.repeat_interleave(128, 0)[:n].repeat_interleave(BLOCK_K, 1)
    return codes.float() * s


#: Weights of at most this many 16-row tiles are stored tile-major (a CTA then holds one or two
#: tiles, and one bulk copy brings a tile's k-blocks of a stage); larger ones k-block-major.
TILE_MAJOR_MAX_TILES = 320


def pack_weight(
    codes: torch.Tensor, scale: torch.Tensor, tile_major: bool | None = None
) -> PackedFp8Weight:
    """e4m3 `[N, K]` (+ its block scales `[ceil(N/128), K/128]` fp32) -> the kernel's layout.

    A unit (k-block kb, 16-row tile T) is 2048 bytes: rows 16T .. 16T + 15, k = 128 kb .. + 127, as
    `[c (4)][lane (32)][16 B]`, lane = 4 g + t, the 16 bytes being
    `W[16T + g][32c + 8t + 0..3], W[16T + g + 8][same], W[16T + g][32c + 8t + 4..7],
    W[16T + g + 8][same]` (rows past N are zero). Units are stored `[K/128][ceil(N/16)]`
    (k-block-major) or, with `tile_major`, `[ceil(N/16)][K/128]` (default: tile-major up to
    TILE_MAJOR_MAX_TILES tiles)."""
    n, k = codes.shape
    assert k % BLOCK_K == 0 and codes.dtype == torch.float8_e4m3fn
    n_tiles = -(-n // TILE_N)
    if tile_major is None:
        tile_major = n_tiles <= TILE_MAJOR_MAX_TILES
    b = torch.zeros(n_tiles * TILE_N, k, dtype=torch.uint8, device=codes.device)
    b[:n] = codes.view(torch.uint8)
    # source (tile, h, g, kb, c, t, jj, e); unit order (c, g, t, jj, h, e)
    v = b.reshape(n_tiles, 2, 8, k // BLOCK_K, 4, 4, 2, 4)
    if tile_major:
        data = v.permute(0, 3, 4, 2, 5, 6, 1, 7).contiguous().reshape(n_tiles, k // BLOCK_K, UNIT)
    else:
        data = v.permute(3, 0, 4, 2, 5, 6, 1, 7).contiguous().reshape(k // BLOCK_K, n_tiles, UNIT)
    nb = -(-n // 128)
    assert scale.shape == (nb, k // BLOCK_K), (scale.shape, nb)
    return PackedFp8Weight(data, scale.float().contiguous(), n, k, tile_major)


def pack_from_bf16(w: torch.Tensor) -> PackedFp8Weight:
    """Converter from a bf16 `[N, K]` weight."""
    codes, scale = quantize_fp8_block(w)
    return pack_weight(codes, scale)


#: Shared memory one CTA may take so that a CTA of the next launch fits beside it (PDL overlap).
SMEM_BUDGET = 100 << 10


#: Measured plans (GB300, 152 SMs, CUDA-graph chain with PDL, weights rotated over > L2) for the
#: qwen3.8-flash-next-nvfp4 tp1 decode projections: (N, K) -> ((largest M, Plan fields), ...).
#: `grid` 0 is "auto" (one CTA per 16-row tile up to 9/8 x SMs, else one per SM); -t is t tiles a
#: CTA (-2: half the tiles).
_TABLE: dict[tuple[int, int], tuple[tuple[int, dict], ...]] = {
    # GDN in_proj (q|k|v|z|a|b)
    (16480, 2560): (
        (8, dict(grid=0, kw=2, tw=4, slots=3, l2_ahead=1, xreg=2)),
        (16, dict(grid=0, kw=2, tw=4, slots=3, xreg=2)),
        (32, dict(grid=0, kw=2, tw=4, slots=2, xreg=2)),
    ),
    # attention in-proj (q|gate|k|v|index)
    (14080, 2560): (
        (8, dict(grid=0, kw=2, tw=6, slots=3, l2_ahead=1, xreg=2)),
        (16, dict(grid=0, kw=2, tw=6, slots=3, xreg=2)),
        (32, dict(grid=0, kw=2, tw=6, slots=2, xreg=2)),
    ),
    # GDN out / attention o
    (2560, 6144): (
        (1, dict(grid=0, kw=12, tw=1, slots=4, xreg=1, xdepth=2)),
        (8, dict(grid=0, kw=12, tw=1, slots=4, xreg=1, xdepth=1)),
        (16, dict(grid=-2, kw=6, tw=2, slots=3, xreg=2)),
        (32, dict(grid=-2, kw=6, tw=2, slots=2, xreg=2)),
    ),
    # shared expert gate_up with its gate row, and without it
    (1281, 2560): ((32, dict(grid=0, kw=20, tw=1, slots=1, xreg=1)),),
    (1280, 2560): ((32, dict(grid=0, kw=20, tw=1, slots=1, xreg=1)),),
    # shared expert down
    (2560, 640): ((32, dict(grid=0, kw=5, tw=1, slots=1, xreg=1)),),
    # vocab projection (MTP draft head / verify head): 15520 tiles, 7 a CTA -> 2218 CTAs (about
    # 15 a multiprocessor over the launch, ~66 KB shared memory each)
    (248320, 2560): ((32, dict(grid=-7, kw=2, tw=4, slots=2, l2_ahead=1, xreg=2)),),
}

#: Most 16-row tiles a CTA takes in the heuristic plan: a ring slot holds `kw` k-blocks of each of
#: them, so a larger N gets more CTAs (several per multiprocessor over the launch) instead.
MAX_TILES_PER_CTA = 7


def default_plan(n: int, k: int, m: int, sms: int) -> Plan:
    """The plan `fp8_gemv` launches with: the measured table entry for (N, K, M), else a heuristic."""
    n_tiles = -(-n // TILE_N)
    nkb = k // BLOCK_K
    auto_grid = n_tiles if n_tiles <= sms + sms // 8 else sms
    for bound, fields in _TABLE.get((n, k), ()):
        if m <= bound:
            f = dict(fields)
            g = f.pop("grid")
            grid = auto_grid if g == 0 else -(-n_tiles // -g)
            return Plan(grid=grid, **f)
    grid = auto_grid
    rt = -(-n_tiles // grid)
    if rt > MAX_TILES_PER_CTA:
        grid = -(-n_tiles // MAX_TILES_PER_CTA)
        rt = -(-n_tiles // grid)
    divs = [d for d in range(1, nkb + 1) if nkb % d == 0]
    if rt >= 4:
        kw = max(d for d in divs if d <= 2)
        tw = min(rt, 4)
    else:
        kw = max(d for d in divs if d <= 12)
        tw = rt
    nt8 = -(-m // 8)
    xreg = (nkb // kw) * nt8 * 16 <= XREG_MAX
    slot = kw * rt * UNIT + (0 if xreg else kw * m * XRS)
    slots = max(1, min(nkb // kw, SMEM_BUDGET // slot))
    return Plan(grid=grid, kw=kw, tw=tw, slots=slots, xreg=1 if xreg else 2)


@functools.lru_cache(maxsize=None)
def _compiled(
    n, k, m, plan: Plan, pdl: bool, tile_major: bool, conv_dim: int = 0, pad_slot: int = -1
):
    kern = _Fp8Gemv(
        n=n,
        k=k,
        m=m,
        plan=plan,
        pdl=pdl,
        tile_major=tile_major,
        conv_dim=conv_dim,
        pad_slot=pad_slot,
    )
    dev = "cuda"
    out = torch.empty(m, n, dtype=torch.bfloat16, device=dev)
    x = torch.empty(m, k, dtype=torch.bfloat16, device=dev)
    w = torch.empty(16, dtype=torch.uint8, device=dev)
    s = torch.empty(4, dtype=torch.float32, device=dev)
    return cute.compile(
        kern,
        from_dlpack(out, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        make_ptr(cutlass.BFloat16, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Uint8, w.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float32, s.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        cutlass.Int32(k),
        make_ptr(cutlass.BFloat16, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=2),
        make_ptr(cutlass.BFloat16, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=2),
        make_ptr(cutlass.Int32, s.data_ptr(), cute.AddressSpace.gmem, assumed_align=4),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--opt-level 2 --enable-tvm-ffi",
    )


@functools.lru_cache(maxsize=None)
def _sms() -> int:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def _env_pdl() -> bool:
    try:
        from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
    except ImportError:
        return True
    return bool(get_env_enable_pdl())


def fp8_gemv(
    x: torch.Tensor, w: PackedFp8Weight, pdl: bool = True, plan: Plan | None = None
) -> torch.Tensor:
    """`x[M, K]` bf16 (row stride >= K, 16-byte aligned rows) @ dequant(w)^T -> fresh
    `[M, N]` bf16. M = 1 .. MAX_ROWS."""
    m, k = x.shape
    assert k == w.k and x.dtype == torch.bfloat16 and x.stride(1) == 1
    assert 1 <= m <= MAX_ROWS
    assert x.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0
    out = torch.empty((m, w.n), dtype=torch.bfloat16, device=x.device)
    if plan is None:
        plan = default_plan(w.n, k, m, _sms())
    fn = _compiled(w.n, k, m, plan, bool(pdl and _env_pdl()), w.tile_major)
    fn(out, x.data_ptr(), w.data.data_ptr(), w.scale.data_ptr(), x.stride(0), 0, 0, 0, 0, 0, 0)
    return out


def fp8_gemv_conv1d(
    x: torch.Tensor,
    w: PackedFp8Weight,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state_indices: torch.Tensor,
    conv_dim: int,
    pad_slot_id: int,
    pdl: bool = True,
    plan: Plan | None = None,
) -> torch.Tensor:
    """FP8-weight `gdn_in_proj_conv1d_pdl`: `p = round_bf16(x @ dequant(w)^T)`; for a column
    c < conv_dim of a row r whose slot s = conv_state_indices[r] is not pad_slot_id,
    `out[r, c] = round_bf16(silu(sum_j conv_weight[c, j] * [conv_state[s, c, :], p[r, c]][j]))` (fp32)
    and `conv_state[s, c, :] = [conv_state[s, c, 1:], p[r, c]]`; every other output is `p`.
    `conv_state` `[slots, conv_dim, 3]` bf16 (any strides), `conv_weight` `[conv_dim, 4]` bf16
    contiguous, `conv_state_indices` int32 `[M]`. Returns the fresh `[M, N]` output."""
    m, k = x.shape
    assert k == w.k and x.dtype == torch.bfloat16 and x.stride(1) == 1 and 1 <= m <= MAX_ROWS
    assert x.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0
    assert conv_state.dtype == torch.bfloat16 and tuple(conv_state.shape[1:]) == (conv_dim, 3)
    assert conv_weight.dtype == torch.bfloat16 and tuple(conv_weight.shape) == (conv_dim, 4)
    assert conv_weight.is_contiguous() and conv_dim <= w.n
    assert conv_state_indices.dtype == torch.int32 and conv_state_indices.is_contiguous()
    assert conv_state_indices.numel() == m
    out = torch.empty((m, w.n), dtype=torch.bfloat16, device=x.device)
    if plan is None:
        plan = default_plan(w.n, k, m, _sms())
    fn = _compiled(
        w.n, k, m, plan, bool(pdl and _env_pdl()), w.tile_major, conv_dim, int(pad_slot_id)
    )
    fn(
        out,
        x.data_ptr(),
        w.data.data_ptr(),
        w.scale.data_ptr(),
        x.stride(0),
        conv_state.data_ptr(),
        conv_weight.data_ptr(),
        conv_state_indices.data_ptr(),
        *conv_state.stride(),
    )
    return out


def _gemv_cell(why: str, n: int, k: int, m: int) -> Cell:
    return Cell(why=why, spec=dict(n=n, k=k, m=m))


class Fp8Gemv(OpWrapper):
    """`fp8_gemv(x, w)`: `y[M, N] = x[M, K] @ dequant(w)^T` in bf16 for 1..MAX_ROWS rows, the
    weight a `PackedFp8Weight` (e4m3 codes with 128x128 fp32 block scales, packed by
    `pack_weight` / `pack_from_bf16`), the bf16 activation never quantized. fp32 accumulation per
    128-wide k-block, block scale applied in fp32, one bf16 rounding at the store.

    Fusion boundary: the projection; `fp8_gemv_conv1d` also fuses the GDN causal-conv step."""

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        _gemv_cell("GDN in_proj, one decode row", 16480, 2560, 1),
        _gemv_cell("attention in-proj, MTP3 verify at con1 (4 rows)", 14080, 2560, 4),
        _gemv_cell("GDN out / attention o, 4 rows", 2560, 6144, 4),
        _gemv_cell("shared expert gate|up, 16 rows", 1280, 2560, 16),
        _gemv_cell("shared expert down, 8 rows", 2560, 640, 8),
        _gemv_cell("vocab projection (draft / verify head), 4 rows", 248320, 2560, 4),
    )

    note = """
    Metadata consumed: none. Call sites: the decode target's dense projections under the dense
    FP8 mode (`TRTLLM_MODELING_V2_DENSE_FP8`) at up to `_FP8_GEMV_MAX_ROWS` rows.
    Preconditions: x bf16 `[M, K]`, unit column stride, row stride a multiple of 8 elements and a
    16-byte aligned base; K a multiple of 128. One PDL launch (weights streamed before the
    wait); CUDA-graph capturable; compiled per (N, K, M, plan) on first use.
    """

    def __call__(self, x: torch.Tensor, w: PackedFp8Weight) -> torch.Tensor:
        return fp8_gemv(x, w)

    def reference(self, x: torch.Tensor, codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """fp32 `[M, N]` over the dequantized weight (a true fp32 matmul)."""
        with true_fp32_matmul():
            return x.float() @ dequantize_fp8_block(codes, scale).t()


class Fp8GemvConv1d(OpWrapper):
    """`fp8_gemv_conv1d`: the FP8-weight `gdn_in_proj_conv1d_pdl` -- the GDN input projection
    of decode rows, then for the conv columns of a live row the causal conv step against its
    paged state (SiLU; the state advanced in place). Returns the `[M, N]` projection row with
    the conv columns replaced by the conv output."""

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="STP decode: one row, live slot", spec=dict(slots=(37,))),
        Cell(why="two rows, one padding slot", spec=dict(slots=(17, -1))),
    )

    note = """
    Call site: the decode target's one-row GDN in_proj + conv step under the dense FP8 mode.
    `conv_state` `[slots, conv_dim, 3]` bf16 (any strides), `conv_weight` `[conv_dim, 4]` bf16
    contiguous, int32 slots; a `pad_slot_id` row gets the plain projection.
    """

    def __call__(self, x, w, conv_state, conv_weight, conv_state_indices, conv_dim, pad_slot_id):
        return fp8_gemv_conv1d(
            x, w, conv_state, conv_weight, conv_state_indices, conv_dim, pad_slot_id
        )

    def reference(self, *args, **kwargs):
        raise NotImplementedError("the test builds the fp32 reference from the dequantized weight")


fp8_gemv_op = Fp8Gemv()
fp8_gemv_conv1d_op = Fp8GemvConv1d()
