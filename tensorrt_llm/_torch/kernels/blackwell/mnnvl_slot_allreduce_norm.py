# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""One-shot all-reduce + residual + RMSNorm on a flag-free slot protocol (CuTe DSL).

``MnnvlSlotAllReduceNormKernel`` has the arithmetic, thread layout and fold options of
``MnnvlAllReduceNormKernel`` (outputs equal its one-shot outputs bit for bit) but runs on its own multicast
workspace without ``buffer_flags``:

* CTA ``b`` owns, per region, the slot block [b][rank][threads] (geometry must be fixed per workspace). Its
  participation count ``n_b`` (int32 at ``ctl + 128 * b``, read by thread 0, written back + 1) selects region
  ``n_b % 3``: multicast into it, clear own region ``(n_b + 2) % 3``, poll ``n_b % 3`` (-0.0 = not written).
* Safety: a peer is at most one participation of CTA ``b`` ahead (it needs this rank's data in between), so
  it writes regions ``n_b`` / ``n_b + 1`` only, never the one being cleared.
* Cross-CTA norm sums use st.async onto a per-CTA mbarrier (no barrier.cluster).

Before ``griddepcontrol.wait`` only the norm weight and shared memory / cluster state are touched.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass import Float32, Int32, Int64, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from .mnnvl_allreduce_norm import (
    _NEG_ZERO,
    VEC,
    _add,
    _bf16r,
    _cluster_sync,
    _div_rn,
    _e4m3x2,
    _fma,
    _hi,
    _ld_s32,
    _ld_shared_f32,
    _ld_u16,
    _ld_v4,
    _ld_volatile_v4,
    _lo,
    _max0,
    _mul,
    _pack_bf16,
    _rsqrt,
    _sanitize,
    _sel_ge0,
    _shfl_bfly,
    _sigmoid_triton,
    _sink4,
    _smem_addr,
    _st_s32,
    _st_shared_f32,
    _st_v2,
    _st_v4,
)
from .utils import griddepcontrol_launch_dependents, griddepcontrol_wait

SLOT_STATE_STRIDE = 128  # bytes between per-CTA state words (one L2 line each)


@dsl_user_op
def _ld_cg_s32(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.cg.b32 $0, [$1];",
            "=r,l,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _mbar_init(addr: Int32, count: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Int32(count).ir_value(loc=loc, ip=ip)],
        "mbarrier.init.shared::cta.b64 [$0], $1;\n\tfence.mbarrier_init.release.cluster;",
        "r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mbar_expect_tx(addr: Int32, nbytes: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Int32(nbytes).ir_value(loc=loc, ip=ip)],
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1;",
        "r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mbar_try_wait(addr: Int32, parity: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(addr).ir_value(loc=loc, ip=ip), Int32(parity).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2;\n\t"
            "selp.u32 $0, 1, 0, p;\n\t}",
            "=r,r,r,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_async_f32(
    addr: Int32, mbar: Int32, cta_rank: Int32, value: Float32, *, loc=None, ip=None
) -> None:
    """st.async one f32 into cluster CTA ``cta_rank``'s smem at ``addr``, completing 4 tx bytes on its ``mbar``."""
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Int32(mbar).ir_value(loc=loc, ip=ip),
            Int32(cta_rank).ir_value(loc=loc, ip=ip),
            Float32(value).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .b32 ra, rb, v;\n\tmapa.shared::cluster.u32 ra, $0, $2;\n\t"
        "mapa.shared::cluster.u32 rb, $1, $2;\n\tmov.b32 v, $3;\n\t"
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [ra], v, [rb];\n\t}",
        "r,r,r,f,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _tree_sum_smem(red_addr, n: int):
    """The CTA butterfly's result over n warp sums at smem ``red_addr``, same additions in the same order
    (the butterfly's extra zero lanes are exact no-ops)."""
    vals = [_ld_shared_f32(red_addr + Int32(4 * i)) for i in range(n)]
    s = n // 2
    while s >= 1:
        vals = [_add(vals[i], vals[i ^ s]) for i in range(n)]
        s //= 2
    return vals[0]


class MnnvlSlotAllReduceNormKernel:
    """Tensors as ``MnnvlAllReduceNormKernel``. ``own`` / ``mcast``: this rank's unicast / multicast base of the
    slot workspace; ``ctl``: the per-CTA state array; ``region_bytes``: size of one of the 3 regions."""

    def __init__(
        self,
        *,
        hidden: int,
        norm_eps: float,
        nranks: int,
        ctas: int,
        use_pdl: bool,
        se_fold: bool,
        norm_out: bool,
        quant: bool,
        region_bytes: int,
        moe_topk: int = 0,
        fc2_row: int = 0,
    ):
        if hidden % (VEC * ctas) or (hidden // VEC // ctas) % 32 or hidden // VEC // ctas > 1024:
            raise ValueError(f"unsupported hidden size {hidden} for {ctas} CTAs per token")
        if nranks not in (2, 4, 8, 16) or ctas not in (1, 2, 4, 8):
            raise ValueError("unsupported group / cluster size")
        if region_bytes <= 0:
            raise ValueError("the slot protocol needs region_bytes")
        self.hidden = hidden
        self.norm_eps = float(norm_eps)
        self.nranks = nranks
        self.ctas = ctas
        self.threads = hidden // VEC // ctas
        self.num_warps = self.threads // 32
        self.use_pdl = use_pdl
        self.se_fold = se_fold
        self.moe_topk = moe_topk
        self.fc2_row = fc2_row or hidden
        self.norm_out = norm_out
        self.quant = quant
        self.region_bytes = region_bytes

    @cute.jit
    def __call__(
        self,
        gX: cute.Tensor,
        gRouted: cute.Tensor,
        gShared: cute.Tensor,
        gGate: cute.Tensor,
        gRes: cute.Tensor,
        gW: cute.Tensor,
        gScale: cute.Tensor,
        gResOut: cute.Tensor,
        gOut: cute.Tensor,
        gQuant: cute.Tensor,
        own: Int64,
        mcast: Int64,
        ctl: Int64,
        fc2_ptr: Int64,
        topw_ptr: Int64,
        perm_ptr: Int64,
        rank: Int32,
        stream,
    ) -> None:
        m = cute.size(gRes, mode=[0])
        self.kernel(
            gX,
            gRouted,
            gShared,
            gGate,
            gRes,
            gW,
            gScale,
            gResOut,
            gOut,
            gQuant,
            own,
            mcast,
            ctl,
            fc2_ptr,
            topw_ptr,
            perm_ptr,
            rank,
        ).launch(
            grid=[m * self.ctas, 1, 1],
            block=[self.threads, 1, 1],
            cluster=[self.ctas, 1, 1],
            smem=512,
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        gX: cute.Tensor,
        gRouted: cute.Tensor,
        gShared: cute.Tensor,
        gGate: cute.Tensor,
        gRes: cute.Tensor,
        gW: cute.Tensor,
        gScale: cute.Tensor,
        gResOut: cute.Tensor,
        gOut: cute.Tensor,
        gQuant: cute.Tensor,
        own: Int64,
        mcast: Int64,
        ctl: Int64,
        fc2_ptr: Int64,
        topw_ptr: Int64,
        perm_ptr: Int64,
        rank: Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        ctas: cutlass.Constexpr = self.ctas
        nranks: cutlass.Constexpr = self.nranks
        vecs: cutlass.Constexpr = self.hidden // VEC
        num_warps: cutlass.Constexpr = self.num_warps
        threads: cutlass.Constexpr = self.threads
        tok = bid // ctas
        part = bid % ctas
        vi = part * threads + tidx
        warp = tidx // 32
        lane = tidx % 32
        row_off = (Int64(tok) * Int64(vecs) + Int64(vi)) * Int64(16)

        smem = cutlass.utils.SmemAllocator()
        red = smem.allocate_tensor(Float32, cute.make_layout((num_warps,)), byte_alignment=16)
        csum = smem.allocate_tensor(Float32, cute.make_layout((ctas,)), byte_alignment=16)
        fl = smem.allocate_tensor(Int32, cute.make_layout((8,)), byte_alignment=16)
        red_addr = _smem_addr(red.iterator)
        csum_addr = _smem_addr(csum.iterator)

        mbar_addr = Int32(0)
        if const_expr(ctas > 1):
            # Initialized cluster-wide before any CTA can send (smem / cluster state only: allowed before
            # the dependency wait).
            mb = smem.allocate_tensor(Int64, cute.make_layout((1,)), byte_alignment=8)
            mbar_addr = _smem_addr(mb.iterator)
            if tidx == 0:
                _mbar_init(mbar_addr, Int32(1))
            _cluster_sync()
            if tidx == 0:
                _mbar_expect_tx(mbar_addr, Int32(4 * ctas))

        # ---- before the dependency wait: only the norm weight ----
        w0, w1, w2, w3 = _ld_v4(Int64(gW.iterator.toint()) + Int64(vi) * Int64(16))
        topk: cutlass.Constexpr = self.moe_topk
        pk = cute.make_rmem_tensor((max(topk, 1),), Int32)
        wk = cute.make_rmem_tensor((max(topk, 1),), Int32)

        if const_expr(self.use_pdl):
            griddepcontrol_wait()

        r0, r1, r2, r3 = _ld_v4(Int64(gRes.iterator.toint()) + row_off)
        if const_expr(self.se_fold):
            if const_expr(topk > 0):
                # routing outputs of this step: only after the dependency wait (RAW)
                kbase = Int64(tok) * Int64(topk)
                for k in cutlass.range_constexpr(topk):
                    pk[k] = _ld_s32(perm_ptr + (kbase + Int64(k)) * Int64(4))
                    wk[k] = _ld_u16(topw_ptr + (kbase + Int64(k)) * Int64(2))
            else:
                f0, f1, f2, f3 = _ld_v4(Int64(gRouted.iterator.toint()) + row_off)
            h0, h1, h2, h3 = _ld_v4(Int64(gShared.iterator.toint()) + row_off)
            gw = _ld_u16(Int64(gGate.iterator.toint()) + Int64(tok) * Int64(2))
        else:
            x0, x1, x2, x3 = _ld_v4(Int64(gX.iterator.toint()) + row_off)

        # ---- this CTA's participation count, read by one thread and shared through smem ----
        if tidx == 0:
            fl[0] = _ld_cg_s32(ctl + Int64(bid) * Int64(SLOT_STATE_STRIDE))
        cute.arch.sync_threads()

        if const_expr(self.se_fold):
            if const_expr(topk > 0):
                ev = cute.make_rmem_tensor((topk * 4,), Int32)
                for k in cutlass.range_constexpr(topk):
                    e0, e1, e2, e3 = _ld_v4(
                        fc2_ptr
                        + (Int64(_max0(pk[k])) * Int64(self.fc2_row) + Int64(vi) * Int64(VEC))
                        * Int64(2)
                    )
                    ev[4 * k + 0] = e0
                    ev[4 * k + 1] = e1
                    ev[4 * k + 2] = e2
                    ev[4 * k + 3] = e3
                acc = cute.make_rmem_tensor((VEC,), Float32)
                for e in cutlass.range_constexpr(VEC):
                    acc[e] = Float32(0.0)
                for k in cutlass.range_constexpr(topk):
                    wf = _lo(wk[k])
                    for i in cutlass.range_constexpr(4):
                        acc[2 * i] = _sel_ge0(
                            pk[k], _fma(wf, _lo(ev[4 * k + i]), acc[2 * i]), acc[2 * i]
                        )
                        acc[2 * i + 1] = _sel_ge0(
                            pk[k], _fma(wf, _hi(ev[4 * k + i]), acc[2 * i + 1]), acc[2 * i + 1]
                        )
                f0 = _pack_bf16(acc[0], acc[1])
                f1 = _pack_bf16(acc[2], acc[3])
                f2 = _pack_bf16(acc[4], acc[5])
                f3 = _pack_bf16(acc[6], acc[7])
            g = _lo(gw)
            s = _sigmoid_triton(g)
            x0 = _pack_bf16(_fma(s, _lo(h0), _lo(f0)), _fma(s, _hi(h0), _hi(f0)))
            x1 = _pack_bf16(_fma(s, _lo(h1), _lo(f1)), _fma(s, _hi(h1), _hi(f1)))
            x2 = _pack_bf16(_fma(s, _lo(h2), _lo(f2)), _fma(s, _hi(h2), _hi(f2)))
            x3 = _pack_bf16(_fma(s, _lo(h3), _lo(f3)), _fma(s, _hi(h3), _hi(f3)))

        # ---- one multicast store into this participation's region of every rank ----
        neg0 = Int32(_NEG_ZERO)
        slot_cnt = fl[0]
        blk_bytes: cutlass.Constexpr = nranks * threads * 16  # [W slots] of this CTA
        data_off = Int64(slot_cnt % Int32(3)) * Int64(self.region_bytes) + Int64(bid) * Int64(
            blk_bytes
        )
        _st_v4(
            mcast + data_off + (Int64(rank) * Int64(threads) + Int64(tidx)) * Int64(16),
            _sanitize(x0),
            _sanitize(x1),
            _sanitize(x2),
            _sanitize(x3),
        )
        poll_base = own + data_off + Int64(tidx) * Int64(16)

        # ---- PDL: release dependents once every recyclable input is consumed (a dependent's pre-wait write
        # into a recycled block must not race these reads) ----
        if const_expr(self.use_pdl):
            _sink4(red_addr, r0, r1, r2, r3)
            cute.arch.sync_threads()
            griddepcontrol_launch_dependents()

        # ---- count the participation; clear this thread's slots of the previous participation's region ----
        if tidx == 0:
            _st_s32(ctl + Int64(bid) * Int64(SLOT_STATE_STRIDE), slot_cnt + Int32(1))
        pq = (slot_cnt + Int32(2)) % Int32(3)
        pbase = (
            own
            + Int64(pq) * Int64(self.region_bytes)
            + Int64(bid) * Int64(blk_bytes)
            + Int64(tidx) * Int64(16)
        )
        for r in cutlass.range_constexpr(nranks):
            _st_v4(pbase + Int64(r * threads * 16), neg0, neg0, neg0, neg0)

        # ---- wait for this vector of every rank ----
        rstride = Int64(threads * 16)
        got = cute.make_rmem_tensor((nranks * 4,), Int32)
        pending = Int32(1)
        while pending != 0:
            pending = Int32(0)
            for r in cutlass.range_constexpr(nranks):
                a, b, cc, d = _ld_volatile_v4(poll_base + Int64(r) * rstride)
                got[r * 4 + 0] = a
                got[r * 4 + 1] = b
                got[r * 4 + 2] = cc
                got[r * 4 + 3] = d
                if a == neg0:
                    pending = Int32(1)
                if b == neg0:
                    pending = Int32(1)
                if cc == neg0:
                    pending = Int32(1)
                if d == neg0:
                    pending = Int32(1)

        # ---- FP32 sum in rank order -> BF16; + residual (BF16) ----
        rw = (r0, r1, r2, r3)
        ww = (w0, w1, w2, w3)
        res = cute.make_rmem_tensor((VEC,), Float32)
        for i in cutlass.range_constexpr(4):
            lo = Float32(0.0)
            hi = Float32(0.0)
            for r in cutlass.range_constexpr(nranks):
                lo = _add(lo, _lo(got[r * 4 + i]))
                hi = _add(hi, _hi(got[r * 4 + i]))
            res[2 * i] = _bf16r(_add(_bf16r(lo), _lo(rw[i])))
            res[2 * i + 1] = _bf16r(_add(_bf16r(hi), _hi(rw[i])))
        out_off = Int64(gResOut.iterator.toint()) + row_off
        _st_v4(
            out_off,
            _pack_bf16(res[0], res[1]),
            _pack_bf16(res[2], res[3]),
            _pack_bf16(res[4], res[5]),
            _pack_bf16(res[6], res[7]),
        )

        # ---- RMSNorm: BF16 squares, per-thread sum in element order, warp / CTA / cluster sums ----
        tsum = Float32(0.0)
        for e in cutlass.range_constexpr(VEC):
            tsum = _add(tsum, _bf16r(_mul(res[e], res[e])))
        for k in cutlass.range_constexpr(5):
            tsum = _add(tsum, _shfl_bfly(tsum, Int32(16 >> k)))
        if lane == 0:
            _st_shared_f32(red_addr + Int32(4) * warp, tsum)
        cute.arch.sync_threads()
        total = _tree_sum_smem(red_addr, num_warps)
        if const_expr(ctas > 1):
            if tidx < ctas:
                _st_async_f32(csum_addr + Int32(4) * part, mbar_addr, Int32(tidx), total)
            ready = _mbar_try_wait(mbar_addr, Int32(0))
            while ready == 0:
                ready = _mbar_try_wait(mbar_addr, Int32(0))
            total = Float32(0.0)
            for p in cutlass.range_constexpr(ctas):
                total = _add(total, _ld_shared_f32(csum_addr + Int32(4 * p)))
        rstd = _rsqrt(_add(_div_rn(total, Float32(float(self.hidden))), Float32(self.norm_eps)))
        y = cute.make_rmem_tensor((VEC,), Float32)
        for i in cutlass.range_constexpr(4):  # (x * rstd) * w
            y[2 * i] = _bf16r(_mul(_mul(res[2 * i], rstd), _lo(ww[i])))
            y[2 * i + 1] = _bf16r(_mul(_mul(res[2 * i + 1], rstd), _hi(ww[i])))
        if const_expr(self.norm_out):
            _st_v4(
                Int64(gOut.iterator.toint()) + row_off,
                _pack_bf16(y[0], y[1]),
                _pack_bf16(y[2], y[3]),
                _pack_bf16(y[4], y[5]),
                _pack_bf16(y[6], y[7]),
            )
        if const_expr(self.quant):
            inv = _div_rn(Float32(1.0), gScale[0])
            q01 = _e4m3x2(_mul(y[0], inv), _mul(y[1], inv))
            q23 = _e4m3x2(_mul(y[2], inv), _mul(y[3], inv))
            q45 = _e4m3x2(_mul(y[4], inv), _mul(y[5], inv))
            q67 = _e4m3x2(_mul(y[6], inv), _mul(y[7], inv))
            _st_v2(
                Int64(gQuant.iterator.toint()) + (Int64(tok) * Int64(vecs) + Int64(vi)) * Int64(8),
                q01 | (q23 << 16),
                q45 | (q67 << 16),
            )
