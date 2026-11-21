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
  participation index ``n_b`` in 0..2 (int32 at ``ctl + 128 * b``, read by thread 0, written back as
  ``(n_b + 1) % 3``) selects region ``n_b``: multicast into it, clear own region ``(n_b + 2) % 3``, poll ``n_b``
  (-0.0 = not written).
* Safety: a peer is at most one participation of CTA ``b`` ahead (it needs this rank's data in between), so
  it writes regions ``n_b`` / ``(n_b + 1) % 3`` only, never the one being cleared.
* Cross-CTA norm sums use st.async onto a per-CTA mbarrier (no barrier.cluster).

``ts_ctas`` > 0 (rows from ``TWO_SHOT_ROWS``): the same one-shot transport with the two-shot kernel's
arithmetic for a two-shot launch of ``ts_ctas`` CTAs per token, so the outputs equal ``MnnvlAllReduceNormKernel``'s
two-shot outputs bit for bit: the rank-order FP32 sum is packed to BF16 and sanitized before the residual add (the
owner's broadcast word), the norm's sum of squares is the two-shot CTA butterfly over that CTA's warps, then summed in
CTA order (the cluster exchange carries every warp's sum, not one per CTA), and the norm output is ``(w * x) * rstd``.

Before ``griddepcontrol.wait`` only the norm weight and shared memory / cluster state are touched, except in the
no-wait Lamport mode (``fc2_lamport``): there the fold reads the residual, the routing tables (relaxed loads), the
shared expert's output and gate, and polls FC2's rows, all before the wait, which comes only before the stores. That
is safe because none of these is written by a kernel still running when this launch starts: FC2 (the PDL primary)
releases it only after its own dependency wait and, for an engine that routes, after it wrote the routing tables, so
the residual's producer has completed; the shared expert runs on the aux stream, which the caller fences and joins
before this launch (a full stream dependency, not PDL); FC2's own rows are polled per 16-byte vector against
0xFFFFFFFF. The stores wait because the outputs may alias allocator blocks FC2 still reads.
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
    l2_prefetch_slice,
)
from .utils import griddepcontrol_launch_dependents, griddepcontrol_wait

SLOT_STATE_STRIDE = 128  # bytes between per-CTA state words (one L2 line each)
#: The word FC2 never writes (0xFFFFFFFF: two BF16 0xFFFF; FC2's ``cvt.rn.satfinite.bf16x2.f32`` NaN is 0x7FFF):
#: with ``fc2_lamport`` the MoE fold reads FC2's rows while FC2 still runs and waits per 16-byte vector for it to go.
_FC2_SENT = -1


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
def _ld_relaxed_s32(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.relaxed.gpu.global.b32 $0, [$1];",
            "=r,l,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_relaxed_u16(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b16 h;\n\tld.relaxed.gpu.global.b16 h, [$1];\n\tcvt.u32.u16 $0, h;\n\t}",
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
        l2pf: bool = False,
        fc2_lamport: bool = False,
        ts_ctas: int = 0,
    ):
        if hidden % (VEC * ctas) or (hidden // VEC // ctas) % 32 or hidden // VEC // ctas > 1024:
            raise ValueError(f"unsupported hidden size {hidden} for {ctas} CTAs per token")
        if nranks not in (2, 4, 8, 16) or ctas not in (1, 2, 4, 8):
            raise ValueError("unsupported group / cluster size")
        if region_bytes <= 0:
            raise ValueError("the slot protocol needs region_bytes")
        token_warps = hidden // VEC // 32
        if ts_ctas and (
            ts_ctas not in (1, 2, 4, 8)
            or token_warps % ts_ctas
            or not 1 <= token_warps // ts_ctas <= 32
            or (token_warps // ts_ctas) & (token_warps // ts_ctas - 1)
        ):
            raise ValueError(f"unsupported two-shot CTA split {ts_ctas} for hidden size {hidden}")
        # the two-shot kernel's arithmetic for its launch of ts_ctas CTAs per token (module docstring)
        self.ts_ctas = ts_ctas
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
        self.l2pf = l2pf  # prefetch pf_ptr's slice into L2 after this CTA's pushes
        # fc2_lamport (moe_topk > 0): fc2 is a Lamport buffer (0xFFFFFFFF where FC2 has not written yet), polled
        # instead of a dependency wait before the fold; FC2 is the PDL primary, the aux-stream inputs are fenced by
        # the caller (MnnvlAllReduceNorm fc2_lamport). The vectors read are written back to 0xFFFFFFFF.
        self.fc2_lamport = fc2_lamport
        # fc2_lamport: outputs come from the caching allocator and may alias blocks FC2 still reads, so every store
        # waits for the dependency (which also orders this kernel's end after FC2's).
        self.store_wait = bool(use_pdl and fc2_lamport)
        if fc2_lamport and not (se_fold and moe_topk > 0):
            raise ValueError("fc2_lamport needs the in-kernel MoE finalize")

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
        pf_ptr: Int64,
        pf_bytes: Int64,
        pf_per: Int64,
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
            pf_ptr,
            pf_bytes,
            pf_per,
        ).launch(
            grid=[m * self.ctas, 1, 1],
            block=[self.threads, 1, 1],
            cluster=[self.ctas, 1, 1],
            smem=self.smem_bytes(),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    def smem_bytes(self) -> int:
        """Dynamic shared memory of a launch: ``kernel``'s allocations (warp sums, cluster sums, flags, mbarrier,
        each 16-byte aligned), and at least 512 bytes. Nothing in the kernel reads beyond its allocations; the floor
        keeps the launch of every hidden size up to 16384, whose allocations fit in it, at the shared-memory size
        its decode-step timings were measured with (occupancy is bound by threads and registers, not by this)."""

        def up16(n: int) -> int:
            return (n + 15) // 16 * 16

        nsum = self.ctas * self.num_warps if self.ts_ctas else self.ctas
        need = up16(4 * self.num_warps) + up16(4 * nsum) + 32 + (8 if self.ctas > 1 else 0)
        return max(512, up16(need))

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
        pf_ptr: Int64,
        pf_bytes: Int64,
        pf_per: Int64,
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

        ts_ctas: cutlass.Constexpr = self.ts_ctas
        # cluster exchange: one sum per CTA, or (ts_ctas) one per warp of the token, in global warp order
        nsum: cutlass.Constexpr = ctas * num_warps if ts_ctas else ctas
        smem = cutlass.utils.SmemAllocator()
        red = smem.allocate_tensor(Float32, cute.make_layout((num_warps,)), byte_alignment=16)
        csum = smem.allocate_tensor(Float32, cute.make_layout((nsum,)), byte_alignment=16)
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
                _mbar_expect_tx(mbar_addr, Int32(4 * nsum))

        # ---- before the dependency wait: only the norm weight (with fc2_lamport also the fold's inputs below: there
        # the wait comes before the stores; see the module docstring) ----
        w0, w1, w2, w3 = _ld_v4(Int64(gW.iterator.toint()) + Int64(vi) * Int64(16))
        topk: cutlass.Constexpr = self.moe_topk
        pk = cute.make_rmem_tensor((max(topk, 1),), Int32)
        wk = cute.make_rmem_tensor((max(topk, 1),), Int32)

        if const_expr(self.use_pdl and not self.fc2_lamport):
            griddepcontrol_wait()

        r0, r1, r2, r3 = _ld_v4(Int64(gRes.iterator.toint()) + row_off)
        if const_expr(self.se_fold):
            if const_expr(topk > 0):
                # routing outputs of this step: after the dependency wait (RAW), or, with fc2_lamport, written before
                # the dependency released this launch (an engine that routes releases it only after its tables)
                kbase = Int64(tok) * Int64(topk)
                for k in cutlass.range_constexpr(topk):
                    pa = perm_ptr + (kbase + Int64(k)) * Int64(4)
                    wa = topw_ptr + (kbase + Int64(k)) * Int64(2)
                    if const_expr(self.fc2_lamport):
                        pk[k] = _ld_relaxed_s32(pa)
                        wk[k] = _ld_relaxed_u16(wa)
                    else:
                        pk[k] = _ld_s32(pa)
                        wk[k] = _ld_u16(wa)
            else:
                f0, f1, f2, f3 = _ld_v4(Int64(gRouted.iterator.toint()) + row_off)
            h0, h1, h2, h3 = _ld_v4(Int64(gShared.iterator.toint()) + row_off)
            gw = _ld_u16(Int64(gGate.iterator.toint()) + Int64(tok) * Int64(2))
        else:
            x0, x1, x2, x3 = _ld_v4(Int64(gX.iterator.toint()) + row_off)

        # ---- this CTA's participation index, read by one thread and shared through smem ----
        if tidx == 0:
            fl[0] = _ld_cg_s32(ctl + Int64(bid) * Int64(SLOT_STATE_STRIDE))
        cute.arch.sync_threads()

        if const_expr(self.se_fold):
            if const_expr(topk > 0):
                ev = cute.make_rmem_tensor((topk * 4,), Int32)
                for k in cutlass.range_constexpr(topk):
                    fa = fc2_ptr + (
                        Int64(_max0(pk[k])) * Int64(self.fc2_row) + Int64(vi) * Int64(VEC)
                    ) * Int64(2)
                    if const_expr(self.fc2_lamport):
                        # real rows only: a word has exactly one reader, which resets it (non-local k stay 0 and
                        # are dropped by the select below)
                        e0 = Int32(0)
                        e1 = Int32(0)
                        e2 = Int32(0)
                        e3 = Int32(0)
                        if pk[k] >= 0:
                            e0, e1, e2, e3 = _ld_volatile_v4(fa)
                    else:
                        e0, e1, e2, e3 = _ld_v4(fa)
                    ev[4 * k + 0] = e0
                    ev[4 * k + 1] = e1
                    ev[4 * k + 2] = e2
                    ev[4 * k + 3] = e3
                if const_expr(self.fc2_lamport):
                    # Lamport: FC2 writes every word of a real row once and never the word 0xFFFFFFFF (its BF16 NaN
                    # is 0x7FFF), so a local row's vector is complete once none of its words is that word. Non-local
                    # k (perm < 0) read row 0 above and are dropped by the select below, as without Lamport.
                    pending = Int32(1)
                    while pending != 0:
                        pending = Int32(0)
                        for k in cutlass.range_constexpr(topk):
                            if pk[k] >= 0:
                                bad = Int32(0)
                                for i in cutlass.range_constexpr(4):
                                    if ev[4 * k + i] == Int32(_FC2_SENT):
                                        bad = Int32(1)
                                if bad != 0:
                                    fb = fc2_ptr + (
                                        Int64(pk[k]) * Int64(self.fc2_row) + Int64(vi) * Int64(VEC)
                                    ) * Int64(2)
                                    g0, g1, g2, g3 = _ld_volatile_v4(fb)
                                    ev[4 * k + 0] = g0
                                    ev[4 * k + 1] = g1
                                    ev[4 * k + 2] = g2
                                    ev[4 * k + 3] = g3
                                    pending = Int32(1)
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
                if const_expr(self.fc2_lamport):
                    # This thread is the only reader of these vectors: hand them back to the next FC2 (it writes
                    # after kernel boundaries that follow this kernel's end).
                    for k in cutlass.range_constexpr(topk):
                        if pk[k] >= 0:
                            fr = fc2_ptr + (
                                Int64(pk[k]) * Int64(self.fc2_row) + Int64(vi) * Int64(VEC)
                            ) * Int64(2)
                            _st_v4(
                                fr,
                                Int32(_FC2_SENT),
                                Int32(_FC2_SENT),
                                Int32(_FC2_SENT),
                                Int32(_FC2_SENT),
                            )
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
        data_off = Int64(slot_cnt) * Int64(self.region_bytes) + Int64(bid) * Int64(blk_bytes)
        _st_v4(
            mcast + data_off + (Int64(rank) * Int64(threads) + Int64(tidx)) * Int64(16),
            _sanitize(x0),
            _sanitize(x1),
            _sanitize(x2),
            _sanitize(x3),
        )
        poll_base = own + data_off + Int64(tidx) * Int64(16)

        if const_expr(self.l2pf):
            # The next GEMM's weight into L2 after this thread's push, so the burst overlaps the poll / reduce /
            # norm phase rather than the fold's gathers (one thread, not thread 0).
            if tidx == self.threads - 1:
                l2_prefetch_slice(pf_ptr, pf_bytes, pf_per, bid)

        # ---- PDL: release dependents once every recyclable input is consumed (a dependent's pre-wait write
        # into a recycled block must not race these reads) ----
        if const_expr(self.use_pdl):
            _sink4(red_addr, r0, r1, r2, r3)
            cute.arch.sync_threads()
            griddepcontrol_launch_dependents()

        # ---- advance the participation index; clear this thread's slots of the previous participation's region ----
        if tidx == 0:
            _st_s32(ctl + Int64(bid) * Int64(SLOT_STATE_STRIDE), (slot_cnt + Int32(1)) % Int32(3))
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
            if const_expr(ts_ctas > 0):
                # the two-shot owner's broadcast word: BF16 pair, each -0.0 as +0.0
                sw = _sanitize(_pack_bf16(lo, hi))
                res[2 * i] = _bf16r(_add(_lo(sw), _lo(rw[i])))
                res[2 * i + 1] = _bf16r(_add(_hi(sw), _hi(rw[i])))
            else:
                res[2 * i] = _bf16r(_add(_bf16r(lo), _lo(rw[i])))
                res[2 * i + 1] = _bf16r(_add(_bf16r(hi), _hi(rw[i])))
        out_off = Int64(gResOut.iterator.toint()) + row_off
        if const_expr(not self.store_wait):
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
        if const_expr(ts_ctas > 0):
            # Two-shot arithmetic: warp sums at their global warp index (part * num_warps + w), a butterfly per
            # two-shot CTA q over warps q * wpc .., the CTA sums in order (lane 0 of a full XOR butterfly stands for
            # every lane: commutative adds, zero lanes add +0.0 to non-negative sums).
            wpc: cutlass.Constexpr = (ctas * num_warps) // ts_ctas
            src = red_addr
            if const_expr(ctas > 1):
                src = csum_addr
                if tidx < ctas * num_warps:
                    _st_async_f32(
                        csum_addr + Int32(4) * (part * Int32(num_warps) + tidx % Int32(num_warps)),
                        mbar_addr,
                        tidx // Int32(num_warps),
                        _ld_shared_f32(red_addr + Int32(4) * (tidx % Int32(num_warps))),
                    )
                ready = _mbar_try_wait(mbar_addr, Int32(0))
                while ready == 0:
                    ready = _mbar_try_wait(mbar_addr, Int32(0))
            if const_expr(ts_ctas == 1):
                total = _tree_sum_smem(src, wpc)
            else:
                total = Float32(0.0)
                for q in cutlass.range_constexpr(ts_ctas):
                    total = _add(total, _tree_sum_smem(src + Int32(4 * q * wpc), wpc))
        else:
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
        if const_expr(ts_ctas > 0):
            for i in cutlass.range_constexpr(4):  # the two-shot's (w * x) * rstd
                y[2 * i] = _bf16r(_mul(_mul(_lo(ww[i]), res[2 * i]), rstd))
                y[2 * i + 1] = _bf16r(_mul(_mul(_hi(ww[i]), res[2 * i + 1]), rstd))
        else:
            for i in cutlass.range_constexpr(4):  # (x * rstd) * w
                y[2 * i] = _bf16r(_mul(_mul(res[2 * i], rstd), _lo(ww[i])))
                y[2 * i + 1] = _bf16r(_mul(_mul(res[2 * i + 1], rstd), _hi(ww[i])))
        if const_expr(self.store_wait):
            # fc2_lamport: every store into allocator memory after the dependency wait (see __init__)
            griddepcontrol_wait()
            _st_v4(
                out_off,
                _pack_bf16(res[0], res[1]),
                _pack_bf16(res[2], res[3]),
                _pack_bf16(res[4], res[5]),
                _pack_bf16(res[6], res[7]),
            )
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
