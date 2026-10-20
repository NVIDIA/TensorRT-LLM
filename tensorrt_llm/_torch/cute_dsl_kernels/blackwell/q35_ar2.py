# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""All-reduce + residual + RMSNorm kernel variants (Qwen3.5 decode, CuTe DSL).

``Q35Ar2Kernel`` keeps the arithmetic, thread layout (``ctas`` CTAs of 16-byte vectors per token, cluster
reduction of the norm) and fold options (shared-expert combine, MoE finalize) of ``Q35AllReduceNormKernel``
(q35_allreduce_norm.py), and adds protocol options:

``proto="flags"``: the library MNNVL workspace protocol (triple buffer + ``buffer_flags``), bitwise the same as
    ``Q35AllReduceNormKernel`` one-shot, with
    * the workspace addresses as kernel arguments (no device table load before the flags load),
    * ``rotate="end"`` (stock: CTA 0 waits for every CTA's arrival at its end and rotates the buffers) or
      ``rotate="poll"`` (thread 0 of CTA 0 checks the arrivals inside its poll loop and rotates as soon as they
      are complete, so the rotation is normally off the kernel's tail; the end check stays as a fallback).
``proto="slot"``: a flag-free one-shot protocol on its own workspace. CTA ``b`` (fixed geometry: every call of a
    workspace uses the same ``threads`` per CTA) owns, per region, the slot block [b][rank][threads] of 16-byte
    vectors. A per-CTA participation count ``n_b`` (int32 at ``ctl + 128 * b``, read once per call by thread 0,
    written back + 1) selects region ``n_b % 3``; the CTA multicasts its vector into that region's slot of every
    rank, clears its own copy of the previous participation's region (``(n_b + 2) % 3``) and polls the current
    one (Lamport: -0.0 pairs = not written). No buffer flags, no arrival counter, no end-of-kernel rotation.
    Safety: a peer can be at most one participation of CTA ``b`` ahead (it needs this rank's data of the
    participation in between), so it writes regions ``n_b`` or ``n_b + 1`` only, never the one being cleared.

Debug: ``stamps`` stores %globaltimer at 8 points per CTA into ``dbg[b * 16 + i]`` (entry, after the dependency
wait, state + inputs ready, multicast issued, clear issued, poll done, outputs stored, end) and %clock64 into
``dbg[b * 16 + 8 + i]`` (entry, poll done, sums + residual out stored, warp sums, CTA sum, cluster sum, outputs, end).
v2 options: ``lean`` (CTA sum without the zero butterfly steps; same result), ``xchg="mbar"`` (the norm's cross-CTA
sums by st.async + an mbarrier instead of barrier.cluster), ``push="uc"`` (slot: unicast stores into every rank).
Before ``griddepcontrol.wait`` only static data is read (norm weight, the host-written peer table).
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass import Float32, Int32, Int64, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from .q35_allreduce_norm import (_NEG_ZERO, _PDL_TEST_ENTRY_TRIGGER, VEC, _add, _bf16r, _cluster_sync, _div_rn, _e4m3x2, _fma, _hi,
                                 _ld_acquire_s32, _ld_s32, _ld_s64, _ld_u16, _ld_v4, _ld_volatile_v4, _lo, _max0, _mul,
                                 _pack_bf16, _red_release_add_s32, _rsqrt, _sanitize, _sel_ge0, _shfl_bfly,
                                 _sigmoid_triton, _sink4, _smem_addr, _st_dsmem_f32, _st_s32, _st_shared_f32, _st_v2,
                                 _st_v4, _ld_shared_f32)
from .utils import griddepcontrol_launch_dependents, griddepcontrol_wait

SLOT_STATE_STRIDE = 128  # bytes between per-CTA state words (one L2 line each)


@dsl_user_op
def _globaltimer(*, loc=None, ip=None) -> Int64:
    return Int64(
        llvm.inline_asm(T.i64(), [], "mov.u64 $0, %globaltimer;", "=l", has_side_effects=True, is_align_stack=False,
                        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _st_s64(addr: Int64, value: Int64, *, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [Int64(addr).ir_value(loc=loc, ip=ip), Int64(value).ir_value(loc=loc, ip=ip)],
                    "st.global.b64 [$0], $1;", "l,l,~{memory}", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def _ld_cg_s32(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(T.i32(), [Int64(addr).ir_value(loc=loc, ip=ip)], "ld.global.cg.b32 $0, [$1];",
                        "=r,l,~{memory}", has_side_effects=True, is_align_stack=False,
                        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _clock64(*, loc=None, ip=None) -> Int64:
    return Int64(
        llvm.inline_asm(T.i64(), [], "mov.u64 $0, %clock64;", "=l", has_side_effects=True, is_align_stack=False,
                        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _mbar_init(addr: Int32, count: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [Int32(addr).ir_value(loc=loc, ip=ip), Int32(count).ir_value(loc=loc, ip=ip)],
                    "mbarrier.init.shared::cta.b64 [$0], $1;\n\tfence.mbarrier_init.release.cluster;",
                    "r,r,~{memory}", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def _mbar_expect_tx(addr: Int32, nbytes: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [Int32(addr).ir_value(loc=loc, ip=ip), Int32(nbytes).ir_value(loc=loc, ip=ip)],
                    "mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1;", "r,r,~{memory}",
                    has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def _mbar_try_wait(addr: Int32, parity: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(T.i32(), [Int32(addr).ir_value(loc=loc, ip=ip), Int32(parity).ir_value(loc=loc, ip=ip)],
                        "{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2;\n\t"
                        "selp.u32 $0, 1, 0, p;\n\t}", "=r,r,r,~{memory}", has_side_effects=True,
                        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _st_async_f32(addr: Int32, mbar: Int32, cta_rank: Int32, value: Float32, *, loc=None, ip=None) -> None:
    """st.async of one f32 into cluster CTA ``cta_rank``'s smem at this CTA's address ``addr``, completing 4 bytes
    of the transaction on that CTA's mbarrier at ``mbar``."""
    llvm.inline_asm(None, [Int32(addr).ir_value(loc=loc, ip=ip), Int32(mbar).ir_value(loc=loc, ip=ip),
                           Int32(cta_rank).ir_value(loc=loc, ip=ip), Float32(value).ir_value(loc=loc, ip=ip)],
                    "{\n\t.reg .b32 ra, rb, v;\n\tmapa.shared::cluster.u32 ra, $0, $2;\n\t"
                    "mapa.shared::cluster.u32 rb, $1, $2;\n\tmov.b32 v, $3;\n\t"
                    "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [ra], v, [rb];\n\t}",
                    "r,r,r,f,~{memory}", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


def _tree_sum_smem(red_addr, n: int):
    """The CTA butterfly's result over n warp sums at smem ``red_addr`` (plain Python at trace time): the
    butterfly on [w_0 .. w_{n-1}, 0, ...] first adds zeros (exact), then pairs lane i with i ^ s for s = n/2 .. 1."""
    vals = [_ld_shared_f32(red_addr + Int32(4 * i)) for i in range(n)]
    s = n // 2
    while s >= 1:
        vals = [_add(vals[i], vals[i ^ s]) for i in range(n)]
        s //= 2
    return vals[0]


def _load_peers(peers, n: int):
    """Every rank's slot workspace base from the host-written table (plain Python at trace time)."""
    return [_ld_s64(peers + Int64(8 * r)) for r in range(n)]


class Q35Ar2Kernel:
    """See the module docstring. Tensors as ``Q35AllReduceNormKernel``; addresses are Int64 arguments:
    ``own`` = this rank's unicast base, ``mcast`` = multicast base, ``ctl`` = buffer_flags (``flags``) or the
    per-CTA state array (``slot``), ``dbg`` = int64 [grid * 8] (``stamps``)."""

    def __init__(self, *, hidden: int, norm_eps: float, nranks: int, ctas: int, use_pdl: bool, se_fold: bool,
                 norm_out: bool, quant: bool, moe_topk: int = 0, fc2_row: int = 0, proto: str = "slot",
                 rotate: str = "end", stamps: bool = False, region_bytes: int = 0,
                 lean: bool = True, xchg: str = "cluster", push: str = "mc", two_shot: bool = False):
        if hidden % (VEC * ctas) or (hidden // VEC // ctas) % 32 or hidden // VEC // ctas > 1024:
            raise ValueError(f"unsupported hidden size {hidden} for {ctas} CTAs per token")
        if nranks not in (2, 4, 8, 16) or ctas not in (1, 2, 4, 8):
            raise ValueError("unsupported group / cluster size")
        if proto not in ("flags", "slot") or rotate not in ("end", "poll"):
            raise ValueError("bad protocol options")
        if proto == "slot" and region_bytes <= 0:
            raise ValueError("slot protocol needs region_bytes")
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
        self.proto = proto
        self.rotate = rotate
        self.stamps = stamps
        self.region_bytes = region_bytes
        # lean: the CTA-level sum of the warp sums without the butterfly's zero steps (same additions, same order)
        self.lean = lean
        if xchg not in ("cluster", "mbar") or push not in ("mc", "uc") or (push == "uc" and proto != "slot"):
            raise ValueError("bad exchange / push options")
        # xchg: the norm's cross-CTA sum by barrier.cluster ("cluster") or st.async + mbarrier ("mbar")
        self.xchg = xchg if ctas > 1 else "cluster"
        # push: one multicast store ("mc") or a unicast store into every rank's slot ("uc", slot protocol)
        self.push = push
        # two_shot (slot protocol): token t is reduced by rank t % W -- every rank unicasts its vector into the
        # owner's scatter slot, the owner sums in rank order (FP32 -> BF16) and multicasts the sum into every
        # rank's broadcast slot; the norm is the library two-shot's Lamport RMSNorm ((w * x) * rstd)
        if two_shot and proto != "slot":
            raise ValueError("two_shot is a slot-protocol mode here")
        self.two_shot = two_shot

    @cute.jit
    def __call__(self, gX: cute.Tensor, gRouted: cute.Tensor, gShared: cute.Tensor, gGate: cute.Tensor,
                 gRes: cute.Tensor, gW: cute.Tensor, gScale: cute.Tensor, gResOut: cute.Tensor, gOut: cute.Tensor,
                 gQuant: cute.Tensor, own: Int64, mcast: Int64, ctl: Int64, fc2_ptr: Int64, topw_ptr: Int64,
                 perm_ptr: Int64, dbg: Int64, peers: Int64, rank: Int32, stream) -> None:
        m = cute.size(gRes, mode=[0])
        self.kernel(gX, gRouted, gShared, gGate, gRes, gW, gScale, gResOut, gOut, gQuant, own, mcast, ctl, fc2_ptr,
                    topw_ptr, perm_ptr, dbg, peers, rank).launch(
            grid=[m * self.ctas, 1, 1],
            block=[self.threads, 1, 1],
            cluster=[self.ctas, 1, 1],
            smem=512,
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(self, gX: cute.Tensor, gRouted: cute.Tensor, gShared: cute.Tensor, gGate: cute.Tensor,
               gRes: cute.Tensor, gW: cute.Tensor, gScale: cute.Tensor, gResOut: cute.Tensor, gOut: cute.Tensor,
               gQuant: cute.Tensor, own: Int64, mcast: Int64, ctl: Int64, fc2_ptr: Int64, topw_ptr: Int64,
               perm_ptr: Int64, dbg: Int64, peers: Int64, rank: Int32) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        ctas: cutlass.Constexpr = self.ctas
        nranks: cutlass.Constexpr = self.nranks
        vecs: cutlass.Constexpr = self.hidden // VEC
        num_warps: cutlass.Constexpr = self.num_warps
        threads: cutlass.Constexpr = self.threads
        tok = bid // ctas
        part = bid % ctas
        m = grid // ctas
        vi = part * threads + tidx
        warp = tidx // 32
        lane = tidx % 32
        row_off = (Int64(tok) * Int64(vecs) + Int64(vi)) * Int64(16)
        dbase = dbg + Int64(bid) * Int64(128)
        cbase = dbase + Int64(64)  # clock64 stamps: entry, poll, sums, warp, cta, cluster, out, end

        smem = cutlass.utils.SmemAllocator()
        red = smem.allocate_tensor(Float32, cute.make_layout((num_warps, )), byte_alignment=16)
        csum = smem.allocate_tensor(Float32, cute.make_layout((ctas, )), byte_alignment=16)
        fl = smem.allocate_tensor(Int32, cute.make_layout((8, )), byte_alignment=16)
        red_addr = _smem_addr(red.iterator)
        csum_addr = _smem_addr(csum.iterator)

        if const_expr(self.stamps):
            if tidx == 0:
                _st_s64(cbase, _clock64())
                _st_s64(dbase, _globaltimer())
        mbar_addr = Int32(0)
        if const_expr(self.xchg == "mbar"):
            # one transaction barrier per CTA for the norm's cross-CTA sums; initialized cluster-wide before any
            # CTA can send (smem / cluster state only: allowed ahead of the dependency wait)
            mb = smem.allocate_tensor(Int64, cute.make_layout((1, )), byte_alignment=8)
            mbar_addr = _smem_addr(mb.iterator)
            if tidx == 0:
                _mbar_init(mbar_addr, Int32(1))
            _cluster_sync()
            if tidx == 0:
                _mbar_expect_tx(mbar_addr, Int32(4 * ctas))
        pb = []
        if const_expr(self.push == "uc"):
            pb = _load_peers(peers, nranks)  # static table: allowed ahead of the dependency wait
        owner = tok % Int32(nranks)
        owner_base = Int64(0)
        if const_expr(self.two_shot):
            owner_base = _ld_s64(peers + Int64(owner) * Int64(8))  # static table

        # ---- ahead of the dependency wait: the norm weight (a parameter) ----
        w0, w1, w2, w3 = _ld_v4(Int64(gW.iterator.toint()) + Int64(vi) * Int64(16))
        topk: cutlass.Constexpr = self.moe_topk
        pk = cute.make_rmem_tensor((max(topk, 1), ), Int32)
        wk = cute.make_rmem_tensor((max(topk, 1), ), Int32)

        if const_expr(self.use_pdl):
            griddepcontrol_wait()
            if const_expr(_PDL_TEST_ENTRY_TRIGGER):  # TEST HOOK ONLY (WAR probe positive control)
                griddepcontrol_launch_dependents()
        if const_expr(self.stamps):
            if tidx == 0:
                _st_s64(dbase + Int64(8), _globaltimer())

        # ---- this rank's inputs (independent of the protocol state) ----
        r0, r1, r2, r3 = _ld_v4(Int64(gRes.iterator.toint()) + row_off)
        if const_expr(self.se_fold):
            if const_expr(topk > 0):
                # routing outputs of this step: read only after the dependency wait (RAW)
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

        # ---- protocol state, read by one thread per CTA and shared through smem ----
        if const_expr(self.proto == "flags"):
            if tidx == 0:
                lf0, lf1, lf2, lf3 = _ld_volatile_v4(ctl)
                lf4, lf5, lf6, lf7 = _ld_volatile_v4(ctl + Int64(16))
                fl[0] = lf0
                fl[1] = lf1
                fl[2] = lf2
                fl[3] = lf3
                fl[4] = lf4
                fl[5] = lf5
                fl[6] = lf6
                fl[7] = lf7
        else:
            if tidx == 0:
                fl[0] = _ld_cg_s32(ctl + Int64(bid) * Int64(SLOT_STATE_STRIDE))
        cute.arch.sync_threads()

        # ---- the shared-expert combine (and the MoE finalize) of the input ----
        if const_expr(self.se_fold):
            if const_expr(topk > 0):
                ev = cute.make_rmem_tensor((topk * 4, ), Int32)
                for k in cutlass.range_constexpr(topk):
                    e0, e1, e2, e3 = _ld_v4(fc2_ptr + (Int64(_max0(pk[k])) * Int64(self.fc2_row) +
                                                       Int64(vi) * Int64(VEC)) * Int64(2))
                    ev[4 * k + 0] = e0
                    ev[4 * k + 1] = e1
                    ev[4 * k + 2] = e2
                    ev[4 * k + 3] = e3
                acc = cute.make_rmem_tensor((VEC, ), Float32)
                for e in cutlass.range_constexpr(VEC):
                    acc[e] = Float32(0.0)
                for k in cutlass.range_constexpr(topk):
                    wf = _lo(wk[k])
                    for i in cutlass.range_constexpr(4):
                        acc[2 * i] = _sel_ge0(pk[k], _fma(wf, _lo(ev[4 * k + i]), acc[2 * i]), acc[2 * i])
                        acc[2 * i + 1] = _sel_ge0(pk[k], _fma(wf, _hi(ev[4 * k + i]), acc[2 * i + 1]),
                                                  acc[2 * i + 1])
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
        if const_expr(self.stamps):
            if tidx == 0:
                _st_s64(dbase + Int64(16), _globaltimer())

        tot = Int64(m) * Int64(vecs)
        idx = Int64(tok) * Int64(vecs) + Int64(vi)
        neg0 = Int32(_NEG_ZERO)
        cur = Int32(0)
        dirty = Int32(0)
        buf_bytes = Int64(0)
        dirty_stages = Int32(0)
        sb0 = Int32(0)
        sb1 = Int32(0)
        sb2 = Int32(0)
        sb3 = Int32(0)
        slot_cnt = Int32(0)
        data_off = Int64(0)
        poll_base = Int64(0)
        if const_expr(self.proto == "flags"):
            cur = fl[0]
            dirty = fl[1]
            buf_bytes = Int64(fl[2])
            dirty_stages = fl[3]
            sb0 = fl[4]
            sb1 = fl[5]
            sb2 = fl[6]
            sb3 = fl[7]
            if tidx == 0:
                _red_release_add_s32(ctl + Int64(32), Int32(1))
            data_off = Int64(cur) * buf_bytes
            # one multicast store reaches every rank's current buffer ([rank][token][H])
            _st_v4(mcast + data_off + (Int64(rank) * tot + idx) * Int64(16), _sanitize(x0), _sanitize(x1),
                   _sanitize(x2), _sanitize(x3))
            poll_base = own + data_off + idx * Int64(16)
        else:
            slot_cnt = fl[0]
            q = slot_cnt % Int32(3)
            blk = Int64(bid) * Int64((nranks + 1) * threads * 16)  # [W scatter / one-shot slots][1 broadcast slot]
            data_off = Int64(q) * Int64(self.region_bytes) + blk
            if const_expr(self.two_shot):
                _st_v4(owner_base + data_off + (Int64(rank) * Int64(threads) + Int64(tidx)) * Int64(16),
                       _sanitize(x0), _sanitize(x1), _sanitize(x2), _sanitize(x3))
            elif const_expr(self.push == "uc"):
                sx0, sx1, sx2, sx3 = _sanitize(x0), _sanitize(x1), _sanitize(x2), _sanitize(x3)
                for r in cutlass.range_constexpr(nranks):
                    _st_v4(pb[r] + data_off + (Int64(rank) * Int64(threads) + Int64(tidx)) * Int64(16), sx0, sx1,
                           sx2, sx3)
            else:
                _st_v4(mcast + data_off + (Int64(rank) * Int64(threads) + Int64(tidx)) * Int64(16), _sanitize(x0),
                       _sanitize(x1), _sanitize(x2), _sanitize(x3))
            poll_base = own + data_off + Int64(tidx) * Int64(16)
        if const_expr(self.stamps):
            if tidx == 0:
                _st_s64(dbase + Int64(24), _globaltimer())

        # ---- PDL: release the dependents once every recyclable input is consumed (PDL rule) ----
        if const_expr(self.use_pdl and not _PDL_TEST_ENTRY_TRIGGER):
            _sink4(red_addr, r0, r1, r2, r3)
            cute.arch.sync_threads()
            griddepcontrol_launch_dependents()

        # ---- clear this thread's share of the previous buffer ----
        if const_expr(self.proto == "flags"):
            for s_i in cutlass.range_constexpr(4):
                if Int32(s_i) < dirty_stages:
                    stage_bytes = buf_bytes // Int64(dirty_stages)
                    stage = own + (Int64(dirty) * Int64(dirty_stages) + Int64(s_i)) * stage_bytes
                    sb = sb0 if s_i == 0 else (sb1 if s_i == 1 else (sb2 if s_i == 2 else sb3))
                    clear_vecs = (Int64(sb) + Int64(15)) // Int64(16)
                    c = idx
                    while c < clear_vecs:
                        _st_v4(stage + c * Int64(16), neg0, neg0, neg0, neg0)
                        c = c + tot
        else:
            if tidx == 0:
                _st_s32(ctl + Int64(bid) * Int64(SLOT_STATE_STRIDE), slot_cnt + Int32(1))
            pq = (slot_cnt + Int32(2)) % Int32(3)
            pbase = own + Int64(pq) * Int64(self.region_bytes) + Int64(bid) * Int64((nranks + 1) * threads * 16) + \
                Int64(tidx) * Int64(16)
            for r in cutlass.range_constexpr(nranks + 1):  # one-shot and two-shot slots of that participation
                _st_v4(pbase + Int64(r * threads * 16), neg0, neg0, neg0, neg0)
        if const_expr(self.stamps):
            if tidx == 0:
                _st_s64(dbase + Int64(32), _globaltimer())

        # ---- wait for this vector of every rank ----
        rstride = tot * Int64(16)
        if const_expr(self.proto == "slot"):
            rstride = Int64(threads * 16)
        got = cute.make_rmem_tensor((nranks * 4, ), Int32)
        rotated = Int32(0)
        pending = Int32(1)
        if const_expr(self.two_shot):
            if owner == rank:
                # owner: every rank's vector, FP32 sum in rank order -> BF16, multicast into the broadcast slots
                pend_o = Int32(1)
                while pend_o != 0:
                    pend_o = Int32(0)
                    for r in cutlass.range_constexpr(nranks):
                        a1, b1, c1, d1 = _ld_volatile_v4(poll_base + Int64(r) * rstride)
                        got[r * 4 + 0] = a1
                        got[r * 4 + 1] = b1
                        got[r * 4 + 2] = c1
                        got[r * 4 + 3] = d1
                        if a1 == neg0:
                            pend_o = Int32(1)
                        if b1 == neg0:
                            pend_o = Int32(1)
                        if c1 == neg0:
                            pend_o = Int32(1)
                        if d1 == neg0:
                            pend_o = Int32(1)
                rsum = cute.make_rmem_tensor((4, ), Int32)
                for i in cutlass.range_constexpr(4):
                    slo = Float32(0.0)
                    shi = Float32(0.0)
                    for r in cutlass.range_constexpr(nranks):
                        slo = _add(slo, _lo(got[r * 4 + i]))
                        shi = _add(shi, _hi(got[r * 4 + i]))
                    rsum[i] = _sanitize(_pack_bf16(slo, shi))
                _st_v4(mcast + data_off + Int64(nranks * threads * 16) + Int64(tidx) * Int64(16), rsum[0], rsum[1],
                       rsum[2], rsum[3])
            # every rank: the reduced vector in its broadcast slot
            while pending != 0:
                pending = Int32(0)
                a2, b2, c2, d2 = _ld_volatile_v4(poll_base + Int64(nranks) * rstride)
                got[0] = a2
                got[1] = b2
                got[2] = c2
                got[3] = d2
                if a2 == neg0:
                    pending = Int32(1)
                if b2 == neg0:
                    pending = Int32(1)
                if c2 == neg0:
                    pending = Int32(1)
                if d2 == neg0:
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
            if const_expr(self.proto == "flags" and self.rotate == "poll"):
                if bid == 0:
                    if tidx == 0:
                        if rotated == 0:
                            arrived0 = _ld_acquire_s32(ctl + Int64(32))
                            if arrived0 == grid:
                                _st_s32(ctl, (cur + Int32(1)) % Int32(3))
                                _st_s32(ctl + Int64(4), cur)
                                _st_s32(ctl + Int64(12), Int32(1))
                                _st_s32(ctl + Int64(16), Int32(m) * Int32(self.hidden * nranks * 2))
                                _st_s32(ctl + Int64(20), Int32(0))
                                _st_s32(ctl + Int64(24), Int32(0))
                                _st_s32(ctl + Int64(28), Int32(0))
                                _st_s32(ctl + Int64(32), Int32(0))
                                rotated = Int32(1)
        if const_expr(self.stamps):
            if tidx == 0:
                _st_s64(dbase + Int64(40), _globaltimer())
                _st_s64(cbase + Int64(8), _clock64())

        # ---- FP32 sum in rank order -> BF16; + residual (BF16) ----
        rw = (r0, r1, r2, r3)
        ww = (w0, w1, w2, w3)
        res = cute.make_rmem_tensor((VEC, ), Float32)
        if const_expr(self.two_shot):
            for i in cutlass.range_constexpr(4):
                res[2 * i] = _bf16r(_add(_lo(got[i]), _lo(rw[i])))
                res[2 * i + 1] = _bf16r(_add(_hi(got[i]), _hi(rw[i])))
        else:
            for i in cutlass.range_constexpr(4):
                lo = Float32(0.0)
                hi = Float32(0.0)
                for r in cutlass.range_constexpr(nranks):
                    lo = _add(lo, _lo(got[r * 4 + i]))
                    hi = _add(hi, _hi(got[r * 4 + i]))
                res[2 * i] = _bf16r(_add(_bf16r(lo), _lo(rw[i])))
                res[2 * i + 1] = _bf16r(_add(_bf16r(hi), _hi(rw[i])))
        out_off = Int64(gResOut.iterator.toint()) + row_off
        _st_v4(out_off, _pack_bf16(res[0], res[1]), _pack_bf16(res[2], res[3]), _pack_bf16(res[4], res[5]),
               _pack_bf16(res[6], res[7]))
        if const_expr(self.stamps):
            if tidx == 0:
                _st_s64(cbase + Int64(16), _clock64())

        # ---- RMSNorm: BF16 squares, per-thread sum in element order, warp / CTA / cluster sums ----
        tsum = Float32(0.0)
        for e in cutlass.range_constexpr(VEC):
            tsum = _add(tsum, _bf16r(_mul(res[e], res[e])))
        for k in cutlass.range_constexpr(5):
            tsum = _add(tsum, _shfl_bfly(tsum, Int32(16 >> k)))
        if lane == 0:
            _st_shared_f32(red_addr + Int32(4) * warp, tsum)
        if const_expr(self.stamps):
            if tidx == 0:
                _st_s64(cbase + Int64(24), _clock64())
        cute.arch.sync_threads()
        bsum = Float32(0.0)
        if const_expr(self.lean):
            # the same additions in the same order as the butterfly below, without its shuffles
            bsum = _tree_sum_smem(red_addr, num_warps)
        else:
            if lane < num_warps:
                bsum = _ld_shared_f32(red_addr + Int32(4) * lane)
            for k in cutlass.range_constexpr(5):
                bsum = _add(bsum, _shfl_bfly(bsum, Int32(16 >> k)))
        if const_expr(self.stamps):
            if tidx == 0:
                _st_s64(cbase + Int64(32), _clock64())
        total = bsum
        if const_expr(ctas > 1):
            if const_expr(self.xchg == "mbar"):
                if tidx < ctas:
                    _st_async_f32(csum_addr + Int32(4) * part, mbar_addr, Int32(tidx), bsum)
                ready = _mbar_try_wait(mbar_addr, Int32(0))
                while ready == 0:
                    ready = _mbar_try_wait(mbar_addr, Int32(0))
            else:
                if tidx < ctas:
                    _st_dsmem_f32(csum_addr + Int32(4) * part, Int32(tidx), bsum)
                _cluster_sync()
            total = Float32(0.0)
            for p in cutlass.range_constexpr(ctas):
                total = _add(total, _ld_shared_f32(csum_addr + Int32(4 * p)))
        if const_expr(self.stamps):
            if tidx == 0:
                _st_s64(cbase + Int64(40), _clock64())
        rstd = _rsqrt(_add(_div_rn(total, Float32(float(self.hidden))), Float32(self.norm_eps)))
        y = cute.make_rmem_tensor((VEC, ), Float32)
        for i in cutlass.range_constexpr(4):
            if const_expr(self.two_shot):  # rmsNormLamport: (w * x) * rstd
                y[2 * i] = _bf16r(_mul(_mul(_lo(ww[i]), res[2 * i]), rstd))
                y[2 * i + 1] = _bf16r(_mul(_mul(_hi(ww[i]), res[2 * i + 1]), rstd))
            else:  # one-shot: (x * rstd) * w
                y[2 * i] = _bf16r(_mul(_mul(res[2 * i], rstd), _lo(ww[i])))
                y[2 * i + 1] = _bf16r(_mul(_mul(res[2 * i + 1], rstd), _hi(ww[i])))
        if const_expr(self.norm_out):
            _st_v4(Int64(gOut.iterator.toint()) + row_off, _pack_bf16(y[0], y[1]), _pack_bf16(y[2], y[3]),
                   _pack_bf16(y[4], y[5]), _pack_bf16(y[6], y[7]))
        if const_expr(self.quant):
            inv = _div_rn(Float32(1.0), gScale[0])
            q01 = _e4m3x2(_mul(y[0], inv), _mul(y[1], inv))
            q23 = _e4m3x2(_mul(y[2], inv), _mul(y[3], inv))
            q45 = _e4m3x2(_mul(y[4], inv), _mul(y[5], inv))
            q67 = _e4m3x2(_mul(y[6], inv), _mul(y[7], inv))
            _st_v2(Int64(gQuant.iterator.toint()) + (Int64(tok) * Int64(vecs) + Int64(vi)) * Int64(8),
                   q01 | (q23 << 16), q45 | (q67 << 16))
        if const_expr(self.stamps):
            if tidx == 0:
                _st_s64(dbase + Int64(48), _globaltimer())
                _st_s64(cbase + Int64(48), _clock64())

        # ---- flags protocol: hand the triple buffer to the next call once every CTA has read the flags ----
        if const_expr(self.proto == "flags"):
            if bid == 0:
                if tidx == 0:
                    if rotated == 0:
                        arrived = _ld_acquire_s32(ctl + Int64(32))
                        while arrived != grid:
                            arrived = _ld_acquire_s32(ctl + Int64(32))
                        _st_s32(ctl, (cur + Int32(1)) % Int32(3))
                        _st_s32(ctl + Int64(4), cur)
                        _st_s32(ctl + Int64(12), Int32(1))
                        _st_s32(ctl + Int64(16), Int32(m) * Int32(self.hidden * nranks * 2))
                        _st_s32(ctl + Int64(20), Int32(0))
                        _st_s32(ctl + Int64(24), Int32(0))
                        _st_s32(ctl + Int64(28), Int32(0))
                        _st_s32(ctl + Int64(32), Int32(0))
        if const_expr(self.stamps):
            if tidx == 0:
                _st_s64(dbase + Int64(56), _globaltimer())
                _st_s64(cbase + Int64(56), _clock64())
