# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent sparse-attention tail, consuming exact indexer-selected rows as the indexer publishes them.

Launched with programmatic dependent launch right behind the indexer grid: CTAs start on SMs as indexer CTAs exit
and claim rows in completion order from the ready queue (selection_lib AQ_*). Warps 0-3 claim rows and issue Q/KV
loads (running ahead into the next row), warp 8 issues the MMAs, warps 4-7 run softmax and the epilogue. S is double
buffered in TMEM, so QK of the next tile and PV of the previous tile run while the softmax warps work.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as xcuda
import cutlass.pipeline as pipeline
from cutlass import Boolean, Float32, Int32, Int64
from cutlass.experimental import primitives as prims

from . import attention_helpers as A
from . import selection_lib as L

# mbarrier slots (all single-arrival)
NKVG = 3  # KV slot groups (one tile each): tile t uses group t % NKVG
B_Q = 0  # four Q slices                                  loader -> MMA warp
B_KV = 4  # 4 * NKVG KV slots                              loader -> MMA warp
B_SFULL = 16  # two S buffers: QK done                         MMA warp -> softmax
B_PV = 18  # PV done                                        MMA warp -> softmax
B_GFREE = 19  # NKVG KV groups, released by the PV that used them  MMA warp -> loader
B_QFREE = 22  # Q slices, released after the last QK of a row  MMA warp -> loader
B_PRDY = 23  # P stored and O rescaled                        softmax -> MMA warp
B_ROW = 24  # two row hand-off slots                         loader -> MMA warp, softmax
B_DRAIN = 26
NBAR = 27
TM_SB = 128  # S buffer t & 1 at column 128 * (t & 1); O at A.TM_O (columns 256-511, both 16-lane panels)
E4M3 = cutlass.Float8E4M3FN


@cute.jit
def _issue_pv(abar, tp, jp, tmem, kv_base, idesc_pv):
    # O += P(tp) V(tp) once the softmax warps have stored P and rescaled O; releases the tile's KV group.
    A._wait(abar + Int32(B_PRDY), tp & Int32(1))
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    if prims.elect_sync():
        gp = tp - tp // Int32(NKVG) * Int32(NKVG)
        g4 = gp * Int32(4)
        s_addr = tmem + (tp & Int32(1)) * Int32(TM_SB)
        for v in cutlass.range_constexpr(A.NSLICE):
            vdesc = prims.Tcgen05SmemDesc.build(
                kv_base + (g4 + Int32(v)) * Int32(A.SLOT_BYTES),
                leading_byte_offset=0,
                stride_byte_offset=1024,
                layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
            )
            d_addr = tmem + Int32(A.TM_O + (v & 1) * 128 + (v >> 1) * A.PANEL1)
            a_addr = s_addr + Int32(A.TILE_N // 4 + (v >> 1) * A.PANEL1)
            for kb in cutlass.range_constexpr(A.TILE_N // 32):
                en = Boolean(True)
                if cutlass.const_expr(kb == 0):
                    en = Boolean(jp > Int32(0))
                prims.tcgen05_mma(
                    prims.Tcgen05MMAKind.F8F6F4,
                    prims.CTAGroup.CTA_1,
                    prims.make_tmem_ptr(d_addr, Float32),
                    prims.make_tmem_ptr(a_addr + Int32(8 * kb), E4M3),
                    vdesc.advance_start_address(kb * 32 * A.SLICE),
                    idesc_pv,
                    en,
                )
        prims.tcgen05_commit(abar + Int32(B_PV), group=prims.CTAGroup.CTA_1)
        prims.tcgen05_commit(abar + Int32(B_GFREE) + gp, group=prims.CTAGroup.CTA_1)


@cute.jit
def _mma_warp(abar, sRow, q_base, kv_base, tmem):
    # Tensor-pipe order per row: QK0, QK1, PV0, QK2, PV1, ..., QK4, PV3, PV4. QK(t) lands in S buffer t & 1 while
    # the softmax warps work on S buffer (t - 1) & 1; the pipe executes in issue order, so QK(t) never overwrites
    # P(t - 2) before PV(t - 2) has read it.
    idesc_qk = prims.Tcgen05InstrDesc.build(
        c_dtype=Float32, a_dtype=E4M3, b_dtype=E4M3, n_dim=A.TILE_N, m_dim=A.HEADS
    )
    idesc_pv = prims.Tcgen05InstrDesc.build(
        c_dtype=Float32,
        a_dtype=E4M3,
        b_dtype=E4M3,
        a_major=0,
        b_major=1,
        n_dim=A.SLICE,
        m_dim=A.HEADS,
    )
    n = Int32(0)
    t = Int32(0)
    ph = Int32(0)
    A._wait(abar + Int32(B_ROW) + (n & Int32(1)), n >> Int32(1) & Int32(1))
    b = Int32(sRow[n & Int32(1)])
    while b >= Int32(0):
        for j in cutlass.range(A.NTILES, unroll=1):
            tt = t + j
            g4 = (tt - tt // Int32(NKVG) * Int32(NKVG)) * Int32(4)
            s_addr = tmem + (tt & Int32(1)) * Int32(TM_SB)
            if j == Int32(0):
                for c in cutlass.range_constexpr(A.NSLICE):
                    A._wait(abar + Int32(B_Q + c), ph >> Int32(c) & Int32(1))
                ph = ph ^ Int32(15)
            for c in cutlass.range_constexpr(A.NSLICE):
                A._wait(
                    abar + Int32(B_KV) + g4 + Int32(c), ph >> Int32(16) + g4 + Int32(c) & Int32(1)
                )
            ph = ph ^ Int32(15) << Int32(16) + g4
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            if prims.elect_sync():
                for c in cutlass.range_constexpr(A.NSLICE):
                    qdesc = prims.Tcgen05SmemDesc.build(
                        q_base + Int32(c * A.Q_SLICE_BYTES),
                        leading_byte_offset=A.Q_SLICE_BYTES,
                        stride_byte_offset=1024,
                        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
                    )
                    kdesc = prims.Tcgen05SmemDesc.build(
                        kv_base + (g4 + Int32(c)) * Int32(A.SLOT_BYTES),
                        leading_byte_offset=A.SLOT_BYTES,
                        stride_byte_offset=1024,
                        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
                    )
                    for kb in cutlass.range_constexpr(A.SLICE // 32):
                        en = Boolean(True)
                        if cutlass.const_expr(kb == 0):
                            en = Boolean(c > 0)
                        prims.tcgen05_mma(
                            prims.Tcgen05MMAKind.F8F6F4,
                            prims.CTAGroup.CTA_1,
                            prims.make_tmem_ptr(s_addr, Float32),
                            qdesc.advance_start_address(kb * 32),
                            kdesc.advance_start_address(kb * 32),
                            idesc_qk,
                            en,
                        )
                prims.tcgen05_commit(
                    abar + Int32(B_SFULL) + (tt & Int32(1)), group=prims.CTAGroup.CTA_1
                )
                if j == Int32(A.NTILES - 1):
                    prims.tcgen05_commit(abar + Int32(B_QFREE), group=prims.CTAGroup.CTA_1)
            if j >= Int32(1):
                _issue_pv(abar, tt - Int32(1), j - Int32(1), tmem, kv_base, idesc_pv)
        _issue_pv(abar, t + Int32(A.NTILES - 1), Int32(A.NTILES - 1), tmem, kv_base, idesc_pv)
        t = t + Int32(A.NTILES)
        n = n + Int32(1)
        A._wait(abar + Int32(B_ROW) + (n & Int32(1)), n >> Int32(1) & Int32(1))
        b = Int32(sRow[n & Int32(1)])
    if prims.elect_sync():
        prims.tcgen05_commit(abar + Int32(B_DRAIN), group=prims.CTAGroup.CTA_1)
    A._wait(abar + Int32(B_DRAIN), Int32(0))


@cute.jit
def _softmax_row(
    rtid,
    b,
    t,
    gv0,
    gv1,
    gv2,
    gv3,
    gv4,
    tmem,
    abar,
    s_msk,
    sink_ptr,
    out_addr: Int64,
    scale_log2: Float32,
):
    # Same arithmetic as attention_helpers.att_row (online softmax with sinks, P in E4M3 scaled by 448), with S read
    # from TMEM once per tile and the PV of the previous tile running underneath.
    wl = rtid >> Int32(5)
    lane = rtid & Int32(31)
    head = wl * Int32(16) + (lane & Int32(15))
    half = lane >> Int32(4)
    lane_base = wl * Int32(32) << Int32(16)
    sink = Float32((sink_ptr + head).load())
    Ls = scale_log2
    m_run = Float32(A.NEG_FLT_MAX)
    l_run = Float32(0.0)
    for j in cutlass.range(A.NTILES, unroll=1):
        tt = t + j
        sb = tt & Int32(1)
        gsel = A._pick(j, gv0, gv1, gv2, gv3, gv4)
        vm = Int32(cute.arch.vote_ballot_sync(gsel >= Int32(0)))
        if lane == Int32(0):
            s_msk[sb * Int32(4) + wl] = vm
        cute.arch.barrier(barrier_id=4, number_of_threads=128)
        mk = Int32(s_msk[sb * Int32(4) + half * Int32(2)])
        mk2 = Int32(s_msk[sb * Int32(4) + half * Int32(2) + Int32(1)])
        A._wait(abar + Int32(B_SFULL) + sb, tt >> Int32(1) & Int32(1))
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        tsa = tmem + sb * Int32(TM_SB) + lane_base
        s0 = prims.tcgen05_ld(
            prims.Tcgen05LdStShape.SHAPE_16X32BX2,
            prims.make_tmem_ptr(tsa, Float32),
            num=32,
            offset=Int64(64),
        )
        s1 = prims.tcgen05_ld(
            prims.Tcgen05LdStShape.SHAPE_16X32BX2,
            prims.make_tmem_ptr(tsa + Int32(32), Float32),
            num=32,
            offset=Int64(64),
        )
        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
        xs = []
        for ch in cutlass.range_constexpr(2):
            for i in cutlass.range_constexpr(32):
                x = Float32(s0[i])
                mword = mk
                if cutlass.const_expr(ch == 1):
                    x = Float32(s1[i])
                    mword = mk2
                if mword >> Int32(i) & Int32(1) == Int32(0):
                    x = Float32(-float("inf"))
                xs.append(x)
        old_max = m_run
        mx0 = old_max
        mx1 = Float32(A.NEG_FLT_MAX)
        mx2 = Float32(A.NEG_FLT_MAX)
        mx3 = Float32(A.NEG_FLT_MAX)
        for i in cutlass.range_constexpr(0, 64, 4):
            mx0 = cute.math.max(mx0, xs[i], ftz=True)
            mx1 = cute.math.max(mx1, xs[i + 1], ftz=True)
            mx2 = cute.math.max(mx2, xs[i + 2], ftz=True)
            mx3 = cute.math.max(mx3, xs[i + 3], ftz=True)
        lmax = cute.math.max(
            cute.math.max(mx0, mx1, ftz=True), cute.math.max(mx2, mx3, ftz=True), ftz=True
        )
        lmax = cute.math.max(lmax, A._shfl_xor_f32(lmax, 16), ftz=True)
        new_max = lmax
        safe_max = new_max
        if safe_max == Float32(A.NEG_FLT_MAX):
            safe_max = Float32(0.0)
        neg_scaled = -Ls * safe_max
        bias = neg_scaled + Float32(A.LOG2_448)
        sm0 = Float32(0.0)
        sm1 = Float32(0.0)
        sm2 = Float32(0.0)
        sm3 = Float32(0.0)
        regs = []
        for pk in cutlass.range_constexpr(16):
            x01 = cute.arch.fma_packed_f32x2(
                (xs[4 * pk], xs[4 * pk + 1]), (Ls, Ls), (bias, bias), ftz=False, rnd="rn"
            )
            x23 = cute.arch.fma_packed_f32x2(
                (xs[4 * pk + 2], xs[4 * pk + 3]), (Ls, Ls), (bias, bias), ftz=False, rnd="rn"
            )
            p0 = cute.math.exp2(x01[0], fastmath=True)
            p1 = cute.math.exp2(x01[1], fastmath=True)
            p2 = cute.math.exp2(x23[0], fastmath=True)
            p3 = cute.math.exp2(x23[1], fastmath=True)
            sm0, sm1 = cute.arch.add_packed_f32x2((sm0, sm1), (p0, p1), ftz=False, rnd="rn")
            sm2, sm3 = cute.arch.add_packed_f32x2((sm2, sm3), (p2, p3), ftz=False, rnd="rn")
            regs.append(A._pack4_e4m3(p0, p1, p2, p3))
        local_sum = sm0 + sm1 + (sm2 + sm3)
        p_addr = tsa + Int32(A.TILE_N // 4)
        A._tmem_st_p8(p_addr, regs[0:8])
        A._tmem_st_p8(p_addr + Int32(8), regs[8:16])
        A._tmem_st_p8(p_addr + Int32(A.PANEL1), regs[0:8])
        A._tmem_st_p8(p_addr + Int32(A.PANEL1 + 8), regs[8:16])
        exp_scale = cute.math.exp2(Ls * (old_max - new_max), fastmath=True)
        l_run = exp_scale * l_run + local_sum
        m_run = new_max
        if j > Int32(0):
            # O is still being accumulated by PV(tt - 1): wait before rescaling it.
            A._wait(abar + Int32(B_PV), tt - Int32(1) & Int32(1))
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            ident = old_max == new_max
            scale = Float32(1.0)
            if not ident:
                scale = exp_scale
            skip_c = prims.vote_sync(4294967295, ident, prims.VoteSync.ALL)
            if not skip_c:
                A._rescale_o(tmem, lane_base, scale)
        prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        cute.arch.barrier(barrier_id=4, number_of_threads=128)
        if rtid == Int32(0):
            prims.mbarrier_arrive(abar + Int32(B_PRDY))
    tl = t + Int32(A.NTILES - 1)
    A._wait(abar + Int32(B_PV), tl & Int32(1))
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    l_tot = l_run + A._shfl_xor_f32(l_run, 16)
    norm = A._final_norm(m_run, l_tot, sink, scale_log2)
    out_base = (
        out_addr
        + Int64(b) * Int64(A.HEADS * A.LATENT * 2)
        + Int64((head * Int32(A.LATENT) + half * Int32(64)) * Int32(2))
    )
    for v in cutlass.range_constexpr(A.NSLICE):
        for hc in cutlass.range_constexpr(2):
            o_addr = tmem + lane_base + Int32(A.TM_O + v % 2 * 128 + v // 2 * A.PANEL1 + 32 * hc)
            o = prims.tcgen05_ld(
                prims.Tcgen05LdStShape.SHAPE_16X32BX2,
                prims.make_tmem_ptr(o_addr, Float32),
                num=32,
                offset=Int64(64),
            )
            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
            ob = []
            for pi in cutlass.range_constexpr(16):
                fp = cute.arch.mul_packed_f32x2(
                    (norm, norm), (Float32(o[2 * pi]), Float32(o[2 * pi + 1])), ftz=False, rnd="rn"
                )
                ob.append(A._pack_bf16x2(fp[0], fp[1]))
            for k in cutlass.range_constexpr(4):
                dst = cutlass.inttoptr(
                    out_base + Int64((v * A.SLICE + hc * 32 + k * 8) * 2), mem_space=1, dtype=Int32
                )
                dst.store(
                    cutlass.Vector.from_elements(tuple((ob[4 * k + i] for i in range(4))), Int32),
                    alignment=16,
                )
    # O has been read: the next row's first PV (after its P-ready barrier) may overwrite it.
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)


@cute.jit
def _issue_kv_tile(slot_base, mbar_base, tma_swa, tma_cmp, use_swa, gidx_t, wl, lane):
    # All four 128-B slices of each gathered 512-B row are requested back to back, so the row's DRAM page is
    # opened once instead of once per slice.
    if wl == Int32(0):
        if prims.elect_sync():
            for c in cutlass.range_constexpr(A.NSLICE):
                prims.mbarrier_arrive_expect_tx(mbar_base + Int32(c), A.SLOT_BYTES)
    dst0 = slot_base + wl * Int32(32 * A.SLICE)
    for q in cutlass.range(8, unroll=1):
        r0 = A._shfl_idx_i32(gidx_t, Int32(4 * q))
        r1 = A._shfl_idx_i32(gidx_t, Int32(4 * q + 1))
        r2 = A._shfl_idx_i32(gidx_t, Int32(4 * q + 2))
        r3 = A._shfl_idx_i32(gidx_t, Int32(4 * q + 3))
        for c in cutlass.range_constexpr(A.NSLICE):
            dst = dst0 + Int32(c * A.SLOT_BYTES) + Int32(4 * q * A.SLICE)
            if use_swa:
                A._tma_gather4(
                    dst, tma_swa.get_ptr(), mbar_base + Int32(c), Int32(c * A.SLICE), r0, r1, r2, r3
                )
            else:
                A._tma_gather4(
                    dst, tma_cmp.get_ptr(), mbar_base + Int32(c), Int32(c * A.SLICE), r0, r1, r2, r3
                )


class AttentionTail:
    def __init__(self):
        self.threads = 288  # warps 0-3 loader, 4-7 softmax/epilogue, 8 MMA issue

    @cute.jit
    def __call__(
        self,
        q_att: cute.Tensor,
        swa_pool: cute.Tensor,
        c_pool: cute.Tensor,
        gidx: cute.Tensor,
        sinks: cute.Tensor,
        out: cute.Tensor,
        aq: cute.Tensor,
        softmax_scale: Float32,
        nctas: Int32,
        stream: cuda.CUstream,
    ):
        tma_q = xcuda.create_tensor_map_tiled(
            global_address=q_att.iterator.toint(),
            dtype=cutlass.Float8E4M3FN,
            global_dims=[A.LATENT, A.HEADS, q_att.shape[0]],
            global_strides=[A.LATENT // 16, A.HEADS * A.LATENT // 16],
            box_dims=[A.SLICE, A.HEADS, 1],
            swizzle=xcuda.TensorMapSwizzle.s128b,
        )
        tma_swa = xcuda.create_tensor_map_tiled(
            global_address=swa_pool.iterator.toint(),
            dtype=cutlass.Float8E4M3FN,
            global_dims=[A.LATENT, swa_pool.shape[0]],
            global_strides=[A.LATENT // 16],
            box_dims=[A.SLICE, 1],
            swizzle=xcuda.TensorMapSwizzle.s128b,
        )
        tma_cmp = xcuda.create_tensor_map_tiled(
            global_address=c_pool.iterator.toint(),
            dtype=cutlass.Float8E4M3FN,
            global_dims=[A.LATENT, c_pool.shape[0]],
            global_strides=[A.LATENT // 16],
            box_dims=[A.SLICE, 1],
            swizzle=xcuda.TensorMapSwizzle.s128b,
        )
        self.kernel(tma_q, tma_swa, tma_cmp, gidx, sinks, out, aq, softmax_scale).launch(
            grid=(nctas, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=True,
        )

    @cute.jit
    def _claim(self, tid, aq_base: Int64, nrows, sRow, sLd, abar, n):
        # Loader thread 0 takes the next completed row (-1 once all rows are taken) and hands it to both groups.
        if tid == Int32(0):
            k = L.atom_add_i32(aq_base + Int64(L.AQ_CLAIM * 4), Int32(1))
            b = Int32(-1)
            if k < nrows:
                ea = aq_base + Int64(L.AQ_ENTRY * 4) + Int64(k) * Int64(4)
                b = L.ld_acquire_i32(ea)
                while b < Int32(0):
                    L.nanosleep(Int32(256))
                    b = L.ld_acquire_i32(ea)
                L.st_relaxed_i32(ea, Int32(-1))
            sRow[n & Int32(1)] = b
            sLd[n & Int32(1)] = b
            prims.mbarrier_arrive(abar + Int32(B_ROW) + (n & Int32(1)))
        cute.arch.barrier(barrier_id=2, number_of_threads=128)
        return Int32(sLd[n & Int32(1)])

    @cute.kernel
    def kernel(
        self,
        tma_q: cutlass.GridConstant[xcuda.TensorMap],
        tma_swa: cutlass.GridConstant[xcuda.TensorMap],
        tma_cmp: cutlass.GridConstant[xcuda.TensorMap],
        gidx: cute.Tensor,
        sinks: cute.Tensor,
        out: cute.Tensor,
        aq: cute.Tensor,
        softmax_scale: Float32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        gdim, _, _ = cute.arch.grid_dim()
        smem = cutlass.utils.SmemAllocator()
        sQ = smem.allocate_tensor(
            cutlass.Float8E4M3FN, cute.make_layout(A.NSLICE * A.Q_SLICE_BYTES), byte_alignment=1024
        )
        sKV = smem.allocate_tensor(
            cutlass.Float8E4M3FN, cute.make_layout(4 * NKVG * A.SLOT_BYTES), byte_alignment=1024
        )
        sBar = smem.allocate_tensor(Int64, cute.make_layout(32), byte_alignment=16)
        sMask = smem.allocate_tensor(Int32, cute.make_layout(8), byte_alignment=16)
        sRow = smem.allocate_tensor(Int32, cute.make_layout(4), byte_alignment=16)
        sLd = smem.allocate_tensor(Int32, cute.make_layout(4), byte_alignment=16)
        sTmem = smem.allocate_tensor(Int32, cute.make_layout(1), byte_alignment=16)
        if tid < Int32(NBAR):
            prims.mbarrier_init(sBar.iterator.raw_ptr() + tid, 1)
        prims.fence_mbarrier_init()
        tmem_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads)
        tmem = cutlass.utils.TmemAllocator(
            sTmem.iterator, barrier_for_retrieve=tmem_barrier, allocator_warp_id=0, is_two_cta=False
        )
        tmem.allocate(512)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(Float32)
        tmem_addr = cute.arch.load(sTmem.iterator, Int32, sem="relaxed", scope="cta")
        q_base = sQ.iterator.raw_ptr()
        kv_base = sKV.iterator.raw_ptr()
        abar = sBar.iterator.raw_ptr()
        aq_base = aq.iterator.toint()
        gidx_base = gidx.iterator.toint()
        nrows = Int32(gidx.shape[0])
        if tid < Int32(128):
            wl = tid >> Int32(5)
            lane = tid & Int32(31)
            n = Int32(0)
            t = Int32(0)
            b = self._claim(tid, aq_base, nrows, sRow, sLd, abar, n)
            while b >= Int32(0):
                if n >= Int32(1):
                    A._wait(abar + Int32(B_QFREE), n - Int32(1) & Int32(1))
                if tid == Int32(0):
                    for c in cutlass.range_constexpr(A.NSLICE):
                        prims.mbarrier_arrive_expect_tx(abar + Int32(B_Q + c), A.Q_SLICE_BYTES)
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            q_base + Int32(c * A.Q_SLICE_BYTES),
                            tma_q.get_ptr(),
                            [Int32(c * A.SLICE), Int32(0), b],
                            abar + Int32(B_Q + c),
                        )
                grow_base = gidx_base + Int64(b) * Int64(A.NROWS * 4) + Int64(tid * Int32(4))
                for j in cutlass.range(A.NTILES, unroll=1):
                    tq = t // Int32(NKVG)
                    g = t - tq * Int32(NKVG)
                    if t >= Int32(NKVG):
                        A._wait(abar + Int32(B_GFREE) + g, tq - Int32(1) & Int32(1))
                    grow = A._grow(L.ld_cg_i32(grow_base + Int64(j * Int32(A.TILE_N * 4))))
                    _issue_kv_tile(
                        kv_base + g * Int32(4 * A.SLOT_BYTES),
                        abar + Int32(B_KV) + g * Int32(4),
                        tma_swa,
                        tma_cmp,
                        j == Int32(0),
                        grow,
                        wl,
                        lane,
                    )
                    t = t + Int32(1)
                n = n + Int32(1)
                b = self._claim(tid, aq_base, nrows, sRow, sLd, abar, n)
        elif tid < Int32(256):
            rtid = tid - Int32(128)
            scale_log2 = softmax_scale * Float32(A.LOG2_E)
            out_addr = out.iterator.toint()
            n = Int32(0)
            t = Int32(0)
            A._wait(abar + Int32(B_ROW) + (n & Int32(1)), n >> Int32(1) & Int32(1))
            b = Int32(sRow[n & Int32(1)])
            while b >= Int32(0):
                gb = gidx_base + Int64(b) * Int64(A.NROWS * 4) + Int64(rtid * Int32(4))
                gv0 = L.ld_cg_i32(gb)
                gv1 = L.ld_cg_i32(gb + Int64(A.TILE_N * 4))
                gv2 = L.ld_cg_i32(gb + Int64(2 * A.TILE_N * 4))
                gv3 = L.ld_cg_i32(gb + Int64(3 * A.TILE_N * 4))
                gv4 = L.ld_cg_i32(gb + Int64(4 * A.TILE_N * 4))
                _softmax_row(
                    rtid,
                    b,
                    t,
                    gv0,
                    gv1,
                    gv2,
                    gv3,
                    gv4,
                    tmem_addr,
                    abar,
                    sMask,
                    sinks.iterator.raw_ptr() + b * Int32(A.HEADS),
                    out_addr,
                    scale_log2,
                )
                t = t + Int32(A.NTILES)
                n = n + Int32(1)
                A._wait(abar + Int32(B_ROW) + (n & Int32(1)), n >> Int32(1) & Int32(1))
                b = Int32(sRow[n & Int32(1)])
        else:
            _mma_warp(abar, sRow, q_base, kv_base, tmem_addr)
        cute.arch.fence_view_async_tmem_load()
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        cute.arch.barrier(barrier_id=3, number_of_threads=self.threads)
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)
        if tid == Int32(0):
            # Every CTA has made its final claim; the last one to leave resets the counters for the next call.
            # All rows were published before they could be claimed, so the indexer no longer touches the queue.
            L.fence_acq_rel_gpu()
            d = L.atom_add_i32(aq_base + Int64(L.AQ_DONE * 4), Int32(1))
            if d == Int32(gdim) - Int32(1):
                L.fence_acq_rel_gpu()
                L.st_relaxed_i32(aq_base + Int64(L.AQ_PUB * 4), Int32(0))
                L.st_relaxed_i32(aq_base + Int64(L.AQ_CLAIM * 4), Int32(0))
                L.st_relaxed_i32(aq_base + Int64(L.AQ_DONE * 4), Int32(0))
            # Do not complete before the indexer grid: the next call's indexer reuses its workspaces.
            cute.arch.griddepcontrol_wait()


def compile_tail(tail: AttentionTail):
    rt = cute.runtime
    qa_fake = rt.make_fake_compact_tensor(
        cutlass.Uint8, (cute.sym_int64(), 64, 512), stride_order=(2, 1, 0), assumed_align=16
    )
    sp_fake = rt.make_fake_compact_tensor(
        cutlass.Uint8, (cute.sym_int64(), 512), stride_order=(1, 0), assumed_align=16
    )
    cp_fake = rt.make_fake_compact_tensor(
        cutlass.Uint8, (cute.sym_int64(), 512), stride_order=(1, 0), assumed_align=16
    )
    sb = cute.sym_int()
    gi_fake = rt.make_fake_compact_tensor(cutlass.Int32, (sb, A.NROWS), stride_order=(1, 0))
    sk_fake = rt.make_fake_compact_tensor(
        cutlass.Float32, (sb, 64), stride_order=(1, 0), assumed_align=16
    )
    ou_fake = rt.make_fake_compact_tensor(
        cutlass.BFloat16, (sb, 64, 512), stride_order=(2, 1, 0), assumed_align=16
    )
    aq_fake = rt.make_fake_compact_tensor(
        cutlass.Int32, (sb, 16), stride_order=(1, 0), assumed_align=16
    )
    stream = rt.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        tail,
        qa_fake,
        sp_fake,
        cp_fake,
        gi_fake,
        sk_fake,
        ou_fake,
        aq_fake,
        Float32(1.0),
        Int32(1),
        stream,
        options="--enable-tvm-ffi --generate-line-info",
    )
