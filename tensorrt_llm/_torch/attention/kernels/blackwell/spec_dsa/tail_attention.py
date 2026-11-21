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
"""specDSA tail attention (K4'): the decode sparse MLA of all rows in parallel, the KV tiles of a row split over a
cluster of CTAs.

Replaces the chain's K3 (local-to-global mapping) + K4 (sparse MLA) + K4b (reduction) behind any exact top-K kernel:
the input is the exact top-512 selection (local compressed positions, ``topk`` [B, 512] int32, as the top-K kernel
writes it) and the attention inputs of the large-batch tail; the output is ``out`` [B, 64 * 512] bf16, the same result
as ``AttentionTail`` of the large-batch kernels.

One cluster of 5 CTAs serves a row. The row has five tiles of 128 KV rows: tile 0 = the sliding window, tiles 1-4 = the
selected rows (entries 128 (j - 1) .. 128 j - 1 of ``topk``), one tile per CTA. Every CTA runs the ``AttentionTail``
pipeline (loader warps: local-to-global mapping + TMA gather4, MMA warp, softmax warps) on its tile with its own
running max, P quantized to E4M3 at that max, and fp32 O in TMEM. The partials are combined in the cluster: the
(max, sum) pairs go to the CTAs that own an output slice, the partial O (bf16, 128-column slice v -> CTA v % 5) goes
to the owner through distributed shared memory (``st.async`` completing the owner's mbarrier), and the owner combines
with 2^(L (m_i - m*)) weights (L = scale * log2 e), applies the sink and the normalization (the arithmetic of
``megakernel_attention._merge_*``) and stores.

Only the cluster size 5 is built and validated (``SUPPORTED_CLUSTER_SIZES``): it serves up to 26 rows, the clusters of
5 CTAs that a 148-SM B200 holds at once at one CTA per SM. The tile split and the slice ownership are written for any
cluster size of 3 to 5, but the clusters of 4 and 3 CTAs (27 to 45 rows) were measured to win nothing over the
large-batch kernels and to leave only 0.5 to 0.8 % of the headroom to the accuracy bar of the real-capture gates (5
CTAs leave about 9 %), so the constructor refuses them.

The launch is a plain stream-ordered launch, or, with ``pdl``, a programmatic dependent launch: the kernel waits for
the grid dependency before it reads the selection (``griddepcontrol.wait``, a no-op in a plain launch), so its CTAs can
start while the top-K kernel still ranks its crossing bin (the multi-CTA top-K of ``topk_multicta`` triggers the launch
after its stream phase) and run the barrier / TMEM setup, the Q load and the sliding-window tile in the shadow of that
kernel. Behind a top-K kernel that never triggers early (the production one) the attribute only costs 0.7 to 0.8 us at
8 to 16 rows (measured): ``pdl`` is for the multi-CTA top-K.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as xcuda
import cutlass.pipeline as pipeline
from cutlass import Boolean, Float32, Int32, Int64, Uint32
from cutlass.experimental import primitives as prims

from . import large_batch_lib as L
from . import megakernel_lib as ML
from .config import DS_V4_FLASH as SPEC
from .config import DS_V4_FLASH_FP4 as SPEC_FP4
from .config import SpecDsaConfig, require_built_for
from .large_batch import (
    HEADS,
    LATENT,
    LOG2_E,
    NEG_FLT_MAX,
    NSLICE,
    PANEL1,
    Q_SLICE_BYTES,
    SLICE,
    SLOT_BYTES,
    TILE_N,
    TM_O,
    TM_SB,
    _final_norm,
    _grow,
    _pack4_e4m3,
    _pack_bf16x2,
    _rescale_o,
    _shfl_idx_i32,
    _shfl_xor_f32,
    _tma_gather4,
    _tmem_st_p8,
    _wait,
)

TOPK = SPEC.topk
NTILES = (SPEC.window + SPEC.topk) // TILE_N  # 5
SLICE_BYTES_O = 16384  # one 128-column slice of the 64-head O partial in bf16
# CTAs per row that are built and validated (see the module docstring).
SUPPORTED_CLUSTER_SIZES = (5,)
E4M3 = cutlass.Float8E4M3FN


class TailAttention:
    """The tail attention kernel; ``compile_tail_attention`` compiles it.

    Args:
        g: CTAs (the cluster) per row, one of ``SUPPORTED_CLUSTER_SIZES``; another value raises ValueError.
        config: a spec of the DS-V4-Flash attention (default: DS-V4-Flash), with either indexer K dtype since the
            kernel does not read the indexer; a config with another attention raises NotImplementedError.
        pdl: launch with the programmatic dependent launch attribute (see the module docstring).

    The compiled function takes, in order: ``q_att`` uint8 [B, 64, 512], ``swa_pool`` and ``c_pool`` uint8 [rows, 512],
    ``topk`` int32 [B, 512], ``kv_lens`` int32 [B], ``bt_swa`` / ``bt_c`` int32 [B, pages], ``sinks`` float32 [64],
    ``out`` bf16 [B, 64, 512], the softmax scale and the number of rows B.
    """

    def __init__(self, g: int, config: SpecDsaConfig | None = None, pdl: bool = False):
        require_built_for(
            config, SPEC_FP4 if config is not None and config.index_k_dtype == "fp4" else SPEC
        )
        if g not in SUPPORTED_CLUSTER_SIZES:
            raise ValueError(
                f"cluster size {g} not supported (built and validated for {SUPPORTED_CLUSTER_SIZES})"
            )
        self.g = g
        self.pdl = pdl
        self.base, self.extra = divmod(NTILES, g)
        self.nkv = self.base + (1 if self.extra else 0)  # most tiles of one CTA
        self.no_max = -(-NSLICE // g)  # most output slices of one CTA
        self.owners = min(g, NSLICE)
        self.threads = 288  # warps 0-3 loader, 4-7 softmax / combine, 8 MMA issue
        # mbarrier slots (8 B each; all single-arrival except B_MASK)
        self.B_Q = 0  # four Q slices                       loader -> MMA
        self.B_KV = 4  # 4 slices per local tile           loader -> MMA
        self.B_SFULL = self.B_KV + 4 * self.nkv  # two S buffers: QK done   MMA -> softmax
        self.B_PV = self.B_SFULL + 2  # PV done                              MMA -> softmax
        self.B_PRDY = self.B_PV + 1  # P stored and O rescaled              softmax -> MMA
        self.B_MASK = self.B_PRDY + 1  # validity mask of a tile (4 arrivals)  loader -> softmax
        self.B_STAT = self.B_MASK + self.nkv  # (max, sum) of every CTA           peers -> owner
        self.B_RX = self.B_STAT + 1  # partial O slices                      peers -> owner
        self.B_PVS = self.B_RX + (
            g - 1
        )  # PV of one output slice done (slice-wise export)   MMA -> softmax
        self.nbar = self.B_PVS + NSLICE

    @cute.jit
    def __call__(
        self,
        q_att: cute.Tensor,
        swa_pool: cute.Tensor,
        c_pool: cute.Tensor,
        topk: cute.Tensor,
        kv_lens: cute.Tensor,
        bt_swa: cute.Tensor,
        bt_c: cute.Tensor,
        sinks: cute.Tensor,
        out: cute.Tensor,
        softmax_scale: Float32,
        nrows: Int32,
        stream: cuda.CUstream,
    ):
        tma_q = xcuda.create_tensor_map_tiled(
            global_address=q_att.iterator.toint(),
            dtype=cutlass.Float8E4M3FN,
            global_dims=[LATENT, HEADS, q_att.shape[0]],
            global_strides=[LATENT // 16, HEADS * LATENT // 16],
            box_dims=[SLICE, HEADS, 1],
            swizzle=xcuda.TensorMapSwizzle.s128b,
        )
        tma_swa = xcuda.create_tensor_map_tiled(
            global_address=swa_pool.iterator.toint(),
            dtype=cutlass.Float8E4M3FN,
            global_dims=[LATENT, swa_pool.shape[0]],
            global_strides=[LATENT // 16],
            box_dims=[SLICE, 1],
            swizzle=xcuda.TensorMapSwizzle.s128b,
        )
        tma_cmp = xcuda.create_tensor_map_tiled(
            global_address=c_pool.iterator.toint(),
            dtype=cutlass.Float8E4M3FN,
            global_dims=[LATENT, c_pool.shape[0]],
            global_strides=[LATENT // 16],
            box_dims=[SLICE, 1],
            swizzle=xcuda.TensorMapSwizzle.s128b,
        )
        self.kernel(
            tma_q,
            tma_swa,
            tma_cmp,
            topk,
            kv_lens,
            bt_swa,
            bt_c,
            sinks,
            out,
            softmax_scale,
        ).launch(
            grid=(nrows * Int32(self.g), 1, 1),
            block=(self.threads, 1, 1),
            cluster=(self.g, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.pdl,
        )

    # ------------------------------------------------------------------------------------------------------------
    @cute.jit
    def _loader(
        self,
        tid,
        row,
        j0,
        nt,
        tma_q,
        tma_swa,
        tma_cmp,
        topk,
        kv_lens,
        bt_swa,
        bt_c,
        q_base,
        kv_base,
        abar,
        s_msk,
    ):
        wl = tid >> Int32(5)
        lane = tid & Int32(31)
        if tid == Int32(0):
            for c in cutlass.range_constexpr(NSLICE):
                prims.mbarrier_arrive_expect_tx(abar + Int32(self.B_Q + c), Q_SLICE_BYTES)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    q_base + Int32(c * Q_SLICE_BYTES),
                    tma_q.get_ptr(),
                    [Int32(c * SLICE), Int32(0), row],
                    abar + Int32(self.B_Q + c),
                )
        mbs = Int32(cute.size(bt_swa, mode=[1]))
        mbc = Int32(cute.size(bt_c, mode=[1]))
        topk_base = topk.iterator.toint() + Int64(row) * Int64(TOPK * 4) + Int64(tid * Int32(4))
        t0 = Int32(0)
        if j0 == Int32(0):
            # the window tile does not depend on the top-K kernel: gathered before the wait; the block-table row of
            # request ``row`` is row ``row`` (the specDSA contract)
            pos = Int32(kv_lens[row]) - Int32(1)
            grow = L.l2g_swa(tid, pos, row, bt_swa, mbs, Int32(0))
            vm = Int32(cute.arch.vote_ballot_sync(grow >= Int32(0)))
            if lane == Int32(0):
                s_msk[wl] = vm
                prims.mbarrier_arrive(abar + Int32(self.B_MASK))
            self._gather(
                kv_base,
                abar + Int32(self.B_KV),
                tma_swa,
                tma_cmp,
                Boolean(True),
                _grow(grow),
                wl,
                lane,
            )
            t0 = Int32(1)
        cute.arch.griddepcontrol_wait()
        for t in cutlass.range(t0, nt, unroll=1):
            jg = j0 + t
            ta = topk_base + Int64((jg - Int32(1)) * Int32(TILE_N * 4))
            c = L.ld_cg_i32(ta)
            grow = L.l2g_compressed(c, row, bt_c, mbc, Int32(0))
            vm = Int32(cute.arch.vote_ballot_sync(grow >= Int32(0)))
            if lane == Int32(0):
                s_msk[t * Int32(4) + wl] = vm
                prims.mbarrier_arrive(abar + Int32(self.B_MASK) + t)
            self._gather(
                kv_base + t * Int32(4 * SLOT_BYTES),
                abar + Int32(self.B_KV) + t * Int32(4),
                tma_swa,
                tma_cmp,
                Boolean(False),
                _grow(grow),
                wl,
                lane,
            )

    @cute.jit
    def _gather(self, slot_base, mbar_base, tma_swa, tma_cmp, use_swa, gidx_t, wl, lane):
        # slice by slice (all rows of slice 0 first): the MMA warp starts QK on a slice as soon as it lands
        if wl == Int32(0):
            if prims.elect_sync():
                for c in cutlass.range_constexpr(NSLICE):
                    prims.mbarrier_arrive_expect_tx(mbar_base + Int32(c), SLOT_BYTES)
        dst0 = slot_base + wl * Int32(32 * SLICE)
        rows = []
        for q in cutlass.range_constexpr(8):
            for k in cutlass.range_constexpr(4):
                rows.append(_shfl_idx_i32(gidx_t, Int32(4 * q + k)))
        for c in cutlass.range_constexpr(NSLICE):
            for q in cutlass.range_constexpr(8):
                dst = dst0 + Int32(c * SLOT_BYTES + 4 * q * SLICE)
                if use_swa:
                    _tma_gather4(
                        dst,
                        tma_swa.get_ptr(),
                        mbar_base + Int32(c),
                        Int32(c * SLICE),
                        rows[4 * q],
                        rows[4 * q + 1],
                        rows[4 * q + 2],
                        rows[4 * q + 3],
                    )
                else:
                    _tma_gather4(
                        dst,
                        tma_cmp.get_ptr(),
                        mbar_base + Int32(c),
                        Int32(c * SLICE),
                        rows[4 * q],
                        rows[4 * q + 1],
                        rows[4 * q + 2],
                        rows[4 * q + 3],
                    )

    # ------------------------------------------------------------------------------------------------------------
    @cute.jit
    def _pv(self, abar, tp, tmem, kv_base, idesc_pv):
        # O += P(tp) V(tp) once the softmax warps have stored P and rescaled O.
        _wait(abar + Int32(self.B_PRDY), tp & Int32(1))
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        if prims.elect_sync():
            s_addr = tmem + (tp & Int32(1)) * Int32(TM_SB)
            for v in cutlass.range_constexpr(NSLICE):
                vdesc = prims.Tcgen05SmemDesc.build(
                    kv_base + (tp * Int32(4) + Int32(v)) * Int32(SLOT_BYTES),
                    leading_byte_offset=0,
                    stride_byte_offset=1024,
                    layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
                )
                d_addr = tmem + Int32(TM_O + (v & 1) * 128 + (v >> 1) * PANEL1)
                a_addr = s_addr + Int32(TILE_N // 4 + (v >> 1) * PANEL1)
                for kb in cutlass.range_constexpr(TILE_N // 32):
                    en = Boolean(True)
                    if cutlass.const_expr(kb == 0):
                        en = Boolean(tp > Int32(0))
                    prims.tcgen05_mma(
                        prims.Tcgen05MMAKind.F8F6F4,
                        prims.CTAGroup.CTA_1,
                        prims.make_tmem_ptr(d_addr, Float32),
                        prims.make_tmem_ptr(a_addr + Int32(8 * kb), E4M3),
                        vdesc.advance_start_address(kb * 32 * SLICE),
                        idesc_pv,
                        en,
                    )
                prims.tcgen05_commit(abar + Int32(self.B_PVS + v), group=prims.CTAGroup.CTA_1)
            prims.tcgen05_commit(abar + Int32(self.B_PV), group=prims.CTAGroup.CTA_1)

    @cute.jit
    def _mma(self, nt, abar, q_base, kv_base, tmem):
        idesc_qk = prims.Tcgen05InstrDesc.build(
            c_dtype=Float32, a_dtype=E4M3, b_dtype=E4M3, n_dim=TILE_N, m_dim=HEADS
        )
        idesc_pv = prims.Tcgen05InstrDesc.build(
            c_dtype=Float32,
            a_dtype=E4M3,
            b_dtype=E4M3,
            a_major=0,
            b_major=1,
            n_dim=SLICE,
            m_dim=HEADS,
        )
        for t in cutlass.range(nt, unroll=1):
            g4 = t * Int32(4)
            s_addr = tmem + (t & Int32(1)) * Int32(TM_SB)
            if t == Int32(0):
                for c in cutlass.range_constexpr(NSLICE):
                    _wait(abar + Int32(self.B_Q + c), Int32(0))
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            for c in cutlass.range_constexpr(NSLICE):
                # QK on slice c as soon as its KV rows have landed
                _wait(abar + Int32(self.B_KV) + g4 + Int32(c), Int32(0))
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                if prims.elect_sync():
                    qdesc = prims.Tcgen05SmemDesc.build(
                        q_base + Int32(c * Q_SLICE_BYTES),
                        leading_byte_offset=Q_SLICE_BYTES,
                        stride_byte_offset=1024,
                        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
                    )
                    kdesc = prims.Tcgen05SmemDesc.build(
                        kv_base + (g4 + Int32(c)) * Int32(SLOT_BYTES),
                        leading_byte_offset=SLOT_BYTES,
                        stride_byte_offset=1024,
                        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
                    )
                    for kb in cutlass.range_constexpr(SLICE // 32):
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
            if prims.elect_sync():
                prims.tcgen05_commit(
                    abar + Int32(self.B_SFULL) + (t & Int32(1)), group=prims.CTAGroup.CTA_1
                )
            if t >= Int32(1):
                self._pv(abar, t - Int32(1), tmem, kv_base, idesc_pv)
        self._pv(abar, nt - Int32(1), tmem, kv_base, idesc_pv)

    # ------------------------------------------------------------------------------------------------------------
    @cute.jit
    def _softmax(
        self,
        rtid,
        r,
        nt,
        tmem,
        abar,
        s_msk,
        sink_ptr,
        out_addr: Int64,
        row,
        scale_log2: Float32,
        s_stat,
        s_rx,
    ):
        wl = rtid >> Int32(5)
        lane = rtid & Int32(31)
        head = wl * Int32(16) + (lane & Int32(15))
        half = lane >> Int32(4)
        lane_base = wl * Int32(32) << Int32(16)
        sink = Float32((sink_ptr + head).load())
        Ls = scale_log2
        G = self.g
        stat_a = Int32(s_stat.iterator.toint())
        rx_a = Int32(s_rx.iterator.toint())
        bar_stat_a = Int32((abar + Int32(self.B_STAT)).toint())
        bar_rx_a = Int32((abar + Int32(self.B_RX)).toint())
        # receive accounting of an owner: (max, sum) of the G CTAs and (G - 1) partial slices per owned slice
        n_own = Int32(0)
        for oi in cutlass.range_constexpr(self.no_max):
            if r + Int32(oi * G) < Int32(NSLICE):
                n_own = n_own + Int32(1)
        if rtid == Int32(0):
            if r < Int32(self.owners):
                prims.mbarrier_arrive_expect_tx(abar + Int32(self.B_STAT), Int32(G * 512))
                for k in cutlass.range_constexpr(G - 1):
                    prims.mbarrier_arrive_expect_tx(
                        abar + Int32(self.B_RX + k), n_own * Int32(SLICE_BYTES_O)
                    )
        m_run = Float32(NEG_FLT_MAX)
        l_run = Float32(0.0)
        for t in cutlass.range(nt, unroll=1):
            sb = t & Int32(1)
            _wait(abar + Int32(self.B_MASK) + t, Int32(0))
            mk = Int32(s_msk[t * Int32(4) + half * Int32(2)])
            mk2 = Int32(s_msk[t * Int32(4) + half * Int32(2) + Int32(1)])
            _wait(abar + Int32(self.B_SFULL) + sb, t >> Int32(1) & Int32(1))
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
            mx1 = Float32(NEG_FLT_MAX)
            mx2 = Float32(NEG_FLT_MAX)
            mx3 = Float32(NEG_FLT_MAX)
            for i in cutlass.range_constexpr(0, 64, 4):
                mx0 = cute.math.max(mx0, xs[i], ftz=True)
                mx1 = cute.math.max(mx1, xs[i + 1], ftz=True)
                mx2 = cute.math.max(mx2, xs[i + 2], ftz=True)
                mx3 = cute.math.max(mx3, xs[i + 3], ftz=True)
            lmax = cute.math.max(
                cute.math.max(mx0, mx1, ftz=True), cute.math.max(mx2, mx3, ftz=True), ftz=True
            )
            lmax = cute.math.max(lmax, _shfl_xor_f32(lmax, 16), ftz=True)
            new_max = lmax
            safe_max = new_max
            if safe_max == Float32(NEG_FLT_MAX):
                safe_max = Float32(0.0)
            neg_scaled = -Ls * safe_max
            bias = neg_scaled + Float32(8.8073549)
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
                regs.append(_pack4_e4m3(p0, p1, p2, p3))
            local_sum = sm0 + sm1 + (sm2 + sm3)
            p_addr = tsa + Int32(TILE_N // 4)
            _tmem_st_p8(p_addr, regs[0:8])
            _tmem_st_p8(p_addr + Int32(8), regs[8:16])
            _tmem_st_p8(p_addr + Int32(PANEL1), regs[0:8])
            _tmem_st_p8(p_addr + Int32(PANEL1 + 8), regs[8:16])
            exp_scale = cute.math.exp2(Ls * (old_max - new_max), fastmath=True)
            l_run = exp_scale * l_run + local_sum
            m_run = new_max
            if t > Int32(0):
                # O is still being accumulated by PV(t - 1): wait before rescaling it.
                _wait(abar + Int32(self.B_PV), t - Int32(1) & Int32(1))
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                ident = old_max == new_max
                scale = Float32(1.0)
                if not ident:
                    scale = exp_scale
                skip_c = prims.vote_sync(4294967295, ident, prims.VoteSync.ALL)
                if not skip_c:
                    _rescale_o(tmem, lane_base, scale)
            prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            cute.arch.barrier(barrier_id=4, number_of_threads=128)
            if rtid == Int32(0):
                prims.mbarrier_arrive(abar + Int32(self.B_PRDY))
        # ------------------------------------------------------------------------------------------------------
        # combine: (max, sum) to the owners (every CTA sends to every owner, itself included)
        l_tot = l_run + _shfl_xor_f32(l_run, 16)
        cute.arch.cluster_wait()
        if half == Int32(0):
            for d in cutlass.range_constexpr(self.owners):
                rc = ML.mapa_u32(stat_a + (r * Int32(64) + head) * Int32(8), Int32(d))
                mc = ML.mapa_u32(bar_stat_a, Int32(d))
                ML.st_async_v2(rc, mc, L.u32_of_f32(m_run), L.u32_of_f32(l_tot))
        # owners: weights of the G partials (the (max, sum) pairs arrive while the last PV runs)
        wn_me = Float32(0.0)
        wn = []
        if r < Int32(self.owners):
            ML.wait_cl(bar_stat_a, Int32(0))
        ms = []
        ls = []
        for s in cutlass.range_constexpr(G):
            sa = stat_a + (Int32(s * 64) + head) * Int32(8)
            ms.append(L.f32_of_u32(Uint32(ML.lds_u32(sa))))
            ls.append(L.f32_of_u32(Uint32(ML.lds_u32(sa + Int32(4)))))
        m_star = ms[0]
        for s in cutlass.range_constexpr(1, G):
            m_star = cute.math.max(m_star, ms[s], ftz=True)
        ws = []
        for s in cutlass.range_constexpr(G):
            ws.append(cute.math.exp2(Ls * (ms[s] - m_star), fastmath=True))
        l_star = ws[0] * ls[0]
        for s in cutlass.range_constexpr(1, G):
            l_star = l_star + ws[s] * ls[s]
        norm = _final_norm(m_star, l_star, sink, scale_log2)
        for s in cutlass.range_constexpr(G):
            wn.append(ws[s] * norm)
        wn_me = wn[0]
        for s in cutlass.range_constexpr(1, G):
            wn_me = cutlass.select_(r == Int32(s), wn[s], wn_me)
        # partial O slices that another CTA owns: as soon as the PV MMAs of a slice have completed, its 64 heads x 128
        # columns (bf16) go from the registers into the owner's receive buffer (``st.async``), completing the owner's
        # per-source mbarrier
        for v in cutlass.range_constexpr(NSLICE):
            ov = v % G
            oi = v // G
            if r != Int32(ov):
                _wait(abar + Int32(self.B_PVS + v), nt - Int32(1) & Int32(1))
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                raws = []
                for hc in cutlass.range_constexpr(2):
                    o_addr = (
                        tmem + lane_base + Int32(TM_O + v % 2 * 128 + v // 2 * PANEL1 + 32 * hc)
                    )
                    raws.append(
                        prims.tcgen05_ld(
                            prims.Tcgen05LdStShape.SHAPE_16X32BX2,
                            prims.make_tmem_ptr(o_addr, Float32),
                            num=32,
                            offset=Int64(64),
                        )
                    )
                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                words = []
                for hc in cutlass.range_constexpr(2):
                    o = raws[hc]
                    for pi in cutlass.range_constexpr(16):
                        words.append(_pack_bf16x2(Float32(o[2 * pi]), Float32(o[2 * pi + 1])))
                sidx = r - cutlass.select_(r > Int32(ov), Int32(1), Int32(0))
                off = (sidx * Int32(self.no_max) + Int32(oi)) * Int32(SLICE_BYTES_O)
                rc0 = ML.mapa_u32(rx_a + off + rtid * Int32(16), Int32(ov))
                mc = ML.mapa_u32(bar_rx_a + sidx * Int32(8), Int32(ov))
                for q in cutlass.range_constexpr(8):
                    ML.st_async_v4(
                        rc0 + Int32(q * 2048),
                        mc,
                        words[4 * q],
                        words[4 * q + 1],
                        words[4 * q + 2],
                        words[4 * q + 3],
                    )
        # owners: combine
        if r < Int32(self.owners):
            for oi in cutlass.range_constexpr(self.no_max):
                vown = r + Int32(oi * G)
                if vown < Int32(NSLICE):
                    _wait(abar + Int32(self.B_PVS) + vown, nt - Int32(1) & Int32(1))
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                    own = []
                    for hc in cutlass.range_constexpr(2):
                        o_addr = (
                            tmem
                            + lane_base
                            + Int32(TM_O)
                            + (vown & Int32(1)) * Int32(128)
                            + (vown >> Int32(1)) * Int32(PANEL1)
                            + Int32(32 * hc)
                        )
                        o = prims.tcgen05_ld(
                            prims.Tcgen05LdStShape.SHAPE_16X32BX2,
                            prims.make_tmem_ptr(o_addr, Float32),
                            num=32,
                            offset=Int64(64),
                        )
                        own.append(o)
                    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                    acc = []
                    for hc in cutlass.range_constexpr(2):
                        for i in cutlass.range_constexpr(32):
                            acc.append(wn_me * Float32(own[hc][i]))
                    for s in cutlass.range_constexpr(G):
                        if r != Int32(s):
                            slot = Int32(s) - cutlass.select_(Int32(s) > r, Int32(1), Int32(0))
                            ML.wait_cl(bar_rx_a + slot * Int32(8), Int32(0))
                            base = rx_a + (slot * Int32(self.no_max) + Int32(oi)) * Int32(
                                SLICE_BYTES_O
                            )
                            for q in cutlass.range_constexpr(8):
                                xq = ML.lds_v4(base + Int32(q * 2048) + rtid * Int32(16))
                                for e in cutlass.range_constexpr(4):
                                    wd = Uint32(xq[e])
                                    acc[8 * q + 2 * e] = acc[8 * q + 2 * e] + wn[s] * L.f32_of_u32(
                                        wd << Uint32(16)
                                    )
                                    acc[8 * q + 2 * e + 1] = acc[8 * q + 2 * e + 1] + wn[
                                        s
                                    ] * L.f32_of_u32(wd & Uint32(4294901760))
                    ob = []
                    for pi in cutlass.range_constexpr(32):
                        ob.append(_pack_bf16x2(acc[2 * pi], acc[2 * pi + 1]))
                    rowb = (
                        out_addr
                        + Int64(row) * Int64(HEADS * LATENT * 2)
                        + Int64(
                            (head * Int32(LATENT) + vown * Int32(SLICE) + half * Int32(64))
                            * Int32(2)
                        )
                    )
                    for ch in cutlass.range_constexpr(8):
                        ML.st_g_v4(
                            rowb + Int64(ch * 16),
                            ob[4 * ch],
                            ob[4 * ch + 1],
                            ob[4 * ch + 2],
                            ob[4 * ch + 3],
                        )

    # ------------------------------------------------------------------------------------------------------------
    @cute.kernel
    def kernel(
        self,
        tma_q: cutlass.GridConstant[xcuda.TensorMap],
        tma_swa: cutlass.GridConstant[xcuda.TensorMap],
        tma_cmp: cutlass.GridConstant[xcuda.TensorMap],
        topk: cute.Tensor,
        kv_lens: cute.Tensor,
        bt_swa: cute.Tensor,
        bt_c: cute.Tensor,
        sinks: cute.Tensor,
        out: cute.Tensor,
        softmax_scale: Float32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        G = self.g
        row = bidx // Int32(G)
        r = bidx - row * Int32(G)  # rank in the cluster
        smem = cutlass.utils.SmemAllocator()
        sQ = smem.allocate_tensor(
            cutlass.Float8E4M3FN, cute.make_layout(NSLICE * Q_SLICE_BYTES), byte_alignment=1024
        )
        sKV = smem.allocate_tensor(
            cutlass.Float8E4M3FN, cute.make_layout(self.nkv * 4 * SLOT_BYTES), byte_alignment=1024
        )
        sRX = smem.allocate_tensor(
            cutlass.Uint8,
            cute.make_layout((G - 1) * self.no_max * 8 * 2048),
            byte_alignment=1024,
        )
        sBar = smem.allocate_tensor(Int64, cute.make_layout(32), byte_alignment=16)
        sMask = smem.allocate_tensor(Int32, cute.make_layout(16), byte_alignment=16)
        sStat = smem.allocate_tensor(Float32, cute.make_layout(G * 128), byte_alignment=16)
        sTmem = smem.allocate_tensor(Int32, cute.make_layout(1), byte_alignment=16)
        abar = sBar.iterator.raw_ptr()
        if tid < Int32(self.nbar):
            cnt = Int32(1)
            if (tid >= Int32(self.B_MASK)) & (tid < Int32(self.B_MASK + self.nkv)):
                cnt = Int32(4)
            ML.mbar_init(Int32((abar + tid).toint()), cnt)
            ML.fence_mbar_init_cluster()
        cute.arch.cluster_arrive_relaxed()
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
        j0 = r * Int32(self.base) + cutlass.min(r, Int32(self.extra))
        nt = Int32(self.base) + cutlass.select_(r < Int32(self.extra), Int32(1), Int32(0))
        if tid < Int32(128):
            self._loader(
                tid,
                row,
                j0,
                nt,
                tma_q,
                tma_swa,
                tma_cmp,
                topk,
                kv_lens,
                bt_swa,
                bt_c,
                q_base,
                kv_base,
                abar,
                sMask,
            )
        elif tid < Int32(256):
            rtid = tid - Int32(128)
            self._softmax(
                rtid,
                r,
                nt,
                tmem_addr,
                abar,
                sMask,
                sinks.iterator.raw_ptr(),
                out.iterator.toint(),
                row,
                softmax_scale * Float32(LOG2_E),
                sStat,
                sRX,
            )
        else:
            self._mma(nt, abar, q_base, kv_base, tmem_addr)
        if tid < Int32(128) or tid >= Int32(256):
            # the softmax warps waited for the start barrier before their first remote operation; the others do it now
            # (one wait per arrive)
            cute.arch.cluster_wait()
        cute.arch.fence_view_async_tmem_load()
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        cute.arch.barrier(barrier_id=3, number_of_threads=self.threads)
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)


def compile_tail_attention(tail: TailAttention):
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
    tk_fake = rt.make_fake_compact_tensor(cutlass.Int32, (sb, TOPK), stride_order=(1, 0))
    kvl_fake = rt.make_fake_compact_tensor(cutlass.Int32, (sb,), stride_order=(0,))
    bts_fake = rt.make_fake_compact_tensor(cutlass.Int32, (sb, cute.sym_int()), stride_order=(1, 0))
    btc_fake = rt.make_fake_compact_tensor(cutlass.Int32, (sb, cute.sym_int()), stride_order=(1, 0))
    sk_fake = rt.make_fake_compact_tensor(
        cutlass.Float32, (64,), stride_order=(0,), assumed_align=16
    )
    ou_fake = rt.make_fake_compact_tensor(
        cutlass.BFloat16, (sb, 64, 512), stride_order=(2, 1, 0), assumed_align=16
    )
    stream = rt.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        tail,
        qa_fake,
        sp_fake,
        cp_fake,
        tk_fake,
        kvl_fake,
        bts_fake,
        btc_fake,
        sk_fake,
        ou_fake,
        Float32(1.0),
        Int32(1),
        stream,
        options="--enable-tvm-ffi --generate-line-info",
    )
