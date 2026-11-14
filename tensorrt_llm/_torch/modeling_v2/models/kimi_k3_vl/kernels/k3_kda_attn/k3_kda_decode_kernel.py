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
"""Kimi K3 KDA plain decode in one launch: the fused projection of ``k3_kda_attn`` (TP16 rank slice, its 26 stream
clusters and three Lamport phases, unchanged) and the gated-delta decode of R <= 8 requests of one token each, on the
pools of ``trtllm::kda_decode``.

The projection's 8 token rows are the R requests' tokens (rows >= R read as zero). Six head clusters (local heads
0-5) of 4 CTAs, cluster rank = V quarter; CTA (h, q) owns V rows [32 q, 32 q + 32) of local head h for every request.
Before ``griddepcontrol.wait`` (nothing there is written by this launch or its predecessors): the slots, the head's
W_fb block (TMA), the conv weights and gate constants, each request's conv windows (the head's q / k channels, this
CTA's 32 v channels) and each request's 32 state rows (16 KB) into shared memory.
After it, per request (warp r = request r where a step is per request):
  phase 1  q, k and f_a rows; f_b on the tensor cores (M = 128 gate channels, N = 8 rows);
           q, k = L2norm(SiLU(conv4(window, new))) (q also scaled); the q / k windows of this CTA's quarter rewritten
           once all four CTAs of the head have read the old ones (cluster barrier);
  phase 2  v and b (bf16 of the two K-half partials): v = SiLU(conv4) of this CTA's 32 channels, its windows
           rewritten; beta = sigmoid(b); decay = exp(lower_bound * sigmoid(exp(A_log) * (g + dt_bias)));
  update   ``kda_decode``'s arithmetic and row / key layout: lane l keys 4 l .. 4 l + 3, warp w rows w, w + 8,
           w + 16, w + 24; S *= decay; r = (v - S k) beta; S += k r; o = S q; S back to the pool;
  phase 3  the output gate; the gated RMSNorm over V (each CTA's two 16-row sums of squares per row to the four CTAs
           of the head); y = o * rms * w_norm * sigmoid(gate), bf16.
Pools: conv bf16 [slots][3 * 768][3] (q | k | v channels, the last three raw inputs, oldest first; slot stride in
elements, a multiple of 8); state fp32 [slots][6][128][128] (V rows, K contiguous; slot stride a multiple of 4).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cutlass._mlir.dialects import llvm as _llvm
from cutlass._mlir.extras import types as _T
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims

try:
    from cutlass.memory.smem import SmemAllocator
except ImportError:  # older DSL layout
    from cutlass.utils import SmemAllocator

from ..k3_kda_verify.k3_kda_verify_kernel import _bf16, _butterfly, _st_async_f32, _store8
from .k3_kda_attn_kernel import (
    BOX_CH,
    BOX_ELEMS,
    BOX_K,
    BUFFERS,
    CLUSTER,
    CONV_W,
    CP_CG,
    FB_TMEM_COLS,
    FUSED_RAW_BYTES,
    FUSED_STAGES,
    HD,
    HEAD_CLUSTERS,
    HK,
    K_IN,
    K_STEP_U,
    MMA_K,
    MMA_N,
    P1_ROWS,
    PART_ROWS,
    PROJ_ROWS,
    SENTINEL,
    STREAM_CLUSTERS,
    THREADS,
    TILE,
    V_CTA,
    X_ELEMS,
    _bf16_bits,
    _mapa_u32,
    _poll_partials,
    _sent16,
    _sent32,
    _SmemCarver,
    _st_b16,
    _stream_role,
)

NR = MMA_N  # request rows of a launch (the projection's token rows)
WIN = CONV_W - 1  # raw inputs a conv window keeps
WIN_QK = HD * WIN  # bf16 of one request's q (or k) window for a head: 128 channels x 3
WIN_V = V_CTA * WIN  # bf16 of one request's v window for a CTA's 32 channels
QK_CHUNKS = WIN_QK // 8  # 16-byte copies per window
V_CHUNKS = WIN_V // 8
REQ_CHUNKS = 2 * QK_CHUNKS + V_CHUNKS
ST_ROWS = V_CTA * HD  # fp32 of one request's state rows owned by a CTA (16 KB)
ST_CHUNKS = ST_ROWS // 4
LANE_KEYS = HD // 32  # keys per lane in the update (kda_decode's float4)
CW_ITEMS = 2 * V_CTA * WIN  # q and k window entries a CTA rewrites per request


@dsl_user_op
def _ld_bf16(smem_ptr, *, loc=None, ip=None):
    """fp32 of one bf16 in shared memory by ld.shared.b16: a channel's three window entries are only 2-byte aligned,
    so their loads must not be merged into one wider load."""
    return cutlass.Float32(
        _llvm.inline_asm(
            _T.f32(), [smem_ptr.toint(loc=loc, ip=ip).ir_value()],
            "{ .reg .b16 h; ld.shared.b16 h, [$1]; cvt.f32.bf16 $0, h; }", "=f,r", has_side_effects=True,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _test_wait_cluster(mbar, parity, *, loc=None, ip=None):
    """mbarrier.test_wait.parity.acquire.cluster on a barrier of this CTA (``mbar``, its shared-memory pointer): whether
    phase ``parity`` has completed, acquiring at cluster scope. The barrier is completed by other CTAs' st.async, whose
    complete_tx releases at cluster scope."""
    done = cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [mbar.toint(loc=loc, ip=ip).ir_value(), cutlass.Int32(parity).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .pred p;\n\tmbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 p, [$1], $2;\n\t"
            "selp.u32 $0, 1, 0, p;\n\t}", "=r,r,r,~{memory}", has_side_effects=True,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip
    return done != cutlass.Int32(0)


@cute.jit
def _decode_head_role(
    tma_wfb,
    p1w: cutlass.Array,
    part: cutlass.Array,
    w_q: cutlass.Array,
    w_k: cutlass.Array,
    w_v: cutlass.Array,
    a_log: cutlass.Array,
    dt_bias: cutlass.Array,
    onorm_w: cutlass.Array,
    conv: cutlass.Array,
    ssm: cutlass.Array,
    slots: cutlass.Array,
    out: cutlass.Array,
    epoch: cutlass.Array,
    smem_a: cutlass.Array,
    smem_b: cutlass.Array,
    bars: cutlass.Array,
    tmem_holder: cutlass.Array,
    s_slot: cutlass.Array,
    s_winq: cutlass.Array,
    s_wink: cutlass.Array,
    s_winv: cutlass.Array,
    s_wq: cutlass.Array,
    s_wk: cutlass.Array,
    s_wv: cutlass.Array,
    s_dtb: cutlass.Array,
    s_onw: cutlass.Array,
    s_nq: cutlass.Array,
    s_nk: cutlass.Array,
    s_gr: cutlass.Array,
    s_q: cutlass.Array,
    s_k: cutlass.Array,
    s_dec: cutlass.Array,
    s_beta: cutlass.Array,
    s_braw: cutlass.Array,
    s_v: cutlass.Array,
    s_o: cutlass.Array,
    s_ss: cutlass.Array,
    s_rs: cutlass.Array,
    s_vp: cutlass.Array,
    s_bp: cutlass.Array,
    s_st: cutlass.Array,
    n_req,
    ssm_stride,
    conv_stride,
    lower_bound: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    eps: cutlass.Constexpr[float],
    USE_PDL: cutlass.Constexpr[bool],
    lbx,
):
    """One CTA of a head cluster (logical clusters 26-31): local head h = cluster - 26, V rows [32 q, 32 q + 32)
    (q = cluster rank) of every request, fed by the stream clusters' Lamport buffers of this launch."""
    tidx, _, _ = cute.arch.thread_idx()
    bx = lbx
    lane = tidx % 32
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    vq = cute.arch.block_idx_in_cluster()
    h = bx // cutlass.Int32(CLUSTER) - cutlass.Int32(STREAM_CLUSTERS)
    v0 = vq * V_CTA
    ch0 = h * HD
    w_full = bars.subview(0)
    acc_done = bars.subview(1)
    ss_ready = bars.subview(2)

    tma_ptr_w = tma_wfb.get_ptr()
    if warp == 0:
        prims.prefetch_tensormap(tma_ptr_w)
        if prims.elect_sync():
            prims.mbarrier_init(w_full, 1)
            prims.mbarrier_init(acc_done, 1)
    elif warp == 2:
        prims.tcgen05_alloc(tmem_holder, FB_TMEM_COLS)
        prims.tcgen05_relinquish_alloc_permit()
    elif warp == 3:
        if prims.elect_sync():
            prims.mbarrier_init(ss_ready, 1)
            prims.mbarrier_arrive_expect_tx(ss_ready, CLUSTER * 2 * NR * 4)
    prims.fence_mbarrier_init()
    prims.barrier_cluster_arrive_relaxed()
    prims.barrier_cluster_wait()
    prims.barrier_cta_sync(0)
    if warp == 0:
        if prims.elect_sync():
            prims.mbarrier_arrive_expect_tx(w_full, HD * HD * 2)
            prims.cp_async_bulk_tensor_shared_cta_global(
                smem_a, tma_ptr_w, (cutlass.Int32(0), ch0, cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0)),
                w_full,
            )  # fmt: skip

    # ---- Before the grid dependency: the slots, the constants, then (asynchronous copies) every request's conv
    # windows and its state rows.
    if tidx < n_req:
        s_slot.store(slots.load(idx=tidx), idx=tidx)
    a_raw = a_log.load(idx=h)
    if tidx < HD:
        wq4 = w_q.load(idx=(ch0 + tidx) * CONV_W, vector_size=CONV_W, alignment=16)
        wk4 = w_k.load(idx=(ch0 + tidx) * CONV_W, vector_size=CONV_W, alignment=16)
        for w in cutlass.range_constexpr(CONV_W):
            s_wq.store(cutlass.Float32(wq4[w]), idx=w * HD + tidx)
            s_wk.store(cutlass.Float32(wk4[w]), idx=w * HD + tidx)
        s_dtb.store(dt_bias.load(idx=ch0 + tidx), idx=tidx)
    elif tidx < HD + V_CTA:
        cv_w = tidx - HD
        wv4 = w_v.load(idx=(ch0 + v0 + cv_w) * CONV_W, vector_size=CONV_W, alignment=16)
        for w in cutlass.range_constexpr(CONV_W):
            s_wv.store(cutlass.Float32(wv4[w]), idx=w * V_CTA + cv_w)
        s_onw.store(onorm_w.load(idx=v0 + cv_w), idx=cv_w)
    for i_o in cutlass.range_constexpr(NR * V_CTA // THREADS):
        s_o.store(cutlass.Float32(0.0), idx=tidx + i_o * THREADS)
    prims.barrier_cta_sync(0)
    for it_w in cutlass.range_constexpr((NR * REQ_CHUNKS + THREADS - 1) // THREADS):
        q_w = tidx + it_w * THREADS
        r_w = q_w // REQ_CHUNKS
        u_w = q_w % REQ_CHUNKS
        if r_w < n_req:
            conv_w = conv.subview(s_slot.load(idx=r_w) * conv_stride)
            if u_w < QK_CHUNKS:
                prims.cp_async_shared_global(
                    s_winq.subview(r_w * WIN_QK + u_w * 8),
                    conv_w.subview(ch0 * WIN + u_w * 8),
                    16,
                    CP_CG,
                )
            elif u_w < 2 * QK_CHUNKS:
                u_k = u_w - QK_CHUNKS
                prims.cp_async_shared_global(
                    s_wink.subview(r_w * WIN_QK + u_k * 8),
                    conv_w.subview((HK + ch0) * WIN + u_k * 8),
                    16,
                    CP_CG,
                )
            else:
                u_v = u_w - 2 * QK_CHUNKS
                prims.cp_async_shared_global(
                    s_winv.subview(r_w * WIN_V + u_v * 8),
                    conv_w.subview((2 * HK + ch0 + v0) * WIN + u_v * 8),
                    16,
                    CP_CG,
                )
    prims.cp_async_commit_group()
    for r_s in cutlass.range_constexpr(NR):
        if r_s < n_req:
            st_src = ssm.subview(s_slot.load(idx=r_s) * ssm_stride + (ch0 + v0) * HD)
            for j_s in cutlass.range_constexpr(ST_CHUNKS // THREADS):
                c_s = (tidx + j_s * THREADS) * 4
                prims.cp_async_shared_global(
                    s_st.subview(r_s * ST_ROWS + c_s), st_src.subview(c_s), 16, CP_CG
                )
    prims.cp_async_commit_group()
    exp_a = cute.math.exp(a_raw, fastmath=True)
    # The windows (the first group) have landed: every CTA of the head has read the head's q / k windows once all
    # four arrive here (release; the acquiring wait follows phase 1), so each may then rewrite its quarter.
    prims.cp_async_wait_group(1)
    prims.barrier_cta_sync(0)
    prims.barrier_cluster_arrive()

    if cutlass.const_expr(USE_PDL):
        prims.griddepcontrol(prims.GridDepAction.WAIT)
    e = epoch.load(idx=bx)
    buf = e % cutlass.Int32(BUFFERS)
    if cutlass.const_expr(USE_PDL):
        if tidx == 0:
            prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)

    # ---- Phase 1 (q, k, f_a): thread i polls 16-byte chunk i of the q | k rows, threads < 128 also chunk i of the
    # f_a rows; q and k as fp32, f_a into the MMA's B operand (128-byte swizzle).
    st_qk = tidx // HD
    t_qk = (tidx % HD) // 16
    j_qk = tidx % 16
    a0 = cutlass.Int32(SENTINEL)
    a1 = cutlass.Int32(SENTINEL)
    a2 = cutlass.Int32(SENTINEL)
    a3 = cutlass.Int32(SENTINEL)
    qk_idx = ((buf * NR + t_qk) * P1_ROWS + st_qk * HK + ch0 + j_qk * 8) // 2
    while _sent16(a0) | _sent16(a1) | _sent16(a2) | _sent16(a3):
        vqk = prims.load_ext(
            p1w.subview(qk_idx), dtype=cutlass.Int32, count=4, order="relaxed", scope="gpu"
        )
        a0 = cutlass.Int32(vqk[0])
        a1 = cutlass.Int32(vqk[1])
        a2 = cutlass.Int32(vqk[2])
        a3 = cutlass.Int32(vqk[3])
    if st_qk == 0:
        _store8(s_nq, (a0, a1, a2, a3), t_qk * HD + j_qk * 8)
    else:
        _store8(s_nk, (a0, a1, a2, a3), t_qk * HD + j_qk * 8)
    if tidx < HD:
        t_fa = tidx // 16
        j_fa = tidx % 16
        f0 = cutlass.Int32(SENTINEL)
        f1 = cutlass.Int32(SENTINEL)
        f2 = cutlass.Int32(SENTINEL)
        f3 = cutlass.Int32(SENTINEL)
        fa_idx = ((buf * NR + t_fa) * P1_ROWS + 2 * HK + j_fa * 8) // 2
        while _sent16(f0) | _sent16(f1) | _sent16(f2) | _sent16(f3):
            vfa = prims.load_ext(
                p1w.subview(fa_idx), dtype=cutlass.Int32, count=4, order="relaxed", scope="gpu"
            )
            f0 = cutlass.Int32(vfa[0])
            f1 = cutlass.Int32(vfa[1])
            f2 = cutlass.Int32(vfa[2])
            f3 = cutlass.Int32(vfa[3])
        # Row t_fa, 16-byte unit u of 64-column chunk c: byte 1024 c + 128 t_fa + 16 (u ^ t_fa).
        sw_word = ((j_fa // 8) * 1024 + t_fa * 128 + ((j_fa % 8) ^ t_fa) * 16) // 4
        smem_b.store((f0, f1, f2, f3), idx=sw_word, alignment=16)
        prims.fence_proxy("async_shared", space=prims.SharedSpace.shared_cta)
    prims.barrier_cta_sync(0)
    prims.barrier_cluster_wait()

    # ---- f_b on the tensor cores (warp 2 issues, warps 4-7 read TMEM), rounded to bf16 as the unfused GEMM's output.
    if warp == 2:
        idesc = prims.Tcgen05InstrDesc.build(
            c_dtype=cutlass.Float32,
            a_dtype=cutlass.BFloat16,
            b_dtype=cutlass.BFloat16,
            n_dim=MMA_N,
            m_dim=HD,
        )
        desc_a_base = prims.Tcgen05SmemDesc.build(
            start_address=smem_a, leading_byte_offset=16, stride_byte_offset=1024,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )  # fmt: skip
        desc_b_base = prims.Tcgen05SmemDesc.build(
            start_address=smem_b, leading_byte_offset=16, stride_byte_offset=1024,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )  # fmt: skip
        tmem_acc = cutlass.inttoptr(tmem_holder.load(), 6, cutlass.Int32)
        while not cute.arch.mbarrier_test_wait(w_full.data_ptr(), 0):
            pass
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        for kb in cutlass.range_constexpr(2 * (BOX_K // MMA_K)):
            box = kb // (BOX_K // MMA_K)
            within = kb % (BOX_K // MMA_K)
            if prims.elect_sync():
                prims.tcgen05_mma(
                    prims.Tcgen05MMAKind.F16, prims.CTAGroup.CTA_1, tmem_acc,
                    desc_a_base + (box * ((HD * BOX_K * 2) >> 4) + within * K_STEP_U),
                    desc_b_base + (box * ((MMA_N * BOX_K * 2) >> 4) + within * K_STEP_U),
                    idesc, kb != 0,
                )  # fmt: skip
        if prims.elect_sync():
            prims.tcgen05_commit(acc_done)
    if warp >= 4:
        while not cute.arch.mbarrier_test_wait(acc_done.data_ptr(), 0):
            pass
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        acc = prims.tcgen05_ld(
            "32x32b", cutlass.inttoptr(tmem_holder.load(), 6, cutlass.Float32), num=MMA_N
        )
        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
        c_fb = (warp - 4) * 32 + lane
        for t_fb in cutlass.range_constexpr(NR):
            s_gr.store(_bf16(cutlass.Float32(acc[t_fb])), idx=t_fb * HD + c_fb)
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        prims.barrier_cta_sync(1, thread_count=128)
        if warp == 4:
            prims.tcgen05_dealloc(
                cutlass.inttoptr(tmem_holder.load(), 6, cutlass.Int32), FB_TMEM_COLS
            )

    # ---- q and k of request w (warp w): conv4 over the window and the new input, SiLU, then the L2 norm with
    # kda_decode's sum order (per 32-channel group a butterfly, then (g0 + g2) + (g1 + g3)); q also scaled.
    if warp < n_req:
        tk = warp
        pq = [cutlass.Float32(0.0)] * LANE_KEYS
        pk = [cutlass.Float32(0.0)] * LANE_KEYS
        for i in cutlass.range_constexpr(LANE_KEYS):
            c = i * 32 + lane
            cq = cutlass.Float32(0.0)
            ck = cutlass.Float32(0.0)
            for w in cutlass.range_constexpr(WIN):
                cq += _ld_bf16(s_winq.subview(tk * WIN_QK + c * WIN + w).data_ptr()) * s_wq.load(
                    idx=w * HD + c
                )
                ck += _ld_bf16(s_wink.subview(tk * WIN_QK + c * WIN + w).data_ptr()) * s_wk.load(
                    idx=w * HD + c
                )
            cq += s_nq.load(idx=tk * HD + c) * s_wq.load(idx=WIN * HD + c)
            ck += s_nk.load(idx=tk * HD + c) * s_wk.load(idx=WIN * HD + c)
            pq[i] = cq * (
                cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.math.exp(-cq, fastmath=True))
            )
            pk[i] = ck * (
                cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.math.exp(-ck, fastmath=True))
            )
        gq = [_butterfly(pq[i] * pq[i]) for i in range(LANE_KEYS)]
        gk = [_butterfly(pk[i] * pk[i]) for i in range(LANE_KEYS)]
        rq = (
            cute.math.rsqrt(
                (gq[0] + gq[2]) + (gq[1] + gq[3]) + cutlass.Float32(1e-6), fastmath=True
            )
            * scale
        )
        rk = cute.math.rsqrt(
            (gk[0] + gk[2]) + (gk[1] + gk[3]) + cutlass.Float32(1e-6), fastmath=True
        )
        for i in cutlass.range_constexpr(LANE_KEYS):
            s_q.store(pq[i] * rq, idx=tk * HD + i * 32 + lane)
            s_k.store(pk[i] * rk, idx=tk * HD + i * 32 + lane)
    # The q / k windows of this CTA's quarter: the two newer raw inputs, then the token's (all four CTAs of the head
    # read the old ones before the cluster wait above).
    for it_c in cutlass.range_constexpr((NR * CW_ITEMS + THREADS - 1) // THREADS):
        q_c = tidx + it_c * THREADS
        r_c = q_c // CW_ITEMS
        u_c = q_c % CW_ITEMS
        if r_c < n_req:
            sec_c = u_c // (V_CTA * WIN)
            ch_c = vq * V_CTA + (u_c % (V_CTA * WIN)) // WIN
            w_c = u_c % WIN
            dst_c = (
                conv.subview(
                    s_slot.load(idx=r_c) * conv_stride + (sec_c * HK + ch0 + ch_c) * WIN + w_c
                )
                .data_ptr()
                .toint()
            )
            # Single 16-bit stores: a channel's entries are only 2-byte aligned.
            if w_c < WIN - 1:
                if sec_c == 0:
                    _st_b16(
                        dst_c,
                        _bf16_bits(
                            _ld_bf16(s_winq.subview(r_c * WIN_QK + ch_c * WIN + w_c + 1).data_ptr())
                        ),
                    )
                else:
                    _st_b16(
                        dst_c,
                        _bf16_bits(
                            _ld_bf16(s_wink.subview(r_c * WIN_QK + ch_c * WIN + w_c + 1).data_ptr())
                        ),
                    )
            else:
                if sec_c == 0:
                    _st_b16(dst_c, _bf16_bits(s_nq.load(idx=r_c * HD + ch_c)))
                else:
                    _st_b16(dst_c, _bf16_bits(s_nk.load(idx=r_c * HD + ch_c)))
    prims.barrier_cta_sync(0)

    # ---- Phase 2 (v, b): the two K-half partials of this CTA's 32 v rows and of this head's b; the projection is
    # their bf16-rounded sum.
    if tidx < 2 * NR * (V_CTA // 4):
        half_v = tidx // (NR * (V_CTA // 4))
        t_v = (tidx // (V_CTA // 4)) % NR
        j_v = tidx % (V_CTA // 4)
        _poll_partials(
            part,
            ((buf * 3 + 0) * 2 + half_v) * (8 * PART_ROWS) + t_v * PART_ROWS + ch0 + v0 + j_v * 4,
            s_vp,
            (half_v * NR + t_v) * V_CTA + j_v * 4,
        )
    elif tidx < 2 * NR * (V_CTA // 4) + 2 * NR:
        ib = tidx - 2 * NR * (V_CTA // 4)
        half_b = ib // NR
        t_b = ib % NR
        wb = cutlass.Int32(SENTINEL)
        while _sent32(wb):
            wb = prims.load_ext(
                part.subview(((buf * 3 + 2) * 2 + half_b) * (8 * PART_ROWS) + t_b * PART_ROWS + h),
                dtype=cutlass.Int32,
                order="relaxed",
                scope="gpu",
            )
        s_bp.store(wb.bitcast(cutlass.Float32), idx=half_b * NR + t_b)
    prims.barrier_cta_sync(0)
    t_cv = tidx // V_CTA
    c_cv = tidx % V_CTA
    nv = _bf16(s_vp.load(idx=t_cv * V_CTA + c_cv) + s_vp.load(idx=(NR + t_cv) * V_CTA + c_cv))
    if tidx < NR:
        s_braw.store(_bf16(s_bp.load(idx=tidx) + s_bp.load(idx=NR + tidx)), idx=tidx)
    if t_cv < n_req:
        # v (one (request, channel) per thread), then its window: the two newer raw inputs and the token's.
        cv = cutlass.Float32(0.0)
        wv_old = [
            _ld_bf16(s_winv.subview(t_cv * WIN_V + c_cv * WIN + w).data_ptr()) for w in range(WIN)
        ]
        for w in cutlass.range_constexpr(WIN):
            cv += wv_old[w] * s_wv.load(idx=w * V_CTA + c_cv)
        cv += nv * s_wv.load(idx=WIN * V_CTA + c_cv)
        s_v.store(
            cv
            * (cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.math.exp(-cv, fastmath=True))),
            idx=t_cv * V_CTA + c_cv,
        )
        dst_v = (
            conv.subview(s_slot.load(idx=t_cv) * conv_stride + (2 * HK + ch0 + v0 + c_cv) * WIN)
            .data_ptr()
            .toint()
        )
        _st_b16(dst_v, _bf16_bits(wv_old[1]))
        _st_b16(dst_v + cutlass.Int64(2), _bf16_bits(wv_old[2]))
        _st_b16(dst_v + cutlass.Int64(4), _bf16_bits(nv))
    prims.barrier_cta_sync(0)
    if tidx < n_req:
        s_beta.store(
            cutlass.Float32(1.0)
            / (cutlass.Float32(1.0) + cute.math.exp(-s_braw.load(idx=tidx), fastmath=True)),
            idx=tidx,
        )
    # The decay of request w's keys (warp w; lane l keys 4 l .. 4 l + 3, the update's layout).
    if warp < n_req:
        for i in cutlass.range_constexpr(LANE_KEYS):
            c = lane * LANE_KEYS + i
            xg = exp_a * (s_gr.load(idx=warp * HD + c) + s_dtb.load(idx=c))
            sig = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.math.exp(-xg, fastmath=True))
            s_dec.store(cute.math.exp(lower_bound * sig, fastmath=True), idx=warp * HD + c)
    prims.cp_async_wait_group(0)
    prims.barrier_cta_sync(0)

    # ---- The update, request by request: warp w rows w, w + 8 and w + 16, w + 24 of this CTA's 32, lane l keys
    # 4 l .. 4 l + 3 (kda_decode's layout and sum order); the rows back to the pool, the outputs to s_o.
    for t in cutlass.range(n_req, unroll=1):
        q4 = s_q.load(idx=t * HD + lane * LANE_KEYS, vector_size=4, alignment=16)
        k4 = s_k.load(idx=t * HD + lane * LANE_KEYS, vector_size=4, alignment=16)
        d4 = s_dec.load(idx=t * HD + lane * LANE_KEYS, vector_size=4, alignment=16)
        beta_t = s_beta.load(idx=t)
        st_dst = ssm.subview(s_slot.load(idx=t) * ssm_stride + (ch0 + v0) * HD)
        for pr in cutlass.range_constexpr(2):
            ra = warp + 16 * pr
            rb = ra + 8
            sa4 = s_st.load(
                idx=t * ST_ROWS + ra * HD + lane * LANE_KEYS, vector_size=4, alignment=16
            )
            sb4 = s_st.load(
                idx=t * ST_ROWS + rb * HD + lane * LANE_KEYS, vector_size=4, alignment=16
            )
            sa = [cutlass.Float32(sa4[i]) * cutlass.Float32(d4[i]) for i in range(LANE_KEYS)]
            sb = [cutlass.Float32(sb4[i]) * cutlass.Float32(d4[i]) for i in range(LANE_KEYS)]
            ska = sa[0] * cutlass.Float32(k4[0])
            skb = sb[0] * cutlass.Float32(k4[0])
            for i in cutlass.range_constexpr(1, LANE_KEYS):
                ska = ska + sa[i] * cutlass.Float32(k4[i])
                skb = skb + sb[i] * cutlass.Float32(k4[i])
            for offset in [16, 8, 4, 2, 1]:
                ska = ska + cute.arch.shuffle_sync_bfly(
                    ska, offset=offset, mask=-1, mask_and_clamp=31
                )
                skb = skb + cute.arch.shuffle_sync_bfly(
                    skb, offset=offset, mask=-1, mask_and_clamp=31
                )
            res_a = (s_v.load(idx=t * V_CTA + ra) - ska) * beta_t
            res_b = (s_v.load(idx=t * V_CTA + rb) - skb) * beta_t
            for i in cutlass.range_constexpr(LANE_KEYS):
                sa[i] = sa[i] + cutlass.Float32(k4[i]) * res_a
                sb[i] = sb[i] + cutlass.Float32(k4[i]) * res_b
            st_dst.store((sa[0], sa[1], sa[2], sa[3]), idx=ra * HD + lane * LANE_KEYS, alignment=16)
            st_dst.store((sb[0], sb[1], sb[2], sb[3]), idx=rb * HD + lane * LANE_KEYS, alignment=16)
            sqa = sa[0] * cutlass.Float32(q4[0])
            sqb = sb[0] * cutlass.Float32(q4[0])
            for i in cutlass.range_constexpr(1, LANE_KEYS):
                sqa = sqa + sa[i] * cutlass.Float32(q4[i])
                sqb = sqb + sb[i] * cutlass.Float32(q4[i])
            for offset in [16, 8, 4, 2, 1]:
                sqa = sqa + cute.arch.shuffle_sync_bfly(
                    sqa, offset=offset, mask=-1, mask_and_clamp=31
                )
                sqb = sqb + cute.arch.shuffle_sync_bfly(
                    sqb, offset=offset, mask=-1, mask_and_clamp=31
                )
            if lane == 0:
                s_o.store(sqa, idx=t * V_CTA + ra)
                s_o.store(sqb, idx=t * V_CTA + rb)

    # ---- Phase 3 (the output gate), then the gated RMSNorm over V: every CTA sends its two 16-row sums of squares
    # per row to all four CTAs of the head. Thread (row t, V row v) reads its gate's two K-half partials first, so
    # their round trip overlaps the norm exchange.
    t_out = tidx // V_CTA
    v_y = tidx % V_CTA
    og_at = (buf * 3 + 1) * 2 * (8 * PART_ROWS) + t_out * PART_ROWS + ch0 + v0 + v_y
    og_h0 = cutlass.Int32(
        prims.load_ext(part.subview(og_at), dtype=cutlass.Int32, order="relaxed", scope="gpu")
    )
    og_h1 = cutlass.Int32(prims.load_ext(part.subview(og_at + 8 * PART_ROWS), dtype=cutlass.Int32, order="relaxed",
                                         scope="gpu"))  # fmt: skip
    prims.barrier_cta_sync(0)
    x_o = s_o.load(idx=tidx)
    ss = x_o * x_o
    for off_ss in [8, 4, 2, 1]:
        ss = ss + cute.arch.shuffle_sync_bfly(ss, offset=off_ss, mask=-1, mask_and_clamp=31)
    if lane % 16 == 0:
        src_slot = vq * 2 + lane // 16
        for r in cutlass.range_constexpr(CLUSTER):
            _st_async_f32(
                _mapa_u32(s_ss.subview(src_slot * NR + t_out).data_ptr(), r),
                ss,
                _mapa_u32(ss_ready.data_ptr(), r),
            )
    # Completed by the head's st.async: acquire at cluster scope.
    while not _test_wait_cluster(ss_ready.data_ptr(), 0):
        pass
    while _sent32(og_h0) | _sent32(og_h1):
        og_h0 = cutlass.Int32(
            prims.load_ext(part.subview(og_at), dtype=cutlass.Int32, order="relaxed", scope="gpu")
        )
        og_h1 = cutlass.Int32(prims.load_ext(part.subview(og_at + 8 * PART_ROWS), dtype=cutlass.Int32,
                                             order="relaxed", scope="gpu"))  # fmt: skip
    if tidx < NR:
        total = cutlass.Float32(0.0)
        for r in cutlass.range_constexpr(2 * CLUSTER):
            total = total + s_ss.load(idx=r * NR + tidx)
        s_rs.store(cute.math.rsqrt(total / HD + eps), idx=tidx)
    prims.barrier_cta_sync(0)
    if t_out < n_req:
        z = _bf16(og_h0.bitcast(cutlass.Float32) + og_h1.bitcast(cutlass.Float32))
        gate = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.math.exp(-z, fastmath=True))
        y = s_o.load(idx=tidx) * s_rs.load(idx=t_out) * s_onw.load(idx=v_y) * gate
        out.store(cutlass.BFloat16(y), idx=(t_out * (HK // HD) + h) * HD + v0 + v_y)

    if tidx == 0:
        epoch.store((e + cutlass.Int32(1)) % cutlass.Int32(BUFFERS), idx=bx)


@cute.kernel
def k3_kda_decode_fused_kernel(
    tma_w: cutlass.GridConstant[
        cuda.TensorMap
    ],  # W [3208, 7168] bf16, 5-D, box 64 cols x 64 rows x 2 chunks
    tma_x: cutlass.GridConstant[
        cuda.TensorMap
    ],  # x [R, 7168] bf16, box 64 cols x 8 rows (rows >= R read as 0)
    tma_wfb: cutlass.GridConstant[
        cuda.TensorMap
    ],  # W_fb [768, 128] bf16, a head's 128 rows by one call
    p1: cutlass.Array,  # int16 bits of bf16 [3][8][1664] (Lamport, written by the stream role)
    p1w: cutlass.Array,  # the same memory as int32 words (polled by the head role)
    part: cutlass.Array,  # int32 bits of fp32 [3][3][2][8][768] (Lamport)
    epoch: cutlass.Array,  # int32 [128]: each CTA's buffer index (launches completed mod 3)
    w_q: cutlass.Array,  # fp32 [768, 4]
    w_k: cutlass.Array,
    w_v: cutlass.Array,
    a_log: cutlass.Array,  # fp32 [6]
    dt_bias: cutlass.Array,  # fp32 [768]
    onorm_w: cutlass.Array,  # fp32 [128]
    conv: cutlass.Array,  # bf16 [slots][2304][3], slot stride conv_stride elements
    ssm: cutlass.Array,  # fp32 [slots][6][128][128], slot stride ssm_stride elements
    slots: cutlass.Array,  # int32 [R]
    out: cutlass.Array,  # bf16 [R][6][128]
    n_req: cutlass.Int32,
    ssm_stride: cutlass.Int64,
    conv_stride: cutlass.Int64,
    lower_bound: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    eps: cutlass.Constexpr[float],
    USE_PDL: cutlass.Constexpr[bool],
):
    # The stream CTAs (clusters 0-25) and the head CTAs (26-31) never share a cluster, so their large arrays alias one
    # raw buffer (k3_kda_attn's carve).
    raw = SmemAllocator().allocate(FUSED_RAW_BYTES, byte_alignment=1024)
    stream_carve = _SmemCarver(raw, FUSED_RAW_BYTES)
    ring_w = stream_carve.view(cutlass.BFloat16, FUSED_STAGES * BOX_ELEMS, 1024)
    ring_x = stream_carve.view(cutlass.BFloat16, FUSED_STAGES * X_ELEMS, 1024)
    mbox = stream_carve.view(cutlass.Float32, 3 * CLUSTER * 32 * 4, 16)
    head_carve = _SmemCarver(raw, FUSED_RAW_BYTES)
    smem_a = head_carve.view(cutlass.BFloat16, HD * HD, 1024)
    smem_b = head_carve.view(cutlass.Int32, MMA_N * HD // 2, 1024)
    s_st = head_carve.view(cutlass.Float32, NR * ST_ROWS)
    s_winq = head_carve.view(cutlass.BFloat16, NR * WIN_QK)
    s_wink = head_carve.view(cutlass.BFloat16, NR * WIN_QK)
    s_winv = head_carve.view(cutlass.BFloat16, NR * WIN_V)
    s_slot = head_carve.view(cutlass.Int32, NR)
    s_wq = head_carve.view(cutlass.Float32, CONV_W * HD)
    s_wk = head_carve.view(cutlass.Float32, CONV_W * HD)
    s_wv = head_carve.view(cutlass.Float32, CONV_W * V_CTA)
    s_dtb = head_carve.view(cutlass.Float32, HD)
    s_onw = head_carve.view(cutlass.Float32, V_CTA)
    s_nq = head_carve.view(cutlass.Float32, NR * HD)
    s_nk = head_carve.view(cutlass.Float32, NR * HD)
    s_gr = head_carve.view(cutlass.Float32, NR * HD)
    s_q = head_carve.view(cutlass.Float32, NR * HD)
    s_k = head_carve.view(cutlass.Float32, NR * HD)
    s_dec = head_carve.view(cutlass.Float32, NR * HD)
    s_beta = head_carve.view(cutlass.Float32, NR)
    s_braw = head_carve.view(cutlass.Float32, NR)
    s_v = head_carve.view(cutlass.Float32, NR * V_CTA)
    s_o = head_carve.view(cutlass.Float32, NR * V_CTA)
    s_ss = head_carve.view(cutlass.Float32, 2 * CLUSTER * NR)
    s_rs = head_carve.view(cutlass.Float32, NR)
    s_vp = head_carve.view(cutlass.Float32, 2 * NR * V_CTA)
    s_bp = head_carve.view(cutlass.Float32, 2 * NR)
    # Barriers and the TMEM holder stay static (both roles initialize their own).
    full = cutlass.Array(cutlass.Int64, FUSED_STAGES, space=cutlass.AddressSpace.smem, alignment=8)
    empty = cutlass.Array(cutlass.Int64, FUSED_STAGES, space=cutlass.AddressSpace.smem, alignment=8)
    acc_done = cutlass.Array(cutlass.Int64, 3, space=cutlass.AddressSpace.smem, alignment=8)
    mbox_bar = cutlass.Array(cutlass.Int64, 3, space=cutlass.AddressSpace.smem, alignment=8)
    tmem_holder = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
    bars = cutlass.Array(cutlass.Int64, 3, space=cutlass.AddressSpace.smem, alignment=8)

    # Launch order = logical order: the f_a / b stream clusters, the q/k/v/og stream clusters, then the head clusters.
    # Beside a predecessor that still holds SMs (the residual-update sandwich before this layer), the CTAs that launch
    # first are the stream CTAs, which have the most bytes to move; the head CTAs' prologue is short. Stream CTAs 0-103,
    # head CTAs 104-127.
    lbx = cutlass.Int32(cute.arch.block_idx()[0])
    if lbx < cutlass.Int32(STREAM_CLUSTERS * CLUSTER):
        _stream_role(tma_w, tma_x, p1, part, epoch, ring_w, ring_x, full, empty, acc_done, mbox_bar, mbox, tmem_holder,
                     USE_PDL, FUSED_STAGES, lbx)  # fmt: skip
    else:
        _decode_head_role(tma_wfb, p1w, part, w_q, w_k, w_v, a_log, dt_bias, onorm_w, conv, ssm, slots, out, epoch,
                          smem_a, smem_b, bars, tmem_holder, s_slot, s_winq, s_wink, s_winv, s_wq, s_wk, s_wv, s_dtb,
                          s_onw, s_nq, s_nk, s_gr, s_q, s_k, s_dec, s_beta, s_braw, s_v, s_o, s_ss, s_rs, s_vp,
                          s_bp, s_st, n_req, ssm_stride, conv_stride, lower_bound, scale, eps, USE_PDL,
                          lbx)  # fmt: skip


@cute.jit
def k3_kda_decode(
    w: cute.Tensor,  # bf16 [3208, 7168]
    x: cute.Tensor,  # bf16 [R, 7168]
    w_fb: cute.Tensor,  # bf16 [768, 128]
    p1: cute.Tensor,  # int16 [3 * 8 * 1664]
    p1w: cute.Tensor,  # the same memory as int32
    part: cute.Tensor,  # int32 [3 * 3 * 2 * 8 * 768]
    epoch: cute.Tensor,  # int32 [128]
    w_q: cute.Tensor,
    w_k: cute.Tensor,
    w_v: cute.Tensor,
    a_log: cute.Tensor,
    dt_bias: cute.Tensor,
    onorm_w: cute.Tensor,
    conv: cute.Tensor,
    ssm: cute.Tensor,
    slots: cute.Tensor,
    out: cute.Tensor,
    n_req: cutlass.Int32,
    ssm_stride: cutlass.Int64,
    conv_stride: cutlass.Int64,
    lower_bound: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    eps: cutlass.Constexpr[float],
    USE_PDL: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    """One launch of the fused KDA projection + plain decode of ``n_req`` requests of one token."""
    tma_w = cuda.create_tensor_map_tiled(
        global_address=w.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=[BOX_K, PROJ_ROWS, K_IN // BOX_K, 1, 1],
        global_strides=[(K_IN * 2) // 16, (BOX_K * 2) // 16, (PROJ_ROWS * K_IN * 2) // 16,
                        (PROJ_ROWS * K_IN * 2) // 16],
        box_dims=[BOX_K, TILE, BOX_CH, 1, 1],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )  # fmt: skip
    tma_x = cuda.create_tensor_map_tiled(
        global_address=x.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=[K_IN, n_req],
        global_strides=[(K_IN * 2) // 16],
        box_dims=[BOX_K, MMA_N],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    tma_wfb = cuda.create_tensor_map_tiled(
        global_address=w_fb.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=[BOX_K, HK, HD // BOX_K, 1, 1],
        global_strides=[(HD * 2) // 16, (BOX_K * 2) // 16, (HK * HD * 2) // 16, (HK * HD * 2) // 16],
        box_dims=[BOX_K, HD, HD // BOX_K, 1, 1],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )  # fmt: skip
    k3_kda_decode_fused_kernel(
        tma_w, tma_x, tma_wfb, p1, p1w, part, epoch, w_q, w_k, w_v, a_log, dt_bias, onorm_w, conv, ssm, slots, out,
        n_req, ssm_stride, conv_stride, lower_bound, scale, eps, USE_PDL,
    ).launch(
        grid=((STREAM_CLUSTERS + HEAD_CLUSTERS) * CLUSTER, 1, 1), block=(THREADS, 1, 1), cluster=(CLUSTER, 1, 1),
        stream=stream, use_pdl=USE_PDL,
        # One CTA per SM (the shared-memory carve); without it ptxas may pick an occupancy-driven register target.
        min_blocks_per_mp=1,
    )  # fmt: skip
