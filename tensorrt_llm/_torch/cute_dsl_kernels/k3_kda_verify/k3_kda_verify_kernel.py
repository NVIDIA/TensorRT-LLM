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
"""Kimi K3 KDA speculative verify as one kernel: the forget-gate up-projection (f_b), the per-token pre-compute and
the delta-rule recurrence over the 1 + NUM_SPEC verify tokens. It commits the golden token's state and each draft's
records (vn, beta * k, decay), from which the next round replays the accepted drafts.

Grid (H, N, 8), cluster (1, 1, 8), 256 threads. CTA (h, n, s) owns V rows [16 s, 16 s + 16) of local head h for
request n; warp w owns rows 16 s + 2 w and 16 s + 2 w + 1, and lane l keys k = 32 i + l (i < 4) of each.

Before ``griddepcontrol.wait`` (written by the previous verify step, or constant):
  the slot and P, the drafts the sampler accepted last round; the starting state into registers (the pool state,
  committed after the last golden token, with P drafts replayed from their records); the raw conv inputs at positions
  -3..-1 before the first new token (conv-cache columns P..P+2) for q and k (128 channels) and v (this CTA's 16);
  the conv weights, dt_bias, A_log and the output-norm weight; the head's W_fb^T block by TMA.
After the wait (this step's fused projection rows [q | k | v | onorm gate | f_a | b]): the new tokens' raw q, k, v,
  f_a, b and output gate.
f_b:          g[t, c] = bf16(sum_i f_a[t, i] W_fb[128 h + c, i]), rounded as the unfused bf16 GEMV output is.
Pre-compute:  warp t = verify token t: q, k = l2norm(silu(conv4(u))) (q also scaled), the lower-bound gate, beta and
              this CTA's 16 v channels.
Recurrence:   the arithmetic of ``kda_mtp_decode``'s V-split path over the verify tokens, unrolled; the state after
              token 0 (the golden token) is committed to the pool; token t >= 1 leaves its draft's records (vn,
              beta * k, decay) in ``state_tok[slot]``.
Epilogue:     the gated RMSNorm of the outputs over V (per-token sums of squares stored into every peer's shared
              memory, one cluster barrier), bf16 output rows; CTA 0 rewrites the q/k conv cache (the window at the
              golden token and the raw inputs of the drafts), every CTA the same for its v channels.

With FOLD_FB False the gate comes from ``g_ext`` (the unfused f_b output); the kernel is then bit-exact against
``kda_mtp_decode`` fed the same gate and replaying the same accepted drafts.
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

K = 128  # key head dim; the value head dim V is the same
V = 128
CONV_W = 4
THREADS = 256
WARPS = THREADS // 32
V_SPLIT = 8
V_CTA = V // V_SPLIT  # V rows per CTA
ROWS = V_CTA // WARPS  # V rows per warp
VEC = K // 32  # keys per lane
BF16_BYTES = 2
CHUNK = 8  # bf16 per 16-byte load
SMEM = cutlass.AddressSpace.smem
RMEM = cutlass.AddressSpace.rmem

assert ROWS == 2, "the recurrence below processes one pair of V rows per warp"

# f_b on the tensor cores: D[c, t] = sum_i W_fb[128 h + c, i] f_a[t, i], M = 128 channels, N = 8 tokens, K = 128.
CTA_M = 128
MMA_N = 8
MMA_K = 16
TMA_K_BOX = 64
TMA_COPY_ITERS = K // TMA_K_BOX
K_BLOCKS_PER_HALF = TMA_K_BOX // MMA_K
LEADING = 16
STRIDE = 8 * TMA_K_BOX * BF16_BYTES
A_HALF_ELEMS = CTA_M * TMA_K_BOX
B_HALF_ELEMS = MMA_N * TMA_K_BOX
STEP = (MMA_K * BF16_BYTES) >> 4
A_BOX = A_HALF_ELEMS >> 3
B_BOX = B_HALF_ELEMS >> 3
TMEM_COLS = 32


@dsl_user_op
def _mapa_u32(smem_ptr, peer, *, loc=None, ip=None):
    """The shared::cluster address of this CTA's shared-memory location in cluster CTA ``peer``."""
    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [smem_ptr.toint(loc=loc, ip=ip).ir_value(), cutlass.Int32(peer).ir_value(loc=loc, ip=ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;", "=r,r,r", has_side_effects=False,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _test_wait_cluster(mbar, parity, *, loc=None, ip=None):
    """mbarrier.test_wait.parity.acquire.cluster on a barrier of this CTA (``mbar``, its shared-memory pointer): whether
    phase ``parity`` has completed, acquiring at cluster scope. For barriers that other CTAs' st.async complete (their
    complete_tx releases at cluster scope; a CTA-scope acquire does not synchronize with it)."""
    done = cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [mbar.toint(loc=loc, ip=ip).ir_value(), cutlass.Int32(parity).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .pred p;\n\tmbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 p, [$1], $2;\n\t"
            "selp.u32 $0, 1, 0, p;\n\t}", "=r,r,r,~{memory}", has_side_effects=True,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip
    return done != cutlass.Int32(0)


@dsl_user_op
def _try_wait_cluster(mbar, parity, *, loc=None, ip=None):
    """mbarrier.try_wait.parity.acquire.cluster on a barrier of this CTA (``mbar``, its shared-memory pointer): as
    ``_test_wait_cluster``, with try_wait's bounded suspend."""
    done = cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [mbar.toint(loc=loc, ip=ip).ir_value(), cutlass.Int32(parity).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [$1], $2;\n\t"
            "selp.u32 $0, 1, 0, p;\n\t}", "=r,r,r,~{memory}", has_side_effects=True,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip
    return done != cutlass.Int32(0)


@dsl_user_op
def _st_async_f32(dst, value, mbar, *, loc=None, ip=None):
    """st.async of one fp32 to a shared::cluster address, completing ``mbar`` (a shared::cluster address) by 4 bytes."""
    _llvm.inline_asm(
        None,
        [cutlass.Int32(dst).ir_value(loc=loc, ip=ip), cutlass.Float32(value).ir_value(loc=loc, ip=ip),
         cutlass.Int32(mbar).ir_value(loc=loc, ip=ip)],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [$0], $1, [$2];", "r,f,r", has_side_effects=True,
        is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


def _lo(word):
    return (word << cutlass.Int32(16)).bitcast(cutlass.Float32)


def _hi(word):
    return (word & cutlass.Int32(-65536)).bitcast(cutlass.Float32)


def _bf16(x):
    return cutlass.Float32(cutlass.BFloat16(x))


def _butterfly(value):
    for offset in [16, 8, 4, 2, 1]:
        value += cute.arch.shuffle_sync_bfly(value, offset=offset, mask=-1, mask_and_clamp=31)
    return value


def _store8(dst, words, idx):
    """The 8 bf16 of four int32 words, as fp32, into dst[idx : idx + 8]."""
    dst.store(
        (_lo(cutlass.Int32(words[0])), _hi(cutlass.Int32(words[0])), _lo(cutlass.Int32(words[1])),
         _hi(cutlass.Int32(words[1]))),
        idx=idx, alignment=16,
    )  # fmt: skip
    dst.store(
        (_lo(cutlass.Int32(words[2])), _hi(cutlass.Int32(words[2])), _lo(cutlass.Int32(words[3])),
         _hi(cutlass.Int32(words[3]))),
        idx=idx + 4, alignment=16,
    )  # fmt: skip


def _qk_pre(tk, lane, s_uq, s_wq, s_uk, s_wk, s_q, s_k, scale):
    """q and k of verify token ``tk`` from the conv history in shared memory: conv4 + SiLU, then the L2 norm (q also
    scaled), lane l keys 32 i + l (the arithmetic of kda_mtp_decode's pre-compute warps). Traced inline: its Python
    loops unroll."""
    pq = [cutlass.Float32(0.0)] * VEC
    for i in range(VEC):
        c = i * 32 + lane
        conv = cutlass.Float32(0.0)
        for w in range(CONV_W - 1):
            conv += s_uq.load(idx=(tk + w) * K + c) * s_wq.load(idx=w * K + c)
        conv += s_uq.load(idx=(tk + CONV_W - 1) * K + c) * s_wq.load(idx=(CONV_W - 1) * K + c)
        e = cute.math.exp(-conv, fastmath=True)
        pq[i] = conv * cute.arch.rcp_approx(cutlass.Float32(1.0) + e)
    sum_q = cutlass.Float32(0.0)
    for i in range(VEC):
        sum_q += pq[i] * pq[i]
    sum_q = _butterfly(sum_q)
    rnorm_q = cute.math.rsqrt(sum_q + 1e-06, fastmath=True) * scale
    for i in range(VEC):
        s_q.store(pq[i] * rnorm_q, idx=tk * K + i * 32 + lane)
    pk = [cutlass.Float32(0.0)] * VEC
    for i in range(VEC):
        c = i * 32 + lane
        conv = s_uk.load(idx=tk * K + c) * s_wk.load(idx=c)
        for w in range(1, CONV_W - 1):
            conv += s_uk.load(idx=(tk + w) * K + c) * s_wk.load(idx=w * K + c)
        conv += s_uk.load(idx=(tk + CONV_W - 1) * K + c) * s_wk.load(idx=(CONV_W - 1) * K + c)
        pk[i] = conv * cute.arch.rcp_approx(
            cutlass.Float32(1.0) + cute.math.exp(-conv, fastmath=True)
        )
    sum_k = cutlass.Float32(0.0)
    for i in range(VEC):
        sum_k += pk[i] * pk[i]
    sum_k = _butterfly(sum_k)
    rnorm_k = cute.math.rsqrt(sum_k + 1e-06, fastmath=True)
    for i in range(VEC):
        s_k.store(pk[i] * rnorm_k, idx=tk * K + i * 32 + lane)


@cute.kernel
def k3_kda_verify_kernel(
    tma_wfb: cutlass.GridConstant[
        cuda.TensorMap
    ],  # W_fb [H * K, K] bf16, the head's 128 rows by one call
    tma_fa: cutlass.GridConstant[
        cuda.TensorMap
    ],  # the projection rows' f_a columns, box 8 tokens x 64
    proj: cutlass.Array,  # int32 words of the fused projection rows, bf16 [T, 2 * proj_words]
    g_ext: cutlass.Array,  # int32 words of bf16 [T, H * K], the unfused f_b output (FOLD_FB False only)
    w_q: cutlass.Array,  # fp32 [H * K, CONV_W]
    w_k: cutlass.Array,
    w_v: cutlass.Array,
    a_log: cutlass.Array,  # fp32 [H]
    dt_bias: cutlass.Array,  # fp32 [H * K]
    onorm_w: cutlass.Array,  # fp32 [V]
    cs_q: cutlass.Array,  # fp32 [pool][S][H * K]: raw conv inputs, column s = position s - 2 from the golden token
    cs_k: cutlass.Array,
    cs_v: cutlass.Array,
    ssm: cutlass.Array,  # fp32 [pool][H][V][K] at slot stride ssm_stride: the state after the last golden token
    state_tok: cutlass.Array,  # fp32 [pool][3][NUM_SPEC][H][K]: the records of the last round's drafts
    slots: cutlass.Array,  # int32 [N]
    pending: cutlass.Array,  # int32 [pool]: drafts the sampler accepted last round
    out: cutlass.Array,  # bf16 [T][H][V]
    proj_words: cutlass.Int32,
    ssm_stride: cutlass.Int64,  # fp32 elements between the pool's slots (the manager interleaves conv states)
    H: cutlass.Constexpr[int],
    NUM_SPEC: cutlass.Constexpr[int],
    lower_bound: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    eps: cutlass.Constexpr[float],
    FOLD_FB: cutlass.Constexpr[bool],
    USE_PDL: cutlass.Constexpr[bool],
):
    NT = NUM_SPEC + 1  # verify tokens per request
    S = CONV_W - 1 + NUM_SPEC  # conv-cache columns
    HK = H * K
    ROWS_U = (
        CONV_W - 1 + NT
    )  # raw conv inputs by position: rows 0..2 before token 0, then the tokens
    # The fused projection row [q | k | v | onorm gate | f_a | b | pad], in int32 words (bf16 pairs).
    Q_W = 0
    K_W = HK // 2
    V_W = HK
    OG_W = 3 * HK // 2
    FA_W = 2 * HK
    B_COL = 4 * HK + K  # bf16 column
    assert NT <= WARPS and NT % 2 == 0

    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx % 32
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    h, n, vs = cute.arch.block_idx()
    v0 = vs * V_CTA
    ch0 = h * K
    row0 = n * NT

    smem_a = cutlass.Array(
        cutlass.BFloat16, CTA_M * K, space=SMEM, alignment=1024
    )  # W_fb rows, 128B-swizzled
    smem_b = cutlass.Array(
        cutlass.BFloat16, MMA_N * K, space=SMEM, alignment=1024
    )  # f_a, 128B-swizzled
    w_full = cutlass.Array(cutlass.Int64, 1, space=SMEM, alignment=8)
    x_full = cutlass.Array(cutlass.Int64, 1, space=SMEM, alignment=8)
    acc_done = cutlass.Array(cutlass.Int64, 1, space=SMEM, alignment=8)
    ss_ready = cutlass.Array(
        cutlass.Int64, 1, space=SMEM, alignment=8
    )  # the peers' norm partials have landed
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=SMEM)
    s_uq = cutlass.Array(cutlass.Float32, ROWS_U * K, space=SMEM, alignment=16)
    s_uk = cutlass.Array(cutlass.Float32, ROWS_U * K, space=SMEM, alignment=16)
    s_uv = cutlass.Array(cutlass.Float32, ROWS_U * V_CTA, space=SMEM, alignment=16)
    s_wq = cutlass.Array(cutlass.Float32, CONV_W * K, space=SMEM, alignment=16)  # [tap][channel]
    s_wk = cutlass.Array(cutlass.Float32, CONV_W * K, space=SMEM, alignment=16)
    s_wv = cutlass.Array(cutlass.Float32, CONV_W * V_CTA, space=SMEM, alignment=16)
    s_dtb = cutlass.Array(cutlass.Float32, K, space=SMEM, alignment=16)
    s_onw = cutlass.Array(cutlass.Float32, V_CTA, space=SMEM, alignment=16)
    s_gr = cutlass.Array(
        cutlass.Float32, NT * K, space=SMEM, alignment=16
    )  # f_b output, bf16-rounded
    s_braw = cutlass.Array(cutlass.Float32, NT, space=SMEM, alignment=16)
    s_og = cutlass.Array(cutlass.Float32, NT * V_CTA, space=SMEM, alignment=16)
    s_q = cutlass.Array(cutlass.Float32, NT * K, space=SMEM, alignment=16)
    s_k = cutlass.Array(cutlass.Float32, NT * K, space=SMEM, alignment=16)
    s_dec = cutlass.Array(cutlass.Float32, NT * K, space=SMEM, alignment=16)  # exp(gate)
    s_kd = cutlass.Array(cutlass.Float32, NT * K, space=SMEM, alignment=16)  # exp(gate) * k
    s_bk = cutlass.Array(cutlass.Float32, NT * K, space=SMEM, alignment=16)  # beta * k
    s_beta = cutlass.Array(cutlass.Float32, NT, space=SMEM, alignment=16)
    s_v = cutlass.Array(cutlass.Float32, NT * V_CTA, space=SMEM, alignment=16)
    s_o = cutlass.Array(
        cutlass.Float32, NT * V_CTA, space=SMEM, alignment=16
    )  # bf16-rounded raw outputs
    s_ss = cutlass.Array(
        cutlass.Float32, V_SPLIT * NT, space=SMEM, alignment=16
    )  # [source CTA][token]
    s_rs = cutlass.Array(cutlass.Float32, NT, space=SMEM, alignment=16)
    r_st = cutlass.Array(cutlass.Float32, ROWS * VEC, space=RMEM)

    # ---- Prologue: nothing here depends on the predecessor grid.
    if cutlass.const_expr(USE_PDL):
        prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
    tma_ptr_w = tma_wfb.get_ptr()
    tma_ptr_x = tma_fa.get_ptr()
    if cutlass.const_expr(FOLD_FB):
        if warp == 0:
            prims.prefetch_tensormap(tma_ptr_w)
            prims.prefetch_tensormap(tma_ptr_x)
            if prims.elect_sync():
                prims.mbarrier_init(w_full, 1)
                prims.mbarrier_init(x_full, 1)
                prims.mbarrier_init(acc_done, 1)
        if warp == 2:
            prims.tcgen05_alloc(tmem_ptr_i32, TMEM_COLS)
            prims.tcgen05_relinquish_alloc_permit()
    if warp == 3:
        if prims.elect_sync():
            prims.mbarrier_init(ss_ready, 1)
            prims.mbarrier_arrive_expect_tx(ss_ready, V_SPLIT * NT * 4)
    prims.fence_mbarrier_init()
    # Cluster formation: the peers' shared memory is addressable from here on.
    prims.barrier_cluster_arrive_relaxed()
    prims.barrier_cluster_wait()
    prims.barrier_cta_sync(0)
    if cutlass.const_expr(FOLD_FB):
        if warp == 0:
            if prims.elect_sync():
                prims.mbarrier_arrive_expect_tx(w_full, CTA_M * K * BF16_BYTES)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    smem_a,
                    tma_ptr_w,
                    (cutlass.Int32(0), ch0, cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0)),
                    w_full,
                )

    slot = slots.load(idx=n)
    pend = pending.load(idx=slot)
    pend = cutlass.Int32(
        cutlass.select_(pend > cutlass.Int32(NUM_SPEC), cutlass.Int32(NUM_SPEC), pend)
    )
    exp_a = cute.math.exp(a_log.load(idx=h), fastmath=True)
    row_a = v0 + warp * ROWS
    # The warp owns both rows and lane l keys 32 i + l (the arithmetic of kda_mtp_decode).
    # The drafts' records, in each slot's region of state_tok: the row innovations vn [NUM_SPEC][H][V], then beta * k
    # and the decay [NUM_SPEC][H][K]. The state after the accepted drafts is the pool's (the golden token's) with each
    # accepted draft's update replayed, S = fma(decay, S, vn * (beta * k)): the recurrence's own arithmetic, so it is
    # bit-identical to the one the drafts reached. k3_kda_attn uses the same records.
    CT_VN = 0
    CT_WB = NUM_SPEC * H * V
    CT_WD = CT_WB + NUM_SPEC * H * K
    CT_SLOT = CT_WD + NUM_SPEC * H * K  # a slot's records
    # The slot's pool state and records from their first element: the slot offset in 64 bits, once.
    pool = ssm.subview(cutlass.Int64(slot) * ssm_stride)
    tok = state_tok.subview(cutlass.Int64(slot) * CT_SLOT)
    st_base = (h * V + row_a) * K
    for r in cutlass.range_constexpr(ROWS):
        for i in cutlass.range_constexpr(VEC):
            r_st.store(pool.load(idx=st_base + r * K + i * 32 + lane), idx=r * VEC + i)
    for t_acc in cutlass.range(pend, unroll=1):
        rec_acc = (t_acc * H + h) * V
        vns_acc = [tok.load(idx=rec_acc + CT_VN + row_a + r) for r in range(ROWS)]
        wbs_acc = [tok.load(idx=CT_WB + (t_acc * H + h) * K + i * 32 + lane) for i in range(VEC)]
        wds_acc = [tok.load(idx=CT_WD + (t_acc * H + h) * K + i * 32 + lane) for i in range(VEC)]
        sts_acc = [r_st.load(idx=j) for j in range(ROWS * VEC)]
        for r in cutlass.range_constexpr(ROWS):
            for _pi in cutlass.range_constexpr(VEC // 2):
                _p = _pi * 2
                vb0_acc, vb1_acc = cute.arch.mul_packed_f32x2(
                    (vns_acc[r], vns_acc[r]), (wbs_acc[_p], wbs_acc[_p + 1])
                )
                sts_acc[r * VEC + _p], sts_acc[r * VEC + _p + 1] = cute.arch.fma_packed_f32x2(
                    src_a=(wds_acc[_p], wds_acc[_p + 1]), src_b=(sts_acc[r * VEC + _p], sts_acc[r * VEC + _p + 1]),
                    src_c=(vb0_acc, vb1_acc),
                )  # fmt: skip
        for j in cutlass.range_constexpr(ROWS * VEC):
            r_st.store(sts_acc[j], idx=j)

    # Conv weights, transposed to [tap][channel]: q by threads 0..127, k by 128..255.
    if tidx < K:
        wq4 = w_q.load(idx=(ch0 + tidx) * CONV_W, vector_size=CONV_W, alignment=16)
        for w in cutlass.range_constexpr(CONV_W):
            s_wq.store(cutlass.Float32(wq4[w]), idx=w * K + tidx)
    else:
        ck_w = tidx - K
        wk4 = w_k.load(idx=(ch0 + ck_w) * CONV_W, vector_size=CONV_W, alignment=16)
        for w in cutlass.range_constexpr(CONV_W):
            s_wk.store(cutlass.Float32(wk4[w]), idx=w * K + ck_w)
    # Conv history (positions -3..-1 = columns P..P+2), the v weights, the output-norm weight, dt_bias.
    QH = 3 * K // 4  # float4 loads per q (or k) history
    VH = 3 * V_CTA // 4
    if tidx < QH:
        m_q = tidx // (K // 4)
        c4_q = (tidx % (K // 4)) * 4
        s_uq.store(cs_q.load(idx=(slot * S + pend + m_q) * HK + ch0 + c4_q, vector_size=4, alignment=16),
                   idx=m_q * K + c4_q, alignment=16)  # fmt: skip
    elif tidx < 2 * QH:
        m_k = (tidx - QH) // (K // 4)
        c4_k = ((tidx - QH) % (K // 4)) * 4
        s_uk.store(cs_k.load(idx=(slot * S + pend + m_k) * HK + ch0 + c4_k, vector_size=4, alignment=16),
                   idx=m_k * K + c4_k, alignment=16)  # fmt: skip
    elif tidx < 2 * QH + VH:
        m_v = (tidx - 2 * QH) // (V_CTA // 4)
        c4_v = ((tidx - 2 * QH) % (V_CTA // 4)) * 4
        s_uv.store(cs_v.load(idx=(slot * S + pend + m_v) * HK + ch0 + v0 + c4_v, vector_size=4, alignment=16),
                   idx=m_v * V_CTA + c4_v, alignment=16)  # fmt: skip
    elif tidx < 2 * QH + VH + V_CTA:
        cv_w = tidx - (2 * QH + VH)
        wv4 = w_v.load(idx=(ch0 + v0 + cv_w) * CONV_W, vector_size=CONV_W, alignment=16)
        for w in cutlass.range_constexpr(CONV_W):
            s_wv.store(cutlass.Float32(wv4[w]), idx=w * V_CTA + cv_w)
    elif tidx < 2 * QH + VH + 2 * V_CTA:
        cv_n = tidx - (2 * QH + VH + V_CTA)
        s_onw.store(onorm_w.load(idx=v0 + cv_n), idx=cv_n)
    if tidx < K // 4:
        s_dtb.store(
            dt_bias.load(idx=ch0 + tidx * 4, vector_size=4, alignment=16),
            idx=tidx * 4,
            alignment=16,
        )

    # ---- This step's projection rows.
    if cutlass.const_expr(USE_PDL):
        prims.griddepcontrol(prims.GridDepAction.WAIT)
    if cutlass.const_expr(FOLD_FB):
        if warp == 1:
            if prims.elect_sync():
                prims.mbarrier_arrive_expect_tx(x_full, MMA_N * K * BF16_BYTES)
                for half in cutlass.range_constexpr(TMA_COPY_ITERS):
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_b.subview(half * B_HALF_ELEMS), tma_ptr_x,
                        (cutlass.Int32(FA_W * 2 + half * TMA_K_BOX), row0), x_full,
                    )  # fmt: skip
    CH = K // CHUNK  # 16-byte chunks per 128-channel row
    for it in cutlass.range_constexpr((2 * NT * CH + THREADS - 1) // THREADS):
        item = tidx + it * THREADS
        if item < 2 * NT * CH:
            stream_qk = item // (NT * CH)
            t_qk = (item // CH) % NT
            j_qk = item % CH
            words_qk = proj.load(
                idx=(row0 + t_qk) * proj_words + Q_W + ch0 // 2 + stream_qk * (K_W - Q_W) + j_qk * (CHUNK // 2),
                vector_size=4, alignment=16,
            )  # fmt: skip
            if stream_qk == 0:
                _store8(s_uq, words_qk, (CONV_W - 1 + t_qk) * K + j_qk * CHUNK)
            else:
                _store8(s_uk, words_qk, (CONV_W - 1 + t_qk) * K + j_qk * CHUNK)
    VCH = V_CTA // CHUNK  # 16-byte chunks per 16-channel v (or gate) slice
    n_v = NT * VCH
    n_g = 0 if FOLD_FB else NT * CH
    for it in cutlass.range_constexpr((2 * n_v + NT + n_g + THREADS - 1) // THREADS):
        item = tidx + it * THREADS
        if item < n_v:
            t_v = item // VCH
            j_v = item % VCH
            words_v = proj.load(idx=(row0 + t_v) * proj_words + V_W + (ch0 + v0) // 2 + j_v * (CHUNK // 2),
                                vector_size=4, alignment=16)  # fmt: skip
            _store8(s_uv, words_v, (CONV_W - 1 + t_v) * V_CTA + j_v * CHUNK)
        elif item < 2 * n_v:
            t_og = (item - n_v) // VCH
            j_og = (item - n_v) % VCH
            words_og = proj.load(idx=(row0 + t_og) * proj_words + OG_W + (ch0 + v0) // 2 + j_og * (CHUNK // 2),
                                 vector_size=4, alignment=16)  # fmt: skip
            _store8(s_og, words_og, t_og * V_CTA + j_og * CHUNK)
        elif item < 2 * n_v + NT:
            t_b = item - 2 * n_v
            word_b = proj.load(idx=(row0 + t_b) * proj_words + (B_COL + h) // 2)
            b_in = _lo(word_b)
            if (B_COL + h) % 2 == 1:
                b_in = _hi(word_b)
            s_braw.store(b_in, idx=t_b)
        elif item < 2 * n_v + NT + n_g:
            gi = item - (2 * n_v + NT)
            t_g = gi // CH
            j_g = gi % CH
            words_g = g_ext.load(idx=(row0 + t_g) * (HK // 2) + ch0 // 2 + j_g * (CHUNK // 2), vector_size=4,
                                 alignment=16)  # fmt: skip
            _store8(s_gr, words_g, t_g * K + j_g * CHUNK)
    prims.barrier_cta_sync(0)

    # ---- f_b on the tensor cores (warp 2 issues, warps 4-7 read TMEM) beside the q/k pre-compute (warps 0-3).
    if cutlass.const_expr(FOLD_FB):
        if warp == 2:
            idesc = prims.Tcgen05InstrDesc.build(
                c_dtype=cutlass.Float32,
                a_dtype=cutlass.BFloat16,
                b_dtype=cutlass.BFloat16,
                n_dim=MMA_N,
                m_dim=CTA_M,
            )
            desc_a_base = prims.Tcgen05SmemDesc.build(
                start_address=smem_a, leading_byte_offset=LEADING, stride_byte_offset=STRIDE,
                layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
            )  # fmt: skip
            desc_b_base = prims.Tcgen05SmemDesc.build(
                start_address=smem_b, leading_byte_offset=LEADING, stride_byte_offset=STRIDE,
                layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
            )  # fmt: skip
            tmem_acc = cutlass.inttoptr(tmem_ptr_i32.load(), 6, cutlass.Int32)
            while not cute.arch.mbarrier_try_wait(w_full.data_ptr(), 0):
                pass
            while not cute.arch.mbarrier_try_wait(x_full.data_ptr(), 0):
                pass
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            for kb in cutlass.range_constexpr(TMA_COPY_ITERS * K_BLOCKS_PER_HALF):
                box = kb // K_BLOCKS_PER_HALF
                within = kb % K_BLOCKS_PER_HALF
                if prims.elect_sync():
                    prims.tcgen05_mma(
                        prims.Tcgen05MMAKind.F16, prims.CTAGroup.CTA_1, tmem_acc,
                        desc_a_base + (box * A_BOX + within * STEP), desc_b_base + (box * B_BOX + within * STEP),
                        idesc, kb != 0,
                    )  # fmt: skip
            if prims.elect_sync():
                prims.tcgen05_commit(acc_done)
    if warp < 4:
        # q and k of tokens 2w and 2w + 1 (the arithmetic of kda_mtp_decode's pre-compute warps); NT not a multiple of 4
        # rounds the tokens per warp up and the warps past NT idle.
        qk_per_warp = (NT + 3) // 4
        for sub in cutlass.range_constexpr(qk_per_warp):
            tk = warp * qk_per_warp + sub
            if cutlass.const_expr(NT % 4 == 0):
                _qk_pre(tk, lane, s_uq, s_wq, s_uk, s_wk, s_q, s_k, scale)
            else:
                if tk < NT:
                    _qk_pre(tk, lane, s_uq, s_wq, s_uk, s_wk, s_q, s_k, scale)
    else:
        if cutlass.const_expr(FOLD_FB):
            while not cute.arch.mbarrier_try_wait(acc_done.data_ptr(), 0):
                pass
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            acc = prims.tcgen05_ld(
                "32x32b", cutlass.inttoptr(tmem_ptr_i32.load(), 6, cutlass.Float32), num=MMA_N
            )
            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
            c_fb = (warp - 4) * 32 + lane
            for t_fb in cutlass.range_constexpr(NT):
                s_gr.store(_bf16(cutlass.Float32(acc[t_fb])), idx=t_fb * K + c_fb)
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            prims.barrier_cta_sync(1, thread_count=128)
            if warp == 4:
                prims.tcgen05_dealloc(
                    cutlass.inttoptr(tmem_ptr_i32.load(), 6, cutlass.Int32), TMEM_COLS
                )
        # v (this CTA's 16 channels) and beta, one (token, channel) per thread of warps 4-7.
        item_v = tidx - 128
        if item_v < NT * V_CTA:
            t_cv = item_v // V_CTA
            c_cv = item_v % V_CTA
            conv_v = cutlass.Float32(0.0)
            for w in cutlass.range_constexpr(CONV_W):
                conv_v += s_uv.load(idx=(t_cv + w) * V_CTA + c_cv) * s_wv.load(idx=w * V_CTA + c_cv)
            conv_v = conv_v * cute.arch.rcp_approx(
                cutlass.Float32(1.0) + cute.math.exp(-conv_v, fastmath=True)
            )
            s_v.store(conv_v, idx=t_cv * V_CTA + c_cv)
        if item_v < NT:
            b_pre = s_braw.load(idx=item_v)
            s_beta.store(
                cute.arch.rcp_approx(cutlass.Float32(1.0) + cute.math.exp(-b_pre, fastmath=True)),
                idx=item_v,
            )
    prims.barrier_cta_sync(0)

    # ---- The recurrence's per-token operands: warp t, token t: decay = exp(gate), decay * k, beta * k.
    if warp < NT:
        tg = warp
        r_beta_g = s_beta.load(idx=tg)
        for i_pair in cutlass.range_constexpr(VEC // 2):
            dks = []
            for i in (i_pair * 2, i_pair * 2 + 1):
                c = i * 32 + lane
                g_raw = s_gr.load(idx=tg * K + c) + s_dtb.load(idx=c)
                x = exp_a * g_raw
                sig = cute.arch.rcp_approx(cutlass.Float32(1.0) + cute.math.exp(-x, fastmath=True))
                dks.append(
                    (cute.math.exp(lower_bound * sig, fastmath=True), s_k.load(idx=tg * K + c), c)
                )
            (d0, k0v, c0), (d1, k1v, c1) = dks
            bk0, bk1 = cute.arch.mul_packed_f32x2((r_beta_g, r_beta_g), (k0v, k1v))
            kd0, kd1 = cute.arch.mul_packed_f32x2((d0, d1), (k0v, k1v))
            s_dec.store(d0, idx=tg * K + c0)
            s_dec.store(d1, idx=tg * K + c1)
            s_kd.store(kd0, idx=tg * K + c0)
            s_kd.store(kd1, idx=tg * K + c1)
            s_bk.store(bk0, idx=tg * K + c0)
            s_bk.store(bk1, idx=tg * K + c1)
    prims.barrier_cta_sync(0)

    # ---- Recurrence over the verify tokens (register-resident state).
    st = [r_st.load(idx=j) for j in range(ROWS * VEC)]
    outs = []
    for tr in cutlass.range_constexpr(NT):
        r_v_val = s_v.load(idx=tr * V_CTA + warp * ROWS + lane % ROWS)
        r_q = [cutlass.Float32(0.0)] * VEC
        r_k = [cutlass.Float32(0.0)] * VEC
        r_decay = [cutlass.Float32(0.0)] * VEC
        r_bk = [cutlass.Float32(0.0)] * VEC
        for i in cutlass.range_constexpr(VEC):
            ki = i * 32 + lane
            r_q[i] = s_q.load(idx=tr * K + ki)
            r_k[i] = s_kd.load(idx=tr * K + ki)
            r_bk[i] = s_bk.load(idx=tr * K + ki)
            r_decay[i] = s_dec.load(idx=tr * K + ki)
        ra = 0
        rb = 1
        r_va = cute.arch.shuffle_sync(r_v_val, ra)
        r_vb = cute.arch.shuffle_sync(r_v_val, rb)
        shk_a1 = cutlass.Float32(0.0)
        shk_a2 = cutlass.Float32(0.0)
        shk_b1 = cutlass.Float32(0.0)
        shk_b2 = cutlass.Float32(0.0)
        for _pi in cutlass.range_constexpr(VEC // 2):
            _p = _pi * 2
            shk_a1, shk_a2 = cute.arch.fma_packed_f32x2(
                src_a=(st[ra * VEC + _p], st[ra * VEC + _p + 1]),
                src_b=(r_k[_p], r_k[_p + 1]),
                src_c=(shk_a1, shk_a2),
            )
            shk_b1, shk_b2 = cute.arch.fma_packed_f32x2(
                src_a=(st[rb * VEC + _p], st[rb * VEC + _p + 1]),
                src_b=(r_k[_p], r_k[_p + 1]),
                src_c=(shk_b1, shk_b2),
            )
        shk_a = shk_a1 + shk_a2
        shk_b = shk_b1 + shk_b2
        for offset in [16, 8, 4, 2, 1]:
            shk_a += cute.arch.shuffle_sync_bfly(shk_a, offset=offset, mask=-1, mask_and_clamp=31)
            shk_b += cute.arch.shuffle_sync_bfly(shk_b, offset=offset, mask=-1, mask_and_clamp=31)
        vn_a = r_va - shk_a
        vn_b = r_vb - shk_b
        shq_a1 = cutlass.Float32(0.0)
        shq_a2 = cutlass.Float32(0.0)
        shq_b1 = cutlass.Float32(0.0)
        shq_b2 = cutlass.Float32(0.0)
        for _pi in cutlass.range_constexpr(VEC // 2):
            _p = _pi * 2
            vnbk_a0, vnbk_a1 = cute.arch.mul_packed_f32x2((vn_a, vn_a), (r_bk[_p], r_bk[_p + 1]))
            vnbk_b0, vnbk_b1 = cute.arch.mul_packed_f32x2((vn_b, vn_b), (r_bk[_p], r_bk[_p + 1]))
            st[ra * VEC + _p], st[ra * VEC + _p + 1] = cute.arch.fma_packed_f32x2(
                src_a=(r_decay[_p], r_decay[_p + 1]),
                src_b=(st[ra * VEC + _p], st[ra * VEC + _p + 1]),
                src_c=(vnbk_a0, vnbk_a1),
            )
            st[rb * VEC + _p], st[rb * VEC + _p + 1] = cute.arch.fma_packed_f32x2(
                src_a=(r_decay[_p], r_decay[_p + 1]),
                src_b=(st[rb * VEC + _p], st[rb * VEC + _p + 1]),
                src_c=(vnbk_b0, vnbk_b1),
            )
            shq_a1, shq_a2 = cute.arch.fma_packed_f32x2(
                src_a=(st[ra * VEC + _p], st[ra * VEC + _p + 1]),
                src_b=(r_q[_p], r_q[_p + 1]),
                src_c=(shq_a1, shq_a2),
            )
            shq_b1, shq_b2 = cute.arch.fma_packed_f32x2(
                src_a=(st[rb * VEC + _p], st[rb * VEC + _p + 1]),
                src_b=(r_q[_p], r_q[_p + 1]),
                src_c=(shq_b1, shq_b2),
            )
        # The q-dot feeds only the output, not the next token: its reduction waits until after the loop.
        outs.append([shq_a1 + shq_a2, shq_b1 + shq_b2])
        # The golden token's state to the pool; a draft's record: the warp's row innovations (lane r, row r).
        if cutlass.const_expr(tr == 0):
            for r in cutlass.range_constexpr(ROWS):
                for i in cutlass.range_constexpr(VEC):
                    pool.store(st[r * VEC + i], idx=st_base + r * K + i * 32 + lane)
        else:
            if lane < ROWS:
                tok.store(
                    cutlass.Float32(cutlass.select_(lane == 0, vn_a, vn_b)),
                    idx=CT_VN + ((tr - 1) * H + h) * V + row_a + lane,
                )

    for offset in [16, 8, 4, 2, 1]:
        for tr in cutlass.range_constexpr(NT):
            for r in cutlass.range_constexpr(ROWS):
                outs[tr][r] += cute.arch.shuffle_sync_bfly(
                    outs[tr][r], offset=offset, mask=-1, mask_and_clamp=31
                )
    if lane == 0:
        for tr in cutlass.range_constexpr(NT):
            s_o.store(_bf16(outs[tr][0]), idx=tr * V_CTA + warp * ROWS + 0)
            s_o.store(_bf16(outs[tr][1]), idx=tr * V_CTA + warp * ROWS + 1)

    # ---- Output gated RMSNorm over V: per-token partial sums of squares to every CTA of the cluster.
    prims.barrier_cta_sync(0)
    if tidx < NT * V_CTA:
        t_out = tidx // V_CTA
        x_o = s_o.load(idx=tidx)
        ss = x_o * x_o
        for off_ss in [V_CTA >> d for d in range(1, V_CTA.bit_length()) if V_CTA >> d > 0]:
            ss = ss + cute.arch.shuffle_sync_bfly(ss, offset=off_ss, mask=-1, mask_and_clamp=31)
        if tidx % V_CTA == 0:
            # st.async into every CTA of the cluster (this one too), completing on its mailbox barrier: no cluster
            # barrier, so nothing waits for this CTA's state stores to drain.
            for r in cutlass.range_constexpr(V_SPLIT):
                _st_async_f32(
                    _mapa_u32(s_ss.subview(vs * NT + t_out).data_ptr(), r),
                    ss,
                    _mapa_u32(ss_ready.data_ptr(), r),
                )
    # Every peer's partials are in this CTA's shared memory once its mailbox completes. Each peer sends only after
    # its recurrence, i.e. after it consumed its reads of the q/k conv history, so CTA 0's rewrite below cannot race
    # them; and no CTA exits before its own mailbox has received every peer's partials.
    while not _try_wait_cluster(ss_ready.data_ptr(), 0):
        pass
    if tidx < NT:
        total = cutlass.Float32(0.0)
        for r in cutlass.range_constexpr(V_SPLIT):
            total = total + s_ss.load(idx=r * NT + tidx)
        s_rs.store(cute.math.rsqrt(total / V + eps), idx=tidx)
    prims.barrier_cta_sync(0)
    if tidx < NT * V_CTA:
        t_y = tidx // V_CTA
        v_y = tidx % V_CTA
        z = s_og.load(idx=tidx)
        gate = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.math.exp(-z, fastmath=True))
        y = s_o.load(idx=tidx) * s_rs.load(idx=t_y) * s_onw.load(idx=v_y) * gate
        out.store(cutlass.BFloat16(y), idx=((row0 + t_y) * H + h) * V + v0 + v_y)

    # ---- The drafts' per-key records (beta * k and the decay), one CTA per head, after every peer's norm partials
    # have arrived: each peer sends them after its recurrence, i.e. after its prologue read the last round's records.
    if vs == 0:
        for it_r in cutlass.range_constexpr((NUM_SPEC * K + THREADS - 1) // THREADS):
            item_r = tidx + it_r * THREADS
            if item_r < NUM_SPEC * K:
                t_r = item_r // K
                key_r = item_r % K
                tok.store(
                    s_bk.load(idx=(t_r + 1) * K + key_r),
                    idx=CT_WB + (t_r * H + h) * K + key_r,
                )
                tok.store(
                    s_dec.load(idx=(t_r + 1) * K + key_r),
                    idx=CT_WD + (t_r * H + h) * K + key_r,
                )

    # ---- Conv caches for the next round: column s = position s - 2 = row s + 1 of the position-indexed inputs.
    if vs == 0:
        if tidx < K:
            for s in cutlass.range_constexpr(S):
                cs_q.store(s_uq.load(idx=(s + 1) * K + tidx), idx=(slot * S + s) * HK + ch0 + tidx)
        else:
            ck_c = tidx - K
            for s in cutlass.range_constexpr(S):
                cs_k.store(s_uk.load(idx=(s + 1) * K + ck_c), idx=(slot * S + s) * HK + ch0 + ck_c)
    for it in cutlass.range_constexpr((S * V_CTA + THREADS - 1) // THREADS):
        item = tidx + it * THREADS
        if item < S * V_CTA:
            s_col = item // V_CTA
            v_col = item % V_CTA
            cs_v.store(
                s_uv.load(idx=(s_col + 1) * V_CTA + v_col),
                idx=(slot * S + s_col) * HK + ch0 + v0 + v_col,
            )


@cute.jit
def k3_kda_verify(
    w_fb: cute.Tensor,  # bf16 [H * K, K]: the f_b weight (out, in)
    proj: cute.Tensor,  # int32 words of the fused projection [T, proj_words]
    g_ext: cute.Tensor,  # int32 words of the unfused f_b output [T, H * K / 2] (FOLD_FB False)
    w_q: cute.Tensor,
    w_k: cute.Tensor,
    w_v: cute.Tensor,
    a_log: cute.Tensor,
    dt_bias: cute.Tensor,
    onorm_w: cute.Tensor,
    cs_q: cute.Tensor,
    cs_k: cute.Tensor,
    cs_v: cute.Tensor,
    ssm: cute.Tensor,
    state_tok: cute.Tensor,
    slots: cute.Tensor,
    pending: cute.Tensor,
    out: cute.Tensor,
    proj_words: cutlass.Int32,
    n_req: cutlass.Int32,
    ssm_stride: cutlass.Int64,
    H: cutlass.Constexpr[int],
    NUM_SPEC: cutlass.Constexpr[int],
    lower_bound: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    eps: cutlass.Constexpr[float],
    FOLD_FB: cutlass.Constexpr[bool],
    USE_PDL: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    """One launch of the verify kernel for ``n_req`` requests of 1 + NUM_SPEC tokens each."""
    # W_fb [H K, K] as five TMA dimensions (64-element column chunk, row, chunk index, 1, 1): one call lands the
    # head's 128 rows, both 128-byte-swizzled halves; strides in 16-byte units.
    tma_wfb = cuda.create_tensor_map_tiled(
        global_address=w_fb.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=[TMA_K_BOX, H * K, K // TMA_K_BOX, 1, 1],
        global_strides=[(K * BF16_BYTES) // 16, (TMA_K_BOX * BF16_BYTES) // 16, (H * K * K * BF16_BYTES) // 16,
                        (H * K * K * BF16_BYTES) // 16],
        box_dims=[TMA_K_BOX, CTA_M, TMA_COPY_ITERS, 1, 1],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )  # fmt: skip
    # The projection rows as bf16 [T, 2 proj_words]: an 8-token x 64-column box of f_a per call.
    tma_fa = cuda.create_tensor_map_tiled(
        global_address=proj.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=[proj_words * 2, n_req * (NUM_SPEC + 1)],
        global_strides=[(proj_words * 4) // 16],
        box_dims=[TMA_K_BOX, MMA_N],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    k3_kda_verify_kernel(
        tma_wfb, tma_fa, proj, g_ext, w_q, w_k, w_v, a_log, dt_bias, onorm_w, cs_q, cs_k, cs_v, ssm, state_tok, slots,
        pending, out, proj_words, ssm_stride, H, NUM_SPEC, lower_bound, scale, eps, FOLD_FB, USE_PDL,
    ).launch(
        grid=(H, n_req, V_SPLIT), block=(THREADS, 1, 1), cluster=(1, 1, V_SPLIT), stream=stream, use_pdl=USE_PDL,
    )  # fmt: skip
