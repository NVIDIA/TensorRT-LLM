# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""KDA multi-token speculative verify as an FP32 CUDA-core recurrence (packed rows, fused gated RMSNorm).

The TEP target runs it for every fused verify outside torch.compile that :func:`supported` admits. The launcher
is :func:`k31_kda_verify_rec` (end of this file); :func:`supported` also gates it to SM100 / SM103 / SM107.

Same contract and replay semantics as ``kda_mtp_decode_v2`` (see ``kda_mtp_decode``): per request the accepted
drafts of the previous round are replayed from the k / g / v / beta caches, then the ``1 + NUM_SPEC`` new tokens
run; the FP32 state and the conv windows are committed in place after the first new (golden) token and the new
drafts are cached for the next round (same cache dtypes and layouts). The recurrence is the delta rule token by
token in FP32 instead of the 3xTF32 chunk (WY) form; outputs and committed state agree with the tensor-core kernels
to FP32 rounding, not bitwise.

Work split. A (head, request) tile runs on a cluster of ``C`` CTAs (1, 2, 4 or 8) of ``NTHR`` threads; CTA ``r``
owns state rows (value channels) ``[r * 128 / C, +128 / C)``, which evolve independently of the other rows. ``LPR``
lanes share a row (``128 / LPR`` keys each, in 16-byte chunks), ``NTHR / LPR`` rows run at once, so every thread
keeps ``RHO`` rows of the state in registers for the whole call. Every CTA computes the key side (q / k / gate of
all 128 keys, one key per thread) itself; the output RMSNorm exchanges one partial sum of squares per token through
distributed shared memory. The two dot products of a token are reduced with a reduce-scatter over the lanes of a
row (each lane ends with one of the ``2 RHO`` sums), the dots and the rank-1 update run as packed ``f32x2`` FMAs.
Launches of up to 32 rows per CTA (``C >= 4``) run :func:`kda_verify_rec_ws_kernel`, the same arithmetic with
specialized warps (see its section).

Deferred decay. With ``P_t = exp(G_t)`` (``G_t`` the cumulative gate since the segment start) the kernel keeps
``S~_t = S_t / P_t`` (per key), for which a token costs two dot products and one rank-1 update and no decay pass:

    u = S~_{t-1}^T (P_t k_t),   w = S~_{t-1}^T (P_t q_t),   v' = beta_t (v_t - u),
    o_t = w + (k_t . q_t) v',   S~_t = S~_{t-1} + (k_t / P_t) v'^T.

Segments: the replayed drafts plus the golden token (the commit stores ``P_t S~_t``), then the remaining new
tokens from the committed state; a segment holds at most 8 tokens, i.e. ``|G| <= 8 |lower_bound|``.

PDL: launched as a programmatic dependent, the CTA loads its state rows, the replay entries, the conv windows and
the weights before ``griddepcontrol.wait`` and only the new-token rows after it, and releases the next kernel right
after the wait (a dependent must ``griddepcontrol.wait`` before it reads the output).

The committed state is closer to an FP64 reference than the 3xTF32 kernels'.
"""

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import Int64
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.ssm.kda_mtp_decode_v2 import (
    gaddr,
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
    ld_global_b32x4,
    ld_shared_f32x4,
    mbarrier_wait_cluster,
    saddr,
    st_async_f32,
    st_global_b32x4,
)

K_DIM = 128
V_DIM = 128
CONV_WIDTH = 4
#: largest NUM_SPEC (a replay segment plus the golden token holds at most 8 tokens)
MAX_NUM_SPEC = 7


@dsl_user_op
def _asm(asm, constraints, operands, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        operands,
        asm,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cp_async(dst_smem, src_gmem, nbytes: int, *, loc=None, ip=None):
    """``nbytes`` (4, 8 or 16) from global to shared memory, asynchronously (``cp.async.ca``)."""
    _asm(
        f"cp.async.ca.shared.global [$0], [$1], {nbytes};",
        "r,l",
        [
            cutlass.Int32(dst_smem).ir_value(loc=loc, ip=ip),
            Int64(src_gmem).ir_value(loc=loc, ip=ip),
        ],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cp_async_commit(*, loc=None, ip=None):
    _asm("cp.async.commit_group;", "", [], loc=loc, ip=ip)


@dsl_user_op
def cp_async_wait_all(*, loc=None, ip=None):
    _asm("cp.async.wait_all;", "", [], loc=loc, ip=ip)


@dsl_user_op
def selp_f32(pred, a, b, *, loc=None, ip=None):
    """``a`` where the int32 ``pred`` is nonzero, else ``b`` (no branch)."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [
                cutlass.Float32(a).ir_value(loc=loc, ip=ip),
                cutlass.Float32(b).ir_value(loc=loc, ip=ip),
                cutlass.Int32(pred).ir_value(loc=loc, ip=ip),
            ],
            "{\n.reg .pred p;\nsetp.ne.s32 p, $3, 0;\nselp.f32 $0, $1, $2, p;\n}",
            "=f,f,f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _f32x2_op(asm, ins, *, loc, ip):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [cutlass.Float32(x).ir_value(loc=loc, ip=ip) for x in ins],
        asm,
        "=f,=f," + ",".join(["f"] * len(ins)),
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return [cutlass.Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(2)]


@dsl_user_op
def ffma2(a0, a1, b0, b1, c0, c1, *, loc=None, ip=None):
    """``(a0 * b0 + c0, a1 * b1 + c1)`` with one packed ``fma.rn.f32x2`` (IEEE fma per lane)."""
    return _f32x2_op(
        "{\n.reg .b64 a, b, c, d;\nmov.b64 a, {$2, $3};\nmov.b64 b, {$4, $5};\nmov.b64 c, {$6, $7};\n"
        "fma.rn.f32x2 d, a, b, c;\nmov.b64 {$0, $1}, d;\n}",
        (a0, a1, b0, b1, c0, c1),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def fmul2(a0, a1, b0, b1, *, loc=None, ip=None):
    """``(a0 * b0, a1 * b1)`` with one packed ``mul.rn.f32x2``."""
    return _f32x2_op(
        "{\n.reg .b64 a, b, d;\nmov.b64 a, {$2, $3};\nmov.b64 b, {$4, $5};\nmul.rn.f32x2 d, a, b;\n"
        "mov.b64 {$0, $1}, d;\n}",
        (a0, a1, b0, b1),
        loc=loc,
        ip=ip,
    )


def _dot8(h, base, v):
    """Dot product of ``h[base : base + 8]`` and ``v[0 : 8]`` (FP32, packed pairs, four steps)."""
    s0, s1 = fmul2(h[base], h[base + 1], v[0], v[1])
    for e in range(2, 8, 2):
        s0, s1 = ffma2(h[base + e], h[base + e + 1], v[e], v[e + 1], s0, s1)
    return s0 + s1


@dsl_user_op
def st_global_b16(addr, value, *, loc=None, ip=None):
    """One bf16 (fp32 ``value`` rounded to nearest even)."""
    _asm(
        "{\n.reg .b16 h;\ncvt.rn.bf16.f32 h, $1;\nst.global.b16 [$0], h;\n}",
        "l,f",
        [Int64(addr).ir_value(loc=loc, ip=ip), cutlass.Float32(value).ir_value(loc=loc, ip=ip)],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_global_f32(addr, value, *, loc=None, ip=None):
    _asm(
        "st.global.f32 [$0], $1;",
        "l,f",
        [Int64(addr).ir_value(loc=loc, ip=ip), cutlass.Float32(value).ir_value(loc=loc, ip=ip)],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_shared_f32_if(addr, value, pred, *, loc=None, ip=None):
    """Store ``value`` to shared memory where the int32 ``pred`` is nonzero (predicated, no branch)."""
    _asm(
        "{\n.reg .pred p;\nsetp.ne.s32 p, $2, 0;\n@p st.shared.f32 [$0], $1;\n}",
        "r,f,r",
        [
            cutlass.Int32(addr).ir_value(loc=loc, ip=ip),
            cutlass.Float32(value).ir_value(loc=loc, ip=ip),
            cutlass.Int32(pred).ir_value(loc=loc, ip=ip),
        ],
        loc=loc,
        ip=ip,
    )


def _smem_view(base, byte_off, dtype, n):
    """``n`` elements of ``dtype`` at byte ``byte_off`` (a multiple of 16) of the FP32 smem buffer ``base``."""
    return cute.make_tensor(
        cute.recast_ptr(base.iterator + byte_off // 4, dtype=dtype), cute.make_layout((n,))
    )


def _cp_row4(dst, dst_off, src, coord):
    """cp.async of four consecutive elements of ``src`` at ``coord`` (8 B bf16 / 16 B fp32) to ``dst[dst_off]``."""
    cp_async(saddr(dst, dst_off), gaddr(src, coord), 4 * (src.element_type.width // 8))


def _sigmoid(x):
    return cute.arch.rcp_approx(cutlass.Float32(1.0) + cute.math.exp(-x, fastmath=True))


def _reduce_scatter(x, lane, offsets):
    """Sum each of the ``len(x)`` (a power of two) per-lane values over the lanes spanned by ``offsets`` (xor
    butterfly levels, in order). At a level with ``n > 1`` values left, lanes whose ``offset`` bit is set keep the
    upper half; once one value is left the remaining levels reduce it in full. Returns this lane's sum and the index
    (into ``x``) it belongs to."""
    n = len(x)
    idx = cutlass.Int32(0)
    for off in offsets:
        hi = lane & off
        if n > 1:
            half = n // 2
            nxt = []
            for i in range(half):
                send = selp_f32(hi, x[i], x[half + i])
                keep = selp_f32(hi, x[half + i], x[i])
                nxt.append(
                    keep + cute.arch.shuffle_sync_bfly(send, offset=off, mask=-1, mask_and_clamp=31)
                )
            x = nxt
            idx = idx + cutlass.Int32(hi != 0) * half
            n = half
        else:
            x = [x[0] + cute.arch.shuffle_sync_bfly(x[0], offset=off, mask=-1, mask_and_clamp=31)]
    return x[0], idx


def _prefetch(t, ctx, pre):
    """Issue the shared-memory loads of token ``t``'s operands into the prefetch registers ``pre``: ``P k`` / ``k / P``
    chunks, ``P q`` chunks of new-token row ``max(t - a, 0)``, beta, and the value of the lane's u-row."""
    (
        hs,
        sA,
        sC,
        sB,
        sV,
        sBeta,
        sKQ,
        sO,
        sPc,
        h0,
        slot,
        i_h,
        vbase,
        group,
        lr,
        a,
        R,
        RHO,
        LPR,
        NCH,
        vrow_u,
        GRP,
    ) = ctx
    pa, pc, pb, ps = pre
    jb = t - a
    jb = jb * cutlass.Int32(jb > 0)
    for j in range(NCH):
        col = 4 * (lr + LPR * j)
        x4 = ld_shared_f32x4(saddr(sA, t * K_DIM + col))
        c4 = ld_shared_f32x4(saddr(sC, t * K_DIM + col))
        b4 = ld_shared_f32x4(saddr(sB, jb * K_DIM + col))
        for e in range(4):
            pa[4 * j + e] = x4[e]
            pc[4 * j + e] = c4[e]
            pb[4 * j + e] = b4[e]
    ps[0] = sBeta[t]
    ps[1] = sV[t * R + vrow_u]


def _token(t, ctx, pre, new):
    """One token of the recurrence, the same code for every token (``t``: its row in the per-token key-side
    buffers; replayed drafts ``t < a`` have no output, the golden token ``t == a`` commits the state). Takes the
    operands from the prefetch registers and prefetches token ``t + 1``'s. Returns the commit flag."""
    (
        hs,
        sA,
        sC,
        sB,
        sV,
        sBeta,
        sKQ,
        sO,
        sPc,
        h0,
        slot,
        i_h,
        vbase,
        group,
        lr,
        a,
        R,
        RHO,
        LPR,
        NCH,
        vrow_u,
        GRP,
    ) = ctx
    pa, pc, pb, ps = pre
    KL = 4 * NCH  # keys per lane
    av = [pa[i] for i in range(KL)]
    cv = [pc[i] for i in range(KL)]
    bv = [pb[i] for i in range(KL)]
    beta_t = ps[0]
    v_u = ps[1]
    _prefetch(t + 1, ctx, pre)
    x = []
    for vec in (av, bv) if new else (av,):
        for p in range(RHO):
            parts = [_dot8(hs, KL * p + 8 * c, vec[8 * c : 8 * c + 8]) for c in range(KL // 8)]
            while len(parts) > 1:
                parts = [parts[2 * i] + parts[2 * i + 1] for i in range(len(parts) // 2)]
            x.append(parts[0])
    if not new:
        # replayed drafts have no output: zeros keep the lane layout of the new tokens
        x += [cutlass.Float32(0.0)] * RHO
    offs = [LPR >> (i + 1) for i in range(LPR.bit_length() - 1)]
    val, idx = _reduce_scatter(x, lr, offs)
    # u-lanes (idx < RHO) hold u of row idx, the lanes LPR / 2 above them w of the same row
    vn_lane = beta_t * (v_u - val)
    lane = group * LPR + lr
    base = lane - lr
    step = (LPR // 2) // RHO
    vn = [
        cute.arch.shuffle_sync(vn_lane, offset=base + step * p, mask=-1, mask_and_clamp=31)
        for p in range(RHO)
    ]
    vn_w = cute.arch.shuffle_sync(
        vn_lane, offset=lane - LPR // 2 * cutlass.Int32(lr >= LPR // 2), mask=-1, mask_and_clamp=31
    )
    if new:
        jb = t - a
        ov = cutlass.Float32(cutlass.BFloat16(val + sKQ[jb] * vn_w))
        is_w = cutlass.Int32(idx >= RHO)
        p_w = idx - RHO * is_w
        st_shared_f32_if(saddr(sO, jb * R + p_w * GRP + group), ov, is_w)
    for p in range(RHO):
        for e in range(0, KL, 2):
            hs[KL * p + e], hs[KL * p + e + 1] = ffma2(
                cv[e], cv[e + 1], vn[p], vn[p], hs[KL * p + e], hs[KL * p + e + 1]
            )


def _commit(ctx):
    """Scale the state rows by ``P`` of the golden token and store them (the committed state)."""
    (
        hs,
        sA,
        sC,
        sB,
        sV,
        sBeta,
        sKQ,
        sO,
        sPc,
        h0,
        slot,
        i_h,
        vbase,
        group,
        lr,
        a,
        R,
        RHO,
        LPR,
        NCH,
        vrow_u,
        GRP,
    ) = ctx
    KL = 4 * NCH
    for j in range(NCH):
        pcm = ld_shared_f32x4(saddr(sPc, 4 * (lr + LPR * j)))
        for p in range(RHO):
            for e in range(4):
                hs[KL * p + 4 * j + e] = hs[KL * p + 4 * j + e] * pcm[e]
    for p in range(RHO):
        for j in range(NCH):
            st_global_b32x4(
                gaddr(h0, (slot, i_h, vbase + p * GRP + group, 4 * (lr + LPR * j))),
                *[hs[KL * p + 4 * j + e].bitcast(cutlass.Int32) for e in range(4)],
            )


@cute.kernel
def kda_verify_rec_kernel(
    h0: cute.Tensor,
    x_q: cute.Tensor,
    x_k: cute.Tensor,
    x_v: cute.Tensor,
    w_q: cute.Tensor,
    w_k: cute.Tensor,
    w_v: cute.Tensor,
    cs_q: cute.Tensor,
    cs_k: cute.Tensor,
    cs_v: cute.Tensor,
    A_log: cute.Tensor,
    g: cute.Tensor,
    dt_bias: cute.Tensor,
    beta: cute.Tensor,
    onorm_g: cute.Tensor,
    onorm_weight: cute.Tensor,
    o: cute.Tensor,
    k_cache: cute.Tensor,
    g_cache: cute.Tensor,
    v_cache: cute.Tensor,
    beta_cache: cute.Tensor,
    ssm_state_indices: cute.Tensor,
    num_accepted_tokens: cute.Tensor,
    scale: cutlass.Constexpr[float],
    NUM_SPEC: cutlass.Constexpr[int],
    lower_bound: cutlass.Constexpr[float],
    onorm_eps: cutlass.Constexpr[float],
    C: cutlass.Constexpr[int],
    LPR: cutlass.Constexpr[int] = 16,
    NTHR: cutlass.Constexpr[int] = 128,
):
    """Grid ``(C, H, N)``, cluster ``(C, 1, 1)``. Operands (bf16 unless noted): ``h0`` FP32 ``[pool, H, V, K]``
    (V-first, strided slots); ``x_q`` / ``x_k`` / ``x_v`` / ``g`` / ``onorm_g`` ``[1, T, H * 128]`` row views of
    the projection outputs; ``w_*`` FP32 ``[H * 128, 4]``; ``cs_*`` ``[pool, H * 128, >= 3 + NUM_SPEC]``
    dim-contiguous; ``A_log`` FP32 ``[H]``; ``dt_bias`` FP32 ``[H * 128]``; ``beta`` ``[1, T, H]``;
    ``onorm_weight`` FP32 ``[V]``; ``o`` ``[1, T, H * V]``; ``k_cache`` / ``v_cache`` ``[pool, M, H * 128]``,
    ``g_cache`` FP32 ``[pool, M, H * 128]``, ``beta_cache`` ``[pool, M, H]``; ``ssm_state_indices`` int32 ``[N]``,
    ``num_accepted_tokens`` int32 ``[pool]``."""
    K = K_DIM
    V = V_DIM
    R = V // C  # state rows of this CTA
    GRP = NTHR // LPR  # state rows updated at once
    NW = NTHR // 32
    VOFF = K if NTHR >= 2 * K else 0  # first value-side thread of the precompute
    NCH = 32 // LPR  # 16-byte key chunks per lane and row
    RHO = R // GRP  # state rows per thread
    NT = NUM_SPEC + 1  # new tokens per request
    TMAX = 2 * NUM_SPEC + 1  # replayed drafts + new tokens
    SROWS = 3 + NUM_SPEC + NT  # raw conv staging rows: window, replay tail, new tokens
    VL = max(1, R // 32)  # own value rows per lane in warp-per-token loops
    NRED = (
        32  # block-reduced values per key thread: sum q^2, sum k^2, sum q k per new token (padded)
    )
    assert (
        NUM_SPEC <= MAX_NUM_SPEC
        and C in (1, 2, 4, 8)
        and RHO >= 1
        and 2 * RHO <= LPR
        and NTHR in (128, 256, 512)
        and 3 * NT <= NRED
        and LPR in (4, 8, 16)
    ), (
        "NUM_SPEC <= MAX_NUM_SPEC, C in (1, 2, 4, 8), NTHR in (128, 256, 512), LPR in (4, 8, 16), and "
        "1 <= RHO <= LPR / 2: every thread holds a state row and a row's reduce-scatter leaves each of its LPR "
        "lanes at most one of the 2 RHO dot-product sums"
    )

    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx % 32
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    rank, i_h, i_n = cute.arch.block_idx()
    slot32 = ssm_state_indices[i_n]
    hk = i_h * K
    vbase = rank * R
    hv = i_h * V + vbase
    k4 = 4 * lane
    v4 = 4 * lane  # first own value row of a lane's 4-row chunk (lanes < R / 4)

    # ---- shared memory. The raw staging rows (conv windows / tails / new rows, replay caches, raw gates) are only
    #      read before the precompute barrier, after which the per-token key-side rows (P k, k / P, P q) overwrite
    #      them.
    BF = 2  # bytes
    st_bytes = [
        2 * SROWS * K * BF,
        SROWS * R * BF,
        NUM_SPEC * K * BF,
        NUM_SPEC * K * 4,
        NUM_SPEC * R * BF,
        NT * K * BF,
    ]
    post_bytes = [TMAX * K * 4, TMAX * K * 4, NT * K * 4]
    u_floats = (max(sum(st_bytes), sum(post_bytes)) + 15) // 16 * 4
    smem = cutlass.utils.SmemAllocator()
    sU = smem.allocate_tensor(cutlass.Float32, cute.make_layout((u_floats,)), 16)

    off = 0
    sXQ = _smem_view(sU, off, cutlass.BFloat16, SROWS * K)
    off += SROWS * K * BF
    sXK = _smem_view(sU, off, cutlass.BFloat16, SROWS * K)
    off += SROWS * K * BF
    sXV = _smem_view(sU, off, cutlass.BFloat16, SROWS * R)
    off += SROWS * R * BF
    sKc = _smem_view(sU, off, cutlass.BFloat16, NUM_SPEC * K)
    off += NUM_SPEC * K * BF
    sGc = _smem_view(sU, off, cutlass.Float32, NUM_SPEC * K)
    off += NUM_SPEC * K * 4
    sVc = _smem_view(sU, off, cutlass.BFloat16, NUM_SPEC * R)
    off += NUM_SPEC * R * BF
    sGraw = _smem_view(sU, off, cutlass.BFloat16, NT * K)
    sA = _smem_view(sU, 0, cutlass.Float32, TMAX * K)
    sC = _smem_view(sU, TMAX * K * 4, cutlass.Float32, TMAX * K)
    sB = _smem_view(sU, 2 * TMAX * K * 4, cutlass.Float32, NT * K)
    sOG = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((NT * R,)), 16)
    sV = smem.allocate_tensor(cutlass.Float32, cute.make_layout((TMAX * R,)), 16)
    sW = smem.allocate_tensor(cutlass.Float32, cute.make_layout((2 * K * 4 + R * 4,)), 16)
    sDtb = smem.allocate_tensor(cutlass.Float32, cute.make_layout((K,)), 16)
    sOnW = smem.allocate_tensor(cutlass.Float32, cute.make_layout((R,)), 16)
    sPc = smem.allocate_tensor(cutlass.Float32, cute.make_layout((K,)), 16)
    sO = smem.allocate_tensor(cutlass.Float32, cute.make_layout((NT * R,)), 16)
    sRed = smem.allocate_tensor(cutlass.Float32, cute.make_layout((NRED * 4,)), 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, cute.make_layout((16,)), 16)
    sBraw = smem.allocate_tensor(cutlass.Float32, cute.make_layout((8,)), 16)
    sBc = smem.allocate_tensor(cutlass.Float32, cute.make_layout((8,)), 16)
    sKQ = smem.allocate_tensor(cutlass.Float32, cute.make_layout((8,)), 16)
    sPart = smem.allocate_tensor(cutlass.Float32, cute.make_layout((NT * C,)), 16)
    sAl = smem.allocate_tensor(cutlass.Float32, cute.make_layout((4,)), 16)
    mbar_part = smem.allocate_array(cutlass.Int64, 1)

    if tidx == 0:
        cute.arch.mbarrier_init(mbar_part, 1)
        cute.arch.mbarrier_arrive_and_expect_tx(mbar_part, NT * C * 4)
    cute.arch.mbarrier_init_fence()

    # ---- load phase 1 (independent of the projection GEMM): weights, slot-dependent pools and caches
    if tidx < K:
        cp_async(saddr(sW, 4 * tidx), gaddr(w_q, (hk + tidx, 0)), 16)
        cp_async(saddr(sW, 4 * K + 4 * tidx), gaddr(w_k, (hk + tidx, 0)), 16)
        sDtb[tidx] = cutlass.Float32(dt_bias[hk + tidx])
    if tidx < R:
        cp_async(saddr(sW, 8 * K + 4 * tidx), gaddr(w_v, (hv + tidx, 0)), 16)
        cp_async(saddr(sOnW, tidx), gaddr(onorm_weight, (vbase + tidx,)), 4)
    if tidx == 0:
        sAl[0] = cutlass.Float32(A_log[i_h])
    slot = Int64(slot32)
    commit_raw = num_accepted_tokens[slot]
    # conv windows (rows 0..2) and replay tails (rows 3 .. 3 + NUM_SPEC) of q / k / v, replay caches
    for r in cutlass.range_constexpr(3 + NUM_SPEC):
        if warp == r % NW:
            _cp_row4(sXQ, r * K + k4, cs_q, (slot, hk + k4, r))
            _cp_row4(sXK, r * K + k4, cs_k, (slot, hk + k4, r))
            if lane < R // 4:
                _cp_row4(sXV, r * R + v4, cs_v, (slot, hv + v4, r))
            if cutlass.const_expr(r >= 3):
                i = r - 3
                _cp_row4(sKc, i * K + k4, k_cache, (slot, i, hk + k4))
                _cp_row4(sGc, i * K + k4, g_cache, (slot, i, hk + k4))
                if lane < R // 4:
                    _cp_row4(sVc, i * R + v4, v_cache, (slot, i, hv + v4))
    if tidx < NUM_SPEC:
        sBc[tidx] = cutlass.Float32(beta_cache[slot, tidx, i_h])
    cp_async_commit()
    # this thread's state rows (value row vbase + p * GRP + group, keys 4 (lr + LPR j) .. + 3)
    group = tidx // LPR
    lr = tidx % LPR
    hs = cute.make_rmem_tensor(cute.make_layout((RHO * 4 * NCH,)), cutlass.Float32)
    if warp < NW:
        for p in cutlass.range_constexpr(RHO):
            for j in cutlass.range_constexpr(NCH):
                wv = ld_global_b32x4(
                    gaddr(h0, (slot, i_h, vbase + p * GRP + group, 4 * (lr + LPR * j)))
                )
                for e in cutlass.range_constexpr(4):
                    hs[4 * NCH * p + 4 * j + e] = wv[e].bitcast(cutlass.Float32)

    # ---- load phase 2: the new-token rows (the projection GEMM's outputs)
    griddepcontrol_wait()
    griddepcontrol_launch_dependents()
    for j in cutlass.range_constexpr(NT):
        if warp == (3 + NUM_SPEC + j) % NW:
            row = i_n * NT + j
            sr = 3 + NUM_SPEC + j
            _cp_row4(sXQ, sr * K + k4, x_q, (0, row, hk + k4))
            _cp_row4(sXK, sr * K + k4, x_k, (0, row, hk + k4))
            _cp_row4(sGraw, j * K + k4, g, (0, row, hk + k4))
            if lane < R // 4:
                _cp_row4(sXV, sr * R + v4, x_v, (0, row, hv + v4))
                _cp_row4(sOG, j * R + v4, onorm_g, (0, row, hv + v4))
    if tidx < NT:
        sBraw[tidx] = cutlass.Float32(beta[0, i_n * NT + tidx, i_h])
    cp_async_commit()
    commit_len = commit_raw
    if commit_len > NUM_SPEC:
        commit_len = cutlass.Int32(NUM_SPEC)
    a = commit_len
    cp_async_wait_all()
    cute.arch.barrier()
    if cutlass.const_expr(C > 1):
        # every replay entry / conv window this CTA reads is in smem now; the partial-sum mbarrier is armed
        cute.arch.cluster_arrive_relaxed()
    if warp < NW:
        # ---- precompute, one key per thread: conv + SiLU of q / k, gates of the new tokens (raw staging reads only)
        kx = tidx % K
        wq = ld_shared_f32x4(saddr(sW, 4 * kx))
        wk = ld_shared_f32x4(saddr(sW, 4 * K + 4 * kx))
        dtb = sDtb[kx]
        exp_a = cute.math.exp(sAl[0], fastmath=True)
        # raw sequence positions a + m: rows a + m (window / replay tail) for m < 3, the new-token rows after
        seqq = []
        seqk = []
        for m in cutlass.range_constexpr(NT + 3):
            row = (a + m) if m < 3 else (NUM_SPEC + m)
            seqq.append(cutlass.Float32(sXQ[row * K + kx]))
            seqk.append(cutlass.Float32(sXK[row * K + kx]))
        qv = []
        kv = []
        gv = []
        red = []
        for j in cutlass.range_constexpr(NT):
            rq = cutlass.Float32(0.0)
            for w in cutlass.range_constexpr(CONV_WIDTH):
                rq += seqq[j + w] * wq[w]
            rk = seqk[j] * wk[0]
            for w in cutlass.range_constexpr(1, CONV_WIDTH):
                rk += seqk[j + w] * wk[w]
            q_s = rq * cute.arch.rcp_approx(
                cutlass.Float32(1.0) + cute.math.exp(-rq, fastmath=True)
            )
            k_s = rk * cute.arch.rcp_approx(
                cutlass.Float32(1.0) + cute.math.exp(-rk, fastmath=True)
            )
            qv.append(q_s)
            kv.append(k_s)
            graw = cutlass.Float32(sGraw[j * K + kx]) + dtb
            gv.append(lower_bound * _sigmoid(exp_a * graw))
        for j in cutlass.range_constexpr(NT):
            red.append(qv[j] * qv[j])
        for j in cutlass.range_constexpr(NT):
            red.append(kv[j] * kv[j])
        for j in cutlass.range_constexpr(NT):
            red.append(qv[j] * kv[j])
        for _ in cutlass.range_constexpr(NRED - 3 * NT):
            red.append(cutlass.Float32(0.0))
        part, ridx = _reduce_scatter(red, lane, [16, 8, 4, 2, 1])
        if warp < 4:
            sRed[ridx * 4 + warp] = part
        # replay drafts' cached key-side rows
        kr = []
        gr = []
        for i in cutlass.range_constexpr(NUM_SPEC):
            kr.append(cutlass.Float32(sKc[i * K + kx]))
            gr.append(sGc[i * K + kx])
        # value side (own rows), betas
        # (every thread computes a value row, tidx % R; only threads < R store)
        vr_t = (tidx + NTHR - VOFF) % R
        v_thr = (tidx >= VOFF) & (tidx < VOFF + R)
        seqv = []
        vnew = []
        for m in cutlass.range_constexpr(NT + 3):
            row = (a + m) if m < 3 else (NUM_SPEC + m)
            seqv.append(cutlass.Float32(sXV[row * R + vr_t]))
        for j in cutlass.range_constexpr(NT):
            vc = cutlass.Float32(0.0)
            for w in cutlass.range_constexpr(CONV_WIDTH):
                vc += seqv[j + w] * sW[8 * K + 4 * vr_t + w]
            vnew.append(vc * _sigmoid(vc))
        vrep = []
        for i in cutlass.range_constexpr(NUM_SPEC):
            vrep.append(cutlass.Float32(sVc[i * R + vr_t]))
        if v_thr:
            for j in cutlass.range_constexpr(NT):
                sV[(a + j) * R + vr_t] = vnew[j]
            for i in cutlass.range_constexpr(NUM_SPEC):
                if i < a:
                    sV[i * R + vr_t] = vrep[i]
        if tidx < NT:
            sBeta[a + tidx] = _sigmoid(sBraw[tidx])
        if (tidx >= 16) & (tidx < 16 + NUM_SPEC):
            if tidx - 16 < a:
                sBeta[tidx - 16] = sBc[tidx - 16]
        cute.arch.barrier()

        # ---- norms (block totals in warp order), the per-token key-side rows over the staging area
        tot = []
        for v in cutlass.range_constexpr(3 * NT):
            p4 = ld_shared_f32x4(saddr(sRed, 4 * v))
            tot.append((p4[0] + p4[1]) + (p4[2] + p4[3]))
        qn = []
        kn = []
        for j in cutlass.range_constexpr(NT):
            rnq = cute.math.rsqrt(tot[j] + 1e-06, fastmath=True) * scale
            rnk = cute.math.rsqrt(tot[NT + j] + 1e-06, fastmath=True)
            qn.append(qv[j] * rnq)
            kn.append(kv[j] * rnk)
            if tidx == 0:
                sKQ[j] = tot[2 * NT + j] * rnq * rnk
        gcum = cute.make_rmem_tensor(cute.make_layout((1,)), cutlass.Float32)
        gcum[0] = cutlass.Float32(0.0)
        if tidx < K:
            for i in cutlass.range_constexpr(NUM_SPEC):
                if i < a:
                    gc = gcum[0] + gr[i]
                    gcum[0] = gc
                    sA[i * K + kx] = cute.math.exp(gc, fastmath=True) * kr[i]
                    sC[i * K + kx] = kr[i] * cute.math.exp(-gc, fastmath=True)
            for j in cutlass.range_constexpr(NT):
                if cutlass.const_expr(j == 1):
                    gcum[0] = cutlass.Float32(0.0)
                gc = gcum[0] + gv[j]
                gcum[0] = gc
                pt = cute.math.exp(gc, fastmath=True)
                sA[(a + j) * K + kx] = pt * kn[j]
                sC[(a + j) * K + kx] = kn[j] * cute.math.exp(-gc, fastmath=True)
                sB[j * K + kx] = pt * qn[j]
                if cutlass.const_expr(j == 0):
                    sPc[kx] = pt
        cute.arch.barrier()

        # ---- replay-cache rows of the new drafts and the conv-window commit, from registers, once every CTA of the
        #      tile has read the old rows (asynchronous stores; key-side drafts split over the cluster)
        if cutlass.const_expr(C > 1):
            cute.arch.cluster_wait()
        for pd in cutlass.range_constexpr(NUM_SPEC):
            if (rank == pd % C) & (tidx < K):
                st_global_b16(gaddr(k_cache, (slot, pd, hk + kx)), kn[pd + 1])
                st_global_f32(gaddr(g_cache, (slot, pd, hk + kx)), gv[pd + 1])
                st_global_b16(gaddr(cs_q, (slot, hk + kx, 3 + pd)), seqq[4 + pd])
                st_global_b16(gaddr(cs_k, (slot, hk + kx, 3 + pd)), seqk[4 + pd])
                if tidx == 0:
                    beta_cache[slot, pd, i_h] = sBeta[a + 1 + pd].to(beta_cache.element_type)
        for w in cutlass.range_constexpr(3):
            if (rank == (NUM_SPEC + w) % C) & (tidx < K):
                st_global_b16(gaddr(cs_q, (slot, hk + kx, w)), seqq[1 + w])
                st_global_b16(gaddr(cs_k, (slot, hk + kx, w)), seqk[1 + w])
        if v_thr:
            for pd in cutlass.range_constexpr(NUM_SPEC):
                st_global_b16(gaddr(v_cache, (slot, pd, hv + vr_t)), vnew[pd + 1])
                st_global_b16(gaddr(cs_v, (slot, hv + vr_t, 3 + pd)), seqv[4 + pd])
            for w in cutlass.range_constexpr(3):
                st_global_b16(gaddr(cs_v, (slot, hv + vr_t, w)), seqv[1 + w])

        # ---- recurrence (token rows 0 .. a + NT - 1 in order: replayed drafts, golden, drafts)
        # value row whose u this lane holds after the reduce-scatter (u-lanes)
        vrow_u = (lr // ((LPR // 2) // RHO)) % RHO * GRP + group
        ctx = (
            hs,
            sA,
            sC,
            sB,
            sV,
            sBeta,
            sKQ,
            sO,
            sPc,
            h0,
            slot,
            i_h,
            vbase,
            group,
            lr,
            a,
            R,
            RHO,
            LPR,
            NCH,
            vrow_u,
            GRP,
        )
        pre = (
            cute.make_rmem_tensor(cute.make_layout((4 * NCH,)), cutlass.Float32),
            cute.make_rmem_tensor(cute.make_layout((4 * NCH,)), cutlass.Float32),
            cute.make_rmem_tensor(cute.make_layout((4 * NCH,)), cutlass.Float32),
            cute.make_rmem_tensor(cute.make_layout((2,)), cutlass.Float32),
        )
        _prefetch(0, ctx, pre)
        for t in cutlass.range(0, a, 1):
            _token(t, ctx, pre, False)
        _token(a, ctx, pre, True)
        _commit(ctx)
        for t in cutlass.range(a + 1, a + NT, 1):
            _token(t, ctx, pre, True)
        cute.arch.barrier()

        # ---- gated RMSNorm: per-token partial sums of squares of this CTA's rows, exchanged over the cluster
        for jj in cutlass.range_constexpr((NT + NW - 1) // NW):
            j = warp + NW * jj
            if j < NT:
                ss = cutlass.Float32(0.0)
                for vi in cutlass.range_constexpr(VL):
                    vr = lane + 32 * vi
                    if vr < R:
                        xo = sO[j * R + vr]
                        ss += xo * xo
                for offx in [16, 8, 4, 2, 1]:
                    ss += cute.arch.shuffle_sync_bfly(ss, offset=offx, mask=-1, mask_and_clamp=31)
                if cutlass.const_expr(C > 1):
                    if lane < C:
                        st_async_f32(saddr(sPart, j * C + rank), lane, mbar_part.toint(), ss)
                else:
                    if lane == 0:
                        sPart[j] = ss
        if cutlass.const_expr(C > 1):
            mbarrier_wait_cluster(mbar_part.toint())
        else:
            cute.arch.barrier()
        for it in cutlass.range_constexpr((NT * R + NTHR - 1) // NTHR):
            item = tidx + NTHR * it
            if item < NT * R:
                j = item // R
                vr = item % R
                tot_s = cutlass.Float32(0.0)
                for rr in cutlass.range_constexpr(C):
                    tot_s += sPart[j * C + rr]
                inv_rms = cute.math.rsqrt(
                    tot_s / cutlass.Float32(V) + cutlass.Float32(onorm_eps), fastmath=True
                )
                og = cutlass.Float32(sOG[j * R + vr])
                gated = sO[j * R + vr] * sOnW[vr] * _sigmoid(og)
                o[0, i_n * NT + j, hv + vr] = cutlass.BFloat16(gated * inv_rms)


# ---------------------------------------------------------------------------------------------------------------
# Specialized warps
#
# The same arithmetic as ``kda_verify_rec_kernel`` (NTHR 128, LPR 16) with 160 threads in two roles. Warps 0-3 hold
# one key each in the precompute and the key-side rows and run the recurrence on the same state layout with the same
# token code. Warp 4 holds the value rows (conv + SiLU of v, betas), so the key warps' precompute no longer carries
# them, and it commits the replay-cache / conv-window rows: the key-side ones (which every CTA of the tile reads)
# from a small shared-memory stage (its own drafts and window only) after the cluster barrier while the recurrence
# runs, its own value rows last. The conv weights, gate bias and A_log go straight to registers and the loads of the
# prologue are all issued before any of them is waited on, so the kernel needs less shared memory per CTA than
# ``kda_verify_rec_kernel`` (an 8-CTA cluster fits beside the same neighbours). The block reduction of the norms pads
# to 16 values instead of 32 at up to five new tokens (every value's sum is the same 32-lane butterfly). Outputs and
# every written buffer are bitwise those of ``kda_verify_rec_kernel`` at the same cluster size.

#: Threads of :func:`kda_verify_rec_ws_kernel`: four key warps and the value warp.
WS_THREADS = K_DIM + 32


@cute.kernel
def kda_verify_rec_ws_kernel(
    h0: cute.Tensor,
    x_q: cute.Tensor,
    x_k: cute.Tensor,
    x_v: cute.Tensor,
    w_q: cute.Tensor,
    w_k: cute.Tensor,
    w_v: cute.Tensor,
    cs_q: cute.Tensor,
    cs_k: cute.Tensor,
    cs_v: cute.Tensor,
    A_log: cute.Tensor,
    g: cute.Tensor,
    dt_bias: cute.Tensor,
    beta: cute.Tensor,
    onorm_g: cute.Tensor,
    onorm_weight: cute.Tensor,
    o: cute.Tensor,
    k_cache: cute.Tensor,
    g_cache: cute.Tensor,
    v_cache: cute.Tensor,
    beta_cache: cute.Tensor,
    ssm_state_indices: cute.Tensor,
    num_accepted_tokens: cute.Tensor,
    scale: cutlass.Constexpr[float],
    NUM_SPEC: cutlass.Constexpr[int],
    lower_bound: cutlass.Constexpr[float],
    onorm_eps: cutlass.Constexpr[float],
    C: cutlass.Constexpr[int],
):
    """:func:`kda_verify_rec_kernel` with specialized warps (see above), for up to 32 state rows per CTA
    (``C >= 4``). Grid ``(C, H, N)``, cluster ``(C, 1, 1)``, :data:`WS_THREADS` threads; same operands."""
    K = K_DIM
    V = V_DIM
    R = V // C  # state rows of this CTA
    NTHR = WS_THREADS
    NW = NTHR // 32
    LPR = 16  # lanes per state row (16-byte key chunks: NCH per lane)
    NCH = 2
    GRP = K // LPR  # state rows updated at once by the key warps
    RHO = R // GRP  # state rows per key thread
    NT = NUM_SPEC + 1
    TMAX = 2 * NUM_SPEC + 1
    SROWS = 3 + NUM_SPEC + NT
    VL = max(1, R // 32)
    # block-reduced values per key thread (sum q^2, sum k^2, sum q k per new token), padded to a power of two: the
    # sum of each value is the same 32-lane butterfly whatever the padding
    NRED = 16 if 3 * NT <= 16 else 32
    assert NUM_SPEC <= MAX_NUM_SPEC and C in (4, 8) and 3 * NT <= NRED

    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx % 32
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    rank, i_h, i_n = cute.arch.block_idx()
    slot32 = ssm_state_indices[i_n]
    hk = i_h * K
    vbase = rank * R
    hv = i_h * V + vbase
    k4 = 4 * lane
    v4 = 4 * lane

    BF = 2
    st_bytes = [
        2 * SROWS * K * BF,
        SROWS * R * BF,
        NUM_SPEC * K * BF,
        NUM_SPEC * K * 4,
        NUM_SPEC * R * BF,
        NT * K * BF,
    ]
    post_bytes = [TMAX * K * 4, TMAX * K * 4, NT * K * 4]
    u_floats = (max(sum(st_bytes), sum(post_bytes)) + 15) // 16 * 4
    smem = cutlass.utils.SmemAllocator()
    sU = smem.allocate_tensor(cutlass.Float32, cute.make_layout((u_floats,)), 16)

    off = 0
    sXQ = _smem_view(sU, off, cutlass.BFloat16, SROWS * K)
    off += SROWS * K * BF
    sXK = _smem_view(sU, off, cutlass.BFloat16, SROWS * K)
    off += SROWS * K * BF
    sXV = _smem_view(sU, off, cutlass.BFloat16, SROWS * R)
    off += SROWS * R * BF
    sKc = _smem_view(sU, off, cutlass.BFloat16, NUM_SPEC * K)
    off += NUM_SPEC * K * BF
    sGc = _smem_view(sU, off, cutlass.Float32, NUM_SPEC * K)
    off += NUM_SPEC * K * 4
    sVc = _smem_view(sU, off, cutlass.BFloat16, NUM_SPEC * R)
    off += NUM_SPEC * R * BF
    sGraw = _smem_view(sU, off, cutlass.BFloat16, NT * K)
    sA = _smem_view(sU, 0, cutlass.Float32, TMAX * K)
    sC = _smem_view(sU, TMAX * K * 4, cutlass.Float32, TMAX * K)
    sB = _smem_view(sU, 2 * TMAX * K * 4, cutlass.Float32, NT * K)
    sOG = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((NT * R,)), 16)
    sV = smem.allocate_tensor(cutlass.Float32, cute.make_layout((TMAX * R,)), 16)
    sOnW = smem.allocate_tensor(cutlass.Float32, cute.make_layout((R,)), 16)
    sPc = smem.allocate_tensor(cutlass.Float32, cute.make_layout((K,)), 16)
    sO = smem.allocate_tensor(cutlass.Float32, cute.make_layout((NT * R,)), 16)
    sRed = smem.allocate_tensor(cutlass.Float32, cute.make_layout((NRED * 4,)), 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, cute.make_layout((16,)), 16)
    sBraw = smem.allocate_tensor(cutlass.Float32, cute.make_layout((8,)), 16)
    sBc = smem.allocate_tensor(cutlass.Float32, cute.make_layout((8,)), 16)
    sKQ = smem.allocate_tensor(cutlass.Float32, cute.make_layout((8,)), 16)
    sPart = smem.allocate_tensor(cutlass.Float32, cute.make_layout((NT * C,)), 16)
    # this CTA's key-side replay-cache / conv-window rows for the value warp's stores: per own draft (draft pd is
    # CTA pd % C's (pd // C)-th) its k / q-tail / k-tail columns in bf16 and its gate in fp32, then its conv window
    # (at most one per CTA) q / k columns in bf16
    DPC = (NUM_SPEC + C - 1) // C
    sStageB = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout(((3 * DPC + 2) * K,)), 16)
    sStageG = smem.allocate_tensor(cutlass.Float32, cute.make_layout((DPC * K,)), 16)
    mbar_part = smem.allocate_array(cutlass.Int64, 1)

    if tidx == 0:
        cute.arch.mbarrier_init(mbar_part, 1)
        cute.arch.mbarrier_arrive_and_expect_tx(mbar_part, NT * C * 4)
    cute.arch.mbarrier_init_fence()

    # ---- load phase 1 (independent of the projection GEMM): every load is issued before any is waited on; the
    #      scalar ones land in registers and reach shared memory after the new-token rows are in flight
    # a thread's conv weights, gate bias and A_log straight into registers (key threads: key tidx % K; the value
    # warp: value row lane % R)
    wq_w = ld_global_b32x4(gaddr(w_q, (hk + tidx % K, 0)))
    wk_w = ld_global_b32x4(gaddr(w_k, (hk + tidx % K, 0)))
    wv_w = ld_global_b32x4(gaddr(w_v, (hv + lane % R, 0)))
    dtb_r = cutlass.Float32(dt_bias[hk + tidx % K])
    al_r = cutlass.Float32(A_log[i_h])
    if tidx < R:
        cp_async(saddr(sOnW, tidx), gaddr(onorm_weight, (vbase + tidx,)), 4)
    slot = Int64(slot32)
    commit_raw = num_accepted_tokens[slot]
    bc_r = cutlass.Float32(0.0)
    if tidx < NUM_SPEC:
        bc_r = cutlass.Float32(beta_cache[slot, tidx, i_h])
    for r in cutlass.range_constexpr(3 + NUM_SPEC):
        if warp == r % NW:
            _cp_row4(sXQ, r * K + k4, cs_q, (slot, hk + k4, r))
            _cp_row4(sXK, r * K + k4, cs_k, (slot, hk + k4, r))
            if lane < R // 4:
                _cp_row4(sXV, r * R + v4, cs_v, (slot, hv + v4, r))
            if cutlass.const_expr(r >= 3):
                i = r - 3
                _cp_row4(sKc, i * K + k4, k_cache, (slot, i, hk + k4))
                _cp_row4(sGc, i * K + k4, g_cache, (slot, i, hk + k4))
                if lane < R // 4:
                    _cp_row4(sVc, i * R + v4, v_cache, (slot, i, hv + v4))
    cp_async_commit()
    # a key thread's state rows (value row vbase + p * GRP + group, keys 4 (lr + LPR j) .. + 3), as in
    # kda_verify_rec_kernel
    group = tidx // LPR
    lr = tidx % LPR
    hs = cute.make_rmem_tensor(cute.make_layout((RHO * 4 * NCH,)), cutlass.Float32)
    if warp < 4:
        for p in cutlass.range_constexpr(RHO):
            for j in cutlass.range_constexpr(NCH):
                wv = ld_global_b32x4(
                    gaddr(h0, (slot, i_h, vbase + p * GRP + group, 4 * (lr + LPR * j)))
                )
                for e in cutlass.range_constexpr(4):
                    hs[4 * NCH * p + 4 * j + e] = wv[e].bitcast(cutlass.Float32)

    # ---- load phase 2: the new-token rows (the projection GEMM's outputs)
    griddepcontrol_wait()
    griddepcontrol_launch_dependents()
    for j in cutlass.range_constexpr(NT):
        if warp == (3 + NUM_SPEC + j) % NW:
            rown = i_n * NT + j
            sr = 3 + NUM_SPEC + j
            _cp_row4(sXQ, sr * K + k4, x_q, (0, rown, hk + k4))
            _cp_row4(sXK, sr * K + k4, x_k, (0, rown, hk + k4))
            _cp_row4(sGraw, j * K + k4, g, (0, rown, hk + k4))
            if lane < R // 4:
                _cp_row4(sXV, sr * R + v4, x_v, (0, rown, hv + v4))
                _cp_row4(sOG, j * R + v4, onorm_g, (0, rown, hv + v4))
    braw_r = cutlass.Float32(0.0)
    if tidx < NT:
        braw_r = cutlass.Float32(beta[0, i_n * NT + tidx, i_h])
    cp_async_commit()
    if tidx < NUM_SPEC:
        sBc[tidx] = bc_r
    if tidx < NT:
        sBraw[tidx] = braw_r
    commit_len = commit_raw
    if commit_len > NUM_SPEC:
        commit_len = cutlass.Int32(NUM_SPEC)
    a = commit_len
    cp_async_wait_all()
    cute.arch.barrier()
    # every replay entry / conv window this CTA reads is in smem now; the partial-sum mbarrier is armed
    cute.arch.cluster_arrive_relaxed()
    # ---- warp roles: warps 0-3 hold one key each (precompute, key-side rows) and run the recurrence; warp 4 holds
    #      the value rows (precompute, betas) and commits the replay-cache / conv-window rows while the recurrence
    #      runs (the key-side values pass through sStageB / sStageG)
    is_key = warp < 4
    kx = tidx % K
    vr = lane  # value row of the value warp's lane (lanes < R)
    # key-side values the key warps carry from the precompute to the key-side rows
    kq = cute.make_rmem_tensor(cute.make_layout((3 * NT + 2 * NUM_SPEC,)), cutlass.Float32)
    # value-side values the value warp carries to its stores
    vst = cute.make_rmem_tensor(cute.make_layout((2 * NUM_SPEC + 3,)), cutlass.Float32)
    if is_key:
        # ---- precompute, one key per thread: conv + SiLU of q / k, gates of the new tokens (raw staging reads only)
        wq = [wq_w[w].bitcast(cutlass.Float32) for w in range(CONV_WIDTH)]
        wk = [wk_w[w].bitcast(cutlass.Float32) for w in range(CONV_WIDTH)]
        dtb = dtb_r
        exp_a = cute.math.exp(al_r, fastmath=True)
        seqq = []
        seqk = []
        for m in cutlass.range_constexpr(NT + 3):
            rowq = (a + m) if m < 3 else (NUM_SPEC + m)
            seqq.append(cutlass.Float32(sXQ[rowq * K + kx]))
            seqk.append(cutlass.Float32(sXK[rowq * K + kx]))
        qv = []
        kv = []
        red = []
        for j in cutlass.range_constexpr(NT):
            rq = cutlass.Float32(0.0)
            for w in cutlass.range_constexpr(CONV_WIDTH):
                rq += seqq[j + w] * wq[w]
            rk = seqk[j] * wk[0]
            for w in cutlass.range_constexpr(1, CONV_WIDTH):
                rk += seqk[j + w] * wk[w]
            q_s = rq * cute.arch.rcp_approx(
                cutlass.Float32(1.0) + cute.math.exp(-rq, fastmath=True)
            )
            k_s = rk * cute.arch.rcp_approx(
                cutlass.Float32(1.0) + cute.math.exp(-rk, fastmath=True)
            )
            qv.append(q_s)
            kv.append(k_s)
            graw = cutlass.Float32(sGraw[j * K + kx]) + dtb
            kq[2 * NT + j] = lower_bound * _sigmoid(exp_a * graw)
        for j in cutlass.range_constexpr(NT):
            red.append(qv[j] * qv[j])
        for j in cutlass.range_constexpr(NT):
            red.append(kv[j] * kv[j])
        for j in cutlass.range_constexpr(NT):
            red.append(qv[j] * kv[j])
        for _ in cutlass.range_constexpr(NRED - 3 * NT):
            red.append(cutlass.Float32(0.0))
        part, ridx = _reduce_scatter(red, lane, [16, 8, 4, 2, 1])
        sRed[ridx * 4 + warp] = part
        for j in cutlass.range_constexpr(NT):
            kq[j] = qv[j]
            kq[NT + j] = kv[j]
        for i in cutlass.range_constexpr(NUM_SPEC):
            kq[3 * NT + i] = cutlass.Float32(sKc[i * K + kx])
            kq[3 * NT + NUM_SPEC + i] = sGc[i * K + kx]
        # raw conv columns and gates of the new drafts, for the stores (this CTA's share)
        for pd in cutlass.range_constexpr(NUM_SPEC):
            if rank == pd % C:
                sStageG[pd // C * K + kx] = kq[2 * NT + pd + 1]
                sStageB[(3 * (pd // C) + 1) * K + kx] = cutlass.BFloat16(seqq[4 + pd])
                sStageB[(3 * (pd // C) + 2) * K + kx] = cutlass.BFloat16(seqk[4 + pd])
        for w in cutlass.range_constexpr(3):
            if rank == (NUM_SPEC + w) % C:
                sStageB[3 * DPC * K + kx] = cutlass.BFloat16(seqq[1 + w])
                sStageB[(3 * DPC + 1) * K + kx] = cutlass.BFloat16(seqk[1 + w])
    else:
        # ---- value side (own rows, one per lane) and betas
        seqv = []
        for m in cutlass.range_constexpr(NT + 3):
            rowv = (a + m) if m < 3 else (NUM_SPEC + m)
            seqv.append(cutlass.Float32(sXV[rowv * R + vr % R]))
        vnew = []
        for j in cutlass.range_constexpr(NT):
            vc = cutlass.Float32(0.0)
            for w in cutlass.range_constexpr(CONV_WIDTH):
                vc += seqv[j + w] * wv_w[w].bitcast(cutlass.Float32)
            vnew.append(vc * _sigmoid(vc))
        vrep = []
        for i in cutlass.range_constexpr(NUM_SPEC):
            vrep.append(cutlass.Float32(sVc[i * R + vr % R]))
        if vr < R:
            for j in cutlass.range_constexpr(NT):
                sV[(a + j) * R + vr] = vnew[j]
            for i in cutlass.range_constexpr(NUM_SPEC):
                if i < a:
                    sV[i * R + vr] = vrep[i]
        if lane < NT:
            sBeta[a + lane] = _sigmoid(sBraw[lane])
        if (lane >= 16) & (lane < 16 + NUM_SPEC):
            if lane - 16 < a:
                sBeta[lane - 16] = sBc[lane - 16]
        for pd in cutlass.range_constexpr(NUM_SPEC):
            vst[pd] = vnew[pd + 1]
            vst[NUM_SPEC + pd] = seqv[4 + pd]
        for w in cutlass.range_constexpr(3):
            vst[2 * NUM_SPEC + w] = seqv[1 + w]
    cute.arch.barrier()

    # ---- norms (block totals in warp order), the per-token key-side rows over the staging area
    if is_key:
        tot = []
        for v in cutlass.range_constexpr(3 * NT):
            p4 = ld_shared_f32x4(saddr(sRed, 4 * v))
            tot.append((p4[0] + p4[1]) + (p4[2] + p4[3]))
        qn = []
        kn = []
        for j in cutlass.range_constexpr(NT):
            rnq = cute.math.rsqrt(tot[j] + 1e-06, fastmath=True) * scale
            rnk = cute.math.rsqrt(tot[NT + j] + 1e-06, fastmath=True)
            qn.append(kq[j] * rnq)
            kn.append(kq[NT + j] * rnk)
            if tidx == 0:
                sKQ[j] = tot[2 * NT + j] * rnq * rnk
        for pd in cutlass.range_constexpr(NUM_SPEC):
            if rank == pd % C:
                sStageB[3 * (pd // C) * K + kx] = cutlass.BFloat16(kn[pd + 1])
        gcum = cute.make_rmem_tensor(cute.make_layout((1,)), cutlass.Float32)
        gcum[0] = cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(NUM_SPEC):
            if i < a:
                kr = kq[3 * NT + i]
                gc = gcum[0] + kq[3 * NT + NUM_SPEC + i]
                gcum[0] = gc
                sA[i * K + kx] = cute.math.exp(gc, fastmath=True) * kr
                sC[i * K + kx] = kr * cute.math.exp(-gc, fastmath=True)
        for j in cutlass.range_constexpr(NT):
            if cutlass.const_expr(j == 1):
                gcum[0] = cutlass.Float32(0.0)
            gc = gcum[0] + kq[2 * NT + j]
            gcum[0] = gc
            pt = cute.math.exp(gc, fastmath=True)
            sA[(a + j) * K + kx] = pt * kn[j]
            sC[(a + j) * K + kx] = kn[j] * cute.math.exp(-gc, fastmath=True)
            sB[j * K + kx] = pt * qn[j]
            if cutlass.const_expr(j == 0):
                sPc[kx] = pt
    cute.arch.barrier()

    if is_key:
        # ---- recurrence (token rows 0 .. a + NT - 1 in order: replayed drafts, golden, drafts) with the token code
        #      of kda_verify_rec_kernel
        vrow_u = (lr // ((LPR // 2) // RHO)) % RHO * GRP + group
        ctx = (
            hs,
            sA,
            sC,
            sB,
            sV,
            sBeta,
            sKQ,
            sO,
            sPc,
            h0,
            slot,
            i_h,
            vbase,
            group,
            lr,
            a,
            R,
            RHO,
            LPR,
            NCH,
            vrow_u,
            GRP,
        )
        pre = (
            cute.make_rmem_tensor(cute.make_layout((4 * NCH,)), cutlass.Float32),
            cute.make_rmem_tensor(cute.make_layout((4 * NCH,)), cutlass.Float32),
            cute.make_rmem_tensor(cute.make_layout((4 * NCH,)), cutlass.Float32),
            cute.make_rmem_tensor(cute.make_layout((2,)), cutlass.Float32),
        )
        _prefetch(0, ctx, pre)
        for t in cutlass.range(0, a, 1):
            _token(t, ctx, pre, False)
        _token(a, ctx, pre, True)
        _commit(ctx)
        for t in cutlass.range(a + 1, a + NT, 1):
            _token(t, ctx, pre, True)
        # the cluster barrier before the norm exchange's remote stores
        cute.arch.cluster_wait()
    else:
        # ---- replay-cache rows of the new drafts and the conv-window commit, key side, once every CTA of the tile has
        #      read the old rows (the value rows, read by this CTA only, are committed last)
        cute.arch.cluster_wait()
        for pd in cutlass.range_constexpr(NUM_SPEC):
            if rank == pd % C:
                for c4 in cutlass.range_constexpr(K // 32):
                    kk = lane + 32 * c4
                    j3 = 3 * (pd // C) * K + kk
                    st_global_b16(gaddr(k_cache, (slot, pd, hk + kk)), cutlass.Float32(sStageB[j3]))
                    st_global_f32(gaddr(g_cache, (slot, pd, hk + kk)), sStageG[pd // C * K + kk])
                    st_global_b16(
                        gaddr(cs_q, (slot, hk + kk, 3 + pd)), cutlass.Float32(sStageB[j3 + K])
                    )
                    st_global_b16(
                        gaddr(cs_k, (slot, hk + kk, 3 + pd)), cutlass.Float32(sStageB[j3 + 2 * K])
                    )
                if lane == 0:
                    beta_cache[slot, pd, i_h] = sBeta[a + 1 + pd].to(beta_cache.element_type)
        for w in cutlass.range_constexpr(3):
            if rank == (NUM_SPEC + w) % C:
                for c4 in cutlass.range_constexpr(K // 32):
                    kk = lane + 32 * c4
                    st_global_b16(
                        gaddr(cs_q, (slot, hk + kk, w)), cutlass.Float32(sStageB[3 * DPC * K + kk])
                    )
                    st_global_b16(
                        gaddr(cs_k, (slot, hk + kk, w)),
                        cutlass.Float32(sStageB[(3 * DPC + 1) * K + kk]),
                    )
    cute.arch.barrier()

    # ---- gated RMSNorm: per-token partial sums of squares of this CTA's rows, exchanged over the cluster
    for jj in cutlass.range_constexpr((NT + NW - 1) // NW):
        j = warp + NW * jj
        if j < NT:
            ss = cutlass.Float32(0.0)
            for vi in cutlass.range_constexpr(VL):
                vrr = lane + 32 * vi
                if vrr < R:
                    xo = sO[j * R + vrr]
                    ss += xo * xo
            for offx in [16, 8, 4, 2, 1]:
                ss += cute.arch.shuffle_sync_bfly(ss, offset=offx, mask=-1, mask_and_clamp=31)
            if lane < C:
                st_async_f32(saddr(sPart, j * C + rank), lane, mbar_part.toint(), ss)
    mbarrier_wait_cluster(mbar_part.toint())
    for it in cutlass.range_constexpr((NT * R + NTHR - 1) // NTHR):
        item = tidx + NTHR * it
        if item < NT * R:
            j = item // R
            vro = item % R
            tot_s = cutlass.Float32(0.0)
            for rr in cutlass.range_constexpr(C):
                tot_s += sPart[j * C + rr]
            inv_rms = cute.math.rsqrt(
                tot_s / cutlass.Float32(V) + cutlass.Float32(onorm_eps), fastmath=True
            )
            og = cutlass.Float32(sOG[j * R + vro])
            gated = sO[j * R + vro] * sOnW[vro] * _sigmoid(og)
            o[0, i_n * NT + j, hv + vro] = cutlass.BFloat16(gated * inv_rms)
    # ---- the value rows' replay-cache / conv-window commit
    if (warp == 4) & (vr < R):
        for pd in cutlass.range_constexpr(NUM_SPEC):
            st_global_b16(gaddr(v_cache, (slot, pd, hv + vr)), vst[pd])
            st_global_b16(gaddr(cs_v, (slot, hv + vr, 3 + pd)), vst[NUM_SPEC + pd])
        for w in cutlass.range_constexpr(3):
            st_global_b16(gaddr(cs_v, (slot, hv + vr, w)), vst[2 * NUM_SPEC + w])


# ---------------------------------------------------------------------------------------------------------------
# Host side

_compiled = {}


@cute.jit
def _run_kda_verify_rec(
    h0: cute.Tensor,
    x_q: cute.Tensor,
    x_k: cute.Tensor,
    x_v: cute.Tensor,
    w_q: cute.Tensor,
    w_k: cute.Tensor,
    w_v: cute.Tensor,
    cs_q: cute.Tensor,
    cs_k: cute.Tensor,
    cs_v: cute.Tensor,
    A_log: cute.Tensor,
    g: cute.Tensor,
    dt_bias: cute.Tensor,
    beta: cute.Tensor,
    onorm_g: cute.Tensor,
    onorm_weight: cute.Tensor,
    o: cute.Tensor,
    k_cache: cute.Tensor,
    g_cache: cute.Tensor,
    v_cache: cute.Tensor,
    beta_cache: cute.Tensor,
    ssm_state_indices: cute.Tensor,
    num_accepted_tokens: cute.Tensor,
    N: cutlass.Int32,
    scale: cutlass.Constexpr[float],
    H: cutlass.Constexpr[int],
    NUM_SPEC: cutlass.Constexpr[int],
    lower_bound: cutlass.Constexpr[float],
    onorm_eps: cutlass.Constexpr[float],
    C: cutlass.Constexpr[int],
    LPR: cutlass.Constexpr[int],
    NTHR: cutlass.Constexpr[int],
    WS: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    if cutlass.const_expr(WS):
        kda_verify_rec_ws_kernel(
            h0,
            x_q,
            x_k,
            x_v,
            w_q,
            w_k,
            w_v,
            cs_q,
            cs_k,
            cs_v,
            A_log,
            g,
            dt_bias,
            beta,
            onorm_g,
            onorm_weight,
            o,
            k_cache,
            g_cache,
            v_cache,
            beta_cache,
            ssm_state_indices,
            num_accepted_tokens,
            scale,
            NUM_SPEC,
            lower_bound,
            onorm_eps,
            C,
        ).launch(
            grid=(C, H, N),
            block=[WS_THREADS, 1, 1],
            cluster=(C, 1, 1),
            stream=stream,
            use_pdl=True,
        )
    else:
        kda_verify_rec_kernel(
            h0,
            x_q,
            x_k,
            x_v,
            w_q,
            w_k,
            w_v,
            cs_q,
            cs_k,
            cs_v,
            A_log,
            g,
            dt_bias,
            beta,
            onorm_g,
            onorm_weight,
            o,
            k_cache,
            g_cache,
            v_cache,
            beta_cache,
            ssm_state_indices,
            num_accepted_tokens,
            scale,
            NUM_SPEC,
            lower_bound,
            onorm_eps,
            C,
            LPR,
            NTHR,
        ).launch(
            grid=(C, H, N),
            block=[NTHR, 1, 1],
            cluster=(C, 1, 1),
            stream=stream,
            use_pdl=True,
        )


def default_cluster(num_heads: int, num_requests: int, num_sms: int) -> int:
    """CTAs per (head, request) tile: the widest split with at most ~2.6 CTAs per SM."""
    tiles = num_heads * num_requests
    for c in (8, 4):
        if tiles * c <= (num_sms * 21) // 8:
            return c
    return 2


def use_specialized_kernel(cluster: int) -> bool:
    """Whether a launch at this cluster size runs :func:`kda_verify_rec_ws_kernel` (up to 32 state rows per CTA),
    else :func:`kda_verify_rec_kernel`."""
    return cluster >= 4


def check_launch(cluster: int, specialized: bool, lpr: int, nthr: int) -> None:
    """Raise ``ValueError`` where a launch of ``cluster`` CTAs per (head, request) tile breaks the work split the
    kernels assume (module docstring). :func:`kda_verify_rec_ws_kernel` (``specialized``) takes 4 or 8 CTAs.
    :func:`kda_verify_rec_kernel` takes ``nthr`` threads with ``lpr`` lanes per state row; a CTA owns
    ``R = 128 / cluster`` rows, so each thread holds ``RHO = R / (nthr / lpr) = 128 lpr / (cluster nthr)`` rows."""
    if cluster not in (1, 2, 4, 8):
        raise ValueError(f"a (head, request) tile runs on 1, 2, 4 or 8 CTAs, got {cluster}")
    if specialized:
        if not use_specialized_kernel(cluster):
            raise ValueError("the specialized kernel takes clusters of 4 or 8 CTAs")
        return
    if nthr not in (128, 256, 512) or lpr not in (4, 8, 16):
        raise ValueError(f"NTHR is 128, 256 or 512 and LPR 4, 8 or 16, got NTHR {nthr}, LPR {lpr}")
    if cluster * nthr < 256:
        raise ValueError(
            "a state row's reduce-scatter leaves each of its LPR lanes one of the row's 2 RHO sums "
            f"(2 RHO <= LPR, so C * NTHR >= 256), got C {cluster} x NTHR {nthr}"
        )
    if cluster * nthr > 128 * lpr:
        raise ValueError(
            "every thread holds at least one of the CTA's state rows (RHO >= 1, so C * NTHR <= 128 * LPR), "
            f"got C {cluster} x NTHR {nthr} > 128 x LPR {lpr}"
        )


@functools.cache
def _is_supported_device(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) in ((10, 0), (10, 3), (10, 7))


def supported(
    recurrent_state, num_spec: int, head_k_dim: int, head_v_dim: int, conv_width: int
) -> bool:
    """Whether :func:`kda_verify_rec` takes this layer's verify (SM100 / SM103 / SM107, FP32 V-first state with
    16-byte rows, up to ``MAX_NUM_SPEC`` drafts, K = V = 128, conv width 4)."""
    return (
        recurrent_state.is_cuda
        and _is_supported_device(
            recurrent_state.device.index
            if recurrent_state.device.index is not None
            else torch.cuda.current_device()
        )
        and recurrent_state.dtype == torch.float32
        and recurrent_state.dim() == 4
        and recurrent_state.stride(-1) == 1
        and recurrent_state.stride(-2) % 4 == 0
        and recurrent_state.data_ptr() % 16 == 0
        and 1 <= num_spec <= MAX_NUM_SPEC
        and head_k_dim == K_DIM
        and head_v_dim == V_DIM
        and conv_width == CONV_WIDTH
    )


def kda_verify_rec(
    x_q,
    x_k,
    x_v,
    w_q,
    w_k,
    w_v,
    cs_q,
    cs_k,
    cs_v,
    g,
    beta,
    A_log,
    dt_bias,
    recurrent_state,
    k_cache,
    g_cache,
    v_cache,
    beta_cache,
    ssm_state_indices,
    num_spec,
    num_accepted_tokens,
    lower_bound,
    scale=None,
    out=None,
    onorm_g=None,
    onorm_weight=None,
    onorm_eps=1e-6,
    cluster=None,
    lpr=16,
    nthr=128,
    specialized=None,
):
    """FP32 recurrent KDA verify on the packed fused-norm contract of ``kda_mtp_decode_impl`` (see the module
    docstring); returns the BF16 ``[1, T, H, V]`` output (``out`` if given). ``cluster``: CTAs per (head, request)
    tile (None: :func:`default_cluster`). ``specialized``: run :func:`kda_verify_rec_ws_kernel` (default:
    :func:`use_specialized_kernel`; same bits either way; ``lpr`` and ``nthr`` apply to :func:`kda_verify_rec_kernel`
    only). :func:`check_launch` rejects a launch shape outside the kernels' work split."""
    from tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_kda_mtp_ops import _dlpack_arg, _layout_key

    _, T_total, H, Kd = x_q.shape
    if Kd != K_DIM or x_v.shape[-1] != V_DIM or w_q.shape[1] != CONV_WIDTH:
        raise ValueError("kda_verify_rec expects K = V = 128 and conv width 4")
    if not 1 <= num_spec <= MAX_NUM_SPEC:
        raise ValueError(f"kda_verify_rec supports 1..{MAX_NUM_SPEC} drafts")
    if recurrent_state.dtype != torch.float32:
        raise ValueError("kda_verify_rec takes an FP32 state")
    N = ssm_state_indices.shape[0]
    if T_total != N * (num_spec + 1):
        raise ValueError("packed rows: T must be N * (1 + num_spec)")
    if scale is None:
        scale = K_DIM**-0.5
    if out is None:
        out = torch.empty((1, T_total, H, V_DIM), dtype=torch.bfloat16, device=x_q.device)
    if num_accepted_tokens.dtype != torch.int32:
        num_accepted_tokens = num_accepted_tokens.to(torch.int32)
    num_sms = torch.cuda.get_device_properties(x_q.device).multi_processor_count
    C = default_cluster(H, N, num_sms) if cluster is None else int(cluster)
    ws = use_specialized_kernel(C) if specialized is None else bool(specialized)
    check_launch(C, ws, int(lpr), int(nthr))
    flat = lambda t: t.view(1, T_total, H * 128)  # noqa: E731
    args = [
        (recurrent_state, 16),
        (flat(x_q), 16),
        (flat(x_k), 16),
        (flat(x_v), 16),
        (w_q, 16),
        (w_k, 16),
        (w_v, 16),
        (cs_q, 16),
        (cs_k, 16),
        (cs_v, 16),
        (A_log, 16),
        (flat(g), 16),
        (dt_bias, 16),
        (beta, 2),
        (flat(onorm_g), 16),
        (onorm_weight, 16),
        (flat(out), 16),
        (k_cache, 16),
        (g_cache, 16),
        (v_cache, 16),
        (beta_cache, 2),
        (ssm_state_indices, 4),
        (num_accepted_tokens, 4),
    ]
    constexprs = dict(
        scale=float(scale),
        H=H,
        NUM_SPEC=int(num_spec),
        lower_bound=float(lower_bound),
        onorm_eps=float(onorm_eps),
        C=int(C),
        LPR=16 if ws else int(lpr),
        NTHR=WS_THREADS if ws else int(nthr),
        WS=ws,
    )
    key = (
        tuple(_layout_key(t, dynamic_layout=True, assumed_align=al) for t, al in args),
        tuple(sorted(constexprs.items())),
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if key not in _compiled:
        _compiled[key] = cute.compile(
            _run_kda_verify_rec,
            *[_dlpack_arg(t, assumed_align=al) for t, al in args],
            N,
            **constexprs,
            stream=stream,
        )
    _compiled[key](*[_dlpack_arg(t, assumed_align=al) for t, al in args], N, stream)
    return out


def k31_kda_verify_rec(
    x_q: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    cs_q: torch.Tensor,
    cs_k: torch.Tensor,
    cs_v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent_state: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    v_cache: torch.Tensor,
    beta_cache: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_spec: int,
    num_accepted_tokens: torch.Tensor,
    lower_bound: float,
    scale: float,
    onorm_g: torch.Tensor,
    onorm_weight: torch.Tensor,
    onorm_eps: float,
    out: torch.Tensor,
) -> None:
    """:func:`kda_verify_rec` writing the BF16 ``[1, T, H, V]`` output into ``out`` (default cluster)."""
    kda_verify_rec(
        x_q,
        x_k,
        x_v,
        w_q,
        w_k,
        w_v,
        cs_q,
        cs_k,
        cs_v,
        g,
        beta,
        A_log,
        dt_bias,
        recurrent_state,
        k_cache,
        g_cache,
        v_cache,
        beta_cache,
        ssm_state_indices,
        num_spec,
        num_accepted_tokens,
        lower_bound,
        scale=scale,
        out=out,
        onorm_g=onorm_g,
        onorm_weight=onorm_weight,
        onorm_eps=onorm_eps,
    )
