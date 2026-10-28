# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused Conv4, native 2FP4 KDA replay, and gated output RMSNorm."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda_exp
import cutlass.utils as utils
import torch
from cutlass._mlir import ir as mlir_ir
from cutlass.cute.runtime import make_fake_tensor
from cutlass.experimental import primitives as prims

THREADS = 128
K = 128
N = 16
STATE_BYTES = 18448
PRIMARY_SCALE = 16384
OUTER_SCALE = 18432

_compiled = {}


@cute.jit
def _warp_sum_squares(value):
    partner = cute.arch.shuffle_sync_bfly(value * value, 16)
    total = cute.fma(value, value, partner)
    for offset in (8, 4, 2, 1):
        total = total + cute.arch.shuffle_sync_bfly(total, offset)
    return total


@cute.jit
def _warp_sum_products(lhs, rhs):
    partner = cute.arch.shuffle_sync_bfly(lhs * rhs, 16)
    total = cute.fma(lhs, rhs, partner)
    for offset in (8, 4, 2, 1):
        total = total + cute.arch.shuffle_sync_bfly(total, offset)
    return total


@cute.jit
def _store_bf16(ptr, value):
    typed_ptr = cutlass.inttoptr(ptr.toint(), cutlass.AddressSpace.smem, cutlass.BFloat16)
    typed_ptr.store(cutlass.Vector.from_elements((value,), dtype=cutlass.BFloat16))


@cute.jit
def _store_bf16x2(ptr, value0, value1):
    typed_ptr = cutlass.inttoptr(ptr.toint(), cutlass.AddressSpace.smem, cutlass.BFloat16)
    typed_ptr.store(
        cutlass.Vector.from_elements((value0, value1), dtype=cutlass.BFloat16),
        alignment=4,
    )


@cute.jit
def _kda_scalar_active(
    tma_state: cutlass.GridConstant[cuda_exp.TensorMap],
    tma_hk: cutlass.GridConstant[cuda_exp.TensorMap],
    tma_hu: cutlass.GridConstant[cuda_exp.TensorMap],
    tma_hg: cutlass.GridConstant[cuda_exp.TensorMap],
    x,
    conv_state,
    conv_weight,
    gate,
    beta,
    z,
    norm_weight,
    pool,
    slots,
    dummy,
    slot_map,
    hk,
    hu,
    hg,
    lengths,
    a_log,
    dt_bias,
    out,
    old_lengths,
    candidate_x,
    lower_bound: cutlass.Float32,
    norm_eps: cutlass.Float32,
    heads: cutlass.Constexpr,
    bounded: cutlass.Constexpr,
    tokens: cutlass.Constexpr,
    mtp: cutlass.Constexpr,
    token: cutlass.Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    batch = block // heads
    head = block - batch * heads
    row = batch * tokens + token

    slot = slots[batch].to(cutlass.Int32)
    live = slot_map[slot].to(cutlass.Int32)
    accepted_length = lengths[live].to(cutlass.Int32)
    length = accepted_length + token

    smem = utils.SmemAllocator()
    s_q = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((K,)), byte_alignment=16)
    s_k = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((K,)), byte_alignment=16)
    s_cumulative = smem.allocate_tensor(cutlass.Float32, cute.make_layout((K,)), byte_alignment=16)
    s_core = cute.make_tensor(
        cute.recast_ptr(s_cumulative.iterator, dtype=cutlass.BFloat16),
        cute.make_layout((K,)),
    )
    # Four warp partials for two simultaneous reductions plus CTA scalars.
    s_red = smem.allocate_tensor(cutlass.Float32, cute.make_layout((16,)), byte_alignment=16)
    s_scalar = smem.allocate_tensor(cutlass.Float32, cute.make_layout((8,)), byte_alignment=16)

    mma_a = cutlass.Array(cutlass.Int8, 128 * 64, space=cutlass.AddressSpace.smem, alignment=128)
    mma_b = cutlass.Array(cutlass.Int8, 8 * 64, space=cutlass.AddressSpace.smem, alignment=128)
    mma_sfa = cutlass.Array(
        cutlass.Float8E4M3FN, 1024, space=cutlass.AddressSpace.smem, alignment=128
    )
    mma_sfb = cutlass.Array(
        cutlass.Float8E4M3FN, 1024, space=cutlass.AddressSpace.smem, alignment=128
    )
    acc_mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    state_mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    coeff_mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    history_mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    history_load_mbar = cutlass.Array(
        cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8
    )
    tmem_holding = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4)
    dense_a_storage = cutlass.Array(
        cutlass.BFloat16, 144 * 64, space=cutlass.AddressSpace.smem, alignment=128
    )
    # The live coefficient panels occupy only the first 4 KB. Reuse padding
    # consumed solely by ignored MMA rows for early HK/HG TMA staging.
    cache_hk = cutlass.Array(
        dense_a_storage.subview(2048).data_ptr(),
        shape=N * K,
        dtype=cutlass.BFloat16,
    )
    cache_hg = cutlass.Array(
        dense_a_storage.subview(4096).data_ptr(),
        shape=N * K,
        dtype=cutlass.Float32,
    )
    cache_hu = cutlass.Array(
        cutlass.BFloat16, N * K, space=cutlass.AddressSpace.smem, alignment=128
    )
    dense_a0 = cutlass.Array(dense_a_storage.data_ptr(), shape=128 * 64, dtype=cutlass.BFloat16)
    dense_a1 = cutlass.Array(
        dense_a_storage.subview(16 * 64).data_ptr(),
        shape=128 * 64,
        dtype=cutlass.BFloat16,
    )
    dense_b_storage = cutlass.Array(
        cutlass.BFloat16, 10 * 64, space=cutlass.AddressSpace.smem, alignment=128
    )
    dense_b0 = cutlass.Array(dense_b_storage.data_ptr(), shape=8 * 64, dtype=cutlass.BFloat16)
    dense_b1 = cutlass.Array(
        dense_b_storage.subview(2 * 64).data_ptr(),
        shape=8 * 64,
        dtype=cutlass.BFloat16,
    )

    if warp == 0:
        if prims.elect_sync():
            prims.mbarrier_init(acc_mbar, 1)
            prims.mbarrier_init(state_mbar, 1)
            prims.mbarrier_init(coeff_mbar, 1)
            prims.mbarrier_init(history_mbar, 1)
            prims.mbarrier_init(history_load_mbar, 1)
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync(0)
    if warp == 0:
        prims.tcgen05_alloc(tmem_holding, 32, group="cta_1")
    prims.barrier_cta_sync(0)
    tmem_raw = tmem_holding.load()
    prims.tcgen05_relinquish_alloc_permit(group="cta_1")
    if warp == 0:
        if prims.elect_sync():
            prims.mbarrier_arrive_expect_tx(state_mbar, tma_state.global_tx_bytes())
            prims.cp_async_bulk_tensor_shared_cta_global(
                mma_a,
                tma_state.get_ptr(),
                (
                    cutlass.Int32(0),
                    cutlass.Int32(0),
                    head,
                    slot,
                ),
                state_mbar,
            )
            if cutlass.const_expr(not mtp):
                if length > 0:
                    history_bytes = (
                        tma_hk.global_tx_bytes()
                        + tma_hu.global_tx_bytes()
                        + tma_hg.global_tx_bytes()
                    )
                    prims.mbarrier_arrive_expect_tx(history_load_mbar, history_bytes)
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        cache_hk,
                        tma_hk.get_ptr(),
                        (
                            cutlass.Int32(0),
                            cutlass.Int32(0),
                            head,
                            live,
                        ),
                        history_load_mbar,
                    )
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        cache_hg,
                        tma_hg.get_ptr(),
                        (
                            cutlass.Int32(0),
                            cutlass.Int32(0),
                            cutlass.Int32(0),
                            head,
                            live,
                        ),
                        history_load_mbar,
                    )
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        cache_hu,
                        tma_hu.get_ptr(),
                        (
                            cutlass.Int32(0),
                            cutlass.Int32(0),
                            head,
                            live,
                        ),
                        history_load_mbar,
                    )

    width = heads * K
    q_channel = head * K + tidx
    k_channel = width + head * K + tidx
    v_channel = 2 * width + head * K + tidx

    # Conv4 uses the old window, then publishes [old1, old2, raw].
    raw_q = x[(row, q_channel)]
    raw_k = x[(row, k_channel)]
    raw_v = x[(row, v_channel)]
    z_bf16 = z[(row, head, tidx)]
    norm_bf16 = norm_weight[tidx]
    beta_bf16 = beta[(row, head)]
    gate_bf16 = gate[(row, head * K + tidx)]
    dt_value = dt_bias[head * K + tidx]
    a_log_value = a_log[head]
    pool_off = cute.crd2idx((slot, head, 0), pool.layout)
    pool_base = pool.iterator + pool_off
    state_sf = cute.recast_ptr(pool_base + PRIMARY_SCALE, dtype=cutlass.Float8E4M3FN)
    state_scale_vec = (state_sf.raw_ptr() + tidx * 8).load(count=8, alignment=8)
    state_outer = cute.recast_ptr(pool_base + OUTER_SCALE, dtype=cutlass.Float32)[0]
    prev_g_prefetch = cutlass.Float32(0.0)
    if length > 0:
        prev_g_prefetch = hg[(live, head, length - 1, tidx)]

    state_raw = conv_state.iterator.raw_ptr()
    q_state_ptr = state_raw + cute.crd2idx((slot, q_channel, 0), conv_state.layout)
    k_state_ptr = state_raw + cute.crd2idx((slot, k_channel, 0), conv_state.layout)
    v_state_ptr = state_raw + cute.crd2idx((slot, v_channel, 0), conv_state.layout)
    q_state = q_state_ptr.load(count=3, alignment=2)
    k_state = k_state_ptr.load(count=3, alignment=2)
    v_state = v_state_ptr.load(count=3, alignment=2)
    q0, q1, q2 = q_state[0], q_state[1], q_state[2]
    k0, k1, k2 = k_state[0], k_state[1], k_state[2]
    v0, v1, v2 = v_state[0], v_state[1], v_state[2]

    if cutlass.const_expr(mtp):
        if token > 0:
            if token >= 3:
                q0 = x[(row - 3, q_channel)]
            elif token == 2:
                q0 = q_state[2]
            else:
                q0 = q_state[1]
            if token >= 2:
                q1 = x[(row - 2, q_channel)]
            else:
                q1 = q_state[2]
            q2 = x[(row - 1, q_channel)]
            if token >= 3:
                k0 = x[(row - 3, k_channel)]
            elif token == 2:
                k0 = k_state[2]
            else:
                k0 = k_state[1]
            if token >= 2:
                k1 = x[(row - 2, k_channel)]
            else:
                k1 = k_state[2]
            k2 = x[(row - 1, k_channel)]
            if token >= 3:
                v0 = x[(row - 3, v_channel)]
            elif token == 2:
                v0 = v_state[2]
            else:
                v0 = v_state[1]
            if token >= 2:
                v1 = x[(row - 2, v_channel)]
            else:
                v1 = v_state[2]
            v2 = x[(row - 1, v_channel)]

    weight_raw = conv_weight.iterator.raw_ptr()
    wq = (weight_raw + q_channel * 4).load(count=4, alignment=8)
    wk = (weight_raw + k_channel * 4).load(count=4, alignment=8)
    wv = (weight_raw + v_channel * 4).load(count=4, alignment=8)
    aq = cute.fma(wq[0].to(cutlass.Float32), q0.to(cutlass.Float32), cutlass.Float32(0.0))
    aq = cute.fma(wq[1].to(cutlass.Float32), q1.to(cutlass.Float32), aq)
    aq = cute.fma(wq[2].to(cutlass.Float32), q2.to(cutlass.Float32), aq)
    aq = cute.fma(wq[3].to(cutlass.Float32), raw_q.to(cutlass.Float32), aq)
    ak = cute.fma(wk[0].to(cutlass.Float32), k0.to(cutlass.Float32), cutlass.Float32(0.0))
    ak = cute.fma(wk[1].to(cutlass.Float32), k1.to(cutlass.Float32), ak)
    ak = cute.fma(wk[2].to(cutlass.Float32), k2.to(cutlass.Float32), ak)
    ak = cute.fma(wk[3].to(cutlass.Float32), raw_k.to(cutlass.Float32), ak)
    av = cute.fma(wv[0].to(cutlass.Float32), v0.to(cutlass.Float32), cutlass.Float32(0.0))
    av = cute.fma(wv[1].to(cutlass.Float32), v1.to(cutlass.Float32), av)
    av = cute.fma(wv[2].to(cutlass.Float32), v2.to(cutlass.Float32), av)
    av = cute.fma(wv[3].to(cutlass.Float32), raw_v.to(cutlass.Float32), av)

    aq = aq * cute.rcp(cutlass.Float32(1.0) + cute.exp(-aq, fastmath=True))
    ak = ak * cute.rcp(cutlass.Float32(1.0) + cute.exp(-ak, fastmath=True))
    av = av * cute.rcp(cutlass.Float32(1.0) + cute.exp(-av, fastmath=True))
    s_q[tidx] = aq.to(cutlass.BFloat16)
    s_k[tidx] = ak.to(cutlass.BFloat16)
    v_bf16 = av.to(cutlass.BFloat16)

    if cutlass.const_expr(mtp):
        candidate_x[(batch, token, q_channel)] = raw_q
        candidate_x[(batch, token, k_channel)] = raw_k
        candidate_x[(batch, token, v_channel)] = raw_v
    else:
        q_state_ptr.store(
            cutlass.Vector.from_elements((q1, q2, raw_q), dtype=cutlass.BFloat16),
            alignment=2,
        )
        k_state_ptr.store(
            cutlass.Vector.from_elements((k1, k2, raw_k), dtype=cutlass.BFloat16),
            alignment=2,
        )
        v_state_ptr.store(
            cutlass.Vector.from_elements((v1, v2, raw_v), dtype=cutlass.BFloat16),
            alignment=2,
        )
    cute.arch.barrier()

    # Simultaneous Q/K norm reductions.
    qf = s_q[tidx].to(cutlass.Float32)
    kf = s_k[tidx].to(cutlass.Float32)
    q0s = _warp_sum_squares(s_q[lane].to(cutlass.Float32))
    q1s = _warp_sum_squares(s_q[lane + 32].to(cutlass.Float32))
    q2s = _warp_sum_squares(s_q[lane + 64].to(cutlass.Float32))
    q3s = _warp_sum_squares(s_q[lane + 96].to(cutlass.Float32))
    k0s = _warp_sum_squares(s_k[lane].to(cutlass.Float32))
    k1s = _warp_sum_squares(s_k[lane + 32].to(cutlass.Float32))
    k2s = _warp_sum_squares(s_k[lane + 64].to(cutlass.Float32))
    k3s = _warp_sum_squares(s_k[lane + 96].to(cutlass.Float32))
    qr = (q0s + q2s) + (q1s + q3s)
    kr = (k0s + k2s) + (k1s + k3s)
    qden = cutlass.Float32(0.0)
    kden = cutlass.Float32(0.0)
    if lane == 0:
        qden = cute.sqrt(qr + cutlass.Float32(1.0e-6), approx=True, ftz=True)
        kden = cute.sqrt(kr + cutlass.Float32(1.0e-6), approx=True, ftz=True)
    qden = cute.arch.shuffle_sync(qden, 0)
    kden = cute.arch.shuffle_sync(kden, 0)

    qn = cute.div(qf * cutlass.Float32(0.08838834764831845), qden, full=True).to(cutlass.BFloat16)
    kn = cute.div(kf, kden, full=True).to(cutlass.BFloat16)
    s_q[tidx] = qn
    s_k[tidx] = kn

    prev_g = prev_g_prefetch
    shifted = gate_bf16.to(cutlass.Float32) + dt_value
    a = cute.exp(a_log_value, fastmath=True)
    step = cutlass.Float32(0.0)
    if bounded:
        step = lower_bound * cute.div(
            cutlass.Float32(1.0),
            cutlass.Float32(1.0) + cute.exp(-(a * shifted), fastmath=True),
            full=True,
        )
    else:
        softplus = shifted
        if shifted <= cutlass.Float32(20.0):
            softplus = cute.log(
                cutlass.Float32(1.0) + cute.exp(shifted, fastmath=True),
                fastmath=True,
            )
        step = -a * softplus
    cumulative = prev_g + step
    s_cumulative[tidx] = cumulative
    decay = cute.exp(cumulative, fastmath=True)
    dyn_k = kn.to(cutlass.Float32) * decay
    dyn_q = qn.to(cutlass.Float32) * decay

    # Shared outer scale and kn.qn dot in one block reduction.
    local_max = cute.arch.fmax(cute.abs(dyn_k), cute.abs(dyn_q))
    wmax = cute.arch.warp_reduction_max(local_max)
    wdot = _warp_sum_products(kn.to(cutlass.Float32), qn.to(cutlass.Float32))
    if lane == 0:
        s_red[warp] = wmax
        s_red[4 + warp] = wdot
    cute.arch.barrier()
    if warp == 0:
        mr = cutlass.Float32(0.0)
        dr = cutlass.Float32(0.0)
        if lane < 4:
            mr = s_red[lane]
            dr = s_red[4 + lane]
        mr = cute.arch.warp_reduction_max(mr)
        dr = cute.arch.warp_reduction_sum(dr)
        if lane == 0:
            s_scalar[2] = mr * cutlass.Float32(1.0 / (6.0 * 448.0))
            s_scalar[3] = dr
    cute.arch.barrier()

    outer = s_scalar[2]
    safe_outer = outer
    if outer == cutlass.Float32(0.0):
        safe_outer = cutlass.Float32(1.0)
    # Each half-warp is one 16-element FP4 scale group.
    group_max_k = cute.arch.warp_reduction_max(cute.abs(dyn_k), threads_in_group=16)
    group_max_q = cute.arch.warp_reduction_max(cute.abs(dyn_q), threads_in_group=16)
    for ii in cutlass.range_constexpr(8):
        zero_idx = tidx + ii * 128
        if zero_idx < 126 * 8:
            zero_row = zero_idx // 8 + 2
            zero_group = zero_idx % 8
            zero_offset = (
                (zero_group // 4) * 512
                + (zero_row % 32) * 16
                + (zero_row // 32) * 4
                + (zero_group % 4)
            )
            mma_sfb.subview(zero_offset).store(cutlass.Float8E4M3FN(0.0))
    sfk_raw = cute.div(group_max_k, cutlass.Float32(6.0) * safe_outer, full=True)
    sfq_raw = cute.div(group_max_q, cutlass.Float32(6.0) * safe_outer, full=True)
    sfk_e4 = cute.arch.fmin(sfk_raw, cutlass.Float32(448.0)).to(cutlass.Float8E4M3FN)
    sfq_e4 = cute.arch.fmin(sfq_raw, cutlass.Float32(448.0)).to(cutlass.Float8E4M3FN)
    if (lane % 16) == 0:
        group = warp * 2 + lane // 16
        sf_offset_k = (group // 4) * 512 + (group % 4)
        sf_offset_q = sf_offset_k + 16
        mma_sfb.subview(sf_offset_k).store(sfk_e4)
        mma_sfb.subview(sf_offset_q).store(sfq_e4)

    sfk = sfk_e4.to(cutlass.Float32)
    sfq = sfq_e4.to(cutlass.Float32)
    denom_k = sfk * outer
    denom_q = sfq * outer
    normalized_k = cutlass.Float32(0.0)
    normalized_q = cutlass.Float32(0.0)
    if denom_k != cutlass.Float32(0.0):
        normalized_k = cute.div(dyn_k, denom_k, full=True)
    if denom_q != cutlass.Float32(0.0):
        normalized_q = cute.div(dyn_q, denom_q, full=True)

    # Native NVFP4 checkpoint projection. Operands use an s128b K-major
    # image padded privately to 256 logical K; only the first 128 K values
    # and two of the hardware-minimum eight N rows participate.
    swizzle_s128 = cutlass.Swizzle.from_name("s128b")
    swizzle_s64 = cutlass.Swizzle.from_name("s64b")

    # Dynamic K/Q pair: exact RNE satfinite E2M1x2, earlier K in low nibble.
    e2m1_type = mlir_ir.TypeAttr.get(cutlass.Float4E2M1FN.mlir_type)
    odd_k = cute.arch.shuffle_sync_down(normalized_k, 1)
    odd_q = cute.arch.shuffle_sync_down(normalized_q, 1)
    if (lane % 2) == 0:
        packed_k_ir = prims.dialect.convert_f32x2_to_f4x2(odd_k, normalized_k, e2m1_type)
        packed_q_ir = prims.dialect.convert_f32x2_to_f4x2(odd_q, normalized_q, e2m1_type)
        byte = tidx // 2
        (mma_b.subview(byte)).data_ptr().apply_swizzle(swizzle_s64).store(cutlass.Int8(packed_k_ir))
        (mma_b.subview(64 + byte)).data_ptr().apply_swizzle(swizzle_s64).store(
            cutlass.Int8(packed_q_ir)
        )
    if tidx < 64:
        # Private N padding rows must be true zero regardless of their scales.
        for padding_row in cutlass.range_constexpr(2, 8):
            (mma_b.subview(padding_row * 64 + tidx)).data_ptr().apply_swizzle(swizzle_s64).store(
                cutlass.Int8(0)
            )
    # Repack plain row-major E4M3 scales into two 128x4 scale atoms.
    for group in cutlass.range_constexpr(8):
        sf_offset = (group // 4) * 512 + (tidx % 32) * 16 + (tidx // 32) * 4 + (group % 4)
        mma_sfa.subview(sf_offset).store(state_scale_vec[group])
    while not prims.mbarrier_try_wait_parity(state_mbar, cutlass.Int32(0)):
        pass
    cute.arch.fence_view_async_shared()
    prims.barrier_cta_sync(0)

    if warp == 0:
        idesc = prims.Tcgen05MxOmmaInstrDesc.build(
            a_dtype=cutlass.Float4E2M1FN,
            b_dtype=cutlass.Float4E2M1FN,
            scale_format=0,
            n_dim=8,
            m_dim=128,
        )
        desc_a = prims.Tcgen05SmemDesc.build(
            mma_a,
            leading_byte_offset=0,
            stride_byte_offset=512,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_64B,
        )
        desc_b = prims.Tcgen05SmemDesc.build(
            mma_b,
            leading_byte_offset=0,
            stride_byte_offset=512,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_64B,
        )
        desc_sfa = prims.Tcgen05SmemDesc.build(
            mma_sfa,
            leading_byte_offset=16,
            stride_byte_offset=128,
            layout=0,
        )
        desc_sfb = prims.Tcgen05SmemDesc.build(
            mma_sfb,
            leading_byte_offset=16,
            stride_byte_offset=128,
            layout=0,
        )
        base_col = tmem_raw & cutlass.Int32(0xFFFF)
        base_row = tmem_raw >> cutlass.Int32(16)
        acc_addr = (base_row << cutlass.Int32(16)) | base_col
        sfa_addr = (base_row << cutlass.Int32(16)) | (base_col + cutlass.Int32(8))
        sfb_addr = (base_row << cutlass.Int32(16)) | (base_col + cutlass.Int32(16))
        acc_ptr = cutlass.inttoptr(acc_addr, 6, cutlass.Float32)
        sf_shape, sf_multicast = prims.S2TCopyMode.S2T_32x128b_WARPX4
        if prims.elect_sync():
            for kblock in cutlass.range_constexpr(2):
                sfa_ptr = cutlass.inttoptr(sfa_addr + kblock * 4, 6, cutlass.Int32)
                sfb_ptr = cutlass.inttoptr(sfb_addr + kblock * 4, 6, cutlass.Int32)
                prims.tcgen05_cp(
                    sf_shape,
                    sfa_ptr,
                    desc_sfa.advance_start_address(kblock * 512),
                    multicast=sf_multicast,
                )
                prims.tcgen05_cp(
                    sf_shape,
                    sfb_ptr,
                    desc_sfb.advance_start_address(kblock * 512),
                    multicast=sf_multicast,
                )
            for kblock in cutlass.range_constexpr(2):
                prims.tcgen05_mma_block_scale(
                    prims.MMABlockScaleKind.MXF4NVF4,
                    prims.CTAGroup.CTA_1,
                    acc_ptr,
                    desc_a.advance_start_address(kblock * 32),
                    desc_b.advance_start_address(kblock * 32),
                    idesc,
                    enable_input_d=(kblock != 0),
                    scale_a=cutlass.inttoptr(sfa_addr + kblock * 4, 6, cutlass.Int32),
                    scale_b=cutlass.inttoptr(sfb_addr + kblock * 4, 6, cutlass.Int32),
                    scale_vec_size=prims.Tcgen05MMABlockScale.BLOCK16,
                )
            prims.tcgen05_commit(acc_mbar, group=prims.CTAGroup.CTA_1)

    if cutlass.const_expr(not mtp):
        if length > 0:
            while not prims.mbarrier_try_wait_parity(history_load_mbar, cutlass.Int32(0)):
                pass

    # First BF16 MMA: adjusted K [16,128] @ [kn,qn]^T. Two s128b
    # K64 panels avoid any padding in the instruction's K dimension.
    dense_col = tidx
    if tidx >= 64:
        dense_col = tidx - 64
    for hist in cutlass.range_constexpr(N):
        adjusted_bf16 = cutlass.BFloat16(0.0)
        if hist < accepted_length:
            if cutlass.const_expr(mtp):
                cached_k = hk[(live, head, hist, tidx)].to(cutlass.Float32)
                cached_g = hg[(live, head, hist, tidx)]
            else:
                cached_k = cache_hk[hist * K + tidx].to(cutlass.Float32)
                cached_g = cache_hg[hist * K + tidx]
            adjusted_bf16 = (cached_k * cute.exp(cumulative - cached_g, fastmath=True)).to(
                cutlass.BFloat16
            )
        if tidx < 64:
            dst = (dense_a0.subview(hist * 64 + dense_col)).data_ptr().apply_swizzle(swizzle_s128)
            _store_bf16(dst, adjusted_bf16)
        else:
            dst = (dense_a1.subview(hist * 64 + dense_col)).data_ptr().apply_swizzle(swizzle_s128)
            _store_bf16(dst, adjusted_bf16)
    if tidx < 64:
        _store_bf16((dense_b0.subview(dense_col)).data_ptr().apply_swizzle(swizzle_s128), kn)
        _store_bf16((dense_b0.subview(64 + dense_col)).data_ptr().apply_swizzle(swizzle_s128), qn)
    else:
        _store_bf16((dense_b1.subview(dense_col)).data_ptr().apply_swizzle(swizzle_s128), kn)
        _store_bf16((dense_b1.subview(64 + dense_col)).data_ptr().apply_swizzle(swizzle_s128), qn)
    cute.arch.fence_view_async_shared()
    prims.barrier_cta_sync(0)

    if warp == 0:
        dense_idesc = prims.Tcgen05InstrDesc.build(
            c_dtype=cutlass.Float32,
            a_dtype=cutlass.BFloat16,
            b_dtype=cutlass.BFloat16,
            n_dim=8,
            m_dim=128,
        )
        da0 = prims.Tcgen05SmemDesc.build(
            dense_a0, leading_byte_offset=128, stride_byte_offset=1024, layout=2
        )
        da1 = prims.Tcgen05SmemDesc.build(
            dense_a1, leading_byte_offset=128, stride_byte_offset=1024, layout=2
        )
        db0 = prims.Tcgen05SmemDesc.build(
            dense_b0, leading_byte_offset=128, stride_byte_offset=1024, layout=2
        )
        db1 = prims.Tcgen05SmemDesc.build(
            dense_b1, leading_byte_offset=128, stride_byte_offset=1024, layout=2
        )
        base_col_dense = tmem_raw & cutlass.Int32(0xFFFF)
        base_row_dense = tmem_raw >> cutlass.Int32(16)
        coeff_addr = (base_row_dense << cutlass.Int32(16)) | (base_col_dense + cutlass.Int32(24))
        coeff_ptr = cutlass.inttoptr(coeff_addr, 6, cutlass.Float32)
        if prims.elect_sync():
            for kb in cutlass.range_constexpr(4):
                prims.tcgen05_mma(
                    prims.Tcgen05MMAKind.F16,
                    prims.CTAGroup.CTA_1,
                    coeff_ptr,
                    da0.advance_start_address(kb * 32),
                    db0.advance_start_address(kb * 32),
                    dense_idesc,
                    kb != 0,
                )
            for kb in cutlass.range_constexpr(4):
                prims.tcgen05_mma(
                    prims.Tcgen05MMAKind.F16,
                    prims.CTAGroup.CTA_1,
                    coeff_ptr,
                    da1.advance_start_address(kb * 32),
                    db1.advance_start_address(kb * 32),
                    dense_idesc,
                    True,
                )
            prims.tcgen05_commit(coeff_mbar, group=prims.CTAGroup.CTA_1)

    # Drain the independent checkpoint result while the coefficient MMA runs.
    while not prims.mbarrier_try_wait_parity(acc_mbar, cutlass.Int32(0)):
        pass
    base_col = tmem_raw & cutlass.Int32(0xFFFF)
    base_row = tmem_raw >> cutlass.Int32(16)
    tmem_row = base_row + warp * cutlass.Int32(32)
    tmem_addr = (tmem_row << cutlass.Int32(16)) | base_col
    checkpoint_vec = prims.tcgen05_ld(
        "32x32b",
        cutlass.inttoptr(tmem_addr, 6, cutlass.Float32),
        num=8,
    )
    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
    checkpoint_rescale = state_outer * outer
    cp_k = checkpoint_vec[0] * checkpoint_rescale
    cp_q = checkpoint_vec[1] * checkpoint_rescale

    while not prims.mbarrier_try_wait_parity(coeff_mbar, cutlass.Int32(0)):
        pass
    coeff_row = (tmem_raw >> cutlass.Int32(16)) + warp * cutlass.Int32(32)
    coeff_col = (tmem_raw & cutlass.Int32(0xFFFF)) + cutlass.Int32(24)
    coeff_addr = (coeff_row << cutlass.Int32(16)) | coeff_col
    coeff_vec = prims.tcgen05_ld("32x32b", cutlass.inttoptr(coeff_addr, 6, cutlass.Float32), num=8)
    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
    coeff_k = cutlass.BFloat16(0.0)
    coeff_q = cutlass.BFloat16(0.0)
    if tidx < N:
        coeff_k = coeff_vec[0].to(cutlass.BFloat16)
        coeff_q = coeff_vec[1].to(cutlass.BFloat16)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)

    # Second BF16 MMA: U^T [128,16] @ coefficients [16,2]. Reuse the
    # first dense panels and the coefficient accumulator columns.
    for hist in cutlass.range_constexpr(N):
        uval = cutlass.BFloat16(0.0)
        if hist < accepted_length:
            if cutlass.const_expr(mtp):
                uval = hu[(live, head, hist, tidx)]
            else:
                uval = cache_hu[hist * K + tidx]
        dst = (dense_a0.subview(tidx * 64 + hist)).data_ptr().apply_swizzle(swizzle_s128)
        _store_bf16(dst, uval)
    if tidx < N:
        _store_bf16(
            (dense_b0.subview(tidx)).data_ptr().apply_swizzle(swizzle_s128),
            coeff_k,
        )
        _store_bf16(
            (dense_b0.subview(64 + tidx)).data_ptr().apply_swizzle(swizzle_s128),
            coeff_q,
        )
    cute.arch.fence_view_async_shared()
    prims.barrier_cta_sync(0)

    if warp == 0:
        dense_idesc2 = prims.Tcgen05InstrDesc.build(
            c_dtype=cutlass.Float32,
            a_dtype=cutlass.BFloat16,
            b_dtype=cutlass.BFloat16,
            n_dim=8,
            m_dim=128,
        )
        da_hist = prims.Tcgen05SmemDesc.build(
            dense_a0,
            leading_byte_offset=128,
            stride_byte_offset=1024,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )
        db_hist = prims.Tcgen05SmemDesc.build(
            dense_b0,
            leading_byte_offset=128,
            stride_byte_offset=1024,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )
        base_col_hist = tmem_raw & cutlass.Int32(0xFFFF)
        base_row_hist = tmem_raw >> cutlass.Int32(16)
        history_addr = (base_row_hist << cutlass.Int32(16)) | (base_col_hist + cutlass.Int32(24))
        if prims.elect_sync():
            prims.tcgen05_mma(
                prims.Tcgen05MMAKind.F16,
                prims.CTAGroup.CTA_1,
                cutlass.inttoptr(history_addr, 6, cutlass.Float32),
                da_hist,
                db_hist,
                dense_idesc2,
                False,
            )
            prims.tcgen05_commit(history_mbar, group=prims.CTAGroup.CTA_1)

    while not prims.mbarrier_try_wait_parity(history_mbar, cutlass.Int32(0)):
        pass
    history_col = base_col + cutlass.Int32(24)
    history_addr = (tmem_row << cutlass.Int32(16)) | history_col
    history_vec = prims.tcgen05_ld(
        "32x32b", cutlass.inttoptr(history_addr, 6, cutlass.Float32), num=8
    )
    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
    hist_k = history_vec[0]
    hist_q = history_vec[1]
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    prims.barrier_cta_sync(0)
    if warp == 0:
        prims.tcgen05_dealloc(cutlass.inttoptr(tmem_raw, 6, cutlass.Float32), 32, group="cta_1")

    # Candidates after the accepted prefix are private to this CTA until sampling.
    if cutlass.const_expr(mtp):
        for prior in cutlass.range(token):
            hist = accepted_length + prior
            adjusted = (
                (
                    hk[(live, head, hist, tidx)].to(cutlass.Float32)
                    * cute.exp(cumulative - hg[(live, head, hist, tidx)], fastmath=True)
                )
                .to(cutlass.BFloat16)
                .to(cutlass.Float32)
            )
            ck = _warp_sum_products(adjusted, kn.to(cutlass.Float32))
            cq = _warp_sum_products(adjusted, qn.to(cutlass.Float32))
            if lane == 0:
                s_red[warp] = ck
                s_red[4 + warp] = cq
            cute.arch.barrier()
            if tidx == 0:
                s_scalar[6] = (s_red[0] + s_red[1]) + (s_red[2] + s_red[3])
                s_scalar[7] = (s_red[4] + s_red[5]) + (s_red[6] + s_red[7])
            cute.arch.barrier()
            uk = hu[(live, head, hist, tidx)].to(cutlass.Float32)
            hist_k = cute.fma(uk, s_scalar[6].to(cutlass.BFloat16).to(cutlass.Float32), hist_k)
            hist_q = cute.fma(uk, s_scalar[7].to(cutlass.BFloat16).to(cutlass.Float32), hist_q)

    beta_v = beta_bf16.to(cutlass.Float32)
    beta_sigmoid = cute.div(
        cutlass.Float32(1.0),
        cutlass.Float32(1.0) + cute.exp(-beta_v, fastmath=True),
        full=True,
    )
    update = beta_sigmoid * (v_bf16.to(cutlass.Float32) - cp_k - hist_k)
    core = cp_q + hist_q + update * s_scalar[3]
    core_bf16 = core.to(cutlass.BFloat16)
    s_core[tidx] = core_bf16

    hk[(live, head, length, tidx)] = kn
    hu[(live, head, length, tidx)] = update.to(cutlass.BFloat16)
    hg[(live, head, length, tidx)] = cumulative
    cute.arch.barrier()

    # RMSNorm is intentionally after the BF16 core boundary.
    core_f = s_core[tidx].to(cutlass.Float32)
    rms0 = _warp_sum_squares(s_core[lane].to(cutlass.Float32))
    rms1 = _warp_sum_squares(s_core[lane + 32].to(cutlass.Float32))
    rms2 = _warp_sum_squares(s_core[lane + 64].to(cutlass.Float32))
    rms3 = _warp_sum_squares(s_core[lane + 96].to(cutlass.Float32))
    total = (rms0 + rms2) + (rms1 + rms3)
    rstd = cutlass.Float32(0.0)
    if lane == 0:
        variance = cute.div(total, cutlass.Float32(128.0), full=True)
        rstd = cute.div(
            cutlass.Float32(1.0),
            cute.sqrt(
                variance + norm_eps,
                approx=True,
                ftz=True,
            ),
            full=True,
        )
    rstd = cute.arch.shuffle_sync(rstd, 0)
    zv = z_bf16.to(cutlass.Float32)
    zg = cute.div(
        cutlass.Float32(1.0),
        cutlass.Float32(1.0) + cute.exp(-zv, fastmath=True),
        full=True,
    )
    result = core_f * rstd * norm_bf16.to(cutlass.Float32) * zg
    out[(row, head, tidx)] = result.to(cutlass.BFloat16)
    if head == 0 and tidx == 0:
        old_lengths[batch] = accepted_length
    cute.arch.barrier()


@cute.kernel
def _kda_scalar_kernel(
    tma_state: cutlass.GridConstant[cuda_exp.TensorMap],
    tma_hk: cutlass.GridConstant[cuda_exp.TensorMap],
    tma_hu: cutlass.GridConstant[cuda_exp.TensorMap],
    tma_hg: cutlass.GridConstant[cuda_exp.TensorMap],
    x,
    conv_state,
    conv_weight,
    gate,
    beta,
    z,
    norm_weight,
    pool,
    slots,
    dummy,
    slot_map,
    hk,
    hu,
    hg,
    lengths,
    a_log,
    dt_bias,
    out,
    old_lengths,
    candidate_x,
    lower_bound: cutlass.Float32,
    norm_eps: cutlass.Float32,
    heads: cutlass.Constexpr,
    bounded: cutlass.Constexpr,
    tokens: cutlass.Constexpr,
    mtp: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    batch = block // heads
    head = block - batch * heads
    if dummy[batch]:
        for token in cutlass.range_constexpr(tokens):
            out[(batch * tokens + token, head, tidx)] = cutlass.BFloat16(0.0)
            if cutlass.const_expr(mtp):
                for section in cutlass.range_constexpr(3):
                    candidate_x[(batch, token, (section * heads + head) * K + tidx)] = (
                        cutlass.BFloat16(0.0)
                    )
        if head == 0 and tidx == 0:
            old_lengths[batch] = cutlass.Int32(0)
    else:
        if cutlass.const_expr(mtp):
            for token in cutlass.range(tokens):
                _kda_scalar_active(
                    tma_state,
                    tma_hk,
                    tma_hu,
                    tma_hg,
                    x,
                    conv_state,
                    conv_weight,
                    gate,
                    beta,
                    z,
                    norm_weight,
                    pool,
                    slots,
                    dummy,
                    slot_map,
                    hk,
                    hu,
                    hg,
                    lengths,
                    a_log,
                    dt_bias,
                    out,
                    old_lengths,
                    candidate_x,
                    lower_bound,
                    norm_eps,
                    heads,
                    bounded,
                    tokens,
                    mtp,
                    token,
                )
        else:
            _kda_scalar_active(
                tma_state,
                tma_hk,
                tma_hu,
                tma_hg,
                x,
                conv_state,
                conv_weight,
                gate,
                beta,
                z,
                norm_weight,
                pool,
                slots,
                dummy,
                slot_map,
                hk,
                hu,
                hg,
                lengths,
                a_log,
                dt_bias,
                out,
                old_lengths,
                candidate_x,
                lower_bound,
                norm_eps,
                heads,
                bounded,
                tokens,
                mtp,
                cutlass.Int32(0),
            )


@cute.jit
def _launch_scalar(
    x,
    conv_state,
    conv_weight,
    gate,
    beta,
    z,
    norm_weight,
    pool,
    slots,
    dummy,
    slot_map,
    hk,
    hu,
    hg,
    lengths,
    a_log,
    dt_bias,
    out,
    old_lengths,
    candidate_x,
    lower_bound: cutlass.Float32,
    norm_eps: cutlass.Float32,
    heads: cutlass.Constexpr,
    bounded: cutlass.Constexpr,
    tokens: cutlass.Constexpr,
    mtp: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    primary_layout = cute.make_layout(
        (64, 128, heads, pool.shape[0]),
        stride=(1, 64, STATE_BYTES, pool.layout.stride[0]),
    )
    primary_tensor = cute.make_tensor(pool.iterator, primary_layout)
    tma_state = cuda_exp.create_tensor_map_tiled_from_view(
        primary_tensor,
        box_dims=(64, 128, 1, 1),
        stride_order=(0, 1, 2, 3),
        swizzle=cuda_exp.TensorMapSwizzle.s64b,
    )
    hk_layout = cute.make_layout(
        (K, N, heads, hk.shape[0]),
        stride=(1, K, hk.layout.stride[1], hk.layout.stride[0]),
    )
    hk_tensor = cute.make_tensor(hk.iterator, hk_layout)
    tma_hk = cuda_exp.create_tensor_map_tiled_from_view(
        hk_tensor,
        box_dims=(K, N, 1, 1),
        stride_order=(0, 1, 2, 3),
        swizzle=cuda_exp.TensorMapSwizzle.none,
    )
    hu_layout = cute.make_layout(
        (K, N, heads, hu.shape[0]),
        stride=(1, K, hu.layout.stride[1], hu.layout.stride[0]),
    )
    hu_tensor = cute.make_tensor(hu.iterator, hu_layout)
    tma_hu = cuda_exp.create_tensor_map_tiled_from_view(
        hu_tensor,
        box_dims=(K, N, 1, 1),
        stride_order=(0, 1, 2, 3),
        swizzle=cuda_exp.TensorMapSwizzle.none,
    )
    hg_layout = cute.make_layout(
        (64, 2, N, heads, hg.shape[0]),
        stride=(1, 64, K, hg.layout.stride[1], hg.layout.stride[0]),
    )
    hg_tensor = cute.make_tensor(hg.iterator, hg_layout)
    tma_hg = cuda_exp.create_tensor_map_tiled_from_view(
        hg_tensor,
        box_dims=(64, 2, N, 1, 1),
        stride_order=(0, 1, 2, 3, 4),
        swizzle=cuda_exp.TensorMapSwizzle.none,
    )
    kernel = _kda_scalar_kernel(
        tma_state,
        tma_hk,
        tma_hu,
        tma_hg,
        x,
        conv_state,
        conv_weight,
        gate,
        beta,
        z,
        norm_weight,
        pool,
        slots,
        dummy,
        slot_map,
        hk,
        hu,
        hg,
        lengths,
        a_log,
        dt_bias,
        out,
        old_lengths,
        candidate_x,
        lower_bound,
        norm_eps,
        heads,
        bounded,
        tokens,
        mtp,
    )
    if cutlass.const_expr(heads == 96 and not mtp):
        kernel.launch(
            grid=[slots.shape[0] * heads, 1, 1],
            block=[THREADS, 1, 1],
            min_blocks_per_mp=6,
            stream=stream,
        )
    else:
        kernel.launch(
            grid=[slots.shape[0] * heads, 1, 1],
            block=[THREADS, 1, 1],
            stream=stream,
        )


def _fake_tensor_args(heads: int, tokens: int = 1, capacity: int = N) -> tuple:
    """Compile once per stable H while B/S/R and all padded strides stay runtime."""
    b = cute.sym_int32(divisibility=1)
    s = cute.sym_int32(divisibility=1)
    r = cute.sym_int32(divisibility=1)

    def row(divisibility: int = 1) -> cute.SymInt:
        return cute.sym_int64(divisibility=divisibility)

    rows = cute.sym_int32(divisibility=1)
    width = heads * K
    channels = 3 * width
    hist_row = heads * capacity * K
    return (
        make_fake_tensor(cutlass.BFloat16, (rows, channels), stride=(row(16), 1)),
        make_fake_tensor(cutlass.BFloat16, (s, channels, 3), stride=(row(16), 3, 1)),
        make_fake_tensor(cutlass.BFloat16, (channels, 4), stride=(4, 1)),
        make_fake_tensor(cutlass.BFloat16, (rows, width), stride=(row(8), 1)),
        make_fake_tensor(cutlass.BFloat16, (rows, heads), stride=(row(1), 1)),
        make_fake_tensor(cutlass.BFloat16, (rows, heads, K), stride=(row(8), K, 1)),
        make_fake_tensor(cutlass.BFloat16, (K,), stride=(1,)),
        make_fake_tensor(cutlass.Uint8, (s, heads, STATE_BYTES), stride=(row(16), STATE_BYTES, 1)),
        make_fake_tensor(cutlass.Int32, (b,), stride=(1,)),
        make_fake_tensor(cutlass.Boolean, (b,), stride=(1,)),
        make_fake_tensor(cutlass.Int32, (s,), stride=(1,)),
        make_fake_tensor(
            cutlass.BFloat16, (r, heads, capacity, K), stride=(hist_row, capacity * K, K, 1)
        ),
        make_fake_tensor(
            cutlass.BFloat16, (r, heads, capacity, K), stride=(hist_row, capacity * K, K, 1)
        ),
        make_fake_tensor(
            cutlass.Float32, (r, heads, capacity, K), stride=(hist_row, capacity * K, K, 1)
        ),
        make_fake_tensor(cutlass.Int32, (r,), stride=(1,)),
        make_fake_tensor(cutlass.Float32, (heads,), stride=(1,)),
        make_fake_tensor(cutlass.Float32, (width,), stride=(1,)),
        make_fake_tensor(cutlass.BFloat16, (rows, heads, K), stride=(width, K, 1)),
        make_fake_tensor(cutlass.Int32, (b,), stride=(1,)),
        make_fake_tensor(cutlass.BFloat16, (b, tokens, channels), stride=(row(1), channels, 1)),
    )


@torch.no_grad()
def fused_replay(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    pool: torch.Tensor,
    slots: torch.Tensor,
    dummy: torch.Tensor,
    slot_map: torch.Tensor,
    hk: torch.Tensor,
    hu: torch.Tensor,
    hg: torch.Tensor,
    lengths: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
    norm_eps: float,
    out: torch.Tensor,
    old_lengths: torch.Tensor,
    *,
    candidate_x: torch.Tensor | None = None,
    tokens_per_step: int = 1,
) -> None:
    """Write BF16 [B*T,H,128] output and immutable [B] history-length snapshots."""
    heads = a_log.numel()
    bounded = lower_bound is not None
    lower_bound = 0.0 if lower_bound is None else lower_bound
    mtp = candidate_x is not None
    if candidate_x is None:
        candidate_x = x.unsqueeze(1)
    key = (heads, bounded, mtp, tokens_per_step, hk.shape[2])
    tensor_args = (
        x,
        conv_state,
        conv_weight,
        gate,
        beta,
        z,
        norm_weight,
        pool,
        slots,
        dummy,
        slot_map,
        hk,
        hu,
        hg,
        lengths,
        a_log,
        dt_bias,
        out,
        old_lengths,
        candidate_x,
    )
    if key not in _compiled:
        compile_args = _fake_tensor_args(heads, tokens_per_step, hk.shape[2])
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        _compiled[key] = cute.compile(
            _launch_scalar,
            *compile_args,
            float(lower_bound),
            float(norm_eps),
            heads,
            bounded,
            tokens_per_step,
            mtp,
            stream,
            options="--enable-tvm-ffi",
        )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    _compiled[key](*tensor_args, float(lower_bound), float(norm_eps), stream)
