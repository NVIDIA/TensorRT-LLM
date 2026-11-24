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
"""Decode GEMV for Kimi K3 projections: ``y[M, N] = x[M, K] @ W[N, K]^T``, M <= 8, bf16 in and out.

tcgen05 with the weight as the MMA's A operand (128 output features per CTA) and the activation as B
(8 token columns), fp32 accumulation in TMEM. For the short-K projections (K <= 6 * 128: the MoE
tail, the attention o_proj) one CTA owns one 128-row weight tile over the whole K extent, and every
weight k-tile of that CTA gets its own shared-memory stage. The weight TMA for all stages is issued
at launch, before ``griddepcontrol.wait``: launched early under PDL, the kernel streams its whole
weight while its predecessor still runs, and after the wait only the activation (8 x K) remains to
load. The activation box is always 8 rows; rows past M are zero-filled by the TMA, so one compiled
kernel serves every M <= 8.

Warps: 0 weight TMA, 1 activation TMA (after the grid dependency), 2 TMEM allocation and MMA, 3 idle,
4-7 epilogue (TMEM -> registers -> bf16 stores; warp 4 + i reads TMEM lanes 32i..32i+31).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cutlass.experimental import primitives as prims

CTA_M = 128  # weight rows (output features) per CTA = the MMA's M
MMA_N = 8  # token columns
CTA_K = 128  # one k-tile: two 64-element halves of the 128-byte swizzle
MMA_K = 16
TMA_K_BOX = 64
TMA_COPY_ITERS = CTA_K // TMA_K_BOX
K_BLOCKS_PER_HALF = TMA_K_BOX // MMA_K
MAX_K_TILES = 6  # 6 x (32 KB weight + 2 KB activation) of shared memory
THREADS = 256
TMEM_COLS = 32
ELEM_BYTES = 2
EVICT_FIRST = 0x12F0000000000000  # createpolicy.fractional.L2::evict_first, fraction 1.0 (sm_100)

# Shared-memory descriptor strides for the 128-byte swizzle, in 16-byte units.
LEADING = 16
STRIDE = 8 * TMA_K_BOX * ELEM_BYTES
A_HALF_ELEMS = CTA_M * TMA_K_BOX
B_HALF_ELEMS = MMA_N * TMA_K_BOX
STEP = (MMA_K * ELEM_BYTES) >> 4
A_BOX = A_HALF_ELEMS >> 3
B_BOX = B_HALF_ELEMS >> 3
STAGE_A = (CTA_M * CTA_K * ELEM_BYTES) >> 4
STAGE_B = (MMA_N * CTA_K * ELEM_BYTES) >> 4

io_dtype = cutlass.BFloat16


def num_k_tiles(k_in: int) -> int:
    return (k_in + CTA_K - 1) // CTA_K


def supports(n_out: int, k_in: int) -> bool:
    """Shapes this kernel runs: whole 128-row tiles and at most MAX_K_TILES k-tiles."""
    return n_out % CTA_M == 0 and k_in % TMA_K_BOX == 0 and num_k_tiles(k_in) <= MAX_K_TILES


@cute.kernel
def k3_decode_gemv_kernel(
    tma_desc_w: cutlass.GridConstant[cuda.TensorMap],  # W [N, K] bf16, 5-D, one call per k-tile
    tma_desc_x: cutlass.GridConstant[
        cuda.TensorMap
    ],  # x [M, *] bf16, box 8 x 64: k-tiles [0, x_tiles)
    tma_desc_x2: cutlass.GridConstant[
        cuda.TensorMap
    ],  # x2 [M, *] bf16, box 8 x 64: k-tiles [x_tiles, K/128)
    y: cutlass.Array,  # [M * N] bf16, token-major
    rms_src: cutlass.Array,  # int32 words of the bf16 [M, rms_cols] rows whose RMS scales accumulator 0
    num_tokens: cutlass.Int32,
    x_col0: cutlass.Int32,  # column of x where k-tile 0 starts
    eps: cutlass.Float32,
    n_out: cutlass.Constexpr[int],
    k_in: cutlass.Constexpr[int],
    x_tiles: cutlass.Constexpr[int],
    split_acc: cutlass.Constexpr[bool],  # the x2 k-tiles accumulate into a second TMEM accumulator
    rms_cols: cutlass.Constexpr[int],  # > 0: y = rsqrt(mean(rms_src row^2) + eps) * acc0 + acc1
    trigger_early: cutlass.Constexpr[bool],
):
    k_tiles = num_k_tiles(k_in)
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    warp_id = cute.arch.warp_idx()
    tma_ptr_w = tma_desc_w.get_ptr()
    tma_ptr_x = tma_desc_x.get_ptr()
    tma_ptr_x2 = tma_desc_x2.get_ptr()
    m_offset = bx * cutlass.Int32(CTA_M)

    smem_a = cutlass.Array(
        io_dtype, k_tiles * CTA_M * CTA_K, space=cutlass.AddressSpace.smem, alignment=1024
    )
    smem_b = cutlass.Array(
        io_dtype, k_tiles * MMA_N * CTA_K, space=cutlass.AddressSpace.smem, alignment=1024
    )
    weight_full = cutlass.Array(
        cutlass.Int64, k_tiles, space=cutlass.AddressSpace.smem, alignment=8
    )
    act_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    acc_done = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
    row_scale = cutlass.Array(cutlass.Float32, MMA_N, space=cutlass.AddressSpace.smem, alignment=16)

    if warp_id == 0:
        prims.prefetch_tensormap(tma_ptr_w)
        prims.prefetch_tensormap(tma_ptr_x)
        if cutlass.const_expr(x_tiles < k_tiles):
            prims.prefetch_tensormap(tma_ptr_x2)
        if prims.elect_sync():
            for k in cutlass.range_constexpr(k_tiles):
                prims.mbarrier_init(weight_full.subview(k), 1)
            prims.mbarrier_init(act_full, 1)
            prims.mbarrier_init(acc_done, 1)
    if warp_id == 2:
        prims.tcgen05_alloc(tmem_ptr_i32, TMEM_COLS)
        prims.tcgen05_relinquish_alloc_permit()
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync(0)
    tmem_ptr = cutlass.inttoptr(tmem_ptr_i32.load(), 6, cutlass.Int32)

    if warp_id == 0:
        # The whole weight slice of this CTA, ahead of the grid dependency.
        if prims.elect_sync():
            for k in cutlass.range_constexpr(k_tiles):
                prims.mbarrier_arrive_expect_tx(weight_full.subview(k), CTA_M * CTA_K * ELEM_BYTES)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    smem_a.subview(k * CTA_M * CTA_K),
                    tma_ptr_w,
                    (
                        cutlass.Int32(0),
                        m_offset,
                        cutlass.Int32(k * TMA_COPY_ITERS),
                        cutlass.Int32(0),
                        cutlass.Int32(0),
                    ),
                    weight_full.subview(k),
                    l2_cache_hint=EVICT_FIRST,
                )
        if cutlass.const_expr(trigger_early):
            # Dependents may launch now; they wait for this whole grid before reading y.
            prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
    elif warp_id == 1:
        # The activations are written by the predecessor.
        prims.griddepcontrol(prims.GridDepAction.WAIT)
        if prims.elect_sync():
            prims.mbarrier_arrive_expect_tx(act_full, k_tiles * MMA_N * CTA_K * ELEM_BYTES)
            for k in cutlass.range_constexpr(k_tiles):
                for half in cutlass.range_constexpr(TMA_COPY_ITERS):
                    if cutlass.const_expr(k < x_tiles):
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            smem_b.subview(k * MMA_N * CTA_K + half * B_HALF_ELEMS),
                            tma_ptr_x,
                            (
                                x_col0 + cutlass.Int32(k * CTA_K + half * TMA_K_BOX),
                                cutlass.Int32(0),
                            ),
                            act_full,
                        )
                    else:
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            smem_b.subview(k * MMA_N * CTA_K + half * B_HALF_ELEMS),
                            tma_ptr_x2,
                            (
                                cutlass.Int32((k - x_tiles) * CTA_K + half * TMA_K_BOX),
                                cutlass.Int32(0),
                            ),
                            act_full,
                        )
    elif warp_id == 2:
        idesc = prims.Tcgen05InstrDesc.build(
            c_dtype=cutlass.Float32, a_dtype=io_dtype, b_dtype=io_dtype, n_dim=MMA_N, m_dim=CTA_M
        )
        desc_a_base = prims.Tcgen05SmemDesc.build(
            start_address=smem_a, leading_byte_offset=LEADING, stride_byte_offset=STRIDE,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )  # fmt: skip
        desc_b_base = prims.Tcgen05SmemDesc.build(
            start_address=smem_b, leading_byte_offset=LEADING, stride_byte_offset=STRIDE,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )  # fmt: skip
        tmem_acc1 = tmem_ptr
        if cutlass.const_expr(split_acc):
            tmem_acc1 = cutlass.inttoptr(
                tmem_ptr_i32.load() + cutlass.Int32(MMA_N), 6, cutlass.Int32
            )
        while not cute.arch.mbarrier_try_wait(act_full.data_ptr(), 0):
            pass
        for k in cutlass.range_constexpr(k_tiles):
            second = split_acc and k >= x_tiles
            first_tile = k == 0 or (split_acc and k == x_tiles)
            while not cute.arch.mbarrier_try_wait(weight_full.subview(k).data_ptr(), 0):
                pass
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            for kb in cutlass.range_constexpr(TMA_COPY_ITERS * K_BLOCKS_PER_HALF):
                box = kb // K_BLOCKS_PER_HALF
                within = kb % K_BLOCKS_PER_HALF
                desc_a = desc_a_base + (k * STAGE_A + box * A_BOX + within * STEP)
                desc_b = desc_b_base + (k * STAGE_B + box * B_BOX + within * STEP)
                if prims.elect_sync():
                    prims.tcgen05_mma(
                        prims.Tcgen05MMAKind.F16, prims.CTAGroup.CTA_1, tmem_acc1 if second else tmem_ptr,
                        desc_a, desc_b, idesc, not (first_tile and kb == 0),
                    )  # fmt: skip
        if prims.elect_sync():
            prims.tcgen05_commit(acc_done)
    elif warp_id >= 4:
        lane = tx % 32
        if cutlass.const_expr(rms_cols > 0):
            # Per-token RMS of the rms_src rows while the MMA runs: epilogue warp w owns tokens 2w, 2w+1.
            prims.griddepcontrol(prims.GridDepAction.WAIT)
            for j in cutlass.range_constexpr(MMA_N // 4):
                t = (warp_id % 4) * cutlass.Int32(MMA_N // 4) + cutlass.Int32(j)
                sum_sq = cutlass.Float32(0.0)
                if t < num_tokens:
                    for v in cutlass.range_constexpr(rms_cols // (8 * 32)):
                        words = rms_src.load(
                            idx=t * cutlass.Int32(rms_cols // 2)
                            + cutlass.Int32(v * 32 * 4)
                            + lane * cutlass.Int32(4),
                            vector_size=4,
                            alignment=16,
                        )
                        for i in cutlass.range_constexpr(4):
                            lo = (words[i] << cutlass.Int32(16)).bitcast(cutlass.Float32)
                            hi = (words[i] & cutlass.Int32(-65536)).bitcast(cutlass.Float32)
                            sum_sq = sum_sq + lo * lo + hi * hi
                for offset in (16, 8, 4, 2, 1):
                    sum_sq = sum_sq + cute.arch.shuffle_sync_bfly(sum_sq, offset=offset)
                if lane == 0:
                    row_scale.store(
                        cute.math.rsqrt(sum_sq * cutlass.Float32(1.0 / rms_cols) + eps), idx=t
                    )
            prims.barrier_cta_sync(1, thread_count=128)
        while not cute.arch.mbarrier_try_wait(acc_done.data_ptr(), 0):
            pass
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        acc0 = prims.tcgen05_ld(
            "32x32b", cutlass.inttoptr(tmem_ptr_i32.load(), 6, cutlass.Float32), num=MMA_N
        )
        acc1 = acc0
        if cutlass.const_expr(split_acc):
            acc1 = prims.tcgen05_ld(
                "32x32b",
                cutlass.inttoptr(tmem_ptr_i32.load() + cutlass.Int32(MMA_N), 6, cutlass.Float32),
                num=MMA_N,
            )
        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
        row = (warp_id % 4) * 32 + lane
        n = m_offset + row
        for t in cutlass.range_constexpr(MMA_N):
            if cutlass.Int32(t) < num_tokens:
                value = cutlass.Float32(acc0[t])
                if cutlass.const_expr(rms_cols > 0):
                    value = value * row_scale.load(idx=t)
                if cutlass.const_expr(split_acc):
                    value = value + cutlass.Float32(acc1[t])
                y.store(cutlass.BFloat16(value), idx=cutlass.Int32(t * n_out) + n)
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        prims.barrier_cta_sync(1, thread_count=128)
        if warp_id == 4:
            prims.tcgen05_dealloc(tmem_ptr, TMEM_COLS)


# =============================================================================
# Long-K split-K variant (e.g. the KDA qkvg projection, 3208 x 7168): the fuse_o_linear_ra structure.
# A cluster of SPLIT CTAs shares one 128-row weight tile; rank r streams the interleaved k-tiles r,
# r + SPLIT, ... through a RING_STAGES ring. Before the grid dependency each CTA fills its ring from HBM and
# prefetches the rest of its k-tiles into L2. Rows [32r, 32r + 32) of the tile are owned by rank r: every lane
# of the other ranks' epilogue warp r stores its fp32 partials into rank r's shared memory and arrives on its
# barrier (release, cluster scope); the owner waits (acquire, cluster scope), adds the SPLIT partials in rank
# order and stores bf16. (fuse_o_linear_ra pushes with st.async, which this DSL build does not expose.)
# =============================================================================
SPLIT = 4
RING_STAGES = 6
ROWS_PER_RANK = CTA_M // SPLIT


def splitk_supports(n_out: int, k_in: int) -> bool:
    k_tiles = num_k_tiles(k_in)
    return k_in % CTA_K == 0 and k_tiles > MAX_K_TILES and k_tiles % SPLIT == 0 and n_out > 0


@cute.kernel
def k3_decode_gemv_splitk_kernel(
    tma_desc_w: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_x: cutlass.GridConstant[cuda.TensorMap],
    y: cutlass.Array,  # [M * N] bf16, token-major
    num_tokens: cutlass.Int32,
    n_out: cutlass.Constexpr[int],
    k_in: cutlass.Constexpr[int],
    trigger_early: cutlass.Constexpr[bool],
):
    k_tiles = num_k_tiles(k_in)
    my_tiles = k_tiles // SPLIT
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    warp_id = cute.arch.warp_idx()
    rank = cute.arch.block_idx_in_cluster()
    tma_ptr_w = tma_desc_w.get_ptr()
    tma_ptr_x = tma_desc_x.get_ptr()
    m_offset = (bx // cutlass.Int32(SPLIT)) * cutlass.Int32(CTA_M)

    smem_a = cutlass.Array(
        io_dtype, RING_STAGES * CTA_M * CTA_K, space=cutlass.AddressSpace.smem, alignment=1024
    )
    smem_b = cutlass.Array(
        io_dtype, RING_STAGES * MMA_N * CTA_K, space=cutlass.AddressSpace.smem, alignment=1024
    )
    tma_done = cutlass.Array(
        cutlass.Int64, RING_STAGES, space=cutlass.AddressSpace.smem, alignment=8
    )
    mma_done = cutlass.Array(
        cutlass.Int64, RING_STAGES, space=cutlass.AddressSpace.smem, alignment=8
    )
    acc_done = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    reduce_ready = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
    # Mailbox of the rows this rank owns: [source rank][row][token] fp32 (its own slot stays unused).
    mailbox = cutlass.Array(
        cutlass.Float32,
        SPLIT * ROWS_PER_RANK * MMA_N,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )

    if warp_id == 0:
        prims.prefetch_tensormap(tma_ptr_w)
        prims.prefetch_tensormap(tma_ptr_x)
        if prims.elect_sync():
            for s in cutlass.range_constexpr(RING_STAGES):
                prims.mbarrier_init(tma_done.subview(s), 2)  # the weight and the activation TMA
                prims.mbarrier_init(mma_done.subview(s), 1)
            prims.mbarrier_init(acc_done, 1)
            prims.mbarrier_init(reduce_ready, (SPLIT - 1) * 32)  # every lane of the 3 pushing warps
    if warp_id == 2:
        prims.tcgen05_alloc(tmem_ptr_i32, TMEM_COLS)
        prims.tcgen05_relinquish_alloc_permit()
    prims.fence_mbarrier_init()
    # Cluster formation: makes the peers' shared memory and barriers addressable.
    prims.barrier_cluster_arrive_relaxed()
    prims.barrier_cluster_wait()
    prims.barrier_cta_sync(0)
    tmem_ptr = cutlass.inttoptr(tmem_ptr_i32.load(), 6, cutlass.Int32)

    if warp_id == 0:
        if prims.elect_sync():
            # k-tiles past the ring go to L2 now, so the refills after the dependency read L2.
            for i in cutlass.range_constexpr(RING_STAGES, my_tiles):
                k_global = rank + cutlass.Int32(i * SPLIT)
                prims.cp_async_bulk_tensor_prefetch(
                    tma_ptr_w,
                    [
                        cutlass.Int32(0),
                        m_offset,
                        k_global * cutlass.Int32(TMA_COPY_ITERS),
                        cutlass.Int32(0),
                        cutlass.Int32(0),
                    ],
                    [],  # tile mode: no im2col offsets
                )
            for i in cutlass.range_constexpr(my_tiles):
                stage = i % RING_STAGES
                if cutlass.const_expr(i >= RING_STAGES):
                    while not cute.arch.mbarrier_try_wait(
                        mma_done.subview(stage).data_ptr(), (i // RING_STAGES - 1) % 2
                    ):
                        pass
                k_global = rank + cutlass.Int32(i * SPLIT)
                prims.mbarrier_arrive_expect_tx(tma_done.subview(stage), CTA_M * CTA_K * ELEM_BYTES)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    smem_a.subview(stage * CTA_M * CTA_K),
                    tma_ptr_w,
                    (
                        cutlass.Int32(0),
                        m_offset,
                        k_global * cutlass.Int32(TMA_COPY_ITERS),
                        cutlass.Int32(0),
                        cutlass.Int32(0),
                    ),
                    tma_done.subview(stage),
                    l2_cache_hint=EVICT_FIRST,
                )
                if cutlass.const_expr(trigger_early and i == RING_STAGES - 1):
                    prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
            if cutlass.const_expr(trigger_early and my_tiles < RING_STAGES):
                prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
    elif warp_id == 1:
        prims.griddepcontrol(prims.GridDepAction.WAIT)
        if prims.elect_sync():
            for i in cutlass.range_constexpr(my_tiles):
                stage = i % RING_STAGES
                if cutlass.const_expr(i >= RING_STAGES):
                    while not cute.arch.mbarrier_try_wait(
                        mma_done.subview(stage).data_ptr(), (i // RING_STAGES - 1) % 2
                    ):
                        pass
                k_global = rank + cutlass.Int32(i * SPLIT)
                prims.mbarrier_arrive_expect_tx(tma_done.subview(stage), MMA_N * CTA_K * ELEM_BYTES)
                for half in cutlass.range_constexpr(TMA_COPY_ITERS):
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_b.subview(stage * MMA_N * CTA_K + half * B_HALF_ELEMS),
                        tma_ptr_x,
                        (
                            k_global * cutlass.Int32(CTA_K) + cutlass.Int32(half * TMA_K_BOX),
                            cutlass.Int32(0),
                        ),
                        tma_done.subview(stage),
                    )
    elif warp_id == 2:
        idesc = prims.Tcgen05InstrDesc.build(
            c_dtype=cutlass.Float32, a_dtype=io_dtype, b_dtype=io_dtype, n_dim=MMA_N, m_dim=CTA_M
        )
        desc_a_base = prims.Tcgen05SmemDesc.build(
            start_address=smem_a, leading_byte_offset=LEADING, stride_byte_offset=STRIDE,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )  # fmt: skip
        desc_b_base = prims.Tcgen05SmemDesc.build(
            start_address=smem_b, leading_byte_offset=LEADING, stride_byte_offset=STRIDE,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )  # fmt: skip
        for i in cutlass.range_constexpr(my_tiles):
            stage = i % RING_STAGES
            while not cute.arch.mbarrier_try_wait(
                tma_done.subview(stage).data_ptr(), (i // RING_STAGES) % 2
            ):
                pass
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            for kb in cutlass.range_constexpr(TMA_COPY_ITERS * K_BLOCKS_PER_HALF):
                box = kb // K_BLOCKS_PER_HALF
                within = kb % K_BLOCKS_PER_HALF
                desc_a = desc_a_base + (stage * STAGE_A + box * A_BOX + within * STEP)
                desc_b = desc_b_base + (stage * STAGE_B + box * B_BOX + within * STEP)
                if prims.elect_sync():
                    prims.tcgen05_mma(
                        prims.Tcgen05MMAKind.F16, prims.CTAGroup.CTA_1, tmem_ptr, desc_a, desc_b, idesc,
                        i > 0 or kb > 0,
                    )  # fmt: skip
            if prims.elect_sync():
                prims.tcgen05_commit(mma_done.subview(stage))
        if prims.elect_sync():
            prims.tcgen05_commit(acc_done)
    elif warp_id >= 4:
        lane = tx % 32
        owner = warp_id % 4  # TMEM lanes 32 * owner .. + 31 = the rows rank `owner` reduces
        while not cute.arch.mbarrier_try_wait(acc_done.data_ptr(), 0):
            pass
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        acc = prims.tcgen05_ld(
            "32x32b", cutlass.inttoptr(tmem_ptr_i32.load(), 6, cutlass.Float32), num=MMA_N
        )
        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        slot_row = lane * MMA_N
        if owner != rank:
            base = rank * cutlass.Int32(ROWS_PER_RANK * MMA_N) + slot_row
            for t in cutlass.range_constexpr(MMA_N):
                prims.mapa(mailbox.subview(base + cutlass.Int32(t)), owner).store(
                    cutlass.Float32(acc[t])
                )
            # Release at cluster scope: this lane's partials are visible to the owner's acquire.
            prims.mbarrier_arrive(prims.mapa(reduce_ready, owner), scope=prims.MemScope.CLUSTER)
        else:
            while not prims.mbarrier_try_wait_parity(
                reduce_ready, 0, scope=prims.MBarrierScope.CLUSTER
            ):
                pass
            n = m_offset + owner * cutlass.Int32(ROWS_PER_RANK) + lane
            for t in cutlass.range_constexpr(MMA_N):
                total = cutlass.Float32(0.0)
                for q in cutlass.range_constexpr(SPLIT):
                    part = mailbox.load(
                        idx=cutlass.Int32(q * ROWS_PER_RANK * MMA_N) + slot_row + cutlass.Int32(t)
                    )
                    total = total + cutlass.Float32(
                        cutlass.select_(rank == cutlass.Int32(q), cutlass.Float32(acc[t]), part)
                    )
                if cutlass.Int32(t) < num_tokens:
                    if n < cutlass.Int32(n_out):
                        y.store(cutlass.BFloat16(total), idx=cutlass.Int32(t * n_out) + n)
        prims.barrier_cta_sync(1, thread_count=128)
        if warp_id == 4:
            prims.tcgen05_dealloc(tmem_ptr, TMEM_COLS)


def _weight_tensor_map(w, n_out, k_in):
    """W as five TMA dimensions (64-element column chunk, row, 64-element chunk index, 1, 1) so one call
    per k-tile lands both 128-byte-swizzled halves; strides in 16-byte units."""
    return cuda.create_tensor_map_tiled(
        global_address=w.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=[TMA_K_BOX, n_out, k_in // TMA_K_BOX, 1, 1],
        global_strides=[
            (k_in * ELEM_BYTES) // 16,
            (TMA_K_BOX * ELEM_BYTES) // 16,
            (n_out * k_in * ELEM_BYTES) // 16,
            (n_out * k_in * ELEM_BYTES) // 16,
        ],
        box_dims=[TMA_K_BOX, CTA_M, TMA_COPY_ITERS, 1, 1],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )


def _activation_tensor_map(x, cols, num_tokens):
    """x [M, cols] as (cols, M) with an 8-row box: rows past num_tokens arrive as zeros."""
    return cuda.create_tensor_map_tiled(
        global_address=x.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=[cols, num_tokens],
        global_strides=[(cols * ELEM_BYTES) // 16],
        box_dims=[TMA_K_BOX, MMA_N],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )


@cute.jit
def k3_decode_gemv(
    w: cute.Tensor,  # [N, K] bf16, K contiguous
    x: cute.Tensor,  # [M, K] bf16, K contiguous, M <= 8
    y: cute.Tensor,  # [M * N] bf16
    num_tokens: cutlass.Int32,
    n_out: cutlass.Constexpr[int],
    k_in: cutlass.Constexpr[int],
    trigger_early: cutlass.Constexpr[bool],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    tma_desc_w = _weight_tensor_map(w, n_out, k_in)
    tma_desc_x = _activation_tensor_map(x, k_in, num_tokens)
    k3_decode_gemv_kernel(
        tma_desc_w, tma_desc_x, tma_desc_x, y, y, num_tokens, cutlass.Int32(0), cutlass.Float32(0.0),
        n_out, k_in, num_k_tiles(k_in), False, 0, trigger_early,
    ).launch(grid=(n_out // CTA_M, 1, 1), block=(THREADS, 1, 1), stream=stream, use_pdl=use_pdl)  # fmt: skip


@cute.jit
def k3_decode_gemv_tail(
    w: cute.Tensor,  # [N, K_lat + K_act] bf16: latent-up columns of this rank's slice (zero-padded) | shared down
    latent: cute.Tensor,  # [M, rms_cols] bf16, the whole reduced latent row
    latent_words: cute.Tensor,  # the same memory as int32 words, for the RMS
    act: cute.Tensor,  # [M, K_act] bf16, the shared-expert activation
    y: cute.Tensor,  # [M * N] bf16
    num_tokens: cutlass.Int32,
    lat_col0: cutlass.Int32,  # first latent column of this rank's slice
    eps: cutlass.Float32,
    n_out: cutlass.Constexpr[int],
    k_in: cutlass.Constexpr[int],
    lat_tiles: cutlass.Constexpr[int],
    rms_cols: cutlass.Constexpr[int],
    trigger_early: cutlass.Constexpr[bool],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    """``y = rmsnorm(latent)[:, slice] @ W_lat^T + act @ W_act^T`` (the RMS applied to the latent accumulator)."""
    tma_desc_w = _weight_tensor_map(w, n_out, k_in)
    tma_desc_lat = _activation_tensor_map(latent, rms_cols, num_tokens)
    tma_desc_act = _activation_tensor_map(act, k_in - lat_tiles * CTA_K, num_tokens)
    k3_decode_gemv_kernel(
        tma_desc_w, tma_desc_lat, tma_desc_act, y, latent_words, num_tokens, lat_col0, eps,
        n_out, k_in, lat_tiles, True, rms_cols, trigger_early,
    ).launch(grid=(n_out // CTA_M, 1, 1), block=(THREADS, 1, 1), stream=stream, use_pdl=use_pdl)  # fmt: skip


@cute.jit
def k3_decode_gemv_splitk(
    w: cute.Tensor,  # [N, K] bf16, K contiguous, K / 128 a multiple of SPLIT
    x: cute.Tensor,  # [M, K] bf16
    y: cute.Tensor,  # [M * N] bf16
    num_tokens: cutlass.Int32,
    n_out: cutlass.Constexpr[int],
    k_in: cutlass.Constexpr[int],
    trigger_early: cutlass.Constexpr[bool],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    tma_desc_w = _weight_tensor_map(w, n_out, k_in)
    tma_desc_x = _activation_tensor_map(x, k_in, num_tokens)
    k3_decode_gemv_splitk_kernel(
        tma_desc_w, tma_desc_x, y, num_tokens, n_out, k_in, trigger_early
    ).launch(
        grid=(((n_out + CTA_M - 1) // CTA_M) * SPLIT, 1, 1),
        block=(THREADS, 1, 1),
        cluster=(SPLIT, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )
