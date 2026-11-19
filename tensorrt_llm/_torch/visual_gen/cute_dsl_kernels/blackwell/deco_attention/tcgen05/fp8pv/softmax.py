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
"""Softmax warpgroups of the attention kernel: S_j(i) -> P_j(i) for the KV tiles i of a work tile, one 128-row Q tile
per warpgroup (a thread per row).

Skip correction: a row keeps its anchor m (the running row maximum) unless the tile maximum exceeds it by more than
T = RESCALE_THRESHOLD binades; only then is O_j rescaled. P is pre-scaled by s_P = 2^P_PRESCALE_LOG2 = 2^floor(log2
448 - T) = 16, so P = s_P 2^(c (s - m)) stays below the largest E4M3 value while the anchor lags by up to T binades.

I-ExpCast: c, the multiplier of the raw scores in log2 units, is a power of two of at least 2^MIN_SCORE_LOG2 (the
preparation's score scale). c SUBNORMAL_ULPS 2^-149 is then an f32 subnormal, and each score pair lands with one
fma.rn.f32x2 on the subnormal grid at SUBNORMAL_ULPS ulps per log2 unit, where one saturating u16 pack reads the f16
bits of 2^t, without an exponential instruction. Two pairs become four E4M3 codes with two converts.

The row sum of P is not formed here: the PV MMA accumulates the sums of P over each KV tile in the columns of O_j
after V (mma.py). V smoothing reads them back: after every batch of vs_tiles KV tiles a thread loads its row's eight
tile sums, writes them as BF16 into its row of the warpgroup's means A operand, zeroes the columns for the next batch
and adds them to the row sum it keeps in a register. At the end of the work tile it leaves that row sum in column 128
of O_j for the epilogue."""

import math
from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.typing import Float32, Int32, Uint32

from ...common.dsl import Tcgen05Fence, tcgen05_fence
from ...common.expmath import (
    bf16x2,
    ex2_sub_f16x2,
    f16x2x2_to_e4m3x4,
    subnormal_anchor_bias,
    subnormal_scale,
    tree_sum,
)
from ...common.mem import sptr, store_vector
from ...constants import RESCALE_THRESHOLD, SUBNORMAL_ULPS
from ..tmem import ld_tmem_f32, st_tmem_f32
from .sched import SplitTileScheduler

E4M3_MAX = 448.0
# P pre-scale: s_P = 2^floor(log2(largest finite E4M3) - T)
P_PRESCALE_LOG2 = int(math.log2(E4M3_MAX) - RESCALE_THRESHOLD)
# The score-scale floor of this family: log2 of the smallest multiplier of the raw scores the landing takes exactly.
# Its fma.rn.f32x2 multiplier is the multiplier x SUBNORMAL_ULPS x 2^-149, an f32 subnormal for a power of two from
# 1 / SUBNORMAL_ULPS up; the preparation raises the score scale to at least this.
MIN_SCORE_LOG2 = -(SUBNORMAL_ULPS.bit_length() - 1)


class SoftmaxRole:
    """The softmax warpgroups' methods of PingPongAttention (kernel.py), which holds the attributes they read."""

    def softmax_sched(self, tile_sched_params, tile_sched, work_tile):
        """In split mode the softmax warps get the 5-value work coordinate."""
        if cutlass.const_expr(self.split):
            sched = SplitTileScheduler(
                tile_sched_params,
                cute.arch.block_idx()[0] + tile_sched_params.n_full,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
                1,
            )
            return sched, sched.initial_work_tile_info()
        return tile_sched, work_tile

    @cute.jit
    def exp_cast(self, tTMEM_LOADrS, tTMEM_STORErS_x4, fma_ab):
        """I-ExpCast of a tile's scores: each score pair lands on the f16 bits of P with one fma.rn.f32x2 and one
        saturating pack ((t + 15) * 2^-139 = S * a + b from the raw scores, straight onto the subnormal grid), and two
        landed pairs become four E4M3 codes of the P store registers."""
        p8 = cute.recast_tensor(tTMEM_STORErS_x4, Int32)
        for q in cutlass.range_constexpr(cute.size(tTMEM_LOADrS) // 4):
            e01 = ex2_sub_f16x2(tTMEM_LOADrS[4 * q], tTMEM_LOADrS[4 * q + 1], fma_ab[0], fma_ab[1])
            e23 = ex2_sub_f16x2(
                tTMEM_LOADrS[4 * q + 2], tTMEM_LOADrS[4 * q + 3], fma_ab[0], fma_ab[1]
            )
            p8[q] = f16x2x2_to_e4m3x4(e01, e23)

    @cute.jit
    def softmax_step(
        self,
        whether_apply_mask: bool,
        iter_args: Tuple,
        stats_args: Tuple,
        pipeline_args: Tuple,
        value_args: Tuple,
        atom_args: Tuple,
        tensor_args: Tuple,
        vs: Tuple,
    ):
        """One KV tile of a softmax warpgroup: S_j(i) -> P_j(i). stats_args: (anchor, row sum of P so far); vs:
        (vs_row, i, first KV tile of the work tile): when tile i - 1 ended a means batch, its block sums go from O_j
        into the means A operand once PV_j(i - 1) completed."""
        row_max, row_sum = stats_args
        vs_row, step, kv_begin = vs
        cS, is_last_iter = iter_args
        seqlen_k, score_scale, o_done_ptr, p_tail_stage_ptr, thr_inc, scale_sub = value_args
        si_peek_status, mma_si_consumer, si_epi_producer, pi_mma_producer, o_phase = pipeline_args
        (
            qk_thr_mma,
            tiled_tmem_load,
            tiled_tmem_store,
            thr_tmem_load,
            thr_tmem_store,
        ) = atom_args
        tTMEM_LOADtS, tTMEM_STOREtS_x4, tOtOi, pv_thr_mma = tensor_args
        tilePlikeFP32 = self.qk_mma_tiler[1] // Float32.width * self.v_dtype.width
        tScS = qk_thr_mma.partition_C(cS)
        tScS_P_layout = cute.composition(tScS.layout, cute.make_layout((128, tilePlikeFP32)))
        tScS_P = cute.make_tensor(tScS.iterator, tScS_P_layout)
        tTMEM_LOADcS = thr_tmem_load.partition_D(tScS)
        tTMEM_STOREcS = thr_tmem_store.partition_S(tScS_P)

        si_handle = mma_si_consumer.wait_and_advance(si_peek_status)
        tTMEM_LOADrS = cute.make_rmem_tensor(tTMEM_LOADcS.shape, self.qk_acc_dtype)
        old_row_max = row_max
        tile_row_max = -Float32.inf
        if cutlass.const_expr(self.use_ld_red):
            n_ld = cute.size(tTMEM_LOADrS, mode=[1])
            tTMEM_LOADrMax = cute.make_rmem_tensor(cute.make_layout((1, n_ld)), self.qk_acc_dtype)
            # S is a shared physical tile: issue every chunk before the final TMEM-load fence and only then return
            # its logical empty credit
            for i in cutlass.range_constexpr(n_ld):
                cute.copy_atom_call(
                    tiled_tmem_load,
                    tTMEM_LOADtS[None, i, 0, 0],
                    (tTMEM_LOADrS[None, i, 0, 0], tTMEM_LOADrMax[None, i]),
                )
            cute.arch.fence_view_async_tmem_load()
            si_handle.release()
            if whether_apply_mask:
                for i in cutlass.range_constexpr(cute.size(tTMEM_LOADrS)):
                    _, index_k = tTMEM_LOADcS[i]
                    tTMEM_LOADrS[i] = Float32(
                        cutlass.select_(index_k < seqlen_k, tTMEM_LOADrS[i], -math.inf)
                    )
                tile_row_max = tTMEM_LOADrS.load().reduce(cute.ReductionOp.MAX, -Float32.inf, 0)
            else:
                tile_row_max = tTMEM_LOADrMax.load().reduce(cute.ReductionOp.MAX, -Float32.inf, 0)
        else:
            cute.copy(tiled_tmem_load, tTMEM_LOADtS, tTMEM_LOADrS)
            cute.arch.fence_view_async_tmem_load()
            si_handle.release()
            if whether_apply_mask:
                for i in cutlass.range_constexpr(cute.size(tTMEM_LOADrS)):
                    _, index_k = tTMEM_LOADcS[i]
                    tTMEM_LOADrS[i] = Float32(
                        cutlass.select_(index_k < seqlen_k, tTMEM_LOADrS[i], -math.inf)
                    )
            tile_row_max = tTMEM_LOADrS.load().reduce(cute.ReductionOp.MAX, -Float32.inf, 0)
        # the logical empty credit is returned right after the complete load fence: the masking, the reductions and
        # the exponentiation below work on registers

        # skip correction: keep the anchor unless the tile max exceeds it by more than T binades (every tile has a
        # finite max)
        tTMEM_STORErS_x4 = cute.make_rmem_tensor(tTMEM_STOREcS.shape, self.qk_acc_dtype)
        moved = tile_row_max > old_row_max + thr_inc
        row_max_safe = cutlass.select_(moved, tile_row_max, old_row_max)
        # the addend takes the subnormal multiplier of the exp, and 15 is the exponent bias of the f16 pattern: the
        # anchor score maps exactly onto s_P = 2^P_PRESCALE_LOG2 (exact in the P format)
        self.exp_cast(
            tTMEM_LOADrS,
            tTMEM_STORErS_x4,
            (scale_sub, subnormal_anchor_bias(row_max_safe, scale_sub, P_PRESCALE_LOG2)),
        )
        # Rare: the anchor moved for a row of this warp -> rescale O_j once PV_j(i-1) has landed,
        # before P_j(i) is published (PV_j(i) accumulates on top of the rescaled O_j). The factor is
        # only formed on that path (rows that kept their anchor, or start a work tile, use 1).
        need_rescale = moved & (old_row_max != -Float32.inf)
        if cute.arch.vote_ballot_sync(need_rescale) != 0:
            acc_scale = Float32(1.0)
            if need_rescale:
                acc_scale = cute.math.exp2(
                    score_scale * (old_row_max - row_max_safe), fastmath=True
                )
            cute.arch.mbarrier_wait(o_done_ptr, o_phase ^ 1)
            self.rescale_o(pv_thr_mma, tOtOi, acc_scale)
            # the row sum to the new anchor (no block sums wait in the A operand across a rescale: a batch's sums
            # are read after this step's rescale, and its means MMA completes before the next step's)
            row_sum = row_sum * acc_scale
        o_phase = o_phase ^ 1
        # PV_j(i-1) has finished reading P_j: its own P-empty commit
        pi_handle = pi_mma_producer.acquire_and_advance()
        tile = kv_begin + step - 1
        take = (step >= 1) & (tile & (self.vs_tiles - 1) == self.vs_tiles - 1)
        if cutlass.const_expr(self.vs_batched):
            # once per batch (tile i - 1 ended it): its block sums from O_j (PV_j(i - 1) completed: the credit above)
            # into the means A operand and the row sum, and the columns zeroed before PV_j(i) adds to them
            if take:
                tcgen05_fence(Tcgen05Fence.AFTER_THREAD_SYNC)
                sums = self.vs_load(vs_row)
                cute.arch.fence_view_async_tmem_load()
                st_tmem_f32(vs_row[2], [Float32(0.0)] * 8)
                self.vs_put(vs_row, sums)
                cute.arch.fence_view_async_shared()
                row_sum = row_sum + tree_sum(sums)
        else:
            # the block sums the PVs of a batch left in O_j (PV_j(i - 1) completed: the credit above), loaded here,
            # ahead of the P stores. The columns PV_j(i) adds to start at zero: stored here, ahead of the P stores
            # (not between them and the P-full commit), while no PV_j runs (a PV writes all its columns), and never
            # in a work tile's first step (column 128 may still hold the previous row sum for the epilogue)
            tcgen05_fence(Tcgen05Fence.AFTER_THREAD_SYNC)
            sums = self.vs_load(vs_row)
            if take:
                cute.arch.fence_view_async_tmem_load()
                st_tmem_f32(vs_row[2], [Float32(0.0)] * 8)
        # P_j(i): three of its four 32-key slices before the P-full commit, the last one after it (the last K phase
        # of PV_j(i) waits for the P-tail arrive)
        p_slices = cute.size(tTMEM_STOREtS_x4.shape[2])
        p_main_slices = p_slices // 4 * 3
        for i in cutlass.range_constexpr(p_main_slices):
            cute.copy(
                tiled_tmem_store,
                tTMEM_STORErS_x4[None, None, i],
                tTMEM_STOREtS_x4[None, None, i],
            )
        cute.arch.fence_view_async_tmem_store()
        pi_handle.commit()
        for i in cutlass.range_constexpr(p_main_slices, p_slices):
            cute.copy(
                tiled_tmem_store,
                tTMEM_STORErS_x4[None, None, i],
                tTMEM_STOREtS_x4[None, None, i],
            )
        if cutlass.const_expr(not self.vs_batched):
            # the sums of the batch reach the A operand before the P-tail arrive: the means MMA follows PV_j(i)
            if take:
                self.vs_put(vs_row, sums)
                cute.arch.fence_view_async_shared()
        cute.arch.fence_view_async_tmem_store()
        cute.arch.mbarrier_arrive(p_tail_stage_ptr)
        if cutlass.const_expr(not self.vs_batched):
            if take:
                cute.arch.fence_view_async_tmem_load()
                row_sum = row_sum + tree_sum(sums)
        if not is_last_iter:
            si_peek_status = mma_si_consumer.try_wait()

        return (row_max_safe, row_sum), (
            si_peek_status,
            mma_si_consumer,
            si_epi_producer,
            pi_mma_producer,
            o_phase,
        )

    @cute.jit
    def rescale_chunk(
        self,
        tiled_tmem_load,
        tiled_tmem_store,
        tTMEM_LOADtO,
        tTMEM_LOADcO,
        tTMEM_STOREtO,
        col,
        acc_scale,
    ):
        """One column chunk of O_j (columns col..) times acc_scale."""
        tTMrO = cute.make_rmem_tensor(tTMEM_LOADcO.shape, self.pv_acc_dtype)
        cute.copy(
            tiled_tmem_load,
            cute.make_tensor(tTMEM_LOADtO.iterator + col, tTMEM_LOADtO.layout),
            tTMrO,
        )
        for j in cutlass.range_constexpr(0, cute.size(tTMrO), 2):
            tTMrO[j], tTMrO[j + 1] = cute.arch.mul_packed_f32x2(
                (tTMrO[j], tTMrO[j + 1]), (acc_scale, acc_scale)
            )
        cute.copy(
            tiled_tmem_store,
            tTMrO,
            cute.make_tensor(tTMEM_STOREtO.iterator + col, tTMEM_STOREtO.layout),
        )

    @cute.jit
    def rescale_o(self, pv_thr_mma, tOtO, acc_scale):
        """O_j rows *= acc_scale (rows whose anchor did not move pass acc_scale = 1), over all of O_j's columns: V
        and the block sums of P after it."""
        cO = cute.make_identity_tensor((self.pv_mma_tiler[0], self.pv_mma_tiler[1]))
        tOcO = pv_thr_mma.partition_C(cO)
        chunk_cols = 16
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(chunk_cols)), self.pv_acc_dtype
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(chunk_cols)), self.pv_acc_dtype
        )
        tOtO_i = cute.make_tensor(
            tOtO.iterator, cute.composition(tOtO.layout, cute.make_layout((128, chunk_cols)))
        )
        tOcO_i = cute.make_tensor(
            tOcO.iterator, cute.composition(tOcO.layout, cute.make_layout((128, chunk_cols)))
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_i)
        tiled_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_i)
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * 4)
        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
        thr_tmem_store = tiled_tmem_store.get_slice(thread_idx)
        tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_i)
        tTMEM_LOADcO = thr_tmem_load.partition_D(tOcO_i)
        tTMEM_STOREtO = thr_tmem_store.partition_D(tOtO_i)
        n_chunks = self.pv_n // chunk_cols
        if cutlass.const_expr(self.rescale_loop):
            for i in cutlass.range(n_chunks, unroll=1):
                self.rescale_chunk(
                    tiled_tmem_load,
                    tiled_tmem_store,
                    tTMEM_LOADtO,
                    tTMEM_LOADcO,
                    tTMEM_STOREtO,
                    i * chunk_cols,
                    acc_scale,
                )
        else:
            for i in cutlass.range_constexpr(n_chunks):
                self.rescale_chunk(
                    tiled_tmem_load,
                    tiled_tmem_store,
                    tTMEM_LOADtO,
                    tTMEM_LOADcO,
                    tTMEM_STOREtO,
                    i * chunk_cols,
                    acc_scale,
                )

    def vs_load(self, vs_row):
        """TMEM load of the eight block sums of this thread's row (columns 128..135 of O_j), waited for by vs_put."""
        return ld_tmem_f32(vs_row[2], 8)

    @cute.jit
    def vs_put(self, vs_row, sums):
        """The block sums `sums` (vs_load) as the BF16 K slots 0..7 of this thread's row of the means A operand."""
        cute.arch.fence_view_async_tmem_load()
        words = tuple(bf16x2(sums[2 * w], sums[2 * w + 1]) for w in range(4))
        store_vector(sptr(Uint32, vs_row[0], 16), words)

    @cute.jit
    def softmax(
        self,
        stage: int,
        tensor_args: Tuple,
        pipeline_args: Tuple,
        qk_thr_mma: cute.ThrMma,
        value_args: Tuple,
        sched_args: Tuple,
        vs_row: Tuple,
    ):
        """Softmax warpgroup `stage` over its work tiles. vs_row: the shared addresses of K slots 0..7 and 8..15 of
        this thread's row of its warpgroup's means A operand and the TMEM address of its lane's O_j columns from 128
        on."""
        tStS, sVecWG, tOtOi, pv_thr_mma, o_done_ptr, p_tail_ptr = tensor_args
        p_tail_stage_ptr = p_tail_ptr + stage
        mma_si_consumer, si_epi_producer, pi_mma_producer = pipeline_args
        seqlen_k, n_kv_tiles, score_scale = value_args
        tile_sched, work_tile = sched_args

        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * 4)
        sVecRow = None
        if cutlass.const_expr(self.split):
            # this thread's row of the warpgroup's anchor exchange with the epilogue warpgroup
            sVecRow = sVecWG[thread_idx, None, None]

        cS_base = cute.make_identity_tensor((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
        tilePlikeFP32 = self.qk_mma_tiler[1] // 32 * self.v_dtype.width
        tStS_P_layout = cute.composition(tStS.layout, cute.make_layout((128, tilePlikeFP32)))
        tmem_p_offset = self.tmem_p0_offset + stage * (self.tmem_p1_offset - self.tmem_p0_offset)
        tStS_P = cute.make_tensor(tStS.iterator + tmem_p_offset, tStS_P_layout)
        if cutlass.const_expr(self.use_ld_red):
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.LdRed32x32bOp(
                    tcgen05.copy.Repetition(64), redOp=tcgen05.TmemLoadRedOp.MAX
                ),
                self.qk_acc_dtype,
            )
        else:
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.qk_acc_dtype
            )
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tStS)
        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
        # TMEM load/store addresses are collective per warp: partition the memory side with the warp's first
        # thread so the address is warp-uniform; register coordinates keep the true lane.
        copy_thread_idx = (
            cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        ) * self.threads_per_warp
        tTMEM_LOADtS = tiled_tmem_load.get_slice(copy_thread_idx).partition_S(tStS)
        # 8-column P stores (four per tile, three before the P-full commit)
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(8)), self.qk_acc_dtype
        )
        tiled_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStS_P)
        thr_tmem_store = tiled_tmem_store.get_slice(thread_idx)
        tTMEM_STOREtS_x4 = tiled_tmem_store.get_slice(copy_thread_idx).partition_D(tStS_P)
        o_phase = Int32(0)
        # skip-correction threshold in raw score units (T binades)
        thr_inc = Float32(RESCALE_THRESHOLD) / score_scale
        scale_sub = subnormal_scale(score_scale)
        # K slots 8..15 of the means A operand, which no batch writes, stay zero
        z = Int32(0)
        for half in cutlass.range_constexpr(2):
            store_vector(sptr(Int32, vs_row[half], 16), (z, z, z, z))
        while work_tile.is_valid_tile:
            curr_block_coord = work_tile.tile_idx
            logical_offset = (
                curr_block_coord[0] * self.cta_tiler[0] + stage * self.qk_mma_tiler[0],
                0,
            )
            cS = cute.domain_offset(logical_offset, cS_base)
            value_args_ = (seqlen_k, score_scale, o_done_ptr, p_tail_stage_ptr, thr_inc, scale_sub)
            atom_args = (
                qk_thr_mma,
                tiled_tmem_load,
                tiled_tmem_store,
                thr_tmem_load,
                thr_tmem_store,
            )
            tensor_args_ = (tTMEM_LOADtS, tTMEM_STOREtS_x4, tOtOi, pv_thr_mma)
            stats_args = (-Float32.inf, Float32(0.0))
            si_peek_status = mma_si_consumer.try_wait()
            pipeline_args_ = (
                si_peek_status,
                mma_si_consumer,
                si_epi_producer,
                pi_mma_producer,
                o_phase,
            )
            kv_begin, n_local = self.kv_range(work_tile, n_kv_tiles)
            n_unmasked = n_local
            if cutlass.const_expr(self.split):
                # only the piece holding the last KV tile masks
                if seqlen_k % self.qk_mma_tiler[1] != 0:
                    if kv_begin + n_local == n_kv_tiles:
                        n_unmasked = n_local - 1
            else:
                if seqlen_k % self.qk_mma_tiler[1] != 0:
                    n_unmasked = n_kv_tiles - 1
            for i in cutlass.range(0, n_unmasked, 1, unroll=1):
                cS_iter = cute.domain_offset((0, (kv_begin + i) * self.qk_mma_tiler[1]), cS)
                stats_args, pipeline_args_ = self.softmax_step(
                    False,
                    (cS_iter, i == n_local - 1),
                    stats_args,
                    pipeline_args_,
                    value_args_,
                    atom_args,
                    tensor_args_,
                    (vs_row, i, kv_begin),
                )
            for i in cutlass.range(n_unmasked, n_local, 1, unroll=1):
                cS_iter = cute.domain_offset((0, (kv_begin + i) * self.qk_mma_tiler[1]), cS)
                stats_args, pipeline_args_ = self.softmax_step(
                    True,
                    (cS_iter, i == n_local - 1),
                    stats_args,
                    pipeline_args_,
                    value_args_,
                    atom_args,
                    tensor_args_,
                    (vs_row, i, kv_begin),
                )
            _, mma_si_consumer, si_epi_producer, pi_mma_producer, o_phase = pipeline_args_
            # the open batch's block sums once the last PV completed, for the means MMA of the last batch, and the
            # row sum of P into column 128 of O_j for the epilogue
            cute.arch.mbarrier_wait(o_done_ptr, o_phase ^ 1)
            tcgen05_fence(Tcgen05Fence.AFTER_THREAD_SYNC)
            row_max, row_sum = stats_args
            sums = self.vs_load(vs_row)
            self.vs_put(vs_row, sums)
            cute.arch.fence_view_async_shared()
            st_tmem_f32(vs_row[2], [row_sum + tree_sum(sums)])
            cute.arch.fence_view_async_tmem_store()
            tcgen05_fence(Tcgen05Fence.BEFORE_THREAD_SYNC)
            cute.arch.barrier_arrive(
                barrier_id=self.vs_bar_id + stage, number_of_threads=self.vs_bar_threads
            )
            # final stats for the epilogue
            vec_i_handle = si_epi_producer.acquire_and_advance()
            if cutlass.const_expr(self.split):
                # final anchor of this KV piece: O and its row-sum column are relative to it
                sVecRow[0, vec_i_handle.index] = row_max
            vec_i_handle.commit()
            cute.arch.barrier_arrive(
                barrier_id=self.stats_bar_id + stage, number_of_threads=self.stats_bar_threads
            )
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()
