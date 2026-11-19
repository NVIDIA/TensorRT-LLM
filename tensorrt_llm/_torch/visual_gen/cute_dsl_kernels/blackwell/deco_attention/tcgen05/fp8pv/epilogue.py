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
"""Epilogue of the attention kernel, run by the epilogue warpgroup (a thread per row): O_j times v_scale over the row
sum of P (column 128 of O_j), as BF16 into a shared-memory staging tile that the store warp stores by TMA.

Main launch (epilogue_tile): one O tile at a time. Split launch (epilogue_split): the units of the last partial
wave are split along KV into pieces that run on different SMs, each with its own anchor. A piece that does not merge
an O tile publishes its unnormalized rows, anchor and row sum to global memory and sets a ready flag; the merging
piece waits for the flags, brings every piece to the largest anchor and normalizes the sum."""

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.typing import Float32, Int32, Int64

from ...common.mem import gptr, store_vector
from ...common.ptx import ld_global_acquire_b32, ld_global_cg_v4
from ...constants import HEAD_DIM


def _merge_slot(slot_base, j, merger, stage):
    """Merge slot of O tile `stage` of the j-th piece other than its merging piece `merger`: piece j + (j >= merger)
    of the unit whose slots start at slot_base."""
    return (slot_base + j + cutlass.select_(j >= merger, 1, 0)) * 2 + stage


class EpilogueRole:
    """The epilogue warpgroup's methods of PingPongAttention (kernel.py), which holds the attributes they read."""

    def _epi_copy(self, thr_mma, tOtO, dest_O):
        """TMEM-load / SMEM-store partitions of one O tile for the epilogue warps."""
        cO = cute.make_identity_tensor((self.pv_mma_tiler[0], self.pv_mma_tiler[1]))
        chunk_cols = 32 * 8 // self.o_dtype.width
        tOdO = thr_mma.partition_C(dest_O)
        tOcO = thr_mma.partition_C(cO)
        tOtO_i = cute.logical_divide(tOtO, cute.make_layout((128, chunk_cols)))
        tOcO_i = cute.logical_divide(tOcO, cute.make_layout((128, chunk_cols)))
        tOdO_i = cute.logical_divide(tOdO, cute.make_layout((128, chunk_cols)))
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.epilogue_warp_ids))
        epi_subtile = (self.epi_tile[0], chunk_cols)
        tmem_copy_atom = sm100_utils.get_tmem_load_op(
            self.pv_mma_tiler,
            self.o_layout,
            self.o_dtype,
            self.pv_acc_dtype,
            epi_subtile,
            use_2cta_instrs=False,
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO_i[(None, None), 0])
        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
        tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_i[(None, None), None])
        tTMEM_LOADdO = thr_tmem_load.partition_D(tOdO_i[(None, None), None])
        tTMEM_LOADoO = thr_tmem_load.partition_D(tOcO_i[(None, None), None])
        smem_copy_atom = sm100_utils.get_smem_store_op(
            self.o_layout, self.o_dtype, self.pv_acc_dtype, tiled_tmem_load
        )
        tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)
        return (
            tiled_tmem_load,
            tTMEM_LOADtO,
            tTMEM_LOADdO,
            tTMEM_LOADoO,
            tiled_smem_store,
            thread_idx,
        )

    @cute.jit
    def epilogue_split(
        self,
        thr_mma,
        tensor_args,
        pipeline_args,
        scale_output,
        split_info,
        split_args,
        score_scale,
    ):
        """Split launch epilogue for both O tiles of a KV piece. O tile s is merged by piece s % nparts; every other
        piece publishes its unnormalized O rows, anchor and row sum into its (unit, piece, tile) slot and then sets
        the slot's ready flag. All publishes come first, so the two merges (on different SMs) run in parallel. The
        flags are zero on entry (the host zeroes them before the launches of a call) and on exit: the merger returns
        each flag it waited for to zero."""
        tOtO0, tOtO1, vec0, vec1, sO = tensor_args
        s0_epi_consumer, s1_epi_consumer, mma_epi_consumer, epi_store_producer = pipeline_args
        part, nparts, slot_base = split_info
        split_o_addr, split_ml_addr, split_flag_addr = split_args
        split_ready = Int32(1)  # the flag of a published slot
        eps = [
            self._epi_copy(thr_mma, tOtO0, sO[None, None, 0]),
            self._epi_copy(thr_mma, tOtO1, sO[None, None, 1]),
        ]
        thread_idx = eps[0][5]
        n_chunks = self.cta_tiler[2] // (32 * 8 // self.o_dtype.width)
        # final anchors of both tiles (published by the softmax warpgroups at the end of the piece)
        cute.arch.barrier(barrier_id=self.stats_bar_id, number_of_threads=self.stats_bar_threads)
        h0 = s0_epi_consumer.wait_and_advance()
        rm0 = vec0[0, h0.index]
        h0.release()
        cute.arch.barrier(
            barrier_id=self.stats_bar_id + 1, number_of_threads=self.stats_bar_threads
        )
        h1 = s1_epi_consumer.wait_and_advance()
        rm1 = vec1[0, h1.index]
        h1.release()
        o0_handle = mma_epi_consumer.wait_and_advance()
        o1_handle = mma_epi_consumer.wait_and_advance()
        rms = [rm0, rm1]
        rss = [self.load_row_sum(thr_mma, tOtO0), self.load_row_sum(thr_mma, tOtO1)]
        e0_handle = epi_store_producer.acquire_and_advance()
        e1_handle = epi_store_producer.acquire_and_advance()
        mergers = [0, 1 % nparts]
        # publish phase
        for stage in cutlass.range_constexpr(2):
            tiled_tmem_load, tTMEM_LOADtO, tTMEM_LOADdO, tTMEM_LOADoO, tiled_smem_store, _ = eps[
                stage
            ]
            if part != mergers[stage]:
                slot = (slot_base + part) * 2 + stage
                row_base = Int64(slot * 128 + thread_idx)
                cute.arch.store(gptr(Float32, split_ml_addr + row_base * 8), rms[stage])
                cute.arch.store(gptr(Float32, split_ml_addr + row_base * 8 + 4), rss[stage])
                for i in cutlass.range_constexpr(n_chunks):
                    tTMEM_LOADtO_i = tTMEM_LOADtO[None, 0, 0, i]
                    tTMEM_LOADoO_i = tTMEM_LOADoO[None, 0, 0, i]
                    tTMrO = cute.make_rmem_tensor(tTMEM_LOADoO_i.shape, self.pv_acc_dtype)
                    cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO)
                    for j in cutlass.range_constexpr(0, cute.size(tTMrO), 4):
                        _, col = tTMEM_LOADoO_i[j]
                        # checked while tracing: the 16-byte store of four columns starts at a multiple of four
                        assert col % 4 == 0
                        dst = split_o_addr + (row_base * 128 + col) * 4
                        vals = (tTMrO[j], tTMrO[j + 1], tTMrO[j + 2], tTMrO[j + 3])
                        store_vector(gptr(Float32, dst, 16), vals)
                cute.arch.fence_acq_rel_gpu()
                self.split_barrier.arrive_and_wait()
                if thread_idx == 0:
                    cute.arch.store(
                        gptr(Int32, split_flag_addr + Int64(slot) * 4),
                        split_ready,
                        sem="release",
                        scope="gpu",
                    )
        # merge phase: the other pieces are j = 0 .. nparts - 2 -> piece j + (j >= merger)
        n_other = nparts - 1
        for stage in cutlass.range_constexpr(2):
            tiled_tmem_load, tTMEM_LOADtO, tTMEM_LOADdO, tTMEM_LOADoO, tiled_smem_store, _ = eps[
                stage
            ]
            merger = mergers[stage]
            if part == merger:
                row_max = rms[stage]
                m_fin = row_max
                for j in cutlass.range(0, n_other, 1, unroll=1):
                    slot_p = _merge_slot(slot_base, j, merger, stage)
                    flag = ld_global_acquire_b32(split_flag_addr + Int64(slot_p) * 4, Int32)
                    while flag != split_ready:
                        flag = ld_global_acquire_b32(split_flag_addr + Int64(slot_p) * 4, Int32)
                    m_p = cute.arch.load(
                        gptr(Float32, split_ml_addr + Int64(slot_p * 128 + thread_idx) * 8),
                        Float32,
                        cop="cg",
                    )
                    m_fin = cute.arch.fmax(m_fin, m_p)
                # every thread has seen the flags it polls: zero them for the next call (nothing in this launch reads
                # them again; the next call's publishes follow this launch in stream order)
                self.split_barrier.arrive_and_wait()
                for j in cutlass.range(thread_idx, n_other, 128, unroll=1):
                    cute.arch.store(
                        gptr(
                            Int32,
                            split_flag_addr + Int64(_merge_slot(slot_base, j, merger, stage)) * 4,
                        ),
                        Int32(0),
                    )
                s_own = cute.math.exp2(score_scale * (row_max - m_fin), fastmath=True)
                l_fin = rss[stage] * s_own
                for j in cutlass.range(0, n_other, 1, unroll=1):
                    rb = Int64(_merge_slot(slot_base, j, merger, stage) * 128 + thread_idx)
                    m_p = cute.arch.load(gptr(Float32, split_ml_addr + rb * 8), Float32, cop="cg")
                    l_p = cute.arch.load(
                        gptr(Float32, split_ml_addr + rb * 8 + 4), Float32, cop="cg"
                    )
                    l_fin = l_fin + l_p * cute.math.exp2(score_scale * (m_p - m_fin), fastmath=True)
                scale = scale_output / l_fin
                scale_o = scale * s_own
                for i in cutlass.range_constexpr(n_chunks):
                    tTMEM_LOADtO_i = tTMEM_LOADtO[None, 0, 0, i]
                    tTMEM_LOADdO_i = tTMEM_LOADdO[None, 0, 0, i]
                    tTMEM_LOADoO_i = tTMEM_LOADoO[None, 0, 0, i]
                    tTMrO = cute.make_rmem_tensor(tTMEM_LOADoO_i.shape, self.pv_acc_dtype)
                    cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO)
                    for jj in cutlass.range_constexpr(0, cute.size(tTMrO), 2):
                        tTMrO[jj], tTMrO[jj + 1] = cute.arch.mul_packed_f32x2(
                            (tTMrO[jj], tTMrO[jj + 1]), (scale_o, scale_o)
                        )
                    # two pieces per iteration: both loads are issued before either is used; an odd last pair
                    # re-reads its first piece with weight 0
                    for j in cutlass.range(0, n_other, 2, unroll=1):
                        has_b = j + 1 < n_other
                        slot_a = _merge_slot(slot_base, j, merger, stage)
                        slot_b = cutlass.select_(
                            has_b, _merge_slot(slot_base, j + 1, merger, stage), slot_a
                        )
                        rb_a = Int64(slot_a * 128 + thread_idx)
                        rb_b = Int64(slot_b * 128 + thread_idx)
                        m_a = cute.arch.load(
                            gptr(Float32, split_ml_addr + rb_a * 8), Float32, cop="cg"
                        )
                        m_b = cute.arch.load(
                            gptr(Float32, split_ml_addr + rb_b * 8), Float32, cop="cg"
                        )
                        vals_a = []
                        vals_b = []
                        for jj in cutlass.range_constexpr(0, cute.size(tTMrO), 4):
                            _, col = tTMEM_LOADoO_i[jj]
                            vals_a += ld_global_cg_v4(
                                split_o_addr + (rb_a * 128 + col) * 4, Float32
                            )
                            vals_b += ld_global_cg_v4(
                                split_o_addr + (rb_b * 128 + col) * 4, Float32
                            )
                        s_a = scale * cute.math.exp2(score_scale * (m_a - m_fin), fastmath=True)
                        s_b = cutlass.select_(
                            has_b,
                            scale * cute.math.exp2(score_scale * (m_b - m_fin), fastmath=True),
                            Float32(0.0),
                        )
                        for jj in cutlass.range_constexpr(cute.size(tTMrO)):
                            tTMrO[jj] = tTMrO[jj] + s_a * vals_a[jj] + s_b * vals_b[jj]
                    tDMrO = cute.make_rmem_tensor(tTMrO.shape, self.o_dtype)
                    o_vec = tTMrO.load()
                    tDMrO.store(o_vec.to(self.o_dtype))
                    cute.copy(tiled_smem_store, tDMrO, tTMEM_LOADdO_i)
        cute.arch.fence_view_async_shared()
        o0_handle.release()
        o1_handle.release()
        e0_handle.commit()
        e1_handle.commit()
        cute.arch.barrier_arrive(barrier_id=self.epi_bar_id, number_of_threads=self.epi_bar_threads)
        cute.arch.barrier_arrive(
            barrier_id=self.epi_bar_id + 1, number_of_threads=self.epi_bar_threads
        )
        return s0_epi_consumer, s1_epi_consumer, mma_epi_consumer, epi_store_producer

    @cute.jit
    def load_row_sum(self, pv_thr_mma, tOtO):
        """Row sum of P: column HEAD_DIM of O_j, where the softmax warpgroup left it at the end of the work tile."""
        cO = cute.make_identity_tensor((self.pv_mma_tiler[0], self.pv_mma_tiler[1]))
        tOcO = pv_thr_mma.partition_C(cO)
        atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(1)), self.pv_acc_dtype
        )
        tOtO_i = cute.make_tensor(
            tOtO.iterator, cute.composition(tOtO.layout, cute.make_layout((128, 1)))
        )
        tOcO_i = cute.make_tensor(
            tOcO.iterator, cute.composition(tOcO.layout, cute.make_layout((128, 1)))
        )
        tiled = tcgen05.make_tmem_copy(atom, tOtO_i)
        tidx, _, _ = cute.arch.thread_idx()
        thr = tiled.get_slice(tidx % (self.threads_per_warp * 4))
        tT, tC = thr.partition_S(tOtO_i), thr.partition_D(tOcO_i)
        tTMrS = cute.make_rmem_tensor(tC.shape, self.pv_acc_dtype)
        cute.copy(tiled, cute.make_tensor(tT.iterator + HEAD_DIM, tT.layout), tTMrS)
        return tTMrS[0]

    @cute.jit
    def epilogue_tile(self, thr_mma, tensor_args, pipeline_args, scale_output, stage: int):
        """Main launch epilogue of O tile `stage`: O times v_scale over the row sum (column 128), BF16 into its SMEM
        staging tile for the store warp's TMA store."""
        tOtO, dest_O = tensor_args
        si_epi_consumer, mma_epi_consumer, epi_store_producer = pipeline_args
        chunk_cols = 32 * 8 // self.o_dtype.width
        tiled_tmem_load, tTMEM_LOADtO, tTMEM_LOADdO, tTMEM_LOADoO, tiled_smem_store, _ = (
            self._epi_copy(thr_mma, tOtO, dest_O)
        )

        cute.arch.barrier(
            barrier_id=self.stats_bar_id + stage, number_of_threads=self.stats_bar_threads
        )
        vec_i_handle = si_epi_consumer.wait_and_advance()
        vec_i_handle.release()

        oi_handle = mma_epi_consumer.wait_and_advance()
        row_sum = self.load_row_sum(thr_mma, tOtO)
        oi_final_handle = epi_store_producer.acquire_and_advance()
        n_chunks = self.cta_tiler[2] // chunk_cols
        scale = scale_output / row_sum
        for i in cutlass.range_constexpr(n_chunks):
            tTMEM_LOADtO_i = tTMEM_LOADtO[None, 0, 0, i]
            tTMEM_LOADdO_i = tTMEM_LOADdO[None, 0, 0, i]
            tTMEM_LOADoO_i = tTMEM_LOADoO[None, 0, 0, i]
            tTMrO = cute.make_rmem_tensor(tTMEM_LOADoO_i.shape, self.pv_acc_dtype)
            cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO)
            for j in cutlass.range_constexpr(0, cute.size(tTMrO), 2):
                tTMrO[j], tTMrO[j + 1] = cute.arch.mul_packed_f32x2(
                    (tTMrO[j], tTMrO[j + 1]), (scale, scale)
                )
            tDMrO = cute.make_rmem_tensor(tTMrO.shape, self.o_dtype)
            o_vec = tTMrO.load()
            tDMrO.store(o_vec.to(self.o_dtype))
            cute.copy(tiled_smem_store, tDMrO, tTMEM_LOADdO_i)
        cute.arch.fence_view_async_shared()
        oi_handle.release()
        oi_final_handle.commit()
        cute.arch.barrier_arrive(
            barrier_id=self.epi_bar_id + stage, number_of_threads=self.epi_bar_threads
        )
        return si_epi_consumer, mma_epi_consumer, epi_store_producer

    @cute.jit
    def store_warp(self, mO_qdl, tma_atom_o, sO, epi_store_consumer, sched_args):
        """The store warp over its work tiles: the TMA stores of both staged O tiles of a work tile (in the split
        launch, of the tiles this KV piece merges)."""
        tile_sched, work_tile = sched_args
        while work_tile.is_valid_tile:
            curr_block_coord = work_tile.tile_idx
            o0_coord = 2 * curr_block_coord[0]
            o1_coord = o0_coord + 1
            gO_qdl = cute.flat_divide(mO_qdl, cute.select(self.pv_mma_tiler, mode=[0, 1]))
            gO = gO_qdl[None, None, None, 0, curr_block_coord[2]]
            tOsO, tOgO = cute.nvgpu.cpasync.tma_partition(
                tma_atom_o,
                0,
                cute.make_layout(1),
                cute.group_modes(sO, 0, 2),
                cute.group_modes(gO, 0, 2),
            )
            cute.arch.barrier(barrier_id=self.epi_bar_id, number_of_threads=self.epi_bar_threads)
            o0_handle = epi_store_consumer.wait_and_advance()
            if cutlass.const_expr(self.split):
                if curr_block_coord[5] == 0:  # O0 is merged by piece 0
                    cute.copy(tma_atom_o, tOsO[None, 0], tOgO[None, o0_coord])
            else:
                cute.copy(tma_atom_o, tOsO[None, 0], tOgO[None, o0_coord])
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.barrier(
                barrier_id=self.epi_bar_id + 1, number_of_threads=self.epi_bar_threads
            )
            o1_handle = epi_store_consumer.wait_and_advance()
            if cutlass.const_expr(self.split):
                if (
                    curr_block_coord[5] == 1 % curr_block_coord[6]
                ):  # O1 is merged by piece 1 % nparts
                    cute.copy(tma_atom_o, tOsO[None, 1], tOgO[None, o1_coord])
            else:
                cute.copy(tma_atom_o, tOsO[None, 1], tOgO[None, o1_coord])
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(1, read=True)
            o0_handle.release()
            cute.arch.cp_async_bulk_wait_group(0, read=True)
            o1_handle.release()
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()
        cute.arch.cp_async_bulk_wait_group(0, read=False)
