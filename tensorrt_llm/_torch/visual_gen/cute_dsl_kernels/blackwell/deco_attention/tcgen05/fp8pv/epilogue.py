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
"""Epilogue of the attention kernel, run by the epilogue warpgroup (a thread per row).

O_j times v_scale over the row sum of P (column 128 of O_j), as BF16 into a shared-memory staging tile that the store
warp stores by TMA. Main launch (epilogue_tile): one O tile at a time. Split launch (epilogue_split): the units of
the last partial wave are split along KV into pieces that run on different SMs, each with its own anchor, and the
pieces of an O tile are merged through global memory.
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.typing import Float32, Int32, Int64, Uint32

from ...common.mem import gptr, load_vector, sptr, store_vector
from ...common.state import State
from ...constants import HEAD_DIM
from ..valid_len import valid_len
from .sched import PART


class EpilogueRole:
    """The epilogue warpgroup's methods of PingPongAttention (kernel.py), which holds the attributes they read."""

    def _epi_copy(self, thr_mma, tOtO, dest_O):
        """TMEM-load / SMEM-store partitions of one O tile for the epilogue warps, by name (a State)."""
        cO = cute.make_identity_tensor((self.pv_mma_tiler[0], self.pv_mma_tiler[1]))
        chunk_cols = 32 * 8 // self.o_dtype.width
        tOdO = thr_mma.partition_C(dest_O)
        tOcO = thr_mma.partition_C(cO)
        tOtO_i = cute.logical_divide(tOtO, cute.make_layout((128, chunk_cols)))
        tOcO_i = cute.logical_divide(tOcO, cute.make_layout((128, chunk_cols)))
        tOdO_i = cute.logical_divide(tOdO, cute.make_layout((128, chunk_cols)))
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_wg)
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
        return State(
            tiled_tmem_load=tiled_tmem_load,
            tTMEM_LOADtO=tTMEM_LOADtO,
            tTMEM_LOADdO=tTMEM_LOADdO,
            tTMEM_LOADoO=tTMEM_LOADoO,
            tiled_smem_store=tiled_smem_store,
            thread_idx=thread_idx,
            n_chunks=self.cta_tiler[2] // chunk_cols,
        )

    def _load_chunk(self, ep, i, staged=True):
        """Return column chunk i of an O tile in registers, its staging chunk (staged) and its (row, column) views."""
        src = ep.tTMEM_LOADtO[None, 0, 0, i]
        dst = ep.tTMEM_LOADdO[None, 0, 0, i] if staged else None
        coords = ep.tTMEM_LOADoO[None, 0, 0, i]
        tTMrO = cute.make_rmem_tensor(coords.shape, self.pv_acc_dtype)
        cute.copy(ep.tiled_tmem_load, src, tTMrO)
        return tTMrO, dst, coords

    def _stage_chunk(self, ep, tTMrO, dst, stage, live_rows):
        """Stage a chunk of O tile `stage` as BF16 in dst; with lengths, zeros in the rows from live_rows on."""
        if live_rows is not None:
            live = stage * 128 + ep.thread_idx < live_rows
            for j in range(cute.size(tTMrO)):
                tTMrO[j] = cutlass.select_(live, tTMrO[j], Float32(0.0))
        tDMrO = cute.make_rmem_tensor(tTMrO.shape, self.o_dtype)
        o_vec = tTMrO.load()
        tDMrO.store(o_vec.to(self.o_dtype))
        cute.copy(ep.tiled_smem_store, tDMrO, dst)

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
        live_rows=None,
    ):
        """Split launch epilogue for both O tiles of a KV piece: all publishes, then the merges.

        * Every piece publishes its unnormalized O rows, anchor and row sum of both tiles into its (unit, piece,
          tile) slots, then counts itself on the (unit, tile) counter. The piece that arrives last at a counter
          merges that tile: no CTA waits for another, so the launch needs no co-residency. All publishes come
          first, so the two merges (on different SMs when different pieces arrive last) run in parallel.
        * The merger reads the slots of all pieces in piece order, its own among them, brings them to the largest
          anchor and normalizes the sum: the bits do not depend on which piece merges. The counters are zero on entry
          (the host zeroes them before the launches of a call) and the merger zeroes its counter again. The smem
          words at last_addr (one per tile) tell the store warp which tiles this piece merged.
        """
        tOtO0, tOtO1, vec0, vec1, sO = tensor_args
        s0_epi_consumer, s1_epi_consumer, mma_epi_consumer, epi_store_producer = pipeline_args
        part, nparts, slot_base = split_info
        split_o_addr, split_ml_addr, split_count_addr, last_addr = split_args
        eps = [
            self._epi_copy(thr_mma, tOtO0, sO[None, None, 0]),
            self._epi_copy(thr_mma, tOtO1, sO[None, None, 1]),
        ]
        thread_idx = eps[0].thread_idx
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
        # publish phase; the counter of (unit, tile) is the word of the unit's first slot of that tile
        for stage in cutlass.range_constexpr(2):
            ep = eps[stage]
            slot = (slot_base + part) * 2 + stage
            row_base = Int64(slot * 128 + thread_idx)
            cute.arch.store(gptr(Float32, split_ml_addr + row_base * 8), rms[stage])
            cute.arch.store(gptr(Float32, split_ml_addr + row_base * 8 + 4), rss[stage])
            for i in cutlass.range_constexpr(ep.n_chunks):
                tTMrO, _, coords = self._load_chunk(ep, i, staged=False)
                for j in cutlass.range_constexpr(0, cute.size(tTMrO), 4):
                    _, col = coords[j]
                    # checked while tracing: the 16-byte store of four columns starts at a multiple of four
                    assert col % 4 == 0
                    slot_addr = split_o_addr + (row_base * 128 + col) * 4
                    vals = (tTMrO[j], tTMrO[j + 1], tTMrO[j + 2], tTMrO[j + 3])
                    store_vector(gptr(Float32, slot_addr, 16), vals)
            cute.arch.fence_acq_rel_gpu()
            self.split_barrier.arrive_and_wait()
            if thread_idx == 0:
                arrived = cute.arch.atomic_add(
                    gptr(Uint32, split_count_addr + Int64(slot_base * 2 + stage) * 4),
                    Uint32(1),
                    sem="acq_rel",
                    scope="gpu",
                )
                cute.arch.fence_acq_rel_gpu()
                cute.arch.store(
                    sptr(Int32, last_addr + stage * 4), Int32(Int32(arrived) == nparts - 1)
                )
        self.split_barrier.arrive_and_wait()
        # merge phase, by the last piece of each tile, over the slots of all pieces in piece order (its own among
        # them), so that the result does not depend on which piece arrives last
        for stage in cutlass.range_constexpr(2):
            ep = eps[stage]
            if cute.arch.load(sptr(Int32, last_addr + stage * 4), Int32) != 0:
                if thread_idx == 0:
                    # the counter is zero for the next call (nothing in this launch reads it again)
                    cute.arch.store(
                        gptr(Int32, split_count_addr + Int64(slot_base * 2 + stage) * 4), Int32(0)
                    )
                m_fin = rms[stage]
                for j in cutlass.range(0, nparts, 1, unroll=1):
                    rb = Int64(((slot_base + j) * 2 + stage) * 128 + thread_idx)
                    m_p = cute.arch.load(gptr(Float32, split_ml_addr + rb * 8), Float32, cop="cg")
                    m_fin = cute.arch.fmax(m_fin, m_p)
                l_fin = Float32(0.0)
                for j in cutlass.range(0, nparts, 1, unroll=1):
                    rb = Int64(((slot_base + j) * 2 + stage) * 128 + thread_idx)
                    m_p = cute.arch.load(gptr(Float32, split_ml_addr + rb * 8), Float32, cop="cg")
                    l_p = cute.arch.load(
                        gptr(Float32, split_ml_addr + rb * 8 + 4), Float32, cop="cg"
                    )
                    l_fin = l_fin + l_p * cute.math.exp2(score_scale * (m_p - m_fin), fastmath=True)
                scale = scale_output / l_fin
                for i in cutlass.range_constexpr(ep.n_chunks):
                    tTMrO, dst, coords = self._load_chunk(ep, i)
                    for jj in cutlass.range_constexpr(cute.size(tTMrO)):
                        tTMrO[jj] = Float32(0.0)
                    # two pieces per iteration: both loads are issued before either is used; an odd last pair
                    # re-reads its first piece with weight 0
                    for j in cutlass.range(0, nparts, 2, unroll=1):
                        has_b = j + 1 < nparts
                        slot_a = (slot_base + j) * 2 + stage
                        slot_b = cutlass.select_(has_b, slot_a + 2, slot_a)
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
                            _, col = coords[jj]
                            vals_a += load_vector(
                                gptr(Float32, split_o_addr + (rb_a * 128 + col) * 4, 16),
                                4,
                                cop="cg",
                            )
                            vals_b += load_vector(
                                gptr(Float32, split_o_addr + (rb_b * 128 + col) * 4, 16),
                                4,
                                cop="cg",
                            )
                        s_a = scale * cute.math.exp2(score_scale * (m_a - m_fin), fastmath=True)
                        s_b = cutlass.select_(
                            has_b,
                            scale * cute.math.exp2(score_scale * (m_b - m_fin), fastmath=True),
                            Float32(0.0),
                        )
                        for jj in cutlass.range_constexpr(cute.size(tTMrO)):
                            tTMrO[jj] = tTMrO[jj] + s_a * vals_a[jj] + s_b * vals_b[jj]
                    self._stage_chunk(ep, tTMrO, dst, stage, live_rows)
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
        thr = tiled.get_slice(tidx % self.threads_per_wg)
        tT, tC = thr.partition_S(tOtO_i), thr.partition_D(tOcO_i)
        tTMrS = cute.make_rmem_tensor(tC.shape, self.pv_acc_dtype)
        cute.copy(tiled, cute.make_tensor(tT.iterator + HEAD_DIM, tT.layout), tTMrS)
        return tTMrS[0]

    @cute.jit
    def epilogue_tile(
        self, thr_mma, tensor_args, pipeline_args, scale_output, stage: int, live_rows=None
    ):
        """Main launch epilogue of O tile `stage`: O times v_scale over the row sum, as BF16 into its staging tile."""
        tOtO, dest_O = tensor_args
        si_epi_consumer, mma_epi_consumer, epi_store_producer = pipeline_args
        ep = self._epi_copy(thr_mma, tOtO, dest_O)

        cute.arch.barrier(
            barrier_id=self.stats_bar_id + stage, number_of_threads=self.stats_bar_threads
        )
        vec_i_handle = si_epi_consumer.wait_and_advance()
        vec_i_handle.release()

        oi_handle = mma_epi_consumer.wait_and_advance()
        row_sum = self.load_row_sum(thr_mma, tOtO)
        oi_final_handle = epi_store_producer.acquire_and_advance()
        scale = scale_output / row_sum
        for i in cutlass.range_constexpr(ep.n_chunks):
            tTMrO, dst, _ = self._load_chunk(ep, i)
            for j in cutlass.range_constexpr(0, cute.size(tTMrO), 2):
                tTMrO[j], tTMrO[j + 1] = cute.arch.mul_packed_f32x2(
                    (tTMrO[j], tTMrO[j + 1]), (scale, scale)
                )
            self._stage_chunk(ep, tTMrO, dst, stage, live_rows)
        cute.arch.fence_view_async_shared()
        oi_handle.release()
        oi_final_handle.commit()
        cute.arch.barrier_arrive(
            barrier_id=self.epi_bar_id + stage, number_of_threads=self.epi_bar_threads
        )
        return si_epi_consumer, mma_epi_consumer, epi_store_producer

    @cute.jit
    def store_warp(self, mO_qdl, tma_atom_o, sO, epi_store_consumer, sched_args, last_addr=None):
        """Store warp: TMA stores of the staged O tiles of a work tile (split launch: the tiles its piece merged, the
        smem words at last_addr that epilogue_split wrote before the epilogue barrier of the tile)."""
        tile_sched, work_tile = sched_args
        while work_tile.is_valid_tile:
            curr_block_coord = work_tile.tile_idx
            o0_coord = 2 * curr_block_coord[0]
            o_coords = (o0_coord, o0_coord + 1)
            gO_qdl = cute.flat_divide(mO_qdl, cute.select(self.pv_mma_tiler, mode=[0, 1]))
            gO = gO_qdl[None, None, None, 0, curr_block_coord[2]]
            tOsO, tOgO = cute.nvgpu.cpasync.tma_partition(
                tma_atom_o,
                0,
                cute.make_layout(1),
                cute.group_modes(sO, 0, 2),
                cute.group_modes(gO, 0, 2),
            )
            handles = []
            for j in cutlass.range_constexpr(2):
                cute.arch.barrier(
                    barrier_id=self.epi_bar_id + j, number_of_threads=self.epi_bar_threads
                )
                handles.append(epi_store_consumer.wait_and_advance())
                if cutlass.const_expr(self.split):
                    if cute.arch.load(sptr(Int32, last_addr + j * 4), Int32) != 0:
                        cute.copy(tma_atom_o, tOsO[None, j], tOgO[None, o_coords[j]])
                else:
                    cute.copy(tma_atom_o, tOsO[None, j], tOgO[None, o_coords[j]])
                cute.arch.cp_async_bulk_commit_group()
            for j in cutlass.range_constexpr(2):
                cute.arch.cp_async_bulk_wait_group(1 - j, read=True)
                handles[j].release()
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()
        cute.arch.cp_async_bulk_wait_group(0, read=False)

    @cute.jit
    def zero_rows(self, tile_sched_params, lens_addr, o_addr, seqlen, tidx):
        """The idle warp of a launch with valid prefix lengths: zeros in the O rows of the units that no role runs.

        * Those units have no query row below the length of their batch element (with CTA pairs: their pair has none).
        * The warp walks the units of the CTA's share of the static schedule (in the split launch the work of the
          CTA, where the first piece of a unit stores its rows); each thread stores its share of the 16-byte chunks.
        """
        lane = tidx % self.threads_per_warp
        heads = tile_sched_params.problem_shape_mhb[1]
        group = self.cluster_shape_mn[0]
        sched = self._tile_scheduler(tile_sched_params)
        work = sched.get_current_work()
        z = Int32(0)
        while work.is_valid_tile:
            m = work.tile_idx[0]
            h, b = work.tile_idx[2]
            no_row = (m - m % group) * self.cta_tiler[0] >= valid_len(lens_addr, b, seqlen)
            if cutlass.const_expr(self.split):
                no_row = no_row & (work.tile_idx[PART] == 0)
            if no_row:
                row0 = m * self.cta_tiler[0]
                rows = cutlass.min(Int32(self.cta_tiler[0]), seqlen - row0)
                row_bytes = Int64(heads) * Int64(2 * HEAD_DIM)
                base = (
                    Int64(o_addr)
                    + (Int64(b) * Int64(seqlen) + Int64(row0)) * row_bytes
                    + Int64(h) * Int64(2 * HEAD_DIM)
                )
                for c in cutlass.range(lane, rows * 16, self.threads_per_warp, unroll=1):
                    dst = base + Int64(c >> 4) * row_bytes + Int64((c & 15) * 16)
                    store_vector(gptr(Int32, dst, 16), (z, z, z, z))
            sched.advance_to_next_work()
            work = sched.get_current_work()
