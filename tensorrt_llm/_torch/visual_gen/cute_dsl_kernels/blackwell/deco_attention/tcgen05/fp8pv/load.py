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
"""Load warp of the attention kernel: the TMA loads of a work tile's Q' tiles, of the K/V ring and of the v_mean rows.

A ring slot holds K'(i) with its scale factors and V(i-1): the QK of KV tile i and the PV of tile i - 1 are issued in
the same step of the MMA warp, so both operands arrive on one barrier (the first slot of a work tile has only K', and
a last slot only V). K', its scale factors and V are multicast to both CTAs of a cluster. The v_mean rows of a batch
of vs_tiles KV tiles go to a ring of two stages of the means MMA's B operand and complete on the barrier of the K/V
slot they are loaded with."""

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32

from ...common.dsl import nanosleep
from ...common.mem import sptr, store_vector
from ...constants import HEAD_DIM
from .mma import MEANS_TILER


@cute.jit
def poll_slot(producer, empty_ptr, sleep_ns: cutlass.Constexpr):
    """Wait for the producer's next free ring slot by try_wait probes with a fixed nanosleep between them, instead
    of the pipeline's blocking wait."""
    state = producer._PipelineProducer__state
    bar = empty_ptr + state.index
    done = cutlass.select_(cute.arch.mbarrier_try_wait(bar, state.phase), Int32(1), Int32(0))
    while done == 0:
        nanosleep(sleep_ns)
        done = cutlass.select_(cute.arch.mbarrier_try_wait(bar, state.phase), Int32(1), Int32(0))


@cute.jit
def spin_slot(producer, empty_ptr):
    """Wait for the producer's next free ring slot by a loop on test_wait."""
    state = producer._PipelineProducer__state
    bar = empty_ptr + state.index
    phase = state.phase
    done = Int32(cute.arch.mbarrier_test_wait(bar, phase))
    while done == 0:
        done = Int32(cute.arch.mbarrier_test_wait(bar, phase))


class LoadRole:
    """The load warp's methods of PingPongAttention (kernel.py), which holds the attributes they read."""

    @cute.jit
    def load_warp(
        self, qk_args, v_args, vm_args, pipeline_args, cluster_args, sched_args, n_kv_tiles, tidx
    ):
        """The load warp over its work tiles. qk_args, v_args: the TMA atoms, global tensors and shared-memory stages
        of Q' / K' and of V; vm_args: those of the v_mean ring, the shared address of the means B operand and the
        ring's empty barriers; pipeline_args: the Q and K/V producers and the K/V ring's empty barriers."""
        (
            tma_atom_q,
            mQ_qdl,
            q_sf_tma,
            tma_atom_k,
            mK_kdl,
            k_sf_tma,
            sQ,
            sK,
            qk_sf_smem,
            qk_thr_mma,
        ) = qk_args
        tma_atom_v, mV_dkl, sV, pv_thr_mma = v_args
        tma_atom_vm, tma_tensor_vm, sMB_tma, sMB_addr, vm_empty = vm_args
        load_q_producer, load_kv_producer, kv_empty_ptr = pipeline_args
        cta_coord_vmnk, kv_cta_layout, kv_mcast_mask = cluster_args
        tile_sched, work_tile = sched_args
        vm_par = Int32(3)  # wait parity of the two ring stages' empty barriers
        gVM = cute.local_tile(tma_tensor_vm, (HEAD_DIM, self.vs_rows), (None, None, None))
        tVMsVM, tVMgVM = cute.nvgpu.cpasync.tma_partition(
            tma_atom_vm,
            0,
            cute.make_layout(1),
            cute.group_modes(sMB_tma, 0, 2),
            cute.group_modes(gVM, 0, 2),
        )
        vm_load = (tma_atom_vm, tVMgVM, tVMsVM, vm_empty)
        # the TMA fills K rows 0..vs_rows - 1 of a B stage: the others stay zero (K slots the A operand leaves
        # zero)
        z = Int32(0)
        half = 64 * MEANS_TILER[2] * 2  # bytes of a 64-channel atom of a stage
        for st in cutlass.range_constexpr(2):
            for a in cutlass.range_constexpr(2):
                for c in cutlass.range_constexpr(half // 2 // 512):
                    store_vector(
                        sptr(
                            Int32,
                            sMB_addr
                            + st * 2 * half
                            + a * half
                            + half // 2
                            + c * 512
                            + (tidx % self.threads_per_warp) * 16,
                            16,
                        ),
                        (z, z, z, z),
                    )
        cute.arch.fence_view_async_shared()
        while work_tile.is_valid_tile:
            curr_block_coord = work_tile.tile_idx
            qk_load = self.qk.load_partitions(
                tma_atom_q,
                mQ_qdl,
                q_sf_tma,
                tma_atom_k,
                mK_kdl,
                k_sf_tma,
                sQ,
                sK,
                qk_sf_smem,
                qk_thr_mma,
                curr_block_coord[2],
                cta_coord_vmnk[1],
                kv_cta_layout,
            )
            gV_dkl = cute.flat_divide(mV_dkl, cute.select(self.pv_mma_tiler, mode=[1, 2]))
            tSgV_dkl = pv_thr_mma.partition_B(gV_dkl)
            tVsV, tVgV_dkl = cute.nvgpu.cpasync.tma_partition(
                tma_atom_v,
                cta_coord_vmnk[1],
                kv_cta_layout,
                cute.group_modes(sV, 0, 3),
                cute.group_modes(tSgV_dkl, 0, 3),
            )
            tVgV = tVgV_dkl[None, 0, None, curr_block_coord[2]]
            # Q0
            q0_coord = 2 * curr_block_coord[0]
            q0_handle = load_q_producer.acquire_and_advance()
            self.qk.copy_q(qk_load, q0_coord, q0_handle)
            # K0 (a slot with only K)
            kv_begin, n_local = self.kv_range(work_tile, n_kv_tiles)
            kv_coord = kv_begin
            self.poll_empty(load_kv_producer, kv_empty_ptr)
            k_handle = load_kv_producer.acquire_and_advance(True, expected_tx=self.tma_copy_k_bytes)
            self.qk.copy_k(qk_load, kv_coord, k_handle, kv_mcast_mask)
            # Q1
            q1_coord = q0_coord + 1
            q1_handle = load_q_producer.acquire_and_advance()
            self.qk.copy_q(qk_load, q1_coord, q1_handle)
            kv_coord += 1
            if cutlass.const_expr(not self.split):
                # prefetch the next work tile's Q tiles (each Q tile is read by one CTA only)
                next_idx = tile_sched._current_work_linear_idx + tile_sched.num_persistent_sm
                if next_idx < tile_sched._num_blocks:
                    nb = tile_sched._problem_shape_mhb.get_hier_coord(next_idx)
                    self.qk.prefetch_q(qk_load, nb)
            for i in cutlass.range(1, n_local, 1, unroll=1):
                self.poll_empty(load_kv_producer, kv_empty_ptr)
                # a slot that starts a means batch also brings v_mean of the batch before (its means MMAs
                # follow PV(i), so the wait for this slot covers it)
                vm_here = kv_coord & (self.vs_tiles - 1) == 0
                k_handle = load_kv_producer.acquire_and_advance(
                    True,
                    expected_tx=cutlass.select_(
                        vm_here,
                        Int32(
                            self.tma_copy_k_bytes + self.tma_copy_v_bytes + self.tma_copy_vm_bytes
                        ),
                        Int32(self.tma_copy_k_bytes + self.tma_copy_v_bytes),
                    ),
                )
                self.qk.copy_k(qk_load, kv_coord, k_handle, kv_mcast_mask)
                # V(i-1) shares the slot (and the barrier) of K(i)
                cute.copy(
                    tma_atom_v,
                    tVgV[None, kv_coord - 1],
                    tVsV[None, k_handle.index],
                    tma_bar_ptr=k_handle.barrier,
                    mcast_mask=kv_mcast_mask,
                )
                if vm_here:
                    vm_par = self.load_vm(
                        vm_load,
                        vm_par,
                        (kv_coord >> self.vs_shift) - 1,
                        curr_block_coord[2],
                        k_handle.barrier,
                    )
                kv_coord += 1
            self.poll_empty(load_kv_producer, kv_empty_ptr)
            v_handle = load_kv_producer.acquire_and_advance(
                True, expected_tx=self.tma_copy_v_bytes + self.tma_copy_vm_bytes
            )
            cute.copy(
                tma_atom_v,
                tVgV[None, kv_coord - 1],
                tVsV[None, v_handle.index],
                tma_bar_ptr=v_handle.barrier,
                mcast_mask=kv_mcast_mask,
            )
            # v_mean of the last batch, with the last V: its means MMAs follow the work tile's last PVs
            vm_par = self.load_vm(
                vm_load,
                vm_par,
                (kv_coord - 1) >> self.vs_shift,
                curr_block_coord[2],
                v_handle.barrier,
            )
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()
        if cutlass.const_expr(self.cluster_shape_mn[0] > 1):
            # observe both CTAs' final slot releases before either CTA can retire its SMEM
            load_kv_producer.tail()

    @cute.jit
    def poll_empty(self, producer, empty_ptr):
        """Wait for the producer's next free slot. With load_sleep_ns = 0 by test_wait (spin_slot); otherwise by
        try_wait probes with a fixed nanosleep between them (poll_slot)."""
        if cutlass.const_expr(self.load_sleep_ns == 0):
            spin_slot(producer, empty_ptr)
        else:
            poll_slot(producer, empty_ptr, self.load_sleep_ns)

    @cute.jit
    def load_vm(self, vm_load, par, batch, head, bar):
        """TMA of the v_mean rows of means batch `batch` (head coordinate `head`) into ring stage batch % 2, once the
        means MMAs of the batch before in that stage released it (par: the stages' wait parities), completing on the
        K/V slot barrier `bar` (whose expected bytes include it); returns par with the stage's bit flipped."""
        tma_atom_vm, tVMgVM, tVMsVM, vm_empty = vm_load
        stage = batch & 1
        cute.arch.mbarrier_wait(vm_empty + stage, (par >> stage) & 1)
        cute.copy(tma_atom_vm, tVMgVM[None, 0, batch, head], tVMsVM[None, stage], tma_bar_ptr=bar)
        return par ^ (1 << stage)
