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
"""MMA warp of the attention kernel: its loop over the KV tiles of a work tile (mma_warp), which issues the QKs of
both Q tiles (qk.py), their PVs and the means MMAs of V smoothing.

Per KV tile i the warp issues QK0(i), PV0(i - 1), QK1(i), PV1(i - 1): both QKs write the one physical S tile, so each
waits for the other softmax warpgroup's S-empty credit, and the PV of the previous tile overlaps that hand-off. K(i)
and V(i - 1) come from one ring slot.

PV_j: [O_j | sums] += P_j [V | ones]. P_j is E4M3 in TMEM (K-major), V a ring stage of MN-major E4M3 in shared
memory, issued as four K = 32 kind::f8f6f4 MMAs per 128-key tile (two K = 64 where the target has them) with N = 144:
the N columns 128..143 of the B operand are a constant block of ones, so the columns of O_j after V take sums of P
over the tile's keys. The eight chunks of a row of that block hold eight variants, each 1.0 in one of the columns
128..135, and the PV of KV tile t starts its ones 16 (t % 8) bytes into the block: the row sum of P over tile t lands
in column 128 + t % 8.

Means MMA: V is stored minus its BF16 means of every 128 keys, so after each batch of eight KV tiles O_j +=
(the batch's eight tile sums of P, BF16) x (the eight v_mean rows of the batch). Both operands are in shared memory:
A is one K = 16 row of sums per query row and softmax warpgroup (the softmax warps write it), B a ring of two stages
of v_mean rows (the load warp fills it)."""

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.typing import Int32, Int64

from ...common.mem import sptr
from ..ptx import opaque_i32, smem_desc_hi, tc_commit_means, tc_mma_f8f6f4, tc_mma_means

# means MMA: 128 rows x 128 channels over 16 K slots of BF16 block sums. Its descriptors without the address fields:
# A, the K-major block sums (32-byte swizzle, SBO 256); B, the MN-major v_mean rows (128-byte swizzle, SBO 1024, the
# second 64-channel atom 2 KB on); kind::f16 BF16 x BF16 -> F32, K-major A, MN-major B, N = M = 128
MEANS_TILER = (128, 128, 16)
MEANS_A_DESC = (smem_desc_hi(256, 6) << 32) | (1 << 16)
MEANS_B_DESC = (smem_desc_hi(1024, 2) << 32) | ((2048 >> 4) << 16)
MEANS_IDESC = (
    (1 << 4)
    | (1 << 7)
    | (1 << 10)
    | (1 << 16)
    | ((MEANS_TILER[1] >> 3) << 17)
    | ((MEANS_TILER[0] >> 4) << 24)
)
# bytes of one softmax warpgroup's A operand
MEANS_A_BYTES = MEANS_TILER[0] * MEANS_TILER[2] * 2


# ------------------------------------------------------------------------------------------------
# The MMA warp
# ------------------------------------------------------------------------------------------------
class MmaRole:
    """The MMA warp's methods of PingPongAttention (kernel.py), which holds the attributes they read."""

    def pv_idesc(self):
        """kind::f8f6f4 instruction descriptor of the PV: F32 accumulator (bits 4-5 = 1), E4M3 A = P (bits 7-9 = 0),
        E4M3 B (bits 10-12 = 0), K-major A (P in TMEM), MN-major B (bit 16), N = 144 (>> 3 at bit 17), M = 128 (>> 4
        at bit 24), K = 64 (bit 29) where pv_k is 64."""
        return Int32(
            (1 << 4)
            | (1 << 16)
            | ((self.pv_n >> 3) << 17)
            | ((self.pv_mma_tiler[0] >> 4) << 24)
            | ((1 << 29) if self.pv_k == 64 else 0)
        )

    @staticmethod
    def v_desc(v_addr, ones_addr):
        """Descriptor of K phase 0 of a V stage: MN-major 128B-swizzled E4M3 (SBO 1 KB, one K phase = v_phase_bytes
        bytes); the second 128-column N atom (columns 128..143 = ones) sits LBO bytes after the first, so LBO
        points at the ones block."""
        a0 = Int32(v_addr)
        lbo = Int32(ones_addr) - a0
        return (Int64(smem_desc_hi(1024, 2)) << 32) | Int64(
            ((a0 >> 4) & 0x3FFF) | (((lbo >> 4) & 0x3FFF) << 16)
        )

    def vs_ones_offset(self, tile):
        """The bytes into the ones block where the PV of KV tile `tile` (global index) starts: variant tile % 8, whose
        1.0 is in column 128 + tile % 8."""
        return (tile & (self.vs_tiles - 1)) << 4

    @cute.jit
    def mma_pv(
        self, whether_acc, tensor_args, pi_mma_consumer, p_tail_stage_ptr, means, p_token=None
    ):
        """PV_j: [O_j | sums] += P_j [V | ones] over the K phases of the tile, then the release of P_j. means: (issue,
        shared memory base, j, v_mean ring stage): when issue is set, the means MMA of softmax warpgroup j after the
        PV (before the release, which also covers it), and for j = 1 the release of the ring stage."""
        o_addr, p_addr, v_desc0, idesc_pv = tensor_args
        # the P-tail barrier completes once per PV_j, like P_j's full barrier: its wait parity is the P-full parity,
        # read before the P-full wait
        p_phase = pi_mma_consumer._PipelineConsumer__state.phase
        pi_handle = pi_mma_consumer.wait_and_advance(p_token)
        num_kphases = self.pv_mma_tiler[2] // self.pv_k
        for kphase_idx in cutlass.range_constexpr(num_kphases):
            if cutlass.const_expr(kphase_idx == num_kphases - 1):
                # the softmax warps store the last quarter of P_j after its full commit: the last K phase waits
                # for it
                cute.arch.mbarrier_wait(p_tail_stage_ptr, p_phase)
            acc = whether_acc
            if cutlass.const_expr(kphase_idx != 0):
                acc = True
            # K phase k of the V stage starts v_phase_bytes later and as much closer to the ones block (LBO)
            step = self.v_phase_bytes
            desc = v_desc0 + Int64(kphase_idx * ((step >> 4) - ((step >> 4) << 16)))
            # P_k starts pv_k / 4 TMEM columns later (four E4M3 keys per column)
            p_col = (self.pv_k // 4) * kphase_idx
            with cute.arch.elect_one():
                tc_mma_f8f6f4(o_addr, p_addr + p_col, desc, idesc_pv, acc, "discard")
        issue, smem_base, j, stage = means
        if issue:
            a_off, b_off, e_off = self.vs_offs
            tc_mma_means(
                o_addr,
                smem_base,
                stage,
                a_off + j * MEANS_A_BYTES,
                b_off,
                MEANS_A_DESC,
                MEANS_B_DESC,
                MEANS_IDESC,
            )
            if cutlass.const_expr(j == 1):
                tc_commit_means(smem_base, stage, e_off)
        # this release (a tcgen05.commit on the P slot's empty barrier) also tells a softmax warp that moves its
        # anchor that O_j is updated: the rare rescale waits on that barrier, so no separate O commit is needed.
        pi_handle.release()
        return pi_mma_consumer

    @cute.jit
    def mma_warp(
        self,
        qk_mma,
        tStS,
        tmem_args,
        smem_args,
        ones_u32,
        pipeline_args,
        sched_args,
        n_kv_tiles,
        tidx,
    ):
        """The MMA warp over its work tiles: per KV tile i, QK0(i), PV0(i - 1), QK1(i), PV1(i - 1), with the means
        MMAs of a batch after the PVs of its last tile. tmem_args: the TMEM base, the addresses of O_0, O_1, P_0 and
        P_1 and the PV instruction descriptor; smem_args: the shared addresses of the V half of ring slot 0, of Q
        stage 0 (with Q' in TMEM), of the K/V slots' empty barriers, of the ones block and of the first field (the
        base of the means MMA's offsets), and the P-tail barriers; ones_u32: the ones block as words."""
        tmem_base, o0_addr, o1_addr, p0_addr, p1_addr, idesc_pv = tmem_args
        v_addr0, sq_addr, kv_empty_addr, ones_addr, smem_base, p_tail_ptr = smem_args
        (
            mma_s0_producer,
            mma_s1_producer,
            load_q_consumer,
            load_kv_consumer,
            mma_epi_producer,
            p0_mma_consumer,
            p1_mma_consumer,
        ) = pipeline_args
        tile_sched, work_tile = sched_args
        lane = tidx % self.threads_per_warp
        # The ones block: the N columns 128..143 of the PV B operand. A tile picks its column by where its
        # descriptor starts in the block: 16 v bytes in, key row r reads physical chunk v ^ (r % 8) (128B-swizzled
        # MN-major), so the eight chunks of a row hold eight variants, variant v being 1.0 in column 128 + v and
        # zero elsewhere. Word `lane` of key row j is word lane % 4 of chunk lane // 4.
        for j in cutlass.range_constexpr(self.v_phase_bytes // 4 // self.threads_per_warp):
            col = (lane // 4) ^ (j % 8)
            ones_u32[lane + j * self.threads_per_warp] = cutlass.select_(
                col // 4 == lane % 4, Int32(0x38) << (8 * (col % 4)), Int32(0)
            )
        cute.arch.fence_view_async_shared()
        cute.arch.sync_warp()

        # Keep one reserved output credit for each logical S pipeline. After
        # committing QKj, reacquiring that same pipeline waits until softmax-j
        # has drained the shared physical S tile and reserves its next credit.
        s0_out_handle = mma_s0_producer.acquire_and_advance()
        s1_out_handle = mma_s1_producer.acquire_and_advance()
        while work_tile.is_valid_tile:
            q0_handle = load_q_consumer.wait_and_advance()
            if cutlass.const_expr(self.qk.q_tmem):
                # the previous work tile's QKs are complete (its last S1 credit was taken after its PV tail)
                self.qk.q_to_tmem(tmem_base, 0, sq_addr, q0_handle.index)
            tSrQ0 = self.qk.q_operand(qk_mma, q0_handle.index)
            k_handle = load_kv_consumer.wait_and_advance()
            tSrK0 = self.qk.k_operand(qk_mma, k_handle.index)
            q0_stage = q0_handle.index
            k_stage = k_handle.index
            # QK00, its scale factors right before the MMAs
            self.qk.stage_scales(qk_mma, q0_stage, k_stage)
            self.qk.issue_qk(qk_mma, 0, tStS, self.qk.descs(tSrQ0, tSrK0), s0_out_handle)
            q1_handle = load_q_consumer.wait_and_advance()
            if cutlass.const_expr(self.qk.q_tmem):
                self.qk.q_to_tmem(tmem_base, 1, sq_addr, q1_handle.index)
            tSrQ1 = self.qk.q_operand(qk_mma, q1_handle.index)
            q1_stage = q1_handle.index
            # QK1 shares the physical S tile. This next-empty credit proves
            # every WG0 T2R load completed before QK1 overwrites it.
            s0_out_handle = mma_s0_producer.acquire_and_advance()
            # QK10 (QK00 staged the K scale factors)
            self.qk.stage_scales(qk_mma, q1_stage)
            self.qk.issue_qk(qk_mma, 1, tStS, self.qk.descs(tSrQ1, tSrK0), s1_out_handle)
            k_handle.release()
            # O_j must be drained by the previous work tile's epilogue before PV_j(0)
            o0_handle = mma_epi_producer.acquire_and_advance()
            o1_handle = mma_epi_producer.acquire_and_advance()
            pv_whether_acc = False
            kv_tok = load_kv_consumer.try_wait()
            kv_begin, n_local = self.kv_range(work_tile, n_kv_tiles)
            for i in cutlass.range(1, n_local, 1, unroll=1):
                # slot [K(i) | V(i-1)] (its full barrier was probed during the previous S0 wait)
                k_handle = load_kv_consumer.wait_and_advance(kv_tok)
                v_desc0 = self.v_desc(
                    v_addr0 + k_handle.index * (2 * self.kv_half_bytes),
                    ones_addr + self.vs_ones_offset(kv_begin + i - 1),
                )
                tSrKi = self.qk.k_operand(qk_mma, k_handle.index)
                k_stage = k_handle.index
                # The QK0 descriptors and the scale-factor copies are independent of the shared score tile.
                # Issue the async S2T copies before blocking on WG1's S-empty credit so that they run while
                # this warp waits.
                if cutlass.const_expr(not self.split and not self.qk.q_tmem):
                    # the Q descriptor of a QK is built from an opaque stage index at the point where the QK is
                    # issued (not in the split launch; with Q' in TMEM there is no Q descriptor)
                    qk0_descs = self.qk.descs(
                        self.qk.q_operand(qk_mma, opaque_i32(q0_handle.index)), tSrKi
                    )
                else:
                    qk0_descs = self.qk.descs(tSrQ0, tSrKi)
                self.qk.stage_scales(qk_mma, q0_stage, k_stage)
                # WG1 has had the preceding PV/TMA interval to drain S. Its
                # next empty credit is the overwrite fence for QK0.
                s1_out_handle = mma_s1_producer.acquire_and_advance()
                self.qk.issue_qk(qk_mma, 0, tStS, qk0_descs, s0_out_handle)
                # probe WG0's S-empty credit before PV0 (not after it)
                s0_tok = None
                if cutlass.const_expr(self.s_probe):
                    s0_tok = mma_s0_producer.try_acquire()
                # after PV_j(i - 1): the means MMA of the batch that ended with tile i - 2
                vs_issue = (i >= 2) & ((kv_begin + i - 1) & (self.vs_tiles - 1) == 0)
                vm_st = (((kv_begin + i - 1) >> self.vs_shift) - 1) & 1
                # P1's consumer phase at its probe below equals P0's before PV0 (both 1-stage, one advance per
                # PV, PV0 first): hand it the same value
                p_step_phase = p0_mma_consumer._PipelineConsumer__state._phase
                p0_mma_consumer = self.mma_pv(
                    pv_whether_acc,
                    (o0_addr, p0_addr, v_desc0, idesc_pv),
                    p0_mma_consumer,
                    p_tail_ptr,
                    (vs_issue, smem_base, 0, vm_st),
                )
                # PV0 overlaps WG0's score drain; consume its next-empty credit immediately before QK1
                # reuses the shared tile. P1(i-1) (needed right after QK1) and the next K/V slot are probed
                # back to back with it.
                if cutlass.const_expr(not self.split and not self.qk.q_tmem):
                    qk1_descs = self.qk.descs(
                        self.qk.q_operand(qk_mma, opaque_i32(q1_handle.index)), tSrKi
                    )
                else:
                    qk1_descs = self.qk.descs(tSrQ1, tSrKi)
                self.qk.stage_scales(qk_mma, q1_stage)
                p1_mma_consumer._PipelineConsumer__state._phase = p_step_phase
                p1_tok = p1_mma_consumer.try_wait()
                kv_tok = load_kv_consumer.try_wait()
                s0_out_handle = mma_s0_producer.acquire_and_advance(s0_tok)
                self.qk.issue_qk(qk_mma, 1, tStS, qk1_descs, s1_out_handle)
                p1_mma_consumer = self.mma_pv(
                    pv_whether_acc,
                    (o1_addr, p1_addr, v_desc0, idesc_pv),
                    p1_mma_consumer,
                    p_tail_ptr + 1,
                    (vs_issue, smem_base, 1, vm_st),
                    p1_tok,
                )
                pv_whether_acc = True
                if cutlass.const_expr(not self.split and self.cluster_shape_mn[0] > 1):
                    # the release arrives on the slot's empty barrier in both CTAs of the 2x1 cluster (mask 3)
                    with cute.arch.elect_one():
                        tcgen05.commit(
                            sptr(Int64, kv_empty_addr + opaque_i32(k_handle.index) * 8), 3
                        )
                else:
                    k_handle.release()
            q0_handle.release()
            q1_handle.release()
            v_handle = load_kv_consumer.wait_and_advance(kv_tok)
            vs_issue = (n_local >= 2) & ((kv_begin + n_local - 1) & (self.vs_tiles - 1) == 0)
            vm_last = (kv_begin + n_local - 1) >> self.vs_shift
            vm_st = (vm_last - 1) & 1
            v_desc0 = self.v_desc(
                v_addr0 + v_handle.index * (2 * self.kv_half_bytes),
                ones_addr + self.vs_ones_offset(kv_begin + n_local - 1),
            )
            p0_mma_consumer = self.mma_pv(
                pv_whether_acc,
                (o0_addr, p0_addr, v_desc0, idesc_pv),
                p0_mma_consumer,
                p_tail_ptr,
                (vs_issue, smem_base, 0, vm_st),
            )
            p1_mma_consumer = self.mma_pv(
                pv_whether_acc,
                (o1_addr, p1_addr, v_desc0, idesc_pv),
                p1_mma_consumer,
                p_tail_ptr + 1,
                (vs_issue, smem_base, 1, vm_st),
            )
            v_handle.release()
            # the last (possibly partial) batch, once softmax warpgroup j wrote the block sums of the last
            # tile (its v_mean came with the last V); then O_j goes to the epilogue warpgroup
            a_off, b_off, e_off = self.vs_offs
            for j, o_handle in ((0, o0_handle), (1, o1_handle)):
                cute.arch.barrier(
                    barrier_id=self.vs_bar_id + j, number_of_threads=self.vs_bar_threads
                )
                tc_mma_means(
                    o0_addr if j == 0 else o1_addr,
                    smem_base,
                    vm_last & 1,
                    a_off + j * MEANS_A_BYTES,
                    b_off,
                    MEANS_A_DESC,
                    MEANS_B_DESC,
                    MEANS_IDESC,
                )
                o_handle.commit()
            tc_commit_means(smem_base, vm_last & 1, e_off)
            # Reserve QK1's next logical credit and, more importantly,
            # prove the last QK1 score tile is no longer being read before
            # the next persistent work tile starts with QK0.
            s1_out_handle = mma_s1_producer.acquire_and_advance()
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()
