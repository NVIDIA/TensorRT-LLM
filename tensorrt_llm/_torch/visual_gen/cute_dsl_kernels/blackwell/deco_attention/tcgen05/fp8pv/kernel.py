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
"""The attention kernel of the tcgen05 targets (sm_100a, sm_103a, sm_107a): NVFP4 Q'K' + FP8 P x E4M3 V with BF16
output, in a ping-pong schedule. Persistent warp-specialized CTAs hold two 128-row Q tiles (one softmax warpgroup
each) that share one physical S tile in TMEM, with independent FP8 P tiles and O accumulators:

  warps 0-3, 4-7   softmax warpgroups 0 and 1: S_j -> P_j (softmax.py)
  warps 8-11       epilogue warpgroup: the epilogue of both O tiles (epilogue.py)
  warp 12          MMA warp: QK0, PV0, QK1, PV1 per KV tile (../qk.py, mma.py), K(i) and V(i-1) from one ring slot
  warp 13          load warp: the TMA loads of Q', K', V and the v_mean rows
  warp 14          store warp: the TMA stores of O
  warp 15          idle (it hands its registers to the softmax warpgroups)

This module holds the kernel object: its configuration (PingPongAttention.__init__), the host-side setup of a launch
(__call__: operand layouts, TMA atoms, the CTA's shared memory) and the CTA body (kernel: the setup of the CTA, then
one branch per warp role; the _run_* methods give the load, MMA and softmax roles their operands, the epilogue's loop
over the work tiles is in its branch, and the roles' loops and steps are the role methods of load.py, mma.py,
softmax.py and epilogue.py). The host entry is launch.py; the per-target schedule constants are in tuning.py.

The QK is the compile target's (../qk.py); the PV is issued in K phases of pv_k keys (mma.py). The PV runs with N =
144 against [V | ones], so the columns of O_j after V take the sums of P over each KV tile: the row sum of exactly the
P that the PV multiplies, and the weights of the means of V smoothing (mma.py)."""

import dataclasses
from typing import Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.nvgpu.common import OperandMajorMode
from cutlass.cute.typing import Float32, Int32, Int64
from cutlass.cutlass_dsl import min as dsl_min
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from ... import target
from ...common.dsl import Tcgen05Fence, tcgen05_fence
from ...common.mem import load_invariant
from ...common.state import State
from ...constants import HEAD_DIM
from .. import qk
from ..ptx import ld_global_nc_f32
from . import tuning
from .epilogue import EpilogueRole
from .load import LoadRole
from .mma import MEANS_A_BYTES, MEANS_TILER, MmaRole
from .sched import (
    SplitTileSchedParams,
    SplitTileScheduler,
    TileSchedParams,
    TileScheduler,
    thread_group,
)
from .softmax import SoftmaxRole


def _rebase_mbar_array(arr, off):
    """The same mbarrier array shifted by off barriers (no re-initialization)."""
    return arr.__new_from_mlir_values__([arr.barrier_storage + off])


def _rebase_pipe(pipe, off):
    """A pipeline over the barriers off entries after pipe's (its full and empty arrays shift together)."""
    return dataclasses.replace(
        pipe,
        sync_object_full=_rebase_mbar_array(pipe.sync_object_full, off),
        sync_object_empty=_rebase_mbar_array(pipe.sync_object_empty, off),
    )


class PingPongAttention(LoadRole, MmaRole, SoftmaxRole, EpilogueRole):
    def __init__(
        self, sm_count: int, rank: int, cluster_m: int = 1, split: bool = False, pdl: bool = True
    ):
        self.sm_count = sm_count
        # split: only the tail units after n_valid, each split along KV into n_parts pieces merged in the epilogue
        self.split = split
        # pdl: the split launch follows a main launch, so it launches as its programmatic dependent; without a main
        # launch it follows the kernels that wrote its scale and flags (the preparation) and must see their writes,
        # so it launches normally
        self.pdl = pdl
        self.qk_acc_dtype = Float32
        self.pv_acc_dtype = Float32
        self.v_dtype = cutlass.Float8E4M3FN
        self.o_dtype = cutlass.BFloat16
        mma_tiler = (128, 128)
        tune = tuning.tuning()
        # the QK component of the target at this compensation rank
        self.qk = qk.for_target(mma_tiler, rank)
        self.load_sleep_ns = tune.load_sleep_ns
        self.s_probe = tune.s_probe
        self.rescale_loop = tune.rescale_loop
        self.vs_batched = tune.vs_batched
        self.kv_stage = tuning.KV_STAGE
        # V smoothing: the PV leaves the sum of P over KV tile t in column 128 + t % 8 of O_j, read once per batch of
        # vs_tiles tiles. The sums become K slots of the means MMA's A operand (K slot = column), whose B takes the
        # batch's vs_rows v_mean rows from a ring of two stages
        self.vs_tiles = 8
        self.vs_shift = self.vs_tiles.bit_length() - 1
        self.vs_rows = 8
        self.head_dim_v = HEAD_DIM
        self.cta_tiler = (2 * mma_tiler[0], mma_tiler[1], self.head_dim_v)
        self.qk_mma_tiler = mma_tiler
        self.pv_mma_tiler = (mma_tiler[0], self.head_dim_v, mma_tiler[1])
        # keys per PV MMA (its K): 32, or 64 where the target has the K = 64 FP8 MMA
        self.pv_k = 64 if target.has_k64_fp8_mma() else 32
        # S loads that also return the row maximum (tcgen05.ld.red)
        self.use_ld_red = target.has_tmem_ld_red()
        # bytes of one K phase of a V stage (pv_k rows of 128 MN-major E4M3 channels), and of the ones block
        self.v_phase_bytes = 128 * self.pv_k
        # Two independent attention CTAs of the same head share K/V (and their scale factors) through TMA multicast
        # in a (2,1) cluster; Q, TMEM and all softmax/MMA handshakes stay local.
        self.cluster_shape_mn = (cluster_m, 1)
        # the TMEM allocation is sized for the compile target ("sm_100", "sm_103", ...)
        self.arch_str = target.arch_name()
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols(self.arch_str)

        self.softmax0_warp_ids = (0, 1, 2, 3)
        self.epilogue_warp_ids = (8, 9, 10, 11)
        self.mma_warp_id = 12
        self.load_warp_id = 13
        self.store_warp_id = 14
        self.idle_warp_id = 15

        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * 16
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_warp * (1 + 4 + 4 + 4),
        )
        self.split_barrier = pipeline.NamedBarrier(
            barrier_id=9, num_threads=self.threads_per_warp * 4
        )
        self.tmem_dealloc_barrier = pipeline.NamedBarrier(
            barrier_id=7, num_threads=self.threads_per_warp * 4
        )
        # Long waits block on named barriers instead of polling mbarriers:
        # softmax WG j -> epilogue WG (final row stats), epilogue WG -> store warp (sO ready)
        self.stats_bar_id = 3  # + stage
        self.stats_bar_threads = self.threads_per_warp * 8
        self.epi_bar_id = 5  # + stage
        self.epi_bar_threads = self.threads_per_warp * 5
        # softmax warpgroup j -> MMA warp: its last block sums of a work tile are in the means A operand
        self.vs_bar_id = 10  # + stage
        self.vs_bar_threads = self.threads_per_warp * 5

        # One physical S tile is alternated between the two logical score pipelines: each softmax warpgroup drains all
        # four 32-column chunks before releasing its logical pipeline, and the MMA warp consumes that next-empty
        # credit before overwriting S.
        self.tmem_s_offset = 0
        self.tmem_o0_offset = 224
        self.tmem_o1_offset = 368
        # Independent FP8 P tiles (32 columns each), followed by the QK component's scale factors
        # (columns 192..223).
        self.tmem_p0_offset = 128
        self.tmem_p1_offset = 160
        # Each O_j accumulator is 144 columns: the PV MMA runs with N = 144 against [V | ones], so the columns from
        # 128 on take sums of P (N must be a multiple of 16; both accumulators are 16-aligned).
        self.pv_n = HEAD_DIM + 16

        self.num_regs_softmax = 192
        self.num_regs_epilogue = 72
        self.num_regs_other = 56
        self.buffer_align_bytes = 1024

        # pipeline depths
        self.q_stage = 2
        self.p_mma_stage = 1
        self.softmax_epi_stage = 2
        self.mma_epi_stage = 2
        self.mma_softmax_stage = 1
        self.epi_stage = 2

    def _shared_storage(self, o_smem_layout_staged, qk_fields, vs_layouts):
        """The CTA's shared memory: pipeline barriers, the split launch's softmax -> epilogue exchange, the O staging
        tiles, the QK component's operand fields, then the constant block of the PV B operand's ones columns and the
        means MMA's barriers and operands (vs_layouts: their A and B layouts)."""
        MemRange, Align = cute.struct.MemRange, cute.struct.Align
        sMA_l, sMB_l = vs_layouts
        fields = dict(
            load_q_mbar_ptr=MemRange[Int64, self.q_stage * 2],
            load_kv_mbar_ptr=MemRange[Int64, self.kv_stage * 2],
            mma_s0_mbar_ptr=MemRange[Int64, self.mma_softmax_stage * 2],
            mma_s1_mbar_ptr=MemRange[Int64, self.mma_softmax_stage * 2],
            p0_mma_mbar_ptr=MemRange[Int64, self.p_mma_stage * 2],
            p1_mma_mbar_ptr=MemRange[Int64, self.p_mma_stage * 2],
            s0_epi_mbar_ptr=MemRange[Int64, self.softmax_epi_stage * 2],
            s1_epi_mbar_ptr=MemRange[Int64, self.softmax_epi_stage * 2],
            epi_store_mbar_ptr=MemRange[Int64, self.epi_stage * 2],
            mma_epi_mbar_ptr=MemRange[Int64, self.mma_epi_stage * 2],
            tmem_holding_buf=Int32,
            # softmax warpgroup j -> MMA warp: the last quarter of P_j is stored
            p_tail_mbar_ptr=MemRange[Int64, 2],
        )
        if self.split:
            # softmax -> epilogue exchange of the split launch (row, 2 values, stage, warpgroup): the final anchor
            # of a KV piece in value 0
            fields["sVec"] = Align[MemRange[Float32, 128 * 2 * 2 * 2], 16]
        fields.update(
            sO=Align[
                MemRange[self.o_dtype, cute.cosize(o_smem_layout_staged)], self.buffer_align_bytes
            ],
            **qk_fields,
            # constant E4M3 ones: the N columns 128..143 of the PV B operand (the 1 KB swizzle atoms one K phase
            # spans: four at K = 32, eight at K = 64)
            sOnes=Align[MemRange[cutlass.Uint8, self.v_phase_bytes], self.buffer_align_bytes],
            # the v_mean ring's empty barriers (its TMA completes on the K/V slot loaded with it)
            vm_mbar_ptr=MemRange[Int64, 2],
            # means MMA operands: one K16 operand of BF16 block sums per softmax warpgroup, the v_mean ring
            sMA=Align[MemRange[cutlass.BFloat16, cute.cosize(sMA_l)], self.buffer_align_bytes],
            sMB=Align[MemRange[cutlass.BFloat16, cute.cosize(sMB_l)], self.buffer_align_bytes],
        )
        return cute.struct(type("SharedStorage", (), {"__annotations__": fields}))

    @cute.jit
    def __call__(
        self,
        qk_ptrs: Tuple,
        v_ptr: cute.Pointer,
        v_mean_ptr: cute.Pointer,
        o_ptr: cute.Pointer,
        s: Int32,
        h: Int32,
        b: Int32,
        score_scale_addr: Int64,
        v_scale_addr: Int64,
        n_valid: Int32,
        n_parts: Int32,
        split_o_addr: Int64,
        split_ml_addr: Int64,
        split_flag_addr: Int64,
        stream: cuda.CUstream,
    ):
        dv = self.head_dim_v
        h_r = 1
        # Q, K (and their scale factors) as the QK component lays them out
        qk_tensors = self.qk.global_tensors(qk_ptrs, s, h, b)
        # V: [B, S, H, 128] -> (dv, s, ((h_r, h), b))
        v_layout = cute.make_layout(
            (dv, s, ((h_r, h), b)),
            stride=(1, dv * h, ((0, dv), s * h * dv)),
        )
        v = cute.make_tensor(v_ptr, v_layout)
        # O: [B, S, H, 128] -> (s, dv, ((h_r, h), b))
        o_layout = cute.make_layout(
            (s, dv, ((h_r, h), b)),
            stride=(dv * h, 1, ((dv, dv), s * h * dv)),
        )
        o = cute.make_tensor(o_ptr, o_layout)

        o_shape = cute.shape((s, dv, ((h_r, h), b)))
        problem_shape_mhb = (
            cute.ceil_div(cute.size(o_shape[0]), self.cta_tiler[0]),
            cute.size(o_shape[2][0]),
            cute.size(o_shape[2][1]),
        )
        if cutlass.const_expr(self.split):
            # n_valid whole units are done by the main kernel; this launch covers the rest, n_parts KV pieces each
            self.tile_sched_params = SplitTileSchedParams(
                problem_shape_mhb, n_valid, n_parts, cute.ceil_div(s, self.qk_mma_tiler[1])
            )
            n_items = n_valid + (cute.size(problem_shape_mhb) - n_valid) * n_parts
            grid = (dsl_min(self.sm_count, n_items - n_valid), 1, 1)
        else:
            self.tile_sched_params = TileSchedParams(problem_shape_mhb, n_valid)
            grid = (dsl_min(self.sm_count, n_valid), 1, 1)

        self.v_major_mode = OperandMajorMode.MN
        self.o_layout = utils.LayoutEnum.from_tensor(o)

        cta_group = tcgen05.CtaGroup.ONE
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = OperandMajorMode.K
        qk_tiled_mma = self.qk.make_tiled_mma(cta_group)
        pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.v_dtype,
            self.v_dtype,
            p_major_mode,
            self.v_major_mode,
            self.pv_acc_dtype,
            cta_group,
            self.pv_mma_tiler[:2],
            p_source,
        )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (qk_tiled_mma.thr_id.shape,),
        )
        self.epi_tile = self.pv_mma_tiler[:2]

        q_smem_layout_staged, k_stage_layout, qk_sf_smem_layouts = self.qk.smem_layouts(
            qk_tiled_mma, self.q_stage, self.kv_stage
        )
        v_smem_layout_staged_origin = sm100_utils.make_smem_layout_b(
            pv_tiled_mma, self.pv_mma_tiler, self.v_dtype, self.kv_stage
        )
        # K(i) and V(i-1) share a ring slot: both stages must have the same footprint, and both views get one
        # common byte stride so the K/V pipeline can alias them.
        v_stage_layout = cute.select(v_smem_layout_staged_origin, mode=[0, 1, 2])
        k_stage_bytes = cute.size_in_bytes(self.qk.k_smem_dtype, k_stage_layout)
        v_stage_bytes = cute.size_in_bytes(self.v_dtype, v_stage_layout)
        if cutlass.const_expr(k_stage_bytes != v_stage_bytes):
            raise RuntimeError("K and V stages must have the same 16 KiB footprint")
        kv_stage_bytes = k_stage_bytes
        self.kv_half_bytes = kv_stage_bytes
        # a ring slot is [K(i) | V(i-1)]: V sits kv_stage_bytes after K, slots are 2 * kv_stage_bytes apart
        k_smem_layout_staged = cute.append(
            k_stage_layout,
            cute.make_layout(
                self.kv_stage, stride=2 * kv_stage_bytes * 8 // self.qk.k_smem_dtype.width
            ),
        )
        v_smem_layout_staged = cute.append(
            v_stage_layout,
            cute.make_layout(
                self.kv_stage,
                stride=2 * kv_stage_bytes * 8 // self.v_dtype.width,
            ),
        )
        sK_cosize = self.kv_stage * 2 * kv_stage_bytes

        o_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.o_dtype, self.o_layout, self.epi_tile, self.epi_stage
        )

        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_kv_load_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, qk_tiled_mma.thr_id
        )
        tma_store_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()

        qk_atoms, self.tma_copy_q_bytes, self.tma_copy_k_bytes = self.qk.tma_atoms(
            tma_load_op,
            tma_kv_load_op,
            qk_tensors,
            q_smem_layout_staged,
            k_smem_layout_staged,
            qk_sf_smem_layouts,
            qk_tiled_mma,
            self.cluster_shape_mn,
            self.cluster_layout_vmnk,
            s,
            h,
            b,
        )
        tma_atom_q, tma_tensor_q, q_sf_tma, tma_atom_k, tma_tensor_k, k_sf_tma = qk_atoms
        v_smem_layout = cute.select(v_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
            tma_kv_load_op,
            v,
            v_smem_layout,
            self.pv_mma_tiler,
            pv_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        o_smem_layout = cute.select(o_smem_layout_staged, mode=[0, 1])
        tma_atom_o, tma_tensor_o = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_store_op, o, o_smem_layout, self.epi_tile
        )
        self.tma_copy_v_bytes = cute.size_in_bytes(self.v_dtype, v_smem_layout)

        # means MMA of V smoothing: O_j[:, 0..127] += (K16 BF16 block sums, K-major in SMEM) x (16 v_mean rows,
        # MN-major in SMEM; the batches fill K slots 0..7 of both)
        bf16 = cutlass.BFloat16
        mn_mma = cute.make_tiled_mma(
            tcgen05.MmaF16BF16Op(
                bf16,
                Float32,
                MEANS_TILER,
                cta_group,
                tcgen05.OperandSource.SMEM,
                OperandMajorMode.K,
                OperandMajorMode.MN,
            )
        )
        # A: one stage per softmax warpgroup (32-byte swizzle); B: the v_mean ring
        sMA_l = sm100_utils.make_smem_layout_a(mn_mma, MEANS_TILER, bf16, 2)
        sMB_l = sm100_utils.make_smem_layout_b(mn_mma, MEANS_TILER, bf16, 2)
        # the first vs_rows rows of the B stages as (channel, row, stage) for the TMA of a batch's v_mean rows
        sMB_tma_l = cute.make_composed_layout(
            sMB_l.inner,
            0,
            cute.make_layout(((64, 2), self.vs_rows, 2), stride=((1, 1024), 64, 2048)),
        )
        # v_mean [B, H, ceil(S / 128), 128] -> (128, rows, ((h_r, h), b)): one row per KV tile, loaded vs_rows rows
        # at a time (the TMA fills the rows of a last batch past the last tile with zeros)
        vm_rows = cute.ceil_div(s, self.qk_mma_tiler[1])
        mVM = cute.make_tensor(
            v_mean_ptr,
            cute.make_layout(
                (dv, vm_rows, ((h_r, h), b)), stride=(1, dv, ((0, vm_rows * dv), h * vm_rows * dv))
            ),
        )
        tma_atom_vm, tma_tensor_vm = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_load_op,
            mVM,
            cute.select(sMB_tma_l, mode=[0, 1]),
            cute.composition(cute.make_identity_layout(mVM.shape), (dv, self.vs_rows)),
        )
        self.tma_copy_vm_bytes = dv * self.vs_rows * 2
        vs_args = (tma_atom_vm, tma_tensor_vm, sMB_tma_l)

        self.shared_storage = self._shared_storage(
            o_smem_layout_staged,
            self.qk.storage_fields(
                q_smem_layout_staged, sK_cosize, qk_sf_smem_layouts, self.buffer_align_bytes
            ),
            (sMA_l, sMB_l),
        )
        if cutlass.const_expr(self.shared_storage.size_in_bytes() + 1024 > 256 * 1024):
            raise ValueError(
                f"shared memory of {self.shared_storage.size_in_bytes()} B exceeds the range that the kernel's "
                "shared-memory descriptors address"
            )
        # shared memory offsets of the means MMA operands and the v_mean ring's empty barriers
        offs = self.shared_storage._offsets
        assert offs["load_q_mbar_ptr"] == 0  # the base the MMA warp offsets from
        self.vs_offs = (offs["sMA"], offs["sMB"], offs["vm_mbar_ptr"])

        self.kernel(
            qk_tiled_mma,
            pv_tiled_mma,
            tma_atom_q,
            tma_tensor_q,
            q_sf_tma,
            tma_atom_k,
            tma_tensor_k,
            k_sf_tma,
            tma_atom_v,
            tma_tensor_v,
            tma_atom_o,
            tma_tensor_o,
            score_scale_addr,
            v_scale_addr,
            q_smem_layout_staged,
            k_smem_layout_staged,
            qk_sf_smem_layouts,
            v_smem_layout_staged,
            o_smem_layout_staged,
            self.tile_sched_params,
            (split_o_addr, split_ml_addr, split_flag_addr),
            vs_args,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
            # the KV-split launch reads only the inputs and writes other O rows than the main launch, so it is
            # launched as its programmatic dependent
            use_pdl=self.split and self.pdl,
        )

    def _make_pipelines(self, storage, cluster_layout_vmnk):
        """The pipelines of the CTA on the barrier arrays of the shared storage, as an object with one attribute per
        participant the roles use. The three *_pipe attributes are whole pipelines: the softmax role rebases those of
        warpgroup 0 to its own warpgroup; `epilogue` is the tuple of the epilogue warpgroup's participants (the
        consumers of s0_epi, s1_epi and mma_epi, the producer of epi_store).

          load_q, load_kv     load warp (TMA) -> MMA warp: the Q' stages and the K/V ring
          mma_s0, mma_s1      MMA warp (QK_j) -> softmax warpgroup j: S
          p0_mma, p1_mma      softmax warpgroup j -> MMA warp (PV_j): P_j
          s0_epi, s1_epi      softmax warpgroup j -> epilogue warpgroup: the final row statistics
          mma_epi             MMA warp (last PV_j) -> epilogue warpgroup: O_j
          epi_store           epilogue warpgroup -> store warp: the staged O tiles

        Every pipeline defers the synchronization of its barrier initialization to the kernel (pipeline_init_arrive,
        pipeline_init_wait)."""
        load_q_producer, load_q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.q_stage,
            producer_group=thread_group(1),
            consumer_group=thread_group(1),
            tx_count=self.tma_copy_q_bytes,
            barrier_storage=storage.load_q_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        load_kv_producer, load_kv_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.kv_stage,
            producer_group=thread_group(1),
            consumer_group=(
                pipeline.CooperativeGroup(pipeline.Agent.Warp)
                if self.cluster_shape_mn[0] > 1
                else thread_group(1)
            ),
            tx_count=self.tma_copy_k_bytes + self.tma_copy_v_bytes,
            barrier_storage=storage.load_kv_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            # K, KSF and V are B operands shared along cluster M: with multicast signalling each CTA's empty
            # barrier expects both MMA warps' tcgen05 completion arrivals before a ring slot is refilled.
            mcast_mode_mn=(0, 1),
            enable_multicast_signaling=self.cluster_shape_mn[0] > 1,
            defer_sync=True,
        ).make_participants()
        mma_s0_pipe = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_softmax_stage,
            producer_group=thread_group(1),
            consumer_group=thread_group(self.threads_per_warp * 4),
            barrier_storage=storage.mma_s0_mbar_ptr.data_ptr(),
            defer_sync=True,
        )
        mma_s0_producer, _ = mma_s0_pipe.make_participants()
        mma_s1_producer, _ = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_softmax_stage,
            producer_group=thread_group(1),
            consumer_group=thread_group(self.threads_per_warp * 4),
            barrier_storage=storage.mma_s1_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        p0_mma_pipe = pipeline.PipelineAsyncUmma.create(
            num_stages=self.p_mma_stage,
            producer_group=thread_group(self.threads_per_warp * 4),
            consumer_group=thread_group(1),
            barrier_storage=storage.p0_mma_mbar_ptr.data_ptr(),
            defer_sync=True,
        )
        _, p0_mma_consumer = p0_mma_pipe.make_participants()
        _, p1_mma_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.p_mma_stage,
            producer_group=thread_group(self.threads_per_warp * 4),
            consumer_group=thread_group(1),
            barrier_storage=storage.p1_mma_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        s0_epi_pipe = pipeline.PipelineAsync.create(
            num_stages=self.softmax_epi_stage,
            producer_group=thread_group(self.threads_per_warp * 4),
            consumer_group=thread_group(self.threads_per_warp * 4),
            barrier_storage=storage.s0_epi_mbar_ptr.data_ptr(),
            defer_sync=True,
        )
        _, s0_epi_consumer = s0_epi_pipe.make_participants()
        _, s1_epi_consumer = pipeline.PipelineAsync.create(
            num_stages=self.softmax_epi_stage,
            producer_group=thread_group(self.threads_per_warp * 4),
            consumer_group=thread_group(self.threads_per_warp * 4),
            barrier_storage=storage.s1_epi_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        epi_store_producer, epi_store_consumer = pipeline.PipelineAsync.create(
            num_stages=self.epi_stage,
            producer_group=thread_group(self.threads_per_warp * 4),
            consumer_group=thread_group(self.threads_per_warp * 1),
            barrier_storage=storage.epi_store_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        mma_epi_producer, mma_epi_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_epi_stage,
            producer_group=thread_group(1),
            consumer_group=thread_group(self.threads_per_warp * 4),
            barrier_storage=storage.mma_epi_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        return State(
            load_q_producer=load_q_producer,
            load_q_consumer=load_q_consumer,
            load_kv_producer=load_kv_producer,
            load_kv_consumer=load_kv_consumer,
            mma_s0_pipe=mma_s0_pipe,
            mma_s0_producer=mma_s0_producer,
            mma_s1_producer=mma_s1_producer,
            p0_mma_pipe=p0_mma_pipe,
            p0_mma_consumer=p0_mma_consumer,
            p1_mma_consumer=p1_mma_consumer,
            s0_epi_pipe=s0_epi_pipe,
            epi_store_consumer=epi_store_consumer,
            mma_epi_producer=mma_epi_producer,
            # the participants of the epilogue warpgroup, which its loop over the work tiles carries
            epilogue=(s0_epi_consumer, s1_epi_consumer, mma_epi_consumer, epi_store_producer),
        )

    def _smem_tensors(self, storage, q_layout, k_layout, qk_sf_layouts, v_layout, o_layout):
        """The tensors of the shared storage by their staged layouts: the Q' stages, the K' halves of the K/V ring
        slots, the QK component's scale factors, the V halves of the ring slots (V sits kv_half_bytes after K in a
        slot), the O staging tiles and, in the split launch, the anchor exchange (row, 2 values, stage, warpgroup)."""
        sQ, sK, qk_sf_smem = self.qk.smem_tensors(storage, q_layout, k_layout, qk_sf_layouts)
        sV_ptr = cute.recast_ptr(
            cute.recast_ptr(storage.sK.data_ptr() + self.kv_half_bytes, dtype=self.v_dtype),
            v_layout.inner,
        )
        sV = cute.make_tensor(sV_ptr, v_layout.outer)
        sO = storage.sO.get_tensor(o_layout.outer, swizzle=o_layout.inner)
        sVec = None
        if cutlass.const_expr(self.split):
            sVec = storage.sVec.get_tensor(
                cute.make_layout((128, 2, 2, 2), stride=(2, 1, 256, 512))
            )
        return sQ, sK, qk_sf_smem, sV, sO, sVec

    def _role_storage(self, st):
        """The shared storage as the calling role's own instance: its field pointers are derived at this call, in the
        branch of the role."""
        return self.shared_storage(st.storage.base)

    def _tile_scheduler(self, tile_sched_params):
        """The scheduler of the work tiles of this CTA; in the split launch its work items come after the n_full whole
        units of the main launch."""
        if cutlass.const_expr(self.split):
            return SplitTileScheduler(
                tile_sched_params,
                cute.arch.block_idx()[0] + tile_sched_params.n_full,
                cute.arch.grid_dim(),
            )
        return TileScheduler(tile_sched_params, cute.arch.block_idx()[0], cute.arch.grid_dim())

    @cute.kernel
    def kernel(
        self,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom,
        mQ_qdl: cute.Tensor,
        q_sf_tma: Tuple,
        tma_atom_k: cute.CopyAtom,
        mK_kdl: cute.Tensor,
        k_sf_tma: Tuple,
        tma_atom_v: cute.CopyAtom,
        mV_dkl: cute.Tensor,
        tma_atom_o: cute.CopyAtom,
        mO_qdl: cute.Tensor,
        score_scale_addr: Int64,
        v_scale_addr: Int64,
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        qk_sf_smem_layouts: Tuple,
        v_smem_layout_staged: cute.ComposedLayout,
        o_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: TileSchedParams,
        split_args: Tuple,
        vs_args: Tuple,
    ):
        """The CTA body: the setup every thread runs (cluster, shared storage, pipelines, barriers, TMEM, shared-memory
        tensors, the tile scheduler), then the role of the thread's warp over the work tiles of the CTA. The roles
        read what the setup made from one state object: the _run_* methods assemble the arguments of the load, MMA
        and softmax roles from it, and the epilogue's steps take it."""
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        if cutlass.const_expr(not self.split):
            # nothing in the dependent KV-split launch reads this launch's output: release it right away
            cute.arch.griddepcontrol_launch_dependents()
        # the layout of self.cluster_layout_vmnk, built again: the host's value belongs to the host function
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (qk_tiled_mma.thr_id.shape,)
        )
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cta_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank)
        kv_cta_layout = cute.make_layout(self.cluster_shape_mn[0])
        kv_mcast_mask = None
        if cutlass.const_expr(self.cluster_shape_mn[0] > 1):
            kv_mcast_mask = cute.nvgpu.cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, cta_coord_vmnk, mcast_mode=1
            )

        if warp_idx == self.load_warp_id:
            self.qk.prefetch_descriptors(tma_atom_q, q_sf_tma, tma_atom_k, k_sf_tma)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_o)
            cute.nvgpu.cpasync.prefetch_descriptor(vs_args[0])

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        pipes = self._make_pipelines(storage, cluster_layout_vmnk)
        p_tail_ptr = storage.p_tail_mbar_ptr.data_ptr()
        # (computed here, outside the warp-role branches)
        tma_atom_vm, tma_tensor_vm, sMB_tma_l = vs_args
        sMB_tma = storage.sMB.get_tensor(sMB_tma_l.outer, swizzle=sMB_tma_l.inner)
        sMA_addr = Int32(storage.sMA.data_ptr().toint())
        sMB_addr = Int32(storage.sMB.data_ptr().toint())
        vm_empty = storage.vm_mbar_ptr.data_ptr()
        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(p_tail_ptr, self.threads_per_warp * 4)
                cute.arch.mbarrier_init(p_tail_ptr + 1, self.threads_per_warp * 4)
                # v_mean ring stage empty: the MMA warp's commit
                for ring_stage in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(vm_empty + ring_stage, 1)
            cute.arch.mbarrier_init_fence()
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_ids[0],
            arch=self.arch_str,
        )
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sQ, sK, qk_sf_smem, sV, sO, sVec = self._smem_tensors(
            storage,
            q_smem_layout_staged,
            k_smem_layout_staged,
            qk_sf_smem_layouts,
            v_smem_layout_staged,
            o_smem_layout_staged,
        )

        qk_thr_mma = qk_tiled_mma.get_slice(0)
        pv_thr_mma = pv_tiled_mma.get_slice(0)
        # TMEM accumulator layouts of S (one physical tile, which both QKs write) and of O_j: each role builds the
        # views it uses from the allocation base
        s_tmem_layout = qk_thr_mma.make_fragment_C(
            qk_thr_mma.partition_shape_C(self.qk_mma_tiler)
        ).layout
        o_tmem_layout = pv_thr_mma.make_fragment_C(
            pv_thr_mma.partition_shape_C(self.pv_mma_tiler[:2])
        ).layout

        tile_sched = self._tile_scheduler(tile_sched_params)
        work_tile = tile_sched.get_current_work()
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        # TMEM: the allocator warp allocates, then every warp of the CTA passes one barrier at this common PC
        tmem.allocate(self.num_tmem_alloc_cols)
        tcgen05_fence(Tcgen05Fence.BEFORE_THREAD_SYNC)
        cute.arch.barrier()
        tcgen05_fence(Tcgen05Fence.AFTER_THREAD_SYNC)

        seqlen_k = mK_kdl.shape[0]
        n_kv_tiles = cute.ceil_div(seqlen_k, self.qk_mma_tiler[1])

        st = State(
            warp_idx=warp_idx,
            tidx=tidx,
            # the global operands the roles load: TMA atoms and tensors, the addresses of the device scalars and of
            # the split launch's exchange buffers
            tma_atom_q=tma_atom_q,
            mQ_qdl=mQ_qdl,
            q_sf_tma=q_sf_tma,
            tma_atom_k=tma_atom_k,
            mK_kdl=mK_kdl,
            k_sf_tma=k_sf_tma,
            tma_atom_v=tma_atom_v,
            mV_dkl=mV_dkl,
            tma_atom_vm=tma_atom_vm,
            tma_tensor_vm=tma_tensor_vm,
            score_scale_addr=score_scale_addr,
            v_scale_addr=v_scale_addr,
            split_args=split_args,
            # the CTA in its cluster
            cta_coord_vmnk=cta_coord_vmnk,
            kv_cta_layout=kv_cta_layout,
            kv_mcast_mask=kv_mcast_mask,
            # shared memory: the storage (a role reads its fields from its own instance, _role_storage), its
            # pipelines and P-tail barriers, the operand stages, the O staging tiles, the split launch's anchor
            # exchange, the means MMA's operands and the v_mean ring's empty barriers
            storage=storage,
            pipes=pipes,
            p_tail_ptr=p_tail_ptr,
            sQ=sQ,
            sK=sK,
            qk_sf_smem=qk_sf_smem,
            qk_sf_smem_layouts=qk_sf_smem_layouts,
            sV=sV,
            sO=sO,
            sVec=sVec,
            sMB_tma=sMB_tma,
            sMA_addr=sMA_addr,
            sMB_addr=sMB_addr,
            vm_empty=vm_empty,
            # the MMAs and tensor memory
            qk_tiled_mma=qk_tiled_mma,
            qk_thr_mma=qk_thr_mma,
            pv_thr_mma=pv_thr_mma,
            tmem=tmem,
            s_tmem_layout=s_tmem_layout,
            o_tmem_layout=o_tmem_layout,
            # the work of the CTA
            tile_sched=tile_sched,
            work_tile=work_tile,
            seqlen_k=seqlen_k,
            n_kv_tiles=n_kv_tiles,
        )

        # One role per warp, each with its register budget. The idle warp only hands its registers to the softmax
        # warpgroups.
        if warp_idx == self.idle_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
        if warp_idx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self._run_load_warp(st)
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self._run_mma_warp(st)
        if warp_idx == self.store_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self.store_warp(
                mO_qdl, tma_atom_o, sO, pipes.epi_store_consumer, (tile_sched, work_tile)
            )
        if warp_idx < self.epilogue_warp_ids[0]:
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            self._run_softmax(st)
        if warp_idx >= self.epilogue_warp_ids[0] and warp_idx < self.mma_warp_id:
            # The epilogue warpgroup: both O tiles of every work tile, then the release of tensor memory. Its loop
            # is in this branch: it carries the warpgroup's pipeline participants and the scheduler's position.
            cute.arch.setmaxregister_decrease(self.num_regs_epilogue)
            epi = self._epilogue_operands(st, tmem)
            epi_pipes = pipes.epilogue
            while work_tile.is_valid_tile:
                epi_pipes = self._epilogue_work_tile(st, epi, epi_pipes, work_tile)
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            tmem.relinquish_alloc_permit()
            self.tmem_dealloc_barrier.arrive_and_wait()
            tmem.free(epi.tmem_ptr)
        if cutlass.const_expr(self.cluster_shape_mn[0] > 1):
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
        return

    @cute.jit
    def _run_load_warp(self, st):
        """The load warp (load.py) with its operands: the TMA atoms, global tensors and shared-memory stages of Q', K'
        and V, the v_mean ring, the Q and K/V producers with the K/V ring's empty barriers, and the CTA's place in
        its cluster for the K/V multicast."""
        storage = self._role_storage(st)
        self.load_warp(
            (
                st.tma_atom_q,
                st.mQ_qdl,
                st.q_sf_tma,
                st.tma_atom_k,
                st.mK_kdl,
                st.k_sf_tma,
                st.sQ,
                st.sK,
                st.qk_sf_smem,
                st.qk_thr_mma,
            ),
            (st.tma_atom_v, st.mV_dkl, st.sV, st.pv_thr_mma),
            (st.tma_atom_vm, st.tma_tensor_vm, st.sMB_tma, st.sMB_addr, st.vm_empty),
            (
                st.pipes.load_q_producer,
                st.pipes.load_kv_producer,
                storage.load_kv_mbar_ptr.data_ptr() + self.kv_stage,
            ),
            (st.cta_coord_vmnk, st.kv_cta_layout, st.kv_mcast_mask),
            (st.tile_sched, st.work_tile),
            st.n_kv_tiles,
            st.tidx,
        )

    @cute.jit
    def _run_mma_warp(self, st):
        """The MMA warp (mma.py) with its operands: the QK component's MMA setup on the S tile, the TMEM addresses of
        O_j and P_j, the shared addresses the PV and means MMAs take, the ones block, and the pipelines the warp
        produces into and consumes from."""
        storage, pipes = self._role_storage(st), st.pipes
        tmem_ptr = st.tmem.retrieve_ptr(self.qk_acc_dtype)
        tStS = cute.make_tensor(tmem_ptr + self.tmem_s_offset, st.s_tmem_layout)
        qk_mma = self.qk.mma_setup(
            st.qk_tiled_mma, tmem_ptr, st.sQ, st.sK, st.qk_sf_smem, st.qk_sf_smem_layouts
        )

        idesc_pv = self.pv_idesc()
        v_addr0 = Int32(storage.sK.data_ptr().toint()) + self.kv_half_bytes
        sq_addr = None
        if cutlass.const_expr(self.qk.q_tmem):
            sq_addr = Int32(storage.sQ.data_ptr().toint())
        # K/V ring slot empty barriers (the load warp's kv_empty_ptr)
        kv_empty_addr = Int32(storage.load_kv_mbar_ptr.data_ptr().toint()) + self.kv_stage * 8
        ones_addr = Int32(storage.sOnes.data_ptr().toint())
        smem_base = Int32(storage.load_q_mbar_ptr.data_ptr().toint())  # the first field: offset 0
        tmem_base = Int32(tmem_ptr.toint())
        o0_addr = tmem_base + self.tmem_o0_offset
        o1_addr = tmem_base + self.tmem_o1_offset
        p0_addr = tmem_base + self.tmem_p0_offset
        p1_addr = tmem_base + self.tmem_p1_offset
        ones_u32 = cute.make_tensor(
            cute.recast_ptr(storage.sOnes.data_ptr(), dtype=Int32),
            cute.make_layout(self.v_phase_bytes // 4),
        )
        self.mma_warp(
            qk_mma,
            tStS,
            (tmem_base, o0_addr, o1_addr, p0_addr, p1_addr, idesc_pv),
            (v_addr0, sq_addr, kv_empty_addr, ones_addr, smem_base, st.p_tail_ptr),
            ones_u32,
            (
                pipes.mma_s0_producer,
                pipes.mma_s1_producer,
                pipes.load_q_consumer,
                pipes.load_kv_consumer,
                pipes.mma_epi_producer,
                pipes.p0_mma_consumer,
                pipes.p1_mma_consumer,
            ),
            (st.tile_sched, st.work_tile),
            st.n_kv_tiles,
            st.tidx,
        )

    @cute.jit
    def _run_softmax(self, st):
        """The softmax warpgroup of the thread's warp (softmax.py). Both softmax warpgroups run one traced body: the
        warpgroup's S/P/stats pipelines are the S0/P0/stats0 ones rebased by the stage at runtime (the stage-1
        barriers sit right after the stage-0 ones in SMEM), and so are its P/O TMEM columns."""
        storage, pipes = self._role_storage(st), st.pipes
        # log2-domain softmax multiplier of the raw scores, a device scalar (the score scale from the
        # preparation); never read on the host
        score_scale = load_invariant(Float32, st.score_scale_addr)
        stage = st.warp_idx // len(self.softmax0_warp_ids)
        tmem_ptr = st.tmem.retrieve_ptr(self.qk_acc_dtype)
        tStS = cute.make_tensor(tmem_ptr + self.tmem_s_offset, st.s_tmem_layout)
        tOtOi = cute.make_tensor(
            tmem_ptr + self.tmem_o0_offset + stage * (self.tmem_o1_offset - self.tmem_o0_offset),
            st.o_tmem_layout,
        )
        _, mma_si_consumer = _rebase_pipe(pipes.mma_s0_pipe, stage * 2).make_participants()
        pi_mma_producer, _ = _rebase_pipe(pipes.p0_mma_pipe, stage * 2).make_participants()
        si_epi_producer, _ = _rebase_pipe(
            pipes.s0_epi_pipe, stage * 2 * self.softmax_epi_stage
        ).make_participants()
        # the addresses of K slots 0..7 and 8..15 of this thread's row of its warpgroup's means A operand
        # (K-major BF16 rows of 32 bytes, 32-byte swizzle: the two 16-byte halves of row r swap when bit 2 of
        # r is set) and the TMEM address of its lane's O_j columns from HEAD_DIM on
        row = st.tidx % (self.threads_per_warp * 4)
        a_row = st.sMA_addr + stage * MEANS_A_BYTES + row * 32
        swz = ((row >> 2) & 1) * 16
        vs_row = (
            a_row + swz,
            a_row + (16 ^ swz),
            Int32(tmem_ptr.toint())
            + (((st.warp_idx % 4) * 32) << 16)
            + self.tmem_o0_offset
            + stage * (self.tmem_o1_offset - self.tmem_o0_offset)
            + HEAD_DIM,
        )
        self.softmax(
            stage=stage,
            tensor_args=(
                tStS,
                st.sVec[None, None, None, stage] if self.split else None,
                tOtOi,
                st.pv_thr_mma,
                # the warpgroup's own P-empty barrier (committed after each PV_j)
                storage.p0_mma_mbar_ptr.data_ptr() + self.p_mma_stage + stage * 2,
                st.p_tail_ptr,
            ),
            pipeline_args=(mma_si_consumer, si_epi_producer, pi_mma_producer),
            qk_thr_mma=st.qk_thr_mma,
            value_args=(st.seqlen_k, st.n_kv_tiles, score_scale),
            sched_args=(st.tile_sched, st.work_tile),
            vs_row=vs_row,
        )

    def _epilogue_operands(self, st, tmem):
        """What the epilogue warpgroup reads for every work tile: the device scalars (score scale, output scale), the
        TMEM base (retrieved from tmem, the allocator), the O_0 and O_1 accumulators and, in the split launch, this
        thread's rows of the anchor exchange of softmax warpgroups 0 and 1."""
        score_scale = ld_global_nc_f32(st.score_scale_addr)
        scale_output = load_invariant(Float32, st.v_scale_addr)
        tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
        tOtO0 = cute.make_tensor(tmem_ptr + self.tmem_o0_offset, st.o_tmem_layout)
        tOtO1 = cute.make_tensor(tmem_ptr + self.tmem_o1_offset, st.o_tmem_layout)
        vec0 = vec1 = None
        if cutlass.const_expr(self.split):
            thread_idx = st.tidx % (self.threads_per_warp * len(self.epilogue_warp_ids))
            vec0 = st.sVec[thread_idx, None, None, 0]
            vec1 = st.sVec[thread_idx, None, None, 1]
        return State(
            score_scale=score_scale,
            scale_output=scale_output,
            tmem_ptr=tmem_ptr,
            tOtO0=tOtO0,
            tOtO1=tOtO1,
            vec0=vec0,
            vec1=vec1,
        )

    @cute.jit
    def _epilogue_work_tile(self, st, epi, epi_pipes, work_tile):
        """The epilogue of both O tiles of a work tile (epilogue.py), with the operands of _epilogue_operands (epi).
        epi_pipes: the consumers of the row statistics of softmax warpgroups 0 and 1 and of O, and the producer of the
        staged O tiles; returns them as they are after the work tile."""
        s0_epi_consumer, s1_epi_consumer, mma_epi_consumer, epi_store_producer = epi_pipes
        if cutlass.const_expr(self.split):
            # both O tiles in one pass: publish first, then merge (the two merges run on different SMs)
            s0_epi_consumer, s1_epi_consumer, mma_epi_consumer, epi_store_producer = (
                self.epilogue_split(
                    st.pv_thr_mma,
                    (epi.tOtO0, epi.tOtO1, epi.vec0, epi.vec1, st.sO),
                    (
                        s0_epi_consumer,
                        s1_epi_consumer,
                        mma_epi_consumer,
                        epi_store_producer,
                    ),
                    epi.scale_output,
                    (work_tile.tile_idx[5], work_tile.tile_idx[6], work_tile.tile_idx[7]),
                    st.split_args,
                    epi.score_scale,
                )
            )
        else:
            # one O tile at a time (the softmax warpgroups rescale O themselves)
            s0_epi_consumer, mma_epi_consumer, epi_store_producer = self.epilogue_tile(
                st.pv_thr_mma,
                (epi.tOtO0, st.sO[None, None, 0]),
                (s0_epi_consumer, mma_epi_consumer, epi_store_producer),
                epi.scale_output,
                0,
            )
            s1_epi_consumer, mma_epi_consumer, epi_store_producer = self.epilogue_tile(
                st.pv_thr_mma,
                (epi.tOtO1, st.sO[None, None, 1]),
                (s1_epi_consumer, mma_epi_consumer, epi_store_producer),
                epi.scale_output,
                1,
            )
        return s0_epi_consumer, s1_epi_consumer, mma_epi_consumer, epi_store_producer

    def kv_range(self, work_tile, n_kv_tiles):
        """(first KV tile, KV tile count) of a work item: the whole row, or one KV piece of a split unit."""
        if cutlass.const_expr(self.split):
            return work_tile.tile_idx[3], work_tile.tile_idx[4]
        return 0, n_kv_tiles
