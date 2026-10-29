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
"""FP8 query of the absorbed NoPE MLA decode as a persistent tcgen05 BMM (SM100/SM103/SM107).

Computes, for a query ``q[T, H, nope + rope]`` (BF16) and the absorbed key weight
``w_kc[H, lora, nope]`` (BF16)::

    out[t, h, :lora] = e4m3(clamp(q[t, h, :nope] @ w_kc[h].T, -448, 448))
    out[t, h, lora:] = e4m3(clamp(float(q[t, h, nope:]), -448, 448))

into ``out[T, H, lora + rope]`` (FP8 E4M3), the contract of
:func:`tensorrt_llm._torch.modules.mla_fp8_q_absorb.mla_fp8_q_absorb_nope`.

The absorbed part is a batched GEMM over heads (``M`` = tokens, ``N`` = lora, ``K`` = nope)
run by :class:`K3MlaQAbsorbKernel`, a warp-specialized persistent kernel built on
:class:`PersistentDenseGemmKernel` (TMA warp, MMA warp, four epilogue warps):
``tcgen05.mma`` kind::f16 with FP32 accumulation in tensor memory, double-buffered
accumulators, and a TMA-store epilogue in 64-column sub-tiles. Its stage counts leave
:data:`SMEM_HEADROOM` of shared memory on each SM so that the MLA RoPE / KV-append kernel on the
aux stream runs beside the persistent CTAs instead of after the whole grid. Each output element
is the same K-ordered chain of 16-wide MMA steps as the Triton kernel's ``tl.dot``; the
epilogue takes ``max.f32(acc, -448)`` (NaN -> -448) and converts with ``cvt.rn.satfinite``
(clamps the rest to +-448), which is the Triton kernel's ``min(max(acc, -448), 448)`` rounded to
nearest even, so the bytes are identical. Four more warps per CTA copy ``q_pe`` (grid-stride
over 16-byte chunks, overlapping the GEMM). With PDL the TMA warp issues the weight stages of
its first tile before ``griddepcontrol.wait``.
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import torch
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from .custom_pipeline import PipelineTmaUmma, PipelineUmmaAsync
from .dense_gemm_persistent import PersistentDenseGemmKernel, _compute_stages
from .utils import griddepcontrol_launch_dependents, griddepcontrol_wait

#: Largest token count the op accepts.
MAX_TOKENS = 16384
#: Epilogue rows per CTA tile (UMMA M per CTA).
CTA_M = 128
#: q_pe elements per 16-byte load.
PE_VEC = 8
#: Shared memory left free per SM. The MLA RoPE / KV-append kernel that runs next to this one on
#: the aux stream (``applyMLARopeAndAssignQKVKernelGeneration``: 2208 B static + 1 KB reserved
#: per 256-thread block) must fit beside the persistent CTA, or it waits for the whole grid.
SMEM_HEADROOM = 4096


@dsl_user_op
def _max_neg448(x, *, loc=None, ip=None):
    """``max.f32(x, -448)``: maps NaN to -448; the satfinite E4M3 conversion does the rest."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
            "max.f32 $0, $1, 0fC3E00000;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ldg_nc_v4(addr, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return [Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _bf16x4_to_e4m3x4(w0, w1, *, loc=None, ip=None):
    """Four BF16 (two little-endian words) -> four clamped E4M3 bytes in one word."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(w0).ir_value(loc=loc, ip=ip), Int32(w1).ir_value(loc=loc, ip=ip)],
            "{\n"
            ".reg .b32 x0, x1, x2, x3;\n"
            ".reg .b16 h0, h1;\n"
            "shl.b32 x0, $1, 16;\n"
            "and.b32 x1, $1, 0xFFFF0000;\n"
            "shl.b32 x2, $2, 16;\n"
            "and.b32 x3, $2, 0xFFFF0000;\n"
            "max.f32 x0, x0, 0fC3E00000;\n"
            "max.f32 x1, x1, 0fC3E00000;\n"
            "max.f32 x2, x2, 0fC3E00000;\n"
            "max.f32 x3, x3, 0fC3E00000;\n"
            "min.f32 x0, x0, 0f43E00000;\n"
            "min.f32 x1, x1, 0f43E00000;\n"
            "min.f32 x2, x2, 0f43E00000;\n"
            "min.f32 x3, x3, 0f43E00000;\n"
            "cvt.rn.satfinite.e4m3x2.f32 h0, x1, x0;\n"
            "cvt.rn.satfinite.e4m3x2.f32 h1, x3, x2;\n"
            "mov.b32 $0, {h0, h1};\n"
            "}",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _stg_v2(addr, lo, hi, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Int32(lo).ir_value(loc=loc, ip=ip),
            Int32(hi).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class K3MlaQAbsorbKernel(PersistentDenseGemmKernel):
    """Per-head batched ``q_nope @ w_kc[h].T`` with a clamped E4M3 output and the q_pe copy.

    ``tile_n``: UMMA N (lora columns per tile). ``use_2cta``: ``cta_group::2`` with 256 tokens
    per CTA pair. Tiles are walked M (token block) fastest, then N, then head.
    """

    def __init__(
        self,
        num_heads: int,
        nope_dim: int,
        rope_dim: int,
        lora_dim: int,
        tile_n: int,
        use_2cta: bool,
        use_pdl: bool,
        pe_warps: int = 4,
        pe_unroll: int = 8,
        epi_n: int = 64,
        smem_headroom: int = SMEM_HEADROOM,
    ):
        assert pe_warps > 0 and rope_dim % PE_VEC == 0
        super().__init__(
            cutlass.Float32,
            use_2cta,
            (2 * CTA_M if use_2cta else CTA_M, tile_n),
            (2, 1) if use_2cta else (1, 1),
            use_tma_store=True,
            raster_along="m",
        )
        self.num_heads = num_heads
        self.nope_dim = nope_dim
        self.rope_dim = rope_dim
        self.lora_dim = lora_dim
        self.use_pdl = use_pdl
        # Extra warps copy q_pe (grid-stride over 16-byte chunks, independent of the tiles).
        self.pe_warp_id = tuple(range(6, 6 + pe_warps))
        self.pe_unroll = pe_unroll
        self.epi_n = epi_n
        self.smem_headroom = smem_headroom
        self.threads_per_cta = 32 * (len(self.epilogue_warp_id) + 2 + len(self.pe_warp_id))

    def _setup_attributes(self):
        """The parent's setup with an ``epi_n``-column epilogue sub-tile (fewer, larger TMA
        stores than its 32-column default for an 8-bit output) and the stage counts sized to
        leave ``smem_headroom`` bytes of shared memory per SM."""
        super()._setup_attributes()
        self.epi_tile = (cute.make_layout(CTA_M), cute.make_layout(self.epi_n))
        tiled_mma = self._create_tiled_mma()
        c_smem_layout = utils.sm100.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, 1
        )
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = _compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.c_dtype,
            self.smem_capacity - self.smem_headroom,
            self.occupancy,
            True,
            c_smem_layout,
        )
        self.a_smem_layout_staged = utils.sm100.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = utils.sm100.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage
        )
        self.c_smem_layout_staged = utils.sm100.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage
        )

    @cute.jit
    def run(
        self,
        num_tokens: Int32,
        q_tok_stride: Int32,
        out_tok_stride: Int32,
        q_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        head_dim = self.nope_dim + self.rope_dim
        out_head_dim = self.lora_dim + self.rope_dim
        a = cute.make_tensor(
            q_ptr,
            cute.make_layout(
                (num_tokens, self.nope_dim, self.num_heads), stride=(q_tok_stride, 1, head_dim)
            ),
        )
        b = cute.make_tensor(
            w_ptr,
            cute.make_layout(
                (self.lora_dim, self.nope_dim, self.num_heads),
                stride=(self.nope_dim, 1, self.lora_dim * self.nope_dim),
            ),
        )
        c = cute.make_tensor(
            out_ptr,
            cute.make_layout(
                (num_tokens, self.lora_dim, self.num_heads),
                stride=(out_tok_stride, 1, out_head_dim),
            ),
        )
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        tiled_mma = self._create_tiled_mma()
        self._setup_attributes()
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        a_op = utils.sm100.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, a, a_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape
        )
        b_op = utils.sm100.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b, b_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape
        )
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        epi_smem_layout = cute.select(self.c_smem_layout_staged, mode=[0, 1])
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c, epi_smem_layout, self.epi_tile
        )

        self.tile_sched_params, grid = self._compute_grid(
            c,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            self.swizzle_size,
            self.raster_along,
            max_active_clusters,
        )

        self.qabs_kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            q_ptr,
            out_ptr,
            num_tokens,
            q_tok_stride,
            out_tok_stride,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def qabs_kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        q_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        num_tokens: Int32,
        q_tok_stride: Int32,
        out_tok_stride: Int32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_pipeline = PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilogue_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )

        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = Int32(cute.size(gA_mkl, mode=[3]))

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)

        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        #
        # TMA warp: weight stages of the first tile before the PDL wait, then everything else.
        #
        if warp_idx == self.tma_warp_id:
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            n_pre = Int32(0)
            if work_tile.is_valid_tile:
                n_pre = k_tile_cnt
                if n_pre > self.num_ab_stage:
                    n_pre = Int32(self.num_ab_stage)
                coord0 = work_tile.tile_idx
                tBgB0 = tBgB[(None, coord0[1], None, coord0[2])]
                for kt in cutlass.range(n_pre, unroll=1):
                    ab_pipeline.producer_acquire(ab_producer_state)
                    cute.copy(
                        tma_atom_b,
                        tBgB0[(None, kt)],
                        tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=b_full_mcast_mask,
                    )
                    ab_producer_state.advance()
            if cutlass.const_expr(self.use_pdl):
                griddepcontrol_wait()
            k_pre = n_pre
            pre_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
                tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
                # Query halves of the prefetched stages.
                for kt in cutlass.range(k_pre, unroll=1):
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, kt)],
                        tAsA[(None, pre_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(pre_state),
                        mcast_mask=a_full_mcast_mask,
                    )
                    pre_state.advance()
                for kt in cutlass.range(k_pre, k_tile_cnt, 1, unroll=1):
                    ab_pipeline.producer_acquire(ab_producer_state)
                    bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, kt)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=bar,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, kt)],
                        tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=bar,
                        mcast_mask=b_full_mcast_mask,
                    )
                    ab_producer_state.advance()
                k_pre = Int32(0)
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            ab_pipeline.producer_tail(ab_producer_state)

        #
        # MMA warp
        #
        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            while work_tile.is_valid_tile:
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_tile in range(k_tile_cnt):
                    if is_leader_cta:
                        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_crd = (None, None, kblock_idx, ab_consumer_state.index)
                            cute.gemm(tiled_mma, tCtAcc, tCrA[kblock_crd], tCrB[kblock_crd], tCtAcc)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        ab_pipeline.consumer_release(ab_consumer_state)
                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            acc_pipeline.producer_tail(acc_producer_state)

        sC = smem.allocate_tensor(
            element_type=self.c_dtype,
            layout=c_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=c_smem_layout_staged.inner,
        )

        #
        # Epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)
            if cutlass.const_expr(self.use_pdl):
                # q_pe comes from the previous grid; ``out`` may still be read by earlier ones.
                griddepcontrol_wait()
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            copy_atom_t2r = utils.sm100.get_tmem_load_op(
                self.cta_tile_shape_mnk,
                self.c_layout,
                self.c_dtype,
                self.acc_dtype,
                epi_tile,
                use_2cta_instrs,
            )
            tAcc_epi = cute.flat_divide(tCtAcc_base[((None, None), 0, 0, None)], epi_tile)
            tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAcc_base = thr_copy_t2r.partition_S(tAcc_epi)
            gC_mnl_epi = cute.flat_divide(tCgC[((None, None), 0, 0, None, None, None)], epi_tile)
            tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
            tTR_rAcc = cute.make_fragment(
                tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
            )
            copy_atom_r2s = utils.sm100.get_smem_store_op(
                self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
            )
            tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
            thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
            tRS_sC = thr_copy_r2s.partition_D(sC)
            tTR_rC = cute.make_fragment(tTR_rAcc.shape, self.c_dtype)
            tRS_rC = tiled_copy_r2s.retile(tTR_rC)
            sC_for_tma = cute.group_modes(sC, 0, 2)
            gC_for_tma = cute.group_modes(gC_mnl_epi, 0, 2)
            bSG_sC, bSG_gC_partitioned = cpasync.tma_partition(
                tma_atom_c, 0, cute.make_layout(1), sC_for_tma, gC_for_tma
            )
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, 32 * len(self.epilogue_warp_id)
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage, producer_group=c_producer_group
            )
            epilog_threads = 32 * len(self.epilogue_warp_id)

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt

                acc_pipeline.consumer_wait(acc_consumer_state)
                for subtile_idx in cutlass.range(subtile_cnt):
                    cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, subtile_idx)], tTR_rAcc)
                    for i in cutlass.range(cute.size(tTR_rAcc), unroll_full=True):
                        tTR_rAcc[i] = _max_neg448(tTR_rAcc[i])
                    acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                    tRS_rC.store(acc_vec.to(self.c_dtype))
                    c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
                    cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id, number_of_threads=epilog_threads
                    )
                    if warp_idx == self.epilogue_warp_id[0]:
                        cute.copy(tma_atom_c, bSG_sC[(None, c_buffer)], bSG_gC[(None, subtile_idx)])
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()
                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id, number_of_threads=epilog_threads
                    )
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            c_pipeline.producer_tail()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        #
        # q_pe warps: out[t, h, lora:] = e4m3(clamp(q[t, h, nope:])), 16-byte chunks grid-stride
        # (adjacent threads take adjacent chunks of a row).
        #
        if warp_idx >= self.pe_warp_id[0]:
            if cutlass.const_expr(self.use_pdl):
                griddepcontrol_wait()
            head_dim = self.nope_dim + self.rope_dim
            out_head_dim = self.lora_dim + self.rope_dim
            chunks_per_row = self.rope_dim // PE_VEC
            pe_threads = 32 * len(self.pe_warp_id)
            q_base = q_ptr.toint()
            out_base = out_ptr.toint()
            # The persistent grid is (cluster M, cluster N, clusters): linearize it.
            grid_x, grid_y, grid_z = cute.arch.grid_dim()
            blk_x, blk_y, blk_z = cute.arch.block_idx()
            cta_id = Int32(blk_x) + Int32(grid_x) * (Int32(blk_y) + Int32(grid_y) * Int32(blk_z))
            step = Int32(grid_x) * Int32(grid_y) * Int32(grid_z) * pe_threads
            total = num_tokens * (self.num_heads * chunks_per_row)
            first = cta_id * pe_threads + (tidx - 32 * self.pe_warp_id[0])
            for base in cutlass.range(first, total, step * self.pe_unroll, unroll=1):
                words = []
                for u in cutlass.range_constexpr(self.pe_unroll):
                    idx = base + u * step
                    if idx > total - 1:
                        idx = total - 1
                    row = idx // chunks_per_row
                    col = (idx - row * chunks_per_row) * PE_VEC
                    tok = row // self.num_heads
                    head = row - tok * self.num_heads
                    words.append(
                        _ldg_nc_v4(
                            q_base
                            + (Int64(tok) * q_tok_stride + head * head_dim + self.nope_dim + col)
                            * 2
                        )
                    )
                for u in cutlass.range_constexpr(self.pe_unroll):
                    idx = base + u * step
                    if idx < total:
                        row = idx // chunks_per_row
                        col = (idx - row * chunks_per_row) * PE_VEC
                        tok = row // self.num_heads
                        head = row - tok * self.num_heads
                        _stg_v2(
                            out_base
                            + Int64(tok) * out_tok_stride
                            + head * out_head_dim
                            + self.lora_dim
                            + col,
                            _bf16x4_to_e4m3x4(words[u][0], words[u][1]),
                            _bf16x4_to_e4m3x4(words[u][2], words[u][3]),
                        )

        if cutlass.const_expr(self.use_pdl):
            griddepcontrol_launch_dependents()


def _gmem_ptr(dtype, tensor: torch.Tensor):
    return make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)


@functools.cache
def _max_active_clusters(cluster_size: int) -> int:
    return cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_size)


@functools.cache
def _compiled(
    num_heads: int,
    nope_dim: int,
    rope_dim: int,
    lora_dim: int,
    tile_n: int,
    use_2cta: bool,
    use_pdl: bool,
    pe_warps: int = 4,
    pe_unroll: int = 8,
    epi_n: int = 64,
    smem_headroom: int = SMEM_HEADROOM,
):
    kernel = K3MlaQAbsorbKernel(
        num_heads,
        nope_dim,
        rope_dim,
        lora_dim,
        tile_n,
        use_2cta,
        use_pdl,
        pe_warps,
        pe_unroll,
        min(epi_n, tile_n),
        smem_headroom,
    )
    dummy = 1 << 20
    bf = make_ptr(cutlass.BFloat16, dummy, cute.AddressSpace.gmem, assumed_align=16)
    f8 = make_ptr(cutlass.Float8E4M3FN, dummy, cute.AddressSpace.gmem, assumed_align=16)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute.compile(
        kernel.run,
        CTA_M,
        num_heads * (nope_dim + rope_dim),
        num_heads * (lora_dim + rope_dim),
        bf,
        bf,
        f8,
        _max_active_clusters(2 if use_2cta else 1),
        stream,
        options="--opt-level 2",
    )


#: (token-count bound, (UMMA N, 2-CTA)) in order; measured on GB300 with 96 heads.
_TILE_TABLE = ((448, (128, False)), (MAX_TOKENS, (256, False)))


def tile_for(num_tokens: int) -> Tuple[int, bool]:
    """(UMMA N, 2-CTA) by token count."""
    for bound, tile in _TILE_TABLE:
        if num_tokens <= bound:
            return tile
    return _TILE_TABLE[-1][1]


@functools.cache
def _compile_all(
    num_heads: int, nope_dim: int, rope_dim: int, lora_dim: int, use_pdl: bool
) -> None:
    # Every tile the table can pick, at the first (eager) call, so a later CUDA graph capture
    # of another token count never compiles.
    for _, (tile_n, use_2cta) in _TILE_TABLE:
        _compiled(num_heads, nope_dim, rope_dim, lora_dim, tile_n, use_2cta, use_pdl)


def _is_supported_device(device_index: int) -> bool:
    # SM100 / SM103 (Blackwell) and SM107 (Rubin): tcgen05 kind::f16 is shared by all three.
    return torch.cuda.get_device_capability(device_index) in ((10, 0), (10, 3), (10, 7))


def can_run(
    q: torch.Tensor,
    w_kc: torch.Tensor,
    out: torch.Tensor,
    nope_dim: int,
    rope_dim: int,
) -> bool:
    """Whether :func:`k3_mla_q_absorb` supports these operands (else use the Triton kernel)."""
    if not q.is_cuda or not _is_supported_device(q.device.index):
        return False
    if q.dim() != 3 or w_kc.dim() != 3 or out.dim() != 3:
        return False
    num_tokens, num_heads, head_dim = q.shape
    lora_dim = w_kc.shape[1]
    if not (0 < num_tokens <= MAX_TOKENS) or head_dim != nope_dim + rope_dim:
        return False
    if q.dtype != torch.bfloat16 or w_kc.dtype != torch.bfloat16:
        return False
    if out.dtype not in (torch.float8_e4m3fn, torch.uint8):
        return False
    if w_kc.shape != (num_heads, lora_dim, nope_dim) or not w_kc.is_contiguous():
        return False
    if out.shape != (num_tokens, num_heads, lora_dim + rope_dim):
        return False
    if nope_dim % 64 != 0 or rope_dim % PE_VEC != 0 or lora_dim % 256 != 0:
        return False
    if q.stride(2) != 1 or q.stride(1) != head_dim or q.stride(0) % 8 != 0:
        return False
    if out.stride(2) != 1 or out.stride(1) != lora_dim + rope_dim or out.stride(0) % 16 != 0:
        return False
    return all(t.data_ptr() % 16 == 0 for t in (q, w_kc, out))


def k3_mla_q_absorb(
    q: torch.Tensor,
    w_kc: torch.Tensor,
    out: torch.Tensor,
    nope_dim: int,
    rope_dim: int,
    *,
    tile: Optional[Tuple[int, bool]] = None,
    use_pdl: bool = True,
) -> None:
    """Write the FP8 absorbed query into ``out`` (see the module docstring); check ``can_run``."""
    num_tokens, num_heads, _ = q.shape
    lora_dim = w_kc.shape[1]
    _compile_all(num_heads, nope_dim, rope_dim, lora_dim, use_pdl)
    tile_n, use_2cta = tile if tile is not None else tile_for(num_tokens)
    fn = _compiled(num_heads, nope_dim, rope_dim, lora_dim, tile_n, use_2cta, use_pdl)
    fn(
        num_tokens,
        q.stride(0),
        out.stride(0),
        _gmem_ptr(cutlass.BFloat16, q),
        _gmem_ptr(cutlass.BFloat16, w_kc),
        _gmem_ptr(cutlass.Float8E4M3FN, out),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
