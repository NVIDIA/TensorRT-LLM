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

from .. import memory as indexer_memory
from .. import tensor_utils as indexer_tensor_utils
from ..packed import fma_packed_f32x2 as indexer_fma_packed_f32x2
from ..packed import add_packed_f32x2 as indexer_add_packed_f32x2
from ..ordering import PipelineTmaAsync as IndexerPipelineTmaAsync
from ..pdl import pdl_prologue
from .weighted_heads import weighted_heads
from .subset_barrier import subset_barrier
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Contiguous FP8 MQA stages with separate FP32 K scales.

Persistent CTAs share Q across KV tiles. TMA uses true operand extents,
zero-filling ragged tiles; per-row visible bounds mask output stores."""

import argparse
import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Float16, testing
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.arch import get_max_tmem_alloc_cols
from cutlass.cutlass_dsl import BaseDSL
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from ..mqa_logits_util import (
    calc_diff,
    KernelLaunch,
)

from .selection import emit, sample_tile, publish_private, select, select_candidates, propose

_RND_RN = "rn"


def _subview(tensor, offset, count):
    """Load count contiguous elements starting at the element offset."""
    return cute.make_tensor(tensor.iterator + offset, cute.make_layout(count)).load()


class FP8PrefillMQALogitsKernel:
    """Contiguous FP8 MQA stages over grid-strided query/KV tiles.

    Two math warp groups share Q and process adjacent KV blocks."""

    def __init__(
        self,
        block_kv: int = 128,
        num_heads: int = 64,
        head_dim: int = 128,
        # Default UMMA N is 128; callers may override the tile and register budget.
        umma_n: int = 128,
        max_w_in_reg: int = 0,  # 0 = auto (128->40, else min(48,144//block_q))
        num_sms: int = 148,
        remove_kv_wait_in_epilogue: bool = False,
        early_tmem_copy: bool = False,
        smem_subpartition_opt: bool = False,
        max_kv_pipeline: bool = False,
        max_umma_pipeline: bool = False,
        num_epi_subtiles: int = 1,
        epi_dtype=cutlass.Float32,
        weight_dtype=cutlass.Float32,
        acc_dtype=cutlass.Float32,
        output_dtype=cutlass.Float32,
        use_relu_trick: bool = False,
        mode: str = "filter",
        top_k: int = 2048,
        capacity: int = 8192,
        splits: int = 1,
        packed: bool = False,
        sample_size: int = 512,
        math_registers: int = 160,
        store_registers: int = 168,
        weights_prepared: bool = False,
        dsv4_positions: bool = False,
        dsv4_bulk_positions: bool = False,
        dsv4: bool = False,
        pro_common_domain: bool = False,
        fixed_samples: int = 0,
        qsa_weights: int = 0,
        qsa_unit_scale: bool = False,
        persistent_sample: bool = True,
        sample_sigma: int = 4,
        sample_rank_sigma: float = 0.0,
        sample_count_override: int = 0,
        num_kv_stages: int = 0,
        sample_shared: bool = False,
        glm_short_quantile: bool = False,
        pdl: bool = False,
        native_bf16: bool = False,
        native_fp16: bool = False,
    ):
        self.pdl = bool(pdl)
        # Match native K2/K4 head math without changing the sampled positions.
        self.native_lowp = bool(native_bf16 or native_fp16)
        assert not (native_bf16 and native_fp16)
        assert not native_fp16 or output_dtype == cutlass.Float16
        assert not self.native_lowp or (num_heads in (4, 32, 64) and output_dtype == (cutlass.Float16 if native_fp16 else cutlass.BFloat16) and not use_relu_trick)
        if (acc_dtype, epi_dtype) not in (
            (cutlass.Float32, cutlass.Float32),
            (cutlass.Float32, cutlass.Float16),
            (cutlass.Float16, cutlass.Float16),
        ):
            raise ValueError(
                "FP8PrefillMQALogitsKernel supports only (acc_dtype, epi_dtype) "
                "combinations (Float32, Float32), (Float32, Float16), and "
                f"(Float16, Float16); got ({acc_dtype}, {epi_dtype})"
            )
        if output_dtype not in (cutlass.Float32, cutlass.Float16, cutlass.BFloat16):
            raise ValueError(
                "FP8PrefillMQALogitsKernel output_dtype must be Float32, Float16, "
                f"or BFloat16; got {output_dtype}"
            )

        assert umma_n % num_heads == 0, (
            f"num_heads must divide umma_n (UMMA N={umma_n})"
        )
        assert mode=="sample" and (umma_n==256 or num_heads in (4,32)) and epi_dtype==cutlass.Float32 and splits==1
        self.mode, self.top_k, self.capacity = mode, top_k, capacity
        self.splits, self.packed = splits, packed
        self.sample_sigma = sample_sigma
        self.sample_rank_sigma = sample_rank_sigma
        self.sample_count_override = sample_count_override
        self.dsv4_positions = dsv4_positions
        self.dsv4_bulk_positions = dsv4_bulk_positions
        self.glm_bulk_positions = num_heads==32 and top_k==512 and fixed_samples in (128,256)
        self.dsv4 = dsv4
        self.pro_common_domain = pro_common_domain
        self.fixed_samples = fixed_samples
        self.fixed_persistent = (dsv4 or (num_heads==32 and top_k==512)) and fixed_samples in (128,256)
        self.qsa_weights = qsa_weights
        self.qsa_unit_scale = qsa_unit_scale
        self.persistent_sample = persistent_sample
        self.weights_prepared = weights_prepared
        self.sample_size = sample_size
        self.math_registers, self.store_registers = math_registers, store_registers
        self.block_kv = block_kv
        # block_q: number of Q tokens per MMA tile. N = block_q * num_heads.
        # Halving umma_n doubles Q-task count and halves the epilogue
        # register footprint per CTA.
        self.block_q = umma_n // num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.N = self.block_q * num_heads
        self.max_w_in_reg = max_w_in_reg
        self.num_sms = num_sms
        self.remove_kv_wait_in_epilogue = remove_kv_wait_in_epilogue
        self.early_tmem_copy = early_tmem_copy
        self.smem_subpartition_opt = smem_subpartition_opt
        self.num_epi_subtiles = num_epi_subtiles
        self.epi_dtype = epi_dtype
        self.weight_dtype = weight_dtype
        self.weight_bytes = 2 if weight_dtype == cutlass.BFloat16 else 4
        # The optional ReLU identity (x+abs(x))/2 can overflow for x >= 2**127.
        # Enable it only when input scales and weights keep the intermediate finite.
        self.use_relu_trick = use_relu_trick
        self.epi_bytes = 2 if epi_dtype == cutlass.Float16 else 4
        w_stage_bytes = self.N * self.weight_bytes
        self.w_stage_stride = ((w_stage_bytes + 127) // 128 * 128) // self.weight_bytes
        self.output_dtype = output_dtype
        if num_epi_subtiles > 1 and num_heads % num_epi_subtiles != 0:
            raise ValueError("num_heads must be divisible by num_epi_subtiles")
        if (num_heads // num_epi_subtiles) % 4 != 0:
            raise ValueError("num_heads // num_epi_subtiles must be divisible by 4")
        self.num_groups = 1
        self.sample_shared = sample_shared
        self.glm_short_quantile = glm_short_quantile

        self.num_math_threads = 512
        self.num_specialized_threads = 96
        self.threads_per_cta = 608
        self.num_math_warps = 16
        self.tma_warp_base = 8
        self.umma_warp_base = 10

        # Reload Q when the Q block changes; reuse it across KV blocks.
        self.num_q_stages = 2 if self.fixed_persistent else 1
        self.sample_position_stride = fixed_samples if self.fixed_persistent and num_heads==32 and top_k==512 else 1024

        arch = BaseDSL._get_dsl().get_arch_enum()
        TMEM_COLS = get_max_tmem_alloc_cols(f"sm_{arch.major}{arch.minor}")
        if max_umma_pipeline:
            self.num_umma_stages = min(2, TMEM_COLS // self.N)
        else:
            self.num_umma_stages = 1

        if max_kv_pipeline:
            smem_capacity = indexer_memory.get_smem_capacity_in_bytes()
            SMEM_BUDGET = smem_capacity - 1024
            # KV+Scale per stage (×2 groups): 2 * (block_kv*head_dim + block_kv*4)
            kv_scale_per_stage = 2 * (block_kv * head_dim + block_kv * 4)
            qw_per_stage = self.N * head_dim + self.w_stage_stride * self.weight_bytes
            qw_total = qw_per_stage * self.num_q_stages
            self.num_kv_stages = (SMEM_BUDGET - qw_total) // kv_scale_per_stage
        else:
            self.num_kv_stages = num_kv_stages if num_kv_stages > 0 else 4

        if self.smem_subpartition_opt:
            BOUNDARY = 128 * 1024
            used = (
                256
                + 2 * (block_kv * head_dim * self.num_kv_stages)
                + self.N * head_dim * self.num_q_stages
            )
            used = ((used + 127) // 128) * 128
            if used < BOUNDARY:
                self.smem_pad_bytes = ((BOUNDARY - used + 1023) // 1024) * 1024
            else:
                self.smem_pad_bytes = 0
        else:
            self.smem_pad_bytes = 0

        self.acc_dtype = acc_dtype
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (1, 1)
        self.mma_tiler_mn = (block_kv, self.N)

    def _setup_mma(self, a_dtype, b_dtype, a_major, b_major):
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.a_major_mode = a_major
        self.b_major_mode = b_major

        self.mma_tiler = (*self.mma_tiler_mn, 1)
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            a_dtype,
            b_dtype,
            a_major,
            b_major,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_mn,
        )
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = self.head_dim // mma_inst_shape_k
        full_k = mma_inst_shape_k * mma_inst_tile_k
        self.mma_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], full_k)

        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.epi_tile = self.cta_tile_shape_mnk[:2]

        # KV SMEM: num_kv_stages per group, each [block_kv, head_dim]
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            a_dtype,
            self.num_kv_stages,
        )
        # Q SMEM: num_q_stages, each [N, head_dim]
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            b_dtype,
            self.num_q_stages,
        )

        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        self.num_tmem_alloc_cols = indexer_memory.get_num_tmem_alloc_cols(tCtAcc_fake)
        self.num_tmem_alloc_cols_total = (
            self.num_tmem_alloc_cols * self.num_groups * self.num_umma_stages
        )
        return tiled_mma

    @cute.jit
    def __call__(
        self,
        kv: cute.Tensor,  # [num_kv_tokens, head_dim] FP8, true extent (no host padding)
        kv_scales: cute.Tensor,  # [num_kv_tokens] float32, true extent
        b: cute.Tensor,  # Q: [num_q_tokens * num_heads, head_dim] (uint8/FP8), true extent
        weights: cute.Tensor,  # [num_q_tokens * num_heads] epi_dtype, true extent
        logits: cute.Tensor,  # [num_q_tokens, logits_stride]
        cu_seq_len_k_start: cute.Tensor,  # [num_q_tokens] int32
        cu_seq_len_k_end: cute.Tensor,  # [num_q_tokens] int32
        thresholds: cute.Tensor,
        counts: cute.Tensor,
        candidates: cute.Tensor,
        output: cute.Tensor,
        recovery_queue: cute.Tensor,
        num_q_tokens: cutlass.Int32,
        num_kv_tokens: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        # TMA descriptors use true operand extents. Hardware zero-fills ragged
        # tiles while completing the full transaction byte count.
        b = cute.recast_tensor(b, cutlass.Float8E4M3FN)
        kv_fp8 = cute.recast_tensor(kv, cutlass.Float8E4M3FN)

        a_dtype = kv_fp8.element_type
        b_dtype = b.element_type
        a_major = indexer_tensor_utils.LayoutEnum.from_tensor(kv_fp8).mma_major_mode()
        b_major = indexer_tensor_utils.LayoutEnum.ROW_MAJOR.mma_major_mode()
        tiled_mma = self._setup_mma(a_dtype, b_dtype, a_major, b_major)
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()

        # ---- TMA for KV (A): [num_kv_tokens, head_dim], box [block_kv, head_dim] ----
        a = kv_fp8

        self.a_tma_smem_layout = sm100_utils.make_smem_layout(
            cute.nvgpu.OperandMajorMode.K,
            (self.block_kv, self.head_dim),
            a_dtype,
            self.num_kv_stages,
        )
        a_smem_per_stage = cute.slice_(self.a_tma_smem_layout, (None, None, 0))
        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            tma_load_op,
            a,
            a_smem_per_stage,
            (self.block_kv, self.head_dim),
        )

        # ---- TMA for Q (B): [num_q_tokens * num_heads, head_dim], box [N, head_dim] ----
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # ---- TMA for Weights: [num_q_tokens * num_heads], box [N] ----
        self.w_smem_layout_staged = cute.make_layout(
            (self.N, self.num_q_stages),
            stride=(1, self.w_stage_stride),
        )
        w_smem_per_stage = cute.select(self.w_smem_layout_staged, mode=[0])
        tma_atom_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
            tma_load_op,
            weights,
            w_smem_per_stage,
            self.w_smem_layout_staged.shape[:1],
        )

        # ---- TMA for KV Scales: [num_kv_tokens] float32, box [block_kv] ----
        scales = kv_scales
        self.s_smem_layout_staged = cute.make_layout(
            (self.block_kv, self.num_kv_stages)
        )
        s_smem_per_block = cute.make_layout((self.block_kv,))
        tma_atom_s, tma_tensor_s = cpasync.make_tiled_tma_atom(
            tma_load_op,
            scales,
            s_smem_per_block,
            (self.block_kv,),
        )

        b_copy_size = cute.size_in_bytes(b_dtype, b_smem_layout)
        w_copy_size = self.N * self.weight_bytes
        # Per KV block: block_kv * head_dim (KV) + block_kv * 4 (scales)
        self.num_kv_scale_tma_bytes = self.block_kv * self.head_dim + self.block_kv * 4
        self.num_q_tma_bytes = b_copy_size * atom_thr_size + w_copy_size

        num_q_blocks = (num_q_tokens + self.block_q - 1) // self.block_q
        num_ctas = num_q_blocks * self.splits
        if cutlass.const_expr(self.mode == "filter" or (self.mode == "sample" and self.persistent_sample)):
            num_ctas = num_q_blocks * self.splits

        if cutlass.const_expr(self.fixed_persistent):num_ctas=min(num_q_blocks,self.num_sms)*self.splits

        @cute.struct
        class SharedStorage:
            kv_mbar_0: cute.struct.MemRange[cutlass.Int64, self.num_kv_stages * 2]
            kv_mbar_1: cute.struct.MemRange[cutlass.Int64, self.num_kv_stages * 2]
            q_mbar: cute.struct.MemRange[cutlass.Int64, self.num_q_stages * 2]
            umma_mbar_0: cute.struct.MemRange[cutlass.Int64, self.num_umma_stages * 2]
            umma_mbar_1: cute.struct.MemRange[cutlass.Int64, self.num_umma_stages * 2]
            tmem_holding_buf: cutlass.Int32

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_w,
            tma_tensor_w,
            weights,
            tma_atom_s,
            tma_tensor_s,
            logits,
            cu_seq_len_k_start,
            cu_seq_len_k_end,
            thresholds, counts, candidates, output, recovery_queue,
            num_q_tokens,
            num_q_blocks,
            num_kv_tokens,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.a_tma_smem_layout,
            self.b_smem_layout_staged,
            self.w_smem_layout_staged,
            self.s_smem_layout_staged,
            self.epi_tile,
            kv_fp8, kv_scales,
            SharedStorage,
        ).launch(
            grid=(1, 1, num_ctas),
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=2,
            preferred_smem_carveout=100,
        use_pdl=self.pdl)

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,  # KV TMA tensor: [num_kv_tokens, head_dim]
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,  # Q TMA tensor: [num_q_tokens * num_heads, head_dim]
        tma_atom_w: cute.CopyAtom,
        mW_tma: cute.Tensor,  # Weights TMA tensor: [num_q_tokens * num_heads]
        mW_raw: cute.Tensor,
        tma_atom_s: cute.CopyAtom,
        mS_tma: cute.Tensor,  # Scales TMA tensor: [num_kv_tokens]
        mLogits: cute.Tensor,  # [num_q_tokens, logits_stride]
        mCuSeqLenKStart: cute.Tensor,  # [num_q_tokens] int32
        mCuSeqLenKEnd: cute.Tensor,  # [num_q_tokens] int32
        thresholds: cute.Tensor,
        counts: cute.Tensor,
        candidates: cute.Tensor,
        output: cute.Tensor,
        recovery_queue: cute.Tensor,
        num_q_tokens: cutlass.Int32,
        num_q_blocks: cutlass.Int32,
        num_kv_tokens: cutlass.Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        a_tma_smem_layout: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        w_smem_layout_staged: cute.Layout,
        s_smem_layout_staged: cute.Layout,
        epi_tile: cute.Tile,
        gather_kv: cute.Tensor, gather_scales: cute.Tensor,
        SharedStorage: cutlass.Constexpr,
    ):
        pdl_prologue(self.pdl)
        if cute.arch.thread_idx()[0]==0 and cute.arch.block_idx()[2]==0:
            recovery_queue[0]=0
            recovery_queue[recovery_queue.shape[0]-1]=0
        self._gemm_body(tiled_mma, tma_atom_a, mA_mkl, tma_atom_b, mB_nkl, tma_atom_w, mW_tma, mW_raw, tma_atom_s, mS_tma, mLogits, mCuSeqLenKStart, mCuSeqLenKEnd, thresholds, counts, candidates, output, num_q_tokens, num_q_blocks, num_kv_tokens, cluster_layout_vmnk, a_smem_layout_staged, a_tma_smem_layout, b_smem_layout_staged, w_smem_layout_staged, s_smem_layout_staged, epi_tile, gather_kv, gather_scales, SharedStorage)

    @cute.jit
    def _gemm_body(self, tiled_mma: cute.TiledMma, tma_atom_a: cute.CopyAtom, mA_mkl: cute.Tensor, tma_atom_b: cute.CopyAtom, mB_nkl: cute.Tensor, tma_atom_w: cute.CopyAtom, mW_tma: cute.Tensor, mW_raw: cute.Tensor, tma_atom_s: cute.CopyAtom, mS_tma: cute.Tensor, mLogits: cute.Tensor, mCuSeqLenKStart: cute.Tensor, mCuSeqLenKEnd: cute.Tensor, thresholds: cute.Tensor, counts: cute.Tensor, candidates: cute.Tensor, output: cute.Tensor, num_q_tokens: cutlass.Int32, num_q_blocks: cutlass.Int32, num_kv_tokens: cutlass.Int32, cluster_layout_vmnk: cute.Layout, a_smem_layout_staged: cute.ComposedLayout, a_tma_smem_layout: cute.ComposedLayout, b_smem_layout_staged: cute.ComposedLayout, w_smem_layout_staged: cute.Layout, s_smem_layout_staged: cute.Layout, epi_tile: cute.Tile, gather_kv: cute.Tensor, gather_scales: cute.Tensor, SharedStorage: cutlass.Constexpr):
        smem = indexer_memory.SmemAllocator()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()

        warpgroup_idx = warp_idx // 4
        is_math_warp = warp_idx < 16
        is_tma_warp_0 = warp_idx == 16
        is_tma_warp_1 = warp_idx == 17
        is_tma_warp = is_tma_warp_0 | is_tma_warp_1
        is_umma_warp_0 = warp_idx == 18
        is_umma_warp_1 = warp_idx == 19

        NUM_MATH_WG = 1
        block_kv_val = self.block_kv
        block_q_val = self.block_q
        num_heads = self.num_heads
        num_epi_subtiles = self.num_epi_subtiles

        # Grid-stride scheduler: this SM handles q_block_idx = sm_idx, sm_idx+num_sms, ...
        sm_idx = bidz // self.splits

        # Start at the minimum visible K position, aligned down to SPLIT_KV to
        # preserve the two warp groups' tile ownership. Bind captured values as
        # default arguments because staged control flow cannot capture closures.
        def _kv_lo_for_block(
            qb,
            _ks=mCuSeqLenKStart,
            _nqt=num_q_tokens,
            _bq=block_q_val,
            _split=block_kv_val * NUM_MATH_WG,
            _nwg=NUM_MATH_WG,
            _part=bidz % self.splits,
            _sample=self.mode == "sample",
        ):
            if cutlass.const_expr(_sample):return cutlass.Int32(0)
            ks_min = _ks[min(qb * _bq, _nqt - 1)]
            for t_i in cutlass.range_constexpr(1, _bq):
                ks_min = min(ks_min, _ks[min(qb * _bq + t_i, _nqt - 1)])
            return (ks_min // _split) * _nwg + _part * _nwg

        # Early load of num_kv for the first assigned Q block (hide L2 latency)
        start_q_block = sm_idx
        start_q_block_clamped = min(start_q_block, num_q_blocks - 1)
        q_last_token_clamped = min(
            start_q_block_clamped * block_q_val + block_q_val - 1,
            num_q_tokens - 1,
        )
        current_num_kv = (
            mCuSeqLenKEnd[q_last_token_clamped] + block_kv_val - 1
        ) // block_kv_val
        current_kv_lo = _kv_lo_for_block(start_q_block_clamped)
        if cutlass.const_expr(self.mode == "sample"):
            current_num_kv = (cutlass.Int32(self.fixed_samples) if cutlass.const_expr(self.fixed_samples>0) else runtime_samples((num_kv_tokens if cutlass.const_expr(self.pro_common_domain) else mCuSeqLenKEnd[q_last_token_clamped]),self.top_k,heads=self.num_heads,coverage=4 if self.num_heads==32 else 8,dsv4=self.dsv4))//128
            current_kv_lo = cutlass.Int32(0)

        sample_lower = cutlass.Int32(0)
        sample_upper = cutlass.Int32(0)
        if cutlass.const_expr(self.num_heads in (4,32) or self.dsv4_positions):
            sample_first = sm_idx*self.block_q
            sample_lower = mCuSeqLenKStart[min(sample_first,num_q_tokens-1)]
            for token in cutlass.range_constexpr(1,self.block_q):
                sample_lower = min(sample_lower,mCuSeqLenKStart[min(sample_first+token,num_q_tokens-1)])
            sample_upper = mCuSeqLenKEnd[min(sample_first+self.block_q-1,num_q_tokens-1)]
            if cutlass.const_expr(self.pro_common_domain):
                sample_lower=cutlass.Int32(0)
                sample_upper=num_kv_tokens

        if is_tma_warp:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_w)
            cpasync.prefetch_descriptor(tma_atom_s)

        storage = smem.allocate(SharedStorage)

        # === Pipelines ===
        prod_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)

        q_cons_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 16)
        q_pipeline = IndexerPipelineTmaAsync.create(
            barrier_storage=storage.q_mbar.data_ptr(),
            num_stages=self.num_q_stages,
            producer_group=prod_group,
            consumer_group=q_cons_group,
            tx_count=self.num_q_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            defer_sync=True,
        )
        q_prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_q_stages
        )
        q_cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_q_stages
        )
        q_cons_state_umma_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_q_stages
        )
        q_cons_state_umma_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_q_stages
        )

        math_warps_per_group = self.num_math_warps
        kv_cons_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, math_warps_per_group
        )
        kv_pipeline_0=pipeline.PipelineAsync.create(
            barrier_storage=storage.kv_mbar_0.data_ptr(),
            num_stages=self.num_kv_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread,64),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread,512),
            defer_sync=True)
        kv_pipeline_1=pipeline.PipelineAsync.create(
            barrier_storage=storage.kv_mbar_1.data_ptr(),
            num_stages=self.num_kv_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread,64),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread,512),
            defer_sync=True)
        kv_prod_state_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_kv_stages
        )
        kv_prod_state_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_kv_stages
        )
        kv_cons_state_umma_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )
        kv_cons_state_umma_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )
        kv_cons_state_math_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )
        kv_cons_state_math_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )

        math_threads_per_group = self.num_math_threads
        umma_pipeline_0 = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.umma_mbar_0.data_ptr(),
            num_stages=self.num_umma_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, math_threads_per_group
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        umma_pipeline_1 = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.umma_mbar_1.data_ptr(),
            num_stages=self.num_umma_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, math_threads_per_group
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        umma_prod_state_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_umma_stages
        )
        umma_prod_state_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_umma_stages
        )
        umma_cons_state_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_umma_stages
        )
        umma_cons_state_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_umma_stages
        )

        tmem_alloc_num_threads = 544
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=tmem_alloc_num_threads
        )
        tmem = indexer_memory.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=0,
            is_two_cta=False,
        )


        # Persistent K1 keeps per-query scratch in the math groups. TMA/UMMA
        # continue independently; barrier 12 includes only the 256 math lanes.
        if cutlass.const_expr(self.mode == 'sample' and self.persistent_sample):
            sample_hist=smem.allocate_tensor(cutlass.Int32,cute.make_layout(self.block_q*256 if self.num_heads==32 else 4096 if self.num_heads==4 else 1056),byte_alignment=128)
            sample_scalar=smem.allocate_tensor(cutlass.Int32,cute.make_layout(8),byte_alignment=16)
        # H4 samples are capped at1024. Each producer writes a unique slot;
        # math reads it after the KV/UMMA acquire, and the whole Q tile owns
        # the array until all math consumers release Q. No ring reuse race.
        if cutlass.const_expr(self.num_heads in (4,32) or self.dsv4_positions):
            sample_positions=smem.allocate_tensor(cutlass.Int32,cute.make_layout(self.sample_position_stride*self.num_q_stages),byte_alignment=128)
        if cutlass.const_expr(self.sample_shared):
            sample_scores=smem.allocate_tensor(self.output_dtype,cute.make_layout((self.block_q,self.fixed_samples if self.fixed_samples>0 else 256 if self.top_k==2048 and self.num_heads==32 else 128),stride=(self.fixed_samples if self.fixed_samples>0 else 256 if self.top_k==2048 and self.num_heads==32 else 128,1)),byte_alignment=128)
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # SMEM allocation: per-group KV + shared Q + Weights + Scales
        sKV_0 = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sKV_1 = sKV_0
        sQ = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        if cutlass.const_expr(self.smem_pad_bytes > 0):
            _ = smem.allocate(self.smem_pad_bytes)
        sW = smem.allocate_tensor(
            element_type=self.weight_dtype,
            layout=w_smem_layout_staged,
            byte_alignment=128,
        )
        sScales_0 = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=s_smem_layout_staged,
            byte_alignment=128,
        )
        sScales_1 = sScales_0

        a_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
        )
        b_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
        )

        # Partition KV (A): [num_kv_tokens, head_dim] tiled by [block_kv, head_dim]
        # -> (block_kv, head_dim, kv_block_idx, 1); the last KV block may be partial
        # (TMA zero-fills the rows beyond num_kv_tokens).
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        gA = cute.local_tile(
            mA_mkl,
            (self.block_kv, self.head_dim),
            coord=(None, None),
        )
        # sKV SMEM view for TMA: group (block_kv, head_dim) into one mode to match
        # GMEM's ((block_kv,head_dim), ...) shape from group_modes(gA, 0, 2).
        sKV_0_tma = cute.make_tensor(
            sKV_0.iterator, cute.group_modes(a_tma_smem_layout.outer, 0, 2)
        )
        sKV_1_tma = cute.make_tensor(
            sKV_1.iterator, cute.group_modes(a_tma_smem_layout.outer, 0, 2)
        )
        tAsA_0, tAgA_0 = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            sKV_0_tma,
            cute.group_modes(gA, 0, 2),
        )
        tAsA_1, tAgA_1 = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            sKV_1_tma,
            cute.group_modes(gA, 0, 2),
        )

        # Partition Q (B): [num_q_tokens * num_heads, head_dim] tiled by the (N, K)
        # MMA tile -> (N, K, q_block_idx, 1); the last Q block may be partial (TMA
        # zero-fills the rows beyond num_q_tokens * num_heads).
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.mma_tiler, (0, None, None)),
            (None, None),
        )
        tCgB = thr_mma.partition_B(gB_nkl)
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tBgB = tBgB[(None, None, 0)]  # [tma, q_block_idx]

        # Partition Weights: [num_q_tokens * num_heads] tiled by (N,) -> (N, q_block_idx)
        gW = cute.local_tile(mW_tma, (self.N,), coord=(None,))
        w_cta_layout = cute.make_layout((1,))
        tWsW, tWgW = cpasync.tma_partition(
            tma_atom_w,
            0,
            w_cta_layout,
            cute.group_modes(sW, 0, 1),
            cute.group_modes(gW, 0, 1),
        )

        # Partition Scales: [num_kv_tokens] tiled by (block_kv,) -> (block_kv, kv_block_idx)
        gS = cute.local_tile(mS_tma, (self.block_kv,), coord=(None,))
        s_tma_smem_layout = cute.make_layout(
            (self.block_kv, self.num_kv_stages),
            stride=(1, self.block_kv),
        )
        sScales_0_tma = cute.make_tensor(sScales_0.iterator, s_tma_smem_layout)
        sScales_1_tma = cute.make_tensor(sScales_1.iterator, s_tma_smem_layout)
        tSsS_0, tSgS_0 = cpasync.tma_partition(
            tma_atom_s,
            0,
            cute.make_layout(1),
            sScales_0_tma,
            gS,
        )
        tSsS_1, tSgS_1 = cpasync.tma_partition(
            tma_atom_s,
            0,
            cute.make_layout(1),
            sScales_1_tma,
            gS,
        )

        # MMA fragments
        tCrA_0 = tiled_mma.make_fragment_A(sKV_0)
        tCrA_1 = tiled_mma.make_fragment_A(sKV_1)
        tCrB = tiled_mma.make_fragment_B(sQ)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])

        us = self.num_umma_stages
        cols = self.num_tmem_alloc_cols
        acc_shape_staged = cute.append(acc_shape, us)
        tCtAcc_fake_staged = tiled_mma.make_fragment_C(acc_shape_staged)

        cols_per_group = cols * us * (32 // self.acc_dtype.width)
        num_tmem_alloc_cols_total = self.num_tmem_alloc_cols_total

        c_layout = indexer_tensor_utils.LayoutEnum.ROW_MAJOR
        epi_sub_mn = (epi_tile[0], num_heads // num_epi_subtiles)
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            c_layout,
            self.acc_dtype,
            self.acc_dtype,
            epi_sub_mn,
            use_2cta_instrs,
        )

        # Grid-stride Q scheduling advances by num_sms. Math warps nest
        # KV-pair loops inside Q blocks; TMA/UMMA use a flat loop.
        q_block_idx = cutlass.Int32(sm_idx)
        kv_idx = cutlass.Int32(current_kv_lo)
        num_kv = current_num_kv
        # Sentinel: no previous Q block (forces initial Q load)
        prev_q_block_idx = num_q_blocks  # out-of-range sentinel

        next_q_block_idx = q_block_idx
        next_kv_idx = kv_idx
        next_num_kv = num_kv
        has_work = q_block_idx < num_q_blocks

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # ===== WARP-SPECIALIZED EXECUTION =====

        if is_tma_warp:
            # Two gather warps; warp 16 also issues Q TMA.
            # TMA warp 0: loads Q+Weights (on Q-block change) + KV group 0
            cute.arch.warpgroup_reg_dealloc(self.store_registers)

            if cutlass.const_expr(self.fixed_persistent):
                q_iteration=cutlass.Int32(0)
                while has_work:
                    sample_q_stage=q_iteration%self.num_q_stages
                    sample_first=next_q_block_idx*self.block_q
                    sample_lower=mCuSeqLenKStart[min(sample_first,num_q_tokens-1)]
                    for token in cutlass.range_constexpr(1,self.block_q):
                        sample_lower=min(sample_lower,mCuSeqLenKStart[min(sample_first+token,num_q_tokens-1)])
                    sample_upper=mCuSeqLenKEnd[min(sample_first+self.block_q-1,num_q_tokens-1)]
                    if cutlass.const_expr(self.pro_common_domain):
                        sample_lower=cutlass.Int32(0)
                        sample_upper=num_kv_tokens
                    # Prefetch first Q block before the loop to hide TMA latency.
                    # Guard with has_work: idle CTAs (bidz >= num_q_blocks) must not
                    # issue a TMA load with an out-of-range Q block index.
                    if has_work and is_tma_warp_0:
                        q_pipeline.producer_acquire(q_prod_state)
                        q_bar = q_pipeline.producer_get_barrier(q_prod_state)
                        cute.copy(
                            tma_atom_b,
                            tBgB[(None, next_q_block_idx)],
                            tBsB[(None, q_prod_state.index)],
                            tma_bar_ptr=q_bar,
                            mcast_mask=b_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_w,
                            tWgW[(None, next_q_block_idx)],
                            tWsW[(None, q_prod_state.index)],
                            tma_bar_ptr=q_bar,
                        )
                        q_prod_state.advance()
                    subset_barrier(barrier_id=2,number_of_threads=64)

                    # Each fixed-sample Q tile has a continuous circular-buffer ordinal.
                    # Keep at most2 outstanding cp.async groups; last groups drain.
                    if has_work:
                        q_block_idx=next_q_block_idx
                        num_kv=next_num_kv
                        for kv_idx in cutlass.range(0,min(1,num_kv)):
                            kv_prod_state_0=pipeline.PipelineState(self.num_kv_stages,(q_iteration*(self.fixed_samples//128)+kv_idx),(q_iteration*(self.fixed_samples//128)+kv_idx)%self.num_kv_stages,1^(((q_iteration*(self.fixed_samples//128)+kv_idx)//self.num_kv_stages)%2))
                            kv_pipeline_0.producer_acquire(kv_prod_state_0)
                            stage=kv_prod_state_0.index
                            lane=cute.arch.lane_idx()
                            data_atom=cute.make_copy_atom(cpasync.CopyG2SOp(),cutlass.Float8E4M3FN,num_bits_per_copy=128)
                            scale_atom=cute.make_copy_atom(cpasync.CopyG2SOp(),cutlass.Float32,num_bits_per_copy=32)
                            if cutlass.const_expr(self.dsv4_bulk_positions or self.glm_bulk_positions):
                                for position_half in cutlass.range_constexpr(2):
                                    owned_row=(warp_idx-16)*64+position_half*32+lane
                                    if cutlass.const_expr(self.glm_bulk_positions):sample_positions[sample_q_stage*self.sample_position_stride+kv_idx*128+owned_row]=shape_sample_key(self.sample_count_override,kv_idx*128+owned_row,q_block_idx,sample_lower,sample_upper,self.top_k,heads=self.num_heads,coverage=4,common=False)
                                    else:sample_positions[sample_q_stage*self.sample_position_stride+kv_idx*128+owned_row]=dsv4_cached_sample_key(kv_idx*128+owned_row,0 if cutlass.const_expr(self.pro_common_domain) else q_block_idx,sample_lower,sample_upper,self.top_k,coverage=4,sample_count=(256 if self.dsv4 and self.top_k==512 and self.fixed_samples==256 else 0))
                                cute.arch.sync_warp()
                            for row_group in cutlass.range(16):
                                local_row=row_group*4+lane//8+(warp_idx-16)*64
                                if cutlass.const_expr(self.num_heads in (4,32)):
                                    if cutlass.const_expr(self.glm_bulk_positions):physical=sample_positions[sample_q_stage*self.sample_position_stride+kv_idx*128+local_row]
                                    else:physical = shape_sample_key(self.sample_count_override,kv_idx*128+local_row,q_block_idx,sample_lower,sample_upper,self.top_k,heads=self.num_heads,coverage=4 if self.num_heads==32 else 8,common=(self.num_heads==32 and self.top_k==2048 and self.output_dtype==cutlass.Float16))
                                    if (not self.dsv4_bulk_positions) and (not self.glm_bulk_positions) and lane%8==0:sample_positions[sample_q_stage*self.sample_position_stride+kv_idx*128+local_row]=physical
                                else:
                                    if cutlass.const_expr(self.dsv4_positions):
                                        if cutlass.const_expr(self.dsv4_bulk_positions or self.glm_bulk_positions):physical=sample_positions[sample_q_stage*self.sample_position_stride+kv_idx*128+local_row]
                                        else:
                                            physical=dsv4_cached_sample_key(kv_idx*128+local_row,0 if cutlass.const_expr(self.pro_common_domain) else q_block_idx,sample_lower,sample_upper,self.top_k,coverage=4,sample_count=(256 if self.dsv4 and self.top_k==512 and self.fixed_samples==256 else 0))
                                        if (not self.dsv4_bulk_positions) and (not self.glm_bulk_positions) and lane%8==0:sample_positions[sample_q_stage*self.sample_position_stride+kv_idx*128+local_row]=physical
                                    else:
                                        physical=individual_sample_key(kv_idx*128+local_row,q_block_idx,mCuSeqLenKStart,mCuSeqLenKEnd,num_q_tokens,self.block_q,self.top_k,dsv4=self.dsv4)
                                feature=(lane%8)*16
                                src=cute.local_tile(gather_kv[physical,None],(16,),(lane%8,))
                                row_view=cute.slice_(sKV_0,((local_row,None),0,feature//32,stage))
                                dst=cute.local_tile(row_view,(16,),((lane%8)%2,))
                                cute.copy(data_atom,src,dst)
                                if lane%8==0:
                                    src_scale=cute.local_tile(gather_scales,(1,),(physical,))
                                    dst_scale=cute.local_tile(sScales_0[None,stage],(1,),(local_row,))
                                    cute.copy(scale_atom,src_scale,dst_scale)
                            cute.arch.cp_async_commit_group()
                        pending=cutlass.Int32(0)
                        while pending<num_kv:
                            kv_idx=pending+1
                            if kv_idx<num_kv:
                                kv_prod_state_0=pipeline.PipelineState(self.num_kv_stages,(q_iteration*(self.fixed_samples//128)+kv_idx),(q_iteration*(self.fixed_samples//128)+kv_idx)%self.num_kv_stages,1^(((q_iteration*(self.fixed_samples//128)+kv_idx)//self.num_kv_stages)%2))
                                kv_pipeline_0.producer_acquire(kv_prod_state_0)
                                stage=kv_prod_state_0.index
                                lane=cute.arch.lane_idx()
                                data_atom=cute.make_copy_atom(cpasync.CopyG2SOp(),cutlass.Float8E4M3FN,num_bits_per_copy=128)
                                scale_atom=cute.make_copy_atom(cpasync.CopyG2SOp(),cutlass.Float32,num_bits_per_copy=32)
                                if cutlass.const_expr(self.dsv4_bulk_positions or self.glm_bulk_positions):
                                    for position_half in cutlass.range_constexpr(2):
                                        owned_row=(warp_idx-16)*64+position_half*32+lane
                                        if cutlass.const_expr(self.glm_bulk_positions):sample_positions[sample_q_stage*self.sample_position_stride+kv_idx*128+owned_row]=shape_sample_key(self.sample_count_override,kv_idx*128+owned_row,q_block_idx,sample_lower,sample_upper,self.top_k,heads=self.num_heads,coverage=4,common=False)
                                        else:sample_positions[sample_q_stage*self.sample_position_stride+kv_idx*128+owned_row]=dsv4_cached_sample_key(kv_idx*128+owned_row,0 if cutlass.const_expr(self.pro_common_domain) else q_block_idx,sample_lower,sample_upper,self.top_k,coverage=4,sample_count=(256 if self.dsv4 and self.top_k==512 and self.fixed_samples==256 else 0))
                                    cute.arch.sync_warp()
                                for row_group in cutlass.range(16):
                                    local_row=row_group*4+lane//8+(warp_idx-16)*64
                                    if cutlass.const_expr(self.num_heads in (4,32)):
                                        if cutlass.const_expr(self.glm_bulk_positions):physical=sample_positions[sample_q_stage*self.sample_position_stride+kv_idx*128+local_row]
                                        else:physical = shape_sample_key(self.sample_count_override,kv_idx*128+local_row,q_block_idx,sample_lower,sample_upper,self.top_k,heads=self.num_heads,coverage=4 if self.num_heads==32 else 8,common=(self.num_heads==32 and self.top_k==2048 and self.output_dtype==cutlass.Float16))
                                        if (not self.dsv4_bulk_positions) and (not self.glm_bulk_positions) and lane%8==0:sample_positions[sample_q_stage*self.sample_position_stride+kv_idx*128+local_row]=physical
                                    else:
                                        if cutlass.const_expr(self.dsv4_positions):
                                            if cutlass.const_expr(self.dsv4_bulk_positions or self.glm_bulk_positions):physical=sample_positions[sample_q_stage*self.sample_position_stride+kv_idx*128+local_row]
                                            else:
                                                physical=dsv4_cached_sample_key(kv_idx*128+local_row,0 if cutlass.const_expr(self.pro_common_domain) else q_block_idx,sample_lower,sample_upper,self.top_k,coverage=4,sample_count=(256 if self.dsv4 and self.top_k==512 and self.fixed_samples==256 else 0))
                                            if (not self.dsv4_bulk_positions) and (not self.glm_bulk_positions) and lane%8==0:sample_positions[sample_q_stage*self.sample_position_stride+kv_idx*128+local_row]=physical
                                        else:
                                            physical=individual_sample_key(kv_idx*128+local_row,q_block_idx,mCuSeqLenKStart,mCuSeqLenKEnd,num_q_tokens,self.block_q,self.top_k,dsv4=self.dsv4)
                                    feature=(lane%8)*16
                                    src=cute.local_tile(gather_kv[physical,None],(16,),(lane%8,))
                                    row_view=cute.slice_(sKV_0,((local_row,None),0,feature//32,stage))
                                    dst=cute.local_tile(row_view,(16,),((lane%8)%2,))
                                    cute.copy(data_atom,src,dst)
                                    if lane%8==0:
                                        src_scale=cute.local_tile(gather_scales,(1,),(physical,))
                                        dst_scale=cute.local_tile(sScales_0[None,stage],(1,),(local_row,))
                                        cute.copy(scale_atom,src_scale,dst_scale)
                                cute.arch.cp_async_commit_group()
                                cute.arch.cp_async_wait_group(1)
                            else:
                                cute.arch.cp_async_wait_group(0)
                            cute.arch.sync_warp()
                            cute.arch.fence_view_async_shared()
                            commit_state=pipeline.PipelineState(self.num_kv_stages,(q_iteration*(self.fixed_samples//128)+pending),(q_iteration*(self.fixed_samples//128)+pending)%self.num_kv_stages,1^(((q_iteration*(self.fixed_samples//128)+pending)//self.num_kv_stages)%2))
                            kv_pipeline_0.producer_commit(commit_state)
                            pending+=1

                    next_q_block_idx=next_q_block_idx+cute.arch.grid_dim()[2]
                    q_iteration=q_iteration+1
                    has_work=next_q_block_idx<num_q_blocks
            else:
                # Prefetch first Q block before the loop to hide TMA latency.
                # Guard with has_work: idle CTAs (bidz >= num_q_blocks) must not
                # issue a TMA load with an out-of-range Q block index.
                if has_work and is_tma_warp_0:
                    q_pipeline.producer_acquire(q_prod_state)
                    q_bar = q_pipeline.producer_get_barrier(q_prod_state)
                    cute.copy(
                        tma_atom_b,
                        tBgB[(None, next_q_block_idx)],
                        tBsB[(None, q_prod_state.index)],
                        tma_bar_ptr=q_bar,
                        mcast_mask=b_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_w,
                        tWgW[(None, next_q_block_idx)],
                        tWsW[(None, q_prod_state.index)],
                        tma_bar_ptr=q_bar,
                    )
                    q_prod_state.advance()

                # Each fixed-sample Q tile has a continuous circular-buffer ordinal.
                # Keep at most2 outstanding cp.async groups; last groups drain.
                if has_work:
                    q_block_idx=next_q_block_idx
                    num_kv=next_num_kv
                    for kv_idx in cutlass.range(0,min(1,num_kv)):
                        kv_prod_state_0=pipeline.PipelineState(self.num_kv_stages,kv_idx,kv_idx%self.num_kv_stages,1^((kv_idx//self.num_kv_stages)%2))
                        kv_pipeline_0.producer_acquire(kv_prod_state_0)
                        stage=kv_prod_state_0.index
                        lane=cute.arch.lane_idx()
                        data_atom=cute.make_copy_atom(cpasync.CopyG2SOp(),cutlass.Float8E4M3FN,num_bits_per_copy=128)
                        scale_atom=cute.make_copy_atom(cpasync.CopyG2SOp(),cutlass.Float32,num_bits_per_copy=32)
                        if cutlass.const_expr(self.dsv4_bulk_positions or self.glm_bulk_positions):
                            for position_half in cutlass.range_constexpr(2):
                                owned_row=(warp_idx-16)*64+position_half*32+lane
                                if cutlass.const_expr(self.glm_bulk_positions):sample_positions[kv_idx*128+owned_row]=shape_sample_key(self.sample_count_override,kv_idx*128+owned_row,q_block_idx,sample_lower,sample_upper,self.top_k,heads=self.num_heads,coverage=4,common=False)
                                else:sample_positions[kv_idx*128+owned_row]=dsv4_cached_sample_key(kv_idx*128+owned_row,0 if cutlass.const_expr(self.pro_common_domain) else q_block_idx,sample_lower,sample_upper,self.top_k,coverage=4,sample_count=(256 if self.dsv4 and self.top_k==512 and self.fixed_samples==256 else 0))
                            cute.arch.sync_warp()
                        for row_group in cutlass.range(16):
                            local_row=row_group*4+lane//8+(warp_idx-16)*64
                            if cutlass.const_expr(self.num_heads in (4,32)):
                                if cutlass.const_expr(self.glm_bulk_positions):physical=sample_positions[kv_idx*128+local_row]
                                else:physical = shape_sample_key(self.sample_count_override,kv_idx*128+local_row,q_block_idx,sample_lower,sample_upper,self.top_k,heads=self.num_heads,coverage=4 if self.num_heads==32 else 8,common=(self.num_heads==32 and self.top_k==2048 and self.output_dtype==cutlass.Float16))
                                if (not self.dsv4_bulk_positions) and (not self.glm_bulk_positions) and lane%8==0:sample_positions[kv_idx*128+local_row]=physical
                            else:
                                if cutlass.const_expr(self.dsv4_positions):
                                    if cutlass.const_expr(self.dsv4_bulk_positions or self.glm_bulk_positions):physical=sample_positions[kv_idx*128+local_row]
                                    else:
                                        physical=dsv4_cached_sample_key(kv_idx*128+local_row,0 if cutlass.const_expr(self.pro_common_domain) else q_block_idx,sample_lower,sample_upper,self.top_k,coverage=4,sample_count=(256 if self.dsv4 and self.top_k==512 and self.fixed_samples==256 else 0))
                                    if (not self.dsv4_bulk_positions) and (not self.glm_bulk_positions) and lane%8==0:sample_positions[kv_idx*128+local_row]=physical
                                else:
                                    physical=individual_sample_key(kv_idx*128+local_row,q_block_idx,mCuSeqLenKStart,mCuSeqLenKEnd,num_q_tokens,self.block_q,self.top_k,dsv4=self.dsv4)
                            feature=(lane%8)*16
                            src=cute.local_tile(gather_kv[physical,None],(16,),(lane%8,))
                            row_view=cute.slice_(sKV_0,((local_row,None),0,feature//32,stage))
                            dst=cute.local_tile(row_view,(16,),((lane%8)%2,))
                            cute.copy(data_atom,src,dst)
                            if lane%8==0:
                                src_scale=cute.local_tile(gather_scales,(1,),(physical,))
                                dst_scale=cute.local_tile(sScales_0[None,stage],(1,),(local_row,))
                                cute.copy(scale_atom,src_scale,dst_scale)
                        cute.arch.cp_async_commit_group()
                    pending=cutlass.Int32(0)
                    while pending<num_kv:
                        kv_idx=pending+1
                        if kv_idx<num_kv:
                            kv_prod_state_0=pipeline.PipelineState(self.num_kv_stages,kv_idx,kv_idx%self.num_kv_stages,1^((kv_idx//self.num_kv_stages)%2))
                            kv_pipeline_0.producer_acquire(kv_prod_state_0)
                            stage=kv_prod_state_0.index
                            lane=cute.arch.lane_idx()
                            data_atom=cute.make_copy_atom(cpasync.CopyG2SOp(),cutlass.Float8E4M3FN,num_bits_per_copy=128)
                            scale_atom=cute.make_copy_atom(cpasync.CopyG2SOp(),cutlass.Float32,num_bits_per_copy=32)
                            if cutlass.const_expr(self.dsv4_bulk_positions or self.glm_bulk_positions):
                                for position_half in cutlass.range_constexpr(2):
                                    owned_row=(warp_idx-16)*64+position_half*32+lane
                                    if cutlass.const_expr(self.glm_bulk_positions):sample_positions[kv_idx*128+owned_row]=shape_sample_key(self.sample_count_override,kv_idx*128+owned_row,q_block_idx,sample_lower,sample_upper,self.top_k,heads=self.num_heads,coverage=4,common=False)
                                    else:sample_positions[kv_idx*128+owned_row]=dsv4_cached_sample_key(kv_idx*128+owned_row,0 if cutlass.const_expr(self.pro_common_domain) else q_block_idx,sample_lower,sample_upper,self.top_k,coverage=4,sample_count=(256 if self.dsv4 and self.top_k==512 and self.fixed_samples==256 else 0))
                                cute.arch.sync_warp()
                            for row_group in cutlass.range(16):
                                local_row=row_group*4+lane//8+(warp_idx-16)*64
                                if cutlass.const_expr(self.num_heads in (4,32)):
                                    if cutlass.const_expr(self.glm_bulk_positions):physical=sample_positions[kv_idx*128+local_row]
                                    else:physical = shape_sample_key(self.sample_count_override,kv_idx*128+local_row,q_block_idx,sample_lower,sample_upper,self.top_k,heads=self.num_heads,coverage=4 if self.num_heads==32 else 8,common=(self.num_heads==32 and self.top_k==2048 and self.output_dtype==cutlass.Float16))
                                    if (not self.dsv4_bulk_positions) and (not self.glm_bulk_positions) and lane%8==0:sample_positions[kv_idx*128+local_row]=physical
                                else:
                                    if cutlass.const_expr(self.dsv4_positions):
                                        if cutlass.const_expr(self.dsv4_bulk_positions or self.glm_bulk_positions):physical=sample_positions[kv_idx*128+local_row]
                                        else:
                                            physical=dsv4_cached_sample_key(kv_idx*128+local_row,0 if cutlass.const_expr(self.pro_common_domain) else q_block_idx,sample_lower,sample_upper,self.top_k,coverage=4,sample_count=(256 if self.dsv4 and self.top_k==512 and self.fixed_samples==256 else 0))
                                        if (not self.dsv4_bulk_positions) and (not self.glm_bulk_positions) and lane%8==0:sample_positions[kv_idx*128+local_row]=physical
                                    else:
                                        physical=individual_sample_key(kv_idx*128+local_row,q_block_idx,mCuSeqLenKStart,mCuSeqLenKEnd,num_q_tokens,self.block_q,self.top_k,dsv4=self.dsv4)
                                feature=(lane%8)*16
                                src=cute.local_tile(gather_kv[physical,None],(16,),(lane%8,))
                                row_view=cute.slice_(sKV_0,((local_row,None),0,feature//32,stage))
                                dst=cute.local_tile(row_view,(16,),((lane%8)%2,))
                                cute.copy(data_atom,src,dst)
                                if lane%8==0:
                                    src_scale=cute.local_tile(gather_scales,(1,),(physical,))
                                    dst_scale=cute.local_tile(sScales_0[None,stage],(1,),(local_row,))
                                    cute.copy(scale_atom,src_scale,dst_scale)
                            cute.arch.cp_async_commit_group()
                            cute.arch.cp_async_wait_group(1)
                        else:
                            cute.arch.cp_async_wait_group(0)
                        cute.arch.sync_warp()
                        cute.arch.fence_view_async_shared()
                        commit_state=pipeline.PipelineState(self.num_kv_stages,pending,pending%self.num_kv_stages,1^((pending//self.num_kv_stages)%2))
                        kv_pipeline_0.producer_commit(commit_state)
                        pending+=1

        elif is_umma_warp_0:
            cute.arch.warpgroup_reg_dealloc(self.store_registers)
            subset_barrier(barrier_id=1,number_of_threads=544)
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base_0 = cute.make_tensor(tmem_ptr, tCtAcc_fake_staged.layout)
            tCtAcc_base_1 = cute.make_tensor(
                tmem_ptr + cols_per_group, tCtAcc_fake_staged.layout
            )

            if is_leader_cta:
                num_k_blocks = cute.size(tCrA_0.shape[2])
                q_stage_0 = cutlass.Int32(0)

                while has_work:
                    q_block_idx_old = prev_q_block_idx
                    prev_q_block_idx = next_q_block_idx
                    q_block_idx = next_q_block_idx
                    kv_idx = next_kv_idx
                    num_kv = next_num_kv
                    if cutlass.const_expr(self.mode == "sample"):
                        num_kv = (cutlass.Int32(self.fixed_samples) if cutlass.const_expr(self.fixed_samples>0) else block_samples(q_block_idx,mCuSeqLenKEnd,num_q_tokens,block_q_val,self.top_k,heads=self.num_heads,coverage=4 if self.num_heads==32 else 8,dsv4=self.dsv4,anchor_upper=(num_kv_tokens if cutlass.const_expr(self.pro_common_domain) else 0)))//128

                    if q_block_idx != q_block_idx_old:
                        if q_block_idx_old < num_q_blocks:
                            q_cons_state_umma_0.advance()
                        q_pipeline.consumer_wait(q_cons_state_umma_0)
                        q_stage_0 = q_cons_state_umma_0.index

                    kv_pipeline_0.consumer_wait(kv_cons_state_umma_0)
                    umma_pipeline_0.producer_acquire(umma_prod_state_0)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    kv_stage = kv_cons_state_umma_0.index
                    tCtAcc_0 = tCtAcc_base_0[
                        (None, None, None, umma_prod_state_0.index)
                    ]
                    for k_block in cutlass.range_constexpr(num_k_blocks):
                        cute.gemm(
                            tiled_mma,
                            tCtAcc_0,
                            tCrA_0[None, None, k_block, kv_stage],
                            tCrB[None, None, k_block, q_stage_0],
                            tCtAcc_0,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    kv_cons_state_umma_0.advance()
                    umma_pipeline_0.producer_commit(umma_prod_state_0)
                    umma_prod_state_0.advance()

                    next_kv_idx = kv_idx + NUM_MATH_WG * self.splits
                    if next_kv_idx >= num_kv:
                        next_q_block_idx = q_block_idx+cute.arch.grid_dim()[2] if cutlass.const_expr(self.fixed_persistent) else num_q_blocks
                        next_kv_idx = _kv_lo_for_block(next_q_block_idx)
                        if next_q_block_idx < num_q_blocks:
                            q_last_next = min(
                                next_q_block_idx * block_q_val + block_q_val - 1,
                                num_q_tokens - 1,
                            )
                            next_num_kv = ((cutlass.Int32(self.fixed_samples) if cutlass.const_expr(self.fixed_samples>0) else runtime_samples((num_kv_tokens if cutlass.const_expr(self.pro_common_domain) else mCuSeqLenKEnd[q_last_next]),self.top_k,heads=self.num_heads,coverage=4 if self.num_heads==32 else 8,dsv4=self.dsv4))//128 if cutlass.const_expr(self.mode=="sample") else ((mCuSeqLenKEnd[q_last_next] + block_kv_val - 1) // block_kv_val))
                    has_work = next_q_block_idx < num_q_blocks

        elif is_math_warp:
            cute.arch.warpgroup_reg_alloc(self.math_registers)
            tmem.allocate(num_tmem_alloc_cols_total)
            subset_barrier(barrier_id=1,number_of_threads=544)
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tmem.relinquish_alloc_permit()
            tCtAcc_base_0 = cute.make_tensor(tmem_ptr, tCtAcc_fake_staged.layout)
            local_tidx = tidx % 128
            query_slot = warpgroup_idx
            cC = cute.make_identity_tensor(epi_sub_mn)
            tAcc_ref = tCtAcc_base_0[(None,None,None,0)][((None,None),0,0)]
            tAcc_ref_epi = cute.flat_divide(tAcc_ref,epi_sub_mn)
            tiled_copy_ref = tcgen05.make_tmem_copy(copy_atom_t2r,tAcc_ref_epi[(None,None,0,0)])
            thr_copy_ref = tiled_copy_ref.get_slice(local_tidx)
            tTR_cC = thr_copy_ref.partition_D(cC)
            m_coord = tTR_cC[0][0]
            tTR_rAcc = cute.make_fragment_like(tTR_cC,self.acc_dtype)
            w_cache = cute.make_rmem_tensor((1,),cutlass.Float32)
            zero4 = cute.full((4,),0.0,dtype=cutlass.Float32)
            if cutlass.const_expr(self.num_heads == 64):
                while has_work:
                    q_block_idx = next_q_block_idx
                    num_kv = next_num_kv
                    q_pipeline.consumer_wait(q_cons_state)
                    q_stage_local = q_cons_state.index
                    outrow = q_block_idx*self.block_q+query_slot
                    token_idx = min(outrow,num_q_tokens-1)
                    kv_start = mCuSeqLenKStart[token_idx]
                    kv_end = mCuSeqLenKEnd[token_idx]
                    w = cutlass.Float32(0.0)
                    if outrow<num_q_tokens:
                        if local_tidx==0:counts[outrow]=0
                        if cutlass.const_expr(self.use_relu_trick):
                            if local_tidx<64:
                                offset=(num_q_tokens+31)//32*32+outrow*64+local_tidx
                                thresholds[offset]=mW_raw[outrow*64+local_tidx]*0.5
                        for col in cutlass.range(local_tidx,512,128):mLogits[outrow,col]=self.output_dtype(float('-inf'))
                        for col in cutlass.range(local_tidx,256,128):candidates[outrow,col]=self.output_dtype(float('-inf'))
                    kv_idx = _kv_lo_for_block(q_block_idx)
                    # Every producer and both math WGs traverse the same KV tiles.
                    while kv_idx < num_kv:
                        sample_pos=kv_idx*128+m_coord
                        if cutlass.const_expr(self.num_heads in (4,32)):
                            kv_pos = shape_sample_key(self.sample_count_override,sample_pos,q_block_idx,sample_lower,sample_upper,self.top_k,heads=self.num_heads,coverage=4 if self.num_heads==32 else 8,common=(self.num_heads==32 and self.top_k==2048 and self.output_dtype==cutlass.Float16))
                        else:
                            kv_pos=cutlass.Int32(0)
                            if cutlass.const_expr(not self.dsv4_positions):
                                kv_pos=individual_sample_key(sample_pos,q_block_idx,mCuSeqLenKStart,mCuSeqLenKEnd,num_q_tokens,self.block_q,self.top_k,dsv4=self.dsv4)
                        kv_ready=kv_pipeline_0.consumer_try_wait(kv_cons_state_math_0)
                        umma_ready=umma_pipeline_0.consumer_try_wait(umma_cons_state_0)
                        kv_pipeline_0.consumer_wait(kv_cons_state_math_0,kv_ready)
                        scale_val=sScales_0[(m_coord,kv_cons_state_math_0.index)]
                        umma_pipeline_0.consumer_wait(umma_cons_state_0,umma_ready)
                        if cutlass.const_expr(self.dsv4_positions):kv_pos=sample_positions[(q_stage_local*self.sample_position_stride if cutlass.const_expr(self.fixed_persistent) else 0)+sample_pos]
                        # Both query WGs have loaded scales, and UMMA completion
                        # proves it has finished reading the same KV stage.
                        kv_pipeline_0.consumer_release(kv_cons_state_math_0)
                        kv_cons_state_math_0.advance()
                        tAcc = tCtAcc_base_0[(None,None,None,umma_cons_state_0.index)][((None,None),0,0)]
                        tTR = thr_copy_ref.partition_S(cute.flat_divide(tAcc,epi_sub_mn))
                        if cutlass.const_expr(self.native_lowp):
                            s_acc=cute.make_rmem_tensor((2,),cutlass.Uint32)
                            s_acc.fill(0)
                        else:
                            s_acc = cute.full((4,),0.0,dtype=cutlass.Float32)
                        for i in cutlass.range_constexpr(num_epi_subtiles):
                            cute.copy(tiled_copy_ref,tTR[(None,None,None,0,query_slot*num_epi_subtiles+i)],tTR_rAcc)
                            cute.arch.fence_view_async_tmem_load()
                            if cutlass.const_expr(i==num_epi_subtiles-1):
                                # 256 consumers release only after both query slices
                                # are in registers. Reuse also protects staged SFA.
                                umma_pipeline_0.consumer_release(umma_cons_state_0)
                                umma_cons_state_0.advance()
                            if cutlass.const_expr(self.native_lowp):
                                for h in cutlass.range_constexpr(0,64//num_epi_subtiles,4):
                                    base_head=query_slot*64+i*(64//num_epi_subtiles)+h
                                    a01=native_fp8_pack_relu(self.output_dtype,tTR_rAcc[h],tTR_rAcc[h+1])
                                    a23=native_fp8_pack_relu(self.output_dtype,tTR_rAcc[h+2],tTR_rAcc[h+3])
                                    w01=native_weight_pair(sW,base_head,q_stage_local,self.output_dtype,fp8=True)
                                    w23=native_weight_pair(sW,base_head+2,q_stage_local,self.output_dtype,fp8=True)
                                    s_acc[0]=native_fma(self.output_dtype,a01,w01,s_acc[0])
                                    s_acc[1]=native_fma(self.output_dtype,a23,w23,s_acc[1])
                            else:
                                for h in cutlass.range_constexpr(0,64//num_epi_subtiles,4):
                                    x4 = _subview(tTR_rAcc,h,4)
                                    if cutlass.const_expr(self.use_relu_trick):
                                        a4 = indexer_add_packed_f32x2(x4,abs(x4),rnd=_RND_RN)
                                    else:
                                        a4 = cute.max(x4,zero4,propagate_nan=True)
                                    if cutlass.const_expr(i*(64//num_epi_subtiles)+h<0):
                                        w4 = _subview(w_cache,i*(64//num_epi_subtiles)+h,4)
                                    else:
                                        shared_w = cute.make_rmem_tensor((4,),cutlass.Float32)
                                        for j in cutlass.range_constexpr(4):
                                            w=cutlass.Float32(sW[(query_slot*64+i*(64//num_epi_subtiles)+h+j,q_stage_local)])
                                            if cutlass.const_expr(self.use_relu_trick and not self.weights_prepared):w=w*0.5
                                            shared_w[j]=w
                                        w4=shared_w.load()
                                    s_acc = indexer_fma_packed_f32x2(a4,w4,s_acc,rnd=_RND_RN)
                        if cutlass.const_expr(self.native_lowp):
                            reduced=native_add(self.output_dtype,s_acc[0],s_acc[1])
                            score=cutlass.Float32(self.output_dtype(native_lo(self.output_dtype,reduced)+native_hi(self.output_dtype,reduced)))*scale_val
                        else:
                            score = (s_acc[0]+s_acc[1]+s_acc[2]+s_acc[3])*scale_val
                        if outrow<num_q_tokens:
                            rounded=self.output_dtype(float('-inf'))
                            if (kv_pos>=kv_start)&(kv_pos<kv_end):rounded=self.output_dtype(score)
                            mLogits[outrow,sample_pos]=rounded
                            if kv_idx<2:candidates[outrow,sample_pos]=rounded
                        kv_idx=kv_idx+1
                    subset_barrier(barrier_id=cutlass.Int32(10)+query_slot,number_of_threads=128)
                    if outrow<num_q_tokens:
                        hist=cute.make_tensor(sample_hist.iterator+query_slot*264,cute.make_layout(264))
                        scalar=cute.make_tensor(sample_scalar.iterator+query_slot*2,cute.make_layout(2))
                        samples=(cutlass.Int32(self.fixed_samples) if cutlass.const_expr(self.fixed_samples>0) else block_samples(q_block_idx,mCuSeqLenKEnd,num_q_tokens,self.block_q,self.top_k,heads=self.num_heads,coverage=4 if self.num_heads==32 else 8,dsv4=self.dsv4,anchor_upper=(num_kv_tokens if cutlass.const_expr(self.pro_common_domain) else 0)))
                        rank=model_proposal_rank((max(1,num_kv_tokens) if cutlass.const_expr(self.pro_common_domain) else max(1,kv_end-kv_start)),samples,self.top_k,self.sample_sigma,self.num_heads,dsv4=self.dsv4)
                        if cutlass.const_expr(self.sample_rank_sigma>0):
                            rank=shape_margin_rank(max(1,kv_end-kv_start),samples,self.top_k,self.sample_rank_sigma)
                        # Use a quantile only when enough samples are expected above it;
                        # sparse tails use the maximum.
                        high_rank=min(samples,max(1,self.top_k*samples//(2*(max(1,num_kv_tokens) if cutlass.const_expr(self.pro_common_domain) else max(1,kv_end-kv_start)))))
                        if high_rank<8:high_rank=cutlass.Int32(1)
                        propose_parallel(outrow,local_tidx,mLogits,hist,scalar,thresholds,samples,rank,cutlass.Int32(10)+query_slot,high_rank,candidates,ignore_padding=self.dsv4,extend_high=self.dsv4,narrow_range=(samples*self.top_k>=6*(max(1,num_kv_tokens) if cutlass.const_expr(self.pro_common_domain) else max(1,kv_end-kv_start))))
                    q_pipeline.consumer_release(q_cons_state)
                    q_cons_state.advance()
                    next_q_block_idx=q_block_idx+cute.arch.grid_dim()[2] if cutlass.const_expr(self.fixed_persistent) else num_q_blocks
                    if next_q_block_idx<num_q_blocks:
                        q_last_next=min(next_q_block_idx*self.block_q+self.block_q-1,num_q_tokens-1)
                        next_num_kv=(cutlass.Int32(self.fixed_samples) if cutlass.const_expr(self.fixed_samples>0) else runtime_samples((num_kv_tokens if cutlass.const_expr(self.pro_common_domain) else mCuSeqLenKEnd[q_last_next]),self.top_k,heads=self.num_heads,coverage=4 if self.num_heads==32 else 8,dsv4=self.dsv4))//128
                    has_work=next_q_block_idx<num_q_blocks
            else:
                while has_work:
                    q_block_idx = next_q_block_idx
                    num_kv = next_num_kv
                    q_pipeline.consumer_wait(q_cons_state)
                    q_stage_local = q_cons_state.index
                    queries_per_wg = self.block_q // 4
                    cursors = cute.make_rmem_tensor((self.block_q // 4,), cutlass.Int32)
                    for local_q in cutlass.range_constexpr(self.block_q // 4):
                        cursors[local_q] = 0
                        slot = query_slot * queries_per_wg + local_q
                        outrow = q_block_idx * self.block_q + slot
                        if outrow < num_q_tokens:
                            if cutlass.const_expr(self.num_heads==32):
                                candidates[outrow,128+local_tidx]=self.output_dtype(float('-inf'))
                            # All remaining retained slots are overwritten below.
                            if local_tidx == 0:counts[outrow] = 0

                            if cutlass.const_expr(self.use_relu_trick):
                                if local_tidx < self.num_heads:
                                    offset = (num_q_tokens+31)//32*32 + outrow*self.num_heads + local_tidx
                                    thresholds[offset] = mW_raw[outrow*self.num_heads+local_tidx]*0.5
                    kv_idx = _kv_lo_for_block(q_block_idx)
                    while kv_idx < num_kv:
                        sample_pos = kv_idx*128 + m_coord
                        if cutlass.const_expr(self.num_heads in (4,32)):
                            kv_pos = cutlass.Int32(0)
                        else:
                            kv_pos = individual_sample_key(sample_pos,q_block_idx,mCuSeqLenKStart,mCuSeqLenKEnd,num_q_tokens,self.block_q,self.top_k,dsv4=self.dsv4)

                        kv_pipeline_0.consumer_wait(kv_cons_state_math_0)
                        scale_val = cutlass.Float32(1) if cutlass.const_expr(self.qsa_unit_scale) else sScales_0[(m_coord,kv_cons_state_math_0.index)]
                        umma_pipeline_0.consumer_wait(umma_cons_state_0)
                        if cutlass.const_expr(self.num_heads in (4,32)):kv_pos=sample_positions[(q_stage_local*self.sample_position_stride if cutlass.const_expr(self.fixed_persistent) else 0)+sample_pos]

                        kv_pipeline_0.consumer_release(kv_cons_state_math_0)
                        kv_cons_state_math_0.advance()
                        tAcc = tCtAcc_base_0[(None,None,None,umma_cons_state_0.index)][((None,None),0,0)]
                        tTR = thr_copy_ref.partition_S(cute.flat_divide(tAcc,epi_sub_mn))
                        for local_q in cutlass.range_constexpr(self.block_q // 4):
                            slot = query_slot * queries_per_wg + local_q
                            outrow = q_block_idx * self.block_q + slot
                            token_idx = min(outrow,num_q_tokens-1)
                            if cutlass.const_expr(self.num_heads==32):
                                head_acc=head_zero(self.output_dtype if self.native_lowp else None)
                                for epi_i in cutlass.range_constexpr(num_epi_subtiles):
                                    cute.copy(tiled_copy_ref,tTR[(None,None,None,0,slot*num_epi_subtiles+epi_i)],tTR_rAcc)
                                    cute.arch.fence_view_async_tmem_load()
                                    if cutlass.const_expr(local_q==queries_per_wg-1 and epi_i==num_epi_subtiles-1):
                                        umma_pipeline_0.consumer_release(umma_cons_state_0)
                                        umma_cons_state_0.advance()
                                    head_acc=weighted_heads_partial(tTR_rAcc,sW,slot,q_stage_local,epi_i*(32//num_epi_subtiles),32//num_epi_subtiles,32,head_acc,self.use_relu_trick,native_dtype=self.output_dtype if self.native_lowp else None,fp8=True)
                                score=head_sum(head_acc,self.output_dtype if self.native_lowp else None)
                                score=score*scale_val
                            else:
                                cute.copy(tiled_copy_ref,tTR[(None,None,None,0,slot)],tTR_rAcc)
                                cute.arch.fence_view_async_tmem_load()
                                if cutlass.const_expr(local_q == self.block_q // 4 - 1):
                                    umma_pipeline_0.consumer_release(umma_cons_state_0)
                                    umma_cons_state_0.advance()
                                x4 = _subview(tTR_rAcc,0,4)
                                if cutlass.const_expr(self.use_relu_trick):
                                    a4 = indexer_add_packed_f32x2(x4,abs(x4),rnd=_RND_RN)
                                else:
                                    a4 = cute.max(x4,zero4,propagate_nan=True)
                                shared_w = cute.make_rmem_tensor((4,),cutlass.Float32)
                                for h in cutlass.range_constexpr(4):
                                    w = cutlass.Float32(0.0883883461356163) if cutlass.const_expr(self.qsa_weights==2) else sW[(slot*4+h,q_stage_local)]
                                    if cutlass.const_expr(self.use_relu_trick and not self.weights_prepared):w = w*0.5
                                    shared_w[h] = w
                                if cutlass.const_expr(self.native_lowp):
                                    score=head_sum(native_head_fma4(x4,shared_w.load(),head_zero(self.output_dtype),self.output_dtype,True),self.output_dtype) *scale_val
                                else:
                                    acc4 = indexer_fma_packed_f32x2(a4,shared_w.load(),zero4,rnd=_RND_RN)
                                    score = (acc4[0]+acc4[1]+acc4[2]+acc4[3]) *scale_val
                            if outrow < num_q_tokens:
                                kv_start = mCuSeqLenKStart[token_idx]
                                kv_end = mCuSeqLenKEnd[token_idx]
                                rounded = self.output_dtype(float('-inf'))
                                if (kv_pos>=kv_start)&(kv_pos<kv_end):rounded = self.output_dtype(score)
                                if cutlass.const_expr(self.sample_shared):sample_scores[slot,sample_pos]=rounded
                                else:mLogits[outrow,sample_pos] = rounded
                                if kv_idx<2:candidates[outrow,sample_pos] = rounded
                        kv_idx = kv_idx + 1
                    subset_barrier(barrier_id=cutlass.Int32(10)+query_slot,number_of_threads=128)
                    # H4/N64 has four queries per math WG. Each warp builds
                    # one private histogram concurrently, using the same bins/ranks.
                    slot = query_slot*queries_per_wg + local_tidx//32
                    outrow = q_block_idx*self.block_q + slot
                    if outrow<num_q_tokens and (cutlass.const_expr(self.num_heads!=32) or local_tidx//32<queries_per_wg):
                        kv_start = mCuSeqLenKStart[outrow]
                        kv_end = mCuSeqLenKEnd[outrow]
                        hist = cute.make_tensor(sample_hist.iterator+slot*256,cute.make_layout(256))
                        samples = (cutlass.Int32(self.fixed_samples) if cutlass.const_expr(self.fixed_samples>0) else block_samples(q_block_idx,mCuSeqLenKEnd,num_q_tokens,self.block_q,self.top_k,heads=self.num_heads,coverage=4 if self.num_heads==32 else 8,dsv4=self.dsv4,anchor_upper=(num_kv_tokens if cutlass.const_expr(self.pro_common_domain) else 0)))
                        rank = model_proposal_rank((max(1,num_kv_tokens) if cutlass.const_expr(self.pro_common_domain) else max(1,kv_end-kv_start)),samples,self.top_k,self.sample_sigma,self.num_heads,dsv4=self.dsv4)
                        if cutlass.const_expr(self.sample_rank_sigma>0):
                            rank=shape_margin_rank(max(1,kv_end-kv_start),samples,self.top_k,self.sample_rank_sigma)
                        high_rank = min(samples,max(1,self.top_k*samples//(2*(max(1,num_kv_tokens) if cutlass.const_expr(self.pro_common_domain) else max(1,kv_end-kv_start)))))
                        if high_rank<8:high_rank=cutlass.Int32(1)
                        if cutlass.const_expr(self.sample_shared):
                            propose_warp(outrow,cute.arch.lane_idx(),sample_scores,hist,thresholds,samples,rank,high_rank,candidates,extend_short=self.num_heads==32 and not self.glm_short_quantile,score_row=slot)
                        else:
                            propose_warp(outrow,cute.arch.lane_idx(),mLogits,hist,thresholds,samples,rank,high_rank,candidates,extend_short=self.num_heads==32 and not self.glm_short_quantile)
                    subset_barrier(barrier_id=cutlass.Int32(10)+query_slot,number_of_threads=128)
                    q_pipeline.consumer_release(q_cons_state)
                    q_cons_state.advance()
                    next_q_block_idx = q_block_idx+cute.arch.grid_dim()[2] if cutlass.const_expr(self.fixed_persistent) else num_q_blocks
                    has_work = next_q_block_idx<num_q_blocks
            subset_barrier(barrier_id=15,number_of_threads=512)
            tmem.free(tmem_ptr)

        else:
            # All four producer warps donate registers, including the two
            # idle roles, so every SM register partition can serve its math WG.
            cute.arch.warpgroup_reg_dealloc(self.store_registers)



from .selection import finish_recovery, select_or_prepare_retry as select_candidates

from .selection import runtime_samples, block_samples

from .selection import encode
@cute.jit
def publish_bank(score,active,tau,row,index,cursor,bank_rows,dtype:cutlass.Constexpr):
    value=cutlass.Float32(dtype(score));hit=active&((value>=tau)|(value!=value))
    mask=cute.arch.vote_ballot_sync(hit);total=cutlass.Int32(cute.arch.popc(mask))
    lane=cute.arch.lane_idx()
    if hit:
        pos=cursor+cutlass.Int32(cute.arch.popc(mask&((cutlass.Uint32(1)<<lane)-1)))
        bank_rows[row,pos]=cutlass.Int64(encode(value,index,dtype,False))
    return cursor+total


from .policy_dsv4 import individual_sample_key
from .policy_qwen import runtime_samples,block_samples
from .sample_policy import proposal_rank
from .proposal_parallel import propose as propose_parallel

from .policy_qwen import qwen_cached_sample_key

from .proposal_warp import propose_warp

from .weighted_heads import weighted_heads_partial

from .policy_dsv4 import runtime_samples,block_samples,model_sample_key,model_proposal_rank

from .policy_dsv4 import dsv4_cached_sample_key

from .sample_rank_margin import shape_margin_rank

from .sample_count_policy import shape_sample_key

from .native_lowp_math import native_pack, native_pack_relu, native_fma, native_add, native_lo, native_hi

from .native_lowp_math import native_fp8_pack, native_fp8_pack_relu

from .native_heads import head_zero, head_sum, native_head_fma4

from .native_heads import native_weight_pair
