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
from .. import memory as indexer_memory
from .. import tensor_utils as indexer_tensor_utils
from ..packed import fma_packed_f32x2 as indexer_fma_packed_f32x2
from ..packed import add_packed_f32x2 as indexer_add_packed_f32x2
from ..ordering import PipelineTmaAsync as IndexerPipelineTmaAsync

from ..pdl import pdl_prologue
import argparse
import functools

import cuda.bindings.driver as cuda
import cutlass
from .subset_barrier import subset_barrier
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
        persistent_sample: bool = True,
        pdl: bool = False,
        native_bf16: bool = False,
        native_fp16: bool = False,
    ):
        self.pdl = bool(pdl)
        # Opt-in BF16 head math; the default FP32 epilogue is retained.
        self.native_lowp = bool(native_bf16 or native_fp16)
        assert not (native_bf16 and native_fp16)
        assert not native_fp16 or output_dtype == cutlass.Float16
        assert not self.native_lowp or (num_heads in (4, 32, 64) and output_dtype == (cutlass.Float16 if native_fp16 else cutlass.BFloat16) and not use_relu_trick and not weights_prepared)
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
        assert mode=="recover" and splits in (1,4,8,16) and epi_dtype==cutlass.Float32
        self.mode, self.top_k, self.capacity = mode, top_k, capacity
        self.splits, self.packed = splits, packed
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
        self.num_groups = 2

        self.num_math_threads = 256
        self.num_specialized_threads = 128
        self.threads_per_cta = 384
        self.num_math_warps = 8
        self.tma_warp_base = 8
        self.umma_warp_base = 10

        # Reload Q when the Q block changes; reuse it across KV blocks.
        self.num_q_stages = 2 if mode == "filter" else 3

        arch = BaseDSL._get_dsl().get_arch_enum()
        TMEM_COLS = get_max_tmem_alloc_cols(f"sm_{arch.major}{arch.minor}")
        if max_umma_pipeline:
            self.num_umma_stages = min(2, TMEM_COLS // (2 * self.N))
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
            self.num_kv_stages = 3

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
        recovery_counters: cute.Tensor,
        recovery_queue: cute.Tensor,
        recovery_sample_info: cute.Tensor,
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
        num_ctas = min(num_q_blocks,max(1,self.num_sms//self.splits))*self.splits
        if cutlass.const_expr(self.mode == "filter" or (self.mode == "sample" and self.persistent_sample)):
            num_ctas = min(num_q_blocks, self.num_sms) * self.splits

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
            thresholds, counts, candidates, output, recovery_counters, recovery_queue, recovery_sample_info,
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
            SharedStorage,
        ).launch(
            grid=(1, 1, num_ctas),
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, self.splits),
            stream=stream,
            min_blocks_per_mp=1,
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
        recovery_counters: cute.Tensor,
        recovery_queue: cute.Tensor,
        recovery_sample_info: cute.Tensor,
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
        SharedStorage: cutlass.Constexpr,
    ):
        pdl_prologue(self.pdl)
        tid=cute.arch.thread_idx()[0]
        partition=cute.arch.block_idx()[2]%self.splits
        slot=cute.arch.block_idx()[2]//self.splits
        queue_count=recovery_queue[0]
        smem=indexer_memory.SmemAllocator()
        hist=smem.allocate_tensor(cutlass.Int32,cute.make_layout(1538),byte_alignment=128)
        scalar=smem.allocate_tensor(cutlass.Int32,cute.make_layout(2),byte_alignment=16)
        recovery_samples=smem.allocate_tensor(cutlass.Float32,cute.make_layout(768),byte_alignment=128)
        last=smem.allocate_tensor(cutlass.Int32,cute.make_layout(1),byte_alignment=16)
        holding=smem.allocate_tensor(cutlass.Int32,cute.make_layout(1),byte_alignment=16)
        if slot<queue_count:
            task_allocator=indexer_memory.TmemAllocator(
                holding.iterator,
                barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=15,num_threads=384),
                allocator_warp_id=0,is_two_cta=False)
            task_allocator.allocate(self.num_tmem_alloc_cols_total)
            task_allocator.wait_for_alloc()
            task_tmem_ptr=task_allocator.retrieve_ptr(self.acc_dtype)
            while slot<queue_count:
                task=recovery_queue[slot+1]
                row_base=task*self.block_q
                delta=cutlass.Int64((cute.arch.block_idx()[2]//self.splits)*self.block_q-row_base)
                task_logits=cute.make_tensor(mLogits.iterator+delta*mLogits.stride[0],mLogits.layout)
                task_candidates=cute.make_tensor(candidates.iterator+delta*candidates.stride[0],candidates.layout)
                if tid<self.block_q:
                    init_row=row_base+tid
                    if init_row<num_q_tokens:
                        for bank in cutlass.range(partition,8,self.splits):
                            task_candidates[init_row,bank]=cutlass.Int64(0)
                cute.arch.fence_acq_rel_gpu()
                subset_barrier(barrier_id=14,number_of_threads=384)
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
                first=mCuSeqLenKStart[row_base]
                for token in cutlass.range_constexpr(1,self.block_q):
                    first=min(first,mCuSeqLenKStart[min(num_q_tokens-1,row_base+token)])
                last_row=min(num_q_tokens-1,row_base+self.block_q-1)
                lo=(first//256)*2+partition*2
                hi=(mCuSeqLenKEnd[last_row]+127)//128
                if lo<hi:
                    self._gemm_body(tiled_mma, tma_atom_a, mA_mkl, tma_atom_b, mB_nkl, tma_atom_w, mW_tma, mW_raw, tma_atom_s, mS_tma, task_logits, mCuSeqLenKStart, mCuSeqLenKEnd, thresholds, counts, task_candidates, output, num_q_tokens, task+1, num_kv_tokens, cluster_layout_vmnk, a_smem_layout_staged, a_tma_smem_layout, b_smem_layout_staged, w_smem_layout_staged, s_smem_layout_staged, epi_tile, task,partition,task_tmem_ptr,recovery_sample_info,SharedStorage)
                cute.arch.fence_acq_rel_gpu()
                subset_barrier(barrier_id=14,number_of_threads=384)
                if tid==0:
                    prior=cute.arch.atomic_add((recovery_counters.iterator+task).llvm_ptr,cutlass.Int32(1),sem='acq_rel',scope='gpu')
                    cute.arch.atomic_exch(last.iterator.llvm_ptr,cutlass.Int32(prior==self.splits-1),sem='release',scope='cta')
                # Dedicated full-CTA publication and reader-retirement phases.
                # Keep the shared decision immutable until every warp snapshots it.
                cute.arch.barrier(barrier_id=4,number_of_threads=384)
                cute.arch.fence_acq_rel_gpu()
                decision=cutlass.Int32(0)
                if cute.arch.lane_idx()==0:decision=cute.arch.atomic_add(last.iterator.llvm_ptr,cutlass.Int32(0),sem='acquire',scope='cta')
                is_last=cute.arch.shuffle_sync(decision,0)!=0
                cute.arch.barrier(barrier_id=4,number_of_threads=384)
                if is_last:
                    if tid<256:
                        if cutlass.const_expr(self.num_heads == 64):
                            for offset in cutlass.range_constexpr(self.block_q):
                                row=row_base+offset
                                if row<num_q_tokens:
                                    required=min(self.top_k,mCuSeqLenKEnd[row]-mCuSeqLenKStart[row])
                                    if counts[row]<required:
                                        finish_recovery(row,tid,task_candidates,counts,mCuSeqLenKStart,mCuSeqLenKEnd,task_logits,hist,scalar,output,thresholds,self.top_k,self.capacity,self.output_dtype,self.packed,recovery_samples)
                        else:
                            for offset in cutlass.range(self.block_q):
                                row=row_base+offset
                                if row<num_q_tokens:
                                    required=min(self.top_k,mCuSeqLenKEnd[row]-mCuSeqLenKStart[row])
                                    if counts[row]<required:
                                        finish_recovery(row,tid,task_candidates,counts,mCuSeqLenKStart,mCuSeqLenKEnd,task_logits,hist,scalar,output,thresholds,self.top_k,self.capacity,self.output_dtype,self.packed,recovery_samples)
                    subset_barrier(barrier_id=14,number_of_threads=384)
                    if tid==0:recovery_counters[task]=0
                subset_barrier(barrier_id=14,number_of_threads=384)
                # All eight CTAs finish consuming this task before the
                # cluster reuses its fixed Q2 scratch for the next queued task.
                # Hardware clusters guarantee scheduling; no software spin.
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
                slot=slot+cute.arch.grid_dim()[2]//self.splits
            subset_barrier(barrier_id=14,number_of_threads=384)
            task_allocator.free(task_tmem_ptr)
            subset_barrier(barrier_id=14,number_of_threads=384)
        if cute.arch.warp_idx()==0:
            cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)

    @cute.jit
    def _gemm_body(self, tiled_mma: cute.TiledMma, tma_atom_a: cute.CopyAtom, mA_mkl: cute.Tensor, tma_atom_b: cute.CopyAtom, mB_nkl: cute.Tensor, tma_atom_w: cute.CopyAtom, mW_tma: cute.Tensor, mW_raw: cute.Tensor, tma_atom_s: cute.CopyAtom, mS_tma: cute.Tensor, mLogits: cute.Tensor, mCuSeqLenKStart: cute.Tensor, mCuSeqLenKEnd: cute.Tensor, thresholds: cute.Tensor, counts: cute.Tensor, candidates: cute.Tensor, output: cute.Tensor, num_q_tokens: cutlass.Int32, num_q_blocks: cutlass.Int32, num_kv_tokens: cutlass.Int32, cluster_layout_vmnk: cute.Layout, a_smem_layout_staged: cute.ComposedLayout, a_tma_smem_layout: cute.ComposedLayout, b_smem_layout_staged: cute.ComposedLayout, w_smem_layout_staged: cute.Layout, s_smem_layout_staged: cute.Layout, epi_tile: cute.Tile, task_q_block: cutlass.Int32, task_partition: cutlass.Int32, task_tmem_ptr: cute.Pointer, recovery_sample_info: cute.Tensor, SharedStorage: cutlass.Constexpr):
        smem = indexer_memory.SmemAllocator()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        bidx, bidy, bidz = cute.arch.block_idx()
        bidz=task_q_block*self.splits+task_partition
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
        is_math_warp = warp_idx < 8
        is_tma_warp_0 = warp_idx == 8
        is_tma_warp_1 = warp_idx == 9
        is_tma_warp = is_tma_warp_0 | is_tma_warp_1
        is_umma_warp_0 = warp_idx == 10
        is_umma_warp_1 = warp_idx == 11

        NUM_MATH_WG = 2
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
            current_num_kv = runtime_samples(mCuSeqLenKEnd[q_last_token_clamped],self.top_k, sample_factor=16)//128
            current_kv_lo = cutlass.Int32(0)

        if is_tma_warp:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_w)
            cpasync.prefetch_descriptor(tma_atom_s)

        storage = smem.allocate(SharedStorage)

        # === Pipelines ===
        prod_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)

        q_cons_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 8)
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

        math_warps_per_group = self.num_math_warps // 2
        kv_cons_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, math_warps_per_group
        )
        kv_pipeline_0 = IndexerPipelineTmaAsync.create(
            barrier_storage=storage.kv_mbar_0.data_ptr(),
            num_stages=self.num_kv_stages,
            producer_group=prod_group,
            consumer_group=kv_cons_group,
            tx_count=self.num_kv_scale_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            defer_sync=True,
        )
        kv_pipeline_1 = IndexerPipelineTmaAsync.create(
            barrier_storage=storage.kv_mbar_1.data_ptr(),
            num_stages=self.num_kv_stages,
            producer_group=prod_group,
            consumer_group=kv_cons_group,
            tx_count=self.num_kv_scale_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            defer_sync=True,
        )
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

        math_threads_per_group = self.num_math_threads // 2
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

        tmem_alloc_num_threads = 320
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=tmem_alloc_num_threads
        )



        # Persistent K1 keeps per-query scratch in the math groups. TMA/UMMA
        # continue independently; barrier 12 includes only the 256 math lanes.
        if cutlass.const_expr(self.mode == 'sample' and self.persistent_sample):
            sample_hist=smem.allocate_tensor(cutlass.Int32,cute.make_layout(1538),byte_alignment=128)
            sample_scalar=smem.allocate_tensor(cutlass.Int32,cute.make_layout(2),byte_alignment=16)
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # SMEM allocation: per-group KV + shared Q + Weights + Scales
        sKV_0 = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sKV_1 = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
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
        sScales_1 = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=s_smem_layout_staged,
            byte_alignment=128,
        )

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
        subset_barrier(barrier_id=14,number_of_threads=384)


        # ===== WARP-SPECIALIZED EXECUTION =====

        if is_tma_warp_0:
            # TMA warp 0: loads Q+Weights (on Q-block change) + KV group 0
            cute.arch.warpgroup_reg_dealloc(self.store_registers)

            # Prefetch first Q block before the loop to hide TMA latency.
            # Guard with has_work: idle CTAs (bidz >= num_q_blocks) must not
            # issue a TMA load with an out-of-range Q block index.
            if has_work:
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

            while has_work:
                q_block_idx_old = prev_q_block_idx
                prev_q_block_idx = next_q_block_idx
                q_block_idx = next_q_block_idx
                kv_idx = next_kv_idx
                num_kv = next_num_kv

                # On Q-block change: prefetch Q for the NEXT Q block
                if q_block_idx != q_block_idx_old:
                    prefetch_next_q = q_block_idx + cute.arch.grid_dim()[2] // self.splits
                    if prefetch_next_q < num_q_blocks:
                        q_pipeline.producer_acquire(q_prod_state)
                        q_bar = q_pipeline.producer_get_barrier(q_prod_state)
                        cute.copy(
                            tma_atom_b,
                            tBgB[(None, prefetch_next_q)],
                            tBsB[(None, q_prod_state.index)],
                            tma_bar_ptr=q_bar,
                            mcast_mask=b_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_w,
                            tWgW[(None, prefetch_next_q)],
                            tWsW[(None, q_prod_state.index)],
                            tma_bar_ptr=q_bar,
                        )
                        q_prod_state.advance()

                # Load KV group 0: block at kv_idx (contiguous, no block_table)
                kv_pipeline_0.producer_acquire(kv_prod_state_0)
                bar = kv_pipeline_0.producer_get_barrier(kv_prod_state_0)
                stage = kv_prod_state_0.index
                cute.copy(
                    tma_atom_a,
                    tAgA_0[(None, sample_tile(kv_idx, q_block_idx, mCuSeqLenKEnd, num_q_tokens, self.block_q, self.sample_size, self.top_k, sample_factor=16) if cutlass.const_expr(self.mode == "sample") else kv_idx, 0)],
                    tAsA_0[(None, stage)],
                    tma_bar_ptr=bar,
                    mcast_mask=a_mcast_mask,
                )
                cute.copy(
                    tma_atom_s,
                    tSgS_0[(None, sample_tile(kv_idx, q_block_idx, mCuSeqLenKEnd, num_q_tokens, self.block_q, self.sample_size, self.top_k, sample_factor=16) if cutlass.const_expr(self.mode == "sample") else kv_idx)],
                    tSsS_0[(None, stage)],
                    tma_bar_ptr=bar,
                )
                kv_prod_state_0.advance()

                # Scheduler advance
                next_kv_idx = kv_idx + NUM_MATH_WG * self.splits
                if next_kv_idx >= num_kv:
                    next_q_block_idx = q_block_idx + cute.arch.grid_dim()[2] // self.splits  # grid-stride
                    next_kv_idx = _kv_lo_for_block(next_q_block_idx)
                    if next_q_block_idx < num_q_blocks:
                        q_last_next = min(
                            next_q_block_idx * block_q_val + block_q_val - 1,
                            num_q_tokens - 1,
                        )
                        next_num_kv = (runtime_samples(mCuSeqLenKEnd[q_last_next],self.top_k, sample_factor=16)//128 if cutlass.const_expr(self.mode=="sample") else ((mCuSeqLenKEnd[q_last_next] + block_kv_val - 1) // block_kv_val))
                has_work = next_q_block_idx < num_q_blocks

        elif is_tma_warp_1:
            # TMA warp 1: loads KV group 1 only (contiguous, no block_table)
            cute.arch.warpgroup_reg_dealloc(self.store_registers)

            while has_work:
                q_block_idx = next_q_block_idx
                kv_idx = next_kv_idx
                num_kv = next_num_kv

                # Load KV group 1: block at kv_idx + 1
                kv_idx_1 = kv_idx + 1
                kv_pipeline_1.producer_acquire(kv_prod_state_1)
                bar = kv_pipeline_1.producer_get_barrier(kv_prod_state_1)
                stage = kv_prod_state_1.index
                cute.copy(
                    tma_atom_a,
                    tAgA_1[(None, sample_tile(kv_idx_1, q_block_idx, mCuSeqLenKEnd, num_q_tokens, self.block_q, self.sample_size, self.top_k, sample_factor=16) if cutlass.const_expr(self.mode == "sample") else kv_idx_1, 0)],
                    tAsA_1[(None, stage)],
                    tma_bar_ptr=bar,
                    mcast_mask=a_mcast_mask,
                )
                cute.copy(
                    tma_atom_s,
                    tSgS_1[(None, sample_tile(kv_idx_1, q_block_idx, mCuSeqLenKEnd, num_q_tokens, self.block_q, self.sample_size, self.top_k, sample_factor=16) if cutlass.const_expr(self.mode == "sample") else kv_idx_1)],
                    tSsS_1[(None, stage)],
                    tma_bar_ptr=bar,
                )
                kv_prod_state_1.advance()

                next_kv_idx = kv_idx + NUM_MATH_WG * self.splits
                if next_kv_idx >= num_kv:
                    next_q_block_idx = q_block_idx + cute.arch.grid_dim()[2] // self.splits
                    next_kv_idx = _kv_lo_for_block(next_q_block_idx)
                    if next_q_block_idx < num_q_blocks:
                        q_last_next = min(
                            next_q_block_idx * block_q_val + block_q_val - 1,
                            num_q_tokens - 1,
                        )
                        next_num_kv = (runtime_samples(mCuSeqLenKEnd[q_last_next],self.top_k, sample_factor=16)//128 if cutlass.const_expr(self.mode=="sample") else ((mCuSeqLenKEnd[q_last_next] + block_kv_val - 1) // block_kv_val))
                has_work = next_q_block_idx < num_q_blocks

        elif is_umma_warp_0:
            cute.arch.warpgroup_reg_dealloc(self.store_registers)
            tmem_ptr = task_tmem_ptr
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
                        num_kv = block_samples(q_block_idx,mCuSeqLenKEnd,num_q_tokens,block_q_val,self.top_k, sample_factor=16)//128

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
                        next_q_block_idx = q_block_idx + cute.arch.grid_dim()[2] // self.splits
                        next_kv_idx = _kv_lo_for_block(next_q_block_idx)
                        if next_q_block_idx < num_q_blocks:
                            q_last_next = min(
                                next_q_block_idx * block_q_val + block_q_val - 1,
                                num_q_tokens - 1,
                            )
                            next_num_kv = (runtime_samples(mCuSeqLenKEnd[q_last_next],self.top_k, sample_factor=16)//128 if cutlass.const_expr(self.mode=="sample") else ((mCuSeqLenKEnd[q_last_next] + block_kv_val - 1) // block_kv_val))
                    has_work = next_q_block_idx < num_q_blocks

        elif is_umma_warp_1:
            cute.arch.warpgroup_reg_dealloc(self.store_registers)
            tmem_ptr = task_tmem_ptr
            tCtAcc_base_0 = cute.make_tensor(tmem_ptr, tCtAcc_fake_staged.layout)
            tCtAcc_base_1 = cute.make_tensor(
                tmem_ptr + cols_per_group, tCtAcc_fake_staged.layout
            )

            if is_leader_cta:
                num_k_blocks_1 = cute.size(tCrA_1.shape[2])
                q_stage_1 = cutlass.Int32(0)

                while has_work:
                    q_block_idx_old = prev_q_block_idx
                    prev_q_block_idx = next_q_block_idx
                    q_block_idx = next_q_block_idx
                    kv_idx = next_kv_idx
                    num_kv = next_num_kv
                    if cutlass.const_expr(self.mode == "sample"):
                        num_kv = block_samples(q_block_idx,mCuSeqLenKEnd,num_q_tokens,block_q_val,self.top_k, sample_factor=16)//128

                    if q_block_idx != q_block_idx_old:
                        if q_block_idx_old < num_q_blocks:
                            q_cons_state_umma_1.advance()
                        q_pipeline.consumer_wait(q_cons_state_umma_1)
                        q_stage_1 = q_cons_state_umma_1.index

                    kv_pipeline_1.consumer_wait(kv_cons_state_umma_1)
                    umma_pipeline_1.producer_acquire(umma_prod_state_1)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    kv_stage_1 = kv_cons_state_umma_1.index
                    tCtAcc_1 = tCtAcc_base_1[
                        (None, None, None, umma_prod_state_1.index)
                    ]
                    for k_block in cutlass.range_constexpr(num_k_blocks_1):
                        cute.gemm(
                            tiled_mma,
                            tCtAcc_1,
                            tCrA_1[None, None, k_block, kv_stage_1],
                            tCrB[None, None, k_block, q_stage_1],
                            tCtAcc_1,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    kv_cons_state_umma_1.advance()
                    umma_pipeline_1.producer_commit(umma_prod_state_1)
                    umma_prod_state_1.advance()

                    next_kv_idx = kv_idx + NUM_MATH_WG * self.splits
                    if next_kv_idx >= num_kv:
                        next_q_block_idx = q_block_idx + cute.arch.grid_dim()[2] // self.splits
                        next_kv_idx = _kv_lo_for_block(next_q_block_idx)
                        if next_q_block_idx < num_q_blocks:
                            q_last_next = min(
                                next_q_block_idx * block_q_val + block_q_val - 1,
                                num_q_tokens - 1,
                            )
                            next_num_kv = (runtime_samples(mCuSeqLenKEnd[q_last_next],self.top_k, sample_factor=16)//128 if cutlass.const_expr(self.mode=="sample") else ((mCuSeqLenKEnd[q_last_next] + block_kv_val - 1) // block_kv_val))
                    has_work = next_q_block_idx < num_q_blocks

        elif is_math_warp:
            cute.arch.warpgroup_reg_alloc(self.math_registers)

            tmem_ptr = task_tmem_ptr
            tCtAcc_base_0 = cute.make_tensor(tmem_ptr, tCtAcc_fake_staged.layout)
            tCtAcc_base_1 = cute.make_tensor(
                tmem_ptr + cols_per_group, tCtAcc_fake_staged.layout
            )

            local_tidx = tidx % 128
            cC = cute.make_identity_tensor(epi_sub_mn)

            # Keep the full row offset and flat extent Int64 because Q*K can exceed 2**31.
            # Only per-tile offsets use Int32.
            logits_stride0 = mLogits.layout.stride[0]
            logits_stride0_i32 = cutlass.Int32(logits_stride0)
            mLogits_flat = cute.make_tensor(
                mLogits.iterator,
                cute.make_layout(cute.size(mLogits, mode=[0]) * logits_stride0),
            )

            if warpgroup_idx == 0:
                tAcc_0_ref = tCtAcc_base_0[(None, None, None, 0)][((None, None), 0, 0)]
                tAcc_0_ref_epi = cute.flat_divide(tAcc_0_ref, epi_sub_mn)
                tiled_copy_ref_0 = tcgen05.make_tmem_copy(
                    copy_atom_t2r, tAcc_0_ref_epi[(None, None, 0, 0)]
                )
                thr_copy_ref_0 = tiled_copy_ref_0.get_slice(local_tidx)
                tTR_cC = thr_copy_ref_0.partition_D(cC)
                m_coord = tTR_cC[0][0]  # this thread's KV position within the block

                tTR_rAcc = cute.make_fragment_like(tTR_cC, self.acc_dtype)

                # The 144-register weight budget caches 36 heads at Q4 or all 64
                # heads at Q2; remaining weights use SMEM.
                if cutlass.const_expr(self.max_w_in_reg > 0):
                    MAX_NUM_W_IN_REG = self.max_w_in_reg
                elif cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                    MAX_NUM_W_IN_REG = 48
                elif cutlass.const_expr(block_q_val == 2):
                    # Use a 40-weight register limit for block_q=2.
                    MAX_NUM_W_IN_REG = 40
                else:
                    MAX_NUM_W_IN_REG = min(48, (144 // block_q_val) // 4 * 4)
                NUM_W_IN_REG = min(MAX_NUM_W_IN_REG, num_heads)
                w_cache = cute.make_rmem_tensor(
                    (NUM_W_IN_REG * block_q_val,), self.epi_dtype
                )
                # Per-token KV boundary cache (loaded on Q-block change)
                kv_start_cache = [cutlass.Int32(0) for _ in range(block_q_val)]
                kv_end_cache = [cutlass.Int32(0) for _ in range(block_q_val)]
                q_stage_local = cutlass.Int32(0)
                if cutlass.const_expr(self.mode == "filter"):
                    private_cursors=cute.make_rmem_tensor((block_q_val,),cutlass.Int32)
                threshold_cache=[cutlass.Float32(0) for _ in range(block_q_val)]
                out_row_off = cutlass.Int64(0)
                kv_lo_cur = cutlass.Int32(0)

                # Outer loop: one iteration per Q block
                while has_work:
                    if cutlass.const_expr(self.mode == "filter"):
                        for token in cutlass.range_constexpr(block_q_val):private_cursors[token]=0
                    q_block_idx = next_q_block_idx
                    num_kv = next_num_kv
                    if cutlass.const_expr(self.mode == "sample"):
                        num_kv = block_samples(q_block_idx,mCuSeqLenKEnd,num_q_tokens,block_q_val,self.top_k, sample_factor=16)//128

                    if q_block_idx != prev_q_block_idx:
                        if prev_q_block_idx < num_q_blocks:
                            q_pipeline.consumer_release(q_cons_state)
                            q_cons_state.advance()
                        q_pipeline.consumer_wait(q_cons_state)
                        q_stage_local = q_cons_state.index
                        # Flat logits view: carry out_row_off for this Q block
                        out_row_off = q_block_idx * block_q_val * logits_stride0
                        # Load weights and per-token KV boundaries
                        q_base = q_block_idx * block_q_val
                        for t_i in cutlass.range_constexpr(block_q_val):
                            for w_j in cutlass.range_constexpr(NUM_W_IN_REG):
                                w_cache[t_i * NUM_W_IN_REG + w_j] = (sW[t_i * num_heads + w_j, q_stage_local]) * (0.5 if cutlass.const_expr(self.use_relu_trick and not self.weights_prepared) else 1.0)
                            # Padded Q slots inherit the last live KV boundary and write only
                            # the allocated padding rows.
                            token_idx = min(q_base + t_i, num_q_tokens - 1)
                            kv_start_cache[t_i] = mCuSeqLenKStart[token_idx]
                            kv_end_cache[t_i] = mCuSeqLenKEnd[token_idx]
                            if cutlass.const_expr(self.mode == "filter"):
                                threshold_cache[t_i]=thresholds[token_idx]
                            elif cutlass.const_expr(self.mode == "recover"):
                                threshold_cache[t_i]=prepare_threshold(token_idx,recovery_sample_info,counts,self.top_k)
                        # First KV tile of this block: min(ks) aligned down to
                        # SPLIT_KV (must match _kv_lo_for_block in the TMA/UMMA
                        # warps). Tiles below it are fully predicated away.
                        ks_min = kv_start_cache[0]
                        for t_j in cutlass.range_constexpr(1, block_q_val):
                            ks_min = min(ks_min, kv_start_cache[t_j])
                        kv_lo_cur = (
                            ks_min // (block_kv_val * NUM_MATH_WG)
                        ) * NUM_MATH_WG + (bidz % self.splits) * NUM_MATH_WG
                    if cutlass.const_expr(self.mode == "sample"):
                        kv_lo_cur = cutlass.Int32(0)
                    prev_q_block_idx = q_block_idx

                    # Fold the Int64 row offset into a per-Q base pointer; tile indices stay Int32.
                    mLogits_blk = cute.make_tensor(
                        mLogits_flat.iterator + out_row_off,
                        cute.make_layout(block_q_val * logits_stride0_i32),
                    )

                    # Inner loop: this Q block's KV tiles [kv_lo_cur, num_kv)
                    kv_idx = kv_lo_cur + cutlass.Int32(0)
                    kv_go = cutlass.Boolean(True)
                    if cutlass.const_expr(self.mode == 'sample' and self.persistent_sample):
                        for token in cutlass.range_constexpr(block_q_val):
                            sample_row=q_block_idx*block_q_val+token
                            if sample_row<num_q_tokens:
                                for col in cutlass.range(m_coord+0*128,self.sample_size,256):
                                    mLogits[sample_row,col]=self.output_dtype(float('-inf'))
                                if cutlass.const_expr(0==0 and self.use_relu_trick):
                                    if local_tidx<64:
                                        woff=(num_q_tokens+31)//32*32+sample_row*64+local_tidx
                                        thresholds[woff]=mW_raw[sample_row*64+local_tidx]*0.5
                    if cutlass.const_expr(self.mode == 'filter'):
                        bank_span=(candidates.shape[1]-8)//8
                        bank_id=warpgroup_idx*4+local_tidx//32
                        bank_rows=cute.make_tensor(candidates.iterator + cutlass.Int64(q_block_idx*block_q_val)*candidates.stride[0]+8+bank_id*bank_span,cute.make_layout((block_q_val,bank_span),stride=(candidates.stride[0],1)))
                    while kv_go:
                        sample_pos = kv_idx * block_kv_val + m_coord
                        kv_pos = sample_pos
                        if cutlass.const_expr(self.mode == "sample"):
                            kv_pos = sample_tile(kv_idx, q_block_idx, mCuSeqLenKEnd, num_q_tokens, self.block_q, self.sample_size, self.top_k, sample_factor=16) * block_kv_val + m_coord

                        # Paired pipeline polls: overlap KV+UMMA mbarrier round-trips
                        if cutlass.const_expr(self.remove_kv_wait_in_epilogue):
                            umma_pipeline_0.consumer_wait(umma_cons_state_0)
                        else:
                            kv_ready_0 = kv_pipeline_0.consumer_try_wait(
                                kv_cons_state_math_0
                            )
                            umma_ready_0 = umma_pipeline_0.consumer_try_wait(
                                umma_cons_state_0
                            )
                            kv_pipeline_0.consumer_wait(
                                kv_cons_state_math_0, kv_ready_0
                            )
                            sc_stage_0 = kv_cons_state_math_0.index
                            scale_val = sScales_0[(m_coord, sc_stage_0)]
                            umma_pipeline_0.consumer_wait(
                                umma_cons_state_0, umma_ready_0
                            )

                        tCtAcc_c0 = tCtAcc_base_0[
                            (None, None, None, umma_cons_state_0.index)
                        ]
                        tAcc_c0 = tCtAcc_c0[((None, None), 0, 0)]
                        tAcc_c0_epi = cute.flat_divide(tAcc_c0, epi_sub_mn)
                        # Every staged fragment has the same layout. Reuse the
                        # copy descriptor and thread slice built from stage 0;
                        # only the TMEM base address changes between stages.
                        tTR_0 = thr_copy_ref_0.partition_S(tAcc_c0_epi)

                        if cutlass.const_expr(self.early_tmem_copy):
                            cute.copy(
                                tiled_copy_ref_0,
                                tTR_0[(None, None, None, 0, 0)],
                                tTR_rAcc,
                            )
                            if cutlass.const_expr(self.remove_kv_wait_in_epilogue):
                                sc_stage_0 = kv_cons_state_math_0.index
                                scale_val = sScales_0[(m_coord, sc_stage_0)]
                            kv_pipeline_0.consumer_release(kv_cons_state_math_0)
                            kv_cons_state_math_0.advance()
                            cute.arch.fence_view_async_tmem_load()
                        else:
                            if cutlass.const_expr(self.remove_kv_wait_in_epilogue):
                                sc_stage_0 = kv_cons_state_math_0.index
                                scale_val = sScales_0[(m_coord, sc_stage_0)]
                            kv_pipeline_0.consumer_release(kv_cons_state_math_0)
                            kv_cons_state_math_0.advance()
                            cute.copy(
                                tiled_copy_ref_0,
                                tTR_0[(None, None, None, 0, 0)],
                                tTR_rAcc,
                            )
                            cute.arch.fence_view_async_tmem_load()

                        subtile_n = num_heads // num_epi_subtiles
                        if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                            packed_zero = cute.full((4,), 0.0, dtype=cutlass.Float16)

                        sw_col = sW[(None, q_stage_local)]
                        # When the relu trick is on, folds its missing /2 in
                        # here; off by default, so this is just scale_val.
                        if cutlass.const_expr(self.use_relu_trick):
                            scale_half = scale_val
                        else:
                            scale_half = scale_val

                        if cutlass.const_expr(False):
                            row_acc=cute.make_rmem_tensor((cute.size(tTR_rAcc)*block_q_val,),cutlass.Float32)
                            row_sums=cute.make_rmem_tensor((4*block_q_val,),cutlass.Float32)
                            for clear in cutlass.range_constexpr(4*block_q_val):row_sums[clear]=0.0
                            for head_tile in cutlass.range_constexpr(num_epi_subtiles):
                                for token in cutlass.range_constexpr(block_q_val):
                                    token_acc=cute.make_tensor(row_acc.iterator+token*cute.size(tTR_rAcc),tTR_rAcc.layout)
                                    cute.copy(tiled_copy_ref_0,tTR_0[(None,None,None,0,head_tile+token*num_epi_subtiles)],token_acc)
                                    cute.arch.fence_view_async_tmem_load()
                                if cutlass.const_expr(head_tile==num_epi_subtiles-1):
                                    umma_pipeline_0.consumer_release(umma_cons_state_0)
                                    umma_cons_state_0.advance()
                                for head in cutlass.range_constexpr(0,subtile_n,4):
                                    h_global=head_tile*subtile_n+head
                                    for token in cutlass.range_constexpr(block_q_val):
                                        dots=_subview(row_acc,token*cute.size(tTR_rAcc)+head,4)
                                        if cutlass.const_expr(self.use_relu_trick):
                                            activated=indexer_add_packed_f32x2(dots,abs(dots),rnd=_RND_RN)
                                        else:
                                            activated=cute.max(dots,cute.full((4,),0.0,dtype=cutlass.Float32),propagate_nan=True)
                                        if cutlass.const_expr(h_global<NUM_W_IN_REG):
                                            weight=_subview(w_cache,token*NUM_W_IN_REG+h_global,4)
                                        else:
                                            weight=_subview(sw_col,token*num_heads+h_global,4)
                                            if cutlass.const_expr(self.use_relu_trick and not self.weights_prepared):weight=weight*0.5
                                        partial=_subview(row_sums,token*4,4)
                                        updated=indexer_fma_packed_f32x2(activated,weight,partial,rnd=_RND_RN)
                                        cute.make_tensor(row_sums.iterator+token*4,cute.make_layout(4)).store(updated)
                            for token in cutlass.range_constexpr(block_q_val):
                                offset=token*4
                                result=row_sums[offset]+row_sums[offset+1]+row_sums[offset+2]+row_sums[offset+3]
                                outrow=q_block_idx*block_q_val+token
                                if outrow<num_q_tokens:
                                    private_cursors[token]=publish_bank(result*scale_half,(kv_pos>=kv_start_cache[token])&(kv_pos<kv_end_cache[token]),threshold_cache[token],token,kv_pos,private_cursors[token],bank_rows,self.output_dtype)
                        else:
                            for t in cutlass.range_constexpr(block_q_val):
                                if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                                    # Keep each token's accumulator independent of
                                    # the shared ReLU zero under staged assignment.
                                    ps = cute.full((4,), 0.0, dtype=cutlass.Float16)
                                else:
                                    if cutlass.const_expr(self.native_lowp):
                                        s_acc=cute.make_rmem_tensor((2,),cutlass.Uint32)
                                        s_acc.fill(0)
                                    else:
                                        s_acc = cute.full((4,), 0.0, dtype=cutlass.Float32)
                                    if cutlass.const_expr(not self.use_relu_trick):
                                        zero4 = cute.full((4,), 0.0, dtype=cutlass.Float32)
                                for i in cutlass.range_constexpr(num_epi_subtiles):
                                    if t > 0 or i > 0:
                                        cute.copy(
                                            tiled_copy_ref_0,
                                            tTR_0[
                                                (
                                                    None,
                                                    None,
                                                    None,
                                                    0,
                                                    t * num_epi_subtiles + i,
                                                )
                                            ],
                                            tTR_rAcc,
                                        )
                                        cute.arch.fence_view_async_tmem_load()
                                    if t == block_q_val - 1 and i == num_epi_subtiles - 1:
                                        umma_pipeline_0.consumer_release(umma_cons_state_0)
                                        umma_cons_state_0.advance()
                                    # Reg-path: weights from registers
                                    reg_h_end = min(
                                        subtile_n, max(0, NUM_W_IN_REG - i * subtile_n)
                                    )
                                    if cutlass.const_expr(self.native_lowp):
                                        for h in cutlass.range_constexpr(0, reg_h_end, 4):
                                            h_g=i*subtile_n+h
                                            x4=_subview(tTR_rAcc,h,4)
                                            w4=_subview(w_cache, t*NUM_W_IN_REG+h_g,4)
                                            s_acc[0]=native_fma(self.output_dtype,native_fp8_pack_relu(self.output_dtype,x4[0],x4[1]),native_fp8_pack(self.output_dtype,w4[0],w4[1]),s_acc[0])
                                            s_acc[1]=native_fma(self.output_dtype,native_fp8_pack_relu(self.output_dtype,x4[2],x4[3]),native_fp8_pack(self.output_dtype,w4[2],w4[3]),s_acc[1])
                                    else:
                                        for h in cutlass.range_constexpr(0, reg_h_end, 4):
                                            n0 = h
                                            h_g = i * subtile_n + h
                                            if cutlass.const_expr(
                                                self.epi_dtype == cutlass.Float16
                                            ):
                                                pa = _subview(tTR_rAcc, n0, 4).to(
                                                    cutlass.Float16
                                                )
                                                pa = cute.max(pa, packed_zero)
                                                r0 = t * NUM_W_IN_REG + h_g
                                                pw = _subview(w_cache, r0, 4)
                                                ps = cute.fma(pa, pw, ps)
                                            else:
                                                if cutlass.const_expr(self.use_relu_trick):
                                                    # Doubled ReLU folds its factor of 1/2 into scale_half.
                                                    # It requires bounded inputs because x+abs(x) can overflow.
                                                    x4 = _subview(tTR_rAcc, n0, 4)
                                                    a4 = indexer_add_packed_f32x2(
                                                        x4, abs(x4), rnd=_RND_RN
                                                    )
                                                else:
                                                    a4 = cute.max(
                                                        _subview(tTR_rAcc, n0, 4),
                                                        zero4,
                                                        propagate_nan=True,
                                                    )
                                                r0 = t * NUM_W_IN_REG + h_g
                                                w4 = _subview(w_cache, r0, 4)
                                                s_acc = indexer_fma_packed_f32x2(
                                                    a4, w4, s_acc, rnd=_RND_RN
                                                )
                                    # SMEM-path: weights from shared mem
                                    smem_h_start = max(0, NUM_W_IN_REG - i * subtile_n)
                                    if cutlass.const_expr(self.native_lowp):
                                        for h in cutlass.range_constexpr(smem_h_start, subtile_n, 4):
                                            h_g=i*subtile_n+h
                                            x4=_subview(tTR_rAcc,h,4)
                                            w4=_subview(sw_col, t*num_heads+h_g,4)
                                            s_acc[0]=native_fma(self.output_dtype,native_fp8_pack_relu(self.output_dtype,x4[0],x4[1]),native_fp8_pack(self.output_dtype,w4[0],w4[1]),s_acc[0])
                                            s_acc[1]=native_fma(self.output_dtype,native_fp8_pack_relu(self.output_dtype,x4[2],x4[3]),native_fp8_pack(self.output_dtype,w4[2],w4[3]),s_acc[1])
                                    else:
                                        for h in cutlass.range_constexpr(
                                            smem_h_start, subtile_n, 4
                                        ):
                                            n0 = h
                                            h_g = i * subtile_n + h
                                            if cutlass.const_expr(
                                                self.epi_dtype == cutlass.Float16
                                            ):
                                                pa = _subview(tTR_rAcc, n0, 4).to(
                                                    cutlass.Float16
                                                )
                                                pa = cute.max(pa, packed_zero)
                                                w_i = t * num_heads + h_g
                                                pw = _subview(sw_col, w_i, 4)
                                                ps = cute.fma(pa, pw, ps)
                                            else:
                                                if cutlass.const_expr(self.use_relu_trick):
                                                    # relu(x) = (x + |x|) / 2 (see
                                                    # reg-path above).
                                                    x4 = _subview(tTR_rAcc, n0, 4)
                                                    a4 = indexer_add_packed_f32x2(
                                                        x4, abs(x4), rnd=_RND_RN
                                                    )
                                                else:
                                                    a4 = cute.max(
                                                        _subview(tTR_rAcc, n0, 4),
                                                        zero4,
                                                        propagate_nan=True,
                                                    )
                                                w_i = t * num_heads + h_g
                                                w4 = _subview(sw_col, w_i, 4).to(cutlass.Float32)
                                                if cutlass.const_expr(self.use_relu_trick and not self.weights_prepared):
                                                    w4 = w4 * 0.5
                                                s_acc = indexer_fma_packed_f32x2(
                                                    a4, w4, s_acc, rnd=_RND_RN
                                                )

                                if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                                    result_t = (ps[0] + ps[2]) + (ps[1] + ps[3])
                                else:
                                    if cutlass.const_expr(self.native_lowp):
                                        reduced=native_add(self.output_dtype,s_acc[0],s_acc[1])
                                        result_t=cutlass.Float32(self.output_dtype(native_lo(self.output_dtype,reduced)+native_hi(self.output_dtype,reduced)))
                                    else:
                                        result_t = s_acc[0] + s_acc[1] + s_acc[2] + s_acc[3]

                                # Prefill boundary check + flat logits store
                                if cutlass.const_expr(self.mode == "filter"):
                                    outrow=q_block_idx*block_q_val+t
                                    if outrow<num_q_tokens:
                                        private_cursors[t]=publish_bank(result_t*scale_half,(kv_pos>=kv_start_cache[t])&(kv_pos<kv_end_cache[t]),threshold_cache[t],t,kv_pos,private_cursors[t],bank_rows,self.output_dtype)
                                else:
                                    emit(result_t * scale_half, q_block_idx * block_q_val + t, kv_pos, sample_pos, (kv_pos >= kv_start_cache[t]) & (kv_pos < kv_end_cache[t]), num_q_tokens, mLogits, thresholds, counts, candidates, self.mode, self.output_dtype, self.packed, self.capacity,task_partition%8,recovery_sample_info,self.top_k,threshold_cache[t])

                        # Advance inner KV loop
                        kv_idx = kv_idx + NUM_MATH_WG * self.splits
                        kv_go = kv_idx < num_kv
                    if cutlass.const_expr(self.mode == "filter"):
                        for token in cutlass.range_constexpr(block_q_val):
                            row=q_block_idx*block_q_val+token
                            if row<num_q_tokens and cute.arch.lane_idx()==0:
                                candidates[row,warpgroup_idx*4+local_tidx//32]=cutlass.Int64(private_cursors[token])
                    if cutlass.const_expr(self.mode == 'sample' and self.persistent_sample):
                        subset_barrier(barrier_id=12,number_of_threads=256)
                        for token in cutlass.range_constexpr(block_q_val):
                            sample_row=q_block_idx*block_q_val+token
                            if sample_row<num_q_tokens:
                                if tidx==0:counts[sample_row]=0
                                width=max(1,mCuSeqLenKEnd[sample_row]-mCuSeqLenKStart[sample_row])
                                active_samples=block_samples(sample_row//self.block_q,mCuSeqLenKEnd,num_q_tokens,self.block_q,self.top_k, sample_factor=16)
                                multiple=2 if active_samples==512 else 6
                                rank=min(active_samples,max(1,(multiple*self.top_k*active_samples+width-1)//width))
                                propose(sample_row,tidx,mLogits,sample_hist,sample_scalar,thresholds,active_samples,rank)
                        subset_barrier(barrier_id=12,number_of_threads=256)

                    # Outer: advance to next Q block (grid-stride)
                    next_q_block_idx = q_block_idx + cute.arch.grid_dim()[2] // self.splits
                    if next_q_block_idx < num_q_blocks:
                        q_last_next = min(
                            next_q_block_idx * block_q_val + block_q_val - 1,
                            num_q_tokens - 1,
                        )
                        next_num_kv = (runtime_samples(mCuSeqLenKEnd[q_last_next],self.top_k, sample_factor=16)//128 if cutlass.const_expr(self.mode=="sample") else ((mCuSeqLenKEnd[q_last_next] + block_kv_val - 1) // block_kv_val))
                    has_work = next_q_block_idx < num_q_blocks

                if q_block_idx < num_q_blocks:
                    q_pipeline.consumer_release(q_cons_state)
                    q_cons_state.advance()

            else:
                # Math WG 1: process group 1 (kv_idx + 1)
                tAcc_1_ref = tCtAcc_base_1[(None, None, None, 0)][((None, None), 0, 0)]
                tAcc_1_ref_epi = cute.flat_divide(tAcc_1_ref, epi_sub_mn)
                tiled_copy_ref_1 = tcgen05.make_tmem_copy(
                    copy_atom_t2r, tAcc_1_ref_epi[(None, None, 0, 0)]
                )
                thr_copy_ref_1 = tiled_copy_ref_1.get_slice(local_tidx)
                tTR_cC = thr_copy_ref_1.partition_D(cC)
                m_coord = tTR_cC[0][0]

                tTR_rAcc = cute.make_fragment_like(tTR_cC, self.acc_dtype)

                if cutlass.const_expr(self.max_w_in_reg > 0):
                    MAX_NUM_W_IN_REG = self.max_w_in_reg
                elif cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                    MAX_NUM_W_IN_REG = 48
                else:
                    MAX_NUM_W_IN_REG = 36
                NUM_W_IN_REG = min(MAX_NUM_W_IN_REG, num_heads)
                w_cache = cute.make_rmem_tensor(
                    (NUM_W_IN_REG * block_q_val,), self.epi_dtype
                )
                kv_start_cache = [cutlass.Int32(0) for _ in range(block_q_val)]
                kv_end_cache = [cutlass.Int32(0) for _ in range(block_q_val)]
                q_stage_local = cutlass.Int32(0)
                if cutlass.const_expr(self.mode == "filter"):
                    private_cursors=cute.make_rmem_tensor((block_q_val,),cutlass.Int32)
                threshold_cache=[cutlass.Float32(0) for _ in range(block_q_val)]
                out_row_off = cutlass.Int64(0)
                kv_lo_cur = cutlass.Int32(0)

                while has_work:
                    if cutlass.const_expr(self.mode == "filter"):
                        for token in cutlass.range_constexpr(block_q_val):private_cursors[token]=0
                    q_block_idx = next_q_block_idx
                    num_kv = next_num_kv
                    if cutlass.const_expr(self.mode == "sample"):
                        num_kv = block_samples(q_block_idx,mCuSeqLenKEnd,num_q_tokens,block_q_val,self.top_k, sample_factor=16)//128

                    if q_block_idx != prev_q_block_idx:
                        if prev_q_block_idx < num_q_blocks:
                            q_pipeline.consumer_release(q_cons_state)
                            q_cons_state.advance()
                        q_pipeline.consumer_wait(q_cons_state)
                        q_stage_local = q_cons_state.index
                        out_row_off = q_block_idx * block_q_val * logits_stride0
                        q_base = q_block_idx * block_q_val
                        for t_i in cutlass.range_constexpr(block_q_val):
                            for w_j in cutlass.range_constexpr(NUM_W_IN_REG):
                                w_cache[t_i * NUM_W_IN_REG + w_j] = (sW[t_i * num_heads + w_j, q_stage_local]) * (0.5 if cutlass.const_expr(self.use_relu_trick and not self.weights_prepared) else 1.0)
                            # Padded Q slots inherit the last live KV boundary and write only
                            # the allocated padding rows.
                            token_idx = min(q_base + t_i, num_q_tokens - 1)
                            kv_start_cache[t_i] = mCuSeqLenKStart[token_idx]
                            kv_end_cache[t_i] = mCuSeqLenKEnd[token_idx]
                            if cutlass.const_expr(self.mode == "filter"):
                                threshold_cache[t_i]=thresholds[token_idx]
                            elif cutlass.const_expr(self.mode == "recover"):
                                threshold_cache[t_i]=prepare_threshold(token_idx,recovery_sample_info,counts,self.top_k)
                        # First KV tile of this block: min(ks) aligned down to
                        # SPLIT_KV (must match _kv_lo_for_block in the TMA/UMMA
                        # warps). Tiles below it are fully predicated away.
                        ks_min = kv_start_cache[0]
                        for t_j in cutlass.range_constexpr(1, block_q_val):
                            ks_min = min(ks_min, kv_start_cache[t_j])
                        kv_lo_cur = (
                            ks_min // (block_kv_val * NUM_MATH_WG)
                        ) * NUM_MATH_WG + (bidz % self.splits) * NUM_MATH_WG
                    if cutlass.const_expr(self.mode == "sample"):
                        kv_lo_cur = cutlass.Int32(0)
                    prev_q_block_idx = q_block_idx

                    # Fold the Int64 row offset into a per-Q base pointer; tile indices stay Int32.
                    mLogits_blk = cute.make_tensor(
                        mLogits_flat.iterator + out_row_off,
                        cute.make_layout(block_q_val * logits_stride0_i32),
                    )

                    kv_idx = kv_lo_cur + cutlass.Int32(0)
                    kv_go = cutlass.Boolean(True)
                    if cutlass.const_expr(self.mode == 'sample' and self.persistent_sample):
                        for token in cutlass.range_constexpr(block_q_val):
                            sample_row=q_block_idx*block_q_val+token
                            if sample_row<num_q_tokens:
                                for col in cutlass.range(m_coord+1*128,self.sample_size,256):
                                    mLogits[sample_row,col]=self.output_dtype(float('-inf'))
                                if cutlass.const_expr(1==0 and self.use_relu_trick):
                                    if local_tidx<64:
                                        woff=(num_q_tokens+31)//32*32+sample_row*64+local_tidx
                                        thresholds[woff]=mW_raw[sample_row*64+local_tidx]*0.5
                    if cutlass.const_expr(self.mode == 'filter'):
                        bank_span=(candidates.shape[1]-8)//8
                        bank_id=warpgroup_idx*4+local_tidx//32
                        bank_rows=cute.make_tensor(candidates.iterator + cutlass.Int64(q_block_idx*block_q_val)*candidates.stride[0]+8+bank_id*bank_span,cute.make_layout((block_q_val,bank_span),stride=(candidates.stride[0],1)))
                    while kv_go:
                        kv_idx_1 = kv_idx + 1
                        sample_pos = kv_idx_1 * block_kv_val + m_coord
                        kv_pos = sample_pos
                        if cutlass.const_expr(self.mode == "sample"):
                            kv_pos = sample_tile(kv_idx_1, q_block_idx, mCuSeqLenKEnd, num_q_tokens, self.block_q, self.sample_size, self.top_k, sample_factor=16) * block_kv_val + m_coord

                        if cutlass.const_expr(self.remove_kv_wait_in_epilogue):
                            umma_pipeline_1.consumer_wait(umma_cons_state_1)
                        else:
                            kv_ready_1 = kv_pipeline_1.consumer_try_wait(
                                kv_cons_state_math_1
                            )
                            umma_ready_1 = umma_pipeline_1.consumer_try_wait(
                                umma_cons_state_1
                            )
                            kv_pipeline_1.consumer_wait(
                                kv_cons_state_math_1, kv_ready_1
                            )
                            sc_stage_1 = kv_cons_state_math_1.index
                            scale_val = sScales_1[(m_coord, sc_stage_1)]
                            umma_pipeline_1.consumer_wait(
                                umma_cons_state_1, umma_ready_1
                            )

                        tCtAcc_c1 = tCtAcc_base_1[
                            (None, None, None, umma_cons_state_1.index)
                        ]
                        tAcc_c1 = tCtAcc_c1[((None, None), 0, 0)]
                        tAcc_c1_epi = cute.flat_divide(tAcc_c1, epi_sub_mn)
                        # Match WG0: reuse the stage-invariant copy descriptor
                        # while partitioning the current stage's TMEM tensor.
                        tTR_1 = thr_copy_ref_1.partition_S(tAcc_c1_epi)

                        if cutlass.const_expr(self.early_tmem_copy):
                            cute.copy(
                                tiled_copy_ref_1,
                                tTR_1[(None, None, None, 0, 0)],
                                tTR_rAcc,
                            )
                            if cutlass.const_expr(self.remove_kv_wait_in_epilogue):
                                sc_stage_1 = kv_cons_state_math_1.index
                                scale_val = sScales_1[(m_coord, sc_stage_1)]
                            kv_pipeline_1.consumer_release(kv_cons_state_math_1)
                            kv_cons_state_math_1.advance()
                            cute.arch.fence_view_async_tmem_load()
                        else:
                            if cutlass.const_expr(self.remove_kv_wait_in_epilogue):
                                sc_stage_1 = kv_cons_state_math_1.index
                                scale_val = sScales_1[(m_coord, sc_stage_1)]
                            kv_pipeline_1.consumer_release(kv_cons_state_math_1)
                            kv_cons_state_math_1.advance()
                            cute.copy(
                                tiled_copy_ref_1,
                                tTR_1[(None, None, None, 0, 0)],
                                tTR_rAcc,
                            )
                            cute.arch.fence_view_async_tmem_load()

                        subtile_n = num_heads // num_epi_subtiles
                        if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                            packed_zero = cute.full((4,), 0.0, dtype=cutlass.Float16)

                        sw_col = sW[(None, q_stage_local)]
                        # When the relu trick is on, folds its missing /2 in
                        # here; off by default, so this is just scale_val.
                        if cutlass.const_expr(self.use_relu_trick):
                            scale_half = scale_val
                        else:
                            scale_half = scale_val

                        if cutlass.const_expr(False):
                            row_acc=cute.make_rmem_tensor((cute.size(tTR_rAcc)*block_q_val,),cutlass.Float32)
                            row_sums=cute.make_rmem_tensor((4*block_q_val,),cutlass.Float32)
                            for clear in cutlass.range_constexpr(4*block_q_val):row_sums[clear]=0.0
                            for head_tile in cutlass.range_constexpr(num_epi_subtiles):
                                for token in cutlass.range_constexpr(block_q_val):
                                    token_acc=cute.make_tensor(row_acc.iterator+token*cute.size(tTR_rAcc),tTR_rAcc.layout)
                                    cute.copy(tiled_copy_ref_1,tTR_1[(None,None,None,0,head_tile+token*num_epi_subtiles)],token_acc)
                                    cute.arch.fence_view_async_tmem_load()
                                if cutlass.const_expr(head_tile==num_epi_subtiles-1):
                                    umma_pipeline_1.consumer_release(umma_cons_state_1)
                                    umma_cons_state_1.advance()
                                for head in cutlass.range_constexpr(0,subtile_n,4):
                                    h_global=head_tile*subtile_n+head
                                    for token in cutlass.range_constexpr(block_q_val):
                                        dots=_subview(row_acc,token*cute.size(tTR_rAcc)+head,4)
                                        if cutlass.const_expr(self.use_relu_trick):
                                            activated=indexer_add_packed_f32x2(dots,abs(dots),rnd=_RND_RN)
                                        else:
                                            activated=cute.max(dots,cute.full((4,),0.0,dtype=cutlass.Float32),propagate_nan=True)
                                        if cutlass.const_expr(h_global<NUM_W_IN_REG):
                                            weight=_subview(w_cache,token*NUM_W_IN_REG+h_global,4)
                                        else:
                                            weight=_subview(sw_col,token*num_heads+h_global,4)
                                            if cutlass.const_expr(self.use_relu_trick and not self.weights_prepared):weight=weight*0.5
                                        partial=_subview(row_sums,token*4,4)
                                        updated=indexer_fma_packed_f32x2(activated,weight,partial,rnd=_RND_RN)
                                        cute.make_tensor(row_sums.iterator+token*4,cute.make_layout(4)).store(updated)
                            for token in cutlass.range_constexpr(block_q_val):
                                offset=token*4
                                result=row_sums[offset]+row_sums[offset+1]+row_sums[offset+2]+row_sums[offset+3]
                                outrow=q_block_idx*block_q_val+token
                                if outrow<num_q_tokens:
                                    private_cursors[token]=publish_bank(result*scale_half,(kv_pos>=kv_start_cache[token])&(kv_pos<kv_end_cache[token]),threshold_cache[token],token,kv_pos,private_cursors[token],bank_rows,self.output_dtype)
                        else:
                            for t in cutlass.range_constexpr(block_q_val):
                                if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                                    # Keep each token's accumulator independent of
                                    # the shared ReLU zero under staged assignment.
                                    ps = cute.full((4,), 0.0, dtype=cutlass.Float16)
                                else:
                                    if cutlass.const_expr(self.native_lowp):
                                        s_acc=cute.make_rmem_tensor((2,),cutlass.Uint32)
                                        s_acc.fill(0)
                                    else:
                                        s_acc = cute.full((4,), 0.0, dtype=cutlass.Float32)
                                    if cutlass.const_expr(not self.use_relu_trick):
                                        zero4 = cute.full((4,), 0.0, dtype=cutlass.Float32)
                                for i in cutlass.range_constexpr(num_epi_subtiles):
                                    if t > 0 or i > 0:
                                        cute.copy(
                                            tiled_copy_ref_1,
                                            tTR_1[
                                                (
                                                    None,
                                                    None,
                                                    None,
                                                    0,
                                                    t * num_epi_subtiles + i,
                                                )
                                            ],
                                            tTR_rAcc,
                                        )
                                        cute.arch.fence_view_async_tmem_load()
                                    if t == block_q_val - 1 and i == num_epi_subtiles - 1:
                                        umma_pipeline_1.consumer_release(umma_cons_state_1)
                                        umma_cons_state_1.advance()
                                    reg_h_end = min(
                                        subtile_n, max(0, NUM_W_IN_REG - i * subtile_n)
                                    )
                                    if cutlass.const_expr(self.native_lowp):
                                        for h in cutlass.range_constexpr(0, reg_h_end, 4):
                                            h_g=i*subtile_n+h
                                            x4=_subview(tTR_rAcc,h,4)
                                            w4=_subview(w_cache, t*NUM_W_IN_REG+h_g,4)
                                            s_acc[0]=native_fma(self.output_dtype,native_fp8_pack_relu(self.output_dtype,x4[0],x4[1]),native_fp8_pack(self.output_dtype,w4[0],w4[1]),s_acc[0])
                                            s_acc[1]=native_fma(self.output_dtype,native_fp8_pack_relu(self.output_dtype,x4[2],x4[3]),native_fp8_pack(self.output_dtype,w4[2],w4[3]),s_acc[1])
                                    else:
                                        for h in cutlass.range_constexpr(0, reg_h_end, 4):
                                            n0 = h
                                            h_g = i * subtile_n + h
                                            if cutlass.const_expr(
                                                self.epi_dtype == cutlass.Float16
                                            ):
                                                pa = _subview(tTR_rAcc, n0, 4).to(
                                                    cutlass.Float16
                                                )
                                                pa = cute.max(pa, packed_zero)
                                                r0 = t * NUM_W_IN_REG + h_g
                                                pw = _subview(w_cache, r0, 4)
                                                ps = cute.fma(pa, pw, ps)
                                            else:
                                                if cutlass.const_expr(self.use_relu_trick):
                                                    # Doubled ReLU folds its factor of 1/2 into scale_half.
                                                    # It requires bounded inputs because x+abs(x) can overflow.
                                                    x4 = _subview(tTR_rAcc, n0, 4)
                                                    a4 = indexer_add_packed_f32x2(
                                                        x4, abs(x4), rnd=_RND_RN
                                                    )
                                                else:
                                                    a4 = cute.max(
                                                        _subview(tTR_rAcc, n0, 4),
                                                        zero4,
                                                        propagate_nan=True,
                                                    )
                                                r0 = t * NUM_W_IN_REG + h_g
                                                w4 = _subview(w_cache, r0, 4)
                                                s_acc = indexer_fma_packed_f32x2(
                                                    a4, w4, s_acc, rnd=_RND_RN
                                                )
                                    smem_h_start = max(0, NUM_W_IN_REG - i * subtile_n)
                                    if cutlass.const_expr(self.native_lowp):
                                        for h in cutlass.range_constexpr(smem_h_start, subtile_n, 4):
                                            h_g=i*subtile_n+h
                                            x4=_subview(tTR_rAcc,h,4)
                                            w4=_subview(sw_col, t*num_heads+h_g,4)
                                            s_acc[0]=native_fma(self.output_dtype,native_fp8_pack_relu(self.output_dtype,x4[0],x4[1]),native_fp8_pack(self.output_dtype,w4[0],w4[1]),s_acc[0])
                                            s_acc[1]=native_fma(self.output_dtype,native_fp8_pack_relu(self.output_dtype,x4[2],x4[3]),native_fp8_pack(self.output_dtype,w4[2],w4[3]),s_acc[1])
                                    else:
                                        for h in cutlass.range_constexpr(
                                            smem_h_start, subtile_n, 4
                                        ):
                                            n0 = h
                                            h_g = i * subtile_n + h
                                            if cutlass.const_expr(
                                                self.epi_dtype == cutlass.Float16
                                            ):
                                                pa = _subview(tTR_rAcc, n0, 4).to(
                                                    cutlass.Float16
                                                )
                                                pa = cute.max(pa, packed_zero)
                                                w_i = t * num_heads + h_g
                                                pw = _subview(sw_col, w_i, 4)
                                                ps = cute.fma(pa, pw, ps)
                                            else:
                                                if cutlass.const_expr(self.use_relu_trick):
                                                    # relu(x) = (x + |x|) / 2 (see
                                                    # reg-path above).
                                                    x4 = _subview(tTR_rAcc, n0, 4)
                                                    a4 = indexer_add_packed_f32x2(
                                                        x4, abs(x4), rnd=_RND_RN
                                                    )
                                                else:
                                                    a4 = cute.max(
                                                        _subview(tTR_rAcc, n0, 4),
                                                        zero4,
                                                        propagate_nan=True,
                                                    )
                                                w_i = t * num_heads + h_g
                                                w4 = _subview(sw_col, w_i, 4).to(cutlass.Float32)
                                                if cutlass.const_expr(self.use_relu_trick and not self.weights_prepared):
                                                    w4 = w4 * 0.5
                                                s_acc = indexer_fma_packed_f32x2(
                                                    a4, w4, s_acc, rnd=_RND_RN
                                                )

                                if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                                    result_t = (ps[0] + ps[2]) + (ps[1] + ps[3])
                                else:
                                    if cutlass.const_expr(self.native_lowp):
                                        reduced=native_add(self.output_dtype,s_acc[0],s_acc[1])
                                        result_t=cutlass.Float32(self.output_dtype(native_lo(self.output_dtype,reduced)+native_hi(self.output_dtype,reduced)))
                                    else:
                                        result_t = s_acc[0] + s_acc[1] + s_acc[2] + s_acc[3]

                                if cutlass.const_expr(self.mode == "filter"):
                                    outrow=q_block_idx*block_q_val+t
                                    if outrow<num_q_tokens:
                                        private_cursors[t]=publish_bank(result_t*scale_half,(kv_pos>=kv_start_cache[t])&(kv_pos<kv_end_cache[t]),threshold_cache[t],t,kv_pos,private_cursors[t],bank_rows,self.output_dtype)
                                else:
                                    emit(result_t * scale_half, q_block_idx * block_q_val + t, kv_pos, sample_pos, (kv_pos >= kv_start_cache[t]) & (kv_pos < kv_end_cache[t]), num_q_tokens, mLogits, thresholds, counts, candidates, self.mode, self.output_dtype, self.packed, self.capacity,task_partition%8,recovery_sample_info,self.top_k,threshold_cache[t])

                        kv_idx = kv_idx + NUM_MATH_WG * self.splits
                        kv_go = kv_idx < num_kv
                    if cutlass.const_expr(self.mode == "filter"):
                        for token in cutlass.range_constexpr(block_q_val):
                            row=q_block_idx*block_q_val+token
                            if row<num_q_tokens and cute.arch.lane_idx()==0:
                                candidates[row,warpgroup_idx*4+local_tidx//32]=cutlass.Int64(private_cursors[token])
                    if cutlass.const_expr(self.mode == 'sample' and self.persistent_sample):
                        subset_barrier(barrier_id=12,number_of_threads=256)
                        for token in cutlass.range_constexpr(block_q_val):
                            sample_row=q_block_idx*block_q_val+token
                            if sample_row<num_q_tokens:
                                if tidx==0:counts[sample_row]=0
                                width=max(1,mCuSeqLenKEnd[sample_row]-mCuSeqLenKStart[sample_row])
                                active_samples=block_samples(sample_row//self.block_q,mCuSeqLenKEnd,num_q_tokens,self.block_q,self.top_k, sample_factor=16)
                                multiple=2 if active_samples==512 else 6
                                rank=min(active_samples,max(1,(multiple*self.top_k*active_samples+width-1)//width))
                                propose(sample_row,tidx,mLogits,sample_hist,sample_scalar,thresholds,active_samples,rank)
                        subset_barrier(barrier_id=12,number_of_threads=256)

                    next_q_block_idx = q_block_idx + cute.arch.grid_dim()[2] // self.splits
                    if next_q_block_idx < num_q_blocks:
                        q_last_next = min(
                            next_q_block_idx * block_q_val + block_q_val - 1,
                            num_q_tokens - 1,
                        )
                        next_num_kv = (runtime_samples(mCuSeqLenKEnd[q_last_next],self.top_k, sample_factor=16)//128 if cutlass.const_expr(self.mode=="sample") else ((mCuSeqLenKEnd[q_last_next] + block_kv_val - 1) // block_kv_val))
                    has_work = next_q_block_idx < num_q_blocks

                if q_block_idx < num_q_blocks:
                    q_pipeline.consumer_release(q_cons_state)
                    q_cons_state.advance()

            # Retire once at the full CTA boundary below.

        else:
            if cutlass.const_expr(self.mode == "filter"):
                if cutlass.const_expr(self.store_registers >= 128):
                    cute.arch.warpgroup_reg_alloc(self.store_registers)
                else:
                    cute.arch.warpgroup_reg_dealloc(self.store_registers)
            else:
                cute.arch.warpgroup_reg_dealloc(self.store_registers)




        subset_barrier(barrier_id=14,number_of_threads=384)
        subset_barrier(barrier_id=14,number_of_threads=384)

from .recovery_selection import finish_recovery, select_or_prepare_retry as select_candidates

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

from .fused_recovery_emit import emit,finish_recovery

from .fused_recovery_emit import prepare_threshold

from .native_lowp_math import native_pack, native_pack_relu, native_fma, native_add, native_lo, native_hi

from .native_lowp_math import native_fp8_pack, native_fp8_pack_relu

from .native_heads import native_weight_pair
