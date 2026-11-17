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
from .qsa_publish import publish as fused_publish, pack_half_index
from .weighted_heads import weighted_heads, weighted_heads_cached
from .half_filter_boundary import filter_boundary

from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm

@dsl_user_op
def predicated_store64(ptr,value,predicate,*,loc=None,ip=None):
    llvm.inline_asm(None,[ptr.toint(loc=loc,ip=ip).ir_value(loc=loc,ip=ip),
        cutlass.Int64(value).ir_value(loc=loc,ip=ip),cutlass.Int32(predicate).ir_value(loc=loc,ip=ip)],
        "{ .reg .pred p; setp.ne.u32 p, $2, 0; @p st.global.b64 [$0], $1; }",
        "l,l,r,~{memory}",has_side_effects=True,loc=loc,ip=ip)
@dsl_user_op
def predicated_store32(ptr,value,predicate,*,loc=None,ip=None):
    llvm.inline_asm(None,[ptr.toint(loc=loc,ip=ip).ir_value(loc=loc,ip=ip),
        cutlass.Int32(value).ir_value(loc=loc,ip=ip),cutlass.Int32(predicate).ir_value(loc=loc,ip=ip)],
        "{ .reg .pred p; setp.ne.u32 p, $2, 0; @p st.global.b32 [$0], $1; }",
        "l,r,r,~{memory}",has_side_effects=True,loc=loc,ip=ip)

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

from .integer_relu_fast import integer_relu4

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
        offsets32: bool = False,
        dsv4_direct_sync: bool = False,
        dsv4_full_tiles: bool = False,
        dsv4_aligned_k: bool = False,
        dsv4_unordered: bool = False,
        dsv4_cast: bool = False,
        pro_packed_store: bool = False,
        dsv4_alias_kv: bool = False,
        dsv4_integer_period: int = 0,
        qsa_weights: int = 0,
        qsa_unit_scale: bool = False,
        persistent_sample: bool = True,
        full_q_blocks: bool = False,
        glm_skip_empty: bool = True,
        glm_integer_period: int = 0,
        glm_single_q_stage: bool = False,
        glm_two_math_wg: bool = False,
        glm_pair_queries: bool = False,
        q_stage_override: int = 0,
        pdl: bool = False,
        native_bf16: bool = False,
        native_fp16: bool = False,
    ):
        self.pdl = bool(pdl)
        # Opt-in BF16 head math; the default FP32 epilogue is retained.
        self.native_lowp = bool(native_bf16 or native_fp16)
        assert not (native_bf16 and native_fp16)
        assert not native_fp16 or output_dtype == cutlass.Float16
        assert not self.native_lowp or (num_heads in (4, 32, 64) and output_dtype == (cutlass.Float16 if native_fp16 else cutlass.BFloat16))
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
        assert mode=="filter" and (umma_n==256 or num_heads in (4,32)) and epi_dtype==cutlass.Float32 and (splits==1 or (num_heads==32 and splits==2))
        self.mode, self.top_k, self.capacity = mode, top_k, capacity
        self.splits, self.packed = splits, packed
        self.offsets32 = offsets32
        self.dsv4_direct_sync = dsv4_direct_sync
        self.dsv4_full_tiles = dsv4_full_tiles
        self.dsv4_aligned_k = dsv4_aligned_k
        self.dsv4_unordered = dsv4_unordered
        self.dsv4_cast = dsv4_cast
        self.pro_packed_store = pro_packed_store
        self.dsv4_alias_kv = dsv4_alias_kv
        # A nonzero period routes fixed four-head groups through the integer pipe.
        # The raw E4M3FN D128 dot is bounded before scales.
        assert dsv4_integer_period in (0, 32, 64)
        assert not dsv4_integer_period or (num_heads == 64 and head_dim == 128)
        self.dsv4_integer_head_mask = sum(
            1 << h for h in range(64)
            if dsv4_integer_period and h % dsv4_integer_period < 4
        )
        self.qsa_weights = qsa_weights
        self.qsa_unit_scale = qsa_unit_scale
        self.persistent_sample = persistent_sample
        self.weights_prepared = weights_prepared
        self.full_q_blocks = full_q_blocks
        self.glm_skip_empty = glm_skip_empty
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
        # Pair Q columns before MMA and load 32 FP32 values per epilogue step.
        # The paired layout avoids spills while preserving four chains per query.
        self.interleaved_queries = (num_heads == 32 and top_k == 512
            and umma_n == 256 and not glm_two_math_wg and glm_integer_period == 0
            and weights_prepared and use_relu_trick)
        if self.interleaved_queries:
            num_epi_subtiles = 1
        self.num_epi_subtiles = num_epi_subtiles
        self.epi_dtype = epi_dtype
        self.weight_dtype = weight_dtype
        self.packed_bf16_weights = weight_dtype == cutlass.BFloat16 and native_bf16
        self.weight_bytes = 2 if weight_dtype == cutlass.BFloat16 else 4
        # The optional ReLU identity (x+abs(x))/2 can overflow for x >= 2**127.
        # Enable it only when input scales and weights keep the intermediate finite.
        self.use_relu_trick = use_relu_trick
        self.epi_bytes = 2 if epi_dtype == cutlass.Float16 else 4
        w_stage_bytes = self.N * self.weight_bytes
        self.w_stage_stride = ((w_stage_bytes + 127) // 128 * 128) // self.weight_bytes
        self.output_dtype = output_dtype
        self.glm_integer_period = glm_integer_period
        if num_epi_subtiles > 1 and num_heads % num_epi_subtiles != 0:
            raise ValueError("num_heads must be divisible by num_epi_subtiles")
        if (num_heads // num_epi_subtiles) % 4 != 0:
            raise ValueError("num_heads // num_epi_subtiles must be divisible by 4")
        self.num_groups = 1

        self.glm_pair_queries=glm_pair_queries
        assert not glm_pair_queries or (num_heads==32 and umma_n==256 and not glm_two_math_wg and glm_integer_period==0)
        self.glm_two_math_wg = glm_two_math_wg
        assert not glm_two_math_wg or (num_heads==32 and umma_n==128)
        self.num_math_threads = 256 if glm_two_math_wg else 512
        self.num_specialized_threads = 128
        self.threads_per_cta = self.num_math_threads+128
        self.num_math_warps = self.num_math_threads//32
        self.tma_warp_base = 8
        self.umma_warp_base = 10

        # Reload Q when the Q block changes; reuse it across KV blocks.
        self.num_q_stages = 2 if mode == "filter" else 3
        if num_heads==32 and glm_single_q_stage and (top_k==512 or (top_k==2048 and output_dtype in (cutlass.Float16,cutlass.BFloat16))):self.num_q_stages=1

        if q_stage_override:self.num_q_stages=q_stage_override
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
            self.num_kv_stages = 4

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
        num_q_tokens: cutlass.Int32,
        num_kv_tokens: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        # TMA descriptors use true operand extents. Hardware zero-fills ragged
        # tiles while completing the full transaction byte count.
        b = cute.recast_tensor(b, cutlass.Float8E4M3FN)
        if cutlass.const_expr(self.interleaved_queries):
            # TMA reads adjacent queries in alternating columns, without a copy.
            b = cute.make_tensor(
                b.iterator,
                cute.make_layout(
                    ((2, 32, b.shape[0] // 64), b.shape[1]),
                    stride=((32 * b.stride[0], b.stride[0], 64 * b.stride[0]),
                            b.stride[1]),
                ),
            )
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
        self.num_kv_scale_tma_bytes = self.block_kv * self.head_dim
        self.num_q_tma_bytes = b_copy_size * atom_thr_size + w_copy_size

        num_q_blocks = (num_q_tokens + self.block_q - 1) // self.block_q
        num_ctas = num_q_blocks * self.splits
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
            kv_scales,
            logits,
            cu_seq_len_k_start,
            cu_seq_len_k_end,
            thresholds, counts, candidates, output,
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
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=(2 if self.glm_two_math_wg else 1) if self.num_heads==32 else 0,
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
        mS_raw: cute.Tensor,
        mLogits: cute.Tensor,  # [num_q_tokens, logits_stride]
        mCuSeqLenKStart: cute.Tensor,  # [num_q_tokens] int32
        mCuSeqLenKEnd: cute.Tensor,  # [num_q_tokens] int32
        thresholds: cute.Tensor,
        counts: cute.Tensor,
        candidates: cute.Tensor,
        output: cute.Tensor,
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
        tid = cute.arch.thread_idx()[0]
        bid = cute.arch.block_idx()[2]
        row_base = (bid // self.splits) * self.block_q
        row = cutlass.Int32(0)
        required = cutlass.Int32(0)
        smem = indexer_memory.SmemAllocator()
        hist = smem.allocate_tensor(cutlass.Int32, cute.make_layout(1538), byte_alignment=128)
        scalar = smem.allocate_tensor(cutlass.Int32, cute.make_layout(2), byte_alignment=16)
        if cutlass.const_expr(self.mode == 'sample' and not self.persistent_sample):
            for offset in cutlass.range_constexpr(self.block_q):
                row = row_base + offset
                if row < num_q_tokens:
                    for j in cutlass.range(tid, self.sample_size, 384):
                        mLogits[row, j] = self.output_dtype(float('-inf'))
                    if tid == 0:
                        counts[row] = 0
                    if cutlass.const_expr(self.use_relu_trick):
                        if tid < self.num_heads:
                            prepared_offset=(num_q_tokens+31)//32*32
                            thresholds[prepared_offset+row*self.num_heads+tid]=mW_raw[row*self.num_heads+tid]*0.5
            cute.arch.barrier(barrier_id=14, number_of_threads=self.threads_per_cta)
        work = cutlass.Boolean(True)
        if cutlass.const_expr(self.mode == 'recover'):
            work = cutlass.Boolean(False)
            for offset in cutlass.range_constexpr(self.block_q):
                row = row_base + offset
                if row < num_q_tokens:
                    required = min(self.top_k, mCuSeqLenKEnd[row] - mCuSeqLenKStart[row])
                    work |= counts[row] < required
        if work:
            self._gemm_body(tiled_mma, tma_atom_a, mA_mkl, tma_atom_b, mB_nkl, tma_atom_w, mW_tma, mW_raw, tma_atom_s, mS_tma, mS_raw, mLogits, mCuSeqLenKStart, mCuSeqLenKEnd, thresholds, counts, candidates, output, num_q_tokens, num_q_blocks, num_kv_tokens, cluster_layout_vmnk, a_smem_layout_staged, a_tma_smem_layout, b_smem_layout_staged, w_smem_layout_staged, s_smem_layout_staged, epi_tile, SharedStorage)
        if cutlass.const_expr(self.mode in ('sample', 'recover')):
            cute.arch.barrier(barrier_id=14, number_of_threads=self.threads_per_cta)
            if tid < 256:
                for offset in cutlass.range_constexpr(self.block_q):
                    row = row_base + offset
                    if row < num_q_tokens:
                        start = mCuSeqLenKStart[row]
                        end = mCuSeqLenKEnd[row]
                        required = min(self.top_k, end - start)
                        if cutlass.const_expr(self.mode == 'sample' and not self.persistent_sample):
                            active_samples=block_samples(row//self.block_q,mCuSeqLenKEnd,num_q_tokens,self.block_q,self.top_k, sample_factor=16)
                            multiple=2 if active_samples==512 else 6
                            rank = min(active_samples, max(1, (multiple * self.top_k * active_samples + max(1, end - start) - 1) // max(1, end - start)))
                            propose(row, tid, mLogits, hist, scalar, thresholds, active_samples, rank)
                        elif cutlass.const_expr(self.mode != 'sample'):
                            if counts[row] < required:
                                finish_recovery(row, tid, candidates, counts, mCuSeqLenKStart, mCuSeqLenKEnd, mLogits, hist, scalar, output, thresholds, self.top_k, self.capacity, self.output_dtype, self.packed)

    @cute.jit
    def _gemm_body(self, tiled_mma: cute.TiledMma, tma_atom_a: cute.CopyAtom, mA_mkl: cute.Tensor, tma_atom_b: cute.CopyAtom, mB_nkl: cute.Tensor, tma_atom_w: cute.CopyAtom, mW_tma: cute.Tensor, mW_raw: cute.Tensor, tma_atom_s: cute.CopyAtom, mS_tma: cute.Tensor, mS_raw: cute.Tensor, mLogits: cute.Tensor, mCuSeqLenKStart: cute.Tensor, mCuSeqLenKEnd: cute.Tensor, thresholds: cute.Tensor, counts: cute.Tensor, candidates: cute.Tensor, output: cute.Tensor, num_q_tokens: cutlass.Int32, num_q_blocks: cutlass.Int32, num_kv_tokens: cutlass.Int32, cluster_layout_vmnk: cute.Layout, a_smem_layout_staged: cute.ComposedLayout, a_tma_smem_layout: cute.ComposedLayout, b_smem_layout_staged: cute.ComposedLayout, w_smem_layout_staged: cute.Layout, s_smem_layout_staged: cute.Layout, epi_tile: cute.Tile, SharedStorage: cutlass.Constexpr):
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
        is_math_warp = warp_idx < self.num_math_warps
        is_tma_warp_0 = warp_idx == self.num_math_warps
        is_tma_warp_1 = warp_idx == self.num_math_warps+1
        is_tma_warp = is_tma_warp_0 | is_tma_warp_1
        is_umma_warp_0 = warp_idx == self.num_math_warps+2
        is_umma_warp_1 = warp_idx == self.num_math_warps+3

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
            qb=max(cutlass.Int32(0),qb)
            ks_min = _ks[min(qb * _bq, _nqt - 1)]
            for t_i in cutlass.range_constexpr(1, _bq):
                ks_min = min(ks_min, _ks[min(qb * _bq + t_i, _nqt - 1)])
            return (ks_min // _split) * _nwg + _part * _nwg

        # Early load of num_kv for the first assigned Q block (hide L2 latency)
        start_q_block = num_q_blocks-1-sm_idx
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

        q_cons_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_math_warps)
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
            pipeline.Agent.Thread, 1
        )
        kv_pipeline_0 = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.kv_mbar_0.data_ptr(),
            num_stages=self.num_kv_stages,
            producer_group=prod_group,
            consumer_group=kv_cons_group,
            tx_count=self.num_kv_scale_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        kv_pipeline_1 = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.kv_mbar_1.data_ptr(),
            num_stages=self.num_kv_stages,
            producer_group=prod_group,
            consumer_group=kv_cons_group,
            tx_count=self.num_kv_scale_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
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

        tmem_alloc_num_threads = self.num_math_threads+32
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
        if cutlass.const_expr(self.num_heads in (4,32) or self.dsv4_alias_kv):
            sKV_1 = sKV_0
        else:
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
        if cutlass.const_expr(self.num_heads in (4,32) or self.dsv4_alias_kv):
            sScales_1 = sScales_0
        else:
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
        epi_sub_mn = (epi_tile[0], 16 if self.num_heads==4 and self.N==128 else num_heads // num_epi_subtiles)
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
        q_block_idx = cutlass.Int32(num_q_blocks-1-sm_idx)
        kv_idx = cutlass.Int32(current_kv_lo)
        num_kv = current_num_kv
        # Sentinel: no previous Q block (forces initial Q load)
        prev_q_block_idx = num_q_blocks  # out-of-range sentinel

        next_q_block_idx = q_block_idx
        next_kv_idx = kv_idx
        next_num_kv = num_kv
        has_work = q_block_idx >= 0

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # ===== WARP-SPECIALIZED EXECUTION =====

        # H64 native math needs 112 registers: producers release to 32 first.
        # 512 * 112 + 128 * 32 fits the SM register file without spills.
        if cutlass.const_expr(self.num_heads==32 or self.native_lowp):
            if not is_math_warp:cute.arch.warpgroup_reg_dealloc(32)
        if is_tma_warp_0:
            # TMA warp 0: loads Q+Weights (on Q-block change) + KV group 0

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
                    prefetch_next_q = q_block_idx - cute.arch.grid_dim()[2] // self.splits
                    if cutlass.const_expr(self.num_q_stages==1):
                        # One buffer cannot prefetch the next Q before current KV:
                        # consumers release this buffer only after current KV is done.
                        # First Q was loaded above; later Q blocks load on demand.
                        prefetch_next_q=q_block_idx if q_block_idx_old<num_q_blocks else cutlass.Int32(-1)
                    if prefetch_next_q >= 0:
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
                kv_prod_state_0.advance()

                # Scheduler advance
                next_kv_idx = kv_idx + NUM_MATH_WG * self.splits
                if next_kv_idx >= num_kv:
                    next_q_block_idx = q_block_idx - cute.arch.grid_dim()[2] // self.splits  # grid-stride
                    next_kv_idx = _kv_lo_for_block(next_q_block_idx)
                    if next_q_block_idx >= 0:
                        q_last_next = min(
                            next_q_block_idx * block_q_val + block_q_val - 1,
                            num_q_tokens - 1,
                        )
                        next_num_kv = (runtime_samples(mCuSeqLenKEnd[q_last_next],self.top_k, sample_factor=16)//128 if cutlass.const_expr(self.mode=="sample") else ((mCuSeqLenKEnd[q_last_next] + block_kv_val - 1) // block_kv_val))
                has_work = next_q_block_idx >= 0

        elif is_umma_warp_0:
            subset_barrier(barrier_id=1,number_of_threads=self.num_math_threads+32)
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
                    kv_pipeline_0.consumer_release(kv_cons_state_umma_0)
                    kv_cons_state_umma_0.advance()
                    umma_pipeline_0.producer_commit(umma_prod_state_0)
                    umma_prod_state_0.advance()

                    next_kv_idx = kv_idx + NUM_MATH_WG * self.splits
                    if next_kv_idx >= num_kv:
                        next_q_block_idx = q_block_idx - cute.arch.grid_dim()[2] // self.splits
                        next_kv_idx = _kv_lo_for_block(next_q_block_idx)
                        if next_q_block_idx >= 0:
                            q_last_next = min(
                                next_q_block_idx * block_q_val + block_q_val - 1,
                                num_q_tokens - 1,
                            )
                            next_num_kv = (runtime_samples(mCuSeqLenKEnd[q_last_next],self.top_k, sample_factor=16)//128 if cutlass.const_expr(self.mode=="sample") else ((mCuSeqLenKEnd[q_last_next] + block_kv_val - 1) // block_kv_val))
                    has_work = next_q_block_idx >= 0

        elif is_math_warp:
            if cutlass.const_expr(self.num_heads==32 or self.native_lowp):cute.arch.warpgroup_reg_alloc(104 if self.glm_two_math_wg else 112)
            tmem.allocate(num_tmem_alloc_cols_total)
            subset_barrier(barrier_id=1,number_of_threads=self.num_math_threads+32)
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            if cutlass.const_expr(self.num_heads in (4,32)):tmem.relinquish_alloc_permit()
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
            w_cache = cute.make_rmem_tensor((64,),cutlass.Float32)
            zero4 = cute.full((4,),0.0,dtype=cutlass.Float32)
            if cutlass.const_expr(self.num_heads == 64):
                while has_work:
                    q_block_idx = next_q_block_idx
                    # Retire the producer-issued dummy even for an empty query block.
                    num_kv = next_num_kv
                    q_pipeline.consumer_wait(q_cons_state)
                    q_stage_local = q_cons_state.index
                    outrow = q_block_idx*self.block_q+query_slot
                    token_idx = outrow if cutlass.const_expr(self.full_q_blocks) else min(outrow,num_q_tokens-1)
                    kv_start = mCuSeqLenKStart[token_idx]
                    kv_end = mCuSeqLenKEnd[token_idx]
                    tau = filter_boundary(thresholds[token_idx],self.output_dtype) if cutlass.const_expr(self.dsv4_cast) else thresholds[token_idx]
                    # A NaN threshold conservatively accepts the whole interval.
                    # K3/K4 still select exact top-k, including overflow recovery.
                    if cutlass.const_expr(self.dsv4_unordered):
                        if tau!=tau:tau=cutlass.Float32(float('-inf'))
                    if cutlass.const_expr(self.native_lowp):
                        native_weights=cute.make_rmem_tensor((32,),cutlass.Uint32)
                        for h in cutlass.range_constexpr(32):
                            wa=cutlass.Float32(sW[(query_slot*64+h*2,q_stage_local)])
                            wb=cutlass.Float32(sW[(query_slot*64+h*2+1,q_stage_local)])
                            if cutlass.const_expr(self.use_relu_trick and self.weights_prepared):
                                wa=wa*2.0
                                wb=wb*2.0
                            native_weights[h]=native_weight_pair(sW,query_slot*64+h*2,q_stage_local,self.output_dtype,fp8=True)
                    else:
                        for h in cutlass.range_constexpr(64):
                            w = cutlass.Float32(sW[(query_slot*64+h,q_stage_local)])
                            if cutlass.const_expr(self.use_relu_trick and not self.weights_prepared):w = w*0.5
                            if cutlass.const_expr(self.use_relu_trick and ((self.dsv4_integer_head_mask >> h) & 1)):
                                w = w * 2.0
                            w_cache[h] = w
                    bank_span = (candidates.shape[1]-8)//4
                    bank_id = local_tidx//32
                    bank_rows = cute.make_tensor(candidates.iterator+cutlass.Int64(outrow)*candidates.stride[0]+8+bank_id*bank_span,cute.make_layout((1,bank_span),stride=(bank_span,1)))
                    cursor_even = cutlass.Int32(0)
                    kv_idx = _kv_lo_for_block(q_block_idx)
                    # Retire empty producer tiles after Q/weight setup so the
                    # empty-range branch does not affect scoring-loop scheduling.
                    if kv_idx >= num_kv:
                        if cutlass.const_expr(self.num_heads == 64 and self.dsv4_direct_sync):
                            cute.arch.mbarrier_wait(storage.umma_mbar_0.data_ptr()+umma_cons_state_0.index,umma_cons_state_0.phase)
                            cute.arch.mbarrier_arrive(storage.umma_mbar_0.data_ptr()+self.num_umma_stages+umma_cons_state_0.index)
                        else:
                            umma_pipeline_0.consumer_wait(umma_cons_state_0)
                            umma_pipeline_0.consumer_release(umma_cons_state_0)
                        umma_cons_state_0.advance()
                    # Every producer and both math WGs traverse the same KV tiles.
                    if (kv_idx%2!=0)&(kv_idx<num_kv):
                        kv_pos = kv_idx*128+m_coord
                        if cutlass.const_expr(self.dsv4_direct_sync):
                            scale_val=mS_raw[kv_pos if cutlass.const_expr(self.dsv4_aligned_k) else min(kv_pos,num_kv_tokens-1)]
                            cute.arch.mbarrier_wait(storage.umma_mbar_0.data_ptr()+umma_cons_state_0.index,umma_cons_state_0.phase)
                        else:
                            umma_ready=umma_pipeline_0.consumer_try_wait(umma_cons_state_0)
                            scale_val=mS_raw[kv_pos if cutlass.const_expr(self.dsv4_aligned_k) else min(kv_pos,num_kv_tokens-1)]
                            umma_pipeline_0.consumer_wait(umma_cons_state_0,umma_ready)
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
                                if cutlass.const_expr(self.dsv4_direct_sync):
                                    cute.arch.mbarrier_arrive(storage.umma_mbar_0.data_ptr()+self.num_umma_stages+umma_cons_state_0.index)
                                else:
                                    umma_pipeline_0.consumer_release(umma_cons_state_0)
                                umma_cons_state_0.advance()
                            if cutlass.const_expr(self.native_lowp):
                                for h in cutlass.range_constexpr(0,64//num_epi_subtiles,4):
                                    x4=_subview(tTR_rAcc,h,4)
                                    woff=(i*(64//num_epi_subtiles)+h)//2
                                    s_acc[0]=native_fma(self.output_dtype,native_fp8_pack_relu(self.output_dtype,x4[0],x4[1]),native_weights[woff],s_acc[0])
                                    s_acc[1]=native_fma(self.output_dtype,native_fp8_pack_relu(self.output_dtype,x4[2],x4[3]),native_weights[woff+1],s_acc[1])
                            else:
                                for h in cutlass.range_constexpr(0,64//num_epi_subtiles,4):
                                    x4 = _subview(tTR_rAcc,h,4)
                                    if cutlass.const_expr(self.use_relu_trick):
                                        if cutlass.const_expr((self.dsv4_integer_head_mask >> (i*(64//num_epi_subtiles)+h)) & 1):
                                            a4 = integer_relu4(x4)
                                        else:
                                            a4 = indexer_add_packed_f32x2(x4,abs(x4),rnd=_RND_RN)
                                    else:
                                        a4 = cute.max(x4,zero4,propagate_nan=True)
                                    w4 = _subview(w_cache,i*(64//num_epi_subtiles)+h,4)
                                    s_acc = indexer_fma_packed_f32x2(a4,w4,s_acc,rnd=_RND_RN)
                        if cutlass.const_expr(self.native_lowp):
                            reduced=native_add(self.output_dtype,s_acc[0],s_acc[1])
                            score=native_sum_fp32(self.output_dtype,reduced)*scale_val
                        else:
                            score = (s_acc[0]+s_acc[1]+s_acc[2]+s_acc[3])*scale_val
                        if cutlass.const_expr(self.full_q_blocks) or outrow < num_q_tokens:
                            valid=(kv_pos>=kv_start)&(kv_pos<kv_end)
                            if kv_idx%2==0:
                                cursor_even=publish_bank(score,valid,tau,0,kv_pos,cursor_even,bank_rows,self.output_dtype,pred_store=(self.pro_packed_store and bank_rows.element_type==cutlass.Int32),raw_half=self.dsv4_cast,unordered=self.dsv4_unordered,pro_raw_packed=(self.pro_packed_store and bank_rows.element_type==cutlass.Int32))
                            else:
                                cursor_even=publish_bank(score,valid,tau,0,kv_pos,cursor_even,bank_rows,self.output_dtype,pred_store=(self.pro_packed_store and bank_rows.element_type==cutlass.Int32),raw_half=self.dsv4_cast,unordered=self.dsv4_unordered,pro_raw_packed=(self.pro_packed_store and bank_rows.element_type==cutlass.Int32))
                        kv_idx=kv_idx+1
                    if cutlass.const_expr(self.dsv4_full_tiles):
                        while (kv_idx < num_kv)&(kv_idx*128<kv_start):
                            kv_pos = kv_idx*128+m_coord
                            if cutlass.const_expr(self.dsv4_direct_sync):
                                scale_val=mS_raw[kv_pos if cutlass.const_expr(self.dsv4_aligned_k) else min(kv_pos,num_kv_tokens-1)]
                                cute.arch.mbarrier_wait(storage.umma_mbar_0.data_ptr()+umma_cons_state_0.index,umma_cons_state_0.phase)
                            else:
                                umma_ready=umma_pipeline_0.consumer_try_wait(umma_cons_state_0)
                                scale_val=mS_raw[kv_pos if cutlass.const_expr(self.dsv4_aligned_k) else min(kv_pos,num_kv_tokens-1)]
                                umma_pipeline_0.consumer_wait(umma_cons_state_0,umma_ready)
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
                                    if cutlass.const_expr(self.dsv4_direct_sync):
                                        cute.arch.mbarrier_arrive(storage.umma_mbar_0.data_ptr()+self.num_umma_stages+umma_cons_state_0.index)
                                    else:
                                        umma_pipeline_0.consumer_release(umma_cons_state_0)
                                    umma_cons_state_0.advance()
                                if cutlass.const_expr(self.native_lowp):
                                    for h in cutlass.range_constexpr(0,64//num_epi_subtiles,4):
                                        x4=_subview(tTR_rAcc,h,4)
                                        woff=(i*(64//num_epi_subtiles)+h)//2
                                        s_acc[0]=native_fma(self.output_dtype,native_fp8_pack_relu(self.output_dtype,x4[0],x4[1]),native_weights[woff],s_acc[0])
                                        s_acc[1]=native_fma(self.output_dtype,native_fp8_pack_relu(self.output_dtype,x4[2],x4[3]),native_weights[woff+1],s_acc[1])
                                else:
                                    for h in cutlass.range_constexpr(0,64//num_epi_subtiles,4):
                                        x4 = _subview(tTR_rAcc,h,4)
                                        if cutlass.const_expr(self.use_relu_trick):
                                            if cutlass.const_expr((self.dsv4_integer_head_mask >> (i*(64//num_epi_subtiles)+h)) & 1):
                                                a4 = integer_relu4(x4)
                                            else:
                                                a4 = indexer_add_packed_f32x2(x4,abs(x4),rnd=_RND_RN)
                                        else:
                                            a4 = cute.max(x4,zero4,propagate_nan=True)
                                        w4 = _subview(w_cache,i*(64//num_epi_subtiles)+h,4)
                                        s_acc = indexer_fma_packed_f32x2(a4,w4,s_acc,rnd=_RND_RN)
                            if cutlass.const_expr(self.native_lowp):
                                reduced=native_add(self.output_dtype,s_acc[0],s_acc[1])
                                score=native_sum_fp32(self.output_dtype,reduced)*scale_val
                            else:
                                score = (s_acc[0]+s_acc[1]+s_acc[2]+s_acc[3])*scale_val
                            if cutlass.const_expr(self.full_q_blocks) or outrow < num_q_tokens:
                                valid=(kv_pos>=kv_start)&(kv_pos<kv_end)
                                if kv_idx%2==0:
                                    cursor_even=publish_bank(score,valid,tau,0,kv_pos,cursor_even,bank_rows,self.output_dtype,pred_store=(self.pro_packed_store and bank_rows.element_type==cutlass.Int32),raw_half=self.dsv4_cast,unordered=self.dsv4_unordered,pro_raw_packed=(self.pro_packed_store and bank_rows.element_type==cutlass.Int32))
                                else:
                                    cursor_even=publish_bank(score,valid,tau,0,kv_pos,cursor_even,bank_rows,self.output_dtype,pred_store=(self.pro_packed_store and bank_rows.element_type==cutlass.Int32),raw_half=self.dsv4_cast,unordered=self.dsv4_unordered,pro_raw_packed=(self.pro_packed_store and bank_rows.element_type==cutlass.Int32))
                            kv_idx=kv_idx+1
                    while kv_idx+(15 if self.native_lowp else 7)<(min(num_kv,kv_end//128) if cutlass.const_expr(self.dsv4_full_tiles) else num_kv):
                        for kv_unroll in cutlass.range_constexpr(16 if self.native_lowp else 8):
                            kv_pos = kv_idx*128+m_coord
                            if cutlass.const_expr(self.dsv4_direct_sync):
                                scale_val=mS_raw[kv_pos if cutlass.const_expr(self.dsv4_aligned_k) else min(kv_pos,num_kv_tokens-1)]
                                cute.arch.mbarrier_wait(storage.umma_mbar_0.data_ptr()+umma_cons_state_0.index,umma_cons_state_0.phase)
                            else:
                                umma_ready=umma_pipeline_0.consumer_try_wait(umma_cons_state_0)
                                scale_val=mS_raw[kv_pos if cutlass.const_expr(self.dsv4_aligned_k) else min(kv_pos,num_kv_tokens-1)]
                                umma_pipeline_0.consumer_wait(umma_cons_state_0,umma_ready)
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
                                    if cutlass.const_expr(self.dsv4_direct_sync):
                                        cute.arch.mbarrier_arrive(storage.umma_mbar_0.data_ptr()+self.num_umma_stages+umma_cons_state_0.index)
                                    else:
                                        umma_pipeline_0.consumer_release(umma_cons_state_0)
                                    umma_cons_state_0.advance()
                                if cutlass.const_expr(self.native_lowp):
                                    for h in cutlass.range_constexpr(0,64//num_epi_subtiles,4):
                                        x4=_subview(tTR_rAcc,h,4)
                                        woff=(i*(64//num_epi_subtiles)+h)//2
                                        s_acc[0]=native_fma(self.output_dtype,native_fp8_pack_relu(self.output_dtype,x4[0],x4[1]),native_weights[woff],s_acc[0])
                                        s_acc[1]=native_fma(self.output_dtype,native_fp8_pack_relu(self.output_dtype,x4[2],x4[3]),native_weights[woff+1],s_acc[1])
                                else:
                                    for h in cutlass.range_constexpr(0,64//num_epi_subtiles,4):
                                        x4 = _subview(tTR_rAcc,h,4)
                                        if cutlass.const_expr(self.use_relu_trick):
                                            if cutlass.const_expr((self.dsv4_integer_head_mask >> (i*(64//num_epi_subtiles)+h)) & 1):
                                                a4 = integer_relu4(x4)
                                            else:
                                                a4 = indexer_add_packed_f32x2(x4,abs(x4),rnd=_RND_RN)
                                        else:
                                            a4 = cute.max(x4,zero4,propagate_nan=True)
                                        w4 = _subview(w_cache,i*(64//num_epi_subtiles)+h,4)
                                        s_acc = indexer_fma_packed_f32x2(a4,w4,s_acc,rnd=_RND_RN)
                            if cutlass.const_expr(self.native_lowp):
                                reduced=native_add(self.output_dtype,s_acc[0],s_acc[1])
                                score=native_sum_fp32(self.output_dtype,reduced)*scale_val
                            else:
                                score = (s_acc[0]+s_acc[1]+s_acc[2]+s_acc[3])*scale_val
                            if cutlass.const_expr(self.full_q_blocks) or outrow < num_q_tokens:
                                valid=True if cutlass.const_expr(self.dsv4_full_tiles) else (kv_pos>=kv_start)&(kv_pos<kv_end)
                                if cutlass.const_expr(kv_unroll%2==0):
                                    cursor_even=publish_bank(score,valid,tau,0,kv_pos,cursor_even,bank_rows,self.output_dtype,pred_store=(self.pro_packed_store and bank_rows.element_type==cutlass.Int32),raw_half=self.dsv4_cast,unordered=self.dsv4_unordered,pro_raw_packed=(self.pro_packed_store and bank_rows.element_type==cutlass.Int32))
                                else:
                                    cursor_even=publish_bank(score,valid,tau,0,kv_pos,cursor_even,bank_rows,self.output_dtype,pred_store=(self.pro_packed_store and bank_rows.element_type==cutlass.Int32),raw_half=self.dsv4_cast,unordered=self.dsv4_unordered,pro_raw_packed=(self.pro_packed_store and bank_rows.element_type==cutlass.Int32))
                            kv_idx=kv_idx+1
                    while kv_idx < num_kv:
                        kv_pos = kv_idx*128+m_coord
                        if cutlass.const_expr(self.dsv4_direct_sync):
                            scale_val=mS_raw[kv_pos if cutlass.const_expr(self.dsv4_aligned_k) else min(kv_pos,num_kv_tokens-1)]
                            cute.arch.mbarrier_wait(storage.umma_mbar_0.data_ptr()+umma_cons_state_0.index,umma_cons_state_0.phase)
                        else:
                            umma_ready=umma_pipeline_0.consumer_try_wait(umma_cons_state_0)
                            scale_val=mS_raw[kv_pos if cutlass.const_expr(self.dsv4_aligned_k) else min(kv_pos,num_kv_tokens-1)]
                            umma_pipeline_0.consumer_wait(umma_cons_state_0,umma_ready)
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
                                if cutlass.const_expr(self.dsv4_direct_sync):
                                    cute.arch.mbarrier_arrive(storage.umma_mbar_0.data_ptr()+self.num_umma_stages+umma_cons_state_0.index)
                                else:
                                    umma_pipeline_0.consumer_release(umma_cons_state_0)
                                umma_cons_state_0.advance()
                            if cutlass.const_expr(self.native_lowp):
                                for h in cutlass.range_constexpr(0,64//num_epi_subtiles,4):
                                    x4=_subview(tTR_rAcc,h,4)
                                    woff=(i*(64//num_epi_subtiles)+h)//2
                                    s_acc[0]=native_fma(self.output_dtype,native_fp8_pack_relu(self.output_dtype,x4[0],x4[1]),native_weights[woff],s_acc[0])
                                    s_acc[1]=native_fma(self.output_dtype,native_fp8_pack_relu(self.output_dtype,x4[2],x4[3]),native_weights[woff+1],s_acc[1])
                            else:
                                for h in cutlass.range_constexpr(0,64//num_epi_subtiles,4):
                                    x4 = _subview(tTR_rAcc,h,4)
                                    if cutlass.const_expr(self.use_relu_trick):
                                        if cutlass.const_expr((self.dsv4_integer_head_mask >> (i*(64//num_epi_subtiles)+h)) & 1):
                                            a4 = integer_relu4(x4)
                                        else:
                                            a4 = indexer_add_packed_f32x2(x4,abs(x4),rnd=_RND_RN)
                                    else:
                                        a4 = cute.max(x4,zero4,propagate_nan=True)
                                    w4 = _subview(w_cache,i*(64//num_epi_subtiles)+h,4)
                                    s_acc = indexer_fma_packed_f32x2(a4,w4,s_acc,rnd=_RND_RN)
                        if cutlass.const_expr(self.native_lowp):
                            reduced=native_add(self.output_dtype,s_acc[0],s_acc[1])
                            score=native_sum_fp32(self.output_dtype,reduced)*scale_val
                        else:
                            score = (s_acc[0]+s_acc[1]+s_acc[2]+s_acc[3])*scale_val
                        if cutlass.const_expr(self.full_q_blocks) or outrow < num_q_tokens:
                            valid=(kv_pos>=kv_start)&(kv_pos<kv_end)
                            if kv_idx%2==0:
                                cursor_even=publish_bank(score,valid,tau,0,kv_pos,cursor_even,bank_rows,self.output_dtype,pred_store=(self.pro_packed_store and bank_rows.element_type==cutlass.Int32),raw_half=self.dsv4_cast,unordered=self.dsv4_unordered,pro_raw_packed=(self.pro_packed_store and bank_rows.element_type==cutlass.Int32))
                            else:
                                cursor_even=publish_bank(score,valid,tau,0,kv_pos,cursor_even,bank_rows,self.output_dtype,pred_store=(self.pro_packed_store and bank_rows.element_type==cutlass.Int32),raw_half=self.dsv4_cast,unordered=self.dsv4_unordered,pro_raw_packed=(self.pro_packed_store and bank_rows.element_type==cutlass.Int32))
                        kv_idx=kv_idx+1
                    if (cutlass.const_expr(self.full_q_blocks) or outrow<num_q_tokens) and cute.arch.lane_idx()==0:
                        candidates[outrow,bank_id]=candidates.element_type(cursor_even)
                        candidates[outrow,bank_id+4]=candidates.element_type(0)
                        cute.arch.atomic_add((counts.iterator+outrow).llvm_ptr,cursor_even-cutlass.Int32(cursor_even>bank_span)*16777216,sem="relaxed",scope="gpu")
                    q_pipeline.consumer_release(q_cons_state)
                    q_cons_state.advance()
                    next_q_block_idx=q_block_idx-cute.arch.grid_dim()[2]
                    if next_q_block_idx>=0:
                        q_last_next=min(next_q_block_idx*self.block_q+self.block_q-1,num_q_tokens-1)
                        next_num_kv=(mCuSeqLenKEnd[q_last_next]+127)//128
                    has_work=next_q_block_idx>=0
            else:
                while has_work:
                    q_block_idx = next_q_block_idx
                    num_kv = next_num_kv
                    q_pipeline.consumer_wait(q_cons_state)
                    q_stage_local = q_cons_state.index
                    queries_per_wg = self.block_q // (self.num_math_threads//128)
                    cursors = cute.make_rmem_tensor((self.block_q // (self.num_math_threads//128),), cutlass.Int32)
                    scores = cute.make_rmem_tensor((self.block_q // (self.num_math_threads//128),), cutlass.Float32)
                    start_cache = cute.make_rmem_tensor((self.block_q // (self.num_math_threads//128),), cutlass.Int32)
                    end_cache = cute.make_rmem_tensor((self.block_q // (self.num_math_threads//128),), cutlass.Int32)
                    tau_cache = cute.make_rmem_tensor((self.block_q // (self.num_math_threads//128),), cutlass.Float32)
                    if cutlass.const_expr(self.num_heads==32):
                        glm_weights=cute.make_rmem_tensor((queries_per_wg*(32//2 if self.packed_bf16_weights else 32),),cutlass.Uint32 if self.packed_bf16_weights else cutlass.Float32)
                    if cutlass.const_expr(self.qsa_weights==1):
                        uniform_cache = cute.make_rmem_tensor((queries_per_wg,),cutlass.Float32)
                    for local_q in cutlass.range_constexpr(self.block_q // (self.num_math_threads//128)):
                        cursors[local_q] = 0
                        slot = query_slot * queries_per_wg + local_q
                        outrow = q_block_idx * self.block_q + slot
                        token_idx = outrow if cutlass.const_expr(self.full_q_blocks) else min(outrow,num_q_tokens-1)
                        start_cache[local_q] = mCuSeqLenKStart[token_idx]
                        end_cache[local_q] = mCuSeqLenKEnd[token_idx]
                        tau_cache[local_q] = filter_boundary(thresholds[token_idx],self.output_dtype)
                        if cutlass.const_expr(self.num_heads==32):
                            if tau_cache[local_q]!=tau_cache[local_q]:tau_cache[local_q]=cutlass.Float32(float("inf"))
                        if cutlass.const_expr(self.num_heads==32):
                            if cutlass.const_expr(self.packed_bf16_weights):
                                for cached_h in cutlass.range_constexpr(32//2):
                                    glm_weights[local_q*(32//2)+cached_h]=native_weight_pair(sW,slot*32+2*cached_h,q_stage_local,cutlass.BFloat16)
                            else:
                                if cutlass.const_expr(self.interleaved_queries):
                                    for cached_h in cutlass.range_constexpr(32):glm_weights[cached_h*2+local_q]=_uniform_weight(mW_raw.iterator+(q_block_idx*self.block_q+slot)*self.num_heads+cached_h)
                                else:
                                    for cached_h in cutlass.range_constexpr(32):glm_weights[local_q*32+cached_h]=sW[(slot*self.num_heads+cached_h,q_stage_local)]
                        if cutlass.const_expr(self.qsa_weights==1):
                            uniform_cache[local_q] = sW[(slot*4,q_stage_local)]
                    kv_idx = _kv_lo_for_block(q_block_idx)
                    # Retire empty producer tiles after Q/weight setup so the
                    # empty-range branch does not affect scoring-loop scheduling.
                    if kv_idx >= num_kv:
                        if cutlass.const_expr(self.num_heads == 64 and self.dsv4_direct_sync):
                            cute.arch.mbarrier_wait(storage.umma_mbar_0.data_ptr()+umma_cons_state_0.index,umma_cons_state_0.phase)
                            cute.arch.mbarrier_arrive(storage.umma_mbar_0.data_ptr()+self.num_umma_stages+umma_cons_state_0.index)
                        else:
                            umma_pipeline_0.consumer_wait(umma_cons_state_0)
                            umma_pipeline_0.consumer_release(umma_cons_state_0)
                        umma_cons_state_0.advance()
                    if cutlass.const_expr(self.num_heads==32):
                        while kv_idx+self.splits*1<num_kv:
                            for kv_unroll in cutlass.range_constexpr(2):
                                kv_pos = kv_idx*128+m_coord
                                scale_val = cutlass.Float32(1) if cutlass.const_expr(self.qsa_unit_scale) else mS_raw[kv_pos if cutlass.const_expr(self.dsv4_aligned_k) else min(kv_pos,num_kv_tokens-1)]
                                umma_pipeline_0.consumer_wait(umma_cons_state_0)
                                tAcc = tCtAcc_base_0[(None,None,None,umma_cons_state_0.index)][((None,None),0,0)]
                                tTR = thr_copy_ref.partition_S(cute.flat_divide(tAcc,epi_sub_mn))
                                if cutlass.const_expr(self.interleaved_queries):
                                    slot0=query_slot*queries_per_wg
                                    pair_acc=cute.make_rmem_tensor((8,),cutlass.Float32)
                                    pair_acc.fill(0.0)
                                    for chunk in cutlass.range_constexpr(2*num_epi_subtiles):
                                        cute.copy(tiled_copy_ref,tTR[(None,None,None,0,slot0*num_epi_subtiles+chunk)],tTR_rAcc)
                                        cute.arch.fence_view_async_tmem_load()
                                        if cutlass.const_expr(chunk==2*num_epi_subtiles-1):
                                            umma_pipeline_0.consumer_release(umma_cons_state_0)
                                            umma_cons_state_0.advance()
                                        pair_acc=interleaved_partial(tTR_rAcc,glm_weights,pair_acc,chunk*(16//num_epi_subtiles),16//num_epi_subtiles,self.use_relu_trick)
                                    result=cute.make_tensor(pair_acc.iterator,cute.make_layout(2)).load()
                                    for chain in cutlass.range_constexpr(1,4):
                                        result=indexer_add_packed_f32x2(result,cute.make_tensor(pair_acc.iterator+chain*2,cute.make_layout(2)).load(),rnd="rn")
                                    scores[0]=result[0]*scale_val
                                    scores[1]=result[1]*scale_val
                                elif cutlass.const_expr(self.glm_pair_queries):
                                    pair_fragment=cute.make_rmem_tensor(tTR_rAcc.layout,cutlass.Float32)
                                    slot0=query_slot*queries_per_wg
                                    acc0=head_zero(self.output_dtype if self.native_lowp else None)
                                    acc1=head_zero(self.output_dtype if self.native_lowp else None)
                                    for epi_i in cutlass.range_constexpr(num_epi_subtiles):
                                        cute.copy(tiled_copy_ref,tTR[(None,None,None,0,slot0*num_epi_subtiles+epi_i)],tTR_rAcc)
                                        cute.copy(tiled_copy_ref,tTR[(None,None,None,0,(slot0+1)*num_epi_subtiles+epi_i)],pair_fragment)
                                        cute.arch.fence_view_async_tmem_load()
                                        if cutlass.const_expr(epi_i==num_epi_subtiles-1):
                                            umma_pipeline_0.consumer_release(umma_cons_state_0)
                                            umma_cons_state_0.advance()
                                        acc0,acc1=weighted_pair_partial_cached(tTR_rAcc,pair_fragment,glm_weights,0,
                                            epi_i*(32//num_epi_subtiles),32//num_epi_subtiles,acc0,acc1,
                                            self.use_relu_trick,self.weights_prepared,integer_period=0,native_dtype=self.output_dtype if self.native_lowp else None,fp8=True)
                                    scores[0]=(head_sum(acc0,self.output_dtype if self.native_lowp else None))*scale_val
                                    scores[1]=(head_sum(acc1,self.output_dtype if self.native_lowp else None))*scale_val
                                else:
                                    for local_q in cutlass.range_constexpr(self.block_q // (self.num_math_threads//128)):
                                        slot = query_slot * queries_per_wg + local_q
                                        outrow = q_block_idx * self.block_q + slot
                                        token_idx = outrow if cutlass.const_expr(self.full_q_blocks) else min(outrow,num_q_tokens-1)
                                        if cutlass.const_expr(self.num_heads==32):
                                            head_acc=head_zero(self.output_dtype if self.native_lowp else None)
                                            for epi_i in cutlass.range_constexpr(num_epi_subtiles):
                                                cute.copy(tiled_copy_ref,tTR[(None,None,None,0,slot*num_epi_subtiles+epi_i)],tTR_rAcc)
                                                cute.arch.fence_view_async_tmem_load()
                                                if cutlass.const_expr(local_q==queries_per_wg-1 and epi_i==num_epi_subtiles-1):
                                                    umma_pipeline_0.consumer_release(umma_cons_state_0)
                                                    umma_cons_state_0.advance()
                                                head_acc=weighted_partial_cached(tTR_rAcc,sW,glm_weights,slot,local_q,q_stage_local,epi_i*(32//num_epi_subtiles),32//num_epi_subtiles,32,head_acc,self.use_relu_trick,self.weights_prepared,integer_period=self.glm_integer_period,native_dtype=self.output_dtype if self.native_lowp else None,fp8=True)
                                            score=(head_sum(head_acc,self.output_dtype if self.native_lowp else None))*scale_val
                                        else:
                                            if cutlass.const_expr(self.N==64):
                                                cute.copy(tiled_copy_ref,tTR[(None,None,None,0,slot)],tTR_rAcc)
                                                cute.arch.fence_view_async_tmem_load()
                                            else:
                                                if cutlass.const_expr(local_q%4==0):
                                                    cute.copy(tiled_copy_ref,tTR[(None,None,None,0,slot//4)],tTR_rAcc)
                                                    cute.arch.fence_view_async_tmem_load()
                                            x4 = _subview(tTR_rAcc,0 if self.N==64 else (local_q%4)*4,4)
                                            if cutlass.const_expr(self.use_relu_trick):
                                                a4 = indexer_add_packed_f32x2(x4,abs(x4),rnd=_RND_RN)
                                            else:
                                                a4 = cute.max(x4,zero4,propagate_nan=True)
                                            shared_w = cute.make_rmem_tensor((4,),cutlass.Float32)
                                            for h in cutlass.range_constexpr(4):
                                                if cutlass.const_expr(self.qsa_weights==2):
                                                    w = cutlass.Float32(0.04419417306780815 if self.use_relu_trick else 0.0883883461356163)
                                                elif cutlass.const_expr(self.qsa_weights==1):
                                                    w = uniform_cache[local_q]
                                                else:
                                                    w = cutlass.Float32(sW[(slot*4+h,q_stage_local)])
                                                if cutlass.const_expr(self.use_relu_trick and not self.weights_prepared):w = w*0.5
                                                shared_w[h] = w
                                            if cutlass.const_expr(self.native_lowp):
                                                score=head_sum(native_head_fma4(x4,shared_w.load(),head_zero(self.output_dtype),self.output_dtype,True),self.output_dtype) *scale_val
                                            else:
                                                acc4 = indexer_fma_packed_f32x2(a4,shared_w.load(),zero4,rnd=_RND_RN)
                                                score = (acc4[0]+acc4[1]+acc4[2]+acc4[3]) *scale_val
                                        scores[local_q] = score
                                if cutlass.const_expr(self.num_heads!=32):
                                    umma_pipeline_0.consumer_release(umma_cons_state_0)
                                    umma_cons_state_0.advance()
                                if cutlass.const_expr(self.num_heads==32 and self.output_dtype in (cutlass.Float16,cutlass.BFloat16,cutlass.Float32)):
                                    pair_row=q_block_idx*self.block_q+query_slot*queries_per_wg
                                    bank_span=(candidates.shape[1]-8)//(4*self.splits)
                                    bank_id=(bidz%self.splits)*4+local_tidx//32
                                    pair_ptr=candidates.iterator+((pair_row*cutlass.Int32(candidates.stride[0])+8+bank_id*bank_span) if cutlass.const_expr(self.offsets32) else (cutlass.Int64(pair_row)*candidates.stride[0]+8+bank_id*bank_span))
                                    valid0=(kv_pos>=start_cache[0])&(kv_pos<end_cache[0])
                                    valid1=(kv_pos>=start_cache[1])&(kv_pos<end_cache[1])
                                    if cutlass.const_expr(not self.full_q_blocks):
                                        valid0=valid0&(pair_row<num_q_tokens)
                                        valid1=valid1&(pair_row+1<num_q_tokens)
                                    cursors[0],cursors[1]=fused_publish_pair(pair_ptr,pair_ptr+candidates.stride[0],scores[0],scores[1],tau_cache[0],tau_cache[1],valid0,valid1,kv_pos,cursors[0],cursors[1],bank_span,self.output_dtype,self.glm_skip_empty,packed=candidates.element_type==cutlass.Int32,ordered_threshold=self.num_heads==32)
                                else:
                                    for local_q in cutlass.range_constexpr(self.block_q // (self.num_math_threads//128)):
                                        slot = query_slot * queries_per_wg + local_q
                                        outrow = q_block_idx * self.block_q + slot
                                        token_idx = outrow if cutlass.const_expr(self.full_q_blocks) else min(outrow,num_q_tokens-1)
                                        score = scores[local_q]
                                        if cutlass.const_expr(self.full_q_blocks) or outrow < num_q_tokens:
                                            kv_start = start_cache[local_q]
                                            kv_end = end_cache[local_q]
                                            bank_span = (candidates.shape[1]-8)//(4*self.splits)
                                            bank_id = (bidz%self.splits)*4+local_tidx//32
                                            bank_rows = cute.make_tensor(candidates.iterator+((outrow*cutlass.Int32(candidates.stride[0])+8+bank_id*bank_span) if cutlass.const_expr(self.offsets32) else (cutlass.Int64(outrow)*candidates.stride[0]+8+bank_id*bank_span)),cute.make_layout((1,bank_span),stride=(bank_span,1)))
                                            if cutlass.const_expr(self.num_heads==32 and self.output_dtype in (cutlass.Float16,cutlass.BFloat16,cutlass.Float32)):
                                                cursors[local_q]=fused_publish(bank_rows.iterator,score,tau_cache[local_q],(kv_pos>=kv_start)&(kv_pos<kv_end),kv_pos,cursors[local_q],bank_span,self.output_dtype,False,True,packed=candidates.element_type==cutlass.Int32)
                                            else:
                                                cursors[local_q] = publish_bank(score,(kv_pos>=kv_start)&(kv_pos<kv_end),tau_cache[local_q],0,kv_pos,cursors[local_q],bank_rows,self.output_dtype,True,True)
                                kv_idx = kv_idx + self.splits
                    while kv_idx < num_kv:
                        kv_pos = kv_idx*128+m_coord
                        scale_val = cutlass.Float32(1) if cutlass.const_expr(self.qsa_unit_scale) else mS_raw[kv_pos if cutlass.const_expr(self.dsv4_aligned_k) else min(kv_pos,num_kv_tokens-1)]
                        umma_pipeline_0.consumer_wait(umma_cons_state_0)
                        tAcc = tCtAcc_base_0[(None,None,None,umma_cons_state_0.index)][((None,None),0,0)]
                        tTR = thr_copy_ref.partition_S(cute.flat_divide(tAcc,epi_sub_mn))
                        if cutlass.const_expr(self.interleaved_queries):
                            slot0=query_slot*queries_per_wg
                            pair_acc=cute.make_rmem_tensor((8,),cutlass.Float32)
                            pair_acc.fill(0.0)
                            for chunk in cutlass.range_constexpr(2*num_epi_subtiles):
                                cute.copy(tiled_copy_ref,tTR[(None,None,None,0,slot0*num_epi_subtiles+chunk)],tTR_rAcc)
                                cute.arch.fence_view_async_tmem_load()
                                if cutlass.const_expr(chunk==2*num_epi_subtiles-1):
                                    umma_pipeline_0.consumer_release(umma_cons_state_0)
                                    umma_cons_state_0.advance()
                                pair_acc=interleaved_partial(tTR_rAcc,glm_weights,pair_acc,chunk*(16//num_epi_subtiles),16//num_epi_subtiles,self.use_relu_trick)
                            result=cute.make_tensor(pair_acc.iterator,cute.make_layout(2)).load()
                            for chain in cutlass.range_constexpr(1,4):
                                result=indexer_add_packed_f32x2(result,cute.make_tensor(pair_acc.iterator+chain*2,cute.make_layout(2)).load(),rnd="rn")
                            scores[0]=result[0]*scale_val
                            scores[1]=result[1]*scale_val
                        elif cutlass.const_expr(self.glm_pair_queries):
                            pair_fragment=cute.make_rmem_tensor(tTR_rAcc.layout,cutlass.Float32)
                            slot0=query_slot*queries_per_wg
                            acc0=head_zero(self.output_dtype if self.native_lowp else None)
                            acc1=head_zero(self.output_dtype if self.native_lowp else None)
                            for epi_i in cutlass.range_constexpr(num_epi_subtiles):
                                cute.copy(tiled_copy_ref,tTR[(None,None,None,0,slot0*num_epi_subtiles+epi_i)],tTR_rAcc)
                                cute.copy(tiled_copy_ref,tTR[(None,None,None,0,(slot0+1)*num_epi_subtiles+epi_i)],pair_fragment)
                                cute.arch.fence_view_async_tmem_load()
                                if cutlass.const_expr(epi_i==num_epi_subtiles-1):
                                    umma_pipeline_0.consumer_release(umma_cons_state_0)
                                    umma_cons_state_0.advance()
                                acc0,acc1=weighted_pair_partial_cached(tTR_rAcc,pair_fragment,glm_weights,0,
                                    epi_i*(32//num_epi_subtiles),32//num_epi_subtiles,acc0,acc1,
                                    self.use_relu_trick,self.weights_prepared,integer_period=0,native_dtype=self.output_dtype if self.native_lowp else None,fp8=True)
                            scores[0]=(head_sum(acc0,self.output_dtype if self.native_lowp else None))*scale_val
                            scores[1]=(head_sum(acc1,self.output_dtype if self.native_lowp else None))*scale_val
                        else:
                            for local_q in cutlass.range_constexpr(self.block_q // (self.num_math_threads//128)):
                                slot = query_slot * queries_per_wg + local_q
                                outrow = q_block_idx * self.block_q + slot
                                token_idx = outrow if cutlass.const_expr(self.full_q_blocks) else min(outrow,num_q_tokens-1)
                                if cutlass.const_expr(self.num_heads==32):
                                    head_acc=head_zero(self.output_dtype if self.native_lowp else None)
                                    for epi_i in cutlass.range_constexpr(num_epi_subtiles):
                                        cute.copy(tiled_copy_ref,tTR[(None,None,None,0,slot*num_epi_subtiles+epi_i)],tTR_rAcc)
                                        cute.arch.fence_view_async_tmem_load()
                                        if cutlass.const_expr(local_q==queries_per_wg-1 and epi_i==num_epi_subtiles-1):
                                            umma_pipeline_0.consumer_release(umma_cons_state_0)
                                            umma_cons_state_0.advance()
                                        head_acc=weighted_partial_cached(tTR_rAcc,sW,glm_weights,slot,local_q,q_stage_local,epi_i*(32//num_epi_subtiles),32//num_epi_subtiles,32,head_acc,self.use_relu_trick,self.weights_prepared,integer_period=self.glm_integer_period,native_dtype=self.output_dtype if self.native_lowp else None,fp8=True)
                                    score=(head_sum(head_acc,self.output_dtype if self.native_lowp else None))*scale_val
                                else:
                                    if cutlass.const_expr(self.N==64):
                                        cute.copy(tiled_copy_ref,tTR[(None,None,None,0,slot)],tTR_rAcc)
                                        cute.arch.fence_view_async_tmem_load()
                                    else:
                                        if cutlass.const_expr(local_q%4==0):
                                            cute.copy(tiled_copy_ref,tTR[(None,None,None,0,slot//4)],tTR_rAcc)
                                            cute.arch.fence_view_async_tmem_load()
                                    x4 = _subview(tTR_rAcc,0 if self.N==64 else (local_q%4)*4,4)
                                    if cutlass.const_expr(self.use_relu_trick):
                                        a4 = indexer_add_packed_f32x2(x4,abs(x4),rnd=_RND_RN)
                                    else:
                                        a4 = cute.max(x4,zero4,propagate_nan=True)
                                    shared_w = cute.make_rmem_tensor((4,),cutlass.Float32)
                                    for h in cutlass.range_constexpr(4):
                                        if cutlass.const_expr(self.qsa_weights==2):
                                            w = cutlass.Float32(0.04419417306780815 if self.use_relu_trick else 0.0883883461356163)
                                        elif cutlass.const_expr(self.qsa_weights==1):
                                            w = uniform_cache[local_q]
                                        else:
                                            w = cutlass.Float32(sW[(slot*4+h,q_stage_local)])
                                        if cutlass.const_expr(self.use_relu_trick and not self.weights_prepared):w = w*0.5
                                        shared_w[h] = w
                                    if cutlass.const_expr(self.native_lowp):
                                        score=head_sum(native_head_fma4(x4,shared_w.load(),head_zero(self.output_dtype),self.output_dtype,True),self.output_dtype) *scale_val
                                    else:
                                        acc4 = indexer_fma_packed_f32x2(a4,shared_w.load(),zero4,rnd=_RND_RN)
                                        score = (acc4[0]+acc4[1]+acc4[2]+acc4[3]) *scale_val
                                scores[local_q] = score
                        if cutlass.const_expr(self.num_heads!=32):
                            umma_pipeline_0.consumer_release(umma_cons_state_0)
                            umma_cons_state_0.advance()
                        if cutlass.const_expr(self.num_heads==32 and self.output_dtype in (cutlass.Float16,cutlass.BFloat16,cutlass.Float32)):
                            pair_row=q_block_idx*self.block_q+query_slot*queries_per_wg
                            bank_span=(candidates.shape[1]-8)//(4*self.splits)
                            bank_id=(bidz%self.splits)*4+local_tidx//32
                            pair_ptr=candidates.iterator+((pair_row*cutlass.Int32(candidates.stride[0])+8+bank_id*bank_span) if cutlass.const_expr(self.offsets32) else (cutlass.Int64(pair_row)*candidates.stride[0]+8+bank_id*bank_span))
                            valid0=(kv_pos>=start_cache[0])&(kv_pos<end_cache[0])
                            valid1=(kv_pos>=start_cache[1])&(kv_pos<end_cache[1])
                            if cutlass.const_expr(not self.full_q_blocks):
                                valid0=valid0&(pair_row<num_q_tokens)
                                valid1=valid1&(pair_row+1<num_q_tokens)
                            cursors[0],cursors[1]=fused_publish_pair(pair_ptr,pair_ptr+candidates.stride[0],scores[0],scores[1],tau_cache[0],tau_cache[1],valid0,valid1,kv_pos,cursors[0],cursors[1],bank_span,self.output_dtype,self.glm_skip_empty,packed=candidates.element_type==cutlass.Int32,ordered_threshold=self.num_heads==32)
                        else:
                            for local_q in cutlass.range_constexpr(self.block_q // (self.num_math_threads//128)):
                                slot = query_slot * queries_per_wg + local_q
                                outrow = q_block_idx * self.block_q + slot
                                token_idx = outrow if cutlass.const_expr(self.full_q_blocks) else min(outrow,num_q_tokens-1)
                                score = scores[local_q]
                                if cutlass.const_expr(self.full_q_blocks) or outrow < num_q_tokens:
                                    kv_start = start_cache[local_q]
                                    kv_end = end_cache[local_q]
                                    bank_span = (candidates.shape[1]-8)//(4*self.splits)
                                    bank_id = (bidz%self.splits)*4+local_tidx//32
                                    bank_rows = cute.make_tensor(candidates.iterator+((outrow*cutlass.Int32(candidates.stride[0])+8+bank_id*bank_span) if cutlass.const_expr(self.offsets32) else (cutlass.Int64(outrow)*candidates.stride[0]+8+bank_id*bank_span)),cute.make_layout((1,bank_span),stride=(bank_span,1)))
                                    if cutlass.const_expr(self.num_heads==32 and self.output_dtype in (cutlass.Float16,cutlass.BFloat16,cutlass.Float32)):
                                        cursors[local_q]=fused_publish(bank_rows.iterator,score,tau_cache[local_q],(kv_pos>=kv_start)&(kv_pos<kv_end),kv_pos,cursors[local_q],bank_span,self.output_dtype,False,True,packed=candidates.element_type==cutlass.Int32)
                                    else:
                                        cursors[local_q] = publish_bank(score,(kv_pos>=kv_start)&(kv_pos<kv_end),tau_cache[local_q],0,kv_pos,cursors[local_q],bank_rows,self.output_dtype,True,True)
                        kv_idx = kv_idx + self.splits
                    for local_q in cutlass.range_constexpr(self.block_q // (self.num_math_threads//128)):
                        outrow = q_block_idx*self.block_q+query_slot*queries_per_wg+local_q
                        if (cutlass.const_expr(self.full_q_blocks) or outrow<num_q_tokens) and cute.arch.lane_idx()==0:
                            bank_span = (candidates.shape[1]-8)//(4*self.splits)
                            bank_id = (bidz%self.splits)*4+local_tidx//32
                            cursor = cursors[local_q]
                            candidates[outrow,bank_id] = candidates.element_type(cursor)
                            if cutlass.const_expr(self.splits==1):candidates[outrow,bank_id+4] = candidates.element_type(0)
                            cute.arch.atomic_add((counts.iterator+outrow).llvm_ptr,cursor-cutlass.Int32(cursor>bank_span)*16777216,sem="relaxed",scope="gpu")
                    q_pipeline.consumer_release(q_cons_state)
                    q_cons_state.advance()
                    next_q_block_idx = q_block_idx-cute.arch.grid_dim()[2]//self.splits
                    if next_q_block_idx>=0:
                        q_last_next=min(next_q_block_idx*self.block_q+self.block_q-1,num_q_tokens-1)
                        next_num_kv=(mCuSeqLenKEnd[q_last_next]+127)//128
                    has_work=next_q_block_idx>=0
            subset_barrier(barrier_id=15,number_of_threads=self.num_math_threads)
            if cutlass.const_expr(self.num_heads != 4):tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)




from .selection import finish_recovery, select_or_prepare_retry as select_candidates

from .selection import runtime_samples, block_samples

from .selection import encode
@cute.jit
def publish_bank(score,active,tau,row,index,cursor,bank_rows,dtype:cutlass.Constexpr,pred_store:cutlass.Constexpr=False,raw_half:cutlass.Constexpr=False,unordered:cutlass.Constexpr=False,pro_raw_packed:cutlass.Constexpr=False):
    value=score
    if cutlass.const_expr(not raw_half or dtype not in (cutlass.Float16,cutlass.BFloat16)):value=cutlass.Float32(dtype(score))
    hit=active&(~(value<tau)) if cutlass.const_expr(unordered) else active&((value>=tau)|(value!=value))
    mask=cute.arch.vote_ballot_sync(hit)
    # Warp-uniform empty ballots have no stores or cursor updates.
    next_cursor=cursor
    if mask!=0:
        if cutlass.const_expr(raw_half and dtype in (cutlass.Float16,cutlass.BFloat16) and not pro_raw_packed):value=cutlass.Float32(dtype(score))
        total=cutlass.Int32(cute.arch.popc(mask))
        lane=cute.arch.lane_idx()
        pos=cursor+cutlass.Int32(cute.arch.popc(mask&((cutlass.Uint32(1)<<lane)-1)))
        if cutlass.const_expr(pred_store and bank_rows.element_type==cutlass.Int32):
            predicated_store32(bank_rows.iterator+row*bank_rows.stride[0]+pos,
                pack_half_index(value,index,dtype),hit&(pos<bank_rows.shape[1]))
        elif cutlass.const_expr(pred_store):
            predicated_store64(bank_rows.iterator+row*bank_rows.stride[0]+pos,
                cutlass.Int64(encode(value,index,cutlass.Float32,False)),hit&(pos<bank_rows.shape[1]))
        else:
            if hit&(pos<bank_rows.shape[1]):
                bank_rows[row,pos]=bank_rows.element_type(encode(value,index,dtype,bank_rows.element_type==cutlass.Int32))
        next_cursor=cursor+total
    return next_cursor


from .weighted_heads import weighted_partial_cached

from .qsa_publish import publish_pair as fused_publish_pair

from .weighted_heads import weighted_pair_partial_cached

from .fp4_safe_filter import _uniform_weight
@cute.jit
def interleaved_partial(fragment,weights,acc,offset:cutlass.Constexpr,width:cutlass.Constexpr,relu:cutlass.Constexpr):
    # Pair across queries; keep the original head order in each FP32 chain.
    for h in cutlass.range_constexpr(width):
        x=cute.make_tensor(fragment.iterator+h*2,cute.make_layout(2)).load()
        if cutlass.const_expr(relu):positive=indexer_add_packed_f32x2(x,abs(x),rnd="rn")
        else:positive=cute.max(x,cute.full((2,),0.0,dtype=cutlass.Float32),propagate_nan=True)
        w=cute.make_tensor(weights.iterator+(offset+h)*2,cute.make_layout(2)).load()
        chain=cute.make_tensor(acc.iterator+((offset+h)%4)*2,cute.make_layout(2))
        chain.store(indexer_fma_packed_f32x2(positive,w,chain.load(),rnd="rn"))
    return acc

from .native_lowp_math import native_pack, native_pack_relu, native_fma, native_add, native_lo, native_hi, native_sum_fp32

from .native_lowp_math import native_fp8_pack, native_fp8_pack_relu

from .native_heads import head_zero, head_sum, native_head_fma4

from .native_heads import native_weight_pair
