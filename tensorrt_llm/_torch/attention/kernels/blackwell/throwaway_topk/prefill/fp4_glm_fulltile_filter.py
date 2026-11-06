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
from .weighted_heads import weighted_heads, weighted_heads_cached
from .qsa_publish import publish as fused_publish, pack_half_index
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

"""MXFP4 contiguous MQA stages with block-scaled MMA.

Q/K use packed E2M1 with four UE8M0 scales per token. Two math warp groups
process adjacent KV blocks; UMMA owns the KV/scale pipeline. Per-row bounds
mask the epilogue, whose output type is FP16, BF16 or FP32."""

import argparse
import functools
import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
from cutlass import Float4E2M1FN, Float8E8M0FNU, testing
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.arch import get_max_tmem_alloc_cols
from cutlass.cutlass_dsl import BaseDSL
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from ..mqa_logits_util import (
    align,
    calc_diff,
    KernelLaunch,
    ld_shared_b32,
    st_shared_b32,
)

# CuTe DSL expects rounding modes as string literals.
from .selection import emit, sample_tile, publish_private, select, select_candidates, propose

from .integer_relu_fast import integer_relu4, finite_integer_relu4

_RND_RN = "rn"


def _subview(tensor, offset, count):
    """Load count contiguous elements starting at the element offset."""
    return cute.make_tensor(tensor.iterator + offset, cute.make_layout(count)).load()


@cute.jit
def utccp_required_smem_warp_transpose(smem_ptr) -> None:
    """Transpose a 512-byte-aligned, 128-Int32 scale atom into UTCCP chunk layout.

    The warp-cooperative XOR permutation avoids shared-memory bank conflicts."""
    lane_idx = cute.arch.lane_idx()
    values = cute.make_rmem_tensor((4,), cutlass.Int32)
    for i in cutlass.range_constexpr(4):
        offset = (i ^ (lane_idx >> 3)) * 32 + lane_idx
        values[i] = ld_shared_b32(smem_ptr + offset)
    # Explicit -1 (all-lanes mask): the wrapper default 0xFFFFFFFF triggers a
    # noisy Int32-truncation DSLWarning on every compile; -1 is the same bits.
    cute.arch.sync_warp(-1)
    for i in cutlass.range_constexpr(4):
        offset = lane_idx * 4 + (i ^ (lane_idx >> 3))
        st_shared_b32(smem_ptr + offset, values[i])


def _target_is_rubin() -> bool:
    """Identify Rubin targets (major 10, minor >= 7) for TMEM capacity dispatch.

    Family-level architecture checks do not distinguish these targets from SM100."""
    arch = BaseDSL._get_dsl().get_arch_enum()
    return arch.major == 10 and arch.minor >= 7


class FP4PrefillMQALogitsKernel:
    """Contiguous MXFP4 MQA stages over grid-strided query/KV tiles.

    Two math warp groups share Q and process adjacent KV blocks. UMMA consumes
    UE8M0 scales through UTCCP; the epilogue uses the scaled accumulator."""

    def __init__(
        self,
        block_kv: int = 128,
        num_heads: int = 64,
        head_dim: int = 128,
        num_sms: int = 148,
        num_epi_subtiles: int = 1,
        epi_dtype=cutlass.Float32,
        output_dtype=cutlass.Float32,
        remove_online_sf_transpose: bool = False,
        use_batched_store: bool = True,
        use_flat_logits_view=None,
        use_two_level_task_loop=None,
        num_kv_stages: int = 0,  # 0 = auto (6); 0 selects the default depth.
        num_umma_stages: int = 0,  # 0 = auto (1); 0 selects the default depth.
        num_q_stages: int = 0,  # 0 selects the default depth.
        max_w_in_reg: int = 0,  # 0 = auto (formula below); tuning knob
        use_relu_trick: bool = False,
        mode: str = "filter",
        top_k: int = 2048,
        capacity: int = 8192,
        splits: int = 1,
        packed: bool = False,
        sample_size: int = 512,
        math_registers: int = 160,
        store_registers: int = 168,
        full_q_blocks: bool = False,
        weights_prepared: bool = False,
        branchless_publish: bool = False,
        offsets32: bool = False,
        glm_mid_kv: bool = False,
        glm_dual_kv: bool = False,
        glm_pair_skip_empty: bool = False,
        glm_direct_bounds: bool = False,
        dsv4_unroll: bool = False,
        dsv4_packed_store: bool = False,
        dsv4_full_tiles: bool = False,
        dsv4_unordered: bool = False,
        dsv4_cast: bool = False,
        dsv4_integer_period: int = 0,
        dsv4_finite_integer: bool = False,
        dsv4_integer_offset: int = 0,
        qsa_weights: int = 0,
        qsa_unit_scale: bool = False,
        persistent_sample: bool = True,
        pdl: bool = False,
    ):
        # Static FP4 invariants imposed by the TMEM and SMEM layouts below.
        self.pdl = bool(pdl)
        assert num_heads in (4, 32, 64), "prefill supports H4 and H64"
        assert head_dim == 128, "FP4 kernel hardcodes head_dim=128"
        assert epi_dtype in (
            cutlass.Float32,
            cutlass.BFloat16,
            cutlass.Float16,
        ), f"FP4 epi_dtype must be fp32/bf16/fp16; got {epi_dtype}"
        assert output_dtype in (
            cutlass.Float32,
            cutlass.BFloat16,
            cutlass.Float16,
        ), f"FP4 output_dtype must be fp32/bf16/fp16; got {output_dtype}"
        assert block_kv == 128, "FP4 compute tile (block_kv) hardcoded to 128"
        assert mode=="filter" and epi_dtype==cutlass.Float32 and splits==1
        self.mode, self.top_k, self.capacity = mode, top_k, capacity
        self.splits, self.packed = splits, packed
        self.sample_size = sample_size
        self.full_q_blocks = full_q_blocks
        self.weights_prepared = weights_prepared
        self.branchless_publish = branchless_publish
        self.offsets32 = offsets32
        self.glm_mid_kv = glm_mid_kv
        self.glm_pair_skip_empty = glm_pair_skip_empty
        self.glm_direct_bounds = glm_direct_bounds
        self.dsv4_unroll = dsv4_unroll
        self.dsv4_packed_store = dsv4_packed_store
        self.dsv4_full_tiles = dsv4_full_tiles
        self.dsv4_unordered = dsv4_unordered
        self.dsv4_cast = dsv4_cast
        # API enables use_relu_trick only when all scale bytes are <=159.
        # Input updates must retain that existing numerical-range contract.
        assert dsv4_integer_period in (0, 4, 8, 16, 32, 64)
        self.dsv4_finite_integer = dsv4_finite_integer
        assert not dsv4_integer_period or num_heads == 64
        assert (dsv4_integer_offset == 0 if not dsv4_integer_period else
                0 <= dsv4_integer_offset <= dsv4_integer_period - 4 and dsv4_integer_offset % 4 == 0)
        self.dsv4_integer_head_mask = sum(
            1 << h for h in range(64)
            if dsv4_integer_period and dsv4_integer_offset <= h % dsv4_integer_period < dsv4_integer_offset + 4
        )
        self.qsa_weights = qsa_weights
        self.qsa_unit_scale = qsa_unit_scale
        self.persistent_sample = persistent_sample
        self.math_registers, self.store_registers = math_registers, store_registers
        self.block_kv = block_kv
        self.num_heads = num_heads
        self.head_dim = head_dim
        # block_q: N = block_q * num_heads tokens per MMA tile.
        # Blackwell (SM100): TMEM = 512 cols; acc(N=256)+SFA+SFB = 544 > 512, so N=128.
        # Rubin (SM107/109): TMEM = 576/832 cols; N=256 fits with room to spare.
        is_rubin = _target_is_rubin()
        self.block_q = (64 if num_heads==4 else (256 if is_rubin else 128)) // num_heads
        if mode=='filter' and ((num_heads==64 and self.block_q!=2) or splits!=1):
            raise ValueError('two-warp prefill publisher requires block_q=2 and splits=1')
        self.N = self.block_q * num_heads  # 128 on Blackwell, 256 on Rubin
        self.num_sms = num_sms
        self.num_epi_subtiles = num_epi_subtiles
        self.epi_dtype = epi_dtype
        # Skip KV-scale transpose when storage uses UTCCP chunk layout
        # (128 keys per atom). Q-scale transpose is independent.
        self.remove_online_sf_transpose = remove_online_sf_transpose
        # When True, defer per-t STG to register array and emit all STGs in
        # one contiguous LSU phase after the for-t loop (epilogue micro-opt).
        self.use_batched_store = use_batched_store
        # Architecture-specific defaults for the flat logits view and nested task loop.
        is_rubin = _target_is_rubin()
        if use_flat_logits_view is None:
            use_flat_logits_view = not is_rubin
        if use_two_level_task_loop is None:
            use_two_level_task_loop = not (is_rubin and self.block_q == 2)
        self.use_flat_logits_view = use_flat_logits_view
        self.use_two_level_task_loop = use_two_level_task_loop
        # The optional ReLU identity (x+abs(x))/2 can overflow for x >= 2**127.
        # Enable it only when input scales and weights keep the intermediate finite.
        self.use_relu_trick = use_relu_trick
        # epi_bytes covers fp16 and bf16 (FP8 only handled fp16).
        self.epi_bytes = 2 if epi_dtype in (cutlass.Float16, cutlass.BFloat16) else 4
        # Align every weight stage to the 128-byte TMA SMEM requirement.
        w_stage_bytes = self.N * self.epi_bytes
        self.w_stage_stride = ((w_stage_bytes + 127) // 128 * 128) // self.epi_bytes
        self.output_dtype = output_dtype
        if num_epi_subtiles > 1 and num_heads % num_epi_subtiles != 0:
            raise ValueError("num_heads must be divisible by num_epi_subtiles")
        if (num_heads // num_epi_subtiles) % 4 != 0:
            raise ValueError(
                "num_heads // num_epi_subtiles must be divisible by 4 (FMA unroll granularity)"
            )
        self.num_groups = 1 if num_heads==32 and top_k!=2048 and not glm_dual_kv else 2

        self.num_math_threads = self.num_groups*256
        self.num_specialized_threads = 128
        self.threads_per_cta = self.num_math_threads+128
        self.num_math_warps = self.num_math_threads//32
        self.tma_warp_base = 8
        self.umma_warp_base = 10

        # Q pipeline depth; the default is two stages.
        self.num_q_stages = num_q_stages if num_q_stages > 0 else 2
        # UMMA pipeline depth; the default is one stage.
        self.num_umma_stages = num_umma_stages if num_umma_stages > 0 else 1
        # KV pipeline depth
        self.num_kv_stages = num_kv_stages if num_kv_stages > 0 else 6
        self.max_w_in_reg = max_w_in_reg

        # acc_dtype is locked to fp32 for FP4 MXF4 SS (cannot be exposed).
        self.acc_dtype = cutlass.Float32
        # SF (UE8M0) static knobs.
        self.sf_dtype = cutlass.Float8E8M0FNU
        self.sf_vec_size = 32
        # MXF4 instruction K is 64.
        self.umma_inst_k = 64
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (1, 1)
        self.mma_tiler_mn = (block_kv, self.N)

    def _setup_mma(self, a_dtype, b_dtype, a_major, b_major):
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.a_major_mode = a_major
        self.b_major_mode = b_major

        self.mma_tiler = (*self.mma_tiler_mn, 1)
        # MXF4 uses K=64 instructions, 32-element scale groups and
        # 64-byte swizzling for the packed D128 row.
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            a_dtype,
            b_dtype,
            a_major,
            b_major,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_tiler_mn,
        )
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])  # 64 for MXF4

        # Full-K: tile K = head_dim (128 FP4 elem), 1 TMA per block.
        # mma_inst_tile_k = 128 / 64 = 2 (FP4) vs 128 / 32 = 4 (FP8).
        mma_inst_tile_k = self.head_dim // mma_inst_shape_k
        full_k = mma_inst_shape_k * mma_inst_tile_k  # 128
        self.mma_tiler = (
            self.mma_tiler_mn[0],
            self.mma_tiler_mn[1],
            full_k,
        )

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

        # KV SMEM: the helper picks SWIZZLE_64B for packed FP4 rows.
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            a_dtype,
            self.num_kv_stages,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            b_dtype,
            self.num_q_stages,
        )

        # acc TMEM (per math WG, per UMMA stage)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        self.num_tmem_alloc_cols = indexer_memory.get_num_tmem_alloc_cols(
            tCtAcc_fake, rounding=False
        )

        # Use a virtual single-stage SMEM layout to derive the TMEM scale-factor layout.
        # The actual scale-factor SMEM allocation is flat.
        sfa_chunk_smem_for_inference = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, self.sf_vec_size, 1
        )
        sfb_chunk_smem_for_inference = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, self.sf_vec_size, 1
        )
        # The TMEM layout helper consumes rank-3 layouts. Restore a size-one stage
        # mode in its output for the four-coordinate slices used by S2T copies.
        _tmem_sfa_rank3 = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            cute.slice_(sfa_chunk_smem_for_inference, (None, None, None, 0)),
        )
        _tmem_sfb_rank3 = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            cute.slice_(sfb_chunk_smem_for_inference, (None, None, None, 0)),
        )
        self.tmem_sfa_layout = cute.append(
            _tmem_sfa_rank3, cute.make_layout(1, stride=0)
        )
        self.tmem_sfb_layout = cute.append(
            _tmem_sfb_rank3, cute.make_layout(1, stride=0)
        )
        # Derive the scale-factor K stride from the returned layout.
        _sf_kmode_size = cute.size(_tmem_sfa_rank3, mode=[2])
        _mma_inst_tile_k_for_sf = self.head_dim // self.umma_inst_k  # = 2 for FP4
        assert _sf_kmode_size % _mma_inst_tile_k_for_sf == 0, (
            f"SF K-mode ({_sf_kmode_size}) must be divisible by UMMA K-inst count "
            f"({_mma_inst_tile_k_for_sf})"
        )
        self.sf_k_step = _sf_kmode_size // _mma_inst_tile_k_for_sf
        # Count SF TMEM columns from the instruction layout;
        # cute.cosize reports the stride span, not occupied cells.
        sf_atom_mn = 32  # BlockScaledBasicChunk M atom
        mma_inst_tile_k = self.head_dim // self.umma_inst_k  # 128/64 = 2
        N_padded = ((self.N + 127) // 128) * 128
        self.num_sfa_tmem_cols = (
            self.cta_tile_shape_mnk[0] // sf_atom_mn
        ) * mma_inst_tile_k
        self.num_sfb_tmem_cols = (N_padded // sf_atom_mn) * mma_inst_tile_k

        # TMEM holds staged accumulators, per-WG SFA and shared Q SFB.
        # Separate SFA regions prevent concurrent UTCCP writes from racing.
        raw_total = (
            self.num_tmem_alloc_cols * self.num_groups * self.num_umma_stages
            + self.num_sfa_tmem_cols * self.num_groups
            + self.num_sfb_tmem_cols
        )
        # SM100 requires a power-of-two TMEM allocation of at most 512 columns.
        # Rubin permits larger, 32-column-aligned exclusive allocations; the DSL
        # selects the exclusive mode from the target and column count.
        arch = BaseDSL._get_dsl().get_arch_enum()
        self.arch_str = f"sm_{arch.major}{arch.minor}"  # family key, e.g. "sm_107"
        max_tmem_cols = get_max_tmem_alloc_cols(self.arch_str)
        if raw_total <= 512:
            # SM100 rule (also valid everywhere): round up to pow2, min 32.
            self.num_tmem_alloc_cols_total = max(
                1 << math.ceil(math.log2(raw_total)), 32
            )
        else:
            # >512: exclusive alloc, 32-col aligned (no pow2 rounding).
            self.num_tmem_alloc_cols_total = ((raw_total + 31) // 32) * 32
        assert self.num_tmem_alloc_cols_total <= max_tmem_cols, (
            f"FP4 TMEM {self.num_tmem_alloc_cols_total} cols (raw={raw_total}) "
            f"exceeds {self.arch_str} capacity {max_tmem_cols} for "
            f"block_q={self.block_q}: acc="
            f"{self.num_tmem_alloc_cols * self.num_groups * self.num_umma_stages}, "
            f"sfa_per_wg={self.num_sfa_tmem_cols} x{self.num_groups}, "
            f"sfb={self.num_sfb_tmem_cols}."
        )

        # Stash the chunk SMEM layouts; UMMA warp uses them as a reference
        # view onto our flat SF SMEM after the in-place transpose.
        self.sfa_chunk_smem_layout = sfa_chunk_smem_for_inference
        self.sfb_chunk_smem_layout = sfb_chunk_smem_for_inference

        # UTCCP S2T copy atoms for SF SMEM -> TMEM SF.
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        # Partition these atoms against the transposed SMEM view at each UTCCP.
        self.copy_atom_s2t = copy_atom_s2t

        return tiled_mma

    @cute.jit
    def __call__(
        self,
        kv: cute.Tensor,  # KV: [num_kv_tokens, head_dim/2] uint8 (FP4 packed)
        kv_scales: cute.Tensor,  # [num_kv_tokens] int32 (4 UE8M0 packed per token)
        b: cute.Tensor,  # Q: [N, head_dim/2, num_q_blocks] uint8 (FP4 packed)
        sf_q: cute.Tensor,  # Q SF: [N_padded, num_q_blocks] int32 (4 UE8M0 per token)
        weights: cute.Tensor,  # [N, num_q_blocks] epi_dtype (cast by host wrapper)
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
        half_head_dim = self.head_dim // 2  # FP4 packed bytes per row

        # Recast KV to FP4. Each uint8 byte becomes 2 FP4 elements,
        # so layout positions and the iterator scale accordingly.
        kv_fp4 = cute.recast_tensor(kv, Float4E2M1FN)

        # Q (b) was passed as uint8 (FP4-packed bytes); recast to FP4 so MMA
        # type inference and TMA descriptors are correct.
        b = cute.recast_tensor(b, Float4E2M1FN)

        # Contiguous KV view: [block_kv, head_dim, num_kv_blocks] FP4 elements.
        # Two elements share a byte; each row occupies head_dim/2 bytes.
        num_kv_blocks = num_kv_tokens // self.block_kv
        kv_layout = cute.make_layout(
            (self.block_kv, self.head_dim, num_kv_blocks),
            stride=(self.head_dim, 1, self.block_kv * self.head_dim),
        )
        a = cute.make_tensor(kv_fp4.iterator, kv_layout)

        # SF KV view: int32 (4 UE8M0 packed), contiguous per token.
        # [block_kv, num_kv_blocks] stride (1, block_kv)
        sf_kv_layout = cute.make_layout(
            (self.block_kv, num_kv_blocks),
            stride=(1, self.block_kv),
        )
        sf_kv = cute.make_tensor(kv_scales.iterator, sf_kv_layout)

        a_dtype = a.element_type
        b_dtype = b.element_type
        a_major = indexer_tensor_utils.LayoutEnum.from_tensor(a).mma_major_mode()
        b_major = indexer_tensor_utils.LayoutEnum.ROW_MAJOR.mma_major_mode()

        tiled_mma = self._setup_mma(a_dtype, b_dtype, a_major, b_major)
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # ---- TMA for KV (A): contiguous [block_kv, head_dim, num_kv_blocks] ----
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
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

        # TMA for Q (B) — full K=128, L dim = num_q_blocks
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

        # TMA for Weights — [N, num_q_blocks], tile [N], L=num_q_blocks
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

        # TMA loads one int32 scale pack per key; UMMA transposes the
        # row-major SMEM tile into UTCCP chunk layout.
        self.sf_kv_smem_layout_staged = cute.make_layout(
            (self.block_kv, self.num_kv_stages)
        )
        sf_kv_smem_per_block = cute.make_layout((self.block_kv,))
        tma_atom_sf_kv, tma_tensor_sf_kv = cpasync.make_tiled_tma_atom(
            tma_load_op,
            sf_kv,
            sf_kv_smem_per_block,
            (self.block_kv,),
        )

        # TMA loads N Q-scale entries into an N_padded allocation aligned for UTCCP.
        # UTCCP copies the padded region, but MMA and the epilogue consume only N entries.
        N_padded = ((self.N + 127) // 128) * 128
        self.N_padded = N_padded
        # sf_q stage stride in int32 elem (= 4 bytes/elem). Pad to 128B (= 32 int32).
        sf_q_stage_bytes = N_padded * 4
        sf_q_stage_stride_int32 = ((sf_q_stage_bytes + 127) // 128 * 128) // 4
        self.sf_q_smem_layout_staged = cute.make_layout(
            (N_padded, self.num_q_stages),
            stride=(1, sf_q_stage_stride_int32),
        )
        # Use an N-wide TMA view with the padded allocation's stage stride.
        # This satisfies the TMA layout symmetry check while preserving UTCCP alignment.
        self.sf_q_tma_smem_layout_staged = cute.make_layout(
            (self.N, self.num_q_stages),
            stride=(1, sf_q_stage_stride_int32),
        )
        sf_q_tma_smem_per_stage = cute.select(
            self.sf_q_tma_smem_layout_staged, mode=[0]
        )
        tma_atom_sf_q, tma_tensor_sf_q = cpasync.make_tiled_tma_atom(
            tma_load_op,
            sf_q,
            sf_q_tma_smem_per_stage,
            (self.N,),
        )

        b_copy_size = cute.size_in_bytes(b_dtype, b_smem_layout)
        w_copy_size = self.N * self.epi_bytes
        # Per KV block (FP4):
        #   block_kv * (head_dim/2) bytes data + block_kv * 4 bytes SF
        self.num_kv_sf_tma_bytes = self.block_kv * half_head_dim + self.block_kv * 4
        # Q, Q scales and weights share a barrier; count the Q-scale
        # transfer as N * 4 bytes to match the TMA descriptor.
        sf_q_tma_bytes = self.N * 4
        self.num_q_tma_bytes = (
            b_copy_size * atom_thr_size + sf_q_tma_bytes + w_copy_size
        )

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
            # SFB WAR-closure mbarrier pair (1 stage: full + unused empty).
            sfb_war_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_holding_buf: cutlass.Int32

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_w,
            tma_tensor_w,
            tma_atom_sf_kv,
            tma_tensor_sf_kv,
            tma_atom_sf_q,
            tma_tensor_sf_q,
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
            self.sf_kv_smem_layout_staged,
            self.sf_q_smem_layout_staged,
            self.sf_q_tma_smem_layout_staged,
            self.tmem_sfa_layout,
            self.tmem_sfb_layout,
            self.sfa_chunk_smem_layout,
            self.sfb_chunk_smem_layout,
            self.copy_atom_s2t,
            self.epi_tile,
            SharedStorage,
        ).launch(
            grid=(1, 1, num_ctas),
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=2 if self.num_groups==1 else 1,
        use_pdl=self.pdl)

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,  # KV: [block_kv, head_dim, num_kv_blocks] (FP4)
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,  # Q (L dim = num_q_blocks, FP4)
        tma_atom_w: cute.CopyAtom,
        mW_tma: cute.Tensor,  # Weights TMA coord tensor [N, num_q_blocks]
        tma_atom_sf_kv: cute.CopyAtom,
        mSF_KV_tma: cute.Tensor,  # SF KV TMA coord tensor [block_kv, num_kv_blocks] int32
        tma_atom_sf_q: cute.CopyAtom,
        mSF_Q_tma: cute.Tensor,  # SF Q TMA coord tensor [N_padded, num_q_blocks] int32
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
        sf_kv_smem_layout_staged: cute.Layout,
        sf_q_smem_layout_staged: cute.Layout,
        sf_q_tma_smem_layout_staged: cute.Layout,
        tmem_sfa_layout: cute.Layout,
        tmem_sfb_layout: cute.Layout,
        sfa_chunk_smem_layout: cute.Layout,
        sfb_chunk_smem_layout: cute.Layout,
        copy_atom_s2t: cute.CopyAtom,
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
            self._gemm_body(tiled_mma, tma_atom_a, mA_mkl, tma_atom_b, mB_nkl, tma_atom_w, mW_tma, tma_atom_sf_kv, mSF_KV_tma, tma_atom_sf_q, mSF_Q_tma, mLogits, mCuSeqLenKStart, mCuSeqLenKEnd, thresholds, counts, candidates, output, num_q_tokens, num_q_blocks, num_kv_tokens, cluster_layout_vmnk, a_smem_layout_staged, a_tma_smem_layout, b_smem_layout_staged, w_smem_layout_staged, sf_kv_smem_layout_staged, sf_q_smem_layout_staged, sf_q_tma_smem_layout_staged, tmem_sfa_layout, tmem_sfb_layout, sfa_chunk_smem_layout, sfb_chunk_smem_layout, copy_atom_s2t, epi_tile, SharedStorage)
        if cutlass.const_expr(self.mode in ('sample', 'recover')):
            cute.arch.barrier(barrier_id=14, number_of_threads=self.threads_per_cta)
            if cutlass.const_expr(self.mode!='sample'):
                cute.arch.fence_acq_rel_gpu()
                cute.arch.barrier(barrier_id=14,number_of_threads=self.threads_per_cta)
            if tid < 256:
                for offset in cutlass.range_constexpr(self.block_q):
                    row = row_base + offset
                    if row < num_q_tokens:
                        start = mCuSeqLenKStart[row]
                        end = mCuSeqLenKEnd[row]
                        required = min(self.top_k, end - start)
                        if cutlass.const_expr(self.mode == 'sample' and not self.persistent_sample):
                            active_samples=block_samples(row//self.block_q,mCuSeqLenKEnd,num_q_tokens,self.block_q,self.top_k)
                            multiple=2 if active_samples==512 else (4 if active_samples<=2048 else 6)
                            rank = min(active_samples, max(1, (multiple * self.top_k * active_samples + max(1, end - start) - 1) // max(1, end - start)))
                            propose(row, tid, mLogits, hist, scalar, thresholds, active_samples, rank)
                        elif cutlass.const_expr(self.mode != 'sample'):
                            if counts[row] < required:
                                finish_recovery(row, tid, candidates, counts, mCuSeqLenKStart, mCuSeqLenKEnd, mLogits, hist, scalar, output, thresholds, self.top_k, self.capacity, self.output_dtype, self.packed)

    @cute.jit
    def _gemm_body(self, tiled_mma: cute.TiledMma, tma_atom_a: cute.CopyAtom, mA_mkl: cute.Tensor, tma_atom_b: cute.CopyAtom, mB_nkl: cute.Tensor, tma_atom_w: cute.CopyAtom, mW_tma: cute.Tensor, tma_atom_sf_kv: cute.CopyAtom, mSF_KV_tma: cute.Tensor, tma_atom_sf_q: cute.CopyAtom, mSF_Q_tma: cute.Tensor, mLogits: cute.Tensor, mCuSeqLenKStart: cute.Tensor, mCuSeqLenKEnd: cute.Tensor, thresholds: cute.Tensor, counts: cute.Tensor, candidates: cute.Tensor, output: cute.Tensor, num_q_tokens: cutlass.Int32, num_q_blocks: cutlass.Int32, num_kv_tokens: cutlass.Int32, cluster_layout_vmnk: cute.Layout, a_smem_layout_staged: cute.ComposedLayout, a_tma_smem_layout: cute.ComposedLayout, b_smem_layout_staged: cute.ComposedLayout, w_smem_layout_staged: cute.Layout, sf_kv_smem_layout_staged: cute.Layout, sf_q_smem_layout_staged: cute.Layout, sf_q_tma_smem_layout_staged: cute.Layout, tmem_sfa_layout: cute.Layout, tmem_sfb_layout: cute.Layout, sfa_chunk_smem_layout: cute.Layout, sfb_chunk_smem_layout: cute.Layout, copy_atom_s2t: cute.CopyAtom, epi_tile: cute.Tile, SharedStorage: cutlass.Constexpr):
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

        # Warp roles
        warpgroup_idx = warp_idx // 4
        is_math_warp = warp_idx < self.num_math_warps
        is_tma_warp_0 = warp_idx == self.num_math_warps+0
        is_tma_warp_1 = warp_idx == self.num_math_warps+1
        is_tma_warp = is_tma_warp_0 | is_tma_warp_1
        is_umma_warp_0 = warp_idx == self.num_math_warps+2
        is_umma_warp_1 = warp_idx == self.num_math_warps+3

        # Grid-stride scheduler: this SM handles q_block_idx = sm_idx,
        # sm_idx+num_sms, ...
        NUM_MATH_WG = self.num_groups  # kNumMathWarpGroups
        block_kv_val = self.block_kv
        block_q_val = self.block_q
        sm_idx = bidz // self.splits

        # Prefetch the first Q block's KV length during prologue setup.
        # Clamp zero-work CTA indices; those CTAs never consume the loaded value.
        start_q_block = sm_idx
        start_q_block_clamped = min(start_q_block, num_q_blocks - 1)
        q_last_token_clamped = min(
            start_q_block_clamped * block_q_val + block_q_val - 1,
            num_q_tokens - 1,
        )
        current_num_kv = (
            mCuSeqLenKEnd[q_last_token_clamped] + block_kv_val - 1
        ) // block_kv_val

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

        current_kv_lo = _kv_lo_for_block(start_q_block_clamped)
        if cutlass.const_expr(self.mode == "sample"):
            current_num_kv = runtime_samples(mCuSeqLenKEnd[q_last_token_clamped],self.top_k)//128
            current_kv_lo = cutlass.Int32(0)

        if is_tma_warp:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_w)
            cpasync.prefetch_descriptor(tma_atom_sf_kv)
            cpasync.prefetch_descriptor(tma_atom_sf_q)

        storage = smem.allocate(SharedStorage)

        num_heads = self.num_heads
        num_epi_subtiles = self.num_epi_subtiles

        # === Pipelines ===
        prod_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)

        # Q pipeline: TMA producer → Math consumer (8 math warps)
        # PipelineTmaAsync: consumer_release uses is_signalling_thread
        # (lane 0 per warp). 8 math warps × 1 lane-0 = 8 arrives.
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
        # Both Math WGs share the same pipeline state (advance in lockstep)
        q_cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_q_stages
        )
        # UMMA warps observe Q pipeline (wait only, no release)
        # to ensure Q is in SMEM before GEMM. Critical for UMMA warp 1
        # since TMA warp 1 only loads KV1 (not Q).
        q_cons_state_umma_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_q_stages
        )
        q_cons_state_umma_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_q_stages
        )

        # UMMA waits on and releases the shared KV/scale pipeline.
        # Math consumes accumulators and does not release this pipeline.
        kv_cons_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        kv_pipeline_0 = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.kv_mbar_0.data_ptr(),
            num_stages=self.num_kv_stages,
            producer_group=prod_group,
            consumer_group=kv_cons_group,
            tx_count=self.num_kv_sf_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        kv_pipeline_1 = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.kv_mbar_1.data_ptr(),
            num_stages=self.num_kv_stages,
            producer_group=prod_group,
            consumer_group=kv_cons_group,
            tx_count=self.num_kv_sf_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        kv_prod_state_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_kv_stages
        )
        kv_prod_state_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_kv_stages
        )
        # UMMA consumer states (wait + release, owns the pipeline).
        # Math reads block-scaled accumulators and does not consume the KV+SF pipeline.
        kv_cons_state_umma_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )
        kv_cons_state_umma_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )

        # UMMA pipelines (per-group)
        math_threads_per_group = self.num_math_threads // self.num_groups
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

        # Both UMMA warps must finish reading SFB before warp 0 overwrites it.
        # Each warp commits its final MMA to this single-stage mbarrier; warp 0
        # waits for both commits. A named barrier orders issue, not MMA completion.
        sfb_war_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.sfb_war_mbar.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_groups),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        sfb_war_prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, 1
        )
        sfb_war_cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, 1
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

        # TMEM synchronization covers 320 math/UMMA threads, excluding TMA.
        # Math warp 0 allocates and waits for the final epilogue consumers.
        tmem_alloc_num_threads = self.num_math_threads+32*self.num_groups  # 10 warps: warp 0-7 (math) + warp 10-11 (umma)
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=tmem_alloc_num_threads
        )
        # Warp 0 copies Q scales into shared SFB TMEM; warp 1 reads that region.
        # Synchronize both UMMA warps after the copy: the Q pipeline orders TMA-to-SMEM,
        # not cross-warp TMEM visibility.
        sfb_sync_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=32*self.num_groups)
        tmem = indexer_memory.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=0,  # math warp 0 does alloc+free (last TMEM consumer)
            is_two_cta=False,
            # Arch-aware: sm_107 exposes 576 cols and needs exclusive alloc for
            # >512 (next_n=4 -> 544). Defaults to sm_100 (512) otherwise.
            arch=self.arch_str,
        )


        # Persistent K1 keeps per-query scratch in the math groups. TMA/UMMA
        # continue independently; barrier 12 includes only the 256 math lanes.
        if cutlass.const_expr(self.mode == 'sample' and self.persistent_sample):
            sample_hist=smem.allocate_tensor(cutlass.Int32,cute.make_layout(1538),byte_alignment=128)
            sample_scalar=smem.allocate_tensor(cutlass.Int32,cute.make_layout(2),byte_alignment=16)
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # SMEM allocation: per-group KV + shared Q
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
        # SF_Q SMEM (int32, packed UE8M0; flat layout, transposed
        # in-place by UMMA warp before UTCCP).
        sSF_Q = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=sf_q_smem_layout_staged,
            byte_alignment=128,
        )
        # Weights SMEM: [N, num_q_stages], shared Q barrier
        sW = smem.allocate_tensor(
            element_type=self.epi_dtype,
            layout=w_smem_layout_staged,
            byte_alignment=128,
        )
        # SF_KV SMEM (int32, packed UE8M0; flat layout, transposed
        # in-place by UMMA warp before UTCCP).
        sSF_KV_0 = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=sf_kv_smem_layout_staged,
            byte_alignment=128,
        )
        sSF_KV_1 = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=sf_kv_smem_layout_staged,
            byte_alignment=128,
        )

        a_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
        )
        b_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
        )

        # Partition KV (A): tile [block_kv, head_dim], coord = (0, 0, kv_block_idx)
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        gA = cute.local_tile(
            mA_mkl,
            (self.block_kv, self.head_dim),
            coord=(None, None, None),
        )
        # sKV SMEM view for TMA: group (block_kv, head_dim) into one mode to match
        # GMEM's ((block_kv,head_dim), ...) shape from group_modes(gA, 0, 2).
        # Use .outer (plain layout); swizzle is captured by sKV_0's iterator.
        sKV_0_for_tma = cute.make_tensor(
            sKV_0.iterator, cute.group_modes(a_tma_smem_layout.outer, 0, 2)
        )
        sKV_1_for_tma = cute.make_tensor(
            sKV_1.iterator, cute.group_modes(a_tma_smem_layout.outer, 0, 2)
        )
        tAsA_0, tAgA_0 = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            sKV_0_for_tma,
            cute.group_modes(gA, 0, 2),
        )
        tAsA_1, tAgA_1 = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            sKV_1_for_tma,
            cute.group_modes(gA, 0, 2),
        )

        # Partition Q (B): shared SMEM, L dim = num_q_blocks
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.mma_tiler, (0, None, None)),
            (None, None, None),
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
        tBgB = tBgB[(None, 0, None, None)]  # [tma, K, L]

        # Partition Weights: standalone TMA, [N, num_q_blocks] → [N] per stage
        w_cta_layout = cute.make_layout((1,))
        tWsW, tWgW = cpasync.tma_partition(
            tma_atom_w,
            0,
            w_cta_layout,
            cute.group_modes(sW, 0, 1),
            cute.group_modes(mW_tma, 0, 1),
        )

        # Partition SF KV: [block_kv, num_kv_blocks]; tile by (block_kv,).
        # Strides are int32-element units.
        sf_kv_tma_view_layout = cute.make_layout(
            (self.block_kv, self.num_kv_stages),
            stride=(1, self.block_kv),
        )
        sSF_KV_0_for_tma = cute.make_tensor(sSF_KV_0.iterator, sf_kv_tma_view_layout)
        sSF_KV_1_for_tma = cute.make_tensor(sSF_KV_1.iterator, sf_kv_tma_view_layout)
        # GMEM: local_tile by block_kv to match atom's tile size
        gSF_KV = cute.local_tile(mSF_KV_tma, (self.block_kv,), coord=(None, None))
        tSsSF_KV_0, tSgSF_KV_0 = cpasync.tma_partition(
            tma_atom_sf_kv,
            0,
            cute.make_layout(1),
            sSF_KV_0_for_tma,
            gSF_KV,
        )
        tSsSF_KV_1, tSgSF_KV_1 = cpasync.tma_partition(
            tma_atom_sf_kv,
            0,
            cute.make_layout(1),
            sSF_KV_1_for_tma,
            gSF_KV,
        )

        # Partition the N-wide TMA view using the padded SMEM stage stride.
        # UTCCP uses the full N_padded allocation; TMA leaves the tail untouched.
        sSF_Q_for_tma = cute.make_tensor(sSF_Q.iterator, sf_q_tma_smem_layout_staged)
        sf_q_cta_layout = cute.make_layout((1,))
        tSF_Q_sSF_Q, tSF_Q_gSF_Q = cpasync.tma_partition(
            tma_atom_sf_q,
            0,
            sf_q_cta_layout,
            cute.group_modes(sSF_Q_for_tma, 0, 1),
            cute.group_modes(mSF_Q_tma, 0, 1),
        )

        # MMA fragments
        tCrA_0 = tiled_mma.make_fragment_A(sKV_0)
        tCrA_1 = tiled_mma.make_fragment_A(sKV_1)
        tCrB = tiled_mma.make_fragment_B(sQ)  # shared
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])

        # Staged acc (fp16_gemm_3 pattern): append UMMA stage dim
        # shape: (*acc_shape, STAGE) — dynamic index on last dim reduces rank
        us = self.num_umma_stages
        cols = self.num_tmem_alloc_cols
        acc_shape_staged = cute.append(acc_shape, us)
        tCtAcc_fake_staged = tiled_mma.make_fragment_C(acc_shape_staged)

        # TMEM layout info (allocation deferred to UMMA/Math warp branches)
        cols_per_group = cols * us * (32 // self.acc_dtype.width)
        num_tmem_alloc_cols_total = self.num_tmem_alloc_cols_total

        # Epilogue setup
        c_layout = indexer_tensor_utils.LayoutEnum.ROW_MAJOR
        epi_sub_mn = (epi_tile[0], (16 if self.branchless_publish else 32) if self.num_heads==4 else num_heads // num_epi_subtiles)
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

        if is_tma_warp_0:
            # TMA warp 0: loads Q+SF_Q+Weights (on Q-block change) + KV group 0
            cute.arch.warpgroup_reg_dealloc(self.store_registers)

            # Prefetch first Q block before the loop
            q_pipeline.producer_acquire(q_prod_state)
            q_bar = q_pipeline.producer_get_barrier(q_prod_state)
            cute.copy(
                tma_atom_b,
                tBgB[(None, 0, next_q_block_idx)],
                tBsB[(None, q_prod_state.index)],
                tma_bar_ptr=q_bar,
                mcast_mask=b_mcast_mask,
            )
            # SF_Q TMA load — under same q_bar as Q + W.
            cute.copy(
                tma_atom_sf_q,
                tSF_Q_gSF_Q[(None, next_q_block_idx)],
                tSF_Q_sSF_Q[(None, q_prod_state.index)],
                tma_bar_ptr=q_bar,
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
                    if cutlass.const_expr(self.num_q_stages==1):prefetch_next_q=q_block_idx
                    if prefetch_next_q < num_q_blocks and (cutlass.const_expr(self.num_q_stages>1) or q_block_idx_old<num_q_blocks):
                        q_pipeline.producer_acquire(q_prod_state)
                        q_bar = q_pipeline.producer_get_barrier(q_prod_state)
                        cute.copy(
                            tma_atom_b,
                            tBgB[(None, 0, prefetch_next_q)],
                            tBsB[(None, q_prod_state.index)],
                            tma_bar_ptr=q_bar,
                            mcast_mask=b_mcast_mask,
                        )
                        # SF_Q TMA load — under same q_bar.
                        cute.copy(
                            tma_atom_sf_q,
                            tSF_Q_gSF_Q[(None, prefetch_next_q)],
                            tSF_Q_sSF_Q[(None, q_prod_state.index)],
                            tma_bar_ptr=q_bar,
                        )
                        cute.copy(
                            tma_atom_w,
                            tWgW[(None, prefetch_next_q)],
                            tWsW[(None, q_prod_state.index)],
                            tma_bar_ptr=q_bar,
                        )
                        q_prod_state.advance()

                # Load KV + SF for group 0: block at kv_idx (contiguous,
                # no block_table)
                kv_pipeline_0.producer_acquire(kv_prod_state_0)
                bar = kv_pipeline_0.producer_get_barrier(kv_prod_state_0)
                stage = kv_prod_state_0.index
                cute.copy(
                    tma_atom_a,
                    tAgA_0[(None, 0, 0, sample_tile(kv_idx, q_block_idx, mCuSeqLenKEnd, num_q_tokens, self.block_q, self.sample_size, self.top_k) if cutlass.const_expr(self.mode == "sample") else kv_idx)],
                    tAsA_0[(None, stage)],
                    tma_bar_ptr=bar,
                    mcast_mask=a_mcast_mask,
                )
                cute.copy(
                    tma_atom_sf_kv,
                    tSgSF_KV_0[(None, 0, sample_tile(kv_idx, q_block_idx, mCuSeqLenKEnd, num_q_tokens, self.block_q, self.sample_size, self.top_k) if cutlass.const_expr(self.mode == "sample") else kv_idx)],
                    tSsSF_KV_0[(None, stage)],
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
                        next_num_kv = (runtime_samples(mCuSeqLenKEnd[q_last_next],self.top_k)//128 if cutlass.const_expr(self.mode=="sample") else ((mCuSeqLenKEnd[q_last_next] + block_kv_val - 1) // block_kv_val))
                has_work = next_q_block_idx < num_q_blocks

        elif is_tma_warp_1 and cutlass.const_expr(self.num_groups==2):
            # TMA warp 1: loads KV + SF for group 1 only (contiguous,
            # no block_table)
            cute.arch.warpgroup_reg_dealloc(self.store_registers)

            while has_work:
                q_block_idx = next_q_block_idx
                kv_idx = next_kv_idx
                num_kv = next_num_kv

                # Load KV + SF for group 1: block at kv_idx + 1
                kv_idx_1 = kv_idx + 1
                kv_pipeline_1.producer_acquire(kv_prod_state_1)
                bar = kv_pipeline_1.producer_get_barrier(kv_prod_state_1)
                stage = kv_prod_state_1.index
                cute.copy(
                    tma_atom_a,
                    tAgA_1[(None, 0, 0, sample_tile(kv_idx_1, q_block_idx, mCuSeqLenKEnd, num_q_tokens, self.block_q, self.sample_size, self.top_k) if cutlass.const_expr(self.mode == "sample") else kv_idx_1)],
                    tAsA_1[(None, stage)],
                    tma_bar_ptr=bar,
                    mcast_mask=a_mcast_mask,
                )
                cute.copy(
                    tma_atom_sf_kv,
                    tSgSF_KV_1[(None, 0, sample_tile(kv_idx_1, q_block_idx, mCuSeqLenKEnd, num_q_tokens, self.block_q, self.sample_size, self.top_k) if cutlass.const_expr(self.mode == "sample") else kv_idx_1)],
                    tSsSF_KV_1[(None, stage)],
                    tma_bar_ptr=bar,
                )
                kv_prod_state_1.advance()

                # Scheduler advance
                next_kv_idx = kv_idx + NUM_MATH_WG * self.splits
                if next_kv_idx >= num_kv:
                    next_q_block_idx = q_block_idx + cute.arch.grid_dim()[2] // self.splits
                    next_kv_idx = _kv_lo_for_block(next_q_block_idx)
                    if next_q_block_idx < num_q_blocks:
                        q_last_next = min(
                            next_q_block_idx * block_q_val + block_q_val - 1,
                            num_q_tokens - 1,
                        )
                        next_num_kv = (runtime_samples(mCuSeqLenKEnd[q_last_next],self.top_k)//128 if cutlass.const_expr(self.mode=="sample") else ((mCuSeqLenKEnd[q_last_next] + block_kv_val - 1) // block_kv_val))
                has_work = next_q_block_idx < num_q_blocks

        elif is_umma_warp_0:
            # Wait for Q independently of KV: separate TMA barriers do not order
            # one another's SMEM writes.
            cute.arch.warpgroup_reg_dealloc(self.store_registers)

            # TMEM: wait for math warp 0's allocation, retrieve pointer
            subset_barrier(barrier_id=1,number_of_threads=self.num_math_threads+32*self.num_groups)
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base_0 = cute.make_tensor(tmem_ptr, tCtAcc_fake_staged.layout)
            tCtAcc_base_1 = cute.make_tensor(
                tmem_ptr + cols_per_group, tCtAcc_fake_staged.layout
            )

            # SF TMEM tensors. Per-WG SFA region (group 0 owns
            # cols [sf_base..sf_base+num_sfa_tmem_cols)). SFB is shared (Q SF)
            # and starts after BOTH groups' SFA regions.
            sf_base_offset = (
                self.num_tmem_alloc_cols * self.num_groups * self.num_umma_stages
            )
            sfa_tmem_ptr = cute.recast_ptr(
                tmem_ptr + sf_base_offset, dtype=self.sf_dtype
            )
            sfb_tmem_ptr = cute.recast_ptr(
                tmem_ptr + sf_base_offset + self.num_sfa_tmem_cols * self.num_groups,
                dtype=self.sf_dtype,
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tmem_sfa_layout)
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tmem_sfb_layout)

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
                        num_kv = block_samples(q_block_idx,mCuSeqLenKEnd,num_q_tokens,block_q_val,self.top_k)//128

                    # Wait for Q pipeline when the Q block changes
                    if q_block_idx != q_block_idx_old:
                        if q_block_idx_old < num_q_blocks:
                            # Commit the preceding Q MMAs to the WAR barrier before replacing
                            # scales; wait::st covers s2t stores, not MMA completion.
                            sfb_war_pipeline.producer_commit(sfb_war_prod_state)
                            q_cons_state_umma_0.advance()
                        q_pipeline.consumer_wait(q_cons_state_umma_0)
                        q_stage_0 = q_cons_state_umma_0.index

                        # New Q stage → re-issue UTCCP for SF Q.
                        # SMEM Q is N_padded tokens; loop over UTCCP atoms (128
                        # tokens each).
                        sf_q_atoms = self.N_padded // 128
                        for atom_idx in cutlass.range_constexpr(sf_q_atoms):
                            atom_offset = atom_idx * 128
                            stage_offset = q_stage_0 * sSF_Q.layout.stride[1]
                            utccp_required_smem_warp_transpose(
                                sSF_Q.iterator + stage_offset + atom_offset
                            )
                        cute.arch.fence_view_async_shared()
                        # Wait for both UMMA commits before overwriting SFB.
                        # The preceding SMEM transpose can overlap the MMA drain.
                        if q_block_idx_old < num_q_blocks:
                            sfb_war_pipeline.consumer_wait(sfb_war_cons_state)
                            sfb_war_cons_state.advance()
                        # Recast int32 scale packs as four UE8M0 cells for UTCCP.
                        sSF_Q_ue8m0 = cute.recast_tensor(sSF_Q, Float8E8M0FNU)
                        stage_off_ue8m0 = q_stage_0 * sSF_Q_ue8m0.layout.stride[1]
                        sSF_Q_chunk = cute.make_tensor(
                            sSF_Q_ue8m0.iterator + stage_off_ue8m0,
                            sfb_chunk_smem_layout,
                        )
                        tCsSFB_compact = cute.filter_zeros(sSF_Q_chunk)
                        tCtSFB_compact = cute.filter_zeros(tCtSFB)
                        tiled_copy_s2t_sfb = tcgen05.make_s2t_copy(
                            copy_atom_s2t, tCtSFB_compact
                        )
                        thr_copy_s2t_sfb = tiled_copy_s2t_sfb.get_slice(0)
                        tCsSFB_s2t_ = thr_copy_s2t_sfb.partition_S(tCsSFB_compact)
                        tCsSFB_s2t = tcgen05.get_s2t_smem_desc_tensor(
                            tiled_copy_s2t_sfb, tCsSFB_s2t_
                        )
                        tCtSFB_s2t = thr_copy_s2t_sfb.partition_D(tCtSFB_compact)
                        cute.copy(tiled_copy_s2t_sfb, tCsSFB_s2t, tCtSFB_s2t)
                        # Make SFB TMEM write visible to umma_warp_1 before its
                        # MMA reads the same SFB region. fence orders the async
                        # s2t; barrier crosses the warp boundary.
                        cute.arch.fence_view_async_tmem_store()
                        subset_barrier(barrier_id=2,number_of_threads=32*self.num_groups)

                    # Wait for KV, then TMEM capacity. Padded iterations compute unused results.
                    kv_pipeline_0.consumer_wait(kv_cons_state_umma_0)
                    umma_pipeline_0.producer_acquire(umma_prod_state_0)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    kv_stage = kv_cons_state_umma_0.index

                    # Transpose KV scales into UTCCP layout unless the host supplied that layout.
                    if cutlass.const_expr(not self.remove_online_sf_transpose):
                        sf_kv_atoms = self.block_kv // 128
                        for atom_idx in cutlass.range_constexpr(sf_kv_atoms):
                            atom_offset = atom_idx * 128
                            stage_offset = kv_stage * sSF_KV_0.layout.stride[1]
                            utccp_required_smem_warp_transpose(
                                sSF_KV_0.iterator + stage_offset + atom_offset
                            )
                        cute.arch.fence_view_async_shared()
                    # int32 SMEM → UE8M0 view for UTCCP atom + chunk layout.
                    sSF_KV_0_ue8m0 = cute.recast_tensor(sSF_KV_0, Float8E8M0FNU)
                    stage_off_kv0_ue8m0 = kv_stage * sSF_KV_0_ue8m0.layout.stride[1]
                    sSF_KV_0_chunk = cute.make_tensor(
                        sSF_KV_0_ue8m0.iterator + stage_off_kv0_ue8m0,
                        sfa_chunk_smem_layout,
                    )
                    tCsSFA_compact = cute.filter_zeros(sSF_KV_0_chunk)
                    tCtSFA_compact = cute.filter_zeros(tCtSFA)
                    tiled_copy_s2t_sfa = tcgen05.make_s2t_copy(
                        copy_atom_s2t, tCtSFA_compact
                    )
                    thr_copy_s2t_sfa = tiled_copy_s2t_sfa.get_slice(0)
                    tCsSFA_s2t_ = thr_copy_s2t_sfa.partition_S(tCsSFA_compact)
                    tCsSFA_s2t = tcgen05.get_s2t_smem_desc_tensor(
                        tiled_copy_s2t_sfa, tCsSFA_s2t_
                    )
                    tCtSFA_s2t = thr_copy_s2t_sfa.partition_D(tCtSFA_compact)
                    cute.copy(tiled_copy_s2t_sfa, tCsSFA_s2t, tCtSFA_s2t)

                    tCtAcc_0 = tCtAcc_base_0[
                        (None, None, None, umma_prod_state_0.index)
                    ]
                    # Slice the scale-factor TMEM K mode using the layout-derived sf_k_step.
                    # The last mode is the size-one stage dimension.
                    for k_block in cutlass.range_constexpr(num_k_blocks):
                        tCtSFA_k = cute.slice_(
                            tCtSFA, (None, None, k_block * self.sf_k_step, None)
                        )
                        tCtSFB_k = cute.slice_(
                            tCtSFB, (None, None, k_block * self.sf_k_step, None)
                        )
                        tiled_mma.set(tcgen05.Field.SFA, tCtSFA_k.iterator)
                        tiled_mma.set(tcgen05.Field.SFB, tCtSFB_k.iterator)
                        cute.gemm(
                            tiled_mma,
                            tCtAcc_0,
                            tCrA_0[None, None, k_block, kv_stage],
                            tCrB[None, None, k_block, q_stage_0],
                            tCtAcc_0,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    # UMMA releases the KV+SF stage after consuming its scales.
                    kv_pipeline_0.consumer_release(kv_cons_state_umma_0)
                    kv_cons_state_umma_0.advance()

                    umma_pipeline_0.producer_commit(umma_prod_state_0)
                    umma_prod_state_0.advance()

                    # Synchronize the UMMA warps on every tile so warp 0 cannot reach the next
                    # Q-scale overwrite before warp 1 has committed its MMA.
                    subset_barrier(barrier_id=2,number_of_threads=32*self.num_groups)

                    # Scheduler advance (grid-stride)
                    next_kv_idx = kv_idx + NUM_MATH_WG * self.splits
                    if next_kv_idx >= num_kv:
                        next_q_block_idx = q_block_idx + cute.arch.grid_dim()[2] // self.splits
                        next_kv_idx = _kv_lo_for_block(next_q_block_idx)
                        if next_q_block_idx < num_q_blocks:
                            q_last_next = min(
                                next_q_block_idx * block_q_val + block_q_val - 1,
                                num_q_tokens - 1,
                            )
                            next_num_kv = (runtime_samples(mCuSeqLenKEnd[q_last_next],self.top_k)//128 if cutlass.const_expr(self.mode=="sample") else ((mCuSeqLenKEnd[q_last_next] + block_kv_val - 1) // block_kv_val))
                    has_work = next_q_block_idx < num_q_blocks

        elif is_umma_warp_1 and cutlass.const_expr(self.num_groups==2):
            # Group 1 must wait on Q separately: its TMA warp loads only KV1.
            cute.arch.warpgroup_reg_dealloc(self.store_registers)

            # TMEM: wait for umma_warp_0's allocation, retrieve pointer
            subset_barrier(barrier_id=1,number_of_threads=self.num_math_threads+32*self.num_groups)
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base_0 = cute.make_tensor(tmem_ptr, tCtAcc_fake_staged.layout)
            tCtAcc_base_1 = cute.make_tensor(
                tmem_ptr + cols_per_group, tCtAcc_fake_staged.layout
            )

            # Group 1 uses the second SFA region. Only warp 0 writes shared SFB.
            sf_base_offset = (
                self.num_tmem_alloc_cols * self.num_groups * self.num_umma_stages
            )
            sfa_tmem_ptr = cute.recast_ptr(
                tmem_ptr + sf_base_offset + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            sfb_tmem_ptr = cute.recast_ptr(
                tmem_ptr + sf_base_offset + self.num_sfa_tmem_cols * self.num_groups,
                dtype=self.sf_dtype,
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tmem_sfa_layout)
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tmem_sfb_layout)

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
                        num_kv = block_samples(q_block_idx,mCuSeqLenKEnd,num_q_tokens,block_q_val,self.top_k)//128

                    # Wait for Q pipeline when the Q block changes
                    if q_block_idx != q_block_idx_old:
                        if q_block_idx_old < num_q_blocks:
                            # Commit the final MMA to sfb_war so warp 0 can wait before overwriting SFB.
                            # The named barrier alone does not establish MMA completion.
                            sfb_war_pipeline.producer_commit(sfb_war_prod_state)
                            q_cons_state_umma_1.advance()
                        q_pipeline.consumer_wait(q_cons_state_umma_1)
                        q_stage_1 = q_cons_state_umma_1.index
                        # Warp 0 owns the Q-scale copy. Warp 1 waits for cross-warp TMEM visibility
                        # before reading SFB; the Q pipeline covers only TMA-to-SMEM ordering.
                        subset_barrier(barrier_id=2,number_of_threads=32*self.num_groups)

                    # Process KV block for group 1 (kv_idx + 1)
                    # Unconditional UMMA
                    # Wait KV first, then TMEM empty
                    kv_pipeline_1.consumer_wait(kv_cons_state_umma_1)
                    umma_pipeline_1.producer_acquire(umma_prod_state_1)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    kv_stage_1 = kv_cons_state_umma_1.index

                    # Skip the transpose and its fence for prearranged UTCCP scales.
                    if cutlass.const_expr(not self.remove_online_sf_transpose):
                        sf_kv_atoms_1 = self.block_kv // 128
                        for atom_idx in cutlass.range_constexpr(sf_kv_atoms_1):
                            atom_offset = atom_idx * 128
                            stage_offset = kv_stage_1 * sSF_KV_1.layout.stride[1]
                            utccp_required_smem_warp_transpose(
                                sSF_KV_1.iterator + stage_offset + atom_offset
                            )
                        cute.arch.fence_view_async_shared()
                    sSF_KV_1_ue8m0 = cute.recast_tensor(sSF_KV_1, Float8E8M0FNU)
                    stage_off_kv1_ue8m0 = kv_stage_1 * sSF_KV_1_ue8m0.layout.stride[1]
                    sSF_KV_1_chunk = cute.make_tensor(
                        sSF_KV_1_ue8m0.iterator + stage_off_kv1_ue8m0,
                        sfa_chunk_smem_layout,
                    )
                    tCsSFA_compact_1 = cute.filter_zeros(sSF_KV_1_chunk)
                    tCtSFA_compact_1 = cute.filter_zeros(tCtSFA)
                    tiled_copy_s2t_sfa_1 = tcgen05.make_s2t_copy(
                        copy_atom_s2t, tCtSFA_compact_1
                    )
                    thr_copy_s2t_sfa_1 = tiled_copy_s2t_sfa_1.get_slice(0)
                    tCsSFA_s2t_1_ = thr_copy_s2t_sfa_1.partition_S(tCsSFA_compact_1)
                    tCsSFA_s2t_1 = tcgen05.get_s2t_smem_desc_tensor(
                        tiled_copy_s2t_sfa_1, tCsSFA_s2t_1_
                    )
                    tCtSFA_s2t_1 = thr_copy_s2t_sfa_1.partition_D(tCtSFA_compact_1)
                    cute.copy(tiled_copy_s2t_sfa_1, tCsSFA_s2t_1, tCtSFA_s2t_1)

                    tCtAcc_1 = tCtAcc_base_1[
                        (None, None, None, umma_prod_state_1.index)
                    ]
                    # Rank-4 slice mode 2 at k_block*sf_k_step per K-instr.
                    # See umma_warp_0 for sf_k_step rationale.
                    for k_block in cutlass.range_constexpr(num_k_blocks_1):
                        tCtSFA_k = cute.slice_(
                            tCtSFA, (None, None, k_block * self.sf_k_step, None)
                        )
                        tCtSFB_k = cute.slice_(
                            tCtSFB, (None, None, k_block * self.sf_k_step, None)
                        )
                        tiled_mma.set(tcgen05.Field.SFA, tCtSFA_k.iterator)
                        tiled_mma.set(tcgen05.Field.SFB, tCtSFB_k.iterator)
                        cute.gemm(
                            tiled_mma,
                            tCtAcc_1,
                            tCrA_1[None, None, k_block, kv_stage_1],
                            tCrB[None, None, k_block, q_stage_1],
                            tCtAcc_1,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    # UMMA owns the KV+SF release.
                    kv_pipeline_1.consumer_release(kv_cons_state_umma_1)
                    kv_cons_state_umma_1.advance()

                    umma_pipeline_1.producer_commit(umma_prod_state_1)
                    umma_prod_state_1.advance()

                    # Synchronize both UMMA warps per tile to prevent SFB overwrite
                    # while warp 1's MMA still reads it.
                    subset_barrier(barrier_id=2,number_of_threads=32*self.num_groups)

                    # Scheduler advance (grid-stride)
                    next_kv_idx = kv_idx + NUM_MATH_WG * self.splits
                    if next_kv_idx >= num_kv:
                        next_q_block_idx = q_block_idx + cute.arch.grid_dim()[2] // self.splits
                        next_kv_idx = _kv_lo_for_block(next_q_block_idx)
                        if next_q_block_idx < num_q_blocks:
                            q_last_next = min(
                                next_q_block_idx * block_q_val + block_q_val - 1,
                                num_q_tokens - 1,
                            )
                            next_num_kv = (runtime_samples(mCuSeqLenKEnd[q_last_next],self.top_k)//128 if cutlass.const_expr(self.mode=="sample") else ((mCuSeqLenKEnd[q_last_next] + block_kv_val - 1) // block_kv_val))
                    has_work = next_q_block_idx < num_q_blocks

        elif is_math_warp:
            cute.arch.warpgroup_reg_alloc(self.math_registers if self.mode == "filter" else 240)

            # TMEM: math warp 0 is the allocator; all math warps wait + retrieve
            tmem.allocate(num_tmem_alloc_cols_total)
            if cutlass.const_expr(self.num_heads==32):tmem.relinquish_alloc_permit()
            subset_barrier(barrier_id=1,number_of_threads=self.num_math_threads+32*self.num_groups)
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base_0 = cute.make_tensor(tmem_ptr, tCtAcc_fake_staged.layout)
            tCtAcc_base_1 = cute.make_tensor(
                tmem_ptr + cols_per_group, tCtAcc_fake_staged.layout
            )

            math_group=warpgroup_idx//(self.num_math_threads//128//self.num_groups)
            math_barriers=storage.umma_mbar_0.data_ptr()
            if math_group==1:math_barriers=storage.umma_mbar_1.data_ptr()
            tCtAcc_base_0=cute.make_tensor(tmem_ptr+math_group*cols_per_group,tCtAcc_fake_staged.layout)
            local_tidx = tidx % 128
            query_slot = warpgroup_idx%(self.num_math_threads//128//self.num_groups)
            cC = cute.make_identity_tensor(epi_sub_mn)
            tAcc_ref = tCtAcc_base_0[(None,None,None,0)][((None,None),0,0)]
            tAcc_ref_epi = cute.flat_divide(tAcc_ref,epi_sub_mn)
            tiled_copy_ref = tcgen05.make_tmem_copy(copy_atom_t2r,tAcc_ref_epi[(None,None,0,0)])
            thr_copy_ref = tiled_copy_ref.get_slice(local_tidx)
            tTR_cC = thr_copy_ref.partition_D(cC)
            m_coord = tTR_cC[0][0]
            tTR_rAcc = cute.make_fragment_like(tTR_cC,self.acc_dtype)
            w_cache = cute.make_rmem_tensor((self.max_w_in_reg,),cutlass.Float32)
            zero4 = cute.full((4,),0.0,dtype=cutlass.Float32)
            # Producers issue one tile pair even for an empty query range.
            # Retire it before advancing the persistent consumer pipeline.
            if cutlass.const_expr(self.num_heads == 64):
                while has_work:
                    q_block_idx = next_q_block_idx
                    num_kv = max(next_num_kv, _kv_lo_for_block(q_block_idx) + 1)
                    q_pipeline.consumer_wait(q_cons_state)
                    q_stage_local = q_cons_state.index
                    outrow = q_block_idx*2+query_slot
                    token_idx = min(outrow,num_q_tokens-1)
                    kv_start = mCuSeqLenKStart[token_idx]
                    kv_end = mCuSeqLenKEnd[token_idx]
                    tau = filter_boundary(thresholds[token_idx],self.output_dtype) if cutlass.const_expr(self.dsv4_cast) else thresholds[token_idx]
                    # A NaN threshold conservatively accepts the whole interval.
                    # K3/K4 still select exact top-k, including overflow recovery.
                    if cutlass.const_expr(self.dsv4_unordered):
                        if tau!=tau:tau=cutlass.Float32(float('-inf'))
                    for h in cutlass.range_constexpr(self.max_w_in_reg):
                        w = sW[(query_slot*64+h,q_stage_local)]
                        if cutlass.const_expr(self.use_relu_trick and not self.weights_prepared):w = w*0.5
                        if cutlass.const_expr(self.use_relu_trick and ((self.dsv4_integer_head_mask >> h) & 1)):
                            w = w * 2.0
                        w_cache[h] = w
                    bank_span = (candidates.shape[1]-8)//(4*self.num_groups)
                    bank_id = math_group*4+local_tidx//32
                    bank_rows = cute.make_tensor(candidates.iterator+cutlass.Int64(outrow)*candidates.stride[0]+8+bank_id*bank_span,cute.make_layout((1,bank_span),stride=(bank_span,1)))
                    cursor = cutlass.Int32(0)
                    kv_idx = _kv_lo_for_block(q_block_idx)
                    # Every producer and both math WGs traverse the same KV tiles.
                    if cutlass.const_expr(self.dsv4_full_tiles):
                        while (kv_idx < num_kv)&((kv_idx+math_group)*128<kv_start):
                            kv_pos = (kv_idx+math_group)*128+m_coord
                            cute.arch.mbarrier_wait(math_barriers+umma_cons_state_0.index,umma_cons_state_0.phase)
                            tAcc = tCtAcc_base_0[(None,None,None,umma_cons_state_0.index)][((None,None),0,0)]
                            tTR = thr_copy_ref.partition_S(cute.flat_divide(tAcc,epi_sub_mn))
                            s_acc = cute.full((4,),0.0,dtype=cutlass.Float32)
                            for i in cutlass.range_constexpr(num_epi_subtiles):
                                cute.copy(tiled_copy_ref,tTR[(None,None,None,0,query_slot*num_epi_subtiles+i)],tTR_rAcc)
                                cute.arch.fence_view_async_tmem_load()
                                if cutlass.const_expr(i==num_epi_subtiles-1):
                                    # 256 consumers release only after both query slices
                                    # are in registers. Reuse also protects staged SFA.
                                    cute.arch.mbarrier_arrive(math_barriers+self.num_umma_stages+umma_cons_state_0.index)
                                    umma_cons_state_0.advance()
                                for h in cutlass.range_constexpr(0,64//num_epi_subtiles,4):
                                    x4 = _subview(tTR_rAcc,h,4)
                                    if cutlass.const_expr(self.use_relu_trick):
                                        if cutlass.const_expr((self.dsv4_integer_head_mask >> (i*(64//num_epi_subtiles)+h)) & 1):
                                            a4 = finite_integer_relu4(x4) if cutlass.const_expr(self.dsv4_finite_integer) else integer_relu4(x4)
                                        else:
                                            a4 = indexer_add_packed_f32x2(x4,abs(x4),rnd=_RND_RN)
                                    else:
                                        a4 = cute.max(x4,zero4,propagate_nan=True)
                                    if cutlass.const_expr(i*(64//num_epi_subtiles)+h<self.max_w_in_reg):
                                        w4 = _subview(w_cache,i*(64//num_epi_subtiles)+h,4)
                                    else:
                                        w4 = cute.make_rmem_tensor((4,),cutlass.Float32)
                                        for j in cutlass.range_constexpr(4):
                                            w = sW[(query_slot*64+i*(64//num_epi_subtiles)+h+j,q_stage_local)]
                                            if cutlass.const_expr(self.use_relu_trick and not self.weights_prepared):w=w*0.5
                                            if cutlass.const_expr(self.use_relu_trick and ((self.dsv4_integer_head_mask >> (i*(64//num_epi_subtiles)+h+j)) & 1)):
                                                w=w*2.0
                                            w4[j]=w
                                        w4=w4.load()
                                    s_acc = indexer_fma_packed_f32x2(a4,w4,s_acc,rnd=_RND_RN)
                            score = s_acc[0]+s_acc[1]+s_acc[2]+s_acc[3]
                            if outrow < num_q_tokens:
                                valid=(kv_pos>=kv_start)&(kv_pos<kv_end)
                                cursor=publish_bank(score,valid,tau,0,kv_pos,cursor,bank_rows,self.output_dtype,raw_half=self.dsv4_cast,unordered=self.dsv4_unordered,raw_packed=(self.dsv4_packed_store and bank_rows.element_type==cutlass.Int32))
                            kv_idx=kv_idx+2
                    while kv_idx+(6 if cutlass.const_expr(self.dsv4_unroll) else 30) < (min(num_kv,kv_end//128-math_group) if cutlass.const_expr(self.dsv4_full_tiles) else num_kv):
                        for kv_unroll in cutlass.range_constexpr(4 if self.dsv4_unroll else 16):
                            kv_pos = (kv_idx+math_group)*128+m_coord
                            cute.arch.mbarrier_wait(math_barriers+umma_cons_state_0.index,umma_cons_state_0.phase)
                            tAcc = tCtAcc_base_0[(None,None,None,umma_cons_state_0.index)][((None,None),0,0)]
                            tTR = thr_copy_ref.partition_S(cute.flat_divide(tAcc,epi_sub_mn))
                            s_acc = cute.full((4,),0.0,dtype=cutlass.Float32)
                            for i in cutlass.range_constexpr(num_epi_subtiles):
                                cute.copy(tiled_copy_ref,tTR[(None,None,None,0,query_slot*num_epi_subtiles+i)],tTR_rAcc)
                                cute.arch.fence_view_async_tmem_load()
                                if cutlass.const_expr(i==num_epi_subtiles-1):
                                    # 256 consumers release only after both query slices
                                    # are in registers. Reuse also protects staged SFA.
                                    cute.arch.mbarrier_arrive(math_barriers+self.num_umma_stages+umma_cons_state_0.index)
                                    umma_cons_state_0.advance()
                                for h in cutlass.range_constexpr(0,64//num_epi_subtiles,4):
                                    x4 = _subview(tTR_rAcc,h,4)
                                    if cutlass.const_expr(self.use_relu_trick):
                                        if cutlass.const_expr((self.dsv4_integer_head_mask >> (i*(64//num_epi_subtiles)+h)) & 1):
                                            a4 = finite_integer_relu4(x4) if cutlass.const_expr(self.dsv4_finite_integer) else integer_relu4(x4)
                                        else:
                                            a4 = indexer_add_packed_f32x2(x4,abs(x4),rnd=_RND_RN)
                                    else:
                                        a4 = cute.max(x4,zero4,propagate_nan=True)
                                    if cutlass.const_expr(i*(64//num_epi_subtiles)+h<self.max_w_in_reg):
                                        w4 = _subview(w_cache,i*(64//num_epi_subtiles)+h,4)
                                    else:
                                        w4 = cute.make_rmem_tensor((4,),cutlass.Float32)
                                        for j in cutlass.range_constexpr(4):
                                            w = sW[(query_slot*64+i*(64//num_epi_subtiles)+h+j,q_stage_local)]
                                            if cutlass.const_expr(self.use_relu_trick and not self.weights_prepared):w=w*0.5
                                            if cutlass.const_expr(self.use_relu_trick and ((self.dsv4_integer_head_mask >> (i*(64//num_epi_subtiles)+h+j)) & 1)):
                                                w=w*2.0
                                            w4[j]=w
                                        w4=w4.load()
                                    s_acc = indexer_fma_packed_f32x2(a4,w4,s_acc,rnd=_RND_RN)
                            score = s_acc[0]+s_acc[1]+s_acc[2]+s_acc[3]
                            if outrow < num_q_tokens:
                                valid=True if cutlass.const_expr(self.dsv4_full_tiles) else (kv_pos>=kv_start)&(kv_pos<kv_end)
                                cursor=publish_bank(score,valid,tau,0,kv_pos,cursor,bank_rows,self.output_dtype,raw_half=self.dsv4_cast,unordered=self.dsv4_unordered,raw_packed=(self.dsv4_packed_store and bank_rows.element_type==cutlass.Int32))
                            kv_idx=kv_idx+2
                    while kv_idx < num_kv:
                        kv_pos = (kv_idx+math_group)*128+m_coord
                        cute.arch.mbarrier_wait(math_barriers+umma_cons_state_0.index,umma_cons_state_0.phase)
                        tAcc = tCtAcc_base_0[(None,None,None,umma_cons_state_0.index)][((None,None),0,0)]
                        tTR = thr_copy_ref.partition_S(cute.flat_divide(tAcc,epi_sub_mn))
                        s_acc = cute.full((4,),0.0,dtype=cutlass.Float32)
                        for i in cutlass.range_constexpr(num_epi_subtiles):
                            cute.copy(tiled_copy_ref,tTR[(None,None,None,0,query_slot*num_epi_subtiles+i)],tTR_rAcc)
                            cute.arch.fence_view_async_tmem_load()
                            if cutlass.const_expr(i==num_epi_subtiles-1):
                                # 256 consumers release only after both query slices
                                # are in registers. Reuse also protects staged SFA.
                                cute.arch.mbarrier_arrive(math_barriers+self.num_umma_stages+umma_cons_state_0.index)
                                umma_cons_state_0.advance()
                            for h in cutlass.range_constexpr(0,64//num_epi_subtiles,4):
                                x4 = _subview(tTR_rAcc,h,4)
                                if cutlass.const_expr(self.use_relu_trick):
                                    if cutlass.const_expr((self.dsv4_integer_head_mask >> (i*(64//num_epi_subtiles)+h)) & 1):
                                        a4 = finite_integer_relu4(x4) if cutlass.const_expr(self.dsv4_finite_integer) else integer_relu4(x4)
                                    else:
                                        a4 = indexer_add_packed_f32x2(x4,abs(x4),rnd=_RND_RN)
                                else:
                                    a4 = cute.max(x4,zero4,propagate_nan=True)
                                if cutlass.const_expr(i*(64//num_epi_subtiles)+h<self.max_w_in_reg):
                                    w4 = _subview(w_cache,i*(64//num_epi_subtiles)+h,4)
                                else:
                                    w4 = cute.make_rmem_tensor((4,),cutlass.Float32)
                                    for j in cutlass.range_constexpr(4):
                                        w = sW[(query_slot*64+i*(64//num_epi_subtiles)+h+j,q_stage_local)]
                                        if cutlass.const_expr(self.use_relu_trick and not self.weights_prepared):w=w*0.5
                                        if cutlass.const_expr(self.use_relu_trick and ((self.dsv4_integer_head_mask >> (i*(64//num_epi_subtiles)+h+j)) & 1)):
                                            w=w*2.0
                                        w4[j]=w
                                    w4=w4.load()
                                s_acc = indexer_fma_packed_f32x2(a4,w4,s_acc,rnd=_RND_RN)
                        score = s_acc[0]+s_acc[1]+s_acc[2]+s_acc[3]
                        if outrow < num_q_tokens:
                            valid=(kv_pos>=kv_start)&(kv_pos<kv_end)
                            cursor=publish_bank(score,valid,tau,0,kv_pos,cursor,bank_rows,self.output_dtype,raw_half=self.dsv4_cast,unordered=self.dsv4_unordered,raw_packed=(self.dsv4_packed_store and bank_rows.element_type==cutlass.Int32))
                        kv_idx=kv_idx+2
                    if outrow<num_q_tokens and cute.arch.lane_idx()==0:
                        # Each bank has an even span. Publish initialized
                        # padding for K3's final 128-bit candidate pair.
                        if ((cursor&1)!=0)&(cursor<bank_span):bank_rows[0,cursor]=bank_rows.element_type(0)
                        candidates[outrow,bank_id]=candidates.element_type(cursor)
                        cute.arch.atomic_add((counts.iterator+outrow).llvm_ptr,cursor-cutlass.Int32(cursor>bank_span)*16777216,sem="relaxed",scope="gpu")
                    q_pipeline.consumer_release(q_cons_state)
                    q_cons_state.advance()
                    next_q_block_idx=q_block_idx+cute.arch.grid_dim()[2]
                    if next_q_block_idx<num_q_blocks:
                        q_last_next=min(next_q_block_idx*2+1,num_q_tokens-1)
                        next_num_kv=(mCuSeqLenKEnd[q_last_next]+127)//128
                    has_work=next_q_block_idx<num_q_blocks

            else:
                if cutlass.const_expr(self.glm_mid_kv):
                    while has_work:
                        q_block_idx = next_q_block_idx
                        num_kv = next_num_kv
                        q_pipeline.consumer_wait(q_cons_state)
                        q_stage_local = q_cons_state.index
                        queries_per_wg = self.block_q // (self.num_math_threads//128//self.num_groups)
                        cursors = cute.make_rmem_tensor((self.block_q // (self.num_math_threads//128//self.num_groups),), cutlass.Int32)
                        scores = cute.make_rmem_tensor((self.block_q // (self.num_math_threads//128//self.num_groups),), cutlass.Float32)
                        start_cache = cute.make_rmem_tensor((self.block_q // (self.num_math_threads//128//self.num_groups),), cutlass.Int32)
                        end_cache = cute.make_rmem_tensor((self.block_q // (self.num_math_threads//128//self.num_groups),), cutlass.Int32)
                        tau_cache = cute.make_rmem_tensor((self.block_q // (self.num_math_threads//128//self.num_groups),), cutlass.Float32)
                        if cutlass.const_expr(self.num_heads==32):
                            glm_weights=cute.make_rmem_tensor((queries_per_wg*32,),cutlass.Float32)
                        if cutlass.const_expr(self.qsa_weights==1):
                            uniform_cache = cute.make_rmem_tensor((queries_per_wg,),cutlass.Float32)
                        for local_q in cutlass.range_constexpr(self.block_q // (self.num_math_threads//128//self.num_groups)):
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
                                for cached_h in cutlass.range_constexpr(32):
                                    cached_w=sW[(slot*self.num_heads+cached_h,q_stage_local)]
                                    if cutlass.const_expr(self.use_relu_trick and cached_h%(32 if self.top_k==2048 or self.num_groups==2 else 16)<(4 if self.top_k==2048 or self.num_groups==2 else 2)):cached_w=cached_w*2.0
                                    glm_weights[local_q*32+cached_h]=cached_w
                            if cutlass.const_expr(self.qsa_weights==1):
                                uniform_cache[local_q] = sW[(slot*4,q_stage_local)]
                        kv_idx = _kv_lo_for_block(q_block_idx)
                        # Retire empty capacity tiles without scoring or reordering Q/weight loads.
                        if kv_idx >= num_kv:
                            cute.arch.mbarrier_wait(
                                math_barriers + umma_cons_state_0.index,
                                umma_cons_state_0.phase,
                            )
                            cute.arch.mbarrier_arrive(
                                math_barriers
                                + self.num_umma_stages
                                + umma_cons_state_0.index
                            )
                            umma_cons_state_0.advance()
                        while (kv_idx < num_kv)&((kv_idx+math_group)*128<max(start_cache[0],start_cache[1])):
                            kv_pos = (kv_idx+math_group)*128+m_coord
                            cute.arch.mbarrier_wait(math_barriers+umma_cons_state_0.index,umma_cons_state_0.phase)
                            tAcc = tCtAcc_base_0[(None,None,None,umma_cons_state_0.index)][((None,None),0,0)]
                            tTR = thr_copy_ref.partition_S(cute.flat_divide(tAcc,epi_sub_mn))
                            if cutlass.const_expr(self.num_heads==32):
                                pair_fragment=cute.make_rmem_tensor(tTR_rAcc.layout,cutlass.Float32)
                                for pair_q in cutlass.range_constexpr(queries_per_wg//2):
                                    local0=pair_q*2
                                    slot0=query_slot*queries_per_wg+local0
                                    acc0=cute.full((4,),0.0,dtype=cutlass.Float32)
                                    acc1=cute.full((4,),0.0,dtype=cutlass.Float32)
                                    for epi_i in cutlass.range_constexpr(num_epi_subtiles):
                                        cute.copy(tiled_copy_ref,tTR[(None,None,None,0,slot0*num_epi_subtiles+epi_i)],tTR_rAcc)
                                        cute.copy(tiled_copy_ref,tTR[(None,None,None,0,(slot0+1)*num_epi_subtiles+epi_i)],pair_fragment)
                                        cute.arch.fence_view_async_tmem_load()
                                        if cutlass.const_expr(pair_q==queries_per_wg//2-1 and epi_i==num_epi_subtiles-1):
                                            # Both final Q slices have reached registers before reuse.
                                            cute.arch.mbarrier_arrive(math_barriers+self.num_umma_stages+umma_cons_state_0.index)
                                            umma_cons_state_0.advance()
                                        acc0,acc1=weighted_pair_partial_cached(tTR_rAcc,pair_fragment,glm_weights,
                                            local0,epi_i*(32//num_epi_subtiles),32//num_epi_subtiles,acc0,acc1,
                                            self.use_relu_trick,self.weights_prepared,integer_period=(32 if self.top_k==2048 or self.num_groups==2 else 16),integer_width=(4 if self.top_k==2048 or self.num_groups==2 else 2))
                                    scores[local0]=acc0[0]+acc0[1]+acc0[2]+acc0[3]
                                    scores[local0+1]=acc1[0]+acc1[1]+acc1[2]+acc1[3]
                            else:
                                for local_q in cutlass.range_constexpr(self.block_q // (self.num_math_threads//128//self.num_groups)):
                                    slot = query_slot * queries_per_wg + local_q
                                    outrow = q_block_idx * self.block_q + slot
                                    token_idx = outrow if cutlass.const_expr(self.full_q_blocks) else min(outrow,num_q_tokens-1)
                                    if cutlass.const_expr(self.num_heads==32):
                                        head_acc=cute.full((4,),0.0,dtype=cutlass.Float32)
                                        for epi_i in cutlass.range_constexpr(num_epi_subtiles):
                                            cute.copy(tiled_copy_ref,tTR[(None,None,None,0,slot*num_epi_subtiles+epi_i)],tTR_rAcc)
                                            cute.arch.fence_view_async_tmem_load()
                                            if cutlass.const_expr(local_q==queries_per_wg-1 and epi_i==num_epi_subtiles-1):
                                                cute.arch.mbarrier_arrive(math_barriers+self.num_umma_stages+umma_cons_state_0.index)
                                                umma_cons_state_0.advance()
                                            head_acc=weighted_partial_cached(tTR_rAcc,sW,glm_weights,slot,local_q,q_stage_local,epi_i*(32//num_epi_subtiles),32//num_epi_subtiles,32,head_acc,self.use_relu_trick,self.weights_prepared)
                                    else:
                                        if cutlass.const_expr(local_q%(4 if self.branchless_publish else 8)==0):
                                            cute.copy(tiled_copy_ref,tTR[(None,None,None,0,slot//(4 if self.branchless_publish else 8))],tTR_rAcc)
                                            cute.arch.fence_view_async_tmem_load()
                                    if cutlass.const_expr(self.num_heads==32):
                                        score=head_acc[0]+head_acc[1]+head_acc[2]+head_acc[3]
                                    else:
                                        x4 = _subview(tTR_rAcc,(local_q%(4 if self.branchless_publish else 8))*4,4)
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
                                                w = sW[(slot*4+h,q_stage_local)]
                                            if cutlass.const_expr(self.use_relu_trick and not self.weights_prepared):w = w*0.5
                                            shared_w[h] = w
                                        acc4 = indexer_fma_packed_f32x2(a4,shared_w.load(),zero4,rnd=_RND_RN)
                                        score = (acc4[0]+acc4[1]+acc4[2]+acc4[3])
                                    scores[local_q] = score
                            if cutlass.const_expr(self.num_heads!=32):
                                cute.arch.mbarrier_arrive(math_barriers+self.num_umma_stages+umma_cons_state_0.index)
                                umma_cons_state_0.advance()
                            if cutlass.const_expr(self.num_heads==32 and self.output_dtype in (cutlass.Float16,cutlass.BFloat16,cutlass.Float32)):
                                for pair_q in cutlass.range_constexpr(queries_per_wg//2):
                                    pair_row=q_block_idx*self.block_q+query_slot*queries_per_wg+pair_q*2
                                    bank_span=(candidates.shape[1]-8)//(4*self.num_groups)
                                    bank_id=math_group*4+local_tidx//32
                                    pair_ptr=candidates.iterator+((pair_row*cutlass.Int32(candidates.stride[0])+8+bank_id*bank_span) if cutlass.const_expr(self.offsets32) else (cutlass.Int64(pair_row)*candidates.stride[0]+8+bank_id*bank_span))
                                    if cutlass.const_expr(self.glm_direct_bounds):
                                        bound_end0=end_cache[pair_q*2]
                                        bound_end1=end_cache[pair_q*2+1]
                                        if cutlass.const_expr(not self.full_q_blocks):
                                            if pair_row>=num_q_tokens:bound_end0=cutlass.Int32(0)
                                            if pair_row+1>=num_q_tokens:bound_end1=cutlass.Int32(0)
                                        cursors[pair_q*2],cursors[pair_q*2+1]=fused_publish_pair(pair_ptr,pair_ptr+candidates.stride[0],scores[pair_q*2],scores[pair_q*2+1],tau_cache[pair_q*2],tau_cache[pair_q*2+1],cutlass.Int32(1),cutlass.Int32(1),kv_pos,cursors[pair_q*2],cursors[pair_q*2+1],bank_span,self.output_dtype,self.glm_pair_skip_empty,packed=candidates.element_type==cutlass.Int32,ordered_threshold=self.num_heads==32,bounds=(start_cache[pair_q*2],bound_end0,start_cache[pair_q*2+1],bound_end1))
                                    else:
                                        valid0=(kv_pos>=start_cache[pair_q*2])&(kv_pos<end_cache[pair_q*2])
                                        valid1=(kv_pos>=start_cache[pair_q*2+1])&(kv_pos<end_cache[pair_q*2+1])
                                        if cutlass.const_expr(not self.full_q_blocks):
                                            valid0=valid0&(pair_row<num_q_tokens)
                                            valid1=valid1&(pair_row+1<num_q_tokens)
                                        cursors[pair_q*2],cursors[pair_q*2+1]=fused_publish_pair(pair_ptr,pair_ptr+candidates.stride[0],scores[pair_q*2],scores[pair_q*2+1],tau_cache[pair_q*2],tau_cache[pair_q*2+1],valid0,valid1,kv_pos,cursors[pair_q*2],cursors[pair_q*2+1],bank_span,self.output_dtype,self.glm_pair_skip_empty,packed=candidates.element_type==cutlass.Int32,ordered_threshold=self.num_heads==32)
                            else:
                                for local_q in cutlass.range_constexpr(self.block_q // (self.num_math_threads//128//self.num_groups)):
                                    slot = query_slot * queries_per_wg + local_q
                                    outrow = q_block_idx * self.block_q + slot
                                    token_idx = outrow if cutlass.const_expr(self.full_q_blocks) else min(outrow,num_q_tokens-1)
                                    score = scores[local_q]
                                    if cutlass.const_expr(self.full_q_blocks) or outrow < num_q_tokens:
                                        kv_start = start_cache[local_q]
                                        kv_end = end_cache[local_q]
                                        bank_span = (candidates.shape[1]-8)//(4*self.num_groups)
                                        bank_id = math_group*4+local_tidx//32
                                        bank_rows = cute.make_tensor(candidates.iterator+((outrow*cutlass.Int32(candidates.stride[0])+8+bank_id*bank_span) if cutlass.const_expr(self.offsets32) else (cutlass.Int64(outrow)*candidates.stride[0]+8+bank_id*bank_span)),cute.make_layout((1,bank_span),stride=(bank_span,1)))
                                        if cutlass.const_expr(self.num_heads==32 and self.output_dtype in (cutlass.Float16,cutlass.BFloat16,cutlass.Float32)):
                                            cursors[local_q]=fused_publish(bank_rows.iterator,score,tau_cache[local_q],(kv_pos>=kv_start)&(kv_pos<kv_end),kv_pos,cursors[local_q],bank_span,self.output_dtype,False,False,packed=candidates.element_type==cutlass.Int32)
                                        elif cutlass.const_expr(self.qsa_weights==2 and self.branchless_publish and self.use_relu_trick and self.output_dtype in (cutlass.Float16, cutlass.BFloat16, cutlass.Float32)):
                                            cursors[local_q]=fused_publish(bank_rows.iterator,score,tau_cache[local_q],(kv_pos>=kv_start)&(kv_pos<kv_end),kv_pos,cursors[local_q],bank_span,self.output_dtype,True,packed=candidates.element_type==cutlass.Int32)
                                        else:
                                            cursors[local_q] = publish_bank(score,(kv_pos>=kv_start)&(kv_pos<kv_end),tau_cache[local_q],0,kv_pos,cursors[local_q],bank_rows,self.output_dtype,True,self.branchless_publish,True)
                            kv_idx = kv_idx + self.num_groups
                        if cutlass.const_expr(self.num_heads==32):
                            while kv_idx+3*self.num_groups+math_group < min(num_kv,min(end_cache[0],end_cache[1])//128):
                                for kv_unroll in cutlass.range_constexpr(4):
                                    kv_pos = (kv_idx+math_group)*128+m_coord
                                    cute.arch.mbarrier_wait(math_barriers+umma_cons_state_0.index,umma_cons_state_0.phase)
                                    tAcc = tCtAcc_base_0[(None,None,None,umma_cons_state_0.index)][((None,None),0,0)]
                                    tTR = thr_copy_ref.partition_S(cute.flat_divide(tAcc,epi_sub_mn))
                                    if cutlass.const_expr(self.num_heads==32):
                                        pair_fragment=cute.make_rmem_tensor(tTR_rAcc.layout,cutlass.Float32)
                                        for pair_q in cutlass.range_constexpr(queries_per_wg//2):
                                            local0=pair_q*2
                                            slot0=query_slot*queries_per_wg+local0
                                            acc0=cute.full((4,),0.0,dtype=cutlass.Float32)
                                            acc1=cute.full((4,),0.0,dtype=cutlass.Float32)
                                            for epi_i in cutlass.range_constexpr(num_epi_subtiles):
                                                cute.copy(tiled_copy_ref,tTR[(None,None,None,0,slot0*num_epi_subtiles+epi_i)],tTR_rAcc)
                                                cute.copy(tiled_copy_ref,tTR[(None,None,None,0,(slot0+1)*num_epi_subtiles+epi_i)],pair_fragment)
                                                cute.arch.fence_view_async_tmem_load()
                                                if cutlass.const_expr(pair_q==queries_per_wg//2-1 and epi_i==num_epi_subtiles-1):
                                                    # Both final Q slices have reached registers before reuse.
                                                    cute.arch.mbarrier_arrive(math_barriers+self.num_umma_stages+umma_cons_state_0.index)
                                                    umma_cons_state_0.advance()
                                                acc0,acc1=weighted_pair_partial_cached(tTR_rAcc,pair_fragment,glm_weights,
                                                    local0,epi_i*(32//num_epi_subtiles),32//num_epi_subtiles,acc0,acc1,
                                                    self.use_relu_trick,self.weights_prepared,integer_period=(32 if self.top_k==2048 or self.num_groups==2 else 16),integer_width=(4 if self.top_k==2048 or self.num_groups==2 else 2))
                                            scores[local0]=acc0[0]+acc0[1]+acc0[2]+acc0[3]
                                            scores[local0+1]=acc1[0]+acc1[1]+acc1[2]+acc1[3]
                                    else:
                                        for local_q in cutlass.range_constexpr(self.block_q // (self.num_math_threads//128//self.num_groups)):
                                            slot = query_slot * queries_per_wg + local_q
                                            outrow = q_block_idx * self.block_q + slot
                                            token_idx = outrow if cutlass.const_expr(self.full_q_blocks) else min(outrow,num_q_tokens-1)
                                            if cutlass.const_expr(self.num_heads==32):
                                                head_acc=cute.full((4,),0.0,dtype=cutlass.Float32)
                                                for epi_i in cutlass.range_constexpr(num_epi_subtiles):
                                                    cute.copy(tiled_copy_ref,tTR[(None,None,None,0,slot*num_epi_subtiles+epi_i)],tTR_rAcc)
                                                    cute.arch.fence_view_async_tmem_load()
                                                    if cutlass.const_expr(local_q==queries_per_wg-1 and epi_i==num_epi_subtiles-1):
                                                        cute.arch.mbarrier_arrive(math_barriers+self.num_umma_stages+umma_cons_state_0.index)
                                                        umma_cons_state_0.advance()
                                                    head_acc=weighted_partial_cached(tTR_rAcc,sW,glm_weights,slot,local_q,q_stage_local,epi_i*(32//num_epi_subtiles),32//num_epi_subtiles,32,head_acc,self.use_relu_trick,self.weights_prepared)
                                            else:
                                                if cutlass.const_expr(local_q%(4 if self.branchless_publish else 8)==0):
                                                    cute.copy(tiled_copy_ref,tTR[(None,None,None,0,slot//(4 if self.branchless_publish else 8))],tTR_rAcc)
                                                    cute.arch.fence_view_async_tmem_load()
                                            if cutlass.const_expr(self.num_heads==32):
                                                score=head_acc[0]+head_acc[1]+head_acc[2]+head_acc[3]
                                            else:
                                                x4 = _subview(tTR_rAcc,(local_q%(4 if self.branchless_publish else 8))*4,4)
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
                                                        w = sW[(slot*4+h,q_stage_local)]
                                                    if cutlass.const_expr(self.use_relu_trick and not self.weights_prepared):w = w*0.5
                                                    shared_w[h] = w
                                                acc4 = indexer_fma_packed_f32x2(a4,shared_w.load(),zero4,rnd=_RND_RN)
                                                score = (acc4[0]+acc4[1]+acc4[2]+acc4[3])
                                            scores[local_q] = score
                                    if cutlass.const_expr(self.num_heads!=32):
                                        cute.arch.mbarrier_arrive(math_barriers+self.num_umma_stages+umma_cons_state_0.index)
                                        umma_cons_state_0.advance()
                                    if cutlass.const_expr(self.num_heads==32 and self.output_dtype in (cutlass.Float16,cutlass.BFloat16,cutlass.Float32)):
                                        for pair_q in cutlass.range_constexpr(queries_per_wg//2):
                                            pair_row=q_block_idx*self.block_q+query_slot*queries_per_wg+pair_q*2
                                            bank_span=(candidates.shape[1]-8)//(4*self.num_groups)
                                            bank_id=math_group*4+local_tidx//32
                                            pair_ptr=candidates.iterator+((pair_row*cutlass.Int32(candidates.stride[0])+8+bank_id*bank_span) if cutlass.const_expr(self.offsets32) else (cutlass.Int64(pair_row)*candidates.stride[0]+8+bank_id*bank_span))
                                            if cutlass.const_expr(False):
                                                bound_end0=end_cache[pair_q*2]
                                                bound_end1=end_cache[pair_q*2+1]
                                                if cutlass.const_expr(not self.full_q_blocks):
                                                    if pair_row>=num_q_tokens:bound_end0=cutlass.Int32(0)
                                                    if pair_row+1>=num_q_tokens:bound_end1=cutlass.Int32(0)
                                                cursors[pair_q*2],cursors[pair_q*2+1]=fused_publish_pair(pair_ptr,pair_ptr+candidates.stride[0],scores[pair_q*2],scores[pair_q*2+1],tau_cache[pair_q*2],tau_cache[pair_q*2+1],cutlass.Int32(1),cutlass.Int32(1),kv_pos,cursors[pair_q*2],cursors[pair_q*2+1],bank_span,self.output_dtype,self.glm_pair_skip_empty,packed=candidates.element_type==cutlass.Int32,ordered_threshold=self.num_heads==32,bounds=(start_cache[pair_q*2],bound_end0,start_cache[pair_q*2+1],bound_end1))
                                            else:
                                                valid0=cutlass.Boolean(True)
                                                valid1=cutlass.Boolean(True)
                                                if cutlass.const_expr(not self.full_q_blocks):
                                                    valid0=valid0&(pair_row<num_q_tokens)
                                                    valid1=valid1&(pair_row+1<num_q_tokens)
                                                cursors[pair_q*2],cursors[pair_q*2+1]=fused_publish_pair(pair_ptr,pair_ptr+candidates.stride[0],scores[pair_q*2],scores[pair_q*2+1],tau_cache[pair_q*2],tau_cache[pair_q*2+1],valid0,valid1,kv_pos,cursors[pair_q*2],cursors[pair_q*2+1],bank_span,self.output_dtype,self.glm_pair_skip_empty,packed=candidates.element_type==cutlass.Int32,ordered_threshold=self.num_heads==32)
                                    else:
                                        for local_q in cutlass.range_constexpr(self.block_q // (self.num_math_threads//128//self.num_groups)):
                                            slot = query_slot * queries_per_wg + local_q
                                            outrow = q_block_idx * self.block_q + slot
                                            token_idx = outrow if cutlass.const_expr(self.full_q_blocks) else min(outrow,num_q_tokens-1)
                                            score = scores[local_q]
                                            if cutlass.const_expr(self.full_q_blocks) or outrow < num_q_tokens:
                                                kv_start = start_cache[local_q]
                                                kv_end = end_cache[local_q]
                                                bank_span = (candidates.shape[1]-8)//(4*self.num_groups)
                                                bank_id = math_group*4+local_tidx//32
                                                bank_rows = cute.make_tensor(candidates.iterator+((outrow*cutlass.Int32(candidates.stride[0])+8+bank_id*bank_span) if cutlass.const_expr(self.offsets32) else (cutlass.Int64(outrow)*candidates.stride[0]+8+bank_id*bank_span)),cute.make_layout((1,bank_span),stride=(bank_span,1)))
                                                if cutlass.const_expr(self.num_heads==32 and self.output_dtype in (cutlass.Float16,cutlass.BFloat16,cutlass.Float32)):
                                                    cursors[local_q]=fused_publish(bank_rows.iterator,score,tau_cache[local_q],(kv_pos>=kv_start)&(kv_pos<kv_end),kv_pos,cursors[local_q],bank_span,self.output_dtype,False,False,packed=candidates.element_type==cutlass.Int32)
                                                elif cutlass.const_expr(self.qsa_weights==2 and self.branchless_publish and self.use_relu_trick and self.output_dtype in (cutlass.Float16, cutlass.BFloat16, cutlass.Float32)):
                                                    cursors[local_q]=fused_publish(bank_rows.iterator,score,tau_cache[local_q],(kv_pos>=kv_start)&(kv_pos<kv_end),kv_pos,cursors[local_q],bank_span,self.output_dtype,True,packed=candidates.element_type==cutlass.Int32)
                                                else:
                                                    cursors[local_q] = publish_bank(score,(kv_pos>=kv_start)&(kv_pos<kv_end),tau_cache[local_q],0,kv_pos,cursors[local_q],bank_rows,self.output_dtype,True,self.branchless_publish,True)
                                    kv_idx = kv_idx + self.num_groups
                        while kv_idx < num_kv:
                            kv_pos = (kv_idx+math_group)*128+m_coord
                            cute.arch.mbarrier_wait(math_barriers+umma_cons_state_0.index,umma_cons_state_0.phase)
                            tAcc = tCtAcc_base_0[(None,None,None,umma_cons_state_0.index)][((None,None),0,0)]
                            tTR = thr_copy_ref.partition_S(cute.flat_divide(tAcc,epi_sub_mn))
                            if cutlass.const_expr(self.num_heads==32):
                                pair_fragment=cute.make_rmem_tensor(tTR_rAcc.layout,cutlass.Float32)
                                for pair_q in cutlass.range_constexpr(queries_per_wg//2):
                                    local0=pair_q*2
                                    slot0=query_slot*queries_per_wg+local0
                                    acc0=cute.full((4,),0.0,dtype=cutlass.Float32)
                                    acc1=cute.full((4,),0.0,dtype=cutlass.Float32)
                                    for epi_i in cutlass.range_constexpr(num_epi_subtiles):
                                        cute.copy(tiled_copy_ref,tTR[(None,None,None,0,slot0*num_epi_subtiles+epi_i)],tTR_rAcc)
                                        cute.copy(tiled_copy_ref,tTR[(None,None,None,0,(slot0+1)*num_epi_subtiles+epi_i)],pair_fragment)
                                        cute.arch.fence_view_async_tmem_load()
                                        if cutlass.const_expr(pair_q==queries_per_wg//2-1 and epi_i==num_epi_subtiles-1):
                                            # Both final Q slices have reached registers before reuse.
                                            cute.arch.mbarrier_arrive(math_barriers+self.num_umma_stages+umma_cons_state_0.index)
                                            umma_cons_state_0.advance()
                                        acc0,acc1=weighted_pair_partial_cached(tTR_rAcc,pair_fragment,glm_weights,
                                            local0,epi_i*(32//num_epi_subtiles),32//num_epi_subtiles,acc0,acc1,
                                            self.use_relu_trick,self.weights_prepared,integer_period=(32 if self.top_k==2048 or self.num_groups==2 else 16),integer_width=(4 if self.top_k==2048 or self.num_groups==2 else 2))
                                    scores[local0]=acc0[0]+acc0[1]+acc0[2]+acc0[3]
                                    scores[local0+1]=acc1[0]+acc1[1]+acc1[2]+acc1[3]
                            else:
                                for local_q in cutlass.range_constexpr(self.block_q // (self.num_math_threads//128//self.num_groups)):
                                    slot = query_slot * queries_per_wg + local_q
                                    outrow = q_block_idx * self.block_q + slot
                                    token_idx = outrow if cutlass.const_expr(self.full_q_blocks) else min(outrow,num_q_tokens-1)
                                    if cutlass.const_expr(self.num_heads==32):
                                        head_acc=cute.full((4,),0.0,dtype=cutlass.Float32)
                                        for epi_i in cutlass.range_constexpr(num_epi_subtiles):
                                            cute.copy(tiled_copy_ref,tTR[(None,None,None,0,slot*num_epi_subtiles+epi_i)],tTR_rAcc)
                                            cute.arch.fence_view_async_tmem_load()
                                            if cutlass.const_expr(local_q==queries_per_wg-1 and epi_i==num_epi_subtiles-1):
                                                cute.arch.mbarrier_arrive(math_barriers+self.num_umma_stages+umma_cons_state_0.index)
                                                umma_cons_state_0.advance()
                                            head_acc=weighted_partial_cached(tTR_rAcc,sW,glm_weights,slot,local_q,q_stage_local,epi_i*(32//num_epi_subtiles),32//num_epi_subtiles,32,head_acc,self.use_relu_trick,self.weights_prepared)
                                    else:
                                        if cutlass.const_expr(local_q%(4 if self.branchless_publish else 8)==0):
                                            cute.copy(tiled_copy_ref,tTR[(None,None,None,0,slot//(4 if self.branchless_publish else 8))],tTR_rAcc)
                                            cute.arch.fence_view_async_tmem_load()
                                    if cutlass.const_expr(self.num_heads==32):
                                        score=head_acc[0]+head_acc[1]+head_acc[2]+head_acc[3]
                                    else:
                                        x4 = _subview(tTR_rAcc,(local_q%(4 if self.branchless_publish else 8))*4,4)
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
                                                w = sW[(slot*4+h,q_stage_local)]
                                            if cutlass.const_expr(self.use_relu_trick and not self.weights_prepared):w = w*0.5
                                            shared_w[h] = w
                                        acc4 = indexer_fma_packed_f32x2(a4,shared_w.load(),zero4,rnd=_RND_RN)
                                        score = (acc4[0]+acc4[1]+acc4[2]+acc4[3])
                                    scores[local_q] = score
                            if cutlass.const_expr(self.num_heads!=32):
                                cute.arch.mbarrier_arrive(math_barriers+self.num_umma_stages+umma_cons_state_0.index)
                                umma_cons_state_0.advance()
                            if cutlass.const_expr(self.num_heads==32 and self.output_dtype in (cutlass.Float16,cutlass.BFloat16,cutlass.Float32)):
                                for pair_q in cutlass.range_constexpr(queries_per_wg//2):
                                    pair_row=q_block_idx*self.block_q+query_slot*queries_per_wg+pair_q*2
                                    bank_span=(candidates.shape[1]-8)//(4*self.num_groups)
                                    bank_id=math_group*4+local_tidx//32
                                    pair_ptr=candidates.iterator+((pair_row*cutlass.Int32(candidates.stride[0])+8+bank_id*bank_span) if cutlass.const_expr(self.offsets32) else (cutlass.Int64(pair_row)*candidates.stride[0]+8+bank_id*bank_span))
                                    if cutlass.const_expr(self.glm_direct_bounds):
                                        bound_end0=end_cache[pair_q*2]
                                        bound_end1=end_cache[pair_q*2+1]
                                        if cutlass.const_expr(not self.full_q_blocks):
                                            if pair_row>=num_q_tokens:bound_end0=cutlass.Int32(0)
                                            if pair_row+1>=num_q_tokens:bound_end1=cutlass.Int32(0)
                                        cursors[pair_q*2],cursors[pair_q*2+1]=fused_publish_pair(pair_ptr,pair_ptr+candidates.stride[0],scores[pair_q*2],scores[pair_q*2+1],tau_cache[pair_q*2],tau_cache[pair_q*2+1],cutlass.Int32(1),cutlass.Int32(1),kv_pos,cursors[pair_q*2],cursors[pair_q*2+1],bank_span,self.output_dtype,self.glm_pair_skip_empty,packed=candidates.element_type==cutlass.Int32,ordered_threshold=self.num_heads==32,bounds=(start_cache[pair_q*2],bound_end0,start_cache[pair_q*2+1],bound_end1))
                                    else:
                                        valid0=(kv_pos>=start_cache[pair_q*2])&(kv_pos<end_cache[pair_q*2])
                                        valid1=(kv_pos>=start_cache[pair_q*2+1])&(kv_pos<end_cache[pair_q*2+1])
                                        if cutlass.const_expr(not self.full_q_blocks):
                                            valid0=valid0&(pair_row<num_q_tokens)
                                            valid1=valid1&(pair_row+1<num_q_tokens)
                                        cursors[pair_q*2],cursors[pair_q*2+1]=fused_publish_pair(pair_ptr,pair_ptr+candidates.stride[0],scores[pair_q*2],scores[pair_q*2+1],tau_cache[pair_q*2],tau_cache[pair_q*2+1],valid0,valid1,kv_pos,cursors[pair_q*2],cursors[pair_q*2+1],bank_span,self.output_dtype,self.glm_pair_skip_empty,packed=candidates.element_type==cutlass.Int32,ordered_threshold=self.num_heads==32)
                            else:
                                for local_q in cutlass.range_constexpr(self.block_q // (self.num_math_threads//128//self.num_groups)):
                                    slot = query_slot * queries_per_wg + local_q
                                    outrow = q_block_idx * self.block_q + slot
                                    token_idx = outrow if cutlass.const_expr(self.full_q_blocks) else min(outrow,num_q_tokens-1)
                                    score = scores[local_q]
                                    if cutlass.const_expr(self.full_q_blocks) or outrow < num_q_tokens:
                                        kv_start = start_cache[local_q]
                                        kv_end = end_cache[local_q]
                                        bank_span = (candidates.shape[1]-8)//(4*self.num_groups)
                                        bank_id = math_group*4+local_tidx//32
                                        bank_rows = cute.make_tensor(candidates.iterator+((outrow*cutlass.Int32(candidates.stride[0])+8+bank_id*bank_span) if cutlass.const_expr(self.offsets32) else (cutlass.Int64(outrow)*candidates.stride[0]+8+bank_id*bank_span)),cute.make_layout((1,bank_span),stride=(bank_span,1)))
                                        if cutlass.const_expr(self.num_heads==32 and self.output_dtype in (cutlass.Float16,cutlass.BFloat16,cutlass.Float32)):
                                            cursors[local_q]=fused_publish(bank_rows.iterator,score,tau_cache[local_q],(kv_pos>=kv_start)&(kv_pos<kv_end),kv_pos,cursors[local_q],bank_span,self.output_dtype,False,False,packed=candidates.element_type==cutlass.Int32)
                                        elif cutlass.const_expr(self.qsa_weights==2 and self.branchless_publish and self.use_relu_trick and self.output_dtype in (cutlass.Float16, cutlass.BFloat16, cutlass.Float32)):
                                            cursors[local_q]=fused_publish(bank_rows.iterator,score,tau_cache[local_q],(kv_pos>=kv_start)&(kv_pos<kv_end),kv_pos,cursors[local_q],bank_span,self.output_dtype,True,packed=candidates.element_type==cutlass.Int32)
                                        else:
                                            cursors[local_q] = publish_bank(score,(kv_pos>=kv_start)&(kv_pos<kv_end),tau_cache[local_q],0,kv_pos,cursors[local_q],bank_rows,self.output_dtype,True,self.branchless_publish,True)
                            kv_idx = kv_idx + self.num_groups
                        for local_q in cutlass.range_constexpr(self.block_q // (self.num_math_threads//128//self.num_groups)):
                            outrow = q_block_idx*self.block_q+query_slot*queries_per_wg+local_q
                            if (cutlass.const_expr(self.full_q_blocks) or outrow<num_q_tokens) and cute.arch.lane_idx()==0:
                                bank_span = (candidates.shape[1]-8)//(4*self.num_groups)
                                bank_id = math_group*4+local_tidx//32
                                cursor = cursors[local_q]
                                candidates[outrow,bank_id] = candidates.element_type(cursor)
                                if cutlass.const_expr(self.num_groups==1):candidates[outrow,bank_id+4]=candidates.element_type(0)
                                if ((cursor&1)!=0)&(cursor<bank_span):candidates[outrow,8+bank_id*bank_span+cursor] = candidates.element_type(0)
                                cute.arch.atomic_add((counts.iterator+outrow).llvm_ptr,cursor-cutlass.Int32(cursor>bank_span)*16777216,sem="relaxed",scope="gpu")
                        q_pipeline.consumer_release(q_cons_state)
                        q_cons_state.advance()
                        next_q_block_idx=q_block_idx+cute.arch.grid_dim()[2]
                        if next_q_block_idx<num_q_blocks:
                            q_last_next=min(next_q_block_idx*self.block_q+self.block_q-1,num_q_tokens-1)
                            next_num_kv=(mCuSeqLenKEnd[q_last_next]+127)//128
                        has_work=next_q_block_idx<num_q_blocks
                else:
                    while has_work:
                        q_block_idx = next_q_block_idx
                        num_kv = next_num_kv
                        q_pipeline.consumer_wait(q_cons_state)
                        q_stage_local = q_cons_state.index
                        queries_per_wg = self.block_q // (self.num_math_threads//128//self.num_groups)
                        cursors = cute.make_rmem_tensor((self.block_q // (self.num_math_threads//128//self.num_groups),), cutlass.Int32)
                        scores = cute.make_rmem_tensor((self.block_q // (self.num_math_threads//128//self.num_groups),), cutlass.Float32)
                        start_cache = cute.make_rmem_tensor((self.block_q // (self.num_math_threads//128//self.num_groups),), cutlass.Int32)
                        end_cache = cute.make_rmem_tensor((self.block_q // (self.num_math_threads//128//self.num_groups),), cutlass.Int32)
                        tau_cache = cute.make_rmem_tensor((self.block_q // (self.num_math_threads//128//self.num_groups),), cutlass.Float32)
                        if cutlass.const_expr(self.num_heads==32):
                            glm_weights=cute.make_rmem_tensor((queries_per_wg*28,),cutlass.Float32)
                        if cutlass.const_expr(self.qsa_weights==1):
                            uniform_cache = cute.make_rmem_tensor((queries_per_wg,),cutlass.Float32)
                        for local_q in cutlass.range_constexpr(self.block_q // (self.num_math_threads//128//self.num_groups)):
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
                                for cached_h in cutlass.range_constexpr(28):glm_weights[local_q*28+cached_h]=sW[(slot*self.num_heads+cached_h,q_stage_local)]
                            if cutlass.const_expr(self.qsa_weights==1):
                                uniform_cache[local_q] = sW[(slot*4,q_stage_local)]
                        kv_idx = _kv_lo_for_block(q_block_idx)
                        # Retire empty capacity tiles without scoring or reordering Q/weight loads.
                        if kv_idx >= num_kv:
                            cute.arch.mbarrier_wait(
                                math_barriers + umma_cons_state_0.index,
                                umma_cons_state_0.phase,
                            )
                            cute.arch.mbarrier_arrive(
                                math_barriers
                                + self.num_umma_stages
                                + umma_cons_state_0.index
                            )
                            umma_cons_state_0.advance()
                        while kv_idx < num_kv:
                            kv_pos = (kv_idx+math_group)*128+m_coord
                            cute.arch.mbarrier_wait(math_barriers+umma_cons_state_0.index,umma_cons_state_0.phase)
                            tAcc = tCtAcc_base_0[(None,None,None,umma_cons_state_0.index)][((None,None),0,0)]
                            tTR = thr_copy_ref.partition_S(cute.flat_divide(tAcc,epi_sub_mn))
                            for local_q in cutlass.range_constexpr(self.block_q // (self.num_math_threads//128//self.num_groups)):
                                slot = query_slot * queries_per_wg + local_q
                                outrow = q_block_idx * self.block_q + slot
                                token_idx = outrow if cutlass.const_expr(self.full_q_blocks) else min(outrow,num_q_tokens-1)
                                if cutlass.const_expr(self.num_heads==32):
                                    cute.copy(tiled_copy_ref,tTR[(None,None,None,0,slot)],tTR_rAcc)
                                    cute.arch.fence_view_async_tmem_load()
                                    if cutlass.const_expr(local_q==queries_per_wg-1):
                                        cute.arch.mbarrier_arrive(math_barriers+self.num_umma_stages+umma_cons_state_0.index)
                                        umma_cons_state_0.advance()
                                else:
                                    if cutlass.const_expr(local_q%(4 if self.branchless_publish else 8)==0):
                                        cute.copy(tiled_copy_ref,tTR[(None,None,None,0,slot//(4 if self.branchless_publish else 8))],tTR_rAcc)
                                        cute.arch.fence_view_async_tmem_load()
                                if cutlass.const_expr(self.num_heads==32):
                                    score=weighted_heads_cached(tTR_rAcc,sW,glm_weights,slot,local_q,q_stage_local,self.num_heads,28,self.use_relu_trick,self.weights_prepared)
                                else:
                                    x4 = _subview(tTR_rAcc,(local_q%(4 if self.branchless_publish else 8))*4,4)
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
                                            w = sW[(slot*4+h,q_stage_local)]
                                        if cutlass.const_expr(self.use_relu_trick and not self.weights_prepared):w = w*0.5
                                        shared_w[h] = w
                                    acc4 = indexer_fma_packed_f32x2(a4,shared_w.load(),zero4,rnd=_RND_RN)
                                    score = (acc4[0]+acc4[1]+acc4[2]+acc4[3])
                                scores[local_q] = score
                            if cutlass.const_expr(self.num_heads!=32):
                                cute.arch.mbarrier_arrive(math_barriers+self.num_umma_stages+umma_cons_state_0.index)
                                umma_cons_state_0.advance()
                            if cutlass.const_expr(self.num_heads==32 and not self.branchless_publish and self.output_dtype in (cutlass.Float16,cutlass.BFloat16,cutlass.Float32)):
                                for pair_q in cutlass.range_constexpr(queries_per_wg//2):
                                    pair_row=q_block_idx*self.block_q+query_slot*queries_per_wg+pair_q*2
                                    bank_span=(candidates.shape[1]-8)//(4*self.num_groups)
                                    bank_id=math_group*4+local_tidx//32
                                    pair_ptr=candidates.iterator+((pair_row*cutlass.Int32(candidates.stride[0])+8+bank_id*bank_span) if cutlass.const_expr(self.offsets32) else (cutlass.Int64(pair_row)*candidates.stride[0]+8+bank_id*bank_span))
                                    if cutlass.const_expr(self.glm_direct_bounds):
                                        bound_end0=end_cache[pair_q*2]
                                        bound_end1=end_cache[pair_q*2+1]
                                        if cutlass.const_expr(not self.full_q_blocks):
                                            if pair_row>=num_q_tokens:bound_end0=cutlass.Int32(0)
                                            if pair_row+1>=num_q_tokens:bound_end1=cutlass.Int32(0)
                                        cursors[pair_q*2],cursors[pair_q*2+1]=fused_publish_pair(pair_ptr,pair_ptr+candidates.stride[0],scores[pair_q*2],scores[pair_q*2+1],tau_cache[pair_q*2],tau_cache[pair_q*2+1],cutlass.Int32(1),cutlass.Int32(1),kv_pos,cursors[pair_q*2],cursors[pair_q*2+1],bank_span,self.output_dtype,True,packed=candidates.element_type==cutlass.Int32,ordered_threshold=self.num_heads==32,bounds=(start_cache[pair_q*2],bound_end0,start_cache[pair_q*2+1],bound_end1))
                                    else:
                                        valid0=(kv_pos>=start_cache[pair_q*2])&(kv_pos<end_cache[pair_q*2])
                                        valid1=(kv_pos>=start_cache[pair_q*2+1])&(kv_pos<end_cache[pair_q*2+1])
                                        if cutlass.const_expr(not self.full_q_blocks):
                                            valid0=valid0&(pair_row<num_q_tokens)
                                            valid1=valid1&(pair_row+1<num_q_tokens)
                                        cursors[pair_q*2],cursors[pair_q*2+1]=fused_publish_pair(pair_ptr,pair_ptr+candidates.stride[0],scores[pair_q*2],scores[pair_q*2+1],tau_cache[pair_q*2],tau_cache[pair_q*2+1],valid0,valid1,kv_pos,cursors[pair_q*2],cursors[pair_q*2+1],bank_span,self.output_dtype,True,packed=candidates.element_type==cutlass.Int32,ordered_threshold=self.num_heads==32)
                            else:
                                for local_q in cutlass.range_constexpr(self.block_q // (self.num_math_threads//128//self.num_groups)):
                                    slot = query_slot * queries_per_wg + local_q
                                    outrow = q_block_idx * self.block_q + slot
                                    token_idx = outrow if cutlass.const_expr(self.full_q_blocks) else min(outrow,num_q_tokens-1)
                                    score = scores[local_q]
                                    if cutlass.const_expr(self.full_q_blocks) or outrow < num_q_tokens:
                                        kv_start = start_cache[local_q]
                                        kv_end = end_cache[local_q]
                                        bank_span = (candidates.shape[1]-8)//(4*self.num_groups)
                                        bank_id = math_group*4+local_tidx//32
                                        bank_rows = cute.make_tensor(candidates.iterator+((outrow*cutlass.Int32(candidates.stride[0])+8+bank_id*bank_span) if cutlass.const_expr(self.offsets32) else (cutlass.Int64(outrow)*candidates.stride[0]+8+bank_id*bank_span)),cute.make_layout((1,bank_span),stride=(bank_span,1)))
                                        if cutlass.const_expr(self.qsa_weights==2 and self.branchless_publish and self.use_relu_trick and self.output_dtype in (cutlass.Float16, cutlass.BFloat16, cutlass.Float32)):
                                            cursors[local_q]=fused_publish(bank_rows.iterator,score,tau_cache[local_q],(kv_pos>=kv_start)&(kv_pos<kv_end),kv_pos,cursors[local_q],bank_span,self.output_dtype,True,packed=candidates.element_type==cutlass.Int32)
                                        else:
                                            cursors[local_q] = publish_bank(score,(kv_pos>=kv_start)&(kv_pos<kv_end),tau_cache[local_q],0,kv_pos,cursors[local_q],bank_rows,self.output_dtype,True,self.branchless_publish,True)
                            kv_idx = kv_idx + self.num_groups
                        for local_q in cutlass.range_constexpr(self.block_q // (self.num_math_threads//128//self.num_groups)):
                            outrow = q_block_idx*self.block_q+query_slot*queries_per_wg+local_q
                            if (cutlass.const_expr(self.full_q_blocks) or outrow<num_q_tokens) and cute.arch.lane_idx()==0:
                                bank_span = (candidates.shape[1]-8)//(4*self.num_groups)
                                bank_id = math_group*4+local_tidx//32
                                cursor = cursors[local_q]
                                candidates[outrow,bank_id] = candidates.element_type(cursor)
                                if cutlass.const_expr(self.num_groups==1):candidates[outrow,bank_id+4]=candidates.element_type(0)
                                if ((cursor&1)!=0)&(cursor<bank_span):candidates[outrow,8+bank_id*bank_span+cursor] = candidates.element_type(0)
                                cute.arch.atomic_add((counts.iterator+outrow).llvm_ptr,cursor-cutlass.Int32(cursor>bank_span)*16777216,sem="relaxed",scope="gpu")
                        q_pipeline.consumer_release(q_cons_state)
                        q_cons_state.advance()
                        next_q_block_idx=q_block_idx+cute.arch.grid_dim()[2]
                        if next_q_block_idx<num_q_blocks:
                            q_last_next=min(next_q_block_idx*self.block_q+self.block_q-1,num_q_tokens-1)
                            next_num_kv=(mCuSeqLenKEnd[q_last_next]+127)//128
                        has_work=next_q_block_idx<num_q_blocks
            if cutlass.const_expr(self.num_heads!=32):tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        else:
            if cutlass.const_expr(self.mode == "filter"):
                if cutlass.const_expr(self.store_registers >= 128):
                    cute.arch.warpgroup_reg_alloc(self.store_registers)
                else:
                    cute.arch.warpgroup_reg_dealloc(self.store_registers)
            else:
                cute.arch.warpgroup_reg_dealloc(self.store_registers)




from .selection import finish_recovery, select_or_prepare_retry as select_candidates

from .selection import runtime_samples, block_samples

from .selection import encode
@cute.jit
def publish_bank(score,active,tau,row,index,cursor,bank_rows,dtype:cutlass.Constexpr,pred_store:cutlass.Constexpr=False,branchless:cutlass.Constexpr=False,raw_half:cutlass.Constexpr=False,unordered:cutlass.Constexpr=False,raw_packed:cutlass.Constexpr=False):
    value=score
    if cutlass.const_expr(not raw_half or dtype not in (cutlass.Float16,cutlass.BFloat16)):value=cutlass.Float32(dtype(score))
    hit=active&(~(value<tau)) if cutlass.const_expr(unordered) else active&((value>=tau)|(value!=value))
    mask=cute.arch.vote_ballot_sync(hit)
    # Warp-uniform empty ballots have no stores or cursor updates.
    next_cursor=cursor
    if cutlass.const_expr(branchless) or mask!=0:
        if cutlass.const_expr(raw_half and dtype in (cutlass.Float16,cutlass.BFloat16) and not raw_packed):value=cutlass.Float32(dtype(score))
        total=cutlass.Int32(cute.arch.popc(mask))
        lane=cute.arch.lane_idx()
        pos=cursor+cutlass.Int32(cute.arch.popc(mask&((cutlass.Uint32(1)<<lane)-1)))
        if cutlass.const_expr(raw_packed):
            predicated_store32(bank_rows.iterator+row*bank_rows.stride[0]+pos,pack_half_index(value,index,dtype),hit&(pos<bank_rows.shape[1]))
        elif cutlass.const_expr(pred_store and bank_rows.element_type!=cutlass.Int32):
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
