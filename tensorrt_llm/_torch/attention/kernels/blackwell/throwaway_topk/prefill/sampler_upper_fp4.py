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
        weight_dtype=cutlass.Float32,
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
        sample_shared_capacity: int = 0,
        math_registers: int = 160,
        store_registers: int = 168,
        dsv4_positions: bool = False,
        dsv4_prepare_sf: bool = False,
        dsv4_bulk_positions: bool = False,
        dsv4_shared_scores: bool = False,
        dsv4_short_top16: bool = False,
        dsv4: bool = False,
        pro_common_domain: bool = False,
        fixed_samples: int = 0,
        qsa_weights: int = 0,
        qsa_unit_scale: bool = False,
        persistent_sample: bool = True,
        sample_sigma: int = 3,
        sample_rank_sigma: float = 0.0,
        sample_count_override: int = 0,
        pdl: bool = False,
        native_bf16: bool = False,
        native_fp16: bool = False,
        input_dtype=cutlass.Float4E2M1FN,
    ):
        # Static FP4 invariants imposed by the TMEM and SMEM layouts below.
        self.pdl = bool(pdl)
        assert input_dtype in (cutlass.Float4E2M1FN, cutlass.Float8E4M3FN)
        self.input_dtype = input_dtype
        self.native_lowp = bool(native_bf16 or native_fp16)
        assert not (native_bf16 and native_fp16)
        assert not native_fp16 or output_dtype == cutlass.Float16
        assert not self.native_lowp or (num_heads in (4, 32, 64) and output_dtype == (cutlass.Float16 if native_fp16 else cutlass.BFloat16) and not use_relu_trick)
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
        self.mode, self.top_k, self.capacity = mode, top_k, capacity
        self.splits, self.packed = splits, packed
        self.sample_size = sample_size
        self.dsv4_positions = dsv4_positions
        self.dsv4_prepare_sf = dsv4_prepare_sf
        self.dsv4_bulk_positions = dsv4_bulk_positions
        self.dsv4_shared_scores = dsv4_shared_scores
        self.dsv4_short_top16 = dsv4_short_top16
        self.dsv4 = dsv4
        self.pro_common_domain = pro_common_domain
        self.fixed_samples = fixed_samples
        self.qsa_weights = qsa_weights
        self.qsa_unit_scale = qsa_unit_scale
        self.persistent_sample = persistent_sample
        self.sample_sigma = sample_sigma
        self.sample_rank_sigma = sample_rank_sigma
        self.sample_count_override = sample_count_override
        self.math_registers, self.store_registers = math_registers, store_registers
        self.block_kv = block_kv
        self.num_heads = num_heads
        self.head_dim = head_dim
        # block_q: N = block_q * num_heads tokens per MMA tile.
        # Blackwell (SM100): TMEM = 512 cols; acc(N=256)+SFA+SFB = 544 > 512, so N=128.
        # Rubin (SM107/109): TMEM = 576/832 cols; N=256 fits with room to spare.
        is_rubin = _target_is_rubin()
        self.block_q = (64 if num_heads==4 else 256) // num_heads
        if mode=='filter' and ((num_heads==64 and self.block_q!=2) or splits!=1):
            raise ValueError('two-warp prefill publisher requires block_q=2 and splits=1')
        self.N = self.block_q * num_heads  # 128 on Blackwell, 256 on Rubin
        self.num_sms = num_sms
        self.num_epi_subtiles = num_epi_subtiles
        self.epi_dtype = epi_dtype
        self.weight_dtype = weight_dtype
        self.weight_bytes = 2 if weight_dtype == cutlass.BFloat16 else 4
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
        w_stage_bytes = self.N * self.weight_bytes
        self.w_stage_stride = ((w_stage_bytes + 127) // 128 * 128) // self.weight_bytes
        self.output_dtype = output_dtype
        if num_epi_subtiles > 1 and num_heads % num_epi_subtiles != 0:
            raise ValueError("num_heads must be divisible by num_epi_subtiles")
        if (num_heads // num_epi_subtiles) % 4 != 0:
            raise ValueError(
                "num_heads // num_epi_subtiles must be divisible by 4 (FMA unroll granularity)"
            )
        self.num_groups = 1
        self.sample_shared_capacity = sample_shared_capacity
        self.glm_persistent = num_heads==32 and mode=="sample"

        self.num_math_threads = 512
        self.num_specialized_threads = 128
        self.threads_per_cta = 640
        self.num_math_warps = 16
        self.tma_warp_base = 16
        self.umma_warp_base = 18

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
        self.umma_inst_k = 64 if input_dtype == cutlass.Float4E2M1FN else 32
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
        recovery_queue: cute.Tensor,
        num_q_tokens: cutlass.Int32,
        num_kv_tokens: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        half_head_dim = self.head_dim * self.input_dtype.width // 8

        # Recast KV to FP4. Each uint8 byte becomes 2 FP4 elements,
        # so layout positions and the iterator scale accordingly.
        kv_fp4 = cute.recast_tensor(kv, self.input_dtype)

        # Q (b) was passed as uint8 (FP4-packed bytes); recast to FP4 so MMA
        # type inference and TMA descriptors are correct.
        b = cute.recast_tensor(b, self.input_dtype)

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
        w_copy_size = self.N * self.weight_bytes
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
        if cutlass.const_expr(self.num_heads==4 or (self.num_heads==32 and not self.glm_persistent)):num_ctas = num_q_blocks*self.splits

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
            weights,
            tma_atom_sf_kv,
            tma_tensor_sf_kv,
            tma_atom_sf_q,
            tma_tensor_sf_q,
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
            self.sf_kv_smem_layout_staged,
            self.sf_q_smem_layout_staged,
            self.sf_q_tma_smem_layout_staged,
            self.tmem_sfa_layout,
            self.tmem_sfb_layout,
            self.sfa_chunk_smem_layout,
            self.sfb_chunk_smem_layout,
            self.copy_atom_s2t,
            self.epi_tile,
            kv, kv_scales,
            SharedStorage,
        ).launch(
            grid=(1, 1, num_ctas),
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=1,
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
        mW_raw: cute.Tensor,
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
        recovery_queue: cute.Tensor,
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
        gather_kv: cute.Tensor, gather_scales: cute.Tensor, SharedStorage: cutlass.Constexpr,
    ):
        pdl_prologue(self.pdl)
        if cutlass.const_expr(self.dsv4_prepare_sf):
            # Transpose each 4x32 int32 scale atom into the 32x4 UTCCP layout.
            # Refresh the workspace before K2 without modifying input scales.
            tid=cute.arch.thread_idx()[0]
            lane=cute.arch.lane_idx()
            warp_global=cute.arch.block_idx()[2]*(self.threads_per_cta//32)+tid//32
            warp_stride=cute.arch.grid_dim()[2]*(self.threads_per_cta//32)
            offset=(num_q_tokens+31)//32*32
            if cutlass.const_expr(self.num_heads==32):offset+=((num_q_tokens+3)//4*4)*32
            prepared=cute.make_tensor(cute.recast_ptr(thresholds.iterator,dtype=cutlass.Int32),cute.make_layout(thresholds.shape[0]))
            copy_atom=cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),cutlass.Int32,num_bits_per_copy=128)
            for tile in cutlass.range(warp_global,num_kv_tokens//128,warp_stride):
                values=cute.make_rmem_tensor((4,),cutlass.Int32)
                for j in cutlass.range_constexpr(4):values[j]=gather_scales[tile*128+j*32+lane]
                destination=cute.make_tensor((prepared.iterator+offset+tile*128+lane*4).align(16),cute.make_layout(4))
                cute.copy(copy_atom,values,destination)
        if cute.arch.thread_idx()[0]==0 and cute.arch.block_idx()[2]==0:
            recovery_queue[0]=0
            recovery_queue[recovery_queue.shape[0]-1]=0
        self._gemm_body(tiled_mma, tma_atom_a, mA_mkl, tma_atom_b, mB_nkl, tma_atom_w, mW_tma, mW_raw, tma_atom_sf_kv, mSF_KV_tma, tma_atom_sf_q, mSF_Q_tma, mLogits, mCuSeqLenKStart, mCuSeqLenKEnd, thresholds, counts, candidates, output, num_q_tokens, num_q_blocks, num_kv_tokens, cluster_layout_vmnk, a_smem_layout_staged, a_tma_smem_layout, b_smem_layout_staged, w_smem_layout_staged, sf_kv_smem_layout_staged, sf_q_smem_layout_staged, sf_q_tma_smem_layout_staged, tmem_sfa_layout, tmem_sfb_layout, sfa_chunk_smem_layout, sfb_chunk_smem_layout, copy_atom_s2t, epi_tile, gather_kv, gather_scales, SharedStorage)

    @cute.jit
    def _gemm_body(self, tiled_mma: cute.TiledMma, tma_atom_a: cute.CopyAtom, mA_mkl: cute.Tensor, tma_atom_b: cute.CopyAtom, mB_nkl: cute.Tensor, tma_atom_w: cute.CopyAtom, mW_tma: cute.Tensor, mW_raw: cute.Tensor, tma_atom_sf_kv: cute.CopyAtom, mSF_KV_tma: cute.Tensor, tma_atom_sf_q: cute.CopyAtom, mSF_Q_tma: cute.Tensor, mLogits: cute.Tensor, mCuSeqLenKStart: cute.Tensor, mCuSeqLenKEnd: cute.Tensor, thresholds: cute.Tensor, counts: cute.Tensor, candidates: cute.Tensor, output: cute.Tensor, num_q_tokens: cutlass.Int32, num_q_blocks: cutlass.Int32, num_kv_tokens: cutlass.Int32, cluster_layout_vmnk: cute.Layout, a_smem_layout_staged: cute.ComposedLayout, a_tma_smem_layout: cute.ComposedLayout, b_smem_layout_staged: cute.ComposedLayout, w_smem_layout_staged: cute.Layout, sf_kv_smem_layout_staged: cute.Layout, sf_q_smem_layout_staged: cute.Layout, sf_q_tma_smem_layout_staged: cute.Layout, tmem_sfa_layout: cute.Layout, tmem_sfb_layout: cute.Layout, sfa_chunk_smem_layout: cute.Layout, sfb_chunk_smem_layout: cute.Layout, copy_atom_s2t: cute.CopyAtom, epi_tile: cute.Tile, gather_kv: cute.Tensor, gather_scales: cute.Tensor, SharedStorage: cutlass.Constexpr):
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
        is_math_warp = warp_idx < 16
        is_tma_warp_0 = warp_idx == 16
        is_tma_warp_1 = warp_idx == 17
        is_tma_warp = is_tma_warp_0 | is_tma_warp_1
        is_umma_warp_0 = warp_idx == 18
        is_umma_warp_1 = warp_idx == 19

        # Grid-stride scheduler: this SM handles q_block_idx = sm_idx,
        # sm_idx+num_sms, ...
        NUM_MATH_WG = 1  # kNumMathWarpGroups
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
            current_num_kv = (cutlass.Int32(self.fixed_samples) if cutlass.const_expr(self.fixed_samples>0) else runtime_samples((num_kv_tokens if cutlass.const_expr(self.pro_common_domain) else mCuSeqLenKEnd[q_last_token_clamped]),self.top_k,heads=self.num_heads,coverage=4 if self.qsa_weights==2 or self.num_heads==32 else 8,dsv4=self.dsv4))//128
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
        kv_pipeline_0=pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.kv_mbar_0.data_ptr(),
            num_stages=self.num_kv_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread,64),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread,1),
            defer_sync=True)
        kv_pipeline_1=pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.kv_mbar_1.data_ptr(),
            num_stages=self.num_kv_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread,64),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread,1),
            defer_sync=True)

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

        # Both UMMA warps must finish reading SFB before warp 0 overwrites it.
        # Each warp commits its final MMA to this single-stage mbarrier; warp 0
        # waits for both commits. A named barrier orders issue, not MMA completion.
        sfb_war_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.sfb_war_mbar.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
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
        tmem_alloc_num_threads = 544  # 10 warps: warp 0-7 (math) + warp 10-11 (umma)
        tmem_alloc_barrier = SubsetNamedBarrier(
            barrier_id=1, num_threads=tmem_alloc_num_threads
        )
        # Warp 0 copies Q scales into shared SFB TMEM; warp 1 reads that region.
        # Synchronize both UMMA warps after the copy: the Q pipeline orders TMA-to-SMEM,
        # not cross-warp TMEM visibility.
        sfb_sync_barrier = SubsetNamedBarrier(barrier_id=2, num_threads=32)
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
            sample_hist=smem.allocate_tensor(cutlass.Int32,cute.make_layout(4096 if self.num_heads in (4,32) else 1056),byte_alignment=128)
            sample_scalar=smem.allocate_tensor(cutlass.Int32,cute.make_layout(8),byte_alignment=16)
        # H4 samples are capped at1024. Each producer writes a unique slot;
        # math reads it after the KV/UMMA acquire, and the whole Q tile owns
        # the array until all math consumers release Q. No ring reuse race.
        if cutlass.const_expr(self.num_heads in (4,32) or self.dsv4_positions):
            sample_positions=smem.allocate_tensor(cutlass.Int32,cute.make_layout(1024*(self.num_q_stages if self.glm_persistent or self.dsv4_positions else 1)),byte_alignment=128)
        if cutlass.const_expr(self.dsv4_shared_scores):
            dsv4_scores=smem.allocate_tensor(self.output_dtype,cute.make_layout((self.block_q,1024),stride=(1024,1)),byte_alignment=128)
        if cutlass.const_expr(self.sample_shared_capacity > 0):
            sample_scores=smem.allocate_tensor(self.output_dtype,cute.make_layout((self.block_q,self.sample_shared_capacity),stride=(self.sample_shared_capacity,1)),byte_alignment=128)
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # SMEM allocation: per-group KV + shared Q
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
        # SF_Q SMEM (int32, packed UE8M0; flat layout, transposed
        # in-place by UMMA warp before UTCCP).
        sSF_Q = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=sf_q_smem_layout_staged,
            byte_alignment=128,
        )
        # Weights SMEM: [N, num_q_stages], shared Q barrier
        sW = smem.allocate_tensor(
            element_type=self.weight_dtype,
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
        sSF_KV_1 = sSF_KV_0

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
            # Two gather warps, one Q TMA owner.
            # TMA warp 0: loads Q+SF_Q+Weights (on Q-block change) + KV group 0
            cute.arch.warpgroup_reg_dealloc(self.store_registers)

            if is_tma_warp_0:
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

            sample_q_stage = cutlass.Int32(0)
            while has_work:
                q_block_idx_old = prev_q_block_idx
                prev_q_block_idx = next_q_block_idx
                q_block_idx = next_q_block_idx
                kv_idx = next_kv_idx
                num_kv = next_num_kv

                if cutlass.const_expr(self.glm_persistent or self.dsv4_positions):
                    if q_block_idx != q_block_idx_old:
                        if q_block_idx_old < num_q_blocks:
                            sample_q_stage=(sample_q_stage+1)%self.num_q_stages
                        sample_first=q_block_idx*self.block_q
                        sample_lower=mCuSeqLenKStart[min(sample_first,num_q_tokens-1)]
                        for token in cutlass.range_constexpr(1,self.block_q):
                            sample_lower=min(sample_lower,mCuSeqLenKStart[min(sample_first+token,num_q_tokens-1)])
                        sample_upper=mCuSeqLenKEnd[min(sample_first+self.block_q-1,num_q_tokens-1)]
                        if cutlass.const_expr(self.pro_common_domain):
                            sample_lower=cutlass.Int32(0)
                            sample_upper=num_kv_tokens

                # On Q-block change: prefetch Q for the NEXT Q block
                if (q_block_idx != q_block_idx_old) and is_tma_warp_0:
                    prefetch_next_q = q_block_idx + cute.arch.grid_dim()[2] // self.splits
                    if prefetch_next_q < num_q_blocks:
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

                if cutlass.const_expr(self.glm_persistent or self.dsv4_positions):
                    if q_block_idx != q_block_idx_old:
                        # Only warp 0 acquires Q stages. Warp 1 must not write
                        # a recycled position buffer before that acquisition.
                        subset_barrier(barrier_id=3,number_of_threads=64)

                # Load KV + SF for group 0: block at kv_idx (contiguous,
                # no block_table)
                kv_pipeline_0.producer_acquire(kv_prod_state_0)
                stage=kv_prod_state_0.index
                data_atom=cute.make_copy_atom(cpasync.CopyG2SOp(),self.input_dtype,num_bits_per_copy=128)
                scale_atom=cute.make_copy_atom(cpasync.CopyG2SOp(),cutlass.Int32,num_bits_per_copy=32)
                lane=cute.arch.lane_idx()
                if cutlass.const_expr(self.dsv4_bulk_positions):
                    for position_half in cutlass.range_constexpr(2):
                        owned_row=(warp_idx-self.tma_warp_base)*64+position_half*32+lane
                        sample_positions[sample_q_stage*1024+kv_idx*128+owned_row]=dsv4_cached_sample_key(kv_idx*128+owned_row,0 if cutlass.const_expr(self.pro_common_domain) else q_block_idx,sample_lower,sample_upper,self.top_k,coverage=4)
                    cute.arch.sync_warp()
                for chunk in cutlass.range(8):
                    local_row=chunk*8+lane//4+(warp_idx-self.tma_warp_base)*64
                    if cutlass.const_expr(self.num_heads in (4,32)):
                        physical = shape_sample_key(self.sample_count_override,kv_idx*128+local_row,q_block_idx,sample_lower,sample_upper,self.top_k,heads=self.num_heads,coverage=4 if self.qsa_weights==2 or self.num_heads==32 else 8,common=(self.num_heads==32 and self.top_k==2048))
                        position_index=kv_idx*128+local_row
                        if cutlass.const_expr(self.glm_persistent or self.dsv4_positions):position_index+=sample_q_stage*1024
                        if lane%4==0:sample_positions[position_index]=physical
                    else:
                        if cutlass.const_expr(self.dsv4_positions):
                            if cutlass.const_expr(self.dsv4_bulk_positions):physical=sample_positions[sample_q_stage*1024+kv_idx*128+local_row]
                            else:
                                physical=dsv4_cached_sample_key(kv_idx*128+local_row,0 if cutlass.const_expr(self.pro_common_domain) else q_block_idx,sample_lower,sample_upper,self.top_k,coverage=4)
                            if (not self.dsv4_bulk_positions) and lane%4==0:sample_positions[sample_q_stage*1024+kv_idx*128+local_row]=physical
                        else:
                            physical=individual_sample_key(kv_idx*128+local_row,q_block_idx,mCuSeqLenKStart,mCuSeqLenKEnd,num_q_tokens,self.block_q,self.top_k,dsv4=self.dsv4)
                    source_fp4=cute.recast_tensor(gather_kv,self.input_dtype)
                    src=cute.local_tile(source_fp4[physical,None],(32,),(lane%4,))
                    row_view=cute.slice_(sKV_0,((local_row,None),0,(lane%4)//(self.umma_inst_k//32),stage))
                    dst=cute.local_tile(row_view,(32,),((lane%4)%(self.umma_inst_k//32),))
                    if cutlass.const_expr(self.input_dtype == cutlass.Float8E4M3FN):
                        # CopyG2S atom moves 128 bits per invocation. A lane's
                        # 32 FP8 elements need two copies (FP4 needed one).
                        for part in cutlass.range_constexpr(2):
                            cute.copy(data_atom, cute.local_tile(src, (16,), (part,)),
                                      cute.local_tile(dst, (16,), (part,)))
                    else:
                        cute.copy(data_atom,src,dst)
                    if lane%4==0:
                        src_scale=cute.local_tile(gather_scales,(1,),(physical,))
                        dst_scale=cute.local_tile(sSF_KV_0[None,stage],(1,),(local_row,))
                        cute.copy(scale_atom,src_scale,dst_scale)
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_warp()
                cute.arch.fence_view_async_shared()
                kv_pipeline_0.producer_commit(kv_prod_state_0)
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
                        next_num_kv = ((cutlass.Int32(self.fixed_samples) if cutlass.const_expr(self.fixed_samples>0) else runtime_samples((num_kv_tokens if cutlass.const_expr(self.pro_common_domain) else mCuSeqLenKEnd[q_last_next]),self.top_k,heads=self.num_heads,coverage=4 if self.qsa_weights==2 or self.num_heads==32 else 8,dsv4=self.dsv4))//128 if cutlass.const_expr(self.mode=="sample") else ((mCuSeqLenKEnd[q_last_next] + block_kv_val - 1) // block_kv_val))
                has_work = next_q_block_idx < num_q_blocks

        elif is_umma_warp_0:
            # Wait for Q independently of KV: separate TMA barriers do not order
            # one another's SMEM writes.
            cute.arch.warpgroup_reg_dealloc(self.store_registers)

            # TMEM: wait for math warp 0's allocation, retrieve pointer
            subset_barrier(barrier_id=1,number_of_threads=544)
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tmem.relinquish_alloc_permit()
            tCtAcc_base_0 = cute.make_tensor(tmem_ptr, tCtAcc_fake_staged.layout)
            tCtAcc_base_1 = tCtAcc_base_0

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
                        num_kv = (cutlass.Int32(self.fixed_samples) if cutlass.const_expr(self.fixed_samples>0) else block_samples(q_block_idx,mCuSeqLenKEnd,num_q_tokens,block_q_val,self.top_k,heads=self.num_heads,coverage=4 if self.qsa_weights==2 or self.num_heads==32 else 8,dsv4=self.dsv4,anchor_upper=(num_kv_tokens if cutlass.const_expr(self.pro_common_domain) else 0)))//128

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
                        subset_barrier(barrier_id=2,number_of_threads=32)

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
                    subset_barrier(barrier_id=2,number_of_threads=32)

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
                            next_num_kv = ((cutlass.Int32(self.fixed_samples) if cutlass.const_expr(self.fixed_samples>0) else runtime_samples((num_kv_tokens if cutlass.const_expr(self.pro_common_domain) else mCuSeqLenKEnd[q_last_next]),self.top_k,heads=self.num_heads,coverage=4 if self.qsa_weights==2 or self.num_heads==32 else 8,dsv4=self.dsv4))//128 if cutlass.const_expr(self.mode=="sample") else ((mCuSeqLenKEnd[q_last_next] + block_kv_val - 1) // block_kv_val))
                    has_work = next_q_block_idx < num_q_blocks

        elif is_math_warp:
            cute.arch.warpgroup_reg_alloc(self.math_registers)

            # TMEM: math warp 0 is the allocator; all math warps wait + retrieve
            tmem.allocate(num_tmem_alloc_cols_total)
            subset_barrier(barrier_id=1,number_of_threads=544)
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tmem.relinquish_alloc_permit()
            tCtAcc_base_0 = cute.make_tensor(tmem_ptr, tCtAcc_fake_staged.layout)
            tCtAcc_base_1 = tCtAcc_base_0

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
            w_cache = cute.make_rmem_tensor((32 if self.dsv4 else 56,),cutlass.Float32)
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
                    if cutlass.const_expr(self.native_lowp):
                        native_weights = cute.make_rmem_tensor((32,), cutlass.Uint32)
                        for h in cutlass.range_constexpr(32):
                            native_weights[h] = native_weight_pair(sW,query_slot*64+2*h,q_stage_local,self.output_dtype,fp8=False)
                    else:
                        for h in cutlass.range_constexpr(32 if self.dsv4 else 56):
                            w = cutlass.Float32(sW[(query_slot*64+h,q_stage_local)])
                            if cutlass.const_expr(self.use_relu_trick):w = w*0.5
                            w_cache[h] = w
                    if outrow<num_q_tokens:
                        if local_tidx==0:counts[outrow]=0
                        if cutlass.const_expr(not self.dsv4_shared_scores):
                            for col in cutlass.range(local_tidx,512,128):mLogits[outrow,col]=self.output_dtype(float('-inf'))
                        for col in cutlass.range(local_tidx,256,128):candidates[outrow,col]=self.output_dtype(float('-inf'))
                    kv_idx = _kv_lo_for_block(q_block_idx)
                    # Every producer and both math WGs traverse the same KV tiles.
                    while kv_idx < num_kv:
                        sample_pos=kv_idx*128+m_coord
                        if cutlass.const_expr(self.num_heads in (4,32)):
                            kv_pos = shape_sample_key(self.sample_count_override,sample_pos,q_block_idx,sample_lower,sample_upper,self.top_k,heads=self.num_heads,coverage=4 if self.qsa_weights==2 or self.num_heads==32 else 8,common=(self.num_heads==32 and self.top_k==2048))
                        else:
                            kv_pos=cutlass.Int32(0)
                            if cutlass.const_expr(not self.dsv4_positions):
                                kv_pos=individual_sample_key(sample_pos,q_block_idx,mCuSeqLenKStart,mCuSeqLenKEnd,num_q_tokens,self.block_q,self.top_k,dsv4=self.dsv4)
                        umma_ready=umma_pipeline_0.consumer_try_wait(umma_cons_state_0)
                        umma_pipeline_0.consumer_wait(umma_cons_state_0,umma_ready)
                        if cutlass.const_expr(self.dsv4_positions):kv_pos=sample_positions[q_stage_local*1024+sample_pos]
                        # Both query WGs have loaded scales, and UMMA completion
                        # proves it has finished reading the same KV stage.
                        tAcc = tCtAcc_base_0[(None,None,None,umma_cons_state_0.index)][((None,None),0,0)]
                        tTR = thr_copy_ref.partition_S(cute.flat_divide(tAcc,epi_sub_mn))
                        if cutlass.const_expr(self.native_lowp):
                            s_acc = cute.make_rmem_tensor((2,), cutlass.Uint32)
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
                                    x4 = _subview(tTR_rAcc,h,4)
                                    woff = (i*(64//num_epi_subtiles)+h)//2
                                    s_acc[0]=native_fma(self.output_dtype,native_pack_relu(self.output_dtype,x4[0],x4[1]),native_weights[woff],s_acc[0])
                                    s_acc[1]=native_fma(self.output_dtype,native_pack_relu(self.output_dtype,x4[2],x4[3]),native_weights[woff+1],s_acc[1])
                            else:
                                for h in cutlass.range_constexpr(0,64//num_epi_subtiles,4):
                                    x4 = _subview(tTR_rAcc,h,4)
                                    if cutlass.const_expr(self.use_relu_trick):
                                        a4 = indexer_add_packed_f32x2(x4,abs(x4),rnd=_RND_RN)
                                    else:
                                        a4 = cute.max(x4,zero4,propagate_nan=True)
                                    if cutlass.const_expr(i*(64//num_epi_subtiles)+h<(32 if self.dsv4 else 56)):
                                        w4 = _subview(w_cache,i*(64//num_epi_subtiles)+h,4)
                                    else:
                                        shared_w = cute.make_rmem_tensor((4,),cutlass.Float32)
                                        for j in cutlass.range_constexpr(4):
                                            w=cutlass.Float32(sW[(query_slot*64+i*(64//num_epi_subtiles)+h+j,q_stage_local)])
                                            if cutlass.const_expr(self.use_relu_trick):w=w*0.5
                                            shared_w[j]=w
                                        w4=shared_w.load()
                                    s_acc = indexer_fma_packed_f32x2(a4,w4,s_acc,rnd=_RND_RN)
                        if cutlass.const_expr(self.native_lowp):
                            reduced = native_add(self.output_dtype,s_acc[0],s_acc[1])
                            score = cutlass.Float32(self.output_dtype(native_lo(self.output_dtype,reduced)+native_hi(self.output_dtype,reduced)))
                        else:
                            score = (s_acc[0]+s_acc[1]+s_acc[2]+s_acc[3])
                        if outrow<num_q_tokens:
                            rounded=self.output_dtype(float('-inf'))
                            if (kv_pos>=kv_start)&(kv_pos<kv_end):rounded=self.output_dtype(score)
                            if cutlass.const_expr(self.dsv4_shared_scores):dsv4_scores[query_slot,sample_pos]=rounded
                            else:mLogits[outrow,sample_pos]=rounded
                            if kv_idx<2:candidates[outrow,sample_pos]=rounded
                        kv_idx=kv_idx+1
                    subset_barrier(barrier_id=cutlass.Int32(10)+query_slot,number_of_threads=128)
                    if outrow<num_q_tokens:
                        hist=cute.make_tensor(sample_hist.iterator+query_slot*264,cute.make_layout(264))
                        scalar=cute.make_tensor(sample_scalar.iterator+query_slot*2,cute.make_layout(2))
                        samples=(cutlass.Int32(self.fixed_samples) if cutlass.const_expr(self.fixed_samples>0) else block_samples(q_block_idx,mCuSeqLenKEnd,num_q_tokens,self.block_q,self.top_k,heads=self.num_heads,coverage=4 if self.qsa_weights==2 or self.num_heads==32 else 8,dsv4=self.dsv4,anchor_upper=(num_kv_tokens if cutlass.const_expr(self.pro_common_domain) else 0)))
                        rank=model_proposal_rank((max(1,num_kv_tokens) if cutlass.const_expr(self.pro_common_domain) else max(1,kv_end-kv_start)),samples,self.top_k,self.sample_sigma,self.num_heads,dsv4=self.dsv4)
                        if cutlass.const_expr(self.sample_rank_sigma>0):
                            rank=shape_margin_rank(max(1,kv_end-kv_start),samples,self.top_k,self.sample_rank_sigma)
                        # Use a quantile only when enough samples are expected above it;
                        # sparse tails use the maximum.
                        high_rank=min(samples,max(1,self.top_k*samples//(2*(max(1,num_kv_tokens) if cutlass.const_expr(self.pro_common_domain) else max(1,kv_end-kv_start)))))
                        if high_rank<8:high_rank=cutlass.Int32(1)
                        if cutlass.const_expr(self.dsv4_shared_scores):
                            propose_parallel(outrow,local_tidx,dsv4_scores,hist,scalar,thresholds,samples,rank,cutlass.Int32(10)+query_slot,high_rank,candidates,ignore_padding=self.dsv4,extend_high=self.dsv4,short_top16=self.dsv4_short_top16,narrow_range=(samples*self.top_k>=6*(max(1,num_kv_tokens) if cutlass.const_expr(self.pro_common_domain) else max(1,kv_end-kv_start))),score_row=query_slot)
                        else:
                            propose_parallel(outrow,local_tidx,mLogits,hist,scalar,thresholds,samples,rank,cutlass.Int32(10)+query_slot,high_rank,candidates,ignore_padding=self.dsv4,extend_high=self.dsv4,short_top16=self.dsv4_short_top16,narrow_range=(samples*self.top_k>=6*(max(1,num_kv_tokens) if cutlass.const_expr(self.pro_common_domain) else max(1,kv_end-kv_start))))
                    q_pipeline.consumer_release(q_cons_state)
                    q_cons_state.advance()
                    next_q_block_idx=q_block_idx+cute.arch.grid_dim()[2]
                    if next_q_block_idx<num_q_blocks:
                        q_last_next=min(next_q_block_idx*self.block_q+self.block_q-1,num_q_tokens-1)
                        next_num_kv=(cutlass.Int32(self.fixed_samples) if cutlass.const_expr(self.fixed_samples>0) else runtime_samples((num_kv_tokens if cutlass.const_expr(self.pro_common_domain) else mCuSeqLenKEnd[q_last_next]),self.top_k,heads=self.num_heads,coverage=4 if self.qsa_weights==2 or self.num_heads==32 else 8,dsv4=self.dsv4))//128
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
                            # All other retained slots are overwritten by the sampler.
                            if local_tidx == 0:counts[outrow] = 0
                            if cutlass.const_expr(self.use_relu_trick):
                                if local_tidx<self.num_heads:
                                    offset=(num_q_tokens+31)//32*32+outrow*self.num_heads+local_tidx
                                    raw=cute.make_tensor(mW_raw.iterator,cute.make_layout(cute.size(mW_raw)))
                                    thresholds[offset]=raw[outrow*self.num_heads+local_tidx]*0.5
                    kv_idx = _kv_lo_for_block(q_block_idx)
                    while kv_idx < num_kv:
                        sample_pos = kv_idx*128 + m_coord
                        if cutlass.const_expr(self.num_heads in (4,32)):
                            kv_pos = cutlass.Int32(0)
                        else:
                            kv_pos = individual_sample_key(sample_pos,q_block_idx,mCuSeqLenKStart,mCuSeqLenKEnd,num_q_tokens,self.block_q,self.top_k,dsv4=self.dsv4)
                        umma_pipeline_0.consumer_wait(umma_cons_state_0)
                        if cutlass.const_expr(self.num_heads in (4,32)):
                            position_index=sample_pos
                            if cutlass.const_expr(self.glm_persistent):position_index+=q_stage_local*1024
                            kv_pos=sample_positions[position_index]
                        tAcc = tCtAcc_base_0[(None,None,None,umma_cons_state_0.index)][((None,None),0,0)]
                        tTR = thr_copy_ref.partition_S(cute.flat_divide(tAcc,epi_sub_mn))
                        for local_q in cutlass.range_constexpr(self.block_q // 4):
                            slot = query_slot * queries_per_wg + local_q
                            outrow = q_block_idx * self.block_q + slot
                            token_idx = min(outrow,num_q_tokens-1)
                            cute.copy(tiled_copy_ref,tTR[(None,None,None,0,slot)],tTR_rAcc)
                            cute.arch.fence_view_async_tmem_load()
                            if cutlass.const_expr(local_q == self.block_q // 4 - 1):
                                umma_pipeline_0.consumer_release(umma_cons_state_0)
                                umma_cons_state_0.advance()
                            if cutlass.const_expr(self.num_heads==32):
                                score=weighted_heads(tTR_rAcc,sW,slot,q_stage_local,self.num_heads,self.use_relu_trick,False,native_dtype=self.output_dtype if self.native_lowp else None,fp8=False)
                            else:
                                x4 = _subview(tTR_rAcc,0,4)
                                if cutlass.const_expr(self.use_relu_trick):
                                    a4 = indexer_add_packed_f32x2(x4,abs(x4),rnd=_RND_RN)
                                else:
                                    a4 = cute.max(x4,zero4,propagate_nan=True)
                                shared_w = cute.make_rmem_tensor((4,),cutlass.Float32)
                                for h in cutlass.range_constexpr(4):
                                    w = cutlass.Float32(0.0883883461356163) if cutlass.const_expr(self.qsa_weights==2) else sW[(slot*4+h,q_stage_local)]
                                    if cutlass.const_expr(self.use_relu_trick ):w = w*0.5
                                    shared_w[h] = w
                                if cutlass.const_expr(self.native_lowp):
                                    score=head_sum(native_head_fma4(x4,shared_w.load(),head_zero(self.output_dtype),self.output_dtype,False),self.output_dtype)
                                else:
                                    acc4 = indexer_fma_packed_f32x2(a4,shared_w.load(),zero4,rnd=_RND_RN)
                                    score = (acc4[0]+acc4[1]+acc4[2]+acc4[3])
                            if outrow < num_q_tokens:
                                kv_start = mCuSeqLenKStart[token_idx]
                                kv_end = mCuSeqLenKEnd[token_idx]
                                rounded = self.output_dtype(float('-inf'))
                                if (kv_pos>=kv_start)&(kv_pos<kv_end):rounded = self.output_dtype(score)
                                if cutlass.const_expr(self.sample_shared_capacity > 0):sample_scores[slot,sample_pos]=rounded
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
                        samples = (cutlass.Int32(self.fixed_samples) if cutlass.const_expr(self.fixed_samples>0) else block_samples(q_block_idx,mCuSeqLenKEnd,num_q_tokens,self.block_q,self.top_k,heads=self.num_heads,coverage=4 if self.qsa_weights==2 or self.num_heads==32 else 8,dsv4=self.dsv4,anchor_upper=(num_kv_tokens if cutlass.const_expr(self.pro_common_domain) else 0)))
                        rank = model_proposal_rank((max(1,num_kv_tokens) if cutlass.const_expr(self.pro_common_domain) else max(1,kv_end-kv_start)),samples,self.top_k,self.sample_sigma,self.num_heads,dsv4=self.dsv4)
                        if cutlass.const_expr(self.sample_rank_sigma>0):
                            rank=shape_margin_rank(max(1,kv_end-kv_start),samples,self.top_k,self.sample_rank_sigma)
                        if cutlass.const_expr(self.qsa_weights==2):
                            if (kv_end-kv_start>32*self.top_k)&(kv_end-kv_start<=128*self.top_k):
                                rank=model_proposal_rank((max(1,num_kv_tokens) if cutlass.const_expr(self.pro_common_domain) else max(1,kv_end-kv_start)),samples,self.top_k,5.5,self.num_heads,dsv4=self.dsv4)
                                if cutlass.const_expr(self.sample_rank_sigma>0):
                                    rank=shape_margin_rank(max(1,kv_end-kv_start),samples,self.top_k,self.sample_rank_sigma)
                        high_rank = min(samples,max(1,self.top_k*samples//(2*(max(1,num_kv_tokens) if cutlass.const_expr(self.pro_common_domain) else max(1,kv_end-kv_start)))))
                        if high_rank<8:high_rank=cutlass.Int32(1)
                        if cutlass.const_expr(self.sample_shared_capacity > 0):
                            if cutlass.const_expr(self.fixed_samples == 128 and self.num_heads == 32 and self.top_k == 512):
                                propose_warp_128(outrow,cute.arch.lane_idx(),sample_scores,hist,thresholds,samples,rank,high_rank,candidates,extend_short=self.qsa_weights==2 or self.num_heads==32,score_row=slot)
                            else:
                                propose_warp(outrow,cute.arch.lane_idx(),sample_scores,hist,thresholds,samples,rank,high_rank,candidates,extend_short=self.qsa_weights==2 or self.num_heads==32,score_row=slot)
                        else:
                            if cutlass.const_expr(self.fixed_samples == 128 and self.num_heads == 32 and self.top_k == 512):
                                propose_warp_128(outrow,cute.arch.lane_idx(),mLogits,hist,thresholds,samples,rank,high_rank,candidates,extend_short=self.qsa_weights==2 or self.num_heads==32)
                            else:
                                propose_warp(outrow,cute.arch.lane_idx(),mLogits,hist,thresholds,samples,rank,high_rank,candidates,extend_short=self.qsa_weights==2 or self.num_heads==32)
                    subset_barrier(barrier_id=cutlass.Int32(10)+query_slot,number_of_threads=128)
                    q_pipeline.consumer_release(q_cons_state)
                    q_cons_state.advance()
                    next_q_block_idx=q_block_idx+cute.arch.grid_dim()[2]
                    if next_q_block_idx<num_q_blocks:
                        q_last_next=min(next_q_block_idx*self.block_q+self.block_q-1,num_q_tokens-1)
                        next_num_kv=(cutlass.Int32(self.fixed_samples) if cutlass.const_expr(self.fixed_samples>0) else runtime_samples((num_kv_tokens if cutlass.const_expr(self.pro_common_domain) else mCuSeqLenKEnd[q_last_next]),self.top_k,heads=self.num_heads,coverage=4 if self.qsa_weights==2 or self.num_heads==32 else 8,dsv4=self.dsv4))//128
                    has_work=next_q_block_idx<num_q_blocks
            tmem.relinquish_alloc_permit()
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

from .policy_qwen import runtime_samples,block_samples

from .selection import publish_bank
from .sample_policy import proposal_rank

from .policy_dsv4 import individual_sample_key

from .upper_proposal import propose

from .subset_named_barrier import SubsetNamedBarrier
from .subset_barrier import subset_barrier

from .proposal_parallel import propose as propose_parallel

from .policy_qwen import qwen_cached_sample_key

from .proposal_warp import propose_warp
from .proposal_warp_128 import propose_warp_128

from .policy_dsv4 import runtime_samples,block_samples,model_sample_key,model_proposal_rank

from .policy_dsv4 import dsv4_cached_sample_key

from .sample_rank_margin import shape_margin_rank

from .sample_count_policy import shape_sample_key

from .native_lowp_math import native_pack, native_pack_relu, native_fma, native_add, native_lo, native_hi

from .native_heads import head_zero, head_sum, native_head_fma4

from .native_heads import native_weight_pair
