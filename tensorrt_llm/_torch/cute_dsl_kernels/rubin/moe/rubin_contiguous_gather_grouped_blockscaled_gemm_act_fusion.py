# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import os
import re
from typing import NamedTuple, Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.rubin_helpers as sm107_utils
import torch
from cutlass._mlir.dialects import math, nvvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.tcgen05.mma import CollectorOp
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils.gemm.sm100 import (
    epilogue_smem_copy_and_partition,
    transform_partitioned_tensor_layout,
)

from ....utils import ActivationType, is_gated_activation

try:
    from .custom_pipeline import PipelineCpAsyncUmma
    from .inline_ptx import sm100_tcgen05_st_32x32b_x4, sm100_tma_gather4_load
    from .utils import TRTLLM_ENABLE_PDL
except ImportError:
    from custom_pipeline import PipelineCpAsyncUmma
    from inline_ptx import sm100_tcgen05_st_32x32b_x4, sm100_tma_gather4_load
    from utils import TRTLLM_ENABLE_PDL


# ============================================================================
# Inline utility functions
# ============================================================================


@dsl_user_op
def fmin(
    a: Union[float, cutlass.Float32],
    b: Union[float, cutlass.Float32],
    *,
    nan=False,
    loc=None,
    ip=None,
) -> cutlass.Float32:
    return cutlass.Float32(
        nvvm.fmin(
            cutlass.Float32(a).ir_value(loc=loc, ip=ip),
            cutlass.Float32(b).ir_value(loc=loc, ip=ip),
            nan=nan,
            loc=loc,
            ip=ip,
        )
    )


def sigmoid_f32(
    a: Union[float, cutlass.Float32], fastmath: bool = False
) -> Union[float, cutlass.Float32]:
    """Compute the sigmoid of the input tensor."""
    return cute.arch.rcp_approx(1.0 + cute.math.exp(-a, fastmath=fastmath))


def silu_f32(
    a: Union[float, cutlass.Float32], fastmath: bool = False
) -> Union[float, cutlass.Float32]:
    """Compute the silu of the input tensor."""
    return a * sigmoid_f32(a, fastmath=fastmath)


SUPPORTED_ACTIVATION_TYPES = (ActivationType.Swiglu, ActivationType.Relu2)


def validate_activation_type(activation_type) -> ActivationType:
    """Normalize and validate the fused FC1 activation."""
    activation_type = ActivationType(int(activation_type))
    assert activation_type in SUPPORTED_ACTIVATION_TYPES, (
        f"Unsupported activation type {activation_type}; "
        f"expected one of {SUPPORTED_ACTIVATION_TYPES}"
    )
    return activation_type


class S2TCopyBundle(NamedTuple):
    """Bundle of tiled copy and partitioned tensors for smem-to-tmem copies."""

    tiled_copy: cute.TiledCopy
    sSF_compact: cute.Tensor  # Partitioned source (smem)
    tSF_compact: cute.Tensor  # Partitioned destination (tmem)


"""
Rubin (SM107) persistent blockscaled contiguous grouped GEMM with token gather
and fused SwiGLU or Relu2 activation (FC1 of MoE).

Compute:
  acc = alpha * (SFA * A[token_ids]) * (SFB * B)         # GEMM
  C   = up * silu(gate) or relu(acc)^2                    # selected activation
  + optional NVFP4 quantization (generates SFC) when c_dtype == Float4E2M1FN.

Shapes: A is M×K×1; B is N×K×L (L = num experts). SwiGLU uses interleaved
[up, gate] weights at granularity=64 and produces M×(N/2)×1; Relu2 uses
plain weights and produces M×N×1. SFA/SFB layouts follow BlockScaledBasicChunk.
token_id_mapping drives the row gather for A/SFA; token_id == -1 marks padding.

Within a tile, valid_m varies per group; padding rows are handled at load:
TMA gather4 passes -1 to zero-fill; CpAsync predicates on `abs_row < mn_limit`.

Constraints: A/B share dtype (mxf8 | mxf4 | nvf4); mma_tiler M in {128, 256};
mma_tiler N in {64, 128, 192, 256}; cluster M/N pow-2, total ≤ 16;
contiguous dim ≥ 16B aligned (16/32 elems for f8/f4).

For CUDA graph, A/C/SFA/token_id_mapping/tile_idx_to_expert_idx can be padded
to permuted_m; padded tiles are filtered by the scheduler.
"""


class Sm107BlockScaledContiguousGatherGroupedGemmActFusionKernel:
    """Rubin (SM107) FC1: contiguous grouped blockscaled GEMM with token
    gather on A/SFA and activation fusion in the epilogue.

    Supports both SwiGLU (gated) and Relu2 (non-gated) activations.

    Builds on Sm107BlockScaledContiguousGroupedGemmKernel (persistent tile
    scheduling, warp specialization, B-reuse, tcgen05.mma block-scale, TMA
    B/SFB with M-multicast, per-group alpha). Refer to backbone for those.

    Additions on top of backbone:
      - Token gather: A/SFA rows are gathered by token_id_mapping
        (token_id == -1 marks padding rows).
      - A load path (knob `a_path`):
          * cpasync — CpAsync128.CG per-thread (default); separate
            a_pipeline; in 2CTA, warp 11 relays per-CTA a_pipeline to a
            cluster-wide a_sync_transform_pipeline so MMA cta_group::2 sees
            both CTAs' A.
          * tma     — TMA gather4 with HW multicast; A and B share a single
            merged ab_pipeline (no relay warp needed).
        SFA is always loaded via CpAsync128.CG, then reorganized into SFA
        TMEM by transform warps via LDS + STTM (sfa_transform_pipeline).
      - SwiGLU epilogue: C = up * silu(gate), where up/gate come from
        interleaved accumulator at granularity=64 → output N is halved.
      - Optional NVFP4 quant: when c_dtype == Float4E2M1FN, the epilogue
        also generates SFC and quantizes the output.

    Extra warp roles (20 warps total; 4-19 are FC1-only):
      - 0-3   epilogue (LDTM → SwiGLU → optional quant → TMA store)
      - 4-7   gather A         (CpAsync128.CG or TMA gather4)
      - 8     MMA
      - 9     TMA B / SFB
      - 10    scheduler
      - 11    cpasync 2CTA A sync-transform relay (idle on 1CTA / tma)
      - 12-15 gather SFA       (CpAsync128.CG)
      - 16-19 SFA transform    (LDS + STTM into SFA TMEM)

    :param sf_vec_size: Scale factor vector size (16 or 32).
    :param mma_inst_shape: MMA instruction shape (M, N, K).
    :param mma_tiler: MMA tiler shape (M, N, K).
    :param cluster_shape_mn: Cluster dimensions (M, N).
    :param vectorized_f32: Use vectorized f32x2 ops in epilogue.
    :param topk: Experts selected per token.
    :param raster_along_m: If True, raster tiles along M first.
    :param a_path: "cpasync" or "tma" — A load implementation.
    """

    def __init__(
        self,
        sf_vec_size: int,
        mma_inst_shape: Tuple[int, int, int],
        mma_tiler: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        vectorized_f32: bool,
        topk: cutlass.Int64,
        raster_along_m: bool = False,
        a_path: str = "cpasync",
        use_pdl: bool = True,
        locality_domain_half_gemm: bool = False,
        activation_type: ActivationType = ActivationType.Swiglu,
    ):
        self.a_path = a_path
        # locality domain half-GEMM: two partitions write their N-half into a shared
        # full-width C/SFC buffer at a column offset (see wrapper/__call__).
        self.locality_domain_half_gemm = locality_domain_half_gemm
        self.sf_vec_size = sf_vec_size
        self.topk = topk
        self.activation_type = validate_activation_type(activation_type)
        self.is_gated = is_gated_activation(self.activation_type)
        if locality_domain_half_gemm and not self.is_gated:
            raise ValueError("Rubin locality domain half-GEMM currently supports SwiGLU only")
        self.acc_dtype = cutlass.Float32
        self.mma_inst_shape = mma_inst_shape
        self.mma_tiler = mma_tiler
        self.cluster_shape_mn = cluster_shape_mn
        self.raster_along_m = raster_along_m
        # Honor the TRTLLM_ENABLE_PDL env flag (PDL on by default); a caller
        # passing use_pdl=False still disables PDL.
        self.use_pdl = use_pdl and TRTLLM_ENABLE_PDL

        self.use_2cta_instrs = mma_inst_shape[0] == 256
        self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        self.arch = "sm_107"
        self.smem_capacity = utils.get_smem_capacity_in_bytes(self.arch)
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols(self.arch)

        self.occupancy = 1
        self.epilog_warp_id = (0, 1, 2, 3)
        self.gather_a_warp_id = (
            4,
            5,
            6,
            7,
        )
        self.mma_warp_id = 8
        self.tma_b_warp_id = 9
        self.sched_warp_id = 10
        # Warp 11: cpasync 2CTA A peer-sync relay (sync_transform_warp_id) /
        # idle on tma-A (dummy_warp_id). Slot reserved for SM occupancy.
        # sync_transform_warp_id is always defined so downstream traces
        # resolve in tma mode (body is const-gated out).
        self.sync_transform_warp_id = 11
        if self.a_path == "tma":
            self.dummy_warp_id = 11
        self.gather_sfa_warp_id = (
            12,
            13,
            14,
            15,
        )
        # 4 SFA transform warps (LDS source + STTM destination into SFA TMEM).
        self.sfa_transform_warp_id = (
            16,
            17,
            18,
            19,
        )
        # Register reconfig (setmaxnreg) per warpgroup. Default 128/thread.
        self.num_regs_epilogue_warps = 168
        self.num_regs_gather_a_warps = 80
        self.num_regs_gather_sfa_warps = 80
        self.num_regs_sfa_transform_warps = 48
        self.num_regs_mma_group_warps = 128
        self.threads_per_warp = 32
        # warp 11 slot is always counted in threads_per_cta (SM occupancy)
        # regardless of cpasync-A peer-sync role vs TMA-A idle role.
        _warp11_id = self.sync_transform_warp_id if self.a_path == "cpasync" else self.dummy_warp_id
        self.threads_per_cta = self.threads_per_warp * len(
            (
                self.mma_warp_id,
                *self.gather_a_warp_id,
                self.tma_b_warp_id,
                *self.epilog_warp_id,
                self.sched_warp_id,
                _warp11_id,
                *self.gather_sfa_warp_id,
                *self.sfa_transform_warp_id,
            )
        )
        # warps_wo_sched = tile_info_pipeline consumers (all warps except
        # scheduler). Warp 11 counts only on cpasync 2CTA (relay needs tiles);
        # excluded on 1CTA or tma-A (warp 11 idle).
        if self.use_2cta_instrs and self.a_path == "cpasync":
            _wo_sched_warps = (
                *self.epilog_warp_id,
                self.mma_warp_id,
                self.tma_b_warp_id,
                self.sync_transform_warp_id,
                *self.gather_a_warp_id,
                *self.gather_sfa_warp_id,
                *self.sfa_transform_warp_id,
            )
        else:
            _wo_sched_warps = (
                *self.epilog_warp_id,
                self.mma_warp_id,
                self.tma_b_warp_id,
                *self.gather_a_warp_id,
                *self.gather_sfa_warp_id,
                *self.sfa_transform_warp_id,
            )
        self.warps_wo_sched = len(_wo_sched_warps)
        self.threads_wo_sched = self.threads_per_warp * self.warps_wo_sched

        # Set barrier for cta sync, epilogue sync and tmem ptr sync
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=32 * len(self.epilog_warp_id),
        )
        # tmem_alloc_barrier participants: epi (allocator) + mma (consumer)
        # + transform warps (STTM producers, need TMEM ptr to write SFA).
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=32
            * len(
                (
                    self.mma_warp_id,
                    *self.epilog_warp_id,
                    *self.sfa_transform_warp_id,
                )
            ),
        )
        self.sched_sync_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=self.threads_per_warp,
        )

        self.num_smem_capacity = self.smem_capacity
        # num_tmem_alloc_cols already set in __init__

        self.vectorized_f32 = vectorized_f32

        # For epilogue compatibility
        self.epilogue_warp_id = self.epilog_warp_id

        # B-reuse pattern control
        self.enable_breuse = True if mma_tiler[0] // mma_inst_shape[0] == 2 else False

        # Overlapping ACC TMEM: acc[0]/acc[1] share 64 cols. Epilogue
        # iterates the overlap region first (reverse for acc[0]) and
        # early-releases so MMA can write the next stage. Frees enough TMEM
        # for 4 SFA stages. Auto-on for non-breuse cta_tile_N=256.
        self.use_overlap_accum = (not self.enable_breuse) and (mma_tiler[1] == 256)

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B
        - Computing epilogue subtile
        - Setting up A/B/C stage counts in shared memory
        - Computing A/B/C shared memory layout
        - Computing tensor memory allocation columns
        """

        self.mma_inst_shape_sfb = (
            self.mma_inst_shape[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape[1], 128),
            self.mma_inst_shape[2],
        )

        # Configure tiled mma (Rubin SM107)
        tiled_mma = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
            atom_layout_mnk=(1, 1, 1),
            permutation_mnk=self._get_mma_permutation_mnk(),
        )

        tiled_mma_sfb = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_sfb,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
        )

        # Compute mma/cluster/tile shapes
        self.mma_tiler_sfb = (
            self.mma_inst_shape_sfb[0],
            self.mma_inst_shape_sfb[1],
            self.mma_tiler[2],
        )

        self.mma_tiler_c = (
            self.mma_tiler[0],
            self.mma_tiler[1] // 2 if self.is_gated else self.mma_tiler[1],
            self.mma_tiler[2],
        )

        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        # Number of CpAsync128.CG loads per thread for A matrix (each loads 16 M-rows)
        self.a_num_loads = self.cta_tile_shape_mnk[0] // 16

        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
        )

        self.cta_tile_shape_mnk_c = (
            self.mma_tiler_c[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_c[1],
            self.mma_tiler_c[2],
        )

        # Compute SFA tiler for CpAsync gather (use mma_inst_shape for M/N, scaled K for SF)
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = self.mma_tiler[2] // mma_inst_shape_k
        self.mma_tiler_sfa = (
            self.mma_inst_shape[0],
            self.mma_inst_shape[1],
            mma_inst_shape_k * mma_inst_tile_k // 16,
        )
        self.cta_tile_shape_mnk_sfa = (
            self.mma_tiler_sfa[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfa[1],
            self.mma_tiler_sfa[2],
        )

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        # A multicast: cluster_N CTAs share A along N dim. Only meaningful when
        # cluster_N > 1. SFA multicast intentionally NOT enabled — was buggy
        # on cta_tile_N >= 256 and not worth the complexity.
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.is_a_mcast = self.num_mcast_ctas_a > 1

        # Fixed epilogue tile (128, 64). SwiGLU halves N, so the default
        # SM107_TILES lookup (keyed on full cta_n) can pick epi_tile_n too
        # small (wrong TMA store strides + insufficient SFC for cvt_fptrunc
        # 32-bit alignment). (128, 64) works for all configs.
        self.epi_tile = (128, 64)
        self.epi_tile_n = cute.size(self.epi_tile[1])
        self.epi_tile_cnt = (
            self.cta_tile_shape_mnk_c[0] // cute.size(self.epi_tile[0]),
            self.cta_tile_shape_mnk_c[1] // cute.size(self.epi_tile[1]),
        )

        # Setup A/B/C/Scale stage count in shared memory and ACC stage count in tensor memory
        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.num_c_stage,
            self.num_tile_stage,
        ) = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.cta_tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.sf_dtype,
            self.sf_vec_size,
            self.smem_capacity,
            self.occupancy,
            self.enable_breuse,
        )

        # Compute A/B/C/Scale shared memory layout
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        # Canonical SFA SMEM layout (from blockscaled_utils), used only to
        # derive tCtSFA_layout below. The actual SFA SMEM uses the linear
        # layout built next; this canonical one isn't allocated.
        sfa_canon_smem_layout = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )

        # SFA SMEM is plain linear (M_per_cta, tile_K_sf, stage), no pad.
        # Each thread does one CpAsync128.CG (16B = tile_K_sf=16 × FP8) per row.
        # Layout exposes (row, k_sf_byte, stage) with byte strides.
        sfa_tile_k_sf = self.cta_tile_shape_mnk[2] // self.sf_vec_size
        sf_bytes_per_row = sfa_tile_k_sf * self.sf_dtype.width // 8
        sfa_bytes_per_stage = self.cta_tile_shape_mnk[0] * sf_bytes_per_row
        self.sfa_smem_layout_staged = cute.make_layout(
            (self.cta_tile_shape_mnk[0], sfa_tile_k_sf, self.num_ab_stage),
            stride=(sf_bytes_per_row, 1, sfa_bytes_per_stage),
        )
        self.sfa_smem_alloc_bytes = self.num_ab_stage * sfa_bytes_per_stage

        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_c_stage,
        )

        # Compute TMEM layouts for SFA/SFB (Rubin precomputed)
        self.tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            cute.slice_(sfa_canon_smem_layout, (None, None, None, 0)),
        )
        self.tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0)),
        )

        # Compute TMEM column counts.
        # SFA TMEM holds num_sfa_tmem_stage stages, each tCtSFA_layout wide.
        # TMEM layout: [acc | sfa (N stages) | sfb].
        self.num_sfa_tmem_cols_per_stage = (
            cute.cosize(cute.recast_layout(32, self.sf_dtype.width, self.tCtSFA_layout))
            & 0x0000FFFF
        )
        # SFA TMEM stages: 4 for non-breuse, 1 for breuse (576-col TMEM
        # already saturated with 1-stage SFA at 32 cols). At non-breuse
        # N=256, use_overlap_accum is auto-on to free TMEM for 4 stages.
        self.num_sfa_tmem_stage = 1 if self.enable_breuse else 4
        self.num_sfa_tmem_cols = self.num_sfa_tmem_cols_per_stage * self.num_sfa_tmem_stage
        self.num_sfb_tmem_cols = (
            cute.cosize(cute.recast_layout(32, self.sf_dtype.width, self.tCtSFB_layout))
            & 0x0000FFFF
        )
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        # use_overlap_accum: pipeline tracks 1 acc stage; physically 2 stages
        # share TMEM with one epi_tile_n of overlap. Otherwise:
        # tile_N × num_acc_stage × (2 if breuse).
        if self.use_overlap_accum:
            self.num_acc_stage = 1  # logical 2 via overlap
            self.num_accumulator_tmem_cols = self.cta_tile_shape_mnk[1] * 2 - self.epi_tile_n
        else:
            self.num_accumulator_tmem_cols = (
                self.cta_tile_shape_mnk[1] * self.num_acc_stage * (2 if self.enable_breuse else 1)
            )
        # SFA TMEM offset (cols, 32-bit each): right after acc.
        self.sfa_tmem_offset = self.num_accumulator_tmem_cols
        # Validation: 512 + 32 + 32 = 576 (exact fit for main target on sm_107)
        _total_used = (
            self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        )
        if _total_used > self.num_tmem_alloc_cols:
            raise ValueError(
                f"TMEM overflow: acc({self.num_accumulator_tmem_cols}) + "
                f"sfa({self.num_sfa_tmem_cols}) + "
                f"sfb({self.num_sfb_tmem_cols}) = {_total_used} > "
                f"max {self.num_tmem_alloc_cols}"
            )

    def _get_mma_permutation_mnk(self):
        if cutlass.const_expr(self.use_2cta_instrs and self.enable_breuse):
            m_layout = cute.make_layout(
                shape=(self.mma_inst_shape[0] // 2, 2, 2),
                stride=(1, self.mma_inst_shape[0], self.mma_inst_shape[0] // 2),
            )
            return (m_layout, self.mma_inst_shape[1], self.mma_inst_shape[2])
        else:
            return (1, 1, 1)

    def _is_interleaved_utccp(self) -> bool:
        """Enable interleaving UTCCP for Bkeep-Breuse case for 4xFP4 kernel."""
        return self.a_dtype.width == 4 and self.b_dtype.width == 4 and self.enable_breuse

    def _mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> S2TCopyBundle:
        """Make tiledCopy for smem to tmem load for scale factor tensor."""
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)

        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        def appendMNBroadcastMode(smem_layout: cute.Layout):
            mn_dim = cute.get(smem_layout, mode=[0, 0])
            mn_dim = cute.append(mn_dim, cute.make_layout((4), stride=(0)))
            layout = cute.append(cute.group_modes(mn_dim, 0), cute.get(smem_layout, mode=[0, 1]))
            layout = cute.append(cute.group_modes(layout, 0), cute.get(smem_layout, mode=[1]))
            layout = cute.append(layout, cute.get(smem_layout, mode=[2]))
            layout = cute.append(layout, cute.get(smem_layout, mode=[3]))
            return layout

        tCsSF_compact_bcast = cute.make_tensor(
            tCsSF_compact.iterator, appendMNBroadcastMode(tCsSF_compact.layout)
        )

        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact_bcast)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return S2TCopyBundle(tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t)

    def _mainloop_s2t_copies(
        self,
        stage_idx: int,
        sfb_s2t_bundle: S2TCopyBundle,
    ):
        """Copy SFB from smem to tmem (UTCCP). SFA path now uses LDS+STTM
        from transform warps, no UTCCP needed here."""
        s2t_stage_coord = (None, None, None, None, stage_idx)

        cute.copy(
            sfb_s2t_bundle.tiled_copy,
            sfb_s2t_bundle.sSF_compact[s2t_stage_coord],
            sfb_s2t_bundle.tSF_compact,
        )

    def _mainloop_s2t_interleaved_copies(
        self,
        k_block: int,
        stage_idx: int,
        sfa_s2t_bundle: S2TCopyBundle,
        sfb_s2t_bundle: S2TCopyBundle,
    ):
        """Interleaved UTCCP for Bkeep-Breuse pattern."""
        s_sfa_crd_keep = (None, 0, None, k_block, stage_idx)
        s_sfa_crd_reuse = (None, 1, None, k_block, stage_idx)
        s_sfb_crd = (None, None, None, k_block, stage_idx)

        t_sfa_crd_keep = (None, 0, None, k_block)
        t_sfa_crd_reuse = (None, 1, None, k_block)
        t_sfb_crd = (None, None, None, k_block)

        cute.copy(
            sfa_s2t_bundle.tiled_copy,
            sfa_s2t_bundle.sSF_compact[s_sfa_crd_keep],
            sfa_s2t_bundle.tSF_compact[t_sfa_crd_keep],
        )
        cute.copy(
            sfb_s2t_bundle.tiled_copy,
            sfb_s2t_bundle.sSF_compact[s_sfb_crd],
            sfb_s2t_bundle.tSF_compact[t_sfb_crd],
        )
        cute.copy(
            sfa_s2t_bundle.tiled_copy,
            sfa_s2t_bundle.sSF_compact[s_sfa_crd_reuse],
            sfa_s2t_bundle.tSF_compact[t_sfa_crd_reuse],
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        sfc_tensor: Optional[cute.Tensor],
        full_c_shape: Optional[cute.Shape],
        norm_const_tensor: Optional[cute.Tensor],
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        token_id_mapping_tensor: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        alpha: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
        c_sf_n_tile_offset: cutlass.Int64 = cutlass.Int64(0),
    ):
        """Execute the contiguous grouped GEMM with gather operation and SwiGLU fusion.

        This method performs FC1 layer computation:
        1. GEMM: acc = alpha * (SFA * A[token_ids]) * (SFB * B)
        2. SwiGLU: C = up * silu(gate), where up/gate are extracted from interleaved acc (granularity=64)
        3. Optional Quant: When c_dtype is Float4E2M1FN, generates SFC and quantizes output

        Data loading:
        - A and SFA are loaded using CpAsync instructions with token-based gather
        - B and SFB are loaded using TMA instructions with multicast
        - B weights are interleaved: [up_0:64, gate_64:128, up_128:192, gate_192:256, ...]

        Execution steps:
        1. Setup static attributes before smem/grid computation
        2. Setup TMA load/store atoms for B, SFB, and C (no TMA for A/SFA)
        3. Compute grid size with regard to hardware constraints
        4. Define shared storage for kernel
        5. Launch the kernel synchronously with warp specialization:
           - Scheduler warp: Dispatches tile information
           - CpAsync warps: Load A and SFA with gather
           - A Sync Transform warps: Transform the sync signal of A and SFA from global to
             shared memory when use_2cta_instrs is True
           - TMA warp: Load B and SFB with multicast
           - MMA warp: Perform matrix multiply-accumulate
           - Epilogue warps: Apply SwiGLU activation, optional quantization, and store results

        :param a: Input tensor A (MxKx1), will be gathered using token_id_mapping
        :type a: cute.Tensor
        :param b: Input tensor B (NxKxL), L is the number of experts/groups, weights are interleaved for SwiGLU
        :type b: cute.Tensor
        :param c: Output tensor C (Mx(N/2)x1), N is halved due to SwiGLU fusion
        :type c: cute.Tensor
        :param sfa: Scale factor tensor A, will be gathered using token_id_mapping
        :type sfa: cute.Tensor
        :param sfb: Scale factor tensor B
        :type sfb: cute.Tensor
        :param sfc_tensor: Scale factor tensor C for quantized output (None if not quantizing)
        :type sfc_tensor: Optional[cute.Tensor]
        :param norm_const_tensor: Normalization constant for scale factor generation
            (None if not quantizing)
        :type norm_const_tensor: Optional[cute.Tensor]
        :param tile_idx_to_expert_idx: Mapping from tile index to expert ID,
            shape (permuted_m/cta_tile_m,) where cta_tile_m is the CTA tile M size
        :type tile_idx_to_expert_idx: cute.Tensor
        :param tile_idx_to_mn_limit: Mapping from tile index to M-N dimension limit
            for boundary checking, shape (permuted_m/cta_tile_m,)
        :type tile_idx_to_mn_limit: cute.Tensor
        :param token_id_mapping_tensor: Token ID mapping for gather operation, shape (permuted_m,)
        :type token_id_mapping_tensor: cute.Tensor
        :param num_non_exiting_tiles: Number of valid tiles to process (valid_m/cta_tile_m), shape (1,)
        :type num_non_exiting_tiles: cute.Tensor
        :param alpha: Alpha tensor for each group
        :type alpha: cute.Tensor
        :param max_active_clusters: Maximum number of active clusters
        :type max_active_clusters: cutlass.Constexpr
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        :param epilogue_op: Optional elementwise lambda function to apply to the output tensor
        :type epilogue_op: cutlass.Constexpr
        :raises TypeError: If input data types are incompatible with the MMA instruction.
        """
        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        # Note: Rubin supports mixed A/B dtypes (e.g., Float8E4M3FN x Float8E5M2)

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        # Setup sfb tensor by filling B tensor to scale factor atom layout
        # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b.shape, self.sf_vec_size)
        sfb = cute.make_tensor(sfb.iterator, sfb_layout)

        # Setup sfc tensor by filling C tensor to scale factor atom layout.
        # For locality domain, full_c_shape carries the full N dimension so sfc gets the
        # correct M-tile stride (two locality domains write their N-half into the shared
        # SF buffer without copy-back); None → use c.shape (non-locality domain).
        self.generate_sfc = sfc_tensor is not None and norm_const_tensor is not None
        if cutlass.const_expr(self.generate_sfc):
            sfc_shape = c.shape if full_c_shape is None else full_c_shape
            sfc_layout = blockscaled_utils.tile_atom_to_shape_SF(sfc_shape, self.sf_vec_size)
            sfc_tensor = cute.make_tensor(sfc_tensor.iterator, sfc_layout)

        atom_layout_mnk = (1, 1, 1)
        permutation_mnk = self._get_mma_permutation_mnk()

        tiled_mma = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
            atom_layout_mnk=atom_layout_mnk,
            permutation_mnk=permutation_mnk,
        )
        tiled_mma.set(tcgen05.Field.NEGATE_A, False)
        tiled_mma.set(tcgen05.Field.NEGATE_B, False)

        # For 2CTA blockscaled kernels, SFB needs to be replicated across peer CTAs.
        tiled_mma_sfb = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_sfb,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
        )
        tiled_mma_sfb.set(tcgen05.Field.NEGATE_A, False)
        tiled_mma_sfb.set(tcgen05.Field.NEGATE_B, False)

        tiled_mma_bkeep = None
        tiled_mma_breuse = None
        if cutlass.const_expr(self.enable_breuse):
            tiled_mma_bkeep = sm107_utils.make_blockscaled_trivial_tiled_mma(
                self.a_dtype,
                self.b_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                self.mma_inst_shape,
                a_collector_op=CollectorOp.DISCARD,
                b_collector_op=CollectorOp.FILL,
                atom_layout_mnk=atom_layout_mnk,
                permutation_mnk=permutation_mnk,
            )
            tiled_mma_bkeep.set(tcgen05.Field.NEGATE_A, False)
            tiled_mma_bkeep.set(tcgen05.Field.NEGATE_B, False)

            tiled_mma_breuse = sm107_utils.make_blockscaled_trivial_tiled_mma(
                self.a_dtype,
                self.b_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                self.mma_inst_shape,
                a_collector_op=CollectorOp.DISCARD,
                b_collector_op=CollectorOp.LASTUSE,
                atom_layout_mnk=atom_layout_mnk,
                permutation_mnk=permutation_mnk,
            )
            tiled_mma_breuse.set(tcgen05.Field.NEGATE_A, False)
            tiled_mma_breuse.set(tcgen05.Field.NEGATE_B, False)
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # cpasync-A: CpAsync128.CG gmem → sA SMEM, no TMA. 4 gather_a warps × 32
        # threads issue cp.async.cg.16B per (token row, k chunk). a_num_loads
        # (in _setup_attributes) controls CpAsync iterations per thread per k_tile.
        tma_atom_a = None
        tma_tensor_a = None
        # tma-A mode: build a 2D gather4 TMA atom for A (box_rows = 1,
        # SW128-permuted base) so gather4 writes sA at the same SW128 offsets
        # UMMA's K_SW128 reads expect.
        if cutlass.const_expr(self.a_path == "tma"):
            a_2d = a[(None, None, 0)]
            a_gather_base = cute.make_layout(
                (1, self.mma_tiler[2]),
                stride=(self.mma_tiler[2], 1),
            )
            sw128 = cute.make_swizzle(3, 4, 3)  # K_SW128
            a_gather_smem_layout = cute.make_composed_layout(sw128, 0, a_gather_base)
            tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                a_2d,
                a_gather_smem_layout,
                (1, self.mma_tiler[2]),  # cta_tiler, box_rows = 1
            )
            # tx_count split evenly across the 4 gather warps. Total = full A
            # tile bytes (× atom_thr_size for 2CTA's leader-mbar collapse).
            self.tma_gather_num_warps = len(self.gather_a_warp_id)
            self.a_num_tma_load_bytes_total = (
                self.cta_tile_shape_mnk[0]
                * self.cta_tile_shape_mnk[2]
                * self.a_dtype.width
                // 8
                * atom_thr_size
            )
            self.a_num_tma_load_bytes = self.a_num_tma_load_bytes_total // self.tma_gather_num_warps

        # cpasync-A SFA path is also CpAsync128.CG (no TMA). gather_sfa warps
        # issue cp.async per row; one row per thread (128 rows / 128 threads
        # in 4 warps). No tma_atom_sfa needed in either A mode.

        # Setup TMA load for B
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for SFB
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma.thr_id)
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # This modifies the layout to handle overlapping 256x(# of scale factors for a single column of B (nNSF))
        # logical blocks for SFB when cta_tile_shape_n=192.
        if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
            x = tma_tensor_sfb.stride[0][1]
            y = cute.ceil_div(tma_tensor_sfb.shape[0][1], 4)

            new_shape = (
                (tma_tensor_sfb.shape[0][0], ((2, 2), y)),
                tma_tensor_sfb.shape[1],
                tma_tensor_sfb.shape[2],
            )
            # Use right multiplication for ScaledBasis (3 * x instead of x * 3)
            x_times_3 = 3 * x
            new_stride = (
                (tma_tensor_sfb.stride[0][0], ((x, x), x_times_3)),
                tma_tensor_sfb.stride[1],
                tma_tensor_sfb.stride[2],
            )
            tma_tensor_sfb_new_layout = cute.make_layout(new_shape, stride=new_stride)
            tma_tensor_sfb = cute.make_tensor(tma_tensor_sfb.iterator, tma_tensor_sfb_new_layout)

        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (b_copy_size + sfb_copy_size) * atom_thr_size

        # Setup TMA store for C
        tma_atom_c = None
        tma_tensor_c = None
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c,
            epi_smem_layout,
            self.epi_tile,
        )

        # Compute grid size
        self.tile_sched_params, grid = self._compute_grid(
            c,
            self.cta_tile_shape_mnk_c,
            self.cluster_shape_mn,
            max_active_clusters,
            self.raster_along_m,
        )

        self.buffer_align_bytes = 1024

        # Define shared storage for kernel.
        @cute.struct
        class SharedStorageCpasync1cta:
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 5 * self.num_tile_stage],
                1,
            ]
            # cpasync mode: A and B use separate pipelines (CpAsync A is
            # CpAsync type, B is TmaUmma type — they can't share one mbar).
            # Each mbar set holds num_ab_stage * 2 (full + empty per stage).
            a_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            b_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            sfa_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            sfa_transform_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_sfa_tmem_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tile_info_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_tile_stage * 2]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # sSFA placed BEFORE sA so SFA gets a low SMEM offset: CpAsync128.CG
            # destination addr stays under the 248KB threshold (above which the
            # .CG cache-mode hint would degrade to a plain CpAsync).
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, self.sfa_smem_alloc_bytes],
                self.buffer_align_bytes,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        # 2CTA variant: adds a_sync_transform_mbar_ptr for the warp-11 relay
        # pipeline (PipelineAsyncUmma) that bridges per-CTA `a_pipeline` to
        # the cluster-wide MMA consumer (tcgen05.mma.cta_group::2).
        @cute.struct
        class SharedStorageCpasync2cta:
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 5 * self.num_tile_stage],
                1,
            ]
            a_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            b_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            sfa_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            sfa_transform_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_sfa_tmem_stage * 2]
            a_sync_transform_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tile_info_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_tile_stage * 2]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, self.sfa_smem_alloc_bytes],
                self.buffer_align_bytes,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        # TMA-A variant: A and B share one ab_pipeline mbar (4 gather_a + 1
        # tma_b producers arrive on the same mbar). No a_sync_transform mbar
        # needed (TMA gather4 has HW `.multicast::cluster`, no peer-sync relay).
        # Storage layout is independent of 1CTA/2CTA mode in TMA-A mode.
        @cute.struct
        class SharedStorageTma:
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 5 * self.num_tile_stage],
                1,
            ]
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            sfa_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            sfa_transform_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_sfa_tmem_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tile_info_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_tile_stage * 2]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # sSFA placed BEFORE sA so SFA gets a low SMEM offset (cpasync.128
            # destination addr stays under the 248KB threshold).
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, self.sfa_smem_alloc_bytes],
                self.buffer_align_bytes,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        if cutlass.const_expr(self.a_path == "cpasync"):
            self.shared_storage = (
                SharedStorageCpasync2cta if self.use_2cta_instrs else SharedStorageCpasync1cta
            )
        else:  # "tma"
            self.shared_storage = SharedStorageTma

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tiled_mma_bkeep,
            tiled_mma_breuse,
            tiled_mma_sfb,
            a,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
            sfc_tensor,
            norm_const_tensor,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            token_id_mapping_tensor,
            num_non_exiting_tiles,
            alpha,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.tCtSFA_layout,
            self.tCtSFB_layout,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            epilogue_op,
            c_sf_n_tile_offset,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.use_pdl,
        )
        return

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for smem to tmem load for scale factor tensor, then use it to
        partition smem memory (source) and tensor memory (destination).

        :param sSF: The scale factor tensor in smem
        :type sSF: cute.Tensor
        :param tSF: The scale factor tensor in tmem
        :type tSF: cute.Tensor

        :return: A tuple containing (tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t) where:
            - tiled_copy_s2t: The tiled copy operation for smem to tmem load for scale factor tensor(s2t)
            - tCsSF_compact_s2t: The partitioned scale factor tensor in smem
            - tSF_compact_s2t: The partitioned scale factor tensor in tmem
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # (MMA, MMA_MN, MMA_K, STAGE)
        tCsSF_compact = cute.filter_zeros(sSF)
        # (MMA, MMA_MN, MMA_K)
        tCtSF_compact = cute.filter_zeros(tSF)

        # Make S2T CopyAtom and tiledCopy
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_bkeep: Optional[cute.TiledMma],
        tiled_mma_breuse: Optional[cute.TiledMma],
        tiled_mma_sfb: cute.TiledMma,
        mA_mkl: cute.Tensor,
        tma_atom_a: Optional[cute.CopyAtom],
        tma_tensor_a: Optional[cute.Tensor],
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        mSFC_mnl: Optional[cute.Tensor],
        norm_const_tensor: Optional[cute.Tensor],
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        token_id_mapping_tensor: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        alpha: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        tCtSFA_layout: cute.Layout,
        tCtSFB_layout: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
        c_sf_n_tile_offset: cutlass.Int64 = cutlass.Int64(0),
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_b_warp_id:
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_c)
            if cutlass.const_expr(self.a_path == "tma"):
                cpasync.prefetch_descriptor(tma_atom_a)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )

        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # (a_pipeline created below alongside b_pipeline.)

        # SFA pipeline (PipelineCpAsync): gather_sfa warps → transform warps.
        # Producer: 4 gather_sfa warps × 32 threads (one CpAsync128.CG per row).
        # Consumer: 4 transform warps × 32 threads.
        # MMA waits on sfa_transform_pipeline downstream (after LDS+STTM).
        sfa_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len(self.gather_sfa_warp_id) * self.threads_per_warp,
        )
        sfa_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len(self.sfa_transform_warp_id) * self.threads_per_warp,
        )
        sfa_pipeline = pipeline.PipelineCpAsync.create(
            barrier_storage=storage.sfa_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=sfa_pipeline_producer_group,
            consumer_group=sfa_pipeline_consumer_group,
            defer_sync=True,
        )

        # SFA transform pipeline: transform warps (STTM) → MMA. PipelineAsyncUmma
        # with cta_layout_vmnk so 2CTA peer-CTA arrives route to leader's mbar.
        # Producer: 4 transform warps × 32 threads × cta_v_size. Consumer: MMA.
        # num_stages = num_sfa_tmem_stage SFA TMEM slots, rotated.
        cta_v_size = cute.size(cluster_layout_vmnk, mode=[0])
        sfa_transform_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len(self.sfa_transform_warp_id) * self.threads_per_warp * cta_v_size,
        )
        sfa_transform_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        sfa_transform_pipeline = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.sfa_transform_mbar_ptr.data_ptr(),
            num_stages=self.num_sfa_tmem_stage,
            producer_group=sfa_transform_pipeline_producer_group,
            consumer_group=sfa_transform_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # A/B pipeline topology — branch on a_path:
        #   cpasync-A: separate a_pipeline (CpAsyncUmma) + b_pipeline (TmaUmma)
        #     + a_sync_transform_pipeline (AsyncUmma, 2CTA peer-sync relay).
        #   tma-A: single ab_pipeline (TmaUmma) shared by A and B producers.
        if cutlass.const_expr(self.a_path == "cpasync"):
            # cpasync-A: A producer = 128 threads (4 gather_a warps) issuing
            # cp.async.cg.16B; each thread arrives once per stage. Consumer =
            # MMA (UMMA), 1 thread.
            a_pipeline_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_warp * len(self.gather_a_warp_id),
            )
            a_pipeline = PipelineCpAsyncUmma.create(
                barrier_storage=storage.a_mbar_ptr.data_ptr(),
                num_stages=self.num_ab_stage,
                producer_group=a_pipeline_producer_group,
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )

            # cpasync 2CTA A sync-transform relay. Producer thread count
            # = 1 warp × cta_v_size so the cluster-wide arrive_count matches
            # warp 11 across CTAs.
            if cutlass.const_expr(self.use_2cta_instrs):
                a_sync_transform_pipeline_producer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    self.threads_per_warp * cta_v_size,
                )
                a_sync_transform_pipeline = pipeline.PipelineAsyncUmma.create(
                    barrier_storage=storage.a_sync_transform_mbar_ptr.data_ptr(),
                    num_stages=self.num_ab_stage,
                    producer_group=a_sync_transform_pipeline_producer_group,
                    consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                    cta_layout_vmnk=cluster_layout_vmnk,
                    defer_sync=True,
                )

            # B/SFB pipeline (TMA → UMMA), 1 producer thread (tma_b warp).
            # mcast_mode_mn=(0, 1): A is per-CTA cpasync (no N-multicast); B
            # is TMA per-CTA or M-multicast. Default (1,1) would release
            # across N peers — wrong since N-peers hold different B tiles.
            b_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            num_tma_producer = self.num_mcast_ctas_b
            b_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_tma_producer
            )
            b_pipeline = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.b_mbar_ptr.data_ptr(),
                num_stages=self.num_ab_stage,
                producer_group=b_pipeline_producer_group,
                consumer_group=b_pipeline_consumer_group,
                tx_count=self.num_tma_load_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                mcast_mode_mn=(0, 1),
                defer_sync=True,
            )
        else:  # tma — merged ab_pipeline (TmaUmma): 4 gather_a + 1 tma_b
            # on one mbar. Per-call expected_tx accumulates 4 × a_num_tma_load
            # + num_tma_load per stage. Consumer release routes A's N peers
            # + B's M peers → consumer_group = num_mcast_ctas_a + num_mcast_ctas_b - 1.
            ab_pipeline_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                len(self.gather_a_warp_id) + 1,  # 4 gather_a + 1 tma_b
            )
            num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
            ab_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_tma_producer
            )
            ab_pipeline = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.ab_mbar_ptr.data_ptr(),
                num_stages=self.num_ab_stage,
                producer_group=ab_pipeline_producer_group,
                consumer_group=ab_pipeline_consumer_group,
                tx_count=0,  # per-producer expected_tx overrides drive accumulation
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )

        # Pipeline Init: Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Pipeline Init:Initialize tile info pipeline (barrier) and states
        tile_info_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * 1,
        )
        # All 4 gather A warps consume tile_info in both CpAsync and TMA paths.
        tile_info_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_wo_sched,
        )
        tile_info_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.tile_info_mbar_ptr.data_ptr(),
            num_stages=self.num_tile_stage,
            producer_group=tile_info_pipeline_producer_group,
            consumer_group=tile_info_pipeline_consumer_group,
        )

        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
            arch=self.arch,
        )

        # Cluster arrive after barrier init (Rubin uses pipeline_init_arrive)
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        #
        # Setup smem tensor A/B/C/Scale
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sC = storage.sC.get_tensor(c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner)
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        # SFA SMEM (linear+pad layout for TMA gather4).
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        # (granularity_n, repeat_n), (granularity_k, repeat_k), num_scale_stage)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        # (bidx, bidy, bidz, valid, mn_limit)
        info_layout = cute.make_layout((5, self.num_tile_stage), stride=(1, 5))
        sInfo = storage.sInfo.get_tensor(info_layout)

        #
        # Compute multicast mask for A/B buffer full
        #
        b_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_b_mcast or use_2cta_instrs):
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
            )
        # A multicast mask (tma-A only). cpasync-A path has no TMA multicast
        # (each thread issues its own cp.async.cg.16B). tma-A broadcasts A
        # along cluster N (mcast_mode=2); sm100_tma_gather4_load picks the
        # `.multicast::cluster` PTX variant when this mask is non-None.
        a_full_mcast_mask = None
        if cutlass.const_expr(self.a_path == "tma" and self.is_a_mcast):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, loopM, loopK, loopL)
        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_(self.cta_tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        # (bN, bK, loopN, loopK, loopL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )

        # (bM, bK, RestM, RestK, RestL)
        gSFA_mkl = cute.local_tile(
            mSFA_mkl,
            cute.slice_(self.cta_tile_shape_mnk_sfa, (None, 0, None)),
            (None, None, None),
        )

        # (bN, bK, RestN, RestK, RestL)
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )

        gToken_ml = cute.local_tile(
            token_id_mapping_tensor,
            cute.slice_(self.cta_tile_shape_mnk, (None, 0, 0)),
            (None,),
        )

        # (bM, bN, loopM, loopN, loopL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler_c, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cutlass.Int32(cute.size(gA_mkl, mode=[3]))

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        # (MMA, MMA_N, MMA_K, loopN, loopK, loopL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        # (MMA, MMA_M, MMA_N, loopM, loopN, loopL)
        tCgC = thr_mma.partition_C(gC_mnl)

        #
        # Partition global/shared tensor for TMA load B
        #
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), loopM, loopK, loopL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        # TMA load SFB partition_S/D
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        if cutlass.const_expr(self.use_overlap_accum):
            # Pipeline tracks 1 stage but TMEM has 2 physical regions overlap-
            # ping by 64 cols. Build fragment with 2 stages and stride-hack
            # the stage dim to (256 - 64) = 192 cols (= cta_tile_N - overlap).
            num_acc_stage_overlapped = 2
            tCtAcc_fake = tiled_mma.make_fragment_C(
                cute.append(acc_shape, num_acc_stage_overlapped)
            )
            tCtAcc_fake = cute.make_tensor(
                tCtAcc_fake.iterator,
                cute.make_layout(
                    tCtAcc_fake.shape,
                    stride=(
                        tCtAcc_fake.stride[0],
                        tCtAcc_fake.stride[1],
                        tCtAcc_fake.stride[2],
                        (256 - 64) * tCtAcc_fake.stride[0][1],
                    ),
                ),
            )
        else:
            tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        cute.arch.griddepcontrol_wait()

        #
        # Specialized Schedule Warp
        #
        if warp_idx == self.sched_warp_id:
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            # First tile
            work_tile = tile_sched.initial_work_tile_info()

            tile_info_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_tile_stage
            )

            num_non_exiting_tiles_value = num_non_exiting_tiles[0]

            if cutlass.const_expr(self.raster_along_m):
                while work_tile.is_valid_tile:
                    cur_tile_coord = work_tile.tile_idx
                    mma_tile_coord_m = cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape)
                    if mma_tile_coord_m < num_non_exiting_tiles_value:
                        tile_info_pipeline.producer_acquire(tile_info_producer_state)
                        cur_tile_coord = work_tile.tile_idx
                        expert_idx = tile_idx_to_expert_idx[mma_tile_coord_m]
                        mn_limit = tile_idx_to_mn_limit[mma_tile_coord_m]
                        with cute.arch.elect_one():
                            sInfo[(0, tile_info_producer_state.index)] = cur_tile_coord[0]
                            sInfo[(1, tile_info_producer_state.index)] = cur_tile_coord[1]
                            sInfo[(2, tile_info_producer_state.index)] = expert_idx
                            sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(
                                work_tile.is_valid_tile
                            )
                            sInfo[(4, tile_info_producer_state.index)] = mn_limit
                            # fence view async shared
                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )

                        self.sched_sync_barrier.arrive_and_wait()
                        tile_info_pipeline.producer_commit(tile_info_producer_state)
                        tile_info_producer_state.advance()

                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
            else:
                is_continue = cutlass.Boolean(1)
                while work_tile.is_valid_tile and is_continue:
                    cur_tile_coord = work_tile.tile_idx
                    mma_tile_coord_m = cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape)
                    if mma_tile_coord_m < num_non_exiting_tiles_value:
                        tile_info_pipeline.producer_acquire(tile_info_producer_state)
                        cur_tile_coord = work_tile.tile_idx
                        expert_idx = tile_idx_to_expert_idx[mma_tile_coord_m]
                        mn_limit = tile_idx_to_mn_limit[mma_tile_coord_m]
                        with cute.arch.elect_one():
                            sInfo[(0, tile_info_producer_state.index)] = cur_tile_coord[0]
                            sInfo[(1, tile_info_producer_state.index)] = cur_tile_coord[1]
                            sInfo[(2, tile_info_producer_state.index)] = expert_idx
                            sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(
                                work_tile.is_valid_tile
                            )
                            sInfo[(4, tile_info_producer_state.index)] = mn_limit
                            # fence view async shared
                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )

                        self.sched_sync_barrier.arrive_and_wait()
                        tile_info_pipeline.producer_commit(tile_info_producer_state)
                        tile_info_producer_state.advance()
                    else:
                        is_continue = cutlass.Boolean(0)

                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()

            tile_info_pipeline.producer_acquire(tile_info_producer_state)
            with cute.arch.elect_one():
                sInfo[(0, tile_info_producer_state.index)] = work_tile.tile_idx[0]
                sInfo[(1, tile_info_producer_state.index)] = work_tile.tile_idx[1]
                sInfo[(2, tile_info_producer_state.index)] = -1
                sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(0)
                sInfo[(4, tile_info_producer_state.index)] = -1
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            self.sched_sync_barrier.arrive_and_wait()
            tile_info_pipeline.producer_commit(tile_info_producer_state)
            tile_info_producer_state.advance()
            tile_info_pipeline.producer_tail(tile_info_producer_state)

        # Gather A warps (warps 4-7). cpasync / tma bodies static-gated by
        # cutlass.const_expr — only one is traced. cpasync: 4 warps × 32
        # threads issue a_num_loads CpAsync128.CG per k_tile (thread layout
        # (16, 8) covers 16 M-rows × 8 K-chunks); padded rows predicate off.
        if warp_idx <= self.gather_a_warp_id[-1] and warp_idx >= self.gather_a_warp_id[0]:
            cute.arch.setmaxregister_decrease(self.num_regs_gather_a_warps)
            if cutlass.const_expr(self.a_path == "cpasync"):
                a_atom_copy = cute.make_copy_atom(
                    cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                    mA_mkl.element_type,
                    num_bits_per_copy=128,
                )
                a_thread_layout = cute.make_layout((16, 8), stride=(8, 1))
                a_value_layout = cute.make_layout((1, 32), stride=(32, 1))
                a_tiled_copy = cute.make_tiled_copy_tv(
                    a_atom_copy,
                    a_thread_layout,
                    a_value_layout,
                )
                tidx_in_warpgroup = tidx % 128

                sA_tiled = cute.make_tensor(
                    sA.iterator,
                    layout=cute.make_layout(
                        (
                            self.cta_tile_shape_mnk[0],
                            self.cta_tile_shape_mnk[2],
                            self.num_ab_stage,
                        ),
                        stride=(
                            self.cta_tile_shape_mnk[2],
                            1,
                            self.cta_tile_shape_mnk[0] * self.cta_tile_shape_mnk[2],
                        ),
                    ),
                )
                a_thr_copy = a_tiled_copy.get_slice(tidx_in_warpgroup)
                tAsA_tiled = a_thr_copy.partition_D(sA_tiled)

                a_token_offset_tensor = cute.make_rmem_tensor(
                    cute.make_layout((self.a_num_loads,)),
                    cutlass.Int32,
                )
                a_predicate_tensor = cute.make_rmem_tensor(
                    cute.make_layout((self.a_num_loads,)),
                    cutlass.Boolean,
                )

                tile_sched = utils.StaticPersistentTileScheduler.create(
                    tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
                )
                work_tile = tile_sched.initial_work_tile_info()

                a_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_ab_stage
                )
                tile_info_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_tile_stage
                )

                tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(5, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

                while is_valid_tile:
                    gToken_ml_tile = gToken_ml[(None, tile_info[0])]
                    for i in range(self.a_num_loads):
                        token_ml_tile_offset = (tidx_in_warpgroup // 8) + i * 16
                        a_token_offset_tensor[i] = gToken_ml_tile[token_ml_tile_offset]
                        a_predicate_tensor[i] = (
                            cutlass.Boolean(1)
                            if tile_info[0] * self.cta_tile_shape_mnk[0] + token_ml_tile_offset
                            < tile_info[4]
                            else cutlass.Boolean(0)
                        )
                        a_token_offset_tensor[i] = (
                            a_token_offset_tensor[i] // self.topk
                            if tile_info[0] * self.cta_tile_shape_mnk[0] + token_ml_tile_offset
                            < tile_info[4]
                            else 0
                        )

                    tAgA = gA_mkl[(None, None, 0, None, 0)]
                    A_gmem_thread_offset = cute.assume((tidx_in_warpgroup % 8) * 32, divby=32)

                    a_producer_state.reset_count()
                    peek_a_empty_status = cutlass.Boolean(1)
                    if a_producer_state.count < k_tile_cnt:
                        peek_a_empty_status = a_pipeline.producer_try_acquire(a_producer_state)

                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        a_pipeline.producer_acquire(a_producer_state, peek_a_empty_status)

                        tAgA_ktile = tAgA[(None, None, a_producer_state.count)]
                        tAsA_ktile = tAsA_tiled[(None, None, None, a_producer_state.index)]

                        for i in range(self.a_num_loads):
                            A_gmem_slice_offset = A_gmem_thread_offset + cute.assume(
                                a_token_offset_tensor[i] * tAgA_ktile.layout[0].stride,
                                divby=32,
                            )
                            A_gmem_slice_offset = cute.assume(A_gmem_slice_offset, divby=32)
                            tAgA_slice_ptr = tAgA_ktile.iterator + A_gmem_slice_offset
                            tAgA_slice = cute.make_tensor(
                                tAgA_slice_ptr, layout=cute.make_layout((32,))
                            )
                            tAsA_slice = cute.make_tensor(
                                tAsA_ktile[(None, i, None)].iterator,
                                layout=cute.make_layout((32,)),
                            )
                            a_predicate_slice = cute.make_rmem_tensor(
                                cute.make_layout((1,)), cutlass.Boolean
                            )
                            a_predicate_slice[0] = a_predicate_tensor[i]
                            cute.copy_atom_call(
                                a_atom_copy,
                                tAgA_slice,
                                tAsA_slice,
                                pred=a_predicate_slice,
                            )

                        a_pipeline.producer_commit(a_producer_state)

                        a_producer_state.advance()
                        peek_a_empty_status = cutlass.Boolean(1)
                        if a_producer_state.count < k_tile_cnt:
                            peek_a_empty_status = a_pipeline.producer_try_acquire(a_producer_state)

                    tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                    for idx in cutlass.range(5, unroll_full=True):
                        tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                    is_valid_tile = tile_info[3] == 1
                    cute.arch.fence_proxy("async.shared", space="cta")
                    tile_info_pipeline.consumer_release(tile_info_consumer_state)
                    tile_info_consumer_state.advance()

                a_pipeline.producer_tail(a_producer_state)

            # tma-A: 4 warps × elect-one issuing TMA gather4. Each warp owns
            # 1/4 of the M rows and issues n_gather_per_warp gather4 calls
            # (each pulling 4 rows × cta_tile_K). token_id == -1 → TMA
            # zero-fills the row. Signals merged ab_pipeline mbar (shared w/ B).
            elif cutlass.const_expr(self.a_path == "tma"):
                warp_rel = warp_idx - self.gather_a_warp_id[0]
                rows_per_warp = self.cta_tile_shape_mnk[0] // self.tma_gather_num_warps
                n_gather_per_warp = rows_per_warp // 4

                a_row_ids = cute.make_rmem_tensor(
                    cute.make_layout((rows_per_warp,)),
                    cutlass.Int32,
                )

                tile_sched = utils.StaticPersistentTileScheduler.create(
                    tile_sched_params,
                    cute.arch.block_idx(),
                    cute.arch.grid_dim(),
                )
                work_tile = tile_sched.initial_work_tile_info()

                ab_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_ab_stage
                )
                tile_info_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_tile_stage
                )

                tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(5, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

                while is_valid_tile:
                    gToken_ml_tile = gToken_ml[(None, tile_info[0])]

                    # Each warp computes its own row range (1/4 of the tile).
                    # The routing helper initializes padded mapping entries to
                    # zero, so use mn_limit as the authoritative valid-row
                    # predicate before passing -1 to gather4 for zero-fill.
                    for i in range(rows_per_warp):
                        row_global = warp_rel * rows_per_warp + i
                        token_id = gToken_ml_tile[row_global]
                        valid_row = (
                            tile_info[0] * self.cta_tile_shape_mnk[0] + row_global < tile_info[4]
                        )
                        row_id = token_id // self.topk if valid_row else cutlass.Int32(-1)
                        a_row_ids[i] = cutlass.Int32(-1) if token_id == -1 else row_id

                    # A multicast leader gate: when A is N-multicast, only the N=0
                    # CTA issues the mcast PTX; HW broadcasts data + mbar tx_count
                    # to the N-peer. The peer's producer_acquire still sets its
                    # mbar's expect_tx; that mbar fires via HW routing.
                    is_a_mcast_leader = (not self.is_a_mcast) or (
                        block_in_cluster_coord_vmnk[2] == 0
                    )
                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        # gather_a contributes a_num_tma_load_bytes per warp to
                        # the merged ab_pipeline mbar (4 such arrivals per stage).
                        ab_pipeline.producer_acquire(
                            ab_producer_state,
                            expected_tx=self.a_num_tma_load_bytes,
                        )

                        if is_a_mcast_leader:
                            with cute.arch.elect_one():
                                col_k = k_tile * self.cta_tile_shape_mnk[2]
                                mbar_ptr = ab_pipeline.producer_get_barrier(ab_producer_state)
                                stage_base_elements = (
                                    ab_producer_state.index
                                    * self.cta_tile_shape_mnk[0]
                                    * self.cta_tile_shape_mnk[2]
                                )
                                warp_base_elements = (
                                    warp_rel * rows_per_warp * self.cta_tile_shape_mnk[2]
                                )

                                for g in range(n_gather_per_warp):
                                    row_start = g * 4
                                    dst_offset = (
                                        stage_base_elements
                                        + warp_base_elements
                                        + row_start * self.cta_tile_shape_mnk[2]
                                    )
                                    dst_ptr = sA.iterator + dst_offset
                                    sm100_tma_gather4_load(
                                        tma_atom_a,
                                        dst_ptr,
                                        mbar_ptr,
                                        col_k,
                                        a_row_ids[row_start],
                                        a_row_ids[row_start + 1],
                                        a_row_ids[row_start + 2],
                                        a_row_ids[row_start + 3],
                                        use_cta_group_2=self.use_2cta_instrs,
                                        mcast_mask=a_full_mcast_mask,
                                    )

                        ab_pipeline.producer_commit(ab_producer_state)
                        ab_producer_state.advance()

                    tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                    for idx in cutlass.range(5, unroll_full=True):
                        tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                    is_valid_tile = tile_info[3] == 1
                    cute.arch.fence_proxy("async.shared", space="cta")
                    tile_info_pipeline.consumer_release(tile_info_consumer_state)
                    tile_info_consumer_state.advance()

                ab_pipeline.producer_tail(ab_producer_state)

        # Gather SFA warps (warps 12-15): CpAsync128.CG per row. 4 warps × 32
        # threads, each loads one row (16B = 16 FP8 SFs). sSFA is plain linear
        # (M, tile_K_sf, stage). tiled_copy_tv for SMEM dest + manual per-thread
        # GMEM source tensor.
        if warp_idx <= self.gather_sfa_warp_id[-1] and warp_idx >= self.gather_sfa_warp_id[0]:
            cute.arch.setmaxregister_decrease(self.num_regs_gather_sfa_warps)

            sfa_tile_k_sf = self.cta_tile_shape_mnk[2] // self.sf_vec_size  # 16
            sfa_gather_threads = len(self.gather_sfa_warp_id) * self.threads_per_warp
            # Rows per thread = cta_tile_M / gather_threads.
            # - non-breuse cta_tile_M=128: 1 row per thread
            # - breuse cta_tile_M=256: 2 rows per thread (outer r-loop in inner k loop)
            sfa_rows_per_thread = self.cta_tile_shape_mnk[0] // sfa_gather_threads

            # One CpAsync128.CG per thread per row per k_tile.
            sfa_atom_copy = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                mSFA_mkl.element_type,
                num_bits_per_copy=128,
            )

            sSFA_tiled = cute.make_tensor(
                sSFA.iterator,
                layout=cute.make_layout(
                    (
                        self.cta_tile_shape_mnk[0],
                        sfa_tile_k_sf,
                        self.num_ab_stage,
                    ),
                    stride=(
                        sfa_tile_k_sf,
                        1,
                        self.cta_tile_shape_mnk[0] * sfa_tile_k_sf,
                    ),
                ),
            )

            tidx_in_warpgroup = tidx - self.gather_sfa_warp_id[0] * self.threads_per_warp

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            sfa_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(5, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                gToken_ml_tile = gToken_ml[(None, tile_info[0])]

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    sfa_pipeline.producer_acquire(sfa_producer_state)

                    # For each row this thread is responsible for
                    # (1 row at non-breuse cta_tile_M=128; 2 rows at breuse
                    # cta_tile_M=256). Inner r-loop fully unrolls.
                    for r in cutlass.range_constexpr(sfa_rows_per_thread):
                        row_in_cta = tidx_in_warpgroup + r * sfa_gather_threads

                        # Per-row token id → SFA gmem row index.
                        tok = gToken_ml_tile[row_in_cta]
                        sfa_row_id = cutlass.Int32(-1) if tok == -1 else tok // self.topk

                        # OOB predicate: rows past mn_limit do not store.
                        sfa_pred = cute.make_rmem_tensor(cute.make_layout((1,)), cutlass.Boolean)
                        sfa_pred[0] = (
                            cutlass.Boolean(1)
                            if (
                                tile_info[0] * self.cta_tile_shape_mnk[0] + row_in_cta
                                < tile_info[4]
                            )
                            else cutlass.Boolean(0)
                        )

                        # GMEM src for this row + k_tile.
                        tAgSFA_row = gSFA_mkl[(sfa_row_id, None, 0, None, 0)]
                        tAgSFA_ktile = tAgSFA_row[(None, k_tile)]
                        tAgSFA_slice = cute.make_tensor(
                            tAgSFA_ktile.iterator,
                            layout=cute.make_layout((sfa_tile_k_sf,)),
                        )

                        # SMEM dst: direct row-indexed slice (bypass partition_D
                        # so we naturally handle multiple rows per thread).
                        tAsSFA_slice = cute.make_tensor(
                            sSFA_tiled[(row_in_cta, None, sfa_producer_state.index)].iterator,
                            cute.make_layout((sfa_tile_k_sf,)),
                        )

                        cute.copy_atom_call(
                            sfa_atom_copy,
                            tAgSFA_slice,
                            tAsSFA_slice,
                            pred=sfa_pred,
                        )

                    sfa_pipeline.producer_commit(sfa_producer_state)
                    sfa_producer_state.advance()

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(5, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            sfa_pipeline.producer_tail(sfa_producer_state)

        #
        # SFA Transform warps (warps 16-19) — LDS + STTM consumer
        #
        if warp_idx >= self.sfa_transform_warp_id[0] and warp_idx <= self.sfa_transform_warp_id[-1]:
            cute.arch.setmaxregister_decrease(self.num_regs_sfa_transform_warps)

            if cutlass.const_expr(True):
                # Transform warps wait for TMEM alloc and compute
                # sfa_tmem_ptr = acc_tmem_ptr + offset for STTM destination.
                tmem.wait_for_alloc()
                acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                sfa_tmem_ptr = cute.recast_ptr(
                    acc_tmem_ptr + self.sfa_tmem_offset,
                    dtype=self.sf_dtype,
                )

                tile_sched = utils.StaticPersistentTileScheduler.create(
                    tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
                )
                _ = tile_sched.initial_work_tile_info()

                tile_info_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_tile_stage
                )
                sfa_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_ab_stage
                )
                # Producer state on sfa_transform_pipeline: tracks which TMEM
                # stage (of num_sfa_tmem_stage) is being filled this iter.
                sfa_transform_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer,
                    self.num_sfa_tmem_stage,
                )
                tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(5, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

                # All 4 transform warps run identical address logic: each does
                # the full cta_tile_M LDS+STTM into its own per-warp TMEM lane;
                # UMMA reads all 4 to cover M. M-blocks = cta_tile_M / 32 (4
                # non-breuse, 8 breuse). Per warp: num_m_blocks LDS.128 + 4 STTM.
                num_m_blocks = self.cta_tile_shape_mnk[0] // 32
                lane_in_warp = tidx % 32

                while is_valid_tile:
                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        sfa_pipeline.consumer_wait(sfa_consumer_state)

                        # LDS.128 → RMEM Uint32. Per ci read 16 SFs = 4 u32.
                        # Issue LDS before producer_acquire so SMEM→RMEM moves
                        # overlap with prior MMA still holding the next TMEM
                        # stage; STTM still gates on producer_acquire below.
                        sfa_rmem_u32 = cute.make_rmem_tensor(
                            cute.make_layout((num_m_blocks, 4)),  # (ci, K-group)
                            cutlass.Uint32,
                        )
                        # Linear sSFA (M, K_sf, stage). Each LDS reads 16
                        # contiguous FP8 SFs (4 u32) at row_global of the
                        # current stage.
                        for ci in range(num_m_blocks):
                            row_global = 32 * ci + lane_in_warp
                            smem_slice = sSFA[
                                (
                                    row_global,
                                    None,
                                    sfa_consumer_state.index,
                                )
                            ]
                            smem_slice_u32 = cute.make_tensor(
                                cute.recast_ptr(smem_slice.iterator, dtype=cutlass.Uint32),
                                cute.make_layout((4,)),
                            )
                            cute.autovec_copy(smem_slice_u32, sfa_rmem_u32[ci, None])

                        # Acquire SFA TMEM slot (MMA's UMMA consumer release
                        # frees it after consuming). Done after LDS so the
                        # SMEM→RMEM stage isn't blocked on TMEM availability.
                        sfa_transform_pipeline.producer_acquire(sfa_transform_producer_state)

                        # STTM via inline PTX. tCtSFA_layout cols:
                        #   non-breuse (M=128): 1 half, cols 0..15 = 4 K-groups
                        #     × 4 M-blocks (gi*4 stride). 4 STTM x4.
                        #   breuse (M=256): 2 halves at +16 col offset (keep
                        #     ci 0..3 / reuse ci 4..7), each 4 STTM x4. 8 STTM x4 total.
                        stage_idx_in_tmem = sfa_transform_producer_state.index
                        sfa_tmem_addr_base = (
                            acc_tmem_ptr
                            + self.sfa_tmem_offset
                            + stage_idx_in_tmem * self.num_sfa_tmem_cols_per_stage
                        ).toint()
                        # Number of "halves" (keep/reuse splits) and the stride
                        # between them. non-breuse: 1 half, no second offset.
                        # breuse: 2 halves, second at +16 cols.
                        num_halves = num_m_blocks // 4  # 1 or 2
                        half_col_stride = 16  # cols between keep and reuse
                        for half in range(num_halves):
                            half_addr_base = sfa_tmem_addr_base + half * half_col_stride
                            ci_base = half * 4
                            for gi in range(4):
                                sm100_tcgen05_st_32x32b_x4(
                                    half_addr_base + gi * 4,
                                    sfa_rmem_u32[ci_base + 0, gi],
                                    sfa_rmem_u32[ci_base + 1, gi],
                                    sfa_rmem_u32[ci_base + 2, gi],
                                    sfa_rmem_u32[ci_base + 3, gi],
                                )
                        # Make TMEM stores visible to UMMA, then commit the
                        # transform pipeline producer slot.
                        cute.arch.fence_view_async_tmem_store()
                        sfa_transform_pipeline.producer_commit(sfa_transform_producer_state)
                        sfa_transform_producer_state.advance()

                        sfa_pipeline.consumer_release(sfa_consumer_state)
                        sfa_consumer_state.advance()

                    tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                    for idx in cutlass.range(5, unroll_full=True):
                        tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                    is_valid_tile = tile_info[3] == 1
                    cute.arch.fence_proxy("async.shared", space="cta")
                    tile_info_pipeline.consumer_release(tile_info_consumer_state)
                    tile_info_consumer_state.advance()

                # Drain transform pipeline before exit.
                sfa_transform_pipeline.producer_tail(sfa_transform_producer_state)

        # A Sync-Transform Warp (warp 11). Active only on cpasync 2CTA.
        # Consumes per-CTA `a_pipeline` and re-produces cluster-wide
        # `a_sync_transform_pipeline` so MMA's cta_group::2 sees both CTAs' A.
        # SFA needs no relay — its transform warps + sfa_transform_pipeline
        # handle cluster arrives. Idle on 1CTA / tma-A.
        if warp_idx == self.sync_transform_warp_id:
            if cutlass.const_expr(self.a_path == "cpasync" and self.use_2cta_instrs):
                tile_sched = utils.StaticPersistentTileScheduler.create(
                    tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
                )
                work_tile = tile_sched.initial_work_tile_info()

                a_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_ab_stage
                )
                a_sync_transform_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_ab_stage
                )
                tile_info_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_tile_stage
                )

                # First tile info (only need the validity flag).
                valid_tile_info = cute.make_rmem_tensor((1,), cutlass.Int32)
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                valid_tile_info[0] = sInfo[(3, tile_info_consumer_state.index)]
                is_valid_tile = valid_tile_info[0] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

                while is_valid_tile:
                    a_consumer_state.reset_count()
                    peek_a_full_status = cutlass.Boolean(1)
                    if a_consumer_state.count < k_tile_cnt:
                        peek_a_full_status = a_pipeline.consumer_try_wait(a_consumer_state)
                    a_sync_transform_producer_state.reset_count()

                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        # Wait per-CTA A full → commit cluster-wide A sync-transform
                        # full. We do NOT release a_pipeline here; MMA owns its
                        # consumer_release so each CTA's producer sees the empty arrive.
                        a_pipeline.consumer_wait(a_consumer_state, peek_a_full_status)
                        a_sync_transform_pipeline.producer_commit(a_sync_transform_producer_state)
                        a_sync_transform_producer_state.advance()
                        a_consumer_state.advance()
                        peek_a_full_status = cutlass.Boolean(1)
                        if a_consumer_state.count < k_tile_cnt:
                            peek_a_full_status = a_pipeline.consumer_try_wait(a_consumer_state)

                    # Advance to next tile
                    tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                    valid_tile_info[0] = sInfo[(3, tile_info_consumer_state.index)]
                    is_valid_tile = valid_tile_info[0] == 1
                    cute.arch.fence_proxy("async.shared", space="cta")
                    tile_info_pipeline.consumer_release(tile_info_consumer_state)
                    tile_info_consumer_state.advance()

                # Drain A sync-transform pipeline before exit.
                a_sync_transform_pipeline.producer_tail(a_sync_transform_producer_state)

        # TMA B/SFB load warp (warp 9). Loads B/SFB GMEM → SMEM with multicast.
        if warp_idx == self.tma_b_warp_id:
            # B producer signals the pipeline owning A+B (tma → ab_pipeline)
            # or B alone (cpasync → b_pipeline). Body is the same; only the
            # pipeline alias and producer_acquire's expected_tx differ.
            if cutlass.const_expr(self.a_path == "cpasync"):
                _b_pipe = b_pipeline
            else:
                _b_pipe = ab_pipeline
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            # First tile
            work_tile = tile_sched.initial_work_tile_info()

            b_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get the first tile info
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )
                #
                # Slice to per mma tile index
                #
                # ((atom_v, rest_v), loopK)
                tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

                # Apply SFB slicing hack when cta_tile_shape_n=64
                slice_n = mma_tile_coord_mnl[1]
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    slice_n = mma_tile_coord_mnl[1] // 2

                # ((atom_v, rest_v), RestK)
                tBgSFB_slice = tBgSFB[(None, slice_n, None, mma_tile_coord_mnl[2])]

                # Peek (try_wait) B buffer empty for k_tile = prefetch_k_tile_cnt
                b_producer_state.reset_count()
                peek_b_empty_status = cutlass.Boolean(1)
                if b_producer_state.count < k_tile_cnt:
                    peek_b_empty_status = _b_pipe.producer_try_acquire(b_producer_state)
                #
                # Tma load loop
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # Conditionally wait for B buffer empty.
                    # tma-A mode passes expected_tx (ab_pipeline has tx_count=0
                    # at create and accumulates per producer call); cpasync-A
                    # mode's b_pipeline has tx_count fixed at create.
                    if cutlass.const_expr(self.a_path == "cpasync"):
                        _b_pipe.producer_acquire(b_producer_state, peek_b_empty_status)
                    else:
                        _b_pipe.producer_acquire(
                            b_producer_state,
                            peek_b_empty_status,
                            expected_tx=self.num_tma_load_bytes,
                        )

                    tBgB_k = tBgB_slice[(None, b_producer_state.count)]
                    tBgSFB_k = tBgSFB_slice[(None, b_producer_state.count)]
                    tBsB_pipe = tBsB[(None, b_producer_state.index)]
                    tBsSFB_pipe = tBsSFB[(None, b_producer_state.index)]

                    tma_bar = _b_pipe.producer_get_barrier(b_producer_state)

                    # TMA load B
                    cute.copy(
                        tma_atom_b,
                        tBgB_k,
                        tBsB_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=b_full_mcast_mask,
                    )

                    # TMA load SFB
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_k,
                        tBsSFB_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=sfb_full_mcast_mask,
                    )

                    # Peek (try_wait) B buffer empty for k_tile + 1
                    b_producer_state.advance()
                    peek_b_empty_status = cutlass.Boolean(1)
                    if b_producer_state.count < k_tile_cnt:
                        peek_b_empty_status = _b_pipe.producer_try_acquire(b_producer_state)

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            #
            # Wait B buffer empty
            #
            _b_pipe.producer_tail(b_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # Bar sync for retrieve tensor memory ptr from shared mem
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # SFA TMEM base ptr (sf_dtype). Stage-indexed tCtSFA is rebuilt
            # per k_tile inside the loop using sfa_transform_consumer_state.index.
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols,
                dtype=self.sf_dtype,
            )

            # Make SFB tmem tensor (using precomputed layout)
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            # SFA TMEM is filled by transform warps via LDS+STTM (no UTCCP).
            # Only SFB uses UTCCP.
            sfb_s2t_bundle = self._mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            # MMA consumer states. cpasync: separate a / b states
            # (+ a_sync_transform in 2CTA). tma: single ab_consumer_state.
            if cutlass.const_expr(self.a_path == "cpasync"):
                a_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_ab_stage
                )
                b_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_ab_stage
                )
                if cutlass.const_expr(self.use_2cta_instrs):
                    a_sync_transform_consumer_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, self.num_ab_stage
                    )
            else:  # "tma"
                ab_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_ab_stage
                )
                # a_consumer_state and b_consumer_state both aliased to
                # ab_consumer_state so existing `.index` reads in shared
                # SMEM-slice code (sA[stage] / sB[stage] in the MMA mainloop)
                # still resolve (A and B share stages in tma-A mode).
                a_consumer_state = ab_consumer_state
                b_consumer_state = ab_consumer_state
            # sfa_transform_pipeline has num_sfa_tmem_stage slots, not num_ab_stage.
            sfa_transform_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_sfa_tmem_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get the first tile info from pipeline (scheduler has filtered out tiles >= num_non_exiting_tiles)
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                # Peek (try_wait) A / B / SFA buffer full for k_tile = 0.
                # cpasync-A 1CTA: A peek on a_pipeline.
                # cpasync-A 2CTA: A peek on a_sync_transform_pipeline (cluster-wide).
                # tma-A: A+B peek on ab_pipeline (single shared mbar).
                if cutlass.const_expr(self.a_path == "cpasync"):
                    if cutlass.const_expr(self.use_2cta_instrs):
                        a_sync_transform_consumer_state.reset_count()
                        peek_a_sync_transform_full_status = cutlass.Boolean(1)
                        if a_sync_transform_consumer_state.count < k_tile_cnt and is_leader_cta:
                            peek_a_sync_transform_full_status = (
                                a_sync_transform_pipeline.consumer_try_wait(
                                    a_sync_transform_consumer_state
                                )
                            )
                        a_consumer_state.reset_count()
                    else:
                        a_consumer_state.reset_count()
                        peek_a_full_status = cutlass.Boolean(1)
                        if a_consumer_state.count < k_tile_cnt:
                            peek_a_full_status = a_pipeline.consumer_try_wait(a_consumer_state)
                    b_consumer_state.reset_count()
                    peek_b_full_status = cutlass.Boolean(1)
                    if b_consumer_state.count < k_tile_cnt and is_leader_cta:
                        peek_b_full_status = b_pipeline.consumer_try_wait(b_consumer_state)
                else:  # "tma"
                    ab_consumer_state.reset_count()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                        peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)
                sfa_transform_consumer_state.reset_count()
                peek_sfa_full_status = cutlass.Boolean(1)
                if sfa_transform_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_sfa_full_status = sfa_transform_pipeline.consumer_try_wait(
                        sfa_transform_consumer_state
                    )

                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )

                # Accumulator stage. Overlap mode: pipeline tracks 1 stage
                # but TMEM has 2 regions (acc[0] cols 0..255, acc[1] 192..447);
                # use phase to pick. producer_state.phase starts at 1, so XOR
                # with 1 to align with TMEM slot 0 on tile 0.
                if cutlass.const_expr(self.use_overlap_accum):
                    acc_stage_index = acc_producer_state.phase ^ 1
                else:
                    acc_stage_index = acc_producer_state.index

                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                # Apply TMEM pointer offset hack when cta_tile_shape_n=192 or
                # cta_tile_shape_n=64
                tCtSFB_mma = tCtSFB
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
                    # If this is an ODD tile, shift the TMEM start address for
                    # cta_tile_shape_n=192 case by two words
                    # (ignores first 64 columns of SFB)
                    offset = (
                        cutlass.Int32(2) if mma_tile_coord_mnl[1] % 2 == 1 else cutlass.Int32(0)
                    )
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + self.num_accumulator_tmem_cols
                        + self.num_sfa_tmem_cols
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)
                elif cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    # Move in increments of 64 columns of SFB
                    offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + self.num_accumulator_tmem_cols
                        + self.num_sfa_tmem_cols
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)
                    #
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)
                #
                # Mma mainloop
                #

                for k_tile in cutlass.range(k_tile_cnt):
                    # Set tensor memory buffer for current tile
                    # (MMA, MMA_M, MMA_N)

                    if is_leader_cta:
                        # Wait for A / B / SFA buffer full.
                        # cpasync-A: 2 separate waits (A side + B side).
                        #   2CTA A wait uses a_sync_transform_pipeline (relay).
                        # tma-A: single ab_pipeline wait covers A and B.
                        if cutlass.const_expr(self.a_path == "cpasync"):
                            if cutlass.const_expr(self.use_2cta_instrs):
                                a_sync_transform_pipeline.consumer_wait(
                                    a_sync_transform_consumer_state,
                                    peek_a_sync_transform_full_status,
                                )
                            else:
                                a_pipeline.consumer_wait(a_consumer_state, peek_a_full_status)
                            b_pipeline.consumer_wait(b_consumer_state, peek_b_full_status)
                            a_stage_idx = a_consumer_state.index
                            b_stage_idx = b_consumer_state.index
                        else:  # "tma"
                            ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)
                            # Read ab_consumer_state.index directly (NOT via
                            # the aliased a_/b_consumer_state names): CuteDSL
                            # caches .index SSA values per variable *name*,
                            # so aliased reads go stale after ab.advance().
                            a_stage_idx = ab_consumer_state.index
                            b_stage_idx = ab_consumer_state.index
                        sfa_transform_pipeline.consumer_wait(
                            sfa_transform_consumer_state, peek_sfa_full_status
                        )

                        # Rebuild tCtSFA pointing at the current SFA TMEM
                        # stage (transform warps rotate through stages;
                        # sfa_transform_consumer_state.index tracks which to read).
                        tCtSFA = cute.make_tensor(
                            sfa_tmem_ptr
                            + sfa_transform_consumer_state.index
                            * self.num_sfa_tmem_cols_per_stage
                            * 4,
                            tCtSFA_layout,
                        )

                        # SFB UTCCP (SFA is already in TMEM from transform warps).
                        self._mainloop_s2t_copies(b_stage_idx, sfb_s2t_bundle)

                        num_kblocks = cute.size(tCrA, mode=[2])

                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            if cutlass.const_expr(
                                self.enable_breuse
                                and cute.size(tCtAcc.layout, mode=[1]) == 2
                                and cute.size(tCtAcc.layout, mode=[2]) == 1
                            ):
                                tCtAcc_bkeep = tCtAcc[(None, 0, 0)]
                                tCtAcc_breuse = tCtAcc[(None, 1, 0)]

                                a_kblk_crd_keep = (None, 0, kblock_idx, a_stage_idx)
                                a_kblk_crd_reuse = (None, 1, kblock_idx, a_stage_idx)
                                b_kblk_crd = (None, 0, kblock_idx, b_stage_idx)

                                sfa_kblk_crd_keep = (None, 0, kblock_idx)
                                sfa_kblk_crd_reuse = (None, 1, kblock_idx)
                                sfb_kblk_crd = (None, 0, kblock_idx)

                                # Bkeep
                                tiled_mma_bkeep.set(
                                    tcgen05.Field.ACCUMULATE,
                                    k_tile != 0 or kblock_idx != 0,
                                )
                                cute.gemm(
                                    tiled_mma_bkeep,
                                    tCtAcc_bkeep,
                                    [tCrA[a_kblk_crd_keep], tCtSFA[sfa_kblk_crd_keep]],
                                    [tCrB[b_kblk_crd], tCtSFB_mma[sfb_kblk_crd]],
                                    tCtAcc_bkeep,
                                )
                                # Breuse
                                tiled_mma_breuse.set(
                                    tcgen05.Field.ACCUMULATE,
                                    k_tile != 0 or kblock_idx != 0,
                                )
                                cute.gemm(
                                    tiled_mma_breuse,
                                    tCtAcc_breuse,
                                    [
                                        tCrA[a_kblk_crd_reuse],
                                        tCtSFA[sfa_kblk_crd_reuse],
                                    ],
                                    [tCrB[b_kblk_crd], tCtSFB_mma[sfb_kblk_crd]],
                                    tCtAcc_breuse,
                                )
                            else:
                                a_kblock_coord = (None, None, kblock_idx, a_stage_idx)
                                b_kblock_coord = (None, None, kblock_idx, b_stage_idx)
                                sf_kblock_coord = (None, None, kblock_idx)

                                tiled_mma.set(
                                    tcgen05.Field.ACCUMULATE,
                                    k_tile != 0 or kblock_idx != 0,
                                )
                                cute.gemm(
                                    tiled_mma,
                                    tCtAcc,
                                    [tCrA[a_kblock_coord], tCtSFA[sf_kblock_coord]],
                                    [tCrB[b_kblock_coord], tCtSFB_mma[sf_kblock_coord]],
                                    tCtAcc,
                                )

                        # Release A/B/SFA buffer-empty (async arrive).
                        # cpasync: a_pipeline + a_sync_transform_pipeline
                        # (2CTA only) + b_pipeline. tma: single ab_pipeline.
                        if cutlass.const_expr(self.a_path == "cpasync"):
                            a_pipeline.consumer_release(a_consumer_state)
                            if cutlass.const_expr(self.use_2cta_instrs):
                                a_sync_transform_pipeline.consumer_release(
                                    a_sync_transform_consumer_state
                                )
                            b_pipeline.consumer_release(b_consumer_state)
                        else:  # "tma"
                            ab_pipeline.consumer_release(ab_consumer_state)
                        sfa_transform_pipeline.consumer_release(sfa_transform_consumer_state)

                    # Peek (try_wait) A / B / SFA buffer full for k_tile + 1.
                    if cutlass.const_expr(self.a_path == "cpasync"):
                        if cutlass.const_expr(self.use_2cta_instrs):
                            a_sync_transform_consumer_state.advance()
                            peek_a_sync_transform_full_status = cutlass.Boolean(1)
                            if a_sync_transform_consumer_state.count < k_tile_cnt and is_leader_cta:
                                peek_a_sync_transform_full_status = (
                                    a_sync_transform_pipeline.consumer_try_wait(
                                        a_sync_transform_consumer_state
                                    )
                                )
                            a_consumer_state.advance()
                        else:
                            a_consumer_state.advance()
                            peek_a_full_status = cutlass.Boolean(1)
                            if a_consumer_state.count < k_tile_cnt:
                                peek_a_full_status = a_pipeline.consumer_try_wait(a_consumer_state)
                        b_consumer_state.advance()
                        peek_b_full_status = cutlass.Boolean(1)
                        if b_consumer_state.count < k_tile_cnt and is_leader_cta:
                            peek_b_full_status = b_pipeline.consumer_try_wait(b_consumer_state)
                    else:  # "tma"
                        ab_consumer_state.advance()
                        peek_ab_full_status = cutlass.Boolean(1)
                        if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)
                    sfa_transform_consumer_state.advance()
                    peek_sfa_full_status = cutlass.Boolean(1)
                    if sfa_transform_consumer_state.count < k_tile_cnt and is_leader_cta:
                        peek_sfa_full_status = sfa_transform_pipeline.consumer_try_wait(
                            sfa_transform_consumer_state
                        )

                #
                # Async arrive accumulator buffer full(each kblock)
                #
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)

                # Peek (try_wait) Acc buffer empty for k_tile = k_tile + 1
                acc_producer_state.advance()

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            #
            # Wait for accumulator buffer empty
            #
            acc_pipeline.producer_tail(acc_producer_state)

        #
        # Specialized epilogue warps
        #
        if warp_idx <= self.epilog_warp_id[-1]:
            # Register reconfig: epilogue needs many regs for SwiGLU
            cute.arch.setmaxregister_increase(self.num_regs_epilogue_warps)
            #
            # Alloc tensor memory buffer
            #
            tmem.allocate(self.num_tmem_alloc_cols)

            #
            # Bar sync for retrieve tensor memory ptr from shared memory
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            # Epilogue partition: transform both accumulator and C layout.
            # transform_partitioned_tensor_layout merges (MMA_ATOM, MMA_M) → flat M.
            tCtAcc_transformed = transform_partitioned_tensor_layout(tCtAcc_base)
            tCgC_for_epi = transform_partitioned_tensor_layout(tCgC)

            epi_tidx = tidx % 128
            (
                tiled_copy_t2r,
                tTR_tAcc_base,
                tTR_rAcc_up,
                tTR_rAcc_gate,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_transformed, tCgC_for_epi, epi_tile, use_2cta_instrs
            )

            tTR_rC = None
            tiled_copy_r2s = None
            tRS_rC = None
            tRS_sC = None
            bSG_sC = None
            bSG_gC_partitioned = None
            tTR_rC = cute.make_rmem_tensor(tTR_rAcc_up.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = epilogue_smem_copy_and_partition(
                self, tiled_copy_t2r, tTR_rC, epi_tidx, sC
            )
            (
                tma_atom_c,
                bSG_sC,
                bSG_gC_partitioned,
            ) = self.epilog_gmem_copy_and_partition(
                epi_tidx, tma_atom_c, tCgC_for_epi, epi_tile, sC
            )

            if cutlass.const_expr(self.generate_sfc):
                norm_const = norm_const_tensor[0]
                # (EPI_TILE_M, EPI_TILE_N, RestM, RestN, RestL)
                gSFC_mnl = cute.local_tile(mSFC_mnl, epi_tile, (None, None, None))

                thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
                # (T2R, T2R_M, T2R_N, RestM, RestN, RestL)
                tCgSFC_mnl = thr_copy_t2r.partition_D(gSFC_mnl)
                tCgSFC_mnl = cute.filter_zeros(tCgSFC_mnl)
                # (T2R, T2R_M, T2R_N)
                tCrSFC = cute.make_rmem_tensor(
                    tCgSFC_mnl[(None, None, None, 0, 0, 0)].layout, self.sf_dtype
                )
                tCrSFC_pvscale = cute.make_rmem_tensor_like(tCrSFC, cutlass.Float32)

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            c_pipeline = None
            # Threads/warps participating in tma store pipeline
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=c_producer_group,
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get the first tile info
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)

            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            num_prev_subtiles = cutlass.Int32(0)
            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )
                #
                # Get alpha for current group
                #

                expert_idx = mma_tile_coord_mnl[2]
                alpha_val = alpha[expert_idx]

                #
                # Slice to per mma tile index
                #
                bSG_gC = None
                # ((ATOM_V, REST_V), EPI_M, EPI_N)
                bSG_gC = bSG_gC_partitioned[
                    (
                        None,
                        None,
                        None,
                        mma_tile_coord_mnl[0],
                        mma_tile_coord_mnl[1],
                        0,
                    )
                ]

                # Get accumulator stage index. Overlap mode uses phase
                # (alternates 0/1) and reverse-iterates acc[0]'s subtiles so
                # the overlap region (high cols of acc[0]) is consumed first
                # → early-release lets MMA write acc[1] in those cols.
                if cutlass.const_expr(self.use_overlap_accum):
                    # Consumer state starts at phase=0 (per make_pipeline_state)
                    # so phase directly maps to TMEM slot index.
                    acc_stage_index = acc_consumer_state.phase
                    reverse_subtile = (
                        cutlass.Boolean(True) if acc_stage_index == 0 else cutlass.Boolean(False)
                    )
                else:
                    acc_stage_index = acc_consumer_state.index

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_stage_index)]

                if cutlass.const_expr(self.generate_sfc):
                    # (T2R, T2R_M, T2R_N, RestM, RestN)
                    tCgSFC_mn = tCgSFC_mnl[
                        (
                            None,
                            None,
                            None,
                            None,
                            None,
                            0,
                        )
                    ]

                #
                acc_pipeline.consumer_wait(acc_consumer_state)

                # Activation epilogue. SwiGLU consumes interleaved [up, gate]
                # accumulator subtiles and halves N; Relu2 consumes each
                # accumulator subtile directly and preserves N.
                #   tTR_tAcc: (T2R, T2R_M, T2R_N, EPI_M, EPI_N, STAGE), sliced on STAGE.
                #   bSG_gC:   ((ATOM_V, REST_V), EPI_M, EPI_N, loopM, loopN, loopL).
                interleave_granularity = 64
                gate_offset = interleave_granularity // self.epi_tile_n
                epi_m_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                acc_n_subtile_cnt = cute.size(tTR_tAcc.shape, mode=[4])
                out_n_subtile_cnt = acc_n_subtile_cnt // 2 if self.is_gated else acc_n_subtile_cnt

                for epi_m_idx in cutlass.range(epi_m_cnt):
                    for out_n_idx in cutlass.range(out_n_subtile_cnt):
                        # Map output N subtile → acc N subtile. Each
                        # interleave block of 2*gate_offset subtiles is
                        # [up*gate_offset, gate*gate_offset]. acc[0] in
                        # overlap mode iterates in reverse (consume high-col
                        # overlap region first).
                        if cutlass.const_expr(self.use_overlap_accum):
                            real_out_n_idx = (
                                (out_n_subtile_cnt - 1 - out_n_idx)
                                if reverse_subtile
                                else out_n_idx
                            )
                        else:
                            real_out_n_idx = out_n_idx
                        if cutlass.const_expr(self.is_gated):
                            block_idx = real_out_n_idx // gate_offset
                            within_block = real_out_n_idx % gate_offset
                            up_n_subtile = block_idx * 2 * gate_offset + within_block
                            gate_n_subtile = (
                                block_idx * 2 * gate_offset + gate_offset + within_block
                            )
                        else:
                            up_n_subtile = real_out_n_idx
                        #
                        # Load accumulator from tensor memory buffer to register
                        #
                        tTR_tAcc_mn_up = tTR_tAcc[(None, None, None, epi_m_idx, up_n_subtile)]

                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn_up, tTR_rAcc_up)
                        if cutlass.const_expr(self.is_gated):
                            tTR_tAcc_mn_gate = tTR_tAcc[
                                (None, None, None, epi_m_idx, gate_n_subtile)
                            ]
                            cute.copy(tiled_copy_t2r, tTR_tAcc_mn_gate, tTR_rAcc_gate)

                        # Overlap mode: after iter 0 the up/gate LDTM has
                        # covered cols 192..255 (reverse for acc[0], forward
                        # for acc[1]). Fence + early-release so MMA can write
                        # the next stage into the overlap region without racing.
                        if cutlass.const_expr(self.use_overlap_accum):
                            if out_n_idx == 0:
                                cute.arch.fence_view_async_tmem_load()
                                with cute.arch.elect_one():
                                    acc_pipeline.consumer_release(acc_consumer_state)
                                acc_consumer_state.advance()

                        acc_vec_up = tTR_rAcc_up.load()
                        tCompute = cute.make_rmem_tensor(acc_vec_up.shape, self.acc_dtype)
                        if cutlass.const_expr(self.activation_type == ActivationType.Swiglu):
                            acc_vec_gate = tTR_rAcc_gate.load()
                            self._apply_swiglu_epilogue(
                                acc_vec_up, acc_vec_gate, alpha_val, tCompute
                            )
                        elif cutlass.const_expr(self.activation_type == ActivationType.Relu2):
                            self._apply_relu2_epilogue(acc_vec_up, alpha_val, tCompute)

                        if cutlass.const_expr(self.generate_sfc):
                            # Float4E2M1FN quantization: per-vector absmax →
                            # SFC → store SFC to gmem → quantize output by
                            # reciprocal of SFC. (Subtile is partitioned on N.)
                            # locality domain: shift the SFC N-subtile by c_sf_n_tile_offset
                            # so this partition writes into the shared full-width
                            # SF buffer at the correct N tile (0 in non-locality domain).
                            sfc_subtile_idx_mn = (
                                tile_info[0] * self.epi_tile_cnt[0] + epi_m_idx,
                                c_sf_n_tile_offset
                                + tile_info[1] * self.epi_tile_cnt[1]
                                + real_out_n_idx,
                            )
                            tCgSFC = tCgSFC_mn[
                                (
                                    None,
                                    None,
                                    None,
                                    *sfc_subtile_idx_mn,
                                )
                            ]

                            #
                            # Get absolute max across a vector and Compute SFC
                            #
                            tTR_rAcc_frg = cute.logical_divide(
                                tCompute, cute.make_layout(self.sf_vec_size)
                            )
                            acc_frg = tTR_rAcc_frg.load()
                            acc_frg = epilogue_op(acc_frg)

                            # Apply element-wise absolute value using math.absf (supports vectors)
                            abs_acc_frg_ir = math.absf(acc_frg.ir_value())
                            abs_acc_frg = type(acc_frg)(
                                abs_acc_frg_ir, acc_frg.shape, acc_frg.dtype
                            )

                            if cutlass.const_expr(self.vectorized_f32):
                                for vi in cutlass.range_constexpr(abs_acc_frg.shape[1]):
                                    tCrSFC_pvscale[vi] = abs_acc_frg[None, vi].reduce(
                                        cute.ReductionOp.MAX,
                                        cutlass.Float32(0.0),
                                        0,  # Use 0.0 as init for abs values
                                    )
                                for vi in cutlass.range_constexpr(0, abs_acc_frg.shape[1], 2):
                                    tCrSFC_pvscale[vi], tCrSFC_pvscale[vi + 1] = (
                                        cute.arch.mul_packed_f32x2(
                                            (
                                                tCrSFC_pvscale[vi],
                                                tCrSFC_pvscale[vi + 1],
                                            ),
                                            (
                                                self.get_dtype_rcp_limits(self.c_dtype),
                                                self.get_dtype_rcp_limits(self.c_dtype),
                                            ),
                                        )
                                    )
                                    tCrSFC_pvscale[vi], tCrSFC_pvscale[vi + 1] = (
                                        cute.arch.mul_packed_f32x2(
                                            (
                                                tCrSFC_pvscale[vi],
                                                tCrSFC_pvscale[vi + 1],
                                            ),
                                            (norm_const, norm_const),
                                        )
                                    )
                            else:
                                for vi in cutlass.range_constexpr(abs_acc_frg.shape[1]):
                                    tCrSFC_pvscale[vi] = (
                                        abs_acc_frg[None, vi].reduce(
                                            cute.ReductionOp.MAX,
                                            cutlass.Float32(0.0),
                                            0,  # Use 0.0 as init for abs values
                                        )
                                        * self.get_dtype_rcp_limits(self.c_dtype)
                                        * norm_const
                                    )

                            # TODO: f32x2 -> f8x2 conversion
                            tCrSFC.store(tCrSFC_pvscale.load().to(self.sf_dtype))

                            # Store SFC to gmem.
                            # TODO: predicate (cute.elem_less)
                            cute.autovec_copy(tCrSFC, tCgSFC)

                            # Quantize output and convert to c_dtype.
                            # TODO: need to add f8x2 -> f32x2 conversion
                            tCrSFC_qpvscale_up = tCrSFC.load().to(cutlass.Float32)
                            fp32_max = cutlass.Float32(3.40282346638528859812e38)
                            if cutlass.const_expr(self.vectorized_f32):
                                for vi in cutlass.range_constexpr(0, cute.size(tCrSFC), 2):
                                    acc_scale = cute.arch.mul_packed_f32x2(
                                        (
                                            cute.arch.rcp_approx(tCrSFC_qpvscale_up[vi]),
                                            cute.arch.rcp_approx(tCrSFC_qpvscale_up[vi + 1]),
                                        ),
                                        (norm_const, norm_const),
                                    )
                                    acc_scale_min0 = fmin(acc_scale[0], fp32_max, nan=True)
                                    acc_scale_min1 = fmin(acc_scale[1], fp32_max, nan=True)

                                    vec0 = tTR_rAcc_frg[None, vi]
                                    vec1 = tTR_rAcc_frg[None, vi + 1]
                                    for ei in cutlass.range_constexpr(self.sf_vec_size):
                                        vec0[ei], vec1[ei] = cute.arch.mul_packed_f32x2(
                                            (vec0[ei], vec1[ei]),
                                            (acc_scale_min0, acc_scale_min1),
                                        )
                            else:
                                for vi in cutlass.range_constexpr(cute.size(tCrSFC)):
                                    # TODO:Need to add E8M0 rcp approximation
                                    acc_scale = norm_const * cute.arch.rcp_approx(
                                        tCrSFC_qpvscale_up[vi]
                                    )
                                    acc_scale = fmin(acc_scale, fp32_max, nan=True)

                                    vec = tTR_rAcc_frg[None, vi]
                                    for ei in cutlass.range_constexpr(self.sf_vec_size):
                                        vec[ei] = vec[ei] * acc_scale

                            acc_vec = tiled_copy_r2s.retile(tCompute).load()
                            tRS_rC.store(acc_vec.to(self.c_dtype))
                        else:
                            #
                            # Convert to C type
                            #
                            acc_vec = tiled_copy_r2s.retile(tCompute).load()
                            acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
                            tRS_rC.store(acc_vec)

                        #
                        # Store C to shared memory
                        #
                        num_prev_subtiles = num_prev_subtiles + 1
                        c_buffer = num_prev_subtiles % self.num_c_stage

                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rC,
                            tRS_sC[(None, None, None, c_buffer)],
                        )
                        # Fence and barrier to make sure shared memory store is visible to TMA store
                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )
                        self.epilog_sync_barrier.arrive_and_wait()
                        #
                        # TMA store C to global memory
                        #
                        if warp_idx == self.epilog_warp_id[0]:
                            cute.copy(
                                tma_atom_c,
                                bSG_sC[(None, c_buffer)],
                                bSG_gC[(None, epi_m_idx, real_out_n_idx)],
                            )
                            # Fence and barrier to make sure shared memory store is visible to TMA store
                            c_pipeline.producer_commit()
                            c_pipeline.producer_acquire()
                        self.epilog_sync_barrier.arrive_and_wait()

                #
                # Async arrive accumulator buffer empty. Overlap mode already
                # released early inside the subtile loop; skip the final one.
                #
                if cutlass.const_expr(not self.use_overlap_accum):
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
            #
            # Wait for C store complete
            #
            c_pipeline.producer_tail()

        cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def _apply_swiglu_epilogue(
        self,
        acc_vec_up: cute.Tensor,
        acc_vec_gate: cute.Tensor,
        alpha_val,
        tCompute: cute.Tensor,
    ):
        """SwiGLU: ``tCompute[i] = (alpha * up[i]) * silu(alpha * gate[i])``."""
        if cutlass.const_expr(self.vectorized_f32):
            LOG2_E = cutlass.Float32(1.4426950408889634)
            for i in cutlass.range_constexpr(0, cute.size(acc_vec_up.shape), 2):
                acc_vec_up_alpha = cute.arch.mul_packed_f32x2(
                    (acc_vec_up[i], acc_vec_up[i + 1]),
                    (cutlass.Float32(alpha_val), cutlass.Float32(alpha_val)),
                )
                acc_vec_gate_alpha = cute.arch.mul_packed_f32x2(
                    (acc_vec_gate[i], acc_vec_gate[i + 1]),
                    (cutlass.Float32(alpha_val), cutlass.Float32(alpha_val)),
                )
                tCompute_log2e = cute.arch.mul_packed_f32x2(
                    (acc_vec_gate_alpha[0], acc_vec_gate_alpha[1]),
                    (-LOG2_E, -LOG2_E),
                )
                (
                    tCompute[i],
                    tCompute[i + 1],
                ) = cute.arch.add_packed_f32x2(
                    (
                        cute.math.exp2(tCompute_log2e[0], fastmath=True),
                        cute.math.exp2(tCompute_log2e[1], fastmath=True),
                    ),
                    (1.0, 1.0),
                )
                tCompute[i] = cute.arch.rcp_approx(tCompute[i])
                tCompute[i + 1] = cute.arch.rcp_approx(tCompute[i + 1])
                (
                    tCompute[i],
                    tCompute[i + 1],
                ) = cute.arch.mul_packed_f32x2(
                    (tCompute[i], tCompute[i + 1]),
                    (acc_vec_gate_alpha[0], acc_vec_gate_alpha[1]),
                )
                (
                    tCompute[i],
                    tCompute[i + 1],
                ) = cute.arch.mul_packed_f32x2(
                    (tCompute[i], tCompute[i + 1]),
                    (acc_vec_up_alpha[0], acc_vec_up_alpha[1]),
                )
        else:
            for i in cutlass.range_constexpr(cute.size(acc_vec_up.shape)):
                acc_vec_up_alpha = acc_vec_up[i] * cutlass.Float32(alpha_val)
                acc_vec_gate_alpha = acc_vec_gate[i] * cutlass.Float32(alpha_val)
                tCompute[i] = acc_vec_up_alpha * silu_f32(acc_vec_gate_alpha, fastmath=True)

    @cute.jit
    def _apply_relu2_epilogue(
        self,
        acc_vec_up: cute.Tensor,
        alpha_val,
        tCompute: cute.Tensor,
    ):
        """Relu2: ``tCompute[i] = relu(alpha * up[i]) ** 2``."""
        if cutlass.const_expr(self.vectorized_f32):
            for i in cutlass.range_constexpr(0, cute.size(acc_vec_up.shape), 2):
                scaled = cute.arch.mul_packed_f32x2(
                    (acc_vec_up[i], acc_vec_up[i + 1]),
                    (cutlass.Float32(alpha_val), cutlass.Float32(alpha_val)),
                )
                relu0 = cute.arch.fmax(scaled[0], 0.0)
                relu1 = cute.arch.fmax(scaled[1], 0.0)
                (
                    tCompute[i],
                    tCompute[i + 1],
                ) = cute.arch.mul_packed_f32x2(
                    (relu0, relu1),
                    (relu0, relu1),
                )
        else:
            for i in cutlass.range_constexpr(cute.size(acc_vec_up.shape)):
                scaled = acc_vec_up[i] * cutlass.Float32(alpha_val)
                relu_val = cute.arch.fmax(scaled, 0.0)
                tCompute[i] = relu_val * relu_val

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for tensor memory load, then use it to partition tensor memory
        (source) and register array (destination).

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param tAcc: The accumulator tensor to be copied and partitioned
        :type tAcc: cute.Tensor
        :param gC_mnl: The global tensor C
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param use_2cta_instrs: Whether use_2cta_instrs is enabled
        :type use_2cta_instrs: bool

        :return: A tuple containing (tiled_copy_t2r, tTR_tAcc, tTR_rAcc_up, tTR_rAcc_gate) where:
            - tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
            - tTR_tAcc: The partitioned accumulator tensor
            - tTR_rAcc_up: The partitioned accumulator tensor for acc up
            - tTR_rAcc_gate: The partitioned accumulator tensor for acc gate
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor, cute.Tensor]
        """
        # Make tiledCopy for tensor memory load (Rubin uses transformed layout)
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )

        # tAcc is already transformed: (M, N, STAGE) layout
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(
            tAcc,
            epi_tile,
        )
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # gC_mnl is already transformed: (M, N_half, loopM, loopN, loopL)
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN, loopL)
        gC_mnl_epi = cute.flat_divide(gC_mnl, epi_tile)

        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, loopM, loopN, loopL)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)

        # (T2R, T2R_M, T2R_N)
        tTR_rAcc_up = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc_gate = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc_up, tTR_rAcc_gate

    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for shared memory store, then use it to partition register
        array (source) and shared memory (destination).

        :param tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
        :type tiled_copy_t2r: cute.TiledCopy
        :param tTR_rC: The partitioned accumulator tensor
        :type tTR_rC: cute.Tensor
        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor
        :type sepi: cute.Tensor

        :return: A tuple containing (tiled_copy_r2s, tRS_rC, tRS_sC) where:
            - tiled_copy_r2s: The tiled copy operation for register to smem copy(r2s)
            - tRS_rC: The partitioned tensor C (register source)
            - tRS_sC: The partitioned tensor C (smem destination)
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        # (R2S, R2S_M, R2S_N)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def epilog_gmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        atom: Union[cute.CopyAtom, cute.TiledCopy],
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
    ) -> Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        """Make tiledCopy for global memory store, then use it to:
        - partition register array (source) and global memory (destination) for none TMA store version;
        - partition shared memory (source) and global memory (destination) for TMA store version.

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param atom: The copy_atom_c to be used for TMA store version, or tiled_copy_t2r for none TMA store version
        :type atom: cute.CopyAtom or cute.TiledCopy
        :param gC_mnl: The global tensor C
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor

        :return: A tuple containing :
            - For TMA store: (tma_atom_c, bSG_sC, bSG_gC) where:
                - tma_atom_c: The TMA copy atom
                - bSG_sC: The partitioned shared memory tensor C
                - bSG_gC: The partitioned global tensor C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]
        """
        # gC_mnl is already transformed: (M, N_half, loopM, loopN, loopL)
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN, loopL)
        gC_epi = cute.flat_divide(gC_mnl, epi_tile)
        tma_atom_c = atom
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        # ((ATOM_V, REST_V), EPI_M, EPI_N, loopM, loopN, loopL)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            gC_for_tma_partition,
        )
        return tma_atom_c, bSG_sC, bSG_gC

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        cta_tile_shape_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        epi_tile: cute.Tile,
        c_dtype: Type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        num_smem_capacity: int,
        occupancy: int,
        with_breuse: bool = False,
    ) -> Tuple[int, int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler_mnk: The shape (M, N, K) of the MMA tiler.
        :type mma_tiler_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param epi_tile: The epilogue tile shape.
        :type epi_tile: cute.Tile
        :param c_dtype: Data type of operand C (output).
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: Layout of operand C.
        :type c_layout: utils.LayoutEnum
        :param sf_dtype: Data type of scale factor.
        :type sf_dtype: type[cutlass.Numeric]
        :param sf_vec_size: Vector size of scale factor.
        :type sf_vec_size: int
        :param num_smem_capacity: Total available shared memory capacity in bytes.
        :type num_smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (ACC stages, A/B operand stages, C stages)
        :rtype: tuple[int, int, int]
        """
        # Default ACC stages
        num_acc_stage = 1 if (with_breuse and mma_tiler_mnk[1] in {192, 256}) else 2

        # Default C stages
        num_c_stage = 2

        # Default Tile info stages
        num_tile_stage = 2

        # Calculate smem layout and size for one stage of A, B, and C
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,  # a tmp 1 stage is provided
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,  # a tmp 1 stage is provided
        )

        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )

        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,
        )

        # SFA SMEM is plain linear (M, tile_K_sf), no pad.
        # Per stage = cta_tile_M × tile_K_sf bytes (FP8 = 1 byte/element).
        sfa_tile_k_sf = cta_tile_shape_mnk[2] // sf_vec_size
        sf_bytes_per_row = sfa_tile_k_sf * sf_dtype.width // 8
        sfa_bytes_per_stage_one = cta_tile_shape_mnk[0] * sf_bytes_per_row

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + sfa_bytes_per_stage_one
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        )
        # 1024B alignment
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage

        # Calculate A/B stages:
        # Start with total smem per CTA (capacity / occupancy)
        # Subtract reserved bytes and initial C stages bytes
        # Divide remaining by bytes needed per A/B stage
        num_ab_stage = (
            num_smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B stages and reserved bytes
        # Add remaining unused smem to epilogue
        num_c_stage += (
            num_smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes)
        ) // (occupancy * c_bytes_per_stage)
        return num_acc_stage, num_ab_stage, num_c_stage, num_tile_stage

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
        raster_along_m: bool = False,
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        """Use persistent tile scheduler to compute the grid size for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type cta_tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Constexpr

        :return: A tuple containing:
            - tile_sched_params: Parameters for the persistent tile scheduler.
            - grid: Grid shape for kernel launch.
        :rtype: Tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]
        """
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl, raster_along_m=raster_along_m
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    @staticmethod
    def _get_tma_atom_kind(
        atom_sm_cnt: cutlass.Int32, mcast: cutlass.Boolean
    ) -> Union[cpasync.CopyBulkTensorTileG2SMulticastOp, cpasync.CopyBulkTensorTileG2SOp]:
        """
        Select the appropriate TMA copy atom based on the number of SMs and the multicast flag.

        :param atom_sm_cnt: The number of SMs
        :type atom_sm_cnt: cutlass.Int32
        :param mcast: The multicast flag
        :type mcast: cutlass.Boolean

        :return: The appropriate TMA copy atom kind
        :rtype: cpasync.CopyBulkTensorTileG2SMulticastOp or cpasync.CopyBulkTensorTileG2SOp

        :raise ValueError: If the atom_sm_cnt is invalid
        """
        if atom_sm_cnt == 2 and mcast:
            return cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.TWO)
        elif atom_sm_cnt == 2 and not mcast:
            return cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO)
        elif atom_sm_cnt == 1 and mcast:
            return cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.ONE)
        elif atom_sm_cnt == 1 and not mcast:
            return cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)

        raise ValueError(f"Invalid atom_sm_cnt: {atom_sm_cnt} and {mcast}")

    @staticmethod
    def get_dtype_rcp_limits(dtype: Type[cutlass.Numeric]) -> float:
        """
        Calculates the reciprocal of the maximum absolute value for a given data type.

        :param dtype: Data type
        :type dtype: Type[cutlass.Numeric]

        :return: An float representing the reciprocal of the maximum absolute value
        :rtype: float
        """
        if dtype == cutlass.Float4E2M1FN:
            return 1 / 6.0
        if dtype == cutlass.Float8E4M3FN:
            return 1 / 448.0
        if dtype == cutlass.Float8E5M2:
            return 1 / 128.0
        return 1.0

    @staticmethod
    def is_valid_dtypes_and_scale_factor_vec_size(
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """
        Check if the dtypes are valid

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factor
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: The vector size of the scale factor
        :type sf_vec_size: int
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]

        :return: True if the dtypes are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        if ab_dtype not in {
            cutlass.Float4E2M1FN,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
            is_valid = False

        # Check valid sf_vec_size
        if sf_vec_size not in {16, 32}:
            is_valid = False

        # Check valid sf_dtype
        if sf_dtype not in {cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN}:
            is_valid = False

        # Check valid sf_dtype and sf_vec_size combinations
        if sf_dtype == cutlass.Float8E4M3FN and sf_vec_size == 32:
            is_valid = False
        if ab_dtype in {cutlass.Float8E5M2, cutlass.Float8E4M3FN} and sf_vec_size == 16:
            is_valid = False

        # Check valid c_dtype
        if c_dtype not in {
            cutlass.Float32,
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
            cutlass.Float4E2M1FN,
        }:
            is_valid = False

        return is_valid

    @staticmethod
    def is_valid_layouts(
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if layouts and dtypes are valid combinations

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: The major dimension of the A tensor
        :type a_major: str
        :param b_major: The major dimension of the B tensor
        :type b_major: str
        :param c_major: The major dimension of the C tensor
        :type c_major: str

        :return: True if the layouts are valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        if ab_dtype is cutlass.Float4E2M1FN and not (a_major == "k" and b_major == "k"):
            is_valid = False
        if c_dtype is cutlass.Float4E2M1FN and c_major == "m":
            is_valid = False
        return is_valid

    @staticmethod
    def is_valid_mma_tiler_and_cluster_shape(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        mma_inst_shape: Tuple[int, int, int],
        mma_tiler: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
    ) -> bool:
        """Check if the mma tiler and cluster shape are valid."""
        # Check valid mma_inst_shape
        if mma_inst_shape[0] not in [128, 256]:
            return False
        # SwiGLU Fusion requires even epi_tile counts
        if mma_inst_shape[1] not in [128, 256]:
            return False

        # Check valid mma_tiler
        if mma_tiler[0] not in [128, 256, 512]:
            return False
        if mma_tiler[1] not in [128, 256]:
            return False

        # Check MMA tiler vs MMA instruction relationship
        # mma_tiler[0] == mma_inst_shape[0] (no B-reuse) or 2 * mma_inst_shape[0] (B-reuse)
        if mma_tiler[0] not in (mma_inst_shape[0], 2 * mma_inst_shape[0]):
            return False
        if mma_tiler[1] != mma_inst_shape[1]:
            return False

        # Check K-dimension constraints based on data type
        if a_dtype in {cutlass.Float8E4M3FN, cutlass.Float8E5M2} and b_dtype in {
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }:
            if mma_tiler[2] != 128 or mma_inst_shape[2] != 64:
                return False
        else:
            if mma_tiler[2] != 256 or mma_inst_shape[2] != 128:
                return False

        # Cluster-M constraint: cluster_M must EQUAL the MMA CTA-group size along M
        # (atom_cta_m): 1 for 1-CTA MMA, 2 for 2-CTA MMA (mma_inst_shape[0] == 256).
        # Splitting the gathered-token (M) dimension across MORE cluster CTAs than the
        # MMA group (cluster_M > atom_cta_m) is not correctly handled by the gather /
        # tile-scheduler row mapping and produces wrong output rows (verified: ~1-2%
        # of rows mismatch on the nvf4 accuracy sweep), so reject it.
        # Cluster-N multicast of A is unaffected and remains supported.
        atom_cta_m = 2 if mma_inst_shape[0] == 256 else 1
        if cluster_shape_mn[0] != atom_cta_m:
            return False

        # Check cluster shape validity
        def _is_power_of_2(x):
            return x > 0 and (x & (x - 1)) == 0

        if (
            cluster_shape_mn[0] * cluster_shape_mn[1] > 16
            or cluster_shape_mn[0] <= 0
            or cluster_shape_mn[1] <= 0
            or cluster_shape_mn[0] > 4
            or cluster_shape_mn[1] > 4
            or not _is_power_of_2(cluster_shape_mn[0])
            or not _is_power_of_2(cluster_shape_mn[1])
        ):
            return False

        return True

    @staticmethod
    def is_valid_tensor_alignment(
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,  # noqa: E741
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if the tensor alignment is valid

        :param m: The number of rows in the A tensor
        :type m: cutlass.Int64
        :param n: The number of columns in the B tensor
        :type n: cutlass.Int64
        :param k: The number of columns in the A tensor
        :type k: cutlass.Int64
        :param l: The number of columns in the C tensor
        :type l: cutlass.Int64
        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param c_major: The major axis of the C tensor
        :type c_major: str

        :return: True if the problem shape is valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        if (
            not check_contigous_16B_alignment(ab_dtype, a_major == "m", (m, k, l))
            or not check_contigous_16B_alignment(ab_dtype, b_major == "n", (n, k, l))
            or not check_contigous_16B_alignment(c_dtype, c_major == "m", (m, n, l))
        ):
            is_valid = False
        return is_valid

    @classmethod
    def can_implement(
        cls,
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
        mma_inst_shape: Tuple[int, int, int],
        mma_tiler: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,  # noqa: E741
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if the gemm can be implemented

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factor
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: The vector size of the scale factor
        :type sf_vec_size: int
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]
        :param m: The number of rows in the A tensor
        :type m: cutlass.Int64
        :param n: The number of columns in the B tensor
        :type n: cutlass.Int64
        :param k: The number of columns in the A tensor
        :type k: cutlass.Int64
        :param l: The number of columns in the C tensor
        :type l: cutlass.Int64
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param c_major: The major axis of the C tensor
        :type c_major: str

        :return: True if the gemm can be implemented, False otherwise
        :rtype: bool
        """
        # Check data types
        if not cls.is_valid_dtypes_and_scale_factor_vec_size(
            a_dtype, sf_dtype, sf_vec_size, c_dtype
        ):
            return False

        # Check layouts
        if not cls.is_valid_layouts(a_dtype, c_dtype, a_major, b_major, c_major):
            return False

        # Check MMA tiler and cluster shape
        if not cls.is_valid_mma_tiler_and_cluster_shape(
            a_dtype, b_dtype, mma_inst_shape, mma_tiler, cluster_shape_mn
        ):
            return False

        # Check tensor alignment
        if not cls.is_valid_tensor_alignment(
            m, n, k, l, a_dtype, c_dtype, a_major, b_major, c_major
        ):
            return False

        # Check A/B layout
        if not (a_major == "k" and b_major == "k"):
            return False
        return True

    @cute.jit
    def wrapper(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        c_sf_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        tile_idx_to_group_idx_ptr: cute.Pointer,
        tile_idx_to_mn_limit_ptr: cute.Pointer,
        token_id_mapping_ptr: cute.Pointer,
        num_non_exiting_tiles_ptr: cute.Pointer,
        global_sf_ptr: cute.Pointer,
        orig_m: cutlass.Int64,
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,  # noqa: E741
        tile_size: cutlass.Constexpr,
        scaling_vector_size: cutlass.Constexpr,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
        c_stride_m: cutlass.Int64 = cutlass.Int64(0),
        c_sf_n_tile_offset: cutlass.Int64 = cutlass.Int64(0),
    ):
        scale_k = k // scaling_vector_size
        interm_size = n // 2 if self.is_gated else n
        num_tiles = m // tile_size
        a = cute.make_tensor(
            a_ptr, layout=cute.make_ordered_layout((orig_m, k, 1), order=(1, 0, 2))
        )
        b = cute.make_tensor(b_ptr, layout=cute.make_ordered_layout((n, k, l), order=(1, 0, 2)))
        a_sf = cute.make_tensor(
            a_sf_ptr,
            layout=cute.make_ordered_layout((orig_m, scale_k, 1), order=(1, 0, 2)),
        )
        b_sf = cute.make_tensor(
            b_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, n // 128, 4, scale_k // 4, l), order=(2, 1, 4, 0, 3, 5)
            ),
        )
        # c: runtime Int64 row stride. For locality domain half-GEMM, two partitions
        # interleave their N-halves into one shared full-width buffer
        # (c_stride_m = full intermediate size). A runtime stride also avoids a
        # cutlass-dsl MLIR alignment bug seen with
        # make_layout(..., stride=ordered_layout.stride). c_stride_m == 0 ->
        # natural interm_size stride (non-locality domain, == make_ordered_layout).
        actual_c_stride_m = interm_size if c_stride_m == 0 else c_stride_m
        c = cute.make_tensor(
            c_ptr,
            layout=cute.make_layout(
                (m, interm_size, 1),
                stride=(actual_c_stride_m, 1, m * actual_c_stride_m),
            ),
        )
        # full_c_shape gives SFC the full-N M-tile stride in locality domain mode so the
        # shared SF buffer is written without copy-back; None → use c.shape.
        if cutlass.const_expr(not self.locality_domain_half_gemm):
            full_c_shape = None
        else:
            full_interm_size = 2 * interm_size
            full_c_shape = cute.make_ordered_layout((m, full_interm_size, 1), order=(0, 1, 2)).shape
        c_sf = cute.make_tensor(
            c_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, m // 128, 4, interm_size // (scaling_vector_size * 4), l),
                order=(2, 1, 4, 0, 3, 5),
            ),
        )
        alpha = cute.make_tensor(alpha_ptr, layout=cute.make_layout((l,)))

        tile_idx_to_group_idx = cute.make_tensor(
            tile_idx_to_group_idx_ptr, layout=cute.make_layout((num_tiles,))
        )
        tile_idx_to_mn_limit = cute.make_tensor(
            tile_idx_to_mn_limit_ptr, layout=cute.make_layout((num_tiles,))
        )
        token_id_mapping = cute.make_tensor(token_id_mapping_ptr, layout=cute.make_layout((m,)))
        num_non_exiting_tiles = cute.make_tensor(
            num_non_exiting_tiles_ptr, layout=cute.make_layout((1,))
        )
        global_sf = cute.make_tensor(global_sf_ptr, layout=cute.make_layout((1,)))

        return self(
            a,
            b,
            c,
            a_sf,
            b_sf,
            c_sf,
            full_c_shape,
            global_sf,
            tile_idx_to_group_idx,
            tile_idx_to_mn_limit,
            token_id_mapping,
            num_non_exiting_tiles,
            alpha,
            max_active_clusters=max_active_clusters,
            stream=stream,
            epilogue_op=epilogue_op,
            c_sf_n_tile_offset=c_sf_n_tile_offset,
        )


@cute.jit
def cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
    sf_ref_tensor: cute.Tensor,
    sf_mma_tensor: cute.Tensor,
):
    """Convert scale factors from MKL layout to the MMA scale-factor layout."""
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 0, 3)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 1, 3)
    for i in cutlass.range(cute.size(sf_ref_tensor)):
        mkl_coord = sf_ref_tensor.layout.get_hier_coord(i)
        sf_mma_tensor[mkl_coord] = sf_ref_tensor[mkl_coord]


@cute.jit
def cvt_sf_M32x4xrm_K4xrk_L_to_MKL(
    sf_swizzled_tensor: cute.Tensor,
    sf_unswizzled_tensor: cute.Tensor,
):
    """Convert scale factors from the MMA scale-factor layout to MKL layout."""
    sf_swizzled_tensor = cute.group_modes(sf_swizzled_tensor, 0, 3)
    sf_swizzled_tensor = cute.group_modes(sf_swizzled_tensor, 1, 3)
    for i in cutlass.range(cute.size(sf_unswizzled_tensor)):
        mkl_coord = sf_unswizzled_tensor.layout.get_hier_coord(i)
        sf_unswizzled_tensor[mkl_coord] = sf_swizzled_tensor[mkl_coord]


def create_mask(group_m_list, mma_tiler_m, permuted_m=None):
    """Create group metadata for contiguous grouped GEMM with gather."""
    valid_m = 0
    aligned_group_m_list = []
    tile_idx_to_expert_idx = []
    tile_idx_to_mn_limit = []

    for i, group_m in enumerate(group_m_list):
        aligned_group_m = ((group_m + mma_tiler_m - 1) // mma_tiler_m) * mma_tiler_m
        aligned_group_m_list.append(aligned_group_m)

        num_tiles_in_group = aligned_group_m // mma_tiler_m
        tile_idx_to_expert_idx.extend([i] * num_tiles_in_group)
        for tile_idx_in_group in range(num_tiles_in_group):
            tile_idx_to_mn_limit.append(
                valid_m + min(tile_idx_in_group * mma_tiler_m + mma_tiler_m, group_m)
            )
        valid_m += aligned_group_m

    num_non_exiting_tiles = len(tile_idx_to_expert_idx)

    if permuted_m is not None:
        if permuted_m < valid_m:
            raise ValueError(f"permuted_m ({permuted_m}) must be >= valid_m ({valid_m}).")
        if (permuted_m - valid_m) % mma_tiler_m != 0:
            raise ValueError(
                f"permuted_m ({permuted_m}) must be aligned to tile M "
                f"({mma_tiler_m}) after valid_m ({valid_m})."
            )
        if permuted_m > valid_m:
            num_padding_tiles = (permuted_m - valid_m) // mma_tiler_m
            tile_idx_to_expert_idx.extend([int(-2e9)] * num_padding_tiles)
            tile_idx_to_mn_limit.extend([int(-2e9)] * num_padding_tiles)

    tile_idx_to_expert_idx = torch.tensor(tile_idx_to_expert_idx, device="cuda", dtype=torch.int32)
    num_non_exiting_tiles_tensor = torch.tensor(
        [num_non_exiting_tiles], device="cuda", dtype=torch.int32
    )
    tile_idx_to_mn_limit_tensor = torch.tensor(
        tile_idx_to_mn_limit, device="cuda", dtype=torch.int32
    )

    return (
        valid_m,
        aligned_group_m_list,
        tile_idx_to_expert_idx,
        num_non_exiting_tiles_tensor,
        tile_idx_to_mn_limit_tensor,
    )


def create_scale_factor_tensor(num_groups, mn, k, sf_vec_size, dtype):
    def ceil_div(a, b):
        return (a + b - 1) // b

    sf_k = ceil_div(k, sf_vec_size)
    ref_shape = (num_groups, mn, sf_k)

    atom_m = (32, 4)
    atom_k = 4
    mma_shape = (
        num_groups,
        ceil_div(mn, atom_m[0] * atom_m[1]),
        ceil_div(sf_k, atom_k),
        atom_m[0],
        atom_m[1],
        atom_k,
    )

    ref_permute_order = (1, 2, 0)
    mma_permute_order = (3, 4, 1, 5, 2, 0)

    ref_f32_torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
        ref_shape,
        torch.float32,
        permute_order=ref_permute_order,
        init_type=cutlass_torch.TensorInitType.RANDOM,
        init_config=cutlass_torch.RandomInitConfig(min_val=1, max_val=3),
    )

    cute_f32_torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
        mma_shape,
        torch.float32,
        permute_order=mma_permute_order,
        init_type=cutlass_torch.TensorInitType.RANDOM,
        init_config=cutlass_torch.RandomInitConfig(min_val=0, max_val=1),
    )

    cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
        from_dlpack(ref_f32_torch_tensor_cpu),
        from_dlpack(cute_f32_torch_tensor_cpu),
    )

    cute_f32_torch_tensor = cute_f32_torch_tensor_cpu.cuda()

    ref_f32_torch_tensor_cpu = (
        ref_f32_torch_tensor_cpu.permute(2, 0, 1)
        .unsqueeze(-1)
        .expand(num_groups, mn, sf_k, sf_vec_size)
        .reshape(num_groups, mn, sf_k * sf_vec_size)
        .permute(*ref_permute_order)
    )
    ref_f32_torch_tensor_cpu = ref_f32_torch_tensor_cpu[:, :k, :]

    cute_tensor, cute_torch_tensor = cutlass_torch.cute_tensor_like(
        cute_f32_torch_tensor_cpu,
        dtype,
        is_dynamic_layout=True,
        assumed_align=16,
    )

    cute_tensor = cutlass_torch.convert_cute_tensor(
        cute_f32_torch_tensor,
        cute_tensor,
        dtype,
        is_dynamic_layout=True,
    )
    return ref_f32_torch_tensor_cpu, cute_tensor, cute_torch_tensor


def create_scale_factor_tensor_unswizzled(num_groups, mn, k, sf_vec_size, dtype):
    def ceil_div(a, b):
        return (a + b - 1) // b

    sf_k = ceil_div(k, sf_vec_size)
    sf_ref = cutlass_torch.matrix(
        num_groups,
        mn,
        sf_k,
        False,
        cutlass.Float32,
        init_type=cutlass_torch.TensorInitType.RANDOM,
        init_config=cutlass_torch.RandomInitConfig(min_val=1, max_val=3),
    )
    sf_tensor, sf_torch = cutlass_torch.cute_tensor_like(
        sf_ref, dtype, is_dynamic_layout=True, assumed_align=16
    )

    sf_ref = (
        sf_ref.permute(2, 0, 1)
        .unsqueeze(-1)
        .expand(num_groups, mn, sf_k, sf_vec_size)
        .reshape(num_groups, mn, sf_k * sf_vec_size)
        .permute(1, 2, 0)
    )
    sf_ref = sf_ref[:, :k, :]
    return sf_ref, sf_tensor, sf_torch


def create_sf_layout_tensor(num_groups, mn, nk, sf_vec_size):
    def ceil_div(a, b):
        return (a + b - 1) // b

    sf_k = ceil_div(nk, sf_vec_size)

    atom_m = (32, 4)
    atom_k = 4
    mma_shape = (
        num_groups,
        ceil_div(mn, atom_m[0] * atom_m[1]),
        ceil_div(sf_k, atom_k),
        atom_m[0],
        atom_m[1],
        atom_k,
    )

    mma_permute_order = (3, 4, 1, 5, 2, 0)

    cute_f32_torch_tensor = cutlass_torch.create_and_permute_torch_tensor(
        mma_shape,
        torch.float32,
        permute_order=mma_permute_order,
        init_type=cutlass_torch.TensorInitType.RANDOM,
        init_config=cutlass_torch.RandomInitConfig(min_val=0, max_val=1),
    )
    return cute_f32_torch_tensor, sf_k


def create_token_id_mapping_tensor(group_m_list, mma_tiler_m, max_token_id, permuted_m=None):
    """Create token_id_mapping tensor for gather with random token IDs."""
    valid_m = 0
    for group_m in group_m_list:
        valid_m += ((group_m + mma_tiler_m - 1) // mma_tiler_m) * mma_tiler_m

    tensor_m = permuted_m if permuted_m is not None else valid_m
    base_data = torch.full((tensor_m,), -1, dtype=torch.int32)

    accumulated_m = 0
    for group_m in group_m_list:
        start_idx = accumulated_m
        rounded_group_m = ((group_m + mma_tiler_m - 1) // mma_tiler_m) * mma_tiler_m
        random_token_ids = torch.randint(0, max_token_id, (group_m,), dtype=torch.int32)
        base_data[start_idx : start_idx + group_m] = random_token_ids
        accumulated_m += rounded_group_m

    token_id_mapping_ref = base_data.clone()
    token_id_mapping_tensor, token_id_mapping_torch = cutlass_torch.cute_tensor_like(
        token_id_mapping_ref, cutlass.Int32, is_dynamic_layout=True, assumed_align=4
    )
    return token_id_mapping_ref, token_id_mapping_tensor, token_id_mapping_torch


def create_tensors(
    num_groups,
    group_m_list,
    n,
    k,
    a_major,
    b_major,
    cd_major,
    a_dtype,
    b_dtype,
    c_dtype,
    sf_dtype,
    sf_vec_size,
    mma_tiler_m,
    permuted_m=None,
):
    """Create tensors for grouped blockscaled GEMM with gather and SwiGLU fusion."""
    torch.manual_seed(1111)

    alpha_torch_cpu = torch.randn((num_groups,), dtype=torch.float32)

    (
        valid_m,
        aligned_group_m_list,
        _tile_idx_to_expert_idx,
        _num_non_exiting_tiles,
        _tile_idx_to_mn_limit,
    ) = create_mask(group_m_list, mma_tiler_m, permuted_m)

    max_m = max(group_m_list)
    tensor_m = permuted_m if permuted_m is not None else valid_m

    a_torch_cpu = cutlass_torch.matrix(1, max_m, k, a_major == "m", cutlass.Float32)
    b_torch_cpu = cutlass_torch.matrix(num_groups, n, k, b_major == "n", cutlass.Float32)
    c_torch_cpu = cutlass_torch.matrix(1, tensor_m, n // 2, cd_major == "m", cutlass.Float32)

    a_tensor, a_torch_gpu = cutlass_torch.cute_tensor_like(
        a_torch_cpu, a_dtype, is_dynamic_layout=True, assumed_align=16
    )
    b_tensor, b_torch_gpu = cutlass_torch.cute_tensor_like(
        b_torch_cpu, b_dtype, is_dynamic_layout=True, assumed_align=16
    )
    c_tensor, c_torch_gpu = cutlass_torch.cute_tensor_like(
        c_torch_cpu, c_dtype, is_dynamic_layout=True, assumed_align=16
    )

    a_tensor.mark_compact_shape_dynamic(
        mode=1 if a_major == "k" else 0,
        stride_order=(2, 0, 1) if a_major == "k" else (2, 1, 0),
        divisibility=32 if a_dtype == cutlass.Float4E2M1FN else 16,
    )
    b_tensor.mark_compact_shape_dynamic(
        mode=1 if b_major == "k" else 0,
        stride_order=(2, 0, 1) if b_major == "k" else (2, 1, 0),
        divisibility=32 if b_dtype == cutlass.Float4E2M1FN else 16,
    )
    c_tensor.mark_compact_shape_dynamic(
        mode=1 if cd_major == "n" else 0,
        stride_order=(2, 0, 1) if cd_major == "n" else (2, 1, 0),
        divisibility=32 if c_dtype == cutlass.Float4E2M1FN else 16,
    )

    sfa_torch_cpu, sfa_tensor, sfa_torch_gpu = create_scale_factor_tensor_unswizzled(
        1, max_m, k, sf_vec_size, sf_dtype
    )
    sfb_torch_cpu, sfb_tensor, sfb_torch_gpu = create_scale_factor_tensor(
        num_groups, n, k, sf_vec_size, sf_dtype
    )

    token_id_mapping_cpu, token_id_mapping, token_id_mapping_torch = create_token_id_mapping_tensor(
        group_m_list, mma_tiler_m, max_token_id=max_m, permuted_m=permuted_m
    )

    tile_idx_to_expert_idx = from_dlpack(_tile_idx_to_expert_idx).mark_layout_dynamic()
    tile_idx_to_mn_limit = from_dlpack(_tile_idx_to_mn_limit).mark_layout_dynamic()
    num_non_exiting_tiles = from_dlpack(_num_non_exiting_tiles).mark_layout_dynamic()
    alpha = from_dlpack(alpha_torch_cpu.cuda()).mark_layout_dynamic()

    sfc_torch_cpu = None
    sfc_tensor = None
    sfc_torch_gpu = None
    norm_const_torch_cpu = None
    norm_const_tensor = None
    norm_const_torch_gpu = None
    n_out = n // 2
    if c_dtype == cutlass.Float4E2M1FN:
        sfc_torch_cpu, sfc_tensor, sfc_torch_gpu = create_scale_factor_tensor(
            1, tensor_m, n_out, sf_vec_size, sf_dtype
        )
        norm_const_torch_gpu = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        norm_const_tensor = from_dlpack(norm_const_torch_gpu).mark_layout_dynamic()
        norm_const_torch_cpu = norm_const_torch_gpu.cpu()

    return (
        a_tensor,
        b_tensor,
        c_tensor,
        sfa_tensor,
        sfb_tensor,
        sfc_tensor,
        norm_const_tensor,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        token_id_mapping,
        num_non_exiting_tiles,
        alpha,
        a_torch_cpu,
        b_torch_cpu,
        c_torch_cpu,
        sfa_torch_cpu,
        sfb_torch_cpu,
        sfc_torch_cpu,
        norm_const_torch_cpu,
        alpha_torch_cpu,
        a_torch_gpu,
        b_torch_gpu,
        c_torch_gpu,
        sfa_torch_gpu,
        sfb_torch_gpu,
        sfc_torch_gpu,
        norm_const_torch_gpu,
        aligned_group_m_list,
        valid_m,
        token_id_mapping_cpu,
    )


def run(
    nkl: Tuple[int, int, int],
    group_m_list: Tuple[int, ...],
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    a_major: str,
    b_major: str,
    c_major: str,
    mma_inst_shape: Tuple[int, int, int],
    mma_tiler: Tuple[int, int, int],
    cluster_shape_mn: Tuple[int, int],
    tolerance: float,
    warmup_iterations: int = 0,
    iterations: int = 1,
    skip_ref_check: bool = False,
    use_cold_l2: bool = False,
    permuted_m: Optional[int] = None,
    raster_along_m: bool = False,
    use_cupti: bool = False,
    a_path: str = "cpasync",
    use_pdl: bool = True,
):
    """Run the Rubin blockscaled contiguous gather grouped GEMM SwiGLU kernel."""
    mma_tiler_m = mma_tiler[0]

    print(
        "Running Rubin Persistent Contiguous Grouped BlockScaled GEMM with "
        "Gather and SwiGLU Fusion:"
    )
    print(f"nkl: {nkl}")
    print(f"group_m_list: {group_m_list}")
    print(
        f"A dtype: {a_dtype}, B dtype: {b_dtype}, C dtype: {c_dtype}, "
        f"SF dtype: {sf_dtype}, SF Vec size: {sf_vec_size}"
    )
    if permuted_m is not None:
        print(f"Padded M (CUDA graph support): {permuted_m}")
    print(f"Matrix majors - A: {a_major}, B: {b_major}, C: {c_major}")
    print(f"MMA Inst Shape: {mma_inst_shape}, MMA Tiler: {mma_tiler}")
    print(f"Cluster Shape: {cluster_shape_mn}")
    print(f"Raster along M: {raster_along_m}")
    print(f"A path: {a_path}")
    print(f"Use PDL: {use_pdl}")
    print(f"Use CUPTI: {use_cupti}")

    n, k, num_groups = nkl

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    if not Sm107BlockScaledContiguousGatherGroupedGemmActFusionKernel.can_implement(
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        sf_dtype=sf_dtype,
        sf_vec_size=sf_vec_size,
        c_dtype=c_dtype,
        mma_inst_shape=mma_inst_shape,
        mma_tiler=mma_tiler,
        cluster_shape_mn=cluster_shape_mn,
        m=mma_tiler_m,
        n=n,
        k=k,
        l=num_groups,
        a_major=a_major,
        b_major=b_major,
        c_major=c_major,
    ):
        raise TypeError(
            f"Unsupported testcase a_dtype={a_dtype}, b_dtype={b_dtype}, "
            f"sf_dtype={sf_dtype}, sf_vec_size={sf_vec_size}, c_dtype={c_dtype}, "
            f"mma_inst_shape={mma_inst_shape}, mma_tiler={mma_tiler}, "
            f"cluster_shape_mn={cluster_shape_mn}"
        )

    (
        a_tensor,
        b_tensor,
        c_tensor,
        sfa_tensor,
        sfb_tensor,
        sfc_tensor,
        norm_const_tensor,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        token_id_mapping,
        num_non_exiting_tiles,
        alpha,
        a_torch_cpu,
        b_torch_cpu,
        c_torch_cpu,
        sfa_torch_cpu,
        sfb_torch_cpu,
        sfc_torch_cpu,
        norm_const_torch_cpu,
        alpha_torch_cpu,
        a_torch_gpu,
        b_torch_gpu,
        c_torch_gpu,
        sfa_torch_gpu,
        sfb_torch_gpu,
        sfc_torch_gpu,
        norm_const_torch_gpu,
        aligned_group_m_list,
        valid_m,
        token_id_mapping_cpu,
    ) = create_tensors(
        num_groups,
        group_m_list,
        n,
        k,
        a_major,
        b_major,
        c_major,
        a_dtype,
        b_dtype,
        c_dtype,
        sf_dtype,
        sf_vec_size,
        mma_tiler_m,
        permuted_m,
    )

    gemm = Sm107BlockScaledContiguousGatherGroupedGemmActFusionKernel(
        sf_vec_size,
        mma_inst_shape,
        mma_tiler,
        cluster_shape_mn,
        True,
        topk=1,
        raster_along_m=raster_along_m,
        a_path=a_path,
        use_pdl=use_pdl,
    )
    hardware_info = cutlass.utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )

    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)
    full_c_shape = None

    compiled_gemm = cute.compile(
        gemm,
        a_tensor,
        b_tensor,
        c_tensor,
        sfa_tensor,
        sfb_tensor,
        sfc_tensor,
        full_c_shape,
        norm_const_tensor,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        token_id_mapping,
        num_non_exiting_tiles,
        alpha,
        max_active_clusters,
        current_stream,
    )

    compiled_gemm(
        a_tensor,
        b_tensor,
        c_tensor,
        sfa_tensor,
        sfb_tensor,
        sfc_tensor,
        full_c_shape,
        norm_const_tensor,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        token_id_mapping,
        num_non_exiting_tiles,
        alpha,
        current_stream,
    )

    torch.cuda.synchronize()

    if not skip_ref_check:
        print("Verifying results...")
        interleave_granularity = 64
        n_out = n // 2

        gemm_result = torch.empty((1, valid_m, n), dtype=torch.float32)
        start = 0
        a_torch_cpu_f32 = torch.einsum("mk,mk->mk", a_torch_cpu[:, :, 0], sfa_torch_cpu[:, :, 0])
        for i, group_m in enumerate(aligned_group_m_list):
            end = start + group_m
            res_a = a_torch_cpu_f32[token_id_mapping_cpu[start:end]]
            res_b = torch.einsum("nk,nk->nk", b_torch_cpu[:, :, i], sfb_torch_cpu[:, :, i])
            gemm_result[0, start:end, :] = (
                torch.einsum("mk,nk->mn", res_a, res_b) * alpha_torch_cpu[i]
            )
            start = end

        assert n % (2 * interleave_granularity) == 0
        ref = torch.empty((1, valid_m, n_out), dtype=torch.float32)
        for n_block in range(0, n, 2 * interleave_granularity):
            up_result = gemm_result[0, :, n_block : n_block + interleave_granularity]
            gate_result = gemm_result[
                0,
                :,
                n_block + interleave_granularity : n_block + 2 * interleave_granularity,
            ]
            silu_gate = gate_result * torch.sigmoid(gate_result)
            output_block = up_result * silu_gate
            out_start = n_block // 2
            out_end = out_start + interleave_granularity
            ref[0, :, out_start:out_end] = output_block

        ref = ref.permute((1, 2, 0))

        res = c_torch_cpu.cuda()
        cute.testing.convert(
            c_tensor,
            from_dlpack(res, assumed_align=16).mark_layout_dynamic(
                leading_dim=(1 if c_major == "n" else 0)
            ),
        )

        res = res[:valid_m]
        mask = token_id_mapping_cpu[:valid_m] >= 0
        res = res.cpu()[mask]
        ref = ref[mask]

        print(f"valid_m: {valid_m}, ref.shape: {ref.shape}, res.shape: {res.shape}")

        if c_dtype in (cutlass.Float32, cutlass.Float16, cutlass.BFloat16):
            torch.testing.assert_close(res.cpu(), ref.cpu(), atol=tolerance, rtol=1e-02)
        elif c_dtype in (cutlass.Float8E5M2, cutlass.Float8E4M3FN):
            ref_f8_ = torch.empty(*(1, valid_m, n_out), dtype=torch.uint8, device="cuda").permute(
                1, 2, 0
            )
            ref_f8 = from_dlpack(ref_f8_, assumed_align=16).mark_layout_dynamic(leading_dim=1)
            ref_f8.element_type = c_dtype
            ref_device = ref.cuda()
            ref_tensor = from_dlpack(ref_device, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            )
            cute.testing.convert(ref_tensor, ref_f8)
            cute.testing.convert(ref_f8, ref_tensor)
            torch.testing.assert_close(res.cpu(), ref_device.cpu(), atol=tolerance, rtol=1e-02)
        elif c_dtype is cutlass.Float4E2M1FN:

            def ceil_div(a, b):
                return (a + b - 1) // b

            def simulate_f8_quantization(tensor_f32, f8_dtype):
                shape = tensor_f32.shape
                f8_torch = torch.empty(*shape, dtype=torch.uint8, device="cuda")
                f8_tensor = from_dlpack(f8_torch, assumed_align=16).mark_layout_dynamic(
                    leading_dim=1
                )
                f8_tensor.element_type = f8_dtype
                f32_device = tensor_f32.cuda()
                f32_tensor = from_dlpack(f32_device, assumed_align=16).mark_layout_dynamic(
                    leading_dim=1
                )
                cute.testing.convert(f32_tensor, f8_tensor)
                cute.testing.convert(f8_tensor, f32_tensor)
                return f32_device.cpu()

            def simulate_nvfp4_quantization(tensor_f32):
                m_dim, n_dim, ng = tensor_f32.shape
                ref_f32_torch = cutlass_torch.matrix(ng, m_dim, n_dim, False, cutlass.Float32)
                f4_tensor, _ = cutlass_torch.cute_tensor_like(
                    ref_f32_torch,
                    cutlass.Float4E2M1FN,
                    is_dynamic_layout=True,
                    assumed_align=16,
                )
                f32_device = tensor_f32.cuda()
                f32_tensor = from_dlpack(f32_device, assumed_align=16).mark_layout_dynamic(
                    leading_dim=1
                )
                cute.testing.convert(f32_tensor, f4_tensor)
                cute.testing.convert(f4_tensor, f32_tensor)
                return f32_device.cpu()

            def compute_scale_factor(tensor_f32, sf_vec_size_local, norm_const, rcp_limits):
                m_dim, n_dim, ng = tensor_f32.shape
                sfn = ceil_div(n_dim, sf_vec_size_local)
                padded_n = sfn * sf_vec_size_local
                if padded_n > n_dim:
                    tensor_padded = torch.zeros(m_dim, padded_n, ng, dtype=tensor_f32.dtype)
                    tensor_padded[:, :n_dim, :] = tensor_f32
                else:
                    tensor_padded = tensor_f32
                tensor_reshaped = tensor_padded.view(m_dim, sfn, sf_vec_size_local, ng)
                abs_max, _ = torch.abs(tensor_reshaped).max(dim=2)
                return abs_max * norm_const * rcp_limits

            def apply_quantization_scale(tensor_f32, scale_factor, sf_vec_size_local, norm_const):
                m_dim, n_dim, ng = tensor_f32.shape
                sfn = scale_factor.shape[1]
                fp32_max = torch.tensor(3.40282346638528859812e38, dtype=torch.float32)
                scale_rcp = norm_const * scale_factor.reciprocal()
                scale_rcp = torch.where(torch.isinf(scale_rcp), fp32_max, scale_rcp)
                scale_rcp_expanded = scale_rcp.unsqueeze(2).expand(
                    m_dim, sfn, sf_vec_size_local, ng
                )
                scale_rcp_expanded = scale_rcp_expanded.reshape(m_dim, sfn * sf_vec_size_local, ng)
                scale_rcp_expanded = scale_rcp_expanded[:, :n_dim, :]
                return tensor_f32 * scale_rcp_expanded

            def unswizzle_kernel_sfc(
                sfc_tensor_local, permuted_m_local, n_out_local, sf_vec_size_local
            ):
                sfn = ceil_div(n_out_local, sf_vec_size_local)
                unswizzled_sfc = torch.empty(permuted_m_local, sfn, 1, dtype=torch.float32)
                swizzled_sfc_cpu, _ = create_sf_layout_tensor(
                    1, permuted_m_local, n_out_local, sf_vec_size_local
                )
                swizzled_sfc_tensor, swizzled_sfc_torch = cutlass_torch.cute_tensor_like(
                    swizzled_sfc_cpu,
                    cutlass.Float32,
                    is_dynamic_layout=True,
                    assumed_align=16,
                )
                cute.testing.convert(sfc_tensor_local, swizzled_sfc_tensor)
                swizzled_sfc_cpu = swizzled_sfc_torch.cpu()
                cvt_sf_M32x4xrm_K4xrk_L_to_MKL(
                    from_dlpack(swizzled_sfc_cpu),
                    from_dlpack(unswizzled_sfc),
                )
                return unswizzled_sfc

            norm_const = norm_const_torch_cpu.item()
            rcp_limits = gemm.get_dtype_rcp_limits(c_dtype)

            ref_sfc_f32 = compute_scale_factor(ref, sf_vec_size, norm_const, rcp_limits)
            ref_sfc_f32 = simulate_f8_quantization(ref_sfc_f32, sf_dtype)

            permuted_m_val = token_id_mapping_cpu.shape[0]
            kernel_sfc = unswizzle_kernel_sfc(sfc_tensor, permuted_m_val, n_out, sf_vec_size)
            torch.testing.assert_close(
                ref_sfc_f32, kernel_sfc[:valid_m][mask], atol=tolerance, rtol=1e-02
            )
            print("SFC Tensor comparison passed!")

            ref_scaled = apply_quantization_scale(ref, ref_sfc_f32, sf_vec_size, norm_const)
            ref_quantized = simulate_nvfp4_quantization(ref_scaled)

            print("Verifying C Tensor...")
            res_cpu = res.cpu()
            diff = torch.abs(res_cpu - ref_quantized)
            within_tolerance = (diff <= tolerance) | (diff <= torch.abs(ref_quantized) * 1e-02)
            pass_rate = within_tolerance.float().mean().item()
            print(f"C Tensor pass rate: {pass_rate * 100:.2f}% (threshold: 95%)")
            assert pass_rate >= 0.95, (
                f"Only {pass_rate * 100:.2f}% elements within tolerance, expected >= 95%"
            )

    def generate_tensors():
        (
            a_tensor,
            b_tensor,
            c_tensor,
            sfa_tensor,
            sfb_tensor,
            sfc_tensor,
            norm_const_tensor,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            token_id_mapping,
            num_non_exiting_tiles,
            alpha,
            *_,
        ) = create_tensors(
            num_groups,
            group_m_list,
            n,
            k,
            a_major,
            b_major,
            c_major,
            a_dtype,
            b_dtype,
            c_dtype,
            sf_dtype,
            sf_vec_size,
            mma_tiler_m,
            permuted_m,
        )
        return cute.testing.JitArguments(
            a_tensor,
            b_tensor,
            c_tensor,
            sfa_tensor,
            sfb_tensor,
            sfc_tensor,
            full_c_shape,
            norm_const_tensor,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            token_id_mapping,
            num_non_exiting_tiles,
            alpha,
            current_stream,
        )

    workspace_count = 1
    if use_cold_l2:
        tensor_m = permuted_m if permuted_m is not None else valid_m
        one_workspace_bytes = (
            a_torch_gpu.numel() * a_torch_gpu.element_size()
            + b_torch_gpu.numel() * b_torch_gpu.element_size()
            + c_torch_gpu.numel() * c_torch_gpu.element_size()
            + sfa_torch_gpu.numel() * sfa_torch_gpu.element_size()
            + sfb_torch_gpu.numel() * sfb_torch_gpu.element_size()
            + (
                sfc_torch_gpu.numel() * sfc_torch_gpu.element_size()
                if sfc_torch_gpu is not None
                else 0
            )
            + (
                norm_const_torch_gpu.numel() * norm_const_torch_gpu.element_size()
                if norm_const_torch_gpu is not None
                else 0
            )
            + (tensor_m // mma_tiler_m) * 4
            + (tensor_m // mma_tiler_m) * 4
            + tensor_m * 4
            + 1 * 4
            + alpha_torch_cpu.numel() * alpha_torch_cpu.element_size()
        )
        workspace_count = cute.testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations
        )

    exec_time = cute.testing.benchmark(
        compiled_gemm,
        workspace_generator=generate_tensors,
        workspace_count=workspace_count,
        stream=current_stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
        use_cupti=use_cupti,
    )

    return exec_time


def parse_comma_separated_ints(s: str) -> Tuple[int, ...]:
    try:
        return tuple(int(x.strip()) for x in s.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Invalid format. Expected comma-separated integers."
        ) from exc


def read_benchmark_file(
    filepath: str,
) -> Tuple[Tuple[int, int, int], Tuple[int, ...]]:
    """Read benchmark file and return nkl plus per-group M values."""
    problems = []
    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                dims = parts[1].split("x")
                if len(dims) == 3:
                    m, n, k = int(dims[0]), int(dims[1]), int(dims[2])
                    problems.append((m, n, k))

        if not problems:
            raise ValueError(f"No valid problems found in benchmark file: {filepath}")

        _, n, k = problems[0]
        num_groups = len(problems)
        m_values = tuple(m for m, _, _ in problems)

        print(f"Loaded {num_groups} problems from benchmark file")
        print(f"Using N={n}, K={k}, L={num_groups}")
        print(f"M values per group: {m_values}")

        return ((n, k, num_groups), m_values)

    except FileNotFoundError as exc:
        raise argparse.ArgumentTypeError(f"Benchmark file not found: {filepath}") from exc
    except (OSError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"Error reading benchmark file: {exc}") from exc


def parse_benchmark_arg(
    arg: str,
) -> Tuple[Tuple[int, int, int], Tuple[int, ...]]:
    """Parse benchmark argument string."""
    match_list = re.match(r"\[([\d,\s]+)\]\s*x\s*(\d+)\s*x\s*(\d+)", arg)
    if match_list:
        m_str = match_list.group(1)
        n = int(match_list.group(2))
        k = int(match_list.group(3))
        try:
            m_values = tuple(int(x.strip()) for x in m_str.split(","))
            num_groups = len(m_values)
            return ((n, k, num_groups), m_values)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid integer list in benchmark argument: {arg}"
            ) from exc

    parts = arg.split("x")
    if len(parts) == 4:
        try:
            m, n, k, num_groups = [int(x.strip()) for x in parts]
            m_values = tuple([m] * num_groups)
            return ((n, k, num_groups), m_values)
        except ValueError:
            pass

    raise argparse.ArgumentTypeError(f"Invalid benchmark argument format. Got: {arg}")


def main():
    """Main entry point for running the Rubin blockscaled SwiGLU fusion kernel."""
    parser = argparse.ArgumentParser(
        description=("Rubin BlockScaled Contiguous Gather Grouped GEMM with SwiGLU Fusion.")
    )

    parser.add_argument("--nkl", type=parse_comma_separated_ints, default=(256, 512, 1))
    parser.add_argument("--fixed_m", type=int, default=None)
    parser.add_argument("--custom_mask", type=parse_comma_separated_ints, default=None)
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument("--permuted_m", type=int, default=None)
    parser.add_argument(
        "--mma_inst_shape", type=parse_comma_separated_ints, default=(128, 128, 128)
    )
    parser.add_argument("--mma_tiler", type=parse_comma_separated_ints, default=(128, 128, 256))
    parser.add_argument("--cluster_shape_mn", type=parse_comma_separated_ints, default=(1, 1))
    parser.add_argument("--a_dtype", type=cutlass.dtype, default=cutlass.Float4E2M1FN)
    parser.add_argument("--b_dtype", type=cutlass.dtype, default=cutlass.Float4E2M1FN)
    parser.add_argument("--c_dtype", type=cutlass.dtype, default=cutlass.BFloat16)
    parser.add_argument("--sf_dtype", type=cutlass.dtype, default=cutlass.Float8E4M3FN)
    parser.add_argument("--sf_vec_size", type=int, default=16)
    parser.add_argument("--a_major", choices=["k"], type=str, default="k")
    parser.add_argument("--b_major", choices=["k"], type=str, default="k")
    parser.add_argument("--c_major", choices=["n", "m"], type=str, default="n")
    parser.add_argument("--tolerance", type=float, default=1e-01)
    parser.add_argument("--warmup_iterations", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--skip_ref_check", action="store_true")
    parser.add_argument("--use_cold_l2", action="store_true", default=False)
    parser.add_argument("--raster_along_m", action="store_true", default=False)
    parser.add_argument("--use_cupti", action="store_true", default=False)
    parser.add_argument(
        "--a_path",
        choices=["cpasync", "tma"],
        default="cpasync",
        help=(
            "A load path: 'cpasync' = per-thread cp.async.cg.16B; "
            "'tma' = TMA gather4. SFA path is always cpasync.128."
        ),
    )
    parser.add_argument(
        "--use_pdl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable/disable PDL (Programmatic Dependent Launch). "
            "Default: on. Use --no-use_pdl to disable."
        ),
    )
    args = parser.parse_args()

    if args.benchmark:
        if os.path.isfile(args.benchmark):
            nkl, group_m_list = read_benchmark_file(args.benchmark)
        else:
            nkl, group_m_list = parse_benchmark_arg(args.benchmark)
    else:
        if len(args.nkl) != 3:
            parser.error("--nkl must contain exactly 3 values")
        n, k, num_groups = args.nkl
        nkl = (n, k, num_groups)

        if args.custom_mask is not None:
            group_m_list = args.custom_mask
            if len(group_m_list) != num_groups:
                parser.error(f"--custom_mask must have exactly {num_groups} values")
        elif args.fixed_m is not None:
            group_m_list = tuple([args.fixed_m] * num_groups)
        else:
            group_m_list = tuple([128] * num_groups)

    if len(args.mma_inst_shape) != 3:
        parser.error("--mma_inst_shape must contain exactly 3 values")
    if len(args.mma_tiler) != 3:
        parser.error("--mma_tiler must contain exactly 3 values")
    if len(args.cluster_shape_mn) != 2:
        parser.error("--cluster_shape_mn must contain exactly 2 values")

    exec_time = run(
        nkl=nkl,
        group_m_list=group_m_list,
        a_dtype=args.a_dtype,
        b_dtype=args.b_dtype,
        c_dtype=args.c_dtype,
        sf_dtype=args.sf_dtype,
        sf_vec_size=args.sf_vec_size,
        a_major=args.a_major,
        b_major=args.b_major,
        c_major=args.c_major,
        mma_inst_shape=args.mma_inst_shape,
        mma_tiler=args.mma_tiler,
        cluster_shape_mn=args.cluster_shape_mn,
        tolerance=args.tolerance,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
        skip_ref_check=args.skip_ref_check,
        use_cold_l2=args.use_cold_l2,
        permuted_m=args.permuted_m,
        raster_along_m=args.raster_along_m,
        use_cupti=args.use_cupti,
        a_path=args.a_path,
        use_pdl=args.use_pdl,
    )
    print(f"Execution time: {exec_time:.2f} us")
    print("PASS")


if __name__ == "__main__":
    main()
