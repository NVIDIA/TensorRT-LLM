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

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# This file is derived from dense_gemm_persistent.py (BF16-in, BF16-out batched
# persistent dense GEMM) by grafting the NVFP4-emitting epilogue from
# dense_blockscaled_gemm_swiglu_fusion.py (minus the SwiGLU activation).
#
# Purpose (DeepSeek-V3.2 MLA decode, v_b_proj V-up absorption): fuse the
# standalone BF16->NVFP4 quantize (quantize_with_block_size<0>) into the BF16
# batched GEMM epilogue so the V-up bmm directly emits packed E2M1 + E4M3
# block scale factors in the TensorRT-LLM swizzled SF layout.
#
# Inputs : A [L, M, K] bf16, B [L, N, K] bf16 (K-major), scalar sf_scale (=
#          o_proj.input_scale, a per-tensor float32 device scalar).
# Outputs: packed E2M1 C [L, M, N//2] (2 values/byte) and ONE monolithic
#          swizzled E4M3 SFC over the logical [M, L*N] activation.
#
# v_head_dim = N = 128 = 8 * 16 (sf_vec_size). Block-scale boundaries align
# exactly with head (N) boundaries, AND each head spans N//16 = 8 SF
# col-vectors which is a multiple of the swizzle K-tile width (4), so head
# boundaries fall on K-tile boundaries: heads remain independent with no
# cross-head SF reduction.
#
# The downstream o_proj NVFP4 GEMM treats the V-up output as a SINGLE
# [M, L*N] matrix (batchIdx=None) and expects ONE swizzled SF over [M, L*N] --
# exactly what fp4_quantize([M, L*N]) produces. A per-head SF stacked along L
# is byte-identical to this ONLY when M <= 128 (numMTiles == 1); for M > 128
# the monolithic layout interleaves M-tiles across ALL heads' K-tiles while a
# per-head layout keeps each head contiguous (see get_sf_out_offset_128x4 in
# cpp/tensorrt_llm/kernels/quantization.cuh). This kernel therefore emits the
# monolithic SF so decode M > 128 is correct.

from typing import Literal, Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass._mlir.dialects import math
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from .custom_pipeline import PipelineTmaUmma, PipelineUmmaAsync
from .utils import (
    TRTLLM_ENABLE_PDL,
    fmin,
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
    is_power_of_2,
)


def _compute_stages(
    tiled_mma: cute.TiledMma,
    mma_tiler_mnk: Tuple[int, int, int],
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    smem_capacity: int,
    occupancy: int,
    c_smem_layout: Union[cute.Layout, None],
) -> Tuple[int, int, int]:
    """Computes the number of stages for A/B/C operands based on heuristics.

    Identical to the BF16 base kernel: the FP4 epilogue does not add any
    A/B mainloop SMEM, only changes the C SMEM element type (FP4 vs BF16).
    """
    num_acc_stage = 2
    num_c_stage = 2

    a_smem_layout_stage_one = utils.sm100.make_smem_layout_a(tiled_mma, mma_tiler_mnk, a_dtype, 1)
    b_smem_layout_staged_one = utils.sm100.make_smem_layout_b(tiled_mma, mma_tiler_mnk, b_dtype, 1)

    ab_bytes_per_stage = cute.size_in_bytes(a_dtype, a_smem_layout_stage_one) + cute.size_in_bytes(
        b_dtype, b_smem_layout_staged_one
    )
    mbar_helpers_bytes = 1024

    c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout)
    c_bytes = c_bytes_per_stage * num_c_stage

    num_ab_stage = (
        smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
    ) // ab_bytes_per_stage

    num_c_stage += (
        smem_capacity
        - occupancy * ab_bytes_per_stage * num_ab_stage
        - occupancy * (mbar_helpers_bytes + c_bytes)
    ) // (occupancy * c_bytes_per_stage)
    return num_acc_stage, num_ab_stage, num_c_stage


class PersistentDenseGemmNVFP4OutKernel:
    """Persistent batched dense GEMM (C = A x B) with a fused NVFP4 output
    epilogue for Blackwell SM100 using CuTe DSL.

    A and B are BF16 (FP32 accumulator). The epilogue quantizes the FP32
    accumulator to NVFP4: packed E2M1 values (2/byte) plus E4M3 block scale
    factors (one per ``sf_vec_size``-element block) written in TensorRT-LLM's
    swizzled SF layout. A single per-tensor scale ``sf_scale`` (= the
    downstream GEMM's ``input_scale``) parameterizes the quantization,
    matching ``quantize_with_block_size<BlockScaleQuantizationType=0>`` /
    ``cvt_warp_fp16_to_fp4``.

    Notes:
        - A and B must both be BF16 (or FP16); accumulator is FP32.
        - Output C dtype is Float4E2M1FN; SFC dtype is Float8E4M3FN.
        - sf_vec_size = 16 (NVFP4).
        - MMA tiler M: 64/128 (1CTA) or 128/256 (2CTA).
        - MMA tiler N: multiple of 32; for the V-up bmm N = v_head_dim = 128.
        - N must be a multiple of sf_vec_size so SF blocks align with N tiles.
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        n: int,
        sf_vec_size: int = 16,
        sf_dtype: Type[cutlass.Numeric] = cutlass.Float8E4M3FN,
        swizzle_size: int = 1,
        raster_along: Literal["m", "n"] = "m",
        vectorized_f32: bool = True,
    ):
        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        self.swizzle_size = swizzle_size
        self.raster_along = raster_along
        self.mma_tiler_mn = mma_tiler_mn
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.sf_vec_size = sf_vec_size
        self.sf_dtype = sf_dtype
        self.vectorized_f32 = vectorized_f32
        self.arch = "sm_100"
        # STATIC per-head N (= v_head_dim). Threaded down from the runner as a
        # Python int (weight.shape[1]); MUST be a true constexpr int so the
        # head-fold SF offset (head * (N // cta_tile_N)) can be read inside the
        # warp-specialized epilogue region with no `cute.*` op. Do NOT source
        # this from c.shape -- c is a __call__ param whose shape is a DYNAMIC
        # SSA int_tuple, and multiplying it in-region emits a `cute.tuple_mul`
        # consuming an outside-region SSA value -> region-isolation failure.
        self.n_static = int(n)

        self.cta_group = tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE

        self.occupancy = 1
        self.epilogue_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * len(
            (self.mma_warp_id, self.tma_warp_id, *self.epilogue_warp_id)
        )
        self.epilog_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2
        self.tmem_dealloc_sync_bar_id = 3

    def _create_tiled_mma(self):
        return utils.sm100.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs."""
        tiled_mma = self._create_tiled_mma()

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        # Number of CTA n-tiles per head (== 1 when cta_tile_N == N): used to
        # offset each head's epilogue columns into the monolithic SF tensor.
        #
        # Computed from STATIC Python ints only:
        #   - self.n_static is the per-head N (= v_head_dim), threaded from the
        #     runner as weight.shape[1] (a Python int), NOT from c.shape.
        #   - self.cta_tile_shape_mnk[1] == self.mma_tiler[1] == mma_tiler_mn[1]
        #     is a static int built from the constructor mma_tiler.
        # Both operands are constexpr ints, so floor-division yields a plain
        # Python int with NO MLIR op emitted. The device kernel then READS this
        # as a constexpr int attr inside the warp-specialized epilogue `if`
        # region (legal -- same as self.epi_tile_cnt_m, self.sf_vec_size).
        #
        # Deriving it from c_n // cta_tile_N (c_n = c.shape[1], a DYNAMIC SSA
        # int_tuple) instead would make self.sf_n_tiles_per_head a dynamic SSA
        # value defined in __call__; the in-region `head * sf_n_tiles_per_head`
        # at the SF-offset site would then emit a `cute.tuple_mul` op consuming
        # that outside-region value -> region-isolation failure.
        self.sf_n_tiles_per_head = self.n_static // self.cta_tile_shape_mnk[1]

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Epilogue tile: derive from the BF16 base heuristic, then clamp the N
        # extent to a multiple of sf_vec_size so each SF block lives entirely
        # within one epilogue subtile (required for the per-block amax/store).
        self.epi_tile = utils.sm100.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )

        c_smem_layout = utils.sm100.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, 1
        )

        self.smem_capacity = utils.get_smem_capacity_in_bytes()

        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = _compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
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

        self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(
            tiled_mma, self.mma_tiler, self.num_acc_stage
        )

        # Epilogue subtile counts (CTA tile / epilogue tile), per spatial mode.
        # MUST be computed HERE (host side, outside any kernel region):
        # self.cta_tile_shape_mnk[i] is a plain int and self.epi_tile[i] is a
        # STATIC layout that exists in __call__/_setup_attributes, so
        # cute.size(static_layout) returns a PLAIN PYTHON INT (no MLIR op
        # emitted) and the floor-division yields a plain int. Storing these as
        # plain-int attrs lets the device kernel READ them as constexpr ints
        # inside the warp-specialized `if` region -- reading a constexpr int attr
        # inside a region is legal (same as self.mma_warp_id, self.sf_vec_size,
        # self.num_c_stage). Computing `cute.size(self.epi_tile[i])` INSIDE the
        # region would instead emit a `cute.size` MLIR op that consumes an SSA
        # layout value defined OUTSIDE the region -> region-isolation failure.
        self.epi_tile_cnt_m = self.cta_tile_shape_mnk[0] // cute.size(self.epi_tile[0])
        self.epi_tile_cnt_n = self.cta_tile_shape_mnk[1] // cute.size(self.epi_tile[1])

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        sfc: cute.Tensor,
        sf_scale: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the GEMM with a fused NVFP4 output epilogue.

        Args:
            a: BF16 input A, logical layout (M, K, L), K-major.
            b: BF16 input B, logical layout (N, K, L), K-major.
            c: Float4E2M1FN packed output, logical layout (M, N, L). N is the
                contiguous (unit-stride) minor mode; M and L carry arbitrary
                strides supplied by the caller. For the V-up fusion the runner
                passes an M-MAJOR layout (M stride = L*N, L stride = N, in FP4
                elements) so the underlying buffer, viewed as [M, L*(N//2)]
                row-major, is the head-major monolithic activation the o_proj
                NVFP4 GEMM consumes with zero copy. The TMA store only requires
                N (the box minor mode) to be unit-stride, so any M/L strides
                are stored directly without a post-kernel transpose.
            sfc: E4M3 scale-factor output. ONE monolithic swizzled SF over the
                logical (M, L*N) activation (derived via tile_atom_to_shape_SF
                on the 2D shape (M, N*L)), matching what fp4_quantize produces
                for a [M, L*N] input -- the layout the downstream o_proj NVFP4
                GEMM expects (batchIdx=None).
            sf_scale: single-element float32 device tensor, the per-tensor
                quantization scale (= downstream input_scale).
            max_active_clusters: Maximum number of active clusters.
            stream: CUDA stream.
            epilogue_op: optional element-wise op applied to the FP32
                accumulator before quantization (default identity).
        """
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        tiled_mma = self._create_tiled_mma()
        self._setup_attributes()

        # Re-lay the SFC iterator into the swizzled scale-factor atom layout.
        #
        # MONOLITHIC SF: the downstream o_proj NVFP4 GEMM treats the V-up
        # output as a single [M, L*N] activation matrix (L = num_heads,
        # N = v_head_dim) with batchIdx=None and expects ONE swizzled SF over
        # [M, L*N] -- exactly what fp4_quantize([M, L*N]) produces. A per-batch
        # (per-head) swizzled SF stacked along L is byte-identical to this ONLY
        # when M <= 128 (numMTiles == 1); for M > 128 the monolithic layout
        # interleaves M-tiles across ALL heads' K-tiles while a per-head layout
        # keeps each head contiguous, so they diverge (see
        # get_sf_out_offset_128x4 in quantization.cuh). We therefore derive the
        # SF layout from the 2D monolithic shape (M, L*N), NOT (M, N, L).
        #
        # N = v_head_dim is a multiple of sf_vec_size (128 = 8 * 16), so each
        # head spans exactly N // sf_vec_size SF col-vectors and head
        # boundaries fall on K-tile (width-4) boundaries: heads stay
        # independent, no cross-head SF reduction. This only changes the SF
        # tensor's LOGICAL shape/indexing, not the per-block amax math.
        c_m, c_n, c_l = c.shape
        # Build the swizzled SF layout over the 2D monolithic (M, L*N) shape.
        #
        # We cannot use blockscaled_utils.tile_atom_to_shape_SF here: that
        # helper hardcodes a RANK-3 order=(2,1,3) (for the standard per-batch
        # (M, N, L) SF), which mismatches our rank-2 (M, L*N) target shape and
        # makes tile_to_shape raise "target shape and order operands have same
        # rank, but got (?,?) vs (2,1,3)". Instead we tile the SAME 128x4
        # BlockScaledBasicChunk atom directly to (M, L*N) with the rank-2
        # order (2, 1) -- byte-identical to what tile_atom_to_shape_SF would
        # produce for a degenerate L=1 batch, and to fp4_quantize([M, L*N]).
        sfc_layout = cute.tile_to_shape(
            blockscaled_utils.BlockScaledBasicChunk(self.sf_vec_size).layout,
            (c_m, c_n * c_l),
            (2, 1),
        )
        sfc = cute.make_tensor(sfc.iterator, sfc_layout)
        # NOTE: self.sf_n_tiles_per_head is computed in _setup_attributes() from
        # STATIC Python ints (self.n_static // self.cta_tile_shape_mnk[1]), NOT
        # here from c_n (= c.shape[1], a DYNAMIC SSA int_tuple). The rank-2
        # monolithic SF layout above legitimately uses dynamic c_m/c_n*c_l: that
        # is on the HOST (inside __call__, consumed by cute.make_tensor on the
        # host), never inside a device kernel region, so it is fine. Only the
        # IN-REGION head-fold offset (which multiplies sf_n_tiles_per_head) must
        # read a constexpr int -- hence the static derivation in
        # _setup_attributes.

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for A
        a_op = utils.sm100.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(cutlass.TFloat32 if a.element_type is cutlass.Float32 else None),
        )

        # Setup TMA load for B
        b_op = utils.sm100.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(cutlass.TFloat32 if b.element_type is cutlass.Float32 else None),
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        # Setup TMA store for C (packed FP4 output)
        epi_smem_layout = cute.select(self.c_smem_layout_staged, mode=[0, 1])
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c, epi_smem_layout, self.epi_tile
        )

        # Compute grid size
        self.tile_sched_params, grid = self._compute_grid(
            c,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            self.swizzle_size,
            self.raster_along,
            max_active_clusters,
        )

        # Launch the kernel
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            sfc,
            sf_scale,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            epilogue_op,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            use_pdl=TRTLLM_ENABLE_PDL,
        )
        return

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        mSFC_mnl: cute.Tensor,
        sf_scale: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
    ):
        """GPU device kernel: persistent batched GEMM with NVFP4 epilogue.

        The mainloop (TMA load A/B, UMMA accumulate) is identical to the BF16
        base kernel. The epilogue quantizes the FP32 accumulator to NVFP4.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # Per-tensor quantization scale (== cvt_warp_fp16_to_fp4 SFScaleVal).
        norm_const = sf_scale[0]

        # Prefetch tma desc
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        # Setup cta/thread coordinates
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()

        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Initialize mainloop ab_pipeline (barrier) and states
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_producer, ab_consumer = PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        ).make_participants()

        # Initialize acc_pipeline (barrier) and states
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
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # Setup smem tensor A/B
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

        # Compute multicast mask for A/B buffer full
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        # Local_tile partition global tensors
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        # Partition global tensor for TiledMMA_A/B/C
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)

        # Partition global/shared tensor for TMA load A/B
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

        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        # Cluster wait before tensor memory alloc
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # Construct the scheduler
        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
        )
        work_tile = tile_sched.initial_work_tile_info()

        # PDL: Wait for previous grid to finish
        griddepcontrol_wait()

        # ---------------- Specialized TMA load warp ----------------
        if warp_idx == self.tma_warp_id:
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
                tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

                ab_producer.reset()
                peek_ab_empty_status = ab_producer.try_acquire()

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    handle = ab_producer.acquire_and_advance(peek_ab_empty_status)

                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, handle.count)],
                        tAsA[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, handle.count)],
                        tBsB[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=b_full_mcast_mask,
                    )

                    peek_ab_empty_status = cutlass.Boolean(1)
                    if handle.count + 1 < k_tile_cnt:
                        peek_ab_empty_status = ab_producer.try_acquire()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            ab_producer.tail()

        # ---------------- Specialized MMA warp ----------------
        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            while work_tile.is_valid_tile:
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                ab_consumer.reset()
                peek_ab_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    peek_ab_full_status = ab_consumer.try_wait()

                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                # Reset ACCUMULATE for each new output tile
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                for k_tile in range(k_tile_cnt):
                    if is_leader_cta:
                        handle = ab_consumer.wait_and_advance(peek_ab_full_status)

                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_crd = (None, None, kblock_idx, handle.index)
                            cute.gemm(tiled_mma, tCtAcc, tCrA[kblock_crd], tCrB[kblock_crd], tCtAcc)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        handle.release()

                        peek_ab_full_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_full_status = ab_consumer.try_wait()

                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            acc_pipeline.producer_tail(acc_producer_state)

        # C output smem (FP4 packed)
        sC = smem.allocate_tensor(
            element_type=self.c_dtype,
            layout=c_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=c_smem_layout_staged.inner,
        )

        # ---------------- Specialized epilogue warps (NVFP4 quant) ----------------
        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)

            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            # TMEM -> RMEM copy setup
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

            # RMEM -> SMEM copy setup
            copy_atom_r2s = utils.sm100.get_smem_store_op(
                self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
            )
            tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
            thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
            tRS_sC = thr_copy_r2s.partition_D(sC)
            tTR_rC = cute.make_fragment(tTR_rAcc.shape, self.c_dtype)
            tRS_rC = tiled_copy_r2s.retile(tTR_rC)

            # SMEM -> GMEM TMA store setup (FP4 C)
            sC_for_tma = cute.group_modes(sC, 0, 2)
            gC_for_tma = cute.group_modes(gC_mnl_epi, 0, 2)
            bSG_sC, bSG_gC_partitioned = cpasync.tma_partition(
                tma_atom_c, 0, cute.make_layout(1), sC_for_tma, gC_for_tma
            )

            # SFC store setup: partition the MONOLITHIC swizzled SFC global
            # tensor (2D [M, L*N], no batch mode) with the same t2r thread
            # layout, then write E4M3 block scales per epilogue subtile.
            # epi_tile_cnt_m/n map a CTA tile to its subtiles. These are PLAIN
            # PYTHON INTS pre-computed on the HOST in _setup_attributes (see the
            # note there). We only READ the constexpr int attrs here; we do NOT
            # emit any cute op on an outside-region SSA value. Computing
            # `cute.size(self.epi_tile[i])` INSIDE this warp-specialized `if`
            # region would emit a `cute.size` MLIR op consuming the self.epi_tile
            # layout SSA value defined OUTSIDE the region -> region-isolation
            # failure ("'cute.size' op using value defined outside the region").
            epi_tile_cnt_m = self.epi_tile_cnt_m
            epi_tile_cnt_n = self.epi_tile_cnt_n
            # mSFC_mnl is the monolithic (M, L*N) swizzled SF tensor, so it
            # tiles in 2D (M, N) only -- the head (L) index is folded into the
            # N column offset below.
            # (EPI_TILE_M, EPI_TILE_N, RestM, RestN)
            gSFC_mn = cute.local_tile(mSFC_mnl, epi_tile, (None, None))
            # (T2R, T2R_M, T2R_N, RestM, RestN)
            tCgSFC_mn = thr_copy_t2r.partition_D(gSFC_mn)
            tCgSFC_mn = cute.filter_zeros(tCgSFC_mn)
            # (T2R, T2R_M, T2R_N) register SF staging
            tCrSFC = cute.make_rmem_tensor(
                tCgSFC_mn[(None, None, None, 0, 0)].layout, self.sf_dtype
            )
            tCrSFC_pvscale = cute.make_rmem_tensor_like(tCrSFC, cutlass.Float32)

            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilogue_warp_id),
                32 * len(self.epilogue_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage, producer_group=c_producer_group
            )

            # E2M1 reciprocal-of-max-abs (1/6) used in the SF recipe.
            c_rcp_limit = self.get_dtype_rcp_limits(self.c_dtype)
            fp32_max = cutlass.Float32(3.40282346638528859812e38)

            # -- Epilogue tile scheduling loop --
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

                num_tiles_executed = tile_sched.num_tiles_executed

                # Slice to per mma tile
                bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]
                # Monolithic SF: no per-batch slice. Each head's columns live
                # at a fixed offset (head * sf_n_tiles_per_head CTA-n-tiles)
                # within the single (M, L*N) swizzled tensor; that offset is
                # applied to the N subtile index below.
                acc_stage_index = acc_consumer_state.index
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_stage_index)]

                # Wait for accumulator buffer full
                acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                # Store accumulator (quantized to NVFP4) in sub-tiles
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = (num_tiles_executed - 1) * subtile_cnt

                for subtile_idx in cutlass.range(subtile_cnt):
                    # Load accumulator from TMEM to RMEM
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                    # Apply optional element-wise epilogue op on FP32 acc.
                    tCompute = cute.make_rmem_tensor(tTR_rAcc.shape, self.acc_dtype)
                    acc_pre = tTR_rAcc.load()
                    acc_pre = epilogue_op(acc_pre)
                    tCompute.store(acc_pre)

                    # ---- NVFP4 quantization of this subtile ----
                    # SFC subtile coordinate in the MONOLITHIC swizzled SF
                    # tensor. M index is unchanged (M is shared across heads).
                    # The N index folds in the head offset: head L occupies
                    # CTA-n-tiles [L*sf_n_tiles_per_head, (L+1)*...) of the
                    # logical [M, L*N] column space, so the effective CTA-n-tile
                    # coordinate is (cur_tile_coord[1] + head*sf_n_tiles_per_head).
                    # Because N = v_head_dim is a multiple of sf_vec_size and a
                    # multiple of the K-tile width (4), head boundaries fall on
                    # K-tile boundaries -- the column offset stays head-local.
                    sf_n_tile_coord = (
                        cur_tile_coord[1]
                        + mma_tile_coord_mnl[2] * self.sf_n_tiles_per_head
                    )
                    sfc_subtile_idx_mn = (
                        cur_tile_coord[0] * epi_tile_cnt_m,
                        sf_n_tile_coord * epi_tile_cnt_n + subtile_idx,
                    )
                    tCgSFC = tCgSFC_mn[(None, None, None, *sfc_subtile_idx_mn)]

                    # Group the compute fragment into sf_vec_size blocks.
                    # (sf_vec_size, num_vecs)
                    tTR_rAcc_frg = cute.logical_divide(
                        tCompute, cute.make_layout(self.sf_vec_size)
                    )
                    acc_frg = tTR_rAcc_frg.load()

                    # Per-block absolute max.
                    abs_acc_frg_ir = math.absf(acc_frg.ir_value())
                    abs_acc_frg = type(acc_frg)(abs_acc_frg_ir, acc_frg.shape, acc_frg.dtype)

                    # SFValue = sf_scale * (vecMax / 6), stored as E4M3.
                    if cutlass.const_expr(self.vectorized_f32):
                        for vi in cutlass.range_constexpr(abs_acc_frg.shape[1]):
                            tCrSFC_pvscale[vi] = abs_acc_frg[None, vi].reduce(
                                cute.ReductionOp.MAX,
                                cutlass.Float32(0.0),
                                0,
                            )
                        for vi in cutlass.range_constexpr(0, abs_acc_frg.shape[1], 2):
                            tCrSFC_pvscale[vi], tCrSFC_pvscale[vi + 1] = (
                                cute.arch.mul_packed_f32x2(
                                    (tCrSFC_pvscale[vi], tCrSFC_pvscale[vi + 1]),
                                    (c_rcp_limit, c_rcp_limit),
                                )
                            )
                            tCrSFC_pvscale[vi], tCrSFC_pvscale[vi + 1] = (
                                cute.arch.mul_packed_f32x2(
                                    (tCrSFC_pvscale[vi], tCrSFC_pvscale[vi + 1]),
                                    (norm_const, norm_const),
                                )
                            )
                    else:
                        for vi in cutlass.range_constexpr(abs_acc_frg.shape[1]):
                            tCrSFC_pvscale[vi] = (
                                abs_acc_frg[None, vi].reduce(
                                    cute.ReductionOp.MAX,
                                    cutlass.Float32(0.0),
                                    0,
                                )
                                * c_rcp_limit
                                * norm_const
                            )

                    # Round SFValue to E4M3 and store the block scales.
                    tCrSFC.store(tCrSFC_pvscale.load().to(self.sf_dtype))
                    cute.autovec_copy(tCrSFC, tCgSFC)

                    # outputScale = sf_scale * rcp(fp8(SFValue))  (== cvt recipe:
                    # reciprocal(fp8(SFValue) * reciprocal(sf_scale)) ), then
                    # rescale and convert each value to E2M1.
                    tCrSFC_qpvscale = tCrSFC.load().to(cutlass.Float32)
                    if cutlass.const_expr(self.vectorized_f32):
                        for vi in cutlass.range_constexpr(0, cute.size(tCrSFC), 2):
                            acc_scale = cute.arch.mul_packed_f32x2(
                                (
                                    cute.arch.rcp_approx(tCrSFC_qpvscale[vi]),
                                    cute.arch.rcp_approx(tCrSFC_qpvscale[vi + 1]),
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
                            acc_scale = norm_const * cute.arch.rcp_approx(tCrSFC_qpvscale[vi])
                            acc_scale = fmin(acc_scale, fp32_max, nan=True)

                            vec = tTR_rAcc_frg[None, vi]
                            for ei in cutlass.range_constexpr(self.sf_vec_size):
                                vec[ei] = vec[ei] * acc_scale

                    # Convert rescaled FP32 to packed E2M1 and stage to SMEM.
                    acc_vec = tiled_copy_r2s.retile(tCompute).load()
                    tRS_rC.store(acc_vec.to(self.c_dtype))

                    # Store FP4 C to SMEM
                    c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
                    cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])

                    # Fence and barrier
                    cute.arch.fence_proxy("async.shared", space="cta")
                    epilog_threads = 32 * len(self.epilogue_warp_id)
                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id,
                        number_of_threads=epilog_threads,
                    )

                    # TMA store FP4 C from SMEM to GMEM
                    if warp_idx == self.epilogue_warp_id[0]:
                        cute.copy(tma_atom_c, bSG_sC[(None, c_buffer)], bSG_gC[(None, subtile_idx)])
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()
                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id,
                        number_of_threads=epilog_threads,
                    )

                # Release accumulator buffer
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

            # Wait for C store complete and deallocate TMEM
            c_pipeline.producer_tail()

            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        # PDL: Launch dependent kernels
        griddepcontrol_launch_dependents()

    @staticmethod
    def get_dtype_rcp_limits(dtype: Type[cutlass.Numeric]) -> float:
        """Reciprocal of the max representable absolute value for a dtype.

        For NVFP4 (Float4E2M1FN) this is 1/6, matching the
        ``reciprocal_approximate_ftz(6.0f)`` factor in cvt_warp_fp16_to_fp4.
        """
        if dtype == cutlass.Float4E2M1FN:
            return 1 / 6.0
        if dtype == cutlass.Float8E4M3FN:
            return 1 / 448.0
        if dtype == cutlass.Float8E5M2:
            return 1 / 128.0
        return 1.0

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        swizzle_size: int,
        raster_along: Literal["m", "n"],
        max_active_clusters: cutlass.Constexpr,
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        """Compute grid size using persistent tile scheduler."""
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl, swizzle_size, raster_along == "m"
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    @staticmethod
    def _compute_num_tmem_alloc_cols(
        tiled_mma: cute.TiledMma,
        mma_tiler: Tuple[int, int, int],
        num_acc_stage: int,
    ) -> int:
        """Compute the number of tensor memory allocation columns."""
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, num_acc_stage))
        num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake)
        return num_tmem_alloc_cols

    @staticmethod
    def check_supported_dtypes(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """Check if the dtypes are valid for the NVFP4-out kernel."""
        valid_ab_dtypes = {cutlass.Float16, cutlass.BFloat16}
        if a_dtype not in valid_ab_dtypes or b_dtype not in valid_ab_dtypes:
            return False
        if a_dtype != b_dtype:
            return False
        if acc_dtype is not cutlass.Float32:
            return False
        if c_dtype is not cutlass.Float4E2M1FN:
            return False
        return True

    @staticmethod
    def is_valid_mma_tiler_and_cluster_shape(
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
    ) -> bool:
        """Check if the mma tiler and cluster shape are valid."""
        if not (
            (not use_2cta_instrs and mma_tiler_mn[0] in [64, 128])
            or (use_2cta_instrs and mma_tiler_mn[0] in [128, 256])
        ):
            return False
        if mma_tiler_mn[1] not in range(32, 257, 32):
            return False
        if cluster_shape_mn[0] % (2 if use_2cta_instrs else 1) != 0:
            return False
        if (
            cluster_shape_mn[0] * cluster_shape_mn[1] > 16
            or cluster_shape_mn[0] <= 0
            or cluster_shape_mn[1] <= 0
            or not is_power_of_2(cluster_shape_mn[0])
            or not is_power_of_2(cluster_shape_mn[1])
        ):
            return False
        return True

    @staticmethod
    def sf_rows_written(
        m: int,
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
    ) -> int:
        """Number of swizzled-SF M-rows the epilogue actually writes for ``m``.

        The SFC store is indexed by the persistent scheduler's CTA M-tile
        coordinate ``cur_tile_coord[0]`` times ``epi_tile_cnt_m`` (see the
        epilogue ``sfc_subtile_idx_mn`` site). That coordinate ranges over
        ``[0, n_cluster_m * cluster_m)`` where::

            cta_tile_m  = mma_tiler_mn[0] // (2 if use_2cta else 1)
            num_ctas_m  = ceil_div(m, cta_tile_m)          # from _compute_grid
            n_cluster_m = ceil_div(num_ctas_m, cluster_m)  # scheduler clustering

        and each tile coordinate maps to ``cta_tile_m`` SF rows
        (``epi_tile_cnt_m * EPI_M == cta_tile_m``). The store is NOT bounded by
        the SF tensor's own (pad_up(m,128)) M extent -- ``cute.autovec_copy``
        writes unconditionally -- so the highest row touched is::

            n_cluster_m * cluster_m * cta_tile_m

        This can exceed ``pad_up(m, 128)`` when ``cta_tile_m`` or the cluster
        grouping rounds ``m`` up past the next 128-row boundary, which is why an
        exact-size (``pad_up(m,128)``) SF buffer is only safe for tactics where
        this equals ``pad_up(m, 128)`` (see ``is_sf_write_in_bounds``).
        """
        cta_tile_m = mma_tiler_mn[0] // (2 if use_2cta_instrs else 1)
        cluster_m = cluster_shape_mn[0]
        num_ctas_m = (m + cta_tile_m - 1) // cta_tile_m
        n_cluster_m = (num_ctas_m + cluster_m - 1) // cluster_m
        return n_cluster_m * cluster_m * cta_tile_m

    @staticmethod
    def is_sf_write_in_bounds(
        m: int,
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
    ) -> bool:
        """True iff the epilogue's SF writes stay within ``pad_up(m, 128)`` rows.

        The downstream o_proj NVFP4 GEMM requires the activation SF buffer to be
        EXACTLY ``pad_up(M,128) * pad_up(K//16, 4)`` elements (no slack). A
        tactic is only usable with that exact-size buffer when the CTA-M tile /
        cluster grouping does not round ``m`` up past its ``pad_up(m,128)`` SF
        extent -- otherwise ``cute.autovec_copy`` in the SF epilogue writes out
        of bounds. Restricting tactics to this set (instead of padding the SF
        buffer with overrun slack) keeps the SF buffer exact-size and o_proj
        consumable.
        """
        padded_rows = ((m + 127) // 128) * 128
        return (
            PersistentDenseGemmNVFP4OutKernel.sf_rows_written(
                m, use_2cta_instrs, mma_tiler_mn, cluster_shape_mn
            )
            <= padded_rows
        )

    @staticmethod
    def can_implement(
        ab_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        m: int,
        n: int,
        k: int,
        batch_size: int,
        a_major: str,
        b_major: str,
        c_major: str,
        sf_vec_size: int = 16,
    ) -> bool:
        """Check if the NVFP4-out gemm can be implemented for these params."""
        if not PersistentDenseGemmNVFP4OutKernel.check_supported_dtypes(
            ab_dtype, ab_dtype, acc_dtype, c_dtype
        ):
            return False
        if not PersistentDenseGemmNVFP4OutKernel.is_valid_mma_tiler_and_cluster_shape(
            use_2cta_instrs, mma_tiler_mn, cluster_shape_mn
        ):
            return False
        # SF blocks must align with N tiles so per-head scales are independent.
        if n % sf_vec_size != 0:
            return False
        # Contiguous (K) dim must be 16B-aligned for BF16 A/B TMA.
        num_contig_ab = 16 * 8 // ab_dtype.width
        if k % num_contig_ab != 0:
            return False
        # FP4 packed C contiguous (N) dim: 2 values/byte, TMA needs 32-elem
        # alignment for the packed minor dim.
        if n % 32 != 0:
            return False
        # The activation SF buffer is sized EXACTLY pad_up(M,128)*pad_up(K//16,4)
        # (no overrun slack) so the downstream o_proj NVFP4 GEMM consumes it
        # directly. Reject any tactic whose CTA-M tile / cluster grouping would
        # make the SF epilogue write past pad_up(M,128) rows.
        if not PersistentDenseGemmNVFP4OutKernel.is_sf_write_in_bounds(
            m, use_2cta_instrs, mma_tiler_mn, cluster_shape_mn
        ):
            return False
        return True

    @cute.jit
    def wrapper_strided(
        self,
        m: cutlass.Int32,
        n: cutlass.Int32,
        k: cutlass.Int32,
        batch_size: cutlass.Int32,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        sfc_ptr: cute.Pointer,
        sf_scale_tensor: cute.Tensor,
        a_stride_m: cutlass.Int32,
        a_stride_batch: cutlass.Int32,
        c_stride_m: cutlass.Int32,
        c_stride_n: cutlass.Int32,
        c_stride_l: cutlass.Int32,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Execute the NVFP4-out GEMM with explicit A and C strides.

        Mirrors PersistentDenseGemmKernel.wrapper_strided but adds the SFC
        output pointer and the per-tensor sf_scale tensor. A may be a
        non-contiguous view (e.g. [M, B, K].transpose(0, 1)); B is contiguous
        and K-major.

        Args:
            m, n, k, batch_size: GEMM problem dims (per-batch M/N/K, L = batch).
            a_ptr: BF16 A data pointer.
            b_ptr: BF16 B data pointer.
            c_ptr: packed FP4 (Float4E2M1FN) output data pointer. The C cute
                tensor (M, N, L) is built INSIDE this jit-traced wrapper from
                c_ptr + the caller-supplied strides (cute.make_layout cannot run
                at Python/runtime level -- it needs an active MLIR context).
            sfc_ptr: E4M3 scale-factor output pointer (swizzled SF layout).
            sf_scale_tensor: single-element float32 per-tensor scale.
            a_stride_m: A stride along M (elements).
            a_stride_batch: A stride along batch (elements).
            c_stride_m: C stride along M (FP4 elements). For the V-up fusion the
                runner passes the M-major value L*N so the underlying buffer,
                viewed as [M, L*(N//2)] row-major, is the head-major monolithic
                activation the o_proj NVFP4 GEMM consumes with zero copy.
            c_stride_n: C stride along N (FP4 elements); 1 (unit-stride minor
                mode -- the TMA box minor mode).
            c_stride_l: C stride along L (FP4 elements); N for the V-up fusion.
            max_active_clusters: Maximum number of active clusters.
            stream: CUDA stream.
        """
        # A with explicit strides: (M, K, batch), K stride = 1.
        a_tensor = cute.make_tensor(
            a_ptr,
            layout=cute.make_layout(
                (m, k, batch_size),
                stride=(a_stride_m, 1, a_stride_batch),
            ),
        )
        # B contiguous: (N, K, batch), K innermost.
        b_tensor = cute.make_tensor(
            b_ptr,
            layout=cute.make_ordered_layout(
                (n, k, batch_size),
                order=(1, 0, 2),
            ),
        )
        # C packed FP4 (M, N, L), built INSIDE the jit. N is the unit-stride
        # minor mode (the TMA box minor mode); M/L carry the caller-supplied
        # M-major strides (c_stride_m == L*N, c_stride_l == N) for the zero-copy
        # monolithic reshape. cute.make_layout requires an active MLIR context,
        # so this construction MUST happen here (jit-traced), not at runtime.
        #
        # The N stride is pinned to the STATIC literal 1 rather than the dynamic
        # c_stride_n parameter (which is always 1 -- the TMA box minor mode
        # contract). This is required for utils.LayoutEnum.from_tensor(c) in
        # __call__: cute.leading_dim() classifies the layout by finding the mode
        # whose stride is the *static* constant 1. If N's stride is a dynamic
        # Int32, is_dynamic_expression() makes leading_dim() return None and
        # from_tensor() raises "Invalid leading dimension: None". With the static
        # 1 here, mode 1 (N) is recognized as the unit-stride minor mode, so C
        # classifies as ROW_MAJOR (is_n_major_c) -- the correct major-ness for
        # selecting the epilogue TMEM->RMEM / RMEM->SMEM copy atoms and smem
        # layout, regardless of the dynamic M/L batch strides. The physical TMA
        # store byte order is unchanged: it reads M/L strides from the layout and
        # writes element (m,n,l) at FP4 offset m*c_stride_m + n*1 + l*c_stride_l.
        # (c_stride_n is accepted for signature symmetry but is unused here; the
        # caller always passes 1, matching the static literal below.)
        c_tensor = cute.make_tensor(
            c_ptr,
            layout=cute.make_layout(
                (m, n, batch_size),
                stride=(c_stride_m, 1, c_stride_l),
            ),
        )
        # SFC iterator only; the swizzled layout is rebuilt via
        # tile_atom_to_shape_SF over the MONOLITHIC (M, L*N) shape in __call__.
        # This placeholder layout exists solely to carry the pointer.
        sfc_tensor = cute.make_tensor(
            sfc_ptr,
            layout=cute.make_layout((m, n * batch_size)),
        )

        self(
            a_tensor,
            b_tensor,
            c_tensor,
            sfc_tensor,
            sf_scale_tensor,
            max_active_clusters,
            stream,
        )
