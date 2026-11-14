# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# This file is ported from TensorRT-LLM's dense_blockscaled_gemm_persistent.py
# with modifications for FlashInfer integration.
# Original: https://github.com/NVIDIA/TensorRT-LLM

# FlashInfer's CuTe DSL MXFP8 dense GEMM (flashinfer/gemm/kernels/dense_blockscaled_gemm_sm100.py,
# https://github.com/flashinfer-ai/flashinfer) with one change, in the epilogue: each staged C subtile has every
# 32-bit word 0x80000000 (-0, the Lamport "empty" marker) replaced by +0, and is stored with TMA into this rank's
# section of the current Lamport data slot of both TP2 ranks (the AllReduce workspace) instead of into C. The
# mainloop, tile scheduling and the C conversion are FlashInfer's, so the stored bytes are the stock GEMM's output
# after the stock one-shot all-reduce's own push rule. The prebuild (OPROJ_PUSH_GEMM in modeling.py) compares this
# kernel's slot bytes with the stock GEMM on the real weights at every tactic before it is used.
# ruff: noqa: E731, E741

from pathlib import Path
from typing import Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch as _torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.arch import griddepcontrol_launch_dependents, griddepcontrol_wait
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.pipeline import PipelineTmaUmma, PipelineUmmaAsync
from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm100_common import _Sm100BlockScaledGemmCommon

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import cute_aot

# ``per_token_alpha`` value for one alpha per row of C (``"n"``: per column).
_ALPHA_PER_ROW = "m"


def _per_token_fragment_plan(shape, stride, token_axis):
    """Static analysis of a t2r fragment of the identity tensor over C.

    ``shape``/``stride`` are the (hierarchical, static) layout of the
    per-thread fragment; strides are ``ScaledBasis`` values ``k@axis`` or 0.
    Returns ``(offsets, distinct)``: for every fragment element (in the
    colexicographic order the fragment is indexed with) its token-axis
    offset relative to element 0, and the sorted distinct offsets. Returns
    ``None`` when the layout is not fully static, so the caller falls back
    to one load per element.
    """
    leaves = []

    def walk(sh, st):
        if isinstance(sh, (tuple, list)):
            if not isinstance(st, (tuple, list)) or len(sh) != len(st):
                return False
            return all(walk(a, b) for a, b in zip(sh, st, strict=True))
        if not isinstance(sh, int):
            return False
        mode = getattr(st, "mode", None)
        value = getattr(st, "value", st)
        if mode is None:
            if not isinstance(value, int) or value != 0:
                return sh == 1 or value == 0
            leaves.append((sh, 0))
            return True
        if not isinstance(value, int):
            return False
        leaves.append((sh, value if list(mode)[0] == token_axis else 0))
        return True

    if not walk(shape, stride):
        return None
    offsets = []
    total = 1
    for extent, _ in leaves:
        total *= extent
    for i in range(total):
        rem = i
        off = 0
        for extent, scale in leaves:
            off += (rem % extent) * scale
            rem //= extent
        offsets.append(off)
    return offsets, sorted(set(offsets))


@dsl_user_op
def _l2_cache_policy(evict: str, *, loc=None, ip=None) -> cutlass.Int64:
    """Runtime ``createpolicy`` L2 eviction policy (fraction 1.0) for TMA loads.

    ``evict`` is ``"evict_first"`` (stream-once operand) or ``"evict_last"``
    (operand re-read by every CTA). Kept a runtime instruction on purpose: a
    constant policy makes ptxas emit an invalid encoding on some toolchains.
    """
    return cutlass.Int64(
        llvm.inline_asm(
            cutlass.Int64.mlir_type,
            [],
            f"createpolicy.fractional.L2::{evict}.b64 $0, 1.0;",
            "=l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


class LamportPushGemmKernel(_Sm100BlockScaledGemmCommon):
    """Implements batched matrix multiplication (C = A x SFA x B x SFB) with support for various data types
    and Blackwell GPU architectural features, including persistent tile scheduling and warp specialization.

    Notes:
        - In the current version, tensors A and B must have the same data type.
          For example, using Float8E4M3FN for A and Float8E5M2 for B is not supported.
        - Supported combinations of A/B data types, scale factor (SF) data types, and SF vector size:
            * MXF8:   A/B: Float8E5M2/Float8E4M3FN, SF: Float8E8M0FNU, sf_vec_size: 32
            * MXF4:   A/B: Float4E2M1FN,            SF: Float8E8M0FNU, sf_vec_size: 32
            * NVF4:   A/B: Float4E2M1FN,            SF: Float8E8M0FNU/Float8E4M3FN, sf_vec_size: 16
        - Supported accumulator data types:
            * Float32
        - Supported C data types:
            * Float32
            * Float16/BFloat16
            * Float8E4M3FN/Float8E5M2
        - Constraints:
            * MMA tiler M must be 128 or 256 (use_2cta_instrs).
            * MMA tiler N must be 64/128/192/256
            * Cluster shape M must be a multiple of 2 if MMA tiler M is 256.
            * Cluster shape M/N must be positive and a power of 2, with total cluster size <= 16.
            * Cluster shape M/N must be <= 4 for scale factor multicasts due to limited scale factor size.

    Example:
        >>> gemm = Sm100BlockScaledPersistentDenseGemmKernel(
        ...     sf_vec_size=16, mma_tiler_mn=(256, 128), cluster_shape_mn=(2, 1)
        ... )
        >>> gemm(a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_tensor, max_active_clusters, stream)
    """

    # Largest kernel N a narrow (< 64) N tile may run at. A narrow tile is a
    # sub-tile of the 128-token SFB tile: the mainloop shifts the MMA's SFB
    # TMEM column by tok_off // 32, and an odd shift makes tcgen05.mma fault,
    # so every sub-tile must stay within the first 32 tokens. See can_implement.
    NARROW_TILE_MAX_N = 32

    @classmethod
    def narrow_tile_ok(cls, tile_n: int, kernel_n: int) -> bool:
        """Whether an MMA N tile of width ``tile_n`` may run at ``kernel_n``.

        Tiles of 64 or wider are unrestricted; narrow tiles must stay within
        ``NARROW_TILE_MAX_N`` kernel-N columns (kernel N is the token count
        under swap_ab). Shared by can_implement and by the mm_mxfp8 / mm_fp4
        runners, which re-check a tactic before launching it.
        """
        return tile_n >= 64 or kernel_n <= cls.NARROW_TILE_MAX_N

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
        per_token_alpha: Optional[str] = None,
        mma_inst_tile_k: int = 4,
        a_l2_evict_first: bool = False,
        b_l2_evict_last: bool = False,
        b_l2_evict_first: bool = False,
        a_l2_evict_last: bool = False,
    ):
        """Initializes the configuration for a Blackwell dense GEMM kernel.

        This configuration includes several key aspects:

        1.  MMA Instruction Settings (tcgen05):
            - acc_dtype: Data types for MMA accumulator, always set to Float32
            - sf_vec_size: Scalefactor A/B vector size.
            - mma_tiler_mn: The (M, N) shape of the MMA instruction tiler.

        2.  Cluster Shape:
            - cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster.

        Args:
            sf_vec_size (int): Scalefactor vector size.
            mma_tiler_mn (Tuple[int, int]): Shape of the Matrix Multiply-Accumulate (MMA) tile (M, N).
            cluster_shape_mn (Tuple[int, int]): Cluster dimensions (M, N) for parallel processing.
            per_token_alpha (Optional[str]): ``None`` for a scalar alpha, else
                the extent ``alpha`` holds one scale per coordinate of: ``"m"``
                per row of C, ``"n"`` per column (callers that swap A and B).
            mma_inst_tile_k (int): MMA K instructions per pipeline stage (4 or 8).
            a_l2_evict_first (bool): issue the A (and SFA) TMA loads with an L2
                ``evict_first`` policy: A is streamed exactly once per CTA.
            b_l2_evict_last (bool): issue the B (and SFB) TMA loads with an L2
                ``evict_last`` policy: every CTA of a column re-reads the tile.
            b_l2_evict_first (bool): issue the B (and SFB) TMA loads with an L2
                ``evict_first`` policy (B streamed once, e.g. weights without
                swap_ab when M fits one tile). Exclusive with b_l2_evict_last.
        """

        self.per_token_alpha = per_token_alpha
        if mma_inst_tile_k not in (4, 8):
            raise ValueError(f"mma_inst_tile_k must be 4 or 8, got {mma_inst_tile_k}")
        # MMA K instructions per pipeline stage: 4 (K tile 256 for FP4) or 8
        # (K tile 512, a 256 B TMA row per operand row; used for narrow N tiles
        # whose weight stream is DRAM-efficiency bound).
        self.mma_inst_tile_k = mma_inst_tile_k
        if b_l2_evict_first and b_l2_evict_last:
            raise ValueError("b_l2_evict_first and b_l2_evict_last are exclusive")
        if a_l2_evict_first and a_l2_evict_last:
            raise ValueError("a_l2_evict_first and a_l2_evict_last are exclusive")
        self.a_l2_evict_last = a_l2_evict_last
        self.a_l2_evict_first = a_l2_evict_first
        self.b_l2_evict_last = b_l2_evict_last
        self.b_l2_evict_first = b_l2_evict_first
        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = sf_vec_size
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        self.cluster_shape_mn = cluster_shape_mn
        # K dimension is deferred in _setup_attributes
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.use_prefetch = use_prefetch
        self.enable_pdl = enable_pdl
        self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE

        self.occupancy = 1
        # Set specialized warp ids
        self.epilog_warp_id = (
            0,
            1,
            2,
            3,
        )
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * len((self.mma_warp_id, self.tma_warp_id, *self.epilog_warp_id))
        # Set barrier id for cta sync, epilogue sync and tmem ptr sync
        self.cta_sync_bar_id = 0
        self.epilog_sync_bar_id = 1
        self.tmem_ptr_sync_bar_id = 2
        # NamedBarrier equivalents of the two ids above, for subclasses written
        # against the newer CuTe DSL API that take the object rather than the id.
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=self.epilog_sync_bar_id,
            num_threads=32 * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_ptr_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.num_tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B/SFA/SFB
        - Computing epilogue subtile
        - Setting up A/B/SFA/SFB/C stage counts in shared memory
        - Computing A/B/SFA/SFB/C shared memory layout
        - Computing tensor memory allocation columns
        """
        # Compute mma instruction shapes
        mma_inst_bits_k = 256
        # (MMA_Tile_Shape_M, MMA_Tile_Shape_N, MMA_Inst_Shape_K)
        self.mma_inst_shape_mnk = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_bits_k // self.a_dtype.width,
        )
        # (CTA_Tile_Shape_M, Round_Up(MMA_Tile_Shape_N, 128), MMA_Inst_Shape_K)
        self.mma_inst_shape_mnk_sfb = (
            self.mma_inst_shape_mnk[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mnk[1], 128),
            self.mma_inst_shape_mnk[2],
        )

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mnk[:2],
        )

        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mnk_sfb[:2],
        )

        # Compute mma/cluster/tile shapes
        mma_inst_tile_k = self.mma_inst_tile_k
        self.mma_tiler = (
            self.mma_inst_shape_mnk[0],
            self.mma_inst_shape_mnk[1],
            self.mma_inst_shape_mnk[2] * mma_inst_tile_k,
        )
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mnk_sfb[0],
            self.mma_inst_shape_mnk_sfb[1],
            self.mma_inst_shape_mnk_sfb[2] * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
        )
        # Number of CTA N tiles covered by one (128-wide) SFB tile
        self.sfb_sub_tiles_per_tile = max(
            1, self.cta_tile_shape_mnk_sfb[1] // self.cta_tile_shape_mnk[1]
        )
        # The S2T copy of a sub-tile starts up to 31 token rows (16 B each) into
        # the 512 B SF block and reads the same length, so pad sSFB by one block.
        self.sfb_smem_pad_bytes = 512 if self.cta_tile_shape_mnk[1] < 64 else 0

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
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        # Compute epilogue subtile
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )

        self.epi_tile_n = cute.size(self.epi_tile[1])

        # Setup A/B/C stage count in shared memory and ACC stage count in tensor memory
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.a_major_mode,
            self.b_dtype,
            self.b_major_mode,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.sf_dtype,
            self.sf_vec_size,
            self.smem_capacity,
            self.occupancy,
            1024 + self.sfb_smem_pad_bytes,
        )

        # Compute A/B/SFA/SFB/C shared memory layout
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
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
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
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_c_stage,
        )

        self.overlapping_accum = self.num_acc_stage == 1
        sf_atom_mn = 32
        self.num_sfa_tmem_cols = (self.cta_tile_shape_mnk[0] // sf_atom_mn) * mma_inst_tile_k
        self.num_sfb_tmem_cols = (self.cta_tile_shape_mnk_sfb[1] // sf_atom_mn) * mma_inst_tile_k
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        self.num_accumulator_tmem_cols = (
            self.cta_tile_shape_mnk[1] * self.num_acc_stage
            if not self.overlapping_accum
            else self.cta_tile_shape_mnk[1] * 2 - self.num_sf_tmem_cols
        )

        # Only when overlapping_accum is enabled, we need to release accumulator buffer early in epilogue
        self.iter_acc_early_release_in_epilogue = self.num_sf_tmem_cols // self.epi_tile_n

        # TODO: Currently set prefetch dist to num_ab_stage, we may have more options for prefetch dist auto
        # tuning
        self.prefetch_dist = self.num_ab_stage

    @cute.jit
    def __call__(
        self,
        a_tensor: cute.Tensor,
        b_tensor: cute.Tensor,
        sfa_tensor: cute.Tensor,
        sfb_tensor: cute.Tensor,
        c_tensor: cute.Tensor,
        alpha: cute.Tensor,  # (1,) scalar alpha, or (m,)/(n,) per-token alpha
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        c_s0l: cute.Tensor,
        c_s0p: cute.Tensor,
        c_s1l: cute.Tensor,
        c_s1p: cute.Tensor,
        c_s2l: cute.Tensor,
        c_s2p: cute.Tensor,
        flag_tensor: cute.Tensor,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes before smem/grid/tma computation
        - Setup TMA load/store atoms and tensors
        - Compute grid size with regard to hardware constraints
        - Define shared storage for kernel
        - Launch the kernel synchronously

        Args:
            a_tensor (cute.Tensor): Input tensor A
            b_tensor (cute.Tensor): Input tensor B
            sfa_tensor (cute.Tensor): Scale factor tensor A
            sfb_tensor (cute.Tensor): Scale factor tensor B
            c_tensor (cute.Tensor): Output tensor C
            max_active_clusters (cutlass.Constexpr): Maximum number of active clusters
            stream (cuda.CUstream): CUDA stream for asynchronous execution
            epilogue_op (cutlass.Constexpr): Optional elementwise lambda function to apply to the output tensor

        Raises:
            TypeError: If input data types are incompatible with the MMA instruction.
        """
        (
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
        ) = self._prepare_tma(
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            c_tensor,
        )

        # Lamport push: one TMA store atom per Lamport data slot (3 rotating slots x local / peer section of this rank)
        push_epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        push_atoms = []
        for push_t in (c_s0l, c_s0p, c_s1l, c_s1p, c_s2l, c_s2p):
            push_atom, _ = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(), push_t, push_epi_smem_layout, self.epi_tile
            )
            push_atoms.append(push_atom)

        # Compute grid size
        self.tile_sched_params, grid = self._compute_grid(
            c_tensor,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            max_active_clusters,
        )

        self.buffer_align_bytes = 1024

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype,
                    cute.cosize(self.sfb_smem_layout_staged) + self.sfb_smem_pad_bytes,
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            epilogue_op,
            alpha,
            push_atoms[0],
            push_atoms[1],
            push_atoms[2],
            push_atoms[3],
            push_atoms[4],
            push_atoms[5],
            flag_tensor,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            smem=self.shared_storage.size_in_bytes(),  # type: ignore[attr-defined]
            min_blocks_per_mp=1,
            stream=stream,
            use_pdl=self.enable_pdl,
        )
        return

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
        alpha: cute.Tensor,
        push_s0l: cute.CopyAtom,
        push_s0p: cute.CopyAtom,
        push_s1l: cute.CopyAtom,
        push_s1p: cute.CopyAtom,
        push_s2l: cute.CopyAtom,
        push_s2p: cute.CopyAtom,
        push_flag: cute.Tensor,
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        """
        # Keep alpha in FP32 for precision: the accumulator is in FP32 and alpha
        # may be a very small scaling factor. Converting to c_dtype (e.g., FP16)
        # before multiplication could cause overflow when acc values are large.
        # In per-token mode the epilogue folds the scale into the accumulators
        # once the output coordinate is known, so this stays neutral.
        if cutlass.const_expr(self.per_token_alpha is not None):
            alpha_value = cutlass.Float32(1.0)
        else:
            alpha_value = alpha[0].to(cutlass.Float32)

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_c)
            cpasync.prefetch_descriptor(push_s0l)
            cpasync.prefetch_descriptor(push_s0p)
            cpasync.prefetch_descriptor(push_s1l)
            cpasync.prefetch_descriptor(push_s1p)
            cpasync.prefetch_descriptor(push_s2l)
            cpasync.prefetch_descriptor(push_s2p)

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

        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr.ptr
        tmem_holding_buf = storage.tmem_holding_buf.ptr

        # Initialize mainloop ab_pipeline (barrier) and states
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

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (2 if use_2cta_instrs else 1)
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

        # Tensor memory dealloc barrier init
        if use_2cta_instrs:
            if warp_idx == self.tma_warp_id:
                num_tmem_dealloc_threads = 32
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(tmem_dealloc_mbar_ptr, num_tmem_dealloc_threads)
        cute.arch.mbarrier_init_fence()

        # Cluster arrive after barrier init
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed(aligned=True)

        #
        # Setup smem tensor A/B/SFA/SFB/C
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sC = storage.sC.get_tensor(c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner)
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        # (MMA, MMA_M, MMA_K, STAGE)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        # (MMA, MMA_N, MMA_K, STAGE)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        #
        # Compute multicast mask for A/B/SFA/SFB buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
            )

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bK, RestM, RestK, RestL)
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )
        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_block_cnt = cutlass.Int32(cute.size(gA_mkl, mode=[3]))

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCgC = thr_mma.partition_C(gC_mnl)

        if cutlass.const_expr(self.per_token_alpha is not None):
            # Identity tensor over C, partitioned like tCgC, so the epilogue can
            # recover each accumulator element's (m, n) coordinate.
            cC_mnl = cute.local_tile(
                cute.make_identity_tensor(mC_mnl.shape),
                cute.slice_(self.mma_tiler, (None, None, 0)),
                (None, None, None),
            )
            # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
            tCcC = thr_mma.partition_C(cC_mnl)

        #
        # Partition global/shared tensor for TMA load A/B
        #
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        #  TMA load SFA partition_S/D
        sfa_cta_layout = a_cta_layout
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            sfa_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

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
        if cutlass.const_expr(self.overlapping_accum):
            num_acc_stage_overlapped = 2
            tCtAcc_fake = tiled_mma.make_fragment_C(
                cute.append(acc_shape, num_acc_stage_overlapped)
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_fake = cute.make_tensor(
                tCtAcc_fake.iterator,
                cute.make_layout(
                    tCtAcc_fake.shape,
                    stride=(
                        tCtAcc_fake.stride[0],
                        tCtAcc_fake.stride[1],
                        tCtAcc_fake.stride[2],
                        (256 - self.num_sf_tmem_cols) * tCtAcc_fake.stride[0][1],
                    ),
                ),
            )
        else:
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        #
        # Cluster wait before tensor memory alloc
        #
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier(
                barrier_id=self.cta_sync_bar_id, number_of_threads=self.threads_per_cta
            )

        # PDL bookend: griddepcontrol_wait is always emitted; the actual PDL
        # behavior is controlled by use_pdl= in .launch(). The griddepcontrol
        # instructions are effectively no-ops when PDL is not enabled at
        # the driver level.
        griddepcontrol_wait()

        #
        # Specialized TMA load warp
        #
        if warp_idx == self.tma_warp_id:
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            # L2 eviction policies for the operand streams (None = default).
            a_cache_policy = None
            b_cache_policy = None
            if cutlass.const_expr(self.a_l2_evict_first):
                a_cache_policy = _l2_cache_policy("evict_first")
            if cutlass.const_expr(self.a_l2_evict_last):
                a_cache_policy = _l2_cache_policy("evict_last")
            if cutlass.const_expr(self.b_l2_evict_last):
                b_cache_policy = _l2_cache_policy("evict_last")
            if cutlass.const_expr(self.b_l2_evict_first):
                b_cache_policy = _l2_cache_policy("evict_first")

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                #
                # Slice to per mma tile index
                #
                # ((atom_v, rest_v), RestK)
                tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
                # ((atom_v, rest_v), RestK)
                tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

                # ((atom_v, rest_v), RestK)
                tAgSFA_slice = tAgSFA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
                slice_n = mma_tile_coord_mnl[1]
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    slice_n = mma_tile_coord_mnl[1] // 2
                elif cutlass.const_expr(self.cta_tile_shape_mnk[1] < 64):
                    # Several narrow N tiles share one 128-wide SFB tile; the MMA
                    # reads its sub-tile through a shifted TMEM address below.
                    slice_n = mma_tile_coord_mnl[1] // self.sfb_sub_tiles_per_tile
                # ((atom_v, rest_v), RestK)
                tBgSFB_slice = tBgSFB[(None, slice_n, None, mma_tile_coord_mnl[2])]

                if cutlass.const_expr(self.use_prefetch):
                    # Prefetch both A and B (default behavior)
                    for pf_k_block in cutlass.range(
                        0, min(self.prefetch_dist, k_block_cnt), unroll=1
                    ):
                        cute.prefetch(
                            tma_atom_a,
                            tAgA_slice[(None, pf_k_block)],
                        )
                        cute.prefetch(
                            tma_atom_b,
                            tBgB_slice[(None, pf_k_block)],
                        )
                        cute.prefetch(
                            tma_atom_sfa,
                            tAgSFA_slice[(None, pf_k_block)],
                        )
                        cute.prefetch(
                            tma_atom_sfb,
                            tBgSFB_slice[(None, pf_k_block)],
                        )

                # Peek (try_wait) AB buffer empty for k_block = prefetch_k_block_cnt
                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_block_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
                #
                # Tma load loop
                #
                for k_block in cutlass.range(0, k_block_cnt, 1, unroll=1):
                    # Conditionally wait for AB buffer empty
                    ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)

                    # TMA load A/B/SFA/SFB
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, ab_producer_state.count)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=a_full_mcast_mask,
                        cache_policy=a_cache_policy,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, ab_producer_state.count)],
                        tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=b_full_mcast_mask,
                        cache_policy=b_cache_policy,
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_slice[(None, ab_producer_state.count)],
                        tAsSFA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfa_full_mcast_mask,
                        cache_policy=a_cache_policy,
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_slice[(None, ab_producer_state.count)],
                        tBsSFB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfb_full_mcast_mask,
                        cache_policy=b_cache_policy,
                    )

                    # Prefetch: Rolling prefetch for next tiles
                    if cutlass.const_expr(self.use_prefetch):
                        if k_block < k_block_cnt - self.prefetch_dist:
                            future_k_block = ab_producer_state.count + self.prefetch_dist
                            # Prefetch both A and B (default behavior)
                            cute.prefetch(
                                tma_atom_a,
                                tAgA_slice[(None, future_k_block)],
                            )
                            cute.prefetch(
                                tma_atom_b,
                                tBgB_slice[(None, future_k_block)],
                            )
                            cute.prefetch(
                                tma_atom_sfa,
                                tAgSFA_slice[(None, future_k_block)],
                            )
                            cute.prefetch(
                                tma_atom_sfb,
                                tBgSFB_slice[(None, future_k_block)],
                            )

                    # Peek (try_wait) AB buffer empty for k_block = prefetch_k_block_cnt + k_block + 1
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_block_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Wait A/B buffer empty
            #
            ab_pipeline.producer_tail(ab_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # Bar sync for retrieve tensor memory ptr from shared mem
            #
            tmem_ptr_read_threads = 32 * len((self.mma_warp_id, *self.epilog_warp_id))
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id,
                number_of_threads=tmem_ptr_read_threads,
            )

            #
            # Retrieving tensor memory ptr and make accumulator/SFA/SFB tensor
            #
            # Make accumulator tmem tensor
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # Make SFA tmem tensor
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols,
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_M, MMA_K)
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            # Make SFB tmem tensor
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_N, MMA_K)
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
            #
            # Partition for S2T copy of SFA/SFB
            #
            tiled_copy_s2t_sfa, tCsSFA_compact_s2t, tCtSFA_compact_s2t = (
                self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            )
            tiled_copy_s2t_sfb, tCsSFB_compact_s2t, tCtSFB_compact_s2t = (
                self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)
            )

            #
            # Persistent tile scheduling loop
            #
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
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_producer_state.phase ^ 1
                else:
                    acc_stage_index = acc_producer_state.index

                # Set tensor memory buffer for current tile
                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                # Peek (try_wait) AB buffer full for k_block = 0
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_block_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                #
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                tCtSFB_mma = tCtSFB
                tCsSFB_s2t_tile = tCsSFB_compact_s2t
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
                    # If this is an ODD tile, shift the TMEM start address for cta_tile_shape_n=192 case by two
                    # words (ignores first 64 columns of SFB)
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
                elif cutlass.const_expr(self.cta_tile_shape_mnk[1] < 64):
                    # Sub-tile of the 128-wide SFB tile. SF for token 32*c + l
                    # lives in TMEM column c, lane l: shift the MMA's SFB address
                    # by one column per 32 tokens and start the S2T copy's smem
                    # source at the remaining token row (16 B per row of the
                    # 32x4 SF block) so the sub-tile's first token lands in lane 0.
                    # can_implement limits narrow tiles to a kernel-N extent of
                    # 32, so the column shift below is always 0 in practice: an
                    # odd column shift makes tcgen05.mma fault (misaligned
                    # address), the 64-wide tile above shifts by two columns.
                    tok_off = (
                        mma_tile_coord_mnl[1] % self.sfb_sub_tiles_per_tile
                    ) * self.cta_tile_shape_mnk[1]
                    offset = cutlass.Int32(tok_off // 32)
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + self.num_accumulator_tmem_cols
                        + self.num_sfa_tmem_cols
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)
                    sSFB_shifted = cute.make_tensor(
                        sSFB.iterator + cutlass.Int32(tok_off % 32) * 16,
                        sSFB.layout,
                    )
                    _, tCsSFB_s2t_tile, _ = self.mainloop_s2t_copy_and_partition(
                        sSFB_shifted, tCtSFB
                    )

                #
                # Reset the ACCUMULATE field for each tile
                #
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                #
                # Mma mainloop
                #
                for _k_block in range(k_block_cnt):
                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)

                        #  Copy SFA/SFB from smem to tmem
                        s2t_stage_coord = (
                            None,
                            None,
                            None,
                            None,
                            ab_consumer_state.index,
                        )
                        tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
                        tCsSFB_compact_s2t_staged = tCsSFB_s2t_tile[s2t_stage_coord]
                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_compact_s2t_staged,
                            tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t_staged,
                            tCtSFB_compact_s2t,
                        )

                        # tCtAcc += tCrA * tCrSFA * tCrB * tCrSFB
                        num_kphases = cute.size(tCrA, mode=[2])
                        for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                            kphase_coord = (
                                None,
                                None,
                                kphase_idx,
                                ab_consumer_state.index,
                            )

                            # Set SFA/SFB tensor to tiled_mma
                            sf_kphase_coord = (None, None, kphase_idx)
                            tiled_mma.set(
                                tcgen05.Field.SFA,
                                tCtSFA[sf_kphase_coord].iterator,
                            )
                            tiled_mma.set(
                                tcgen05.Field.SFB,
                                tCtSFB_mma[sf_kphase_coord].iterator,
                            )

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kphase_coord],
                                tCrB[kphase_coord],
                                tCtAcc,
                            )

                            # Enable accumulate on tCtAcc after first kphase
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # Async arrive AB buffer empty
                        ab_pipeline.consumer_release(ab_consumer_state)

                    # Peek (try_wait) AB buffer full for k_block = k_block + 1
                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_block_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                #
                # Async arrive accumulator buffer full
                #
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Wait for accumulator buffer empty
            #
            acc_pipeline.producer_tail(acc_producer_state)
        #
        # Specialized epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            #
            # Alloc tensor memory buffer
            #
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.alloc_tmem(
                    self.num_tmem_alloc_cols,
                    tmem_holding_buf,
                    is_two_cta=use_2cta_instrs,
                )

            #
            # Bar sync for retrieve tensor memory ptr from shared memory
            #
            tmem_ptr_read_threads = 32 * len((self.mma_warp_id, *self.epilog_warp_id))
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id,
                number_of_threads=tmem_ptr_read_threads,
            )

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            #
            # Partition for epilogue
            #
            epi_tidx = tidx
            tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs
            )

            if cutlass.const_expr(self.per_token_alpha is not None):
                # Partitioned like tCgC, so this mirrors the t2r register
                # fragment element for element.
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
                tTR_cC_partitioned = tiled_copy_t2r.get_slice(epi_tidx).partition_D(
                    cute.flat_divide(tCcC[((None, None), 0, 0, None, None, None)], epi_tile)
                )
                alpha_extent = (
                    mC_mnl.shape[0]
                    if cutlass.const_expr(self.per_token_alpha == _ALPHA_PER_ROW)
                    else mC_mnl.shape[1]
                )
                token_axis = 0 if self.per_token_alpha == _ALPHA_PER_ROW else 1
                # Static plan: which fragment elements share a token, and
                # whether the token is constant across the epilogue subtiles.
                _frag_layout = tTR_cC_partitioned.layout
                _frag_shape = _frag_layout.shape
                _frag_stride = _frag_layout.stride
                alpha_plan = _per_token_fragment_plan(
                    tuple(_frag_shape[0:3]), tuple(_frag_stride[0:3]), token_axis
                )
                _subtile_plan = _per_token_fragment_plan(
                    tuple(_frag_shape[3:5]), tuple(_frag_stride[3:5]), token_axis
                )
                alpha_token_per_tile = (
                    alpha_plan is not None
                    and len(alpha_plan[1]) == 1
                    and _subtile_plan is not None
                    and _subtile_plan[1] == [0]
                )

            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r, tTR_rC, epi_tidx, sC
            )
            tma_atom_c, bSG_sC, bSG_gC_partitioned = self.epilog_gmem_copy_and_partition(
                epi_tidx, tma_atom_c, tCgC, epi_tile, sC
            )
            # Lamport push: the Lamport slot of this call, and the C staging buffer as 32-bit words
            push_slot = push_flag[0] % 3
            push_stage_words = cute.cosize(c_smem_layout_staged.outer) // (2 * self.num_c_stage)
            push_words = cute.make_tensor(
                cute.recast_ptr(storage.sC.data_ptr(), dtype=cutlass.Int32),
                cute.make_layout(push_stage_words * self.num_c_stage),
            )
            push_epi_threads = 32 * len(self.epilog_warp_id)

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

            # Threads/warps participating in tma store pipeline
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=c_producer_group,
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                #
                # Slice to per mma tile index
                #
                # ((ATOM_V, REST_V), EPI_M, EPI_N)
                bSG_gC = bSG_gC_partitioned[
                    (
                        None,
                        None,
                        None,
                        *mma_tile_coord_mnl,
                    )
                ]

                if cutlass.const_expr(self.per_token_alpha is not None):
                    # (T2R, T2R_M, T2R_N, (EPI_M, EPI_N))
                    tTR_cC = tTR_cC_partitioned[
                        (
                            None,
                            None,
                            None,
                            None,
                            None,
                            *mma_tile_coord_mnl,
                        )
                    ]
                    tTR_cC = cute.group_modes(tTR_cC, 3, cute.rank(tTR_cC))
                    if cutlass.const_expr(alpha_token_per_tile):
                        # Every element this thread holds in this tile has
                        # the same token: one clamped load per tile, applied
                        # through the scalar alpha multiply below.
                        tile_token = tTR_cC[(0, 0, 0, 0)][token_axis]
                        tile_token = cutlass.min(tile_token, alpha_extent - 1)
                        alpha_value = alpha[tile_token].to(cutlass.Float32)

                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_consumer_state.phase
                    reverse_subtile = (
                        cutlass.Boolean(True) if acc_stage_index == 0 else cutlass.Boolean(False)
                    )
                else:
                    acc_stage_index = acc_consumer_state.index

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_stage_index)]

                #
                # Wait for accumulator buffer full
                #
                acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                #
                # Store accumulator to global memory in sub-tiles
                #
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt

                for subtile_idx in cutlass.range(subtile_cnt):
                    real_subtile_idx = subtile_idx
                    if cutlass.const_expr(self.overlapping_accum):
                        if reverse_subtile:
                            real_subtile_idx = (
                                self.cta_tile_shape_mnk[1] // self.epi_tile_n - 1 - subtile_idx
                            )
                    #
                    # Load accumulator from tensor memory buffer to register
                    #
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, real_subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                    #
                    # Async arrive accumulator buffer empty earlier when overlapping_accum is enabled
                    #
                    if cutlass.const_expr(self.overlapping_accum):
                        if subtile_idx == self.iter_acc_early_release_in_epilogue:
                            # Fence for TMEM load
                            cute.arch.fence_view_async_tmem_load()
                            with cute.arch.elect_one():
                                acc_pipeline.consumer_release(acc_consumer_state)
                            acc_consumer_state.advance()
                    #
                    # Fold the per-token scale into the accumulators
                    #
                    if cutlass.const_expr(
                        self.per_token_alpha is not None and not alpha_token_per_tile
                    ):
                        tTR_cC_subtile = tTR_cC[(None, None, None, real_subtile_idx)]
                        if cutlass.const_expr(alpha_plan is not None):
                            # The fragment's token offsets are static: load
                            # each distinct token once per subtile (clamped so
                            # the rows past the token extent, which the TMA
                            # store drops, never read alpha out of bounds).
                            offsets, distinct = alpha_plan
                            token0 = tTR_cC_subtile[0][token_axis]
                            loaded = {}
                            for d in distinct:
                                tok = cutlass.min(token0 + d, alpha_extent - 1)
                                loaded[d] = alpha[tok].to(cutlass.Float32)
                            for i in cutlass.range_constexpr(len(offsets)):
                                tTR_rAcc[i] = tTR_rAcc[i] * loaded[offsets[i]]
                        else:
                            for i in cutlass.range_constexpr(cute.size(tTR_cC_subtile)):
                                coord = tTR_cC_subtile[i]
                                token = coord[token_axis]
                                token = cutlass.min(token, alpha_extent - 1)
                                tTR_rAcc[i] = tTR_rAcc[i] * alpha[token].to(cutlass.Float32)

                    #
                    # Convert to C type
                    #
                    acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                    # Multiply alpha in FP32 before converting to c_dtype to
                    # avoid overflow when c_dtype is FP16 and acc values are large.
                    acc_vec = epilogue_op((alpha_value * acc_vec).to(self.c_dtype))
                    tRS_rC.store(acc_vec)

                    #
                    # Store C to shared memory
                    #
                    c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rC,
                        tRS_sC[(None, None, None, c_buffer)],
                    )
                    # Lamport push: the stock push's -0 word rule on the whole staged subtile (other threads' words too)
                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id,
                        number_of_threads=push_epi_threads,
                    )
                    for push_i in cutlass.range(epi_tidx, push_stage_words, push_epi_threads):
                        push_j = c_buffer * push_stage_words + push_i
                        if push_words[push_j] == cutlass.Int32(-2147483648):
                            push_words[push_j] = cutlass.Int32(0)
                    # Fence and barrier to make sure shared memory store is visible to TMA store
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    epilog_threads = 32 * len(self.epilog_warp_id)
                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id,
                        number_of_threads=epilog_threads,
                    )

                    #
                    # TMA store C to global memory
                    #
                    if warp_idx == self.epilog_warp_id[0]:
                        if push_slot == 0:
                            cute.copy(
                                push_s0l, bSG_sC[(None, c_buffer)], bSG_gC[(None, real_subtile_idx)]
                            )
                            cute.copy(
                                push_s0p, bSG_sC[(None, c_buffer)], bSG_gC[(None, real_subtile_idx)]
                            )
                        elif push_slot == 1:
                            cute.copy(
                                push_s1l, bSG_sC[(None, c_buffer)], bSG_gC[(None, real_subtile_idx)]
                            )
                            cute.copy(
                                push_s1p, bSG_sC[(None, c_buffer)], bSG_gC[(None, real_subtile_idx)]
                            )
                        else:
                            cute.copy(
                                push_s2l, bSG_sC[(None, c_buffer)], bSG_gC[(None, real_subtile_idx)]
                            )
                            cute.copy(
                                push_s2p, bSG_sC[(None, c_buffer)], bSG_gC[(None, real_subtile_idx)]
                            )
                        # Fence and barrier to make sure shared memory store is visible to TMA store
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()
                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id,
                        number_of_threads=epilog_threads,
                    )

                #
                # Async arrive accumulator buffer empty
                #
                if cutlass.const_expr(not self.overlapping_accum):
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Dealloc the tensor memory buffer
            #
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=use_2cta_instrs)
            epilog_threads = 32 * len(self.epilog_warp_id)
            cute.arch.barrier(barrier_id=self.epilog_sync_bar_id, number_of_threads=epilog_threads)
            if warp_idx == self.epilog_warp_id[0]:
                if use_2cta_instrs:
                    cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr, cta_rank_in_cluster ^ 1)
                    cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
                cute.arch.dealloc_tmem(
                    acc_tmem_ptr, self.num_tmem_alloc_cols, is_two_cta=use_2cta_instrs
                )
            #
            # Wait for C store complete
            #
            c_pipeline.producer_tail()

        griddepcontrol_launch_dependents()

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        """Computes the grid size for the kernel launch.

        This method uses a persistent tile scheduler to determine the grid
        dimensions based on the output tensor shape and hardware constraints.

        Args:
            c (cute.Tensor): The output tensor C.
            cta_tile_shape_mnk (Tuple[int, int, int]): The shape (M, N, K) of the
                CTA tile.
            cluster_shape_mn (Tuple[int, int]): The shape of a CTA cluster.
            max_active_clusters (cutlass.Constexpr): The maximum number of
                active clusters supported by the hardware.

        Returns:
            A tuple containing the tile scheduler parameters and the computed
            grid shape.
        """
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(num_ctas_mnl, cluster_shape_mnl)
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    @staticmethod
    def is_valid_dtypes_and_scale_factor_vec_size(
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """Checks if the data types and scale factor vector size are a valid combination.

        Args:
            ab_dtype (Type[cutlass.Numeric]): Data type of operands A and B.
            sf_dtype (Type[cutlass.Numeric]): Data type of the scale factors.
            sf_vec_size (int): Vector size of the scale factors.
            c_dtype (Type[cutlass.Numeric]): Data type of the output tensor C.

        Returns:
            bool: True if the combination is valid, False otherwise.
        """
        is_valid = True

        # Check valid ab_dtype
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
        """Checks if the tensor layouts are valid for the given data types.

        Args:
            ab_dtype (Type[cutlass.Numeric]): Data type of operands A and B.
            c_dtype (Type[cutlass.Numeric]): Data type of the output tensor C.
            a_major (str): The major layout of tensor A ('k' or 'm').
            b_major (str): The major layout of tensor B ('k' or 'n').
            c_major (str): The major layout of tensor C ('n' or 'm').

        Returns:
            bool: True if the layouts are valid, False otherwise.
        """
        is_valid = True

        if ab_dtype is cutlass.Float4E2M1FN and not (a_major == "k" and b_major == "k"):
            is_valid = False
        return is_valid

    @staticmethod
    def is_valid_mma_tiler_and_cluster_shape(
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
    ) -> bool:
        """Checks if the MMA tiler and cluster shape are a valid combination.

        Args:
            mma_tiler_mn (Tuple[int, int]): The (M, N) shape of the MMA tiler.
            cluster_shape_mn (Tuple[int, int]): The (M, N) shape of the CTA cluster.

        Returns:
            bool: True if the combination is valid, False otherwise.
        """
        is_valid = True
        # Skip invalid mma tile shape
        if mma_tiler_mn[0] not in [128, 256]:
            is_valid = False
        if mma_tiler_mn[1] not in [8, 16, 32, 64, 128, 192, 256]:
            is_valid = False
        # Skip illegal cluster shape
        if cluster_shape_mn[0] % (2 if mma_tiler_mn[0] == 256 else 1) != 0:
            is_valid = False
        # Skip invalid cluster shape
        _is_power_of_2 = lambda x: x > 0 and (x & (x - 1)) == 0
        if (
            cluster_shape_mn[0] * cluster_shape_mn[1] > 16
            or cluster_shape_mn[0] <= 0
            or cluster_shape_mn[1] <= 0
            # Special cluster shape check for scale factor multicasts.
            # Due to limited size of scale factors, we can't multicast among more than 4 CTAs.
            or cluster_shape_mn[0] > 4
            or cluster_shape_mn[1] > 4
            or not _is_power_of_2(cluster_shape_mn[0])
            or not _is_power_of_2(cluster_shape_mn[1])
        ):
            is_valid = False
        return is_valid

    @staticmethod
    def is_valid_tensor_alignment(
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """Checks if the tensor dimensions are valid for memory alignment.

        Args:
            m (cutlass.Int64): The M dimension of the GEMM problem.
            n (cutlass.Int64): The N dimension of the GEMM problem.
            k (cutlass.Int64): The K dimension of the GEMM problem.
            l (cutlass.Int64): The batch dimension (L) of the GEMM problem.
            ab_dtype (Type[cutlass.Numeric]): Data type of operands A and B.
            c_dtype (Type[cutlass.Numeric]): Data type of the output tensor C.
            a_major (str): The major layout of tensor A ('k' or 'm').
            b_major (str): The major layout of tensor B ('k' or 'n').
            c_major (str): The major layout of tensor C ('n' or 'm').

        Returns:
            bool: True if the tensor alignment is valid, False otherwise.
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
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """Checks if the kernel can implement the given GEMM problem.

        This method aggregates checks for data types, layouts, tile shapes,
        and alignment to determine if the configuration is supported.

        Args:
            ab_dtype (Type[cutlass.Numeric]): Data type of operands A and B.
            sf_dtype (Type[cutlass.Numeric]): Data type of the scale factors.
            sf_vec_size (int): Vector size of the scale factors.
            c_dtype (Type[cutlass.Numeric]): Data type of the output tensor C.
            mma_tiler_mn (Tuple[int, int]): The (M, N) shape of the MMA tiler.
            cluster_shape_mn (Tuple[int, int]): The (M, N) shape of the CTA
                cluster.
            m (cutlass.Int64): The M dimension of the GEMM problem.
            n (cutlass.Int64): The N dimension of the GEMM problem.
            k (cutlass.Int64): The K dimension of the GEMM problem.
            l (cutlass.Int64): The batch dimension (L) of the GEMM problem.
            a_major (str): The major layout of tensor A ('k' or 'm').
            b_major (str): The major layout of tensor B ('k' or 'n').
            c_major (str): The major layout of tensor C ('n' or 'm').

        Returns:
            bool: True if the configuration is supported, False otherwise.
        """
        can_implement = True
        # Skip unsupported types
        if not cls.is_valid_dtypes_and_scale_factor_vec_size(
            ab_dtype, sf_dtype, sf_vec_size, c_dtype
        ):
            can_implement = False
        # Skip unsupported layouts
        if not cls.is_valid_layouts(ab_dtype, c_dtype, a_major, b_major, c_major):
            can_implement = False
        # Skip invalid mma tile shape and cluster shape
        if not cls.is_valid_mma_tiler_and_cluster_shape(mma_tiler_mn, cluster_shape_mn):
            can_implement = False
        # Skip illegal problem shape for load/store alignment
        if not cls.is_valid_tensor_alignment(
            m, n, k, l, ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            can_implement = False

        # Narrow N tiles (8/16/32) read their tokens' scale factors out of the
        # 128-wide SFB tile through the sub-tile addressing in the mainloop:
        # the S2T copy's smem source moves by whole token rows inside the
        # first 32-token group and the MMA's SFB TMEM address moves by one
        # column per 32 tokens. tcgen05.mma faults with a misaligned address
        # when that column shift is odd (offsets 1 and 3 fault; the 64-wide
        # tile's two-column shift is fine), so every sub-tile has to stay in
        # the first 32-token group: the kernel-N extent is limited to 32 and
        # the SFB tile is never multicast. Wider N takes the 64-wide or
        # 128-wide tiles. narrow_tile_ok holds the 32-column bound, shared with
        # the mm_mxfp8 / mm_fp4 runners' pre-launch re-check of replayed tactics.
        if not cls.narrow_tile_ok(mma_tiler_mn[1], n) or (
            mma_tiler_mn[1] < 64 and cluster_shape_mn[1] > 1
        ):
            can_implement = False
        return can_implement

    @cute.jit
    def push_wrapper(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        sf_m: cutlass.Int64,
        sf_n: cutlass.Int64,
        sf_k: cutlass.Int64,
        l: cutlass.Constexpr,
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        alpha_tensor: cute.Tensor,
        p_s0l: cute.Pointer,
        p_s0p: cute.Pointer,
        p_s1l: cute.Pointer,
        p_s1p: cute.Pointer,
        p_s2l: cute.Pointer,
        p_s2p: cute.Pointer,
        p_flag: cute.Pointer,
        max_active_clusters: cutlass.Constexpr,
        current_stream,
        swap_ab: cutlass.Constexpr = False,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """The common wrapper (MXFP8 path) plus 6 Lamport slot tensors with C's layout and the flag tensor."""
        m = cute.size(mA, mode=[0])
        k = cute.size(mA, mode=[1])
        n = cute.size(mB, mode=[0])
        a_tensor = cute.make_tensor(
            mA.iterator, layout=cute.make_ordered_layout((m, k, l), order=(1, 0, 2))
        )
        b_tensor = cute.make_tensor(
            mB.iterator, layout=cute.make_ordered_layout((n, k, l), order=(1, 0, 2))
        )
        if cutlass.const_expr(swap_ab):
            c_order = (0, 1, 2)
        else:
            c_order = (1, 0, 2)
        c_tensor = cute.make_tensor(
            mC.iterator, layout=cute.make_ordered_layout((m, n, l), order=c_order)
        )
        slots = [
            cute.make_tensor(p, layout=cute.make_ordered_layout((m, n, l), order=c_order))
            for p in (p_s0l, p_s0p, p_s1l, p_s1p, p_s2l, p_s2p)
        ]
        flag_tensor = cute.make_tensor(p_flag, layout=cute.make_layout(1))
        sfa_tensor = cute.make_tensor(
            a_sf_ptr,
            layout=cute.make_ordered_layout((32, 4, sf_m, 4, sf_k, l), order=(2, 1, 4, 0, 3, 5)),
        )
        sfb_tensor = cute.make_tensor(
            b_sf_ptr,
            layout=cute.make_ordered_layout((32, 4, sf_n, 4, sf_k, l), order=(2, 1, 4, 0, 3, 5)),
        )
        self(
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            c_tensor,
            alpha_tensor,
            max_active_clusters,
            current_stream,
            slots[0],
            slots[1],
            slots[2],
            slots[3],
            slots[4],
            slots[5],
            flag_tensor,
            epilogue_op,
        )


# ---------------------------------------------------------------------------------------------------------------------
# The launcher the TP2 modeling uses (OPROJ_PUSH_GEMM).
# One compiled kernel per pinned tactic (ops/gemm/mxfp8_cute_pinned_tactics; non-split-K only); the 6 Lamport slot
# pointers (3 rotating slots x local / peer section of this rank) and the flag address come from the AUTO all-reduce's
# workspace and are plain kernel params (graph-capture-safe: the workspace never moves; the slot rotates on the device).
# ---------------------------------------------------------------------------------------------------------------------


class _DeviceBytes:
    """``nbytes`` bytes of device memory at ``ptr`` (on the current device), as a CUDA array."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


def read_device_i64(addr: int) -> int:
    """The int64 at a device address of the current device (the Lamport workspace keeps its comm size in device
    memory). Set-up only: it synchronizes, so never inside a CUDA graph."""
    return int(_torch.as_tensor(_DeviceBytes(addr, 8), device="cuda").view(_torch.int64).item())


class LamportPushGemmLauncher:
    NR = 2

    def __init__(
        self, workspace: _torch.Tensor, rank: int, hidden: int, k: int, peek_i64=read_device_i64
    ):
        """``workspace``: get_allreduce_workspace(mapping) (int64 device pointers); ``peek_i64``: reads 8 bytes at a
        device address (the comm size lives in device memory)."""
        self.rank, self.hidden, self.k = rank, hidden, k
        self.wsp = [int(v) for v in workspace.cpu().tolist()]
        self.comm_size = int(peek_i64(self.wsp[self.NR * 3 + 1] + 8))
        self.flag_addr = self.wsp[self.NR * 3] + 8  # int index 2 = the Lamport flag
        self._compiled = {}
        self._alpha = {}

    def slots(self, rows: int):
        base = self.rank * rows * self.hidden * 2
        return [
            self.wsp[2 * self.NR + d] + s * self.comm_size + base
            for s in range(3)
            for d in (self.rank, 1 - self.rank)
        ]

    def compile(self, tactic):
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import make_ptr
        from flashinfer.cute_dsl.utils import get_max_active_clusters

        mma_tiler_mn, cluster_mn, swap_ab, use_prefetch, split_k = tactic
        assert split_k == 1
        key = (tuple(mma_tiler_mn), tuple(cluster_mn), bool(swap_ab), bool(use_prefetch))
        if key in self._compiled:
            return self._compiled[key]
        gemm = LamportPushGemmKernel(
            32, tuple(mma_tiler_mn), tuple(cluster_mn), bool(use_prefetch), True
        )
        mac = get_max_active_clusters(cluster_mn[0] * cluster_mn[1])
        sm, sk, sn = cute.sym_int(), cute.sym_int(), cute.sym_int()
        a_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Float8E4M3FN, (sm, sk), stride_order=(1, 0), assumed_align=16
        )
        b_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Float8E4M3FN, (sn, sk), stride_order=(1, 0), assumed_align=16
        )
        if swap_ab:
            c_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.BFloat16, (sn, sm), stride_order=(0, 1), assumed_align=16
            )
        else:
            c_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.BFloat16, (sm, sn), stride_order=(1, 0), assumed_align=16
            )
        sfp = [make_ptr(cutlass.Float8E8M0FNU, 16, cute.AddressSpace.gmem, 16) for _ in range(2)]
        cps = [make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, 16) for _ in range(6)]
        fp = make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, 4)
        alpha_fake = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (1,), assumed_align=4)
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        # The kernel class comes from FlashInfer, so its version is part of the shipped object's key.
        fn = cute_aot.compile_or_load(
            f"lamport_push_gemm {key} mac={mac} flashinfer={_flashinfer_version()}",
            lambda: cute.compile(
                gemm.push_wrapper,
                a_fake,
                b_fake,
                c_fake,
                1,
                1,
                1,
                1,
                sfp[0],
                sfp[1],
                alpha_fake,
                *cps,
                fp,
                mac,
                stream_fake,
                swap_ab,
                options="--opt-level 2 --enable-tvm-ffi",
            ),
            sources=_aot_sources(),
            tvm_ffi=True,
        )
        self._compiled[key] = fn
        return fn

    def run(
        self, tactic, act, act_scale, w, w_scale, dummy_out, alpha=None, slots=None, flag_addr=None
    ):
        """``act @ w^T`` (MXFP8, the pinned tactic) stored into this rank's section of the current Lamport slot on
        both ranks, with the stock push's -0 word rule. ``dummy_out``: a [rows, hidden] bf16 tensor (shape only).
        ``slots`` / ``flag_addr``: six [rows, hidden] bf16 destinations and an int32 slot selector in place of the
        workspace's (the prebuild's byte check stores into scratch buffers this way)."""
        fn = self.compile(tactic)
        rows = act.shape[0]
        swap_ab = tactic[2]
        if swap_ab:
            km, kn, ka, kb, kasf, kbsf = self.hidden, rows, w, act, w_scale, act_scale
        else:
            km, kn, ka, kb, kasf, kbsf = rows, self.hidden, act, w, act_scale, w_scale
        sf_m, sf_n, sf_k = (km + 127) // 128, (kn + 127) // 128, (self.k // 32 + 3) // 4
        out = (
            dummy_out.as_strided(dummy_out.shape, (1, dummy_out.shape[0])) if swap_ab else dummy_out
        )
        if alpha is None:
            alpha = self._alpha.get(act.device)
            if alpha is None:
                alpha = self._alpha[act.device] = _torch.ones(
                    1, dtype=_torch.float32, device=act.device
                )
        fn(
            ka,
            kb,
            out,
            sf_m,
            sf_n,
            sf_k,
            kasf.data_ptr(),
            kbsf.data_ptr(),
            alpha,
            *(self.slots(rows) if slots is None else slots),
            self.flag_addr if flag_addr is None else flag_addr,
        )


def _flashinfer_version() -> str:
    import importlib.metadata as md

    return md.version("flashinfer-python")


def _aot_sources():
    """The Python files this module's CuTe DSL kernels are traced from (the target package and the shared Blackwell
    kernel helpers): the ``sources`` of their shipped AOT objects."""
    target = Path(__file__).resolve().parents[1]
    shared = cute_aot.BLACKWELL_DIR
    return cute_aot.py_files(str(target), str(shared))
