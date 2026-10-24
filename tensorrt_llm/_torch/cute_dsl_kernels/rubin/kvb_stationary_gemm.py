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

# ruff: noqa: E501,E731,E741

"""B-stationary persistent block-scaled GEMM for a short K (Kimi K3 MLA kv_b decompression).

``C[M, N] = A[M, K] x B[N, K]^T`` with ``K`` a small multiple of the 128-wide K tile (512 for kv_b), FP8 (E4M3)
operands with UE8M0 block scales and an FP8 (E4M3) output. The dense persistent kernel
(``Sm107BlockScaledPersistentDenseGemmKernel``) stages A and B in lockstep, so each 128 x 256 output tile
re-reads a 64 KB A block and a 128 KB B block through the TMA unit: with K = 512 that is 192 KB of ingress for
33.5 MFLOP, and the per-SM load port, not the tensor cores, sets the speed. Its epilogue then
adds the FP8 conversion and the TMA stores to the same critical path.

This kernel changes three things and keeps the math (same MMA instructions in the same K order into the same
accumulator layout, same BF16 rounding / alpha multiply / E4M3 conversion), so its output is bitwise equal to
the dense kernel's:

* B is resident. The B tiles of ``n_res`` N blocks (all K tiles of a 256-wide block plus their scale factors)
  stay in shared memory; only A and its scale factors stream through a ring. Every cluster owns a contiguous
  range of steps (one M block against the ``n_res`` resident N blocks) in N-group-major order, so it reloads B
  at most a handful of times, and every A block is loaded once per ``n_res`` output tiles.
* The epilogue converts in registers (TMEM -> BF16 -> BF16 alpha multiply -> E4M3 in 5 instructions per 4
  elements) and writes each 32-column sub-tile into its own shared-memory buffer; a dedicated store warp issues
  the TMA stores and frees the buffers a few stores behind. The epilogue warps never wait for a TMA store and
  no barrier couples the warps: buffers are handed over with per-buffer mbarriers.
* The A ring is as deep as the shared memory left by B and the C buffers allows (it has to cover the TMA
  latency); with ``n_res`` = 2 and K = 512 that is 9 stages of the 2-CTA kernel.

Supported: ``cta_tile_n == 256``, no B reuse (one 128-row MMA per K block), cluster = the MMA's CTA group
((1, 1) for 128-row tiles, (2, 1) for 2-CTA MMA), static contiguous step ranges (no tile scheduler), FP8
output. ``k_tiles`` (K / 128) and ``n_res`` are compile-time constants of the kernel object.
"""

from typing import Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.rubin_helpers as sm107_utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.nvgpu.tcgen05.mma import CollectorOp
from cutlass.cutlass_dsl import Int32, T, dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils.gemm.sm100 import (
    epilogue_smem_copy_and_partition,
    epilogue_tmem_copy_and_partition,
    transform_partitioned_tensor_layout,
)

from .dense_blockscaled_gemm_persistent import Sm107BlockScaledPersistentDenseGemmKernel


@dsl_user_op
def _bf16x2_splat(value: cutlass.Float32, *, loc=None, ip=None) -> Int32:
    """``{bf16(value), bf16(value)}`` packed in one register (round to nearest even)."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Float32(value).ir_value(loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $1, $1;",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _scale_bf16x4_to_e4m3x4(
    values01: Int32, values23: Int32, scale: Int32, *, loc=None, ip=None
) -> Int32:
    """Four BF16 values (two packed registers) times a packed BF16 scale, converted to four E4M3 bytes.

    The BF16 multiply and the saturating E4M3 conversion straight from BF16 are the dense epilogue's math (BF16
    rounding of the accumulator, BF16 alpha multiply, clamp to +-448, E4M3 round to nearest even) in 5 instructions
    per 4 elements instead of unpacking to FP32 and clamping with min/max.
    """
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Int32(values01).ir_value(loc=loc, ip=ip),
                Int32(values23).ir_value(loc=loc, ip=ip),
                Int32(scale).ir_value(loc=loc, ip=ip),
            ],
            "{\n"
            "  .reg .b32 scaled01;\n"
            "  .reg .b32 scaled23;\n"
            "  .reg .b16 lo;\n"
            "  .reg .b16 hi;\n"
            "  mul.rn.bf16x2 scaled01, $1, $3;\n"
            "  mul.rn.bf16x2 scaled23, $2, $3;\n"
            "  cvt.rn.satfinite.e4m3x2.bf16x2 lo, scaled01;\n"
            "  cvt.rn.satfinite.e4m3x2.bf16x2 hi, scaled23;\n"
            "  mov.b32 $0, {lo, hi};\n"
            "}",
            "=r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


# Shared memory kept for the pipeline barriers, the TMEM pointer and the 1 KB buffer alignment.
_SMEM_RESERVE_BYTES = 2048
# Most A stages worth building (the A ring only has to cover the TMA latency).
_MAX_A_STAGES = 16
# Stores the store warp keeps in flight before it frees the oldest C buffer.
_STORE_LAG = 2


class Sm107KvBStationaryGemmKernel(Sm107BlockScaledPersistentDenseGemmKernel):
    """Persistent block-scaled GEMM with a resident B tile and a streamed A ring (see the module docstring)."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_inst_shape: Tuple[int, int, int],
        mma_tiler: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        k_tiles: int = 4,
        n_res: int = 2,
        num_a_stage: int = 0,
    ):
        """
        :param k_tiles: K / ``mma_tiler[2]``: the number of K tiles of one B tile held in shared memory.
        :param n_res: N blocks (of 256) whose B tiles are resident at once. Every A tile (one M block, streamed
            through the ring) is multiplied with all of them before its stages are released, so A is loaded once
            per ``n_res`` output tiles. The last group of a problem whose N-block count is not a multiple of
            ``n_res`` runs its missing blocks on zero-filled B (TMA out-of-bounds) and stores nothing for them.
        :param num_a_stage: A-ring depth in K tiles (0 = as many as the shared memory left by B and C allows). At
            least ``k_tiles`` when ``n_res`` > 1 (the A block stays resident until its last output tile), else 2.
        """
        super().__init__(
            sf_vec_size,
            mma_inst_shape,
            mma_tiler,
            cluster_shape_mn,
            0,
            raster_order="m",
            use_pdl=False,
        )
        if mma_tiler[1] != 256:
            raise ValueError(f"cta tile N must be 256, got {mma_tiler[1]}")
        if self.enable_breuse:
            raise ValueError("B reuse (two 128-row MMAs per K block) is not supported")
        if tuple(cluster_shape_mn) != (2 if self.use_2cta_instrs else 1, 1):
            raise ValueError(f"cluster must be the MMA CTA group, got {cluster_shape_mn}")
        if k_tiles < 1:
            raise ValueError(f"k_tiles must be >= 1, got {k_tiles}")
        if n_res < 1:
            raise ValueError(f"n_res must be >= 1, got {n_res}")
        self.k_tiles = k_tiles
        self.n_res = n_res
        self.num_a_stage_request = num_a_stage
        # A seventh warp issues the epilogue's TMA stores (see ``_epilogue``).
        self.store_warp_id = 6
        self.threads_per_cta = self.threads_per_warp * 7

    def _compute_stages(
        self,
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        epi_tile: cute.Tile,
        c_dtype: Type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        smem_capacity: int,
        occupancy: int,
        with_breuse: bool,
    ) -> Tuple[int, int, int]:
        """(ACC stages, A ring stages, C stages). B and SFB take ``n_res * k_tiles`` resident stages off the top."""
        num_acc_stage = 2
        a_one = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, a_dtype, 1)
        b_one = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, b_dtype, 1)
        sfa_one = blockscaled_utils.make_smem_layout_sfa(tiled_mma, mma_tiler_mnk, sf_vec_size, 1)
        sfb_one = blockscaled_utils.make_smem_layout_sfb(tiled_mma, mma_tiler_mnk, sf_vec_size, 1)
        c_one = sm100_utils.make_smem_layout_epi(c_dtype, c_layout, epi_tile, 1)
        a_stage_bytes = cute.size_in_bytes(a_dtype, a_one) + cute.size_in_bytes(sf_dtype, sfa_one)
        b_stage_bytes = cute.size_in_bytes(b_dtype, b_one) + cute.size_in_bytes(sf_dtype, sfb_one)
        c_stage_bytes = cute.size_in_bytes(c_dtype, c_one)

        # One C buffer per epilogue sub-tile: the whole output tile is staged in shared memory.
        num_c_stage = (self.cta_tile_shape_mnk[0] * self.cta_tile_shape_mnk[1]) // (
            cute.size(epi_tile[0]) * cute.size(epi_tile[1])
        )
        free = (
            smem_capacity // occupancy
            - _SMEM_RESERVE_BYTES
            - self.n_res * self.k_tiles * b_stage_bytes
            - num_c_stage * c_stage_bytes
        )
        num_a_stage = min(free // a_stage_bytes, _MAX_A_STAGES)
        if self.num_a_stage_request > 0:
            num_a_stage = min(num_a_stage, self.num_a_stage_request)
        # With n_res > 1 a whole A block (k_tiles stages) stays in the ring until its last output tile has been
        # multiplied, so a shallower ring would deadlock the TMA warp against the MMA warp.
        min_a_stage = self.k_tiles if self.n_res > 1 else 2
        if num_a_stage < min_a_stage:
            raise ValueError(
                f"A ring of {num_a_stage} stages is shorter than the {min_a_stage} this kernel needs (n_res "
                f"{self.n_res}, k_tiles {self.k_tiles}); shared memory leaves {free} B for the ring after "
                f"{self.n_res * self.k_tiles} resident B stages ({self.n_res * self.k_tiles * b_stage_bytes} B), "
                f"a stage is {a_stage_bytes} B"
            )
        return num_acc_stage, num_a_stage, num_c_stage

    def _setup_attributes(self):
        super()._setup_attributes()
        # The parent staged A, B, SFA and SFB with the A-ring depth (self.num_ab_stage). B and SFB are resident:
        # one stage per K tile of each of the n_res N blocks.
        self.num_a_stage = self.num_ab_stage
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
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.n_res * self.k_tiles
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, self.sf_vec_size, self.n_res * self.k_tiles
        )

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        alpha: cute.Tensor,
        layouts: cutlass.Constexpr[Tuple[OperandMajorMode, OperandMajorMode, utils.LayoutEnum]],
        problem_mnkl: Tuple[int, int, int, int],
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,  # interface of the dense wrapper; the epilogue is fixed
        c_ld: cutlass.Int64 = cutlass.Int64(0),
        split_k: cutlass.Constexpr = 1,
    ):
        """Set up the TMA atoms, the shared storage and the launch (same interface as the dense kernel)."""
        assert split_k == 1, "split-K is not supported"
        self.a_dtype: Type[cutlass.Numeric] = a_ptr.value_type
        self.b_dtype: Type[cutlass.Numeric] = b_ptr.value_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa_ptr.value_type
        self.c_dtype: Type[cutlass.Numeric] = c_ptr.value_type
        assert self.c_dtype == cutlass.Float8E4M3FN, "FP8 output only"
        assert self.sf_dtype == cutlass.Float8E8M0FNU
        assert self.a_dtype == cutlass.Float8E4M3FN
        assert self.b_dtype == cutlass.Float8E4M3FN

        m, n, k, l = problem_mnkl
        # One batch only; K must equal ``k_tiles`` * the K tile (the runner derives ``k_tiles`` from K and keys its
        # compile cache on it); ``epilogue_op`` is not applied (the epilogue is fixed, see ``_scale_bf16x4_to_e4m3x4``).
        assert l == 1, "batched problems are not supported"
        self.a_major_mode, self.b_major_mode, self.c_layout = layouts
        self._setup_attributes()

        a_layout = cute.make_ordered_layout((cute.assume(m, 32), k, l), order=(1, 0, 2))
        b_layout = cute.make_ordered_layout((cute.assume(n, 32), k, l), order=(1, 0, 2))
        if cutlass.const_expr(self.c_layout == utils.LayoutEnum.ROW_MAJOR):
            actual_c_ld = c_ld + (n - c_ld) * (c_ld == 0)
            c_out_layout = cute.make_layout(
                (m, cute.assume(n, 32), l), stride=(actual_c_ld, 1, m * actual_c_ld)
            )
        else:
            actual_c_ld = c_ld + (m - c_ld) * (c_ld == 0)
            c_out_layout = cute.make_layout(
                (cute.assume(m, 32), n, l), stride=(1, actual_c_ld, n * actual_c_ld)
            )
        a_tensor = cute.make_tensor(a_ptr, a_layout)
        b_tensor = cute.make_tensor(b_ptr, b_layout)
        c_tensor = cute.make_tensor(c_ptr, c_out_layout)
        sfa_tensor = cute.make_tensor(
            sfa_ptr, blockscaled_utils.tile_atom_to_shape_SF(a_tensor.shape, self.sf_vec_size)
        )
        sfb_tensor = cute.make_tensor(
            sfb_ptr, blockscaled_utils.tile_atom_to_shape_SF(b_tensor.shape, self.sf_vec_size)
        )

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
        # For 2-CTA MMA the scale factors of B are replicated across the CTA pair.
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
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, a_tensor, a_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b_tensor, b_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape
        )
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa_tensor,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma.thr_id)
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb_tensor,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        # Bytes one A-ring stage and the whole resident B tile expect from the TMA unit (both CTAs of a pair).
        self.num_tma_load_bytes_a = (a_copy_size + sfa_copy_size) * atom_thr_size
        self.num_tma_load_bytes_b = (
            self.n_res * self.k_tiles * (b_copy_size + sfb_copy_size) * atom_thr_size
        )

        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c_tensor, epi_smem_layout, self.epi_tile
        )

        # Persistent grid: one cluster per tile range. Tiles are counted in MMA tiles (one per CTA pair for 2-CTA MMA).
        c_tile = cute.slice_(self.mma_tiler, (None, None, 0))
        gc = cute.zipped_divide(c_tensor, tiler=c_tile)
        num_tiles_mnl = gc[(0, (None, None, None))].shape
        # A step is one M block times n_res N blocks (the work between two A loads); CTAs own contiguous step ranges.
        total_steps = (
            (cute.size(num_tiles_mnl, mode=[1]) + self.n_res - 1)
            // self.n_res
            * cute.size(num_tiles_mnl, mode=[0])
        )
        num_clusters = cutlass.min(cutlass.Int32(max_active_clusters), total_steps)

        self.buffer_align_bytes = 1024

        @cute.struct
        class SharedStorage:
            a_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_a_stage]
            a_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_a_stage]
            b_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
            b_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            c_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_c_stage]
            c_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_c_stage]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[self.c_dtype, cute.cosize(self.c_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, A-ring STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, K tile): resident
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

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
            self.tCtSFA_layout,
            self.tCtSFB_layout,
            self.c_smem_layout_staged,
            self.epi_tile,
            alpha,
        ).launch(
            grid=(*self.cluster_shape_mn, num_clusters),
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )
        return

    @cute.jit
    def _epilogue(
        self,
        epi_tidx: Int32,
        tCtAcc_base: cute.Tensor,
        sC: cute.Tensor,
        tCgC_base: cute.Tensor,
        epi_tile: cute.Tile,
        alpha2: Int32,
        acc_consumer_state: pipeline.PipelineState,
        acc_pipeline: pipeline.PipelineAsync,
        c_pipe: pipeline.PipelineAsync,
        c_producer_state: pipeline.PipelineState,
    ) -> Tuple[pipeline.PipelineState, pipeline.PipelineState]:
        """One output tile without any cross-warp synchronisation: the C buffers are handed to the store warp.

        Each epilogue warp converts its 32 TMEM lanes (TMEM -> BF16 -> alpha -> E4M3 in registers, the TMEM load of
        sub-tile i + 1 issued before sub-tile i is converted) and writes every sub-tile into its own shared-memory
        buffer; after the buffer is complete (the four warps arrive on the pipeline's full barrier) the store warp
        issues the TMA store and, once the store has read the buffer, frees it. One C buffer per sub-tile.
        """
        tCgC = transform_partitioned_tensor_layout(tCgC_base)
        tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)
        tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = epilogue_tmem_copy_and_partition(
            self, epi_tidx, tCtAcc, tCgC, epi_tile, self.use_2cta_instrs
        )
        tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
        tiled_copy_r2s, tRS_rC, tRS_sC = epilogue_smem_copy_and_partition(
            self, tiled_copy_t2r, tTR_rC, epi_tidx, sC
        )
        tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        sub_tiles = cute.size(tTR_tAcc.shape, mode=[3])
        num_elems = cute.size(tTR_rAcc)

        acc_pipeline.consumer_wait(acc_consumer_state)
        acc_bufs = [tTR_rAcc, cute.make_rmem_tensor(tTR_rAcc.shape, self.acc_dtype)]
        c_words = cute.recast_tensor(tTR_rC, cutlass.Int32)
        cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 0)], acc_bufs[0])
        for i in cutlass.range_constexpr(sub_tiles):
            if cutlass.const_expr(i + 1 < sub_tiles):
                cute.copy(
                    tiled_copy_t2r, tTR_tAcc[(None, None, None, i + 1)], acc_bufs[(i + 1) % 2]
                )
            acc_flat = cute.make_tensor(acc_bufs[i % 2].iterator, cute.make_layout((num_elems,)))
            rounded = cute.make_rmem_tensor((num_elems,), cutlass.BFloat16)
            rounded.store(acc_flat.load().to(cutlass.BFloat16))
            packed = cute.recast_tensor(rounded, cutlass.Int32)
            for g in cutlass.range_constexpr(num_elems // 4):
                c_words[g] = _scale_bf16x4_to_e4m3x4(packed[2 * g], packed[2 * g + 1], alpha2)
            # The buffer must have been read by the previous tile's TMA store before it is overwritten.
            c_pipe.producer_acquire(c_producer_state)
            cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, i)])
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                c_pipe.producer_commit(c_producer_state)
            c_producer_state.advance()

        # Every TMEM read of this warp has been consumed: hand the accumulator back to the MMA warp.
        with cute.arch.elect_one():
            acc_pipeline.consumer_release(acc_consumer_state)
        acc_consumer_state.advance()
        return acc_consumer_state, c_producer_state

    @cute.jit
    def _store_warp(
        self,
        tma_atom_c: cute.CopyAtom,
        sC: cute.Tensor,
        tCgC_base: cute.Tensor,
        epi_tile: cute.Tile,
        c_pipe: pipeline.PipelineAsync,
        step_begin: Int32,
        step_end: Int32,
        m_tiles: Int32,
    ):
        """Issue the TMA store of every finished C buffer and free the buffers a few stores behind.

        Stores are committed one bulk group each; after committing store j, ``cp.async.bulk.wait_group.read``
        with ``lag`` (``_STORE_LAG``, at most one less than the number of buffers) pending guarantees stores
        <= j - ``lag`` have read their buffers, which are then released to the epilogue warps.
        """
        tCgC = transform_partitioned_tensor_layout(tCgC_base)
        tCgC_epi = cute.flat_divide(tCgC, epi_tile)
        bSG_sC, bSG_gC_partitioned = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            cute.group_modes(sC, 0, 2),
            cute.group_modes(tCgC_epi, 0, 2),
        )
        sub_tiles = self.num_c_stage
        lag = min(_STORE_LAG, sub_tiles - 1)
        c_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, sub_tiles
        )
        for step in cutlass.range(step_begin, step_end, 1, unroll=1):
            g_idx = step // m_tiles
            m_idx = step - g_idx * m_tiles
            for j in cutlass.range_constexpr(self.n_res):
                bSG_gC = bSG_gC_partitioned[
                    (None, None, None, m_idx, g_idx * self.n_res + j, cutlass.Int32(0))
                ]
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
                for i in cutlass.range_constexpr(sub_tiles):
                    c_pipe.consumer_wait(c_consumer_state)
                    cute.copy(tma_atom_c, bSG_sC[(None, i)], bSG_gC[(None, i)])
                    cute.arch.cp_async_bulk_commit_group()
                    c_consumer_state.advance()
                    cute.arch.cp_async_bulk_wait_group(lag, read=True)
                    if cutlass.const_expr(i >= lag):
                        with cute.arch.elect_one():
                            c_pipe.sync_object_empty.arrive(i - lag, None)
                    else:
                        # The buffers freed here belong to the previous tile (none before the first one).
                        if cutlass.const_expr(j > 0):
                            with cute.arch.elect_one():
                                c_pipe.sync_object_empty.arrive(sub_tiles + i - lag, None)
                        else:
                            if step > step_begin:
                                with cute.arch.elect_one():
                                    c_pipe.sync_object_empty.arrive(sub_tiles + i - lag, None)
        cute.arch.cp_async_bulk_wait_group(0, read=False)

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
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        tCtSFA_layout: cute.Layout,
        tCtSFB_layout: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        alpha: cute.Tensor,
    ):
        """Device kernel: TMA warp, MMA warp, epilogue warps and store warp, each walking the CTA's step range."""
        alpha_value = alpha[0].to(cutlass.BFloat16)
        alpha2 = _bf16x2_splat(alpha_value.to(cutlass.Float32))

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2
        bidx, bidy, bidz = cute.arch.block_idx()
        num_clusters = cute.arch.grid_dim()[2]
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Shared storage and barriers: A ring (full/empty per stage), resident B (one stage), accumulators.
        #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # The A and B pipelines have one producer thread (the TMA warp) and, per CTA, one consumer arrival: the
        # (leader) MMA warp's commit.
        a_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.a_full_mbar_ptr.data_ptr(),
            num_stages=self.num_a_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.num_tma_load_bytes_a,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        b_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.b_full_mbar_ptr.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.num_tma_load_bytes_b,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        num_acc_consumer_threads = len(self.epilog_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_acc_consumer_threads
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Epilogue C buffers: the epilogue warps (one arrival per warp) fill a buffer, the store warp
        # (one arrival) frees it once its TMA store has read it.
        c_async_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.c_full_mbar_ptr.data_ptr(),
            num_stages=self.num_c_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len(self.epilog_warp_id)
            ),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            defer_sync=True,
        )

        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
            arch=self.arch,
        )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sC = storage.sC.get_tensor(c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(use_2cta_instrs):
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
        # Global tensors per tile and the MMA/TMA partitions (as in the dense kernel).
        #
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl, cute.slice_(self.mma_tiler_sfb, (0, None, None)), (None, None, None)
        )
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        m_tiles = cute.size(gA_mkl, mode=[2])
        n_tiles = cute.size(gB_nkl, mode=[2])

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
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
        tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        #
        # This cluster's contiguous range of steps. A step is one M block against the n_res N blocks of a group; steps
        # are ordered group-major (step = g * m_tiles + m), so consecutive steps share the resident B tiles. The range
        # crosses at most a few groups (num_runs); the last group may run past n_tiles (those tiles are clipped).
        #
        n_groups = (n_tiles + self.n_res - 1) // self.n_res
        total_steps = m_tiles * n_groups
        step_begin = bidz * total_steps // num_clusters
        step_end = (bidz + 1) * total_steps // num_clusters
        g_first = step_begin // m_tiles
        num_runs = (step_end - 1) // m_tiles - g_first + 1

        #
        # TMA warp
        #
        if warp_idx == self.tma_warp_id:
            a_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_a_stage
            )
            b_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            for run in cutlass.range(num_runs, unroll=1):
                g_idx = g_first + run
                # The B tiles of the whole group (every K tile and its scale factors of the n_res N blocks) land
                # under one barrier; the MMA warp releases them when the last step of the previous group has been
                # multiplied.
                b_pipeline.producer_acquire(b_producer_state)
                b_bar = b_pipeline.producer_get_barrier(b_producer_state)
                for j in cutlass.range_constexpr(self.n_res):
                    tBgB_n = tBgB[(None, g_idx * self.n_res + j, None, 0)]
                    tBgSFB_n = tBgSFB[(None, g_idx * self.n_res + j, None, 0)]
                    for k_tile in cutlass.range(self.k_tiles, unroll_full=True):
                        cute.copy(
                            tma_atom_b,
                            tBgB_n[(None, k_tile)],
                            tBsB[(None, j * self.k_tiles + k_tile)],
                            tma_bar_ptr=b_bar,
                            mcast_mask=b_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_sfb,
                            tBgSFB_n[(None, k_tile)],
                            tBsSFB[(None, j * self.k_tiles + k_tile)],
                            tma_bar_ptr=b_bar,
                            mcast_mask=sfb_full_mcast_mask,
                        )
                b_producer_state.advance()

                run_begin = cutlass.max(step_begin, g_idx * m_tiles)
                run_end = cutlass.min(step_end, (g_idx + 1) * m_tiles)
                for step in cutlass.range(run_begin, run_end, 1, unroll=1):
                    m_idx = step - g_idx * m_tiles
                    tAgA_m = tAgA[(None, m_idx, None, 0)]
                    tAgSFA_m = tAgSFA[(None, m_idx, None, 0)]
                    for k_tile in cutlass.range(self.k_tiles, unroll=1):
                        a_pipeline.producer_acquire(a_producer_state)
                        a_bar = a_pipeline.producer_get_barrier(a_producer_state)
                        cute.copy(
                            tma_atom_a,
                            tAgA_m[(None, k_tile)],
                            tAsA[(None, a_producer_state.index)],
                            tma_bar_ptr=a_bar,
                            mcast_mask=a_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_sfa,
                            tAgSFA_m[(None, k_tile)],
                            tAsSFA[(None, a_producer_state.index)],
                            tma_bar_ptr=a_bar,
                            mcast_mask=sfa_full_mcast_mask,
                        )
                        a_producer_state.advance()
            a_pipeline.producer_tail(a_producer_state)
            b_pipeline.producer_tail(b_producer_state)

        #
        # MMA warp
        #
        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols, dtype=self.sf_dtype
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
            sfa_s2t_bundle = self._mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            sfb_s2t_bundle = self._mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

            a_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_a_stage
            )
            b_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            num_kblocks = cute.size(tCrA, mode=[2])

            for run in cutlass.range(num_runs, unroll=1):
                g_idx = g_first + run
                if is_leader_cta:
                    b_pipeline.consumer_wait(b_consumer_state)
                run_begin = cutlass.max(step_begin, g_idx * m_tiles)
                run_end = cutlass.min(step_end, (g_idx + 1) * m_tiles)
                for step in cutlass.range(run_begin, run_end, 1, unroll=1):
                    # The ring stages of this step's A block stay held until the last of its n_res tiles has been
                    # multiplied: the first tile waits for them, the others re-read them.
                    a_step_state = a_consumer_state.clone()
                    for j in cutlass.range_constexpr(self.n_res):
                        tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]
                        if is_leader_cta:
                            acc_pipeline.producer_acquire(acc_producer_state)
                        a_state = a_step_state.clone()
                        for k_tile in cutlass.range(self.k_tiles, unroll=1):
                            if is_leader_cta:
                                if cutlass.const_expr(j == 0):
                                    a_pipeline.consumer_wait(a_state)
                                # SFA of this ring stage and SFB of this K tile into tensor memory.
                                cute.copy(
                                    sfa_s2t_bundle.tiled_copy,
                                    sfa_s2t_bundle.sSF_compact[
                                        (None, None, None, None, a_state.index)
                                    ],
                                    sfa_s2t_bundle.tSF_compact,
                                )
                                cute.copy(
                                    sfb_s2t_bundle.tiled_copy,
                                    sfb_s2t_bundle.sSF_compact[
                                        (None, None, None, None, j * self.k_tiles + k_tile)
                                    ],
                                    sfb_s2t_bundle.tSF_compact,
                                )
                                for k_block in cutlass.range(num_kblocks, unroll_full=True):
                                    tiled_mma.set(
                                        tcgen05.Field.ACCUMULATE, k_tile != 0 or k_block != 0
                                    )
                                    cute.gemm(
                                        tiled_mma,
                                        tCtAcc,
                                        [
                                            tCrA[(None, None, k_block, a_state.index)],
                                            tCtSFA[(None, None, k_block)],
                                        ],
                                        [
                                            tCrB[(None, None, k_block, j * self.k_tiles + k_tile)],
                                            tCtSFB[(None, None, k_block)],
                                        ],
                                        tCtAcc,
                                    )
                                if cutlass.const_expr(j == self.n_res - 1):
                                    a_pipeline.consumer_release(a_state)
                            a_state.advance()
                        if is_leader_cta:
                            acc_pipeline.producer_commit(acc_producer_state)
                        acc_producer_state.advance()
                    # PipelineState.advance() takes no count in the DSL version of the image: step it k_tiles times.
                    for _ in cutlass.range_constexpr(self.k_tiles):
                        a_consumer_state.advance()
                # Every MMA of this group has been issued: free the B tiles once they complete.
                if is_leader_cta:
                    b_pipeline.consumer_release(b_consumer_state)
                b_consumer_state.advance()
            acc_pipeline.producer_tail(acc_producer_state)

        #
        # Epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            c_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_c_stage
            )
            # One accumulator per (step, resident N block), in the order the MMA warp produces them.
            for _ in cutlass.range(step_begin * self.n_res, step_end * self.n_res, 1, unroll=1):
                acc_consumer_state, c_producer_state = self._epilogue(
                    tidx,
                    tCtAcc,
                    sC,
                    tCgC,
                    epi_tile,
                    alpha2,
                    acc_consumer_state,
                    acc_pipeline,
                    c_async_pipeline,
                    c_producer_state,
                )
            # The epilogue warps share no per-tile barrier here: all of them must have issued their last tcgen05.ld
            # before warp 0 deallocates tensor memory (the dense epilogue syncs the warps after every sub-tile).
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.relinquish_alloc_permit()
            tmem.free(acc_tmem_ptr)

        #
        # Store warp
        #
        if warp_idx == self.store_warp_id:
            self._store_warp(
                tma_atom_c, sC, tCgC, epi_tile, c_async_pipeline, step_begin, step_end, m_tiles
            )
