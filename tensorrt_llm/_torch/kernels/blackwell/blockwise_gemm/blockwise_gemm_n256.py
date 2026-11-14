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
"""2-CTA 256x256-tile blockwise FP8 GEMM with the MXFP8 output epilogue.

``Sm100BlockwiseGemmMxfp8N256Kernel`` computes what ``Sm100BlockwiseGemmKernel(mxfp8_out=True)`` computes with a
128x128 CTA tile, bit for bit: every 128-deep k-block's MMA result is promoted into an FP32 accumulator with
``final = acc * (scale_a * scale_b) + final`` in k order, and the result is rounded to BF16 and quantized to E4M3 with
per-32-column UE8M0 scales in the swizzled 128x4 layout. Only the schedule differs:

- One 2-CTA MMA covers a 256x256 output tile (each CTA of the pair holds 128 rows x 256 columns), so every A/B byte
  loaded into shared memory feeds twice as many MACs as with a 256x128 tile.
- A k-block's 256-column accumulator takes 256 TMEM columns: two accumulator stages fill the 512 columns, and the
  final accumulator stays in registers. Eight promotion warps (two warpgroups, one per 128-column half) hold 128 FP32
  values per thread, the same per-thread state as the 128x128 kernel's accumulator-update warps.
- The promotion warps run the MXFP8 epilogue from registers at the end of each tile and store each warpgroup's
  128x128 E4M3 tile with TMA from a 128B-swizzled shared-memory tile, while the MMA warp starts the next tile.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import Float32, Int32, Int64, T, dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from .blockwise_gemm import Sm100BlockwiseGemmKernel


@dsl_user_op
def _mxfp8_store_global_b32(gmem_ptr, value: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [
            gmem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            Int32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mxfp8_store_shared_b128(
    smem_ptr, value0: Int32, value1: Int32, value2: Int32, value3: Int32, *, loc=None, ip=None
) -> None:
    llvm.inline_asm(
        None,
        [
            smem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            Int32(value0).ir_value(loc=loc, ip=ip),
            Int32(value1).ir_value(loc=loc, ip=ip),
            Int32(value2).ir_value(loc=loc, ip=ip),
            Int32(value3).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mxfp8_block_scale(block_max_bits: Int32, *, loc=None, ip=None) -> Int32:
    """UE8M0 code (0-255) of one 32-element block from the FP32 bits of its absolute maximum,
    exactly as ``cvt_warp_fp16_to_mxfp8`` derives it: ``rcp.approx.ftz(448)``, an FP32 product,
    then ``cvt.rp.satfinite`` to E8M0."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(block_max_bits).ir_value(loc=loc, ip=ip)],
            "{\n"
            "  .reg .f32 rcp448, sf;\n"
            "  .reg .b16 codes;\n"
            "  .reg .b32 wide;\n"
            "  rcp.approx.ftz.f32 rcp448, 0f43E00000;\n"
            "  mul.rn.f32 sf, $1, rcp448;\n"
            "  cvt.rp.satfinite.ue8m0x2.f32 codes, 0f00000000, sf;\n"
            "  cvt.u32.u16 wide, codes;\n"
            "  and.b32 $0, wide, 255;\n"
            "}",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _mxfp8_quant_multiplier(code: Int32, block_max_bits: Int32, *, loc=None, ip=None) -> Int32:
    """FP32 bits of ``block_max != 0 ? rcp.approx.ftz(float(E8M0 code)) : 0``."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(code).ir_value(loc=loc, ip=ip), Int32(block_max_bits).ir_value(loc=loc, ip=ip)],
            "{\n"
            "  .reg .b32 bits;\n"
            "  .reg .f32 scale, inv;\n"
            "  .reg .pred is_zero_code, is_zero_max;\n"
            "  shl.b32 bits, $1, 23;\n"
            "  setp.eq.u32 is_zero_code, $1, 0;\n"
            "  selp.b32 bits, 4194304, bits, is_zero_code;\n"
            "  mov.b32 scale, bits;\n"
            "  rcp.approx.ftz.f32 inv, scale;\n"
            "  setp.eq.u32 is_zero_max, $2, 0;\n"
            "  mov.b32 $0, inv;\n"
            "  @is_zero_max mov.b32 $0, 0;\n"
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
def _mxfp8_quant_bf16x4(
    values01: Int32, values23: Int32, multiplier: Int32, *, loc=None, ip=None
) -> Int32:
    """Four BF16 values (two packed words) times the power-of-two multiplier (packed BF16x2), each
    converted with ``cvt.rn.satfinite.e4m3x2``; byte i holds value i. A BF16 times a power of two
    is exact wherever the E4M3 result can be nonzero, so this equals ``fp32_vec_to_e4m3`` of the
    FP32 products."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Int32(values01).ir_value(loc=loc, ip=ip),
                Int32(values23).ir_value(loc=loc, ip=ip),
                Int32(multiplier).ir_value(loc=loc, ip=ip),
            ],
            "{\n"
            "  .reg .b32 s01, s23;\n"
            "  .reg .b16 lo, hi;\n"
            "  mul.rn.bf16x2 s01, $1, $3;\n"
            "  mul.rn.bf16x2 s23, $2, $3;\n"
            "  cvt.rn.satfinite.e4m3x2.bf16x2 lo, s01;\n"
            "  cvt.rn.satfinite.e4m3x2.bf16x2 hi, s23;\n"
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


@dsl_user_op
def _mxfp8_bf16x2_absmax(values01: Int32, running: Int32, *, loc=None, ip=None) -> Int32:
    """Packed BF16x2 running maximum of absolute values."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(values01).ir_value(loc=loc, ip=ip), Int32(running).ir_value(loc=loc, ip=ip)],
            "{\n"
            "  .reg .b32 magnitude;\n"
            "  and.b32 magnitude, $1, 2147450879;\n"
            "  max.bf16x2 $0, $2, magnitude;\n"
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
def _mxfp8_bf16x2_to_f32_max_bits(running: Int32, *, loc=None, ip=None) -> Int32:
    """FP32 bits of the larger of the two BF16 halves."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(running).ir_value(loc=loc, ip=ip)],
            "{\n"
            "  .reg .b32 lo, hi;\n"
            "  .reg .f32 flo, fhi, r;\n"
            "  shl.b32 lo, $1, 16;\n"
            "  and.b32 hi, $1, -65536;\n"
            "  mov.b32 flo, lo;\n"
            "  mov.b32 fhi, hi;\n"
            "  max.f32 r, flo, fhi;\n"
            "  mov.b32 $0, r;\n"
            "}",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _pack_f32x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Int64:
    """Two FP32 values as one 64-bit register pair (``lo`` in the low half)."""
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [Float32(lo).ir_value(loc=loc, ip=ip), Float32(hi).ir_value(loc=loc, ip=ip)],
            "mov.b64 $0, {$1, $2};",
            "=l,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fma_f32x2(a: Int64, b: Int64, c: Int64, *, loc=None, ip=None) -> Int64:
    """``a * b + c`` on FP32 pairs, each half one ``fma.rn.f32`` (no flush to zero)."""
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [
                Int64(a).ir_value(loc=loc, ip=ip),
                Int64(b).ir_value(loc=loc, ip=ip),
                Int64(c).ir_value(loc=loc, ip=ip),
            ],
            "fma.rn.f32x2 $0, $1, $2, $3;",
            "=l,l,l,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _f32x2_to_bf16x2(pair: Int64, *, loc=None, ip=None) -> Int32:
    """An FP32 pair rounded to BF16 (``cvt.rn``) and packed: the low half's value in the low 16 bits."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int64(pair).ir_value(loc=loc, ip=ip)],
            "{\n  .reg .f32 lo, hi;\n  mov.b64 {lo, hi}, $1;\n  cvt.rn.bf16x2.f32 $0, hi, lo;\n}",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class Sm100BlockwiseGemmMxfp8N256Kernel(Sm100BlockwiseGemmKernel):
    """Blockwise FP8 GEMM (C = (SFA * A) * (SFB * B)), 2-CTA 256x256 MMA tile, E4M3 C with UE8M0 scales.

    A [M, K, L] and B [N, K, L] are K-major E4M3 with FP32 1x128 (A) and 128x128 (B) scales; C [M, N, L] is a
    row-major (N-contiguous) E4M3 view and ``out_sf`` the flat swizzled 128x4 UE8M0 scales of the (M, L * N) matrix.
    Needs a static N that is a multiple of 256.
    """

    def __init__(self):
        # (2, 1) clusters in N-major persistent order (the 256x128 kernel's bits); swizzle 8 measured fastest.
        super().__init__(
            cutlass.Float32, True, (256, 256), (2, 1), swizzle_size=8, raster_along_m=False
        )
        # Warps 0-3 promote and store columns 0-127 of the CTA tile, warps 4-7 columns 128-255.
        self.acc_update_warp_id = (0, 1, 2, 3, 4, 5, 6, 7)
        self.mma_warp_id = 8
        self.tma_warp_id = 9
        self.scale_warp_id = 10
        self.sched_warp_id = 11
        self.threads_per_cta = self.threads_per_warp * 12
        self.threads_wo_sched = self.threads_per_warp * 11
        self.num_acc_update_threads = self.threads_per_warp * len(self.acc_update_warp_id)
        # 216 registers for each promotion thread (128 FP32 accumulators), 64 for the rest.
        self.tmem_load_columns = 16
        self.num_regs_uniform_warps = 64
        self.num_regs_sched_warps = 64
        self.num_regs_acc_update_warps = 216
        self.acc_update_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.num_acc_update_threads
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_warp * (1 + len(self.acc_update_warp_id)),
        )
        self.sched_sync_barrier = pipeline.NamedBarrier(
            barrier_id=3, num_threads=self.threads_per_warp
        )
        # One barrier per warpgroup around its shared-memory output tile.
        self.out_tile_barriers = (
            pipeline.NamedBarrier(barrier_id=4, num_threads=128),
            pipeline.NamedBarrier(barrier_id=5, num_threads=128),
        )

    def _setup_attributes(self):
        """Tiled MMA, tile shapes, stage counts and shared-memory layouts."""
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )
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
        if self.cta_tile_shape_mnk != (128, 256, 128):
            raise ValueError(
                f"Unexpected CTA tile {self.cta_tile_shape_mnk}, expected (128, 256, 128)"
            )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        self.scale_granularity_m = 1
        self.scale_granularity_n = 128
        self.scale_granularity_k = 128
        self.scale_m_per_tile = self.cta_tile_shape_mnk[0] // self.scale_granularity_m
        self.scale_n_per_tile = self.cta_tile_shape_mnk[1] // self.scale_granularity_n
        self.scale_k_per_tile = self.cta_tile_shape_mnk[2] // self.scale_granularity_k

        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Promotion subtile: one 32x32b TMEM load of ``tmem_load_columns`` columns of the thread's row.
        self.epi_tile = (self.cta_tile_shape_mnk[0], self.tmem_load_columns)
        # Each warpgroup's output tile (128 rows x 128 E4M3 columns), stored by TMA.
        self.out_tile = (self.cta_tile_shape_mnk[0], 128)

        self.num_acc_stage = 2
        self.num_scale_stage = 8
        self.num_tile_stage = 2

        self.out_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            cutlass.Float8E4M3FN,
            utils.LayoutEnum.ROW_MAJOR,
            self.out_tile,
            2,
        )
        a_smem_layout_one = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, 1
        )
        b_smem_layout_one = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, 1
        )
        ab_bytes_per_stage = cute.size_in_bytes(
            self.a_dtype, a_smem_layout_one
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout_one)
        out_bytes = cute.size_in_bytes(cutlass.Float8E4M3FN, self.out_smem_layout_staged)
        scale_bytes = (
            (
                (
                    self.scale_m_per_tile * (self.sfa_dtype.width // 8)
                    + self.scale_n_per_tile * (self.sfb_dtype.width // 8)
                )
                * self.num_scale_stage
                + 1023
            )
            // 1024
            * 1024
        )
        # Barriers, tile info and the TMEM holding word, rounded up to the 1 KiB buffer alignment.
        reserved_bytes = 1024
        self.num_ab_stage = (
            self.num_smem_capacity - reserved_bytes - out_bytes - scale_bytes
        ) // ab_bytes_per_stage

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage
        )
        self.sfa_smem_layout_staged = cute.make_layout(
            (
                (self.scale_granularity_m, self.scale_m_per_tile),
                (self.scale_granularity_k, self.scale_k_per_tile),
                self.num_scale_stage,
            ),
            stride=(
                (0, self.scale_k_per_tile),
                (0, 1),
                self.scale_k_per_tile * self.scale_m_per_tile,
            ),
        )
        self.sfb_smem_layout_staged = cute.make_layout(
            (
                (self.scale_granularity_n, self.scale_n_per_tile),
                (self.scale_granularity_k, self.scale_k_per_tile),
                self.num_scale_stage,
            ),
            stride=(
                (0, self.scale_k_per_tile),
                (0, 1),
                self.scale_k_per_tile * self.scale_n_per_tile,
            ),
        )
        self.num_tmem_alloc_cols = 512

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        out_sf: cute.Tensor,
    ):
        """Set up the TMA atoms, the grid and the shared storage, then launch the kernel."""
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.sfa_dtype = sfa.element_type
        self.sfb_dtype = sfb.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)
        if cutlass.const_expr(c.element_type is not cutlass.Float8E4M3FN):
            raise TypeError("Sm100BlockwiseGemmMxfp8N256Kernel needs a Float8E4M3FN C")
        # A raw-pointer tensor of bytes may carry a signless 8-bit type; the epilogue stores bytes.
        if cutlass.const_expr(out_sf.element_type.width != 8):
            raise TypeError("Sm100BlockwiseGemmMxfp8N256Kernel needs byte output scales")
        if cutlass.const_expr(self.c_layout != utils.LayoutEnum.ROW_MAJOR):
            raise TypeError("Sm100BlockwiseGemmMxfp8N256Kernel needs a row-major (N-contiguous) C")
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")
        # The SFB copy is unpredicated, so every 256-column tile must lie inside N.
        if cutlass.const_expr(not cute.is_static(b.shape[0]) or b.shape[0] % 256 != 0):
            raise ValueError(
                "Sm100BlockwiseGemmMxfp8N256Kernel needs a static N that is a multiple of 256"
            )

        self._setup_attributes()

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        a_op = self._get_tma_atom_kind(atom_thr_size, self.is_a_mcast)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        b_op = self._get_tma_atom_kind(atom_thr_size, self.is_b_mcast)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        # TMA store of one warpgroup's 128x128 E4M3 tile.
        out_smem_layout = cute.slice_(self.out_smem_layout_staged, (None, None, 0))
        c_cta_v_layout = cute.composition(cute.make_identity_layout(c.shape), self.out_tile)
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c,
            out_smem_layout,
            c_cta_v_layout,
        )

        tensor_sfa = cute.make_tensor(
            sfa.iterator,
            cute.make_layout(
                (
                    (self.scale_granularity_m, sfa.shape[0]),
                    (self.scale_granularity_k, sfa.shape[1]),
                    sfa.shape[2],
                ),
                stride=(
                    (0, sfa.layout.stride[0]),
                    (0, sfa.layout.stride[1]),
                    sfa.layout.stride[2],
                ),
            ),
        )
        tensor_sfb = cute.make_tensor(
            sfb.iterator,
            cute.make_layout(
                (
                    (self.scale_granularity_n, sfb.shape[0]),
                    (self.scale_granularity_k, sfb.shape[1]),
                    sfb.shape[2],
                ),
                stride=(
                    (0, sfb.layout.stride[0]),
                    (0, sfb.layout.stride[1]),
                    sfb.layout.stride[2],
                ),
            ),
        )

        self.tile_sched_params, grid = self._compute_grid(
            c,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            max_active_clusters,
            self.swizzle_size,
            self.raster_along_m,
        )

        self.buffer_align_bytes = 1024

        @cute.struct
        class SharedStorage:
            # (bidx, bidy, bidz, valid)
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 4 * self.num_tile_stage],
                1,
            ]
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            scale_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_scale_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tile_info_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_tile_stage * 2]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # Two 128x128 E4M3 output tiles (one per warpgroup), 128B-swizzled for the TMA store.
            sOut: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float8E4M3FN, cute.cosize(self.out_smem_layout_staged.outer)
                ],
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
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sfa_dtype, cute.cosize(self.sfa_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sfb_dtype, cute.cosize(self.sfb_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            tensor_sfa,
            tensor_sfb,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.out_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.tile_sched_params,
            out_sf,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )
        return

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
        mSFA_mkl: cute.Tensor,
        mSFB_nkl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        out_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        out_sf: cute.Tensor,
    ):
        """Persistent warp-specialized kernel: scheduler, TMA, scale-load and MMA warps as in
        ``Sm100BlockwiseGemmKernel``; eight promotion warps that also run the MXFP8 epilogue."""
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        lane_idx = cute.arch.lane_idx()

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr
        tmem_holding_buf = storage.tmem_holding_buf

        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        scale_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * 1,
        )
        scale_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_acc_update_threads,
        )
        scale_pipeline = pipeline.PipelineCpAsync.create(
            barrier_storage=storage.scale_mbar_ptr.data_ptr(),
            num_stages=self.num_scale_stage,
            producer_group=scale_pipeline_producer_group,
            consumer_group=scale_pipeline_consumer_group,
            defer_sync=True,
        )

        # Every promotion warp of both CTAs of the pair releases each accumulator stage once.
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.acc_update_warp_id) * (2 if use_2cta_instrs else 1)
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

        tile_info_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * 1,
        )
        tile_info_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_wo_sched,
        )
        tile_info_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.tile_info_mbar_ptr.data_ptr(),
            num_stages=self.num_tile_stage,
            producer_group=tile_info_pipeline_producer_group,
            consumer_group=tile_info_pipeline_consumer_group,
            defer_sync=True,
        )

        tmem = utils.TmemAllocator(
            tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.acc_update_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=tmem_dealloc_mbar_ptr,
        )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # (OUT_TILE_M, OUT_TILE_N, WARPGROUP)
        sOut = storage.sOut.get_tensor(
            out_smem_layout_staged.outer, swizzle=out_smem_layout_staged.inner
        )
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        info_layout = cute.make_layout((4, self.num_tile_stage), stride=(1, 4))
        sInfo = storage.sInfo.get_tensor(info_layout)

        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        # (bM, bK, loopM, loopK, loopL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, loopN, loopK, loopL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bN, loopM, loopN, loopL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        gSFA_mkl = cute.local_tile(
            mSFA_mkl,
            cute.slice_(self.cta_tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.cta_tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        cSFA_mkl = cute.make_identity_tensor(cute.shape(mSFA_mkl))
        cSFA = cute.local_tile(
            cSFA_mkl,
            cute.slice_(self.cta_tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, loopM, loopK, loopL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, loopN, loopK, loopL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_N, loopM, loopN, loopL)
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

        atom_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mSFA_mkl.element_type,
            num_bits_per_copy=mSFA_mkl.element_type.width,
        )
        tiled_copy_sfa = cute.make_tiled_copy_tv(
            atom_copy, cute.make_layout((32,)), cute.make_layout((1,))
        )
        tiled_copy_sfb = cute.make_tiled_copy_tv(
            atom_copy, cute.make_layout((32,)), cute.make_layout((1,))
        )
        thr_copy_sfa = tiled_copy_sfa.get_slice(lane_idx)
        thr_copy_sfb = tiled_copy_sfb.get_slice(lane_idx)
        tAgSFA_mkl = thr_copy_sfa.partition_S(gSFA_mkl)
        tAsSFA = thr_copy_sfa.partition_D(sSFA)
        tAcSFA = thr_copy_sfa.partition_S(cSFA)
        tBgSFB_nkl = thr_copy_sfb.partition_S(gSFB_nkl)
        tBsSFB = thr_copy_sfb.partition_D(sSFB)

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        #
        # Scheduler warp
        #
        if warp_idx == self.sched_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_sched_warps)
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            tile_info_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_tile_stage
            )
            while work_tile.is_valid_tile:
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                tile_info_pipeline.producer_acquire(tile_info_producer_state)
                cur_tile_coord = work_tile.tile_idx
                with cute.arch.elect_one():
                    sInfo[(0, tile_info_producer_state.index)] = cur_tile_coord[0]
                    sInfo[(1, tile_info_producer_state.index)] = cur_tile_coord[1]
                    sInfo[(2, tile_info_producer_state.index)] = cur_tile_coord[2]
                    sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(
                        work_tile.is_valid_tile
                    )
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                self.sched_sync_barrier.arrive_and_wait()
                tile_info_pipeline.producer_commit(tile_info_producer_state)
                tile_info_producer_state.advance()
            tile_info_pipeline.producer_tail(tile_info_producer_state)

        #
        # TMA load warp
        #
        if warp_idx == self.tma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_uniform_warps)
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            cur_tile_coord = work_tile.tile_idx
            tile_info[0] = cur_tile_coord[0]
            tile_info[1] = cur_tile_coord[1]
            tile_info[2] = cur_tile_coord[2]
            tile_info[3] = work_tile.is_valid_tile
            is_valid_tile = cutlass.Boolean(1)
            is_valid_tile = tile_info[3] == 1

            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )
                tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
                tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    tAgA_k = tAgA_slice[(None, ab_producer_state.count)]
                    tBgB_k = tBgB_slice[(None, ab_producer_state.count)]
                    tAsA_pipe = tAsA[(None, ab_producer_state.index)]
                    tBsB_pipe = tBsB[(None, ab_producer_state.index)]
                    tma_bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                    ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
                    cute.copy(
                        tma_atom_a,
                        tAgA_k,
                        tAsA_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_k,
                        tBsB_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=b_full_mcast_mask,
                    )

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

            ab_pipeline.producer_tail(ab_producer_state)

        #
        # Scale load warp
        #
        if warp_idx == self.scale_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_uniform_warps)
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            scale_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_scale_stage
            )
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            cur_tile_coord = work_tile.tile_idx
            tile_info[0] = cur_tile_coord[0]
            tile_info[1] = cur_tile_coord[1]
            tile_info[2] = cur_tile_coord[2]
            tile_info[3] = work_tile.is_valid_tile
            is_valid_tile = cutlass.Boolean(1)
            is_valid_tile = tile_info[3] == 1

            while is_valid_tile:
                tApSFA = cute.make_rmem_tensor(
                    cute.make_layout(
                        cute.filter_zeros(cute.slice_(tAsSFA, (None, None, None, 0))).shape
                    ),
                    cutlass.Boolean,
                )
                scale_producer_state.reset_count()
                peek_scale_empty_status = cutlass.Boolean(1)
                if scale_producer_state.count < k_tile_cnt:
                    peek_scale_empty_status = scale_pipeline.producer_try_acquire(
                        scale_producer_state
                    )
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    tAsSFA_pipe = cute.filter_zeros(
                        tAsSFA[(None, None, None, scale_producer_state.index)]
                    )
                    tBsSFB_pipe = cute.filter_zeros(
                        tBsSFB[(None, None, None, scale_producer_state.index)]
                    )
                    tAgSFA_k = cute.filter_zeros(
                        tAgSFA_mkl[
                            (
                                None,
                                None,
                                None,
                                tile_info[0],
                                scale_producer_state.count,
                                tile_info[2],
                            )
                        ]
                    )
                    tBgSFB_k = cute.filter_zeros(
                        tBgSFB_nkl[
                            (
                                None,
                                None,
                                None,
                                tile_info[1],
                                scale_producer_state.count,
                                tile_info[2],
                            )
                        ]
                    )
                    tAcSFA_compact = cute.filter_zeros(
                        cute.slice_(
                            tAcSFA,
                            (
                                None,
                                None,
                                None,
                                tile_info[0],
                                scale_producer_state.count,
                                tile_info[2],
                            ),
                        )
                    )
                    for i in cutlass.range_constexpr(cute.size(tApSFA, mode=[1])):
                        tApSFA[((0, 0), i, (0, 0))] = cute.elem_less(
                            tAcSFA_compact[(i)][0], mSFA_mkl.shape[0]
                        )
                    scale_pipeline.producer_acquire(scale_producer_state, peek_scale_empty_status)
                    cute.copy(tiled_copy_sfa, tAgSFA_k, tAsSFA_pipe, pred=tApSFA)
                    # N % 256 == 0: every tile's SFB blocks are in bounds.
                    cute.copy(tiled_copy_sfb, tBgSFB_k, tBsSFB_pipe)
                    scale_pipeline.producer_commit(scale_producer_state)
                    scale_producer_state.advance()
                    peek_scale_empty_status = cutlass.Boolean(1)
                    if scale_producer_state.count < k_tile_cnt:
                        peek_scale_empty_status = scale_pipeline.producer_try_acquire(
                            scale_producer_state
                        )

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

            scale_pipeline.producer_tail(scale_producer_state)

        #
        # MMA warp
        #
        if warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_uniform_warps)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
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
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            cur_tile_coord = work_tile.tile_idx
            tile_info[0] = cur_tile_coord[0]
            tile_info[1] = cur_tile_coord[1]
            tile_info[2] = cur_tile_coord[2]
            tile_info[3] = work_tile.is_valid_tile
            is_valid_tile = cutlass.Boolean(1)
            is_valid_tile = tile_info[3] == 1

            while is_valid_tile:
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)
                acc_producer_state.reset_count()
                peek_acc_empty_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_acc_empty_status = acc_pipeline.producer_try_acquire(acc_producer_state)

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]
                    if is_leader_cta:
                        acc_pipeline.producer_acquire(acc_producer_state, peek_acc_empty_status)
                    # Every k-block starts a fresh accumulation: the promotion warps apply its scales.
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    if is_leader_cta:
                        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                ab_consumer_state.index,
                            )
                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblock_coord],
                                tCrB[kblock_coord],
                                tCtAcc,
                            )
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
                    if acc_producer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_acc_empty_status = acc_pipeline.producer_try_acquire(
                                acc_producer_state
                            )

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

            acc_pipeline.producer_tail(acc_producer_state)

        #
        # Promotion + MXFP8 epilogue warps
        #
        if warp_idx <= self.acc_update_warp_id[-1]:
            cute.arch.warpgroup_reg_alloc(self.num_regs_acc_update_warps)
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            # Warpgroup 0 owns columns 0-127 of the CTA tile, warpgroup 1 columns 128-255; within a warpgroup, thread
            # t owns row t (TMEM lane t) of the CTA tile.
            warpgroup_idx = warp_idx // 4
            warp_in_group = warp_idx % 4
            epi_tidx = tidx % 128

            # (MMA, MMA_M, MMA_N, STAGE) from this warpgroup's first column: the column offset stays out of the loop.
            tCtAcc_base = cute.make_tensor(
                tmem_ptr + warpgroup_idx * self.out_tile[1], tCtAcc_fake.layout
            )
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.tmem_load_columns)),
                self.acc_dtype,
            )
            # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
            tAcc_epi = cute.flat_divide(tCtAcc_base[((None, None), 0, 0, None)], self.epi_tile)
            tiled_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tAcc_epi[(None, None, 0, 0, 0)])
            thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
            # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, STAGE)
            tTR_tAcc_base = thr_copy_t2r.partition_S(tAcc_epi)
            # One warpgroup's 128 columns: (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN, loopL)
            gC_half = cute.flat_divide(tCgC[((None, None), 0, 0, None, None, None)], self.out_tile)[
                (None, None, 0, 0, None, None, None)
            ]
            gC_half_epi = cute.flat_divide(gC_half, self.epi_tile)
            tTR_gC = thr_copy_t2r.partition_D(gC_half_epi)
            # (T2R, T2R_M, T2R_N): one subtile of the thread's row, in column order
            tTR_rAcc = cute.make_rmem_tensor(
                tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
            )
            acc_values = cute.make_tensor(
                tTR_rAcc.iterator, cute.make_layout((self.tmem_load_columns,))
            )
            num_subtiles = self.out_tile[1] // self.tmem_load_columns
            pairs_per_subtile = self.tmem_load_columns // 2
            # The thread's 128 promoted values as 64 FP32 pairs (columns 2j, 2j + 1), updated in place by
            # ``fma.rn.f32x2`` so that the loop-carried pairs keep their registers.
            final_pairs = cute.make_rmem_tensor((64,), cutlass.Int64)

            # Per-stage scale factors: FP32 per row (A) and per 128-column block (B).
            sSFA_rows = cute.make_tensor(
                sSFA.iterator,
                cute.make_layout(
                    (self.scale_m_per_tile, self.num_scale_stage), stride=(1, self.scale_m_per_tile)
                ),
            )
            sSFB_blocks = cute.make_tensor(
                sSFB.iterator,
                cute.make_layout(
                    (self.scale_n_per_tile, self.num_scale_stage), stride=(1, self.scale_n_per_tile)
                ),
            )

            # TMA store partition: ((ATOM_V, REST_V), WARPGROUP) and
            # ((ATOM_V, REST_V), OUT_M, OUT_N, loopM, loopN, loopL).
            gC_mnl_out = cute.flat_divide(
                tCgC[((None, None), 0, 0, None, None, None)], self.out_tile
            )
            bSG_sOut, bSG_gOut = cpasync.tma_partition(
                tma_atom_c,
                0,
                cute.make_layout(1),
                cute.group_modes(sOut, 0, 2),
                cute.group_modes(gC_mnl_out, 0, 2),
            )
            out_base_ptr = storage.sOut.data_ptr()

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            scale_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_scale_stage
            )
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            cur_tile_coord = work_tile.tile_idx
            tile_info[0] = cur_tile_coord[0]
            tile_info[1] = cur_tile_coord[1]
            tile_info[2] = cur_tile_coord[2]
            tile_info[3] = work_tile.is_valid_tile
            is_valid_tile = cutlass.Boolean(1)
            is_valid_tile = tile_info[3] == 1

            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )
                final_pairs.fill(0)

                scale_consumer_state.reset_count()
                peek_scale_full_status = cutlass.Boolean(1)
                if scale_consumer_state.count < k_tile_cnt:
                    peek_scale_full_status = scale_pipeline.consumer_try_wait(scale_consumer_state)
                acc_consumer_state.reset_count()
                peek_acc_full_status = cutlass.Boolean(1)
                if acc_consumer_state.count < k_tile_cnt:
                    peek_acc_full_status = acc_pipeline.consumer_try_wait(acc_consumer_state)

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # (T2R, T2R_M, T2R_N, (EPI_M, EPI_N))
                    tTR_tAcc = tTR_tAcc_base[
                        (None, None, None, None, None, acc_consumer_state.index)
                    ]
                    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                    scale_pipeline.consumer_wait(scale_consumer_state, peek_scale_full_status)
                    scale_a = sSFA_rows[(epi_tidx, scale_consumer_state.index)]
                    scale_b = sSFB_blocks[(warpgroup_idx, scale_consumer_state.index)]
                    scale = scale_a * scale_b
                    scale_pair = _pack_f32x2(scale, scale)

                    acc_pipeline.consumer_wait(acc_consumer_state, peek_acc_full_status)

                    for subtile_idx in cutlass.range_constexpr(num_subtiles):
                        cute.copy(
                            tiled_copy_t2r, tTR_tAcc[(None, None, None, subtile_idx)], tTR_rAcc
                        )
                        # final = acc * (scale_a * scale_b) + final, one FMA per element in k order.
                        for pair_idx in cutlass.range_constexpr(pairs_per_subtile):
                            final_idx = subtile_idx * pairs_per_subtile + pair_idx
                            final_pairs[final_idx] = _fma_f32x2(
                                scale_pair,
                                _pack_f32x2(acc_values[2 * pair_idx], acc_values[2 * pair_idx + 1]),
                                final_pairs[final_idx],
                            )

                    scale_pipeline.consumer_release(scale_consumer_state)
                    scale_consumer_state.advance()
                    peek_scale_full_status = cutlass.Boolean(1)
                    if scale_consumer_state.count < k_tile_cnt:
                        peek_scale_full_status = scale_pipeline.consumer_try_wait(
                            scale_consumer_state
                        )

                    cute.arch.fence_view_async_tmem_load()
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()
                    peek_acc_full_status = cutlass.Boolean(1)
                    if acc_consumer_state.count < k_tile_cnt:
                        peek_acc_full_status = acc_pipeline.consumer_try_wait(acc_consumer_state)

                self._epilog_mxfp8_from_registers(
                    final_pairs,
                    out_base_ptr,
                    tma_atom_c,
                    bSG_sOut,
                    bSG_gOut,
                    warpgroup_idx,
                    warp_in_group,
                    lane_idx,
                    mma_tile_coord_mnl,
                    mma_tile_coord_v,
                    mC_mnl,
                    out_sf,
                )

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

            # The last TMA stores must finish reading shared memory before the CTA exits.
            if warp_in_group == 0:
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            tmem.relinquish_alloc_permit()
            self.acc_update_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

    @cute.jit
    def _out_tile_sync(self, warpgroup_idx):
        """Named barrier over one warpgroup (128 threads)."""
        if warpgroup_idx == 0:
            self.out_tile_barriers[0].arrive_and_wait()
        else:
            self.out_tile_barriers[1].arrive_and_wait()

    @cute.jit
    def _epilog_mxfp8_from_registers(
        self,
        final_pairs,
        out_base_ptr,
        tma_atom_c,
        bSG_sOut,
        bSG_gOut,
        warpgroup_idx,
        warp_in_group,
        lane_idx,
        mma_tile_coord_mnl,
        mma_tile_coord_v,
        mC_mnl,
        out_sf,
    ):
        """Quantize the thread's 128 promoted values (one row of its warpgroup's 128x128 tile) to E4M3 with
        per-32-column UE8M0 scales, exactly as ``Sm100BlockwiseGemmKernel._epilog_mxfp8``: BF16 rounding, block
        absolute maximum, ``rcp.approx.ftz(448)`` / ``cvt.rp.satfinite`` E8M0 code, ``cvt.rn.satfinite`` E4M3 of
        ``bf16 * rcp.approx.ftz(2^(code - 127))`` (0 for an all-zero block). The row's four codes land as one 32-bit
        word in the swizzled 128x4 scale layout (zero codes for rows past M up to the 128-row padding); the E4M3 row
        goes to the warpgroup's 128B-swizzled shared-memory tile, which one TMA store writes out (rows past M are
        clipped by the tensor map)."""
        tile_m_base = (
            mma_tile_coord_mnl[0] * self.mma_tiler[0]
            + mma_tile_coord_v * self.cta_tile_shape_mnk[0]
        )
        num_rows = mC_mnl.shape[0]
        valid_rows = num_rows - tile_m_base
        padded_rows = (num_rows + 127) // 128 * 128
        global_n_base = mma_tile_coord_mnl[1] * self.mma_tiler[1] + warpgroup_idx * self.out_tile[1]
        num_sf_cols = (mC_mnl.shape[1] * mC_mnl.shape[2]) // 32
        global_sf_col = (mma_tile_coord_mnl[2] * mC_mnl.shape[1] + global_n_base) // 32
        num_sf_col_tiles = (num_sf_cols + 3) // 4
        output_sf_tile_base = (tile_m_base // 128) * num_sf_col_tiles * 512 + (
            global_sf_col // 4
        ) * 512

        # Round the row to BF16 and keep it packed; one 32-column block per segment.
        row_values = cute.make_rmem_tensor(
            cute.make_layout((16, 4), stride=(1, 16)),
            cutlass.Int32,
        )
        block_max_bits = cute.make_rmem_tensor((4,), cutlass.Int32)
        for segment_idx in cutlass.range_constexpr(4):
            running = cutlass.Int32(0)
            for packed_idx in cutlass.range_constexpr(16):
                packed = _f32x2_to_bf16x2(final_pairs[segment_idx * 16 + packed_idx])
                row_values[(packed_idx, segment_idx)] = packed
                running = _mxfp8_bf16x2_absmax(packed, running)
            block_max_bits[segment_idx] = _mxfp8_bf16x2_to_f32_max_bits(running)

        # The previous tile's TMA store must have read this warpgroup's shared-memory tile.
        if warp_in_group == 0:
            cute.arch.cp_async_bulk_wait_group(0, read=True)
        self._out_tile_sync(warpgroup_idx)

        epi_row = warp_in_group * 32 + lane_idx
        out_tile_bytes = self.out_tile[0] * self.out_tile[1]
        scratch_i32_ptr = cute.recast_ptr(
            out_base_ptr + warpgroup_idx * out_tile_bytes, dtype=cutlass.Int32
        )
        sf_word = cutlass.Int32(0)
        for segment_idx in cutlass.range_constexpr(4):
            code = _mxfp8_block_scale(block_max_bits[segment_idx])
            multiplier_f32 = _mxfp8_quant_multiplier(code, block_max_bits[segment_idx])
            # A power of two (or 0 / inf): its BF16 is the top half, exact.
            multiplier_bf16 = (multiplier_f32 >> 16) & 0xFFFF
            multiplier = multiplier_bf16 | (multiplier_bf16 << 16)
            sf_word = sf_word | (code << (8 * segment_idx))
            packed_output = cute.make_rmem_tensor((8,), cutlass.Int32)
            for output_idx in cutlass.range_constexpr(8):
                packed_output[output_idx] = _mxfp8_quant_bf16x4(
                    row_values[(output_idx * 2, segment_idx)],
                    row_values[(output_idx * 2 + 1, segment_idx)],
                    multiplier,
                )
            # 128B swizzle: the 16-byte chunk c of row r sits at chunk slot c ^ (r % 8).
            row_swizzle = epi_row % 8
            _mxfp8_store_shared_b128(
                scratch_i32_ptr + epi_row * 32 + ((segment_idx * 2) ^ row_swizzle) * 4,
                packed_output[0],
                packed_output[1],
                packed_output[2],
                packed_output[3],
            )
            _mxfp8_store_shared_b128(
                scratch_i32_ptr + epi_row * 32 + ((segment_idx * 2 + 1) ^ row_swizzle) * 4,
                packed_output[4],
                packed_output[5],
                packed_output[6],
                packed_output[7],
            )

        global_row = tile_m_base + epi_row
        if global_row < padded_rows:
            output_sf_offset = output_sf_tile_base + (epi_row % 32) * 16 + (epi_row // 32) * 4
            if epi_row >= valid_rows:
                sf_word = cutlass.Int32(0)
            _mxfp8_store_global_b32(out_sf.iterator + output_sf_offset, sf_word)

        # Shared-memory writes visible to the TMA (async proxy), then one warp stores the tile.
        cute.arch.fence_proxy(
            "async.shared",
            space="cta",
        )
        self._out_tile_sync(warpgroup_idx)
        if warp_in_group == 0:
            cute.copy(
                tma_atom_c,
                bSG_sOut[(None, warpgroup_idx)],
                bSG_gOut[
                    (
                        None,
                        0,
                        warpgroup_idx,
                        mma_tile_coord_mnl[0],
                        mma_tile_coord_mnl[1],
                        mma_tile_coord_mnl[2],
                    )
                ],
            )
            cute.arch.cp_async_bulk_commit_group()
