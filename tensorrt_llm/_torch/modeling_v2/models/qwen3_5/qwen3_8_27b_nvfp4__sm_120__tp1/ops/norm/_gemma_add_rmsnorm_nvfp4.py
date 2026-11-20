# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from FlashInfer (https://github.com/flashinfer-ai/flashinfer) under the Apache-2.0
# license.  Original file: flashinfer/norm/kernels/fused_add_rmsnorm.py
# Original copyright: Copyright (c) 2025 by FlashInfer team.
"""Gemma fused add + RMSNorm with an NVFP4 (group-16, E4M3 SF, 128x4 swizzle) epilogue.

A subclass of FlashInfer's CuTe-DSL ``FusedAddRMSNormKernel``, imported from the
installed FlashInfer and left unmodified: the kernel body below is that kernel's body,
unchanged up to ``y = h * rstd * (w + weight_bias)``, so every other user of the stock
norm keeps the stock kernel by construction.

Bitwise contract: equal to
    flashinfer gemma_fused_add_rmsnorm (CuTe-DSL, weight_bias=1.0, BF16 y)
    + torch.ops.trtllm.fp4_quantize(y, global_scale, 16, False)
on the residual (bf16(x + r), in place), the FP4 bytes [M, H/2] and the
swizzled SF buffer, including the zeroed SF padding rows [M, ceil128(M)).
x (the norm input) is NOT overwritten with y here.

The NVFP4 tail reproduces cvt_warp_fp16_to_fp4 (quantization.cuh, built
without fast math): vecMax over 16 BF16 values (lanes 2k/2k+1, one xor-1
shuffle), SF = e4m3_satfinite(mul.rn(gs, mul.rn(vmax, rcp.approx.ftz(6)))),
outScale = vmax != 0 ? rcp.approx.ftz(mul.rn(f32(SF), rcp.approx.ftz(gs))) : 0,
codes = cvt.rn.satfinite.e2m1x2(mul.rn(y, outScale)).

Supported: BF16/FP16, H where the parent geometry is cluster_n == 1,
vec_size == 8, threads_per_row * 8 * num_vec_blocks == H (e.g. H = 5120).
"""

import functools
import math

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint8, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from flashinfer.norm.kernels.fused_add_rmsnorm import FusedAddRMSNormKernel
from flashinfer.norm.kernels.rmsnorm import RMSNormKernel
from flashinfer.norm.utils import (
    _torch_dtype_to_str,
    get_cutlass_dtype,
    get_ptr_as_int64,
    get_sm_version,
    predicate_k,
    row_reduce_sum_multirow,
)

_SF_VEC = 16
# CTAs that zero the SF padding rows of the last 128-row tile.
_N_PAD_CTAS = 8


# =============================================================================
# PTX helpers
# =============================================================================


@dsl_user_op
def _fabs_f32(a: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "abs.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _fmax_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "max.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _nvfp4_sf_and_scale(vmax: Float32, gs: Float32, *, loc=None, ip=None):
    """Returns (SF byte in the low 8 bits of a u32, outScale)."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.f32()]),
        [Float32(vmax).ir_value(loc=loc, ip=ip), Float32(gs).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .f32 c6, r6, t, sfv, zero, sff, igs, p, osc;
            .reg .b16 pair, hlo, hhi;
            .reg .b32 h2;
            .reg .pred nz;
            mov.f32 c6, 0f40C00000;
            mov.f32 zero, 0f00000000;
            rcp.approx.ftz.f32 r6, c6;
            mul.rn.f32 t, $2, r6;
            mul.rn.f32 sfv, $3, t;
            cvt.rn.satfinite.e4m3x2.f32 pair, zero, sfv;
            cvt.u32.u16 $0, pair;
            cvt.rn.f16x2.e4m3x2 h2, pair;
            mov.b32 {hlo, hhi}, h2;
            cvt.f32.f16 sff, hlo;
            rcp.approx.ftz.f32 igs, $3;
            mul.rn.f32 p, sff, igs;
            rcp.approx.ftz.f32 osc, p;
            setp.neu.f32 nz, $2, zero;
            selp.f32 $1, osc, zero, nz;
        }
        """,
        "=r,=f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    sf = Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip))
    osc = Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip))
    return sf, osc


@dsl_user_op
def _e2m1x8(
    y0: Float32,
    y1: Float32,
    y2: Float32,
    y3: Float32,
    y4: Float32,
    y5: Float32,
    y6: Float32,
    y7: Float32,
    osc: Float32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """8 values * outScale (mul.rn) -> 8 packed E2M1 codes, same byte/nibble
    order as fp32_vec_to_e2m1 (element 0 in the low nibble of byte 0)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(v).ir_value(loc=loc, ip=ip) for v in (y0, y1, y2, y3, y4, y5, y6, y7, osc)],
            """
            {
                .reg .f32 q0, q1, q2, q3, q4, q5, q6, q7;
                .reg .b8 b0, b1, b2, b3;
                mul.rn.f32 q0, $1, $9;
                mul.rn.f32 q1, $2, $9;
                mul.rn.f32 q2, $3, $9;
                mul.rn.f32 q3, $4, $9;
                mul.rn.f32 q4, $5, $9;
                mul.rn.f32 q5, $6, $9;
                mul.rn.f32 q6, $7, $9;
                mul.rn.f32 q7, $8, $9;
                cvt.rn.satfinite.e2m1x2.f32 b0, q1, q0;
                cvt.rn.satfinite.e2m1x2.f32 b1, q3, q2;
                cvt.rn.satfinite.e2m1x2.f32 b2, q5, q4;
                cvt.rn.satfinite.e2m1x2.f32 b3, q7, q6;
                mov.b32 $0, {b0, b1, b2, b3};
            }
            """,
            "=r,f,f,f,f,f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_global_u32(addr: Int64, value: Uint32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Uint32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


# =============================================================================
# Kernel
# =============================================================================


class FusedAddRMSNormNVFP4Kernel(FusedAddRMSNormKernel):
    """FusedAddRMSNormKernel (pinned) + NVFP4 group-16 swizzled epilogue."""

    def __init__(
        self,
        dtype: cutlass.Numeric,
        H: int,
        weight_bias: float = 1.0,
        sm_version: int | None = None,
    ):
        super().__init__(dtype, H, weight_bias, sm_version=sm_version)
        if not (
            self.cluster_n == 1
            and self.vec_size == 8
            and self.threads_per_row % 2 == 0
            and self.threads_per_row * self.vec_size * self.num_vec_blocks == H
            and H % (4 * _SF_VEC) == 0
        ):
            raise ValueError(
                f"FusedAddRMSNormNVFP4Kernel: unsupported geometry for H={H} "
                f"(cluster_n={self.cluster_n}, vec={self.vec_size}, "
                f"tpr={self.threads_per_row}, nvb={self.num_vec_blocks})"
            )
        self.num_k_tiles = (H // _SF_VEC + 3) // 4

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        mQ: cute.Tensor,
        mSF: cute.Tensor,
        mS: cute.Tensor,
        M: Int64,
        n_row_ctas: Int32,
        grid_x: Int32,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        tv_shape, tv_stride = RMSNormKernel._make_tv_layout(
            self.threads_per_row,
            self.rows_per_block,
            self.vec_size,
            self.num_vec_blocks,
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (self.rows_per_block, self.cols_per_tile)

        self.kernel_nvfp4(
            mX, mR, mW, mQ, mSF, mS, M, n_row_ctas, eps, enable_pdl, tv_layout, tiler_mn
        ).launch(
            grid=[grid_x, 1, 1],
            block=[self.num_threads, 1, 1],
            smem=self._smem_size_in_bytes(),
            stream=stream,
            use_pdl=enable_pdl,
        )

    @cute.kernel
    def kernel_nvfp4(
        self,
        mX: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        mQ: cute.Tensor,
        mSF: cute.Tensor,
        mS: cute.Tensor,
        M: Int64,
        n_row_ctas: Int32,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        # PDL: Wait for previous kernel (SM90+ only)
        if enable_pdl:
            cute.arch.griddepcontrol_wait()

        H = self.H
        cluster_n = self.cluster_n
        weight_bias = self.weight_bias
        copy_bits = self.copy_bits
        threads_per_row = tv_layout.shape[0][0]
        rows_per_block = tiler_mn[0]
        warps_per_row = max(threads_per_row // 32, 1)

        cluster_y = cutlass.const_expr(0)

        # ===== Allocate shared memory =====
        smem = cutlass.utils.SmemAllocator()

        if cutlass.const_expr(self.use_async_copy):
            sX = smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )
            sR = smem.allocate_tensor(
                mR.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )

        reduction_buffer = smem.allocate_tensor(
            Float32,
            cute.make_layout((rows_per_block, warps_per_row)),
            byte_alignment=4,
        )
        mbar_ptr = None

        # ===== Coordinate tracking and tiling =====
        idX = cute.make_identity_tensor(mX.shape)

        gX = cute.local_tile(mX, tiler_mn, (bidx, cluster_y))
        gR = cute.local_tile(mR, tiler_mn, (bidx, cluster_y))
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

        mW_expanded_layout = cute.prepend(mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,)))
        mW_2d = cute.make_tensor(mW.iterator, mW_expanded_layout)
        gW = cute.local_tile(mW_2d, tiler_mn, (0, cluster_y))

        # ===== Create TiledCopy atoms =====
        copy_atom_sync = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=copy_bits,
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=copy_bits,
        )

        if cutlass.const_expr(self.use_async_copy):
            copy_atom_async = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mX.element_type,
                num_bits_per_copy=copy_bits,
            )
            tiled_copy_load = cute.make_tiled_copy(copy_atom_async, tv_layout, tiler_mn)
        else:
            tiled_copy_load = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)

        tiled_copy_W = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)
        tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)

        thr_copy_X = tiled_copy_load.get_slice(tidx)
        thr_copy_W = tiled_copy_W.get_slice(tidx)
        thr_copy_O = tiled_copy_store.get_slice(tidx)

        # Partition input
        tXgX = thr_copy_X.partition_S(gX)
        tXcX = thr_copy_X.partition_S(cX)
        tXrX = cute.make_fragment_like(tXgX)

        # Partition residual (same load tiled copy)
        tRgR = thr_copy_X.partition_S(gR)
        tRrR = cute.make_fragment_like(tRgR)

        if cutlass.const_expr(self.use_async_copy):
            tXsX = thr_copy_X.partition_D(sX)
            tRsR = thr_copy_X.partition_D(sR)

        # Partition weight (sync, separate tiled copy)
        tWgW = thr_copy_W.partition_S(gW)
        tWrW = cute.make_fragment_like(tWgW)
        tXrW = thr_copy_X.retile(tWrW)

        # Partition output destinations
        tXgO = thr_copy_O.partition_D(gX)
        tRgO = thr_copy_O.partition_D(gR)
        tXrO = cute.make_fragment_like(tXgO)

        # ===== Bounds checking =====
        tXpX = predicate_k(tXcX, limit=H)
        tWpW = predicate_k(thr_copy_W.partition_S(cX), limit=H)
        row_coord = tXcX[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < M

        # ===== Pass 1: Load input + residual, compute h, reduce =====
        if cutlass.const_expr(self.use_async_copy):
            if row_in_bounds:
                cute.copy(copy_atom_async, tXgX, tXsX, pred=tXpX)
                cute.copy(copy_atom_async, tRgR, tRsR, pred=tXpX)
            cute.arch.cp_async_commit_group()

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

            cute.arch.cp_async_wait_group(0)

            cute.autovec_copy(tXsX, tXrX)
            cute.autovec_copy(tRsR, tRrR)
        else:
            tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))
            tRrR.store(cute.zeros_like(tRrR, dtype=mR.element_type))
            if row_in_bounds:
                cute.copy(copy_atom_sync, tXgX, tXrX, pred=tXpX)
                cute.copy(copy_atom_sync, tRgR, tRrR, pred=tXpX)

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

        x_in = tXrX.load().to(Float32)
        r_in = tRrR.load().to(Float32)
        h = x_in + r_in

        # Write h to residual (global)
        tXrO.store(h.to(mR.element_type))
        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tRgO, pred=tXpX)

        h_sq = h * h
        sum_sq = row_reduce_sum_multirow(
            h_sq, threads_per_row, reduction_buffer, mbar_ptr, cluster_n
        )

        mean_sq = sum_sq / Float32(H)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        cute.arch.barrier()

        # ===== Pass 2: Normalize, round to the BF16 y, NVFP4-quantize =====
        w = tXrW.load().to(Float32)
        y = h * rstd * (w + Float32(weight_bias))

        tXrO.store(y.to(mX.element_type))
        tYr = cute.make_fragment_like(tXrO, Float32)
        tYr.store(tXrO.load().to(Float32))

        gs = mS[0]
        lane = tidx % threads_per_row
        row = bidx * rows_per_block + tidx // threads_per_row
        is_even = (lane % 2) == 0
        num_k_tiles = self.num_k_tiles
        half_h = H // 2
        vec = self.vec_size
        cols_per_vec_block = vec * threads_per_row

        for j in cutlass.range_constexpr(self.num_vec_blocks):
            y0 = tYr[vec * j + 0]
            y1 = tYr[vec * j + 1]
            y2 = tYr[vec * j + 2]
            y3 = tYr[vec * j + 3]
            y4 = tYr[vec * j + 4]
            y5 = tYr[vec * j + 5]
            y6 = tYr[vec * j + 6]
            y7 = tYr[vec * j + 7]
            a = _fmax_f32(
                _fmax_f32(
                    _fmax_f32(_fabs_f32(y0), _fabs_f32(y1)),
                    _fmax_f32(_fabs_f32(y2), _fabs_f32(y3)),
                ),
                _fmax_f32(
                    _fmax_f32(_fabs_f32(y4), _fabs_f32(y5)),
                    _fmax_f32(_fabs_f32(y6), _fabs_f32(y7)),
                ),
            )
            vmax = _fmax_f32(cute.arch.shuffle_sync_bfly(a, offset=1), a)
            sf_u32, osc = _nvfp4_sf_and_scale(vmax, gs)
            packed = _e2m1x8(y0, y1, y2, y3, y4, y5, y6, y7, osc)
            if row_in_bounds:
                col0 = lane * vec + cols_per_vec_block * j
                q_off = Int64(row) * Int64(half_h) + Int64(col0 // 2)
                _st_global_u32(get_ptr_as_int64(mQ, q_off), packed)
                if is_even:
                    c = col0 // _SF_VEC
                    sf_off = (
                        ((row // 128) * num_k_tiles + c // 4) * 512
                        + (row % 32) * 16
                        + ((row % 128) // 32) * 4
                        + c % 4
                    )
                    mSF[sf_off] = Uint8(sf_u32 & Uint32(0xFF))

        # ===== SF padding rows [M, ceil128(M)) of the last tile -> 0 =====
        # In the 128x4 swizzle every 32-bit word holds the 4 SF bytes of one
        # row: within a 512-byte k-tile chunk, word wi -> row (wi % 4) * 32 +
        # wi // 4. Row CTAs never write padding rows, so no word is shared.
        if bidx >= n_row_ctas:
            tile_words = num_k_tiles * 128
            pad_threads = _N_PAD_CTAS * self.num_threads
            pad_tid = (bidx - n_row_ctas) * self.num_threads + tidx
            m_tile = M // 128
            m0 = m_tile * 128
            base_w = m_tile * tile_words
            n_iter = (tile_words + pad_threads - 1) // pad_threads
            for it in cutlass.range_constexpr(n_iter):
                wi = pad_tid + it * pad_threads
                if wi < tile_words:
                    within = wi % 128
                    rr = (within % 4) * 32 + within // 4
                    if m0 + Int64(rr) >= M:
                        _st_global_u32(
                            get_ptr_as_int64(mSF, (base_w + Int64(wi)) * 4),
                            Uint32(0),
                        )

        # PDL: Signal dependent kernels (SM90+ only)
        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()


# =============================================================================
# Compiled kernel getter and API
# =============================================================================


@functools.cache
def _get_compiled_fused_add_rmsnorm_nvfp4_kernel(
    dtype_str: str,
    H: int,
    weight_bias: float,
    enable_pdl: bool,
    sm_version: int,
):
    dtype = get_cutlass_dtype(dtype_str)
    kernel_obj = FusedAddRMSNormNVFP4Kernel(dtype, H, weight_bias, sm_version=sm_version)

    sym_m = cute.sym_int(64)
    sym_sf = cute.sym_int(64)
    elem_bytes = dtype.width // 8
    tensor_align = math.gcd(128, H * elem_bytes)
    x_fake = cute.runtime.make_fake_compact_tensor(
        dtype, (sym_m, H), stride_order=(1, 0), assumed_align=tensor_align
    )
    r_fake = cute.runtime.make_fake_compact_tensor(
        dtype, (sym_m, H), stride_order=(1, 0), assumed_align=tensor_align
    )
    w_fake = cute.runtime.make_fake_compact_tensor(dtype, (H,), assumed_align=16)
    q_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (sym_m, H // 2), stride_order=(1, 0), assumed_align=16
    )
    sf_fake = cute.runtime.make_fake_compact_tensor(cutlass.Uint8, (sym_sf,), assumed_align=16)
    s_fake = cute.runtime.make_fake_compact_tensor(Float32, (1,), assumed_align=4)
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    return kernel_obj, cute.compile(
        kernel_obj,
        x_fake,
        r_fake,
        w_fake,
        q_fake,
        sf_fake,
        s_fake,
        Int64(1),
        Int32(1),
        Int32(1),
        Float32(1e-6),
        enable_pdl,
        stream_fake,
        options="--enable-tvm-ffi",
    )


def nvfp4_sf_buffer_numel(M: int, H: int) -> int:
    m_pad = (M + 127) // 128 * 128
    return m_pad * ((H // _SF_VEC + 3) // 4) * 4


def gemma_fused_add_rmsnorm_nvfp4_cute(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    global_scale: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: bool = False,
    fp4_out: torch.Tensor | None = None,
    sf_out: torch.Tensor | None = None,
):
    """residual = bf16(input + residual) in place; returns (fp4 [M, H/2] uint8,
    swizzled SF uint8 [ceil128(M) * ceil4(H/16)]) of
    NVFP4(bf16(rmsnorm(residual) * (weight + 1)), global_scale).
    input is read only. Contiguous 2-D BF16/FP16 tensors only."""
    assert input.dim() == 2 and input.shape == residual.shape
    assert input.is_contiguous() and residual.is_contiguous()
    assert weight.is_contiguous() and global_scale.dtype == torch.float32
    M, H = input.shape
    if fp4_out is None:
        fp4_out = torch.empty((M, H // 2), dtype=torch.uint8, device=input.device)
    if sf_out is None:
        sf_out = torch.empty((nvfp4_sf_buffer_numel(M, H),), dtype=torch.uint8, device=input.device)
    if M == 0:
        return fp4_out, sf_out
    kernel_obj, kernel = _get_compiled_fused_add_rmsnorm_nvfp4_kernel(
        _torch_dtype_to_str(input.dtype),
        H,
        1.0,
        bool(enable_pdl),
        get_sm_version(input.device),
    )
    rpb = kernel_obj.rows_per_block
    n_row_ctas = (M + rpb - 1) // rpb
    grid_x = n_row_ctas + (_N_PAD_CTAS if M % 128 != 0 else 0)
    kernel(input, residual, weight, fp4_out, sf_out, global_scale, M, n_row_ctas, grid_x, eps)
    return fp4_out, sf_out
