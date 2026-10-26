"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

FlashInfer's fused add + RMSNorm that also writes the NVFP4 of its output (MoE-entry dispatch input, opt-in).

Gate: ``TRTLLM_MOE_ENTRY_NORM_NVFP4`` (models/modeling_deepseekv3.py; default 0 = off, nothing here runs).

One kernel computes, in place and bit for bit::

    flashinfer.norm.fused_add_rmsnorm(input, residual, weight, eps)  # its CuTe DSL path
    q, sf = fp4_quantize(input, global_scale, 16, False, False)  # linear scale factors

The norm part is FlashInfer's public ``FusedAddRMSNormKernel.kernel`` body unchanged (flashinfer-python 0.6.18: same
tiling, thread-value layout, ``row_reduce_sum_multirow`` reduction, ``rsqrt`` and store order, same compile spec),
so ``input`` and ``residual`` match it exactly. The epilogue then quantizes the BF16-rounded normalized row with
TensorRT-LLM's NVFP4 activation quantizer arithmetic (``cvt_warp_fp16_to_fp4`` in
cpp/tensorrt_llm/kernels/quantization.cuh, the same PTX as modules/fused_ops/nvfp4_quantize_triton.py): the exact
max of each 16-value block's |x|, ``rcp.approx.ftz.f32``, ``mul.rn.f32``, ``cvt.rn.satfinite.e4m3x2.f32`` for the
scale factor and ``cvt.rn.satfinite.e2m1x2.f32`` for the values; linear (unswizzled) scale factors, as
``CuteDslFusedMoE.quantize_input`` produces before the all-to-all dispatch.

At H = 7168 each thread holds 7 vectors of 8 consecutive BF16 values; a 16-value block spans lanes 2j and 2j+1, so
its max is one butterfly shuffle. Prior art for the fusion pattern is FlashInfer
``add_rmsnorm_fp4quant`` and TensorRT-LLM #14848, neither bitwise with the two-kernel sequence this replaces.

Upstream source (v0.6.18, FusedAddRMSNormKernel.kernel):
https://github.com/flashinfer-ai/flashinfer/blob/v0.6.18/flashinfer/norm/kernels/fused_add_rmsnorm.py
"""
# SPDX-FileCopyrightText: Copyright (c) 2025 by FlashInfer team.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import math
from collections.abc import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
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
    rcp_approx_ftz,
    row_reduce_sum_multirow,
)

# =============================================================================
# PTX helpers (TensorRT-LLM cvt_warp_fp16_to_fp4 op for op)
# =============================================================================


@dsl_user_op
def _absmax8_f32(
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    v4: Float32,
    v5: Float32,
    v6: Float32,
    v7: Float32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    """max(|v0|, ..., |v7|), exact: max.xorsign.abs returns max(|a|, |b|) with a xor-ed sign, cleared at the end
    (a max of magnitudes does not depend on order)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(v).ir_value(loc=loc, ip=ip) for v in (v0, v1, v2, v3, v4, v5, v6, v7)],
            "{\n"
            ".reg .f32 m0, m1, m2, m3;\n"
            "max.xorsign.abs.f32 m0, $1, $2;\n"
            "max.xorsign.abs.f32 m1, $3, $4;\n"
            "max.xorsign.abs.f32 m2, $5, $6;\n"
            "max.xorsign.abs.f32 m3, $7, $8;\n"
            "max.xorsign.abs.f32 m0, m0, m1;\n"
            "max.xorsign.abs.f32 m2, m2, m3;\n"
            "max.xorsign.abs.f32 m0, m0, m2;\n"
            "abs.f32 $0, m0;\n"
            "}",
            "=f,f,f,f,f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _max_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
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
def _nvfp4_sf_bits(
    vmax: Float32, global_scale: Float32, rcp6: Float32, *, loc=None, ip=None
) -> Int32:
    """The E4M3 scale-factor byte: e4m3(gs * (vmax * rcp6)), round-to-nearest, satfinite; rcp6 = rcp.approx.ftz(6)."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(vmax).ir_value(loc=loc, ip=ip),
                Float32(global_scale).ir_value(loc=loc, ip=ip),
                Float32(rcp6).ir_value(loc=loc, ip=ip),
            ],
            "{\n"
            ".reg .f32 t;\n"
            ".reg .b16 p;\n"
            ".reg .b32 w;\n"
            "mul.rn.f32 t, $1, $3;\n"
            "mul.rn.f32 t, $2, t;\n"
            "cvt.rn.satfinite.e4m3x2.f32 p, t, t;\n"
            "cvt.u32.u16 w, p;\n"
            "and.b32 $0, w, 255;\n"
            "}",
            "=r,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _nvfp4_out_scale(
    sf_bits: Int32, vmax: Float32, rcp_global_scale: Float32, *, loc=None, ip=None
) -> Float32:
    """vmax != 0 ? rcp(float(e4m3) * rcp(gs)) : 0, with the exact E4M3 -> FP32 decode; rcp_global_scale = rcp(gs)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Int32(sf_bits).ir_value(loc=loc, ip=ip),
                Float32(vmax).ir_value(loc=loc, ip=ip),
                Float32(rcp_global_scale).ir_value(loc=loc, ip=ip),
            ],
            "{\n"
            ".reg .b16 e8, hl, hh;\n"
            ".reg .b32 h2;\n"
            ".reg .f32 sv, t;\n"
            ".reg .pred p;\n"
            "cvt.u16.u32 e8, $1;\n"
            "cvt.rn.f16x2.e4m3x2 h2, e8;\n"
            "mov.b32 {hl, hh}, h2;\n"
            "cvt.f32.f16 sv, hl;\n"
            "mul.rn.f32 t, sv, $3;\n"
            "rcp.approx.ftz.f32 t, t;\n"
            "setp.neu.f32 p, $2, 0f00000000;\n"
            "selp.f32 $0, t, 0f00000000, p;\n"
            "}",
            "=f,r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _nvfp4_cvt8_store(
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    v4: Float32,
    v5: Float32,
    v6: Float32,
    v7: Float32,
    out_scale: Float32,
    addr: Int64,
    *,
    loc=None,
    ip=None,
):
    """Scale 8 values (mul.rn), convert pairs to E2M1 (odd value high nibble) and store one 32-bit word."""
    llvm.inline_asm(
        None,
        [Float32(v).ir_value(loc=loc, ip=ip) for v in (v0, v1, v2, v3, v4, v5, v6, v7, out_scale)]
        + [Int64(addr).ir_value(loc=loc, ip=ip)],
        "{\n"
        ".reg .f32 s0, s1, s2, s3, s4, s5, s6, s7;\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        ".reg .b32 w;\n"
        "mul.rn.f32 s0, $0, $8;\n"
        "mul.rn.f32 s1, $1, $8;\n"
        "mul.rn.f32 s2, $2, $8;\n"
        "mul.rn.f32 s3, $3, $8;\n"
        "mul.rn.f32 s4, $4, $8;\n"
        "mul.rn.f32 s5, $5, $8;\n"
        "mul.rn.f32 s6, $6, $8;\n"
        "mul.rn.f32 s7, $7, $8;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0, s1, s0;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b1, s3, s2;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b2, s5, s4;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b3, s7, s6;\n"
        "mov.b32 w, {b0, b1, b2, b3};\n"
        "st.global.b32 [$9], w;\n"
        "}",
        "f,f,f,f,f,f,f,f,f,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _store_sf_if_even_lane(val: Int32, addr: Int64, lane: Int32, *, loc=None, ip=None):
    """Lanes 2j and 2j+1 hold the same scale-factor byte; the even lane stores it (predicated, no branch)."""
    llvm.inline_asm(
        None,
        [
            Int32(val).ir_value(loc=loc, ip=ip),
            Int64(addr).ir_value(loc=loc, ip=ip),
            Int32(lane).ir_value(loc=loc, ip=ip),
        ],
        "{\n"
        ".reg .pred p;\n"
        ".reg .b32 t;\n"
        ".reg .b16 b;\n"
        "and.b32 t, $2, 1;\n"
        "setp.eq.u32 p, t, 0;\n"
        "cvt.u16.u32 b, $0;\n"
        "@p st.global.b8 [$1], b;\n"
        "}",
        "r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


# =============================================================================
# Kernel
# =============================================================================


class FusedAddRMSNormNvfp4OutKernel(FusedAddRMSNormKernel):
    """FlashInfer's fused add + RMSNorm plus the BF16-rounded output's NVFP4 (linear scale factors)."""

    @staticmethod
    def supported(H: int, kernel_obj: "FusedAddRMSNormKernel") -> bool:
        # The epilogue's block math assumes one row per CTA, no cluster split and 8-value vectors, so that a 16-value
        # scale block is exactly lanes (2j, 2j+1) of one vector block.
        return (
            H % 16 == 0
            and kernel_obj.cluster_n == 1
            and kernel_obj.vec_size == 8
            and kernel_obj.rows_per_block == 1
            and kernel_obj.threads_per_row % 32 == 0
        )

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        M: Int64,
        eps: Float32,
        mS: cute.Tensor,
        mQ: cute.Tensor,
        mSF: cute.Tensor,
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

        cluster_n = self.cluster_n

        self.kernel(mX, mR, mW, M, eps, mS, mQ, mSF, enable_pdl, tv_layout, tiler_mn).launch(
            grid=[cute.ceil_div(M, self.rows_per_block), cluster_n, 1],
            block=[self.num_threads, 1, 1],
            cluster=[1, cluster_n, 1] if cutlass.const_expr(cluster_n > 1) else None,
            smem=self._smem_size_in_bytes(),
            stream=stream,
            use_pdl=enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        M: Int64,
        eps: Float32,
        mS: cute.Tensor,
        mQ: cute.Tensor,
        mSF: cute.Tensor,
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
        vec_size = self.vec_size
        num_vec_blocks = self.num_vec_blocks
        threads_per_row = tv_layout.shape[0][0]
        rows_per_block = tiler_mn[0]
        warps_per_row = max(threads_per_row // 32, 1)

        if cutlass.const_expr(cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
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

        if cutlass.const_expr(cluster_n == 1):
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, warps_per_row)),
                byte_alignment=4,
            )
            mbar_ptr = None
        else:
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, (warps_per_row, cluster_n))),
                byte_alignment=4,
            )
            mbar_ptr = smem.allocate_array(cutlass.Int64, num_elems=1)

        # ===== Initialize cluster =====
        if cutlass.const_expr(cluster_n > 1):
            if tidx == 0:
                cute.arch.mbarrier_init(mbar_ptr, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

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

        if cutlass.const_expr(cluster_n > 1):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier()

        # ===== Pass 2: Normalize and store output =====
        w = tXrW.load().to(Float32)
        y = h * rstd * (w + Float32(weight_bias))

        tXrO.store(y.to(mX.element_type))

        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tXgO, pred=tXpX)

        # ===== Epilogue: NVFP4 of the BF16-rounded output (linear SF) =====
        # Branch-free so the 7 vector blocks interleave: every row of the grid is < M (rows_per_block == 1 and the
        # grid is M rows, FusedAddRMSNormNvfp4OutKernel.supported), and the scale-factor store is predicated.
        tQ = cute.make_rmem_tensor(tXrX.shape, Float32)
        tQ.store(tXrO.load().to(Float32))
        global_scale = mS[0]
        rcp6 = rcp_approx_ftz(Float32(6.0))
        rcp_global_scale = rcp_approx_ftz(global_scale)
        lane_in_row = tidx % threads_per_row
        row_in_block = tidx // threads_per_row
        actual_row = bidx * rows_per_block + row_in_block
        vmax8 = []
        for v in cutlass.range_constexpr(num_vec_blocks):
            base = v * vec_size
            vmax8.append(
                _absmax8_f32(
                    tQ[base],
                    tQ[base + 1],
                    tQ[base + 2],
                    tQ[base + 3],
                    tQ[base + 4],
                    tQ[base + 5],
                    tQ[base + 6],
                    tQ[base + 7],
                )
            )
        sf_bits, out_scale = [], []
        for v in cutlass.range_constexpr(num_vec_blocks):
            # The 16-value scale block is this lane's 8 values and its neighbour's (lane ^ 1).
            vmax = _max_f32(vmax8[v], cute.arch.shuffle_sync_bfly(vmax8[v], offset=1))
            bits_v = _nvfp4_sf_bits(vmax, global_scale, rcp6)
            sf_bits.append(bits_v)
            out_scale.append(_nvfp4_out_scale(bits_v, vmax, rcp_global_scale))
        for v in cutlass.range_constexpr(num_vec_blocks):
            base = v * vec_size
            col = (
                cluster_y * self.cols_per_tile
                + lane_in_row * vec_size
                + v * threads_per_row * vec_size
            )
            _nvfp4_cvt8_store(
                tQ[base],
                tQ[base + 1],
                tQ[base + 2],
                tQ[base + 3],
                tQ[base + 4],
                tQ[base + 5],
                tQ[base + 6],
                tQ[base + 7],
                out_scale[v],
                get_ptr_as_int64(mQ, cute.crd2idx((Int32(actual_row), Int32(col // 2)), mQ.layout)),
            )
            _store_sf_if_even_lane(
                sf_bits[v],
                get_ptr_as_int64(
                    mSF, cute.crd2idx((Int32(actual_row), Int32(col // 16)), mSF.layout)
                ),
                lane_in_row,
            )

        # PDL: Signal dependent kernels (SM90+ only)
        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()


@functools.cache
def _get_compiled_fused_add_rmsnorm_nvfp4_kernel(
    dtype_str: str,
    hidden_size: int,
    weight_bias: float,
    enable_pdl: bool,
    sm_version: int,
) -> Callable[..., None]:
    dtype = get_cutlass_dtype(dtype_str)
    kernel_obj = FusedAddRMSNormNvfp4OutKernel(
        dtype, hidden_size, weight_bias, sm_version=sm_version
    )
    if not FusedAddRMSNormNvfp4OutKernel.supported(hidden_size, kernel_obj):
        raise ValueError(
            f"fused add-RMSNorm + NVFP4 does not support H={hidden_size} "
            f"(cluster_n={kernel_obj.cluster_n}, vec_size={kernel_obj.vec_size}, "
            f"rows_per_block={kernel_obj.rows_per_block})"
        )
    # Same compile spec as FlashInfer's contiguous fused_add_rmsnorm kernel (64-bit M, compact layouts).
    sym_m = cute.sym_int(64)
    tensor_align = math.gcd(128, hidden_size * (dtype.width // 8))
    x_fake = cute.runtime.make_fake_compact_tensor(
        dtype, (sym_m, hidden_size), stride_order=(1, 0), assumed_align=tensor_align
    )
    r_fake = cute.runtime.make_fake_compact_tensor(
        dtype, (sym_m, hidden_size), stride_order=(1, 0), assumed_align=tensor_align
    )
    w_fake = cute.runtime.make_fake_compact_tensor(dtype, (hidden_size,), assumed_align=16)
    s_fake = cute.runtime.make_fake_compact_tensor(Float32, (1,), assumed_align=4)
    q_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (sym_m, hidden_size // 2), stride_order=(1, 0), assumed_align=16
    )
    sf_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (sym_m, hidden_size // 16), stride_order=(1, 0), assumed_align=4
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel_obj,
        x_fake,
        r_fake,
        w_fake,
        Int64(1),
        Float32(1e-6),
        s_fake,
        q_fake,
        sf_fake,
        enable_pdl,
        stream_fake,
        options="--enable-tvm-ffi",
    )


def fused_add_rmsnorm_nvfp4_supported(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, global_scale: torch.Tensor
) -> bool:
    """Shapes this kernel reproduces bit for bit (else the caller keeps norm + quantize)."""
    return (
        input.is_cuda
        and input.dtype == torch.bfloat16
        and input.dim() == 2
        and input.shape[0] > 0
        and input.shape[1] == 7168
        and input.is_contiguous()
        and residual.dtype == input.dtype
        and residual.shape == input.shape
        and residual.is_contiguous()
        and residual.device == input.device
        and weight.dtype == input.dtype
        and weight.dim() == 1
        and weight.shape[0] == input.shape[1]
        and weight.is_contiguous()
        and weight.device == input.device
        and global_scale is not None
        and global_scale.is_cuda
        and global_scale.dtype == torch.float32
        and global_scale.numel() == 1
        and global_scale.is_contiguous()
        and input.shape[0] * input.shape[1] <= 2**31 - 1
    )


def fused_add_rmsnorm_nvfp4_cute(
    input: torch.Tensor,  # noqa: A002 - mirrors fused_add_rmsnorm
    residual: torch.Tensor,
    weight: torch.Tensor,
    global_scale: torch.Tensor,
    eps: float = 1e-6,
    weight_bias: float = 0.0,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """In place: residual += input; input = RMSNorm(residual) * weight (FlashInfer). Returns the NVFP4 of the new
    input as (q [M, H/2] uint8, sf [M * H/16] uint8, linear), the bytes fp4_quantize(input, global_scale, 16, False,
    False) returns."""
    num_rows, hidden_size = input.shape
    q = torch.empty((num_rows, hidden_size // 2), dtype=torch.uint8, device=input.device)
    sf = torch.empty((num_rows * (hidden_size // 16),), dtype=torch.uint8, device=input.device)
    kernel = _get_compiled_fused_add_rmsnorm_nvfp4_kernel(
        _torch_dtype_to_str(input.dtype),
        hidden_size,
        weight_bias,
        enable_pdl,
        get_sm_version(input.device),
    )
    # mS is compiled as a 1-D Tensor([1]); the NVFP4 MoE's fc31_input_scale is a 0-d Parameter
    # (NVFP4FusedMoEMethod.create_weights: torch.tensor(1.)). view(1) is metadata only (same storage, no copy, so a
    # captured graph still reads the live scale) and works for every contiguous one-element scale supported() admits.
    kernel(
        input,
        residual,
        weight,
        num_rows,
        eps,
        global_scale.view(1),
        q,
        sf.view(num_rows, hidden_size // 16),
    )
    return q, sf


__all__ = [
    "FusedAddRMSNormNvfp4OutKernel",
    "fused_add_rmsnorm_nvfp4_cute",
    "fused_add_rmsnorm_nvfp4_supported",
]
