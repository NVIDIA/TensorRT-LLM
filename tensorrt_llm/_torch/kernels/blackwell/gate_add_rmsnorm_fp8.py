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
"""Shared-expert gate + residual add + RMSNorm + static E4M3 quant in one launch (CuTe DSL).

Bitwise equal to the three launches it replaces: ``fused_sigmoid_gate_mul_add`` (``x = bf16(f + sigmoid(g) * s)``),
FlashInfer's fused add + RMSNorm (``FusedAddRMSNormKernel``, subclassed here for its layout and row reduction) and
``static_quantize_e4m3_per_tensor`` of the BF16 norm output. One CTA covers whole rows (``supports_hidden``).
"""

import functools
import math
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from flashinfer.norm.kernels.fused_add_rmsnorm import FusedAddRMSNormKernel
from flashinfer.norm.kernels.rmsnorm import RMSNormKernel
from flashinfer.norm.utils import (
    get_ptr_as_int64,
    get_sm_version,
    predicate_k,
    row_reduce_sum_multirow,
)

from .low_m_mma_gemm import _rcp_rn
from .mnnvl_allreduce_norm import (
    _e4m3x2,
    _fma,
    _hi,
    _ld_u16,
    _lo,
    _pack_bf16,
    _sigmoid_triton,
    _st_v2,
)


class GateAddRMSNormFp8Kernel(FusedAddRMSNormKernel):
    """``h = bf16(f + sigmoid(g) * s) + r`` (FP32), ``r <- bf16(h)``, ``y = bf16(h * rstd * (w + weight_bias))``,
    ``q = e4m3(y / scale)``; ``f`` is overwritten by ``y`` when ``write_y``."""

    def __init__(
        self,
        hidden: int,
        weight_bias: float = 1.0,
        write_y: bool = True,
        sm_version: Optional[int] = None,
    ):
        super().__init__(cutlass.BFloat16, hidden, weight_bias, sm_version=sm_version)
        if self.cluster_n != 1 or self.vec_size != 8 or self.cols_per_tile != hidden:
            raise ValueError(f"hidden={hidden} does not fit one CTA per row group")
        self.write_y = write_y

    def _smem_size_in_bytes(self) -> int:
        # Rows load straight into registers; only the row reduction is shared.
        return self.rows_per_block * self.warps_per_row * 4

    @cute.jit
    def __call__(
        self,
        mF: cute.Tensor,
        mS: cute.Tensor,
        mG: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        mQ: cute.Tensor,
        mScale: cute.Tensor,
        M: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        tv_shape, tv_stride = RMSNormKernel._make_tv_layout(
            self.threads_per_row, self.rows_per_block, self.vec_size, self.num_vec_blocks
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (self.rows_per_block, self.cols_per_tile)
        self.kernel(mF, mS, mG, mR, mW, mQ, mScale, M, eps, enable_pdl, tv_layout, tiler_mn).launch(
            grid=[cute.ceil_div(M, self.rows_per_block), 1, 1],
            block=[self.num_threads, 1, 1],
            smem=self._smem_size_in_bytes(),
            stream=stream,
            use_pdl=enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mF: cute.Tensor,
        mS: cute.Tensor,
        mG: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        mQ: cute.Tensor,
        mScale: cute.Tensor,
        M: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        H = self.H
        threads_per_row = tv_layout.shape[0][0]
        rows_per_block = tiler_mn[0]
        warps_per_row = max(threads_per_row // 32, 1)
        vec = self.vec_size

        smem = cutlass.utils.SmemAllocator()
        reduction_buffer = smem.allocate_tensor(
            Float32, cute.make_layout((rows_per_block, warps_per_row)), byte_alignment=4
        )

        idX = cute.make_identity_tensor(mF.shape)
        gF = cute.local_tile(mF, tiler_mn, (bidx, 0))
        gS = cute.local_tile(mS, tiler_mn, (bidx, 0))
        gR = cute.local_tile(mR, tiler_mn, (bidx, 0))
        cX = cute.local_tile(idX, tiler_mn, (bidx, 0))
        mW_2d = cute.make_tensor(
            mW.iterator, cute.prepend(mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,)))
        )
        gW = cute.local_tile(mW_2d, tiler_mn, (0, 0))

        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mF.element_type, num_bits_per_copy=self.copy_bits
        )
        tiled_copy = cute.make_tiled_copy(copy_atom, tv_layout, tiler_mn)
        thr_copy = tiled_copy.get_slice(tidx)

        tFgF = thr_copy.partition_S(gF)
        tXcX = thr_copy.partition_S(cX)
        tXrX = cute.make_fragment_like(tFgF)
        tSgS = thr_copy.partition_S(gS)
        tSrS = cute.make_fragment_like(tSgS)
        tRgR = thr_copy.partition_S(gR)
        tRrR = cute.make_fragment_like(tRgR)
        tWgW = thr_copy.partition_S(gW)
        tWrW = cute.make_fragment_like(tWgW)
        tXgO = thr_copy.partition_D(gF)
        tRgO = thr_copy.partition_D(gR)
        tXrO = cute.make_fragment_like(tXgO)

        tXpX = predicate_k(tXcX, limit=H)
        row = tXcX[(0, 0), 0, 0][0]
        row_in_bounds = row < M
        t0 = tidx % threads_per_row

        # The norm weight and the quant scale are model weights: load them before the PDL wait.
        cute.copy(copy_atom, tWgW, tWrW, pred=tXpX)
        factor = _rcp_rn(mScale[0])

        if enable_pdl:
            cute.arch.griddepcontrol_wait()

        tXrX.store(cute.zeros_like(tXrX, dtype=mF.element_type))
        tSrS.store(cute.zeros_like(tSrS, dtype=mS.element_type))
        tRrR.store(cute.zeros_like(tRrR, dtype=mR.element_type))
        g = Float32(0.0)
        if row_in_bounds:
            cute.copy(copy_atom, tFgF, tXrX, pred=tXpX)
            cute.copy(copy_atom, tSgS, tSrS, pred=tXpX)
            cute.copy(copy_atom, tRgR, tRrR, pred=tXpX)
            g = _lo(_ld_u16(get_ptr_as_int64(mG, Int64(row))))
        s = _sigmoid_triton(g)

        # fused_sigmoid_gate_mul_add: x = bf16(fma(sigmoid(g), s, f)), two BF16 per word.
        wF = cute.recast_tensor(tXrX, Int32)
        wS = cute.recast_tensor(tSrS, Int32)
        for i in cutlass.range_constexpr(cute.size(wF)):
            fw = wF[i]
            sw = wS[i]
            wF[i] = _pack_bf16(_fma(s, _lo(sw), _lo(fw)), _fma(s, _hi(sw), _hi(fw)))

        # FusedAddRMSNormKernel's arithmetic from here on.
        h = tXrX.load().to(Float32) + tRrR.load().to(Float32)
        tXrO.store(h.to(mR.element_type))
        if row_in_bounds:
            cute.copy(copy_atom, tXrO, tRgO, pred=tXpX)
        sum_sq = row_reduce_sum_multirow(h * h, threads_per_row, reduction_buffer, None, 1)
        rstd = cute.math.rsqrt(sum_sq / Float32(H) + eps, fastmath=True)
        cute.arch.barrier()
        w = thr_copy.retile(tWrW).load().to(Float32)
        y = h * rstd * (w + Float32(self.weight_bias))
        tXrO.store(y.to(mF.element_type))
        if cutlass.const_expr(self.write_y):
            if row_in_bounds:
                cute.copy(copy_atom, tXrO, tXgO, pred=tXpX)

        # E4M3 of the BF16 y: fragment words 4j .. 4j + 3 hold columns vec * (t0 + threads_per_row * j) .. + 7.
        if row_in_bounds:
            wY = cute.recast_tensor(tXrO, Int32)
            q_base = get_ptr_as_int64(mQ, Int64(row) * Int64(H) + Int64(vec * t0))
            for j in cutlass.range_constexpr(self.num_vec_blocks):
                y0 = wY[4 * j]
                y1 = wY[4 * j + 1]
                y2 = wY[4 * j + 2]
                y3 = wY[4 * j + 3]
                q0 = _e4m3x2(_lo(y0) * factor, _hi(y0) * factor)
                q1 = _e4m3x2(_lo(y1) * factor, _hi(y1) * factor)
                q2 = _e4m3x2(_lo(y2) * factor, _hi(y2) * factor)
                q3 = _e4m3x2(_lo(y3) * factor, _hi(y3) * factor)
                _st_v2(q_base + Int64(j * vec * threads_per_row), q0 | (q1 << 16), q2 | (q3 << 16))

        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()


@functools.lru_cache(maxsize=None)
def supports_hidden(hidden: int) -> bool:
    """Whether one CTA covers a row of ``hidden`` BF16 columns on the current device."""
    try:
        GateAddRMSNormFp8Kernel(hidden, sm_version=get_sm_version(torch.device("cuda")))
    except ValueError:
        return False
    return True


@functools.lru_cache(maxsize=None)
def _compiled(hidden: int, weight_bias: float, write_y: bool, enable_pdl: bool, sm_version: int):
    kernel = GateAddRMSNormFp8Kernel(hidden, weight_bias, write_y, sm_version=sm_version)
    mk = cute.runtime.make_fake_compact_tensor
    m = cute.sym_int(64)
    align = math.gcd(128, hidden * 2)

    def rows(dtype, row_align):
        return mk(dtype, (m, hidden), stride_order=(1, 0), assumed_align=row_align)

    return cute.compile(
        kernel,
        rows(cutlass.BFloat16, align),
        rows(cutlass.BFloat16, align),
        mk(cutlass.BFloat16, (m,), assumed_align=2),
        rows(cutlass.BFloat16, align),
        mk(cutlass.BFloat16, (hidden,), assumed_align=16),
        rows(cutlass.Float8E4M3FN, math.gcd(128, hidden)),
        mk(Float32, (1,), assumed_align=4),
        Int64(1),
        Float32(1e-6),
        enable_pdl,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def gate_add_rmsnorm_fp8(
    routed: torch.Tensor,
    shared: torch.Tensor,
    gate_logits: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scale: torch.Tensor,
    write_y: bool = True,
    weight_bias: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``routed`` [M, H] BF16 (overwritten by the norm output when ``write_y``), ``shared`` [M, H] BF16,
    ``gate_logits`` [M, 1] BF16, ``residual`` [M, H] BF16 (updated in place), ``weight`` [H] BF16 (the norm adds
    ``weight_bias`` in FP32), ``scale`` the static E4M3 scale. Returns (``routed``, E4M3 [M, H])."""
    from ...flashinfer_utils import get_env_enable_pdl

    m, hidden = routed.shape
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=routed.device)
    if m == 0:
        return routed, q
    fn = _compiled(
        hidden,
        float(weight_bias),
        bool(write_y),
        bool(get_env_enable_pdl()),
        get_sm_version(routed.device),
    )
    fn(
        routed,
        shared,
        gate_logits.reshape(-1),
        residual,
        weight,
        q,
        scale.reshape(1),
        m,
        float(eps),
    )
    return routed, q
