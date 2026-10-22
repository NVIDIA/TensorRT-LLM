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
"""CuTe DSL kernel for the draft_front: RMSNorm of the needed source + TP-local packing.

The MTP draft layer feeds ``eh_proj`` with column chunk ``tp_rank`` of
``concat([enorm(embeds), hnorm(hidden)], dim=-1)``. This kernel computes only
the columns of that chunk. Each CTA handles ``rows_per_block`` rows of one
source segment: it reads the full source row (the RMS needs it), and stores the
normalized values of the segment's columns into the caller's concat-layout
buffer.

The per-row arithmetic deliberately mirrors the FlashInfer CuTe DSL
``RMSNormKernel`` that TensorRT-LLM's ``RMSNorm`` runs for BF16 inputs: the same
thread-value layout and vector width, the per-thread ``TensorSSA`` reduction of
``x * x``, a butterfly warp reduction with offsets 1, 2, 4, 8, 16, a
shared-memory combine across the warps of a row, ``rsqrt(sum / H + eps)`` with
the fast-math rsqrt, and ``y = x * rstd * (w + 0)`` rounded once to the output
dtype. With the same reduction tree the normalized values are bit-identical to
the native norm, which the unit tests check.
"""

import math
from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int64

_COPY_BITS = 128

# One segment: (source, first column, end column) in source coordinates. Source 0
# is the embedding half of the concatenation, source 1 the hidden half.
Segment = tuple[int, int, int]


def threads_per_row(hidden_size: int) -> int:
    """Threads that cooperate on one row; same table as the native norm kernel."""
    if hidden_size <= 64:
        return 8
    if hidden_size <= 128:
        return 16
    if hidden_size <= 3072:
        return 32
    if hidden_size <= 6144:
        return 64
    if hidden_size <= 16384:
        return 128
    return 256


def threads_per_block(hidden_size: int) -> int:
    """CTA size; same table as the native norm kernel."""
    return 128 if hidden_size <= 16384 else 256


@cute.jit
def _warp_reduce_sum(val: Float32, width: cutlass.Constexpr[int]) -> Float32:
    for i in cutlass.range_constexpr(int(math.log2(width))):
        val = val + cute.arch.shuffle_sync_bfly(val, offset=1 << i)
    return val


@cute.jit
def _row_reduce_sum(
    x_sq: cute.TensorSSA,
    row_threads: cutlass.Constexpr[int],
    reduction_buffer: cute.Tensor,
) -> Float32:
    """Sum ``x_sq`` over the threads of one row (the row's warps share an smem slot row)."""
    local_val = x_sq.reduce(cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0)
    warp_val = _warp_reduce_sum(local_val, min(row_threads, 32))
    result = warp_val
    if cutlass.const_expr(row_threads > 32):
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()
        warps_per_row = cute.size(reduction_buffer.shape[1])
        row_idx = warp_idx // warps_per_row
        col_idx = warp_idx % warps_per_row
        if lane_idx == 0:
            reduction_buffer[row_idx, col_idx] = warp_val
        cute.arch.barrier()
        block_val = Float32(0.0)
        if lane_idx < warps_per_row:
            block_val = reduction_buffer[row_idx, lane_idx]
        result = _warp_reduce_sum(block_val, 32)
    return result


@cute.jit
def _column_predicate(
    tXcX: cute.Tensor, begin: cutlass.Constexpr[int], end: cutlass.Constexpr[int]
) -> cute.Tensor:
    """Per-vector predicate ``begin <= column < end`` for a partitioned coordinate tensor."""
    pred = cute.make_rmem_tensor(
        cute.make_layout(
            (
                cute.size(tXcX, mode=[0, 1]),
                cute.size(tXcX, mode=[1]),
                cute.size(tXcX, mode=[2]),
            ),
            stride=(cute.size(tXcX, mode=[2]), 0, 1),
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(pred.shape[0]):
        for rest_k in cutlass.range_constexpr(pred.shape[2]):
            col = tXcX[(0, rest_v), 0, rest_k][1]
            if cutlass.const_expr(begin == 0):
                pred[rest_v, 0, rest_k] = cute.elem_less(col, end)
            else:
                pred[rest_v, 0, rest_k] = cute.elem_less(col, end) & cute.elem_less(begin - 1, col)
    return pred


class DraftFrontPackKernel:
    """Normalize the needed concat source and write this rank's ``eh_proj`` input columns.

    Args:
        dtype: element type of the activations, weights and output (BF16 in the
            supported domain).
        hidden_size: ``H``; each source row has ``H`` elements.
        segments: one or two ``(source, begin, end)`` triples; grid dimension y
            enumerates them.
        enable_pdl: wait on / trigger programmatic dependent launch like the
            native norm kernel.
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        hidden_size: int,
        segments: tuple[Segment, ...],
        enable_pdl: bool,
    ):
        if len(segments) not in (1, 2):
            raise ValueError(f"expected one or two segments, got {segments}")
        self.dtype = dtype
        self.H = hidden_size
        self.segments = segments
        self.enable_pdl = enable_pdl

        elem_bits = dtype.width
        max_vec_size = _COPY_BITS // elem_bits
        h_align = hidden_size & (-hidden_size)
        self.vec_size = min(h_align, max_vec_size)
        self.copy_bits = self.vec_size * elem_bits
        self.threads_per_row = threads_per_row(hidden_size)
        self.num_threads = threads_per_block(hidden_size)
        self.rows_per_block = self.num_threads // self.threads_per_row
        self.warps_per_row = max(self.threads_per_row // 32, 1)
        self.num_vec_blocks = max(
            1, (hidden_size // self.vec_size + self.threads_per_row - 1) // self.threads_per_row
        )
        self.cols_per_tile = self.vec_size * self.num_vec_blocks * self.threads_per_row

    def _tv_layout(self) -> cute.Layout:
        shape = (
            (self.threads_per_row, self.rows_per_block),
            (self.vec_size, self.num_vec_blocks),
        )
        stride = (
            (self.vec_size * self.rows_per_block, 1),
            (self.rows_per_block, self.rows_per_block * self.vec_size * self.threads_per_row),
        )
        return cute.make_layout(shape, stride=stride)

    def smem_bytes(self) -> int:
        return self.rows_per_block * self.warps_per_row * 4

    @cute.jit
    def __call__(
        self,
        mE: cute.Tensor,
        mHs: cute.Tensor,
        mWe: cute.Tensor,
        mWh: cute.Tensor,
        mOutE: cute.Tensor,
        mOutH: cute.Tensor,
        M: Int64,
        eps_e: Float32,
        eps_h: Float32,
        stream,
    ):
        tv_layout = self._tv_layout()
        tiler_mn = (self.rows_per_block, self.cols_per_tile)
        self.kernel(mE, mHs, mWe, mWh, mOutE, mOutH, M, eps_e, eps_h, tv_layout, tiler_mn).launch(
            grid=[cute.ceil_div(M, self.rows_per_block), len(self.segments), 1],
            block=[self.num_threads, 1, 1],
            smem=self.smem_bytes(),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mE: cute.Tensor,
        mHs: cute.Tensor,
        mWe: cute.Tensor,
        mWh: cute.Tensor,
        mOutE: cute.Tensor,
        mOutH: cute.Tensor,
        M: Int64,
        eps_e: Float32,
        eps_h: Float32,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, seg_idx, _ = cute.arch.block_idx()

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        smem = cutlass.utils.SmemAllocator()
        reduction_buffer = smem.allocate_tensor(
            Float32,
            cute.make_layout((self.rows_per_block, self.warps_per_row)),
            byte_alignment=4,
        )

        # Source 0 is the embedding half of the concatenation, source 1 the hidden half.
        operands = ((mE, mWe, mOutE, eps_e), (mHs, mWh, mOutH, eps_h))
        seg0 = self.segments[0]
        if cutlass.const_expr(len(self.segments) == 1):
            self._normalize_segment(
                *operands[seg0[0]],
                M,
                seg0[1],
                seg0[2],
                tidx,
                bidx,
                reduction_buffer,
                tv_layout,
                tiler_mn,
            )
        else:
            seg1 = self.segments[1]
            if seg_idx == 0:
                self._normalize_segment(
                    *operands[seg0[0]],
                    M,
                    seg0[1],
                    seg0[2],
                    tidx,
                    bidx,
                    reduction_buffer,
                    tv_layout,
                    tiler_mn,
                )
            else:
                self._normalize_segment(
                    *operands[seg1[0]],
                    M,
                    seg1[1],
                    seg1[2],
                    tidx,
                    bidx,
                    reduction_buffer,
                    tv_layout,
                    tiler_mn,
                )

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def _normalize_segment(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        eps: Float32,
        M: Int64,
        begin: cutlass.Constexpr[int],
        end: cutlass.Constexpr[int],
        tidx,
        bidx,
        reduction_buffer: cute.Tensor,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        H = self.H

        idX = cute.make_identity_tensor(mX.shape)
        gX = cute.local_tile(mX, tiler_mn, (bidx, 0))
        gY = cute.local_tile(mY, tiler_mn, (bidx, 0))
        cX = cute.local_tile(idX, tiler_mn, (bidx, 0))

        mW_expanded_layout = cute.prepend(mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,)))
        mW_2d = cute.make_tensor(mW.iterator, mW_expanded_layout)
        gW = cute.local_tile(mW_2d, tiler_mn, (0, 0))

        copy_atom_load = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mX.element_type, num_bits_per_copy=self.copy_bits
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mY.element_type, num_bits_per_copy=self.copy_bits
        )
        tiled_copy_load = cute.make_tiled_copy(copy_atom_load, tv_layout, tiler_mn)
        tiled_copy_w = cute.make_tiled_copy(copy_atom_load, tv_layout, tiler_mn)
        tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)

        thr_copy_x = tiled_copy_load.get_slice(tidx)
        thr_copy_w = tiled_copy_w.get_slice(tidx)
        thr_copy_o = tiled_copy_store.get_slice(tidx)

        tXgX = thr_copy_x.partition_S(gX)
        tXcX = thr_copy_x.partition_S(cX)
        tXrX = cute.make_fragment_like(tXgX)

        tWgW = thr_copy_w.partition_S(gW)
        tWrW = cute.make_fragment_like(tWgW)
        tXrW = thr_copy_x.retile(tWrW)

        tXgO = thr_copy_o.partition_D(gY)
        tXrO = cute.make_fragment_like(tXgO)

        # Full-row columns feed the reduction; only [begin, end) is weighted and stored.
        tXpX = _column_predicate(tXcX, 0, H)
        tWcW = thr_copy_w.partition_S(cX)
        tWpW = _column_predicate(tWcW, begin, end)
        tXpO = _column_predicate(tXcX, begin, end)
        row_coord = tXcX[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < M

        tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))
        if row_in_bounds:
            cute.copy(copy_atom_load, tXgX, tXrX, pred=tXpX)
        tWrW.store(cute.zeros_like(tWrW, dtype=mW.element_type))
        cute.copy(copy_atom_load, tWgW, tWrW, pred=tWpW)

        x = tXrX.load().to(Float32)
        x_sq = x * x
        sum_sq = _row_reduce_sum(x_sq, self.threads_per_row, reduction_buffer)

        mean_sq = sum_sq / Float32(H)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        w = tXrW.load().to(Float32)
        y = x * rstd * (w + Float32(0.0))
        tXrO.store(y.to(mY.element_type))

        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tXgO, pred=tXpO)


def _torch_to_cutlass_dtype(dtype: torch.dtype) -> cutlass.Numeric:
    mapping: dict[torch.dtype, cutlass.Numeric] = {
        torch.bfloat16: cutlass.BFloat16,
        torch.float16: cutlass.Float16,
    }
    if dtype not in mapping:
        raise ValueError(f"unsupported dtype {dtype}")
    return mapping[dtype]


_COMPILED: dict[tuple, Callable] = {}


def get_compiled_kernel(
    dtype: torch.dtype,
    hidden_size: int,
    segments: tuple[Segment, ...],
    enable_pdl: bool,
    allow_compile: bool = True,
) -> Callable | None:
    """Return the draft-front pack kernel, compiling it once per process and configuration.

    The returned callable takes torch tensors through TVM-FFI and launches on the
    current torch stream:
    ``fn(embeds[M,H], hidden[M,H], enorm_w[H], hnorm_w[H], out_e[M,H], out_h[M,H], M, eps_e,
    eps_h)``. Every ``[M, H]`` operand needs unit column stride and a row stride
    divisible by the vector width.

    Args:
        allow_compile: when False and the kernel is not compiled yet, return
            ``None`` instead of compiling (used during CUDA-graph capture).
    """
    key = (dtype, hidden_size, segments, enable_pdl)
    kernel = _COMPILED.get(key)
    if kernel is None and allow_compile:
        kernel = _compile_kernel(*key)
        _COMPILED[key] = kernel
    return kernel


def _compile_kernel(
    dtype: torch.dtype, hidden_size: int, segments: tuple[Segment, ...], enable_pdl: bool
) -> Callable:
    cutlass_dtype = _torch_to_cutlass_dtype(dtype)
    kernel_obj = DraftFrontPackKernel(cutlass_dtype, hidden_size, segments, enable_pdl)
    vec = kernel_obj.vec_size
    sym_m = cute.sym_int(64)

    def row_major_fake() -> cute.Tensor:
        return cute.runtime.make_fake_tensor(
            cutlass_dtype,
            (sym_m, hidden_size),
            (cute.sym_int64(divisibility=vec), 1),
            assumed_align=16,
        )

    def weight_fake() -> cute.Tensor:
        return cute.runtime.make_fake_compact_tensor(
            cutlass_dtype, (hidden_size,), assumed_align=16
        )

    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel_obj,
        row_major_fake(),
        row_major_fake(),
        weight_fake(),
        weight_fake(),
        row_major_fake(),
        row_major_fake(),
        Int64(1),
        Float32(1e-6),
        Float32(1e-6),
        stream_fake,
        options="--enable-tvm-ffi",
    )


__all__ = [
    "DraftFrontPackKernel",
    "Segment",
    "get_compiled_kernel",
    "threads_per_block",
    "threads_per_row",
]
