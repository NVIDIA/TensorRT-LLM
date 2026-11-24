# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MegaMoE's top-k combine reduce and the latent RMSNorm in one CuTe DSL kernel (K3.1 Rubin).

On a generation server MegaMoE returns each row as ``top_k`` BF16 expert outputs (the combine
plane, ``[rows, top_k, hidden]``) and the FP32 routing weights; the engine's ``TopkReduce`` sums
them into a BF16 row and FlashInfer's RMSNorm normalizes it before ``latent_up``. This kernel
writes the normalized row directly, with the bits of those two kernels:

* the reduce is ``TopkReduce``'s per-element FP32 chain: ``acc = v[0] * s[0]``, then
  ``acc = fma(v[k], s[k], acc)`` for ``k = 1 .. top_k - 1``, rounded once to BF16;
* the sum of squares is FlashInfer's ``RMSNormKernel`` at this width: the same thread-value layout
  (``threads_per_row`` threads of a row, ``vec_size`` values per vector, ``num_vec_blocks``
  vectors per thread, two rows per 128 threads), the same per-thread fragment reduction, the
  32-lane butterfly and the two-warp combine through shared memory, ``sum / hidden``,
  ``rsqrt.approx`` of ``mean + eps``, and ``x * rstd * (w + 0)`` rounded to BF16.

A CTA holds ``rows_per_cta`` (1 or 2) rows, ``hidden / 8`` threads per row: each thread reduces
eight consecutive values of its row into the CTA's shared row tile (FlashInfer's ``[2, hidden]``
tile), the first 128 threads take FlashInfer's sum of squares over that tile, and each thread
normalizes and stores its eight values. With ``pdl`` the launch is programmatic: each CTA loads its
norm weights (which no kernel writes), waits for the previous grid, then releases its own
dependents at once; a programmatic dependent still waits for this grid to finish before it reads
the output.
"""

from __future__ import annotations

import functools
import operator

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32
from cutlass.cute.runtime import from_dlpack

#: Values one thread reduces and normalizes (one 16-byte BF16 vector).
VEC = 8
#: FlashInfer's 128-bit vector width, in bits.
_COPY_BITS = 128
#: Rows of FlashInfer's tile per 128-thread block at the widths served here.
_FI_ROWS_PER_BLOCK = 2
#: Most threads a CTA may launch with.
_MAX_THREADS = 1024


def flashinfer_threads_per_row(hidden: int) -> int:
    """FlashInfer ``RMSNormKernel._compute_threads_per_row``: the threads that sum one row."""
    if hidden <= 64:
        return 8
    if hidden <= 128:
        return 16
    if hidden <= 3072:
        return 32
    if hidden <= 6144:
        return 64
    if hidden <= 16384:
        return 128
    return 256


def flashinfer_geometry(hidden: int) -> tuple[int, int, int]:
    """FlashInfer's ``(threads_per_row, vec_size, num_vec_blocks)`` of a BF16 row of ``hidden``
    on one CTA (no cluster), or ValueError where this kernel does not reproduce it: the row must
    fill its tile exactly and two rows must share a 128-thread block."""
    threads_per_row = flashinfer_threads_per_row(hidden)
    vec_size = min(hidden & -hidden, _COPY_BITS // 16)
    num_vec_blocks = max(1, (hidden // vec_size + threads_per_row - 1) // threads_per_row)
    if vec_size != VEC or vec_size * num_vec_blocks * threads_per_row != hidden:
        raise ValueError(
            f"hidden {hidden}: FlashInfer's row tile does not cover it in 16-byte vectors"
        )
    if 128 // threads_per_row != _FI_ROWS_PER_BLOCK:
        raise ValueError(
            f"hidden {hidden}: FlashInfer puts {128 // threads_per_row} rows per block"
        )
    return threads_per_row, vec_size, num_vec_blocks


def rows_per_cta_for(rows: int, sms: int) -> int:
    """Rows per CTA: one up to two CTAs per SM, else two."""
    return 1 if rows <= 2 * sms else 2


@cute.jit
def _warp_sum(val: Float32) -> Float32:
    """FlashInfer's ``warp_reduce(val, add, 32)``: the butterfly over offsets 1, 2, 4, 8, 16."""
    for i in cutlass.range_constexpr(5):
        val = operator.add(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


class MoeBackNormKernel:
    """``out[r] = rmsnorm(sum_k scores[r, k] * plane[r, k], weight, eps)`` in BF16."""

    def __init__(self, hidden: int, top_k: int, rows_per_cta: int, pdl: bool):
        if rows_per_cta not in (1, 2):
            raise ValueError(f"rows_per_cta {rows_per_cta}: 1 or 2")
        self.hidden = hidden
        self.top_k = top_k
        self.rows_per_cta = rows_per_cta
        self.pdl = pdl
        self.fi_threads_per_row, self.fi_vec, self.fi_blocks = flashinfer_geometry(hidden)
        self.row_threads = hidden // VEC
        if self.row_threads % 32:
            raise ValueError(
                f"hidden {hidden}: {self.row_threads} threads per row is not whole warps"
            )
        self.threads = self.row_threads * rows_per_cta
        if self.threads > _MAX_THREADS or self.row_threads < 128:
            raise ValueError(
                f"hidden {hidden} x {rows_per_cta} rows: {self.threads} threads per CTA"
            )
        self.fi_warps_per_row = self.fi_threads_per_row // 32

    def smem_bytes(self) -> int:
        tile = _FI_ROWS_PER_BLOCK * self.hidden * 2
        return tile + _FI_ROWS_PER_BLOCK * self.fi_warps_per_row * 4 + 64

    @cute.jit
    def __call__(
        self,
        plane: cute.Tensor,  # [>= rows, top_k, hidden] BF16, hidden contiguous
        scores: cute.Tensor,  # [>= rows, top_k] FP32
        weight: cute.Tensor,  # [hidden] BF16
        out: cute.Tensor,  # [rows, hidden] BF16, hidden contiguous
        eps: Float32,
        stream: cuda.CUstream,
    ):
        rows = out.shape[0]
        self.kernel(plane, scores, weight, out, eps).launch(
            grid=[(rows + self.rows_per_cta - 1) // self.rows_per_cta, 1, 1],
            block=[self.threads, 1, 1],
            smem=self.smem_bytes(),
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.kernel
    def kernel(
        self,
        plane: cute.Tensor,
        scores: cute.Tensor,
        weight: cute.Tensor,
        out: cute.Tensor,
        eps: Float32,
    ):
        hidden: cutlass.Constexpr[int] = self.hidden
        top_k: cutlass.Constexpr[int] = self.top_k
        row_threads: cutlass.Constexpr[int] = self.row_threads
        tidx = cute.arch.thread_idx()[0]
        group = tidx // row_threads
        tile = tidx - group * row_threads
        row = cute.arch.block_idx()[0] * self.rows_per_cta + group
        live = row < out.shape[0]

        # The norm weights of this thread's eight values: no kernel writes them.
        w_r = cute.make_rmem_tensor((VEC,), cutlass.BFloat16)
        cute.autovec_copy(cute.zipped_divide(weight, (VEC,))[(None,), (tile,)], w_r)
        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

        smem = cutlass.utils.SmemAllocator()
        # FlashInfer's row tile: [2, hidden], row-major.
        tiler = (_FI_ROWS_PER_BLOCK, hidden)
        s_x = smem.allocate_tensor(
            cutlass.BFloat16, cute.make_ordered_layout(tiler, order=(1, 0)), byte_alignment=16
        )
        red = smem.allocate_tensor(
            Float32, cute.make_layout((_FI_ROWS_PER_BLOCK, self.fi_warps_per_row)), byte_alignment=4
        )

        # ---- 1. TopkReduce's FP32 chain over this thread's eight values, rounded to BF16.
        acc = cute.make_rmem_tensor((VEC,), Float32)
        for i in cutlass.range_constexpr(VEC):
            acc[i] = Float32(0.0)
        if live:
            score = cute.make_rmem_tensor((top_k,), Float32)
            cute.autovec_copy(scores[row, None], score)
            load = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128
            )
            src = cute.zipped_divide(plane[row, None, None], (top_k, VEC))[(None, None), (0, tile)]
            # Every term's 16-byte load is issued before the first product.
            terms = []
            for k in cutlass.range_constexpr(top_k):
                term = cute.make_rmem_tensor((VEC,), cutlass.BFloat16)
                cute.copy(load, src[k, None], term)
                terms.append(term)
            for k in cutlass.range_constexpr(top_k):
                s_pair = (score[k], score[k])
                for i in cutlass.range_constexpr(0, VEC, 2):
                    v_pair = (Float32(terms[k][i]), Float32(terms[k][i + 1]))
                    if cutlass.const_expr(k == 0):
                        acc[i], acc[i + 1] = cute.arch.mul_packed_f32x2(v_pair, s_pair)
                    else:
                        acc[i], acc[i + 1] = cute.arch.fma_packed_f32x2(
                            v_pair, s_pair, (acc[i], acc[i + 1])
                        )
        x_bf16 = cute.make_rmem_tensor((VEC,), cutlass.BFloat16)
        x_bf16.store(acc.load().to(cutlass.BFloat16))
        dst = cute.zipped_divide(s_x[group, None], (VEC,))[(None,), (tile,)]
        cute.autovec_copy(x_bf16, dst)
        cute.arch.barrier()

        # ---- 2. FlashInfer's sum of squares over the tile: the first 128 threads, its layout.
        tv = cute.make_layout(
            ((self.fi_threads_per_row, _FI_ROWS_PER_BLOCK), (self.fi_vec, self.fi_blocks)),
            stride=(
                (self.fi_vec * _FI_ROWS_PER_BLOCK, 1),
                (_FI_ROWS_PER_BLOCK, _FI_ROWS_PER_BLOCK * self.fi_vec * self.fi_threads_per_row),
            ),
        )
        fi_copy = cute.make_tiled_copy(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=_COPY_BITS
            ),
            tv,
            tiler,
        )
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        if tidx < self.fi_threads_per_row * _FI_ROWS_PER_BLOCK:
            t_s = fi_copy.get_slice(tidx).partition_S(s_x)
            t_r = cute.make_fragment_like(t_s)
            cute.autovec_copy(t_s, t_r)
            xf = t_r.load().to(Float32)
            x_sq = xf * xf
            local = x_sq.reduce(cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0)
            warp_val = _warp_sum(local)
            if lane == 0:
                red[warp // self.fi_warps_per_row, warp % self.fi_warps_per_row] = warp_val
        cute.arch.barrier()

        # Every warp of a row combines the row's warp sums as FlashInfer's row warps do.
        total = Float32(0.0)
        if lane < self.fi_warps_per_row:
            total = red[group, lane]
        sum_sq = _warp_sum(total)
        mean_sq = sum_sq / Float32(hidden)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        # ---- 3. x * rstd * (w + 0), this thread's eight values.
        if live:
            x = x_bf16.load().to(Float32)
            w = w_r.load().to(Float32)
            y = x * rstd * (w + Float32(0.0))
            y_r = cute.make_rmem_tensor((VEC,), cutlass.BFloat16)
            y_r.store(y.to(cutlass.BFloat16))
            cute.autovec_copy(y_r, cute.zipped_divide(out[row, None], (VEC,))[(None,), (tile,)])


def _rows_dynamic(t: torch.Tensor):
    """``t`` (dense) as a CuTe tensor whose leading extent is a runtime value."""
    return from_dlpack(t, assumed_align=16).mark_compact_shape_dynamic(
        mode=0, stride_order=t.dim_order()
    )


@functools.cache
def _compiled(hidden: int, top_k: int, rows_per_cta: int, pdl: bool, device: int):
    kernel = MoeBackNormKernel(hidden, top_k, rows_per_cta, pdl)
    dev = torch.device("cuda", device)
    plane = torch.empty(2, top_k, hidden, dtype=torch.bfloat16, device=dev)
    scores = torch.empty(2, top_k, dtype=torch.float32, device=dev)
    weight = torch.empty(hidden, dtype=torch.bfloat16, device=dev)
    out = torch.empty(2, hidden, dtype=torch.bfloat16, device=dev)
    return cute.compile(
        kernel,
        _rows_dynamic(plane),
        _rows_dynamic(scores),
        from_dlpack(weight, assumed_align=16),
        _rows_dynamic(out),
        Float32(1e-6),
        cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream),
    )


def check_operands(
    plane: torch.Tensor, scores: torch.Tensor, weight: torch.Tensor, out: torch.Tensor
) -> None:
    """Raise ValueError unless the operands are what the kernel reads and writes: ``plane``
    ``[T, top_k, hidden]`` BF16 with a dense ``[top_k, hidden]`` row, ``scores`` ``[T, top_k]``
    FP32 with unit inner stride, ``weight`` ``[hidden]`` BF16 dense, ``out`` ``[rows, hidden]``
    BF16 dense with ``rows <= T``, all on one CUDA device and 16-byte aligned."""
    if plane.dim() != 3 or plane.dtype != torch.bfloat16:
        raise ValueError(f"plane {tuple(plane.shape)} {plane.dtype}: a BF16 [T, top_k, hidden]")
    t, top_k, hidden = plane.shape
    if plane.stride(2) != 1 or plane.stride(1) != hidden:
        raise ValueError(
            f"plane strides {plane.stride()}: each row's [top_k, hidden] must be dense"
        )
    if scores.dtype != torch.float32 or scores.dim() != 2 or scores.shape[1] != top_k:
        raise ValueError(f"scores {tuple(scores.shape)} {scores.dtype}: FP32 [T, {top_k}]")
    if scores.stride(1) != 1:
        raise ValueError(f"scores strides {scores.stride()}: each row's weights must be dense")
    if (
        weight.dtype != torch.bfloat16
        or tuple(weight.shape) != (hidden,)
        or not weight.is_contiguous()
    ):
        raise ValueError(f"weight {tuple(weight.shape)} {weight.dtype}: dense BF16 [{hidden}]")
    if (
        out.dtype != torch.bfloat16
        or out.dim() != 2
        or out.shape[1] != hidden
        or not out.is_contiguous()
    ):
        raise ValueError(f"out {tuple(out.shape)} {out.dtype}: dense BF16 [rows, {hidden}]")
    if out.shape[0] > min(t, scores.shape[0]):
        raise ValueError(
            f"out has {out.shape[0]} rows, the plane {t} and the scores {scores.shape[0]}"
        )
    tensors = (plane, scores, weight, out)
    if any(x.device != out.device or not x.is_cuda for x in tensors):
        raise ValueError("plane, scores, weight and out must be on one CUDA device")
    if any(x.data_ptr() % 16 for x in tensors):
        raise ValueError("plane, scores, weight and out must be 16-byte aligned")


def run_moe_back_norm(
    plane: torch.Tensor,
    scores: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: torch.Tensor,
    *,
    pdl: bool = True,
    rows_per_cta: int | None = None,
) -> torch.Tensor:
    """``out[r] = rmsnorm(sum_k scores[r, k] * plane[r, k], weight, eps)`` for ``r < out.shape[0]``,
    on the current stream; the bits of ``TopkReduce`` (BF16 combine, FP32 scores) followed by
    FlashInfer's RMSNorm. ``pdl``: a programmatic launch that waits for the previous grid and
    releases its dependents at once. ``rows_per_cta``: None picks :func:`rows_per_cta_for`.

    Returns:
        ``out``.
    """
    check_operands(plane, scores, weight, out)
    rows = out.shape[0]
    if rows == 0:
        return out
    dev = out.device
    if rows_per_cta is None:
        rows_per_cta = rows_per_cta_for(
            rows, torch.cuda.get_device_properties(dev).multi_processor_count
        )
    compiled = _compiled(plane.shape[2], plane.shape[1], rows_per_cta, pdl, dev.index)
    compiled(
        _rows_dynamic(plane),
        _rows_dynamic(scores),
        from_dlpack(weight, assumed_align=16),
        _rows_dynamic(out),
        Float32(eps),
        cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream),
    )
    return out


__all__ = [
    "VEC",
    "MoeBackNormKernel",
    "check_operands",
    "flashinfer_geometry",
    "flashinfer_threads_per_row",
    "rows_per_cta_for",
    "run_moe_back_norm",
]
