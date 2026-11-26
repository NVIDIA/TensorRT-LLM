# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Row-count-agnostic GLM-5.2 MoE front-end pieces for row counts the fused
decode kernels do not cover (the MTP module's context steps).

* ``sigmoid_topk``: the ungrouped ``noaux_tc`` routing of GLM-5.2 -- 256
  experts, top 8::

      raw  = sigmoid(logits)                  (as 0.5 * tanh(0.5 x) + 0.5)
      ids  = top-8 of raw + bias              (descending; equal -> smaller id)
      w    = raw[ids] * scale / (sum(raw[ids]) + 1e-20)

  One warp per row, eight experts per lane; eight rounds of a warp-wide
  ``(score, -id)`` max pick the experts in descending order.  ``ids`` match
  the reference exactly; the weights' only freedom is the order of the
  8-term sum.
* ``swiglu``: ``out[r, c] = silu(src[r, c]) * src[r, W + c]`` for ``c < W``
  (gate half first, the order FlashInfer's ``silu_and_mul`` reads), FP32
  math, BF16 out.  One CTA of 256 threads per 2048 columns of a row.

"""

from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

EXPERTS = 256
TOP_K = 8
_WARP_SIZE = 32
_PER_LANE = EXPERTS // _WARP_SIZE
_ROWS_PER_CTA = 8
_THREADS = _WARP_SIZE * _ROWS_PER_CTA
_SWIGLU_THREADS = 256
_SWIGLU_PER_THREAD = 8
_SWIGLU_COLS = _SWIGLU_THREADS * _SWIGLU_PER_THREAD
_NEG_INF = float("-inf")

CompiledKernel = Callable[..., object]
_COMPILED: dict[str, CompiledKernel] = {}


@cute.kernel
def _topk_kernel(
    logits: cute.Tensor,
    bias: cute.Tensor,
    weights: cute.Tensor,
    ids: cute.Tensor,
    rows: cutlass.Int32,
    scale: cutlass.Float32,
) -> None:
    bidx, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx % _WARP_SIZE
    row = bidx * _ROWS_PER_CTA + tidx // _WARP_SIZE
    if row < rows:
        raw = cute.make_rmem_tensor((_PER_LANE,), cutlass.Float32)
        sel = cute.make_rmem_tensor((_PER_LANE,), cutlass.Float32)
        for k in cutlass.range_constexpr(_PER_LANE):
            e = lane * _PER_LANE + k
            x = logits[row, e]
            r = cutlass.Float32(0.5) * cute.math.tanh(cutlass.Float32(0.5) * x) + cutlass.Float32(
                0.5
            )
            raw[k] = r
            sel[k] = r + bias[e]
        total = cutlass.Float32(0.0)
        picked_raw = cute.make_rmem_tensor((TOP_K,), cutlass.Float32)
        picked_id = cute.make_rmem_tensor((TOP_K,), cutlass.Int32)
        for t in cutlass.range_constexpr(TOP_K):
            # This lane's best unpicked expert: the largest score, the smallest id on ties.
            best = cutlass.Float32(_NEG_INF)
            best_id = cutlass.Int32(EXPERTS)
            best_raw = cutlass.Float32(0.0)
            for k in cutlass.range_constexpr(_PER_LANE):
                e = lane * _PER_LANE + k
                if (sel[k] > best) | ((sel[k] == best) & (e < best_id)):
                    best = sel[k]
                    best_id = e
                    best_raw = raw[k]
            for offset in (16, 8, 4, 2, 1):
                o_best = cute.arch.shuffle_sync_bfly(best, offset)
                o_id = cute.arch.shuffle_sync_bfly(best_id, offset)
                o_raw = cute.arch.shuffle_sync_bfly(best_raw, offset)
                if (o_best > best) | ((o_best == best) & (o_id < best_id)):
                    best = o_best
                    best_id = o_id
                    best_raw = o_raw
            for k in cutlass.range_constexpr(_PER_LANE):
                if lane * _PER_LANE + k == best_id:
                    sel[k] = cutlass.Float32(_NEG_INF)
            picked_raw[t] = best_raw
            picked_id[t] = best_id
            total = total + best_raw
        if lane < TOP_K:
            for t in cutlass.range_constexpr(TOP_K):
                if lane == t:
                    weights[row, t] = picked_raw[t] * scale / (total + cutlass.Float32(1.0e-20))
                    ids[row, t] = picked_id[t]


@cute.kernel
def _swiglu_kernel(src: cute.Tensor, out: cute.Tensor, col_tiles: cutlass.Int32) -> None:
    bidx, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    row = bidx // col_tiles
    tile = bidx - row * col_tiles
    width = cute.size(out, mode=[1])
    for k in cutlass.range_constexpr(_SWIGLU_PER_THREAD):
        c = tile * _SWIGLU_COLS + k * _SWIGLU_THREADS + tidx
        if c < width:
            g = src[row, c].to(cutlass.Float32)
            u = src[row, width + c].to(cutlass.Float32)
            s = g / (cutlass.Float32(1.0) + cute.math.exp(-g))
            out[row, c] = (s * u).to(cutlass.BFloat16)


@cute.jit
def _launch_topk(
    logits, bias, weights, ids, rows: cutlass.Int32, scale: cutlass.Float32, stream: cuda.CUstream
):
    _topk_kernel(logits, bias, weights, ids, rows, scale).launch(
        grid=((rows + _ROWS_PER_CTA - 1) // _ROWS_PER_CTA, 1, 1),
        block=(_THREADS, 1, 1),
        stream=stream,
    )


@cute.jit
def _launch_swiglu(
    src, out, col_tiles: cutlass.Int32, num_ctas: cutlass.Int32, stream: cuda.CUstream
):
    _swiglu_kernel(src, out, col_tiles).launch(
        grid=(num_ctas, 1, 1), block=(_SWIGLU_THREADS, 1, 1), stream=stream
    )


def compile_topk() -> CompiledKernel:
    if "topk" not in _COMPILED:
        r = cute.sym_int32()
        _COMPILED["topk"] = cute.compile(
            _launch_topk,
            make_fake_compact_tensor(
                cutlass.Float32, (r, EXPERTS), stride_order=(1, 0), assumed_align=16
            ),
            make_fake_compact_tensor(cutlass.Float32, (EXPERTS,), assumed_align=16),
            make_fake_compact_tensor(
                cutlass.Float32, (cute.sym_int32(), TOP_K), stride_order=(1, 0), assumed_align=16
            ),
            make_fake_compact_tensor(
                cutlass.Int32, (cute.sym_int32(), TOP_K), stride_order=(1, 0), assumed_align=16
            ),
            cutlass.Int32(1),
            cutlass.Float32(1.0),
            make_fake_stream(),
            options="--enable-tvm-ffi",
        )
    return _COMPILED["topk"]


def compile_swiglu() -> CompiledKernel:
    if "swiglu" not in _COMPILED:
        _COMPILED["swiglu"] = cute.compile(
            _launch_swiglu,
            make_fake_compact_tensor(
                cutlass.BFloat16,
                (cute.sym_int32(), cute.sym_int32()),
                stride_order=(1, 0),
                assumed_align=16,
            ),
            make_fake_compact_tensor(
                cutlass.BFloat16,
                (cute.sym_int32(), cute.sym_int32()),
                stride_order=(1, 0),
                assumed_align=16,
            ),
            cutlass.Int32(1),
            cutlass.Int32(1),
            make_fake_stream(),
            options="--enable-tvm-ffi",
        )
    return _COMPILED["swiglu"]


def sigmoid_topk(fn: CompiledKernel, logits: torch.Tensor, bias: torch.Tensor, scale: float):
    """``(weights [T, 8] fp32, ids [T, 8] int32)`` for fp32 ``logits [T, 256]``."""

    if (
        not logits.is_cuda
        or logits.dtype != torch.float32
        or logits.shape[1:] != (EXPERTS,)
        or not logits.is_contiguous()
    ):
        raise ValueError("sigmoid_topk: fp32 contiguous CUDA [T, 256] logits expected")
    # The launch runs on logits' device; bias must be readable there.
    if (
        not bias.is_cuda
        or bias.device != logits.device
        or bias.dtype != torch.float32
        or bias.shape != (EXPERTS,)
        or not bias.is_contiguous()
    ):
        raise ValueError("sigmoid_topk: fp32 contiguous [256] bias on the logits device expected")
    rows = int(logits.shape[0])
    w = torch.empty((rows, TOP_K), dtype=torch.float32, device=logits.device)
    i = torch.empty((rows, TOP_K), dtype=torch.int32, device=logits.device)
    if rows == 0:
        # An empty grid is a launch error; the empty result is exact.
        return w, i
    fn(logits, bias, w, i, rows, float(scale), torch.cuda.current_stream(logits.device).cuda_stream)
    return w, i


def swiglu(fn: CompiledKernel, src: torch.Tensor) -> torch.Tensor:
    """``silu(src[:, :W]) * src[:, W:]`` as new BF16 ``[T, W]``."""

    if (
        not src.is_cuda
        or src.dtype != torch.bfloat16
        or src.ndim != 2
        or src.shape[1] % 2
        or not src.is_contiguous()
    ):
        raise ValueError("swiglu: contiguous CUDA BF16 [T, 2W] expected")
    rows, width = int(src.shape[0]), int(src.shape[1]) // 2
    out = torch.empty((rows, width), dtype=torch.bfloat16, device=src.device)
    tiles = (width + _SWIGLU_COLS - 1) // _SWIGLU_COLS
    if rows == 0 or width == 0:
        # An empty grid is a launch error; the empty result is exact.
        return out
    fn(src, out, tiles, rows * tiles, torch.cuda.current_stream(src.device).cuda_stream)
    return out
