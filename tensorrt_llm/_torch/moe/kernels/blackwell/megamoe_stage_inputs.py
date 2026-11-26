# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage one MegaMoE call's routing and NVFP4 activation into the kernel's bucket buffers."""

import functools
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda

Int32, Int64, Float32 = cutlass.Int32, cutlass.Int64, cutlass.Float32


class MegaMoEStageInputsKernel:
    """One CTA per bucket row: live rows get the caller's routing (and activation) rows, the
    others expert -1 and weight 0 (dispatch skips -1; combine reads every weight)."""

    num_threads = 128

    def __init__(self, top_k: int, act_words: int, sf_words: int):
        self.top_k = top_k
        self.act_words = act_words
        self.sf_words = sf_words

    @cute.jit
    def __call__(
        self,
        idx_src: cute.Tensor,
        weights_src: cute.Tensor,
        act_src: Optional[cute.Tensor],
        sf_src: Optional[cute.Tensor],
        idx_dst: cute.Tensor,
        weights_dst: cute.Tensor,
        act_dst: Optional[cute.Tensor],
        sf_dst: Optional[cute.Tensor],
        stream: cuda.CUstream,
    ):
        self.kernel(
            idx_src, weights_src, act_src, sf_src, idx_dst, weights_dst, act_dst, sf_dst
        ).launch(grid=[idx_dst.shape[0], 1, 1], block=[self.num_threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        idx_src: cute.Tensor,
        weights_src: cute.Tensor,
        act_src: Optional[cute.Tensor],
        sf_src: Optional[cute.Tensor],
        idx_dst: cute.Tensor,
        weights_dst: cute.Tensor,
        act_dst: Optional[cute.Tensor],
        sf_dst: Optional[cute.Tensor],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        live = row < idx_src.shape[0]
        if tidx < self.top_k:
            expert = idx_dst.element_type(-1)
            weight = Float32(0.0)
            if live:
                expert = idx_src[row, tidx].to(idx_dst.element_type)
                weight = weights_src[row, tidx]
            idx_dst[row, tidx] = expert
            weights_dst[row, tidx] = weight
        if cutlass.const_expr(self.act_words > 0):
            if live:
                self.copy_row(act_src, act_dst, row, tidx, self.act_words)
                self.copy_row(sf_src, sf_dst, row, tidx, self.sf_words)

    @cute.jit
    def copy_row(
        self,
        src: cute.Tensor,
        dst: cute.Tensor,
        row: Int32,
        tidx: Int32,
        words: cutlass.Constexpr,
    ):
        for step in cutlass.range_constexpr((words + self.num_threads - 1) // self.num_threads):
            word = tidx + step * self.num_threads
            if word < words:
                dst[row, word] = src[row, word]


_INDEX_TYPES = {torch.int32: Int32, torch.int64: Int64}


def _fake_rows(dtype, rows, width: int):
    return cute.runtime.make_fake_compact_tensor(dtype, (rows, width), stride_order=(1, 0))


@functools.cache
def _compile_stage_inputs(top_k: int, index_dtype: torch.dtype, act_words: int, sf_words: int):
    live, rows = cute.sym_int(), cute.sym_int()
    staged = act_words > 0
    return cute.compile(
        MegaMoEStageInputsKernel(top_k, act_words, sf_words),
        _fake_rows(Int32, live, top_k),
        _fake_rows(Float32, live, top_k),
        _fake_rows(Int32, live, act_words) if staged else None,
        _fake_rows(Int32, live, sf_words) if staged else None,
        _fake_rows(_INDEX_TYPES[index_dtype], rows, top_k),
        _fake_rows(Float32, rows, top_k),
        _fake_rows(Int32, rows, act_words) if staged else None,
        _fake_rows(Int32, rows, sf_words) if staged else None,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--opt-level 2 --enable-tvm-ffi",
    )


def stage_megamoe_inputs(
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    idx_buffer: torch.Tensor,
    weights_buffer: torch.Tensor,
    x: Optional[torch.Tensor] = None,
    x_sf: Optional[torch.Tensor] = None,
    act_buffer: Optional[torch.Tensor] = None,
    sf_buffer: Optional[torch.Tensor] = None,
) -> None:
    """Fill a bucket's routing (and, with ``x``, activation) buffers; rows past ``num_tokens`` get
    expert -1 and weight 0. Byte rows must be whole 32-bit words."""
    act_src = sf_src = act_dst = sf_dst = None
    act_words = sf_words = 0
    if x is not None:
        act_src, act_dst = x.view(torch.int32), act_buffer.view(torch.int32)
        sf_src, sf_dst = x_sf.view(torch.int32), sf_buffer.view(torch.int32)
        act_words, sf_words = act_dst.shape[1], sf_dst.shape[1]
    _compile_stage_inputs(idx_buffer.shape[1], idx_buffer.dtype, act_words, sf_words)(
        topk_idx, topk_weights, act_src, sf_src, idx_buffer, weights_buffer, act_dst, sf_dst
    )
