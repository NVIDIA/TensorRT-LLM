# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Embedding lookup of a vocabulary slice in one launch: tokens outside the slice get zero rows."""

import functools

import cutlass
import cutlass.cute as cute
import torch

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda

VEC = 8  # BF16 elements per 16-byte vector
# One vector per thread: a row spans several CTAs, so a decode step's few rows fill more SMs.
THREADS = 128

_TOKEN_TYPES = {torch.int32: cutlass.Int32, torch.int64: cutlass.Int64}


class MaskedEmbeddingKernel:
    """``output[i] = weight[tokens[i] - start]`` when the token is in ``[start, start + vocab)``,
    else zeros."""

    def __init__(self, start: int):
        self.start = start

    @cute.jit
    def __call__(
        self,
        tokens: cute.Tensor,
        weight: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ):
        chunks = output.shape[1] // VEC
        self.kernel(tokens, weight, output).launch(
            grid=[output.shape[0], (chunks + THREADS - 1) // THREADS, 1],
            block=[THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, tokens: cute.Tensor, weight: cute.Tensor, output: cute.Tensor):
        row, block, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        vocab, width = weight.shape
        chunk = block * THREADS + tidx
        if chunk < width // VEC:
            index = cutlass.Int64(tokens[row]) - self.start
            valid = (index >= 0) & (index < vocab)
            safe_index = cutlass.Int32(0)
            if valid:
                safe_index = cutlass.Int32(index)
            source = cute.zipped_divide(weight[safe_index, None], (VEC,))
            destination = cute.zipped_divide(output[row, None], (VEC,))
            fragment = cute.make_rmem_tensor((VEC,), weight.element_type)
            fragment.fill(0)
            if valid:
                cute.autovec_copy(source[None, chunk], fragment)
            cute.autovec_copy(fragment, destination[None, chunk])


def _rows(rows, width: int):
    return cute.runtime.make_fake_tensor(
        cutlass.BFloat16, (rows, width), stride=(width, 1), assumed_align=16
    )


@functools.cache
def _compile(vocab: int, width: int, start: int, token_dtype: torch.dtype):
    rows = cute.sym_int()
    return cute.compile(
        MaskedEmbeddingKernel(start),
        cute.runtime.make_fake_compact_tensor(
            _TOKEN_TYPES[token_dtype], (rows,), stride_order=(0,)
        ),
        _rows(vocab, width),
        _rows(rows, width),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--opt-level 2 --enable-tvm-ffi",
    )


def supports_masked_embedding(tokens: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        tokens.is_cuda
        and tokens.dtype in _TOKEN_TYPES
        and weight.dtype == torch.bfloat16
        and weight.dim() == 2
        and weight.is_contiguous()
        and weight.shape[1] % VEC == 0
        and weight.data_ptr() % 16 == 0
    )


def masked_embedding(tokens: torch.Tensor, weight: torch.Tensor, start: int) -> torch.Tensor:
    """Rows ``weight[token - start]`` for the tokens in the slice, zeros for the others."""
    flat = tokens.reshape(-1).contiguous()
    output = weight.new_empty((flat.shape[0], weight.shape[1]))
    if flat.shape[0]:
        _compile(*weight.shape, start, flat.dtype)(flat, weight, output)
    return output.view(*tokens.shape, weight.shape[1])
