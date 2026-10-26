# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compact rank-major DSpark TP argmax: local candidates, global argmax and Markov embedding."""

import functools
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils.smem_allocator import SmemAllocator

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda

WORDS_PER_PART = 4096
# BF16 elements per 16-byte vector.
_VEC = 8
_THREADS = 128
_ROWS_PER_CTA = 4
_COMPILE_OPTIONS = "--opt-level 2 --enable-tvm-ffi"


@dsl_user_op
def _add_rn(lhs: cutlass.Float32, rhs: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Float32:
    # Round to nearest and keep subnormals, matching an unfused FP32 torch addition.
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [lhs.ir_value(loc=loc, ip=ip), rhs.ir_value(loc=loc, ip=ip)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _takes(value, index, other, other_index):
    """Whether (other, other_index) precedes (value, index) in torch.argmax order.

    NaNs precede other values and ties choose the smaller index.
    """
    number = value == value
    other_number = other == other
    other_nan = other != other
    smaller = other_index < index
    return (
        (number & other_nan)
        | ((value != value) & other_nan & smaller)
        | (number & other_number & ((other > value) | ((other == value) & smaller)))
    )


@cute.jit
def _warp_argmax(value, index, width: cutlass.Constexpr):
    for step in cutlass.range_constexpr(width.bit_length() - 1):
        other = cute.arch.shuffle_sync_bfly(value, offset=1 << step)
        other_index = cute.arch.shuffle_sync_bfly(index, offset=1 << step)
        if _takes(value, index, other, other_index):
            value = other
            index = other_index
    return value, index


@cute.jit
def _copy_embedding_row(index, weight: cute.Tensor, output: cute.Tensor, row, lane):
    """Copy weight[index] to output[row] with one warp, zeroing an out-of-range index."""
    vocab, width = weight.shape
    chunks = width // _VEC
    valid = (index >= 0) & (index < vocab)
    safe_index = cutlass.Int32(0)
    if valid:
        safe_index = cutlass.Int32(index)
    source = cute.zipped_divide(weight[safe_index, None], (_VEC,))
    destination = cute.zipped_divide(output[row, None], (_VEC,))
    fragment = cute.make_rmem_tensor((_VEC,), weight.element_type)
    for item in cutlass.range_constexpr((chunks + cute.arch.WARP_SIZE - 1) // cute.arch.WARP_SIZE):
        chunk = lane + item * cute.arch.WARP_SIZE
        if chunk < chunks:
            fragment.fill(0)
            if valid:
                cute.autovec_copy(source[None, chunk], fragment)
            cute.autovec_copy(fragment, destination[None, chunk])


class DSparkLocalCandidatesKernel:
    """Reduce each 4096-column part of base + bias to one (value, float(global id)) packet."""

    def __init__(self, local_vocab: int, global_vocab: int, offset: int):
        self.local_vocab = local_vocab
        self.global_vocab = global_vocab
        self.offset = offset
        self.parts = (local_vocab + WORDS_PER_PART - 1) // WORDS_PER_PART
        self.items = WORDS_PER_PART // (_THREADS * _VEC)

    @cute.jit
    def __call__(
        self, base: cute.Tensor, bias: cute.Tensor, packet: cute.Tensor, stream: cuda.CUstream
    ):
        self.kernel(base, bias, packet).launch(
            grid=[self.parts, base.shape[0], 1], block=[_THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, base: cute.Tensor, bias: cute.Tensor, packet: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        part, row, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        base_row = cute.zipped_divide(base[row, None], (_VEC,))
        bias_row = cute.zipped_divide(bias[row, None], (_VEC,))
        base_values = cute.make_rmem_tensor((_VEC, self.items), base.element_type)
        bias_values = cute.make_rmem_tensor((_VEC, self.items), bias.element_type)
        first_chunk = part * (WORDS_PER_PART // _VEC) + tidx
        # Issue every load before reducing.
        for item in cutlass.range_constexpr(self.items):
            chunk = first_chunk + item * _THREADS
            if chunk < self.local_vocab // _VEC:
                cute.autovec_copy(base_row[None, chunk], base_values[None, item])
                cute.autovec_copy(bias_row[None, chunk], bias_values[None, item])

        # Each thread visits its columns in increasing order, so a strict
        # comparison keeps the first maximum; the first valid column always
        # replaces the sentinel, and the first NaN is never replaced.
        best = -cutlass.Float32.inf
        best_id = cutlass.Int32(self.global_vocab)
        for item in cutlass.range_constexpr(self.items):
            for element in cutlass.range_constexpr(_VEC):
                column = (first_chunk + item * _THREADS) * _VEC + element
                global_id = column + self.offset
                value = _add_rn(
                    cutlass.Float32(base_values[element, item]),
                    cutlass.Float32(bias_values[element, item]),
                )
                if (
                    (column < self.local_vocab)
                    & (global_id < self.global_vocab)
                    & (
                        (best_id == self.global_vocab)
                        | ((best == best) & ((value > best) | (value != value)))
                    )
                ):
                    best = value
                    best_id = global_id

        best, best_id = _warp_argmax(best, best_id, cute.arch.WARP_SIZE)
        smem = SmemAllocator()
        warps = _THREADS // cute.arch.WARP_SIZE
        shared_values = smem.allocate_tensor(cutlass.Float32, cute.make_layout(warps))
        shared_ids = smem.allocate_tensor(cutlass.Int32, cute.make_layout(warps))
        if lane == 0:
            shared_values[warp] = best
            shared_ids[warp] = best_id
        cute.arch.barrier()
        if warp == 0:
            best = -cutlass.Float32.inf
            best_id = cutlass.Int32(self.global_vocab)
            if lane < warps:
                best = shared_values[lane]
                best_id = shared_ids[lane]
            best, best_id = _warp_argmax(best, best_id, warps)
            if lane == 0:
                packet[row, part, 0] = best
                packet[row, part, 1] = best_id.to(cutlass.Float32)


class DSparkGlobalArgmaxKernel:
    """Reduce rank-major packets to global IDs, optionally gathering the selected embedding."""

    def __init__(self, ranks: int, parts: int, global_vocab: int, gather: bool):
        self.ranks = ranks
        self.parts = parts
        self.global_vocab = global_vocab
        self.gather = gather

    @cute.jit
    def __call__(
        self,
        packet: cute.Tensor,
        output: cute.Tensor,
        weight: Optional[cute.Tensor],
        embedding: Optional[cute.Tensor],
        stream: cuda.CUstream,
    ):
        self.kernel(packet, output, weight, embedding).launch(
            grid=[(output.shape[0] + _ROWS_PER_CTA - 1) // _ROWS_PER_CTA, 1, 1],
            block=[cute.arch.WARP_SIZE * _ROWS_PER_CTA, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        packet: cute.Tensor,
        output: cute.Tensor,
        weight: Optional[cute.Tensor],
        embedding: Optional[cute.Tensor],
    ):
        cta, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        row = cta * _ROWS_PER_CTA + cute.arch.warp_idx()
        rows = output.shape[0]
        if row < rows:
            best = -cutlass.Float32.inf
            best_id = cutlass.Int32(self.global_vocab)
            candidates = self.ranks * self.parts
            for item in cutlass.range_constexpr(
                (candidates + cute.arch.WARP_SIZE - 1) // cute.arch.WARP_SIZE
            ):
                candidate = lane + item * cute.arch.WARP_SIZE
                if candidate < candidates:
                    source = (candidate // self.parts) * rows + row
                    part = candidate % self.parts
                    value = packet[source, part, 0]
                    index = cutlass.Int32(packet[source, part, 1])
                    if _takes(best, best_id, value, index):
                        best = value
                        best_id = index
            best, best_id = _warp_argmax(best, best_id, cute.arch.WARP_SIZE)
            if lane == 0:
                result = cutlass.Int64(-1)
                if best_id < self.global_vocab:
                    result = cutlass.Int64(best_id)
                output[row] = result
            if cutlass.const_expr(self.gather):
                _copy_embedding_row(best_id, weight, embedding, row, lane)


class DSparkMaskedEmbeddingKernel:
    """Gather weight rows for INT64 tokens, zeroing out-of-range tokens."""

    @cute.jit
    def __call__(
        self,
        tokens: cute.Tensor,
        weight: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(tokens, weight, output).launch(
            grid=[(output.shape[0] + _ROWS_PER_CTA - 1) // _ROWS_PER_CTA, 1, 1],
            block=[cute.arch.WARP_SIZE * _ROWS_PER_CTA, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, tokens: cute.Tensor, weight: cute.Tensor, output: cute.Tensor):
        cta, _, _ = cute.arch.block_idx()
        row = cta * _ROWS_PER_CTA + cute.arch.warp_idx()
        if row < output.shape[0]:
            _copy_embedding_row(tokens[row], weight, output, row, cute.arch.lane_idx())


_CUTE_DTYPES = {torch.bfloat16: cutlass.BFloat16, torch.float32: cutlass.Float32}


def _fake_rows(dtype, rows, width: int, row_stride=None):
    """Fake [rows, width] tensor with 16-byte aligned rows and unit column stride."""
    if row_stride is None:
        row_stride = cute.sym_int64(divisibility=_VEC)
    return cute.runtime.make_fake_tensor(
        dtype, (rows, width), stride=(row_stride, 1), assumed_align=16
    )


@functools.cache
def _compile_local_candidates(bias_dtype: torch.dtype, local_vocab: int, vocab: int, offset: int):
    rows = cute.sym_int()
    parts = (local_vocab + WORDS_PER_PART - 1) // WORDS_PER_PART
    return cute.compile(
        DSparkLocalCandidatesKernel(local_vocab, vocab, offset),
        _fake_rows(cutlass.BFloat16, rows, local_vocab),
        _fake_rows(_CUTE_DTYPES[bias_dtype], rows, local_vocab),
        cute.runtime.make_fake_compact_tensor(
            cutlass.Float32, (rows, parts, 2), stride_order=(2, 1, 0)
        ),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options=_COMPILE_OPTIONS,
    )


@functools.cache
def _compile_global_argmax(ranks: int, parts: int, vocab: int, width: int):
    rows = cute.sym_int()
    gather = width > 0
    return cute.compile(
        DSparkGlobalArgmaxKernel(ranks, parts, vocab, gather),
        cute.runtime.make_fake_compact_tensor(
            cutlass.Float32, (cute.sym_int(), parts, 2), stride_order=(2, 1, 0)
        ),
        cute.runtime.make_fake_compact_tensor(cutlass.Int64, (rows,), stride_order=(0,)),
        _fake_rows(cutlass.BFloat16, vocab, width, width) if gather else None,
        _fake_rows(cutlass.BFloat16, rows, width, width) if gather else None,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options=_COMPILE_OPTIONS,
    )


@functools.cache
def _compile_masked_embedding(vocab: int, width: int):
    rows = cute.sym_int()
    return cute.compile(
        DSparkMaskedEmbeddingKernel(),
        cute.runtime.make_fake_tensor(cutlass.Int64, (rows,), stride=(cute.sym_int64(),)),
        _fake_rows(cutlass.BFloat16, vocab, width, width),
        _fake_rows(cutlass.BFloat16, rows, width, width),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options=_COMPILE_OPTIONS,
    )


def masked_embedding(tokens: torch.Tensor, weight: torch.Tensor, output: torch.Tensor) -> None:
    """Gather BF16 weight [vocab, width] into output [B, width], zeroing invalid IDs.

    The caller supplies contiguous weight and output storage on the same CUDA
    device, with width a multiple of 8. INT64 tokens [B] may be strided. Output
    must not alias either input.
    """
    _compile_masked_embedding(*weight.shape)(tokens, weight, output)


def local_candidates(
    base: torch.Tensor,
    bias: torch.Tensor,
    offset: int,
    vocab: int,
    packet: torch.Tensor,
) -> None:
    """Reduce [B, local_vocab] BF16 base + BF16/FP32 bias into [B, parts, 2].

    Caller provides distinct contiguous FP32 packet storage. Input rows may be
    strided; columns are contiguous, and every row starts 16-byte aligned with
    local_vocab a multiple of 8. Values and exact global IDs share each packet,
    so the vocabulary must contain fewer than 2**24 entries.
    """
    _compile_local_candidates(bias.dtype, base.shape[1], vocab, offset)(base, bias, packet)


def global_argmax(
    packet: torch.Tensor,
    output: torch.Tensor,
    vocab: int,
    *,
    embedding_weight: torch.Tensor | None = None,
    embedding_output: torch.Tensor | None = None,
) -> None:
    """Reduce [ranks * B, parts, 2] packets into B global INT64 token IDs.

    Caller provides distinct contiguous packet/output storage. NaNs precede
    other values and ties choose the smallest global ID. A row without any
    valid candidate produces -1.

    When both embedding tensors are provided, also gather the selected rows of
    contiguous BF16 weight [vocab, width] into contiguous output [B, width],
    zeroing rows with no valid candidate. Embedding output must not alias other
    operands.
    """
    rows, parts = output.numel(), packet.shape[1]
    width = 0 if embedding_weight is None else embedding_weight.shape[1]
    _compile_global_argmax(packet.shape[0] // rows, parts, vocab, width)(
        packet, output, embedding_weight, embedding_output
    )
