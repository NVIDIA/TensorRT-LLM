# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engram n-gram hashing over the per-request compressed-token history."""

import functools

import cutlass
import cutlass.cute as cute
import torch

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda

Int32, Int64, Uint8 = cutlass.Int32, cutlass.Int64, cutlass.Uint8


class EngramDecodeHashKernel:
    """Hash each token's n-grams against ``history[row, position - offset]``.

    Grid ``y`` selects the n-gram order, so every head's modulus is a
    compile-time constant. Without ``read_current_history`` the token's own
    compressed id comes from the lookup table (and order 0 writes it to
    history with PyTorch's negative-index semantics); with it, the caller has
    already scattered the step's tokens and the kernel reads them back. Output
    rows past ``num_tokens`` are zero.
    """

    num_threads = 32

    def __init__(
        self,
        moduli: tuple[tuple[int, ...], ...],
        pad_id: int,
        dead: int,
        write_history: bool,
        read_current_history: bool,
        has_mask: bool,
    ):
        self.moduli = moduli
        self.max_ngram = len(moduli) + 1
        self.n_head = len(moduli[0])
        self.pad_id = pad_id
        self.dead = dead
        self.write_history = write_history
        self.read_current_history = read_current_history
        self.has_mask = has_mask

    @cute.jit
    def __call__(
        self,
        input_ids: cute.Tensor,
        positions: cute.Tensor,
        rows: cute.Tensor,
        token_mask: cute.Tensor | None,
        lookup: cute.Tensor,
        history: cute.Tensor,
        multipliers: cute.Tensor,
        output: cute.Tensor,
        num_tokens: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            input_ids, positions, rows, token_mask, lookup, history, multipliers, output, num_tokens
        ).launch(
            grid=[
                (output.shape[0] + self.num_threads - 1) // self.num_threads,
                self.max_ngram - 1,
                1,
            ],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        input_ids: cute.Tensor,
        positions: cute.Tensor,
        rows: cute.Tensor,
        token_mask: cute.Tensor | None,
        lookup: cute.Tensor,
        history: cute.Tensor,
        multipliers: cute.Tensor,
        output: cute.Tensor,
        num_tokens: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        cta, order, _ = cute.arch.block_idx()
        token = cta * self.num_threads + tidx
        if token < output.shape[0]:
            if token < num_tokens:
                columns = Int64(history.shape[1])
                position = Int64(positions[token])
                row = Int64(rows[token])
                compressed = Int64(self.dead)
                if cutlass.const_expr(self.read_current_history):
                    if position < columns:
                        current = position
                        if current < 0:
                            current = Int64(0)
                        compressed = Int64(history[row, current])
                else:
                    raw = Int64(input_ids[token])
                    if raw < 0:
                        raw = Int64(0)
                    if raw > Int64(lookup.shape[0] - 1):
                        raw = Int64(lookup.shape[0] - 1)
                    compressed = Int64(lookup[raw])
                    if cutlass.const_expr(self.has_mask):
                        if Uint8(token_mask[token]) == 0:
                            compressed = Int64(self.dead)
                    # History stores int32, so hash the truncated value.
                    compressed = Int64(Int32(compressed))
                if cutlass.const_expr(self.write_history):
                    if (order == 0) & (position >= -columns) & (position < columns):
                        target = position
                        if target < 0:
                            target = target + columns
                        history[row, target] = Int32(compressed)

                blocked = (position < 0) | (compressed == self.dead)
                source = compressed
                if blocked:
                    source = Int64(self.pad_id)
                mix = source * Int64(multipliers[0])
                for offset in cutlass.range_constexpr(1, self.max_ngram):
                    if order >= offset - 1:
                        previous = position - offset
                        source = Int64(self.dead)
                        if (previous >= 0) & (previous < columns):
                            source = Int64(history[row, previous])
                        blocked = blocked | (previous < 0) | (source == self.dead)
                        if blocked:
                            source = Int64(self.pad_id)
                        mix = mix ^ (source * Int64(multipliers[offset]))
                        if order == offset - 1:
                            self.store_hashes(output, token, mix, offset - 1)
            else:
                for head in cutlass.range_constexpr(self.n_head):
                    output[token, order * self.n_head + head] = Int64(0)

    @cute.jit
    def store_hashes(self, output: cute.Tensor, token: Int32, mix: Int64, order: cutlass.Constexpr):
        """PyTorch's nonnegative ``mix % modulus`` for each head of one n-gram order."""
        for head in cutlass.range_constexpr(self.n_head):
            remainder = mix % Int64(self.moduli[order][head])
            if remainder < 0:
                remainder = remainder + self.moduli[order][head]
            output[token, order * self.n_head + head] = remainder


_INDEX_DTYPES = {torch.int32: Int32, torch.int64: Int64}


@functools.cache
def compile_engram_decode_hash(
    moduli: tuple[tuple[int, ...], ...],
    pad_id: int,
    dead: int,
    write_history: bool,
    read_current_history: bool,
    has_mask: bool,
    ids_dtype: torch.dtype,
    positions_dtype: torch.dtype,
):
    """Compile one layer's hash; tensors use TVM-FFI calling and the mask is uint8."""
    tokens = cute.sym_int()
    input_ids = cute.runtime.make_fake_compact_tensor(
        _INDEX_DTYPES[ids_dtype], (tokens,), stride_order=(0,)
    )
    positions = cute.runtime.make_fake_compact_tensor(
        _INDEX_DTYPES[positions_dtype], (tokens,), stride_order=(0,)
    )
    rows = cute.runtime.make_fake_compact_tensor(Int64, (tokens,), stride_order=(0,))
    token_mask = (
        cute.runtime.make_fake_compact_tensor(Uint8, (tokens,), stride_order=(0,))
        if has_mask
        else None
    )
    lookup = cute.runtime.make_fake_compact_tensor(Int64, (cute.sym_int(),), stride_order=(0,))
    history = cute.runtime.make_fake_compact_tensor(
        Int32, (cute.sym_int(), cute.sym_int()), stride_order=(1, 0)
    )
    multipliers = cute.runtime.make_fake_compact_tensor(
        Int64, (len(moduli) + 1,), stride_order=(0,)
    )
    output = cute.runtime.make_fake_compact_tensor(
        Int64, (cute.sym_int(), len(moduli) * len(moduli[0])), stride_order=(1, 0)
    )
    kernel = EngramDecodeHashKernel(
        moduli, pad_id, dead, write_history, read_current_history, has_mask
    )
    return cute.compile(
        kernel,
        input_ids,
        positions,
        rows,
        token_mask,
        lookup,
        history,
        multipliers,
        output,
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--opt-level 2 --enable-tvm-ffi",
    )
