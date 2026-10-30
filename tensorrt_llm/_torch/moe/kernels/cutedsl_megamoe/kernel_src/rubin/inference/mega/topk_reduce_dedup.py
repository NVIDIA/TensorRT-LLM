# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # ctm-lint: allow-internal-marker
# SPDX-License-Identifier: BSD-3-Clause
"""Combine reduce for the dedup path: sums the ``rank_slots[token][rank]`` rows named by ``valid_rank_mask`` in
ascending rank order into ``reduced_output[token]``; top-k weights are already folded in."""

from typing import ClassVar

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Float32, Int32

from .....helpers.dsl_helpers import mark_alignment
from .....quant_def import CombineFormat


class TopkReduceDedup:
    _threads: ClassVar[int] = 128
    _hidden_per_thread: ClassVar[int] = 16  # one 32 B (ld/st.256) access of bf16 per slot

    def __init__(self, hidden: int, world_size: int, combine_format: CombineFormat) -> None:
        if combine_format.is_quantized or combine_format.act_dtype is not cutlass.BFloat16:
            raise ValueError("TopkReduceDedup supports the BF16 combine format only.")
        if world_size > 32:
            raise ValueError("TopkReduceDedup keeps the rank mask in an Int32; world_size must be <= 32.")
        self.hidden = int(hidden)
        self.world_size = int(world_size)
        self.combine_format = combine_format
        if self.hidden % self._hidden_per_thread != 0:
            raise ValueError(f"hidden ({self.hidden}) must be divisible by {self._hidden_per_thread}.")
        self.hidden_tiles = self.hidden // self._hidden_per_thread
        self.require_predicate = self.hidden_tiles % self._threads != 0

    @cute.jit
    def __call__(
        self,
        rank_slots: cute.Tensor,  # (token, world_size, hidden) bf16
        valid_rank_mask: cute.Tensor,  # (token,) i32 bitmask over ranks
        reduced_output: cute.Tensor,  # (token, hidden)
        stream: cuda.CUstream,
    ):
        threads = self._threads
        total_workers = reduced_output.shape[0] * self.hidden_tiles
        grid = [(total_workers + threads - 1) // threads, 1, 1]
        block = [threads, 1, 1]
        rank_slots = cute.make_tensor(
            rank_slots.iterator,
            cute.make_layout((rank_slots.shape[0], self.world_size, self.hidden), stride=rank_slots.stride),
        )
        reduced_output = cute.make_tensor(
            reduced_output.iterator,
            cute.make_layout((reduced_output.shape[0], self.hidden), stride=reduced_output.stride),
        )
        self._reduce_bf16(rank_slots, valid_rank_mask, reduced_output).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def _reduce_bf16(self, rank_slots: cute.Tensor, valid_rank_mask: cute.Tensor, reduced_output: cute.Tensor):
        threads = self._threads
        hidden_per_thread = self._hidden_per_thread
        hidden_tiles = self.hidden_tiles
        world_size: cutlass.Constexpr[int] = self.world_size
        needs_guard = self.require_predicate
        out_dtype = reduced_output.element_type

        worker_idx = cute.arch.block_idx()[0] * Int32(threads) + cute.arch.thread_idx()[0]
        token_idx = worker_idx // hidden_tiles
        hidden_tile_idx = worker_idx % hidden_tiles

        if (not needs_guard) or token_idx < reduced_output.shape[0]:
            # (token, rank, hidden) -> (rank, hidden_per_thread) slice owned by this worker
            slots = cute.zipped_divide(rank_slots[token_idx, None, None], (world_size, hidden_per_thread))[
                (None, None), (0, hidden_tile_idx)
            ]
            dst = cute.zipped_divide(reduced_output[token_idx, None], (hidden_per_thread,))[(None,), (hidden_tile_idx,)]
            rank_mask = valid_rank_mask[token_idx]

            access_bits: cutlass.Constexpr[int] = hidden_per_thread * cutlass.BFloat16.width
            load_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=access_bits)
            acc = cute.make_rmem_tensor((hidden_per_thread,), cutlass.Float32)
            for i in cutlass.range_constexpr(hidden_per_thread):
                acc[i] = Float32(0.0)
            one_pair = (Float32(1.0), Float32(1.0))

            # Issue every rank slot's load first (predicated, one register buffer per rank) so all of them are in
            # flight together, then accumulate ranks ascending; fma by 1.0 is an exact fp32 add.
            terms = [cute.make_rmem_tensor((hidden_per_thread,), cutlass.BFloat16) for _ in range(world_size)]
            for rank in cutlass.range_constexpr(world_size):
                if ((rank_mask >> Int32(rank)) & Int32(1)) != Int32(0):
                    cute.copy(
                        load_atom,
                        mark_alignment(slots[rank, None], hidden_per_thread * cutlass.BFloat16.width // 8),
                        terms[rank],
                    )
            for rank in cutlass.range_constexpr(world_size):
                if ((rank_mask >> Int32(rank)) & Int32(1)) != Int32(0):
                    for i in cutlass.range_constexpr(0, hidden_per_thread, 2):
                        value_pair = (Float32(terms[rank][i]), Float32(terms[rank][i + 1]))
                        acc[i], acc[i + 1] = cute.arch.fma_packed_f32x2(value_pair, one_pair, (acc[i], acc[i + 1]))

            out = cute.make_rmem_tensor((hidden_per_thread,), out_dtype)
            out.store(acc.load().to(out_dtype))
            cute.copy(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), out_dtype, num_bits_per_copy=hidden_per_thread * out_dtype.width),
                out,
                mark_alignment(dst, hidden_per_thread * out_dtype.width // 8),
            )


__all__ = ["TopkReduceDedup"]
