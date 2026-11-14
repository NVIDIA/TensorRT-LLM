# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pack each row's selected latent-cache row ids to the front, in one launch.

The expanded selection of a sparse MLA row (``kpool_expand``) holds ``-1`` wherever a pool slot
or a tail slot selected nothing, while the native sparse attention reads each row's first
``lengths[r]`` ids. Per row this moves the nonnegative ids to the front in their order
(duplicates kept), pads with ``-1`` to the next multiple of 64 columns, and gives a row with no
valid id one dummy id ``0``, length 1 and ``nonempty`` False: the attention kernel takes no
empty row, so its caller discards that row's output.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


@triton.jit
def _compact_sparse_rows_kernel(
    Indices,
    Packed,
    Lengths,
    Nonempty,
    Stride: tl.constexpr,
    Width: tl.constexpr,
    OutWidth: tl.constexpr,
    Block: tl.constexpr,
    OutBlock: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, Block)
    value = tl.load(Indices + row * Stride + col, col < Width, -1)
    valid = (col < Width) & (value >= 0)
    destination = tl.cumsum(valid.to(tl.int32)) - 1
    length = tl.sum(valid.to(tl.int32))
    tl.store(Packed + row * OutWidth + destination, value, valid)
    tail = tl.arange(0, OutBlock)
    tl.store(
        Packed + row * OutWidth + tail, -1, (tail >= tl.maximum(length, 1)) & (tail < OutWidth)
    )
    if length == 0:
        tl.store(Packed + row * OutWidth, 0)
    tl.store(Lengths + row, tl.maximum(length, 1))
    tl.store(Nonempty + row, length > 0)


class CompactSparseRows(OpWrapper):
    """``indices`` int32 ``[R, W]`` -> (``packed`` int32 ``[R, cdiv(W, 64) * 64]``, ``lengths``
    int32 ``[R]``, ``nonempty`` bool ``[R]``).

    ``packed[r, :lengths[r]]`` are the ids ``>= 0`` of ``indices[r]`` in column order, the rest
    of the row is ``-1``; a row with none is ``[0, -1, ...]`` with length 1 and ``nonempty``
    False. Negative ids are the only invalid ones: an id past the latent cache is kept.
    """

    ARCHS = frozenset({Arch.SM_100})
    # GLM-5.3-Flash selects index_topk = 2048 positions as 512 pools of 4 plus a tail of up to
    # 3, so the expanded rows are kernel_output_width = cdiv(2051, 64) * 64 = 2112 wide.
    # `kv` is the visible length of the cell's last row; the test builds each row's expansion
    # from its own visible length.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="MTP-3 verify of one 13K-token request",
            spec=dict(rows=4, width=2112, kv=13312),
        ),
        Cell(
            why="MTP-3 verify of eight 13K-token requests",
            spec=dict(rows=32, width=2112, kv=13312),
        ),
        Cell(
            why="MTP-3 verify of 64 requests, the largest CUDA-graph batch",
            spec=dict(rows=256, width=2112, kv=13312),
        ),
        Cell(
            why="single-token decode of eight 13K-token requests",
            spec=dict(rows=8, width=2112, kv=13312),
        ),
        Cell(
            why="one rank's sequence-parallel quarter of a 13,312-token prompt (rows 0-3327)",
            spec=dict(rows=3328, width=2112, kv=3328),
        ),
        Cell(
            why="rank 1's quarter (rows 10-19) of a 37-token prompt: short rows",
            spec=dict(rows=10, width=2112, kv=20),
        ),
    )

    note = """
    The row stride of `indices` is a compile-time constant of the kernel, so every distinct
    stride compiles once; the callers pass the contiguous output of `kpool_expand`.

    An id at or past the latent cache's row count is packed like any other: the native sparse
    attention would read past the cache. `kpool_expand` writes in-range ids or -1 only.
    """

    def __call__(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = indices.shape[0]
        width = triton.cdiv(indices.shape[1], 64) * 64
        packed = torch.empty((rows, width), dtype=torch.int32, device=indices.device)
        lengths = torch.empty(rows, dtype=torch.int32, device=indices.device)
        nonempty = torch.empty(rows, dtype=torch.bool, device=indices.device)
        if rows == 0:
            return packed, lengths, nonempty
        _compact_sparse_rows_kernel[(rows,)](
            indices,
            packed,
            lengths,
            nonempty,
            indices.stride(0),
            indices.shape[1],
            width,
            triton.next_power_of_2(indices.shape[1]),
            triton.next_power_of_2(width),
        )
        return packed, lengths, nonempty

    def reference(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """A stable partition of each row (valid ids first, in order), then the padding and the
        dummy id of an empty row."""
        rows, width = indices.shape
        out_width = triton.cdiv(width, 64) * 64
        valid = indices >= 0
        # Stable sort on "is invalid" keeps the valid ids in their column order.
        order = torch.sort((~valid).to(torch.int8), dim=1, stable=True).indices
        moved = torch.gather(indices, 1, order)
        count = valid.sum(dim=1)
        col = torch.arange(out_width, device=indices.device)
        packed = torch.full((rows, out_width), -1, dtype=torch.int32, device=indices.device)
        packed[:, :width] = moved.to(torch.int32)
        packed = torch.where(col[None, :] < count[:, None], packed, -1)
        packed[:, 0] = torch.where(count > 0, packed[:, 0], 0)
        return packed, count.clamp(min=1).to(torch.int32), count > 0

    def is_valid(self, indices: torch.Tensor) -> None:
        assert indices.dim() == 2 and indices.dtype == torch.int32, (
            f"indices must be int32 [R, W]; got {indices.dtype} {tuple(indices.shape)}: the "
            "kernel stores the loaded ids into an int32 output"
        )
        assert indices.stride(1) == 1, f"indices rows must be unit-stride; got {indices.stride()}"
        # Row offsets are int32 products of the program id and a row width.
        out_width = triton.cdiv(indices.shape[1], 64) * 64
        assert indices.shape[0] * max(indices.stride(0), out_width) < 2**31


compact_sparse_rows = CompactSparseRows()
