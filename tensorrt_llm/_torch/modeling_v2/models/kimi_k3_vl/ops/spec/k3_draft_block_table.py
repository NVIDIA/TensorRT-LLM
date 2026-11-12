# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash drafter's page-table decode: layer-relative block offsets to pool pages, and the
pages each row holds (Triton, one row per program or a row's columns over programs)."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_spec.spec_tail_kernels import (
    decode_draft_block_table,
    decode_draft_block_table_parallel,
)


class K3DraftBlockTable(OpWrapper):
    """For ``encoded = src - block_offset``: ``table[:rows] = encoded.clamp(min=0) // divisor``
    (int32) and ``counts[:rows] = (encoded >= 0).sum(-1)`` (int64). Returns None.

    ``src`` is ``[rows, blocks]`` of any integer dtype and strides and is not written; ``table`` is
    ``[>= rows, blocks]`` int32 with a unit inner stride, ``counts`` ``[>= rows]`` int64. With
    ``partial`` (an int32 scratch of at least ``rows * cdiv(blocks, 2048)`` elements) a row's
    columns are spread over programs and a second launch sums the counts; the values are the same.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS = (
        Cell(
            why="tp1 and dep16 (c383): 16 rows of the 7795-page table (max_seq_len 997696, 128 "
            "tokens per page), one program per row",
            spec=dict(rows=16, blocks=7795, partial=False, divisor=5, block_offset=0),
        ),
        Cell(
            why="helix8 c24/c42: 2 rows, each row's columns over 4 programs, then the count sum",
            spec=dict(rows=2, blocks=7795, partial=True, divisor=1, block_offset=3),
        ),
        Cell(
            why="helix16_moetp4ep4 (ULL c1/c2/c4): 4 rows, the parallel decode",
            spec=dict(rows=4, blocks=7795, partial=True, divisor=1, block_offset=3),
        ),
        Cell(
            why="helix8 c240: 20 rows, the parallel decode at the largest helix batch",
            spec=dict(rows=20, blocks=7795, partial=True, divisor=1, block_offset=3),
        ),
    )

    note = """
    Integer arithmetic only; rows past ``rows`` of ``table`` and ``counts`` are not touched. The
    caller passes the metadata's ``kv_cache_block_offsets`` plane of the drafter pool and the
    worker's persistent table, counts and scratch. Launch shapes depend on host sizes only.
    """

    def __call__(
        self,
        src: torch.Tensor,
        divisor: int,
        table: torch.Tensor,
        counts: torch.Tensor,
        partial: Optional[torch.Tensor] = None,
        block_offset: int = 0,
    ) -> None:
        if partial is None:
            decode_draft_block_table(src, divisor, table, counts, block_offset=block_offset)
        else:
            decode_draft_block_table_parallel(
                src, divisor, table, counts, partial, block_offset=block_offset
            )

    def reference(
        self,
        src: torch.Tensor,
        divisor: int,
        table: torch.Tensor,
        counts: torch.Tensor,
        partial: Optional[torch.Tensor] = None,
        block_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(table, counts)`` after the call; the arguments are not written, ``partial`` is
        scratch and not read."""
        rows = src.shape[0]
        encoded = src.to(torch.int64) - int(block_offset)
        table_out, counts_out = table.clone(), counts.clone()
        table_out[:rows] = torch.div(encoded.clamp(min=0), int(divisor), rounding_mode="floor").to(
            torch.int32
        )
        counts_out[:rows] = (encoded >= 0).sum(dim=-1).to(counts.dtype)
        return table_out, counts_out

    def is_valid(
        self,
        src: torch.Tensor,
        divisor: int,
        table: torch.Tensor,
        counts: torch.Tensor,
        partial: Optional[torch.Tensor] = None,
        block_offset: int = 0,
    ) -> None:
        # The kernels index `table` with its row stride only and store int32 / int64 through raw
        # pointers: another dtype is written with the wrong width.
        assert table.dtype == torch.int32, "table must be int32"
        assert counts.dtype == torch.int64 and counts.stride(-1) == 1, "counts must be int64"
        assert int(divisor) > 0, "divisor must be positive"


k3_draft_block_table = K3DraftBlockTable()
