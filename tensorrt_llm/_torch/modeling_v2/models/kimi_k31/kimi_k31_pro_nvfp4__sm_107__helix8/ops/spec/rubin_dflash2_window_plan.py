# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 drafter's window plan on SM107: which keys each block query attends, once a step.

One launch per generation step, before the five drafter layers, from the history length
``num_ctx`` the fused prepare advertises (the block's RoPE base). The plan splits each request's
window into a body shared by its block queries, a short per-query head and the block itself
(``kernels/spec/rubin_dflash2_window``), keeps only this CP rank's keys under helix, and cuts the
body into the step's fixed number of page-aligned chunks. The window entry
(``rubin_dflash2_window_triton``) reads it.
"""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Cell, OpWrapper

from ...kernels.spec.rubin_dflash2_window import (
    DFlash2WindowPlan,
    body_page_count,
    head_width,
    launch_window_plan,
    plan_buffers,
    plan_view,
    window_plan_torch,
)

#: The K3.1 DFlash2 drafter's sliding window and the draft pool's page size.
_WINDOW = 1024
_PAGE = 64
#: One history per request at every window boundary: empty, one key, a page edge, the window
#: edge (``W - blk`` .. ``W + 7``), several pages past it, and the 190k contexts of the workload.
_DEP16_LENGTHS = (0, 1, 63, 65, 1017, 1023, 1024, 1031, 4097, 9000, 65536, 190000)
_HELIX8_LENGTHS = (0, 1, 5, 60, 63, 64, 66, 130, 1017, 1022, 1025, 1031, 4097, 9000, 65536, 190000)


def _spec(batch: int, block_size: int, cp_size: int, num_splits: int, **overrides) -> dict:
    spec = dict(
        batch=batch,
        block_size=block_size,
        window=_WINDOW,
        page_size=_PAGE,
        cp_size=cp_size,
        num_splits=num_splits,
        lengths={12: _DEP16_LENGTHS, 16: _HELIX8_LENGTHS}.get(batch, _DEP16_LENGTHS[:batch]),
        table_rows=False,
    )
    spec.update(overrides)
    return spec


class RubinDFlash2WindowPlan(OpWrapper):
    """Fill ``plan`` for the ``B = num_ctx.numel()`` requests of a step; returns its views.

    For request ``b`` with history ``L = num_ctx[b]`` and block queries ``j < block_size`` at
    ``L + j``: ``body_pages[b]`` / ``body_len[b]`` hold this rank's body (history positions
    ``[S, L)``, ``S = min(L, ceil(lo_max / P) * P)``), cut into ``num_splits`` page-aligned
    chunks by ``chunk_bounds[b]``; ``head_rows[b]`` the pool rows of this rank's head keys
    (positions ``[lo_0, S)``) and ``head_first[b, j]`` the first one inside query ``j``'s window;
    ``block_bits[b]`` the block positions this rank owns. With ``cp_size > 1`` position ``x``
    belongs to rank ``(x // P) % cp_size`` at local page column ``x // P // cp_size``.

    ``page_table`` is the draft cache's page table (int32, unit inner stride) and ``table_rows``
    the row of each request in it (None: row ``b``); only the columns of the window are read.
    ``plan`` comes from ``plan_buffers`` (persistent, allocated before capture); ``num_splits``
    is the step's static split count.
    """

    ARCHS = frozenset()

    CELLS: tuple[Cell, ...] = (
        Cell(why="DEP16 generation, 12 requests at draft 7, 2 chunks", spec=_spec(12, 8, 1, 2)),
        Cell(
            why="helix8 generation, 16 requests at draft 5, every CP rank, 2 chunks",
            spec=_spec(16, 6, 8, 2),
        ),
        Cell(
            why="TEP8 generation, one request at draft 7 cut into 16 chunks",
            spec=_spec(1, 8, 1, 16, lengths=(190000,)),
        ),
        Cell(
            why="CP4: empty shards, a body page and a block split between ranks",
            spec=_spec(
                15,
                6,
                4,
                3,
                lengths=(0, 1, 60, 64, 66, 127, 128, 130, 191, 255, 256, 1000, 1030, 4097, 9000),
            ),
        ),
        Cell(
            why="CP2, a batch at rows of the shared table (the unified draft cache)",
            spec=_spec(4, 8, 2, 4, lengths=(5, 1500, 1031, 9000), table_rows=True),
        ),
        Cell(
            why="dummy rows of a padded graph: every history shorter than the block",
            spec=_spec(8, 8, 1, 2, lengths=tuple(range(8))),
        ),
    )

    note = """
    Plain integer work, one program per request; exact against `reference`. The plan writes
    every field of every planned row, so stale rows of a larger earlier batch are never read.
    It reads `num_ctx` as the history length: the fused prepare advertises
    `min(new_len, capacity * cp - blk)`, which equals the block's RoPE base while the draft
    reserve stays at K + 1 (the op test checks `num_ctx == query_positions[:, 0]`).
    """

    def __call__(
        self,
        num_ctx: torch.Tensor,
        page_table: torch.Tensor,
        table_rows: Optional[torch.Tensor],
        block_size: int,
        window: int,
        page_size: int,
        cp_size: int,
        cp_rank: int,
        num_splits: int,
        plan: DFlash2WindowPlan,
    ) -> DFlash2WindowPlan:
        return launch_window_plan(
            num_ctx,
            page_table,
            table_rows,
            block_size,
            window,
            page_size,
            cp_size,
            cp_rank,
            num_splits,
            plan,
        )

    def reference(
        self,
        num_ctx: torch.Tensor,
        page_table: torch.Tensor,
        table_rows: Optional[torch.Tensor],
        block_size: int,
        window: int,
        page_size: int,
        cp_size: int,
        cp_rank: int,
        num_splits: int,
        plan: DFlash2WindowPlan,
    ) -> DFlash2WindowPlan:
        """The plan in plain torch from the window definition (``plan`` unused: new tensors)."""
        return window_plan_torch(
            num_ctx,
            page_table,
            table_rows,
            block_size,
            window,
            page_size,
            cp_size,
            cp_rank,
            num_splits,
        )

    def is_valid(
        self,
        num_ctx: torch.Tensor,
        page_table: torch.Tensor,
        table_rows: Optional[torch.Tensor],
        block_size: int,
        window: int,
        page_size: int,
        cp_size: int,
        cp_rank: int,
        num_splits: int,
        plan: DFlash2WindowPlan,
    ) -> None:
        batch = num_ctx.shape[0]
        assert num_ctx.dim() == 1 and not num_ctx.is_floating_point(), "[B] integer lengths"
        assert page_table.dim() == 2 and page_table.dtype in (torch.int32, torch.int64)
        assert page_table.stride(1) == 1, "unit inner stride"
        assert table_rows is None or table_rows.shape == (batch,)
        assert 0 < block_size < window and block_size <= 31, "a block bitmask per request"
        assert page_size & (page_size - 1) == 0, "power-of-two pages"
        assert 0 <= cp_rank < cp_size
        assert plan.body_pages.shape[1] == body_page_count(window, page_size, block_size, cp_size)
        assert plan.head_rows.shape[1] == head_width(page_size, block_size)
        assert plan.head_first.shape[1] == block_size
        assert 1 <= num_splits < plan.chunk_bounds.shape[1], "buffers hold num_splits chunks"
        assert all(t.shape[0] >= batch for t in plan), "buffers hold every request"
        assert all(t.dtype == torch.int32 and t.is_contiguous() for t in plan)
        # The table must cover the history: a column past it is clamped, not an error.
        if batch:
            longest = int(num_ctx.max())
            need = -(-longest // page_size)
            assert -(-need // cp_size) <= page_table.shape[1], "page table shorter than history"


rubin_dflash2_window_plan = RubinDFlash2WindowPlan()

__all__ = [
    "DFlash2WindowPlan",
    "plan_buffers",
    "plan_view",
    "rubin_dflash2_window_plan",
]
