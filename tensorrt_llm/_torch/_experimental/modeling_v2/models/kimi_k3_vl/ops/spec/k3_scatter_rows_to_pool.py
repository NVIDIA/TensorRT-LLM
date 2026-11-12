# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scatter of the MLA drafter's block latents into its paged pool (Triton, one program per
row)."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_spec.spec_tail_kernels import scatter_rows_to_pool
from .k3_store_context_latents import to_e4m3_saturated


class K3ScatterRowsToPool(OpWrapper):
    """``pool[pages[i], offsets[i]] = src[i]`` for a ``[pages, page_size, D]`` pool view; an FP8
    E4M3 pool stores ``to_draft_fp8(src[i])``. With ``rows_mask`` (bool or integer ``[rows]``,
    contiguous) rows where it is zero write nothing. Returns None.

    ``src`` is ``[rows, D]`` with a unit inner stride, the pool's dtype or BF16 into FP8.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS = (
        Cell(
            why="dep16 (c383): 16 blocks of 6 latents (576 wide) into the FP8 pool, every row",
            spec=dict(rows=96, rows_mask=False),
        ),
        Cell(
            why="helix8 c75/c174: 8 blocks of 8, only the rows on this rank's pages",
            spec=dict(rows=64, rows_mask=True),
        ),
        Cell(
            why="helix16_moetp4ep4 (ULL c4): 4 blocks of 8, only this rank's rows",
            spec=dict(rows=32, rows_mask=True),
        ),
    )

    note = """
    The eager advanced-index assignment as one launch; two rows on one slot race (the drafter's
    block positions are distinct). A Helix drafter passes the block positions this rank owns as
    ``rows_mask``.
    """

    def __call__(
        self,
        src: torch.Tensor,
        pool: torch.Tensor,
        pages: torch.Tensor,
        offsets: torch.Tensor,
        rows_mask: Optional[torch.Tensor] = None,
    ) -> None:
        scatter_rows_to_pool(src, pool, pages, offsets, rows_mask=rows_mask)

    def reference(
        self,
        src: torch.Tensor,
        pool: torch.Tensor,
        pages: torch.Tensor,
        offsets: torch.Tensor,
        rows_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The pool after the call (a clone)."""
        new = pool.clone()
        write = torch.ones(src.shape[0], dtype=torch.bool, device=src.device)
        if rows_mask is not None:
            write = rows_mask.reshape(-1) != 0
        vals, target = src, new
        if new.dtype == torch.float8_e4m3fn:
            # Assigned as bytes: the bits are the same.
            vals, target = to_e4m3_saturated(src).view(torch.uint8), new.view(torch.uint8)
        target[pages.to(torch.int64)[write], offsets.to(torch.int64)[write]] = vals[write]
        return new

    def is_valid(
        self,
        src: torch.Tensor,
        pool: torch.Tensor,
        pages: torch.Tensor,
        offsets: torch.Tensor,
        rows_mask: Optional[torch.Tensor] = None,
    ) -> None:
        keys = pages.reshape(-1).to(torch.int64) * pool.shape[1] + offsets.reshape(-1)
        if rows_mask is not None:
            keys = keys[rows_mask.reshape(-1) != 0]
        # Two rows on one slot: which one lands is unspecified.
        assert keys.unique().numel() == keys.numel(), "rows target the same slot"


k3_scatter_rows_to_pool = K3ScatterRowsToPool()
