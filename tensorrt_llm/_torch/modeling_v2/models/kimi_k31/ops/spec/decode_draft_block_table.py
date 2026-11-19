# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode a draft KV pool's encoded block offsets into pool block indices and per-row block extents."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.speculative.spec_tail_kernels import (
    decode_draft_block_table as _decode_draft_block_table,
)


class DecodeDraftBlockTable(OpWrapper):
    """Write `table[i, j] = max(src[i, j], 0) // divisor` (int32) and `counts[i]` = one past the last
    `j` with `src[i, j] >= 0`, 0 when none (int64), for every row i of `src`, in place; returns None.

    `src` is [rows, blocks] (any strides, any integer dtype; negative entries are unused slots);
    `table` is [>= rows, blocks] with a unit inner stride, `counts` [>= rows]. Rows of `table` and
    `counts` past `rows` are left as they are.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"{rows} rows x {blocks} blocks, divisor {divisor}: {why}",
            spec=dict(rows=rows, blocks=blocks, divisor=divisor, table_rows=table_rows),
        )
        for rows, blocks, divisor, table_rows, why in (
            (1, 64, 2, 8, "one decode request, K/V pair pool (kv_factor 2)"),
            (16, 257, 2, 64, "a decode batch just past 32k context over 128-token pages"),
            (64, 1024, 2, 64, "64 slots x 128k context over 128-token pages"),
            (8, 40, 1, 32, "kv_factor 1 pool, a block count that is not a power of two"),
            (2, 4100, 2, 4, "more than 2048 blocks: the kernel loops over its 2048-wide row block"),
        )
    )
    note = """
    The DFlash worker of the Kimi K3.1 target calls it once per step, before the
    drafter runs, when the drafter's context K / V lives in a draft KV pool of
    its own (not the unified pool and not the private arena): `src` is that
    pool's plane-0 rows of the TRTLLM metadata's `draft_kv_cache_block_offsets`,
    `divisor` the pool's kv factor, `table` / `counts` the worker's persistent
    block-table buffers (fixed addresses, so a captured graph keeps reading them).
    The K3.1 DFlash2 MLA drafter runs on the VANILLA drafter backend, for which the
    engine builds no separate draft KV manager: its context latents live in the
    worker's private paged arena and this call is not reached on the target's
    default path. It is certified because the worker's separate-pool path calls it.

    One Triton program per row, a fixed block width per launch; integer
    arithmetic only, exact and deterministic. It compiles on first use (warm it
    outside CUDA-graph capture). No metadata, workspace or other state.

    ARCHS lists only what was driven: sm_103 (GB300) and sm_107 (VR200).
    """

    def __call__(
        self, src: torch.Tensor, divisor: int, table: torch.Tensor, counts: torch.Tensor
    ) -> None:
        _decode_draft_block_table(src, divisor, table, counts)

    def reference(
        self, src: torch.Tensor, divisor: int, table: torch.Tensor, counts: torch.Tensor
    ) -> None:
        """Writes the same rows of `table` and `counts` with torch ops."""
        rows = src.shape[0]
        encoded = src.to(torch.int64)
        table[:rows] = torch.div(encoded.clamp(min=0), divisor, rounding_mode="floor").to(
            table.dtype
        )
        # One past the last backed entry, as positions are indexed from entry 0.
        ends = torch.arange(1, encoded.shape[1] + 1, device=encoded.device).expand_as(encoded)
        extent = torch.where(encoded >= 0, ends, torch.zeros_like(ends))
        if rows and encoded.shape[1]:
            counts[:rows] = extent.max(dim=1).values.to(counts.dtype)
        elif rows:
            counts[:rows] = 0

    def is_valid(
        self, src: torch.Tensor, divisor: int, table: torch.Tensor, counts: torch.Tensor
    ) -> None:
        # One program per row of `src` writes row `i` of `table` and entry `i` of `counts`
        # unguarded: shorter outputs are written past their end.
        rows = src.shape[0]
        assert table.shape[0] >= rows and counts.shape[0] >= rows, (
            f"table ({table.shape[0]} rows) and counts ({counts.shape[0]}) must cover the "
            f"{rows} rows of src"
        )
        assert divisor > 0, "divisor must be positive"


decode_draft_block_table = DecodeDraftBlockTable()
