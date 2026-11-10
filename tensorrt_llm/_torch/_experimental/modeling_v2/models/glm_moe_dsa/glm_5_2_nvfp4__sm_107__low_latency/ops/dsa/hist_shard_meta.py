# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""This rank's key shard of the generation request: its context length and block table."""

from __future__ import annotations

from typing import Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._hist_topk import shard_capacity, shard_size

#: c1: 16 ranks, 64-token pages, and the shard table of the gen server's indexer capacity.
_C1 = dict(group_size=16, tokens_per_block=64, capacity=shard_capacity(1_048_596, 16))


def _spec(kv_len, rank, **overrides) -> dict:
    spec = dict(_C1, kv_len=kv_len, rank=rank)
    spec.update(overrides)
    return spec


class GlmHistShardMeta(OpWrapper):
    """``shard_ctx[0]`` and ``shard_block_table[0, :]`` of rank ``rank`` for one request.

    With ``S = shard_size(kv_lens[0], group_size)`` (ceil(kv_len / group_size) rounded up to 128),
    the rank holds the keys [rank * S, rank * S + n), ``n = clamp(kv_len - rank * S, 0, S)``:

    - ``shard_ctx[0] = max(n, 1)`` (an empty shard scores one key no row reaches);
    - ``shard_block_table[0, j] = block_table[rank * S / tokens_per_block + j]`` for the shard's
      ``ceil(n / tokens_per_block)`` pages, 0 for the rest of the row.

    ``block_table`` is the request's INDEX_KEY block-table row (one-dimensional). One CTA. Returns
    ``(shard_ctx, shard_block_table)``.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(why="c1: a 530,000-key request on rank 0", spec=_spec(530_000, 0)),
        Cell(why="c1: the same request's last rank, a partial shard", spec=_spec(530_000, 15)),
        Cell(why="the full 1,048,596-key capacity on rank 15", spec=_spec(1_048_596, 15)),
        Cell(why="a short request: rank 13 holds no key, one unread key", spec=_spec(3_000, 13)),
        Cell(
            why="a shard ending inside a page: 1,000,003 keys on rank 15",
            spec=_spec(1_000_003, 15),
        ),
    )

    note = """
    Metadata consumed: none. The target passes the generation request's indexer KV length (int32,
    on the device), its row of the INDEX_KEY block offsets, persistent ``shard_ctx`` [1] and
    ``shard_block_table`` [1, ceil(capacity / 64)] int32 buffers (captured CUDA graphs read
    them; capacity = ``_hist_topk.shard_capacity(indexer max_seq_len, 16)``), the group size
    (16), this rank's TP rank and the page size (64).

    The kernel refuses a group above 16 ranks and a page size that does not divide 128.
    """

    def __call__(
        self,
        kv_lens: torch.Tensor,
        block_table: torch.Tensor,
        shard_ctx: torch.Tensor,
        shard_block_table: torch.Tensor,
        group_size: int,
        rank: int,
        tokens_per_block: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        torch.ops.trtllm.glm_dsa_hist_shard_meta(
            kv_lens, block_table, shard_ctx, shard_block_table, group_size, rank, tokens_per_block
        )
        return shard_ctx, shard_block_table

    def reference(
        self,
        kv_lens: torch.Tensor,
        block_table: torch.Tensor,
        shard_ctx: torch.Tensor,
        shard_block_table: torch.Tensor,
        group_size: int,
        rank: int,
        tokens_per_block: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fresh ``(shard_ctx, shard_block_table)`` computed on the host."""
        kv_len = int(kv_lens[0])
        size = shard_size(kv_len, group_size)
        base = rank * size
        n = min(max(kv_len - base, 0), size)
        first_page = base // tokens_per_block
        pages = -(-n // tokens_per_block)
        row = block_table.tolist()
        width = shard_block_table.numel()
        table = [
            row[first_page + j] if j < pages and first_page + j < len(row) else 0
            for j in range(width)
        ]
        ctx = torch.tensor([max(n, 1)], dtype=torch.int32, device=shard_ctx.device)
        out = torch.tensor(table, dtype=torch.int32, device=shard_block_table.device)
        return ctx, out.view(shard_block_table.shape)

    def is_valid(
        self,
        kv_lens: torch.Tensor,
        block_table: torch.Tensor,
        shard_ctx: torch.Tensor,
        shard_block_table: torch.Tensor,
        group_size: int,
        rank: int,
        tokens_per_block: int,
    ) -> None:
        # The kernel reads the first row of a two-dimensional table: another request's pages.
        assert block_table.dim() == 1, "block_table must be the request's one-dimensional row"
        # A table narrower than the shard's pages drops pages the logits then read as page 0.
        kv_len = int(kv_lens[0])
        size = shard_size(kv_len, group_size)
        n = min(max(kv_len - rank * size, 0), size)
        assert shard_block_table.numel() * tokens_per_block >= n, (
            f"shard_block_table holds {shard_block_table.numel()} pages, the shard {n} keys"
        )
        # Pages past the request's row are read as 0 instead of failing.
        if n > 0:
            first_page = rank * size // tokens_per_block
            assert first_page + -(-n // tokens_per_block) <= block_table.numel(), (
                "the shard's pages run past the request's block-table row"
            )


hist_shard_meta = GlmHistShardMeta()
