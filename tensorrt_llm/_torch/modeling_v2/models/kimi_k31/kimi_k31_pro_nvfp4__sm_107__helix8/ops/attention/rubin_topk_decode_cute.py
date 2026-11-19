# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact radix Top-K of short decode rows (CuTe DSL): the helix key shard's local Top-L.

Under helix CP each rank scores only its own pages of every sequence, so its rows are about
``1 / cp`` of the context; the radix kernel is fast on such rows and its tie order does not matter
to the key-shard merge, whose certificate compares scores only. The engine op is called as it is.
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class RubinTopkDecodeCute(OpWrapper):
    """``indices[r]`` = ``k`` columns of ``logits[r, :n_r]`` holding its ``k`` largest values
    (``k`` = ``top_k``), ``-1`` padded, ``n_r = row_lens[r // next_n] - next_n + r % next_n + 1``;
    a row with ``n_r <= k`` gets ``0 .. n_r - 1`` then ``-1``. Among equal values at the boundary
    the kernel picks any. ``logits`` FP32 ``[R, >= n_r]`` with unit column stride; ``row_lens``
    int32; ``indices`` int32 ``[R, k]``, written in place and returned."""

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="helix8 key shard b16 k5: 96 rows of about 1/8 of 24k keys, local Top-512",
            spec=dict(rows=96, row_lens=(1500, 4000), width=4096, top_k=512, next_n=1),
        ),
        Cell(
            why="rows shorter than the local Top-K: the identity and -1 tail",
            spec=dict(rows=12, row_lens=(100, 600), width=1024, top_k=512, next_n=1),
        ),
        Cell(
            why="empty rows (a rank that owns none of a short sequence's pages)",
            spec=dict(rows=8, row_lens=(0, 0), width=256, top_k=512, next_n=1),
        ),
    )
    note = """
    ``trtllm::cute_dsl_indexer_topk_decode`` (engine). The test compares each row's selected
    values (a multiset), since the kernel may resolve boundary ties either way.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        row_lens: torch.Tensor,
        indices: torch.Tensor,
        top_k: int,
        next_n: int,
    ) -> torch.Tensor:
        torch.ops.trtllm.cute_dsl_indexer_topk_decode(logits, row_lens, indices, top_k, next_n)
        return indices

    def reference(
        self,
        logits: torch.Tensor,
        row_lens: torch.Tensor,
        indices: torch.Tensor,
        top_k: int,
        next_n: int,
    ) -> torch.Tensor:
        """A new ``[R, top_k]`` tensor: each row's Top-K columns (ties by the smaller column) in
        ascending order, then -1; ``indices`` is left untouched."""
        rows = logits.shape[0]
        out = torch.full((rows, top_k), -1, dtype=torch.int32, device=logits.device)
        for r in range(rows):
            n = int(row_lens[r // next_n]) - next_n + r % next_n + 1
            if n <= 0:
                continue
            if n <= top_k:
                out[r, :n] = torch.arange(n, dtype=torch.int32, device=logits.device)
                continue
            order = torch.sort(-logits[r, :n].double(), stable=True).indices[:top_k]
            out[r] = torch.sort(order).values.to(torch.int32)
        return out

    def is_valid(
        self,
        logits: torch.Tensor,
        row_lens: torch.Tensor,
        indices: torch.Tensor,
        top_k: int,
        next_n: int,
    ) -> None:
        # The kernel reads n_r columns of each row without checking the width.
        assert int(row_lens.max()) <= logits.shape[1], "a row length past the logits' width"
        assert logits.dtype == torch.float32 and logits.stride(1) == 1
        assert indices.shape == (logits.shape[0], top_k) and indices.dtype == torch.int32


rubin_topk_decode_cute = RubinTopkDecodeCute()
