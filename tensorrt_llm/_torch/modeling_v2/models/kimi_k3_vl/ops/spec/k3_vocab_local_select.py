# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-rank half of vocab-parallel token selection: each split of this rank's logits shard
reduced to a ``(value, global index)`` pair in the exchange buffer (Triton)."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_spec.vocab_parallel_kernels import _PairLayout, local_select
from ._reference import gumbel, philox4x32, publish_pairs


def sampled_scores(local: torch.Tensor, col_base: int, sampling: tuple) -> torch.Tensor:
    """FP64 ``[rows, shard]`` scores whose argmax is the draw: the logits for rows at or below
    the greedy threshold, else ``logits / T`` plus the Philox Gumbel noise of the global column."""
    temperatures, seeds, offsets, threshold = sampling
    rows, n_cols = local.shape
    device = local.device
    t = temperatures[:rows].double()
    greedy = t <= threshold
    inv_t = 1.0 / torch.where(greedy, torch.ones_like(t), t)
    cols = col_base + torch.arange(n_cols, device=device, dtype=torch.int64)
    seed = seeds[:rows].to(torch.int64)[:, None]
    offset = offsets[:rows].to(torch.int64)[:, None]
    zeros = torch.zeros((rows, n_cols), dtype=torch.int64, device=device)
    c0 = zeros + (cols // 4)[None, :]
    c1 = zeros + torch.arange(rows, device=device)[:, None]
    c2 = zeros + (offset & 0xFFFFFFFF)
    c3 = zeros + ((offset >> 32) & 0xFFFFFFFF)
    words = torch.stack(philox4x32(seed, c0, c1, c2, c3), dim=-1)
    noise = gumbel(words.gather(-1, (cols % 4).expand(rows, -1).unsqueeze(-1)).squeeze(-1))
    x = local.double()
    return torch.where(greedy[:, None], x, x * inv_t[:, None] + noise)


class K3VocabLocalSelect(OpWrapper):
    """Publish, for every row of ``local`` ``[rows, shard]`` (unit inner stride) and every split
    ``s`` of ``layout`` (columns ``[s * chunk, (s + 1) * chunk)``), the split's winner as float32
    ``(value, col_base + column)`` at columns ``2 (slot0 + s)`` and ``2 (slot0 + s) + 1`` of the
    row of ``layout.buffer``; with ``zero_width > 0`` the split-0 program also zeroes the row's
    columns outside ``[zero_lo, zero_hi)``, so that an all-reduce sum gathers. Returns None.

    The winner is the ``torch.argmax`` of the FP32 logits, or with ``sampling = (temperatures,
    seeds, offsets, greedy_threshold)`` the Gumbel-max draw ``argmax(logits / T + G)`` for rows
    with ``T > greedy_threshold``. The noise of global column ``v`` of row ``r`` is
    ``-log(-log(u))`` of output ``v % 4`` of ``Philox4x32-10(seed[r], (v // 4, r, offset[r]
    lo, offset[r] hi))``; every rank draws the same noise for a column.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    # One rank of the vocab-sharded drafter / target head (163840 columns over the helix group)
    # with the MNNVL exchange's layout; rows are requests x (draft length + 1).
    CELLS = (
        Cell(
            why="helix16 ULL c1, rank 0 of 16: 8 rows over a 10240-column shard, greedy, 5 splits",
            spec=dict(tp=16, rank=0, rows=8, sampling=False, ties=False),
        ),
        Cell(
            why="helix16 ULL c4, last rank: 32 rows, greedy",
            spec=dict(tp=16, rank=15, rows=32, sampling=False, ties=False),
        ),
        Cell(
            why="helix16 ULL c4, rank 5: 32 rows of Gumbel-max sampling, some rows greedy",
            spec=dict(tp=16, rank=5, rows=32, sampling=True, ties=False),
        ),
        Cell(
            why="helix8 c24/c42, rank 0 of 8: 16 rows over a 20480-column shard, 10 splits",
            spec=dict(tp=8, rank=0, rows=16, sampling=False, ties=False),
        ),
        Cell(
            why="helix8 c75/c174, last rank: 64 rows, 4 splits of 6144 columns, the last partial",
            spec=dict(tp=8, rank=7, rows=64, sampling=False, ties=False),
        ),
        Cell(
            why="helix8 c75/c174, rank 2: maxima tied across splits and NaN rows (argmax order)",
            spec=dict(tp=8, rank=2, rows=64, sampling=False, ties=True),
        ),
        Cell(
            why="helix8 c240, rank 3: 120 rows (20 x 6) of sampling, 3 splits of 8192",
            spec=dict(tp=8, rank=3, rows=120, sampling=True, ties=False),
        ),
    )

    note = """
    ``layout`` is a ``_PairLayout`` of ``_PairExchange.layout``: its buffer is new each call, and
    the exchange (MNNVL all-reduce, NCCL all-gather, or none on one rank) follows. Grid (rows,
    splits), 8 warps, 2048 columns per iteration. ``col_base`` and ``chunk`` are multiples of 4,
    so Philox groups never straddle splits. The draws follow ``softmax(logits / T)``; they are
    not the gathered sampler's draws.
    """

    def __call__(
        self,
        local: torch.Tensor,
        col_base: int,
        layout: _PairLayout,
        sampling: Optional[tuple] = None,
    ) -> None:
        local_select(local, col_base, layout, sampling)

    def reference(
        self,
        local: torch.Tensor,
        col_base: int,
        layout: _PairLayout,
        sampling: Optional[tuple] = None,
    ) -> torch.Tensor:
        """``layout.buffer`` after the call (a clone). Sampled values are computed in FP64."""
        if sampling is None:
            return publish_pairs(layout, local.float(), col_base)
        return publish_pairs(layout, sampled_scores(local, col_base, sampling), col_base)

    def is_valid(
        self,
        local: torch.Tensor,
        col_base: int,
        layout: _PairLayout,
        sampling: Optional[tuple] = None,
    ) -> None:
        assert local.dim() == 2 and local.stride(-1) == 1, "local must be [rows, shard], unit inner"
        assert layout.buffer.dtype == torch.float32, "pairs are float32"
        assert layout.chunk % 4 == 0 and col_base % 4 == 0, "Philox groups straddle splits"
        assert layout.splits * layout.chunk >= local.shape[1], "the splits miss columns"
        # Global indices travel as float32: exact only below 2**24.
        assert col_base + local.shape[1] <= 2**24, "vocab too large for float32 indices"


k3_vocab_local_select = K3VocabLocalSelect()
