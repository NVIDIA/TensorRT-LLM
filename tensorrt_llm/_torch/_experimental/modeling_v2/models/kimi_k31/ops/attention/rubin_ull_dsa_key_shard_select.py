# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Helix key-shard decode indexer's local select + exact global Top-K merge: one call of the
key-shard workspace's select/pack runner (``kernels/attention/rubin_ull_dsa_index_select_pack.py``
``KeyShardSelectPack``; with the LL exchange, ``rubin_ull_dsa_index_local_topk.py``'s select / push grid and
its merge grid, or ``rubin_ull_dsa_index_exchange.py``)."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

PAGE = 64


def _cell(why: str, world: int, requests: int, next_n: int, context: int, **kw) -> Cell:
    spec = dict(world=world, requests=requests, next_n=next_n, context=context, kv_rows=True,
                bounds=True, hot=False)  # fmt: skip
    spec.update(kw)
    return Cell(why=why, spec=spec)


class DsaKeyShardSelect(OpWrapper):
    """Rows ``0 <= t < rows`` of the workspace ``runner.merge``: this rank's local logits
    ``merge.local_logits[t, :row_lens[t]]`` (local token ``c`` of this rank's KV shard at column
    ``c``) are one shard of row ``t``'s keys over the ``merge.world`` ranks of the group. The call
    pushes this rank's local Top-``merge.local_k`` to every rank and, with ``out``, writes this
    rank's part of the exact global Top-``merge.top_k`` of the union:

    - ``out[t, :counts[t]]``: the kept local tokens (with ``kv_rows``: their KV pool rows,
      ``block_table[req_idx[t], c // 64] * stride_rows + row_offset + c % 64``), ``-1`` after;
    - ``counts[t]``: how many this rank keeps; ``mask_out[t] = bounds[t] == 0 or counts[t] == 0``
      when ``mask_out`` is given.

    ``exchange``: through the LL exchange (each 16-byte pair tagged with the call's epoch; the
    served path); ``tops``: ``runner.tops`` holds this step's vector maxima (the scorer's,
    ``dsa_indexer_score(tops=...)``), from which rows of at least ``runner.tops_from`` keys select.
    Without ``out`` the call only publishes (``KeyShardTopKMerge.merge_owned`` follows).

    Collective over the group: every rank makes the same call with the same ``rows`` (each with its
    own runner), and a rank's result depends on every rank's logits.
    """

    ARCHS = frozenset({Arch.SM_103})

    # One GPU stands in for the group (every rank's workspace on it, every rank on its own stream):
    # a select and a merge CTA per row and rank must be resident at once, so world x rows <= 64.
    CELLS: tuple[Cell, ...] = (
        _cell(
            "16 ranks (served width), 225k context: 1 request x 4 rows from the maxima, KV "
            "pool rows + combine mask",
            16,
            1,
            4,
            225_000,
        ),  # fmt: skip
        _cell(
            "16 ranks, q1 row, local token indices, no bounds",
            16,
            1,
            1,
            70_000,
            kv_rows=False,
            bounds=False,
        ),  # fmt: skip
        _cell(
            "16 ranks, one hot rank holds most of the Top-K (exact re-selection)",
            16,
            1,
            4,
            100_000,
            hot=True,
        ),  # fmt: skip
        _cell("8 ranks, served b1: 1 request x next_n 8 rows (long rows)", 8, 1, 8, 120_000),
        _cell(
            "4 ranks, served b2 shape: 2 requests x 8 rows, unequal contexts",
            4,
            2,
            8,
            30_000,
            contexts=(30_000, 9_000),
        ),  # fmt: skip
        _cell(
            "2 ranks, served b4 shape: 4 requests x 8 rows (32 rows)",
            2,
            4,
            8,
            12_000,
            contexts=(12_000, 2_500, 700, 6_000),
        ),  # fmt: skip
    )

    note = """
    The K3.1 ULL generation target's indexer select + merge (Helix CP16: 16 ranks, local_k 256,
    top_k 2048) after ``dsa_indexer_score`` and before the sparse decode. The exchange path is exact:
    a rank with more of the global Top-K than it pushed is re-selected from its own logits (counted
    in ``merge.state[2]``). LL polls have a deadline of ``merge.max_spin`` microseconds; a timed-out
    poll counts in ``merge.state[3]``, then writes the ``kernels/common/rubin_ull_spin_trap.py``
    record (sites 171-174: flag merge, own LL slots, exchange merge, rescue; row / thread / epoch)
    and traps: its row's result would be undefined.

    The epoch: every select and merge CTA reads ``merge.state[0] + 1`` (the merge before its
    ``griddepcontrol.wait``, launched early by PDL) and the merge's last CTA publishes it into
    ``state[0]``. The two reads agree only because the previous call's merge completed before this
    call's select released its dependents; were the merge's read stale it would settle at once on
    the other parity block's pairs from two calls back, which carry the matching tag: stale keys,
    no timeout. The test asserts the agreement per call.

    Not certified here: 16 ranks at 8 rows (too many CTAs for one GPU), the flag merge
    (``exchange=False`` with ``out``), the cluster select (``tops`` off).
    """

    def __call__(
        self,
        runner,
        row_lens: torch.Tensor,
        rows: int,
        out: torch.Tensor | None = None,
        counts: torch.Tensor | None = None,
        bounds: torch.Tensor | None = None,
        mask_out: torch.Tensor | None = None,
        exchange: bool = False,
        tops: bool = False,
        kv_rows=None,
    ) -> None:
        runner(row_lens, rows, out, counts, bounds, mask_out, exchange=exchange, tops=tops,
               kv_rows=kv_rows)  # fmt: skip

    def reference(
        self,
        runner,
        row_lens: torch.Tensor,
        rows: int,
        out: torch.Tensor | None = None,
        counts: torch.Tensor | None = None,
        bounds: torch.Tensor | None = None,
        mask_out: torch.Tensor | None = None,
        exchange: bool = False,
        tops: bool = False,
        kv_rows=None,
        *,
        group: Sequence[tuple[torch.Tensor, torch.Tensor]] = (),
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor | None]:
        """A collective: ``group`` holds every rank's ``(local_logits, row_lens)`` in rank order.

        Returns ``(kept, counts, mask)``: per row the ascending kept local tokens of this rank (KV
        pool rows with ``kv_rows``) among the global Top-``top_k`` by score, their count, and the
        combine mask (None without ``mask_out``). ``out``, ``counts``, ``exchange`` and ``tops`` are
        not read: every path computes the same rows."""
        merge = runner.merge
        rank, k = merge.rank, merge.top_k
        kept, sizes = [], []
        for t in range(rows):
            scores, owner, column = [], [], []
            for r, (logits, lens) in enumerate(group):
                n = int(lens[t])
                scores.append(logits[t, :n].double())
                owner.append(torch.full((n,), r, dtype=torch.long, device=logits.device))
                column.append(torch.arange(n, device=logits.device))
            scores, owner, column = map(torch.cat, (scores, owner, column))
            top = torch.topk(scores, min(k, scores.numel())).indices
            mine = column[top[owner[top] == rank]].sort().values
            if kv_rows is not None:
                req = int(kv_rows.req_idx[t])
                table = kv_rows.block_table[req].long()
                page = mine // PAGE
                inside = page < table.shape[0]
                entry = torch.where(inside, table[page.clamp(max=table.shape[0] - 1)], -1)
                mine = torch.where(
                    entry >= 0, entry * kv_rows.stride_rows + kv_rows.row_offset + mine % PAGE, -1
                )
            kept.append(mine)
            sizes.append(mine.numel())
        sizes = torch.tensor(sizes, dtype=torch.int32, device=row_lens.device)
        mask = None
        if mask_out is not None:
            mask = (bounds[:rows] == 0) | (sizes == 0)
        return kept, sizes, mask

    def is_valid(
        self,
        runner,
        row_lens: torch.Tensor,
        rows: int,
        out: torch.Tensor | None = None,
        counts: torch.Tensor | None = None,
        bounds: torch.Tensor | None = None,
        mask_out: torch.Tensor | None = None,
        exchange: bool = False,
        tops: bool = False,
        kv_rows=None,
    ) -> None:
        """The runner asserts its operands' layout; it does not check that a row's keys fit the
        workspace, that the workspace's layout takes the LL exchange, or that the KV rows cover
        ``rows``."""
        merge = runner.merge
        lens = row_lens[:rows]
        if int(lens.max()) > merge.local_width:
            raise ValueError("dsa_key_shard_select: a row's keys run past the workspace width")
        if exchange and not runner.supports_exchange():
            raise ValueError("dsa_key_shard_select: the workspace does not support the LL exchange")
        if kv_rows is not None and kv_rows.req_idx.shape[0] < rows:
            raise ValueError("dsa_key_shard_select: fewer request indices than rows")


dsa_key_shard_select = DsaKeyShardSelect()
