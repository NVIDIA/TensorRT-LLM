# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Key-sharded decode Top-K of the helix8 target: the pack and merge entries, and the workspace builder.

Under helix CP each rank scores only its own pages of every sequence (the paged logits entry writes
them into ``ws.local_logits``), takes its local Top-``local_k`` (``rubin_topk_decode_cute`` into
``ws.local_topk``), then:

* ``rubin_dsa_key_shard_pack`` pushes those candidates into every rank's receive slot;
* ``rubin_dsa_key_shard_merge`` selects each row's global Top-K and writes this rank's share of it
  as local token indices, their counts and, with per-token bounds, the combine mask.

The workspace is symmetric memory over the CP group, so ``rubin_dsa_key_shard_ws`` (a builder, not
an entry) allocates it once, collectively, in ``post_load_weights`` and outside any CUDA graph
capture. Every launch takes it as an argument.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Cell, OpWrapper

from ...kernels.attention.rubin_dsa_key_shard import (
    KEY_SHARD_TOKENS_PER_BLOCK,
    KeyShardWorkspace,
    key_shard_bytes_needed,
    key_shard_local_k,
    key_shard_merge,
    key_shard_pack,
    key_shard_workspace,
)

# MNNVL allocations are made in units of this many bytes.
_MNNVL_GRANULARITY = 1 << 21


def _order_keys(scores: torch.Tensor, gidx: torch.Tensor) -> torch.Tensor:
    """The kernels' candidate key: ``(ordered score bits << 32) | (2^32 - 1 - global index)``, so a
    larger key is a better candidate (``-0.0`` ranks below ``+0.0``) and ties go to the smaller index."""
    bits = scores.float().contiguous().view(torch.int32).long()
    ordered = torch.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
    return (ordered << 32) | (4294967295 - gidx.long())


def _global_index(local: torch.Tensor, rank: int, world: int) -> torch.Tensor:
    tpb = KEY_SHARD_TOKENS_PER_BLOCK
    return ((local // tpb) * world + rank) * tpb + local % tpb


def _exact_shares(logits_by_rank, row_lens_by_rank, top_k: int) -> List[List[torch.Tensor]]:
    """Every rank's share of each row's exact global Top-K, as ascending local indices."""
    world = len(logits_by_rank)
    rows = logits_by_rank[0].shape[0]
    shares = [[None] * rows for _ in range(world)]
    for r in range(rows):
        keys, owner, local = [], [], []
        for w in range(world):
            n = int(row_lens_by_rank[w][r])
            cols = torch.arange(n, device=logits_by_rank[w].device)
            keys.append(_order_keys(logits_by_rank[w][r, :n], _global_index(cols, w, world)))
            owner.append(torch.full((n,), w, dtype=torch.long))
            local.append(cols)
        keys, owner, local = torch.cat(keys), torch.cat(owner), torch.cat(local)
        chosen = torch.argsort(keys, descending=True)[:top_k]
        for w in range(world):
            mine = local[chosen][owner[chosen] == w]
            shares[w][r] = torch.sort(mine).values.to(torch.int32)
    return shares


class RubinDsaKeyShardPack(OpWrapper):
    """Push this rank's candidates of every row into every rank's receive slot of ``ws``.

    ``local_topk`` int32 ``[rows, ws.local_k]``: this rank's Top-``local_k`` local columns of each
    row of ``ws.local_logits`` (-1 padded); ``row_lens`` int32 ``[rows]``: each row's local key
    count. Writes only ``ws``; returns None.
    """

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="helix8 GEN b16 k5: 96 rows, 8 ranks, local Top-512 of about 1/8 of 24k keys",
            spec=dict(world=8, rows=96, row_lens=(1500, 4000), top_k=2048),
        ),
        Cell(
            why="rows where one rank holds more than its 512 of the global Top-K (exact fallback)",
            spec=dict(world=4, rows=8, row_lens=(3000, 6000), top_k=2048, skew=True),
        ),
        Cell(
            why="short rows: fewer keys than the Top-K on every rank",
            spec=dict(world=8, rows=12, row_lens=(10, 200), top_k=2048),
        ),
    )
    note = """
    Collective with ``rubin_dsa_key_shard_merge``: every rank of the group packs, then every rank
    merges, in the same order of calls. Multi-rank: certified by the multi-rank test, which runs
    every rank of a group on one GPU over plain local buffers and on 4 / 8 GPUs over MNNVL.
    """

    def __call__(
        self, local_topk: torch.Tensor, row_lens: torch.Tensor, ws: KeyShardWorkspace
    ) -> None:
        key_shard_pack(ws, local_topk, row_lens)

    def reference(
        self,
        local_logits_by_rank: Sequence[torch.Tensor],
        local_topk_by_rank: Sequence[torch.Tensor],
        row_lens_by_rank: Sequence[torch.Tensor],
    ) -> List[torch.Tensor]:
        """The group's inputs in rank order (the collective exception): each rank's candidate keys
        as it pushes them, ``[rows, local_k]`` int64 in ascending local position, padded with the
        smallest int64."""
        world = len(local_logits_by_rank)
        out = []
        for w in range(world):
            topk = local_topk_by_rank[w]
            idx = torch.where(topk >= 0, topk, 2147483647).sort(dim=1).values
            valid = idx != 2147483647
            safe = torch.where(valid, idx, 0).long()
            scores = torch.gather(local_logits_by_rank[w], 1, safe)
            keys = _order_keys(scores, _global_index(safe, w, world))
            out.append(torch.where(valid, keys, torch.iinfo(torch.int64).min))
        return out

    def is_valid(self, local_topk: torch.Tensor, row_lens: torch.Tensor, ws: Any) -> None:
        # Rows past the workspace's capacity land in another row's receive cells.
        assert local_topk.shape[0] <= ws.rows_max, "more rows than the workspace holds"
        assert local_topk.shape[1] == ws.local_k and local_topk.is_contiguous()
        assert row_lens.dtype == torch.int32 and int(row_lens.max()) <= ws.local_width


class RubinDsaKeyShardMerge(OpWrapper):
    """This rank's share of each row's global Top-K, from the candidates every rank packed.

    Writes ``decode_rows`` int32 ``[rows, ws.top_k]``: this rank's winners as local token indices,
    in ascending order, then -1; ``counts`` int32 ``[rows]`` (or ``ws.counts`` when None): their
    number; and with ``row_bounds`` (each row's rank-local KV bound) ``mask_out`` bool ``[rows]``:
    ``(row_bounds == 0) | (count == 0)``. Selection order: score descending, then the global token
    index ascending; a row where a rank's unsent keys could win is re-selected exactly from every
    rank's local logits, or with ``failing_only`` from those ranks' local logits and the other
    ranks' candidates (the same values; every rank's logits when every rank's keys could win).
    Returns None.
    """

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="helix8 GEN b16 k5: 96 rows over 8 ranks, with per-token bounds and the mask",
            spec=dict(world=8, rows=96, row_lens=(1500, 4000), top_k=2048, bounds=True),
        ),
        Cell(
            why="the exact fallback: one rank holds more than 512 of a row's Top-K",
            spec=dict(world=4, rows=8, row_lens=(3000, 6000), top_k=2048, skew=True, bounds=True),
        ),
        Cell(
            why="no per-token bounds: counts only, no mask",
            spec=dict(world=8, rows=16, row_lens=(1000, 3000), top_k=2048, bounds=False),
        ),
        Cell(
            why="ties: integer scores, so the global index decides",
            spec=dict(world=4, rows=8, row_lens=(800, 1600), top_k=2048, bounds=True, ties=True),
        ),
        Cell(
            why="failing_only: one rank holds more than 512 of the Top-K; only its logits are read",
            spec=dict(
                world=4,
                rows=8,
                row_lens=(3000, 6000),
                top_k=2048,
                skew=True,
                bounds=True,
                failing_only=True,
            ),
        ),
        Cell(
            why="failing_only at 8 ranks: two ranks' logits are read, the others' candidates",
            spec=dict(
                world=8,
                rows=16,
                row_lens=(3000, 6000),
                top_k=2048,
                skew=2,
                bounds=True,
                failing_only=True,
            ),
        ),
        Cell(
            why="failing_only with integer ties: every rank fails, so every rank's logits are read",
            spec=dict(
                world=4,
                rows=8,
                row_lens=(800, 1600),
                top_k=2048,
                bounds=True,
                ties=True,
                failing_only=True,
            ),
        ),
    )
    note = """
    Collective (see ``rubin_dsa_key_shard_pack``). The selected set equals the engine's
    ``merge_helix_topk`` of each rank's exact local Top-K wherever the scores are distinct; the
    multi-rank test checks both. The epoch and the counters (merges, fallback rows, spin timeouts)
    are in ``ws.state``. ``failing_only`` changes which logits a re-selected row reads, not its
    values: the fallback-row counter counts the same rows either way.
    """

    def __call__(
        self,
        decode_rows: torch.Tensor,
        counts: Optional[torch.Tensor],
        row_bounds: Optional[torch.Tensor],
        mask_out: Optional[torch.Tensor],
        ws: KeyShardWorkspace,
        failing_only: bool = False,
    ) -> None:
        key_shard_merge(ws, decode_rows, counts, row_bounds, mask_out, failing_only)

    def reference(
        self,
        local_logits_by_rank: Sequence[torch.Tensor],
        row_lens_by_rank: Sequence[torch.Tensor],
        rank: int,
        top_k: int,
        row_bounds: Optional[torch.Tensor],
    ):
        """The group's inputs in rank order (the collective exception): ``(decode_rows, counts,
        mask)`` of rank ``rank``; ``mask`` is None without ``row_bounds``. The same for either
        ``failing_only``."""
        shares = _exact_shares(local_logits_by_rank, row_lens_by_rank, top_k)[rank]
        rows = len(shares)
        device = local_logits_by_rank[0].device
        out = torch.full((rows, top_k), -1, dtype=torch.int32, device=device)
        counts = torch.zeros(rows, dtype=torch.int32, device=device)
        for r, mine in enumerate(shares):
            out[r, : mine.numel()] = mine
            counts[r] = mine.numel()
        mask = None if row_bounds is None else (row_bounds[:rows] == 0) | (counts == 0)
        return out, counts, mask

    def is_valid(
        self,
        decode_rows: torch.Tensor,
        counts: Optional[torch.Tensor],
        row_bounds: Optional[torch.Tensor],
        mask_out: Optional[torch.Tensor],
        ws: Any,
        failing_only: bool = False,
    ) -> None:
        assert decode_rows.shape[0] <= ws.rows_max and decode_rows.shape[1] == ws.top_k
        # The mask needs the bounds, and a mask without them would be left unwritten.
        assert (mask_out is None) == (row_bounds is None), "mask_out goes with row_bounds"


rubin_dsa_key_shard_pack = RubinDsaKeyShardPack()
rubin_dsa_key_shard_merge = RubinDsaKeyShardMerge()


def rubin_dsa_key_shard_ws(
    mapping: Any, rows_max: int, top_k: int, local_width: int
) -> KeyShardWorkspace:
    """The merge workspace over ``mapping``'s TP group (under helix the repurposed mapping, whose
    TP group is the CP group). Collective: every rank of the group calls it at the same point,
    once, outside CUDA graph capture. Fabric handles across nodes, POSIX FD handles on one node, as
    the mapping says."""
    from tensorrt_llm._torch.distributed.ops import (
        _get_mnnvl_workspace_comm,
        _make_mnnvl_mcast_buffer,
        _mnnvl_workspace_all_succeeded,
    )

    local_k = key_shard_local_k()
    size = key_shard_bytes_needed(rows_max, top_k, mapping.tp_size, local_k, local_width)
    size = -(-size // _MNNVL_GRANULARITY) * _MNNVL_GRANULARITY
    comm = _get_mnnvl_workspace_comm(mapping)
    handle = _make_mnnvl_mcast_buffer(comm, size, mapping, mapping.is_multi_node())
    buffers = [handle.get_uc_buffer(r, (size,), torch.uint8, 0) for r in range(mapping.tp_size)]
    buffers[mapping.tp_rank].zero_()
    torch.cuda.synchronize()
    # Every rank has zeroed its flags before any peer publishes into the slots.
    if not _mnnvl_workspace_all_succeeded(comm, True):
        raise RuntimeError("key-sharded indexer workspace setup failed on a peer")
    return key_shard_workspace(
        buffers,
        mapping.tp_rank,
        rows_max,
        top_k,
        local_width,
        local_k,
        owner=(handle, comm),
        mapping=mapping,
    )
