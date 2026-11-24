# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host plans of a context-parallel (cep) DSA layer: a runtime helper, not an OpWrapper.

Under cep every rank appends every new token of the step to its latent and index-key caches (the
KV stays laid out as under TP), and scores and attends only its own query rows ``[begin, end)`` of
the step's context rows. Those rows cut the context requests into slices: a request whose new
tokens straddle two ranks' rows gives each of them one slice. A slice is a chunked-context request
of its own: the request's cached tokens and the new tokens before the slice are its past KV, the
slice's rows its new tokens. The sparse MLA of a rank's rows is the attention op's context call
over its slices, after every token was appended (the call appends nothing: no latent rows).

Everything here is host arithmetic (Python ints, numpy), once a step; the forward builds the
device tensors from it.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np


class RowSlice(NamedTuple):
    """The new tokens of context request ``req`` that one rank's rows hold.

    ``offset`` is the slice's first token among the request's new tokens, ``length`` its token
    count, ``row`` its first row in the step's context rows.
    """

    req: int
    offset: int
    length: int
    row: int


def own_row_slices(new_tokens: Sequence[int], begin: int, end: int) -> tuple[RowSlice, ...]:
    """The slices of the context requests (``new_tokens[r]`` new tokens each, in row order) that
    rows ``[begin, end)`` hold, in row order; no empty slice."""
    if not 0 <= begin <= end:
        raise ValueError(f"rows [{begin}, {end})")
    slices = []
    start = 0
    for req, count in enumerate(new_tokens):
        stop = start + int(count)
        lo, hi = max(start, begin), min(stop, end)
        if lo < hi:
            slices.append(RowSlice(req, lo - start, hi - lo, lo))
        start = stop
    if end > start:
        raise ValueError(f"rows [{begin}, {end}) past the {start} context rows")
    return tuple(slices)


class SliceLengths(NamedTuple):
    """Per slice: its new tokens, its past KV (cached + new tokens before it), its KV length."""

    q: tuple[int, ...]
    past: tuple[int, ...]
    kv: tuple[int, ...]


def slice_lengths(slices: Sequence[RowSlice], cached: Sequence[int]) -> SliceLengths:
    """The chunked-context lengths of each slice; ``cached[r]``: request ``r``'s cached tokens."""
    q = tuple(s.length for s in slices)
    past = tuple(int(cached[s.req]) + s.offset for s in slices)
    return SliceLengths(q, past, tuple(p + n for p, n in zip(past, q)))


def own_row_tiles(
    token_start: int, token_end: int, begin: int, end: int, keys: int, budget: int
) -> tuple[tuple[int, int], ...]:
    """The query tiles ``(g0, g1)`` (step rows) of rows ``[begin, end)`` within an indexer chunk of
    rows ``[token_start, token_end)`` that scores ``keys`` keys, each tile's logits at most
    ``budget`` elements (one row at least)."""
    lo, hi = max(token_start, begin), min(token_end, end)
    if lo >= hi:
        return ()
    step = max(1, budget // max(1, keys))
    return tuple((g, min(g + step, hi)) for g in range(lo, hi, step))


def balanced_query_bounds(
    new_tokens: Sequence[int], cached: Sequence[int], size: int, row_cost: float = 0.0
) -> tuple[int, ...]:
    """``size + 1`` row bounds that cut the step's context rows into ``size`` blocks of equal
    indexer cost (the indexer's ``balanced_query_bounds``, scope ``indexer``).

    Row ``i`` of a request with ``cached`` cached tokens scores ``cached + i + 1`` keys, plus
    ``row_cost``; the indexer's cost grows with the history, the sparse MLA's does not (a fixed
    Top-K), so only the indexer takes this split. A rank's block ends at the first row whose
    cumulative cost reaches its share (FP64 running sums in row order).
    """
    if len(new_tokens) != len(cached):
        raise ValueError(f"{len(new_tokens)} requests' new tokens, {len(cached)} cached counts")
    new = np.asarray([int(n) for n in new_tokens], dtype=np.int64)
    count = int(new.sum())
    if count == 0:
        return (0,) * (size + 1)
    first = np.cumsum(new) - new
    position = np.arange(count, dtype=np.int64) - np.repeat(first, new)
    keys = np.repeat(np.asarray([int(c) for c in cached], dtype=np.int64), new) + position + 1
    cum = np.cumsum(keys.astype(np.float64) + row_cost)
    total = float(cum[-1])
    targets = np.asarray([total * r / size for r in range(1, size)], dtype=np.float64)
    cuts = np.minimum(np.searchsorted(cum, targets, side="left") + 1, count)
    bounds = np.maximum.accumulate(np.concatenate([np.zeros(1, dtype=np.int64), cuts]))
    return (*(int(b) for b in bounds), count)


class RowMove(NamedTuple):
    """This rank's part of moving a step's rows from one contiguous split to another.

    Rank ``r`` holds rows ``[src[r], src[r + 1])`` of the step (as its first rows) and gets rows
    ``[dst[r], dst[r + 1])`` back, in row order. Both splits are contiguous and in rank order, so
    the rows for each rank are a contiguous run of the held rows, in rank order, and the rows from
    each rank a contiguous run of the rows gotten: one all-to-all with these split sizes moves them
    (the rows that stay here are this rank's own split size).
    """

    held: int  # rows this rank holds: src[r + 1] - src[r]
    rows: int  # rows this rank gets: dst[r + 1] - dst[r]
    send: tuple[int, ...]  # rows to each rank, in rank order
    recv: tuple[int, ...]  # rows from each rank, in rank order


def row_move_blocks(
    src: Sequence[int], dst: Sequence[int], rank: int
) -> tuple[tuple[tuple[int, int, int], ...], tuple[int, ...]]:
    """``rank``'s push of moving rows from split ``src`` to split ``dst``: per destination ``d``,
    ``(s0, d0, n)``, the held rows ``[s0, s0 + n)`` that land at row ``d0`` of ``d``'s rows of the
    destination split (rows arrive in row order); and every rank's rows of ``dst``."""
    world = len(src) - 1
    blocks = []
    for d in range(world):
        lo = max(src[rank], dst[d])
        n = max(0, min(src[rank + 1], dst[d + 1]) - lo)
        blocks.append((lo - src[rank] if n else 0, lo - dst[d] if n else 0, n))
    return tuple(blocks), tuple(dst[d + 1] - dst[d] for d in range(world))


def row_move(src: Sequence[int], dst: Sequence[int], rank: int) -> RowMove:
    """``rank``'s part of moving rows from split ``src`` to split ``dst`` (``world + 1`` row bounds
    each, over the same rows), as the indexer's ``_move_rows``."""
    world = len(src) - 1
    if len(dst) != world + 1 or src[0] != dst[0] or src[-1] != dst[-1]:
        raise ValueError(f"splits {tuple(src)} and {tuple(dst)} of different rows")

    def overlap(holder: int, getter: int) -> int:
        return max(0, min(src[holder + 1], dst[getter + 1]) - max(src[holder], dst[getter]))

    return RowMove(
        held=src[rank + 1] - src[rank],
        rows=dst[rank + 1] - dst[rank],
        send=tuple(overlap(rank, other) for other in range(world)),
        recv=tuple(overlap(other, rank) for other in range(world)),
    )
