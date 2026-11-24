# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The cep DSA layers' host plans (``rubin_cp_mla_rows``): row slices, their chunked-context
lengths, this rank's indexer query tiles, the cost-balanced split and the moves between splits,
on CPU."""

import itertools
import random

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_cp_mla_rows import (
    RowSlice,
    balanced_query_bounds,
    own_row_slices,
    own_row_tiles,
    row_move,
    slice_lengths,
)

pytestmark = pytest.mark.cpu_only

# Context requests' new tokens: a short one, one longer than a rank's rows, one row, a long tail.
_NEW = (5, 40, 1, 97)
_WORLDS = (8, 16)


def _split(total: int, world: int) -> list[tuple[int, int]]:
    rows = -(-total // world)
    return [(min(r * rows, total), min((r + 1) * rows, total)) for r in range(world)]


@pytest.mark.parametrize("world", _WORLDS, ids=lambda n: f"cep{n}")
def test_the_ranks_slices_cover_every_new_token_once(world):
    """Over the N ranks' rows, every new token of every request is in exactly one slice, in
    order, and each slice's first row is the token's row of the step."""
    total = sum(_NEW)
    seen = {r: [] for r in range(len(_NEW))}
    starts = [0, *itertools.accumulate(_NEW)]
    for begin, end in _split(total, world):
        slices = own_row_slices(_NEW, begin, end)
        assert sum(s.length for s in slices) == end - begin
        for s in slices:
            assert s.length > 0
            assert s.row == starts[s.req] + s.offset
            seen[s.req].extend(range(s.offset, s.offset + s.length))
    assert all(seen[r] == list(range(n)) for r, n in enumerate(_NEW))


def test_a_request_across_two_ranks_gives_each_one_slice():
    # Rows [0, 10) and [10, 20) of requests (5, 40): the second request straddles the boundary.
    assert own_row_slices(_NEW, 0, 10) == (RowSlice(0, 0, 5, 0), RowSlice(1, 0, 5, 5))
    assert own_row_slices(_NEW, 10, 20) == (RowSlice(1, 5, 10, 10),)


def test_a_rank_without_rows_holds_no_slice():
    total = sum(_NEW)
    assert own_row_slices(_NEW, total, total) == ()
    with pytest.raises(ValueError):
        own_row_slices(_NEW, 0, total + 1)


def test_a_slice_is_a_chunked_context_request():
    """Past KV: the request's cached tokens and its new tokens before the slice."""
    cached = (0, 128, 7, 1000)
    slices = own_row_slices(_NEW, 10, 50)
    lengths = slice_lengths(slices, cached)
    assert slices == (RowSlice(1, 5, 35, 10), RowSlice(2, 0, 1, 45), RowSlice(3, 0, 4, 46))
    assert lengths.q == (35, 1, 4)
    assert lengths.past == (133, 7, 1000)
    assert lengths.kv == (168, 8, 1004)


@pytest.mark.parametrize("keys", [1, 2048, 131072])
def test_the_query_tiles_cover_the_ranks_rows_of_a_chunk_under_the_budget(keys):
    budget = 536870912
    tiles = own_row_tiles(100, 9000, 0, 8192, keys, budget)
    rows = [r for g0, g1 in tiles for r in range(g0, g1)]
    assert rows == list(range(100, 8192))
    assert all((g1 - g0) * keys <= budget for g0, g1 in tiles)


def test_a_chunk_outside_the_ranks_rows_has_no_tile():
    assert own_row_tiles(0, 100, 100, 200, 4096, 1 << 20) == ()
    # A budget below one row of keys still scores one row a tile.
    assert own_row_tiles(0, 3, 0, 3, 1 << 30, 1) == ((0, 1), (1, 2), (2, 3))


def _cost(new, cached, lo, hi):
    rows = [c + i + 1 for n, c in zip(new, cached) for i in range(n)]
    return sum(rows[lo:hi])


def test_balanced_bounds_give_each_rank_an_equal_share_of_the_indexer_cost():
    """A long-history request's few rows weigh as much as many short-history rows."""
    new, cached = [100, 4000], [50000, 0]
    bounds = balanced_query_bounds(new, cached, 8)
    assert bounds == (0, 33, 65, 98, 1831, 2600, 3182, 3671, 4100)
    total = _cost(new, cached, 0, 4100)
    for lo, hi in zip(bounds, bounds[1:]):
        # Each block reaches its share with its last row (the cut is the first row at or past it).
        assert _cost(new, cached, lo, hi - 1) <= total / 8 + 50100


def test_balanced_bounds_of_one_fresh_request_follow_the_growing_history():
    bounds = balanced_query_bounds([1024], [0], 4)
    assert bounds[0] == 0 and bounds[-1] == 1024
    sizes = [hi - lo for lo, hi in zip(bounds, bounds[1:])]
    assert sizes == sorted(sizes, reverse=True)  # later rows score more keys


def test_balanced_bounds_of_an_empty_step():
    assert balanced_query_bounds([0, 0], [10, 20], 4) == (0, 0, 0, 0, 0)


def _bounds_by_row(new_tokens, cached, size, row_cost=0.0):
    """The row-by-row bounds (the reference the vectorized ``balanced_query_bounds`` must equal)."""
    count = sum(int(n) for n in new_tokens)
    if count == 0:
        return (0,) * (size + 1)
    cum, total, cuts = [], 0.0, []
    for n, c in zip(new_tokens, cached):
        for i in range(int(n)):
            total += int(c) + i + 1 + row_cost
            cum.append(total)
    targets = [total * r / size for r in range(1, size)]
    j = 0
    for target in targets:
        while j < count and cum[j] < target:
            j += 1
        cuts.append(min(j + 1, count))
    bounds = [0]
    for cut in cuts:
        bounds.append(max(bounds[-1], cut))
    return (*bounds, count)


def _random_steps():
    rng = random.Random(7)
    steps = [
        ([5461] * 12, [148_000] * 12),  # mid-curve: equal new tokens on long histories
        ([8192] * 16, [0, 200_000] * 8),  # alternating fresh and long histories
        ([57_000] + [4_000] * 18, [0] + [150_000] * 18),  # one fresh prompt among cached turns
        ([1, 0, 3], [10**9, 5, 0]),  # tiny, an empty request, a history past 2**30 keys
    ]
    for _ in range(40):
        k = rng.randint(1, 12)
        steps.append(
            (
                [rng.choice([0, 1, rng.randint(1, 9000)]) for _ in range(k)],
                [rng.choice([0, rng.randint(0, 250_000)]) for _ in range(k)],
            )
        )
    return steps


@pytest.mark.parametrize("size", [4, 8, 16])
@pytest.mark.parametrize("row_cost", [0.0, 0.37, 3.0])
def test_vectorized_bounds_equal_the_row_by_row_ones(size, row_cost):
    for new, cached in _random_steps():
        got = balanced_query_bounds(new, cached, size, row_cost)
        assert got == _bounds_by_row(new, cached, size, row_cost), (new, cached)


def _moved(src, dst, world):
    """Every rank's rows of ``dst`` after ``row_move`` from ``src``, the all-to-all simulated."""
    moves = [row_move(src, dst, r) for r in range(world)]
    # Rank r cuts its held rows by its send sizes, in rank order; rank j gets chunk j of each rank.
    chunks = []
    for r, m in enumerate(moves):
        held = list(range(src[r], src[r + 1]))
        assert m.held == len(held) and sum(m.send) == m.held
        cuts = list(itertools.accumulate(m.send, initial=0))
        chunks.append([held[a:b] for a, b in zip(cuts, cuts[1:])])
    out = []
    for r, m in enumerate(moves):
        got = [row for s in range(world) for row in chunks[s][r]]
        assert [len(chunks[s][r]) for s in range(world)] == list(m.recv) and len(got) == m.rows
        out.append(got)
    return out, moves


@pytest.mark.parametrize("world", [4, 8, 16])
def test_a_row_move_hands_each_rank_its_rows_of_the_new_split(world):
    for new, cached in _random_steps():
        count = sum(new)
        rows = -(-count // world)
        equal = tuple(min(r * rows, count) for r in range(world)) + (count,)
        balanced = balanced_query_bounds(new, cached, world)
        for src, dst in ((equal, balanced), (balanced, equal)):
            out, moves = _moved(src, dst, world)
            for r in range(world):
                assert out[r] == list(range(dst[r], dst[r + 1])), (src, dst, r)
            # The rows that change rank: those sent to another rank.
            changed = sum(n for r, m in enumerate(moves) for j, n in enumerate(m.send) if j != r)
            assert changed == sum(
                max(0, min(src[a + 1], dst[b + 1]) - max(src[a], dst[b]))
                for a in range(world)
                for b in range(world)
                if a != b
            )
    with pytest.raises(ValueError, match="different rows"):
        row_move((0, 5, 10), (0, 4, 9), 0)


def test_balanced_bounds_differ_from_the_equal_split_by_few_rows_mid_curve():
    """A mid-curve case: the balanced split is off the equal one by a few dozen rows per rank,
    so the move carries a few dozen rows, not the step."""
    new, cached = [5461] * 12, [148_000] * 12
    count = sum(new)
    rows = -(-count // 16)
    equal = tuple(min(r * rows, count) for r in range(16)) + (count,)
    balanced = balanced_query_bounds(new, cached, 16)
    assert balanced != equal
    moved = [
        sum(n for j, n in enumerate(row_move(equal, balanced, r).send) if j != r) for r in range(16)
    ]
    assert 0 < max(moved) < 64, moved
