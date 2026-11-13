# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Staging sizing and the slot queue alone, over plain layouts, the usable end resume_bounds finds and
how it and fetch ends keep out of runs of multimodal tokens, with the runtime's own life cycles
behind a stand-in manager, the delivered rows a shrink hook drops when its own work raises, the
slots a read's grant or a write's marks keep when a copy's completion cannot be recorded, also when
the quarantine itself raises, a write whose grant and failure both raise, a lender stopped by an
error its free or shutdown hook logs, the records of a replaced cache resume_bounds drops, the states
a fetch walks through, and the bytes the test kit stages: no real manager, no memory, no GPU."""

import ast
import gc
import inspect
import itertools
import random
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import (
    StagingOptions,
    _fetch_ledger,
    _lender,
    _ranges,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing._slots import (
    SlotQueue,
    SlotRuns,
    slot_counts,
    slots_per_fetch,
)
from tensorrt_llm.runtime.kv_cache_manager_v2 import AttnLifeCycle

pytestmark = pytest.mark.cpu_only

MiB = 1 << 20
GiB = 1 << 30


def layout(tokens_per_block, groups):
    """What the sizing reads of a manager's layout, for layer groups ``(window, sink blocks, pool
    group, page bytes)``."""
    return SimpleNamespace(
        tokens_per_block=tokens_per_block,
        windows=tuple(window for window, _, _, _ in groups),
        sink_blocks=tuple(sinks for _, sinks, _, _ in groups),
        pool_group_of=tuple(g for _, _, g, _ in groups),
        page_bytes={g: page for _, _, g, page in groups},
    )


def one_fetch(lay, tokens_per_fetch):
    return sum(
        rows * lay.page_bytes[g] for g, rows in slots_per_fetch(lay, tokens_per_fetch).items()
    )


# TinyLlama-1.1B, TP=1, 32 tokens per block: 22 layers of K and V, 4 heads of 64 bf16 each.
TINYLLAMA_PAGE = 22 * 2 * 4 * 64 * 2 * 32
TINYLLAMA = layout(32, [(None, 0, 0, TINYLLAMA_PAGE)])

# DeepSeek-V4-Pro as its cache manager lays it out on every rank, 128 tokens per block: a
# 128-token window, compressed full attention, and the compressors' state with an 8-token window.
DSV4_PAGES = {1: 20250624, 0: 572672, 2: 39321600}
DSV4_PRO = layout(
    128, [(128, 0, 1, DSV4_PAGES[1]), (None, 0, 0, DSV4_PAGES[0]), (8, 0, 2, DSV4_PAGES[2])]
)


# -- sizing -----------------------------------------------------------------------------------


def test_tinyllama_stages_eight_whole_prompts():
    assert slots_per_fetch(TINYLLAMA, 2048) == {0: 64}
    assert one_fetch(TINYLLAMA, 2048) == 44 * MiB
    assert slot_counts(TINYLLAMA, StagingOptions(2048, planned_fetches=8)) == {0: 8 * 64}


def test_deepseek_v4_windows_take_their_window_and_the_full_group_the_range():
    assert slots_per_fetch(DSV4_PRO, 4096) == {1: 1, 0: 32, 2: 1}
    assert 74 * MiB < one_fetch(DSV4_PRO, 4096) < 75 * MiB
    assert slot_counts(DSV4_PRO, StagingOptions(4096, planned_fetches=8)) == {1: 8, 0: 256, 2: 8}


def test_deepseek_v4_long_ranges_capped_at_one_fetch_keep_one_fetch():
    one = one_fetch(DSV4_PRO, 1 << 20)
    assert slots_per_fetch(DSV4_PRO, 1 << 20) == {1: 1, 0: 8192, 2: 1}
    assert 4 * GiB < one < 5 * GiB
    options = StagingOptions(1 << 20, planned_fetches=8, max_bytes=one)
    assert slot_counts(DSV4_PRO, options) == {1: 1, 0: 8192, 2: 1}
    # A cap between one and eight fetches: every group keeps its rows of one fetch and more.
    cap = 960 * GiB // 8 // 4
    counts = slot_counts(DSV4_PRO, StagingOptions(1 << 20, planned_fetches=8, max_bytes=cap))
    assert counts[0] >= 8192 and counts[1] >= 1 and counts[2] >= 1
    assert sum(counts[g] * DSV4_PAGES[g] for g in counts) <= cap


def test_short_ranges_take_no_more_than_their_blocks_even_in_a_window():
    assert slots_per_fetch(DSV4_PRO, 100) == {1: 1, 0: 1, 2: 1}
    assert slot_counts(DSV4_PRO, StagingOptions(100)) == {1: 1, 0: 1, 2: 1}


def test_a_window_wider_than_the_range_takes_the_range_blocks():
    lay = layout(32, [(1024, 0, 0, 100)])  # a window of 32 blocks over a range of 8
    assert slots_per_fetch(lay, 256) == {0: 8}
    assert slot_counts(lay, StagingOptions(256, planned_fetches=2)) == {0: 16}


def test_sink_blocks_count_with_the_window():
    lay = layout(32, [(64, 1, 0, 100), (None, 0, 1, 10)])
    assert slots_per_fetch(lay, 320) == {0: 1 + 2, 1: 10}
    assert one_fetch(lay, 320) == 3 * 100 + 10 * 10
    assert slot_counts(lay, StagingOptions(320, planned_fetches=3)) == {0: 9, 1: 30}


def test_layer_groups_sharing_a_pool_group_add_up():
    lay = layout(16, [(None, 0, 0, 8), (32, 0, 0, 8)])
    assert slots_per_fetch(lay, 160) == {0: 10 + 2}


def test_a_partial_block_counts_as_a_whole_one():
    assert slots_per_fetch(TINYLLAMA, 33) == {0: 2}
    assert slots_per_fetch(TINYLLAMA, 1) == {0: 1}


@pytest.mark.parametrize("tokens_per_fetch", [0, -32])
def test_tokens_per_fetch_must_be_positive(tokens_per_fetch):
    with pytest.raises(ValueError):
        slots_per_fetch(TINYLLAMA, tokens_per_fetch)


def test_max_bytes_below_one_fetch_is_rejected():
    one = one_fetch(DSV4_PRO, 4096)
    with pytest.raises(ValueError, match="one fetch"):
        slot_counts(DSV4_PRO, StagingOptions(4096, max_bytes=one - 1))
    assert slot_counts(DSV4_PRO, StagingOptions(4096, max_bytes=one)) == slots_per_fetch(
        DSV4_PRO, 4096
    )


def test_max_bytes_above_planned_fetches_changes_nothing():
    options = StagingOptions(4096, planned_fetches=2, max_bytes=100 * one_fetch(DSV4_PRO, 4096))
    assert slot_counts(DSV4_PRO, options) == {1: 2, 0: 64, 2: 2}


def random_layout(rng):
    """A layout with up to four layer groups over up to three pool groups, and its life cycles."""
    tpb = rng.choice([4, 8, 16, 32])
    pages = {g: rng.randint(1, 1000) for g in range(rng.randint(1, 3))}
    groups, life_cycles = [], []
    for _ in range(rng.randint(1, 4)):
        window = rng.choice([None, rng.randint(1, 6 * tpb)])
        sink_tokens = 0 if window is None else rng.choice([0, 0, rng.randint(1, 3 * tpb)])
        life_cycle = AttnLifeCycle.make(window, sink_tokens, tpb)
        g = rng.choice(sorted(pages))
        groups.append((window, int(life_cycle.num_sink_blocks), g, pages[g]))
        life_cycles.append(life_cycle)
    return layout(tpb, groups), life_cycles


@pytest.mark.parametrize("seed", range(40))
def test_any_whole_block_range_of_at_most_tokens_per_fetch_fits_one_fetch(seed):
    # Rows a range [start, end) needs: its blocks that a history of end still reads, as the
    # runtime's own life cycle says; the longest range ending at end needs the most.
    rng = random.Random(seed)
    lay, life_cycles = random_layout(rng)
    tpb = lay.tokens_per_block
    tokens_per_fetch = rng.randint(1, 12 * tpb)
    bound = slots_per_fetch(lay, tokens_per_fetch)
    span = tokens_per_fetch // tpb
    for end_block in range(1, 40):
        end = end_block * tpb
        need = dict.fromkeys(bound, 0)
        for lg, life_cycle in enumerate(life_cycles):
            stale = life_cycle.get_stale_range(end, tpb)
            blocks = range(max(0, end_block - span), end_block)
            need[lay.pool_group_of[lg]] += sum(not stale.beg <= b < stale.end for b in blocks)
        assert all(need[g] <= bound[g] for g in bound), (end, need, bound)


@pytest.mark.parametrize("seed", range(40))
def test_counts_are_planned_fetches_fetches_or_the_cap_rounded_down(seed):
    rng = random.Random(seed)
    lay, _ = random_layout(rng)
    tokens_per_fetch = rng.randint(1, 12 * lay.tokens_per_block)
    rows = slots_per_fetch(lay, tokens_per_fetch)
    one = one_fetch(lay, tokens_per_fetch)
    planned_fetches = rng.randint(1, 9)
    assert slot_counts(lay, StagingOptions(tokens_per_fetch, planned_fetches)) == {
        g: planned_fetches * rows[g] for g in rows
    }
    cap = rng.randint(one, planned_fetches * one)
    counts = slot_counts(lay, StagingOptions(tokens_per_fetch, planned_fetches, max_bytes=cap))
    assert sum(counts[g] * lay.page_bytes[g] for g in counts) <= cap
    # As many whole fetches as the cap holds hold slots at once.
    whole = cap // one
    assert all(whole * rows[g] <= counts[g] <= planned_fetches * rows[g] for g in rows)
    assert all(counts[g] * one <= cap * rows[g] < (counts[g] + 1) * one for g in rows)


def test_a_fractional_fetch_budget_keeps_slots_for_a_shorter_range() -> None:
    lay = layout(32, [(None, 0, 0, 8)])
    slots = SlotQueue(slot_counts(lay, StagingOptions(320, planned_fetches=3, max_bytes=200)))
    assert slots.num_slots(0) == 25
    for rows in (10, 10, 5):
        assert slots.try_grant(slots.enqueue({0: rows})) is not None
    assert slots.try_grant(slots.enqueue({0: 1})) is None


@pytest.mark.parametrize("max_bytes_in_fetches", [None, 1, 2.5])
def test_planned_fetches_ranges_hold_slots_at_once_and_one_more_waits(max_bytes_in_fetches):
    rows = slots_per_fetch(DSV4_PRO, 4096)
    max_bytes = None
    at_once = 3
    if max_bytes_in_fetches is not None:
        max_bytes = int(max_bytes_in_fetches * one_fetch(DSV4_PRO, 4096))
        at_once = int(max_bytes_in_fetches)
    slots = SlotQueue(
        slot_counts(DSV4_PRO, StagingOptions(4096, planned_fetches=3, max_bytes=max_bytes))
    )
    for _ in range(at_once):
        assert slots.try_grant(slots.enqueue(rows)) is not None
    assert slots.try_grant(slots.enqueue(rows)) is None
    assert slots.num_waiting == 1


# -- the slot queue ---------------------------------------------------------------------------


def take_now(slots, counts):
    ticket = slots.enqueue(counts)
    runs = slots.try_grant(ticket)
    assert runs is not None, f"{counts} did not fit"
    return runs


def test_slots_of_one_lease_are_contiguous_per_group():
    slots = SlotQueue({0: 8, 1: 8})
    assert (slots.num_slots(0), slots.free_slots(1)) == (8, 8)
    runs = take_now(slots, {0: 3, 1: 5})
    assert runs.slots(0).tolist() == [0, 1, 2]
    assert runs.slots(1).tolist() == [0, 1, 2, 3, 4]
    assert runs.slots(7).tolist() == []
    assert runs.slots(0).dtype.name == "int64"


def test_a_lease_is_all_or_nothing_across_groups():
    slots = SlotQueue({0: 8, 1: 4})
    held = take_now(slots, {1: 3})
    ticket = slots.enqueue({0: 2, 1: 2})

    assert slots.try_grant(ticket) is None
    # Group 0 had room, but nothing was taken from it.
    assert slots.free_slots(0) == 8
    assert slots.free_slots(1) == 1

    slots.give_back(held)
    assert slots.try_grant(ticket) is not None
    assert (slots.free_slots(0), slots.free_slots(1)) == (6, 2)


def test_contiguity_waits_for_a_run_even_when_enough_slots_are_free():
    slots = SlotQueue({0: 6})
    a, b, c = (take_now(slots, {0: 2}) for _ in range(3))
    slots.give_back(a)
    slots.give_back(c)
    assert slots.free_slots(0) == 4

    ticket = slots.enqueue({0: 3})
    assert slots.try_grant(ticket) is None, "two free runs of 2 are not a run of 3"

    slots.give_back(b)
    assert slots.try_grant(ticket).slots(0).tolist() == [0, 1, 2]


def test_waiting_leases_are_served_strictly_in_order():
    slots = SlotQueue({0: 4})
    held = take_now(slots, {0: 3})
    first = slots.enqueue({0: 2})
    second = slots.enqueue({0: 1})

    # One slot is free and ``second`` would fit, but ``first`` asked earlier.
    assert slots.try_grant(second) is None
    assert slots.try_grant(first) is None
    assert slots.num_waiting == 2

    slots.give_back(held)
    assert slots.try_grant(second) is None, "still behind the first"
    assert slots.try_grant(first).slots(0).tolist() == [0, 1]
    assert slots.try_grant(second).slots(0).tolist() == [2]
    assert slots.num_waiting == 0


def test_cancelling_the_head_lets_the_next_lease_through():
    slots = SlotQueue({0: 4})
    held = take_now(slots, {0: 3})
    first = slots.enqueue({0: 4})
    second = slots.enqueue({0: 1})
    assert slots.try_grant(second) is None

    slots.cancel(first)
    assert slots.try_grant(second).slots(0).tolist() == [3]
    slots.give_back(held)
    # Cancelling an unknown or granted ticket does nothing.
    slots.cancel(first)
    slots.cancel(12345)
    assert (slots.free_slots(0), slots.num_waiting) == (3, 0)


def test_a_ticket_that_is_not_waiting_is_a_key_error():
    slots = SlotQueue({0: 8})
    with pytest.raises(KeyError):
        slots.try_grant(12345)
    ticket = slots.enqueue({0: 1})
    assert slots.try_grant(ticket) is not None
    with pytest.raises(KeyError):
        slots.try_grant(ticket)


def test_frees_coalesce_into_one_run():
    slots = SlotQueue({0: 8})
    held = [take_now(slots, {0: 2}) for _ in range(4)]
    for runs in (held[1], held[3], held[0], held[2]):
        slots.give_back(runs)
    assert slots.free_slots(0) == 8
    # Only a single coalesced run can serve the whole group at once.
    assert take_now(slots, {0: 8}).slots(0).tolist() == list(range(8))


def test_a_double_free_is_rejected_and_changes_nothing():
    slots = SlotQueue({0: 4, 1: 4})
    runs = take_now(slots, {0: 2, 1: 2})
    slots.give_back(SlotRuns({1: (0, 2)}))
    assert (slots.free_slots(0), slots.free_slots(1)) == (2, 4)

    with pytest.raises(ValueError, match="freed twice"):
        slots.give_back(runs)
    # Group 0's run was not returned on the way to finding group 1's double free.
    assert (slots.free_slots(0), slots.free_slots(1)) == (2, 4)


@pytest.mark.parametrize("run", [(3, 2), (-1, 1)], ids=["past_the_end", "negative"])
def test_freeing_slots_outside_the_group_is_rejected(run):
    slots = SlotQueue({0: 4})
    take_now(slots, {0: 4})
    with pytest.raises(ValueError, match="outside"):
        slots.give_back(SlotRuns({0: run}))
    assert slots.free_slots(0) == 0


def test_freeing_slots_of_an_unknown_group_is_rejected():
    slots = SlotQueue({0: 4})
    take_now(slots, {0: 4})
    with pytest.raises(ValueError, match="pool group 5"):
        slots.give_back(SlotRuns({0: (0, 4), 5: (0, 1)}))
    assert slots.free_slots(0) == 0


@pytest.mark.parametrize(
    "counts", [{0: 5}, {3: 1}, {0: -1}], ids=["too_many", "unknown", "negative"]
)
def test_a_request_that_can_never_be_granted_is_rejected_at_once(counts):
    slots = SlotQueue({0: 4})
    with pytest.raises(ValueError):
        slots.check_fits_capacity(counts)
    with pytest.raises(ValueError):
        slots.enqueue(counts)
    assert slots.num_waiting == 0
    assert slots.free_slots(0) == 4


def test_check_queues_nothing():
    slots = SlotQueue({0: 4})
    slots.check_fits_capacity({0: 4})
    assert slots.num_waiting == 0
    # A later ask is first in line.
    assert take_now(slots, {0: 4}).slots(0).tolist() == [0, 1, 2, 3]


def test_zero_counts_need_no_slots():
    slots = SlotQueue({0: 1})
    take_now(slots, {0: 1})
    assert take_now(slots, {0: 0, 9: 0}).by_pool_group == {}


class _Owner:
    """What a stand-in lender's weak references point at."""


_OWNER = _Owner()


def staging_over(windows, sinks, tpb):
    """A staging lender and a manager stand-in answering what resume_bounds asks: the tokens per block
    and each layer group's stale range, from the runtime's own life cycle."""
    cycles = [AttnLifeCycle.make(w, s * tpb, tpb) for w, s in zip(windows, sinks)]

    def stale_block_range(lg, history):
        stale = cycles[lg].get_stale_range(history, tpb)
        return stale.beg, stale.end

    manager = SimpleNamespace(tokens_per_block=tpb, _stale_block_range=stale_block_range)
    lay = SimpleNamespace(
        tokens_per_block=tpb, windows=tuple(windows), pool_groups=(), num_layer_groups=len(windows)
    )
    ref = weakref.ref(_OWNER)
    return _lender.Staging(ref, lay, None, (), None, ref), manager


def reads_all(manager, windows, blocks, first, end_block, tpb):
    """Every layer group has, among the blocks from ``first`` and the delivered ones, every block a
    start at ``end_block * tpb`` reads: all of them for full attention, else all but the stale."""
    for lg, delivered in enumerate(blocks):
        beg, end = (0, 0)
        if windows[lg] is not None:
            beg, end = manager._stale_block_range(lg, end_block * tpb)
        for ordinal in range(first, end_block):
            if not beg <= ordinal < end and not (ordinal < len(delivered) and delivered[ordinal]):
                return False
    return True


def test_max_resume_position_is_the_largest_start_every_layer_group_reads():
    """Random layouts, delivered rows and computed prefixes: the usable end is the largest
    whole-block start past the computed prefix where every layer group has what it reads, whatever
    the lowest fetch start, else the computed prefix."""
    rng = random.Random(0)
    for _ in range(3000):
        tpb = rng.choice([16, 32])
        windows = [None if rng.random() < 0.25 else rng.randint(1, 200) for _ in range(3)]
        windows = windows[: rng.randint(1, 3)]
        sinks = [0 if w is None else rng.choice([0, 0, 1, 2]) for w in windows]
        known = rng.choice([0, rng.randint(0, 12) * tpb, rng.randint(0, 12 * tpb)])
        origin = rng.choice([known // tpb * tpb, rng.randint(0, 14) * tpb])
        length = rng.randint(0, 20)
        blocks = []
        for _ in windows:
            row = np.zeros(max(0, length + rng.choice([0, -1, 1])), dtype=bool)
            row[rng.choice([0, origin // tpb, rng.randint(0, 20)]) :] = True
            for _ in range(rng.choice([0, 0, 1, 2])):
                if len(row):
                    row[rng.randrange(len(row))] = False
            row[: origin // tpb] = False  # fetches start at origin or later
            blocks.append(row)
        lender, manager = staging_over(windows, sinks, tpb)
        first = known // tpb
        last = max((len(b) for b in blocks), default=0)
        ends = range(first + 1, last + 1)
        starts = [e for e in ends if reads_all(manager, windows, blocks, first, e, tpb)]
        want = starts[-1] * tpb if starts else known
        delivered = _fetch_ledger._Deliveries(None, origin, [b.copy() for b in blocks])
        got = lender._ledger._max_resume_position(manager, delivered, known)
        assert got == want, (
            f"windows {windows}, sinks {sinks}, tpb {tpb}, origin {origin}, known {known}, "
            f"delivered {[np.nonzero(b)[0].tolist() for b in blocks]}: usable until {got}, "
            f"not {want}"
        )


def test_max_resume_position_after_a_whole_fetch_checks_its_end_alone(monkeypatch):
    """Two windowed layer groups, nothing computed and 4096 blocks delivered in each: the end of the
    delivered rows is usable, found with one stale-range lookup per layer group, not one per block
    below it, so resume_bounds after each lease of a split fetch costs no more as the fetch grows."""
    lender, manager = staging_over([4096, 8192], [0, 0], 32)
    blocks = [np.ones(4096, dtype=bool), np.ones(4096, dtype=bool)]
    real_stale, calls = _ranges._stale_range, []

    def counted(*args):
        calls.append(args)
        return real_stale(*args)

    monkeypatch.setattr(_ranges, "_stale_range", counted)
    got = lender._ledger._max_resume_position(
        manager, _fetch_ledger._Deliveries(None, 0, blocks), 0
    )
    assert got == 4096 * 32, f"usable until {got}"
    assert len(calls) == 2, f"{len(calls)} stale-range lookups for one usable end"


def test_max_resume_position_checks_no_start_past_a_missing_full_attention_block(monkeypatch):
    """One full-attention layer group, nothing computed and 8192 blocks delivered but one: no start
    past the missing block has every block it reads, so none is checked, and resume_bounds after each
    later lease of a split fetch that lost an early block checks no more starts as it grows."""
    lender, manager = staging_over([None], [0], 32)
    real_stale, calls = _ranges._stale_range, []

    def counted(*args):
        calls.append(args)
        return real_stale(*args)

    monkeypatch.setattr(_ranges, "_stale_range", counted)
    for hole, usable, checked in ((0, 0, 0), (4000, 4000 * 32, 1)):
        blocks = np.ones(8192, dtype=bool)
        blocks[hole] = False
        calls.clear()
        got = lender._ledger._max_resume_position(
            manager, _fetch_ledger._Deliveries(None, 0, [blocks]), 0
        )
        assert got == usable, f"usable until {got} with block {hole} missing"
        assert len(calls) == checked, f"{len(calls)} starts checked with block {hole} missing"


def test_a_shrink_hook_that_raises_forgets_what_the_request_was_delivered(monkeypatch):
    """A shrink hook whose own work raises, which it logs: the request's delivered rows all go,
    since the cache may grow back before resume_bounds sees the freed ones past its blocks."""
    lender, _ = staging_over([None], [0], 32)
    kv = object()
    lender._ledger._deliveries[7] = _fetch_ledger._Deliveries(kv, 0, [np.ones(3, dtype=bool)])
    lender._ledger._mixed_tail_caches[7] = kv
    raised = []

    def num_blocks(cache):
        raised.append(cache)
        raise MemoryError("planted")

    monkeypatch.setattr(_lender._manager, "num_blocks", num_blocks)
    lender._on_shrink(7, kv)
    assert raised == [kv], "the hook's work did not raise: the check proves nothing"
    assert 7 not in lender._ledger._deliveries, "the rows the shrink freed stay delivered"
    assert lender._ledger._mixed_tail_caches.get(7) is kv, "the hook dropped the request's taint"


# -- runs of multimodal tokens ----------------------------------------------------------------


def multimodal(runs, length, flagged=True):
    """A request stand-in whose multimodal data marks ``runs`` as tokens that attend both ways."""
    mask = np.zeros(length, dtype=np.int64)
    for beg, end in runs:
        mask[beg:end] = 1
    data = {"mm_bidirectional_blocks": flagged, "multimodal_embed_mask_cumsum": np.cumsum(mask)}
    return SimpleNamespace(py_multimodal_data=data)


def strictly_inside(runs, position):
    return any(beg < position < end for beg, end in runs)


def test_resume_bounds_and_fetch_ends_keep_out_of_bidirectional_runs():
    """Random bidirectional runs of multimodal tokens: a fetch end strictly inside a run is refused,
    others not; resume_bounds lowers its end to the largest position at or below it not strictly inside
    one, with a floor at the end of the last run at or below that, so none from floor to end is."""
    rng = random.Random(0)
    # With a window too, which the refusal must not read: the model's window can be narrower.
    lender, windowed = (staging_over(w, [0] * len(w), 32)[0] for w in ([None], [64, None]))
    for _ in range(3000):
        length = rng.randint(1, 400)
        runs, pos = [], rng.randint(0, 20)
        while pos < length - 1 and rng.random() < 0.7:
            stop = min(length, pos + rng.randint(1, 120))
            runs.append((pos, stop))
            pos = stop + rng.randint(1, 40)
        request = multimodal(runs, length)
        usable = rng.randint(0, length)
        got, floor = lender._ledger._clip_to_bidirectional_runs(request, usable)
        want = max(p for p in range(usable + 1) if not strictly_inside(runs, p))
        case = f"runs {runs}, usable {usable}"
        assert got == want, f"{case}: the end lowered to {got}, not {want}"
        assert floor == max((stop for _, stop in runs if stop <= want), default=0), (
            f"{case}: a floor of {floor}"
        )
        assert not any(strictly_inside(runs, p) for p in range(floor, got + 1)), (
            f"{case}: [{floor}, {got}] holds a position strictly inside a run"
        )
        end = rng.randint(0, length)
        for each in (lender, windowed):
            refused = each._bidirectional_split_refusal(request, end) is not None
            assert refused == strictly_inside(runs, end), (
                f"{case}: an end at {end} refused: {refused}"
            )


def test_a_request_without_bidirectional_runs_keeps_its_interval_and_ends():
    """No flag, no multimodal data, or no mask: nothing lowers, rises or fails."""
    lender, _ = staging_over([64, None], [0, 0], 32)
    for request in (
        multimodal([(1, 300)], 400, flagged=False),
        SimpleNamespace(py_multimodal_data=None),
        SimpleNamespace(py_multimodal_data={"mm_bidirectional_blocks": True}),
    ):
        assert lender._ledger._clip_to_bidirectional_runs(request, 200) == (200, 0)
        assert lender._bidirectional_split_refusal(request, 200) is None


# -- copies whose completion cannot be recorded ----------------------------------------------


def raising_once(call, error, fired):
    """``call``, except that its first call raises ``error`` and notes it in ``fired``."""

    def once(*args, **kwargs):
        if not fired:
            fired.append(True)
            raise error
        return call(*args, **kwargs)

    return once


class _AppendRaisingOnce(list):
    """A record list whose first append raises ``MemoryError`` and notes it in ``fired``."""

    def __init__(self, fired):
        super().__init__()
        self.fired = fired

    def append(self, item):
        if not self.fired:
            self.fired.append(True)
            raise MemoryError("planted")
        super().append(item)


def _stand_in_view(rows, keys):
    return "view"


@pytest.mark.parametrize("fault", ["none", "before", "record", "after"])
def test_a_read_grant_keeps_the_slots_its_queued_copy_may_write_out_of_use(monkeypatch, fault):
    """A read granted 2 of 4 slots queues into them a copy whose completion cannot be recorded: the
    read fails and its slots stay out of use, also with a second fault in the quarantine's record or
    the read's failure after it. A fault in its view, before it takes slots, returns them."""
    lay = SimpleNamespace(
        pool_groups=(0,),
        pool_group_of=(0,),
        windows=(None,),
        num_layer_groups=1,
        tokens_per_block=1,
    )
    ref = weakref.ref(_OWNER)
    lender = _lender.Staging(ref, lay, None, (), SlotQueue({0: 4}), ref)
    fired = []
    view = _stand_in_view
    if fault == "before":
        view = raising_once(view, MemoryError("planted"), fired)
    monkeypatch.setattr(lender, "_build_view", view)
    monkeypatch.setattr(lender, "_copy_segments", lambda rows: [])
    monkeypatch.setattr(
        lender, "_queue_copy", lambda segments, to_staging: (None, "planted: no event")
    )
    if fault == "record":
        lender._quarantined = _AppendRaisingOnce(fired)
    elif fault == "after":
        failing = raising_once(lender._fail, MemoryError("planted"), fired)
        monkeypatch.setattr(lender, "_fail", failing)
    rows = _lender._LentBlockEntries([0], [np.arange(2)], [np.arange(2)])
    read = _lender._StagingLease(lender, "read", 7, None, rows, [np.zeros((2, 32), np.uint8)])
    lender._admit(read, {0: 2})
    assert bool(fired) == (fault != "none"), "the planted fault did not strike: proves nothing"
    assert read.failure is not None and "planted" in read.failure, read.failure
    if fault == "before":
        assert lender._free_slots(0) == 4, "slots no copy reached stayed out of use"
        assert not lender._quarantined, "slots no copy reached were recorded as lost"
        return
    assert lender._free_slots(0) == 2, "slots a queued copy may still write came back"
    assert any(r is read._slot_runs for r in lender._quarantined), "the lost slots are not recorded"


class _IterRaisingOnce(list):
    """A holding list whose first iteration raises ``MemoryError`` and notes it in ``fired``."""

    def __init__(self, fired):
        super().__init__()
        self.fired = fired

    def __iter__(self):
        if not self.fired:
            self.fired.append(True)
            raise MemoryError("planted")
        return super().__iter__()


def _unrecorded_copy(segments, to_staging):
    raise MemoryError("planted: the copy is queued, its record is not")


def _one_pool_group():
    return SimpleNamespace(
        pool_groups=(0,),
        pool_group_of=(0,),
        windows=(None,),
        num_layer_groups=1,
        tokens_per_block=1,
    )


def _write_target():
    """A stand-in manager whose request 7 holds blocks 0 and 1 on pages 0 and 1."""
    kv = SimpleNamespace(
        num_committed_tokens=0,
        history_length=0,
        is_active=True,
        num_blocks=2,
        get_base_page_indices=lambda lg: [0, 1],
    )
    manager = _Owner()
    manager.kv_cache_map = {7: kv}
    return kv, manager


@pytest.mark.parametrize("fault", ["none", "call", "record", "holding", "log", "append", "partial"])
def test_a_read_whose_grant_and_quarantine_both_raise_is_never_ready(monkeypatch, fault):
    """A read granted 2 of 4 slots: its grant raises once its view is set (recording its copy,
    entering the holding list or failing a partly failed copy), then its quarantine raises too (at
    its call, the slots' record, the holding list or its log line). As after a manager hook logged
    the error, it fails at its next poll, returning no view, its slots stay out of use after
    release, and where a copy may be queued the shutdown would keep the staging memory. With no
    fault it is ready once its copy landed."""
    ref = weakref.ref(_OWNER)
    lender = _lender.Staging(ref, _one_pool_group(), None, (), SlotQueue({0: 4}), ref)
    planted = [[], []]  # the grant's fault, then the quarantine's
    landed = SimpleNamespace(query=lambda: True)
    monkeypatch.setattr(lender, "_build_view", _stand_in_view)
    monkeypatch.setattr(lender, "_copy_segments", lambda rows: [])
    if fault == "none":
        copied = (_lender._QueuedCopy(landed), None)
        monkeypatch.setattr(lender, "_queue_copy", lambda segments, to_staging: copied)
    elif fault == "partial":
        copied = (_lender._QueuedCopy(landed), "planted: a copy call failed")
        monkeypatch.setattr(lender, "_queue_copy", lambda segments, to_staging: copied)
        failing = raising_once(lender._fail, MemoryError("planted"), planted[0])
        monkeypatch.setattr(lender, "_fail", failing)
    else:
        monkeypatch.setattr(lender, "_queue_copy", _unrecorded_copy)
        planted[0].append(True)
    if fault == "append":
        planted[0].clear()
        lender._leases_holding_slots = _AppendRaisingOnce(planted[0])
    if fault in ("call", "append", "partial"):
        failing = raising_once(lender._quarantine, MemoryError("planted"), planted[1])
        monkeypatch.setattr(lender, "_quarantine", failing)
    elif fault == "record":
        lender._quarantined = _AppendRaisingOnce(planted[1])
    elif fault == "holding":
        lender._leases_holding_slots = _IterRaisingOnce(planted[1])
    elif fault == "log":
        failing = raising_once(_lender.logger.warning, MemoryError("planted"), planted[1])
        monkeypatch.setattr(_lender.logger, "warning", failing)
    rows = _lender._LentBlockEntries([0], [np.arange(2)], [np.arange(2)])
    read = _lender._StagingLease(lender, "read", 7, None, rows, [np.zeros((2, 32), np.uint8)])
    if fault == "none":
        lender._admit(read, {0: 2})
    else:
        with pytest.raises(MemoryError, match="planted"):  # what a manager hook logs and goes on
            lender._admit(read, {0: 2})
        assert all(planted), "a planted fault did not strike: proves nothing"
    view = read.poll()
    read.release()
    if fault == "none":
        assert view == "view" and read.failure is None, "a read whose copy landed was not ready"
        assert lender._free_slots(0) == 4, "the slots of a read whose copy landed stayed out of use"
        assert not lender._must_retain_memory(), "nothing open, yet the staging memory would stay"
        return
    assert view is None, "a read with no record of its copy was ready"
    assert read.failure is not None, "a read with no record of its copy did not fail"
    assert lender._free_slots(0) == 2, "slots a queued copy may still write came back"
    if fault != "append":  # past the append, a copy may be queued
        assert lender._must_retain_memory(), (
            "the shutdown would free staging a queued copy may write"
        )


@pytest.mark.parametrize("fault", ["call", "record"])
@pytest.mark.parametrize("copy", ["raised", "unrecorded"])
def test_marks_whose_copy_and_quarantine_both_fail_keep_the_slots_out_of_use(
    monkeypatch, copy, fault
):
    """A write granted 2 of 4 slots is marked; its marks' copy raises once queued or returns with no
    record of its completion, then its quarantine raises too, at its call or the slots' record:
    after the release the slots stay out of use and the shutdown would keep the staging memory."""
    kv, manager = _write_target()
    ref = weakref.ref(manager)
    lender = _lender.Staging(ref, _one_pool_group(), None, (), SlotQueue({0: 4}), ref)
    fired = []
    monkeypatch.setattr(
        lender, "_build_view", lambda rows, keys: SimpleNamespace(groups=rows.block_ordinals)
    )
    monkeypatch.setattr(lender, "_copy_segments", lambda rows, mask: [])
    if copy == "raised":
        monkeypatch.setattr(lender, "_queue_copy", _unrecorded_copy)
    else:
        monkeypatch.setattr(lender, "_queue_copy", lambda segments, to_staging: (None, "no event"))
    if fault == "call":
        failing = raising_once(lender._quarantine, MemoryError("planted"), fired)
        monkeypatch.setattr(lender, "_quarantine", failing)
    else:
        lender._quarantined = _AppendRaisingOnce(fired)
    rows = _lender._LentBlockEntries([0], [np.arange(2)], [np.arange(2)])
    write = _lender._StagingLease(lender, "write", 7, kv, rows)
    lender._admit(write, {0: 2})
    view = write.poll()
    with pytest.raises(MemoryError, match="planted"):
        write.mark_arrived([np.ones(len(run), dtype=bool) for run in view.groups])
    assert fired, "the quarantine did not raise: proves nothing"
    write.release()
    assert lender._free_slots(0) == 2, "slots a queued copy may still read came back"
    assert lender._must_retain_memory(), (
        "the shutdown would free staging a queued copy may still read"
    )


class _HoldingAppendRaisingOnce(list):
    """A holding list, with the leases of ``held``, whose first append raises ``MemoryError`` and
    notes it in ``fired``."""

    def __init__(self, held, fired):
        super().__init__(held)
        self.fired = fired

    def append(self, item):
        if not self.fired:
            self.fired.append(True)
            raise MemoryError("planted")
        super().append(item)


def test_a_write_whose_grant_and_failure_both_raise_is_never_ready(monkeypatch):
    """A write waits for the slots a read holds, and the read's release grants it: entering the
    holding list raises, then failing the write raises too. As after a manager hook logged that
    error, the write never returns a view over the slots it gave back for the next grant."""
    kv, manager = _write_target()
    ref = weakref.ref(manager)
    lender = _lender.Staging(ref, _one_pool_group(), None, (), SlotQueue({0: 2}), ref)
    landed = SimpleNamespace(query=lambda: True)
    monkeypatch.setattr(
        lender, "_build_view", lambda rows, keys: SimpleNamespace(groups=rows.block_ordinals)
    )
    monkeypatch.setattr(lender, "_copy_segments", lambda rows, mask=None: [])
    copied = (_lender._QueuedCopy(landed), None)
    monkeypatch.setattr(lender, "_queue_copy", lambda segments, to_staging: copied)
    read = _lender._StagingLease(
        lender,
        "read",
        8,
        None,
        _lender._LentBlockEntries([0], [np.arange(2)], [np.arange(2)]),
        None,
    )
    lender._admit(read, {0: 2})
    assert read.poll() is not None, "the read was not ready: proves nothing"
    write = _lender._StagingLease(
        lender, "write", 7, kv, _lender._LentBlockEntries([0], [np.arange(2)], [np.arange(2)])
    )
    lender._admit(write, {0: 2})
    assert write.poll() is None and write.failure is None, "the write did not wait"
    appended, failed = [], []
    recycle = lender._recycle

    def recycling():  # the holding list, once rebuilt, raises at the write's grant
        recycle()
        lender._leases_holding_slots = _HoldingAppendRaisingOnce(
            lender._leases_holding_slots, appended
        )

    monkeypatch.setattr(lender, "_recycle", recycling)
    monkeypatch.setattr(lender, "_fail", raising_once(lender._fail, MemoryError("planted"), failed))
    with pytest.raises(MemoryError, match="planted"):  # what a manager hook logs and goes on
        read.release()
    assert appended and failed, "a planted fault did not strike: proves nothing"
    assert write.poll() is None, "a write was ready over slots it gave back"


# -- a lender stopped by an error its free or shutdown hook logs ------------------------------


class _Memory:
    """The staging memory a stand-in lender finds at its shutdown; records each free."""

    def __init__(self):
        self.freed = []

    def free(self):
        self.freed.append(True)


def _cache(blocks=2, committed=0):
    """A stand-in cache whose blocks ``0..blocks-1`` lock pages ``0..blocks-1`` in every layer
    group, with ``committed`` committed tokens, which its history also has."""
    pages = list(range(blocks))
    return SimpleNamespace(
        num_committed_tokens=committed,
        history_length=committed,
        is_active=True,
        num_blocks=blocks,
        get_base_page_indices=lambda lg: pages,
        get_aggregated_page_indices=lambda lg, valid_only=False: pages,
    )


def _rows_of(count):
    """Blocks ``0..count-1`` on pages ``0..count-1``, in each of two layer groups."""
    ordinals = [np.arange(count), np.arange(count)]
    return _lender._LentBlockEntries([0, 1], ordinals, [o.copy() for o in ordinals])


def _request(request_id, prompt_len=4):
    return SimpleNamespace(
        py_request_id=request_id,
        prompt_len=prompt_len,
        py_multimodal_data=None,
        try_get_encoder_output_len=lambda: None,
    )


def _a_write_waits(monkeypatch, fault, error=MemoryError):
    """Two layer groups on pool groups 0 and 1 of four slots each: a read holds slots 0 and 1 of
    both with its copy pending, a write into request 8 whose view a poll returned holds slot 2, a
    write whose view no poll returned holds slot 3, and a write into request 7 waits for 2 + 2;
    request 10's cache has 8 committed blocks. The read is released and its copy lands. ``error``
    is planted in pool group 1, changed second, for its next take or return of slots as ``fault``
    says: the take of the waiting write's slots or the return of the read's."""
    manager = _Owner()
    manager.kv_cache_map = {7: _cache(), 8: _cache(), 10: _cache(blocks=8, committed=8)}
    layout = SimpleNamespace(
        pool_groups=(0, 1),
        pool_group_of=(0, 1),
        windows=(None, None),
        num_layer_groups=2,
        tokens_per_block=1,
    )
    memory = _Memory()
    lender = _lender.Staging(
        weakref.ref(manager), layout, None, (), SlotQueue({0: 4, 1: 4}), weakref.ref(memory)
    )
    s = SimpleNamespace(
        lender=lender, manager=manager, memory=memory, copies=[], errors=[], fired=[]
    )
    landed = []
    event = SimpleNamespace(query=lambda: bool(landed), synchronize=lambda: None)

    def memcpy(segments, to_staging):
        s.copies.append(to_staging)
        return _lender._QueuedCopy(event), None

    monkeypatch.setattr(
        lender, "_build_view", lambda rows, keys: SimpleNamespace(groups=rows.block_ordinals)
    )
    monkeypatch.setattr(lender, "_copy_segments", lambda rows, mask=None: [])
    monkeypatch.setattr(lender, "_queue_copy", memcpy)
    monkeypatch.setattr(_lender.logger, "error", s.errors.append)

    def lent(kind, request_id, count):
        kv = manager.kv_cache_map.get(request_id)
        fetch = None
        if kind == "write" and kv is not None:  # recorded as lend_write leaves it
            fetch = lender._ledger._latest_fetches[request_id] = _fetch_ledger._FetchRecord(0, kv)
        lease = _lender._StagingLease(lender, kind, request_id, kv, _rows_of(count), None, fetch)
        lender._admit(lease, {0: count, 1: count})
        return lease

    s.read = lent("read", 11, 2)
    s.seen = lent("write", 8, 1)
    s.seen_view = s.seen.poll()
    s.granted = lent("write", 12, 1)
    s.waiting = lent("write", 7, 2)
    assert s.seen_view is not None and s.granted._view is not None
    assert s.waiting.poll() is None and s.waiting.failure is None, "the write did not wait"
    s.read.release()  # its copy pends, so its slots stay out of use
    landed.append(True)
    if fault != "none":
        alloc = lender._slot_queue._allocators[1]
        planted = raising_once(getattr(alloc, fault), error("planted"), s.fired)
        monkeypatch.setattr(alloc, fault, planted)
    return s


def _freed_while_a_write_waits(monkeypatch, fault, error=MemoryError):
    """``_a_write_waits``, then request 9's free runs the manager's hook."""
    s = _a_write_waits(monkeypatch, fault, error)
    s.lender._on_free(9, object())  # the manager's hook: it raises nothing into the free
    assert bool(s.fired) == (fault != "none"), "the planted fault did not strike: proves nothing"
    return s


@pytest.mark.parametrize(
    "fault, error",
    [
        ("none", None),
        ("take", MemoryError),
        ("give", MemoryError),
        ("take", KeyError),
        ("give", KeyError),
    ],
    ids=["none", "take", "give", "take_key_error", "give_key_error"],
)
def test_an_error_the_free_hook_logs_stops_the_lender(monkeypatch, fault, error):
    """Another request's free returns the slots of a released read whose copy landed and grants
    them to the write waiting for them, and a planted MemoryError or KeyError strikes between the
    changes of the two pool groups: at the take of the write's slots or at the return of the read's.
    The hook logs it, raises nothing into the free, and stops the lender for good: the next lender
    call fails the waiting write, a poll fails a lease no poll returned the view of and returns a
    view one did, lends fail at the call after their range checks, resume_bounds is an empty interval,
    a mark copies nothing, no slot returns or is granted, and the shutdown keeps the staging memory
    with nothing open. ResumeBounds of a request without a cache still raises. A range needing more
    slots than a part has still raises in ``lend_write``, which checks the stop after its slot
    check, and fails at the call in ``lend_read``, which checks the stop before any check of the
    cache, each where it checks a reuse reset. Without the fault the free grants the waiting write,
    and that range raises in both lends."""
    s = _freed_while_a_write_waits(monkeypatch, fault, error)
    lender, waiting = s.lender, s.waiting
    big = _request(10, prompt_len=9)  # [0, 8) needs 8 slots of each pool group, which has 4
    if fault == "none":
        assert not s.errors, s.errors
        assert waiting.poll() is not None and waiting.failure is None, "the free granted nothing"
        assert s.granted.poll() is not None and s.seen.poll() is s.seen_view
        for lend in (lender.lend_read, lender.lend_write):
            with pytest.raises(ValueError, match="more staging slots"):
                lend(big, 0, 8)
        for lease in (s.seen, s.granted, waiting):
            lease.mark_arrived([np.full(len(run), lease is s.seen) for run in lease._view.groups])
            lease.release()
        assert s.copies == [True, False], "no copy of the marked rows into the request's pages"
        lender._on_shutdown()
        assert s.memory.freed == [True], "the shutdown kept staging that nothing holds"
        return
    assert len(s.errors) == 1 and "planted" in s.errors[0], s.errors
    free = (0, 2) if fault == "take" else (2, 0)  # pool group 0 changed, pool group 1 not
    assert (lender._free_slots(0), lender._free_slots(1)) == free, "no fault between the groups"
    requests = [_request(7), _request(8)]
    intervals = [lender.resume_bounds(request) for request in requests]  # progress comes first
    assert waiting.failure is not None, "the waiting write neither became ready nor failed"
    stop = waiting.failure
    assert stop == _lender._REASON_STOPPED, stop
    assert waiting.poll() is None
    for request, interval in zip(requests, intervals):
        assert interval is not None and interval.max_position < interval.min_position, (
            f"request {request.py_request_id}: resume_bounds {interval}, not an empty interval"
        )
    no_cache = _request(9)
    with pytest.raises(ValueError, match="has no KV cache"):
        lender.resume_bounds(no_cache)
    assert s.granted.poll() is None and s.granted.failure == stop, "a view went out after the stop"
    assert s.seen.poll() is s.seen_view, "a view a poll returned before the stop was taken back"
    for lend in (lender.lend_read, lender.lend_write):
        lease = lend(no_cache, 0, 2)
        assert lease.failure == stop, f"{lend.__name__} did not fail at the call: {lease.failure}"
        lease.release()
    with pytest.raises(ValueError):
        lender.lend_read(no_cache, 2, 0)
    with pytest.raises(ValueError):
        lender.lend_write(no_cache, 0, 4)  # past the whole blocks before the last prompt token
    with pytest.raises(ValueError, match="more staging slots"):
        lender.lend_write(big, 0, 8)
    read = lender.lend_read(big, 0, 8)
    assert read.failure == stop, f"lend_read of the range did not fail at the call: {read.failure}"
    read.release()
    marks = [np.ones(len(run), dtype=bool) for run in s.seen_view.groups]
    s.seen.mark_arrived(marks)
    assert s.copies == [True], "a mark queued a copy into the request's pages"
    with pytest.raises(RuntimeError, match="twice"):
        s.seen.mark_arrived(marks)
    for lease in (s.seen, s.granted, waiting):
        lease.release()
    assert lender._open_lease_count() == 0
    assert (lender._free_slots(0), lender._free_slots(1)) == free, "slots returned or granted"
    assert lender._must_retain_memory(), "the shutdown would free the staging memory"
    warnings = []
    monkeypatch.setattr(_lender.logger, "warning", warnings.append)
    lender._on_shutdown()
    assert not s.memory.freed, "the shutdown freed staging whose slot records may be wrong"
    assert any(stop in warning for warning in warnings), f"the stop is not logged: {warnings}"
    assert (lender._free_slots(0), lender._free_slots(1)) == free, "the shutdown returned slots"
    assert len(s.errors) == 1, f"the shutdown logged an error: {s.errors[1:]}"


@pytest.mark.parametrize("fault", ["take", "give"])
def test_a_shutdown_right_after_the_free_hook_stopped_the_lender(monkeypatch, fault):
    """As above, with the manager's shutdown the first call after the free: the shutdown itself
    fails the waiting write, before any poll, a later poll fails a lease no poll returned the view
    of and returns a view one did, the staging memory stays, and the shutdown logs no error of its
    own."""
    s = _freed_while_a_write_waits(monkeypatch, fault)
    s.lender._on_shutdown()
    assert s.waiting.failure == _lender._REASON_SHUT_DOWN, (
        "the shutdown left the waiting write in line"
    )
    assert s.waiting.poll() is None
    assert s.granted.poll() is None and s.granted.failure == _lender._REASON_STOPPED, (
        "a view went out"
    )
    assert s.seen.poll() is s.seen_view, "a view a poll returned before the stop was taken back"
    assert not s.memory.freed, "the shutdown freed staging whose slot records may be wrong"
    assert len(s.errors) == 1, f"the shutdown logged an error: {s.errors[1:]}"


@pytest.mark.parametrize("before", ["live", "stopped"])
def test_an_error_the_shutdown_logs_stops_the_lender(monkeypatch, before):
    """The manager's shutdown meets an error in its own work and logs it, raising nothing into the
    shutdown: on a live lender a planted MemoryError between the returns of the two pool groups of
    a released read whose copy landed; on a lender the free hook stopped, a planted MemoryError in
    the warning that it keeps the staging memory. The lender stops, its first reason staying: the
    waiting write fails by its next poll, a poll fails a lease no poll returned the view of rather
    than handing the view out, a view a poll returned keeps returning, and the staging memory
    stays."""
    if before == "live":
        s = _a_write_waits(monkeypatch, "give")
        struck = s.fired
    else:
        s = _freed_while_a_write_waits(monkeypatch, "take")
        struck = []
        warned = raising_once(_lender.logger.warning, MemoryError("planted"), struck)
        monkeypatch.setattr(_lender.logger, "warning", warned)
    s.lender._on_shutdown()  # the manager's hook: it raises nothing into the shutdown
    assert struck, "the planted fault did not strike: proves nothing"
    assert len(s.errors) == 1 + (before == "stopped") and "planted" in s.errors[-1], s.errors
    assert s.waiting.poll() is None and s.waiting.failure is not None, "the waiting write pends"
    assert s.granted.poll() is None and s.granted.failure is not None, "a view went out"
    assert s.seen.poll() is s.seen_view, "a view a poll returned was taken back"
    assert not s.memory.freed, "the shutdown freed staging whose slot records may be wrong"
    with pytest.raises(ValueError, match="shut down"):
        s.lender.resume_bounds(_request(7))
    first = _lender._REASON_STOPPED_AT_SHUTDOWN if before == "live" else _lender._REASON_STOPPED
    assert s.granted.failure == first, f"not the first reason: {s.granted.failure}"


def test_a_clean_shutdown_does_not_stop_the_lender(monkeypatch):
    """Without a fault the shutdown fails the waiting write with its own reason and keeps the
    staging memory for the open leases, and a write no poll returned the view of returns its view
    at its first poll after it: only an error the shutdown logs stops the lender."""
    s = _a_write_waits(monkeypatch, "none")
    warnings = []
    monkeypatch.setattr(_lender.logger, "warning", warnings.append)
    s.lender._on_shutdown()
    assert not s.errors, s.errors
    assert s.waiting.failure == _lender._REASON_SHUT_DOWN and s.waiting.poll() is None
    assert s.granted.poll() is not None and s.granted.failure is None, "a clean shutdown stopped it"
    assert s.seen.poll() is s.seen_view
    assert warnings and not s.memory.freed, "leases are open: the memory stays"


@pytest.mark.parametrize("hook", ["free", "shutdown"])
def test_a_hook_whose_error_log_raises_still_stops_the_lender(monkeypatch, hook):
    """A planted MemoryError strikes in a hook's own work, at the take in the free or at the return
    in the shutdown as above, and the hook's error log line raises a KeyError, which goes out into
    the manager's free or shutdown: the lender has stopped all the same, so the waiting write and a
    lease no poll returned the view of fail at their next poll, and a view a poll returned keeps
    returning."""
    s = _a_write_waits(monkeypatch, "take" if hook == "free" else "give")
    logged = []
    log = raising_once(s.errors.append, KeyError("log line"), logged)
    monkeypatch.setattr(_lender.logger, "error", log)
    with pytest.raises(KeyError, match="log line"):
        if hook == "free":
            s.lender._on_free(9, object())
        else:
            s.lender._on_shutdown()
    assert s.fired and logged, "a planted fault did not strike: proves nothing"
    assert s.waiting.poll() is None and s.waiting.failure is not None, "the waiting write pends"
    assert s.granted.poll() is None and s.granted.failure is not None, "a view went out"
    assert s.seen.poll() is s.seen_view, "a view a poll returned was taken back"
    stop = _lender._REASON_STOPPED if hook == "free" else _lender._REASON_STOPPED_AT_SHUTDOWN
    assert s.waiting.failure == s.granted.failure == stop, (s.waiting.failure, s.granted.failure)


# -- records of a cache the request no longer holds -------------------------------------------

# ResumeBounds of request 7 in ``_records_of``: without records, and with one record of each kind
# describing the request's current cache.
NO_RECORDS = (4, 4)
CURRENT_RECORDS = {"fetch": None, "delivered": (6, 4), "advanced": (2, 4), "tainted": (0, 4)}


def _records_of(monkeypatch, kind, owner):
    """A lender over a stand-in manager whose request 7 holds a cache of 8 one-token blocks in two
    full-attention layer groups, with 2 committed tokens, a history of 4 and the floor at the
    history. One record of ``kind`` describes that cache, or with ``owner`` set to ``"replaced"``
    the cache the request held before: an undelivered fetch, blocks 2 to 5 delivered, a fetch from
    2 that kept the 2 tokens below it at a history of 4, or a taint."""
    monkeypatch.setattr(_ranges, "_floor_follows_history", lambda manager: True)
    manager = _Owner()
    kv, old = _cache(blocks=8, committed=2), _cache(blocks=8, committed=2)
    kv.history_length = old.history_length = 4
    manager.kv_cache_map = {7: kv}
    layout = SimpleNamespace(
        pool_groups=(0, 1),
        pool_group_of=(0, 1),
        windows=(None, None),
        num_layer_groups=2,
        tokens_per_block=1,
    )
    ref = weakref.ref(manager)
    lender = _lender.Staging(ref, layout, None, (), SlotQueue({0: 4, 1: 4}), ref)
    cache = kv if owner == "current" else old
    rows = np.zeros(6, dtype=bool)
    rows[2:] = True
    records, record = {
        "fetch": (lender._ledger._latest_fetches, _fetch_ledger._FetchRecord(2, cache)),
        "delivered": (
            lender._ledger._deliveries,
            _fetch_ledger._Deliveries(cache, 2, [rows, rows.copy()]),
        ),
        "advanced": (lender._ledger._kept_prefixes, (cache, 2, 4)),
        "tainted": (lender._ledger._mixed_tail_caches, cache),
    }[kind]
    records[7] = record
    return SimpleNamespace(lender=lender, manager=manager, record=record, records=records)


@pytest.mark.parametrize("owner", ["replaced", "current"])
@pytest.mark.parametrize("kind", list(CURRENT_RECORDS))
def test_resume_bounds_drops_the_records_of_a_cache_the_request_no_longer_holds(
    monkeypatch, kind, owner
):
    """The manager's free takes a request's cache out of its map, then calls
    ``discard_pending_stats`` and then the lender's free hook, so where ``discard_pending_stats``
    raises, the lender keeps the request's records of a cache the request no longer holds.
    ResumeBounds of its next cache ignores and drops such a record of each kind the lender keeps per
    request, answering as without records; the same record of the current cache changes resume_bounds
    and stays."""
    s = _records_of(monkeypatch, kind, owner)
    resume_bounds = s.lender.resume_bounds(_request(7))
    if owner == "current":
        assert resume_bounds == CURRENT_RECORDS[kind] != NO_RECORDS, (
            f"resume_bounds {resume_bounds}"
        )
        assert s.records.get(7) is s.record, "resume_bounds dropped a record of the current cache"
        return
    assert resume_bounds == NO_RECORDS, (
        f"resume_bounds {resume_bounds} reads a record of a replaced cache"
    )
    assert 7 not in s.records, "resume_bounds kept a record of a replaced cache"


# -- a fetch's states -------------------------------------------------------------------------


def test_a_fetch_walks_through_the_states_its_enum_names(monkeypatch):
    """Write leases into a stand-in cache, ``_fetch_state`` read after each call: the first fetch
    open, delivered with its copy pending, settled once the copy completed, then replaced by the
    next, which is abandoned when its lease is released before a poll returned the view; a third is
    abandoned when the copy of its marks fails. A new fetch is refused while one is unsettled."""
    states = _fetch_ledger._FetchState
    lay = SimpleNamespace(
        pool_groups=(0,),
        pool_group_of=(0,),
        windows=(None,),
        num_layer_groups=1,
        tokens_per_block=1,
    )
    kv = SimpleNamespace(
        num_committed_tokens=0,
        history_length=0,
        is_active=True,
        num_blocks=2,
        get_base_page_indices=lambda lg: [0, 1],
    )
    manager = _Owner()
    manager.kv_cache_map = {7: kv}
    ref = weakref.ref(manager)
    lender = _lender.Staging(ref, lay, None, (), SlotQueue({0: 4}), ref)
    landed, failure = [], {}
    event = SimpleNamespace(query=lambda: bool(landed))
    monkeypatch.setattr(
        lender, "_build_view", lambda rows, keys: SimpleNamespace(groups=rows.block_ordinals)
    )
    monkeypatch.setattr(lender, "_copy_segments", lambda rows, mask: [])
    monkeypatch.setattr(
        lender,
        "_queue_copy",
        lambda segments, to_staging: (_lender._QueuedCopy(event), failure.get("error")),
    )
    seen = set()

    def state(lease, expected):
        got = lender._ledger._fetch_state(7, lease._fetch)
        seen.add(got)
        assert got is expected, f"{got.name}, not {expected.name}"

    def lent():
        """A write lease as ``lend_write`` leaves it once the cache grew: recorded, granted."""
        fetch = _fetch_ledger._FetchRecord(0, kv)
        lender._ledger._latest_fetches[7] = fetch
        rows = _lender._LentBlockEntries([0], [np.arange(2)], [np.arange(2)])
        lease = _lender._StagingLease(lender, "write", 7, kv, rows, None, fetch)
        lender._admit(lease, {0: 2})
        return lease

    def marked(lease):
        lease.mark_arrived([np.ones(len(run), dtype=bool) for run in lease.poll().groups])

    first = lent()
    state(first, states.AWAITING_MARKS)
    assert lender._ledger._unsettled_fetch_refusal(7, kv), "a fetch started while another was open"
    marked(first)
    state(first, states.COPY_PENDING)
    assert lender._ledger._unsettled_fetch_refusal(7, kv), (
        "a fetch started while another's copy was pending"
    )
    first.release()
    landed.append(True)
    lender._progress()  # as a lend, a poll or a resume_bounds call does first; it asks the copy again
    state(first, states.COPY_DONE)
    assert lender._ledger._unsettled_fetch_refusal(7, kv) is None, (
        "a settled fetch kept a new one from starting"
    )
    second = lent()
    state(first, states.RETIRED)
    state(second, states.AWAITING_MARKS)
    second.release()  # before a poll returned its view
    state(second, states.ABANDONED)
    third = lent()
    failure["error"] = "planted"
    marked(third)
    state(third, states.ABANDONED)
    third.release()
    assert seen == set(states), f"states never seen: {set(states) - seen}"


# -- the hooks the manager calls --------------------------------------------------------------

# The manager's module, beside the sharing package.
MANAGER_SOURCE = Path(_lender.__file__).resolve().parents[1] / "kv_cache_manager_v2.py"
# Each lender method the manager calls on its ``_sharing``: positional arguments, keywords.
HOOKS = {
    "_on_free": (2, ()),
    "_on_shrink": (2, ()),
    "_on_shutdown": (0, ()),
    "_on_caches_closed": (0, ()),
    "_on_reset": (0, ()),
}


def hook_calls(source):
    """Each method ``source`` calls on an attribute named ``_sharing``, with the positional
    arguments and the keywords of each of its calls."""
    calls = {}
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "_sharing"
        ):
            shape = (len(node.args), tuple(k.arg for k in node.keywords))
            calls.setdefault(node.func.attr, set()).add(shape)
    return calls


def unbound_hooks(cls, calls):
    """The methods of ``calls`` that ``cls`` lacks, or whose signature cannot bind one of their
    calls."""
    unbound = []
    for name, shapes in sorted(calls.items()):
        try:
            signature = inspect.signature(getattr(cls, name))
            for args, keywords in shapes:
                signature.bind(None, *[None] * args, **dict.fromkeys(keywords))
        except (AttributeError, TypeError, ValueError):
            unbound.append(name)
    return unbound


def test_the_lender_has_every_hook_the_manager_calls(monkeypatch):
    """The manager calls five lender methods on its ``_sharing``, and ``Staging`` has each, taking
    the arguments the manager passes. The check finds a ``Staging`` planted without one of them,
    and one planted with a hook that takes fewer arguments."""
    calls = hook_calls(MANAGER_SOURCE.read_text())
    assert calls == {name: {shape} for name, shape in HOOKS.items()}, calls
    assert unbound_hooks(_lender.Staging, calls) == []
    with monkeypatch.context() as patched:
        patched.delattr(_lender.Staging, "_on_reset")
        assert unbound_hooks(_lender.Staging, calls) == ["_on_reset"], "a missing hook passed"
    with monkeypatch.context() as patched:
        patched.setattr(_lender.Staging, "_on_free", lambda self, request_id: None)
        assert unbound_hooks(_lender.Staging, calls) == ["_on_free"], "a short hook passed"


# -- the bytes the test kit stages ------------------------------------------------------------


def test_the_pieces_of_a_staged_row_differ(kit):
    """The 32-byte pieces of a staged row are pairwise distinct, so two of them trading places
    shows in the row's bytes."""
    row = kit.staged_row(0x5A, 1, 3, 4096)
    pieces = [row[i : i + 32] for i in range(0, len(row), 32)]
    assert len(set(pieces)) == len(pieces), "a staged row repeats a piece, so a swap would not show"


# -- a lease's axes ------------------------------------------------------------------------------


def test_each_axis_of_a_lease_follows_its_own_fields_alone():
    """Each axis of a staging lease follows its own fields alone: the holder axis the failure and
    whether a poll returned the view, the slot axis the ticket, the copy axis the copy. Checked
    over every combination of those four fields, ticket 0 (the first a slot queue issues) and an
    empty failure among them, and of four lease fields no axis reads."""
    lender, _ = staging_over([None], [0], 32)
    holder, slots, copies = _lender._HolderState, _lender._SlotState, _lender._CopyState
    queued = _lender._QueuedCopy(SimpleNamespace(query=lambda: False))
    recorded = {None: True, _lender._COPY_WITHOUT_RECORD: False, queued: True}
    for failure, seen, ticket, copy, released, marked, runs, doomed in itertools.product(
        [None, "failed", ""],
        [False, True],
        [None, 0, 3],
        recorded,
        [False, True],
        [False, True],
        [None, SlotRuns({0: (0, 1)})],
        [None, "doomed"],
    ):
        lease = _lender._StagingLease(lender, "write", 7)
        lease._failure, lease._view_returned, lease._ticket = failure, seen, ticket
        lease._copy, lease._released, lease._marked = copy, released, marked
        lease._slot_runs, lease._first_poll_failure = runs, doomed
        failed = failure is not None
        want = (
            holder.FAILED if failed else holder.VIEW_RETURNED if seen else holder.VIEW_PENDING,
            slots.WAITING if ticket is not None else slots.NOT_WAITING,
            copies.NONE_OR_RECORDED if recorded[copy] else copies.WITHOUT_RECORD,
        )
        got = (lease._holder_state(), lease._slot_state(), lease._copy_state())
        assert got == want, (
            f"failure {failure}, seen {seen}, ticket {ticket}, copy {copy}, released {released}, "
            f"marked {marked}, runs {runs}, doomed {doomed}: {got}"
        )


class _Cache:
    """A stand-in cache a weak reference can follow."""


def test_a_lender_and_its_records_go_with_its_last_reference():
    """With the cyclic collector off, a lender and a cache its fetch records hold go as soon as the
    last reference to the lender does: its ledger reaches it only through a weak reference, so no
    reference cycle keeps either until a collection."""
    lender, _ = staging_over([None], [0], 32)
    cache = _Cache()
    lender._ledger._kept_prefixes[7] = (cache, 0, 1)
    lender_ref, cache_ref = weakref.ref(lender), weakref.ref(cache)
    collecting = gc.isenabled()
    gc.disable()
    try:
        del lender, cache
        assert lender_ref() is None, "a reference cycle keeps the lender after its last reference"
        assert cache_ref() is None, "the lender's records keep a cache after its last reference"
    finally:
        if collecting:
            gc.enable()
