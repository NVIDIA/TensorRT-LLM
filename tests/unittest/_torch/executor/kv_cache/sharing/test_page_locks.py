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
"""The page-lock registry, the manager's lock hooks and the generation update under locks."""

from __future__ import annotations

import ast
import copy
import enum
import gc
import numbers
import random
import traceback
import types
import weakref
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import (
    ALL_ARRIVED,
    HoldReason,
    StagingOptions,
    _lender,
    _manager,
    _page_locks,
    attach_lender,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing._page_locks import LockMode, PageLocks

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="allocates KV cache pools")

READ, WRITE = LockMode.READ, LockMode.WRITE
TPB = 32


# -- the registry over stand-ins ------------------------------------------------------------------


class Manager:
    """A stand-in manager: three layer groups, the first and last on pool group 0. It notes each
    cache it retries a held-back commit of, with that cache's closes then; with ``held``, a record
    of held commits by cache ``id`` as the KV cache manager keeps, it retries only those."""

    def __init__(self, pool_group_of=(0, 1, 0), held=None):
        self.impl = SimpleNamespace(get_life_cycle_pool_group_indices=lambda: list(pool_group_of))
        self.retried = []
        if held is not None:
            self._held_commits = held

    def _commit_held_back(self, kv):
        held = getattr(self, "_held_commits", None)
        if held is None or held.pop(id(kv), None) is not None:
            self.retried.append((kv, kv.closes))


class Cache:
    """A stand-in runtime cache that counts its closes: its committed tokens in each of three layer
    groups and, per beam, a page per block of each layer group, -1 for none."""

    class Status(enum.Enum):
        OPEN = 0
        CLOSED = 1

    def __init__(self, committed_tokens=0, beams=None):
        self.status = self.Status.OPEN
        self.num_committed_tokens = committed_tokens
        self.reuse_status = [SimpleNamespace(complete=True, endpoint=committed_tokens)] * 3
        self.tokens_per_block = TPB
        self._beams = beams or [{}]
        self.beam_width = len(self._beams)
        self.closes = 0

    def close(self):
        self.closes += 1
        self.status = self.Status.CLOSED

    def get_scratch_desc(self, lg):
        return None

    def get_aggregated_page_indices(self, lg, beam_id=0, valid_only=False):
        pages = self._beams[beam_id].get(lg, [])
        return [p for p in pages if p >= 0] if valid_only else pages


class Event:
    """A stand-in CUDA event: complete once ``done`` is set; waiting on it fails the test."""

    def __init__(self, done=False):
        self.done = done
        self.queries = 0

    def query(self):
        self.queries += 1
        return self.done

    def synchronize(self):
        raise AssertionError("a lock waited on its event")


@pytest.fixture
def recorded(monkeypatch):
    """The CUDA events made while the test runs."""
    events, make = [], torch.cuda.Event

    def counted(*args, **kwargs):
        events.append(make(*args, **kwargs))
        return events[-1]

    monkeypatch.setattr(torch.cuda, "Event", counted)
    return events


MANAGER = Manager()  # the registry holds its manager weakly


def registry(manager=MANAGER, recurrent=()):
    return PageLocks(weakref.ref(manager), recurrent)


def locked(owner):
    """The locked pages per pool group ``owner.held_page_counts()`` gives."""
    return dict(owner.held_page_counts().by_reason.get(HoldReason.LOCKED, {}))


def deducted(owner):
    return dict(owner.held_page_counts().deduct)


@pytest.mark.cpu_only
def test_readers_share_blocks_and_a_writer_shares_none(recorded):
    locks, kv, other = registry(), Cache(), Cache()
    first = locks.lock(kv, {0: [1, 2, 3]}, READ)
    second = locks.lock(kv, {0: [3, 4], 1: [0]}, READ)
    assert {first.mode, second.mode} == {READ}
    for blocks in ({0: [3]}, {0: [0, 4]}, {1: [0]}):
        with pytest.raises(ValueError, match="locked for read"):
            locks.lock(kv, blocks, WRITE)
    writer = locks.lock(kv, {0: [5, 6], 2: [3]}, WRITE)  # other blocks, another layer group
    for blocks, mode in (({0: [6]}, READ), ({0: [6]}, WRITE), ({2: [0, 3]}, READ)):
        with pytest.raises(ValueError, match="locked for write"):
            locks.lock(kv, blocks, mode)
    # Another cache's blocks and another layer group's ordinals are other blocks.
    locks.lock(other, {0: [1, 2, 3, 5, 6]}, WRITE)
    locks.lock(kv, {1: [5, 6]}, WRITE)
    locks.unlock(writer)
    again = locks.lock(kv, {0: [5, 6], 2: [3]}, WRITE)
    assert first.is_ready() and second.is_ready() and again.is_ready(), "a lock without an event"
    assert recorded == [], "a lock made an event"


@pytest.mark.cpu_only
def test_a_refused_lock_changes_nothing(recorded):
    locks, kv = registry(), Cache(committed_tokens=2 * TPB)
    reader = locks.lock(kv, {0: [2, 3]}, READ)
    before = (locked(locks), locks.is_locked(kv, 4), len(recorded))
    refused = [
        lambda: locks.lock(kv, {0: [3, 9]}, WRITE),  # a reader holds block 3
        lambda: locks.lock(kv, {0: [1, 9]}, WRITE),  # block 1 is committed
        lambda: locks.lock(kv, {1: [9], 0: [-1]}, READ),
        lambda: locks.lock(kv, {3: [9]}, READ),  # three layer groups
        lambda: locks.lock(kv, {0: [9]}, READ, level=1),
    ]
    for call in refused:
        with pytest.raises(ValueError):
            call()
    assert (locked(locks), locks.is_locked(kv, 4), len(recorded)) == before
    locks.unlock(reader)
    assert not locks.is_locked(kv) and locked(locks) == {}


@pytest.mark.cpu_only
def test_a_lock_another_registry_holds_is_not_released_here():
    """Unlocking another registry's lock raises and changes neither registry."""
    mine, theirs, kv = registry(), registry(), Cache()
    held = mine.lock(kv, {0: [1, 2]}, READ)
    foreign = theirs.lock(kv, {0: [5]}, READ)
    before = (mine.holding, mine.version, mine.is_locked(kv, 1, 3))
    with pytest.raises(ValueError, match="not held in this registry"):
        mine.unlock(foreign)
    assert (mine.holding, mine.version, mine.is_locked(kv, 1, 3)) == before
    assert not foreign.released and theirs.is_locked(kv, 5, 6)
    for locks, lock in ((mine, held), (theirs, foreign)):
        locks.unlock(lock)
    assert not mine.is_locked(kv) and not theirs.is_locked(kv)


@pytest.mark.cpu_only
def test_a_writer_locks_only_blocks_past_the_committed_whole_blocks():
    locks = registry()
    kv = Cache(committed_tokens=2 * TPB + 5)  # 2 whole blocks, and 5 tokens of the third
    with pytest.raises(ValueError, match="past the 2 committed whole blocks"):
        locks.lock(kv, {0: [2, 3], 1: [1, 2]}, WRITE)
    writer = locks.lock(kv, {0: [2, 3], 1: [2]}, WRITE)
    reader = locks.lock(kv, {0: [0, 1], 1: [0, 1]}, READ)  # committed blocks are read
    assert locks.is_locked(kv, 0, 1) and locks.is_locked(kv, 3, 4)
    for lock in (writer, reader):
        locks.unlock(lock)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "blocks, mode, level, error",
    [
        ({0: [1]}, "read", None, TypeError),
        ({0: [1]}, READ, 0, ValueError),
        ({True: [1]}, READ, None, TypeError),
        ({1.0: [1]}, READ, None, TypeError),
        ({-1: [1]}, READ, None, ValueError),
        ({3: [1]}, READ, None, ValueError),
        ({0: [1.5]}, READ, None, ValueError),
        ({0: [[1]]}, READ, None, ValueError),
        ({0: [True]}, READ, None, ValueError),
        ({0: [0, -2]}, READ, None, ValueError),
    ],
)
def test_a_lock_takes_integer_blocks_of_known_layer_groups_in_place(
    recorded, blocks, mode, level, error
):
    locks, kv = registry(), Cache()
    with pytest.raises(error):
        locks.lock(kv, blocks, mode, level)
    assert not locks.is_locked(kv) and recorded == []


@pytest.mark.cpu_only
def test_a_lock_holds_sorted_distinct_read_only_ordinals():
    locks, kv = registry(), Cache()
    given = np.array([7, 3, 3, 5], dtype=np.int32)
    lock = locks.lock(kv, {2: given, 0: range(2), 1: []}, READ)
    assert {lg: o.tolist() for lg, o in lock.blocks.items()} == {2: [3, 5, 7], 0: [0, 1], 1: []}
    for ordinals in lock.blocks.values():
        assert ordinals.dtype == np.int64 and not ordinals.flags.writeable
    with pytest.raises(TypeError):
        lock.blocks[3] = np.zeros(1)  # a read-only mapping
    given[0] = 0
    assert lock.blocks[2].tolist() == [3, 5, 7], "the lock shares the caller's array"


def frozen(values):
    """Ordinals as the lender makes them: sorted, distinct, ``int64`` on immutable bytes."""
    return np.frombuffer(np.asarray(values, dtype=np.int64).tobytes(), dtype=np.int64)


@pytest.mark.cpu_only
def test_a_lock_of_ordered_ordinals_takes_them_as_they_are_and_still_refuses(recorded):
    """A lock takes the lender's ordered arrays as they are and refuses them as any blocks."""
    locks, kv = registry(), Cache(committed_tokens=2 * TPB)
    given = {0: frozen([2, 3, 5]), 1: frozen([]), 2: frozen([0])}
    reader = locks.lock(kv, given, READ, ordered=True)
    assert all(reader.blocks[lg] is given[lg] for lg in given)
    assert locks.is_locked(kv, 5, 6) and not locks.is_locked(kv, 4, 5)
    refused = (
        ({0: frozen([3, 9])}, WRITE, "locked for read"),
        ({0: frozen([1, 9])}, WRITE, "past the 2 committed whole blocks"),
        ({3: frozen([9])}, READ, "the manager has 3 layer groups"),
    )
    for blocks, mode, match in refused:
        with pytest.raises(ValueError, match=match):
            locks.lock(kv, blocks, mode, ordered=True)
        found = locks.check(kv, blocks, mode, ordered=True)
        assert isinstance(found, str) and found == locks.refusal(kv, blocks, mode)
    # Committed tokens the caller read stand for the cache's own count.
    past = {0: frozen([1, 9])}
    for committed in (None, (2 * TPB,) * 3):
        assert "past the 2" in locks.check(kv, past, WRITE, ordered=True, committed=committed)
    assert isinstance(
        locks.check(kv, past, WRITE, ordered=True, committed=(0,) * 3), _page_locks.CheckedBlocks
    )
    assert reader.is_ready()
    locks.unlock(reader)
    assert recorded == []


@pytest.mark.cpu_only
def test_a_checked_lock_takes_what_its_check_found_until_a_lock_or_unlock_comes_between():
    """A lock takes its check's result while no lock or unlock came between, else checks again."""
    locks, kv, other = registry(), Cache(), Cache()
    blocks = {0: frozen([4, 5]), 2: frozen([1])}
    checked = locks.check(kv, blocks, WRITE, ordered=True)
    assert isinstance(checked, _page_locks.CheckedBlocks)
    assert (checked.kv, checked.mode, checked.version) == (kv, WRITE, locks.version)
    assert all(checked.blocks[lg] is blocks[lg] for lg in blocks)
    writer = locks.lock(kv, blocks, WRITE, ordered=True, checked=checked)
    assert all(writer.blocks[lg] is blocks[lg] for lg in blocks)
    locks.unlock(writer)
    checked = locks.check(kv, blocks, WRITE, ordered=True)
    reader = locks.lock(kv, {0: [5]}, READ)  # comes between the check and the lock
    with pytest.raises(ValueError, match="locked for read"):
        locks.lock(kv, blocks, WRITE, ordered=True, checked=checked)
    elsewhere = locks.check(other, blocks, WRITE, ordered=True)
    as_reader = locks.check(kv, {0: frozen([5])}, READ, ordered=True)
    for found in (elsewhere, as_reader):
        assert isinstance(found, _page_locks.CheckedBlocks)
        with pytest.raises(ValueError, match="locked for read"):
            locks.lock(kv, {0: frozen([5])}, WRITE, ordered=True, checked=found)
    locks.unlock(reader)
    assert not locks.is_locked(kv) and not locks.is_locked(other)


@pytest.mark.cpu_only
def test_a_checked_lock_cannot_reopen_a_cache_deferred_at_free():
    locks, kv = registry(), Cache()
    first = locks.lock(kv, {0: [0]}, READ)
    checked = locks.check(kv, {0: [1]}, WRITE)
    assert isinstance(checked, _page_locks.CheckedBlocks)

    assert locks.defer_close(kv)
    with pytest.raises(ValueError, match="freed"):
        locks.lock(kv, {0: [1]}, WRITE, checked=checked)
    assert locks.holding == 1
    locks.unlock(first)
    assert kv.closes == 1


@pytest.mark.cpu_only
def test_a_closed_cache_refuses_checked_and_new_locks():
    locks, kv = registry(), Cache()
    checked = locks.check(kv, {0: [0]}, WRITE)
    assert isinstance(checked, _page_locks.CheckedBlocks)
    assert not locks.defer_close(kv)
    kv.close()

    with pytest.raises(ValueError, match="closed"):
        locks.lock(kv, {0: [0]}, WRITE, checked=checked)
    with pytest.raises(ValueError, match="closed"):
        locks.lock(kv, {0: [0]}, READ)
    assert "closed" in locks.check(kv, {0: [0]}, READ)
    assert locks.lock_fresh(kv, {0: frozen([0])}, READ, None) is None
    assert not locks.is_locked(kv)


@pytest.mark.cpu_only
@pytest.mark.parametrize("intervening", ["none", "held", "released"])
def test_a_checked_write_rechecks_blocks_committed_since_check(intervening):
    locks, kv = registry(), Cache()
    committed = (0,) * 3
    checked = locks.check(kv, {0: [1]}, WRITE, ordered=True, committed=committed)
    assert isinstance(checked, _page_locks.CheckedBlocks)
    if intervening != "none":
        other = locks.lock(kv, {0: [3]}, READ)
        if intervening == "released":
            locks.unlock(other)
    kv.num_committed_tokens = 2 * TPB

    with pytest.raises(ValueError, match="past the 2 committed whole blocks"):
        locks.lock(kv, {0: [1]}, WRITE, checked=checked, ordered=True, committed=committed)
    assert not locks.is_locked(kv, 1, 2)


@pytest.mark.cpu_only
def test_an_ordered_lock_on_an_unlocked_cache_takes_and_refuses_what_its_check_does():
    """Random ordered blocks: a lock on an unlocked cache takes and refuses what its check does."""
    rng = random.Random(0)
    locks = registry()
    for _ in range(600):
        tokens = rng.choice((0, 0, rng.randint(1, 3 * TPB)))
        kv = Cache(committed_tokens=tokens)
        groups = rng.sample(range(4), rng.randint(0, 3))
        blocks = {lg: frozen(sorted(rng.sample(range(6), rng.randint(0, 3)))) for lg in groups}
        mode = rng.choice((READ, WRITE))
        committed = rng.choice((None, (0,) * 3, (tokens,) * 3))
        found = locks.check(kv, blocks, mode, ordered=True, committed=committed)
        try:
            lock = locks.lock(kv, blocks, mode, ordered=True, committed=committed)
        except ValueError as error:
            assert found == str(error)
        else:
            assert isinstance(found, _page_locks.CheckedBlocks)
            assert lock.blocks.keys() == blocks.keys()
            assert all(lock.blocks[lg] is blocks[lg] for lg in blocks)
            locks.unlock(lock)
        assert not locks.is_locked(kv)


@pytest.mark.cpu_only
def test_readiness_only_queries_the_event_and_drops_it_once_complete(recorded):
    locks, kv = registry(), Cache()
    given = Event()
    lock = locks.lock(kv, {0: [0]}, WRITE, ready=given)
    assert recorded == [], "a lock made an event besides the one it was given"
    assert not lock.is_ready() and not lock.is_ready()
    assert given.queries == 2
    given.done = True
    assert lock.is_ready() and lock.is_ready()
    assert given.queries == 3, "a completed event was queried again"
    held = weakref.ref(given)
    del given
    assert held() is None, "the lock keeps its event after it completed"
    # Without an event, nothing in flight touches the pages: the lock is ready at once.
    mine = locks.lock(kv, {0: [1]}, WRITE)
    assert recorded == [] and mine.is_ready()


@pytest.mark.cpu_only
def test_unlock_is_idempotent_and_ends_access():
    locks, kv = registry(), Cache()
    event = Event(done=True)
    lock = locks.lock(kv, {0: [0, 1]}, WRITE, ready=event)
    held = weakref.ref(event)
    del event
    locks.unlock(lock)
    locks.unlock(lock)
    assert lock.released and not lock.is_ready()
    assert held() is None, "an unlocked lock keeps its event"
    assert not locks.is_locked(kv)
    again = locks.lock(kv, {0: [0, 1]}, WRITE)  # the blocks are free again
    with pytest.raises(ValueError, match="not held"):
        registry(Manager()).unlock(again)
    assert locks.is_locked(kv)


@pytest.mark.cpu_only
def test_the_holding_count_follows_every_lock_and_unlock(recorded):
    """Through random locks and unlocks, ``holding`` counts the locks that hold a block."""
    rng = random.Random(5)
    locks = registry()
    caches = [Cache() for _ in range(3)]
    held = []
    for _ in range(400):
        if held and rng.random() < 0.4:
            locks.unlock(held.pop(rng.randrange(len(held))))
        else:
            blocks = {
                lg: sorted(rng.sample(range(8), rng.randint(0, 2)))
                for lg in range(3)
                if rng.random() < 0.6
            }
            try:
                held.append(locks.lock(rng.choice(caches), blocks, rng.choice((READ, WRITE))))
            except ValueError:
                pass
        assert locks.holding == sum(any(map(len, lock.blocks.values())) for lock in held)
        assert locks.holds_any() == any(locks.is_locked(kv) for kv in caches)
    assert recorded == []


@pytest.mark.cpu_only
def test_is_locked_reads_the_ordinal_range_across_layer_groups():
    locks, kv = registry(), Cache()
    assert not locks.is_locked(kv) and not locks.holds_any()
    lock = locks.lock(kv, {0: [4, 9], 2: [6]}, READ)
    cases = {(0, None): True, (0, 4): False, (4, 5): True, (5, 6): False, (5, 7): True}
    cases.update({(7, 9): False, (10, None): False, (9, 10): True, (6, 6): False})
    assert {r: locks.is_locked(kv, *r) for r in cases} == cases
    assert not locks.is_locked(Cache()) and locks.holds_any()
    empty = locks.lock(kv, {1: []}, READ)
    locks.unlock(lock)
    assert not locks.is_locked(kv), "a lock of no block locks the cache"
    assert not locks.holds_any(), "a lock of no block counts as holding one"
    assert not locks.defer_close(kv)
    locks.unlock(empty)


@pytest.mark.cpu_only
def test_a_freed_locked_cache_closes_once_at_its_last_unlock():
    locks, kv, free = registry(), Cache(), Cache()
    first = locks.lock(kv, {0: [0, 1]}, READ)
    second = locks.lock(kv, {1: [0]}, READ)
    assert not locks.defer_close(free), "an unlocked cache was kept for the manager to close"
    assert locks.defer_close(kv) and locks.defer_close(kv)
    assert locks.holds_open_caches() and kv.closes == 0
    with pytest.raises(ValueError, match="freed"):
        locks.lock(kv, {2: [0]}, READ)
    locks.unlock(first)
    assert kv.closes == 0 and locks.holds_open_caches(), "closed while a lock still holds it"
    assert HoldReason.LOCKED not in locks.held_page_counts().by_reason, "counted as locked"
    assert locks.holds_any(), "a freed cache's lock went unseen"
    locks.unlock(second)
    locks.unlock(second)
    locks.unlock(first)
    assert kv.closes == 1 and not locks.holds_open_caches() and not locks.holds_any()
    assert free.closes == 0, "the registry closed a cache it did not keep"


@pytest.mark.cpu_only
def test_shutdown_keeps_each_locked_cache_until_its_release():
    """At the manager's shutdown every mapped cache is offered; the locked ones stay open."""
    manager = Manager()
    locks = registry(manager)
    caches = [Cache(), Cache(), Cache()]
    lock = locks.lock(caches[1], {0: [3]}, WRITE)
    kept = [locks.defer_close(kv) for kv in caches]
    assert kept == [False, True, False] and locks.holds_open_caches()
    del manager  # the manager is gone after its shutdown; the close does not need it
    locks.unlock(lock)
    assert [kv.closes for kv in caches] == [0, 1, 0] and not locks.holds_open_caches()


@pytest.mark.cpu_only
def test_the_lender_refuses_a_reuse_reset_while_a_lock_keeps_a_freed_cache_open():
    """The lender's reset hook refuses only while a lock keeps a freed cache open."""
    locks, kv = registry(), Cache()
    lender = SimpleNamespace(_locks=locks)  # all the hook reads
    lock = locks.lock(kv, {0: [0]}, READ)
    _lender._ManagerLender._before_reset(lender)  # the request still holds its cache
    assert locks.defer_close(kv)
    with pytest.raises(RuntimeError, match="until its lease is released"):
        _lender._ManagerLender._before_reset(lender)
    locks.unlock(lock)
    assert kv.closes == 1
    _lender._ManagerLender._before_reset(lender)


@pytest.mark.cpu_only
def test_a_state_row_is_locked_whatever_the_range_and_written_past_any_commit():
    """A recurrent layer group's row 0 is locked for any range and written past any commit."""
    locks = registry(recurrent=(False, True, False))
    kv = Cache(committed_tokens=2 * TPB)
    writer = locks.lock(kv, {1: [0]}, WRITE)
    with pytest.raises(ValueError, match="committed whole blocks"):
        locks.lock(kv, {0: [0]}, WRITE)
    assert locks.is_locked(kv, 7, 9) and locks.is_locked(kv)
    with pytest.raises(ValueError, match="locked for write"):
        locks.lock(kv, {1: [0]}, READ)
    assert locked(locks) == {1: 1}
    locks.unlock(writer)
    assert not locks.is_locked(kv, 7, 9)
    with pytest.raises(ValueError, match="committed whole blocks"):
        registry().lock(Cache(committed_tokens=2 * TPB), {1: [0]}, WRITE)


@pytest.mark.cpu_only
def test_deferred_close_counts_include_every_page_of_each_cache_until_close():
    """A kept cache counts every page of each beam until it closes, a live locked one none."""
    locks = registry()
    kept = Cache(
        beams=[{0: [3, 4, 5, -1], 1: [7, 8], 2: [9]}, {0: [3, 4, 10, -1], 1: [7, 8], 2: [9]}]
    )
    other = Cache(beams=[{0: [3, 11], 1: [12]}])
    live = Cache(beams=[{0: [20, 21], 1: [22], 2: [23]}])
    assert deducted(locks) == {}
    first = locks.lock(kept, {0: [0]}, READ)
    second = locks.lock(kept, {1: [1]}, READ)
    sharing = locks.lock(other, {0: [0]}, READ)
    locks.lock(live, {0: [0, 1], 1: [0]}, READ)
    assert deducted(locks) == {}, "a live cache's pages counted as retained"
    assert locks.defer_close(kept) and locks.defer_close(other)
    # kept: layer group 0 holds 3, 4, 5 and beam 1's 10, layer group 2 holds 9; other: 3 and 11.
    assert deducted(locks) == {0: 4 + 1 + 2, 1: 2 + 1}
    locks.unlock(first)
    assert deducted(locks) == {0: 7, 1: 3}, "an unlocked page of a kept cache freed"
    locks.unlock(sharing)
    assert other.closes == 1 and deducted(locks) == {0: 5, 1: 2}
    locks.unlock(second)
    assert kept.closes == 1 and deducted(locks) == {}


@pytest.mark.cpu_only
def test_the_lowest_locked_block_past_each_floor_spans_every_lock():
    locks, kv = registry(), Cache()
    assert locks.lowest_locked_blocks(kv, (0, 0, 0)) == {}
    first = locks.lock(kv, {0: [0, 5, 9], 2: [1]}, READ)
    locks.lock(kv, {0: [3, 4], 1: []}, READ)
    locks.lock(Cache(), {1: [0]}, READ)  # another cache's lock counts for that cache alone
    assert locks.lowest_locked_blocks(kv, (0, 0, 0)) == {0: 0, 2: 1}
    # Blocks below a floor, a window's sinks, are passed over; a group with none past it is left out.
    assert locks.lowest_locked_blocks(kv, (1, 0, 2)) == {0: 3}
    assert locks.lowest_locked_blocks(kv, (6, 0, 0)) == {0: 9, 2: 1}
    locks.unlock(first)
    assert locks.lowest_locked_blocks(kv, (0, 0, 0)) == {0: 3}


@pytest.mark.cpu_only
def test_each_unlock_has_the_manager_retry_a_held_commit_before_the_close():
    """Every unlock retries a held commit; the last closes the cache after it, even if it raises."""
    manager = Manager()
    locks, kv = registry(manager), Cache()
    first, last = locks.lock(kv, {0: [0]}, READ), locks.lock(kv, {2: [1]}, READ)
    assert locks.defer_close(kv)
    locks.unlock(first)
    assert manager.retried == [(kv, 0)] and kv.closes == 0
    locks.unlock(last)
    assert manager.retried == [(kv, 0), (kv, 0)], "the cache closed before the retry"
    assert kv.closes == 1
    locks.unlock(last)
    assert len(manager.retried) == 2, "a repeated unlock retried again"

    def failing(kv):
        raise RuntimeError("retry failed")

    manager._commit_held_back = failing
    other = Cache()
    lock = locks.lock(other, {0: [0]}, READ)
    assert locks.defer_close(other)
    with pytest.raises(RuntimeError, match="retry failed"):
        locks.unlock(lock)
    assert other.closes == 1 and not locks.is_locked(other), "a failed retry kept the cache"


@pytest.mark.cpu_only
def test_an_unlock_retries_a_commit_only_where_the_manager_holds_one_back():
    """An unlock retries only a commit the manager's record holds back, the last unlock too."""
    manager = Manager(held={})
    locks, kv, other = registry(manager), Cache(), Cache()
    first, last = locks.lock(kv, {0: [0]}, READ), locks.lock(kv, {2: [1]}, READ)
    unheld = locks.lock(other, {0: [0]}, READ)
    locks.unlock(unheld)
    assert manager.retried == [], "an unlock retried a commit nobody held back"
    manager._held_commits[id(kv)] = (kv, None)
    assert locks.defer_close(kv)
    locks.unlock(first)
    assert manager.retried == [(kv, 0)] and kv.closes == 0, "a held commit was not retried"
    manager._held_commits[id(kv)] = (kv, None)
    locks.unlock(last)
    assert manager.retried == [(kv, 0), (kv, 0)], "the last unlock closed before the retry"
    assert kv.closes == 1 and manager._held_commits == {}


@pytest.mark.cpu_only
def test_a_registry_needs_its_manager_to_lock():
    manager = Manager()
    locks = registry(manager)
    del manager
    with pytest.raises(RuntimeError, match="gone"):
        locks.lock(Cache(), {0: [0]}, READ, ready=Event())


@pytest.mark.cpu_only
def test_the_module_reads_the_manager_and_caches_through_the_facade_alone():
    """The registry reaches a manager or a cache only through ``_manager``."""
    names = {"layer_group_pool_indices", "committed_blocks", "close_cache", "commit_held_back"}
    source = Path(_page_locks.__file__).read_text()
    assert {n for n in names if f"_manager.{n}(" in source} == names
    assert all(callable(getattr(_manager, name)) for name in names)


@pytest.mark.cpu_only
@pytest.mark.parametrize("state_slots", [(17, 18), (17, -1)], ids=["every_beam", "one_beam"])
def test_retained_counts_include_active_state_and_scratch_slots_across_beams(state_slots):
    """A kept cache counts its scratch slots and each beam's state slot, -1 for none."""
    locks = registry(recurrent=(False, True, False))
    kv = Cache(beams=[{0: [3, 4]}, {0: [3, 10]}])
    kv.get_ssm_block_base_index = lambda lg, beam_id=0: state_slots[beam_id]
    kv.get_scratch_desc = lambda lg: SimpleNamespace(slot_ids=[12, 13]) if lg == 0 else None
    held = locks.lock(kv, {0: [0], 1: [0]}, READ)
    assert locks.defer_close(kv)
    assert deducted(locks) == {0: 5, 1: sum(slot >= 0 for slot in state_slots)}
    locks.unlock(held)
    assert kv.closes == 1 and deducted(locks) == {}


@pytest.mark.cpu_only
def test_a_failed_deferred_close_keeps_its_cache_and_capacity_registered():
    locks, kv = registry(), Cache(beams=[{0: [3]}])
    held = locks.lock(kv, {0: [0]}, READ)
    assert locks.defer_close(kv)

    def fail_close():
        raise RuntimeError("injected close failure")

    kv.close = fail_close
    with pytest.raises(RuntimeError, match="injected close failure"):
        locks.unlock(held)
    assert held.released and not locks.is_locked(kv)
    assert locks.holds_open_caches() and deducted(locks) == {0: 1}
    with pytest.raises(ValueError, match="freed"):
        locks.lock(kv, {0: [0]}, READ)


def scanned(locks):
    """What a full scan of the registry finds: per pool group, the distinct locked blocks of each
    open cache, and every page each cache closing at its last unlock holds."""
    locked, deferred = {}, {}
    for cache in locks._caches.values():
        if cache.close_at_last_unlock:
            for lg, group in enumerate(locks._pool_group_of):
                pages = _manager.held_pages(cache.kv, lg, recurrent=lg in locks._recurrent)
                if pages:
                    deferred[group] = deferred.get(group, 0) + pages
            continue
        by_layer_group = {}
        for lock in cache.locks:
            for lg, ordinals in lock._blocks.items():
                by_layer_group.setdefault(lg, []).append(ordinals)
        for lg, arrays in by_layer_group.items():
            pages = len(np.unique(np.concatenate(arrays)))
            if pages:
                group = locks._pool_group_of[lg]
                locked[group] = locked.get(group, 0) + pages
    return locked, deferred


@pytest.mark.cpu_only
@pytest.mark.parametrize("counted", [False, True], ids=["never_counted", "counted"])
def test_a_cache_whose_last_lock_ended_is_not_kept_for_a_later_count(counted):
    """A cache whose last lock ended is no longer referenced by the registry, counted or not."""
    manager = Manager(held={})  # no commit held back, so none retried that would keep the cache
    locks = registry(manager)
    kv = Cache()
    gone = weakref.ref(kv)
    lock = locks.lock(kv, {0: [0]}, READ)
    if counted:
        assert locked(locks) == {0: 1}
    locks.unlock(lock)
    del kv, lock
    gc.collect()
    assert gone() is None, "the registry kept a cache whose last lock ended"
    assert locked(locks) == {}


@pytest.mark.cpu_only
@pytest.mark.parametrize("seed", range(6))
def test_held_page_counts_follow_each_change_as_a_full_scan_finds_them(seed):
    """Random locks, unlocks and frees: the counts equal a full scan and change only with it."""
    rng = random.Random(seed)
    locks = registry(recurrent=(False, True, False))
    caches = []
    for i in range(4):
        cache = Cache(beams=[{lg: [rng.randrange(-1, 30) for _ in range(6)] for lg in range(3)}])
        cache.get_ssm_block_base_index = lambda lg, beam_id=0, slot=40 + i: slot
        caches.append(cache)
    held, changes = [], 0
    value = locks.held_page_counts()
    for _ in range(120):
        roll, cache = rng.random(), rng.choice(caches)
        if roll < 0.5:
            blocks = {lg: rng.sample(range(8), rng.randint(0, 3)) for lg in (0, 2)}
            if rng.random() < 0.3:
                blocks[1] = [0]
            try:
                held.append(locks.lock(cache, blocks, rng.choice([READ, READ, WRITE])))
            except ValueError:
                pass
        elif roll < 0.85 and held:
            locks.unlock(held.pop(rng.randrange(len(held))))
        else:
            locks.defer_close(cache)
        new = locks.held_page_counts()
        locked, deferred = scanned(locks)
        reasons = {HoldReason.LOCKED: locked, HoldReason.DEFERRED_CLOSE: deferred}
        assert {r: dict(c) for r, c in new.by_reason.items()} == {
            r: c for r, c in reasons.items() if c
        }
        assert dict(new.deduct) == deferred
        unreclaimable = dict(locked)
        for group, pages in deferred.items():
            unreclaimable[group] = unreclaimable.get(group, 0) + pages
        assert dict(new.unreclaimable) == unreclaimable
        if (new.by_reason, new.deduct) == (value.by_reason, value.deduct):
            assert new is value, "an unchanged count gave a new value"
        else:
            assert new.version > value.version, "a changed count kept its version"
            changes += 1
        value = new
    assert changes >= 20, changes


# -- the generation update over cache models, its methods compiled from the package's sources ---

_CACHE = Path(_page_locks.__file__).resolve().parents[1]
_TPB = 4
_STATES = SimpleNamespace(GENERATION_COMPLETE="complete", CONTEXT_INIT="context_init")
_TYPES = SimpleNamespace(CROSS="cross", SELF="self")
_LIFECYCLE_HOOKS = (
    "on_free defer_close before_reset on_reset on_shutdown keeps_pools on_caches_closed"
)


def _source(path: Path, qualified: str) -> ast.AST:
    """The source node of a module-level or class-level definition."""
    node = ast.parse(path.read_text())
    for name in qualified.split("."):
        node = next(child for child in node.body if getattr(child, "name", None) == name)
    return node


def _reads_holding(test: ast.AST) -> bool:
    """Whether an ``if`` test reads the lock state's ``holding``."""
    return any(isinstance(n, ast.Attribute) and n.attr == "holding" for n in ast.walk(test))


def _method(path: Path, qualified: str, namespace: dict, eager: bool = False) -> Callable:
    """A source definition compiled into ``namespace``. ``eager`` opens both lender gates of
    ``update_resources``: every active cache asks the lent history and every resize notifies."""
    node = copy.deepcopy(_source(path, qualified))
    if eager:
        changed = Counter()
        for child in ast.walk(node):
            if isinstance(child, ast.If) and _reads_holding(child.test):
                child.test = ast.Name("has_lender", ast.Load())
                changed["holding gate"] += 1
            elif isinstance(child, ast.If) and any(
                isinstance(n, ast.Name) and n.id == "num_blocks_before"
                for n in ast.walk(child.test)
            ):
                child.test = ast.parse("has_lender and new_capacity is not None", mode="eval").body
                changed["shrink gate"] += 1
        assert changed == {"holding gate": 1, "shrink gate": 1}, (qualified, dict(changed))
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.Module(body=[future, node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[node.name]


@pytest.fixture(scope="module")
def sources() -> SimpleNamespace:
    namespace = {
        "np": np,
        "SimpleNamespace": SimpleNamespace,
        "traceback": traceback,
        "logger": SimpleNamespace(
            error=lambda *args: pytest.fail(f"unexpected lender error: {args}"),
            warning=lambda *args: None,
        ),
        "CacheTypeCpp": _TYPES,
        "LlmRequestState": _STATES,
        "StreamingKVCacheEventManager": type("StreamingEvents", (), {}),
        "_update_kv_cache_draft_token_location": lambda *args, **kwargs: None,
        "_manager": SimpleNamespace(num_blocks=lambda kv: kv.num_blocks),
        "_ranges": SimpleNamespace(
            _last_history_keeping=lambda manager, layout, lg, block: (
                (block + 1) * _TPB + layout.windows[lg] - 2
            )
        ),
    }
    hooks = SimpleNamespace()
    for name in (
        _LIFECYCLE_HOOKS + " on_shrink lent_history holds_page_locks generation_lock_state"
    ).split():
        setattr(hooks, name, _method(_CACHE / "_lender_hooks.py", name, namespace))
    namespace["_lender_hooks"] = hooks
    methods = SimpleNamespace(hooks=hooks, namespace=namespace)
    for name in ("is_locked", "holds_any", "lowest_locked_blocks"):
        locks = _CACHE / "sharing/_page_locks.py"
        setattr(methods, name, _method(locks, f"PageLocks.{name}", namespace))
    for name in ("_history_cap", "_on_shrink", "_page_lock_state"):
        lender = _CACHE / "sharing/_lender.py"
        setattr(methods, name, _method(lender, f"_ManagerLender.{name}", namespace))
    ledger = _CACHE / "sharing/_fetch_ledger.py"
    namespace["_writable_mask"] = _method(ledger, "_writable_mask", namespace)
    methods.invalidate = _method(
        ledger, "_FetchLedger._invalidate_delivered_blocks_past_capacity", namespace
    )
    manager = _CACHE / "kv_cache_manager_v2.py"
    for name in ("update_resources", "free_resources", "reset_reuse_state", "shutdown"):
        setattr(methods, name, _method(manager, f"KVCacheManagerV2.{name}", namespace))
    update = "KVCacheManagerV2.update_resources"
    methods.eager_update = _method(manager, update, dict(namespace), eager=True)
    return methods


class _Locks:
    """The real lock queries over admissions and unlocks modeled between generation updates."""

    def __init__(self, sources: SimpleNamespace) -> None:
        self._caches = {}
        self.locked_caches = self._caches
        self.caps_history = True
        self.unreusable = {}
        self.version = self.holding = 0
        self._recurrent = frozenset()
        for name in ("is_locked", "holds_any", "lowest_locked_blocks"):
            setattr(self, name, getattr(sources, name).__get__(self))

    def add(self, kv: _Cache, block: int) -> None:
        entry = self._caches.setdefault(id(kv), SimpleNamespace(kv=kv, locks=[]))
        entry.locks.append(SimpleNamespace(_blocks={0: np.array([block], dtype=np.int64)}))
        self.version += 1
        self.holding += 1

    def clear(self) -> None:
        if self._caches:
            self._caches.clear()
            self.version += 1
            self.holding = 0

    def locked(self, kv: _Cache, block: int) -> bool:
        return self.is_locked(kv, block, block + 1)


class _Cache:
    """Native resize rules relevant here: block rounding, fresh regrowth and stale windows."""

    def __init__(self, rid: int, window: int | None, locks: _Locks) -> None:
        self.id = rid
        self.capacity = self.history_length = self.serial = 0
        self.pages = []
        self.window, self.locks = window, locks
        self.is_active = True
        self.fail_next = False
        self.trace, self.violations = [], []
        self.drops = 0

    @property
    def num_blocks(self) -> int:
        return len(self.pages)

    def stale_end(self, history: int) -> int:
        return 0 if self.window is None else max(0, (history + 1 - self.window) // _TPB)

    def resize(self, capacity: int | None, history: int | None) -> bool:
        if self.fail_next:
            self.fail_next = False
            return False
        end = self.capacity if capacity is None else capacity
        past = self.history_length if history is None else history
        assert self.history_length <= past <= end
        before, after = self.num_blocks, -(-end // _TPB)
        if after < before:
            self.drops += 1
            for block in range(after, before):
                if self.locks.locked(self, block):
                    self.violations.append(("tail", block))
            del self.pages[after:]
        while len(self.pages) < after:
            self.pages.append(("fresh", self.serial))
            self.serial += 1
        for block in range(min(self.stale_end(past), after)):
            if self.pages[block] is not None and self.locks.locked(self, block):
                self.violations.append(("window", block))
            self.pages[block] = None
        self.capacity, self.history_length = end, past
        self.trace.append((capacity, history, tuple(self.pages)))
        return True


class _Ledger:
    def __init__(self, layout: SimpleNamespace, sources: SimpleNamespace) -> None:
        self._layout = layout
        self._deliveries = {}
        self._invalidate_delivered_blocks_past_capacity = sources.invalidate.__get__(self)

    def _deliveries_of(self, rid: int) -> SimpleNamespace | None:
        return self._deliveries.get(rid)

    def _drop_deliveries(self, rid: int) -> None:
        self._deliveries.pop(rid, None)

    def deliver(self, kv: _Cache, prompt: int) -> None:
        top = min(kv.num_blocks, -(-prompt // _TPB))
        delivered = self._deliveries.setdefault(
            kv.id,
            SimpleNamespace(
                kv=kv,
                delivered_block_masks=[np.zeros(0, dtype=bool)],
                end=0,
                partial_ends={},
                resume_bounds_memo=("memo",),
            ),
        )
        mask = delivered.delivered_block_masks[0]
        if len(mask) < top:
            mask = np.pad(mask, (0, top - len(mask)))
        for block in range(kv.stale_end(kv.history_length), top):
            mask[block] = True
            kv.pages[block] = ("fetched", kv.id, block)
        delivered.delivered_block_masks[0] = mask
        delivered.end = max(delivered.end, min(prompt, top * _TPB))
        if delivered.end % _TPB:
            delivered.partial_ends[(0, delivered.end // _TPB)] = delivered.end


class _Manager:
    def __init__(
        self,
        sources: SimpleNamespace,
        window: int | None,
        *,
        attached: bool = True,
        draft: bool = False,
        managed: bool = False,
        helix: bool = False,
    ) -> None:
        self.sources, self.counts = sources, Counter()
        self.kv_cache_map, self._allocated_draft_lens = {}, {}
        self._inflight_draft_lens = {}
        self.tokens_per_block = _TPB
        self.is_draft = draft
        self.kv_compression_manages_history, self._has_cp_helix = managed, helix
        self.kv_cache_type, self._kv_reserve_draft_tokens = _TYPES.SELF, 3
        self.locks = _Locks(sources)
        self.locks.caps_history = window is not None
        layout = SimpleNamespace(windows=[window], sink_blocks=[0], tokens_per_block=_TPB)
        lender = SimpleNamespace(
            _layout=layout,
            _locks=self.locks,
            _any_window=window is not None,
            _shrink_epochs={},
            _writes_at_end={},
            _lose_end_resume=lambda kv_cache: None,
        )
        lender._manager_ref = lambda: self
        lender._serving_manager = lambda: self
        lender._page_lock_state = sources._page_lock_state.__get__(lender)
        lender._history_cap = sources._history_cap.__get__(lender)
        lender._on_shrink = sources._on_shrink.__get__(lender)
        lender._is_locked = lambda kv, begin=0, end=None: (
            self.locks.holds_any() if kv is None else self.locks.is_locked(kv, begin, end)
        )
        self.ledger = lender._ledger = _Ledger(layout, sources)
        self._sharing = lender if attached else None

    def _holds_page_locks(self, rid: int | None) -> bool:
        return self.sources.hooks.holds_page_locks(self, rid)

    def _lent_history(self, kv: _Cache, history: int) -> int:
        return self.sources.hooks.lent_history(self, kv, history)

    def _after_shrink(self, rid: int, kv: _Cache) -> None:
        self.counts["shrink"] += 1
        self.sources.hooks.on_shrink(self, rid, kv)

    def try_commit_generation_blocks(self, request: SimpleNamespace) -> None:
        """Commits nothing, as a manager with a lender attached does."""


def _request(rid: int, tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        py_request_id=rid,
        max_beam_num_tokens=tokens,
        py_rewind_len=0,
        py_num_accepted_draft_tokens=0,
        state="generation",
    )


def _observe(manager: _Manager) -> tuple:
    caches = tuple(
        (rid, kv.capacity, kv.history_length, *map(tuple, (kv.pages, kv.trace, kv.violations)))
        for rid, kv in manager.kv_cache_map.items()
    )
    records = tuple(
        (
            rid,
            tuple(tuple(mask) for mask in d.delivered_block_masks),
            d.end,
            tuple(sorted(d.partial_ends.items())),
            d.resume_bounds_memo,
        )
        for rid, d in manager.ledger._deliveries.items()
    )
    return caches, records, tuple(sorted(manager._allocated_draft_lens.items()))


def _run(
    sources: SimpleNamespace,
    update: Callable,
    seed: int,
    window: int | None,
    *,
    attached: bool = True,
    draft: bool = False,
    managed: bool = False,
) -> tuple:
    rng = random.Random(seed)
    manager = _Manager(sources, window, attached=attached, draft=draft, managed=managed)
    requests, prompts, observations = [], {}, []
    for rid in range(4):
        prompt = prompts[rid] = rng.randint(5, 23)
        kv = manager.kv_cache_map[rid] = _Cache(rid, window, manager.locks)
        kv.resize(prompt, 0 if managed else prompt)
        requests.append(_request(rid, prompt))
        if attached:
            manager.ledger.deliver(kv, prompt)
    batch = SimpleNamespace(generation_requests=requests)
    for _ in range(12):
        # No admission or unlock occurs within update_resources.
        manager.locks.clear()
        for req in requests:
            kv = manager.kv_cache_map[req.py_request_id]
            runtime = rng.choice([0, 0, 1, 3, 5])
            accepted = rng.randint(0, runtime)
            allocated = max(runtime, manager._kv_reserve_draft_tokens) if draft else runtime
            kv.resize(kv.capacity + 1 + allocated, None)
            req.py_rewind_len, req.py_num_accepted_draft_tokens = runtime - accepted, accepted
            req.max_beam_num_tokens += 1 + accepted
            if rng.random() < 0.5:
                manager._allocated_draft_lens[req.py_request_id] = allocated
            if attached and rng.random() < 0.3:
                manager.ledger.deliver(kv, prompts[kv.id])
            low = kv.stale_end(kv.history_length)
            top = min(kv.num_blocks, prompts[kv.id] // _TPB)
            if attached and low < top and rng.random() < 0.3:
                manager.locks.add(kv, rng.randrange(low, top))
        update(manager, batch)
        observations.append(_observe(manager))
    assert not any(kv.violations for kv in manager.kv_cache_map.values())
    return observations, manager.counts, sum(kv.drops for kv in manager.kv_cache_map.values())


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "window,draft,managed,attached",
    [
        (None, False, False, True),
        (8, False, False, True),
        (8, True, False, True),
        (None, False, True, True),
        (None, False, False, False),
    ],
)
def test_generation_matches_eager_reference(sources, window, draft, managed, attached):
    """Schedules of reserves, fallback widths, fetch masks, windows, locks and regrowth."""
    totals = Counter()
    for seed in range(40):
        options = dict(draft=draft, managed=managed, attached=attached)
        reference, old_counts, _ = _run(sources, sources.eager_update, seed, window, **options)
        observed, new_counts, drops = _run(
            sources, sources.update_resources, seed, window, **options
        )
        assert observed == reference, (seed, options)
        assert new_counts["shrink"] == (drops if attached else 0)
        totals.update(new_counts)
        totals["old_shrink"] += old_counts["shrink"]
    if attached:
        assert 0 < totals["shrink"] < totals["old_shrink"]
    else:
        assert totals["shrink"] == 0


@pytest.mark.cpu_only
@pytest.mark.parametrize("batch_size", [64, 256])
def test_unlocked_batch_preserves_generation_results(sources, batch_size):
    managers = [_Manager(sources, None) for _ in range(2)]
    for manager, update in zip(managers, (sources.eager_update, sources.update_resources)):
        requests = []
        for rid in range(batch_size):
            kv = manager.kv_cache_map[rid] = _Cache(rid, None, manager.locks)
            kv.resize(8, 0)
            requests.append(_request(rid, 8))
        update(manager, SimpleNamespace(generation_requests=requests))
    assert _observe(managers[0]) == _observe(managers[1])
    assert managers[0].counts["shrink"] == batch_size
    assert managers[1].counts["shrink"] == 0


@pytest.mark.cpu_only
def test_in_flight_draft_width_sets_the_rewind(sources):
    """The oldest in-flight width (2), not the next step's (4), sets the rewind and is consumed."""
    for update in (sources.eager_update, sources.update_resources):
        manager = _Manager(sources, None)
        kv = manager.kv_cache_map[1] = _Cache(1, None, manager.locks)
        kv.resize(20, 12)
        manager._allocated_draft_lens[1] = 4
        manager._inflight_draft_lens[1] = [2]
        req = _request(1, 14)
        req.py_num_accepted_draft_tokens = 1
        update(manager, SimpleNamespace(generation_requests=[req]))
        assert (kv.capacity, kv.history_length) == (19, 13), update
        assert manager._inflight_draft_lens == {} and manager._allocated_draft_lens == {}, update


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "capacity,rewind,state,notifies",
    [
        (10, 1, "generation", 0),
        (9, 1, "generation", 1),
        (17, 4, "generation", 1),
        (17, 0, "generation", 0),
        (17, 4, _STATES.GENERATION_COMPLETE, 0),
        (17, 4, _STATES.CONTEXT_INIT, 0),
    ],
)
def test_shrink_notification_tracks_actual_blocks(sources, capacity, rewind, state, notifies):
    manager = _Manager(sources, None, managed=True)
    kv = manager.kv_cache_map[1] = _Cache(1, None, manager.locks)
    kv.resize(capacity, 0)
    manager.ledger.deliver(kv, capacity)
    req = _request(1, 16)
    req.py_rewind_len, req.state = rewind, state
    manager._allocated_draft_lens[1] = rewind
    sources.update_resources(manager, SimpleNamespace(generation_requests=[req]))
    assert manager.counts["shrink"] == notifies == kv.drops
    assert not manager._allocated_draft_lens


@pytest.mark.cpu_only
def test_failed_resize_keeps_allocation_and_skips_notification(sources):
    manager = _Manager(sources, None)
    kv = manager.kv_cache_map[1] = _Cache(1, None, manager.locks)
    kv.resize(17, 0)
    kv.fail_next = True
    manager._allocated_draft_lens[1] = 4
    with pytest.raises(ValueError, match="generation update"):
        sources.update_resources(manager, SimpleNamespace(generation_requests=[_request(1, 12)]))
    assert kv.capacity == 17 and kv.history_length == 0
    assert manager._allocated_draft_lens == {1: 4}
    assert manager.counts["shrink"] == 0


def _changing_batch(sources, update, *, unlock=False):
    manager = _Manager(sources, 4)
    requests = []
    for rid in (1, 2):
        kv = manager.kv_cache_map[rid] = _Cache(rid, 4, manager.locks)
        kv.resize(20, 0)
        requests.append(_request(rid, 16))
    if unlock:
        manager.locks.add(manager.kv_cache_map[2], 0)

    def changing_requests():
        yield requests[0]
        if unlock:
            manager.locks.clear()
        else:
            manager.locks.add(manager.kv_cache_map[2], 0)
        yield requests[1]

    update(manager, SimpleNamespace(generation_requests=changing_requests()))
    return manager


@pytest.mark.cpu_only
@pytest.mark.parametrize("unlock", [False, True], ids=["lock", "unlock"])
def test_a_lock_or_unlock_during_the_batch_counts_for_the_requests_after_it(sources, unlock):
    """A lock or an unlock between two requests of a batch counts for the second."""
    reference = _changing_batch(sources, sources.eager_update, unlock=unlock)
    current = _changing_batch(sources, sources.update_resources, unlock=unlock)
    assert _observe(current) == _observe(reference)
    if not unlock:
        assert current.kv_cache_map[2].history_length == 6


@pytest.mark.cpu_only
def test_missing_and_suspended_caches_keep_their_allocations(sources):
    manager = _Manager(sources, None)
    kv = manager.kv_cache_map[1] = _Cache(1, None, manager.locks)
    kv.resize(17, 0)
    kv.is_active = False
    manager._allocated_draft_lens.update({1: 4, 2: 4})
    sources.update_resources(
        manager, SimpleNamespace(generation_requests=[_request(1, 12), _request(2, 12)])
    )
    assert manager._allocated_draft_lens == {1: 4, 2: 4}
    assert kv.trace == [(17, 0, tuple(kv.pages))]


def _lifecycle(*, deferred=False, refuse=False, runtime_error=False):
    """Minimal lifecycle dependencies; assertions in callbacks observe the actual call order."""
    events = []
    manager = SimpleNamespace(
        _pending_preemption={},
        kv_connector_manager=None,
        is_draft=False,
        conversation_manager=None,
        _waiter_snapshot_points=None,
        _allocated_draft_lens={},
        _inflight_draft_lens={},
        _request_stats_enabled_ids=set(),
        _fresh_pages_filled={},
        _disagg_receive_ready={},
        _on_allocation_stream=set(),
        _allocation_ready={},
        _early_freed_index_requests=set(),
        _held_commits={},
        sparse_metadata_batch=None,
        _stream=None,
        event_manager=None,
        index_mapper=SimpleNamespace(remove_sequence=lambda rid: None),
    )
    kv = SimpleNamespace(close=lambda: events.append("close"), discard_pending_stats=lambda: None)
    manager.kv_cache_map = {1: kv}
    manager._detach_index_slot = lambda cache: events.append("detach")

    def on_free(rid, cache):
        assert rid not in manager.kv_cache_map
        assert "close" not in events
        events.append("free")

    def before_reset():
        events.append("before_reset")
        if refuse:
            raise RuntimeError("locked cache")

    def shutdown():
        events.append("runtime_shutdown")
        if runtime_error:
            raise RuntimeError("runtime shutdown failed")

    manager._sharing = SimpleNamespace(
        _on_free=on_free,
        _defer_close=lambda cache: (events.append("defer_close"), deferred)[1],
        _before_reset=before_reset,
        _on_reset=lambda: events.append("reset"),
        _on_shutdown=lambda: events.append("shutdown"),
        _keeps_pools=lambda: (events.append("keeps_pools"), deferred)[1],
        _on_caches_closed=lambda: events.append("caches_closed"),
    )
    manager.impl = SimpleNamespace(
        clear_stats_excluded=lambda rid: None,
        clear_reusable_blocks=lambda: events.append("clear_reuse"),
        shutdown=shutdown,
    )
    return manager, events


# Per double, the source functions that run with it as their first argument.
_DOUBLED = {
    "manager": (
        ("kv_cache_manager_v2.py", "KVCacheManagerV2.update_resources"),
        ("_lender_hooks.py", "generation_lock_state"),
        ("_lender_hooks.py", "holds_page_locks"),
        ("_lender_hooks.py", "lent_history"),
        ("_lender_hooks.py", "on_shrink"),
    ),
    "lifecycle": tuple(
        ("kv_cache_manager_v2.py", f"KVCacheManagerV2.{name}")
        for name in ("free_resources", "reset_reuse_state", "shutdown")
    )
    + tuple(("_lender_hooks.py", name) for name in _LIFECYCLE_HOOKS.split()),
    "lender": tuple(
        ("sharing/_lender.py", f"_ManagerLender.{name}")
        for name in ("_history_cap", "_on_shrink", "_page_lock_state")
    ),
    "locks": tuple(
        ("sharing/_page_locks.py", f"PageLocks.{name}")
        for name in ("is_locked", "holds_any", "lowest_locked_blocks")
    ),
    "ledger": (
        ("sharing/_fetch_ledger.py", "_FetchLedger._invalidate_delivered_blocks_past_capacity"),
    ),
}


def _first_argument_reads(node: ast.FunctionDef) -> set[str]:
    """Attributes a function loads from its first argument: ``self``, or a hook's manager."""
    first = node.args.args[0].arg
    return {
        n.attr
        for n in ast.walk(node)
        if isinstance(n, ast.Attribute)
        and isinstance(n.ctx, ast.Load)
        and isinstance(n.value, ast.Name)
        and n.value.id == first
    }


@pytest.mark.cpu_only
def test_doubles_carry_every_attribute_their_source_functions_read(sources):
    """A source function that reads a new attribute fails here, naming the double and attribute."""
    manager = _Manager(sources, 8)
    doubles = {
        "manager": manager,
        "lifecycle": _lifecycle()[0],
        "lender": manager._sharing,
        "locks": manager.locks,
        "ledger": manager.ledger,
    }
    missing = [
        f"{kind} double lacks {attr}, which {qualified} reads"
        for kind, functions in _DOUBLED.items()
        for path, qualified in functions
        for attr in sorted(_first_argument_reads(_source(_CACHE / path, qualified)))
        if not hasattr(doubles[kind], attr)
    ]
    assert not missing, missing


@pytest.mark.cpu_only
@pytest.mark.parametrize("deferred", [False, True])
def test_free_notifies_before_close_or_deferred_detach(sources, deferred):
    manager, events = _lifecycle(deferred=deferred)
    sources.free_resources(manager, _request(1, 8))
    assert events == ["free", "defer_close", "detach" if deferred else "close"]
    assert manager.kv_cache_map == {}


@pytest.mark.cpu_only
@pytest.mark.parametrize("refuse", [False, True])
def test_reset_preflight_precedes_any_reset(sources, refuse):
    manager, events = _lifecycle(refuse=refuse)
    if refuse:
        with pytest.raises(RuntimeError, match="locked cache"):
            sources.reset_reuse_state(manager)
        assert events == ["before_reset"]
    else:
        sources.reset_reuse_state(manager)
        assert events == ["before_reset", "clear_reuse", "reset"]


@pytest.mark.cpu_only
@pytest.mark.parametrize("deferred,runtime_error", [(False, False), (True, False), (False, True)])
def test_shutdown_notifies_caches_closed_only_after_runtime_returns(
    sources, deferred, runtime_error
):
    manager, events = _lifecycle(deferred=deferred, runtime_error=runtime_error)
    if runtime_error:
        with pytest.raises(RuntimeError, match="runtime shutdown failed"):
            sources.shutdown(manager)
    else:
        sources.shutdown(manager)
    if deferred:
        assert events == ["shutdown", "defer_close", "detach", "keeps_pools"]
    else:
        expected = ["shutdown", "defer_close", "close", "keeps_pools", "runtime_shutdown"]
        assert events == expected + ([] if runtime_error else ["caches_closed"])


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "hook,adapter,args",
    [
        ("_defer_close", "defer_close", (object(),)),
        ("_keeps_pools", "keeps_pools", ()),
        ("_history_cap", "lent_history", (SimpleNamespace(history_length=0), 8)),
        ("_is_locked", "holds_page_locks", (None,)),
        ("_page_lock_state", "generation_lock_state", ()),
    ],
)
def test_query_errors_propagate_to_the_manager(sources, hook, adapter, args):
    def failed_query(*_query_args):
        raise ValueError("query failed")

    queries = {hook: failed_query}
    if hook != "_page_lock_state":
        # A lock holds a block of every cache and can cap a history, so the adapters ask their
        # query.
        every = type("EveryCache", (dict,), {"__contains__": lambda self, key: True})()
        queries.setdefault(
            "_page_lock_state",
            lambda: SimpleNamespace(
                holding=1, locked_caches=every, caps_history=True, unreusable={}
            ),
        )
    manager = SimpleNamespace(_sharing=SimpleNamespace(**queries))
    with pytest.raises(ValueError, match="query failed"):
        getattr(sources.hooks, adapter)(manager, *args)


def _real_registry(sources, manager):
    """The registry compiled from its source; only manager primitives and events are modeled."""
    primitives = SimpleNamespace(
        layer_group_pool_indices=lambda owner: [0],
        committed_blocks=lambda kv, count: [0] * count,
        commit_held_back=lambda owner, kv: owner._commit_held_back(kv),
        # As the facade answers: a manager keeping no record of its held commits retries each.
        has_held_commit=lambda owner, kv: (
            getattr(owner, "_held_commits", None) is None or id(kv) in owner._held_commits
        ),
        is_closed=lambda kv: False,
        close_cache=lambda kv: None,
    )
    namespace = dict(
        sources.namespace,
        enum=enum,
        numbers=numbers,
        types=types,
        _manager=primitives,
        _NO_BLOCKS=np.zeros(0, dtype=np.int64),
    )
    _method(_CACHE / "sharing/_types.py", "_integer", namespace)
    for name in ("LockMode", "PageLock", "_CacheLocks", "_ordinals", "PageLocks"):
        _method(_CACHE / "sharing/_page_locks.py", name, namespace)
    registry = namespace["PageLocks"](weakref.ref(manager))
    registry.locked = lambda kv, block: registry.is_locked(kv, block, block + 1)
    manager._commit_held_back = lambda kv: None
    manager.locks = manager._sharing._locks = registry
    return registry, namespace["LockMode"]


def _real_generation_change(sources, update, change):
    manager = _Manager(sources, 4)
    registry, mode = _real_registry(sources, manager)
    first = manager.kv_cache_map[1] = _Cache(1, 4, registry)
    second = manager.kv_cache_map[2] = _Cache(2, 4, registry)
    first.resize(17, 0)
    second.resize(20, 0)
    req_one, req_two = _request(1, 12), _request(2, 16)
    req_one.py_rewind_len = 4
    held = registry.lock(first, {0: [0]}, mode.READ) if change in ("retry", "unlock") else None
    admitted = []

    def admit():
        if not admitted:
            admitted.append(registry.lock(second, {0: [0]}, mode.READ))

    if change == "retry":
        manager._commit_held_back = lambda kv: admit() if kv is first else None
    if change in ("resize", "retry", "unlock"):
        original_resize = first.resize

        def resize(capacity, history_length):
            result = original_resize(capacity, history_length)
            if change == "resize":
                admit()
            else:
                registry.unlock(held)
            return result

        first.resize = resize
    else:
        original_shrink = manager._after_shrink

        def after_shrink(rid, kv):
            original_shrink(rid, kv)
            if kv is first:
                admit()

        manager._after_shrink = after_shrink
    update(manager, SimpleNamespace(generation_requests=[req_one, req_two]))
    return manager


@pytest.mark.cpu_only
@pytest.mark.parametrize("change", ["resize", "after_shrink", "retry", "unlock"])
def test_production_registry_reentry_preserves_generation_results(sources, change):
    reference = _real_generation_change(sources, sources.eager_update, change)
    current = _real_generation_change(sources, sources.update_resources, change)
    assert _observe(current) == _observe(reference)
    if change != "unlock":
        assert current.kv_cache_map[2].history_length == 6
        assert not any(kv.violations for kv in current.kv_cache_map.values())


@pytest.mark.cpu_only
def test_production_registry_refusal_idempotence_and_retry_exception(sources):
    manager = _Manager(sources, None)
    registry, mode = _real_registry(sources, manager)
    kv = _Cache(1, None, registry)
    state = sources.hooks.generation_lock_state(manager)
    before = state.version
    assert state.holding == 0
    held = registry.lock(kv, {0: [0]}, mode.READ)
    assert state.version != before and state.holding == 1
    stable = state.version
    assert registry.refusal(kv, {0: [0]}, mode.WRITE) is not None
    with pytest.raises(ValueError, match="conflicts"):
        registry.lock(kv, {0: [0]}, mode.WRITE)
    with pytest.raises(ValueError, match="only level=None"):
        registry.lock(kv, {0: [0]}, mode.READ, level=0)
    assert state.version == stable
    callback_observed = []

    def failed_retry(cache):
        callback_observed.append((state.version != stable, registry.is_locked(cache)))
        raise RuntimeError("retry failed")

    manager._commit_held_back = failed_retry
    with pytest.raises(RuntimeError, match="retry failed"):
        registry.unlock(held)
    assert callback_observed == [(True, False)]
    assert held.released and not registry.holds_any() and state.holding == 0
    stable = state.version
    registry.unlock(held)
    assert state.version == stable and callback_observed == [(True, False)]


@pytest.mark.cpu_only
def test_window_model_and_history_cap_follow_native_token_windows(sources):
    for window in (1, 3, 4, 5, 8, 13):
        manager = _Manager(sources, window)
        cache = _Cache(1, window, manager.locks)
        layout = manager._sharing._layout

        def retained(history):
            return {token // _TPB for token in list(range(history + 1))[-window:]}

        for history in range(64):
            assert cache.stale_end(history) == min(retained(history))
        namespace = dict(
            sources.namespace,
            _manager=SimpleNamespace(
                stale_range=lambda owner, lg, history: (0, min(retained(history))),
                _MAX_TOKENS=2**31 - 1,
            ),
        )
        _method(_CACHE / "sharing/_ranges.py", "_stale_range", namespace)
        actual_cap = _method(_CACHE / "sharing/_ranges.py", "_last_history_keeping", namespace)
        model_cap = sources.namespace["_ranges"]._last_history_keeping
        for block in range(5):
            last = max(
                history
                for history in range((block + 2) * _TPB + window)
                if block in retained(history)
            )
            assert actual_cap(manager, layout, 0, block) == last
            assert model_cap(manager, layout, 0, block) == last


# -- the manager's lock hooks over real managers ------------------------------------------------

PROMPT = list(range(1000, 1097))  # three whole blocks and one token
OTHER_PROMPT = list(range(5000, 5097))
ALIGNED = list(range(2000, 2000 + 4 * TPB))  # four whole blocks
SOURCE, OTHER, THIRD = 1, 2, 3


def attach(mgr):
    return attach_lender(mgr, scope=b"page-lock-hooks", staging=StagingOptions(3 * TPB))


def warnings_of(monkeypatch) -> List[str]:
    """The warnings the manager module logs from now on."""
    seen: List[str] = []
    warn = kv_cache_manager_v2.logger.warning

    def spy(message, *args, **kwargs):
        seen.append(str(message))
        return warn(message, *args, **kwargs)

    monkeypatch.setattr(kv_cache_manager_v2.logger, "warning", spy)
    return seen


def compute_context(mgr, request, tokens=None):
    """The request's next context chunk, ``tokens`` or the rest, computed and its update run."""
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

    request.context_chunk_size = tokens or request.context_remaining_length
    request.move_to_next_context_chunk()
    batch = ScheduledRequests()
    batch.append_context_request(request)
    mgr.update_context_resources(batch)


def computed_under_a_lock(kit, mgr, lender, request_id):
    """A request computed whole while a lock holds its first block, its last commit held back:
    the request, its cache and the lock."""
    request = kit.admitted(mgr, request_id, PROMPT)
    kv = kit.kv(mgr, request)
    assert mgr.resize_context(request, request.context_remaining_length)
    lock = lender._locks.lock(kv, {0: [0]}, READ)
    compute_context(mgr, request)
    assert kv.num_committed_tokens == 0, "inconclusive: the lock held no commit back"
    return request, kv, lock


def matched(kit, mgr, request_id, tokens=PROMPT) -> int:
    """The prompt tokens a new request with ``tokens`` finds in the reuse tree; freed again."""
    probe = kit.admitted(mgr, request_id, tokens)
    tokens = kit.kv(mgr, probe).num_committed_tokens
    mgr.free_resources(probe)
    return tokens


@gpu
def test_a_locked_cache_refuses_suspension_until_its_unlock(kit, real_manager):
    with real_manager() as mgr:
        request = kit.published(mgr, SOURCE, PROMPT)
        lender = attach(mgr)
        kv = kit.kv(mgr, request)
        lock = lender._locks.lock(kv, {0: [1]}, READ)
        with pytest.raises(RuntimeError, match="cannot be suspended"):
            mgr.suspend_request(request)
        assert kv.is_active, "the refused suspension changed the cache"
        lender._locks.unlock(lock)
        mgr.suspend_request(request)
        assert not kv.is_active
        assert mgr.resume_request(request)


@gpu
def test_a_freed_locked_cache_stays_open_until_its_last_unlock(kit, real_manager):
    """A freed locked cache keeps its pages until its last unlock, whose close spares its row."""
    with real_manager() as mgr:
        request = kit.published(mgr, SOURCE, PROMPT)
        lender = attach(mgr)
        kv = kit.kv(mgr, request)
        index = mgr.index_mapper.get_index(SOURCE)
        pages = [kit.pages(kv, lg) for lg in range(kit.num_layer_groups(mgr))]
        locks = [lender._locks.lock(kv, {0: [0, 1]}, READ), lender._locks.lock(kv, {0: [2]}, READ)]
        assert locked(lender) == {kit.pool_group_of(mgr)[0]: 3}
        mgr.free_resources(request)
        assert kit.kv(mgr, request) is None and not kit.closed(kv), "the free closed a locked cache"
        assert [kit.pages(kv, lg) for lg in range(kit.num_layer_groups(mgr))] == pages
        assert lender._keeps_pools()
        taker = kit.admitted(mgr, OTHER, OTHER_PROMPT)
        assert mgr.index_mapper.get_index(OTHER) == index, "inconclusive: the row went elsewhere"
        assert mgr.resize_context(taker, taker.context_remaining_length)
        row = kit.index_row(mgr, taker).values
        assert -1 not in row[: len(pages[0])], "the taker's row shows no pages of its own"
        lender._locks.unlock(locks[0])
        assert not kit.closed(kv), "closed while a lock still holds it"
        lender._locks.unlock(locks[1])
        lender._locks.unlock(locks[1])
        assert kit.closed(kv) and not lender._keeps_pools()
        assert kit.index_row(mgr, taker).values == row, "the close wrote into the taker's row"


@gpu
def test_a_shutdown_keeps_a_locked_cache_and_the_pools_until_its_unlock(
    kit, real_manager, monkeypatch
):
    seen = warnings_of(monkeypatch)
    with real_manager() as mgr:
        request = kit.published(mgr, SOURCE, PROMPT)
        other = kit.published(mgr, OTHER, OTHER_PROMPT)
        lender = attach(mgr)
        kv, other_kv = kit.kv(mgr, request), kit.kv(mgr, other)
        lock = lender._locks.lock(kv, {0: [0]}, READ)
        mgr.shutdown()  # the runtime's own shutdown raises while a cache is open
        assert kit.closed(other_kv) and not kit.closed(kv)
        assert any("pools stay allocated" in message for message in seen), seen
        assert lender._keeps_pools() and mgr.kv_cache_map == {}
        lender._locks.unlock(lock)
        assert kit.closed(kv) and not lender._keeps_pools()
    # The block's exit shuts the manager down again; with no cache open the runtime's shutdown runs.


@gpu
def test_a_commit_that_reaches_a_locked_block_waits_for_its_unlock(kit, real_manager, monkeypatch):
    """A commit reaching a locked block is skipped, with a warning, until the unlock freeing it."""
    seen = warnings_of(monkeypatch)
    with real_manager() as mgr:
        lender = attach(mgr)
        request = kit.admitted(mgr, SOURCE, PROMPT)
        kv = kit.kv(mgr, request)
        assert mgr.resize_context(request, request.context_remaining_length)
        compute_context(mgr, request, 2 * TPB)
        assert kv.num_committed_tokens == 2 * TPB
        # Locks on committed blocks leave a commit of later blocks alone.
        on_committed = lender._locks.lock(kv, {0: [0, 1]}, READ)
        on_tail = lender._locks.lock(kv, {0: [3]}, READ)
        compute_context(mgr, request)
        assert request.context_remaining_length == 0
        assert kv.num_committed_tokens == 2 * TPB, "a commit reached a locked block"
        assert any("skipped while a lender locks" in message for message in seen), seen
        lender._locks.unlock(on_tail)
        assert kv.num_committed_tokens == request.prompt_len, "the unlock left the commit undone"
        lender._locks.unlock(on_committed)


@gpu
def test_a_commit_held_back_reaches_the_reuse_tree_at_the_last_unlock(kit, real_manager):
    with real_manager() as mgr:
        lender = attach(mgr)
        request, kv, lock = computed_under_a_lock(kit, mgr, lender, SOURCE)
        assert matched(kit, mgr, OTHER) == 0, "the held-back blocks reached the reuse tree"
        assert not mgr.try_commit_blocks(request)
        assert kv.num_committed_tokens == 0, "a commit reached a locked block"
        lender._locks.unlock(lock)
        assert kv.num_committed_tokens == request.prompt_len
        assert matched(kit, mgr, THIRD) == 3 * TPB, "the prompt missed the reuse tree"


@gpu
@pytest.mark.parametrize("freed", [False, True], ids=["active", "freed_before_unlock"])
def test_a_held_last_commit_saves_a_conversation_plan_only_for_an_active_request(
    kit, real_manager, freed
):
    """The unlock saves an active conversation's plan; a freed request commits before its close."""
    with real_manager(block_reuse_config={"policy": "per_conversation"}) as mgr:
        lender = attach(mgr)
        request = kit.make_request(SOURCE, PROMPT)
        request.py_conversation_params = SimpleNamespace(conversation_id="talk")
        assert mgr.prepare_context(request)
        kv = kit.kv(mgr, request)
        assert mgr.resize_context(request, request.context_remaining_length)
        lock = lender._locks.lock(kv, {0: [0]}, READ)
        compute_context(mgr, request)  # saving the plan here would raise
        state = mgr.conversation_manager._conversation_states["talk"]
        assert kv.num_committed_tokens == 0 and not state.planned_drop_handles
        if freed:
            mgr.free_resources(request)
            assert "talk" not in mgr.conversation_manager._conversation_states
            assert not kit.closed(kv), "the free closed a locked cache"
        lender._locks.unlock(lock)
        if freed:
            assert kit.closed(kv), "the last unlock left the freed cache open"
            assert "talk" not in mgr.conversation_manager._conversation_states
            assert matched(kit, mgr, OTHER) == 3 * TPB, "the freed request's prompt missed the tree"
        else:
            assert kv.num_committed_tokens == request.prompt_len
            assert len(state.planned_drop_handles) == 1, "the unlock saved no drop plan"
        assert state.current_request_id is None


@gpu
@pytest.mark.parametrize("path", ["fast", "rules"])
@pytest.mark.parametrize("order", ["returned_then_unlocked", "unlocked_then_returned", "abandoned"])
def test_a_fetch_over_rows_a_held_commit_covers_keeps_them_out_of_the_reuse_tree(
    kit, real_manager, monkeypatch, path, order
):
    """A fetch over held-back rows, however it ends and is lent, leaves none in the reuse tree."""
    fast_lends = []
    fast = _lender._ManagerLender._write_every_row_fast

    def counted(self, *args, **kwargs):
        lease = fast(self, *args, **kwargs)
        fast_lends.append(lease is not None)
        return lease

    monkeypatch.setattr(_lender._ManagerLender, "_write_every_row_fast", counted)
    with real_manager() as mgr:
        lender = attach(mgr)
        if path == "rules":
            lender._lends_fast = lambda: False
        request, kv, lock = computed_under_a_lock(kit, mgr, lender, SOURCE)
        lease = lender.lend_write(request, TPB, len(PROMPT), **kit.DEVICE_ALL)
        assert lease.failure is None, lease.failure
        assert any(fast_lends) == (path == "fast"), f"inconclusive: fast lends {fast_lends}"
        view = kit.ready_view(lease, mgr)
        arrived = [np.isin(rows.page_ordinals, (1,)) for rows in view.groups]
        if order == "unlocked_then_returned":
            lender._locks.unlock(lock)
        if order == "abandoned":
            lease.release()
        else:
            lender.return_writes([(lease, arrived, True)])
        if order != "unlocked_then_returned":
            lender._locks.unlock(lock)
        assert kv.num_committed_tokens <= TPB, "the retried commit took rows the fetch overwrote"
        assert matched(kit, mgr, OTHER) <= TPB, "rows the fetch overwrote reached the reuse tree"


@gpu
@pytest.mark.parametrize("path", ["fast", "rules"])
@pytest.mark.parametrize("finish", ["mark_before_release", "release_before_mark"])
def test_a_commit_held_after_lending_is_dropped_before_a_write_unlock(
    kit, real_manager, monkeypatch, path, finish
):
    """A commit held after the fetch was lent must not enter reuse when that fetch unlocks."""
    fast_lends = []
    fast = _lender._ManagerLender._write_every_row_fast

    def counted(self, *args, **kwargs):
        lease = fast(self, *args, **kwargs)
        fast_lends.append(lease is not None)
        return lease

    monkeypatch.setattr(_lender._ManagerLender, "_write_every_row_fast", counted)
    with real_manager() as mgr:
        lender = attach(mgr)
        if path == "rules":
            lender._lends_fast = lambda: False
        request, kv, lock = computed_under_a_lock(kit, mgr, lender, SOURCE)
        lease = lender.lend_write(request, TPB, len(PROMPT), **kit.DEVICE_ALL)
        assert lease.failure is None, lease.failure
        assert any(fast_lends) == (path == "fast"), f"inconclusive: fast lends {fast_lends}"
        assert kit.ready_view(lease, mgr) is not None
        assert id(kv) not in mgr._held_commits, "the lend left the earlier commit held"
        assert not mgr.try_commit_blocks(request)
        assert id(kv) in mgr._held_commits, "inconclusive: no commit was held after the lend"

        if finish == "mark_before_release":
            lender.return_writes([(lease, ALL_ARRIVED, True)])
            lease.release()
        else:
            lease.release()
        lender._locks.unlock(lock)

        assert id(kv) not in mgr._held_commits
        assert kv.num_committed_tokens <= TPB, "the unlock committed rows the fetch could overwrite"
        assert matched(kit, mgr, OTHER) <= TPB, "rows the fetch could overwrite reached reuse"


@gpu
@pytest.mark.parametrize("lock_count", [1, 2])
def test_a_fetch_from_where_the_computed_rows_end_keeps_the_held_commit(
    kit, real_manager, lock_count
):
    """A fetch from where the computed tokens end leaves the held commit to the last unlock."""
    with real_manager() as mgr:
        lender = attach(mgr)
        request = kit.admitted(mgr, SOURCE, PROMPT)
        kv = kit.kv(mgr, request)
        assert mgr.resize_context(request, request.context_remaining_length)
        locks = [lender._locks.lock(kv, {0: [0]}, READ) for _ in range(lock_count)]
        compute_context(mgr, request, 2 * TPB)
        assert kv.num_committed_tokens == 0, "inconclusive: the lock held no commit back"
        lease = lender.lend_write(request, 2 * TPB, len(PROMPT), **kit.DEVICE_ALL)
        assert lease.failure is None, lease.failure
        kit.ready_view(lease, mgr)
        lender.return_writes([(lease, ALL_ARRIVED, True)])
        request.context_current_position = len(PROMPT)
        for lock in locks[:-1]:
            lender._locks.unlock(lock)
            assert kv.num_committed_tokens == 0
            assert mgr._held_commits[id(kv)][2:] == (2 * TPB, True)
        lender._locks.unlock(locks[-1])
        assert kv.num_committed_tokens == 2 * TPB, "a fetch past the held rows dropped their commit"


@pytest.mark.cpu_only
@pytest.mark.parametrize("manager", ["paged", "hybrid"])
@pytest.mark.parametrize("case", ["committed", "held_back", "no_reuse"])
def test_a_context_end_saves_its_drop_plan_unless_a_lock_holds_its_commit_back(manager, case):
    """A context's end saves its drop plan unless a lock holds its commit back."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
        BlockReusePolicy,
        CacheTypeCpp,
        KVCacheManagerV2,
    )
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
        MambaHybridCacheManagerV2,
    )

    no_reuse = case == "no_reuse"
    plans = SimpleNamespace(saved=[])
    plans.save_drop_plan = lambda request, kv_cache: plans.saved.append(request.py_request_id)
    request = SimpleNamespace(
        py_request_id=SOURCE,
        is_dummy_request=no_reuse and manager == "paged",
        context_current_position=len(PROMPT),
        context_remaining_length=0,
        expect_snapshot_points=[],
    )
    stand_in = SimpleNamespace(
        kv_cache_map={SOURCE: SimpleNamespace(is_active=True, resize=lambda *args: True)},
        kv_cache_type=CacheTypeCpp.SELF,
        enable_block_reuse=True,
        _can_publish_block_reuse=True,
        _commit_branch_snapshots=False,
        is_draft=no_reuse and manager == "hybrid",
        block_reuse_policy=BlockReusePolicy.PER_CONVERSATION,
        conversation_manager=plans,
        _lent_history=lambda kv_cache, history: history,
        _history_lock_state=lambda: None,
        _resize_context_history=lambda req, kv_cache, history: None,
        try_commit_blocks=lambda req, kv_cache=None: case == "committed",
    )
    cls = KVCacheManagerV2 if manager == "paged" else MambaHybridCacheManagerV2
    cls.update_context_resources(stand_in, SimpleNamespace(context_requests=[request]))
    assert plans.saved == ([] if case == "held_back" else [SOURCE]), plans.saved


@gpu
def test_a_first_chunk_out_of_pages_keeps_its_locked_cache_active(kit, real_manager):
    """A first chunk out of pages is suspended unless a lease lends its matched prefix in place."""
    # Index slots for a one-block request per page, so other requests can take every page.
    with real_manager(max_tokens=kit.POOL_TOKENS, host_cache_size=0, max_batch_size=64) as mgr:
        kit.published(mgr, SOURCE, PROMPT)
        lender = attach(mgr)
        # The source's three whole blocks, then two blocks of its own: its first chunk needs pages.
        target = kit.make_request(OTHER, PROMPT[: 3 * TPB] + list(range(7000, 7000 + 2 * TPB)))
        assert mgr.prepare_context(target) and target.is_first_context_chunk
        kv = kit.kv(mgr, target)
        assert kv.num_committed_tokens == 3 * TPB, "inconclusive: no prefix matched"
        held = [page for page in kit.pages(kv, 0) if page >= 0]
        assert len(held) < target.prompt_len // TPB, "inconclusive: the chunk needs no new page"
        lease = lender.lend_read(target, 0, 3 * TPB, **kit.DEVICE_NAMED)
        assert lease.failure is None, lease.failure
        others = kit.Requests(mgr)
        while others.allocate(1, chunk=1):
            pass
        # A pool group with fewer free or evictable pages than the chunk's new blocks refuses it.
        needed = target.prompt_len // TPB - len(held)
        tiers = [str(tier) for tier in mgr.impl.cache_tier_list]
        gpu_pages = [
            (int(s.total), int(s.free), int(s.evictable))
            for s in mgr.impl.get_storage_statistics(0)
        ]
        assert len(tiers) == 1 and any(free + evict < needed for _, free, evict in gpu_pages), (
            f"inconclusive: the chunk could still get its {needed} new pages; tiers {tiers}, GPU "
            f"(total, free, evictable) per pool group {gpu_pages}, {len(others.held)} others"
        )
        assert not mgr.resize_context(target, target.context_remaining_length)
        assert kv.is_active, "the failed resize suspended a cache a lease lends"
        lease.release()
        others.free()


@gpu
def test_a_reverted_context_chunk_keeps_its_locked_cache_active(kit, real_manager):
    """A reverted context allocation keeps its cache active while a lease lends its blocks."""
    with real_manager(enable_block_reuse=False) as mgr:
        lender = attach(mgr)
        request = kit.admitted(mgr, OTHER, OTHER_PROMPT)
        assert mgr.resize_context(request, 2 * TPB)
        compute_context(mgr, request, 2 * TPB)
        kv = kit.kv(mgr, request)
        lease = lender.lend_read(request, 0, 2 * TPB, **kit.DEVICE_ALL)
        assert lease.failure is None, lease.failure
        assert mgr.resize_context(request, TPB) and kv.capacity > 2 * TPB
        assert mgr.revert_allocate_context(request)
        assert kv.is_active and kv.capacity == 2 * TPB
        lease.release()


@gpu
@pytest.mark.parametrize("hybrid", [False, True], ids=["paged", "hybrid"])
def test_no_context_step_commit_or_stop_makes_an_event_for_a_lender(
    kit, real_manager, hybrid_manager, recorded, hybrid
):
    """A chunked context's steps, commits and stop make as many events with a lender as without."""
    made = []
    for lender in (False, True):
        with hybrid_manager() if hybrid else real_manager() as mgr:
            if lender:
                attach(mgr)
            before = len(recorded)
            request = kit.admitted(mgr, SOURCE, PROMPT)
            assert mgr.resize_context(request, request.context_remaining_length)
            compute_context(mgr, request, 2 * TPB)
            compute_context(mgr, request)
            if not hybrid:
                assert kit.kv(mgr, request).num_committed_tokens == request.prompt_len
            made.append(len(recorded) - before)
            mgr.free_resources(request)
    assert made[0] == made[1], f"the lender made events: {made}"


@gpu
def test_a_read_in_place_lends_what_the_context_steps_computed(kit, real_manager):
    """A read past a held-back history lends up to the computed tokens and waits for its event."""
    with real_manager(windows=[2 * TPB, _manager._MAX_TOKENS], enable_block_reuse=False) as mgr:
        lender = attach(mgr)
        short = lender._layout.windows.index(2 * TPB)
        request = kit.admitted(mgr, OTHER, OTHER_PROMPT)
        kv = kit.kv(mgr, request)
        assert mgr.resize_context(request, request.context_remaining_length)
        held = lender._locks.lock(kv, {short: [0]}, READ)
        compute_context(mgr, request, 3 * TPB)
        computed = request.context_current_position
        assert computed == 3 * TPB and kv.history_length < computed, (
            f"inconclusive: the history {kv.history_length} reached the context position"
        )

        def lend(end):
            ready = kit.stream_event(mgr)
            return lender.lend_read(request, 0, end, **{**kit.DEVICE_ALL, "ready_event": ready})

        lease = lend(computed)
        assert lease.failure is None, lease.failure
        lent = {rows.layer_group_index: len(rows) for rows in kit.ready_view(lease, mgr).groups}
        assert lent[short] == 0 and sum(lent.values()) == 3, lent
        lease.release()
        past = lend(computed + 1)
        assert "past the" in (past.failure or ""), past.failure
        past.release()
        inside = lend(kv.history_length - 1)
        assert "ends at a whole block" in (inside.failure or ""), inside.failure
        inside.release()
        lender._locks.unlock(held)
        compute_context(mgr, request)
        whole = lend(len(OTHER_PROMPT))
        assert whole.failure is None, whole.failure
        lent = {rows.layer_group_index: len(rows) for rows in kit.ready_view(whole, mgr).groups}
        assert lent[short] > 0 and sum(lent.values()) > 4, lent
        whole.release()
        mgr._stream.synchronize()
        mgr.free_resources(request)


@gpu
@pytest.mark.parametrize("lender", [True, False], ids=["lender", "no_lender"])
def test_a_hybrid_manager_records_no_receive_event_after_its_slot_reset(
    kit, hybrid_manager, recorded, lender
):
    """A generation-init admission's resource preparation makes no event and leaves none."""
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestType
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

    with hybrid_manager() as mgr:
        if lender:
            attach(mgr)
        request = kit.make_request(
            SOURCE, PROMPT, llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY
        )
        assert request.is_disagg_generation_init_state
        assert mgr.prepare_disagg_gen_init(request)
        assert SOURCE in mgr._disagg_receive_ready
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [request]
        before = len(recorded)
        mgr.prepare_resources(batch)
        assert len(recorded) == before, "the slot reset made an event"
        assert mgr.take_disagg_receive_ready(request) is None, "an event is left to take"


@pytest.mark.cpu_only
def test_kda_suspend_checks_page_locks_before_changing_replay_state():
    """A borrowed KDA cache cannot lose replay bookkeeping during a rejected suspension."""
    from tensorrt_llm._torch.modules.kimi_kda.cache_manager import KimiK3HybridCacheManager

    manager = object.__new__(KimiK3HybridCacheManager)
    replay = SimpleNamespace(
        acceptance_pending=set(), unseeded_requests={SOURCE}, suspended_requests=set()
    )
    cache = SimpleNamespace(is_active=True)
    manager._kda_replay = replay
    manager.kv_cache_map = {SOURCE: cache}
    manager._request_id_to_state_index = {SOURCE: 0}

    def reject_suspend(candidate):
        assert candidate is cache
        raise RuntimeError("borrowed cache")

    manager._check_suspendable = reject_suspend
    with pytest.raises(RuntimeError, match="borrowed cache"):
        manager.suspend_request(SimpleNamespace(py_request_id=SOURCE))

    assert replay.unseeded_requests == {SOURCE}
    assert replay.suspended_requests == set()
    assert manager._request_id_to_state_index == {SOURCE: 0}


@pytest.mark.cpu_only
@pytest.mark.parametrize("locked", [False, True])
def test_disagg_resize_pressure_returns_false_and_preserves_a_locked_cache(kit, locked):
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestType

    request = kit.make_request(
        SOURCE, PROMPT, llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY
    )
    suspended = []
    kv = SimpleNamespace(
        capacity=TPB, resize=lambda *args: False, suspend=lambda: suspended.append(True)
    )

    def check_suspendable(cache):
        assert not locked, "ordinary allocation pressure must not try suspending a locked cache"

    manager = SimpleNamespace(
        prepare_context_cache=lambda req: 0,
        enable_block_reuse=False,
        kv_cache_map={SOURCE: kv},
        _has_cp_helix=False,
        num_extra_kv_tokens=0,
        _holds_page_locks=lambda request_id: locked,
        _check_suspendable=check_suspendable,
    )
    assert request.is_first_context_chunk
    assert not kv_cache_manager_v2.KVCacheManagerV2.prepare_disagg_gen_init(manager, request)
    assert manager.kv_cache_map[SOURCE] is kv
    assert suspended == ([] if locked else [True])


@gpu
@pytest.mark.parametrize("windows", [None, [2 * TPB, 4096]], ids=["full", "windowed"])
def test_a_context_step_asks_no_lock_query_while_no_lock_holds_a_block_of_its_cache(
    kit, real_manager, monkeypatch, windows
):
    """Only a locked cache's step asks about locks, and about its history only through a window."""
    asked = {"_is_locked": 0, "_history_cap": 0}
    for name in asked:
        query = getattr(_lender._ManagerLender, name)

        def counted(self, *args, _query=query, _name=name, **kwargs):
            asked[_name] += 1
            return _query(self, *args, **kwargs)

        monkeypatch.setattr(_lender._ManagerLender, name, counted)
    with real_manager(windows=windows) as mgr:
        lender = attach(mgr)
        source = kit.published(mgr, SOURCE, PROMPT)
        kv_cache = kit.kv(mgr, source)
        # Asked directly too: a manager with block reuse moves the history through its commits.
        assert mgr._lent_history(kv_cache, len(PROMPT)) == len(PROMPT)
        assert not mgr._commit_gate(kv_cache, source, len(PROMPT))
        assert asked == {"_is_locked": 0, "_history_cap": 0}, asked
        read = lender.lend_read(source, 0, 3 * TPB, **kit.DEVICE_ALL)
        assert read.failure is None and lender._locks.holding == 1
        other = kit.published(mgr, OTHER, OTHER_PROMPT)
        other_cache = kit.kv(mgr, other)
        assert mgr._lent_history(other_cache, len(OTHER_PROMPT)) == len(OTHER_PROMPT)
        assert not mgr._commit_gate(other_cache, other, len(OTHER_PROMPT))
        assert asked == {"_is_locked": 0, "_history_cap": 0}, ("asked about another cache", asked)
        mgr._commit_gate(kv_cache, source, len(PROMPT))
        mgr._lent_history(kv_cache, len(PROMPT))
        assert asked["_is_locked"] > 0, asked
        assert (asked["_history_cap"] > 0) == (windows is not None), asked
        read.release()
        before = dict(asked)
        third = kit.published(mgr, THIRD, PROMPT[:40])
        assert asked == before, (before, asked)
        mgr._stream.synchronize()
        for request in (source, other, third):
            mgr.free_resources(request)


@gpu
@pytest.mark.parametrize("windows", [None, [2 * TPB, 4096]], ids=["full", "windowed"])
def test_a_context_step_keeps_a_block_a_lock_holds_inside_its_window(kit, real_manager, windows):
    """A context step moves a history only as far as its window keeps a locked block."""
    with real_manager(windows=windows, enable_block_reuse=False) as mgr:
        lender = attach(mgr)
        state = mgr._history_lock_state()
        assert (state is lender._locks) == (windows is not None), state
        windowed = [lg for lg, window in enumerate(kit.windows(mgr)) if window is not None]
        assert bool(windowed) == (windows is not None), "inconclusive: the windows were not kept"
        lg = windowed[0] if windowed else 0
        request = kit.admitted(mgr, SOURCE, PROMPT)
        kv = kit.kv(mgr, request)
        assert mgr.resize_context(request, request.context_remaining_length)
        lock = lender._locks.lock(kv, {lg: [0]}, READ)
        compute_context(mgr, request)
        cap = lender._history_cap(kv)
        if windows is None:
            assert cap is None and kv.history_length == len(PROMPT), (cap, kv.history_length)
        else:
            assert cap is not None and cap < len(PROMPT), "inconclusive: the window keeps block 0"
            assert kv.history_length == cap, (kv.history_length, cap)
        lender._locks.unlock(lock)
        mgr._stream.synchronize()
        mgr.free_resources(request)


@gpu
@pytest.mark.parametrize("caps", [True, False], ids=["a_lock_can_cap", "no_lock_can_cap"])
def test_a_hybrid_context_step_moves_a_history_only_as_far_as_the_lender_lets_it(
    kit, hybrid_manager, monkeypatch, caps
):
    """A hybrid context step asks the lender's cap only where its lock state says a lock can cap."""
    asked = []

    def lent_history(kv_cache, history):
        asked.append(history)
        return min(history, TPB)

    with hybrid_manager() as mgr:
        attach(mgr)
        state = SimpleNamespace(holding=1, locked_caches={}, caps_history=True, unreusable={})
        monkeypatch.setattr(mgr, "_history_lock_state", lambda: state if caps else None)
        monkeypatch.setattr(mgr, "_lent_history", lent_history)
        request = kit.admitted(mgr, SOURCE, PROMPT)
        assert mgr.resize_context(request, request.context_remaining_length)
        compute_context(mgr, request, 2 * TPB)
        history = kit.kv(mgr, request).history_length
        if caps:
            assert asked == [2 * TPB] and history == TPB, (asked, history)
        else:
            assert asked == [] and history == 2 * TPB, (asked, history)
        mgr._stream.synchronize()
        mgr.free_resources(request)


def refuse_reuse(kit, lender, request, end, reusable=False):
    """A write lease of ``[0, end)`` in place, returned with every row and ``reusable``."""
    lease = lender.lend_write(request, 0, end, **kit.DEVICE_ALL)
    assert lease.poll() is not None, lease.failure
    lender.return_writes([(lease, ALL_ARRIVED, reusable)])


@gpu
@pytest.mark.parametrize(
    "path", ["context", "native_entry", "later_true", "retry", "retry_after_exit", "retired"]
)
def test_a_cache_kept_out_of_reuse_commits_nothing_on_any_path(kit, real_manager, path):
    """A ``reusable=False`` return keeps its cache out of the reuse tree on every commit path."""
    with real_manager(block_reuse_config={"policy": "per_conversation"}) as mgr:
        lender = attach(mgr)
        request = kit.make_request(SOURCE, ALIGNED)
        request.py_conversation_params = SimpleNamespace(conversation_id="talk")
        assert mgr.prepare_context(request)
        kv = kit.kv(mgr, request)
        kv.enable_swa_scratch_reuse = False
        state = mgr.conversation_manager._conversation_states["talk"]
        assert mgr.resize_context(request, request.context_remaining_length)
        again = None
        if path == "retired":
            lease = lender.lend_write(request, 0, len(ALIGNED), **kit.DEVICE_ALL)
            assert lease.poll() is not None, lease.failure
            mgr.free_resources(request)
            again = kit.make_request(SOURCE, ALIGNED)
            assert mgr.prepare_context(again)
            lender.return_writes([(lease, ALL_ARRIVED, False)])
            assert kit.closed(kv)
        elif path.startswith("retry"):
            lock = lender._locks.lock(kv, {0: [0]}, READ)
            compute_context(mgr, request)
            assert id(kv) in mgr._held_commits, "inconclusive: no commit held back"
            lease = lender.lend_write(request, 4 * TPB, 4 * TPB, **kit.DEVICE_ALL)
            assert lease.poll() is not None, lease.failure
            if path == "retry_after_exit":
                mgr.free_resources(request)
            lender.return_writes([(lease, ALL_ARRIVED, False)])
            lender._locks.unlock(lock)
        else:
            refuse_reuse(kit, lender, request, len(ALIGNED))
            if path == "later_true":
                refuse_reuse(kit, lender, request, len(ALIGNED), reusable=True)
            if path == "native_entry":
                request.context_chunk_size = request.context_remaining_length
                request.move_to_next_context_chunk()
                assert mgr.try_commit_blocks(request), "the commit did not stop"
            else:
                compute_context(mgr, request)
            if path != "native_entry":
                assert state.current_request_id is None, "a refused commit saved no plan"
            assert kv.num_committed_tokens == 0, "a cache kept out of reuse committed"
        assert matched(kit, mgr, OTHER, ALIGNED) == 0, "a cache kept out of reuse was matched"
        if again is None:
            if path != "retry_after_exit":
                mgr.free_resources(request)
            again = kit.make_request(SOURCE, ALIGNED)
            assert mgr.prepare_context(again)
        assert mgr.resize_context(again, again.context_remaining_length)
        compute_context(mgr, again)
        assert kit.kv(mgr, again).num_committed_tokens == len(ALIGNED), "a new cache stayed out"
        mgr.free_resources(again)


@gpu
def test_a_staged_refusal_keeps_a_held_commit_out_of_reuse_after_the_free(kit, real_manager):
    """A staged ``reusable=False`` return, which holds no lock, keeps a held commit out after a free."""
    with real_manager() as mgr:
        lender = attach(mgr)
        request = kit.admitted(mgr, SOURCE, ALIGNED)
        kv = kit.kv(mgr, request)
        assert mgr.resize_context(request, request.context_remaining_length)
        lock = lender._locks.lock(kv, {0: [0]}, READ)
        compute_context(mgr, request, 2 * TPB)
        assert id(kv) in mgr._held_commits, "inconclusive: no commit held back"
        staged = lender.lend_write(request, 2 * TPB, 3 * TPB, **kit.HOST_NAMED)
        view = None
        for _ in range(4):
            lender.progress()
            view = staged.poll()
            if view is not None:
                break
            mgr._stream.synchronize()
        assert view is not None, staged.failure
        lender.return_writes([(staged, ALL_ARRIVED, False)])
        mgr._stream.synchronize()
        lender.progress()
        assert id(kv) in mgr._held_commits, "inconclusive: the return retried the commit"
        mgr.free_resources(request)
        lender._locks.unlock(lock)
        assert kit.closed(kv)
        assert matched(kit, mgr, OTHER, ALIGNED) == 0, "a cache kept out of reuse was matched"


@gpu
@pytest.mark.parametrize("replacement_freed", [False, True])
def test_a_held_commit_of_a_freed_cache_commits_nothing_into_the_request_s_next_cache(
    kit, real_manager, replacement_freed
):
    """Replacing a cache invalidates its held commit, also after the replacement is freed."""
    with real_manager() as mgr:
        lender = attach(mgr)
        request = kit.admitted(mgr, SOURCE, ALIGNED)
        kv = kit.kv(mgr, request)
        assert mgr.resize_context(request, request.context_remaining_length)
        lock = lender._locks.lock(kv, {0: [0]}, READ)
        compute_context(mgr, request, 2 * TPB)
        assert id(kv) in mgr._held_commits, "inconclusive: no commit held back"
        mgr.free_resources(request)
        again = kit.admitted(mgr, SOURCE, ALIGNED)
        assert kit.kv(mgr, again).num_committed_tokens == 0, "inconclusive: the new cache reused"
        request.context_chunk_size = request.context_remaining_length
        request.move_to_next_context_chunk()
        if replacement_freed:
            mgr.free_resources(again)
        lender._locks.unlock(lock)
        assert kit.closed(kv)
        if not replacement_freed:
            assert kit.kv(mgr, again).num_committed_tokens == 0
            mgr.free_resources(again)
        assert matched(kit, mgr, OTHER, ALIGNED) == 0, "the retired cache entered the reuse tree"


@gpu
@pytest.mark.parametrize("path", ["context", "free"])
def test_a_hybrid_cache_kept_out_of_reuse_commits_no_snapshot(kit, hybrid_manager, path):
    """A hybrid cache a ``reusable=False`` return keeps out of reuse commits no snapshot."""
    with hybrid_manager() as mgr:
        lender = attach(mgr)
        request = kit.admitted(mgr, SOURCE, ALIGNED)
        assert mgr.resize_context(request, request.context_remaining_length)
        refuse_reuse(kit, lender, request, len(ALIGNED))
        request.expect_snapshot_points = [2 * TPB, len(ALIGNED)]
        kv = kit.kv(mgr, request)
        compute_context(mgr, request, 2 * TPB)
        if path == "context":
            compute_context(mgr, request)
        else:
            mgr.free_resources(request)
        assert kv.num_committed_tokens == 0, "a snapshot of a cache kept out of reuse committed"
        assert matched(kit, mgr, OTHER, ALIGNED) == 0
        if path == "context":
            mgr.free_resources(request)
