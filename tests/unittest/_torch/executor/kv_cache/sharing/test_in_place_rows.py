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
"""Rows lent in place: recurrent state, Helix pages, windowed fetches, index and parked rows."""

import itertools
import random
import weakref
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.pyexecutor.kv_cache import _lender_hooks
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import (
    ALL_ARRIVED,
    MemoryKind,
    RowSelection,
    StagingOptions,
    _fetch_ledger,
    _lender,
    _manager,
    _ranges,
    attach_lender,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing._identity import Identity
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing._layout import DevicePool, derive_layout
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing._slots import SlotQueue

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="allocates KV cache pools")

TPB = 32
PROMPT = 97  # three whole blocks and one token
TOKENS = list(range(1000, 1000 + PROMPT))
STATE_SLOT = 5
DEVICE, HOST = MemoryKind.DEVICE, MemoryKind.HOST
NAMED, ALL = RowSelection.NAMED, RowSelection.ALL
NO_NAMES = "layer groups [1] hold recurrent state, which no name covers"


class Manager:
    """A stand-in manager: its layer groups' pool groups, its caches by request."""

    _on_allocation_stream = frozenset()

    def _commit_held_back(self, kv):
        """No commit is held back here."""

    def __init__(self, pool_group_of):
        self.impl = SimpleNamespace(get_life_cycle_pool_group_indices=lambda: list(pool_group_of))
        self.kv_cache_map = {}
        self.kv_connector_manager = None


def layout(count=1, tpb=TPB, pages=16, **fields):
    """A stand-in layout of ``count`` full-attention layer groups, each on a pool of its own."""
    lay = SimpleNamespace(
        tokens_per_block=tpb,
        windows=(None,) * count,
        sink_blocks=(0,) * count,
        recurrent=(False,) * count,
        pool_groups=tuple(range(count)),
        pool_group_of=tuple(range(count)),
        num_layer_groups=count,
        named_groups=(True,) * count,
        cp=(0, 1),
        device_pools={g: (DevicePool(g, 0, (g + 1) << 30, 256, pages),) for g in range(count)},
    )
    vars(lay).update(fields)
    return lay


def hybrid_layout():
    """Layer group 0 of attention; layer group 1 of recurrent state, whose slots span two pools."""
    lay = layout(2, recurrent=(False, True))
    lay.device_pools[1] = (DevicePool(1, 0, 2 << 30, 4096, 8), DevicePool(1, 1, 3 << 30, 512, 8))
    return lay


def lender_over(manager, lay, no_names=NO_NAMES):
    count = lay.num_layer_groups
    ident = Identity(b"scope", bytes(range(32)), [(lg,) for lg in range(count)], [(1, 0)] * count)
    ref = weakref.ref(manager)
    regions = tuple(_lender._device_regions(manager, lay, ident))
    return _lender._ManagerLender(ref, lay, ident, regions, SlotQueue({}), ref, no_names=no_names)


def helix(rank):
    """A stand-in manager under Helix over two ranks, as rank ``rank`` sees it, and its lender."""
    manager = Manager((0,))
    manager.enable_block_reuse = manager.is_draft = False
    manager._has_cp_helix, manager._helix_cp_rank, manager._helix_cp_size = True, rank, 2
    lay = layout(tpb=2 * TPB, cp=(rank, 2))
    return manager, lender_over(manager, lay, no_names="under Helix")


def cache(blocks, pages=None, history=PROMPT, committed=0, tpb=TPB, groups=1, paged=None):
    """A stand-in active cache of ``blocks`` blocks on ``pages`` in the groups ``paged`` flags, its
    claim complete in each, its state page ``kv.state_slot``."""
    pages = list(range(blocks) if pages is None else pages)

    def pages_of(lg, valid_only=False):
        return pages if paged is None or paged[lg] else [-1] * blocks

    kv = SimpleNamespace(
        Status=SimpleNamespace(CLOSED="CLOSED"),
        status="OPEN",
        num_committed_tokens=committed,
        reuse_status=[SimpleNamespace(complete=True, endpoint=committed)] * groups,
        history_length=history,
        is_active=True,
        num_blocks=blocks,
        tokens_per_block=tpb,
        enable_swa_scratch_reuse=False,
        state_slot=STATE_SLOT,
        get_base_page_indices=pages_of,
        get_aggregated_page_indices=pages_of,
    )
    kv.get_ssm_block_base_index = lambda lg: kv.state_slot
    return kv


def hybrid_cache(blocks=4, committed=0, history=PROMPT):
    return cache(blocks, range(10, 10 + blocks), history, committed, groups=2, paged=(True, False))


class Event:
    """A stand-in CUDA event, complete at once; waiting on it fails the test."""

    def query(self):
        return True

    def synchronize(self):
        raise AssertionError("a lease waited on its event")


READY = Event()


@pytest.fixture
def floor_follows_history(monkeypatch):
    """The stand-in managers' resume floor follows the history."""
    monkeypatch.setattr(_ranges, "_floor_follows_history", lambda manager: True)


def return_write(lender, req, lease, arrived):
    """The return's bounds, which ``poll_resume_bounds`` gives too, memoized and computed afresh."""
    (bounds,) = lender.return_writes([(lease, arrived, True)])
    assert lender.poll_resume_bounds(req) == bounds, bounds
    delivered = lender._ledger._deliveries_of(int(req.py_request_id))
    if delivered is not None:
        delivered.resume_bounds_memo = None
        assert lender.poll_resume_bounds(req) == bounds, ("computed afresh", bounds)
    return bounds


def request(request_id=7, prompt_len=PROMPT):
    return SimpleNamespace(
        py_request_id=request_id,
        prompt_len=prompt_len,
        py_multimodal_data=None,
        context_current_position=0,
        context_remaining_length=prompt_len,
    )


# -- recurrent state -----------------------------------------------------------------------------


@pytest.mark.cpu_only
def test_a_state_group_lends_one_row_its_state_page_as_block_zero():
    manager = Manager((0, 1))
    lender = lender_over(manager, hybrid_layout())
    kv = hybrid_cache()
    entries = lender._block_entries_of_range(manager, kv, 0, PROMPT)
    assert [o.tolist() for o in entries.block_ordinals] == [[0, 1, 2, 3], [0]]
    assert [p.tolist() for p in entries.device_pages] == [[10, 11, 12, 13], [STATE_SLOT]]
    view = lender._in_place_view(entries, None, 0, 0)
    attention, state = view.groups
    assert (attention.region_indices, state.region_indices) == ((0,), (1, 2))
    assert state.page_ordinals.tolist() == [0] and state.slots.tolist() == [STATE_SLOT]
    assert state.object_keys is None
    kv.state_slot = -1
    assert lender._block_entries_of_range(manager, kv, 0, PROMPT).device_pages[1].tolist() == [-1]


@pytest.mark.cpu_only
def test_a_lender_naming_nothing_refuses_named_rows_and_lends_every_row_in_place():
    lender = lender_over(Manager((0, 1)), hybrid_layout())
    for memory in (DEVICE, HOST):
        with pytest.raises(ValueError, match="lends nothing by name, as layer groups"):
            lender._check_memory_and_rows(memory, NAMED)
    lender._check_memory_and_rows(DEVICE, ALL)
    with pytest.raises(ValueError, match="staging relays named whole blocks"):
        lender._check_memory_and_rows(HOST, ALL)
    named = lender_over(Manager((0, 1)), hybrid_layout(), no_names=None)
    named._check_memory_and_rows(DEVICE, NAMED)  # the check reads the lender's reason alone


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "history,end,lent",
    [
        (2 * TPB, 2 * TPB, False),
        (PROMPT, PROMPT, True),
        (PROMPT + 3, PROMPT + 3, False),
        (PROMPT, 2 * TPB, False),
    ],
    ids=["mid_context", "context_computed", "after_generation", "short_of_the_state"],
)
def test_a_read_lends_a_state_row_only_where_the_state_stands_at_the_prompt_end(
    floor_follows_history, history, end, lent
):
    """A read with a state row ends at the prompt's end; elsewhere it fails at the call."""
    manager = Manager((0, 1))
    lender = lender_over(manager, hybrid_layout())
    kv = manager.kv_cache_map[7] = hybrid_cache(blocks=5, history=history)
    lease = lender.lend_read(request(), 0, end, memory_kind=DEVICE, rows=ALL, ready_event=READY)
    if lent:
        assert lease.failure is None, lease.failure
    else:
        assert lease.failure is not None and "where the state stands" in lease.failure
        assert not lender._is_locked(kv), "a lease failed at the call locks blocks"
    lease.release()


@pytest.mark.cpu_only
def test_a_state_row_lock_holds_every_commit_and_takes_a_write_past_any_commit(
    floor_follows_history,
):
    manager = Manager((0, 1))
    lender = lender_over(manager, hybrid_layout())
    kv = hybrid_cache(committed=2 * TPB)  # two whole blocks committed
    entries = lender._block_entries_of_range(manager, kv, 2 * TPB, PROMPT)
    lease = lender._lend_in_place(
        manager, request(), kv, "write", entries, None, start=0, end=PROMPT, ready_event=READY
    )
    assert lease.failure is None, lease.failure
    assert lease._lock.blocks[1].tolist() == [0]
    # A commit of any blocks snapshots the state page, so the lock holds them all.
    assert lender._is_locked(kv, 100, 200) and lender._is_locked(kv, 0, 1)
    lease.release()
    assert not lender._is_locked(kv, 100, 200)


def deliver(lender, manager, kv, masks_of):
    """A fetch of every row of ``[0, PROMPT)`` returned with the rows ``masks_of`` gives."""
    committed = (kv.num_committed_tokens,) * 2
    fetch = lender._ledger._record_fetch(
        7, kv, 0, (0, 0), kv.history_length, committed, state_end=PROMPT, end=PROMPT, prompt=PROMPT
    )
    entries = lender._block_entries_of_range(manager, kv, 0, PROMPT)
    req = request()
    lease = lender._lend_in_place(
        manager, req, kv, "write", entries, None, fetch, start=0, end=PROMPT, ready_event=READY
    )
    view = lease.poll()
    assert view is not None, lease.failure
    return_write(lender, req, lease, masks_of(view))


@pytest.mark.cpu_only
@pytest.mark.parametrize("case", ["everything", "state_missing", "block_missing", "state_moved"])
def test_a_fetch_with_recurrent_state_resumes_where_the_state_stands_or_nowhere(
    floor_follows_history, case
):
    """Resume at the prompt's end with the state and every block below, else nowhere."""
    manager = Manager((0, 1))
    lender = lender_over(manager, hybrid_layout())
    kv = manager.kv_cache_map[7] = hybrid_cache()

    def masks_of(view):
        masks = view.row_masks(True)
        if case == "state_missing":
            masks[1][:] = False
        if case == "block_missing":
            masks[0][1] = False
        if case == "state_moved":
            kv.state_slot = STATE_SLOT + 1  # the state now lives on another page
        return masks

    deliver(lender, manager, kv, masks_of)
    got = lender.poll_resume_bounds(request())
    if case == "everything":
        assert tuple(got) == (PROMPT, PROMPT)
    else:
        assert got.max_position < got.min_position, got


@pytest.mark.cpu_only
@pytest.mark.parametrize("state_only", [True, False])
def test_replacing_a_delivered_state_revokes_resume_until_the_new_state_arrives(
    floor_follows_history, state_only
):
    manager = Manager((0, 1))
    lender = lender_over(manager, hybrid_layout())
    kv = manager.kv_cache_map[7] = hybrid_cache()
    req = request()
    deliver(lender, manager, kv, lambda view: view.row_masks(True))
    assert tuple(lender.poll_resume_bounds(req)) == (PROMPT, PROMPT)

    entries = lender._block_entries_of_range(manager, kv, 0, PROMPT)
    lease = SimpleNamespace(_request_id=7, _kv=kv, _block_entries=entries)
    arrived = [np.ones(len(ordinals), dtype=bool) for ordinals in entries.block_ordinals]
    if state_only:
        arrived[0][:] = False
    lender._ledger._invalidate_delivered_blocks(lease, arrived if state_only else None)
    bounds = lender.poll_resume_bounds(req)
    assert bounds.max_position < bounds.min_position

    fetch = _fetch_ledger._FetchRecord(0, kv, state_end=PROMPT, end=PROMPT)
    lender._ledger._record_delivered_blocks(7, fetch, entries, arrived)
    assert tuple(lender.poll_resume_bounds(req)) == (PROMPT, PROMPT)


@pytest.mark.cpu_only
@pytest.mark.parametrize("committed", [0, PROMPT])
@pytest.mark.parametrize("delivery", ["local", "complete", "missing", "overwritten"])
def test_a_read_requires_delivered_state_even_with_a_computed_attention_prefix(
    floor_follows_history, committed, delivery
):
    manager = Manager((0, 1))
    manager.enable_block_reuse = manager._can_publish_block_reuse = manager.is_draft = False
    lender = lender_over(manager, hybrid_layout())
    manager.kv_cache_map[7] = hybrid_cache(committed=committed)
    req = request()

    def fetch(complete):
        write = lender.lend_write(req, 0, PROMPT, memory_kind=DEVICE, rows=ALL, ready_event=READY)
        view = write.poll()
        assert view is not None, write.failure
        masks = view.row_masks(True)
        masks[1][:] = complete
        return_write(lender, req, write, masks)

    if delivery == "overwritten":
        fetch(True)
    if delivery != "local":
        fetch(delivery == "complete")
    read = lender.lend_read(req, 0, PROMPT, memory_kind=DEVICE, rows=ALL, ready_event=READY)
    try:
        if delivery in ("missing", "overwritten"):
            assert read.failure is not None and "recurrent state" in read.failure
            assert read.poll() is None
            bounds = lender.poll_resume_bounds(req)
            assert bounds.max_position < bounds.min_position
        else:
            assert read.poll() is not None, read.failure
    finally:
        read.release()


@pytest.mark.cpu_only
def test_a_committed_tail_check_reads_no_state_row():
    lender = lender_over(Manager((0, 1)), hybrid_layout())
    kv = hybrid_cache(committed=5)  # five tokens of block 0 committed: its tail block is 0
    lent = _lender._LentBlockEntries(
        [0, 1], [np.array([1], np.int64), np.array([0], np.int64)], [np.array([11]), np.array([5])]
    )
    assert not lender._covers_committed_tail(kv, lent, [np.array([True]), np.array([True])])


# -- Helix ---------------------------------------------------------------------------------------


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "end,ordinals",
    [(PROMPT, ([0, 2], [1, 3])), (90, ([0, 2], [1])), (64, ([0], [1]))],
    ids=["prompt", "inside_rank_0_page", "whole_block"],
)
def test_each_helix_rank_lends_its_pages_of_the_range_under_their_own_blocks(end, ordinals):
    """Page ``j`` of a block ``b`` of 64 tokens lies on rank ``j``, lent as block ``b * 2 + j``."""
    for rank in (0, 1):
        manager, lender = helix(rank)
        kv = cache(2, (3 + rank, 9 + rank), tpb=2 * TPB)
        entries = lender._block_entries_of_range(manager, kv, 0, end)
        (run,) = lender._in_place_view(entries, None, 0, 0).groups
        assert run.page_ordinals.tolist() == ordinals[rank], (rank, end)
        assert run.slots.tolist() == [(3, 9)[b // 2] + rank for b in ordinals[rank]]


@pytest.mark.cpu_only
@pytest.mark.parametrize("kind", ["recurrent", "window", "sinks"])
def test_helix_refuses_recurrent_state_and_windows_at_the_attach(monkeypatch, kind):
    """Under Helix the attach refuses a state row, a window or sinks, and passes a plain layout."""
    lay = SimpleNamespace(
        cp=(0, 2),
        num_layer_groups=2,
        recurrent=(False, kind == "recurrent"),
        windows=(None, 64 if kind == "window" else None),
        sink_blocks=(0, 1 if kind == "sinks" else 0),
        sparse=(False, False),
        prefix_required=(),
    )
    plain = SimpleNamespace(**vars(lay))
    plain.recurrent, plain.windows, plain.sink_blocks = (False, False), (None, None), (0, 0)
    manager = SimpleNamespace(
        mapping=SimpleNamespace(cp_size=2, pp_size=1, has_cp_helix=lambda: True),
        _model_has_attention_window=False,
        _model_has_sparse_buffers=False,
        _model_attention_window_pattern=(),
        num_layers=2,
        kv_cache_manager_py_config=SimpleNamespace(layers=[]),
        impl=SimpleNamespace(layer_grouping=[[], []]),
    )
    monkeypatch.setattr(_manager, "require_v2", lambda manager: None)
    monkeypatch.setattr(_lender, "derive_layout", lambda manager: lay)
    monkeypatch.setattr(_manager, "model_has_recurrent_state", lambda manager: False)
    with pytest.raises(
        ValueError, match="under Helix the lender lends no recurrent state"
    ) as raised:
        _lender._checked_layout(manager)
    assert "not shown to carry them end to end" in str(raised.value), "the refusal gives no reason"
    monkeypatch.setattr(_lender, "derive_layout", lambda manager: plain)
    assert _lender._checked_layout(manager) is plain


@pytest.mark.cpu_only
def test_a_helix_range_starts_at_a_whole_block_of_every_rank():
    _owner, lender = helix(0)
    with pytest.raises(ValueError, match="starts at a whole block of 64 tokens"):
        lender._whole_blocks(TPB, PROMPT, whole_end=False)
    assert lender._whole_blocks(2 * TPB, PROMPT, whole_end=False) == (2 * TPB, PROMPT)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "rank,delivered,cap,usable",
    [
        (0, [True, True], PROMPT, PROMPT),
        (1, [True, True], PROMPT, PROMPT),
        (1, [True, False], PROMPT, 96),  # its page of block 1 starts at token 96
        (0, [True, False], PROMPT, 64),
        (1, [True, False], 90, 90),  # nothing of its page of block 1 lies below 90
        (1, [False, True], PROMPT, TPB),
    ],
)
def test_a_helix_rank_resumes_up_to_its_first_missing_page_below_the_prompt(
    rank, delivered, cap, usable
):
    _owner, lender = helix(rank)
    deliveries = _fetch_ledger._Deliveries(None, 0, [np.array(delivered)])
    assert lender._ledger._max_resume_position(None, deliveries, (0,), cap) == usable


@pytest.mark.cpu_only
def test_a_helix_lease_marks_and_resumes_per_rank(floor_follows_history):
    for rank, expected in ((0, PROMPT), (1, 96)):
        manager, lender = helix(rank)
        kv = manager.kv_cache_map[7] = cache(2, (3 + rank, 9 + rank), tpb=2 * TPB)
        fetch = lender._ledger._record_fetch(
            7, kv, 0, (0,), kv.history_length, (0,), end=PROMPT, prompt=PROMPT
        )
        entries = lender._block_entries_of_range(manager, kv, 0, PROMPT)
        req = SimpleNamespace(
            py_request_id=7, prompt_len=49, total_input_len_cp=PROMPT, py_multimodal_data=None
        )
        lease = lender._lend_in_place(
            manager, req, kv, "write", entries, None, fetch, start=0, end=PROMPT, ready_event=READY
        )
        masks = lease.poll().row_masks(True)
        masks[0][-1] = rank == 0  # rank 1's page of block 1, token 96, does not arrive
        got = return_write(lender, req, lease, masks)
        assert got.max_position == expected, (rank, got)
        assert got.min_position == PROMPT  # the history, where the floor follows it


# -- windowed fetches ---------------------------------------------------------------------------


def _native_stale(history, tpb, window, sinks):
    """AttnLifeCycle::getStaleRange, including C++ division's truncation toward zero."""
    begin = min(-(-history // tpb), sinks)
    if window is None:
        return begin, begin
    numerator = history + 1 - window
    end = numerator // tpb if numerator >= 0 else -((-numerator) // tpb)
    return begin, max(begin, end)


def _native_needed(position, tpb, window, sinks):
    """Required ordinals at the actual token position, independently of the ledger's search."""
    begin, end = _native_stale(position, tpb, window, sinks)
    return [b for b in range(-(-position // tpb)) if not begin <= b < end]


def _window_case(windows, prompt, *, tpb=TPB, sinks=None, known=0, history=None, state=False):
    sinks = tuple(sinks or (0,) * len(windows))
    windows = tuple(windows)
    count = len(windows) + int(state)
    manager = Manager(tuple(range(count)))
    manager.tokens_per_block = tpb
    manager.enable_block_reuse = False
    manager._stale_block_range = lambda lg, h: _native_stale(h, tpb, windows[lg], sinks[lg])
    blocks = -(-prompt // tpb)
    lay = layout(
        count,
        tpb,
        max(blocks + 1, STATE_SLOT + 1),
        windows=windows + (None,) * int(state),
        sink_blocks=sinks + (0,) * int(state),
        recurrent=(False,) * len(windows) + (True,) * int(state),
    )
    lender = lender_over(manager, lay, no_names="this manager commits no blocks")
    kv = cache(blocks, None, known if history is None else history, known, tpb, count)
    manager.kv_cache_map[7] = kv
    return manager, lender, kv, request(prompt_len=prompt)


def _fetch_all(lender, req, start, end, masks_of=lambda view: view.row_masks(True)):
    lease = lender.lend_write(req, start, end, memory_kind=DEVICE, rows=ALL, ready_event=READY)
    assert lease.failure is None, lease.failure
    view = lease.poll()
    assert view is not None
    masks = masks_of(view)
    return_write(lender, req, lease, masks)
    return view, masks


@pytest.mark.cpu_only
@pytest.mark.parametrize("delivery", ["local", "complete", "missing"])
def test_a_read_refuses_a_cache_whose_committed_tail_was_not_delivered(
    delivery, floor_follows_history
):
    prompt = TPB + 4
    _owner, lender, kv, req = _window_case((None,), prompt, known=prompt, history=prompt)
    if delivery != "local":
        _fetch_all(lender, req, TPB, prompt, lambda view: view.row_masks(delivery == "complete"))
    lease = lender.lend_read(req, TPB, prompt, memory_kind=DEVICE, rows=ALL, ready_event=READY)
    try:
        if delivery == "missing":
            assert lease.failure is not None
            assert lease.poll() is None and not lender._is_locked(kv)
        else:
            assert lease.poll() is not None, lease.failure
    finally:
        lease.release()


def _assert_interval(bounds, tpb, windows, sinks, known, delivered, runs=()):
    def safe(position):
        if any(begin < position < end for begin, end in runs):
            return False
        for lg, (window, sink) in enumerate(zip(windows, sinks)):
            missing = [
                b
                for b in _native_needed(position, tpb, window, sink)
                if min((b + 1) * tpb, position) > known and b not in delivered[lg]
            ]
            if missing:
                return False
        return True

    if bounds.max_position < bounds.min_position:
        return
    assert bounds.min_position >= known, (bounds, known)
    for position in range(bounds.min_position, bounds.max_position + 1):
        assert safe(position), (position, bounds, known, delivered, runs)
    # Completed bidirectional runs set the scheduler's floor even for a one-token run.
    floor_limit = max(known, max((e for _, e in runs if e <= bounds.max_position), default=0))
    floor = bounds.max_position
    while floor > floor_limit and safe(floor - 1):
        floor -= 1
    assert bounds.min_position == floor, (bounds, floor, known, delivered, runs)


@pytest.mark.cpu_only
def test_a_fetch_of_every_row_may_end_at_the_history_its_cache_declares(floor_follows_history):
    """A fetch may end at the declared history inside a block, and resumes there at most."""
    manager, lender, kv, req = _window_case((None,), 5 * TPB, history=100)
    lease = lender.lend_write(req, 0, 90, memory_kind=DEVICE, rows=ALL, ready_event=READY)
    assert lease.failure is not None and "ends at 90" in lease.failure, lease.failure
    assert lease._lock is None, "a lease failed at the call holds a lock"
    lease.release()
    with pytest.raises(ValueError, match="past the request's"):
        lender.lend_write(req, 0, 5 * TPB + 1, memory_kind=DEVICE, rows=ALL, ready_event=READY)
    view, _ = _fetch_all(lender, req, 0, 100)
    assert view.groups[0].page_ordinals.tolist() == [0, 1, 2, 3]
    assert lender.poll_resume_bounds(req).max_position == 100
    assert kv.history_length == 100


@pytest.mark.cpu_only
def test_a_fetch_skips_a_group_committed_past_its_end_and_fills_the_other_group(
    floor_follows_history,
):
    """A 34-token fetch must not overwrite another group's 36-token committed prefix."""
    _owner, lender, kv, req = _window_case((None, None), 64, tpb=8, known=36, history=0)
    kv.reuse_status = [
        SimpleNamespace(complete=True, endpoint=36),
        SimpleNamespace(complete=False, endpoint=36),
    ]
    prefix, _ = _fetch_all(lender, req, 0, 32)
    assert [g.page_ordinals.tolist() for g in prefix.groups] == [[], [0, 1, 2, 3]]
    kv.history_length = 34
    view, _ = _fetch_all(lender, req, 32, 34)
    assert [g.page_ordinals.tolist() for g in view.groups] == [[], [4]]
    assert lender.poll_resume_bounds(req) == (34, 34)

    view, _ = _fetch_all(lender, req, 32, 40)
    assert [g.layer_group_index for g in view.groups] == [0, 1]
    assert all(g.page_ordinals.tolist() == [4] for g in view.groups)


def _names_in_place(lender, monkeypatch):
    """``lender`` lending by name, with a grow that moves the cache's history where asked."""
    lender._no_names = None
    monkeypatch.setattr(
        _manager, "block_keys", lambda manager, req, kv, n: [bytes([i]) * 32 for i in range(n)]
    )

    def grow(manager, req, kv, position, end):
        kv.history_length = position
        return True

    monkeypatch.setattr(_manager, "grow", grow)


@pytest.mark.cpu_only
@pytest.mark.parametrize("fetch", ["named_window", "every_row"])
def test_a_read_of_every_row_fails_over_rows_a_fetch_moving_the_history_left_unwritten(
    floor_follows_history, monkeypatch, fetch
):
    """A read through the history fails over a block no fetch delivered, until a refetch does."""
    if fetch == "named_window":
        _owner, lender, kv, req = _window_case((80,), 80, tpb=8)
        _names_in_place(lender, monkeypatch)
        req.try_get_encoder_output_len = lambda: None
        req.py_return_context_logits, req.py_additional_outputs = False, None
        end, missing, rows = 64, 6, NAMED
    else:
        _owner, lender, kv, req = _window_case((None,), 64, tpb=8, history=48)
        end, missing, rows = 48, 3, ALL

    def deliver(missing_blocks):
        lease = lender.lend_write(req, 0, end, memory_kind=DEVICE, rows=rows, ready_event=READY)
        view = lease.poll()
        assert view is not None, lease.failure
        return_write(lender, req, lease, [~np.isin(view.groups[0].page_ordinals, missing_blocks)])

    deliver([missing])
    assert kv.history_length == end
    assert lender.poll_resume_bounds(req).max_position < end
    read = lender.lend_read(req, 0, end, memory_kind=DEVICE, rows=ALL, ready_event=READY)
    assert read.failure is not None and f"block {missing} holds rows" in read.failure, read.failure
    assert read._lock is None, "a lease failed at the call holds a lock"
    read.release()
    # A request that computed on past the fetch's history wrote every row below it.
    kv.history_length = end + 8
    read = lender.lend_read(req, 0, end, memory_kind=DEVICE, rows=ALL, ready_event=READY)
    assert read.failure is None, read.failure
    read.release()
    kv.history_length = end
    deliver([])
    assert lender.poll_resume_bounds(req).max_position == end
    read = lender.lend_read(req, 0, end, memory_kind=DEVICE, rows=ALL, ready_event=READY)
    assert read.failure is None and read.poll() is not None
    assert missing in read.poll().groups[0].page_ordinals.tolist()
    read.release()


@pytest.mark.cpu_only
def test_a_read_of_every_row_fails_over_committed_rows_a_fetch_wrote_and_did_not_deliver(
    floor_follows_history,
):
    """An undelivered partial committed tail cannot be read; a disjoint delivered block can."""
    _owner, lender, kv, req = _window_case((None,), 64, tpb=8, known=28, history=40)
    _fetch_all(lender, req, 24, 48, lambda view: [view.groups[0].page_ordinals != 3])
    read = lender.lend_read(req, 0, 40, memory_kind=DEVICE, rows=ALL, ready_event=READY)
    assert read.failure is not None and "block 3 holds rows" in read.failure, read.failure
    read.release()
    read = lender.lend_read(req, 32, 40, memory_kind=DEVICE, rows=ALL, ready_event=READY)
    assert read.failure is None and read.poll().groups[0].page_ordinals.tolist() == [4]
    read.release()


@pytest.mark.cpu_only
@pytest.mark.parametrize("boundary_arrives", [False, True], ids=["boundary_missing", "complete"])
def test_a_partial_prompt_fetch_checks_the_window_at_its_actual_end(
    floor_follows_history, boundary_arrives
):
    manager, lender, kv, req = _window_case((64,), PROMPT)

    def masks_of(view):
        assert view.groups[0].page_ordinals.tolist() == [1, 2, 3]
        return [np.array([boundary_arrives, True, True])]

    _fetch_all(lender, req, 0, PROMPT, masks_of)
    bounds = lender.poll_resume_bounds(req)
    assert bounds.max_position == (PROMPT if boundary_arrives else 0), bounds
    delivered = [{2, 3} | ({1} if boundary_arrives else set())]
    _assert_interval(bounds, TPB, (64,), (0,), 0, delivered)
    assert kv.history_length == 0 and manager.kv_cache_map[7] is kv


@pytest.mark.cpu_only
def test_every_position_in_a_windowed_fetch_s_resume_interval_has_its_rows(floor_follows_history):
    manager, lender, kv, req = _window_case((64,), 4 * TPB)
    view, _ = _fetch_all(lender, req, 0, req.prompt_len)
    assert view.groups[0].page_ordinals.tolist() == [2, 3]
    bounds = lender.poll_resume_bounds(req)
    assert tuple(bounds) == (128, 127), bounds
    _assert_interval(bounds, TPB, (64,), (0,), 0, [{2, 3}])
    # A second manager may lower the combined maximum to any point in this interval.
    assert bounds.min_position <= 127 < bounds.max_position
    assert kv.history_length == 0 and manager.kv_cache_map[7] is kv


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "kept,reads,sinks,prompt,runs,segments,expected",
    [
        pytest.param(
            (64,),
            (64,),
            (0,),
            160,
            ((96, 150),),
            ((0, 96, ({2},)), (96, 160, ({3},))),
            (0, 0),
            id="one_run",
        ),
        pytest.param(
            (64, 160),
            (64, 96),
            (0, 1),
            384,
            ((128, 184), (224, 280), (320, 360)),
            (
                (0, 64, ({0, 1}, None)),
                (64, 128, ({3}, None)),
                (128, 192, ({4}, None)),
                (192, 224, ({6}, None)),
                (224, 288, ({7}, None)),
                (288, 320, ({9}, None)),
                (320, 384, ({10}, None)),
            ),
            (64, 0),
            id="three_runs",
        ),
    ],
)
def test_bidirectional_clips_recheck_rows_from_public_fetches(
    floor_follows_history, kept, reads, sinks, prompt, runs, segments, expected
):
    manager, lender, kv, req = _window_case(kept, prompt, sinks=sinks)
    lender._layout.read_windows = reads
    mask = np.zeros(prompt, dtype=np.int64)
    for begin, end in runs:
        mask[begin:end] = 1
    req.py_multimodal_data = {
        "mm_bidirectional_blocks": True,
        "multimodal_embed_mask_cumsum": np.cumsum(mask),
    }
    valid = [{} for _ in kept]
    for start, end, arrived in segments:
        view, marks = _fetch_all(
            lender,
            req,
            start,
            end,
            lambda view: [
                np.ones(len(g.page_ordinals), dtype=bool)
                if arrived[g.layer_group_index] is None
                else np.isin(g.page_ordinals, list(arrived[g.layer_group_index]))
                for g in view.groups
            ],
        )
        for group, delivered in zip(view.groups, marks):
            pages = valid[group.layer_group_index]
            for ordinal, got in zip(group.page_ordinals.tolist(), delivered.tolist()):
                pages.pop(ordinal, None)
                if got:
                    pages[ordinal] = min(end, (ordinal + 1) * TPB)

    def safe(position):
        if any(begin < position < end for begin, end in runs):
            return False
        for lg, window in enumerate(reads):
            # Intersect actual token intervals; include native's in-flight token.
            needed = set(range(sinks[lg])) | {
                token // TPB for token in range(max(0, position + 1 - window), position + 1)
            }
            for ordinal in needed:
                if ordinal * TPB < position and valid[lg].get(ordinal, 0) < min(
                    position, (ordinal + 1) * TPB
                ):
                    return False
        return True

    permitted = [position for position in range(prompt + 1) if safe(position)]
    high = max(permitted)
    low = high
    floor = max((end for _, end in runs if end <= high), default=0)
    while low > floor and safe(low - 1):
        low -= 1
    assert (high, low) == expected
    bounds = lender.poll_resume_bounds(req)
    assert tuple(bounds) == expected, bounds
    assert all(safe(position) for position in range(bounds.min_position, bounds.max_position + 1))
    assert kv.history_length == 0 and manager.kv_cache_map[7] is kv


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "attention_arrives,state_arrives",
    [(False, True), (True, False), (True, True)],
    ids=["attention_boundary_missing", "state_missing", "complete"],
)
def test_a_state_fetch_needs_the_attention_window_at_the_state_end(
    floor_follows_history, attention_arrives, state_arrives
):
    manager, lender, kv, req = _window_case((64,), PROMPT, history=PROMPT, state=True)

    def masks_of(view):
        assert [g.page_ordinals.tolist() for g in view.groups] == [[1, 2, 3], [0]]
        return [np.array([attention_arrives, True, True]), np.array([state_arrives])]

    _fetch_all(lender, req, 0, PROMPT, masks_of)
    assert kv.history_length == PROMPT
    bounds = lender.poll_resume_bounds(req)
    if attention_arrives and state_arrives:
        assert tuple(bounds) == (PROMPT, PROMPT), bounds
    else:
        assert bounds.max_position < bounds.min_position, bounds
    assert manager.kv_cache_map[7] is kv


@pytest.mark.cpu_only
def test_public_windowed_fetch_bounds_match_a_native_formula_oracle(floor_follows_history):
    """Small exhaustive masks and seeded layouts check every permitted token, not only the end."""
    cases = []
    tpb = 4
    for window, sink, known in itertools.product((None, 1, 3, 4, 5, 7, 8), (0, 1), (0, 3, 5)):
        prompt = 3 * tpb + 1
        ordinals = [b for b in _native_needed(prompt, tpb, window, sink) if b >= known // tpb]
        for bits in itertools.product((False, True), repeat=len(ordinals)):
            cases.append((tpb, (window,), (sink,), known, prompt, (bits,), ()))
    rng = random.Random(290053)
    for _ in range(200):
        tpb = rng.choice((4, 8, 16, 32))
        windows = tuple(
            rng.choice((None, 1, tpb - 1, tpb + 1, 2 * tpb - 1, 3 * tpb + 3))
            for _ in range(rng.randint(1, 3))
        )
        sinks = tuple(rng.randrange(3) for _ in windows)
        prompt = rng.randint(2 * tpb, 7 * tpb + 1)
        known = rng.choice((0, tpb - 1, tpb + 1, prompt // 2))
        bits = tuple(
            tuple(
                rng.random() < 0.7 for b in _native_needed(prompt, tpb, w, s) if b >= known // tpb
            )
            for w, s in zip(windows, sinks)
        )
        runs = ()
        if rng.random() < 0.4 and prompt > known + 2:
            begin = rng.randint(known, prompt - 2)
            runs = ((begin, rng.randint(begin + 1, prompt)),)
        cases.append((tpb, windows, sinks, known, prompt, bits, runs))

    for case in cases:
        tpb, windows, sinks, known, prompt, bits, runs = case
        manager, lender, kv, req = _window_case(windows, prompt, tpb=tpb, sinks=sinks, known=known)
        if runs:
            mask = np.zeros(prompt, dtype=np.int64)
            for begin, end in runs:
                mask[begin:end] = 1
            req.py_multimodal_data = {
                "mm_bidirectional_blocks": True,
                "multimodal_embed_mask_cumsum": np.cumsum(mask),
            }
        view, masks = _fetch_all(
            lender, req, known // tpb * tpb, prompt, lambda view: [np.array(b, bool) for b in bits]
        )
        delivered = [
            {int(b) for b, arrived in zip(g.page_ordinals, m) if arrived}
            for g, m in zip(view.groups, masks)
        ]
        bounds = lender.poll_resume_bounds(req)
        tainted_tail = known % tpb and any(
            known // tpb in g.page_ordinals and known // tpb not in got
            for g, got in zip(view.groups, delivered)
        )
        if tainted_tail:
            assert bounds.max_position < bounds.min_position, (case, bounds)
            continue
        _assert_interval(bounds, tpb, windows, sinks, known, delivered, runs)
        assert bounds.max_position <= prompt, (case, bounds)
        if any(len(g) for g in view.groups) and all(m.all() for m in masks):
            assert bounds.max_position == prompt and bounds.min_position <= prompt, (case, bounds)
        assert kv.history_length == known and manager.kv_cache_map[7] is kv


@pytest.mark.cpu_only
@pytest.mark.parametrize("window", [None, 72])
def test_a_complete_refetch_replaces_an_earlier_partial_page(floor_follows_history, window):
    _owner, lender, _kv, req = _window_case((window,), 160, history=65)
    _fetch_all(lender, req, 0, 65)
    assert lender.poll_resume_bounds(req) == (65, 65)
    _fetch_all(lender, req, 0, 160)
    assert lender.poll_resume_bounds(req) == (160, 65)


@pytest.mark.cpu_only
def test_a_partial_page_behind_the_window_still_bounds_every_rewind(floor_follows_history):
    _owner, lender, _kv, req = _window_case((72,), 192, history=65)
    lender._layout.read_windows = (64,)
    _fetch_all(lender, req, 0, 65)
    _fetch_all(lender, req, 96, 192, lambda view: [view.groups[0].page_ordinals < 5])
    assert lender.poll_resume_bounds(req) == (160, 159)
    mask = np.zeros(193, dtype=int)
    mask[152:168] = 1
    req.py_multimodal_data = {
        "mm_bidirectional_blocks": True,
        "multimodal_embed_mask_cumsum": np.cumsum(mask),
    }
    lender._ledger._deliveries_of(7).resume_bounds_memo = None
    assert lender.poll_resume_bounds(req) == (65, 65)


@pytest.mark.cpu_only
def test_a_fetch_with_no_read_history_uses_its_declared_extent(floor_follows_history):
    _owner, lender, _kv, req = _window_case((9,), 160)
    lender._layout.read_windows = (1,)
    _fetch_all(lender, req, 0, 32, lambda view: view.row_masks(False))
    assert lender.poll_resume_bounds(req) == (32, 32)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arrived", [True, False])
def test_a_state_only_stage_constrains_the_pipeline_at_its_delivered_state(
    floor_follows_history, arrived
):
    """A stage of state alone pins the pipeline's resume to the prompt's end, or empties it."""
    _owner, lender, _kv, req = _window_case((), PROMPT, history=PROMPT, state=True)
    _fetch_all(lender, req, 0, PROMPT, lambda view: view.row_masks(arrived))
    state = lender.poll_resume_bounds(req)
    _other, attention_lender, _other_kv, other_req = _window_case((None,), PROMPT)
    _fetch_all(attention_lender, other_req, 0, PROMPT)
    attention = attention_lender.poll_resume_bounds(other_req)
    assert attention.max_position == PROMPT and attention.min_position < PROMPT, attention
    pipeline = (
        min(state.max_position, attention.max_position),
        max(state.min_position, attention.min_position),
    )
    if arrived:
        assert state == (PROMPT, PROMPT) and pipeline == (PROMPT, PROMPT)
    else:
        assert state.max_position < state.min_position and pipeline[0] < pipeline[1]


@pytest.mark.cpu_only
def test_a_holder_changing_its_view_cannot_change_the_arrival_mask_shape(floor_follows_history):
    """Arrivals are checked against the lease's own entries; a refused return leaves it lent."""
    _owner, lender, _kv, req = _window_case((None,), PROMPT)
    lease = lender.lend_write(req, 0, 96, memory_kind=DEVICE, rows=ALL, ready_event=READY)
    view = lease.poll()
    assert view is not None
    object.__setattr__(view.groups[0], "page_ordinals", np.zeros(1, dtype=np.int64))
    with pytest.raises(ValueError, match="shape"):
        lender.return_writes([(lease, [np.array([True])], True)])
    assert return_write(lender, req, lease, [np.zeros(3, dtype=bool)]).max_position == 0


@pytest.mark.cpu_only
@pytest.mark.parametrize("stop", ["error", "shutdown", "gone"])
def test_invalid_write_ends_are_checked_before_the_lender_stops(floor_follows_history, stop):
    manager, lender, _, req = _window_case((None,), PROMPT)
    if stop == "error":
        lender._stop_reason = "injected"
    elif stop == "shutdown":
        lender._shut_down = True
    else:
        del manager
    with pytest.raises(ValueError, match="past the request"):
        lender.lend_write(req, 0, PROMPT + 1, memory_kind=DEVICE, rows=ALL, ready_event=READY)
    with pytest.raises(RuntimeError, match="lends no more"):
        lender.lend_write(req, 0, 96, memory_kind=DEVICE, rows=ALL, ready_event=READY)


@pytest.mark.cpu_only
def test_partial_refetch_intervals_match_the_native_window_rule_at_every_position(
    floor_follows_history,
):
    rng = random.Random(31)
    for _ in range(300):
        read = rng.randint(1, 160)
        kept = read + rng.choice((0, 1, 8, 32))
        partial = rng.randint(1, 95)
        prompt = 192
        sink = rng.choice((0, 0, 1))
        _owner, lender, _kv, req = _window_case((kept,), prompt, history=partial, sinks=(sink,))
        lender._layout.read_windows = (read,)
        valid = {}
        for start, end in ((0, partial), (rng.choice((0, 96)), prompt)):
            lease = lender.lend_write(
                req, start, end, memory_kind=DEVICE, rows=ALL, ready_event=READY
            )
            view = lease.poll()
            assert view is not None, lease.failure
            (group,) = view.groups
            marks = np.array([rng.random() < 0.8 for _ in group.page_ordinals], dtype=bool)
            for ordinal, arrived in zip(group.page_ordinals.tolist(), marks):
                valid.pop(ordinal, None)
                if arrived:
                    valid[ordinal] = min(end, (ordinal + 1) * TPB)
            return_write(lender, req, lease, [marks])
        safe = [
            p
            for p in range(partial, prompt + 1)
            if all(
                valid.get(b, 0) >= min(p, (b + 1) * TPB) for b in _native_needed(p, TPB, read, sink)
            )
        ]
        bounds = lender.poll_resume_bounds(req)
        if not safe:
            assert bounds.max_position < bounds.min_position, (read, kept, partial, valid, bounds)
            continue
        assert bounds.max_position == max(safe), (read, kept, partial, valid, bounds, safe)
        assert all(p in safe for p in range(bounds.min_position, bounds.max_position + 1))


@pytest.mark.cpu_only
def test_refetch_progress_tracks_helix_physical_pages(floor_follows_history):
    for rank in (0, 1):
        manager, lender = helix(rank)
        manager.kv_cache_map[7] = cache(3, (3, 9, 12), history=96, tpb=2 * TPB)
        req = request(prompt_len=80)
        req.total_input_len_cp = 160
        _fetch_all(lender, req, 0, 96)
        assert lender.poll_resume_bounds(req) == (96, 96)
        _fetch_all(lender, req, 64, 160)
        assert lender.poll_resume_bounds(req) == (160, 96)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "kept,reads,floor",
    [((None, 64), (), 191), ((128, 64), (64, 32), 159)],
    ids=["full_attention_and_window", "two_windows"],
)
def test_a_partial_page_a_refetch_leaves_raises_the_floor_of_its_own_layer_group_alone(
    floor_follows_history, kept, reads, floor
):
    """A refetch from 96 re-lends block 3 in the wider group alone; the other's stops at 100."""
    _owner, lender, _kv, req = _window_case(kept, 192, history=100)
    lender._layout.read_windows = reads
    first, _ = _fetch_all(lender, req, 0, 100)
    assert first.groups[1].page_ordinals.tolist() == [1, 2, 3]
    second, _ = _fetch_all(lender, req, 96, 192)
    assert [g.page_ordinals.tolist() for g in second.groups] == [[3, 4, 5], [4, 5]]
    assert lender.poll_resume_bounds(req) == (192, floor)


@pytest.mark.cpu_only
def test_a_group_whose_block_a_refetch_left_undelivered_keeps_the_resume_below_it(
    floor_follows_history,
):
    _owner, lender, _kv, req = _window_case((None, None), 192, history=100)
    _fetch_all(lender, req, 0, 100)
    _fetch_all(lender, req, 96, 192, lambda view: [view.row_masks(True)[0], view.row_masks()[1]])
    assert lender.poll_resume_bounds(req).max_position == 96


@pytest.mark.cpu_only
@pytest.mark.parametrize("history", [80, 95])
@pytest.mark.parametrize("rank", [0, 1])
def test_a_helix_rank_s_partly_fetched_page_bounds_a_later_fetch_s_resume(
    floor_follows_history, rank, history
):
    """A rank resumes where its own pages hold every token: rank 0 at the history, rank 1 at 96."""
    manager, lender = helix(rank)
    manager.kv_cache_map[7] = cache(3, (3, 9, 12), history=history, tpb=2 * TPB)
    req = request(prompt_len=80)
    req.total_input_len_cp = 160
    first, _ = _fetch_all(lender, req, 0, history)
    assert first.groups[0].page_ordinals.tolist() == ([0, 2] if rank == 0 else [1])
    second, _ = _fetch_all(lender, req, 128, 160)
    assert second.groups[0].page_ordinals.tolist() == ([4] if rank == 0 else [])
    assert lender.poll_resume_bounds(req).max_position == (history if rank == 0 else 96)


@pytest.mark.cpu_only
def test_a_shrink_past_a_partial_page_bounds_the_resume_at_the_blocks_it_keeps(
    floor_follows_history,
):
    _owner, lender, kv, req = _window_case((None,), 160, history=100)
    _fetch_all(lender, req, 0, 100)
    assert lender.poll_resume_bounds(req).max_position == 100
    kv.num_blocks = 3
    lender._on_shrink(7, kv)
    assert lender.poll_resume_bounds(req).max_position == 96


@pytest.mark.cpu_only
def test_a_shorter_refetch_keeps_what_the_earlier_fetch_delivered_past_it(floor_follows_history):
    _owner, lender, _kv, req = _window_case((None,), 160)
    _fetch_all(lender, req, 0, 160, lambda view: [view.groups[0].page_ordinals > 0])
    assert lender.poll_resume_bounds(req).max_position == 0
    _fetch_all(lender, req, 0, 32)
    assert lender.poll_resume_bounds(req) == (160, 0)


@pytest.mark.cpu_only
def test_refetch_intervals_match_the_native_rule_over_layer_groups_and_shorter_refetches(
    floor_follows_history,
):
    """Random refetches into one to three groups resume where the native window rule permits."""
    rng = random.Random(83)
    prompt = 192
    checked = 0
    for _ in range(600):
        count = rng.randint(1, 3)
        reads, kept, sinks = [], [], []
        for _ in range(count):
            if rng.random() < 0.4:
                reads.append(None)
                kept.append(None)
                sinks.append(0)
            else:
                read = rng.randint(1, 160)
                reads.append(read)
                kept.append(read + rng.choice((0, 1, 8, 32)))
                sinks.append(rng.choice((0, 0, 1)))
        known = rng.choice((0, 0, rng.randint(1, 95), rng.randint(1, 160)))
        complete = [rng.random() < 0.6 for _ in range(count)]
        history = rng.choice((0, known, rng.randint(0, known)))
        _owner, lender, kv, req = _window_case(
            tuple(kept), prompt, history=history, sinks=tuple(sinks), known=known
        )
        lender._layout.read_windows = tuple(reads)
        kv.reuse_status = [SimpleNamespace(complete=c, endpoint=known) for c in complete]
        committed = [known if c else 0 for c in complete]
        computed = [max(c, history) for c in committed]
        valid = [{} for _ in range(count)]
        tainted = False
        extent = 0  # the furthest end the fetches declare
        ends = sorted({e for e in (history, prompt, *range(TPB, prompt, TPB)) if e >= history})
        for _ in range(rng.randint(1, 4)):
            end = rng.choice(ends)
            start = rng.choice([s for s in range(0, prompt, TPB) if s <= end])
            lease = lender.lend_write(
                req, start, end, memory_kind=DEVICE, rows=ALL, ready_event=READY
            )
            if lease.failure is not None and "committed tokens" in lease.failure:
                # A block the fetch lends would hold committed tokens past its end.
                assert any(end < c and c // TPB == end // TPB for c in committed), lease.failure
                lease.release()
                continue
            view = lease.poll()
            assert view is not None, (lease.failure, start, end, history, known, complete)
            extent = max(extent, end)
            if history > min(committed):
                computed = [max(c, min(t, start)) for c, t in zip(committed, computed)]
            masks = []
            for group in view.groups:
                lg = group.layer_group_index
                arrived = np.array([rng.random() < 0.8 for _ in group.page_ordinals], dtype=bool)
                for ordinal, got in zip(group.page_ordinals.tolist(), arrived):
                    if got:
                        valid[lg][ordinal] = min(end, (ordinal + 1) * TPB)
                    else:
                        valid[lg].pop(ordinal, None)
                        tainted |= ordinal * TPB < committed[lg]
                masks.append(arrived)
            return_write(lender, req, lease, masks)

        def safe(position):
            for lg in range(count):
                for b in _native_needed(position, TPB, reads[lg], sinks[lg]):
                    need = min(position, (b + 1) * TPB)
                    if need > computed[lg] and valid[lg].get(b, 0) < need:
                        return False
            return True

        bounds = lender.poll_resume_bounds(req)
        case = (reads, kept, sinks, known, complete, history, valid, computed, bounds)
        # No resume passes the furthest end the fetches declare, or what every group computed.
        top = min(prompt, max(min(computed), extent))
        permitted = [p for p in range(history, top + 1) if safe(p)]
        if tainted or not permitted:
            assert bounds.max_position < bounds.min_position, case
            continue
        assert bounds.max_position == max(permitted), case
        low = bounds.max_position
        while low > history and safe(low - 1):
            low -= 1
        assert bounds.min_position == low, case
        checked += 1
    assert checked > 150, f"only {checked} cases resumed anywhere"


@pytest.mark.cpu_only
@pytest.mark.parametrize("prefix", ["fetched", "computed"])
@pytest.mark.parametrize("valid_tokens", [TPB - 2, TPB - 1], ids=["read_hole", "last_read_covered"])
def test_a_partial_prefix_keeps_the_pipeline_resume_only_when_it_covers_the_last_read(
    prefix, valid_tokens, floor_follows_history
):
    _owner, lender, _kv, req = _window_case((2,), 64, history=valid_tokens)
    lender._layout.read_windows = (1,)
    if prefix == "fetched":
        _fetch_all(lender, req, 0, valid_tokens)
    _fetch_all(lender, req, 32, 64)
    bounds = lender.poll_resume_bounds(req)
    covered = valid_tokens == TPB - 1
    assert bounds == (64, valid_tokens if covered else 32), (prefix, valid_tokens, bounds)
    for position in range(bounds.min_position, bounds.max_position + 1):
        required = _native_needed(position, TPB, 1, 0)
        assert all(
            min(position, (b + 1) * TPB) <= (valid_tokens if b == 0 else 64) for b in required
        )

    _peer_owner, peer, _, peer_req = _window_case((None,), 64, history=valid_tokens)
    if prefix == "fetched":
        _fetch_all(peer, peer_req, 0, valid_tokens)
    _fetch_all(peer, peer_req, 32, 64, lambda view: view.row_masks(False))
    peer_bounds = peer.poll_resume_bounds(peer_req)
    assert peer_bounds == (valid_tokens, valid_tokens), peer_bounds
    intersection = (
        min(bounds.max_position, peer_bounds.max_position),
        max(bounds.min_position, peer_bounds.min_position),
    )
    assert (intersection[0] >= intersection[1]) == covered, intersection


# -- page-index rows, parked rows and state slots of real managers -------------------------------


def attach(mgr, staging_tokens=3 * TPB):
    return attach_lender(mgr, scope=b"rows", staging=StagingOptions(staging_tokens))


@pytest.fixture
def cache_page_reads(monkeypatch):
    """How many times the lender asked a cache for its pages."""
    reads = [0]
    for name in ("pages", "page_list"):
        ask = getattr(_manager, name)

        def counted(kv, layer_group, _ask=ask):
            reads[0] += 1
            return _ask(kv, layer_group)

        monkeypatch.setattr(_manager, name, counted)
    return reads


def row_matches_cache(mgr, lender, request_id):
    """Whether the request's page-index row holds its cache's pages; ``None`` without one."""
    row = _manager.index_row(mgr, request_id)
    if row is None:
        return None
    kv = mgr.kv_cache_map[request_id]
    blocks = int(kv.num_blocks)
    for lg in range(len(mgr.impl.layer_grouping)):
        cache = np.asarray(kv.get_base_page_indices(lg))[:blocks]
        if not np.array_equal(cache, lender._index_rows[lg, row, 0, :blocks]):
            return False
    return True


def parked_matches_cache(mgr, lender, request_id):
    """Whether the request's parked row holds its cache's pages, then -1; ``None`` without a row."""
    row = lender._parked.get(request_id)
    if row is None:
        return None
    kv = mgr.kv_cache_map[request_id]
    blocks = int(kv.num_blocks)
    for lg in range(len(mgr.impl.layer_grouping)):
        cache = np.asarray(kv.get_base_page_indices(lg))[:blocks]
        if (
            not np.array_equal(cache, row.pages[lg, :blocks])
            or (row.pages[lg, blocks:] != -1).any()
        ):
            return False
    return True


def assert_lent_pages_are_the_cache_s(kit, mgr, lender, request, view):
    """Each row of ``view``, recurrent state's aside, lends the page the cache gives its block."""
    kv = kit.kv(mgr, request)
    for run in view.groups:
        if run.layer_group_index not in lender._recurrent_groups:
            cache = np.asarray(kv.get_base_page_indices(run.layer_group_index))
            assert run.slots.tolist() == cache[run.page_ordinals].tolist()


def receive_target(kit, mgr, request_id, tokens):
    """A generation-init request admitted to receive ``tokens``, or ``None`` without room."""
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestType

    request = kit.make_request(
        request_id, tokens, llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY
    )
    if not mgr.prepare_disagg_gen_init(request):
        return None
    mgr.take_disagg_receive_ready(request)
    return request


def read_every_row(kit, mgr, lender, request, tokens, cache_page_reads):
    """A fast read of every row of ``request``'s first ``tokens``, asking the cache for no page."""
    before = cache_page_reads[0]
    lease = lender.lend_read(request, 0, tokens, **kit.DEVICE_ALL)
    assert lease.failure is None and lease._block_entries.spans is not None, lease.failure
    assert cache_page_reads[0] == before, "the read asked the cache for pages"
    assert_lent_pages_are_the_cache_s(kit, mgr, lender, request, kit.ready_view(lease, mgr))
    return lease


def prepare(mgr, *requests):
    """The requests' resources prepared as for a batch's context requests before a forward."""
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

    batch = ScheduledRequests()
    for request in requests:
        batch.append_context_request(request)
    mgr.prepare_resources(batch)


def chunk(mgr, request, tokens):
    """One context chunk of ``tokens``: allocated, prepared, computed and updated."""
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

    assert mgr.resize_context(request, tokens)
    prepare(mgr, request)
    request.context_chunk_size = tokens
    request.move_to_next_context_chunk()
    batch = ScheduledRequests()
    batch.append_context_request(request)
    mgr.update_context_resources(batch)


def computed_request(kit, mgr, request_id, tokens):
    """A request whose prompt the manager prepared and computed in one chunk."""
    request = kit.admitted(mgr, request_id, tokens)
    chunk(mgr, request, request.context_remaining_length)
    assert request.context_remaining_length == 0
    return request


@gpu
@pytest.mark.parametrize("groups", [1, 2], ids=["one_group", "two_groups"])
@pytest.mark.parametrize("blocks", [3, _lender._LISTED_BLOCKS + 6], ids=["listed", "viewed"])
def test_a_write_of_every_row_lends_its_row_s_pages_without_asking_the_cache(
    kit, real_manager, cache_page_reads, blocks, groups
):
    """A receive's write of every row lends its row's pages, asking the cache for none."""
    tokens = list(range(2000, 2000 + blocks * TPB + 1))
    windows = None if groups == 1 else [(blocks + 3) * TPB, (blocks + 4) * TPB]
    with real_manager(
        max_tokens=4 * (blocks + 2) * TPB, max_seq_len=(blocks + 4) * TPB, windows=windows
    ) as mgr:
        lender = attach(mgr)
        assert len(mgr.impl.layer_grouping) == groups, "inconclusive: another number of groups"
        target = receive_target(kit, mgr, 7, tokens)
        assert target is not None
        mgr._stream.synchronize()
        assert row_matches_cache(mgr, lender, 7)
        lease = lender.lend_write(target, 0, len(tokens), **kit.DEVICE_ALL)
        assert lease.failure is None and lease._block_entries.spans is not None, lease.failure
        assert cache_page_reads[0] == 0, "the write asked the cache for pages"
        assert_lent_pages_are_the_cache_s(kit, mgr, lender, target, lease.poll())
        bounds = kit.return_write(lender, lease, ALL_ARRIVED)
        assert bounds.max_position == len(tokens), bounds
        mgr._stream.synchronize()
        mgr.free_resources(target)


@gpu
def test_the_rows_hold_the_cache_s_pages_through_the_executor_s_operations(kit, host_tier_manager):
    """A random walk of the executor's operations: every request's row holds its cache's pages."""
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

    rng = random.Random(11)
    done = dict.fromkeys(
        ("compared", "released", "reused", "resumed", "generated", "parked_compared"), 0
    )
    done.update(dict.fromkeys(("parked_resumed", "deferred", "parked_reused"), 0))
    with host_tier_manager() as mgr:
        lender = attach(mgr)
        live, rows_of, seen_rows, receives, next_id = {}, {}, set(), set(), 100
        held, parked_seen, closing = {}, set(), []
        for _ in range(1000):
            action = rng.random()
            if action < 0.2 and len(live) < 6:
                next_id += 1
                tokens = list(range(next_id * 1000, next_id * 1000 + rng.choice((17, 33, 64))))
                if rng.random() < 0.5:
                    request = receive_target(kit, mgr, next_id, tokens)
                    if request is None:
                        continue
                    receives.add(next_id)
                else:
                    request = kit.make_request(next_id, tokens)
                    if not mgr.prepare_context(request):
                        continue
                    kit.kv(mgr, request).enable_swa_scratch_reuse = False
                live[next_id] = request
                row = _manager.index_row(mgr, next_id)
                done["reused"] += row in seen_rows
                seen_rows.add(row)
                rows_of[next_id] = row
            elif action < 0.42 and live:
                rid = rng.choice(list(live))
                request = live[rid]
                kv = kit.kv(mgr, request)
                if rid not in receives and request.context_remaining_length and kv.is_active:
                    step = min(TPB, request.context_remaining_length)
                    if mgr.resize_context(request, step):
                        request.context_chunk_size = step
                        request.move_to_next_context_chunk()
                        batch = ScheduledRequests()
                        batch.append_context_request(request)
                        mgr.update_context_resources(batch)
            elif action < 0.6 and live:
                rid = rng.choice(list(live))
                request = live[rid]
                computed = rid not in receives and not request.context_remaining_length
                if computed and mgr.try_allocate_generation(request):
                    request.add_new_token(7, 0)
                    batch = ScheduledRequests()
                    batch.append_generation_request(request)
                    mgr.update_resources(batch)
                    done["generated"] += 1
            elif action < 0.68 and live:
                rid = rng.choice(list(live))
                request = live[rid]
                kv = kit.kv(mgr, request)
                # A lender refuses the suspension of a cache whose blocks a read holds.
                if kv.is_active and rid not in held:
                    mgr._stream.synchronize()
                    mgr.suspend_request(request)
                    # Others push the suspended pages to host, so the resume can move them.
                    for orid, other in live.items():
                        computed = orid not in receives and not other.context_remaining_length
                        if other is not request and computed and kit.kv(mgr, other).is_active:
                            mgr.try_allocate_generation(other)
                    if mgr.resume_request(request):
                        done["resumed"] += 1
                        done["parked_resumed"] += rows_of[rid] is None
            elif action < 0.76 and live:
                rid = rng.choice(list(live))
                request = live[rid]
                if not request.context_remaining_length and rows_of[rid] is not None:
                    mgr.release_index_slot(rid)
                    rows_of[rid] = None
                    done["released"] += 1
                    parked = lender._parked.get(rid)
                    assert parked is not None, f"request {rid}: no row parked at its release"
                    done["parked_reused"] += id(parked) in parked_seen
                    parked_seen.add(id(parked))
                    assert all(parked is not row for row, _ in closing), "an open cache's row"
            elif action < 0.82 and held:
                lease = held.pop(rng.choice(list(held)))
                lease.release()
            elif live:
                rid = rng.choice(list(live))
                request = live[rid]
                computed = rid not in receives and not request.context_remaining_length
                kv = kit.kv(mgr, request)
                if computed and kv.is_active and rid not in held and rng.random() < 0.4:
                    lease = lender.lend_read(request, 0, request.prompt_len, **kit.DEVICE_ALL)
                    if lease.failure is None:
                        kit.ready_view(lease, mgr)
                        held[rid] = lease
                    else:
                        lease.release()
                parked = lender._parked.get(rid)
                mgr._stream.synchronize()
                mgr.free_resources(live.pop(rid))
                if rid in held:
                    done["deferred"] += 1
                    assert not kit.closed(kv), "a free closed a cache whose blocks a read holds"
                if parked is not None:
                    closing.append((parked, kv))
                rows_of.pop(rid)
                receives.discard(rid)
            closing = [(row, kv) for row, kv in closing if not kit.closed(kv)]
            mgr._stream.synchronize()
            for rid in live:
                if not kit.kv(mgr, live[rid]).is_active:
                    continue
                if rows_of[rid] is None:
                    assert row_matches_cache(mgr, lender, rid) is None, f"request {rid}: a row"
                    assert parked_matches_cache(mgr, lender, rid), f"request {rid}: parked row"
                    done["parked_compared"] += 1
                else:
                    assert row_matches_cache(mgr, lender, rid), f"request {rid}: its row differs"
                    done["compared"] += 1
        mgr._stream.synchronize()
        for lease in held.values():
            lease.release()
        for request in live.values():
            mgr.free_resources(request)
    counts = str(done)
    assert done["compared"] >= 500 and done["released"] >= 5 and done["reused"] >= 5, counts
    assert done["resumed"] >= 5 and done["generated"] >= 20, counts
    assert done["parked_compared"] >= 50 and done["parked_resumed"] >= 2, counts
    assert done["deferred"] >= 3 and done["parked_reused"] >= 2, counts


def manager_of(kind, real_manager, hybrid_manager, deepseek_v4_manager):
    """A manager of ``kind`` and the prompt length its tests read."""
    if kind == "one_group":
        return real_manager(), 3 * TPB + 1
    if kind == "two_groups":
        # Windows longer than any prompt here give two layer groups that keep every block.
        return real_manager(windows=[6 * TPB, 7 * TPB], max_seq_len=8 * TPB), 3 * TPB + 1
    if kind == "hybrid":
        return hybrid_manager(), 3 * TPB + 1
    return deepseek_v4_manager(), 3 * 128 + 5


@gpu
@pytest.mark.parametrize(
    "kind",
    ["one_group", "two_groups", "hybrid", pytest.param("deepseek_v4", marks=skip_pre_blackwell)],
)
def test_a_read_after_the_slot_s_release_lends_the_parked_row_s_pages(
    kit, real_manager, hybrid_manager, deepseek_v4_manager, cache_page_reads, kind
):
    """A released slot's parked row keeps its pages past the slot's next request, and lends them."""
    context, length = manager_of(kind, real_manager, hybrid_manager, deepseek_v4_manager)
    with context as mgr:
        lender = attach(mgr, staging_tokens=128 if kind == "deepseek_v4" else 3 * TPB)
        if kind != "one_group":
            assert len(mgr.impl.layer_grouping) > 1, "inconclusive: one layer group"
        publish = kit.published if kind != "hybrid" else partial(computed_request, kit)
        source = publish(mgr, 7, list(range(1000, 1000 + length)))
        slot_row = _manager.index_row(mgr, 7)
        mgr._stream.synchronize()
        mgr.release_index_slot(7)
        assert parked_matches_cache(mgr, lender, 7), "the parked row differs from the cache"
        nxt = publish(mgr, 8, list(range(5000, 5000 + length)))
        assert _manager.index_row(mgr, 8) == slot_row, "inconclusive: the slot's row went unused"
        assert parked_matches_cache(mgr, lender, 7), "the slot's next request changed the row"
        read_every_row(kit, mgr, lender, source, length, cache_page_reads).release()
        mgr._stream.synchronize()
        for request in (source, nxt):
            mgr.free_resources(request)


@gpu
def test_a_freed_request_s_row_goes_back_to_use_only_once_its_cache_closes(
    kit, real_manager, cache_page_reads
):
    """A row parked for a cache whose close a read defers goes back to use only once it closes."""
    prompts = {rid: list(range(rid * 1000, rid * 1000 + 2 * TPB + 1)) for rid in (1, 2, 3)}
    with real_manager() as mgr:
        lender = attach(mgr)
        first = kit.published(mgr, 1, prompts[1])
        mgr.release_index_slot(1)
        row = lender._parked[1]
        held = read_every_row(kit, mgr, lender, first, len(prompts[1]), cache_page_reads)
        closing = kit.kv(mgr, first)
        mgr._stream.synchronize()
        mgr.free_resources(first)
        assert not kit.closed(closing), "inconclusive: the free closed the cache"
        second = kit.published(mgr, 2, prompts[2])
        mgr.release_index_slot(2)
        assert lender._parked[2] is not row, "a row went back to use before its cache closed"
        held.release()
        assert kit.closed(closing), "inconclusive: the last unlock left the cache open"
        third = kit.published(mgr, 3, prompts[3])
        mgr.release_index_slot(3)
        assert lender._parked[3] is row, "inconclusive: the closed cache's row was not reused"
        for request in (second, third):
            rid = request.py_request_id
            assert parked_matches_cache(mgr, lender, rid), rid
            read_every_row(kit, mgr, lender, request, len(prompts[rid]), cache_page_reads).release()
        mgr._stream.synchronize()
        for request in (second, third):
            mgr.free_resources(request)


@gpu
def test_a_row_handed_out_again_gives_no_page_until_its_cache_binds_it(
    kit, real_manager, monkeypatch
):
    """A row handed out again holds no page; unbound, a read lends the cache's own pages."""
    with real_manager() as mgr:
        lender = attach(mgr)
        first = kit.published(mgr, 1, list(range(1000, 1000 + 2 * TPB + 1)))
        mgr.release_index_slot(1)
        row = lender._parked[1]
        mgr._stream.synchronize()
        mgr.suspend_request(first)
        assert (row.pages[0, :2] >= 0).all(), "inconclusive: the suspension left no page"
        mgr.free_resources(first)
        second = kit.published(mgr, 2, list(range(5000, 5000 + 2 * TPB + 1)))
        give = _lender_hooks.parking_row

        def unbound(manager, request_id, kv_cache):
            give(manager, request_id, kv_cache)
            return None

        monkeypatch.setattr(_lender_hooks, "parking_row", unbound)
        mgr.release_index_slot(2)
        assert lender._parked[2] is row, "inconclusive: the closed cache's row was not reused"
        assert (row.pages == -1).all(), "a row went back to use holding pages"
        lease = lender.lend_read(second, 0, 2 * TPB + 1, **kit.DEVICE_ALL)
        assert_lent_pages_are_the_cache_s(kit, mgr, lender, second, kit.ready_view(lease, mgr))
        lease.release()
        mgr._stream.synchronize()
        mgr.free_resources(second)


@gpu
def test_a_parked_row_bound_to_another_cache_lends_nothing(kit, real_manager, cache_page_reads):
    """A row parked for another cache than the request's lends no page: the read asks the cache."""
    with real_manager() as mgr:
        lender = attach(mgr)
        prompt = list(range(1000, 1000 + 2 * TPB + 1))
        source = kit.published(mgr, 1, prompt)
        mgr.release_index_slot(1)
        lender._parked[1].kv = object()
        before = cache_page_reads[0]
        lease = lender.lend_read(source, 0, len(prompt), **kit.DEVICE_ALL)
        view = kit.ready_view(lease, mgr)
        assert cache_page_reads[0] > before, "the read lent a row bound to another cache"
        assert_lent_pages_are_the_cache_s(kit, mgr, lender, source, view)
        lease.release()
        lender._parked[1].kv = kit.kv(mgr, source)
        mgr._stream.synchronize()
        mgr.free_resources(source)


def state_group(mgr) -> int:
    (lg,) = [lg for lg, flag in enumerate(derive_layout(mgr).recurrent) if flag]
    return lg


@pytest.fixture
def cache_reads(monkeypatch):
    """How many times the lender asked a cache for its state slot."""
    reads = [0]
    ask = _manager.state_slot

    def counted(kv, layer_group):
        reads[0] += 1
        return ask(kv, layer_group)

    monkeypatch.setattr(_manager, "state_slot", counted)
    return reads


def slot_of(kit, mgr, lender, request, cache_reads):
    """The state slot a lend of ``request`` reads, checked, and which of record or cache gave it."""
    lg = state_group(mgr)
    kv = kit.kv(mgr, request)
    before = cache_reads[0]
    slot = lender._state_slot(kv, request.py_request_id, lg)
    assert slot == kv.get_ssm_block_base_index(lg), (request.py_request_id, slot)
    return slot, ("cache" if cache_reads[0] > before else "record")


def lent_slot(kit, lender, mgr, request, kind):
    """The state row's slot of an in-place lease of every row of ``request``'s prompt."""
    lend = lender.lend_read if kind == "read" else lender.lend_write
    lease = lend(request, 0, PROMPT, **kit.DEVICE_ALL)
    assert lease.failure is None and lease._first_poll_failure is None, lease.failure
    view = lease.poll()
    assert view is not None, lease.failure
    (row,) = [run for run in view.groups if run.layer_group_index == state_group(mgr)]
    return lease, int(row.slots[0])


@gpu
def test_the_record_answers_after_each_preparation_of_a_chunked_prefill(
    kit, hybrid_manager, cache_reads
):
    """The cache answers before a request's first preparation, its record at every chunk after."""
    with hybrid_manager() as mgr:
        lender = attach(mgr)
        first, second = kit.admitted(mgr, 7, TOKENS), kit.admitted(mgr, 8, TOKENS)
        assert slot_of(kit, mgr, lender, first, cache_reads)[1] == "cache"
        sources = []
        for tokens in (TPB, TPB, PROMPT - 2 * TPB):
            chunk(mgr, first, tokens)
            sources.append(slot_of(kit, mgr, lender, first, cache_reads)[1])
            if second.context_remaining_length:
                chunk(mgr, second, second.context_remaining_length)
            sources.append(slot_of(kit, mgr, lender, second, cache_reads)[1])
        assert set(sources) == {"record"}, sources
        mine = slot_of(kit, mgr, lender, first, cache_reads)[0]
        theirs = slot_of(kit, mgr, lender, second, cache_reads)[0]
        assert mine != theirs, "inconclusive: both requests hold one slot"
        reads = cache_reads[0]
        lease, slot = lent_slot(kit, lender, mgr, first, "read")
        assert slot == mine and cache_reads[0] == reads, "the read asked the cache"
        lease.release()
        mgr._stream.synchronize()
        for request in (first, second):
            mgr.free_resources(request)


@gpu
def test_a_resume_leaves_the_record_unanswered_until_the_cache_confirms_it(
    kit, hybrid_manager, cache_reads
):
    """After a resume the cache answers until it confirms the record; a stale one is never lent."""
    with hybrid_manager() as mgr:
        lender = attach(mgr)
        request = kit.admitted(mgr, 7, TOKENS)
        chunk(mgr, request, PROMPT)
        slot, source = slot_of(kit, mgr, lender, request, cache_reads)
        assert source == "record"
        mgr.suspend_request(request)
        assert mgr.resume_request(request)
        assert slot_of(kit, mgr, lender, request, cache_reads)[1] == "cache"
        assert slot_of(kit, mgr, lender, request, cache_reads)[1] == "record", "never confirmed"
        mgr.suspend_request(request)
        assert mgr.resume_request(request)
        record = mgr._request_id_to_state_index
        record[request.py_request_id] = slot + 1  # the slot before a move the record missed
        for _ in range(2):
            assert slot_of(kit, mgr, lender, request, cache_reads)[1] == "cache"
        lease, lent = lent_slot(kit, lender, mgr, request, "read")
        assert lent == slot, "a lease lent the record's stale slot"
        lease.release()
        prepare(mgr, request)
        assert record[request.py_request_id] == slot
        assert slot_of(kit, mgr, lender, request, cache_reads)[1] == "cache"
        assert slot_of(kit, mgr, lender, request, cache_reads)[1] == "record"
        mgr.suspend_request(request)
        assert mgr.resume_request(request)
        assert request.py_request_id in lender._state_unverified
        mgr._stream.synchronize()
        mgr.free_resources(request)
        assert request.py_request_id not in lender._state_unverified, "the free kept its mark"


@gpu
def test_a_freed_slot_serves_its_next_request_from_that_request_s_record(
    kit, hybrid_manager, cache_reads
):
    """The sole freed slot has no new record until its next request's preparation."""
    with hybrid_manager() as mgr:
        lender = attach(mgr)
        first = kit.admitted(mgr, 7, TOKENS)
        chunk(mgr, first, PROMPT)
        slot = slot_of(kit, mgr, lender, first, cache_reads)[0]
        second = kit.admitted(mgr, 8, list(range(5000, 5000 + PROMPT)))
        chunk(mgr, second, PROMPT)
        assert slot_of(kit, mgr, lender, second, cache_reads)[0] != slot
        mgr._stream.synchronize()
        mgr.free_resources(first)
        assert first.py_request_id not in mgr._request_id_to_state_index
        nxt = kit.admitted(mgr, 9, list(range(9000, 9000 + PROMPT)))
        assert slot_of(kit, mgr, lender, nxt, cache_reads)[1] == "cache"
        chunk(mgr, nxt, PROMPT)
        again, source = slot_of(kit, mgr, lender, nxt, cache_reads)
        assert again == slot, "the sole freed state slot was not reused"
        assert source == "record"
        lease, lent = lent_slot(kit, lender, mgr, nxt, "read")
        assert lent == again
        lease.release()
        mgr._stream.synchronize()
        for request in (second, nxt):
            mgr.free_resources(request)


@gpu
def test_a_receive_target_lends_the_cache_s_slot_until_its_preparation_then_the_record_s(
    kit, hybrid_manager, cache_reads
):
    """A receive target's write lends the cache's slot before its preparation, its record after."""
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestType
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

    with hybrid_manager() as mgr:
        lender = attach(mgr)
        target = kit.make_request(
            7, TOKENS, llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY
        )
        assert mgr.prepare_disagg_gen_init(target)
        mgr._stream.synchronize()
        assert slot_of(kit, mgr, lender, target, cache_reads)[1] == "cache"
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [target]
        mgr.prepare_resources(batch)
        mgr._stream.synchronize()
        slot, source = slot_of(kit, mgr, lender, target, cache_reads)
        assert source == "record"
        reads = cache_reads[0]
        lease, lent = lent_slot(kit, lender, mgr, target, "write")
        assert lent == slot and cache_reads[0] == reads, "the write asked the cache"
        bounds = kit.return_write(lender, lease, ALL_ARRIVED)
        assert bounds is not None and bounds.max_position == PROMPT, bounds
        mgr._stream.synchronize()
        mgr.free_resources(target)
