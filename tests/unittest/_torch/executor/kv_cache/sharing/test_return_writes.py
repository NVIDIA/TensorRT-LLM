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
"""Returned write leases and the resume bounds they leave: windows, claimed prefixes, stages."""

import random
import weakref
from contextlib import ExitStack
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.pyexecutor.kv_cache import _lender_hooks
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import (
    ALL_ARRIVED,
    HoldReason,
    MemoryKind,
    ResumeBounds,
    RowSelection,
    StagingOptions,
    _fetch_ledger,
    _lender,
    _manager,
    _ranges,
    attach_lender,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing._identity import Identity
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing._layout import DevicePool
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing._page_locks import LockMode
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing._slots import SlotQueue
from tensorrt_llm.runtime.kv_cache_manager_v2 import AttnLifeCycle

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="allocates KV cache pools")

TPB = 32
PROMPT = 97  # three whole blocks and one token
TOKENS = list(range(1000, 1000 + PROMPT))
OTHER_TOKENS = list(range(5000, 5000 + PROMPT))  # no block shared with TOKENS
DEVICE, HOST = MemoryKind.DEVICE, MemoryKind.HOST
NAMED, ALL = RowSelection.NAMED, RowSelection.ALL


class Rules(_lender._ManagerLender):
    """A lender of a class of its own: every lend and every return goes through the rules."""


def attach(mgr, rules=False, staging_tokens=4 * TPB):
    staging = StagingOptions(staging_tokens)
    if rules:
        return _lender._attach_lender(mgr, scope=b"returns", staging=staging, cls=Rules)
    return attach_lender(mgr, scope=b"returns", staging=staging)


def receiving(kit, mgr, request_id, tokens=TOKENS, hybrid=False):
    """A generation-init request admitted to receive ``tokens``, with recurrent state prepared."""
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestType
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

    request = kit.make_request(
        request_id, tokens, llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY
    )
    assert mgr.prepare_disagg_gen_init(request)
    mgr.take_disagg_receive_ready(request)
    if hybrid:
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [request]
        mgr.prepare_resources(batch)
    return request


def ready_write(lender, mgr, request, end=None, start=0, memory_kind=DEVICE, rows=ALL):
    """A write lease of ``[start, end)`` and its view once ready."""
    end = request.prompt_len if end is None else end
    event = torch.cuda.Event()
    event.record(mgr._stream)
    lease = lender.lend_write(
        request, start, end, memory_kind=memory_kind, rows=rows, ready_event=event
    )
    assert lease.failure is None and lease._first_poll_failure is None, lease.failure
    mgr._stream.synchronize()
    for _ in range(4):
        lender.progress()
        view = lease.poll()
        if view is not None:
            assert (view.start, view.end) == (start, end), (view.start, view.end)
            return lease, view
        mgr._stream.synchronize()
    raise AssertionError(f"the write never became ready: {lease.failure}")


def ready_in_place(kit, lender, request, start=0, end=None):
    """A fast write lease of ``[start, end)`` and its view, ready without a round of progress."""
    end = request.prompt_len if end is None else end
    lease = lender.lend_write(request, start, end, **kit.DEVICE_ALL)
    view = lease.poll()
    assert view is not None, lease.failure
    return lease, view


def records(lender, request_id):
    """The ledger's records of the request, as plain values."""
    ledger = lender._ledger
    fetch = ledger._latest_fetches.get(request_id)
    delivered = ledger._deliveries.get(request_id)
    return dict(
        latest=None
        if fetch is None
        else (fetch.start, fetch.end, fetch.state_end, fetch.marks_queued),
        deliveries=None
        if delivered is None
        else (
            delivered.lowest_start,
            [mask.tolist() for mask in delivered.delivered_block_masks],
            delivered.state_end,
            delivered.end,
            sorted(delivered.partial_ends.items()),
        ),
        mixed=request_id in ledger._mixed_tail_caches,
    )


@pytest.fixture
def fast_returns(monkeypatch):
    """The returns that took the fast path: whole deliveries recorded, and bounds in closed form."""
    taken = {"recorded": 0, "bounds": 0}
    ledger = _fetch_ledger._FetchLedger
    record, bounds = ledger._record_whole_delivery, ledger._whole_delivery_bounds

    def counted_record(self, *args, **kwargs):
        taken["recorded"] += 1
        return record(self, *args, **kwargs)

    def counted_bounds(self, *args, **kwargs):
        found = bounds(self, *args, **kwargs)
        taken["bounds"] += found is not None
        return found

    monkeypatch.setattr(ledger, "_record_whole_delivery", counted_record)
    monkeypatch.setattr(ledger, "_whole_delivery_bounds", counted_bounds)
    return taken


def arrivals_of(view, kind):
    """``view``'s arrivals: the constant, masks writable or read-only, or all but each last row."""
    if kind == "constant":
        return ALL_ARRIVED
    masks = [np.ones(len(rows), dtype=bool) for rows in view.groups]
    if kind == "all_but_last":
        for mask in masks:
            if len(mask):
                mask[-1] = False
    if kind == "read_only":
        for mask in masks:
            mask.flags.writeable = False
    return masks


@gpu
@pytest.mark.parametrize("arrived", ["constant", "masks", "read_only", "all_but_last"])
@pytest.mark.parametrize("hybrid", [False, True], ids=["paged", "hybrid"])
def test_a_fast_return_gives_what_the_rules_give(
    kit, real_manager, hybrid_manager, fast_returns, hybrid, arrived
):
    """A receive returned fast and through the rules gives the same bounds and records."""
    outcomes = []
    for rules in (False, True):
        with hybrid_manager() if hybrid else real_manager() as mgr:
            lender = attach(mgr, rules=rules)
            request = receiving(kit, mgr, 7, hybrid=hybrid)
            lease, view = ready_write(lender, mgr, request)
            bounds = lender.return_writes([(lease, arrivals_of(view, arrived), True)])
            outcomes.append(
                (bounds, lender.poll_resume_bounds(request), records(lender, 7), lease._released)
            )
            mgr._stream.synchronize()
            mgr.free_resources(request)
    assert outcomes[0] == outcomes[1]
    (bounds,), again, _, released = outcomes[0]
    assert bounds == again and released
    if arrived != "all_but_last":
        assert bounds.max_position == PROMPT
        assert fast_returns == {"recorded": 1, "bounds": 1}, fast_returns
    else:
        assert fast_returns["recorded"] == 0


def invalid_entries(write, view, request, read):
    """Entries of a batch that is invalid in one place each, with what it raises."""
    good = (write, ALL_ARRIVED, True)
    return [
        ("read lease", [good, (read, ALL_ARRIVED, True)], RuntimeError, "for write leases"),
        ("duplicate", [good, good], RuntimeError, "returned already"),
        ("mask count", [(write, [], True)], ValueError, "masks for"),
        (
            "mask dtype",
            [(write, [np.ones(len(r), dtype=np.int8) for r in view.groups], True)],
            ValueError,
            "must be bool",
        ),
        (
            "mask shape",
            [(write, [np.ones(len(r) + 1, dtype=bool) for r in view.groups], True)],
            ValueError,
            "must be bool",
        ),
        ("not a triple", [good, (write, ALL_ARRIVED)], TypeError, "an entry is"),
        ("reusable not a bool", [(write, ALL_ARRIVED, 1)], TypeError, "reusable must be a bool"),
        ("request first", [(request, write, ALL_ARRIVED)], TypeError, "reusable must be a bool"),
        ("not a lease", [(object(), ALL_ARRIVED, True)], ValueError, "not lent by this"),
    ]


@gpu
@pytest.mark.parametrize("hybrid", [False, True], ids=["paged", "hybrid"])
def test_a_batch_with_one_invalid_entry_raises_and_changes_nothing(
    kit, real_manager, hybrid_manager, hybrid
):
    """A batch or a call with one invalid entry raises before any entry is returned."""
    with hybrid_manager() if hybrid else real_manager() as mgr:
        lender = attach(mgr)
        request = receiving(kit, mgr, 7, hybrid=hybrid)
        other = receiving(kit, mgr, 8, hybrid=hybrid)
        write, view = ready_write(lender, mgr, request)
        second, _ = ready_write(lender, mgr, other)
        source = receiving(kit, mgr, 9, hybrid=hybrid)
        read = lender.lend_read(source, 0, PROMPT, **kit.DEVICE_ALL)
        assert read.poll() is not None, read.failure
        before = {7: records(lender, 7), 8: records(lender, 8)}
        version = lender._locks.version
        for name, batch, error, match in invalid_entries(write, view, request, read):
            alone = [batch] if len(batch) == 1 else []
            for entries in [[(second, ALL_ARRIVED, True), *batch], *alone]:
                with pytest.raises(error, match=match):
                    lender.return_writes(entries)
                assert not write._released and not write._marked, name
                assert not second._released, name
                assert {7: records(lender, 7), 8: records(lender, 8)} == before, name
                assert lender._locks.version == version, name
                lender.progress()
                assert lender.poll_resume_bounds(request) is None, name
        bounds = lender.return_writes([(write, ALL_ARRIVED, True), (second, ALL_ARRIVED, True)])
        assert all(isinstance(b, ResumeBounds) for b in bounds), bounds
        for lease in (write, read):
            with pytest.raises(RuntimeError):
                lender.return_writes([(lease, ALL_ARRIVED, True)])
        read.release()
        unpolled = receiving(kit, mgr, 10, hybrid=hybrid)
        pending = lender.lend_write(unpolled, 0, PROMPT, **kit.DEVICE_ALL)
        with pytest.raises(RuntimeError, match="before poll"):
            lender.return_writes([(pending, ALL_ARRIVED, True)])
        pending.release()
        with pytest.raises(RuntimeError, match="returned already"):
            lender.return_writes([(pending, ALL_ARRIVED, True)])
        mgr._stream.synchronize()
        for r in (request, other, source, unpolled):
            mgr.free_resources(r)


@gpu
def test_a_partial_arrival_resumes_where_the_delivered_prefix_ends(kit, real_manager):
    """Rows that did not arrive count nothing: the bounds stop at the first block missing."""
    with real_manager() as mgr:
        lender = attach(mgr)
        request = receiving(kit, mgr, 7)
        lease, view = ready_write(lender, mgr, request)
        masks = view.row_masks(True)
        masks[0][2:] = False  # blocks 2 and 3 of the only layer group missing
        bounds = kit.return_write(lender, lease, masks)
        assert bounds.max_position == 2 * TPB, bounds
        assert lease._released and not lender._locks.is_locked(kit.kv(mgr, request))
        mgr._stream.synchronize()
        mgr.free_resources(request)


@gpu
@pytest.mark.parametrize("hybrid", [False, True], ids=["paged", "hybrid"])
def test_a_lease_of_a_retired_cache_counts_nothing_and_gives_no_bounds(
    kit, real_manager, hybrid_manager, hybrid
):
    """A lease of a freed request's old cache keeps its pages, counts no row and gives no bounds."""
    with hybrid_manager() if hybrid else real_manager() as mgr:
        lender = attach(mgr)
        moved = receiving(kit, mgr, 7, hybrid=hybrid)
        lease, _ = ready_write(lender, mgr, moved)
        old = kit.kv(mgr, moved)
        mgr._stream.synchronize()
        mgr.free_resources(moved)
        assert not kit.closed(old), "the lock kept no cache open"
        held = lender.held_page_counts()
        assert HoldReason.LOCKED not in held.by_reason and held.deduct == held.unreclaimable
        assert sum(held.deduct.values()) > 0, held
        again = receiving(kit, mgr, 7, hybrid=hybrid)
        assert kit.kv(mgr, again) is not old
        fresh = lender.poll_resume_bounds(again)
        assert not lease.return_resumes_at_end, "a retired cache's lease resumes at its end"
        bounds = kit.return_write(lender, lease, ALL_ARRIVED)
        assert bounds is None and kit.closed(old) and lease._released
        assert lender.held_page_counts().unreclaimable == {}
        assert lender.poll_resume_bounds(again) == fresh, "a retired cache's rows counted"
        lease, _ = ready_write(lender, mgr, again)
        bounds = kit.return_write(lender, lease, ALL_ARRIVED)
        assert bounds is not None and bounds.max_position == PROMPT, bounds
        mgr._stream.synchronize()
        mgr.free_resources(again)
        gone = receiving(kit, mgr, 8, hybrid=hybrid)
        lease, _ = ready_write(lender, mgr, gone)
        mgr._stream.synchronize()
        mgr.free_resources(gone)
        assert lender.return_writes([(lease, ALL_ARRIVED, True)]) == [None]
        with pytest.raises(ValueError, match="has no KV cache"):
            lender.poll_resume_bounds(gone)


@gpu
def test_staged_and_in_place_leases_return_in_one_batch(kit, real_manager):
    """The in-place fetch of a batch settles in the call, the staged one after its copy's round."""
    with real_manager() as mgr:
        lender = attach(mgr)
        in_place = receiving(kit, mgr, 7)
        staged = kit.admitted(mgr, 8, OTHER_TOKENS)
        place, _ = ready_write(lender, mgr, in_place)
        stage, view = ready_write(lender, mgr, staged, end=3 * TPB, memory_kind=HOST, rows=NAMED)
        assert place.return_resumes_at_end and not stage.return_resumes_at_end
        bounds = lender.return_writes([(place, ALL_ARRIVED, True), (stage, ALL_ARRIVED, True)])
        assert bounds[0] is not None and bounds[0].max_position == PROMPT, bounds
        settled = bounds[1]
        for _ in range(8):
            if settled is not None:
                break
            mgr._stream.synchronize()
            lender.progress()
            settled = lender.poll_resume_bounds(staged)
        assert settled is not None and settled.max_position >= 3 * TPB, settled
        mgr._stream.synchronize()
        for r in (in_place, staged):
            mgr.free_resources(r)


@gpu
@pytest.mark.parametrize("request_count", [1, 3], ids=["single", "batch"])
def test_an_error_in_one_return_raises_once_every_entry_is_returned(
    kit, real_manager, monkeypatch, request_count
):
    """An unlock error still returns all entries; the caller then drops fetched caches."""
    with real_manager() as mgr:
        lender = attach(mgr)
        requests = [receiving(kit, mgr, rid) for rid in range(7, 7 + request_count)]
        leases = [ready_write(lender, mgr, r)[0] for r in requests]
        broken = kit.kv(mgr, requests[0])
        retry = mgr._commit_held_back

        def failing(kv):
            if kv is broken:
                raise RuntimeError("the retried commit failed")
            return retry(kv)

        monkeypatch.setattr(mgr, "_commit_held_back", failing)
        # The unlock retries a commit only where the manager holds one back for the cache.
        monkeypatch.setitem(mgr._held_commits, id(broken), (broken, requests[0], 0, False))
        with pytest.raises(RuntimeError, match="the retried commit failed"):
            lender.return_writes([(lease, ALL_ARRIVED, True) for lease in leases])
        assert all(lease._released for lease in leases)
        assert not lender._locks.holds_any()
        mgr._stream.synchronize()
        for r in requests:
            mgr.free_resources(r)


class Interrupted(BaseException):
    """An interrupt, as ``KeyboardInterrupt`` is, raised where a call runs."""


@gpu
@pytest.mark.parametrize("error", [MemoryError, Interrupted])
def test_a_return_whose_count_raises_still_returns_its_lease(kit, real_manager, monkeypatch, error):
    """A copy error still returns every lease; the caller drops caches without querying bounds."""
    with real_manager() as mgr:
        lender = attach(mgr)
        in_place = receiving(kit, mgr, 7)
        staged = kit.admitted(mgr, 8, OTHER_TOKENS)
        stage, _ = ready_write(lender, mgr, staged, end=3 * TPB, memory_kind=HOST, rows=NAMED)
        place, _ = ready_write(lender, mgr, in_place)
        queue, tried = lender._queue_copy, []

        def failing(segments, to_staging, after=None):
            if not to_staging:
                tried.append(len(segments))
                raise error("the copy into the pages failed")
            return queue(segments, to_staging, after=after)

        monkeypatch.setattr(lender, "_queue_copy", failing)
        with pytest.raises(error, match="into the pages"):
            lender.return_writes([(stage, ALL_ARRIVED, True), (place, ALL_ARRIVED, True)])
        assert tried and tried[0] > 0, "inconclusive: no copy into the pages was queued"
        assert stage._released and place._released, "a return left its lease open"
        assert not lender._locks.holds_any()
        mgr._stream.synchronize()
        for r in (in_place, staged):
            mgr.free_resources(r)


@gpu
def test_a_release_whose_count_raises_still_ends_its_lock(kit, real_manager, monkeypatch):
    """A release whose count of arrivals raises still ends the lease's lock before raising."""
    with real_manager() as mgr:
        lender = attach(mgr)
        request = receiving(kit, mgr, 7)
        lease, _ = ready_write(lender, mgr, request)

        def failing(*args, **kwargs):
            raise MemoryError("counting the arrivals failed")

        monkeypatch.setattr(lender, "_apply_in_place_marks", failing)
        assert lender._locks.holds_any()
        with pytest.raises(MemoryError, match="counting the arrivals"):
            lease.release()
        assert not lender._locks.holds_any(), "the release kept the lease's lock"
        mgr._stream.synchronize()
        mgr.free_resources(request)


PAGE_READS = ("pages", "page_list", "locked_pages", "state_slot")


@pytest.fixture
def page_reads(monkeypatch):
    """Calls of the manager facade's page and state-slot reads, by name, while counting is on."""
    counts = {name: 0 for name in PAGE_READS}
    counting = [False]
    for name in PAGE_READS:
        read = getattr(_manager, name)

        def counted(*args, _read=read, _name=name, **kwargs):
            if counting[0]:
                counts[_name] += 1
            return _read(*args, **kwargs)

        monkeypatch.setattr(_manager, name, counted)
    return counts, counting


@gpu
@pytest.mark.parametrize("hybrid", [False, True], ids=["paged", "hybrid"])
def test_a_fast_write_lend_and_its_return_ask_the_cache_for_no_page(
    kit, real_manager, hybrid_manager, page_reads, fast_returns, hybrid
):
    """A fast write lend and its return of every row read no page and no state slot of the cache."""
    counts, counting = page_reads
    with hybrid_manager() if hybrid else real_manager() as mgr:
        lender = attach(mgr)
        recurrent = sum(lender._layout.recurrent)
        request = receiving(kit, mgr, 7, hybrid=hybrid)
        counting[0] = True
        lease = lender.lend_write(request, 0, PROMPT, **kit.DEVICE_ALL)
        assert lease._block_entries.spans is not None, "inconclusive: the lend was not fast"
        assert sum(counts.values()) == 0, counts
        assert recurrent == hybrid, "inconclusive: no recurrent state where expected"
        for name in counts:
            counts[name] = 0
        assert lease.poll() is not None
        lender.return_writes([(lease, ALL_ARRIVED, True)])
        lender.progress()
        counting[0] = False
        assert sum(counts.values()) == 0, counts
        assert fast_returns == {"recorded": 0, "bounds": 1}, fast_returns
        mgr._stream.synchronize()
        mgr.free_resources(request)


@gpu
@skip_pre_blackwell
@pytest.mark.parametrize("tokens", [128 + 5, 3 * 128, 7 * 128 + 1])
def test_deepseek_v4_windowed_groups_take_the_fast_paths_as_the_rules_lend_and_return(
    kit, deepseek_v4_manager, fast_returns, monkeypatch, tokens
):
    """DeepSeek-V4's windowed layer groups lend and return fast as the rules do."""
    taken = {"read": 0, "write": 0}
    for name in ("_read_every_row_fast", "_write_every_row_fast"):
        fast = getattr(_lender._ManagerLender, name)

        def counted(self, *args, _fast=fast, _kind=name.split("_")[1], **kwargs):
            lease = _fast(self, *args, **kwargs)
            taken[_kind] += lease is not None
            return lease

        monkeypatch.setattr(_lender._ManagerLender, name, counted)
    prompt = list(range(20000, 20000 + tokens))
    outcomes = []
    for rules in (False, True):
        with deepseek_v4_manager() as mgr:
            lender = attach(mgr, rules=rules, staging_tokens=128)
            assert any(w is not None for w in lender._layout.windows), "inconclusive: no window"
            assert lender._layout.stale_ranges_by_formula
            source = kit.published(mgr, 9, prompt)
            read = lender.lend_read(
                source, 0, tokens, **{**kit.DEVICE_ALL, "ready_event": kit.stream_event(mgr)}
            )
            read_view = kit.ready_view(read, mgr)
            target = receiving(kit, mgr, 7, prompt)
            write, view = ready_write(lender, mgr, target)
            key = [
                [(rows.layer_group_index, rows.page_ordinals.tolist()) for rows in v.groups]
                for v in (read_view, view)
            ]
            locks = [
                {lg: b.tolist() for lg, b in lease._lock.blocks.items()} for lease in (read, write)
            ]
            bounds = lender.return_writes([(write, ALL_ARRIVED, True)])
            outcomes.append((key, locks, bounds, records(lender, 7)))
            read.release()
            mgr._stream.synchronize()
            for r in (source, target):
                mgr.free_resources(r)
    assert outcomes[0] == outcomes[1]
    assert taken == {"read": 1, "write": 1}, taken
    assert fast_returns == {"recorded": 1, "bounds": 1}, fast_returns
    assert outcomes[0][2][0].max_position == tokens


@gpu
def test_a_staged_read_becomes_ready_only_after_a_round_of_progress(kit, real_manager):
    """A staged read gives its view once a round of progress saw its copy done, never by polling."""
    with real_manager() as mgr:
        lender = attach(mgr)
        source = kit.published(mgr, 9, TOKENS)
        with ExitStack() as held:
            held.enter_context(kit.held_stream(mgr._stream, strict=False))
            lease = lender.lend_read(source, 0, 3 * TPB, **kit.HOST_NAMED)
            assert lease.failure is None, lease.failure
            assert lease.poll() is None
        mgr._stream.synchronize()
        for _ in range(3):
            assert lease.poll() is None, "a poll advanced the lender"
        lender.progress()
        assert lease.poll() is not None
        lease.release()
        mgr._stream.synchronize()
        mgr.free_resources(source)


@gpu
def test_a_new_lender_counts_the_pages_its_fast_lend_keeps_and_locks(kit, real_manager):
    """A fast write lend's pages count as locked, and all its cache's once a free keeps it open."""
    with real_manager() as mgr:
        lender = attach(mgr)
        request = receiving(kit, mgr, 7)
        lease, view = ready_write(lender, mgr, request)
        assert lender._locks.holds_any(), "inconclusive: the lend took no lock"
        kv = kit.kv(mgr, request)
        held = {}
        for lg, group in enumerate(kit.pool_group_of(mgr)):
            held[group] = held.get(group, 0) + sum(page >= 0 for page in kit.pages(kv, lg))
        locked = dict(lender.held_page_counts().by_reason.get(HoldReason.LOCKED, {}))
        assert sum(locked.values()) == sum(len(rows) for rows in view.groups) > 0, locked
        mgr._stream.synchronize()
        mgr.free_resources(request)
        assert not kit.closed(kv), "the free closed a cache whose pages are lent"
        assert dict(lender.held_page_counts().deduct) == {g: n for g, n in held.items() if n}
        lender.return_writes([(lease, ALL_ARRIVED, True)])
        assert kit.closed(kv) and dict(lender.held_page_counts().deduct) == {}


@gpu
@pytest.mark.parametrize("hybrid", [False, True], ids=["paged", "hybrid"])
def test_a_whole_return_holds_its_record_until_a_reader_needs_it(
    kit, real_manager, hybrid_manager, fast_returns, monkeypatch, hybrid
):
    """A fast whole return holds its record: rounds keep it, a reader writes it, a free drops it."""
    with hybrid_manager() if hybrid else real_manager() as mgr:
        lender = attach(mgr)
        ledger = lender._ledger
        request = receiving(kit, mgr, 7, hybrid=hybrid)
        lease, _ = ready_in_place(kit, lender, request)
        bounds = kit.return_write(lender, lease, ALL_ARRIVED)
        assert fast_returns == {"recorded": 0, "bounds": 1}, fast_returns
        assert 7 in ledger._pending and 7 not in ledger._delivered, "the return wrote its record"
        assert ledger._fetch_settled(ledger._latest_fetches[7]), "the fetch settled later"
        assert ledger._has_records(7)
        rounds = lender._progress_round
        with monkeypatch.context() as patched:
            patched.setattr(ledger, "_write_pending", lambda: pytest.fail("a round wrote a record"))
            lender.progress()
            lender.progress()
        assert lender._progress_round == rounds + 2 and 7 in ledger._pending
        assert lender.poll_resume_bounds(request) == bounds
        assert not ledger._pending and 7 in ledger._delivered, "the reader left the record held"
        assert fast_returns == {"recorded": 1, "bounds": 1}, fast_returns
        other = receiving(kit, mgr, 8, hybrid=hybrid)
        lease, _ = ready_in_place(kit, lender, other)
        lender.return_writes([(lease, ALL_ARRIVED, True)])
        assert 8 in ledger._pending, "inconclusive: the return held no record"
        mgr._stream.synchronize()
        mgr.free_resources(other)
        assert not ledger._pending and not ledger._has_records(8), "the free kept a held record"
        assert fast_returns == {"recorded": 1, "bounds": 2}, fast_returns
        mgr.free_resources(request)
        assert not ledger._has_records(7)


@gpu
def test_a_held_record_names_its_request_without_its_fetch(kit, real_manager):
    """A held whole delivery names its request and cache, and goes with the request's deliveries."""
    with real_manager() as mgr:
        lender = attach(mgr)
        ledger = lender._ledger
        request = receiving(kit, mgr, 7)
        lease, _ = ready_in_place(kit, lender, request)
        lender.return_writes([(lease, ALL_ARRIVED, True)])
        # The held record alone left to name the request.
        fetch = ledger._latest_fetches.pop(7)
        ledger._kept_prefixes.pop(7, None)
        assert 7 in ledger._pending and 7 not in ledger._delivered
        assert 7 not in ledger._mixed_tail_caches
        assert ledger._has_records(7), "a held record does not name its request"
        assert ledger._delivered_kv(7) is fetch.kv
        ledger._drop_deliveries(7)
        assert 7 not in ledger._deliveries and ledger._delivered_kv(7) is None
        mgr._stream.synchronize()
        mgr.free_resources(request)
        assert not ledger._has_records(7) and not ledger._pending


@gpu
@pytest.mark.parametrize(
    "reader", ["second_fetch", "masks", "partial", "staged_none", "free", "shrink", "read_in_place"]
)
def test_what_reads_a_held_record_sees_it_written(kit, real_manager, reader):
    """Every reader of a held record sees what the rules, which write it at once, would give."""
    outcomes = []
    for rules in (False, True):
        with real_manager() as mgr:
            lender = attach(mgr, rules=rules)
            request = receiving(kit, mgr, 7)
            first, _ = ready_in_place(kit, lender, request)
            outcome = [lender.return_writes([(first, ALL_ARRIVED, True)])]
            if reader in ("second_fetch", "masks", "partial"):
                second, view = ready_in_place(kit, lender, request, start=TPB)
                arrived = ALL_ARRIVED if reader == "second_fetch" else view.row_masks(True)
                if reader == "partial":
                    for mask in arrived:
                        mask[-1:] = False
                outcome.append(lender.return_writes([(second, arrived, True)]))
            elif reader == "staged_none":
                staged, view = ready_write(
                    lender, mgr, request, end=3 * TPB, memory_kind=HOST, rows=NAMED
                )
                outcome.append(lender.return_writes([(staged, view.row_masks(False), True)]))
            elif reader == "shrink":
                # The manager's report that the cache may have shrunk in place.
                lender._on_shrink(7, kit.kv(mgr, request))
            elif reader == "read_in_place":
                read = lender.lend_read(request, 0, 3 * TPB, **kit.DEVICE_ALL)
                outcome.append(read.failure)
                read.release()
            mgr._stream.synchronize()
            if reader == "free":
                mgr.free_resources(request)
                outcome.append(lender._ledger._has_records(7))
            else:
                outcome.append(lender.poll_resume_bounds(request))
            outcome.append(records(lender, 7))
            outcomes.append(outcome)
            if reader != "free":
                mgr.free_resources(request)
    assert outcomes[0] == outcomes[1]
    if reader == "free":
        assert outcomes[0][1] is False, outcomes[0]


@gpu
def test_a_shrink_hook_that_raises_drops_a_held_record(kit, real_manager, monkeypatch):
    """A shrink report whose own work raises drops the request's deliveries, a held record too."""
    with real_manager() as mgr:
        lender = attach(mgr)
        request = receiving(kit, mgr, 7)
        lease, _ = ready_in_place(kit, lender, request)
        lender.return_writes([(lease, ALL_ARRIVED, True)])
        assert 7 in lender._ledger._pending, "inconclusive: the return held no record"
        raised = []

        def num_blocks(cache):
            raised.append(cache)
            raise MemoryError("the block count failed")

        monkeypatch.setattr(_lender._manager, "num_blocks", num_blocks)
        lender._on_shrink(7, kit.kv(mgr, request))
        monkeypatch.undo()
        assert raised, "inconclusive: the hook's work did not raise"
        assert 7 not in lender._ledger._deliveries, "the rows the shrink freed stay delivered"
        mgr._stream.synchronize()
        mgr.free_resources(request)


@gpu
@pytest.mark.parametrize("hybrid", [False, True], ids=["paged", "hybrid"])
def test_a_write_that_resumes_at_its_end_returns_bounds_holding_it(
    kit, real_manager, hybrid_manager, monkeypatch, hybrid
):
    """A whole return of a write that says it resumes at its end gives bounds holding that end."""
    calls = []
    for name, value in list(vars(_manager).items()):
        if not isinstance(value, type) and getattr(value, "__module__", None) == _manager.__name__:

            def counted(*args, _value=value, _name=name, **kwargs):
                calls.append(_name)
                return _value(*args, **kwargs)

            monkeypatch.setattr(_manager, name, counted)
    ends = [PROMPT] if hybrid else [PROMPT, 2 * TPB, 3 * TPB]
    said = []
    with hybrid_manager() if hybrid else real_manager() as mgr:
        lender = attach(mgr)
        rid = 20
        for admit in ("received", "admitted"):
            for start in (0, TPB):
                for end in ends:
                    rid += 1
                    if admit == "received":
                        request = receiving(kit, mgr, rid, hybrid=hybrid)
                    else:
                        request = kit.admitted(mgr, rid, TOKENS)
                        assert mgr.resize_context(request, request.context_remaining_length)
                    lease = lender.lend_write(request, start, end, **kit.DEVICE_ALL)
                    assert not lease.return_resumes_at_end, "a lease before its view says so"
                    if lease.poll() is not None:
                        before = len(calls)
                        at_end = lease.return_resumes_at_end
                        assert len(calls) == before, calls[before:]
                        bounds = kit.return_write(lender, lease, ALL_ARRIVED)
                        if at_end:
                            assert bounds is not None, (admit, start, end)
                            assert bounds.min_position <= end <= bounds.max_position, (
                                admit,
                                start,
                                end,
                                bounds,
                            )
                        assert not lease.return_resumes_at_end, "a returned lease says so"
                        said.append(at_end)
                    else:
                        lease.release()
                    mgr._stream.synchronize()
                    mgr.free_resources(request)
    assert said.count(True) >= 2 and said.count(False) >= (0 if hybrid else 2), said


@gpu
@pytest.mark.parametrize("event", ["shrink", "free", "reset", "stop", "shutdown"])
def test_a_lease_stops_resuming_at_its_end_once_its_cache_can_move(
    kit, real_manager, monkeypatch, event
):
    """A shrink, a free, a reuse reset, the lender's stop or the shutdown ends it for good."""
    with real_manager() as mgr:
        lender = attach(mgr)
        request = receiving(kit, mgr, 7)
        lease, _ = ready_in_place(kit, lender, request)
        assert lease.return_resumes_at_end, "inconclusive: the lease does not resume at its end"
        mgr._stream.synchronize()
        if event == "shrink":
            lender._on_shrink(7, kit.kv(mgr, request))
        elif event == "free":
            mgr.free_resources(request)
        elif event == "reset":
            # The manager's report; its runtime resets nothing while a cache is open.
            lender._on_reset()
        elif event == "stop":

            def failing(request_id):
                raise MemoryError("dropping the records failed")

            monkeypatch.setattr(lender._ledger, "_drop_records", failing)
            other = receiving(kit, mgr, 8)
            mgr.free_resources(other)
            assert lender._stop_reason is not None, "inconclusive: the lender did not stop"
        else:
            mgr.shutdown()
        assert not lease.return_resumes_at_end, event
        lease.release()
        if event not in ("free", "shutdown"):
            mgr.free_resources(request)


# -- the ledger's helpers ------------------------------------------------------------------------

LEDGER_TPB = 4
RUN = [False] * 2 + [True] * 6  # blocks 2 to 7 of 8


def ledger(recurrent=(False, False)):
    layout = SimpleNamespace(
        tokens_per_block=LEDGER_TPB, num_layer_groups=len(recurrent), recurrent=recurrent, cp=(0, 1)
    )
    return _fetch_ledger._FetchLedger(layout, lambda copy: True)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("recurrent", "computed", "known"),
    [
        ((False,), (7,), 7),
        ((False, False, False), (5, 9, 2), 2),
        ((False, True, False), (5, 1, 9), 5),
        ((True, False, False), (0, 9, 4), 4),
        ((True, True), (3, 8), 8),
    ],
)
def test_the_known_prefix_is_the_least_a_group_without_recurrent_state_computed(
    recurrent, computed, known
):
    """The least any group but those of recurrent state computed; with state alone, the most."""
    assert ledger(recurrent)._known(computed) == known


def request_with_runs(runs, length):
    """A request whose multimodal tokens form ``runs``; ``None`` for one without multimodal data."""
    if runs is None:
        return SimpleNamespace(py_multimodal_data=None)
    mask = np.zeros(length, dtype=np.int64)
    for begin, end in runs:
        mask[begin:end] = 1
    data = {"mm_bidirectional_blocks": True, "multimodal_embed_mask_cumsum": np.cumsum(mask)}
    return SimpleNamespace(py_multimodal_data=data)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("runs", "usable", "clipped", "floor"),
    [
        (None, 40, 40, 0),
        ([], 40, 40, 0),
        ([(10, 20)], 40, 40, 20),
        ([(10, 20)], 15, 10, 0),
        ([(2, 5), (10, 20)], 15, 10, 5),
        ([(2, 5), (10, 20)], 20, 20, 20),
        ([(2, 5), (30, 50)], 40, 30, 5),
    ],
)
def test_an_end_is_clipped_to_the_multimodal_runs_with_its_floor(runs, usable, clipped, floor):
    """An end inside a run drops to its start; the floor is the end of the last run below it."""
    runs = _manager.bidirectional_runs(request_with_runs(runs, 60))
    assert ledger((False,))._clip_to_bidirectional_runs(runs, usable) == (clipped, floor)


def whole(led, request_id, kv):
    """The record of a whole delivery of blocks 2 to 7 of both layer groups into ``kv``."""
    fetch = _fetch_ledger._FetchRecord(2 * LEDGER_TPB, kv, end=8 * LEDGER_TPB)
    led._record_whole_delivery(request_id, fetch, [(2, 8), (2, 8)])
    return led._deliveries[request_id]


def masks_of(delivered):
    return [mask.tolist() for mask in delivered.delivered_block_masks]


@pytest.mark.cpu_only
@pytest.mark.parametrize("change", ["shrink", "invalidate", "deliver"])
def test_a_change_to_one_record_of_a_run_leaves_the_others_as_they_were(change):
    """Records of one run share a read-only mask; a change to one leaves the other and the mask."""
    led = ledger()
    kv1, kv2 = SimpleNamespace(num_blocks=8), SimpleNamespace(num_blocks=8)
    one, two = whole(led, 1, kv1), whole(led, 2, kv2)
    shared = one.delivered_block_masks[0]
    assert two.delivered_block_masks[0] is shared, "inconclusive: the records share no mask"
    assert not shared.flags.writeable
    if change == "shrink":
        kv1.num_blocks = 5
        led._invalidate_delivered_blocks_past_capacity(one, kv1)
        expected = [RUN[:5] + [False] * 3] * 2
    elif change == "invalidate":
        ordinals = np.arange(2, 8)
        lease = SimpleNamespace(
            _request_id=1,
            _kv=kv1,
            _block_entries=SimpleNamespace(
                layer_group_indices=[0, 1], block_ordinals=[ordinals, ordinals]
            ),
        )
        copied = [ordinals == 2, np.zeros(6, dtype=bool)]
        led._invalidate_delivered_blocks(lease, copied)
        expected = [[False] * 3 + [True] * 5, RUN]
    else:
        # A later fetch into the same cache delivers blocks 0 and 1 of the first layer group.
        fetch = _fetch_ledger._FetchRecord(0, kv1, end=2 * LEDGER_TPB)
        entries = SimpleNamespace(
            layer_group_indices=[0, 1], block_ordinals=[np.array([0, 1]), np.array([0, 1])]
        )
        copied = [np.array([True, True]), np.array([False, False])]
        led._record_delivered_blocks(1, fetch, entries, copied)
        expected = [[True] * 8, RUN]
    assert masks_of(one) == expected
    assert masks_of(two) == [RUN, RUN], "the change reached another record of the run"
    assert shared.tolist() == RUN and not shared.flags.writeable


@pytest.mark.cpu_only
def test_the_shared_masks_hold_a_bounded_number_of_bytes(monkeypatch):
    """Masks of ever longer runs are the runs' blocks, and never hold more bytes than the bound."""
    monkeypatch.setattr(_fetch_ledger, "_run_masks", {})
    monkeypatch.setattr(_fetch_ledger, "_run_mask_bytes", 0)
    monkeypatch.setattr(_fetch_ledger, "_RUN_MASK_BYTES", 64)
    emptied = 0
    for last in range(1, 40):
        held = len(_fetch_ledger._run_masks)
        assert _fetch_ledger._run_mask(1, last).tolist() == [False] + [True] * (last - 1)
        emptied += len(_fetch_ledger._run_masks) <= held
        kept = sum(len(mask) for mask in _fetch_ledger._run_masks.values())
        assert _fetch_ledger._run_mask_bytes == kept <= 64
    assert emptied >= 5, emptied


# -- stand-in managers: kept and read windows, claimed prefixes, pipeline stages ------------------

LONG_PROMPT = 5 * TPB
RID = 7
KEPT, READ = 72, 64  # a window kept eight tokens wider than the one attention reads
GROUPS = 3
# CSA2 with block reuse on: the global layer group REQUIRED, the encoder's OPTIONAL and not
# selected, the decoder's sliding window PRIVATE.
GLOBAL_ONLY = (True, False, False)


class Event:
    """A stand-in CUDA event, complete once ``done`` is set; waiting on it fails the test."""

    def __init__(self, done=True):
        self.done = done

    def query(self):
        return self.done

    def synchronize(self):
        raise AssertionError("a lease waited on its event")


class Manager:
    """A stand-in manager: a layer group per window, each on a pool group of its own, the runtime's
    stale ranges for the windows, no block reuse; its caches by request."""

    tokens_per_block = TPB
    enable_block_reuse = False
    _can_publish_block_reuse = False
    kv_connector_manager = None
    _on_allocation_stream = frozenset()

    def __init__(self, windows=(None,) * GROUPS, sinks=None):
        sinks = sinks or (0,) * len(windows)
        cycles = [AttnLifeCycle.make(w, s * TPB, TPB) for w, s in zip(windows, sinks)]
        self.impl = SimpleNamespace(
            get_life_cycle_pool_group_indices=lambda: list(range(len(windows)))
        )
        self.kv_cache_map = {}

        def stale_block_range(lg, history):
            stale = cycles[lg].get_stale_range(history, TPB)
            return stale.beg, stale.end

        self._stale_block_range = stale_block_range

    def _commit_held_back(self, kv):
        """No commit is held back here."""


def layout(windows=(None,) * GROUPS, read_windows=(), sinks=None, named=None):
    count = len(windows)
    return SimpleNamespace(
        tokens_per_block=TPB,
        windows=tuple(windows),
        read_windows=tuple(read_windows),
        sink_blocks=tuple(sinks or (0,) * count),
        recurrent=(False,) * count,
        pool_groups=tuple(range(count)),
        pool_group_of=tuple(range(count)),
        num_layer_groups=count,
        named_groups=tuple(named or (True,) * count),
        cp=(0, 1),
        device_pools={g: (DevicePool(g, 0, (g + 1) << 30, 256, 16),) for g in range(count)},
    )


def lender_over(manager, lay, layers=None):
    count = lay.num_layer_groups
    layers = layers or [(lg,) for lg in range(count)]
    ident = Identity(b"scope", bytes(range(32)), layers, [(1, 0)] * count)
    ref = weakref.ref(manager)
    regions = tuple(_lender._device_regions(manager, lay, ident))
    return _lender._ManagerLender(ref, lay, ident, regions, SlotQueue({}), ref)


def cache(committed=0, complete=(), endpoint=None, history=LONG_PROMPT, blocks=5):
    """A stand-in active cache on pages ``10, 11, ...``, ``committed`` tokens committed, its claim
    ending at ``endpoint`` and complete in the layer groups ``complete`` flags."""
    pages = list(range(10, 10 + blocks))
    claim = committed if endpoint is None else endpoint
    return SimpleNamespace(
        Status=SimpleNamespace(OPEN=0, CLOSED=1),
        status=0,
        num_committed_tokens=committed,
        reuse_status=[SimpleNamespace(complete=c, endpoint=claim) for c in complete],
        history_length=history,
        is_active=True,
        num_blocks=blocks,
        tokens_per_block=TPB,
        enable_swa_scratch_reuse=False,
        get_base_page_indices=lambda lg: pages,
        get_aggregated_page_indices=lambda lg, valid_only=False: pages,
    )


def request(prompt=LONG_PROMPT):
    return SimpleNamespace(
        py_request_id=RID,
        prompt_len=prompt,
        py_multimodal_data=None,
        py_served_prefix_end=0,
        context_current_position=0,
        context_remaining_length=prompt,
    )


def return_write(lender, req, lease, arrived):
    """The bounds ``return_writes`` gives for ``lease``, which ``req``'s query gives too."""
    (bounds,) = lender.return_writes([(lease, arrived, True)])
    assert bounds == lender.poll_resume_bounds(req), bounds
    return bounds


def lent(lease):
    """The view of ``lease`` and its block ordinals per layer group."""
    view = lease.poll()
    assert view is not None, lease.failure
    return view, {run.layer_group_index: run.page_ordinals.tolist() for run in view.groups}


def unnamed_keys(lender, monkeypatch):
    """``lender`` lending by name keys of zeros, its request's input needing no name."""
    monkeypatch.setattr(lender, "_unnamed_input_refusal", lambda request: None)
    monkeypatch.setattr(
        lender, "_reuse_keys_for", lambda m, r, k, o: [np.zeros((len(v), 32), np.uint8) for v in o]
    )


@pytest.mark.cpu_only
def test_the_read_window_leaves_out_what_the_runtime_s_stale_range_for_it_does():
    """The read window's stale range is the runtime's own for a window of its length."""
    rng = random.Random(0)
    for _ in range(3000):
        read = rng.randint(1, 300)
        kept = read + rng.choice([0, 1, 2, 3, 8, 40])
        sinks = rng.choice([0, 0, 1, 2])
        history = rng.randint(0, 1200)
        manager = Manager(windows=(kept,), sinks=(sinks,))
        lay = layout((kept,), (read,), (sinks,))
        cycle = AttnLifeCycle.make(read, sinks * TPB, TPB)
        stale = cycle.get_stale_range(history, TPB)
        got = _ranges._read_stale_range(manager, lay, 0, history)
        if stale.end > stale.beg or got[1] > got[0]:
            assert got == (stale.beg, stale.end), (kept, read, sinks, history)
        block = rng.randint(sinks, 40)
        last = _ranges._last_history_reading(manager, lay, 0, block)
        if last >= block * TPB:
            assert cycle.get_stale_range(last, TPB).end <= block, (kept, read, sinks, block, last)
        assert cycle.get_stale_range(last + 1, TPB).end > block, (kept, read, sinks, block, last)


def fetch_window(kit, delivered, history=LONG_PROMPT, prompt=LONG_PROMPT, world=None):
    """Blocks a fetch of five lends under the window pair, and its bounds with ``delivered`` in."""
    manager = Manager(windows=(KEPT,))
    lender = lender_over(manager, layout((KEPT,), (READ,)))
    manager.kv_cache_map[RID] = cache(history=history)
    req = request(prompt)
    lease = lender.lend_write(req, 0, 5 * TPB, **kit.DEVICE_ALL)
    view = lease.poll()
    assert view is not None, lease.failure
    (run,) = view.groups
    bounds = return_write(lender, req, lease, [np.isin(run.page_ordinals, list(delivered))])
    if world is not None:
        world.update(manager=manager, lender=lender, kv=manager.kv_cache_map[RID])
    return run.page_ordinals.tolist(), tuple(bounds)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "delivered",
    [(3, 4), (2, 3, 4), (2, 4)],
    ids=["margin_missing", "every_row", "read_row_missing"],
)
def test_a_margin_row_that_did_not_arrive_lowers_no_bound_and_keeps_the_cache_out_of_reuse(
    kit, delivered
):
    """Block 2 of the kept window is a margin row the read window leaves out at the prompt's end."""
    world = {}
    lent_blocks, bounds = fetch_window(kit, delivered, world=world)
    assert lent_blocks == [2, 3, 4], f"the fetch lends what the kept window holds: {lent_blocks}"
    if delivered == (2, 4):
        assert bounds[0] < bounds[1] and bounds[0] < LONG_PROMPT, bounds
        return
    assert bounds == (LONG_PROMPT, LONG_PROMPT), bounds
    manager = world["manager"]
    manager._sharing = world["lender"]
    gate = _lender_hooks.commit_gate(manager, world["kv"], None, LONG_PROMPT)
    assert gate == (_lender_hooks.REFUSE if delivered == (3, 4) else _lender_hooks.COMMIT), gate


@pytest.mark.cpu_only
def test_a_resume_below_the_end_starts_where_the_read_window_leaves_the_margin_behind(
    kit, monkeypatch
):
    """A resume below the end starts past 158, the last history reading the undelivered block 2."""
    monkeypatch.setattr(_ranges, "_floor_follows_history", lambda manager: False)
    lent_blocks, bounds = fetch_window(kit, (3, 4), history=0, prompt=LONG_PROMPT + 1)
    assert lent_blocks == [2, 3, 4], lent_blocks
    assert bounds == (LONG_PROMPT, LONG_PROMPT - 1), bounds


@pytest.mark.cpu_only
def test_a_group_counts_a_claimed_prefix_as_native_s_transfer_does_and_its_own_commits_always():
    """Each group's committed blocks of a claim are what native's transfer counts as cached."""
    from tensorrt_llm._torch.disaggregation.resource.cache_reuse import _CacheReuseAdapterV2
    from tensorrt_llm._torch.disaggregation.resource.page import AttentionLayerGroup

    manager = Manager()
    manager.enable_block_reuse = True
    adapter = _CacheReuseAdapterV2(manager)
    groups = [AttentionLayerGroup(pool_group_idx=g, kv_head_num_per_rank=1) for g in range(GROUPS)]
    req = request()
    rng = random.Random(0)
    for _ in range(2000):
        endpoint = rng.choice([0, rng.randint(1, 8) * TPB, rng.randint(1, 8 * TPB)])
        committed = rng.choice([0, endpoint, endpoint // TPB * TPB, rng.randint(0, 9 * TPB)])
        complete = [endpoint > 0 and rng.random() < 0.6 for _ in range(GROUPS)]
        kv = cache(committed, complete, endpoint)
        got = _manager.committed_by_group(kv, GROUPS)
        if committed <= endpoint:
            manager.kv_cache_map[RID] = kv
            want = adapter.get_cached_token_count_per_layer_group(req, groups)
            assert [t // TPB * TPB for t in got] == want, (committed, endpoint, complete, got)
        else:
            assert got == (committed,) * GROUPS, (committed, endpoint, complete, got)


@pytest.mark.cpu_only
@pytest.mark.parametrize("history", [LONG_PROMPT, 2 * TPB], ids=["generation_init", "context"])
@pytest.mark.parametrize("missing", [None, 1, 0], ids=["all_delivered", "private_hole", "global"])
def test_each_layer_group_is_fetched_from_its_own_committed_blocks(kit, missing, history):
    """A claim complete in the global group alone: it is fetched from block 2, the others from 0."""
    manager = Manager()
    lender = lender_over(manager, layout())
    kv = manager.kv_cache_map[RID] = cache(2 * TPB, GLOBAL_ONLY, history=history)
    lease = lender.lend_write(request(), 0, LONG_PROMPT, **kit.DEVICE_ALL)
    view, blocks = lent(lease)
    assert blocks == {0: [2, 3, 4], 1: [0, 1, 2, 3, 4], 2: [0, 1, 2, 3, 4]}, blocks
    assert {lg: b.tolist() for lg, b in lease._lock.blocks.items()} == blocks
    masks = view.row_masks(True)
    if missing is not None:
        masks[missing][0] = False
    bounds = return_write(lender, request(), lease, masks)
    if missing == 1:
        assert bounds.max_position == 0, f"a group without block 0 resumes: {bounds}"
    elif missing == 0:
        assert bounds.max_position == 2 * TPB, f"block 2 of the global group: {bounds}"
    else:
        assert tuple(bounds) == (LONG_PROMPT, history), bounds
    assert kv.history_length == history


@pytest.mark.cpu_only
def test_blocks_the_cache_committed_past_its_claim_are_fetched_in_no_layer_group(kit):
    """Blocks committed past a claim the global group alone took whole are held in every group."""
    manager = Manager()
    lender = lender_over(manager, layout())
    manager.kv_cache_map[RID] = cache(3 * TPB, GLOBAL_ONLY, endpoint=2 * TPB)
    lease = lender.lend_write(request(), 0, LONG_PROMPT, **kit.DEVICE_ALL)
    _, blocks = lent(lease)
    assert blocks == {lg: [3, 4] for lg in range(GROUPS)}, blocks
    lease.release()


@pytest.mark.cpu_only
def test_a_write_lock_keeps_off_each_layer_group_s_own_committed_blocks():
    manager = Manager()
    lender = lender_over(manager, layout())
    kv = cache(2 * TPB, GLOBAL_ONLY)
    refusal = lender._locks.refusal(kv, {0: [1], 1: [0]}, LockMode.WRITE)
    assert refusal is not None and "layer group 0" in refusal, refusal
    assert lender._locks.refusal(kv, {0: [2], 1: [0], 2: [1]}, LockMode.WRITE) is None


@pytest.mark.cpu_only
def test_a_named_read_lends_no_block_a_layer_group_does_not_hold(kit, monkeypatch):
    """A read by name of the two claimed blocks lends them in the global group alone."""
    manager = Manager()
    lender = lender_over(manager, layout())
    manager.kv_cache_map[RID] = cache(2 * TPB, GLOBAL_ONLY)
    unnamed_keys(lender, monkeypatch)
    lease = lender.lend_read(request(), 0, 2 * TPB, **kit.DEVICE_NAMED)
    _, blocks = lent(lease)
    assert blocks == {0: [0, 1], 1: [], 2: []}, blocks
    lease.release()


@pytest.mark.cpu_only
def test_an_undelivered_block_taints_only_where_a_group_s_committed_tokens_end_inside_it(kit):
    """A block missing where a group committed none of its tokens only keeps the resume below it."""
    manager = Manager()
    lender = lender_over(manager, layout())
    manager.kv_cache_map[RID] = cache(2 * TPB + 5, GLOBAL_ONLY)
    lease = lender.lend_write(request(), 0, LONG_PROMPT, **kit.DEVICE_ALL)
    view, blocks = lent(lease)
    assert blocks == {0: [2, 3, 4], 1: [0, 1, 2, 3, 4], 2: [0, 1, 2, 3, 4]}, blocks
    masks = view.row_masks(True)
    masks[1][2] = False
    assert return_write(lender, request(), lease, masks).max_position == 2 * TPB


@pytest.mark.cpu_only
def test_the_floor_reads_each_window_from_the_tokens_its_layer_group_computed(kit, monkeypatch):
    """A window whose group committed the blocks behind it lets a resume start at the history."""
    monkeypatch.setattr(_ranges, "_floor_follows_history", lambda manager: False)
    manager = Manager(windows=(64, None, None))
    lender = lender_over(manager, layout((64, None, None)))
    manager.kv_cache_map[RID] = cache(3 * TPB, (True, False, False), history=3 * TPB)
    lease = lender.lend_write(request(), 0, LONG_PROMPT, **kit.DEVICE_ALL)
    view, blocks = lent(lease)
    assert blocks == {0: [3, 4], 1: [0, 1, 2, 3, 4], 2: [0, 1, 2, 3, 4]}, blocks
    bounds = return_write(lender, request(), lease, view.row_masks(True))
    assert tuple(bounds) == (LONG_PROMPT, 3 * TPB), bounds


@pytest.mark.cpu_only
def test_a_window_s_unwritten_pages_are_found_from_the_tokens_its_layer_group_computed(
    monkeypatch,
):
    """A grow past what a window's group computed refuses, naming that group."""
    monkeypatch.setattr(_manager, "keeps_stale_pages", lambda manager: True)
    manager = Manager(windows=(64, 64, None))
    lender = lender_over(manager, layout((64, 64, None)))
    kv = manager.kv_cache_map[RID] = cache(2 * TPB, (True, False, False), history=2 * TPB)
    lender._ledger._kept_prefixes[RID] = (kv, (0, 0, 0), 2 * TPB)
    assert lender._ledger._computed_tokens(RID, kv) == (2 * TPB, 0, 0)
    refusal = lender._unwritten_pages_refusal(manager, RID, kv, 4 * TPB)
    assert refusal is not None and refusal.startswith("layer group 1 "), refusal


@pytest.mark.cpu_only
@pytest.mark.parametrize("memory", [HOST, DEVICE])
def test_named_reads_exclude_private_rows_even_after_the_global_commit_advanced(
    kit, monkeypatch, memory
):
    manager = Manager()
    lender = lender_over(manager, layout(named=(True, True, False)))
    manager.kv_cache_map[RID] = cache(2 * TPB, (False,) * GROUPS, endpoint=0)
    unnamed_keys(lender, monkeypatch)
    admitted = []
    if memory is HOST:
        monkeypatch.setattr(lender, "_fits_refusal", lambda counts: None)
        monkeypatch.setattr(lender, "_admit", lambda lease, counts: admitted.append(lease))
    named = kit.HOST_NAMED if memory is HOST else kit.DEVICE_NAMED
    lease = lender.lend_read(request(), 0, 2 * TPB, **named)
    if memory is HOST:
        assert admitted == [lease] and lease.failure is None
        entries = lease._block_entries
        blocks = {
            lg: o.tolist() for lg, o in zip(entries.layer_group_indices, entries.block_ordinals)
        }
    else:
        _, blocks = lent(lease)
    assert blocks == {0: [0, 1], 1: [0, 1], 2: []}
    lease.release()


@pytest.mark.cpu_only
@pytest.mark.parametrize("memory", [HOST, DEVICE])
def test_named_writes_lend_no_rows_of_a_private_layer_group(kit, monkeypatch, memory):
    """A named fetch lends a PRIVATE layer group nothing in either memory; the others arrive."""
    manager = Manager()
    lender = lender_over(manager, layout(named=(True, True, False)))
    lender._slot_queue = SlotQueue({g: 4 for g in range(GROUPS)})
    manager.kv_cache_map[RID] = cache(2 * TPB, (False,) * GROUPS, endpoint=0, history=2 * TPB)
    unnamed_keys(lender, monkeypatch)
    monkeypatch.setattr(lender, "_returns_context_outputs", lambda request: False)
    monkeypatch.setattr(_manager, "grow", lambda manager, request, kv, position, end: True)
    monkeypatch.setattr(
        lender,
        "_queue_copy",
        lambda segments, to_staging, after=None: (_lender._QueuedCopy(Event()), None),
    )
    named = kit.HOST_NAMED if memory is HOST else kit.DEVICE_NAMED
    lease = lender.lend_write(request(), 0, 4 * TPB, **named)
    view, blocks = lent(lease)
    assert blocks == {0: [2, 3], 1: [2, 3], 2: []}, blocks
    assert [len(run.object_keys) for run in view.groups] == [2, 2, 0]
    if memory is HOST:
        assert [lender._free_slots(g) for g in range(GROUPS)] == [2, 2, 4]
    lender.return_writes([(lease, view.row_masks(True), True)])
    delivered = lender._ledger._deliveries_of(RID).delivered_block_masks
    assert [m.nonzero()[0].tolist() for m in delivered] == [[2, 3], [2, 3], []]


BLOCKS = -(-PROMPT // TPB)
STEPS = (TPB, 2 * TPB, PROMPT)  # the context's steps: the position each computes through


class Stage:
    """One pipeline stage, ``lag`` iterations behind the first in processing each context step: its
    manager, lender and request, the event recorded after its last step, and its lease."""

    def __init__(self, index, lag=0):
        self.index = index
        self.lag = lag
        self.manager = Manager(windows=(None,))
        self.lender = lender_over(self.manager, layout((None,)), [(index,)])
        self.request = request(PROMPT)
        self.ready = None
        self.lease = None
        self.seen = False  # a poll returned the view; the backend's access ends one iteration later
        self.released_in = None

    def process(self, iteration):
        """The context step this stage reaches in ``iteration``, and the event recorded after it."""
        step = iteration - self.lag
        if 0 <= step < len(STEPS):
            self.manager.kv_cache_map[RID].history_length = STEPS[step]
            self.request.context_current_position = STEPS[step]
            self.request.context_remaining_length = PROMPT - STEPS[step]
            self.ready = Event()

    def processed_the_prompt(self):
        return self.manager.kv_cache_map[RID].history_length == PROMPT

    def status(self):
        """This stage's part of the per-iteration allgather: delivered once its backend's access
        to a ready view ended, failed once its lease failed, else pending."""
        if self.lease is None:
            return "pending"
        view = self.lease.poll()  # every open lease, every iteration
        if self.lease.failure is not None:
            return "failed"
        if view is None:
            return "pending"
        if self.seen:
            return "delivered"
        self.seen = True
        return "pending"


def agree(stages, iteration):
    """Once every stage's lease settled, the outcome and every stage's release in this iteration."""
    statuses = [stage.status() for stage in stages]
    if "pending" in statuses:
        return None
    for stage in stages:
        stage.lease.release()
        stage.released_in = iteration
    return "delivered" if set(statuses) == {"delivered"} else "failed"


def send_the_prompt(stages, early_release=None, lend_together=False):
    """Each stage lends a read of the prompt once it processed it (``lend_together``: once the first
    did), polled until the stages agree; ``early_release``'s stage releases its lease early."""
    for stage in stages:
        stage.manager.kv_cache_map[RID] = cache(history=0, blocks=BLOCKS)
    for iteration in range(len(STEPS) + max(stage.lag for stage in stages) + 4):
        for stage in stages:
            stage.process(iteration)
        ready = [stage.processed_the_prompt() for stage in stages]
        for stage in stages:
            due = any(ready) if lend_together else stage.processed_the_prompt()
            if stage.lease is None and due:
                stage.lease = stage.lender.lend_read(
                    stage.request, 0, PROMPT, memory_kind=DEVICE, rows=ALL, ready_event=stage.ready
                )
        outcome = agree(stages, iteration)
        if outcome is not None:
            return outcome
        early = stages[early_release] if early_release is not None else None
        if early is not None and early.seen and early.released_in is None:
            early.lease.release()  # its own backend is done, the other stage's is not
            early.released_in = iteration
    raise AssertionError("the stages never agreed")


@pytest.mark.cpu_only
def test_each_stage_lends_once_it_processed_the_step_and_all_release_in_one_iteration():
    stages = [Stage(0), Stage(1, lag=1)]
    assert send_the_prompt(stages) == "delivered"
    assert stages[0].released_in == stages[1].released_in is not None
    assert not any(stage.lender._is_locked(None) for stage in stages), "a lock outlived its lease"


@pytest.mark.cpu_only
def test_a_stage_that_has_not_processed_the_step_fails_its_lease_at_the_call():
    """A read lent past what a stage holds fails at the call; the stages agree it failed."""
    stages = [Stage(0), Stage(1, lag=1)]
    assert send_the_prompt(stages, lend_together=True) == "failed"
    assert "past the" in stages[1].lease.failure and stages[0].lease.failure is None
    assert stages[0].released_in == stages[1].released_in is not None


@pytest.mark.cpu_only
def test_a_lease_released_before_the_stages_agree_fails_the_driver_loudly():
    """The driver's next poll of a lease released before the stages agree raises."""
    stages = [Stage(0), Stage(1, lag=2)]
    with pytest.raises(RuntimeError, match="poll after release"):
        send_the_prompt(stages, early_release=0)
    assert stages[0].released_in is not None and stages[1].released_in is None
    if stages[1].lease is not None:
        stages[1].lease.release()


def fetch_the_prompt(kit, stages, delivered_blocks, histories=(PROMPT, PROMPT), end=PROMPT):
    """Each stage's write of ``[0, end)`` returned with ``delivered_blocks``; the bounds."""
    bounds = []
    for stage, blocks, history in zip(stages, delivered_blocks, histories):
        stage.manager.kv_cache_map[RID] = cache(history=history, blocks=BLOCKS)
        req = request(PROMPT)
        lease = stage.lender.lend_write(req, 0, end, **kit.DEVICE_ALL)
        view = lease.poll()
        assert view is not None, lease.failure
        arrived = [np.isin(run.page_ordinals, list(blocks)) for run in view.groups]
        bounds.append(return_write(stage.lender, req, lease, arrived))
    return bounds


def intersect(bounds):
    """The rule ``ResumeBounds`` states across ranks: the largest minimum, the smallest maximum."""
    return ResumeBounds(
        max_position=min(b.max_position for b in bounds),
        min_position=max(b.min_position for b in bounds),
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("partial", [False, True], ids=["every_block", "one_stage_short"])
def test_the_stages_resume_at_the_intersection_of_their_bounds(kit, partial):
    """A stage short of a block keeps both from resuming at the prompt: the bounds are disjoint."""
    stages = [Stage(0), Stage(1)]
    second = range(2) if partial else range(BLOCKS)
    bounds = fetch_the_prompt(kit, stages, [range(BLOCKS), second])
    agreed = intersect(bounds)
    if not partial:
        assert bounds == [(PROMPT, PROMPT)] * 2 and agreed == (PROMPT, PROMPT)
        return
    assert bounds[0] == (PROMPT, PROMPT), bounds[0]
    assert bounds[1].max_position < PROMPT, f"a short fetch resumes at the prompt: {bounds[1]}"
    assert agreed.max_position < agreed.min_position, f"the stages could diverge: {agreed}"


@pytest.mark.cpu_only
def test_a_stage_s_lender_refuses_a_lease_another_stage_lent(kit):
    """A lease another lender lent is refused and left to return through its own."""
    first, second = Stage(0), Stage(1)
    second.manager.kv_cache_map[RID] = cache(history=PROMPT, blocks=BLOCKS)
    lease = second.lender.lend_write(request(PROMPT), 0, PROMPT, **kit.DEVICE_ALL)
    assert lease.poll() is not None, lease.failure
    with pytest.raises(ValueError, match="not lent by this lender"):
        first.lender.return_writes([(lease, ALL_ARRIVED, True)])
    assert return_write(second.lender, request(PROMPT), lease, ALL_ARRIVED) == (PROMPT, PROMPT)


@pytest.mark.cpu_only
def test_pipeline_intersection_uses_the_larger_of_real_stage_lower_bounds(kit):
    bounds = fetch_the_prompt(kit, [Stage(0), Stage(1)], [range(BLOCKS)] * 2, (0, 64), end=96)
    assert bounds == [(96, 0), (96, 64)]
    assert intersect(bounds) == (96, 64)


@pytest.mark.cpu_only
@pytest.mark.parametrize("hybrid", [False, True], ids=["attention_model", "hybrid_model"])
def test_a_stage_of_a_model_with_recurrent_state_names_nothing_whatever_layers_it_holds(
    monkeypatch, hybrid
):
    """Every stage of a model with recurrent state names nothing; a model without names blocks."""
    stage = SimpleNamespace(cp=(0, 1), recurrent=(False, False))
    monkeypatch.setattr(_manager, "model_has_recurrent_state", lambda manager: hybrid)
    monkeypatch.setattr(_manager, "commits_blocks", lambda manager: True)
    monkeypatch.setattr(_manager, "prompt_lookahead", lambda manager: 0)
    monkeypatch.setattr(_manager, "has_lossy_cold_page_codec", lambda manager: False)
    reason = _lender._no_names(SimpleNamespace(), stage)
    if hybrid:
        assert reason is not None and "recurrent state" in reason, reason
    else:
        assert reason is None, reason


@pytest.mark.cpu_only
@pytest.mark.parametrize("feature", ["window", "state", "sparse"])
@pytest.mark.parametrize("pp_rank", [0, 1])
def test_attach_rejects_model_wide_features_on_every_pipeline_stage(monkeypatch, feature, pp_rank):
    """A stage without the unsupported layer rejects the same model as the stage holding it."""
    stage = SimpleNamespace(
        cp=(0, 2),
        num_layer_groups=1,
        recurrent=(False,),
        windows=(None,),
        sink_blocks=(0,),
        sparse=(False,),
    )
    manager = SimpleNamespace(
        mapping=SimpleNamespace(cp_size=2, pp_rank=pp_rank, has_cp_helix=lambda: True),
        _model_has_attention_window=feature == "window",
        _model_has_sparse_buffers=feature == "sparse",
        _model_attention_window_pattern=(),
        num_layers=2,
    )
    monkeypatch.setattr(_manager, "require_v2", lambda manager: None)
    monkeypatch.setattr(_manager, "layer_configs", lambda manager: {})
    monkeypatch.setattr(_manager, "layer_group_layer_ids", lambda manager: [])
    monkeypatch.setattr(_lender, "derive_layout", lambda manager: stage)
    monkeypatch.setattr(_manager, "model_has_recurrent_state", lambda manager: feature == "state")
    with pytest.raises(ValueError, match="sparse buffers" if feature == "sparse" else "Helix"):
        _lender._checked_layout(manager)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "pattern,num_layers,expected",
    [((256, 64), 2, True), ((256, 64), 1, False), ((256,), 2, False), ((), 2, False)],
)
def test_model_window_detection_keeps_windows_on_other_pipeline_stages(
    pattern, num_layers, expected
):
    manager = SimpleNamespace(
        max_seq_len=256,
        num_layers=num_layers,
        _model_has_attention_window=False,
        _model_has_sparse_buffers=False,
        _model_attention_window_pattern=pattern,
    )
    assert _manager.model_has_attention_window(manager) is expected


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "missing, detector",
    [
        ("_model_has_attention_window", _manager.model_has_attention_window),
        ("_model_attention_window_pattern", _manager.model_has_attention_window),
        ("num_layers", _manager.model_has_attention_window),
        ("_model_has_sparse_buffers", _manager.model_has_sparse_buffers),
    ],
)
def test_model_features_require_explicit_metadata(missing, detector):
    manager = SimpleNamespace(
        max_seq_len=256,
        num_layers=2,
        _model_has_attention_window=False,
        _model_has_sparse_buffers=False,
        _model_attention_window_pattern=(256, 64),
    )
    delattr(manager, missing)
    with pytest.raises(AttributeError, match=missing):
        detector(manager)
