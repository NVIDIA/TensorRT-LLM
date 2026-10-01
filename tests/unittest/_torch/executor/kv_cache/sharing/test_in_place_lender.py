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
"""The in-place lender over real KV cache managers, through the public API: views of a request's
own pages, the two explicit failures, and loans kept across the request's free and the manager's
shutdown. Oracles read pages and the allocator directly; each has a negative control."""

import gc
import threading
import weakref
from contextlib import contextmanager

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import (
    InPlaceLender,
    Lease,
    StagingLender,
    StagingOptions,
    attach_in_place,
    attach_staging,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="allocates KV cache pools")

TPB = 32
WINDOW = 64
PROMPT = list(range(1000, 1097))  # three whole blocks and a one-token tail: four pages
END = len(PROMPT)
WINDOWED_PROMPT = list(range(2000, 2161))  # five whole blocks and a one-token tail
WINDOWED_END = len(WINDOWED_PROMPT)
SOURCE, TARGET = 1, 2


def ceil_blocks(tokens: int) -> int:
    return -(-tokens // TPB)


def check_view(kit, view, kv, expected):
    """``expected``: layer group -> ordinals, every layer group listed. Rows carry no names,
    addresses or part, and each is a block with a page of its own."""
    assert {run.layer_group: run.ordinals.tolist() for run in view.runs} == {
        lg: list(ordinals) for lg, ordinals in expected.items()
    }
    for run in view.runs:
        assert run.names is None and run.addresses is None and run.part is None
        own = kit.pages(kv, run.layer_group)
        assert all(own[o] >= 0 for o in run.ordinals.tolist())


def in_turn(kit, call, name):
    """``call`` on its own thread, joined; like the executor's threads, it used CUDA before."""

    def run():
        torch.cuda.synchronize()
        return call()

    return kit.on_thread(run, name)


@contextmanager
def lent_and_freed(kit, mgr, lender, write=False):
    """``SOURCE`` computed ``PROMPT``, a lease lends all of it, and ``SOURCE`` is freed. Yields the
    lease, the freed cache and its pages."""
    request = kit.published(mgr, SOURCE, PROMPT)
    kv = kit.kv(mgr, request)
    lent = set(kit.pages(kv, 0))
    assert len(lent) == ceil_blocks(END) and kit.pool_pages(mgr) - len(lent) >= 1
    lease = (lender.lend_write if write else lender.lend_read)(request, 0, END)
    assert lease.poll() is not None
    mgr.free_resources(request)
    assert kit.kv(mgr, request) is None
    yield lease, kv, lent


# -- what it lends ----------------------------------------------------------------------------


def test_an_in_place_lender_lends_and_promises_nothing_more(real_manager):
    with real_manager() as mgr:
        lender = attach_in_place(mgr)
        assert isinstance(lender, InPlaceLender) and not isinstance(lender, StagingLender)
        assert not hasattr(lender, "readiness") and not hasattr(lender, "parts")
        with pytest.raises(ValueError, match="already attached"):
            attach_in_place(mgr)
        with pytest.raises(ValueError, match="already attached"):
            attach_staging(mgr, scope=b"scope", staging=StagingOptions(TPB))


def test_a_read_is_ready_at_its_first_poll_without_waiting_for_the_stream(kit, real_manager):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        kv = kit.kv(mgr, source)
        lender = attach_in_place(mgr)
        with kit.held_stream(mgr._stream) as gate:
            lease = lender.lend_read(source, 0, END)
            assert isinstance(lease, Lease)
            view = lease.poll()
            assert view is not None, "an in-place lease waited for the stream"
            assert lease.poll() is view
            gate.open()
        check_view(kit, view, kv, {0: range(ceil_blocks(END))})
        assert END % TPB and view.runs[0].ordinals[-1] == END // TPB, "the partial last block"
        pieces = [
            (lender.lend_read(source, TPB, 2 * TPB), [1]),
            (lender.lend_read(source, 2 * TPB, END), [2, 3]),
            (lender.lend_read(source, 5, 40), [0, 1]),  # any token range
        ]
        for piece, ordinals in pieces:
            check_view(kit, piece.poll(), kv, {0: ordinals})
        for held in [lease] + [piece for piece, _ in pieces]:
            held.release()
        with pytest.raises(RuntimeError):
            lease.poll()


def test_a_windowed_view_keeps_only_the_blocks_the_window_reads(kit, real_manager):
    with real_manager(windows=[WINDOW, 256]) as mgr:
        source = kit.published(mgr, SOURCE, WINDOWED_PROMPT)
        kv = kit.kv(mgr, source)
        lender = attach_in_place(mgr)
        windows = kit.windows(mgr)
        sliding, full = windows.index(WINDOW), windows.index(None)
        beg, end = kit.stale_blocks(mgr, sliding, WINDOWED_END)
        assert end > beg == 0, "the window must have left some blocks behind"
        blocks = range(ceil_blocks(WINDOWED_END))
        lease = lender.lend_read(source, 0, WINDOWED_END)
        in_window = [o for o in blocks if not beg <= o < end]
        check_view(kit, lease.poll(), kv, {full: blocks, sliding: in_window})
        lease.release()


def test_a_generation_view_leaves_out_paged_blocks_the_window_no_longer_reads(kit, real_manager):
    with real_manager(windows=[WINDOW, 256]) as mgr:
        request = kit.make_request(TARGET, WINDOWED_PROMPT)
        assert mgr.prepare_context(request)
        assert mgr.resize_context(request, request.context_remaining_length)
        kv = kit.kv(mgr, request)
        lender = attach_in_place(mgr)
        sliding = kit.windows(mgr).index(WINDOW)
        beg, end = kit.stale_blocks(mgr, sliding, WINDOWED_END)
        own = kit.pages(kv, sliding)
        assert end > beg and all(own[o] >= 0 for o in range(beg, end)), (
            "the blocks below the window must still hold pages here"
        )
        blocks = range(ceil_blocks(WINDOWED_END))
        lease = lender.lend_write(request, 0, WINDOWED_END)
        in_window = [o for o in blocks if not beg <= o < end]
        check_view(kit, lease.poll(), kv, {1 - sliding: blocks, sliding: in_window})
        lease.release()


def test_an_mla_view_lists_every_block_with_its_tail(kit, real_manager):
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.bindings.internal.batch_manager import CacheType

    mla = dict(kv_cache_type=CacheType.SELFKONLY, num_kv_heads=1, head_dim=128, dtype=DataType.BF16)
    with real_manager(**mla) as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        lender = attach_in_place(mgr)
        lease = lender.lend_read(source, 0, END)
        check_view(kit, lease.poll(), kit.kv(mgr, source), {0: range(ceil_blocks(END))})
        lease.release()


def test_a_write_changes_neither_the_cache_nor_its_pages(kit, real_manager):
    with real_manager() as mgr:
        kit.published(mgr, SOURCE, PROMPT)
        request = kit.make_request(TARGET, PROMPT)
        assert mgr.prepare_context(request)
        kv = kit.kv(mgr, request)
        assert kv.num_committed_tokens >= TPB, "the prefix must be reused"
        assert mgr.resize_context(request, request.context_remaining_length)
        lender = attach_in_place(mgr)
        shape = (kv.capacity, kv.history_length, kv.num_committed_tokens)
        lease = lender.lend_write(request, 0, END)  # starts inside the reused prefix
        view = lease.poll()
        check_view(kit, view, kv, {0: range(ceil_blocks(END))})
        assert (kv.capacity, kv.history_length, kv.num_committed_tokens) == shape, "grown"
        dev = kit.DevicePages(mgr)
        slots = [kit.pages(kv, 0)[o] for o in view.runs[0].ordinals.tolist()]
        content = [dev.read(0, slot) for slot in slots]
        lease.mark_arrived(view.row_masks(True))
        mgr._stream.synchronize()
        now = kit.digest([dev.read(0, slot) for slot in slots])
        assert now == kit.digest(content), "marks copied something"
        assert (kv.capacity, kv.history_length, kv.num_committed_tokens) == shape
        lease.release()


def test_a_view_may_end_past_the_committed_tokens_with_reuse_off(kit, real_manager):
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

    with real_manager(enable_block_reuse=False) as mgr:
        request = kit.make_request(SOURCE, PROMPT)
        assert mgr.prepare_context(request)
        assert mgr.resize_context(request, request.context_remaining_length)
        request.move_to_next_context_chunk()
        batch = ScheduledRequests()
        batch.append_context_request(request)
        mgr.update_context_resources(batch)
        kv = kit.kv(mgr, request)
        assert kv.num_committed_tokens < END
        lender = attach_in_place(mgr)
        lease = lender.lend_read(request, 0, END)
        check_view(kit, lease.poll(), kv, {0: range(ceil_blocks(END))})
        lease.release()


# -- failures ---------------------------------------------------------------------------------


def test_blocks_without_a_page_are_left_out_of_a_read_and_fail_a_write(kit, real_manager):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        kv = kit.kv(mgr, source)
        blocks = ceil_blocks(END)
        assert len(kit.pages(kv, 0)) == blocks
        lender = attach_in_place(mgr)
        beyond = (blocks + 1) * TPB
        read = lender.lend_read(source, 0, beyond)
        check_view(kit, read.poll(), kv, {0: range(blocks)})
        empty = lender.lend_read(source, blocks * TPB, beyond)
        assert empty.poll().num_rows == 0
        write = lender.lend_write(source, 0, beyond)
        assert write.failure is not None and write.poll() is None
        whole = lender.lend_write(source, 0, blocks * TPB)  # every block it touches has a page
        assert whole.poll() is not None
        for lease in (read, empty, whole):
            lease.release()
        mgr.free_resources(source)
        assert kit.closed(kv), "a failed lease holds no loan"
        write.release()


def test_a_suspended_cache_fails_both_lends_at_the_call(kit, real_manager):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        lender = attach_in_place(mgr)
        mgr.suspend_request(source)
        assert not kit.kv(mgr, source).is_active
        for lend in (lender.lend_read, lender.lend_write):
            lease = lend(source, 0, END)
            assert lease.failure is not None and lease.poll() is None
            lease.release()
        assert mgr.resume_request(source)  # active again, both go through
        for lend in (lender.lend_read, lender.lend_write):
            lease = lend(source, 0, END)
            assert lease.failure is None and lease.poll() is not None
            lease.release()


def test_only_a_negative_or_reversed_range_is_an_argument_error(kit, real_manager):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        lender = attach_in_place(mgr)
        for lend in (lender.lend_read, lender.lend_write):
            for start, end in ((-1, TPB), (TPB, 0), (0, -1)):
                with pytest.raises(ValueError):
                    lend(source, start, end)
            nobody = lend(kit.make_request(9, PROMPT), 0, TPB)
            assert nobody.failure is not None and nobody.poll() is None
            nobody.release()
        mgr.shutdown()
        for lend in (lender.lend_read, lender.lend_write):
            lease = lend(source, 0, END)
            assert lease.failure is not None and lease.poll() is None
            lease.release()


def test_mark_arrived_checks_the_shapes_and_nothing_else(kit, real_manager):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        lender = attach_in_place(mgr)
        read = lender.lend_read(source, 0, END)
        read_view = read.poll()
        with pytest.raises(RuntimeError):
            read.mark_arrived(read_view.row_masks())
        write = lender.lend_write(source, 0, END)
        with pytest.raises(RuntimeError):
            write.mark_arrived(())  # before poll() gave the view
        view = write.poll()
        for masks in ((), (np.ones(len(view.runs[0]) + 1, bool),)):
            with pytest.raises(ValueError):
                write.mark_arrived(masks)
        write.release()
        write.mark_arrived(view.row_masks(True))  # after the release too
        with pytest.raises(RuntimeError):
            write.mark_arrived(view.row_masks(True))
        read.release()


# -- the request's free -----------------------------------------------------------------------


def check_a_freed_request_keeps_its_lent_pages_until_the_release(kit, real_manager):
    with real_manager(max_tokens=kit.POOL_TOKENS) as mgr:
        dev = kit.DevicePages(mgr)
        lender = attach_in_place(mgr)
        with lent_and_freed(kit, mgr, lender) as (lease, kv, lent):
            content = {slot: dev.read(0, slot) for slot in lent}
            assert not kit.closed(kv), "a freed cache stays open while lent"
            assert kit.taken_by_others(mgr, lent) == set(), "another request got a lent page"
            now = kit.digest({slot: dev.read(0, slot) for slot in lent})
            assert now == kit.digest(content), "a lent page changed"
            lease.release()
            assert kit.closed(kv), "the last release closes the cache in its call"
            assert kit.whole_pool_goes_to_others(mgr, lent), "the freed pages are reused"


def test_a_freed_request_keeps_its_lent_pages_until_the_release(kit, real_manager):
    check_a_freed_request_keeps_its_lent_pages_until_the_release(kit, real_manager)


def test_the_check_catches_a_lender_letting_the_free_close_a_lent_cache(
    kit, real_manager, monkeypatch
):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    monkeypatch.setattr(_lender.InPlace, "_on_free", lambda self, rid, kv, after: False)
    with pytest.raises(AssertionError, match="a freed cache stays open while lent"):
        check_a_freed_request_keeps_its_lent_pages_until_the_release(kit, real_manager)


def test_the_page_oracle_sees_a_page_the_free_returned(kit, real_manager, monkeypatch):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    monkeypatch.setattr(_lender.InPlace, "_on_free", lambda self, rid, kv, after: False)
    with real_manager(max_tokens=kit.POOL_TOKENS) as mgr:
        lender = attach_in_place(mgr)
        with lent_and_freed(kit, mgr, lender) as (lease, kv, lent):
            assert kit.closed(kv)
            assert kit.taken_by_others(mgr, lent), "the oracle cannot see a returned page"
            lease.release()


@pytest.mark.parametrize("mark_first", [True, False], ids=["mark_first", "release_first"])
def test_marks_and_the_release_may_come_in_either_order(kit, real_manager, mark_first):
    with real_manager(max_tokens=kit.POOL_TOKENS) as mgr:
        lender = attach_in_place(mgr)
        with lent_and_freed(kit, mgr, lender, write=True) as (lease, kv, lent):
            view = lease.poll()
            if mark_first:
                lease.mark_arrived(view.row_masks(True))
                assert not kit.closed(kv), "marks do not end the loan"
                assert kit.taken_by_others(mgr, lent) == set()
                lease.release()
                assert kit.closed(kv)
            else:
                lease.release()
                assert kit.closed(kv), "the release ends the loan"
                lease.mark_arrived(view.row_masks(True))
                assert kit.closed(kv)
            assert kit.whole_pool_goes_to_others(mgr, lent)


def test_the_pages_stay_until_the_last_of_several_leases_ends(kit, real_manager):
    with real_manager(max_tokens=kit.POOL_TOKENS) as mgr:
        request = kit.published(mgr, SOURCE, PROMPT)
        kv = kit.kv(mgr, request)
        lent = set(kit.pages(kv, 0))
        lender = attach_in_place(mgr)
        # Never polled: a loan opens at the call.
        write = lender.lend_write(request, 0, END)
        read = lender.lend_read(request, 0, 2 * TPB)
        mgr.free_resources(request)
        write.release()
        assert not kit.closed(kv) and kit.taken_by_others(mgr, lent) == set()
        read.release()
        assert kit.closed(kv)


def next_owner_s_row_across_the_kept_close(kit, mgr):
    """``SOURCE`` is lent, freed, and its index slot taken by another request; then the lease ends
    and the kept cache closes. The new owner's host page table row before and after."""
    request = kit.published(mgr, SOURCE, PROMPT)
    kv = kit.kv(mgr, request)
    index = mgr.index_mapper.get_index(SOURCE)
    lender = attach_in_place(mgr)
    lease = lender.lend_read(request, 0, END)
    mgr.free_resources(request)
    others = kit.Requests(mgr)
    assert others.allocate(3)
    (other,) = others.held
    assert mgr.index_mapper.get_index(other.py_request_id) == index, "the slot is reused"
    row = mgr.host_kv_cache_block_offsets[0, index * mgr.max_beam_width]
    before = row.clone()
    lease.release()
    assert kit.closed(kv)
    after = row.clone()
    others.free()
    return before, after


def test_a_lent_request_gives_up_its_index_slot_at_its_free(kit, real_manager):
    with real_manager(max_tokens=kit.POOL_TOKENS) as mgr:
        before, after = next_owner_s_row_across_the_kept_close(kit, mgr)
    assert torch.equal(after, before), "closing the kept cache wrote into the next owner's row"


def test_the_row_oracle_sees_a_close_writing_into_the_next_owner(kit, real_manager, monkeypatch):
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

    def free_lent_attached(self, request_id, kv_cache):
        if request_id in self._early_freed_index_requests:
            self._early_freed_index_requests.discard(request_id)
            return
        self.index_mapper.remove_sequence(request_id)

    monkeypatch.setattr(KVCacheManagerV2, "_free_lent", free_lent_attached)
    with real_manager(max_tokens=kit.POOL_TOKENS) as mgr:
        before, after = next_owner_s_row_across_the_kept_close(kit, mgr)
    assert not torch.equal(after, before), "the oracle cannot see a write into the row"


def test_a_kept_cache_closes_before_its_stats_exclusion_is_cleared(kit, real_manager):
    with real_manager(max_tokens=kit.POOL_TOKENS) as mgr:
        request = kit.published(mgr, SOURCE, PROMPT)
        kv = kit.kv(mgr, request)
        mgr.impl.mark_stats_excluded(SOURCE)
        lender = attach_in_place(mgr)
        lease = lender.lend_read(request, 0, END)
        impl = mgr.impl
        spy = mgr.impl = kit.StatsSpy(impl, kv)
        try:
            mgr.free_resources(request)
            assert spy.cleared == [] and impl.is_stats_excluded(SOURCE)
            lease.release()
            assert spy.cleared == [(SOURCE, True)], "cleared before the close"
            assert not impl.is_stats_excluded(SOURCE)
        finally:
            mgr.impl = impl


# -- the manager's shutdown -------------------------------------------------------------------


@pytest.mark.parametrize("freed", [True, False], ids=["request_freed", "request_live"])
def test_shutdown_keeps_what_a_lease_still_lends_until_exit(kit, real_manager, monkeypatch, freed):
    with real_manager(max_tokens=kit.POOL_TOKENS) as mgr:
        request = kit.published(mgr, SOURCE, PROMPT)
        kv = kit.kv(mgr, request)
        dev = kit.DevicePages(mgr)
        lender = attach_in_place(mgr)
        lease = lender.lend_read(request, 0, END)
        slots = [kit.pages(kv, 0)[o] for o in lease.poll().runs[0].ordinals.tolist()]
        content = [dev.read(0, slot) for slot in slots]
        if freed:
            mgr.free_resources(request)
        spy = mgr.impl = kit.ShutdownSpy(mgr.impl)
        warnings = kit.lender_warnings(monkeypatch)
        mgr.shutdown()
        assert not spy.shut_down, "the pools holding a lent page were destroyed"
        assert warnings, "keeping caches until exit is logged"
        kept = kit.retained()
        assert any(o is spy for o in kept) and any(o is kv for o in kept)
        assert not kit.closed(kv)
        now = kit.digest([dev.read(0, slot) for slot in slots])
        assert now == kit.digest(content), "the lent bytes stay readable"
        lease.release()
        assert not kit.closed(kv), "a release after the shutdown closes nothing"
        mgr.shutdown()
        assert not spy.shut_down, "a later shutdown keeps them too"


@pytest.mark.parametrize("leases", [0, 2], ids=["never_lent", "every_lease_ended"])
def test_without_an_open_loan_free_and_shutdown_are_as_without_a_lender(
    kit, real_manager, monkeypatch, leases
):
    with real_manager(max_tokens=kit.POOL_TOKENS) as mgr:
        request = kit.published(mgr, SOURCE, PROMPT)
        kv = kit.kv(mgr, request)
        lent = set(kit.pages(kv, 0))
        lender = attach_in_place(mgr)
        for lend in (lender.lend_read, lender.lend_write)[:leases]:
            lend(request, 0, END).release()
        mgr.free_resources(request)
        assert kit.closed(kv), "freed at once"
        assert kit.taken_by_others(mgr, lent), "the freed pages go to the next requests"
        kept = len(kit.retained())
        spy = mgr.impl = kit.ShutdownSpy(mgr.impl)
        warnings = kit.lender_warnings(monkeypatch)
        mgr.shutdown()
        assert spy.shut_down and warnings == [] and len(kit.retained()) == kept


# -- references and threads -------------------------------------------------------------------


def check_a_dropped_lease_ends_no_loan(kit, real_manager, monkeypatch):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    ended = []
    end_loan = _lender.InPlace._end_loan

    def spy(self, kv_cache):
        ended.append(threading.get_ident())
        return end_loan(self, kv_cache)

    monkeypatch.setattr(_lender.InPlace, "_end_loan", spy)
    with real_manager(max_tokens=kit.POOL_TOKENS) as mgr:
        request = kit.published(mgr, SOURCE, PROMPT)
        kv = kit.kv(mgr, request)
        lender = attach_in_place(mgr)
        cycle = [lender.lend_read(request, 0, END)]
        cycle.append(cycle)  # only a collector frees it
        mgr.free_resources(request)
        del cycle
        in_turn(kit, gc.collect, "collector")
        assert ended == [], "collecting a lease ended its loan"
        assert not kit.closed(kv) and kit.kv(mgr, request) is None
        mgr.shutdown()
        assert any(o is kv for o in kit.retained()), "the shutdown keeps what is still lent"


def test_a_dropped_lease_ends_no_loan_whatever_thread_collects_it(kit, real_manager, monkeypatch):
    check_a_dropped_lease_ends_no_loan(kit, real_manager, monkeypatch)


def test_the_check_catches_a_lease_whose_collection_ends_its_loan(kit, real_manager, monkeypatch):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    init = _lender._InPlaceLease.__init__

    def hold_lender_strongly(self, lender, *args, **kwargs):
        init(self, lender, *args, **kwargs)
        # The collector clears a weak reference inside the garbage before any finalizer runs.
        self._lender = lambda: lender

    def release_when_collected(self):
        self.release()

    monkeypatch.setattr(_lender._InPlaceLease, "__init__", hold_lender_strongly)
    monkeypatch.setattr(_lender._InPlaceLease, "__del__", release_when_collected, raising=False)
    with pytest.raises(AssertionError, match="collecting a lease ended its loan"):
        check_a_dropped_lease_ends_no_loan(kit, real_manager, monkeypatch)


def test_open_leases_keep_neither_the_lender_nor_the_manager_alive(kit):
    torch.cuda.init()
    gc.collect()
    mgr = kit.make_manager()
    request = kit.published(mgr, SOURCE, PROMPT)
    lender = attach_in_place(mgr)
    read = lender.lend_read(request, 0, END)
    write = lender.lend_write(request, 0, END)
    view = write.poll()
    watched = (weakref.ref(mgr), weakref.ref(lender))
    mgr._stream.synchronize()
    del mgr, lender
    gc.collect()
    assert [ref() for ref in watched] == [None, None], "a lease kept its lender or manager alive"
    write.mark_arrived(view.row_masks(True))
    for lease in (write, write, read):
        lease.release()
    gc.collect()
    torch.cuda.empty_cache()


def test_threads_take_turns_and_a_release_closes_a_freed_cache_on_its_own(kit, real_manager):
    with real_manager(max_tokens=kit.POOL_TOKENS) as mgr:
        request = kit.published(mgr, SOURCE, PROMPT)
        kv = kit.kv(mgr, request)
        before = set(threading.enumerate())
        lender = in_turn(kit, lambda: attach_in_place(mgr), "builder")["value"]

        def executor_loop():
            lease = lender.lend_read(request, 0, END)
            rows = lease.poll().num_rows
            mgr.free_resources(request)
            lease.release()  # closes the freed cache inside this call, on this thread
            return rows, kit.closed(kv)

        looped = in_turn(kit, executor_loop, "executor-loop")
        assert looped.get("value") == (ceil_blocks(END), True)
        assert set(threading.enumerate()) <= before, "the lender started a thread"
        assert "error" not in in_turn(kit, mgr.shutdown, "shutdown")


# -- the manager's page-index buffer under a cache that outlives the manager --------------------
# A cache writes -1 for every block into its manager's host page-index buffer as it closes. A new
# tensor reclaims a freed buffer's address with a canary the cache must neither read nor write.

FREED = "touched the page-index buffer freed with its manager"


def fresh_device():
    torch.cuda.init()
    gc.collect()
    torch.cuda.empty_cache()


def check_a_lease_outliving_its_manager(kit, free_first):
    """A lease outlives its manager and lender, collected without a shutdown; ``free_first`` frees
    the request first, which detaches its cache from the buffer."""
    fresh_device()
    mgr = kit.make_manager()
    request = kit.published(mgr, SOURCE, PROMPT)
    kv = kit.kv(mgr, request)
    blocks = int(kv.num_blocks)
    lender = attach_in_place(mgr)
    lease = lender.lend_write(request, 0, END)
    assert lease.poll() is not None
    row = kit.index_row(mgr, request)
    own = list(kv.get_base_page_indices(0)[:blocks])
    assert row.values[:blocks] == own and -1 not in own, "the cache writes elsewhere"
    if free_first:
        mgr.free_resources(request)
    watched = [weakref.ref(o) for o in (mgr.host_kv_cache_block_offsets, mgr, lender)]
    mgr._stream.synchronize()
    del mgr, lender
    gc.collect()
    assert watched[1]() is None and watched[2]() is None, "the manager or lender was not collected"
    canary = None if watched[0]() is not None else kit.reclaim(row)
    if watched[0]() is None:
        assert canary is not None, "inconclusive: no allocation reused the freed buffer's address"
    read = list(kv.get_base_page_indices(0)[:blocks])
    del kv  # the lease now holds the cache's last reference
    lease.release()
    gc.collect()
    written = kit.canary_written(canary, row)
    assert not (canary is not None and read == [kit.CANARY] * blocks) and not written, (
        f"the lent cache {FREED}: it read {read} and its close wrote {written} (cell, value)"
    )


def test_a_lease_outliving_its_manager_touches_no_freed_index_buffer(kit):
    check_a_lease_outliving_its_manager(kit, free_first=False)


def test_a_freed_request_s_cache_is_detached_before_its_manager_goes(kit):
    check_a_lease_outliving_its_manager(kit, free_first=True)


def check_a_cache_kept_at_shutdown(kit):
    """The manager shuts down with a loan open, which keeps the cache until exit; the lease is
    released and the manager collected. Dropping the kept cache, as the process exit does, must
    write nothing into the freed buffer."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    fresh_device()
    mgr = kit.make_manager()
    request = kit.published(mgr, SOURCE, PROMPT)
    kv = kit.kv(mgr, request)
    blocks = int(kv.num_blocks)
    lender = attach_in_place(mgr)
    lease = lender.lend_read(request, 0, END)
    assert lease.poll() is not None
    row = kit.index_row(mgr, request)
    impl = mgr.impl
    mgr._stream.synchronize()
    mgr.shutdown()
    assert any(o is kv for o in kit.retained()), "the shutdown keeps the lent cache"
    lease.release()
    buffer = weakref.ref(mgr.host_kv_cache_block_offsets)
    del mgr, lender
    gc.collect()
    canary = None if buffer() is not None else kit.reclaim(row)
    if buffer() is None:
        assert canary is not None, "inconclusive: no allocation reused the freed buffer's address"
    read = list(kv.get_base_page_indices(0)[:blocks])
    _lender._let_go(kv)  # what the process exit does to the keep list
    _lender._let_go(impl)
    del kv, impl
    gc.collect()
    written = kit.canary_written(canary, row)
    assert not (canary is not None and read == [kit.CANARY] * blocks) and not written, (
        f"the kept cache {FREED}: it read {read} and its close wrote {written} (cell, value)"
    )


def test_a_cache_kept_at_shutdown_touches_no_freed_index_buffer(kit):
    check_a_cache_kept_at_shutdown(kit)


class _Reclaimer:
    """A manager attribute set after the buffer and before the lender, so the manager's collection
    drops it between the two: it reclaims the freed buffer's address before the lender goes."""

    def __init__(self, kit, row, out):
        self._kit, self._row, self._out = kit, row, out

    def __del__(self):
        self._out.append(self._kit.reclaim(self._row))


def check_a_dropped_lease_on_a_collected_manager(kit):
    """A lease dropped unreleased leaves its loan with the lender; the manager is collected without
    a shutdown, its attributes in order. The cache the lender held closes after the buffer went."""
    fresh_device()
    mgr = kit.make_manager()
    request = kit.published(mgr, SOURCE, PROMPT)
    row = kit.index_row(mgr, request)
    out = []
    mgr._reclaimer = _Reclaimer(kit, row, out)
    lender = attach_in_place(mgr)
    lender.lend_read(request, 0, END)  # dropped unreleased
    del lender
    mgr._stream.synchronize()
    watched, buffer = weakref.ref(mgr), weakref.ref(mgr.host_kv_cache_block_offsets)
    del mgr
    gc.collect()
    assert watched() is None
    canary = out[0] if out else None
    if buffer() is None:
        assert canary is not None, "inconclusive: the freed address was not reclaimed"
    written = kit.canary_written(canary, row)
    assert not written, f"the cache the lender held {FREED}: its close wrote {written}"


def test_a_dropped_lease_on_a_collected_manager_touches_no_freed_index_buffer(kit):
    check_a_dropped_lease_on_a_collected_manager(kit)


def check_a_last_release_after_its_manager(kit, cache_held):
    """The manager is collected without a shutdown or a free while a loan is open; the caller still
    holds the lender and releases the last lease. The cache closes as the release drops its last
    reference or, with ``cache_held``, as the test drops its own."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    fresh_device()
    mgr = kit.make_manager()
    request = kit.published(mgr, SOURCE, PROMPT)
    kv = kit.kv(mgr, request)
    blocks = int(kv.num_blocks)
    lender = attach_in_place(mgr)
    lease = lender.lend_write(request, 0, END)
    assert lease.poll() is not None
    row = kit.index_row(mgr, request)
    own = list(kv.get_base_page_indices(0)[:blocks])
    assert row.values[:blocks] == own and -1 not in own, "the cache writes elsewhere"
    buffer, manager = weakref.ref(mgr.host_kv_cache_block_offsets), weakref.ref(mgr)
    mgr._stream.synchronize()
    del mgr
    gc.collect()
    assert manager() is None, "the manager was not collected"
    out = []

    def reclaim_if_freed():
        gc.collect()
        if buffer() is None and not out:
            out.append(kit.reclaim(row))

    reclaim_if_freed()  # a buffer the manager's collection freed
    if cache_held:
        lease.release()
        reclaim_if_freed()
        del kv
    else:
        end_loan = _lender.InPlace._end_loan

        def end_loan_then_reclaim(self, kv_cache):
            # Reclaims a buffer freed here, before the release drops the cache's last reference.
            end_loan(self, kv_cache)
            reclaim_if_freed()

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(_lender.InPlace, "_end_loan", end_loan_then_reclaim)
            del kv  # the lease and the loan now hold the cache's last references
            lease.release()
    gc.collect()
    canary = out[0] if out else None
    if buffer() is None:
        assert canary is not None, "inconclusive: no allocation reused the freed buffer's address"
    written = kit.canary_written(canary, row)
    assert not written, f"the lent cache {FREED}: its close wrote {written} (cell, value)"


@pytest.mark.parametrize("cache_held", [False, True], ids=["release_drops_cache", "cache_held"])
def test_a_last_release_after_its_manager_touches_no_freed_index_buffer(kit, cache_held):
    check_a_last_release_after_its_manager(kit, cache_held)


@pytest.mark.parametrize("cache_held", [False, True], ids=["release_drops_cache", "cache_held"])
def test_the_check_catches_a_lender_letting_the_buffer_go_after_its_manager(
    kit, monkeypatch, cache_held
):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    def let_go_regardless(self):
        _lender._let_go(self._index_buffer)
        self._index_buffer = None

    monkeypatch.setattr(_lender.InPlace, "_let_go_index_buffer", let_go_regardless)
    with pytest.raises(AssertionError, match=FREED):
        check_a_last_release_after_its_manager(kit, cache_held)


@pytest.mark.parametrize(
    "check",
    [
        lambda kit: check_a_lease_outliving_its_manager(kit, free_first=False),
        check_a_cache_kept_at_shutdown,
        check_a_dropped_lease_on_a_collected_manager,
        lambda kit: check_a_last_release_after_its_manager(kit, cache_held=False),
        lambda kit: check_a_last_release_after_its_manager(kit, cache_held=True),
    ],
    ids=[
        "lease_outlives",
        "kept_at_shutdown",
        "dropped_lease",
        "last_release_drops_cache",
        "last_release_cache_held",
    ],
)
def test_the_checks_catch_a_lender_not_keeping_the_index_buffer(kit, monkeypatch, check):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    monkeypatch.setattr(_lender.InPlace, "_keep_index_buffer", lambda self, manager: None)
    with pytest.raises(AssertionError, match=FREED):
        check(kit)


def check_the_index_buffer_is_kept_only_while_lent(kit, real_manager):
    with real_manager(max_tokens=kit.POOL_TOKENS) as mgr:
        request = kit.published(mgr, SOURCE, PROMPT)
        before = len(kit.retained())
        buffer = mgr.host_kv_cache_block_offsets
        lender = attach_in_place(mgr)
        assert not any(o is buffer for o in kit.retained()), "kept with no loan open"
        leases = [lender.lend_read(request, 0, END), lender.lend_write(request, 0, END)]
        assert any(o is buffer for o in kit.retained()), "not kept while a loan is open"
        leases[0].release()
        assert any(o is buffer for o in kit.retained()), "let go with a loan still open"
        leases[1].release()
        assert len(kit.retained()) == before, "kept after the last loan ended"


def test_the_index_buffer_is_kept_only_while_a_loan_is_open(kit, real_manager):
    check_the_index_buffer_is_kept_only_while_lent(kit, real_manager)


def test_the_check_catches_a_lender_keeping_the_index_buffer_for_good(
    kit, real_manager, monkeypatch
):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    monkeypatch.setattr(_lender.InPlace, "_let_go_index_buffer", lambda self: None)
    with pytest.raises(AssertionError, match="kept after the last loan ended"):
        check_the_index_buffer_is_kept_only_while_lent(kit, real_manager)
