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
"""The lenders beside the KV cache manager's automatic host tier: pages that move to host and back,
under their own request's suspension, other requests' pressure and a pool rebalance. Each check
first proves the pages did move, and runs once more against a lender breaking the rule it checks."""

import ctypes
from contextlib import contextmanager

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import (
    StagingOptions,
    attach_in_place,
    attach_staging,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="allocates KV cache pools")

TPB = 32
WINDOW = 64
PROMPT = list(range(1000, 1097))  # three whole blocks and one token
END = 96
BLOCKS = END // TPB
BYSTANDERS = [list(range(3000 + 200 * i, 3097 + 200 * i)) for i in range(2)]
WINDOWED_PROMPT = list(range(2000, 2161))  # five whole blocks and one token
WINDOWED_END = 160
SOURCE, TARGET, SECOND = 1, 2, 3
SCOPE = b"host-tier-suite"
# Only an assertion counts as a catch: a timeout or any other error fails the liar test.
CAUGHT = (AssertionError,)


def attach(mgr, *, fetch_tokens=END, max_fetches=1):
    return attach_staging(mgr, scope=SCOPE, staging=StagingOptions(fetch_tokens, max_fetches))


def attach_breaking(rules):
    """An attach installing a staging lender whose ``rules`` (method name -> function) replace the
    real ones."""

    def attach_rule_breaker(mgr, *, fetch_tokens=END, max_fetches=1):
        from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

        cls = type("RuleBreaker", (_lender.Staging,), dict(rules))
        options = StagingOptions(fetch_tokens, max_fetches)
        return _lender._attach_staging(mgr, scope=SCOPE, staging=options, cls=cls)

    return attach_rule_breaker


def real(name):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    return getattr(_lender.Staging, name)


def ready_view(lease, mgr, tries=3):
    for _ in range(tries):
        mgr._stream.synchronize()
        view = lease.poll()
        if view is not None:
            return view
    raise AssertionError(f"the lease never became ready: {lease.failure}")


def staged(lender, view):
    """Per run, the bytes of every row's staging slot."""
    out = []
    for run in view.runs:
        length = lender.parts[run.part].slot_bytes
        out.append([ctypes.string_at(a, length) for a in run.addresses.tolist()])
    return out


def names(view):
    return [[name.tobytes() for name in run.names] for run in view.runs]


@contextmanager
def to_host_and_back(kit, mgr, request, dev=None):
    """Suspend ``request``; one-block requests take every GPU page, its pages go to host; free the
    others but those on its old pages, resume it. Yields those others and the old pages. With
    ``dev`` the others fill their pages on the stream without waiting for it."""
    kv = kit.kv(mgr, request)
    old = [p for p in kit.pages(kv, 0) if p >= 0]
    host_before = kit.tier_used(mgr, 1)
    others = kit.Requests(mgr, dev)
    try:
        mgr.suspend_request(request)
        assert others.allocate(kit.pool_pages(mgr), fill=dev is None, chunk=1)
        assert set(old) <= others.pages(), "the request's pages stayed on the GPU"
        assert kit.tier_used(mgr, 1) >= host_before + len(old), "its pages did not reach host"
        others.free_all_but(set(old))
        assert mgr.resume_request(request)
        now = [p for p in kit.pages(kv, 0) if p >= 0]
        assert len(now) == len(old) and not set(now) & set(old), "it came back to its old pages"
        yield others, old
    finally:
        others.free()


# -- (a) a fetch and a later publish across the request's own trip to host ---------------------


def check_a_fetch_stays_usable_across_a_trip_to_host(kit, host_tier_manager, attach):
    with host_tier_manager() as mgr_a, host_tier_manager() as mgr_b:
        source = kit.published(mgr_a, SOURCE, PROMPT)
        lender_a = attach(mgr_a)
        publish = lender_a.lend_read(source, 0, END)
        publish_view = ready_view(publish, mgr_a)
        target = kit.admitted(mgr_b, TARGET, PROMPT)
        lender_b = attach(mgr_b)
        lease = lender_b.lend_write(target, 0, END)
        view = lease.poll()
        assert view is not None
        kit.fill_sentinel(mgr_b, target)
        masks = kit.relay(lender_a, publish_view, lender_b, view)
        dev = kit.DevicePages(mgr_b)
        kit.warm_tier_moves(mgr_b)
        with kit.held_stream(mgr_b._stream, strict=False) as gate:
            # The copy into the lent pages waits behind the gate; the suspension and the move to
            # host come after it on the stream, so the move carries the fetched bytes.
            lease.mark_arrived(masks)
            lease.release()
            with to_host_and_back(kit, mgr_b, target, dev) as (others, old):
                assert lender_b.readiness(target) is None, "settled before its copy ran"
                gate.open()
                mgr_b._stream.synchronize()
                assert lender_b.readiness(target) == (END, 0), "not usable once its copy ran"
                for ordinal in range(BLOCKS):
                    got = kit.digest(kit.page(mgr_b, target, 0, ordinal))
                    want = kit.digest(kit.page(mgr_a, source, 0, ordinal))
                    assert got == want, "the fetched bytes are lost"
                sentinel = bytes([kit.SENTINEL]) * dev.page_bytes(0)
                assert all(dev.read(0, slot) == sentinel for slot in old)
        publish.release()


def test_a_fetch_stays_usable_across_its_request_s_trip_to_host(kit, host_tier_manager):
    check_a_fetch_stays_usable_across_a_trip_to_host(kit, host_tier_manager, attach)


def test_the_check_catches_a_lender_tying_a_fetch_to_where_its_pages_were(kit, host_tier_manager):
    def deliver(self, request_id, fetch, rows, copied):  # also remembers the slots the rows had
        real("_deliver")(self, request_id, fetch, rows, copied)
        self.lent_rows = rows

    def where_they_were(self, manager, delivered, committed):
        from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _manager

        rows = self.lent_rows
        for lg, ordinals, slots in zip(rows.layer_groups, rows.ordinals, rows.device_slots):
            moved = ordinals[_manager.pages(delivered.kv, lg)[ordinals] != slots]
            delivered.blocks[lg][moved[moved < len(delivered.blocks[lg])]] = False
        return real("_usable_until")(self, manager, delivered, committed)

    liar = attach_breaking({"_deliver": deliver, "_usable_until": where_they_were})
    with pytest.raises(CAUGHT, match="not usable once its copy ran"):
        check_a_fetch_stays_usable_across_a_trip_to_host(kit, host_tier_manager, liar)


def check_a_publish_after_a_trip_to_host_reads_the_pages_where_they_are(
    kit, host_tier_manager, attach
):
    with host_tier_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        original = [kit.page(mgr, source, 0, o) for o in range(BLOCKS)]
        lender = attach(mgr)
        first = lender.lend_read(source, 0, END)
        assert kit.digest(staged(lender, ready_view(first, mgr))) == kit.digest([original])
        first.release()
        with to_host_and_back(kit, mgr, source):
            again = lender.lend_read(source, 0, END)
            got = kit.digest(staged(lender, ready_view(again, mgr)))
            assert got == kit.digest([original]), "read a stale page"
            again.release()


def test_a_publish_after_a_trip_to_host_reads_the_pages_where_they_are(kit, host_tier_manager):
    check_a_publish_after_a_trip_to_host_reads_the_pages_where_they_are(
        kit, host_tier_manager, attach
    )


def test_the_check_catches_a_lender_keeping_page_indices_across_lends(kit, host_tier_manager):
    seen = {}

    def remembered(self, manager, kv, start, end):
        key = (id(kv), start, end)
        if key not in seen:
            seen[key] = real("_rows")(self, manager, kv, start, end)
        return seen[key]

    with pytest.raises(CAUGHT, match="read a stale page"):
        check_a_publish_after_a_trip_to_host_reads_the_pages_where_they_are(
            kit, host_tier_manager, attach_breaking({"_rows": remembered})
        )


# -- (b) other requests' pressure while a fetch is lent -----------------------------------------


def check_pressure_while_a_fetch_is_lent(kit, host_tier_manager, attach, moment):
    """``moment``: ``copy_queued``, the target stays active and its marks' copy waits on the stream
    while other requests push committed blocks to host; ``before_marks``, the target itself goes to
    host and back between the grant and the marks."""
    with host_tier_manager() as mgr_a, host_tier_manager() as mgr_b:
        source = kit.published(mgr_a, SOURCE, PROMPT)
        lender_a = attach(mgr_a)
        publish = lender_a.lend_read(source, 0, END)
        publish_view = ready_view(publish, mgr_a)
        kit.warm_tier_moves(mgr_b)
        for i, tokens in enumerate(BYSTANDERS):  # committed blocks, then no request holds them
            mgr_b.free_resources(kit.published(mgr_b, 10 + i, tokens))
        target = kit.admitted(mgr_b, TARGET, PROMPT)
        lender_b = attach(mgr_b)
        lease = lender_b.lend_write(target, 0, END)
        view = lease.poll()
        assert view is not None
        kit.fill_sentinel(mgr_b, target)
        masks = kit.relay(lender_a, publish_view, lender_b, view)
        kv = kit.kv(mgr_b, target)
        lent = kit.pages(kv, 0)[:BLOCKS]
        dev = kit.DevicePages(mgr_b)
        sentinel = bytes([kit.SENTINEL]) * dev.page_bytes(0)
        if moment == "copy_queued":
            others = kit.Requests(mgr_b, dev)
            host_before = kit.tier_used(mgr_b, 1)
            try:
                with kit.held_stream(mgr_b._stream, strict=False) as gate:
                    lease.mark_arrived(masks)
                    lease.release()
                    assert others.allocate(kit.pool_pages(mgr_b) - BLOCKS, fill=False, chunk=1)
                    moved = kit.tier_used(mgr_b, 1) - host_before
                    assert moved >= len(BYSTANDERS) * BLOCKS, "no bystander block reached host"

                    assert kit.pages(kv, 0)[:BLOCKS] == lent, "the lent pages moved"
                    assert lender_b.readiness(target) is None, "settled before its copy ran"
                    gate.open()
                assert lender_b.readiness(target) == (END, 0)
                assert all(dev.read(0, slot) == sentinel for slot in others.pages())
            finally:
                others.free()
        else:
            with to_host_and_back(kit, mgr_b, target) as (others, old):
                assert old == lent
                lease.mark_arrived(masks)
                lease.release()
                mgr_b._stream.synchronize()
                assert all(dev.read(0, slot) == sentinel for slot in old), (
                    "the marks' copy reached another request's page"
                )
                assert lender_b.readiness(target) == (0, 0), "counts rows that never landed"
        for ordinal in range(BLOCKS):
            got = kit.digest(kit.page(mgr_b, target, 0, ordinal))
            expected = kit.page(mgr_a, source, 0, ordinal) if moment == "copy_queued" else sentinel
            assert got == kit.digest(expected), "a page does not hold what its fetch says"
        publish.release()


@pytest.mark.parametrize("moment", ["copy_queued", "before_marks"])
def test_other_requests_pushing_blocks_to_host_leave_a_fetch_where_its_readiness_says(
    kit, host_tier_manager, moment
):
    check_pressure_while_a_fetch_is_lent(kit, host_tier_manager, attach, moment)


def test_the_check_catches_a_lender_settling_a_fetch_before_its_copy_ran(kit, host_tier_manager):
    def at_the_marks(self, fetch):  # settled once marked, whether or not the copy has run
        return fetch.delivered

    with pytest.raises(CAUGHT, match="settled before its copy ran"):
        check_pressure_while_a_fetch_is_lent(
            kit,
            host_tier_manager,
            attach_breaking({"_report_settled": at_the_marks}),
            "copy_queued",
        )


def test_the_check_catches_a_lender_trusting_the_cache_but_not_its_pages(kit, host_tier_manager):
    def same_cache(self, kv, lease):
        from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _manager

        alive = kv is not None and kv is lease._kv and _manager.cache_state(kv).active
        return [np.full(len(o), alive, dtype=bool) for o in lease._rows.ordinals]

    with pytest.raises(CAUGHT, match="the marks' copy reached another request's page"):
        check_pressure_while_a_fetch_is_lent(
            kit, host_tier_manager, attach_breaking({"_still_lent": same_cache}), "before_marks"
        )


# -- (c) a cache whose pages are on host fails every lend at the call ---------------------------

SERVED = "served bytes that are not its blocks'"
REACHED = "reached another request's page"


def use_as_a_backend(kit, mgr, lender, lease, reading, kind, kv, original):
    """What a backend does with a lease that became ready: a read's rows are compared with the
    blocks' bytes, a write's rows are filled. In place, rows are found in the manager's page table."""
    mgr._stream.synchronize()
    view = lease.poll()
    if view is None:
        return
    if kind == "staging":
        if reading:
            (run,) = view.runs
            want = [[original[o] for o in run.ordinals.tolist()]]
            assert kit.digest(staged(lender, view)) == kit.digest(want), SERVED
        else:
            kit.stage(lender, view, 0x5A)
            lease.mark_arrived(view.row_masks(True))
        return
    dev = kit.DevicePages(mgr)
    for run in view.runs:
        lg, slots = run.layer_group, kit.pages(kv, run.layer_group)
        for ordinal in run.ordinals.tolist():
            if reading:
                got = dev.read(lg, slots[ordinal])
                assert kit.digest(got) == kit.digest(original[ordinal]), SERVED
            else:
                dev.write(lg, slots[ordinal], bytes([0x5A]) * dev.page_bytes(lg))


def check_a_cache_on_host_fails_lends_at_the_call(kit, host_tier_manager, kind, to_host=True):
    """Both caches go to host before the lends. ``to_host=False`` leaves their pages on the GPU,
    for a lender's rule breach to show without the host tier."""
    with host_tier_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        original = [kit.page(mgr, source, 0, o) for o in range(BLOCKS)]
        target = kit.admitted(mgr, TARGET, BYSTANDERS[0])
        assert mgr.resize_context(target, target.context_remaining_length)
        lender = attach(mgr, max_fetches=2) if kind == "staging" else attach_in_place(mgr)
        writer = target if kind == "staging" else source
        kv, kv_writer = kit.kv(mgr, source), kit.kv(mgr, writer)
        old = [p for p in kit.pages(kv, 0) if p >= 0]
        old_writer = [p for p in kit.pages(kv_writer, 0) if p >= 0]
        host_before = kit.tier_used(mgr, 1)
        others = kit.Requests(mgr)
        try:
            mgr.suspend_request(target)
            mgr.suspend_request(source)
            if to_host:
                assert others.allocate(kit.pool_pages(mgr), chunk=1)
                assert set(old) | set(old_writer) <= others.pages()
                moved = len(set(old) | set(old_writer))
                assert kit.tier_used(mgr, 1) >= host_before + moved, "its pages did not reach host"
            dev = kit.DevicePages(mgr)
            theirs = {slot: dev.read(0, slot) for slot in others.pages()}
            failed = [lender.lend_read(source, 0, END), lender.lend_write(writer, 0, END)]
            for lease, reading, lent in zip(failed, (True, False), (kv, kv_writer)):
                use_as_a_backend(kit, mgr, lender, lease, reading, kind, lent, original)
            mgr._stream.synchronize()
            now = {slot: dev.read(0, slot) for slot in theirs}
            assert kit.digest(now) == kit.digest(theirs), REACHED
            for lease in failed:
                assert lease.failure is not None, "not failed at the call"
                assert lease.poll() is None
            if kind == "staging":
                groups = kit.pool_group_ids(mgr)
                assert [lender._free_slots(g) for g in groups] == [p.slots for p in lender.parts]
            for lease in failed:
                lease.release()
            others.free_all_but(set(old))
            assert mgr.resume_request(source) and mgr.resume_request(target)
            now = [p for p in kit.pages(kv, 0) if p >= 0]
            assert not set(now) & set(old), "it came back to its old pages"
            read = lender.lend_read(source, 0, END)
            view = ready_view(read, mgr)
            if kind == "staging":
                assert kit.digest(staged(lender, view)) == kit.digest([original])
            else:
                (run,) = view.runs
                assert run.ordinals.tolist() == list(range(BLOCKS))
            write = lender.lend_write(writer, 0, END)
            write_view = ready_view(write, mgr)
            write.mark_arrived(write_view.row_masks())
            write.release()
            read.release()
        finally:
            others.free()


@pytest.mark.parametrize("kind", ["staging", "in_place"])
def test_a_cache_on_host_fails_every_lend_at_the_call_and_lends_again_once_resumed(
    kit, host_tier_manager, kind
):
    check_a_cache_on_host_fails_lends_at_the_call(kit, host_tier_manager, kind)


@pytest.mark.parametrize("pages", ["on_host", "on_gpu"])
@pytest.mark.parametrize("kind", ["staging", "in_place"])
def test_the_check_catches_a_lender_lending_a_suspended_cache_by_the_pages_it_lists(
    kit, host_tier_manager, monkeypatch, kind, pages
):
    """The lender reads a suspended cache as active and, in place, lends every page it lists. On
    host those pages are others' and the contents catch it; on the GPU only the call's rule does."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _manager

    state = _manager.cache_state
    monkeypatch.setattr(_manager, "cache_state", lambda kv: state(kv)._replace(active=True))
    if kind == "in_place":
        monkeypatch.setattr(_manager, "locked_pages", _manager.pages)
    caught = f"{SERVED}|{REACHED}" if pages == "on_host" else "not failed at the call"
    with pytest.raises(CAUGHT, match=caught):
        check_a_cache_on_host_fails_lends_at_the_call(
            kit, host_tier_manager, kind, to_host=pages == "on_host"
        )


# -- (d) a committed block's name does not depend on its tier -----------------------------------


def check_a_block_keeps_its_name_on_either_tier(kit, host_tier_manager, attach):
    with host_tier_manager() as mgr:
        first = kit.published(mgr, SOURCE, PROMPT)
        lender = attach(mgr)
        lease = lender.lend_read(first, 0, END)
        view = ready_view(lease, mgr)
        first_names, first_bytes = names(view), staged(lender, view)
        lease.release()
        old = kit.pages(kit.kv(mgr, first), 0)[:BLOCKS]
        mgr.free_resources(first)  # its whole blocks stay committed in the reuse tree
        host_before = kit.tier_used(mgr, 1)
        others = kit.Requests(mgr)
        try:
            assert others.allocate(kit.pool_pages(mgr), chunk=1)
            assert set(old) <= others.pages()
            assert kit.tier_used(mgr, 1) >= host_before + BLOCKS, "the blocks did not reach host"
            others.free_all_but(set(old))
            second = kit.make_request(SECOND, PROMPT)
            assert mgr.prepare_context(second)
            kv = kit.kv(mgr, second)
            assert kv.num_committed_tokens == END, "the blocks on host were not reused"
            now = kit.pages(kv, 0)[:BLOCKS]
            assert not set(now) & set(old), "the blocks came back to their old pages"
            again = lender.lend_read(second, 0, END)
            view = ready_view(again, mgr)
            assert names(view) == first_names, "a block's name changed with its tier"
            assert kit.digest(staged(lender, view)) == kit.digest(first_bytes)
            again.release()
        finally:
            others.free()


def test_a_committed_block_has_one_name_on_the_gpu_and_after_a_trip_to_host(kit, host_tier_manager):
    check_a_block_keeps_its_name_on_either_tier(kit, host_tier_manager, attach)


def test_the_check_catches_a_lender_naming_a_block_by_its_slot(kit, host_tier_manager):
    def by_slot(self, rows, keys):
        slot_keys = [
            np.repeat(slots.astype("<i8").view(np.uint8).reshape(-1, 8), 4, axis=1)
            for slots in rows.device_slots
        ]
        return real("_view")(self, rows, slot_keys)

    with pytest.raises(CAUGHT, match="a block's name changed with its tier"):
        check_a_block_keeps_its_name_on_either_tier(
            kit, host_tier_manager, attach_breaking({"_view": by_slot})
        )


# -- (e) window blocks behind the history only hold their pages ---------------------------------


def check_in_place_below_the_history_lends_only_locked_pages(kit, host_tier_manager):
    with host_tier_manager(windows=[WINDOW, 256]) as mgr:
        request = kit.make_request(TARGET, WINDOWED_PROMPT)
        assert mgr.prepare_context(request)
        assert mgr.resize_context(request, request.context_remaining_length)
        kv = kit.kv(mgr, request)
        sliding = kit.windows(mgr).index(WINDOW)
        # The history moves past the window: the blocks behind it keep their pages only held.
        assert mgr._resize_for_connector_prefix(request, kv, WINDOWED_END, WINDOWED_END + 1)
        beg, end = kit.stale_blocks(mgr, sliding, WINDOWED_END)
        assert (beg, end) == (0, 3)
        held = kit.pages(kv, sliding)[:end]
        assert all(p >= 0 for p in held), "the blocks behind the window hold no pages here"
        lender = attach_in_place(mgr)
        host_before = kit.tier_used(mgr, 1)
        # Still on the GPU, before any pressure, a held page is left out all the same.
        early = lender.lend_read(request, 0, 2 * TPB)
        assert kit.pages(kv, sliding)[:end] == held and kit.tier_used(mgr, 1) == host_before
        runs = {run.layer_group: run.ordinals.tolist() for run in early.poll().runs}
        assert runs == {1 - sliding: [0, 1], sliding: []}, "lent a held page still on the GPU"
        early.release()
        others = kit.Requests(mgr)
        try:
            while others.allocate(1, chunk=1):
                pass
            gone = [o for o in range(end) if kit.pages(kv, sliding)[o] != held[o]]
            assert set(gone) >= {0, 1}, "the held pages stayed on the GPU"
            assert kit.tier_used(mgr, 1) >= host_before + len(gone)
            read = lender.lend_read(request, 0, 2 * TPB)  # a history of 64 reads blocks 0 and 1
            runs = {run.layer_group: run.ordinals.tolist() for run in read.poll().runs}
            assert runs == {1 - sliding: [0, 1], sliding: []}, "lent a page it only holds"
            read.release()
            write = lender.lend_write(request, 0, 2 * TPB)
            assert write.failure is not None, "a write into pages only held went through"
            write.release()
        finally:
            others.free()


def test_in_place_below_the_history_lends_only_pages_its_cache_locks(kit, host_tier_manager):
    check_in_place_below_the_history_lends_only_locked_pages(kit, host_tier_manager)


def test_the_check_catches_a_lender_lending_held_pages_below_the_history(
    kit, host_tier_manager, monkeypatch
):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender, _manager

    locked = _manager.locked_pages

    def every_page(kv, layer_group):  # every block's page, held ones included
        if not kv.is_active:
            return locked(kv, layer_group)
        return _manager.pages(kv, layer_group)

    monkeypatch.setattr(_lender._manager, "locked_pages", every_page)
    with pytest.raises(CAUGHT, match="lent a held page still on the GPU"):
        check_in_place_below_the_history_lends_only_locked_pages(kit, host_tier_manager)


def test_the_check_catches_a_lender_lending_held_pages_while_they_stay_on_the_gpu(
    kit, host_tier_manager, monkeypatch
):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender, _manager

    locked = _manager.locked_pages

    def while_nothing_is_on_host(kv, layer_group):  # a held page taken as safe on the GPU
        stats = kv.manager.get_storage_statistics(1)
        if any(int(s.total) != int(s.free) for s in stats):
            return locked(kv, layer_group)
        return _manager.pages(kv, layer_group)

    monkeypatch.setattr(_lender._manager, "locked_pages", while_nothing_is_on_host)
    with pytest.raises(CAUGHT, match="lent a held page still on the GPU"):
        check_in_place_below_the_history_lends_only_locked_pages(kit, host_tier_manager)


def test_the_check_catches_a_lender_lending_held_pages_once_some_are_on_host(
    kit, host_tier_manager, monkeypatch
):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender, _manager

    locked = _manager.locked_pages

    def once_on_host(kv, layer_group):  # right before any pressure, wrong after it
        stats = kv.manager.get_storage_statistics(1)
        if any(int(s.total) != int(s.free) for s in stats):
            return _manager.pages(kv, layer_group)
        return locked(kv, layer_group)

    monkeypatch.setattr(_lender._manager, "locked_pages", once_on_host)
    with pytest.raises(CAUGHT, match="lent a page it only holds"):
        check_in_place_below_the_history_lends_only_locked_pages(kit, host_tier_manager)


# -- (e) a pool rebalance while a fetch is lent -------------------------------------------------


class DeviceCopyGuard:
    """The CUDA driver, except that a memcpy into device memory outside ``allowed()`` is recorded and
    not run: after a pool shrinks, the slot it names may be unmapped."""

    def __init__(self, real_driver, device_ranges, allowed):
        self._real = real_driver
        self._device = device_ranges  # every byte a device pool ever held
        self._allowed = allowed  # () -> [(begin, end)] the copies may write
        self.into_device = 0
        self.stray = []

    @staticmethod
    def _covered(begin, end, ranges):
        for lo, hi in sorted(ranges):
            if lo <= begin < hi:
                begin = hi
            if begin >= end:
                return True
        return False

    def __getattr__(self, name):
        found = getattr(self._real, name)
        if name != "cuMemcpyAsync":
            return found

        def call(dst, src, nbytes, stream):
            begin = int(dst)
            if any(lo <= begin < hi for lo, hi in self._device):
                self.into_device += 1
                if not self._covered(begin, begin + int(nbytes), self._allowed()):
                    self.stray.append((begin, int(nbytes)))
                    return (self._real.CUresult.CUDA_SUCCESS,)
            return found(dst, src, nbytes, stream)

        return call


def page_ranges(kit, mgr, kv, layer_groups):
    """Byte ranges of the pages ``kv`` locks in ``layer_groups``, in every pool of their groups."""
    group_of = {int(pg.pool_group_index): pg for pg in mgr.impl.pool_group_descs}
    pool_group = kit.pool_group_of(mgr)
    out = []
    for lg in layer_groups:
        pg = group_of[pool_group[lg]]
        slots = [int(s) for s in kv.get_base_page_indices(lg)[: kv.num_blocks] if int(s) >= 0]
        for pool in pg.pools:
            base, width = int(pool.base_address), int(pool.slot_bytes)
            out.extend((base + s * width, base + (s + 1) * width) for s in slots)
    return out


def check_a_rebalance_under_a_fetch_moves_no_copy_off_the_pages_still_lent(
    kit, host_tier_manager, monkeypatch, attach
):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender
    from tensorrt_llm.runtime.kv_cache_manager_v2 import _introspection

    shape = dict(windows=[WINDOW, 256], num_layers=3, max_tokens=4 * kit.POOL_TOKENS)
    with host_tier_manager(**shape) as mgr_a, host_tier_manager(**shape) as mgr_b:
        source = kit.published(mgr_a, SOURCE, WINDOWED_PROMPT)
        lender_a = attach(mgr_a, fetch_tokens=WINDOWED_END)
        publish = lender_a.lend_read(source, 0, WINDOWED_END)
        publish_view = ready_view(publish, mgr_a)
        full = kit.windows(mgr_b).index(None)
        pg_full = kit.pool_group_of(mgr_b)[full]
        device = []
        for pg in mgr_b.impl.pool_group_descs:
            for pool in pg.pools:
                base = int(pool.base_address)
                device.append((base, base + int(pg.num_slots) * int(pool.slot_bytes)))
        # Fillers take the low slots, so the target's full-attention pages sit high in their pool.
        fillers = kit.Requests(mgr_b)
        stats = mgr_b.impl.get_storage_statistics
        while stats(0)[pg_full].total - stats(0)[pg_full].free < 72 and fillers.allocate(7):
            pass
        target = kit.admitted(mgr_b, TARGET, WINDOWED_PROMPT)
        lender_b = attach(mgr_b, fetch_tokens=WINDOWED_END)
        lease = lender_b.lend_write(target, 0, WINDOWED_END)
        view = lease.poll()
        assert view is not None
        kit.fill_sentinel(mgr_b, target)
        masks = kit.relay(lender_a, publish_view, lender_b, view)
        kv = kit.kv(mgr_b, target)
        lent = [kit.pages(kv, lg) for lg in range(2)]
        fillers.free()
        # The executor's rebalance: suspend every active request, adjust the pools, resume them.
        mgr_b.suspend_request(target)
        _introspection.force_rebalance_precondition(mgr_b.impl, skew=0.05 if pg_full == 0 else 20)
        assert mgr_b.impl.need_adjustment
        mgr_b.impl.adjust()
        assert mgr_b.resume_request(target)
        now = [kit.pages(kv, lg) for lg in range(2)]
        slots_now = {
            int(pg.pool_group_index): int(pg.num_slots) for pg in mgr_b.impl.pool_group_descs
        }
        gone = [p for p in lent[full] if p >= slots_now[pg_full]]
        assert now[full] != lent[full] and gone, "the rebalance left the lent pages where they were"
        guard = DeviceCopyGuard(_lender.drv, device, lambda: page_ranges(kit, mgr_b, kv, range(2)))
        monkeypatch.setattr(_lender, "drv", guard)
        lease.mark_arrived(masks)
        lease.release()
        mgr_b._stream.synchronize()
        assert guard.stray == [], "the marks' copy wrote outside the pages the target locks"
        assert guard.into_device, "nothing was copied: the check saw no copy"
        assert lender_b.readiness(target) == (0, WINDOWED_END), "counts rows that never landed"
        publish.release()


def test_a_rebalance_under_a_fetch_moves_no_copy_off_the_pages_still_lent(
    kit, host_tier_manager, monkeypatch
):
    check_a_rebalance_under_a_fetch_moves_no_copy_off_the_pages_still_lent(
        kit, host_tier_manager, monkeypatch, attach
    )


def test_the_check_catches_a_lender_copying_into_pages_a_rebalance_moved(
    kit, host_tier_manager, monkeypatch
):
    def same_cache(self, kv, lease):
        from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _manager

        alive = kv is not None and kv is lease._kv and _manager.cache_state(kv).active
        return [np.full(len(o), alive, dtype=bool) for o in lease._rows.ordinals]

    with pytest.raises(CAUGHT, match="the marks' copy wrote outside the pages the target locks"):
        check_a_rebalance_under_a_fetch_moves_no_copy_off_the_pages_still_lent(
            kit, host_tier_manager, monkeypatch, attach_breaking({"_still_lent": same_cache})
        )


# -- (f) a mark arriving after the window passed the lent blocks --------------------------------

LATE_PROMPT = list(range(2000, 2161))  # five whole blocks and one token
LATE_END = 2 * TPB  # the lease covers blocks 0 and 1
LATE_HISTORY = 5 * TPB  # a served prefix moves the history here: blocks 0 to 2 leave the window
FETCHED = 0xAB


class CopyLog:
    """The CUDA driver, recording every ``cuMemcpyAsync`` destination; every copy still runs."""

    def __init__(self, real_driver):
        self._real = real_driver
        self.copies = []

    def __getattr__(self, name):
        found = getattr(self._real, name)
        if name != "cuMemcpyAsync":
            return found

        def call(dst, src, nbytes, stream):
            self.copies.append(int(dst))
            return found(dst, src, nbytes, stream)

        return call


def device_slot(mgr, address):
    """``(pool group, slot)`` of a device address, or ``None`` outside every device pool."""
    for pg in mgr.impl.pool_group_descs:
        for pool in pg.pools:
            base, width = int(pool.base_address), int(pool.slot_bytes)
            if base <= address < base + int(pg.num_slots) * width:
                return int(pg.pool_group_index), (address - base) // width
    return None


def locked_by(kit, mgr, requests):
    """``{(pool group, slot): request id}`` of every page ``requests`` lock, every layer group."""
    group_of = kit.pool_group_of(mgr)
    out = {}
    for request in requests:
        kv = kit.kv(mgr, request)
        for lg in range(kit.num_layer_groups(mgr)):
            for slot in kv.get_base_page_indices(lg)[: kv.num_blocks]:
                if int(slot) >= 0:
                    out[(group_of[lg], int(slot))] = request.py_request_id
    return out


def late_mark(kit, host_tier_manager, monkeypatch, attach, windows, fillers, keep):
    """One run, ``fillers`` one-block requests allocated first and kept or freed by ``keep``, which
    moves the slots the target gets. What the marks' copy wrote, or ``None`` when no lent block went
    to host under its lent GPU slot's number."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    with host_tier_manager(windows=windows) as mgr:
        sliding = kit.windows(mgr).index(WINDOW)
        group = kit.pool_group_of(mgr)[sliding]
        first, others = kit.Requests(mgr), kit.Requests(mgr)
        others._next = 500
        try:
            assert first.allocate(fillers, chunk=1) if fillers else True
            if not keep:
                first.free()
            target = kit.admitted(mgr, TARGET, LATE_PROMPT)
            lender = attach(mgr, fetch_tokens=LATE_END)
            lease = lender.lend_write(target, 0, LATE_END)
            view = lease.poll()
            assert view is not None, lease.failure
            kv = kit.kv(mgr, target)
            lent = kit.pages(kv, sliding)[:2]
            assert [int(s) for s in kv.get_base_page_indices(sliding)[:2]] == lent
            kit.stage(lender, view, FETCHED)
            # Before the marks, a served prefix moves the history past the lent blocks of the
            # sliding group, as a KV connector's reservation does: they keep their pages only held.
            assert mgr._resize_for_connector_prefix(target, kv, LATE_HISTORY, LATE_HISTORY + 1)
            stale = kit.stale_blocks(mgr, sliding, kv.history_length)
            assert stale[0] <= 0 and stale[1] >= 2, "the window did not pass the lent blocks"
            assert all(int(s) < 0 for s in kv.get_base_page_indices(sliding)[:2]), "still locked"
            assert kit.pages(kv, sliding)[:2] == lent, "the lent blocks do not hold their pages"
            # Other requests take every GPU page left: the held pages go to host.
            host_before = kit.tier_used(mgr, 1)
            while others.allocate(4, chunk=4):
                pass
            while others.allocate(1, chunk=1):
                pass
            now = kit.pages(kv, sliding)[:2]
            owners = locked_by(kit, mgr, others.held + first.held)
            moved = [o for o in range(2) if (group, lent[o]) in owners and now[o] >= 0]
            assert kit.tier_used(mgr, 1) >= host_before + len(moved), "nothing reached host"
            aliased = [o for o in moved if now[o] == lent[o]]
            if not aliased:
                lease.mark_arrived(view.row_masks(False))
                lease.release()
                return None
            dev = kit.DevicePages(mgr)
            watched = {lent[o]: owners[(group, lent[o])] for o in aliased}
            before = {s: dev.read(sliding, s) for s in watched}
            log = CopyLog(_lender.drv)
            monkeypatch.setattr(_lender, "drv", log)
            lease.mark_arrived(view.row_masks(True))
            lease.release()
            mgr._stream.synchronize()
            monkeypatch.undo()
            written = [device_slot(mgr, dst) for dst in log.copies]
            hit = sorted({w for w in written if w is not None and w in owners})
            changed = sorted(watched[s] for s in watched if dev.read(sliding, s) != before[s])
            return dict(hit=hit, changed=changed)
        finally:
            others.free()
            first.free()


def check_a_late_mark_writes_no_slot_its_window_left(kit, host_tier_manager, monkeypatch, attach):
    # Slots go out lowest first, so a short search finds a lent block on host under its lent GPU
    # slot's number while another request locks that GPU slot.
    runs = [(w, n, k) for w in ([WINDOW], [WINDOW, 256]) for k in (True, False) for n in range(4)]
    for windows, fillers, keep in runs:
        result = late_mark(kit, host_tier_manager, monkeypatch, attach, windows, fillers, keep)
        if result is not None:
            break
    else:
        raise RuntimeError("no run put a lent block on host under its lent GPU slot's number")
    assert not result["hit"], f"the marks' copy wrote slots other requests lock: {result['hit']}"
    assert not result["changed"], f"requests {result['changed']} now hold the fetch"


def test_a_late_mark_writes_no_slot_its_window_left(kit, host_tier_manager, monkeypatch):
    check_a_late_mark_writes_no_slot_its_window_left(kit, host_tier_manager, monkeypatch, attach)


def test_the_check_catches_a_lender_matching_pages_by_slot_number_alone(
    kit, host_tier_manager, monkeypatch
):
    def same_number(self, kv, lease):  # any page of the block, on any tier, under the lent number
        from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _manager

        rows = lease._rows
        alive = kv is not None and kv is lease._kv and _manager.cache_state(kv).active
        masks = []
        for lg, ordinals, slots in zip(rows.layer_groups, rows.ordinals, rows.device_slots):
            same = np.zeros(len(ordinals), dtype=bool)
            if alive:
                pages = _manager.pages(kv, lg)
                inside = ordinals < len(pages)
                same[inside] = pages[ordinals[inside]] == slots[inside]
            masks.append(same)
        return masks

    with pytest.raises(CAUGHT, match="the marks' copy wrote slots other requests lock"):
        check_a_late_mark_writes_no_slot_its_window_left(
            kit, host_tier_manager, monkeypatch, attach_breaking({"_still_lent": same_number})
        )
