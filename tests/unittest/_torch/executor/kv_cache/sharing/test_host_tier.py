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
"""The staging lender beside the KV cache manager's host tier. Each check first proves the move it
covers: pages to host and back under their own request's suspension or other requests' pressure,
bystander blocks to host while lent pages stay, lent pages to other slots under a pool rebalance,
and held pages and a late mark's pages to host."""

import ctypes
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import StagingOptions, attach_staging

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


def attach(mgr, *, fetch_tokens=END, max_fetches=1):
    return attach_staging(mgr, scope=SCOPE, staging=StagingOptions(fetch_tokens, max_fetches))


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


def test_a_fetch_stays_usable_across_its_request_s_trip_to_host(
    kit, host_tier_manager, attach=attach
):
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


def test_a_publish_after_a_trip_to_host_reads_the_pages_where_they_are(
    kit, host_tier_manager, attach=attach
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


# -- (b) other requests' pressure while a fetch is lent -----------------------------------------


@pytest.mark.parametrize("moment", ["copy_queued", "before_marks"])
def test_other_requests_pushing_blocks_to_host_leave_a_fetch_where_its_readiness_says(
    kit, host_tier_manager, moment, attach=attach
):
    """``moment``: ``copy_queued``, the target stays active and its marks' copy waits on the stream
    while other requests push committed blocks to host; ``before_marks``, the target itself goes to
    host and back between the grant and the marks."""
    with host_tier_manager() as mgr_a, host_tier_manager() as mgr_b:
        source = kit.published(mgr_a, SOURCE, PROMPT)
        lender_a = attach(mgr_a)
        publish = lender_a.lend_read(source, 0, END)
        publish_view = ready_view(publish, mgr_a)
        original = [kit.page(mgr_a, source, 0, o) for o in range(BLOCKS)]
        got = kit.digest(staged(lender_a, publish_view))
        assert got == kit.digest([original]), "the relay carries rows off their blocks"
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


# -- (c) a cache whose pages are on host fails every lend at the call ---------------------------

SERVED = "served bytes that are not its blocks'"
REACHED = "reached another request's page"


def use_as_a_backend(kit, mgr, lender, lease, reading, original):
    """What a backend does with a lease that became ready: a read's rows are compared with the
    blocks' bytes, a write's rows are filled."""
    mgr._stream.synchronize()
    view = lease.poll()
    if view is None:
        return
    if reading:
        (run,) = view.runs
        want = [[original[o] for o in run.ordinals.tolist()]]
        assert kit.digest(staged(lender, view)) == kit.digest(want), SERVED
    else:
        kit.stage(lender, view, 0x5A)
        lease.mark_arrived(view.row_masks(True))


def test_a_cache_on_host_fails_every_lend_at_the_call_and_lends_again_once_resumed(
    kit, host_tier_manager, to_host=True
):
    """Both caches go to host before the lends. ``to_host=False`` serves only the counter-examples:
    it leaves their pages on the GPU, for a lender's rule breach to show without the host tier, and
    the real lender then fails the check where the resumed cache is on its old pages."""
    with host_tier_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        original = [kit.page(mgr, source, 0, o) for o in range(BLOCKS)]
        target = kit.admitted(mgr, TARGET, BYSTANDERS[0])
        assert mgr.resize_context(target, target.context_remaining_length)
        lender = attach(mgr, max_fetches=2)
        writer = target
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
            for lease, reading in zip(failed, (True, False)):
                use_as_a_backend(kit, mgr, lender, lease, reading, original)
            mgr._stream.synchronize()
            now = {slot: dev.read(0, slot) for slot in theirs}
            assert kit.digest(now) == kit.digest(theirs), REACHED
            for lease in failed:
                assert lease.failure is not None, "not failed at the call"
                assert lease.poll() is None
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
            assert kit.digest(staged(lender, view)) == kit.digest([original])
            write = lender.lend_write(writer, 0, END)
            write_view = ready_view(write, mgr)
            write.mark_arrived(write_view.row_masks())
            write.release()
            read.release()
        finally:
            others.free()


# -- (d) a committed block's name does not depend on its tier -----------------------------------


def test_a_committed_block_has_one_name_on_the_gpu_and_after_a_trip_to_host(
    kit, host_tier_manager, attach=attach
):
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


# -- (f) a pool rebalance while a fetch is lent -------------------------------------------------


class DeviceCopyGuard:
    """The CUDA driver, recording every ``cuMemcpyAsync`` destination. With ``allowed``, a memcpy
    into device memory outside ``allowed()`` is recorded as stray and not run: after a pool shrinks,
    the slot it names may be unmapped."""

    def __init__(self, real_driver, device_ranges=(), allowed=None):
        self._real = real_driver
        self._device = device_ranges  # every byte a device pool ever held
        self._allowed = allowed  # () -> [(begin, end)] the copies may write
        self.copies = []
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
            self.copies.append(begin)
            if self._allowed is not None and any(lo <= begin < hi for lo, hi in self._device):
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


def test_a_rebalance_under_a_fetch_moves_no_copy_off_the_pages_still_lent(
    kit, host_tier_manager, monkeypatch, attach=attach
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
        stayed = [
            (run.layer_group, o)
            for run in view.runs
            for o in run.ordinals.tolist()
            if now[run.layer_group][o] == lent[run.layer_group][o] >= 0
        ]
        wrong = [
            f"{lg}:{o}"
            for lg, o in stayed
            if kit.page(mgr_b, target, lg, o) != kit.page(mgr_a, source, lg, o)
        ]
        assert stayed and not wrong, (
            f"rows {wrong} of {stayed}, whose pages stayed, hold other bytes"
        )
        assert lender_b.readiness(target) == (0, WINDOWED_END), "counts rows that never landed"
        publish.release()


# -- (f2) a pool rebalance with every kind of lease in flight ------------------------------------

REB_END = 96  # three whole blocks
REB_PROMPTS = {
    "queued_read": list(range(11_000, 11_097)),  # a read whose copy is queued behind the stream
    "waiting_read": list(range(12_000, 12_097)),  # a read waiting for slots
    "unmarked_write": list(range(13_000, 13_097)),  # a write granted, marked after the rebalance
    "marked_write": list(range(14_000, 14_097)),  # a write marked before, its copy queued
}
REB_GATE_SECONDS = 4.0
# One pool group, its quota halved with resize(level 0): every page above the new capacity moves
# or goes to host. Two pool groups, the target ratio moved so that adjust(), the executor's own
# rebalance call, shrinks the full-attention group.
MECHANISMS = {
    "resize": dict(max_tokens=2048),
    "adjust": dict(windows=[WINDOW, 256], num_layers=3, max_tokens=2048),
}
STRAY = "a copy touched device memory outside the leased pages"


class PageCopyGuard:
    """The CUDA driver, recording every ``cuMemcpyAsync``. A copy whose device side lies in a device
    pool but outside ``allowed()`` (the pages the leased requests lock at the call) is stray; with
    ``block`` it is recorded and not run, since its slot may be unmapped."""

    def __init__(self, real_driver, device_ranges, allowed, block):
        self._real = real_driver
        self._device = device_ranges
        self._allowed = allowed
        self._block = block
        self.copies = 0
        self.stray = []

    def __getattr__(self, name):
        found = getattr(self._real, name)
        if name != "cuMemcpyAsync":
            return found

        def guarded(dst, src, nbytes, stream):
            self.copies += 1
            for side, address in (("dst", int(dst)), ("src", int(src))):
                if not any(lo <= address < hi for lo, hi in self._device):
                    continue
                if not DeviceCopyGuard._covered(address, address + int(nbytes), self._allowed()):
                    self.stray.append((side, address, int(nbytes)))
                    if self._block:
                        return (self._real.CUresult.CUDA_SUCCESS,)
            return found(dst, src, nbytes, stream)

        return guarded


def gpu_slot(kv, lg, ordinal):
    """The GPU slot the active cache ``kv`` locks for block ``ordinal`` of ``lg``, or -1."""
    if kv is None or not kv.is_active or ordinal >= kv.num_blocks:
        return -1
    return int(kv.get_base_page_indices(lg)[ordinal])


def published_bytes(kit, mgr, request, lg, ordinal, nbytes):
    """What ``published`` wrote into block ``ordinal`` of ``lg``."""
    keys = kit.chain_keys(kit.kv(mgr, request), request.get_tokens(0))
    return kit.fake_page(lg, keys[ordinal], nbytes)


def staged_right(kit, mgr, lender, view, request):
    """Per row of a read's view, whether its slot holds the block's published bytes."""
    out = {}
    for run in view.runs:
        nbytes = lender.parts[run.part].slot_bytes
        for ordinal, address in zip(run.ordinals.tolist(), run.addresses.tolist()):
            want = published_bytes(kit, mgr, request, run.layer_group, ordinal, nbytes)
            out[f"{run.layer_group}:{ordinal}"] = kit.host_bytes(address, nbytes) == want
    return out


def landed(kit, mgr, request, rows, pub, pub_request, dev, dev_pub):
    """Per row of ``rows``, whether the request's page holds the publisher's bytes; ``None`` where
    its cache locks no page."""
    kv, kv_pub = kit.kv(mgr, request), kit.kv(pub, pub_request)
    out = {}
    for lg, ordinals in rows:
        for o in ordinals:
            slot, slot_pub = gpu_slot(kv, lg, o), gpu_slot(kv_pub, lg, o)
            assert slot_pub >= 0, f"the publisher holds no page for {lg}:{o}"
            out[f"{lg}:{o}"] = (
                None if slot < 0 else dev.read(lg, slot) == dev_pub.read(lg, slot_pub)
            )
    return out


def read_at(kit, mgr, lg, position):
    """The block ordinals a start at ``position`` reads in layer group ``lg``."""
    blocks = list(range(position // TPB))
    if kit.windows(mgr)[lg] is None:
        return blocks
    beg, end = kit.stale_blocks(mgr, lg, position)
    return [o for o in blocks if not beg <= o < end]


def locked(kit, mgr, request):
    kv = kit.kv(mgr, request)
    return [
        [int(s) for s in kv.get_base_page_indices(lg)[: kv.num_blocks]]
        for lg in range(kit.num_layer_groups(mgr))
    ]


def leased_ranges(kit, mgr, caches):
    """Byte ranges of the pages the active ``caches`` lock, in every pool of their groups."""
    out = []
    for kv in caches:
        if kv is not None and kv.is_active:
            out.extend(page_ranges(kit, mgr, kv, range(kit.num_layer_groups(mgr))))
    return out


def rebalance_with_leases_in_flight(
    kit, host_tier_manager, monkeypatch, mechanism, attach=attach, block=False, stay=False
):
    """The executor's rebalance (suspend every cache, ``resize`` or ``adjust`` the pools, resume)
    with four leases in flight: a read whose copy is queued behind a held stream, a read waiting for
    slots, a granted write marked after the rebalance, and a write marked before, its copy queued.
    The rebalance moves their pages, or with ``stay`` leaves the waiting read's and the unmarked
    write's in place. Bytes are checked against the pages ``published`` wrote, and a guard on the
    driver sees every copy into or out of a device pool."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender
    from tensorrt_llm.runtime.kv_cache_manager_v2 import _introspection

    shape = MECHANISMS[mechanism]
    with host_tier_manager(**shape) as mgr_a, host_tier_manager(**shape) as mgr_b:
        num_lg = kit.num_layer_groups(mgr_b)
        full = kit.windows(mgr_b).index(None)
        moving = kit.pool_group_of(mgr_b)[full]  # the pool group the rebalance shrinks
        # The publisher of what the writes fetch.
        pubs = {
            k: kit.published(mgr_a, 30 + i, REB_PROMPTS[k])
            for i, k in enumerate(("unmarked_write", "marked_write"))
        }
        lender_a = attach(mgr_a, fetch_tokens=REB_END, max_fetches=2)
        pub_leases = {k: lender_a.lend_read(p, 0, REB_END) for k, p in pubs.items()}
        pub_views = {k: ready_view(lease, mgr_a) for k, lease in pub_leases.items()}
        stats = mgr_b.impl.get_storage_statistics

        def used(group):
            return int(stats(0)[group].total) - int(stats(0)[group].free)

        groups = range(len(stats(0)))
        totals = [int(stats(0)[g].total) for g in groups]
        total = totals[moving]
        goal = int(total * 0.6)
        lender_b = attach(mgr_b, fetch_tokens=REB_END, max_fetches=3)
        leases, views, masks, early = {}, {}, {}, {}

        def lend_write(k, request):
            leases[k] = lender_b.lend_write(request, 0, REB_END)
            views[k] = leases[k].poll()
            assert views[k] is not None, leases[k].failure
            kit.fill_sentinel(mgr_b, request)  # a row that never lands shows
            masks[k] = kit.relay(lender_a, pub_views[k], lender_b, views[k])

        if stay:
            # Admitted and grown before the fillers: their pages sit low, below what moves.
            early["waiting_read"] = kit.published(mgr_b, 22, REB_PROMPTS["waiting_read"])
            early["unmarked_write"] = kit.admitted(mgr_b, 23, REB_PROMPTS["unmarked_write"])
            lend_write("unmarked_write", early["unmarked_write"])
        # Fillers take the low slots (a fresh pool hands out slot ids in order), so the leased
        # requests' pages sit above them; another pool group may fill first, so stop at 70% of it.
        fillers = kit.Requests(mgr_b)
        while (
            used(moving) < goal
            and all(used(g) < int(totals[g] * 0.7) for g in groups if g != moving)
            and fillers.allocate(min(7, goal - used(moving)))
        ):
            pass
        reqs = {
            "queued_read": kit.published(mgr_b, 21, REB_PROMPTS["queued_read"]),
            "waiting_read": early.get("waiting_read")
            or kit.published(mgr_b, 22, REB_PROMPTS["waiting_read"]),
            "unmarked_write": early.get("unmarked_write")
            or kit.admitted(mgr_b, 23, REB_PROMPTS["unmarked_write"]),
            "marked_write": kit.admitted(mgr_b, 24, REB_PROMPTS["marked_write"]),
        }
        bystanders = [kit.published(mgr_b, 25 + i, t) for i, t in enumerate(BYSTANDERS)]
        for k in ("unmarked_write", "marked_write"):
            if k not in leases:
                lend_write(k, reqs[k])
        rows = {
            k: [(run.layer_group, run.ordinals.tolist()) for run in views[k].runs]
            for k in ("unmarked_write", "marked_write")
        }
        fillers.free()
        dev = kit.DevicePages(mgr_b)  # built before the gate: building it waits for the device
        before = {k: locked(kit, mgr_b, r) for k, r in reqs.items()}
        for k in ("queued_read", "waiting_read"):  # the oracle holds before anything moves
            kv = kit.kv(mgr_b, reqs[k])
            for lg in range(num_lg):
                for o in range(REB_END // TPB):
                    slot = gpu_slot(kv, lg, o)
                    if slot >= 0:
                        want = published_bytes(kit, mgr_b, reqs[k], lg, o, dev.page_bytes(lg))
                        assert dev.read(lg, slot) == want, f"the oracle is wrong for {k} {lg}:{o}"
        device = []
        for pg in mgr_b.impl.pool_group_descs:
            for pool in pg.pools:
                base = int(pool.base_address)
                device.append((base, base + int(pg.num_slots) * int(pool.slot_bytes)))
        caches = [kit.kv(mgr_b, r) for r in reqs.values()]
        guard = PageCopyGuard(_lender.drv, device, lambda: leased_ranges(kit, mgr_b, caches), block)
        monkeypatch.setattr(_lender, "drv", guard)
        with kit.gated_stream(mgr_b._stream, open_after=REB_GATE_SECONDS) as gate:
            leases["marked_write"].mark_arrived(masks["marked_write"])  # its copy queues
            leases["marked_write"].release()
            leases["queued_read"] = lender_b.lend_read(reqs["queued_read"], 0, REB_END)
            leases["waiting_read"] = lender_b.lend_read(reqs["waiting_read"], 0, REB_END)
            waiting = (
                leases["waiting_read"].poll() is None and leases["waiting_read"].failure is None
            )
            granted = leases["queued_read"].failure is None
            ids = list(mgr_b.kv_cache_map)
            for rid in ids:  # the executor's rebalance: suspend every cache, rebalance, resume
                mgr_b.suspend_request(SimpleNamespace(py_request_id=rid))
            if mechanism == "adjust":
                current = _introspection.current_gpu_ratio(mgr_b.impl)
                assert len(current) >= 2, f"one pool group: {current}"
                # Shrink the group below the lowest slot a leased request holds there, so every
                # leased page in it moves.
                lowest = min(
                    s
                    for k in reqs
                    if not (stay and k in ("waiting_read", "unmarked_write"))
                    for s in before[k][full]
                    if s >= 0
                )
                factor = min(0.45, max(lowest - 1, 1) / total)
                target = list(current)
                target[moving] = current[moving] * factor
                scale = (1.0 - target[moving]) / (sum(target) - target[moving])
                target = [t if g == moving else t * scale for g, t in enumerate(target)]
                _introspection.set_num_sampled_kv_caches(mgr_b.impl, 2001)
                _introspection.set_last_adjustment_time(mgr_b.impl, 0.0)
                _introspection.set_target_ratio_list_gpu(mgr_b.impl, target)
                assert mgr_b.impl.need_adjustment
                mgr_b.impl.adjust()
                rebalanced = True
            else:
                quota = int(mgr_b.impl.get_quota(0))
                rebalanced = any(
                    bool(mgr_b.impl.resize(0, int(quota * fraction))) for fraction in (0.5, 0.55)
                )
            resumed = [
                bool(mgr_b.resume_request(SimpleNamespace(py_request_id=rid))) for rid in ids
            ]
            if not gate.opened:
                gate.open()
        mgr_b._stream.synchronize()
        after = {k: locked(kit, mgr_b, r) for k, r in reqs.items()}
        moved = {
            k: [sorted(set(b) - {-1}) != sorted(set(a) - {-1}) for b, a in zip(before[k], after[k])]
            for k in reqs
        }
        dev = kit.DevicePages(mgr_b)  # the pools changed size
        dev_a = kit.DevicePages(mgr_a)
        # The queued read's copy was queued before the rebalance.
        view_q = ready_view(leases["queued_read"], mgr_b) if granted else None
        staged_q = (
            staged_right(kit, mgr_b, lender_b, view_q, reqs["queued_read"]) if view_q else None
        )
        # The unmarked write is marked after the rebalance.
        leases["unmarked_write"].mark_arrived(masks["unmarked_write"])
        leases["unmarked_write"].release()
        mgr_b._stream.synchronize()
        # The waiting read is granted once the queued read's slots return.
        leases["queued_read"].release()
        view_w = None
        for _ in range(3):
            mgr_b._stream.synchronize()
            view_w = leases["waiting_read"].poll()
            if view_w is not None or leases["waiting_read"].failure is not None:
                break
        staged_w = (
            staged_right(kit, mgr_b, lender_b, view_w, reqs["waiting_read"]) if view_w else None
        )
        mgr_b._stream.synchronize()
        readiness = {}
        for k in ("unmarked_write", "marked_write"):
            got = lender_b.readiness(reqs[k])
            readiness[k] = None if got is None else tuple(got)
        arrived = {
            k: landed(kit, mgr_b, reqs[k], rows[k], mgr_a, pubs[k], dev, dev_a)
            for k in ("unmarked_write", "marked_write")
        }
        # Bystanders keep their own bytes wherever their pages went.
        bystanders_kept = []
        for b in bystanders:
            kv = kit.kv(mgr_b, b)
            for lg in range(num_lg):
                for o in range(len(BYSTANDERS[0]) // TPB):
                    slot = gpu_slot(kv, lg, o)
                    if slot >= 0:
                        want = published_bytes(kit, mgr_b, b, lg, o, dev.page_bytes(lg))
                        bystanders_kept.append(dev.read(lg, slot) == want)
        # Readiness counts only what landed: every block a start at usable_until reads.
        unbacked = {}
        for k in ("unmarked_write", "marked_write"):
            if readiness[k] is None:
                unbacked[k] = ["unsettled"]
                continue
            unbacked[k] = [
                f"{lg}:{o}"
                for lg in range(num_lg)
                for o in read_at(kit, mgr_b, lg, readiness[k][0])
                if arrived[k].get(f"{lg}:{o}") is not True
            ]
        for lease in list(leases.values()) + list(pub_leases.values()):
            lease.release()
        # The rebalance ran and moved what the scenario needs.
        assert rebalanced, "the rebalance did not run"
        assert all(resumed), f"a cache did not resume: {resumed}"
        assert waiting and granted, "the leases were not in the states the scenario needs"
        if stay:
            assert any(moved["queued_read"]) and any(moved["marked_write"]), moved
            assert not any(moved["waiting_read"]) and not any(moved["unmarked_write"]), moved
        else:
            assert all(any(m) for m in moved.values()), f"leased pages stayed: {moved}"
        # What the docstrings promise.
        assert guard.stray == [], f"{STRAY}: {guard.stray[:4]}"
        assert staged_q is not None and all(staged_q.values()), (
            f"the queued read staged wrong bytes: {staged_q}"
        )
        assert view_w is not None or leases["waiting_read"].failure is not None, (
            "the waiting read never settled"
        )
        if view_w is not None:
            assert all(staged_w.values()), f"the waiting read staged wrong bytes: {staged_w}"
        if stay:
            assert view_w is not None, (
                f"the waiting read failed with its pages in place: {leases['waiting_read'].failure}"
            )
            assert all(arrived["unmarked_write"].values()), (
                f"the unmarked write lost rows that stayed: {arrived['unmarked_write']}"
            )
            assert readiness["unmarked_write"][0] == REB_END, (
                f"readiness misses rows that landed: {readiness['unmarked_write']}"
            )
        assert all(not v for v in unbacked.values()), (
            f"readiness counts rows that never landed: {unbacked}"
        )
        assert all(arrived["marked_write"].values()), (
            f"the queued write's rows were lost: {arrived['marked_write']}"
        )
        assert all(bystanders_kept), "a bystander's pages changed"


@pytest.mark.parametrize("stay", [False, True], ids=["pages_move", "pages_stay"])
@pytest.mark.parametrize("mechanism", sorted(MECHANISMS))
def test_a_rebalance_with_every_kind_of_lease_in_flight(
    kit, host_tier_manager, monkeypatch, mechanism, stay
):
    rebalance_with_leases_in_flight(kit, host_tier_manager, monkeypatch, mechanism, stay=stay)


# -- (g) a mark arriving after the window passed the lent blocks --------------------------------

LATE_PROMPT = list(range(2000, 2161))  # five whole blocks and one token
LATE_END = 2 * TPB  # the lease covers blocks 0 and 1
LATE_HISTORY = 5 * TPB  # a served prefix moves the history here: blocks 0 to 2 leave the window
FETCHED = 0xAB


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


def late_mark(kit, host_tier_manager, attach, windows, fillers, keep):
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
            log = DeviceCopyGuard(_lender.drv)
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(_lender, "drv", log)
                lease.mark_arrived(view.row_masks(True))
                lease.release()
                mgr._stream.synchronize()
            written = [device_slot(mgr, dst) for dst in log.copies]
            hit = sorted({w for w in written if w is not None and w in owners})
            changed = sorted(watched[s] for s in watched if dev.read(sliding, s) != before[s])
            return dict(hit=hit, changed=changed)
        finally:
            others.free()
            first.free()


def test_a_late_mark_writes_no_slot_its_window_left(kit, host_tier_manager, attach=attach):
    # Slots go out lowest first, so a short search finds a lent block on host under its lent GPU
    # slot's number while another request locks that GPU slot.
    runs = [(w, n, k) for w in ([WINDOW], [WINDOW, 256]) for k in (True, False) for n in range(4)]
    for windows, fillers, keep in runs:
        result = late_mark(kit, host_tier_manager, attach, windows, fillers, keep)
        if result is not None:
            break
    else:
        raise RuntimeError("no run put a lent block on host under its lent GPU slot's number")
    assert not result["hit"], f"the marks' copy wrote slots other requests lock: {result['hit']}"
    assert not result["changed"], f"requests {result['changed']} now hold the fetch"
