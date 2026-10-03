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
"""The staging lender's contract over real KV cache managers, through the public API alone. A rule
a lender could break unnoticed has its check run twice: against the real lender, and against a
subclass breaking exactly that rule, which the check must catch. Allocates device pools."""

import gc
import mmap
import threading
import weakref
from types import SimpleNamespace
from typing import List, Set

import numpy as np
import pytest
import torch
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import (
    InPlaceLender,
    Lease,
    PartsHold,
    Readiness,
    RegionView,
    StagingLender,
    StagingOptions,
    attach_in_place,
    attach_staging,
)
from tensorrt_llm._utils import prefer_pinned

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="allocates KV cache pools")

TPB = 32
WINDOW = 64
PROMPT = list(range(1000, 1097))  # three whole blocks and one token
OTHER_PROMPT = list(range(5000, 5097))
END = 96
BLOCKS = END // TPB
WINDOWED_PROMPT = list(range(2000, 2161))  # five whole blocks and one token
OTHER_WINDOWED_PROMPT = list(range(6000, 6161))
WINDOWED_END = 160
SOURCE, TARGET, OTHER = 1, 2, 3
SCOPE = b"lender-contract-suite"
# What a check raises when the lender under it breaks the rule it checks; a timeout or any other
# error fails the liar test instead.
CAUGHT = (AssertionError,)


def attach(mgr, *, fetch_tokens, max_fetches=1, max_bytes=None, scope=SCOPE):
    """The public attach, as an integrator makes it."""
    options = StagingOptions(fetch_tokens, max_fetches, max_bytes)
    return attach_staging(mgr, scope=scope, staging=options)


def attach_breaking(rules):
    """An attach installing a lender whose ``rules`` (method name -> function, or a function of the
    manager returning them) replace the real ones: a lender breaking exactly those rules."""

    def attach_rule_breaker(mgr, *, fetch_tokens, max_fetches=1, max_bytes=None, scope=SCOPE):
        from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

        chosen = rules(mgr) if callable(rules) else rules
        cls = type("RuleBreaker", (_lender.Staging,), dict(chosen))
        options = StagingOptions(fetch_tokens, max_fetches, max_bytes)
        return _lender._attach_staging(mgr, scope=scope, staging=options, cls=cls)

    return attach_rule_breaker


def real(name):
    """The real lender's rule ``name``, for a rule breaker to call around."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    return getattr(_lender.Staging, name)


def ready_view(lease, mgr, tries=3):
    """The lease's view once the copies queued so far have run; fails if it never comes."""
    for _ in range(tries):
        mgr._stream.synchronize()
        view = lease.poll()
        if view is not None:
            return view
    raise AssertionError(f"the lease never became ready: {lease.failure}")


def by_group(view):
    return {run.layer_group: run.ordinals.tolist() for run in view.runs}


def one_fetch_bytes(kit, mgr, fetch_tokens):
    """Bytes one fetch of ``fetch_tokens`` tokens needs, from the device pages: every block for full
    attention, a window's blocks for a window (no sinks here)."""
    dev = kit.DevicePages(mgr)
    total = 0
    for lg, window in enumerate(kit.windows(mgr)):
        blocks = -(-fetch_tokens // TPB)
        if window is not None:
            blocks = min(blocks, -(-window // TPB))
        total += blocks * dev.page_bytes(lg)
    return total


def check_staged(kit, mgr, lender, request, view):
    """Every row's slot, inside its part, holds the request's device page of that block."""
    kv = kit.kv(mgr, request)
    dev = kit.DevicePages(mgr)
    rows = 0
    for run in view.runs:
        part = lender.parts[run.part]
        assert part.slot_bytes == dev.page_bytes(run.layer_group)
        slots = kit.pages(kv, run.layer_group)
        for address, ordinal in zip(run.addresses.tolist(), run.ordinals.tolist()):
            assert part.address <= address < part.address + part.nbytes
            assert (address - part.address) % part.slot_bytes == 0
            device = kit.digest(dev.read(run.layer_group, slots[ordinal]))
            assert kit.digest(kit.host_bytes(address, part.slot_bytes)) == device
            rows += 1
    return rows


def state_of(kit, mgr, lender, request):
    """Capacity, history, committed tokens and readiness of the request, or what stands for none."""
    kv = kit.kv(mgr, request)
    try:
        readiness = lender.readiness(request)
    except ValueError:
        readiness = "no cache"
    if kv is None:
        return None, readiness
    return (kv.capacity, kv.history_length, kv.num_committed_tokens), readiness


# -- attach -----------------------------------------------------------------------------------


def test_attach_takes_only_a_v2_manager():
    with pytest.raises(TypeError, match="KVCacheManagerV2"):
        attach(SimpleNamespace(), fetch_tokens=END)
    with pytest.raises(TypeError, match="KVCacheManagerV2"):
        attach_in_place(SimpleNamespace())


def test_a_manager_takes_one_lender_for_its_life(real_manager):
    with real_manager() as mgr:
        with pytest.raises(TypeError):
            attach_staging(mgr, scope="text", staging=StagingOptions(END))
        lender = attach(mgr, fetch_tokens=END)  # the refused call attached nothing
        assert isinstance(lender, StagingLender)
        with pytest.raises(ValueError, match="already attached"):
            attach(mgr, fetch_tokens=END)
        with pytest.raises(ValueError, match="already attached"):
            attach_in_place(mgr)


def test_recurrent_state_is_refused(kit, hybrid_manager):
    before = len(kit.retained())
    with hybrid_manager() as mgr:
        with pytest.raises(ValueError, match="recurrent"):
            attach(mgr, fetch_tokens=256)
        with pytest.raises(ValueError, match="recurrent"):
            attach_in_place(mgr)
    assert len(kit.retained()) == before, "a refused attach allocated staging"


def stand_in(cp_type=None):
    """The smallest manager the attach path reads before any other work: an uninitialised
    ``KVCacheManagerV2`` whose mapping is rank 0 of two ``cp_type`` ranks, or has no mapping."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm.mapping import Mapping

    manager = KVCacheManagerV2.__new__(KVCacheManagerV2)
    if cp_type is not None:
        manager.mapping = Mapping(world_size=2, cp_size=2, rank=0, cp_config={"cp_type": cp_type})
    return manager


def refusal(call, kind=ValueError) -> str:
    """The message of the ``kind`` error ``call`` raises; the check fails on anything else."""
    try:
        call()
    except kind as error:
        return str(error)
    except AssertionError:
        raise
    except Exception as error:  # handed to the check as its failure
        raise AssertionError(f"expected a {kind.__name__}, got {error!r}") from error
    raise AssertionError(NOT_REFUSED)


PAST_THE_CHECKS = "the attach got past its checks"
NOT_REFUSED = "the call went through"


def check_refused_before_any_other_work(monkeypatch, cases):
    """Each ``(manager, kind, words)`` is refused by both attaches with a ``kind`` error naming
    ``words``, before the layout is derived: a tripwire there fails the check."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    def tripwire(manager):
        raise AssertionError(PAST_THE_CHECKS)

    monkeypatch.setattr(_lender, "derive_layout", tripwire)
    before = len(_lender._retained())
    for make, kind, words in cases:
        staging, in_place = make(), make()
        assert words in refusal(lambda: attach(staging, fetch_tokens=END), kind)
        assert words in refusal(lambda: attach_in_place(in_place), kind)
        assert "_sharing" not in vars(staging) and "_sharing" not in vars(in_place)
    assert len(_lender._retained()) == before, "a refused attach allocated staging"


CONTEXT_PARALLEL = [
    (lambda: stand_in("HELIX"), ValueError, "context parallelism"),
    (lambda: stand_in("ULYSSES"), ValueError, "context parallelism"),
]
NO_MAPPING = [(stand_in, TypeError, "mapping")]


def test_context_parallelism_is_refused(monkeypatch):
    check_refused_before_any_other_work(monkeypatch, CONTEXT_PARALLEL)


def test_the_check_catches_a_lender_accepting_context_parallelism(monkeypatch):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    monkeypatch.setattr(_lender, "_context_parallel_size", lambda manager: 1)
    with pytest.raises(CAUGHT, match=PAST_THE_CHECKS):
        check_refused_before_any_other_work(monkeypatch, CONTEXT_PARALLEL)


def test_a_manager_without_a_mapping_is_refused(monkeypatch):
    check_refused_before_any_other_work(monkeypatch, NO_MAPPING)


def test_the_check_catches_a_lender_defaulting_a_missing_mapping(monkeypatch):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    def defaulting(manager):
        return int(getattr(getattr(manager, "mapping", None), "cp_size", 1))

    monkeypatch.setattr(_lender, "_context_parallel_size", defaulting)
    with pytest.raises(CAUGHT, match=PAST_THE_CHECKS):
        check_refused_before_any_other_work(monkeypatch, NO_MAPPING)


def unpaired_draft(mgr) -> None:
    """The state of a draft manager without joint reuse: it never publishes blocks for reuse."""
    mgr._can_publish_block_reuse = False


COMMITS_NOTHING = {
    "block_reuse_off": ({"enable_block_reuse": False}, None),
    "unpaired_draft": ({}, unpaired_draft),
}


def check_staging_refuses_a_manager_that_commits_nothing(kit, real_manager, attach, case):
    """Staging publishes committed blocks only, so a manager that commits none is refused at the
    attach, which then changed nothing; the in-place lender, which lends pages, attaches."""
    manager_kwargs, adjust = COMMITS_NOTHING[case]
    with real_manager(**manager_kwargs) as mgr:
        if adjust is not None:
            adjust(mgr)
        before = len(kit.retained())
        message = refusal(lambda: attach(mgr, fetch_tokens=END))
        assert "block reuse" in message, message
        assert "_sharing" not in vars(mgr) and len(kit.retained()) == before
        assert isinstance(attach_in_place(mgr), InPlaceLender)


@pytest.mark.parametrize("case", list(COMMITS_NOTHING))
def test_staging_refuses_a_manager_that_commits_no_blocks(kit, real_manager, case):
    check_staging_refuses_a_manager_that_commits_nothing(kit, real_manager, attach, case)


@pytest.mark.parametrize("case", list(COMMITS_NOTHING))
def test_the_check_catches_a_staging_attach_ignoring_block_reuse(
    kit, real_manager, monkeypatch, case
):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    monkeypatch.setattr(_lender, "_check_commits", lambda manager: None)
    with pytest.raises(CAUGHT, match=NOT_REFUSED):
        check_staging_refuses_a_manager_that_commits_nothing(kit, real_manager, attach, case)


# -- sizing -----------------------------------------------------------------------------------


def test_staging_holds_max_fetches_fetches_at_once(kit, real_manager):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        one = one_fetch_bytes(kit, mgr, END)
        lender = attach(mgr, fetch_tokens=END, max_fetches=2)
        (part,) = lender.parts
        assert part.slot_bytes == kit.DevicePages(mgr).page_bytes(0)
        assert part.nbytes == part.slots * part.slot_bytes == 2 * one
        leases = [lender.lend_read(source, 0, END) for _ in range(3)]
        for lease in leases[:2]:
            ready_view(lease, mgr)
        assert leases[2].poll() is None and leases[2].failure is None, "three fetches in two"
        leases[0].release()
        check_staged(kit, mgr, lender, source, ready_view(leases[2], mgr))
        for lease in leases[1:]:
            lease.release()


@pytest.mark.parametrize("cap", ["one_fetch", "two_and_a_half_fetches"])
def test_max_bytes_caps_staging_but_never_below_one_fetch(kit, real_manager, cap):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        one = one_fetch_bytes(kit, mgr, END)
        with pytest.raises(ValueError, match="one fetch"):
            attach(mgr, fetch_tokens=END, max_fetches=4, max_bytes=one - 1)
        max_bytes, holding = (one, 1) if cap == "one_fetch" else (2 * one + one // 2, 2)
        lender = attach(mgr, fetch_tokens=END, max_fetches=4, max_bytes=max_bytes)
        (part,) = lender.parts
        assert holding * one <= part.nbytes <= max_bytes
        leases = [lender.lend_read(source, 0, END) for _ in range(holding + 1)]
        for lease in leases[:holding]:
            ready_view(lease, mgr)
        assert leases[-1].poll() is None and leases[-1].failure is None
        for lease in leases:
            lease.release()


@pytest.mark.parametrize("num_layers", [2, 3], ids=["one_pool_group", "two_pool_groups"])
def test_a_windowed_range_of_fetch_tokens_always_fits(kit, real_manager, num_layers):
    with real_manager(windows=[WINDOW, 256], num_layers=num_layers) as mgr:
        source = kit.published(mgr, SOURCE, WINDOWED_PROMPT)
        target = kit.admitted(mgr, TARGET, OTHER_WINDOWED_PROMPT)
        lender = attach(mgr, fetch_tokens=WINDOWED_END)
        dev = kit.DevicePages(mgr)
        rows = {}
        for lg, (window, g) in enumerate(zip(kit.windows(mgr), kit.pool_group_of(mgr))):
            rows[g] = rows.get(g, 0) + (WINDOW // TPB if window else WINDOWED_END // TPB)
        groups = kit.pool_group_ids(mgr)
        assert len(lender.parts) == len(groups) == num_layers - 1
        for g, part in zip(groups, lender.parts):
            assert (part.slots, part.slot_bytes) == (rows[g], dev.group_page_bytes(g))
            assert part.nbytes == part.slots * part.slot_bytes
        read = lender.lend_read(source, 0, WINDOWED_END)  # takes every slot
        assert check_staged(kit, mgr, lender, source, ready_view(read, mgr)) == sum(rows.values())
        read.release()
        write = lender.lend_write(target, 0, WINDOWED_END)
        view = ready_view(write, mgr)
        assert view.num_rows == sum(rows.values())
        write.mark_arrived(view.row_masks())
        write.release()


def check_the_staging_memory_is_the_parts_and_nothing_more(kit, real_manager, attach):
    # Three layers: two pool groups, so two parts.
    with real_manager(windows=[WINDOW, 256], num_layers=3) as mgr:
        lender = attach(mgr, fetch_tokens=WINDOWED_END, max_fetches=2)
        parts = lender.parts
        assert len(parts) == len(kit.pool_group_ids(mgr)) == 2
        (memory,) = kit.staging_memory(parts)
        assert memory.address <= parts[0].address < memory.address + memory.nbytes
        assert memory.nbytes == sum(p.nbytes for p in parts), "staging is not the sum of its parts"


def test_the_staging_memory_is_the_parts_and_nothing_more(kit, real_manager):
    check_the_staging_memory_is_the_parts_and_nothing_more(kit, real_manager, attach)


def test_the_check_catches_a_lender_allocating_more_than_its_parts(kit, real_manager, monkeypatch):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    allocate = _lender._allocate
    monkeypatch.setattr(_lender, "_allocate", lambda nbytes: allocate(nbytes + 4096))
    with pytest.raises(CAUGHT, match="not the sum of its parts"):
        check_the_staging_memory_is_the_parts_and_nothing_more(kit, real_manager, attach)


class FreeSpy:
    """The CUDA runtime, recording the address of every page-locked host allocation freed."""

    def __init__(self, runtime):
        self._runtime = runtime
        self.freed = []

    def __getattr__(self, name):
        found = getattr(self._runtime, name)
        if name != "cudaFreeHost":
            return found

        def free(address):
            self.freed.append(int(address))
            return found(address)

        return free


def check_the_staging_memory_is_pinned_at_its_size_and_freed_once(
    kit, real_manager, attach, monkeypatch
):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    spy = FreeSpy(_lender.cudart)
    monkeypatch.setattr(_lender, "cudart", spy)
    with real_manager() as mgr:
        parts = attach(mgr, fetch_tokens=END, max_fetches=3).parts
        base, total = parts[0].address, sum(p.nbytes for p in parts)
        whole_pages = -(-total // mmap.PAGESIZE) * mmap.PAGESIZE
        assert 1 << (total - 1).bit_length() > whole_pages, "a power of two would pass unseen"
        start, pinned = kit.pinned_range(base)
        assert start == base and total <= pinned <= whole_pages, (
            f"{pinned} bytes pinned for {total}"
        )
        mgr.shutdown()
        assert spy.freed == [base] and kit.pinned_range(base) is None, "not freed at the shutdown"
        mgr.shutdown()
    gc.collect()
    assert spy.freed == [base], "freed more than once"


def test_the_staging_memory_is_pinned_at_its_size_and_freed_once(kit, real_manager, monkeypatch):
    if not prefer_pinned():
        pytest.skip("staging is pageable where pinning does not pay off")
    check_the_staging_memory_is_pinned_at_its_size_and_freed_once(
        kit, real_manager, attach, monkeypatch
    )


def test_the_check_catches_a_lender_pinning_through_torch_s_rounding_allocator(
    kit, real_manager, monkeypatch
):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    class ThroughTorch(_lender._HostMemory):
        def __init__(self, nbytes):
            self.nbytes = nbytes
            self._tensor = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
            self.address = self._tensor.data_ptr()

        def free(self):
            self.address, self._tensor = 0, None

    monkeypatch.setattr(_lender, "_HostMemory", ThroughTorch)
    with pytest.raises(CAUGHT, match="bytes pinned for"):
        check_the_staging_memory_is_pinned_at_its_size_and_freed_once(
            kit, real_manager, attach, monkeypatch
        )


def test_the_check_catches_a_lender_never_freeing_the_staging_memory(
    kit, real_manager, monkeypatch
):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    if not prefer_pinned():
        pytest.skip("staging is pageable where pinning does not pay off")
    monkeypatch.setattr(_lender._HostMemory, "free", lambda self: None)
    with pytest.raises(CAUGHT, match="not freed at the shutdown"):
        check_the_staging_memory_is_pinned_at_its_size_and_freed_once(
            kit, real_manager, attach, monkeypatch
        )


# -- publish ----------------------------------------------------------------------------------


def test_a_publish_lends_the_committed_pages_under_their_reuse_keys(kit, real_manager):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        lender = attach(mgr, fetch_tokens=END)
        lease = lender.lend_read(source, 0, END)
        assert isinstance(lease, Lease)
        view = ready_view(lease, mgr)
        assert isinstance(view, RegionView) and lease.poll() is view, "one view, every poll"
        (run,) = view.runs
        assert run.ordinals.tolist() == list(range(BLOCKS)) and run.part == 0
        assert check_staged(kit, mgr, lender, source, view) == BLOCKS
        kv = kit.kv(mgr, source)
        assert [bytes(name[16:48]) for name in run.names] == kit.chain_keys(kv, PROMPT)[:BLOCKS]
        assert len({bytes(name[:16]) for name in run.names}) == 1, "one namespace"
        assert len({bytes(name[48:]) for name in run.names}) == 1, "one layer group, one shard"
        # The manager committed under those keys: its tree finds the block after them by them.
        extra = list(range(7000, 7000 + TPB))
        probed = mgr.impl.probe_first_new_block_key(kv.reuse_scope, PROMPT[:END] + extra)
        assert probed == kit.chain_keys(kv, PROMPT[:END] + extra)[BLOCKS]
        lease.release()
        with pytest.raises(RuntimeError):
            lease.poll()


def test_a_prompt_ending_on_a_block_boundary_publishes_its_last_block(kit, real_manager):
    with real_manager() as mgr:
        tokens = PROMPT[:END]
        source = kit.published(mgr, SOURCE, tokens)
        lender = attach(mgr, fetch_tokens=END)
        lease = lender.lend_read(source, 0, END)
        (run,) = ready_view(lease, mgr).runs
        keys = kit.chain_keys(kit.kv(mgr, source), tokens)
        assert len(keys) == BLOCKS and [bytes(n[16:48]) for n in run.names] == keys
        lease.release()


def test_a_multimodal_publish_is_named_by_the_keys_the_manager_commits(kit, real_manager):
    image = dict(multimodal_positions=[40], multimodal_lengths=[16])
    with real_manager() as mgr:
        first = kit.published(mgr, SOURCE, PROMPT, multimodal_hashes=[list(range(1, 9))], **image)
        second = kit.published(
            mgr, OTHER, PROMPT, multimodal_hashes=[list(range(8, 0, -1))], **image
        )
        lender = attach(mgr, fetch_tokens=END, max_fetches=2)
        leases = [lender.lend_read(r, 0, END) for r in (first, second)]
        names = [ready_view(lease, mgr).runs[0].names for lease in leases]
        keys = [[bytes(n[16:48]) for n in run_names] for run_names in names]
        kv = kit.kv(mgr, first)
        augmented = list(mgr._augment_tokens_for_block_reuse(PROMPT, first))
        assert keys[0] == kit.chain_keys(kv, augmented)[:BLOCKS]
        extra = list(range(7000, 7000 + TPB))
        probed = mgr.impl.probe_first_new_block_key(kv.reuse_scope, augmented[:END] + extra)
        assert probed == kit.chain_keys(kv, augmented[:END] + extra)[BLOCKS]
        # The image sits in block 1: the block before it is shared, the rest is not.
        assert keys[0][0] == keys[1][0] and not set(keys[0][1:]) & set(keys[1][1:])
        assert keys[0][1:] != kit.chain_keys(kv, PROMPT)[1:BLOCKS], "the image changes the keys"
        for lease in leases:
            lease.release()


def test_a_windowed_publish_lends_only_what_the_window_at_its_end_reads(kit, real_manager):
    with real_manager(windows=[WINDOW, 256]) as mgr:
        source = kit.published(mgr, SOURCE, WINDOWED_PROMPT)
        lender = attach(mgr, fetch_tokens=WINDOWED_END)
        windows = kit.windows(mgr)
        sliding, full = windows.index(WINDOW), windows.index(None)
        lease = lender.lend_read(source, 0, WINDOWED_END)
        view = ready_view(lease, mgr)
        beg, end = kit.stale_blocks(mgr, sliding, WINDOWED_END)
        assert end > beg, "the window must leave early blocks behind"
        blocks = range(WINDOWED_END // TPB)
        in_window = [o for o in blocks if not beg <= o < end]
        assert by_group(view) == {full: list(blocks), sliding: in_window}
        addresses = np.concatenate([run.addresses for run in view.runs])
        assert len(set(addresses.tolist())) == len(addresses), "every row has its own slot"
        assert check_staged(kit, mgr, lender, source, view) == len(addresses)
        lease.release()

        # The request's own window has since passed what a window at 96 reads: none is lent.
        early = lender.lend_read(source, 0, END)
        rows = by_group(ready_view(early, mgr))
        history = kit.kv(mgr, source).history_length
        passed_beg, passed_end = kit.stale_blocks(mgr, sliding, history)
        assert rows[full] == list(range(BLOCKS))
        assert [o for o in rows.get(sliding, []) if passed_beg <= o < passed_end] == []
        assert rows.get(sliding, []) == []
        early.release()


def test_a_range_without_rows_is_ready_at_once_and_waits_for_no_one(kit, real_manager):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        lender = attach(mgr, fetch_tokens=END)
        holder = lender.lend_read(source, 0, END)  # every slot
        ready_view(holder, mgr)
        empty = lender.lend_read(source, END, END)
        view = empty.poll()
        assert view is not None and view.num_rows == 0
        assert len(view.runs) == kit.num_layer_groups(mgr)
        empty.release()
        holder.release()


# -- ready means copied -----------------------------------------------------------------------


def check_ready_means_copied(kit, real_manager, attach):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        lender = attach(mgr, fetch_tokens=END)
        with kit.held_stream(mgr._stream) as gate:
            lease = lender.lend_read(source, 0, END)
            assert lease.poll() is None, "ready before its copy into staging ran"
            assert lease.poll() is None and lease.failure is None
            gate.open()
        check_staged(kit, mgr, lender, source, ready_view(lease, mgr))
        lease.release()


def test_a_publish_is_ready_only_once_its_copy_ran(kit, real_manager):
    check_ready_means_copied(kit, real_manager, attach)


def test_the_check_catches_a_lender_ready_before_its_copy(kit, real_manager):
    liar = attach_breaking({"_copy_landed": lambda self, lease: True})
    with pytest.raises(CAUGHT, match="ready before its copy into staging ran"):
        check_ready_means_copied(kit, real_manager, liar)


# -- no slot reused while a copy or a backend may touch it -----------------------------------


def check_slots_are_not_reused_while_touched(kit, real_manager, attach):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        first = kit.admitted(mgr, TARGET, OTHER_PROMPT)
        second = kit.admitted(mgr, OTHER, OTHER_WINDOWED_PROMPT)
        for request in (first, second):  # grown ahead, so the held section only lends
            assert mgr._resize_for_connector_prefix(request, kit.kv(mgr, request), 0, END)
        lender = attach(mgr, fetch_tokens=END)  # one fetch: three slots
        with kit.held_stream(mgr._stream) as gate:
            read = lender.lend_read(source, 0, END)  # every slot, its copy held
            read.release()
            write = lender.lend_write(first, 0, END)
            assert write.poll() is None, "slots handed out while a copy into them was queued"
            gate.open()
        view = ready_view(write, mgr)
        write.release()  # seen ready and not marked: its backend may still write the slots
        waiting = lender.lend_write(second, 0, END)
        assert waiting.poll() is None, "slots handed out while a backend could still write them"
        write.mark_arrived(view.row_masks())
        waiting_view = ready_view(waiting, mgr)
        waiting.mark_arrived(waiting_view.row_masks())
        waiting.release()


def test_a_slot_returns_only_after_its_copy_and_its_marks(kit, real_manager):
    check_slots_are_not_reused_while_touched(kit, real_manager, attach)


def test_the_check_catches_a_lender_recycling_before_the_copy_lands(kit, real_manager):
    def recycle_early(self, lease):
        with kit.events_report_done():
            return real("_recyclable")(self, lease)

    with pytest.raises(CAUGHT, match="slots handed out while a copy into them was queued"):
        check_slots_are_not_reused_while_touched(
            kit, real_manager, attach_breaking({"_recyclable": recycle_early})
        )


def test_leases_wait_for_slots_strictly_in_order(kit, real_manager):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        lender = attach(mgr, fetch_tokens=END)  # three slots
        first = lender.lend_read(source, 0, 2 * TPB)
        ready_view(first, mgr)
        second = lender.lend_read(source, 0, END)
        third = lender.lend_read(source, 0, TPB)
        assert second.poll() is None
        assert third.poll() is None, "a later lease that fits jumped the one waiting ahead"
        first.release()
        ready_view(second, mgr)
        assert third.poll() is None
        second.release()
        assert by_group(ready_view(third, mgr)) == {0: [0]}
        third.release()


def test_max_fetches_is_a_budget_and_holes_can_make_a_lease_wait(kit, real_manager):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        lender = attach(mgr, fetch_tokens=END, max_fetches=2)
        (group,) = kit.pool_group_ids(mgr)
        (part,) = lender.parts
        assert part.slots == 2 * BLOCKS
        first = lender.lend_read(source, 0, 2 * TPB)  # slots 0 and 1
        second = lender.lend_read(source, 0, TPB)  # slot 2
        third = lender.lend_read(source, 0, TPB)  # slot 3
        for lease in (first, second, third):
            ready_view(lease, mgr)
        first.release()
        assert lender._free_slots(group) == 4  # slots 0, 1, 4 and 5
        fetch = lender.lend_read(source, 0, END)
        assert fetch.poll() is None and fetch.failure is None, "no three free slots in a row"
        second.release()  # slots 0 to 2 are free in a row
        (run,) = ready_view(fetch, mgr).runs
        assert ((run.addresses - part.address) // part.slot_bytes).tolist() == [0, 1, 2]
        for lease in (third, fetch):
            lease.release()


# -- fetch ------------------------------------------------------------------------------------


def check_a_fetch_settles_when_its_copy_lands(kit, real_manager, attach, local_blocks=0):
    with real_manager() as mgr_a, real_manager() as mgr_b:
        source = kit.published(mgr_a, SOURCE, PROMPT)
        lender_a = attach(mgr_a, fetch_tokens=END)
        publish = lender_a.lend_read(source, 0, END)
        publish_view = ready_view(publish, mgr_a)
        if local_blocks:
            kit.published(mgr_b, 7, PROMPT[: local_blocks * TPB] + list(range(9000, 9033)))
        target = kit.admitted(mgr_b, TARGET, PROMPT)
        kv = kit.kv(mgr_b, target)
        local, history = kv.num_committed_tokens, kv.history_length
        assert local == local_blocks * TPB
        lender_b = attach(mgr_b, fetch_tokens=END)
        assert lender_b.readiness(target) == (max(local, history), history)

        lease = lender_b.lend_write(target, local, END)
        assert lender_b.readiness(target) is None, "a fetch counts from its call"
        assert kv.capacity >= END and kv.history_length == history
        view = lease.poll()
        assert view is not None, "a write is ready at its first poll after the grant"
        assert by_group(view) == {0: list(range(local_blocks, BLOCKS))}
        kit.fill_sentinel(mgr_b, target, first_block=local_blocks)
        masks = kit.relay(lender_a, publish_view, lender_b, view)
        with kit.held_stream(mgr_b._stream) as gate:
            lease.mark_arrived(masks)
            assert lender_b.readiness(target) is None, "settled before the copy into pages ran"
            gate.open()
        readiness = lender_b.readiness(target)
        assert isinstance(readiness, Readiness) and all(type(v) is int for v in readiness)
        assert readiness == (END, min(local, history))
        for ordinal in range(BLOCKS):
            got = kit.digest(kit.page(mgr_b, target, 0, ordinal))
            assert got == kit.digest(kit.page(mgr_a, source, 0, ordinal)), "a fetched page differs"
        lease.release()
        publish.release()


@pytest.mark.parametrize("local_blocks", [0, 2], ids=["cold", "local_prefix"])
def test_a_fetch_is_usable_once_its_copy_lands(kit, real_manager, local_blocks):
    check_a_fetch_settles_when_its_copy_lands(kit, real_manager, attach, local_blocks)


def test_the_check_catches_readiness_before_the_copy(kit, real_manager):
    def settled_early(self, fetch):
        with kit.events_report_done():
            return real("_report_settled")(self, fetch)

    with pytest.raises(CAUGHT, match="settled before the copy into pages ran"):
        check_a_fetch_settles_when_its_copy_lands(
            kit, real_manager, attach_breaking({"_report_settled": settled_early})
        )


@pytest.mark.parametrize(
    "rows,usable",
    [((0, 1), 64), ((0, 2), 32), ((), 0), ((1, 2), 0)],
    ids=["prefix", "gap", "nothing", "no_first_block"],
)
def test_a_partial_arrival_is_usable_up_to_its_first_gap(kit, real_manager, rows, usable):
    with real_manager() as mgr_a, real_manager() as mgr_b:
        source = kit.published(mgr_a, SOURCE, PROMPT)
        lender_a = attach(mgr_a, fetch_tokens=END)
        publish = lender_a.lend_read(source, 0, END)
        publish_view = ready_view(publish, mgr_a)
        target = kit.admitted(mgr_b, TARGET, PROMPT)
        lender_b = attach(mgr_b, fetch_tokens=END)
        lease = lender_b.lend_write(target, 0, END)
        view = lease.poll()
        kit.fill_sentinel(mgr_b, target)
        kit.relay(lender_a, publish_view, lender_b, view, deliver=lambda run, row: row in rows)
        lease.mark_arrived((np.isin(np.arange(BLOCKS), rows),))
        mgr_b._stream.synchronize()
        assert lender_b.readiness(target) == (usable, 0)
        sentinel = bytes([kit.SENTINEL]) * kit.DevicePages(mgr_b).page_bytes(0)
        for ordinal in range(BLOCKS):
            expected = kit.page(mgr_a, source, 0, ordinal) if ordinal in rows else sentinel
            got = kit.digest(kit.page(mgr_b, target, 0, ordinal))
            assert got == kit.digest(expected), "an unmarked row was copied"
        lease.release()
        publish.release()


@pytest.mark.parametrize("partial", [False, True], ids=["complete", "partial"])
def test_a_windowed_fetch_is_usable_only_from_its_floor(kit, real_manager, partial):
    windows = [WINDOW, 256]
    with real_manager(windows=windows) as mgr_a, real_manager(windows=windows) as mgr_b:
        source = kit.published(mgr_a, SOURCE, WINDOWED_PROMPT)
        lender_a = attach(mgr_a, fetch_tokens=WINDOWED_END)
        publish = lender_a.lend_read(source, 0, WINDOWED_END)
        publish_view = ready_view(publish, mgr_a)
        target = kit.admitted(mgr_b, TARGET, WINDOWED_PROMPT)
        lender_b = attach(mgr_b, fetch_tokens=WINDOWED_END)
        lease = lender_b.lend_write(target, 0, WINDOWED_END)
        assert kit.kv(mgr_b, target).history_length == WINDOWED_END, "the history moves to the end"
        view = lease.poll()
        assert by_group(view) == by_group(publish_view)
        kit.fill_sentinel(mgr_b, target)
        deliver = (lambda run, row: row < 2) if partial else None
        masks = kit.relay(lender_a, publish_view, lender_b, view, deliver)
        lease.mark_arrived(masks)
        mgr_b._stream.synchronize()
        readiness = lender_b.readiness(target)
        assert readiness.restart_floor == WINDOWED_END, "the window released the early blocks"
        if partial:
            assert readiness.usable_until < readiness.restart_floor, "empty: compute from 0"
        else:
            assert readiness.usable_until == WINDOWED_END
            for run in view.runs:
                for ordinal in run.ordinals.tolist():
                    lg = run.layer_group
                    got = kit.digest(kit.page(mgr_b, target, lg, ordinal))
                    assert got == kit.digest(kit.page(mgr_a, source, lg, ordinal))
        lease.release()
        publish.release()


def test_mark_arrived_takes_one_mask_per_run_once_on_a_write_seen_ready(kit, real_manager):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        target = kit.admitted(mgr, TARGET, OTHER_PROMPT)
        lender = attach(mgr, fetch_tokens=END, max_fetches=2)
        read = lender.lend_read(source, 0, END)
        read_view = ready_view(read, mgr)
        with pytest.raises(RuntimeError):
            read.mark_arrived(read_view.row_masks())
        write = lender.lend_write(target, 0, END)
        with pytest.raises(RuntimeError):
            write.mark_arrived(())  # before poll() gave the view
        view = write.poll()
        bad = ((), (np.ones(BLOCKS - 1, bool),), (np.ones(BLOCKS, bool),) * 2)
        for masks in bad:
            with pytest.raises(ValueError):
                write.mark_arrived(masks)
            assert lender.readiness(target) is None, "a refused mask recorded something"
        write.mark_arrived(view.row_masks())
        with pytest.raises(RuntimeError):
            write.mark_arrived(view.row_masks())
        mgr._stream.synchronize()
        kv = kit.kv(mgr, target)
        assert lender.readiness(target) == (kv.num_committed_tokens, kv.history_length)
        write.release()
        read.release()


@pytest.mark.parametrize(
    "mark_first", [True, False], ids=["mark_then_release", "release_then_mark"]
)
def test_either_order_of_marks_and_release_copies_the_same(kit, real_manager, mark_first):
    with real_manager() as mgr:
        first = kit.admitted(mgr, TARGET, PROMPT)
        second = kit.admitted(mgr, OTHER, OTHER_PROMPT)
        lender = attach(mgr, fetch_tokens=END)  # the second waits for the first's slots
        lease = lender.lend_write(first, 0, END)
        view = lease.poll()
        kit.fill_sentinel(mgr, first)
        kit.stage(lender, view, 0x5A)
        waiting = lender.lend_write(second, 0, END)
        if not mark_first:
            lease.release()
            assert waiting.poll() is None, "slots returned before the marks"
        with kit.held_stream(mgr._stream) as gate:
            lease.mark_arrived(view.row_masks(True))
            lease.release()
            assert waiting.poll() is None, "slots returned before the copy out of them ran"
            gate.open()
        waiting_view = ready_view(waiting, mgr)
        staged = bytes([0x5A]) * kit.DevicePages(mgr).page_bytes(0)
        got = kit.digest([kit.page(mgr, first, 0, o) for o in range(BLOCKS)])
        assert got == kit.digest([staged] * BLOCKS)
        assert lender.readiness(first) == (END, 0)
        waiting.mark_arrived(waiting_view.row_masks())
        waiting.release()


# -- the request exits during a lease ---------------------------------------------------------


def check_arrivals_reach_only_pages_still_lent(kit, real_manager, attach):
    with real_manager(max_tokens=kit.POOL_TOKENS) as mgr:
        target = kit.admitted(mgr, TARGET, PROMPT)
        lender = attach(mgr, fetch_tokens=END)
        lease = lender.lend_write(target, 0, END)
        view = lease.poll()
        assert view is not None
        kv = kit.kv(mgr, target)
        lent = {kit.pages(kv, 0)[o] for o in view.runs[0].ordinals.tolist()}
        kit.stage(lender, view, 0x5A)
        mgr.free_resources(target)
        with pytest.raises(ValueError):
            lender.readiness(target)
        others = kit.Requests(mgr)
        try:
            assert others.allocate(kit.pool_pages(mgr))
            assert lent <= others.pages(), "no lent page was reused: the check proves nothing"
            lease.mark_arrived(view.row_masks(True))
            mgr._stream.synchronize()
            dev = kit.DevicePages(mgr)
            sentinel = bytes([kit.SENTINEL]) * dev.page_bytes(0)
            for slot in sorted(lent):
                got = kit.digest(dev.read(0, slot))
                assert got == kit.digest(sentinel), "an arrival overwrote another request's page"
        finally:
            lease.release()
            others.free()


def test_arrivals_for_a_freed_request_copy_nothing(kit, real_manager):
    check_arrivals_reach_only_pages_still_lent(kit, real_manager, attach)


def test_the_check_catches_a_lender_copying_into_pages_no_longer_lent(kit, real_manager):
    def always_lent(self, kv, lease):
        return [np.ones_like(m, dtype=bool) for m in real("_still_lent")(self, kv, lease)]

    with pytest.raises(CAUGHT, match="an arrival overwrote another request's page"):
        check_arrivals_reach_only_pages_still_lent(
            kit, real_manager, attach_breaking({"_still_lent": always_lent})
        )


def check_a_free_fails_the_request_s_waiting_leases(kit, real_manager, attach):
    with real_manager() as mgr:
        holder = kit.published(mgr, SOURCE, PROMPT)
        leaving = kit.published(mgr, OTHER, OTHER_PROMPT)
        lender = attach(mgr, fetch_tokens=END)
        head = lender.lend_read(holder, 0, END)
        ready_view(head, mgr)
        waiting = lender.lend_read(leaving, 0, 2 * TPB)
        assert waiting.poll() is None and waiting.failure is None
        mgr.free_resources(leaving)
        assert waiting.failure is not None, "a lease waiting for slots outlived its request"
        assert waiting.poll() is None
        waiting.release()
        head.release()
        again = lender.lend_read(holder, 0, END)
        ready_view(again, mgr)
        again.release()


def test_a_free_fails_the_request_s_waiting_leases(kit, real_manager):
    check_a_free_fails_the_request_s_waiting_leases(kit, real_manager, attach)


def test_the_check_catches_a_lender_keeping_a_freed_request_s_leases_in_line(kit, real_manager):
    liar = attach_breaking({"_fail_waiting": lambda self, request_id, reason: None})
    with pytest.raises(CAUGHT, match="a lease waiting for slots outlived its request"):
        check_a_free_fails_the_request_s_waiting_leases(kit, real_manager, liar)


def test_a_publish_granted_before_its_request_is_freed_copies_what_it_lent(kit, real_manager):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        original = [kit.page(mgr, source, 0, o) for o in range(BLOCKS)]
        lender = attach(mgr, fetch_tokens=END)
        with kit.held_stream(mgr._stream, strict=False) as gate:
            lease = lender.lend_read(source, 0, END)  # granted, its copy queued behind the gate
            mgr.free_resources(source)
            assert lease.failure is None and kit.kv(mgr, source) is None
            gate.open()
        (run,) = ready_view(lease, mgr).runs
        length = lender.parts[run.part].slot_bytes
        got = kit.digest([kit.host_bytes(a, length) for a in run.addresses.tolist()])
        assert got == kit.digest(original)
        lease.release()


def test_a_waiting_read_fails_when_its_cache_changed_before_its_grant(kit, real_manager):
    with real_manager() as mgr:
        holder = kit.published(mgr, SOURCE, PROMPT)
        other = kit.published(mgr, OTHER, OTHER_PROMPT)
        lender = attach(mgr, fetch_tokens=END)
        head = lender.lend_read(holder, 0, END)
        ready_view(head, mgr)
        waiting = lender.lend_read(other, 0, 2 * TPB)
        assert waiting.poll() is None and waiting.failure is None
        mgr.suspend_request(other)
        head.release()
        assert waiting.poll() is None and waiting.failure is not None, "copied a moved cache"
        waiting.release()
        again = lender.lend_read(holder, 0, END)  # its slots came back
        ready_view(again, mgr)
        again.release()


# -- abandoned fetches ------------------------------------------------------------------------


def check_an_abandoned_fetch_leaves_only_its_growth(kit, real_manager, attach, windowed):
    tokens, other, end = (
        (WINDOWED_PROMPT, OTHER_WINDOWED_PROMPT, WINDOWED_END)
        if windowed
        else (PROMPT, OTHER_PROMPT, END)
    )
    with real_manager(windows=[WINDOW, 256] if windowed else None) as mgr:
        target = kit.admitted(mgr, TARGET, tokens)
        second = kit.admitted(mgr, OTHER, other)
        lender = attach(mgr, fetch_tokens=end)
        lease = lender.lend_write(target, 0, end)
        assert lender.readiness(target) is None
        lease.release()  # before any poll: no backend wrote, the fetch is abandoned
        kv = kit.kv(mgr, target)
        if windowed:
            assert kv.history_length == end, "the growth moved the window's history"
        own = (kv.num_committed_tokens, kv.history_length)
        assert lender.readiness(target) == own, "the abandoned fetch still counts"
        again = lender.lend_write(second, 0, end)
        view = again.poll()
        assert view is not None, "the abandoned fetch kept its slots"
        again.mark_arrived(view.row_masks())
        again.release()


@pytest.mark.parametrize("windowed", [False, True], ids=["full", "windowed"])
def test_an_abandoned_fetch_leaves_only_its_growth(kit, real_manager, windowed):
    check_an_abandoned_fetch_leaves_only_its_growth(kit, real_manager, attach, windowed)


def test_the_check_catches_a_lender_keeping_an_abandoned_fetch(kit, real_manager):
    liar = attach_breaking({"_abandon": lambda self, lease: None})
    with pytest.raises(CAUGHT, match="the abandoned fetch still counts"):
        check_an_abandoned_fetch_leaves_only_its_growth(kit, real_manager, liar, windowed=True)


# -- a fetch split into leases, and a shrink of the cache it went into -------------------------

SPLIT_PROMPT = list(range(3000, 3129))  # four whole blocks and one token
SMALL_POOL = dict(max_tokens=256, max_batch_size=64)


def stale_anywhere(kit, mgr, history):
    """Some windowed layer group has released blocks behind its window at ``history``."""
    for lg, window in enumerate(kit.windows(mgr)):
        if window is not None:
            beg, end = kit.stale_blocks(mgr, lg, history)
            if end > beg:
                return True
    return False


def fetch_segment(kit, mgr, lender, target, start, end, byte, arrived=None):
    """One write lease over ``[start, end)``: every slot staged with ``byte``, the rows of block
    ordinals ``arrived`` (default all) marked, released, the copy into pages run."""
    lease = lender.lend_write(target, start, end)
    view = lease.poll()
    assert view is not None, f"the write lease was not ready: {lease.failure}"
    kit.stage(lender, view, byte)
    masks = tuple(
        np.ones(len(run), dtype=bool) if arrived is None else np.isin(run.ordinals, arrived)
        for run in view.runs
    )
    lease.mark_arrived(masks)
    lease.release()
    mgr._stream.synchronize()


SEGMENTS = {
    # name: (manager kwargs, segments [(start, end, arrived)], usable_until after each segment)
    "two_halves": ({}, [(0, 64, None), (64, 128, None)], [64, 128]),
    "three_segments": ({}, [(0, 32, None), (32, 96, None), (96, 128, None)], [32, 96, 128]),
    "windowed": ({"windows": [WINDOW, 256]}, [(0, 64, None), (64, 128, None)], [64, 128]),
    # A gap stays a gap across segments.
    "first_landed_nothing": ({}, [(0, 64, []), (64, 128, None)], [0, 0]),
    "first_landed_one_block": ({}, [(0, 64, [0]), (64, 128, None)], [32, 32]),
}


def check_segments_add_up(kit, real_manager, attach, case):
    """Consecutive leases into one cache, nothing committed in between: readiness covers every
    segment that landed, and nothing more."""
    manager_kwargs, segments, expected = SEGMENTS[case]
    with real_manager(**manager_kwargs) as mgr:
        target = kit.admitted(mgr, TARGET, SPLIT_PROMPT)
        kv = kit.kv(mgr, target)
        assert kv.num_committed_tokens == 0, "no local prefix: every usable token was fetched"
        lender = attach(mgr, fetch_tokens=64)
        bytes_of = {}
        for i, ((start, end, arrived), usable) in enumerate(zip(segments, expected)):
            byte = 0x11 * (i + 1)
            fetch_segment(kit, mgr, lender, target, start, end, byte, arrived)
            for ordinal in range(start // TPB, end // TPB):
                if arrived is None or ordinal in arrived:
                    bytes_of[ordinal] = byte
            assert kv.num_committed_tokens == 0, "a fetch committed"
            readiness = lender.readiness(target)
            floor = kv.history_length if stale_anywhere(kit, mgr, kv.history_length) else 0
            assert readiness is not None and readiness.restart_floor == floor
            assert readiness.usable_until >= usable, (
                f"after segment {i} [{start}, {end}): readiness {tuple(readiness)}; an earlier "
                "segment's delivered prefix was dropped"
            )
            assert readiness.usable_until == usable, (
                f"after segment {i}: readiness {tuple(readiness)} counts what never landed"
            )
        # The bytes are in the pages: only the bookkeeping decides whether they count.
        dev = kit.DevicePages(mgr)
        for lg, window in enumerate(kit.windows(mgr)):
            if window is not None and window < 128:
                continue
            slots = kit.pages(kv, lg)
            for ordinal, byte in bytes_of.items():
                got = kit.digest(dev.read(lg, slots[ordinal]))
                assert got == kit.digest(bytes([byte]) * dev.page_bytes(lg))


@pytest.mark.parametrize("case", list(SEGMENTS))
def test_consecutive_leases_add_up_to_one_fetch(kit, real_manager, case):
    check_segments_add_up(kit, real_manager, attach, case)


def test_the_check_catches_a_lender_keeping_only_the_latest_segment(kit, real_manager):
    def latest_only(self, request_id, fetch, rows, copied):
        self._delivered.pop(request_id, None)
        real("_deliver")(self, request_id, fetch, rows, copied)

    with pytest.raises(CAUGHT, match="an earlier segment's delivered prefix was dropped"):
        check_segments_add_up(
            kit, real_manager, attach_breaking({"_deliver": latest_only}), "two_halves"
        )


@pytest.mark.parametrize("case", ["first_landed_nothing", "first_landed_one_block"])
def test_the_check_catches_a_lender_counting_rows_that_never_landed(kit, real_manager, case):
    def whole_ranges(self, manager, delivered, committed):  # up to the last row any lease covered
        return max(committed, max(len(b) for b in delivered.blocks) * TPB)

    with pytest.raises(CAUGHT, match="counts what never landed"):
        check_segments_add_up(
            kit, real_manager, attach_breaking({"_usable_until": whole_ranges}), case
        )


def check_an_abandoned_segment_keeps_earlier_deliveries(kit, real_manager, attach):
    with real_manager() as mgr:
        target = kit.admitted(mgr, TARGET, SPLIT_PROMPT)
        lender = attach(mgr, fetch_tokens=64)
        fetch_segment(kit, mgr, lender, target, 0, 64, 0x11)
        assert lender.readiness(target) == (64, 0)
        second = lender.lend_write(target, 64, 128)
        assert lender.readiness(target) is None, "a fetch counts from its call"
        second.release()  # before anyone saw it ready: abandoned, nothing written
        readiness = lender.readiness(target)
        assert readiness == (64, 0), (
            f"readiness {tuple(readiness)} after an abandoned segment; the first segment's "
            "delivered prefix was dropped"
        )


def test_an_abandoned_segment_keeps_what_earlier_ones_delivered(kit, real_manager):
    check_an_abandoned_segment_keeps_earlier_deliveries(kit, real_manager, attach)


def test_the_check_catches_a_lender_dropping_every_delivery_on_abandon(kit, real_manager):
    def abandon_all(self, lease):
        real("_abandon")(self, lease)
        self._delivered.pop(lease._request_id, None)

    with pytest.raises(CAUGHT, match="the first segment's delivered prefix was dropped"):
        check_an_abandoned_segment_keeps_earlier_deliveries(
            kit, real_manager, attach_breaking({"_abandon": abandon_all})
        )


FRESH_FILL = 2.5  # every fp16 element 0x4100: no staged byte pattern reads as it


def test_a_resume_with_the_fresh_page_fill_on_keeps_the_fetched_blocks(
    kit, real_manager, monkeypatch
):
    """The manager's fresh-page fill writes pages as a request first gets them. The pages a fetch
    grew are the request's own once the fetch lands: resuming fills only pages past them."""
    monkeypatch.setenv("TRTLLM_KV_FRESH_PAGE_FILL", str(FRESH_FILL))
    with real_manager() as mgr:
        target = kit.admitted(mgr, TARGET, PROMPT)
        lender = attach(mgr, fetch_tokens=END)
        fetch_segment(kit, mgr, lender, target, 0, 64, 0x5A)
        assert lender.readiness(target).usable_until == 64
        # The executor resumes the request where the fetch left it usable.
        target.context_chunk_size = target.prompt_len - target.context_current_position
        target.set_prepopulated_prompt_len(64, TPB)
        target.context_chunk_size = target.prompt_len - 64
        assert mgr.resize_context(target, target.context_remaining_length)
        mgr._stream.synchronize()
        dev = kit.DevicePages(mgr)
        fetched = kit.digest(bytes([0x5A]) * dev.page_bytes(0))
        filled = np.full(dev.page_bytes(0) // 2, FRESH_FILL, dtype=np.float16).tobytes()
        got = [kit.digest(kit.page(mgr, target, 0, ordinal)) for ordinal in range(3)]
        assert got[2] == kit.digest(filled), "the fill did not run on the resume"
        assert got[:2] == [fetched, fetched], "the resume filled the fetched blocks as fresh pages"
        mgr.free_resources(target)


def check_a_rollback_voids_the_delivery(kit, real_manager, attach, ask_between):
    """The manager's own context rollback undoes the growth ``lend_write`` made: the same cache
    shrinks and frees the fetched pages, which others overwrite before the cache grows back.
    ``ask_between``: readiness is asked between the rollback and the regrow."""
    with real_manager() as mgr_a, real_manager(**SMALL_POOL) as mgr_b:
        source = kit.published(mgr_a, SOURCE, PROMPT)
        lender_a = attach(mgr_a, fetch_tokens=END)
        publish = lender_a.lend_read(source, 0, END)
        publish_view = ready_view(publish, mgr_a)
        source_pages = [kit.digest(kit.page(mgr_a, source, 0, b)) for b in range(BLOCKS)]
        target = kit.admitted(mgr_b, TARGET, PROMPT)
        lender_b = attach(mgr_b, fetch_tokens=END)
        kv = kit.kv(mgr_b, target)
        lease = lender_b.lend_write(target, 0, END)
        assert target.py_ctx_pre_resize_cap == 0, "the rollback would not undo the growth"
        view = lease.poll()
        kit.fill_sentinel(mgr_b, target)
        lease.mark_arrived(kit.relay(lender_a, publish_view, lender_b, view))
        lease.release()
        publish.release()
        mgr_b._stream.synchronize()
        assert [kit.digest(kit.page(mgr_b, target, 0, b)) for b in range(BLOCKS)] == source_pages
        assert lender_b.readiness(target) == (END, 0)

        assert mgr_b.revert_allocate_context(target) is True
        assert kit.kv(mgr_b, target) is kv, "the rollback replaced the cache"
        assert kv.capacity == 0 and kit.pages(kv, 0) == [] and kv.num_committed_tokens == 0
        if ask_between:
            readiness = lender_b.readiness(target)
            assert readiness == (0, 0), f"readiness {tuple(readiness)} counts freed blocks"
        assert kit.overwrite_free_pages(mgr_b) >= BLOCKS
        assert mgr_b.resize_context(target, END)
        regrown = [kit.digest(kit.page(mgr_b, target, 0, b)) for b in range(BLOCKS)]
        assert regrown != source_pages, "the regrown pages still hold the fetched bytes"
        readiness = lender_b.readiness(target)
        assert readiness == (0, 0), (
            f"readiness {tuple(readiness)} counts blocks whose pages now hold {regrown}"
        )
        mgr_b.free_resources(target)


@pytest.mark.parametrize("ask_between", [False, True], ids=["regrown_unasked", "asked_between"])
def test_a_rollback_voids_what_the_fetch_delivered(kit, real_manager, ask_between):
    check_a_rollback_voids_the_delivery(kit, real_manager, attach, ask_between)


def test_the_check_catches_a_lender_ignoring_the_manager_s_shrink(kit, real_manager):
    liar = attach_breaking({"_on_shrink": lambda self, request_id, kv_cache: None})
    with pytest.raises(CAUGHT, match="counts blocks whose pages now hold"):
        check_a_rollback_voids_the_delivery(kit, real_manager, liar, ask_between=False)


def test_the_check_catches_a_lender_never_voiding_delivered_rows(kit, real_manager):
    liar = attach_breaking({"_void_past": lambda self, delivered, kv: None})
    with pytest.raises(CAUGHT, match="counts freed blocks"):
        check_a_rollback_voids_the_delivery(kit, real_manager, liar, ask_between=True)


def check_a_rollback_to_a_local_prefix_keeps_only_the_prefix(kit, real_manager, attach):
    """The rollback's other branch: a local match of two blocks, a fetch of the third; the rollback
    shrinks to the prefix and suspends the cache."""
    local = 2 * TPB
    with real_manager() as mgr_a, real_manager(**SMALL_POOL) as mgr_b:
        source = kit.published(mgr_a, SOURCE, PROMPT)
        lender_a = attach(mgr_a, fetch_tokens=END)
        publish = lender_a.lend_read(source, 0, END)
        publish_view = ready_view(publish, mgr_a)
        kit.published(mgr_b, 7, PROMPT[:local] + list(range(9000, 9033)))
        target = kit.admitted(mgr_b, TARGET, PROMPT)
        lender_b = attach(mgr_b, fetch_tokens=END)
        kv = kit.kv(mgr_b, target)
        assert kv.num_committed_tokens == local
        lease = lender_b.lend_write(target, local, END)
        view = lease.poll()
        kit.fill_sentinel(mgr_b, target, first_block=2)
        lease.mark_arrived(kit.relay(lender_a, publish_view, lender_b, view))
        lease.release()
        publish.release()
        mgr_b._stream.synchronize()
        assert lender_b.readiness(target) == (END, local)
        assert mgr_b.revert_allocate_context(target) is True
        assert kit.kv(mgr_b, target) is kv
        assert (kv.capacity, kv.is_active, kv.num_committed_tokens) == (local, False, local)
        readiness = lender_b.readiness(target)
        assert readiness == (local, local), (
            f"readiness {tuple(readiness)} counts block 2, which the rollback freed"
        )
        mgr_b.free_resources(target)


def test_a_rollback_to_a_local_prefix_keeps_only_the_prefix(kit, real_manager):
    check_a_rollback_to_a_local_prefix_keeps_only_the_prefix(kit, real_manager, attach)


def test_the_check_catches_a_lender_counting_rows_past_the_shrunk_cache(kit, real_manager):
    liar = attach_breaking(
        {
            "_on_shrink": lambda self, request_id, kv_cache: None,
            "_void_past": lambda self, delivered, kv: None,
        }
    )
    with pytest.raises(CAUGHT, match="counts block 2, which the rollback freed"):
        check_a_rollback_to_a_local_prefix_keeps_only_the_prefix(kit, real_manager, liar)


# -- failures at the call ---------------------------------------------------------------------


@pytest.mark.parametrize("state", ["no_cache", "suspended", "shut_down"])
def test_a_lease_failed_at_the_call_changed_nothing(kit, real_manager, state):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        target = kit.admitted(mgr, TARGET, OTHER_PROMPT)
        lender = attach(mgr, fetch_tokens=END, max_fetches=2)
        if state == "no_cache":
            source, target = kit.make_request(8, PROMPT), kit.make_request(9, OTHER_PROMPT)
        elif state == "suspended":
            for request in (source, target):
                mgr.suspend_request(request)
                assert not kit.kv(mgr, request).is_active
        else:
            mgr.shutdown()
        before = [state_of(kit, mgr, lender, r) for r in (source, target)]
        leases = [lender.lend_read(source, 0, END), lender.lend_write(target, 0, END)]
        for lease in leases:
            assert lease.failure is not None, "not failed at the call"
            assert lease.poll() is None
        assert [state_of(kit, mgr, lender, r) for r in (source, target)] == before
        for lease in leases:
            lease.release()
            lease.release()


def test_a_write_the_pool_cannot_grow_fails_at_the_call_and_changed_nothing(kit, real_manager):
    with real_manager(max_tokens=kit.POOL_TOKENS) as mgr:
        target = kit.admitted(mgr, TARGET, PROMPT)
        lender = attach(mgr, fetch_tokens=END)
        others = kit.Requests(mgr)
        others.allocate(kit.pool_pages(mgr))
        before = state_of(kit, mgr, lender, target)
        lease = lender.lend_write(target, 0, END)
        assert lease.failure is not None and lease.poll() is None
        assert state_of(kit, mgr, lender, target) == before
        lease.release()
        others.free()
        lease = lender.lend_write(target, 0, END)  # with pages free again it goes through
        view = lease.poll()
        assert view is not None
        lease.mark_arrived(view.row_masks())
        lease.release()


def test_a_write_left_with_a_block_without_a_page_fails_at_its_first_poll(
    kit, real_manager, monkeypatch
):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _manager

    with real_manager() as mgr:
        target = kit.admitted(mgr, TARGET, PROMPT)
        lender = attach(mgr, fetch_tokens=END)
        kv = kit.kv(mgr, target)

        def without_block_one(read):
            def pages(kv_cache, lg):
                found = np.array(read(kv_cache, lg), dtype=np.int64)
                if kv_cache is kv and len(found) > 1:
                    found[1] = -1
                return found

            return pages

        with monkeypatch.context() as patched:
            for name in ("pages", "locked_pages"):
                patched.setattr(_manager, name, without_block_one(getattr(_manager, name)))
            lease = lender.lend_write(target, 0, END)
            assert lease.failure is None, "the cache grew: nothing fails at the call after that"
            assert lender.readiness(target) is None
            assert lease.poll() is None and lease.failure is not None
        assert lender.readiness(target) == (kv.num_committed_tokens, kv.history_length)
        lease.release()


# -- argument errors --------------------------------------------------------------------------


def test_a_read_refuses_a_bad_range_at_the_call(kit, real_manager):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        lender = attach(mgr, fetch_tokens=4 * TPB)
        for start, end in ((-TPB, TPB), (TPB, 0), (0, 50), (16, 48), (0, 4 * TPB)):
            with pytest.raises(ValueError):
                lender.lend_read(source, start, end)
        lease = lender.lend_read(source, 0, END)
        ready_view(lease, mgr)
        lease.release()


def test_a_write_refuses_a_bad_range_before_changing_anything(kit, real_manager):
    with real_manager() as mgr:
        kit.published(mgr, 7, PROMPT[: 2 * TPB] + list(range(9000, 9033)))
        matched = kit.admitted(mgr, TARGET, PROMPT)  # its first two blocks match locally
        fresh = kit.admitted(mgr, OTHER, OTHER_PROMPT)
        assert kit.kv(mgr, matched).num_committed_tokens == 2 * TPB
        lender = attach(mgr, fetch_tokens=4 * TPB, max_fetches=2)
        before = [state_of(kit, mgr, lender, r) for r in (matched, fresh)]
        bad = [
            (fresh, -TPB, TPB),
            (fresh, TPB, 0),
            (fresh, 0, 50),
            (fresh, 16, 48),
            (matched, TPB, END),  # starts inside the committed blocks
            (fresh, 0, 4 * TPB),  # past the whole blocks of the prompt
        ]
        for request, start, end in bad:
            with pytest.raises(ValueError):
                lender.lend_write(request, start, end)
        assert [state_of(kit, mgr, lender, r) for r in (matched, fresh)] == before
        leases = [lender.lend_write(matched, 2 * TPB, END), lender.lend_write(fresh, 0, END)]
        with pytest.raises(ValueError):
            lender.lend_write(fresh, 0, END)  # one unsettled fetch per cache
        for lease in leases:
            view = lease.poll()
            lease.mark_arrived(view.row_masks())
            lease.release()


def test_a_replaced_cache_takes_a_new_fetch(kit, real_manager):
    with real_manager() as mgr:
        target = kit.admitted(mgr, TARGET, PROMPT)
        lender = attach(mgr, fetch_tokens=END, max_fetches=2)
        first = lender.lend_write(target, 0, END)
        first_view = first.poll()  # unsettled: seen ready, not marked
        mgr.free_resources(target)
        target = kit.admitted(mgr, TARGET, PROMPT)
        second = lender.lend_write(target, 0, END)
        view = second.poll()
        assert view is not None and lender.readiness(target) is None
        second.mark_arrived(view.row_masks())
        first.mark_arrived(first_view.row_masks())
        for lease in (first, second):
            lease.release()


def test_a_windowed_write_may_not_end_below_the_history(kit, real_manager):
    with real_manager(windows=[WINDOW, 256]) as mgr:
        target = kit.admitted(mgr, TARGET, WINDOWED_PROMPT)
        lender = attach(mgr, fetch_tokens=WINDOWED_END)
        lender.lend_write(target, 0, WINDOWED_END).release()  # abandoned; the history stays
        assert kit.kv(mgr, target).history_length == WINDOWED_END
        before = state_of(kit, mgr, lender, target)
        with pytest.raises(ValueError):
            lender.lend_write(target, 0, END)
        assert state_of(kit, mgr, lender, target) == before


def check_a_windowed_target_with_scratch_reuse_fails_at_the_call(kit, real_manager, attach):
    with real_manager(windows=[WINDOW, 256], swa_scratch_reuse=True) as mgr:
        target = kit.make_request(TARGET, WINDOWED_PROMPT)
        assert mgr.prepare_context(target)
        kv = kit.kv(mgr, target)
        assert kv.enable_swa_scratch_reuse, "the target must start with scratch reuse on"
        lender = attach(mgr, fetch_tokens=WINDOWED_END)
        before = state_of(kit, mgr, lender, target)
        try:
            lease = lender.lend_write(target, 0, WINDOWED_END)
        except RuntimeError as error:  # the manager refusing the grow a lender let through
            raise AssertionError(f"not failed at the call: {error}") from error
        assert lease.failure is not None and "scratch" in lease.failure, "not failed at the call"
        assert lease.poll() is None
        assert state_of(kit, mgr, lender, target) == before
        lease.release()
        kv.enable_swa_scratch_reuse = False
        lease = lender.lend_write(target, 0, WINDOWED_END)  # with it off the write goes through
        view = lease.poll()
        assert view is not None
        lease.mark_arrived(view.row_masks())
        lease.release()


def test_a_windowed_target_with_scratch_reuse_fails_at_the_call(kit, real_manager):
    check_a_windowed_target_with_scratch_reuse_fails_at_the_call(kit, real_manager, attach)


def test_the_check_catches_a_lender_ignoring_scratch_reuse(kit, real_manager):
    liar = attach_breaking({"_scratch_reuse_on": lambda self, kv: False})
    with pytest.raises(CAUGHT, match="not failed at the call"):
        check_a_windowed_target_with_scratch_reuse_fails_at_the_call(kit, real_manager, liar)


class _TooBig(Exception):
    pass


def _oversize_fails_the_lease():
    """Rules turning the refusal of a range staging cannot hold into a lease failed at the call."""
    nobody = SimpleNamespace(py_request_id=10**9)

    def check_fits(self, counts):
        try:
            real("_check_fits")(self, counts)
        except ValueError as error:
            raise _TooBig() from error

    def lend(name):
        def call(self, request, start, end):
            try:
                return real(name)(self, request, start, end)
            except _TooBig:
                return real(name)(self, nobody, 0, 0)

        return call

    return {
        "_check_fits": check_fits,
        "lend_read": lend("lend_read"),
        "lend_write": lend("lend_write"),
    }


def check_a_range_staging_cannot_hold_is_refused(kit, real_manager, attach):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        target = kit.admitted(mgr, TARGET, OTHER_PROMPT)
        lender = attach(mgr, fetch_tokens=2 * TPB)  # two slots
        before = state_of(kit, mgr, lender, target)
        refusal(lambda: lender.lend_read(source, 0, END))
        refusal(lambda: lender.lend_write(target, 0, END))
        assert state_of(kit, mgr, lender, target) == before
        # A range of fetch_tokens tokens fits exactly.
        read = lender.lend_read(source, 0, 2 * TPB)
        ready_view(read, mgr)
        read.release()
        write = lender.lend_write(target, 0, 2 * TPB)
        view = write.poll()
        assert view is not None
        write.mark_arrived(view.row_masks())
        write.release()


def test_a_range_longer_than_staging_holds_is_refused(kit, real_manager):
    check_a_range_staging_cannot_hold_is_refused(kit, real_manager, attach)


def test_the_check_catches_a_lender_failing_the_lease_instead(kit, real_manager):
    liar = attach_breaking(_oversize_fails_the_lease())
    with pytest.raises(CAUGHT, match=NOT_REFUSED):
        check_a_range_staging_cannot_hold_is_refused(kit, real_manager, liar)


# -- readiness --------------------------------------------------------------------------------


def test_readiness_without_a_fetch_is_the_cache_s_own_interval(kit, real_manager):
    with real_manager(windows=[WINDOW, 256]) as mgr:
        source = kit.published(mgr, SOURCE, WINDOWED_PROMPT)
        lender = attach(mgr, fetch_tokens=WINDOWED_END)
        kv = kit.kv(mgr, source)
        readiness = lender.readiness(source)
        assert readiness == (max(kv.num_committed_tokens, kv.history_length), kv.history_length)
        assert all(type(v) is int for v in readiness)
        with pytest.raises(ValueError):
            lender.readiness(kit.make_request(9, PROMPT))


def test_readiness_waits_on_nothing_and_settles_by_being_asked(kit, real_manager):
    with real_manager() as mgr:
        target = kit.admitted(mgr, TARGET, PROMPT)
        lender = attach(mgr, fetch_tokens=END)
        lease = lender.lend_write(target, 0, END)
        view = lease.poll()
        with kit.held_stream(mgr._stream) as gate:
            lease.mark_arrived(view.row_masks(True))
            assert lender.readiness(target) is None
            assert lender.readiness(target) is None
            gate.open()
        assert kit.wait_until(lambda: lender.readiness(target) is not None)
        assert lender.readiness(target) == (END, 0)
        lease.release()


# -- shutdown ---------------------------------------------------------------------------------


def test_the_manager_s_shutdown_waits_for_the_lender_s_copies(kit, real_manager):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        lender = attach(mgr, fetch_tokens=END)
        parts = lender.parts
        with kit.gated_stream(mgr._stream, open_after=1.0) as gate:
            lender.lend_read(source, 0, END).release()  # its copy waits behind the gate
            mgr.shutdown()
            assert gate.opened, "shutdown returned while a copy into staging was queued"
        assert not kit.staging_kept(parts), "no lease was open: the memory is freed"


def test_shutdown_fails_waiting_leases_and_later_calls_only_end_records(kit, real_manager):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        target = kit.admitted(mgr, TARGET, OTHER_PROMPT)
        lender = attach(mgr, fetch_tokens=END)
        held = lender.lend_read(source, 0, END)
        ready_view(held, mgr)
        waiting = lender.lend_read(source, 0, END)
        assert waiting.poll() is None
        mgr.shutdown()
        assert waiting.failure is not None and waiting.poll() is None
        for lease in (lender.lend_read(source, 0, END), lender.lend_write(target, 0, END)):
            assert lease.failure is not None and lease.poll() is None
            lease.release()
        with pytest.raises(ValueError):
            lender.readiness(source)
        for lease in (held, held, waiting):
            lease.release()
        mgr.shutdown()  # a second shutdown frees nothing and returns


def check_an_open_lease_keeps_the_staging_memory(kit, real_manager, attach, monkeypatch, failed):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        lender = attach(mgr, fetch_tokens=END)
        parts = lender.parts
        assert kit.staging_memory(parts)
        if failed:
            lease = lender.lend_read(kit.make_request(9, PROMPT), 0, END)
            assert lease.failure is not None
            address, content = None, None
        else:
            lease = lender.lend_read(source, 0, END)
            address = int(ready_view(lease, mgr).runs[0].addresses[0])
            content = kit.host_bytes(address, parts[0].slot_bytes)
        warnings = kit.lender_warnings(monkeypatch)
        mgr.shutdown()
        assert kit.staging_kept(parts), "staging freed at shutdown with a lease open"
        assert warnings, "keeping the memory until exit is logged"
        if address is not None:
            got = kit.digest(kit.host_bytes(address, parts[0].slot_bytes))
            assert got == kit.digest(content), "the kept staging memory changed"
        lease.release()
        assert kit.staging_kept(parts), "a release after the shutdown freed it"


@pytest.mark.parametrize("failed", [False, True], ids=["ready_lease", "failed_lease"])
def test_an_unreleased_lease_keeps_the_staging_memory_until_exit(
    kit, real_manager, monkeypatch, failed
):
    check_an_open_lease_keeps_the_staging_memory(kit, real_manager, attach, monkeypatch, failed)


def test_the_check_catches_a_lender_freeing_memory_a_lease_holds(kit, real_manager, monkeypatch):
    liar = attach_breaking({"_memory_in_use": lambda self: False})
    with pytest.raises(CAUGHT, match="staging freed at shutdown with a lease open"):
        check_an_open_lease_keeps_the_staging_memory(
            kit, real_manager, liar, monkeypatch, failed=False
        )


def test_with_every_lease_released_shutdown_frees_the_staging_memory(
    kit, real_manager, monkeypatch
):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        lender = attach(mgr, fetch_tokens=END)
        parts = lender.parts
        refs = [weakref.ref(m) for m in kit.staging_memory(parts)]
        assert refs
        lease = lender.lend_read(source, 0, END)
        ready_view(lease, mgr)
        lease.release()
        lender.lend_read(kit.make_request(9, PROMPT), 0, END).release()  # failed, released
        warnings = kit.lender_warnings(monkeypatch)
        mgr.shutdown()
        assert not kit.staging_kept(parts) and warnings == []
    gc.collect()
    assert all(ref() is None for ref in refs), "something kept the staging memory alive"


def test_a_shutdown_hook_that_fails_keeps_the_memory_and_raises_nothing(
    kit, real_manager, monkeypatch
):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    def planted(self):
        raise RuntimeError("planted")

    with real_manager() as mgr:
        parts = attach(mgr, fetch_tokens=END).parts
        monkeypatch.setattr(_lender.Staging, "_memory_in_use", planted)
        mgr.shutdown()
        assert kit.staging_kept(parts)


# -- parts holds ------------------------------------------------------------------------------


def released(hold) -> None:
    """``hold.release()``; the check fails if it raises, since release is legal in every state."""
    try:
        hold.release()
    except Exception as error:  # handed to the check as its failure
        raise AssertionError(f"release raised {error!r}") from error


def check_an_open_hold_keeps_the_staging_memory(kit, real_manager, attach, monkeypatch):
    with real_manager() as mgr:
        lender = attach(mgr, fetch_tokens=END)
        parts = lender.parts
        assert kit.staging_memory(parts)
        hold = lender.hold_parts()
        assert isinstance(hold, PartsHold)
        warnings = kit.lender_warnings(monkeypatch)
        mgr.shutdown()
        assert kit.staging_kept(parts), "staging freed at shutdown with a hold open"
        assert warnings, "keeping the memory until exit is logged"
        released(hold)
        mgr.shutdown()
        assert kit.staging_kept(parts), "a release after the shutdown freed it"


def test_an_open_hold_keeps_the_staging_memory_until_exit(kit, real_manager, monkeypatch):
    check_an_open_hold_keeps_the_staging_memory(kit, real_manager, attach, monkeypatch)


def test_the_check_catches_a_lender_ignoring_holds(kit, real_manager, monkeypatch):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    liar = attach_breaking({"hold_parts": lambda self: _lender._PartsHold(None)})
    with pytest.raises(CAUGHT, match="staging freed at shutdown with a hold open"):
        check_an_open_hold_keeps_the_staging_memory(kit, real_manager, liar, monkeypatch)


def check_a_hold_released_before_shutdown_lets_the_memory_go(
    kit, real_manager, attach, monkeypatch
):
    with real_manager() as mgr:
        lender = attach(mgr, fetch_tokens=END)
        parts = lender.parts
        refs = [weakref.ref(m) for m in kit.staging_memory(parts)]
        assert refs
        lender.hold_parts().release()
        warnings = kit.lender_warnings(monkeypatch)
        mgr.shutdown()
        assert not kit.staging_kept(parts), "a released hold kept the memory"
        assert warnings == []
    gc.collect()
    assert all(ref() is None for ref in refs), "something kept the staging memory alive"


def test_a_hold_released_before_shutdown_lets_the_memory_go(kit, real_manager, monkeypatch):
    check_a_hold_released_before_shutdown_lets_the_memory_go(kit, real_manager, attach, monkeypatch)


def test_the_check_catches_a_lender_never_letting_a_hold_go(kit, real_manager, monkeypatch):
    liar = attach_breaking({"_end_hold": lambda self, hold: None})
    with pytest.raises(CAUGHT, match="a released hold kept the memory"):
        check_a_hold_released_before_shutdown_lets_the_memory_go(
            kit, real_manager, liar, monkeypatch
        )


def check_a_dropped_hold_keeps_the_staging_memory(kit, real_manager, attach):
    with real_manager() as mgr:
        lender = attach(mgr, fetch_tokens=END)
        parts = lender.parts
        assert kit.staging_memory(parts)
        lender.hold_parts()  # dropped unreleased
        gc.collect()
        mgr.shutdown()
        assert kit.staging_kept(parts), "a dropped hold let the memory go"


def test_a_hold_dropped_unreleased_still_keeps_the_staging_memory(kit, real_manager):
    check_a_dropped_hold_keeps_the_staging_memory(kit, real_manager, attach)


def test_the_check_catches_a_lender_holding_holds_weakly(kit, real_manager):
    def weakly(self, *args):
        real("__init__")(self, *args)
        self._holds = weakref.WeakSet()

    with pytest.raises(CAUGHT, match="a dropped hold let the memory go"):
        check_a_dropped_hold_keeps_the_staging_memory(
            kit, real_manager, attach_breaking({"__init__": weakly})
        )


def check_a_hold_releases_once_in_every_state(kit, real_manager, attach):
    with real_manager() as mgr:
        lender = attach(mgr, fetch_tokens=END)
        parts = lender.parts
        assert kit.staging_memory(parts)
        first, second = lender.hold_parts(), lender.hold_parts()
        released(first)
        released(first)  # does nothing: the second hold still holds
        mgr.shutdown()
        assert kit.staging_kept(parts), "a second release let the other hold go"
        late = lender.hold_parts()  # after the shutdown: inert
        assert isinstance(late, PartsHold)
        for hold in (late, late):
            released(hold)
    gone = weakref.ref(lender)
    del mgr, lender
    gc.collect()
    assert gone() is None, "the lender outlived its manager"
    for hold in (second, second):  # its lender is gone
        released(hold)


def test_a_hold_releases_once_in_every_state(kit, real_manager):
    check_a_hold_releases_once_in_every_state(kit, real_manager, attach)


def test_the_check_catches_a_hold_releasing_again(kit, real_manager, monkeypatch):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    def unguarded(self):
        lender = self._lender() if self._lender is not None else None
        if lender is not None:
            lender._end_hold(self)

    monkeypatch.setattr(_lender._PartsHold, "release", unguarded)
    with pytest.raises(CAUGHT, match="release raised"):
        check_a_hold_releases_once_in_every_state(kit, real_manager, attach)


class _CountedHold:
    """A hold whose every release reaches its lender, which counts holds with an integer."""

    def __init__(self, lender):
        self._lender = weakref.ref(lender)

    def release(self):
        lender = self._lender()
        if lender is not None:
            lender._end_hold(self)


def _counting_holds(self, *args):
    real("__init__")(self, *args)
    self._count = 0


def _take_counted(self):
    self._count += 1
    return _CountedHold(self)


def _drop_counted(self, hold):
    self._count -= 1


def _in_use_counted(self):
    return bool(self._unreleased) or self._count > 0 or bool(self._quarantined)


def test_the_check_catches_a_lender_counting_holds_with_an_integer(kit, real_manager):
    liar = attach_breaking(
        {
            "__init__": _counting_holds,
            "hold_parts": _take_counted,
            "_end_hold": _drop_counted,
            "_memory_in_use": _in_use_counted,
        }
    )
    with pytest.raises(CAUGHT, match="a second release let the other hold go"):
        check_a_hold_releases_once_in_every_state(kit, real_manager, liar)


# What is released before the manager's shutdown, in order; the rest is released after it.
ORDERS = {
    "neither": (),
    "lease": ("lease",),
    "hold": ("hold",),
    "lease_then_hold": ("lease", "hold"),
    "hold_then_lease": ("hold", "lease"),
}


def check_a_lease_and_a_hold_keep_the_memory_until_both_go(
    kit, real_manager, attach, monkeypatch, order
):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        lender = attach(mgr, fetch_tokens=END)
        parts = lender.parts
        assert kit.staging_memory(parts)
        hold = lender.hold_parts()
        lease = lender.lend_read(source, 0, END)
        ready_view(lease, mgr)
        ends = {"lease": lease.release, "hold": lambda: released(hold)}
        for what in order:
            ends[what]()
        warnings = kit.lender_warnings(monkeypatch)
        mgr.shutdown()
        both = len(order) == 2
        if both:
            assert not kit.staging_kept(parts) and warnings == [], "kept with both released"
        else:
            assert kit.staging_kept(parts), f"freed at shutdown after releasing {order}"
            assert warnings, "keeping the memory until exit is logged"
        for what in [w for w in ("lease", "hold") if w not in order]:
            ends[what]()
        assert kit.staging_kept(parts) == (not both), "a release after the shutdown changed it"


@pytest.mark.parametrize("order", list(ORDERS))
def test_a_lease_and_a_hold_keep_the_memory_until_both_go(kit, real_manager, monkeypatch, order):
    check_a_lease_and_a_hold_keep_the_memory_until_both_go(
        kit, real_manager, attach, monkeypatch, ORDERS[order]
    )


def _in_every_order(check, *args):
    for order in ORDERS.values():
        check(*args, order)


def test_the_check_catches_a_hold_release_that_forgets_open_leases(kit, real_manager, monkeypatch):
    def end_hold(self, hold):
        real("_end_hold")(self, hold)
        self._unreleased.clear()

    liar = attach_breaking({"_end_hold": end_hold})
    with pytest.raises(CAUGHT, match="freed at shutdown"):
        _in_every_order(
            check_a_lease_and_a_hold_keep_the_memory_until_both_go,
            kit,
            real_manager,
            liar,
            monkeypatch,
        )


def test_the_check_catches_a_lease_release_that_forgets_open_holds(kit, real_manager, monkeypatch):
    def on_release(self, lease):
        real("_on_release")(self, lease)
        self._holds.clear()

    def end_hold(self, hold):  # a hold the lease's release already ended
        self._holds.discard(hold)

    liar = attach_breaking({"_on_release": on_release, "_end_hold": end_hold})
    with pytest.raises(CAUGHT, match="freed at shutdown"):
        _in_every_order(
            check_a_lease_and_a_hold_keep_the_memory_until_both_go,
            kit,
            real_manager,
            liar,
            monkeypatch,
        )


def check_a_failed_copy_and_a_hold_keep_the_memory_in_every_order(
    kit, real_manager, attach, monkeypatch, order
):
    def no_event(self, stream=None):
        raise RuntimeError("planted")

    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        lender = attach(mgr, fetch_tokens=END)
        parts = lender.parts
        assert kit.staging_memory(parts)
        hold = lender.hold_parts()
        with monkeypatch.context() as patched:
            patched.setattr(torch.cuda.Event, "record", no_event)
            lease = lender.lend_read(source, 0, END)  # its copy has no event: slots lost
        assert lease.failure is not None
        mgr._stream.synchronize()
        ends = {"lease": lease.release, "hold": lambda: released(hold)}
        for what in order:
            ends[what]()
        warnings = kit.lender_warnings(monkeypatch)
        mgr.shutdown()
        assert kit.staging_kept(parts), f"freed at shutdown after releasing {order}"
        assert warnings, "keeping the memory until exit is logged"
        for what in [w for w in ("lease", "hold") if w not in order]:
            ends[what]()
        assert kit.staging_kept(parts), "a release after the shutdown freed it"


@pytest.mark.parametrize("order", list(ORDERS))
def test_a_failed_copy_and_a_hold_keep_the_memory_in_every_order(
    kit, real_manager, monkeypatch, order
):
    check_a_failed_copy_and_a_hold_keep_the_memory_in_every_order(
        kit, real_manager, attach, monkeypatch, ORDERS[order]
    )


def test_the_check_catches_a_hold_release_that_forgets_lost_slots(kit, real_manager, monkeypatch):
    def end_hold(self, hold):
        real("_end_hold")(self, hold)
        self._quarantined.clear()

    liar = attach_breaking({"_end_hold": end_hold})
    with pytest.raises(CAUGHT, match="freed at shutdown"):
        _in_every_order(
            check_a_failed_copy_and_a_hold_keep_the_memory_in_every_order,
            kit,
            real_manager,
            liar,
            monkeypatch,
        )


def test_staging_outlives_a_manager_dropped_without_shutdown(kit):
    torch.cuda.init()
    gc.collect()
    mgr = kit.make_manager()
    source = kit.published(mgr, SOURCE, PROMPT)
    lender = attach(mgr, fetch_tokens=END)
    parts = lender.parts
    lease = lender.lend_read(source, 0, END)
    address = int(ready_view(lease, mgr).runs[0].addresses[0])
    content = kit.host_bytes(address, parts[0].slot_bytes)
    mgr.free_resources(source)
    mgr._stream.synchronize()
    watched = weakref.ref(mgr)
    del mgr, lender
    gc.collect()
    torch.cuda.empty_cache()
    assert watched() is None, "an open lease kept the lender or the manager alive"
    assert kit.staging_kept(parts)
    got = kit.digest(kit.host_bytes(address, parts[0].slot_bytes))
    assert got == kit.digest(content), "the kept staging memory changed"
    lease.release()
    lease.release()


# -- progress ---------------------------------------------------------------------------------


class EventQueries:
    """Counts ``torch.cuda.Event.query`` per event within each lender call made through ``call``."""

    def __init__(self, monkeypatch):
        self.asked: List[int] = []
        self.last: Set[int] = set()  # the events the latest call asked
        query = torch.cuda.Event.query

        def counted(event):
            self.asked.append(id(event))
            return query(event)

        monkeypatch.setattr(torch.cuda.Event, "query", counted)

    def call(self, method, *args):
        first = len(self.asked)
        result = method(*args)
        asked = self.asked[first:]
        assert len(asked) == len(set(asked)), "a copy's event was asked twice in one lender call"
        self.last = set(asked)
        return result


HELD_READS = 4


def check_each_call_asks_each_copy_once(kit, real_manager, attach, monkeypatch):
    """Each lender call asks each copy's event at most once, a lease still lent costs no query
    beyond its own poll's, and a copy reported complete is never asked again: a call asks only its
    own copy and the pending copies of released leases, not one per held lease."""
    with real_manager() as mgr:
        sources = [
            kit.published(mgr, 10 + i, [10000 * (i + 1) + t for t in range(END + 1)])
            for i in range(HELD_READS)
        ]
        target = kit.admitted(mgr, TARGET, PROMPT)
        assert mgr._resize_for_connector_prefix(target, kit.kv(mgr, target), 0, END)
        lender = attach(mgr, fetch_tokens=END, max_fetches=HELD_READS + 1)
        events = EventQueries(monkeypatch)
        with kit.held_stream(mgr._stream) as gate:
            reads = [events.call(lender.lend_read, source, 0, END) for source in sources]
            write = events.call(lender.lend_write, target, 0, END)
            view = events.call(write.poll)
            events.call(write.mark_arrived, view.row_masks(True))
            events.call(write.release)
            released = id(write._copy._event)
            for _ in range(2):
                for lease in reads:
                    assert events.call(lease.poll) is None
                    own = id(lease._copy._event)
                    assert events.last == {own, released}, (
                        f"a read's poll asked {len(events.last)} copies, not its own and the "
                        "released one"
                    )
                assert events.call(lender.readiness, target) is None
                assert events.last == {released}, (
                    f"readiness asked {len(events.last)} copies, not the released one"
                )
            gate.open()
        mgr._stream.synchronize()
        assert events.call(lender.readiness, target) is not None
        for lease in reads:
            assert events.call(lease.poll) is not None
        first = len(events.asked)
        for _ in range(3):
            for lease in reads:
                assert events.call(lease.poll) is not None
            assert events.call(lender.readiness, target) is not None
        assert len(events.asked) == first, "a copy reported complete was asked again"
        for lease in reads:
            events.call(lease.release)
        assert len(events.asked) == first, "a copy reported complete was asked again"


def test_each_call_asks_each_copy_s_event_at_most_once(kit, real_manager, monkeypatch):
    check_each_call_asks_each_copy_once(kit, real_manager, attach, monkeypatch)


def test_the_check_catches_a_lender_asking_every_held_lease_on_every_call(
    kit, real_manager, monkeypatch
):
    def asks_first(self, lease):
        if lease._copy is not None:
            lease._copy._event.query()
        return real("_recyclable")(self, lease)

    with pytest.raises(CAUGHT, match="asked twice in one lender call"):
        check_each_call_asks_each_copy_once(
            kit, real_manager, attach_breaking({"_recyclable": asks_first}), monkeypatch
        )


def test_the_check_catches_a_lender_asking_every_held_lease_s_copy_once_per_call(
    kit, real_manager, monkeypatch
):
    def lent_first(self, lease):  # asks through the per-round cache, then applies the rule
        if not self._landed(lease._copy):
            return False
        return real("_recyclable")(self, lease)

    with pytest.raises(CAUGHT, match="a read's poll asked 5 copies"):
        check_each_call_asks_each_copy_once(
            kit, real_manager, attach_breaking({"_recyclable": lent_first}), monkeypatch
        )


def test_the_check_catches_a_lender_asking_a_completed_copy_again(kit, real_manager, monkeypatch):
    def forgets_completion(self, copy):  # asks once per round, whatever it learnt before
        if copy is not None and copy._round != self._round:
            copy._round = self._round
            copy._done = bool(copy._event.query())
        return copy is None or copy._done

    with pytest.raises(CAUGHT, match="a copy reported complete was asked again"):
        check_each_call_asks_each_copy_once(
            kit, real_manager, attach_breaking({"_landed": forgets_completion}), monkeypatch
        )


def check_progress_needs_no_new_traffic(kit, real_manager, attach):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        lender = attach(mgr, fetch_tokens=END)
        holder = lender.lend_read(source, 0, END)
        ready_view(holder, mgr)
        waiting = lender.lend_read(source, 0, END)
        assert waiting.poll() is None
        holder.release()
        ready_view(waiting, mgr)  # its own polls alone grant it and see its copy land
        waiting.release()


def test_progress_needs_no_new_traffic(kit, real_manager):
    check_progress_needs_no_new_traffic(kit, real_manager, attach)


def test_the_check_catches_a_lender_progressing_only_when_lending(kit, real_manager):
    lending = [0]

    def lend(name):
        def call(self, *args):
            lending[0] += 1
            try:
                return real(name)(self, *args)
            finally:
                lending[0] -= 1

        return call

    def progress(self):
        if lending[0]:
            real("_progress")(self)

    liar = attach_breaking(
        {"_progress": progress, "lend_read": lend("lend_read"), "lend_write": lend("lend_write")}
    )
    with pytest.raises(CAUGHT, match="the lease never became ready"):
        check_progress_needs_no_new_traffic(kit, real_manager, liar)


# -- a copy that fails partway ----------------------------------------------------------------


class FailingDriver:
    """The CUDA driver, except that its ``fail_on``-th memcpy call, of any flavour, fails."""

    def __init__(self, real_driver, fail_on):
        self._real = real_driver
        self._fail_on = fail_on
        self.calls = 0

    def __getattr__(self, name):
        found = getattr(self._real, name)
        if not name.startswith("cuMemcpy"):
            return found

        def call(*args, **kwargs):
            self.calls += 1
            if self.calls == self._fail_on:
                return (self._real.CUresult.CUDA_ERROR_INVALID_VALUE,)
            return found(*args, **kwargs)

        return call


def test_a_publish_whose_copy_fails_partway_fails_and_frees_its_slots_after_the_rest(
    kit, real_manager, monkeypatch
):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    # Three layers: two pool groups, so the copy is several memcpys.
    with real_manager(windows=[WINDOW, 256], num_layers=3) as mgr:
        source = kit.published(mgr, SOURCE, WINDOWED_PROMPT)
        lender = attach(mgr, fetch_tokens=WINDOWED_END)
        groups = kit.pool_group_ids(mgr)
        driver = FailingDriver(_lender.drv, fail_on=2)
        monkeypatch.setattr(_lender, "drv", driver)
        with kit.held_stream(mgr._stream) as gate:
            lease = lender.lend_read(source, 0, WINDOWED_END)  # every slot
            assert driver.calls >= 2, "a single memcpy: nothing failed partway"
            assert lease.failure is not None and lease.poll() is None
            lease.release()
            assert [lender._free_slots(g) for g in groups] == [0] * len(groups), (
                "slots came back while a copy into them was queued"
            )
            gate.open()
        lender.readiness(source)  # any call makes progress
        assert [lender._free_slots(g) for g in groups] == [p.slots for p in lender.parts]


def test_marks_whose_copy_fails_partway_abandon_the_fetch_and_keep_the_lease_ready(
    kit, real_manager, monkeypatch
):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    with real_manager(windows=[WINDOW, 256], num_layers=3) as mgr:
        target = kit.admitted(mgr, TARGET, WINDOWED_PROMPT)
        lender = attach(mgr, fetch_tokens=WINDOWED_END)
        groups = kit.pool_group_ids(mgr)
        lease = lender.lend_write(target, 0, WINDOWED_END)
        view = lease.poll()
        kit.stage(lender, view, 0x5A)
        driver = FailingDriver(_lender.drv, fail_on=2)
        monkeypatch.setattr(_lender, "drv", driver)
        kv = kit.kv(mgr, target)
        with kit.held_stream(mgr._stream) as gate:
            lease.mark_arrived(view.row_masks(True))  # raises nothing
            assert driver.calls >= 2, "a single memcpy: nothing failed partway"
            assert lender.readiness(target) == (kv.num_committed_tokens, kv.history_length)
            assert lease.poll() is view, "the lease stays ready"
            lease.release()
            assert [lender._free_slots(g) for g in groups] == [0] * len(groups)
            gate.open()
        lender.readiness(target)
        assert [lender._free_slots(g) for g in groups] == [p.slots for p in lender.parts]


def test_a_copy_without_an_event_loses_its_slots_and_keeps_the_memory(
    kit, real_manager, monkeypatch
):
    def no_event(self, stream=None):
        raise RuntimeError("planted")

    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        lender = attach(mgr, fetch_tokens=END)
        parts = lender.parts
        (group,) = kit.pool_group_ids(mgr)
        with monkeypatch.context() as patched:
            patched.setattr(torch.cuda.Event, "record", no_event)
            lease = lender.lend_read(source, 0, END)
        assert lease.failure is not None
        mgr._stream.synchronize()
        lease.release()
        lender.readiness(source)
        assert lender._free_slots(group) == 0, "slots a copy may still touch were reused"
        warnings = kit.lender_warnings(monkeypatch)
        mgr.shutdown()
        assert kit.staging_kept(parts) and warnings


# -- errors inside the lender -----------------------------------------------------------------


def test_marks_that_raise_leave_the_fetch_abandoned_and_the_slots_returnable(
    kit, real_manager, monkeypatch
):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    def planted(*args):
        raise RuntimeError("planted")

    with real_manager() as mgr:
        target = kit.admitted(mgr, TARGET, PROMPT)
        other = kit.admitted(mgr, OTHER, OTHER_PROMPT)
        lender = attach(mgr, fetch_tokens=END)
        (group,) = kit.pool_group_ids(mgr)
        lease = lender.lend_write(target, 0, END)
        view = lease.poll()
        kv = kit.kv(mgr, target)
        with monkeypatch.context() as patched:
            patched.setattr(_lender.Staging, "_still_lent", planted)
            with pytest.raises(RuntimeError, match="planted"):
                lease.mark_arrived(view.row_masks(True))
        assert lender.readiness(target) == (kv.num_committed_tokens, kv.history_length)
        lease.release()
        waiting = lender.lend_write(other, 0, END)
        waiting_view = waiting.poll()
        assert waiting_view is not None, "the lease's slots never came back"
        opened = lender._open_count()
        with monkeypatch.context() as patched:
            patched.setattr(_lender.Staging, "_progress", planted)
            try:
                waiting.release()
            except RuntimeError:
                pass  # the planted error may surface; the release must count all the same
        assert lender._open_count() == opened - 1
        waiting.mark_arrived(waiting_view.row_masks())
        lender.readiness(other)
        assert lender._free_slots(group) == lender.parts[0].slots


def test_a_grant_that_raises_fails_only_its_own_lease(kit, real_manager, monkeypatch):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _manager

    with real_manager() as mgr:
        holder = kit.published(mgr, SOURCE, PROMPT)
        broken = kit.published(mgr, OTHER, OTHER_PROMPT)
        fetched = kit.admitted(mgr, TARGET, list(range(8000, 8097)))
        lender = attach(mgr, fetch_tokens=END)
        write = lender.lend_write(fetched, 0, END)
        write.mark_arrived(write.poll().row_masks())
        write.release()
        head = lender.lend_read(holder, 0, END)
        ready_view(head, mgr)
        waiting = lender.lend_read(broken, 0, 2 * TPB)
        assert waiting.poll() is None
        broken_kv = kit.kv(mgr, broken)

        real_kv_of = _manager.kv_of

        def kv_of(manager, request_id):
            if request_id == OTHER:
                raise RuntimeError("planted")
            return real_kv_of(manager, request_id)

        def raising_for_broken(read):
            def call(kv_cache, *rest):
                if kv_cache is broken_kv:
                    raise RuntimeError("planted")
                return read(kv_cache, *rest)

            return call

        with monkeypatch.context() as patched:
            patched.setattr(_manager, "kv_of", kv_of)
            for name in ("cache_state", "pages", "locked_pages"):
                patched.setattr(_manager, name, raising_for_broken(getattr(_manager, name)))
            head.release()  # grants the one waiting, whose grant raises
            assert waiting.poll() is None and waiting.failure is not None
            assert isinstance(lender.readiness(fetched), Readiness)
        waiting.release()
        again = lender.lend_read(holder, 0, END)
        ready_view(again, mgr)
        again.release()


# -- names ------------------------------------------------------------------------------------


def _tp2(rank):
    from tensorrt_llm.mapping import Mapping

    return Mapping(world_size=2, tp_size=2, rank=rank)


def names_by_rank(kit, real_manager, attach, num_kv_heads, scopes=(SCOPE, SCOPE)):
    """Each rank of a TP2 pair publishes ``PROMPT``: its names, and its part names."""
    names, part_names = [], []
    for rank, scope in enumerate(scopes):
        with real_manager(num_kv_heads=num_kv_heads, mapping=_tp2(rank)) as mgr:
            source = kit.published(mgr, SOURCE, PROMPT)
            lender = attach(mgr, fetch_tokens=END, scope=scope)
            lease = lender.lend_read(source, 0, END)
            (run,) = ready_view(lease, mgr).runs
            names.append(np.array(run.names))
            part_names.append([p.name for p in lender.parts])
            lease.release()
    return names, part_names


def check_a_replicated_group_is_named_alike_on_every_rank(kit, real_manager, attach):
    (rank0, rank1), (parts0, parts1) = names_by_rank(kit, real_manager, attach, num_kv_heads=1)
    assert rank0.shape == (BLOCKS, 54) and rank0.dtype == np.uint8
    assert np.array_equal(rank0, rank1), "ranks holding the same bytes must share their names"
    assert parts0 == parts1


def test_a_replicated_group_is_named_alike_on_every_rank(kit, real_manager):
    check_a_replicated_group_is_named_alike_on_every_rank(kit, real_manager, attach)


def test_the_check_catches_a_lender_naming_each_rank_apart(kit, real_manager):
    def per_rank(mgr):
        shard = np.frombuffer(
            mgr.mapping.tp_size.to_bytes(2, "big") + mgr.mapping.tp_rank.to_bytes(2, "big"),
            np.uint8,
        )

        def names(self, layer_group, keys):
            out = np.array(real("_names")(self, layer_group, keys))
            out[:, 50:54] = shard
            return out

        return {"_names": names}

    with pytest.raises(CAUGHT, match="ranks holding the same bytes must share their names"):
        check_a_replicated_group_is_named_alike_on_every_rank(
            kit, real_manager, attach_breaking(per_rank)
        )


def test_a_head_sharded_group_names_each_rank_s_share(kit, real_manager):
    (rank0, rank1), (parts0, parts1) = names_by_rank(kit, real_manager, attach, num_kv_heads=4)
    assert np.array_equal(rank0[:, :50], rank1[:, :50]), "same scope, layout and blocks"
    for rank, names in enumerate((rank0, rank1)):
        shard = (2).to_bytes(2, "big") + rank.to_bytes(2, "big")
        assert {bytes(n[50:54]) for n in names} == {shard}
    assert parts0 == parts1


def test_a_different_scope_shares_no_name(kit, real_manager):
    names = []
    for scope in (b"model-a", b"model-b"):
        with real_manager() as mgr:
            source = kit.published(mgr, SOURCE, PROMPT)
            lender = attach(mgr, fetch_tokens=END, scope=scope)
            lease = lender.lend_read(source, 0, END)
            names.append(([bytes(n) for n in ready_view(lease, mgr).runs[0].names], lender.parts))
            lease.release()
    (first, first_parts), (second, second_parts) = names
    assert [n[16:] for n in first] == [n[16:] for n in second], "same blocks, same keys"
    assert not set(first) & set(second)
    assert [p.name for p in first_parts] == [p.name for p in second_parts], "laid out alike"


def test_part_names_follow_the_layout_alone(real_manager):
    names = []
    for tokens_per_block, scope in ((TPB, b"model-a"), (TPB, b"model-b"), (2 * TPB, b"model-a")):
        with real_manager(tokens_per_block=tokens_per_block) as mgr:
            lender = attach(mgr, fetch_tokens=END, scope=scope)
            names.append([p.name for p in lender.parts])
    assert names[0] == names[1], "instances laid out alike name their parts alike"
    assert names[0] != names[2], "another layout, other part names"


# -- threads ----------------------------------------------------------------------------------


def test_threads_take_turns_and_the_lender_starts_none(kit, real_manager):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        target = kit.admitted(mgr, TARGET, OTHER_PROMPT)
        before = set(threading.enumerate())

        def in_turn(call, name):
            """``call`` on its own thread, joined; like the executor's, it used CUDA before."""

            def run():
                torch.cuda.synchronize()
                return call()

            return kit.on_thread(run, name)

        built = in_turn(lambda: attach(mgr, fetch_tokens=END, max_fetches=2), "builder")
        lender = built["value"]

        def executor_loop():
            read = lender.lend_read(source, 0, END)
            view = ready_view(read, mgr)
            # A backend reads the arrays on its own thread until the release.
            seen = kit.on_thread(lambda: [bytes(n) for n in view.runs[0].names], "backend")
            write = lender.lend_write(target, 0, END)
            write.mark_arrived(write.poll().row_masks(True))
            read.release()
            write.release()
            return seen["value"]

        looped = in_turn(executor_loop, "executor-loop")
        assert "error" not in looped and len(looped["value"]) == BLOCKS
        assert set(threading.enumerate()) <= before, "the lender started a thread"
        assert "error" not in in_turn(mgr.shutdown, "shutdown")
        with pytest.raises(ValueError):
            lender.readiness(source)


# -- the manager's page-index buffer under a cache that outlives the manager --------------------

FREED = "touched the page-index buffer freed with its manager"


def check_a_staging_lease_outliving_its_manager(kit, attach):
    """A read lease, which holds its request's cache, outlives its manager and lender, collected
    without a shutdown or a free; a new tensor reclaims the freed buffer's address with a canary.
    Dropping the lease closes the cache, which must write nothing there."""
    torch.cuda.init()
    gc.collect()
    torch.cuda.empty_cache()
    mgr = kit.make_manager()
    source = kit.published(mgr, SOURCE, PROMPT)
    kv = kit.kv(mgr, source)
    blocks = int(kv.num_blocks)
    lender = attach(mgr, fetch_tokens=END)
    lease = lender.lend_read(source, 0, END)
    ready_view(lease, mgr)
    row = kit.index_row(mgr, source)
    assert row.values[:blocks] == list(kv.get_base_page_indices(0)[:blocks])
    watched = [weakref.ref(o) for o in (mgr.host_kv_cache_block_offsets, mgr, lender)]
    mgr._stream.synchronize()
    del mgr, lender
    gc.collect()
    assert watched[1]() is None and watched[2]() is None, "the manager or lender was not collected"
    canary = None if watched[0]() is not None else kit.reclaim(row)
    if watched[0]() is None:
        assert canary is not None, "inconclusive: no allocation reused the freed buffer's address"
    read = list(kv.get_base_page_indices(0)[:blocks])
    del kv
    lease.release()
    del lease  # it held the cache's last reference
    gc.collect()
    written = kit.canary_written(canary, row)
    assert not (canary is not None and read == [kit.CANARY] * blocks) and not written, (
        f"the leased cache {FREED}: it read {read} and its close wrote {written} (cell, value)"
    )


def test_a_lease_outliving_its_manager_touches_no_freed_index_buffer(kit):
    check_a_staging_lease_outliving_its_manager(kit, attach)


def test_the_check_catches_a_lender_not_keeping_the_index_buffer(kit):
    liar = attach_breaking({"_keep_index_buffer": lambda self, manager: None})
    with pytest.raises(CAUGHT, match=FREED):
        check_a_staging_lease_outliving_its_manager(kit, liar)


def check_the_index_buffer_is_kept_until_the_shutdown(kit, real_manager, attach):
    with real_manager() as mgr:
        before = len(kit.retained())
        buffer = mgr.host_kv_cache_block_offsets
        attach(mgr, fetch_tokens=END)
        assert any(o is buffer for o in kit.retained()), "not kept from the attach"
        mgr.shutdown()
        assert not any(o is buffer for o in kit.retained()), "kept past the shutdown"
        assert len(kit.retained()) == before


def test_the_index_buffer_is_kept_from_the_attach_until_the_shutdown(kit, real_manager):
    check_the_index_buffer_is_kept_until_the_shutdown(kit, real_manager, attach)


def test_the_check_catches_a_lender_keeping_the_index_buffer_past_the_shutdown(kit, real_manager):
    liar = attach_breaking({"_let_go_index_buffer": lambda self: None})
    with pytest.raises(CAUGHT, match="kept past the shutdown"):
        check_the_index_buffer_is_kept_until_the_shutdown(kit, real_manager, liar)


# -- DeepSeek-V4 ------------------------------------------------------------------------------


@skip_pre_blackwell
def test_a_deepseek_v4_prompt_is_fetched_through_staging_byte_for_byte(kit, deepseek_v4_manager):
    tpb = 128
    end = 3 * tpb
    prompt = list(range(3000, 3000 + end + 1))
    with deepseek_v4_manager() as mgr_a, deepseek_v4_manager() as mgr_b:
        source = kit.published(mgr_a, SOURCE, prompt)
        lender_a = attach(mgr_a, fetch_tokens=end)
        publish = lender_a.lend_read(source, 0, end)
        publish_view = ready_view(publish, mgr_a)
        assert len(publish_view.runs) == kit.num_layer_groups(mgr_a)
        assert check_staged(kit, mgr_a, lender_a, source, publish_view) == publish_view.num_rows
        target = kit.admitted(mgr_b, TARGET, prompt)
        lender_b = attach(mgr_b, fetch_tokens=end)
        assert [p.name for p in lender_a.parts] == [p.name for p in lender_b.parts]
        lease = lender_b.lend_write(target, 0, end)
        view = lease.poll()
        assert by_group(view) == by_group(publish_view)
        kit.fill_sentinel(mgr_b, target)
        lease.mark_arrived(kit.relay(lender_a, publish_view, lender_b, view))
        mgr_b._stream.synchronize()
        assert lender_b.readiness(target) == (end, end)
        for run in view.runs:
            for ordinal in run.ordinals.tolist():
                lg = run.layer_group
                got = kit.digest(kit.page(mgr_b, target, lg, ordinal))
                assert got == kit.digest(kit.page(mgr_a, source, lg, ordinal))
        lease.release()
        publish.release()
