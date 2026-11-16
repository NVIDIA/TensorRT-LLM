# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fast slot lookups for the slot-divergence fix (TRTLLM_MOONCAKE_STORE_PGFIX_FAST, lanes/mc/PGFIX_FAST.md).

`_localize` (rank 0's slots -> this rank's) and the commit-swap re-check must produce exactly the same transfers
whether they build each request's whole unmasked page table (switch off) or look up just the listed block ordinals
in the page index row the KV cache manager publishes for locked pages (switch on), with the full query answering any
ordinal the row does not publish. These tests run both on the same fake caches, built so that ranks diverge (rank 0's
slots differ from this rank's) or not, pages are attached without a lock (window-passed blocks, suspended caches),
ordinals run past a request's blocks, groups are out of range and requests have no cache here.

CPU only: the store is the in-process fake from test_mooncake_store_connector; the KV cache manager methods under
test run on a stand-in manager whose caches answer the two C++ accessors they use.
"""

import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
from test_mooncake_store_connector import (  # noqa: F401 (fixtures)
    fake_store,
    make_layout,
    make_worker,
    store_config,
)

from tensorrt_llm._torch.pyexecutor.connectors import kv_cache_connector as kvc
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import worker as worker_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.metadata import (
    MooncakeStoreMetadata,
    PageTransfer,
    PrefixLoadTransfers,
    RequestTransfers,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

pytestmark = pytest.mark.cpu_only

BAD = -1
NUM_GROUPS = 2


class FakeCache:
    """The two page-index accessors of a C++ KvCache, over an explicit attached/locked state.

    `attached[g][o]` is the slot of the page attached at block ordinal `o` (BAD when none); `locked[g][o]` whether the
    cache holds a lock on it. The published (base) row holds the slot of locked pages and BAD elsewhere, padded past
    num_blocks like the external host row. `lie` makes the row publish a wrong slot (an invariant violation).
    """

    def __init__(self, attached, locked, pad=0, lie=None):
        self.attached, self.locked, self.pad, self.lie = attached, locked, pad, lie or {}
        self.history_length = 0
        self.aggregated_calls = self.base_calls = 0

    @property
    def num_blocks(self):
        return len(self.attached[0])

    def get_aggregated_page_indices(self, layer_group_id, valid_only=False):
        assert valid_only is False
        self.aggregated_calls += 1
        return list(self.attached[layer_group_id])

    def get_base_page_indices(self, layer_group_id):
        self.base_calls += 1
        row = [s if lk else BAD for s, lk in zip(self.attached[layer_group_id], self.locked[layer_group_id])]
        for (g, o), v in self.lie.items():
            if g == layer_group_id:
                row[o] = v
        return np.array(row + [BAD] * self.pad, dtype=np.int32)


def make_manager(caches):
    return SimpleNamespace(impl=SimpleNamespace(layer_grouping=[[0], [1]]), kv_cache_map=dict(caches))


def resolver(manager):
    return lambda request_id: KVCacheManagerV2.get_page_indices_by_layer_group_for_id(
        manager, request_id, mask_stale=False
    )


def lookup(manager):
    return lambda request_id, wanted, verify=False: KVCacheManagerV2.lookup_local_page_indices_for_id(
        manager, request_id, wanted, verify
    )


def random_cache(rng, num_blocks, unlocked_share, pad):
    attached, locked = [], []
    for _ in range(NUM_GROUPS):
        slots = rng.sample(range(10 * num_blocks + 10), num_blocks)
        a = [s if rng.random() > 0.1 else BAD for s in slots]  # some blocks have no page
        lk = [s != BAD and rng.random() >= unlocked_share for s in a]  # attached but unlocked (window-passed)
        attached.append(a)
        locked.append(lk)
    return FakeCache(attached, locked, pad=pad)


def random_pages(rng, cache, divergent, *, load, tag_share=0.95):
    """Pages naming blocks of `cache`, with rank 0's slot equal to this rank's unless `divergent` remaps some."""
    pages = []
    for _ in range(rng.randint(0, 40)):
        g = rng.randrange(NUM_GROUPS + (0 if load else 1))  # saves may name a group this rank does not have
        n = cache.num_blocks if cache is not None else 4
        o = rng.randrange(n + (0 if load else 3))  # saves may run past the blocks
        local = cache.attached[g][o] if cache is not None and g < NUM_GROUPS and o < n else BAD
        if load and local == BAD:
            continue  # a load always has a destination here (the unresolved-load case raises, tested below)
        rank0 = local if not divergent or rng.random() > 0.3 else local + 1000 + rng.randrange(50)
        if rank0 < 0:
            rank0 = rng.randrange(5000)
        ordinal = o if rng.random() < tag_share else -1
        pages.append(PageTransfer(bytes([rng.randrange(256)]) * 16, g, rank0, ordinal))
    return pages


def snapshot(metadata):
    def entry(e):
        for p in e.pages:
            assert type(p.page_index) is int and type(p.block_ordinal) is int
        return (e.request_id, [(p.block_hash, p.layer_group_id, p.page_index, p.block_ordinal) for p in e.pages])

    return (
        [entry(e) for e in metadata.loads],
        [entry(e) for e in metadata.saves],
        [(x.reservation_id, x.is_async, entry(x.transfers)) for x in metadata.prefix_loads],
    )


def run_localize(worker, metadata, manager, mode):
    worker._pgfix_fast = mode
    worker._remapped_pages = worker._unresolved_save_pages = 0
    worker._pgfix_lookups = worker._pgfix_fallbacks = worker._pgfix_mismatches = 0
    worker.bind_local_page_indices(resolver(manager))
    worker.bind_local_page_lookup(lookup(manager))
    worker._localize(metadata)
    return snapshot(metadata), (worker._remapped_pages, worker._unresolved_save_pages)


@pytest.mark.parametrize("divergent", [False, True], ids=["same_slots", "divergent"])
@pytest.mark.parametrize("unlocked_share", [0.0, 0.3], ids=["all_locked", "some_unlocked"])
def test_fast_localize_equals_the_full_table(store_config, fake_store, divergent, unlocked_share):
    rng = random.Random(1234 + 7 * divergent + int(10 * unlocked_share))
    with make_worker(fake_store, layout=make_layout()) as worker:
        for trial in range(60):
            caches = {
                rid: random_cache(rng, rng.randint(1, 60), unlocked_share, rng.choice([0, 7])) for rid in range(5)
            }
            manager = make_manager(caches)
            rids = list(range(7))  # 5 and 6 have no cache on this rank

            def transfers(rid, load):
                return RequestTransfers(rid, random_pages(rng, caches.get(rid), divergent, load=load))

            metadata = MooncakeStoreMetadata(
                loads=[transfers(r, True) for r in rng.sample(rids[:5], 2)],
                saves=[transfers(r, False) for r in rng.sample(rids, 4)],
                prefix_loads=[
                    PrefixLoadTransfers(100 + r, bool(r % 2), transfers(r, True)) for r in rng.sample(rids[:5], 2)
                ],
            )
            want = run_localize(worker, copy.deepcopy(metadata), manager, 0)
            for mode in (1, 2):
                got = run_localize(worker, copy.deepcopy(metadata), manager, mode)
                assert got == want, (trial, mode)
                assert worker._pgfix_mismatches == 0
            if not divergent:
                assert want[1][0] == 0  # nothing to remap when rank 0's slots are this rank's


def test_fast_localize_reads_only_the_published_row_when_every_page_is_locked(store_config, fake_store):
    cache = FakeCache([[10, 11, 12, 13], [20, 21, 22, 23]], [[True] * 4, [True] * 4], pad=4)
    manager = make_manager({7: cache})
    metadata = MooncakeStoreMetadata(
        saves=[RequestTransfers(7, [PageTransfer(b"h" * 16, 1, 21, 1), PageTransfer(b"i" * 16, 1, 99, 2)])],
        loads=[RequestTransfers(7, [PageTransfer(b"j" * 16, 0, 55, 3)])],
    )
    with make_worker(fake_store, layout=make_layout()) as worker:
        snap, counts = run_localize(worker, metadata, manager, 1)
        assert snap[1] == [(7, [(b"h" * 16, 1, 21, 1), (b"i" * 16, 1, 22, 2)])]
        assert snap[0] == [(7, [(b"j" * 16, 0, 13, 3)])]
        assert counts == (2, 0)
        # One published-row read per (request, group) named; the full table was never built.
        assert cache.aggregated_calls == 0 and cache.base_calls == 2
        assert (worker._pgfix_lookups, worker._pgfix_fallbacks) == (2, 0)


def test_an_unlocked_page_is_answered_by_the_full_query(store_config, fake_store):
    # Block 0 of group 0 has a page attached but not locked (the window passed it): the row publishes BAD there.
    cache = FakeCache([[10, 11], [20, 21]], [[False, True], [True, True]])
    manager = make_manager({7: cache})
    metadata = MooncakeStoreMetadata(saves=[RequestTransfers(7, [PageTransfer(b"h" * 16, 0, 99, 0)])])
    with make_worker(fake_store, layout=make_layout()) as worker:
        snap, counts = run_localize(worker, metadata, manager, 1)
        assert snap[1] == [(7, [(b"h" * 16, 0, 10, 0)])] and counts == (1, 0)
        assert worker._pgfix_fallbacks == 1 and cache.aggregated_calls == 1


def test_a_load_with_no_slot_here_raises_the_same_error(store_config, fake_store):
    cache = FakeCache([[10, BAD], [20, 21]], [[True, False], [True, True]])
    errors = []
    with make_worker(fake_store, layout=make_layout()) as worker:
        for mode in (0, 1):
            metadata = MooncakeStoreMetadata(
                loads=[RequestTransfers(7, [PageTransfer(b"h" * 16, 1, 99, 0), PageTransfer(b"h" * 16, 0, 5, 1)])]
            )
            with pytest.raises(RuntimeError, match="no page for block 1 of layer group 0") as info:
                run_localize(worker, metadata, make_manager({7: cache}), mode)
            errors.append((str(info.value), worker._remapped_pages))
    assert errors[0] == errors[1]


def test_switch_off_never_calls_the_lookup(store_config, fake_store, monkeypatch):
    monkeypatch.delenv(worker_module.PGFIX_FAST_ENV, raising=False)
    cache = FakeCache([[10, 11]], [[True, True]])
    manager = SimpleNamespace(impl=SimpleNamespace(layer_grouping=[[0]]), kv_cache_map={7: cache})
    with make_worker(fake_store, layout=make_layout()) as worker:
        assert worker._pgfix_fast == 0
        worker.bind_local_page_indices(resolver(manager))
        worker.bind_local_page_lookup(lambda *args: pytest.fail("lookup called with the switch off"))
        metadata = MooncakeStoreMetadata(saves=[RequestTransfers(7, [PageTransfer(b"h" * 16, 0, 3, 1)])])
        worker._localize(metadata)
        assert [p.page_index for p in metadata.saves[0].pages] == [11]
        assert cache.base_calls == 0 and cache.aggregated_calls == 1


def test_switch_on_arms_with_a_receipt(store_config, fake_store, monkeypatch):
    monkeypatch.setenv(worker_module.PGFIX_FAST_ENV, "2")
    with make_worker(fake_store, layout=make_layout()) as worker:
        assert worker._pgfix_fast == 2


def test_verify_counts_a_published_slot_that_differs_and_uses_the_full_query(store_config, fake_store):
    # An invariant violation (the row publishes slot 77 for a page attached on slot 11): verify catches it.
    cache = FakeCache([[10, 11]], [[True, True]], lie={(0, 1): 77})
    manager = SimpleNamespace(impl=SimpleNamespace(layer_grouping=[[0]]), kv_cache_map={7: cache})
    with make_worker(fake_store, layout=make_layout()) as worker:
        metadata = MooncakeStoreMetadata(saves=[RequestTransfers(7, [PageTransfer(b"h" * 16, 0, 11, 1)])])
        snap, _ = run_localize(worker, metadata, manager, 2)
        assert snap[1] == [(7, [(b"h" * 16, 0, 11, 1)])]
        assert worker._pgfix_mismatches == 1


@pytest.mark.parametrize("retarget", [True, False])
@pytest.mark.parametrize("unlocked_share", [0.0, 0.3], ids=["all_locked", "some_unlocked"])
def test_fast_recheck_after_commit_equals_the_full_table(store_config, fake_store, retarget, unlocked_share):
    rng = random.Random(99 + int(retarget) + int(10 * unlocked_share))
    with make_worker(fake_store, layout=make_layout()) as worker:
        for trial in range(60):
            caches = {rid: random_cache(rng, rng.randint(1, 50), unlocked_share, 3) for rid in range(4)}
            manager = make_manager(caches)
            saves = [
                RequestTransfers(r, random_pages(rng, caches.get(r), True, load=False)) for r in rng.sample(range(6), 4)
            ]
            overrides = {}
            for entry in saves:
                for page in entry.pages[: rng.randint(0, 3)]:
                    overrides[(page.layer_group_id, page.page_index)] = (len(overrides), 0, 0)
            results = []
            for mode in (0, 1, 2):
                worker._pgfix_fast = mode
                worker._swap_checked = worker._swap_retargeted = worker._swap_dropped = 0
                worker._pgfix_mismatches = 0
                saves_copy = copy.deepcopy(saves)
                if mode == 0:
                    out = worker._recheck_after_commit(saves_copy, overrides, resolver(manager), retarget)
                else:
                    out = worker._recheck_after_commit_fast(saves_copy, overrides, lookup(manager), retarget)
                snap = [
                    (e.request_id, [(p.block_hash, p.layer_group_id, p.page_index, p.block_ordinal) for p in e.pages])
                    for e in out
                ]
                for e in out:
                    assert all(type(p.page_index) is int for p in e.pages)
                results.append((snap, worker._swap_checked, worker._swap_retargeted, worker._swap_dropped))
                assert worker._pgfix_mismatches == 0
            assert results[1] == results[0] and results[2] == results[0], trial


def test_localize_then_commit_swaps_then_recheck_equals_the_full_table(store_config, fake_store):
    """The re-check of entries `_localize_fast` produced reuses their columns; commits swap slots in between."""
    rng = random.Random(4242)
    with make_worker(fake_store, layout=make_layout()) as worker:
        reused = 0
        for trial in range(60):
            caches = {rid: random_cache(rng, rng.randint(1, 50), rng.choice([0.0, 0.3]), 5) for rid in range(5)}
            def transfers(rid, load):
                return RequestTransfers(rid, random_pages(rng, caches.get(rid), True, load=load))

            metadata = MooncakeStoreMetadata(
                loads=[transfers(r, True) for r in rng.sample(range(5), 2)],
                saves=[transfers(r, False) for r in rng.sample(range(7), 4)],
            )
            swaps = [
                (r, g, o, rng.choice([BAD, 90000 + rng.randrange(10**5)]))
                for r, c in caches.items()
                for g in range(NUM_GROUPS)
                for o in range(c.num_blocks)
                if c.attached[g][o] != BAD and rng.random() < 0.15
            ]
            results = []
            for mode in (0, 1, 2):
                state = copy.deepcopy(caches)
                manager = make_manager(state)
                meta = copy.deepcopy(metadata)
                run_localize(worker, meta, manager, mode)
                overrides = {
                    (p.layer_group_id, p.page_index): (0, 0, 0)
                    for e in meta.saves
                    for p in e.pages[:2]
                    if p.layer_group_id == 0
                }
                for r, g, o, slot in swaps:  # the commit after the pass rebases blocks (or a block loses its page)
                    state[r].attached[g][o] = slot
                    state[r].locked[g][o] = slot != BAD
                worker._swap_checked = worker._swap_retargeted = worker._swap_dropped = 0
                if mode == 0:
                    out = worker._recheck_after_commit(meta.saves, overrides, resolver(manager), True)
                else:
                    reused += sum(getattr(e, "_page_cols", None) is not None for e in meta.saves)
                    out = worker._recheck_after_commit_fast(meta.saves, overrides, lookup(manager), True)
                snap = [
                    (e.request_id, [(p.block_hash, p.layer_group_id, p.page_index, p.block_ordinal) for p in e.pages])
                    for e in out
                ]
                results.append((snap, worker._swap_checked, worker._swap_retargeted, worker._swap_dropped))
            assert results[1] == results[0] and results[2] == results[0], trial
        assert reused > 0


def test_lookup_equals_the_unmasked_table_entry_for_entry():
    """The KV cache manager method alone, fuzzed: every ordinal, every group, missing requests, padding."""
    rng = random.Random(5)
    for trial in range(300):
        caches = {
            rid: random_cache(rng, rng.randint(1, 40), rng.choice([0.0, 0.2, 1.0]), rng.choice([0, 5]))
            for rid in range(3)
        }
        manager = make_manager(caches)
        for rid in range(4):
            table = KVCacheManagerV2.get_page_indices_by_layer_group_for_id(manager, rid, mask_stale=False)
            wanted = {
                g: np.array([rng.randrange(45) for _ in range(rng.randint(0, 12))], dtype=np.int64) for g in range(3)
            }
            for verify in (False, True):
                found, fallbacks, mismatches = KVCacheManagerV2.lookup_local_page_indices_for_id(
                    manager, rid, wanted, verify
                )
                assert mismatches == 0
                for g, ordinals in wanted.items():
                    row = table[g] if g < len(table) else []
                    want = [row[o] if o < len(row) else BAD for o in ordinals.tolist()]
                    assert found[g].tolist() == want, (trial, rid, g)


def test_handle_metadata_binds_the_lookup():
    bound = {}
    probe = lambda *args: None  # noqa: E731
    manager = SimpleNamespace(
        _scheduler_output=SimpleNamespace(prefix_loads=[]),
        _bound_prefix_load_ids=set(),
        _bound_scheduler_output=None,
        _allocation_fence=None,
        _execution_stream=None,
        _local_page_indices=None,
        _local_page_lookup=probe,
        _run_on_leader=lambda f: f(),
        scheduler=SimpleNamespace(build_connector_meta=lambda output: "meta"),
        worker=SimpleNamespace(
            bind_allocation_fence=lambda fence: None,
            bind_local_page_lookup=lambda f: bound.__setitem__("lookup", f),
            bind_connector_meta=lambda meta: bound.__setitem__("meta", meta),
        ),
    )
    kvc.KvCacheConnectorManager.handle_metadata(manager)
    assert bound == {"lookup": probe, "meta": "meta"}
