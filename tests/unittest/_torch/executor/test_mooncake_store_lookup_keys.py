# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""lkp: fewer keys per store lookup (L1 staged probe, L2 run reuse, L3 loaded keys into the local-stored set).

Every lookup must answer exactly what the one-round probe of every key answers. CPU only: the store is the
in-process fake from test_mooncake_store_connector, with the worker's peer namespaces widened to several ranks
so partial-rank saves can be modelled.
"""

import random

import pytest
from test_mooncake_store_connector import (  # noqa: F401 (fixtures)
    TOKENS_PER_BLOCK,
    fake_store,
    make_layout,
    make_window_layout,
    make_worker,
    store_config,
)

from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import worker as worker_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.stored_set import LocalStoredSet
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.worker import PrefixHit

pytestmark = pytest.mark.cpu_only

RANKS = 4
SWITCHES = (
    worker_module.LOOKUP_STAGED_ENV,
    worker_module.LOOKUP_REUSE_MS_ENV,
    worker_module.STORED_FROM_LOADS_ENV,
)


@pytest.fixture(autouse=True)
def _switches_off(monkeypatch):
    for name in SWITCHES:
        monkeypatch.delenv(name, raising=False)


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def widen(worker, ranks=RANKS):
    """Give the worker `ranks` peer namespaces per group, as under TP, and return them by group."""
    worker._world_size = ranks
    for group in list(worker._peer_namespaces):
        bytes_per_page = worker._addressing.bytes_per_page(group)
        worker._peer_namespaces[group] = tuple(
            worker._namespace(rank, group, bytes_per_page) for rank in range(ranks)
        )
    return worker._peer_namespaces


def set_mode(worker, staged, reuse_ms=0.0, clock=None):
    worker._lookup_staged = staged
    worker._run_reuse = (
        worker_module._RunReuse(reuse_ms / 1e3, clock=clock or Clock()) if reuse_ms else None
    )


def hashes_for(n, salt=0):
    return [bytes([salt, index % 256, index // 256]) * 5 + b"\x00" for index in range(n)]


def random_store(rng, fake_store, peers, hashes, window_group=None, worker=None):
    """Full-attention runs per rank with holes and partial-rank saves; windows at a few prompt ends."""
    fake_store.objects.clear()
    n = len(hashes)
    full_groups = [g for g in peers if g != window_group]
    base_run = rng.choice([0, 1, n // 3, n // 2, n - 1, n]) if n else 0
    for group in full_groups:
        for rank, namespace in enumerate(peers[group]):
            run = base_run if rng.random() < 0.8 else rng.randint(0, n)  # a rank whose saves lag
            for block in range(run):
                fake_store.objects.add(namespace.key(hashes[block]))
            for _ in range(rng.choice([0, 0, 1, 2])):  # holes: an evicted page inside the run
                if n:
                    fake_store.objects.discard(namespace.key(hashes[rng.randrange(n)]))
            for _ in range(rng.choice([0, 1])):  # stragglers past the run
                if n:
                    fake_store.objects.add(namespace.key(hashes[rng.randrange(n)]))
    if window_group is not None and n:
        ends = sorted({rng.randint(1, n) for _ in range(rng.choice([0, 1, 2, 3]))})
        for end in ends:
            _, stale_end = worker.stale_blocks(window_group, end * TOKENS_PER_BLOCK)
            for rank, namespace in enumerate(peers[window_group]):
                if rng.random() < 0.1:
                    continue  # this rank's window save never landed
                for block in range(stale_end, end):
                    fake_store.objects.add(namespace.key(hashes[block]))
        for _ in range(rng.choice([0, 0, 1])):
            namespace = rng.choice(peers[window_group])
            fake_store.objects.discard(namespace.key(hashes[rng.randrange(n)]))


# ---- switches ----


def test_off_by_default(store_config, fake_store):  # noqa: F811
    with make_worker(fake_store, layout=make_layout()) as worker:
        assert worker._lookup_staged is False
        assert worker._run_reuse is None
        assert worker._stored_from_loads is False
        hashes = hashes_for(100)
        assert worker.count_prefix_hit(hashes) == 0
        # The one-round probe of every key, as before.
        assert len(fake_store.exist_calls) == 1 and len(fake_store.exist_calls[0]) == 100


def test_switches_read_from_env(store_config, fake_store, monkeypatch):  # noqa: F811
    monkeypatch.setenv(worker_module.LOOKUP_STAGED_ENV, "1")
    monkeypatch.setenv(worker_module.LOOKUP_REUSE_MS_ENV, "750")
    monkeypatch.setenv(worker_module.STORED_FROM_LOADS_ENV, "1")
    with make_worker(fake_store, layout=make_layout()) as worker:
        assert worker._lookup_staged is True
        assert abs(worker._run_reuse._ttl - 0.75) < 1e-9
        assert worker._stored_from_loads is False  # F3 is off: nothing to add to
    monkeypatch.setenv("TRTLLM_MOONCAKE_STORE_LOCAL_STORED_SET", "1")
    with make_worker(fake_store, layout=make_layout()) as worker:
        assert worker._stored_from_loads is True


# ---- L1 staged probe ----


def test_staged_search_matches_a_full_scan():
    rng = random.Random(3)
    for _ in range(5000):
        n = rng.choice([1, 2, 63, 64, 65, 66, 129, 500, 3000])
        start = rng.randint(0, n)
        complete = [True] * n
        for _ in range(rng.choice([0, 1, 4])):
            complete[rng.randrange(n)] = False
        for block in range(start):
            complete[block] = True
        expected = next((b for b in range(n) if not complete[b]), n)
        asked = []

        def probe(blocks):
            asked.extend(blocks)
            return [complete[b] for b in blocks]

        assert worker_module._leading_run_staged(n, start, probe) == expected
        assert len(asked) == len(set(asked))  # no block asked twice
        assert all(block >= start for block in asked)
        assert worker_module._leading_run_staged(n, start, lambda blocks: None) in (None, n)


def test_staged_cold_prompt_asks_only_its_sentinels(store_config, fake_store):  # noqa: F811
    with make_worker(fake_store, layout=make_layout()) as worker:
        peers = widen(worker)
        set_mode(worker, staged=True)
        hashes = hashes_for(3000)
        assert worker.count_prefix_hit(hashes) == 0
        probed = sum(len(call) for call in fake_store.exist_calls)
        stride = 54  # isqrt(3000)
        assert probed == RANKS * (len(range(stride, 3000, stride)) + 2)
        assert probed < 3000 * RANKS / 20
        # A warm prompt: all blocks below the miss, plus at most a stride and the sentinels past it.
        for block in range(2000):
            for namespace in peers[0]:
                fake_store.objects.add(namespace.key(hashes[block]))
        fake_store.exist_calls.clear()
        rounds = worker._probe_calls
        assert worker.count_prefix_hit(hashes) == 2000
        probed = sum(len(call) for call in fake_store.exist_calls)
        assert 2000 * RANKS <= probed <= (2000 + stride + 3000 // stride + 2) * RANKS
        # Two rounds (the second one goes out as concurrent PROBE_CHUNK-key calls).
        assert worker._probe_calls - rounds == 2


@pytest.mark.parametrize("seed", range(6))
def test_count_prefix_hit_matches_the_one_round_probe(store_config, fake_store, seed):  # noqa: F811
    rng = random.Random(seed)
    with make_worker(fake_store, layout=make_layout(num_groups=2)) as worker:
        peers = widen(worker)
        for _ in range(150):
            hashes = hashes_for(rng.choice([1, 7, 64, 65, 200, 700]), salt=rng.randrange(256))
            random_store(rng, fake_store, peers, hashes)
            set_mode(worker, staged=False)
            expected = worker.count_prefix_hit(hashes)
            for staged, reuse_ms in ((True, 0), (False, 1000), (True, 1000)):
                set_mode(worker, staged=staged, reuse_ms=reuse_ms)
                assert worker.count_prefix_hit(hashes) == expected
                # A repeat answered from the reused run, with the store unchanged, still agrees.
                assert worker.count_prefix_hit(hashes) == expected


@pytest.mark.parametrize("seed", range(8))
def test_find_prefix_hit_matches_with_windows_rewinds_and_fallback(store_config, fake_store, seed):  # noqa: F811
    rng = random.Random(100 + seed)
    with make_worker(fake_store, layout=make_window_layout(num_slots=64)) as worker:
        peers = widen(worker)
        for _ in range(120):
            n = rng.choice([2, 9, 40, 70, 150])
            hashes = hashes_for(n, salt=rng.randrange(256))
            random_store(rng, fake_store, peers, hashes, window_group=1, worker=worker)
            first_block = rng.choice([0, 0, rng.randrange(n)])
            rewind = rng.choice([0, TOKENS_PER_BLOCK, 3 * TOKENS_PER_BLOCK, 40 * TOKENS_PER_BLOCK])
            worker._window_fallback = rng.random() < 0.8
            candidates = hashes[first_block:]

            set_mode(worker, staged=False)
            before = dict(worker._lookup_stats.counts)
            expected = worker.find_prefix_hit(candidates, first_block, rewind)
            outcome = {k: v - before[k] for k, v in worker._lookup_stats.counts.items() if v != before[k]}
            for staged, reuse_ms in ((True, 0), (False, 1000), (True, 1000)):
                set_mode(worker, staged=staged, reuse_ms=reuse_ms)
                for _repeat in range(2):
                    before = dict(worker._lookup_stats.counts)
                    assert worker.find_prefix_hit(candidates, first_block, rewind) == expected
                    got = {k: v - before[k] for k, v in worker._lookup_stats.counts.items() if v != before[k]}
                    assert got == outcome


def test_partial_rank_save_ends_the_run_where_one_rank_misses(store_config, fake_store):  # noqa: F811
    with make_worker(fake_store, layout=make_layout()) as worker:
        peers = widen(worker)
        hashes = hashes_for(300)
        for rank, namespace in enumerate(peers[0]):
            for block in range(250 if rank != 2 else 117):
                fake_store.objects.add(namespace.key(hashes[block]))
        for staged in (False, True):
            set_mode(worker, staged=staged)
            assert worker.count_prefix_hit(hashes) == 117


# ---- L2 run reuse ----


def test_reuse_probes_only_past_the_confirmed_run(store_config, fake_store):  # noqa: F811
    clock = Clock()
    with make_worker(fake_store, layout=make_layout()) as worker:
        peers = widen(worker)
        set_mode(worker, staged=False, reuse_ms=1000, clock=clock)
        hashes = hashes_for(200)
        for block in range(150):
            for namespace in peers[0]:
                fake_store.objects.add(namespace.key(hashes[block]))
        assert worker.count_prefix_hit(hashes) == 150
        fake_store.exist_calls.clear()
        clock.now += 0.5
        # Five more blocks landed since: the repeat sees them, asking only from block 150 on.
        for block in range(150, 155):
            for namespace in peers[0]:
                fake_store.objects.add(namespace.key(hashes[block]))
        assert worker.count_prefix_hit(hashes) == 155
        assert sum(len(call) for call in fake_store.exist_calls) == (200 - 150) * RANKS
        assert worker._run_reuse.reused_blocks == 150


def test_reuse_expires_and_keeps_the_oldest_confirmation(store_config, fake_store):  # noqa: F811
    clock = Clock()
    with make_worker(fake_store, layout=make_layout()) as worker:
        peers = widen(worker)
        set_mode(worker, staged=True, reuse_ms=1000, clock=clock)
        hashes = hashes_for(200)
        for block in range(100):
            for namespace in peers[0]:
                fake_store.objects.add(namespace.key(hashes[block]))
        assert worker.count_prefix_hit(hashes) == 100  # confirmed at t
        clock.now += 0.8
        assert worker.count_prefix_hit(hashes) == 100  # reused; the run keeps time t, not t + 0.8
        clock.now += 0.3  # 1.1 s after t: expired, although the last lookup was 0.3 s ago
        fake_store.objects.discard(peers[0][3].key(hashes[40]))  # evicted since
        fake_store.exist_calls.clear()
        assert worker.count_prefix_hit(hashes) == 40
        assert any(peers[0][0].key(hashes[0]) in call for call in fake_store.exist_calls)


def test_reuse_needs_the_same_blocks(store_config, fake_store):  # noqa: F811
    with make_worker(fake_store, layout=make_layout()) as worker:
        peers = widen(worker)
        set_mode(worker, staged=False, reuse_ms=60000)
        hashes = hashes_for(100)
        for block in range(80):
            for namespace in peers[0]:
                fake_store.objects.add(namespace.key(hashes[block]))
        assert worker.count_prefix_hit(hashes) == 80
        # Same first block, different continuation from block 50 on: nothing reused past what matches.
        other = hashes[:50] + hashes_for(50, salt=9)
        assert worker.count_prefix_hit(other) == 50
        assert worker._run_reuse.reused_blocks == 0
        # A shorter candidate list that still covers the run's last block reuses it; one that ends sooner does not.
        assert worker.count_prefix_hit(hashes[:90]) == 80
        assert worker._run_reuse.reused_blocks == 80
        assert worker.count_prefix_hit(hashes[:60]) == 60


def test_failed_probe_is_a_miss_and_is_not_remembered(store_config, fake_store):  # noqa: F811
    with make_worker(fake_store, layout=make_layout()) as worker:
        peers = widen(worker)
        set_mode(worker, staged=True, reuse_ms=60000)
        hashes = hashes_for(300)
        for block in range(300):
            for namespace in peers[0]:
                fake_store.objects.add(namespace.key(hashes[block]))
        real = fake_store.batch_is_exist
        fake_store.batch_is_exist = lambda keys: (_ for _ in ()).throw(RuntimeError("fabric down"))
        assert worker.count_prefix_hit(hashes) == 0
        fake_store.batch_is_exist = real
        assert worker.count_prefix_hit(hashes) == 300
        assert worker._run_reuse.reused_blocks == 0


# ---- statistics ----


def test_lookup_stats_report_keys_rounds_and_reuse(store_config, fake_store, monkeypatch):  # noqa: F811
    monkeypatch.setattr(worker_module, "LOAD_STATS_INTERVAL", 0.0)
    lines = []
    monkeypatch.setattr(worker_module.logger, "info", lambda message: lines.append(message))
    with make_worker(fake_store, layout=make_window_layout()) as worker:
        set_mode(worker, staged=True, reuse_ms=1000)
        worker.find_prefix_hit(hashes_for(10), 0)
    report = [line for line in lines if "lookups:" in line][-1]
    assert "1 lookups (" in report and "blocks hit" in report
    assert "keys probed 10 (10/lookup), probe rounds 1 (1.00/lookup), reused blocks 0" in report


# ---- L3 loaded keys ----


def test_loaded_pages_join_the_local_stored_set(store_config, fake_store, monkeypatch):  # noqa: F811
    from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.metadata import (
        PageTransfer,
        RequestTransfers,
    )

    monkeypatch.setenv("TRTLLM_MOONCAKE_STORE_LOCAL_STORED_SET", "1")
    for on in (False, True):
        if on:
            monkeypatch.setenv(worker_module.STORED_FROM_LOADS_ENV, "1")
        with make_worker(fake_store, layout=make_layout()) as worker:
            assert isinstance(worker._local_stored, LocalStoredSet)
            hashes = hashes_for(3)
            keys = [worker._namespaces[0].key(h) for h in hashes]
            fake_store.objects.update(keys)
            transfers = RequestTransfers(
                request_id=1,
                pages=[PageTransfer(layer_group_id=0, page_index=i, block_hash=h) for i, h in enumerate(hashes)],
            )
            worker._load_pages([transfers], None, 0)
            assert worker._local_stored.known(keys) == [on] * 3
