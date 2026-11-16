# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rank-local store lookups (TRTLLM_MOONCAKE_STORE_RANK_LOCAL_LOOKUP=1).

Every rank probes only its own shard's keys and the ranks agree by an
allreduce(MIN). These tests run four worker "ranks" in one process against one
fake store, with the allreduce done across threads, and check that every rank
agrees on exactly what the leader's probe of all shards answers today, that a
missing shard or a failed probe stays conservative, and that the follower
ranks join the collective for exactly the requests the leader does.
"""

import contextlib
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from test_mooncake_store_connector import TOKENS_PER_BLOCK as TPB
from test_mooncake_store_connector import (  # noqa: F401
    fake_store,
    make_layout,
    make_llm_args,
    make_request,
    make_window_layout,
    store_config,
)

from tensorrt_llm._torch.pyexecutor.connectors import kv_cache_connector as kvc
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import worker as worker_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.keys import BlockHashChain
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.scheduler import (
    MooncakeStoreConnectorScheduler,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.worker import (
    MooncakeStoreConnectorWorker,
    PrefixHit,
)

pytestmark = pytest.mark.cpu_only

WORLD = 4


class ThreadMin:
    """An allreduce(MIN) across `n` threads, one reducer per rank; a desync times out instead of hanging."""

    def __init__(self, n, timeout=20.0):
        self.n = n
        self.barrier = threading.Barrier(n, timeout=timeout)
        self.slots = [None] * n
        self.calls = [0] * n

    def for_rank(self, rank):
        def reduce(values):
            self.calls[rank] += 1
            self.slots[rank] = list(values)
            self.barrier.wait()
            lengths = {len(slot) for slot in self.slots}
            assert len(lengths) == 1, f"ranks reduced different lengths {lengths}"
            out = [min(column) for column in zip(*self.slots)]
            self.barrier.wait()
            return out

        return reduce


@contextlib.contextmanager
def make_ranks(fake_store, monkeypatch, layout, *, enabled=True, rewind_tokens=0):  # noqa: F811
    """`WORLD` workers sharing one fake store, rank r built as `mpi_rank() == r`."""
    if enabled:
        monkeypatch.setenv(worker_module.RANK_LOCAL_LOOKUP_ENV, "1")
    else:
        monkeypatch.delenv(worker_module.RANK_LOCAL_LOOKUP_ENV, raising=False)
    monkeypatch.setattr(worker_module, "mpi_world_size", lambda: WORLD)
    workers = []
    try:
        for rank in range(WORLD):
            monkeypatch.setattr(worker_module, "mpi_rank", lambda rank=rank: rank)
            llm_args = make_llm_args()
            llm_args.kv_cache_config.block_reuse_config = SimpleNamespace(
                swa_endpoint_rewind_tokens=rewind_tokens
            )
            worker = MooncakeStoreConnectorWorker(llm_args)
            fake_store.workers.append(worker)
            worker.register_kv_cache_layout(layout)
            workers.append(worker)
        if enabled:
            reducer = ThreadMin(WORLD)
            for rank, worker in enumerate(workers):
                worker._lookup_reducer = reducer.for_rank(rank)
                worker._test_reducer = reducer
        yield workers
    finally:
        for worker in workers:
            worker.shutdown()


def on_all_ranks(workers, call):
    """Run `call(rank, worker)` on every rank concurrently, as the ranks' executor threads would."""
    with ThreadPoolExecutor(max_workers=len(workers)) as pool:
        futures = [pool.submit(call, rank, worker) for rank, worker in enumerate(workers)]
        return [future.result(timeout=60) for future in futures]


@contextlib.contextmanager
def leader_view(worker):
    """The leader's probe of every shard, as with the switch off."""
    saved = worker._rank_local_lookup
    worker._rank_local_lookup = False
    try:
        yield worker
    finally:
        worker._rank_local_lookup = saved


def put(fake_store, worker, group, block_hash):  # noqa: F811
    fake_store.objects.add(worker._namespaces[group].key(block_hash))


def hashes_of(n, salt=0):
    return [bytes([salt, index]) * 8 for index in range(n)]


# ---- the switch ----


def test_switch_is_off_by_default(store_config, fake_store, monkeypatch):  # noqa: F811
    with make_ranks(fake_store, monkeypatch, make_layout(), enabled=False) as workers:
        assert not any(worker.rank_local_lookup for worker in workers)
        # The follower does nothing and the lookup probes every shard from one rank.
        workers[1].follow_prefix_lookup(make_request(1, list(range(5 * TPB))), 0)
        fake_store.exist_calls.clear()
        workers[0].count_prefix_hit(hashes_of(3))
        assert len(fake_store.exist_calls) == 1 and len(fake_store.exist_calls[0]) == 3 * WORLD


def test_switch_needs_more_than_one_rank_and_no_attention_dp(store_config, fake_store, monkeypatch):  # noqa: F811
    monkeypatch.setenv(worker_module.RANK_LOCAL_LOOKUP_ENV, "1")
    single = MooncakeStoreConnectorWorker(make_llm_args())
    fake_store.workers.append(single)
    assert not single.rank_local_lookup
    monkeypatch.setattr(worker_module, "mpi_world_size", lambda: WORLD)
    llm_args = make_llm_args()
    llm_args.enable_attention_dp = True
    adp = MooncakeStoreConnectorWorker(llm_args)
    fake_store.workers.append(adp)
    assert not adp.rank_local_lookup
    tp = MooncakeStoreConnectorWorker(make_llm_args())
    fake_store.workers.append(tp)
    assert tp.rank_local_lookup


# ---- full attention: count_prefix_hit ----


def _fill_full(fake_store, workers, hashes, rng, num_groups, pattern):  # noqa: F811
    """Store pages per shard; `pattern` picks how the shards differ."""
    n = len(hashes)
    for rank, worker in enumerate(workers):
        if pattern == "missing_shard" and rank == 2:
            continue
        upto = n
        if pattern == "ragged":
            upto = rng.randrange(0, n + 1)
        for block in range(upto):
            for group in range(num_groups):
                if pattern == "holes" and rng.random() < 0.08:
                    continue
                put(fake_store, worker, group, hashes[block])


@pytest.mark.parametrize("pattern", ["all", "ragged", "holes", "missing_shard"])
@pytest.mark.parametrize("num_groups", [1, 2])
def test_rank_local_count_matches_the_leaders_probe(
    store_config,
    fake_store,
    monkeypatch,
    pattern,
    num_groups,  # noqa: F811
):
    rng = random.Random(f"{pattern}-{num_groups}")
    with make_ranks(fake_store, monkeypatch, make_layout(num_groups=num_groups)) as workers:
        for trial in range(25):
            fake_store.objects.clear()
            hashes = hashes_of(rng.randrange(1, 40), salt=trial)
            _fill_full(fake_store, workers, hashes, rng, num_groups, pattern)
            with leader_view(workers[0]) as leader:
                expected = leader.count_prefix_hit(hashes)
            fake_store.exist_calls.clear()
            agreed = on_all_ranks(workers, lambda rank, worker: worker.count_prefix_hit(hashes))
            assert agreed == [expected] * WORLD, (trial, pattern)
            if pattern == "missing_shard":
                assert expected == 0
            # Each rank asked only about its own shard's keys.
            assert sum(len(call) for call in fake_store.exist_calls) == len(hashes) * num_groups * WORLD
            for call in fake_store.exist_calls:
                assert len({re.search(r"/w\d+r(\d+)/", key).group(1) for key in call}) == 1


def test_a_failed_probe_on_any_rank_is_a_miss_everywhere(store_config, fake_store, monkeypatch):  # noqa: F811
    with make_ranks(fake_store, monkeypatch, make_layout()) as workers:
        hashes = hashes_of(6)
        for worker in workers:
            for block_hash in hashes:
                put(fake_store, worker, 0, block_hash)
        assert on_all_ranks(workers, lambda r, w: w.count_prefix_hit(hashes)) == [6] * WORLD

        real = fake_store.batch_is_exist
        failing = workers[3]._namespaces[0].prefix

        def flaky(keys):
            if any(key.startswith(failing) for key in keys):
                raise RuntimeError("master unreachable")
            return real(keys)

        monkeypatch.setattr(fake_store, "batch_is_exist", flaky)
        assert on_all_ranks(workers, lambda r, w: w.count_prefix_hit(hashes)) == [0] * WORLD


def test_rank_local_count_of_a_long_prompt_matches(store_config, fake_store, monkeypatch):  # noqa: F811
    """A ~130k-token prompt (32k blocks at 4 tokens/block), probed in chunks, with one shard stopping early."""
    monkeypatch.setenv(worker_module.PROBE_CHUNK_ENV, "4096")
    with make_ranks(fake_store, monkeypatch, make_layout()) as workers:
        tokens = [random.Random(3).randrange(200000) for _ in range(32 * 1024 * TPB + 3)]
        chain = BlockHashChain(TPB)
        chain.extend(tokens)
        hashes = list(chain.hashes)
        for rank, worker in enumerate(workers):
            for block_hash in hashes[: len(hashes) - 1000 * rank]:
                put(fake_store, worker, 0, block_hash)
        with leader_view(workers[0]) as leader:
            expected = leader.count_prefix_hit(hashes)
        assert expected == len(hashes) - 3000
        assert on_all_ranks(workers, lambda r, w: w.count_prefix_hit(hashes)) == [expected] * WORLD


# ---- with the lkp lookup-key switches (L1 staged probes, L2 run reuse) stacked on ----


@contextlib.contextmanager
def plain_leader_view(worker):
    """The leader's one-round probe of every shard: rank-local lookups and lkp L1/L2 off."""
    saved = (worker._rank_local_lookup, worker._lookup_staged, worker._run_reuse)
    worker._rank_local_lookup, worker._lookup_staged, worker._run_reuse = False, False, None
    try:
        yield worker
    finally:
        worker._rank_local_lookup, worker._lookup_staged, worker._run_reuse = saved


def long_hashes(n, salt=0):
    return [(salt.to_bytes(2, "little") + index.to_bytes(2, "little")) * 4 for index in range(n)]


@pytest.mark.parametrize("pattern", ["all", "ragged", "holes", "missing_shard"])
@pytest.mark.parametrize("staged,reuse_ms", [(True, 0), (False, 60000), (True, 60000)])
def test_rank_local_count_with_lookup_key_switches_matches_the_leaders_probe(
    store_config,
    fake_store,
    monkeypatch,
    pattern,
    staged,
    reuse_ms,  # noqa: F811
):
    """Each rank's staged / reused run over its own keys, agreed by MIN, is the leader's one-round answer.

    A repeat lookup after more pages land (the reuse case) still finds them, and every rank joins exactly one
    agreement per lookup whatever its local probe rounds were.
    """
    monkeypatch.setenv(worker_module.LOOKUP_STAGED_ENV, "1" if staged else "0")
    monkeypatch.setenv(worker_module.LOOKUP_REUSE_MS_ENV, str(reuse_ms))
    rng = random.Random(f"lkp-{pattern}-{staged}-{reuse_ms}")
    num_groups = 2
    with make_ranks(fake_store, monkeypatch, make_layout(num_groups=num_groups)) as workers:
        assert all(w._lookup_staged == staged for w in workers)
        assert all((w._run_reuse is not None) == bool(reuse_ms) for w in workers)
        lookups = 0
        for trial in range(12):
            fake_store.objects.clear()
            n = rng.randrange(1, 300)
            hashes = long_hashes(n, salt=trial)
            _fill_full(fake_store, workers, hashes, rng, num_groups, pattern)
            for repeat in range(2):
                if repeat:
                    # More pages land on every shard (never fewer): a reused run must not hide them.
                    grow = rng.randrange(0, n + 1)
                    for rank, worker in enumerate(workers):
                        if pattern == "missing_shard" and rank == 2:
                            continue
                        for block in range(grow):
                            for group in range(num_groups):
                                put(fake_store, worker, group, hashes[block])
                with plain_leader_view(workers[0]) as leader:
                    expected = leader.count_prefix_hit(hashes)
                fake_store.exist_calls.clear()
                agreed = on_all_ranks(workers, lambda rank, worker: worker.count_prefix_hit(hashes))
                lookups += 1
                assert agreed == [expected] * WORLD, (trial, repeat, pattern, expected, agreed)
                if pattern == "missing_shard":
                    assert expected == 0
                probed = sum(len(call) for call in fake_store.exist_calls)
                assert probed <= n * num_groups * WORLD
                for call in fake_store.exist_calls:
                    assert len({re.search(r"/w\d+r(\d+)/", key).group(1) for key in call}) == 1
                if staged and pattern == "missing_shard" and n > worker_module.STAGED_MIN_SPAN:
                    # The empty shard asks only its sentinels.
                    assert probed < n * num_groups * WORLD
        calls = workers[0]._test_reducer.calls
        assert calls == [lookups] * WORLD


def test_rank_local_window_hit_with_lookup_key_switches_matches(
    store_config, fake_store, monkeypatch  # noqa: F811
):
    monkeypatch.setenv(worker_module.LOOKUP_STAGED_ENV, "1")
    monkeypatch.setenv(worker_module.LOOKUP_REUSE_MS_ENV, "60000")
    rng = random.Random("lkp-window")
    with make_ranks(fake_store, monkeypatch, make_window_layout(num_slots=64)) as workers:
        outcomes = set()
        for trial in range(60):
            fake_store.objects.clear()
            # Short candidate lists (one probe round) and long ones (staged: two rounds).
            n = rng.randrange(1, 30) if trial % 2 else rng.randrange(65, 140)
            hashes = long_hashes(n, salt=trial)
            _fill_window(fake_store, workers, hashes, rng)
            first_block = rng.choice([0, 0, rng.randrange(0, n)])
            candidates = hashes[first_block:]
            for _ in range(2):
                with plain_leader_view(workers[0]) as leader:
                    expected = leader.find_prefix_hit(candidates, first_block, TPB)
                agreed = on_all_ranks(
                    workers, lambda r, w: w.find_prefix_hit(candidates, first_block, TPB)
                )
                assert agreed == [expected] * WORLD, (trial, expected, agreed)
            outcomes.add(expected.blocks > 0)
        assert outcomes == {True, False}


# ---- sliding windows: find_prefix_hit ----


def _fill_window(fake_store, workers, hashes, rng):  # noqa: F811
    """Full-attention pages up to a ragged end, and the windows of a few earlier prompt ends, per shard."""
    n = len(hashes)
    fa_end = rng.randrange(0, n + 1)
    ends = sorted(rng.sample(range(1, n + 1), k=min(n, rng.randrange(0, 4))))
    for rank, worker in enumerate(workers):
        lose_fa = rng.random() < 0.2
        for block in range(fa_end - (rng.randrange(0, 3) if lose_fa else 0)):
            put(fake_store, worker, 0, hashes[block])
        for end in ends:
            _, stale_end = worker.stale_blocks(1, end * TPB)
            for block in range(max(0, stale_end - rng.randrange(0, 2)), end):
                if rng.random() < 0.06:
                    continue  # a window page this shard lost
                put(fake_store, worker, 1, hashes[block])


@pytest.mark.parametrize("fallback", [True, False])
@pytest.mark.parametrize("rewind_tokens", [0, TPB, 3 * TPB])
def test_rank_local_window_hit_matches_the_leaders_probe(
    store_config,
    fake_store,
    monkeypatch,
    fallback,
    rewind_tokens,  # noqa: F811
):
    rng = random.Random(f"{fallback}-{rewind_tokens}")
    with make_ranks(fake_store, monkeypatch, make_window_layout(num_slots=64)) as workers:
        for worker in workers:
            worker._window_fallback = fallback
        outcomes = set()
        for trial in range(60):
            fake_store.objects.clear()
            n = rng.randrange(1, 30)
            hashes = hashes_of(n, salt=trial)
            _fill_window(fake_store, workers, hashes, rng)
            first_block = rng.choice([0, 0, rng.randrange(0, n)])
            candidates = hashes[first_block:]
            with leader_view(workers[0]) as leader:
                expected = leader.find_prefix_hit(candidates, first_block, rewind_tokens)
            agreed = on_all_ranks(
                workers,
                lambda r, w: w.find_prefix_hit(candidates, first_block, rewind_tokens),
            )
            assert agreed == [expected] * WORLD, (trial, expected, agreed)
            outcomes.add(expected.blocks > 0)
        # Both hits and misses were exercised.
        assert outcomes == {True, False}


def test_window_hit_with_a_missing_window_shard_is_a_miss(store_config, fake_store, monkeypatch):  # noqa: F811
    with make_ranks(fake_store, monkeypatch, make_window_layout()) as workers:
        hashes = hashes_of(6)
        for rank, worker in enumerate(workers):
            for block_hash in hashes:
                put(fake_store, worker, 0, block_hash)
            if rank != 1:
                _, stale_end = worker.stale_blocks(1, 6 * TPB)
                for block in range(stale_end, 6):
                    put(fake_store, worker, 1, hashes[block])
        with leader_view(workers[0]) as leader:
            assert leader.find_prefix_hit(hashes, 0) == PrefixHit(0)
        assert on_all_ranks(workers, lambda r, w: w.find_prefix_hit(hashes, 0)) == [PrefixHit(0)] * WORLD


# ---- the leader's scheduler with follower ranks ----


def _requests():
    """Requests covering every early return of the leader's lookup, plus hits."""
    long_tokens = list(range(131072 + 5))
    return [
        (make_request(1, list(range(9 * TPB + 1))), 0),
        (make_request(2, list(range(9 * TPB + 1))), 2 * TPB),
        (make_request(3, list(range(9 * TPB + 1))), 2 * TPB - 1),  # partial local block: no lookup
        (make_request(4, list(range(TPB))), 0),  # nothing before the last token's block: no lookup
        (make_request(5, list(range(9 * TPB + 1)), lora_task_id=3), 0),  # no key may name it
        (make_request(6, list(range(9 * TPB + 1)), multimodal_positions=[0]), 0),  # unhashed media
        (make_request(7, list(range(9 * TPB + 1)), cache_salt="tenant-b"), 0),
        (make_request(8, long_tokens), 0),
        (make_request(1, list(range(9 * TPB + 1))), 4 * TPB),  # the same request again, cached chain
    ]


def _store_prompts(fake_store, workers, requests, upto_blocks):  # noqa: F811
    from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.scheduler import _reuse_scope

    for request, _ in requests:
        scope = _reuse_scope(request)
        if scope is None:
            continue
        chain = BlockHashChain(TPB, scope=scope)
        chain.extend(request.get_tokens(0))
        for rank, worker in enumerate(workers):
            for block_hash in chain.hashes[: max(0, upto_blocks(request.request_id) - rank)]:
                for group in worker._namespaces:
                    put(fake_store, worker, group, block_hash)
            if worker.windowed_group_ids:
                end = min(len(chain.hashes), upto_blocks(request.request_id))
                _, stale_end = worker.stale_blocks(1, end * TPB)
                for block in range(stale_end, end):
                    put(fake_store, worker, 1, chain.hashes[block])


@pytest.mark.parametrize("windowed", [False, True])
@pytest.mark.parametrize("entry", ["reserve_prefix", "get_num_new_matched_tokens"])
def test_leader_with_rank_local_followers_offers_what_it_offers_today(
    store_config,
    fake_store,
    monkeypatch,
    windowed,
    entry,  # noqa: F811
):
    layout = make_window_layout(num_slots=64) if windowed else make_layout(num_groups=2)
    requests = _requests()
    with make_ranks(fake_store, monkeypatch, layout, rewind_tokens=TPB) as workers:
        _store_prompts(fake_store, workers, requests, lambda rid: {1: 7, 2: 9, 8: 30000}.get(rid, 5))

        def leader_scheduler(worker):
            llm_args = make_llm_args()
            llm_args.kv_cache_config.block_reuse_config = SimpleNamespace(
                swa_endpoint_rewind_tokens=TPB
            )
            scheduler = MooncakeStoreConnectorScheduler(llm_args)
            scheduler._worker = worker
            return scheduler

        def ask(scheduler, request, local, index):
            if entry == "reserve_prefix":
                return scheduler.reserve_prefix(request, local, index)
            return scheduler.get_num_new_matched_tokens(request, local)

        with leader_view(workers[0]):
            today = leader_scheduler(workers[0])
            expected = [ask(today, request, local, i) for i, (request, local) in enumerate(requests)]
        assert any(count for count, _ in expected) and not all(count for count, _ in expected)

        rank_local = leader_scheduler(workers[0])

        def run(rank, worker):
            answers = []
            for i, (request, local) in enumerate(requests):
                if rank == 0:
                    answers.append(ask(rank_local, request, local, i))
                else:
                    worker.follow_prefix_lookup(request, local)
            return answers

        assert on_all_ranks(workers, run)[0] == expected
        # Every rank joined the same number of agreements (a desync would have timed out first).
        calls = workers[0]._test_reducer.calls
        assert len(set(calls)) == 1 and calls[0] > 0

        def reservations(scheduler):
            return sorted(
                (rid, res_id, r.start, r.end, sorted(r.window_present_from.items()))
                for rid, state in scheduler._requests.items()
                for res_id, r in state.reservations.items()
            )

        assert reservations(rank_local) == reservations(today)
        assert {rid: (s.load_first_block, s.load_blocks) for rid, s in rank_local._requests.items()} == {
            rid: (s.load_first_block, s.load_blocks) for rid, s in today._requests.items()
        }


def test_follower_chain_cache_is_bounded_and_reused(store_config, fake_store, monkeypatch):  # noqa: F811
    monkeypatch.setattr(worker_module, "RANK_LOCAL_CHAIN_CACHE_SIZE", 2)
    with make_ranks(fake_store, monkeypatch, make_layout()) as workers:
        follower = workers[1]
        from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.keys import ReuseScope

        scope = ReuseScope()
        tokens = list(range(5 * TPB))
        first = follower._follower_chain(1, scope, tokens)
        assert follower._follower_chain(1, scope, tokens + [1, 2, 3, 4]) is first
        assert len(first.hashes) == 6
        follower._follower_chain(2, scope, tokens)
        follower._follower_chain(3, scope, tokens)
        assert list(follower._follower_chains) == [2, 3]
        # A different scope, or a shorter token list, rebuilds rather than reuses.
        assert follower._follower_chain(3, ReuseScope(cache_salt="x"), tokens) is not first
        rebuilt = follower._follower_chain(3, ReuseScope(cache_salt="x"), tokens[:TPB])
        assert len(rebuilt.hashes) == 1


# ---- the connector manager ----


def _manager(scheduler, worker, *, adp=False):
    manager = SimpleNamespace(scheduler=scheduler, worker=worker, enable_attention_dp=adp)
    manager._run_on_leader = lambda f, follower=None: kvc.KvCacheConnectorManager._run_on_leader(
        manager, f, follower
    )
    manager._lookup_follower = lambda request, n: kvc.KvCacheConnectorManager._lookup_follower(
        manager, request, n
    )
    return manager


def test_manager_runs_the_follower_only_on_non_leaders_with_the_switch(monkeypatch):
    sent = []
    monkeypatch.setattr(kvc, "mpi_broadcast", lambda obj, root=0: sent.append(obj) or (5, False))
    follows = []
    on = SimpleNamespace(rank_local_lookup=True, follow_prefix_lookup=lambda r, n: follows.append((r, n)))
    off = SimpleNamespace(rank_local_lookup=False, follow_prefix_lookup=lambda r, n: follows.append("off"))
    request = SimpleNamespace(request_id=1)

    follower = _manager(None, on)
    assert follower._run_on_leader(lambda: 1 / 0, follower._lookup_follower(request, 8)) == (5, False)
    assert follows == [(request, 8)] and sent == [None]

    # Switch off, attention DP, or the leader itself: no follower, the path is today's.
    assert _manager(None, off)._lookup_follower(request, 8) is None
    assert _manager(None, on, adp=True)._lookup_follower(request, 8) is None
    assert _manager(object(), on)._lookup_follower(request, 8) is None
    assert _manager(None, SimpleNamespace())._lookup_follower(request, 8) is None
    assert follows == [(request, 8)]
