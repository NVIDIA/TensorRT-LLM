# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ldfix F3: the local stored-key set on the save path (TRTLLM_MOONCAKE_STORE_LOCAL_STORED_SET).

CPU only: the store is the in-process fake from test_mooncake_store_connector.
"""

import pytest
from test_mooncake_store_connector import (  # noqa: F401 (fixtures)
    fake_store,
    make_layout,
    make_worker,
    store_config,
)

from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.metadata import (
    PageTransfer,
    RequestTransfers,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.stored_set import (
    LOCAL_STORED_SET_ENV,
    LocalStoredSet,
)

pytestmark = pytest.mark.cpu_only


# ---- F3: LocalStoredSet ----


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def test_stored_set_remembers_expires_and_forgets():
    clock = Clock()
    stored = LocalStoredSet(capacity=10, ttl=5.0, clock=clock)
    stored.add_many(["a", "b"])
    assert stored.known(["a", "b", "c"]) == [True, True, False]
    clock.now += 4.0
    stored.add_many(["b"])  # refreshed
    clock.now += 2.0
    assert stored.known(["a", "b"]) == [False, True]  # a expired after 6 s, b confirmed 2 s ago
    stored.discard_many(["b", "zzz"])
    assert stored.known(["b"]) == [False]
    assert stored.invalidated == 1 and stored.expired == 1


def test_stored_set_is_bounded_lru():
    stored = LocalStoredSet(capacity=3, ttl=0)  # ttl 0: no expiry
    stored.add_many(["a", "b", "c"])
    stored.add_many(["a"])  # a is now the most recent
    stored.add_many(["d"])  # evicts b, the least recently confirmed
    assert stored.known(["a", "b", "c", "d"]) == [True, False, True, True]
    assert len(stored) == 3


def test_stored_set_is_off_by_default(monkeypatch):
    monkeypatch.delenv(LOCAL_STORED_SET_ENV, raising=False)
    assert LocalStoredSet.from_env() is None
    monkeypatch.setenv(LOCAL_STORED_SET_ENV, "1")
    assert isinstance(LocalStoredSet.from_env(), LocalStoredSet)


def _transfers(hashes, start=0):
    return [RequestTransfers(1, [PageTransfer(h, 0, start + i) for i, h in enumerate(hashes)])]


@pytest.mark.parametrize("enabled", [False, True])
def test_save_skips_the_probe_for_keys_it_already_confirmed(store_config, fake_store, monkeypatch, enabled):
    if enabled:
        monkeypatch.setenv(LOCAL_STORED_SET_ENV, "1")
    else:
        monkeypatch.delenv(LOCAL_STORED_SET_ENV, raising=False)
    with make_worker(fake_store, layout=make_layout()) as worker:
        worker._save_staging = None  # put/probe logic only, no device copy
        hashes = [bytes([i]) * 16 for i in range(4)]
        assert worker._put(_transfers(hashes)) [:2] == (4, 0)
        probes_after_first = len(fake_store.exist_calls)
        # The resave of the same pages: identical outcome either way.
        assert worker._put(_transfers(hashes))[:2] == (0, 4)
        if enabled:
            assert len(fake_store.exist_calls) == probes_after_first  # answered locally
        else:
            assert len(fake_store.exist_calls) == probes_after_first + 1
        # A mix: two known, two new. Only the new ones are probed, and only they are written.
        more = hashes[:2] + [bytes([9]) * 16, bytes([10]) * 16]
        assert worker._put(_transfers(more))[:2] == (2, 2)
        if enabled:
            assert len(fake_store.exist_calls[-1]) == 2
        assert len(fake_store.put_calls[-1][0]) == 2


def test_an_absent_lookup_probe_invalidates_a_confirmed_key(store_config, fake_store, monkeypatch):
    monkeypatch.setenv(LOCAL_STORED_SET_ENV, "1")
    with make_worker(fake_store, layout=make_layout()) as worker:
        worker._save_staging = None
        block_hash = b"\x05" * 16
        key = worker._namespaces[0].key(block_hash)
        worker._put(_transfers([block_hash]))
        assert worker._local_stored.known([key]) == [True]
        # The pool evicts it; the next lookup on this rank sees it absent.
        fake_store.objects.discard(key)
        assert worker.count_prefix_hit([block_hash]) == 0
        assert worker._local_stored.known([key]) == [False]
        # So the next save probes and rewrites it instead of trusting the stale entry.
        written, present, _ = worker._put(_transfers([block_hash]))
        assert (written, present) == (1, 0)
        assert key in fake_store.objects


def test_a_failed_write_is_not_remembered(store_config, fake_store, monkeypatch):
    monkeypatch.setenv(LOCAL_STORED_SET_ENV, "1")
    monkeypatch.setenv("TRTLLM_DISABLE_STORE_SAVE_RETRY", "1")
    monkeypatch.setattr(
        fake_store, "batch_put_from_multi_buffers", lambda keys, *_a, **_k: [-1] * len(keys)
    )
    with make_worker(fake_store, layout=make_layout()) as worker:
        worker._save_staging = None
        block_hash = b"\x06" * 16
        worker._put(_transfers([block_hash]))
        assert worker._local_stored.known([worker._namespaces[0].key(block_hash)]) == [False]
