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
"""CPU concurrency/ownership tests; no assertion of real RDMA or GPU correctness."""

import hashlib
import json
import multiprocessing
import os
import threading
from pathlib import Path

import pytest

from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import retention

pytestmark = pytest.mark.cpu_only

NS = "trtllm/test-retention"


def keys(number):
    digest = hashlib.sha256(str(number).encode()).hexdigest()
    return (
        NS + "/complete/" + digest,
        NS + "/group/3/" + digest,
        NS + "/group/2/prefix/endpoint/" + digest,
    )


class FileStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(exist_ok=True)
        self.fail_key = None
        self.removed = []

    def path(self, key):
        return self.root / hashlib.sha256(key.encode()).hexdigest()

    def put(self, key):
        self.path(key).write_text("payload")

    def exists(self, key):
        return self.path(key).exists()

    def remove(self, key, force):
        assert force is False
        if key == self.fail_key:
            return -1
        self.removed.append(key)
        try:
            self.path(key).unlink()
        except FileNotFoundError:
            return -704
        return 0


def save(catalog, store, number, conversation="conversation", turn=None):
    marker, *private = keys(number)
    lease = catalog.acquire(marker)
    try:
        catalog.prepare_save(lease, marker, tuple(private))
        for key in (*private, marker):
            store.put(key)
        catalog.publish(lease, conversation, turn or str(number))
    finally:
        lease.close(retired=True)


def drain(catalog, store):
    totals = {}
    for _ in range(4):
        for key, value in catalog.collect(store, limit=100).items():
            totals[key] = totals.get(key, 0) + value
    return totals


@pytest.fixture
def setup(tmp_path):
    return retention.CheckpointRetention(str(tmp_path / "catalog"), NS), FileStore(
        tmp_path / "store"
    )


def _owner_save(root, number, ready=None):
    catalog = retention.CheckpointRetention(str(Path(root) / "catalog"), NS)
    save(catalog, FileStore(Path(root) / "store"), number)
    if ready:
        ready.set()


def _crash_pin(root):
    catalog = retention.CheckpointRetention(str(Path(root) / "catalog"), NS)
    catalog.acquire(keys(0)[0])
    os._exit(0)


def test_twelve_distinct_owner_processes_keep_five_globally(setup, tmp_path):
    catalog, store = setup
    context = multiprocessing.get_context("spawn")
    for number in range(12):
        process = context.Process(target=_owner_save, args=(str(tmp_path), number))
        process.start()
        process.join(15)
        assert process.exitcode == 0
    stats = drain(catalog, store)
    assert stats["gc_deleted_checkpoints"] == 7
    for number in range(12):
        assert all(store.exists(key) == (number >= 7) for key in keys(number))


def test_multiple_snapshots_of_one_turn_are_one_retention_unit(setup):
    catalog, store = setup
    save(catalog, store, 0, turn="first")
    save(catalog, store, 1, turn="first")
    for number in range(2, 6):
        save(catalog, store, number)
    drain(catalog, store)
    assert all(store.exists(key) for key in keys(0) + keys(1))
    save(catalog, store, 6)
    drain(catalog, store)
    assert all(not store.exists(key) for key in keys(0) + keys(1))


def test_shared_checkpoint_survives_until_both_conversations_retire(setup):
    catalog, store = setup
    save(catalog, store, 0, "A")
    save(catalog, store, 0, "B")
    for number in range(1, 6):
        save(catalog, store, number, "A")
    drain(catalog, store)
    assert all(store.exists(key) for key in keys(0))
    for number in range(6, 11):
        save(catalog, store, number, "B")
    drain(catalog, store)
    assert all(not store.exists(key) for key in keys(0))


def test_shared_attention_prefix_is_never_deleted(setup):
    catalog, store = setup
    prefix = NS + "/group/0/shared-prefix"
    store.put(prefix)
    for number in range(6):
        save(catalog, store, number)
    drain(catalog, store)
    assert store.exists(prefix)
    assert prefix not in store.removed


def test_inflight_get_delays_gc_until_retirement(setup):
    catalog, store = setup
    save(catalog, store, 0)
    active = catalog.acquire(keys(0)[0])
    for number in range(1, 6):
        save(catalog, store, number)
    stats = drain(catalog, store)
    assert stats["gc_deferred_active"] > 0
    assert store.exists(keys(0)[0])
    active.close(retired=True)
    drain(catalog, store)
    assert not store.exists(keys(0)[0])


def test_uncertain_completion_and_process_death_leave_durable_pins(setup, tmp_path):
    catalog, store = setup
    save(catalog, store, 0)
    process = multiprocessing.get_context("spawn").Process(target=_crash_pin, args=(str(tmp_path),))
    process.start()
    process.join(15)
    assert process.exitcode == 0
    for number in range(1, 6):
        save(catalog, store, number)
    assert drain(catalog, store)["gc_deferred_active"] > 0
    assert store.exists(keys(0)[0])


def test_republish_while_old_gc_is_pending_does_not_delete_new_publication(setup):
    catalog, store = setup
    for number in range(6):
        save(catalog, store, number)
    save(catalog, store, 0, turn="new-publication")
    drain(catalog, store)
    assert all(store.exists(key) for key in keys(0))
    assert not store.exists(keys(1)[0])


def test_partial_deletion_quarantines_endpoint_and_removes_marker_first(setup):
    catalog, store = setup
    for number in range(6):
        save(catalog, store, number)
    marker, recurrent, _ = keys(0)
    store.fail_key = recurrent
    assert drain(catalog, store)["gc_errors"] > 0
    assert not store.exists(marker)
    assert store.exists(recurrent)
    assert store.removed[0] == marker
    store.fail_key = None
    assert drain(catalog, store)["gc_deleted_checkpoints"] == 0
    with pytest.raises(retention.RetentionError):
        catalog.acquire(marker)


def test_unscoped_publication_conservatively_protects_shared_endpoint(setup):
    catalog, store = setup
    save(catalog, store, 0, "")
    for number in range(6):
        save(catalog, store, number)
    drain(catalog, store)
    assert store.exists(keys(0)[0])


def test_manifest_mismatch_and_cross_namespace_rejected(setup):
    catalog, _ = setup
    marker, *private = keys(0)
    with pytest.raises(retention.RetentionError):
        catalog.acquire("other/" + marker)
    lease = catalog.acquire(marker)
    try:
        catalog.prepare_save(lease, marker, tuple(private))
        with pytest.raises(retention.RetentionError):
            catalog.prepare_save(lease, marker, tuple(private[:1]))
    finally:
        lease.close(retired=True)


def test_policy_mismatch_rejected(setup, tmp_path):
    with pytest.raises(retention.RetentionError):
        retention.CheckpointRetention(str(tmp_path / "catalog"), NS, 4)


def test_crash_between_new_reference_and_ledger_only_leaks(setup, monkeypatch):
    catalog, store = setup
    for number in range(5):
        save(catalog, store, number)
    original = retention._atomic

    def fail_ledger(path, value):
        if path.parent.name == "conversations":
            raise OSError("simulated crash before ledger commit")
        original(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(retention, "_atomic", fail_ledger)
        with pytest.raises(OSError):
            save(catalog, store, 5)
    drain(catalog, store)
    assert all(store.exists(keys(number)[0]) for number in range(6))


def test_many_concurrent_publications_still_keep_five(setup, tmp_path):
    catalog, store = setup
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_owner_save, args=(str(tmp_path), i)) for i in range(12)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(20)
        assert process.exitcode == 0
    drain(catalog, store)
    assert sum(store.exists(keys(i)[0]) for i in range(12)) == 5
    ledger = json.loads(next((catalog.root / "conversations").glob("*.json")).read_text())
    assert len(ledger) == 5


def test_gc_exclusive_lock_blocks_republication_during_delete(setup):
    catalog, store = setup
    for number in range(6):
        save(catalog, store, number)
    entered, release, republished = threading.Event(), threading.Event(), threading.Event()
    remove = store.remove

    def blocked_remove(key, force):
        if key == keys(0)[0]:
            entered.set()
            assert release.wait(5)
        return remove(key, force)

    store.remove = blocked_remove
    collector = threading.Thread(target=drain, args=(catalog, store))
    collector.start()
    assert entered.wait(5)

    other = retention.CheckpointRetention(str(catalog.root.parent), NS)
    with pytest.raises(retention.EndpointUnavailable):
        save(other, store, 0, turn="republished")
    assert not republished.is_set()
    release.set()
    collector.join(5)
    save(other, store, 0, turn="republished")
    assert all(store.exists(key) for key in keys(0))


@pytest.mark.parametrize("marker", ["a" * 64, "other/complete/" + "a" * 64])
def test_bare_digest_and_foreign_marker_cannot_acquire_endpoint(setup, marker):
    catalog, _ = setup
    with pytest.raises(retention.RetentionError):
        catalog.acquire(marker)


@pytest.mark.parametrize(
    "payload",
    [
        NS + "/model/w1r0/lg0/t64b128/" + "a" * 32,
        NS + "/group/0/shared-prefix",
        "other/group/0/" + "a" * 64,
    ],
)
def test_shared_or_foreign_page_cannot_enter_reclaim_manifest(setup, payload):
    catalog, store = setup
    marker = keys(0)[0]
    store.put(payload)
    lease = catalog.acquire(marker)
    try:
        with pytest.raises(retention.RetentionError):
            catalog.prepare_save(lease, marker, (payload,))
    finally:
        lease.close(retired=True)
    drain(catalog, store)
    assert store.exists(payload)
    assert payload not in store.removed
