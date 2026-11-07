# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Signed records in the etcd disagg cluster storage."""

import asyncio
import json
import secrets
from types import SimpleNamespace
from unittest import mock

import pytest

from tensorrt_llm.serve.cluster_storage import (
    Etcd3ClusterStorage,
    StorageItem,
    WatchEvent,
    WatchEventType,
)

pytestmark = pytest.mark.cpu_only

RECORD_KEY = secrets.token_hex(16)
OTHER_RECORD_KEY = secrets.token_hex(16)
PREFIX = "/trtllm-disagg/c/workers"


class FakeEtcdClient:
    """The calls of the etcd client that Etcd3ClusterStorage makes, on a dict."""

    def __init__(self):
        self.data = {}
        self.callbacks = {}

    def put(self, key, value, lease=None):
        self.data[key] = value

    def put_if_not_exists(self, key, value, lease=None):
        if key in self.data:
            return False
        self.data[key] = value
        return True

    def get(self, key):
        value = self.data.get(key)
        return (None if value is None else value.encode()), None

    def get_prefix(self, prefix):
        return [
            (value.encode(), SimpleNamespace(key=key.encode()))
            for key, value in self.data.items()
            if key.startswith(prefix)
        ]

    def add_watch_prefix_callback(self, prefix, callback):
        self.callbacks[prefix] = callback
        return prefix

    def cancel_watch(self, watch_id):
        self.callbacks.pop(watch_id, None)

    def close(self):
        pass


class PutEvent:
    def __init__(self, key, value):
        self.key, self.value = key.encode(), value.encode()


class DeleteEvent(PutEvent):
    pass


def _storage(record_auth_key=RECORD_KEY, client=None):
    client = client or FakeEtcdClient()
    with mock.patch.object(Etcd3ClusterStorage, "_connect", return_value=client):
        return client, Etcd3ClusterStorage(
            "etcd://etcd-host:2379", "c", record_auth_key=record_auth_key
        )


@pytest.mark.asyncio
async def test_signed_round_trip():
    client, storage = _storage()
    assert await storage.set(f"{PREFIX}/a", "value-a")
    assert not await storage.set(f"{PREFIX}/a", "other")
    stored = json.loads(client.data[f"{PREFIX}/a"])
    assert stored["value"] == "value-a"
    assert stored["signature"].startswith("sha256=")
    assert await storage.get(f"{PREFIX}/a") == "value-a"
    assert await storage.get(f"{PREFIX}/missing") is None
    assert await storage.get_prefix(PREFIX) == {f"{PREFIX}/a": "value-a"}
    assert await storage.get_prefix(PREFIX, keys_only=True) == {f"{PREFIX}/a": ""}


@pytest.mark.asyncio
async def test_records_without_a_valid_signature_are_ignored():
    client, storage = _storage()
    good = f"{PREFIX}/good"
    await storage.set(good, "good")
    await _storage(OTHER_RECORD_KEY, client)[1].set(f"{PREFIX}/other-key", "x")
    edited = json.loads(client.data[good])
    edited["value"] = "edited"
    client.data[f"{PREFIX}/edited"] = json.dumps(edited)
    client.data[f"{PREFIX}/copy"] = client.data[good]
    client.data[f"{PREFIX}/not-json"] = "plain"
    client.data[f"{PREFIX}/no-signature"] = json.dumps({"value": "x"})
    client.data[f"{PREFIX}/not-an-object"] = json.dumps(["x"])

    assert await storage.get(good) == "good"
    for key in client.data:
        if key != good:
            assert await storage.get(key) is None, key
    assert await storage.get_prefix(PREFIX) == {good: "good"}


@pytest.mark.asyncio
async def test_watch_delivers_only_valid_records():
    client, storage = _storage()
    await storage.set(f"{PREFIX}/a", "value-a")
    handle = await storage.watch(PREFIX)
    client.callbacks[PREFIX](
        SimpleNamespace(
            events=[
                PutEvent(f"{PREFIX}/plain", "plain"),
                PutEvent(f"{PREFIX}/a", client.data[f"{PREFIX}/a"]),
                DeleteEvent(f"{PREFIX}/gone", ""),
            ]
        )
    )
    events = await asyncio.wait_for(handle.drain(), 1)
    assert [(e.event_type, e.storage_item.key, e.storage_item.value) for e in events] == [
        (WatchEventType.SET, f"{PREFIX}/a", "value-a"),
        (WatchEventType.DELETE, f"{PREFIX}/gone", ""),
    ]


@pytest.mark.asyncio
async def test_events_added_by_the_caller_are_not_verified():
    # The cluster manager puts the records it has read into the queue before draining it.
    _, storage = _storage()
    handle = await storage.watch(PREFIX)
    await handle.add_events(
        [
            WatchEvent(
                storage_item=StorageItem(key=f"{PREFIX}/a", value="value-a"),
                event_type=WatchEventType.SET,
            )
        ]
    )
    events = await asyncio.wait_for(handle.drain(), 1)
    assert [e.storage_item.value for e in events] == ["value-a"]


@pytest.mark.asyncio
async def test_without_a_key_records_are_stored_as_given():
    client, storage = _storage(record_auth_key=None)
    await storage.set(f"{PREFIX}/a", "value-a")
    assert client.data == {f"{PREFIX}/a": "value-a"}
    assert await storage.get(f"{PREFIX}/a") == "value-a"
    assert await storage.get_prefix(PREFIX) == {f"{PREFIX}/a": "value-a"}
    handle = await storage.watch(PREFIX)
    client.callbacks[PREFIX](SimpleNamespace(events=[PutEvent(f"{PREFIX}/b", "plain")]))
    events = await asyncio.wait_for(handle.drain(), 1)
    assert [e.storage_item.value for e in events] == ["plain"]
