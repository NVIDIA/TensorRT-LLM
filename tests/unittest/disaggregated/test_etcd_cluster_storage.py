# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Etcd3ClusterStorage on a fake etcd client: connecting, logging in and signed records."""

import asyncio
import json
import secrets
import threading
from types import SimpleNamespace
from unittest import mock

import grpc
import pytest

from tensorrt_llm.serve import cluster_storage
from tensorrt_llm.serve.cluster_storage import (
    Etcd3ClusterStorage,
    StorageItem,
    WatchEvent,
    WatchEventType,
)
from tensorrt_llm.serve.record_signing import sign_record

pytestmark = pytest.mark.cpu_only

ETCD_USER = secrets.token_hex(4)
ETCD_SECRET = secrets.token_hex(16)
ETCD_URI = "etcd://etcd-host:2379"
LOGIN = {"user": ETCD_USER, "password": ETCD_SECRET, "ca_cert": "/ca.pem"}
RECORD_KEY = secrets.token_hex(16)
OTHER_RECORD_KEY = secrets.token_hex(16)
PREFIX = "/trtllm-disagg/c/workers"


class _RpcError(grpc.RpcError):
    def __init__(self, code, details=""):
        super().__init__()
        self._code, self._details = code, details

    def code(self):
        return self._code

    def details(self):
        return self._details


def _connect(*clients):
    return mock.patch.object(Etcd3ClusterStorage, "_connect", side_effect=list(clients))


@pytest.mark.asyncio
async def test_etcd_storage_without_options_connects_as_before():
    with (
        mock.patch.object(cluster_storage.etcd3, "Endpoint") as endpoint,
        mock.patch.object(cluster_storage.etcd3, "MultiEndpointEtcd3Client") as client,
    ):
        client.return_value.get.return_value = (None, None)
        storage = Etcd3ClusterStorage("etcd://h1:2379,etcd://h2:2380", "c")
        client.assert_not_called()
        await storage.get("k")
    assert endpoint.call_args_list == [
        mock.call("h1", 2379, secure=False, creds=None),
        mock.call("h2", 2380, secure=False, creds=None),
    ]
    client.assert_called_once_with(
        [endpoint.return_value, endpoint.return_value], timeout=10, user=None, password=None
    )


@pytest.mark.asyncio
async def test_etcd_storage_passes_credentials_and_tls_files_and_keeps_them_for_reconnect():
    with (
        mock.patch.object(cluster_storage.etcd3.Etcd3Client, "get_secure_creds") as secure_creds,
        mock.patch.object(cluster_storage.etcd3, "Endpoint") as endpoint,
        mock.patch.object(cluster_storage.etcd3, "MultiEndpointEtcd3Client") as client,
    ):
        client.return_value.get.return_value = (None, None)
        storage = Etcd3ClusterStorage(
            ETCD_URI, "c", **LOGIN, cert_key="/key.pem", cert_cert="/cert.pem"
        )
        await storage.get("k")
        storage.reconnect()
        await storage.get("k")
    secure_creds.assert_called_with("/ca.pem", "/key.pem", "/cert.pem")
    endpoint.assert_called_with("etcd-host", 2379, secure=True, creds=secure_creds.return_value)
    assert client.call_count == 2
    client.assert_called_with(
        [endpoint.return_value], timeout=10, user=ETCD_USER, password=ETCD_SECRET
    )
    client.return_value.close.assert_called_once()


@pytest.mark.parametrize(
    "create",
    [cluster_storage.create_cluster_storage, cluster_storage.create_cluster_storage_client],
)
def test_storage_with_an_unreadable_tls_file_is_not_created(create, tmp_path):
    files = {}
    for name in ("ca_cert", "cert_key", "cert_cert"):
        files[name] = str(tmp_path / f"{name}.pem")
        (tmp_path / f"{name}.pem").write_text("pem")
    with mock.patch.object(Etcd3ClusterStorage, "_connect") as connect:
        assert isinstance(create(ETCD_URI, "c", **files), Etcd3ClusterStorage)
    connect.assert_not_called()
    for name in files:
        for path in (str(tmp_path / "missing.pem"), str(tmp_path)):
            with pytest.raises(ValueError) as excinfo:
                create(ETCD_URI, "c", **{**files, name: path})
            assert str(excinfo.value) == f"disagg_cluster {name} is not a readable file: {path!r}"


@pytest.mark.asyncio
async def test_connect_warning_names_the_file_that_cannot_be_read():
    missing = FileNotFoundError(2, "No such file or directory", "/tls/ca.pem")
    with _connect(missing), mock.patch.object(cluster_storage.logger, "warning") as warning:
        storage = Etcd3ClusterStorage(ETCD_URI, "c", **LOGIN)
        assert await storage.set("k", "v") is False
    (message,), _ = warning.call_args
    assert message.endswith("FileNotFoundError: [Errno 2] No such file or directory: '/tls/ca.pem'")


@pytest.mark.asyncio
async def test_login_runs_on_the_first_call_off_the_event_loop():
    loop_thread = threading.get_ident()
    login_threads = []

    def connect(*args, **kwargs):
        login_threads.append(threading.get_ident())
        client = mock.MagicMock()
        client.get.return_value = (None, None)
        return client

    with mock.patch.object(Etcd3ClusterStorage, "_connect", side_effect=connect):
        storage = Etcd3ClusterStorage(ETCD_URI, "c", **LOGIN)
        assert login_threads == []
        await storage.get("k")
        await storage.get("k")
    assert len(login_threads) == 1 and login_threads[0] != loop_thread


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        _RpcError(grpc.StatusCode.UNAVAILABLE),
        cluster_storage.etcd3.exceptions.ConnectionFailedError(),
        FileNotFoundError("/ca.pem"),
    ],
)
async def test_failed_login_returns_the_error_value_and_is_tried_again(error):
    client = mock.MagicMock()
    with _connect(error, client) as connect:
        storage = Etcd3ClusterStorage(ETCD_URI, "c", **LOGIN)
        assert await storage.set("k", "v") is False
        assert await storage.set("k", "v", overwrite_if_exists=True) is True
    assert connect.call_count == 2
    client.put.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code, details",
    [
        (grpc.StatusCode.UNAUTHENTICATED, "etcdserver: invalid auth token"),
        (grpc.StatusCode.INVALID_ARGUMENT, "etcdserver: revision of auth store is old"),
        (grpc.StatusCode.INVALID_ARGUMENT, "etcdserver: user name is empty"),
    ],
)
async def test_rejected_token_logs_in_again(code, details):
    stale, fresh = mock.MagicMock(), mock.MagicMock()
    stale.get.side_effect = _RpcError(code, details)
    fresh.get.return_value = (b"value", None)
    with _connect(stale, fresh) as connect:
        storage = Etcd3ClusterStorage(ETCD_URI, "c", **LOGIN)
        assert await storage.get("k") is None
        assert await storage.get("k") == "value"
    assert connect.call_count == 2
    stale.close.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "login, code, details",
    [
        (LOGIN, grpc.StatusCode.PERMISSION_DENIED, "etcdserver: permission denied"),
        (LOGIN, grpc.StatusCode.INVALID_ARGUMENT, "etcdserver: key is not provided"),
        ({}, grpc.StatusCode.UNAUTHENTICATED, "etcdserver: invalid auth token"),
    ],
)
async def test_other_errors_are_raised_without_logging_in_again(login, code, details):
    client = mock.MagicMock()
    client.get.side_effect = _RpcError(code, details)
    with _connect(client) as connect:
        storage = Etcd3ClusterStorage(ETCD_URI, "c", **login)
        with pytest.raises(grpc.RpcError):
            await storage.get("k")
    assert connect.call_count == 1
    client.close.assert_not_called()


def _bytes(data):
    return data if isinstance(data, bytes) else data.encode()


class FakeEtcdClient:
    """The calls of the etcd client that Etcd3ClusterStorage makes, on a dict."""

    def __init__(self, data=None):
        self.data = {} if data is None else data
        # etcd revision, counted by the writes made through this client.
        self.revision = 0
        self.callbacks = {}
        # start_revision of each watch created, in order.
        self.start_revisions = []
        self.cancelled = []

    def put(self, key, value, lease=None):
        self.data[key] = value
        self.revision += 1

    def put_if_not_exists(self, key, value, lease=None):
        if key in self.data:
            return False
        self.put(key, value)
        return True

    def get(self, key):
        value = self.data.get(key)
        return (None if value is None else _bytes(value)), None

    def get_prefix_response(self, prefix):
        return SimpleNamespace(
            header=SimpleNamespace(revision=self.revision),
            kvs=[
                SimpleNamespace(key=_bytes(key), value=_bytes(value))
                for key, value in self.data.items()
                if _bytes(key).startswith(prefix.encode())
            ],
        )

    def add_watch_prefix_callback(self, prefix, callback, start_revision=None):
        self.callbacks[prefix] = callback
        self.start_revisions.append(start_revision)
        return prefix

    def cancel_watch(self, watch_id):
        self.cancelled.append(watch_id)
        self.callbacks.pop(watch_id, None)

    def close(self):
        # Closing the channel ends the watch streams, as with the real client.
        callbacks, self.callbacks = self.callbacks, {}
        for callback in callbacks.values():
            callback(None)

    def send(self, prefix, *events):
        self.callbacks[prefix](SimpleNamespace(events=list(events)))


class PutEvent:
    def __init__(self, key, value):
        self.key, self.value = _bytes(key), _bytes(value)


class DeleteEvent(PutEvent):
    pass


def _storage(record_auth_key=RECORD_KEY, client=None):
    client = client or FakeEtcdClient()
    storage = Etcd3ClusterStorage("etcd://etcd-host:2379", "c", record_auth_key=record_auth_key)
    storage._connect = lambda *args, **kwargs: client
    return client, storage


def test_signature_format_is_stable():
    # Records signed by one release must still verify on the next one.
    assert sign_record("test-record-auth-key", f"{PREFIX}/a", "value-a") == (
        "sha256=0f22fee56aca4beb74fd61e86842a93a8b1eff74ba82420e0a58e8e6f5f42479"
    )


@pytest.mark.asyncio
async def test_signed_round_trip():
    client, storage = _storage()
    assert await storage.set(f"{PREFIX}/a", "value-a")
    assert not await storage.set(f"{PREFIX}/a", "other")
    stored = json.loads(client.data[f"{PREFIX}/a"])
    assert stored == {
        "value": "value-a",
        "signature": sign_record(RECORD_KEY, f"{PREFIX}/a", "value-a"),
    }
    assert await storage.get(f"{PREFIX}/a") == "value-a"
    assert await storage.get(f"{PREFIX}/missing") is None
    assert await storage.get_prefix(PREFIX) == {f"{PREFIX}/a": "value-a"}
    assert await storage.get_prefix(PREFIX, keys_only=True) == {f"{PREFIX}/a": ""}


@pytest.mark.asyncio
async def test_a_record_changed_in_place_is_ignored():
    client, storage = _storage()
    key = f"{PREFIX}/a"
    await storage.set(key, "value-a")
    record = json.loads(client.data[key])
    record["value"] = "edited"
    client.data[key] = json.dumps(record)
    assert await storage.get(key) is None
    assert await storage.get_prefix(PREFIX) == {}


@pytest.mark.asyncio
async def test_records_without_a_valid_signature_are_ignored():
    client, storage = _storage()
    good = f"{PREFIX}/good"
    await storage.set(good, "good")
    await _storage(OTHER_RECORD_KEY, client)[1].set(f"{PREFIX}/other-key", "x")
    client.data[f"{PREFIX}/copy"] = client.data[good]
    client.data[f"{PREFIX}/not-json"] = "plain"
    client.data[f"{PREFIX}/no-signature"] = json.dumps({"value": "x"})
    client.data[f"{PREFIX}/not-an-object"] = json.dumps(["x"])
    client.data[f"{PREFIX}/not-utf-8"] = b"\xff"

    assert await storage.get(good) == "good"
    for key in client.data:
        if key != good:
            assert await storage.get(key) is None, key
    assert await storage.get_prefix(PREFIX) == {good: "good"}


@pytest.mark.asyncio
@pytest.mark.parametrize("record_auth_key", [RECORD_KEY, None])
async def test_undecodable_records_do_not_hide_valid_ones(record_auth_key):
    client, storage = _storage(record_auth_key)
    await storage.set(f"{PREFIX}/a", "value-a")
    client.data[f"{PREFIX}/bad-value"] = b"\xff\xfe"
    client.data[f"{PREFIX}/bad-key-\xff".encode("latin-1")] = client.data[f"{PREFIX}/a"]
    assert await storage.get_prefix(PREFIX) == {f"{PREFIX}/a": "value-a"}

    handle = await storage.watch(PREFIX)
    client.callbacks[PREFIX](
        SimpleNamespace(
            events=[
                PutEvent(f"{PREFIX}/bad-value", b"\xff\xfe"),
                PutEvent(f"{PREFIX}/bad-key-\xff".encode("latin-1"), client.data[f"{PREFIX}/a"]),
                PutEvent(f"{PREFIX}/a", client.data[f"{PREFIX}/a"]),
            ]
        )
    )
    events = await asyncio.wait_for(handle.drain(), 1)
    assert [(e.storage_item.key, e.storage_item.value) for e in events] == [
        (f"{PREFIX}/a", "value-a")
    ]
    assert PREFIX in client.callbacks  # the watch is still active


@pytest.mark.asyncio
async def test_ignored_records_are_logged_with_the_key_escaped():
    client, storage = _storage()
    key = f"{PREFIX}/a\nfake log line"
    client.data[key] = "plain"
    with mock.patch.object(cluster_storage.logger, "warning_once") as warning_once:
        assert await storage.get(key) is None
    (message,), kwargs = warning_once.call_args
    assert repr(key) in message and "\n" not in message
    assert "record_auth_key" in message and kwargs == {"key": message}


def _changes(events):
    return [(e.event_type.name, e.storage_item.key, e.storage_item.value) for e in events]


@pytest.mark.asyncio
async def test_watch_delivers_only_valid_records():
    client, storage = _storage()
    await storage.set(f"{PREFIX}/a", "value-a")
    await storage.set(f"{PREFIX}/b", "value-b")
    handle = await storage.watch(PREFIX)
    client.send(
        PREFIX,
        PutEvent(f"{PREFIX}/plain", "plain"),
        DeleteEvent(f"{PREFIX}/plain", ""),
        PutEvent(f"{PREFIX}/a", client.data[f"{PREFIX}/a"]),
        PutEvent(f"{PREFIX}/a", client.data[f"{PREFIX}/a"]),
        DeleteEvent(f"{PREFIX}/never-seen", ""),
        PutEvent(f"{PREFIX}/b", client.data[f"{PREFIX}/b"]),
        DeleteEvent(f"{PREFIX}/a", ""),
    )
    events = await asyncio.wait_for(handle.drain(), 1)
    # Removals of records that were never queued, and repeated values, are not queued.
    assert _changes(events) == [
        ("SET", f"{PREFIX}/a", "value-a"),
        ("SET", f"{PREFIX}/b", "value-b"),
        ("DELETE", f"{PREFIX}/a", ""),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream_end", [None, _RpcError(grpc.StatusCode.UNAVAILABLE)])
async def test_watch_continues_on_a_new_stream_with_the_changes_made_meanwhile(stream_end):
    client, storage = _storage()
    for name in "abc":
        await storage.set(f"{PREFIX}/{name}", f"value-{name}")
    handle = await storage.watch(PREFIX)
    first_stream = client.callbacks[PREFIX]
    client.send(PREFIX, *(PutEvent(f"{PREFIX}/{n}", client.data[f"{PREFIX}/{n}"]) for n in "abc"))
    assert len(await asyncio.wait_for(handle.drain(), 1)) == 3

    # While the stream is down: b is removed, c changes, d is added (for example, b's worker
    # came back on the same address under the new key d).
    del client.data[f"{PREFIX}/b"]
    await storage.set(f"{PREFIX}/c", "value-c2", overwrite_if_exists=True)
    await storage.set(f"{PREFIX}/d", "value-d")
    first_stream(stream_end)
    await asyncio.sleep(0)
    await asyncio.wait_for(handle.restart_task, 1)

    # Removals come first, so a consumer that keys workers by address drops b before adding d.
    assert _changes(await asyncio.wait_for(handle.drain(), 1)) == [
        ("DELETE", f"{PREFIX}/b", ""),
        ("SET", f"{PREFIX}/c", "value-c2"),
        ("SET", f"{PREFIX}/d", "value-d"),
    ]
    # The new stream starts after the read, which saw revision 5.
    assert client.start_revisions == [None, 6]
    # Events of the ended stream are ignored; the new stream fills the queue.
    late = {"value": "late", "signature": sign_record(RECORD_KEY, f"{PREFIX}/e", "late")}
    first_stream(SimpleNamespace(events=[PutEvent(f"{PREFIX}/e", json.dumps(late))]))
    client.send(PREFIX, DeleteEvent(f"{PREFIX}/d", ""))
    assert _changes(await asyncio.wait_for(handle.drain(), 1)) == [("DELETE", f"{PREFIX}/d", "")]


@pytest.mark.asyncio
async def test_rejected_token_moves_an_active_watch_to_the_new_login():
    stale = FakeEtcdClient()
    fresh = FakeEtcdClient(stale.data)
    stale.data[f"{PREFIX}/a"] = "value-a"
    with _connect(stale, fresh) as connect:
        storage = Etcd3ClusterStorage(ETCD_URI, "c", **LOGIN)
        handle = await storage.watch(PREFIX)
        stale.get = mock.Mock(side_effect=_RpcError(grpc.StatusCode.UNAUTHENTICATED))
        assert await storage.get(f"{PREFIX}/a") is None
        await asyncio.sleep(0)
        await asyncio.wait_for(handle.restart_task, 1)
    assert connect.call_count == 2
    assert stale.cancelled == [PREFIX] and PREFIX in fresh.callbacks
    assert _changes(await asyncio.wait_for(handle.drain(), 1)) == [
        ("SET", f"{PREFIX}/a", "value-a")
    ]
    fresh.send(PREFIX, PutEvent(f"{PREFIX}/b", "value-b"))
    assert _changes(await asyncio.wait_for(handle.drain(), 1)) == [
        ("SET", f"{PREFIX}/b", "value-b")
    ]
    await storage.unwatch(PREFIX)
    assert fresh.cancelled == [PREFIX] and fresh.callbacks == {}


@pytest.mark.asyncio
async def test_watch_is_tried_again_until_etcd_answers(monkeypatch):
    monkeypatch.setattr(cluster_storage, "_WATCH_RETRY_MIN_SEC", 0)
    first, second = FakeEtcdClient(), FakeEtcdClient()
    # etcd goes down: the stream ends, watching on the old client fails, and so does
    # the next login; the login after that succeeds.
    with _connect(first, _RpcError(grpc.StatusCode.UNAVAILABLE), second) as connect:
        storage = Etcd3ClusterStorage(ETCD_URI, "c", **LOGIN)
        handle = await storage.watch(PREFIX)
        first.add_watch_prefix_callback = mock.Mock(
            side_effect=cluster_storage.etcd3.exceptions.ConnectionFailedError()
        )
        first.callbacks[PREFIX](_RpcError(grpc.StatusCode.UNAVAILABLE))
        await asyncio.sleep(0)
        await asyncio.wait_for(handle.restart_task, 1)
    assert connect.call_count == 3 and PREFIX in second.callbacks


class _OnWatch(FakeEtcdClient):
    """A client that calls ``hooks[n]`` with the callback of its n-th watch as it creates it."""

    def __init__(self, hooks):
        super().__init__()
        self.hooks = hooks

    def add_watch_prefix_callback(self, prefix, callback, start_revision=None):
        watch_id = super().add_watch_prefix_callback(prefix, callback, start_revision)
        hook = self.hooks.get(len(self.start_revisions))
        if hook:
            hook(callback)
        return watch_id


@pytest.mark.asyncio
async def test_watch_whose_stream_ends_before_watch_returns_is_watched_again():
    # The first stream ends while the watch is being created.
    client, storage = _storage(client=_OnWatch({1: lambda callback: callback(None)}))
    await storage.set(f"{PREFIX}/a", "value-a")
    handle = await storage.watch(PREFIX)
    assert handle.restart_task is not None
    await asyncio.wait_for(handle.restart_task, 1)
    assert client.start_revisions == [None, 2]
    assert _changes(await asyncio.wait_for(handle.drain(), 1)) == [
        ("SET", f"{PREFIX}/a", "value-a")
    ]


@pytest.mark.asyncio
async def test_watch_that_etcd_did_not_start_for_the_login_token_is_created_again(monkeypatch):
    monkeypatch.setattr(cluster_storage, "_WATCH_RETRY_MIN_SEC", 0)

    def reject_login(callback):
        # etcd stops accepting the login as the new watch is created, and answers the watch as
        # if it had started it; reads with that login fail.
        stale.get = stale.get_prefix_response = mock.Mock(
            side_effect=_RpcError(
                grpc.StatusCode.INVALID_ARGUMENT, "etcdserver: revision of auth store is old"
            )
        )

    stale, fresh = _OnWatch({2: reject_login}), FakeEtcdClient()
    with _connect(stale, fresh) as connect:
        storage = Etcd3ClusterStorage(ETCD_URI, "c", **LOGIN)
        handle = await storage.watch(PREFIX)
        stale.callbacks[PREFIX](None)
        await asyncio.sleep(0)
        await asyncio.wait_for(handle.restart_task, 1)
    assert connect.call_count == 2 and PREFIX in fresh.callbacks


@pytest.mark.asyncio
async def test_changes_reported_by_the_new_stream_are_queued_after_the_resync():
    old, new = f"{PREFIX}/old", f"{PREFIX}/new"

    def worker_comes_back(callback):
        # The new stream reports a worker that came back on the address of the old one.
        record = {"value": "host:8001", "signature": sign_record(RECORD_KEY, new, "host:8001")}
        client.put(new, json.dumps(record))
        callback(SimpleNamespace(events=[PutEvent(new, client.data[new])]))

    client, storage = _storage(client=_OnWatch({2: worker_comes_back}))
    await storage.set(old, "host:8001")
    handle = await storage.watch(PREFIX)
    client.send(PREFIX, PutEvent(old, client.data[old]))
    assert len(await asyncio.wait_for(handle.drain(), 1)) == 1

    del client.data[old]  # while the stream is down
    client.callbacks[PREFIX](None)
    await asyncio.sleep(0)
    await asyncio.wait_for(handle.restart_task, 1)
    # A consumer that keys workers by address removes the old worker before adding the new.
    assert _changes(await asyncio.wait_for(handle.drain(), 1)) == [
        ("DELETE", old, ""),
        ("SET", new, "host:8001"),
    ]


@pytest.mark.asyncio
async def test_unwatched_prefix_is_not_watched_again():
    client, storage = _storage()
    handle = await storage.watch(PREFIX)
    client.callbacks[PREFIX](None)  # the stream ends just before the unwatch
    await storage.unwatch(PREFIX)
    await asyncio.sleep(0)
    assert handle.restart_task is None and client.callbacks == {}


@pytest.mark.asyncio
async def test_stream_started_after_an_unwatch_is_cancelled():
    def unwatch(callback):
        # The prefix is unwatched while the new stream starts.
        asyncio.run_coroutine_threadsafe(storage.unwatch(PREFIX), loop).result(5)

    loop = asyncio.get_running_loop()
    client, storage = _storage(client=_OnWatch({2: unwatch}))
    handle = await storage.watch(PREFIX)
    client.callbacks[PREFIX](None)
    await asyncio.sleep(0)
    await asyncio.wait_for(handle.restart_task, 1)
    assert len(client.start_revisions) == 2 and client.callbacks == {}


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
