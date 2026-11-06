# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How Etcd3ClusterStorage connects to etcd: credentials, TLS files and reconnecting."""

import secrets
import threading
from unittest import mock

import grpc
import pytest

from tensorrt_llm.serve import cluster_storage
from tensorrt_llm.serve.cluster_storage import Etcd3ClusterStorage

pytestmark = pytest.mark.cpu_only

ETCD_USER = secrets.token_hex(4)
ETCD_SECRET = secrets.token_hex(16)
ETCD_URI = "etcd://etcd-host:2379"
LOGIN = {"user": ETCD_USER, "password": ETCD_SECRET, "ca_cert": "/ca.pem"}


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
