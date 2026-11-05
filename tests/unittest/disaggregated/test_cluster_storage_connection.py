# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How Etcd3ClusterStorage connects to etcd: credentials, TLS files and reconnecting."""

import secrets
from unittest import mock

import grpc
import pytest

from tensorrt_llm.serve import cluster_storage
from tensorrt_llm.serve.cluster_storage import Etcd3ClusterStorage

pytestmark = pytest.mark.cpu_only

ETCD_USER = secrets.token_hex(4)
ETCD_SECRET = secrets.token_hex(16)
ETCD_URI = "etcd://etcd-host:2379"
OPTIONS = {
    "user": ETCD_USER,
    "password": ETCD_SECRET,
    "ca_cert": "/ca.pem",
    "cert_key": "/key.pem",
    "cert_cert": "/cert.pem",
    "record_auth_key": secrets.token_hex(16),
}


class _RpcError(grpc.RpcError):
    def __init__(self, code):
        super().__init__()
        self._code = code

    def code(self):
        return self._code


def test_etcd_storage_without_options_connects_as_before():
    with (
        mock.patch.object(cluster_storage.etcd3, "Endpoint") as endpoint,
        mock.patch.object(cluster_storage.etcd3, "MultiEndpointEtcd3Client") as client,
    ):
        Etcd3ClusterStorage("etcd://h1:2379,etcd://h2:2380", "c")
    assert endpoint.call_args_list == [
        mock.call("h1", 2379, secure=False, creds=None),
        mock.call("h2", 2380, secure=False, creds=None),
    ]
    client.assert_called_once_with(
        [endpoint.return_value, endpoint.return_value], timeout=10, user=None, password=None
    )


def test_etcd_storage_passes_credentials_and_tls_files_and_keeps_them_for_reconnect():
    with (
        mock.patch.object(cluster_storage.etcd3.Etcd3Client, "get_secure_creds") as secure_creds,
        mock.patch.object(cluster_storage.etcd3, "Endpoint") as endpoint,
        mock.patch.object(cluster_storage.etcd3, "MultiEndpointEtcd3Client") as client,
    ):
        storage = Etcd3ClusterStorage(ETCD_URI, "c", **OPTIONS)
        storage.reconnect()
    secure_creds.assert_called_with("/ca.pem", "/key.pem", "/cert.pem")
    endpoint.assert_called_with("etcd-host", 2379, secure=True, creds=secure_creds.return_value)
    assert client.call_count == 2
    client.assert_called_with(
        [endpoint.return_value], timeout=10, user=ETCD_USER, password=ETCD_SECRET
    )


@pytest.mark.asyncio
async def test_rejected_token_connects_again():
    stale, fresh = mock.MagicMock(), mock.MagicMock()
    stale.get.side_effect = _RpcError(grpc.StatusCode.UNAUTHENTICATED)
    fresh.get.return_value = (b"value", None)
    with mock.patch.object(Etcd3ClusterStorage, "_connect", side_effect=[stale, fresh]) as connect:
        storage = Etcd3ClusterStorage(ETCD_URI, "c", user=ETCD_USER, password=ETCD_SECRET)
        assert await storage.get("k") is None
        assert await storage.get("k") == "value"
    assert connect.call_count == 2
