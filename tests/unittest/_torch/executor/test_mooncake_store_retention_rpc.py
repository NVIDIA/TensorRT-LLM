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
"""Real localhost HTTP tests for the job-scoped coordinator (CPU only)."""

import hashlib
import secrets
import threading
import uuid
from http.client import HTTPConnection, HTTPResponse
from io import BytesIO
from socket import socket
from unittest.mock import create_autospec

import pytest
from test_mooncake_store_retention import FileStore

from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import retention_rpc as rpc

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def service(tmp_path):
    epoch, token = uuid.uuid4().hex, secrets.token_hex(32)
    coordinator = rpc.RetentionCoordinator(str(tmp_path / "local"), epoch)
    server = rpc.make_server(("127.0.0.1", 0), coordinator, token)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    descriptor = dict(host="127.0.0.1", port=server.server_port, epoch=epoch, token=token)
    namespace = "trtllm/rpc-test/retention/" + epoch
    clients = []

    def client(**overrides):
        value = rpc.RemoteCheckpointRetention(descriptor | overrides, namespace)
        clients.append(value)
        return value

    yield client, FileStore(tmp_path / "store"), coordinator
    for value in clients:
        value.close()
    server.shutdown()
    server.server_close()
    thread.join(2)


def keys(client, number):
    digest = hashlib.sha256(str(number).encode()).hexdigest()
    return client.namespace + "/complete/" + digest, client.namespace + "/group/3/" + digest


def save(client, store, number):
    marker, private = keys(client, number)
    lease = client.acquire(marker)
    try:
        client.prepare_save(lease, marker, (private,))
        store.put(private)
        store.put(marker)
        client.publish(lease, "conversation", str(number))
    finally:
        lease.close(retired=True)


def drain(client, store):
    for _ in range(4):
        client.collect(store)


def test_distinct_rpc_clients_share_one_five_turn_limit(service):
    factory, store, _ = service
    clients = [factory() for _ in range(12)]
    for number, client in enumerate(clients):
        save(client, store, number)
    drain(clients[0], store)
    for number in range(12):
        assert store.exists(keys(clients[0], number)[0]) == (number >= 7)


def test_other_client_cannot_collect_active_get(service):
    factory, store, _ = service
    writer, reader = factory(), factory()
    save(writer, store, 0)
    lease = reader.acquire(keys(reader, 0)[0])
    for number in range(1, 6):
        save(writer, store, number)
    drain(writer, store)
    assert store.exists(keys(writer, 0)[0])
    lease.close(retired=True)
    drain(writer, store)
    assert not store.exists(keys(writer, 0)[0])


def test_lost_client_has_no_lease_timeout(service):
    factory, store, coordinator = service
    writer, lost = factory(), factory()
    save(writer, store, 0)
    lost.acquire(keys(lost, 0)[0])
    lost.close()
    for number in range(1, 6):
        save(writer, store, number)
    drain(writer, store)
    assert store.exists(keys(writer, 0)[0])
    assert len(coordinator._leases) == 1
    # Explicit test teardown after all simulated I/O, not an automatic TTL.
    _, lease = next(iter(coordinator._leases.values()))
    lease.close(retired=True)


@pytest.mark.parametrize("override", ["token", "epoch"])
def test_bad_capability_or_epoch_fails_closed(service, override):
    factory, _, _ = service
    client = factory(**{override: "wrong"})
    with pytest.raises(RuntimeError):
        client.acquire(keys(client, 0)[0])


def test_acquired_gc_claim_excludes_new_save_until_delete_finishes(service):
    factory, store, _ = service
    collector, writer = factory(), factory()
    for number in range(6):
        save(writer, store, number)
    response = collector._call("claim")
    if not response["claims"]:
        response = collector._call("claim")
    claim = response["claims"][0]
    # An abandoned exclusive claim remains fenced, but callers never block.
    for _ in range(3):
        with pytest.raises(rpc.EndpointUnavailable):
            save(writer, store, 0)
    assert claim["token"] in service[2]._claims
    for key in [claim["manifest"]["marker"], *claim["manifest"]["private_keys"]]:
        store.remove(key, False)
    collector._call("finish_claim", token=claim["token"], success=True)
    save(writer, store, 0)
    assert store.exists(keys(writer, 0)[0])


def test_wrong_namespace_cannot_release_lease(service):
    factory, _, _ = service
    client, other = factory(), factory()
    lease = client.acquire(keys(client, 0)[0])
    other.namespace += "/different"
    with pytest.raises(RuntimeError):
        other._call("release", token=lease.token, retired=True)
    lease.close(retired=True)


@pytest.mark.parametrize(
    "status,body",
    [
        (200, b"null"),
        (200, b"{}"),
        (200, b"not-json"),
        (409, b'{"error":"RetentionError"}'),
        (403, b"denied"),
    ],
)
def test_protocol_errors_are_not_endpoint_unavailable(status, body):
    client = rpc.RemoteCheckpointRetention(
        dict(host="localhost", port=1, epoch="e", token="t"), "e"
    )
    transport = create_autospec(socket, instance=True, spec_set=True)
    transport.makefile.return_value = BytesIO(
        f"HTTP/1.1 {status} Test\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body
    )
    response = HTTPResponse(transport)
    response.begin()
    client._connection = create_autospec(HTTPConnection, instance=True, spec_set=True)
    client._connection.getresponse.return_value = response
    with pytest.raises(rpc.RetentionError) as caught:
        client.acquire("marker")
    assert not isinstance(caught.value, rpc.EndpointUnavailable)
