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
"""Exercise producer ownership through real local and HTTP retention catalogs."""

import hashlib
import threading
import uuid

import pytest
from test_mooncake_store_retention import FileStore, drain

from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.checkpoint_lifecycle import (
    CheckpointPublication,
    CheckpointTransfer,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.retention import (
    CheckpointRetention,
    EndpointUnavailable,
    RetentionError,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.retention_rpc import (
    RemoteCheckpointRetention,
    RetentionCoordinator,
    make_server,
)

pytestmark = pytest.mark.cpu_only


@pytest.fixture(params=["local", "rpc"])
def backend(request, tmp_path):
    epoch = uuid.uuid4().hex
    namespace = "checkpoint-test/" + epoch
    store = FileStore(tmp_path / "store")
    if request.param == "local":
        catalog = CheckpointRetention(str(tmp_path / "catalog"), namespace, 2)
        yield catalog, store
        catalog.close()
        return
    coordinator = RetentionCoordinator(str(tmp_path / "catalog"), epoch)
    server = make_server(("127.0.0.1", 0), coordinator, "test-capability")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    catalog = RemoteCheckpointRetention(
        dict(host="127.0.0.1", port=server.server_port, epoch=epoch, token="test-capability"),
        namespace,
        2,
    )
    try:
        yield catalog, store
    finally:
        catalog.close()
        server.shutdown()
        server.server_close()
        thread.join(5)
        assert not thread.is_alive()
        # Test I/O has stopped. Do not model production lease expiry here.
        for _, lease in coordinator._leases.values():
            lease.close(retired=False)


def publication(catalog, number, *, turn=None):
    digest = hashlib.sha256(str(number).encode()).hexdigest()
    return CheckpointPublication(
        catalog.namespace + "/complete/" + digest,
        (catalog.namespace + "/group/1/" + digest,),
        "conversation",
        str(number) if turn is None else turn,
    )


def save(catalog, store, number, *, turn=None):
    checkpoint = publication(catalog, number, turn=turn)
    transfer = CheckpointTransfer.begin_save(catalog, checkpoint)
    for key in (*checkpoint.private_keys, checkpoint.marker):
        store.put(key)
    expired = transfer.finish(retired=True, success=True)
    return checkpoint, expired


def test_load_pin_spans_lookup_through_transfer_retirement(backend):
    catalog, store = backend
    first, _ = save(catalog, store, 0)
    load = CheckpointTransfer.begin_load(catalog, first.marker)
    assert store.exists(first.marker)  # Lookup occurs under the lease.
    save(catalog, store, 1)
    _, expired = save(catalog, store, 2)
    assert expired == 1
    drain(catalog, store)
    assert all(store.exists(key) for key in (*first.private_keys, first.marker))
    assert load.finish(retired=True, success=True) == 0
    drain(catalog, store)
    assert not store.exists(first.marker)


def test_save_is_not_published_before_retirement(backend):
    catalog, store = backend
    first, _ = save(catalog, store, 0)
    save(catalog, store, 1)
    checkpoint = publication(catalog, 2)
    transfer = CheckpointTransfer.begin_save(catalog, checkpoint)
    for key in (*checkpoint.private_keys, checkpoint.marker):
        store.put(key)
    with pytest.raises(RetentionError, match="requires retired"):
        transfer.finish(retired=False, success=True)
    drain(catalog, store)
    assert store.exists(first.marker)
    assert transfer.finish(retired=True, success=True) == 1
    drain(catalog, store)
    assert not store.exists(first.marker)


@pytest.mark.parametrize("retired", [True, False])
def test_failed_or_cancelled_save_never_advances_retention(backend, retired):
    catalog, store = backend
    first, _ = save(catalog, store, 0)
    save(catalog, store, 1)
    checkpoint = publication(catalog, 2)
    transfer = CheckpointTransfer.begin_save(catalog, checkpoint)
    store.put(checkpoint.private_keys[0])  # Partial PUT, no completion marker.
    assert transfer.finish(retired=retired, success=False) == 0
    drain(catalog, store)
    assert store.exists(first.marker)
    assert store.exists(checkpoint.private_keys[0]) is (not retired)
    with pytest.raises(RetentionError, match="already finished"):
        transfer.finish(retired=True, success=True)


def test_metadata_failure_keeps_pin_and_does_not_retry_publication(backend, monkeypatch):
    catalog, store = backend
    checkpoint = publication(catalog, 0)
    transfer = CheckpointTransfer.begin_save(catalog, checkpoint)
    for key in (*checkpoint.private_keys, checkpoint.marker):
        store.put(key)

    def unavailable(*args):
        raise OSError("metadata unavailable")

    monkeypatch.setattr(catalog, "publish", unavailable)
    with pytest.raises(OSError, match="metadata unavailable"):
        transfer.finish(retired=True, success=True)
    drain(catalog, store)
    assert store.exists(checkpoint.marker)
    with pytest.raises(RetentionError, match="already finished"):
        transfer.finish(retired=True, success=True)


def test_rejected_manifest_submits_no_io_and_leaves_no_active_lease(backend):
    catalog, store = backend
    checkpoint = publication(catalog, 0)
    invalid = CheckpointPublication(checkpoint.marker, ("shared-prefix",), "c", "t")
    with pytest.raises(RetentionError):
        CheckpointTransfer.begin_save(catalog, invalid)
    save(catalog, store, 0)
    save(catalog, store, 1)
    save(catalog, store, 2)
    drain(catalog, store)
    assert not store.exists(checkpoint.marker)


def test_quarantined_endpoint_is_an_admission_failure(backend):
    catalog, store = backend
    first, _ = save(catalog, store, 0)
    save(catalog, store, 1)
    save(catalog, store, 2)
    store.fail_key = first.private_keys[0]
    drain(catalog, store)
    with pytest.raises(EndpointUnavailable):
        CheckpointTransfer.begin_load(catalog, first.marker)
    with pytest.raises(EndpointUnavailable):
        CheckpointTransfer.begin_save(catalog, first)


def test_turns_use_publication_order_and_keep_shared_pages(backend):
    catalog, store = backend
    shared = catalog.namespace + "/shared-attention-page"
    store.put(shared)
    first, _ = save(catalog, store, 10, turn="turn10")
    save(catalog, store, 11, turn="turn11")
    late, expired = save(catalog, store, 1, turn="turn1")
    assert expired == 1
    drain(catalog, store)
    assert not store.exists(first.marker)
    assert store.exists(late.marker)
    assert store.exists(shared)


def test_missing_turn_rejected_before_admission():
    with pytest.raises(ValueError, match="turn identity"):
        CheckpointPublication("marker", ("private",), "conversation", "")
