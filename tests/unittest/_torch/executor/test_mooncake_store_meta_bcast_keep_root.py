# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The TP leader keeps the connector metadata it built (TRTLLM_KV_CONNECTOR_META_BCAST_KEEP_ROOT=1).

mpi4py's pkl5 ``bcast`` returns ``loads(dumps(obj))`` on the root as well, so the leader used to unpickle its own
metadata every iteration. With the switch the leader pickles once, broadcasts the bytes and keeps (a shallow copy
of) its object; the other ranks unpickle the bytes. CPU only, with a fake communicator.
"""

import pickle
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.connectors import kv_cache_connector as kvc
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.metadata import (
    MooncakeStoreMetadata,
    PageTransfer,
    PrefixLoadTransfers,
    RequestTransfers,
)

pytestmark = pytest.mark.cpu_only


class FakeComm:
    """pkl5-like bcast: the root's call returns loads(dumps(obj)); a non-root call returns `incoming`."""

    def __init__(self, incoming=None):
        self.sent = []
        self.incoming = incoming

    def bcast(self, obj, root=0):
        if obj is not None:
            self.sent.append(obj)
            return pickle.loads(pickle.dumps(obj))
        return pickle.loads(pickle.dumps(self.incoming))


def _metadata():
    page = PageTransfer(b"\x01" * 16, 0, 7, 3)
    return MooncakeStoreMetadata(
        loads=[RequestTransfers(1, [page])],
        saves=[RequestTransfers(2, [PageTransfer(b"\x02" * 16, 1, 9, 4)])],
        prefix_loads=[PrefixLoadTransfers(5, True, RequestTransfers(3, [page]))],
    )


def _manager(*, leader, keep_root, adp=False):
    return SimpleNamespace(
        scheduler=object() if leader else None,
        enable_attention_dp=adp,
        _meta_bcast_keep_root=keep_root,
    )


def _run(manager, f, **kwargs):
    return kvc.KvCacheConnectorManager._run_on_leader(manager, f, **kwargs)


@pytest.fixture
def world(monkeypatch):
    comm = FakeComm()
    monkeypatch.setattr(kvc, "mpi_comm", lambda: comm)
    monkeypatch.setattr(kvc, "mpi_rank", lambda: 0)
    monkeypatch.setattr(kvc, "global_mpi_size", lambda: 4)
    broadcasts = []

    def mpi_broadcast(obj, root=0):
        broadcasts.append(obj)
        return pickle.loads(pickle.dumps(obj))

    monkeypatch.setattr(kvc, "mpi_broadcast", mpi_broadcast)
    return SimpleNamespace(comm=comm, broadcasts=broadcasts)


def test_switch_name_and_default_off(monkeypatch):
    assert kvc.META_BCAST_KEEP_ROOT_ENV == "TRTLLM_KV_CONNECTOR_META_BCAST_KEEP_ROOT"
    monkeypatch.delenv(kvc.META_BCAST_KEEP_ROOT_ENV, raising=False)
    # The manager reads it in __init__; the attribute is absent on stubs and then counts as off.
    assert not getattr(SimpleNamespace(), "_meta_bcast_keep_root", False)


@pytest.mark.parametrize("keep_root", [False, True])
def test_switch_off_or_not_metadata_is_todays_broadcast(world, keep_root):
    meta = _metadata()
    # Switch off: the pkl5 broadcast, the leader gets a deep copy.
    out = _run(_manager(leader=True, keep_root=False), lambda: meta, keep_root=keep_root)
    assert out == meta and out is not meta and out.loads[0] is not meta.loads[0]
    # Switch on but not a metadata call: today's broadcast too.
    out = _run(_manager(leader=True, keep_root=True), lambda: (5, False))
    assert out == (5, False)
    assert world.broadcasts == [meta, (5, False)] and world.comm.sent == []


def test_leader_keeps_its_object_and_sends_the_pickled_bytes_once(world):
    meta = _metadata()
    out = _run(_manager(leader=True, keep_root=True), lambda: meta, keep_root=True)
    assert world.broadcasts == [] and len(world.comm.sent) == 1
    payload = world.comm.sent[0]
    assert isinstance(payload, bytes) and pickle.loads(payload) == meta
    # A shallow copy: same entries, no unpickle, but rebinding its attributes leaves the built object alone.
    assert out == meta and out is not meta
    assert out.loads is meta.loads and out.loads[0] is meta.loads[0]
    out.loads = []
    out._localized = True
    assert meta.loads and not getattr(meta, "_localized", False)


def test_followers_get_the_same_object_graph(world, monkeypatch):
    meta = _metadata()
    payload = pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL)
    world.comm.incoming = payload
    followed = []
    out = _run(
        _manager(leader=False, keep_root=True),
        lambda: pytest.fail("a follower never builds"),
        follower=lambda: followed.append(1),
        keep_root=True,
    )
    assert out == meta and followed == [1] and world.comm.sent == [] and world.broadcasts == []


def test_attention_dp_and_single_rank_never_broadcast(world, monkeypatch):
    meta = _metadata()
    assert _run(_manager(leader=True, keep_root=True, adp=True), lambda: meta, keep_root=True) is meta
    monkeypatch.setattr(kvc, "global_mpi_size", lambda: 1)
    assert _run(_manager(leader=True, keep_root=True), lambda: meta, keep_root=True) is meta
    assert world.broadcasts == [] and world.comm.sent == []


@pytest.mark.parametrize("keep_root", [False, True])
def test_handle_metadata_uses_it_only_with_the_switch(keep_root):
    calls = []
    bound = {}

    def run_on_leader(f, follower=None, **kwargs):
        calls.append(kwargs)
        return f()

    manager = SimpleNamespace(
        _scheduler_output=SimpleNamespace(prefix_loads=[]),
        _bound_prefix_load_ids=set(),
        _bound_scheduler_output=None,
        _allocation_fence=None,
        _execution_stream=None,
        _local_page_indices=None,
        _meta_bcast_keep_root=keep_root,
        _run_on_leader=run_on_leader,
        scheduler=SimpleNamespace(build_connector_meta=lambda output: "meta"),
        worker=SimpleNamespace(
            bind_allocation_fence=lambda fence: None,
            bind_connector_meta=lambda meta: bound.__setitem__("meta", meta),
        ),
    )
    kvc.KvCacheConnectorManager.handle_metadata(manager)
    assert bound == {"meta": "meta"}
    assert calls == ([{"keep_root": True}] if keep_root else [{}])
