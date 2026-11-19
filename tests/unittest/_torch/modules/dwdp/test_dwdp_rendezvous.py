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
"""Tests for the store-based DWDP rendezvous (independently launched context workers)."""

import gc
import pickle
import threading
from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest
import torch

from tensorrt_llm._torch.modules.dwdp.rendezvous import (
    DWDP_MASTER_ADDR_ENV,
    DWDP_MASTER_PORT_ENV,
    DWDP_RANK_ENV,
    DWDP_TIMEOUT_ENV,
    StoreDwdpGroup,
    dwdp_global_rank,
    store_rendezvous_enabled,
)
from tensorrt_llm._torch.pyexecutor.dwdp import DwdpManager, set_global_dwdp_manager
from tensorrt_llm.llmapi.llm_args import DwdpConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

_MOD = "tensorrt_llm._torch.modules.dwdp.rendezvous"
_MGR_MOD = "tensorrt_llm._torch.pyexecutor.dwdp"
_LOCALHOST = "127.0.0.1"


class _FakeStore:
    """The TCPStore methods the group uses, without a socket."""

    def __init__(self, seeded=None):
        self.data = dict(seeded or {})
        self.waited = []

    def set(self, key, value):
        self.data[key] = value

    def get(self, key):
        return self.data[key]

    def add(self, key, amount):
        self.data[key] = self.data.get(key, 0) + amount
        return self.data[key]

    def wait(self, keys, timeout=None):
        self.waited.append(list(keys))
        missing = [key for key in keys if key not in self.data]
        if missing:
            raise TimeoutError(f"keys never arrived: {missing}")


def _group(global_rank, dwdp_size, store):
    return StoreDwdpGroup(
        global_rank=global_rank, dwdp_size=dwdp_size, master_addr="unused", store=store
    )


def _seed_peers(prefix, values, skip):
    """Pickled payloads for every rank but ``skip`` (the rank under test)."""
    return {
        f"{prefix}{rank}": pickle.dumps(value) for rank, value in enumerate(values) if rank != skip
    }


def _set_env(monkeypatch, name, value):
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)


@pytest.fixture
def clean_env(monkeypatch):
    for name in (DWDP_RANK_ENV, DWDP_MASTER_ADDR_ENV, DWDP_MASTER_PORT_ENV, DWDP_TIMEOUT_ENV):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


# ----------------------------------------------------------------------
# Environment parsing
# ----------------------------------------------------------------------


@pytest.mark.parametrize("rank,addr,expected", [(None, None, False), ("0", "node-a", True)])
def test_enabled_needs_both_rank_and_addr(clean_env, rank, addr, expected):
    _set_env(clean_env, DWDP_RANK_ENV, rank)
    _set_env(clean_env, DWDP_MASTER_ADDR_ENV, addr)
    assert store_rendezvous_enabled() is expected


@pytest.mark.parametrize(
    "rank,addr,missing",
    [("0", None, DWDP_MASTER_ADDR_ENV), (None, "node-a", DWDP_RANK_ENV)],
    ids=["rank-only", "addr-only"],
)
def test_half_configured_env_is_rejected(clean_env, rank, addr, missing):
    """A rank from the env must never be paired with the MPI sub-communicator."""
    _set_env(clean_env, DWDP_RANK_ENV, rank)
    _set_env(clean_env, DWDP_MASTER_ADDR_ENV, addr)
    with pytest.raises(ValueError, match=missing):
        store_rendezvous_enabled()
    with pytest.raises(ValueError, match=missing):
        dwdp_global_rank()


@pytest.mark.parametrize(
    "rank_env,expected", [("0", 0), ("13", 13), (None, 7)], ids=["zero", "env-wins", "mpi-fallback"]
)
def test_global_rank_prefers_env_over_mpi(clean_env, rank_env, expected):
    _set_env(clean_env, DWDP_RANK_ENV, rank_env)
    _set_env(clean_env, DWDP_MASTER_ADDR_ENV, None if rank_env is None else "node-a")
    with patch(f"{_MOD}.global_mpi_rank", return_value=7) as mock_mpi:
        assert dwdp_global_rank() == expected
    assert mock_mpi.called is (rank_env is None)


def test_negative_rank_rejected(clean_env):
    clean_env.setenv(DWDP_RANK_ENV, "-1")
    clean_env.setenv(DWDP_MASTER_ADDR_ENV, "node-a")
    with pytest.raises(ValueError, match="non-negative"):
        dwdp_global_rank()


def test_from_env_requires_master_addr(clean_env):
    with pytest.raises(RuntimeError, match=DWDP_MASTER_ADDR_ENV):
        StoreDwdpGroup.from_env(global_rank=0, dwdp_size=4)


def test_from_env_reads_port_and_timeout(clean_env):
    clean_env.setenv(DWDP_MASTER_ADDR_ENV, "node-a")
    clean_env.setenv(DWDP_MASTER_PORT_ENV, "31000")
    clean_env.setenv(DWDP_TIMEOUT_ENV, "60")
    with patch(f"{_MOD}.StoreDwdpGroup.__init__", return_value=None) as mock_init:
        StoreDwdpGroup.from_env(global_rank=2, dwdp_size=4, num_groups=3)
    kwargs = mock_init.call_args.kwargs
    assert kwargs["master_addr"] == "node-a"
    assert (kwargs["master_port"], kwargs["timeout_s"], kwargs["num_groups"]) == (31000, 60, 3)


# ----------------------------------------------------------------------
# Group semantics over an in-memory store
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "global_rank,expected_group,expected_rank",
    [(0, 0, 0), (3, 0, 3), (4, 1, 0), (5, 1, 1), (11, 2, 3)],
)
def test_group_rank_and_keys_follow_global_rank(global_rank, expected_group, expected_rank):
    """Peers are seeded only under the expected group id; results come back in rank order."""
    prefix = f"dwdp/{expected_group}/0/"
    payloads = [f"from-{r}" for r in range(4)]
    store = _FakeStore(_seed_peers(prefix, payloads, skip=expected_rank))
    group = _group(global_rank, 4, store)
    assert (group.Get_rank(), group.Get_size()) == (expected_rank, 4)
    assert group.allgather(payloads[expected_rank]) == payloads
    assert all(key.startswith(f"dwdp/{expected_group}/") for key in store.data)


@pytest.mark.parametrize("global_rank,dwdp_size", [(-1, 4), (0, 0), (0, -2)])
def test_invalid_topology_rejected(global_rank, dwdp_size):
    with pytest.raises(ValueError):
        _group(global_rank, dwdp_size, _FakeStore())


def test_duplicate_rank_rejected():
    store = _FakeStore()
    _group(5, 4, store)
    with pytest.raises(RuntimeError, match="distinct"):
        _group(5, 4, store)
    # The same in-group rank in another group is a different worker.
    _group(1, 4, store)


def test_sequence_advances_per_collective():
    store = _FakeStore()
    group = _group(0, 1, store)
    group.allgather("first")
    group.Barrier()
    group.allgather("third")
    assert [keys[0] for keys in store.waited] == ["dwdp/0/0/0", "dwdp/0/1/0", "dwdp/0/2/0"]
    assert pickle.loads(store.data["dwdp/0/1/0"]) is None


def test_tensor_payloads_survive_the_round_trip():
    shard = torch.arange(8, dtype=torch.float32)
    (gathered,) = _group(0, 1, _FakeStore()).allgather(shard)
    assert torch.equal(gathered, shard)


def test_missing_peer_surfaces_as_timeout():
    with pytest.raises(TimeoutError):
        _group(0, 4, _FakeStore()).allgather("only-me")


def test_free_is_idempotent_and_blocks_reuse():
    group = _group(0, 1, _FakeStore())
    group.Free()
    group.Free()
    with pytest.raises(RuntimeError, match="freed"):
        group.allgather("x")


# ----------------------------------------------------------------------
# Real TCPStore on localhost
# ----------------------------------------------------------------------


def _run_workers(target, ranks):
    errors = {}

    def wrap(rank):
        try:
            target(rank)
        except BaseException as exc:  # surfaced below on the main thread
            errors[rank] = exc

    # Daemon threads: a peer stuck in ``store.wait`` must not keep pytest alive.
    threads = [threading.Thread(target=wrap, args=(r,), daemon=True) for r in ranks]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
        assert not thread.is_alive(), "rendezvous deadlocked"
    assert not errors, errors


def test_two_groups_rendezvous_over_a_localhost_tcpstore():
    """Two groups of two workers share one TCPStore hosted by DWDP rank 0."""
    master = StoreDwdpGroup(
        global_rank=0,
        dwdp_size=2,
        master_addr=_LOCALHOST,
        master_port=0,
        timeout_s=60,
        num_groups=2,
    )
    port = master._store.port
    groups = {0: master}
    results = {}

    def worker(rank):
        group = groups.get(rank) or StoreDwdpGroup(
            global_rank=rank,
            dwdp_size=2,
            master_addr=_LOCALHOST,
            master_port=port,
            timeout_s=60,
            num_groups=2,
        )
        groups[rank] = group
        results[rank] = group.allgather(torch.tensor([rank]))
        group.Barrier()
        results[rank] = [int(t) for t in results[rank]] + group.allgather(f"pid-{rank}")

    _run_workers(worker, ranks=(0, 1, 2, 3))
    # Rank 0 hosts the store both groups use, so it is freed last, after every worker is done.
    for rank in (3, 2, 1, 0):
        groups[rank].Free()

    assert results == {
        0: [0, 1, "pid-0", "pid-1"],
        1: [0, 1, "pid-0", "pid-1"],
        2: [2, 3, "pid-2", "pid-3"],
        3: [2, 3, "pid-2", "pid-3"],
    }


def test_free_on_rank0_shuts_the_store_down():
    """Teardown releases the TCPStore: once rank 0 frees, nothing can connect."""
    from torch.distributed import TCPStore

    master = StoreDwdpGroup(
        global_rank=0, dwdp_size=2, master_addr=_LOCALHOST, master_port=0, timeout_s=60
    )
    port = master._store.port
    peer = StoreDwdpGroup(
        global_rank=1, dwdp_size=2, master_addr=_LOCALHOST, master_port=port, timeout_s=10
    )
    peer.Free()
    master.Free()
    gc.collect()
    with pytest.raises((RuntimeError, TimeoutError)):
        TCPStore(_LOCALHOST, port, is_master=False, timeout=timedelta(seconds=2))


def test_rank0_free_keeps_the_store_until_every_group_is_set_up():
    """Group 0 finishing first must not cut off group 1, which still exchanges handles."""
    groups = {
        0: StoreDwdpGroup(
            global_rank=0,
            dwdp_size=2,
            master_addr=_LOCALHOST,
            master_port=0,
            timeout_s=60,
            num_groups=2,
        )
    }
    port = groups[0]._store.port
    for rank in (1, 2, 3):
        groups[rank] = StoreDwdpGroup(
            global_rank=rank,
            dwdp_size=2,
            master_addr=_LOCALHOST,
            master_port=port,
            timeout_s=60,
            num_groups=2,
        )

    # Group 0 is set up and shuts down; rank 0's Free must wait for group 1.
    groups[1].Free()
    rank0_free = threading.Thread(target=groups[0].Free, daemon=True)
    rank0_free.start()
    rank0_free.join(timeout=1)
    assert rank0_free.is_alive(), "rank 0 released the store while group 1 was still setting up"

    # Group 1 can still run its setup collectives over the store.
    results = {}

    def group1_setup(rank):
        results[rank] = groups[rank].allgather(f"handle-{rank}")
        groups[rank].mark_setup_done()

    _run_workers(group1_setup, ranks=(2, 3))
    assert results == {2: ["handle-2", "handle-3"], 3: ["handle-2", "handle-3"]}

    rank0_free.join(timeout=30)
    assert not rank0_free.is_alive(), "rank 0 never released the store"
    assert groups[0]._store is None
    for rank in (3, 2):
        groups[rank].Free()


def test_rank0_free_gives_up_after_the_timeout():
    store = _FakeStore()
    group = StoreDwdpGroup(
        global_rank=0, dwdp_size=2, master_addr="unused", timeout_s=0, store=store, num_groups=2
    )
    with patch(f"{_MOD}.logger") as mock_logger:
        group.Free()
    assert group._store is None
    assert store.data["dwdp/setup_done"] == 1
    mock_logger.warning.assert_called_once()


def test_setup_done_counts_each_worker_once():
    store = _FakeStore()
    group = _group(1, 2, store)
    group.mark_setup_done()
    group.mark_setup_done()
    group.Free()
    assert store.data["dwdp/setup_done"] == 1


def test_non_positive_num_groups_rejected():
    with pytest.raises(ValueError, match="num_groups"):
        StoreDwdpGroup(
            global_rank=0, dwdp_size=2, master_addr="unused", store=_FakeStore(), num_groups=0
        )


# ----------------------------------------------------------------------
# DwdpManager under a store rendezvous
# ----------------------------------------------------------------------


def _mapping(dwdp_size=2, dwdp_rank=0):
    return Mapping(world_size=1, rank=0, tp_size=1, dwdp_size=dwdp_size, dwdp_rank=dwdp_rank)


def _config(num_groups=1, dwdp_size=2):
    return DwdpConfig(
        dwdp_size=dwdp_size,
        num_groups=num_groups,
        num_experts_per_worker=4,
        num_prefetch_experts=4,
    )


@pytest.fixture
def no_global_manager():
    set_global_dwdp_manager(None)
    yield
    set_global_dwdp_manager(None)


def test_manager_uses_store_group_without_mpi_world(clean_env, no_global_manager):
    """Neither COMM_WORLD's size nor its sub-communicators are consulted; cleanup frees the store."""
    clean_env.setenv(DWDP_RANK_ENV, "2")
    clean_env.setenv(DWDP_MASTER_ADDR_ENV, "node-a")
    store = _FakeStore()
    real_init = StoreDwdpGroup.__init__

    def init_with_fake_store(self, **kwargs):
        real_init(self, **{**kwargs, "store": store})

    with (
        patch(f"{_MGR_MOD}.COMM_WORLD") as mock_world,
        patch(f"{_MOD}.global_mpi_rank", side_effect=AssertionError("MPI rank consulted")),
        patch.object(StoreDwdpGroup, "__init__", init_with_fake_store),
    ):
        mgr = DwdpManager(
            config=_config(num_groups=2),
            # A non-MPI dist is accepted under a store rendezvous.
            dist=MagicMock(),
            mapping=_mapping(dwdp_rank=0),
        )
        assert mgr.rank == 2
        group = mgr.dwdp_group
        assert isinstance(group, StoreDwdpGroup)
        assert (group.Get_rank(), group.Get_size(), group._group_id) == (0, 2, 1)
        assert group._num_workers == 4
        mgr.cleanup()
        mgr.cleanup()

    mock_world.Get_size.assert_not_called()
    mock_world.Create_group.assert_not_called()
    assert mgr.dwdp_group is None
    assert group._store is None


def test_manager_rejects_rank_beyond_num_groups(clean_env, no_global_manager):
    clean_env.setenv(DWDP_RANK_ENV, "4")
    clean_env.setenv(DWDP_MASTER_ADDR_ENV, "node-a")
    with patch.object(StoreDwdpGroup, "from_env") as mock_from_env:
        with pytest.raises(ValueError, match="num_groups"):
            DwdpManager(config=_config(num_groups=2), dist=MagicMock(), mapping=_mapping())
    mock_from_env.assert_not_called()


def test_manager_without_store_env_still_requires_mpi(clean_env, no_global_manager):
    with pytest.raises(RuntimeError, match="MPIDist"):
        DwdpManager(config=_config(), dist=MagicMock(), mapping=_mapping())
