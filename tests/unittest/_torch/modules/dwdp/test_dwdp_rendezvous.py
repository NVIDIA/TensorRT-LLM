# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for the store-based DWDP rendezvous (plain trtllm-serve launches)."""

import pickle
import threading
from unittest.mock import patch

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

pytestmark = pytest.mark.cpu_only

_MOD = "tensorrt_llm._torch.modules.dwdp.rendezvous"


class _FakeStore:
    """The three TCPStore methods the group uses, without a socket."""

    def __init__(self, seeded=None):
        self.data = dict(seeded or {})
        self.waited = []

    def set(self, key, value):
        self.data[key] = value

    def get(self, key):
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


@pytest.mark.parametrize(
    "rank,addr,expected",
    [(None, None, False), ("0", None, False), (None, "node-a", False), ("0", "node-a", True)],
    ids=["neither", "rank-only", "addr-only", "rank0-and-addr"],
)
def test_enabled_needs_both_rank_and_addr(monkeypatch, rank, addr, expected):
    _set_env(monkeypatch, DWDP_RANK_ENV, rank)
    _set_env(monkeypatch, DWDP_MASTER_ADDR_ENV, addr)
    assert store_rendezvous_enabled() is expected


@pytest.mark.parametrize(
    "rank_env,expected", [("0", 0), ("13", 13), (None, 7)], ids=["zero", "env-wins", "mpi-fallback"]
)
def test_global_rank_prefers_env_over_mpi(monkeypatch, rank_env, expected):
    _set_env(monkeypatch, DWDP_RANK_ENV, rank_env)
    with patch(f"{_MOD}.global_mpi_rank", return_value=7) as mock_mpi:
        assert dwdp_global_rank() == expected
    assert mock_mpi.called is (rank_env is None)


def test_negative_rank_rejected(monkeypatch):
    monkeypatch.setenv(DWDP_RANK_ENV, "-1")
    with pytest.raises(ValueError, match="non-negative"):
        dwdp_global_rank()


def test_from_env_requires_master_addr(monkeypatch):
    monkeypatch.delenv(DWDP_MASTER_ADDR_ENV, raising=False)
    with pytest.raises(RuntimeError, match=DWDP_MASTER_ADDR_ENV):
        StoreDwdpGroup.from_env(global_rank=0, dwdp_size=4)


def test_from_env_reads_port_and_timeout(monkeypatch):
    monkeypatch.setenv(DWDP_MASTER_ADDR_ENV, "node-a")
    monkeypatch.setenv(DWDP_MASTER_PORT_ENV, "31000")
    monkeypatch.setenv(DWDP_TIMEOUT_ENV, "60")
    with patch(f"{_MOD}.StoreDwdpGroup.__init__", return_value=None) as mock_init:
        StoreDwdpGroup.from_env(global_rank=2, dwdp_size=4)
    kwargs = mock_init.call_args.kwargs
    assert kwargs["master_addr"] == "node-a"
    assert (kwargs["master_port"], kwargs["timeout_s"]) == (31000, 60)


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
    assert all(key.startswith(prefix) for key in store.data)


@pytest.mark.parametrize("global_rank,dwdp_size", [(-1, 4), (0, 0), (0, -2)])
def test_invalid_topology_rejected(global_rank, dwdp_size):
    with pytest.raises(ValueError):
        _group(global_rank, dwdp_size, _FakeStore())


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


def test_two_ranks_rendezvous_over_a_real_store():
    """End-to-end through torch's own store, one thread per worker."""
    torch_dist = pytest.importorskip("torch.distributed")
    store = torch_dist.HashStore()
    results = {}
    barrier_seen = threading.Event()

    def worker(rank):
        group = StoreDwdpGroup(
            global_rank=rank, dwdp_size=2, master_addr="unused", timeout_s=60, store=store
        )
        results[rank] = group.allgather(f"handle-{rank}")
        group.Barrier()
        if rank == 1:
            barrier_seen.set()
        group.Free()

    threads = [threading.Thread(target=worker, args=(r,)) for r in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
        assert not thread.is_alive(), "rendezvous deadlocked"

    assert results == {0: ["handle-0", "handle-1"], 1: ["handle-0", "handle-1"]}
    assert barrier_seen.is_set()
