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
"""Tests for the startup of Wide-EP fault tolerance, where every rank starts as its own process.

Threads stand in for ranks, and the rendezvous store is a real TCPStore.
"""

import functools
import os
import pickle
import re
import socket
import subprocess  # nosec B404
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from _torch.distributed.store_group_test_utils import Unpickled
from mpi4py import MPI
from torch.distributed import DistStoreError, TCPStore

from tensorrt_llm import _utils
from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.distributed import Distributed
from tensorrt_llm._torch.distributed.host_exchange import MnnvlDistFactory
from tensorrt_llm._torch.distributed.membership import (
    LIVENESS_KEY,
    RENDEZVOUS_ENV,
    REPLACEMENT_CANNOT_START,
    REPORT_INTERVAL_S,
    SHARED_SECRET_ENV,
    MembershipRecord,
    MembershipState,
    SignedStore,
    StoreGroup,
    open_store,
    rendezvous_host_port,
    wait_for_keys,
)
from tensorrt_llm._torch.distributed.membership import startup as startup_module
from tensorrt_llm._torch.distributed.membership.startup import (
    AGREED_STARTUP_KEY,
    WORKER_ARGS_KEY,
    join_startup_group,
    publish_startup_record,
    startup_fingerprint,
)
from tensorrt_llm._torch.distributed.mnnvl_memory import TpMnnvlMemory
from tensorrt_llm._torch.distributed.startup_record import ReplacementCannotStart, StartupRecord
from tensorrt_llm._torch.pyexecutor import hang_detector
from tensorrt_llm._torch.pyexecutor.hang_detector import (
    RANK_CRASH_KILL_GRACE_ENV,
    hard_kill_on_rank_crash,
    start_rank_crash_kill_watchdog,
)
from tensorrt_llm._utils import OMPI_COMM_TYPE_HOST
from tensorrt_llm._wide_ep_fault_tolerance import (
    FAULT_TOLERANCE_RANK_ENV,
    TURN_FAULT_TOLERANCE_OFF,
    RankIdentity,
)
from tensorrt_llm.llmapi.llm import BaseLLM
from tensorrt_llm.llmapi.llm_args import CheckpointIOPolicy
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

WORLD = 4
JOIN_TIMEOUT_S = 30.0


def _client(port):
    return TCPStore("127.0.0.1", port, is_master=False, timeout=timedelta(seconds=30))


@pytest.fixture(autouse=True)
def fresh_start(monkeypatch):
    """Give each test the startup state of a process that has not started a rank yet."""
    record = StartupRecord(report_progress=lambda: startup_module._report_startup_wait(0))
    monkeypatch.setattr(startup_module, "_state", startup_module.RankStartState(record))


@pytest.fixture
def store(monkeypatch):
    """Host the rendezvous store, give each thread its own client, and return the store's port."""
    server = TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False)
    clients = threading.local()

    def client():
        if not hasattr(clients, "store"):
            clients.store = _client(server.port)
        return clients.store

    monkeypatch.setattr(startup_module, "_startup_store", client)
    monkeypatch.setattr(startup_module._state, "client", Mock())
    yield server.port


def test_starting_ranks_gather_through_store(store):
    """Check the collectives, splits and subgroups of ranks that start together."""

    def rank(r):
        world = StoreGroup(_client(store), "w", r, range(WORLD), gpus_per_node=2)
        pairs = StoreGroup(_client(store), "pair", r, range(WORLD), active=[1, 3])
        node = world.Split_type(OMPI_COMM_TYPE_HOST)
        reversed_group = world.subgroup([3, 2, 1, 0])
        world.Barrier()
        return (
            world.allgather(f"x{r}"),
            r in (1, 3) and pairs.allgather(r),
            node.membership,
            node.allreduce(r, op=MPI.MIN),
            (world.allreduce(r), world.allreduce(r, op=MPI.MAX)),
            (reversed_group.Get_rank(), reversed_group.allgather(r)),
        )

    with ThreadPoolExecutor(WORLD) as pool:
        results = list(pool.map(rank, range(WORLD)))
    for r in range(WORLD):
        node = (0, 1) if r < 2 else (2, 3)
        assert results[r][0] == ["x0", "x1", "x2", "x3"]
        assert results[r][2:] == (node, node[0], (6, 3), (3 - r, [3, 2, 1, 0]))
    assert results[1][1] == [None, 1, None, 3]
    assert results[3][1] == results[1][1]


def test_startup_collective_reports_waits_and_steps(store, monkeypatch):
    """Check that a startup collective reports each change of its awaited peers, then its step."""
    client = startup_module._state.client
    waiting = threading.Event()
    client.report.side_effect = lambda **report: report["waiting_for"] and waiting.set()

    def late_peer():  # Rank 1 arrives once rank 0 reported that it waits for it.
        waiting.wait(JOIN_TIMEOUT_S)
        StoreGroup(_client(store), "g", 1, range(2)).allgather(1)

    peer = threading.Thread(target=late_peer)
    peer.start()
    try:
        group = StoreGroup(_client(store), "g", 0, range(2), startup_module._report_startup_wait)
        assert group.allgather(0) == [0, 1]
    finally:
        peer.join(JOIN_TIMEOUT_S)
    assert [call.kwargs for call in client.report.call_args_list] == [
        dict(waiting_for=0b10, progress=0),
        dict(waiting_for=0, progress=1),
    ]


def test_startup_reports_host_collective_waits(store, monkeypatch):
    """Report the peers and round a startup host collective waits in, and the startup group's."""
    monkeypatch.setattr(startup_module._state, "waiting_for", 0b0010)
    startup_module._on_tick_at_startup(SimpleNamespace(waited_epochs=((2, 7),)))
    startup_module._state.client.report.assert_called_once_with(
        waiting_for=0b0110, waited_epochs=((2, 7),)
    )


def test_host_weights_wait_reports_startup_progress(store, monkeypatch):
    """Report each poll for the node's host weights as a step of the replacement's startup."""
    monkeypatch.setattr(startup_module, "_HOST_WEIGHTS_CHECK_S", 0.0)
    monkeypatch.setattr(Distributed, "startup_record", startup_module._state.record)
    client = startup_module._state.client
    client.record = MembershipRecord(
        generation=1, state=MembershipState.COMMITTED, active_mask=0b1011
    )
    checks = iter([[2], [2], []])
    startup_module.wait_for_node_host_weights(3, lambda: next(checks))
    assert [call.kwargs for call in client.report.call_args_list] == [
        dict(waiting_for=0, progress=1),
        dict(waiting_for=0, progress=2),
    ]


def test_startup_ends_when_authority_fails():
    """Check that a wait for a peer ends at once when the authority hosting the store is gone."""
    authority = subprocess.Popen(  # nosec B603
        [
            sys.executable,
            "-c",
            "import time; from torch.distributed import TCPStore; "
            "s = TCPStore('127.0.0.1', 0, is_master=True, wait_for_workers=False); "
            "print(s.port, flush=True); time.sleep(600)",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        port = int(authority.stdout.readline())
        # The authority ends while rank 0 waits for rank 1.
        group = StoreGroup(
            _client(port), "g", 0, range(2), lambda waiting: waiting and authority.kill()
        )
        with pytest.raises(RuntimeError, match="lost the rendezvous store"):
            group.allgather(0)
        assert group.exchange_failed
    finally:
        authority.kill()
        authority.wait()


def test_replacement_groups_hold_only_itself(monkeypatch):
    """Check that a replacement's groups have the deployment's layout with only itself active."""
    client = Mock()
    monkeypatch.setattr(startup_module._state, "client", client)
    world = StoreGroup(
        None, "w", 5, range(8), startup_module._report_startup_wait, active=[5], gpus_per_node=4
    )
    assert (world.Get_rank(), world.Get_size()) == (5, 8)
    assert world.allgather("v") == [None] * 5 + ["v", None, None]
    world.Barrier()
    node = world.Split_type(MPI.COMM_TYPE_SHARED)
    assert (node.membership, node.Get_rank(), node.Get_size()) == ((4, 5, 6, 7), 1, 4)
    assert node.allgather(True) == [None, True, None, None]
    assert node.allreduce(3, op=MPI.MIN) == 3
    assert [call.kwargs["progress"] for call in client.report.call_args_list] == [1, 2, 3, 4]


def test_replacement_startup_group_refuses_value_collectives(store, monkeypatch):
    """Check that a replacement's mpi_comm() refuses a collective of the others' values."""
    monkeypatch.setenv(FAULT_TOLERANCE_RANK_ENV, "2,4,4,1")
    monkeypatch.setenv("TLLM_RANK", "-")
    monkeypatch.setattr(_utils, "comm", _utils.comm)
    monkeypatch.setattr(_utils, "world_comm", _utils.world_comm)
    monkeypatch.setattr(_utils, "_local_comm", None)
    monkeypatch.setattr(Distributed, "_create", None)
    monkeypatch.setattr(Distributed, "startup_record", None)
    Distributed.get.cache_clear()
    record = startup_module._state.record
    record.replay({"values": {}, "exchanges": []})
    join_startup_group(
        Mapping(
            world_size=WORLD, rank=2, tp_size=WORLD, moe_ep_size=WORLD, enable_attention_dp=True
        )
    )
    world = _utils.mpi_comm()
    world.Barrier()
    for collective in (world.allgather, world.bcast, world.allreduce):
        with pytest.raises(ReplacementCannotStart, match="needs the original ranks' values"):
            collective(2)
    with Distributed.rank_varying("KV cache load sample"):
        assert world.allgather("sample") == [None, None, "sample", None]
    record.close()
    assert world.allreduce(2) == 2


@pytest.mark.parametrize(
    "grace, warned", [(None, False), ("-1", False), ("10", True)], ids=["unset", "off", "set"]
)
def test_startup_turns_off_rank_crash_kill(store, monkeypatch, grace, warned):
    """Check that a crashed rank kills no peer, since the membership removes a rank that ends."""
    monkeypatch.setenv(FAULT_TOLERANCE_RANK_ENV, "2,4,4,1")
    monkeypatch.setenv("TLLM_RANK", "-")
    if grace is None:
        monkeypatch.delenv(RANK_CRASH_KILL_GRACE_ENV, raising=False)
    else:
        monkeypatch.setenv(RANK_CRASH_KILL_GRACE_ENV, grace)
    monkeypatch.setattr(_utils, "comm", _utils.comm)
    monkeypatch.setattr(_utils, "world_comm", _utils.world_comm)
    monkeypatch.setattr(_utils, "_local_comm", None)
    monkeypatch.setattr(Distributed, "_create", None)
    monkeypatch.setattr(Distributed, "startup_record", None)
    Distributed.get.cache_clear()
    warning = Mock()
    monkeypatch.setattr(startup_module.logger, "warning", warning)
    record = startup_module._state.record
    record.replay({"values": {}, "exchanges": []})
    join_startup_group(
        Mapping(
            world_size=WORLD, rank=2, tp_size=WORLD, moe_ep_size=WORLD, enable_attention_dp=True
        )
    )
    record.close()
    kills = []
    monkeypatch.setattr(hang_detector, "propagate_hard_kill", lambda: kills.append(1))
    monkeypatch.setattr(hang_detector.time, "sleep", lambda s: None)
    assert os.environ[RANK_CRASH_KILL_GRACE_ENV] == "-1"
    assert start_rank_crash_kill_watchdog(WORLD) is None
    assert hard_kill_on_rank_crash(WORLD) is False
    assert kills == []
    named = [call for call in warning.call_args_list if RANK_CRASH_KILL_GRACE_ENV in call.args[0]]
    assert len(named) == int(warned)


@pytest.mark.parametrize(
    "requested, logged",
    [
        (CheckpointIOPolicy.RANK_STRIPED_READ_AHEAD, "warning"),
        (CheckpointIOPolicy.AUTO, "info"),
        (CheckpointIOPolicy.NATIVE, None),
    ],
)
def test_replacement_starts_alone(monkeypatch, requested, logged):
    """Check that a replacement loads its checkpoint alone and skips the KV cache estimation."""
    monkeypatch.setenv("TRTLLM_SKIP_KV_CACHE_ESTIMATION", "0")
    logs = {level: Mock() for level in ("warning", "info")}
    for level, log in logs.items():
        monkeypatch.setattr(startup_module.logger, level, log)
    llm_args = Mock(checkpoint_io_policy=requested, env_overrides=None)
    worker_main_args = {"llm_args": llm_args}
    startup_module._start_alone(worker_main_args)
    llm_args.model_copy.assert_called_once_with(
        update={"checkpoint_io_policy": CheckpointIOPolicy.NATIVE}
    )
    assert worker_main_args["llm_args"] is llm_args.model_copy.return_value
    assert os.environ["TRTLLM_SKIP_KV_CACHE_ESTIMATION"] == "1"
    for level, log in logs.items():
        if level == logged:
            message = log.call_args.args[0]
            assert f"requested={requested.value}, selected=native, reason=a replacement" in message
        else:
            log.assert_not_called()


def test_replacement_drops_autotuner_cache_override(monkeypatch):
    """Check that a replacement's environment overrides do not set the autotuner cache file again."""
    monkeypatch.setenv("TRTLLM_SKIP_KV_CACHE_ESTIMATION", "0")
    monkeypatch.setenv("TLLM_AUTOTUNER_CACHE_PATH", "/cache/autotuner.json")
    monkeypatch.setenv("TLLM_LOG_LEVEL", "info")
    llm_args = Mock(
        checkpoint_io_policy=CheckpointIOPolicy.NATIVE,
        env_overrides={
            "TLLM_AUTOTUNER_CACHE_PATH": "/cache/autotuner.json",
            "TLLM_LOG_LEVEL": "debug",
        },
    )
    startup_module._start_alone({"llm_args": llm_args})
    env_overrides = llm_args.model_copy.call_args.kwargs["update"]["env_overrides"]
    # As worker_main applies them.
    os.environ.update(env_overrides)
    assert "TLLM_AUTOTUNER_CACHE_PATH" not in os.environ
    assert os.environ["TLLM_LOG_LEVEL"] == "debug"


class _CountingStore:
    """In-memory TCPStore that records its calls, with a fake clock that a wait advances."""

    def __init__(self, late_rank, arrival_s):
        self.values = {}
        self.calls = []
        self.now_s = 0.0
        self._late_rank, self._arrival_s = late_rank, arrival_s

    def set(self, key, value):
        self.values[key] = value

    def get(self, key):
        return self.values[key]

    def add(self, key, amount):
        self.calls.append(("add", key, amount))
        return self.bump(key, amount)

    def bump(self, key, amount):
        """Add ``amount`` to ``key`` as another process does, without recording the call."""
        self.values[key] = self.values.get(key, 0) + amount
        return self.values[key]

    def wait(self, keys, timeout):
        self.calls.append(("wait", len(keys)))
        if self.now_s + timeout.total_seconds() < self._arrival_s:
            self.now_s += timeout.total_seconds()
            raise DistStoreError("wait timeout")
        self.now_s = max(self.now_s, self._arrival_s)
        _arrive(self, self._late_rank)

    def multi_get(self, keys):
        self.calls.append(("multi_get", len(keys)))
        return [self.values[key] for key in keys]

    def delete_key(self, key):
        return self.values.pop(key, None) is not None


def _arrive(store, rank):
    """Set ``rank``'s value of the first "rows" call and mark it arrived, as its StoreGroup does."""
    store.set(f"rows/0/{rank}", pickle.dumps(f"rows-{rank}"))
    store.bump(StoreGroup._counter("rows/0", rank), 1 << rank % 63)


@pytest.mark.parametrize("world", [4, 256])
def test_gather_waits_in_store(world):
    """Check that a waiting gather blocks in the store and reads its counters per timeout."""
    last = world - 1
    tcp_store = _CountingStore(last, arrival_s=1.2)
    for r in range(world):
        if r not in (1, last):
            _arrive(tcp_store, r)
    gathered = StoreGroup(tcp_store, "rows", 1, range(world)).allgather("rows-1")
    assert gathered == [f"rows-{r}" for r in range(world)]
    timeouts = int(1.2 // REPORT_INTERVAL_S)
    counters = -(-world // 63)
    counter_reads = [c for c in tcp_store.calls if c[0] == "add" and c[2] == 0]
    assert len(counter_reads) == counters * timeouts
    assert tcp_store.calls.count(("wait", world)) == timeouts + 1
    assert tcp_store.calls[-1] == ("multi_get", world)
    # Besides these, the rank only marks its own arrival.
    assert len(tcp_store.calls) == len(counter_reads) + timeouts + 3


def test_wait_returns_at_last_value(store):
    """Check that a wait returns when its last key is set, with no check in between."""
    setter = threading.Timer(0.2, _client(store).set, ("key", b"value"))
    setter.start()
    checks = []
    assert wait_for_keys(_client(store), ["key"], 30.0, lambda: checks.append(None)) == [b"value"]
    setter.join()
    assert checks == []


def _mirror(store_port, rank, steps):
    """Run the collectives of rank 1's peers in test_an_original_rank_starts_without_mpi."""
    world = StoreGroup(
        startup_module._startup_store(), "startup/world", rank, range(WORLD), gpus_per_node=WORLD
    )
    # The group of local_mpi_comm(). TpMnnvlMemory.get_comm() splits the TP group at its step.
    node = world.Split_type(OMPI_COMM_TYPE_HOST)
    for step in steps:
        {
            "barrier": world.barrier,
            "allgather": lambda: world.allgather(rank),
            "node_barrier": node.barrier,
            "node_min": lambda: node.allreduce(rank, op=MPI.MIN),
            "tp_allgather": lambda: world.Split(0, rank).allgather(rank),
        }[step]()


class _RecordingComm:
    """Record every attribute that an MPI communicator is asked for."""

    def __init__(self, comm, calls):
        self._comm, self._calls = comm, calls

    def __getattr__(self, name):
        self._calls.append(name)
        return getattr(self._comm, name)


def test_an_original_rank_starts_without_mpi(store, monkeypatch):
    """Check that rank 1 of 4, alone in its MPI world, reaches its peers through the store."""
    calls = []
    monkeypatch.setenv(FAULT_TOLERANCE_RANK_ENV, "1,4,4,0")
    monkeypatch.setenv("TLLM_RANK", "-")
    monkeypatch.setattr(_utils, "comm", _RecordingComm(_utils.comm, calls))
    monkeypatch.setattr(_utils, "world_comm", _utils.world_comm)
    monkeypatch.setattr(_utils, "_local_comm", None)
    monkeypatch.setattr(MPI, "COMM_WORLD", _RecordingComm(MPI.COMM_WORLD, calls))
    monkeypatch.setattr(TpMnnvlMemory, "comm", None)
    monkeypatch.setattr(TpMnnvlMemory, "comm_signature", None)
    monkeypatch.setattr(Distributed, "_create", None)
    monkeypatch.setattr(Distributed, "startup_record", None)
    Distributed.get.cache_clear()
    # Every rank on one node.
    mapping = Mapping(
        world_size=WORLD, rank=1, tp_size=WORLD, moe_ep_size=WORLD, enable_attention_dp=True
    )
    steps = ["barrier", "allgather", "node_barrier", "node_min", "tp_allgather"]
    peers = ThreadPoolExecutor(3)
    mirrored = [peers.submit(_mirror, store, rank, steps) for rank in (0, 2, 3)]

    join_startup_group(mapping)
    _utils.mpi_barrier()
    gathered = _utils.mpi_comm().allgather(1)
    _utils.local_mpi_barrier()
    least = _utils.local_mpi_comm().allreduce(1, op=MPI.MIN)
    tp_values = TpMnnvlMemory.get_comm(mapping).allgather(1)
    for peer in mirrored:
        peer.result(timeout=60)
    peers.shutdown()

    assert calls == []
    assert gathered == [0, 1, 2, 3]
    assert tp_values == gathered
    assert least == 0
    assert (_utils.mpi_rank(), _utils.mpi_world_size()) == (1, 4)
    assert (_utils.global_mpi_rank(), _utils.global_mpi_size()) == (1, 4)
    assert _utils.local_mpi_size() == 4
    assert startup_module.os.environ["TLLM_RANK"] == "1"
    assert _utils.mpi_comm() is _utils.world_comm
    assert isinstance(Distributed._create, MnnvlDistFactory)
    # The membership client reports the waits of the host collectives.
    assert startup_module._state.client.bind.call_args.args[1].args == (Distributed._create,)
    assert Distributed.startup_record is startup_module._state.record


@pytest.mark.parametrize(
    "world_size, gpus_per_node, message",
    [
        (8, 8, "gpus_per_node 4, the GPUs that the node supervisor sees"),
        (
            4,
            4,
            "a world size of 8, the ranks that the node supervisors start (2 nodes with 4 GPUs ",
        ),
    ],
    ids=["gpus_per_node", "world_size"],
)
def test_mapping_must_match_node_supervisors(monkeypatch, world_size, gpus_per_node, message):
    """Reject a mapping whose layout differs from the ranks that the node supervisors start."""
    monkeypatch.setenv(FAULT_TOLERANCE_RANK_ENV, "2,8,4,0")
    mapping = Mapping(
        world_size=world_size,
        rank=2,
        gpus_per_node=gpus_per_node,
        tp_size=world_size,
        moe_ep_size=world_size,
        enable_attention_dp=True,
    )
    with pytest.raises(ValueError, match=re.escape(message)) as error:
        join_startup_group(mapping)
    assert ("not the model's 4" in str(error.value)) == (world_size == 4)
    assert ("gpus_per_node" in str(error.value)) == (gpus_per_node == 8)


def test_original_rank_waits_for_every_rank(monkeypatch):
    """Check that the wait returns once a record past STARTING arrives on the client's thread."""
    starting = MembershipRecord(
        generation=0, state=MembershipState.STARTING, active_mask=(1 << WORLD) - 1
    )

    class Started(threading.Event):
        """The event of every rank's start, which sets ``entered`` once the waiter waits on it."""

        def __init__(self):
            super().__init__()
            self.entered = threading.Event()

        def wait(self, timeout=None):
            self.entered.set()
            return super().wait(timeout)

    client = Mock(record=starting, rank=1)
    started = Started()
    monkeypatch.setattr(startup_module._state, "client", client)
    monkeypatch.setattr(startup_module._state, "every_rank_started", started)
    waiter = threading.Thread(target=startup_module.wait_for_every_rank, daemon=True)
    waiter.start()
    startup_module._on_record_at_startup(starting)
    assert started.entered.wait(JOIN_TIMEOUT_S)
    assert waiter.is_alive() and not started.is_set()
    startup_module._on_record_at_startup(
        starting.model_copy(update={"state": MembershipState.COMMITTED})
    )
    waiter.join(JOIN_TIMEOUT_S)
    assert not waiter.is_alive()
    client.report.assert_called_once_with(started=True)


# The entry of each rank's worker, before its startup.


@pytest.fixture
def launched(monkeypatch):
    """Set the environment of rank 2 of 4 as trtllm-llmapi-launch starts it, with no authority."""
    monkeypatch.setenv(SHARED_SECRET_ENV, "ab" * 32)
    monkeypatch.setenv(RENDEZVOUS_ENV, "head-node:29600")
    monkeypatch.setenv(FAULT_TOLERANCE_RANK_ENV, "2,4,4,0")
    monkeypatch.setattr(startup_module, "_join_at_entry", lambda store, identity: None)


_SECRET_OR_RENDEZVOUS = f"shared secret in {SHARED_SECRET_ENV} and host:port in {RENDEZVOUS_ENV}"
_SHORT_SECRET = f"shared secret of at least 32 bytes (64 hex digits) in {SHARED_SECRET_ENV}"


@pytest.mark.parametrize(
    "env, fix",
    [
        ({RENDEZVOUS_ENV: None}, _SECRET_OR_RENDEZVOUS),
        ({RENDEZVOUS_ENV: "head-node"}, _SECRET_OR_RENDEZVOUS),
        ({RENDEZVOUS_ENV: "head-node:port"}, _SECRET_OR_RENDEZVOUS),
        ({SHARED_SECRET_ENV: None}, _SECRET_OR_RENDEZVOUS),
        # The authority admits only processes that hold the secret.
        ({SHARED_SECRET_ENV: "not hex"}, _SHORT_SECRET),
        ({SHARED_SECRET_ENV: ""}, _SHORT_SECRET),
        ({SHARED_SECRET_ENV: "ab" * 31}, _SHORT_SECRET),
    ],
)
def test_launch_rejects_bad_rendezvous(launched, monkeypatch, env, fix):
    """Name the fix for a missing or malformed shared secret or rendezvous address."""
    for name, value in env.items():
        if value is None:
            monkeypatch.delenv(name)
        else:
            monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=re.escape(fix)) as error:
        startup_module.worker_node_main()
    assert TURN_FAULT_TOLERANCE_OFF in str(error.value)


def test_launch_needs_rank_identity(launched, monkeypatch):
    """Name the rank identity format when a rank starts without it."""
    monkeypatch.delenv(FAULT_TOLERANCE_RANK_ENV)
    with pytest.raises(ValueError, match="'<rank>,<world size>,<GPUs per node>,<replacement>'"):
        startup_module.worker_node_main()


@pytest.mark.parametrize(
    "identity, address, refused",
    [
        ("2,8,4,0", "127.0.0.1:29600", True),
        # The Debian and Ubuntu /etc/hosts default for the host's own name.
        ("2,8,4,0", "127.0.1.1:29600", True),
        ("2,8,4,0", "192.0.2.1:29600", False),
        ("2,4,4,0", "127.0.0.1:29600", False),
    ],
    ids=["loopback", "debian_hostname", "remote", "one_node"],
)
def test_multi_node_launch_rejects_loopback_rendezvous(
    launched, monkeypatch, identity, address, refused
):
    """Refuse a loopback rendezvous address on several nodes, which the other nodes cannot reach."""
    monkeypatch.setenv(FAULT_TOLERANCE_RANK_ENV, identity)
    monkeypatch.setenv(RENDEZVOUS_ENV, address)
    if not refused:
        assert startup_module._launched_rank().world_size == int(identity.split(",")[1])
        return
    host, _ = rendezvous_host_port(address)
    with pytest.raises(ValueError, match=re.escape(f"resolves to the loopback address {host} ")):
        startup_module._launched_rank()


def test_multi_node_launch_rejects_unresolvable_rendezvous(launched, monkeypatch):
    """Name the rendezvous setting when its host does not resolve on a node of several."""

    def unresolvable(host):
        raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")

    monkeypatch.setenv(FAULT_TOLERANCE_RANK_ENV, "2,8,4,0")
    monkeypatch.setattr(startup_module.socket, "gethostbyname", unresolvable)
    with pytest.raises(
        ValueError, match=re.escape(f"Cannot resolve {RENDEZVOUS_ENV}=head-node:29600")
    ):
        startup_module._launched_rank()


@pytest.mark.parametrize(
    "orchestrator_type, mpi_size, spawn_proxy, message",
    [
        ("rpc", 1, False, "orchestrator_type unset"),
        ("ray", 1, False, "orchestrator_type unset"),
        (None, 2, False, "not one on every rank of an MPI job"),
        (None, 1, True, f"without {SHARED_SECRET_ENV} and {RENDEZVOUS_ENV} exported"),
    ],
    # Not "ray": the conftest skips a test whose id is "ray" unless Ray tests run.
    ids=["rpc_executor", "ray_executor", "mpi_job", "llmapi_launch_without_supervisor"],
)
def test_rendezvous_refuses_unsupported_launch(
    monkeypatch, orchestrator_type, mpi_size, spawn_proxy, message
):
    """Refuse another executor, a start on every rank of an MPI job, or ranks without the secret."""
    monkeypatch.delenv("TLLM_EXECUTOR_ATTACH_INFO", raising=False)
    monkeypatch.delenv(FAULT_TOLERANCE_RANK_ENV, raising=False)
    monkeypatch.setenv("TLLM_SPAWN_PROXY_PROCESS", "1" if spawn_proxy else "0")
    monkeypatch.setattr(startup_module, "global_mpi_size", lambda: mpi_size)
    with pytest.raises(ValueError, match=message):
        startup_module.ensure_rendezvous(orchestrator_type)


def test_attached_frontend_gets_no_rendezvous(monkeypatch):
    """Return no values to an attached frontend, which starts no rank, where the LLM sets "ray"."""
    monkeypatch.setenv("TLLM_EXECUTOR_ATTACH_INFO", "{}")
    assert startup_module.ensure_rendezvous("ray") == ({}, None)


@pytest.mark.parametrize("secret", [None, ""], ids=["unset", "empty"])
def test_rendezvous_port_reserved_until_shutdown(monkeypatch, secret):
    """Check that no other process takes the rendezvous port until the LLM's shutdown."""
    for name in (SHARED_SECRET_ENV, RENDEZVOUS_ENV):
        monkeypatch.delenv(name, raising=False)
    if secret is not None:
        monkeypatch.setenv(SHARED_SECRET_ENV, secret)
    llm = object.__new__(BaseLLM)
    env, llm._rendezvous_socket = startup_module.ensure_rendezvous(None)
    try:
        assert len(bytes.fromhex(env[SHARED_SECRET_ENV])) == 32
        _, port = rendezvous_host_port(env[RENDEZVOUS_ENV])
        with socket.socket() as other, pytest.raises(OSError):
            other.bind(("", port))
    finally:
        llm.shutdown()
    llm.shutdown()
    with socket.socket() as other:
        other.bind(("", port))


@pytest.mark.parametrize(
    "port, warned", [(29600, False), (32767, False), (32768, True), (60999, True), (61000, False)]
)
def test_exported_rendezvous_in_ephemeral_range_warns(monkeypatch, tmp_path, port, warned):
    """Check that an exported port in the ephemeral range is warned about, not refused."""
    port_range = tmp_path / "ip_local_port_range"
    port_range.write_text("32768\t60999\n")
    monkeypatch.setattr(startup_module, "_EPHEMERAL_PORT_RANGE", str(port_range))
    monkeypatch.setattr(startup_module.logger, "warning", warning := Mock())
    monkeypatch.setenv(SHARED_SECRET_ENV, "00" * 32)
    monkeypatch.setenv(RENDEZVOUS_ENV, f"node0:{port}")
    monkeypatch.delenv("TLLM_SPAWN_PROXY_PROCESS", raising=False)

    env, reserved = startup_module.ensure_rendezvous(None)

    assert (env[RENDEZVOUS_ENV], reserved) == (f"node0:{port}", None)
    assert warning.called == warned
    if warned:
        assert "outside net.ipv4.ip_local_port_range" in warning.call_args.args[0]


def test_single_node_authority_binds_loopback_only(monkeypatch):
    """Check that the authority of one node binds only the rendezvous port, on loopback only."""
    for name in (SHARED_SECRET_ENV, RENDEZVOUS_ENV):
        monkeypatch.delenv(name, raising=False)
    env, reserved = startup_module.ensure_rendezvous(None)
    with reserved:
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        host, port = rendezvous_host_port(env[RENDEZVOUS_ENV])
        assert host == "127.0.0.1"
        try:
            taken = socket.create_server(("127.0.0.1", port + 1))
        except OSError:
            taken = None  # Another process holds it already.
        process = startup_module._start_authority(world_size=1, gpus_per_node=1, timeout_s=60.0)
        try:
            store = open_store(env[RENDEZVOUS_ENV], timeout_s=60.0)
            _, liveness_port = rendezvous_host_port(store.get(LIVENESS_KEY).decode())
            assert liveness_port != port + 1
            # A listener on every interface would take a connection to another loopback address.
            for listened in (port, liveness_port):
                with pytest.raises(ConnectionRefusedError):
                    socket.create_connection(("127.0.0.2", listened), timeout=10).close()
        finally:
            process.kill()
            process.wait()
            if taken is not None:
                taken.close()


@pytest.mark.parametrize(
    "numa_aware_affinity", [None, "1", "0"], ids=["unset", "opted_in", "opted_out"]
)
def test_authority_removes_cpu_affinity_constraints(monkeypatch, numa_aware_affinity):
    """Run the authority on every CPU, though rank 0 starts it bound to one.

    With ``TLLM_NUMA_AWARE_WORKER_AFFINITY`` set to ``0``, the authority keeps the binding it
    inherits.
    """
    allowed = os.sched_getaffinity(0)
    if len(allowed) < 2:
        pytest.skip("needs two CPUs")
    for name in (SHARED_SECRET_ENV, RENDEZVOUS_ENV, "TLLM_NUMA_AWARE_WORKER_AFFINITY"):
        monkeypatch.delenv(name, raising=False)
    if numa_aware_affinity is not None:
        monkeypatch.setenv("TLLM_NUMA_AWARE_WORKER_AFFINITY", numa_aware_affinity)
    env, reserved = startup_module.ensure_rendezvous(None)
    with reserved:
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        bound = {min(allowed)}
        os.sched_setaffinity(0, bound)
        try:
            process = startup_module._start_authority(world_size=1, gpus_per_node=1, timeout_s=60.0)
        finally:
            os.sched_setaffinity(0, allowed)
        try:
            affinity = os.sched_getaffinity(process.pid)
            assert affinity == bound if numa_aware_affinity == "0" else affinity >= allowed
        finally:
            process.kill()
            process.wait()


@pytest.mark.parametrize("forger_secret", [None, "cd" * 32], ids=["unsigned", "other_secret"])
def test_forged_worker_args_are_never_unpickled(launched, monkeypatch, forger_secret):
    """Reject worker args that a process without the deployment's secret set, before unpickling them."""
    server = TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False)
    forger = _client(server.port)
    if forger_secret is not None:
        forger = SignedStore(forger, forger_secret)
    signed = open_store(f"127.0.0.1:{server.port}", timeout_s=30.0)
    monkeypatch.setattr(Unpickled, "calls", [])
    forged = pickle.dumps(Unpickled())
    forger.set(WORKER_ARGS_KEY, forged)
    monkeypatch.setattr(startup_module, "_startup_store", lambda: signed)
    with pytest.raises(ValueError, match="is not signed with this deployment's"):
        startup_module.worker_node_main()
    assert Unpickled.calls == []


def test_wait_for_keys_checks_while_blocked(store):
    """Check that a wait blocks in the store and checks once per interval until the key is set."""
    client = _client(store)
    checks = []

    def check():
        checks.append(None)
        if len(checks) == 2:
            client.set("key", b"value")

    assert wait_for_keys(client, ["key"], 0.2, check) == [b"value"]
    assert len(checks) == 2


def test_authority_starts_outside_launcher_job(monkeypatch):
    """Check that the authority does not inherit the launcher's rank identity."""
    for name in ("OMPI_COMM_WORLD_RANK", "PMIX_RANK", "PMI_RANK", "PRTE_LAUNCHED", "SLURM_PROCID"):
        monkeypatch.setenv(name, "0")
    launched = {}

    def popen(args, env, pass_fds, stdin):
        launched.update(env)
        os.write(pass_fds[0], b"host:1\n")
        return Mock()

    monkeypatch.setattr(startup_module.subprocess, "Popen", popen)
    startup_module._start_authority(world_size=4, gpus_per_node=4, timeout_s=120.0)
    assert not [
        name for name in launched if name.startswith(("OMPI_", "PMIX_", "PMI_", "PRTE_", "SLURM_"))
    ]
    assert launched["CUDA_VISIBLE_DEVICES"] == ""
    assert launched["MPI4PY_RC_INITIALIZE"] == "false"
    assert launched["TRT_LLM_NO_LIB_INIT"] == "1"  # Importing tensorrt_llm would run MPI_Init.


def _rank_0_args(**fields):
    """Publish rank 0's worker arguments, as start_rank does, and return them."""
    args = {"log_level": "info", "hf_model_dir": None, "llm_args": None, **fields}
    startup_module._startup_store().set(WORKER_ARGS_KEY, pickle.dumps(args))
    return args


class _LlmArgs:
    """Rank 0's LLM arguments, whose copies keep their update."""

    checkpoint_io_policy = CheckpointIOPolicy.NATIVE
    env_overrides = None

    def __init__(self, update=None):
        self.update = update

    def model_copy(self, update):
        return _LlmArgs(update)

    def __eq__(self, other):
        return isinstance(other, _LlmArgs) and self.update == other.update


def _tactics(rank):
    """Return the tuned tactics of ``rank``, whose MoE cache keys hold its expert offset."""
    return {("moe", "MoERunner", f"(local_expert_offset={rank},)", ((16,),)): (0, rank, 0.1)}


def _publish_tactics(monkeypatch):
    """Publish every rank's tuned tactics, as their startup does.

    Return the autotuner that a replacement of rank 2 then starts with.
    """
    for rank in range(WORLD):
        startup_module._state.record.tuned_tactics = _tactics(rank)
        startup_module.publish_tuned_tactics(startup_module._startup_store(), rank)
    tuner = Mock()
    monkeypatch.setattr(AutoTuner, "get", Mock(return_value=tuner))
    return tuner


def _start_rank_2(monkeypatch, replacement):
    """Start rank 2, or a replacement of it, up to its worker.

    Return rank 0's arguments and the arguments that the worker started with.
    """
    monkeypatch.setenv("TRTLLM_SKIP_KV_CACHE_ESTIMATION", "0")
    args = _rank_0_args(marker="rank 0", llm_args=_LlmArgs())
    if replacement:
        monkeypatch.setenv(FAULT_TOLERANCE_RANK_ENV, "2,4,4,1")
        agreed = {"values": {}, "exchanges": []}
        startup_module._startup_store().set(
            AGREED_STARTUP_KEY, pickle.dumps({**agreed, "fingerprint": startup_fingerprint(args)})
        )
    started = []
    monkeypatch.setattr(startup_module, "worker_main", lambda **args: started.append(args))
    startup_module.worker_node_main()
    return args, started


@pytest.mark.parametrize("replacement", [False, True])
def test_rank_takes_flag_from_rank0_args(store, launched, monkeypatch, replacement):
    """Check that a rank other than 0 runs with rank 0's arguments, which enable fault tolerance."""
    _publish_tactics(monkeypatch)
    args, started = _start_rank_2(monkeypatch, replacement)
    native = _LlmArgs({"checkpoint_io_policy": CheckpointIOPolicy.NATIVE})
    assert started == [{**args, "llm_args": native} if replacement else args]
    assert os.environ["TRTLLM_SKIP_KV_CACHE_ESTIMATION"] == ("1" if replacement else "0")


@pytest.mark.parametrize("replacement", [False, True])
def test_replacement_takes_its_predecessors_tactics(store, launched, monkeypatch, replacement):
    """Check that a replacement of rank 2 starts from rank 2's tactics and keeps them to publish.

    An original rank 2 takes none.
    """
    tuner = _publish_tactics(monkeypatch)
    _start_rank_2(monkeypatch, replacement)
    taken = [call.args for call in tuner.profiling_cache.merge_cache_data.call_args_list]
    assert taken == ([(_tactics(2),)] if replacement else [])
    if replacement:
        assert startup_module._state.record.tuned_tactics == _tactics(2)


def test_replacement_reports_replayed_collectives_as_progress(store, launched, monkeypatch):
    """Check that each startup collective a replacement replays counts as startup progress."""
    monkeypatch.setenv("TRTLLM_SKIP_KV_CACHE_ESTIMATION", "0")
    args = _rank_0_args(llm_args=_LlmArgs())
    original = StartupRecord()
    original.record()
    original.agree("max_seq_len", lambda: 8)
    original.exchange(0, pickle.dumps(1), None, lambda: [pickle.dumps(1)] * WORLD)
    original.close()
    startup_module._startup_store().set(
        AGREED_STARTUP_KEY,
        pickle.dumps({**original.published(), "fingerprint": startup_fingerprint(args)}),
    )
    _publish_tactics(monkeypatch)
    monkeypatch.setenv(FAULT_TOLERANCE_RANK_ENV, "2,4,4,1")

    def worker_main(**worker_main_args):
        record = startup_module._state.record
        assert record.agree("max_seq_len", lambda: 9) == 8
        record.exchange(2, pickle.dumps(1), None, lambda: [])
        record.close()

    monkeypatch.setattr(startup_module, "worker_main", worker_main)
    startup_module.worker_node_main()
    assert [call.kwargs for call in startup_module._state.client.report.call_args_list] == [
        dict(waiting_for=0, progress=1),
        dict(waiting_for=0, progress=2),
    ]


def test_mismatched_replacement_refuses_to_start(store, launched, monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text('{"num_hidden_layers": 2}')
    # The configuration comes from rank 0, so only the version and the checkpoint can differ.
    args = {"hf_model_dir": str(tmp_path), "llm_args": SimpleNamespace(max_batch_size=8)}
    fingerprint = startup_fingerprint(args)
    assert startup_fingerprint({**args, "llm_args": None}) == fingerprint
    startup_module._startup_store().set(
        AGREED_STARTUP_KEY, pickle.dumps({"fingerprint": fingerprint})
    )
    (tmp_path / "config.json").write_text('{"num_hidden_layers": 3}')
    _rank_0_args(**args)
    monkeypatch.setenv(FAULT_TOLERANCE_RANK_ENV, "2,4,4,1")
    ended = []
    monkeypatch.setattr(
        startup_module, "end_process", lambda rank, status, reason: ended.append((status, reason))
    )
    startup_module.worker_node_main()
    # Another replacement would not match either, so its supervisor starts none.
    assert [status for status, _ in ended] == [REPLACEMENT_CANNOT_START]
    assert "TensorRT-LLM version or checkpoint" in ended[0][1]


def test_rank0_publishes_args_without_ipc_keys(store, launched, monkeypatch):
    """Check that rank 0 starts the authority, publishes arguments without IPC keys, and joins."""
    from tensorrt_llm.executor.utils import WorkerCommIpcAddrs

    events = []
    authority = Mock()
    monkeypatch.setenv(FAULT_TOLERANCE_RANK_ENV, "0,8,4,0")
    # Two nodes, so the rendezvous host must resolve to an address that the other node reaches.
    monkeypatch.setenv(RENDEZVOUS_ENV, "192.0.2.1:29600")
    monkeypatch.setattr(
        startup_module,
        "_start_authority",
        lambda world_size, gpus_per_node, timeout_s: (
            events.append(("authority", world_size, gpus_per_node)) or authority
        ),
    )
    monkeypatch.setattr(
        startup_module,
        "_join_at_entry",
        lambda joined, identity: events.append(
            ("join", joined.check([WORKER_ARGS_KEY]), identity.rank)
        ),
    )
    queues = WorkerCommIpcAddrs(
        ("ipc://request", b"request-key"),
        ("ipc://status", b"status-key"),
        ("ipc://result", b"result-key"),
    )
    args = {
        "worker_queues": queues,
        "rpc_addr": "ipc://rpc",
        "hmac_key": b"rpc-key",
        "log_level": "info",
        "hf_model_dir": None,
        "llm_args": None,
    }
    startup_module.start_rank(args)
    assert events == [("authority", 8, 4), ("join", True, 0)]
    # Rank 0 hands its authority process to the executor, which waits for it to end at shutdown.
    assert startup_module.authority_process() is authority
    # Rank 0 records its startup for the replacements.
    record = startup_module._state.record
    assert record.agree("rank_hosts", lambda: ["a"] * WORLD) == ["a"] * WORLD
    assert record.values == {"rank_hosts": [["a"] * WORLD]}
    client = startup_module._startup_store()
    assert pickle.loads(client.get(WORKER_ARGS_KEY)) == {
        **args,
        "worker_queues": WorkerCommIpcAddrs(None, None, None),
        "rpc_addr": None,
        "hmac_key": b"",
    }
    publish_startup_record(client)
    assert pickle.loads(client.get(AGREED_STARTUP_KEY))["fingerprint"] == startup_fingerprint(args)


# The membership without rank 2, whose replacement starts.
_REMOVAL = MembershipRecord(generation=1, state=MembershipState.INSTALL, active_mask=0b1011)


@pytest.mark.parametrize("committed", [True, False])
def test_replacement_starts_after_removal_commit(store, monkeypatch, committed):
    """Check that a replacement starts only once a commit without its rank arrived."""
    records = [_REMOVAL]
    if committed:
        records.append(_REMOVAL.model_copy(update={"state": MembershipState.COMMITTED}))

    def membership_client(rank, store, on_record, **kwargs):
        return Mock(rank=rank, incarnation=1, start=lambda: [on_record(r) for r in records])

    monkeypatch.setattr(startup_module, "MembershipClient", membership_client)
    # The join waits no time, so it times out unless the records released the predecessor's memory.
    monkeypatch.setattr(startup_module, "RENDEZVOUS_TIMEOUT_S", 0.0)
    join = functools.partial(
        startup_module._join_at_entry, _client(store), RankIdentity(2, WORLD, WORLD, True)
    )
    if committed:
        join()
    else:
        with pytest.raises(TimeoutError, match="release its predecessor's memory"):
            join()
    assert startup_module._state.predecessor_memory_released.is_set() == committed
