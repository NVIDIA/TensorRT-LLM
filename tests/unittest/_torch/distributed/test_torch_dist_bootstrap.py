# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The torch.distributed bootstrap under MPI: which rendezvous address rank 0 publishes.

Each MPI rank is a thread of this process with its own environment, host name and network
interfaces; bcast and allgather meet at a barrier (a stand-in ``tensorrt_llm._utils`` serves the
MPI rank, size and communicator). Every fake host name resolves to 127.0.1.1,
as on hosts whose /etc/hosts maps their own name there. The reachability probe connects for real
to rank 0's listener (a fake host address is dialed as 127.0.0.1); ``init_process_group`` is
recorded, not run.
"""

from __future__ import annotations

import socket
import sys
import threading
from collections.abc import MutableMapping
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from tensorrt_llm._torch.distributed import torch_dist_bootstrap as boot

pytestmark = pytest.mark.cpu_only

_A, _B = "node-a", "node-b"
_NETS = {
    _A: {"lo": "127.0.0.1", "eth0": "10.0.0.1"},
    _B: {"lo": "127.0.0.1", "enp2": "10.0.0.2"},
}
_ROUTES = {_A: "10.0.0.1", _B: "10.0.0.2"}


class _World:
    """In-process MPI ranks (threads) on fake hosts; mpi4py's object bcast / allgather."""

    def __init__(self, hosts, nets, envs=None, routes=None, fqdn=None):
        self.hosts = hosts
        self.size = len(hosts)
        self.nets = nets
        self.routes = routes or {}
        self.fqdn = fqdn or {}
        self.envs = [dict(env) for env in (envs or [{}] * self.size)]
        self.initialized = False
        self.unreachable = set()  # hosts whose connections to rank 0 time out
        self.inits = {}  # rank -> init_process_group kwargs and the env it saw
        self.probes = []  # (rank, address, timeout)
        self._local = threading.local()
        self._barrier = threading.Barrier(self.size, timeout=60)
        self._slots = [None] * self.size

    @property
    def rank(self):
        return self._local.rank

    @property
    def host(self):
        return self.hosts[self.rank]

    def allgather(self, obj):
        self._barrier.wait()
        self._slots[self.rank] = obj
        self._barrier.wait()
        return list(self._slots)

    def bcast(self, obj, root=0):
        return self.allgather(obj)[root]

    def run(self, fn):
        """``fn()`` on every rank: each rank's return value or exception."""
        out = [None] * self.size

        def body(rank):
            self._local.rank = rank
            try:
                out[rank] = fn()
            except Exception as e:
                out[rank] = e

        threads = [threading.Thread(target=body, args=(r,), daemon=True) for r in range(self.size)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)
        assert not any(t.is_alive() for t in threads), "a rank hung"
        return out


class _RankEnv(MutableMapping):
    """``os.environ`` of the calling rank."""

    def __init__(self, world):
        self._world = world

    def _env(self):
        return self._world.envs[self._world.rank]

    def __getitem__(self, key):
        return self._env()[key]

    def __setitem__(self, key, value):
        self._env()[key] = value

    def __delitem__(self, key):
        del self._env()[key]

    def __iter__(self):
        return iter(self._env())

    def __len__(self):
        return len(self._env())


class _RankSocket:
    """The ``socket`` module as the calling rank's host sees it."""

    def __init__(self, world):
        self._world = world

    def __getattr__(self, name):
        return getattr(socket, name)

    def gethostname(self):
        return self._world.host

    def gethostbyname(self, name):
        return "127.0.1.1"

    def create_connection(self, address, timeout=None):
        world = self._world
        addr, port = address
        world.probes.append((world.rank, addr, timeout))
        if world.host in world.unreachable:
            raise TimeoutError("timed out")
        assert addr in {a for net in world.nets.values() for a in net.values()}, addr
        return socket.create_connection(("127.0.0.1", port), timeout=timeout)


@pytest.fixture
def make_world(monkeypatch):
    def make(hosts, nets=_NETS, **kwargs):
        world = _World(hosts, nets, **kwargs)

        def net_if_addrs():
            return {
                name: [SimpleNamespace(family=socket.AF_INET, address=addr)]
                for name, addr in world.nets[world.host].items()
            }

        def init_process_group(**kw):
            env = world.envs[world.rank]
            seen = {k: env.get(k) for k in ("MASTER_ADDR", "MASTER_PORT", "GLOO_SOCKET_IFNAME")}
            world.inits[world.rank] = {**kw, **seen}

        monkeypatch.setattr(boot, "os", SimpleNamespace(environ=_RankEnv(world)))
        monkeypatch.setattr(boot, "socket", _RankSocket(world))
        monkeypatch.setattr(boot, "psutil", SimpleNamespace(net_if_addrs=net_if_addrs))
        monkeypatch.setattr(boot, "_default_route_ipv4", lambda: world.routes.get(world.host))
        monkeypatch.setattr(boot, "_fqdn_ipv4", lambda: world.fqdn.get(world.host, []))
        monkeypatch.setattr(dist, "is_initialized", lambda: world.initialized)
        monkeypatch.setattr(dist, "init_process_group", init_process_group)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        mpi = ModuleType("tensorrt_llm._utils")
        mpi.mpi_comm = lambda: world
        mpi.mpi_rank = lambda: world.rank
        mpi.mpi_world_size = lambda: world.size
        monkeypatch.setitem(sys.modules, "tensorrt_llm._utils", mpi)
        return world

    return make


def _bootstrap(world):
    return world.run(lambda: boot.ensure_torch_dist_under_mpi(world.size))


def test_two_hosts_publish_a_routable_address_and_pin_gloo(make_world):
    """Not the 127.0.1.1 the host name resolves to; each rank pins Gloo to its own interface."""
    world = make_world([_A, _A, _B, _B], routes=_ROUTES)
    assert _bootstrap(world) == [None] * 4
    inits = [world.inits[r] for r in range(4)]
    assert {i["MASTER_ADDR"] for i in inits} == {"10.0.0.1"}
    assert len({i["MASTER_PORT"] for i in inits}) == 1
    assert [i["GLOO_SOCKET_IFNAME"] for i in inits] == ["eth0", "eth0", "enp2", "enp2"]
    assert [(i["rank"], i["world_size"], i["backend"]) for i in inits] == [
        (r, 4, "cuda:nccl,cpu:gloo") for r in range(4)
    ]
    assert all("timeout" not in i for i in inits)
    assert sorted(world.probes) == [(r, "10.0.0.1", boot._PROBE_TIMEOUT_S) for r in range(4)]


def test_the_interface_the_job_names_comes_first(make_world):
    """NCCL_SOCKET_IFNAME's interface (matched as a prefix, as NCCL does) over the default route."""
    nets = {
        _A: {"lo": "127.0.0.1", "eth0": "10.0.0.1", "ib0": "10.1.0.1"},
        _B: {"lo": "127.0.0.1", "eth0": "10.0.0.2", "ib0": "10.1.0.2"},
    }
    envs = [{"NCCL_SOCKET_IFNAME": "ib"}, {"NCCL_SOCKET_IFNAME": "ib"}]
    world = make_world([_A, _B], nets, envs=envs, routes={_A: "10.0.0.1", _B: "10.0.0.2"})
    assert _bootstrap(world) == [None, None]
    assert [world.inits[r]["MASTER_ADDR"] for r in range(2)] == ["10.1.0.1", "10.1.0.1"]
    assert [world.inits[r]["GLOO_SOCKET_IFNAME"] for r in range(2)] == ["ib0", "ib0"]


def test_the_fqdn_serves_when_there_is_no_default_route(make_world):
    world = make_world([_A, _B], fqdn={_A: ["127.0.1.1", "10.0.0.1"], _B: ["10.0.0.2"]})
    assert _bootstrap(world) == [None, None]
    assert [world.inits[r]["MASTER_ADDR"] for r in range(2)] == ["10.0.0.1", "10.0.0.1"]
    assert [world.inits[r]["GLOO_SOCKET_IFNAME"] for r in range(2)] == ["eth0", "enp2"]


def test_two_hosts_refuse_loopback(make_world):
    """Rank 0's host has only loopback: every rank raises before any probe or init."""
    nets = {_A: {"lo": "127.0.0.1", "link": "169.254.0.7"}, _B: _NETS[_B]}
    world = make_world([_A, _B], nets, routes={_B: "10.0.0.2"}, fqdn={_A: ["127.0.1.1"]})
    out = _bootstrap(world)
    assert all(isinstance(e, RuntimeError) for e in out), out
    assert all("no routable IPv4" in str(e) and _A in str(e) for e in out)
    assert not world.probes and not world.inits


def test_ranks_that_cannot_reach_rank0_are_named(make_world):
    """A probe that times out fails every rank at once, naming the ranks that could not connect."""
    world = make_world([_A, _A, _B, _B], routes=_ROUTES)
    world.unreachable = {_B}
    out = _bootstrap(world)
    assert all(isinstance(e, RuntimeError) for e in out), out
    message = str(out[0])
    assert all(str(e) == message for e in out)
    assert "rank 2 on node-b" in message and "rank 3 on node-b" in message
    assert "rank 0 on node-a published 10.0.0.1" in message and "rank 1 on" not in message
    assert "TimeoutError" in message
    assert {timeout for _, _, timeout in world.probes} == {boot._PROBE_TIMEOUT_S}
    assert not world.inits


def test_a_rank_without_a_routable_interface_for_gloo_is_named(make_world):
    nets = {_A: _NETS[_A], _B: {"lo": "127.0.0.1"}}
    world = make_world([_A, _B], nets, routes={_A: "10.0.0.1"})
    out = _bootstrap(world)
    assert all(isinstance(e, RuntimeError) for e in out), out
    assert "rank 1 on node-b: no routable interface for Gloo" in str(out[0])
    assert not world.inits


def test_one_host_uses_the_host_name_address(make_world):
    """All ranks on one host: the host name's address, no probe, Gloo left alone."""
    world = make_world([_A, _A], routes=_ROUTES)
    assert _bootstrap(world) == [None, None]
    assert [world.inits[r]["MASTER_ADDR"] for r in range(2)] == ["127.0.1.1", "127.0.1.1"]
    assert [world.inits[r]["GLOO_SOCKET_IFNAME"] for r in range(2)] == [None, None]
    assert not world.probes


def test_rank0_master_addr_and_port_are_used_untouched(make_world):
    envs = [{"MASTER_ADDR": "10.9.9.9", "MASTER_PORT": "29511"}, {}]
    world = make_world([_A, _B], envs=envs, routes=_ROUTES)
    assert _bootstrap(world) == [None, None]
    seen = [
        (i["MASTER_ADDR"], i["MASTER_PORT"], i["GLOO_SOCKET_IFNAME"]) for i in world.inits.values()
    ]
    assert seen == [("10.9.9.9", "29511", None)] * 2
    assert not world.probes


def test_an_initialized_group_or_a_single_rank_is_left_alone(make_world):
    world = make_world([_A, _B], routes=_ROUTES)
    world.initialized = True
    assert _bootstrap(world) == [None, None]
    single = make_world([_A], routes=_ROUTES)
    assert _bootstrap(single) == [None]
    assert not world.inits and not single.inits and not world.probes


def test_a_group_smaller_than_the_world_raises_on_every_rank(make_world):
    world = make_world([_A, _B], routes=_ROUTES)
    out = world.run(lambda: boot.ensure_torch_dist_under_mpi(1))
    assert all(isinstance(e, ValueError) for e in out), out
    assert not world.probes and not world.inits


@pytest.mark.parametrize(
    "addr, routable",
    [
        ("10.0.0.1", True),
        ("127.0.1.1", False),
        ("127.0.0.1", False),
        ("0.0.0.0", False),
        ("169.254.3.1", False),
        ("fe80::1", False),
        ("node-a", False),
    ],
)
def test_routable(addr, routable):
    assert boot._routable(addr) is routable


@pytest.mark.parametrize(
    "env, names",
    [
        ({"NCCL_SOCKET_IFNAME": "^lo,docker"}, []),
        (
            {"NCCL_SOCKET_IFNAME": "=eth1", "UCX_NET_DEVICES": "mlx5_0:1,enP13p65s0f0np0"},
            ["eth1", "mlx5_0", "enP13p65s0f0np0"],
        ),
        ({"GLOO_SOCKET_IFNAME": "ib0", "OMPI_MCA_btl_tcp_if_include": "ib0,eth0"}, ["ib0", "eth0"]),
    ],
)
def test_named_interfaces(monkeypatch, env, names):
    monkeypatch.setattr(boot, "os", SimpleNamespace(environ=env))
    assert boot._named_interfaces() == names


def test_this_hosts_lookups_return_addresses_or_nothing():
    """The real lookups on this host: no exception, and only routable interface addresses."""
    assert all(boot._routable(addr) for _, addr in boot._local_ipv4())
    route = boot._default_route_ipv4()
    assert route is None or isinstance(route, str)
    assert all(isinstance(addr, str) for addr in boot._fqdn_ipv4())
