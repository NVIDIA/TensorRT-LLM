# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""torch.distributed under the MPI orchestrator, and the MoE expert-parallel process group.

Kernels that exchange tokens over symmetric memory (MegaMoE) need a torch.distributed process
group for their build-time rendezvous. The MPI orchestrator (trtllm-serve, trtllm-bench,
mpirun) never initializes torch.distributed, so the first such builder bootstraps it.
Collective: every rank calls these at the same point of model construction.
"""

from __future__ import annotations

import ipaddress
import os
import socket

import psutil
import torch
import torch.distributed as dist

from tensorrt_llm.logger import logger

# Seconds a rank may take to reach rank 0's rendezvous address before the bootstrap fails. The
# store itself would retry silently for 30 minutes.
_PROBE_TIMEOUT_S = 60.0
# Interface lists the job may set, in order of preference: comma-separated names; NCCL matches
# a name as a prefix ("=" for exact, a leading "^" excludes), UCX writes a device as dev:port.
_IFNAME_VARS = (
    "GLOO_SOCKET_IFNAME",
    "NCCL_SOCKET_IFNAME",
    "UCX_NET_DEVICES",
    "OMPI_MCA_btl_tcp_if_include",
)


def _free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return str(sock.getsockname()[1])


def _rendezvous_address() -> tuple[str, str]:
    """``(host, port)`` rank 0 publishes: its MASTER_ADDR / MASTER_PORT, else a free local port."""
    addr = os.environ.get("MASTER_ADDR")
    port = os.environ.get("MASTER_PORT")
    if addr and port:
        return addr, port
    port = _free_port()
    try:
        addr = socket.gethostbyname(socket.gethostname())
    except OSError:
        # An unresolvable hostname (minimal containers): the ranks share this host.
        addr = "127.0.0.1"
    return addr, port


def _routable(addr: str) -> bool:
    """Whether another host can reach the IPv4 address ``addr``."""
    try:
        ip = ipaddress.IPv4Address(addr)
    except ValueError:
        return False
    return not (ip.is_loopback or ip.is_link_local or ip.is_unspecified)


def _local_ipv4() -> list[tuple[str, str]]:
    """``(interface, address)`` for each routable IPv4 address of this host."""
    return [
        (name, addr.address)
        for name, addrs in psutil.net_if_addrs().items()
        for addr in addrs
        if addr.family == socket.AF_INET and _routable(addr.address)
    ]


def _default_route_ipv4() -> str | None:
    """The source address the routing table picks for an off-host destination."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("192.0.2.1", 9))  # TEST-NET-1; connecting a UDP socket sends nothing
            return sock.getsockname()[0]
    except OSError:  # no route
        return None


def _fqdn_ipv4() -> list[str]:
    """The IPv4 addresses this host's fully qualified name resolves to."""
    try:
        return [info[4][0] for info in socket.getaddrinfo(socket.getfqdn(), None, socket.AF_INET)]
    except OSError:
        return []


def _named_interfaces() -> list[str]:
    """The interface names the job's settings list, in ``_IFNAME_VARS`` order."""
    names = []
    for var in _IFNAME_VARS:
        value = os.environ.get(var, "")
        if value.startswith("^"):
            continue
        for token in value.split(","):
            name = token.strip().lstrip("=").split(":")[0]
            if name and name not in names:
                names.append(name)
    return names


def _pick_interface() -> tuple[str, str, str] | None:
    """``(interface, IPv4 address, how it was chosen)`` of this host for the rendezvous, or None.

    The first interface the job's settings name (exactly, else as a prefix), else the default
    route's, else the one holding the FQDN's address. Never loopback or link-local.
    """
    local = _local_ipv4()
    for want in _named_interfaces():
        match = [x for x in local if x[0] == want] or [x for x in local if x[0].startswith(want)]
        if match:
            return (*match[0], "named by the job")
    for how, wanted in (("default route", [_default_route_ipv4()]), ("FQDN", _fqdn_ipv4())):
        for name, addr in local:
            if addr in wanted:
                return name, addr, how
    return None


def _offer(spans_hosts: bool, world: int) -> tuple[tuple, socket.socket | None]:
    """What rank 0 broadcasts, ``(host, port, probe port, error)``, and its probe listener.

    With MASTER_ADDR / MASTER_PORT set, or all ranks on this host, the address is published as
    is and nothing is probed. Otherwise loopback cannot serve (a Debian-style /etc/hosts maps the
    host's own name to 127.0.1.1): rank 0 publishes a routable address and listens on a probe
    port every rank must reach.
    """
    if not spans_hosts or (os.environ.get("MASTER_ADDR") and os.environ.get("MASTER_PORT")):
        return (*_rendezvous_address(), None, None), None
    picked = _pick_interface()
    if picked is None:
        return (
            None,
            None,
            None,
            f"the MPI ranks span several hosts and rank 0's host {socket.gethostname()} has no "
            f"routable IPv4 address (interfaces named by the job: {_named_interfaces()}); set "
            f"MASTER_ADDR / MASTER_PORT, or NCCL_SOCKET_IFNAME to the interface to use",
        ), None
    name, addr, how = picked
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(("", 0))
        # Connections complete in the accept queue; nothing needs to accept them.
        listener.listen(world)
        offer = (addr, _free_port(), listener.getsockname()[1], None)
    except OSError as e:
        listener.close()
        return (None, None, None, f"rank 0 cannot open a probe port ({e!r})"), None
    logger.info(f"torch.distributed bootstrap: rank 0 publishes {addr} ({name}, {how})")
    return offer, listener


def _check_reachable(comm, hosts: list[str], addr: str, probe_port: int) -> None:
    """Pin Gloo to a routable interface unless GLOO_SOCKET_IFNAME is set, and connect to rank 0's
    probe port. Collective: every rank raises, naming the ranks that failed, if any did."""
    problems = []
    if not os.environ.get("GLOO_SOCKET_IFNAME"):
        picked = _pick_interface()
        if picked is None:
            problems.append("no routable interface for Gloo")
        else:
            # Unset, Gloo binds to the address the host name resolves to. Gloo reads the
            # variable in this process when the group is created, so each rank names its own.
            os.environ["GLOO_SOCKET_IFNAME"] = picked[0]
    try:
        with socket.create_connection((addr, probe_port), timeout=_PROBE_TIMEOUT_S):
            pass
    except OSError as e:
        problems.append(f"cannot connect to {addr}:{probe_port} ({e!r})")
    failed = [
        f"rank {r} on {hosts[r]}: {', '.join(p)}"
        for r, p in enumerate(comm.allgather(problems))
        if p
    ]
    if failed:
        raise RuntimeError(
            f"torch.distributed bootstrap: rank 0 on {hosts[0]} published {addr}, which not every "
            f"rank can use (probe timeout {_PROBE_TIMEOUT_S:.0f} s): {'; '.join(failed)}. Set "
            f"MASTER_ADDR / MASTER_PORT to an address every rank reaches, or NCCL_SOCKET_IFNAME / "
            f"GLOO_SOCKET_IFNAME to the interface to use"
        )


def ensure_torch_dist_under_mpi(group_size: int) -> None:
    """Initialize a torch.distributed NCCL / Gloo WORLD group over the MPI ranks, if none exists.

    A no-op when torch.distributed is already initialized (Ray, torchrun), unavailable, or the job
    has one rank. ``group_size`` is the group the caller exchanges over; it must be the whole MPI
    world, so that every rank reaches this call (checked before any collective, so a mismatch
    raises on every rank instead of hanging). Rank 0 picks the rendezvous address and broadcasts
    it over MPI; the NCCL group binds to the device this rank already uses. When the ranks span
    several hosts and rank 0 has no MASTER_ADDR / MASTER_PORT, the address is a routable one,
    each rank checks it can connect to it within ``_PROBE_TIMEOUT_S``, and Gloo is pinned to a
    routable interface (see ``_offer``).
    """
    if not dist.is_available() or dist.is_initialized():
        return
    from tensorrt_llm._utils import mpi_comm, mpi_rank, mpi_world_size

    try:
        world, rank = mpi_world_size(), mpi_rank()
    except Exception as e:  # not under MPI either: leave it uninitialized
        logger.debug(f"torch.distributed bootstrap skipped: no MPI rank ({e!r})")
        return
    if world <= 1:
        return
    if world != group_size:
        raise ValueError(
            f"torch.distributed bootstrap needs the whole MPI world in the group: MPI world "
            f"{world}, group {group_size}"
        )
    comm = mpi_comm()
    hosts = comm.allgather(socket.gethostname())
    offer, listener = _offer(len(set(hosts)) > 1, world) if rank == 0 else (None, None)
    try:
        host, port, probe_port, error = comm.bcast(offer, root=0)
        if error is not None:
            raise RuntimeError(f"torch.distributed bootstrap: {error}")
        if probe_port is not None:
            _check_reachable(comm, hosts, host, probe_port)
    finally:
        if listener is not None:
            listener.close()
    os.environ["MASTER_ADDR"] = str(host)
    os.environ["MASTER_PORT"] = str(port)
    device_id = (
        torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else None
    )
    logger.info(
        f"torch.distributed not initialized under MPI; bootstrapping the NCCL WORLD group "
        f"(rank {rank}/{world}, device {device_id}, {host}:{port})"
    )
    dist.init_process_group(
        backend="cuda:nccl,cpu:gloo", rank=rank, world_size=world, device_id=device_id
    )


def moe_ep_process_group(mapping) -> dist.ProcessGroup:
    """The torch.distributed group of ``mapping``'s MoE expert-parallel ranks.

    Bootstraps torch.distributed under MPI first. The mapping's own EP group where it has one
    (DeviceMesh), else WORLD when the EP group is the whole world; anything else raises.
    """
    ensure_torch_dist_under_mpi(mapping.moe_ep_size)
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "the MoE expert-parallel group needs torch.distributed (mpirun or Ray) for "
            f"{mapping.moe_ep_size} ranks"
        )
    try:
        return mapping.moe_ep_group_pg
    except (NotImplementedError, AttributeError):
        pass
    if mapping.moe_ep_size == dist.get_world_size():
        return dist.group.WORLD
    raise RuntimeError(
        f"no process group for the MoE expert-parallel group: the mapping has none and EP "
        f"({mapping.moe_ep_size}) is a strict subset of the world ({dist.get_world_size()})"
    )
