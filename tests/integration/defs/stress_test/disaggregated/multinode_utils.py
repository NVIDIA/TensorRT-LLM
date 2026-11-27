# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded coordination for two-node stress tests with one pytest task per node."""

import hashlib
import re
import select
import socket
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager


def _expand_nodelist(nodelist: str) -> list[str]:
    """Expand Slurm's comma-separated hosts and bracketed numeric ranges."""
    groups = re.split(r",(?![^\[]*\])", nodelist)
    hosts = []
    for group in groups:
        match = re.fullmatch(r"(.+?)\[([\d,-]+)\]", group)
        if match is None:
            if not group or "[" in group or "]" in group:
                raise ValueError(f"Unsupported Slurm node list: {nodelist}")
            hosts.append(group)
            continue
        prefix, suffixes = match.groups()
        for suffix in suffixes.split(","):
            start, separator, end = suffix.partition("-")
            if separator:
                if not start.isdigit() or not end.isdigit() or int(end) < int(start):
                    raise ValueError(f"Invalid Slurm node range: {suffix}")
                hosts.extend(
                    f"{prefix}{number:0{len(start)}d}" for number in range(int(start), int(end) + 1)
                )
            elif suffix.isdigit():
                hosts.append(f"{prefix}{suffix}")
            else:
                raise ValueError(f"Invalid Slurm node suffix: {suffix}")
    return hosts


def slurm_topology(environ: Mapping[str, str]) -> tuple[int, list[str]]:
    """Validate the active Slurm step and resolve its two hostnames."""
    expected = {"SLURM_STEP_NUM_NODES": "2", "SLURM_STEP_NUM_TASKS": "2", "SLURM_LOCALID": "0"}
    for name, value in expected.items():
        if environ.get(name) != value:
            raise ValueError(
                f"Requires two Slurm nodes and one task per node: {name} must be {value}"
            )
    rank = int(environ["SLURM_NODEID"])
    if rank not in (0, 1) or int(environ["SLURM_PROCID"]) != rank:
        raise ValueError("Slurm process rank must equal node rank (0 or 1)")
    hosts = _expand_nodelist(environ["SLURM_STEP_NODELIST"])
    if len(hosts) != 2 or len(set(hosts)) != 2:
        raise ValueError(f"Expected two distinct Slurm step hosts, got {hosts}")
    return rank, hosts


def wait_for_signal(
    connection: socket.socket,
    expected: bytes,
    timeout: float,
    check: Callable[[], None],
) -> None:
    """Wait for one protocol message while checking the locally owned processes."""
    deadline = time.monotonic() + timeout
    received = bytearray()
    while len(received) < len(expected):
        check()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Peer did not send {expected!r} within {timeout}s")
        if not select.select([connection], [], [], min(1.0, remaining))[0]:
            continue
        chunk = connection.recv(len(expected) - len(received))
        if not chunk:
            raise RuntimeError(f"Peer exited before sending {expected!r}")
        received.extend(chunk)
    if received != expected:
        raise RuntimeError(f"Expected peer message {expected!r}, received {bytes(received)!r}")


def check_peer(connection: socket.socket) -> None:
    """Fail if the peer task exits or sends an unexpected message during load."""
    if select.select([connection], [], [], 0)[0]:
        message = connection.recv(1, socket.MSG_PEEK)
        raise RuntimeError(f"Peer task stopped during stress work (message={message!r})")


@contextmanager
def peer_connection(
    rank: int, host: str, port: int, test_id: str, timeout: float = 300
) -> Iterator[socket.socket]:
    """Connect both tasks before model loading, verifying they selected the same test."""
    deadline = time.monotonic() + timeout
    identity = hashlib.sha256(test_id.encode()).digest()
    if rank == 0:
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("0.0.0.0", port))
            listener.listen(1)
            listener.settimeout(timeout)
            connection, _ = listener.accept()
    else:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Could not connect to coordinator at {host}:{port}")
            try:
                connection = socket.create_connection((host, port), timeout=min(5, remaining))
                break
            except (ConnectionRefusedError, TimeoutError):
                time.sleep(min(1, remaining))
    with connection:
        connection.settimeout(5)
        connection.sendall(identity)
        wait_for_signal(connection, identity, max(0, deadline - time.monotonic()), lambda: None)
        connection.settimeout(5)
        yield connection
