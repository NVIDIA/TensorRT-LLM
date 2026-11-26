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
"""Launch an isolated single-node etcd server for tests.

Each instance gets its own data directory and explicit client and peer listener
URLs so concurrent tests or unrelated host services on ports 2379/2380 do not
collide. Dynamic port allocation closes its probe socket before etcd binds, so
startup retries with fresh ports and a fresh data directory when etcd exits
before reporting healthy.
"""

import contextlib
import os
import shutil
import socket
import subprocess
import tempfile
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass

from test_common.http_utils import wait_for_endpoint_ready

__all__ = [
    "ETCD_HEALTH_POLL_INTERVAL_S",
    "ETCD_READY_TIMEOUT_S",
    "ETCD_START_ATTEMPTS",
    "ETCD_STOP_TIMEOUT_S",
    "LaunchedEtcd",
    "running_etcd",
]

ETCD_START_ATTEMPTS = 3
ETCD_READY_TIMEOUT_S = 30
ETCD_STOP_TIMEOUT_S = 10
ETCD_HEALTH_POLL_INTERVAL_S = 0.2
_DISTINCT_PORT_ATTEMPTS = 16


@dataclass(frozen=True)
class LaunchedEtcd:
    """Handle to a running test etcd instance."""

    process: subprocess.Popen
    client_port: int
    peer_port: int
    data_dir: str
    client_host: str = "127.0.0.1"

    @property
    def client_url(self) -> str:
        return f"http://{self.client_host}:{self.client_port}"

    @property
    def uri(self) -> str:
        return f"etcd://{self.client_host}:{self.client_port}"


def _bind_free_ports(count: int) -> list[int]:
    """Reserve ``count`` distinct ephemeral loopback ports via simultaneous bind."""
    sockets: list[socket.socket] = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def _allocate_distinct_port(
    exclude: int,
    port_allocator: Callable[[], int] | None,
) -> int:
    for _ in range(_DISTINCT_PORT_ATTEMPTS):
        port = port_allocator() if port_allocator is not None else _bind_free_ports(1)[0]
        if port != exclude:
            return port
    raise RuntimeError(f"Failed to allocate a port distinct from {exclude}")


def _resolve_ports(
    client_port: int | None,
    peer_port: int | None,
    port_allocator: Callable[[], int] | None,
) -> tuple[int, int]:
    if client_port is not None and peer_port is not None:
        if client_port == peer_port:
            raise ValueError("client_port and peer_port must be distinct")
        return client_port, peer_port

    if client_port is None and peer_port is None:
        if port_allocator is None:
            allocated_client, allocated_peer = _bind_free_ports(2)
            return allocated_client, allocated_peer
        allocated_client = port_allocator()
        allocated_peer = _allocate_distinct_port(allocated_client, port_allocator)
        return allocated_client, allocated_peer

    if client_port is None:
        assert peer_port is not None
        return _allocate_distinct_port(peer_port, port_allocator), peer_port

    return client_port, _allocate_distinct_port(client_port, port_allocator)


def _make_data_dir(data_dir_root: str | None) -> str:
    if data_dir_root is not None:
        return os.path.join(data_dir_root, f"disagg_test-etcd-{uuid.uuid4()}")
    return tempfile.mkdtemp(prefix="trtllm-etcd-")


def _stop_etcd(
    proc: subprocess.Popen,
    data_dir: str,
    stop_timeout: float = ETCD_STOP_TIMEOUT_S,
) -> None:
    try:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=stop_timeout)
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


def _health_probe_url(listen_client_host: str, client_port: int) -> str:
    probe_host = (
        "127.0.0.1" if listen_client_host in ("0.0.0.0", "localhost") else listen_client_host
    )
    return f"http://{probe_host}:{client_port}/health"


@contextlib.contextmanager
def running_etcd(
    *,
    data_dir_root: str | None = None,
    client_port: int | None = None,
    peer_port: int | None = None,
    listen_client_host: str = "127.0.0.1",
    advertise_client_host: str | None = None,
    port_allocator: Callable[[], int] | None = None,
    extra_args: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    cwd: str | None = None,
    ready_timeout: int = ETCD_READY_TIMEOUT_S,
    poll_interval: float = ETCD_HEALTH_POLL_INTERVAL_S,
    stop_timeout: float = ETCD_STOP_TIMEOUT_S,
    attempts: int = ETCD_START_ATTEMPTS,
) -> Iterator[LaunchedEtcd]:
    """Start a single-member etcd process, wait for health, and always clean up.

    Args:
        data_dir_root: Parent directory for the per-attempt etcd data directory.
            When None, a dedicated temporary directory is created and removed on
            teardown.
        client_port: Fixed client port, or None to allocate a free port.
        peer_port: Fixed peer port, or None to allocate a free loopback port.
        listen_client_host: Interface etcd binds for client traffic.
        advertise_client_host: Host advertised to clients and returned in
            ``LaunchedEtcd.uri``. Defaults to ``listen_client_host``.
        port_allocator: Optional zero-arg callable used to allocate dynamic
            ports (for example ``defs.common.get_free_port_in_ci``).
        extra_args: Additional CLI flags appended to the ``etcd`` invocation.
        env: Optional environment mapping passed to ``subprocess.Popen``.
        cwd: Optional working directory passed to ``subprocess.Popen``.
        ready_timeout: Seconds to wait for ``/health`` to return HTTP 200 per
            attempt.
        poll_interval: Seconds between ``/health`` probes.
        stop_timeout: Seconds to wait for ``etcd`` to exit on teardown.
        attempts: Maximum startup attempts when at least one port is dynamically
            allocated and etcd exits before becoming healthy.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    max_attempts = 1 if (client_port is not None and peer_port is not None) else attempts
    resolved_advertise_host = advertise_client_host or listen_client_host
    launched: LaunchedEtcd | None = None

    for attempt in range(1, max_attempts + 1):
        chosen_client_port, chosen_peer_port = _resolve_ports(
            client_port, peer_port, port_allocator
        )
        data_dir = _make_data_dir(data_dir_root)
        listen_client_url = f"http://{listen_client_host}:{chosen_client_port}"
        advertise_client_url = f"http://{resolved_advertise_host}:{chosen_client_port}"
        peer_url = f"http://127.0.0.1:{chosen_peer_port}"
        health_url = _health_probe_url(listen_client_host, chosen_client_port)
        cmd = [
            "etcd",
            "--data-dir",
            data_dir,
            "--listen-client-urls",
            listen_client_url,
            "--advertise-client-urls",
            advertise_client_url,
            "--listen-peer-urls",
            peer_url,
            "--initial-advertise-peer-urls",
            peer_url,
            "--initial-cluster",
            f"default={peer_url}",
            *(extra_args or ()),
        ]
        try:
            proc = subprocess.Popen(cmd, env=env, cwd=cwd)
        except BaseException:
            shutil.rmtree(data_dir, ignore_errors=True)
            raise

        try:
            wait_for_endpoint_ready(
                health_url,
                timeout=ready_timeout,
                server_proc=proc,
                poll_interval=poll_interval,
                trust_env=False,
                allow_redirects=False,
            )
        except RuntimeError:
            exited_early = proc.poll() is not None
            _stop_etcd(proc, data_dir, stop_timeout=stop_timeout)
            if exited_early and attempt < max_attempts:
                print(
                    f"etcd exited before becoming healthy on client={chosen_client_port} "
                    f"peer={chosen_peer_port} (attempt {attempt} of {max_attempts}); "
                    "retrying with fresh ports."
                )
                continue
            raise
        except BaseException:
            _stop_etcd(proc, data_dir, stop_timeout=stop_timeout)
            raise

        launched = LaunchedEtcd(
            process=proc,
            client_port=chosen_client_port,
            peer_port=chosen_peer_port,
            data_dir=data_dir,
            client_host=resolved_advertise_host,
        )
        break

    assert launched is not None
    try:
        yield launched
    finally:
        _stop_etcd(launched.process, launched.data_dir, stop_timeout=stop_timeout)
