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
"""Listening sockets and bound-address reporting for trtllm-serve.

Every listener that trtllm-serve binds takes its address family from
_listener_address_family(), so the policy and its reasons live in one place.
"""

import contextlib
import os
import socket
import tempfile
from typing import Literal, Optional

from tensorrt_llm.logger import logger

AddressFamilyPolicy = Literal["auto", "ipv4", "first"]


def _publish_bound_address(report_addr: Optional[str], host: str,
                           port: int) -> None:
    """Write the address this server actually bound to ``report_addr``.

    Lets a launcher pass ``--port 0`` and learn the kernel-assigned port
    afterwards, instead of picking a port up front and racing whoever grabs it
    before this process binds. The write is atomic (temp file in the same
    directory, then rename) so a reader never observes a partial line, which
    matters on the shared filesystems multi-node tests coordinate through.

    The caller is responsible for making the path unique per run: a stale file
    from an earlier run points at a dead server, which fails far less obviously
    than a port conflict.

    A wildcard bind host is replaced by this machine's hostname, since readers
    use the published value as a URL authority and cannot dial 0.0.0.0 or ::.
    """
    if not report_addr:
        return
    if host in ("0.0.0.0", "::", ""):  # nosec B104 - reporting, not binding
        resolved = socket.gethostname()
        logger.info(f"Reporting hostname {resolved} instead of wildcard bind "
                    f"address {host!r}, which a reader cannot dial")
        host = resolved
    report_addr = os.path.abspath(report_addr)
    parent = os.path.dirname(report_addr)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=parent or None,
                                    prefix=os.path.basename(report_addr) + ".",
                                    suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            # Bracket IPv6 literals so the value is a usable URL authority:
            # readers build "http://<reported>/..." from it verbatim.
            reported_host = f"[{host}]" if ":" in host else host
            f.write(f"{reported_host}:{port}\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, report_addr)
    finally:
        # A successful replace already consumed tmp_path; this only cleans up
        # after a failed write so a partial file is not left behind.
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
    logger.info(f"Reported bound address {host}:{port} to {report_addr}")


def _diagnose_port_in_use(port: int) -> str:
    """Describe which process currently holds the given port, best effort."""
    try:
        import psutil
    except ImportError:
        return "psutil unavailable; cannot identify the process holding the port"

    details = []
    try:
        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr and conn.laddr.port == port:
                holder = ""
                if conn.pid is not None:
                    try:
                        proc = psutil.Process(conn.pid)
                        holder = (f" pid={conn.pid} name={proc.name()} "
                                  f"cmdline={' '.join(proc.cmdline())}")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        holder = f" pid={conn.pid}"
                details.append(
                    f"{conn.laddr.ip}:{conn.laddr.port} status={conn.status}{holder}"
                )
    except (psutil.Error, OSError) as e:
        return f"failed to inspect port holders: {e}"

    if not details:
        return ("no listening socket found for this port; it may have already "
                "been released, indicating a transient race")
    return "; ".join(details)


def _bind_failure(kind: str, host: str, port: int,
                  error: OSError) -> RuntimeError:
    """Log a failed listener bind and return the exception to raise.

    An address lookup failure (``socket.gaierror``) is an ``OSError`` too, but
    no port was ever tried, so it is reported as such instead of being
    diagnosed as a port conflict. It covers both a name that does not resolve
    and an address of a family the socket cannot use, such as an IPv6 literal
    on an IPv4-only listener.
    """
    if isinstance(error, socket.gaierror):
        reason = f"address lookup failed for host {host!r}: {error}"
        logger.error(f"Failed to bind {kind} socket to {host}:{port} "
                     f"(pid={os.getpid()}): {reason}")
        return RuntimeError(f"Failed to bind socket to {host}:{port}: {reason}")
    holder = _diagnose_port_in_use(port)
    logger.error(
        f"Failed to bind {kind} socket to {host}:{port} (pid={os.getpid()}): "
        f"{error}. Current port holder(s): {holder}")
    return RuntimeError(f"Failed to bind socket to {host}:{port}: {error}. "
                        f"Port holder(s): {holder}")


def _listener_address_family(
        kind: str,
        host: str,
        port: int,
        policy: AddressFamilyPolicy = "auto") -> socket.AddressFamily:
    """The address family a trtllm-serve listener binds ``host`` with.

    The three policies differ on purpose; making them one is a behavior
    change decided on its own:

    - ``"auto"`` (the OpenAI HTTP server, the VisualGen server): IPv6 only if
      ``host`` resolves to IPv6 addresses only, otherwise IPv4, so a
      dual-stack host binds IPv4. A failed lookup is reported through
      _bind_failure().
    - ``"ipv4"`` (the disaggregated server, its coordinator, its fleet
      workers): always IPv4, without a lookup. When a hostname's AAAA record
      is link-local it resolves to fe80:: with scope id 0, which the kernel
      rejects with "invalid argument", because the scope can only come from
      an interface name.
    - ``"first"`` (OpenEngine's pre-start trial bind): the family of the
      first getaddrinfo() result; a failed lookup raises socket.gaierror.
    """
    if policy == "ipv4":
        return socket.AF_INET
    if policy == "first":
        return socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)[0][0]
    if policy != "auto":
        raise ValueError(f"Unknown address-family policy {policy!r}")
    try:
        addr_info = socket.getaddrinfo(host, port, socket.AF_UNSPEC,
                                       socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise _bind_failure(kind, host, port, e) from e
    if all(info[0] == socket.AF_INET6 for info in addr_info):
        return socket.AF_INET6
    return socket.AF_INET


def _bind_listener(kind: str,
                   host: str,
                   port: int,
                   *,
                   family: socket.AddressFamily,
                   reuse_port: bool = False) -> socket.socket:
    """Create a TCP socket, set its reuse options and bind it to (host, port).

    SO_REUSEADDR is always set: without it a restart is refused for the
    whole TIME_WAIT window (~60s) by the tombstones of connections this
    server accepted, and it has to be set on the socket that owns the port
    first, since the TIME_WAIT entry inherits it. SO_REUSEPORT is set when
    several processes serve one port. A failed bind is reported through
    _bind_failure(kind, ...). The socket is closed on any failure; on
    success the caller owns it (use it as a context manager or close it).
    """
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if reuse_port:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        try:
            sock.bind((host, port))
        except OSError as e:
            raise _bind_failure(kind, host, port, e) from e
    except BaseException:
        sock.close()
        raise
    return sock
