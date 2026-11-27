# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process supervision across MPI hosts, independent of MPI progress."""

from __future__ import annotations

import errno
import inspect
import ipaddress
import json
import math
import os
import socket

# Runs this interpreter and the helper script with fixed argv, without a shell.
import subprocess  # nosec B404
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, TypeVar

import zmq

from .. import _utils
from .._errors import ConfigInputError, ResourceError, RuntimeInitError
from ..logger import logger
from . import process_watch_helper
from .hmac_framing import HmacVerificationError
from .ipc import IpcQueue
from .process_watch_helper import HEARTBEAT_INTERVAL as _HEARTBEAT_INTERVAL
from .process_watch_helper import START_TIMEOUT as _START_TIMEOUT
from .process_watch_helper import wait_readable as _wait_readable

# Default lease. TCP retransmission backoff (0.2, 0.4, 0.8, 1.6, 3.2 s, ...)
# leaves gaps, so a lease of N seconds tolerates network outages of only about
# N - 4 seconds. Worker process death is detected in about one interval
# regardless, because the helper notices its worker's exit directly. A helper
# that dies itself sends no report, so only the lease detects it.
_HEARTBEAT_TIMEOUT = 15.0
# The receiver wakes once per interval and drains every queued report, so its
# wakeups do not grow with the number of ranks or reports. Detection of a loss
# is delayed by at most one interval.
_RECEIVE_WAIT = 0.1
# Upper bound on one authenticated report frame; real reports are far smaller.
_MAX_REPORT_BYTES = 4096
# A worker call that returns or raises reports its outcome through MPI, which
# arrives after the helper's final report. Defer to that outcome briefly.
_RETURN_GRACE = 2.0
# MPI sessions start a task on all ranks together after a barrier. Each live rank
# starts its helper first in the worker call, and reports within _START_TIMEOUT or
# raises. A rank still missing this long after the first report exited before
# reporting, or reports elsewhere, while its peers wait in an init collective.
_REGISTRATION_TIMEOUT = 3 * _START_TIMEOUT
WORKER_WATCH_HINT = (
    "Set TRTLLM_WORKER_WATCH_HOST to this proxy's address reachable from every MPI worker "
    "and allow worker-to-proxy TCP traffic through the firewall. "
    "TRTLLM_WORKER_WATCH_PORT selects a fixed listener port. "
    "Unset TRTLLM_WORKER_WATCH (or set it to 0) to disable independent remote process "
    "supervision."
)


# Interfaces a deployment already configured for inter-node traffic, in order.
_INTERFACE_HINT_VARIABLES = (
    "NCCL_SOCKET_IFNAME",
    "GLOO_SOCKET_IFNAME",
    "OMPI_MCA_oob_tcp_if_include",
)


def _usable_ipv4(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    return ip.version == 4 and not (ip.is_loopback or ip.is_link_local or ip.is_unspecified)


def _interface_ipv4s() -> dict[str, list[str]]:
    """Return each interface's non-loopback, non-link-local IPv4 addresses."""
    import psutil

    return {
        name: [a.address for a in addrs if a.family == socket.AF_INET and _usable_ipv4(a.address)]
        for name, addrs in psutil.net_if_addrs().items()
    }


def _discover_host() -> tuple[str, str]:
    """Choose a worker-reachable IPv4 address and describe where it came from.

    Prefer an interface the deployment configured for inter-node traffic, then
    the default route, then any interface. Failed steps log at debug level only;
    the caller warns once if the result is loopback.
    """
    try:
        interfaces = _interface_ipv4s()
    except (ImportError, OSError) as error:
        logger.debug(f"Cannot list network interfaces for process watch: {error}")
        interfaces = {}
    for variable in _INTERFACE_HINT_VARIABLES:
        value = os.environ.get(variable, "").strip()
        # Exclusion lists ("^name") do not name an interface to use.
        if not value or value.startswith("^"):
            continue
        # NCCL matches interface-name prefixes unless the list starts with "=".
        prefix = variable == "NCCL_SOCKET_IFNAME" and not value.startswith("=")
        for name in value.lstrip("=").split(","):
            name = name.strip()
            # Skip subnet (CIDR) entries, which some MPI options also accept.
            if not name or "/" in name:
                continue
            for interface, addresses in interfaces.items():
                matches = interface.startswith(name) if prefix else interface == name
                if matches and addresses:
                    return addresses[0], f"{variable}={value}"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            # Connecting a UDP socket sends nothing; it only selects a route.
            probe.connect(("10.255.255.255", 1))
            address = probe.getsockname()[0]
        if _usable_ipv4(address):
            return address, "the default route"
    except OSError as error:
        logger.debug(f"Cannot find the default route for process watch: {error}")
    for interface, addresses in interfaces.items():
        if addresses:
            return addresses[0], f"interface {interface}"
    return "127.0.0.1", "no non-loopback interface"


def _resolve_host(host: str) -> tuple[int, str]:
    """Resolve host, preferring IPv4 and non-loopback results.

    A dual-stack name keeps the listener on IPv4, and a name whose first entry
    is a loopback alias (such as Debian's 127.0.1.1 hostname entry) uses a
    routable address when the name also has one.
    """
    infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)

    def preference(info):
        return (ipaddress.ip_address(info[4][0]).is_loopback, info[0] != socket.AF_INET)

    family, _, _, _, address = min(infos, key=preference)
    return family, address[0]


def _is_loopback_name(host: str | None) -> bool:
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _heartbeat_timeout_from_env() -> float:
    """Read TRTLLM_WORKER_WATCH_TIMEOUT, the lease in seconds, at construction."""
    value = os.environ.get("TRTLLM_WORKER_WATCH_TIMEOUT")
    if not value:
        return _HEARTBEAT_TIMEOUT
    minimum = 2 * _HEARTBEAT_INTERVAL
    try:
        timeout = float(value)
    except ValueError as error:
        raise ConfigInputError(
            f"TRTLLM_WORKER_WATCH_TIMEOUT must be a number of seconds greater than {minimum:g}"
        ) from error
    if not math.isfinite(timeout) or timeout <= minimum:
        raise ConfigInputError(
            f"TRTLLM_WORKER_WATCH_TIMEOUT must be a number of seconds greater than {minimum:g}"
        )
    return timeout


def _port_from_env() -> str:
    """Return TRTLLM_WORKER_WATCH_PORT, or "*" for an OS-assigned port when unset or empty."""
    value = os.environ.get("TRTLLM_WORKER_WATCH_PORT", "").strip()
    if not value:
        return "*"
    try:
        port = int(value)
    except ValueError:
        port = 0
    if not 1 <= port <= 65535:
        raise ConfigInputError(
            f"TRTLLM_WORKER_WATCH_PORT must be a port number from 1 through 65535, got "
            f"{value!r}. {WORKER_WATCH_HINT}",
            hint=WORKER_WATCH_HINT,
        )
    return str(port)


def start_remote_monitor(
    world_size: int, wake: threading.Event | None = None
) -> RemoteWorkerProcessMonitor | None:
    """Start remote supervision when TRTLLM_WORKER_WATCH=1 opts in.

    Args:
        world_size: Number of MPI worker ranks required for readiness.
        wake: Event to set when a worker or its supervision is lost.

    Returns:
        The started monitor, or None when TRTLLM_WORKER_WATCH is unset, empty
        or 0.

    Raises:
        ConfigInputError: If TRTLLM_WORKER_WATCH is anything other than 0 or
            1, or another TRTLLM_WORKER_WATCH_* setting is invalid.
    """
    switch = os.environ.get("TRTLLM_WORKER_WATCH", "").strip()
    if switch not in ("", "0", "1"):
        raise ConfigInputError(
            "TRTLLM_WORKER_WATCH must be 1 to enable remote worker process supervision "
            f"or 0 to disable it, got {switch!r}"
        )
    if switch == "1":
        return RemoteWorkerProcessMonitor(world_size, wake)
    configured = [
        name
        for name in ("TRTLLM_WORKER_WATCH_HOST", "TRTLLM_WORKER_WATCH_PORT")
        if os.environ.get(name)
    ]
    if configured:
        logger.warning(
            f"{' and '.join(configured)} is set but remote worker process supervision is "
            "disabled; set TRTLLM_WORKER_WATCH=1 to enable it"
        )
    return None


def describe_failure(report: dict) -> str:
    """Describe a supervision failure report for logs and fatal errors."""
    if report.get("missing"):
        return (
            f"MPI worker ranks {report['missing']} did not report to process-watch endpoint "
            f"{report['endpoint']}: {report['reason']}"
        )
    if report["rank"] < 0:
        return f"Remote MPI worker process supervision failed: {report['reason']}"
    return (
        f"MPI worker rank {report['rank']} (pid {report['pid']}) lost process liveness "
        f"on {report['host']}: {report['reason']}"
    )


@dataclass(frozen=True)
class ProcessWatchEndpoint:
    """Authenticated listener address and address family sent to each worker."""

    address: str
    key: bytes = field(repr=False)
    ipv6: bool = False


class RemoteWorkerProcessMonitor:
    """Receive authenticated process-watch reports on a dedicated socket owner."""

    def __init__(self, world_size: int, wake: threading.Event | None = None) -> None:
        """Start the proxy listener selected by the worker-watch environment.

        Args:
            world_size: Number of MPI worker ranks required for readiness.
            wake: Event to set when a worker or its supervision is lost.
        """
        self._world_size = world_size
        self._wake = wake
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        # Each rank's newest report and when it arrived. A "returned" report is
        # always a helper's last one.
        self._seen: dict[int, tuple[dict, float]] = {}
        self._first_report_at: float | None = None
        self._failure: dict | None = None
        self._heartbeat_timeout = _heartbeat_timeout_from_env()
        port = _port_from_env()
        override = os.environ.get("TRTLLM_WORKER_WATCH_HOST", "").strip()
        if override:
            host, source = override, "TRTLLM_WORKER_WATCH_HOST"
        else:
            host, source = _discover_host()
        try:
            family, host = _resolve_host(host)
        except socket.gaierror as error:
            if override:
                brackets = (
                    " Supply IPv6 literals without brackets." if override.startswith("[") else ""
                )
                raise ConfigInputError(
                    f"Cannot resolve TRTLLM_WORKER_WATCH_HOST={override!r}: {error}."
                    f"{brackets} {WORKER_WATCH_HINT}",
                    hint=WORKER_WATCH_HINT,
                ) from error
            raise RuntimeInitError(
                f"Cannot resolve process-watch host {host!r} from {source}. {WORKER_WATCH_HINT}",
                hint=WORKER_WATCH_HINT,
            ) from error
        # An explicit loopback choice is valid when every rank shares this
        # network namespace. Anything else that ends at loopback is usually a
        # discovery or /etc/hosts problem that fails startup on other hosts.
        if ipaddress.ip_address(host).is_loopback and not _is_loopback_name(override):
            logger.warning(
                f"Process-watch listener advertises loopback address {host} (from {source}); "
                "ranks on other hosts cannot reach it. Set TRTLLM_WORKER_WATCH_HOST to an "
                "address every MPI rank can reach."
            )
        # Reject wildcard advertised addresses; this comparison does not bind.
        # Loopback is valid when every rank can report to this endpoint.
        if ipaddress.ip_address(host).is_unspecified:
            error_type = ConfigInputError if override else RuntimeInitError
            raise error_type(
                f"Cannot advertise wildcard process-watch host {host!r} from {source}. "
                f"{WORKER_WATCH_HINT}",
                hint=WORKER_WATCH_HINT,
            )
        ipv6 = family == socket.AF_INET6
        advertised_host = f"[{host}]" if ipv6 else host
        queue_options = dict(
            is_server=True,
            socket_type=zmq.PULL,
            name="remote_worker_process_watch",
            # Several queued reports per rank while the receiver is paused.
            receive_hwm=max(64, world_size * 4),
            ipv6=ipv6,
            # libzmq drops a connection that announces a larger frame before
            # allocating it. Reports are a few hundred bytes.
            socket_options={zmq.MAXMSGSIZE: _MAX_REPORT_BYTES},
        )
        # Listen only on the advertised address. IpcQueue checks HMAC before
        # unpickling; restrict ingress to MPI workers.
        bind_address = f"tcp://{advertised_host}:{port}"
        try:
            try:
                self._queue = IpcQueue((bind_address, None), **queue_options)
            except zmq.ZMQError as error:
                if error.errno != errno.EADDRNOTAVAIL:
                    raise
                # The advertised address is not on this host, for example a
                # forwarded or translated address from TRTLLM_WORKER_WATCH_HOST.
                bind_address = f"tcp://{'[::]' if ipv6 else '0.0.0.0'}:{port}"  # nosec B104
                logger.warning(
                    f"Process-watch host {host} is not a local address; listening on all "
                    f"interfaces at {bind_address}. Restrict ingress to MPI workers."
                )
                self._queue = IpcQueue((bind_address, None), **queue_options)
        except zmq.ZMQError as error:
            raise ResourceError(
                f"Cannot bind process-watch listener {bind_address}. {WORKER_WATCH_HINT}",
                reason="port",
                hint=WORKER_WATCH_HINT,
            ) from error
        port = self._queue.address[0].rsplit(":", 1)[1]
        self.endpoint = ProcessWatchEndpoint(
            f"tcp://{advertised_host}:{port}", self._queue.address[1], ipv6
        )
        self._thread = threading.Thread(
            target=self._run, name="remote_worker_process_watch", daemon=True
        )
        self._thread.start()

    @property
    def failure(self) -> dict | None:
        """Return the first worker/helper loss report, or None while healthy."""
        with self._lock:
            return self._failure

    def _fail(self, report: dict) -> None:
        with self._lock:
            if self._stopped.is_set() or self._failure is not None:
                return
            self._failure = report
        hint = f" {WORKER_WATCH_HINT}" if report.get("missing") else ""
        logger.error(f"{describe_failure(report)}{hint}")
        if self._wake is not None:
            self._wake.set()

    def wait_ready(self, timeout: float = _START_TIMEOUT) -> bool:
        """Wait at most timeout seconds for authenticated reports from all ranks.

        A timeout records a failure that names the missing ranks and this
        listener's endpoint.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._failure is not None:
                    return False
                if len(self._seen) == self._world_size:
                    return True
            if self._stopped.wait(0.01):
                return False
        with self._lock:
            missing = sorted(set(range(self._world_size)) - self._seen.keys())
        if missing:
            self._fail(self._missing_report(missing, f"no report within {timeout:g} s"))
        return False

    def _missing_report(self, missing: list[int], reason: str) -> dict:
        # No process identity is known for a rank that never reported.
        return {
            "rank": missing[0],
            "pid": 0,
            "host": "unknown",
            "missing": missing,
            "endpoint": self.endpoint.address,
            "reason": reason,
        }

    def _accept_report(self, report: dict) -> None:
        if not isinstance(report, dict):
            return
        rank, sequence = report.get("rank"), report.get("sequence")
        if type(rank) is not int or not 0 <= rank < self._world_size:
            return
        if type(sequence) is not int or sequence < 0:
            return
        if type(report.get("pid")) is not int or not isinstance(report.get("host"), str):
            return
        with self._lock:
            previous = self._seen.get(rank)
            if previous is not None and sequence <= previous[0]["sequence"]:
                return
            self._seen[rank] = (report, time.monotonic())
            if self._first_report_at is None:
                self._first_report_at = self._seen[rank][1]
        if report.get("kind") not in ("alive", "returned"):
            self._fail({**report, "reason": "worker exited"})

    def _check_leases(self, now: float) -> None:
        """Fail on the first returned, expired or never-registered rank at ``now``."""
        with self._lock:
            returned = [
                report
                for report, seen in self._seen.values()
                if report.get("kind") == "returned" and now - seen > _RETURN_GRACE
            ]
            expired = [
                report
                for report, seen in self._seen.values()
                if report.get("kind") != "returned" and now - seen > self._heartbeat_timeout
            ]
            first = self._first_report_at
            unregistered = (
                sorted(set(range(self._world_size)) - self._seen.keys())
                if first is not None and now - first > _REGISTRATION_TIMEOUT
                else []
            )
        if returned:
            self._fail({**returned[0], "reason": "worker returned without an MPI result"})
        elif expired:
            self._fail({**expired[0], "reason": "heartbeat timed out"})
        elif unregistered:
            self._fail(
                self._missing_report(
                    unregistered,
                    f"no report within {_REGISTRATION_TIMEOUT:g} s of the first rank's report",
                )
            )

    def _run(self) -> None:
        try:
            while True:
                waited_from = time.monotonic()
                if self._stopped.wait(_RECEIVE_WAIT):
                    break
                # Fix the lease check time before reading. The libzmq I/O
                # thread queues reports without the GIL, so reports that
                # arrived before ``now`` already wait in the socket.
                now = time.monotonic()
                # After a pause longer than the report interval, give the I/O
                # thread one receive interval to surface reports buffered while
                # the whole process was stopped.
                first_wait = (
                    _RECEIVE_WAIT if now - waited_from > _RECEIVE_WAIT + _HEARTBEAT_INTERVAL else 0
                )
                # At least the receive HWM, so one pass empties a normal
                # backlog; bounded so the lease check below still runs.
                for index in range(max(64, self._world_size * 64)):
                    if self._stopped.is_set() or not self._queue.poll(
                        first_wait if index == 0 else 0
                    ):
                        break
                    try:
                        report = self._queue.get()
                    except HmacVerificationError:
                        continue
                    self._accept_report(report)
                # Check leases after each bounded drain, even if traffic keeps
                # arriving. A busy peer must not hide a silent rank forever.
                self._check_leases(now)
        except Exception as error:
            # Losing the supervisor is itself a terminal loss of worker coverage.
            self._fail(
                {"rank": -1, "pid": 0, "host": "process-watch receiver", "reason": str(error)}
            )
        finally:
            self._queue.socket.setsockopt(zmq.LINGER, 0)
            self._queue.close()

    def close(self) -> None:
        """Stop the receiver and close its listener without waiting for workers."""
        self._stopped.set()
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=2)


class WorkerProcessWatch:
    """Keep process liveness observable while the worker is in Python or CUDA.

    Use as a context manager, or call close(), to stop the helper.
    """

    def __init__(self, endpoint: ProcessWatchEndpoint, rank: int) -> None:
        """Start an independent helper and require its first report to connect.

        Args:
            endpoint: Proxy-owned authenticated process-watch listener.
            rank: MPI rank of the calling worker process.
        """
        self._process = subprocess.Popen(
            [sys.executable, process_watch_helper.__file__, "--watch"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        config = {
            "address": endpoint.address,
            "key": endpoint.key.hex(),
            "rank": rank,
            "pid": os.getpid(),
            "ipv6": endpoint.ipv6,
        }
        try:
            self._process.stdin.write(json.dumps(config) + "\n")
            self._process.stdin.flush()
            readable = _wait_readable(self._process.stdout, _START_TIMEOUT)
            if not readable or self._process.stdout.readline().strip() != "ready":
                address_host = endpoint.address[len("tcp://") :].rsplit(":", 1)[0].strip("[]")
                loopback = (
                    " That is a loopback address, reachable only from the proxy's network "
                    "namespace."
                    if _is_loopback_name(address_host)
                    else ""
                )
                # The hint stays in the message for launchers that forward only text.
                raise RuntimeInitError(
                    f"MPI rank {rank} process watch failed to connect to {endpoint.address}."
                    f"{loopback} {WORKER_WATCH_HINT}",
                    hint=WORKER_WATCH_HINT,
                )
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> WorkerProcessWatch:
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()

    def close(self) -> None:
        """Stop and reap this worker's helper, escalating after a bounded wait."""
        process = self._process
        try:
            if process.poll() is None:
                try:
                    process.stdin.write("stop\n")
                    process.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        finally:
            for stream in (process.stdin, process.stdout):
                try:
                    stream.close()
                except OSError:
                    # Closing flushes a "stop" left buffered by a failed write
                    # to a helper that already exited (BrokenPipeError). That
                    # must not replace the error that is closing this watch.
                    pass


_T = TypeVar("_T")


def supervise_worker_process(function: Callable[..., _T]) -> Callable[..., _T]:
    """Add a ``process_watch_endpoint`` keyword argument that supervises the call.

    With an endpoint, every MPI rank starts a WorkerProcessWatch, the ranks
    agree on the outcome before calling ``function``, and the watch stays open
    for the whole call. Without one, ``function`` runs unsupervised. The
    parameter appears in the wrapper's signature.
    """

    @wraps(function)
    def supervised(
        *args: Any, process_watch_endpoint: ProcessWatchEndpoint | None = None, **kwargs: Any
    ) -> _T:
        if process_watch_endpoint is None:
            return function(*args, **kwargs)
        watch, error = None, None
        try:
            watch = WorkerProcessWatch(process_watch_endpoint, _utils.mpi_rank())
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        # Agree on the outcome before the worker's first collective. A rank
        # that raised here alone would leave its peers waiting in that
        # collective, and the session could then not shut down.
        errors = _utils.mpi_comm().allgather(error)
        failed = [rank for rank, message in enumerate(errors) if message is not None]
        if failed:
            if watch is not None:
                watch.close()
            # Every rank raises the same error, so whichever rank's result the
            # caller reads first reports the root cause.
            raise RuntimeInitError(
                f"Process watch failed to start on MPI ranks {failed}: {errors[failed[0]]}",
                hint=WORKER_WATCH_HINT,
            )
        with watch:
            return function(*args, **kwargs)

    signature = inspect.signature(function)
    parameters = list(signature.parameters.values())
    endpoint_parameter = inspect.Parameter(
        "process_watch_endpoint",
        inspect.Parameter.KEYWORD_ONLY,
        default=None,
        annotation=ProcessWatchEndpoint | None,
    )
    position = len(parameters)
    if parameters and parameters[-1].kind == inspect.Parameter.VAR_KEYWORD:
        position -= 1
    parameters.insert(position, endpoint_parameter)
    supervised.__signature__ = signature.replace(parameters=parameters)
    return supervised


def describe_lost_worker(identity, remote_failure: dict | None = None) -> str:
    """Describe a worker returned by WorkerProcessMonitor.find_dead_worker().

    Args:
        identity: The lost worker's WorkerProcessIdentity.
        remote_failure: WorkerProcessMonitor.remote_failure. When it describes
            this worker, the result adds the reporting host and the reason.
    """
    if isinstance(remote_failure, dict) and (
        remote_failure.get("rank"),
        remote_failure.get("pid"),
    ) == (identity.rank, identity.pid):
        return describe_failure(remote_failure)
    return f"MPI worker rank {identity.rank} (pid {identity.pid}) lost process liveness"
