# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process supervision across MPI hosts, independent of MPI progress."""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import logging
import os

# Only serializes reports; IpcQueue authenticates frames before deserialization.
import pickle  # nosec B403
import select
import socket

# Runs this interpreter and this helper with fixed argv, without a shell.
import subprocess  # nosec B404
import sys
import threading
import time
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path

_HEARTBEAT_INTERVAL = 0.5
_HEARTBEAT_TIMEOUT = 5.0
_RECEIVE_WAIT = 0.1
# A worker call that returns or raises reports its outcome through MPI, which
# arrives after the helper's final report. Defer to that outcome briefly.
_RETURN_GRACE = 2.0
_START_TIMEOUT = 10.0
WORKER_WATCH_HINT = (
    "Set TRTLLM_WORKER_WATCH_HOST to this proxy's address reachable from every MPI worker "
    "and allow worker-to-proxy TCP traffic through the firewall. "
    "TRTLLM_WORKER_WATCH_PORT selects a fixed listener port. "
    "TRTLLM_WORKER_WATCH=0 disables independent remote process supervision."
)
logger = logging.getLogger(__name__)


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
        import zmq

        from tensorrt_llm._errors import ConfigInputError, ResourceError, RuntimeInitError
        from tensorrt_llm._torch.disaggregation.native.utils import get_local_ip

        from .ipc import IpcQueue

        self._world_size = world_size
        self._wake = wake
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._seen: dict[int, tuple[dict, float]] = {}
        self._returned: dict[int, tuple[dict, float]] = {}
        self._failure: dict | None = None
        host = os.environ.get("TRTLLM_WORKER_WATCH_HOST") or get_local_ip()
        try:
            family, _, _, _, address = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)[0]
        except socket.gaierror as error:
            raise RuntimeInitError(
                f"Cannot resolve process-watch host {host!r}. {WORKER_WATCH_HINT}",
                hint=WORKER_WATCH_HINT,
            ) from error
        host = address[0]
        # Reject wildcard advertised addresses; this comparison does not bind.
        # Loopback is valid when every rank can report to this endpoint.
        if ipaddress.ip_address(host).is_unspecified:
            raise RuntimeInitError(
                f"Cannot advertise wildcard process-watch host {host!r}. {WORKER_WATCH_HINT}",
                hint=WORKER_WATCH_HINT,
            )
        port = os.environ.get("TRTLLM_WORKER_WATCH_PORT", "*")
        if port != "*":
            try:
                port_number = int(port)
            except ValueError as error:
                raise ConfigInputError(
                    "TRTLLM_WORKER_WATCH_PORT must be between 1 and 65535"
                ) from error
            if not 1 <= port_number <= 65535:
                raise ConfigInputError("TRTLLM_WORKER_WATCH_PORT must be between 1 and 65535")
            port = str(port_number)
        ipv6 = family == socket.AF_INET6
        # The advertised address may be forwarded from outside this namespace.
        # IpcQueue checks HMAC before unpickling; restrict ingress to MPI workers.
        bind_address = f"tcp://{'[::]' if ipv6 else '0.0.0.0'}:{port}"  # nosec B104
        try:
            self._queue = IpcQueue(
                (bind_address, None),
                is_server=True,
                socket_type=zmq.PULL,
                name="remote_worker_process_watch",
                receive_hwm=max(64, world_size * 4),
                ipv6=ipv6,
            )
        except zmq.ZMQError as error:
            raise ResourceError(
                f"Cannot bind process-watch listener {bind_address}. {WORKER_WATCH_HINT}",
                reason="port",
                hint=WORKER_WATCH_HINT,
            ) from error
        port = self._queue.address[0].rsplit(":", 1)[1]
        advertised_host = f"[{host}]" if ipv6 else host
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
        logger.error("Remote MPI worker process supervision failed: %s", report)
        if self._wake is not None:
            self._wake.set()

    def wait_ready(self, timeout: float = _START_TIMEOUT) -> bool:
        """Wait at most timeout seconds for authenticated reports from all ranks."""
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
        logger.error(
            "MPI ranks %s did not report to process-watch endpoint %s. %s",
            missing,
            self.endpoint.address,
            WORKER_WATCH_HINT,
        )
        return False

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
            if report.get("kind") == "returned":
                self._returned.setdefault(rank, (report, time.monotonic()))
        if report.get("kind") not in ("alive", "returned"):
            self._fail({**report, "reason": "worker exited"})

    def _run(self) -> None:
        import zmq

        try:
            while not self._stopped.is_set():
                waited_from = time.monotonic()
                self._queue.poll(_RECEIVE_WAIT)
                # Fix the lease check time before reading. The libzmq I/O
                # thread queues reports without the GIL, so after this thread
                # was paused, the earlier poll result is stale while reports
                # that arrived before ``now`` already wait in the socket.
                now = time.monotonic()
                # After a pause longer than the report interval, give the I/O
                # thread one receive interval to surface reports buffered while
                # the whole process was stopped.
                first_wait = (
                    _RECEIVE_WAIT
                    if now - waited_from > _RECEIVE_WAIT + _HEARTBEAT_INTERVAL
                    else 0
                )
                for index in range(max(64, self._world_size * 64)):
                    if self._stopped.is_set() or not self._queue.poll(
                        first_wait if index == 0 else 0
                    ):
                        break
                    try:
                        report = self._queue.get()
                    except RuntimeError as error:
                        if str(error) != "HMAC verification failed":
                            raise
                    else:
                        self._accept_report(report)
                # Check leases after each bounded drain, even if traffic keeps
                # arriving. A busy peer must not hide a silent rank forever.
                with self._lock:
                    returned = [
                        r for r, seen in self._returned.values() if now - seen > _RETURN_GRACE
                    ]
                    expired = [
                        r for r, seen in self._seen.values() if now - seen > _HEARTBEAT_TIMEOUT
                    ]
                if returned:
                    self._fail({**returned[0], "reason": "worker returned without an MPI result"})
                elif expired:
                    self._fail({**expired[0], "reason": "heartbeat timed out"})
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
    """Keep process liveness observable while the worker is in Python or CUDA."""

    def __init__(self, endpoint: ProcessWatchEndpoint, rank: int) -> None:
        """Start an independent helper and require its first report to connect.

        Args:
            endpoint: Proxy-owned authenticated process-watch listener.
            rank: MPI rank of the calling worker process.
        """
        self._process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--watch"],
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
            readable, _, _ = select.select([self._process.stdout], [], [], _START_TIMEOUT)
            if not readable or self._process.stdout.readline().strip() != "ready":
                from tensorrt_llm._errors import RuntimeInitError

                # The hint stays in the message for launchers that forward only text.
                raise RuntimeInitError(
                    f"MPI rank {rank} process watch failed to connect to {endpoint.address}. "
                    f"{WORKER_WATCH_HINT}",
                    hint=WORKER_WATCH_HINT,
                )
        except BaseException:
            self.close()
            raise

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
            process.stdin.close()
            process.stdout.close()


def supervise_worker_process(function):
    """Keep remote MPI worker supervision alive for the complete worker call."""

    @wraps(function)
    def supervised(*args, _process_watch=None, **kwargs):
        if _process_watch is None:
            return function(*args, **kwargs)
        from .._utils import mpi_rank

        watch = WorkerProcessWatch(_process_watch, mpi_rank())
        try:
            return function(*args, **kwargs)
        finally:
            watch.close()

    return supervised


def _watch_process() -> int:
    # This file is executed directly; it must not import TensorRT-LLM or MPI.
    import zmq

    config = json.loads(sys.stdin.readline())
    key = bytes.fromhex(config.pop("key"))
    report = {
        "rank": config["rank"],
        "pid": config["pid"],
        "host": socket.gethostname(),
        "sequence": 0,
        "kind": "alive",
    }
    with zmq.Context() as context, context.socket(zmq.PUSH) as channel:
        channel.setsockopt(zmq.IPV6, int(config.get("ipv6", False)))
        channel.setsockopt(zmq.IMMEDIATE, 1)
        channel.setsockopt(zmq.LINGER, 250)
        channel.setsockopt(zmq.SNDHWM, 4)
        channel.setsockopt(zmq.SNDTIMEO, 100)
        channel.connect(config["address"])
        deadline = time.monotonic() + _START_TIMEOUT - 1
        ready = False
        while True:
            if os.getppid() != config["pid"]:
                report["kind"] = "exited"
            try:
                # Match IpcQueue's authenticated framing without importing its
                # package (which would import the inference stack in this helper).
                payload = pickle.dumps(report)
                channel.send(payload + hmac.digest(key, payload, hashlib.sha256))
                if not ready:
                    print("ready", flush=True)
                    ready = True
            except zmq.Again:
                if not ready and time.monotonic() >= deadline:
                    return 1
            if report["kind"] != "alive":
                return 0
            report["sequence"] += 1
            readable, _, _ = select.select([sys.stdin], [], [], _HEARTBEAT_INTERVAL)
            if readable:
                # A stop request means the worker call ended in a live process;
                # end of input without one means the worker process is gone.
                stopped = sys.stdin.readline().strip() == "stop"
                report["kind"] = "returned" if stopped else "exited"


if __name__ == "__main__":
    if sys.argv[1:] != ["--watch"]:
        raise SystemExit("Expected --watch")
    raise SystemExit(_watch_process())
