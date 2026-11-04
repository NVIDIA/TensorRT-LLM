# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process supervision across MPI hosts, independent of MPI progress."""

from __future__ import annotations

import hashlib
import hmac
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
_START_TIMEOUT = 10.0
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcessWatchEndpoint:
    address: str
    key: bytes = field(repr=False)


class RemoteWorkerProcessMonitor:
    """Receive authenticated process-watch reports on a dedicated socket owner."""

    def __init__(self, world_size: int, wake: threading.Event | None = None) -> None:
        import zmq

        from .ipc import IpcQueue

        self._world_size = world_size
        self._wake = wake
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._seen: dict[int, tuple[dict, float]] = {}
        self._failure: dict | None = None
        host_override = os.environ.get("TRTLLM_WORKER_WATCH_HOST")
        host = socket.gethostbyname(host_override or socket.gethostname())
        # Reject wildcard advertised addresses; this comparison does not bind.
        if (
            host == "0.0.0.0"  # nosec B104
            or (host.startswith("127.") and not host_override)
        ):
            raise RuntimeError(
                "Set TRTLLM_WORKER_WATCH_HOST to a leader address reachable from all MPI workers"
            )
        self._queue = IpcQueue(
            ("tcp://0.0.0.0:*", None),
            is_server=True,
            socket_type=zmq.PULL,
            name="remote_worker_process_watch",
            receive_hwm=max(64, world_size * 4),
        )
        port = self._queue.address[0].rsplit(":", 1)[1]
        self.endpoint = ProcessWatchEndpoint(f"tcp://{host}:{port}", self._queue.address[1])
        self._thread = threading.Thread(
            target=self._run, name="remote_worker_process_watch", daemon=True
        )
        self._thread.start()

    @property
    def failure(self) -> dict | None:
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
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._failure is not None:
                    return False
                if len(self._seen) == self._world_size:
                    return True
            if self._stopped.wait(0.01):
                return False
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
        if report.get("kind") != "alive":
            self._fail({**report, "reason": "worker exited"})

    def _run(self) -> None:
        import zmq

        try:
            while not self._stopped.is_set():
                readable = self._queue.poll(0.1)
                # After a receiver scheduling pause, other ranks may already
                # have fresh reports waiting. Read them before expiring leases.
                for _ in range(max(64, self._world_size * 64)):
                    if not readable or self._stopped.is_set():
                        break
                    try:
                        report = self._queue.get()
                    except RuntimeError as error:
                        if str(error) != "HMAC verification failed":
                            raise
                    else:
                        self._accept_report(report)
                    readable = self._queue.poll(0)
                if readable:
                    continue
                now = time.monotonic()
                with self._lock:
                    expired = [
                        r for r, seen in self._seen.values() if now - seen > _HEARTBEAT_TIMEOUT
                    ]
                if expired:
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
        self._stopped.set()
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=2)


class WorkerProcessWatch:
    """Keep process liveness observable while the worker is in Python or CUDA."""

    def __init__(self, endpoint: ProcessWatchEndpoint, rank: int) -> None:
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
        }
        try:
            self._process.stdin.write(json.dumps(config) + "\n")
            self._process.stdin.flush()
            readable, _, _ = select.select([self._process.stdout], [], [], _START_TIMEOUT)
            if not readable or self._process.stdout.readline().strip() != "ready":
                raise RuntimeError(f"MPI rank {rank} process watch failed to start")
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
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
                sys.stdin.readline()
                report["kind"] = "exited"


if __name__ == "__main__":
    if sys.argv[1:] != ["--watch"]:
        raise SystemExit("Expected --watch")
    raise SystemExit(_watch_process())
