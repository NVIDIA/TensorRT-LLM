# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Remote rank death must be observable even when MPI makes no progress."""

import json
import os
import signal
import socket
import subprocess
import sys
import threading

import pytest
import zmq

from tensorrt_llm.executor import remote_worker_process_monitor as remote
from tensorrt_llm.executor.proxy import GenerationExecutorProxy
from tensorrt_llm.executor.worker_process_monitor import WorkerProcessMonitor

pytestmark = pytest.mark.cpu_only

_WORKER = """
import json, runpy, sys
config = json.loads(sys.stdin.readline())
namespace = runpy.run_path(config.pop('module'))
endpoint = namespace['ProcessWatchEndpoint'](config['address'], bytes.fromhex(config['key']))
watch = namespace['WorkerProcessWatch'](endpoint, 0)
print(watch._process.pid, flush=True)
try:
    sys.stdin.readline()
finally:
    watch.close()
"""


@pytest.mark.parametrize("lost_process", ["worker", "watch"])
def test_remote_rank_loss_wakes_existing_monitor_without_mpi(monkeypatch, lost_process):
    # Regression: a remote rank's SIGKILL leaves MPI futures and its root alive,
    # so serving must fail through the independent process monitor instead.
    if lost_process == "worker":
        monkeypatch.delenv("TRTLLM_WORKER_WATCH_HOST", raising=False)
        monkeypatch.setattr(remote.socket, "gethostbyname", lambda _host: "127.0.1.1")
    else:
        monkeypatch.setenv("TRTLLM_WORKER_WATCH_HOST", "127.0.0.1")
    monkeypatch.setattr(remote, "_HEARTBEAT_TIMEOUT", 1.5)
    wake = threading.Event()
    monitor = WorkerProcessMonitor()
    endpoint = monitor.start_remote(1, wake)
    worker = subprocess.Popen(
        [sys.executable, "-c", _WORKER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    try:
        worker.stdin.write(
            json.dumps(
                {"module": remote.__file__, "address": endpoint.address, "key": endpoint.key.hex()}
            )
            + "\n"
        )
        worker.stdin.flush()
        watch_pid = int(worker.stdout.readline())
        assert monitor.wait_remote_ready()
        # Unauthenticated traffic must not report a dead worker.
        with zmq.Context() as context, context.socket(zmq.PUSH) as channel:
            channel.setsockopt(zmq.LINGER, 250)
            channel.connect(endpoint.address)
            channel.send(b"invalid process-watch signature")
        assert monitor.find_dead_worker() is None
        if lost_process == "worker":
            worker.kill()
            worker.wait(timeout=3)
        else:
            os.kill(watch_pid, signal.SIGKILL)
        assert wake.wait(4), "Dead rank was invisible without an MPI future"
        dead = monitor.find_dead_worker()
        assert dead is not None and dead.rank == 0 and dead.pid == worker.pid
    finally:
        monitor.close()
        if worker.poll() is None:
            worker.stdin.write("stop\n")
            worker.stdin.flush()
            worker.wait(timeout=3)
        worker.stdin.close()
        worker.stdout.close()


def test_remote_readiness_requires_every_rank(monkeypatch):
    monkeypatch.delenv("TRTLLM_WORKER_WATCH_HOST", raising=False)
    monkeypatch.setattr(remote.socket, "gethostbyname", lambda _host: "127.0.1.1")
    monitor = remote.RemoteWorkerProcessMonitor(2)
    watch = remote.WorkerProcessWatch(monitor.endpoint, 0)
    try:
        assert not monitor.wait_ready(timeout=0.2)
        assert monitor.failure is None
    finally:
        monitor.close()
        watch.close()


@pytest.mark.parametrize("override", [None, "missing.invalid"])
def test_unresolvable_watch_host_reports_configuration_error(monkeypatch, override):
    if override is None:
        monkeypatch.delenv("TRTLLM_WORKER_WATCH_HOST", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_WORKER_WATCH_HOST", override)
    failure = socket.gaierror("hostname cannot be resolved")

    def resolve(_host):
        raise failure

    monkeypatch.setattr(remote.socket, "gethostbyname", resolve)
    with pytest.raises(RuntimeError, match="TRTLLM_WORKER_WATCH_HOST") as error:
        remote.RemoteWorkerProcessMonitor(1)
    assert error.value.__cause__ is failure


def test_shutdown_before_worker_submission_closes_remote_monitor(monkeypatch):
    monkeypatch.setenv("TRTLLM_WORKER_WATCH_HOST", "127.0.0.1")
    monitor = WorkerProcessMonitor()
    monitor.start_remote(1, threading.Event())
    receiver = monitor._remote._thread
    proxy = GenerationExecutorProxy.__new__(GenerationExecutorProxy)
    proxy.workers_started = False
    proxy._request_sender = None
    proxy._multi_frontend_ipc_dir = None
    proxy._worker_process_monitor = monitor
    try:
        # A failed MPI submit never sets workers_started, but the receiver
        # already owns a socket and must still be stopped by proxy teardown.
        proxy.shutdown()
        assert not receiver.is_alive()
    finally:
        monitor.close()


def test_queued_heartbeats_prevent_false_death_after_receiver_pause(monkeypatch):
    # A receiver scheduling pause must not expire a rank whose newer heartbeat
    # is already queued behind a different rank's report.
    from collections import deque
    from types import SimpleNamespace

    monkeypatch.setattr(remote.time, "monotonic", lambda: 100.0)
    monitor = remote.RemoteWorkerProcessMonitor.__new__(remote.RemoteWorkerProcessMonitor)
    monitor._world_size = 2
    monitor._wake = None
    monitor._lock = threading.Lock()
    monitor._stopped = threading.Event()
    monitor._failure = None
    reports = [dict(rank=r, pid=r + 1, host="worker", sequence=0, kind="alive") for r in (0, 1)]
    monitor._seen = {r["rank"]: (r, 0.0) for r in reports}
    frames = deque(dict(r, sequence=1) for r in reports)

    def poll(_timeout):
        if not frames:
            monitor._stopped.set()
        return bool(frames)

    monitor._queue = SimpleNamespace(
        poll=poll,
        get=frames.popleft,
        close=lambda: None,
        socket=SimpleNamespace(setsockopt=lambda *_args: None),
    )
    monitor._run()
    assert monitor.failure is None
    assert all(report["sequence"] == 1 for report, _seen in monitor._seen.values())
