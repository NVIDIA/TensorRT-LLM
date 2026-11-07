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

from tensorrt_llm._errors import ConfigInputError, RuntimeInitError
from tensorrt_llm._startup_errors import classify_startup_error
from tensorrt_llm._torch.disaggregation.native import utils as network_utils
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
        monkeypatch.setattr(network_utils, "get_local_ip", lambda: "127.0.1.1")
        monkeypatch.setattr(remote.socket, "gethostbyname", lambda _host: "127.0.2.1")
    else:
        monkeypatch.setenv("TRTLLM_WORKER_WATCH_HOST", "127.0.0.1")
    monkeypatch.setattr(remote, "_HEARTBEAT_TIMEOUT", 1.5)
    wake = threading.Event()
    monitor = WorkerProcessMonitor()
    endpoint = monitor.start_remote(1, wake)
    if lost_process == "worker":
        assert endpoint.address.startswith("tcp://127.0.1.1:")
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


def test_remote_readiness_requires_every_rank(monkeypatch, caplog):
    monkeypatch.delenv("TRTLLM_WORKER_WATCH_HOST", raising=False)
    monkeypatch.setattr(network_utils, "get_local_ip", lambda: "127.0.1.1")
    monitor = remote.RemoteWorkerProcessMonitor(2)
    watch = remote.WorkerProcessWatch(monitor.endpoint, 0)
    try:
        assert not monitor.wait_ready(timeout=0.2)
        assert monitor.failure is None
        assert "MPI ranks [1]" in caplog.text
        assert monitor.endpoint.address in caplog.text
        assert "firewall" in caplog.text
    finally:
        monitor.close()
        watch.close()


@pytest.mark.parametrize("override", [None, "missing.invalid"])
def test_unresolvable_watch_host_reports_configuration_error(monkeypatch, override):
    if override is None:
        monkeypatch.delenv("TRTLLM_WORKER_WATCH_HOST", raising=False)
        monkeypatch.setattr(network_utils, "get_local_ip", lambda: "missing.invalid")
    else:
        monkeypatch.setenv("TRTLLM_WORKER_WATCH_HOST", override)
    failure = socket.gaierror("hostname cannot be resolved")

    def resolve(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(remote.socket, "getaddrinfo", resolve)
    with pytest.raises(RuntimeInitError, match="TRTLLM_WORKER_WATCH_HOST") as error:
        remote.RemoteWorkerProcessMonitor(1)
    assert error.value.__cause__ is failure
    assert error.value.hint == remote.WORKER_WATCH_HINT


@pytest.mark.parametrize("port", ["not-a-port", "0", "65536"])
def test_invalid_watch_port_is_a_configuration_error(monkeypatch, port):
    monkeypatch.setenv("TRTLLM_WORKER_WATCH_HOST", "127.0.0.1")
    monkeypatch.setenv("TRTLLM_WORKER_WATCH_PORT", port)
    with pytest.raises(ConfigInputError, match="TRTLLM_WORKER_WATCH_PORT"):
        remote.RemoteWorkerProcessMonitor(1)


@pytest.mark.parametrize("host", ["127.0.0.1", "::1"])
def test_explicit_watch_host_and_fixed_port_reach_standalone_helper(monkeypatch, host):
    # Regression: IPv6 overrides used to fail resolution, and clusters could not
    # allowlist a fixed watch port while retaining independent supervision.
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as reservation:
        try:
            reservation.bind((host, 0))
        except OSError:
            if family == socket.AF_INET6:
                pytest.skip("IPv6 loopback is unavailable")
            raise
        port = reservation.getsockname()[1]
    monkeypatch.setenv("TRTLLM_WORKER_WATCH_HOST", host)
    monkeypatch.setenv("TRTLLM_WORKER_WATCH_PORT", str(port))

    def unexpected_discovery():
        raise AssertionError("Explicit watch host must take precedence over discovery")

    monkeypatch.setattr(network_utils, "get_local_ip", unexpected_discovery)
    monitor = remote.RemoteWorkerProcessMonitor(1)
    advertised_host = f"[{host}]" if family == socket.AF_INET6 else host
    try:
        assert monitor.endpoint.address == f"tcp://{advertised_host}:{port}"
        watch = remote.WorkerProcessWatch(monitor.endpoint, 0)
        try:
            assert monitor.wait_ready(timeout=2)
        finally:
            watch.close()
    finally:
        monitor.close()


def test_watch_opt_out_preserves_existing_checks_without_opening_a_listener(monkeypatch, caplog):
    monkeypatch.setenv("TRTLLM_WORKER_WATCH", "0")
    # Invalid network config would fail if opt-out still constructed a listener.
    monkeypatch.setenv("TRTLLM_WORKER_WATCH_HOST", "missing.invalid")
    monitor = WorkerProcessMonitor()
    try:
        assert monitor.start_remote(1, threading.Event()) is None
        assert monitor._remote is None
        assert monitor.wait_remote_ready()
        assert monitor.find_dead_worker() is None
        assert "supervision is disabled" in caplog.text
        assert "may not detect" in caplog.text
    finally:
        monitor.close()


def test_unreachable_watch_helper_reports_rank_endpoint_and_configuration_hint(monkeypatch):
    monkeypatch.setattr(remote, "_START_TIMEOUT", 0.1)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
        reservation.bind(("127.0.0.1", 0))
        endpoint = remote.ProcessWatchEndpoint(
            f"tcp://127.0.0.1:{reservation.getsockname()[1]}", b"test-key"
        )
        with pytest.raises(RuntimeInitError) as error:
            remote.WorkerProcessWatch(endpoint, 7)
    message = str(error.value)
    assert "MPI rank 7" in message
    assert endpoint.address in message
    # Launchers that forward only text keep the hint in the message.
    assert "TRTLLM_WORKER_WATCH_HOST" in message and "firewall" in message
    info = classify_startup_error(error.value)
    assert info.category == "runtime_init"
    assert info.hint == remote.WORKER_WATCH_HINT
    assert "firewall" not in info.summary


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
    monitor._returned = {}
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


@pytest.mark.parametrize("invalid_signature", [False, True])
def test_continuous_traffic_cannot_hide_an_expired_rank(monkeypatch, invalid_signature):
    # Regression: a continuously readable listener must still expire a silent
    # rank, even if the incoming frames fail authentication.
    from types import SimpleNamespace

    monkeypatch.setattr(remote.time, "monotonic", lambda: 100.0)
    monitor = remote.RemoteWorkerProcessMonitor.__new__(remote.RemoteWorkerProcessMonitor)
    monitor._world_size = 2
    monitor._wake = threading.Event()
    monitor._lock = threading.Lock()
    monitor._stopped = threading.Event()
    monitor._failure = None
    reports = [dict(rank=r, pid=r + 1, host="worker", sequence=0, kind="alive") for r in (0, 1)]
    monitor._seen = {0: (reports[0], 100.0), 1: (reports[1], 0.0)}
    monitor._returned = {}
    received = 0

    def poll(_timeout):
        # Bound the pre-fix failure without waiting for a wall-clock timeout.
        if monitor.failure is not None or received >= 1024:
            monitor._stopped.set()
        return True

    def get():
        nonlocal received
        received += 1
        if invalid_signature:
            raise RuntimeError("HMAC verification failed")
        return dict(reports[0], sequence=received)

    monitor._queue = SimpleNamespace(
        poll=poll,
        get=get,
        close=lambda: None,
        socket=SimpleNamespace(setsockopt=lambda *_args: None),
    )
    monitor._run()
    assert monitor._wake.is_set(), "Continuous traffic suppressed heartbeat expiry"
    assert monitor.failure["rank"] == 1
    assert monitor.failure["reason"] == "heartbeat timed out"


def _paused_receiver(monkeypatch, buffered_until_first_wait):
    """Build a receiver whose first poll returns stale data after a long pause.

    During the pause, the libzmq I/O thread keeps queueing reports without the
    GIL. A paused thread's poll result still says nothing is readable.
    """
    from collections import deque
    from types import SimpleNamespace

    clock = [100.0]
    monkeypatch.setattr(remote.time, "monotonic", lambda: clock[0])
    monitor = remote.RemoteWorkerProcessMonitor.__new__(remote.RemoteWorkerProcessMonitor)
    monitor._world_size = 2
    monitor._wake = threading.Event()
    monitor._lock = threading.Lock()
    monitor._stopped = threading.Event()
    monitor._failure = None
    reports = [dict(rank=r, pid=r + 1, host="worker", sequence=0, kind="alive") for r in (0, 1)]
    monitor._seen = {r["rank"]: (r, clock[0]) for r in reports}
    monitor._returned = {}
    frames = deque()
    paused = []
    empty_polls = []

    def poll(timeout):
        if not paused:
            # This thread waits seven seconds for the GIL after its wait ends.
            paused.append(True)
            clock[0] += 7.0
            if not buffered_until_first_wait:
                frames.extend(monitor.fresh_reports)
            return False
        if timeout and not frames and buffered_until_first_wait:
            # A stopped process resumes; the I/O thread then reads its socket.
            frames.extend(monitor.fresh_reports)
        if not frames:
            # Stop on the next wait, after this pass has checked its leases.
            empty_polls.append(True)
            if len(empty_polls) > 1:
                monitor._stopped.set()
        return bool(frames)

    monitor.fresh_reports = [dict(r, sequence=1) for r in reports]
    monitor._queue = SimpleNamespace(
        poll=poll,
        get=frames.popleft,
        close=lambda: None,
        socket=SimpleNamespace(setsockopt=lambda *_args: None),
    )
    return monitor


@pytest.mark.parametrize("buffered_until_first_wait", [False, True])
def test_reports_received_during_a_receiver_pause_prevent_false_death(
    monkeypatch, buffered_until_first_wait
):
    # Regression: after a GIL or process pause, the receiver trusted its stale
    # poll result, skipped the drain and expired every live rank.
    monitor = _paused_receiver(monkeypatch, buffered_until_first_wait)
    monitor._run()
    assert monitor.failure is None
    assert not monitor._wake.is_set()
    assert all(report["sequence"] == 1 for report, _seen in monitor._seen.values())


def test_silent_rank_still_expires_after_a_receiver_pause(monkeypatch):
    monitor = _paused_receiver(monkeypatch, buffered_until_first_wait=False)
    monitor.fresh_reports = monitor.fresh_reports[:1]
    monitor._run()
    assert monitor._wake.is_set()
    assert monitor.failure["rank"] == 1
    assert monitor.failure["reason"] == "heartbeat timed out"


def test_worker_call_outcome_precedes_a_returned_report(monkeypatch):
    # Regression: a worker call that raised in a live process sent its final
    # helper report before MPI delivered the exception, so a generic liveness
    # loss replaced the worker's classified startup or runtime error.
    monkeypatch.setenv("TRTLLM_WORKER_WATCH_HOST", "127.0.0.1")
    monkeypatch.setattr(remote, "_RETURN_GRACE", 1.5)
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
        worker.stdout.readline()
        assert monitor.wait_remote_ready()
        # The worker call ends while its process stays alive.
        worker.stdin.write("stop\n")
        worker.stdin.flush()
        assert worker.wait(timeout=3) == 0
        assert not wake.wait(0.4), "A returned call failed before MPI could report it"
        assert monitor.find_dead_worker() is None
        assert wake.wait(3), "A returned call without an MPI result stayed invisible"
        dead = monitor.find_dead_worker()
        assert dead is not None and dead.rank == 0 and dead.pid == worker.pid
        assert monitor._remote.failure["reason"] == "worker returned without an MPI result"
    finally:
        monitor.close()
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=3)
        worker.stdin.close()
        worker.stdout.close()


@pytest.mark.timeout(20, method="thread")
def test_forwarded_worker_error_precedes_a_returned_call_report(monkeypatch):
    # Regression: under trtllm-llmapi-launch the forwarded worker exception
    # waited for the error monitor's poll, while the returned call's report
    # woke it first and recorded a generic liveness loss.
    import queue as queue_module
    import time

    from tensorrt_llm.llmapi.mpi_session import RemoteMpiCommSessionClient, RemoteWorkerDeath

    class _Forwarded:
        def __init__(self):
            self.messages = []
            self.lock = threading.Lock()

        def poll(self, _timeout):
            with self.lock:
                return bool(self.messages)

        def get(self):
            with self.lock:
                return self.messages.pop(0)

    monkeypatch.setenv("TRTLLM_WORKER_WATCH_HOST", "127.0.0.1")
    monkeypatch.setattr(remote, "_RETURN_GRACE", 0.5)
    proxy = GenerationExecutorProxy.__new__(GenerationExecutorProxy)
    proxy.workers_started = False
    proxy.doing_shutdown = False
    proxy._multi_frontend_ipc_dir = None
    proxy._owns_mpi_session = False
    proxy._engine_dead = False
    proxy._fatal_error = None
    proxy._results = {}
    proxy._error_queue = queue_module.Queue()
    proxy._request_sender = None
    proxy.mpi_futures = []
    shutdowns = []
    proxy.pre_shutdown = lambda: shutdowns.append(True)
    proxy._shutdown_event = threading.Event()
    proxy._worker_process_monitor = WorkerProcessMonitor()
    endpoint = proxy._worker_process_monitor.start_remote(1, proxy._shutdown_event)
    client = object.__new__(RemoteMpiCommSessionClient)
    client._is_shutdown = False
    client._pending_responses = []
    client.queue = _Forwarded()
    proxy.mpi_session = client
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
        worker.stdout.readline()
        assert proxy._worker_process_monitor.wait_remote_ready()
        monitor_thread = threading.Thread(target=proxy._error_monitor_loop, daemon=True)
        monitor_thread.start()
        time.sleep(0.2)
        # The worker call raises: its helper reports the return, and the
        # launcher forwards the actual exception shortly afterward.
        worker.stdin.write("stop\n")
        worker.stdin.flush()
        assert worker.wait(timeout=3) == 0
        time.sleep(0.1)
        client.queue.messages.append(
            RemoteWorkerDeath.from_exception(ValueError("actual rank failure"))
        )
        # A health check while both are pending must not record the generic loss.
        time.sleep(remote._RETURN_GRACE + 0.2)
        proxy.check_health()
        monitor_thread.join(timeout=8)
        assert not monitor_thread.is_alive()
        assert "actual rank failure" in str(proxy._fatal_error)
        assert shutdowns
    finally:
        proxy._worker_process_monitor.close()
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=3)
        worker.stdin.close()
        worker.stdout.close()
