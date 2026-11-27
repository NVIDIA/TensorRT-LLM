# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Remote rank death must be observable even when MPI makes no progress."""

import gc
import inspect
import json
import os
import pickle  # nosec B403
import signal
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest
import zmq
from utils.executor_stubs import bare_proxy

from tensorrt_llm._errors import ConfigInputError, ResourceError, RuntimeInitError
from tensorrt_llm._startup_errors import classify_startup_error
from tensorrt_llm.executor import remote_worker_process_monitor as remote
from tensorrt_llm.executor.hmac_framing import HmacVerificationError, sign_frame
from tensorrt_llm.executor.ipc import IpcQueue
from tensorrt_llm.executor.proxy import GenerationExecutorProxy
from tensorrt_llm.executor.remote_worker_process_monitor import describe_lost_worker
from tensorrt_llm.executor.worker_process_monitor import WorkerProcessIdentity, WorkerProcessMonitor

pytestmark = pytest.mark.cpu_only

_WORKER = """
import json, os, sys
from tensorrt_llm.executor.remote_worker_process_monitor import (
    ProcessWatchEndpoint, WorkerProcessWatch)
config = json.loads(sys.stdin.readline())
endpoint = ProcessWatchEndpoint(config['address'], bytes.fromhex(config['key']))
with WorkerProcessWatch(endpoint, 0) as watch:
    print("watch pid", watch._process.pid, flush=True)
    sys.stdin.readline()
# Skip interpreter teardown, which can outlast the return grace under test.
os._exit(0)
"""


def _wait_until(condition, message, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, message
        time.sleep(0.01)


def _start_worker(endpoint):
    """Start a worker process that runs a process watch until told to stop.

    Returns:
        The worker process and its helper's pid, once the helper has reported.
    """
    worker = subprocess.Popen(
        [sys.executable, "-c", _WORKER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    worker.stdin.write(json.dumps({"address": endpoint.address, "key": endpoint.key.hex()}) + "\n")
    worker.stdin.flush()
    # Importing TensorRT-LLM prints a banner first.
    for line in worker.stdout:
        if line.startswith("watch pid "):
            return worker, int(line.split()[-1])
    raise AssertionError(f"Worker exited before its watch started: {worker.wait()}")


class _Log:
    """Records the TRT-LLM logger output of the process-watch module."""

    def __init__(self):
        self.records = []

    def _record(self, level):
        return lambda message: self.records.append((level, message))

    def __getattr__(self, level):
        return self._record(level)

    @property
    def text(self):
        return "\n".join(message for _level, message in self.records)


@pytest.fixture
def log(monkeypatch):
    recorder = _Log()
    monkeypatch.setattr(remote, "logger", recorder)
    return recorder


@pytest.fixture(autouse=True)
def _enable_remote_supervision(monkeypatch):
    # Remote supervision is opt-in; these tests exercise it when enabled.
    monkeypatch.setenv("TRTLLM_WORKER_WATCH", "1")
    for name in ("TRTLLM_WORKER_WATCH_PORT", "TRTLLM_WORKER_WATCH_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("lost_process", ["worker", "watch"])
def test_remote_rank_loss_wakes_existing_monitor_without_mpi(monkeypatch, lost_process):
    # A remote rank's SIGKILL leaves MPI futures and its root alive, so serving
    # fails through the independent process monitor instead.
    if lost_process == "worker":
        monkeypatch.delenv("TRTLLM_WORKER_WATCH_HOST", raising=False)
        monkeypatch.setattr(remote, "_discover_host", lambda: ("127.0.1.1", "test"))
    else:
        monkeypatch.setenv("TRTLLM_WORKER_WATCH_HOST", "127.0.0.1")
    monkeypatch.setenv("TRTLLM_WORKER_WATCH_TIMEOUT", "1.5")
    wake = threading.Event()
    monitor = WorkerProcessMonitor()
    endpoint = monitor.start_remote(1, wake)
    if lost_process == "worker":
        assert endpoint.address.startswith("tcp://127.0.1.1:")
    worker, watch_pid = _start_worker(endpoint)
    try:
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
        # The fatal error names the reporting host and why the rank was lost.
        reason = "worker exited" if lost_process == "worker" else "heartbeat timed out"
        assert describe_lost_worker(dead, monitor.remote_failure) == (
            f"MPI worker rank 0 (pid {worker.pid}) lost process liveness "
            f"on {socket.gethostname()}: {reason}"
        )
    finally:
        monitor.close()
        if worker.poll() is None:
            worker.stdin.write("stop\n")
            worker.stdin.flush()
            worker.wait(timeout=10)
        worker.stdin.close()
        worker.stdout.close()


def test_remote_readiness_requires_every_rank(monkeypatch, log):
    monkeypatch.delenv("TRTLLM_WORKER_WATCH_HOST", raising=False)
    monkeypatch.setattr(remote, "_discover_host", lambda: ("127.0.1.1", "test"))
    monitor = WorkerProcessMonitor()
    endpoint = monitor.start_remote(2, threading.Event())
    watch = remote.WorkerProcessWatch(endpoint, 0)
    try:
        _wait_until(lambda: 0 in monitor._remote._seen, "rank 0's report was not received")
        # Rank 1 never reports, so this wait fails however long it is.
        assert not monitor._remote.wait_ready(timeout=0.2)
        # The startup error names the missing ranks and the endpoint they
        # could not reach.
        failure = monitor.remote_failure
        assert failure["missing"] == [1] and failure["endpoint"] == endpoint.address
        assert failure["reason"] == "no report within 0.2 s"
        assert "MPI worker ranks [1]" in log.text and endpoint.address in log.text
        assert "firewall" in log.text
        proxy = GenerationExecutorProxy.__new__(GenerationExecutorProxy)
        proxy.mpi_futures = []
        proxy.mpi_session = object()
        proxy._command_failure = None
        proxy._worker_process_monitor = monitor
        cause = proxy._detect_worker_death_during_init()
        assert isinstance(cause, RuntimeInitError)
        assert f"ranks [1] did not report to process-watch endpoint {endpoint.address}" in str(
            cause
        )
    finally:
        monitor.close()
        watch.close()


@pytest.mark.timeout(20, method="thread")
def test_rank_missing_after_the_first_report_fails_startup(monkeypatch):
    # A rank that exits before its helper's first report still fails startup,
    # so the proxy does not wait for READY while the surviving ranks block in
    # the init collective.
    monkeypatch.setenv("TRTLLM_WORKER_WATCH_HOST", "127.0.0.1")
    monkeypatch.setattr(remote, "_REGISTRATION_TIMEOUT", 2.0)
    wake = threading.Event()
    monitor = WorkerProcessMonitor()
    endpoint = monitor.start_remote(2, wake)
    watch = None
    try:
        # The deadline starts with the first report, not with the listener.
        assert not wake.wait(2.5), "Ranks failed before any rank reported"
        watch = remote.WorkerProcessWatch(endpoint, 0)
        assert not wake.wait(0.5), "A missing rank failed before its deadline"
        assert monitor.find_dead_worker() is None
        assert wake.wait(6), "A rank that never reported stayed invisible"
        dead = monitor.find_dead_worker()
        assert dead is not None and dead.rank == 1
        failure = monitor.remote_failure
        assert failure["missing"] == [1] and failure["endpoint"] == endpoint.address
        assert "no report within 2 s" in failure["reason"]
    finally:
        monitor.close()
        if watch is not None:
            watch.close()


@pytest.mark.parametrize(
    "override, error_type",
    [
        (None, RuntimeInitError),
        # A bad override is user input, so it is a configuration error, not a
        # runtime initialization failure.
        ("missing.invalid", ConfigInputError),
        ("[::1]", ConfigInputError),
    ],
)
def test_unresolvable_watch_host_reports_its_source(monkeypatch, override, error_type):
    if override is None:
        monkeypatch.delenv("TRTLLM_WORKER_WATCH_HOST", raising=False)
        monkeypatch.setattr(remote, "_discover_host", lambda: ("missing.invalid", "test"))
    else:
        monkeypatch.setenv("TRTLLM_WORKER_WATCH_HOST", override)
    failure = socket.gaierror("hostname cannot be resolved")

    def resolve(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(remote.socket, "getaddrinfo", resolve)
    with pytest.raises(error_type, match="TRTLLM_WORKER_WATCH_HOST") as error:
        remote.RemoteWorkerProcessMonitor(1)
    assert type(error.value) is error_type
    assert error.value.__cause__ is failure
    assert error.value.hint == remote.WORKER_WATCH_HINT
    if override is not None:
        assert repr(override) in str(error.value)
    assert ("without brackets" in str(error.value)) is (override == "[::1]")


@pytest.mark.parametrize(
    "override, error_type", [("0.0.0.0", ConfigInputError), (None, RuntimeInitError)]
)
def test_wildcard_watch_host_reports_its_source(monkeypatch, override, error_type):
    if override is None:
        monkeypatch.delenv("TRTLLM_WORKER_WATCH_HOST", raising=False)
        monkeypatch.setattr(remote, "_discover_host", lambda: ("0.0.0.0", "test"))
    else:
        monkeypatch.setenv("TRTLLM_WORKER_WATCH_HOST", override)
    with pytest.raises(error_type, match="wildcard") as error:
        remote.RemoteWorkerProcessMonitor(1)
    assert type(error.value) is error_type


@pytest.mark.parametrize("port", ["not-a-port", "0", "65536", "-1"])
def test_invalid_watch_port_is_a_configuration_error(monkeypatch, port):
    monkeypatch.setenv("TRTLLM_WORKER_WATCH_HOST", "127.0.0.1")
    monkeypatch.setenv("TRTLLM_WORKER_WATCH_PORT", port)
    with pytest.raises(ConfigInputError, match="TRTLLM_WORKER_WATCH_PORT") as error:
        remote.RemoteWorkerProcessMonitor(1)
    # The message shows the rejected value.
    assert repr(port) in str(error.value)
    assert error.value.hint == remote.WORKER_WATCH_HINT


def test_empty_watch_port_selects_an_ephemeral_port(monkeypatch):
    # An exported but empty port means "unset", as an empty host does.
    monkeypatch.setenv("TRTLLM_WORKER_WATCH_HOST", "127.0.0.1")
    monkeypatch.setenv("TRTLLM_WORKER_WATCH_PORT", "")
    monitor = remote.RemoteWorkerProcessMonitor(1)
    try:
        assert int(monitor.endpoint.address.rsplit(":", 1)[1]) > 0
    finally:
        monitor.close()


@pytest.mark.parametrize("host", ["127.0.0.1", "::1"])
def test_explicit_watch_host_and_fixed_port_reach_standalone_helper(monkeypatch, host):
    # An explicit IPv4 or IPv6 watch host and a fixed port, which a cluster can
    # allowlist, reach a standalone helper.
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    reservation = None
    try:
        reservation = socket.socket(family, socket.SOCK_STREAM)
        reservation.bind((host, 0))
        port = reservation.getsockname()[1]
    except OSError:
        if family == socket.AF_INET6:
            pytest.skip("IPv6 loopback is unavailable")
        raise
    finally:
        if reservation is not None:
            reservation.close()
    monkeypatch.setenv("TRTLLM_WORKER_WATCH_HOST", host)
    monkeypatch.setenv("TRTLLM_WORKER_WATCH_PORT", str(port))

    def unexpected_discovery():
        raise AssertionError("Explicit watch host must take precedence over discovery")

    monkeypatch.setattr(remote, "_discover_host", unexpected_discovery)
    monitor = remote.RemoteWorkerProcessMonitor(1)
    advertised_host = f"[{host}]" if family == socket.AF_INET6 else host
    try:
        assert monitor.endpoint.address == f"tcp://{advertised_host}:{port}"
        watch = remote.WorkerProcessWatch(monitor.endpoint, 0)
        try:
            assert monitor.wait_ready(timeout=10)
        finally:
            watch.close()
    finally:
        monitor.close()


@pytest.mark.parametrize("setting", [None, "", "0"])
def test_remote_supervision_is_off_unless_explicitly_enabled(monkeypatch, log, setting):
    # Supervision needs a worker-to-proxy TCP path that remote MPI deployments
    # may not have, so it stays off unless TRTLLM_WORKER_WATCH=1.
    if setting is None:
        monkeypatch.delenv("TRTLLM_WORKER_WATCH", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_WORKER_WATCH", setting)
    # Invalid network config would fail if a disabled monitor still built a listener.
    monkeypatch.setenv("TRTLLM_WORKER_WATCH_HOST", "missing.invalid")
    monitor = WorkerProcessMonitor()
    try:
        assert monitor.start_remote(1, threading.Event()) is None
        assert monitor._remote is None
        assert monitor.wait_remote_ready()
        assert monitor.find_dead_worker() is None
        # Network settings without the switch are almost certainly a mistake.
        assert "TRTLLM_WORKER_WATCH_HOST is set" in log.text
        assert "TRTLLM_WORKER_WATCH=1" in log.text
    finally:
        monitor.close()


@pytest.mark.parametrize("setting", ["true", "false", "off", "2"])
def test_unrecognized_supervision_switch_is_a_configuration_error(monkeypatch, setting):
    # Only unset, empty, "0" and "1" are valid; any other value fails startup
    # instead of silently choosing a mode.
    monkeypatch.setenv("TRTLLM_WORKER_WATCH", setting)
    monitor = WorkerProcessMonitor()
    try:
        with pytest.raises(ConfigInputError, match="TRTLLM_WORKER_WATCH") as error:
            monitor.start_remote(1, threading.Event())
        assert repr(setting) in str(error.value)
        assert monitor._remote is None
    finally:
        monitor.close()


def test_disabled_supervision_is_silent_without_network_settings(monkeypatch, log):
    monkeypatch.delenv("TRTLLM_WORKER_WATCH", raising=False)
    monkeypatch.delenv("TRTLLM_WORKER_WATCH_HOST", raising=False)
    monkeypatch.delenv("TRTLLM_WORKER_WATCH_PORT", raising=False)
    monitor = WorkerProcessMonitor()
    try:
        assert monitor.start_remote(1, threading.Event()) is None
        assert "TRTLLM_WORKER_WATCH" not in log.text
    finally:
        monitor.close()


@pytest.mark.timeout(30, method="thread")
def test_watch_starts_when_worker_descriptors_exceed_fd_setsize(monkeypatch):
    # select.select() raises ValueError for descriptors >= 1024, and workers
    # often hold that many (CUDA, NCCL, sockets). The helper's pipe must still
    # work on such a rank.
    resource = pytest.importorskip("resource")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    wanted = 1100
    if hard != resource.RLIM_INFINITY and hard < wanted:
        pytest.skip("hard RLIMIT_NOFILE is too low to cross FD_SETSIZE")
    resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, wanted), hard))
    monkeypatch.setenv("TRTLLM_WORKER_WATCH_HOST", "127.0.0.1")
    monitor = remote.RemoteWorkerProcessMonitor(1)
    filler = []
    watch = None
    try:
        while True:
            fd = os.open(os.devnull, os.O_RDONLY)
            if fd >= 1030:
                os.close(fd)
                break
            filler.append(fd)
        watch = remote.WorkerProcessWatch(monitor.endpoint, 0)
        assert watch._process.stdout.fileno() >= 1024
        assert monitor.wait_ready(timeout=5)
    finally:
        if watch is not None:
            watch.close()
        monitor.close()
        for fd in filler:
            os.close(fd)
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


def test_heartbeat_lease_defaults_to_tolerate_brief_network_outages(monkeypatch):
    # The default lease outlasts a network blip that TCP retransmission takes
    # several seconds to recover from, and TRTLLM_WORKER_WATCH_TIMEOUT sets it.
    monkeypatch.setenv("TRTLLM_WORKER_WATCH_HOST", "127.0.0.1")
    monkeypatch.delenv("TRTLLM_WORKER_WATCH_TIMEOUT", raising=False)
    monitor = remote.RemoteWorkerProcessMonitor(1)
    try:
        assert monitor._heartbeat_timeout >= 15.0
    finally:
        monitor.close()
    monkeypatch.setenv("TRTLLM_WORKER_WATCH_TIMEOUT", "45")
    monitor = remote.RemoteWorkerProcessMonitor(1)
    try:
        assert monitor._heartbeat_timeout == 45.0
    finally:
        monitor.close()


@pytest.mark.parametrize("value", ["soon", "nan", "inf", "0", "1", "-3"])
def test_invalid_heartbeat_lease_is_a_configuration_error(monkeypatch, value):
    monkeypatch.setenv("TRTLLM_WORKER_WATCH_HOST", "127.0.0.1")
    monkeypatch.setenv("TRTLLM_WORKER_WATCH_TIMEOUT", value)
    with pytest.raises(ConfigInputError, match="TRTLLM_WORKER_WATCH_TIMEOUT"):
        remote.RemoteWorkerProcessMonitor(1)


@pytest.mark.timeout(30, method="thread")
def test_listener_binds_only_the_advertised_address_and_caps_frames(monkeypatch):
    # The listener binds only the advertised address and caps frame size, so
    # other hosts cannot make it buffer arbitrary data.
    monkeypatch.setenv("TRTLLM_WORKER_WATCH_HOST", "127.0.0.1")
    monkeypatch.setattr(remote, "_START_TIMEOUT", 1.0)
    monitor = remote.RemoteWorkerProcessMonitor(1)
    try:
        assert monitor._queue.address_endpoint.startswith("tcp://127.0.0.1:")
        assert monitor._queue.socket.getsockopt(zmq.MAXMSGSIZE) == remote._MAX_REPORT_BYTES
        port = monitor.endpoint.address.rsplit(":", 1)[1]
        # Another local address does not reach a listener bound to 127.0.0.1.
        other = remote.ProcessWatchEndpoint(f"tcp://127.0.0.2:{port}", monitor.endpoint.key)
        with pytest.raises(RuntimeInitError):
            remote.WorkerProcessWatch(other, 0)
        # An oversized frame is dropped without disturbing real reports.
        with zmq.Context() as context, context.socket(zmq.PUSH) as channel:
            channel.setsockopt(zmq.LINGER, 250)
            channel.connect(monitor.endpoint.address)
            channel.send(b"x" * (remote._MAX_REPORT_BYTES * 2))
        watch = remote.WorkerProcessWatch(monitor.endpoint, 0)
        try:
            assert monitor.wait_ready(timeout=5)
            assert monitor.failure is None
        finally:
            watch.close()
    finally:
        monitor.close()


def test_listener_falls_back_to_all_interfaces_for_a_non_local_address(monkeypatch, log):
    # A forwarded or translated address cannot be bound; keep it advertised.
    monkeypatch.setenv("TRTLLM_WORKER_WATCH_HOST", "192.0.2.1")
    monitor = remote.RemoteWorkerProcessMonitor(1)
    try:
        assert monitor._queue.address_endpoint.startswith("tcp://0.0.0.0:")
        assert monitor.endpoint.address.startswith("tcp://192.0.2.1:")
        assert "not a local address" in log.text
    finally:
        monitor.close()


def test_failed_bind_releases_the_queue_without_a_finalizer_error(monkeypatch):
    # A failed bind releases the socket handle, so ZeroMqQueue.__del__ does not
    # raise on a closed socket during the all-interfaces fallback.
    gc.collect()
    unraisable = []
    monkeypatch.setattr(sys, "unraisablehook", unraisable.append)
    with pytest.raises(zmq.ZMQError):
        IpcQueue(("tcp://192.0.2.1:0", None), is_server=True, socket_type=zmq.PULL)
    gc.collect()
    from_queue = [u.exc_value for u in unraisable if "ZeroMqQueue" in repr(u.object)]
    assert not from_queue


def test_fixed_watch_port_in_use_is_still_a_resource_error(monkeypatch):
    monkeypatch.setenv("TRTLLM_WORKER_WATCH_HOST", "127.0.0.1")
    first = remote.RemoteWorkerProcessMonitor(1)
    try:
        monkeypatch.setenv("TRTLLM_WORKER_WATCH_PORT", first.endpoint.address.rsplit(":", 1)[1])
        with pytest.raises(ResourceError, match="Cannot bind process-watch listener"):
            remote.RemoteWorkerProcessMonitor(1)
    finally:
        first.close()


def _addrinfo(*addresses):
    return [
        (socket.AF_INET6 if ":" in a else socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, 0))
        for a in addresses
    ]


@pytest.mark.parametrize(
    "addresses, expected",
    [
        # A dual-stack name keeps the listener on IPv4.
        (("::1", "127.0.0.1"), "tcp://127.0.0.1:"),
        # A routable address wins over Debian's 127.0.1.1 hostname alias.
        (("127.0.1.1", "192.0.2.1"), "tcp://192.0.2.1:"),
    ],
)
def test_watch_host_name_prefers_ipv4_and_non_loopback(monkeypatch, addresses, expected):
    monkeypatch.setenv("TRTLLM_WORKER_WATCH_HOST", "worker-host")
    monkeypatch.setattr(remote.socket, "getaddrinfo", lambda *_a, **_k: _addrinfo(*addresses))
    monitor = remote.RemoteWorkerProcessMonitor(1)
    try:
        assert monitor.endpoint.address.startswith(expected)
        assert not monitor.endpoint.ipv6
    finally:
        monitor.close()


@pytest.mark.parametrize(
    "override, warned",
    [
        # A hostname whose only entry is the Debian/Ubuntu 127.0.1.1 alias.
        ("worker-host", True),
        # Discovery that found no routable interface.
        (None, True),
        # Explicit loopback is a valid choice when all ranks share the namespace.
        ("127.0.0.1", False),
        ("localhost", False),
    ],
)
def test_loopback_listener_address_warns_unless_chosen_explicitly(
    monkeypatch, log, override, warned
):
    if override is None:
        monkeypatch.delenv("TRTLLM_WORKER_WATCH_HOST", raising=False)
        monkeypatch.setattr(remote, "_discover_host", lambda: ("127.0.0.1", "test discovery"))
    else:
        monkeypatch.setenv("TRTLLM_WORKER_WATCH_HOST", override)
    address = "127.0.1.1" if override == "worker-host" else "127.0.0.1"
    monkeypatch.setattr(remote.socket, "getaddrinfo", lambda *_a, **_k: _addrinfo(address))
    monitor = remote.RemoteWorkerProcessMonitor(1)
    try:
        assert monitor.endpoint.address.startswith(f"tcp://{address}:")
        assert ("advertises loopback address" in log.text) is warned
        if warned:
            assert address in log.text and "TRTLLM_WORKER_WATCH_HOST" in log.text
    finally:
        monitor.close()


@pytest.mark.parametrize(
    "variable, value, expected",
    [
        ("NCCL_SOCKET_IFNAME", "ib", "198.51.100.7"),
        ("NCCL_SOCKET_IFNAME", "=ib0", "198.51.100.7"),
        ("NCCL_SOCKET_IFNAME", "^docker", "203.0.113.5"),
        ("GLOO_SOCKET_IFNAME", "ib0", "198.51.100.7"),
        ("OMPI_MCA_oob_tcp_if_include", "10.0.0.0/8,ib0", "198.51.100.7"),
    ],
)
def test_discovery_prefers_the_configured_inter_node_interface(
    monkeypatch, variable, value, expected
):
    # Discovery uses the interface NCCL/MPI are pinned to rather than the
    # default route, which may be firewalled between nodes.
    for name in remote._INTERFACE_HINT_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(variable, value)
    monkeypatch.setattr(
        remote,
        "_interface_ipv4s",
        lambda: {"docker0": [], "eth0": ["192.0.2.10"], "ib0": ["198.51.100.7"]},
    )

    class RouteProbe:
        def __init__(self, *_args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def connect(self, _address):
            pass

        def getsockname(self):
            return ("203.0.113.5", 40000)

    monkeypatch.setattr(remote.socket, "socket", RouteProbe)
    assert remote._discover_host()[0] == expected


def test_discovery_without_a_route_uses_an_interface_quietly(monkeypatch, log):
    # Without a route, discovery uses an interface address without logging
    # errors, and falls back to 127.0.0.1 only when no interface has one.
    for name in remote._INTERFACE_HINT_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(remote, "_interface_ipv4s", lambda: {"lo": [], "eth0": ["192.0.2.10"]})

    class NoRoute:
        def __init__(self, *_args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def connect(self, _address):
            raise OSError("Network is unreachable")

    monkeypatch.setattr(remote.socket, "socket", NoRoute)
    assert remote._discover_host() == ("192.0.2.10", "interface eth0")
    assert not [level for level, _message in log.records if level in ("error", "critical")]
    monkeypatch.setattr(remote, "_interface_ipv4s", lambda: {"lo": []})
    assert remote._discover_host()[0] == "127.0.0.1"


class _FakeComm:
    """Stand-in MPI communicator whose allgather returns the peers' outcomes."""

    def __init__(self, rank, peers):
        self.rank = rank
        self.peers = peers
        self.gathered = []

    def allgather(self, value):
        self.gathered.append(value)
        outcomes = list(self.peers)
        outcomes[self.rank] = value
        return outcomes


def _supervised_rank(monkeypatch, rank, peers):
    import tensorrt_llm._utils as utils

    comm = _FakeComm(rank, peers)
    monkeypatch.setattr(utils, "mpi_comm", lambda: comm)
    monkeypatch.setattr(utils, "mpi_rank", lambda: rank)
    calls = []

    @remote.supervise_worker_process
    def worker_main():
        # Stands in for worker_main, whose first step is an MPI barrier.
        calls.append(True)
        return "served"

    return comm, calls, worker_main


def test_peer_watch_failure_fails_every_rank_before_the_worker_collective(monkeypatch):
    # Ranks agree on watch startup before worker_main. When one rank fails,
    # every rank raises instead of waiting in worker_main's barrier, so a
    # caller-owned session can still shut down.
    monkeypatch.setenv("TRTLLM_WORKER_WATCH_HOST", "127.0.0.1")
    monitor = remote.RemoteWorkerProcessMonitor(2)
    try:
        comm, calls, worker_main = _supervised_rank(
            monkeypatch, 0, [None, "RuntimeInitError: MPI rank 1 process watch failed"]
        )
        with pytest.raises(RuntimeInitError, match=r"ranks \[1\]: RuntimeInitError: MPI rank 1"):
            worker_main(process_watch_endpoint=monitor.endpoint)
        assert calls == [], "a healthy rank entered the worker collective"
        assert comm.gathered == [None]
    finally:
        monitor.close()


def test_local_watch_failure_reaches_the_collective_before_raising(monkeypatch):
    monkeypatch.setattr(remote, "_START_TIMEOUT", 0.2)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
        reservation.bind(("127.0.0.1", 0))
        unreachable = remote.ProcessWatchEndpoint(
            f"tcp://127.0.0.1:{reservation.getsockname()[1]}", b"test-key"
        )
        comm, calls, worker_main = _supervised_rank(monkeypatch, 1, [None, None])
        with pytest.raises(RuntimeInitError, match=r"ranks \[1\]") as error:
            worker_main(process_watch_endpoint=unreachable)
    assert calls == []
    assert len(comm.gathered) == 1 and "failed to connect" in comm.gathered[0]
    assert error.value.hint == remote.WORKER_WATCH_HINT


def test_supervised_worker_signature_declares_the_endpoint():
    # The decorator's keyword appears in worker_main's signature, which
    # functools.wraps alone would hide.
    from tensorrt_llm.executor.worker import worker_main

    parameter = inspect.signature(worker_main).parameters["process_watch_endpoint"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


def test_supervised_worker_runs_when_every_watch_starts(monkeypatch):
    monkeypatch.setenv("TRTLLM_WORKER_WATCH_HOST", "127.0.0.1")
    monitor = remote.RemoteWorkerProcessMonitor(1)
    try:
        _comm, calls, worker_main = _supervised_rank(monkeypatch, 0, [None])
        assert worker_main(process_watch_endpoint=monitor.endpoint) == "served"
        assert calls == [True]
    finally:
        monitor.close()


def test_watch_close_does_not_raise_for_a_helper_that_closed_its_input():
    # Closing a helper whose input is gone does not raise BrokenPipeError from
    # a buffered "stop", which would replace the startup error.
    watch = remote.WorkerProcessWatch.__new__(remote.WorkerProcessWatch)
    watch._process = subprocess.Popen(
        [sys.executable, "-c", "import os, time; os.close(0); time.sleep(5)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        time.sleep(0.5)
        watch.close()
        assert watch._process.stdin.closed
    finally:
        if watch._process.poll() is None:
            watch._process.kill()
            watch._process.wait()


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
    assert "loopback address" in message
    info = classify_startup_error(error.value)
    assert info.category == "runtime_init"
    assert info.hint == remote.WORKER_WATCH_HINT
    assert "firewall" not in info.summary


def test_shutdown_before_worker_submission_closes_remote_monitor(monkeypatch):
    monkeypatch.setenv("TRTLLM_WORKER_WATCH_HOST", "127.0.0.1")
    monitor = WorkerProcessMonitor()
    monitor.start_remote(1, threading.Event())
    receiver = monitor._remote._thread
    proxy = bare_proxy(_worker_process_monitor=monitor)
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

    monkeypatch.setattr(remote.time, "monotonic", lambda: 100.0)
    monitor = remote.RemoteWorkerProcessMonitor.__new__(remote.RemoteWorkerProcessMonitor)
    monitor._world_size = 2
    monitor._wake = None
    monitor._lock = threading.Lock()
    monitor._stopped = threading.Event()
    monitor._failure = None
    monitor._heartbeat_timeout = 5.0
    reports = [dict(rank=r, pid=r + 1, host="worker", sequence=0, kind="alive") for r in (0, 1)]
    monitor._seen = {r["rank"]: (r, 0.0) for r in reports}
    monitor._first_report_at = 0.0
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
    # A continuously readable listener still expires a silent rank, even if
    # the incoming frames fail authentication.
    monkeypatch.setattr(remote.time, "monotonic", lambda: 100.0)
    monitor = remote.RemoteWorkerProcessMonitor.__new__(remote.RemoteWorkerProcessMonitor)
    monitor._world_size = 2
    monitor._wake = threading.Event()
    monitor._lock = threading.Lock()
    monitor._stopped = threading.Event()
    monitor._failure = None
    monitor._heartbeat_timeout = 5.0
    reports = [dict(rank=r, pid=r + 1, host="worker", sequence=0, kind="alive") for r in (0, 1)]
    monitor._seen = {0: (reports[0], 100.0), 1: (reports[1], 0.0)}
    monitor._first_report_at = 0.0
    received = 0

    def poll(_timeout):
        # Stop after a bounded number of frames if no rank has expired.
        if monitor.failure is not None or received >= 1024:
            monitor._stopped.set()
        return True

    def get():
        nonlocal received
        received += 1
        if invalid_signature:
            raise HmacVerificationError("HMAC verification failed")
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


class _PausingStop:
    """Stop event whose first wait returns seven seconds late, as after a pause."""

    def __init__(self, clock, on_pause):
        self._clock = clock
        self._on_pause = on_pause
        self._set = False
        self.paused = False

    def is_set(self):
        return self._set

    def set(self):
        self._set = True

    def wait(self, _timeout):
        if not self.paused:
            # This thread waits seven seconds for the GIL after its wait ends.
            self.paused = True
            self._clock[0] += 7.0
            self._on_pause()
        return self._set


def _paused_receiver(monkeypatch, buffered_until_first_wait):
    """Build a receiver that resumes from a long pause with reports pending.

    During the pause, the libzmq I/O thread keeps queueing reports without the
    GIL, or, if the whole process was stopped, reads them only after it resumes.
    """
    from collections import deque

    clock = [100.0]
    monkeypatch.setattr(remote.time, "monotonic", lambda: clock[0])
    monitor = remote.RemoteWorkerProcessMonitor.__new__(remote.RemoteWorkerProcessMonitor)
    monitor._world_size = 2
    monitor._wake = threading.Event()
    monitor._lock = threading.Lock()
    monitor._failure = None
    monitor._heartbeat_timeout = 5.0
    reports = [dict(rank=r, pid=r + 1, host="worker", sequence=0, kind="alive") for r in (0, 1)]
    monitor._seen = {r["rank"]: (r, clock[0]) for r in reports}
    monitor._first_report_at = clock[0]
    frames = deque()
    empty_polls = []

    def queued_during_pause():
        if not buffered_until_first_wait:
            frames.extend(monitor.fresh_reports)

    monitor._stopped = _PausingStop(clock, queued_during_pause)

    def poll(timeout):
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
    # After a GIL or process pause, the receiver drains reports that arrived
    # during the pause before it checks leases, instead of trusting its stale
    # poll result and expiring every live rank.
    monitor = _paused_receiver(monkeypatch, buffered_until_first_wait)
    monitor._run()
    assert monitor._stopped.paused
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


def test_unauthenticated_frames_are_recognized_by_type(monkeypatch):
    # The receiver recognizes a rejected frame by its exception type, not its
    # text, and keeps supervising.
    monkeypatch.setattr(remote.time, "monotonic", lambda: 100.0)
    monitor = remote.RemoteWorkerProcessMonitor.__new__(remote.RemoteWorkerProcessMonitor)
    monitor._world_size = 1
    monitor._wake = threading.Event()
    monitor._lock = threading.Lock()
    monitor._stopped = threading.Event()
    monitor._failure = None
    monitor._heartbeat_timeout = 5.0
    monitor._seen = {0: (dict(rank=0, pid=1, host="worker", sequence=0, kind="alive"), 100.0)}
    monitor._first_report_at = 100.0
    frames = [HmacVerificationError("frame signature mismatch")] * 3

    def poll(_timeout):
        if not frames:
            monitor._stopped.set()
        return bool(frames)

    def get():
        raise frames.pop()

    monitor._queue = SimpleNamespace(
        poll=poll,
        get=get,
        close=lambda: None,
        socket=SimpleNamespace(setsockopt=lambda *_args: None),
    )
    monitor._run()
    assert monitor.failure is None
    assert not monitor._wake.is_set()


def test_ipc_queue_rejects_a_bad_signature_with_a_dedicated_error():
    server = IpcQueue(("tcp://127.0.0.1:0", None), is_server=True, socket_type=zmq.PULL)
    try:
        with pytest.raises(HmacVerificationError):
            server._parse_data(sign_frame(b"other key", pickle.dumps("payload")))
        assert server._parse_data(sign_frame(server.hmac_key, pickle.dumps("ok"))) == "ok"
    finally:
        server.close()


@pytest.mark.timeout(30, method="thread")
def test_receiver_wakeups_do_not_scale_with_report_rate(monkeypatch):
    # The receiver wakes once per interval however many reports arrive, so its
    # cost does not grow with the number of ranks.
    monkeypatch.setenv("TRTLLM_WORKER_WATCH_HOST", "127.0.0.1")
    world_size = 8
    monitor = remote.RemoteWorkerProcessMonitor(world_size)
    scans = []
    check_leases = monitor._check_leases
    monkeypatch.setattr(
        monitor, "_check_leases", lambda now: (scans.append(now), check_leases(now))
    )
    try:
        with zmq.Context() as context, context.socket(zmq.PUSH) as channel:
            channel.setsockopt(zmq.LINGER, 1000)
            channel.connect(monitor.endpoint.address)
            sequence = 0
            started = time.monotonic()
            scans.clear()
            # About 500 reports per second, far above the receive cadence.
            while time.monotonic() - started < 1.0:
                for rank in range(world_size):
                    report = dict(
                        rank=rank, pid=rank + 1, host="worker", sequence=sequence, kind="alive"
                    )
                    channel.send(sign_frame(monitor.endpoint.key, pickle.dumps(report)))
                sequence += 1
                time.sleep(0.016)
            elapsed = time.monotonic() - started
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                with monitor._lock:
                    newest = [report["sequence"] for report, _seen in monitor._seen.values()]
                if len(newest) == world_size and min(newest) == sequence - 1:
                    break
                time.sleep(0.05)
            scanned = len([now for now in scans if now <= started + elapsed])
        assert monitor.failure is None
        assert min(newest) == sequence - 1, "Reports were not all received"
        assert sequence * world_size > 200
        assert scanned <= elapsed / remote._RECEIVE_WAIT + 3, (
            f"{scanned} lease scans for {sequence * world_size} reports in {elapsed:.2f} s"
        )
    finally:
        monitor.close()


def test_runtime_liveness_error_names_the_host_and_reason():
    # The fatal error names the reporting host and the reason, and a receiver
    # crash reads as a supervision failure, not as a lost rank.
    proxy = GenerationExecutorProxy.__new__(GenerationExecutorProxy)
    proxy.mpi_futures = []
    proxy.doing_shutdown = True
    recorded = []
    proxy._set_fatal_error = recorded.append
    monitor = WorkerProcessMonitor()
    failure = dict(rank=2, pid=77, host="node-b", sequence=4, kind="alive")
    monitor._remote = SimpleNamespace(
        failure=dict(failure, reason="heartbeat timed out"), close=lambda: None
    )
    proxy._worker_process_monitor = monitor
    assert proxy._check_mpi_workers(include_remote=True)
    assert str(recorded[0]) == (
        "MPI worker rank 2 (pid 77) lost process liveness on node-b: heartbeat timed out"
    )
    crash = {"rank": -1, "pid": 0, "host": "process-watch receiver", "reason": "boom"}
    identity = WorkerProcessIdentity(-1, 0, None, "process-watch receiver", None)
    assert describe_lost_worker(identity, crash) == (
        "Remote MPI worker process supervision failed: boom"
    )
    local = WorkerProcessIdentity(3, 12345, 1, "localhost", 1)
    assert describe_lost_worker(local, None) == (
        "MPI worker rank 3 (pid 12345) lost process liveness"
    )


def test_liveness_loss_during_init_precedes_command_connection_loss():
    # When a worker dies during initialization, its liveness loss, which names
    # the rank and pid, is reported before the lost command connection.
    proxy = GenerationExecutorProxy.__new__(GenerationExecutorProxy)
    proxy.mpi_futures = []
    proxy.mpi_session = object()
    proxy._command_failure = ConnectionError("Executor command connection lost")
    monitor = WorkerProcessMonitor()
    monitor._dead_identity = WorkerProcessIdentity(0, 4321, 1, "localhost", 1)
    proxy._worker_process_monitor = monitor
    death = proxy._detect_worker_death_during_init()
    assert isinstance(death, RuntimeInitError) and death.reason == "worker_died"
    assert "rank 0 (pid 4321) lost process liveness" in str(death)
    monitor._dead_identity = None
    assert proxy._detect_worker_death_during_init() is proxy._command_failure


def test_worker_call_outcome_precedes_a_returned_report(monkeypatch):
    # A worker call that returns or raises in a live process sends its final
    # helper report before MPI delivers the outcome. The monitor defers the
    # generic loss for _RETURN_GRACE so the worker's own error can win.
    monkeypatch.setenv("TRTLLM_WORKER_WATCH_HOST", "127.0.0.1")
    # The receiver reads the grace on every pass. Holding it long, then
    # ending it, orders the steps without depending on wall-clock margins.
    monkeypatch.setattr(remote, "_RETURN_GRACE", 60.0)
    wake = threading.Event()
    monitor = WorkerProcessMonitor()
    endpoint = monitor.start_remote(1, wake)
    worker, _watch_pid = _start_worker(endpoint)
    try:
        assert monitor.wait_remote_ready()
        # The worker call ends while its process stays alive.
        worker.stdin.write("stop\n")
        worker.stdin.flush()
        assert worker.wait(timeout=10) == 0
        _wait_until(
            lambda: any(
                report.get("kind") == "returned" for report, _ in monitor._remote._seen.values()
            ),
            "the returned report was not received",
        )
        assert not wake.wait(0.5), "A returned call failed before MPI could report it"
        assert monitor.find_dead_worker() is None
        monkeypatch.setattr(remote, "_RETURN_GRACE", 0.0)
        assert wake.wait(10), "A returned call without an MPI result stayed invisible"
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


@pytest.mark.timeout(60, method="thread")
def test_forwarded_worker_error_precedes_a_returned_call_report(monkeypatch):
    # Under trtllm-llmapi-launch, the launcher forwards a worker's exception
    # after the helper reports that the call returned. With both pending, the
    # error monitor records the forwarded exception, not the generic liveness
    # loss, and a health check records neither.
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
    # Hold the returned report until both outcomes are pending; see
    # test_worker_call_outcome_precedes_a_returned_report.
    monkeypatch.setattr(remote, "_RETURN_GRACE", 60.0)
    shutdowns = []
    client = object.__new__(RemoteMpiCommSessionClient)
    client._is_shutdown = False
    client._pending_responses = []
    client.queue = _Forwarded()
    proxy = bare_proxy(pre_shutdown=lambda: shutdowns.append(True), mpi_session=client)
    endpoint = proxy._worker_process_monitor.start_remote(1, proxy._shutdown_event)
    worker, _watch_pid = _start_worker(endpoint)
    try:
        assert proxy._worker_process_monitor.wait_remote_ready()
        # The worker call raises: its helper reports the return, and the
        # launcher forwards the actual exception afterward.
        worker.stdin.write("stop\n")
        worker.stdin.flush()
        assert worker.wait(timeout=10) == 0
        remote_monitor = proxy._worker_process_monitor._remote
        _wait_until(
            lambda: any(
                report.get("kind") == "returned" for report, _ in remote_monitor._seen.values()
            ),
            "the returned report was not received",
        )
        client.queue.messages.append(
            RemoteWorkerDeath.from_exception(ValueError("actual rank failure"))
        )
        # End the grace: the returned report becomes a liveness loss and sets
        # the error monitor's wake event.
        monkeypatch.setattr(remote, "_RETURN_GRACE", 0.0)
        _wait_until(lambda: proxy._shutdown_event.is_set(), "the returned report did not expire")
        # A health check while both are pending does not record the generic loss.
        assert proxy.check_health()
        assert proxy._fatal_error is None
        monitor_thread = threading.Thread(target=proxy._error_monitor_loop, daemon=True)
        monitor_thread.start()
        monitor_thread.join(timeout=20)
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
