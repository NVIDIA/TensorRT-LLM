# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real frontend processes must fail and shut down as one serving instance."""

import asyncio
import fcntl
import os
import resource
import signal
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import grpc
import pytest

pytest.importorskip("grpc_health", reason="gRPC health runtime not installed")

from tensorrt_llm.grpc.openengine import server as openengine_server  # noqa: E402
from tensorrt_llm.grpc.openengine.bindings import lifecycle_pb2 as lifecycle  # noqa: E402
from tensorrt_llm.grpc.openengine.bindings import openengine_pb2_grpc as rpc  # noqa: E402
from tensorrt_llm.grpc.openengine.coordinator import _Connection  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]


def test_second_frontend_group_rejects_occupied_port_before_model_load(monkeypatch):
    monkeypatch.delenv("TLLM_EXECUTOR_ATTACH_INFO", raising=False)
    model_loaded = Mock(side_effect=AssertionError("model loaded before port check"))
    monkeypatch.setattr(openengine_server, "PyTorchLLM", model_loaded)
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with pytest.raises(RuntimeError, match="Failed to bind"):
            openengine_server.launch_server(
                "127.0.0.1", port, {"model": "test", "backend": "pytorch", "num_serve_frontends": 2}
            )
    model_loaded.assert_not_called()


@pytest.mark.parametrize("exits", [True, False])
def test_readiness_eof_waits_for_exit_status(exits):
    """EOF can precede a reapable exit, but waiting for telemetry stays bounded."""
    from tensorrt_llm.serve import _frontend_processes as processes

    child = Mock(pid=1)
    child.poll.return_value = None
    if exits:
        child.wait.return_value = 7
    else:
        child.wait.side_effect = subprocess.TimeoutExpired("frontend", 1.0)
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    report_failure = Mock()
    try:
        with pytest.raises(RuntimeError, match="before signaling READY"):
            processes._wait_attached_frontends_ready(
                [child], [read_fd], report_failure=report_failure
            )
        child.wait.assert_called_once_with(timeout=1.0)
        if exits:
            report_failure.assert_called_once_with(7, "server", "model_initialization")
        else:
            report_failure.assert_not_called()
    finally:
        os.close(read_fd)


def test_readiness_pipe_supports_high_file_descriptors():
    """Loaded engines may allocate pipe descriptors beyond select's FD_SETSIZE."""
    from tensorrt_llm.serve import _frontend_processes as processes

    if resource.getrlimit(resource.RLIMIT_NOFILE)[0] <= 1024:
        pytest.skip("This host cannot allocate descriptor 1024")
    read_fd, write_fd = os.pipe()
    high_fd = fcntl.fcntl(read_fd, fcntl.F_DUPFD, 1024)
    child = Mock(pid=1, returncode=None)
    child.poll.return_value = None
    try:
        os.write(write_fd, b"R")
        processes._wait_attached_frontends_ready([child], [high_fd])
    finally:
        os.close(high_fd)
        os.close(read_fd)
        os.close(write_fd)


@pytest.mark.parametrize("last_child_ready", [False, True])
def test_ready_child_death_fails_startup(monkeypatch, last_child_ready):
    """READY children must remain supervised until every sibling has started."""
    from tensorrt_llm.serve import _frontend_processes as processes

    children = [Mock(pid=1, returncode=None), Mock(pid=2, returncode=None)]
    for child in children:
        child.poll.side_effect = lambda child=child: child.returncode
    pipes = [os.pipe(), os.pipe()]
    calls = 0

    class ReadySelector:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def register(self, fd, events):
            pass

        def unregister(self, fd):
            pass

        def select(self, timeout):
            nonlocal calls
            calls += 1
            if calls == 1:
                return [(Mock(fd=pipes[0][0]), None)]
            assert calls == 2, "Startup kept waiting after a READY child died"
            children[0].returncode = -9
            return [(Mock(fd=pipes[1][0]), None)] if last_child_ready else []

    monkeypatch.setattr(processes.selectors, "DefaultSelector", ReadySelector)
    report_failure = Mock()
    try:
        for _, write_fd in pipes:
            os.write(write_fd, b"R")
        with pytest.raises(RuntimeError, match="exited with code -9"):
            processes._wait_attached_frontends_ready(
                children, [read_fd for read_fd, _ in pipes], report_failure=report_failure
            )
        report_failure.assert_called_once_with(-9, "server", "model_initialization")
    finally:
        for pair in pipes:
            for fd in pair:
                os.close(fd)


# Replace the GPU engine and gate failure callbacks where needed. The real
# launcher, readiness pipes, gRPC servers, coordinator, and cleanup remain in use.
_LAUNCHER = """
import asyncio
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock
from tensorrt_llm.executor.proxy import GenerationExecutorProxy
from tensorrt_llm.grpc.openengine import server

root = Path(sys.argv[1])
frontend = int(os.environ.get("TLLM_EXECUTOR_FRONTEND_ID", "0"))
(root / f"pid-{frontend}").write_text(str(os.getpid()))

class Engine:
    def __init__(self, **kwargs):
        if sys.argv[3] == "startup-signal" or (
            frontend and sys.argv[3] in ("child-startup-signal", "child-startup-engine-failure")
        ):
            if frontend:
                previous = signal.getsignal(signal.SIGTERM)
                def on_term(signum, frame):
                    previous(signum, frame)
                    (root / "child-stopping").touch()
                signal.signal(signal.SIGTERM, on_term)
            (root / "initializing").touch()
            deadline = time.monotonic() + 45
            while not (root / "continue-initialization").exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError("test did not release initialization")
                time.sleep(0.01)
        if frontend and sys.argv[3] == "child-startup":
            raise RuntimeError("injected attach failure")
        self.llm_id = "shared-engine"
        self.args = SimpleNamespace(guided_decoding_backend=None)
        self.tokenizer = None
        self._executor = Mock(spec=GenerationExecutorProxy)
        self._executor.multi_frontend_attach_info.return_value = {"mode": "classic"}
    def _check_health(self):
        return not (
            frontend == 0 and sys.argv[3] == "child-startup-engine-failure"
            and (root / "initializing").exists()
        )
    def shutdown(self):
        (root / f"shutdown-{frontend}").touch()

server.PyTorchLLM = Engine
if sys.argv[3] == "blocked-failure-log":
    from tensorrt_llm.grpc.openengine import coordinator
    original_dispatch = coordinator.Coordinator._dispatch
    async def pause_reservation(self, message, context):
        if message.get("request_id") != "before-terminal-failure":
            return await original_dispatch(self, message, context)
        original_healthy = self._healthy
        def paused_health():
            healthy = original_healthy()
            (root / "reservation-health-paused").touch()
            deadline = time.monotonic() + 10
            while not (root / "release-reservation-health").exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError("test did not release reservation health")
                time.sleep(0.01)
            (root / "health-before-publication").write_text(str(original_healthy()))
            return healthy
        self._healthy = paused_health
        try:
            return await original_dispatch(self, message, context)
        finally:
            self._healthy = original_healthy
    coordinator.Coordinator._dispatch = pause_reservation
    original_reactor = server.CoordinatorReactor
    class FailureReactor(original_reactor):
        def report_owner_failure(self, reason):
            (root / "publication-entered").touch()
            return super().report_owner_failure(reason)
    server.CoordinatorReactor = FailureReactor
    original_error = server.logger.error
    def blocked_error(reason, *args, **kwargs):
        if frontend == 0 and reason == "injected first terminal failure":
            (root / "failure-logging").touch()
            deadline = time.monotonic() + 10
            while not (root / "release-failure-log").exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError("test did not release failure logging")
                time.sleep(0.01)
        return original_error(reason, *args, **kwargs)
    server.logger.error = blocked_error
    original_frontend = server.FrontendClient
    class FailureFrontend(original_frontend):
        def __init__(self, directory, frontend_id, fail):
            super().__init__(directory, frontend_id, fail)
            if frontend == 0:
                (root / "coordinator-directory").write_text(directory)
                signal.signal(signal.SIGUSR1,
                              lambda signum, frame: fail("injected first terminal failure"))
                def repeat_failure(signum, frame):
                    fail("injected second terminal failure")
                    (root / "reentrant-returned").touch()
                signal.signal(signal.SIGUSR2, repeat_failure)
    server.FrontendClient = FailureFrontend
if sys.argv[3] == "admission-cancel":
    original_stop = server.CoordinatorReactor.stop_admission
    async def cancel_admission_shutdown(self):
        await original_stop(self)
        raise asyncio.CancelledError()
    server.CoordinatorReactor.stop_admission = cancel_admission_shutdown
server.launch_server("127.0.0.1", int(sys.argv[2]),
                     {"model": "test", "backend": "pytorch", "num_serve_frontends": 2})
"""


def _wait_until(predicate, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise AssertionError("frontend lifecycle operation timed out")


@pytest.mark.parametrize(
    "failure",
    [
        "child-startup",
        "child-runtime",
        "child-hang",
        "launcher-death",
        "shutdown",
        "admission-cancel",
        "blocked-failure-log",
        "startup-signal",
        "child-startup-signal",
        "child-startup-engine-failure",
    ],
)
def test_frontend_group_cleanup(tmp_path, failure):
    """A failed/terminated frontend must not leave siblings serving indefinitely."""
    script = tmp_path / "launch.py"
    script.write_text(_LAUNCHER)
    log_path = tmp_path / "server.log"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(script), str(tmp_path), str(port), failure],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "TLLM_FRONTEND_READY_TIMEOUT": "45"},
        )
    try:
        if failure == "startup-signal":
            _wait_until(lambda: (tmp_path / "initializing").exists())
            process.terminate()
            (tmp_path / "continue-initialization").touch()
            assert process.wait(timeout=20) == 0
            assert (tmp_path / "shutdown-0").exists()
        elif failure in ("child-startup-signal", "child-startup-engine-failure"):
            _wait_until(lambda: (tmp_path / "initializing").exists())
            if failure == "child-startup-signal":
                process.terminate()
            # Resume only after the launcher starts terminating its child:
            # otherwise READY could race cancellation and bypass the wait path.
            _wait_until(lambda: (tmp_path / "child-stopping").exists(), timeout=10)
            (tmp_path / "continue-initialization").touch()
            return_code = process.wait(timeout=20)
            assert (return_code == 0) == (failure == "child-startup-signal")
            assert (tmp_path / "shutdown-0").exists()
            assert (tmp_path / "shutdown-1").exists()
        elif failure == "child-startup":
            assert process.wait(timeout=90) != 0
            assert (tmp_path / "shutdown-0").exists()
        else:
            with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                control = rpc.ControlStub(channel)

                def ready():
                    try:
                        return (
                            control.Health(lifecycle.HealthRequest(), timeout=0.5).state
                            == lifecycle.HEALTH_STATE_READY
                        )
                    except grpc.RpcError:
                        return False

                _wait_until(ready)
            child_pid = int((tmp_path / "pid-1").read_text())
            if failure == "child-runtime":
                os.kill(child_pid, signal.SIGKILL)
                assert process.wait(timeout=30) != 0
                assert (tmp_path / "shutdown-0").exists()
            elif failure == "child-hang":
                os.kill(child_pid, signal.SIGSTOP)
                assert process.wait(timeout=40) != 0
                assert (tmp_path / "shutdown-0").exists()
            elif failure == "launcher-death":
                process.kill()
                process.wait(timeout=10)
                _wait_until(lambda: (tmp_path / "shutdown-1").exists(), timeout=20)
            elif failure == "blocked-failure-log":

                async def reserve(request_id):
                    directory = (tmp_path / "coordinator-directory").read_text()
                    connection = _Connection(f"unix:{directory}/coordinator")
                    try:
                        return await connection.request(
                            {
                                "operation": "reserve",
                                "request_id": request_id,
                                "token": "generation",
                                "frontend": 1,
                            }
                        )
                    finally:
                        await connection.close()

                with ThreadPoolExecutor(max_workers=1) as executor:
                    earlier = executor.submit(asyncio.run, reserve("before-terminal-failure"))
                    try:
                        _wait_until(
                            lambda: (tmp_path / "reservation-health-paused").exists(), timeout=3
                        )
                        os.kill(process.pid, signal.SIGUSR1)
                        _wait_until(
                            lambda: (tmp_path / "publication-entered").exists()
                            or (tmp_path / "failure-logging").exists(),
                            timeout=3,
                        )
                        (tmp_path / "release-reservation-health").touch()
                        assert earlier.result(timeout=3) == {"reserved": True}
                        assert (tmp_path / "health-before-publication").read_text() == "True"
                        _wait_until(lambda: (tmp_path / "failure-logging").exists(), timeout=3)
                        assert asyncio.run(reserve("after-terminal-failure")) == {
                            "ready": False,
                            "reserved": False,
                        }
                        os.kill(process.pid, signal.SIGUSR2)
                        _wait_until(lambda: (tmp_path / "reentrant-returned").exists(), timeout=3)
                    finally:
                        (tmp_path / "release-reservation-health").touch()
                        (tmp_path / "release-failure-log").touch()
                assert process.wait(timeout=30) != 0
                assert "RuntimeError: injected first terminal failure" in log_path.read_text()
                assert (tmp_path / "shutdown-0").exists()
                assert (tmp_path / "shutdown-1").exists()
            else:
                process.terminate()
                return_code = process.wait(timeout=30)
                assert (return_code == 0) == (failure == "shutdown")
                assert (tmp_path / "shutdown-0").exists()
                assert (tmp_path / "shutdown-1").exists()

        # Binding with reuse disabled proves no sibling still holds the port.
        def port_released():
            with socket.socket() as listener:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    listener.bind(("127.0.0.1", port))
                    return True
                except OSError:
                    return False

        _wait_until(port_released, timeout=20)
    except BaseException:
        print(log_path.read_text())
        raise
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)
