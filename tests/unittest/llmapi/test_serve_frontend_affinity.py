# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Socket handoff and affinity ports of a multi-frontend trtllm-serve."""

import os
import socket
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm.serve import _frontend_processes as processes
from tensorrt_llm.serve._frontend_processes import MultiFrontendMode

pytestmark = pytest.mark.cpu_only

_HOST = "127.0.0.1"


def _is_free(port: int) -> bool:
    with socket.socket() as probe:
        try:
            probe.bind((_HOST, port))
        except OSError:
            return False
    return True


def _free_port_layout(num_frontends: int, stride: int = 7) -> int:
    """A base port such that base + stride * i is free for i in [0, num_frontends]."""
    for _ in range(50):
        with socket.socket() as probe:
            probe.bind((_HOST, 0))
            base = probe.getsockname()[1]
        ports = [base + stride * i for i in range(num_frontends + 1)]
        if ports[-1] < 65536 and all(_is_free(p) for p in ports):
            return base
    pytest.skip("no free port layout found")


def _describe_holder(port: int) -> str:
    return f"holder-of-{port}"


@pytest.fixture(autouse=True)
def _no_ephemeral_range(monkeypatch):
    monkeypatch.setattr(processes, "_ephemeral_port_range", lambda: None)


def test_failed_bind_names_the_holder():
    num_frontends, stride = 3, 7
    base = _free_port_layout(num_frontends, stride)
    mode = MultiFrontendMode(num_frontends, False)
    with socket.socket() as holder:
        holder.bind((_HOST, base + 2 * stride))
        holder.listen()
        with pytest.raises(RuntimeError) as excinfo:
            with processes._frontend_affinity_sockets(
                _HOST, base, socket.AF_INET, mode, stride, _describe_holder
            ):
                pass
    message = str(excinfo.value)
    assert f"affinity socket of frontend 1 to {_HOST}:{base + 2 * stride}" in message
    assert f"holder-of-{base + 2 * stride}" in message
    # Frontend 0's port was released when frontend 1's bind failed.
    assert _is_free(base + stride)


def test_attached_frontend_adopts_inherited_socket(monkeypatch):
    with socket.socket() as reserved_by_launcher:
        reserved_by_launcher.bind((_HOST, 0))
        port = reserved_by_launcher.getsockname()[1]
        inherited_fd = os.dup(reserved_by_launcher.fileno())
    monkeypatch.setenv("TLLM_FRONTEND_AFFINITY_FD", str(inherited_fd))
    monkeypatch.setenv("TLLM_EXECUTOR_FRONTEND_ID", "2")
    mode = MultiFrontendMode(4, True)
    with processes._frontend_affinity_sockets(
        _HOST, 8000, socket.AF_INET, mode, 7, _describe_holder
    ) as (own, reserved):
        assert own.fileno() == inherited_fd
        assert own.getsockname()[1] == port
        assert not own.get_inheritable()
        assert reserved == {}
        assert "TLLM_FRONTEND_AFFINITY_FD" not in os.environ
    assert own.fileno() == -1


def test_launch_server_hands_its_sockets_to_the_server(monkeypatch, tmp_path):
    from tensorrt_llm.commands import serve

    num_frontends, stride = 3, 7
    base = _free_port_layout(num_frontends, stride)
    monkeypatch.delenv("TLLM_EXECUTOR_ATTACH_INFO", raising=False)
    monkeypatch.setenv("TRTLLM_RESPONSES_API_DISABLE_STORE", "1")
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    llm = Mock()
    # The launcher binds its forwarding socket in the executor's ipc dir.
    llm._executor._multi_frontend_ipc_dir = str(tmp_path)
    monkeypatch.setattr(serve, "PyTorchLLM", Mock(return_value=llm))
    handed_off = {}

    def fake_spawn(
        llm, num_frontends, *, extra_env, report_failure, affinity_sockets, launcher_socket=None
    ):
        handed_off.update({i: s.getsockname()[1] for i, s in affinity_sockets.items()})
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as frontend:
            frontend.settimeout(5)
            frontend.connect(extra_env[serve.LAUNCHER_UDS_ENV])
            frontend.sendall(b"queued startup request")
        return []

    served = {}

    class FakeServer:
        def __init__(self, **kwargs):
            served["multi_frontend_serving"] = kwargs["multi_frontend_serving"]
            self.app = Mock()

        def __call__(self, host, port, sockets):
            served["ports"] = [s.getsockname()[1] for s in sockets if s.family != socket.AF_UNIX]
            served["uds"] = [s.getsockname() for s in sockets if s.family == socket.AF_UNIX]
            launcher_socket = next(
                listener for listener in sockets if listener.family == socket.AF_UNIX
            )
            launcher_socket.settimeout(5)
            connection, _ = launcher_socket.accept()
            with connection:
                assert connection.recv(128) == b"queued startup request"

    monkeypatch.setattr(serve, "_spawn_attached_frontends", fake_spawn)
    monkeypatch.setattr(serve, "OpenAIServer", FakeServer)
    monkeypatch.setattr(serve.uvloop, "run", lambda coro: None)

    serve.launch_server(
        _HOST,
        base,
        {
            "model": "test",
            "backend": "pytorch",
            "num_serve_frontends": num_frontends,
            "serve_frontend_affinity_port_stride": stride,
        },
    )

    assert served["ports"] == [base, base + stride]
    # Tokenization routing needs the launcher's frontend id and the frontend count.
    serving = served["multi_frontend_serving"]
    assert (serving.is_launcher, serving.frontend_id, serving.num_frontends) == (
        True,
        0,
        num_frontends,
    )
    assert served["uds"] == [str(tmp_path / serve.LAUNCHER_UDS_NAME)]
    assert handed_off == {1: base + 2 * stride, 2: base + 3 * stride}
    # Everything the launcher bound is released once it stops serving.
    assert all(_is_free(base + stride * i) for i in range(num_frontends + 1))


@pytest.mark.parametrize("failure_stage", ["server_setup", "frontend_startup"])
def test_failed_launcher_start_releases_queued_connections_before_stopping_frontends(
    monkeypatch, tmp_path, failure_stage
) -> None:
    from tensorrt_llm.commands import serve
    from tensorrt_llm.executor.proxy import GenerationExecutorProxy

    base = _free_port_layout(2)
    monkeypatch.delenv("TLLM_EXECUTOR_ATTACH_INFO", raising=False)
    monkeypatch.setenv("TRTLLM_RESPONSES_API_DISABLE_STORE", "1")
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    llm = Mock()
    llm._executor = Mock(spec=GenerationExecutorProxy)
    llm._executor._multi_frontend_ipc_dir = str(tmp_path)
    llm._executor.multi_frontend_attach_info.return_value = {}
    llm.input_processor = SimpleNamespace(runs_without_gpu=False)
    monkeypatch.setattr(serve, "PyTorchLLM", Mock(return_value=llm))
    stopped = []

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as frontend:
        frontend.settimeout(1)

        def fake_spawn(*args, extra_env, **kwargs):
            frontend.connect(extra_env[serve.LAUNCHER_UDS_ENV])
            frontend.sendall(b"queued startup request")
            return [Mock()]

        def fail_server(**kwargs):
            raise RuntimeError("server setup failed")

        def stop_frontends(children) -> None:
            with pytest.raises(ConnectionResetError):
                frontend.recv(1)
            stopped.extend(children)

        def fake_popen(*args, **kwargs):
            frontend.connect(kwargs["env"][serve.LAUNCHER_UDS_ENV])
            frontend.sendall(b"queued startup request")
            child = Mock(pid=1, returncode=0)
            child.wait.return_value = 0
            return child

        if failure_stage == "server_setup":
            monkeypatch.setattr(serve, "_spawn_attached_frontends", fake_spawn)
            error = "server setup failed"
        else:
            monkeypatch.setattr(processes.subprocess, "Popen", fake_popen)
            error = "before signaling READY"
        monkeypatch.setattr(serve, "_watch_attached_frontends", lambda children: threading.Event())
        monkeypatch.setattr(serve, "_terminate_attached_frontends", stop_frontends)
        monkeypatch.setattr(processes, "_terminate_attached_frontends", stop_frontends)
        monkeypatch.setattr(serve, "OpenAIServer", fail_server)

        with pytest.raises(RuntimeError, match=error):
            serve.launch_server(
                _HOST,
                base,
                {"model": "test", "backend": "pytorch", "num_serve_frontends": 2},
            )

    assert len(stopped) == 1
