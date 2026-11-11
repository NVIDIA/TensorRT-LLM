# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-frontend affinity ports of a multi-frontend trtllm-serve."""

import os
import socket
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


def test_launch_server_serves_the_affinity_port(monkeypatch, tmp_path):
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

    def fake_spawn(llm, num_frontends, *, extra_env, report_failure, affinity_sockets):
        handed_off.update({i: s.getsockname()[1] for i, s in affinity_sockets.items()})
        return []

    served = {}

    class FakeServer:
        def __init__(self, **kwargs):
            self.app = Mock()

        def __call__(self, host, port, sockets):
            served["ports"] = [s.getsockname()[1] for s in sockets if s.family != socket.AF_UNIX]
            served["uds"] = [s.getsockname() for s in sockets if s.family == socket.AF_UNIX]

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
    assert served["uds"] == [str(tmp_path / serve.LAUNCHER_UDS_NAME)]
    assert handed_off == {1: base + 2 * stride, 2: base + 3 * stride}
    # Everything the launcher bound is released once it stops serving.
    assert all(_is_free(base + stride * i) for i in range(num_frontends + 1))
