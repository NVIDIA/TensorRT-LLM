# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Per-frontend affinity ports of a multi-frontend trtllm-serve."""

import os
import socket
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm.serve import _frontend_processes as processes
from tensorrt_llm.serve._frontend_processes import MultiFrontendMode

pytestmark = pytest.mark.cpu_only

_STRIDE_ENV = "TRTLLM_FRONTEND_AFFINITY_PORT_STRIDE"
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


@pytest.mark.parametrize("stride,num_frontends", [(None, 4), ("100", 1)])
def test_affinity_ports_off(monkeypatch, stride, num_frontends):
    if stride is None:
        monkeypatch.delenv(_STRIDE_ENV, raising=False)
    else:
        monkeypatch.setenv(_STRIDE_ENV, stride)
    assert processes._frontend_affinity_ports(8000, num_frontends) == []
    mode = MultiFrontendMode(num_frontends, False)
    with processes._frontend_affinity_sockets(
        _HOST, 8000, socket.AF_INET, mode, _describe_holder
    ) as (own, reserved):
        assert own is None
        assert reserved == {}


def test_launcher_binds_every_frontend_port(monkeypatch):
    num_frontends, stride = 3, 7
    base = _free_port_layout(num_frontends, stride)
    monkeypatch.setenv(_STRIDE_ENV, str(stride))
    mode = MultiFrontendMode(num_frontends, False)
    with processes._frontend_affinity_sockets(
        _HOST, base, socket.AF_INET, mode, _describe_holder
    ) as (own, reserved):
        assert own.getsockname()[1] == base + stride
        assert {i: s.getsockname()[1] for i, s in reserved.items()} == {
            1: base + 2 * stride,
            2: base + 3 * stride,
        }
        sockets = [own, *reserved.values()]
    assert all(s.fileno() == -1 for s in sockets)


def test_failed_bind_names_the_holder(monkeypatch):
    num_frontends, stride = 3, 7
    base = _free_port_layout(num_frontends, stride)
    monkeypatch.setenv(_STRIDE_ENV, str(stride))
    mode = MultiFrontendMode(num_frontends, False)
    with socket.socket() as holder:
        holder.bind((_HOST, base + 2 * stride))
        holder.listen()
        with pytest.raises(RuntimeError) as excinfo:
            with processes._frontend_affinity_sockets(
                _HOST, base, socket.AF_INET, mode, _describe_holder
            ):
                pass
    message = str(excinfo.value)
    assert f"affinity socket of frontend 1 to {_HOST}:{base + 2 * stride}" in message
    assert f"holder-of-{base + 2 * stride}" in message
    # Frontend 0's port was released when frontend 1's bind failed.
    assert _is_free(base + stride)


def test_attached_frontend_adopts_inherited_socket(monkeypatch):
    monkeypatch.setenv(_STRIDE_ENV, "7")
    with socket.socket() as reserved_by_launcher:
        reserved_by_launcher.bind((_HOST, 0))
        port = reserved_by_launcher.getsockname()[1]
        inherited_fd = os.dup(reserved_by_launcher.fileno())
    monkeypatch.setenv("TLLM_FRONTEND_AFFINITY_FD", str(inherited_fd))
    monkeypatch.setenv("TLLM_EXECUTOR_FRONTEND_ID", "2")
    mode = MultiFrontendMode(4, True)
    with processes._frontend_affinity_sockets(
        _HOST, 8000, socket.AF_INET, mode, _describe_holder
    ) as (own, reserved):
        assert own.fileno() == inherited_fd
        assert own.getsockname()[1] == port
        assert not own.get_inheritable()
        assert reserved == {}
        assert "TLLM_FRONTEND_AFFINITY_FD" not in os.environ
    assert own.fileno() == -1


@pytest.mark.parametrize("with_affinity", [True, False])
def test_spawn_hands_each_frontend_its_socket(monkeypatch, with_affinity):
    from tensorrt_llm.executor.proxy import GenerationExecutorProxy

    executor = Mock(spec=GenerationExecutorProxy)
    executor.multi_frontend_attach_info.return_value = {"endpoint": "test"}
    launched = []

    def fake_popen(args, env, pass_fds):
        launched.append((env, list(pass_fds)))
        return Mock(pid=1000 + len(launched))

    monkeypatch.setattr(processes.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(processes, "_wait_attached_frontends_ready", lambda *args: None)
    affinity_sockets = None
    fds = {}
    if with_affinity:
        affinity_sockets = {i: socket.socket() for i in (1, 2)}
        fds = {i: s.fileno() for i, s in affinity_sockets.items()}
        handed_off = list(affinity_sockets.values())

    children = processes._spawn_attached_frontends(
        SimpleNamespace(_executor=executor), 3, affinity_sockets=affinity_sockets
    )

    assert len(children) == 2
    for frontend_id, (env, pass_fds) in enumerate(launched, start=1):
        assert env["TLLM_EXECUTOR_FRONTEND_ID"] == str(frontend_id)
        assert int(env["TLLM_FRONTEND_READY_FD"]) in pass_fds
        if with_affinity:
            assert env["TLLM_FRONTEND_AFFINITY_FD"] == str(fds[frontend_id])
            assert fds[frontend_id] in pass_fds
        else:
            assert "TLLM_FRONTEND_AFFINITY_FD" not in env
            assert len(pass_fds) == 1
    if with_affinity:
        assert affinity_sockets == {}
        assert all(s.fileno() == -1 for s in handed_off)


def test_launch_server_serves_the_affinity_port(monkeypatch):
    from tensorrt_llm.commands import serve

    num_frontends, stride = 3, 7
    base = _free_port_layout(num_frontends, stride)
    monkeypatch.setenv(_STRIDE_ENV, str(stride))
    monkeypatch.delenv("TLLM_EXECUTOR_ATTACH_INFO", raising=False)
    monkeypatch.setenv("TRTLLM_RESPONSES_API_DISABLE_STORE", "1")
    monkeypatch.setattr(serve, "PyTorchLLM", Mock())
    handed_off = {}

    def fake_spawn(llm, num_frontends, *, report_failure, affinity_sockets):
        handed_off.update({i: s.getsockname()[1] for i, s in affinity_sockets.items()})
        return []

    served = {}

    class FakeServer:
        def __init__(self, **kwargs):
            self.app = Mock()

        def __call__(self, host, port, sockets):
            served["ports"] = [s.getsockname()[1] for s in sockets]

    monkeypatch.setattr(serve, "_spawn_attached_frontends", fake_spawn)
    monkeypatch.setattr(serve, "OpenAIServer", FakeServer)
    monkeypatch.setattr(serve.uvloop, "run", lambda coro: None)

    serve.launch_server(
        _HOST,
        base,
        {"model": "test", "backend": "pytorch", "num_serve_frontends": num_frontends},
    )

    assert served["ports"] == [base, base + stride]
    assert handed_off == {1: base + 2 * stride, 2: base + 3 * stride}
    # Everything the launcher bound is released once it stops serving.
    assert all(_is_free(base + stride * i) for i in range(num_frontends + 1))
