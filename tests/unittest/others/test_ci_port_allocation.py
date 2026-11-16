# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CI port allocation must reject reachable services hidden from bind probes."""

import socket

import pytest

__extra_import_path__ = ["~/tests/integration"]
from defs import common

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("forwarded", [False, True])
def test_range_rejects_reachable_service(monkeypatch: pytest.MonkeyPatch, forwarded: bool) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        real_socket = socket.socket

        class ForwardedSocket(real_socket):
            def bind(self, address: tuple[str, int]) -> None:
                # A forwarded endpoint can accept connections without owning a
                # listening socket in the namespace used by the bind probe.
                super().bind((address[0], 0) if forwarded and address[1] == port else address)

        monkeypatch.setattr(common.socket, "socket", ForwardedSocket)
        monkeypatch.setattr(common, "PORTS_IN_USE", set())
        assert common.reserve_port_from_range((port, port), "test") is None
        assert port not in common.PORTS_IN_USE


def test_ephemeral_fallback_rejects_reachable_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        busy_port = listener.getsockname()[1]
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            free_port = probe.getsockname()[1]
        ports = iter([busy_port, free_port])
        monkeypatch.delenv("CONTAINER_PORT_START", raising=False)
        monkeypatch.delenv("CONTAINER_PORT_NUM", raising=False)
        monkeypatch.setattr(common, "get_static_port_range", lambda: None)
        monkeypatch.setattr(common, "get_free_port", lambda: next(ports))
        monkeypatch.setattr(common, "PORTS_IN_USE", set())
        assert common.get_free_port_in_ci(max_attempts=2) == free_port
        assert common.PORTS_IN_USE == {free_port}


def test_range_returns_port_available_for_bind_and_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    monkeypatch.setattr(common, "PORTS_IN_USE", set())
    assert common.reserve_port_from_range((port, port), "test") == port
    assert common.PORTS_IN_USE == {port}
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", port))
