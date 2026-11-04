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
"""The perf-sanity port reservation must pass the trtllm-serve launcher's port probe.

Both sides are loaded from their real sources, without importing the serving
stack: the harness's reserve_multi_frontend_port and serve.py's
_reserve_multi_frontend_port.
"""

import ast
import os
import socket
from pathlib import Path
from typing import List

import pytest

pytestmark = pytest.mark.cpu_only

_REPO = Path(__file__).resolve().parents[3]
_PERF_SANITY = _REPO / "tests" / "integration" / "defs" / "perf" / "test_perf_sanity.py"
_SERVE = _REPO / "tensorrt_llm" / "commands" / "serve.py"
_HOST = "127.0.0.1"


def _load(path: Path, names: set, namespace: dict) -> dict:
    """Execute only the named top-level definitions of *path* in *namespace*."""
    tree = ast.parse(path.read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            nodes.append(node)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in names
        ):
            nodes.append(node)
    found = {n.name if isinstance(n, ast.FunctionDef) else n.target.id for n in nodes}
    assert found == names, f"{path.name} is missing {names - found}"
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.fixture
def harness():
    namespace = _load(
        _PERF_SANITY,
        {"reserve_multi_frontend_port", "_RESERVED_PORT_SOCKETS"},
        {"socket": socket, "List": List, "print_info": lambda *args, **kwargs: None},
    )
    yield namespace
    for sock in namespace["_RESERVED_PORT_SOCKETS"]:
        sock.close()


@pytest.fixture(scope="module")
def serve():
    return _load(
        _SERVE,
        {"_reserve_multi_frontend_port"},
        {"os": os, "socket": socket, "_diagnose_port_in_use": lambda port: "test"},
    )


def _frontend_socket() -> socket.socket:
    """A socket set up the way each multi-frontend server frontend binds."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    return sock


def test_reserved_port_passes_the_launcher_probe(harness, serve):
    port = harness["reserve_multi_frontend_port"](_HOST)

    lock = serve["_reserve_multi_frontend_port"](_HOST, port, socket.AF_INET)
    try:
        with _frontend_socket() as frontend:
            frontend.bind((_HOST, port))
            frontend.listen()
            # The other frontends join the same port while the first listens.
            with _frontend_socket() as second_frontend:
                second_frontend.bind((_HOST, port))
                second_frontend.listen()
    finally:
        lock.close()


def test_launcher_probe_still_refuses_a_listening_server(serve):
    with _frontend_socket() as other_server:
        other_server.bind((_HOST, 0))
        other_server.listen()
        port = other_server.getsockname()[1]

        with pytest.raises(RuntimeError, match="Failed to bind socket"):
            serve["_reserve_multi_frontend_port"](_HOST, port, socket.AF_INET)
