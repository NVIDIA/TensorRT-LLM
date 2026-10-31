# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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
"""Where the implicit coordinator puts its Unix socket, and who can reach it.

The tests run the real ``_serve_coordinator_and_fleet`` with the real
``CoordinatorServer`` and uvicorn listeners; only the coordinator service and
the fleet processes are stand-ins.
"""

import os
import socket
import stat
import tempfile
import threading
import time
from types import SimpleNamespace

import pytest

from tensorrt_llm.commands import serve
from tensorrt_llm.serve import disagg_coordinator

pytestmark = pytest.mark.cpu_only


class _StubCoordinatorService:
    """Coordinator service without routers; always ready."""

    cluster_storage = None

    def __init__(self, *args, **kwargs):
        pass

    async def start(self):
        pass

    async def stop(self):
        pass

    async def is_ready(self):
        return True


class _StubFleetWorker:
    """A fleet process that exits cleanly once ``done`` is set."""

    pid = 0

    def __init__(self, done: threading.Event):
        self._done = done

    def poll(self):
        return 0 if self._done.is_set() else None

    def terminate(self):
        self._done.set()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _http_get(sock: socket.socket, route: str) -> bytes:
    sock.settimeout(10)
    sock.sendall(f"GET {route} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n".encode())
    chunks = []
    while chunk := sock.recv(65536):
        chunks.append(chunk)
    return b"".join(chunks)


def _http_get_unix(path: str, route: str) -> bytes:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(path)
        return _http_get(sock, route)


def _http_get_tcp(port: int, route: str) -> bytes:
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        return _http_get(sock, route)


class _CoordinatorRun:
    """Run ``_serve_coordinator_and_fleet`` in a thread until stopped."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch):
        self.coord_port = _free_port()
        self.public_port = self.coord_port + 1
        self.fleet_urls = []
        self.errors = []
        self._done = threading.Event()
        monkeypatch.setenv("TRTLLM_NO_USAGE_STATS", "1")
        monkeypatch.setenv(
            serve.DisaggWorkerEnvs.TLLM_DISAGG_COORDINATOR_PORT, str(self.coord_port)
        )
        monkeypatch.setattr(disagg_coordinator, "DisaggCoordinatorService", _StubCoordinatorService)
        monkeypatch.setattr(serve, "_launch_disagg_fleet", self._launch_fleet)
        disagg_cfg = SimpleNamespace(
            hostname="127.0.0.1",
            port=self.public_port,
            server_keep_alive_timeout=5,
            internal_request_auth_key=None,
        )
        self._thread = threading.Thread(
            target=self._serve, args=(disagg_cfg,), name="coordinator-under-test"
        )

    def _launch_fleet(self, *args):
        self.fleet_urls.append(args[-1])
        return [_StubFleetWorker(self._done)]

    def _serve(self, disagg_cfg):
        try:
            serve._serve_coordinator_and_fleet(disagg_cfg, "disagg.yaml", None, None, 180, 180, 2)
        except BaseException as e:  # noqa: BLE001 - reported by the test thread
            self.errors.append(e)

    def start(self):
        self._thread.start()

    def wait_for(self, ready) -> None:
        deadline = time.monotonic() + 60
        while not ready():
            assert not self.errors, self.errors
            assert time.monotonic() < deadline, "coordinator did not come up"
            time.sleep(0.05)

    def stop(self) -> None:
        self._done.set()
        self._thread.join(timeout=60)
        assert not self._thread.is_alive(), "coordinator did not stop"
        assert not self.errors, self.errors


def _tcp_ready(port: int) -> bool:
    try:
        return b" 200 " in _http_get_tcp(port, "/health").split(b"\r\n", 1)[0]
    except OSError:
        return False


@pytest.mark.parametrize("umask", [0o000, 0o022, 0o077])
def test_coordinator_unix_socket_is_in_new_owner_only_directory(monkeypatch, umask):
    monkeypatch.delenv(serve.DisaggWorkerEnvs.TLLM_DISAGG_COORDINATOR_UDS, raising=False)
    run = _CoordinatorRun(monkeypatch)
    old_umask = os.umask(umask)
    try:
        run.start()
        run.wait_for(lambda: run.fleet_urls and os.path.exists(run.fleet_urls[0][len("unix:") :]))
    finally:
        os.umask(old_umask)
    try:
        url = run.fleet_urls[0]
        assert url.startswith("unix:")
        uds_path = url[len("unix:") :]
        uds_dir = os.path.dirname(uds_path)
        assert stat.S_ISSOCK(os.stat(uds_path).st_mode)
        dir_stat = os.stat(uds_dir)
        assert oct(stat.S_IMODE(dir_stat.st_mode)) == oct(0o700)
        assert dir_stat.st_uid == os.getuid()
        assert os.path.basename(uds_dir).startswith(f"trtllm_disagg_coord_{run.public_port}_")
        assert os.path.realpath(uds_dir) not in {
            os.path.realpath(tempfile.gettempdir()),
            os.path.realpath("/tmp"),
        }

        # The fleet's route to the coordinator still answers, as does TCP.
        assert b" 200 " in _http_get_unix(uds_path, "/version").split(b"\r\n", 1)[0]
        run.wait_for(lambda: _tcp_ready(run.coord_port))
    finally:
        run.stop()
    assert not os.path.exists(uds_dir)


def test_coordinator_tcp_transport_creates_no_socket_directory(monkeypatch):
    monkeypatch.setenv(serve.DisaggWorkerEnvs.TLLM_DISAGG_COORDINATOR_UDS, "0")
    run = _CoordinatorRun(monkeypatch)
    name_prefix = f"trtllm_disagg_coord_{run.public_port}"

    def socket_entries():
        return {
            os.path.join(directory, name)
            for directory in {tempfile.gettempdir(), "/tmp"}
            for name in os.listdir(directory)
            if name.startswith(name_prefix)
        }

    before = socket_entries()
    run.start()
    try:
        run.wait_for(lambda: run.fleet_urls and _tcp_ready(run.coord_port))
        assert run.fleet_urls == [f"http://127.0.0.1:{run.coord_port}"]
        assert socket_entries() == before
    finally:
        run.stop()


def test_coordinator_socket_directory_fits_unix_socket_address(monkeypatch, tmp_path):
    long_tmpdir = tmp_path / ("d" * 120)
    long_tmpdir.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(long_tmpdir))

    uds_dir = serve._make_coordinator_uds_dir(8000)
    try:
        uds_path = os.path.join(uds_dir.name, serve._COORDINATOR_UDS_NAME)
        assert os.path.dirname(uds_dir.name) == "/tmp"
        assert stat.S_IMODE(os.stat(uds_dir.name).st_mode) == 0o700
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.bind(uds_path)
    finally:
        uds_dir.cleanup()
    assert list(long_tmpdir.iterdir()) == []
    assert not os.path.exists(uds_dir.name)
