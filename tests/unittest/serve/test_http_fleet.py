# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import socket
import subprocess

from tensorrt_llm.serve.http_fleet import (
    HttpFleet,
    bind_reuseport_http_socket,
    launch_subprocess_http_fleet,
)


class _FakeProcess:
    def __init__(self, pid: int, return_code=None):
        self.pid = pid
        self.return_code = return_code
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.return_code

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        if self.return_code is None:
            raise subprocess.TimeoutExpired("worker", timeout)
        return self.return_code

    def kill(self):
        self.killed = True
        self.return_code = -9


def test_frontend_sockets_share_port_with_reuseport():
    with bind_reuseport_http_socket("127.0.0.1", 0) as first:
        port = first.getsockname()[1]
        with bind_reuseport_http_socket("127.0.0.1", port) as second:
            assert second.getsockname()[1] == port
            assert first.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT) == 1
            assert second.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT) == 1


def test_generic_fleet_wait_and_cleanup():
    running = _FakeProcess(1)
    exited = _FakeProcess(2, return_code=3)
    fleet = HttpFleet([running, exited])

    assert fleet.wait(poll_interval=0) == (1, exited, 3)
    fleet.cleanup(timeout=0)

    assert running.terminated
    assert running.killed
    assert not exited.terminated


def test_launch_subprocess_fleet_uses_worker_environments(monkeypatch):
    processes = [_FakeProcess(10), _FakeProcess(11)]
    calls = []

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return processes[len(calls) - 1]

    monkeypatch.setattr("tensorrt_llm.serve.http_fleet.subprocess.Popen", popen)
    fleet = launch_subprocess_http_fleet(
        ["python", "-m", "worker"],
        [{"WORKER_ID": "0"}, {"WORKER_ID": "1"}],
        name="test",
    )

    assert fleet.processes == processes
    assert [call[1]["env"]["WORKER_ID"] for call in calls] == ["0", "1"]
