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
"""CPU-only private transport tests. No Docker daemon, container, or external endpoint."""

import json
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import test_checkpoint_engine_client as lifecycle

pytestmark = pytest.mark.cpu_only
endpoint = lifecycle.endpoint
client = lifecycle.client
_ID = "ab" * 32
_POPEN = subprocess.Popen


class _LocalExec:
    """Check the real CLI invocation, then execute just its in-container command locally."""

    def __init__(self) -> None:
        self.processes: list[subprocess.Popen] = []
        self.routes: list[str] = []
        self.replace_helper: str | None = None

    def __call__(self, args: list[str], **kwargs: object) -> subprocess.Popen:
        assert args[:3] == ["/usr/bin/docker", "--host=unix:///var/run/docker.sock", "--config"]
        assert list(Path(args[3]).iterdir()) == []
        assert args[4:14] == [
            "exec",
            "-i",
            "--env",
            "TRTLLM_RUNTIME_CONTROL_API_KEY=",
            _ID,
            "python3",
            "-I",
            "-S",
            "-c",
            client._EXEC_HELPER.read_text("ascii"),
        ]
        assert lifecycle.KEY not in repr(args)
        assert kwargs["env"] == {"PATH": "/usr/bin:/bin", "HOME": args[3]}
        assert kwargs["stderr"] == subprocess.DEVNULL
        self.routes.append(args[-2])
        cmd = [sys.executable, *args[10:]]
        if self.replace_helper is not None:
            cmd[4] = self.replace_helper
        proc = _POPEN(cmd, **kwargs)
        self.processes.append(proc)
        return proc


@pytest.fixture
def local_exec(monkeypatch: pytest.MonkeyPatch) -> Iterator[_LocalExec]:
    fake = _LocalExec()
    monkeypatch.setattr(client, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(client.subprocess, "Popen", fake)
    for name in ("DOCKER_HOST", "DOCKER_CONTEXT", "HTTP_PROXY", "TRTLLM_RUNTIME_CONTROL_API_KEY"):
        monkeypatch.setenv(name, lifecycle.KEY)
    yield fake
    assert all(proc.poll() is not None for proc in fake.processes)


@pytest.mark.parametrize("target", ["a" * 12, "A" * 64, "--host=tcp://example", "", False])
def test_invalid_container_is_not_loopback(target: object, local_exec: _LocalExec) -> None:
    with pytest.raises(ValueError, match="immutable container ID"):
        client.EngineClient(8000, lifecycle.KEY, container=target)
    assert not local_exec.routes


def test_no_desktop_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client, "sys", SimpleNamespace(platform="darwin"))
    with pytest.raises(RuntimeError, match="local Linux"):
        client.EngineClient(8000, lifecycle.KEY, container=_ID)
    client.EngineClient(8000, lifecycle.KEY)


def test_fresh_status_only_authority(local_exec: _LocalExec) -> None:
    api = client.EngineClient(8000, lifecycle.KEY, container=_ID)
    replay = lifecycle.auth["RuntimeControlReplayCache"]()
    verify = lifecycle.auth["validate_runtime_control_request"]
    first, second = api.status_headers(), api.status_headers()
    assert (
        set(first)
        == {
            "x-trtllm-runtime-control-auth",
            "x-trtllm-runtime-control-nonce",
            "x-trtllm-runtime-control-timestamp",
        }
        == set(second)
    )
    assert first["x-trtllm-runtime-control-nonce"] != second["x-trtllm-runtime-control-nonce"]
    for headers in (first, second):
        with pytest.raises(ValueError):
            verify(lifecycle.KEY, "POST", "/checkpoint/activate", b"{}", headers, replay)
        verify(lifecycle.KEY, "GET", "/checkpoint/status", b"", headers, replay)
        with pytest.raises(ValueError, match="already been used"):
            verify(lifecycle.KEY, "GET", "/checkpoint/status", b"", headers, replay)
    assert not local_exec.routes


def test_private_lifecycle(endpoint: tuple, local_exec: _LocalExec) -> None:
    port, engine = endpoint
    api = client.EngineClient(port, lifecycle.KEY, container=_ID)
    markers = []

    def mark_intent() -> None:
        assert [route for route, _ in engine.calls] == ["status"]
        markers.append("intent")

    capture = api.prepare(
        "template",
        10,
        lifecycle.PROBE,
        before_dispatch=mark_intent,
        on_send=lambda: markers.append("send"),
    )
    assert markers == ["intent", "send"]
    assert capture["capture_deadline"] > time.monotonic()
    assert all(proc.returncode == 0 for proc in local_exec.processes)
    api.validate_and_activate(
        lifecycle.IDENTITY, "restore", "final-host-result", lifecycle.PROBE, 10
    )
    assert (
        [route.rsplit("/", 1)[-1] for route in local_exec.routes]
        == [
            "status",
            "prepare_capture",
            "status",
            "status",
            "validate_runtime",
            "status",
            "activate",
            "status",
            "ready",
        ]
        == [route for route, _ in engine.calls]
    )


@pytest.mark.parametrize("fault", ["journal", "expired_intent", "lost_reply"])
def test_private_dispatch_boundary(
    fault: str, endpoint: tuple, local_exec: _LocalExec, monkeypatch: pytest.MonkeyPatch
) -> None:
    port, engine = endpoint
    api = client.EngineClient(port, lifecycle.KEY, container=_ID)
    events = []

    def intent() -> None:
        events.append("intent")
        if fault == "journal":
            raise FileExistsError("already claimed")
        if fault == "expired_intent":
            monkeypatch.setattr(client, "time", SimpleNamespace(monotonic=lambda: float("inf")))

    engine.drop_prepare_reply = fault == "lost_reply"
    error = {"journal": FileExistsError, "expired_intent": TimeoutError, "lost_reply": RuntimeError}
    with pytest.raises(error[fault]):
        api.prepare(
            "template",
            10,
            lifecycle.PROBE,
            before_dispatch=intent,
            on_send=lambda: events.append("send"),
        )
    assert events == (["intent", "send"] if engine.drop_prepare_reply else ["intent"])
    assert [r for r, _ in engine.calls] == (
        ["status", "prepare_capture"] if engine.drop_prepare_reply else ["status"]
    )


def test_failed_inner_connect_does_not_claim(local_exec: _LocalExec) -> None:
    # A bound, non-listening TCP port cannot be claimed by another server during this test.
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        api = client.EngineClient(reserved.getsockname()[1], lifecycle.KEY, container=_ID)
        with pytest.raises((RuntimeError, TimeoutError)):
            api._exchange(
                "/checkpoint/prepare_capture",
                time.monotonic() + 1,
                lambda: pytest.fail("no intent before connect"),
                lambda: pytest.fail("no dispatch before connect"),
            )
    assert local_exec.routes == ["/checkpoint/prepare_capture"]


@pytest.mark.parametrize("abandoned", [False, True], ids=["late-release", "no-release"])
def test_helper_enforces_its_own_deadline(endpoint: tuple, abandoned: bool) -> None:
    port, engine = endpoint
    cmd = [
        sys.executable,
        "-I",
        "-S",
        str(client._EXEC_HELPER),
        str(port),
        "/ready",
        "0.25" if abandoned else "5",
    ]
    with _POPEN(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    ) as proc:
        try:
            if abandoned:
                # The exec remains attached but receives neither a request nor stdin EOF.
                assert proc.wait(timeout=5) == -signal.SIGALRM
                assert proc.stdout is not None and proc.stdout.read() == b"R"
            else:
                # A release delayed in transit cannot use a new send budget at arrival.
                out, _ = proc.communicate(b'[1e-12, {}, ""]\n', timeout=5)
                assert out == b"R" and proc.returncode != 0
        finally:
            proc.kill()
    assert not engine.calls


@pytest.mark.parametrize(
    ("fault", "tail"),
    [
        ("oversized", "os.write(1, struct.pack('!HI', 200, 2**31))"),
        ("truncated", "os.write(1, struct.pack('!HI', 200, 4) + b'{}')"),
        ("trailing", "os.write(1, struct.pack('!HI', 200, 0) + b'extra')"),
        ("failed_exec", "os.write(1, struct.pack('!HI', 200, 0)); sys.exit(1)"),
        (
            "stuck_exec",
            "os.write(1, struct.pack('!HI', 200, 0)); os.close(0); os.close(1); time.sleep(30)",
        ),
    ],
)
def test_no_success_without_bounded_reply_and_exit(
    fault: str, tail: str, local_exec: _LocalExec
) -> None:
    local_exec.replace_helper = (
        "import os, struct, sys, time; os.write(1, b'R'); sys.stdin.buffer.readline(); " + tail
    )
    deadline = time.monotonic() + (1 if fault == "stuck_exec" else 5)
    expected = TimeoutError if fault == "stuck_exec" else RuntimeError
    with pytest.raises(expected):
        client.EngineClient(8000, lifecycle.KEY, container=_ID)._exchange("/ready", deadline)
    assert len(local_exec.processes) == 1 and local_exec.processes[0].returncode is not None


def test_no_proxy_or_redirect(endpoint: tuple, local_exec: _LocalExec) -> None:
    port, engine = endpoint
    api = client.EngineClient(port, lifecycle.KEY, container=_ID)
    with pytest.raises(RuntimeError):
        api._exchange("http://127.0.0.1/other", time.monotonic() + 5)
    engine.response = lambda _, data: (307, {"private": lifecycle.KEY})
    with pytest.raises(RuntimeError, match=r"^Checkpoint HTTP 307\.$"):
        api._exchange("/checkpoint/status", time.monotonic() + 5)
    assert [route for route, _ in engine.calls] == ["status"]


def test_published_token_probe(local_exec: _LocalExec) -> None:
    requests = []
    token_reply = [3, 4]
    extra_length = 0

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            assert self.path == "/v1/completions"
            assert not any("runtime-control" in name for name in self.headers)
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps({"choices": [{"token_ids": token_reply}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body) + extra_length))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        try:

            def generate(count: int) -> list[int]:
                return client.generate_tokens(_ID, server.server_port, "model", [1, 2], count, 5)

            assert generate(2) == [3, 4]
            token_reply[:] = [True, 4]
            with pytest.raises(RuntimeError, match="token IDs"):
                generate(2)
            token_reply[:] = [3, 4]
            extra_length = 1  # Valid JSON, but the HTTP response was truncated.
            with pytest.raises(RuntimeError, match="outcome unknown"):
                generate(2)
            with pytest.raises(ValueError):
                generate(65)
        finally:
            server.shutdown()
            thread.join()
    assert (
        requests
        == [
            {
                "model": "model",
                "prompt": [1, 2],
                "max_tokens": 2,
                "top_k": 1,
                "ignore_eos": True,
                "detokenize": False,
                "stream": False,
            }
        ]
        * 3
    )
