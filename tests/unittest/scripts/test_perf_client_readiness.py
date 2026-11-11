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
"""Exercise benchmark client readiness against real HTTP rejection and retry responses."""

import importlib.util
import json
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.cpu_only

CLIENT_PATH = Path(__file__).resolve().parents[3] / "tests/integration/defs/perf/agentx_client.py"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location("perf_benchmark_client", CLIENT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("AGENTX_READY_TIMEOUT", "60")
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    return module


@pytest.fixture
def endpoint():
    responses = []
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            requests.append(
                (self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            )
            status, body = responses.pop(0) if responses else (503, "not ready")
            self.send_response(status)
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", responses, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_rejection_preserves_server_reason(client: ModuleType, endpoint, status: int) -> None:
    url, responses, requests = endpoint
    responses.append((status, '{"error":{"message":"unsupported chat template"}}'))
    with pytest.raises(RuntimeError, match=f"HTTP {status}.*unsupported chat template"):
        client.verify_endpoint(url, "model")
    assert len(requests) == 1
    assert requests[0][0] == "/v1/chat/completions"
    assert requests[0][1]["model"] == "model"


@pytest.mark.parametrize("status", [408, 429, 500, 503])
def test_transient_http_failure_still_retries(client: ModuleType, endpoint, status: int) -> None:
    url, responses, requests = endpoint
    responses.extend([(status, "temporarily unavailable"), (200, '{"choices":[{}]}')])
    client.verify_endpoint(url, "model")
    assert len(requests) == 2


class _FakeClock:
    """Stands in for the client's ``time`` module; ``sleep`` advances ``monotonic``."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_retryable_failure_reason_reaches_the_timeout_message(
    client: ModuleType, endpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    url, _, requests = endpoint
    # On the fake clock, a 30 s budget with a 15 s retry interval allows exactly
    # two requests. Both get the stub's default 503.
    monkeypatch.setattr(client, "time", _FakeClock())
    monkeypatch.setenv("AGENTX_READY_TIMEOUT", "30")
    monkeypatch.setenv("AGENTX_READY_INTERVAL", "15")
    with pytest.raises(RuntimeError, match="Last response: HTTP 503: not ready"):
        client.verify_endpoint(url, "model")
    assert len(requests) == 2


def test_error_body_naming_choices_is_not_a_completion(client: ModuleType, endpoint) -> None:
    url, responses, requests = endpoint
    responses.extend(
        [
            (500, '{"error":{"param":"choices"}}'),
            (200, '{"error":{"param":"choices"}}'),
            (200, '{"choices":[{}]}'),
        ]
    )
    client.verify_endpoint(url, "model")
    assert len(requests) == 3


def test_heartbeat_request_remains_best_effort(client: ModuleType, endpoint) -> None:
    url, responses, requests = endpoint
    responses.append((400, "invalid request"))
    assert client.post_one_token(url, "model", timeout=1) == "HTTP 400: invalid request"
    assert len(requests) == 1


def test_transport_failure_remains_retryable(client: ModuleType, monkeypatch) -> None:
    def unavailable(*args: object, **kwargs: object) -> None:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(client.urllib.request, "urlopen", unavailable)
    assert client.post_one_token("http://unused", "model", timeout=1) is None


def test_rejection_body_is_bounded(client: ModuleType, endpoint) -> None:
    url, responses, _ = endpoint
    responses.append((400, "x" * 10000))
    with pytest.raises(RuntimeError, match="HTTP 400") as error:
        client.verify_endpoint(url, "model")
    assert len(str(error.value)) < 2300
