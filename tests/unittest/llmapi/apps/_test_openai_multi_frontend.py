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
"""trtllm-serve with 8 frontend processes on one engine.

Each connection may land on any frontend (SO_REUSEPORT), so every request opens
its own. Server-wide state must not depend on which frontend answers, and the
attached frontends serve without creating a CUDA context.
"""

import http.client
import json
import time
from concurrent.futures import ThreadPoolExecutor

import psutil
import pynvml
import pytest
import requests

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer

NUM_FRONTENDS = 8
NUM_REQUESTS = 64
NUM_STREAMS = 16


@pytest.fixture(scope="module")
def model_path() -> str:
    return get_model_path("Qwen3/Qwen3-0.6B")


@pytest.fixture(scope="module")
def server(model_path: str):
    args = ["--backend", "pytorch", "--num_serve_frontends", str(NUM_FRONTENDS)]
    extra_config = {"return_perf_metrics": True, "enable_iter_perf_stats": True}
    with RemoteOpenAIServer(model_path, args, extra_config=extra_config) as server:
        # /health answers once the first frontend listens.
        deadline = time.monotonic() + 600
        while _listeners(server.port) < NUM_FRONTENDS:
            assert time.monotonic() < deadline, _listeners(server.port)
            time.sleep(1)
        yield server


def _listeners(port: int) -> int:
    return sum(
        1
        for c in psutil.net_connections("tcp")
        if c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == port
    )


def _finish_reason(server: RemoteOpenAIServer, model: str, i: int, stream: bool):
    """Run one completion on a connection of its own."""
    with server.get_client() as client:
        output = client.completions.create(
            model=model,
            prompt=f"Request {i}: the capital of France is",
            max_tokens=16,
            stream=stream,
        )
        if stream:
            choices = [chunk.choices[0] for chunk in output if chunk.choices]
        else:
            choices = output.choices
    text = "".join(choice.text for choice in choices)
    reasons = [choice.finish_reason for choice in choices if choice.finish_reason]
    return reasons[-1] if text and reasons else None


def _success_total(server: RemoteOpenAIServer) -> float:
    text = requests.get(server.url_for("prometheus", "metrics"), timeout=30).text
    return sum(
        float(line.rsplit(" ", 1)[1])
        for line in text.splitlines()
        if line.startswith("trtllm_request_success_total")
    )


def test_frontends_share_one_engine(server: RemoteOpenAIServer, model_path: str):
    with ThreadPoolExecutor(16) as pool:
        finish_reasons = list(
            pool.map(
                lambda i: _finish_reason(server, model_path, i, stream=i >= NUM_REQUESTS),
                range(NUM_REQUESTS + NUM_STREAMS),
            )
        )
    assert all(finish_reasons), finish_reasons

    # Only the launcher drains the engine's stats (every second), so the first
    # poll gets all of them, whichever frontend answers.
    time.sleep(3)
    polls = [len(requests.get(server.url_for("metrics"), timeout=30).json()) for _ in range(16)]
    assert polls[0] > 0 and not any(polls[1:]), polls

    # Every scrape counts the requests of all frontends.
    totals = [_success_total(server) for _ in range(NUM_FRONTENDS)]
    assert totals == [NUM_REQUESTS + NUM_STREAMS] * NUM_FRONTENDS, totals


def _connection_to(process: psutil.Process, port: int) -> http.client.HTTPConnection:
    """Open keep-alive connections until one is accepted by the given frontend process."""
    for _ in range(256):
        conn = http.client.HTTPConnection("localhost", port, timeout=120)
        conn.request("GET", "/health")
        conn.getresponse().read()
        client_port = conn.sock.getsockname()[1]
        if any(
            c.laddr.port == port and c.raddr and c.raddr.port == client_port
            for c in process.net_connections(kind="tcp")
        ):
            return conn
        conn.close()
    raise AssertionError(f"process {process.pid} accepted none of 256 connections")


def _complete(conn: http.client.HTTPConnection, model: str) -> str:
    body = {
        "model": model,
        "prompt": "The capital of France is",
        "max_tokens": 16,
        "temperature": 0.0,
    }
    conn.request("POST", "/v1/completions", json.dumps(body), {"Content-Type": "application/json"})
    response = conn.getresponse()
    payload = response.read()
    assert response.status == 200, payload
    return json.loads(payload)["choices"][0]["text"]


def _gpu_compute_pids() -> set[int]:
    pynvml.nvmlInit()
    try:
        return {
            proc.pid
            for index in range(pynvml.nvmlDeviceGetCount())
            for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(
                pynvml.nvmlDeviceGetHandleByIndex(index)
            )
        }
    finally:
        pynvml.nvmlShutdown()


# After test_frontends_share_one_engine: its request counts must not include these requests.
def test_attached_frontends_serve_without_cuda_context(
    server: RemoteOpenAIServer, model_path: str, process_gpu_memory_info_available: bool
):
    if not process_gpu_memory_info_available:
        pytest.skip("NVML cannot attribute GPU contexts to processes here")
    launcher = psutil.Process(server.proc.pid)
    frontends = [
        child
        for child in launcher.children(recursive=True)
        if "TLLM_EXECUTOR_FRONTEND_ID" in child.environ()
    ]
    assert len(frontends) == NUM_FRONTENDS - 1
    assert all(frontend.environ().get("CUDA_VISIBLE_DEVICES") == "" for frontend in frontends)

    expected = _complete(_connection_to(launcher, server.port), model_path)
    assert expected
    for frontend in frontends:
        assert _complete(_connection_to(frontend, server.port), model_path) == expected

    frontend_pids = {
        proc.pid
        for frontend in frontends
        for proc in (frontend, *frontend.children(recursive=True))
    }
    assert not frontend_pids & _gpu_compute_pids()
