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
"""/health, /version and /v1/models behavior, through the production handlers.

These pin what the system endpoints do today, per serving role, so a later
change can show it changed nothing. An unhealthy engine makes ``/health`` raise
``SIGINT`` (after recording a termination observation) so the process does not
linger as a zombie; the signal is patched out here before any unhealthy request.
"""

import signal

import pytest
from test_common.serve.app_factory import AggConfig

import tensorrt_llm.version
from tensorrt_llm.llmapi.disagg_utils import ServerRole

# threadleak is off: these tests create TestClients and run requests inside the
# test body, so the client's portal thread and the fake server's worker-pool
# threads are still alive when the leak check runs; the serve_env fixture closes
# them in its teardown, which runs after that check (precedent:
# tests/unittest/llmapi/apps/test_serve_lazy_import.py).
pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

UNAVAILABLE_BODY = "LLM is unavailable. Please check the server logs for more details."


@pytest.fixture
def sigint_recorder(monkeypatch):
    calls = []
    monkeypatch.setattr(signal, "raise_signal", calls.append)
    return calls


@pytest.mark.parametrize(
    "role", [None, ServerRole.MM_ENCODER, ServerRole.EMBEDDING], ids=["llm", "mm", "embedding"]
)
def test_health_returns_200_with_empty_body(role, serve_env):
    server = serve_env.build_openai_server(AggConfig(role=role))
    response = serve_env.client(server.app).get("/health")
    assert response.status_code == 200
    assert response.content == b""


@pytest.mark.parametrize("role", [None, ServerRole.MM_ENCODER], ids=["llm", "mm"])
def test_unhealthy_executor_returns_503_without_shutdown(role, serve_env, sigint_recorder):
    server = serve_env.build_openai_server(AggConfig(role=role))
    server.generator._executor.healthy = False
    response = serve_env.client(server.app).get("/health")
    assert response.status_code == 503
    assert response.text == UNAVAILABLE_BODY
    assert sigint_recorder == [], "an unhealthy executor without a fatal error must not SIGINT"


@pytest.mark.parametrize("role", [None, ServerRole.MM_ENCODER], ids=["llm", "mm"])
def test_fatal_error_triggers_sigint_and_termination_record(role, serve_env, sigint_recorder):
    server = serve_env.build_openai_server(AggConfig(role=role))
    client = serve_env.client(server.app)
    seen = serve_env.record_terminations()
    server.generator._executor._fatal_error = RuntimeError("boom")
    server.generator._executor.doing_shutdown = False
    response = client.get("/health")
    assert response.status_code == 503
    assert sigint_recorder == [signal.SIGINT]
    assert len(seen) == 1
    assert seen[0].termination_kind == "exception"
    assert seen[0].component == "llm"
    assert seen[0].reporting_source == "self"


@pytest.mark.parametrize("role", [None, ServerRole.MM_ENCODER], ids=["llm", "mm"])
def test_fatal_error_during_shutdown_signals_once(role, serve_env, sigint_recorder):
    server = serve_env.build_openai_server(AggConfig(role=role))
    client = serve_env.client(server.app)
    seen = serve_env.record_terminations()
    server.generator._executor._fatal_error = RuntimeError("boom")
    server.generator._executor.doing_shutdown = True
    response = client.get("/health")
    assert response.status_code == 503
    assert sigint_recorder == [], "a teardown already in progress must not stack signals"
    assert seen == []


def test_embedding_health_depends_on_batcher_and_encoder(serve_env, sigint_recorder):
    server = serve_env.build_openai_server(AggConfig(role=ServerRole.EMBEDDING))
    # Without the lifespan the encode batcher never started, so every
    # /v1/embeddings request would hang; /health reports unhealthy.
    without_lifespan = serve_env.client(server.app, lifespan=False)
    assert without_lifespan.get("/health").status_code == 503
    with_lifespan = serve_env.client(server.app)
    assert with_lifespan.get("/health").status_code == 200
    # Encode-only generators have no generation executor; health reads the
    # encoder executor instead.
    server.generator._encoder_executor = None
    assert with_lifespan.get("/health").status_code == 503
    assert sigint_recorder == [], (
        "the embedding health path finds no generation executor, so it must not SIGINT"
    )


def test_version_reports_package_version(serve_env):
    server = serve_env.build_openai_server(AggConfig())
    response = serve_env.client(server.app).get("/version")
    assert response.status_code == 200
    assert response.json() == {"version": tensorrt_llm.version.__version__}


def test_models_lists_the_served_model(serve_env):
    server = serve_env.build_openai_server(AggConfig())
    response = serve_env.client(server.app).get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    entry = body["data"][0]
    assert entry["id"] == "fake-model"
    assert entry["object"] == "model"
    assert entry["owned_by"] == "tensorrt_llm"
    assert isinstance(entry["created"], int)  # the request time; never compare the value
