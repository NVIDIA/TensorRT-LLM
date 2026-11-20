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
"""``TRTLLM_KVCACHE_TIME_OUTPUT_PATH`` makes chat requests ask for per-request perf metrics.

With the variable set to a non-empty value, the chat handler submits
``SamplingParams.return_perf_metrics=True`` to the engine, as the completions
handler does. Every request that runs through the chat handler is covered:
``/v1/chat/completions`` (non-streaming and streaming), the Anthropic Messages
routes (single message and message batch) and ``/health_generate``. The tests
read the ``sampling_params`` the handler hands to the scripted ``generate_async``
of the fake generator; they do not assert on the HTTP response, which the flag
leaves unchanged. The non-streaming chat test scripts the fake engine to fail at
the call, so the handler stops right after submitting the request and the
recorded flag is the value the engine was handed.
"""

import os
import time

import pytest
from test_common.serve.app_factory import AggConfig
from test_common.serve.fakes import ScriptedOutput

# threadleak is off: these tests create TestClients and run requests inside the
# test body, so the client's portal thread and the fake server's worker-pool
# threads are still alive when the leak check runs; the serve_env fixture closes
# them in its teardown, which runs after that check (precedent:
# tests/unittest/llmapi/apps/test_serve_lazy_import.py).
pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

_VAR = "TRTLLM_KVCACHE_TIME_OUTPUT_PATH"

_CHAT_BODY = {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]}
_MESSAGES_BODY = {
    "model": "fake-model",
    "max_tokens": 16,
    "messages": [{"role": "user", "content": "hi"}],
}
# A token-id prompt: the completions handler then skips the input processor.
_COMPLETION_BODY = {"model": "fake-model", "prompt": [1, 2, 3]}

_CHAT_MUST_SET_FLAG = (
    "TRTLLM_KVCACHE_TIME_OUTPUT_PATH is set but the chat path did not request per-request "
    "perf metrics; openai_chat must set sampling_params.return_perf_metrics like "
    "openai_completion"
)


def _with_output_path(tmp_path) -> AggConfig:
    # Set through the environment's config, not monkeypatch: isolated_serve_env
    # clears the variable on entry and restores the process environment on exit.
    return AggConfig(env=((_VAR, str(tmp_path)),))


def _only_generate_call(server, handler: str, route: str):
    calls = server.generator.generate_calls
    assert len(calls) == 1, (
        f"{handler} never reached generate_async for {route} ({len(calls)} calls recorded); "
        "build the server with isolated_serve_env (tests/test_common/serve/app_factory.py) "
        "on the fakes in tests/test_common/serve/fakes.py (scripted generate_async, fake "
        "tokenizer, preprocess override)"
    )
    return calls[0]


def test_chat_requests_metrics_when_kvcache_time_output_is_set(serve_env, tmp_path):
    server = serve_env.build_openai_server(_with_output_path(tmp_path))
    # The scripted failure fires inside generate_async, after the call is
    # recorded, so nothing the handler does after submitting the request runs:
    # the recorded flag is the value the engine was handed.
    server.generator.script(ScriptedOutput(raise_on_call=RuntimeError("stop at submission")))
    client = serve_env.client(server.app)
    client.post("/v1/chat/completions", json=_CHAT_BODY)
    call = _only_generate_call(server, "the chat handler", "/v1/chat/completions")
    assert call.sampling_params.return_perf_metrics is True, _CHAT_MUST_SET_FLAG


def test_streaming_chat_requests_metrics_when_kvcache_time_output_is_set(serve_env, tmp_path):
    server = serve_env.build_openai_server(_with_output_path(tmp_path))
    client = serve_env.client(server.app)
    client.post("/v1/chat/completions", json={**_CHAT_BODY, "stream": True})
    call = _only_generate_call(server, "the chat handler", "/v1/chat/completions with stream")
    assert call.kwargs["streaming"] is True, "the request did not reach the engine as a stream"
    assert call.sampling_params.return_perf_metrics is True, _CHAT_MUST_SET_FLAG


def test_chat_does_not_request_metrics_when_unset(serve_env):
    assert _VAR not in os.environ, (
        f"{_VAR} is set in this test's environment although isolated_serve_env clears it on "
        "entry; this test pins the unset case and relies on that clearing"
    )
    server = serve_env.build_openai_server(AggConfig())
    client = serve_env.client(server.app)
    client.post("/v1/chat/completions", json=_CHAT_BODY)
    call = _only_generate_call(server, "the chat handler", "/v1/chat/completions")
    assert call.sampling_params.return_perf_metrics is False, (
        "the chat path requested per-request perf metrics with TRTLLM_KVCACHE_TIME_OUTPUT_PATH "
        "unset and server-level perf metrics off"
    )


def test_anthropic_messages_request_metrics_when_kvcache_time_output_is_set(serve_env, tmp_path):
    server = serve_env.build_openai_server(_with_output_path(tmp_path))
    client = serve_env.client(server.app)
    client.post("/v1/messages", json=_MESSAGES_BODY)
    call = _only_generate_call(server, "the chat handler", "/v1/messages")
    assert call.sampling_params.return_perf_metrics is True, _CHAT_MUST_SET_FLAG


def test_anthropic_batch_requests_metrics_when_kvcache_time_output_is_set(serve_env, tmp_path):
    server = serve_env.build_openai_server(_with_output_path(tmp_path))
    client = serve_env.client(server.app)
    created = client.post(
        "/v1/messages/batches",
        json={"requests": [{"custom_id": "a", "params": _MESSAGES_BODY}]},
    ).json()
    # The batch runs as a task on the app's event loop, which the client keeps
    # alive between requests; wait for the batch to end before reading the call.
    status = created.get("processing_status")
    deadline = time.monotonic() + 10
    while status != "ended" and time.monotonic() < deadline:
        time.sleep(0.01)
        status = client.get(f"/v1/messages/batches/{created['id']}").json()["processing_status"]
    assert status == "ended", f"the message batch did not end within 10 s (status {status!r})"
    call = _only_generate_call(server, "the chat handler", "/v1/messages/batches")
    assert call.sampling_params.return_perf_metrics is True, _CHAT_MUST_SET_FLAG


def test_health_generate_requests_metrics_when_kvcache_time_output_is_set(serve_env, tmp_path):
    server = serve_env.build_openai_server(_with_output_path(tmp_path))
    client = serve_env.client(server.app)
    client.get("/health_generate")
    call = _only_generate_call(server, "the chat handler", "/health_generate")
    assert call.sampling_params.return_perf_metrics is True, _CHAT_MUST_SET_FLAG


def test_completions_requests_metrics_when_kvcache_time_output_is_set(serve_env, tmp_path):
    # The behavior the chat path now matches.
    server = serve_env.build_openai_server(_with_output_path(tmp_path))
    client = serve_env.client(server.app)
    client.post("/v1/completions", json=_COMPLETION_BODY)
    call = _only_generate_call(server, "the completions handler", "/v1/completions")
    assert call.sampling_params.return_perf_metrics is True, (
        "TRTLLM_KVCACHE_TIME_OUTPUT_PATH is set but the completions path did not request "
        "per-request perf metrics; openai_completion sets sampling_params.return_perf_metrics, "
        "and the chat path mirrors it"
    )
