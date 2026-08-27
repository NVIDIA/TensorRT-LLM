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
"""Route-level tests for the Anthropic Messages compatibility handlers."""

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import aiohttp
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

from tensorrt_llm.serve.openai_disagg_server import OpenAIDisaggServer
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    FunctionCall,
    ToolCall,
    UsageInfo,
)
from tensorrt_llm.serve.openai_server import OpenAIServer

MODEL = "test-model"


def _request(**overrides):
    payload = {
        "model": MODEL,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hello"}],
    }
    payload.update(overrides)
    return payload


def _chat_response(*, tool_arguments=None):
    tool_calls = []
    if tool_arguments is not None:
        tool_calls = [
            ToolCall(
                id="call_1",
                function=FunctionCall(name="get_weather", arguments=tool_arguments),
            )
        ]
    return ChatCompletionResponse(
        id="chatcmpl-route-test",
        model=MODEL,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="hello", tool_calls=tool_calls),
                finish_reason="tool_calls" if tool_calls else "stop",
            )
        ],
        usage=UsageInfo(prompt_tokens=3, completion_tokens=2, total_tokens=5),
    )


def _json_chat_response(*, tool_arguments=None, status_code=200):
    return JSONResponse(
        content=_chat_response(tool_arguments=tool_arguments).model_dump(),
        status_code=status_code,
    )


def _streaming_chat_response():
    chunks = [
        ChatCompletionStreamResponse(
            model=MODEL,
            choices=[{"index": 0, "delta": {"role": "assistant"}}],
        ),
        ChatCompletionStreamResponse(
            model=MODEL,
            choices=[{"index": 0, "delta": {"content": "hello"}}],
        ),
        ChatCompletionStreamResponse(
            model=MODEL,
            choices=[
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
            usage=UsageInfo(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        ),
    ]

    async def source():
        for chunk in chunks:
            yield f"data: {chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(source(), media_type="text/event-stream")


def _make_route_client(server_kind, openai_response):
    app = FastAPI()
    if server_kind == "standard":
        server = object.__new__(OpenAIServer)
        server.model = MODEL
        backend = AsyncMock(return_value=openai_response)
        server.openai_chat = backend
    else:
        server = object.__new__(OpenAIDisaggServer)
        server._service = SimpleNamespace(openai_chat_completion=object())
        backend = AsyncMock(return_value=openai_response)
        server._wrap_entry_point = Mock(return_value=backend)
    app.add_api_route("/v1/messages", server.anthropic_messages, methods=["POST"])
    return TestClient(app), backend


@pytest.mark.parametrize("server_kind", ["standard", "disagg"])
def test_messages_route_converts_nonstream_response(server_kind):
    client, backend = _make_route_client(server_kind, _json_chat_response())

    response = client.post("/v1/messages", json=_request())

    assert response.status_code == 200
    assert response.json() | {"id": "ignored"} == {
        "id": "ignored",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }
    chat_request = backend.await_args.args[0]
    assert chat_request.model == MODEL
    assert chat_request.max_completion_tokens == 64
    assert not chat_request.stream


@pytest.mark.parametrize("server_kind", ["standard", "disagg"])
def test_messages_route_rejects_anthropic_server_tools(server_kind):
    client, backend = _make_route_client(server_kind, _json_chat_response())

    response = client.post(
        "/v1/messages",
        json=_request(
            tools=[
                {
                    "name": "web_search",
                    "type": "web_search_20250305",
                }
            ]
        ),
    )

    assert response.status_code == 400
    assert response.json()["type"] == "error"
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "server tool" in response.json()["error"]["message"]
    backend.assert_not_awaited()


@pytest.mark.parametrize("server_kind", ["standard", "disagg"])
def test_messages_route_hides_invalid_generated_tool_arguments(server_kind):
    client, _ = _make_route_client(server_kind, _json_chat_response(tool_arguments="{not-json"))

    response = client.post("/v1/messages", json=_request())

    assert response.status_code == 500
    assert response.json() == {
        "type": "error",
        "error": {"type": "api_error", "message": "Internal server error"},
    }


@pytest.mark.parametrize("server_kind", ["standard", "disagg"])
def test_messages_route_reframes_streaming_response(server_kind):
    client, backend = _make_route_client(server_kind, _streaming_chat_response())

    response = client.post("/v1/messages", json=_request(stream=True))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text.startswith("event: message_start\n")
    assert "event: content_block_delta\n" in response.text
    assert '"text":"hello"' in response.text
    assert response.text.rstrip().endswith('event: message_stop\ndata: {"type":"message_stop"}')
    assert backend.await_args.args[0].stream


def test_standard_and_disagg_register_messages_route(monkeypatch, tmp_path):
    # The disagg server's register_routes mounts a prometheus multiprocess
    # collector, which errors unless PROMETHEUS_MULTIPROC_DIR points at a real
    # directory. Production sets that up in set_prometheus_multiproc_dir()
    # during startup; building the server with object.__new__ skips startup
    # entirely, so the directory has to be supplied here. monkeypatch keeps it
    # from leaking into any other test in the session.
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))

    standard = object.__new__(OpenAIServer)
    standard.app = FastAPI()
    standard.generator = SimpleNamespace(
        _executor=SimpleNamespace(resource_governor_queue=None),
        args=SimpleNamespace(return_perf_metrics=False),
    )
    standard.use_harmony = False
    standard.register_routes()

    disagg = object.__new__(OpenAIDisaggServer)
    disagg.app = FastAPI()
    disagg._service = SimpleNamespace(
        openai_completion=AsyncMock(), openai_chat_completion=AsyncMock()
    )
    disagg._perf_metrics_collector = SimpleNamespace(get_perf_metrics=AsyncMock())
    disagg._disagg_cluster_storage = None
    # register_routes reads this; the server is built with object.__new__ here,
    # so every attribute it touches has to be stubbed explicitly.
    disagg._coordinator = None
    disagg.register_routes()

    for server in (standard, disagg):
        paths = {route.path for route in server.app.routes}
        assert "/v1/messages" in paths

    # Claude Code calls count_tokens before most turns to size its context.
    # An unregistered route 404s every call, which the client cannot act on,
    # so registration is asserted separately from /v1/messages.
    assert "/v1/messages/count_tokens" in {r.path for r in standard.app.routes}

    # The disagg server deliberately does not serve it: it holds no tokenizer,
    # so a count there would need a worker round-trip. Pinned so that adding
    # one is a conscious change rather than an accident.
    assert "/v1/messages/count_tokens" not in {r.path for r in disagg.app.routes}


def _batch_client(runner=None):
    """A TestClient over just the batch routes, with a stubbed item runner."""
    from tensorrt_llm.serve.anthropic_batches import AnthropicBatchStore

    async def ok(request):
        return "succeeded", {"type": "message", "role": "assistant", "content": []}

    app = FastAPI()
    server = object.__new__(OpenAIServer)
    server.model = MODEL
    server._anthropic_batch_store = AnthropicBatchStore(runner=runner or ok)
    app.add_api_route("/v1/messages/batches", server.anthropic_create_batch, methods=["POST"])
    app.add_api_route("/v1/messages/batches", server.anthropic_list_batches, methods=["GET"])
    app.add_api_route(
        "/v1/messages/batches/{batch_id}", server.anthropic_get_batch, methods=["GET"]
    )
    app.add_api_route(
        "/v1/messages/batches/{batch_id}", server.anthropic_delete_batch, methods=["DELETE"]
    )
    app.add_api_route(
        "/v1/messages/batches/{batch_id}/cancel", server.anthropic_cancel_batch, methods=["POST"]
    )
    app.add_api_route(
        "/v1/messages/batches/{batch_id}/results", server.anthropic_batch_results, methods=["GET"]
    )
    return TestClient(app)


def _batch_body(*custom_ids):
    return {
        "requests": [
            {
                "custom_id": cid,
                "params": {
                    "model": MODEL,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            }
            for cid in custom_ids
        ]
    }


def test_batch_routes_are_registered_on_the_standard_server():
    standard = object.__new__(OpenAIServer)
    standard.app = FastAPI()
    standard.generator = SimpleNamespace(
        _executor=SimpleNamespace(resource_governor_queue=None),
        args=SimpleNamespace(return_perf_metrics=False),
    )
    standard.use_harmony = False
    standard.register_routes()

    paths = {route.path for route in standard.app.routes}
    for path in (
        "/v1/messages/batches",
        "/v1/messages/batches/{batch_id}",
        "/v1/messages/batches/{batch_id}/cancel",
        "/v1/messages/batches/{batch_id}/results",
    ):
        assert path in paths, f"{path} not registered"


def test_create_batch_returns_an_in_progress_batch():
    client = _batch_client()
    response = client.post("/v1/messages/batches", json=_batch_body("a", "b"))

    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "message_batch"
    assert body["id"].startswith("msgbatch_")
    assert body["processing_status"] in {"in_progress", "ended"}
    assert body["request_counts"]["processing"] + body["request_counts"]["succeeded"] == 2


def test_create_batch_rejects_duplicate_custom_ids():
    client = _batch_client()
    response = client.post("/v1/messages/batches", json=_batch_body("same", "same"))

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_unknown_batch_says_it_is_not_found_and_explains_why():
    """The in-memory limitation is the likeliest cause, so name it."""
    client = _batch_client()
    response = client.get("/v1/messages/batches/msgbatch_nope")

    assert response.status_code == 404
    error = response.json()["error"]
    assert error["type"] == "not_found_error"
    assert "restart" in error["message"]


def test_results_of_an_unfinished_batch_are_400_not_404():
    """404 would tell a polling client the batch never existed.

    Driven directly rather than through TestClient: each TestClient call runs
    in its own portal, so the batch's background task is not guaranteed to
    still be in flight by the time the second request arrives, and the batch
    would have already ended.
    """
    from tensorrt_llm.serve.anthropic_batches import AnthropicBatchStore
    from tensorrt_llm.serve.anthropic_protocol import AnthropicBatchRequestItem

    async def scenario():
        blocked = asyncio.Event()

        async def never_finishes(request):
            await blocked.wait()
            return "succeeded", {"type": "message"}

        server = object.__new__(OpenAIServer)
        server._anthropic_batch_store = AnthropicBatchStore(runner=never_finishes)
        item = AnthropicBatchRequestItem(
            custom_id="a",
            params={
                "model": MODEL,
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        batch = server._anthropic_batch_store.create([item])
        await asyncio.sleep(0)  # let the worker start and block

        response = await server.anthropic_batch_results(batch.id)
        assert response.status_code == 400
        assert "still processing" in json.loads(response.body)["error"]["message"]

        # An id that was never issued is a genuine 404, and must stay distinct.
        missing = await server.anthropic_batch_results("msgbatch_nope")
        assert missing.status_code == 404

        blocked.set()

    asyncio.run(scenario())


def test_interrupted_batch_still_reports_counts_that_sum():
    """A cancelled worker must not leave counts that under-sum.

    _run finalizes from a finally block, so a torn-down task would otherwise
    end the batch with fewer results than requests - and request_counts is the
    number clients use to decide whether every request was accounted for.
    """
    from tensorrt_llm.serve.anthropic_batches import AnthropicBatchStore
    from tensorrt_llm.serve.anthropic_protocol import AnthropicBatchRequestItem

    async def scenario():
        blocked = asyncio.Event()

        async def never_finishes(request):
            await blocked.wait()
            return "succeeded", {"type": "message"}

        store = AnthropicBatchStore(runner=never_finishes)
        items = [
            AnthropicBatchRequestItem(
                custom_id=f"id-{i}",
                params={
                    "model": MODEL,
                    "max_tokens": 8,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            for i in range(4)
        ]
        batch = store.create(items)
        await asyncio.sleep(0)

        batch_record_task = store._batches[batch.id].task
        batch_record_task.cancel()
        try:
            await batch_record_task
        except asyncio.CancelledError:
            pass

        counts = store.get(batch.id).request_counts
        total = (
            counts.succeeded + counts.errored + counts.canceled + counts.expired + counts.processing
        )
        assert total == 4, f"counts sum to {total}, expected 4"

    asyncio.run(scenario())


def test_finished_batch_serves_ndjson_results():
    client = _batch_client()
    created = client.post("/v1/messages/batches", json=_batch_body("a", "b")).json()

    for _ in range(200):
        if (
            client.get(f"/v1/messages/batches/{created['id']}").json()["processing_status"]
            == "ended"
        ):
            break
        time.sleep(0.01)

    response = client.get(f"/v1/messages/batches/{created['id']}/results")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    lines = [json.loads(line) for line in response.text.splitlines()]
    assert {line["custom_id"] for line in lines} == {"a", "b"}


def test_list_batches_returns_a_page():
    client = _batch_client()
    client.post("/v1/messages/batches", json=_batch_body("a"))
    body = client.get("/v1/messages/batches").json()

    assert isinstance(body["data"], list) and body["data"]
    assert body["first_id"] == body["data"][0]["id"]
    assert body["has_more"] is False


def _handle(exception):
    """Drive OpenAIDisaggServer._handle_exception and capture what it raises."""
    server = object.__new__(OpenAIDisaggServer)
    server._perf_metrics_collector = SimpleNamespace(
        http_exceptions=SimpleNamespace(inc=Mock()),
        internal_errors=SimpleNamespace(inc=Mock()),
    )
    with pytest.raises(Exception) as excinfo:
        server._handle_exception(exception)
    return server, excinfo.value


@pytest.mark.parametrize(
    ("exception", "expected_status", "expected_counter"),
    [
        pytest.param(
            aiohttp.ClientResponseError(
                request_info=Mock(), history=(), status=400, message="bad request"
            ),
            400,
            "http_exceptions",
            id="upstream_http_status_is_propagated_not_masked_as_500",
        ),
        pytest.param(
            RuntimeError("boom"),
            500,
            "internal_errors",
            id="non_http_exceptions_still_become_500",
        ),
    ],
)
def test_handle_exception_maps_status_and_counter(exception, expected_status, expected_counter):
    """Pins a deliberate cross-cutting change, and that it is narrow.

    _handle_exception is shared by every route on the disagg server, including
    /v1/completions, which predates this work. Before the aiohttp branch existed
    an upstream ClientResponseError fell through to the generic handler and every
    upstream failure reached the client as 500. The Anthropic adapter maps
    upstream status onto Anthropic error types, so a worker's 400 arriving as a
    500 would be reported to the client as an api_error rather than an
    invalid_request_error.

    The change is an improvement for the older routes too - a 429 or 503 now
    survives instead of being flattened - but it IS a behaviour change on
    endpoints outside this feature, so it is pinned here rather than left
    implicit. The RuntimeError case pins the other half: the generic path is
    unchanged, only the aiohttp case was carved out of it.

    The counter matters as much as the status. Each exception must increment
    exactly one of the two, so an upstream 400 is not also filed as an internal
    error of this server.
    """
    server, raised = _handle(exception)
    collector = server._perf_metrics_collector
    incremented = {
        name
        for name in ("http_exceptions", "internal_errors")
        if getattr(collector, name).inc.called
    }

    assert raised.status_code == expected_status
    assert incremented == {expected_counter}


class _FakeTokenizer:
    """Renders a prompt the way a chat template would, and counts words."""

    def apply_chat_template(self, messages, **kwargs):
        parts = [str(m.get("content", "")) for m in messages]
        # Tools and the thinking prefix change the real prompt, so reflect them
        # here too - a count that ignored them would not be worth taking.
        for tool in kwargs.get("tools") or []:
            parts.append(str(tool))
        # A real template changes the rendered prompt, so a fake that dropped
        # it would make any test of template plumbing pass vacuously.
        template = kwargs.get("chat_template")
        if template:
            parts.append(str(template))
        return " ".join(parts)

    def encode(self, text):
        return text.split()


def _count_tokens_client():
    app = FastAPI()
    server = object.__new__(OpenAIServer)
    server.model = MODEL
    server.tokenizer = _FakeTokenizer()
    app.add_api_route("/v1/messages/count_tokens", server.anthropic_count_tokens, methods=["POST"])
    return TestClient(app)


def test_count_tokens_route_returns_a_count():
    client = _count_tokens_client()

    response = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "one two three"}],
        },
    )

    assert response.status_code == 200
    assert response.json() == {"input_tokens": 3}


def test_count_tokens_counts_the_rendered_prompt_not_the_raw_messages():
    """Tools inflate the real prompt, so they have to inflate the count."""
    client = _count_tokens_client()
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "one two three"}],
    }

    bare = client.post("/v1/messages/count_tokens", json=body).json()
    with_tools = client.post(
        "/v1/messages/count_tokens",
        json={
            **body,
            "tools": [
                {
                    "name": "get_weather",
                    "input_schema": {"type": "object"},
                }
            ],
        },
    ).json()

    assert with_tools["input_tokens"] > bare["input_tokens"]


def test_count_tokens_honours_the_server_chat_template():
    """A server started with --chat_template must count THAT prompt.

    The renderer only consults request.chat_template; openai_chat resolves
    `request.chat_template or self.chat_template`. Without the same resolution
    here, count_tokens reports a prompt the server never builds - the exact
    drift this endpoint exists to remove.
    """
    app = FastAPI()
    server = object.__new__(OpenAIServer)
    server.model = MODEL
    server.tokenizer = _FakeTokenizer()
    # A template that measurably inflates the rendered prompt.
    server.chat_template = "SERVER-TEMPLATE-MARKER {{ messages }}"
    app.add_api_route("/v1/messages/count_tokens", server.anthropic_count_tokens, methods=["POST"])
    client = TestClient(app)

    with_template = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "one two three"}],
        },
    ).json()

    server.chat_template = None
    without = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "one two three"}],
        },
    ).json()

    assert with_template["input_tokens"] != without["input_tokens"], (
        "server chat_template had no effect on the count, so it was ignored"
    )


def test_count_tokens_rejects_a_malformed_request():
    client = _count_tokens_client()

    response = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "bogus"},
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
