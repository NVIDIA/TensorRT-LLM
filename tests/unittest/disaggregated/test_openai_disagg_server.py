# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import aiohttp
import pytest
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.datastructures import Headers

from tensorrt_llm.llmapi.disagg_utils import ServerRole, extract_disagg_cfg
from tensorrt_llm.serve import openai_disagg_server
from tensorrt_llm.serve.anthropic_protocol import AnthropicMessagesRequest
from tensorrt_llm.serve.openai_disagg_server import OpenAIDisaggServer
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ConversationParams,
    DisaggregatedParams,
    ResponsesRequest,
)

pytestmark = pytest.mark.cpu_only


def _raw_request(headers: dict[str, str]):
    return SimpleNamespace(headers=Headers(headers=headers))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config_kwargs", "expected_timeout"),
    [({}, 10), ({"server_keep_alive_timeout": 3600}, 3600)],
)
async def test_server_keep_alive_timeout_is_passed_to_uvicorn(
    monkeypatch, config_kwargs, expected_timeout
):
    config = extract_disagg_cfg(
        context_servers={"num_instances": 0},
        generation_servers={"num_instances": 0},
        **config_kwargs,
    )
    server = object.__new__(OpenAIDisaggServer)
    server._config = config
    server.app = object()

    uvicorn_config = object()
    config_factory = Mock(return_value=uvicorn_config)
    uvicorn_server = SimpleNamespace(serve=AsyncMock())
    server_factory = Mock(return_value=uvicorn_server)
    monkeypatch.setattr(openai_disagg_server.uvicorn, "Config", config_factory)
    monkeypatch.setattr(openai_disagg_server.uvicorn, "Server", server_factory)

    await server(host="localhost", port=8000)

    assert config_factory.call_args.kwargs["timeout_keep_alive"] == expected_timeout
    server_factory.assert_called_once_with(uvicorn_config)
    uvicorn_server.serve.assert_awaited_once_with(sockets=None)


@pytest.mark.asyncio
async def test_http_cluster_storage_request_is_proxied_to_coordinator():
    payload = b'{"key":"worker","value":"ready"}'

    async def receive():
        return {"type": "http.request", "body": payload, "more_body": False}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/set",
            "query_string": b"source=worker",
            "headers": [(b"content-type", b"application/json")],
        },
        receive,
    )
    server = OpenAIDisaggServer.__new__(OpenAIDisaggServer)
    server._coordinator = SimpleNamespace(
        proxy_cluster_storage_request=AsyncMock(
            return_value=(b'{"result":true}', 200, "application/json")
        )
    )

    response = await server._proxy_cluster_storage_request(request)

    server._coordinator.proxy_cluster_storage_request.assert_awaited_once_with(
        "POST", "/set", [("source", "worker")], payload, "application/json"
    )
    assert response.status_code == 200
    assert response.body == b'{"result":true}'


@pytest.mark.parametrize("bounded_replay", [False, True])
def test_create_client_does_not_register_with_server_metrics_collector(bounded_replay):
    server = OpenAIDisaggServer.__new__(OpenAIDisaggServer)
    server._coordinator = SimpleNamespace(get_disagg_request_id=AsyncMock(return_value=1))
    server._req_timeout_secs = 30
    server._collect_perf_metrics = True
    server._config = SimpleNamespace(
        internal_request_auth_key="key", bounded_replay_on_generation=bounded_replay
    )
    server._perf_metrics_collector = SimpleNamespace()

    with patch("tensorrt_llm.serve.openai_disagg_server.OpenAIHttpClient") as mock_client:
        client = server._create_client(SimpleNamespace(), ServerRole.GENERATION, max_retries=2)

    assert client is mock_client.return_value
    mock_client.assert_called_once()
    assert mock_client.call_args.kwargs["bounded_replay_on_generation"] is bounded_replay


def test_extract_conversation_id_from_headers():
    cases = [
        ({"X-Session-ID": "session-id"}, "session-id"),
        ({"X-Correlation-ID": "correlation-id"}, "correlation-id"),
        ({"x-session-affinity": "session-affinity"}, "session-affinity"),
        ({"x-multi-turn-session-id": "multi-turn-session-id"}, "multi-turn-session-id"),
        (
            {
                "X-Correlation-ID": "correlation-id",
                "X-Session-ID": "session-id",
                "x-session-affinity": "session-affinity",
                "x-multi-turn-session-id": "multi-turn-session-id",
            },
            "session-id",
        ),
        (
            {
                "x-session-affinity": "session-affinity",
                "x-multi-turn-session-id": "multi-turn-session-id",
            },
            "session-affinity",
        ),
        (
            {
                "X-Session-ID": "",
                "X-Correlation-ID": "correlation-id",
            },
            "correlation-id",
        ),
    ]

    for headers, expected_conversation_id in cases:
        request = CompletionRequest(model="test-model", prompt="hello")

        OpenAIDisaggServer._extract_conversation_id(request, _raw_request(headers))

        assert request.disaggregated_params is None
        assert request.conversation_params.conversation_id == expected_conversation_id


def test_extract_conversation_id_ignores_empty_headers():
    request = CompletionRequest(model="test-model", prompt="hello")

    OpenAIDisaggServer._extract_conversation_id(
        request,
        _raw_request(
            {
                "X-Session-ID": "",
                "X-Correlation-ID": " ",
                "x-session-affinity": "",
                "x-multi-turn-session-id": " ",
            }
        ),
    )

    assert request.disaggregated_params is None
    assert request.conversation_params is None


def test_extract_conversation_id_preserves_body_conversation_params():
    request = CompletionRequest(
        model="test-model",
        prompt="hello",
        conversation_params=ConversationParams(conversation_id="body-id"),
        disaggregated_params=DisaggregatedParams(request_type="context_only"),
    )

    OpenAIDisaggServer._extract_conversation_id(
        request,
        _raw_request({"X-Session-ID": "header-id"}),
    )

    assert request.conversation_params.conversation_id == "body-id"


def test_extract_conversation_id_populates_conversation_params_with_existing_disaggregated_params():
    request = CompletionRequest(
        model="test-model",
        prompt="hello",
        disaggregated_params=DisaggregatedParams(request_type="context_only"),
    )

    OpenAIDisaggServer._extract_conversation_id(
        request,
        _raw_request({"x-multi-turn-session-id": "multi-turn-session-id"}),
    )

    assert request.conversation_params.conversation_id == "multi-turn-session-id"


# --- sub-agent conversation affinity (routing key, conversation_id NOT rewritten) ---

_PARENT_HEADER = "X-Dynamo-Parent-Session-ID"


def _routing_id(request):
    from tensorrt_llm.serve.conversation_id import get_request_routing_id

    return get_request_routing_id(request)


def _affinity_id(request):
    from tensorrt_llm.serve.conversation_id import get_request_subagent_affinity_id

    return get_request_subagent_affinity_id(request)


def test_subagent_affinity_sets_routing_key_without_rewriting_conversation_id():
    # A sub-agent request keeps its OWN conversation_id (linear history for the
    # worker's per-conversation KV bookkeeping); only the server-private routing
    # key is set to the parent, so the ConversationRouter co-locates it.
    request = CompletionRequest(
        model="test-model",
        prompt="hello",
        conversation_params=ConversationParams(conversation_id="own-id"),
    )
    OpenAIDisaggServer._extract_conversation_id(
        request,
        _raw_request({_PARENT_HEADER: "parent-id"}),
        _PARENT_HEADER,
    )
    assert request.conversation_params.conversation_id == "own-id"  # NOT rewritten
    assert _affinity_id(request) == "parent-id"
    assert _routing_id(request) == "parent-id"  # routes to the parent's instance


def test_subagent_affinity_main_agent_has_no_routing_key():
    # A main-agent request lacks the parent header -> no affinity; routes by its
    # own id.
    request = CompletionRequest(model="test-model", prompt="hello")
    OpenAIDisaggServer._extract_conversation_id(
        request,
        _raw_request({"X-Session-ID": "own-id"}),
        _PARENT_HEADER,
    )
    assert request.conversation_params.conversation_id == "own-id"
    assert _affinity_id(request) is None
    assert _routing_id(request) == "own-id"


def test_subagent_affinity_feature_off_ignores_parent_header():
    # No configured header name -> the parent header is inert.
    request = CompletionRequest(
        model="test-model",
        prompt="hello",
        conversation_params=ConversationParams(conversation_id="own-id"),
    )
    OpenAIDisaggServer._extract_conversation_id(
        request, _raw_request({_PARENT_HEADER: "parent-id"}), None
    )
    assert request.conversation_params.conversation_id == "own-id"
    assert _affinity_id(request) is None


def test_subagent_affinity_parent_only_synthesizes_own_id():
    # No body id and no session header: synthesize a distinct own id so the
    # worker's bookkeeping / gen fleet see a distinct linear session, while the
    # routing key still pins to the parent.
    request = CompletionRequest(model="test-model", prompt="hello")
    OpenAIDisaggServer._extract_conversation_id(
        request, _raw_request({_PARENT_HEADER: "parent-id"}), _PARENT_HEADER
    )
    assert request.conversation_params.conversation_id.startswith("subagent:")
    assert _affinity_id(request) == "parent-id"
    assert _routing_id(request) == "parent-id"


@pytest.mark.parametrize("header", [None, _PARENT_HEADER])
def test_subagent_affinity_clears_client_supplied_routing_key(header):
    # subagent_affinity_id is server-private: a client cannot enable affinity by
    # putting it in the request body. With the feature OFF (header=None) it ends
    # up None; with a configured header but no parent header present it is also
    # cleared (only the trusted parent header re-sets it).
    request = CompletionRequest(
        model="test-model",
        prompt="hello",
        conversation_params=ConversationParams(
            conversation_id="own-id", subagent_affinity_id="attacker-id"
        ),
    )
    OpenAIDisaggServer._extract_conversation_id(
        request, _raw_request({"X-Session-ID": "own-id"}), header
    )
    assert request.conversation_params.conversation_id == "own-id"
    assert _affinity_id(request) is None


def test_disagg_config_allows_request_chat_template_opt_in():
    config = extract_disagg_cfg(
        context_servers={"num_instances": 0},
        generation_servers={"num_instances": 0},
        allow_request_chat_template=True,
    )

    assert config.allow_request_chat_template is True


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
def test_disagg_config_rejects_non_bool_request_chat_template_opt_in(value):
    with pytest.raises(ValueError, match="allow_request_chat_template must be a boolean"):
        extract_disagg_cfg(
            context_servers={"num_instances": 0},
            generation_servers={"num_instances": 0},
            allow_request_chat_template=value,
        )


def _streaming_server():
    server = OpenAIDisaggServer.__new__(OpenAIDisaggServer)
    server._perf_metrics_collector = MagicMock()
    server._allow_request_chat_template = False
    server._collect_perf_metrics = False
    server._config = SimpleNamespace(conversation_affinity_header_for_subagents=None)
    return server


def _streaming_raw_request():
    return SimpleNamespace(
        headers=Headers(headers={}), state=SimpleNamespace(server_arrival_time=0.0)
    )


def _stream_request(kind):
    if kind == "completion":
        return CompletionRequest(model="test-model", prompt="hello", stream=True)
    assert kind == "chat"
    return ChatCompletionRequest(
        model="test-model", messages=[{"role": "user", "content": "hello"}], stream=True
    )


async def _failing_generation_stream():
    raise aiohttp.ClientPayloadError("Response payload is not completed")
    yield b"unreachable"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["completion", "chat"])
async def test_stream_failing_before_first_chunk_returns_http_error(kind):
    """A generation server that fails its KV transfer yields no chunk; answer 500, not a cut 200."""

    async def entry_point(request, hooks):
        return _failing_generation_stream()

    request = _stream_request(kind)
    wrapper = _streaming_server()._wrap_entry_point(entry_point, type(request))

    with pytest.raises(HTTPException) as exc_info:
        await wrapper(request, _streaming_raw_request())

    assert exc_info.value.status_code == 500
    assert "Response payload is not completed" in exc_info.value.detail


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks", [[b"data: 1\n\n", b"data: 2\n\n", b"data: [DONE]\n\n"], []])
async def test_stream_forwards_every_chunk(chunks):
    closed = []

    async def generation_stream():
        try:
            for chunk in chunks:
                yield chunk
        finally:
            closed.append(True)

    async def entry_point(request, hooks):
        return generation_stream()

    wrapper = _streaming_server()._wrap_entry_point(entry_point, CompletionRequest)
    request = CompletionRequest(model="test-model", prompt="hello", stream=True)

    response = await wrapper(request, _streaming_raw_request())

    assert isinstance(response, StreamingResponse)
    assert [chunk async for chunk in response.body_iterator] == chunks
    assert closed == [True]


# Responses streams are left out: guard_responses_stream does not close its
# source when it is itself closed before the end.
@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["completion", "chat"])
async def test_stream_closed_before_iteration_closes_upstream(kind):
    """A client that leaves before the body is sent still releases the upstream stream."""
    closed = []

    async def generation_stream():
        try:
            yield b"data: 1\n\n"
            yield b"data: 2\n\n"
        finally:
            closed.append(True)

    async def entry_point(request, hooks):
        return generation_stream()

    request = _stream_request(kind)
    wrapper = _streaming_server()._wrap_entry_point(entry_point, type(request))

    response = await wrapper(request, _streaming_raw_request())
    assert isinstance(response, StreamingResponse)
    await response.body_iterator.aclose()

    assert closed == [True]


@pytest.mark.asyncio
async def test_stream_error_after_first_chunk_propagates():
    closed = []

    async def generation_stream():
        try:
            yield b"data: 1\n\n"
            raise RuntimeError("mid-stream")
        finally:
            closed.append(True)

    async def entry_point(request, hooks):
        return generation_stream()

    wrapper = _streaming_server()._wrap_entry_point(entry_point, CompletionRequest)

    response = await wrapper(_stream_request("completion"), _streaming_raw_request())

    assert isinstance(response, StreamingResponse)
    received = []
    with pytest.raises(RuntimeError, match="mid-stream"):
        async for chunk in response.body_iterator:
            received.append(chunk)
    assert received == [b"data: 1\n\n"]
    assert closed == [True]


@pytest.mark.asyncio
async def test_stream_upstream_error_status_keeps_worker_status():
    async def rejected_stream():
        raise aiohttp.ClientResponseError(
            MagicMock(),
            (),
            status=400,
            message='Bad Request: {"error": {"message": "prompt too long"}}',
        )
        yield b"unreachable"

    async def entry_point(request, hooks):
        return rejected_stream()

    wrapper = _streaming_server()._wrap_entry_point(entry_point, CompletionRequest)

    with pytest.raises(HTTPException) as exc_info:
        await wrapper(_stream_request("completion"), _streaming_raw_request())

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "prompt too long"


@pytest.mark.asyncio
async def test_responses_stream_failing_before_first_frame_ends_with_error_event():
    """A Responses relay keeps its 200 and ends with an `error` event."""

    async def failing_relay():
        raise RuntimeError("no generation worker")
        yield b"unreachable"

    async def entry_point(request, hooks):
        return failing_relay()

    wrapper = _streaming_server()._wrap_entry_point(entry_point, ResponsesRequest)
    request = ResponsesRequest.model_validate({"model": "m", "input": "hi", "stream": True})

    response = await wrapper(request, _streaming_raw_request())

    assert isinstance(response, StreamingResponse)
    frames = []
    with pytest.raises(RuntimeError, match="no generation worker"):
        async for frame in response.body_iterator:
            frames.append(frame)
    assert len(frames) == 1
    assert frames[0].startswith(b"event: error")
    assert json.loads(frames[0].decode().split("data: ", 1)[1])["sequence_number"] == 0


@pytest.mark.asyncio
async def test_failed_stream_is_not_counted_as_response():
    async def ok_stream():
        yield b"data: 1\n\n"

    streams = iter([_failing_generation_stream(), ok_stream()])

    async def entry_point(request, hooks):
        return next(streams)

    server = _streaming_server()
    metrics = server._perf_metrics_collector
    wrapper = server._wrap_entry_point(entry_point, CompletionRequest)

    with pytest.raises(HTTPException):
        await wrapper(_stream_request("completion"), _streaming_raw_request())

    metrics.total_responses.inc.assert_not_called()
    metrics.internal_errors.inc.assert_called_once_with()

    response = await wrapper(_stream_request("completion"), _streaming_raw_request())

    assert [chunk async for chunk in response.body_iterator] == [b"data: 1\n\n"]
    metrics.total_responses.inc.assert_called_once_with()


@pytest.mark.asyncio
async def test_anthropic_stream_failing_before_first_chunk_is_api_error():
    """/v1/messages maps a chat stream that fails before its first chunk to an error envelope."""
    server = _streaming_server()
    server._service = SimpleNamespace(
        openai_chat_completion=AsyncMock(return_value=_failing_generation_stream())
    )
    request = AnthropicMessagesRequest.model_validate(
        {
            "model": "test-model",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        }
    )

    response = await server.anthropic_messages(request, _streaming_raw_request())

    assert response.status_code == 500
    assert json.loads(response.body)["error"]["type"] == "api_error"
