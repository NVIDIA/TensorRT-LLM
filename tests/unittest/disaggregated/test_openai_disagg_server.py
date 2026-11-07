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
import asyncio
import json
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from starlette.datastructures import Headers
from starlette.requests import ClientDisconnect

from tensorrt_llm.llmapi.disagg_utils import ServerRole, extract_disagg_cfg
from tensorrt_llm.serve import openai_disagg_server
from tensorrt_llm.serve.anthropic_protocol import AnthropicMessagesRequest
from tensorrt_llm.serve.disagg_request_deadline import DisaggRequestTimeoutError
from tensorrt_llm.serve.openai_disagg_server import OpenAIDisaggServer
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionStreamResponse,
    CompletionRequest,
    ConversationParams,
    DisaggregatedParams,
    ResponsesRequest,
)
from tensorrt_llm.serve.responses_utils import ClosingStreamingResponse

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
        internal_request_auth_key="key",
        bounded_replay_on_generation=bounded_replay,
        ctx_frontend_affinity=None,
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


_Protocol = Literal["completion", "chat", "responses", "anthropic"]
_PROTOCOLS = ["completion", "chat", "responses", "anthropic"]


def _deadline_server(entry_point: AsyncMock, enabled: bool = True) -> OpenAIDisaggServer:
    """Build the real route wrapper without starting workers."""
    server = object.__new__(OpenAIDisaggServer)
    server._config = extract_disagg_cfg(
        context_servers={"num_instances": 0},
        generation_servers={"num_instances": 0},
        request_deadline={"total_seconds": 5.0} if enabled else None,
    )
    server._allow_request_chat_template = False
    server._collect_perf_metrics = False
    server._perf_metrics_collector = Mock()
    server._service = SimpleNamespace(openai_chat_completion=entry_point)
    return server


async def _call_deadline_route(
    server: OpenAIDisaggServer, protocol: _Protocol, *, stream: bool
) -> Response:
    """Exercise the protocol entry point with an accepted HTTP request."""
    raw = Request({"type": "http", "headers": [], "state": {"server_arrival_time": 0}})
    messages = [{"role": "user", "content": "hello"}]
    if protocol == "anthropic":
        return await server.anthropic_messages(
            AnthropicMessagesRequest(
                model="test-model", messages=messages, max_tokens=8, stream=stream
            ),
            raw,
        )
    if protocol == "completion":
        request = CompletionRequest(model="test-model", prompt="hello", stream=stream)
    elif protocol == "chat":
        request = ChatCompletionRequest(model="test-model", messages=messages, stream=stream)
    else:
        request = ResponsesRequest(model="test-model", input="hello", stream=stream)
    return await server._wrap_entry_point(server._service.openai_chat_completion, type(request))(
        request, raw
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", _PROTOCOLS)
@pytest.mark.parametrize("stream", [False, True])
async def test_request_deadline_before_headers_is_http_504(
    protocol: _Protocol, stream: bool
) -> None:
    error = DisaggRequestTimeoutError(5.0)
    server = _deadline_server(AsyncMock(side_effect=error))

    if protocol == "anthropic":
        response = await _call_deadline_route(server, protocol, stream=stream)
        assert response.status_code == 504
        assert json.loads(response.body)["error"]["message"] == str(error)
    else:
        with pytest.raises(HTTPException) as caught:
            await _call_deadline_route(server, protocol, stream=stream)
        assert caught.value.status_code == 504
        assert caught.value.detail == str(error)
        assert caught.value.__cause__ is error
    server._perf_metrics_collector.http_exceptions.inc.assert_called_once()
    server._perf_metrics_collector.internal_errors.inc.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", _PROTOCOLS)
async def test_request_deadline_after_headers_is_terminal_error(protocol: _Protocol) -> None:
    closed = []
    error = DisaggRequestTimeoutError(5.0)

    async def source() -> AsyncGenerator[str, None]:
        try:
            yield ": ready\n\n"
            raise error
        finally:
            closed.append(True)

    server = _deadline_server(AsyncMock(return_value=source()))
    response = await _call_deadline_route(server, protocol, stream=True)
    assert isinstance(response, ClosingStreamingResponse)
    frames = [frame async for frame in response.body_iterator]
    body = "".join(frame.decode() if isinstance(frame, bytes) else frame for frame in frames)

    assert str(error) in body
    if protocol in ("completion", "chat"):
        payload = json.loads(body.split("data: ", 1)[1])
        assert payload["error"]["code"] == "request_timeout"
    else:
        assert "event: error\n" in body
        if protocol == "responses":
            assert '"code":"request_timeout"' in body
    assert "[DONE]" not in body
    assert "event: response.completed" not in body
    assert "event: message_stop" not in body
    assert closed == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", _PROTOCOLS)
async def test_deadline_stream_closes_source_when_sending_headers_fails(
    protocol: _Protocol,
) -> None:
    closed = []
    consumed = []

    async def source() -> AsyncGenerator[str, None]:
        try:
            # The service primes its cleanup scope before returning the stream.
            yield ""
            consumed.append(True)
            yield ": ready\n\n"
        finally:
            closed.append(True)

    upstream = source()
    await upstream.__anext__()
    response = await _call_deadline_route(
        _deadline_server(AsyncMock(return_value=upstream)), protocol, stream=True
    )

    async def send(message: dict[str, object]) -> None:
        assert message["type"] == "http.response.start"
        raise OSError("connection reset")

    scope = {"type": "http", "asgi": {"spec_version": "2.4"}}
    with pytest.raises(ClientDisconnect):
        await response(scope, AsyncMock(), send)

    assert consumed == []
    assert closed == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", _PROTOCOLS)
@pytest.mark.parametrize("during_stream", [False, True])
async def test_caller_cancellation_is_not_request_timeout(
    protocol: _Protocol, during_stream: bool
) -> None:
    closed = []

    async def source() -> AsyncGenerator[str, None]:
        try:
            yield ": ready\n\n"
            raise asyncio.CancelledError
        finally:
            closed.append(True)

    entry_point = (
        AsyncMock(return_value=source())
        if during_stream
        else AsyncMock(side_effect=asyncio.CancelledError)
    )
    server = _deadline_server(entry_point)
    with pytest.raises(asyncio.CancelledError):
        response = await _call_deadline_route(server, protocol, stream=True)
        async for _ in response.body_iterator:
            pass
    assert closed == ([True] if during_stream else [])
    server._perf_metrics_collector.http_exceptions.inc.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["completion", "chat", "responses"])
async def test_deadline_wrapper_preserves_other_stream_errors(protocol: _Protocol) -> None:
    error = TimeoutError("worker timeout")

    async def source() -> AsyncGenerator[str, None]:
        yield ": ready\n\n"
        raise error

    response = await _call_deadline_route(
        _deadline_server(AsyncMock(return_value=source())), protocol, stream=True
    )
    with pytest.raises(TimeoutError) as caught:
        async for _ in response.body_iterator:
            pass
    assert caught.value is error


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["responses", "anthropic"])
async def test_late_deadline_does_not_publish_withheld_terminal_event(protocol: _Protocol) -> None:
    if protocol == "responses":
        worker_frames = [
            ": ready\n\n",
            'event: response.completed\ndata: {"type":"response.completed"}\n\n',
        ]
    else:
        chunk = ChatCompletionStreamResponse(
            model="test-model",
            choices=[{"index": 0, "delta": {"content": "hello"}, "finish_reason": "stop"}],
        )
        worker_frames = [f"data: {chunk.model_dump_json()}\n\n", "data: [DONE]\n\n"]

    async def source() -> AsyncGenerator[str, None]:
        for frame in worker_frames:
            yield frame
        raise DisaggRequestTimeoutError(5.0)

    response = await _call_deadline_route(
        _deadline_server(AsyncMock(return_value=source())), protocol, stream=True
    )
    frames = [frame async for frame in response.body_iterator]
    body = "".join(frame.decode() if isinstance(frame, bytes) else frame for frame in frames)

    if protocol == "responses":
        # This terminal event was already relayed before worker EOF.
        assert "event: error" not in body
        assert "event: response.completed" in body
    else:
        # Anthropic holds message_stop until EOF; expiry must not create success.
        assert "event: error" in body
        assert "E2E budget" in body
        assert "event: message_stop" not in body


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", _PROTOCOLS)
async def test_disabled_request_deadline_keeps_existing_stream_response(
    protocol: _Protocol,
) -> None:
    async def source() -> AsyncGenerator[str, None]:
        yield "data: [DONE]\n\n"

    response = await _call_deadline_route(
        _deadline_server(AsyncMock(return_value=source()), enabled=False), protocol, stream=True
    )
    assert type(response) is StreamingResponse
    await response.body_iterator.aclose()
