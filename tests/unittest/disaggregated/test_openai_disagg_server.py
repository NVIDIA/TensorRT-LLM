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
import contextlib
import json
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from starlette.datastructures import Headers

from tensorrt_llm.executor.utils import context_length_exceeded_message
from tensorrt_llm.llmapi.disagg_utils import DisaggServerConfig, ServerRole, extract_disagg_cfg
from tensorrt_llm.serve import anthropic_adapter, openai_disagg_server
from tensorrt_llm.serve.anthropic_protocol import AnthropicMessagesRequest
from tensorrt_llm.serve.disagg_coordinator import DisaggCoordinatorService
from tensorrt_llm.serve.openai_disagg_server import OpenAIDisaggServer
from tensorrt_llm.serve.openai_disagg_service import OpenAIDisaggregatedService
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    CompletionRequest,
    ConversationParams,
    DisaggregatedParams,
    UsageInfo,
)
from tensorrt_llm.serve.responses_utils import ServerArrivalTimeMiddleware
from tensorrt_llm.serve.router import Router

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


# ---------------------------------------------------------------------------
# Streamed /v1/messages: a slow context phase must not leave the client silent
# ---------------------------------------------------------------------------

_MODEL = "test-model"
_SCHEDULE_STYLES = ["context_first", "generation_first"]
# The events of the message the generation worker streams (_generation_frames).
_MESSAGE_EVENTS = [
    "message_start",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
    "message_delta",
    "message_stop",
]
_OVER_LENGTH = context_length_exceeded_message(8, 9)


def _generation_frames() -> list[bytes]:
    """A generation worker's chat stream answering "hello"."""
    chunks = [
        ChatCompletionStreamResponse(
            model=_MODEL, choices=[{"index": 0, "delta": {"role": "assistant"}}]
        ),
        ChatCompletionStreamResponse(
            model=_MODEL, choices=[{"index": 0, "delta": {"content": "hello"}}]
        ),
        ChatCompletionStreamResponse(
            model=_MODEL,
            choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
            usage=UsageInfo(prompt_tokens=3, completion_tokens=1, total_tokens=4),
        ),
    ]
    frames = [f"data: {chunk.model_dump_json()}\n\n".encode() for chunk in chunks]
    return frames + [b"data: [DONE]\n\n"]


def _worker_rejection(status: int, message: str) -> aiohttp.ClientResponseError:
    """A worker's error response, as the HTTP client raises it."""
    body = json.dumps({"object": "error", "message": message, "code": status})
    return aiohttp.ClientResponseError(
        Mock(real_url="http://ctx:8001/v1/chat/completions"),
        (),
        status=status,
        message=f"Error: {body}",
    )


class _Workers:
    """A context and a generation worker behind the disaggregated service, without a network.

    The context phase runs ``context_phase`` and then answers with the KV-cache
    handoff. The generation worker streams ``_generation_frames`` once the
    context phase has succeeded, as it does once it has the KV cache.
    """

    def __init__(self, context_phase):
        self.context_phase = context_phase
        self.context_done = asyncio.Event()
        self.context_cancelled = asyncio.Event()
        self.generation_cancelled = asyncio.Event()
        # Set once the generation worker's stream has ended, however it ended.
        self.generation_closed = asyncio.Event()

    async def send_context_request(self, request, server=None, hooks=None, req_id=None):
        try:
            await self.context_phase()
        except asyncio.CancelledError:
            self.context_cancelled.set()
            raise
        self.context_done.set()
        disagg_request_id = request.disaggregated_params.disagg_request_id
        return ChatCompletionResponse(
            model=_MODEL,
            usage=UsageInfo(prompt_tokens=3, completion_tokens=1),
            prompt_token_ids=[1, 2, 3],
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=""),
                    finish_reason="length",
                    disaggregated_params=DisaggregatedParams(
                        request_type="context_only",
                        disagg_request_id=disagg_request_id,
                        ctx_request_id=disagg_request_id,
                    ),
                )
            ],
        )

    async def send_generation_request(self, request, server=None, hooks=None, req_id=None):
        async def stream():
            try:
                yield None
                await self.context_done.wait()
                async for frame in self.generation_stream():
                    yield frame
            except asyncio.CancelledError:
                self.generation_cancelled.set()
                raise
            finally:
                self.generation_closed.set()

        # Started before it is returned, as the HTTP client starts it, so that
        # closing it ends it even before it is read.
        response = stream()
        await response.__anext__()
        return response

    async def generation_stream(self):
        """The generation worker's stream once the context phase is done."""
        for frame in _generation_frames():
            yield frame


class _WorkersStillGenerating(_Workers):
    """Workers whose generation worker sends its first frame and then goes on generating."""

    async def generation_stream(self):
        yield _generation_frames()[0]
        await asyncio.Event().wait()


def _messages_server(schedule_style: str, workers: _Workers) -> OpenAIDisaggServer:
    """A disaggregated server in front of ``workers``."""
    config = DisaggServerConfig(server_configs=[], schedule_style=schedule_style)
    service = OpenAIDisaggregatedService(
        config, DisaggCoordinatorService(config, client_factory=Mock()), client_factory=Mock()
    )
    service.is_ready = AsyncMock(return_value=True)
    service._ctx_router = AsyncMock(spec=Router)
    service._ctx_router.get_next_server.return_value = ("ctx:8001", {})
    service._gen_router = AsyncMock(spec=Router)
    service._gen_router.get_next_server.return_value = ("gen:8002", {})
    service._ctx_client = SimpleNamespace(send_request=workers.send_context_request)
    service._gen_client = SimpleNamespace(send_request=workers.send_generation_request)

    server = OpenAIDisaggServer.__new__(OpenAIDisaggServer)
    server._config = config
    server._service = service
    server._perf_metrics_collector = Mock()
    server._allow_request_chat_template = False
    server._collect_perf_metrics = False
    return server


def _messages_app(schedule_style: str, workers: _Workers) -> FastAPI:
    """The /v1/messages route of a disaggregated server in front of ``workers``."""
    server = _messages_server(schedule_style, workers)
    app = FastAPI()
    app.add_middleware(ServerArrivalTimeMiddleware)
    app.add_api_route("/v1/messages", server.anthropic_messages, methods=["POST"])
    return app


async def _post_messages(app: FastAPI, *, stream=True, on_send=None, leave=None) -> list[dict]:
    """Send one /v1/messages request to ``app`` as a client and return the messages sent back.

    ``on_send`` sees each message as it is sent. The client disconnects once
    ``leave`` is set.
    """
    body = {
        "model": _MODEL,
        "max_tokens": 16,
        "stream": stream,
        "messages": [{"role": "user", "content": "hi"}],
    }
    requested = False

    async def receive():
        nonlocal requested
        if not requested:
            requested = True
            return {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}
        await (leave or asyncio.Event()).wait()
        return {"type": "http.disconnect"}

    sent = []

    async def send(message):
        sent.append(message)
        if on_send is not None:
            on_send(message)

    scope = {
        "type": "http",
        # As uvicorn declares it: a disconnect cancels a streaming response.
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/messages",
        "raw_path": b"/v1/messages",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 40000),
        "server": ("127.0.0.1", 8000),
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=10)
    return sent


def _status(sent: list[dict]) -> int:
    return next(message["status"] for message in sent if message["type"] == "http.response.start")


def _body(sent: list[dict]) -> str:
    return b"".join(
        message.get("body", b"") for message in sent if message["type"] == "http.response.body"
    ).decode()


def _sse_events(sent: list[dict]) -> list[tuple[str, dict]]:
    """The (event, data) pairs of a streamed response."""
    events = []
    for frame in _body(sent).split("\n\n"):
        if frame:
            fields = dict(line.split(": ", 1) for line in frame.splitlines())
            events.append((fields["event"], json.loads(fields["data"])))
    return events


def _short_waits(monkeypatch):
    monkeypatch.setattr(openai_disagg_server, "_MESSAGES_STREAM_START_WAIT_S", 0.05)
    monkeypatch.setattr(anthropic_adapter, "STREAM_PING_INTERVAL_S", 0.05)


def _waits_for(event: asyncio.Event):
    """A context phase that lasts until ``event`` is set, or five seconds without it."""

    async def context_phase():
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(event.wait(), 5)

    return context_phase


@pytest.mark.asyncio
@pytest.mark.parametrize("schedule_style", _SCHEDULE_STYLES)
async def test_messages_stream_starts_and_pings_while_the_context_phase_runs(
    monkeypatch, schedule_style
):
    _short_waits(monkeypatch)
    pinged = asyncio.Event()
    workers = _Workers(_waits_for(pinged))
    started_during_context_phase = []

    def on_send(message):
        if message["type"] == "http.response.start":
            started_during_context_phase.append(not workers.context_done.is_set())
        elif message.get("body", b"").startswith(b"event: ping\n"):
            pinged.set()

    sent = await _post_messages(_messages_app(schedule_style, workers), on_send=on_send)

    assert _status(sent) == 200
    assert started_during_context_phase == [True]
    events = [event for event, _ in _sse_events(sent)]
    message_start = events.index("message_start")
    assert message_start > 0
    assert set(events[:message_start]) == {"ping"}
    assert [event for event in events[message_start:] if event != "ping"] == _MESSAGE_EVENTS
    delta = {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "hello"},
    }
    assert ("content_block_delta", delta) in _sse_events(sent)


@pytest.mark.asyncio
@pytest.mark.parametrize("schedule_style", _SCHEDULE_STYLES)
async def test_messages_stream_after_a_fast_context_phase_is_the_reframed_generation(
    schedule_style,
):
    async def context_phase():
        pass

    sent = await _post_messages(_messages_app(schedule_style, _Workers(context_phase)))

    async def generation():
        for frame in _generation_frames():
            yield frame

    reframed = anthropic_adapter.reframe_openai_stream(generation(), model=_MODEL)
    expected = "".join([frame async for frame in reframed])

    def without_message_id(text):
        return re.sub(r'"id":"msg_[0-9a-f]+"', '"id":"msg"', text)

    assert _status(sent) == 200
    assert "event: ping" not in _body(sent)
    assert without_message_id(_body(sent)) == without_message_id(expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("schedule_style", _SCHEDULE_STYLES)
@pytest.mark.parametrize(
    "status, worker_message, error_type",
    [
        pytest.param(400, _OVER_LENGTH, "invalid_request_error", id="over_length_400"),
        pytest.param(503, "no capacity left", "api_error", id="503"),
    ],
)
async def test_messages_stream_with_a_fast_context_failure_is_an_http_error(
    schedule_style, status, worker_message, error_type
):
    async def context_phase():
        raise _worker_rejection(status, worker_message)

    streamed = await _post_messages(_messages_app(schedule_style, _Workers(context_phase)))
    not_streamed = await _post_messages(
        _messages_app(schedule_style, _Workers(context_phase)), stream=False
    )

    assert _status(streamed) == status
    error = json.loads(_body(streamed))
    assert error["type"] == "error"
    assert error["error"]["type"] == error_type
    # The error response of the same failure without a stream.
    assert _status(not_streamed) == status
    assert error == json.loads(_body(not_streamed))


@pytest.mark.asyncio
@pytest.mark.parametrize("schedule_style", _SCHEDULE_STYLES)
@pytest.mark.parametrize(
    "status, worker_message",
    [
        pytest.param(400, _OVER_LENGTH, id="over_length_400"),
        pytest.param(503, "no capacity left", id="503"),
    ],
)
async def test_messages_stream_with_a_late_context_failure_ends_with_an_error_event(
    monkeypatch, schedule_style, status, worker_message
):
    _short_waits(monkeypatch)
    pinged = asyncio.Event()
    wait = _waits_for(pinged)

    async def context_phase():
        await wait()
        raise _worker_rejection(status, worker_message)

    def on_send(message):
        if message.get("body", b"").startswith(b"event: ping\n"):
            pinged.set()

    streamed = await _post_messages(
        _messages_app(schedule_style, _Workers(context_phase)), on_send=on_send
    )
    pinged.set()
    not_streamed = await _post_messages(
        _messages_app(schedule_style, _Workers(context_phase)), stream=False
    )

    assert _status(streamed) == 200
    events = _sse_events(streamed)
    assert {event for event, _ in events[:-1]} == {"ping"}
    assert events[-1][0] == "error"
    # An error event as the stream reports any failure, with the message of
    # the error response of the same failure without a stream.
    assert events[-1][1] == {
        "type": "error",
        "error": {
            "type": "api_error",
            "message": json.loads(_body(not_streamed))["error"]["message"],
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("schedule_style", _SCHEDULE_STYLES)
async def test_client_leaving_after_the_response_starts_cancels_the_context_phase(
    monkeypatch, schedule_style
) -> None:
    _short_waits(monkeypatch)
    workers = _Workers(_waits_for(asyncio.Event()))
    left = asyncio.Event()

    def on_send(message):
        if message["type"] == "http.response.start":
            left.set()

    await _post_messages(_messages_app(schedule_style, workers), on_send=on_send, leave=left)

    await asyncio.wait_for(workers.context_cancelled.wait(), 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("schedule_style", _SCHEDULE_STYLES)
async def test_cancelling_the_messages_handler_during_its_wait_cancels_the_context_phase(
    monkeypatch, schedule_style
) -> None:
    monkeypatch.setattr(openai_disagg_server, "_MESSAGES_STREAM_START_WAIT_S", 60)
    context_started = asyncio.Event()
    release_context = asyncio.Event()

    async def context_phase() -> None:
        context_started.set()
        await release_context.wait()

    workers = _Workers(context_phase)
    server = _messages_server(schedule_style, workers)
    request = AnthropicMessagesRequest(
        model=_MODEL,
        max_tokens=16,
        stream=True,
        messages=[{"role": "user", "content": "hi"}],
    )
    raw_request = Request({"type": "http", "headers": []})
    raw_request.state.server_arrival_time = 0
    handler = asyncio.create_task(server.anthropic_messages(request, raw_request))
    try:
        await asyncio.wait_for(context_started.wait(), 2)
        assert not handler.done()
        handler.cancel()
        with pytest.raises(asyncio.CancelledError):
            await handler
        await asyncio.wait_for(workers.context_cancelled.wait(), 2)
        assert not workers.context_done.is_set()
    finally:
        release_context.set()
        handler.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await handler


@pytest.mark.asyncio
@pytest.mark.parametrize("schedule_style", _SCHEDULE_STYLES)
@pytest.mark.parametrize("asgi_version", ["2.3", "2.4"])
async def test_disconnect_before_the_messages_body_starts_cancels_context(
    monkeypatch, schedule_style, asgi_version
) -> None:
    """Response cleanup must run even if the streaming iterator never starts."""
    from starlette.requests import ClientDisconnect

    _short_waits(monkeypatch)
    release_context = asyncio.Event()
    workers = _Workers(_waits_for(release_context))
    server = _messages_server(schedule_style, workers)
    request = AnthropicMessagesRequest(
        model=_MODEL,
        max_tokens=16,
        stream=True,
        messages=[{"role": "user", "content": "hi"}],
    )
    raw_request = Request({"type": "http", "headers": []})
    raw_request.state.server_arrival_time = 0
    response = await server.anthropic_messages(request, raw_request)
    response_started = asyncio.Event()
    sent = []

    async def send(message):
        sent.append(message)
        response_started.set()
        if asgi_version == "2.4":
            raise OSError("client disconnected")
        await asyncio.Event().wait()

    async def receive():
        await response_started.wait()
        return {"type": "http.disconnect"}

    try:
        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": asgi_version}}
        if asgi_version == "2.4":
            with pytest.raises(ClientDisconnect):
                await response(scope, receive, send)
        else:
            await response(scope, receive, send)
        assert [message["type"] for message in sent] == ["http.response.start"]
        await asyncio.wait_for(workers.context_cancelled.wait(), 2)
        if schedule_style == "generation_first":
            await asyncio.wait_for(workers.generation_cancelled.wait(), 2)
        assert not workers.context_done.is_set()
    finally:
        release_context.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("schedule_style", _SCHEDULE_STYLES)
@pytest.mark.parametrize("asgi_version", ["2.3", "2.4"])
async def test_disconnect_before_the_messages_body_starts_closes_the_stream_it_answered_with(
    monkeypatch, schedule_style, asgi_version
) -> None:
    """The pipeline answers with a stream after the response is returned, which no body reads."""
    from starlette.requests import ClientDisconnect

    _short_waits(monkeypatch)
    release_context = asyncio.Event()
    workers = _WorkersStillGenerating(_waits_for(release_context))
    server = _messages_server(schedule_style, workers)
    request = AnthropicMessagesRequest(
        model=_MODEL,
        max_tokens=16,
        stream=True,
        messages=[{"role": "user", "content": "hi"}],
    )
    raw_request = Request({"type": "http", "headers": []})
    raw_request.state.server_arrival_time = 0
    response = await asyncio.wait_for(server.anthropic_messages(request, raw_request), 2)
    chat = response._chat
    answered = asyncio.Event()
    sent = []

    async def send(message):
        sent.append(message)
        # The pipeline answers while the response starts, before the body does.
        release_context.set()
        await asyncio.wait((chat,), timeout=2)
        answered.set()
        if asgi_version == "2.4":
            raise OSError("client disconnected")
        await asyncio.Event().wait()

    async def receive():
        await answered.wait()
        return {"type": "http.disconnect"}

    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": asgi_version}}
    if asgi_version == "2.4":
        with pytest.raises(ClientDisconnect):
            await response(scope, receive, send)
    else:
        await response(scope, receive, send)

    assert [message["type"] for message in sent] == ["http.response.start"]
    assert isinstance(chat.result(), StreamingResponse)
    # Closed by the time the response is done, while the generation worker goes on.
    assert workers.generation_closed.is_set()
