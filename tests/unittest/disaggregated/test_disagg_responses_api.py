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
"""Offline tests for the Responses API on the disaggregated path.

A Responses response has no ``choices``, so the context-to-generation handoff
that completions and chat carry on ``choices[0]`` sits at its top level.
"""

import asyncio
import base64
import contextlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import msgspec
import numpy as np
import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseTextDeltaEvent,
)
from pydantic import ValidationError

from tensorrt_llm.executor.utils import context_length_exceeded_message
from tensorrt_llm.llmapi.disagg_utils import DisaggServerConfig
from tensorrt_llm.serve import openai_disagg_server, responses_utils
from tensorrt_llm.serve._base.responses_stream import responses_done_generator
from tensorrt_llm.serve.disagg_coordinator import DisaggCoordinatorService
from tensorrt_llm.serve.openai_client import _encode_request_body
from tensorrt_llm.serve.openai_disagg_server import OpenAIDisaggServer
from tensorrt_llm.serve.openai_disagg_service import (
    OpenAIDisaggregatedService,
    _ctx_handoff_slots,
    _ctx_usage_info,
)
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionResponse,
    DisaggregatedParams,
    DisaggScheduleStyle,
    ModelCard,
    ModelList,
    PromptTokensDetails,
    ResponsesRequest,
    ResponsesResponse,
    UsageInfo,
)
from tensorrt_llm.serve.responses_utils import (
    ResponsesStreamingProcessor,
    ServerArrivalTimeMiddleware,
    create_response_non_store,
    finish_reason_mapping,
)
from tensorrt_llm.serve.router import Router

pytestmark = pytest.mark.cpu_only


def _responses_response(finish_reason="length", status="incomplete"):
    return ResponsesResponse(
        model="m",
        output=[],
        parallel_tool_calls=False,
        temperature=1.0,
        tool_choice="auto",
        tools=[],
        top_p=1.0,
        background=False,
        service_tier="auto",
        status=status,
        top_logprobs=0,
        truncation="disabled",
        finish_reason=finish_reason,
        prompt_token_ids=[1, 2],
        prompt_token_ids_b64="AQAAAAIAAAA=",
        disaggregated_params=DisaggregatedParams(
            request_type="context_only", ctx_request_id=7, disagg_request_id=7
        ),
    )


def _service(**attrs):
    service = OpenAIDisaggregatedService.__new__(OpenAIDisaggregatedService)

    async def ready():
        return True

    service.is_ready = ready
    for name, value in attrs.items():
        setattr(service, name, value)
    return service


# ---------------------------------------------------------------------------
# The handoff
# ---------------------------------------------------------------------------


def test_the_handoff_is_read_per_choice_for_chat_and_top_level_for_responses():
    chat = ChatCompletionResponse(
        model="m",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "length",
                "disaggregated_params": {"request_type": "context_only", "ctx_request_id": 3},
            }
        ],
        usage={"prompt_tokens": 1, "total_tokens": 2, "completion_tokens": 1},
    )
    assert [s.disaggregated_params.ctx_request_id for s in _ctx_handoff_slots(chat)] == [3]
    response = _responses_response()
    assert _ctx_handoff_slots(response) == [response]


@pytest.mark.parametrize(
    "finish_reason, need_gen", [("length", True), ("not_finished", True), ("stop", False)]
)
def test_a_finished_context_response_goes_to_the_client_without_the_handoff(
    finish_reason, need_gen
):
    response = _responses_response(finish_reason=finish_reason)
    assert _service()._need_gen(response) is need_gen
    handoff = (
        response.disaggregated_params,
        response.prompt_token_ids,
        response.prompt_token_ids_b64,
        response.finish_reason,
    )
    if need_gen:
        assert all(field is not None for field in handoff)
    else:
        assert handoff == (None, None, None, None)


_HANDOFF_KEYS = {
    "disaggregated_params",
    "prompt_token_ids",
    "prompt_token_ids_b64",
    "finish_reason",
}


def test_a_response_without_the_handoff_serializes_without_its_keys():
    """Bodies, stream events and stored copies carry no null handoff keys."""
    response = _responses_response(finish_reason="stop")
    _service()._need_gen(response)  # a finished context response drops the handoff

    assert not _HANDOFF_KEYS & response.model_dump(by_alias=True).keys()
    assert not _HANDOFF_KEYS & json.loads(response.model_dump_json(by_alias=True)).keys()
    # The stream wraps a dump of the response, as get_initial_responses does.
    created = ResponseCreatedEvent(
        type="response.created", sequence_number=0, response=response.model_dump(by_alias=True)
    )
    assert not _HANDOFF_KEYS & json.loads(created.model_dump_json(by_alias=True))["response"].keys()

    async def frames():
        return [frame async for frame in responses_done_generator(response)]

    for frame in asyncio.run(frames()):
        event = json.loads(frame.decode().split("data: ", 1)[1])
        assert not _HANDOFF_KEYS & event["response"].keys()


def test_a_context_response_still_carries_its_handoff():
    body = _responses_response(finish_reason="length").model_dump(by_alias=True)
    assert _HANDOFF_KEYS <= body.keys()
    assert ResponsesResponse(**body).disaggregated_params.ctx_request_id == 7


@pytest.mark.parametrize(
    "finish_reason, status",
    [("not_finished", "incomplete"), ("length", "incomplete"), ("stop", "completed")],
)
def test_finish_reason_mapping(finish_reason, status):
    assert finish_reason_mapping(finish_reason) == status


def test_an_unknown_finish_reason_names_itself():
    with pytest.raises(RuntimeError, match="wat"):
        finish_reason_mapping("wat")


@pytest.mark.parametrize(
    "fields, expected",
    [
        ({"prompt_token_ids": [1, 2, 3]}, [1, 2, 3]),
        (
            {
                "prompt_token_ids_b64": base64.b64encode(
                    np.asarray([5, 6, 7], dtype=np.int32).tobytes()
                ).decode("ascii")
            },
            [5, 6, 7],
        ),
        ({}, None),
    ],
)
def test_the_generation_worker_uses_the_relayed_prompt(fields, expected):
    request = ResponsesRequest(
        model="m",
        input="hi",
        disaggregated_params=DisaggregatedParams(request_type="generation_only"),
        **fields,
    )
    assert request.relayed_prompt_token_ids() == expected
    assert request.prompt_token_ids == fields.get("prompt_token_ids")


@pytest.mark.parametrize("request_type", [None, "context_only", "context_and_generation"])
@pytest.mark.parametrize("field", ["prompt_token_ids", "prompt_token_ids_b64"])
def test_prompt_ids_outside_a_generation_handoff_are_ignored(request_type, field):
    """A request that is not a generation-only handoff renders its own prompt."""
    value = [1, 2, 3]
    if field == "prompt_token_ids_b64":
        value = base64.b64encode(np.asarray(value, dtype=np.int32).tobytes()).decode("ascii")
    params = None if request_type is None else DisaggregatedParams(request_type=request_type)
    request = ResponsesRequest(model="m", input="hi", disaggregated_params=params, **{field: value})
    assert request.relayed_prompt_token_ids() is None


@pytest.mark.parametrize(
    "encoded, reason",
    [
        ("abc", "not valid base64"),
        ("AQ==!", "not valid base64"),
        (base64.b64encode(b"\x01\x02\x03").decode("ascii"), "int32"),
    ],
)
def test_a_malformed_relayed_prompt_is_a_validation_error(encoded, reason):
    with pytest.raises(ValidationError, match=reason):
        ResponsesRequest(model="m", input="hi", prompt_token_ids_b64=encoded)


@pytest.mark.parametrize(
    "usage, expected",
    [
        (
            SimpleNamespace(
                input_tokens=96,
                input_tokens_details=SimpleNamespace(cached_tokens=64),
                output_tokens=1,
                total_tokens=97,
            ),
            (96, 1, 64),
        ),
        (
            SimpleNamespace(
                input_tokens=96, input_tokens_details=None, output_tokens=1, total_tokens=97
            ),
            (96, 1, 0),
        ),
    ],
)
def test_the_context_usage_is_handed_off_as_usage_info(usage, expected):
    carried = _ctx_usage_info(SimpleNamespace(usage=usage))
    assert (
        carried.prompt_tokens,
        carried.completion_tokens,
        carried.prompt_tokens_details.cached_tokens,
    ) == expected
    already = UsageInfo(
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=0),
    )
    assert _ctx_usage_info(SimpleNamespace(usage=already)) is already


@pytest.mark.parametrize("use_harmony", [False, True])
def test_a_context_only_response_carries_the_handoff(use_harmony):
    """The single handoff token is not a complete Harmony message to parse."""
    request = ResponsesRequest(
        model="m", input="hi", disaggregated_params=DisaggregatedParams(request_type="context_only")
    )
    output = SimpleNamespace(
        index=0, text="", token_ids=[7], finish_reason="length", disaggregated_params=None
    )
    result = SimpleNamespace(outputs=[output], prompt_token_ids=[1, 2], cached_tokens=0)
    with patch(
        "tensorrt_llm.serve.responses_utils._create_output_content_harmony",
        side_effect=AssertionError("parsed a context-only output"),
    ):
        response = create_response_non_store(
            generation_result=result,
            request=request,
            sampling_params=request.to_sampling_params(),
            model_name="m",
            use_harmony=use_harmony,
        )
    assert response.output == []
    assert (response.finish_reason, response.prompt_token_ids) == ("length", [1, 2])


@pytest.mark.parametrize("as_b64", [False, True])
def test_a_context_worker_with_postprocessing_workers_returns_the_prompt(monkeypatch, as_b64):
    """A postprocessing worker's result has no prompt ids; the handoff still does."""
    import tensorrt_llm.serve.openai_server as server_module
    from tensorrt_llm.serve.openai_server import OpenAIServer

    request = ResponsesRequest(
        model="m",
        input="hi",
        store=False,
        disaggregated_params=DisaggregatedParams(
            request_type="context_only", return_prompt_token_ids_b64=as_b64
        ),
    )
    output = SimpleNamespace(
        index=0, text="", token_ids=[7], finish_reason="length", disaggregated_params=None
    )
    # What a postprocessing worker builds: its result carries no prompt_token_ids.
    postprocessed = create_response_non_store(
        generation_result=SimpleNamespace(outputs=[output], cached_tokens=0),
        request=request,
        sampling_params=request.to_sampling_params(),
        model_name="m",
        use_harmony=False,
        num_prompt_tokens=3,
    )
    assert postprocessed.prompt_token_ids is None

    async def nothing(*args, **kwargs):
        return None

    promise = SimpleNamespace(
        outputs=[SimpleNamespace(_postprocess_result=postprocessed)],
        prompt_token_ids=[5, 6, 7],
        aresult=nothing,
    )
    server = object.__new__(OpenAIServer)
    server.model = "m"
    server.enable_store = False
    server._is_visual_gen = False
    server.use_harmony = False
    server.tokenizer = None
    server.model_config = None
    server.processor = None
    server.multimodal_server_config = None
    server.chat_template = None
    server.tool_parser = None
    server.metrics_collector = None
    server.conversation_store = None
    server.generator = SimpleNamespace(
        args=SimpleNamespace(reasoning_parser=None, num_postprocess_workers=1),
        generate_async=lambda **kwargs: promise,
    )
    server._validate_internal_disagg_request = lambda *args: None
    server.await_disconnected = nothing
    server._extract_metrics = nothing

    async def preprocess(**kwargs):
        return [5, 6, 7], kwargs["request"].to_sampling_params()

    monkeypatch.setattr(server_module, "responses_api_request_preprocess", preprocess)
    raw_request = SimpleNamespace(
        state=SimpleNamespace(), headers={}, url=SimpleNamespace(path="/v1/responses")
    )

    body = json.loads(asyncio.run(server.openai_responses(request, raw_request)).body)

    if as_b64:
        ids = np.frombuffer(base64.b64decode(body["prompt_token_ids_b64"]), dtype=np.int32)
        assert (ids.tolist(), "prompt_token_ids" in body) == ([5, 6, 7], False)
    else:
        assert (body["prompt_token_ids"], "prompt_token_ids_b64" in body) == ([5, 6, 7], False)


# ---------------------------------------------------------------------------
# Generation-first scheduling
# ---------------------------------------------------------------------------


class _RecordingClient:
    def __init__(self, response):
        self.requests = []
        self._response = response

    async def send_request(self, request, server=None, hooks=None, req_id=None):
        self.requests.append(request.model_copy(deep=True))
        return self._response


def _gen_first_service(ctx_response, gen_response):
    async def disagg_request_id():
        return 11

    async def ctx_server(request, req_id=None):
        return "ctx0", None

    service = _service(
        _schedule_style=DisaggScheduleStyle.GENERATION_FIRST,
        _coordinator=SimpleNamespace(get_disagg_request_id=disagg_request_id),
        _ctx_router=SimpleNamespace(get_next_server=ctx_server),
        _ctx_client=_RecordingClient(ctx_response),
        _gen_client=_RecordingClient(gen_response),
        _tokids_ctxbytes=False,
        _subagent_affinity_scope="both",
    )
    service._send_disagg_request = service._send_disagg_request_gen_first
    return service


def _assert_generation_first_handoff(service):
    (ctx_request,) = service._ctx_client.requests
    (gen_request,) = service._gen_client.requests
    assert (ctx_request.disaggregated_params.request_type, ctx_request.stream) == (
        "context_only",
        False,
    )
    params = gen_request.disaggregated_params
    assert (params.request_type, params.schedule_style, params.disagg_request_id) == (
        "generation_only",
        DisaggScheduleStyle.GENERATION_FIRST,
        11,
    )
    # Nothing is relayed: the generation worker renders the prompt itself.
    assert gen_request.relayed_prompt_token_ids() is None
    assert gen_request.input == ctx_request.input


def test_a_generation_first_response_comes_from_the_generation_worker():
    final = _responses_response(finish_reason="stop", status="completed")
    service = _gen_first_service(ctx_response=_responses_response(), gen_response=final)

    response = asyncio.run(service.openai_responses(ResponsesRequest(model="m", input="hi")))

    assert response is final
    _assert_generation_first_handoff(service)


def test_a_generation_first_stream_relays_the_generation_frames_in_order():
    frames = [
        b"event: response.created\ndata: {}\n\n",
        b"event: response.output_text.delta\ndata: {}\n\n",
        b"event: response.completed\ndata: {}\n\n",
    ]

    async def gen_frames():
        for frame in frames:
            yield frame

    service = _gen_first_service(ctx_response=_responses_response(), gen_response=gen_frames())

    async def relay():
        stream = await service.openai_responses(
            ResponsesRequest(model="m", input="hi", stream=True)
        )
        return [frame async for frame in stream]

    assert asyncio.run(relay()) == frames
    _assert_generation_first_handoff(service)


# ---------------------------------------------------------------------------
# Streaming a request the context phase finished
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "finish_reason, status, terminal, details",
    [
        ("stop", "completed", "response.completed", None),
        ("length", "incomplete", "response.incomplete", {"reason": "max_output_tokens"}),
    ],
)
def test_a_context_finished_stream_replays_the_response(finish_reason, status, terminal, details):
    response = _responses_response(finish_reason=finish_reason, status=status)
    if details:
        response.incomplete_details = details

    async def collect():
        return [chunk.decode() async for chunk in responses_done_generator(response)]

    frames = asyncio.run(collect())
    assert all(f.startswith("event: ") and f.endswith("\n\n") for f in frames)
    events = [json.loads(f.split("data: ", 1)[1]) for f in frames]
    assert [e["type"] for e in events] == ["response.created", "response.in_progress", terminal]
    assert [e["sequence_number"] for e in events] == [0, 1, 2]
    assert events[-1]["response"]["incomplete_details"] == details


# ---------------------------------------------------------------------------
# What the orchestrator does with a Responses request
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("store, warned", [(None, False), (False, False), (True, True)])
def test_store_is_forwarded_off_and_an_explicit_request_for_it_is_logged(store, warned):
    sent = []

    async def send(request, hooks):
        sent.append(request)
        return "relayed"

    request = ResponsesRequest(model="m", input="hi", **({} if store is None else {"store": store}))
    with patch("tensorrt_llm.serve.openai_disagg_service.logger") as mock_logger:
        assert asyncio.run(_service(_send_disagg_request=send).openai_responses(request)) == (
            "relayed"
        )
    assert sent[0].store is False
    assert mock_logger.warning_once.called is warned


@pytest.mark.parametrize(
    "fields",
    [
        {"prompt_token_ids": [9, 9, 9]},
        {"prompt_token_ids_b64": "CQAAAA=="},
        {"prompt_token_ids": [9, 9, 9], "prompt_token_ids_b64": "CQAAAA=="},
    ],
)
def test_client_prompt_ids_are_dropped_before_dispatch(fields):
    """Only the context response supplies the prompt the generation worker uses."""
    sent = []

    async def send(request, hooks):
        sent.append(request)
        return "relayed"

    request = ResponsesRequest(model="m", input="hi", **fields)
    asyncio.run(_service(_send_disagg_request=send).openai_responses(request))
    assert (sent[0].prompt_token_ids, sent[0].prompt_token_ids_b64) == (None, None)


@pytest.mark.parametrize(
    "fields", [{}, {"request_id": "resp_set_by_the_client"}], ids=["generated", "client_set"]
)
def test_every_worker_is_sent_the_request_id(fields):
    """A worker names its response after request_id, so it goes out even when generated here.

    A stream this server opens itself names that id, and the workers must answer under it.
    """
    request = ResponsesRequest(model="m", input="hi", **fields)
    # The context request is a copy of the request, as the service makes it.
    for forwarded in (request, request.model_copy(update={"stream": False})):
        body = msgspec.msgpack.decode(_encode_request_body(forwarded))
        assert body["request_id"] == request.request_id


def test_a_generation_first_worker_ignores_client_prompt_ids():
    final = _responses_response(finish_reason="stop", status="completed")
    service = _gen_first_service(ctx_response=_responses_response(), gen_response=final)

    request = ResponsesRequest(model="m", input="hi", prompt_token_ids=[9, 9, 9])
    asyncio.run(service.openai_responses(request))

    _assert_generation_first_handoff(service)


def test_previous_response_id_is_rejected_with_a_400():
    request = ResponsesRequest(model="m", input="hi", previous_response_id="resp_1")
    with pytest.raises(HTTPException) as raised:
        asyncio.run(_service().openai_responses(request))
    assert raised.value.status_code == 400


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------


class _StubCtxClient:
    def __init__(self, error=None):
        self.error = error
        self.servers = []

    async def get_json(self, endpoint, response_type, server):
        assert endpoint == "v1/models"
        self.servers.append(server)
        if self.error is not None:
            raise self.error
        return ModelList(data=[ModelCard(id="served-name")])


def _models_service(client, servers=("w0:8001", "w1:8001")):
    return _service(
        _ctx_client=client,
        _ctx_router=SimpleNamespace(servers=list(servers)),
        _count_tokens_rr_counter=0,
    )


def test_models_are_listed_by_a_context_worker_round_robin():
    client = _StubCtxClient()
    service = _models_service(client)
    assert asyncio.run(service.get_model()).data[0].id == "served-name"
    asyncio.run(service.get_model())
    assert client.servers == ["w0:8001", "w1:8001"]


@pytest.mark.parametrize(
    "client, servers, status",
    [
        (_StubCtxClient(), (), 503),
        (
            _StubCtxClient(
                aiohttp.ClientResponseError(
                    SimpleNamespace(real_url="http://w0:8001/v1/models"),
                    (),
                    status=404,
                    message="Not Found",
                )
            ),
            ("w0:8001",),
            404,
        ),
    ],
    ids=["no_context_worker", "worker_error"],
)
def test_a_models_failure_is_an_error_response(client, servers, status):
    server = OpenAIDisaggServer.__new__(OpenAIDisaggServer)
    server._service = _models_service(client, servers)
    response = asyncio.run(server.get_model())
    assert response.status_code == status
    assert json.loads(response.body)["code"] == status


# ---------------------------------------------------------------------------
# Streamed requests: a slow context phase must not leave the client silent
# ---------------------------------------------------------------------------

_MODEL = "test-model"
_SCHEDULE_STYLES = ["context_first", "generation_first"]
_OVER_LENGTH = context_length_exceeded_message(8, 9)


def _as_received(request: ResponsesRequest) -> ResponsesRequest:
    """``request`` as a worker parses it from the body the service forwards."""
    return ResponsesRequest(**msgspec.msgpack.decode(_encode_request_body(request)))


def _response(request: ResponsesRequest, status: str) -> ResponsesResponse:
    """A worker's response to ``request``, without output."""
    return ResponsesResponse.from_request(
        request=request,
        sampling_params=request.to_sampling_params(),
        model_name=_MODEL,
        created_time=0,
        output=[],
        status=status,
    )


def _sse(event) -> str:
    return f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"


def _generation_frames(request: ResponsesRequest) -> list[str]:
    """A generation worker's stream for ``request``: its opening, "hello", the completion.

    Its response objects are created at 0, so they differ from any built elsewhere.
    """
    opening = ResponsesStreamingProcessor(
        request, request.to_sampling_params(), model_name=_MODEL, create_time=0
    ).get_initial_responses()
    delta = ResponseTextDeltaEvent(
        type="response.output_text.delta",
        sequence_number=2,
        item_id="msg_1",
        output_index=0,
        content_index=0,
        delta="hello",
        logprobs=[],
    )
    completed = ResponseCompletedEvent(
        type="response.completed",
        sequence_number=3,
        response=_response(request, "completed").model_dump(by_alias=True),
    )
    return opening + [_sse(delta), _sse(completed)]


def _worker_rejection(status: int, message: str) -> aiohttp.ClientResponseError:
    """A worker's error response, as the HTTP client raises it."""
    body = json.dumps({"object": "error", "message": message, "code": status})
    return aiohttp.ClientResponseError(
        Mock(real_url="http://ctx:8001/v1/responses"),
        (),
        status=status,
        message=f"Error: {body}",
    )


class _Workers:
    """A context and a generation worker behind the disaggregated service, without a network.

    Each parses its request from the body the service forwards. The context phase runs
    ``context_phase`` and then answers with the KV-cache handoff. The generation worker
    answers once the context phase has succeeded, as it does once it has the KV cache, and
    streams its answer in pieces cut anywhere, as the transport delivers it.
    """

    def __init__(self, context_phase):
        self.context_phase = context_phase
        self.context_done = asyncio.Event()
        self.context_cancelled = asyncio.Event()
        self.generation_cancelled = asyncio.Event()
        # Set once the generation worker's stream has ended, however it ended.
        self.generation_closed = asyncio.Event()
        self.generation_requests = []
        # What the generation worker answered: a response, or the frames of a stream.
        self.response = None
        self.frames = None

    async def send_context_request(self, request, server=None, hooks=None, req_id=None):
        received = _as_received(request)
        try:
            await self.context_phase()
        except asyncio.CancelledError:
            self.context_cancelled.set()
            raise
        self.context_done.set()
        response = _response(received, "incomplete")
        response.finish_reason = "length"
        response.prompt_token_ids = [1, 2, 3]
        disagg_request_id = received.disaggregated_params.disagg_request_id
        response.disaggregated_params = DisaggregatedParams(
            request_type="context_only",
            disagg_request_id=disagg_request_id,
            ctx_request_id=disagg_request_id,
        )
        return response

    async def send_generation_request(self, request, server=None, hooks=None, req_id=None):
        received = _as_received(request)
        self.generation_requests.append(received)
        if not received.stream:
            await self.context_done.wait()
            self.response = _response(received, "completed")
            return self.response

        async def stream():
            try:
                yield None
                await self.context_done.wait()
                async for piece in self.generation_stream(received):
                    yield piece
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

    async def generation_stream(self, received: ResponsesRequest):
        """The generation worker's stream, in pieces cut anywhere, as the transport delivers it."""
        self.frames = _generation_frames(received)
        data = "".join(self.frames).encode()
        for start in range(0, len(data), 61):
            yield data[start : start + 61]


class _WorkersSilentAfterTheFirstOutput(_Workers):
    """Workers whose generation worker is silent for a while after its first output.

    The silence outlasts the keepalive interval, so the worker's own keepalive sends its
    response.in_progress again, with its own response object, before the rest of its stream.
    """

    async def generation_stream(self, received: ResponsesRequest):
        frames = _generation_frames(received)

        async def produced():
            # The opening and the first output, then the rest after a silence.
            for frame in frames[:3]:
                yield frame
            await asyncio.sleep(0.3)
            for frame in frames[3:]:
                yield frame

        self.frames = []
        async for frame in responses_utils.keep_responses_stream_alive(produced()):
            self.frames.append(frame)
            yield frame.encode()


class _WorkersFailingAfterTheFirstOutput(_Workers):
    """Workers whose generation worker's connection fails once its first output is relayed."""

    def __init__(self, context_phase):
        super().__init__(context_phase)
        self.output_relayed = asyncio.Event()

    async def generation_stream(self, received: ResponsesRequest):
        self.frames = _generation_frames(received)[:3]
        for frame in self.frames:
            yield frame.encode()
        await self.output_relayed.wait()
        raise aiohttp.ClientPayloadError("Response payload is not completed")

    def on_send(self, message):
        """An ``on_send`` that notes the first output reaching the client."""
        if message.get("body", b"").startswith(b"event: response.output_text.delta\n"):
            self.output_relayed.set()


class _WorkersStillGenerating(_Workers):
    """Workers whose generation worker sends its opening and then goes on generating."""

    async def generation_stream(self, received: ResponsesRequest):
        self.frames = _generation_frames(received)[:2]
        for frame in self.frames:
            yield frame.encode()
        await asyncio.Event().wait()


def _responses_server(schedule_style: str, workers: _Workers) -> OpenAIDisaggServer:
    """A disaggregated server in front of ``workers``, with its routes."""
    config = DisaggServerConfig(server_configs=[], schedule_style=schedule_style)
    coordinator = DisaggCoordinatorService(config, client_factory=Mock())
    service = OpenAIDisaggregatedService(config, coordinator, client_factory=Mock())
    service.is_ready = AsyncMock(return_value=True)
    service._ctx_router = AsyncMock(spec=Router)
    service._ctx_router.get_next_server.return_value = ("ctx:8001", {})
    service._gen_router = AsyncMock(spec=Router)
    service._gen_router.get_next_server.return_value = ("gen:8002", {})
    service._ctx_client = SimpleNamespace(send_request=workers.send_context_request)
    service._gen_client = SimpleNamespace(send_request=workers.send_generation_request)

    server = OpenAIDisaggServer.__new__(OpenAIDisaggServer)
    server._config = config
    server._coordinator = coordinator
    server._disagg_cluster_storage = None
    server._service = service
    server._perf_metrics_collector = Mock()
    server._allow_request_chat_template = False
    server._collect_perf_metrics = False
    server.app = FastAPI()
    server.app.add_middleware(ServerArrivalTimeMiddleware)
    with patch("prometheus_client.multiprocess.MultiProcessCollector"):
        server.register_routes()
    return server


def _responses_route(server: OpenAIDisaggServer):
    """The handler ``server`` routes /v1/responses to."""
    (route,) = [
        route for route in server.app.routes if getattr(route, "path", "") == "/v1/responses"
    ]
    return route.endpoint


async def _post_responses(app: FastAPI, *, stream=True, on_send=None, leave=None, **fields):
    """Send one /v1/responses request to ``app`` as a client and return the messages sent back.

    ``on_send`` sees each message as it is sent. The client disconnects once
    ``leave`` is set.
    """
    body = {"model": _MODEL, "input": "hi", "stream": stream, **fields}
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
        "path": "/v1/responses",
        "raw_path": b"/v1/responses",
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


def _without_number(data: dict) -> dict:
    return {key: value for key, value in data.items() if key != "sequence_number"}


def _short_waits(monkeypatch):
    monkeypatch.setattr(openai_disagg_server, "_RESPONSES_STREAM_START_WAIT_S", 0.05)
    monkeypatch.setattr(responses_utils, "RESPONSES_STREAM_KEEPALIVE_INTERVAL_S", 0.05)


def _waits_for(event: asyncio.Event):
    """A context phase that lasts until ``event`` is set, or five seconds without it."""

    async def context_phase():
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(event.wait(), 5)

    return context_phase


def _on_keepalive(event: asyncio.Event):
    """An ``on_send`` that sets ``event`` once response.in_progress has gone out again."""
    in_progress = 0

    def on_send(message):
        nonlocal in_progress
        if message.get("body", b"").startswith(b"event: response.in_progress\n"):
            in_progress += 1
            if in_progress == 2:
                event.set()

    return on_send


@pytest.mark.asyncio
@pytest.mark.parametrize("schedule_style", _SCHEDULE_STYLES)
async def test_responses_stream_opens_and_keeps_alive_while_the_context_phase_runs(
    monkeypatch, schedule_style
):
    _short_waits(monkeypatch)
    kept_alive = asyncio.Event()
    workers = _Workers(_waits_for(kept_alive))
    keepalive = _on_keepalive(kept_alive)
    started_during_context_phase = []

    def on_send(message):
        if message["type"] == "http.response.start":
            started_during_context_phase.append(not workers.context_done.is_set())
        keepalive(message)

    sent = await _post_responses(_responses_server(schedule_style, workers).app, on_send=on_send)

    assert _status(sent) == 200
    assert started_during_context_phase == [True]
    events = _sse_events(sent)
    types = [event for event, _ in events]
    first_output = types.index("response.output_text.delta")
    # The opening, kept alive while the context phase ran, then the generation
    # worker's events without its own opening.
    assert types[:2] == ["response.created", "response.in_progress"]
    assert first_output > 2
    assert set(types[2:first_output]) == {"response.in_progress"}
    relayed = [data for event, data in events[first_output:] if event != "response.in_progress"]
    answered = [json.loads(frame.split("data: ", 1)[1]) for frame in workers.frames[2:]]
    assert [_without_number(data) for data in relayed] == [
        _without_number(data) for data in answered
    ]
    # One sequence, for one response: the one the generation worker was asked for.
    assert [data["sequence_number"] for _, data in events] == list(range(len(events)))
    (generation_request,) = workers.generation_requests
    assert {data["response"]["id"] for _, data in events if "response" in data} == {
        generation_request.request_id
    }


def _frame_events(frames: list[str]) -> list[tuple[str, dict]]:
    """The (event, data) pairs of SSE frames."""
    return [
        (frame.split("\n", 1)[0].removeprefix("event: "), json.loads(frame.split("data: ", 1)[1]))
        for frame in frames
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("schedule_style", _SCHEDULE_STYLES)
async def test_every_keepalive_of_an_opened_stream_repeats_its_opening(monkeypatch, schedule_style):
    """The worker's own keepalives, which carry its response object, never reach the client."""
    _short_waits(monkeypatch)
    kept_alive = asyncio.Event()
    workers = _WorkersSilentAfterTheFirstOutput(_waits_for(kept_alive))

    sent = await _post_responses(
        _responses_server(schedule_style, workers).app, on_send=_on_keepalive(kept_alive)
    )

    assert _status(sent) == 200
    events = _sse_events(sent)
    types = [event for event, _ in events]
    assert types[:2] == ["response.created", "response.in_progress"]
    first_output = types.index("response.output_text.delta")
    # The worker kept its own stream alive while it was silent after its first output.
    answered = _frame_events(workers.frames)
    answered_types = [event for event, _ in answered]
    assert (
        "response.in_progress"
        in answered_types[answered_types.index("response.output_text.delta") :]
    )
    # The client was kept alive meanwhile, and every response.in_progress it got is
    # the opening's, only numbered on; none is one of the worker's.
    assert "response.in_progress" in types[first_output:]
    opening = _without_number(events[1][1])
    assert all(
        _without_number(data) == opening
        for event, data in events
        if event == "response.in_progress"
    )
    assert opening not in [
        _without_number(data) for event, data in answered if event == "response.in_progress"
    ]
    # The worker's other events follow, in one sequence, for one response.
    relayed = [data for event, data in events[first_output:] if event != "response.in_progress"]
    assert [_without_number(data) for data in relayed] == [
        _without_number(data)
        for event, data in answered
        if event not in ("response.created", "response.in_progress")
    ]
    assert [data["sequence_number"] for _, data in events] == list(range(len(events)))
    (generation_request,) = workers.generation_requests
    assert {data["response"]["id"] for _, data in events if "response" in data} == {
        generation_request.request_id
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("schedule_style", _SCHEDULE_STYLES)
async def test_responses_stream_after_a_fast_context_phase_is_the_generation_stream(
    schedule_style,
):
    async def context_phase():
        pass

    workers = _Workers(context_phase)
    sent = await _post_responses(_responses_server(schedule_style, workers).app)

    assert _status(sent) == 200
    assert [event for event, _ in _sse_events(sent)] == [
        "response.created",
        "response.in_progress",
        "response.output_text.delta",
        "response.completed",
    ]
    assert _body(sent) == "".join(workers.frames)


_CONTEXT_FAILURES = [
    pytest.param(400, _OVER_LENGTH, id="over_length_400"),
    pytest.param(503, "no capacity left", id="503"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("schedule_style", _SCHEDULE_STYLES)
@pytest.mark.parametrize("status, worker_message", _CONTEXT_FAILURES)
async def test_responses_stream_with_a_fast_context_failure_is_an_http_error(
    schedule_style, status, worker_message
):
    async def context_phase():
        raise _worker_rejection(status, worker_message)

    streamed = await _post_responses(_responses_server(schedule_style, _Workers(context_phase)).app)
    not_streamed = await _post_responses(
        _responses_server(schedule_style, _Workers(context_phase)).app, stream=False
    )

    assert _status(streamed) == status
    assert worker_message in _body(streamed)
    # The error response of the same failure without a stream.
    assert _status(not_streamed) == status
    assert json.loads(_body(streamed)) == json.loads(_body(not_streamed))


@pytest.mark.asyncio
@pytest.mark.parametrize("schedule_style", _SCHEDULE_STYLES)
@pytest.mark.parametrize(
    "status, worker_message, error_code, failed_code",
    [
        # An over-length rejection keeps its code in both events: clients act on it,
        # as an agent compacts its history and retries.
        pytest.param(
            400,
            _OVER_LENGTH,
            "context_length_exceeded",
            "context_length_exceeded",
            id="over_length_400",
        ),
        pytest.param(503, "no capacity left", "upstream_error", "server_error", id="503"),
    ],
)
async def test_responses_stream_with_a_late_context_failure_ends_with_response_failed(
    monkeypatch, schedule_style, status, worker_message, error_code, failed_code
):
    _short_waits(monkeypatch)
    kept_alive = asyncio.Event()
    wait = _waits_for(kept_alive)

    async def context_phase():
        await wait()
        raise _worker_rejection(status, worker_message)

    streamed = await _post_responses(
        _responses_server(schedule_style, _Workers(context_phase)).app,
        on_send=_on_keepalive(kept_alive),
    )
    kept_alive.set()
    not_streamed = await _post_responses(
        _responses_server(schedule_style, _Workers(context_phase)).app, stream=False
    )

    assert _status(streamed) == 200
    # The stream ends after its events, as any stream does.
    assert streamed[-1] == {"type": "http.response.body", "body": b"", "more_body": False}
    events = _sse_events(streamed)
    types = [event for event, _ in events]
    assert types[:2] == ["response.created", "response.in_progress"]
    assert set(types[2:-2]) == {"response.in_progress"}
    assert types[-2:] == ["error", "response.failed"]
    assert [data["sequence_number"] for _, data in events] == list(range(len(events)))
    # error and response.failed, as a stream reports any failure, with the
    # message of the error response of the same failure without a stream.
    error_response = json.loads(_body(not_streamed))
    message = error_response.get("detail") or error_response["message"]
    assert worker_message in message
    assert (events[-2][1]["code"], events[-2][1]["message"]) == (error_code, message)
    failed = events[-1][1]["response"]
    assert failed["id"] == events[0][1]["response"]["id"]
    assert failed["status"] == "failed"
    assert (failed["error"]["code"], failed["error"]["message"]) == (failed_code, message)


@pytest.mark.asyncio
@pytest.mark.parametrize("schedule_style", _SCHEDULE_STYLES)
async def test_an_opened_stream_whose_generation_stream_fails_ends_with_response_failed(
    monkeypatch, schedule_style
):
    _short_waits(monkeypatch)

    async def context_phase():
        # Outlasts the start wait.
        await asyncio.sleep(0.2)

    workers = _WorkersFailingAfterTheFirstOutput(context_phase)
    sent = await _post_responses(
        _responses_server(schedule_style, workers).app, on_send=workers.on_send
    )

    assert _status(sent) == 200
    # The stream ends after its events, as any stream does.
    assert sent[-1] == {"type": "http.response.body", "body": b"", "more_body": False}
    events = _sse_events(sent)
    assert events[1][0] == "response.in_progress"
    assert [event for event, _ in events if event != "response.in_progress"] == [
        "response.created",
        "response.output_text.delta",
        "error",
        "response.failed",
    ]
    assert [data["sequence_number"] for _, data in events] == list(range(len(events)))
    assert (events[-2][1]["code"], events[-2][1]["message"]) == (
        "upstream_error",
        "ClientPayloadError",
    )
    # The failed response is the one the stream opened with.
    opened, failed = events[0][1]["response"], events[-1][1]["response"]
    assert (failed["id"], failed["created_at"]) == (opened["id"], opened["created_at"])
    assert (failed["status"], failed["error"]["code"]) == ("failed", "server_error")


@pytest.mark.asyncio
@pytest.mark.parametrize("schedule_style", _SCHEDULE_STYLES)
async def test_a_relayed_generation_stream_that_fails_ends_with_an_error_event(schedule_style):
    """After a fast context phase the worker's own opening is relayed, so only error is added."""

    async def context_phase():
        pass

    workers = _WorkersFailingAfterTheFirstOutput(context_phase)
    sent = []

    def on_send(message):
        sent.append(message)
        workers.on_send(message)

    with contextlib.suppress(aiohttp.ClientPayloadError):
        await _post_responses(_responses_server(schedule_style, workers).app, on_send=on_send)

    events = _sse_events(sent)
    assert [event for event, _ in events] == [
        "response.created",
        "response.in_progress",
        "response.output_text.delta",
        "error",
    ]
    assert [data["sequence_number"] for _, data in events] == [0, 1, 2, 3]
    assert (events[-1][1]["code"], events[-1][1]["message"]) == (
        "upstream_error",
        "ClientPayloadError",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("schedule_style", _SCHEDULE_STYLES)
async def test_a_non_streamed_response_still_waits_for_the_whole_pipeline(
    monkeypatch, schedule_style
):
    _short_waits(monkeypatch)

    async def context_phase():
        # Outlasts the start wait.
        await asyncio.sleep(0.2)

    workers = _Workers(context_phase)
    sent = await _post_responses(_responses_server(schedule_style, workers).app, stream=False)

    assert _status(sent) == 200
    assert json.loads(_body(sent)) == workers.response.model_dump(mode="json", by_alias=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("schedule_style", _SCHEDULE_STYLES)
async def test_a_request_no_opening_can_be_built_for_is_answered_by_its_pipeline(
    monkeypatch, schedule_style
):
    """The opening is built from the request; one it cannot be built from waits as before."""
    _short_waits(monkeypatch)
    rejection = "require 0 <= top_p <= 1, got top_p=2.0"

    async def context_phase():
        # Outlasts the start wait.
        await asyncio.sleep(0.2)
        raise _worker_rejection(400, rejection)

    sent = await _post_responses(
        _responses_server(schedule_style, _Workers(context_phase)).app, top_p=2.0
    )

    assert _status(sent) == 400
    assert json.loads(_body(sent)) == {"detail": rejection}


@pytest.mark.asyncio
@pytest.mark.parametrize("schedule_style", _SCHEDULE_STYLES)
async def test_client_leaving_after_the_responses_stream_starts_cancels_the_context_phase(
    monkeypatch, schedule_style
):
    _short_waits(monkeypatch)
    workers = _Workers(_waits_for(asyncio.Event()))
    left = asyncio.Event()

    def on_send(message):
        if message["type"] == "http.response.start":
            left.set()

    await _post_responses(
        _responses_server(schedule_style, workers).app, on_send=on_send, leave=left
    )

    await asyncio.wait_for(workers.context_cancelled.wait(), 2)


def _raw_request() -> Request:
    raw_request = Request({"type": "http", "headers": []})
    raw_request.state.server_arrival_time = 0
    return raw_request


@pytest.mark.asyncio
@pytest.mark.parametrize("schedule_style", _SCHEDULE_STYLES)
async def test_cancelling_the_responses_handler_during_its_wait_cancels_the_context_phase(
    monkeypatch, schedule_style
):
    monkeypatch.setattr(openai_disagg_server, "_RESPONSES_STREAM_START_WAIT_S", 60)
    context_started = asyncio.Event()
    release_context = asyncio.Event()

    async def context_phase():
        context_started.set()
        await release_context.wait()

    workers = _Workers(context_phase)
    handler = _responses_route(_responses_server(schedule_style, workers))
    request = ResponsesRequest(model=_MODEL, input="hi", stream=True)
    pending = asyncio.create_task(handler(request, _raw_request()))
    try:
        await asyncio.wait_for(context_started.wait(), 2)
        assert not pending.done()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await asyncio.wait_for(workers.context_cancelled.wait(), 2)
        assert not workers.context_done.is_set()
    finally:
        release_context.set()
        pending.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pending


@pytest.mark.asyncio
@pytest.mark.parametrize("schedule_style", _SCHEDULE_STYLES)
@pytest.mark.parametrize("asgi_version", ["2.3", "2.4"])
async def test_disconnect_before_the_responses_body_starts_cancels_the_context_phase(
    monkeypatch, schedule_style, asgi_version
):
    """The pipeline is cancelled even if the body is never iterated."""
    from starlette.requests import ClientDisconnect

    _short_waits(monkeypatch)
    release_context = asyncio.Event()
    workers = _Workers(_waits_for(release_context))
    handler = _responses_route(_responses_server(schedule_style, workers))
    request = ResponsesRequest(model=_MODEL, input="hi", stream=True)
    response = await asyncio.wait_for(handler(request, _raw_request()), 2)
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
async def test_disconnect_before_the_responses_body_starts_closes_the_stream_it_answered_with(
    monkeypatch, schedule_style, asgi_version
):
    """The pipeline answers with a stream after the response is returned, which no body reads."""
    from starlette.requests import ClientDisconnect

    _short_waits(monkeypatch)
    release_context = asyncio.Event()
    workers = _WorkersStillGenerating(_waits_for(release_context))
    handler = _responses_route(_responses_server(schedule_style, workers))
    request = ResponsesRequest(model=_MODEL, input="hi", stream=True)
    response = await asyncio.wait_for(handler(request, _raw_request()), 2)
    pipeline = response._pipeline
    answered = asyncio.Event()
    sent = []

    async def send(message):
        sent.append(message)
        # The pipeline answers while the response starts, before the body does.
        release_context.set()
        await asyncio.wait((pipeline,), timeout=2)
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
    assert isinstance(pipeline.result(), StreamingResponse)
    # Closed by the time the response is done, while the generation worker goes on.
    assert workers.generation_closed.is_set()


def _frame(event_type: str, sequence_number: int, **fields) -> str:
    data = {"type": event_type, "sequence_number": sequence_number, **fields}
    return f"event: {event_type}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


async def _transport(data: str, piece: int):
    """``data`` as a relayed stream delivers it, in pieces of ``piece`` bytes."""
    encoded = data.encode()
    for start in range(0, len(encoded), piece):
        yield encoded[start : start + piece]


@pytest.mark.asyncio
@pytest.mark.parametrize("piece", [1, 7, 4096])
async def test_the_relay_drops_the_worker_opening_and_keepalives_and_numbers_the_rest(piece):
    worker = "".join(
        [
            _frame("response.created", 0),
            _frame("response.in_progress", 1),
            # Kept alive before its first output.
            _frame("response.in_progress", 2),
            _frame("response.output_text.delta", 3, delta="hi"),
            # Kept alive after it.
            _frame("response.in_progress", 4),
            _frame("response.completed", 5),
        ]
    )

    relayed = responses_utils.relay_after_opening(_transport(worker, piece), 2)

    assert [frame async for frame in relayed] == [
        _frame("response.output_text.delta", 2, delta="hi"),
        _frame("response.completed", 3),
    ]


@pytest.mark.asyncio
async def test_the_relay_numbers_a_failure_the_worker_reported_before_its_opening():
    worker = _frame("error", 0, code="upstream_error", message="ClientResponseError")

    relayed = responses_utils.relay_after_opening(_transport(worker, 4096), 3)

    assert [frame async for frame in relayed] == [
        _frame("error", 3, code="upstream_error", message="ClientResponseError")
    ]


@pytest.mark.asyncio
async def test_the_relay_passes_on_an_event_its_stream_stopped_inside():
    cut = 'event: response.output_text.delta\ndata: {"type":"response.out'
    worker = _frame("response.output_text.delta", 2, delta="hi") + cut

    relayed = responses_utils.relay_after_opening(_transport(worker, 7), 2)

    assert [frame async for frame in relayed] == [
        _frame("response.output_text.delta", 2, delta="hi"),
        cut,
    ]


@pytest.mark.asyncio
async def test_the_relay_closes_its_stream_when_it_stops_early():
    closed = asyncio.Event()

    async def worker():
        try:
            yield _frame("response.output_text.delta", 2, delta="hi").encode()
            yield _frame("response.completed", 3).encode()
        finally:
            closed.set()

    relayed = responses_utils.relay_after_opening(worker(), 2)
    await relayed.__anext__()
    await relayed.aclose()

    assert closed.is_set()
