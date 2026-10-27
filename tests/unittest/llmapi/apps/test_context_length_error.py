# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Unit tests for the over-length-prompt (context_length_exceeded) error.

An over-length prompt is rejected (never truncated) with HTTP 400. The
message uses OpenAI's wording so agent frameworks can pattern-match it to
trigger context compaction, and the error payload carries the
machine-readable ``code: "context_length_exceeded"``.

The message is raised in ``_deduce_max_tokens`` (executor/base_worker.py)
and crosses the executor IPC boundary as a plain string, so the server
recognizes it by text in ``OpenAIServer.create_error_response`` — the single
chokepoint through which the chat, completions, and responses endpoints all
render request errors. These tests cover the message builder, the detector,
that chokepoint, and the disaggregated router's forwarding of the worker's
error envelope; they are CPU-only.

A streamed Responses request is the exception: it is answered as OpenAI
answers it, with a stream that opens and ends in ``response.failed`` carrying
the code, because that event, not a 400, is what clients such as the Codex
CLI act on.
"""

import json
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import Mock

import aiohttp
import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

import tensorrt_llm.serve.openai_server as openai_server_module
from tensorrt_llm.executor.executor import GenerationExecutor
from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.executor.result import GenerationResult
from tensorrt_llm.executor.utils import (
    CONTEXT_LENGTH_EXCEEDED_CODE,
    ErrorResponse,
    RequestError,
    context_length_exceeded_message,
    is_context_length_exceeded_message,
)
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.serve.openai_disagg_server import OpenAIDisaggServer
from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest, ResponsesRequest
from tensorrt_llm.serve.openai_server import OpenAIServer

# The CPU stage collects with `-m cpu_only`; unittest/conftest.py also skips
# collecting any file that does not contain the cpu_only marker.
pytestmark = pytest.mark.cpu_only


def test_message_matches_openai_wording():
    msg = context_length_exceeded_message(max_context_length=69632, num_prompt_tokens=69666)
    assert msg == (
        "This model's maximum context length is 69632 tokens. "
        "However, your messages resulted in 69666 tokens. "
        "Please reduce the length of the messages."
    )


def test_detector_matches_exact_and_wrapped():
    msg = context_length_exceeded_message(2048, 4096)
    assert is_context_length_exceeded_message(msg)
    # RequestError text may arrive wrapped in extra context.
    assert is_context_length_exceeded_message(f"Request failed: {msg} (id=7)")


def test_detector_rejects_other_errors():
    assert not is_context_length_exceeded_message(
        "`default_max_tokens` (-2785) must be greater than 0, "
        "`default_max_tokens` (-2785) = max_seq_len (69632) "
        "- `splited_prompt_len` (72417)"
    )
    assert not is_context_length_exceeded_message("`max_tokens` (0) must be greater than 0")
    assert not is_context_length_exceeded_message("")


def _body(response):
    return json.loads(response.body)


def test_error_response_carries_machine_readable_code():
    msg = context_length_exceeded_message(2048, 4096)
    response = OpenAIServer.create_error_response(msg)
    assert response.status_code == HTTPStatus.BAD_REQUEST
    body = _body(response)
    assert body["code"] == CONTEXT_LENGTH_EXCEEDED_CODE
    assert body["type"] == "BadRequestError"
    assert body["message"] == msg
    assert "2048 tokens" in body["message"]
    assert "4096 tokens" in body["message"]


def test_error_response_default_code_unchanged():
    response = OpenAIServer.create_error_response("`max_tokens` (0) must be greater than 0")
    assert response.status_code == HTTPStatus.BAD_REQUEST
    body = _body(response)
    assert body["code"] == 400
    assert body["type"] == "BadRequestError"


def test_error_response_non_400_status_unchanged():
    response = OpenAIServer.create_error_response(
        "Response with id 'resp_x' not found.",
        err_type="InvalidRequestError",
        status_code=HTTPStatus.NOT_FOUND,
    )
    assert response.status_code == HTTPStatus.NOT_FOUND
    body = _body(response)
    assert body["code"] == 404
    assert body["type"] == "InvalidRequestError"


def _upstream_error(worker_body: str, status: int = 400) -> aiohttp.ClientResponseError:
    """A ClientResponseError the way openai_client builds one.

    post_json/_send_request raise with ``message=f"{reason}: {body[:2048]}"``,
    where ``body`` is the worker's HTTP response text.
    """
    return aiohttp.ClientResponseError(
        request_info=Mock(),
        history=(),
        status=status,
        message=f"Bad Request: {worker_body}",
    )


def _disagg_server() -> OpenAIDisaggServer:
    server = object.__new__(OpenAIDisaggServer)
    server._perf_metrics_collector = SimpleNamespace(
        http_exceptions=SimpleNamespace(inc=Mock()),
        internal_errors=SimpleNamespace(inc=Mock()),
    )
    return server


def test_disagg_route_preserves_machine_readable_code():
    """The disagg router must not drop the worker's context-length code.

    The router forwards a worker error as HTTPException(detail=<message>),
    which keeps HTTP 400 and the text but loses ``code``; a client that
    checks ``code == "context_length_exceeded"`` could then never detect the
    rejection through the disaggregated route. _handle_exception must re-emit
    the worker's envelope instead for this error.
    """
    msg = context_length_exceeded_message(2048, 4096)
    # create_error_response's body is the exact JSON a context worker sends.
    worker_body = OpenAIServer.create_error_response(msg).body.decode()

    response = _disagg_server()._handle_exception(_upstream_error(worker_body))

    assert response.status_code == HTTPStatus.BAD_REQUEST
    body = _body(response)
    assert body["code"] == CONTEXT_LENGTH_EXCEEDED_CODE
    assert body["type"] == "BadRequestError"
    assert body["message"] == msg


def test_disagg_route_other_upstream_errors_unchanged():
    """Non-context-length worker errors keep the {"detail": message} contract."""
    worker_body = json.dumps(
        {
            "message": "`max_tokens` (0) must be greater than 0",
            "type": "BadRequestError",
            "code": 400,
        }
    )
    with pytest.raises(HTTPException) as excinfo:
        _disagg_server()._handle_exception(_upstream_error(worker_body))
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "`max_tokens` (0) must be greater than 0"


# ---------------------------------------------------------------------------
# A streamed Responses request fails in-stream
# ---------------------------------------------------------------------------

_MSG = context_length_exceeded_message(max_context_length=8, num_prompt_tokens=9)

_RAW_REQUEST = SimpleNamespace(state=SimpleNamespace(no_client_connection=True), headers={})


class _Executor:
    """Holds the executor's per-request error handler; a result keeps only a weak reference."""

    _handle_background_error = GenerationExecutor._handle_background_error


class _RejectingEngine:
    """LLM stand-in whose executor rejects every prompt with ``error_msg``.

    ``in_worker``: the executor runs in worker processes, so the rejection
    arrives as the worker's error response and is raised while the stream
    iterates the result. Otherwise it runs in this process and
    ``generate_async`` itself raises.
    """

    def __init__(self, error_msg: str, in_worker: bool) -> None:
        self._error_msg = error_msg
        self._in_worker = in_worker
        self._executor = _Executor()
        self.args = SimpleNamespace(num_postprocess_workers=0, reasoning_parser=None)

    def generate_async(self, inputs, sampling_params, streaming, **_kwargs):
        if not self._in_worker:
            raise RequestError(self._error_msg)
        request = GenerationRequest(inputs, sampling_params, streaming=streaming)
        request.set_id(1)
        result = GenerationResult(
            request, background_error_handler=self._executor._handle_background_error
        )
        result.aqueue.put(ErrorResponse(request.id, self._error_msg, request.id))
        return RequestOutput._from_generation_result(result)


def _responses_server(engine: _RejectingEngine, monkeypatch) -> OpenAIServer:
    async def preprocess(**kwargs):
        return [1, 2, 3], kwargs["request"].to_sampling_params()

    monkeypatch.setattr(openai_server_module, "responses_api_request_preprocess", preprocess)
    server = object.__new__(OpenAIServer)
    server.generator = engine
    server.model = "served-model"
    server.enable_store = False
    server._is_visual_gen = False
    server.use_harmony = False
    server.tokenizer = None
    server.model_config = None
    server.processor = None
    server.multimodal_server_config = None
    server.chat_template = None
    server.tool_parser = None
    server.conversation_store = None
    server._validate_internal_disagg_request = lambda *args: None
    return server


def _event(frame):
    if isinstance(frame, bytes):
        frame = frame.decode()
    return json.loads(frame.split("data: ", 1)[1])


async def _events(response: StreamingResponse) -> list:
    # Iterated to the end: a body that raised would abort the connection.
    return [_event(frame) async for frame in response.body_iterator]


def _assert_failed_in_stream(events, request: ResponsesRequest) -> None:
    assert [event["type"] for event in events] == [
        "response.created",
        "response.in_progress",
        "error",
        "response.failed",
    ]
    assert [event["sequence_number"] for event in events] == [0, 1, 2, 3]
    assert {event["response"]["id"] for event in events if "response" in event} == {
        request.request_id
    }
    failed = events[-1]["response"]
    assert (failed["status"], failed["output"]) == ("failed", [])
    assert (failed["error"]["code"], failed["error"]["message"]) == (
        CONTEXT_LENGTH_EXCEEDED_CODE,
        _MSG,
    )


@pytest.mark.parametrize("in_worker", [True, False], ids=["in_worker", "in_process"])
@pytest.mark.asyncio
async def test_a_streamed_responses_request_fails_in_stream(in_worker, monkeypatch):
    server = _responses_server(_RejectingEngine(_MSG, in_worker), monkeypatch)
    request = ResponsesRequest(model="client-model", input="hi", stream=True)

    response = await server.openai_responses(request, _RAW_REQUEST)

    assert isinstance(response, StreamingResponse), getattr(response, "body", response)
    _assert_failed_in_stream(await _events(response), request)


@pytest.mark.parametrize("in_worker", [True, False], ids=["in_worker", "in_process"])
@pytest.mark.asyncio
async def test_a_non_streamed_responses_request_keeps_the_400(in_worker, monkeypatch):
    server = _responses_server(_RejectingEngine(_MSG, in_worker), monkeypatch)
    request = ResponsesRequest(model="client-model", input="hi", stream=False)

    response = await server.openai_responses(request, _RAW_REQUEST)

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert (_body(response)["code"], _body(response)["message"]) == (
        CONTEXT_LENGTH_EXCEEDED_CODE,
        _MSG,
    )


@pytest.mark.parametrize("in_worker", [True, False], ids=["in_worker", "in_process"])
@pytest.mark.asyncio
async def test_other_rejections_of_a_streamed_request_are_unchanged(in_worker, monkeypatch):
    server = _responses_server(_RejectingEngine("Insufficient KV cache", in_worker), monkeypatch)
    request = ResponsesRequest(model="client-model", input="hi", stream=True)

    response = await server.openai_responses(request, _RAW_REQUEST)

    if not in_worker:
        assert response.status_code == HTTPStatus.BAD_REQUEST
        assert _body(response)["code"] == 400
        return
    frames = []
    with pytest.raises(RequestError):
        async for frame in response.body_iterator:
            frames.append(frame)
    assert _event(frames[-1])["response"]["error"]["code"] == "server_error"


def _context_worker_rejection() -> aiohttp.ClientResponseError:
    return _upstream_error(OpenAIServer.create_error_response(_MSG).body.decode())


@pytest.mark.asyncio
async def test_disagg_route_fails_a_streamed_responses_request_in_stream():
    """The context phase never streams, so its worker answers with the 400."""
    request = ResponsesRequest(model="client-model", input="hi", stream=True)

    response = _disagg_server()._handle_exception(_context_worker_rejection(), request)

    assert isinstance(response, StreamingResponse)
    events = await _events(response)
    _assert_failed_in_stream(events, request)
    assert events[-1]["response"]["model"] == "client-model"


@pytest.mark.parametrize(
    "request_",
    [
        ResponsesRequest(model="m", input="hi", stream=False),
        ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}], stream=True),
    ],
    ids=["responses_non_streamed", "chat_streamed"],
)
def test_disagg_route_keeps_the_400_for_other_requests(request_):
    response = _disagg_server()._handle_exception(_context_worker_rejection(), request_)

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert _body(response)["code"] == CONTEXT_LENGTH_EXCEEDED_CODE


@pytest.mark.asyncio
async def test_disagg_route_hands_the_request_to_its_failure_path():
    async def rejected(req, hooks):
        raise _context_worker_rejection()

    server = _disagg_server()
    server._perf_metrics_collector = Mock()
    server._allow_request_chat_template = False
    server._config = SimpleNamespace(conversation_affinity_header_for_subagents=None)
    server._collect_perf_metrics = False
    route = server._wrap_entry_point(rejected, ResponsesRequest)
    request = ResponsesRequest(model="client-model", input="hi", stream=True)
    raw_request = SimpleNamespace(state=SimpleNamespace(server_arrival_time=0.0), headers={})

    response = await route(request, raw_request)

    assert isinstance(response, StreamingResponse)
    _assert_failed_in_stream(await _events(response), request)
