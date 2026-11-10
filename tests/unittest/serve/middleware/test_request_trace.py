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
"""The request trace records each request and the exact reply the client received.

The middleware is driven two ways: through a bare FastAPI app with the real
``register_middlewares`` stack and Starlette's TestClient, for what a client sees,
and at the ASGI level with hand-written ``receive``/``send`` callables, for the
endings a TestClient cannot produce (a send that fails, a disconnect mid-stream, a
cancelled task).
"""

import asyncio
import hashlib
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import msgspec
import pytest
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.testclient import TestClient

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.middleware.arrival_time import ServerArrivalTimeMiddleware
from tensorrt_llm.serve.middleware.perf_metrics import PerfMetricsMiddleware
from tensorrt_llm.serve.middleware.register import register_middlewares
from tensorrt_llm.serve.middleware.request_trace import (
    REQUEST_TRACE_DIR_ENV,
    REQUEST_TRACE_IDENTITY_ENV,
    RequestTrace,
    RequestTraceMiddleware,
    RequestTraceWriter,
    classify_ending,
    load_identity_plugin,
    redact_headers,
    sanitize_session_key,
    summarize_reply,
)
from tensorrt_llm.serve.perf_metrics import RETURN_METRICS_HEADER, SSE_METRICS_EVENT

pytestmark = pytest.mark.cpu_only

CHAT = "/v1/chat/completions"


# -- helpers ------------------------------------------------------------------------


def make_trace(tmp_path, identity=None):
    return RequestTrace(RequestTraceWriter(str(tmp_path), shard_suffix="-w"), identity=identity)


def lines(tmp_path, kind):
    """The ``kind`` ("requests" or "responses") lines of every hour bucket, in order."""
    records = []
    for path in sorted(Path(tmp_path).glob(f"*/{kind}-w.jsonl")):
        records.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    return records


def traced_app(trace, *, perf_metrics=False):
    """A bare FastAPI app with the real built-in middleware stack and the trace."""

    @asynccontextmanager
    async def lifespan(app):
        await trace.start()
        yield
        await trace.close()

    app = FastAPI(lifespan=lifespan)
    register_middlewares(
        app,
        collect_perf_metrics=perf_metrics,
        expose_perf_metrics=perf_metrics,
        perf_metrics_writer=None,
        request_trace=trace,
    )
    return app


def chat_chunk(content=None, finish_reason=None):
    chunk = {
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk)}\n\n".encode()


def sse(frames):
    async def body():
        for frame in frames:
            yield frame

    return StreamingResponse(body(), media_type="text/event-stream")


def http_scope(path=CHAT, method="POST", headers=()):
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(name.encode(), value.encode()) for name, value in headers],
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
    }


def queued_receive(*messages):
    """A ``receive`` that hands out the queued messages and then blocks, as a server does."""
    queue = list(messages)

    async def receive():
        if queue:
            return queue.pop(0)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    return receive


def body_message(body, more_body=False):
    return {"type": "http.request", "body": body, "more_body": more_body}


DISCONNECT = {"type": "http.disconnect"}


async def wait_for_disconnect(receive):
    """What Starlette's disconnect listener does: read until ``http.disconnect``."""
    while True:
        if (await receive())["type"] == "http.disconnect":
            return


async def run_asgi(trace, app, scope, receive, send):
    """Drive ``RequestTraceMiddleware`` around ``app`` for one request, writer included."""
    await trace.start()
    try:
        await RequestTraceMiddleware(app, trace)(scope, receive, send)
    finally:
        await trace.close()


def collecting_send(fail_at=None, error=None):
    """A ``send`` that records messages and raises ``error`` on the ``fail_at``-th one."""
    sent = []

    async def send(message):
        if fail_at is not None and len(sent) == fail_at:
            raise error
        sent.append(message)

    return send, sent


def start_message(status=200, content_type=b"text/event-stream"):
    return {
        "type": "http.response.start",
        "status": status,
        "headers": [(b"content-type", content_type)],
    }


def chunk_message(body, more_body=True):
    return {"type": "http.response.body", "body": body, "more_body": more_body}


@pytest.fixture()
def logged():
    """Warnings the TRT-LLM logger emits during the test, as messages."""
    messages = []

    class Capture(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    handler = Capture(level=logging.WARNING)
    stdlib_logger = logger._logger
    previous_level = stdlib_logger.level
    previous_min_severity = logger._min_severity
    stdlib_logger.addHandler(handler)
    stdlib_logger.setLevel(logging.WARNING)
    logger._min_severity = "warning"
    try:
        yield messages
    finally:
        logger._min_severity = previous_min_severity
        stdlib_logger.setLevel(previous_level)
        stdlib_logger.removeHandler(handler)


# -- what a client sees, through the real stack --------------------------------------


def test_a_json_reply_is_recorded_with_its_request(tmp_path):
    trace = make_trace(tmp_path)
    app = traced_app(trace)
    reply = {
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }

    @app.post(CHAT)
    async def chat(request: Request):
        return reply

    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5, "stream": False}
    raw = json.dumps(body).encode()
    with TestClient(app) as client:
        response = client.post(
            CHAT,
            content=raw,
            headers=[
                ("content-type", "application/json"),
                ("authorization", "Bearer secret"),
                ("x-forwarded-for", "10.0.0.1"),
                ("x-forwarded-for", "10.0.0.2"),
            ],
        )
    assert response.status_code == 200

    (request_line,) = lines(tmp_path, "requests")
    (response_line,) = lines(tmp_path, "responses")
    assert request_line["event"] == "request" and response_line["event"] == "response"
    assert request_line["trace_id"].startswith("tr_")
    assert response_line["trace_id"] == request_line["trace_id"], "the two lines join"
    assert request_line["session"] == response_line["session"] == "_no_session"
    assert request_line["method"] == "POST" and request_line["route"] == CHAT
    assert request_line["body"] == body
    assert request_line["body_sha256"] == hashlib.sha256(raw).hexdigest()
    assert request_line["sampling"] == {"max_tokens": 5, "stream": False}
    assert isinstance(request_line["server_arrival_time"], float), (
        "the arrival stamp of ServerArrivalTimeMiddleware is on the line"
    )
    assert "body_parse_error" not in request_line and "body_complete" not in request_line
    assert ["authorization", "[redacted]"] in request_line["headers"]
    assert ["content-type", "application/json"] in request_line["headers"]
    forwarded = [value for name, value in request_line["headers"] if name == "x-forwarded-for"]
    assert forwarded == ["10.0.0.1", "10.0.0.2"], "repeated headers are kept in order"

    assert response_line["status"] == "completed"
    assert response_line["http_status"] == 200
    assert ["content-type", "application/json"] in response_line["headers"]
    assert response_line["response"] == {"kind": "json", "body": reply}
    assert response_line["finish_reasons"] == ["stop"]
    assert response_line["usage"] == reply["usage"]
    assert "error" not in response_line and "disconnected" not in response_line


def test_repeated_headers_are_all_kept_and_each_credential_is_redacted():
    pairs = redact_headers(
        [
            (b"x-forwarded-for", b"10.0.0.1"),
            (b"x-forwarded-for", b"10.0.0.2"),
            (b"Authorization", b"Bearer a"),
            (b"x-api-key", b"k"),
            (b"X-Auth-Token", b"t"),
            (b"cookie", b"session=1"),
            (b"x-client-secret", b"s"),
            (b"content-type", b"application/json"),
        ]
    )
    assert pairs == [
        ["x-forwarded-for", "10.0.0.1"],
        ["x-forwarded-for", "10.0.0.2"],
        ["Authorization", "[redacted]"],
        ["x-api-key", "[redacted]"],
        ["X-Auth-Token", "[redacted]"],
        ["cookie", "[redacted]"],
        ["x-client-secret", "[redacted]"],
        ["content-type", "application/json"],
    ]


@pytest.mark.parametrize(
    "name",
    [
        "authorization",
        "proxy-authorization",
        "www-authenticate",
        "x-forwarded-authorization",  # Google ESP, Apigee and Envoy forward the bearer here
        "x-original-authorization",
        "x-auth",
        "x-auth-request-access-token",
        "x-goog-iap-jwt-assertion",  # Google IAP's signed identity
        "x-jwt-assertion",
        "x-amz-credential",
        "x-amz-security-token",
        "x-hub-signature-256",
        "x-client-cert",
        "x-ssl-client-cert",
        "x-forwarded-client-cert",
        "cookie",
        "set-cookie",
        "api-key",
        "x-password",
    ],
)
def test_credential_header_spellings_are_redacted(name):
    assert redact_headers([(name.encode(), b"value")]) == [[name, "[redacted]"]]
    assert redact_headers([(name.upper().encode(), b"value")]) == [[name.upper(), "[redacted]"]]


@pytest.mark.parametrize(
    "name",
    [
        "content-type",
        "accept",
        "user-agent",
        "x-request-id",
        "x-session-id",
        "x-forwarded-for",
        "x-trtllm-msgpack",
        "x-trtllm-return-metrics",
        "traceparent",
    ],
)
def test_ordinary_headers_keep_their_values(name):
    assert redact_headers([(name.encode(), b"value")]) == [[name, "value"]]


def test_an_sse_reply_is_recorded_byte_for_byte_with_the_metrics_event(tmp_path):
    """The trace sits outside PerfMetricsMiddleware: what it records is what the client got."""
    trace = make_trace(tmp_path)
    app = traced_app(trace, perf_metrics=True)
    perf_record = {
        "schema_version": 1,
        "request_id": "42",
        "status": "complete",
        "phases": {
            "server": {
                "timing_metrics": {
                    "arrival_time": 1.0,
                    "first_scheduled_time": 1.01,
                    "first_token_time": 1.02,
                    "last_token_time": 1.05,
                    "kv_cache_transfer_start": None,
                    "kv_cache_transfer_end": None,
                },
                "time_breakdown_metrics": {"step_metrics": [], "ctx_chunk_metrics": []},
            }
        },
    }
    frames = [chat_chunk("he"), chat_chunk("llo"), chat_chunk(finish_reason="length")]

    @app.post(CHAT)
    async def chat(request: Request):
        request.state.perf_metrics_records.append(perf_record)
        return sse(frames + [b"data: [DONE]\n\n"])

    with TestClient(app) as client:
        response = client.post(CHAT, json={"stream": True}, headers={RETURN_METRICS_HEADER: "1"})
    assert response.status_code == 200
    assert f"event: {SSE_METRICS_EVENT}" in response.text, (
        "PerfMetricsMiddleware appended its event"
    )

    (response_line,) = lines(tmp_path, "responses")
    assert response_line["response"]["kind"] == "sse_text"
    assert response_line["response"]["body"] == response.text, (
        "the recorded stream must be exactly the bytes the client received, the metrics "
        "event PerfMetricsMiddleware appended included"
    )
    assert response_line["status"] == "completed"
    assert response_line["finish_reasons"] == ["length"]
    assert "usage" not in response_line, "a chat stream without include_usage states none"


def test_a_multibyte_character_split_across_chunks_survives(tmp_path):
    trace = make_trace(tmp_path)
    app = traced_app(trace)
    text = "data: 你好\n\n".encode()

    @app.post(CHAT)
    async def chat(request: Request):
        return sse([text[:7], text[7:]])  # the split falls inside the first character

    with TestClient(app) as client:
        client.post(CHAT, json={})
    (response_line,) = lines(tmp_path, "responses")
    assert response_line["response"]["body"] == "data: 你好\n\n"


def test_a_responses_stream_is_summarized_from_its_terminal_event(tmp_path):
    trace = make_trace(tmp_path)
    app = traced_app(trace)
    terminal = {
        "type": "response.completed",
        "response": {
            "id": "resp_1",
            "object": "response",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "max_output_tokens": 64,
            "usage": {"input_tokens": 10, "output_tokens": 64, "total_tokens": 74},
            "output": [
                {"type": "reasoning"},
                {"type": "function_call", "call_id": "call_1", "name": "f", "arguments": "{}"},
                {"type": "message", "content": []},
            ],
        },
    }
    frames = [
        b'event: response.created\ndata: {"type":"response.created","response":{"id":"resp_1"}}\n\n',
        b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"x"}\n\n',
        f"event: response.completed\ndata: {json.dumps(terminal)}\n\n".encode(),
    ]

    @app.post("/v1/responses")
    async def responses(request: Request):
        return sse(frames)

    with TestClient(app) as client:
        client.post("/v1/responses", json={"input": "x", "stream": True})
    (response_line,) = lines(tmp_path, "responses")
    assert response_line["response_status"] == "incomplete"
    assert response_line["incomplete_reason"] == "max_output_tokens"
    assert response_line["response_id"] == "resp_1"
    assert response_line["usage"] == terminal["response"]["usage"]
    assert response_line["max_output_tokens"] == 64
    assert response_line["output_item_types"] == ["reasoning", "function_call", "message"]
    assert response_line["output_call_ids"] == ["call_1"]
    assert "finish_reasons" not in response_line
    assert "error_code" not in response_line


def test_the_handler_still_reads_the_body_and_the_request_line_precedes_it(tmp_path):
    trace = make_trace(tmp_path)
    app = traced_app(trace)
    submitted_when_the_handler_ran = []

    @app.post(CHAT)
    async def chat(request: Request):
        submitted_when_the_handler_ran.append(trace.writer.submitted)
        return {"echo": await request.json()}

    with TestClient(app) as client:
        response = client.post(CHAT, json={"messages": [1, 2, 3]})
    assert response.json() == {"echo": {"messages": [1, 2, 3]}}
    assert submitted_when_the_handler_ran == [1], (
        "the request line is handed to the writer before the handler starts, so a request "
        "the process does not survive is still on disk"
    )


def test_a_validation_rejection_is_recorded_with_the_body_and_the_error_reply(tmp_path):
    trace = make_trace(tmp_path)
    app = traced_app(trace)

    class ChatRequest(BaseModel):
        messages: list

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc):
        return JSONResponse(status_code=400, content={"error": str(exc)})

    @app.post(CHAT)
    async def chat(request: ChatRequest):
        return {}

    with TestClient(app) as client:
        response = client.post(CHAT, json={"messages": "not a list"})
    assert response.status_code == 400

    (request_line,) = lines(tmp_path, "requests")
    (response_line,) = lines(tmp_path, "responses")
    assert request_line["body"] == {"messages": "not a list"}, "the rejected body is kept"
    assert response_line["status"] == "rejected_400"
    assert response_line["http_status"] == 400
    assert response_line["response"]["kind"] == "json"
    assert response_line["response"]["body"] == response.json()


def test_a_body_that_is_not_json_is_kept_as_text_with_the_parse_error(tmp_path):
    trace = make_trace(tmp_path)
    app = traced_app(trace)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc):
        return JSONResponse(status_code=400, content={"error": "bad json"})

    @app.post(CHAT)
    async def chat(body: dict):
        return {}

    with TestClient(app) as client:
        response = client.post(
            CHAT, content=b'{"messages": [', headers={"content-type": "application/json"}
        )
    assert response.status_code == 400

    (request_line,) = lines(tmp_path, "requests")
    assert request_line["body"] == '{"messages": ['
    assert request_line["body_parse_error"].startswith("JSONDecodeError: ")
    assert request_line["body_sha256"] == hashlib.sha256(b'{"messages": [').hexdigest()
    (response_line,) = lines(tmp_path, "responses")
    assert response_line["status"] == "rejected_400"


def test_a_msgpack_body_is_decoded_like_the_server_decodes_it(tmp_path):
    trace = make_trace(tmp_path)
    app = traced_app(trace)

    @app.post(CHAT)
    async def chat(request: Request):
        return {}

    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3}
    with TestClient(app) as client:
        client.post(
            CHAT,
            content=msgspec.msgpack.encode(body),
            headers={"content-type": "application/json", "x-trtllm-msgpack": "1"},
        )
    (request_line,) = lines(tmp_path, "requests")
    assert request_line["body"] == body
    assert "body_parse_error" not in request_line


@pytest.mark.parametrize("request_type", ["context_only", "generation_only"])
def test_an_internal_disaggregated_hop_leaves_no_trace(tmp_path, request_type):
    trace = make_trace(tmp_path)
    app = traced_app(trace)

    @app.post(CHAT)
    async def chat(request: Request):
        return {}

    with TestClient(app) as client:
        client.post(CHAT, json={"disaggregated_params": {"request_type": request_type}})
        client.post(CHAT, json={"disaggregated_params": {"request_type": "context_and_generation"}})
    assert len(lines(tmp_path, "requests")) == 1, "only the client-facing request is recorded"
    assert len(lines(tmp_path, "responses")) == 1
    assert lines(tmp_path, "requests")[0]["body"]["disaggregated_params"] == {
        "request_type": "context_and_generation"
    }


def test_requests_without_a_body_leave_no_trace(tmp_path):
    trace = make_trace(tmp_path)
    app = traced_app(trace)

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.get("/v1/models")
    async def models():
        return {"data": []}

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/models").status_code == 200
        # A GET that carries a body is still a GET: the method decides, before any read.
        assert client.request("GET", "/v1/models", content=b'{"a":1}').status_code == 200
    assert lines(tmp_path, "requests") == [] and lines(tmp_path, "responses") == []


def test_a_5xx_reply_is_an_error_and_a_404_a_rejection(tmp_path):
    trace = make_trace(tmp_path)
    app = traced_app(trace)

    @app.post("/boom")
    async def boom():
        return JSONResponse(status_code=503, content={"error": "overloaded"})

    with TestClient(app) as client:
        client.post("/boom", json={})
        client.post("/missing", json={})
    statuses = {line["http_status"]: line["status"] for line in lines(tmp_path, "responses")}
    assert statuses == {503: "error", 404: "rejected_404"}


def test_an_exception_out_of_the_handler_is_recorded_as_an_error(tmp_path):
    trace = make_trace(tmp_path)
    app = traced_app(trace)

    @app.post(CHAT)
    async def chat(request: Request):
        raise RuntimeError("handler bug")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(CHAT, json={"messages": []})
    assert response.status_code == 500

    (request_line,) = lines(tmp_path, "requests")
    (response_line,) = lines(tmp_path, "responses")
    assert request_line["body"] == {"messages": []}, (
        "the request line is written before the handler"
    )
    assert response_line["status"] == "error"
    assert response_line["error"] == "RuntimeError: handler bug"
    # The framework's 500 is sent outside the built-in middleware: nothing was observed.
    assert response_line["http_status"] is None and "response" not in response_line


# -- sessions -----------------------------------------------------------------------


def test_the_session_comes_from_the_body_first_then_the_conversation_headers(tmp_path):
    trace = make_trace(tmp_path)
    app = traced_app(trace)

    @app.post(CHAT)
    async def chat(request: Request):
        return {}

    with TestClient(app) as client:
        client.post(CHAT, json={"conversation_params": {"conversation_id": "body/1"}})
        client.post(CHAT, json={}, headers={"x-session-id": "hdr 2"})
        client.post(
            CHAT,
            json={"conversation_params": {"conversation_id": "body-3"}},
            headers={"x-session-id": "hdr-3"},
        )
        client.post(CHAT, json={})
    assert [line["session"] for line in lines(tmp_path, "requests")] == [
        "body_1",
        "hdr_2",
        "body-3",
        "_no_session",
    ]
    assert [line["session"] for line in lines(tmp_path, "responses")] == [
        "body_1",
        "hdr_2",
        "body-3",
        "_no_session",
    ]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "_no_session"),
        ("", "_no_session"),
        ("   ", "_no_session"),
        ("..", "_no_session"),
        (".hidden", "_no_session"),
        ("a/b\\c d\x00e", "a_b_c_d_e"),
        ("x" * 200, "x" * 128),
        ("conv-1.A_b", "conv-1.A_b"),
    ],
)
def test_session_keys_are_sanitized_and_bounded(value, expected):
    assert sanitize_session_key(value) == expected


# -- endings only the ASGI level can produce -----------------------------------------


@pytest.mark.asyncio
async def test_a_send_that_fails_mid_stream_is_a_client_disconnect_with_the_sent_frames(
    tmp_path,
):
    trace = make_trace(tmp_path)
    frames = [chat_chunk("a"), chat_chunk("b"), chat_chunk("c")]

    async def app(scope, receive, send):
        await send(start_message())
        for frame in frames:
            await send(chunk_message(frame))
        await send(chunk_message(b"", more_body=False))

    # A server whose send raises once the client is gone (uvicorn's WebSocket protocols
    # do; its HTTP protocols return silently and the disconnect shows on the receive side).
    send, sent = collecting_send(fail_at=3, error=OSError("client gone"))
    with pytest.raises(OSError):
        await run_asgi(
            trace, app, http_scope(), queued_receive(body_message(b'{"stream":true}')), send
        )

    (response_line,) = lines(tmp_path, "responses")
    assert response_line["status"] == "client_disconnected"
    assert response_line["http_status"] == 200
    assert response_line["response"]["body"] == (frames[0] + frames[1]).decode(), (
        "only the frames the client received are recorded"
    )
    assert response_line["error"] == "OSError: client gone"


@pytest.mark.asyncio
async def test_a_disconnect_seen_by_the_app_mid_stream_is_a_client_disconnect(tmp_path):
    """Starlette's disconnect listener reads http.disconnect through the wrapped receive."""
    trace = make_trace(tmp_path)

    async def app(scope, receive, send):
        await send(start_message())
        await send(chunk_message(chat_chunk("a")))
        await wait_for_disconnect(receive)  # the listener sees it and stops the stream

    send, sent = collecting_send()
    await run_asgi(
        trace,
        app,
        http_scope(),
        queued_receive(body_message(b"{}"), DISCONNECT),
        send,
    )
    (response_line,) = lines(tmp_path, "responses")
    assert response_line["status"] == "client_disconnected"
    assert response_line["response"]["body"] == chat_chunk("a").decode()
    assert "error" not in response_line


@pytest.mark.asyncio
async def test_a_reply_that_never_starts_is_not_started(tmp_path):
    trace = make_trace(tmp_path)

    async def app(scope, receive, send):
        await wait_for_disconnect(receive)  # the client went away before the reply began

    send, sent = collecting_send()
    await run_asgi(
        trace, app, http_scope(), queued_receive(body_message(b'{"a":1}'), DISCONNECT), send
    )
    (request_line,) = lines(tmp_path, "requests")
    (response_line,) = lines(tmp_path, "responses")
    assert request_line["body"] == {"a": 1}
    assert response_line["status"] == "not_started"
    assert response_line["disconnected"] is True
    assert response_line["http_status"] is None
    assert "response" not in response_line and "headers" not in response_line


@pytest.mark.asyncio
async def test_a_cancelled_request_is_cancelled_and_the_exception_propagates(tmp_path):
    trace = make_trace(tmp_path)

    async def app(scope, receive, send):
        await receive()
        raise asyncio.CancelledError()

    send, sent = collecting_send()
    with pytest.raises(asyncio.CancelledError):
        await run_asgi(trace, app, http_scope(), queued_receive(body_message(b"{}")), send)
    (response_line,) = lines(tmp_path, "responses")
    assert response_line["status"] == "cancelled"


@pytest.mark.asyncio
async def test_a_handler_that_never_reads_the_body_still_gets_the_body_recorded(tmp_path):
    """The middleware reads the body itself, so the request line never depends on the handler."""
    trace = make_trace(tmp_path)

    async def app(scope, receive, send):
        await send(start_message(content_type=b"application/json"))
        await send(chunk_message(b"{}", more_body=False))

    send, sent = collecting_send()
    await run_asgi(
        trace,
        app,
        http_scope(),
        queued_receive(body_message(b'{"a":', more_body=True), body_message(b"1}")),
        send,
    )
    (request_line,) = lines(tmp_path, "requests")
    (response_line,) = lines(tmp_path, "responses")
    assert request_line["body"] == {"a": 1}, "both body messages were read and joined"
    assert "body_complete" not in request_line
    assert response_line["status"] == "completed"


@pytest.mark.asyncio
async def test_the_body_is_replayed_to_the_app_as_it_was_received(tmp_path):
    trace = make_trace(tmp_path)
    seen = []

    async def app(scope, receive, send):
        while True:
            message = await receive()
            seen.append(message)
            if not message.get("more_body", False):
                break
        await send(start_message(content_type=b"application/json"))
        await send(chunk_message(b"{}", more_body=False))

    first, second = body_message(b'{"a":', more_body=True), body_message(b"1}")
    send, sent = collecting_send()
    await run_asgi(trace, app, http_scope(), queued_receive(first, second), send)
    assert seen == [first, second], "the app gets the same messages in the same order"


@pytest.mark.asyncio
async def test_a_body_cut_by_a_disconnect_is_recorded_incomplete(tmp_path):
    trace = make_trace(tmp_path)

    async def app(scope, receive, send):
        await wait_for_disconnect(receive)

    send, sent = collecting_send()
    await run_asgi(
        trace,
        app,
        http_scope(),
        queued_receive(body_message(b'{"a":', more_body=True), DISCONNECT),
        send,
    )
    (request_line,) = lines(tmp_path, "requests")
    (response_line,) = lines(tmp_path, "responses")
    assert request_line["body"] == '{"a":' and request_line["body_complete"] is False
    assert request_line["body_parse_error"].startswith("JSONDecodeError")
    assert response_line["status"] == "not_started"


@pytest.mark.asyncio
async def test_is_disconnected_still_fires_through_the_wrapped_receive(tmp_path):
    """Buffering the body must not break Request.is_disconnected().

    A middleware that replaces ``receive`` with one that awaits anything of its own, or
    that swallows ``http.disconnect``, makes ``is_disconnected()`` return False forever.
    """
    trace = make_trace(tmp_path)
    seen = {}

    async def app(scope, receive, send):
        request = Request(scope, receive)
        seen["body"] = await request.body()
        seen["before"] = await request.is_disconnected()
        receive_queue.append(DISCONNECT)
        seen["after"] = await request.is_disconnected()
        await send(start_message(content_type=b"application/json"))
        await send(chunk_message(b"{}", more_body=False))

    receive_queue = [body_message(b'{"a":1}')]

    async def receive():
        if receive_queue:
            return receive_queue.pop(0)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    send, sent = collecting_send()
    await run_asgi(trace, app, http_scope(), receive, send)
    assert seen == {"body": b'{"a":1}', "before": False, "after": True}
    (response_line,) = lines(tmp_path, "responses")
    assert response_line["status"] == "completed", "a disconnect after completion changes nothing"


@pytest.mark.parametrize(
    ("observed", "expected"),
    [
        (dict(started=True, completed=True, http_status=200), "completed"),
        (dict(started=True, completed=True, http_status=302), "completed"),
        (dict(started=True, completed=True, http_status=400), "rejected_400"),
        (dict(started=True, completed=True, http_status=422), "rejected_422"),
        (dict(started=True, completed=True, http_status=500), "error"),
        (dict(started=True, completed=True, http_status=200, disconnected=True), "completed"),
        (dict(started=False, completed=False, http_status=None), "not_started"),
        (dict(started=False, completed=False, http_status=None, disconnected=True), "not_started"),
        (dict(started=False, completed=False, http_status=None, failed=True), "error"),
        (
            dict(started=False, completed=False, http_status=None, failed=True, disconnected=True),
            "client_disconnected",
        ),
        (dict(started=False, completed=False, http_status=None, cancelled=True), "cancelled"),
        (
            dict(
                started=False, completed=False, http_status=None, cancelled=True, disconnected=True
            ),
            "not_started",
        ),
        (
            dict(started=True, completed=False, http_status=200, send_failed=True),
            "client_disconnected",
        ),
        (
            dict(started=True, completed=False, http_status=200, disconnected=True),
            "client_disconnected",
        ),
        (
            dict(started=True, completed=False, http_status=200, send_failed=True, failed=True),
            "client_disconnected",
        ),
        (dict(started=True, completed=False, http_status=200, cancelled=True), "cancelled"),
        (dict(started=True, completed=False, http_status=200, failed=True), "error"),
        (dict(started=True, completed=False, http_status=200), "error"),
    ],
)
def test_classify_ending(observed, expected):
    flags = dict(disconnected=False, send_failed=False, cancelled=False, failed=False)
    flags.update(observed)
    assert classify_ending(**flags) == expected


def test_a_non_streaming_completions_reply_keeps_its_finish_reasons_and_usage():
    reply = {
        "object": "text_completion",
        "choices": [{"index": 0, "text": "x", "finish_reason": "length"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 8, "total_tokens": 11},
    }
    assert summarize_reply({"kind": "json", "body": reply}, "/v1/completions") == {
        "finish_reasons": ["length"],
        "usage": reply["usage"],
    }


def test_the_terminal_event_is_found_behind_trailing_frames():
    """Only the last frame is the terminal event in the common case; a few may follow it."""
    terminal = {"type": "response.completed", "response": {"id": "resp_1", "status": "completed"}}
    stream = "event: response.created\ndata: {}\n\n" + (
        f"event: response.completed\ndata: {json.dumps(terminal)}\n\n"
    )
    trailing = 'event: trtllm.perf_metrics\ndata: {"Server-Timing": "x"}\n\n'
    trailing += 'event: error\ndata: {"type": "error", "message": "relay closed"}\n\n'
    trailing += "data: [DONE]\n\n"
    found = summarize_reply({"kind": "sse_text", "body": stream + trailing}, "/v1/responses")
    assert found["response_id"] == "resp_1" and found["response_status"] == "completed"
    too_far = stream + "data: {}\n\n" * 8
    assert summarize_reply({"kind": "sse_text", "body": too_far}, "/v1/responses") == {}, (
        "the search stops after the last eight frames"
    )


def test_summaries_add_nothing_for_other_replies():
    assert summarize_reply(None, CHAT) == {}
    assert summarize_reply({"kind": "text", "body": "Internal Server Error"}, CHAT) == {}
    assert summarize_reply({"kind": "json", "body": {"error": "bad"}}, CHAT) == {}
    cut = "event: response.created\ndata: {}\n\nevent: response.output_text.delta\ndata: {}\n\n"
    assert summarize_reply({"kind": "sse_text", "body": cut}, "/v1/responses") == {}, (
        "a Responses stream cut before its terminal event states nothing"
    )
    chat_usage_stream = chat_chunk(finish_reason="stop").decode() + (
        'data: {"object":"chat.completion.chunk","choices":[],'
        '"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}\n\n'
    )
    assert summarize_reply({"kind": "sse_text", "body": chat_usage_stream}, CHAT) == {
        "finish_reasons": ["stop"],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }


# -- identity plugin ----------------------------------------------------------------


def identity_fields(headers, body, route):
    return {"deployment": headers.get("x-deployment"), "route_seen": route}


def identity_with_reserved_keys(headers, body, route):
    return {"session": "mine", "trace_id": "mine", "tenant": "t1"}


def failing_identity(headers, body, route):
    raise RuntimeError("plugin bug")


def unserializable_identity(headers, body, route):
    return {"tenant": object()}


def test_the_identity_plugin_adds_fields_to_the_request_line(tmp_path):
    trace = make_trace(tmp_path, identity=load_identity_plugin(f"{__name__}:identity_fields"))
    app = traced_app(trace)

    @app.post(CHAT)
    async def chat(request: Request):
        return {}

    with TestClient(app) as client:
        client.post(CHAT, json={}, headers={"x-deployment": "canary"})
    (request_line,) = lines(tmp_path, "requests")
    assert request_line["deployment"] == "canary" and request_line["route_seen"] == CHAT
    (response_line,) = lines(tmp_path, "responses")
    assert "deployment" not in response_line, "identity fields are request-line fields"


def test_reserved_keys_from_the_plugin_are_refused(tmp_path, logged):
    trace = make_trace(
        tmp_path, identity=load_identity_plugin(f"{__name__}:identity_with_reserved_keys")
    )
    app = traced_app(trace)

    @app.post(CHAT)
    async def chat(request: Request):
        return {}

    with TestClient(app) as client:
        client.post(CHAT, json={}, headers={"x-session-id": "real"})
    (request_line,) = lines(tmp_path, "requests")
    assert request_line["session"] == "real" and request_line["trace_id"].startswith("tr_")
    assert request_line["tenant"] == "t1"
    assert [m for m in logged if "returned 2 reserved key(s), refused" in m]


@pytest.mark.parametrize("plugin", ["failing_identity", "unserializable_identity"])
def test_a_failing_plugin_costs_its_fields_not_the_line(tmp_path, logged, plugin):
    trace = make_trace(tmp_path, identity=load_identity_plugin(f"{__name__}:{plugin}"))
    app = traced_app(trace)

    @app.post(CHAT)
    async def chat(request: Request):
        return {}

    with TestClient(app) as client:
        client.post(CHAT, json={"a": 1})
        client.post(CHAT, json={"a": 2})
    assert [line["body"] for line in lines(tmp_path, "requests")] == [{"a": 1}, {"a": 2}]
    assert all("tenant" not in line for line in lines(tmp_path, "requests"))
    assert len([m for m in logged if "identity plugin failed 1 time(s)" in m]) == 1, (
        "the first failure is logged, the next only counted"
    )


@pytest.mark.parametrize(
    "spec",
    ["", "no_colon", ":attr", "module:", "tests.no_such_module:fn", f"{__name__}:no_such_name"],
)
def test_a_plugin_that_cannot_load_is_a_startup_error(spec):
    with pytest.raises(ValueError):
        load_identity_plugin(spec)


def test_a_plugin_that_is_not_callable_is_a_startup_error():
    with pytest.raises(ValueError, match="not callable"):
        load_identity_plugin(f"{__name__}:CHAT")


# -- configuration and installation -------------------------------------------------


def test_the_trace_is_absent_unless_the_directory_variable_is_set(monkeypatch):
    monkeypatch.delenv(REQUEST_TRACE_DIR_ENV, raising=False)
    monkeypatch.delenv(REQUEST_TRACE_IDENTITY_ENV, raising=False)
    assert RequestTrace.from_env() is None
    monkeypatch.setenv(REQUEST_TRACE_DIR_ENV, "   ")
    assert RequestTrace.from_env() is None, "a blank value is unset"

    app = FastAPI()
    register_middlewares(
        app,
        collect_perf_metrics=True,
        expose_perf_metrics=True,
        perf_metrics_writer=None,
        request_trace=RequestTrace.from_env(),
    )
    assert [m.cls for m in app.user_middleware] == [
        ServerArrivalTimeMiddleware,
        PerfMetricsMiddleware,
    ], "with the variable unset the middleware stack is the one from before the trace existed"


def test_from_env_builds_the_writer_and_the_plugin(tmp_path, monkeypatch):
    monkeypatch.setenv(REQUEST_TRACE_DIR_ENV, str(tmp_path / "trace"))
    monkeypatch.setenv(REQUEST_TRACE_IDENTITY_ENV, f"{__name__}:identity_fields")
    trace = RequestTrace.from_env()
    assert trace is not None
    assert trace.writer._output_dir == tmp_path / "trace"
    assert trace._identity is identity_fields

    monkeypatch.setenv(REQUEST_TRACE_IDENTITY_ENV, "nowhere:nothing")
    with pytest.raises(ValueError):
        RequestTrace.from_env()


def test_the_trace_is_installed_inside_the_arrival_stamp_and_outside_perf_metrics(tmp_path):
    trace = make_trace(tmp_path)
    for adjusted_clock in (None, object()):
        app = FastAPI()
        register_middlewares(
            app,
            collect_perf_metrics=True,
            expose_perf_metrics=False,
            perf_metrics_writer=None,
            adjusted_clock=adjusted_clock,
            request_trace=trace,
        )
        assert [(m.cls, m.kwargs) for m in app.user_middleware][1] == (
            RequestTraceMiddleware,
            {"trace": trace},
        )
        assert [m.cls for m in app.user_middleware] == [
            ServerArrivalTimeMiddleware,
            RequestTraceMiddleware,
            PerfMetricsMiddleware,
        ]


def test_the_trace_survives_a_directory_it_cannot_create(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("")
    trace = RequestTrace(RequestTraceWriter(str(blocker / "trace")))
    app = traced_app(trace)

    @app.post(CHAT)
    async def chat(request: Request):
        return {"ok": True}

    with TestClient(app) as client:
        assert client.post(CHAT, json={}).json() == {"ok": True}
    assert not list(tmp_path.glob("**/*.jsonl"))


def test_records_land_in_the_hour_bucket_of_their_own_timestamp(tmp_path):
    trace = make_trace(tmp_path)
    app = traced_app(trace)

    @app.post(CHAT)
    async def chat(request: Request):
        return {}

    with TestClient(app) as client:
        client.post(CHAT, json={})
    (request_path,) = tmp_path.glob("*/requests-w.jsonl")
    (response_path,) = tmp_path.glob("*/responses-w.jsonl")
    (request_line,) = lines(tmp_path, "requests")
    assert request_path.parent.name == request_line["recorded_at"][:13]
    assert request_path.parent.name == response_path.parent.name
    assert (tmp_path / "_writer").is_dir(), "the writer's accounting sits beside the buckets"
    assert os.environ.get(REQUEST_TRACE_DIR_ENV) is None, "the trace never reads the env at runtime"


# -- the trace never changes a reply ------------------------------------------------


def _deep_body_app(trace):
    app = traced_app(trace) if trace is not None else FastAPI()

    @app.post(CHAT)
    async def chat(body: dict):
        return {}

    return app


def test_a_body_too_deep_to_parse_is_answered_exactly_as_without_the_trace(tmp_path):
    """json.loads raises RecursionError, not ValueError, on a deeply nested body."""
    deep = b"[" * 50_000 + b"]" * 50_000
    headers = {"content-type": "application/json"}
    with TestClient(_deep_body_app(None), raise_server_exceptions=False) as client:
        without_trace = client.post(CHAT, content=deep, headers=headers)
    trace = make_trace(tmp_path)
    with TestClient(_deep_body_app(trace), raise_server_exceptions=False) as client:
        with_trace = client.post(CHAT, content=deep, headers=headers)

    assert with_trace.status_code == without_trace.status_code == 400, (
        "the trace must leave the server's own answer to a hostile body untouched"
    )
    assert with_trace.content == without_trace.content
    (request_line,) = lines(tmp_path, "requests")
    (response_line,) = lines(tmp_path, "responses")
    assert request_line["body"] == deep.decode()
    assert request_line["body_parse_error"].startswith("RecursionError")
    assert response_line["status"] == "rejected_400"


def test_a_failure_in_the_request_line_costs_the_line_not_the_request(tmp_path, logged):
    trace = make_trace(tmp_path)
    app = traced_app(trace)

    def broken(self, **kwargs):
        raise RuntimeError("trace bug")

    @app.post(CHAT)
    async def chat(request: Request):
        return {"ok": True}

    with TestClient(app) as client, pytest.MonkeyPatch.context() as patch:
        patch.setattr(RequestTrace, "record_request", broken)
        for _ in range(3):
            assert client.post(CHAT, json={"a": 1}).json() == {"ok": True}
    assert lines(tmp_path, "requests") == []
    assert [line["status"] for line in lines(tmp_path, "responses")] == ["completed"] * 3, (
        "the response lines still stand, with their trace ids, so the loss is visible"
    )
    failures = [
        m for m in logged if "the request line was not recorded (RuntimeError: trace bug)" in m
    ]
    assert len(failures) == 1, "the first failure is logged, the rest are counted"
    assert trace._failures == 3


def test_a_failure_in_the_response_line_costs_the_line_not_the_request(tmp_path, logged):
    trace = make_trace(tmp_path)
    app = traced_app(trace)

    def broken(self, **kwargs):
        raise RuntimeError("trace bug")

    @app.post(CHAT)
    async def chat(request: Request):
        return {"ok": True}

    with TestClient(app) as client, pytest.MonkeyPatch.context() as patch:
        patch.setattr(RequestTrace, "record_response", broken)
        assert client.post(CHAT, json={"a": 1}).json() == {"ok": True}
    assert [line["body"] for line in lines(tmp_path, "requests")] == [{"a": 1}]
    assert lines(tmp_path, "responses") == []
    assert [
        m for m in logged if "the response line was not recorded (RuntimeError: trace bug)" in m
    ]


def test_a_body_less_post_leaves_no_trace(tmp_path):
    """The disaggregated router polls POST /kv_cache_events on every worker for each request."""
    trace = make_trace(tmp_path)
    app = traced_app(trace)

    @app.post("/kv_cache_events")
    async def kv_cache_events():
        return {"events": []}

    @app.post(CHAT)
    async def chat(request: Request):
        return {}

    with TestClient(app) as client:
        assert client.post("/kv_cache_events").status_code == 200
        assert (
            client.post(CHAT, content=b"", headers={"content-type": "application/json"}).status_code
            == 200
        )
        assert client.post(CHAT, json={"a": 1}).status_code == 200
    assert [line["body"] for line in lines(tmp_path, "requests")] == [{"a": 1}]
    assert len(lines(tmp_path, "responses")) == 1


# -- the reply's work happens on the writer's thread ---------------------------------


def completion_chunk(index, finish_reason=None):
    """A /v1/completions chunk as the server serializes it: every field present."""
    chunk = {
        "id": "cmpl-1",
        "object": "text_completion",
        "choices": [
            {"index": 0, "text": f"t{index}", "logprobs": None, "finish_reason": finish_reason}
        ],
        "usage": None,
    }
    return f"data: {json.dumps(chunk)}\n\n"


def test_frames_that_only_mention_a_marker_are_not_parsed(monkeypatch):
    """Completion chunks carry "finish_reason":null and "usage":null on every chunk."""
    stream = "".join(completion_chunk(i) for i in range(300)) + completion_chunk(300, "stop")
    stream += "data: [DONE]\n\n"
    parses = []
    real_loads = json.loads
    monkeypatch.setattr(
        "tensorrt_llm.serve.middleware.request_trace.json.loads",
        lambda s, *a, **k: parses.append(s) or real_loads(s, *a, **k),
    )
    summary = summarize_reply({"kind": "sse_text", "body": stream}, "/v1/completions")
    assert summary == {}, "a text_completion stream is not a chat completion reply"
    # The two tail frames (the Responses terminal-event probe) plus the one frame whose
    # finish_reason is set; never the 300 frames that only mention the keys.
    assert len(parses) <= 3, f"{len(parses)} frames parsed; only the tail and set values may be"

    chat_with_nulls = (
        "".join(
            chat_chunk(str(i))
            .decode()
            .replace('"finish_reason": null', '"finish_reason": null, "usage": null')
            for i in range(300)
        )
        + chat_chunk(finish_reason="stop").decode()
    )
    parses.clear()
    summary = summarize_reply({"kind": "sse_text", "body": chat_with_nulls}, CHAT)
    assert summary == {"finish_reasons": ["stop"]}
    assert len(parses) <= 3


def test_the_reply_is_joined_decoded_and_summarized_on_the_writer_thread(tmp_path, monkeypatch):
    import threading

    trace = make_trace(tmp_path)
    app = traced_app(trace)
    loop_thread = []
    summary_thread = []
    import tensorrt_llm.serve.middleware.request_trace as module

    real_summarize = module.summarize_reply

    def spying_summarize(payload, route):
        summary_thread.append(threading.get_ident())
        return real_summarize(payload, route)

    monkeypatch.setattr(module, "summarize_reply", spying_summarize)

    @app.post(CHAT)
    async def chat(request: Request):
        loop_thread.append(threading.get_ident())
        return sse([chat_chunk("a"), chat_chunk(finish_reason="stop")])

    with TestClient(app) as client:
        client.post(CHAT, json={})
    (response_line,) = lines(tmp_path, "responses")
    assert response_line["finish_reasons"] == ["stop"], "the deferred fields were completed"
    assert response_line["response"]["kind"] == "sse_text"
    assert summary_thread and loop_thread and summary_thread[0] != loop_thread[0], (
        "joining, decoding and summarizing the reply must not run on the event loop"
    )


# -- the disconnect fact ------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_completed_reply_to_a_client_that_left_is_marked_disconnected(tmp_path):
    """Uvicorn accepts a reply for a gone client; the trace keeps the disconnect it saw."""
    trace = make_trace(tmp_path)

    async def app(scope, receive, send):
        await wait_for_disconnect(receive)  # the handler polls is_disconnected(), sees it
        await send(start_message(content_type=b"application/json"))
        await send(chunk_message(b'{"aborted":true}', more_body=False))

    send, sent = collecting_send()
    await run_asgi(
        trace, app, http_scope(), queued_receive(body_message(b'{"a":1}'), DISCONNECT), send
    )
    (response_line,) = lines(tmp_path, "responses")
    assert response_line["status"] == "completed"
    assert response_line["disconnected"] is True
    assert response_line["response"] == {"kind": "json", "body": {"aborted": True}}


@pytest.mark.asyncio
async def test_a_disconnect_message_after_the_reply_is_complete_says_nothing(tmp_path):
    """An ASGI server answers every receive after the response with http.disconnect."""
    trace = make_trace(tmp_path)

    async def app(scope, receive, send):
        await send(start_message(content_type=b"application/json"))
        await send(chunk_message(b"{}", more_body=False))
        await wait_for_disconnect(receive)  # the synthetic post-response disconnect

    send, sent = collecting_send()
    await run_asgi(
        trace, app, http_scope(), queued_receive(body_message(b'{"a":1}'), DISCONNECT), send
    )
    (response_line,) = lines(tmp_path, "responses")
    assert response_line["status"] == "completed"
    assert "disconnected" not in response_line


@pytest.mark.asyncio
async def test_a_client_disconnect_raised_while_reading_the_body_is_a_client_disconnect(tmp_path):
    from starlette.requests import ClientDisconnect

    trace = make_trace(tmp_path)

    async def app(scope, receive, send):
        await Request(scope, receive).body()  # Starlette raises ClientDisconnect here

    send, sent = collecting_send()
    with pytest.raises(ClientDisconnect):
        await run_asgi(
            trace,
            app,
            http_scope(),
            queued_receive(body_message(b'{"a":', more_body=True), DISCONNECT),
            send,
        )
    (request_line,) = lines(tmp_path, "requests")
    (response_line,) = lines(tmp_path, "responses")
    assert request_line["body_complete"] is False
    assert response_line["status"] == "client_disconnected"
    assert response_line["disconnected"] is True
    assert response_line["error"].startswith("ClientDisconnect")
