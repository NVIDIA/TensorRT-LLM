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
"""``RequestContext``: what the chat core reads from a request, on the HTTP and the internal path."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from starlette.requests import Request
from test_common.serve.app_factory import AggConfig

from tensorrt_llm.serve._base.msgspec_route import _MsgspecRequest
from tensorrt_llm.serve._base.request_context import RequestContext
from tensorrt_llm.serve.openai_client import MSGPACK_HEADERS, _msgpack_encoder

pytestmark = pytest.mark.cpu_only

BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


def _http_request(body: bytes = b"", headers=(), state=None) -> Request:
    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/messages",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers],
        "state": {"perf_metrics_records": []} if state is None else state,
    }
    return Request(scope, receive)


# -- from_http: delegation to the request FastAPI passed ----------------------


def test_from_http_shares_the_request_state():
    request = _http_request()
    ctx = RequestContext.from_http(request)

    assert ctx.http_request is request
    assert ctx.state is request.state
    assert ctx.perf_metrics_records is request.state.perf_metrics_records
    # A write through the context is a write on the request's state, which the
    # perf-metrics middleware reads when the response goes out.
    ctx.server_first_token_time = 2.0
    assert request.state.server_first_token_time == 2.0
    assert ctx.server_first_token_time == 2.0
    assert bool(ctx), "the chat core tests `if raw_request and ...`; a context is truthy"


def test_from_http_reads_headers_and_client_from_the_request():
    request = _http_request(headers=(("X-Conversation-Id", "c1"),))
    ctx = RequestContext.from_http(request)

    assert ctx.headers["x-conversation-id"] == "c1"
    assert ctx.client is None  # the scope above names no client
    assert ctx.server_arrival_time is None
    assert ctx.no_client_connection is False


def test_from_http_reads_nothing_at_construction():
    # Tests hand the chat core a bare namespace as the request; wrapping it
    # must not touch any attribute.
    ctx = RequestContext.from_http(SimpleNamespace())
    assert ctx.http_request is not None


def test_from_http_keeps_the_msgpack_decoding_of_the_route_request():
    """A worker's request decodes msgpack bodies; the context must reuse it, not rebuild one."""
    body = _msgpack_encoder.encode(BODY)
    request = _MsgspecRequest(
        _http_request(body=body, headers=tuple(MSGPACK_HEADERS.items())).scope,
        _http_request(body=body).receive,
    )
    ctx = RequestContext.from_http(request)

    assert asyncio.run(ctx.json()) == BODY


def test_from_http_is_disconnected_asks_the_request():
    request = _http_request()
    disconnected = {"value": False}

    async def is_disconnected():
        return disconnected["value"]

    request.is_disconnected = is_disconnected  # type: ignore[method-assign]
    ctx = RequestContext.from_http(request)

    assert asyncio.run(ctx.is_disconnected()) is False
    disconnected["value"] = True
    assert asyncio.run(ctx.is_disconnected()) is True


# -- internal: a request with no client behind it ------------------------------


def test_internal_context_has_no_client_and_seeds_the_state():
    ctx = RequestContext.internal(BODY)

    assert ctx.http_request is None
    assert dict(ctx.headers) == {}
    assert ctx.client is None
    assert ctx.no_client_connection is True
    assert ctx.perf_metrics_records == []
    assert ctx.server_arrival_time is None
    assert ctx.server_first_token_time is None
    assert asyncio.run(ctx.json()) == BODY
    assert asyncio.run(ctx.is_disconnected()) is False
    assert bool(ctx)


def test_internal_json_is_what_a_wire_body_decodes_to_and_is_computed_once():
    # A body read from the wire has been through JSON: tuples become lists,
    # non-string keys become strings. The internal context answers the same.
    ctx = RequestContext.internal({"a": (1, 2), 3: "x"})

    first = asyncio.run(ctx.json())
    assert first == {"a": [1, 2], "3": "x"}
    assert asyncio.run(ctx.json()) is first


def test_internal_state_writes_stay_in_the_context():
    ctx = RequestContext.internal(BODY)
    ctx.server_first_token_time = 5.0
    ctx.perf_metrics_records.append({"k": "v"})

    assert ctx.state.server_first_token_time == 5.0
    assert ctx.perf_metrics_records == [{"k": "v"}]


def test_a_context_needs_a_request_or_an_internal_state():
    with pytest.raises(ValueError, match="from_http"):
        RequestContext()


def test_await_disconnected_returns_immediately_for_an_internal_context(serve_env):
    """A batched request has no socket, so the watchdog must not poll on it.

    is_disconnected() can never become true without a client, so without the
    no_client_connection opt-out this coroutine would loop at 1Hz for the life
    of the process, one live task per batched request.
    """
    server = serve_env.build_openai_server(AggConfig())
    promise = SimpleNamespace(finished=False, request_id="req", abort=Mock())

    async def run():
        # A socket-backed request would never let this return; a bounded wait
        # keeps the failure a clear timeout rather than a hung suite.
        await asyncio.wait_for(
            server.await_disconnected(RequestContext.internal(BODY), promise), timeout=5.0
        )

    asyncio.run(run())
    # Returning early must not be mistaken for a disconnect.
    promise.abort.assert_not_called()
