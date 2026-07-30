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
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request
from starlette.datastructures import Headers
from starlette.responses import StreamingResponse

from tensorrt_llm.llmapi.disagg_utils import ServerRole, extract_disagg_cfg
from tensorrt_llm.serve.openai_client import UpstreamRequestTimeoutError
from tensorrt_llm.serve.openai_disagg_server import (
    DisaggregatedRequestTimeoutError,
    OpenAIDisaggServer,
    _CleanupStreamingResponse,
)
from tensorrt_llm.serve.openai_protocol import (
    CompletionRequest,
    ConversationParams,
    DisaggregatedParams,
)


def _raw_request(headers: dict[str, str]):
    return SimpleNamespace(headers=Headers(headers=headers))


@pytest.mark.asyncio
async def test_shutdown_drains_request_cleanup_before_service_teardown():
    server = OpenAIDisaggServer.__new__(OpenAIDisaggServer)
    server._cleanup_grace_secs = 0.5
    server._background_cleanup_tasks = set()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    order = []

    async def cleanup():
        cleanup_started.set()
        await release_cleanup.wait()
        order.append("cleanup")

    cleanup_task = asyncio.create_task(cleanup())
    server._track_background_cleanup(cleanup_task, "test cleanup")

    async def teardown():
        order.append("teardown")

    server._service = SimpleNamespace(teardown=AsyncMock(side_effect=teardown))
    server._perf_metrics_collector = SimpleNamespace(_background_tasks=set())

    shutdown_task = asyncio.create_task(server._shutdown())
    await cleanup_started.wait()
    await asyncio.sleep(0)
    server._service.teardown.assert_not_awaited()

    release_cleanup.set()
    await asyncio.wait_for(shutdown_task, timeout=1)

    assert order == ["cleanup", "teardown"]
    assert server._background_cleanup_tasks == set()


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
    assert request.disaggregated_params.conversation_id is None


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
    assert request.disaggregated_params.conversation_id is None


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


@pytest.mark.asyncio
async def test_request_deadline_cancels_upstream_work():
    server = OpenAIDisaggServer.__new__(OpenAIDisaggServer)
    server._req_timeout_secs = 1
    work_started = asyncio.Event()
    work_cancelled = asyncio.Event()

    async def blocking_entry_point(_request, _hooks):
        work_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            work_cancelled.set()

    async def receive():
        await asyncio.Event().wait()

    raw_request = SimpleNamespace(receive=receive)
    deadline = asyncio.get_running_loop().time() + 0.01

    with pytest.raises(DisaggregatedRequestTimeoutError):
        await server._await_response_or_disconnect(
            blocking_entry_point, object(), object(), raw_request, deadline
        )

    assert work_started.is_set()
    assert work_cancelled.is_set()


@pytest.mark.asyncio
async def test_request_deadline_does_not_wait_forever_for_cleanup():
    server = OpenAIDisaggServer.__new__(OpenAIDisaggServer)
    server._req_timeout_secs = 1
    server._cleanup_grace_secs = 0.01
    server._background_cleanup_tasks = set()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()

    async def blocking_entry_point(_request, _hooks):
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await release_cleanup.wait()

    async def receive():
        await asyncio.Event().wait()

    raw_request = SimpleNamespace(receive=receive)
    deadline = asyncio.get_running_loop().time() + 0.01

    with pytest.raises(DisaggregatedRequestTimeoutError):
        await asyncio.wait_for(
            server._await_response_or_disconnect(
                blocking_entry_point, object(), object(), raw_request, deadline
            ),
            timeout=0.5,
        )

    assert cleanup_started.is_set()
    assert len(server._background_cleanup_tasks) == 1
    cleanup_tasks = tuple(server._background_cleanup_tasks)
    release_cleanup.set()
    await asyncio.gather(*cleanup_tasks, return_exceptions=True)
    await asyncio.sleep(0)
    assert server._background_cleanup_tasks == set()


@pytest.mark.asyncio
async def test_client_disconnect_cancels_upstream_work():
    server = OpenAIDisaggServer.__new__(OpenAIDisaggServer)
    server._req_timeout_secs = 180
    work_started = asyncio.Event()
    work_cancelled = asyncio.Event()

    async def blocking_entry_point(_request, _hooks):
        work_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            work_cancelled.set()

    async def receive():
        await work_started.wait()
        return {"type": "http.disconnect"}

    raw_request = Request({"type": "http"}, receive=receive)
    deadline = asyncio.get_running_loop().time() + 10

    with pytest.raises(asyncio.CancelledError):
        await server._await_response_or_disconnect(
            blocking_entry_point, object(), object(), raw_request, deadline
        )

    assert work_cancelled.is_set()


@pytest.mark.asyncio
async def test_completed_stream_setup_cancels_disconnect_listener():
    server = OpenAIDisaggServer.__new__(OpenAIDisaggServer)
    server._req_timeout_secs = 180
    receive_started = asyncio.Event()
    receive_cancelled = asyncio.Event()
    release_receive = asyncio.Event()
    response_stream = object()

    async def receive():
        receive_started.set()
        try:
            await release_receive.wait()
            return {"type": "http.disconnect"}
        finally:
            receive_cancelled.set()

    async def entry_point(_request, _hooks):
        await receive_started.wait()
        return response_stream

    raw_request = Request({"type": "http"}, receive=receive)
    deadline = asyncio.get_running_loop().time() + 10

    request_task = asyncio.create_task(
        server._await_response_or_disconnect(
            entry_point, SimpleNamespace(stream=True), object(), raw_request, deadline
        )
    )
    done, _ = await asyncio.wait((request_task,), timeout=1)
    completed_without_release = request_task in done
    if not completed_without_release:
        release_receive.set()
        await asyncio.wait((request_task,), timeout=1)

    assert completed_without_release
    assert request_task.result() is response_stream
    assert receive_cancelled.is_set()


@pytest.mark.asyncio
async def test_stream_setup_waits_for_exclusive_receive_handoff():
    server = OpenAIDisaggServer.__new__(OpenAIDisaggServer)
    server._req_timeout_secs = 180
    server._cleanup_grace_secs = 0.01
    server._background_cleanup_tasks = set()
    receive_cancelled = asyncio.Event()
    release_receive = asyncio.Event()
    response_stream = object()

    async def receive():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            receive_cancelled.set()
            await release_receive.wait()
            raise

    async def entry_point(_request, _hooks):
        return response_stream

    raw_request = Request({"type": "http"}, receive=receive)
    deadline = asyncio.get_running_loop().time() + 10
    request_task = asyncio.create_task(
        server._await_response_or_disconnect(
            entry_point, SimpleNamespace(stream=True), object(), raw_request, deadline
        )
    )

    await asyncio.wait_for(receive_cancelled.wait(), timeout=1)
    await asyncio.sleep(server._cleanup_grace_secs * 2)
    assert not request_task.done()

    release_receive.set()
    assert await asyncio.wait_for(request_task, timeout=1) is response_stream


@pytest.mark.asyncio
async def test_canceled_stream_setup_closes_completed_unclaimed_stream(monkeypatch):
    class UnclaimedStream:
        def __init__(self):
            self.closed = False

        async def aclose(self):
            self.closed = True

    server = OpenAIDisaggServer.__new__(OpenAIDisaggServer)
    server._req_timeout_secs = 180
    server._cleanup_grace_secs = 0.1
    server._background_cleanup_tasks = set()
    response_stream = UnclaimedStream()

    async def receive():
        await asyncio.Event().wait()

    async def entry_point(_request, _hooks):
        return response_stream

    real_wait = asyncio.wait
    wait_calls = 0

    async def cancel_first_wait(awaitables, *args, **kwargs):
        nonlocal wait_calls
        wait_calls += 1
        if wait_calls == 1:
            request_task, _ = tuple(awaitables)
            await request_task
            raise asyncio.CancelledError()
        return await real_wait(awaitables, *args, **kwargs)

    monkeypatch.setattr(asyncio, "wait", cancel_first_wait)
    raw_request = SimpleNamespace(receive=receive)
    deadline = asyncio.get_running_loop().time() + 10

    with pytest.raises(asyncio.CancelledError):
        await server._await_response_or_disconnect(
            entry_point, object(), object(), raw_request, deadline
        )

    assert response_stream.closed


@pytest.mark.asyncio
async def test_timed_out_stream_setup_closes_stream_returned_on_cancel():
    class UnclaimedStream:
        def __init__(self):
            self.closed = False

        async def aclose(self):
            self.closed = True

    server = OpenAIDisaggServer.__new__(OpenAIDisaggServer)
    server._req_timeout_secs = 1
    server._cleanup_grace_secs = 0.1
    server._background_cleanup_tasks = set()
    response_stream = UnclaimedStream()

    async def entry_point(_request, _hooks):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return response_stream

    async def receive():
        await asyncio.Event().wait()

    raw_request = SimpleNamespace(receive=receive)
    deadline = asyncio.get_running_loop().time() + 0.01

    with pytest.raises(DisaggregatedRequestTimeoutError):
        await server._await_response_or_disconnect(
            entry_point, object(), object(), raw_request, deadline
        )

    assert response_stream.closed
    assert server._background_cleanup_tasks == set()


@pytest.mark.parametrize(
    "exception, expected_message",
    [
        (DisaggregatedRequestTimeoutError(900), "900-second deadline"),
        (UpstreamRequestTimeoutError(ServerRole.CONTEXT, 180), "180-second timeout"),
    ],
)
def test_request_timeout_maps_to_structured_504(exception, expected_message):
    server = OpenAIDisaggServer.__new__(OpenAIDisaggServer)
    server._perf_metrics_collector = SimpleNamespace(http_exceptions=MagicMock())

    response = server._handle_exception(exception)
    body = json.loads(response.body)

    assert response.status_code == 504
    assert body["object"] == "error"
    assert body["type"] == "RequestTimeoutError"
    assert body["code"] == 504
    assert expected_message in body["message"]
    server._perf_metrics_collector.http_exceptions.inc.assert_called_once()


@pytest.mark.asyncio
async def test_stream_timeout_emits_error_then_done():
    server = OpenAIDisaggServer.__new__(OpenAIDisaggServer)
    server._req_timeout_secs = 900
    stream_closed = asyncio.Event()

    async def blocking_stream():
        try:
            await asyncio.Event().wait()
            yield b"unreachable"
        finally:
            stream_closed.set()

    deadline = asyncio.get_running_loop().time() + 0.01
    chunks = [chunk async for chunk in server._stream_with_deadline(blocking_stream(), deadline)]

    assert len(chunks) == 2
    error_event = json.loads(chunks[0].removeprefix("data: ").strip())
    assert error_event["error"]["object"] == "error"
    assert error_event["error"]["type"] == "RequestTimeoutError"
    assert error_event["error"]["code"] == 504
    assert chunks[1] == "data: [DONE]\n\n"
    assert stream_closed.is_set()


@pytest.mark.asyncio
async def test_stream_preserves_chunks():
    server = OpenAIDisaggServer.__new__(OpenAIDisaggServer)
    server._req_timeout_secs = 180

    async def stream():
        for chunk in (b"first", b"second", b"third"):
            yield chunk
            await asyncio.sleep(0)

    deadline = asyncio.get_running_loop().time() + 10
    chunks = [chunk async for chunk in server._stream_with_deadline(stream(), deadline)]

    assert chunks == [b"first", b"second", b"third"]


@pytest.mark.asyncio
async def test_streaming_response_disconnect_closes_upstream_work():
    server = OpenAIDisaggServer.__new__(OpenAIDisaggServer)
    server._req_timeout_secs = 180
    stream_started = asyncio.Event()
    stream_closed = asyncio.Event()

    async def blocking_stream():
        stream_started.set()
        try:
            await asyncio.Event().wait()
            yield b"unreachable"
        finally:
            # Exercise cleanup after Starlette has canceled the stream inside
            # its AnyIO cancel scope.
            await asyncio.sleep(0)
            stream_closed.set()

    async def receive():
        await stream_started.wait()
        return {"type": "http.disconnect"}

    async def send(_message):
        pass

    scope = {
        "type": "http",
        "asgi": {
            "version": "3.0",
            "spec_version": "2.3",
        },
    }
    deadline = asyncio.get_running_loop().time() + 10
    response = StreamingResponse(
        server._stream_with_deadline(blocking_stream(), deadline),
        media_type="text/event-stream",
    )

    await asyncio.wait_for(response(scope, receive, send), timeout=1)

    assert stream_closed.is_set()


@pytest.mark.asyncio
async def test_streaming_response_disconnect_closes_unstarted_upstream_work():
    class UnstartedStream:
        def __init__(self):
            self.started = False
            self.closed = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            self.started = True
            await asyncio.Event().wait()

        async def aclose(self):
            self.closed = True

    stream = UnstartedStream()
    server = OpenAIDisaggServer.__new__(OpenAIDisaggServer)
    server._cleanup_grace_secs = 0.1
    server._background_cleanup_tasks = set()

    async def receive():
        return {"type": "http.disconnect"}

    async def send(_message):
        await asyncio.Event().wait()

    scope = {
        "type": "http",
        "asgi": {
            "version": "3.0",
            "spec_version": "2.3",
        },
    }
    response = _CleanupStreamingResponse(
        stream,
        media_type="text/event-stream",
        cleanup=stream.aclose,
        cleanup_runner=server._run_cleanup_bounded,
    )

    await asyncio.wait_for(response(scope, receive, send), timeout=1)

    assert not stream.started
    assert stream.closed
