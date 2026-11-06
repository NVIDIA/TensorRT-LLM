# Copyright (c) 2025-2026, NVIDIA CORPORATION.
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
import errno
import json
import socket
import struct
import zlib
from contextlib import aclosing
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import aiohttp
import msgspec
import pytest

from tensorrt_llm._utils import AdjustedSteadyClock
from tensorrt_llm.disaggregated_params import DisaggScheduleStyle
from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.serve.conversation_id import SUBAGENT_AFFINITY_HEADER
from tensorrt_llm.serve.disagg_auth import (
    INTERNAL_DISAGG_AUTH_HEADER,
    SUBAGENT_AFFINITY_AUTH_HEADER,
    validate_internal_disagg_request,
    validate_subagent_affinity,
)
from tensorrt_llm.serve.openai_client import OpenAIHttpClient, _encode_request_body
from tensorrt_llm.serve.openai_protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    ConversationParams,
    DisaggregatedParams,
    ResponsesRequest,
    UsageInfo,
)
from tensorrt_llm.serve.perf_metrics import (
    _PERF_METRICS_HEADER_BUDGET_BYTES,
    CLOCK_SYNC_HEADER,
    RETURN_METRICS_HEADER,
    SERVER_TIMING_HEADER,
    SSE_METRICS_EVENT,
    START_END_TIME_HEADER,
    PerfMetricsMiddleware,
    adjusted_clock_from_headers,
)
from tensorrt_llm.serve.responses_utils import ResponseHooks, ServerArrivalTimeMiddleware
from tensorrt_llm.serve.router import Router

pytestmark = pytest.mark.cpu_only


def _reset_prometheus_registry():
    from prometheus_client.registry import REGISTRY

    REGISTRY._names_to_collectors = {}
    REGISTRY._collector_to_names = {}


def test_adjusted_steady_clock_uses_reference_domain():
    source = Mock(return_value=10.0)
    clock = AdjustedSteadyClock(2.0, time_source=source)

    assert clock.now() == 12.0
    assert clock.to_reference_time(20.0) == 22.0

    clock.set_reference_offset(-3.0)
    assert clock.now() == 7.0


@pytest.mark.asyncio
async def test_worker_clock_calibration_uses_global_clock():
    from tensorrt_llm.serve.openai_server import OpenAIServer

    server = object.__new__(OpenAIServer)
    global_clock = Mock(side_effect=[10.0, 10.2])
    delay = AsyncMock()
    with (
        patch(
            "tensorrt_llm.serve.openai_server.get_global_steady_clock_now_in_seconds",
            global_clock,
        ),
        patch("tensorrt_llm.serve.openai_server.asyncio.sleep", delay),
    ):
        response = await server.get_steady_clock_offset()

    assert json.loads(response.body) == {
        "receive_ts": 10.0,
        "transmit_ts": 10.2,
    }
    assert global_clock.call_count == 2
    delay.assert_awaited_once_with(0.2)


@pytest.fixture
def mock_router():
    """Create a mock router."""
    router = AsyncMock(spec=Router)
    router.servers = ["localhost:8000", "localhost:8001"]
    router.get_next_server = AsyncMock(return_value=("localhost:8000", None))
    router.finish_request = AsyncMock()
    return router


@pytest.fixture
def mock_session():
    """Create a mock aiohttp session."""
    return AsyncMock(spec=aiohttp.ClientSession)


@pytest.fixture
def openai_client(mock_router, mock_session):
    """Create an OpenAIHttpClient instance."""
    # uninitialize the prometheus metrics collector or it will raise a duplicate metric error
    _reset_prometheus_registry()
    return OpenAIHttpClient(
        router=mock_router,
        role=ServerRole.CONTEXT,
        timeout_secs=180,
        max_retries=2,
        retry_interval_sec=1,
        session=mock_session,
    )


@pytest.fixture
def completion_request():
    """Create a sample non-streaming CompletionRequest."""
    return CompletionRequest(
        model="test-model",
        prompt="Hello, world!",
        stream=False,
        disaggregated_params=DisaggregatedParams(
            request_type="generation_only", first_gen_tokens=[123], ctx_request_id=123
        ),
    )


@pytest.fixture
def streaming_completion_request():
    """Create a sample streaming CompletionRequest."""
    return CompletionRequest(
        model="test-model",
        prompt="Hello, world!",
        stream=True,
        disaggregated_params=DisaggregatedParams(
            request_type="generation_only", first_gen_tokens=[456], ctx_request_id=456
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("process_offset", [0.0, 1000.0])
async def test_perf_metrics_middleware_reports_effective_frontend_clock(process_offset):
    """Each frontend shard reports timestamps in its effective metrics clock."""
    sent = []

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    async def send(message):
        sent.append(message)

    clock = AdjustedSteadyClock(process_offset, time_source=Mock(side_effect=[10.0, 10.25]))
    middleware = ServerArrivalTimeMiddleware(
        PerfMetricsMiddleware(
            app,
            expose_headers=True,
            adjusted_clock=clock,
        ),
        adjusted_clock=clock,
    )
    scope = {
        "type": "http",
        "headers": [(RETURN_METRICS_HEADER.lower().encode(), b"1")],
    }
    await middleware(scope, AsyncMock(), send)

    response_headers = dict(sent[0]["headers"])
    assert response_headers[CLOCK_SYNC_HEADER.encode()].decode() == (
        f"receive;ts={10.0 + process_offset:.9f}, transmit;ts={10.25 + process_offset:.9f}"
    )


@pytest.mark.parametrize(
    "header",
    [
        {},
        {CLOCK_SYNC_HEADER: "receive;ts=invalid, transmit;ts=1"},
        {CLOCK_SYNC_HEADER: "receive;ts=1"},
        {CLOCK_SYNC_HEADER: "receive;ts=nan, transmit;ts=1"},
    ],
)
def test_invalid_clock_sync_header_is_ignored(header):
    clock = adjusted_clock_from_headers(header, 1.0, 2.0)
    assert clock.to_reference_time(1.0) == 1.0


class TestOpenAIHttpClient:
    """Test OpenAIHttpClient main functionality."""

    def dummy_response(self):
        return CompletionResponse(
            id="test-123",
            object="text_completion",
            created=1234567890,
            model="test-model",
            usage=UsageInfo(prompt_tokens=10, completion_tokens=10),
            choices=[CompletionResponseChoice(index=0, text="Hello!")],
        )

    def test_initialization(self, mock_router, mock_session):
        """Test client initialization."""
        client = OpenAIHttpClient(
            router=mock_router,
            role=ServerRole.GENERATION,
            timeout_secs=300,
            max_retries=5,
            session=mock_session,
        )
        assert client._router == mock_router
        assert client._role == ServerRole.GENERATION
        assert client._session == mock_session
        assert client._max_retries == 5

    @pytest.mark.asyncio
    @pytest.mark.parametrize("clock_delta", [0.0, 1000.0])
    async def test_request_metrics_normalize_frontend_clock_domain(
        self,
        clock_delta,
        openai_client,
        completion_request,
        mock_session,
    ):
        """Metrics from synchronized and unsynchronized shards share one clock."""
        openai_client._request_perf_metrics = True
        response = self.dummy_response()
        http_response = AsyncMock()
        http_response.status = 200
        http_response.headers = {
            "Content-Type": "application/json",
            CLOCK_SYNC_HEADER: (
                f"receive;ts={100.05 + clock_delta:.9f}, transmit;ts={100.35 + clock_delta:.9f}"
            ),
            START_END_TIME_HEADER: (
                f"server-start;ts={100.1 + clock_delta:.9f}, "
                f"server-end;ts={100.3 + clock_delta:.9f}"
            ),
            SERVER_TIMING_HEADER: (
                "server_queue;dur=10.0, server_ttft;dur=50.0, server_e2e;dur=200.0"
            ),
        }
        http_response.json = AsyncMock(return_value=response.model_dump())
        http_response.__aenter__ = AsyncMock(return_value=http_response)
        http_response.__aexit__ = AsyncMock()
        mock_session.post.return_value = http_response
        hooks = MagicMock(spec=ResponseHooks)

        with patch(
            "tensorrt_llm.serve.openai_client.get_steady_clock_now_in_seconds",
            side_effect=[100.0, 100.4, 100.5],
        ):
            await openai_client.send_request(completion_request, hooks=hooks)

        _, role, record = hooks.on_perf_metrics.call_args.args
        timing = record["phases"]["ctx"]["timing_metrics"]
        assert role == "ctx"
        assert timing["arrival_time"] == pytest.approx(100.1)
        assert timing["first_scheduled_time"] == pytest.approx(100.11)
        assert timing["first_token_time"] == pytest.approx(100.15)
        assert timing["last_token_time"] == pytest.approx(100.3)

    @pytest.mark.asyncio
    async def test_streaming_metrics_normalize_frontend_clock_domain(
        self,
        openai_client,
        streaming_completion_request,
        mock_session,
    ):
        openai_client._request_perf_metrics = True
        clock_delta = 1000.0
        http_response = AsyncMock()
        http_response.status = 200
        http_response.headers = {
            "Content-Type": "text/event-stream",
            CLOCK_SYNC_HEADER: (
                f"receive;ts={100.05 + clock_delta:.9f}, transmit;ts={100.35 + clock_delta:.9f}"
            ),
        }
        metrics_headers = {
            START_END_TIME_HEADER: (
                f"server-start;ts={100.1 + clock_delta:.9f}, "
                f"server-end;ts={100.3 + clock_delta:.9f}"
            ),
            SERVER_TIMING_HEADER: (
                "server_queue;dur=10.0, server_ttft;dur=50.0, server_e2e;dur=200.0"
            ),
        }
        metrics_event = (
            f"event: {SSE_METRICS_EVENT}\ndata: {json.dumps(metrics_headers)}\n\n"
        ).encode()

        async def mock_iter_any():
            yield b'data: "Hello"\n\ndata: [DONE]\n\n'
            yield metrics_event

        http_response.content = AsyncMock()
        http_response.content.iter_any = mock_iter_any
        http_response.__aenter__ = AsyncMock(return_value=http_response)
        http_response.__aexit__ = AsyncMock()
        mock_session.post.return_value = http_response
        hooks = MagicMock(spec=ResponseHooks)

        with patch(
            "tensorrt_llm.serve.openai_client.get_steady_clock_now_in_seconds",
            side_effect=[100.0, 100.4, 100.45, 100.5, 100.6],
        ):
            generator = await openai_client.send_request(streaming_completion_request, hooks=hooks)
            chunks = [chunk async for chunk in generator]

        assert b"".join(chunks) == b'data: "Hello"\n\ndata: [DONE]\n\n'
        _, role, record = hooks.on_perf_metrics.call_args.args
        timing = record["phases"]["ctx"]["timing_metrics"]
        assert role == "ctx"
        assert timing["arrival_time"] == pytest.approx(100.1)
        assert timing["first_token_time"] == pytest.approx(100.15)
        assert timing["last_token_time"] == pytest.approx(100.3)

    @pytest.mark.asyncio
    async def test_internal_client_accepts_perf_metrics_header_size(self, mock_router):
        with (
            patch("tensorrt_llm.serve.openai_client.ClientMetricsCollector"),
            patch("tensorrt_llm.serve.openai_client.aiohttp.ClientSession") as session,
        ):
            OpenAIHttpClient(router=mock_router, role=ServerRole.GENERATION)

        assert session.call_args.kwargs["max_field_size"] == _PERF_METRICS_HEADER_BUDGET_BYTES

    @pytest.mark.asyncio
    async def test_generation_request_with_opaque_state_is_signed(self, mock_router, mock_session):
        """Opaque state forwarded to generation workers gets internal auth."""
        _reset_prometheus_registry()
        client = OpenAIHttpClient(
            router=mock_router,
            role=ServerRole.GENERATION,
            timeout_secs=300,
            max_retries=0,
            session=mock_session,
            internal_disagg_auth_key="secret",
        )
        mock_response = self.dummy_response()
        mock_http_response = AsyncMock()
        mock_http_response.status = 200
        mock_http_response.headers = {"Content-Type": "application/json"}
        mock_http_response.json = AsyncMock(return_value=mock_response.model_dump())
        mock_http_response.__aenter__ = AsyncMock(return_value=mock_http_response)
        mock_http_response.__aexit__ = AsyncMock()
        mock_session.post.return_value = mock_http_response

        request = CompletionRequest(
            model="test-model",
            prompt="Hello, world!",
            stream=False,
            disaggregated_params=DisaggregatedParams(
                request_type="generation_only",
                encoded_opaque_state="b3BhcXVl",
            ),
        )

        await client.send_request(request)

        headers = mock_session.post.call_args.kwargs["headers"]
        assert headers[INTERNAL_DISAGG_AUTH_HEADER].startswith("sha256=")

    @pytest.mark.asyncio
    async def test_generation_request_with_opaque_state_without_key_warns(
        self, mock_router, mock_session
    ):
        """Opaque state without auth key emits a transitional warning."""
        _reset_prometheus_registry()
        client = OpenAIHttpClient(
            router=mock_router,
            role=ServerRole.GENERATION,
            timeout_secs=300,
            max_retries=0,
            session=mock_session,
        )
        mock_response = self.dummy_response()
        mock_http_response = AsyncMock()
        mock_http_response.status = 200
        mock_http_response.headers = {"Content-Type": "application/json"}
        mock_http_response.json = AsyncMock(return_value=mock_response.model_dump())
        mock_http_response.__aenter__ = AsyncMock(return_value=mock_http_response)
        mock_http_response.__aexit__ = AsyncMock()
        mock_session.post.return_value = mock_http_response
        request = CompletionRequest(
            model="test-model",
            prompt="Hello, world!",
            stream=False,
            disaggregated_params=DisaggregatedParams(
                request_type="generation_only",
                encoded_opaque_state="b3BhcXVl",
            ),
        )

        warning_message = "In a future release the requirement to use internal_request_auth_key"
        with pytest.warns(FutureWarning, match=warning_message):
            await client.send_request(request)

        headers = mock_session.post.call_args.kwargs["headers"]
        assert INTERNAL_DISAGG_AUTH_HEADER not in headers

    @pytest.mark.asyncio
    async def test_non_streaming_completion_request(
        self, openai_client, completion_request, mock_session, mock_router
    ):
        """Test non-streaming completion request end-to-end."""
        mock_response = self.dummy_response()

        # Mock HTTP response
        mock_http_response = AsyncMock()
        mock_http_response.status = 200
        mock_http_response.headers = {"Content-Type": "application/json"}
        mock_http_response.json = AsyncMock(return_value=mock_response.model_dump())
        mock_http_response.raise_for_status = Mock()
        mock_http_response.__aenter__ = AsyncMock(return_value=mock_http_response)
        mock_http_response.__aexit__ = AsyncMock()

        mock_session.post.return_value = mock_http_response

        # Send request
        response = await openai_client.send_request(completion_request)

        # Assertions
        assert isinstance(response, CompletionResponse)
        assert response.model == "test-model"
        mock_session.post.assert_called_once()
        mock_router.finish_request.assert_called_once_with(
            completion_request, mock_session, success=True
        )

    @pytest.mark.asyncio
    async def test_streaming_completion_request(
        self, openai_client, streaming_completion_request, mock_session, mock_router
    ):
        """Test streaming completion request end-to-end."""
        # Mock HTTP streaming response
        mock_http_response = AsyncMock()
        mock_http_response.status = 200
        mock_http_response.headers = {"Content-Type": "text/event-stream"}

        dummy_data = [
            b'data: "Hello"\n\n',
            b'data: "world"\n\n',
            b'data: "!"\n\n',
        ]

        async def mock_iter_any():
            for data in dummy_data:
                yield data

        mock_http_response.content = AsyncMock()
        mock_http_response.content.iter_any = mock_iter_any
        mock_http_response.__aenter__ = AsyncMock(return_value=mock_http_response)
        mock_http_response.__aexit__ = AsyncMock()

        mock_session.post.return_value = mock_http_response

        # Send streaming request
        response_generator = await openai_client.send_request(streaming_completion_request)

        # Consume the generator
        chunks = []
        async for chunk in response_generator:
            chunks.append(chunk)

        # Assertions
        assert len(chunks) == 3
        for i, chunk in enumerate(chunks):
            assert chunk == dummy_data[i]
        mock_session.post.assert_called_once()
        mock_router.finish_request.assert_called_once_with(
            streaming_completion_request, mock_session, success=True
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("together", [False, True], ids=["separate", "same-chunk"])
    async def test_responses_first_token_follows_the_opening_events(
        self, openai_client, mock_session, together
    ):
        """The worker sends response.created and response.in_progress before generating."""
        request = ResponsesRequest(model="test-model", input="hi", stream=True)
        opening = b"event: response.created\ndata: {}\n\nevent: response.in_progress\ndata: {}\n\n"
        delta = b"event: response.output_text.delta\ndata: {}\n\n"
        sent = [opening + delta] if together else [opening, delta]

        async def mock_iter_any():
            for data in sent:
                yield data

        mock_http_response = AsyncMock()
        mock_http_response.status = 200
        mock_http_response.headers = {"Content-Type": "text/event-stream"}
        mock_http_response.content = AsyncMock()
        mock_http_response.content.iter_any = mock_iter_any
        mock_http_response.__aenter__ = AsyncMock(return_value=mock_http_response)
        mock_http_response.__aexit__ = AsyncMock()
        mock_session.post.return_value = mock_http_response
        received = []
        hooks = MagicMock(spec=ResponseHooks)
        first_token_after = []
        hooks.on_first_token.side_effect = lambda *args: first_token_after.append(len(received))

        response_generator = await openai_client.send_request(request, hooks=hooks)
        async for chunk in response_generator:
            received.append(chunk)

        assert received == sent
        # Fired on the chunk carrying the first output, not on the opening pair.
        assert first_token_after == [len(sent) - 1]

    @pytest.mark.asyncio
    async def test_streaming_perf_metrics_preserve_sse_event_boundaries(
        self, openai_client, streaming_completion_request, mock_session, mock_router
    ):
        openai_client._request_perf_metrics = True
        mock_http_response = AsyncMock()
        mock_http_response.status = 200
        mock_http_response.headers = {"Content-Type": "text/event-stream"}

        response_data = b'data: {"choices":[{"text":"Hello"}]}\n\n'
        done_data = b"data: [DONE]\n\n"
        metrics_data = (
            f'event: {SSE_METRICS_EVENT}\ndata: {{"Server-Timing":"server_ttft;dur=1.0"}}\n\n'
        ).encode()
        marker_split = len("event: trtllm")

        async def mock_iter_any():
            yield response_data
            yield done_data + metrics_data[:marker_split]
            yield metrics_data[marker_split:]

        mock_http_response.content = AsyncMock()
        mock_http_response.content.iter_any = mock_iter_any
        mock_http_response.__aenter__ = AsyncMock(return_value=mock_http_response)
        mock_http_response.__aexit__ = AsyncMock()
        mock_session.post.return_value = mock_http_response
        hooks = MagicMock(spec=ResponseHooks)

        response_generator = await openai_client.send_request(
            streaming_completion_request, hooks=hooks
        )
        chunks = [chunk async for chunk in response_generator]

        assert chunks == [response_data, done_data]
        hooks.on_first_token.assert_called_once_with("localhost:8000", streaming_completion_request)
        hooks.on_perf_metrics.assert_called_once()
        hooks.on_resp_done.assert_called_once_with(
            "localhost:8000", streaming_completion_request, None
        )
        mock_router.finish_request.assert_called_once_with(
            streaming_completion_request, mock_session, success=True
        )

    @pytest.mark.asyncio
    async def test_malformed_streaming_metrics_do_not_fail_request(
        self, openai_client, streaming_completion_request, mock_session, mock_router
    ):
        openai_client._request_perf_metrics = True
        mock_http_response = AsyncMock()
        mock_http_response.status = 200
        mock_http_response.headers = {"Content-Type": "text/event-stream"}

        response_data = b'data: "Hello"\n\ndata: [DONE]\n\n'
        metrics_data = f"event: {SSE_METRICS_EVENT}\ndata: not-json\n\n".encode()

        async def mock_iter_any():
            yield b""
            yield response_data
            yield metrics_data

        mock_http_response.content = AsyncMock()
        mock_http_response.content.iter_any = mock_iter_any
        mock_http_response.__aenter__ = AsyncMock(return_value=mock_http_response)
        mock_http_response.__aexit__ = AsyncMock()
        mock_session.post.return_value = mock_http_response
        hooks = MagicMock(spec=ResponseHooks)

        response_generator = await openai_client.send_request(
            streaming_completion_request, hooks=hooks
        )
        chunks = [chunk async for chunk in response_generator]

        assert b"".join(chunks) == response_data
        hooks.on_first_token.assert_called_once_with("localhost:8000", streaming_completion_request)
        hooks.on_perf_metrics.assert_not_called()
        hooks.on_resp_done.assert_called_once_with(
            "localhost:8000", streaming_completion_request, None
        )
        mock_router.finish_request.assert_called_once_with(
            streaming_completion_request, mock_session, success=True
        )

    @pytest.mark.asyncio
    async def test_request_with_custom_server(
        self, openai_client, completion_request, mock_session, mock_router
    ):
        """Test sending request to a specific server."""
        custom_server = "localhost:9000"
        mock_response = self.dummy_response()

        mock_http_response = AsyncMock()
        mock_http_response.status = 200
        mock_http_response.headers = {"Content-Type": "application/json"}
        mock_http_response.json = AsyncMock(return_value=mock_response.model_dump())
        mock_http_response.__aenter__ = AsyncMock(return_value=mock_http_response)
        mock_http_response.__aexit__ = AsyncMock()

        mock_session.post.return_value = mock_http_response

        await openai_client.send_request(completion_request, server=custom_server)

        # Verify custom server was used in URL
        call_args = mock_session.post.call_args[0][0]
        assert custom_server in call_args
        # Router should not be called when server is specified
        mock_router.get_next_server.assert_not_called()

    @pytest.mark.asyncio
    async def test_request_error_handling(
        self, openai_client, completion_request, mock_session, mock_router
    ):
        """Test error handling when request fails."""
        mock_session.post.side_effect = aiohttp.ClientError("Connection failed")

        with pytest.raises(aiohttp.ClientError):
            await openai_client.send_request(completion_request)

        # Should finish request on error with success=False so the router
        # doesn't record routed-block cache state for a request that didn't complete.
        mock_router.finish_request.assert_called_once_with(
            completion_request, mock_session, success=False
        )

    @pytest.mark.asyncio
    async def test_request_with_retry(
        self, openai_client, completion_request, mock_session, mock_router
    ):
        """Test retry mechanism on transient failures."""
        mock_response = self.dummy_response()

        mock_http_response = AsyncMock()
        mock_http_response.status = 200
        mock_http_response.headers = {"Content-Type": "application/json"}
        mock_http_response.json = AsyncMock(return_value=mock_response.model_dump())
        mock_http_response.__aenter__ = AsyncMock(return_value=mock_http_response)
        mock_http_response.__aexit__ = AsyncMock()

        # First attempt fails, second succeeds
        mock_session.post.side_effect = [
            aiohttp.ClientError("Temporary failure"),
            mock_http_response,
        ]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            response = await openai_client.send_request(completion_request)

        assert isinstance(response, CompletionResponse)
        assert mock_session.post.call_count == 2  # Initial + 1 retry

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(
        self, openai_client, completion_request, mock_session, mock_router
    ):
        """Test that request fails after max retries."""
        mock_session.post.side_effect = aiohttp.ClientError("Connection failed")

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(aiohttp.ClientError):
                await openai_client.send_request(completion_request)

        # Should try max_retries + 1 times
        assert mock_session.post.call_count == openai_client._max_retries + 1
        mock_router.finish_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_request_type(self, openai_client):
        """Test handling of invalid request type."""
        with pytest.raises(ValueError, match="Invalid request type"):
            await openai_client.send_request("invalid_request")

    def test_generation_request_with_ctx_info_endpoint_is_signed(self, mock_router, mock_session):
        _reset_prometheus_registry()
        client = OpenAIHttpClient(
            router=mock_router,
            role=ServerRole.GENERATION,
            session=mock_session,
            internal_disagg_auth_key="secret",
        )
        request = CompletionRequest(
            model="test-model",
            prompt="Hello, world!",
            disaggregated_params=DisaggregatedParams(
                request_type="generation_only",
                ctx_request_id=1,
                disagg_request_id=2,
                ctx_info_endpoint="tcp://10.0.0.1:5000",
            ),
        )

        headers = client._get_request_headers(request)

        assert headers is not None
        assert INTERNAL_DISAGG_AUTH_HEADER in headers

    def test_generation_request_with_ctx_info_endpoint_without_key_warns(
        self, mock_router, mock_session
    ):
        _reset_prometheus_registry()
        client = OpenAIHttpClient(
            router=mock_router,
            role=ServerRole.GENERATION,
            session=mock_session,
        )
        request = CompletionRequest(
            model="test-model",
            prompt="Hello, world!",
            disaggregated_params=DisaggregatedParams(
                request_type="generation_only",
                ctx_request_id=1,
                disagg_request_id=2,
                ctx_info_endpoint="tcp://10.0.0.1:5000",
            ),
        )

        warning_message = "In a future release the requirement to use internal_request_auth_key"
        with pytest.warns(FutureWarning, match=warning_message):
            headers = client._get_request_headers(request)

        assert INTERNAL_DISAGG_AUTH_HEADER not in headers


class TestHttpErrorBodyPreservation:
    """Test that HTTP 4xx/5xx errors include the response body (TRTLLM-11123)."""

    def _mock_http_error(self, status, body):
        r = AsyncMock()
        r.status = status
        r.reason = "Bad Request" if status == 400 else "Internal Server Error"
        r.text = AsyncMock(return_value=body)
        r.content.read = AsyncMock(side_effect=BytesIO(body.encode("utf-8")).read)
        r.headers = {"Content-Type": "application/json"}
        r.request_info = MagicMock()
        r.history = ()
        r.__aenter__ = AsyncMock(return_value=r)
        r.__aexit__ = AsyncMock(return_value=False)
        return r

    def _make_client(self, session, max_retries=0, role=ServerRole.CONTEXT, **kwargs):
        from prometheus_client.registry import REGISTRY

        REGISTRY._names_to_collectors = {}
        REGISTRY._collector_to_names = {}

        router = AsyncMock(spec=Router)
        router.servers = ["localhost:8000"]
        router.get_next_server = AsyncMock(return_value=("localhost:8000", None))
        router.finish_request = AsyncMock()
        return OpenAIHttpClient(
            router=router,
            role=role,
            timeout_secs=10,
            max_retries=max_retries,
            retry_interval_sec=0,
            session=session,
            **kwargs,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("role", [ServerRole.CONTEXT, ServerRole.GENERATION])
    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize(
        "status,body",
        [
            (400, '{"error":"missing field X"}'),
            (500, "internal failure detail"),
        ],
    )
    async def test_error_body_in_exception(self, monkeypatch, status, body, stream, role):
        monkeypatch.delenv("TRTLLM_DISAGG_NO_RETRY", raising=False)
        session = AsyncMock(spec=aiohttp.ClientSession)
        http_response = self._mock_http_error(status, body)
        session.post.return_value = http_response
        next_id = AsyncMock(return_value=2)
        client = self._make_client(session, max_retries=1, role=role, disagg_id_generator=next_id)
        req = CompletionRequest(
            model="m",
            prompt="hi",
            stream=stream,
            disaggregated_params=DisaggregatedParams(
                request_type="context_only" if role == ServerRole.CONTEXT else "generation_only",
                disagg_request_id=1,
            ),
        )
        with pytest.raises(aiohttp.ClientResponseError) as exc_info:
            response = await client.send_request(req)
            if stream:
                async with aclosing(response):
                    async for _ in response:
                        pytest.fail("An HTTP rejection must not yield streaming output")
        assert exc_info.value.status == status
        assert exc_info.value.message == f"{http_response.reason}: {body}"
        assert exc_info.value.headers == http_response.headers
        # Streaming rejections never replay. Non-streaming CTX retains its retry
        # budget, but an accepted GEN leg cannot acquire a new pairing identity.
        retried = not stream and role == ServerRole.CONTEXT
        assert session.post.call_count == (2 if retried else 1)
        assert next_id.await_count == int(retried)
        assert req.disaggregated_params.disagg_request_id == (2 if retried else 1)
        if not retried:
            client._router.renew_request.assert_not_awaited()
        finish_kwargs = {"req_id": 1} if role == ServerRole.CONTEXT else {}
        client._router.finish_request.assert_awaited_once_with(
            req, session, success=False, **finish_kwargs
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("role", [ServerRole.CONTEXT, ServerRole.GENERATION])
    @pytest.mark.parametrize(
        "error",
        [aiohttp.ClientPayloadError("truncated body"), ConnectionResetError("reset while reading")],
    )
    async def test_streaming_rejected_body_failure_is_not_retried(
        self, monkeypatch: pytest.MonkeyPatch, error: Exception, role: ServerRole
    ) -> None:
        """A diagnostic-read failure cannot replay an already rejected stream."""
        monkeypatch.delenv("TRTLLM_DISAGG_NO_RETRY", raising=False)
        session = AsyncMock(spec=aiohttp.ClientSession)
        http_response = self._mock_http_error(500, "unavailable")
        http_response.content.read.side_effect = error
        session.post.return_value = http_response
        next_id = AsyncMock(return_value=2)
        client = self._make_client(session, max_retries=1, role=role, disagg_id_generator=next_id)
        req = CompletionRequest(
            model="m",
            prompt="hi",
            stream=True,
            disaggregated_params=DisaggregatedParams(
                request_type="context_only" if role == ServerRole.CONTEXT else "generation_only",
                disagg_request_id=1,
            ),
        )

        with pytest.raises(type(error)) as exc_info:
            response = await client.send_request(req)
            async with aclosing(response):
                async for _ in response:
                    pytest.fail("A rejected stream must not yield output")

        assert exc_info.value is error
        session.post.assert_called_once()
        next_id.assert_not_awaited()
        assert req.disaggregated_params.disagg_request_id == 1
        client._router.renew_request.assert_not_awaited()
        http_response.__aexit__.assert_awaited_once()
        finish_kwargs = {"req_id": 1} if role == ServerRole.CONTEXT else {}
        client._router.finish_request.assert_awaited_once_with(
            req, session, success=False, **finish_kwargs
        )

    @pytest.mark.asyncio
    async def test_streaming_error_body_read_is_bounded(self) -> None:
        """Read only a byte prefix, even when it splits a UTF-8 character."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        body = "x" * 2047 + "\u20ac" + "y" * 8192
        http_response = self._mock_http_error(500, body)
        session.post.return_value = http_response
        client = self._make_client(session)
        req = CompletionRequest(model="m", prompt="hi", stream=True)

        with pytest.raises(aiohttp.ClientResponseError) as exc_info:
            response = await client.send_request(req)
            async with aclosing(response):
                async for _ in response:
                    pytest.fail("An HTTP rejection must not yield streaming output")

        http_response.content.read.assert_awaited_once_with(2048)
        http_response.text.assert_not_awaited()
        assert exc_info.value.status == 500
        assert exc_info.value.message == f"{http_response.reason}: {'x' * 2047}\ufffd"
        http_response.__aexit__.assert_awaited_once()
        client._router.finish_request.assert_awaited_once_with(req, session, success=False)

    @pytest.mark.asyncio
    async def test_streaming_error_body_preserves_fragmented_diagnostic(self) -> None:
        """A short network read must not truncate the worker's error classification."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        body = '{"error":"input too long","code":"context_length_exceeded"}'
        http_response = self._mock_http_error(400, body)
        body_stream = BytesIO(body.encode("utf-8"))
        http_response.content.read.side_effect = lambda size: body_stream.read(min(size, 9))
        session.post.return_value = http_response
        client = self._make_client(session)
        req = CompletionRequest(model="m", prompt="hi", stream=True)

        with pytest.raises(aiohttp.ClientResponseError) as exc_info:
            response = await client.send_request(req)
            async with aclosing(response):
                async for _ in response:
                    pytest.fail("An HTTP rejection must not yield streaming output")

        assert exc_info.value.status == 400
        assert exc_info.value.message == f"{http_response.reason}: {body}"
        assert http_response.content.read.await_count > 1
        assert body_stream.tell() == len(body.encode("utf-8"))
        http_response.text.assert_not_awaited()
        session.post.assert_called_once()
        http_response.__aexit__.assert_awaited_once()
        client._router.finish_request.assert_awaited_once_with(req, session, success=False)


class TestDisaggIdRegenOnRetry:
    """Test that disagg_request_id is regenerated on retry (TRTLLM-11123)."""

    def _ok_response(self):
        return CompletionResponse(
            model="m",
            usage=UsageInfo(prompt_tokens=1, completion_tokens=1),
            choices=[CompletionResponseChoice(index=0, text="ok")],
        ).model_dump()

    def _mock_http_ok(self, json_val):
        r = AsyncMock()
        r.status = 200
        r.headers = {"Content-Type": "application/json"}
        r.json = AsyncMock(return_value=json_val)
        r.__aenter__ = AsyncMock(return_value=r)
        r.__aexit__ = AsyncMock()
        return r

    def _make_client(self, session, role=ServerRole.CONTEXT, **kwargs):
        from prometheus_client.registry import REGISTRY

        REGISTRY._names_to_collectors = {}
        REGISTRY._collector_to_names = {}

        router = AsyncMock(spec=Router)
        router.servers = ["localhost:8000"]
        router.get_next_server = AsyncMock(return_value=("localhost:8000", None))
        router.finish_request = AsyncMock()
        return OpenAIHttpClient(
            router=router,
            role=role,
            timeout_secs=10,
            max_retries=2,
            retry_interval_sec=0,
            session=session,
            **kwargs,
        )

    @pytest.mark.asyncio
    async def test_retry_regenerates_disagg_id(self):
        session = AsyncMock(spec=aiohttp.ClientSession)
        ids = iter(range(1000, 2000))

        async def next_id():
            return next(ids)

        client = self._make_client(session, disagg_id_generator=next_id)

        session.post.side_effect = [
            aiohttp.ClientError("transient"),
            self._mock_http_ok(self._ok_response()),
        ]
        req = CompletionRequest(
            model="m",
            prompt="hi",
            stream=False,
            disaggregated_params=DisaggregatedParams(
                request_type="context_only", disagg_request_id=42
            ),
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            resp = await client.send_request(req)

        assert req.disaggregated_params.disagg_request_id != 42
        assert isinstance(resp, CompletionResponse)

    @pytest.mark.asyncio
    async def test_no_generator_keeps_original_id(self):
        session = AsyncMock(spec=aiohttp.ClientSession)
        client = self._make_client(session)  # no disagg_id_generator

        session.post.side_effect = [
            aiohttp.ClientError("transient"),
            self._mock_http_ok(self._ok_response()),
        ]
        req = CompletionRequest(
            model="m",
            prompt="hi",
            stream=False,
            disaggregated_params=DisaggregatedParams(
                request_type="context_only", disagg_request_id=42
            ),
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await client.send_request(req)

        assert req.disaggregated_params.disagg_request_id == 42

    @pytest.mark.asyncio
    @pytest.mark.parametrize("role", [ServerRole.CONTEXT, ServerRole.GENERATION])
    @pytest.mark.parametrize("regenerate_id", [False, True])
    async def test_retry_affinity_signature_matches_wire_request(
        self, role: ServerRole, regenerate_id: bool
    ) -> None:
        session = AsyncMock(spec=aiohttp.ClientSession)
        client = self._make_client(
            session,
            role=role,
            internal_disagg_auth_key="secret",
            disagg_id_generator=AsyncMock(return_value=1000) if regenerate_id else None,
        )
        # A failure that leaves the request undelivered, so both roles retry.
        session.post.side_effect = [
            ConnectionResetError(),
            self._mock_http_ok(self._ok_response()),
        ]
        request = CompletionRequest(
            model="m",
            prompt="hi",
            stream=False,
            conversation_params=ConversationParams(
                conversation_id="child", subagent_affinity_id="parent"
            ),
            disaggregated_params=DisaggregatedParams(
                request_type="context_only" if role == ServerRole.CONTEXT else "generation_only",
                disagg_request_id=42,
                encoded_opaque_state="b3BhcXVl" if role == ServerRole.GENERATION else None,
            ),
        )

        await client.send_request(request)
        # Only a context retry regenerates the request ID; a generation leg keeps
        # the ID that names its context KV cache.
        regenerated = regenerate_id and role == ServerRole.CONTEXT
        expected_ids = [42, 1000 if regenerated else 42]

        # Serialized attempts retain their own IDs when a context retry
        # regenerates the request ID.
        assert session.post.call_count == len(expected_ids)
        bodies = [
            msgspec.msgpack.decode(call.kwargs["data"]) for call in session.post.call_args_list
        ]
        headers = [dict(call.kwargs["headers"]) for call in session.post.call_args_list]
        assert [
            body["disaggregated_params"]["disagg_request_id"] for body in bodies
        ] == expected_ids
        wire_requests = [CompletionRequest.model_validate(body) for body in bodies]
        for body, wire_request, attempt_headers in zip(bodies, wire_requests, headers):
            assert "subagent_affinity_id" not in body["conversation_params"]
            assert attempt_headers[SUBAGENT_AFFINITY_HEADER] == "parent"
            assert (
                validate_subagent_affinity("secret", wire_request, role, attempt_headers)
                == "parent"
            )
            if role == ServerRole.GENERATION:
                validate_internal_disagg_request("secret", wire_request, attempt_headers)
        if role == ServerRole.GENERATION:
            assert (
                headers[0][INTERNAL_DISAGG_AUTH_HEADER] == headers[1][INTERNAL_DISAGG_AUTH_HEADER]
            )
        if regenerated:
            assert (
                headers[0][SUBAGENT_AFFINITY_AUTH_HEADER]
                != headers[1][SUBAGENT_AFFINITY_AUTH_HEADER]
            )
            for index in (0, 1):
                with pytest.raises(ValueError, match="Invalid internal subagent"):
                    validate_subagent_affinity(
                        "secret", wire_requests[index], role, headers[1 - index]
                    )
        else:
            assert (
                headers[0][SUBAGENT_AFFINITY_AUTH_HEADER]
                == headers[1][SUBAGENT_AFFINITY_AUTH_HEADER]
            )

    @pytest.mark.asyncio
    async def test_retry_keeps_original_reservation_id_without_explicit_req_id(self):
        """Renew/finish must keep using the id the request was routed with.

        Without an explicit req_id the coordinator router keys a context
        reservation by disagg_request_id, which the retry path re-issues. If the
        client let that leak into renew/finish, the coordinator would look up a
        reservation it never created and the original one would linger until it
        expired.
        """
        session = AsyncMock(spec=aiohttp.ClientSession)
        ids = iter(range(1000, 2000))

        async def next_id():
            return next(ids)

        client = self._make_client(session, disagg_id_generator=next_id)
        session.post.side_effect = [
            aiohttp.ClientError("transient"),
            self._mock_http_ok(self._ok_response()),
        ]
        req = CompletionRequest(
            model="m",
            prompt="hi",
            stream=False,
            disaggregated_params=DisaggregatedParams(
                request_type="context_only", disagg_request_id=42
            ),
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await client.send_request(req)

        assert req.disaggregated_params.disagg_request_id != 42
        assert client._router.renew_request.await_count == 2
        assert all(
            call.kwargs["req_id"] == 42 for call in client._router.renew_request.await_args_list
        )
        assert client._router.finish_request.await_count == 1
        assert client._router.finish_request.await_args.kwargs["req_id"] == 42


class TestSelectiveTransientTcpRetry:
    """Selective retry budget for transient TCP race symptoms.

    ServerDisconnectedError and connection resets, including aiohttp's
    ClientOSError wrapper, get an extended budget of up to 5 retries;
    all other client errors keep the original
    max_retries fail-fast behaviour.
    """

    def _ok_response(self):
        return CompletionResponse(
            id="cmpl-1",
            object="text_completion",
            created=0,
            model="m",
            choices=[CompletionResponseChoice(index=0, text="ok", finish_reason="stop")],
            usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    def _mock_http_ok(self, body):
        r = AsyncMock()
        r.status = 200
        r.headers = {"Content-Type": "application/json"}
        r.json = AsyncMock(return_value=body.model_dump())
        r.__aenter__ = AsyncMock(return_value=r)
        r.__aexit__ = AsyncMock()
        return r

    def _make_client(self, session, max_retries=1):
        from prometheus_client.registry import REGISTRY

        REGISTRY._names_to_collectors = {}
        REGISTRY._collector_to_names = {}

        router = AsyncMock(spec=Router)
        router.servers = ["localhost:8000"]
        router.get_next_server = AsyncMock(return_value=("localhost:8000", None))
        router.finish_request = AsyncMock()
        return OpenAIHttpClient(
            router=router,
            role=ServerRole.CONTEXT,
            timeout_secs=10,
            max_retries=max_retries,
            retry_interval_sec=0,
            session=session,
        )

    def _make_request(self):
        return CompletionRequest(
            model="m",
            prompt="hi",
            stream=False,
            disaggregated_params=DisaggregatedParams(
                request_type="context_only", disagg_request_id=1
            ),
        )

    @pytest.mark.asyncio
    async def test_retry_renews_coordinator_reservation(self):
        session = AsyncMock(spec=aiohttp.ClientSession)
        client = self._make_client(session)
        session.post.side_effect = [
            aiohttp.ClientError("transient"),
            self._mock_http_ok(self._ok_response()),
        ]
        request = self._make_request()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await client.send_request(request, req_id=71)

        assert client._router.renew_request.await_count == 2
        assert all(
            call.args == (request,) and call.kwargs == {"req_id": 71}
            for call in client._router.renew_request.await_args_list
        )

    @pytest.mark.asyncio
    async def test_server_disconnected_gets_extra_retries(self):
        """ServerDisconnectedError: even with max_retries=1, retry up to 5."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        client = self._make_client(session, max_retries=1)

        # 4 disconnect failures then success on the 5th attempt
        session.post.side_effect = [
            aiohttp.ServerDisconnectedError(),
            aiohttp.ServerDisconnectedError(),
            aiohttp.ServerDisconnectedError(),
            aiohttp.ServerDisconnectedError(),
            self._mock_http_ok(self._ok_response()),
        ]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await client.send_request(self._make_request())

        # 1 original + 4 retries = 5 total attempts (extra budget kicked in)
        assert session.post.call_count == 5

    @pytest.mark.asyncio
    @pytest.mark.parametrize("error_type", [ConnectionResetError, aiohttp.ClientOSError])
    async def test_connection_reset_gets_extra_retries(self, error_type):
        """OS connection resets recover after more than the ordinary retry budget."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        client = self._make_client(session, max_retries=1)

        session.post.side_effect = [
            error_type(errno.ECONNRESET, "Connection reset by peer"),
            error_type(errno.ECONNRESET, "Connection reset by peer"),
            self._mock_http_ok(self._ok_response()),
        ]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            response = await client.send_request(self._make_request())

        assert response.choices[0].text == "ok"
        # 1 original + 2 retries = 3 total attempts (within extra budget)
        assert session.post.call_count == 3

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [
            aiohttp.ClientError("transient non-tcp"),
            aiohttp.ClientOSError(errno.ECONNREFUSED, "Connection refused"),
            aiohttp.ClientOSError(errno.EACCES, "Permission denied"),
        ],
    )
    async def test_other_client_error_keeps_fail_fast(self, error):
        """Errors other than a transient reset respect max_retries (=1)."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        client = self._make_client(session, max_retries=1)

        session.post.side_effect = error

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(aiohttp.ClientError):
                await client.send_request(self._make_request())

        # Original + 1 retry = 2 attempts, NOT promoted to 5
        assert session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_zero_still_gets_transient_tcp_budget(self):
        """Even when max_retries=0, transient TCP races still retry up to 5."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        client = self._make_client(session, max_retries=0)

        session.post.side_effect = [
            aiohttp.ServerDisconnectedError(),
            self._mock_http_ok(self._ok_response()),
        ]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await client.send_request(self._make_request())

        assert session.post.call_count == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [
            aiohttp.ServerDisconnectedError(),
            aiohttp.ClientOSError(errno.ECONNRESET, "Connection reset by peer"),
        ],
    )
    async def test_no_retry_env_disables_all_http_retries(self, monkeypatch, error):
        monkeypatch.setenv("TRTLLM_DISAGG_NO_RETRY", "1")
        session = AsyncMock(spec=aiohttp.ClientSession)
        with patch("tensorrt_llm.serve.openai_client.logger.info") as log_info:
            client = self._make_client(session, max_retries=5)
        assert "TRTLLM_DISAGG_NO_RETRY=1" in log_info.call_args.args[0]
        session.post.side_effect = error
        with pytest.raises(type(error)):
            await client.send_request(self._make_request())
        assert session.post.call_count == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [
            aiohttp.ServerDisconnectedError(),
            aiohttp.ClientOSError(errno.ECONNRESET, "Connection reset by peer"),
        ],
    )
    async def test_transient_tcp_capped_at_5_when_max_retries_smaller(self, error):
        """If transient TCP keeps failing, give up after the extended budget."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        client = self._make_client(session, max_retries=1)

        # Always raise — must give up after extended (1 + 5) = 6 attempts
        session.post.side_effect = error

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(type(error)):
                await client.send_request(self._make_request())

        # 1 original + 5 retries
        assert session.post.call_count == 6


def test_a_forwarded_request_keeps_the_field_names_it_arrived_with():
    """`schema` must not reach a worker spelled `schema_`, its pydantic name."""
    request = ResponsesRequest.model_validate(
        {
            "model": "m",
            "input": "hi",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "structured_output",
                    "schema": {"type": "object"},
                    "strict": True,
                }
            },
        }
    )

    body = msgspec.msgpack.decode(_encode_request_body(request))

    assert body["text"]["format"]["schema"] == {"type": "object"}
    assert "schema_" not in body["text"]["format"]
    assert ResponsesRequest.model_validate(body).text.format.schema_ == {"type": "object"}


class TestRequestCleanup:
    @pytest.fixture
    def client(self, openai_client):
        openai_client._max_retries = 1
        openai_client._retry_interval_sec = 0
        openai_client._metrics_collector = MagicMock()
        return openai_client

    @staticmethod
    def _response(chunks):
        response = AsyncMock()
        response.status = 200
        response.headers = {"Content-Type": "text/event-stream"}
        response.content.iter_any = chunks
        response.__aenter__.return_value = response
        response.__aexit__.return_value = False
        return response

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [False, True])
    async def test_preheader_failure_finishes_once(
        self, client, completion_request, mock_session, mock_router, stream
    ):
        completion_request.stream = stream
        mock_session.post.side_effect = aiohttp.ClientOSError(104, "Connection reset by peer")

        with pytest.raises(aiohttp.ClientOSError):
            response = await client.send_request(completion_request, req_id=42)
            if stream:
                _ = [chunk async for chunk in response]

        assert mock_session.post.call_count == 6
        mock_router.finish_request.assert_awaited_once_with(
            completion_request, mock_session, success=False, req_id=42
        )
        client._metrics_collector.error_requests.inc.assert_called_once_with()
        client._metrics_collector.completed_requests.inc.assert_called_once_with()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("failure_stage", ["headers", "body"])
    async def test_retry_keeps_reservation_until_success(
        self, client, streaming_completion_request, mock_session, mock_router, failure_stage
    ):
        async def failed_chunks():
            raise aiohttp.ClientOSError(104, "Connection reset by peer")
            yield  # pragma: no cover

        async def successful_chunks():
            yield b"data: [DONE]\n\n"

        failed_response = self._response(failed_chunks)
        successful_response = self._response(successful_chunks)
        attempts = 0

        def post(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            mock_router.finish_request.assert_not_called()
            if attempts == 1:
                if failure_stage == "headers":
                    raise aiohttp.ClientOSError(104, "Connection reset by peer")
                return failed_response
            return successful_response

        mock_session.post.side_effect = post
        response = await client.send_request(streaming_completion_request, req_id=42)
        assert [chunk async for chunk in response] == [b"data: [DONE]\n\n"]

        assert attempts == 2
        mock_router.finish_request.assert_awaited_once_with(
            streaming_completion_request, mock_session, success=True, req_id=42
        )
        client._metrics_collector.error_requests.inc.assert_not_called()
        client._metrics_collector.completed_requests.inc.assert_called_once_with()
        client._metrics_collector.retry_requests.inc.assert_called_once_with()
        successful_response.__aexit__.assert_awaited_once()
        if failure_stage == "body":
            failed_response.__aexit__.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("error_type", [aiohttp.ClientOSError, RuntimeError])
    async def test_failure_after_first_chunk_finishes_once_without_retry(
        self, client, streaming_completion_request, mock_session, mock_router, error_type
    ):
        error = (
            error_type(errno.ECONNRESET, "Connection reset by peer")
            if issubclass(error_type, OSError)
            else error_type("stream failed")
        )

        async def chunks():
            yield b'data: "first"\n\n'
            raise error

        http_response = self._response(chunks)
        mock_session.post.return_value = http_response
        response = await client.send_request(streaming_completion_request, req_id=42)
        assert await response.__anext__() == b'data: "first"\n\n'
        with pytest.raises(error_type):
            await response.__anext__()
        await response.aclose()

        mock_session.post.assert_called_once()
        http_response.__aexit__.assert_awaited_once()
        mock_router.finish_request.assert_awaited_once_with(
            streaming_completion_request, mock_session, success=False, req_id=42
        )
        client._metrics_collector.error_requests.inc.assert_called_once_with()
        client._metrics_collector.completed_requests.inc.assert_called_once_with()
        client._metrics_collector.retry_requests.inc.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("read_first", [False, True])
    async def test_explicit_close_finishes_once(
        self, client, streaming_completion_request, mock_session, mock_router, read_first
    ):
        async def chunks():
            yield b'data: "first"\n\n'
            yield b"data: [DONE]\n\n"

        http_response = self._response(chunks)
        mock_session.post.return_value = http_response
        response = await client.send_request(streaming_completion_request, req_id=42)
        mock_session.post.assert_not_called()
        if read_first:
            assert await response.__anext__() == b'data: "first"\n\n'
        await response.aclose()
        await response.aclose()

        assert mock_session.post.call_count == int(read_first)
        assert http_response.__aexit__.await_count == int(read_first)
        mock_router.finish_request.assert_awaited_once_with(
            streaming_completion_request, mock_session, success=False, req_id=42
        )
        client._metrics_collector.error_requests.inc.assert_called_once_with()
        client._metrics_collector.completed_requests.inc.assert_called_once_with()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cancel_stage", ["headers", "body"])
    async def test_cancellation_finishes_once(
        self, client, streaming_completion_request, mock_session, mock_router, cancel_stage
    ):
        entered = asyncio.Event()
        blocked = asyncio.Event()

        async def wait_for_cancel():
            entered.set()
            await blocked.wait()

        async def chunks():
            await wait_for_cancel()
            yield b"data: [DONE]\n\n"

        http_response = self._response(chunks)
        if cancel_stage == "headers":
            http_response.__aenter__.side_effect = wait_for_cancel
        mock_session.post.return_value = http_response
        response = await client.send_request(streaming_completion_request, req_id=42)
        pending = asyncio.create_task(response.__anext__())
        try:
            await asyncio.wait_for(entered.wait(), timeout=1)
        finally:
            pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await response.aclose()

        mock_session.post.assert_called_once()
        assert http_response.__aexit__.await_count == int(cancel_stage == "body")
        mock_router.finish_request.assert_awaited_once_with(
            streaming_completion_request, mock_session, success=False, req_id=42
        )
        client._metrics_collector.error_requests.inc.assert_called_once_with()
        client._metrics_collector.completed_requests.inc.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_successful_stream_finishes_once(
        self, client, streaming_completion_request, mock_session, mock_router
    ):
        async def chunks():
            yield b"data: [DONE]\n\n"

        http_response = self._response(chunks)
        mock_session.post.return_value = http_response
        response = await client.send_request(streaming_completion_request, req_id=42)
        assert [chunk async for chunk in response] == [b"data: [DONE]\n\n"]
        await response.aclose()

        mock_router.finish_request.assert_awaited_once_with(
            streaming_completion_request, mock_session, success=True, req_id=42
        )
        client._metrics_collector.error_requests.inc.assert_not_called()
        client._metrics_collector.completed_requests.inc.assert_called_once_with()
        http_response.__aexit__.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cancel_stage", ["headers", "body", "exit"])
    async def test_non_stream_cancellation_finishes_once(
        self, client, completion_request, mock_session, mock_router, cancel_stage
    ):
        entered = asyncio.Event()
        blocked = asyncio.Event()

        async def wait_for_cancel(*args):
            entered.set()
            await blocked.wait()

        http_response = AsyncMock()
        http_response.status = 200
        http_response.headers = {"Content-Type": "application/json"}
        http_response.json.return_value = TestOpenAIHttpClient().dummy_response().model_dump()
        http_response.__aenter__.return_value = http_response
        http_response.__aexit__.return_value = False
        waiting_method = {
            "headers": http_response.__aenter__,
            "body": http_response.json,
            "exit": http_response.__aexit__,
        }[cancel_stage]
        waiting_method.side_effect = wait_for_cancel
        mock_session.post.return_value = http_response
        pending = asyncio.create_task(client.send_request(completion_request, req_id=42))
        try:
            await asyncio.wait_for(entered.wait(), timeout=1)
        finally:
            pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

        mock_session.post.assert_called_once()
        assert http_response.__aexit__.await_count == int(cancel_stage != "headers")
        mock_router.finish_request.assert_awaited_once_with(
            completion_request, mock_session, success=False, req_id=42
        )
        client._metrics_collector.error_requests.inc.assert_called_once_with()
        client._metrics_collector.completed_requests.inc.assert_called_once_with()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("error_type", [aiohttp.ClientOSError, RuntimeError])
    async def test_non_stream_context_exit_error_finishes_once(
        self, client, completion_request, mock_session, mock_router, error_type
    ):
        client._max_retries = 0
        http_response = AsyncMock()
        http_response.status = 200
        http_response.headers = {"Content-Type": "application/json"}
        http_response.json.return_value = TestOpenAIHttpClient().dummy_response().model_dump()
        http_response.__aenter__.return_value = http_response
        http_response.__aexit__.side_effect = error_type("closing HTTP response failed")
        mock_session.post.return_value = http_response

        with pytest.raises(error_type):
            await client.send_request(completion_request, req_id=42)

        mock_session.post.assert_called_once()
        http_response.__aexit__.assert_awaited_once()
        mock_router.finish_request.assert_awaited_once_with(
            completion_request, mock_session, success=False, req_id=42
        )
        client._metrics_collector.error_requests.inc.assert_called_once_with()
        client._metrics_collector.completed_requests.inc.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_non_stream_hook_error_closes_http_response(
        self, client, completion_request, mock_session, mock_router
    ):
        http_response = AsyncMock()
        http_response.status = 200
        http_response.headers = {"Content-Type": "application/json"}
        http_response.json.return_value = TestOpenAIHttpClient().dummy_response().model_dump()
        http_response.__aenter__.return_value = http_response
        http_response.__aexit__.return_value = False
        mock_session.post.return_value = http_response
        hooks = MagicMock(spec=ResponseHooks)
        hooks.on_ctx_resp.side_effect = RuntimeError("response hook failed")

        with pytest.raises(RuntimeError, match="response hook failed"):
            await client.send_request(completion_request, hooks=hooks, req_id=42)

        http_response.__aexit__.assert_awaited_once()
        mock_router.finish_request.assert_awaited_once_with(
            completion_request, mock_session, success=False, req_id=42
        )
        client._metrics_collector.error_requests.inc.assert_called_once_with()
        client._metrics_collector.completed_requests.inc.assert_called_once_with()


class TestRemoteTailHandoffRetry:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize(
        "role,request_type,schedule_style,remote_tail_start,bounded_replay",
        [
            (ServerRole.GENERATION, "generation_only", None, 0, False),
            (ServerRole.GENERATION, "generation_only", DisaggScheduleStyle.CONTEXT_FIRST, 8, False),
            (
                ServerRole.GENERATION,
                "generation_only",
                DisaggScheduleStyle.GENERATION_FIRST,
                None,
                True,
            ),
            (ServerRole.CONTEXT, "context_only", DisaggScheduleStyle.GENERATION_FIRST, None, True),
            (ServerRole.CONTEXT, "context_only", 1, None, True),
        ],
    )
    @pytest.mark.parametrize(
        "error_type", [aiohttp.ClientOSError, aiohttp.ServerDisconnectedError, ConnectionResetError]
    )
    async def test_remote_tail_handoff_does_not_retry_or_change_id(
        self,
        openai_client,
        mock_session,
        mock_router,
        stream,
        role,
        request_type,
        schedule_style,
        remote_tail_start,
        bounded_replay,
        error_type,
    ):
        openai_client._role = role
        openai_client._bounded_replay_on_generation = bounded_replay
        openai_client._metrics_collector = MagicMock()
        openai_client._disagg_id_generator = AsyncMock(return_value=999)
        mock_session.post.side_effect = (
            error_type(errno.ECONNRESET, "Connection reset by peer")
            if issubclass(error_type, OSError)
            else error_type("connection failed")
        )
        request = CompletionRequest(
            model="m",
            prompt="hi",
            stream=stream,
            disaggregated_params=DisaggregatedParams(
                request_type=request_type,
                disagg_request_id=123,
                schedule_style=schedule_style,
                remote_tail_start=remote_tail_start,
            ),
        )
        hooks = MagicMock(spec=ResponseHooks)
        with pytest.raises(error_type):
            response = await openai_client.send_request(request, hooks=hooks, req_id=42)
            if stream:
                _ = [chunk async for chunk in response]

        mock_session.post.assert_called_once()
        openai_client._disagg_id_generator.assert_not_awaited()
        assert request.disaggregated_params.disagg_request_id == 123
        hooks.on_disagg_request_id.assert_not_called()
        openai_client._metrics_collector.retry_requests.inc.assert_not_called()
        openai_client._metrics_collector.error_requests.inc.assert_called_once_with()
        openai_client._metrics_collector.completed_requests.inc.assert_called_once_with()
        mock_router.finish_request.assert_awaited_once_with(
            request, mock_session, success=False, req_id=42
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "role,request_type,schedule_style,bounded_replay",
        [
            (ServerRole.GENERATION, None, None, True),
            (ServerRole.CONTEXT, "context_only", DisaggScheduleStyle.GENERATION_FIRST, False),
            (ServerRole.CONTEXT, "context_only", DisaggScheduleStyle.CONTEXT_FIRST, True),
        ],
    )
    @pytest.mark.parametrize("error_type", [aiohttp.ClientError, aiohttp.ServerDisconnectedError])
    async def test_other_requests_retain_retry(
        self,
        openai_client,
        completion_request,
        mock_session,
        mock_router,
        role,
        request_type,
        schedule_style,
        bounded_replay,
        error_type,
    ):
        openai_client._role = role
        openai_client._bounded_replay_on_generation = bounded_replay
        openai_client._retry_interval_sec = 0
        # Transient TCP failures retain their budget even without normal retries.
        openai_client._max_retries = 0 if error_type is aiohttp.ServerDisconnectedError else 2
        openai_client._disagg_id_generator = AsyncMock(return_value=999)
        completion_request.disaggregated_params = (
            DisaggregatedParams(
                request_type=request_type, disagg_request_id=123, schedule_style=schedule_style
            )
            if request_type is not None
            else None
        )
        response_body = TestOpenAIHttpClient().dummy_response()
        http_response = AsyncMock()
        http_response.status = 200
        http_response.headers = {"Content-Type": "application/json"}
        http_response.json.return_value = response_body.model_dump()
        http_response.__aenter__.return_value = http_response
        http_response.__aexit__.return_value = False
        mock_session.post.side_effect = [error_type("connection failed"), http_response]
        hooks = MagicMock(spec=ResponseHooks)

        response = await openai_client.send_request(completion_request, hooks=hooks, req_id=42)

        assert response == response_body
        assert mock_session.post.call_count == 2
        if request_type is not None:
            assert completion_request.disaggregated_params.disagg_request_id == 999
            openai_client._disagg_id_generator.assert_awaited_once_with()
            hooks.on_disagg_request_id.assert_called_once_with(999)
        else:
            openai_client._disagg_id_generator.assert_not_awaited()
            hooks.on_disagg_request_id.assert_not_called()
        mock_router.finish_request.assert_awaited_once_with(
            completion_request, mock_session, success=True, req_id=42
        )


class TestCtxFrontendAffinity:
    """frontend_affinity=(K, S) picks a CTX frontend port per conversation."""

    @staticmethod
    def _client(mock_router, mock_session, affinity, role=ServerRole.CONTEXT):
        _reset_prometheus_registry()
        return OpenAIHttpClient(
            router=mock_router, role=role, session=mock_session, frontend_affinity=affinity
        )

    @staticmethod
    def _request(conversation_id):
        conversation_params = (
            None if conversation_id is None else ConversationParams(conversation_id=conversation_id)
        )
        return CompletionRequest(
            model="test-model", prompt="Hello", conversation_params=conversation_params
        )

    @pytest.mark.parametrize("server", ["localhost:8000", "[::1]:8000"])
    @pytest.mark.parametrize(
        "affinity,role,conversation_id",
        [
            ((4, 100), ServerRole.CONTEXT, "conv-a"),
            ((4, 100), ServerRole.CONTEXT, "conv-e"),
            (None, ServerRole.CONTEXT, "conv-a"),
            ((4, 100), ServerRole.GENERATION, "conv-a"),
            ((4, 100), ServerRole.CONTEXT, None),
        ],
    )
    def test_frontend_port(
        self, mock_router, mock_session, server, affinity, role, conversation_id
    ):
        client = self._client(mock_router, mock_session, affinity, role)
        expected = server
        if affinity is not None and role == ServerRole.CONTEXT and conversation_id is not None:
            frontend = zlib.crc32(conversation_id.encode()) % 4
            expected = f"{server.rpartition(':')[0]}:{8000 + 100 * (frontend + 1)}"
        assert client._url_server(server, self._request(conversation_id)) == expected

    @pytest.mark.asyncio
    async def test_send_request_posts_to_frontend_port(self, mock_router, mock_session):
        client = self._client(mock_router, mock_session, (8, 10))
        response_body = TestOpenAIHttpClient().dummy_response()
        http_response = AsyncMock()
        http_response.status = 200
        http_response.headers = {"Content-Type": "application/json"}
        http_response.json = AsyncMock(return_value=response_body.model_dump())
        http_response.__aenter__ = AsyncMock(return_value=http_response)
        http_response.__aexit__ = AsyncMock()
        mock_session.post.return_value = http_response
        request = self._request("conv-a")
        hooks = MagicMock(spec=ResponseHooks)

        await client.send_request(request, hooks=hooks)

        frontend = zlib.crc32(b"conv-a") % 8
        url = mock_session.post.call_args[0][0]
        assert url.startswith(f"http://localhost:{8000 + 10 * (frontend + 1)}/")
        # Router bookkeeping keeps the routed server's shared address.
        hooks.on_ctx_resp.assert_called_once_with("localhost:8000", response_body)

    @pytest.mark.asyncio
    async def test_refused_frontend_port_falls_back_to_shared_port(self, mock_router, mock_session):
        _reset_prometheus_registry()
        client = OpenAIHttpClient(
            router=mock_router,
            role=ServerRole.CONTEXT,
            max_retries=0,
            session=mock_session,
            frontend_affinity=(8, 10),
        )
        response_body = TestOpenAIHttpClient().dummy_response()
        http_response = AsyncMock()
        http_response.status = 200
        http_response.headers = {"Content-Type": "application/json"}
        http_response.json = AsyncMock(return_value=response_body.model_dump())
        http_response.__aenter__ = AsyncMock(return_value=http_response)
        http_response.__aexit__ = AsyncMock()
        refused = aiohttp.ClientConnectorError(Mock(), OSError(111, "Connection refused"))
        mock_session.post.side_effect = [refused] + [http_response] * 3
        affinity_port = 8000 + 10 * (zlib.crc32(b"conv-a") % 8 + 1)
        now = 100.0

        def ports():
            return [
                int(call.args[0].split("/")[2].rpartition(":")[2])
                for call in mock_session.post.call_args_list
            ]

        with patch(
            "tensorrt_llm.serve.openai_client.get_steady_clock_now_in_seconds",
            side_effect=lambda: now,
        ):
            # The refused attempt is not a retry: it succeeds with max_retries=0.
            await client.send_request(self._request("conv-a"))
            assert ports() == [affinity_port, 8000]
            mock_router.renew_request.assert_not_called()
            # Later turns of the conversation skip the refused port for a while ...
            now += 1.0
            await client.send_request(self._request("conv-a"))
            assert ports()[2:] == [8000]
            # ... and try it again once that window has passed.
            now += 60.0
            await client.send_request(self._request("conv-a"))
            assert ports()[3:] == [affinity_port]

    @pytest.mark.asyncio
    async def test_refused_shared_port_is_not_retried_elsewhere(self, mock_router, mock_session):
        """Without a conversation id the request already uses the shared port."""
        _reset_prometheus_registry()
        client = OpenAIHttpClient(
            router=mock_router,
            role=ServerRole.CONTEXT,
            max_retries=0,
            session=mock_session,
            frontend_affinity=(8, 10),
        )
        refused = aiohttp.ClientConnectorError(Mock(), OSError(111, "Connection refused"))
        mock_session.post.side_effect = [refused]

        with pytest.raises(aiohttp.ClientConnectorError):
            await client.send_request(self._request(None))
        assert mock_session.post.call_count == 1


class TestGenerationLegRetry:
    """A generation leg is retried only while it cannot have reached the server.

    The generation leg names the context server's KV-cache send session through
    ``disagg_request_id``. Once the generation server may have accepted it, the
    KV cache went to that attempt (or was released after a failed transfer), so
    a second post would reuse the id of a request the worker may still hold, or
    wait for KV cache that is never sent. Only a failure to connect, or a
    connection that closed before the whole request was written, is retried. A
    server-side close before the response (``ServerDisconnectedError``) and a
    connection reset (``ClientOSError``) can follow a request the server already
    read; TestGenerationLegRetryOverTransport shows them against a real server.
    """

    _ORIGINAL_ID = 4242

    def _make_client(self, session, server="localhost:8000"):
        _reset_prometheus_registry()
        router = AsyncMock(spec=Router)
        router.servers = [server]
        router.get_next_server = AsyncMock(return_value=(server, None))
        router.finish_request = AsyncMock()
        client = OpenAIHttpClient(
            router=router,
            role=ServerRole.GENERATION,
            timeout_secs=10,
            max_retries=1,
            retry_interval_sec=0,
            session=session,
            disagg_id_generator=AsyncMock(return_value=9000),
        )
        return client, router

    def _make_request(self, stream: bool) -> CompletionRequest:
        return CompletionRequest(
            model="m",
            prompt=[1, 2, 3],
            stream=stream,
            disaggregated_params=DisaggregatedParams(
                request_type="generation_only",
                first_gen_tokens=[7],
                ctx_request_id=self._ORIGINAL_ID,
                disagg_request_id=self._ORIGINAL_ID,
            ),
        )

    @staticmethod
    def _mock_stream(iter_any):
        r = AsyncMock()
        r.status = 200
        r.headers = {"Content-Type": "text/event-stream"}
        r.content = AsyncMock()
        r.content.iter_any = iter_any
        r.__aenter__ = AsyncMock(return_value=r)
        r.__aexit__ = AsyncMock(return_value=False)
        return r

    def _mock_truncated_stream(self):
        """A 200 event stream that is cut before its first chunk."""

        async def iter_any():
            raise aiohttp.ClientPayloadError("Response payload is not completed")
            yield b""  # pragma: no cover - makes this an async generator

        return self._mock_stream(iter_any)

    def _mock_ok_stream(self):
        async def iter_any():
            yield b'data: "ok"\n\n'

        return self._mock_stream(iter_any)

    @pytest.mark.asyncio
    async def test_truncated_stream_is_not_retried(self):
        """The generation server failed the request after sending its headers."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        session.post.side_effect = [self._mock_truncated_stream(), self._mock_ok_stream()]
        client, router = self._make_client(session)
        req = self._make_request(stream=True)

        generator = await client.send_request(req)
        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(aiohttp.ClientPayloadError):
                async for _ in generator:
                    pass

        assert session.post.call_count == 1
        assert req.disaggregated_params.disagg_request_id == self._ORIGINAL_ID
        assert router.finish_request.await_args.kwargs["success"] is False

    @pytest.mark.asyncio
    async def test_http_error_status_is_not_retried(self):
        session = AsyncMock(spec=aiohttp.ClientSession)
        r = AsyncMock()
        r.status = 500
        r.reason = "Internal Server Error"
        r.headers = {"Content-Type": "application/json"}
        r.text = AsyncMock(return_value="Error in kv cache transfer for generation requests")
        r.__aenter__ = AsyncMock(return_value=r)
        r.__aexit__ = AsyncMock(return_value=False)
        session.post.return_value = r
        client, _ = self._make_client(session)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(aiohttp.ClientResponseError):
                await client.send_request(self._make_request(stream=False))

        assert session.post.call_count == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [
            asyncio.TimeoutError(),
            aiohttp.ServerDisconnectedError(),
            aiohttp.ClientOSError(errno.ECONNRESET, "Connection reset by peer"),
        ],
        ids=["timeout", "server_disconnected", "connection_reset"],
    )
    async def test_possibly_delivered_request_is_not_retried(self, error):
        session = AsyncMock(spec=aiohttp.ClientSession)
        session.post.side_effect = error
        client, _ = self._make_client(session)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(type(error)):
                await client.send_request(self._make_request(stream=False))

        assert session.post.call_count == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [
            aiohttp.ClientConnectorError(Mock(), OSError(111, "Connection refused")),
            # Raised while the request is being written; a ConnectionResetError.
            aiohttp.ClientConnectionResetError("Cannot write to closing transport"),
        ],
        ids=["connect", "write_to_closing_transport"],
    )
    async def test_undelivered_request_is_retried_with_same_id(self, error):
        session = AsyncMock(spec=aiohttp.ClientSession)
        session.post.side_effect = [error, self._mock_ok_stream()]
        client, router = self._make_client(session)
        req = self._make_request(stream=True)
        hooks = MagicMock(spec=ResponseHooks)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            generator = await client.send_request(req, hooks=hooks)
            chunks = [chunk async for chunk in generator]

        assert chunks == [b'data: "ok"\n\n']
        assert session.post.call_count == 2
        bodies = [
            msgspec.msgpack.decode(call.kwargs["data"]) for call in session.post.call_args_list
        ]
        assert [body["disaggregated_params"]["disagg_request_id"] for body in bodies] == [
            self._ORIGINAL_ID,
            self._ORIGINAL_ID,
        ]
        client._disagg_id_generator.assert_not_awaited()
        hooks.on_disagg_request_id.assert_not_called()
        assert router.finish_request.await_args.kwargs["success"] is True


async def _read_http_request(reader: asyncio.StreamReader) -> bytes:
    head = await reader.readuntil(b"\r\n\r\n")
    for line in head.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            return await reader.readexactly(int(line.split(b":", 1)[1]))
    raise AssertionError("request without Content-Length")


def _reset(writer: asyncio.StreamWriter) -> None:
    sock = writer.get_extra_info("socket")
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    writer.transport.abort()


class TestGenerationLegRetryOverTransport:
    """The generation-leg retry policy against a local server.

    With aiohttp 3.13 a refused connection raises ``ClientConnectorError``, so
    the request was never delivered and is retried. Once the connection is
    open, a server that closes or resets it without responding surfaces as
    ``ClientOSError`` (ECONNRESET) or ``ServerDisconnectedError`` whether or not
    it read the request first, so none of those is re-posted.
    """

    _ORIGINAL_ID = TestGenerationLegRetry._ORIGINAL_ID
    _make_client = TestGenerationLegRetry._make_client
    _make_request = TestGenerationLegRetry._make_request

    @staticmethod
    def _free_port() -> int:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    @pytest.mark.asyncio
    async def test_refused_connection_is_retried_with_same_id(self):
        port = self._free_port()
        bodies = []
        servers = []

        async def respond(reader, writer):
            bodies.append(msgspec.msgpack.decode(await _read_http_request(reader)))
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                b'Connection: close\r\n\r\ndata: "ok"\n\n'
            )
            await writer.drain()
            writer.close()

        async def start_server(*args, **kwargs):
            # Nothing listens on the port until the first attempt was refused.
            if not servers:
                servers.append(await asyncio.start_server(respond, "127.0.0.1", port))

        async with aiohttp.ClientSession() as session:
            client, router = self._make_client(session, server=f"127.0.0.1:{port}")
            router.renew_request.side_effect = start_server
            try:
                generator = await client.send_request(self._make_request(stream=True))
                chunks = b"".join([chunk async for chunk in generator])
            finally:
                for server in servers:
                    server.close()

        assert b'data: "ok"' in chunks
        assert [body["disaggregated_params"]["disagg_request_id"] for body in bodies] == [
            self._ORIGINAL_ID
        ]
        client._disagg_id_generator.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("reads_request", "abort"),
        [(False, False), (False, True), (True, False), (True, True)],
        ids=["close_before_read", "reset_before_read", "close_after_read", "reset_after_read"],
    )
    async def test_open_connection_closed_without_response_is_not_reposted(
        self, reads_request, abort
    ):
        connections = []
        complete_requests = []

        async def close_without_response(reader, writer):
            connections.append(True)
            if reads_request:
                complete_requests.append(await _read_http_request(reader))
            if abort:
                _reset(writer)
            else:
                writer.close()

        server = await asyncio.start_server(close_without_response, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            async with aiohttp.ClientSession() as session:
                client, router = self._make_client(session, server=f"127.0.0.1:{port}")
                with pytest.raises((aiohttp.ClientOSError, aiohttp.ServerDisconnectedError)):
                    await client.send_request(self._make_request(stream=False))
        finally:
            server.close()

        assert len(connections) == 1
        assert len(complete_requests) == int(reads_request)
        assert router.finish_request.await_args.kwargs["success"] is False
