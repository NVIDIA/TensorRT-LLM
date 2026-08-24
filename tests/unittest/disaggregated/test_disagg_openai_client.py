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

import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import aiohttp
import pytest

from tensorrt_llm._utils import AdjustedSteadyClock
from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.serve.disagg_auth import INTERNAL_DISAGG_AUTH_HEADER
from tensorrt_llm.serve.openai_client import OpenAIHttpClient
from tensorrt_llm.serve.openai_protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    DisaggregatedParams,
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
from tensorrt_llm.serve.responses_utils import (ResponseHooks,
                                                 ServerArrivalTimeMiddleware)
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
    with patch(
        "tensorrt_llm.serve.openai_server.get_global_steady_clock_now_in_seconds",
        global_clock,
    ), patch("tensorrt_llm.serve.openai_server.asyncio.sleep", delay):
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

    clock = AdjustedSteadyClock(
        process_offset, time_source=Mock(side_effect=[10.0, 10.25])
    )
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
        r.headers = {"Content-Type": "application/json"}
        r.request_info = MagicMock()
        r.history = ()
        r.__aenter__ = AsyncMock(return_value=r)
        r.__aexit__ = AsyncMock(return_value=False)
        return r

    def _make_client(self, session, **kwargs):
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
            max_retries=0,
            session=session,
            **kwargs,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status,body",
        [
            (400, '{"error":"missing field X"}'),
            (500, "internal failure detail"),
        ],
    )
    async def test_error_body_in_exception(self, status, body):
        session = AsyncMock(spec=aiohttp.ClientSession)
        session.post.return_value = self._mock_http_error(status, body)
        client = self._make_client(session)
        req = CompletionRequest(
            model="m",
            prompt="hi",
            stream=False,
            disaggregated_params=DisaggregatedParams(request_type="context_only", ctx_request_id=1),
        )
        with pytest.raises(aiohttp.ClientResponseError) as exc_info:
            await client.send_request(req)
        assert body[:20] in str(exc_info.value.message)


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

    def _make_client(self, session, **kwargs):
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


class TestSelectiveTransientTcpRetry:
    """Selective retry budget for transient TCP race symptoms.

    ServerDisconnectedError and ConnectionResetError (which include
    aiohttp.ClientConnectionResetError via MRO) get an extended retry budget
    of up to 5 attempts; all other client errors keep the original
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
    async def test_connection_reset_gets_extra_retries(self):
        """ConnectionResetError: same extra budget as ServerDisconnectedError."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        client = self._make_client(session, max_retries=1)

        session.post.side_effect = [
            ConnectionResetError(),
            ConnectionResetError(),
            self._mock_http_ok(self._ok_response()),
        ]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await client.send_request(self._make_request())

        # 1 original + 2 retries = 3 total attempts (within extra budget)
        assert session.post.call_count == 3

    @pytest.mark.asyncio
    async def test_other_client_error_keeps_fail_fast(self):
        """Generic aiohttp.ClientError still respects max_retries (=1)."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        client = self._make_client(session, max_retries=1)

        session.post.side_effect = aiohttp.ClientError("transient non-tcp")

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
    async def test_transient_tcp_capped_at_5_when_max_retries_smaller(self):
        """If transient TCP keeps failing, give up after the extended budget."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        client = self._make_client(session, max_retries=1)

        # Always raise — must give up after extended (1 + 5) = 6 attempts
        session.post.side_effect = aiohttp.ServerDisconnectedError()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(aiohttp.ServerDisconnectedError):
                await client.send_request(self._make_request())

        # 1 original + 5 retries
        assert session.post.call_count == 6
