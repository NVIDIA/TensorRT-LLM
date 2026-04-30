# Copyright (c) 2025, NVIDIA CORPORATION.
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

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import aiohttp
import pytest

from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.serve.openai_client import OpenAIHttpClient
from tensorrt_llm.serve.openai_protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    DisaggregatedParams,
    UsageInfo,
)
from tensorrt_llm.serve.router import Router


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
    from prometheus_client.registry import REGISTRY

    REGISTRY._names_to_collectors = {}
    REGISTRY._collector_to_names = {}
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
        mock_router.finish_request.assert_called_once_with(completion_request, mock_session)

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
            streaming_completion_request, mock_session
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

        # Should finish request on error
        mock_router.finish_request.assert_called_once_with(completion_request, mock_session)

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
        client = self._make_client(session, disagg_id_generator=lambda: next(ids))

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
