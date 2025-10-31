"""
End-to-End Negative Tests for trtllm-serve

These tests verify that trtllm-serve handles error conditions gracefully:
- Invalid inputs and malformed requests
- Server stability under stress with invalid requests
- Proper error responses and status codes
- Recovery after encountering errors
"""

import asyncio
import socket
import time
from pathlib import Path

import openai
import pytest
import requests
from defs.conftest import llm_models_root
from defs.trt_test_alternative import popen, print_error, print_info


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class RemoteOpenAIServer:
    DUMMY_API_KEY = "tensorrt_llm"

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port

    @property
    def url_root(self) -> str:
        return f"http://{self.host}:{self.port}"

    def url_for(self, *parts: str) -> str:
        return self.url_root + "/" + "/".join(parts)

    def get_async_client(self, **kwargs):
        return openai.AsyncOpenAI(base_url=self.url_for("v1"),
                                  api_key=self.DUMMY_API_KEY,
                                  **kwargs)


@pytest.fixture(scope="module")
def model_name():
    """Use TinyLlama for faster testing"""
    return "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"


@pytest.fixture(scope="module")
def model_path(model_name):
    """Get the full model path"""
    return str(Path(llm_models_root()) / model_name)


@pytest.fixture(scope="module")
def server(model_path):
    """Start a test server for the module using popen like test_serve.py"""
    host_bind = "0.0.0.0"
    client_host = "localhost"
    port = _find_free_port()
    cmd = [
        "trtllm-serve",
        "serve",
        model_path,
        "--host",
        host_bind,
        "--port",
        str(port),
        "--backend",
        "pytorch",
    ]

    def _wait_until_ready(timeout_secs: int = 600, interval: float = 0.5):
        start = time.time()
        health_url = f"http://{client_host}:{port}/health"
        while True:
            try:
                if requests.get(health_url, timeout=2).status_code == 200:
                    break
            except Exception:
                pass
            if time.time() - start > timeout_secs:
                raise TimeoutError("Error: trtllm-serve health check timed out")
            time.sleep(interval)

    print_info("Launching trtllm-serve (negative tests)...")
    with popen(cmd):
        _wait_until_ready()
        yield RemoteOpenAIServer(client_host, port)


@pytest.fixture
def async_client(server: RemoteOpenAIServer):
    """Get an async OpenAI client"""
    return server.get_async_client()


@pytest.mark.asyncio
async def test_invalid_max_tokens(async_client: openai.AsyncOpenAI,
                                  model_name: str):
    """Test that server rejects invalid max_tokens value."""
    print_info("Testing invalid max_tokens parameter: 0")

    with pytest.raises(openai.BadRequestError) as exc_info:
        await async_client.chat.completions.create(
            model=model_name,
            messages=[{
                "role": "user",
                "content": "Hello"
            }],
            max_tokens=0,
        )
    error_msg = str(exc_info.value).lower()
    assert any(keyword in error_msg for keyword in ("mMaxNewTokens", "failed"))


@pytest.mark.asyncio
async def test_invalid_temperature(async_client: openai.AsyncOpenAI,
                                   model_name: str):
    """Test that server rejects invalid temperature value."""
    print_info("Testing invalid temperature parameter: -0.5")

    with pytest.raises(openai.BadRequestError) as exc_info:
        await async_client.chat.completions.create(
            model=model_name,
            messages=[{
                "role": "user",
                "content": "Hello"
            }],
            temperature=-0.5,
        )
    assert "temperature" in str(exc_info.value).lower() or "invalid" in str(
        exc_info.value).lower()


@pytest.mark.parametrize("top_p_value", [-0.1, 1.1])
@pytest.mark.asyncio
async def test_invalid_top_p(async_client: openai.AsyncOpenAI, model_name: str,
                             top_p_value: float):
    """Test that server rejects invalid top_p values."""
    print_info(f"Testing invalid top_p parameter: {top_p_value}")

    with pytest.raises(openai.BadRequestError) as exc_info:
        await async_client.chat.completions.create(
            model=model_name,
            messages=[{
                "role": "user",
                "content": "Hello"
            }],
            top_p=top_p_value,
        )
    assert "top_p" in str(exc_info.value).lower() or "invalid" in str(
        exc_info.value).lower()


@pytest.mark.asyncio
async def test_empty_messages_array(async_client: openai.AsyncOpenAI,
                                    model_name: str):
    """Test that server rejects empty messages array."""
    print_info("Testing empty messages array...")

    with pytest.raises(openai.BadRequestError) as exc_info:
        await async_client.chat.completions.create(model=model_name,
                                                   messages=[],
                                                   max_tokens=10)
    assert "message" in str(exc_info.value).lower() or "empty" in str(
        exc_info.value).lower()


@pytest.mark.asyncio
async def test_missing_message_role(async_client: openai.AsyncOpenAI,
                                    model_name: str):
    """Test that server rejects messages without role field."""
    print_info("Testing missing message role...")

    with pytest.raises(openai.BadRequestError) as exc_info:
        await async_client.chat.completions.create(
            model=model_name,
            messages=[{
                "content": "Hello"
            }],  # Missing 'role'
            max_tokens=10)
    assert "role" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_invalid_token_ids(async_client: openai.AsyncOpenAI,
                                 model_name: str):
    """Test that server handles invalid token IDs in prompt."""
    print_info("Testing invalid token IDs...")

    # Test negative token ID
    with pytest.raises((openai.BadRequestError, openai.APIError)) as exc_info:
        await async_client.completions.create(
            model=model_name,
            prompt=[1, 2, 3, -1, 5],  # Invalid token ID: -1
            max_tokens=5)
    error_msg = str(exc_info.value).lower()
    assert "token" in error_msg or "invalid" in error_msg or "range" in error_msg


@pytest.mark.asyncio
async def test_extremely_large_token_id(async_client: openai.AsyncOpenAI,
                                        model_name: str):
    """Test that server handles token IDs exceeding vocabulary size."""
    print_info("Testing extremely large token ID...")

    # Test token ID beyond typical vocabulary size
    with pytest.raises((openai.BadRequestError, openai.APIError)) as exc_info:
        await async_client.completions.create(
            model=model_name,
            prompt=[1, 2, 3, 999999],  # Token ID far beyond vocab size
            max_tokens=5)
    error_msg = str(exc_info.value).lower()
    assert "token" in error_msg or "range" in error_msg or "vocabulary" in error_msg or "vocab" in error_msg


@pytest.mark.asyncio
async def test_server_stability_under_invalid_requests(
        server: RemoteOpenAIServer, model_name: str):
    """
    E2E Test: Verify server remains stable after receiving many invalid requests

    Test flow:
    1. Send valid request to verify server is working
    2. Flood server with invalid requests
    3. Send valid request to verify server still works
    4. Check health endpoint
    """
    print_info("Testing server stability under invalid requests...")

    async_client = server.get_async_client()

    # Step 1: Verify server is working with valid request
    response = await async_client.chat.completions.create(model=model_name,
                                                          messages=[{
                                                              "role":
                                                              "user",
                                                              "content":
                                                              "Hello"
                                                          }],
                                                          max_tokens=5)
    assert response is not None
    assert len(response.choices) > 0
    print_info("Initial valid request succeeded")

    # Step 2: Send multiple invalid requests
    invalid_request_types = [
        # Empty messages
        {
            "messages": [],
            "max_tokens": 5
        },
        # Missing role
        {
            "messages": [{
                "content": "test"
            }],
            "max_tokens": 5
        },
        # Invalid temperature
        {
            "messages": [{
                "role": "user",
                "content": "test"
            }],
            "temperature": -1
        },
        # Invalid max_tokens
        {
            "messages": [{
                "role": "user",
                "content": "test"
            }],
            "max_tokens": -10
        },
        # Invalid top_p
        {
            "messages": [{
                "role": "user",
                "content": "test"
            }],
            "top_p": 2.0
        },
    ]

    error_count = 0
    for _ in range(20):  # Send 100 total invalid requests (20 x 5 types)
        for invalid_params in invalid_request_types:
            try:
                await async_client.chat.completions.create(model=model_name,
                                                           **invalid_params)
            except (openai.BadRequestError, openai.APIError):
                error_count += 1
            except Exception as e:
                # Unexpected error - server might be unstable
                pytest.fail(f"Unexpected error during invalid request: {e}")

    print_info(
        f"Sent {error_count} invalid requests, all rejected as expected.")

    # Step 3: Verify server still works with valid request
    response = await async_client.chat.completions.create(model=model_name,
                                                          messages=[{
                                                              "role":
                                                              "user",
                                                              "content":
                                                              "Hello again"
                                                          }],
                                                          max_tokens=5)
    assert response is not None
    assert len(response.choices) > 0
    print_info("Server still responsive after invalid requests.")

    # Step 4: Check health endpoint
    health_url = server.url_for("health")
    health_response = requests.get(health_url)
    assert health_response.status_code == 200
    print_info("Health check passed.")


@pytest.mark.asyncio
async def test_concurrent_invalid_requests(server: RemoteOpenAIServer,
                                           model_name: str):
    """
    E2E Test: Multiple concurrent invalid requests should not crash server

    Simulates multiple clients sending invalid requests simultaneously
    """
    print_info("Testing concurrent invalid requests...")

    async_client = server.get_async_client()

    # Create 50 concurrent invalid requests
    tasks = []
    for i in range(50):
        # Alternate between different types of invalid requests
        if i % 3 == 0:
            task = async_client.chat.completions.create(
                model=model_name,
                messages=[],  # Empty messages
                max_tokens=5)
        elif i % 3 == 1:
            task = async_client.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": "test"
                }],
                temperature=-1  # Invalid temperature
            )
        else:
            task = async_client.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": "test"
                }],
                max_tokens=-5  # Invalid max_tokens
            )
        tasks.append(task)

    # Execute all concurrently and gather results
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # All should be BadRequestError or APIError
    for i, result in enumerate(results):
        assert isinstance(result, (openai.BadRequestError, openai.APIError)), \
            f"Request {i} should have failed with BadRequestError or APIError, got: {type(result)}"

    print_info(
        f"All {len(results)} concurrent invalid requests rejected properly")

    # Verify server still works
    response = await async_client.chat.completions.create(model=model_name,
                                                          messages=[{
                                                              "role":
                                                              "user",
                                                              "content":
                                                              "Final check"
                                                          }],
                                                          max_tokens=5)
    assert response is not None


@pytest.mark.asyncio
async def test_mixed_valid_invalid_requests(server: RemoteOpenAIServer,
                                            model_name: str):
    """
    E2E Test: Mix of valid and invalid requests - server should handle both correctly

    Simulates real-world scenario where some clients send bad requests
    """
    print_info("Testing mixed valid and invalid requests...")

    async_client = server.get_async_client()

    async def send_request(request_id: int) -> dict:
        """Send either valid or invalid request based on request_id"""
        result = {
            "id": request_id,
            "success": False,
            "expected_error": False,
            "unexpected_error": False
        }

        try:
            if request_id % 4 == 0:
                # Send invalid request (25% of requests)
                await async_client.chat.completions.create(
                    model=model_name,
                    messages=[{
                        "role": "user",
                        "content": "test"
                    }],
                    temperature=-1  # Invalid
                )
                result["unexpected_error"] = True  # Shouldn't succeed
            else:
                # Send valid request (75% of requests)
                response = await async_client.chat.completions.create(
                    model=model_name,
                    messages=[{
                        "role": "user",
                        "content": f"Request {request_id}"
                    }],
                    max_tokens=5,
                    temperature=0.5)
                if response and len(response.choices) > 0:
                    result["success"] = True
        except openai.BadRequestError:
            result["expected_error"] = True
        except Exception as e:
            print_error(f"Request {request_id} unexpected error: {e}")
            result["unexpected_error"] = True

        return result

    # Send 100 mixed requests
    tasks = [send_request(i) for i in range(100)]
    results = await asyncio.gather(*tasks)

    # Analyze results
    successful = sum(1 for r in results if r["success"])
    expected_errors = sum(1 for r in results if r["expected_error"])
    unexpected_errors = sum(1 for r in results if r["unexpected_error"])

    print_info(
        f"Results: {successful} successful, {expected_errors} expected errors, {unexpected_errors} unexpected errors"
    )

    # Assertions
    assert successful > 0, "Some valid requests should have succeeded"
    assert expected_errors > 0, "Some invalid requests should have been caught"
    assert unexpected_errors == 0, "No unexpected errors should occur"

    # Roughly 75% should succeed, 25% should fail
    assert successful >= 60, f"Expected ~75 successful requests, got {successful}"
    assert expected_errors >= 20, f"Expected ~25 failed requests, got {expected_errors}"


@pytest.mark.asyncio
async def test_health_check_during_errors(server: RemoteOpenAIServer,
                                          model_name: str):
    """
    E2E Test: Health endpoints should remain functional even when receiving invalid requests
    """
    print_info("Testing health check during error conditions...")

    async def send_invalid_requests():
        """Background task sending invalid requests"""
        async_client = server.get_async_client()
        for _ in range(50):
            try:
                await async_client.chat.completions.create(
                    model=model_name,
                    messages=[],  # Invalid
                    max_tokens=5)
            except:
                pass  # Expected to fail
            await asyncio.sleep(0.05)

    # Start background task sending invalid requests
    background_task = asyncio.create_task(send_invalid_requests())

    # Meanwhile, check health endpoints repeatedly
    health_url = server.url_for("health")
    health_checks_passed = 0

    for _ in range(20):
        await asyncio.sleep(0.1)
        try:
            health_response = requests.get(health_url, timeout=2)
            if health_response.status_code == 200:
                health_checks_passed += 1
        except Exception as e:
            pytest.fail(f"Health check failed during error conditions: {e}")

    # Wait for background task to complete
    await background_task

    assert health_checks_passed == 20, f"All health checks should pass, got {health_checks_passed}/20"


@pytest.mark.asyncio
async def test_request_exceeds_context_length(async_client: openai.AsyncOpenAI,
                                              model_name: str):
    """Test handling of requests exceeding model's max context length"""
    print_info("Testing request exceeding context length...")

    # Generate extremely long prompt (> max_seq_len for TinyLlama)
    # TinyLlama has max_position_embeddings of 2048
    very_long_prompt = "word " * 3000  # ~15000 characters, way over limit

    # Server should either reject or handle gracefully without crashing
    try:
        response = await async_client.chat.completions.create(
            model=model_name,
            messages=[{
                "role": "user",
                "content": very_long_prompt
            }],
            max_tokens=10)
        # If it succeeds, verify response is valid
        assert response is not None
        print_info("Server handled oversized request gracefully")
    except (openai.BadRequestError, openai.APIError) as e:
        # Also acceptable - server rejected the request
        assert "length" in str(e).lower() or "token" in str(
            e).lower() or "context" in str(e).lower()


def test_malformed_json_request(server: RemoteOpenAIServer):
    """Test that server rejects malformed JSON in HTTP requests"""
    print_info("Testing malformed JSON request...")

    chat_url = server.url_for("v1", "chat", "completions")

    # Send invalid JSON
    response = requests.post(
        chat_url,
        headers={"Content-Type": "application/json"},
        data="{invalid json syntax here}",
    )

    # Should return 400 Bad Request
    assert response.status_code == 400


def test_missing_content_type_header(server: RemoteOpenAIServer,
                                     model_name: str):
    """Test server behavior with missing Content-Type header"""
    print_info("Testing missing Content-Type header...")

    chat_url = server.url_for("v1", "chat", "completions")

    # Send request without Content-Type header
    import json
    payload = {
        "model": model_name,
        "messages": [{
            "role": "user",
            "content": "Hello"
        }],
        "max_tokens": 5
    }

    response = requests.post(
        chat_url,
        data=json.dumps(payload),
        # No Content-Type header
    )

    # Server might accept it or reject it - either way it shouldn't crash
    assert response.status_code in [
        200, 400, 415
    ]  # Success, Bad Request, or Unsupported Media Type


@pytest.mark.asyncio
async def test_extremely_large_batch(async_client: openai.AsyncOpenAI,
                                     model_name: str):
    """Test handling of extremely large batch requests for completions"""
    print_info("Testing extremely large batch request...")

    # Try to send batch with many prompts
    large_batch = ["Hello"] * 1000  # 1000 prompts

    try:
        # This should either process or reject gracefully
        response = await async_client.completions.create(model=model_name,
                                                         prompt=large_batch,
                                                         max_tokens=1)
        # If successful, verify we got results
        assert response is not None
        assert hasattr(response, "choices") and len(response.choices) > 0
        print_info("Server processed large batch.")
    except (openai.BadRequestError, openai.APIError) as e:
        # Server rejected - also acceptable
        assert "batch" in str(e).lower() or "too many" in str(
            e).lower() or "limit" in str(e).lower()
