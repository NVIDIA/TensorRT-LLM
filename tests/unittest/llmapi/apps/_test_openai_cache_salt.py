"""Test cache_salt functionality in OpenAI API to ensure it prevents cache reuse"""

import os
import tempfile

import openai
import pytest
import yaml

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer

pytestmark = pytest.mark.threadleak(enabled=False)


@pytest.fixture(scope="module", ids=["TinyLlama-1.1B-Chat"])
def model_name() -> str:
    return "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"


@pytest.fixture(scope="module")
def temp_extra_llm_api_options_file():
    """Create temporary config file with KV cache enabled for testing"""
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, "cache_salt_test_options.yaml")
    try:
        extra_llm_api_options_dict = {
            # Enable KV cache reuse
            "kv_cache_config": {
                "enable_block_reuse": True,
            },
            # Enable performance metrics to get cache hit rate
            "return_perf_metrics": True,
            "enable_iter_perf_stats": True,
            "enable_iter_req_stats": True,
            # Disable CUDA graph for compatibility
            "cuda_graph_config": None,
        }

        with open(temp_file_path, 'w') as f:
            yaml.dump(extra_llm_api_options_dict, f)

        yield temp_file_path
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@pytest.fixture(scope="module")
def server(model_name: str,
           temp_extra_llm_api_options_file: str) -> RemoteOpenAIServer:
    model_path = get_model_path(model_name)
    args = []
    args.extend(["--backend", "pytorch"])
    args.extend(["--extra_llm_api_options", temp_extra_llm_api_options_file])
    with RemoteOpenAIServer(model_path, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer) -> openai.OpenAI:
    return server.get_client()


def get_cache_hit_rate(client: openai.OpenAI) -> float:
    """Get cache hit rate from the metrics endpoint"""
    import httpx

    # Get the base URL from the OpenAI client (it includes /v1)
    # We need to go up one level to access /metrics
    base_url = str(client.base_url).rstrip('/')
    if base_url.endswith('/v1'):
        base_url = base_url[:-3]  # Remove /v1

    # Make a direct HTTP request to the metrics endpoint
    with httpx.Client() as http_client:
        response = http_client.get(f"{base_url}/metrics", timeout=5.0)

        # Check if metrics endpoint is available
        if response.status_code != 200:
            raise RuntimeError(
                f"Metrics endpoint returned status {response.status_code}")

        metrics = response.json()

        # Validate that we have metrics data
        if not isinstance(metrics, list) or len(metrics) == 0:
            raise ValueError("No metrics data available")

        # Get the most recent stats
        latest_stats = metrics[-1]

        # Extract KV cache statistics
        kv_cache_stats = latest_stats.get("kvCacheStats", {})
        if not kv_cache_stats:
            raise ValueError("No KV cache statistics available in metrics")

        try:
            print(f"kv_cache_stats reused: {kv_cache_stats['reusedBlocks']}")
            print(f"kv_cache_stats missed: {kv_cache_stats['missedBlocks']}")
            print(f"kv_cache_stats hit rate: {kv_cache_stats['cacheHitRate']}")
            return kv_cache_stats["cacheHitRate"]
        except Exception as e:
            print(f"Warning: Could not get cache metrics: {e}")
            return 0.0


def test_cache_salt_prevents_reuse_chat(client: openai.OpenAI, model_name: str):
    """Test that different cache_salt values prevent KV cache reuse in chat completions"""

    # Common messages that will be used across all requests
    messages = [{
        "role": "system",
        "content": "You are a helpful assistant. Keep responses brief."
    }, {
        "role":
        "user",
        "content":
        "What is the capital of France? Answer in one sentence."
    }]

    # Test configuration
    max_tokens = 30
    temperature = 0.0  # Deterministic for testing

    # Track responses for comparison
    responses = []

    # Test Case 1: First request without cache_salt (baseline)
    print("\n=== Test Case 1: First request without cache_salt ===")
    response1 = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    responses.append(response1.choices[0].message.content)
    print(f"Response 1: {response1.choices[0].message.content[:100]}...")

    # Display initial cache metrics
    initial_hit_rate = get_cache_hit_rate(client)
    print(f"Initial cache hit rate: {initial_hit_rate:.2%}")

    # Test Case 2: Same messages without cache_salt (should reuse cache)
    print(
        "\n=== Test Case 2: Same messages without cache_salt (should reuse) ==="
    )
    response2 = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    responses.append(response2.choices[0].message.content)
    print(f"Response 2: {response2.choices[0].message.content[:100]}...")

    # Check if metrics are available
    hit_rate_after_reuse = get_cache_hit_rate(client)
    print(f"Cache hit rate after reuse: {hit_rate_after_reuse:.2%}")
    assert hit_rate_after_reuse >= initial_hit_rate, \
        "Cache hit rate should increase when reusing cache without salt"

    # Test Case 3: Same messages with cache_salt="user_123" (should NOT reuse)
    print(
        "\n=== Test Case 3: Same messages with cache_salt='user_123' (no reuse) ==="
    )
    response3 = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={"cache_salt": "user_123"})
    responses.append(response3.choices[0].message.content)
    print(f"Response 3: {response3.choices[0].message.content[:100]}...")

    # Record metrics after request with different salt
    hit_rate_after_salt1 = get_cache_hit_rate(client)
    print(f"Cache hit rate after salt 'user_123': {hit_rate_after_salt1:.2%}")
    assert hit_rate_after_salt1 < hit_rate_after_reuse, \
        "Cache hit rate should decrease when using a different salt"

    # Test Case 4: Same messages with same cache_salt="user_123" (should reuse)
    print(
        "\n=== Test Case 4: Same messages with same cache_salt='user_123' (should reuse) ==="
    )
    response4 = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={"cache_salt": "user_123"}  # Same salt should enable reuse
    )
    responses.append(response4.choices[0].message.content)
    print(f"Response 4: {response4.choices[0].message.content[:100]}...")

    # Cache hit rate should increase again when using same salt
    hit_rate_after_salt1_reuse = get_cache_hit_rate(client)
    print(
        f"Cache hit rate after reusing salt 'user_123': {hit_rate_after_salt1_reuse:.2%}"
    )
    assert hit_rate_after_salt1_reuse >= hit_rate_after_salt1, \
        "Cache hit rate should increase when reusing same salt"

    # Test Case 5: Same messages with different cache_salt="user_456" (should NOT reuse)
    print(
        "\n=== Test Case 5: Same messages with cache_salt='user_456' (no reuse) ==="
    )
    response5 = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={"cache_salt": "user_456"})
    responses.append(response5.choices[0].message.content)
    print(f"Response 5: {response5.choices[0].message.content[:100]}...")

    # Cache hit rate should decrease when using a different salt
    hit_rate_after_salt2 = get_cache_hit_rate(client)
    print(f"Cache hit rate after salt 'user_456': {hit_rate_after_salt2:.2%}")
    assert hit_rate_after_salt2 < hit_rate_after_salt1_reuse, \
        "Cache hit rate should decrease when using a different salt"

    # Test empty string (should be rejected)
    with pytest.raises(Exception) as exc_info:
        client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            extra_body={"cache_salt": ""}  # Empty string should be rejected
        )
    print(f"Empty string rejected as expected: {exc_info.value}")
