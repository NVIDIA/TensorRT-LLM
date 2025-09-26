import concurrent.futures
import time

import pytest
from apps.fastapi_server import LLM, BuildConfig, LlmServer
from fastapi.testclient import TestClient

import tensorrt_llm.profiler as profiler

from ..test_llm import llama_model_path

pytestmark = pytest.mark.threadleak(enabled=False)


@pytest.fixture(scope="module")
def client():
    build_config = BuildConfig()
    build_config.max_batch_size = 8
    build_config.max_seq_len = 512
    llm = LLM(llama_model_path, build_config=build_config)

    app_instance = LlmServer(llm)
    client = TestClient(app_instance.app)
    yield client
    llm.shutdown()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_generate(client):
    response = client.post("/generate", json={"prompt": "A B C"})
    assert response.status_code == 200
    assert "D E F" in response.json()["text"]
    print(response.json())


def test_generate_with_sampling(client):
    response_topk_1 = client.post("/generate",
                                  json={
                                      "prompt": "In this example,",
                                      "top_k": 1
                                  })
    assert response_topk_1.status_code == 200
    response_topk_3 = client.post("/generate",
                                  json={
                                      "prompt": "In this example,",
                                      "top_k": 3
                                  })
    assert response_topk_3.status_code == 200
    print(response_topk_1.json())
    print(response_topk_3.json())


def test_generate_streaming(client):
    with client.stream("POST",
                       "/generate",
                       json={
                           "prompt": "A B C",
                           "streaming": True
                       }) as response:
        assert response.status_code == 200
        chunks = []
        for chunk in response.iter_text():
            chunks.append(chunk)

        whole_text = "".join(chunks)
        assert "D E F" in whole_text


def make_concurrent_requests(client, num_requests):
    """make concurrent requests"""

    def single_request():
        try:
            response = client.post("/generate",
                                   json={
                                       "prompt": "In this example,",
                                       "max_tokens": 2048,
                                       "beam_width": 5,
                                       "temperature": 0,
                                       "repetition_penalty": 1.15,
                                       "presence_penalty": 2,
                                       "frequency_penalty": 2
                                   })
            assert response.status_code == 200
            return response.json()
        except Exception as e:
            print(f"Request failed: {str(e)}")
            return None

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_requests) as executor:
        responses = [None] * num_requests
        future_to_request = {
            executor.submit(single_request): i
            for i in range(num_requests)
        }
        for future in concurrent.futures.as_completed(future_to_request):
            i = future_to_request[future]
            try:
                responses[i] = future.result()
            except Exception as e:
                print(f"Request {i} failed: {str(e)}")
                responses[i] = None
        print(responses)
    return responses


def test_concurrent_requests_memory_leak(client):
    """test memory leak under concurrent requests"""
    num_requests = 10
    num_iterations = 2
    memory_threshold = 1  # GB
    memory_usages = []
    try:
        # multiple rounds of concurrent requests test
        for i in range(num_iterations):
            print(f"\nIteration {i+1}:")
            profiler.print_memory_usage(f'make concurrent requests {i} started')
            current_memory, _, _ = profiler.host_memory_info()
            responses = make_concurrent_requests(client, num_requests)
            assert len(responses) == num_requests
            time.sleep(2)
            profiler.print_memory_usage(f'make concurrent requests {i} ended')
            current_memory, _, _ = profiler.host_memory_info()
            memory_usages.append(current_memory)

        first_round_memory = memory_usages[0] / (1024**3)
        final_memory = memory_usages[-1] / (1024**3)
        memory_diff = final_memory - first_round_memory
        assert memory_diff < memory_threshold, f"Memory leak detected: {memory_diff:.2f} GB increase between first and last round"

    finally:
        import gc
        gc.collect()
