import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples",
                 "apps"))
from fastapi_server import LLM, KvCacheConfig, LlmServer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from test_llm import llama_model_path


@pytest.fixture(scope="module")
def client():
    llm = LLM(llama_model_path)
    kv_cache_config = KvCacheConfig()

    app_instance = LlmServer(llm, kv_cache_config)
    client = TestClient(app_instance.app)
    yield client

    del llm
    del app_instance.llm


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
