import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples",
                 "apps"))
from fastapi_server import LLM, KvCacheConfig, LlmServer, SamplingParams

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from test_llm import llama_model_path


@pytest.fixture
def client():
    llm = LLM(llama_model_path)
    sampling_params = SamplingParams()
    kv_cache_config = KvCacheConfig()

    app_instance = LlmServer(llm, sampling_params, kv_cache_config)
    client = TestClient(app_instance.app)
    return client


def test_generate(client):
    response = client.post("/generate", json={"prompt": "A B C"})
    assert response.status_code == 200
    assert "D E F" in response.json()["text"]
    print(response.json())


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
