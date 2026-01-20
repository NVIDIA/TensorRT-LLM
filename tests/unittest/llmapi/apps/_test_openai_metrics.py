"""Test the metrics endpoint when using OpenAI API to send requests"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from tensorrt_llm import LLM as PyTorchLLM
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.serve.openai_server import OpenAIServer

from ..test_llm import llama_model_path

pytestmark = pytest.mark.threadleak(enabled=False)


@pytest.fixture(scope="module")
def llm():
    llm = PyTorchLLM(model=llama_model_path,
                     kv_cache_config=KvCacheConfig(),
                     enable_iter_perf_stats=True)
    yield llm
    llm.shutdown()


@pytest.fixture(scope="module")
def client(llm):
    app_instance = OpenAIServer(llm,
                                model=llama_model_path,
                                tool_parser=None,
                                server_role=None,
                                metadata_server_cfg=None)
    client = TestClient(app_instance.app)
    yield client


@pytest.mark.parametrize("is_healthy,response_code", [(True, 200),
                                                      (False, 503)])
def test_health(client, llm, is_healthy, response_code):
    if not is_healthy:
        with patch.object(llm._executor, 'is_shutdown', return_value=True):
            response = client.get("/health")
            assert response.status_code == response_code
    else:
        response = client.get("/health")
        assert response.status_code == response_code


def test_version(client):
    response = client.get("/version")
    assert response.status_code == 200


def test_metrics(client):
    response = client.post("/v1/completions",
                           json={
                               "prompt": "A B C",
                               "model": llama_model_path,
                               "max_tokens": 10
                           })
    assert response.status_code == 200
    assert "D E F" in response.json()["choices"][0]["text"]
    response = client.get("/metrics")
    assert response.status_code == 200
    response_dict = response.json()[0]
    assert "cpuMemUsage" in response_dict
    assert "gpuMemUsage" in response_dict
    assert "inflightBatchingStats" in response_dict
    assert "numContextRequests" in response_dict["inflightBatchingStats"]
    assert "numCtxTokens" in response_dict["inflightBatchingStats"]
    assert "numGenRequests" in response_dict["inflightBatchingStats"]
    assert "numPausedRequests" in response_dict["inflightBatchingStats"]
    assert "numScheduledRequests" in response_dict["inflightBatchingStats"]
    assert "iter" in response_dict
    assert "iterLatencyMS" in response_dict
    assert "kvCacheStats" in response_dict
    assert "allocNewBlocks" in response_dict["kvCacheStats"]
    assert "allocTotalBlocks" in response_dict["kvCacheStats"]
    assert "cacheHitRate" in response_dict["kvCacheStats"]
    assert "freeNumBlocks" in response_dict["kvCacheStats"]
    assert "maxNumBlocks" in response_dict["kvCacheStats"]
    assert "missedBlocks" in response_dict["kvCacheStats"]
    assert "reusedBlocks" in response_dict["kvCacheStats"]
    assert "tokensPerBlock" in response_dict["kvCacheStats"]
    assert "usedNumBlocks" in response_dict["kvCacheStats"]
    assert "maxBatchSizeRuntime" in response_dict
    assert "maxBatchSizeStatic" in response_dict
    assert "maxBatchSizeTunerRecommended" in response_dict
    assert "maxNumActiveRequests" in response_dict
    assert "maxNumTokensRuntime" in response_dict
    assert "maxNumTokensStatic" in response_dict
    assert "maxNumTokensTunerRecommended" in response_dict
    assert "newActiveRequestsQueueLatencyMS" in response_dict
    assert "numActiveRequests" in response_dict
    assert "numCompletedRequests" in response_dict
    assert "numNewActiveRequests" in response_dict
    assert "numQueuedRequests" in response_dict
    assert "pinnedMemUsage" in response_dict
    assert "staticBatchingStats" in response_dict
    assert "timestamp" in response_dict
