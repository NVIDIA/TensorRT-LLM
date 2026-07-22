import logging
import os
import tempfile

import pytest
import requests
import yaml
from test_common.perf_metrics_utils import wait_for_perf_metrics_jsonl

from tensorrt_llm.serve import perf_metrics

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module", ids=["TinyLlama-1.1B-Chat"])
def model_name():
    return "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"


@pytest.fixture(scope="module")
def perf_metrics_output_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("perf_metrics")


@pytest.fixture(scope="module")
def temp_extra_llm_api_options_file(perf_metrics_output_dir):
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, "extra_llm_api_options.yaml")
    try:
        extra_llm_api_options_dict = {
            "return_perf_metrics": True,
            "perf_metrics_output_dir": str(perf_metrics_output_dir),
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
    args = ["--backend", "pytorch", "--tp_size", "1"]
    args.extend(["--extra_llm_api_options", temp_extra_llm_api_options_file])
    logger.info(f"Starting server, model: {model_name}, args: {args}")
    with RemoteOpenAIServer(model_path, args) as remote_server:
        yield remote_server
        logger.info("Tests completed, shutting down server")


def test_return_perf_metrics_and_jsonl_dump(server: RemoteOpenAIServer,
                                            perf_metrics_output_dir):
    response = requests.post(
        f"{server.url_root}/v1/completions",
        headers={perf_metrics.RETURN_METRICS_HEADER: "1"},
        json={
            "model": "Server",
            "prompt": "Hello, my name is",
            "max_tokens": 2,
        },
        timeout=120,
    )
    assert response.status_code == 200

    assert response.headers.get(perf_metrics.SERVER_TIMING_HEADER)
    assert response.headers.get(perf_metrics.START_END_TIME_HEADER)
    assert response.headers.get(perf_metrics.STEP_METRICS_HEADER)
    assert response.headers.get(perf_metrics.CTX_CHUNK_METRICS_HEADER)

    records = wait_for_perf_metrics_jsonl(perf_metrics_output_dir,
                                          expected_count=1)
    data = records[-1]
    assert data["status"] == "complete"
    assert set(data) == {
        "request_id",
        "perf_metrics",
        "time_breakdown_metrics",
        "status",
    }

    request_metrics = data["perf_metrics"]
    assert request_metrics["first_iter"] <= request_metrics["last_iter"]

    timing_metrics = request_metrics["timing_metrics"]
    assert timing_metrics["arrival_time"] < timing_metrics[
        "first_scheduled_time"]
    assert timing_metrics["first_scheduled_time"] < timing_metrics[
        "first_token_time"]
    assert timing_metrics["first_token_time"] <= timing_metrics[
        "last_token_time"]

    kv_cache_metrics = request_metrics["kv_cache_metrics"]
    assert kv_cache_metrics["num_new_allocated_blocks"] <= kv_cache_metrics[
        "num_total_allocated_blocks"]

    assert "ctx_request_id" not in data
    assert "kv_cache_transfer_start" not in timing_metrics
    assert "kv_cache_transfer_end" not in timing_metrics

    response = requests.post(
        f"{server.url_root}/v1/completions",
        json={
            "model": "Server",
            "prompt": "Hello, my name is",
            "max_tokens": 2,
        },
        timeout=120,
    )
    assert response.status_code == 200
    assert not response.headers.get(perf_metrics.SERVER_TIMING_HEADER)
    assert not response.headers.get(perf_metrics.START_END_TIME_HEADER)
    assert not response.headers.get(perf_metrics.STEP_METRICS_HEADER)
    assert not response.headers.get(perf_metrics.CTX_CHUNK_METRICS_HEADER)


def test_streaming_metrics_require_request_opt_in(server: RemoteOpenAIServer):
    payload = {
        "model": "Server",
        "prompt": "Hello, my name is",
        "max_tokens": 2,
        "stream": True,
    }
    response = requests.post(f"{server.url_root}/v1/completions",
                             json=payload,
                             timeout=120)
    assert response.status_code == 200
    assert "data: [DONE]" in response.text
    assert f"event: {perf_metrics.SSE_METRICS_EVENT}" not in response.text

    response = requests.post(
        f"{server.url_root}/v1/completions",
        headers={perf_metrics.RETURN_METRICS_HEADER: "1"},
        json=payload,
        timeout=120,
    )
    assert response.status_code == 200
    assert "data: [DONE]" in response.text
    assert f"event: {perf_metrics.SSE_METRICS_EVENT}" in response.text
