import json
import logging
import os
import tempfile
from urllib.request import urlopen

import pytest
import yaml

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module", ids=["TinyLlama-1.1B-Chat"])
def model_name():
    return "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"


@pytest.fixture(scope="module")
def temp_extra_llm_api_options_file(request):
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, "extra_llm_api_options.yaml")
    try:
        extra_llm_api_options_dict = {
            "return_perf_metrics": True,
            "perf_metrics_max_requests": 10
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


def test_metrics_endpoint(server: RemoteOpenAIServer):

    client = server.get_client()
    client.completions.create(
        model="Server",
        prompt="Hello, my name is",
        max_tokens=25,
        stream=False,
    )

    response = urlopen(f'{server.url_root}/perf_metrics')
    assert response.status is 200

    data_list = json.loads(response.read())
    assert len(data_list) == 1
    assert "perf_metrics" in data_list[0]
    assert "request_id" in data_list[0]

    data = data_list[0]["perf_metrics"]
    assert "first_iter" in data
    assert "last_iter" in data
    assert data["first_iter"] <= data["last_iter"]

    timing_metrics = data["timing_metrics"]
    assert "arrival_time" in timing_metrics
    assert "first_scheduled_time" in timing_metrics
    assert "first_token_time" in timing_metrics
    assert "last_token_time" in timing_metrics
    assert timing_metrics["arrival_time"] < timing_metrics[
        "first_scheduled_time"]
    assert timing_metrics["first_scheduled_time"] < timing_metrics[
        "first_token_time"]
    assert timing_metrics["first_token_time"] <= timing_metrics[
        "last_token_time"]

    kv_cache_metrics = data["kv_cache_metrics"]
    assert "num_total_allocated_blocks" in kv_cache_metrics
    assert "num_new_allocated_blocks" in kv_cache_metrics
    assert "num_reused_blocks" in kv_cache_metrics
    assert "num_missed_blocks" in kv_cache_metrics
    assert kv_cache_metrics["num_new_allocated_blocks"] <= kv_cache_metrics[
        "num_total_allocated_blocks"]

    # exclude disagg specific metrics
    assert "ctx_request_id" not in data_list[0]
    assert "kv_cache_size" not in timing_metrics
    assert "kv_cache_transfer_start" not in timing_metrics
    assert "kv_cache_transfer_end" not in timing_metrics
