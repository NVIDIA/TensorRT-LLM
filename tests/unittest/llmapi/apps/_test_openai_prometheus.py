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
        extra_llm_api_options_dict = {"return_perf_metrics": True}

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

    response = urlopen(f'{server.url_root}/prometheus/metrics')
    assert response.status is 200

    data = response.read().decode("utf-8")
    assert "request_success_total" in data
    assert "e2e_request_latency_seconds" in data
    assert "time_to_first_token_seconds" in data
    assert "request_queue_time_seconds" in data
