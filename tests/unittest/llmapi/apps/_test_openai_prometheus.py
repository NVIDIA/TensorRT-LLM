# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
import tempfile
import time
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
            "enable_iter_perf_stats": True
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
    metric_prefix = "trtllm_"

    client = server.get_client()
    # Need to send at least 2 requests to get iteration stats computed
    # so kv_cache Prometheus metrics are available.
    for _ in range(2):
        client.completions.create(
            model="Server",
            prompt="Hello, my name is",
            max_tokens=25,
            stream=False,
        )

    # Wait for background iteration stats collector task to process and log metrics
    # The _iteration_stats_collector_loop runs asynchronously, so we poll until
    # the iteration stats metrics (kv_cache_hit_rate, kv_cache_utilization) appear
    max_wait_time = 10.0  # seconds
    poll_interval = 0.5  # seconds
    start_time = time.time()

    iteration_stats_metrics_found = False
    while time.time() - start_time < max_wait_time:
        response = urlopen(f'{server.url_root}/prometheus/metrics')
        assert response.status == 200

        data = response.read().decode("utf-8")

        # Check if iteration stats metrics are present
        if (metric_prefix + "kv_cache_hit_rate" in data
                and metric_prefix + "kv_cache_utilization" in data):
            iteration_stats_metrics_found = True
            break

        logger.info(
            f"Iteration stats not yet available, waiting {poll_interval}s...")
        time.sleep(poll_interval)

    # Final check: fetch metrics one more time for assertions
    response = urlopen(f'{server.url_root}/prometheus/metrics')
    assert response.status == 200
    data = response.read().decode("utf-8")

    # Assert request-level metrics (these should be available immediately)
    assert metric_prefix + "request_success_total" in data
    assert metric_prefix + "e2e_request_latency_seconds" in data
    assert metric_prefix + "time_to_first_token_seconds" in data
    assert metric_prefix + "request_queue_time_seconds" in data

    # Assert iteration stats metrics (collected by background task)
    assert iteration_stats_metrics_found, \
        f"Iteration stats metrics not found after waiting {max_wait_time}s"
    assert metric_prefix + "kv_cache_hit_rate" in data
    assert metric_prefix + "kv_cache_utilization" in data
