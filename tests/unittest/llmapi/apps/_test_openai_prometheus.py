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
import re
import tempfile
import time
from typing import Dict
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
    """Return the HuggingFace model path used for all tests in this module."""
    return "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"


@pytest.fixture(scope="module")
def temp_extra_llm_api_options_file(request):
    """Create a temporary YAML file with extra LLM API options for metrics collection."""
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
    """Start a RemoteOpenAIServer with the PyTorch backend and metrics enabled."""
    model_path = get_model_path(model_name)
    args = ["--backend", "pytorch", "--tp_size", "1"]
    args.extend(["--extra_llm_api_options", temp_extra_llm_api_options_file])
    logger.info(f"Starting server, model: {model_name}, args: {args}")
    with RemoteOpenAIServer(model_path, args) as remote_server:
        yield remote_server
        logger.info("Tests completed, shutting down server")


def _parse_prometheus_sample(data: str, metric_name: str) -> float | None:
    """Parse Prometheus exposition text and return the sample value for a metric.

    Matches a single sample line in the Prometheus text format, e.g.:
        trtllm_kv_cache_hit_rate{model_name="llama"} 0.5
        trtllm_kv_cache_hit_rate 0.5
    Comment lines (# HELP, # TYPE) are naturally skipped because they
    do not match the pattern.

    Args:
        data: Raw Prometheus exposition text from the /prometheus/metrics endpoint.
        metric_name: Fully qualified metric name to search for (e.g.
            "trtllm_kv_cache_hit_rate").

    Returns:
        The float value of the first matching sample, or None if not found.
    """

    # '^'  # anchor to start of line
    # r'(?:\{[^}]*\})?' for  optional {label="value",...} block (non-capturing)
    # r'\s+' for whitespace separating metric name from value
    # r'(\S+)' for capture the numeric sample value
    # re.MULTILINE so '^' matches each line start, not just the start of `data`
    pattern = re.compile(
        r'^' + re.escape(metric_name) + r'(?:\{[^}]*\})?' + r'\s+' + r'(\S+)',
        re.MULTILINE)
    match = pattern.search(data)
    return float(match.group(1)) if match else None


def _parse_all_kv_metrics(data: str, prefix: str) -> Dict[str, float | None]:
    """Parse and return sample values for all KV cache Prometheus metrics.

    Args:
        data: Raw Prometheus exposition text.
        prefix: Metric name prefix (e.g. "trtllm_").

    Returns:
        Dict mapping each fully qualified metric name to its parsed sample
        value, or None if that metric was not found in the data.
    """
    names = [
        prefix + "kv_cache_hit_rate",
        prefix + "kv_cache_reused_blocks_total",
        prefix + "kv_cache_missed_blocks_total",
        prefix + "kv_cache_utilization",
    ]
    return {name: _parse_prometheus_sample(data, name) for name in names}


def test_metrics_endpoint(server: RemoteOpenAIServer):
    """Verify that Prometheus metrics are correctly exposed after serving requests.

    Sends two identical completion requests, then polls the /prometheus/metrics
    endpoint until iteration-level KV cache metrics appear. Asserts that:
    - Request-level metrics (success count, latencies) are present.
    - KV cache metrics have sample values (not just HELP/TYPE lines).
    - Post-warmup values are correct: the second identical request reuses
      1 block and misses 1 block, yielding a 0.5 hit rate.
    """
    METRIC_PREFIX = "trtllm_"

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

        # Check if iteration stats metrics have sample values
        kv_metrics = _parse_all_kv_metrics(data, METRIC_PREFIX)
        if all(v is not None for v in kv_metrics.values()):
            hit_rate = kv_metrics[METRIC_PREFIX + "kv_cache_hit_rate"]
            if hit_rate > 0.0:
                # Wait until we have some kv cache reuse to check on iteration stats
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
    assert METRIC_PREFIX + "request_success_total" in data
    assert METRIC_PREFIX + "e2e_request_latency_seconds" in data
    assert METRIC_PREFIX + "time_to_first_token_seconds" in data
    assert METRIC_PREFIX + "request_queue_time_seconds" in data

    # Assert iteration stats metrics are eventually available
    assert iteration_stats_metrics_found, \
        f"Iteration stats metrics not found or cache hit rate is always 0 after waiting {max_wait_time}s"
    kv_metrics = _parse_all_kv_metrics(data, METRIC_PREFIX)
    for name, value in kv_metrics.items():
        assert value is not None, f"No sample value found for {name}"

    # Verify post-warmup values match expected behavior:
    # Two identical requests → 1 reused block, 1 missed block, 0.5 hit rate
    hit_rate = kv_metrics[METRIC_PREFIX + "kv_cache_hit_rate"]
    reused = kv_metrics[METRIC_PREFIX + "kv_cache_reused_blocks_total"]
    missed = kv_metrics[METRIC_PREFIX + "kv_cache_missed_blocks_total"]
    utilization = kv_metrics[METRIC_PREFIX + "kv_cache_utilization"]

    assert hit_rate == pytest.approx(0.5), \
        f"Expected kv_cache_hit_rate == 0.5, got {hit_rate}"
    assert reused == 1.0, \
        f"Expected kv_cache_reused_blocks_total == 1.0, got {reused}"
    assert missed == 1.0, \
        f"Expected kv_cache_missed_blocks_total == 1.0, got {missed}"
    assert utilization >= 0, \
        f"Expected kv_cache_utilization >= 0, got {utilization}"
