### :title Prometheus Metrics

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from urllib.request import urlopen

from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="tensorrt_llm",
)

# Prometheus metric prefix used by TensorRT-LLM
METRIC_PREFIX = "trtllm_"

# Base URL for the metrics endpoint
METRICS_URL = "http://localhost:8000/prometheus/metrics"


def fetch_metrics() -> str | None:
    """Fetch raw Prometheus exposition text from the metrics endpoint.

    Returns:
        The decoded response body as a string, or None if the request
        failed or returned a non-200 status.
    """
    try:
        response = urlopen(METRICS_URL)
        if response.status == 200:
            return response.read().decode("utf-8")
        else:
            print(f"Error fetching metrics: HTTP {response.status}")
            return None
    except Exception as e:
        print(f"Error fetching metrics: {e}")
        return None


def parse_and_display_metrics(metrics_data: str) -> None:
    """Parse Prometheus exposition text and print TensorRT-LLM metrics.

    Searches the raw text for a predefined set of metrics (request counts,
    latency histograms, KV cache stats). Found metrics are printed with
    their sample lines; missing metrics are listed separately.

    Args:
        metrics_data: Raw Prometheus exposition text returned by fetch_metrics().
    """
    if not metrics_data:
        return

    print("\n" + "=" * 80)
    print("TensorRT-LLM Prometheus Metrics")
    print("=" * 80)

    # Define metrics to display with descriptions
    metrics_of_interest = {
        f"{METRIC_PREFIX}request_success_total": "Total successful requests",
        f"{METRIC_PREFIX}e2e_request_latency_seconds": "End-to-end request latency",
        f"{METRIC_PREFIX}time_to_first_token_seconds": "Time to first token",
        f"{METRIC_PREFIX}request_queue_time_seconds": "Request queue time",
        f"{METRIC_PREFIX}kv_cache_hit_rate": "KV cache hit rate",
        f"{METRIC_PREFIX}kv_cache_reused_blocks_total": "KV cache reused blocks (cumulative)",
        f"{METRIC_PREFIX}kv_cache_missed_blocks_total": "KV cache missed blocks (cumulative)",
        f"{METRIC_PREFIX}kv_cache_utilization": "KV cache utilization",
    }

    found_metrics = []
    missing_metrics = []

    for metric_name, description in metrics_of_interest.items():
        if metric_name in metrics_data:
            found_metrics.append((metric_name, description))
        else:
            missing_metrics.append((metric_name, description))

    # Display found metrics
    if found_metrics:
        print("\n✓ Available Metrics:")
        print("-" * 80)
        for metric_name, description in found_metrics:
            # Extract the metric lines from the data
            lines = [
                line
                for line in metrics_data.split("\n")
                if line.startswith(metric_name) and not line.startswith("#")
            ]
            print(f"\n{description} ({metric_name}):")
            for line in lines:
                print(f"  {line}")

    # Display missing metrics
    if missing_metrics:
        print("\n✗ Not Yet Available:")
        print("-" * 80)
        for metric_name, description in missing_metrics:
            print(f"  {description} ({metric_name})")

    print("\n" + "=" * 80)


def main():
    """Send completion requests to a running TensorRT-LLM server and display Prometheus metrics.

    Sends 10 completion requests sequentially, fetching and printing
    the Prometheus metrics after each response to show how counters, histograms,
    and gauges evolve over time.
    """
    print("Prometheus Metrics Example")
    print("=" * 80)
    print("This script will:")
    print("1. Send 10 completion requests to a running TensorRT-LLM server")
    print(
        "2. After each response, fetch and display Prometheus metrics from the /prometheus/metrics endpoint"
    )
    print()

    # Make several completion requests to generate metrics
    print("Sending completion requests...")
    NUM_REQUESTS = 10
    for i in range(NUM_REQUESTS):
        try:
            response = client.completions.create(
                model="Server",
                prompt=(
                    f"Hello, this is request {i + 1}. "
                    "Use your greatest imagination in this request. Tell me a lot about"
                ),
                max_tokens=1000,
                stream=False,
            )
            print(
                f"  Request {i + 1}/{NUM_REQUESTS} completed. Response: {response.choices[0].text[:50]}..."
            )

            # Fetch and display metrics after each response
            print(f"\n  Fetching metrics after request {i + 1}...")
            metrics_data = fetch_metrics()
            if metrics_data:
                parse_and_display_metrics(metrics_data)
            else:
                print("  ✗ Failed to fetch metrics")
            print()
        except Exception as e:
            print(f"  Error on request {i + 1}: {e}")
    print("All requests completed.")


if __name__ == "__main__":
    main()
