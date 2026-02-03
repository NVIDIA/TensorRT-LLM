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
"""Utilities for Prometheus Metrics Collection."""

import time
from typing import Dict, Union

from .enums import MetricNames


# Adapted from https://github.com/vllm-project/vllm/blob/v0.10.0rc1/vllm/engine/metrics.py#L30
class MetricsCollector:
    """
    Collects and logs metrics from TensorRT-LLM engine stats and request performance metrics to Prometheus.

    Used by OpenAIServer in tensorrt_llm/serve/openai_server.py.

    Args:
        labels: A key-value dictionary of labels to add as metadata to all created Prometheus metrics. Useful for
        distinguishing between multiple series of the same metric name. Example:
        {"model_name": "nemotron-nano-3", "engine_type": "trtllm"}

    Created Prometheus metrics:
        trtllm_request_success_total
        trtllm_e2e_request_latency_seconds
        trtllm_time_to_first_token_seconds
        trtllm_time_per_output_token_seconds
        trtllm_request_queue_time_seconds
        trtllm_kv_cache_hit_rate
        trtllm_kv_cache_utilization
    """
    labelname_finish_reason = "finished_reason"

    def __init__(self, labels: Dict[str, str]) -> None:
        from prometheus_client import Counter, Gauge, Histogram
        self.last_log_time = time.time()
        self.labels = labels
        self.metric_prefix = "trtllm_"

        self.finish_reason_label = {
            MetricsCollector.labelname_finish_reason: "unknown"
        }
        self.labels_with_finished_reason = {
            **self.labels,
            **self.finish_reason_label
        }

        self.counter_request_success = Counter(
            name=self.metric_prefix + "request_success_total",
            documentation="Count of successfully processed requests.",
            labelnames=self.labels_with_finished_reason.keys())

        self.histogram_e2e_time_request = Histogram(
            name=self.metric_prefix + "e2e_request_latency_seconds",
            documentation="Histogram of end to end request latency in seconds.",
            buckets=[
                0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0,
                40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0
            ],
            labelnames=self.labels.keys())

        self.histogram_time_to_first_token = Histogram(
            name=self.metric_prefix + "time_to_first_token_seconds",
            documentation="Histogram of time to first token in seconds.",
            buckets=[
                0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5,
                0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0,
                2560.0
            ],
            labelnames=self.labels.keys())

        self.histogram_time_per_output_token = Histogram(
            name=self.metric_prefix + "time_per_output_token_seconds",
            documentation="Histogram of time per output token in seconds.",
            buckets=[
                0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75,
                1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0
            ],
            labelnames=self.labels.keys())

        self.histogram_queue_time_request = Histogram(
            name=self.metric_prefix + "request_queue_time_seconds",
            documentation=
            "Histogram of time spent in WAITING phase for request.",
            buckets=[
                0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0,
                40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0
            ],
            labelnames=self.labels.keys())

        self.kv_cache_hit_rate = Gauge(name=self.metric_prefix +
                                       "kv_cache_hit_rate",
                                       documentation="KV cache hit rate",
                                       labelnames=self.labels.keys())
        self.kv_cache_utilization = Gauge(name=self.metric_prefix +
                                          "kv_cache_utilization",
                                          documentation="KV cache utilization",
                                          labelnames=self.labels.keys())

    def _label_merge(self, labels: Dict[str, str]) -> Dict[str, str]:
        if labels is None or len(labels) == 0:
            return self.labels
        return {**self.labels, **labels}

    def _log_counter(self, counter, labels: Dict[str, str],
                     data: Union[int, float]) -> None:
        # Convenience function for logging to counter.
        counter.labels(**self._label_merge(labels)).inc(data)

    def _log_histogram(self, histogram, data: Union[int, float]) -> None:
        # Convenience function for logging to histogram.
        histogram.labels(**self.labels).observe(data)

    def _log_gauge(self, gauge, data: Union[int, float]) -> None:
        # Convenience function for logging to gauge.
        gauge.labels(**self.labels).set(data)

    def log_request_metrics_dict(self, metrics_dict: dict[str, float]) -> None:
        """
        Log per-request metrics from TRTLLM engine responses.

        This method updates Prometheus metrics including:
        - counter_request_success
        - histogram_e2e_time_request
        - histogram_time_to_first_token
        - histogram_time_per_output_token
        - histogram_queue_time_request

        Args:
            metrics_dict: A dictionary containing request metrics with the following expected keys:
                - `MetricsCollector.labelname_finish_reason` (str): Finish reason string indicating
                  request completion status.
                - `MetricNames.E2E` (float): End-to-end request latency in seconds.
                - `MetricNames.TTFT` (float): Time to first token in seconds.
                - `MetricNames.TPOT` (float): Time per output token in seconds.
                - `MetricNames.REQUEST_QUEUE_TIME` (float): Request queue time in seconds.

        Returns:
            None: Metrics are logged to Prometheus; nothing is returned.

        Note:
            - Needs to include `return_perf_metrics: true` in LLM args to populate the metrics_dict field
            from the engine responses.
            - Metrics are only recorded when MetricsCollector.labelname_finish_reason is present
            in the metrics_dict, indicating the request has finished.

        """
        if finish_reason := metrics_dict.get(
                MetricsCollector.labelname_finish_reason):
            # If the request finishes, log per-request metrics
            self._log_counter(
                self.counter_request_success,
                {MetricsCollector.labelname_finish_reason: finish_reason}, 1)
            if e2e := metrics_dict.get(MetricNames.E2E, 0):
                self._log_histogram(self.histogram_e2e_time_request, e2e)
            if ttft := metrics_dict.get(MetricNames.TTFT, 0):
                self._log_histogram(self.histogram_time_to_first_token, ttft)
            if tpot := metrics_dict.get(MetricNames.TPOT, 0):
                self._log_histogram(self.histogram_time_per_output_token, tpot)
            if request_queue_time := metrics_dict.get(
                    MetricNames.REQUEST_QUEUE_TIME, 0):
                self._log_histogram(self.histogram_queue_time_request,
                                    request_queue_time)
            self.last_log_time = time.time()

    def log_iteration_stats(self, iteration_stats: dict) -> None:
        """
        Log iteration-level statistics from TRTLLM engine.

        This method updates Prometheus metrics including:
        - kv_cache_hit_rate
        - kv_cache_utilization

        Args:
            iteration_stats: A dictionary containing iteration-level statistics with the following
                expected structure:
                - "kvCacheStats" (dict): KV cache statistics containing:
                    - "cacheHitRate" (float): Cache hit rate (0.0 to 1.0). If present (including zero),
                      the kv_cache_hit_rate gauge is updated.
                    - "usedNumBlocks" (int): Number of KV cache blocks currently in use.
                    - "maxNumBlocks" (int): Maximum number of KV cache blocks available. Should always be
                      non-zero.

        Returns:
            None: Metrics are logged to Prometheus; nothing is returned.

        Note:
            - Needs to include `enable_iter_perf_stats: true` in LLM args to collect iteration-level stats.
            - KV cache utilization is only calculated and logged when both "usedNumBlocks" and
              "maxNumBlocks" are present in kvCacheStats and "maxNumBlocks" is non-zero.
        """
        if kv_stats := iteration_stats.get("kvCacheStats"):
            cache_hit_rate = kv_stats.get("cacheHitRate")
            if cache_hit_rate is not None:
                self._log_gauge(self.kv_cache_hit_rate, cache_hit_rate)
            if "usedNumBlocks" in kv_stats and "maxNumBlocks" in kv_stats:
                max_num_blocks = kv_stats["maxNumBlocks"]
                if max_num_blocks:
                    utilization = kv_stats["usedNumBlocks"] / max_num_blocks
                    self._log_gauge(self.kv_cache_utilization, utilization)
