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
from typing import Dict, List, Optional, Union

from .enums import MetricNames


# Adapted from https://github.com/vllm-project/vllm/blob/v0.10.0rc1/vllm/engine/metrics.py#L30
class MetricsCollector:
    """Collects and logs metrics from TensorRT-LLM engine stats to Prometheus.

    Used by OpenAIServer in tensorrt_llm/serve/openai_server.py.

    Args:
        labels: A key-value dictionary of labels to add as metadata to all created Prometheus metrics. Useful for
        distinguishing between multiple series of the same metric name. Example:
        {"model_name": "nemotron-nano-3", "engine_type": "trtllm"}

    Created Prometheus metrics:
        Per-request metrics:
            trtllm_request_success_total
            trtllm_e2e_request_latency_seconds
            trtllm_time_to_first_token_seconds
            trtllm_time_per_output_token_seconds
            trtllm_request_queue_time_seconds
            trtllm_request_prefill_time_seconds
            trtllm_request_decode_time_seconds
            trtllm_request_inference_time_seconds
            trtllm_prompt_tokens_total
            trtllm_generation_tokens_total

        Iteration-level metrics:
            trtllm_kv_cache_hit_rate
            trtllm_kv_cache_utilization
            trtllm_kv_cache_host_utilization
            trtllm_kv_cache_iter_reuse_rate
            trtllm_kv_cache_reused_blocks_total
            trtllm_kv_cache_missed_blocks_total
            trtllm_kv_cache_iter_reused_blocks_total
            trtllm_kv_cache_iter_full_reused_blocks_total
            trtllm_kv_cache_iter_partial_reused_blocks_total
            trtllm_kv_cache_iter_missed_blocks_total
            trtllm_kv_cache_gen_alloc_blocks_total
            trtllm_kv_cache_onboard_bytes_total
            trtllm_kv_cache_offload_bytes_total
            trtllm_kv_cache_intra_device_copy_bytes_total
            trtllm_num_requests_running
            trtllm_num_requests_waiting
            trtllm_num_requests_completed_total
            trtllm_max_num_active_requests
            trtllm_iteration_latency_seconds
            trtllm_gpu_memory_usage_bytes
            trtllm_cpu_memory_usage_bytes
            trtllm_pinned_memory_usage_bytes
            trtllm_max_batch_size_static
            trtllm_max_batch_size_runtime
            trtllm_max_num_tokens_runtime
            trtllm_kv_cache_max_blocks
            trtllm_kv_cache_free_blocks
            trtllm_kv_cache_used_blocks
            trtllm_kv_cache_tokens_per_block
            trtllm_num_context_requests
            trtllm_num_generation_requests
            trtllm_num_paused_requests
            trtllm_num_scheduled_requests
            trtllm_total_context_tokens
            trtllm_avg_decoded_tokens_per_iter
            trtllm_spec_decode_num_draft_tokens_total
            trtllm_spec_decode_num_accepted_tokens_total
            trtllm_spec_decode_acceptance_length
            trtllm_spec_decode_draft_overhead

        Config info metrics (logged once at startup via log_config_info):
            trtllm_model_config_info
            trtllm_parallel_config_info
            trtllm_speculative_config_info
            trtllm_kv_cache_config_info
    """
    labelname_finish_reason = "finished_reason"

    def __init__(
        self,
        labels: Dict[str, str],
        e2e_request_latency_buckets: Optional[List[float]] = None,
        time_to_first_token_buckets: Optional[List[float]] = None,
        time_per_output_token_buckets: Optional[List[float]] = None,
        request_queue_time_buckets: Optional[List[float]] = None,
        request_prefill_time_buckets: Optional[List[float]] = None,
        request_decode_time_buckets: Optional[List[float]] = None,
        request_inference_time_buckets: Optional[List[float]] = None,
    ) -> None:
        from prometheus_client import Counter, Gauge, Histogram
        _bucket_params = {
            "e2e_request_latency_buckets": e2e_request_latency_buckets,
            "time_to_first_token_buckets": time_to_first_token_buckets,
            "time_per_output_token_buckets": time_per_output_token_buckets,
            "request_queue_time_buckets": request_queue_time_buckets,
            "request_prefill_time_buckets": request_prefill_time_buckets,
            "request_decode_time_buckets": request_decode_time_buckets,
            "request_inference_time_buckets": request_inference_time_buckets,
        }
        for name, buckets in _bucket_params.items():
            if buckets is None:
                continue
            if len(buckets) == 0:
                raise ValueError(f"{name} must not be empty when provided.")
            if any(a >= b for a, b in zip(buckets, buckets[1:])):
                raise ValueError(
                    f"{name} must be strictly increasing, got {buckets}.")
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
            buckets=e2e_request_latency_buckets or [
                0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0,
                40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0
            ],
            labelnames=self.labels.keys())

        self.histogram_time_to_first_token = Histogram(
            name=self.metric_prefix + "time_to_first_token_seconds",
            documentation="Histogram of time to first token in seconds.",
            buckets=time_to_first_token_buckets or [
                0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5,
                0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0,
                2560.0
            ],
            labelnames=self.labels.keys())

        self.histogram_time_per_output_token = Histogram(
            name=self.metric_prefix + "time_per_output_token_seconds",
            documentation="Histogram of time per output token in seconds.",
            buckets=time_per_output_token_buckets or [
                0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75,
                1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0
            ],
            labelnames=self.labels.keys())

        self.histogram_queue_time_request = Histogram(
            name=self.metric_prefix + "request_queue_time_seconds",
            documentation=
            "Histogram of time spent in WAITING phase for request.",
            buckets=request_queue_time_buckets or [
                0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0,
                40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0
            ],
            labelnames=self.labels.keys())

        self.histogram_prefill_time_request = Histogram(
            name=self.metric_prefix + "request_prefill_time_seconds",
            documentation=
            "Histogram of prefill (context) phase duration in seconds "
            "(first_token_time - first_scheduled_time).",
            buckets=request_prefill_time_buckets or [
                0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
                10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0
            ],
            labelnames=self.labels.keys())

        self.histogram_decode_time_request = Histogram(
            name=self.metric_prefix + "request_decode_time_seconds",
            documentation=
            "Histogram of decode (generation) phase duration in seconds "
            "(last_token_time - first_token_time).",
            buckets=request_decode_time_buckets or [
                0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0,
                40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0
            ],
            labelnames=self.labels.keys())

        self.histogram_inference_time_request = Histogram(
            name=self.metric_prefix + "request_inference_time_seconds",
            documentation="Histogram of total inference duration in seconds "
            "(last_token_time - first_scheduled_time).",
            buckets=request_inference_time_buckets or [
                0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0,
                40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0
            ],
            labelnames=self.labels.keys())

        self.counter_prompt_tokens = Counter(
            name=self.metric_prefix + "prompt_tokens_total",
            documentation=
            "Cumulative number of prompt (input) tokens processed.",
            labelnames=self.labels.keys())

        self.counter_generation_tokens = Counter(
            name=self.metric_prefix + "generation_tokens_total",
            documentation=
            "Cumulative number of generation (output) tokens produced.",
            labelnames=self.labels.keys())

        self.kv_cache_hit_rate = Gauge(name=self.metric_prefix +
                                       "kv_cache_hit_rate",
                                       documentation="KV cache hit rate",
                                       labelnames=self.labels.keys())
        self.kv_cache_reused_blocks = Counter(
            name=self.metric_prefix + "kv_cache_reused_blocks",
            documentation=
            "Cumulative number of KV cache blocks reused (cache hits)",
            labelnames=self.labels.keys())
        self.kv_cache_missed_blocks = Counter(
            name=self.metric_prefix + "kv_cache_missed_blocks",
            documentation=
            "Cumulative number of KV cache blocks missed (cache misses)",
            labelnames=self.labels.keys())
        self.kv_cache_utilization = Gauge(name=self.metric_prefix +
                                          "kv_cache_utilization",
                                          documentation="KV cache utilization",
                                          labelnames=self.labels.keys())
        self._prev_reused_blocks = 0
        self._prev_missed_blocks = 0

        # Per-iteration KV cache gauges
        self.kv_cache_host_utilization = Gauge(
            name=self.metric_prefix + "kv_cache_host_utilization",
            documentation="KV cache host (secondary) pool utilization",
            labelnames=self.labels.keys())
        self.kv_cache_iter_reuse_rate = Gauge(
            name=self.metric_prefix + "kv_cache_iter_reuse_rate",
            documentation="Per-iteration KV cache block reuse rate",
            labelnames=self.labels.keys())

        # Per-iteration KV cache counters (monotonically increasing via accumulated deltas)
        self.kv_cache_iter_reused_blocks = Counter(
            name=self.metric_prefix + "kv_cache_iter_reused_blocks",
            documentation="Total reused KV cache blocks (full + partial)",
            labelnames=self.labels.keys())
        self.kv_cache_iter_full_reused_blocks = Counter(
            name=self.metric_prefix + "kv_cache_iter_full_reused_blocks",
            documentation="Total fully reused KV cache blocks",
            labelnames=self.labels.keys())
        self.kv_cache_iter_partial_reused_blocks = Counter(
            name=self.metric_prefix + "kv_cache_iter_partial_reused_blocks",
            documentation="Total partially reused KV cache blocks",
            labelnames=self.labels.keys())
        self.kv_cache_iter_missed_blocks = Counter(
            name=self.metric_prefix + "kv_cache_iter_missed_blocks",
            documentation="Total missed KV cache blocks (context phase)",
            labelnames=self.labels.keys())
        self.kv_cache_gen_alloc_blocks_total = Counter(
            name=self.metric_prefix + "kv_cache_gen_alloc_blocks_total",
            documentation="Total blocks allocated during generation phase",
            labelnames=self.labels.keys())
        self.kv_cache_onboard_bytes_total = Counter(
            name=self.metric_prefix + "kv_cache_onboard_bytes_total",
            documentation="Total bytes transferred from host to GPU (onboard)",
            labelnames=self.labels.keys())
        self.kv_cache_offload_bytes_total = Counter(
            name=self.metric_prefix + "kv_cache_offload_bytes_total",
            documentation="Total bytes transferred from GPU to host (offload)",
            labelnames=self.labels.keys())
        self.kv_cache_intra_device_copy_bytes_total = Counter(
            name=self.metric_prefix + "kv_cache_intra_device_copy_bytes_total",
            documentation=
            "Total bytes copied within GPU (intra-device block copies)",
            labelnames=self.labels.keys())

        # Queue & load metrics
        self.num_requests_running = Gauge(
            name=self.metric_prefix + "num_requests_running",
            documentation="Number of active requests",
            labelnames=self.labels.keys())
        self.num_requests_waiting = Gauge(
            name=self.metric_prefix + "num_requests_waiting",
            documentation="Number of queued requests",
            labelnames=self.labels.keys())
        self.counter_num_requests_completed = Counter(
            name=self.metric_prefix + "num_requests_completed_total",
            documentation="Total number of completed requests across iterations",
            labelnames=self.labels.keys())
        self.max_num_active_requests = Gauge(
            name=self.metric_prefix + "max_num_active_requests",
            documentation="Maximum number of active requests",
            labelnames=self.labels.keys())

        # Iteration latency
        self.iteration_latency_seconds = Gauge(
            name=self.metric_prefix + "iteration_latency_seconds",
            documentation="Iteration latency in seconds",
            labelnames=self.labels.keys())

        # Memory usage
        self.gpu_memory_usage_bytes = Gauge(
            name=self.metric_prefix + "gpu_memory_usage_bytes",
            documentation="GPU memory usage in bytes",
            labelnames=self.labels.keys())
        self.cpu_memory_usage_bytes = Gauge(
            name=self.metric_prefix + "cpu_memory_usage_bytes",
            documentation="CPU memory usage in bytes",
            labelnames=self.labels.keys())
        self.pinned_memory_usage_bytes = Gauge(
            name=self.metric_prefix + "pinned_memory_usage_bytes",
            documentation="Pinned memory usage in bytes",
            labelnames=self.labels.keys())

        # Batch size
        self.max_batch_size_static = Gauge(
            name=self.metric_prefix + "max_batch_size_static",
            documentation="Static maximum batch size",
            labelnames=self.labels.keys())
        self.max_batch_size_runtime = Gauge(
            name=self.metric_prefix + "max_batch_size_runtime",
            documentation="Runtime maximum batch size",
            labelnames=self.labels.keys())
        self.max_num_tokens_runtime = Gauge(
            name=self.metric_prefix + "max_num_tokens_runtime",
            documentation="Runtime maximum number of tokens",
            labelnames=self.labels.keys())

        # KV cache block metrics
        self.kv_cache_max_blocks = Gauge(
            name=self.metric_prefix + "kv_cache_max_blocks",
            documentation="Maximum number of KV cache blocks",
            labelnames=self.labels.keys())
        self.kv_cache_free_blocks = Gauge(
            name=self.metric_prefix + "kv_cache_free_blocks",
            documentation="Number of free KV cache blocks",
            labelnames=self.labels.keys())
        self.kv_cache_used_blocks = Gauge(
            name=self.metric_prefix + "kv_cache_used_blocks",
            documentation="Number of used KV cache blocks",
            labelnames=self.labels.keys())
        self.kv_cache_tokens_per_block = Gauge(
            name=self.metric_prefix + "kv_cache_tokens_per_block",
            documentation="Number of tokens per KV cache block",
            labelnames=self.labels.keys())

        # Inflight batching metrics
        self.num_context_requests = Gauge(
            name=self.metric_prefix + "num_context_requests",
            documentation="Number of context (prefill) requests",
            labelnames=self.labels.keys())
        self.num_generation_requests = Gauge(
            name=self.metric_prefix + "num_generation_requests",
            documentation="Number of generation (decode) requests",
            labelnames=self.labels.keys())
        self.num_paused_requests = Gauge(
            name=self.metric_prefix + "num_paused_requests",
            documentation="Number of paused requests",
            labelnames=self.labels.keys())
        self.num_scheduled_requests = Gauge(
            name=self.metric_prefix + "num_scheduled_requests",
            documentation="Number of scheduled requests",
            labelnames=self.labels.keys())
        self.total_context_tokens = Gauge(
            name=self.metric_prefix + "total_context_tokens",
            documentation="Total number of context tokens",
            labelnames=self.labels.keys())
        self.avg_decoded_tokens_per_iter = Gauge(
            name=self.metric_prefix + "avg_decoded_tokens_per_iter",
            documentation="Average number of decoded tokens per iteration",
            labelnames=self.labels.keys())

        # Speculative decoding metrics
        self.counter_spec_decode_num_draft_tokens = Counter(
            name=self.metric_prefix + "spec_decode_num_draft_tokens_total",
            documentation="Total number of draft tokens in speculative decoding",
            labelnames=self.labels.keys())
        self.counter_spec_decode_num_accepted_tokens = Counter(
            name=self.metric_prefix + "spec_decode_num_accepted_tokens_total",
            documentation=
            "Total number of accepted tokens in speculative decoding",
            labelnames=self.labels.keys())
        self.spec_decode_acceptance_length = Gauge(
            name=self.metric_prefix + "spec_decode_acceptance_length",
            documentation="Acceptance length in speculative decoding",
            labelnames=self.labels.keys())
        self.spec_decode_draft_overhead = Gauge(
            name=self.metric_prefix + "spec_decode_draft_overhead",
            documentation="Draft overhead in speculative decoding",
            labelnames=self.labels.keys())

    def log_config_info(
            self,
            model_config: Optional[Dict[str, str]] = None,
            parallel_config: Optional[Dict[str, str]] = None,
            speculative_config: Optional[Dict[str, str]] = None,
            kv_cache_config: Optional[Dict[str, str]] = None) -> None:
        """Log static configuration as Prometheus info-style gauges (set to 1 with config labels).

        Should be called once at startup. Each config dict's keys become Prometheus labels.
        Follows the same pattern as vLLM/SGLang config info metrics.

        Args:
            model_config: Model configuration labels (model, dtype, quantization, gpu_type, etc.)
            parallel_config: Parallelism configuration labels (tp_size, pp_size, etc.)
            speculative_config: Speculative decoding configuration labels (method, draft_model, etc.)
            kv_cache_config: KV cache configuration labels (page_size, enable_block_reuse, etc.)
        """
        from prometheus_client import Gauge

        if model_config:
            info_labels = {**self.labels, **model_config}
            gauge = Gauge(name=self.metric_prefix + "model_config_info",
                          documentation="Model configuration info",
                          labelnames=info_labels.keys())
            gauge.labels(**info_labels).set(1)

        if parallel_config:
            info_labels = {**self.labels, **parallel_config}
            gauge = Gauge(name=self.metric_prefix + "parallel_config_info",
                          documentation="Parallelism configuration info",
                          labelnames=info_labels.keys())
            gauge.labels(**info_labels).set(1)

        if speculative_config:
            info_labels = {**self.labels, **speculative_config}
            gauge = Gauge(
                name=self.metric_prefix + "speculative_config_info",
                documentation="Speculative decoding configuration info",
                labelnames=info_labels.keys())
            gauge.labels(**info_labels).set(1)

        if kv_cache_config:
            info_labels = {**self.labels, **kv_cache_config}
            gauge = Gauge(name=self.metric_prefix + "kv_cache_config_info",
                          documentation="KV cache configuration info",
                          labelnames=info_labels.keys())
            gauge.labels(**info_labels).set(1)

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
        """Log per-request metrics from TRTLLM engine responses.

        This method updates Prometheus metrics including:
        - counter_request_success
        - histogram_e2e_time_request
        - histogram_time_to_first_token
        - histogram_time_per_output_token
        - histogram_queue_time_request
        - histogram_prefill_time_request
        - histogram_decode_time_request
        - histogram_inference_time_request
        - counter_prompt_tokens
        - counter_generation_tokens

        Args:
            metrics_dict: A dictionary containing request metrics with the following expected keys:
                - `MetricsCollector.labelname_finish_reason` (str): Finish reason string indicating
                  request completion status.
                - `MetricNames.E2E` (float): End-to-end request latency in seconds.
                - `MetricNames.TTFT` (float): Time to first token in seconds.
                - `MetricNames.TPOT` (float): Time per output token in seconds.
                - `MetricNames.REQUEST_QUEUE_TIME` (float): Request queue time in seconds.
                - `MetricNames.PREFILL_TIME` (float): Prefill phase duration in seconds.
                - `MetricNames.DECODE_TIME` (float): Decode phase duration in seconds.
                - `MetricNames.INFERENCE_TIME` (float): Total inference duration in seconds.
                - `MetricNames.PROMPT_TOKENS` (int): Number of input tokens.
                - `MetricNames.GENERATION_TOKENS` (int): Number of output tokens.

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
            if (request_queue_time := metrics_dict.get(
                    MetricNames.REQUEST_QUEUE_TIME)) is not None:
                self._log_histogram(self.histogram_queue_time_request,
                                    request_queue_time)
            if prefill_time := metrics_dict.get(MetricNames.PREFILL_TIME, 0):
                self._log_histogram(self.histogram_prefill_time_request,
                                    prefill_time)
            if decode_time := metrics_dict.get(MetricNames.DECODE_TIME, 0):
                self._log_histogram(self.histogram_decode_time_request,
                                    decode_time)
            if inference_time := metrics_dict.get(MetricNames.INFERENCE_TIME,
                                                  0):
                self._log_histogram(self.histogram_inference_time_request,
                                    inference_time)
            if prompt_tokens := metrics_dict.get(MetricNames.PROMPT_TOKENS, 0):
                self._log_counter(self.counter_prompt_tokens, {}, prompt_tokens)
            if generation_tokens := metrics_dict.get(
                    MetricNames.GENERATION_TOKENS, 0):
                self._log_counter(self.counter_generation_tokens, {},
                                  generation_tokens)
            self.last_log_time = time.time()

    def log_iteration_stats(self, iteration_stats: dict) -> None:
        """Log iteration-level statistics from TRTLLM engine.

        Updates Prometheus gauges/counters for queue load, memory usage, batch sizes,
        KV cache blocks, inflight batching, and speculative decoding stats.

        Args:
            iteration_stats: A JSON dict returned from `BaseLLM.get_stats()` containing iteration-level statistics.
                Top-level fields: numActiveRequests, numQueuedRequests, numCompletedRequests,
                maxNumActiveRequests, iterLatencyMS, gpuMemUsage, cpuMemUsage, pinnedMemUsage,
                maxBatchSizeStatic, maxBatchSizeRuntime, maxNumTokensRuntime.
                Nested dicts: kvCacheStats, inflightBatchingStats, specDecodingStats.

        Note:
            - Needs `enable_iter_perf_stats: true` in LLM args to collect iteration-level stats.
            - inflightBatchingStats and specDecodingStats are only present when applicable.
        """
        # Top-level queue & load metrics
        if "numActiveRequests" in iteration_stats:
            self._log_gauge(self.num_requests_running,
                            iteration_stats["numActiveRequests"])
        if "numQueuedRequests" in iteration_stats:
            self._log_gauge(self.num_requests_waiting,
                            iteration_stats["numQueuedRequests"])
        if "numCompletedRequests" in iteration_stats:
            completed = iteration_stats["numCompletedRequests"]
            if completed > 0:
                self._log_counter(self.counter_num_requests_completed, {},
                                  completed)
        if "maxNumActiveRequests" in iteration_stats:
            self._log_gauge(self.max_num_active_requests,
                            iteration_stats["maxNumActiveRequests"])

        # Iteration latency (convert ms to seconds)
        if "iterLatencyMS" in iteration_stats:
            self._log_gauge(self.iteration_latency_seconds,
                            iteration_stats["iterLatencyMS"] / 1000.0)

        # Memory usage
        if "gpuMemUsage" in iteration_stats:
            self._log_gauge(self.gpu_memory_usage_bytes,
                            iteration_stats["gpuMemUsage"])
        if "cpuMemUsage" in iteration_stats:
            self._log_gauge(self.cpu_memory_usage_bytes,
                            iteration_stats["cpuMemUsage"])
        if "pinnedMemUsage" in iteration_stats:
            self._log_gauge(self.pinned_memory_usage_bytes,
                            iteration_stats["pinnedMemUsage"])

        # Batch size
        if "maxBatchSizeStatic" in iteration_stats:
            self._log_gauge(self.max_batch_size_static,
                            iteration_stats["maxBatchSizeStatic"])
        if "maxBatchSizeRuntime" in iteration_stats:
            self._log_gauge(self.max_batch_size_runtime,
                            iteration_stats["maxBatchSizeRuntime"])
        if "maxNumTokensRuntime" in iteration_stats:
            self._log_gauge(self.max_num_tokens_runtime,
                            iteration_stats["maxNumTokensRuntime"])

        # KV cache stats
        if kv_stats := iteration_stats.get("kvCacheStats"):
            cache_hit_rate = kv_stats.get("cacheHitRate")
            if cache_hit_rate is not None:
                self._log_gauge(self.kv_cache_hit_rate, cache_hit_rate)
            reused_blocks = kv_stats.get("reusedBlocks")
            if reused_blocks is not None:
                delta = reused_blocks - self._prev_reused_blocks
                if delta > 0:
                    self._log_counter(self.kv_cache_reused_blocks, None, delta)
                self._prev_reused_blocks = reused_blocks
            missed_blocks = kv_stats.get("missedBlocks")
            if missed_blocks is not None:
                delta = missed_blocks - self._prev_missed_blocks
                if delta > 0:
                    self._log_counter(self.kv_cache_missed_blocks, None, delta)
                self._prev_missed_blocks = missed_blocks
            if "usedNumBlocks" in kv_stats and "maxNumBlocks" in kv_stats:
                max_num_blocks = kv_stats["maxNumBlocks"]
                if max_num_blocks:
                    utilization = kv_stats["usedNumBlocks"] / max_num_blocks
                    self._log_gauge(self.kv_cache_utilization, utilization)
            if "maxNumBlocks" in kv_stats:
                self._log_gauge(self.kv_cache_max_blocks,
                                kv_stats["maxNumBlocks"])
            if "freeNumBlocks" in kv_stats:
                self._log_gauge(self.kv_cache_free_blocks,
                                kv_stats["freeNumBlocks"])
            if "usedNumBlocks" in kv_stats:
                self._log_gauge(self.kv_cache_used_blocks,
                                kv_stats["usedNumBlocks"])
            if "tokensPerBlock" in kv_stats:
                self._log_gauge(self.kv_cache_tokens_per_block,
                                kv_stats["tokensPerBlock"])

        # Inflight batching stats
        if ifb_stats := iteration_stats.get("inflightBatchingStats"):
            if "numContextRequests" in ifb_stats:
                self._log_gauge(self.num_context_requests,
                                ifb_stats["numContextRequests"])
            if "numGenRequests" in ifb_stats:
                self._log_gauge(self.num_generation_requests,
                                ifb_stats["numGenRequests"])
            if "numPausedRequests" in ifb_stats:
                self._log_gauge(self.num_paused_requests,
                                ifb_stats["numPausedRequests"])
            if "numScheduledRequests" in ifb_stats:
                self._log_gauge(self.num_scheduled_requests,
                                ifb_stats["numScheduledRequests"])
            if "numCtxTokens" in ifb_stats:
                self._log_gauge(self.total_context_tokens,
                                ifb_stats["numCtxTokens"])
            if "avgNumDecodedTokensPerIter" in ifb_stats:
                self._log_gauge(self.avg_decoded_tokens_per_iter,
                                ifb_stats["avgNumDecodedTokensPerIter"])

        # Speculative decoding stats
        if spec_stats := iteration_stats.get("specDecodingStats"):
            if "numDraftTokens" in spec_stats:
                draft_tokens = spec_stats["numDraftTokens"]
                if draft_tokens > 0:
                    self._log_counter(self.counter_spec_decode_num_draft_tokens,
                                      {}, draft_tokens)
            if "numAcceptedTokens" in spec_stats:
                accepted_tokens = spec_stats["numAcceptedTokens"]
                if accepted_tokens > 0:
                    self._log_counter(
                        self.counter_spec_decode_num_accepted_tokens, {},
                        accepted_tokens)
            if "acceptanceLength" in spec_stats:
                self._log_gauge(self.spec_decode_acceptance_length,
                                spec_stats["acceptanceLength"])
            if "draftOverhead" in spec_stats:
                self._log_gauge(self.spec_decode_draft_overhead,
                                spec_stats["draftOverhead"])

        # Per-iteration KV cache stats (aggregated across window sizes)
        if kv_iter := iteration_stats.get("kvCacheIterationStats"):
            # Aggregate across all window sizes
            total_secondary_max = 0
            total_secondary_used = 0
            total_reused = 0
            total_full_reused = 0
            total_partial_reused = 0
            total_missed = 0
            total_gen_alloc = 0
            total_onboard_bytes = 0
            total_offload_bytes = 0
            total_intra_device_copy_bytes = 0

            for ws_stats in kv_iter.values():
                total_secondary_max += ws_stats.get("secondaryMaxNumBlocks", 0)
                total_secondary_used += ws_stats.get("secondaryUsedNumBlocks",
                                                     0)
                total_reused += ws_stats.get("iterReusedBlocks", 0)
                total_full_reused += ws_stats.get("iterFullReusedBlocks", 0)
                total_partial_reused += ws_stats.get("iterPartialReusedBlocks",
                                                     0)
                total_missed += ws_stats.get("iterMissedBlocks", 0)
                total_gen_alloc += ws_stats.get("iterGenAllocBlocks", 0)
                total_onboard_bytes += ws_stats.get("iterOnboardBytes", 0)
                total_offload_bytes += ws_stats.get("iterOffloadBytes", 0)
                total_intra_device_copy_bytes += ws_stats.get(
                    "iterIntraDeviceCopyBytes", 0)

            # Gauges
            if total_secondary_max > 0:
                self._log_gauge(self.kv_cache_host_utilization,
                                total_secondary_used / total_secondary_max)
            iter_total = total_reused + total_missed
            if iter_total > 0:
                self._log_gauge(self.kv_cache_iter_reuse_rate,
                                total_reused / iter_total)

            # Counters (increment by delta)
            if total_reused > 0:
                self._log_counter(self.kv_cache_iter_reused_blocks, {},
                                  total_reused)
            if total_full_reused > 0:
                self._log_counter(self.kv_cache_iter_full_reused_blocks, {},
                                  total_full_reused)
            if total_partial_reused > 0:
                self._log_counter(self.kv_cache_iter_partial_reused_blocks, {},
                                  total_partial_reused)
            if total_missed > 0:
                self._log_counter(self.kv_cache_iter_missed_blocks, {},
                                  total_missed)
            if total_gen_alloc > 0:
                self._log_counter(self.kv_cache_gen_alloc_blocks_total, {},
                                  total_gen_alloc)
            if total_onboard_bytes > 0:
                self._log_counter(self.kv_cache_onboard_bytes_total, {},
                                  total_onboard_bytes)
            if total_offload_bytes > 0:
                self._log_counter(self.kv_cache_offload_bytes_total, {},
                                  total_offload_bytes)
            if total_intra_device_copy_bytes > 0:
                self._log_counter(self.kv_cache_intra_device_copy_bytes_total,
                                  {}, total_intra_device_copy_bytes)
