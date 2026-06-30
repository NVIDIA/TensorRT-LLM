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

import logging
import math
import threading
import time
from collections import deque
from typing import Dict, List, NamedTuple, Optional, Union

from .enums import MetricNames

_LOGGER = logging.getLogger(__name__)

# Phase-1 WideEP uses the two-word uint64 active-rank-mask ABI. Reject a
# malformed payload before it can create unbounded Prometheus cardinality on
# the serving event loop.
_MAX_EP_HEALTH_RANKS = 128
# Prometheus gauges are IEEE-754 doubles. Keep the monotonic generation exact
# so stale/conflict comparisons have the same meaning before and after export.
_MAX_EP_HEALTH_GENERATION = (1 << 53) - 1
_MAX_EP_HEALTH_SOURCE_EPOCH_LENGTH = 128
_MAX_RETIRED_EP_HEALTH_SOURCE_EPOCHS = 32


class _EPHealthSnapshot(NamedTuple):
    """One immutable snapshot exported by the local Prometheus collector."""

    world_size: int
    active_count: int
    failed_ranks: frozenset[int]
    generation: int


class _EPHealthPrometheusCollector:
    """Export a coherent EP snapshot without multiprocess gauge merging.

    ``prometheus_client`` combines each multiprocess gauge independently.  A
    rank vector, its aggregate counts, and its validity bit therefore cannot
    form an atomic snapshot when more than one process writes the same label
    set.  EP health is polled by the OpenAI server itself, so keep its latest
    immutable value in that process and register this collector directly on
    the server's scrape registry instead.
    """

    def __init__(self, labels: Dict[str, str], metric_prefix: str) -> None:
        self._label_names = list(labels)
        self._label_values = [labels[name] for name in self._label_names]
        self._metric_prefix = metric_prefix
        self._lock = threading.Lock()
        self._snapshot: Optional[_EPHealthSnapshot] = None
        self._available = False

    def publish(self, snapshot: _EPHealthSnapshot) -> None:
        """Atomically replace the snapshot and mark it available."""
        with self._lock:
            self._snapshot = snapshot
            self._available = True

    def mark_unavailable(self) -> None:
        """Atomically invalidate the retained snapshot."""
        with self._lock:
            self._available = False

    def collect(self) -> list:
        """Build all metric families from one locked state read."""
        with self._lock:
            snapshot = self._snapshot
            available = self._available

        # Do not advertise EP health for non-EP or non-FT deployments. Once a
        # producer has published a valid snapshot, retain its samples during an
        # outage and use the availability family as their validity bit.
        if snapshot is None:
            return []

        from prometheus_client.core import GaugeMetricFamily

        metrics = []
        rank_active = GaugeMetricFamily(
            self._metric_prefix + "ep_rank_active",
            "Whether an Expert Parallel rank is included in coordinator-committed "
            "data-plane membership (1) or excluded (0); not physical liveness.",
            labels=[*self._label_names, MetricsCollector.labelname_ep_rank],
        )
        for rank in range(snapshot.world_size):
            rank_active.add_metric(
                [*self._label_values, str(rank)],
                int(rank not in snapshot.failed_ranks),
            )
        metrics.append(rank_active)

        for name, documentation, value in (
            (
                "ep_active_ranks",
                "Number of ranks in coordinator-committed EP data-plane membership.",
                snapshot.active_count,
            ),
            (
                "ep_failed_ranks",
                "Number of ranks excluded from coordinator-committed EP data-plane "
                "membership; not a physical-failure detector.",
                snapshot.world_size - snapshot.active_count,
            ),
            (
                "ep_health_generation",
                "Committed EP membership version; increments on each effective transition.",
                snapshot.generation,
            ),
        ):
            metric = GaugeMetricFamily(
                self._metric_prefix + name,
                documentation,
                labels=self._label_names,
            )
            metric.add_metric(self._label_values, value)
            metrics.append(metric)

        availability = GaugeMetricFamily(
            self._metric_prefix + "ep_health_available",
            "Whether rank-0 committed EP membership telemetry is available. "
            "When 0, ignore all other trtllm_ep_* health samples.",
            labels=self._label_names,
        )
        availability.add_metric(self._label_values, int(available))
        metrics.append(availability)
        return metrics


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
            trtllm_prompt_cached_tokens_total
            trtllm_prompt_cached_tokens_per_request
            trtllm_spec_decode_drafted_tokens_total
            trtllm_spec_decode_accepted_tokens_total
            trtllm_prefill_perplexity
            trtllm_generation_perplexity
            trtllm_request_error_total

        Expert Parallel health metrics:
            trtllm_ep_rank_active
            trtllm_ep_active_ranks
            trtllm_ep_failed_ranks
            trtllm_ep_health_generation
            trtllm_ep_health_available

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
            trtllm_prefill_batch_occupancy
            trtllm_prefill_batch_tokens

        Config info metrics (logged once at startup via log_config_info):
            trtllm_model_config_info
            trtllm_parallel_config_info
            trtllm_speculative_config_info
            trtllm_kv_cache_config_info
    """
    labelname_finish_reason = "finished_reason"
    labelname_ep_rank = "ep_rank"

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

        # Committed Expert Parallel membership is passively observed from the
        # rank-0 worker through a dedicated internal RPC. Unlike independent
        # multiprocess gauges, this local collector preserves the rank vector,
        # aggregate counts, generation, and availability as one snapshot.
        self._last_ep_health_state = None
        self._last_ep_health_rejection = None
        self._retired_ep_health_source_epochs = set()
        self._retired_ep_health_source_epoch_order = deque()
        self._ep_health_prometheus_collector = _EPHealthPrometheusCollector(
            self.labels, self.metric_prefix)

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

        # Prompt cache hit tracking
        self.counter_tokens_cached_prompt = Counter(
            name=self.metric_prefix + "prompt_cached_tokens_total",
            documentation="Total prompt tokens served from KV cache.",
            labelnames=self.labels.keys())
        self.histogram_tokens_cached_prompt = Histogram(
            name=self.metric_prefix + "prompt_cached_tokens_per_request",
            documentation="Histogram of cached prompt tokens per request.",
            buckets=[0, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
            labelnames=self.labels.keys())

        # Per-position speculative decoding acceptance counters
        self.labelname_token_pos = "token_position"  # nosec: B105
        self.labels_with_token_pos = {
            **self.labels, self.labelname_token_pos: ""
        }
        self.counter_tokens_drafted_per_position = Counter(
            name=self.metric_prefix + "spec_decode_drafted_tokens_total",
            documentation=
            "Total drafted tokens per speculative decoding position.",
            labelnames=self.labels_with_token_pos.keys())
        self.counter_tokens_accepted_per_position = Counter(
            name=self.metric_prefix + "spec_decode_accepted_tokens_total",
            documentation=
            "Total accepted tokens per speculative decoding position.",
            labelnames=self.labels_with_token_pos.keys())

        # Per-request perplexity histograms
        self.histogram_prefill_perplexity = Histogram(
            name=self.metric_prefix + "prefill_perplexity",
            documentation="Histogram of prefill perplexity per request.",
            buckets=[1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 500.0, 1000.0],
            labelnames=self.labels.keys())
        self.histogram_generation_perplexity = Histogram(
            name=self.metric_prefix + "generation_perplexity",
            documentation="Histogram of generation perplexity per request.",
            buckets=[1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 500.0, 1000.0],
            labelnames=self.labels.keys())

        # Prefill batch occupancy / context token distribution
        self.gauge_prefill_batch_occupancy = Gauge(
            name=self.metric_prefix + "prefill_batch_occupancy",
            documentation=
            "Fraction of max active slots occupied by context requests.",
            labelnames=self.labels.keys())
        self.histogram_prefill_batch_tokens = Histogram(
            name=self.metric_prefix + "prefill_batch_tokens",
            documentation="Histogram of total context tokens per iteration.",
            buckets=[64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
            labelnames=self.labels.keys())

        # HTTP error counter
        self.labelname_http_code = "http_code"
        self.labels_with_http_code = {
            **self.labels, self.labelname_http_code: ""
        }
        self.counter_request_error = Counter(
            name=self.metric_prefix + "request_error_total",
            documentation="Total request errors, labeled by HTTP status code.",
            labelnames=self.labels_with_http_code.keys())

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
            if MetricNames.PROMPT_CACHE_CACHED_TOKENS in metrics_dict:
                cached_tokens = metrics_dict[
                    MetricNames.PROMPT_CACHE_CACHED_TOKENS]
                if cached_tokens > 0:
                    self._log_counter(self.counter_tokens_cached_prompt,
                                      self.labels, cached_tokens)
                self._log_histogram(self.histogram_tokens_cached_prompt,
                                    cached_tokens)

            per_pos_drafted = metrics_dict.get(
                MetricNames.SPEC_DEC_DRAFTED_PER_POS)
            per_pos_accepted = metrics_dict.get(
                MetricNames.SPEC_DEC_ACCEPTED_PER_POS)
            if per_pos_drafted is not None and per_pos_accepted is not None:
                last_nonzero = -1
                for i in range(len(per_pos_drafted) - 1, -1, -1):
                    if per_pos_drafted[i] > 0:
                        last_nonzero = i
                        break
                for pos in range(last_nonzero + 1):
                    labels_with_pos = {
                        **self.labels, self.labelname_token_pos: pos
                    }
                    if per_pos_drafted[pos] > 0:
                        self.counter_tokens_drafted_per_position.labels(
                            **labels_with_pos).inc(per_pos_drafted[pos])
                    if per_pos_accepted[pos] > 0:
                        self.counter_tokens_accepted_per_position.labels(
                            **labels_with_pos).inc(per_pos_accepted[pos])

            prefill_ppl = metrics_dict.get(MetricNames.PREFILL_PERPLEXITY)
            if prefill_ppl is not None and math.isfinite(prefill_ppl):
                self._log_histogram(self.histogram_prefill_perplexity,
                                    prefill_ppl)
            gen_ppl = metrics_dict.get(MetricNames.GENERATION_PERPLEXITY)
            if gen_ppl is not None and math.isfinite(gen_ppl):
                self._log_histogram(self.histogram_generation_perplexity,
                                    gen_ppl)

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
                num_ctx_tokens = ifb_stats["numCtxTokens"]
                self._log_gauge(self.total_context_tokens, num_ctx_tokens)
                if num_ctx_tokens > 0:
                    self._log_histogram(self.histogram_prefill_batch_tokens,
                                        num_ctx_tokens)
            if "avgNumDecodedTokensPerIter" in ifb_stats:
                self._log_gauge(self.avg_decoded_tokens_per_iter,
                                ifb_stats["avgNumDecodedTokensPerIter"])

            # Prefill batch occupancy: context_requests / max_active_requests
            num_context = ifb_stats.get("numContextRequests", 0)
            max_active = iteration_stats.get("maxNumActiveRequests", 0)
            if max_active and max_active > 0:
                self._log_gauge(self.gauge_prefill_batch_occupancy,
                                num_context / max_active)

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

        # Per-iteration KV cache stats. V2 reports reuse/miss by lifecycle and
        # storage/transfer counters by pool group; legacy V1 uses window stats.
        kv_iter = iteration_stats.get("kvCacheIterationStats")
        kv_iter_by_lifecycle = iteration_stats.get(
            "kvCacheIterationStatsByLifecycle")
        kv_iter_by_pool_group = iteration_stats.get(
            "kvCacheIterationStatsByPoolGroup")
        if kv_iter or kv_iter_by_lifecycle or kv_iter_by_pool_group:
            reuse_stats = kv_iter_by_lifecycle or kv_iter or {}
            pool_group_stats = kv_iter_by_pool_group or kv_iter or {}
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

            for stats in reuse_stats.values():
                total_reused += stats.get("iterReusedBlocks", 0)
                total_full_reused += stats.get("iterFullReusedBlocks", 0)
                total_partial_reused += stats.get("iterPartialReusedBlocks", 0)
                total_missed += stats.get("iterMissedBlocks", 0)

            for stats in pool_group_stats.values():
                total_secondary_max += stats.get("secondaryMaxNumBlocks", 0)
                total_secondary_used += stats.get("secondaryUsedNumBlocks", 0)
                total_gen_alloc += stats.get("iterGenAllocBlocks", 0)
                total_onboard_bytes += stats.get("iterOnboardBytes", 0)
                total_offload_bytes += stats.get("iterOffloadBytes", 0)
                total_intra_device_copy_bytes += stats.get(
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

    def _reject_ep_health_stats(self, message: str, *args: object) -> bool:
        """Mark telemetry unavailable and warn once per repeated rejection."""
        signature = (message, tuple(str(arg) for arg in args))
        if signature != self._last_ep_health_rejection:
            _LOGGER.warning(message, *args)
            self._last_ep_health_rejection = signature
        self.log_ep_health_unavailable()
        return False

    def log_ep_health_stats(self, ep_health_stats: dict) -> bool:
        """Materialize committed ``EPGroupHealth`` membership as gauges.

        Returns ``False`` when the payload is invalid, stale, or conflicts with
        the last accepted snapshot. Callers may retry because a later generation
        or a never-before-seen producer epoch can restore a coherent stream.
        ``sourceEpoch`` is required so a producer restart cannot be confused
        with delayed state from a retired producer. This passive consumer does
        not interpret the snapshot as physical liveness or mutate recovery
        state.
        """
        try:
            world_size = ep_health_stats["worldSize"]
            active_count = ep_health_stats["activeCount"]
            generation = ep_health_stats["generation"]
            failed_rank_list = ep_health_stats["failedRanks"]
            source_epoch = ep_health_stats["sourceEpoch"]
            scalar_values = (world_size, active_count, generation)
            if any(type(value) is not int for value in scalar_values):
                raise TypeError(
                    "worldSize, activeCount, and generation must be integers")
            if (world_size <= 0 or world_size > _MAX_EP_HEALTH_RANKS
                    or not 0 <= active_count <= world_size or generation < 0
                    or generation > _MAX_EP_HEALTH_GENERATION):
                raise ValueError("EP health scalar values are out of range")
            if not isinstance(failed_rank_list, list):
                raise TypeError("failedRanks must be a list")
            if (not isinstance(source_epoch, str) or not source_epoch
                    or len(source_epoch) > _MAX_EP_HEALTH_SOURCE_EPOCH_LENGTH):
                raise TypeError(
                    "sourceEpoch must be a bounded non-empty string")
            if len(failed_rank_list) > world_size:
                raise ValueError("failedRanks contains too many ranks")
            if any(
                    type(rank) is not int or not 0 <= rank < world_size
                    for rank in failed_rank_list):
                raise ValueError("failedRanks contains an invalid rank")
            failed_ranks = set(failed_rank_list)
            if (len(failed_ranks) != len(failed_rank_list)
                    or active_count != world_size - len(failed_ranks)):
                raise ValueError("EP health counts are inconsistent")
        except (KeyError, TypeError, ValueError) as error:
            return self._reject_ep_health_stats(
                "Ignoring invalid epHealthStats payload: %s", error)

        state = (source_epoch, world_size, generation,
                 tuple(sorted(failed_ranks)))
        last_source_epoch = None
        if self._last_ep_health_state is not None:
            last_source_epoch, last_world_size, last_generation, last_failed_ranks = (
                self._last_ep_health_state)
            if world_size != last_world_size:
                return self._reject_ep_health_stats(
                    "Ignoring epHealthStats world-size change from %s to %s",
                    last_world_size, world_size)
            if source_epoch == last_source_epoch and generation < last_generation:
                return self._reject_ep_health_stats(
                    "Ignoring stale epHealthStats generation %s after %s",
                    generation, last_generation)
            if (source_epoch == last_source_epoch
                    and generation == last_generation
                    and state[3] != last_failed_ranks):
                return self._reject_ep_health_stats(
                    "Conflicting epHealthStats payloads at generation %s",
                    generation)
            if source_epoch != last_source_epoch:
                if source_epoch in self._retired_ep_health_source_epochs:
                    return self._reject_ep_health_stats(
                        "Ignoring epHealthStats from retired source epoch %s",
                        source_epoch)

        if last_source_epoch is not None and source_epoch != last_source_epoch:
            if (len(self._retired_ep_health_source_epoch_order) ==
                    _MAX_RETIRED_EP_HEALTH_SOURCE_EPOCHS):
                expired_epoch = self._retired_ep_health_source_epoch_order.popleft(
                )
                self._retired_ep_health_source_epochs.remove(expired_epoch)
            self._retired_ep_health_source_epoch_order.append(last_source_epoch)
            self._retired_ep_health_source_epochs.add(last_source_epoch)
        self._last_ep_health_state = state
        self._last_ep_health_rejection = None
        self._ep_health_prometheus_collector.publish(
            _EPHealthSnapshot(
                world_size=world_size,
                active_count=active_count,
                failed_ranks=frozenset(failed_ranks),
                generation=generation,
            ))
        return True

    def log_ep_health_unavailable(self) -> None:
        """Mark the last EP health telemetry read as unavailable."""
        self._ep_health_prometheus_collector.mark_unavailable()

    def register_ep_health_metrics(self, registry: object) -> None:
        """Register coherent EP health metrics on a scrape registry."""
        registry.register(self._ep_health_prometheus_collector)

    def log_request_error(self, http_code: Union[int, str] = "") -> None:
        """Increment the error counter, labeled by HTTP status code."""
        labels = {**self.labels, self.labelname_http_code: str(http_code)}
        self.counter_request_error.labels(**labels).inc(1)
