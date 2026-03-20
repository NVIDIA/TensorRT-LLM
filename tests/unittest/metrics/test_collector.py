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
"""Unit tests for MetricsCollector iteration stats Prometheus metrics."""

import pytest
from prometheus_client import REGISTRY, CollectorRegistry

from tensorrt_llm.metrics.collector import MetricsCollector
from tensorrt_llm.metrics.enums import MetricNames


@pytest.fixture(autouse=True)
def clean_registry():
    """Unregister all custom collectors between tests to avoid duplicate metric errors."""
    collectors_to_remove = []
    for collector in REGISTRY._names_to_collectors.values():
        if hasattr(collector, '_name') and collector._name.startswith(
                'trtllm_'):
            collectors_to_remove.append(collector)
    for collector in set(collectors_to_remove):
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass
    yield


@pytest.fixture
def collector():
    labels = {"model_name": "test_model"}
    return MetricsCollector(labels)


def _get_gauge_value(gauge, labels):
    return gauge.labels(**labels)._value.get()


def _get_counter_value(counter, labels):
    return counter.labels(**labels)._value.get()


SAMPLE_ITERATION_STATS = {
    "numActiveRequests": 5,
    "numQueuedRequests": 3,
    "numCompletedRequests": 2,
    "maxNumActiveRequests": 10,
    "iterLatencyMS": 15.5,
    "gpuMemUsage": 4_000_000_000,
    "cpuMemUsage": 2_000_000_000,
    "pinnedMemUsage": 500_000_000,
    "maxBatchSizeStatic": 64,
    "maxBatchSizeRuntime": 32,
    "maxNumTokensRuntime": 8192,
    "kvCacheStats": {
        "cacheHitRate": 0.85,
        "maxNumBlocks": 1000,
        "freeNumBlocks": 400,
        "usedNumBlocks": 600,
        "tokensPerBlock": 64,
    },
    "inflightBatchingStats": {
        "numContextRequests": 2,
        "numGenRequests": 3,
        "numPausedRequests": 1,
        "numScheduledRequests": 5,
        "numCtxTokens": 256,
        "avgNumDecodedTokensPerIter": 4.5,
    },
    "specDecodingStats": {
        "numDraftTokens": 20,
        "numAcceptedTokens": 15,
        "acceptanceLength": 3.75,
        "draftOverhead": 1.2,
    },
}


class TestIterationStatsTopLevel:
    """Test top-level iteration stats are correctly exposed as Prometheus metrics."""

    def test_queue_and_load_gauges(self, collector):
        labels = {"model_name": "test_model"}
        collector.log_iteration_stats(SAMPLE_ITERATION_STATS)

        assert _get_gauge_value(collector.num_requests_running, labels) == 5
        assert _get_gauge_value(collector.num_requests_waiting, labels) == 3
        assert _get_gauge_value(collector.max_num_active_requests, labels) == 10

    def test_completed_requests_counter(self, collector):
        labels = {"model_name": "test_model"}
        collector.log_iteration_stats(SAMPLE_ITERATION_STATS)
        assert _get_counter_value(collector.counter_num_requests_completed,
                                  labels) == 2

        # Counter should accumulate across iterations
        collector.log_iteration_stats(SAMPLE_ITERATION_STATS)
        assert _get_counter_value(collector.counter_num_requests_completed,
                                  labels) == 4

    def test_completed_requests_zero_not_incremented(self, collector):
        labels = {"model_name": "test_model"}
        stats = {**SAMPLE_ITERATION_STATS, "numCompletedRequests": 0}
        collector.log_iteration_stats(stats)
        assert _get_counter_value(collector.counter_num_requests_completed,
                                  labels) == 0

    def test_iteration_latency_ms_to_seconds(self, collector):
        labels = {"model_name": "test_model"}
        collector.log_iteration_stats(SAMPLE_ITERATION_STATS)
        assert _get_gauge_value(collector.iteration_latency_seconds,
                                labels) == pytest.approx(0.0155)

    def test_memory_usage(self, collector):
        labels = {"model_name": "test_model"}
        collector.log_iteration_stats(SAMPLE_ITERATION_STATS)

        assert _get_gauge_value(collector.gpu_memory_usage_bytes,
                                labels) == 4_000_000_000
        assert _get_gauge_value(collector.cpu_memory_usage_bytes,
                                labels) == 2_000_000_000
        assert _get_gauge_value(collector.pinned_memory_usage_bytes,
                                labels) == 500_000_000

    def test_batch_size(self, collector):
        labels = {"model_name": "test_model"}
        collector.log_iteration_stats(SAMPLE_ITERATION_STATS)

        assert _get_gauge_value(collector.max_batch_size_static, labels) == 64
        assert _get_gauge_value(collector.max_batch_size_runtime, labels) == 32
        assert _get_gauge_value(collector.max_num_tokens_runtime,
                                labels) == 8192


class TestKVCacheStats:
    """Test KV cache stats are correctly exposed."""

    def test_kv_cache_block_gauges(self, collector):
        labels = {"model_name": "test_model"}
        collector.log_iteration_stats(SAMPLE_ITERATION_STATS)

        assert _get_gauge_value(collector.kv_cache_max_blocks, labels) == 1000
        assert _get_gauge_value(collector.kv_cache_free_blocks, labels) == 400
        assert _get_gauge_value(collector.kv_cache_used_blocks, labels) == 600
        assert _get_gauge_value(collector.kv_cache_tokens_per_block,
                                labels) == 64

    def test_kv_cache_utilization(self, collector):
        labels = {"model_name": "test_model"}
        collector.log_iteration_stats(SAMPLE_ITERATION_STATS)
        assert _get_gauge_value(collector.kv_cache_utilization,
                                labels) == pytest.approx(0.6)

    def test_kv_cache_hit_rate(self, collector):
        labels = {"model_name": "test_model"}
        collector.log_iteration_stats(SAMPLE_ITERATION_STATS)
        assert _get_gauge_value(collector.kv_cache_hit_rate,
                                labels) == pytest.approx(0.85)


class TestInflightBatchingStats:
    """Test inflight batching stats are correctly exposed."""

    def test_inflight_batching_gauges(self, collector):
        labels = {"model_name": "test_model"}
        collector.log_iteration_stats(SAMPLE_ITERATION_STATS)

        assert _get_gauge_value(collector.num_context_requests, labels) == 2
        assert _get_gauge_value(collector.num_generation_requests, labels) == 3
        assert _get_gauge_value(collector.num_paused_requests, labels) == 1
        assert _get_gauge_value(collector.num_scheduled_requests, labels) == 5
        assert _get_gauge_value(collector.total_context_tokens, labels) == 256
        assert _get_gauge_value(collector.avg_decoded_tokens_per_iter,
                                labels) == pytest.approx(4.5)

    def test_missing_inflight_batching_stats(self, collector):
        """No error when inflightBatchingStats is absent."""
        stats = {k: v for k, v in SAMPLE_ITERATION_STATS.items()
                 if k != "inflightBatchingStats"}
        collector.log_iteration_stats(stats)


class TestSpecDecodingStats:
    """Test speculative decoding stats are correctly exposed."""

    def test_spec_decoding_metrics(self, collector):
        labels = {"model_name": "test_model"}
        collector.log_iteration_stats(SAMPLE_ITERATION_STATS)

        assert _get_counter_value(
            collector.counter_spec_decode_num_draft_tokens,
            labels) == 20
        assert _get_counter_value(
            collector.counter_spec_decode_num_accepted_tokens,
            labels) == 15
        assert _get_gauge_value(collector.spec_decode_acceptance_length,
                                labels) == pytest.approx(3.75)
        assert _get_gauge_value(collector.spec_decode_draft_overhead,
                                labels) == pytest.approx(1.2)

    def test_missing_spec_decoding_stats(self, collector):
        """No error when specDecodingStats is absent."""
        stats = {k: v for k, v in SAMPLE_ITERATION_STATS.items()
                 if k != "specDecodingStats"}
        collector.log_iteration_stats(stats)


class TestPartialStats:
    """Test that partial iteration stats don't cause errors."""

    def test_empty_stats(self, collector):
        collector.log_iteration_stats({})

    def test_only_kv_cache(self, collector):
        collector.log_iteration_stats({
            "kvCacheStats": {
                "maxNumBlocks": 100,
                "usedNumBlocks": 50,
            }
        })
        labels = {"model_name": "test_model"}
        assert _get_gauge_value(collector.kv_cache_max_blocks, labels) == 100
        assert _get_gauge_value(collector.kv_cache_used_blocks, labels) == 50


class TestConfigInfoMetrics:
    """Test config info gauges are correctly exposed."""

    def test_model_config_info(self, collector):
        model_config = {
            "model": "meta-llama/Llama-3-8B",
            "served_model_name": "Llama-3-8B",
            "dtype": "float16",
            "quantization": "none",
            "max_model_len": "4096",
            "gpu_type": "NVIDIA H100",
        }
        collector.log_config_info(model_config=model_config)

        from prometheus_client import REGISTRY
        # Verify the metric was registered
        assert "trtllm_model_config_info" in REGISTRY._names_to_collectors

    def test_parallel_config_info(self, collector):
        parallel_config = {
            "tensor_parallel_size": "4",
            "pipeline_parallel_size": "2",
            "gpu_count": "8",
        }
        collector.log_config_info(parallel_config=parallel_config)

        from prometheus_client import REGISTRY
        assert "trtllm_parallel_config_info" in REGISTRY._names_to_collectors

    def test_speculative_config_info(self, collector):
        spec_config = {
            "spec_enabled": "true",
            "spec_method": "Eagle",
            "spec_num_tokens": "5",
            "spec_draft_model": "eagle-model",
        }
        collector.log_config_info(speculative_config=spec_config)

        from prometheus_client import REGISTRY
        assert "trtllm_speculative_config_info" in REGISTRY._names_to_collectors

    def test_cache_config_info(self, collector):
        cache_config = {
            "page_size": "64",
            "enable_block_reuse": "True",
            "cache_dtype": "auto",
        }
        collector.log_config_info(cache_config=cache_config)

        from prometheus_client import REGISTRY
        assert "trtllm_cache_config_info" in REGISTRY._names_to_collectors

    def test_no_config_no_error(self, collector):
        """No error when all configs are None."""
        collector.log_config_info()

    def test_partial_config(self, collector):
        """Only model config provided, others None."""
        collector.log_config_info(
            model_config={"model": "test", "dtype": "auto"})


# ---------------------------------------------------------------------------
# Per-request token counters and phase histograms (Step 1 additions)
# ---------------------------------------------------------------------------

SAMPLE_REQUEST_METRICS_FULL = {
    MetricsCollector.labelname_finish_reason: "end_id",
    # latency fields (seconds) — keys are MetricNames enum members
    MetricNames.E2E: 2.5,
    MetricNames.TTFT: 0.3,
    MetricNames.TPOT: 0.05,
    MetricNames.REQUEST_QUEUE_TIME: 0.1,
    MetricNames.PREFILL_TIME: 0.2,
    MetricNames.DECODE_TIME: 2.2,
    MetricNames.INFERENCE_TIME: 2.4,
    # token counts
    MetricNames.PROMPT_TOKENS: 128,
    MetricNames.GENERATION_TOKENS: 50,
}




class TestPerRequestTokenCounters:
    """Test prompt_tokens_total and generation_tokens_total counters."""

    def test_prompt_tokens_incremented(self, collector):
        labels = {"model_name": "test_model"}
        collector.log_request_metrics_dict(SAMPLE_REQUEST_METRICS_FULL)
        assert _get_counter_value(collector.counter_prompt_tokens,
                                  labels) == 128

    def test_generation_tokens_incremented(self, collector):
        labels = {"model_name": "test_model"}
        collector.log_request_metrics_dict(SAMPLE_REQUEST_METRICS_FULL)
        assert _get_counter_value(collector.counter_generation_tokens,
                                  labels) == 50

    def test_token_counters_accumulate(self, collector):
        """Counters should sum across multiple requests."""
        labels = {"model_name": "test_model"}
        collector.log_request_metrics_dict(SAMPLE_REQUEST_METRICS_FULL)
        collector.log_request_metrics_dict(SAMPLE_REQUEST_METRICS_FULL)
        assert _get_counter_value(collector.counter_prompt_tokens,
                                  labels) == 256
        assert _get_counter_value(collector.counter_generation_tokens,
                                  labels) == 100

    def test_missing_token_counts_no_error(self, collector):
        """No error and no increment when token counts are absent."""
        labels = {"model_name": "test_model"}
        metrics_without_tokens = {
            MetricsCollector.labelname_finish_reason: "end_id",
            MetricNames.E2E: 1.0,
            MetricNames.TTFT: 0.1,
        }
        collector.log_request_metrics_dict(metrics_without_tokens)
        assert _get_counter_value(collector.counter_prompt_tokens, labels) == 0
        assert _get_counter_value(collector.counter_generation_tokens,
                                  labels) == 0

    def test_zero_tokens_not_incremented(self, collector):
        """Zero token counts should not increment the counter."""
        labels = {"model_name": "test_model"}
        metrics = {
            **SAMPLE_REQUEST_METRICS_FULL,
            MetricNames.PROMPT_TOKENS: 0,
            MetricNames.GENERATION_TOKENS: 0,
        }
        collector.log_request_metrics_dict(metrics)
        assert _get_counter_value(collector.counter_prompt_tokens, labels) == 0
        assert _get_counter_value(collector.counter_generation_tokens,
                                  labels) == 0


def _get_histogram_sum(histogram, labels):
    """Return the sum of all observations in a Prometheus histogram."""
    for metric in REGISTRY.collect():
        if metric.name == histogram._name:
            for sample in metric.samples:
                if sample.name.endswith("_sum") and sample.labels == labels:
                    return sample.value
    return 0.0


def _get_histogram_count(histogram, labels):
    """Return the number of observations in a Prometheus histogram."""
    for metric in REGISTRY.collect():
        if metric.name == histogram._name:
            for sample in metric.samples:
                if sample.name.endswith("_count") and sample.labels == labels:
                    return int(sample.value)
    return 0


class TestPerRequestPhaseHistograms:
    """Test request_prefill_time_seconds, _decode_time_seconds, _inference_time_seconds."""

    def test_prefill_time_observed(self, collector):
        labels = {"model_name": "test_model"}
        collector.log_request_metrics_dict(SAMPLE_REQUEST_METRICS_FULL)
        assert _get_histogram_count(collector.histogram_prefill_time_request,
                                    labels) == 1
        assert _get_histogram_sum(collector.histogram_prefill_time_request,
                                  labels) == pytest.approx(0.2)

    def test_decode_time_observed(self, collector):
        labels = {"model_name": "test_model"}
        collector.log_request_metrics_dict(SAMPLE_REQUEST_METRICS_FULL)
        assert _get_histogram_count(collector.histogram_decode_time_request,
                                    labels) == 1
        assert _get_histogram_sum(collector.histogram_decode_time_request,
                                  labels) == pytest.approx(2.2)

    def test_inference_time_observed(self, collector):
        labels = {"model_name": "test_model"}
        collector.log_request_metrics_dict(SAMPLE_REQUEST_METRICS_FULL)
        assert _get_histogram_count(collector.histogram_inference_time_request,
                                    labels) == 1
        assert _get_histogram_sum(collector.histogram_inference_time_request,
                                  labels) == pytest.approx(2.4)

    def test_missing_phase_times_no_observation(self, collector):
        """No histogram observations when phase times are absent."""
        labels = {"model_name": "test_model"}
        metrics = {
            MetricsCollector.labelname_finish_reason: "end_id",
            MetricNames.E2E: 1.0,
            MetricNames.TTFT: 0.1,
        }
        collector.log_request_metrics_dict(metrics)
        assert _get_histogram_count(collector.histogram_prefill_time_request,
                                    labels) == 0
        assert _get_histogram_count(collector.histogram_decode_time_request,
                                    labels) == 0
        assert _get_histogram_count(collector.histogram_inference_time_request,
                                    labels) == 0

    def test_no_observation_without_finish_reason(self, collector):
        """Phase histograms must not be updated for in-progress requests."""
        labels = {"model_name": "test_model"}
        metrics_no_finish = {
            MetricNames.E2E: 1.0,
            MetricNames.PREFILL_TIME: 0.2,
            MetricNames.DECODE_TIME: 0.8,
            MetricNames.INFERENCE_TIME: 1.0,
            MetricNames.PROMPT_TOKENS: 10,
            MetricNames.GENERATION_TOKENS: 5,
        }
        collector.log_request_metrics_dict(metrics_no_finish)
        assert _get_histogram_count(collector.histogram_prefill_time_request,
                                    labels) == 0
        assert _get_counter_value(collector.counter_prompt_tokens, labels) == 0


from tensorrt_llm.metrics.perf_utils import process_req_perf_metrics
from tensorrt_llm.metrics.enums import RequestEventTiming

# Shared timing fixture used across TestProcessReqPerfMetrics tests.
_FULL_TIMESTAMPS = {
    RequestEventTiming.ARRIVAL_TIME: 1000.0,
    RequestEventTiming.FIRST_SCHEDULED_TIME: 1000.05,
    RequestEventTiming.FIRST_TOKEN_TIME: 1000.3,
    RequestEventTiming.LAST_TOKEN_TIME: 1002.5,
}


class TestProcessReqPerfMetrics:
    """Unit tests for process_req_perf_metrics (new phase timings + token counts)."""

    def test_phase_timings_computed(self):
        stat = process_req_perf_metrics(_FULL_TIMESTAMPS, output_length=50)

        assert stat[MetricNames.PREFILL_TIME] == pytest.approx(0.25)
        assert stat[MetricNames.DECODE_TIME] == pytest.approx(2.2)
        assert stat[MetricNames.INFERENCE_TIME] == pytest.approx(2.45)

    def test_base_latencies_computed(self):
        stat = process_req_perf_metrics(_FULL_TIMESTAMPS, output_length=50)

        assert stat[MetricNames.TTFT] == pytest.approx(0.3)
        assert stat[MetricNames.E2E] == pytest.approx(2.5)

    def test_zero_queue_time_is_included(self):
        """REQUEST_QUEUE_TIME=0 (instant scheduling) must not be filtered out."""
        timestamps = {
            RequestEventTiming.ARRIVAL_TIME: 1000.0,
            RequestEventTiming.FIRST_SCHEDULED_TIME: 1000.0,  # same as arrival
            RequestEventTiming.FIRST_TOKEN_TIME: 1000.3,
            RequestEventTiming.LAST_TOKEN_TIME: 1002.5,
        }
        stat = process_req_perf_metrics(timestamps, output_length=50)
        assert MetricNames.REQUEST_QUEUE_TIME in stat
        assert stat[MetricNames.REQUEST_QUEUE_TIME] == pytest.approx(0.0)

    def test_generation_tokens_in_stat(self):
        stat = process_req_perf_metrics(_FULL_TIMESTAMPS, output_length=50)
        assert stat[MetricNames.GENERATION_TOKENS] == 50

    def test_output_length_one_has_tokens_but_no_tpot(self):
        """GENERATION_TOKENS is present but TPOT is excluded for single-token output."""
        stat = process_req_perf_metrics(_FULL_TIMESTAMPS, output_length=1)
        assert stat[MetricNames.GENERATION_TOKENS] == 1
        assert MetricNames.TPOT not in stat

    def test_generation_tokens_excluded_for_multiple_response(self):
        stat = process_req_perf_metrics(_FULL_TIMESTAMPS,
                                        output_length=50,
                                        is_multiple_response=True)
        assert MetricNames.GENERATION_TOKENS not in stat

    def test_zero_output_length_excludes_tokens(self):
        stat = process_req_perf_metrics(_FULL_TIMESTAMPS, output_length=0)
        assert MetricNames.GENERATION_TOKENS not in stat

    def test_missing_timestamps_no_phase_timings(self):
        """When only the arrival timestamp is present, phase metrics are absent."""
        # Key must be RequestEventTiming enum to match dict lookups.
        raw = {RequestEventTiming.ARRIVAL_TIME: 1000.0}
        stat = process_req_perf_metrics(raw, output_length=10)
        assert MetricNames.PREFILL_TIME not in stat
        assert MetricNames.DECODE_TIME not in stat
        assert MetricNames.INFERENCE_TIME not in stat
        assert MetricNames.TTFT not in stat
        assert MetricNames.E2E not in stat

    def test_clock_skew_negative_phase_time_is_dropped(self):
        """Negative phase durations (clock skew) must not appear in output."""
        skewed = {
            RequestEventTiming.ARRIVAL_TIME: 1000.0,
            RequestEventTiming.FIRST_SCHEDULED_TIME: 1000.05,
            # first_token before first_scheduled — invalid clock ordering
            RequestEventTiming.FIRST_TOKEN_TIME: 1000.02,
            RequestEventTiming.LAST_TOKEN_TIME: 1002.5,
        }
        stat = process_req_perf_metrics(skewed, output_length=50)
        # PREFILL_TIME = 1000.02 - 1000.05 = -0.03 → must be absent
        assert MetricNames.PREFILL_TIME not in stat

    def test_empty_stats_returns_empty(self):
        assert process_req_perf_metrics(None, output_length=10) == {}
        assert process_req_perf_metrics({}, output_length=10) == {}
