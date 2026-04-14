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
"""Unit tests for MetricsCollector and process_req_perf_metrics."""

import pytest
from prometheus_client import REGISTRY

from tensorrt_llm.metrics.collector import MetricsCollector
from tensorrt_llm.metrics.enums import MetricNames, RequestEventTiming
from tensorrt_llm.metrics.perf_utils import process_req_perf_metrics


@pytest.fixture(autouse=True)
def clean_registry():
    """Unregister all custom collectors between tests to avoid duplicate metric errors."""
    collectors_to_remove = []
    for collector in REGISTRY._names_to_collectors.values():
        if hasattr(collector, "_name") and collector._name.startswith("trtllm_"):
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


def _get_gauge_value(collector, metric_name: str):
    """Get the current value of a Prometheus gauge."""
    metric = getattr(collector, metric_name)
    return metric.labels(**collector.labels)._value.get()


def _get_counter_value(collector, metric_name: str):
    """Get the current value of a Prometheus counter."""
    metric = getattr(collector, metric_name)
    return metric.labels(**collector.labels)._value.get()


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
        collector.log_iteration_stats(SAMPLE_ITERATION_STATS)

        assert _get_gauge_value(collector, "num_requests_running") == 5
        assert _get_gauge_value(collector, "num_requests_waiting") == 3
        assert _get_gauge_value(collector, "max_num_active_requests") == 10

    def test_completed_requests_counter(self, collector):
        collector.log_iteration_stats(SAMPLE_ITERATION_STATS)
        assert _get_counter_value(collector, "counter_num_requests_completed") == 2

        # Counter should accumulate across iterations
        collector.log_iteration_stats(SAMPLE_ITERATION_STATS)
        assert _get_counter_value(collector, "counter_num_requests_completed") == 4

    def test_completed_requests_zero_not_incremented(self, collector):
        stats = {**SAMPLE_ITERATION_STATS, "numCompletedRequests": 0}
        collector.log_iteration_stats(stats)
        assert _get_counter_value(collector, "counter_num_requests_completed") == 0

    def test_iteration_latency_ms_to_seconds(self, collector):
        collector.log_iteration_stats(SAMPLE_ITERATION_STATS)
        assert _get_gauge_value(collector, "iteration_latency_seconds") == pytest.approx(0.0155)

    def test_memory_usage(self, collector):
        collector.log_iteration_stats(SAMPLE_ITERATION_STATS)

        assert _get_gauge_value(collector, "gpu_memory_usage_bytes") == 4_000_000_000
        assert _get_gauge_value(collector, "cpu_memory_usage_bytes") == 2_000_000_000
        assert _get_gauge_value(collector, "pinned_memory_usage_bytes") == 500_000_000

    def test_batch_size(self, collector):
        collector.log_iteration_stats(SAMPLE_ITERATION_STATS)

        assert _get_gauge_value(collector, "max_batch_size_static") == 64
        assert _get_gauge_value(collector, "max_batch_size_runtime") == 32
        assert _get_gauge_value(collector, "max_num_tokens_runtime") == 8192


class TestKVCacheStats:
    """Test KV cache stats are correctly exposed."""

    def test_kv_cache_block_gauges(self, collector):
        collector.log_iteration_stats(SAMPLE_ITERATION_STATS)

        assert _get_gauge_value(collector, "kv_cache_max_blocks") == 1000
        assert _get_gauge_value(collector, "kv_cache_free_blocks") == 400
        assert _get_gauge_value(collector, "kv_cache_used_blocks") == 600
        assert _get_gauge_value(collector, "kv_cache_tokens_per_block") == 64

    def test_kv_cache_utilization(self, collector):
        collector.log_iteration_stats(SAMPLE_ITERATION_STATS)
        assert _get_gauge_value(collector, "kv_cache_utilization") == pytest.approx(0.6)

    def test_kv_cache_hit_rate(self, collector):
        collector.log_iteration_stats(SAMPLE_ITERATION_STATS)
        assert _get_gauge_value(collector, "kv_cache_hit_rate") == pytest.approx(0.85)


class TestInflightBatchingStats:
    """Test inflight batching stats are correctly exposed."""

    def test_inflight_batching_gauges(self, collector):
        collector.log_iteration_stats(SAMPLE_ITERATION_STATS)

        assert _get_gauge_value(collector, "num_context_requests") == 2
        assert _get_gauge_value(collector, "num_generation_requests") == 3
        assert _get_gauge_value(collector, "num_paused_requests") == 1
        assert _get_gauge_value(collector, "num_scheduled_requests") == 5
        assert _get_gauge_value(collector, "total_context_tokens") == 256
        assert _get_gauge_value(collector, "avg_decoded_tokens_per_iter") == pytest.approx(4.5)

    def test_missing_inflight_batching_stats(self, collector):
        """No error when inflightBatchingStats is absent."""
        stats = {k: v for k, v in SAMPLE_ITERATION_STATS.items() if k != "inflightBatchingStats"}
        collector.log_iteration_stats(stats)


class TestSpecDecodingStats:
    """Test speculative decoding stats are correctly exposed."""

    def test_spec_decoding_metrics(self, collector):
        collector.log_iteration_stats(SAMPLE_ITERATION_STATS)

        assert _get_counter_value(collector, "counter_spec_decode_num_draft_tokens") == 20
        assert _get_counter_value(collector, "counter_spec_decode_num_accepted_tokens") == 15
        assert _get_gauge_value(collector, "spec_decode_acceptance_length") == pytest.approx(3.75)
        assert _get_gauge_value(collector, "spec_decode_draft_overhead") == pytest.approx(1.2)

    def test_missing_spec_decoding_stats(self, collector):
        """No error when specDecodingStats is absent."""
        stats = {k: v for k, v in SAMPLE_ITERATION_STATS.items() if k != "specDecodingStats"}
        collector.log_iteration_stats(stats)


class TestPartialStats:
    """Test that partial iteration stats don't cause errors."""

    def test_empty_stats(self, collector):
        collector.log_iteration_stats({})

    def test_only_kv_cache(self, collector):
        collector.log_iteration_stats(
            {
                "kvCacheStats": {
                    "maxNumBlocks": 100,
                    "usedNumBlocks": 50,
                }
            }
        )
        assert _get_gauge_value(collector, "kv_cache_max_blocks") == 100
        assert _get_gauge_value(collector, "kv_cache_used_blocks") == 50


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
        assert (
            REGISTRY.get_sample_value(
                "trtllm_model_config_info", {"model_name": "test_model", **model_config}
            )
            == 1.0
        )

    def test_parallel_config_info(self, collector):
        parallel_config = {
            "tensor_parallel_size": "4",
            "pipeline_parallel_size": "2",
            "gpu_count": "8",
        }
        collector.log_config_info(parallel_config=parallel_config)
        assert (
            REGISTRY.get_sample_value(
                "trtllm_parallel_config_info", {"model_name": "test_model", **parallel_config}
            )
            == 1.0
        )

    def test_speculative_config_info(self, collector):
        spec_config = {
            "spec_enabled": "true",
            "spec_method": "Eagle",
            "spec_num_tokens": "5",
            "spec_draft_model": "eagle-model",
        }
        collector.log_config_info(speculative_config=spec_config)
        assert (
            REGISTRY.get_sample_value(
                "trtllm_speculative_config_info", {"model_name": "test_model", **spec_config}
            )
            == 1.0
        )

    def test_kv_cache_config_info(self, collector):
        kv_cache_config = {
            "page_size": "64",
            "enable_block_reuse": "True",
            "cache_dtype": "auto",
        }
        collector.log_config_info(kv_cache_config=kv_cache_config)
        assert (
            REGISTRY.get_sample_value(
                "trtllm_kv_cache_config_info", {"model_name": "test_model", **kv_cache_config}
            )
            == 1.0
        )

    def test_no_config_no_error(self, collector):
        """No error when all configs are None."""
        collector.log_config_info()

    def test_partial_config(self, collector):
        """Only model config provided, others None."""
        collector.log_config_info(model_config={"model": "test", "dtype": "auto"})


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


class TestRequestSuccessCounter:
    """Test counter_request_success increments with the correct finished_reason label."""

    def test_success_counter_incremented(self, collector):
        collector.log_request_metrics_dict(SAMPLE_REQUEST_METRICS_FULL)
        labels = {**collector.labels, "finished_reason": "end_id"}
        assert collector.counter_request_success.labels(**labels)._value.get() == 1

    def test_success_counter_tracks_finish_reason_separately(self, collector):
        """Different finish_reason values must be tracked in separate label series."""
        metrics_stop = {
            **SAMPLE_REQUEST_METRICS_FULL,
            MetricsCollector.labelname_finish_reason: "stop_words",
        }
        collector.log_request_metrics_dict(SAMPLE_REQUEST_METRICS_FULL)
        collector.log_request_metrics_dict(metrics_stop)
        end_id_labels = {**collector.labels, "finished_reason": "end_id"}
        stop_labels = {**collector.labels, "finished_reason": "stop_words"}
        assert collector.counter_request_success.labels(**end_id_labels)._value.get() == 1
        assert collector.counter_request_success.labels(**stop_labels)._value.get() == 1


class TestPerRequestTokenCounters:
    """Test prompt_tokens_total and generation_tokens_total counters."""

    def test_prompt_tokens_incremented(self, collector):
        collector.log_request_metrics_dict(SAMPLE_REQUEST_METRICS_FULL)
        assert _get_counter_value(collector, "counter_prompt_tokens") == 128

    def test_generation_tokens_incremented(self, collector):
        collector.log_request_metrics_dict(SAMPLE_REQUEST_METRICS_FULL)
        assert _get_counter_value(collector, "counter_generation_tokens") == 50

    def test_token_counters_accumulate(self, collector):
        """Counters should sum across multiple requests."""
        collector.log_request_metrics_dict(SAMPLE_REQUEST_METRICS_FULL)
        collector.log_request_metrics_dict(SAMPLE_REQUEST_METRICS_FULL)
        assert _get_counter_value(collector, "counter_prompt_tokens") == 256
        assert _get_counter_value(collector, "counter_generation_tokens") == 100

    def test_missing_token_counts_no_error(self, collector):
        """No error and no increment when token counts are absent."""
        metrics_without_tokens = {
            MetricsCollector.labelname_finish_reason: "end_id",
            MetricNames.E2E: 1.0,
            MetricNames.TTFT: 0.1,
        }
        collector.log_request_metrics_dict(metrics_without_tokens)
        assert _get_counter_value(collector, "counter_prompt_tokens") == 0
        assert _get_counter_value(collector, "counter_generation_tokens") == 0

    def test_zero_tokens_not_incremented(self, collector):
        """Zero token counts should not increment the counter."""
        metrics = {
            **SAMPLE_REQUEST_METRICS_FULL,
            MetricNames.PROMPT_TOKENS: 0,
            MetricNames.GENERATION_TOKENS: 0,
        }
        collector.log_request_metrics_dict(metrics)
        assert _get_counter_value(collector, "counter_prompt_tokens") == 0
        assert _get_counter_value(collector, "counter_generation_tokens") == 0

    def test_n_greater_than_1_prompt_tokens_counted_once(self, collector):
        """For n>1, PROMPT_TOKENS should be counted once (candidate 0 only).

        GENERATION_TOKENS accumulates per candidate.

        This simulates what record_stats in executor/result.py produces:
        candidate 0 emits both PROMPT_TOKENS and GENERATION_TOKENS,
        candidates 1+ emit only GENERATION_TOKENS (prompt is shared).
        """
        # Candidate 0: prompt + generation tokens
        collector.log_request_metrics_dict(
            {
                MetricsCollector.labelname_finish_reason: "end_id",
                MetricNames.PROMPT_TOKENS: 128,
                MetricNames.GENERATION_TOKENS: 50,
            }
        )
        # Candidate 1: only generation tokens (shared prompt not re-counted)
        collector.log_request_metrics_dict(
            {
                MetricsCollector.labelname_finish_reason: "end_id",
                MetricNames.GENERATION_TOKENS: 42,
            }
        )
        # Candidate 2: only generation tokens
        collector.log_request_metrics_dict(
            {
                MetricsCollector.labelname_finish_reason: "end_id",
                MetricNames.GENERATION_TOKENS: 38,
            }
        )
        # Prompt counted once, generation tokens summed across all candidates
        assert _get_counter_value(collector, "counter_prompt_tokens") == 128
        assert _get_counter_value(collector, "counter_generation_tokens") == 130  # 50 + 42 + 38


def _get_histogram_sum(collector, metric_name: str):
    """Return the sum of all observations in a Prometheus histogram."""
    histogram = getattr(collector, metric_name)
    for metric in REGISTRY.collect():
        if metric.name == histogram._name:
            for sample in metric.samples:
                if sample.name.endswith("_sum") and sample.labels == collector.labels:
                    return sample.value
    return 0.0


def _get_histogram_count(collector, metric_name: str):
    """Return the number of observations in a Prometheus histogram."""
    histogram = getattr(collector, metric_name)
    for metric in REGISTRY.collect():
        if metric.name == histogram._name:
            for sample in metric.samples:
                if sample.name.endswith("_count") and sample.labels == collector.labels:
                    return int(sample.value)
    return 0


class TestPerRequestPhaseHistograms:
    """Test request_prefill_time_seconds, _decode_time_seconds, _inference_time_seconds."""

    def test_prefill_time_observed(self, collector):
        collector.log_request_metrics_dict(SAMPLE_REQUEST_METRICS_FULL)
        assert _get_histogram_count(collector, "histogram_prefill_time_request") == 1
        assert _get_histogram_sum(collector, "histogram_prefill_time_request") == pytest.approx(0.2)

    def test_decode_time_observed(self, collector):
        collector.log_request_metrics_dict(SAMPLE_REQUEST_METRICS_FULL)
        assert _get_histogram_count(collector, "histogram_decode_time_request") == 1
        assert _get_histogram_sum(collector, "histogram_decode_time_request") == pytest.approx(2.2)

    def test_inference_time_observed(self, collector):
        collector.log_request_metrics_dict(SAMPLE_REQUEST_METRICS_FULL)
        assert _get_histogram_count(collector, "histogram_inference_time_request") == 1
        assert _get_histogram_sum(collector, "histogram_inference_time_request") == pytest.approx(
            2.4
        )

    def test_e2e_time_observed(self, collector):
        collector.log_request_metrics_dict(SAMPLE_REQUEST_METRICS_FULL)
        assert _get_histogram_count(collector, "histogram_e2e_time_request") == 1
        assert _get_histogram_sum(collector, "histogram_e2e_time_request") == pytest.approx(2.5)

    def test_ttft_observed(self, collector):
        collector.log_request_metrics_dict(SAMPLE_REQUEST_METRICS_FULL)
        assert _get_histogram_count(collector, "histogram_time_to_first_token") == 1
        assert _get_histogram_sum(collector, "histogram_time_to_first_token") == pytest.approx(0.3)

    def test_tpot_observed(self, collector):
        collector.log_request_metrics_dict(SAMPLE_REQUEST_METRICS_FULL)
        assert _get_histogram_count(collector, "histogram_time_per_output_token") == 1
        assert _get_histogram_sum(collector, "histogram_time_per_output_token") == pytest.approx(
            0.05
        )

    def test_missing_phase_times_no_observation(self, collector):
        """No histogram observations when phase times are absent."""
        metrics = {
            MetricsCollector.labelname_finish_reason: "end_id",
            MetricNames.E2E: 1.0,
            MetricNames.TTFT: 0.1,
        }
        collector.log_request_metrics_dict(metrics)
        assert _get_histogram_count(collector, "histogram_prefill_time_request") == 0
        assert _get_histogram_count(collector, "histogram_decode_time_request") == 0
        assert _get_histogram_count(collector, "histogram_inference_time_request") == 0

    def test_no_observation_without_finish_reason(self, collector):
        """Phase histograms must not be updated for in-progress requests."""
        metrics_no_finish = {
            MetricNames.E2E: 1.0,
            MetricNames.PREFILL_TIME: 0.2,
            MetricNames.DECODE_TIME: 0.8,
            MetricNames.INFERENCE_TIME: 1.0,
            MetricNames.PROMPT_TOKENS: 10,
            MetricNames.GENERATION_TOKENS: 5,
        }
        collector.log_request_metrics_dict(metrics_no_finish)
        assert _get_histogram_count(collector, "histogram_prefill_time_request") == 0
        assert _get_counter_value(collector, "counter_prompt_tokens") == 0

    def test_queue_time_zero_is_recorded_to_prometheus(self, collector):
        """REQUEST_QUEUE_TIME=0 must reach the Prometheus histogram (not silently dropped)."""
        metrics = {
            MetricsCollector.labelname_finish_reason: "end_id",
            MetricNames.E2E: 2.5,
            MetricNames.TTFT: 0.3,
            MetricNames.REQUEST_QUEUE_TIME: 0.0,  # immediate scheduling
        }
        collector.log_request_metrics_dict(metrics)
        assert _get_histogram_count(collector, "histogram_queue_time_request") == 1
        assert _get_histogram_sum(collector, "histogram_queue_time_request") == pytest.approx(0.0)


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

    def test_per_candidate_metrics_computed_independently(self):
        """Each candidate gets its own GENERATION_TOKENS and TPOT.

        Even when candidates have different output lengths.
        """
        stat_a = process_req_perf_metrics(_FULL_TIMESTAMPS, output_length=50)
        stat_b = process_req_perf_metrics(_FULL_TIMESTAMPS, output_length=30)
        assert stat_a[MetricNames.GENERATION_TOKENS] == 50
        assert stat_b[MetricNames.GENERATION_TOKENS] == 30
        # TPOT differs because output_length differs
        assert stat_a[MetricNames.TPOT] != stat_b[MetricNames.TPOT]

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
        """Negative phase durations (clock skew) must not appear in output.

        Non-negative metrics must still be present.
        """
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
        # Non-negative metrics must still be present
        assert MetricNames.TTFT in stat
        assert stat[MetricNames.TTFT] == pytest.approx(0.02)
        assert MetricNames.E2E in stat
        assert stat[MetricNames.E2E] == pytest.approx(2.5)
        assert MetricNames.DECODE_TIME in stat
        assert stat[MetricNames.DECODE_TIME] == pytest.approx(2.48)

    def test_negative_queue_time_is_dropped(self):
        """Negative REQUEST_QUEUE_TIME due to clock skew must not appear.

        When arrival is after first_scheduled due to clock skew,
        the output should not contain the metric.
        """
        skewed = {
            RequestEventTiming.ARRIVAL_TIME: 1000.10,  # arrives after scheduled
            RequestEventTiming.FIRST_SCHEDULED_TIME: 1000.05,
            RequestEventTiming.FIRST_TOKEN_TIME: 1000.5,
            RequestEventTiming.LAST_TOKEN_TIME: 1002.5,
        }
        stat = process_req_perf_metrics(skewed, output_length=50)
        # REQUEST_QUEUE_TIME = 1000.05 - 1000.10 = -0.05 → must be absent
        assert MetricNames.REQUEST_QUEUE_TIME not in stat
        # Other metrics are still valid
        assert MetricNames.PREFILL_TIME in stat

    def test_empty_stats_returns_empty(self):
        assert process_req_perf_metrics(None, output_length=10) == {}


class TestCustomHistogramBuckets:
    """Tests for configurable Prometheus histogram bucket boundaries."""

    def test_custom_buckets_applied(self):
        """Custom bucket lists are reflected in the created histograms."""
        labels = {"model_name": "test_model"}
        custom_e2e = [0.1, 0.5, 1.0, 5.0, 10.0]
        custom_ttft = [0.001, 0.01, 0.1, 1.0]
        c = MetricsCollector(
            labels, e2e_request_latency_buckets=custom_e2e, time_to_first_token_buckets=custom_ttft
        )
        # prometheus_client appends +Inf automatically
        assert list(c.histogram_e2e_time_request._upper_bounds) == custom_e2e + [float("inf")]
        assert list(c.histogram_time_to_first_token._upper_bounds) == custom_ttft + [float("inf")]

    def test_none_uses_defaults(self):
        """None (unset) falls back to the built-in default bucket boundaries."""
        labels = {"model_name": "test_model"}
        c = MetricsCollector(labels)
        default_e2e = [
            0.3,
            0.5,
            0.8,
            1.0,
            1.5,
            2.0,
            2.5,
            5.0,
            10.0,
            15.0,
            20.0,
            30.0,
            40.0,
            50.0,
            60.0,
            120.0,
            240.0,
            480.0,
            960.0,
            1920.0,
            7680.0,
        ]
        assert list(c.histogram_e2e_time_request._upper_bounds) == default_e2e + [float("inf")]

    def test_empty_bucket_list_raises(self):
        """Passing an empty list must raise ValueError at MetricsCollector init."""
        labels = {"model_name": "test_model"}
        with pytest.raises(ValueError, match="must not be empty"):
            MetricsCollector(labels, e2e_request_latency_buckets=[])

    def test_unsorted_bucket_list_raises(self):
        """Passing an unsorted list must raise ValueError at MetricsCollector init."""
        labels = {"model_name": "test_model"}
        with pytest.raises(ValueError, match="must be strictly increasing"):
            MetricsCollector(labels, time_to_first_token_buckets=[1.0, 0.5, 2.0])

    def test_duplicate_bucket_values_raises(self):
        """Passing duplicate values must raise ValueError (not strictly increasing)."""
        labels = {"model_name": "test_model"}
        with pytest.raises(ValueError, match="must be strictly increasing"):
            MetricsCollector(labels, e2e_request_latency_buckets=[1.0, 1.0, 2.0])


# ---------------------------------------------------------------------------
# Tests for kvCacheIterationStats (from main — TRTLLM-11421)
# ---------------------------------------------------------------------------


def _make_kv_iter_collector() -> MetricsCollector:
    """Construct a fresh MetricsCollector for kv_iter tests.

    Uses a unique label so the autouse clean_registry fixture does not
    collide with collectors from other tests in this module.
    """
    return MetricsCollector(labels={"kv_iter_test": "true"})


class TestLogIterationStatsKvCacheIteration:
    def test_no_kv_cache_iteration_stats(self):
        """When kvCacheIterationStats is absent, new metrics should not error."""
        collector = _make_kv_iter_collector()
        stats = {"kvCacheStats": {"cacheHitRate": 0.5, "usedNumBlocks": 10, "maxNumBlocks": 20}}
        # Should not raise
        collector.log_iteration_stats(stats)

    def test_gauges_updated(self):
        """Host utilization and iter reuse rate gauges should be set."""
        collector = _make_kv_iter_collector()
        stats = {
            "kvCacheIterationStats": {
                "16": {
                    "primaryMaxNumBlocks": 100,
                    "primaryFreeNumBlocks": 60,
                    "primaryUsedNumBlocks": 40,
                    "secondaryMaxNumBlocks": 50,
                    "secondaryFreeNumBlocks": 30,
                    "secondaryUsedNumBlocks": 20,
                    "iterAllocTotalBlocks": 8,
                    "iterAllocNewBlocks": 3,
                    "iterReusedBlocks": 5,
                    "iterFullReusedBlocks": 4,
                    "iterPartialReusedBlocks": 1,
                    "iterMissedBlocks": 3,
                    "iterCacheHitRate": 0.625,
                    "iterGenAllocBlocks": 2,
                    "iterOnboardBlocks": 1,
                    "iterOnboardBytes": 4096,
                    "iterOffloadBlocks": 0,
                    "iterOffloadBytes": 0,
                }
            }
        }
        collector.log_iteration_stats(stats)

        # Host utilization = 20/50 = 0.4
        assert _get_gauge_value(collector, "kv_cache_host_utilization") == pytest.approx(0.4)
        # Iter reuse rate = 5/(5+3) = 0.625
        assert _get_gauge_value(collector, "kv_cache_iter_reuse_rate") == pytest.approx(0.625)

    def test_counters_incremented(self):
        """Counter metrics should accumulate deltas across calls."""
        collector = _make_kv_iter_collector()
        stats = {
            "kvCacheIterationStats": {
                "16": {
                    "primaryMaxNumBlocks": 100,
                    "primaryFreeNumBlocks": 60,
                    "primaryUsedNumBlocks": 40,
                    "secondaryMaxNumBlocks": 0,
                    "secondaryFreeNumBlocks": 0,
                    "secondaryUsedNumBlocks": 0,
                    "iterAllocTotalBlocks": 8,
                    "iterAllocNewBlocks": 3,
                    "iterReusedBlocks": 5,
                    "iterFullReusedBlocks": 4,
                    "iterPartialReusedBlocks": 1,
                    "iterMissedBlocks": 3,
                    "iterCacheHitRate": 0.625,
                    "iterGenAllocBlocks": 2,
                    "iterOnboardBlocks": 1,
                    "iterOnboardBytes": 4096,
                    "iterOffloadBlocks": 1,
                    "iterOffloadBytes": 2048,
                    "iterIntraDeviceCopyBlocks": 2,
                    "iterIntraDeviceCopyBytes": 8192,
                }
            }
        }

        # Read baseline counter values (may be non-zero from prior tests)
        before_reused = _get_counter_value(collector, "kv_cache_iter_reused_blocks")
        before_missed = _get_counter_value(collector, "kv_cache_iter_missed_blocks")
        before_onboard = _get_counter_value(collector, "kv_cache_onboard_bytes_total")
        before_intra_device = _get_counter_value(
            collector, "kv_cache_intra_device_copy_bytes_total"
        )

        # First call
        collector.log_iteration_stats(stats)
        assert _get_counter_value(
            collector, "kv_cache_iter_reused_blocks"
        ) - before_reused == pytest.approx(5)
        assert _get_counter_value(
            collector, "kv_cache_iter_missed_blocks"
        ) - before_missed == pytest.approx(3)
        assert _get_counter_value(
            collector, "kv_cache_onboard_bytes_total"
        ) - before_onboard == pytest.approx(4096)
        assert _get_counter_value(
            collector, "kv_cache_intra_device_copy_bytes_total"
        ) - before_intra_device == pytest.approx(8192)

        # Second call — counters should accumulate further
        collector.log_iteration_stats(stats)
        assert _get_counter_value(
            collector, "kv_cache_iter_reused_blocks"
        ) - before_reused == pytest.approx(10)
        assert _get_counter_value(
            collector, "kv_cache_iter_missed_blocks"
        ) - before_missed == pytest.approx(6)
        assert _get_counter_value(
            collector, "kv_cache_onboard_bytes_total"
        ) - before_onboard == pytest.approx(8192)
        assert _get_counter_value(
            collector, "kv_cache_intra_device_copy_bytes_total"
        ) - before_intra_device == pytest.approx(16384)

    def test_multiple_windows_aggregated(self):
        """Stats from multiple window sizes should be summed."""
        collector = _make_kv_iter_collector()
        ws16 = {
            "primaryMaxNumBlocks": 50,
            "primaryFreeNumBlocks": 30,
            "primaryUsedNumBlocks": 20,
            "secondaryMaxNumBlocks": 10,
            "secondaryFreeNumBlocks": 5,
            "secondaryUsedNumBlocks": 5,
            "iterAllocTotalBlocks": 4,
            "iterAllocNewBlocks": 2,
            "iterReusedBlocks": 2,
            "iterFullReusedBlocks": 2,
            "iterPartialReusedBlocks": 0,
            "iterMissedBlocks": 2,
            "iterCacheHitRate": 0.5,
            "iterGenAllocBlocks": 1,
            "iterOnboardBlocks": 0,
            "iterOnboardBytes": 0,
            "iterOffloadBlocks": 0,
            "iterOffloadBytes": 0,
        }
        ws64 = {
            "primaryMaxNumBlocks": 50,
            "primaryFreeNumBlocks": 40,
            "primaryUsedNumBlocks": 10,
            "secondaryMaxNumBlocks": 10,
            "secondaryFreeNumBlocks": 2,
            "secondaryUsedNumBlocks": 8,
            "iterAllocTotalBlocks": 5,
            "iterAllocNewBlocks": 2,
            "iterReusedBlocks": 3,
            "iterFullReusedBlocks": 1,
            "iterPartialReusedBlocks": 2,
            "iterMissedBlocks": 2,
            "iterCacheHitRate": 0.6,
            "iterGenAllocBlocks": 0,
            "iterOnboardBlocks": 1,
            "iterOnboardBytes": 8192,
            "iterOffloadBlocks": 0,
            "iterOffloadBytes": 0,
        }
        stats = {"kvCacheIterationStats": {"16": ws16, "64": ws64}}

        # Read baseline
        before_reused = _get_counter_value(collector, "kv_cache_iter_reused_blocks")

        collector.log_iteration_stats(stats)

        # Host utilization = (5+8) / (10+10) = 13/20 = 0.65
        assert _get_gauge_value(collector, "kv_cache_host_utilization") == pytest.approx(0.65)
        # Iter reuse rate = (2+3) / (2+3+2+2) = 5/9
        assert _get_gauge_value(collector, "kv_cache_iter_reuse_rate") == pytest.approx(5 / 9)
        # Counters: delta should be 5 (2+3)
        assert _get_counter_value(
            collector, "kv_cache_iter_reused_blocks"
        ) - before_reused == pytest.approx(5)

    def test_zero_deltas_no_counter_increment(self):
        """When all deltas are zero, counters should not increment."""
        collector = _make_kv_iter_collector()
        stats = {
            "kvCacheIterationStats": {
                "16": {
                    "primaryMaxNumBlocks": 100,
                    "primaryFreeNumBlocks": 100,
                    "primaryUsedNumBlocks": 0,
                    "secondaryMaxNumBlocks": 0,
                    "secondaryFreeNumBlocks": 0,
                    "secondaryUsedNumBlocks": 0,
                    "iterAllocTotalBlocks": 0,
                    "iterAllocNewBlocks": 0,
                    "iterReusedBlocks": 0,
                    "iterFullReusedBlocks": 0,
                    "iterPartialReusedBlocks": 0,
                    "iterMissedBlocks": 0,
                    "iterCacheHitRate": 0.0,
                    "iterGenAllocBlocks": 0,
                    "iterOnboardBlocks": 0,
                    "iterOnboardBytes": 0,
                    "iterOffloadBlocks": 0,
                    "iterOffloadBytes": 0,
                }
            }
        }
        before_reused = _get_counter_value(collector, "kv_cache_iter_reused_blocks")
        before_missed = _get_counter_value(collector, "kv_cache_iter_missed_blocks")
        collector.log_iteration_stats(stats)
        assert _get_counter_value(collector, "kv_cache_iter_reused_blocks") == pytest.approx(
            before_reused
        )
        assert _get_counter_value(collector, "kv_cache_iter_missed_blocks") == pytest.approx(
            before_missed
        )
