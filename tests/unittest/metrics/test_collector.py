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
"""Unit tests for MetricsCollector.log_iteration_stats with kvCacheIterationStats."""

import pytest

prometheus_client = pytest.importorskip("prometheus_client")

from tensorrt_llm.metrics.collector import MetricsCollector  # noqa: E402

# Use a single module-level collector to avoid re-registering Prometheus metrics
# (Prometheus does not allow duplicate metric names in the same process).
_collector = MetricsCollector(labels={"test": "true"})


def _make_collector() -> MetricsCollector:
    """Return the shared collector instance."""
    return _collector


def _get_gauge_value(collector, metric_name: str):
    """Get the current value of a Prometheus gauge."""
    metric = getattr(collector, metric_name)
    return metric.labels(**collector.labels)._value.get()


def _get_counter_value(collector, metric_name: str):
    """Get the current value of a Prometheus counter."""
    metric = getattr(collector, metric_name)
    return metric.labels(**collector.labels)._value.get()


class TestLogIterationStatsKvCacheIteration:
    def test_no_kv_cache_iteration_stats(self):
        """When kvCacheIterationStats is absent, new metrics should not error."""
        collector = _make_collector()
        stats = {"kvCacheStats": {"cacheHitRate": 0.5, "usedNumBlocks": 10, "maxNumBlocks": 20}}
        # Should not raise
        collector.log_iteration_stats(stats)

    def test_gauges_updated(self):
        """Host utilization and iter reuse rate gauges should be set."""
        collector = _make_collector()
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
        collector = _make_collector()
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
        before_reused = _get_counter_value(collector, "kv_cache_reused_blocks_total")
        before_missed = _get_counter_value(collector, "kv_cache_missed_blocks_total")
        before_onboard = _get_counter_value(collector, "kv_cache_onboard_bytes_total")
        before_intra_device = _get_counter_value(
            collector, "kv_cache_intra_device_copy_bytes_total"
        )

        # First call
        collector.log_iteration_stats(stats)
        assert _get_counter_value(
            collector, "kv_cache_reused_blocks_total"
        ) - before_reused == pytest.approx(5)
        assert _get_counter_value(
            collector, "kv_cache_missed_blocks_total"
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
            collector, "kv_cache_reused_blocks_total"
        ) - before_reused == pytest.approx(10)
        assert _get_counter_value(
            collector, "kv_cache_missed_blocks_total"
        ) - before_missed == pytest.approx(6)
        assert _get_counter_value(
            collector, "kv_cache_onboard_bytes_total"
        ) - before_onboard == pytest.approx(8192)
        assert _get_counter_value(
            collector, "kv_cache_intra_device_copy_bytes_total"
        ) - before_intra_device == pytest.approx(16384)

    def test_multiple_windows_aggregated(self):
        """Stats from multiple window sizes should be summed."""
        collector = _make_collector()
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
        before_reused = _get_counter_value(collector, "kv_cache_reused_blocks_total")

        collector.log_iteration_stats(stats)

        # Host utilization = (5+8) / (10+10) = 13/20 = 0.65
        assert _get_gauge_value(collector, "kv_cache_host_utilization") == pytest.approx(0.65)
        # Iter reuse rate = (2+3) / (2+3+2+2) = 5/9
        assert _get_gauge_value(collector, "kv_cache_iter_reuse_rate") == pytest.approx(5 / 9)
        # Counters: delta should be 5 (2+3)
        assert _get_counter_value(
            collector, "kv_cache_reused_blocks_total"
        ) - before_reused == pytest.approx(5)

    def test_zero_deltas_no_counter_increment(self):
        """When all deltas are zero, counters should not increment."""
        collector = _make_collector()
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
        before_reused = _get_counter_value(collector, "kv_cache_reused_blocks_total")
        before_missed = _get_counter_value(collector, "kv_cache_missed_blocks_total")
        collector.log_iteration_stats(stats)
        assert _get_counter_value(collector, "kv_cache_reused_blocks_total") == pytest.approx(
            before_reused
        )
        assert _get_counter_value(collector, "kv_cache_missed_blocks_total") == pytest.approx(
            before_missed
        )
