# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for sim_metrics."""

import pytest

from tensorrt_llm._torch.pyexecutor.sim_metrics import (SimIterationRecord,
                                                         SimRequestStats,
                                                         calc_sim_metrics,
                                                         percentile)


class TestSimRequestStats:

    def test_ttft(self):
        s = SimRequestStats(request_id=1, input_length=10)
        s.gen_token_times = [0.010, 0.015, 0.020]
        assert s.ttft_s == pytest.approx(0.010)

    def test_itl(self):
        s = SimRequestStats(request_id=1, input_length=10)
        s.gen_token_times = [0.010, 0.015, 0.020]
        assert s.itl_s == [pytest.approx(0.005), pytest.approx(0.005)]

    def test_tpot(self):
        s = SimRequestStats(request_id=1, input_length=10)
        s.gen_token_times = [0.010, 0.015, 0.020]
        assert s.tpot_s == pytest.approx(0.005)

    def test_e2e(self):
        s = SimRequestStats(request_id=1, input_length=10)
        s.gen_token_times = [0.010, 0.015, 0.020]
        assert s.e2e_s == pytest.approx(0.020)

    def test_empty_tokens(self):
        s = SimRequestStats(request_id=1, input_length=10)
        assert s.ttft_s == 0.0
        assert s.itl_s == []
        assert s.tpot_s == 0.0
        assert s.e2e_s == 0.0

    def test_single_token(self):
        s = SimRequestStats(request_id=1, input_length=10)
        s.gen_token_times = [0.010]
        assert s.ttft_s == pytest.approx(0.010)
        assert s.itl_s == []
        assert s.tpot_s == 0.0
        assert s.e2e_s == pytest.approx(0.010)


class TestSimIterationRecord:

    def test_fields(self):
        r = SimIterationRecord(
            iteration=1,
            sim_time_s=0.010,
            predicted_duration_s=0.010,
            num_context_requests=1,
            num_context_tokens=128,
            num_generation_requests=0)
        assert r.iteration == 1
        assert r.predicted_duration_s == 0.010


class TestPercentile:

    def test_p50(self):
        assert percentile([1, 2, 3, 4, 5], 50) == pytest.approx(3.0)

    def test_p0(self):
        assert percentile([1, 2, 3], 0) == pytest.approx(1.0)

    def test_p100(self):
        assert percentile([1, 2, 3], 100) == pytest.approx(3.0)

    def test_single_element(self):
        assert percentile([42], 50) == pytest.approx(42.0)

    def test_empty_returns_zero(self):
        assert percentile([], 50) == 0.0


class TestCalcSimMetrics:

    def _make_stats(self):
        """Helper: 1 request, prefill=10ms, 7 decodes at 5ms each."""
        s = SimRequestStats(request_id=0, input_length=5, output_length=8)
        # Prefill at t=0.010, then decodes at 0.015, 0.020, ..., 0.045
        s.gen_token_times = [
            0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045
        ]
        return {0: s}

    def test_ttft(self):
        m = calc_sim_metrics(self._make_stats(), [])
        assert m["mean_ttft_ms"] == pytest.approx(10.0)

    def test_tpot(self):
        m = calc_sim_metrics(self._make_stats(), [])
        assert m["mean_tpot_ms"] == pytest.approx(5.0)

    def test_itl(self):
        m = calc_sim_metrics(self._make_stats(), [])
        assert m["mean_itl_ms"] == pytest.approx(5.0)

    def test_e2e(self):
        m = calc_sim_metrics(self._make_stats(), [])
        assert m["mean_e2e_latency_ms"] == pytest.approx(45.0)

    def test_throughput(self):
        m = calc_sim_metrics(self._make_stats(), [])
        # 8 tokens / 0.045s = 177.78
        assert m["output_throughput"] == pytest.approx(177.78, rel=0.01)

    def test_completed(self):
        m = calc_sim_metrics(self._make_stats(), [])
        assert m["completed"] == 1
        assert m["total_output"] == 8
        assert m["total_input"] == 5

    def test_empty(self):
        m = calc_sim_metrics({}, [])
        assert m["completed"] == 0
        assert m["output_throughput"] == 0.0
