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
"""Tests for SimClock."""

import pytest

from tensorrt_llm._torch.pyexecutor.sim_clock import SimClock
from tensorrt_llm._torch.pyexecutor.sim_metrics import (SimIterationRecord,
                                                         SimRequestStats)


class TestSimClock:

    def test_initial_state(self):
        clock = SimClock()
        assert clock.total_time_s == 0.0
        assert clock.num_iterations == 0

    def test_step_accumulates(self):
        clock = SimClock()
        clock.step(0.01)
        clock.step(0.01)
        clock.step(0.01)
        assert clock.total_time_s == pytest.approx(0.03)
        assert clock.num_iterations == 3

    def test_reset(self):
        clock = SimClock()
        clock.step(0.05)
        clock.step(0.05)
        clock.reset()
        assert clock.total_time_s == 0.0
        assert clock.num_iterations == 0

    def test_step_zero_duration(self):
        clock = SimClock()
        clock.step(0.0)
        assert clock.total_time_s == 0.0
        assert clock.num_iterations == 1

    def test_step_fractional(self):
        clock = SimClock()
        clock.step(0.013)
        clock.step(0.0014)
        assert clock.total_time_s == pytest.approx(0.0144)
        assert clock.num_iterations == 2


class TestSimClockRecording:

    def test_record_iteration(self):
        clock = SimClock()
        clock.step(0.010)
        clock.record_iteration(0.010,
                               num_ctx_req=1,
                               num_ctx_tokens=128,
                               num_gen_req=0)
        assert len(clock.iterations) == 1
        rec = clock.iterations[0]
        assert rec.iteration == 1
        assert rec.sim_time_s == pytest.approx(0.010)
        assert rec.predicted_duration_s == pytest.approx(0.010)
        assert rec.num_context_requests == 1

    def test_register_request(self):
        clock = SimClock()
        clock.register_request(42, input_length=10)
        assert 42 in clock.request_stats
        assert clock.request_stats[42].input_length == 10

    def test_record_token(self):
        clock = SimClock()
        clock.register_request(1, input_length=5)
        clock.step(0.010)
        clock.record_token(1)
        assert clock.request_stats[1].gen_token_times == [
            pytest.approx(0.010)
        ]
        assert clock.request_stats[1].output_length == 1

    def test_record_multiple_tokens(self):
        clock = SimClock()
        clock.register_request(1, input_length=5)
        clock.step(0.010)
        clock.record_token(1)
        clock.step(0.005)
        clock.record_token(1)
        clock.step(0.005)
        clock.record_token(1)
        stats = clock.request_stats[1]
        assert len(stats.gen_token_times) == 3
        assert stats.gen_token_times == [
            pytest.approx(0.010),
            pytest.approx(0.015),
            pytest.approx(0.020)
        ]
        assert stats.output_length == 3

    def test_metrics_property(self):
        clock = SimClock()
        clock.register_request(1, input_length=5)
        clock.step(0.010)
        clock.record_iteration(0.010, 1, 128, 0)
        clock.record_token(1)
        for _ in range(7):
            clock.step(0.005)
            clock.record_iteration(0.005, 0, 0, 1)
            clock.record_token(1)
        clock.request_stats[1].output_length = 8
        m = clock.metrics
        assert m["mean_ttft_ms"] == pytest.approx(10.0)
        assert m["mean_tpot_ms"] == pytest.approx(5.0)
        assert m["completed"] == 1

    def test_reset_clears_recordings(self):
        clock = SimClock()
        clock.register_request(1, input_length=5)
        clock.step(0.010)
        clock.record_iteration(0.010, 1, 128, 0)
        clock.record_token(1)
        clock.reset()
        assert clock.iterations == []
        assert clock.request_stats == {}
