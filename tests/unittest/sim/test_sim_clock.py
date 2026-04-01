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
