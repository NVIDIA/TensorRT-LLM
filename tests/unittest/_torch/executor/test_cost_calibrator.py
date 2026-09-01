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
"""Unit tests for the equal-cost chunking online calibrator."""

import random

from tensorrt_llm._torch.pyexecutor.scheduler.cost_calibrator import ContextCostCalibrator

M = 8192


def make_calibrator(**kw):
    defaults = dict(
        max_num_tokens=M,
        kv_depth_threshold_percentile=50.0,
        min_samples=64,
        window=512,
        refresh_interval=64,
    )
    defaults.update(kw)
    return ContextCostCalibrator(**defaults)


def feed_synthetic(cal, n_iters, a_ms, c2_ms, rng, outlier_rate=0.03):
    """Iterations with N=M and kv uniform in [0, 600k]; T from the model."""
    for _ in range(n_iters):
        kv = rng.uniform(0, 600_000)
        s = M * (kv + M / 2)
        t = a_ms + c2_ms * s
        t *= rng.uniform(0.98, 1.02)  # measurement noise
        if rng.random() < outlier_rate:
            t *= 10.0  # eviction/onboard stall
        cal.observe_chunk(int(kv), M)
        cal.observe_iteration(M, s, t)
        cal.maybe_refresh()


class TestContextCostCalibrator:

    def test_converges_with_noise_and_outliers(self):
        # Ground truth kv_cost_offset = a / (c2 * M) = 1638.4 / (1e-6 * 8192) = 200k.
        cal = make_calibrator()
        feed_synthetic(cal, 600, a_ms=1638.4, c2_ms=1e-6, rng=random.Random(0))
        assert cal.budget is not None
        assert 100_000 < cal.kv_cost_offset < 400_000
        # kv uniform in [0, 600k] -> p50 near 300k (16k-bucket resolution).
        assert 250_000 < cal.kv_depth_threshold < 350_000

    def test_stays_off_for_linear_workload(self):
        # c2 = 0: iteration time does not depend on KV depth.
        cal = make_calibrator()
        feed_synthetic(cal, 600, a_ms=1638.4, c2_ms=0.0, rng=random.Random(1))
        assert cal.budget is None

    def test_stays_off_below_min_samples(self):
        cal = make_calibrator(min_samples=256)
        feed_synthetic(cal, 100, a_ms=1638.4, c2_ms=1e-6, rng=random.Random(2))
        assert cal.budget is None

    def test_hysteresis_suppresses_jitter(self):
        cal = make_calibrator()
        feed_synthetic(cal, 300, a_ms=1638.4, c2_ms=1e-6, rng=random.Random(3))
        first = cal.budget
        assert first is not None
        # Same distribution again: refreshes fire but the budget stays put.
        feed_synthetic(cal, 300, a_ms=1638.4, c2_ms=1e-6, rng=random.Random(4))
        assert abs(cal.budget - first) / first < 0.15

    def test_kv_percentile_from_histogram(self):
        cal = make_calibrator(kv_depth_threshold_percentile=25.0)
        for kv in (0, 100_000, 200_000, 300_000):
            cal.observe_chunk(kv, 1000)  # equal token mass in four buckets
        ref = cal._kv_percentile(25.0)
        assert ref < 100_000  # p25 falls inside the first bucket
