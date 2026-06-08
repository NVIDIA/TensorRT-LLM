# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import pytest

from tensorrt_llm.bench.dataclasses.statistics import PercentileStats


@pytest.mark.parametrize(
    "values, weights, expected_average",
    [
        # No weights -> plain arithmetic mean (backward compatible).
        ([1.0, 2.0, 3.0, 4.0], None, 2.5),
        # Weighted mean: sum(w * v) / sum(w).
        ([0.5, 1.0], [1.0, 3.0], 0.875),
        # trtllm-bench AR/AL case: the long (heavy) request pulls the average
        # toward its value instead of being equally weighted (0.75 -> 0.63).
        ([0.9, 0.6], [10, 90], 0.63),
        # Uniform weights reproduce the unweighted mean.
        ([2.0, 4.0, 6.0, 8.0], [5.0, 5.0, 5.0, 5.0], 5.0),
        # Zero total weight must not divide by zero; fall back to plain mean.
        ([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], 2.0),
    ],
)
def test_weighted_average(values, weights, expected_average):
    stats = PercentileStats.from_iterable(values, weights=weights)
    assert stats.average == pytest.approx(expected_average)


def test_weights_only_affect_average():
    """Weights adjust the average but never the percentiles/min/max."""
    values = [1.0, 2.0, 3.0, 4.0]
    weighted = PercentileStats.from_iterable(values, weights=[1.0, 100.0, 1.0, 1.0])
    unweighted = PercentileStats.from_iterable(values)
    assert weighted.average != pytest.approx(unweighted.average)
    assert (
        weighted.p50,
        weighted.p90,
        weighted.p95,
        weighted.p99,
        weighted.minimum,
        weighted.maximum,
    ) == (
        unweighted.p50,
        unweighted.p90,
        unweighted.p95,
        unweighted.p99,
        unweighted.minimum,
        unweighted.maximum,
    )


def test_mismatched_weights_length_raises():
    """zip(strict=True) rejects mismatched values/weights lengths."""
    with pytest.raises(ValueError):
        PercentileStats.from_iterable([1.0, 2.0, 3.0], weights=[1.0, 2.0])
