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

from tensorrt_llm.bench.dataclasses.reporting import PerfItemTuple, StatsKeeper
from tensorrt_llm.bench.dataclasses.statistics import PercentileStats


@pytest.mark.parametrize(
    "values, weights, expected_average",
    [
        # No weights -> plain arithmetic mean (backward compatible).
        ([1.0, 2.0, 3.0, 4.0], None, 2.5),
        # Weighted mean: sum(w * v) / sum(w).
        ([0.5, 1.0], [1.0, 3.0], 0.875),
        # The heavy (large-weight) value pulls the average toward itself
        # instead of being equally weighted (0.75 -> 0.63).
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


def _perf_item(rid, num_out, decoding_iteration):
    """Build a final PerfItemTuple for a single request.

    ``num_total_output_tokens`` is ``len(tokens)``, so ``tokens`` is sized to
    ``num_out``; ``decoding_iteration`` maps onto ``RequestRecord.decode_iteration``.
    """
    return PerfItemTuple(
        start_timestamp=0,
        end_timestamp=1000,
        request_id=rid,
        num_input_tokens=8,
        response_is_final=True,
        error=False,
        tokens=list(range(num_out)),
        decoding_iteration=decoding_iteration,
        time_on_first_token=100,
    )


def test_acceptance_length_weighted_by_iterations():
    """AR/AL averages are weighted by per-request decoding iterations.

    AL_i = num_total_output_tokens / (decode_iteration + 1), weighted by the
    iteration count (decode_iteration + 1). A short request has high AL but few
    iterations; a long request has low AL but many iterations, so it dominates.
    """
    keeper = StatsKeeper()
    # short request: AL = 10 / 2 = 5.0 over 2 iterations
    keeper.register_request_perf_item(_perf_item(0, num_out=10, decoding_iteration=1))
    # long request: AL = 300 / 200 = 1.5 over 200 iterations
    keeper.register_request_perf_item(_perf_item(1, num_out=300, decoding_iteration=199))

    stats = keeper.generate_statistics_summary(max_draft_tokens=3, batch_size=1)

    # iteration-weighted mean: weights are (decode_iteration + 1).
    weighted = (2 * 5.0 + 200 * 1.5) / (2 + 200)
    assert stats.acceptance_length_percentiles.average == pytest.approx(weighted)

    # Design intent: under iteration weighting the per-request AL average
    # equals the globally-computed acceptance_length
    # (= sum(output_tokens) / total_decoding_iterations). This would fail under
    # output-token weighting.
    assert stats.acceptance_length_percentiles.average == pytest.approx(stats.acceptance_length)
