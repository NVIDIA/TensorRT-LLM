# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for shared VisualGen benchmark metrics."""

import pytest

from tensorrt_llm.bench.benchmark.visual_gen_utils import (
    VisualGenRequestOutput,
    build_visual_gen_result_dict,
    calculate_metrics,
    print_visual_gen_results,
)


def test_denoise_metrics_are_aggregated_printed_and_serialized(capsys):
    outputs = [
        VisualGenRequestOutput(
            success=True,
            latency=8.0,
            generation=7.0,
            denoise=2.0,
        ),
        VisualGenRequestOutput(
            success=True,
            latency=10.0,
            generation=9.0,
            denoise=4.0,
        ),
        VisualGenRequestOutput(
            success=False,
            latency=100.0,
            generation=90.0,
            denoise=80.0,
            error="failed",
            exception_type="RuntimeError",
        ),
    ]

    metrics = calculate_metrics(
        outputs,
        dur_s=20.0,
        selected_percentiles=[50.0, 90.0],
    )

    assert metrics.mean_denoise == pytest.approx(3.0)
    assert metrics.median_denoise == pytest.approx(3.0)
    assert metrics.std_denoise == pytest.approx(1.0)
    assert metrics.min_denoise == pytest.approx(2.0)
    assert metrics.max_denoise == pytest.approx(4.0)
    assert metrics.percentiles_denoise[0] == pytest.approx((50.0, 3.0))
    assert metrics.percentiles_denoise[1] == pytest.approx((90.0, 3.8))

    print_visual_gen_results("offline", "test-model", 20.0, metrics)
    printed = capsys.readouterr().out
    assert " Denoise " in printed
    assert "Mean Denoise (s):" in printed
    assert "P90 Denoise (s):" in printed

    result = build_visual_gen_result_dict(
        backend="offline",
        model_id="test-model",
        benchmark_duration=20.0,
        metrics=metrics,
        outputs=outputs,
        gen_params={"seed": 42},
    )
    assert result["mean_denoise"] == pytest.approx(3.0)
    assert result["percentiles_denoise"] == pytest.approx({"p50": 3.0, "p90": 3.8})
    assert result["denoises"] == [2.0, 4.0, 80.0]


def test_unreported_denoise_metrics_remain_zero():
    outputs = [VisualGenRequestOutput(success=True, latency=1.0, generation=0.8)]

    metrics = calculate_metrics(
        outputs,
        dur_s=1.0,
        selected_percentiles=[50.0, 99.0],
    )

    assert metrics.mean_denoise == 0
    assert metrics.median_denoise == 0
    assert metrics.std_denoise == 0
    assert metrics.min_denoise == 0
    assert metrics.max_denoise == 0
    assert metrics.percentiles_denoise == [(50.0, 0.0), (99.0, 0.0)]
