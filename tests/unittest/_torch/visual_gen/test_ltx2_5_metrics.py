# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""L1 metric oracle: transparent BF16 comparisons must detect active tensor drift."""

import math

import pytest
import torch

__extra_import_path__ = ["~/scripts/visualgen_eval"]

from compare_ltx2_5_traces import tensor_metrics
from compare_ltx25_quantized import audio_spectra, validate_policy

pytestmark = pytest.mark.cpu_only


def test_metric_detects_localized_drift() -> None:
    expected = torch.tensor([0, 1, 2, -3], dtype=torch.bfloat16)
    actual = expected.clone()
    actual[1] += 0.125
    result = tensor_metrics(actual, expected)
    assert result["max_abs"] == 0.125
    assert result["rmse"] == 0.0625
    assert result["normalized_rmse"] == pytest.approx(0.125 / math.sqrt(14))
    assert not result["pass"]


def test_small_bf16_rounding_does_not_require_bitwise_identity() -> None:
    expected = torch.tensor([0, 1, 2, -3], dtype=torch.bfloat16)
    actual = expected.clone()
    actual[2] += 0.015625
    result = tensor_metrics(actual, expected)
    assert result["max_abs"] > 0
    assert not result["bitwise_equal"]
    assert result["pass"]


def test_zero_norm_and_invalid_tensors_are_explicit() -> None:
    zeros = torch.zeros(4, dtype=torch.bfloat16)
    assert tensor_metrics(zeros, zeros)["pass"]
    assert not tensor_metrics(zeros + 1, zeros)["pass"]
    assert not tensor_metrics(zeros.float(), zeros)["pass"]
    assert not tensor_metrics(zeros[:2], zeros)["pass"]
    assert not tensor_metrics(zeros + float("nan"), zeros)["pass"]


def test_audio_spectra_preserve_gain_error() -> None:
    """L1: a known gain change has analytic magnitude and logarithmic errors."""
    generator = torch.Generator().manual_seed(42)
    reference = torch.randn(2, 4096, generator=generator) * 0.02
    for row in audio_spectra(reference, reference).values():
        assert row["spectral_convergence"] == 0
        assert row["energy_weighted_log_error"] == 0
    for row in audio_spectra(reference * 1.1, reference).values():
        assert row["spectral_convergence"] == pytest.approx(0.1, abs=1e-6)
        assert row["energy_weighted_log_error"] == pytest.approx(math.log(1.1), abs=1e-6)


@pytest.mark.parametrize("invalid", [True, -0.01, float("nan"), float("inf"), "0.05"])
def test_quality_policy_rejects_invalid_limits(invalid) -> None:
    policy = {
        "limits": {
            "video_lpips": invalid,
            "audio_spectral_convergence": 0.1,
            "audio_energy_weighted_log_error": 0.1,
        }
    }
    with pytest.raises(ValueError, match="Quality limit"):
        validate_policy(policy)


def test_quality_policy_requires_all_metrics() -> None:
    with pytest.raises(ValueError, match="exactly"):
        validate_policy({"limits": {"video_lpips": 0.05}})
