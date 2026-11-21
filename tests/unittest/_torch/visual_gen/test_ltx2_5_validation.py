# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Prevent incomplete or mismatched qualification traces from reporting green."""

import copy

import pytest
import torch

__extra_import_path__ = ["~/scripts/visualgen_eval"]

from compare_ltx2_5_traces import compare

pytestmark = pytest.mark.cpu_only


def _trace():
    return {
        "recipe": {"height": 256, "width": 256, "num_frames": 9, "seed": 42},
        "steps": [
            {"latent": torch.zeros(1), "next": torch.ones(1), "sigma": 1.0} for _ in range(22)
        ],
        "contexts": {"video": torch.ones(1), "audio": torch.ones(1)},
        "stage_outputs": {
            "stage_1": {"video": torch.ones(1), "audio": torch.ones(1)},
            "upsampled_video": torch.ones(1),
            "stage_2": {"video": torch.ones(1), "audio": torch.ones(1)},
        },
        "initializations": [
            {
                key: torch.zeros(1)
                for key in ("noise", "initial_latent", "clean_latent", "denoise_mask")
            }
            for _ in range(4)
        ],
        "video": torch.zeros(9, 16, 16, 3),
        "audio": torch.zeros(2, 4),
        "sample_rate": 48000,
    }


@pytest.mark.parametrize("corruption", ["seed", "missing_step", "missing_stage", "missing_audio"])
def test_incomplete_or_mismatched_reference_is_rejected(corruption):
    reference = _trace()
    candidate = copy.deepcopy(reference)
    if corruption == "seed":
        candidate["recipe"]["seed"] = 17
    elif corruption == "missing_step":
        candidate["steps"].pop()
    elif corruption == "missing_stage":
        candidate["stage_outputs"].pop("upsampled_video")
    else:
        candidate["stage_outputs"]["stage_2"].pop("audio")
    with pytest.raises(ValueError):
        compare(candidate, reference)


@pytest.mark.parametrize("side", ["candidate", "reference"])
def test_empty_upsampler_output_is_rejected(side: str) -> None:
    traces = {"candidate": _trace(), "reference": _trace()}
    traces[side]["stage_outputs"]["upsampled_video"] = {}
    with pytest.raises(ValueError, match="Expected upsampled_video to be a tensor"):
        compare(traces["candidate"], traces["reference"], quality=False)


def test_complete_trace_includes_upsampler_metric() -> None:
    result = compare(_trace(), _trace(), quality=False)
    assert result["pass"]
    assert result["tensors"]["stage_outputs.upsampled_video"]["bitwise_equal"]


def test_audio_sample_rate_mismatch_is_rejected() -> None:
    reference = _trace()
    candidate = copy.deepcopy(reference)
    candidate["sample_rate"] = 24000
    with pytest.raises(ValueError, match="audio sample rates differ"):
        compare(candidate, reference, quality=False)


@pytest.mark.parametrize("side", ["candidate", "reference"])
@pytest.mark.parametrize("rate", [None, 0, -48000, True, 48000.0, "48000"])
def test_invalid_audio_sample_rate_is_rejected(side: str, rate: object) -> None:
    traces = {"candidate": _trace(), "reference": _trace()}
    traces[side]["sample_rate"] = rate
    with pytest.raises(ValueError, match="sample_rate to be a positive integer"):
        compare(traces["candidate"], traces["reference"], quality=False)


@pytest.mark.parametrize("side", ["candidate", "reference"])
def test_missing_audio_sample_rate_is_rejected(side: str) -> None:
    traces = {"candidate": _trace(), "reference": _trace()}
    traces[side].pop("sample_rate")
    with pytest.raises(ValueError, match="sample_rate to be a positive integer"):
        compare(traces["candidate"], traces["reference"], quality=False)


def test_low_lpips_cannot_hide_intermediate_divergence(monkeypatch):
    import lpips

    class IdenticalVideoMetric:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, candidate, reference):
            return torch.zeros(1)

    monkeypatch.setattr(lpips, "LPIPS", lambda **kwargs: IdenticalVideoMetric())
    reference = _trace()
    candidate = copy.deepcopy(reference)
    candidate["steps"][10]["next"].add_(1)
    result = compare(candidate, reference)
    assert result["lpips"] == 0
    assert "steps.10.next" in result["failures"]
    assert not result["pass"]
