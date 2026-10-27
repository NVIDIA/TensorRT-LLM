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
        "video": torch.zeros(9, 16, 16, 3),
        "audio": torch.zeros(2, 4),
    }


@pytest.mark.parametrize("corruption", ["seed", "missing_step"])
def test_incomplete_or_mismatched_reference_is_rejected(corruption):
    reference = _trace()
    candidate = copy.deepcopy(reference)
    if corruption == "seed":
        candidate["recipe"]["seed"] = 17
    else:
        candidate["steps"].pop()
    with pytest.raises(ValueError):
        compare(candidate, reference)


def test_low_lpips_cannot_hide_intermediate_divergence(monkeypatch):
    import compare_ltx2_5_traces as validation

    class IdenticalVideoMetric:
        def eval(self):
            return self

        def __call__(self, candidate, reference):
            return torch.zeros(1)

    monkeypatch.setattr(validation.lpips, "LPIPS", lambda **kwargs: IdenticalVideoMetric())
    reference = _trace()
    candidate = copy.deepcopy(reference)
    candidate["steps"][10]["next"].add_(1)
    result = compare(candidate, reference)
    assert result["lpips"] == 0
    assert not result["steps_exact"]
    assert not result["pass"]
