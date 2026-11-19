# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastH3 migration quality vs the matching pre-refactor eight-step baseline."""

import json
import math
import os
from pathlib import Path

import pytest
import torch
from test_common.llm_data import get_checkpoint

from defs.examples.visual_gen.visual_gen_test_utils import _multi_resolution_log_stft_distance
from defs.visual_gen.minimax_h3_quality_cases import _mean_lpips_distance
from defs.visual_gen.minimax_h3_v2.fasth3.quality_case import generate


@pytest.mark.parametrize("backend", ["TRTLLM", "CUTEDSL"])
def test_video_audio_against_baseline(backend):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("FastH3 VSA quality requires SM100 or SM103")
    checkpoint = Path(
        os.environ.get("FASTH3_CHECKPOINT") or get_checkpoint("FastVideo-FastH3-8-Step-V2")
    )
    refs = Path(os.environ.get("FASTH3_QA_REFERENCES") or get_checkpoint("FastH3-QA-references"))
    manifest = json.loads((refs / "8-step-v2.json").read_text())
    case = manifest[backend]
    assert manifest["baseline_commit"] and manifest["checkpoint_revision"]
    reference = torch.load(refs / case["artifact"], map_location="cpu", weights_only=True)
    actual = generate(checkpoint, backend)
    assert actual["case"] == reference["case"]
    assert actual["target"].endswith(".fasth3.pipeline.FastH3Pipeline")
    for key in ("video", "audio"):
        assert actual[key].shape == reference[key].shape
        assert torch.isfinite(actual[key]).all() and torch.isfinite(reference[key]).all()
    expected_video = reference["video"].permute(0, 1, 4, 2, 3).float().div(255)
    video = _mean_lpips_distance(actual["video"], expected_video)
    audio = _multi_resolution_log_stft_distance(actual["audio"], reference["audio"])
    assert math.isfinite(case["max_lpips"]) and 0 <= video <= case["max_lpips"]
    assert math.isfinite(case["max_audio_log_stft"]) and 0 <= audio <= case["max_audio_log_stft"]
