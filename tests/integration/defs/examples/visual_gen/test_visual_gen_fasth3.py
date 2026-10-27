# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastH3 example regression against a recorded TRTLLM VSA video.

This golden detects output drift; it does not establish FastVideo reference parity.
Stage the checkpoint revision recorded in the golden JSON under LLM_MODELS_ROOT.
"""

import json
from pathlib import Path

import pytest
import torch
from defs.common import venv_check_call
from defs.examples.visual_gen.visual_gen_test_utils import (
    VISUAL_GEN_LPIPS_GOLDEN_DIR,
    _assert_lpips_below_threshold,
    _golden_media_path,
    _preserve_lpips_candidate_on_failure,
    _run_lpips_eval,
)
from test_common.llm_data import get_checkpoint


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fasth3_example_lpips_against_golden(
    request, tmp_path, _visual_gen_deps, llm_root, llm_venv
):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("FastH3 TRTLLM VSA requires Blackwell")
    metadata_path = Path(VISUAL_GEN_LPIPS_GOLDEN_DIR) / "fasth3_lpips_golden_video.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    golden_path = _golden_media_path(tmp_path, metadata["video"], "FastH3 TRTLLM regression golden")
    model_path = get_checkpoint(metadata["model"])
    output_path = tmp_path / "fasth3_output.mp4"
    script_path = Path(llm_root) / "examples/visual_gen/models/fasth3.py"
    config_path = Path(llm_root) / "examples/visual_gen/configs/fasth3-bf16-trtllm-vsa-1gpu.yaml"
    # Exercise the example's published schedule, seed and geometry defaults.
    venv_check_call(
        llm_venv,
        [
            str(script_path),
            "--model",
            model_path,
            "--visual_gen_args",
            str(config_path),
            "--prompt",
            metadata["prompt"],
            "--output_path",
            str(output_path),
        ],
    )
    assert output_path.is_file(), "FastH3 example did not save a video"
    score = _run_lpips_eval(
        tmp_path, "fasth3", "video", metadata["prompt"], golden_path, output_path
    )
    threshold = metadata["lpips_threshold"]
    _preserve_lpips_candidate_on_failure(request, score, threshold, output_path, metadata["video"])
    _assert_lpips_below_threshold(score, threshold)
