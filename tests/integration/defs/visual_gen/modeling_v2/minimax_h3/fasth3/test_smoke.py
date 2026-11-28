# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest
import torch
from test_common.llm_data import get_checkpoint

from defs.visual_gen.modeling_v2.minimax_h3.fasth3.quality_case import generate


def test_released_checkpoint_runs_vsa():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("FastH3 VSA requires SM100 or SM103")
    checkpoint = Path(
        os.environ.get("FASTH3_CHECKPOINT") or get_checkpoint("FastVideo-FastH3-8-Step-V2")
    )
    actual = generate(checkpoint, "TRTLLM", height=128, width=128)
    assert actual["target"].endswith(".fasth3.pipeline.FastH3Pipeline")
    assert actual["video"].shape == (1, 124, 128, 128, 3)
    assert torch.isfinite(actual["video"]).all() and actual["video"].amax() > actual["video"].amin()
    assert torch.isfinite(actual["audio"]).all() and actual["audio"].abs().amax() > 0
