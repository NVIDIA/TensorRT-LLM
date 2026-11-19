# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from test_common.visual_gen.h3_rollout import h3_modeling_v1 as h3_modeling_v1

from defs.visual_gen import minimax_h3_sleep_cases as sleep

pytestmark = pytest.mark.usefixtures("h3_modeling_v1")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("release_host_backup", [False, True])
def test_minimax_h3_sleep_preserves_video_and_audio(release_host_backup: bool) -> None:
    sleep.check_sleep_preserves_video_and_audio(release_host_backup)
