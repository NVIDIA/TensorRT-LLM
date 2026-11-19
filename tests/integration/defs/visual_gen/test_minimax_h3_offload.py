# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch
from test_common.visual_gen.h3_rollout import h3_modeling_v1 as h3_modeling_v1

from defs.visual_gen import minimax_h3_offload_cases as offload

pytestmark = pytest.mark.usefixtures("h3_modeling_v1")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_minimax_h3_offload_memory_and_parity(tmp_path: Path) -> None:
    offload.check_offload_memory_and_parity(tmp_path)
