# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the GLM hist_topk entry's certification matrix.

The matrix itself is ``_hist_topk_op_matrix.py`` beside this file, its own 4-rank launcher (see
``_glm_rank_job`` for why it is left intact). It drives the entry's 4-rank cells; the 16-rank cells
need the 4-node launch its docstring describes.
"""

import _glm_rank_job
import pytest
import torch

from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    get_sm_version() != 107 or torch.cuda.device_count() < _glm_rank_job.WORLD_SIZE,
    reason="requires SM107 and at least four visible GPUs",
)

if not torch.cuda.is_available():
    pytest.skip("hist_topk requires CUDA devices", allow_module_level=True)


# Each case spawns its own 4 rank processes over the visible devices; under xdist several
# workers would fight for the same GPUs.
@pytest.mark.no_xdist
def test_hist_topk_op_matrix() -> None:
    _glm_rank_job.run("hist_topk")
