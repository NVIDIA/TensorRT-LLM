# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the glm_draft_sample_glue entry's multi-rank certification.

The cells, a draft loop of the target's shape (eager and as CUDA-graph replays) and the guards run inside
`_draft_step_op_matrix.py` beside this file, one process per rank over a 4-rank group; see that file for the
16-rank run over four nodes.
"""

__extra_import_path__ = ["."]
import _draft_step_rank_job
import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_107__tp16.ops.spec.draft_sample_glue import (  # noqa: E501
    glm_draft_sample_glue as op,
)
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    get_sm_version() != 107, reason="the GLM-5.2 tp16 target's ops are certified on sm_107"
)


# Each run spawns its own 4 rank processes over the visible devices: under xdist several workers would fight
# for the same GPUs.
@pytest.mark.no_xdist
def test_draft_sample_glue_op_matrix() -> None:
    assert op.CELLS, "the entry declares no cells"
    _draft_step_rank_job.run("draft_sample_glue", expected_cells=len(op.CELLS))
