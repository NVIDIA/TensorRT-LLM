# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the mnnvl_moe_finalize_hop entry's certification matrix.

The matrix is ``_mnnvl_moe_finalize_hop_op_matrix.py`` beside this file, its own 4-rank launcher
on one node; see ``_rank_job`` for why it runs as a separate job. The 16-rank c1 world is run by
hand (``_mnnvl_ranks``).
"""

import _rank_job
import pytest

from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    get_sm_version() != 107, reason="the glm_moe_dsa tp16 target is certified on sm_107 only"
)


# The job spawns 4 ranks over the visible devices; under xdist several workers would fight for
# the same GPUs.
@pytest.mark.no_xdist
def test_mnnvl_moe_finalize_hop_op_matrix() -> None:
    _rank_job.run("mnnvl_moe_finalize_hop")
