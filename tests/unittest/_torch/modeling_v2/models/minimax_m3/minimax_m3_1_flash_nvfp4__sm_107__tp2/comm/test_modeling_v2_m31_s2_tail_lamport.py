# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the TP2W_S2_TAIL Lamport workspace check.

The check is ``_m31_s2_tail_lamport_op_matrix.py`` beside this file, its own 2-rank launcher; see ``_rank_job``. It
alternates the fused MoE tail with the built-in one-shot all-reduce on one workspace, with a late rank per round, and
requires every sum exact.
"""

import _rank_job
import pytest
import torch

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
        reason="the sm_107 TP2 target runs on SM107 only",
    ),
    pytest.mark.skipif(
        torch.cuda.device_count() < _rank_job.WORLD_SIZE,
        reason=f"certified at world size {_rank_job.WORLD_SIZE}",
    ),
]


# Spawns 2 rank processes over the visible devices; under xdist workers would fight for them.
@pytest.mark.no_xdist
def test_m31_s2_tail_lamport_clear_size() -> None:
    out = _rank_job.run("m31_s2_tail_lamport")
    assert "alternating calls exact" in out, f"the launcher reported no result:\n{out}"
