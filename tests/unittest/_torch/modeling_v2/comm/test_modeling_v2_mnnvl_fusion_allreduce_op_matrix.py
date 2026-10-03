# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the mnnvl_fusion_allreduce op's certification matrix.

The matrix is ``_mnnvl_fusion_allreduce_op_matrix.py`` beside this file, its own W-rank launcher (call sequences
over caller-owned workspaces, so one job, not independent cases); see ``_rank_job`` for why that is left intact.
"""

import _rank_job
import pytest
import torch

assert torch.cuda.is_available(), "mnnvl_fusion_allreduce requires CUDA devices"

if torch.cuda.get_device_capability() != (10, 0):
    # The entry is certified on sm_100 (GB200) only; see its contract's receipts.
    pytest.skip("mnnvl_fusion_allreduce is certified on sm_100 only", allow_module_level=True)


# Each case starts its own W-rank mpirun over the visible devices; under xdist several workers would fight for them.
@pytest.mark.no_xdist
def test_mnnvl_fusion_allreduce_op_matrix() -> None:
    _rank_job.run("mnnvl_fusion_allreduce")
