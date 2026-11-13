# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the Kimi K3 family's ``mnnvl_workspace`` on four ranks of one tray.

The steps are ``_mnnvl_mcast_workspace_op_matrix.py`` beside this file, its own W-rank launcher (one sequence of
collective steps, so one job); see ``_rank_job`` for why that is left intact.
"""

from pathlib import Path

__extra_import_path__ = ["../../comm"]
import _rank_job  # noqa: E402
import pytest
import torch

assert torch.cuda.is_available(), "multicast memory requires CUDA devices"

if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
    pytest.skip("multicast memory is exercised on sm_100 and sm_103", allow_module_level=True)

if torch.cuda.device_count() < _rank_job.WORLD_SIZE:
    pytest.skip(
        f"the check runs {_rank_job.WORLD_SIZE} ranks, one per device; "
        f"{torch.cuda.device_count()} visible",
        allow_module_level=True,
    )


# The case starts its own W-rank mpirun over the visible devices; under xdist several workers would fight for them.
@pytest.mark.no_xdist
def test_mnnvl_mcast_workspace_op_matrix() -> None:
    _rank_job.run("mnnvl_mcast_workspace", directory=Path(__file__).resolve().parent)
