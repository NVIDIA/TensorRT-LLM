# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the mnnvl_allgather_split op's certification matrix.

The matrix is ``_mnnvl_allgather_split_op_matrix.py`` beside this file, its own W-rank launcher (call sequences
over caller-owned workspaces, so one job, not independent cases); see ``_rank_job`` for why that is left intact.
"""

from pathlib import Path

__extra_import_path__ = ["../../comm"]
import _rank_job  # noqa: E402
import pytest
import torch

assert torch.cuda.is_available(), "mnnvl_allgather_split requires CUDA devices"

if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
    pytest.skip("mnnvl_allgather_split is certified on sm_100 and sm_103", allow_module_level=True)


# Each case starts its own W-rank mpirun over the visible devices; under xdist several workers would fight for them.
@pytest.mark.no_xdist
def test_mnnvl_allgather_split_op_matrix() -> None:
    _rank_job.run("mnnvl_allgather_split", directory=Path(__file__).resolve().parent)
