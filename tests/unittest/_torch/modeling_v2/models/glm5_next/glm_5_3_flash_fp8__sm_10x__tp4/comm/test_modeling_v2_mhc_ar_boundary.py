# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the fused all-reduce + mHC boundary's 4-rank matrix.

The matrix is ``_mhc_ar_boundary_op_matrix.py`` beside this file, its own 4-rank launcher;
``_rank_job`` (in the shared comm test directory) starts it in a fresh interpreter and turns
its exit code into an assertion.
"""

from pathlib import Path

import pytest
import torch

__extra_import_path__ = ["../../../../comm"]
import _rank_job  # noqa: E402 — needs the path declared above


# Spawns 4 rank processes over the visible devices. Under xdist several workers would fight for
# the same GPUs, so this must run alone, like the shared collective entries.
@pytest.mark.no_xdist
def test_mhc_ar_boundary_op_matrix() -> None:
    devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if devices < _rank_job.WORLD_SIZE:
        pytest.skip(
            f"the op runs on {_rank_job.WORLD_SIZE} ranks in one node; {devices} device(s) visible"
        )
    _rank_job.run("mhc_ar_boundary", directory=Path(__file__).resolve().parent)
