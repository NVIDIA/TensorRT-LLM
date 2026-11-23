# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runs ``_allgather__r20_op_matrix.py`` (4 ranks) and asserts its exit code."""

from pathlib import Path

import pytest
import torch

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2/comm"]
import _rank_job  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "allgather requires CUDA devices"


# Spawns its own ranks over the visible devices; xdist workers would fight for them.
@pytest.mark.no_xdist
def test_allgather__r20_op_matrix() -> None:
    _rank_job.run("allgather__r20", directory=Path(__file__).resolve().parent)
