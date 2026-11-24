# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runs ``_reducescatter__r21_op_matrix.py`` (4 ranks) and asserts its exit code."""

from pathlib import Path

import pytest
import torch

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2/comm"]
import _rank_job  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "reducescatter requires CUDA devices"


# Spawns its own ranks over the visible devices; xdist workers would fight for them.
@pytest.mark.no_xdist
def test_reducescatter__r21_op_matrix() -> None:
    _rank_job.run("reducescatter__r21", directory=Path(__file__).resolve().parent)
