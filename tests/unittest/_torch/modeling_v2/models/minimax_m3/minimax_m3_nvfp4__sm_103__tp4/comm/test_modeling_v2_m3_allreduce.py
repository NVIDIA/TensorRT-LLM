# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the ``m3_allreduce`` certification matrix.

The matrix is ``_m3_allreduce_op_matrix.py`` beside this file, its own 4-rank launcher; see ``_rank_job``. It drives
every cell of the entry against its reference, then requires the fused draft loop's boundaries built on it (plain
all-reduce, then the fused add + RMSNorm) to equal stock Eagle3's bit for bit under every strategy, and reports both
against an fp64 reference as a warning, one line per strategy and case.
"""

import warnings

import _rank_job
import pytest
import torch

assert torch.cuda.is_available(), "the fused all-reduce + RMSNorm requires CUDA devices"

_RESULT_TAG = "[ulp]"


# Spawns 4 rank processes over the visible devices; under xdist workers would fight for them.
@pytest.mark.no_xdist
def test_m3_allreduce_op_matrix() -> None:
    out = _rank_job.run("m3_allreduce")
    lines = [line for line in out.splitlines() if line.startswith(_RESULT_TAG)]
    assert lines, f"the launcher reported no measurements:\n{out}"
    warnings.warn(
        "draft boundary (= stock Eagle3) vs fp64, max BF16 ulps\n" + "\n".join(lines), stacklevel=1
    )
