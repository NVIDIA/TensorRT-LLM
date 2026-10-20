# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the fast draft loop's layer boundaries against stock Eagle3's.

The matrix is ``_m3_draft_ar_norm_op_matrix.py`` beside this file, its own 4-rank launcher; see ``_rank_job``. It
requires the target's drafter boundary (``modeling.M3Eagle3Worker._m3_draft_ar_norm``: the multicast all-reduce +
residual + RMSNorm in its drafter form at the rows ``DRAFT_MC_AR`` covers, else and when that declines
``M3Eagle3Draft.ar_norm`` on the strategy's all-reduce) to equal stock Eagle3's boundary (plain all-reduce, then the
``RMSNorm`` module with the residual) bit for bit under every strategy, and reports both against an fp64 reference as a
warning, one line per strategy, path and case.
"""

import warnings

import _rank_job
import pytest
import torch

assert torch.cuda.is_available(), "the fused all-reduce + RMSNorm requires CUDA devices"

_RESULT_TAG = "[ulp]"


# Spawns 4 rank processes over the visible devices; under xdist workers would fight for them.
@pytest.mark.no_xdist
def test_m3_draft_ar_norm_is_stock_eagle3() -> None:
    out = _rank_job.run("m3_draft_ar_norm")
    lines = [line for line in out.splitlines() if line.startswith(_RESULT_TAG)]
    assert lines, f"the launcher reported no measurements:\n{out}"
    warnings.warn(
        "draft boundary (= stock Eagle3) vs fp64, max BF16 ulps\n" + "\n".join(lines), stacklevel=1
    )
