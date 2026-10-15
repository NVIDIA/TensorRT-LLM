# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the fast draft loop's layer boundaries against stock Eagle3's.

The matrix is ``_m3_draft_ar_norm_op_matrix.py`` beside this file, its own launcher; see ``_rank_job``. It requires
``spec_worker.draft_ar_norm`` to equal stock Eagle3's boundary (plain all-reduce, then the ``RMSNorm`` module with the
residual) bit for bit under every single-node strategy, and reports both against an fp64 reference as a warning, one
line per strategy and case.

The tp8 drafter runs at 8 ranks over two trays; a single-node job cannot form that group, so the matrix runs at each
world size of ``_rank_job.WORLD_SIZES`` the node's devices can form (the claim, draft boundary = stock boundary, does
not depend on the rank count) and skips the others. MNNVL, the strategy the two-tray group may take, is multi-node
only and not driven here.
"""

import warnings

import _rank_job
import pytest
import torch

assert torch.cuda.is_available(), "the fused all-reduce + RMSNorm requires CUDA devices"

_RESULT_TAG = "[ulp]"


# Spawns one rank process per device of the world size; under xdist workers would fight for them.
@pytest.mark.no_xdist
@pytest.mark.parametrize("world_size", _rank_job.WORLD_SIZES, ids=lambda w: f"world{w}")
def test_m3_draft_ar_norm_is_stock_eagle3(world_size) -> None:
    out = _rank_job.run("m3_draft_ar_norm", world_size)
    lines = [line for line in out.splitlines() if line.startswith(_RESULT_TAG)]
    assert lines, f"the launcher reported no measurements:\n{out}"
    warnings.warn(
        f"draft boundary (= stock Eagle3) vs fp64 at {world_size} ranks, max BF16 ulps\n"
        + "\n".join(lines),
        stacklevel=1,
    )
