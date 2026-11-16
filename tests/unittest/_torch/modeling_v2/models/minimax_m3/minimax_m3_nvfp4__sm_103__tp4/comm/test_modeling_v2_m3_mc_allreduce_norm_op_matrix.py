# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the multicast all-reduce + RMSNorm entries' certification matrix.

Covers both ``m3_mc_allreduce_norm`` and ``m3_mc_finalize_allreduce_norm`` (one
kernel, one exchange protocol). The matrix is ``_m3_mc_allreduce_norm_op_matrix.py``
beside this file, its own 4-rank launcher; see ``_rank_job`` for why.
"""

import _rank_job
import pytest
import torch

assert torch.cuda.is_available(), "m3_mc_allreduce_norm requires CUDA devices"


# Spawns 4 rank processes over the visible devices; under xdist workers would fight for them.
@pytest.mark.no_xdist
def test_m3_mc_allreduce_norm_op_matrix() -> None:
    _rank_job.run("m3_mc_allreduce_norm")
