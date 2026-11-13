# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for CAPTURE_NORM_IN_BOUNDARY on the tp4 multicast all-reduce + RMSNorm op.

The matrix is ``_m3_capture_in_boundary_op_matrix.py`` beside this file, its own 4-rank launcher; see ``_rank_job``.
It requires the boundary's capture output to equal ``m3_spec_tail.capture_norm`` of its updated residual bit for bit,
on both kernels the tp4 op runs (the oneshot cluster kernel, the row kernel), and the op's other outputs to be
unchanged by it.
"""

import _rank_job
import pytest
import torch

assert torch.cuda.is_available(), "the multicast all-reduce requires CUDA devices"


# Spawns 4 rank processes over the visible devices; under xdist workers would fight for them.
@pytest.mark.no_xdist
def test_m3_capture_norm_in_boundary_is_capture_norm() -> None:
    out = _rank_job.run("m3_capture_in_boundary")
    assert "[case]" in out, f"the launcher ran no case:\n{out}"
