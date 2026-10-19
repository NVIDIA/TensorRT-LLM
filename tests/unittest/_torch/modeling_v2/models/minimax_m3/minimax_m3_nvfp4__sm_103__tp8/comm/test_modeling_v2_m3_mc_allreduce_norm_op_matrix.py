# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the multicast all-reduce + RMSNorm entries' certification matrix.

Covers both ``m3_mc_allreduce_norm`` and ``m3_mc_finalize_allreduce_norm`` (one
kernel, one exchange protocol). The matrix is ``_m3_mc_allreduce_norm_op_matrix.py``
beside this file, its own launcher; see ``_rank_job`` for why.

The tp8 target runs these entries at 8 ranks over two trays, which a single-node
job cannot form. The world size is a runtime argument of the kernel and no cell's
spec names one, so the matrix certifies every cell at each world size of
``_rank_job.WORLD_SIZES`` the node's devices can form (2 and 4 on a GB300 tray, 8
on an 8-device node) and skips the others; it does not emulate the two-tray group.
"""

import _rank_job
import pytest
import torch

assert torch.cuda.is_available(), "m3_mc_allreduce_norm requires CUDA devices"


# Spawns one rank process per device of the world size; under xdist workers would fight for them.
@pytest.mark.no_xdist
@pytest.mark.parametrize("world_size", _rank_job.WORLD_SIZES, ids=lambda w: f"world{w}")
def test_m3_mc_allreduce_norm_op_matrix(world_size) -> None:
    _rank_job.run("m3_mc_allreduce_norm", world_size)


def test_the_cells_are_world_independent() -> None:
    """What lets a single-node job certify the tp8 entries: no cell's spec names a world size."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops.comm import (  # noqa: E501
        m3_mc_allreduce_norm,
        m3_mc_finalize_allreduce_norm,
    )

    for op in (
        m3_mc_allreduce_norm.m3_mc_allreduce_norm,
        m3_mc_finalize_allreduce_norm.m3_mc_finalize_allreduce_norm,
    ):
        for cell in op.CELLS:
            assert not {"world", "world_size", "rank"} & set(cell.spec), cell.why
