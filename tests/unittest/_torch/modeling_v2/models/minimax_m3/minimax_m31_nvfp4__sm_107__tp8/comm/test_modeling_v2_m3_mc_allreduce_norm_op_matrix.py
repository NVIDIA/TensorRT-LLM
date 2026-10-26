# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the multicast all-reduce + RMSNorm entries' certification matrix.

Covers both ``m3_mc_allreduce_norm`` and ``m3_mc_finalize_allreduce_norm`` (one
kernel, one exchange protocol). The matrix is ``_m3_mc_allreduce_norm_op_matrix.py``
beside this file, its own launcher; it drives every cell of each entry's
``CELLS`` against the entry's ``reference``. See ``_rank_job`` for why.

The tp8 target runs these entries at 8 ranks over two trays, which a single-node
job cannot form. The world size is a runtime argument of the kernel and no cell's
spec names one, so the matrix certifies every cell at each world size of
``_rank_job.WORLD_SIZES`` the node's devices can form (2 and 4 on a 4-device tray,
8 on an 8-device node) and skips the others; it does not emulate the two-tray group.
"""

import _rank_job
import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8.ops.comm.m3_mc_finalize_allreduce_norm import (  # noqa: E501
    finalize_partial,
)

assert torch.cuda.is_available(), "m3_mc_allreduce_norm requires CUDA devices"


# Spawns one rank process per device of the world size; under xdist workers would fight for them.
@pytest.mark.no_xdist
@pytest.mark.parametrize("world_size", _rank_job.WORLD_SIZES, ids=lambda w: f"world{w}")
def test_m3_mc_allreduce_norm_op_matrix(world_size) -> None:
    _rank_job.run("m3_mc_allreduce_norm", world_size)


def test_the_cells_are_world_independent() -> None:
    """What lets a single-node job certify the tp8 entries: no cell's spec names a world size."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8.ops.comm import (  # noqa: E501
        m3_mc_allreduce_norm,
        m3_mc_finalize_allreduce_norm,
    )

    for op in (
        m3_mc_allreduce_norm.m3_mc_allreduce_norm,
        m3_mc_finalize_allreduce_norm.m3_mc_finalize_allreduce_norm,
    ):
        assert op.CELLS
        for cell in op.CELLS:
            assert not {"world", "world_size", "rank"} & set(cell.spec), cell.why


def test_finalize_partial_rounds_the_routed_sum_before_the_shared_expert() -> None:
    """As stock: the finalized routed output is BF16, and the shared expert is added to it (shared.add_(routed)).
    Four routed rows of 1.0 at FP32 routing weight 0.2509765625 sum to 1.00390625, which rounds to BF16 1.0; plus the
    shared -1 that is 0, where adding before the rounding would leave 0.00390625."""
    fc2 = torch.ones((4, 8), dtype=torch.bfloat16)
    perm = torch.arange(4, dtype=torch.int32).view(1, 4)
    scales = torch.full((1, 4), 0.2509765625, dtype=torch.float32)
    shared = torch.full((1, 8), -1.0, dtype=torch.bfloat16)
    out = finalize_partial(fc2, perm, scales, shared)
    routed = finalize_partial(fc2, perm, scales, None)
    assert torch.equal(out, shared.clone().add_(routed))
    assert torch.equal(out, torch.zeros(1, 8, dtype=torch.bfloat16))
