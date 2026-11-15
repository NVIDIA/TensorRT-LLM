# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``m3_mc_finalize_allreduce_norm``: its reference's rounding. The entry's GPU cells (9..48 rows, every one gated on
that reference) run in the multicast all-reduce matrix, ``test_modeling_v2_m3_mc_allreduce_norm_op_matrix.py``, which
launches both entries of the one kernel at 2 ranks."""

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2.ops.comm.m3_mc_finalize_allreduce_norm import (  # noqa: E501
    finalize_partial,
    m3_mc_finalize_allreduce_norm,
)


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


def test_the_cells_cover_every_row_the_target_sends_here() -> None:
    """The decode layer's MC finalize (MC_FINALIZE_MIN_TOKENS 9 .. the b32 plan's 32 rows) and the generic layer's
    (past 32, up to GENERIC_MC_FINALIZE_MAX_TOKENS 48): every one of those row counts is a certified cell."""
    rows = {cell.spec["m"] for cell in m3_mc_finalize_allreduce_norm.CELLS}
    assert set(range(9, 49)) <= rows
