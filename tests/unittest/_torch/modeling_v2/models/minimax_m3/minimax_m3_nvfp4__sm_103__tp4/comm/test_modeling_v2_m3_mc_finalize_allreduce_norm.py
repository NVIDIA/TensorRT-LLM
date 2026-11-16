# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of the ``m3_mc_finalize_allreduce_norm`` entry that need no collective.

Its cells are certified, with ``m3_mc_allreduce_norm``'s (one kernel, one exchange protocol), by the 4-rank matrix
``_m3_mc_allreduce_norm_op_matrix.py`` that ``test_modeling_v2_m3_mc_allreduce_norm_op_matrix.py`` runs.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.comm.m3_mc_finalize_allreduce_norm import (  # noqa: E501
    finalize_partial,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.comm.m3_mc_finalize_allreduce_norm import (  # noqa: E501
    m3_mc_finalize_allreduce_norm as op,
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


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_each_cells_partial_is_the_stock_finalize_then_shared_add(cell) -> None:
    """Per cell shape (rows, top-k, dropped slots), every rank's partial in the entry's reference is the stock MoE's:
    the finalized routed rows rounded to BF16, then ``shared.add_(routed)``. The 4-rank matrix drives the cells on the
    GPU against this reference."""
    spec = cell.spec
    g = torch.Generator().manual_seed(spec["m"] * 31 + spec["dropped"])
    m, topk, hidden = spec["m"], spec["topk"], 64
    fc2 = torch.randn(m * topk, hidden, generator=g).to(torch.bfloat16)
    perm = torch.randperm(m * topk, generator=g).view(m, topk).to(torch.int32)
    perm.view(-1)[: spec["dropped"]] = -1
    scales = torch.rand(m, topk, generator=g)
    shared = torch.randn(m, hidden, generator=g).to(torch.bfloat16)
    routed = finalize_partial(fc2, perm, scales, None)
    assert torch.equal(finalize_partial(fc2, perm, scales, shared), shared.clone().add_(routed))
